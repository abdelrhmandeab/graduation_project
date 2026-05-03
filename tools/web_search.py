"""DuckDuckGo web search — free, no API key, no signup.

Phase 2.7 hardening:
  - Domain allowlist boost (trusted publishers rank first).
  - Domain blocklist filtering (drop low-quality sources outright).
  - Recency scoring using any "date"/"published" field exposed by the backend.
"""

import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
from urllib.parse import urlparse

from core.config import (
    WEB_SEARCH_BLOCKED_DOMAINS,
    WEB_SEARCH_RECENCY_BOOST,
    WEB_SEARCH_TRUSTED_DOMAIN_BOOST,
    WEB_SEARCH_TRUSTED_DOMAINS,
)
from core.logger import logger

try:
    from ddgs import DDGS

    _DDGS_AVAILABLE = True
    _DDGS_PROVIDER = "ddgs"
except ImportError:
    try:
        from duckduckgo_search import DDGS

        _DDGS_AVAILABLE = True
        _DDGS_PROVIDER = "duckduckgo_search"
    except ImportError:
        _DDGS_AVAILABLE = False
        _DDGS_PROVIDER = ""


_SEARCH_TIMEOUT_SECONDS = 6.0

# Recency cliff: results published within ~30 days hit full recency boost,
# items older than ~365 days lose the boost entirely.
_RECENCY_FULL_WINDOW_SECONDS = 30 * 24 * 3600
_RECENCY_DECAY_WINDOW_SECONDS = 335 * 24 * 3600

_DATE_FIELD_CANDIDATES = ("date", "published", "timestamp", "modified")


def _ddgs_text_search(query: str, max_results: int):
    """Run a synchronous DuckDuckGo text search.

    We keep this isolated so the caller can run it on a worker thread and apply
    a hard timeout — the upstream library can hang for 30s+ on flaky DNS.
    """
    # Legacy duckduckgo_search emits a rename RuntimeWarning; suppress it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))


def _normalize_domain(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if "://" not in text:
        text = "http://" + text
    try:
        parsed = urlparse(text)
    except Exception:
        return ""
    netloc = (parsed.netloc or parsed.path or "").strip().lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    # Strip ports.
    if ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    return netloc


def _domain_matches(haystack_domain: str, needle: str) -> bool:
    """Case-insensitive suffix match — `bbc.co.uk` matches `news.bbc.co.uk`."""
    candidate = _normalize_domain(haystack_domain)
    target = _normalize_domain(needle)
    if not candidate or not target:
        return False
    if candidate == target:
        return True
    return candidate.endswith("." + target)


def _is_trusted_domain(domain: str) -> bool:
    return any(_domain_matches(domain, item) for item in WEB_SEARCH_TRUSTED_DOMAINS)


def _is_blocked_domain(domain: str) -> bool:
    return any(_domain_matches(domain, item) for item in WEB_SEARCH_BLOCKED_DOMAINS)


def _parse_published_timestamp(raw_value):
    """Best-effort ISO-8601 / RFC-2822 parsing — tolerant of partial dates."""
    text = str(raw_value or "").strip()
    if not text:
        return 0.0

    # ISO with optional trailing Z
    iso_candidate = text.replace("Z", "+00:00") if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(iso_candidate)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError:
        pass

    # YYYY-MM-DD anywhere in the string
    iso_match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if iso_match:
        try:
            year, month, day = (int(part) for part in iso_match.groups())
            return datetime(year, month, day, tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass

    return 0.0


def _recency_factor(published_ts: float, now_ts: float) -> float:
    if published_ts <= 0.0:
        return 0.0
    age = max(0.0, now_ts - published_ts)
    if age <= _RECENCY_FULL_WINDOW_SECONDS:
        return 1.0
    if age >= _RECENCY_DECAY_WINDOW_SECONDS:
        return 0.0
    span = float(_RECENCY_DECAY_WINDOW_SECONDS - _RECENCY_FULL_WINDOW_SECONDS)
    return max(0.0, 1.0 - (age - _RECENCY_FULL_WINDOW_SECONDS) / span)


def _score_result(result, now_ts: float) -> float:
    """Combined ranking signal: trust + recency + intrinsic order."""
    base_score = float(result.get("_base_rank") or 0.0)
    domain = _normalize_domain(result.get("href") or result.get("url") or "")
    trust_boost = (
        float(WEB_SEARCH_TRUSTED_DOMAIN_BOOST) if _is_trusted_domain(domain) else 0.0
    )

    published_ts = 0.0
    for field in _DATE_FIELD_CANDIDATES:
        published_ts = _parse_published_timestamp(result.get(field))
        if published_ts > 0:
            break
    recency_boost = float(WEB_SEARCH_RECENCY_BOOST) * _recency_factor(published_ts, now_ts)

    return base_score + trust_boost + recency_boost


def _filter_and_rank(raw_results, max_results: int):
    """Drop blocklisted hosts, score the rest, return top-N rows."""
    if not raw_results:
        return []

    now_ts = time.time()
    enriched = []
    for index, row in enumerate(raw_results):
        domain = _normalize_domain((row or {}).get("href") or (row or {}).get("url") or "")
        if domain and _is_blocked_domain(domain):
            continue
        # Keep DDG ordering as a tiebreaker — earlier rows beat later ones at parity.
        scored = dict(row or {})
        scored["_base_rank"] = max(0.0, 1.0 - (index * 0.05))
        scored["_domain"] = domain
        scored["_published_ts"] = 0.0
        for field in _DATE_FIELD_CANDIDATES:
            ts = _parse_published_timestamp(scored.get(field))
            if ts > 0:
                scored["_published_ts"] = ts
                break
        scored["_score"] = _score_result(scored, now_ts)
        enriched.append(scored)

    enriched.sort(key=lambda item: item["_score"], reverse=True)
    return enriched[: max(1, int(max_results))]


def _format_results(rows) -> str:
    lines = []
    for row in rows:
        title = (row.get("title") or "").strip()
        body = (row.get("body") or "").strip()
        domain = (row.get("_domain") or "").strip()

        suffix_bits = []
        if domain:
            suffix_bits.append(domain)
        published_ts = float(row.get("_published_ts") or 0.0)
        if published_ts > 0:
            try:
                suffix_bits.append(
                    datetime.fromtimestamp(published_ts, tz=timezone.utc).strftime("%Y-%m-%d")
                )
            except Exception:
                pass

        suffix = f" [{' · '.join(suffix_bits)}]" if suffix_bits else ""
        if title or body:
            lines.append(f"- {title}: {body}{suffix}")
    return "\n".join(lines)


def search_web(query: str, max_results: int = 3) -> str:
    """Return formatted search results as context for LLM prompt injection.

    Returns an empty string on failure so callers can fall back gracefully.

    Phase 2.7: results are filtered against ``WEB_SEARCH_BLOCKED_DOMAINS``,
    boosted when the host appears in ``WEB_SEARCH_TRUSTED_DOMAINS`` and ranked
    by recency when the backend exposes a publication date.
    """
    if not _DDGS_AVAILABLE:
        logger.warning("duckduckgo-search not installed — web search unavailable")
        return ""

    query = (query or "").strip()
    if not query:
        return ""

    requested = max(1, int(max_results))
    # Pull a few extra rows so the trust/recency reranker has headroom even
    # after blocklist filtering.
    candidate_count = min(10, requested * 3)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_ddgs_text_search, query, candidate_count)
            raw_results = future.result(timeout=_SEARCH_TIMEOUT_SECONDS)
        if not raw_results:
            return ""
        ranked = _filter_and_rank(raw_results, requested)
        return _format_results(ranked)
    except TimeoutError:
        logger.warning("Web search timed out after %.1fs", _SEARCH_TIMEOUT_SECONDS)
        return ""
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return ""
