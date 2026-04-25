"""DuckDuckGo web search — free, no API key, no signup."""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
import warnings

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


def _ddgs_text_search(query: str, max_results: int):
    # Legacy duckduckgo_search emits a rename RuntimeWarning; suppress it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))


def search_web(query: str, max_results: int = 3) -> str:
    """Return formatted search results as context for LLM prompt injection.

    Returns an empty string on failure so callers can fall back gracefully.
    """
    if not _DDGS_AVAILABLE:
        logger.warning("duckduckgo-search not installed — web search unavailable")
        return ""

    query = (query or "").strip()
    if not query:
        return ""

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_ddgs_text_search, query, int(max_results))
            results = future.result(timeout=_SEARCH_TIMEOUT_SECONDS)
        if not results:
            return ""
        lines = []
        for r in results:
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()
            if title or body:
                lines.append(f"- {title}: {body}")
        return "\n".join(lines)
    except TimeoutError:
        logger.warning("Web search timed out after %.1fs", _SEARCH_TIMEOUT_SECONDS)
        return ""
    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return ""
