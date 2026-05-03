"""Phase 2: Live Data Injection Pipeline.

Fetches weather, web search, and other real-time data to augment LLM prompts.

Phase 2.7 hardening:
  - Each tool result is wrapped in a labeled block (``[WEATHER]`` / ``[WEB_SEARCH]``)
    with a short instruction line so the model treats live data as authoritative
    rather than trying to merge it with its prior knowledge.
"""

import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Dict, Optional

from core.config import (
    WEATHER_DEFAULT_CITY,
    WEB_SEARCH_MAX_RESULTS,
)
from core.logger import logger
from tools.weather import get_weather
from tools.web_search import search_web

_LIVE_DATA_TIMEOUT = 3.0  # Total timeout for all live data fetches

# Per-tool framing tells the LLM which block is authoritative for which question.
# Keep these short — Qwen-class models follow brief instructions better than long
# preambles, and the framing lives inside the model's already-tight context budget.
_TOOL_FRAMING = {
    "weather": (
        "Live weather snapshot (use these numbers verbatim — do not invent values):"
    ),
    "search": (
        "Live web search results (cite the most relevant snippet, ignore irrelevant ones):"
    ),
}


def _detect_weather_intent(query_text: str) -> Optional[Dict]:
    """Detect if query asks about weather."""
    query = (query_text or "").lower().strip()
    weather_keywords = [
        "weather", "forecast", "temperature", "temp", "cold", "hot", "rain",
        "snow", "clouds", "sunny", "cloudy", "humidity", "wind",
        # Bilingual Arabic — include both "طقس" alone and "الطقس" with the ال
        # article. Users say "طقس الاسكندرية" (no article) just as often as
        # "ايه الطقس النهارده".
        "طقس", "الطقس", "درجة الحرارة", "درجة", "برد", "حر",
        "مطر", "ثلج", "غيوم", "حرارة", "الحرارة",
    ]

    if any(kw in query for kw in weather_keywords):
        # Extract city if mentioned (optional — use default if not found).
        # First try preposition-anchored forms ("weather in X", "الطقس في X").
        city_match = re.search(
            r"(?:in|at|weather in|forecast for|في|ب)\s+([a-zA-Z؀-ۿ\s]+?)"
            r"(?:\s+(?:today|tomorrow|weather|forecast)|$)",
            query,
            re.IGNORECASE,
        )
        if city_match:
            city = city_match.group(1).strip()
        else:
            # Egyptian Arabic often drops the preposition: "طقس الاسكندرية".
            # Look for an Arabic city token following the weather keyword.
            tail_match = re.search(
                r"(?:طقس|الطقس|حرارة|الحرارة)\s+([؀-ۿ]+(?:\s+[؀-ۿ]+)?)",
                query,
            )
            city = tail_match.group(1).strip() if tail_match else WEATHER_DEFAULT_CITY
        return {"type": "weather", "city": city}

    return None


def _detect_web_search_intent(query_text: str) -> Optional[Dict]:
    """Detect if query needs web search."""
    query = (query_text or "").lower().strip()
    search_keywords = [
        "search", "look up", "find", "what is", "who is", "how to", "latest",
        "current", "today", "news",
        # Egyptian/MSA — include "ابحث" with the alef prefix and "اخبار"/"أخبار"
        # so news queries do not silently fall through to plain LLM mode.
        "بحث", "ابحث", "أبحث", "ادور", "ادوّر", "ايه", "مين", "كيف",
        "اخبار", "أخبار", "آخر",
    ]

    # Exclude queries that are system commands or file operations
    exclude_phrases = [
        "search index", "search file", "find file", "البحث عن ملف", "search drives",
    ]

    if any(exclude in query for exclude in exclude_phrases):
        return None

    if any(kw in query for kw in search_keywords):
        # Extract search terms (remove common prefixes — Arabic keywords may
        # carry the leading alef "ا" / "أ" so we strip those forms too).
        search_terms = re.sub(
            r"^(?:search|google|look\s+up|أبحث|ابحث|بحث|ادور|ادوّر|ايه)\s*(?:عن\s+)?",
            "",
            query,
        ).strip()
        if search_terms:
            return {"type": "search", "query": search_terms}

    return None


def _fetch_weather(city: str) -> str:
    """Fetch weather data for a city."""
    try:
        return get_weather(city=city) or ""
    except Exception as exc:
        logger.debug("Weather fetch failed: %s", exc)
        return ""


def _fetch_web_search(query: str, max_results: int = 3) -> str:
    """Fetch web search results."""
    try:
        return search_web(query, max_results=max_results) or ""
    except Exception as exc:
        logger.debug("Web search failed: %s", exc)
        return ""


_TOOL_LABELS = {
    "weather": "[WEATHER]",
    "search": "[WEB_SEARCH]",
}


def _format_block(tool_kind: str, body: str) -> str:
    """Wrap a tool result in a labeled block + per-tool instruction.

    The label gives the LLM a clear anchor it can refer back to ("according to the
    weather block...") and the framing line steers it toward verbatim use.
    """
    text = str(body or "").strip()
    if not text:
        return ""
    framing = _TOOL_FRAMING.get(tool_kind, "")
    label = _TOOL_LABELS.get(tool_kind, f"[{tool_kind.upper()}]")
    if framing:
        return f"{label} {framing}\n{text}"
    return f"{label}\n{text}"


def gather_live_data(user_query: str, parallel: bool = True) -> str:
    """Fetch weather, search, and other live data in parallel.

    Returns formatted context string ready for prompt injection. Each tool's
    output is wrapped in a labeled block so the LLM can attribute facts back to
    the correct source.
    """
    query = (user_query or "").strip()
    if not query:
        return ""

    weather_intent = _detect_weather_intent(query)
    search_intent = _detect_web_search_intent(query)

    if not weather_intent and not search_intent:
        return ""  # No live data needed

    blocks: list[str] = []

    if parallel:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}
            if weather_intent:
                futures["weather"] = executor.submit(_fetch_weather, weather_intent["city"])
            if search_intent:
                futures["search"] = executor.submit(
                    _fetch_web_search,
                    search_intent["query"],
                    int(WEB_SEARCH_MAX_RESULTS or 3),
                )

            try:
                for kind, future in futures.items():
                    result = future.result(timeout=_LIVE_DATA_TIMEOUT)
                    block = _format_block(kind, result)
                    if block:
                        blocks.append(block)
            except FutureTimeoutError:
                logger.debug("Live data fetch timed out")
    else:
        if weather_intent:
            block = _format_block("weather", _fetch_weather(weather_intent["city"]))
            if block:
                blocks.append(block)
        if search_intent:
            block = _format_block(
                "search",
                _fetch_web_search(search_intent["query"], int(WEB_SEARCH_MAX_RESULTS or 3)),
            )
            if block:
                blocks.append(block)

    return "\n\n".join(blocks) if blocks else ""
