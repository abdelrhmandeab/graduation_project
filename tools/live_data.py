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
    VOICE_NORMALIZER_MAX_SEARCH_RESULTS,
    WEATHER_DEFAULT_CITY,
    WEB_SEARCH_MAX_RESULTS,
    WEB_SEARCH_ENABLED,
)
from core.logger import logger
from core.voice_normalizer import normalize_search_block, normalize_weather_block
from tools.weather import get_weather
from tools.web_search import search_web

_LIVE_DATA_TIMEOUT = 7.0  # Total timeout for all live data fetches; must exceed
# web_search's own internal timeout (6.0s in tools/web_search.py) or this outer
# timeout fires first and silently discards results that were about to land.

# Per-tool framing tells the LLM which block is authoritative for which question.
# Keep these short — Qwen-class models follow brief instructions better than long
# preambles, and the framing lives inside the model's already-tight context budget.
_TOOL_FRAMING = {
    "weather": "Voice-ready live weather:",
    "search": "Voice-ready live search results:",
}


def _detect_weather_intent(query_text: str) -> Optional[Dict]:
    """Detect if query asks about weather."""
    query = (query_text or "").lower().strip()
    arabic_weather_keywords = (
        "\u0637\u0642\u0633",
        "\u0627\u0644\u0637\u0642\u0633",
        "\u0627\u0644\u062c\u0648",
        "\u0623\u062e\u0628\u0627\u0631 \u0627\u0644\u0637\u0642\u0633",
        "\u0627\u062e\u0628\u0627\u0631 \u0627\u0644\u0637\u0642\u0633",
        "\u0623\u062e\u0628\u0627\u0631 \u0627\u0644\u062c\u0648",
        "\u0627\u062e\u0628\u0627\u0631 \u0627\u0644\u062c\u0648",
    )
    weather_keywords = [
        "weather", "forecast", "temperature", "temp", "cold", "hot", "rain",
        "snow", "clouds", "sunny", "cloudy", "humidity", "wind",
        # Bilingual Arabic — include both "طقس" alone and "الطقس" with the ال
        # article. Users say "طقس الاسكندرية" (no article) just as often as
        # "ايه الطقس النهارده".
        "طقس", "الطقس", "درجة الحرارة", "درجة", "برد", "حر",
        "مطر", "ثلج", "غيوم", "حرارة", "الحرارة",
    ]

    if any(kw in query for kw in weather_keywords) or any(kw in query for kw in arabic_weather_keywords):
        # Extract city if mentioned (optional — use default if not found).
        # First try preposition-anchored forms ("weather in X", "الطقس في X").
        city_match = re.search(
            r"\b(?:weather in|forecast for|in|at|في|ب)\s+([a-zA-Z؀-ۿ][a-zA-Z؀-ۿ\s]*?)"
            r"(?:\s+(?:today|tomorrow|weather|forecast|right now|now|in\s+this\s+weather)\b|$|[?.!,])",
            query,
            re.IGNORECASE,
        )
        if city_match:
            city = city_match.group(1).strip()
        else:
            # Egyptian Arabic often drops the preposition: "طقس الاسكندرية".
            # Look for an Arabic city token following the weather keyword.
            # Skip a leading "في"/"ب" so "الطقس في الاسكندرية" doesn't capture
            # the preposition itself as part of the city name.
            tail_match = re.search(
                r"(?:طقس|الطقس|حرارة|الحرارة)\s+(?:في\s+|ب)?([؀-ۿ]+(?:\s+[؀-ۿ]+)?)",
                query,
            )
            city = tail_match.group(1).strip() if tail_match else WEATHER_DEFAULT_CITY
        if city in {
            "\u0627\u0644\u0646\u0647\u0627\u0631\u062f\u0629",
            "\u0627\u0644\u0646\u0647\u0627\u0631\u062f\u0647",
            "\u0627\u0644\u064a\u0648\u0645",
            "\u062f\u0644\u0648\u0642\u062a\u064a",
        }:
            city = WEATHER_DEFAULT_CITY
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
        # NOTE: bare "ايه"/"مين"/"كيف" ("what"/"who"/"how") were removed from
        # here — they match nearly any casual Arabic sentence (including
        # greetings like "عامل ايه") and caused searches to fire on chit-chat.
        # The caller's force_search flag (see command_router._fetch_live_tool_context)
        # is now the primary signal for when to search at all.
        "بحث", "ابحث", "أبحث", "ادور", "ادوّر",
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


def _format_block(tool_kind: str, body: str, language: str = "en") -> str:
    """Wrap a tool result in a labeled block + per-tool instruction.

    The label gives the LLM a clear anchor it can refer back to ("according to the
    weather block...") and the framing line steers it toward verbatim use.
    """
    text = str(body or "").strip()
    if not text:
        return ""
    if tool_kind == "weather":
        text = normalize_weather_block(text, language)
    elif tool_kind == "search":
        text = normalize_search_block(
            text,
            language,
            max_results=int(VOICE_NORMALIZER_MAX_SEARCH_RESULTS or 2),
        )
    framing = _TOOL_FRAMING.get(tool_kind, "")
    label = _TOOL_LABELS.get(tool_kind, f"[{tool_kind.upper()}]")
    if framing:
        return f"{label} {framing}\n{text}"
    return f"{label}\n{text}"


def gather_live_data(user_query: str, parallel: bool = True, force_search: bool = False) -> str:
    """Fetch weather, search, and other live data in parallel.

    Returns formatted context string ready for prompt injection. Each tool's
    output is wrapped in a labeled block so the LLM can attribute facts back to
    the correct source.
    """
    query = (user_query or "").strip()
    if not query:
        return ""
    language = "ar" if re.search(r"[\u0600-\u06FF]", query) else "en"

    weather_intent = _detect_weather_intent(query)
    # When weather already answers the query, skip the generic keyword-based
    # search detector — it matches very broad terms like "today"/"now" that
    # overlap heavily with weather phrasing, which previously caused every
    # weather/clothing question to also fire a parallel web search that timed
    # out 6s later having found nothing useful. The caller (command_router)
    # still controls force_search explicitly when a real search is wanted.
    search_intent = None if weather_intent else _detect_web_search_intent(query)

    if not weather_intent and not search_intent and force_search and WEB_SEARCH_ENABLED:
        search_intent = {"type": "search", "query": query}

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
                    block = _format_block(kind, result, language)
                    if block:
                        blocks.append(block)
            except FutureTimeoutError:
                logger.debug("Live data fetch timed out")
    else:
        if weather_intent:
            block = _format_block("weather", _fetch_weather(weather_intent["city"]), language)
            if block:
                blocks.append(block)
        if search_intent:
            block = _format_block(
                "search",
                _fetch_web_search(search_intent["query"], int(WEB_SEARCH_MAX_RESULTS or 3)),
                language,
            )
            if block:
                blocks.append(block)

    return "\n\n".join(blocks) if blocks else ""
