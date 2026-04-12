"""Normalization and fuzzy matching helpers for noisy STT text."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable, List, Tuple

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - fallback path only when dependency is missing.
    fuzz = None

_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_PUNCT_RE = re.compile(r"[^a-z0-9\u0600-\u06FF\s]")
_SPACE_RE = re.compile(r"\s+")
_ARABIC_CHAR_MAP = str.maketrans(
    {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0624": "\u0648",
        "\u0626": "\u064a",
        "\u0649": "\u064a",
        "\u0629": "\u0647",
    }
)


def normalize_text(text: str) -> str:
    """Normalize Arabic/English text for robust exact+fuzzy matching."""
    value = str(text or "").strip().lower()
    if not value:
        return ""

    value = _ARABIC_DIACRITICS_RE.sub("", value)
    value = value.replace("\u0640", "")
    value = value.translate(_ARABIC_CHAR_MAP)
    value = value.replace("_", " ").replace("-", " ")
    value = _PUNCT_RE.sub(" ", value)
    value = _SPACE_RE.sub(" ", value).strip()
    return value


def fuzzy_score(text: str, keyword: str) -> int:
    """Return a 0-100 fuzzy partial match score between text and keyword."""
    normalized_text = normalize_text(text)
    normalized_keyword = normalize_text(keyword)

    if not normalized_text or not normalized_keyword:
        return 0

    if normalized_keyword in normalized_text:
        return 100

    if fuzz is not None:
        score = fuzz.partial_ratio(normalized_text, normalized_keyword)
        return int(round(score))

    # Lightweight fallback when rapidfuzz is unavailable.
    baseline = SequenceMatcher(None, normalized_text, normalized_keyword).ratio()
    token_scores = [SequenceMatcher(None, token, normalized_keyword).ratio() for token in normalized_text.split()]
    return int(round(max([baseline] + token_scores) * 100))


def fuzzy_contains(text: str, keyword: str, threshold: int = 70) -> bool:
    """True when keyword appears in text by exact or fuzzy partial matching."""
    return fuzzy_score(text, keyword) >= int(threshold)


def find_keyword_matches(text: str, keywords: Iterable[str], threshold: int = 70) -> List[Tuple[str, int]]:
    """Return all keywords that match text with score >= threshold, sorted by score desc."""
    matches: List[Tuple[str, int]] = []
    for keyword in keywords:
        score = fuzzy_score(text, keyword)
        if score >= int(threshold):
            matches.append((str(keyword), int(score)))

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches
