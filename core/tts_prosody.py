"""Deterministic prosody polisher for TTS.

Runs right before SSML wrap / ElevenLabs send.  No LLM call — pure regex.
Handles both English and Egyptian Arabic.
"""

from __future__ import annotations

import re
from typing import Literal

from core.config import (
    TTS_EGY_DISCOURSE_COMMA_ENABLED,
    TTS_FORMAL_CONNECTOR_REWRITE_ENABLED,
    TTS_PROSODY_POLISHER_ENABLED,
    TTS_PUNCTUATION_DEDUP_ENABLED,
)

# ── AR discourse particles that should get a comma after them ──
_AR_DISCOURSE_PARTICLES = (
    "طب",
    "بص",
    "شوف",
    "يلا",
    "تمام كده",
    "يعيش",
    "يعني",
    "بصراحة",
)

_AR_DISCOURSE_RE = re.compile(
    r"(?:^|(?<=\s))(" + "|".join(re.escape(p) for p in _AR_DISCOURSE_PARTICLES) + r")(?=\s+[^،,.:؟?!؛\s])",
    re.UNICODE,
)

# ── Formal MSA connectors → colloquial replacements (small whitelist) ──
_FORMAL_CONNECTOR_REWRITES: tuple[tuple[str, str], ...] = (
    ("بالإضافة إلى", "وكمان"),
    ("علاوة على ذلك", "وكمان"),
    ("بالتالي", "وبكده"),
    ("إذن", "يبقى"),
)

# ── EN compound sentence comma insertion ──
_EN_COMPOUND_RE = re.compile(
    r"(\S+(?:\s+\S+){3,})\s+(and|but|or)\s+(\S+(?:\s+\S+){3,})",
    re.IGNORECASE,
)


def polish_for_voice(
    text: str,
    language: Literal["en", "ar"] = "en",
    *,
    discourse_comma: bool = True,
    formal_connector_rewrite: bool = True,
    punctuation_dedup: bool = True,
) -> str:
    """Apply deterministic prosody fixes to text before TTS synthesis.

    Safe to call on any text — never changes content words,
    never touches numbers/units (the voice normalizer owns those).
    """
    if not TTS_PROSODY_POLISHER_ENABLED:
        return text

    result = str(text or "")
    if not result.strip():
        return result

    # ── Bilingual rules ──

    # Em-dash → comma
    result = result.replace("—", "،" if language == "ar" else ",")
    result = result.replace("–", "،" if language == "ar" else ",")

    # Ellipsis → comma (before period dedup so "..." isn't collapsed first)
    result = re.sub(r"…|\.{3}", "،" if language == "ar" else ",", result)

    if punctuation_dedup and TTS_PUNCTUATION_DEDUP_ENABLED:
        result = re.sub(r"\.{2,}", ".", result)
        result = re.sub(r"؟{2,}", "؟", result)
        result = re.sub(r"\?{2,}", "?", result)
        result = re.sub(r"!{2,}", "!", result)
        result = re.sub(r"،{2,}", "،", result)
        result = re.sub(r",{2,}", ",", result)

    # Residual markdown
    result = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", result)
    result = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", result)
    result = result.replace("`", "")

    if language == "en":
        # Hyphenated terms: collapse stray spaces around hyphens
        result = re.sub(r"(\w)\s+-\s+(\w)", r"\1-\2", result)

        # Compound sentence comma insertion before and/but/or
        # Only when both clauses are >= 4 words
        result = _EN_COMPOUND_RE.sub(r"\1, \2 \3", result)

    elif language == "ar":
        if discourse_comma and TTS_EGY_DISCOURSE_COMMA_ENABLED:
            result = _AR_DISCOURSE_RE.sub(r"\1،", result)

        if formal_connector_rewrite and TTS_FORMAL_CONNECTOR_REWRITE_ENABLED:
            for formal, colloquial in _FORMAL_CONNECTOR_REWRITES:
                result = result.replace(formal, colloquial)

    # Final whitespace cleanup
    result = re.sub(r"\s+", " ", result).strip()

    return result
