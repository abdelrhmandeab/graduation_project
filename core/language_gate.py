import re
from dataclasses import dataclass


_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")

SUPPORTED_LANGUAGES = {"ar", "en"}

UNSUPPORTED_LANGUAGE_MESSAGE = (
    "Sorry, I currently support Arabic and English only. "
    "عذرًا، أنا أتعامل حاليًا مع العربية والإنجليزية فقط."
)


@dataclass
class LanguageGateResult:
    supported: bool
    language: str
    normalized_text: str
    reason: str = ""


def _script_counts(text):
    arabic = 0
    latin = 0
    cyrillic = 0
    other_alpha = 0
    for char in text or "":
        if _ARABIC_CHAR_RE.match(char):
            arabic += 1
        elif _LATIN_CHAR_RE.match(char):
            latin += 1
        elif _CYRILLIC_RE.match(char):
            cyrillic += 1
        elif char.isalpha():
            other_alpha += 1
    return arabic, latin, cyrillic, other_alpha


def _normalize_arabic(text):
    cleaned = (text or "").strip()
    cleaned = cleaned.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    cleaned = cleaned.replace("ـ", "")
    cleaned = _ARABIC_DIACRITICS_RE.sub("", cleaned)
    return " ".join(cleaned.split())


def normalize_text_for_language(text, language):
    if (language or "").lower() == "ar":
        return _normalize_arabic(text)
    return " ".join((text or "").strip().split())


def detect_supported_language(text, previous_language="en"):
    raw = (text or "").strip()
    if not raw:
        fallback_language = previous_language if previous_language in SUPPORTED_LANGUAGES else "en"
        return LanguageGateResult(
            supported=True,
            language=fallback_language,
            normalized_text="",
            reason="empty_text",
        )

    arabic, latin, cyrillic, other_alpha = _script_counts(raw)
    ar_en_total = arabic + latin

    # Any Cyrillic characters are a strong signal the transcript is wrong (Russian
    # hallucination from Whisper, or the user spoke Russian). Two or more is definitive.
    if cyrillic >= 2:
        return LanguageGateResult(
            supported=False,
            language="unsupported",
            normalized_text=raw,
            reason="cyrillic_script_detected",
        )

    if ar_en_total == 0 and (other_alpha > 0 or cyrillic > 0):
        return LanguageGateResult(
            supported=False,
            language="unsupported",
            normalized_text=raw,
            reason="unsupported_script_only",
        )

    if other_alpha > 0 and other_alpha >= max(1, ar_en_total):
        return LanguageGateResult(
            supported=False,
            language="unsupported",
            normalized_text=raw,
            reason="unsupported_script_dominant",
        )

    if arabic > 0 and latin == 0:
        language = "ar"
    elif latin > 0 and arabic == 0:
        language = "en"
    elif arabic > 0 and latin > 0:
        if arabic == latin and previous_language in SUPPORTED_LANGUAGES:
            language = previous_language
        else:
            language = "ar" if arabic > latin else "en"
    else:
        language = previous_language if previous_language in SUPPORTED_LANGUAGES else "en"

    normalized = normalize_text_for_language(raw, language)
    return LanguageGateResult(
        supported=True,
        language=language,
        normalized_text=normalized,
        reason="ok",
    )
