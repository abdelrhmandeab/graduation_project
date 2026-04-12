import re

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_ENGLISH_RE = re.compile(r"[A-Za-z]")


def count_arabic_chars(text: str) -> int:
    return len(_ARABIC_RE.findall(str(text or "")))


def count_english_chars(text: str) -> int:
    return len(_ENGLISH_RE.findall(str(text or "")))


def count_valid_chars(text: str) -> int:
    return count_arabic_chars(text) + count_english_chars(text)


def detect_language(text: str) -> str:
    value = str(text or "")
    arabic_count = count_arabic_chars(value)
    english_count = count_english_chars(value)

    if arabic_count == 0 and english_count == 0:
        return "unknown"
    if arabic_count > 0 and english_count == 0:
        return "ar"
    if english_count > 0 and arabic_count == 0:
        return "en"
    return "mixed"
