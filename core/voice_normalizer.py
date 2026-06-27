"""Deterministic voice-ready text normalization for Jarvis.

This module deliberately avoids LLM calls.  It converts tool output and final
assistant text into phrases that TTS can speak naturally in English or Egyptian
Arabic.
"""

from __future__ import annotations

import re
from datetime import date
from urllib.parse import urlparse

from core.config import (
    VOICE_NORMALIZER_ENABLED,
    VOICE_NORMALIZER_KEEP_URLS,
    VOICE_NORMALIZER_MAX_SEARCH_RESULTS,
)


_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")
_TIME_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")

_AR_PLACE_NAMES = {
    "cairo": "القاهرة",
    "alexandria": "الإسكندرية",
    "giza": "الجيزة",
    "egypt": "مصر",
}

_AR_CONDITIONS = {
    "clear sky": "السما صافية",
    "mainly clear": "الجو صافي في الغالب",
    "partly cloudy": "الجو غائم جزئياً",
    "overcast": "الجو ملبد بالغيوم",
    "foggy": "فيه شبورة",
    "light drizzle": "فيه رذاذ خفيف",
    "moderate drizzle": "فيه رذاذ متوسط",
    "dense drizzle": "فيه رذاذ تقيل",
    "slight rain": "فيه مطر خفيف",
    "moderate rain": "فيه مطر متوسط",
    "heavy rain": "فيه مطر غزير",
    "thunderstorm": "فيه عواصف رعدية",
}

_ONES_AR = {
    0: "صفر",
    1: "واحد",
    2: "اتنين",
    3: "تلاتة",
    4: "أربعة",
    5: "خمسة",
    6: "ستة",
    7: "سبعة",
    8: "تمانية",
    9: "تسعة",
    10: "عشرة",
    11: "حداشر",
    12: "اتناشر",
    13: "تلتاشر",
    14: "أربعتاشر",
    15: "خمستاشر",
    16: "ستاشر",
    17: "سبعتاشر",
    18: "تمنتاشر",
    19: "تسعتاشر",
}
_TENS_AR = {
    20: "عشرين",
    30: "تلاتين",
    40: "أربعين",
    50: "خمسين",
    60: "ستين",
    70: "سبعين",
    80: "تمانين",
    90: "تسعين",
}

_ONES_EN = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
    17: "seventeen",
    18: "eighteen",
    19: "nineteen",
}
_TENS_EN = {
    20: "twenty",
    30: "thirty",
    40: "forty",
    50: "fifty",
    60: "sixty",
    70: "seventy",
    80: "eighty",
    90: "ninety",
}
_MONTHS_EN = [
    "",
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
_MONTHS_AR = [
    "",
    "يناير",
    "فبراير",
    "مارس",
    "أبريل",
    "مايو",
    "يونيو",
    "يوليو",
    "أغسطس",
    "سبتمبر",
    "أكتوبر",
    "نوفمبر",
    "ديسمبر",
]


def _lang(language: str) -> str:
    return "ar" if str(language or "").strip().lower() == "ar" else "en"


def _clean_decimal(value: str) -> str:
    text = str(value or "").strip()
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _intish(value: str) -> int | None:
    try:
        return int(round(float(str(value).replace(",", ""))))
    except Exception:
        return None


def _num_ar(value: str | int | float) -> str:
    text = _clean_decimal(str(value))
    if "." in text:
        whole, frac = text.split(".", 1)
        return f"{_num_ar(whole)} فاصلة {' '.join(_num_ar(ch) for ch in frac[:2])}"
    try:
        n = int(text)
    except Exception:
        return str(value)
    if n < 0:
        return "سالب " + _num_ar(abs(n))
    if n < 20:
        return _ONES_AR[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        return _TENS_AR[tens] if ones == 0 else f"{_ONES_AR[ones]} و{_TENS_AR[tens]}"
    if n < 1000:
        hundreds = n // 100
        rest = n % 100
        if hundreds == 1:
            prefix = "مية"
        elif hundreds == 2:
            prefix = "ميتين"
        else:
            prefix = f"{_ONES_AR[hundreds]} مية"
        return prefix if rest == 0 else f"{prefix} و{_num_ar(rest)}"
    if n < 1000000:
        thousands = n // 1000
        rest = n % 1000
        if thousands == 1:
            prefix = "ألف"
        elif thousands == 2:
            prefix = "ألفين"
        else:
            prefix = f"{_num_ar(thousands)} ألف"
        return prefix if rest == 0 else f"{prefix} و{_num_ar(rest)}"
    return str(n)


def _num_en(value: str | int | float) -> str:
    text = _clean_decimal(str(value))
    if "." in text:
        whole, frac = text.split(".", 1)
        return f"{_num_en(whole)} point {' '.join(_num_en(ch) for ch in frac[:2])}"
    try:
        n = int(text)
    except Exception:
        return str(value)
    if n < 0:
        return "minus " + _num_en(abs(n))
    if n < 20:
        return _ONES_EN[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        return _TENS_EN[tens] if ones == 0 else f"{_TENS_EN[tens]} {_ONES_EN[ones]}"
    if n < 1000:
        rest = n % 100
        prefix = f"{_ONES_EN[n // 100]} hundred"
        return prefix if rest == 0 else f"{prefix} {_num_en(rest)}"
    if n < 1000000:
        rest = n % 1000
        prefix = f"{_num_en(n // 1000)} thousand"
        return prefix if rest == 0 else f"{prefix} {_num_en(rest)}"
    return str(n)


def _place_for_voice(name: str, language: str) -> str:
    text = str(name or "").strip()
    if _lang(language) != "ar":
        return text
    return _AR_PLACE_NAMES.get(text.lower(), text)


def _condition_for_voice(condition: str, language: str) -> str:
    text = str(condition or "").strip()
    if _lang(language) != "ar":
        return text
    return _AR_CONDITIONS.get(text.lower(), text)


def _format_iso_date(match: re.Match, language: str) -> str:
    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3))
    try:
        date(year, month, day)
    except ValueError:
        return match.group(0)
    if _lang(language) == "ar":
        return f"يوم {_num_ar(day)} {_MONTHS_AR[month]}"
    return f"{_MONTHS_EN[month]} {_num_en(day)}"


def _format_time(match: re.Match, language: str) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2))
    if _lang(language) == "ar":
        period = "بعد الظهر" if hour >= 12 else "الصبح"
        hour12 = hour % 12 or 12
        if minute == 0:
            return f"{_num_ar(hour12)} {period}"
        if minute == 30:
            return f"{_num_ar(hour12)} ونص {period}"
        return f"{_num_ar(hour12)} و{_num_ar(minute)} دقيقة {period}"
    suffix = "pm" if hour >= 12 else "am"
    hour12 = hour % 12 or 12
    return f"{hour12}:{minute:02d} {suffix}"


def _normalize_units(text: str, language: str) -> str:
    lang = _lang(language)

    def temp_c(match):
        value = _clean_decimal(match.group(1))
        return f"{_num_ar(value)} درجة" if lang == "ar" else f"{value} degrees"

    def percent(match):
        value = _clean_decimal(match.group(1))
        return f"{_num_ar(value)} في المية" if lang == "ar" else f"{value} percent"

    def kmh(match):
        value = _clean_decimal(match.group(1))
        return f"{_num_ar(value)} كيلومتر في الساعة" if lang == "ar" else f"{value} kilometers per hour"

    def ms(match):
        value = _clean_decimal(match.group(1))
        return f"{_num_ar(value)} متر في الثانية" if lang == "ar" else f"{value} meters per second"

    def money_usd(match):
        value = _clean_decimal(match.group(1))
        return f"{_num_ar(value)} دولار" if lang == "ar" else f"{value} dollars"

    text = re.sub(r"([-+]?\d+(?:\.\d+)?)\s*(?:Â?°C|°C|\?C|C\b)", temp_c, text)
    text = re.sub(r"([-+]?\d+(?:\.\d+)?)\s*%", percent, text)
    text = re.sub(r"([-+]?\d+(?:\.\d+)?)\s*km/?h\b", kmh, text, flags=re.IGNORECASE)
    text = re.sub(r"([-+]?\d+(?:\.\d+)?)\s*m/?s\b", ms, text, flags=re.IGNORECASE)
    text = re.sub(r"\$\s*([-+]?\d+(?:\.\d+)?)", money_usd, text)
    text = re.sub(r"\bEGP\s*([-+]?\d+(?:\.\d+)?)", lambda m: f"{_num_ar(m.group(1))} جنيه" if lang == "ar" else f"{_clean_decimal(m.group(1))} Egyptian pounds", text, flags=re.IGNORECASE)
    return text


def normalize_for_voice(text: str, language: str = "en", persona: dict | None = None) -> str:
    if not VOICE_NORMALIZER_ENABLED:
        return str(text or "")
    lang = _lang(language)
    value = str(text or "")
    if not value:
        return value
    if not bool(VOICE_NORMALIZER_KEEP_URLS):
        value = _URL_RE.sub("", value)
    value = value.replace("→", " ").replace("↑", " ").replace("↓", " ").replace("&", " and ")
    value = re.sub(r"^[\s\-•*]+", "", value, flags=re.MULTILINE)
    value = _ISO_DATE_RE.sub(lambda m: _format_iso_date(m, lang), value)
    value = _TIME_RE.sub(lambda m: _format_time(m, lang), value)
    value = _normalize_units(value, lang)
    value = re.sub(r"\s{2,}", " ", value)
    return value.strip(" \n\t-•")


def normalize_weather_block(raw_block: str, language: str = "en", persona: dict | None = None) -> str:
    text = " ".join(str(raw_block or "").split()).strip()
    if not text:
        return ""
    lang = _lang(language)
    match = re.search(
        r"Weather in (?P<city>.*?): (?P<condition>.*?), (?P<temp>[-+]?\d+(?:\.\d+)?)\s*(?:Â?°C|°C|\?C|C), "
        r"humidity (?P<humidity>\d+(?:\.\d+)?)%, wind (?P<wind>\d+(?:\.\d+)?) km/h",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return normalize_for_voice(text, lang, persona)
    city = _place_for_voice(match.group("city"), lang)
    condition = _condition_for_voice(match.group("condition"), lang)
    temp = _clean_decimal(match.group("temp"))
    humidity = _clean_decimal(match.group("humidity"))
    wind = _clean_decimal(match.group("wind"))
    if lang == "ar":
        return (
            f"الطقس في {city} دلوقتي: {condition}. "
            f"الحرارة {_num_ar(temp)} درجة، الرطوبة {_num_ar(humidity)} في المية، "
            f"والرياح حوالي {_num_ar(wind)} كيلومتر في الساعة."
        )
    return (
        f"Weather in {city}: {condition}. "
        f"Temperature {temp} degrees, humidity {humidity} percent, "
        f"wind about {wind} kilometers per hour."
    )


def _source_name(line: str) -> str:
    match = re.search(r"\|\s*([^:]+):", line)
    if match:
        return match.group(1).strip()
    url = _URL_RE.search(line)
    if url:
        host = urlparse(url.group(0)).netloc.replace("www.", "")
        return host
    return ""


def normalize_search_block(raw_block: str, language: str = "en", persona: dict | None = None, max_results: int | None = None) -> str:
    lang = _lang(language)
    limit = max(1, int(max_results or VOICE_NORMALIZER_MAX_SEARCH_RESULTS or 2))
    lines = [line.strip() for line in str(raw_block or "").splitlines() if line.strip()]
    bullets = [line[1:].strip() if line.startswith("-") else line for line in lines]
    results = []
    for line in bullets:
        if not line or line.startswith("["):
            continue
        line = _URL_RE.sub("", line).strip()
        source = _source_name(line)
        line = re.sub(r"\s*\[[^\]]+\]\s*$", "", line).strip()
        if " | " in line:
            title, rest = line.split(" | ", 1)
        elif ":" in line:
            title, rest = line.split(":", 1)
        else:
            title, rest = line, ""
        title = normalize_for_voice(title.strip(), lang, persona)
        snippet = normalize_for_voice(rest.strip(), lang, persona)
        if not title:
            continue
        if lang == "ar":
            item = f"{title}."
            if snippet:
                item += f" {snippet}"
            if source:
                item += f" من {source}."
        else:
            item = f"{title}."
            if snippet:
                item += f" {snippet}"
            if source:
                item += f" From {source}."
        results.append(item)
        if len(results) >= limit:
            break
    return " ".join(results).strip()
