from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple, Dict

_AR_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_WEEKDAY_ALIASES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
    "الاثنين": 0,
    "الاتنين": 0,
    "اثنين": 0,
    "اتنين": 0,
    "التلات": 1,
    "تلات": 1,
    "الثلاثاء": 1,
    "الاربع": 2,
    "اربع": 2,
    "الاربعاء": 2,
    "الخميس": 3,
    "خميس": 3,
    "الجمعة": 4,
    "جمعة": 4,
    "السبت": 5,
    "سبت": 5,
    "الاحد": 6,
    "الأحد": 6,
    "حد": 6,
}

_RELATIVE_RE = re.compile(
    r"(?:^|\b)(?:in|after|بعد)\s+"
    r"(?P<qty>\d+(?:\.\d+)?|نص|ربع|ساعتين)\s*"
    r"(?P<unit>seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|"
    r"ثانية|ثواني|دقيقة|دقائق|دقايق|ساعة|ساعات|يوم|ايام|أيام|اسبوع|أسبوع|اسابيع|أسابيع)?",
    re.IGNORECASE,
)

_CLOCK_RE = re.compile(
    r"(?:\b(?:at|الساعة|الساعه|ساعه)\s*)?(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>am|pm|صباحا|صباحاً|صبح|مساءا|مساءً|مساء|م|ص)?(?!\d)",
    re.IGNORECASE,
)

_DATE_RE = re.compile(
    r"\b(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})\b|"
    r"\b(?P<day2>\d{1,2})[-/](?P<month2>\d{1,2})(?:[-/](?P<year2>\d{2,4}))?\b"
)

_TODAY_TOKENS = ("today", "النهاردة", "النهارده", "اليوم", "النهارده", "now")
_TOMORROW_TOKENS = ("tomorrow", "بكرة", "بكره", "بكرا")
_DAY_AFTER_TOMORROW_TOKENS = ("after tomorrow", "day after tomorrow", "بعد بكرة", "بعد بكره", "بعد بكرا")
_TONIGHT_TOKENS = ("tonight", "الليل", "بالليل")
_RECURRENCE_TOKENS = (
    "every day",
    "every week",
    "every month",
    "daily",
    "weekly",
    "monthly",
    "each day",
    "each week",
    "each month",
    "كل يوم",
    "كل اسبوع",
    "كل أسبوع",
    "كل شهر",
    "يومي",
    "اسبوعي",
    "شهري",
)

_RECURRING_WEEKDAY_ALIASES = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
    "الاثنين": 0,
    "الاتنين": 0,
    "اثنين": 0,
    "اتنين": 0,
    "التلات": 1,
    "تلات": 1,
    "الثلاثاء": 1,
    "الاربع": 2,
    "اربع": 2,
    "الاربعاء": 2,
    "الخميس": 3,
    "خميس": 3,
    "الجمعة": 4,
    "جمعة": 4,
    "السبت": 5,
    "سبت": 5,
    "الاحد": 6,
    "الأحد": 6,
    "حد": 6,
}


def _norm(text: str) -> str:
    return " ".join(str(text or "").translate(_AR_INDIC).lower().split()).strip()


def _parse_quantity(qty_text: str) -> float:
    value = _norm(qty_text)
    if value == "نص":
        return 0.5
    if value == "ربع":
        return 0.25
    if value == "ساعتين":
        return 2.0
    try:
        return float(value)
    except ValueError:
        match = re.search(r"\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                pass
    return 1.0


def _duration_delta(qty_text: str, unit_text: str) -> Optional[timedelta]:
    qty = _parse_quantity(qty_text)
    unit = _norm(unit_text)
    if not unit:
        unit = "seconds"
    if unit in {"h", "hr", "hrs", "hour", "hours", "ساعة", "ساعات"}:
        return timedelta(hours=qty)
    if unit in {"d", "day", "days", "يوم", "ايام", "أيام"}:
        return timedelta(days=qty)
    if unit in {"w", "week", "weeks", "اسبوع", "أسبوع", "اسابيع", "أسابيع"}:
        return timedelta(weeks=qty)
    if unit in {"m", "min", "mins", "minute", "minutes", "دقيقة", "دقائق", "دقايق"}:
        return timedelta(minutes=qty)
    return timedelta(seconds=qty)


def _extract_clock(text: str):
    match = _CLOCK_RE.search(text)
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    ampm = (match.group("ampm") or "").strip().lower() or None
    if ampm in {"صباحا", "صباحاً", "صبح", "ص"}:
        ampm = "am"
    elif ampm in {"مساءا", "مساءً", "مساء", "م"}:
        ampm = "pm"
    return hour, minute, ampm


def _next_weekday(now: datetime, weekday: int) -> datetime:
    delta = (weekday - now.weekday()) % 7
    if delta == 0:
        delta = 7
    return now + timedelta(days=delta)


def _extract_explicit_date(text: str, now: datetime) -> Optional[datetime]:
    match = _DATE_RE.search(text)
    if not match:
        return None
    if match.group("year"):
        year = int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day"))
    else:
        day = int(match.group("day2"))
        month = int(match.group("month2"))
        year_raw = match.group("year2")
        if year_raw:
            year = int(year_raw)
            if year < 100:
                year += 2000 if year < 70 else 1900
        else:
            year = now.year
            if (month, day) < (now.month, now.day):
                year += 1
    try:
        return datetime(year, month, day)
    except ValueError:
        return None


def _replace_tokens(text: str, tokens: tuple[str, ...]) -> str:
    result = text
    for token in tokens:
        result = re.sub(rf"\b{re.escape(token)}\b", " ", result, flags=re.IGNORECASE)
    return " ".join(result.split()).strip()


def _strip_recurrence_markers(text: str) -> str:
    result = str(text or "")
    for token in sorted(_RECURRENCE_TOKENS, key=len, reverse=True):
        result = re.sub(rf"\b{re.escape(token)}\b", " ", result, flags=re.IGNORECASE)
    result = re.sub(r"\b(?:every|each|كل)\b", " ", result, flags=re.IGNORECASE)
    return " ".join(result.split()).strip()


def parse_recurrence_spec(text: str) -> Tuple[Optional[str], Dict[str, object]]:
    """Extract a normalized recurrence spec from English/Egyptian-Arabic text."""
    normalized = _norm(text)
    if not normalized:
        return None, {}

    metadata: Dict[str, object] = {}
    if re.search(r"\b(?:daily|every day|each day|كل يوم|يومي)\b", normalized, re.IGNORECASE):
        return "daily", metadata
    if re.search(r"\b(?:weekly|every week|each week|كل اسبوع|كل أسبوع|اسبوعي|أسبوعي)\b", normalized, re.IGNORECASE):
        return "weekly", metadata
    if re.search(r"\b(?:monthly|every month|each month|كل شهر|شهري)\b", normalized, re.IGNORECASE):
        return "monthly", metadata

    weekday_match = None
    for name, index in _RECURRING_WEEKDAY_ALIASES.items():
        if re.search(rf"\b(?:every|كل)?\s*{re.escape(name)}\b", normalized, re.IGNORECASE):
            weekday_match = (name, index)
            break
    if weekday_match is not None:
        metadata["weekday"] = weekday_match[1]
        metadata["weekday_name"] = weekday_match[0]
        return "weekly", metadata

    return None, {}


def parse_natural_datetime(time_str: str, now: Optional[datetime] = None) -> Optional[datetime]:
    """Parse an English/Egyptian-Arabic natural language time string."""
    now = now or datetime.now()
    text = _norm(time_str)
    if not text:
        return None

    text = _strip_recurrence_markers(text)

    rel = _RELATIVE_RE.search(text)
    if rel:
        delta = _duration_delta(rel.group("qty") or "1", rel.group("unit") or "seconds")
        return now + delta

    base_date = None
    working = text

    if any(token in working for token in _DAY_AFTER_TOMORROW_TOKENS):
        base_date = now.date() + timedelta(days=2)
        working = _replace_tokens(working, _DAY_AFTER_TOMORROW_TOKENS)
    elif any(token in working for token in _TOMORROW_TOKENS):
        base_date = now.date() + timedelta(days=1)
        working = _replace_tokens(working, _TOMORROW_TOKENS)
    elif any(token in working for token in _TODAY_TOKENS):
        base_date = now.date()
        working = _replace_tokens(working, _TODAY_TOKENS)
    elif any(token in working for token in _TONIGHT_TOKENS):
        base_date = now.date()
        working = _replace_tokens(working, _TONIGHT_TOKENS)

    weekday = None
    for name, index in _WEEKDAY_ALIASES.items():
        if re.search(rf"\b{re.escape(name)}\b", working):
            weekday = index
            working = re.sub(rf"\b{re.escape(name)}\b", " ", working)
            break
    if weekday is not None:
        base_date = _next_weekday(now, weekday).date()

    explicit_date = _extract_explicit_date(working, now)
    if explicit_date is not None:
        base_date = explicit_date.date()

    clock = _extract_clock(working)
    if clock is None:
        if base_date is None:
            return None
        default_hour = 20 if any(token in text for token in _TONIGHT_TOKENS) else 9
        target = datetime(base_date.year, base_date.month, base_date.day, default_hour, 0, 0)
        if target <= now:
            target += timedelta(days=1)
        return target

    hour, minute, ampm = clock
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    elif ampm is None and 1 <= hour <= 6 and base_date is not None:
        hour += 12

    if base_date is None:
        base_date = now.date()

    target = datetime(base_date.year, base_date.month, base_date.day, hour % 24, minute, 0)
    if target <= now and base_date == now.date():
        target += timedelta(days=1)
    return target
