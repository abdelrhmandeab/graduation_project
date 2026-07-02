import os
import re
from dataclasses import dataclass, field

from core.config import CONFIRMATION_TOKEN_BYTES, CONFIRMATION_TOKEN_MIN_HEX_LEN
from nlp.codeswitching import convert_arabic_numerals, normalize_codeswitched, normalize_arabic_preserve_digits
from os_control.path_resolver import (
    DRIVE_ALIASES as _PATH_RESOLVER_DRIVE_ALIASES,
    FOLDER_ALIASES as _PATH_RESOLVER_FOLDER_ALIASES,
    SEARCH_PATH_ALIASES,
)
from os_control.temporal_parser import parse_recurrence_spec
from os_control.system_ops import normalize_system_action


@dataclass
class ParsedCommand:
    intent: str
    raw: str
    normalized: str
    action: str = ""
    args: dict = field(default_factory=dict)
    negated: bool = False


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------
_COLLAPSE_WS_RE = re.compile(r"\s+")
_MATCH_SANITIZE_RE = re.compile(r"[^a-z0-9_\s:\\/.\-\u0600-\u06FF]")
_DRIVE_COLON_RE = re.compile(r"\b([a-z])\s*:", flags=re.IGNORECASE)
_DRIVE_WORD_RE = re.compile(r"\b([a-z])\s+(?:drive|partition)\b", flags=re.IGNORECASE)
_SEA_C_DRIVE_RE = re.compile(r"\b(?:sea|see|cee)\s+(?:drive|partition)\b", flags=re.IGNORECASE)
_OPEN_FILLER_PREFIXES = (
    r"^(?:for me|for us|for me now|for me please)\s+",
    r"^(?:\u0645\u0646 \u0641\u0636\u0644\u0643|\u0644\u0648 \u0633\u0645\u062d\u062a|\u0631\u062c\u0627\u0621|\u0627\u0644\u0631\u062c\u0627\u0621)\s+",
    r"^(?:the)\s+",
)
_FILESYSTEM_OPEN_HINTS = (
    "drive",
    "partition",
    "folder",
    "directory",
    "desktop",
    "downloads",
    "documents",
    "\u0642\u0631\u0635",
    "\u0628\u0627\u0631\u062a\u0634\u0646",
    "\u0642\u0633\u0645",
    "\u062f\u0631\u0627\u064a\u0641",
    "\u0645\u062c\u0644\u062f",
    "\u0645\u0644\u0641",
    "\u0633\u0637\u062d \u0627\u0644\u0645\u0643\u062a\u0628",
    "\u0627\u0644\u062a\u062d\u0645\u064a\u0644\u0627\u062a",
    "\u0627\u0644\u0645\u0633\u062a\u0646\u062f\u0627\u062a",
    "\u0627\u0644\u0635\u0648\u0631",
    "\u0627\u0644\u0641\u064a\u062f\u064a\u0648\u0647\u0627\u062a",
)
# Imported from path_resolver \u2014 single source of truth for folder aliases.
_SPECIAL_FOLDER_ALIASES = _PATH_RESOLVER_FOLDER_ALIASES
_SEARCH_PATH_ALIASES = SEARCH_PATH_ALIASES
_MEDIA_APP_TARGETS = {
    "spotify": "spotify",
    "vlc": "vlc",
    "youtube music": "youtube music",
    "yt music": "youtube music",
    "youtube": "youtube music",
    "music": "spotify",
    "play music": "spotify",
    "\u0633\u0628\u0648\u062a\u064a\u0641\u0627\u064a": "spotify",
    "\u0641\u064a \u0627\u0644 \u0633\u064a": "vlc",
    "\u064a\u0648\u062a\u064a\u0648\u0628 \u0645\u064a\u0648\u0632\u0643": "youtube music",
    "\u0634\u063a\u0644 \u0645\u0648\u0633\u064a\u0642\u0649": "spotify",
}
_NATURAL_APP_ALIASES = {
    "calculator": "calculator",
    "calc": "calculator",
    "notepad": "notepad",
    "text editor": "notepad",
    "editor": "notepad",
    "chrome": "chrome",
    "google chrome": "chrome",
    "edge": "edge",
    "microsoft edge": "edge",
    "spotify": "spotify",
    "vlc": "vlc",
    "firefox": "firefox",
    "fire fox": "firefox",
    "mozilla firefox": "firefox",
    "youtube music": "youtube music",
    "yt music": "youtube music",
    "youtube": "youtube music",
    "explorer": "file explorer",
    "file explorer": "file explorer",
    "\u0627\u0644\u062d\u0627\u0633\u0628\u0629": "calculator",
    "\u0646\u0648\u062a \u0628\u0627\u062f": "notepad",
    "\u0627\u0644\u0645\u0641\u0643\u0631\u0629": "notepad",
    "\u0643\u0631\u0648\u0645": "chrome",
    "\u062c\u0648\u062c\u0644 \u0643\u0631\u0648\u0645": "chrome",
    "\u0633\u0628\u0648\u062a\u064a\u0641\u0627\u064a": "spotify",
    "\u0641\u0627\u064a\u0631\u0641\u0648\u0643\u0633": "firefox",
    "\u0641\u0627\u064a\u0631 \u0641\u0648\u0643\u0633": "firefox",
    "\u0645\u0648\u0632\u064a\u0644\u0627 \u0641\u0627\u064a\u0631\u0641\u0648\u0643\u0633": "firefox",
}
_NATURAL_APP_REQUEST_PATTERNS = (
    re.compile(
        r"^(?:i\s+)?(?:need|want)(?:\s+to\s+(?:use|open|launch|start))?\s+(.+?)(?:\s+(?:now|right\s+now|please))?$",
        re.IGNORECASE,
    ),
    re.compile(r"^(?:can\s+i\s+get|give\s+me)\s+(.+?)(?:\s+(?:now|please))?$", re.IGNORECASE),
    re.compile(
        (
            r"^(?:\u0639\u0627\u064a\u0632|\u0639\u0627\u0648\u0632)"
            r"(?:\s+(?:\u0627\u0646|\u0623\u0646))?\s+(.+)$"
        ),
        re.IGNORECASE,
    ),
)
_URL_RE = re.compile(r"^(?:https?://|www\.)[^\s]+$", flags=re.IGNORECASE)
_DOMAIN_RE = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?$", flags=re.IGNORECASE)
_WINDOW_QUERY_ALIASES = {
    "google chrome": "chrome",
    "chrome window": "chrome",
    "chrome": "chrome",
    "كروم": "chrome",
    "جوجل كروم": "chrome",
    "spotify": "spotify",
    "سبوتيفاي": "spotify",
    "firefox": "firefox",
    "mozilla firefox": "firefox",
    "فايرفوكس": "firefox",
    "فاير فوكس": "firefox",
    "vlc": "vlc",
    "notepad": "notepad",
    "نوت باد": "notepad",
    "المفكرة": "notepad",
}
_DURATION_UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "ثانية": 1,
    "ثواني": 1,
    "ثانيتين": 2,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "دقيقة": 60,
    "دقائق": 60,
    "دقايق": 60,
    "دقيقتين": 120,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "ساعة": 3600,
    "ساعات": 3600,
    "ساعه": 3600,
    "ساعتين": 7200,
}
# Egyptian Arabic fraction words used as duration quantities
_AR_DURATION_FRACTIONS = {"نص": 0.5, "ربع": 0.25}
_NUMBER_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "صفر": 0,
    "واحد": 1,
    "واحدة": 1,
    "اتنين": 2,
    "اثنين": 2,
    "اثنتين": 2,
    "اتنان": 2,
    "ثلاثة": 3,
    "تلاتة": 3,
    "تلاته": 3,
    "تلات": 3,
    "ثلاث": 3,
    "اربعة": 4,
    "أربعة": 4,
    "أربع": 4,
    "اربع": 4,
    "خمسة": 5,
    "خمسه": 5,
    "خمس": 5,
    "ستة": 6,
    "سته": 6,
    "ست": 6,
    "سبعة": 7,
    "سبعه": 7,
    "سبع": 7,
    "ثمانية": 8,
    "تمانية": 8,
    "تمانيه": 8,
    "تماني": 8,
    "تسعة": 9,
    "تسعه": 9,
    "تسع": 9,
    "عشرة": 10,
    "عشره": 10,
    "عشر": 10,
    "حداشر": 11,
    "إحدى عشر": 11,
    "اتناشر": 12,
    "اثنا عشر": 12,
    "تلتاشر": 13,
    "أربعتاشر": 14,
    "خمستاشر": 15,
    "ستاشر": 16,
    "سبعتاشر": 17,
    "تمنتاشر": 18,
    "تسعتاشر": 19,
    "عشرين": 20,
}
_NUMBER_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "عشرين": 20,
    "ثلاثين": 30,
    "اربعين": 40,
    "خمسين": 50,
    "ستين": 60,
    "سبعين": 70,
    "ثمانين": 80,
    "تسعين": 90,
}
_CONFIRMATION_TOKEN_MAX_HEX_LEN = max(int(CONFIRMATION_TOKEN_MIN_HEX_LEN), int(CONFIRMATION_TOKEN_BYTES) * 2)


_TRAILING_PUNCT_RE = re.compile(r"[.,،؟?!]+$")


def _normalize_for_match(text: str) -> str:
    lowered = " ".join((text or "").lower().split()).strip()
    lowered = _TRAILING_PUNCT_RE.sub("", lowered).strip()
    cleaned = _MATCH_SANITIZE_RE.sub(" ", lowered)
    return _COLLAPSE_WS_RE.sub(" ", cleaned).strip()


def _int_from_numeric_text(text: str) -> int:
    return int(convert_arabic_numerals(str(text or "")).strip())


def _looks_like_explicit_level_request(text: str) -> bool:
    normalized = _normalize_for_match(text)
    if not normalized:
        return False
    if re.search(r"\b(?:to|at|=|ل|لـ|الى|إلى|على)\s*[0-9٠-٩]{1,3}\b", normalized, flags=re.IGNORECASE):
        return True
    if re.match(
        r"^(?:volume|sound|brightness|screen\s+brightness|الصوت|السطوع|الإضاءة|الاضاءة|النور)\s+[0-9٠-٩]{1,3}\b",
        normalized,
        flags=re.IGNORECASE,
    ):
        return True
    if re.search(r"\b(?:في\s+)?الم(?:ية|ئه|ئة|يه|ائه)\b", normalized):
        return True
    if any(token in normalized.split() for token in ("نص", "نصف", "ربع", "تلت", "ثلث")):
        return True
    return False


# ---------------------------------------------------------------------------
# Negation detection and handling
# ---------------------------------------------------------------------------

_NEGATION_RE = re.compile(r"^(?:do not|don't|don t|dont)\b|^(?:\u0644\u0627\b|\u0645\u0634\b|\u0645\u0627\b)", flags=re.IGNORECASE)


def _detect_and_strip_negation(text: str):
    """Detect a leading negation token in normalized text and return
    (negated: bool, stripped_text: str).
    Handles simple English (don't, do not) and Arabic (لا, مش, ما) prefixes.
    """
    if not text:
        return False, text
    m = _NEGATION_RE.match(text)
    if not m:
        return False, text
    # strip matched prefix
    stripped = text[m.end():].strip()
    return True, stripped


def _apply_negation_to_parsed(parsed: ParsedCommand) -> ParsedCommand:
    """Apply conservative negation inversions for common intents.
    This function mutates and returns the parsed command.
    """
    if not parsed or not parsed.intent:
        return parsed
    # Invert simple app open -> app close
    if parsed.intent == "OS_APP_OPEN":
        parsed.intent = "OS_APP_CLOSE"
        return parsed

    if parsed.intent == "OS_SYSTEM_COMMAND":
        ak = parsed.args.get("action_key")
        if ak == "wifi_on":
            parsed.args["action_key"] = "wifi_off"
        elif ak == "bluetooth_on":
            parsed.args["action_key"] = "bluetooth_off"
        elif ak == "notifications_on":
            parsed.args["action_key"] = "notifications_off"
        elif ak == "notifications_off":
            parsed.args["action_key"] = "notifications_on"
        elif ak == "media_play":
            parsed.args["action_key"] = "media_stop"
        elif ak == "volume_up":
            parsed.args["action_key"] = "volume_down"
        elif ak == "volume_down":
            parsed.args["action_key"] = "volume_up"
        elif ak == "volume_mute":
            parsed.args["action_key"] = "volume_unmute"
        elif ak == "volume_unmute":
            parsed.args["action_key"] = "volume_mute"
        elif ak == "brightness_up":
            parsed.args["action_key"] = "brightness_down"
        elif ak == "media_play":
            parsed.args["action_key"] = "media_play_pause"
        elif ak == "media_next":
            parsed.args["action_key"] = "media_next_track"
        elif ak == "media_prev":
            parsed.args["action_key"] = "media_previous_track"
    return parsed


def _normalize_audio_profile(mode_str: str) -> str:
    """Normalize audio profile mode names to canonical form."""
    m = _normalize_for_match(mode_str)
    if m in {"fast", "low latency", "low_latency", "responsive"}:
        return "responsive"
    if m in {"balanced", "normal"}:
        return "balanced"
    if m in {"robust", "stable", "reliable", "noisy"}:
        return "robust"
    return m.replace(" ", "_")


def _normalize_browser_action(action_hint: str) -> str:
    """Normalize browser control action to canonical form."""
    m = _normalize_for_match(action_hint)
    if m in {"new", "new tab", "open tab", "create tab", "تاب جديد", "تاب"}:
        return "new_tab"
    if m in {"close", "close tab", "remove tab", "delete tab", "اقفل التاب", "سكر التاب"}:
        return "close_tab"
    if m in {"back", "go back", "previous", "ارجع", "ارجع للخلف"}:
        return "back"
    if m in {"forward", "go forward", "next", "روح لقدام", "قدام"}:
        return "forward"
    return m


def _normalize_window_action(action_hint: str) -> str:
    """Normalize window control action to canonical form."""
    m = _normalize_for_match(action_hint)
    if m in {"maximize", "max", "fullscreen", "كبّر", "أكبر"}:
        return "maximize"
    if m in {"minimize", "min", "shrink", "صغّر"}:
        return "minimize"
    if m in {"snap left", "snap to left", "half left", "left half", "خش للشمال"}:
        return "snap_left"
    if m in {"snap right", "snap to right", "half right", "right half", "خش لليمين"}:
        return "snap_right"
    return m


def _strip_spoken_prefixes(normalized_text: str) -> str:
    candidate = (normalized_text or "").strip()
    # Strip Arabic comma that STT inserts after the wake word ("Jarvis،").
    candidate = re.sub(r"^jarvis\s*[،,]\s*", "", candidate, flags=re.IGNORECASE).strip()
    patterns = (
        r"^(?:hey|ok|okay)\s+jarvis\s+",
        r"^(?:hey|ok|okay)\s+",
        r"^jarvis\s+",
        r"^please\s+",
        r"^(?:please\s+)?(?:can|could|would|will)\s+you\s+",
        r"^(?:please\s+)?(?:i need you to|i want you to|i want to)\s+",
        r"^(?:\u064a\u0627\s+)?\u062c\u0627\u0631\u0641\u064a\u0633\s+",
        r"^(?:\u0645\u0646 \u0641\u0636\u0644\u0643|\u0644\u0648 \u0633\u0645\u062d\u062a|\u0631\u062c\u0627\u0621|\u0627\u0644\u0631\u062c\u0627\u0621)\s+",
        r"^(?:\u0647\u0644 \u064a\u0645\u0643\u0646\u0643|\u0647\u0644 \u062a\u0633\u062a\u0637\u064a\u0639|\u0645\u0645\u0643\u0646)\s+",
        r"^(?:\u0627\u0631\u064a\u062f\u0643 \u0627\u0646|\u0623\u0631\u064a\u062f\u0643 \u0623\u0646|\u0627\u0631\u064a\u062f|\u0623\u0631\u064a\u062f|\u0639\u0627\u064a\u0632\u0643|\u0639\u0627\u064a\u0632)\s+(?:\u0627\u0646|\u0623\u0646)?\s*",
    )
    for pattern in patterns:
        candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()
    return candidate


def _extract_drive_letter(text: str):
    if _SEA_C_DRIVE_RE.search(text or ""):
        return "C"
    for pattern in (_DRIVE_COLON_RE, _DRIVE_WORD_RE):
        match = pattern.search(text or "")
        if match:
            return match.group(1).upper()
    return None


def _is_drive_open_request(text: str) -> bool:
    lowered = (text or "").lower()
    explicit_verbs = (
        "open",
        "show",
        "browse",
        "access",
        "enter",
        "\u0627\u0641\u062a\u062d",
        "\u0627\u0641\u062a\u062d\u0644\u064a",
        "\u0648\u0631\u064a\u0646\u064a",
        "\u0647\u0627\u062a\u0644\u064a",
        "\u062e\u0634",
    )
    if any(verb in lowered for verb in explicit_verbs):
        return True
    if "go to" in lowered and ("drive" in lowered or "partition" in lowered):
        return True
    if "\u0631\u0648\u062d \u0639\u0644\u0649" in lowered and ("\u062f\u0631\u0627\u064a\u0641" in lowered or "\u0642\u0631\u0635" in lowered):
        return True
    return False


def _strip_open_fillers(text: str) -> str:
    candidate = (text or "").strip()
    for pattern in _OPEN_FILLER_PREFIXES:
        candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()
    return candidate


def _special_folder_path(text: str):
    lowered = (text or "").lower()
    user_home = os.path.expanduser("~")
    for key, folder_name in _SPECIAL_FOLDER_ALIASES.items():
        if key in lowered:
            return os.path.join(user_home, folder_name)
    return None


def _looks_like_filesystem_target(text: str) -> bool:
    lowered = (text or "").lower()
    if any(hint in lowered for hint in _FILESYSTEM_OPEN_HINTS):
        return True
    if "\\" in lowered or "/" in lowered:
        return True
    if re.search(r"\b[a-z]:\\", lowered):
        return True
    return False


def _collapse_repeated_phrase(text: str) -> str:
    candidate = " ".join((text or "").split()).strip()
    if not candidate:
        return ""

    tokens = candidate.split(" ")
    if len(tokens) >= 2 and len(tokens) % 2 == 0:
        half = len(tokens) // 2
        if tokens[:half] == tokens[half:]:
            return " ".join(tokens[:half])

    lower = candidate.lower()
    for sep in (" in ", " on ", " inside ", " \u0641\u064a ", " \u062f\u0627\u062e\u0644 "):
        parts = [segment.strip() for segment in lower.split(sep) if segment.strip()]
        if len(parts) >= 2 and len(set(parts)) == 1:
            return parts[0]
    return candidate


_AR_ARTICLE_PREFIX_RE = re.compile(
    r"^(?:الـ|لل|ال|لـ|ل)\s*",
    re.IGNORECASE,
)
_EN_ARTICLE_PREFIX_RE = re.compile(r"^(?:the|my)\s+", re.IGNORECASE)


def _normalize_search_path_hint(path_hint: str):
    candidate = _collapse_repeated_phrase(path_hint)
    if not candidate:
        return None

    # Strip trailing punctuation that STT appends ("Downloads." → "Downloads").
    candidate = re.sub(r"[.,،؟?!]+$", "", candidate).strip()
    if not candidate:
        return None

    lowered = candidate.lower().strip()
    # Try exact alias first.
    alias = _SEARCH_PATH_ALIASES.get(lowered)
    if alias:
        return os.path.join(os.path.expanduser("~"), alias)

    # Strip Arabic definite-article prefix and retry ("الـ desktop" → "desktop").
    stripped = _AR_ARTICLE_PREFIX_RE.sub("", lowered).strip()
    if stripped != lowered:
        alias = _SEARCH_PATH_ALIASES.get(stripped)
        if alias:
            return os.path.join(os.path.expanduser("~"), alias)
        lowered = stripped

    # Strip English article prefix and retry ("the desktop" → "desktop").
    stripped_en = _EN_ARTICLE_PREFIX_RE.sub("", lowered).strip()
    if stripped_en != lowered:
        alias = _SEARCH_PATH_ALIASES.get(stripped_en)
        if alias:
            return os.path.join(os.path.expanduser("~"), alias)
        lowered = stripped_en

    # Drive/partition phrases ("D partition", "قرص د", "C drive") — resolved
    # via the canonical path_resolver table (folders already tried above).
    drive_path = _PATH_RESOLVER_DRIVE_ALIASES.get(lowered)
    if drive_path:
        return drive_path

    return candidate


def _normalize_natural_app_target(value: str):
    candidate = _strip_open_fillers(_normalize_for_match(value))
    if not candidate:
        return ""

    for pattern in (
        r"\b(?:app|application|program)\b",
        r"\b(?:for me|for us|please|now|right now)\b",
        r"(?:\u062a\u0637\u0628\u064a\u0642|\u0628\u0631\u0646\u0627\u0645\u062c)",
        r"(?:\u0645\u0646 \u0641\u0636\u0644\u0643|\u0644\u0648 \u0633\u0645\u062d\u062a|\u0631\u062c\u0627\u0621|\u0627\u0644\u0622\u0646|\u0627\u0644\u0627\u0646)",
    ):
        candidate = re.sub(pattern, " ", candidate, flags=re.IGNORECASE)

    return " ".join(candidate.split()).strip()


def _infer_known_app_name(target_text: str):
    candidate = _normalize_natural_app_target(target_text)
    if not candidate:
        return None

    direct = _NATURAL_APP_ALIASES.get(candidate)
    if direct:
        return direct

    for alias in sorted(_NATURAL_APP_ALIASES.keys(), key=len, reverse=True):
        if (
            candidate.startswith(alias + " ")
            or candidate.endswith(" " + alias)
            or (" " + alias + " ") in (" " + candidate + " ")
        ):
            return _NATURAL_APP_ALIASES[alias]
    return None


def _parse_spoken_int(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(float(value))

    from nlp.codeswitching import convert_arabic_numerals
    text = _normalize_for_match(convert_arabic_numerals(str(value or "")))
    if not text:
        return None

    digit = re.search(r"\d{1,4}", text)
    if digit:
        return int(digit.group(0))

    tokens = text.split()
    total = 0
    current = 0
    found = False
    for token in tokens:
        if token in {"and", "و"}:
            continue
        if token in _NUMBER_ONES:
            current += _NUMBER_ONES[token]
            found = True
            continue
        if token in _NUMBER_TENS:
            current += _NUMBER_TENS[token]
            found = True
            continue
        if token in {"hundred", "مئة", "ماية", "مية"}:
            current = max(1, current) * 100
            found = True
            continue
    if not found:
        return None
    return total + current


def _duration_to_seconds(number_value, unit_text):
    number = _parse_spoken_int(number_value)
    if number is None:
        frac_key = _normalize_for_match(str(number_value or ""))
        frac_val = _AR_DURATION_FRACTIONS.get(frac_key)
        if frac_val is None:
            return None
        number = frac_val
    unit = _normalize_for_match(unit_text)
    factor = _DURATION_UNIT_SECONDS.get(unit, 1)
    return max(1, min(86400, int(number * factor)))


def parse_duration_from_text(text):
    candidate = _normalize_for_match(text)
    if not candidate:
        return None

    candidate = re.sub(r"^(?:for|in|after|على|علي|ل(?:ـ)?|الى|إلى|بعد)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+(?:for|in|after|على|علي|ل(?:ـ)?|الى|إلى|بعد)$", "", candidate, flags=re.IGNORECASE)
    candidate = " ".join(candidate.split()).strip()
    if not candidate:
        return None

    duration_units = r"seconds?|secs?|minutes?|mins?|hours?|hrs?|ثانية|ثواني|ثانيتين|دقيقة|دقائق|دقايق|دقيقتين|ساعة|ساعات|ساعتين"

    explicit_match = re.search(
        rf"\b(.+?)\s+({duration_units})\b",
        candidate,
        flags=re.IGNORECASE,
    )
    if explicit_match:
        seconds = _duration_to_seconds(explicit_match.group(1), explicit_match.group(2))
        if seconds is not None:
            return seconds

    reverse_match = re.search(
        rf"\b({duration_units})\s+(.+?)\b",
        candidate,
        flags=re.IGNORECASE,
    )
    if reverse_match:
        seconds = _duration_to_seconds(reverse_match.group(2), reverse_match.group(1))
        if seconds is not None:
            return seconds

    unit_only = _DURATION_UNIT_SECONDS.get(candidate)
    if unit_only is not None:
        return max(1, min(86400, int(unit_only)))

    spoken = _parse_spoken_int(candidate)
    if spoken is not None:
        return spoken

    return None


_TIMER_FILLER_WORDS = frozenset({
    "the", "a", "an", "my", "for", "of",
    "timer", "alarm", "تايمر", "منبه",
})


def _timer_args_from_text(duration_text, *, label="Timer"):
    seconds = parse_duration_from_text(duration_text)
    if seconds is None:
        return {}
    return {"seconds": seconds, "label": label}


def _extract_named_timer_label(text: str) -> str:
    """Extract a human label from text like 'pasta' or 'the pasta timer'."""
    tokens = [t for t in text.lower().split() if t not in _TIMER_FILLER_WORDS]
    return " ".join(tokens).strip() or "Timer"


def _named_timer_args(m) -> dict:
    """Parse 'set a timer for <label> for <duration>' regexes."""
    label_text = m.group(1).strip()
    duration_text = m.group(2).strip()
    label = _extract_named_timer_label(label_text) or "Timer"
    seconds = parse_duration_from_text(duration_text)
    if seconds is None:
        return {}
    return {"seconds": seconds, "label": label.capitalize()}


def _normalize_url_target(value: str):
    candidate = str(value or "").strip().strip('"').strip("'")
    candidate = re.sub(r"^(?:website|site|url|لينك|ويبسايت)\s+", "", candidate, flags=re.IGNORECASE).strip()
    if not candidate:
        return ""
    if _URL_RE.match(candidate):
        return f"https://{candidate}" if candidate.lower().startswith("www.") else candidate
    if _DOMAIN_RE.match(candidate):
        return f"https://{candidate}"
    return ""


def _canonical_window_query(value: str):
    normalized = _normalize_for_match(value)
    if not normalized:
        return ""
    direct = _WINDOW_QUERY_ALIASES.get(normalized)
    if direct:
        return direct
    for alias, canonical in sorted(_WINDOW_QUERY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if (
            normalized.startswith(alias + " ")
            or normalized.endswith(" " + alias)
            or (" " + alias + " ") in (" " + normalized + " ")
        ):
            return canonical
    return value.strip()


def _strip_create_name(value: str) -> str:
    """Strip leading filler words for folder creation commands.

    "اسمه test" → "test", "called reports" → "reports", "new folder test" → "test".
    """
    candidate = str(value or "").strip()
    # Arabic: "اسمه X", "باسم X", "بالاسم X"
    candidate = re.sub(r"^(?:اسمه|اسمها|باسم|بالاسم)\s+", "", candidate, flags=re.IGNORECASE).strip()
    # English: "called X", "named X"
    candidate = re.sub(r"^(?:called|named)\s+", "", candidate, flags=re.IGNORECASE).strip()
    # Strip leading "folder", "new folder", "new"
    candidate = re.sub(r"^(?:new\s+)?(?:folder|مجلد)\s+", "", candidate, flags=re.IGNORECASE).strip()
    return candidate.strip().strip('"').strip("'")


def _strip_file_target_fillers(value: str):
    candidate = _normalize_for_match(value)
    if not candidate:
        return ""
    candidate = re.sub(r"^(?:the\s+)?(?:file|folder)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^(?:\u0627\u0644)?(?:\u0645\u0644\u0641|\u0627\u0644\u0645\u062c\u0644\u062f|\u0645\u062c\u0644\u062f)\s+", "", candidate, flags=re.IGNORECASE)
    candidate = re.sub(r"^(?:\u062c\u062f\u064a\u062f\s+\u0628\u0627\u0633\u0645\s+|\u0628\u0627\u0633\u0645\s+)", "", candidate, flags=re.IGNORECASE)
    # Strip trailing filler words that the STT often appends ("cv file" \u2192 "cv")
    candidate = re.sub(r"\s+(?:files?|folders?|documents?|docs?)$", "", candidate, flags=re.IGNORECASE)
    return candidate.strip()


def _normalize_language_value(value: str):
    token = _normalize_for_match(value)
    if token in {"ar", "arabic", "عربي", "مصري", "المصري"}:
        return "ar"
    if token in {"en", "english", "انجليزي", "انجلش"}:
        return "en"
    return token


_TRAILING_LOCATION_RE = re.compile(
    r"^(.*?)\s+(?:in|من|في|from)\s+(.+)$",
    re.IGNORECASE,
)
# Greedy version — splits at the LAST في/in so "CV.pdf في الـ Documents"
# keeps "CV.pdf" intact as the filename rather than just "CV".
_TRAILING_LOCATION_RE_GREEDY = re.compile(
    r"^(.*)\s+(?:in|من|في|from)\s+(.+)$",
    re.IGNORECASE,
)


def _split_filename_and_location(text):
    """Split "X in Y" / "X في Y" into (filename, search_path).

    Returns (cleaned_filename, resolved_search_path_or_empty). Used by the
    codeswitched-command tier so OS_FILE_SEARCH results get the same
    filename-filler-stripping and location-alias resolution as the regex
    table, regardless of which tier matched first.
    """
    candidate = str(text or "").strip()
    if not candidate:
        return "", ""

    # Try greedy split first (splits at LAST في/in) so filenames with
    # dots like "CV.pdf في Documents" are not broken after the first word.
    greedy = _TRAILING_LOCATION_RE_GREEDY.match(candidate)
    if greedy:
        filename_part = _strip_file_target_fillers(greedy.group(1))
        location_part = _normalize_search_path_hint(greedy.group(2).strip()) or ""
        if filename_part:
            return filename_part, location_part

    # Fallback to non-greedy (handles edge cases like "X and Y in Z")
    match = _TRAILING_LOCATION_RE.match(candidate)
    if match:
        filename_part = _strip_file_target_fillers(match.group(1))
        location_part = _normalize_search_path_hint(match.group(2).strip()) or ""
        if filename_part:
            return filename_part, location_part
    return _strip_file_target_fillers(candidate), ""


# Trailing STT noise words that sometimes get appended to the end of a
# destructive-command utterance ("...downloads. Done." / "...محذوف.").
_TRAILING_NOISE_RE = re.compile(
    r"[.,،؟?!]*\s*(?:done|ok|okay|please|محذوف|تم|خلاص|كده)?\s*[.,،؟?!]*$",
    re.IGNORECASE,
)


def _split_target_and_location(text):
    """Split a delete/move/rename source phrase into (target, search_path).

    Like _split_filename_and_location but for destructive-op targets: also
    strips trailing STT noise words ("done", "محذوف") that often get
    appended after a location clause.  Returns (target, search_path_or_"").
    """
    candidate = str(text or "").strip()
    if not candidate:
        return "", ""
    match = _TRAILING_LOCATION_RE.match(candidate)
    if match:
        target_part = _strip_file_target_fillers(match.group(1))
        location_raw = _TRAILING_NOISE_RE.sub("", match.group(2).strip()).strip()
        location_part = _normalize_search_path_hint(location_raw) or ""
        if target_part:
            return target_part, location_part
    cleaned = _TRAILING_NOISE_RE.sub("", candidate).strip()
    return _strip_file_target_fillers(cleaned), ""


def _try_codeswitched_command(raw, normalized):
    cs = normalize_codeswitched(raw)
    intent = str((cs or {}).get("intent") or "").strip().lower()
    entity = str((cs or {}).get("entity") or "").strip()
    entity_normalized = _normalize_for_match(entity)
    entities = cs  # alias for legacy references below

    if not intent or not entity:
        return None

    if intent == "open":
        # "افتح tab جديدة في الـ browser" → browser_new_tab
        _tab_tokens = {"tab", "تاب", "new tab", "تاب جديد", "تاب جديدة"}
        if entity_normalized in _tab_tokens or entity_normalized.startswith("tab") or entity_normalized.startswith("تاب"):
            return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "browser_new_tab"})

        app_name = _infer_known_app_name(entity)
        if app_name:
            return ParsedCommand("OS_APP_OPEN", raw, normalized, args={"app_name": app_name})

        if entity_normalized in {"files", "file", "folder", "folders", "directory", "directories", "المفات", "الملفات", "المجلد", "المجلدات"}:
            return ParsedCommand("OS_FILE_NAVIGATION", raw, normalized, action="list_directory", args={"path": ""})

        if entity_normalized in {"music", "spotify", "vlc", "youtube music", "youtube", "song", "songs", "الموسيقى", "المزيكا"}:
            if normalized.startswith(("play ", "start ", "resume ", "pause ")):
                return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "media_play_pause"})
            app_name = _infer_known_app_name(entity) or _infer_known_app_name("spotify")
            if app_name:
                return ParsedCommand("OS_APP_OPEN", raw, normalized, args={"app_name": app_name})

    if intent == "close":
        # "اقفل tab الـ YouTube" / "close YouTube tab" → browser_close_named_tab
        entity_lower = entity_normalized.lower()
        has_tab = (
            any(t in entity_lower.split() for t in ("tab", "تاب"))
            or "tab" in raw.lower()
            or "تاب" in raw
        )
        if has_tab:
            import re as _re
            # The codeswitched parser often captures only "tab" as the entity
            # when the site name follows the Latin word "tab".
            # Try to extract the site name from raw text directly.
            _ar_tab_re = _re.compile(
                r"(?:اقفل|سكر|قفل)\s+tab\s+(?:الـ\s+|ال)?(.+?)(?:\s+في\s+(?:الـ\s+)?(?:browser|متصفح|chrome|firefox|edge))?$",
                _re.IGNORECASE | _re.UNICODE,
            )
            m = _ar_tab_re.search(raw)
            if m:
                tab_query = m.group(1).strip()
            else:
                tab_query = _re.sub(
                    r"\b(tab|تاب|browser|الـ|في|in|the|a|an)\b",
                    " ",
                    entity,
                    flags=_re.IGNORECASE | _re.UNICODE,
                ).strip()
            if not tab_query or tab_query.lower() == "tab":
                # Couldn't extract a site name; fall through to generic close-tab
                return ParsedCommand(
                    "OS_SYSTEM_COMMAND", raw, normalized,
                    args={"action_key": "browser_close_tab"},
                )
            return ParsedCommand(
                "OS_SYSTEM_COMMAND", raw, normalized,
                args={"action_key": "browser_close_named_tab", "tab_query": tab_query.lower()},
            )
        app_name = _infer_known_app_name(entity)
        if app_name:
            return ParsedCommand("OS_APP_CLOSE", raw, normalized, args={"app_name": app_name})

    if intent == "search":
        source_text = str((entities or {}).get("source_text") or raw).strip()
        source_norm = _normalize_for_match(source_text)
        explicit_search_markers = (
            "search",
            "google",
            "look up",
            "ابحث",
            "دور",
            "دوّر",
        )
        question_markers = (
            "tell me",
            "what",
            "who",
            "when",
            "where",
            "why",
            "how",
            "ممكن تقول",
            "قولي",
            "ايه",
            "إيه",
            "اخبار",
            "أخبار",
            "النهاردة",
            "النهارده",
        )
        looks_informational_question = any(marker in source_norm for marker in question_markers)
        has_explicit_search_verb = any(marker in source_norm for marker in explicit_search_markers)
        if looks_informational_question and not has_explicit_search_verb:
            return ParsedCommand("LLM_QUERY", raw, normalized)

        if entity_normalized in {"files", "file", "folder", "folders", "document", "documents", "الملفات", "المستندات", "المجلد", "المجلدات"}:
            query = str((entities or {}).get("source_text") or raw).strip()
            query = re.sub(
                r"^(?:search files? for|search for|look for|find|search|ابحث عن|دور على|دوّر على)\s+",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip()
            if query:
                filename, search_path = _split_filename_and_location(query)
                if filename:
                    return ParsedCommand("OS_FILE_SEARCH", raw, normalized, args={"filename": filename, "search_path": search_path})

        if entity:
            web_terms = {
                "web",
                "google",
                "youtube",
                "gmail",
                "maps",
                "news",
                "weather",
                "images",
                "video",
                "videos",
                "wiki",
            }
            if entity_normalized not in web_terms and not re.search(r"://|\.[a-z0-9]{2,6}\b", entity_normalized, flags=re.IGNORECASE):
                # `entity` from normalize_codeswitched is already the clean
                # filename (verb stripped); only extract the location from the
                # full source text rather than re-splitting the clean entity.
                # Recover lost extension: if "CV" came from "CV.pdf ...", reattach ".pdf".
                entity_with_ext = entity
                _ext_recover = re.search(
                    re.escape(entity) + r"(\.[a-zA-Z0-9]{1,6})\b",
                    raw,
                    re.IGNORECASE,
                )
                if _ext_recover:
                    entity_with_ext = entity + _ext_recover.group(1)
                filename = _strip_file_target_fillers(entity_with_ext) or entity_with_ext
                _, search_path = _split_filename_and_location(source_text)
                if filename:
                    return ParsedCommand("OS_FILE_SEARCH", raw, normalized, args={"filename": filename, "search_path": search_path})

        query = source_text
        if query:
            return ParsedCommand(
                "OS_SYSTEM_COMMAND",
                raw,
                normalized,
                args={"action_key": "browser_search_web", "search_query": query},
            )

    if intent in {"stop", "mute"} and entity_normalized in {"music", "musiqa", "mزيكا", "الموسيقى", "الموسيقي", "المزيكا", "media"}:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "media_stop"})

    if intent in {"increase", "decrease", "set"}:
        _volume_targets = {"volume", "الصوت", "الفوليم", "صوت"}
        _brightness_targets = {"brightness", "السطوع", "سطوع"}
        numbers = (cs or {}).get("numbers") or []
        level = int(numbers[0]) if numbers else None
        if entity_normalized in _volume_targets or entity in _volume_targets:
            if level is not None and _looks_like_explicit_level_request(raw):
                return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized,
                                     args={"action_key": "volume_set", "volume_level": level})
            if intent == "increase":
                return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized,
                                     args={"action_key": "volume_up",
                                           **({} if level is None else {"volume_level": level})})
            if intent == "decrease":
                return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized,
                                     args={"action_key": "volume_down",
                                           **({} if level is None else {"volume_level": level})})
            if intent == "set" and level is not None:
                return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized,
                                     args={"action_key": "volume_set", "volume_level": level})
        if entity_normalized in _brightness_targets or entity in _brightness_targets:
            if level is not None and _looks_like_explicit_level_request(raw):
                return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized,
                                     args={"action_key": "brightness_set", "brightness_level": level})
            if intent == "increase":
                return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized,
                                     args={"action_key": "brightness_up",
                                           **({} if level is None else {"brightness_level": level})})
            if intent == "decrease":
                return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized,
                                     args={"action_key": "brightness_down",
                                           **({} if level is None else {"brightness_level": level})})
            if intent == "set" and level is not None:
                return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized,
                                     args={"action_key": "brightness_set", "brightness_level": level})

    return None


def _contains_any_phrase(text: str, phrases):
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in phrases)


def _recurrence_args(time_phrase: str):
    recurrence, meta = parse_recurrence_spec(time_phrase)
    args = {"recurrence": recurrence} if recurrence else {}
    if meta.get("weekday") is not None:
        args["recurrence_weekday"] = meta["weekday"]
    if meta.get("weekday_name"):
        args["recurrence_weekday_name"] = meta["weekday_name"]
    return args


# ---------------------------------------------------------------------------
# Priority structural patterns
# ---------------------------------------------------------------------------
# These are exact, high-confidence command phrases that should win before the
# broader keyword and regex inventories. Keep this list small and bilingual.
_PRIORITY_STRUCTURAL_TABLE = [
    (
        {
            "turn on notifications",
            "enable notifications",
            "notifications on",
            "allow notifications",
            "open notifications",
            "notifications enable",
            "turn off dnd",
            "disable dnd",
            "dnd off",
            "turn off focus assist",
            "شغل الاشعارات",
            "شغّل الاشعارات",
            "افتح الاشعارات",
            "شغل الإشعارات",
            "شغّل الإشعارات",
            "افتح الإشعارات",
            "turn off do not disturb",
            "disable do not disturb",
            "do not disturb off",
            "dnd off",
            "focus assist off",
            "turn off do not disturb",
            "disable do not disturb",
            "do not disturb off",
            "dnd off",
            "focus assist off",
            "ايقاف وضع عدم الإزعاج",
            "إيقاف وضع عدم الإزعاج",
            "طفي وضع عدم الإزعاج",
            "طفي عدم الإزعاج",
            "اقفل وضع عدم الإزعاج",
            "اقفل عدم الإزعاج",
            "شيل وضع عدم الإزعاج",
            "شيل عدم الإزعاج",
            "قطّع وضع عدم الإزعاج",
            "قطّع عدم الإزعاج",
            "قطع وضع عدم الإزعاج",
            "قطع عدم الإزعاج",
            "ايقاف وضع عدم الازعاج",
            "إيقاف وضع عدم الازعاج",
            "طفي وضع عدم الازعاج",
            "طفي عدم الازعاج",
            "اقفل وضع عدم الازعاج",
            "اقفل عدم الازعاج",
            "شيل وضع عدم الازعاج",
            "شيل عدم الازعاج",
            "قطّع وضع عدم الازعاج",
            "قطّع عدم الازعاج",
            "قطع وضع عدم الازعاج",
            "قطع عدم الازعاج",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "notifications_on"},
        0.95,
    ),
    (
        {
            "turn off notifications",
            "disable notifications",
            "notifications off",
            "mute notifications",
            "silence notifications",
            "turn on do not disturb",
            "enable do not disturb",
            "do not disturb on",
            "dnd on",
            "focus assist on",
            "turn on dnd",
            "enable dnd",
            "dnd on",
            "turn on focus assist",
            "تفعيل وضع عدم الإزعاج",
            "فعّل وضع عدم الإزعاج",
            "فعّل عدم الإزعاج",
            "شغّل وضع عدم الإزعاج",
            "وضع عدم الإزعاج",
            "وضع عدم الازعاج",
            "طيب، ممكن تفعّل وضع عدم الإزعاج؟",
            "ممكن تفعّل وضع عدم الإزعاج؟",
            "تفعّل وضع عدم الإزعاج؟",
            "تفعيل وضع عدم الإزعاج؟",
            "ممكن تفعيل وضع عدم الإزعاج",
            "ممكن تفعيل وضع عدم الازعاج",
            "طيب، ممكن تفعل وضع عدم الازعاج؟",
            "ممكن تفعل وضع عدم الازعاج؟",
            "تفعل وضع عدم الازعاج؟",
            "تفعيل وضع عدم الازعاج؟",
            "خلي الوضع صامت",
            "تفعيل dnd",
            "شغل dnd",
            "notifications disable",
            "notifications mute",
            "اطفي الاشعارات",
            "اطفِ الاشعارات",
            "اقفل الاشعارات",
            "وقف الاشعارات",
            "كتم الاشعارات",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "notifications_off"},
        0.95,
    ),
    (
        {
            "list running apps",
            "show running apps",
            "show running applications",
            "what apps are running",
            "what is open right now",
            "show processes",
            "show running processes",
            "التطبيقات الشغالة",
            "البرامج الشغالة",
            "إيه التطبيقات الشغالة",
            "ايه التطبيقات الشغالة",
            "إيه البرامج الشغالة",
            "اعرض التطبيقات الشغالة",
            "وريني التطبيقات الشغالة",
            "وريني البرامج الشغالة",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "list_processes"},
        0.94,
    ),
    (
        {
            "rescan apps",
            "refresh app list",
            "refresh installed apps",
            "scan apps",
            "find installed apps",
            "find installed app list",
            "اسكن البرامج تاني",
            "اسكن البرامج",
            "سكن البرامج",
            "جدّد قائمة البرامج",
            "جدد قائمة البرامج",
            "تحديث قائمة البرامج",
            "تحديث التطبيقات",
            "اعادة فحص التطبيقات",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "rescan_apps"},
        0.94,
    ),
    (
        {
            "volume up",
            "turn up volume",
            "raise volume",
            "increase volume",
            "louder",
            "volume louder",
            "volume increase",
            "ارفع الصوت",
            "علي الصوت",
            "على الصوت",
            "زوّد الصوت",
            "زود الصوت",
            "صوت أعلى",
            "صوت اعلى",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "volume_up"},
        0.95,
    ),
    (
        {
            "volume down",
            "turn down volume",
            "lower volume",
            "decrease volume",
            "softer",
            "volume lower",
            "volume decrease",
            "وطّي الصوت",
            "وطي الصوت",
            "اخفض الصوت",
            "خفض الصوت",
            "قلل الصوت",
            "صوت واطي",
            "الصوت واطي",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "volume_down"},
        0.95,
    ),
    (
        {
            "mute volume",
            "mute sound",
            "mute audio",
            "turn off sound",
            "silence sound",
            "اكتم الصوت",
            "كتم الصوت",
            "اسكت الصوت",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "volume_mute"},
        0.95,
    ),
    (
        {
            "unmute volume",
            "unmute sound",
            "unmute audio",
            "turn sound on",
            "restore sound",
            "شغل الصوت",
            "شغّل الصوت",
            "ارجع الصوت",
            "فعل الصوت",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "volume_unmute"},
        0.95,
    ),
    (
        {
            "turn on wifi",
            "enable wifi",
            "wifi on",
            "turn on wi fi",
            "enable wi fi",
            "wifi enable",
            "شغل الواي فاي",
            "شغّل الواي فاي",
            "وصل الواي فاي",
            "افتح الانترنت",
            "فتح الانترنت",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "wifi_on"},
        0.95,
    ),
    (
        {
            "turn off wifi",
            "disable wifi",
            "wifi off",
            "turn off wi fi",
            "disable wi fi",
            "wifi disable",
            "turn off internet",
            "disable internet",
            "شيل الواي فاي",
            "اقفل الواي فاي",
            "قطع الانترنت",
            "وقف الواي فاي",
            "افصل الواي فاي",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "wifi_off"},
        0.95,
    ),
    (
        {
            "turn on bluetooth",
            "enable bluetooth",
            "bluetooth on",
            "bluetooth enable",
            "وصل البلوتوث",
            "شغّل البلوتوث",
            "شغل البلوتوث",
            "فتح البلوتوث",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "bluetooth_on"},
        0.95,
    ),
    (
        {
            "turn off bluetooth",
            "disable bluetooth",
            "bluetooth off",
            "bluetooth disable",
            "اقفل البلوتوث",
            "اطفي البلوتوث",
            "اطفِ البلوتوث",
            "قطع البلوتوث",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "bluetooth_off"},
        0.95,
    ),
    # Window management — Egyptian Arabic uses شاشة/شباك/نافذة interchangeably
    (
        {
            "maximize window", "maximize this window", "window maximize",
            "كبر الشباك", "كبر الشبابك", "كبر الشاشة", "كبر النافذة",
            "كبّر الشباك", "كبّر الشاشة", "كبّر النافذة",
            "كبرلي الشاشة", "كبرلي الشباك",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "window_maximize"},
        0.95,
    ),
    (
        {
            "minimize window", "minimize this window", "window minimize",
            "صغر الشباك", "صغر الشبابك", "صغر الشاشة", "صغر النافذة",
            "صغّر الشباك", "صغّر الشاشة", "صغّر النافذة",
            "صغرلي الشاشة", "صغرلي الشباك",
            "اطوي الشاشة", "اطوي الشباك",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "window_minimize"},
        0.95,
    ),
    (
        {
            "open new tab", "new tab", "browser new tab",
            "افتح تاب جديد", "افتح تاب جديدة", "تاب جديد", "تاب جديدة",
            "افتح tab جديد", "افتح tab جديدة",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "browser_new_tab"},
        0.95,
    ),
    (
        {
            "close tab", "close browser tab", "browser close tab",
            "اقفل التاب", "سكر التاب", "اقفل tab", "سكر tab",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "browser_close_tab"},
        0.95,
    ),
    (
        {
            "read clipboard", "show clipboard", "clipboard read",
            "what's in my clipboard", "whats in my clipboard",
        },
        "OS_CLIPBOARD",
        "read",
        {},
        0.95,
    ),
    (
        {
            "clear clipboard", "empty clipboard", "clipboard clear",
            "امسح الكليببورد", "امسح clipboard",
        },
        "OS_CLIPBOARD",
        "clear",
        {},
        0.95,
    ),
    # Screen recording
    (
        {
            "start recording", "start screen recording", "record screen",
            "record my screen", "begin recording",
            "ابدأ التسجيل", "سجّل الشاشة", "سجل الشاشة",
            "ابدأ تسجيل الشاشة", "شغّل التسجيل", "شغل التسجيل",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "screen_record_start"},
        0.95,
    ),
    (
        {
            "stop recording", "stop screen recording", "end recording",
            "finish recording",
            "وقّف التسجيل", "وقف التسجيل", "اوقف التسجيل",
            "خلّص التسجيل", "خلص التسجيل",
        },
        "OS_SYSTEM_COMMAND",
        "",
        {"action_key": "screen_record_stop"},
        0.95,
    ),
]


def _try_priority_structural_table(normalized, raw):
    for entry in _PRIORITY_STRUCTURAL_TABLE:
        phrases, intent, action = entry[0], entry[1], entry[2]
        if normalized in phrases:
            args = entry[3] if len(entry) > 3 else {}
            pattern_confidence = entry[4] if len(entry) > 4 else None
            final_args = dict(args or {})
            if pattern_confidence is not None:
                final_args["pattern_confidence"] = float(pattern_confidence)
            return ParsedCommand(intent, raw, normalized, action=action, args=final_args)
    return None


_PRIORITY_REGEX_TABLE = [
    (
        re.compile(
            r"^(?:please\s+|could\s+you\s+|can\s+you\s+)?(?:set|adjust|change)\s+(?:the\s+)?(?:volume|sound)\s+(?:to|at|=)\s+([0-9٠-٩]{1,3})%?[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda m: {"action_key": "volume_set", "volume_level": _int_from_numeric_text(m.group(1))},
        0.97,
    ),
    (
        re.compile(
            r"^(?:please\s+|could\s+you\s+|can\s+you\s+)?(?:volume|sound)\s+([0-9٠-٩]{1,3})%?[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda m: {"action_key": "volume_set", "volume_level": _int_from_numeric_text(m.group(1))},
        0.96,
    ),
    (
        re.compile(
            r"^(?:please\s+|could\s+you\s+|can\s+you\s+)?(?:set|adjust|change)\s+(?:the\s+)?(?:brightness|screen\s+brightness)\s+(?:to|at|=)\s+([0-9٠-٩]{1,3})%?[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda m: {"action_key": "brightness_set", "brightness_level": _int_from_numeric_text(m.group(1))},
        0.97,
    ),
    (
        re.compile(
            r"^(?:please\s+|could\s+you\s+|can\s+you\s+)?(?:brightness|screen\s+brightness)\s+([0-9٠-٩]{1,3})%?[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda m: {"action_key": "brightness_set", "brightness_level": _int_from_numeric_text(m.group(1))},
        0.96,
    ),
    (
        re.compile(
            r"^(?:ممكن\s+|لو\s+سمحت\s+|please\s+|could\s+you\s+|can\s+you\s+)?(?:(?:increase|raise|turn\s+up|decrease|lower|turn\s+down|set|adjust|change)\s+)?(?:the\s+)?(?:volume|sound)\s+(?:to|at|=|ل|لـ|الى|إلى|على)\s*([0-9٠-٩]{1,3})%?[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda m: {"action_key": "volume_set", "volume_level": _int_from_numeric_text(m.group(1))},
        0.96,
    ),
    (
        re.compile(
            r"^(?:ممكن\s+|لو\s+سمحت\s+|please\s+|could\s+you\s+|can\s+you\s+)?(?:(?:increase|raise|turn\s+up|decrease|lower|turn\s+down|set|adjust|change)\s+)?(?:the\s+)?(?:brightness|screen\s+brightness)\s+(?:to|at|=|ل|لـ|الى|إلى|على)\s*([0-9٠-٩]{1,3})%?[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda m: {"action_key": "brightness_set", "brightness_level": _int_from_numeric_text(m.group(1))},
        0.96,
    ),
    (
        re.compile(
            r"^(?:ممكن\s+|لو\s+سمحت\s+|please\s+|could\s+you\s+|can\s+you\s+)?(?:ارفع|زود|زيد|اخفض|خفض|قلل|وطي|وطّي|اضبط|ظبط|حط|ضع|اعمل|ترفع|تزود|تزيد|تخفض|تقلل|توطي|توطي)\s+(?:الصوت|الفوليم|volume)\s*(?:ل|لـ|على|الى|إلى|to|at|=)?\s*([0-9٠-٩]{1,3})%?[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda m: {"action_key": "volume_set", "volume_level": _int_from_numeric_text(m.group(1))},
        0.97,
    ),
    (
        re.compile(
            r"^(?:ممكن\s+|لو\s+سمحت\s+|please\s+|could\s+you\s+|can\s+you\s+)?(?:ارفع|زود|زيد|اخفض|خفض|قلل|وطي|وطّي|اضبط|ظبط|حط|ضع|اعمل|ترفع|تزود|تزيد|تخفض|تقلل|توطي|توطي)\s+(?:السطوع|الإضاءة|الاضاءة|النور|brightness|screen\s+brightness)\s*(?:ل|لـ|على|الى|إلى|to|at|=)?\s*([0-9٠-٩]{1,3})%?[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda m: {"action_key": "brightness_set", "brightness_level": _int_from_numeric_text(m.group(1))},
        0.97,
    ),
    (
        re.compile(r"^(?:batch add|اضف دفعة|ضيف دفعة)\s+(.+)$", re.IGNORECASE),
        True,
        "BATCH_COMMAND",
        "add",
        lambda m: {"command_text": m.group(1).strip()},
        0.95,
    ),
    (
        re.compile(r"^(?:index find|search indexed|دور في الفهرس|ابحث في الفهرس)\s+(.+?)(?:\s+in\s+(.+))?$", re.IGNORECASE),
        True,
        "SEARCH_INDEX_COMMAND",
        "search",
        lambda m: {"query": m.group(1).strip(), "root": (m.group(2) or "").strip() or None},
        0.95,
    ),    # Phase 3: Batch file delete patterns (highest priority for batch)
    (
        re.compile(
            r"^(?:delete|remove|rm)\s+(?:files?|items?)\s+(.+?)(?:\s+from\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION_BATCH",
        "delete_multiple",
        lambda m: {"files": m.group(1).strip(), "location": (m.group(2) or "").strip()},
        0.92,
    ),
    (
        re.compile(
            r"^(?:احذف|امسح)\s+(?:ملفات|مستندات)\s+(.+?)(?:\s+(?:من|في)\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION_BATCH",
        "delete_multiple",
        lambda m: {"files": m.group(1).strip(), "location": (m.group(2) or "").strip()},
        0.92,
    ),
    # Phase 4: Advanced file search patterns (highest priority for search intent)
    (
        re.compile(
            r"^(?:find|search for|look for|locate)\s+(?:files?|documents?|docs?|pdfs?|images?)\s+(?:about|on|for|containing|with)\s+(.+?)(?:\s+in\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_SEARCH_ADVANCED",
        "search",
        lambda m: {"query": m.group(1).strip(), "search_path": (m.group(2) or "").strip() or None},
        0.94,
    ),
    (
        re.compile(
            r"^(?:find files about|search files about|look for files about)\s+(.+?)(?:\s+in\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_SEARCH_ADVANCED",
        "search",
        lambda m: {"query": m.group(1).strip(), "search_path": (m.group(2) or "").strip() or None},
        0.94,
    ),
    (
        re.compile(
            r"^(?:اوجد|ابحث|دور|دوّر)\s+(?:ملفات|مستندات|وثائق|pdf|بي\s*دي\s*اف)\s+(?:عن|بخصوص|فيها|تحتوي\s+على)\s+(.+?)(?:\s+(?:في|بداخل|داخل)\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_SEARCH_ADVANCED",
        "search",
        lambda m: {"query": m.group(1).strip(), "search_path": (m.group(2) or "").strip() or None},
        0.94,
    ),
    # Phase 5 -- English open-file with extension: "open report.pdf [in downloads]"
    (
        re.compile(
            r"^(?:open|launch|run|start)\s+(?:(?:the|a)\s+)?(?:file\s+)?(.+?\.\w{1,6})(?:\s+(?:in|from|inside)\s+(.+))?[.!?]*$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "open_file",
        lambda m: {"path": (m.group(1).strip() + " في " + m.group(2).strip()) if m.group(2) else m.group(1).strip()},
        0.97,
    ),
    # Phase 5 -- open/launch explorer with a destination arg
    # Must be in the priority table so it fires before _try_codeswitched_command
    # which maps "explorer" -> OS_APP_OPEN.
    (
        re.compile(
            r"^(?:open|launch|show)\s+(?:file\s+)?explorer\s+(?:to|at|on|in)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "open_in_explorer",
        lambda m: {"path": m.group(1).strip()},
        0.97,
    ),

]


def _try_priority_regex_table(normalized, raw):
    for entry in _PRIORITY_REGEX_TABLE:
        pattern, use_raw, intent, action, args_builder, pattern_confidence = entry
        text = raw if use_raw else normalized
        m = pattern.match(text)
        if m:
            args = args_builder(m) if args_builder else {}
            if pattern_confidence is not None:
                args = dict(args or {})
                args["pattern_confidence"] = float(pattern_confidence)
            return ParsedCommand(intent, raw, normalized, action=action, args=args)
    return None


# ---------------------------------------------------------------------------
# Table-driven keyword matching — Phase 1.6 inventory
# ---------------------------------------------------------------------------
# Each entry: (set_of_keywords, intent, action[, args]).
# Matched against `normalized` (lowercased, whitespace-collapsed).
#
# This table is for EXACT phrases that must always resolve deterministically
# (admin commands, runtime toggles, status queries). Conversational paraphrases
# like "can you open chrome please" go through the semantic router instead.

_KEYWORD_TABLE = [
    # Observability
    ({"observability", "observability report", "show observability", "dashboard"}, "OBSERVABILITY_REPORT", ""),
    # Persona
    ({"persona status", "persona show"}, "PERSONA_COMMAND", "status"),
    ({"persona list", "list personas"}, "PERSONA_COMMAND", "list"),
    ({"assistant mode", "assistant mode on"}, "PERSONA_COMMAND", "set", {"profile": "assistant"}),
    # Voice
    ({"voice status", "speech status", "حالة الصوت", "حالة النطق", "الصوت عامل ايه", "النطق عامل ايه", "عامل ايه في الصوت"}, "VOICE_COMMAND", "status"),
    ({"voice diagnostic", "voice diagnostics", "speech diagnostic", "tts diagnostic"}, "VOICE_COMMAND", "diagnostic"),
    (
        {
            "latency status",
            "show latency",
            "pipeline latency status",
            "phase latency status",
            "runtime latency status",
            "حالة الكمون",
            "حالة التأخير",
            "حالة الاستجابة",
            "الاستجابة عاملة ايه",
            "التاخير عامل ايه",
            "التأخير عامل ايه",
            "اظهر الوقت",
            "وريني الوقت",
            "وريني التأخير",
        },
        "VOICE_COMMAND",
        "latency_status",
    ),
    (
        {
            "latency mode fast",
            "low latency mode",
            "speed mode fast",
            "performance mode fast",
            "fast response mode",
            "reduce latency mode",
            "turbo mode",
            "خلي الاستجابة سريعة",
            "خلّي الاستجابة سريعة",
            "السرعة سريع",
        },
        "VOICE_COMMAND",
        "audio_ux_profile_set",
        {"profile": "responsive"},
    ),
    (
        {
            "latency mode balanced",
            "latency mode normal",
            "speed mode normal",
            "performance mode balanced",
            "خلي الاستجابة متوازنة",
            "خلّي الاستجابة متوازنة",
        },
        "VOICE_COMMAND",
        "audio_ux_profile_set",
        {"profile": "balanced"},
    ),
    (
        {
            "latency mode robust",
            "latency mode stable",
            "latency mode reliable",
            "performance mode stable",
            "خلي الاستجابة ثابتة",
            "خلّي الاستجابة ثابتة",
            "خلي الاستجابة قوية",
        },
        "VOICE_COMMAND",
        "audio_ux_profile_set",
        {"profile": "robust"},
    ),
    ({"audio ux status", "audio profile status", "voice audio status", "الصوت عامل ايه دلوقتي"}, "VOICE_COMMAND", "audio_ux_status"),
    ({"audio ux profiles", "audio ux profile list", "list audio ux profiles", "ملفات الصوت ايه"}, "VOICE_COMMAND", "audio_ux_profiles"),
    ({"audio ux profile balanced", "audio profile balanced", "set audio profile balanced", "خلي الصوت متوازن", "خلّي الصوت متوازن"}, "VOICE_COMMAND", "audio_ux_profile_set", {"profile": "balanced"}),
    ({"audio ux profile responsive", "audio profile responsive", "set audio profile responsive", "خلي الصوت سريع", "خلّي الصوت سريع"}, "VOICE_COMMAND", "audio_ux_profile_set", {"profile": "responsive"}),
    ({"audio ux profile robust", "audio profile robust", "set audio profile robust", "خلي الصوت ثابت", "خلّي الصوت ثابت"}, "VOICE_COMMAND", "audio_ux_profile_set", {"profile": "robust"}),
    ({"voice quality status", "speech quality status", "tts quality status", "جودة الصوت عاملة ايه"}, "VOICE_COMMAND", "voice_quality_status"),
    ({"voice quality natural", "speech quality natural", "tts quality natural", "natural voice mode", "خلي الصوت طبيعي", "خلّي الصوت طبيعي"}, "VOICE_COMMAND", "voice_quality_set", {"mode": "natural"}),
    ({"voice quality standard", "speech quality standard", "tts quality standard", "robot voice mode", "robotic voice mode", "خلي الصوت عادي", "خلّي الصوت عادي", "خلي الصوت روبوتي"}, "VOICE_COMMAND", "voice_quality_set", {"mode": "standard"}),
    ({"stt backend hybrid", "speech backend hybrid", "voice stt backend hybrid", "use hybrid stt", "use elevenlabs stt", "set stt backend hybrid", "set stt backend elevenlabs", "محرك الاستماع هجين", "محرك الاستماع اليفن لابس"}, "VOICE_COMMAND", "stt_backend_hybrid"),
    ({"stt backend local", "speech backend local", "voice stt backend local", "stt backend whisper", "set stt backend local", "use local stt", "محرك الاستماع محلي", "محرك الاستماع ويسبر"}, "VOICE_COMMAND", "stt_backend_local"),
    ({"wake triggers", "wake triggers list", "list wake triggers", "wake status", "wake mode status", "كلمات التنبيه", "كلمات الصحوة"}, "VOICE_COMMAND", "wake_status"),
    ({"stop speaking", "interrupt speech", "be quiet", "stop talking"}, "VOICE_COMMAND", "interrupt"),
    ({"speech on", "enable speech", "شغل الصوت"}, "VOICE_COMMAND", "speech_on"),
    ({"speech off", "disable speech", "اطفي الصوت", "اقفل الصوت", "اسكت"}, "VOICE_COMMAND", "speech_off"),
    # Knowledge base
    ({"kb status", "knowledge status", "knowledge base status"}, "KNOWLEDGE_BASE_COMMAND", "status"),
    ({"kb autosync status", "kb auto sync status", "knowledge autosync status"}, "KNOWLEDGE_BASE_COMMAND", "autosync_status"),
    ({"kb autosync on", "kb auto sync on", "knowledge autosync on"}, "KNOWLEDGE_BASE_COMMAND", "autosync_on"),
    ({"kb autosync off", "kb auto sync off", "knowledge autosync off"}, "KNOWLEDGE_BASE_COMMAND", "autosync_off"),
    ({"kb quality", "knowledge quality", "kb quality report"}, "KNOWLEDGE_BASE_COMMAND", "quality"),
    ({"kb clear", "knowledge clear"}, "KNOWLEDGE_BASE_COMMAND", "clear"),
    ({"kb retrieval on", "knowledge retrieval on"}, "KNOWLEDGE_BASE_COMMAND", "retrieval_on"),
    ({"kb retrieval off", "knowledge retrieval off"}, "KNOWLEDGE_BASE_COMMAND", "retrieval_off"),
    # Memory
    ({"memory status", "session memory status"}, "MEMORY_COMMAND", "status"),
    ({"memory clear", "session memory clear"}, "MEMORY_COMMAND", "clear"),
    ({"memory on", "enable memory"}, "MEMORY_COMMAND", "on"),
    ({"memory off", "disable memory"}, "MEMORY_COMMAND", "off"),
    ({"memory show", "show memory"}, "MEMORY_COMMAND", "show"),
    ({"language arabic", "set language arabic", "language ar", "set language ar", "خلي اللغة عربي", "خلّي اللغة عربي", "خلي اللغة مصري", "خلّي اللغة مصري"}, "MEMORY_COMMAND", "set_language", {"language": "ar"}),
    ({"language english", "set language english", "language en", "set language en", "خلي اللغة انجليزي", "خلّي اللغة انجليزي"}, "MEMORY_COMMAND", "set_language", {"language": "en"}),
    # Demo
    ({"demo mode on", "demo on"}, "DEMO_MODE", "on"),
    ({"demo mode off", "demo off"}, "DEMO_MODE", "off"),
    ({"demo mode status", "demo status"}, "DEMO_MODE", "status"),
    # Metrics
    ({"show metrics", "metrics", "metrics report"}, "METRICS_REPORT", ""),
    # Audit
    ({"verify audit", "verify audit log", "audit verify"}, "AUDIT_VERIFY", ""),
    ({"audit reseal", "reseal audit", "repair audit chain"}, "AUDIT_RESEAL", ""),
    # Policy
    ({"policy status"}, "POLICY_COMMAND", "status"),
    ({"policy dry run on", "policy dry-run on", "policy dryrun on"}, "POLICY_COMMAND", "set_dry_run", {"enabled": True}),
    ({"policy dry run off", "policy dry-run off", "policy dryrun off"}, "POLICY_COMMAND", "set_dry_run", {"enabled": False}),
    # Batch
    ({"batch plan", "batch start", "batch begin"}, "BATCH_COMMAND", "plan"),
    ({"batch preview", "batch show"}, "BATCH_COMMAND", "preview"),
    ({"batch status"}, "BATCH_COMMAND", "status"),
    ({"batch commit", "batch run"}, "BATCH_COMMAND", "commit"),
    ({"batch abort", "batch cancel", "batch clear"}, "BATCH_COMMAND", "abort"),
    # Search index
    ({"index status", "search index status"}, "SEARCH_INDEX_COMMAND", "status"),
    ({"index start", "start index"}, "SEARCH_INDEX_COMMAND", "start"),
    # Job queue
    ({"job worker start"}, "JOB_QUEUE_COMMAND", "worker_start"),
    ({"job worker stop"}, "JOB_QUEUE_COMMAND", "worker_stop"),
    ({"job worker status"}, "JOB_QUEUE_COMMAND", "worker_status"),
    # Timer
    (
        {
            "cancel timer",
            "stop timer",
            "cancel alarm",
            "stop alarm",
            "الغي التايمر",
            "وقف التايمر",
            "الغيلي التايمر",
            "بطل التايمر",
            "اوقفلي التايمر",
            "امسح التايمر",
        },
        "OS_TIMER",
        "cancel",
    ),
    (
        {
            "list timers",
            "show timers",
            "active timers",
            "list alarms",
            "show alarms",
            "active alarms",
            "التايمرات",
            "وريني التايمرات",
            "التايمر على كام",
            "كام دقيقة فاضلة",
            "فضل قد ايه",
        },
        "OS_TIMER",
        "list",
    ),
    # Clipboard
    (
        {
            "read clipboard",
            "show clipboard",
            "what's in my clipboard",
            "whats in my clipboard",
            "what's in clipboard",
            "whats in clipboard",
            "read my clipboard",
            "show my clipboard",
            "what's in clipboard?",
            "whats in clipboard?",
            "paste clipboard",
            "اقرا الكليب بورد",
            "اقرأ الكليب بورد",
            "اقرأ clipboard",
            "اقرا clipboard",
            "اقرا ال clipboard",
            "اقرأ ال clipboard",
            "وريني الكليب بورد",
            "ايه في الكليب بورد",
            "إيه اللي في Clipboard",
            "ايه اللي في Clipboard",
            "إيه اللي في clipboard",
            "ايه اللي في clipboard",
            "إيه اللي في clipboard؟",
            "ايه اللي في clipboard؟",
            "إيه المنسوخ",
            "المنسوخ ايه",
            "في الكليبورد ايه",
            "انسخ من الكليب بورد",
            "اللي في الكليب بورد",
            "افتح الكليب بورد",
            "الكليب بورد",
        },
        "OS_CLIPBOARD",
        "read",
    ),
    (
        {
            "clear clipboard",
            "empty clipboard",
            "امسح الكليب بورد",
            "فضي الكليب بورد",
            "مسح الكليب بورد",
            "خليه فاضي",
            "نضف الكليب بورد",
            "افرغ الكليب بورد",
        },
        "OS_CLIPBOARD",
        "clear",
    ),
    # Battery / System info
    (
        {
            "battery status",
            "battery level",
            "how much battery",
            "battery percentage",
            "البطارية كام",
            "نسبة البطارية",
            "حالة البطارية",
            "الشحن كام",
            "الشحن قد ايه",
            "البطارية وصلت كام",
            "البطارية تجيب كام",
            "البطارية عاملة ايه",
            "نسبة الشحن",
        },
        "OS_SYSINFO",
        "battery",
    ),
    (
        {
            "system info",
            "system status",
            "cpu usage",
            "ram usage",
            "disk usage",
            "معلومات النظام",
            "حالة النظام",
            "استهلاك المع��لج",
            "الرام قد ايه",
            "استهلاك الرام",
            "المعالج بياخد قد ايه",
            "الكمبيوتر شغال بكام",
            "المساحة قد ايه",
            "معلومات الجهاز",
            "حالة الجهاز",
            "استهلاك المعالج",
            "الرام كام",
        },
        "OS_SYSINFO",
        "system",
    ),
    # Email
    (
        {
            "compose email",
            "new email",
            "draft email",
            "open email",
            "open mail",
            "open inbox",
            "open outlook",
            "open outlook and draft",
            "open outlook and compose",
            "launch outlook",
            "start outlook",
            "افتح البريد",
            "افتح الايميل",
            "افتح الإيميل",
            "اكتب بريد",
            "افتح ايميل جديد",
            "ايميل جديد",
            "اكتب ايميل",
            "اعمل ايميل",
            "ابعت ايميل",
            "مسودة ايميل",
            "افتح أوتلوك",
            "افتح Outlook",
            " إبعت email",
            "افتح اوتلوك",
            "شغّل أوتلوك",
            "شغل اوتلوك",
            "اعملي ايميل",
            "اعمل إيميل",
            "اكتب إيميل",
        },
        "OS_EMAIL",
        "draft",
    ),
    # Settings (top-level — specific pages are handled by regex fallback below)
    (
        {
            "open settings",
            "open windows settings",
            "settings",
            "windows settings",
            "افتح الاعدادات",
            "افتح الإعدادات",
            "الاعدادات",
            "الإعدادات",
            "ودّيني للاعدادات",
            "روح على الاعدادات",
            "خدني على الاعدادات",
            "عايز الاعدادات",
            "اعداداتك",
            "افتح الضبط",
            "روح للاعدادات",
            "روح للضبط",
        },
        "OS_SETTINGS",
        "open",
    ),
    # Rollback
    (
        {
            "undo",
            "rollback",
            "undo last action",
            "ارجع اخر حاجة",
            "الغي اخر حاجة",
            "رجعني لاخر خطوة",
            "تراجع",
            "ارجع",
            "رجع",
        },
        "OS_ROLLBACK",
        "",
    ),
    # File nav
    (
        {
            "current directory",
            "pwd",
            "احنا فين",
            "انا فين دلوقتي",
            "احنا فين دلوقتي",
            "ده فين",
        },
        "OS_FILE_NAVIGATION",
        "pwd",
    ),
    (
        {
            "list drives",
            "drive list",
            "وريني الدرايفات",
            "هاتلي الدرايفات",
            "الدرايفات ايه",
        },
        "OS_FILE_NAVIGATION",
        "list_drives",
    ),
    # Reminders — list
    (
        {
            "show reminders",
            "list reminders",
            "my reminders",
            "active reminders",
            "وريني التذكيرات",
            "وريني التذكيرات بتاعتي",
            "التذكيرات",
            "التذكيرات النشطة",
            "ايه التذكيرات",
            "عندي تذكيرات ايه",
        },
        "OS_REMINDER",
        "list",
    ),
    # Reminders — cancel
    (
        {
            "cancel reminder",
            "cancel the reminder",
            "delete reminder",
            "remove reminder",
            "الغي التذكير",
            "امسح التذكير",
            "شيل التذكير",
            "الغيلي التذكير",
            "مسح التذكير",
        },
        "OS_REMINDER",
        "cancel",
    ),
    # Phase 5 -- open File Explorer (bare "open explorer" phrases; must come
    # before _try_codeswitched_command which maps "file explorer" → OS_APP_OPEN)
    (
        {
            "open explorer",
            "open file explorer",
            "launch explorer",
            "launch file explorer",
            "show file explorer",
            "start file explorer",
            "افتح المستكشف",
            "افتح مستكشف الملفات",
            "فتح المستكشف",
        },
        "OS_FILE_NAVIGATION",
        "open_in_explorer",
        {"path": ""},
    ),
]


def _try_keyword_table(normalized, raw):
    for entry in _KEYWORD_TABLE:
        keywords, intent, action = entry[0], entry[1], entry[2]
        if normalized in keywords:
            args = entry[3] if len(entry) > 3 else {}
            pattern_confidence = entry[4] if len(entry) > 4 else None
            final_args = dict(args or {})
            if pattern_confidence is not None:
                final_args["pattern_confidence"] = float(pattern_confidence)
            return ParsedCommand(intent, raw, normalized, action=action, args=final_args)
    return None


# ---------------------------------------------------------------------------
# Table-driven regex matching — Phase 1.6 inventory
# ---------------------------------------------------------------------------
# Each entry: (compiled_regex, use_raw, intent, action, args_builder).
#
# Every pattern below is *structural*: it exists to extract a typed argument
# (a hex token, a numeric value, an alarm time, a file path, an email address,
# a settings page name, etc.) that the semantic router and keyword fuzzy tier
# cannot recover from paraphrase similarity alone. Pure paraphrase routes —
# "open chrome", "pause music", "go back", "minimize this window" — were
# removed in favor of the semantic router's ``_ROUTE_DEFINITIONS`` to keep
# this list small and maintainable.
#
# If you find yourself adding a regex that has NO capture groups and only
# matches a fixed phrase, prefer adding it to ``_KEYWORD_TABLE`` instead.

# ---------------------------------------------------------------------------
# Q2 2026 OPTIMIZATION: Parser Reduction Phase 1.6 → 1.7
# ---------------------------------------------------------------------------
# Current state: ~110 regex patterns in _REGEX_TABLE (high maintenance, paraphrase misses)
# Target state: ~40 structural patterns only (unambiguous commands)
#
# Keep ONLY these in regex:
#   - Paths/navigation: open /path, pwd, ls, cd, move /src /dst
#   - Token sequences: set volume 50, set brightness 80, open [appname]
#   - Explicit actions: create_directory, delete_item, send_email, create_calendar_event
#   - System toggles: wifi_on, wifi_off, bluetooth_on, bluetooth_off, mute
#   - Timer/reminder/alarm: set_timer 5m, set_reminder, set_alarm
#   - Confirmations: yes, no, cancel, confirm
#
# MOVE TO SEMANTIC ROUTER (nlp/semantic_router.py):
#   - Music playback (all paraphrases: "play some music", "put on music", etc.)
#   - Media control (play, pause, skip, volume hints)
#   - App commands (all friendly phrasings: "open chrome please", "launch spotify", etc.)
#   - File operations (search, create, delete with natural language)
#   - Every user-friendly paraphrase that embedding similarity can handle
#
# Result: Parser easier to maintain, semantic router handles rich user language,
# combined coverage remains >95% with better recall on natural phrasings.
#
_BROWSER_CLOSE_TAB_RE = re.compile(
    r"^(?:close|shut|kill|exit|اقفل|سكر|قفل)\s+"
    r"(?:the\s+)?(?:الـ\s+|ال)?(.+?)\s+"
    r"(?:tab|window|browser\s+tab|تاب|شبابك)$",
    re.IGNORECASE | re.UNICODE,
)
_BROWSER_CLOSE_TAB_AR_RE = re.compile(
    r"^(?:اقفل|سكر|قفل)\s+tab\s+(?:الـ\s+|ال)?(.+?)(?:\s+في\s+(?:الـ\s+)?(?:browser|متصفح|chrome|firefox|edge))?$",
    re.IGNORECASE | re.UNICODE,
)


def _parse_browser_close_named_tab(m):
    raw_query = m.group(1).strip()
    # Strip residual noise words
    import re as _re
    tab_query = _re.sub(r"\b(tab|تاب|browser|الـ|في|in|the|a|an)\b", " ", raw_query, flags=_re.IGNORECASE | _re.UNICODE).strip()
    return {"action_key": "browser_close_named_tab", "tab_query": (tab_query or raw_query).lower()}


_REGEX_TABLE = [
    # Browser tab close by name
    (
        _BROWSER_CLOSE_TAB_RE,
        True,
        "OS_SYSTEM_COMMAND",
        "",
        _parse_browser_close_named_tab,
        0.93,
    ),
    (
        _BROWSER_CLOSE_TAB_AR_RE,
        True,
        "OS_SYSTEM_COMMAND",
        "",
        _parse_browser_close_named_tab,
        0.93,
    ),
    # Persona
    (
        re.compile(r"^persona set\s+([a-z0-9_-]+)$"),
        False,
        "PERSONA_COMMAND",
        "set",
        lambda m: {"profile": m.group(1)},
    ),
    # Voice
    (
        re.compile(
            r"^(?:set\s+)?(?:(?:voice|speech|stt)\s+)?(?:stt|speech)\s+backend(?:\s+to)?\s+(hybrid|elevenlabs?|arabic(?:\s+hybrid)?)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "stt_backend_hybrid",
        lambda _m: {},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:(?:voice|speech|stt)\s+)?(?:stt|speech)\s+backend(?:\s+to)?\s+(local|whisper|faster(?:[_\s-]?whisper)?)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "stt_backend_local",
        lambda _m: {},
    ),
    (
        re.compile(
            r"^(?:ظبط|ظبّط|غير|غيّر|عدل|عدّل|خلي|خلّي)\s+(?:محرك|باكند)?\s*الاستماع(?:\s+على)?\s+(?:هجين|اليفن\s*لابس|elevenlabs?)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "stt_backend_hybrid",
        lambda _m: {},
    ),
    (
        re.compile(
            r"^(?:ظبط|ظبّط|غير|غيّر|عدل|عدّل|خلي|خلّي)\s+(?:محرك|باكند)?\s*الاستماع(?:\s+على)?\s+(?:محلي|لوكال|ويسبر)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "stt_backend_local",
        lambda _m: {},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:voice|speech|tts)\s+quality(?:\s+to)?\s+(natural|standard|balanced|default|human|robot|robotic)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "voice_quality_set",
        lambda m: {"mode": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:ظبط|ظبّط|غير|غيّر|عدل|عدّل|خلي|خلّي)\s+(?:جودة|وضع)?\s*(?:الصوت|النطق)(?:\s+ل)?\s+(طبيعي|عادي|روبوت|روبوتي)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "voice_quality_set",
        lambda m: {"mode": m.group(1)},
    ),
    # CONSOLIDATED: audio_ux_profile (unified English/Arabic + mode/latency synonyms)
    (
        re.compile(
            r"^(?:set\s+)?(?:(?:audio|voice|latency|performance|speed)\s+)?(?:ux\s+)?(?:profile|mode)(?:\s+to)?\s+(balanced|responsive|robust|fast|low\s*latency|low_latency|stable|reliable|noisy|normal)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_profile_set",
        lambda m: {"profile": _normalize_audio_profile(m.group(1))},
    ),
    (
        re.compile(
            r"^(?:ظبط|ظبّط|غير|غيّر|عدل|عدّل|خلي|خلّي)\s+(?:ملف|وضع|نمط)?\s*(?:تجربة\s+)?(?:الصوت|النطق|الاستجابة|السرعة|الكمون)(?:\s+ل)?\s+(متوازن|سريع(?:\s*الاستجابة)?|قوي|ثابت|طبيعي)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_profile_set",
        lambda m: {"profile": _normalize_audio_profile(m.group(1))},
    ),
    # CONSOLIDATED: latency_status (all synonyms unified)
    (
        re.compile(
            r"^(?:latency|pipeline\s+latency|phase\s+latency|runtime\s+latency|performance|response\s+time)\s+(?:status|state|report)?$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "latency_status",
        lambda _m: {},
    ),
    (
        re.compile(
            r"^(?:الكمون|الاستجابة|التاخير|التأخير|الكمون)\s+(?:عامل|عاملة|اخبار|اخباره)\s+(?:ايه|ه|هو)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "latency_status",
        lambda _m: {},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:audio\s+ux\s+)?(?:mic|microphone|vad)\s+(?:energy\s+)?threshold(?:\s+to)?\s+([0-9]+(?:\.[0-9]+)?)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_mic_threshold_set",
        lambda m: {"value": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:audio\s+ux\s+)?(?:wake(?:\s*[-_]?word)?\s+threshold)(?:\s+to)?\s+([0-9]+(?:\.[0-9]+)?)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_wake_threshold_set",
        lambda m: {"value": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:audio\s+ux\s+)?(?:wake(?:\s*[-_]?word)?\s+gain)(?:\s+to)?\s+([0-9]+(?:\.[0-9]+)?)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_wake_gain_set",
        lambda m: {"value": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:wake(?:\s*[-_]?word)?\s+triggers?)\s+(?:add|insert)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "wake_triggers_add",
        lambda m: {"trigger": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:wake(?:\s*[-_]?word)?\s+triggers?)\s+(?:remove|delete)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "wake_triggers_remove",
        lambda m: {"trigger": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:wake(?:\s*[-_]?word)?\s+mode)(?:\s+to)?\s+(english|arabic|both|en|ar)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "wake_mode_set",
        lambda m: {"mode": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:(?:voice|speech|tts|audio\s+ux)\s+)?pause\s+scale(?:\s+to)?\s+([0-9]+(?:\.[0-9]+)?)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_pause_scale_set",
        lambda m: {"value": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:(?:voice|speech|tts|audio\s+ux)\s+)?rate\s+offset(?:\s+to)?\s+([+-]?\d+)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_rate_offset_set",
        lambda m: {"value": m.group(1)},
    ),
    # Memory
    (
        re.compile(
            r"^(?:(?:set|change|switch)\s+)?(?:the\s+)?language(?:\s+(?:to|into))?\s+(arabic|english|ar|en)(?:\s*[.!?؟،]+)?$",
            re.IGNORECASE,
        ),
        True,
        "MEMORY_COMMAND",
        "set_language",
        lambda m: {"language": _normalize_language_value(m.group(1).strip())},
    ),
    (
        re.compile(
            r"^(?:switch|change)\s+to\s+(arabic|english|ar|en)(?:\s*[.!?؟،]+)?$",
            re.IGNORECASE,
        ),
        True,
        "MEMORY_COMMAND",
        "set_language",
        lambda m: {"language": _normalize_language_value(m.group(1).strip())},
    ),
    (
        re.compile(
            r"^(?:ظبط|ظبّط|غير|غيّر|بدل|بدّل|حول|حوّل|خلي|خلّي)?\s*(?:اللغة)(?:\s+ل)?\s*(عربي|مصري|انجليزي|انجلش|ar|en)(?:\s*[.!?؟،]+)?$",
            re.IGNORECASE,
        ),
        True,
        "MEMORY_COMMAND",
        "set_language",
        lambda m: {"language": _normalize_language_value(m.group(1).strip())},
    ),
    # Knowledge base
    (
        re.compile(r"^(?:kb sync|knowledge sync)\s+(.+)$", re.IGNORECASE),
        True,
        "KNOWLEDGE_BASE_COMMAND",
        "sync_dir",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(r"^(?:kb add|knowledge add)\s+(.+)$", re.IGNORECASE),
        True,
        "KNOWLEDGE_BASE_COMMAND",
        "add_file",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(r"^(?:kb index|knowledge index)\s+(.+)$", re.IGNORECASE),
        True,
        "KNOWLEDGE_BASE_COMMAND",
        "index_dir",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(r"^(?:kb search|knowledge search)\s+(.+)$", re.IGNORECASE),
        True,
        "KNOWLEDGE_BASE_COMMAND",
        "search",
        lambda m: {"query": m.group(1).strip()},
    ),
    (
        re.compile(r"^(?:kb|knowledge)\s+(?:auto\s*sync|autosync)\s+(on|off|status)$", re.IGNORECASE),
        False,
        "KNOWLEDGE_BASE_COMMAND",
        "autosync_toggle",
        lambda m: {"mode": m.group(1).strip().lower()},
    ),
    # System commands: explicit catch-alls for common paraphrases missed by fuzzy aliasing
    (
        re.compile(
            r"^(?:lock(?:\s+the)?\s+(?:screen|computer|pc|workstation)|\u0642\u0641\u0644\s+(?:\u0627\u0644\u0634\u0627\u0634\u0629|\u0627\u0644\u062c\u0647\u0627\u0632)|\u0627\u0642\u0641\u0644\s+\u0627\u0644\u0634\u0627\u0634\u0629)$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "lock"},
    ),
    (
        re.compile(
            r"^(?:put(?:\s+the)?\s+(?:computer|pc)\s+to\s+sleep|sleep\s+(?:pc|computer)|sleep\s+this\s+computer|\u0646\u0627\u0645\s+\u0627\u0644\u0643\u0645\u0628\u064a\u0648\u062a\u0631|\u0646\u0627\u0645\s+\u0627\u0644\u062c\u0647\u0627\u0632)$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "sleep"},
    ),
    (
        re.compile(
            r"^(?:set|adjust|change)\s+(?:the\s+)?brightness\s+(?:to|at)\s+(\d{1,3})%?[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda m: {"action_key": "brightness_set", "brightness_level": int(m.group(1))},
    ),
    (
        re.compile(
            r"^(?:brightness\s+(\d{1,3})%?|set\s+brightness\s+(\d{1,3})%?)[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda m: {
            "action_key": "brightness_set",
            "brightness_level": int(m.group(1) or m.group(2)),
        },
    ),
    (
        re.compile(
            r"^(?:increase|raise|turn\s+up|brighten)\s+(?:the\s+)?(?:screen\s+)?brightness[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "brightness_up"},
    ),
    (
        re.compile(
            r"^(?:decrease|lower|turn\s+down|dim)\s+(?:the\s+)?(?:screen\s+)?brightness[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "brightness_down"},
    ),
    (
        re.compile(
            r"^(?:turn|switch)\s+(?:the\s+)?bluetooth\s+off[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "bluetooth_off"},
    ),
    (
        re.compile(
            r"^(?:turn|switch)\s+(?:the\s+)?bluetooth\s+on[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "bluetooth_on"},
    ),
    (
        re.compile(
            r"^(?:enable|disable)\s+(?:the\s+)?bluetooth[.!?]*$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda m: {"action_key": "bluetooth_on" if m.group(0).lower().startswith("enable") else "bluetooth_off"},
    ),
    # Arabic Bluetooth control — شغل / اقفل البلوتوث variants
    (
        re.compile(
            r"^(?:شغل\s+البلوتوث|شغّل\s+البلوتوث|وصل\s+البلوتوث|فعّل\s+البلوتوث|تفعيل\s+البلوتوث)$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "bluetooth_on"},
    ),
    (
        re.compile(
            r"^(?:اقفل\s+البلوتوث|اطفي\s+البلوتوث|وقف\s+البلوتوث|افصل\s+البلوتوث|تعطيل\s+البلوتوث)$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "bluetooth_off"},
    ),
    # Arabic colloquial volume down mapping and colloquial screenshot phrasing
    (
        re.compile(r"^(?:وطي\s+الصوت|اخفض\s+الصوت|خف\u0651\u0636\s+الصوت|خفف\s+الصوت)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "volume_down"},
    ),
    (
        re.compile(r"^(?:خد\s+سكرين\s+شوت|خد\s+سكرينشوت|خذ\s+سكرينشوت|خذ\s+سكرين\s+شوت|خد\s+screenshot|خذ\s+screenshot|خد\s+سكرين|خذ\s+سكرين)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "screenshot"},
    ),
    # Arabic volume up variants
    (
        re.compile(r"^(?:زود\s+الصوت|ارفع\s+الفوليم|ارفع\s+الصوت|صوت\s+أعلى|صوت\s+اعلى|مش\s+سامع\s+خالص|الصوت\s+واطي\s+جداً?|الصوت\s+واطي)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "volume_up"},
    ),
    # Arabic brightness up variants
    (
        re.compile(
            r"^(?:ارفع\s+السطوع|زود\s+الإضاءة|زود\s+الاضاءة|زود\s+الضوء|زود\s+العضاءة|زود\s+العضرا|زود\s+العضره|السطوع\s+واطي|الشاشة\s+مظلمة|مظلم\s+قوي)$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "brightness_up"},
    ),
    # Arabic brightness down variants
    (
        re.compile(
            r"^(?:خفض\s+السطوع|قلل\s+الإضاءة|قلل\s+الاضاءة|قلل\s+الضوء|قلل\s+العضاءة|قلل\s+العضرا|قلل\s+العضره|السطوع\s+عالي\s+قوي|الشاشة\s+ناصعة)$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "brightness_down"},
    ),
    # Arabic additional screenshot variants
    (
        re.compile(r"^(?:صور\s+الشاشة|خد\s+لقطة\s+شاشة|سكرين\s+شوت|لقطة\s+الشاشة|التقط\s+الشاشة|صورة\s+للشاشة)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "screenshot"},
    ),
    # Arabic lock screen variants
    (
        re.compile(r"^(?:اقفل\s+الجهاز|اقفل\s+الشاشة|قفل\s+الشاشة|قفل\s+الجهاز|لوك\s+الشاشة|قفّل\s+الجهاز)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "lock"},
    ),
    # Arabic sleep variants
    (
        re.compile(r"^(?:نوم\s+الجهاز|حط\s+الجهاز\s+في\s+السليب|سليب\s+الجهاز|وضع\s+السليب|اوضع\s+الكمبيوتر\s+في\s+السليب)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "sleep"},
    ),
    # Arabic shutdown variants
    (
        re.compile(r"^(?:اوقف\s+الكمبيوتر|اغلق\s+الكمبيوتر|شتداون|اطفي\s+الجهاز|اطفئ\s+الجهاز|سكّر\s+الكمبيوتر)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "shutdown"},
    ),
    # Arabic restart variants
    (
        re.compile(r"^(?:اعيد\s+تشغيل\s+الكمبيوتر|ريستارت|عمل\s+ريستارت|اعد\s+تشغيل\s+الجهاز|اعادة\s+تشغيل)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "restart"},
    ),
    # Arabic media stop/pause/next/prev variants
    # Note: ى (U+0649) is normalized to ي (U+064A) by normalize_arabic_preserve_digits
    (
        re.compile(r"^(?:وقف\s+الموسيقي|وقف\s+المزيكا|ايقاف\s+الموسيقي|وقّف\s+الموسيقي|اوقف\s+الموسيقي)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "media_stop"},
    ),
    (
        re.compile(r"^(?:شغل\s+الموسيقي|شغّل\s+الموسيقي|كمل\s+الموسيقي|استانف\s+الموسيقي|play\s+الموسيقي)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "media_play_pause"},
    ),
    # Arabic Wi-Fi toggle variants — includes اقفل (close/lock)
    (
        re.compile(r"^(?:شيل\s+الواي\s+فاي|اقفل\s+الواي\s+فاي|قطع\s+الانترنت|وقف\s+الواي\s+فاي|افصل\s+الواي\s+فاي)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "wifi_off"},
        0.95,
    ),
    (
        re.compile(r"^(?:وصل\s+الواي\s+فاي|شغّل\s+الواي\s+فاي|فتح\s+الانترنت|شغل\s+الانترنت)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "wifi_on"},
        0.95,
    ),
    # Arabic Bluetooth toggle variants
    (
        re.compile(r"^(?:اقفل\s+البلوتوث|اطفي\s+البلوتوث|قطع\s+البلوتوث)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "bluetooth_off"},
        0.95,
    ),
    (
        re.compile(r"^(?:شغّل\s+البلوتوث|وصل\s+البلوتوث|فتح\s+البلوتوث)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "bluetooth_on"},
        0.95,
    ),
    # Arabic notifications toggle variants
    (
        re.compile(r"^(?:اطفي\s+الاشعارات|اقفل\s+الاشعارات|وقف\s+الاشعارات)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "notifications_off"},
        0.95,
    ),
    (
        re.compile(r"^(?:شغّل\s+الاشعارات|وصل\s+الاشعارات|افتح\s+الاشعارات)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "notifications_on"},
        0.95,
    ),
    (
        re.compile(r"^(?:فع.?ل\s+عدم\s+الإزعاج|فع.?ل\s+عدم\s+الازعاج)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "notifications_off"},
        0.95,
    ),
    # Arabic مش سامع / مش شايف — shorthand for volume/brightness help
    (
        re.compile(r"^(?:مش\s+سامع|صوت\s+خفيف\s+قوي|الصوت\s+خفيف)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "volume_up"},
    ),
    # Media control — next/skip forward
    (
        re.compile(
            r"^(?:next\s+(?:track|song)|skip\s+(?:forward|track)|go\s+to\s+next|play\s+next|التالي|الأغنية\s+الجاية|الأغنية\s+التالية|شغل\s+التالي|التراك\s+التالي)$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "media_next_track"},
    ),
    # Media control — previous/skip backward
    (
        re.compile(
            r"^(?:previous\s+(?:track|song)|skip\s+back(?:ward)?|go\s+(?:back|to\s+previous)|play\s+previous|السابق|الأغنية\s+اللي\s+فات|الأغنية\s+السابقة|رجع\s+تراك|التراك\s+السابق)$",
            re.IGNORECASE,
        ),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "media_previous_track"},
    ),
    # Audit
    (
        re.compile(r"^(?:show audit log|عرض سجل التدقيق|وريني سجل التدقيق|اعرض سجل التدقيق)(?:\s+(\d+))?$", re.IGNORECASE),
        False,
        "AUDIT_LOG_REPORT",
        "",
        lambda m: {"limit": int(m.group(1)) if m.group(1) else 10},
    ),
    # Policy
    (
        re.compile(r"^(?:policy profile|ملف السياسة)\s+([a-z0-9_-]+)$", re.IGNORECASE),
        False,
        "POLICY_COMMAND",
        "set_profile",
        lambda m: {"profile": m.group(1)},
    ),
    (
        re.compile(r"^(?:policy (?:read only|readonly)|السياسة قراءة فقط)\s+(on|off)$", re.IGNORECASE),
        False,
        "POLICY_COMMAND",
        "set_read_only",
        lambda m: {"enabled": m.group(1) == "on"},
    ),
    (
        re.compile(r"^(?:policy (?:dry run|dry-run|dryrun)|وضع المحاكاة)\s+(on|off)$", re.IGNORECASE),
        False,
        "POLICY_COMMAND",
        "set_dry_run",
        lambda m: {"enabled": m.group(1) == "on"},
    ),
    (
        re.compile(r"^(?:policy permission|صلاحية السياسة)\s+([a-z_]+)\s+(on|off)$", re.IGNORECASE),
        False,
        "POLICY_COMMAND",
        "set_permission",
        lambda m: {"permission": m.group(1), "enabled": m.group(2) == "on"},
    ),
    # Batch
    (
        re.compile(r"^(?:batch add|اضف دفعة|ضيف دفعة)\s+(.+)$", re.IGNORECASE),
        True,
        "BATCH_COMMAND",
        "add",
        lambda m: {"command_text": m.group(1).strip()},
    ),
    # Search index
    (
        re.compile(r"^(?:index refresh|حدث الفهرس|اعمل تحديث للفهرس)(?:\s+in\s+(.+))?$", re.IGNORECASE),
        True,
        "SEARCH_INDEX_COMMAND",
        "refresh",
        lambda m: {"root": (m.group(1) or "").strip() or None},
    ),
    (
        re.compile(r"^(?:indexed find|index find|search indexed|دور في الفهرس|ابحث في الفهرس)\s+(.+?)(?:\s+in\s+(.+))?$", re.IGNORECASE),
        True,
        "SEARCH_INDEX_COMMAND",
        "search",
        lambda m: {"query": m.group(1).strip(), "root": (m.group(2) or "").strip() or None},
    ),
    # Job queue
    (
        re.compile(r"^(?:queue job|job add|جدولة مهمة)\s+in\s+(\d+)\s*(?:s|sec|secs|seconds)?\s+(.+)$", re.IGNORECASE),
        True,
        "JOB_QUEUE_COMMAND",
        "enqueue",
        lambda m: {"delay_seconds": int(m.group(1)), "command_text": m.group(2).strip()},
    ),
    (
        re.compile(r"^(?:queue job|job add|جدولة مهمة)\s+(.+)$", re.IGNORECASE),
        True,
        "JOB_QUEUE_COMMAND",
        "enqueue",
        lambda m: {"delay_seconds": 0, "command_text": m.group(1).strip()},
    ),
    (
        re.compile(r"^(?:job status|حالة المهمة)\s+(\d+)$", re.IGNORECASE),
        False,
        "JOB_QUEUE_COMMAND",
        "status",
        lambda m: {"job_id": int(m.group(1))},
    ),
    (
        re.compile(r"^(?:job cancel|الغ المهمة|الغ المهمة رقم)\s+(\d+)$", re.IGNORECASE),
        False,
        "JOB_QUEUE_COMMAND",
        "cancel",
        lambda m: {"job_id": int(m.group(1))},
    ),
    (
        re.compile(r"^(?:job retry|أعد المهمة)\s+(\d+)(?:\s+in\s+(\d+)\s*(?:s|sec|secs|seconds)?)?$", re.IGNORECASE),
        False,
        "JOB_QUEUE_COMMAND",
        "retry",
        lambda m: {"job_id": int(m.group(1)), "delay_seconds": int(m.group(2) or 0)},
    ),
    (
        re.compile(r"^(?:job list|قائمة المهام)(?:\s+([a-z]+|\d+))?(?:\s+(\d+))?$", re.IGNORECASE),
        False,
        "JOB_QUEUE_COMMAND",
        "list",
        lambda m: _parse_job_list_args(m),
    ),
    # Confirmation
    (
        re.compile(
            rf"^(?:confirm|\u062a\u0627\u0643\u064a\u062f|\u062a\u0623\u0643\u064a\u062f)\s+([0-9a-f]{{{CONFIRMATION_TOKEN_MIN_HEX_LEN},{_CONFIRMATION_TOKEN_MAX_HEX_LEN}}})(?:\s+(?:with\s+)?(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_CONFIRMATION",
        "",
        lambda m: {"token": m.group(1).lower(), "second_factor": (m.group(2) or "").strip() or None},
    ),
    # File search
    (
        re.compile(
            r"^(?:find file|search file|دور على ملف|دور على|دوّر على ملف|دوّر على|وريني ملف|هاتلي ملف|دورلي على ملف|دورلي على|دورلي ملف|لقيلي ملف|فين ملف)\s+(.+?)(?:\s+(?:in|the|\u0641\u064a|\u0627\u0644))?\s*(.+)?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_SEARCH",
        "",
        lambda m: {"filename": _strip_file_target_fillers(m.group(1)), "search_path": _normalize_search_path_hint((m.group(2) or "").strip()) or None},
    ),
    # Phase 5 -- Arabic reveal (must precede the وريني list_directory catch-all)
    (
        re.compile(
            r"^(?:ورّيني\s+مكان|وريني\s+مكان|وريني\s+الملف\s+|وريني\s+المجلد\s+|فين\s+الملف\s+|فين\s+المجلد\s+|فين\s+|وريني\s+(?!الملفات|المجلدات))(.+)$",
            re.IGNORECASE | re.UNICODE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "reveal_in_explorer",
        lambda m: {"path": m.group(1).strip()},
    ),
    # Phase 5 -- Arabic open-file: "افتح <file>" or "فتح ملف <file>"
    # Must come BEFORE the افتح المستكشف open-in-explorer rule.
    (
        re.compile(
            r"^(?:افتح|فتح|شغل)\s+(?!المستكشف|مستكشف)(?:ملف|الملف|المجلد|مجلد)?\s*(.+?)(?:\s+(?:في|من|inside|in)\s+(.+))?[.!?]*$",
            re.IGNORECASE | re.UNICODE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "open_file",
        lambda m: {"path": (m.group(1).strip() + " في " + m.group(2).strip()) if m.group(2) else m.group(1).strip()},
    ),
    # Phase 5 -- Arabic open-in-explorer (must precede list_directory)
    (
        re.compile(
            r"^(?:افتح|فتح)\s+(?:المستكشف|مستكشف\s+الملفات|explorer)(?:\s+(?:على|علي|في)\s+(.+))?[.!?]*$",
            re.IGNORECASE | re.UNICODE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "open_in_explorer",
        lambda m: {"path": (m.group(1) or "").strip()},
    ),
    # File nav - regex-based
    (
        re.compile(
            r"^(?:list files|list directory|show files|show directory|وريني الملفات|هاتلي الملفات|وريني المجلد|هاتلي المجلد|شوفلي الملفات|ايه في المجلد|ايه في|ايه اللي في|هاتلي|اعرضلي)(?:\s+(?:in|the|\u0641\u064a|\u0627\u0644))?\s*(.+)?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "list_directory",
        lambda m: {"path": _normalize_search_path_hint((m.group(1) or "").strip()) or (m.group(1) or "").strip() or None},
    ),
    (
        re.compile(r"^(?:dir|ls)(?:\s+(.+))?$", re.IGNORECASE),
        True,
        "OS_FILE_NAVIGATION",
        "list_directory",
        lambda m: {"path": (m.group(1) or "").strip() or None},
    ),
    (
        re.compile(
            r"^(?:file info|metadata|معلومات الملف|بيانات الملف)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "file_info",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:create folder|make folder|new folder|mkdir|اعمل مجلد|اعمللي مجلد|انشئ مجلد|عمل مجلد|اعمل folder|اعمللي folder|انشئ folder|اعمل مجلد جديد)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "create_directory",
        lambda m: {"path": _strip_create_name(m.group(1))},
    ),
    (
        re.compile(
            r"^(?:(?:delete|remove)\s+(?:permanently|forever)\s+(.+)|(?:permanent\s+delete|force\s+delete)\s+(.+)|(?:amسح|شيل)\s+(.+?)\s+(?:نهائيا|نهائي|permanently|forever)|(?:delete|remove)\s+(.+?)\s+(?:permanently|forever|نهائيا|نهائي))$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "delete_item_permanent",
        lambda m: dict(zip(("path", "location"), _split_target_and_location((m.group(1) or m.group(2) or m.group(3) or m.group(4) or "").strip()))),
    ),
    (
        re.compile(r"^(?:delete|remove|امسح|شيل)\s+(.+)$", re.IGNORECASE),
        True,
        "OS_FILE_NAVIGATION",
        "delete_item",
        lambda m: dict(zip(("path", "location"), _split_target_and_location(m.group(1)))),
    ),
    (
        re.compile(
            r"^(?:move|put|انقل|انقل|نقل|حرك|ودي|ودّي|غير مكان|حول|شيل)\s+(?:the\s+)?(?:file|folder|ملف|مجلد)?\s*(.+?)\s+(?:from|من)\s+(.+?)\s+(?:to|into|to the|الى|الي|على|علي|ل|للـ?)\s*(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "move_item",
        lambda m: {"source": os.path.join(_normalize_search_path_hint(m.group(2)) or m.group(2), _strip_file_target_fillers(m.group(1))), "destination": _normalize_search_path_hint(m.group(3)) or _strip_file_target_fillers(m.group(3))},
    ),
    (
        re.compile(
            r"^(?:move|put|انقل|انقل|نقل|حرك|ودي|ودّي|غير مكان|حول|شيل)\s+(?:the\s+)?(?:file|folder|ملف|مجلد)?\s*(.+?)\s+(?:to|into|الى|الي|على|علي|ل|للـ?)\s*(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "move_item",
        lambda m: {"source": _strip_file_target_fillers(m.group(1)), "destination": _normalize_search_path_hint(m.group(2)) or _strip_file_target_fillers(m.group(2))},
    ),
    (
        re.compile(
            r"^(?:rename|سمي|سميلي|سمّي|سمّيلي|غير اسم|بدّل اسم|اغير اسم)\s+(.+?)\s+(?:to|as|اسمه|باسم|ل(?:ـ)?|الى|لـ)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "rename_item",
        lambda m: dict(list({"source": _split_target_and_location(m.group(1))[0] or _strip_file_target_fillers(m.group(1)), "new_name": _strip_file_target_fillers(m.group(2)), "location": _split_target_and_location(m.group(1))[1] or None}.items())),
    ),
     # Follow-up rename
    (
        re.compile(
            r"^(?:غير اسمه|سميه|سمّيه|بدّله|غير اسمها|سميها|rename it to|name it)(?:\s+(?:ل(?:ـ)?|لـ|الى|to|as))?\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "rename_item_followup",
        lambda m: {"new_name": re.sub(r"^ل(?:ـ)?", "", _strip_file_target_fillers(m.group(1))).strip()},
    ),
    # Follow-up move
    (
        re.compile(
            r"^(?:انقله|نقله|حركه|انقلها|نقلها|حركها|move it to|put it in)\s+(?:(?:to|into|الى|الي|على|علي|ل|للـ?)\s+)?(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "move_item_followup",
        lambda m: {"destination": _normalize_search_path_hint(m.group(1)) or _strip_file_target_fillers(m.group(1))},
    ),
   # Copy — "copy file X to Y", "انسخ الملف X الى Y", "copy X to Y"
    (
        re.compile(
            r"^(?:copy|انسخ|انسخلي)\s+(?:the\s+)?(?:file|folder)?\s*(.+?)\s+(?:to|الى|الي|ل)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "copy_item",
        lambda m: {"source": _strip_file_target_fillers(m.group(1)), "destination": _strip_file_target_fillers(m.group(2))},
    ),
    (
        re.compile(
            r"^(?:copy|انسخ|انسخلي|كوبي)\s+(.+?)\s+(?:to|الى|الي|ل)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "copy_item",
        lambda m: {"source": _strip_file_target_fillers(m.group(1)), "destination": _strip_file_target_fillers(m.group(2))},
    ),
    # --- Phase 5: Explorer-driven file ops ---
    # reveal in explorer  -- "reveal X in explorer", "show X in file manager", etc.
    (
        re.compile(
            r"^(?:reveal|show|locate|find|highlight)\s+(.+?)\s+(?:in\s+(?:file\s+)?(?:explorer|manager|finder)|in\s+explorer)[.!?]*$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "reveal_in_explorer",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:reveal|show|locate)\s+(.+?)\s+(?:file|folder|ملف|مجلد)?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "reveal_in_explorer",
        lambda m: {"path": m.group(1).strip()},
        0.5,  # lower priority — only fires when nothing else matches
    ),
    # Arabic reveal: "ورّيني مكان X", "فين X", "وريني الملف X"
    (
        re.compile(
            r"^(?:ورّيني\s+مكان|ورّينى\s+مكان|وريني\s+مكان|وريني\s+الملف|وريني\s+المجلد|فين\s+الملف|فين\s+المجلد|فين)\s+(.+)$",
            re.IGNORECASE | re.UNICODE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "reveal_in_explorer",
        lambda m: {"path": m.group(1).strip()},
    ),
    # open in explorer -- "open downloads in explorer", "open folder X in file manager"
    (
        re.compile(
            r"^(?:open|show|launch|browse)\s+(?:the\s+)?(?:folder\s+|directory\s+)?(.+?)\s+(?:in\s+(?:file\s+)?(?:explorer|manager|finder)|in\s+explorer)[.!?]*$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "open_in_explorer",
        lambda m: {"path": m.group(1).strip()},
    ),
    # "open explorer" / "open file explorer" with optional path
    (
        re.compile(
            r"^(?:open|launch|show)\s+(?:file\s+)?explorer(?:\s+(?:to|at|on|in)\s+(.+))?[.!?]*$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "open_in_explorer",
        lambda m: {"path": (m.group(1) or "").strip()},
    ),
    # Arabic open-in-explorer: "افتح المستكشف", "افتح الـ explorer", "افتح المستكشف على X"
    (
        re.compile(
            r"^(?:افتح|فتح)\s+(?:المستكشف|مستكشف الملفات|الـ\s*explorer|explorer)(?:\s+(?:على|علي|في|على\s+فولدر|على\s+ملف)\s+(.+))?[.!?]*$",
            re.IGNORECASE | re.UNICODE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "open_in_explorer",
        lambda m: {"path": (m.group(1) or "").strip()},
    ),

    # Timer — "set timer 5 minutes", "timer 10 seconds", "حط تايمر 5 دقايق"
    (
        re.compile(
            r"^(?:set\s+(?:a\s+)?timer|timer|set\s+(?:an?\s+)?alarm)\s+(?:for\s+)?(\S+)\s+(seconds?|secs?|minutes?|mins?|hours?|hrs?|ثانية|ثواني|دقيقة|دقائق|دقايق|ساعة|ساعات)[.!?]*$",
            re.IGNORECASE,
        ),
        True,
        "OS_TIMER",
        "set",
        lambda m: {"seconds": _duration_to_seconds(m.group(1), m.group(2)), "label": "Timer"},
        0.95,
    ),
    (
        re.compile(
            r"^(?:set\s+(?:a\s+)?timer\s+for|timer\s+for)\s+(\S+)\s+(seconds?|secs?|minutes?|mins?|hours?|hrs?)[.!?]*$",
            re.IGNORECASE,
        ),
        True,
        "OS_TIMER",
        "set",
        lambda m: {"seconds": _duration_to_seconds(m.group(1), m.group(2)), "label": "Timer"},
        0.95,
    ),
    # Timer — "set a 5 minute timer", "set it a 5 minute timer", "5 minutes timer"
    (
        re.compile(
            r"^(?:set\s+(?:(?:it\s+)?(?:an?\s+)?)?)?(\S+)\s+(seconds?|secs?|minutes?|mins?|hours?|hrs?|ثانية|ثواني|دقيقة|دقائق|دقايق|ساعة|ساعات)\s+timer[.!?]*$",
            re.IGNORECASE,
        ),
        True,
        "OS_TIMER",
        "set",
        lambda m: {"seconds": _duration_to_seconds(m.group(1), m.group(2)), "label": "Timer"},
        0.95,
    ),
    (
        re.compile(
            r"^(?:حط|حطلي|ظبط|ظبّط|اعمل|اعمللي|اضبط|اضبطلي)\s+(?:ال)?(?:تايمر|منبه|alarm|timer)\s+(?:على|علي\s+)?(\S+)\s+(ثانية|ثواني|دقيقة|دقائق|دقايق|ساعة|ساعات|seconds?|secs?|minutes?|mins?)$",
            re.IGNORECASE,
        ),
        True,
        "OS_TIMER",
        "set",
        lambda m: {"seconds": _duration_to_seconds(m.group(1), m.group(2)), "label": "Timer"},
        0.95,
    ),
    (
        re.compile(
            r"^(?:صحيني|فكرني|نبهني)\s+(?:بعد\s+)?(\S+)\s+(ثانية|ثواني|دقيقة|دقائق|دقايق|ساعة|ساعات)$",
            re.IGNORECASE,
        ),
        True,
        "OS_TIMER",
        "set",
        lambda m: {"seconds": _duration_to_seconds(m.group(1), m.group(2)), "label": "Reminder"},
        0.95,
    ),
    (
        re.compile(
            r"^(?:set\s+(?:an?\s+)?alarm|alarm)\s+(?:for\s+|at\s+)?((?:\d{1,2}(?::\d{2})?\s*(?:am|pm)?)|(?:\d{1,2}\s*(?:am|pm)))$",
            re.IGNORECASE,
        ),
        True,
        "OS_TIMER",
        "set_alarm",
        lambda m: {"alarm_time": m.group(1).strip(), "label": "Alarm"},
        0.95,
    ),
    (
        re.compile(
            r"^(?:صحيني|نبهني|حط(?:لي)?\s+منبه|اعمل(?:لي)?\s+منبه)\s+(?:الساعة\s+|الساعه\s+)?(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_TIMER",
        "set_alarm",
        lambda m: {"alarm_time": m.group(1).strip(), "label": "Alarm"},
        0.95,
    ),
    # Named timer — "set a timer for the pasta for 10 minutes"
    # label group comes before duration group
    (
        re.compile(
            r"^(?:set\s+(?:a\s+)?timer\s+for\s+(?:the\s+)?)([\w\s]+?)\s+(?:for\s+)(\S+\s+(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|ثانية|ثواني|دقيقة|دقائق|دقايق|ساعة|ساعات))[.!?]*$",
            re.IGNORECASE,
        ),
        True,
        "OS_TIMER",
        "set",
        _named_timer_args,
        0.96,
    ),
    # Named cancel — "cancel the pasta timer"
    (
        re.compile(
            r"^(?:cancel|stop|الغي|بطل|اوقف)\s+(?:the\s+)?([\w\s]+?)\s+(?:timer|alarm|تايمر|منبه)[.!?]*$",
            re.IGNORECASE,
        ),
        True,
        "OS_TIMER",
        "cancel",
        lambda m: {"label": _extract_named_timer_label(m.group(1).strip())},
        0.95,
    ),
    # Loose timer — handles STT garbling like "sit at ten seconds timer"
    # Must come AFTER strict patterns so it only fires as a fallback.
    (
        re.compile(
            r".*?\b(\S+)\s+(seconds?|secs?|minutes?|mins?|hours?|hrs?|ثانية|ثواني|دقيقة|دقائق|دقايق|ساعة|ساعات)\s+timer[.!?]*$",
            re.IGNORECASE | re.DOTALL,
        ),
        True,
        "OS_TIMER",
        "set",
        lambda m: {"seconds": _duration_to_seconds(m.group(1), m.group(2)), "label": "Timer"},
    ),
    (
        re.compile(
            r"^(?:set\s+(?:a\s+)?timer|timer|set\s+(?:an?\s+)?alarm|(?:حط|حطلي|ظبط|ظبّط|اعمل|اعمللي|اضبط|اضبطلي)\s+(?:ال)?(?:تايمر|منبه|alarm|timer))(?:\s+(?:for|at|in|after|على|علي|ل(?:ـ)?|الى|إلى|بعد))?\s+(.+?)(?:[.!?]+)?$",
            re.IGNORECASE,
        ),
        True,
        "OS_TIMER",
        "set",
        lambda m: _timer_args_from_text(m.group(1), label="Timer"),
        0.9,
    ),
    # Reminder — English recurring, time before message:
    #   "remind me every day at 9 to drink water"
    #   "remind me every monday at 9 to call mom"
    (
        re.compile(
            r"^(?:remind(?:\s+me)?|set\s+(?:a\s+)?reminder)\s+"
            r"((?:every\s+.+?|daily|weekly|monthly))\s+to\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_REMINDER",
        "create",
        lambda m: {**{"time_str": m.group(1).strip(), "message": m.group(2).strip()}, **_recurrence_args(m.group(1).strip())},
        0.95,
    ),
    # Reminder — English recurring, message before time:
    (
        re.compile(
            r"^(?:remind(?:\s+me)?|set\s+(?:a\s+)?reminder)\s+to\s+(.+?)\s+((?:every\s+.+?|daily|weekly|monthly))$",
            re.IGNORECASE,
        ),
        True,
        "OS_REMINDER",
        "create",
        lambda m: {**{"time_str": m.group(2).strip(), "message": m.group(1).strip()}, **_recurrence_args(m.group(2).strip())},
        0.95,
    ),
    # Reminder — English, time before message:
    #   "remind me at 3pm to call Ahmed", "remind me in 2 hours to take meds",
    #   "remind me tomorrow at 9 to wake up"
    (
        re.compile(
            r"^(?:remind(?:\s+me)?|set\s+(?:a\s+)?reminder)\s+"
            r"((?:tomorrow\s+)?(?:at|in|by)\s+.+?)\s+to\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_REMINDER",
        "create",
        lambda m: {"time_str": m.group(1).strip(), "message": m.group(2).strip()},
        0.95,
    ),
    # Reminder — Arabic recurring, time before message:
    #   "فكرني كل يوم الساعة ٩ علشان اشرب مية"
    #   "فكرني كل اثنين الساعة ٩ علشان اكلم ماما"
    (
        re.compile(
            r"^(?:فكرني|فكّرني|ذكرني|ذكّرني|نبهني|نبّهني|اعمل(?:لي)?\s+تذكير)\s+"
            r"((?:كل\s+.+?|يومي|اسبوعي|أسبوعي|شهري))\s+"
            r"(?:علشان|عشان|to)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_REMINDER",
        "create",
        lambda m: {**{"time_str": m.group(1).strip(), "message": m.group(2).strip()}, **_recurrence_args(m.group(1).strip())},
        0.95,
    ),
    # Reminder — Arabic recurring, message before time:
    (
        re.compile(
            r"^(?:فكرني|فكّرني|ذكرني|ذكّرني|نبهني|نبّهني|اعمل(?:لي)?\s+تذكير)\s+"
            r"(?:علشان|عشان|to)\s+(.+?)\s+((?:كل\s+.+?|يومي|اسبوعي|أسبوعي|شهري))$",
            re.IGNORECASE,
        ),
        True,
        "OS_REMINDER",
        "create",
        lambda m: {**{"time_str": m.group(2).strip(), "message": m.group(1).strip()}, **_recurrence_args(m.group(2).strip())},
        0.95,
    ),
    # Reminder — English, message before time:
    #   "remind me to call Ahmed at 3pm", "remind me to take meds in 2 hours"
    (
        re.compile(
            r"^(?:remind(?:\s+me)?|set\s+(?:a\s+)?reminder)\s+to\s+(.+?)\s+"
            r"((?:tomorrow\s+)?(?:at|in)\s+.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_REMINDER",
        "create",
        lambda m: {"time_str": m.group(2).strip(), "message": m.group(1).strip()},
        0.95,
    ),
    # Reminder — English, message only (no time) — dispatcher returns "when?" prompt
    (
        re.compile(
            r"^(?:remind(?:\s+me)?|set\s+(?:a\s+)?reminder)\s+to\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_REMINDER",
        "create",
        lambda m: {"time_str": "", "message": m.group(1).strip()},
        0.80,
    ),
    (
        re.compile(
            r"^(?:فكرني|فكّرني|ذكرني|ذكّرني|نبهني|نبّهني|اعمل(?:لي)?\s+تذكير)\s+"
            r"((?:بكرة|بكره|بكرا|النهاردة|النهارده|اليوم|بعد\s+بكرة|بعد\s+بكره|بعد\s+بكرا|بعد)\s+.+?)\s+"
            r"(?:عشان|علشان|to)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_REMINDER",
        "create",
        lambda m: {"time_str": m.group(1).strip(), "message": m.group(2).strip()},
        0.95,
    ),
    # Reminder — Arabic wall-clock time:
    #   "فكرني الساعة ٣ أكلم أحمد", "فكرني بكرة الساعة ٩ أصحى"
    (
        re.compile(
            r"^(?:فكرني|فكّرني|ذكرني|ذكّرني|نبهني|نبّهني)\s+"
            r"((?:بكرة\s+|بكره\s+)?(?:الساعة|الساعه|ساعه?)\s+[\d٠-٩]+(?:[:.،,][\d٠-٩]+)?\s*"
            r"(?:صباحاً|صباحا|صبح|ص|مساءً|مساءا|مساء|م)?)\s+"
            r"(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_REMINDER",
        "create",
        lambda m: {"time_str": m.group(1).strip(), "message": m.group(2).strip()},
        0.95,
    ),
    # Reminder — Arabic relative time:
    #   "فكرني بعد ساعتين أكلم أحمد", "فكرني بعد ٣٠ دقيقة أاخد الدواء"
    (
        re.compile(
            r"^(?:فكرني|فكّرني|ذكرني|ذكّرني|نبهني|نبّهني)\s+"
            r"(بعد\s+(?:[\d٠-٩]+(?:\.\d+)?|نص|ربع|ساعتين)\s*"
            r"(?:ثانية|ثواني|دقيقة|دقائق|دقايق|ساعة|ساعات|ساعه)?)\s+"
            r"(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_REMINDER",
        "create",
        lambda m: {"time_str": m.group(1).strip(), "message": m.group(2).strip()},
        0.95,
    ),
    # Clipboard — "copy this: {text}", "انسخ: {text}"
    (
        re.compile(
            r"^(?:copy(?:\s+this)?|انسخ|انسخلي)\s*[:：]\s*(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_CLIPBOARD",
        "write",
        lambda m: {"text": m.group(1).strip()},
    ),
    # Battery / sysinfo — Phase 1.6: regex variants removed. The keyword table
    # already covers the exact-match phrases ("battery status", "البطارية كام")
    # and the semantic router handles paraphrases like "what's my battery".
    # Email — most-specific patterns first so they win over shorter ones.
    # "draft email to X about Y saying Z" — spoken body via "saying"
    (
        re.compile(
            r"^(?:draft|compose|write|new)\s+(?:an?\s+)?email\s+(?:to\s+)?(\S+@\S+|\S+)\s+(?:about|re|subject)\s+(.+?)\s+saying\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_EMAIL",
        "draft",
        lambda m: {"to": m.group(1).strip(), "subject": m.group(2).strip(), "body": m.group(3).strip()},
    ),
    # "draft email to X with subject Y and body Z"
    (
        re.compile(
            r"^(?:draft|compose|send|write|new)\s+(?:an?\s+)?email\s+(?:to\s+)?(\S+@\S+)(?:\s+(?:subject|about|re)\s+(.+?))?(?:\s+(?:with\s+)?(?:body|message|text)\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_EMAIL",
        "draft",
        lambda m: {"to": m.group(1).strip(), "subject": (m.group(2) or "").strip(), "body": (m.group(3) or "").strip()},
    ),
    # "draft email to X about Y"
    (
        re.compile(
            r"^(?:draft|compose|send|write|new)\s+(?:an?\s+)?email\s+(?:to\s+)?(\S+@\S+)(?:\s+(?:about|subject|re)\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_EMAIL",
        "draft",
        lambda m: {"to": m.group(1).strip(), "subject": (m.group(2) or "").strip()},
    ),
    # Arabic with body — "ابعت ايميل ل X عن Y والرسالة Z"
    (
        re.compile(
            r"^(?:ابعت|اكتب|افتح)\s+(?:ايميل|إيميل)\s+(?:ل|الى|الي)?\s*(\S+@\S+)(?:\s+(?:عن|بخصوص|موضوع)\s+(.+?))?(?:\s+(?:و|الرسالة|الجسم)\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_EMAIL",
        "draft",
        lambda m: {"to": (m.group(1) or "").strip(), "subject": (m.group(2) or "").strip(), "body": (m.group(3) or "").strip()},
    ),
    # Arabic without body — "ابعت ايميل ل X عن Y"
    (
        re.compile(
            r"^(?:ابعت|اكتب|افتح)\s+(?:ايميل|إيميل)\s+(?:ل|الى|الي)?\s*(\S+@\S+)?(?:\s+(?:عن|بخصوص)\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_EMAIL",
        "draft",
        lambda m: {"to": (m.group(1) or "").strip(), "subject": (m.group(2) or "").strip()},
    ),
    # "email X about Y" — short natural form without @-address
    (
        re.compile(
            r"^(?:email|draft|compose)\s+(\S+)\s+(?:about|re|regarding)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_EMAIL",
        "draft",
        lambda m: {"to": m.group(1).strip(), "subject": m.group(2).strip()},
    ),
    # Calendar — "create event meeting at 3pm", "اعمل حدث اجتماع"
    (
        re.compile(
            r"^(?:create|add|schedule|new)\s+(?:a\s+)?(?:calendar\s+)?event\s+(.+?)(?:\s+(?:at|on|for)\s+(.+?))?(?:\s+(?:for|duration)\s+(\d+)\s*(?:minutes?|mins?|hours?|hrs?))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_CALENDAR",
        "create",
        lambda m: {
            "subject": m.group(1).strip(),
            "start_time": (m.group(2) or "").strip(),
            "duration_minutes": int(m.group(3)) if m.group(3) else 60,
        },
    ),
    (
        re.compile(
            r"^(?:اعمل|اعمللي|ضيف|حط)\s+(?:حدث|ايفنت|موعد|اجتماع)\s+(.+?)(?:\s+(?:الساعة|في)\s+(.+?))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_CALENDAR",
        "create",
        lambda m: {
            "subject": m.group(1).strip(),
            "start_time": (m.group(2) or "").strip(),
            "duration_minutes": 60,
        },
    ),
    # Settings — specific page: "open display settings", "open wifi settings",
    # "settings for sound", "افتح اعدادات الشاشة", "روح على اعدادات الواي فاي"
    (
        re.compile(
            r"^(?:open|launch|show|go\s+to|take\s+me\s+to)\s+(?:the\s+)?(.+?)\s+settings[.!?]*$",
            re.IGNORECASE,
        ),
        True,
        "OS_SETTINGS",
        "open",
        lambda m: {"page": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:open|launch|show)\s+(?:windows\s+)?settings\s+(?:for|to)\s+(.+?)[.!?]*$",
            re.IGNORECASE,
        ),
        True,
        "OS_SETTINGS",
        "open",
        lambda m: {"page": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:افتح|افتحلي|روح\s+على|ودّيني\s+(?:على|ل)|خدني\s+(?:على|ل)|روحلي\s+على)\s+(?:اعدادات|إعدادات|صفحة\s+اعدادات|صفحة\s+إعدادات)\s+(.+?)[.!؟]*$",
            re.IGNORECASE,
        ),
        True,
        "OS_SETTINGS",
        "open",
        lambda m: {"page": m.group(1).strip()},
    ),
    # Open app explicit
    (
        re.compile(r"^(?:open app|افتحلي برنامج|شغللي برنامج)\s+(.+)$", re.IGNORECASE),
        True,
        "OS_APP_OPEN",
        "",
        lambda m: {"app_name": re.sub(r"[.!?,;]+$", "", m.group(1).strip())},
    ),
    # Close app explicit
    (
        re.compile(
            r"^(?:close app|اقفللي برنامج|سكرلي برنامج|سكّرلي برنامج)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_APP_CLOSE",
        "",
        lambda m: {"app_name": re.sub(r"[.!?,;]+$", "", m.group(1).strip())},
    ),
]


def _parse_job_list_args(m):
    first = m.group(1)
    second = m.group(2)
    status = None
    limit = 10
    if first:
        if first.isdigit():
            limit = int(first)
        else:
            status = first
    if second:
        limit = int(second)
    return {"status": status, "limit": limit}


def _try_regex_table(normalized, raw):
    for entry in _REGEX_TABLE:
        # support optional pattern_confidence as a 6th element
        if len(entry) == 5:
            pattern, use_raw, intent, action, args_builder = entry
            pattern_confidence = None
        else:
            pattern, use_raw, intent, action, args_builder, pattern_confidence = entry
        text = raw if use_raw else normalized
        m = pattern.match(text)
        if m:
            args = args_builder(m) if args_builder else {}
            if pattern_confidence is not None:
                args = dict(args or {})
                args["pattern_confidence"] = float(pattern_confidence)
            return ParsedCommand(intent, raw, normalized, action=action, args=args)
    return None


# ---------------------------------------------------------------------------
# Heuristic matchers (order-sensitive, cannot be table-driven)
# ---------------------------------------------------------------------------


def _try_drive_open(normalized_match, raw, normalized):
    drive_letter = _extract_drive_letter(normalized_match)
    if drive_letter and _is_drive_open_request(normalized_match):
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="list_directory",
            args={"path": f"{drive_letter}:\\"},
        )
    return None


def _try_open_command(raw, normalized):
    system_action = normalize_system_action(raw) or normalize_system_action(_normalize_for_match(raw))
    if system_action:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": system_action})

    open_match = re.match(
        (
            r"^(?:(?:\u0645\u0645\u0643\u0646|\u0644\u0648\u0020\u0633\u0645\u062d\u062a|\u0628\u0639\u062f\u0020\u0627\u0630\u0646\u0643)\s+)?"
            r"(?:open|launch|start|\u0627\u0641\u062a\u062d|\u062a\u0641\u062a\u062d|\u0627\u0641\u062a\u062d\u0644\u064a|\u0641\u062a\u062d\u0644?\u064a?|\u0634\u063a\u0644|\u0634\u063a\u0644\u0644\u064a|\u062f\u0641\u062a\u062d|\u062f\u0641\u062a\u062d\u0644\u064a|\u062f\u0641\u062a\u062d\u0644?\u064a?)\s+(.+)$"
        ),
        raw,
        flags=re.IGNORECASE,
    )
    if not open_match:
        return None

    target_raw = open_match.group(1).strip()
    target_for_match = _strip_open_fillers(_normalize_for_match(target_raw))

    drive_from_target = _extract_drive_letter(target_for_match)
    if drive_from_target and _is_drive_open_request(f"open {target_for_match}"):
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="list_directory",
            args={"path": f"{drive_from_target}:\\"},
        )

    special_folder = _special_folder_path(target_for_match)
    if special_folder:
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="list_directory",
            args={"path": special_folder},
        )

    if _looks_like_filesystem_target(target_for_match):
        target_path = target_raw
        if target_path.lower().startswith("the "):
            target_path = target_path[4:].strip()
        if target_path.startswith("\u0627\u0644"):
            target_path = target_path[2:].strip()
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="list_directory",
            args={"path": target_path},
        )

    app_name = _infer_known_app_name(target_raw)
    if app_name:
        return ParsedCommand("OS_APP_OPEN", raw, normalized, args={"app_name": app_name})

    return ParsedCommand("OS_APP_OPEN", raw, normalized, args={"app_name": target_raw})


def _try_close_command(raw, normalized):
    close_match = re.match(
        (
            r"^(?:close|terminate|kill|quit|exit|\u0627\u0642\u0641\u0644|\u0627\u0642\u0641\u0644\u0644\u064a|\u0633\u0643\u0631|\u0633\u0643\u0631\u0644\u064a|\u0633\u0643\u0651\u0631\u0644\u064a)\s+"
            r"(?:app\s+|application\s+|program\s+|\u062a\u0637\u0628\u064a\u0642\s+)?(.+)$"
        ),
        raw,
        flags=re.IGNORECASE,
    )
    if not close_match:
        return None

    target_raw = close_match.group(1).strip()
    if not target_raw:
        return None

    blocked_system_targets = {
        "computer",
        "pc",
        "system",
        "الجهاز",
        "الكمبيوتر",
        "النظام",
    }
    normalized_target = _normalize_for_match(target_raw)
    if normalized_target in blocked_system_targets:
        return None

    return ParsedCommand("OS_APP_CLOSE", raw, normalized, args={"app_name": target_raw})


def _looks_explicit_file_search(raw, filename, search_path):
    if str(search_path or "").strip():
        return True

    lowered = _normalize_for_match(raw)
    file_markers = (
        " file ",
        " files ",
        " folder ",
        " directory ",
        " document ",
        " documents ",
        " pdf",
        " doc",
        " txt",
        " ملف",
        " ملفات",
        " مجلد",
        " مستند",
    )
    padded = f" {lowered} "
    if any(marker in padded for marker in file_markers):
        return True

    candidate = str(filename or "").strip().lower()
    if any(token in candidate for token in ("\\", "/", ":")):
        return True
    if re.search(r"\.[a-z0-9]{1,6}$", candidate, flags=re.IGNORECASE):
        return True
    return False


def _clean_browser_search_query(value):
    query = str(value or "").strip().strip(" .,!?؟")
    if not query:
        return ""

    query = re.sub(r"^(?:about|for)\s+", "", query, flags=re.IGNORECASE)
    query = re.sub(r"^(?:عن)\s+", "", query, flags=re.IGNORECASE)
    query = re.sub(
        r"\s+(?:online|on\s+google|على\s+النت|بالنت|اونلاين|أونلاين)$",
        "",
        query,
        flags=re.IGNORECASE,
    )
    return query.strip().strip(" .,!?؟")

def _try_natural_file_search(raw, normalized):
    lowered = _normalize_for_match(raw)
    if _contains_any_phrase(
        lowered,
        (
            "online",
            "web",
            "internet",
            "on google",
            "search web",
            "search online",
            "google",
            "الويب",
            "النت",
            "جوجل",
            "اونلاين",
            "أونلاين",
            "بالنت",
        ),
    ):
        return None

    patterns = (
        re.compile(
            r"^(?:(?:i\s+)?(?:want|need)\s+(?:to\s+)?)?(?:find|search|look\s+for|locate)\s+(?:for\s+)?(?:file\s+)?(.+?)(?:\s+(?:in|on|inside)\s+(.+))?$",
            re.IGNORECASE,
        ),
        re.compile(
            (
                r"^(?:(?:\u0639\u0627\u064a\u0632|\u0639\u0627\u0648\u0632)\s+(?:\u0627\u0646|\u0623\u0646)?\s+)?"
                r"(?:\u062f\u0648\u0631|\u062f\u0648\u0651\u0631|\u062f\u0648\u0631\u0644\u064a|\u062f\u0648\u0651\u0631\u0644\u064a)"
                r"(?:\s+(?:\u0639\u0646|\u0639\u0644\u0649|\u0639\u0644\u0627))?"  # \u0639\u0646 | \u0639\u0644\u0649 | \u0639\u0644\u0627
                r"\s+(?:\u0645\u0644\u0641\s+)?"
                r"(.+?)(?:\s+(?:\u0641\u064a|\u062f\u0627\u062e\u0644)\s+(.+))?$"
            ),
            re.IGNORECASE,
        ),
    )

    for pattern in patterns:
        match = pattern.match(raw)
        if not match:
            continue

        filename = _strip_file_target_fillers(match.group(1) or "")
        filename = _collapse_repeated_phrase(filename)
        filename = filename.strip().strip('"').strip("'")
        if not filename:
            return None

        search_path = _normalize_search_path_hint(match.group(2) or "")
        if not _looks_explicit_file_search(raw, filename, search_path):
            return None
        return ParsedCommand(
            "OS_FILE_SEARCH",
            raw,
            normalized,
            args={"filename": filename, "search_path": search_path},
        )
    return None


def _try_media_open_command(raw, normalized):
    lowered = _normalize_for_match(raw)
    direct_match = re.match(
        r"^(?:play|start|launch|open)\s+(?:some\s+)?(?:music\s+(?:on|in)\s+)?(spotify|vlc|youtube\s+music|yt\s+music|youtube)$",
        lowered,
        flags=re.IGNORECASE,
    )
    if direct_match:
        key = " ".join(direct_match.group(1).lower().split())
        target = _MEDIA_APP_TARGETS.get(key)
        if target:
            return ParsedCommand("OS_APP_OPEN", raw, normalized, args={"app_name": target})

    arabic_match = re.match(
        r"^(?:\u0634\u063a\u0644|\u0627\u0641\u062a\u062d)\s+(?:\u0645\u0648\u0633\u064a\u0642\u0649(?:\s+\u0639\u0644\u0649)?\s*)?(\u0633\u0628\u0648\u062a\u064a\u0641\u0627\u064a|spotify|vlc|\u064a\u0648\u062a\u064a\u0648\u0628\s+\u0645\u064a\u0648\u0632\u0643|youtube\s+music)$",
        lowered,
        flags=re.IGNORECASE,
    )
    if arabic_match:
        key = " ".join(arabic_match.group(1).lower().split())
        target = _MEDIA_APP_TARGETS.get(key)
        if target:
            return ParsedCommand("OS_APP_OPEN", raw, normalized, args={"app_name": target})

    return None


def _try_natural_app_open_command(raw, normalized):
    for pattern in _NATURAL_APP_REQUEST_PATTERNS:
        match = pattern.match(raw)
        if not match:
            continue

        target_text = re.sub(r"[.!?,;]+$", "", (match.group(1) or "").strip()).strip()
        app_name = _infer_known_app_name(target_text)
        if app_name:
            return ParsedCommand("OS_APP_OPEN", raw, normalized, args={"app_name": app_name})
    return None


def _try_app_catalog_refresh_command(raw, normalized):
    patterns = (
        re.compile(r"^(?:rescan|refresh|scan)(?:\s+(?:apps?|installed\s+apps?|app\s+catalog|app\s+list))?(?:\s+now)?[.!?]*$", re.IGNORECASE),
        re.compile(r"^(?:find|locate)(?:\s+(?:installed\s+)?apps?)?(?:\s+now)?[.!?]*$", re.IGNORECASE),
        re.compile(r"^(?:اعادة\s+فحص|حدّث|تحديث|اسكن|سكن)(?:\s+(?:التطبيقات|قائمة\s+التطبيقات|كتالوج\s+التطبيقات|البرامج|قائمة\s+البرامج))?[.!؟]*$", re.IGNORECASE),
        re.compile(r"^(?:جدّد|جدد)(?:\s+(?:قائمة\s+)?(?:التطبيقات|البرامج))?[.!؟]*$", re.IGNORECASE),
    )
    for pattern in patterns:
        if pattern.match(raw) or pattern.match(normalized):
            return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "rescan_apps"})
    return None


def _try_natural_schedule_command(raw, normalized):
    # Note: "remind me in N unit to X" is intentionally excluded here — it is
    # handled earlier by the OS_REMINDER regex patterns in _REGEX_TABLE.
    patterns = (
        re.compile(
            r"^(?:in|after)\s+(.+?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+(.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^بعد\s+(.+?)\s*(ثانية|ثواني|دقيقة|دقائق|ساعة|ساعات)\s+(.+)$",
            re.IGNORECASE,
        ),
    )

    for pattern in patterns:
        match = pattern.match(raw)
        if not match:
            continue
        delay_seconds = _duration_to_seconds(match.group(1), match.group(2))
        command_text = str(match.group(3) or "").strip()
        command_text = re.sub(r"^(?:to\s+|أن\s+|ان\s+)", "", command_text, flags=re.IGNORECASE).strip()
        if delay_seconds is None or not command_text:
            continue
        return ParsedCommand(
            "JOB_QUEUE_COMMAND",
            raw,
            normalized,
            action="enqueue",
            args={"delay_seconds": int(delay_seconds), "command_text": command_text},
        )
    return None


def _try_natural_browser_command(raw, normalized):
    """Phase 1.6 — only structural patterns remain.

    Tab open/close + back/forward used to be matched here with keyword loops,
    but those forms are now resolved by the semantic router (Tier 2) which
    already covers ``OS_SYSTEM_COMMAND`` for browser navigation. We keep the
    two patterns that *extract* an argument the router cannot infer: the
    explicit search query and the destination URL.
    """

    # STRUCTURAL: extract a free-form search query.
    search_pattern = re.compile(
        r"(?:^|\b)(?:search(?:\s+(?:the\s+)?)?(?:(?:web|online|internet)\s*(?:for|about)?|(?:for|about))|google|look\s+up|دور(?:\s+على)?(?:\s+(?:النت|اونلاين|أونلاين))?|دوّر(?:\s+على)?(?:\s+(?:النت|اونلاين|أونلاين))?)\s+(.+)$",
        re.IGNORECASE,
    )
    match = search_pattern.search(raw)
    if match and match.group(1).strip():
        query = _clean_browser_search_query(match.group(1))
        if query:
            return ParsedCommand(
                "OS_SYSTEM_COMMAND",
                raw,
                normalized,
                args={"action_key": "browser_search_web", "search_query": query},
            )

    # STRUCTURAL: extract a URL argument from "open ${URL}" / "visit ${URL}".
    open_pattern = re.compile(
        r"^(?:open|visit|go to|browse to|افتح|افتحلي|روح على|خش على|ادخل على)\s+(?:website|site|url\s+|موقع\s+)?(.+)$",
        re.IGNORECASE,
    )
    match = open_pattern.match(raw)
    if match:
        url = _normalize_url_target(match.group(1))
        if url:
            return ParsedCommand(
                "OS_SYSTEM_COMMAND",
                raw,
                normalized,
                args={"action_key": "browser_open_url", "url": url},
            )
    return None


def _try_natural_window_command(raw, normalized):
    """Phase 1.6 — only the focus-window pattern remains.

    Maximize/minimize/snap/close-active/next-window were keyword loops that
    fully overlap with the semantic router's ``OS_SYSTEM_COMMAND`` coverage,
    so we delegate them. The focus pattern stays here because it has to
    extract a window-title argument (``focus chrome`` → window_query=chrome)
    that the semantic router cannot infer on its own.
    """
    focus_pattern = re.compile(
        r"^(?:focus|switch to|bring|ركز على|روح على|خش على|ادخل على)\s+(?:the\s+|window\s+|شباك\s+)?(.+)$",
        re.IGNORECASE,
    )
    match = focus_pattern.match(raw)
    if match:
        query = _canonical_window_query(match.group(1) or "")
        if query:
            return ParsedCommand(
                "OS_SYSTEM_COMMAND",
                raw,
                normalized,
                args={"action_key": "focus_window", "window_query": query},
            )
    return None


def _try_natural_media_control_command(raw, normalized):
    """Phase 1.6 — only the seek-by-N-seconds patterns remain.

    Pause/play/next/previous/stop are now resolved by the semantic router
    (``OS_SYSTEM_COMMAND`` covers ``pause music``, ``next track`` etc.). We
    keep the seek patterns because they have to extract a numeric seek_seconds
    argument that the router can't recover from a paraphrase alone.
    """
    lowered = _normalize_for_match(raw)
    media_context = any(
        token in lowered for token in ("music", "media", "track", "song", "موسيقى", "اغنية")
    )

    direct_en = re.match(
        r"^(?:play|start|resume|pause|stop)(?:\s+(?:some|a|the|my|your|our|any))?\s*(?:music|media|audio|song|track|songs|tracks)?$",
        lowered,
        flags=re.IGNORECASE,
    )
    if direct_en:
        verb = lowered.split()[0]
        action_key = "media_stop" if verb == "stop" else "media_play_pause"
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": action_key})

    direct_ar = re.match(
        r"^(?:شغل|شغّل|كمل|استأنف|وقف|وقّف|اوقف)\s*(?:الموسيقى|المزيكا|الميديا|الاغاني|الاغنية|الاغانيه)?$",
        lowered,
        flags=re.IGNORECASE,
    )
    if direct_ar:
        verb = lowered.split()[0]
        action_key = "media_stop" if verb in {"وقف", "وقّف", "اوقف"} else "media_play_pause"
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": action_key})

    forward = re.search(
        r"(?:seek|skip|forward|قدم)\s+(?:by\s+)?(.+?)?\s*(seconds?|secs?|ثانية|ثواني)?$",
        raw,
        flags=re.IGNORECASE,
    )
    if forward and media_context and ("forward" in lowered or "seek" in lowered or "قدم" in lowered):
        seconds = _duration_to_seconds(forward.group(1) or 10, forward.group(2) or "seconds") or 10
        return ParsedCommand(
            "OS_SYSTEM_COMMAND",
            raw,
            normalized,
            args={"action_key": "media_seek_forward", "seek_seconds": int(seconds)},
        )

    backward = re.search(
        r"(?:seek|skip|back|rewind|ارجع)\s+(?:by\s+)?(.+?)?\s*(seconds?|secs?|ثانية|ثواني)?$",
        raw,
        flags=re.IGNORECASE,
    )
    if backward and media_context and ("back" in lowered or "rewind" in lowered or "ارجع" in lowered):
        seconds = _duration_to_seconds(backward.group(1) or 10, backward.group(2) or "seconds") or 10
        return ParsedCommand(
            "OS_SYSTEM_COMMAND",
            raw,
            normalized,
            args={"action_key": "media_seek_backward", "seek_seconds": int(seconds)},
        )
    return None


def _try_natural_file_operation(raw, normalized):
    # CONSOLIDATED: file operations (create/move/rename/delete) using action-specific patterns
    
    # CREATE folder unified
    create_patterns = (
        re.compile(
            r"^(?:(?:create|make|new)\s+(?:a\s+)?(?:new\s+)?folder(?:\s+(?:called|named))?|(?:اعمل|اعمللي|انشئ|عمل)\s+(?:a\s+)?(?:new\s+)?(?:مجلد\s+)?(?:باسم\s+|اسمه\s+)?|(?:اعمل|اعمللي|انشئ)\s+folder\s*)\s*(.+?)(?:\s+(?:in|inside|under|في|داخل)\s+(.+))?$",
            re.IGNORECASE,
        ),
    )
    for pattern in create_patterns:
        match = pattern.match(raw)
        if not match:
            continue
        name = _strip_file_target_fillers(match.group(1) or "")
        parent = _normalize_search_path_hint(match.group(2) or "")
        if not name:
            continue
        path = os.path.join(parent, name) if parent else name
        return ParsedCommand("OS_FILE_NAVIGATION", raw, normalized, action="create_directory", args={"path": path})

    # UNIFIED MOVE & RENAME (both require source + target, distinguish by verb)
    move_rename_patterns = (
        re.compile(
            r"^(?:(?:move|put)\s+(?:the\s+)?(?:file|folder)?|(?:انقل|نقل|حرك|ودي|ودّي|غير مكان|حول)\s+(?:the\s+)?(?:file|folder|ملف|مجلد)?)\s*(.+?)\s+(?:from|من)\s+(.+?)\s+(?:to|into|to the|الى|الي|على|علي|ل|للـ?)\s*(.+)$",
            re.IGNORECASE,
        ),
        re.compile(r"^(?:(?:rename|change name of|change name)|(?:سمي|سميلي|سمّي|سمّيلي|غير اسم|بدّل اسم))\s+(?:the\s+)?(?:file|folder|ملف|مجلد)?\s*(.+?)\s+(?:to|as|ل(?:ـ)?|لـ|الى)\s+(.+)$", re.IGNORECASE),
    )
    
    for i, pattern in enumerate(move_rename_patterns):
        match = pattern.match(raw)
        if not match or not match.group(1).strip():
            continue

        # Determine action: rename vs move
        action = "rename_item" if i == 1 else "move_item"
        if action == "move_item":
            # Pattern 0 has 3 groups when "from X to Y" is present.
            has_from_clause = match.lastindex is not None and match.lastindex >= 3
            if has_from_clause and match.group(3):
                # "move ITEM from SOURCE to DEST"
                location_source = _normalize_search_path_hint(match.group(2).strip()) or match.group(2).strip()
                filename = _strip_file_target_fillers(match.group(1))
                source = os.path.join(location_source, filename) if location_source else filename
                dest = _normalize_search_path_hint(match.group(3).strip()) or match.group(3).strip()
            else:
                if not match.group(2):
                    continue
                source = _strip_file_target_fillers(match.group(1))
                dest = _normalize_search_path_hint(match.group(2).strip()) or _strip_file_target_fillers(match.group(2))
            args_dict = {"source": source, "destination": dest}
        else:
            if not match.group(2):
                continue
            args_dict = {
                "source": _strip_file_target_fillers(match.group(1) or ""),
                "new_name": _strip_file_target_fillers(match.group(2) or ""),
            }
        
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action=action,
            args=args_dict,
        )

    # UNIFIED DELETE (with optional permanent flag)
    delete_patterns = (
        re.compile(r"^(?:(?:delete|remove)\s+(?:the\s+)?(?:file|folder)?|(?:امسح|شيل)\s+(?:الملف|المجلد)?)\s*(.+?)(?:\s+(permanently|forever|نهائيا|نهائي))?$", re.IGNORECASE),
    )
    for pattern in delete_patterns:
        match = pattern.match(raw)
        if not match:
            continue
        target = _strip_file_target_fillers(match.group(1) or "")
        if not target:
            continue
        permanent = bool(match.group(2))
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="delete_item_permanent" if permanent else "delete_item",
            args={"path": target},
        )
    return None


def _try_system_action(normalized_match, normalized, raw):
    system_action = normalize_system_action(normalized_match) or normalize_system_action(normalized)
    if system_action:
        return ParsedCommand(
            "OS_SYSTEM_COMMAND",
            raw,
            normalized,
            args={"action_key": system_action},
        )
    return None


def _try_cd_commands(normalized, raw):
    if normalized.startswith("go to "):
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="cd",
            args={"path": raw[6:].strip()},
        )
    if normalized.startswith("change directory "):
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="cd",
            args={"path": raw[len("change directory ") :].strip()},
        )
    if normalized.startswith("cd "):
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="cd",
            args={"path": raw[3:].strip()},
        )

    arabic_match = re.match(
        r"^(?:\u0631\u0648\u062d|\u0627\u062f\u062e\u0644|\u062e\u0634)\s+(?:\u0639\u0644\u0649)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if arabic_match:
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="cd",
            args={"path": arabic_match.group(1).strip()},
        )
    return None


# ---------------------------------------------------------------------------
# Phase 3: Command Chaining and Batch Operations
# ---------------------------------------------------------------------------

def _try_command_chaining(raw, normalized):
    """Detect and parse chained commands with conjunctions (AND/OR/THEN)."""
    if (
        _looks_like_explanatory_llm_query(raw)
        or _looks_like_question_llm_query(raw)
        or _looks_like_career_advice_llm_query(raw)
    ):
        return None

    conjunction_pattern = re.compile(
        r'(?:\s+(?:and|or|then)\s+|\s+(?:\u0648(?!احد)|\u0623\u0648|\u062b\u0645)(?:\s+|(?=[\u0600-\u06FFA-Za-z]))|\s+و(?!احد)(?=[\u0600-\u06FFA-Za-z]))',
        re.IGNORECASE
    )
    
    if not conjunction_pattern.search(raw):
        return None
    
    return ParsedCommand(
        "COMMAND_CHAIN",
        raw,
        normalized,
        action="parse_and_execute",
        args={"command_text": raw},
    )


def _try_batch_file_operations(raw, normalized):
    """Detect batch file operations with multiple targets."""
    delete_pattern = re.compile(
        r"^(?:delete|remove|rm)\s+files?\s+(.+?)(?:\s+from\s+(.+))?$",
        re.IGNORECASE,
    )
    delete_ar_pattern = re.compile(
        r"^(?:\u0627\u062d\u0630\u0641|\u0623\u0632\u0644|\u0627\u0645\u0633\u062d)\s+(?:\u0645\u0644\u0641\u0627\u062a)\s+(.+?)(?:\s+(?:\u0641\\u064a|\u0645\u0646)\s+(.+))?$",
        re.IGNORECASE,
    )
    
    m = delete_pattern.match(raw)
    if m:
        files = m.group(1) or ""
        location = m.group(2) or ""
        return ParsedCommand(
            "OS_FILE_NAVIGATION_BATCH",
            raw,
            normalized,
            action="delete_multiple",
            args={"files": files.strip(), "location": location.strip()},
        )
    
    m = delete_ar_pattern.match(raw)
    if m:
        files = m.group(1) or ""
        location = m.group(2) or ""
        return ParsedCommand(
            "OS_FILE_NAVIGATION_BATCH",
            raw,
            normalized,
            action="delete_multiple",
            args={"files": files.strip(), "location": location.strip()},
        )
    
    return None


def _looks_like_explanatory_llm_query(text: str) -> bool:
    """Return True for advice/explanation questions that must stay LLM_QUERY."""
    normalized = _normalize_for_match(text)
    if not normalized:
        return False

    word_count = len(normalized.split())
    english = normalized.lower()

    explanatory_markers = (
        "how to",
        "how can i",
        "how do i",
        "how should i",
        "step by step",
        "steps",
        "explain",
        "\u0627\u0632\u0627\u064a",          # ازاي
        "\u0625\u0632\u0627\u064a",          # إزاي
        "\u0643\u064a\u0641",                # كيف
        "\u062e\u0637\u0648\u0629 \u062e\u0637\u0648\u0629",
        "\u0627\u0644\u062e\u0637\u0648\u0627\u062a",
        "\u062e\u0637\u0648\u0627\u062a",
    )
    tell_me_markers = (
        "tell me",
        "can you tell me",
        "could you tell me",
        "i want you to tell me",
        "\u0642\u0648\u0644\u064a",       # "tell me"
        "\u0642\u0648\u0644\u0644\u064a",
        "\u0642\u0648\u0644 \u0644\u064a",
        "\u0645\u0645\u0643\u0646 \u062a\u0642\u0648\u0644",   # "can you tell"
        "\u0645\u0645\u0643\u0646 \u062a\u0642\u0648\u0644\u064a",
        "\u0639\u0627\u0648\u0632\u0643 \u062a\u0642\u0648\u0644",  # "I want you to tell"
        "\u0639\u0627\u0648\u0632\u0643 \u062a\u0642\u0648\u0644\u064a",
        "\u0639\u0627\u064a\u0632\u0643 \u062a\u0642\u0648\u0644",
        "\u0639\u0627\u064a\u0632\u0643 \u062a\u0642\u0648\u0644\u064a",
        "\u0627\u062d\u0643\u064a\u0644\u064a",      # "tell me" (colloquial)
        "\u0627\u062e\u0628\u0631\u0646\u064a",
        "\u0623\u062e\u0628\u0631\u0646\u064a",
        "\u062e\u0628\u0631\u0646\u064a",
        "\u0627\u0639\u0631\u0641\u0646\u064a",      # "let me know"
        "\u0627\u0634\u0631\u062d",        # "explain"
        "\u0627\u0634\u0631\u062d\u0644\u064a",
        "\u0627\u0634\u0631\u062d \u0644\u064a",
    )
    advice_markers = (
        "successful",
        "engineer",
        "computer engineer",
        "learn",
        "career",
        "i don't know",
        "\u0645\u0634 \u0639\u0627\u0631\u0641",
        "\u0645\u0634 \u0639\u0627\u0631\u0641\u0629",
        "\u0628\u062d\u0627\u0648\u0644",
        "\u0627\u0639\u0645\u0644 \u0643\u062f\u0647",
        "\u0623\u0639\u0645\u0644 \u0643\u062f\u0647",
        "\u0627\u0643\u0648\u0646",
        "\u0623\u0643\u0648\u0646",
        "\u0646\u0627\u062c\u062d",
        "\u0645\u0647\u0646\u062f\u0633",
        "\u0643\u0645\u0628\u064a\u0648\u062a\u0631",
        "\u0627\u062a\u0639\u0644\u0645",
        "\u0623\u062a\u0639\u0644\u0645",
    )

    has_explanatory = any(marker in english for marker in explanatory_markers)
    has_tell_me = any(marker in normalized for marker in tell_me_markers) or any(
        marker in english for marker in tell_me_markers
    )
    has_advice = any(marker in normalized for marker in advice_markers) or any(
        marker in english for marker in advice_markers
    )

    if has_explanatory and (word_count >= 4 or has_advice):
        return True
    if has_tell_me and word_count >= 5:
        return True
    return False


def _looks_like_question_llm_query(text: str) -> bool:
    normalized = _normalize_for_match(text)
    if not normalized:
        return False
    has_question_mark = "?" in str(text or "") or "\u061f" in str(text or "")
    question_markers = (
        "what",
        "why",
        "how",
        "when",
        "where",
        "who",
        "\u0627\u064a\u0647",
        "\u0625\u064a\u0647",
        "\u0644\u064a\u0647",
        "\u0627\u0632\u0627\u064a",
        "\u0625\u0632\u0627\u064a",
        "\u0643\u064a\u0641",
        "\u0647\u0648 \u0644\u064a\u0647",
    )
    return has_question_mark and any(marker in normalized for marker in question_markers)


def _looks_like_career_advice_llm_query(text: str) -> bool:
    normalized = _normalize_for_match(text)
    if not normalized:
        return False

    career_terms = (
        "computer engineering",
        "computer engineer",
        "software engineer",
        "engineer",
        "career",
        "\u0645\u0647\u0646\u062f\u0633",
        "\u0643\u0645\u0628\u064a\u0648\u062a\u0631",
        "\u0628\u0631\u0645\u062c\u0629",
    )
    quality_terms = (
        "successful",
        "good",
        "better",
        "strong",
        "skilled",
        "\u0646\u0627\u062c\u062d",
        "\u0634\u0627\u0637\u0631",
        "\u0643\u0648\u064a\u0633",
        "\u0642\u0648\u064a",
    )
    intent_terms = (
        "want",
        "become",
        "how",
        "tell me",
        "i want",
        "\u0627\u0631\u064a\u062f",
        "\u0623\u0631\u064a\u062f",
        "\u0639\u0627\u064a\u0632",
        "\u0639\u0627\u0648\u0632",
        "\u0627\u0643\u0648\u0646",
        "\u0623\u0643\u0648\u0646",
        "\u0627\u0628\u0642\u0649",
        "\u0623\u0628\u0642\u0649",
        "\u0642\u0648\u0644",
        "\u062a\u0642\u0648\u0644",
        "\u0627\u0632\u064a\u0643",
        "\u0627\u0632\u0627\u064a",
        "\u0625\u0632\u0627\u064a",
    )

    return (
        any(term in normalized for term in career_terms)
        and any(term in normalized for term in quality_terms)
        and any(term in normalized for term in intent_terms)
    )


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

# STT disfluency fillers ("uh", "um", "ايه") that sometimes appear mid-sentence
# with surrounding commas, e.g. "search for hello folder in, uh, desktop".
# Stripped before parsing so they don't break regex captures.
_DISFLUENCY_RE = re.compile(
    r"(?:^|(?<=[\s,،]))(?:uh+|um+|er+|ايه ده|يعني)(?:[\s,،]+|$)",
    re.IGNORECASE,
)


def _strip_disfluencies(text: str) -> str:
    if not text or ("uh" not in text.lower() and "um" not in text.lower() and "ايه ده" not in text and "يعني" not in text):
        return text
    cleaned = _DISFLUENCY_RE.sub(" ", text)
    # Collapse any leftover comma runs/spacing left by the filler removal.
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"(,\s*)+", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"^,\s*|,\s*$", "", cleaned).strip()
    # Drop commas adjacent to a preposition that the regex tables expect to
    # be followed directly by whitespace ("in, desktop" -> "in desktop").
    cleaned = re.sub(r"\b(in|من|في|from)\s*,\s*", r"\1 ", cleaned, flags=re.IGNORECASE)
    return cleaned or text


_OR_CLAUSE_RE = re.compile(
    r"\s+(?:أو|او|or)\s+.+$",
    re.IGNORECASE,
)


def _strip_or_clause(text: str) -> str:
    """Remove trailing 'أو X' / 'or X' alternative clauses the STT sometimes adds.

    "دور على ملف hello في الـ downloads أو folder hello في الـ downloads"
    → "دور على ملف hello في الـ downloads"

    Only strips when the main command already has a clear filename/location so
    the remaining text is still parseable.
    """
    return _OR_CLAUSE_RE.sub("", text).strip()


_NOTE_CREATE_RE = re.compile(
    r"^(?:"
    r"create\s+(?:a\s+)?(?:new\s+)?note"
    r"|new\s+note"
    r"|take\s+(?:a\s+)?note"
    r"|make\s+(?:a\s+)?note"
    r"|note\s+(?:this\s+)?down"
    r"|write\s+(?:a\s+)?note"
    r"|اعمل(?:ي|لي|ولي)?\s+نوت(?:ة)?(?:\s+جديدة?)?"
    r"|نوت(?:ة)?\s+جديد(?:ة)?"
    r"|اكتب(?:لي)?\s+نوت(?:ة)?"
    r"|سجل\s+نوت(?:ة)?"
    r"|دون\s+ملاحظة"
    r"|حفظ\s+ملاحظة"
    r")"
    r"(?:\s+(?:called|named|اسمها|باسم|اسمه)\s+(?P<name>.+?))?$",
    re.IGNORECASE | re.UNICODE,
)

_NOTE_INLINE_RE = re.compile(
    r"^(?:"
    r"note\s+(?:this\s+)?(?:down\s*[:\-]\s*|:\s*)"
    r"|write\s+(?:a\s+)?note\s*[:\-]\s*"
    r"|اكتب\s+(?:نوتة\s+)?[:\-]\s*"
    r"|سجل\s*[:\-]\s*"
    r")"
    r"(?P<body>.+)$",
    re.IGNORECASE | re.UNICODE,
)


def _try_note_command(raw: str, normalized: str):
    """Match note-creation commands; returns ParsedCommand or None."""
    # Inline body: "note this down: buy milk"
    m = _NOTE_INLINE_RE.match(raw)
    if not m:
        m = _NOTE_INLINE_RE.match(normalized)
    if m:
        body = m.group("body").strip()
        if body:
            return ParsedCommand(
                "OS_NOTE", raw, normalized,
                action="create",
                args={"body": body},
            )

    # Create with optional name: "create a note called groceries"
    m = _NOTE_CREATE_RE.match(normalized)
    if not m:
        m = _NOTE_CREATE_RE.match(raw.lower())
    if m:
        name = (m.group("name") or "").strip() or None
        return ParsedCommand(
            "OS_NOTE", raw, normalized,
            action="create",
            args={"name": name} if name else {},
        )

    return None


_SCREEN_EN_PHRASES = frozenset({
    "what's on my screen", "what is on my screen",
    "whats on my screen", "describe my screen",
    "show me what's open", "show me whats open",
    "what's currently on the screen", "what is currently on the screen",
    "what apps are open", "what windows are open",
    "what am i looking at", "describe what's visible",
    "describe whats visible", "what's the active window",
    "what is the active window", "what program am i in",
    "what window is open", "what's open right now",
    "tell me what's on screen", "tell me whats on screen",
    "what do you see on my screen",
})

_SCREEN_AR_PHRASES = frozenset({
    # pre-normalisation (original alef/ya forms)
    "ايه اللي على الشاشة", "إيه اللي على الشاشة",
    "ايه اللي شايفه", "إيه اللي شايفه",
    "وصف الشاشة", "ايه اللي فاتح", "إيه اللي فاتح",
    "ايه اللي مفتوح دلوقتي", "إيه اللي مفتوح دلوقتي",
    "ايه التطبيق اللي شغال", "إيه التطبيق اللي شغال",
    "انا فين دلوقتي", "أنا فين دلوقتي",
    "ايه اللي بيحصل على الشاشة", "إيه اللي بيحصل على الشاشة",
    "قولي ايه اللي شايفه", "قولي إيه اللي شايفه",
    "ايه اللي واقف", "إيه اللي واقف",
    # post-normalisation (normalize_arabic_preserve_digits converts على -> علي)
    "ايه اللي علي الشاشة", "إيه اللي علي الشاشة",
    "ايه اللي بيحصل علي الشاشة", "إيه اللي بيحصل علي الشاشة",
})


def _is_screen_describe_request(normalized: str) -> bool:
    """Return True when the utterance asks Jarvis to describe the screen."""
    value = str(normalized or "").strip().lower()
    if not value:
        return False
    if value in _SCREEN_EN_PHRASES or value in _SCREEN_AR_PHRASES:
        return True
    en_signals = (
        "on my screen", "on the screen", "describe my screen",
        "what's open", "what is open", "apps are open", "windows are open",
        "what am i looking at", "active window", "describe what",
    )
    ar_signals = (
        "على الشاشة", "علي الشاشة",  # pre- and post-normalisation variants
        "الشاشة دلوقتي", "اللي شايفه",
        "اللي مفتوح", "اللي فاتح", "التطبيق اللي شغال",
        "وصف الشاشة", "اللي بيحصل",
    )
    for phrase in en_signals:
        if phrase in value:
            return True
    for phrase in ar_signals:
        if phrase in value:
            return True
    return False


_IDENTITY_EN_PHRASES = frozenset({
    "who are you", "what are you", "introduce yourself",
    "what can you do", "tell me about yourself", "what's your name",
    "what is your name", "are you jarvis", "who is jarvis",
})

_IDENTITY_AR_PHRASES = frozenset({
    "انت مين", "إنت مين", "انت ايه", "انت الايه",
    "انت بتشتغل ازاي", "عرفني بنفسك", "عرفني عليك",
    "بتعمل ايه", "بتعمل إيه", "تعرف تعمل ايه", "تعرف تعمل إيه",
    "اسمك ايه", "اسمك إيه", "مين انت", "مين إنت",
    "هو انت مين", "انت جارفيس", "انت مساعد ايه",
})


def _is_identity_question(normalized: str) -> bool:
    """Return True when the utterance is asking Jarvis to introduce itself."""
    value = str(normalized or "").strip().lower()
    if not value:
        return False
    # Exact phrase match
    if value in _IDENTITY_EN_PHRASES or value in _IDENTITY_AR_PHRASES:
        return True
    # Prefix / substring match for longer phrasings
    en_signals = ("who are you", "what are you", "introduce yourself",
                  "what can you do", "tell me about yourself")
    ar_signals = ("مين انت", "انت مين", "عرفني بنفسك", "عرفني عليك",
                  "بتعمل ايه", "تعرف تعمل", "اسمك ايه", "اسمك إيه",
                  "انت جارفيس", "انت مساعد")
    for phrase in en_signals:
        if phrase in value:
            return True
    for phrase in ar_signals:
        if phrase in value:
            return True
    return False


def parse_command(text: str) -> ParsedCommand:
    raw = text or ""
    raw = _strip_disfluencies(raw)
    raw = _strip_or_clause(raw)
    # Pre-normalize Arabic text: strip tashkeel, normalize alef/ya/gaf variants.
    # Preserves Arabic-Indic digits (٣) so regex captures like time_str keep the
    # original numeral form (normalize_arabic would convert them to ASCII).
    raw = normalize_arabic_preserve_digits(raw) if any(0x0600 <= ord(c) <= 0x06FF for c in raw) else raw
    normalized = " ".join(raw.lower().split()).strip()
    normalized_match = _normalize_for_match(raw)

    # Screen-describe check before the early LLM guard so "tell me what's on
    # screen" isn't captured by _looks_like_explanatory_llm_query.
    if _is_screen_describe_request(normalized):
        return ParsedCommand("OS_SCREEN_DESCRIBE", raw, normalized)

    if (
        _looks_like_explanatory_llm_query(raw)
        or _looks_like_question_llm_query(raw)
        or _looks_like_career_advice_llm_query(raw)
    ):
        return ParsedCommand("LLM_QUERY", raw, normalized)

    spoken_candidate = _strip_spoken_prefixes(normalized_match)

    # Try stripping spoken prefixes and re-parsing.
    if spoken_candidate and spoken_candidate != normalized_match:
        nested = parse_command(spoken_candidate)
        if nested.intent != "LLM_QUERY":
            return ParsedCommand(
                nested.intent,
                raw,
                normalized,
                action=nested.action,
                args=dict(nested.args),
            )

    # Detect negation early and strip it for downstream parsing. Keep the
    # original raw/normalized values for reporting; we will mark the final
    # ParsedCommand.negated flag if a negation was present.
    negated, stripped_norm = _detect_and_strip_negation(normalized_match)
    if negated:
        # Use stripped normalized forms for matching patterns.
        normalized = " ".join(stripped_norm.split())
        normalized_match = stripped_norm

    def _finalize(p: ParsedCommand) -> ParsedCommand:
        if not p:
            return p
        if negated:
            p.negated = True
            p = _apply_negation_to_parsed(p)
        return p

    match_raw = normalized if negated else raw

    # 0. Priority structural table (exact, bilingual, high-confidence toggles).
    result = _try_priority_structural_table(normalized, raw)
    if result:
        return _finalize(result)

    # 0.5 Priority structural regexes for queue/index actions that need args.
    result = _try_priority_regex_table(normalized, raw)
    if result:
        return _finalize(result)

    # 0.55 Note-taking commands (before chaining so "note this down: …" isn't split).
    result = _try_note_command(raw, normalized)
    if result:
        return _finalize(result)

    # 0.6 Phase 3: Command chaining detection
    result = _try_command_chaining(raw, normalized)
    if result:
        return _finalize(result)

    # 1. Keyword table (exact match on normalized).
    result = _try_keyword_table(normalized, raw)
    if result:
        return _finalize(result)

    # 1.5 Mixed Arabic/English command pass.
    result = _try_codeswitched_command(raw, normalized)
    if result:
        return _finalize(result)

    # 2. Regex table.
    result = _try_regex_table(normalized, raw)
    if result:
        return _finalize(result)

    # 2.5 Natural scheduling phrasing.
    result = _try_natural_schedule_command(raw, normalized)
    if result:
        return _finalize(result)

    # 3. Natural file search phrasing.
    result = _try_natural_file_search(raw, normalized)
    if result:
        return _finalize(result)

    # 3.5 Natural media launch phrasing.
    result = _try_media_open_command(raw, normalized)
    if result:
        return _finalize(result)

    # 3.6 Natural media control phrasing.
    result = _try_natural_media_control_command(raw, normalized)
    if result:
        return _finalize(result)

    # 3.7 Natural browser command phrasing.
    result = _try_natural_browser_command(raw, normalized)
    if result:
        return _finalize(result)

    # 3.8 Natural window command phrasing.
    result = _try_natural_window_command(raw, normalized)
    if result:
        return result

    # 3.9 Natural app-open phrasing.
    result = _try_natural_app_open_command(match_raw, normalized)
    if result:
        return _finalize(result)

    # 3.95 App catalog refresh phrasing.
    result = _try_app_catalog_refresh_command(raw, normalized)
    if result:
        return _finalize(result)

    # 4. Drive open heuristic.
    result = _try_drive_open(normalized_match, raw, normalized)
    if result:
        return _finalize(result)

    # 5. "open ..." disambiguation.
    result = _try_open_command(match_raw, normalized)
    if result:
        return _finalize(result)

    # 5.5 Natural file operation phrasing.
    result = _try_natural_file_operation(raw, normalized)
    if result:
        return _finalize(result)

    # 5.6 Phase 3: Batch file operations
    result = _try_batch_file_operations(raw, normalized)
    if result:
        return _finalize(result)

    # 6. System action aliases.
    result = _try_system_action(normalized_match, normalized, raw)
    if result:
        return _finalize(result)

    # 7. Natural close-app phrasing.
    result = _try_close_command(match_raw, normalized)
    if result:
        return _finalize(result)

    # 8. CD / navigation commands.
    result = _try_cd_commands(normalized, raw)
    if result:
        return _finalize(result)

    # 8.3 Screen-describe request.
    if _is_screen_describe_request(normalized):
        return _finalize(ParsedCommand("OS_SCREEN_DESCRIBE", raw, normalized))

    # 8.5 Identity / self-introduction questions.
    if _is_identity_question(normalized):
        return _finalize(ParsedCommand("IDENTITY", raw, normalized))

    # 9. LLM fallback.
    return _finalize(ParsedCommand("LLM_QUERY", raw, normalized))


# ---------------------------------------------------------------------------
# Spoken-PIN confirmation (Phase 1)
# ---------------------------------------------------------------------------
_PIN_WORD_TOKEN_RE = re.compile(r"[a-zA-Z؀-ۿ]+|\d+")
_PIN_DIGIT_WORDS = {k: v for k, v in _NUMBER_ONES.items() if v <= 9}
# Egyptian colloquial spellings not covered by the formal _NUMBER_ONES table.
_PIN_DIGIT_WORDS.update({
    "صفر": 0,
    "واحد": 1, "وحدة": 1,
    "اتنين": 2, "اتنان": 2,
    "تلاتة": 3, "تلات": 3,
    "اربعة": 4, "اربعه": 4, "اربع": 4,
    "خمسة": 5, "خمسه": 5, "خمس": 5,
    "ستة": 6, "سته": 6, "ست": 6,
    "سبعة": 7, "سبعه": 7, "سبع": 7,
    "تمانية": 8, "تمانيه": 8, "تمن": 8,
    "تسعة": 9, "تسعه": 9, "تسع": 9,
})

# Filler words that may surround a spoken PIN — strip these before parsing.
# Covers "the pin is", "pin:", "my pin is", "الرقم هو", "الرقم السري", etc.
_PIN_FILLER_RE = re.compile(
    r"\b(?:the\s+)?(?:pin|p\.?i\.?n\.?|code|passcode|رقم(?:\s+السري)?|الرقم(?:\s+السري)?)"
    r"(?:\s+(?:is|هو|ده|هي))?\s*[:\-]?\s*",
    re.IGNORECASE,
)
# Trailing filler: "is the pin", "is the pin for the shutdown", etc.
_PIN_TAIL_RE = re.compile(
    r"[\s,،]+(?:is|was|هو|ده)\b.*$",
    re.IGNORECASE,
)


def _extract_pin_digits(text):
    """Return a digit string from a text of digit-words/digits, or None."""
    tokens = _PIN_WORD_TOKEN_RE.findall(text.lower())
    digits = []
    for token in tokens:
        if token.isdigit():
            digits.extend(list(token))
        elif token in _PIN_DIGIT_WORDS:
            digits.append(str(_PIN_DIGIT_WORDS[token]))
        else:
            return None
    return "".join(digits) if digits else None


def try_parse_pin_confirm(text):
    """Extract a spoken PIN from an utterance while a PIN request is pending.

    Accepts all of:
      - Pure digits / number-words:  "1234", "one two three four"
      - PIN with leading filler:     "pin is one two three four", "الرقم 2468"
      - PIN with trailing filler:    "two four six eight is the pin"
      - Arabic-Indic digits:         "٢٤٦٨"

    Returns ParsedCommand(intent="OS_PIN_CONFIRM", args={"pin": "1234"}) or None.
    """
    raw = str(text or "").strip()
    if not raw:
        return None

    candidate = convert_arabic_numerals(raw)

    # Fast path: bare digit string (possibly space-separated).
    collapsed = re.sub(r"[\s,،-]+", "", candidate)
    if collapsed.isdigit():
        return ParsedCommand("OS_PIN_CONFIRM", raw, raw.lower(), args={"pin": collapsed})

    # Strip known PIN intro/outro filler and try again.
    stripped = _PIN_FILLER_RE.sub("", candidate).strip()
    stripped = _PIN_TAIL_RE.sub("", stripped).strip()
    # Also strip trailing punctuation.
    stripped = re.sub(r"[.,،؟?!]+$", "", stripped).strip()

    collapsed2 = re.sub(r"[\s,،-]+", "", stripped)
    if collapsed2.isdigit():
        return ParsedCommand("OS_PIN_CONFIRM", raw, raw.lower(), args={"pin": collapsed2})

    # Try parsing stripped text as number-words only.
    pin = _extract_pin_digits(stripped)
    if pin:
        return ParsedCommand("OS_PIN_CONFIRM", raw, raw.lower(), args={"pin": pin})

    # Last resort: pure number-words with no extra tokens in the full text.
    pin = _extract_pin_digits(candidate)
    if pin:
        return ParsedCommand("OS_PIN_CONFIRM", raw, raw.lower(), args={"pin": pin})

    return None


def extract_pin_from_text(text):
    """Extract a PIN digit string embedded anywhere in a longer utterance.

    Used when a sensitive command and a PIN appear in the same utterance
    (e.g. "shutdown, the pin is 2468"). Returns the digit string or None.
    Deliberately loose — the caller verifies the PIN against the hash.
    """
    if not text:
        return None
    candidate = convert_arabic_numerals(str(text))

    # Look for an explicit "pin is <digits/words>" phrase.
    filler_match = _PIN_FILLER_RE.search(candidate)
    if filler_match:
        after = candidate[filler_match.end():].strip()
        after = _PIN_TAIL_RE.sub("", after).strip()
        after = re.sub(r"[.,،؟?!\s]+$", "", after).strip()
        collapsed = re.sub(r"[\s,،-]+", "", after)
        if collapsed.isdigit():
            return collapsed
        pin = _extract_pin_digits(after)
        if pin:
            return pin

    # Look for a bare digit run of 4+ characters (typical PIN length).
    digit_run = re.search(r"\b(\d{4,})\b", candidate)
    if digit_run:
        return digit_run.group(1)

    # "Two four six eight is the pin for the shutdown" — number-words before tail.
    stripped_tail = _PIN_TAIL_RE.sub("", candidate).strip()
    stripped_tail = re.sub(r"[.,،؟?!]+$", "", stripped_tail).strip()
    if stripped_tail != candidate.strip():
        pin = _extract_pin_digits(stripped_tail)
        if pin:
            return pin

    return None


