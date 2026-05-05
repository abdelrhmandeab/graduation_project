import os
import re
from dataclasses import dataclass, field

from core.config import CONFIRMATION_TOKEN_BYTES, CONFIRMATION_TOKEN_MIN_HEX_LEN
from nlp.codeswitching import normalize_codeswitched
from os_control.system_ops import normalize_system_action


@dataclass
class ParsedCommand:
    intent: str
    raw: str
    normalized: str
    action: str = ""
    args: dict = field(default_factory=dict)


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
    r"^(?:\u0627\u0644)\s+",
)
_FILESYSTEM_OPEN_HINTS = (
    "drive",
    "partition",
    "folder",
    "directory",
    "desktop",
    "downloads",
    "documents",
    "pictures",
    "music",
    "videos",
    "file explorer",
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
    "\u0627\u0644\u0645\u0648\u0633\u064a\u0642\u0649",
    "\u0627\u0644\u0641\u064a\u062f\u064a\u0648\u0647\u0627\u062a",
)
_SPECIAL_FOLDER_ALIASES = {
    "desktop": "Desktop",
    "\u0633\u0637\u062d \u0627\u0644\u0645\u0643\u062a\u0628": "Desktop",
    "downloads": "Downloads",
    "download": "Downloads",
    "\u0627\u0644\u062a\u062d\u0645\u064a\u0644\u0627\u062a": "Downloads",
    "\u0627\u0644\u062a\u0646\u0632\u064a\u0644\u0627\u062a": "Downloads",
    "documents": "Documents",
    "document": "Documents",
    "\u0627\u0644\u0645\u0633\u062a\u0646\u062f\u0627\u062a": "Documents",
    "pictures": "Pictures",
    "picture": "Pictures",
    "\u0627\u0644\u0635\u0648\u0631": "Pictures",
    "music": "Music",
    "\u0627\u0644\u0645\u0648\u0633\u064a\u0642\u0649": "Music",
    "videos": "Videos",
    "video": "Videos",
    "\u0627\u0644\u0641\u064a\u062f\u064a\u0648\u0647\u0627\u062a": "Videos",
}
_SEARCH_PATH_ALIASES = {
    **_SPECIAL_FOLDER_ALIASES,
    "\u0627\u0644\u0645\u0643\u062a\u0628": "Desktop",
}
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
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "دقيقة": 60,
    "دقائق": 60,
    "دقايق": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "ساعة": 3600,
    "ساعات": 3600,
}
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
    "اثنين": 2,
    "ثلاثة": 3,
    "اربعة": 4,
    "خمسة": 5,
    "ستة": 6,
    "سبعة": 7,
    "ثمانية": 8,
    "تسعة": 9,
    "عشرة": 10,
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


def _normalize_for_match(text: str) -> str:
    lowered = " ".join((text or "").lower().split()).strip()
    cleaned = _MATCH_SANITIZE_RE.sub(" ", lowered)
    return _COLLAPSE_WS_RE.sub(" ", cleaned).strip()


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


def _normalize_search_path_hint(path_hint: str):
    candidate = _collapse_repeated_phrase(path_hint)
    if not candidate:
        return None

    lowered = candidate.lower().strip()
    alias = _SEARCH_PATH_ALIASES.get(lowered)
    if alias:
        return os.path.join(os.path.expanduser("~"), alias)
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

    text = _normalize_for_match(str(value or ""))
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
        return None
    unit = _normalize_for_match(unit_text)
    factor = _DURATION_UNIT_SECONDS.get(unit, 1)
    return max(1, min(86400, int(number * factor)))


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


def _try_codeswitched_command(raw, normalized):
    intent_lang, entity_lang, entities = normalize_codeswitched(raw)
    intent = str((entities or {}).get("intent") or "").strip().lower()
    entity = str((entities or {}).get("entity") or "").strip()
    entity_normalized = _normalize_for_match(entity)

    if not intent or not entity:
        return None

    if intent == "open":
        app_name = _infer_known_app_name(entity)
        if app_name:
            return ParsedCommand("OS_APP_OPEN", raw, normalized, args={"app_name": app_name})

        if entity_normalized in {"files", "file", "folder", "folders", "directory", "directories", "المفات", "الملفات", "المجلد", "المجلدات"}:
            return ParsedCommand("OS_FILE_NAVIGATION", raw, normalized, action="list_directory", args={"path": ""})

        if entity_normalized in {"music", "spotify", "vlc", "youtube music", "youtube", "song", "songs", "الموسيقى", "المزيكا"}:
            app_name = _infer_known_app_name(entity) or _infer_known_app_name("spotify")
            if app_name:
                return ParsedCommand("OS_APP_OPEN", raw, normalized, args={"app_name": app_name})

    if intent == "close":
        app_name = _infer_known_app_name(entity)
        if app_name:
            return ParsedCommand("OS_APP_CLOSE", raw, normalized, args={"app_name": app_name})

    if intent == "search":
        if entity_normalized in {"files", "file", "folder", "folders", "document", "documents", "الملفات", "المستندات", "المجلد", "المجلدات"}:
            query = str((entities or {}).get("source_text") or raw).strip()
            query = re.sub(
                r"^(?:search files? for|search for|look for|find|search|ابحث عن|دور على|دوّر على)\s+",
                "",
                query,
                flags=re.IGNORECASE,
            ).strip()
            if query:
                return ParsedCommand("OS_FILE_SEARCH", raw, normalized, args={"filename": query, "search_path": ""})

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
                return ParsedCommand("OS_FILE_SEARCH", raw, normalized, args={"filename": entity, "search_path": ""})

        query = str((entities or {}).get("source_text") or raw).strip()
        if query:
            return ParsedCommand(
                "OS_SYSTEM_COMMAND",
                raw,
                normalized,
                args={"action_key": "browser_search_web", "search_query": query},
            )

    if intent in {"stop", "mute"} and entity_normalized in {"music", "musiqa", "mزيكا", "الموسيقى", "المزيكا", "media"}:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "media_stop"})

    return None


def _contains_any_phrase(text: str, phrases):
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in phrases)


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
            "paste clipboard",
            "اقرا الكليب بورد",
            "وريني الكليب بورد",
            "ايه في الكليب بورد",
            "في الكليبورد ايه",
            "انسخ من الكليب بورد",
            "اللي في الكليب بورد",
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
            "افتح ايميل جديد",
            "ايميل جديد",
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
]


def _try_keyword_table(normalized, raw):
    for entry in _KEYWORD_TABLE:
        keywords, intent, action = entry[0], entry[1], entry[2]
        if normalized in keywords:
            args = entry[3] if len(entry) > 3 else {}
            return ParsedCommand(intent, raw, normalized, action=action, args=dict(args))
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

_REGEX_TABLE = [
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
    # Arabic colloquial volume down mapping and colloquial screenshot phrasing
    (
        re.compile(r"^(?:وطي\s+الصوت|اخفض\s+الصوت|خف\u0651\u0636\s+الصوت|خفف\s+الصوت)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "volume_down"},
    ),
    (
        re.compile(r"^(?:خد\s+سكرين\s+شوت|خد\s+سكرينشوت|خذ\s+سكرينشوت|خذ\s+سكرين\s+شوت)$", re.IGNORECASE),
        False,
        "OS_SYSTEM_COMMAND",
        "",
        lambda _m: {"action_key": "screenshot"},
    ),
    # Audit
    (
        re.compile(r"^show audit log(?:\s+(\d+))?$"),
        False,
        "AUDIT_LOG_REPORT",
        "",
        lambda m: {"limit": int(m.group(1)) if m.group(1) else 10},
    ),
    # Policy
    (
        re.compile(r"^policy profile\s+([a-z0-9_-]+)$"),
        False,
        "POLICY_COMMAND",
        "set_profile",
        lambda m: {"profile": m.group(1)},
    ),
    (
        re.compile(r"^policy (?:read only|readonly)\s+(on|off)$"),
        False,
        "POLICY_COMMAND",
        "set_read_only",
        lambda m: {"enabled": m.group(1) == "on"},
    ),
    (
        re.compile(r"^policy (?:dry run|dry-run|dryrun)\s+(on|off)$"),
        False,
        "POLICY_COMMAND",
        "set_dry_run",
        lambda m: {"enabled": m.group(1) == "on"},
    ),
    (
        re.compile(r"^policy permission\s+([a-z_]+)\s+(on|off)$"),
        False,
        "POLICY_COMMAND",
        "set_permission",
        lambda m: {"permission": m.group(1), "enabled": m.group(2) == "on"},
    ),
    # Batch
    (
        re.compile(r"^batch add\s+(.+)$", re.IGNORECASE),
        True,
        "BATCH_COMMAND",
        "add",
        lambda m: {"command_text": m.group(1).strip()},
    ),
    # Search index
    (
        re.compile(r"^index refresh(?:\s+in\s+(.+))?$", re.IGNORECASE),
        True,
        "SEARCH_INDEX_COMMAND",
        "refresh",
        lambda m: {"root": (m.group(1) or "").strip() or None},
    ),
    (
        re.compile(r"^(?:indexed find|index find|search indexed)\s+(.+?)(?:\s+in\s+(.+))?$", re.IGNORECASE),
        True,
        "SEARCH_INDEX_COMMAND",
        "search",
        lambda m: {"query": m.group(1).strip(), "root": (m.group(2) or "").strip() or None},
    ),
    # Job queue
    (
        re.compile(r"^(?:queue job|job add)\s+in\s+(\d+)\s*(?:s|sec|secs|seconds)?\s+(.+)$", re.IGNORECASE),
        True,
        "JOB_QUEUE_COMMAND",
        "enqueue",
        lambda m: {"delay_seconds": int(m.group(1)), "command_text": m.group(2).strip()},
    ),
    (
        re.compile(r"^(?:queue job|job add)\s+(.+)$", re.IGNORECASE),
        True,
        "JOB_QUEUE_COMMAND",
        "enqueue",
        lambda m: {"delay_seconds": 0, "command_text": m.group(1).strip()},
    ),
    (
        re.compile(r"^job status\s+(\d+)$"),
        False,
        "JOB_QUEUE_COMMAND",
        "status",
        lambda m: {"job_id": int(m.group(1))},
    ),
    (
        re.compile(r"^job cancel\s+(\d+)$"),
        False,
        "JOB_QUEUE_COMMAND",
        "cancel",
        lambda m: {"job_id": int(m.group(1))},
    ),
    (
        re.compile(r"^job retry\s+(\d+)(?:\s+in\s+(\d+)\s*(?:s|sec|secs|seconds)?)?$"),
        False,
        "JOB_QUEUE_COMMAND",
        "retry",
        lambda m: {"job_id": int(m.group(1)), "delay_seconds": int(m.group(2) or 0)},
    ),
    (
        re.compile(r"^job list(?:\s+([a-z]+|\d+))?(?:\s+(\d+))?$"),
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
            r"^(?:find file|search file|دور على ملف|دوّر على ملف|وريني ملف|هاتلي ملف|دورلي على ملف|دورلي ملف|لقيلي ملف|فين ملف)\s+(.+?)(?:\s+(?:in|\u0641\u064a)\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_SEARCH",
        "",
        lambda m: {"filename": _strip_file_target_fillers(m.group(1)), "search_path": (m.group(2) or "").strip() or None},
    ),
    # File nav - regex-based
    (
        re.compile(
            r"^(?:list files|list directory|show files|show directory|وريني الملفات|هاتلي الملفات|وريني المجلد|هاتلي المجلد|شوفلي الملفات|ايه في المجلد)(?:\s+(?:in|\u0641\u064a)\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "list_directory",
        lambda m: {"path": (m.group(1) or "").strip() or None},
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
            r"^(?:create folder|make folder|mkdir|اعمل مجلد|اعمللي مجلد)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "create_directory",
        lambda m: {"path": _strip_file_target_fillers(m.group(1))},
    ),
    (
        re.compile(
            r"^(?:(?:delete|remove)\s+(?:permanently|forever)\s+(.+)|(?:permanent\s+delete|force\s+delete)\s+(.+)|(?:amسح|شيل)\s+(.+?)\s+(?:نهائيا|نهائي|permanently|forever)|(?:delete|remove)\s+(.+?)\s+(?:permanently|forever|نهائيا|نهائي))$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "delete_item_permanent",
        lambda m: {"path": _strip_file_target_fillers((m.group(1) or m.group(2) or m.group(3) or m.group(4) or "").strip())},
    ),
    (
        re.compile(r"^(?:delete|remove|امسح|شيل)\s+(.+)$", re.IGNORECASE),
        True,
        "OS_FILE_NAVIGATION",
        "delete_item",
        lambda m: {"path": _strip_file_target_fillers(m.group(1))},
    ),
    (
        re.compile(
            r"^(?:move|حرك|ودي|ودّي)\s+(.+?)\s+(?:to|على)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "move_item",
        lambda m: {"source": _strip_file_target_fillers(m.group(1)), "destination": _strip_file_target_fillers(m.group(2))},
    ),
    (
        re.compile(
            r"^(?:rename|سمي|سميلي|سمّي|سمّيلي)\s+(.+?)\s+(?:to|ل)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "rename_item",
        lambda m: {"source": _strip_file_target_fillers(m.group(1)), "new_name": _strip_file_target_fillers(m.group(2))},
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
    ),
    (
        re.compile(
            r"^(?:حط|حطلي|ظبط|ظبّط|اعمل|اعمللي|اضبط|اضبطلي)\s+(?:تايمر|منبه|alarm|timer)\s+(\S+)\s+(ثانية|ثواني|دقيقة|دقائق|دقايق|ساعة|ساعات|seconds?|secs?|minutes?|mins?)$",
            re.IGNORECASE,
        ),
        True,
        "OS_TIMER",
        "set",
        lambda m: {"seconds": _duration_to_seconds(m.group(1), m.group(2)), "label": "Timer"},
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
    # Email — "draft email to X about Y", "ابعت ايميل ل X عن Y"
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
    (
        re.compile(
            r"^(?:ابعت|اكتب|افتح)\s+(?:ايميل|إيميل)\s+(?:ل\s*)?(\S+@\S+)?(?:\s+(?:عن|بخصوص)\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_EMAIL",
        "draft",
        lambda m: {"to": (m.group(1) or "").strip(), "subject": (m.group(2) or "").strip()},
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
    for pattern, use_raw, intent, action, args_builder in _REGEX_TABLE:
        text = raw if use_raw else normalized
        m = pattern.match(text)
        if m:
            return ParsedCommand(intent, raw, normalized, action=action, args=args_builder(m))
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
        r"^(?:open|launch|start|\u0627\u0641\u062a\u062d|\u0627\u0641\u062a\u062d\u0644\u064a|\u0634\u063a\u0644|\u0634\u063a\u0644\u0644\u064a)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if not open_match:
        return None

    target_raw = re.sub(r"[.!?,;]+$", "", open_match.group(1).strip()).strip()
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
                r"(?:\u062f\u0648\u0631|\u062f\u0648\u0651\u0631|\u062f\u0648\u0631\u0644\u064a|\u062f\u0648\u0651\u0631\u0644\u064a)(?:\s+\u0639\u0646)?\s+(?:\u0645\u0644\u0641\s+)?"
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

    play_music = re.match(r"^(?:play|start)\s+(?:some\s+)?music$", lowered, flags=re.IGNORECASE)
    if play_music:
        return ParsedCommand("OS_APP_OPEN", raw, normalized, args={"app_name": "spotify"})

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
        re.compile(r"^(?:rescan|refresh|scan)(?:\s+(?:apps?|installed\s+apps?|app\s+catalog))?(?:\s+now)?[.!?]*$", re.IGNORECASE),
        re.compile(r"^(?:اعادة\s+فحص|حدّث|تحديث)(?:\s+(?:التطبيقات|قائمة\s+التطبيقات|كتالوج\s+التطبيقات))?[.!؟]*$", re.IGNORECASE),
    )
    for pattern in patterns:
        if pattern.match(raw) or pattern.match(normalized):
            return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "rescan_apps"})
    return None


def _try_natural_schedule_command(raw, normalized):
    patterns = (
        re.compile(
            r"^(?:in|after)\s+(.+?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+(.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^remind\s+me\s+in\s+(.+?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+to\s+(.+)$",
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
            r"^(?:(?:create|make)\s+(?:a\s+)?(?:new\s+)?folder(?:\s+(?:called|named))?|(?:اعمل|اعمللي)\s+(?:مجلد\s+)?(?:باسم\s+)?)\s+(.+?)(?:\s+(?:in|inside|under|في|داخل)\s+(.+))?$",
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
            r"^(?:(?:move|put)\s+(?:the\s+)?(?:file|folder)?|(?:حرك|ودي|ودّي)\s+(?:الملف|المجلد)?)\s*(.+?)\s+(?:to|into|inside|under|على)\s+(.+)$",
            re.IGNORECASE,
        ),
        re.compile(r"^(?:(?:rename|change name of)|(?:سمي|سميلي|سمّي|سمّيلي))\s+(.+?)\s+(?:to|as|ل)\s+(.+)$", re.IGNORECASE),
    )
    
    for i, pattern in enumerate(move_rename_patterns):
        match = pattern.match(raw)
        if not match or not match.group(1).strip() or not match.group(2).strip():
            continue
        
        # Determine action: rename vs move
        action = "rename_item" if i == 1 else "move_item"
        if action == "move_item":
            args_dict = {
                "source": _strip_file_target_fillers(match.group(1) or ""),
                "destination": _strip_file_target_fillers(match.group(2) or ""),
            }
        else:
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
# Main parser
# ---------------------------------------------------------------------------


def parse_command(text: str) -> ParsedCommand:
    raw = text or ""
    normalized = " ".join(raw.lower().split()).strip()
    normalized_match = _normalize_for_match(raw)
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

    # 1. Keyword table (exact match on normalized).
    result = _try_keyword_table(normalized, raw)
    if result:
        return result

    # 1.5 Mixed Arabic/English command pass.
    result = _try_codeswitched_command(raw, normalized)
    if result:
        return result

    # 2. Regex table.
    result = _try_regex_table(normalized, raw)
    if result:
        return result

    # 2.5 Natural scheduling phrasing.
    result = _try_natural_schedule_command(raw, normalized)
    if result:
        return result

    # 3. Natural file search phrasing.
    result = _try_natural_file_search(raw, normalized)
    if result:
        return result

    # 3.5 Natural media launch phrasing.
    result = _try_media_open_command(raw, normalized)
    if result:
        return result

    # 3.6 Natural media control phrasing.
    result = _try_natural_media_control_command(raw, normalized)
    if result:
        return result

    # 3.7 Natural browser command phrasing.
    result = _try_natural_browser_command(raw, normalized)
    if result:
        return result

    # 3.8 Natural window command phrasing.
    result = _try_natural_window_command(raw, normalized)
    if result:
        return result

    # 3.9 Natural app-open phrasing.
    result = _try_natural_app_open_command(raw, normalized)
    if result:
        return result

    # 3.95 App catalog refresh phrasing.
    result = _try_app_catalog_refresh_command(raw, normalized)
    if result:
        return result

    # 4. Drive open heuristic.
    result = _try_drive_open(normalized_match, raw, normalized)
    if result:
        return result

    # 5. "open ..." disambiguation.
    result = _try_open_command(raw, normalized)
    if result:
        return result

    # 5.5 Natural file operation phrasing.
    result = _try_natural_file_operation(raw, normalized)
    if result:
        return result

    # 6. System action aliases.
    result = _try_system_action(normalized_match, normalized, raw)
    if result:
        return result

    # 7. Natural close-app phrasing.
    result = _try_close_command(raw, normalized)
    if result:
        return result

    # 8. CD / navigation commands.
    result = _try_cd_commands(normalized, raw)
    if result:
        return result

    # 9. LLM fallback.
    return ParsedCommand("LLM_QUERY", raw, normalized)


