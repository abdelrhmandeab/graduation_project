import os
import re
from dataclasses import dataclass, field

from core.config import CONFIRMATION_TOKEN_BYTES, CONFIRMATION_TOKEN_MIN_HEX_LEN
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
            r"^(?:\u0627\u0631\u064a\u062f|\u0623\u0631\u064a\u062f|\u0639\u0627\u064a\u0632|\u0639\u0627\u0648\u0632|\u0627\u0628\u063a\u0649|\u0623\u0628\u063a\u0649)"
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
        "\u0627\u0639\u0631\u0636",
        "\u0627\u0638\u0647\u0631",
        "\u062a\u0635\u0641\u062d",
        "\u0627\u062f\u062e\u0644",
    )
    if any(verb in lowered for verb in explicit_verbs):
        return True
    if "go to" in lowered and ("drive" in lowered or "partition" in lowered):
        return True
    if (
        ("\u0627\u0630\u0647\u0628 \u0627\u0644\u0649" in lowered or "\u0627\u0646\u062a\u0642\u0644 \u0627\u0644\u0649" in lowered)
        and (
            "\u0642\u0631\u0635" in lowered
            or "\u0628\u0627\u0631\u062a\u0634\u0646" in lowered
            or "\u0642\u0633\u0645" in lowered
            or "\u062f\u0631\u0627\u064a\u0641" in lowered
        )
    ):
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
    candidate = re.sub(r"^(?:website|site|url|موقع|رابط)\s+", "", candidate, flags=re.IGNORECASE).strip()
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


def _normalize_language_value(value: str):
    token = _normalize_for_match(value)
    if token in {"ar", "arabic", "العربية", "عربي"}:
        return "ar"
    if token in {"en", "english", "الانجليزية", "الإنجليزية", "انجليزي"}:
        return "en"
    return token


# ---------------------------------------------------------------------------
# Table-driven keyword matching
# ---------------------------------------------------------------------------
# Each entry: (set_of_keywords, intent, action)
# Matched against `normalized`.

_KEYWORD_TABLE = [
    # Observability
    ({"observability", "observability report", "show observability", "dashboard"}, "OBSERVABILITY_REPORT", ""),
    # Benchmark
    ({"benchmark run", "run benchmark", "benchmark quick"}, "BENCHMARK_COMMAND", "run"),
    ({"resilience demo", "run resilience demo", "failure demo"}, "BENCHMARK_COMMAND", "resilience_demo"),
    (
        {
            "benchmark wake",
            "run wake benchmark",
            "wake benchmark",
            "benchmark wake reliability",
            "wake reliability benchmark",
        },
        "BENCHMARK_COMMAND",
        "wake_reliability",
    ),
    (
        {
            "benchmark stt",
            "run stt benchmark",
            "stt benchmark",
            "benchmark speech to text",
            "speech to text benchmark",
            "benchmark wer",
        },
        "BENCHMARK_COMMAND",
        "stt_reliability",
    ),
    (
        {
            "benchmark tts",
            "run tts benchmark",
            "tts benchmark",
            "benchmark speech synthesis",
            "speech synthesis benchmark",
            "benchmark tts quality",
        },
        "BENCHMARK_COMMAND",
        "tts_quality",
    ),
    # Persona
    ({"persona status", "persona show"}, "PERSONA_COMMAND", "status"),
    ({"persona list", "list personas"}, "PERSONA_COMMAND", "list"),
    ({"assistant mode", "assistant mode on"}, "PERSONA_COMMAND", "set", {"profile": "assistant"}),
    # Voice
    ({"voice status", "speech status", "حالة الصوت", "حالة النطق"}, "VOICE_COMMAND", "status"),
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
            "وضع الكمون منخفض",
            "وضع الاستجابة سريع",
            "وضع السرعة سريع",
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
            "وضع الكمون متوازن",
            "وضع الاستجابة متوازن",
            "وضع السرعة متوازن",
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
            "وضع الكمون ثابت",
            "وضع الاستجابة ثابت",
            "وضع الاستجابة قوي",
        },
        "VOICE_COMMAND",
        "audio_ux_profile_set",
        {"profile": "robust"},
    ),
    ({"audio ux status", "audio profile status", "voice audio status", "حالة تجربة الصوت", "حالة ملف تجربة الصوت"}, "VOICE_COMMAND", "audio_ux_status"),
    ({"audio ux profiles", "audio ux profile list", "list audio ux profiles", "قائمة ملفات تجربة الصوت", "ملفات تجربة الصوت"}, "VOICE_COMMAND", "audio_ux_profiles"),
    ({"audio ux profile balanced", "audio profile balanced", "set audio profile balanced", "ملف تجربة الصوت متوازن", "وضع تجربة الصوت متوازن", "وضع الصوت متوازن"}, "VOICE_COMMAND", "audio_ux_profile_set", {"profile": "balanced"}),
    ({"audio ux profile responsive", "audio profile responsive", "set audio profile responsive", "ملف تجربة الصوت سريع", "وضع تجربة الصوت سريع", "وضع الصوت سريع"}, "VOICE_COMMAND", "audio_ux_profile_set", {"profile": "responsive"}),
    ({"audio ux profile robust", "audio profile robust", "set audio profile robust", "ملف تجربة الصوت قوي", "وضع تجربة الصوت قوي", "وضع الصوت قوي", "وضع الصوت ثابت"}, "VOICE_COMMAND", "audio_ux_profile_set", {"profile": "robust"}),
    ({"voice quality status", "speech quality status", "tts quality status", "حالة جودة الصوت", "حالة جودة النطق"}, "VOICE_COMMAND", "voice_quality_status"),
    ({"voice quality natural", "speech quality natural", "tts quality natural", "natural voice mode", "جودة الصوت طبيعي", "وضع الصوت طبيعي", "وضع النطق طبيعي"}, "VOICE_COMMAND", "voice_quality_set", {"mode": "natural"}),
    ({"voice quality standard", "speech quality standard", "tts quality standard", "robot voice mode", "robotic voice mode", "جودة الصوت قياسي", "وضع الصوت قياسي", "وضع الصوت روبوتي"}, "VOICE_COMMAND", "voice_quality_set", {"mode": "standard"}),
    ({"stt profile status", "speech profile status", "voice stt profile status", "حالة ملف الاستماع"}, "VOICE_COMMAND", "stt_profile_status"),
    ({"stt profile quiet", "speech profile quiet", "ملف الاستماع هادئ", "وضع الاستماع هادئ"}, "VOICE_COMMAND", "stt_profile_set", {"profile": "quiet"}),
    ({"stt profile noisy", "speech profile noisy", "ملف الاستماع ضوضاء", "وضع الاستماع ضوضاء"}, "VOICE_COMMAND", "stt_profile_set", {"profile": "noisy"}),
    ({"stt backend status", "speech backend status", "voice stt backend status", "حالة محرك الاستماع"}, "VOICE_COMMAND", "stt_backend_status"),
    ({"stt backend whisper", "stt backend faster whisper", "set stt backend faster whisper", "set stt backend faster_whisper"}, "VOICE_COMMAND", "stt_backend_set", {"backend": "faster_whisper"}),
    ({"wake triggers", "wake triggers list", "list wake triggers", "wake status", "wake mode status"}, "VOICE_COMMAND", "wake_status"),
    ({"stop speaking", "interrupt speech", "be quiet", "stop talking"}, "VOICE_COMMAND", "interrupt"),
    ({"speech on", "enable speech"}, "VOICE_COMMAND", "speech_on"),
    ({"speech off", "disable speech"}, "VOICE_COMMAND", "speech_off"),
    # Knowledge base
    ({"kb status", "knowledge status", "knowledge base status"}, "KNOWLEDGE_BASE_COMMAND", "status"),
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
    ({"language arabic", "set language arabic", "language ar", "set language ar", "اللغة العربية", "اللغة عربي"}, "MEMORY_COMMAND", "set_language", {"language": "ar"}),
    ({"language english", "set language english", "language en", "set language en", "اللغة الانجليزية", "اللغة الإنجليزية", "اللغة انجليزي"}, "MEMORY_COMMAND", "set_language", {"language": "en"}),
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
    # Rollback
    (
        {
            "undo",
            "rollback",
            "undo last action",
            "\u062a\u0631\u0627\u062c\u0639",
            "\u0627\u0644\u063a\u0627\u0621 \u0627\u062e\u0631 \u0639\u0645\u0644\u064a\u0629",
            "\u0627\u0644\u063a\u0627\u0621 \u0627\u062e\u0631 \u0627\u0645\u0631",
        },
        "OS_ROLLBACK",
        "",
    ),
    # File nav
    (
        {
            "current directory",
            "pwd",
            "\u0627\u0644\u0645\u062c\u0644\u062f \u0627\u0644\u062d\u0627\u0644\u064a",
            "\u0627\u064a\u0646 \u0627\u0646\u0627",
        },
        "OS_FILE_NAVIGATION",
        "pwd",
    ),
    (
        {
            "list drives",
            "drive list",
            "\u0627\u0639\u0631\u0636 \u0627\u0644\u0627\u0642\u0631\u0627\u0635",
            "\u0627\u0638\u0647\u0631 \u0627\u0644\u0627\u0642\u0631\u0627\u0635",
            "\u0642\u0627\u0626\u0645\u0629 \u0627\u0644\u0627\u0642\u0631\u0627\u0635",
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
# Table-driven regex matching
# ---------------------------------------------------------------------------
# Each entry: (compiled_regex, use_raw, intent, action, args_builder)
# If use_raw is True, the regex is matched against `raw` (case-insensitive).
# Otherwise it's matched against `normalized`.
# args_builder is a callable: (match) -> dict

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
        re.compile(r"^(?:set\s+)?(?:voice\s+)?(?:stt|speech)\s+profile(?:\s+to)?\s+(quiet|noisy)(?:\s+room)?$"),
        False,
        "VOICE_COMMAND",
        "stt_profile_set",
        lambda m: {"profile": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:(?:voice|speech|stt)\s+)?(?:stt|speech)\s+backend(?:\s+to)?\s+(faster(?:[_\s-]?whisper)?|whisper)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "stt_backend_set",
        lambda m: {"backend": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:(?:voice|speech|stt)\s+)?(?:stt|speech)\s+backend\s+status$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "stt_backend_status",
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
            r"^(?:اضبط|حدد|غير|غيّر|اجعل)\s+(?:جودة|وضع)\s+(?:الصوت|النطق)(?:\s+(?:الى|إلى))?\s+(طبيعي|قياسي|افتراضي|روبوت|روبوتي)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "voice_quality_set",
        lambda m: {"mode": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:audio|voice)\s+(?:ux\s+)?profile(?:\s+to)?\s+(balanced|responsive|robust|fast|low\s*latency|low_latency|stable|reliable|noisy)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_profile_set",
        lambda m: {"profile": m.group(1).replace(" ", "_")},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:latency|performance|speed)\s+mode(?:\s+to)?\s+(fast|balanced|normal|stable|robust|reliable|low\s*latency)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_profile_set",
        lambda m: {"profile": m.group(1).replace(" ", "_")},
    ),
    (
        re.compile(
            r"^(?:latency|pipeline\s+latency|phase\s+latency|runtime\s+latency|performance)\s+status$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "latency_status",
        lambda _m: {},
    ),
    (
        re.compile(
            r"^(?:اضبط|حدد|غير|غيّر|اجعل)\s+(?:ملف|وضع)\s+(?:تجربة\s+)?(?:الصوت|النطق)(?:\s+(?:الى|إلى))?\s+(متوازن|سريع(?:\s*الاستجابة)?|منخفض\s*الكمون|قوي|ثابت|موثوق)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_profile_set",
        lambda m: {"profile": m.group(1).replace(" ", "_")},
    ),
    (
        re.compile(
            r"^(?:اضبط|حدد|غير|غيّر|اجعل)\s+(?:وضع|نمط)\s+(?:السرعة|الكمون|الاستجابة)(?:\s+(?:الى|إلى))?\s+(سريع|متوازن|طبيعي|ثابت|موثوق|منخفض\s*الكمون)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_profile_set",
        lambda m: {"profile": m.group(1).replace(" ", "_")},
    ),
    (
        re.compile(
            r"^(?:حالة\s+)?(?:الكمون|الاستجابة|التأخير)\s*$",
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
            r"^(?:اضبط|حدد|غير|غيّر|بدل|بدّل|حول|حوّل|خلي|خلّي|اجعل)?\s*(?:اللغة)(?:\s+(?:الى|إلى|ل))?\s*(العربية|عربي|الانجليزية|الإنجليزية|انجليزي|ar|en)(?:\s*[.!?؟،]+)?$",
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
            r"^(?:find file|search file|\u0627\u0628\u062d\u062b \u0639\u0646 \u0645\u0644\u0641|\u0627\u0628\u062d\u062b \u0645\u0644\u0641|\u062f\u0648\u0631 \u0639\u0644\u0649 \u0645\u0644\u0641)\s+(.+?)(?:\s+(?:in|\u0641\u064a)\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_SEARCH",
        "",
        lambda m: {"filename": m.group(1).strip(), "search_path": (m.group(2) or "").strip() or None},
    ),
    # File nav - regex-based
    (
        re.compile(
            r"^(?:list files|list directory|show files|show directory|\u0627\u0639\u0631\u0636 \u0627\u0644\u0645\u0644\u0641\u0627\u062a|\u0627\u0638\u0647\u0631 \u0627\u0644\u0645\u0644\u0641\u0627\u062a|\u0627\u0639\u0631\u0636 \u0627\u0644\u0645\u062c\u0644\u062f|\u0627\u0638\u0647\u0631 \u0627\u0644\u0645\u062c\u0644\u062f)(?:\s+(?:in|\u0641\u064a)\s+(.+))?$",
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
            r"^(?:file info|metadata|\u0645\u0639\u0644\u0648\u0645\u0627\u062a \u0645\u0644\u0641|\u0628\u064a\u0627\u0646\u0627\u062a \u0645\u0644\u0641)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "file_info",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:create folder|make folder|mkdir|\u0627\u0646\u0634\u0626 \u0645\u062c\u0644\u062f|\u0627\u0639\u0645\u0644 \u0645\u062c\u0644\u062f|\u0627\u0635\u0646\u0639 \u0645\u062c\u0644\u062f)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "create_directory",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:delete permanently|permanent delete|force delete|\u0627\u062d\u0630\u0641 \u0646\u0647\u0627\u0626\u064a\u0627|\u062d\u0630\u0641 \u0646\u0647\u0627\u0626\u064a)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "delete_item_permanent",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(r"^(?:delete|remove|\u0627\u062d\u0630\u0641|\u0627\u0645\u0633\u062d|\u0627\u0632\u0644)\s+(.+)$", re.IGNORECASE),
        True,
        "OS_FILE_NAVIGATION",
        "delete_item",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:move|\u0627\u0646\u0642\u0644|\u062d\u0631\u0643)\s+(.+?)\s+(?:to|\u0627\u0644\u0649|\u0625\u0644\u0649)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "move_item",
        lambda m: {"source": m.group(1).strip(), "destination": m.group(2).strip()},
    ),
    (
        re.compile(
            r"^(?:rename|\u0627\u0639\u062f \u062a\u0633\u0645\u064a\u0629|\u063a\u064a\u0631 \u0627\u0633\u0645|\u063a\u064a\u0651\u0631 \u0627\u0633\u0645)\s+(.+?)\s+(?:to|\u0627\u0644\u0649|\u0625\u0644\u0649)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "rename_item",
        lambda m: {"source": m.group(1).strip(), "new_name": m.group(2).strip()},
    ),
    # Open app explicit
    (
        re.compile(r"^(?:open app|\u0627\u0641\u062a\u062d \u062a\u0637\u0628\u064a\u0642|\u0634\u063a\u0644 \u062a\u0637\u0628\u064a\u0642)\s+(.+)$", re.IGNORECASE),
        True,
        "OS_APP_OPEN",
        "",
        lambda m: {"app_name": m.group(1).strip()},
    ),
    # Close app explicit
    (
        re.compile(
            r"^(?:close app|\u0627\u063a\u0644\u0642 \u062a\u0637\u0628\u064a\u0642|\u0627\u0642\u0641\u0644 \u062a\u0637\u0628\u064a\u0642|\u0633\u0643\u0631 \u062a\u0637\u0628\u064a\u0642|\u0627\u0646\u0647\u064a \u062a\u0637\u0628\u064a\u0642)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_APP_CLOSE",
        "",
        lambda m: {"app_name": m.group(1).strip()},
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
        r"^(?:open|launch|start|\u0627\u0641\u062a\u062d|\u0634\u063a\u0644)\s+(.+)$",
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
            r"^(?:close|terminate|kill|quit|exit|\u0627\u063a\u0644\u0642|\u0627\u0642\u0641\u0644|\u0633\u0643\u0631|\u0627\u0646\u0647\u064a)\s+"
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
    web_markers = (
        "online",
        "web",
        "internet",
        "on google",
        "search web",
        "search online",
        "google",
        "الويب",
        "النت",
        "الانترنت",
        "الإنترنت",
        "جوجل",
        "اونلاين",
        "أونلاين",
        "بالنت",
    )
    if any(marker in lowered for marker in web_markers):
        return None

    patterns = (
        re.compile(
            r"^(?:find|search|look\s+for|locate)\s+(?:for\s+)?(?:file\s+)?(.+?)(?:\s+(?:in|on|inside)\s+(.+))?$",
            re.IGNORECASE,
        ),
        re.compile(
            (
                r"^(?:(?:i\s+)?(?:want|need)\s+(?:to\s+)?)"
                r"(?:find|search|look\s+for|locate)\s+(?:for\s+)?(?:file\s+)?"
                r"(.+?)(?:\s+(?:in|on|inside)\s+(.+))?$"
            ),
            re.IGNORECASE,
        ),
        re.compile(
            (
                r"^(?:(?:\u0627\u0631\u064a\u062f|\u0623\u0631\u064a\u062f|\u0639\u0627\u064a\u0632|\u0627\u0628\u063a\u0649|\u0623\u0628\u063a\u0649)\s+(?:\u0627\u0646|\u0623\u0646)?\s+)?"
                r"(?:\u0627\u062c\u062f|\u0623\u062c\u062f|\u0627\u062f\u0648\u0631|\u0623\u062f\u0648\u0631|\u0627\u0628\u062d\u062b|\u0623\u0628\u062d\u062b)(?:\s+\u0639\u0646)?\s+(?:\u0645\u0644\u0641\s+)?"
                r"(.+?)(?:\s+(?:\u0641\u064a|\u062f\u0627\u062e\u0644)\s+(.+))?$"
            ),
            re.IGNORECASE,
        ),
    )

    for pattern in patterns:
        match = pattern.match(raw)
        if not match:
            continue

        filename = _collapse_repeated_phrase(match.group(1) or "")
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

        target_text = (match.group(1) or "").strip()
        app_name = _infer_known_app_name(target_text)
        if app_name:
            return ParsedCommand("OS_APP_OPEN", raw, normalized, args={"app_name": app_name})
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
    lowered = _normalize_for_match(raw)

    if re.search(r"\b(new tab|open tab)\b", lowered) or "تبويب جديد" in lowered:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "browser_new_tab"})
    if re.search(r"\b(close tab)\b", lowered) or "اغلق التبويب" in lowered:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "browser_close_tab"})
    if re.search(r"\b(go back|browser back)\b", lowered) or "ارجع للخلف" in lowered:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "browser_back"})
    if re.search(r"\b(go forward|browser forward)\b", lowered) or "اذهب للامام" in lowered:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "browser_forward"})

    search_patterns = (
        re.compile(
            r"(?:^|\b)(?:search(?:\s+(?:the\s+)?)?(?:(?:web|online|internet)\s*(?:for|about)?|(?:for|about))|google|look\s+up)\s+(.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:^|\b)(?:ابحث(?:\s+(?:في\s+)?)?(?:(?:الويب|النت|الانترنت|الإنترنت|جوجل|اونلاين|أونلاين)\s*(?:عن)?|عن)|دور(?:\s+على)?(?:\s+(?:النت|الانترنت|الإنترنت|اونلاين|أونلاين))?)\s+(.+)$",
            re.IGNORECASE,
        ),
    )
    for pattern in search_patterns:
        match = pattern.search(raw)
        if match and match.group(1).strip():
            query = _clean_browser_search_query(match.group(1))
            if not query:
                continue
            return ParsedCommand(
                "OS_SYSTEM_COMMAND",
                raw,
                normalized,
                args={"action_key": "browser_search_web", "search_query": query},
            )

    open_patterns = (
        re.compile(r"^(?:open|visit|go to|browse to)\s+(?:website|site|url\s+)?(.+)$", re.IGNORECASE),
        re.compile(r"^(?:افتح|روح على|اذهب الى)\s+(?:موقع\s+)?(.+)$", re.IGNORECASE),
    )
    for pattern in open_patterns:
        match = pattern.match(raw)
        if not match:
            continue
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
    lowered = _normalize_for_match(raw)
    if re.search(r"\b(maximize)\b", lowered) and "window" in lowered:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "window_maximize"})
    if re.search(r"\b(minimize)\b", lowered) and "window" in lowered:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "window_minimize"})
    if "snap" in lowered and "left" in lowered:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "window_snap_left"})
    if "snap" in lowered and "right" in lowered:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "window_snap_right"})
    if re.search(r"\b(next window|switch window)\b", lowered) or "النافذة التالية" in lowered:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "window_next"})
    if re.search(r"\b(close (?:active|this) window)\b", lowered) or "اغلق النافذة" in lowered:
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "window_close_active"})

    focus_patterns = (
        re.compile(r"^(?:focus|switch to|bring)\s+(?:the\s+)?(?:window\s+)?(.+)$", re.IGNORECASE),
        re.compile(r"^(?:ركز على|روح على)\s+(?:نافذة\s+)?(.+)$", re.IGNORECASE),
    )
    for pattern in focus_patterns:
        match = pattern.match(raw)
        if not match:
            continue
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
    lowered = _normalize_for_match(raw)
    media_context = any(token in lowered for token in ("music", "media", "track", "song", "موسيقى", "اغنية"))

    if re.search(r"\b(pause|play|resume)\b", lowered) and any(token in lowered for token in ("music", "media", "track", "song")):
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "media_play_pause"})
    if any(token in lowered for token in ("next track", "next song", "skip track", "skip song", "الاغنية التالية")):
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "media_next_track"})
    if any(token in lowered for token in ("previous track", "prev track", "previous song", "الاغنية السابقة")):
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "media_previous_track"})
    if any(token in lowered for token in ("stop music", "stop media", "اوقف التشغيل")):
        return ParsedCommand("OS_SYSTEM_COMMAND", raw, normalized, args={"action_key": "media_stop"})

    forward = re.search(r"(?:seek|skip|forward|قدم)\s+(?:by\s+)?(.+?)?\s*(seconds?|secs?|ثانية|ثواني)?$", raw, flags=re.IGNORECASE)
    if forward and media_context and ("forward" in lowered or "seek" in lowered or "قدم" in lowered):
        seconds = _duration_to_seconds(forward.group(1) or 10, forward.group(2) or "seconds") or 10
        return ParsedCommand(
            "OS_SYSTEM_COMMAND",
            raw,
            normalized,
            args={"action_key": "media_seek_forward", "seek_seconds": int(seconds)},
        )

    backward = re.search(r"(?:seek|skip|back|rewind|ارجع)\s+(?:by\s+)?(.+?)?\s*(seconds?|secs?|ثانية|ثواني)?$", raw, flags=re.IGNORECASE)
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
    create_patterns = (
        re.compile(
            r"^(?:create|make)\s+(?:a\s+)?(?:new\s+)?folder(?:\s+(?:called|named))?\s+(.+?)(?:\s+(?:in|inside|under)\s+(.+))?$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:انشئ|اعمل|اصنع)\s+(?:مجلد\s+)?(?:باسم\s+)?(.+?)(?:\s+(?:في|داخل)\s+(.+))?$",
            re.IGNORECASE,
        ),
    )
    for pattern in create_patterns:
        match = pattern.match(raw)
        if not match:
            continue
        name = str(match.group(1) or "").strip()
        parent = _normalize_search_path_hint(match.group(2) or "")
        if not name:
            continue
        path = os.path.join(parent, name) if parent else name
        return ParsedCommand("OS_FILE_NAVIGATION", raw, normalized, action="create_directory", args={"path": path})

    move_patterns = (
        re.compile(
            r"^(?:move|put)\s+(?:the\s+)?(?:file|folder)?\s*(.+?)\s+(?:to|into|inside|under)\s+(.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:انقل|حرك)\s+(?:الملف|المجلد)?\s*(.+?)\s+(?:الى|إلى|داخل)\s+(.+)$",
            re.IGNORECASE,
        ),
    )
    for pattern in move_patterns:
        match = pattern.match(raw)
        if match and match.group(1).strip() and match.group(2).strip():
            return ParsedCommand(
                "OS_FILE_NAVIGATION",
                raw,
                normalized,
                action="move_item",
                args={"source": match.group(1).strip(), "destination": match.group(2).strip()},
            )

    rename_patterns = (
        re.compile(r"^(?:rename|change name of)\s+(.+?)\s+(?:to|as)\s+(.+)$", re.IGNORECASE),
        re.compile(r"^(?:اعد تسمية|غير اسم|غيّر اسم)\s+(.+?)\s+(?:الى|إلى)\s+(.+)$", re.IGNORECASE),
    )
    for pattern in rename_patterns:
        match = pattern.match(raw)
        if match and match.group(1).strip() and match.group(2).strip():
            return ParsedCommand(
                "OS_FILE_NAVIGATION",
                raw,
                normalized,
                action="rename_item",
                args={"source": match.group(1).strip(), "new_name": match.group(2).strip()},
            )

    delete_patterns = (
        re.compile(r"^(?:delete|remove)\s+(?:the\s+)?(?:file|folder)?\s*(.+?)(?:\s+(permanently|forever))?$", re.IGNORECASE),
        re.compile(r"^(?:احذف|امسح|ازل)\s+(?:الملف|المجلد)?\s*(.+?)(?:\s+(نهائيا|نهائي))?$", re.IGNORECASE),
    )
    for pattern in delete_patterns:
        match = pattern.match(raw)
        if not match:
            continue
        target = str(match.group(1) or "").strip()
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
        r"^(?:\u0627\u0630\u0647\u0628|\u0631\u0648\u062d|\u0627\u0646\u062a\u0642\u0644)\s+(?:\u0627\u0644\u0649|\u0625\u0644\u0649)\s+(.+)$",
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


