import json
import re
import time
from collections import OrderedDict

from core.command_parser import parse_command
from core.config import (
    NLU_INTENT_CACHE_ENABLED,
    NLU_INTENT_CACHE_MAX_SIZE,
    NLU_INTENT_CACHE_TTL_SECONDS,
)
from core.intent_confidence import assess_intent_confidence
from llm.ollama_client import ask_llm
from llm.prompt_builder import build_intent_extraction_prompt


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_ALLOWED_INTENTS = {
    "OS_APP_OPEN",
    "OS_APP_CLOSE",
    "OS_FILE_SEARCH",
    "OS_FILE_NAVIGATION",
    "OS_SYSTEM_COMMAND",
    "JOB_QUEUE_COMMAND",
    "VOICE_COMMAND",
    "LLM_QUERY",
}
_ALLOWED_NAV_ACTIONS = {
    "",
    "list_directory",
    "cd",
    "file_info",
    "create_directory",
    "delete_item",
    "move_item",
    "rename_item",
}
_ALLOWED_VOICE_ACTIONS = {"interrupt", "speech_on", "speech_off", "status"}
_ALLOWED_JOB_ACTIONS = {"", "enqueue", "status", "cancel", "retry", "list"}
_ALLOWED_SYSTEM_ACTION_KEYS = {
    "shutdown",
    "restart",
    "sleep",
    "lock",
    "logoff",
    "volume_up",
    "volume_down",
    "volume_mute",
    "volume_set",
    "brightness_up",
    "brightness_down",
    "brightness_set",
    "wifi_on",
    "wifi_off",
    "bluetooth_on",
    "bluetooth_off",
    "notifications_on",
    "notifications_off",
    "screenshot",
    "empty_recycle_bin",
    "list_processes",
    "focus_window",
    "window_maximize",
    "window_minimize",
    "window_snap_left",
    "window_snap_right",
    "window_next",
    "window_close_active",
    "media_play_pause",
    "media_next_track",
    "media_previous_track",
    "media_stop",
    "media_seek_forward",
    "media_seek_backward",
    "browser_new_tab",
    "browser_close_tab",
    "browser_back",
    "browser_forward",
    "browser_open_url",
    "browser_search_web",
}
_WINDOW_QUERY_ALIASES = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "chrome window": "chrome",
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
_URL_RE = re.compile(r"^(?:https?://|www\.)[^\s]+$", flags=re.IGNORECASE)
_DOMAIN_RE = re.compile(r"^[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?$", flags=re.IGNORECASE)
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
_MEDIA_APP_ALIASES = {
    "spotify": "spotify",
    "سبوتيفاي": "spotify",
    "vlc": "vlc",
    "youtube music": "youtube music",
    "yt music": "youtube music",
    "يوتيوب ميوزك": "youtube music",
}
_APP_ALIASES = {
    "calculator": "calculator",
    "calc": "calculator",
    "الحاسبة": "calculator",
    "notepad": "notepad",
    "text editor": "notepad",
    "المفكرة": "notepad",
    "chrome": "chrome",
    "google chrome": "chrome",
    "كروم": "chrome",
    "spotify": "spotify",
    "سبوتيفاي": "spotify",
    "firefox": "firefox",
    "mozilla firefox": "firefox",
    "فايرفوكس": "firefox",
    "فاير فوكس": "firefox",
    "vlc": "vlc",
    "youtube music": "youtube music",
    "يوتيوب ميوزك": "youtube music",
}
_VOICE_SPEECH_ON_PATTERNS = (
    re.compile(r"\b(?:speech|voice)\s+on\b", re.IGNORECASE),
    re.compile(r"\b(?:enable|unmute|activate)\s+(?:speech|voice)\b", re.IGNORECASE),
    re.compile(r"\bturn\s+(?:speech|voice)\s+on\b", re.IGNORECASE),
    re.compile(
        r"(?:\u0634\u063a\u0644|\u0641\u0639\u0644|\u0641\u0639\u0651\u0644|\u0627\u0641\u062a\u062d)\s+(?:\u0627\u0644\u0635\u0648\u062a|\u0627\u0644\u0646\u0637\u0642|\u0627\u0644\u0643\u0644\u0627\u0645)",
        re.IGNORECASE,
    ),
)
_VOICE_SPEECH_OFF_PATTERNS = (
    re.compile(r"\b(?:speech|voice)\s+off\b", re.IGNORECASE),
    re.compile(r"\b(?:disable|mute|deactivate)\s+(?:speech|voice)\b", re.IGNORECASE),
    re.compile(r"\bturn\s+(?:speech|voice)\s+off\b", re.IGNORECASE),
    re.compile(r"\b(?:be quiet|stop talking|stop speaking)\b", re.IGNORECASE),
    re.compile(
        r"(?:\u0627\u0637\u0641\u064a|\u0627\u0637\u0641\u0626|\u0627\u0642\u0641\u0644|\u0627\u0643\u062a\u0645|\u0648\u0642\u0641|\u0623\u0648\u0642\u0641)\s+(?:\u0627\u0644\u0635\u0648\u062a|\u0627\u0644\u0646\u0637\u0642|\u0627\u0644\u0643\u0644\u0627\u0645)",
        re.IGNORECASE,
    ),
)
_SYSTEM_ACTION_MARKERS = {
    "open",
    "launch",
    "start",
    "close",
    "shutdown",
    "restart",
    "lock",
    "mute",
    "unmute",
    "volume",
    "brightness",
    "wifi",
    "bluetooth",
    "notification",
    "notifications",
    "search",
    "google",
    "browser",
    "tab",
    "window",
    "media",
    "music",
    "track",
    "play",
    "pause",
    "افتح",
    "شغل",
    "اغلق",
    "اقفل",
    "اطفي",
    "اطفئ",
    "اقفل",
    "ارفع",
    "اخفض",
    "خفض",
    "زود",
    "قلل",
    "اضبط",
    "حدد",
    "فعل",
    "فعّل",
    "عطل",
    "اكتم",
    "ابحث",
    "دور",
    "جوجل",
    "الصوت",
    "السطوع",
    "النافذة",
    "موسيقى",
    "اغنية",
    "أغنية",
    "ويب",
    "موقع",
    "تبويب",
}
_INFORMATIONAL_QUERY_MARKERS = {
    "tell me",
    "explain",
    "what",
    "why",
    "how",
    "who",
    "where",
    "when",
    "about",
    "اخبرني",
    "أخبرني",
    "اخبر",
    "أخبر",
    "خبرني",
    "حدثني",
    "اشرح",
    "اشرحلي",
    "اشرح لي",
    "ماذا",
    "ما",
    "كيف",
    "لماذا",
    "ليش",
    "من",
    "اين",
    "أين",
    "متى",
    "عن",
    "اخبار",
    "أخبار",
}
_ACTION_KEY_REQUIRED_MARKERS = {
    "volume_set": ("volume", "sound", "audio", "الصوت", "صوت"),
    "volume_up": ("volume", "sound", "audio", "الصوت", "صوت", "turn it up", "ارفع"),
    "volume_down": ("volume", "sound", "audio", "الصوت", "صوت", "turn it down", "اخفض", "خفض"),
    "volume_mute": ("volume", "sound", "audio", "mute", "الصوت", "صوت", "اكتم"),
    "brightness_set": ("brightness", "screen", "display", "السطوع", "الشاشة"),
    "brightness_up": ("brightness", "screen", "display", "السطوع", "الشاشة"),
    "brightness_down": ("brightness", "screen", "display", "السطوع", "الشاشة"),
    "browser_search_web": ("search", "google", "look up", "ابحث", "دور", "جوجل"),
    "browser_open_url": ("http", "www", ".com", ".net", ".org", "site", "website", "موقع", "رابط"),
    "focus_window": ("focus", "switch", "bring", "window", "ركز", "روح", "نافذة"),
    "wifi_on": ("wifi", "wi fi", "wireless", "واي فاي", "الانترنت"),
    "wifi_off": ("wifi", "wi fi", "wireless", "واي فاي", "الانترنت"),
    "bluetooth_on": ("bluetooth", "بلوتوث"),
    "bluetooth_off": ("bluetooth", "بلوتوث"),
    "notifications_on": (
        "notification",
        "notifications",
        "notify",
        "dnd",
        "do not disturb",
        "focus assist",
        "اشعارات",
        "إشعارات",
        "الاشعارات",
        "الإشعارات",
        "عدم الازعاج",
        "عدم الإزعاج",
    ),
    "notifications_off": (
        "notification",
        "notifications",
        "notify",
        "dnd",
        "do not disturb",
        "focus assist",
        "اشعارات",
        "إشعارات",
        "الاشعارات",
        "الإشعارات",
        "عدم الازعاج",
        "عدم الإزعاج",
    ),
    "shutdown": (
        "shutdown",
        "shut down",
        "turn off",
        "power off",
        "turn off pc",
        "turn off the pc",
        "اطفي",
        "اطفئ",
        "اغلق",
        "اقفل",
        "ايقاف تشغيل",
        "إيقاف تشغيل",
    ),
    "restart": ("restart", "reboot", "اعادة تشغيل", "إعادة تشغيل", "ريستارت"),
    "sleep": ("sleep", "suspend", "sleep mode", "سكون", "وضع السكون"),
    "lock": ("lock", "lock screen", "قفل", "اقفل", "قفل الشاشة", "اقفل الشاشة"),
    "logoff": ("log off", "logout", "sign out", "تسجيل الخروج", "سجل خروج", "اخرج"),
    "empty_recycle_bin": (
        "recycle bin",
        "trash",
        "empty recycle",
        "empty bin",
        "سلة المحذوفات",
        "سلة المهملات",
        "افرغ",
        "إفراغ",
        "افراغ",
    ),
}
_NLU_INTENT_CACHE = OrderedDict()
_NLU_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "stores": 0,
    "evictions": 0,
}


def _cache_key(text: str, language: str):
    return (_norm_text(language or "en"), _norm_text(text))


def _clone_nlu_result(payload: dict):
    copied = dict(payload or {})
    copied["args"] = dict(copied.get("args") or {})
    return copied


def _cache_get_nlu_result(text: str, language: str):
    if not NLU_INTENT_CACHE_ENABLED:
        return None

    now = time.time()
    key = _cache_key(text, language)
    entry = _NLU_INTENT_CACHE.get(key)
    if not entry:
        _NLU_CACHE_STATS["misses"] += 1
        return None

    cached_at = float(entry.get("cached_at") or 0.0)
    if cached_at <= 0 or (now - cached_at) > max(1, int(NLU_INTENT_CACHE_TTL_SECONDS)):
        _NLU_INTENT_CACHE.pop(key, None)
        _NLU_CACHE_STATS["misses"] += 1
        _NLU_CACHE_STATS["evictions"] += 1
        return None

    _NLU_INTENT_CACHE.move_to_end(key)
    _NLU_CACHE_STATS["hits"] += 1
    value = _clone_nlu_result(entry.get("value") or {})
    value["cache_hit"] = True
    return value


def _cache_put_nlu_result(text: str, language: str, payload: dict):
    if not NLU_INTENT_CACHE_ENABLED:
        return

    key = _cache_key(text, language)
    _NLU_INTENT_CACHE[key] = {
        "cached_at": time.time(),
        "value": _clone_nlu_result(payload),
    }
    _NLU_INTENT_CACHE.move_to_end(key)
    _NLU_CACHE_STATS["stores"] += 1

    max_size = max(16, int(NLU_INTENT_CACHE_MAX_SIZE or 256))
    while len(_NLU_INTENT_CACHE) > max_size:
        _NLU_INTENT_CACHE.popitem(last=False)
        _NLU_CACHE_STATS["evictions"] += 1


def clear_nlu_cache():
    _NLU_INTENT_CACHE.clear()
    _NLU_CACHE_STATS.update({"hits": 0, "misses": 0, "stores": 0, "evictions": 0})


def get_nlu_cache_stats():
    return {
        "enabled": bool(NLU_INTENT_CACHE_ENABLED),
        "size": len(_NLU_INTENT_CACHE),
        "hits": int(_NLU_CACHE_STATS["hits"]),
        "misses": int(_NLU_CACHE_STATS["misses"]),
        "stores": int(_NLU_CACHE_STATS["stores"]),
        "evictions": int(_NLU_CACHE_STATS["evictions"]),
        "ttl_seconds": int(NLU_INTENT_CACHE_TTL_SECONDS),
        "max_size": int(NLU_INTENT_CACHE_MAX_SIZE),
    }


def _parse_spoken_int(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(float(value))

    text = _norm_text(value)
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


def _coerce_percent_int(value):
    if value is None:
        return None
    numeric = _parse_spoken_int(value)
    if numeric is None:
        return None
    return max(0, min(100, numeric))


def _extract_first_number(text: str):
    parsed = _parse_spoken_int(text)
    if parsed is None:
        return None
    return _coerce_percent_int(parsed)


def _parse_duration_seconds(value, unit_hint="seconds"):
    number = _parse_spoken_int(value)
    if number is None:
        return None
    unit = _norm_text(unit_hint)
    factor = _DURATION_UNIT_SECONDS.get(unit, 1)
    return max(1, min(3600, int(number * factor)))


def _normalize_url(value):
    candidate = str(value or "").strip().strip('"').strip("'")
    candidate = re.sub(
        r"^(?:open|visit|go to|browse to|website|site|url|افتح|روح على|اذهب الى|موقع|رابط)\s+",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    if not candidate:
        return ""
    if _URL_RE.match(candidate):
        return f"https://{candidate}" if candidate.lower().startswith("www.") else candidate
    if _DOMAIN_RE.match(candidate):
        return f"https://{candidate}"
    return ""


def _clean_web_search_query(value):
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


def _extract_search_query(source_text: str):
    text = str(source_text or "").strip()
    patterns = (
        r"(?:^|\b)(?:search(?:\s+(?:the\s+)?)?(?:(?:web|online|internet)\s*(?:for|about)?|(?:for|about))|google|look\s+up)\s+(.+)$",
        r"(?:^|\b)(?:ابحث(?:\s+(?:في\s+)?)?(?:(?:الويب|النت|الانترنت|الإنترنت|جوجل|اونلاين|أونلاين)\s*(?:عن)?|عن)|دور(?:\s+على)?(?:\s+(?:النت|الانترنت|الإنترنت|اونلاين|أونلاين))?)\s+(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            query = _clean_web_search_query(match.group(1))
            if query:
                return query
    return ""


def _extract_schedule_payload(source_text: str):
    text = str(source_text or "").strip()
    patterns = (
        re.compile(
            r"^(?:in|after)\s+(.+?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+(.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^remind\s+me\s+in\s+(.+?)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+to\s+(.+)$",
            re.IGNORECASE,
        ),
        re.compile(r"^بعد\s+(.+?)\s*(ثانية|ثواني|دقيقة|دقائق|ساعة|ساعات)\s+(.+)$", re.IGNORECASE),
    )
    for pattern in patterns:
        match = pattern.match(text)
        if not match:
            continue
        delay_seconds = _parse_duration_seconds(match.group(1), match.group(2))
        command_text = str(match.group(3) or "").strip()
        command_text = re.sub(r"^(?:to\s+|أن\s+|ان\s+)", "", command_text, flags=re.IGNORECASE).strip()
        if delay_seconds is None or not command_text:
            continue
        return {
            "delay_seconds": int(max(0, delay_seconds)),
            "command_text": command_text,
        }
    return None


def _infer_media_app_target(source_text: str, args: dict):
    candidate = str(
        args.get("app_name")
        or args.get("target")
        or args.get("app")
        or source_text
        or ""
    ).strip()
    normalized = _norm_text(candidate)
    for alias, app_name in sorted(_MEDIA_APP_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if (
            normalized == alias
            or normalized.startswith(alias + " ")
            or normalized.endswith(" " + alias)
            or (" " + alias + " ") in (" " + normalized + " ")
        ):
            return app_name
    return ""


def _infer_app_target(source_text: str, args: dict):
    candidate = str(
        args.get("app_name")
        or args.get("target")
        or args.get("app")
        or args.get("name")
        or source_text
        or ""
    ).strip()
    normalized = _norm_text(candidate)
    for alias, app_name in sorted(_APP_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if (
            normalized == alias
            or normalized.startswith(alias + " ")
            or normalized.endswith(" " + alias)
            or (" " + alias + " ") in (" " + normalized + " ")
        ):
            return app_name
    return ""


def _extract_focus_query(text: str):
    value = str(text or "").strip()
    if not value:
        return ""

    patterns = [
        r"(?:focus|switch\s+to|bring)\s+(?:window\s+)?(.+)$",
        r"(?:go\s+to|open)\s+(.+)$",
        r"(?:\u0631\u0648\u062d\s+\u0639\u0644\u0649|\u0631\u0643\u0632\s+\u0639\u0644\u0649)\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).strip().strip(".\"'")
            candidate = re.sub(r"^(?:the\s+)?(?:window\s+)?", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"^(?:\u0646\u0627\u0641\u0630\u0629\s+)", "", candidate, flags=re.IGNORECASE)
            candidate = re.sub(r"(?:\s+window)$", "", candidate, flags=re.IGNORECASE)
            return candidate.strip()
    return ""


def _canonicalize_window_query(value: str):
    query = str(value or "").strip()
    if not query:
        return ""

    normalized = _norm_text(query)
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
    return query


def _infer_system_action_key(source_text: str, args: dict, action_key_hint: str = ""):
    hint = str(action_key_hint or "").strip().lower()
    arg_hint = str(args.get("action_key") or "").strip().lower()
    fallback_hint = ""
    if hint in _ALLOWED_SYSTEM_ACTION_KEYS:
        fallback_hint = hint
    elif arg_hint in _ALLOWED_SYSTEM_ACTION_KEYS:
        fallback_hint = arg_hint

    text = _norm_text(source_text)

    has_volume = any(token in text for token in ("volume", "sound", "الصوت"))
    has_brightness = any(token in text for token in ("brightness", "screen", "السطوع"))
    has_focus = any(
        token in text
        for token in (
            "focus",
            "switch to",
            "bring",
            "ركز",
            "روح على",
        )
    )
    has_window = any(token in text for token in ("window", "نافذة"))
    has_media = any(token in text for token in ("media", "music", "song", "track", "موسيقى", "اغنية"))
    has_browser = any(token in text for token in ("browser", "tab", "website", "site", "url", "ويب", "موقع", "تبويب"))
    has_media_app = any(token in text for token in ("spotify", "vlc", "youtube music", "سبوتيفاي", "يوتيوب ميوزك"))
    has_number = _extract_first_number(text) is not None

    if any(token in text for token in ("screenshot", "screen shot", "take a screenshot", "لقطة شاشة", "صورة للشاشة", "سكرينشوت")):
        return "screenshot"

    if has_volume and has_number:
        return "volume_set"
    if has_brightness and has_number:
        return "brightness_set"

    if has_window and any(token in text for token in ("maximize", "كبر")):
        return "window_maximize"
    if has_window and any(token in text for token in ("minimize", "صغر")):
        return "window_minimize"
    if has_window and "left" in text and "snap" in text:
        return "window_snap_left"
    if has_window and "right" in text and "snap" in text:
        return "window_snap_right"
    if any(token in text for token in ("next window", "switch window", "النافذة التالية")):
        return "window_next"
    if any(token in text for token in ("close active window", "close this window", "اغلق النافذة")):
        return "window_close_active"
    if has_focus:
        return "focus_window"

    if has_media and not has_media_app and any(token in text for token in ("pause", "resume", "play", "اوقف", "شغل")):
        return "media_play_pause"
    if any(token in text for token in ("next track", "next song", "skip song", "الاغنية التالية")):
        return "media_next_track"
    if any(token in text for token in ("previous track", "prev track", "previous song", "الاغنية السابقة")):
        return "media_previous_track"
    if has_media and any(token in text for token in ("stop", "اوقف التشغيل")):
        return "media_stop"
    if any(token in text for token in ("seek forward", "skip forward", "forward", "قدم")) and has_media:
        return "media_seek_forward"
    if any(token in text for token in ("seek back", "seek backward", "rewind", "backward", "ارجع")) and has_media:
        return "media_seek_backward"

    if has_browser and any(token in text for token in ("new tab", "open tab", "تبويب جديد")):
        return "browser_new_tab"
    if has_browser and any(token in text for token in ("close tab", "اغلق التبويب")):
        return "browser_close_tab"
    if any(token in text for token in ("go back", "browser back", "ارجع للخلف")):
        return "browser_back"
    if any(token in text for token in ("go forward", "browser forward", "اذهب للامام")):
        return "browser_forward"
    if _normalize_url(source_text):
        return "browser_open_url"
    if _extract_search_query(source_text):
        return "browser_search_web"

    if any(token in text for token in ("wifi", "wi fi", "wireless", "واي فاي", "الانترنت")):
        if any(token in text for token in ("off", "disable", "turn off", "افصل", "اطفي")):
            return "wifi_off"
        if any(token in text for token in ("on", "enable", "turn on", "شغل", "فعل")):
            return "wifi_on"

    if any(token in text for token in ("bluetooth", "بلوتوث")):
        if any(token in text for token in ("off", "disable", "turn off", "اطفي")):
            return "bluetooth_off"
        if any(token in text for token in ("on", "enable", "turn on", "شغل", "فعل")):
            return "bluetooth_on"

    if any(
        token in text
        for token in (
            "notification",
            "notifications",
            "notify",
            "dnd",
            "do not disturb",
            "focus assist",
            "اشعارات",
            "إشعارات",
            "عدم الازعاج",
            "عدم الإزعاج",
        )
    ):
        if any(token in text for token in ("on", "enable", "turn on", "شغل", "فعل")):
            if any(token in text for token in ("dnd", "do not disturb", "focus assist", "عدم الازعاج", "عدم الإزعاج")):
                return "notifications_off"
            return "notifications_on"
        if any(token in text for token in ("off", "disable", "turn off", "اطفي", "اقفل", "اكتم", "mute", "silence")):
            if any(token in text for token in ("dnd", "do not disturb", "focus assist", "عدم الازعاج", "عدم الإزعاج")):
                return "notifications_on"
            return "notifications_off"

        if any(token in text for token in ("dnd", "do not disturb", "focus assist", "عدم الازعاج", "عدم الإزعاج")):
            return "notifications_off"

    if fallback_hint in _ALLOWED_SYSTEM_ACTION_KEYS:
        return fallback_hint

    return ""


def _norm_text(value):
    return " ".join(str(value or "").strip().lower().split())


def _looks_informational_query(source_text: str) -> bool:
    text = _norm_text(source_text)
    if not text:
        return False

    if any(token in text for token in _SYSTEM_ACTION_MARKERS):
        return False

    if any(marker in text for marker in _INFORMATIONAL_QUERY_MARKERS):
        return True

    return False


def _has_required_markers_for_action(source_text: str, action_key: str) -> bool:
    required = _ACTION_KEY_REQUIRED_MARKERS.get(str(action_key or "").strip().lower())
    if not required:
        return True
    text = _norm_text(source_text)
    if not text:
        return False
    return any(marker in text for marker in required)


def _has_system_action_negation(source_text: str, action_key: str) -> bool:
    text = _norm_text(source_text)
    if not text:
        return False

    if str(action_key or "").strip().lower() in {"notifications_on", "notifications_off"}:
        if "do not disturb" in text or "عدم الازعاج" in text or "عدم الإزعاج" in text:
            return False

    if re.search(r"\b(?:don't|do\s+not|dont)\b", text):
        return True
    if re.search(r"\bnot\b", text):
        return True
    if re.search(r"(?:\bلا\b|\bمو\b|\bليس\b)", text):
        return True
    return False


def _safe_json_object(raw_text: str):
    text = str(raw_text or "").strip()
    if not text:
        return None

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    match = _JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        if isinstance(value, dict):
            return value
    except Exception:
        return None
    return None


def _normalize_nlu_payload(payload: dict, source_text: str = ""):
    intent = str(payload.get("intent") or "LLM_QUERY").strip().upper()
    if intent not in _ALLOWED_INTENTS:
        intent = "LLM_QUERY"

    action = str(payload.get("action") or "").strip().lower()
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    args = dict(args or {})
    source_norm = _norm_text(source_text)
    forced_app_open = False

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if intent == "OS_FILE_NAVIGATION" and action not in _ALLOWED_NAV_ACTIONS:
        intent = "LLM_QUERY"
        action = ""
        args = {}
        confidence = min(confidence, 0.3)

    if intent == "VOICE_COMMAND" and action not in _ALLOWED_VOICE_ACTIONS:
        intent = "LLM_QUERY"
        action = ""
        args = {}
        confidence = min(confidence, 0.3)

    if intent == "JOB_QUEUE_COMMAND" and action not in _ALLOWED_JOB_ACTIONS:
        intent = "LLM_QUERY"
        action = ""
        args = {}
        confidence = min(confidence, 0.3)

    if intent == "VOICE_COMMAND" and action in {"speech_on", "speech_off"}:
        patterns = _VOICE_SPEECH_ON_PATTERNS if action == "speech_on" else _VOICE_SPEECH_OFF_PATTERNS
        if not any(pattern.search(source_norm) for pattern in patterns):
            # Do not let noisy STT text accidentally toggle speech state.
            intent = "LLM_QUERY"
            action = ""
            args = {}
            confidence = min(confidence, 0.3)

    schedule_payload = _extract_schedule_payload(source_text)
    if schedule_payload:
        intent = "JOB_QUEUE_COMMAND"
        action = "enqueue"
        args = dict(schedule_payload)
        confidence = max(confidence, 0.82)

    media_target = _infer_media_app_target(source_text, args)
    media_open_hints = (
        "play music on",
        "music on",
        "open spotify",
        "open vlc",
        "open youtube music",
        "شغل موسيقى",
        "افتح سبوتيفاي",
        "افتح vlc",
        "افتح يوتيوب ميوزك",
    )
    if media_target and any(hint in source_norm for hint in media_open_hints):
        intent = "OS_APP_OPEN"
        action = ""
        args = {"app_name": media_target}
        confidence = max(confidence, 0.80)
        forced_app_open = True

    app_target = _infer_app_target(source_text, args)
    app_open_hints = (
        "i need",
        "i want",
        "open",
        "launch",
        "start",
        "افتح",
        "شغل",
        "اريد",
        "أريد",
        "عايز",
        "عاوز",
    )
    if app_target and any(hint in source_norm for hint in app_open_hints):
        intent = "OS_APP_OPEN"
        action = ""
        args = {"app_name": app_target}
        confidence = max(confidence, 0.80)
        forced_app_open = True

    if intent == "OS_APP_OPEN" and "app_name" not in args:
        for key in ("target", "app", "name"):
            if args.get(key):
                args["app_name"] = str(args.get(key)).strip()
                break

    if intent == "OS_APP_CLOSE" and "app_name" not in args:
        for key in ("target", "app", "name"):
            if args.get(key):
                args["app_name"] = str(args.get(key)).strip()
                break

    if intent == "OS_FILE_SEARCH" and "filename" not in args:
        for key in ("file", "name", "target"):
            if args.get(key):
                args["filename"] = str(args.get(key)).strip()
                break

    inferred_action_key = ""
    if intent != "JOB_QUEUE_COMMAND" and not forced_app_open:
        inferred_action_key = _infer_system_action_key(source_text, args, action_key_hint=action)

    if inferred_action_key:
        intent = "OS_SYSTEM_COMMAND"
        args["action_key"] = inferred_action_key

    if intent == "OS_SYSTEM_COMMAND":
        action_key = str(args.get("action_key") or action or "").strip().lower()
        if action_key not in _ALLOWED_SYSTEM_ACTION_KEYS:
            intent = "LLM_QUERY"
            action = ""
            args = {}
            confidence = min(confidence, 0.3)
        else:
            if _looks_informational_query(source_text):
                intent = "LLM_QUERY"
                action = ""
                args = {}
                confidence = min(confidence, 0.30)
            elif _has_system_action_negation(source_text, action_key):
                intent = "LLM_QUERY"
                action = ""
                args = {}
                confidence = min(confidence, 0.30)
            elif not _has_required_markers_for_action(source_text, action_key):
                # Block high-impact hallucinated system actions when utterance lacks action evidence.
                intent = "LLM_QUERY"
                action = ""
                args = {}
                confidence = min(confidence, 0.30)
            else:
                args["action_key"] = action_key

                if action_key == "volume_set":
                    level = _coerce_percent_int(
                        args.get("volume_level")
                        or args.get("level")
                        or args.get("percent")
                        or args.get("value")
                    )
                    if level is None:
                        level = _extract_first_number(source_text)
                    if level is None:
                        confidence = min(confidence, 0.35)
                    else:
                        args["volume_level"] = level
                        confidence = max(confidence, 0.78)

                if action_key == "brightness_set":
                    level = _coerce_percent_int(
                        args.get("brightness_level")
                        or args.get("level")
                        or args.get("percent")
                        or args.get("value")
                    )
                    if level is None:
                        level = _extract_first_number(source_text)
                    if level is None:
                        confidence = min(confidence, 0.35)
                    else:
                        args["brightness_level"] = level
                        confidence = max(confidence, 0.78)

                if action_key == "focus_window":
                    query = str(
                        args.get("window_query")
                        or args.get("window_title")
                        or args.get("target")
                        or args.get("app_name")
                        or ""
                    ).strip()
                    if not query:
                        query = _extract_focus_query(source_text)
                    if not query:
                        confidence = min(confidence, 0.35)
                    else:
                        args["window_query"] = _canonicalize_window_query(query)
                        confidence = max(confidence, 0.78)

                if action_key in {"media_seek_forward", "media_seek_backward"}:
                    seconds = _parse_duration_seconds(
                        args.get("seek_seconds")
                        or args.get("seconds")
                        or args.get("duration")
                        or args.get("value")
                        or _extract_first_number(source_text)
                        or 10,
                        args.get("unit") or "seconds",
                    )
                    if seconds is None:
                        confidence = min(confidence, 0.35)
                    else:
                        args["seek_seconds"] = int(seconds)
                        confidence = max(confidence, 0.78)

                if action_key == "browser_open_url":
                    url = _normalize_url(
                        args.get("url")
                        or args.get("link")
                        or args.get("target")
                        or args.get("value")
                        or source_text
                    )
                    if not url:
                        confidence = min(confidence, 0.35)
                    else:
                        args["url"] = url
                        confidence = max(confidence, 0.78)

                if action_key == "browser_search_web":
                    query = str(
                        args.get("search_query")
                        or args.get("query")
                        or args.get("text")
                        or args.get("value")
                        or ""
                    ).strip()
                    if not query:
                        query = _extract_search_query(source_text)
                    if not query:
                        confidence = min(confidence, 0.35)
                    else:
                        args["search_query"] = query
                        confidence = max(confidence, 0.78)

                action = ""

    if intent == "JOB_QUEUE_COMMAND":
        normalized_action = action or "enqueue"
        if normalized_action not in _ALLOWED_JOB_ACTIONS:
            normalized_action = "enqueue"

        if normalized_action == "enqueue":
            command_text = str(
                args.get("command_text")
                or args.get("command")
                or args.get("task")
                or args.get("text")
                or ""
            ).strip()
            delay_seconds = _parse_duration_seconds(
                args.get("delay_seconds")
                or args.get("seconds")
                or args.get("delay")
                or args.get("value")
                or _extract_first_number(source_text)
                or 0,
                args.get("unit") or "seconds",
            )

            if not command_text:
                schedule_patterns = (
                    r"(?:in|after)\s+.+?\s+(?:seconds?|secs?|minutes?|mins?|hours?|hrs?)\s+(.+)$",
                    r"(?:بعد)\s+.+?\s+(?:ثانية|ثواني|دقيقة|دقائق|ساعة|ساعات)\s+(.+)$",
                )
                for pattern in schedule_patterns:
                    match = re.search(pattern, str(source_text or ""), flags=re.IGNORECASE)
                    if match:
                        command_text = match.group(1).strip()
                        break

            if not command_text:
                confidence = min(confidence, 0.35)
            else:
                args["command_text"] = command_text
                args["delay_seconds"] = int(max(0, delay_seconds or 0))
                confidence = max(confidence, 0.78)

        action = normalized_action

    if intent == "OS_APP_OPEN" and "app_name" not in args:
        confidence = min(confidence, 0.35)
    if intent == "OS_APP_CLOSE" and "app_name" not in args:
        confidence = min(confidence, 0.35)
    if intent == "OS_FILE_SEARCH" and "filename" not in args:
        confidence = min(confidence, 0.35)
    if intent == "OS_SYSTEM_COMMAND" and "action_key" not in args:
        confidence = min(confidence, 0.35)
    if intent == "JOB_QUEUE_COMMAND" and action == "enqueue" and "command_text" not in args:
        confidence = min(confidence, 0.35)

    return {
        "intent": intent,
        "action": action,
        "args": args,
        "confidence": confidence,
    }


def classify(text: str):
    return parse_command(text).intent


def classify_with_confidence(text: str, language: str = "en"):
    parsed = parse_command(text)
    assessment = assess_intent_confidence(text, parsed, language=language)
    return {
        "intent": parsed.intent,
        "action": parsed.action,
        "args": dict(parsed.args or {}),
        "confidence": float(assessment.confidence),
        "entity_scores": dict(assessment.entity_scores or {}),
        "should_clarify": bool(assessment.should_clarify),
        "reason": assessment.reason,
    }


def classify_with_nlu(text: str, language: str = "en", use_cache: bool = True):
    if use_cache:
        cached = _cache_get_nlu_result(text, language)
        if cached is not None:
            return cached

    prompt = build_intent_extraction_prompt(text, language=language)
    raw = ask_llm(prompt)
    payload = _safe_json_object(raw)
    if not payload:
        return {
            "ok": False,
            "intent": "LLM_QUERY",
            "action": "",
            "args": {},
            "confidence": 0.0,
            "error": "invalid_nlu_response",
            "raw": raw,
            "cache_hit": False,
        }

    normalized = _normalize_nlu_payload(payload, source_text=text)
    normalized["ok"] = True
    normalized["raw"] = raw
    normalized["cache_hit"] = False
    if use_cache:
        _cache_put_nlu_result(text, language, normalized)
    return normalized
