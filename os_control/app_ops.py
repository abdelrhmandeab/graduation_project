import re
import shutil
import threading
import time
from difflib import SequenceMatcher

from core.config import (
    APP_CATALOG_TTL_HOURS,
    APP_REFRESH_ON_MISS,
    APP_RESOLUTION_AVAILABLE_BONUS,
    APP_RESOLUTION_RECENT_BONUS_SECONDS,
    APP_RESOLUTION_RUNNING_BONUS_CLOSE,
    APP_RESOLUTION_RUNNING_BONUS_OPEN,
    APP_RESOLUTION_USAGE_BOOST_CAP,
    APP_RESOLUTION_USAGE_BOOST_PER_HIT,
    CONFIRMATION_TIMEOUT_SECONDS,
)
from core.logger import logger
from core.response_templates import format_confirmation_prompt
from core.session_memory import session_memory
from os_control.action_log import log_action
from os_control.adapter_result import confirmation_result, failure_result, success_result, to_legacy_pair
from os_control.confirmation import confirmation_manager
from os_control.policy import policy_engine
from os_control.app_scanner import get_catalog_age_seconds, scan_installed_apps
from os_control.powershell_bridge import run_template
from os_control.risk_policy import risk_tier_for_app_operation

_REFRESH_LOCK = threading.Lock()


_APP_CATALOG = {
    "notepad.exe": {
        "canonical_name": "Notepad",
        "aliases": [
            "notepad",
            "notes",
            "text editor",
            "\u0627\u0644\u0645\u0641\u0643\u0631\u0629",
            "\u0645\u0641\u0643\u0631\u0629",
            "\u0645\u062d\u0631\u0631 \u0646\u0635\u0648\u0635",
        ],
    },
    "calc.exe": {
        "canonical_name": "Calculator",
        "aliases": [
            "calculator",
            "calc",
            "math",
            "\u0627\u0644\u0622\u0644\u0629 \u0627\u0644\u062d\u0627\u0633\u0628\u0629",
            "\u062d\u0627\u0633\u0628\u0629",
            "\u0627\u0644\u062d\u0627\u0633\u0628\u0629",
        ],
    },
    "mspaint.exe": {
        "canonical_name": "Paint",
        "aliases": [
            "paint",
            "ms paint",
            "\u0628\u064a\u0646\u062a",
            "\u0631\u0633\u0627\u0645",
            "\u0627\u0644\u0631\u0633\u0627\u0645",
        ],
    },
    "cmd.exe": {
        "canonical_name": "Command Prompt",
        "aliases": [
            "cmd",
            "command prompt",
            "terminal",
            "\u0645\u0648\u062c\u0647 \u0627\u0644\u0623\u0648\u0627\u0645\u0631",
            "\u062a\u0631\u0645\u064a\u0646\u0627\u0644",
            "\u0637\u0631\u0641\u064a\u0629",
        ],
    },
    "powershell.exe": {
        "canonical_name": "PowerShell",
        "aliases": [
            "powershell",
            "power shell",
            "ps",
            "\u0628\u0627\u0648\u0631 \u0634\u064a\u0644",
            "\u0628\u0627\u0648\u0631\u0634\u064a\u0644",
        ],
    },
    "explorer.exe": {
        "canonical_name": "File Explorer",
        "aliases": [
            "explorer",
            "file explorer",
            "files",
            "\u0645\u0633\u062a\u0643\u0634\u0641 \u0627\u0644\u0645\u0644\u0641\u0627\u062a",
            "\u0645\u0633\u062a\u0639\u0631\u0636 \u0627\u0644\u0645\u0644\u0641\u0627\u062a",
        ],
    },
    "taskmgr.exe": {
        "canonical_name": "Task Manager",
        "aliases": [
            "task manager",
            "taskmgr",
            "\u0645\u062f\u064a\u0631 \u0627\u0644\u0645\u0647\u0627\u0645",
        ],
    },
    "control.exe": {
        "canonical_name": "Control Panel",
        "aliases": [
            "control panel",
            "settings control panel",
            "\u0644\u0648\u062d\u0629 \u0627\u0644\u062a\u062d\u0643\u0645",
        ],
    },
    "ms-settings:": {
        "canonical_name": "Windows Settings",
        "aliases": [
            "settings",
            "windows settings",
            "system settings",
            "\u0627\u0644\u0627\u0639\u062f\u0627\u062f\u0627\u062a",
            "\u0627\u0644\u0625\u0639\u062f\u0627\u062f\u0627\u062a",
        ],
    },
    "start microsoft-edge:": {
        "canonical_name": "Microsoft Edge",
        "aliases": [
            "edge",
            "microsoft edge",
            "\u0625\u064a\u062f\u062c",
        ],
    },
    "start chrome": {
        "canonical_name": "Google Chrome",
        "aliases": [
            "chrome",
            "google chrome",
            "\u0643\u0631\u0648\u0645",
            "\u062c\u0648\u062c\u0644 \u0643\u0631\u0648\u0645",
        ],
    },
    "powerpnt.exe": {
        "canonical_name": "PowerPoint",
        "aliases": [
            "powerpoint",
            "power point",
            "ppt",
            "presentation",
            "\u0628\u0627\u0648\u0631 \u0628\u0648\u064a\u0646\u062a",
            "\u0639\u0631\u0636 \u062a\u0642\u062f\u064a\u0645\u064a",
        ],
    },
    "wt.exe": {
        "canonical_name": "Windows Terminal",
        "aliases": [
            "windows terminal",
            "terminal app",
            "wt",
            "\u0648\u064a\u0646\u062f\u0648\u0632 \u062a\u0631\u0645\u064a\u0646\u0627\u0644",
            "\u062a\u0631\u0645\u064a\u0646\u0627\u0644 \u0648\u064a\u0646\u062f\u0648\u0632",
        ],
    },
    "code": {
        "canonical_name": "Visual Studio Code",
        "aliases": [
            "visual studio code",
            "vscode",
            "vs code",
            "code editor",
            "\u0641\u064a\u062c\u0648\u0627\u0644 \u0633\u062a\u0648\u062f\u064a\u0648 \u0643\u0648\u062f",
            "\u0641\u064a \u0627\u0633 \u0643\u0648\u062f",
        ],
    },
    "winword.exe": {
        "canonical_name": "Microsoft Word",
        "aliases": [
            "word",
            "microsoft word",
            "ms word",
            "doc editor",
            "\u0648\u0648\u0631\u062f",
            "\u0645\u0627\u064a\u0643\u0631\u0648\u0633\u0648\u0641\u062a \u0648\u0648\u0631\u062f",
        ],
    },
    "excel.exe": {
        "canonical_name": "Microsoft Excel",
        "aliases": [
            "excel",
            "microsoft excel",
            "sheet",
            "spreadsheet",
            "\u0627\u0643\u0633\u0644",
            "\u0645\u0627\u064a\u0643\u0631\u0648\u0633\u0648\u0641\u062a \u0627\u0643\u0633\u0644",
        ],
    },
    "outlook.exe": {
        "canonical_name": "Outlook",
        "aliases": [
            "outlook",
            "mail",
            "email",
            "\u0627\u0648\u062a\u0644\u0648\u0643",
            "\u0627\u0644\u0628\u0631\u064a\u062f",
        ],
    },
    "onenote.exe": {
        "canonical_name": "OneNote",
        "aliases": [
            "onenote",
            "one note",
            "notes app",
            "\u0648\u0646 \u0646\u0648\u062a",
            "\u062a\u0637\u0628\u064a\u0642 \u0627\u0644\u0645\u0644\u0627\u062d\u0638\u0627\u062a",
        ],
    },
    "firefox.exe": {
        "canonical_name": "Mozilla Firefox",
        "aliases": [
            "firefox",
            "mozilla firefox",
            "browser firefox",
            "\u0641\u0627\u064a\u0631\u0641\u0648\u0643\u0633",
        ],
    },
    "vlc.exe": {
        "canonical_name": "VLC",
        "aliases": [
            "vlc",
            "vlc player",
            "video player",
            "\u0641\u064a \u0627\u0644 \u0633\u064a",
            "\u0645\u0634\u063a\u0644 \u0641\u064a\u062f\u064a\u0648",
        ],
    },
    "spotify.exe": {
        "canonical_name": "Spotify",
        "aliases": [
            "spotify",
            "spotify app",
            "music app",
            "\u0633\u0628\u0648\u062a\u064a\u0641\u0627\u064a",
        ],
    },
    "telegram.exe": {
        "canonical_name": "Telegram",
        "aliases": [
            "telegram",
            "telegram desktop",
            "\u062a\u064a\u0644\u064a\u062c\u0631\u0627\u0645",
        ],
    },
    "whatsapp.exe": {
        "canonical_name": "WhatsApp",
        "aliases": [
            "whatsapp",
            "whats app",
            "\u0648\u0627\u062a\u0633 \u0627\u0628",
            "\u0648\u0627\u062a\u0633\u0627\u0628",
        ],
    },
    "discord.exe": {
        "canonical_name": "Discord",
        "aliases": [
            "discord",
            "discord app",
            "\u062f\u064a\u0633\u0643\u0648\u0631\u062f",
        ],
    },
    # Brand aliases for apps whose catalog name differs from what users say
    "claude.exe": {
        "canonical_name": "Claude",
        "aliases": ["claude", "claude ai", "anthropic claude", "\u0643\u0644\u0648\u062f"],
    },
    "cursor.exe": {
        "canonical_name": "Cursor",
        "aliases": ["cursor", "cursor editor", "\u0643\u064a\u0631\u0633\u0648\u0631"],
    },
    "obsidian.exe": {
        "canonical_name": "Obsidian",
        "aliases": ["obsidian", "obsidian notes", "\u0623\u0648\u0628\u0633\u064a\u062f\u064a\u0627\u0646"],
    },
    "notion.exe": {
        "canonical_name": "Notion",
        "aliases": ["notion", "notion app", "\u0646\u0648\u0634\u0646"],
    },
    "slack.exe": {
        "canonical_name": "Slack",
        "aliases": ["slack", "slack app", "\u0633\u0644\u0627\u0643"],
    },
    "zoom.exe": {
        "canonical_name": "Zoom",
        "aliases": ["zoom", "zoom meeting", "\u0632\u0648\u0645"],
    },
    "teams.exe": {
        "canonical_name": "Microsoft Teams",
        "aliases": ["teams", "microsoft teams", "ms teams", "\u062a\u064a\u0645\u0632"],
    },
    "postman.exe": {
        "canonical_name": "Postman",
        "aliases": ["postman", "api client", "\u0628\u0648\u0633\u062a\u0645\u0627\u0646"],
    },
    "figma.exe": {
        "canonical_name": "Figma",
        "aliases": ["figma", "design tool", "\u0641\u064a\u062c\u0645\u0627"],
    },
    "gimp.exe": {
        "canonical_name": "GIMP",
        "aliases": ["gimp", "image editor", "\u062c\u064a\u0645\u0628"],
    },
}

_BASE_APP_CATALOG = dict(_APP_CATALOG)


def _apply_app_catalog(app_catalog):
    global _APP_CATALOG, _EXECUTABLE_TO_CANONICAL, KNOWN_APPS

    _APP_CATALOG = app_catalog or {}
    _EXECUTABLE_TO_CANONICAL = {
        executable: payload.get("canonical_name", executable)
        for executable, payload in _APP_CATALOG.items()
    }
    KNOWN_APPS = _build_known_apps_alias_map()


def refresh_app_catalog(force: bool = True) -> int:
    """Rescan installed apps and apply the result. Returns catalog size. Thread-safe."""
    with _REFRESH_LOCK:
        try:
            catalog = scan_installed_apps(_BASE_APP_CATALOG, force=force)
            _apply_app_catalog(catalog)
            count = len(catalog)
            logger.info("app catalog refreshed: %d entries (force=%s)", count, force)
            return count
        except Exception as exc:
            logger.warning("app catalog refresh failed: %s", exc)
            return len(_APP_CATALOG)


def _normalize_alias(text):
    value = " ".join((text or "").lower().split()).strip()
    value = re.sub(r"^(?:open app|open|start|launch|run|close app|close|kill app|terminate app)\s+", "", value)
    value = re.sub(
        r"^(?:\u0627\u0641\u062a\u062d \u062a\u0637\u0628\u064a\u0642|\u0627\u0641\u062a\u062d|\u0634\u063a\u0644 \u062a\u0637\u0628\u064a\u0642|\u0634\u063a\u0644|\u0627\u063a\u0644\u0642 \u062a\u0637\u0628\u064a\u0642|\u0627\u0642\u0641\u0644 \u062a\u0637\u0628\u064a\u0642|\u0633\u0643\u0631 \u062a\u0637\u0628\u064a\u0642|\u0627\u0646\u0647\u064a \u062a\u0637\u0628\u064a\u0642)\s+",
        "",
        value,
    )
    value = re.sub(r"[^a-z0-9\u0600-\u06FF\s.+_-]", " ", value)
    value = " ".join(value.split())
    return value


def _similarity(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    score = SequenceMatcher(a=a, b=b).ratio()
    if a in b or b in a:
        score = max(score, min(len(a), len(b)) / max(len(a), len(b)))
    return float(score)


def _build_known_apps_alias_map():
    alias_map = {}
    for executable, payload in _APP_CATALOG.items():
        canonical = payload["canonical_name"]
        alias_map[_normalize_alias(canonical)] = executable
        for alias in payload.get("aliases", []):
            alias_map[_normalize_alias(alias)] = executable
    return alias_map


KNOWN_APPS = _build_known_apps_alias_map()
_apply_app_catalog(scan_installed_apps(_BASE_APP_CATALOG))
_RETRYABLE_OPEN_ERRORS = ("timed out", "temporarily unavailable")
_RETRYABLE_CLOSE_ERRORS = ("timed out", "temporarily unavailable")
_PROCESS_NAME_OVERRIDES = {
    "start chrome": "chrome.exe",
    "start microsoft-edge:": "msedge.exe",
    "ms-settings:": "SystemSettings.exe",
    # UWP apps: scanner names them <short>.uwp but the real process is different
    "spotifymusic.uwp": "Spotify.exe",
    "spotify.uwp": "Spotify.exe",
    "whatsappdesktop.uwp": "WhatsApp.exe",
    "whatsapp.uwp": "WhatsApp.exe",
    "microsoftteams.uwp": "ms-teams.exe",
    "msteams.uwp": "ms-teams.exe",
    "windowsterminal.uwp": "WindowsTerminal.exe",
    "microsoftnotepad.uwp": "Notepad.exe",
    "notepad.uwp": "Notepad.exe",
    "clipchamp.uwp": "Clipchamp.exe",
    "microsoftclipchamp.uwp": "Clipchamp.exe",
}
_PROCESS_SNAPSHOT_TTL_SECONDS = 8.0
_PROCESS_SNAPSHOT_CACHE = {"captured_at": 0.0, "names": set()}
_PROCESS_SNAPSHOT_AVAILABLE = True


def _friendly_open_error(target, error_text):
    lowered = (error_text or "").lower()
    if "cannot find the file specified" in lowered or "not recognized" in lowered:
        return (
            f"I could not find an app or executable named '{target}'. "
            "Try `open app notepad` or use a filesystem command like `open C partition`."
        )
    if "access is denied" in lowered:
        return f"Access denied while trying to open '{target}'."
    if "timed out" in lowered:
        return f"Timed out while opening '{target}'. Please try again."
    return "I could not open that app."


def _friendly_close_error(target, error_text):
    lowered = (error_text or "").lower()
    if "cannot find a process" in lowered or "no process" in lowered:
        return f"I could not find a running process for '{target}'."
    if "access is denied" in lowered:
        return f"Access denied while trying to close '{target}'."
    if "timed out" in lowered:
        return f"Timed out while closing '{target}'. Please try again."
    return "I could not close that app."


def _error_code_from_text(error_text):
    lowered = (error_text or "").lower()
    if "access is denied" in lowered:
        return "permission_denied"
    if "cannot find" in lowered or "not recognized" in lowered or "no process" in lowered:
        return "not_found"
    if "timed out" in lowered:
        return "timeout"
    return "execution_failed"


def _to_process_name(target):
    raw = str(target or "").strip()
    lowered = raw.lower()
    if lowered in _PROCESS_NAME_OVERRIDES:
        return _PROCESS_NAME_OVERRIDES[lowered]
    if lowered.startswith("start "):
        raw = raw[6:].strip()
    raw = raw.rstrip(":")
    base = raw.replace("/", "\\").split("\\")[-1]
    if not base:
        return ""
    base_lower = base.lower()
    # UWP apps from the scanner are named "<short>.uwp" but don't have a matching
    # process. Try to find the real running process by matching the short name.
    if base_lower.endswith(".uwp"):
        short = base_lower[:-4]  # strip ".uwp"
        running = _running_process_names_snapshot()
        # Prefer an exact prefix match (e.g. "spotify" → "spotify.exe")
        for proc in sorted(running):
            proc_base = proc.rstrip(".exe").rstrip(".EXE").lower() if proc.lower().endswith(".exe") else proc.lower()
            if proc_base == short or proc_base.startswith(short):
                return proc  # already lowercase with .exe from psutil
        # No match found — return a best-guess so the error message is meaningful
        return f"{short}.exe"
    if not base_lower.endswith(".exe"):
        return f"{base}.exe"
    return base


def _running_process_names_snapshot():
    global _PROCESS_SNAPSHOT_AVAILABLE

    now_ts = time.time()
    cached_at = float(_PROCESS_SNAPSHOT_CACHE.get("captured_at") or 0.0)
    if now_ts - cached_at < _PROCESS_SNAPSHOT_TTL_SECONDS:
        return set(_PROCESS_SNAPSHOT_CACHE.get("names") or set())

    names = set()
    try:
        import psutil  # type: ignore

        for proc in psutil.process_iter(attrs=["name"]):
            name = str((proc.info or {}).get("name") or "").strip().lower()
            if name:
                names.add(name)
        _PROCESS_SNAPSHOT_AVAILABLE = True
    except Exception:
        names = set()
        _PROCESS_SNAPSHOT_AVAILABLE = False

    _PROCESS_SNAPSHOT_CACHE["captured_at"] = now_ts
    _PROCESS_SNAPSHOT_CACHE["names"] = names
    return set(names)


def _is_process_running(process_name):
    normalized = str(process_name or "").strip().lower()
    if not normalized:
        return False

    running_names = _running_process_names_snapshot()
    if not _PROCESS_SNAPSHOT_AVAILABLE:
        # If process snapshots are unavailable (for example psutil is missing),
        # do not block close operations with a hard not-found check.
        return True
    return normalized in running_names


def _is_executable_available(executable):
    value = str(executable or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if lowered.startswith("start ") or lowered.endswith(":"):
        return True
    if shutil.which(value):
        return True
    if lowered.endswith(".exe") and shutil.which(value[:-4]):
        return True
    return False


def _candidate_score(query, alias, executable, operation, running_names):
    similarity = _similarity(query, alias)
    if similarity < 0.46:
        return None

    score = similarity * 0.78
    if query == alias:
        score += 0.20
    elif alias.startswith(query) or query.startswith(alias):
        score += 0.05

    usage = session_memory.get_app_usage_stats(executable)
    usage_count = int(usage.get("count") or 0)
    score += min(float(APP_RESOLUTION_USAGE_BOOST_CAP), usage_count * float(APP_RESOLUTION_USAGE_BOOST_PER_HIT))

    last_used_at = float(usage.get("last_used_at") or 0.0)
    if last_used_at and (time.time() - last_used_at) <= float(APP_RESOLUTION_RECENT_BONUS_SECONDS):
        score += 0.06

    process_name = _to_process_name(executable).lower()
    is_running = bool(process_name and process_name in running_names)
    if is_running:
        if str(operation or "open").lower() == "close":
            score += float(APP_RESOLUTION_RUNNING_BONUS_CLOSE)
        else:
            score += float(APP_RESOLUTION_RUNNING_BONUS_OPEN)

    if _is_executable_available(executable):
        score += float(APP_RESOLUTION_AVAILABLE_BONUS)

    return {
        "score": round(min(1.0, max(0.0, score)), 4),
        "similarity": round(float(similarity), 4),
        "usage_count": usage_count,
        "is_running": is_running,
        "available": _is_executable_available(executable),
    }


def resolve_app_candidates(app_name, limit=5, operation="open"):
    query = _normalize_alias(app_name)
    if not query:
        return []

    if query in KNOWN_APPS:
        executable = KNOWN_APPS[query]
        return [
            {
                "canonical_name": _EXECUTABLE_TO_CANONICAL.get(executable, executable),
                "executable": executable,
                "matched_alias": query,
                "score": 1.0,
                "similarity": 1.0,
                "usage_count": int(session_memory.get_app_usage_stats(executable).get("count") or 0),
                "is_running": _to_process_name(executable).lower() in _running_process_names_snapshot(),
                "available": _is_executable_available(executable),
            }
        ]

    best_by_executable = {}
    running_names = _running_process_names_snapshot()
    for alias, executable in KNOWN_APPS.items():
        scoring = _candidate_score(query, alias, executable, operation, running_names)
        if not scoring:
            continue
        current = best_by_executable.get(executable)
        payload = {
            "canonical_name": _EXECUTABLE_TO_CANONICAL.get(executable, executable),
            "executable": executable,
            "matched_alias": alias,
            "score": scoring["score"],
            "similarity": scoring["similarity"],
            "usage_count": scoring["usage_count"],
            "is_running": scoring["is_running"],
            "available": scoring["available"],
        }
        if not current or payload["score"] > current["score"]:
            best_by_executable[executable] = payload

    candidates = sorted(
        best_by_executable.values(),
        key=lambda item: item["score"],
        reverse=True,
    )
    return candidates[: max(1, int(limit))]


# These words are Windows folder names / generic nouns — they must never fuzzy-match
# an app unless the query IS an exact catalog alias (e.g. "downloads" is a folder, not an app).
_FOLDER_WORDS_BLOCKLIST = {
    "download", "downloads", "documents", "desktop", "pictures", "videos", "music",
    "folder", "directory", "drive", "disk", "partition", "files", "file",
    "تنزيلات", "مستندات", "صور", "فيديوهات", "موسيقى", "سطح المكتب",
}


def resolve_app_request(app_name, operation="open"):
    query = _normalize_alias(app_name)
    if not query:
        return {"status": "none", "query": query, "candidates": []}

    # Block generic folder/noun words from resolving as apps — even if the scanner
    # accidentally aliased something to one of these words (e.g. "desktop" → Telegram).
    # These words should route to OS_FILE_NAVIGATION instead.
    if query in _FOLDER_WORDS_BLOCKLIST:
        return {"status": "none", "query": query, "candidates": []}

    # Single-token queries get a lower similarity floor (brand names like "claude",
    # "figma", "notion" are short and won't score above 0.74 against longer catalog
    # names without this adjustment).
    is_single_token = len(query.split()) == 1

    def _resolve(q):
        if q in KNOWN_APPS:
            cands = resolve_app_candidates(q, limit=5, operation=operation)
            return {"status": "exact", "query": q, "candidates": cands}

        cands = resolve_app_candidates(q, limit=5, operation=operation)
        if not cands:
            return None

        top = cands[0]
        second = cands[1] if len(cands) > 1 else None
        second_score = float(second["score"]) if second else 0.0
        delta = float(top["score"]) - second_score

        high_floor = 0.82 if is_single_token else 0.90
        likely_floor = 0.62 if is_single_token else 0.74

        if float(top["score"]) >= high_floor and delta >= 0.08:
            return {"status": "high_confidence", "query": q, "candidates": [top]}

        if len(cands) > 1 and float(top["score"]) >= 0.62 and delta < 0.14:
            return {"status": "ambiguous", "query": q, "candidates": cands[:3]}

        if float(top["score"]) >= likely_floor:
            return {"status": "likely", "query": q, "candidates": [top]}

        return None

    result = _resolve(query)
    if result is not None:
        return result

    # Refresh-on-miss: if the catalog might be stale, rescan once and retry.
    if APP_REFRESH_ON_MISS:
        ttl_seconds = float(APP_CATALOG_TTL_HOURS) * 3600
        age = get_catalog_age_seconds()
        if age > ttl_seconds:
            logger.info("app_ops: catalog stale (age=%.0fs), refreshing for query=%r", age, query)
            refresh_app_catalog(force=True)
            result = _resolve(query)
            if result is not None:
                result["refresh_used"] = True
                return result

    return {"status": "none", "query": query, "candidates": []}


def _run_open_template(target):
    attempts = 0
    last_error = ""
    while attempts < 2:
        attempts += 1
        ok, error, _output = run_template(
            "open_app",
            env_overrides={"JARVIS_APP_PATH": target},
            timeout_seconds=15,
        )
        if ok:
            return True, "", attempts
        last_error = error or "PowerShell template failed"
        if not any(token in last_error.lower() for token in _RETRYABLE_OPEN_ERRORS):
            break
    return False, last_error, attempts


def _close_explorer_windows():
    """Close open File Explorer windows without killing the shell process."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        WM_CLOSE = 0x0010
        EXPLORER_CLASS = "CabinetWClass"

        closed = [0]

        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_size_t, ctypes.c_size_t)

        def _cb(hwnd, _):
            buf = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, buf, 256)
            if buf.value == EXPLORER_CLASS:
                user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
                closed[0] += 1
            return True

        user32.EnumWindows(EnumWindowsProc(_cb), 0)
        if closed[0]:
            return True, "", 1
        return False, "No open File Explorer windows found.", 1
    except Exception as exc:
        return False, str(exc), 1


def _run_close_template(process_name):
    # explorer.exe is the Windows shell — killing it crashes the desktop.
    # Instead, close only the open File Explorer folder windows via WM_CLOSE.
    if process_name.lower() in ("explorer.exe", "explorer"):
        return _close_explorer_windows()

    attempts = 0
    last_error = ""
    while attempts < 2:
        attempts += 1
        ok, error, _output = run_template(
            "close_app",
            env_overrides={"JARVIS_APP_PROCESS": process_name},
            timeout_seconds=15,
        )
        if ok:
            return True, "", attempts
        last_error = error or "PowerShell template failed"
        if not any(token in last_error.lower() for token in _RETRYABLE_CLOSE_ERRORS):
            break
    return False, last_error, attempts


def _execute_close_app(target, process_name, query, resolution_status, confirmed=False):
    try:
        ok, error, attempts = _run_close_template(process_name)
        if not ok:
            error_code = _error_code_from_text(error)
            log_action(
                "close_app",
                "failed",
                details={
                    "target": target,
                    "process_name": process_name,
                    "query": query,
                    "resolution_status": resolution_status,
                    "confirmed": bool(confirmed),
                    "attempts": attempts,
                },
                error=error,
            )
            return failure_result(
                _friendly_close_error(process_name, error),
                error_code=error_code,
                debug_info={
                    "target": target,
                    "process_name": process_name,
                    "query": query,
                    "resolution_status": resolution_status,
                    "confirmed": bool(confirmed),
                    "attempts": attempts,
                },
            )

        log_action(
            "close_app",
            "success",
            details={
                "target": target,
                "process_name": process_name,
                "query": query,
                "resolution_status": resolution_status,
                "confirmed": bool(confirmed),
                "attempts": attempts,
            },
        )
        return success_result(
            f"Closed {process_name}.",
            debug_info={
                "target": target,
                "process_name": process_name,
                "query": query,
                "resolution_status": resolution_status,
                "confirmed": bool(confirmed),
                "attempts": attempts,
            },
            executed_confirmed_action="app_operation" if confirmed else "",
        )
    except Exception as exc:
        log_action("close_app", "failed", details={"target": target, "query": query}, error=exc)
        logger.error("Close app failed: %s", exc)
        return failure_result(
            str(exc),
            error_code="execution_failed",
            debug_info={"target": target, "query": query, "confirmed": bool(confirmed)},
        )


def _resolve_close_target(app_name):
    query = _normalize_alias(app_name)
    resolution = resolve_app_request(query, operation="close")
    if resolution["status"] == "ambiguous":
        return {
            "ok": False,
            "response": failure_result(
                "Multiple app matches were found. Please clarify.",
                error_code="ambiguous",
                debug_info={"query": query, "candidates": resolution.get("candidates", [])},
            ),
        }

    if resolution["status"] in {"exact", "high_confidence", "likely"} and resolution["candidates"]:
        target = resolution["candidates"][0]["executable"]
    else:
        target = app_name

    process_name = _to_process_name(target)
    if not process_name:
        return {
            "ok": False,
            "response": failure_result(
                "Could not determine the process name for this app.",
                error_code="invalid_input",
                debug_info={"target": target, "query": query},
            ),
        }

    if not _is_process_running(process_name):
        return {
            "ok": False,
            "response": failure_result(
                f"I could not find a running process for '{process_name}'.",
                error_code="not_found",
                debug_info={"target": target, "process_name": process_name, "query": query},
            ),
        }

    return {
        "ok": True,
        "target": target,
        "process_name": process_name,
        "query": query,
        "resolution_status": resolution.get("status"),
    }


def open_app_result(app_name):
    if not app_name:
        return failure_result("No app name provided.", error_code="invalid_input")
    if not policy_engine.is_command_allowed("app_open"):
        return failure_result(
            "Application launch is disabled by policy.",
            error_code="policy_blocked",
        )

    query = _normalize_alias(app_name)
    resolution = resolve_app_request(query, operation="open")

    if resolution["status"] == "ambiguous":
        return failure_result(
            "Multiple app matches were found. Please clarify.",
            error_code="ambiguous",
            debug_info={"query": query, "candidates": resolution.get("candidates", [])},
        )

    if resolution["status"] in {"exact", "high_confidence", "likely"} and resolution["candidates"]:
        executable = resolution["candidates"][0]["executable"]
        # Use the full path stored by the scanner when available; otherwise fall
        # back to the bare executable name (works for apps registered in App Paths
        # or on the system PATH, e.g. the hardcoded catalog entries).
        catalog_entry = _APP_CATALOG.get(executable) or {}
        target = str(catalog_entry.get("path") or "").strip() or executable
    else:
        target = app_name

    try:
        ok, error, attempts = _run_open_template(target)
        if not ok:
            error_code = _error_code_from_text(error)
            log_action(
                "open_app",
                "failed",
                details={"target": target, "query": query, "attempts": attempts},
                error=error,
            )
            try:
                from core.logger import log_structured

                log_structured("open_app", success=False, target=target, attempts=attempts)
            except Exception:
                pass
            return failure_result(
                _friendly_open_error(target, error),
                error_code=error_code,
                debug_info={
                    "target": target,
                    "query": query,
                    "attempts": attempts,
                    "resolution_status": resolution.get("status"),
                },
            )

        log_action(
            "open_app",
            "success",
            details={
                "target": target,
                "query": query,
                "attempts": attempts,
                "resolution_status": resolution.get("status"),
            },
        )
        try:
            from core.logger import log_structured

            log_structured("open_app", success=True, target=target, attempts=attempts)
        except Exception:
            pass
        logger.info("Opened app via template PowerShell: %s", target)
        return success_result(
            f"Opening {app_name}.",
            debug_info={
                "target": target,
                "query": query,
                "attempts": attempts,
                "resolution_status": resolution.get("status"),
            },
        )
    except Exception as exc:
        log_action("open_app", "failed", details={"target": target, "query": query}, error=exc)
        logger.error("Open app failed: %s", exc)
        return failure_result(
            str(exc),
            error_code="execution_failed",
            debug_info={"target": target, "query": query},
        )


def request_close_app_result(app_name):
    if not app_name:
        return failure_result("No app name provided.", error_code="invalid_input")
    if not policy_engine.is_command_allowed("app_close"):
        return failure_result(
            "Application close is disabled by policy.",
            error_code="policy_blocked",
        )

    resolved = _resolve_close_target(app_name)
    if not resolved.get("ok"):
        return resolved.get("response")

    risk_tier = risk_tier_for_app_operation("close_app")
    payload = {
        "kind": "app_operation",
        "operation": "close_app",
        "resolved_args": {
            "target": resolved["target"],
            "process_name": resolved["process_name"],
            "query": resolved["query"],
            "resolution_status": resolved.get("resolution_status"),
        },
        "risk_tier": risk_tier,
        "require_second_factor": False,
    }
    description = f"Close app `{resolved['process_name']}`"
    token = confirmation_manager.create(
        action_name="app_close",
        description=description,
        payload=payload,
    )
    log_action(
        "app_operation_request",
        "pending",
        details={
            "operation": "close_app",
            "risk_tier": risk_tier,
            "token": token,
            "second_factor": False,
            "args": payload["resolved_args"],
        },
    )

    message = format_confirmation_prompt(
        description,
        token,
        risk_tier=risk_tier,
        timeout_seconds=CONFIRMATION_TIMEOUT_SECONDS,
        require_second_factor=False,
    )
    return confirmation_result(
        message,
        token=token,
        second_factor=False,
        risk_tier=risk_tier,
        debug_info={
            "operation": "close_app",
            "resolved_args": dict(payload["resolved_args"]),
        },
    )


def execute_confirmed_app_operation(payload):
    if (payload or {}).get("kind") != "app_operation":
        return failure_result("Unsupported app operation payload.", error_code="unsupported_action")

    operation = (payload or {}).get("operation")
    resolved_args = (payload or {}).get("resolved_args") or {}
    risk_tier = (payload or {}).get("risk_tier") or risk_tier_for_app_operation("close_app")

    if operation != "close_app":
        return failure_result("Unsupported confirmed app operation.", error_code="unsupported_action")

    target = resolved_args.get("target")
    process_name = resolved_args.get("process_name")
    query = resolved_args.get("query")
    resolution_status = resolved_args.get("resolution_status")
    if not target or not process_name:
        return failure_result(
            "Invalid confirmation payload: missing target/process_name.",
            error_code="invalid_payload",
        )

    result = _execute_close_app(
        target=target,
        process_name=process_name,
        query=query,
        resolution_status=resolution_status,
        confirmed=True,
    )
    if isinstance(result, dict):
        result["risk_tier"] = risk_tier
    return result


def close_app_result(app_name):
    if not app_name:
        return failure_result("No app name provided.", error_code="invalid_input")
    if not policy_engine.is_command_allowed("app_close"):
        return failure_result(
            "Application close is disabled by policy.",
            error_code="policy_blocked",
        )

    resolved = _resolve_close_target(app_name)
    if not resolved.get("ok"):
        return resolved.get("response")

    return _execute_close_app(
        target=resolved["target"],
        process_name=resolved["process_name"],
        query=resolved["query"],
        resolution_status=resolved.get("resolution_status"),
        confirmed=False,
    )


def open_app(app_name):
    return to_legacy_pair(open_app_result(app_name))


def close_app(app_name):
    return to_legacy_pair(close_app_result(app_name))


def refresh_app_catalog_result(force=False):
    from core.config import FEATURE_FLAGS

    if not FEATURE_FLAGS.get("AUTO_APP_DISCOVERY_ENABLED", True):
        try:
            from core.logger import log_structured

            log_structured("app_catalog_refresh", success=False, reason="feature_disabled")
        except Exception:
            pass
        return failure_result("Auto app discovery is disabled.", error_code="feature_disabled")
    try:
        refreshed_catalog = scan_installed_apps(_BASE_APP_CATALOG, force=bool(force))
        _apply_app_catalog(refreshed_catalog)
        log_action(
            "app_catalog_refresh",
            "success",
            details={"force": bool(force), "app_count": len(_APP_CATALOG)},
        )
        try:
            from core.logger import log_structured

            log_structured("app_catalog_refresh", success=True, force=bool(force), app_count=len(_APP_CATALOG))
        except Exception:
            pass
        return success_result(
            f"Rescanned installed apps and found {len(_APP_CATALOG)} entries.",
            debug_info={"force": bool(force), "app_count": len(_APP_CATALOG)},
        )
    except Exception as exc:
        logger.error("App catalog rescan failed: %s", exc)
        log_action("app_catalog_refresh", "failed", details={"force": bool(force)}, error=exc)
        try:
            from core.logger import log_structured

            log_structured("app_catalog_refresh", success=False, force=bool(force))
        except Exception:
            pass
        return failure_result(
            "I could not rescan installed apps.",
            error_code="execution_failed",
            debug_info={"force": bool(force)},
        )
