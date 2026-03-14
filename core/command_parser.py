import os
import re
from dataclasses import dataclass, field

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
_MATCH_SANITIZE_RE = re.compile(r"[^a-z0-9_\s:\\/.\-]")
_DRIVE_COLON_RE = re.compile(r"\b([a-z])\s*:", flags=re.IGNORECASE)
_DRIVE_WORD_RE = re.compile(r"\b([a-z])\s+(?:drive|partition)\b", flags=re.IGNORECASE)
_SEA_C_DRIVE_RE = re.compile(r"\b(?:sea|see|cee)\s+(?:drive|partition)\b", flags=re.IGNORECASE)
_OPEN_FILLER_PREFIXES = (
    r"^(?:for me|for us|for me now|for me please)\s+",
    r"^(?:the)\s+",
)
_FILESYSTEM_OPEN_HINTS = (
    "drive", "partition", "folder", "directory", "desktop",
    "downloads", "documents", "pictures", "music", "videos", "file explorer",
)
_SPECIAL_FOLDERS = {
    "desktop": "Desktop",
    "downloads": "Downloads",
    "documents": "Documents",
    "pictures": "Pictures",
    "music": "Music",
    "videos": "Videos",
}


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
        r"^(?:please\s+)?(?:can|could|would|will)\s+you\s+",
        r"^(?:please\s+)?(?:i need you to|i want you to|i want to)\s+",
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
    explicit_verbs = ("open", "show", "browse", "access", "enter")
    if any(verb in lowered for verb in explicit_verbs):
        return True
    if "go to" in lowered and ("drive" in lowered or "partition" in lowered):
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
    for key, folder_name in _SPECIAL_FOLDERS.items():
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


# ---------------------------------------------------------------------------
# Table-driven keyword matching
# ---------------------------------------------------------------------------
# Each entry: (set_of_keywords, intent, action)
# Matched against `normalized`.

_KEYWORD_TABLE = [
    # Observability
    ({"observability", "observability report", "show observability", "dashboard"},
     "OBSERVABILITY_REPORT", ""),
    # Benchmark
    ({"benchmark run", "run benchmark", "benchmark quick"},
     "BENCHMARK_COMMAND", "run"),
    ({"resilience demo", "run resilience demo", "failure demo"},
     "BENCHMARK_COMMAND", "resilience_demo"),
    # Persona
    ({"persona status", "persona show"}, "PERSONA_COMMAND", "status"),
    ({"persona list", "list personas"}, "PERSONA_COMMAND", "list"),
    ({"persona voice status"}, "PERSONA_COMMAND", "voice_status"),
    ({"assistant mode", "assistant mode on"},
     "PERSONA_COMMAND", "set", {"profile": "assistant"}),
    # Voice
    ({"voice status", "speech status"}, "VOICE_COMMAND", "status"),
    ({"voice clone on", "enable voice clone"}, "VOICE_COMMAND", "clone_on"),
    ({"voice clone off", "disable voice clone"}, "VOICE_COMMAND", "clone_off"),
    ({"stop speaking", "interrupt speech", "be quiet", "stop talking"},
     "VOICE_COMMAND", "interrupt"),
    ({"speech on", "enable speech"}, "VOICE_COMMAND", "speech_on"),
    ({"speech off", "disable speech"}, "VOICE_COMMAND", "speech_off"),
    # Knowledge base
    ({"kb status", "knowledge status", "knowledge base status"},
     "KNOWLEDGE_BASE_COMMAND", "status"),
    ({"kb quality", "knowledge quality", "kb quality report"},
     "KNOWLEDGE_BASE_COMMAND", "quality"),
    ({"kb clear", "knowledge clear"}, "KNOWLEDGE_BASE_COMMAND", "clear"),
    ({"kb retrieval on", "knowledge retrieval on"},
     "KNOWLEDGE_BASE_COMMAND", "retrieval_on"),
    ({"kb retrieval off", "knowledge retrieval off"},
     "KNOWLEDGE_BASE_COMMAND", "retrieval_off"),
    # Memory
    ({"memory status", "session memory status"}, "MEMORY_COMMAND", "status"),
    ({"memory clear", "session memory clear"}, "MEMORY_COMMAND", "clear"),
    ({"memory on", "enable memory"}, "MEMORY_COMMAND", "on"),
    ({"memory off", "disable memory"}, "MEMORY_COMMAND", "off"),
    ({"memory show", "show memory"}, "MEMORY_COMMAND", "show"),
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
    ({"undo", "rollback", "undo last action"}, "OS_ROLLBACK", ""),
    # File nav
    ({"current directory", "pwd"}, "OS_FILE_NAVIGATION", "pwd"),
    ({"list drives", "drive list"}, "OS_FILE_NAVIGATION", "list_drives"),
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
    (re.compile(r"^persona voice clone\s+([a-z0-9_-]+)\s+(on|off)$"), False,
     "PERSONA_COMMAND", "set_profile_clone_enabled",
     lambda m: {"profile": m.group(1), "enabled": m.group(2) == "on"}),
    (re.compile(r"^persona voice provider\s+([a-z0-9_-]+)\s+(xtts|voicecraft)$"), False,
     "PERSONA_COMMAND", "set_profile_clone_provider",
     lambda m: {"profile": m.group(1), "provider": m.group(2)}),
    (re.compile(r"^persona voice reference\s+([a-z0-9_-]+)\s+(.+)$", re.IGNORECASE), True,
     "PERSONA_COMMAND", "set_profile_clone_reference",
     lambda m: {"profile": m.group(1).strip().lower(), "path": m.group(2).strip()}),
    (re.compile(r"^persona set\s+([a-z0-9_-]+)$"), False,
     "PERSONA_COMMAND", "set",
     lambda m: {"profile": m.group(1)}),
    # Voice
    (re.compile(r"^voice clone provider\s+(xtts|voicecraft)$"), False,
     "VOICE_COMMAND", "set_provider",
     lambda m: {"provider": m.group(1)}),
    (re.compile(r"^voice clone reference\s+(.+)$", re.IGNORECASE), True,
     "VOICE_COMMAND", "set_reference",
     lambda m: {"path": m.group(1).strip()}),
    # Knowledge base
    (re.compile(r"^(?:kb sync|knowledge sync)\s+(.+)$", re.IGNORECASE), True,
     "KNOWLEDGE_BASE_COMMAND", "sync_dir",
     lambda m: {"path": m.group(1).strip()}),
    (re.compile(r"^(?:kb add|knowledge add)\s+(.+)$", re.IGNORECASE), True,
     "KNOWLEDGE_BASE_COMMAND", "add_file",
     lambda m: {"path": m.group(1).strip()}),
    (re.compile(r"^(?:kb index|knowledge index)\s+(.+)$", re.IGNORECASE), True,
     "KNOWLEDGE_BASE_COMMAND", "index_dir",
     lambda m: {"path": m.group(1).strip()}),
    (re.compile(r"^(?:kb search|knowledge search)\s+(.+)$", re.IGNORECASE), True,
     "KNOWLEDGE_BASE_COMMAND", "search",
     lambda m: {"query": m.group(1).strip()}),
    # Audit
    (re.compile(r"^show audit log(?:\s+(\d+))?$"), False,
     "AUDIT_LOG_REPORT", "",
     lambda m: {"limit": int(m.group(1)) if m.group(1) else 10}),
    # Policy
    (re.compile(r"^policy profile\s+([a-z0-9_-]+)$"), False,
     "POLICY_COMMAND", "set_profile",
     lambda m: {"profile": m.group(1)}),
    (re.compile(r"^policy (?:read only|readonly)\s+(on|off)$"), False,
     "POLICY_COMMAND", "set_read_only",
     lambda m: {"enabled": m.group(1) == "on"}),
    (re.compile(r"^policy permission\s+([a-z_]+)\s+(on|off)$"), False,
     "POLICY_COMMAND", "set_permission",
     lambda m: {"permission": m.group(1), "enabled": m.group(2) == "on"}),
    # Batch
    (re.compile(r"^batch add\s+(.+)$", re.IGNORECASE), True,
     "BATCH_COMMAND", "add",
     lambda m: {"command_text": m.group(1).strip()}),
    # Search index
    (re.compile(r"^index refresh(?:\s+in\s+(.+))?$", re.IGNORECASE), True,
     "SEARCH_INDEX_COMMAND", "refresh",
     lambda m: {"root": (m.group(1) or "").strip() or None}),
    (re.compile(r"^(?:indexed find|index find|search indexed)\s+(.+?)(?:\s+in\s+(.+))?$", re.IGNORECASE), True,
     "SEARCH_INDEX_COMMAND", "search",
     lambda m: {"query": m.group(1).strip(), "root": (m.group(2) or "").strip() or None}),
    # Job queue
    (re.compile(r"^(?:queue job|job add)\s+in\s+(\d+)\s*(?:s|sec|secs|seconds)?\s+(.+)$", re.IGNORECASE), True,
     "JOB_QUEUE_COMMAND", "enqueue",
     lambda m: {"delay_seconds": int(m.group(1)), "command_text": m.group(2).strip()}),
    (re.compile(r"^(?:queue job|job add)\s+(.+)$", re.IGNORECASE), True,
     "JOB_QUEUE_COMMAND", "enqueue",
     lambda m: {"delay_seconds": 0, "command_text": m.group(1).strip()}),
    (re.compile(r"^job status\s+(\d+)$"), False,
     "JOB_QUEUE_COMMAND", "status",
     lambda m: {"job_id": int(m.group(1))}),
    (re.compile(r"^job cancel\s+(\d+)$"), False,
     "JOB_QUEUE_COMMAND", "cancel",
     lambda m: {"job_id": int(m.group(1))}),
    (re.compile(r"^job retry\s+(\d+)(?:\s+in\s+(\d+)\s*(?:s|sec|secs|seconds)?)?$"), False,
     "JOB_QUEUE_COMMAND", "retry",
     lambda m: {"job_id": int(m.group(1)), "delay_seconds": int(m.group(2) or 0)}),
    (re.compile(r"^job list(?:\s+([a-z]+|\d+))?(?:\s+(\d+))?$"), False,
     "JOB_QUEUE_COMMAND", "list",
     lambda m: _parse_job_list_args(m)),
    # Confirmation
    (re.compile(r"^confirm\s+([0-9a-f]{6})(?:\s+(?:with\s+)?(.+))?$", re.IGNORECASE), True,
     "OS_CONFIRMATION", "",
     lambda m: {"token": m.group(1).lower(), "second_factor": (m.group(2) or "").strip() or None}),
    # File search
    (re.compile(r"^find file\s+(.+?)(?:\s+in\s+(.+))?$", re.IGNORECASE), True,
     "OS_FILE_SEARCH", "",
     lambda m: {"filename": m.group(1).strip(), "search_path": (m.group(2) or "").strip() or None}),
    # File nav — regex-based
    (re.compile(r"^(?:list files|list directory|show files|show directory)(?:\s+in\s+(.+))?$", re.IGNORECASE), True,
     "OS_FILE_NAVIGATION", "list_directory",
     lambda m: {"path": (m.group(1) or "").strip() or None}),
    (re.compile(r"^(?:dir|ls)(?:\s+(.+))?$", re.IGNORECASE), True,
     "OS_FILE_NAVIGATION", "list_directory",
     lambda m: {"path": (m.group(1) or "").strip() or None}),
    (re.compile(r"^(?:file info|metadata)\s+(.+)$", re.IGNORECASE), True,
     "OS_FILE_NAVIGATION", "file_info",
     lambda m: {"path": m.group(1).strip()}),
    (re.compile(r"^(?:create folder|make folder|mkdir)\s+(.+)$", re.IGNORECASE), True,
     "OS_FILE_NAVIGATION", "create_directory",
     lambda m: {"path": m.group(1).strip()}),
    (re.compile(r"^(?:delete|remove)\s+(.+)$", re.IGNORECASE), True,
     "OS_FILE_NAVIGATION", "delete_item",
     lambda m: {"path": m.group(1).strip()}),
    (re.compile(r"^move\s+(.+?)\s+to\s+(.+)$", re.IGNORECASE), True,
     "OS_FILE_NAVIGATION", "move_item",
     lambda m: {"source": m.group(1).strip(), "destination": m.group(2).strip()}),
    (re.compile(r"^rename\s+(.+?)\s+to\s+(.+)$", re.IGNORECASE), True,
     "OS_FILE_NAVIGATION", "rename_item",
     lambda m: {"source": m.group(1).strip(), "new_name": m.group(2).strip()}),
    # Open app explicit
    (re.compile(r"^open app\s+(.+)$", re.IGNORECASE), True,
     "OS_APP_OPEN", "",
     lambda m: {"app_name": m.group(1).strip()}),
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
            "OS_FILE_NAVIGATION", raw, normalized,
            action="list_directory", args={"path": f"{drive_letter}:\\"},
        )
    return None


def _try_open_command(raw, normalized):
    open_match = re.match(r"^open\s+(.+)$", raw, flags=re.IGNORECASE)
    if not open_match:
        return None

    target_raw = open_match.group(1).strip()
    target_for_match = _strip_open_fillers(_normalize_for_match(target_raw))

    drive_from_target = _extract_drive_letter(target_for_match)
    if drive_from_target and _is_drive_open_request(f"open {target_for_match}"):
        return ParsedCommand(
            "OS_FILE_NAVIGATION", raw, normalized,
            action="list_directory", args={"path": f"{drive_from_target}:\\"},
        )

    special_folder = _special_folder_path(target_for_match)
    if special_folder:
        return ParsedCommand(
            "OS_FILE_NAVIGATION", raw, normalized,
            action="list_directory", args={"path": special_folder},
        )

    if _looks_like_filesystem_target(target_for_match):
        target_path = target_raw
        if target_path.lower().startswith("the "):
            target_path = target_path[4:].strip()
        return ParsedCommand(
            "OS_FILE_NAVIGATION", raw, normalized,
            action="list_directory", args={"path": target_path},
        )

    return ParsedCommand(
        "OS_APP_OPEN", raw, normalized,
        args={"app_name": target_raw},
    )


def _try_system_action(normalized_match, normalized, raw):
    system_action = (
        normalize_system_action(normalized_match)
        or normalize_system_action(normalized)
    )
    if system_action:
        return ParsedCommand(
            "OS_SYSTEM_COMMAND", raw, normalized,
            args={"action_key": system_action},
        )
    return None


def _try_cd_commands(normalized, raw):
    if normalized.startswith("go to "):
        return ParsedCommand(
            "OS_FILE_NAVIGATION", raw, normalized,
            action="cd", args={"path": raw[6:].strip()},
        )
    if normalized.startswith("change directory "):
        return ParsedCommand(
            "OS_FILE_NAVIGATION", raw, normalized,
            action="cd", args={"path": raw[len("change directory "):].strip()},
        )
    if normalized.startswith("cd "):
        return ParsedCommand(
            "OS_FILE_NAVIGATION", raw, normalized,
            action="cd", args={"path": raw[3:].strip()},
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

    # Try stripping spoken prefixes and re-parsing
    if spoken_candidate and spoken_candidate != normalized_match:
        nested = parse_command(spoken_candidate)
        if nested.intent != "LLM_QUERY":
            return ParsedCommand(
                nested.intent, raw, normalized,
                action=nested.action, args=dict(nested.args),
            )

    # 1. Keyword table (exact match on normalized)
    result = _try_keyword_table(normalized, raw)
    if result:
        return result

    # 2. Regex table
    result = _try_regex_table(normalized, raw)
    if result:
        return result

    # 3. Drive open heuristic
    result = _try_drive_open(normalized_match, raw, normalized)
    if result:
        return result

    # 4. "open ..." disambiguation
    result = _try_open_command(raw, normalized)
    if result:
        return result

    # 5. System action aliases
    result = _try_system_action(normalized_match, normalized, raw)
    if result:
        return result

    # 6. CD / navigation commands
    result = _try_cd_commands(normalized, raw)
    if result:
        return result

    # 7. LLM fallback
    return ParsedCommand("LLM_QUERY", raw, normalized)
