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


def parse_command(text: str) -> ParsedCommand:
    raw = text or ""
    normalized = " ".join(raw.lower().split()).strip()
    normalized_match = _normalize_for_match(raw)
    spoken_candidate = _strip_spoken_prefixes(normalized_match)

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

    if normalized in {"observability", "observability report", "show observability", "dashboard"}:
        return ParsedCommand("OBSERVABILITY_REPORT", raw, normalized)

    if normalized in {"benchmark run", "run benchmark", "benchmark quick"}:
        return ParsedCommand("BENCHMARK_COMMAND", raw, normalized, action="run")
    if normalized in {"resilience demo", "run resilience demo", "failure demo"}:
        return ParsedCommand("BENCHMARK_COMMAND", raw, normalized, action="resilience_demo")

    if normalized in {"persona status", "persona show"}:
        return ParsedCommand("PERSONA_COMMAND", raw, normalized, action="status")
    if normalized in {"persona list", "list personas"}:
        return ParsedCommand("PERSONA_COMMAND", raw, normalized, action="list")
    if normalized in {"persona voice status"}:
        return ParsedCommand("PERSONA_COMMAND", raw, normalized, action="voice_status")
    persona_voice_clone_match = re.match(
        r"^persona voice clone\s+([a-z0-9_-]+)\s+(on|off)$",
        normalized,
    )
    if persona_voice_clone_match:
        return ParsedCommand(
            "PERSONA_COMMAND",
            raw,
            normalized,
            action="set_profile_clone_enabled",
            args={
                "profile": persona_voice_clone_match.group(1),
                "enabled": persona_voice_clone_match.group(2) == "on",
            },
        )
    persona_voice_provider_match = re.match(
        r"^persona voice provider\s+([a-z0-9_-]+)\s+(xtts|voicecraft)$",
        normalized,
    )
    if persona_voice_provider_match:
        return ParsedCommand(
            "PERSONA_COMMAND",
            raw,
            normalized,
            action="set_profile_clone_provider",
            args={
                "profile": persona_voice_provider_match.group(1),
                "provider": persona_voice_provider_match.group(2),
            },
        )
    persona_voice_ref_match = re.match(
        r"^persona voice reference\s+([a-z0-9_-]+)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if persona_voice_ref_match:
        return ParsedCommand(
            "PERSONA_COMMAND",
            raw,
            normalized,
            action="set_profile_clone_reference",
            args={
                "profile": persona_voice_ref_match.group(1).strip().lower(),
                "path": persona_voice_ref_match.group(2).strip(),
            },
        )
    if normalized in {"assistant mode", "assistant mode on"}:
        return ParsedCommand("PERSONA_COMMAND", raw, normalized, action="set", args={"profile": "assistant"})
    persona_set_match = re.match(r"^persona set\s+([a-z0-9_-]+)$", normalized)
    if persona_set_match:
        return ParsedCommand(
            "PERSONA_COMMAND",
            raw,
            normalized,
            action="set",
            args={"profile": persona_set_match.group(1)},
        )

    if normalized in {"voice status", "speech status"}:
        return ParsedCommand("VOICE_COMMAND", raw, normalized, action="status")
    if normalized in {"voice clone on", "enable voice clone"}:
        return ParsedCommand("VOICE_COMMAND", raw, normalized, action="clone_on")
    if normalized in {"voice clone off", "disable voice clone"}:
        return ParsedCommand("VOICE_COMMAND", raw, normalized, action="clone_off")
    voice_provider_match = re.match(r"^voice clone provider\s+(xtts|voicecraft)$", normalized)
    if voice_provider_match:
        return ParsedCommand(
            "VOICE_COMMAND",
            raw,
            normalized,
            action="set_provider",
            args={"provider": voice_provider_match.group(1)},
        )
    voice_ref_match = re.match(r"^voice clone reference\s+(.+)$", raw, flags=re.IGNORECASE)
    if voice_ref_match:
        return ParsedCommand(
            "VOICE_COMMAND",
            raw,
            normalized,
            action="set_reference",
            args={"path": voice_ref_match.group(1).strip()},
        )
    if normalized in {"stop speaking", "interrupt speech", "be quiet", "stop talking"}:
        return ParsedCommand("VOICE_COMMAND", raw, normalized, action="interrupt")
    if normalized in {"speech on", "enable speech"}:
        return ParsedCommand("VOICE_COMMAND", raw, normalized, action="speech_on")
    if normalized in {"speech off", "disable speech"}:
        return ParsedCommand("VOICE_COMMAND", raw, normalized, action="speech_off")

    if normalized in {"kb status", "knowledge status", "knowledge base status"}:
        return ParsedCommand("KNOWLEDGE_BASE_COMMAND", raw, normalized, action="status")
    kb_sync_match = re.match(r"^(?:kb sync|knowledge sync)\s+(.+)$", raw, flags=re.IGNORECASE)
    if kb_sync_match:
        return ParsedCommand(
            "KNOWLEDGE_BASE_COMMAND",
            raw,
            normalized,
            action="sync_dir",
            args={"path": kb_sync_match.group(1).strip()},
        )
    kb_add_match = re.match(r"^(?:kb add|knowledge add)\s+(.+)$", raw, flags=re.IGNORECASE)
    if kb_add_match:
        return ParsedCommand(
            "KNOWLEDGE_BASE_COMMAND",
            raw,
            normalized,
            action="add_file",
            args={"path": kb_add_match.group(1).strip()},
        )
    kb_index_match = re.match(r"^(?:kb index|knowledge index)\s+(.+)$", raw, flags=re.IGNORECASE)
    if kb_index_match:
        return ParsedCommand(
            "KNOWLEDGE_BASE_COMMAND",
            raw,
            normalized,
            action="index_dir",
            args={"path": kb_index_match.group(1).strip()},
        )
    kb_search_match = re.match(r"^(?:kb search|knowledge search)\s+(.+)$", raw, flags=re.IGNORECASE)
    if kb_search_match:
        return ParsedCommand(
            "KNOWLEDGE_BASE_COMMAND",
            raw,
            normalized,
            action="search",
            args={"query": kb_search_match.group(1).strip()},
        )
    if normalized in {"kb quality", "knowledge quality", "kb quality report"}:
        return ParsedCommand("KNOWLEDGE_BASE_COMMAND", raw, normalized, action="quality")
    if normalized in {"kb clear", "knowledge clear"}:
        return ParsedCommand("KNOWLEDGE_BASE_COMMAND", raw, normalized, action="clear")
    if normalized in {"kb retrieval on", "knowledge retrieval on"}:
        return ParsedCommand("KNOWLEDGE_BASE_COMMAND", raw, normalized, action="retrieval_on")
    if normalized in {"kb retrieval off", "knowledge retrieval off"}:
        return ParsedCommand("KNOWLEDGE_BASE_COMMAND", raw, normalized, action="retrieval_off")

    if normalized in {"memory status", "session memory status"}:
        return ParsedCommand("MEMORY_COMMAND", raw, normalized, action="status")
    if normalized in {"memory clear", "session memory clear"}:
        return ParsedCommand("MEMORY_COMMAND", raw, normalized, action="clear")
    if normalized in {"memory on", "enable memory"}:
        return ParsedCommand("MEMORY_COMMAND", raw, normalized, action="on")
    if normalized in {"memory off", "disable memory"}:
        return ParsedCommand("MEMORY_COMMAND", raw, normalized, action="off")
    if normalized in {"memory show", "show memory"}:
        return ParsedCommand("MEMORY_COMMAND", raw, normalized, action="show")

    if normalized in {"demo mode on", "demo on"}:
        return ParsedCommand("DEMO_MODE", raw, normalized, action="on")
    if normalized in {"demo mode off", "demo off"}:
        return ParsedCommand("DEMO_MODE", raw, normalized, action="off")
    if normalized in {"demo mode status", "demo status"}:
        return ParsedCommand("DEMO_MODE", raw, normalized, action="status")

    if normalized in {"show metrics", "metrics", "metrics report"}:
        return ParsedCommand("METRICS_REPORT", raw, normalized)

    audit_match = re.match(r"^show audit log(?:\s+(\d+))?$", normalized)
    if audit_match:
        limit = int(audit_match.group(1)) if audit_match.group(1) else 10
        return ParsedCommand("AUDIT_LOG_REPORT", raw, normalized, args={"limit": limit})

    if normalized in {"verify audit", "verify audit log", "audit verify"}:
        return ParsedCommand("AUDIT_VERIFY", raw, normalized)
    if normalized in {"audit reseal", "reseal audit", "repair audit chain"}:
        return ParsedCommand("AUDIT_RESEAL", raw, normalized)

    if normalized == "policy status":
        return ParsedCommand("POLICY_COMMAND", raw, normalized, action="status")
    profile_match = re.match(r"^policy profile\s+([a-z0-9_-]+)$", normalized)
    if profile_match:
        return ParsedCommand(
            "POLICY_COMMAND",
            raw,
            normalized,
            action="set_profile",
            args={"profile": profile_match.group(1)},
        )
    readonly_match = re.match(r"^policy (?:read only|readonly)\s+(on|off)$", normalized)
    if readonly_match:
        return ParsedCommand(
            "POLICY_COMMAND",
            raw,
            normalized,
            action="set_read_only",
            args={"enabled": readonly_match.group(1) == "on"},
        )
    permission_match = re.match(r"^policy permission\s+([a-z_]+)\s+(on|off)$", normalized)
    if permission_match:
        return ParsedCommand(
            "POLICY_COMMAND",
            raw,
            normalized,
            action="set_permission",
            args={
                "permission": permission_match.group(1),
                "enabled": permission_match.group(2) == "on",
            },
        )

    if normalized in {"batch plan", "batch start", "batch begin"}:
        return ParsedCommand("BATCH_COMMAND", raw, normalized, action="plan")
    batch_add_match = re.match(r"^batch add\s+(.+)$", raw, flags=re.IGNORECASE)
    if batch_add_match:
        return ParsedCommand(
            "BATCH_COMMAND",
            raw,
            normalized,
            action="add",
            args={"command_text": batch_add_match.group(1).strip()},
        )
    if normalized in {"batch preview", "batch show"}:
        return ParsedCommand("BATCH_COMMAND", raw, normalized, action="preview")
    if normalized in {"batch status"}:
        return ParsedCommand("BATCH_COMMAND", raw, normalized, action="status")
    if normalized in {"batch commit", "batch run"}:
        return ParsedCommand("BATCH_COMMAND", raw, normalized, action="commit")
    if normalized in {"batch abort", "batch cancel", "batch clear"}:
        return ParsedCommand("BATCH_COMMAND", raw, normalized, action="abort")

    if normalized in {"index status", "search index status"}:
        return ParsedCommand("SEARCH_INDEX_COMMAND", raw, normalized, action="status")
    if normalized in {"index start", "start index"}:
        return ParsedCommand("SEARCH_INDEX_COMMAND", raw, normalized, action="start")
    index_refresh_match = re.match(r"^index refresh(?:\s+in\s+(.+))?$", raw, flags=re.IGNORECASE)
    if index_refresh_match:
        return ParsedCommand(
            "SEARCH_INDEX_COMMAND",
            raw,
            normalized,
            action="refresh",
            args={"root": (index_refresh_match.group(1) or "").strip() or None},
        )
    indexed_search_match = re.match(
        r"^(?:indexed find|index find|search indexed)\s+(.+?)(?:\s+in\s+(.+))?$",
        raw,
        flags=re.IGNORECASE,
    )
    if indexed_search_match:
        return ParsedCommand(
            "SEARCH_INDEX_COMMAND",
            raw,
            normalized,
            action="search",
            args={
                "query": indexed_search_match.group(1).strip(),
                "root": (indexed_search_match.group(2) or "").strip() or None,
            },
        )

    queue_delayed_match = re.match(
        r"^(?:queue job|job add)\s+in\s+(\d+)\s*(?:s|sec|secs|seconds)?\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if queue_delayed_match:
        return ParsedCommand(
            "JOB_QUEUE_COMMAND",
            raw,
            normalized,
            action="enqueue",
            args={
                "delay_seconds": int(queue_delayed_match.group(1)),
                "command_text": queue_delayed_match.group(2).strip(),
            },
        )
    queue_match = re.match(r"^(?:queue job|job add)\s+(.+)$", raw, flags=re.IGNORECASE)
    if queue_match:
        return ParsedCommand(
            "JOB_QUEUE_COMMAND",
            raw,
            normalized,
            action="enqueue",
            args={"delay_seconds": 0, "command_text": queue_match.group(1).strip()},
        )
    job_status_match = re.match(r"^job status\s+(\d+)$", normalized)
    if job_status_match:
        return ParsedCommand(
            "JOB_QUEUE_COMMAND",
            raw,
            normalized,
            action="status",
            args={"job_id": int(job_status_match.group(1))},
        )
    job_cancel_match = re.match(r"^job cancel\s+(\d+)$", normalized)
    if job_cancel_match:
        return ParsedCommand(
            "JOB_QUEUE_COMMAND",
            raw,
            normalized,
            action="cancel",
            args={"job_id": int(job_cancel_match.group(1))},
        )
    job_retry_match = re.match(
        r"^job retry\s+(\d+)(?:\s+in\s+(\d+)\s*(?:s|sec|secs|seconds)?)?$",
        normalized,
    )
    if job_retry_match:
        return ParsedCommand(
            "JOB_QUEUE_COMMAND",
            raw,
            normalized,
            action="retry",
            args={
                "job_id": int(job_retry_match.group(1)),
                "delay_seconds": int(job_retry_match.group(2) or 0),
            },
        )
    if normalized in {"job worker start"}:
        return ParsedCommand("JOB_QUEUE_COMMAND", raw, normalized, action="worker_start")
    if normalized in {"job worker stop"}:
        return ParsedCommand("JOB_QUEUE_COMMAND", raw, normalized, action="worker_stop")
    if normalized in {"job worker status"}:
        return ParsedCommand("JOB_QUEUE_COMMAND", raw, normalized, action="worker_status")
    job_list_match = re.match(r"^job list(?:\s+([a-z]+|\d+))?(?:\s+(\d+))?$", normalized)
    if job_list_match:
        first = job_list_match.group(1)
        second = job_list_match.group(2)
        status = None
        limit = 10
        if first:
            if first.isdigit():
                limit = int(first)
            else:
                status = first
        if second:
            limit = int(second)
        return ParsedCommand(
            "JOB_QUEUE_COMMAND",
            raw,
            normalized,
            action="list",
            args={"status": status, "limit": limit},
        )

    confirm_match = re.match(
        r"^confirm\s+([0-9a-f]{6})(?:\s+(?:with\s+)?(.+))?$",
        raw,
        flags=re.IGNORECASE,
    )
    if confirm_match:
        secret = (confirm_match.group(2) or "").strip() or None
        return ParsedCommand(
            "OS_CONFIRMATION",
            raw,
            normalized,
            args={"token": confirm_match.group(1).lower(), "second_factor": secret},
        )

    if normalized in {"undo", "rollback", "undo last action"}:
        return ParsedCommand("OS_ROLLBACK", raw, normalized)

    search_match = re.match(r"^find file\s+(.+?)(?:\s+in\s+(.+))?$", raw, flags=re.IGNORECASE)
    if search_match:
        return ParsedCommand(
            "OS_FILE_SEARCH",
            raw,
            normalized,
            args={
                "filename": search_match.group(1).strip(),
                "search_path": (search_match.group(2) or "").strip() or None,
            },
        )

    drive_letter = _extract_drive_letter(normalized_match)
    if drive_letter and _is_drive_open_request(normalized_match):
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="list_directory",
            args={"path": f"{drive_letter}:\\"},
        )

    open_app_match = re.match(r"^open app\s+(.+)$", raw, flags=re.IGNORECASE)
    if open_app_match:
        return ParsedCommand(
            "OS_APP_OPEN",
            raw,
            normalized,
            args={"app_name": open_app_match.group(1).strip()},
        )
    open_match = re.match(r"^open\s+(.+)$", raw, flags=re.IGNORECASE)
    if open_match:
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
            return ParsedCommand(
                "OS_FILE_NAVIGATION",
                raw,
                normalized,
                action="list_directory",
                args={"path": target_path},
            )

        return ParsedCommand(
            "OS_APP_OPEN",
            raw,
            normalized,
            args={"app_name": target_raw},
        )

    system_action = (
        normalize_system_action(normalized_match)
        or normalize_system_action(normalized)
    )
    if system_action:
        return ParsedCommand(
            "OS_SYSTEM_COMMAND",
            raw,
            normalized,
            args={"action_key": system_action},
        )

    if normalized in {"current directory", "pwd"}:
        return ParsedCommand("OS_FILE_NAVIGATION", raw, normalized, action="pwd")

    if normalized.startswith("go to "):
        return ParsedCommand("OS_FILE_NAVIGATION", raw, normalized, action="cd", args={"path": raw[6:].strip()})
    if normalized.startswith("change directory "):
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="cd",
            args={"path": raw[len("change directory ") :].strip()},
        )
    if normalized.startswith("cd "):
        return ParsedCommand("OS_FILE_NAVIGATION", raw, normalized, action="cd", args={"path": raw[3:].strip()})

    if normalized in {"list drives", "drive list"}:
        return ParsedCommand("OS_FILE_NAVIGATION", raw, normalized, action="list_drives")

    list_match = re.match(
        r"^(?:list files|list directory|show files|show directory)(?:\s+in\s+(.+))?$",
        raw,
        flags=re.IGNORECASE,
    )
    if list_match:
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="list_directory",
            args={"path": (list_match.group(1) or "").strip() or None},
        )

    dir_match = re.match(r"^(?:dir|ls)(?:\s+(.+))?$", raw, flags=re.IGNORECASE)
    if dir_match:
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="list_directory",
            args={"path": (dir_match.group(1) or "").strip() or None},
        )

    info_match = re.match(r"^(?:file info|metadata)\s+(.+)$", raw, flags=re.IGNORECASE)
    if info_match:
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="file_info",
            args={"path": info_match.group(1).strip()},
        )

    mkdir_match = re.match(r"^(?:create folder|make folder|mkdir)\s+(.+)$", raw, flags=re.IGNORECASE)
    if mkdir_match:
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="create_directory",
            args={"path": mkdir_match.group(1).strip()},
        )

    delete_match = re.match(r"^(?:delete|remove)\s+(.+)$", raw, flags=re.IGNORECASE)
    if delete_match:
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="delete_item",
            args={"path": delete_match.group(1).strip()},
        )

    move_match = re.match(r"^move\s+(.+?)\s+to\s+(.+)$", raw, flags=re.IGNORECASE)
    if move_match:
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="move_item",
            args={"source": move_match.group(1).strip(), "destination": move_match.group(2).strip()},
        )

    rename_match = re.match(r"^rename\s+(.+?)\s+to\s+(.+)$", raw, flags=re.IGNORECASE)
    if rename_match:
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="rename_item",
            args={"source": rename_match.group(1).strip(), "new_name": rename_match.group(2).strip()},
        )

    return ParsedCommand("LLM_QUERY", raw, normalized)
