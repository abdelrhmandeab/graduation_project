import ctypes
import os
import re
import shutil
import time
import uuid

from core.config import (
    ALLOW_PERMANENT_DELETE,
    CONFIRMATION_TIMEOUT_SECONDS,
    DEFAULT_SEARCH_PATH,
    DEFAULT_WORKING_DIRECTORY,
    FILE_DEFAULT_SEARCH_ROOTS,
    FILE_HUMANIZE_PATHS,
    FILE_SPOKEN_RESULTS_MAX,
    MAX_FILE_RESULTS,
    ROLLBACK_DIR_NAME,
    SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE,
)
from os_control.path_resolver import humanize_path, resolve_location, KNOWN_FOLDERS
from core.logger import logger
from core.response_templates import format_confirmation_prompt
from os_control.action_log import log_action
from os_control.adapter_result import (
    confirmation_result,
    failure_result,
    success_result,
    to_legacy_pair,
)
from os_control.confirmation import confirmation_manager
from os_control.persistence import (
    pop_latest_rollback_action,
    push_rollback_action,
    restore_rollback_action,
)
from os_control.policy import policy_engine
from os_control.risk_policy import risk_tier_for_file_operation

_current_directory = os.path.abspath(DEFAULT_WORKING_DIRECTORY)

_INVALID_PATH_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1F]")
_INVALID_NAME_CHAR_RE = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
_RESERVED_WINDOWS_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    "com1",
    "com2",
    "com3",
    "com4",
    "com5",
    "com6",
    "com7",
    "com8",
    "com9",
    "lpt1",
    "lpt2",
    "lpt3",
    "lpt4",
    "lpt5",
    "lpt6",
    "lpt7",
    "lpt8",
    "lpt9",
}


def _ensure_rollback_dir():
    temp_root = os.environ.get("TEMP") or os.getcwd()
    rollback_root = os.path.join(temp_root, ROLLBACK_DIR_NAME)
    os.makedirs(rollback_root, exist_ok=True)
    return rollback_root


def _validate_raw_path_input(path_value, label):
    if path_value is None:
        return False, f"{label} cannot be empty.", None
    raw = str(path_value).strip()
    if not raw:
        return False, f"{label} cannot be empty.", None
    if _INVALID_PATH_CONTROL_CHAR_RE.search(raw):
        return False, f"{label} contains unsupported control characters.", None
    return True, "", raw


def _validate_windows_name(name_value, label):
    name = (name_value or "").strip().strip(".")
    if not name:
        return False, f"{label} cannot be empty.", None
    if _INVALID_NAME_CHAR_RE.search(name):
        return False, f"{label} contains unsupported characters.", None
    stem = os.path.splitext(name)[0].lower()
    if stem in _RESERVED_WINDOWS_NAMES:
        return False, f"{label} uses a reserved Windows name.", None
    return True, "", name


def _validate_path_segments(path_value, label):
    cleaned = str(path_value or "").strip().strip('"').strip("'")
    if not cleaned:
        return False, f"{label} cannot be empty."
    parts = re.split(r"[\\/]+", cleaned)
    for part in parts:
        if not part or part in {".", ".."}:
            continue
        if re.fullmatch(r"[a-zA-Z]:", part):
            continue
        ok, reason, _name = _validate_windows_name(part, label)
        if not ok:
            return False, reason
    return True, ""


def _is_subpath(path_value, parent_value):
    path_abs = os.path.abspath(path_value)
    parent_abs = os.path.abspath(parent_value)
    if path_abs == parent_abs:
        return True
    return path_abs.startswith(parent_abs + os.sep)


def _resolve_path(path):
    if not path:
        return _current_directory
    cleaned = path.strip().strip('"').strip("'")
    # Try spoken-location resolution first (Desktop, Downloads, C partition …).
    resolved = resolve_location(cleaned)
    if resolved is not None:
        return str(resolved)
    cleaned = os.path.expanduser(cleaned)
    if os.path.isabs(cleaned):
        return os.path.abspath(cleaned)
    return os.path.abspath(os.path.join(_current_directory, cleaned))


def _check_path_policy(path, write=False):
    allowed, reason = policy_engine.can_access_path(path, write=write)
    if not allowed:
        return False, reason
    return True, ""


def _validate_file_write_enabled():
    if not policy_engine.is_command_allowed("file_write"):
        return False, "File write operations are disabled by policy."
    return True, ""


def _risk_tier_for_operation(operation):
    return risk_tier_for_file_operation(operation)


def _prepare_move_paths(source, destination):
    src_raw_ok, src_raw_reason, src_raw = _validate_raw_path_input(source, "Source path")
    if not src_raw_ok:
        return False, src_raw_reason, None, None
    dst_raw_ok, dst_raw_reason, dst_raw = _validate_raw_path_input(destination, "Destination path")
    if not dst_raw_ok:
        return False, dst_raw_reason, None, None
    src_segments_ok, src_segments_reason = _validate_path_segments(src_raw, "Source path")
    if not src_segments_ok:
        return False, src_segments_reason, None, None
    dst_segments_ok, dst_segments_reason = _validate_path_segments(dst_raw, "Destination path")
    if not dst_segments_ok:
        return False, dst_segments_reason, None, None

    src = _resolve_path(src_raw)
    dst = _resolve_path(dst_raw)
    src_ok, src_reason = _check_path_policy(src, write=True)
    if not src_ok:
        return False, src_reason, None, None
    dst_ok, dst_reason = _check_path_policy(dst, write=True)
    if not dst_ok:
        return False, dst_reason, None, None
    if not os.path.exists(src):
        return False, f"Source does not exist: {src}", None, None

    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))

    if os.path.exists(dst):
        return False, f"Destination already exists: {dst}", None, None
    if src == dst:
        return False, "Source and destination are the same path.", None, None
    if os.path.isdir(src) and _is_subpath(dst, src):
        return False, "Destination cannot be inside the source directory.", None, None
    return True, "", src, dst


def resolve_name_in_location(name, location):
    """Resolve a bare file/folder name within a specific directory.

    Used when the user says "delete X from Y" / "rename X in Y" instead of
    giving an exact path — X is searched for inside Y rather than treated
    as a literal subpath of the current directory.

    Returns (status, value):
      "single"    -> value is the resolved absolute path
      "ambiguous" -> value is a list of candidate absolute paths
      "not_found" -> value is None
    """
    if not location or not os.path.isdir(location):
        return "not_found", None

    name = str(name or "").strip()
    if not name:
        return "not_found", None

    # Exact match (file or folder) first — case-insensitive.
    try:
        entries = list(os.scandir(location))
    except Exception:
        return "not_found", None

    name_lower = name.lower()
    exact = [e.path for e in entries if e.name.lower() == name_lower]
    if len(exact) == 1:
        return "single", exact[0]
    if len(exact) > 1:
        return "ambiguous", exact

    # Fuzzy match: name without extension, or name as a substring/prefix.
    name_tokens, ext_tokens = _split_file_query(name)
    candidates = [e.path for e in entries if _file_matches_query(e.name, name_tokens, ext_tokens)]
    if len(candidates) == 1:
        return "single", candidates[0]
    if len(candidates) > 1:
        return "ambiguous", candidates

    return "not_found", None


def _prepare_delete_path(path, location=None):
    raw_ok, raw_reason, raw_path = _validate_raw_path_input(path, "Path")
    if not raw_ok:
        return False, raw_reason, None
    segments_ok, segments_reason = _validate_path_segments(raw_path, "Path")
    if not segments_ok:
        return False, segments_reason, None
    target = _resolve_path(raw_path)
    ok, reason = _check_path_policy(target, write=True)
    if not ok:
        return False, reason, None
    if not os.path.exists(target) and location:
        status, value = resolve_name_in_location(raw_path, location)
        if status == "single":
            target = value
        elif status == "ambiguous":
            return False, "AMBIGUOUS", value
    if not os.path.exists(target):
        return False, f"Path does not exist: {target}", None
    return True, "", target


def _prepare_copy_paths(source, destination):
    """Validate and prepare paths for copy operation."""
    src_raw_ok, src_raw_reason, src_raw = _validate_raw_path_input(source, "Source path")
    if not src_raw_ok:
        return False, src_raw_reason, None, None
    dst_raw_ok, dst_raw_reason, dst_raw = _validate_raw_path_input(destination, "Destination path")
    if not dst_raw_ok:
        return False, dst_raw_reason, None, None
    src_segments_ok, src_segments_reason = _validate_path_segments(src_raw, "Source path")
    if not src_segments_ok:
        return False, src_segments_reason, None, None
    dst_segments_ok, dst_segments_reason = _validate_path_segments(dst_raw, "Destination path")
    if not dst_segments_ok:
        return False, dst_segments_reason, None, None

    src = _resolve_path(src_raw)
    dst = _resolve_path(dst_raw)
    src_ok, src_reason = _check_path_policy(src, write=False)
    if not src_ok:
        return False, src_reason, None, None
    dst_ok, dst_reason = _check_path_policy(dst, write=True)
    if not dst_ok:
        return False, dst_reason, None, None
    if not os.path.exists(src):
        return False, f"Source does not exist: {src}", None, None

    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))

    if os.path.exists(dst):
        return False, f"Destination already exists: {dst}", None, None
    if src == dst:
        return False, "Source and destination are the same path.", None, None
    return True, "", src, dst


def _prepare_rename_paths(source, new_name):
    source_raw_ok, source_raw_reason, source_raw = _validate_raw_path_input(source, "Source path")
    if not source_raw_ok:
        return False, source_raw_reason, None, None
    source_segments_ok, source_segments_reason = _validate_path_segments(source_raw, "Source path")
    if not source_segments_ok:
        return False, source_segments_reason, None, None

    source_abs = _resolve_path(source_raw)
    source_ok, source_reason = _check_path_policy(source_abs, write=True)
    if not source_ok:
        return False, source_reason, None, None
    if not os.path.exists(source_abs):
        return False, f"Source does not exist: {source_abs}", None, None

    name_ok, name_reason, clean_new_name = _validate_windows_name(new_name, "New name")
    if not name_ok:
        return False, name_reason, None, None
    if os.path.sep in clean_new_name:
        return False, "New name must not include path separators.", None, None
    if os.path.altsep and os.path.altsep in clean_new_name:
        return False, "New name must not include path separators.", None, None

    destination_abs = os.path.join(os.path.dirname(source_abs), clean_new_name)
    destination_ok, destination_reason = _check_path_policy(destination_abs, write=True)
    if not destination_ok:
        return False, destination_reason, None, None
    if source_abs == destination_abs:
        return False, "Source and destination are the same path.", None, None
    if os.path.exists(destination_abs):
        return False, f"Destination already exists: {destination_abs}", None, None
    return True, "", source_abs, destination_abs


def _request_file_operation_confirmation(operation, description, resolved_args):
    risk_tier = _risk_tier_for_operation(operation)
    require_second_factor = bool(
        risk_tier == "high" and SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE
    )
    payload = {
        "kind": "file_operation",
        "operation": operation,
        "resolved_args": dict(resolved_args or {}),
        "risk_tier": risk_tier,
        "require_second_factor": require_second_factor,
    }
    token = confirmation_manager.create(
        action_name=f"file_{operation}",
        description=description,
        payload=payload,
    )
    log_action(
        "file_operation_request",
        "pending",
        details={
            "operation": operation,
            "risk_tier": risk_tier,
            "token": token,
            "second_factor": require_second_factor,
            "args": payload["resolved_args"],
        },
    )

    message = format_confirmation_prompt(
        description,
        token,
        risk_tier=risk_tier,
        timeout_seconds=CONFIRMATION_TIMEOUT_SECONDS,
        require_second_factor=require_second_factor,
    )
    return confirmation_result(
        message,
        token=token,
        second_factor=require_second_factor,
        risk_tier=risk_tier,
        debug_info={
            "operation": operation,
            "resolved_args": dict(resolved_args or {}),
        },
    )


def _execute_move_item(src, dst, action_name="move_item"):
    try:
        destination_parent = os.path.dirname(dst)
        if destination_parent:
            os.makedirs(destination_parent, exist_ok=True)
        shutil.move(src, dst)
        action_id = push_rollback_action("move", {"source": dst, "destination": src})
        log_action(
            action_name,
            "success",
            details={"source": src, "destination": dst, "rollback_action_id": action_id},
            rollback_data={"rollback_action_id": action_id},
        )
        verb = "Renamed" if action_name == "rename_item" else "Moved"
        return success_result(
            f"{verb}: {src} -> {dst}",
            debug_info={"source": src, "destination": dst, "operation": action_name},
            executed_confirmed_action="file_operation",
        )
    except Exception as exc:
        log_action(action_name, "failed", details={"source": src, "destination": dst}, error=exc)
        return failure_result(
            f"Failed to execute {action_name}: {exc}",
            error_code="execution_failed",
            debug_info={"source": src, "destination": dst, "operation": action_name},
        )


def _execute_copy_item(src, dst):
    """Copy a file or directory recursively."""
    try:
        destination_parent = os.path.dirname(dst)
        if destination_parent:
            os.makedirs(destination_parent, exist_ok=True)
        
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        
        log_action(
            "copy_item",
            "success",
            details={"source": src, "destination": dst},
        )
        return success_result(
            f"Copied: {src} -> {dst}",
            debug_info={"source": src, "destination": dst, "operation": "copy_item"},
            executed_confirmed_action="file_operation",
        )
    except Exception as exc:
        log_action("copy_item", "failed", details={"source": src, "destination": dst}, error=exc)
        return failure_result(
            f"Failed to copy item: {exc}",
            error_code="execution_failed",
            debug_info={"source": src, "destination": dst, "operation": "copy_item"},
        )



def _execute_delete_item(target, permanent=False):
    operation = "delete_item_permanent" if permanent else "delete_item"
    try:
        if permanent:
            if os.path.isdir(target):
                shutil.rmtree(target)
            else:
                os.remove(target)
            log_action(
                operation,
                "success",
                details={"path": target, "permanent": True},
            )
            return success_result(
                f"Permanently deleted: {target}",
                debug_info={"path": target, "operation": operation, "permanent": True},
                executed_confirmed_action="file_operation",
            )

        rollback_root = _ensure_rollback_dir()
        backup_name = f"{uuid.uuid4().hex}_{os.path.basename(target)}"
        backup_path = os.path.join(rollback_root, backup_name)
        shutil.move(target, backup_path)

        action_id = push_rollback_action(
            "move",
            {"source": backup_path, "destination": target},
        )
        log_action(
            operation,
            "success",
            details={"path": target, "backup_path": backup_path, "rollback_action_id": action_id, "permanent": False},
            rollback_data={"rollback_action_id": action_id},
        )
        return success_result(
            f"Deleted (moved to rollback storage): {target}",
            debug_info={"path": target, "operation": operation, "permanent": False},
            executed_confirmed_action="file_operation",
        )
    except Exception as exc:
        log_action(operation, "failed", details={"path": target, "permanent": bool(permanent)}, error=exc)
        return failure_result(
            f"Failed to delete item: {exc}",
            error_code="execution_failed",
            debug_info={"path": target, "operation": operation, "permanent": bool(permanent)},
        )


def _validation_error_result(message, debug_info=None):
    return failure_result(
        message,
        error_code="validation_error",
        debug_info=dict(debug_info or {}),
    )


def _resolve_validated_path(path_value, label, allow_empty=False):
    if allow_empty and (path_value is None or not str(path_value).strip()):
        return True, "", _current_directory

    raw_ok, raw_reason, raw_path = _validate_raw_path_input(path_value, label)
    if not raw_ok:
        return False, raw_reason, None
    segments_ok, segments_reason = _validate_path_segments(raw_path, label)
    if not segments_ok:
        return False, segments_reason, None
    return True, "", _resolve_path(raw_path)


def get_current_directory():
    return _current_directory


def change_directory_result(path):
    global _current_directory

    path_ok, path_reason, target = _resolve_validated_path(path, "Path", allow_empty=True)
    if not path_ok:
        return _validation_error_result(path_reason, debug_info={"path": path})

    ok, reason = _check_path_policy(target, write=False)
    if not ok:
        return failure_result(
            reason,
            error_code="policy_blocked",
            debug_info={"path": target},
        )
    if not os.path.isdir(target):
        return failure_result(
            f"Directory does not exist: {target}",
            error_code="not_found",
            debug_info={"path": target},
        )

    _current_directory = target
    log_action("change_directory", "success", details={"new_directory": target})
    return success_result(
        f"Current directory set to: {target}",
        debug_info={"path": target},
    )


def list_directory_result(path=None, limit=50):
    path_ok, path_reason, target = _resolve_validated_path(path, "Path", allow_empty=True)
    if not path_ok:
        return _validation_error_result(path_reason, debug_info={"path": path})

    try:
        safe_limit = max(1, min(500, int(limit or 50)))
    except (TypeError, ValueError):
        safe_limit = 50
    ok, reason = _check_path_policy(target, write=False)
    if not ok:
        return failure_result(
            reason,
            error_code="policy_blocked",
            debug_info={"path": target},
        )
    if not os.path.isdir(target):
        return failure_result(
            f"Directory does not exist: {target}",
            error_code="not_found",
            debug_info={"path": target},
        )

    try:
        entries = []
        with os.scandir(target) as it:
            for entry in it:
                prefix = "[D]" if entry.is_dir() else "[F]"
                entries.append(f"{prefix} {entry.name}")
                if len(entries) >= safe_limit:
                    break
        log_action("list_directory", "success", details={"path": target, "count": len(entries)})
        if not entries:
            msg = "Directory is empty."
        else:
            loc = humanize_path(target) if FILE_HUMANIZE_PATHS else {"en": target, "ar": target}
            header = f"Contents of {loc['en']}:"
            msg = header + "\n" + "\n".join(entries)
        return success_result(
            msg,
            debug_info={"path": target, "count": len(entries), "limit": safe_limit},
        )
    except Exception as exc:
        log_action("list_directory", "failed", details={"path": target}, error=exc)
        return failure_result(
            f"Failed to list directory: {exc}",
            error_code="execution_failed",
            debug_info={"path": target},
        )


def list_drives_win32_result():
    if os.name != "nt":
        return failure_result(
            "Drive listing via Win32 is only available on Windows.",
            error_code="unsupported_platform",
        )
    if not policy_engine.is_command_allowed("file_navigation"):
        return failure_result(
            "File navigation is disabled by policy.",
            error_code="policy_blocked",
        )

    try:
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        drives = [f"{chr(65 + index)}:\\" for index in range(26) if bitmask & (1 << index)]
        log_action("list_drives", "success", details={"count": len(drives)})
        return success_result(
            "\n".join(drives) if drives else "No drives found.",
            debug_info={"count": len(drives)},
        )
    except Exception as exc:
        log_action("list_drives", "failed", error=exc)
        return failure_result(
            f"Failed to list drives: {exc}",
            error_code="execution_failed",
        )


def get_file_metadata_result(path):
    path_ok, path_reason, target = _resolve_validated_path(path, "Path", allow_empty=False)
    if not path_ok:
        return _validation_error_result(path_reason, debug_info={"path": path})

    ok, reason = _check_path_policy(target, write=False)
    if not ok:
        return failure_result(
            reason,
            error_code="policy_blocked",
            debug_info={"path": target},
        )
    if not os.path.exists(target):
        return failure_result(
            f"Path does not exist: {target}",
            error_code="not_found",
            debug_info={"path": target},
        )

    try:
        st = os.stat(target)
        metadata = [
            f"Path: {target}",
            f"Type: {'Directory' if os.path.isdir(target) else 'File'}",
            f"Size: {st.st_size} bytes",
            f"Created: {st.st_ctime}",
            f"Modified: {st.st_mtime}",
        ]
        log_action("file_metadata", "success", details={"path": target})
        return success_result("\n".join(metadata), debug_info={"path": target})
    except Exception as exc:
        log_action("file_metadata", "failed", details={"path": target}, error=exc)
        return failure_result(
            f"Failed to read metadata: {exc}",
            error_code="execution_failed",
            debug_info={"path": target},
        )


_FILE_EXTENSION_RE = re.compile(
    r"^(pdf|docx?|xlsx?|pptx?|txt|csv|json|xml|html?|jpg|jpeg|png|gif|bmp|mp3|mp4|wav|zip|rar|7z|exe|py|js|ts|md)$",
    re.IGNORECASE,
)


def _split_file_query(query: str):
    """Split query into (name_tokens, extension_tokens).

    e.g. "cv pdf" -> (["cv"], ["pdf"])
    """
    tokens = (query or "").lower().split()
    exts = [t for t in tokens if _FILE_EXTENSION_RE.match(t)]
    names = [t for t in tokens if not _FILE_EXTENSION_RE.match(t)]
    return names, exts


def _file_matches_query(filename: str, name_tokens: list, ext_tokens: list) -> bool:
    name_lower = filename.lower()
    stem, ext = os.path.splitext(name_lower)
    if name_tokens and not all(tok in name_lower for tok in name_tokens):
        return False
    if ext_tokens and not any(ext == f".{e}" or stem.endswith(e) for e in ext_tokens):
        return False
    return True


def search_windows_index(query, max_results=10):
    """Search using Windows Search Index via ADODB (built into Windows).

    Returns a list of file paths, or empty list on failure.
    Falls back gracefully if win32com is not installed or index is unavailable.
    """
    if not query:
        return []
    try:
        import win32com.client

        conn = win32com.client.Dispatch("ADODB.Connection")
        conn.Open(
            "Provider=Search.CollatorDSO;"
            "Extended Properties='Application=Windows';"
        )
        name_tokens, ext_tokens = _split_file_query(query)
        conditions = []
        for tok in name_tokens:
            safe_tok = tok.replace("'", "''")
            conditions.append(f"System.FileName LIKE '%{safe_tok}%'")
        if ext_tokens:
            ext_conds = " OR ".join(
                f"System.FileExtension = '.{e.replace(chr(39), chr(39)*2)}'" for e in ext_tokens
            )
            conditions.append(f"({ext_conds})")
        if not conditions:
            safe_query = str(query).replace("'", "''")
            conditions.append(f"System.FileName LIKE '%{safe_query}%'")
        where_clause = " AND ".join(conditions)
        sql = (
            f"SELECT TOP {int(max_results)} System.ItemPathDisplay "
            f"FROM SystemIndex "
            f"WHERE {where_clause}"
        )
        rs = conn.Execute(sql)[0]
        results = []
        while not rs.EOF:
            path = rs.Fields("System.ItemPathDisplay").Value
            if path:
                results.append(path)
            rs.MoveNext()
        conn.Close()
        logger.info("Windows Search Index returned %d results for '%s'", len(results), query)
        return results
    except ImportError:
        logger.debug("pywin32 not installed — Windows Search Index unavailable")
        return []
    except Exception as exc:
        logger.debug("Windows Search Index query failed: %s", exc)
        return []


def _search_root_paths(search_path=None):
    """Resolve search_path to a list of absolute root paths to search."""
    if search_path:
        resolved = resolve_location(search_path)
        if resolved:
            return [str(resolved)]
        return [_resolve_path(search_path)]
    # No explicit location — search the configured default roots.
    roots = []
    for name in FILE_DEFAULT_SEARCH_ROOTS:
        folder = KNOWN_FOLDERS.get(name)
        if folder and folder.is_dir():
            roots.append(str(folder))
    return roots or [DEFAULT_SEARCH_PATH]


def find_files(filename, search_path=None):
    if not policy_engine.is_command_allowed("file_search"):
        return []
    if not filename:
        return []

    roots = _search_root_paths(search_path)
    # Use the first valid root for legacy index queries; walk all roots below.
    primary_root = roots[0] if roots else DEFAULT_SEARCH_PATH
    ok, reason = _check_path_policy(primary_root, write=False)
    if not ok:
        logger.warning("Search blocked by policy: %s", reason)
        return []

    try:
        # Fast path: Windows Search Index across all roots.
        indexed_results = search_windows_index(filename, max_results=max(MAX_FILE_RESULTS * 4, 20))
        if indexed_results:
            filtered = []
            for candidate in indexed_results:
                try:
                    candidate_abs = os.path.abspath(candidate)
                except Exception:
                    continue
                in_any_root = any(_is_subpath(candidate_abs, os.path.abspath(r)) for r in roots)
                if not in_any_root:
                    continue
                path_ok, _ = _check_path_policy(candidate_abs, write=False)
                if not path_ok:
                    continue
                filtered.append(candidate_abs)
                if len(filtered) >= MAX_FILE_RESULTS:
                    break

            if filtered:
                log_action("find_files", "success", details={
                    "query": filename, "root": primary_root,
                    "count": len(filtered), "method": "windows_index",
                })
                return filtered

        name_tokens, ext_tokens = _split_file_query(filename)
        matches = []
        deadline = time.monotonic() + 15
        for root in roots:
            for current_root, _, files in os.walk(root):
                if time.monotonic() > deadline:
                    logger.warning("File walk timed out after 15s searching for '%s'", filename)
                    break
                for name in files:
                    if not _file_matches_query(name, name_tokens, ext_tokens):
                        continue
                    path = os.path.join(current_root, name)
                    path_ok, _ = _check_path_policy(path, write=False)
                    if not path_ok:
                        continue
                    matches.append(path)
                    if len(matches) >= MAX_FILE_RESULTS:
                        break
                if len(matches) >= MAX_FILE_RESULTS:
                    break
            if len(matches) >= MAX_FILE_RESULTS:
                break

        log_action("find_files", "success", details={
            "query": filename, "root": primary_root,
            "count": len(matches), "method": "directory_walk",
        })
        return matches
    except Exception as exc:
        log_action("find_files", "failed", details={"query": filename, "root": primary_root}, error=exc)
        logger.error("File search failed: %s", exc)
        return []


def create_directory_result(path):
    write_ok, write_reason = _validate_file_write_enabled()
    if not write_ok:
        return failure_result(write_reason, error_code="policy_blocked")

    raw_ok, raw_reason, raw_path = _validate_raw_path_input(path, "Path")
    if not raw_ok:
        return _validation_error_result(raw_reason, debug_info={"path": path})
    segments_ok, segments_reason = _validate_path_segments(raw_path, "Path")
    if not segments_ok:
        return _validation_error_result(segments_reason, debug_info={"path": path})

    target = _resolve_path(raw_path)
    ok, reason = _check_path_policy(target, write=True)
    if not ok:
        return failure_result(
            reason,
            error_code="policy_blocked",
            debug_info={"path": target},
        )

    try:
        os.makedirs(target, exist_ok=False)
        action_id = push_rollback_action("remove_path", {"path": target})
        log_action(
            "create_directory",
            "success",
            details={"path": target, "rollback_action_id": action_id},
            rollback_data={"rollback_action_id": action_id},
        )
        return success_result(
            f"Created directory: {target}",
            debug_info={"path": target, "rollback_action_id": action_id},
        )
    except FileExistsError:
        return failure_result(
            f"Directory already exists: {target}",
            error_code="already_exists",
            debug_info={"path": target},
        )
    except Exception as exc:
        log_action("create_directory", "failed", details={"path": target}, error=exc)
        return failure_result(
            f"Failed to create directory: {exc}",
            error_code="execution_failed",
            debug_info={"path": target},
        )


def change_directory(path):
    return to_legacy_pair(change_directory_result(path))


def list_directory(path=None, limit=50):
    return to_legacy_pair(list_directory_result(path=path, limit=limit))


def list_drives_win32():
    return to_legacy_pair(list_drives_win32_result())


def get_file_metadata(path):
    return to_legacy_pair(get_file_metadata_result(path))


def create_directory(path):
    return to_legacy_pair(create_directory_result(path))


def request_move_item(source, destination):
    write_ok, write_reason = _validate_file_write_enabled()
    if not write_ok:
        return failure_result(write_reason, error_code="policy_blocked")

    ok, reason, src, dst = _prepare_move_paths(source, destination)
    if not ok:
        return failure_result(reason, error_code="validation_error")
    description = f"Move item from `{src}` to `{dst}`"
    return _request_file_operation_confirmation(
        "move_item",
        description,
        {"source": src, "destination": dst},
    )


def request_rename_item(source, new_name, location=None):
    write_ok, write_reason = _validate_file_write_enabled()
    if not write_ok:
        return failure_result(write_reason, error_code="policy_blocked")

    ok, reason, src, dst = _prepare_rename_paths(source, new_name)
    if not ok and reason.startswith("Source does not exist") and location:
        # Source not found as a literal path — search for it in the given location.
        location_path = resolve_location(location) if location else None
        loc_str = str(location_path) if location_path else location
        status, value = resolve_name_in_location(source, loc_str)
        if status == "single":
            ok, reason, src, dst = _prepare_rename_paths(value, new_name)
        elif status == "ambiguous":
            return failure_result(
                "Multiple items match that name.",
                error_code="ambiguous_target",
                debug_info={"candidates": value or []},
            )

    if not ok:
        if not location:
            # No explicit location — search in default roots.
            for root_name in FILE_DEFAULT_SEARCH_ROOTS:
                folder = KNOWN_FOLDERS.get(root_name)
                if not folder:
                    continue
                status, value = resolve_name_in_location(source, str(folder))
                if status == "single":
                    ok, reason, src, dst = _prepare_rename_paths(value, new_name)
                    if ok:
                        break
        if not ok:
            return failure_result(reason, error_code="validation_error")

    description = f"Rename item `{src}` to `{os.path.basename(dst)}`"
    return _request_file_operation_confirmation(
        "rename_item",
        description,
        {"source": src, "destination": dst},
    )


def request_delete_item(path, permanent=False, location=None):
    write_ok, write_reason = _validate_file_write_enabled()
    if not write_ok:
        return failure_result(write_reason, error_code="policy_blocked")

    if permanent and not ALLOW_PERMANENT_DELETE:
        return failure_result(
            "Permanent delete is disabled by configuration. Use soft delete or enable ALLOW_PERMANENT_DELETE.",
            error_code="policy_blocked",
        )

    location_path = resolve_location(location) if location else None
    ok, reason, target = _prepare_delete_path(path, location=str(location_path) if location_path else location)
    if not ok:
        if reason == "AMBIGUOUS":
            return failure_result(
                "Multiple items match that name.",
                error_code="ambiguous_target",
                debug_info={"candidates": target or []},
            )
        return failure_result(reason, error_code="validation_error")

    operation = "delete_item_permanent" if permanent else "delete_item"
    if permanent:
        description = f"Permanently delete item `{target}` (cannot be undone)."
    else:
        description = f"Delete item `{target}`"
    return _request_file_operation_confirmation(
        operation,
        description,
        {"path": target, "permanent": bool(permanent)},
    )


def request_copy_item(source, destination):
    """Request user confirmation to copy a file or directory."""
    write_ok, write_reason = _validate_file_write_enabled()
    if not write_ok:
        return failure_result(write_reason, error_code="policy_blocked")

    ok, reason, src, dst = _prepare_copy_paths(source, destination)
    if not ok:
        return failure_result(reason, error_code="validation_error")
    description = f"Copy item from `{src}` to `{dst}`"
    return _request_file_operation_confirmation(
        "copy_item",
        description,
        {"source": src, "destination": dst},
    )


def execute_confirmed_file_operation(payload):
    write_ok, write_reason = _validate_file_write_enabled()
    if not write_ok:
        return failure_result(write_reason, error_code="policy_blocked")

    if (payload or {}).get("kind") != "file_operation":
        return failure_result("Unsupported file operation payload.", error_code="unsupported_action")

    operation = (payload or {}).get("operation")
    resolved_args = (payload or {}).get("resolved_args") or {}
    risk_tier = _risk_tier_for_operation(operation)

    if operation in {"delete_item", "delete_item_permanent"}:
        target = resolved_args.get("path")
        if not target:
            return failure_result(
                "Invalid confirmation payload: missing path.",
                error_code="invalid_payload",
            )
        ok, reason = _check_path_policy(target, write=True)
        if not ok:
            return failure_result(reason, error_code="policy_blocked")
        if not os.path.exists(target):
            return failure_result(f"Path does not exist: {target}", error_code="not_found")

        permanent = bool(resolved_args.get("permanent") or operation == "delete_item_permanent")
        if permanent and not ALLOW_PERMANENT_DELETE:
            return failure_result(
                "Permanent delete is disabled by configuration.",
                error_code="policy_blocked",
            )

        result = _execute_delete_item(target, permanent=permanent)
        if isinstance(result, dict):
            result["risk_tier"] = risk_tier
        return result

    if operation in {"move_item", "rename_item"}:
        src = resolved_args.get("source")
        dst = resolved_args.get("destination")
        if not src or not dst:
            return failure_result(
                "Invalid confirmation payload: missing source/destination.",
                error_code="invalid_payload",
            )
        src_ok, src_reason = _check_path_policy(src, write=True)
        if not src_ok:
            return failure_result(src_reason, error_code="policy_blocked")
        dst_ok, dst_reason = _check_path_policy(dst, write=True)
        if not dst_ok:
            return failure_result(dst_reason, error_code="policy_blocked")
        if not os.path.exists(src):
            return failure_result(f"Source does not exist: {src}", error_code="not_found")
        result = _execute_move_item(src, dst, action_name=operation)
        if isinstance(result, dict):
            result["risk_tier"] = risk_tier
        return result

    if operation == "copy_item":
        src = resolved_args.get("source")
        dst = resolved_args.get("destination")
        if not src or not dst:
            return failure_result(
                "Invalid confirmation payload: missing source/destination.",
                error_code="invalid_payload",
            )
        src_ok, src_reason = _check_path_policy(src, write=False)
        if not src_ok:
            return failure_result(src_reason, error_code="policy_blocked")
        dst_ok, dst_reason = _check_path_policy(dst, write=True)
        if not dst_ok:
            return failure_result(dst_reason, error_code="policy_blocked")
        if not os.path.exists(src):
            return failure_result(f"Source does not exist: {src}", error_code="not_found")
        result = _execute_copy_item(src, dst)
        if isinstance(result, dict):
            result["risk_tier"] = risk_tier
        return result

    return failure_result("Unsupported confirmed file operation.", error_code="unsupported_action")


def move_item(source, destination):
    write_ok, write_reason = _validate_file_write_enabled()
    if not write_ok:
        return False, write_reason
    ok, reason, src, dst = _prepare_move_paths(source, destination)
    if not ok:
        return False, reason
    return to_legacy_pair(_execute_move_item(src, dst, action_name="move_item"))


def rename_item(source, new_name):
    write_ok, write_reason = _validate_file_write_enabled()
    if not write_ok:
        return False, write_reason
    ok, reason, src, dst = _prepare_rename_paths(source, new_name)
    if not ok:
        return False, reason
    return to_legacy_pair(_execute_move_item(src, dst, action_name="rename_item"))


def delete_item(path, permanent=False):
    write_ok, write_reason = _validate_file_write_enabled()
    if not write_ok:
        return False, write_reason
    ok, reason, target = _prepare_delete_path(path)
    if not ok:
        return False, reason
    return to_legacy_pair(_execute_delete_item(target, permanent=bool(permanent)))


def undo_last_action():
    if not policy_engine.is_command_allowed("rollback"):
        return False, "Rollback is disabled by policy."

    entry = pop_latest_rollback_action()
    if not entry:
        return False, "Nothing to rollback."

    action_id = entry["id"]
    action_type = entry["action_type"]
    payload = entry["payload"]

    try:
        if action_type == "remove_path":
            path = payload["path"]
            ok, reason = _check_path_policy(path, write=True)
            if not ok:
                raise RuntimeError(reason)
            if os.path.isdir(path):
                os.rmdir(path)
            elif os.path.exists(path):
                os.remove(path)
            else:
                raise RuntimeError("Rollback path no longer exists.")
        elif action_type == "move":
            source = payload["source"]
            destination = payload["destination"]
            src_ok, src_reason = _check_path_policy(source, write=True)
            if not src_ok:
                raise RuntimeError(src_reason)
            dst_ok, dst_reason = _check_path_policy(destination, write=True)
            if not dst_ok:
                raise RuntimeError(dst_reason)
            destination_parent = os.path.dirname(destination)
            if destination_parent:
                os.makedirs(destination_parent, exist_ok=True)
            shutil.move(source, destination)
        else:
            raise RuntimeError("Unsupported rollback action.")

        log_action("undo", "success", details={"rollback_action_id": action_id})
        return True, "Rollback completed."
    except Exception as exc:
        restore_rollback_action(action_id)
        log_action("undo", "failed", details={"rollback_action_id": action_id}, error=exc)
        return False, f"Rollback failed: {exc}"









