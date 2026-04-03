import ctypes
import os
import re
import shutil
import uuid

from core.config import (
    ALLOW_PERMANENT_DELETE,
    CONFIRMATION_TIMEOUT_SECONDS,
    DEFAULT_SEARCH_PATH,
    DEFAULT_WORKING_DIRECTORY,
    MAX_FILE_RESULTS,
    ROLLBACK_DIR_NAME,
    SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE,
)
from core.logger import logger
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

_current_directory = os.path.abspath(DEFAULT_WORKING_DIRECTORY)

_FILE_OPERATION_RISK = {
    "move_item": "medium",
    "rename_item": "medium",
    "delete_item": "high",
    "delete_item_permanent": "high",
}
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
    return _FILE_OPERATION_RISK.get(operation, "low")


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


def _prepare_delete_path(path):
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
    if not os.path.exists(target):
        return False, f"Path does not exist: {target}", None
    return True, "", target


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

    message = (
        f"Confirmation required (risk: {risk_tier}) for: {description}. "
        f"Say `confirm {token}`"
    )
    if require_second_factor:
        message += " and provide PIN/passphrase as second factor."
    message += f" within {CONFIRMATION_TIMEOUT_SECONDS} seconds."
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


def get_current_directory():
    return _current_directory


def change_directory(path):
    global _current_directory
    target = _resolve_path(path)
    ok, reason = _check_path_policy(target, write=False)
    if not ok:
        return False, reason
    if not os.path.isdir(target):
        return False, f"Directory does not exist: {target}"

    _current_directory = target
    log_action("change_directory", "success", details={"new_directory": target})
    return True, f"Current directory set to: {target}"


def list_directory(path=None, limit=50):
    target = _resolve_path(path)
    ok, reason = _check_path_policy(target, write=False)
    if not ok:
        return False, reason
    if not os.path.isdir(target):
        return False, f"Directory does not exist: {target}"

    try:
        entries = []
        with os.scandir(target) as it:
            for entry in it:
                prefix = "[D]" if entry.is_dir() else "[F]"
                entries.append(f"{prefix} {entry.name}")
                if len(entries) >= limit:
                    break
        log_action("list_directory", "success", details={"path": target, "count": len(entries)})
        return (True, "\n".join(entries)) if entries else (True, "Directory is empty.")
    except Exception as exc:
        log_action("list_directory", "failed", details={"path": target}, error=exc)
        return False, f"Failed to list directory: {exc}"


def list_drives_win32():
    if os.name != "nt":
        return False, "Drive listing via Win32 is only available on Windows."
    if not policy_engine.is_command_allowed("file_navigation"):
        return False, "File navigation is disabled by policy."

    try:
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        drives = [f"{chr(65 + index)}:\\" for index in range(26) if bitmask & (1 << index)]
        log_action("list_drives", "success", details={"count": len(drives)})
        return True, "\n".join(drives) if drives else "No drives found."
    except Exception as exc:
        log_action("list_drives", "failed", error=exc)
        return False, f"Failed to list drives: {exc}"


def get_file_metadata(path):
    target = _resolve_path(path)
    ok, reason = _check_path_policy(target, write=False)
    if not ok:
        return False, reason
    if not os.path.exists(target):
        return False, f"Path does not exist: {target}"

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
        return True, "\n".join(metadata)
    except Exception as exc:
        log_action("file_metadata", "failed", details={"path": target}, error=exc)
        return False, f"Failed to read metadata: {exc}"


def find_files(filename, search_path=None):
    if not policy_engine.is_command_allowed("file_search"):
        return []
    if not filename:
        return []

    root = _resolve_path(search_path) if search_path else DEFAULT_SEARCH_PATH
    ok, reason = _check_path_policy(root, write=False)
    if not ok:
        logger.warning("Search blocked by policy: %s", reason)
        return []

    try:
        needle = filename.lower()
        matches = []
        for current_root, _, files in os.walk(root):
            for name in files:
                if needle in name.lower():
                    path = os.path.join(current_root, name)
                    path_ok, _ = _check_path_policy(path, write=False)
                    if not path_ok:
                        continue
                    matches.append(path)
                    if len(matches) >= MAX_FILE_RESULTS:
                        log_action(
                            "find_files",
                            "success",
                            details={"query": filename, "root": root, "count": len(matches)},
                        )
                        return matches
        log_action("find_files", "success", details={"query": filename, "root": root, "count": len(matches)})
        return matches
    except Exception as exc:
        log_action("find_files", "failed", details={"query": filename, "root": root}, error=exc)
        logger.error("File search failed: %s", exc)
        return []


def create_directory(path):
    write_ok, write_reason = _validate_file_write_enabled()
    if not write_ok:
        return False, write_reason

    raw_ok, raw_reason, raw_path = _validate_raw_path_input(path, "Path")
    if not raw_ok:
        return False, raw_reason
    segments_ok, segments_reason = _validate_path_segments(raw_path, "Path")
    if not segments_ok:
        return False, segments_reason

    target = _resolve_path(raw_path)
    ok, reason = _check_path_policy(target, write=True)
    if not ok:
        return False, reason

    try:
        os.makedirs(target, exist_ok=False)
        action_id = push_rollback_action("remove_path", {"path": target})
        log_action(
            "create_directory",
            "success",
            details={"path": target, "rollback_action_id": action_id},
            rollback_data={"rollback_action_id": action_id},
        )
        return True, f"Created directory: {target}"
    except FileExistsError:
        return False, f"Directory already exists: {target}"
    except Exception as exc:
        log_action("create_directory", "failed", details={"path": target}, error=exc)
        return False, f"Failed to create directory: {exc}"


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


def request_rename_item(source, new_name):
    write_ok, write_reason = _validate_file_write_enabled()
    if not write_ok:
        return failure_result(write_reason, error_code="policy_blocked")

    ok, reason, src, dst = _prepare_rename_paths(source, new_name)
    if not ok:
        return failure_result(reason, error_code="validation_error")
    description = f"Rename item `{src}` to `{os.path.basename(dst)}`"
    return _request_file_operation_confirmation(
        "rename_item",
        description,
        {"source": src, "destination": dst},
    )


def request_delete_item(path, permanent=False):
    write_ok, write_reason = _validate_file_write_enabled()
    if not write_ok:
        return failure_result(write_reason, error_code="policy_blocked")

    if permanent and not ALLOW_PERMANENT_DELETE:
        return failure_result(
            "Permanent delete is disabled by configuration. Use soft delete or enable ALLOW_PERMANENT_DELETE.",
            error_code="policy_blocked",
        )

    ok, reason, target = _prepare_delete_path(path)
    if not ok:
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









