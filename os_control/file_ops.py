import ctypes
import os
import shutil
import uuid

from core.config import (
    DEFAULT_SEARCH_PATH,
    DEFAULT_WORKING_DIRECTORY,
    MAX_FILE_RESULTS,
    ROLLBACK_DIR_NAME,
)
from core.logger import logger
from os_control.action_log import log_action
from os_control.persistence import (
    pop_latest_rollback_action,
    push_rollback_action,
    restore_rollback_action,
)
from os_control.policy import policy_engine

_current_directory = os.path.abspath(DEFAULT_WORKING_DIRECTORY)


def _ensure_rollback_dir():
    temp_root = os.environ.get("TEMP") or os.getcwd()
    rollback_root = os.path.join(temp_root, ROLLBACK_DIR_NAME)
    os.makedirs(rollback_root, exist_ok=True)
    return rollback_root


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
    if not policy_engine.is_command_allowed("file_write"):
        return False, "File write operations are disabled by policy."

    target = _resolve_path(path)
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


def move_item(source, destination):
    if not policy_engine.is_command_allowed("file_write"):
        return False, "File write operations are disabled by policy."

    src = _resolve_path(source)
    dst = _resolve_path(destination)
    src_ok, src_reason = _check_path_policy(src, write=True)
    if not src_ok:
        return False, src_reason
    dst_ok, dst_reason = _check_path_policy(dst, write=True)
    if not dst_ok:
        return False, dst_reason
    if not os.path.exists(src):
        return False, f"Source does not exist: {src}"

    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        action_id = push_rollback_action("move", {"source": dst, "destination": src})
        log_action(
            "move_item",
            "success",
            details={"source": src, "destination": dst, "rollback_action_id": action_id},
            rollback_data={"rollback_action_id": action_id},
        )
        return True, f"Moved: {src} -> {dst}"
    except Exception as exc:
        log_action("move_item", "failed", details={"source": src, "destination": dst}, error=exc)
        return False, f"Failed to move item: {exc}"


def delete_item(path):
    if not policy_engine.is_command_allowed("file_write"):
        return False, "File write operations are disabled by policy."

    target = _resolve_path(path)
    ok, reason = _check_path_policy(target, write=True)
    if not ok:
        return False, reason
    if not os.path.exists(target):
        return False, f"Path does not exist: {target}"

    try:
        rollback_root = _ensure_rollback_dir()
        backup_name = f"{uuid.uuid4().hex}_{os.path.basename(target)}"
        backup_path = os.path.join(rollback_root, backup_name)
        shutil.move(target, backup_path)

        action_id = push_rollback_action(
            "move",
            {"source": backup_path, "destination": target},
        )
        log_action(
            "delete_item",
            "success",
            details={"path": target, "backup_path": backup_path, "rollback_action_id": action_id},
            rollback_data={"rollback_action_id": action_id},
        )
        return True, f"Deleted (moved to rollback storage): {target}"
    except Exception as exc:
        log_action("delete_item", "failed", details={"path": target}, error=exc)
        return False, f"Failed to delete item: {exc}"


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
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.move(source, destination)
        else:
            raise RuntimeError("Unsupported rollback action.")

        log_action("undo", "success", details={"rollback_action_id": action_id})
        return True, "Rollback completed."
    except Exception as exc:
        restore_rollback_action(action_id)
        log_action("undo", "failed", details={"rollback_action_id": action_id}, error=exc)
        return False, f"Rollback failed: {exc}"
