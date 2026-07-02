import os
from os_control.adapter_result import to_router_tuple
from os_control.file_ops import (
    change_directory_result,
    create_directory_result,
    get_current_directory,
    get_file_metadata_result,
    list_directory_result,
    list_drives_win32_result,
    request_copy_item,
    request_delete_item,
    request_move_item,
    request_rename_item,
)
from os_control.explorer_ops import open_in_explorer, open_file, reveal_in_explorer
from core.session_memory import session_memory


def _last_file():
    """Return the last-referenced file path from session memory, or ''."""
    return str(session_memory.get_last_file() or "").strip()


def _language(parsed) -> str:
    return str((parsed.args or {}).get("_language") or "en")


def handle(parsed):
    action = parsed.action
    args = parsed.args
    lang = _language(parsed)

    if action == "open_in_explorer":
        return open_in_explorer(args.get("path", ""), language=lang)
    if action == "reveal_in_explorer":
        return reveal_in_explorer(args.get("path", ""), language=lang)
    if action == "open_file":
        return open_file(args.get("path", ""), language=lang)

    if action == "pwd":
        return True, f"Current directory: {get_current_directory()}", {}
    if action == "cd":
        return to_router_tuple(change_directory_result(args.get("path", "")))
    if action == "list_drives":
        return to_router_tuple(list_drives_win32_result())
    if action == "list_directory":
        return to_router_tuple(list_directory_result(args.get("path")))
    if action == "file_info":
        return to_router_tuple(get_file_metadata_result(args.get("path", "")))
    if action == "create_directory":
        return to_router_tuple(create_directory_result(args.get("path", "")))
    if action == "delete_item":
        return to_router_tuple(request_delete_item(args.get("path", ""), permanent=False, location=args.get("location")))
    if action == "delete_item_permanent":
        return to_router_tuple(request_delete_item(args.get("path", ""), permanent=True, location=args.get("location")))
    if action == "move_item":
        return to_router_tuple(request_move_item(args.get("source", ""), args.get("destination", "")))
    if action == "move_item_followup":
        src = _last_file()
        dest = str(args.get("destination") or "").strip()
        if not src:
            return False, "I don't have a recent file reference. Say the full move command with the filename.", {}
        if not dest:
            return False, "Please specify where to move it.", {}
        return to_router_tuple(request_move_item(src, dest))
    if action == "copy_item":
        return to_router_tuple(request_copy_item(args.get("source", ""), args.get("destination", "")))
    if action == "rename_item":
        return to_router_tuple(request_rename_item(args.get("source", ""), args.get("new_name", ""), location=args.get("location")))
    if action == "rename_item_followup":
        src = _last_file()
        new_name = str(args.get("new_name") or "").strip()
        if not src:
            return False, "I don't have a recent file reference. Say the full rename command with the filename.", {}
        if not new_name:
            return False, "Please specify the new name.", {}
        # Preserve the original file extension when the user didn't specify one.
        ext = os.path.splitext(src)[1]
        if not os.path.splitext(new_name)[1] and ext:
            new_name = new_name + ext
        # request_rename_item expects a bare new name (not a full path) —
        # it computes os.path.join(dirname(source), new_name) internally.
        return to_router_tuple(request_rename_item(src, new_name))

    return False, "Unsupported file navigation command.", {}
