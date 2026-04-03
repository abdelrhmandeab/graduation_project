from os_control.adapter_result import to_router_tuple
from os_control.file_ops import (
    change_directory,
    create_directory,
    get_current_directory,
    get_file_metadata,
    list_directory,
    list_drives_win32,
    request_delete_item,
    request_move_item,
    request_rename_item,
)


def handle(parsed):
    action = parsed.action
    args = parsed.args

    if action == "pwd":
        return True, f"Current directory: {get_current_directory()}", {}
    if action == "cd":
        return to_router_tuple(change_directory(args.get("path", "")))
    if action == "list_drives":
        return to_router_tuple(list_drives_win32())
    if action == "list_directory":
        return to_router_tuple(list_directory(args.get("path")))
    if action == "file_info":
        return to_router_tuple(get_file_metadata(args.get("path", "")))
    if action == "create_directory":
        return to_router_tuple(create_directory(args.get("path", "")))
    if action == "delete_item":
        return to_router_tuple(request_delete_item(args.get("path", ""), permanent=False))
    if action == "delete_item_permanent":
        return to_router_tuple(request_delete_item(args.get("path", ""), permanent=True))
    if action == "move_item":
        return to_router_tuple(request_move_item(args.get("source", ""), args.get("destination", "")))
    if action == "rename_item":
        return to_router_tuple(request_rename_item(args.get("source", ""), args.get("new_name", "")))

    return False, "Unsupported file navigation command.", {}
