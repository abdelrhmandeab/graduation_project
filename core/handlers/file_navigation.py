import os

from os_control.file_ops import (
    change_directory,
    create_directory,
    delete_item,
    get_current_directory,
    get_file_metadata,
    list_directory,
    list_drives_win32,
    move_item,
)


def handle(parsed):
    action = parsed.action
    args = parsed.args

    if action == "pwd":
        return True, f"Current directory: {get_current_directory()}", {}
    if action == "cd":
        return (*change_directory(args.get("path", "")), {})
    if action == "list_drives":
        return (*list_drives_win32(), {})
    if action == "list_directory":
        return (*list_directory(args.get("path")), {})
    if action == "file_info":
        return (*get_file_metadata(args.get("path", "")), {})
    if action == "create_directory":
        return (*create_directory(args.get("path", "")), {})
    if action == "delete_item":
        return (*delete_item(args.get("path", "")), {})
    if action == "move_item":
        return (*move_item(args.get("source", ""), args.get("destination", "")), {})
    if action == "rename_item":
        source = args.get("source", "")
        new_name = args.get("new_name", "")
        source_abs = (
            os.path.abspath(source)
            if os.path.isabs(source)
            else os.path.join(get_current_directory(), source)
        )
        destination = os.path.join(os.path.dirname(source_abs), new_name)
        return (*move_item(source, destination), {})

    return False, "Unsupported file navigation command.", {}
