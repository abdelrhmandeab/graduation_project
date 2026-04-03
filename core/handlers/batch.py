from os_control.adapter_result import to_router_tuple
from os_control.batch_ops import batch_manager


def handle(parsed, parse_command_fn, execute_internal_fn):
    if parsed.action == "plan":
        return to_router_tuple(batch_manager.plan_result())
    if parsed.action == "add":
        return to_router_tuple(batch_manager.add_result(parsed.args.get("command_text", "")))
    if parsed.action == "status":
        return to_router_tuple(batch_manager.status_result())
    if parsed.action == "preview":
        return to_router_tuple(batch_manager.preview_result(parse_command_fn))
    if parsed.action == "abort":
        return to_router_tuple(batch_manager.abort_result())
    if parsed.action == "commit":
        return to_router_tuple(batch_manager.commit_result(parse_command_fn, execute_internal_fn))
    return False, "Unsupported batch command.", {}
