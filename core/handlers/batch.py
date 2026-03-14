from os_control.batch_ops import batch_manager


def handle(parsed, parse_command_fn, execute_internal_fn):
    if parsed.action == "plan":
        ok, message = batch_manager.plan()
        return ok, message, {}
    if parsed.action == "add":
        ok, message = batch_manager.add(parsed.args.get("command_text", ""))
        return ok, message, {}
    if parsed.action == "status":
        ok, message = batch_manager.status()
        return ok, message, {}
    if parsed.action == "preview":
        ok, message = batch_manager.preview(parse_command_fn)
        return ok, message, {}
    if parsed.action == "abort":
        ok, message = batch_manager.abort()
        return ok, message, {}
    if parsed.action == "commit":
        ok, message = batch_manager.commit(parse_command_fn, execute_internal_fn)
        return ok, message, {}
    return False, "Unsupported batch command.", {}
