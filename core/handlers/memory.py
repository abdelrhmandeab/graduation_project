from core.session_memory import session_memory
from os_control.action_log import log_action


def handle(parsed):
    action = parsed.action

    if action == "status":
        status = session_memory.status()
        lines = [
            "Memory Status",
            f"enabled: {status['enabled']}",
            f"turn_count: {status['turn_count']}",
            f"max_turns: {status['max_turns']}",
            f"file: {status['file']}",
            f"preferred_language: {status.get('preferred_language', 'en')}",
            f"pending_clarification: {status.get('pending_clarification', False)}",
        ]
        return True, "\n".join(lines), {}

    if action == "clear":
        ok, message = session_memory.clear()
        log_action("memory_clear", "success" if ok else "failed")
        return ok, message, {}

    if action == "on":
        ok, message = session_memory.set_enabled(True)
        log_action("memory_toggle", "success", details={"enabled": True})
        return ok, message, {}

    if action == "off":
        ok, message = session_memory.set_enabled(False)
        log_action("memory_toggle", "success", details={"enabled": False})
        return ok, message, {}

    if action == "show":
        context = session_memory.build_context()
        if not context:
            return True, "Memory is empty.", {}
        return True, "Recent Memory\n" + context, {}

    return False, "Unsupported memory command.", {}
