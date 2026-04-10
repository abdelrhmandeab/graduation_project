from core.persona import persona_manager
from os_control.action_log import log_action


def _format_status():
    status = persona_manager.status()
    lines = [
        "Persona Status",
        f"active_profile: {status['active_profile']}",
        f"speech_style: {status['speech_style']}",
        f"speech_rate: {status['speech_rate']}",
        "available_profiles:",
    ]
    for profile in status["available_profiles"]:
        lines.append(f"- {profile}")
    return "\n".join(lines)


def handle(parsed):
    action = parsed.action
    args = parsed.args

    if action == "status":
        return True, _format_status(), {}
    if action == "list":
        profiles = persona_manager.list_profiles()
        return True, "Available personas:\n" + "\n".join(f"- {p}" for p in profiles), {}
    if action == "set":
        ok, message = persona_manager.set_profile(args.get("profile", ""))
        log_action(
            "persona_set",
            "success" if ok else "failed",
            details={"profile": args.get("profile")},
            error=None if ok else message,
        )
        return ok, message, {"persona": persona_manager.get_profile()}

    return False, "Unsupported persona command.", {}
