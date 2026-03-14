from core.persona import persona_manager
from os_control.action_log import log_action


def _format_status():
    status = persona_manager.status()
    lines = [
        "Persona Status",
        f"active_profile: {status['active_profile']}",
        f"speech_style: {status['speech_style']}",
        f"speech_rate: {status['speech_rate']}",
        f"clone_enabled: {status['clone_enabled']}",
        f"clone_provider: {status['clone_provider']}",
        f"clone_reference_audio: {status['clone_reference_audio'] or 'not_set'}",
        "available_profiles:",
    ]
    for profile in status["available_profiles"]:
        lines.append(f"- {profile}")
    lines.append("profile_voice_map:")
    for profile, voice_cfg in sorted(status.get("voice_profiles", {}).items()):
        lines.append(
            (
                f"- {profile}: clone_enabled={voice_cfg.get('clone_enabled')}, "
                f"clone_provider={voice_cfg.get('clone_provider')}, "
                f"reference_audio={voice_cfg.get('reference_audio') or 'not_set'}"
            )
        )
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

    if action == "voice_status":
        voice_map = persona_manager.profile_voice_map()
        lines = ["Persona Voice Status"]
        for profile, voice_cfg in sorted(voice_map.items()):
            lines.append(
                (
                    f"- {profile}: clone_enabled={voice_cfg.get('clone_enabled')}, "
                    f"clone_provider={voice_cfg.get('clone_provider')}, "
                    f"reference_audio={voice_cfg.get('reference_audio') or 'not_set'}"
                )
            )
        return True, "\n".join(lines), {}

    if action == "set_profile_clone_enabled":
        profile = args.get("profile")
        enabled = bool(args.get("enabled"))
        ok, message = persona_manager.set_profile_clone_enabled(profile, enabled)
        log_action(
            "persona_voice_clone_toggle",
            "success" if ok else "failed",
            details={"profile": profile, "enabled": enabled},
            error=None if ok else message,
        )
        return ok, message, {}

    if action == "set_profile_clone_provider":
        profile = args.get("profile")
        provider = args.get("provider")
        ok, message = persona_manager.set_profile_clone_provider(profile, provider)
        log_action(
            "persona_voice_provider",
            "success" if ok else "failed",
            details={"profile": profile, "provider": provider},
            error=None if ok else message,
        )
        return ok, message, {}

    if action == "set_profile_clone_reference":
        profile = args.get("profile")
        path = args.get("path")
        ok, message = persona_manager.set_profile_clone_reference_audio(profile, path)
        log_action(
            "persona_voice_reference",
            "success" if ok else "failed",
            details={"profile": profile, "path": path},
            error=None if ok else message,
        )
        return ok, message, {}

    return False, "Unsupported persona command.", {}
