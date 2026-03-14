from audio.tts import speech_engine
from core.persona import persona_manager
from os_control.action_log import log_action


def handle(parsed):
    action = parsed.action
    args = parsed.args

    if action == "status":
        clone = persona_manager.get_clone_settings()
        speaking = speech_engine.is_speaking()
        enabled = speech_engine.is_enabled()
        lines = [
            "Voice Status",
            f"speech_enabled: {enabled}",
            f"is_speaking: {speaking}",
            f"active_persona: {clone.get('profile')}",
            f"speech_rate: {persona_manager.get_speech_rate()}",
            f"clone_enabled: {clone['enabled']}",
            f"clone_provider: {clone['provider']}",
            f"clone_reference_audio: {clone['reference_audio'] or 'not_set'}",
        ]
        return True, "\n".join(lines), {}

    if action == "clone_on":
        ok, message = persona_manager.set_clone_enabled(True)
        log_action("voice_clone_toggle", "success", details={"enabled": True})
        return ok, message, {"voice_clone": True}

    if action == "clone_off":
        ok, message = persona_manager.set_clone_enabled(False)
        log_action("voice_clone_toggle", "success", details={"enabled": False})
        return ok, message, {"voice_clone": False}

    if action == "set_provider":
        provider = args.get("provider", "")
        ok, message = persona_manager.set_clone_provider(provider)
        log_action(
            "voice_clone_provider",
            "success" if ok else "failed",
            details={"provider": provider},
            error=None if ok else message,
        )
        return ok, message, {}

    if action == "set_reference":
        ref_path = args.get("path", "")
        ok, message = persona_manager.set_clone_reference_audio(ref_path)
        log_action(
            "voice_clone_reference",
            "success" if ok else "failed",
            details={"path": ref_path},
            error=None if ok else message,
        )
        return ok, message, {}

    if action == "interrupt":
        if not speech_engine.is_speaking():
            return True, "No active speech to interrupt.", {"speech_interrupted": False}
        speech_engine.interrupt()
        log_action("speech_interrupt", "success")
        return True, "Speech interrupted.", {"speech_interrupted": True}

    if action == "speech_on":
        ok, message = speech_engine.set_enabled(True)
        log_action("speech_toggle", "success", details={"enabled": True})
        return ok, message, {"speech_enabled": True}

    if action == "speech_off":
        ok, message = speech_engine.set_enabled(False)
        log_action("speech_toggle", "success", details={"enabled": False})
        return ok, message, {"speech_enabled": False}

    return False, "Unsupported voice command.", {}
