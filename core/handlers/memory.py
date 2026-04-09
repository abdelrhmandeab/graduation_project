from core.session_memory import session_memory
from os_control.action_log import log_action


def _normalize_language(value):
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ar": "ar",
        "arabic": "ar",
        "عربي": "ar",
        "العربية": "ar",
        "en": "en",
        "english": "en",
        "انجليزي": "en",
        "الانجليزية": "en",
        "الإنجليزية": "en",
    }
    return aliases.get(normalized, "")


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
            f"last_app: {status.get('last_app') or 'none'}",
            f"last_app_updated_at: {status.get('last_app_updated_at') or 0.0}",
            f"last_file: {status.get('last_file') or 'none'}",
            f"last_file_updated_at: {status.get('last_file_updated_at') or 0.0}",
            f"pending_confirmation_token: {status.get('pending_confirmation_token') or 'none'}",
            f"pending_confirmation_updated_at: {status.get('pending_confirmation_updated_at') or 0.0}",
            f"stt_profile: {status.get('stt_profile') or 'default'}",
            f"stt_profile_updated_at: {status.get('stt_profile_updated_at') or 0.0}",
            f"hf_profile: {status.get('hf_profile') or 'custom'}",
            f"hf_profile_updated_at: {status.get('hf_profile_updated_at') or 0.0}",
            f"audio_ux_profile: {status.get('audio_ux_profile') or 'custom'}",
            f"audio_ux_profile_updated_at: {status.get('audio_ux_profile_updated_at') or 0.0}",
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

    if action == "set_language":
        requested_language = parsed.args.get("language")
        language = _normalize_language(requested_language)
        if language not in {"ar", "en"}:
            return False, "Unsupported language. Use: arabic or english.", {}

        ok, _message = session_memory.set_preferred_language(language)
        log_action(
            "memory_language_set",
            "success" if ok else "failed",
            details={"requested": requested_language, "language": language},
            error=None if ok else "set_preferred_language_failed",
        )
        if not ok:
            return False, "Failed to update preferred language.", {}
        return True, f"Preferred language: {language}", {"preferred_language": language}

    return False, "Unsupported memory command.", {}
