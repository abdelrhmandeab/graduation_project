from audio import mic as mic_capture
from audio import stt as stt_runtime
from audio import vad as vad_runtime
from audio import wake_word as wake_word_runtime
from audio.tts import speech_engine
from core.logger import logger
from core.persona import persona_manager
from core.session_memory import session_memory
from os_control.action_log import log_action


_STT_PROFILE_PRESETS = {
    "quiet": {
        "mic": {
            "energy_threshold": 0.009,
            "silence_seconds": 0.95,
            "min_speech_seconds": 0.30,
            "start_timeout_seconds": 5.0,
        },
        "stt": {
            "beam_size": 2,
            "vad_filter": False,
            "condition_on_previous_text": False,
            "quality_retry_threshold": 0.50,
            "quality_retry_beam_size": 4,
        },
        "speech_guard_threshold": 0.009,
    },
    "noisy": {
        "mic": {
            "energy_threshold": 0.020,
            "silence_seconds": 0.60,
            "min_speech_seconds": 0.45,
            "start_timeout_seconds": 3.2,
        },
        "stt": {
            "beam_size": 4,
            "vad_filter": True,
            "condition_on_previous_text": False,
            "quality_retry_threshold": 0.62,
            "quality_retry_beam_size": 6,
        },
        "speech_guard_threshold": 0.020,
    },
}
_HF_SPEECH_PROFILE_PRESETS = {
    "arabic": {
        "stt": {
            "model": "openai/whisper-small",
            "mode": "manual",
            "chunk_length_s": 12.0,
            "batch_size": 4,
        },
        "tts": {
            "model": "facebook/mms-tts-ara",
            "sample_rate": 0,
        },
    },
    "english": {
        "stt": {
            "model": "openai/whisper-small.en",
            "mode": "manual",
            "chunk_length_s": 10.0,
            "batch_size": 4,
        },
        "tts": {
            "model": "facebook/mms-tts-eng",
            "sample_rate": 0,
        },
    },
}
_AUDIO_UX_PROFILE_PRESETS = {
    "balanced": {
        "mic": {
            "energy_threshold": 0.012,
            "silence_seconds": 0.80,
            "min_speech_seconds": 0.35,
            "start_timeout_seconds": 4.0,
        },
        "speech_guard_threshold": 0.012,
        "wake_word": {
            "threshold": 0.35,
            "audio_gain": 1.4,
            "detection_cooldown_seconds": 1.0,
        },
        "wake_behavior": {
            "ignore_while_speaking": True,
            "barge_in_interrupt_on_wake": True,
        },
        "tts": {
            "quality_mode": "natural",
            "rate_offset": 0,
            "pause_scale": 1.0,
        },
    },
    "responsive": {
        "mic": {
            "energy_threshold": 0.009,
            "silence_seconds": 0.55,
            "min_speech_seconds": 0.25,
            "start_timeout_seconds": 3.0,
        },
        "speech_guard_threshold": 0.009,
        "wake_word": {
            "threshold": 0.30,
            "audio_gain": 1.55,
            "detection_cooldown_seconds": 0.8,
        },
        "wake_behavior": {
            "ignore_while_speaking": False,
            "barge_in_interrupt_on_wake": True,
        },
        "tts": {
            "quality_mode": "natural",
            "rate_offset": 8,
            "pause_scale": 0.9,
        },
    },
    "robust": {
        "mic": {
            "energy_threshold": 0.020,
            "silence_seconds": 0.95,
            "min_speech_seconds": 0.45,
            "start_timeout_seconds": 4.5,
        },
        "speech_guard_threshold": 0.020,
        "wake_word": {
            "threshold": 0.46,
            "audio_gain": 1.2,
            "detection_cooldown_seconds": 1.4,
        },
        "wake_behavior": {
            "ignore_while_speaking": True,
            "barge_in_interrupt_on_wake": False,
        },
        "tts": {
            "quality_mode": "standard",
            "rate_offset": -6,
            "pause_scale": 1.1,
        },
    },
}
_AUDIO_UX_PROFILE_DESCRIPTIONS = {
    "balanced": "General use with stable latency and natural speech pacing.",
    "responsive": "Faster turn-taking and lower wake/endpoint latency.",
    "robust": "More conservative detection for noisy environments.",
}
_ACTIVE_STT_PROFILE = "default"
_ACTIVE_HF_SPEECH_PROFILE = "custom"
_ACTIVE_AUDIO_UX_PROFILE = "custom"
_RUNTIME_PROFILES_INITIALIZED = False


def _normalize_profile_name(value):
    profile = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "quiet_room": "quiet",
        "noisy_room": "noisy",
    }
    return aliases.get(profile, profile)


def _normalize_hf_profile_name(value):
    profile = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "ar": "arabic",
        "arabic_first": "arabic",
        "arabic_mode": "arabic",
        "arabicprofile": "arabic",
        "arabic_profile": "arabic",
        "عربي": "arabic",
        "العربية": "arabic",
        "en": "english",
        "english_first": "english",
        "english_mode": "english",
        "englishprofile": "english",
        "english_profile": "english",
        "انجليزي": "english",
        "الانجليزية": "english",
        "الإنجليزية": "english",
    }
    return aliases.get(profile, profile)


def _normalize_voice_quality(value):
    mode = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "human": "natural",
        "natural_voice": "natural",
        "طبيعي": "natural",
        "robot": "standard",
        "robotic": "standard",
        "روبوت": "standard",
        "روبوتي": "standard",
        "قياسي": "standard",
        "افتراضي": "standard",
        "default": "standard",
        "balanced": "standard",
    }
    return aliases.get(mode, mode)


def _normalize_audio_ux_profile(value):
    profile = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "default": "balanced",
        "standard": "balanced",
        "متوازن": "balanced",
        "fast": "responsive",
        "low_latency": "responsive",
        "سريع": "responsive",
        "سريع_الاستجابة": "responsive",
        "منخفض_الكمون": "responsive",
        "reliable": "robust",
        "stable": "robust",
        "noisy": "robust",
        "قوي": "robust",
        "ثابت": "robust",
        "موثوق": "robust",
    }
    return aliases.get(profile, profile)


def _stt_runtime_snapshot():
    mic_settings = mic_capture.get_runtime_vad_settings()
    stt_settings = stt_runtime.get_runtime_stt_settings()
    persisted_profile = session_memory.get_stt_profile() or "default"
    return {
        "profile": _ACTIVE_STT_PROFILE,
        "persisted_profile": persisted_profile,
        "mic": mic_settings,
        "stt": stt_settings,
        "speech_guard_threshold": vad_runtime.get_energy_fallback_threshold(),
    }


def _apply_stt_profile(profile_name, *, persist=True):
    global _ACTIVE_STT_PROFILE
    profile = _normalize_profile_name(profile_name)
    preset = _STT_PROFILE_PRESETS.get(profile)
    if not preset:
        return False, "Unsupported STT profile. Use: quiet or noisy.", {}

    mic_capture.set_runtime_vad_settings(**dict(preset["mic"]))
    stt_runtime.set_runtime_stt_settings(**dict(preset["stt"]))
    vad_runtime.set_energy_fallback_threshold(preset["speech_guard_threshold"])
    _ACTIVE_STT_PROFILE = profile
    if persist:
        session_memory.set_stt_profile(profile)

    snapshot = _stt_runtime_snapshot()
    return (
        True,
        (
            f"Active STT profile: {snapshot['profile']}\n"
            f"mic_energy_threshold: {snapshot['mic']['energy_threshold']:.4f}\n"
            f"stt_beam_size: {snapshot['stt']['beam_size']}\n"
            f"stt_vad_filter: {snapshot['stt']['vad_filter']}"
        ),
        snapshot,
    )


def _hf_speech_runtime_snapshot():
    stt_hf = stt_runtime.get_runtime_hf_settings()
    tts_hf = speech_engine.get_hf_runtime_settings()
    persisted_profile = session_memory.get_hf_profile() or "custom"
    return {
        "profile": _ACTIVE_HF_SPEECH_PROFILE,
        "persisted_profile": persisted_profile,
        "stt_backend": stt_runtime.get_runtime_stt_backend(),
        "tts_backend": speech_engine.get_backend(),
        "voice_quality_mode": speech_engine.get_quality_mode(),
        "stt": stt_hf,
        "tts": tts_hf,
    }


def _apply_hf_speech_profile(profile_name, *, persist=True):
    global _ACTIVE_HF_SPEECH_PROFILE
    profile = _normalize_hf_profile_name(profile_name)
    preset = _HF_SPEECH_PROFILE_PRESETS.get(profile)
    if not preset:
        return False, "Unsupported HF speech profile. Use: arabic or english.", {}

    # Keep runtime behavior independent from .env backend defaults.
    stt_runtime.set_runtime_stt_backend("huggingface")
    stt_runtime.set_runtime_hf_settings(**dict(preset["stt"]))
    speech_engine.set_backend("huggingface")
    speech_engine.set_hf_runtime_settings(**dict(preset["tts"]))
    if profile == "english":
        speech_engine.set_quality_mode("natural")
    _ACTIVE_HF_SPEECH_PROFILE = profile
    if persist:
        session_memory.set_hf_profile(profile)
    snapshot = _hf_speech_runtime_snapshot()

    return (
        True,
        (
            f"Active HF speech profile: {snapshot['profile']}\n"
            f"hf_profile_persisted: {snapshot['persisted_profile']}\n"
            f"stt_backend: {snapshot['stt_backend']}\n"
            f"tts_backend: {snapshot['tts_backend']}\n"
            f"voice_quality_mode: {snapshot['voice_quality_mode']}\n"
            f"hf_stt_model: {snapshot['stt']['model']}\n"
            f"hf_stt_mode: {snapshot['stt']['mode']}\n"
            f"hf_tts_model: {snapshot['tts']['model']}"
        ),
        snapshot,
    )


def _audio_ux_runtime_snapshot():
    mic_settings = mic_capture.get_runtime_vad_settings()
    wake_settings = wake_word_runtime.get_runtime_wake_word_settings()
    wake_behavior = wake_word_runtime.get_runtime_wake_word_behavior()
    tts_tuning = speech_engine.get_tuning_settings()
    persisted_profile = session_memory.get_audio_ux_profile() or "custom"
    return {
        "profile": _ACTIVE_AUDIO_UX_PROFILE,
        "persisted_profile": persisted_profile,
        "mic": mic_settings,
        "speech_guard_threshold": vad_runtime.get_energy_fallback_threshold(),
        "wake_word": wake_settings,
        "wake_behavior": wake_behavior,
        "voice_quality_mode": speech_engine.get_quality_mode(),
        "tts_tuning": tts_tuning,
    }


def _apply_audio_ux_profile(profile_name, *, persist=True):
    global _ACTIVE_AUDIO_UX_PROFILE
    profile = _normalize_audio_ux_profile(profile_name)
    preset = _AUDIO_UX_PROFILE_PRESETS.get(profile)
    if not preset:
        return (
            False,
            (
                "Unsupported audio UX profile. Use: balanced, responsive, or robust.\n"
                "Tip: run 'audio ux profiles' to see profile guidance."
            ),
            {},
        )

    mic_capture.set_runtime_vad_settings(**dict(preset["mic"]))
    vad_runtime.set_energy_fallback_threshold(float(preset["speech_guard_threshold"]))
    wake_word_runtime.set_runtime_wake_word_settings(**dict(preset["wake_word"]))
    wake_word_runtime.set_runtime_wake_word_behavior(**dict(preset["wake_behavior"]))
    speech_engine.set_quality_mode(str(preset["tts"].get("quality_mode") or "natural"))
    speech_engine.set_tuning_settings(
        rate_offset=int(preset["tts"].get("rate_offset") or 0),
        pause_scale=float(preset["tts"].get("pause_scale") or 1.0),
    )

    _ACTIVE_AUDIO_UX_PROFILE = profile
    if persist:
        session_memory.set_audio_ux_profile(profile)

    snapshot = _audio_ux_runtime_snapshot()
    return (
        True,
        (
            f"Active audio UX profile: {snapshot['profile']}\n"
            f"audio_ux_profile_persisted: {snapshot['persisted_profile']}\n"
            f"mic_energy_threshold: {snapshot['mic']['energy_threshold']:.4f}\n"
            f"wake_word_threshold: {float(snapshot['wake_word']['threshold']):.2f}\n"
            f"wake_word_gain: {float(snapshot['wake_word']['audio_gain']):.2f}\n"
            f"wake_barge_in_on_wake: {snapshot['wake_behavior']['barge_in_interrupt_on_wake']}\n"
            f"voice_quality_mode: {snapshot['voice_quality_mode']}\n"
            f"tts_rate_offset: {int(snapshot['tts_tuning']['rate_offset'])}"
        ),
        snapshot,
    )


def initialize_runtime_profiles(force=False):
    global _RUNTIME_PROFILES_INITIALIZED
    if _RUNTIME_PROFILES_INITIALIZED and not force:
        return True, "runtime_profiles_already_initialized"

    summary = []
    restored_ok = True

    persisted_stt = _normalize_profile_name(session_memory.get_stt_profile())
    if persisted_stt:
        ok, message, _snapshot = _apply_stt_profile(persisted_stt, persist=False)
        if ok:
            logger.info("Restored persisted STT profile: %s", persisted_stt)
            summary.append(f"restored_stt:{persisted_stt}")
        else:
            logger.warning("Failed to restore persisted STT profile '%s': %s", persisted_stt, message)
            session_memory.set_stt_profile("")
            summary.append("failed_stt_restore")
            restored_ok = False
    else:
        summary.append("no_persisted_stt_profile")

    persisted_hf = _normalize_hf_profile_name(session_memory.get_hf_profile())
    if persisted_hf:
        ok, message, _snapshot = _apply_hf_speech_profile(persisted_hf, persist=False)
        if ok:
            logger.info("Restored persisted HF speech profile: %s", persisted_hf)
            summary.append(f"restored_hf:{persisted_hf}")
        else:
            logger.warning("Failed to restore persisted HF speech profile '%s': %s", persisted_hf, message)
            session_memory.set_hf_profile("")
            summary.append("failed_hf_restore")
            restored_ok = False
    else:
        summary.append("no_persisted_hf_profile")

    persisted_audio_ux = _normalize_audio_ux_profile(session_memory.get_audio_ux_profile())
    if persisted_audio_ux:
        ok, message, _snapshot = _apply_audio_ux_profile(persisted_audio_ux, persist=False)
        if ok:
            logger.info("Restored persisted audio UX profile: %s", persisted_audio_ux)
            summary.append(f"restored_audio_ux:{persisted_audio_ux}")
        else:
            logger.warning("Failed to restore persisted audio UX profile '%s': %s", persisted_audio_ux, message)
            session_memory.set_audio_ux_profile("")
            summary.append("failed_audio_ux_restore")
            restored_ok = False
    else:
        ok, _message, _snapshot = _apply_audio_ux_profile("balanced", persist=False)
        if ok:
            summary.append("default_audio_ux:balanced")
        else:
            summary.append("default_audio_ux_failed")
            restored_ok = False

    _RUNTIME_PROFILES_INITIALIZED = True
    return restored_ok, "; ".join(summary)


def _format_stt_profile_status():
    snapshot = _stt_runtime_snapshot()
    lines = [
        "STT Profile Status",
        f"stt_profile: {snapshot['profile']}",
        f"stt_profile_persisted: {snapshot['persisted_profile']}",
        f"mic_energy_threshold: {snapshot['mic']['energy_threshold']:.4f}",
        f"mic_silence_seconds: {snapshot['mic']['silence_seconds']:.2f}",
        f"mic_min_speech_seconds: {snapshot['mic']['min_speech_seconds']:.2f}",
        f"stt_beam_size: {snapshot['stt']['beam_size']}",
        f"stt_vad_filter: {snapshot['stt']['vad_filter']}",
        f"stt_retry_threshold: {snapshot['stt']['quality_retry_threshold']:.2f}",
        f"speech_guard_threshold: {snapshot['speech_guard_threshold']:.4f}",
    ]
    return "\n".join(lines), snapshot


def _format_hf_profile_status():
    snapshot = _hf_speech_runtime_snapshot()
    lines = [
        "HF Speech Profile Status",
        f"hf_profile: {snapshot['profile']}",
        f"hf_profile_persisted: {snapshot['persisted_profile']}",
        f"stt_backend: {snapshot['stt_backend']}",
        f"tts_backend: {snapshot['tts_backend']}",
        f"voice_quality_mode: {snapshot['voice_quality_mode']}",
        f"hf_stt_model: {snapshot['stt']['model']}",
        f"hf_stt_mode: {snapshot['stt']['mode']}",
        f"hf_stt_chunk_length_s: {float(snapshot['stt']['chunk_length_s']):.1f}",
        f"hf_stt_batch_size: {int(snapshot['stt']['batch_size'])}",
        f"hf_tts_model: {snapshot['tts']['model']}",
        f"hf_tts_sample_rate: {int(snapshot['tts']['sample_rate'])}",
    ]
    return "\n".join(lines), snapshot


def _format_voice_quality_status():
    mode = speech_engine.get_quality_mode()
    lines = [
        "Voice Quality Status",
        f"voice_quality_mode: {mode}",
        f"tts_backend: {speech_engine.get_backend()}",
    ]
    return "\n".join(lines), {"voice_quality_mode": mode}


def _format_audio_ux_status():
    snapshot = _audio_ux_runtime_snapshot()
    lines = [
        "Audio UX Status",
        f"audio_ux_profile: {snapshot['profile']}",
        f"audio_ux_profile_persisted: {snapshot['persisted_profile']}",
        f"mic_energy_threshold: {snapshot['mic']['energy_threshold']:.4f}",
        f"mic_silence_seconds: {snapshot['mic']['silence_seconds']:.2f}",
        f"mic_min_speech_seconds: {snapshot['mic']['min_speech_seconds']:.2f}",
        f"speech_guard_threshold: {snapshot['speech_guard_threshold']:.4f}",
        f"wake_word_threshold: {float(snapshot['wake_word']['threshold']):.2f}",
        f"wake_word_gain: {float(snapshot['wake_word']['audio_gain']):.2f}",
        f"wake_word_cooldown_s: {float(snapshot['wake_word']['detection_cooldown_seconds']):.2f}",
        f"wake_ignore_while_speaking: {snapshot['wake_behavior']['ignore_while_speaking']}",
        f"wake_barge_in_on_wake: {snapshot['wake_behavior']['barge_in_interrupt_on_wake']}",
        f"voice_quality_mode: {snapshot['voice_quality_mode']}",
        f"tts_rate_offset: {int(snapshot['tts_tuning']['rate_offset'])}",
        f"tts_pause_scale: {float(snapshot['tts_tuning']['pause_scale']):.2f}",
    ]
    return "\n".join(lines), snapshot


def _format_audio_ux_profiles():
    lines = ["Audio UX Profiles"]
    for profile_name in ("balanced", "responsive", "robust"):
        lines.append(f"- {profile_name}: {_AUDIO_UX_PROFILE_DESCRIPTIONS[profile_name]}")
    return "\n".join(lines), {
        "profiles": list(_AUDIO_UX_PROFILE_PRESETS.keys()),
        "active_profile": _ACTIVE_AUDIO_UX_PROFILE,
    }


def _parse_float_value(args):
    try:
        return float(args.get("value"))
    except (TypeError, ValueError, AttributeError):
        return None


def _parse_int_value(args):
    try:
        return int(args.get("value"))
    except (TypeError, ValueError, AttributeError):
        return None


def _mark_audio_ux_custom_profile():
    global _ACTIVE_AUDIO_UX_PROFILE
    _ACTIVE_AUDIO_UX_PROFILE = "custom"


def handle(parsed):
    initialize_runtime_profiles()

    action = parsed.action
    args = parsed.args

    if action == "status":
        clone = persona_manager.get_clone_settings()
        speaking = speech_engine.is_speaking()
        enabled = speech_engine.is_enabled()
        stt_profile = _stt_runtime_snapshot()
        hf_profile = _hf_speech_runtime_snapshot()
        audio_ux = _audio_ux_runtime_snapshot()
        lines = [
            "Voice Status",
            f"speech_enabled: {enabled}",
            f"is_speaking: {speaking}",
            f"active_persona: {clone.get('profile')}",
            f"speech_rate: {persona_manager.get_speech_rate()}",
            f"clone_enabled: {clone['enabled']}",
            f"clone_provider: {clone['provider']}",
            f"clone_reference_audio: {clone['reference_audio'] or 'not_set'}",
            f"tts_backend: {speech_engine.get_backend()}",
            f"voice_quality_mode: {speech_engine.get_quality_mode()}",
            f"audio_ux_profile: {audio_ux['profile']}",
            f"stt_backend: {stt_runtime.get_runtime_stt_backend()}",
            f"stt_profile: {stt_profile['profile']}",
            f"stt_profile_persisted: {stt_profile['persisted_profile']}",
            f"stt_beam_size: {stt_profile['stt']['beam_size']}",
            f"mic_energy_threshold: {stt_profile['mic']['energy_threshold']:.4f}",
            f"wake_word_threshold: {float(audio_ux['wake_word']['threshold']):.2f}",
            f"wake_word_gain: {float(audio_ux['wake_word']['audio_gain']):.2f}",
            f"wake_barge_in_on_wake: {audio_ux['wake_behavior']['barge_in_interrupt_on_wake']}",
            f"tts_rate_offset: {int(audio_ux['tts_tuning']['rate_offset'])}",
            f"hf_profile: {hf_profile['profile']}",
            f"hf_profile_persisted: {hf_profile['persisted_profile']}",
            f"hf_stt_model: {hf_profile['stt']['model']}",
            f"hf_tts_model: {hf_profile['tts']['model']}",
        ]
        return True, "\n".join(lines), {}

    if action == "diagnostic":
        ok, message, meta = speech_engine.run_voice_diagnostic()
        log_action("voice_diagnostic", "success" if ok else "failed", details=meta)
        return ok, message, meta

    if action == "voice_quality_status":
        message, snapshot = _format_voice_quality_status()
        return True, message, snapshot

    if action == "audio_ux_status":
        message, snapshot = _format_audio_ux_status()
        return True, message, snapshot

    if action == "audio_ux_profiles":
        message, snapshot = _format_audio_ux_profiles()
        return True, message, snapshot

    if action == "audio_ux_profile_set":
        profile_name = args.get("profile", "")
        ok, message, snapshot = _apply_audio_ux_profile(profile_name)
        log_action(
            "audio_ux_profile_set",
            "success" if ok else "failed",
            details={"requested": profile_name, "active": snapshot.get("profile") if snapshot else None},
            error=None if ok else message,
        )
        return ok, message, snapshot

    if action == "audio_ux_mic_threshold_set":
        value = _parse_float_value(args)
        if value is None:
            return False, "Invalid mic threshold value. Example: set mic threshold to 0.012", {}
        mic_capture.set_runtime_vad_settings(energy_threshold=value)
        vad_runtime.set_energy_fallback_threshold(value)
        _mark_audio_ux_custom_profile()
        message, snapshot = _format_audio_ux_status()
        log_action(
            "audio_ux_mic_threshold_set",
            "success",
            details={"requested": value, "active": snapshot["mic"].get("energy_threshold")},
        )
        return True, message, snapshot

    if action == "audio_ux_wake_threshold_set":
        value = _parse_float_value(args)
        if value is None:
            return False, "Invalid wake threshold value. Example: set wake threshold to 0.38", {}
        wake_word_runtime.set_runtime_wake_word_settings(threshold=value)
        _mark_audio_ux_custom_profile()
        message, snapshot = _format_audio_ux_status()
        log_action(
            "audio_ux_wake_threshold_set",
            "success",
            details={"requested": value, "active": snapshot["wake_word"].get("threshold")},
        )
        return True, message, snapshot

    if action == "audio_ux_wake_gain_set":
        value = _parse_float_value(args)
        if value is None:
            return False, "Invalid wake gain value. Example: set wake gain to 1.6", {}
        wake_word_runtime.set_runtime_wake_word_settings(audio_gain=value)
        _mark_audio_ux_custom_profile()
        message, snapshot = _format_audio_ux_status()
        log_action(
            "audio_ux_wake_gain_set",
            "success",
            details={"requested": value, "active": snapshot["wake_word"].get("audio_gain")},
        )
        return True, message, snapshot

    if action == "audio_ux_pause_scale_set":
        value = _parse_float_value(args)
        if value is None:
            return False, "Invalid pause scale value. Example: set pause scale to 0.9", {}
        speech_engine.set_tuning_settings(pause_scale=value)
        _mark_audio_ux_custom_profile()
        message, snapshot = _format_audio_ux_status()
        log_action(
            "audio_ux_pause_scale_set",
            "success",
            details={"requested": value, "active": snapshot["tts_tuning"].get("pause_scale")},
        )
        return True, message, snapshot

    if action == "audio_ux_rate_offset_set":
        value = _parse_int_value(args)
        if value is None:
            return False, "Invalid rate offset value. Example: set rate offset to -8", {}
        speech_engine.set_tuning_settings(rate_offset=value)
        _mark_audio_ux_custom_profile()
        message, snapshot = _format_audio_ux_status()
        log_action(
            "audio_ux_rate_offset_set",
            "success",
            details={"requested": value, "active": snapshot["tts_tuning"].get("rate_offset")},
        )
        return True, message, snapshot

    if action == "voice_quality_set":
        requested_mode = args.get("mode", "")
        normalized_mode = _normalize_voice_quality(requested_mode)
        if normalized_mode not in {"natural", "standard"}:
            return False, "Unsupported voice quality mode. Use: natural or standard.", {}
        active_mode = speech_engine.set_quality_mode(normalized_mode)
        log_action(
            "voice_quality_set",
            "success",
            details={"requested": requested_mode, "active": active_mode},
        )
        return (
            True,
            f"Voice quality mode: {active_mode}\nHint: natural mode prefers tuned system voices before HF-TTS.",
            {"voice_quality_mode": active_mode},
        )

    if action == "stt_profile_status":
        message, snapshot = _format_stt_profile_status()
        return True, message, snapshot

    if action == "stt_profile_set":
        profile_name = args.get("profile", "")
        ok, message, snapshot = _apply_stt_profile(profile_name)
        log_action(
            "stt_profile_set",
            "success" if ok else "failed",
            details={"requested": profile_name, "active": snapshot.get("profile") if snapshot else None},
            error=None if ok else message,
        )
        return ok, message, snapshot

    if action == "hf_profile_status":
        message, snapshot = _format_hf_profile_status()
        return True, message, snapshot

    if action == "hf_profile_set":
        profile_name = args.get("profile", "")
        ok, message, snapshot = _apply_hf_speech_profile(profile_name)
        log_action(
            "hf_profile_set",
            "success" if ok else "failed",
            details={"requested": profile_name, "active": snapshot.get("profile") if snapshot else None},
            error=None if ok else message,
        )
        return ok, message, snapshot

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
