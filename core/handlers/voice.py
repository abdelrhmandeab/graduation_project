from audio import mic as mic_capture
from audio import stt as stt_runtime
from audio import vad as vad_runtime
from audio import wake_word as wake_word_runtime
from audio.tts import speech_engine
from core.logger import logger
from core.metrics import metrics
from core.persona import persona_manager
from core.session_memory import session_memory
from os_control.action_log import log_action


_STT_PROFILE_PRESETS = {
    "quiet": {
        "mic": {
            "energy_threshold": 0.009,
            "silence_seconds": 1.05,
            "min_speech_seconds": 0.26,
            "pre_roll_seconds": 0.35,
            "start_timeout_seconds": 5.0,
            "max_speech_seconds": 8.5,
        },
        "stt": {
            "beam_size": 5,
            "best_of": 5,
            "vad_filter": True,
            "condition_on_previous_text": True,
            "quality_retry_threshold": 0.50,
            "quality_retry_beam_size": 4,
            "egyptalk_fallback_threshold": 0.48,
            "egyptalk_fallback_low_quality_score": 0.58,
            "egyptalk_fallback_min_text_chars": 4,
            "language_hint": "auto",
            "no_speech_threshold": 0.88,
            "log_prob_threshold": -2.0,
        },
        "speech_guard_threshold": 0.009,
    },
    "noisy": {
        "mic": {
            "energy_threshold": 0.017,
            "silence_seconds": 0.82,
            "min_speech_seconds": 0.30,
            "pre_roll_seconds": 0.50,
            "start_timeout_seconds": 3.4,
            "max_speech_seconds": 7.5,
        },
        "stt": {
            "beam_size": 5,
            "best_of": 5,
            "vad_filter": True,
            "condition_on_previous_text": True,
            "quality_retry_threshold": 0.62,
            "quality_retry_beam_size": 6,
            "egyptalk_fallback_threshold": 0.55,
            "egyptalk_fallback_low_quality_score": 0.66,
            "egyptalk_fallback_min_text_chars": 4,
            "language_hint": "auto",
            "no_speech_threshold": 0.93,
            "log_prob_threshold": -2.4,
        },
        "speech_guard_threshold": 0.017,
    },
    "arabic_egy": {
        "mic": {
            "energy_threshold": 0.014,
            "silence_seconds": 0.88,
            "min_speech_seconds": 0.28,
            "pre_roll_seconds": 0.55,
            "start_timeout_seconds": 3.6,
            "max_speech_seconds": 7.8,
        },
        "stt": {
            "beam_size": 2,
            "best_of": 2,
            "vad_filter": True,
            "condition_on_previous_text": False,
            "quality_retry_threshold": 0.56,
            "quality_retry_beam_size": 6,
            "egyptalk_fallback_threshold": 0.64,
            "egyptalk_fallback_low_quality_score": 0.74,
            "egyptalk_fallback_min_text_chars": 3,
            "language_hint": "auto",
            "no_speech_threshold": 0.95,
            "log_prob_threshold": -2.6,
            "egyptalk_chunk_seconds": 22.0,
            "egyptalk_stride_seconds": 5.0,
        },
        "speech_guard_threshold": 0.013,
    },
    "code_switched": {
        "mic": {
            "energy_threshold": 0.015,
            "silence_seconds": 0.70,
            "min_speech_seconds": 0.28,
            "pre_roll_seconds": 0.45,
            "start_timeout_seconds": 3.0,
            "max_speech_seconds": 7.0,
        },
        "stt": {
            "beam_size": 5,
            "best_of": 5,
            "vad_filter": True,
            "condition_on_previous_text": True,
            "quality_retry_threshold": 0.52,
            "quality_retry_beam_size": 6,
            "egyptalk_fallback_threshold": 0.56,
            "egyptalk_fallback_low_quality_score": 0.66,
            "egyptalk_fallback_min_text_chars": 4,
            "language_hint": "auto",
            "no_speech_threshold": 0.90,
            "log_prob_threshold": -2.2,
            "egyptalk_chunk_seconds": 18.0,
            "egyptalk_stride_seconds": 4.0,
        },
        "speech_guard_threshold": 0.012,
    },
    "auto": {
        "mic": {
            "energy_threshold": 0.015,
            "silence_seconds": 0.72,
            "min_speech_seconds": 0.28,
            "pre_roll_seconds": 0.50,
            "start_timeout_seconds": 3.1,
            "max_speech_seconds": 7.3,
        },
        "stt": {
            "beam_size": 5,
            "best_of": 5,
            "vad_filter": True,
            "condition_on_previous_text": True,
            "quality_retry_threshold": 0.54,
            "quality_retry_beam_size": 6,
            "egyptalk_fallback_threshold": 0.50,
            "egyptalk_fallback_low_quality_score": 0.60,
            "egyptalk_fallback_min_text_chars": 4,
            "language_hint": "auto",
            "no_speech_threshold": 0.90,
            "log_prob_threshold": -2.2,
            "egyptalk_chunk_seconds": 16.0,
            "egyptalk_stride_seconds": 4.0,
        },
        "speech_guard_threshold": 0.012,
    },
}
_AUDIO_UX_PROFILE_PRESETS = {
    "balanced": {
        "mic": {
            "energy_threshold": 0.015,
            "silence_seconds": 0.55,
            "min_speech_seconds": 0.28,
            "start_timeout_seconds": 3.0,
            "max_speech_seconds": 6.0,
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
            "energy_threshold": 0.014,
            "silence_seconds": 0.40,
            "min_speech_seconds": 0.25,
            "start_timeout_seconds": 2.5,
            "max_speech_seconds": 5.0,
        },
        "speech_guard_threshold": 0.009,
        "wake_word": {
            "threshold": 0.30,
            "audio_gain": 1.55,
            "detection_cooldown_seconds": 0.6,
        },
        "wake_behavior": {
            "ignore_while_speaking": False,
            "barge_in_interrupt_on_wake": True,
        },
        "tts": {
            "quality_mode": "natural",
            "rate_offset": 12,
            "pause_scale": 0.85,
        },
    },
    "robust": {
        "mic": {
            "energy_threshold": 0.020,
            "silence_seconds": 0.95,
            "min_speech_seconds": 0.45,
            "start_timeout_seconds": 4.5,
            "max_speech_seconds": 7.5,
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
_ACTIVE_AUDIO_UX_PROFILE = "custom"
_RUNTIME_PROFILES_INITIALIZED = False


def _normalize_profile_name(value):
    profile = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "quiet_room": "quiet",
        "noisy_room": "noisy",
        "هادئ": "quiet",
        "ضوضاء": "noisy",
        "arabic_eg": "arabic_egy",
        "arabic_egyptian": "arabic_egy",
        "arabic": "arabic_egy",
        "egyptian": "arabic_egy",
        "arabic_fast": "arabic_egy",
        "masri_fast": "arabic_egy",
        "egyptian_arabic": "arabic_egy",
        "masri": "arabic_egy",
        "مصري": "arabic_egy",
        "عربي_مصري": "arabic_egy",
        "عربي_مصرى": "arabic_egy",
        "code_switch": "code_switched",
        "codeswitched": "code_switched",
        "mixed": "code_switched",
        "mixed_language": "code_switched",
        "mixed_script": "code_switched",
        "مختلط": "code_switched",
        "auto_select": "auto",
        "automatic": "auto",
        "default": "auto",
        "تلقائي": "auto",
    }
    return aliases.get(profile, profile)


def _normalize_stt_backend_name(value):
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fw": "faster_whisper",
        "faster": "faster_whisper",
        "whisper": "faster_whisper",
        "nemo": "egyptalk_transformers",
        "nemo_egyptalk": "egyptalk_transformers",
        "egyptalk": "egyptalk_transformers",
        "egypt_talk": "egyptalk_transformers",
        "egyptalk_transformer": "egyptalk_transformers",
        "egypt_talk_transformers": "egyptalk_transformers",
        "egyptian": "egyptalk_transformers",
        "masri": "egyptalk_transformers",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"faster_whisper", "egyptalk_transformers"}:
        return ""
    return normalized


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
        "normal": "balanced",
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


def _normalize_wake_mode(value):
    mode = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "en": "english",
        "ar": "arabic",
        "dual": "both",
        "bilingual": "both",
    }
    return aliases.get(mode, mode)


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
        return False, "Unsupported STT profile. Use: quiet, noisy, arabic_egy, code_switched, or auto.", {}

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


def _audio_ux_runtime_snapshot():
    mic_settings = mic_capture.get_runtime_vad_settings()
    wake_settings = wake_word_runtime.get_runtime_wake_word_settings()
    wake_phrase_settings = wake_word_runtime.get_runtime_wake_word_phrase_settings()
    wake_behavior = wake_word_runtime.get_runtime_wake_word_behavior()
    tts_tuning = speech_engine.get_tuning_settings()
    persisted_profile = session_memory.get_audio_ux_profile() or "custom"
    return {
        "profile": _ACTIVE_AUDIO_UX_PROFILE,
        "persisted_profile": persisted_profile,
        "mic": mic_settings,
        "speech_guard_threshold": vad_runtime.get_energy_fallback_threshold(),
        "wake_word": wake_settings,
        "wake_phrase": wake_phrase_settings,
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
            fallback_ok, _fallback_message, _fallback_snapshot = _apply_stt_profile("auto", persist=False)
            if fallback_ok:
                summary.append("fallback_stt_profile:auto")
            else:
                summary.append("fallback_stt_profile_failed")
                restored_ok = False
    else:
        ok, _message, _snapshot = _apply_stt_profile("auto", persist=False)
        if ok:
            summary.append("default_stt_profile:auto")
        else:
            summary.append("default_stt_profile_failed")
            restored_ok = False

    summary.append("legacy_speech_profiles_removed")

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
        "stt_profiles_available: quiet,noisy,arabic_egy,code_switched,auto",
        f"mic_energy_threshold: {snapshot['mic']['energy_threshold']:.4f}",
        f"mic_silence_seconds: {snapshot['mic']['silence_seconds']:.2f}",
        f"mic_min_speech_seconds: {snapshot['mic']['min_speech_seconds']:.2f}",
        f"stt_beam_size: {snapshot['stt']['beam_size']}",
        f"stt_vad_filter: {snapshot['stt']['vad_filter']}",
        f"stt_language_hint: {snapshot['stt'].get('language_hint')}",
        f"stt_no_speech_threshold: {float(snapshot['stt'].get('no_speech_threshold', 0.0)):.2f}",
        f"stt_log_prob_threshold: {float(snapshot['stt'].get('log_prob_threshold', 0.0)):.2f}",
        f"stt_egyptalk_chunk_seconds: {float(snapshot['stt'].get('egyptalk_chunk_seconds', 0.0)):.1f}",
        f"stt_egyptalk_stride_seconds: {float(snapshot['stt'].get('egyptalk_stride_seconds', 0.0)):.1f}",
        f"stt_egyptalk_fallback_threshold: {float(snapshot['stt'].get('egyptalk_fallback_threshold', 0.0)):.2f}",
        f"stt_egyptalk_fallback_quality: {float(snapshot['stt'].get('egyptalk_fallback_low_quality_score', 0.0)):.2f}",
        f"stt_egyptalk_fallback_min_chars: {int(snapshot['stt'].get('egyptalk_fallback_min_text_chars', 0))}",
        f"stt_retry_threshold: {snapshot['stt']['quality_retry_threshold']:.2f}",
        f"speech_guard_threshold: {snapshot['speech_guard_threshold']:.4f}",
    ]
    return "\n".join(lines), snapshot


def _format_stt_backend_status():
    backend_info = stt_runtime.get_runtime_stt_backend_info()
    backend = backend_info.get("backend") or stt_runtime.get_runtime_stt_backend()
    stt_settings = stt_runtime.get_runtime_stt_settings()

    lines = [
        "STT Backend Status",
        f"stt_backend: {backend}",
        f"stt_language_hint: {stt_settings.get('language_hint')}",
        f"stt_egyptalk_enabled: {backend_info.get('egyptalk_enabled')}",
        f"stt_egyptalk_model: {backend_info.get('egyptalk_model')}",
    ]

    return "\n".join(lines), {
        "stt_backend": backend,
        "stt_language_hint": stt_settings.get("language_hint"),
        "stt_egyptalk_enabled": backend_info.get("egyptalk_enabled"),
        "stt_egyptalk_model": backend_info.get("egyptalk_model"),
    }


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
        f"wake_mode: {snapshot['wake_phrase']['mode']}",
        f"wake_arabic_enabled: {snapshot['wake_phrase']['arabic_enabled']}",
        f"wake_arabic_trigger_count: {len(snapshot['wake_phrase']['arabic_triggers'])}",
        f"wake_ignore_while_speaking: {snapshot['wake_behavior']['ignore_while_speaking']}",
        f"wake_barge_in_on_wake: {snapshot['wake_behavior']['barge_in_interrupt_on_wake']}",
        f"voice_quality_mode: {snapshot['voice_quality_mode']}",
        f"tts_rate_offset: {int(snapshot['tts_tuning']['rate_offset'])}",
        f"tts_pause_scale: {float(snapshot['tts_tuning']['pause_scale']):.2f}",
    ]
    return "\n".join(lines), snapshot


def _format_latency_status():
    snapshot = metrics.snapshot()
    stages = dict(snapshot.get("stages") or {})
    stage_order = [
        "wake_word",
        "record_audio",
        "speech_guard",
        "stt",
        "router",
        "pipeline",
        "backpressure_wait",
        "backpressure_drop",
    ]

    lines = [
        "Runtime Latency Status",
        f"audio_ux_profile: {_ACTIVE_AUDIO_UX_PROFILE}",
    ]

    if not stages:
        lines.append("pipeline_stage_metrics: no data yet")
    else:
        lines.append("pipeline_stage_metrics:")
        for stage_name in stage_order:
            stat = stages.get(stage_name)
            if not stat:
                continue
            lines.append(
                (
                    f"- {stage_name}: count={int(stat.get('count') or 0)}, "
                    f"success={float(stat.get('success_rate') or 0.0):.2%}, "
                    f"p50={float(stat.get('p50_ms') or 0.0):.1f}ms, "
                    f"p95={float(stat.get('p95_ms') or 0.0):.1f}ms"
                )
            )

    lines.append("Hint: use 'latency mode fast' for lower endpoint and phase transition delay.")
    return "\n".join(lines), {"audio_ux_profile": _ACTIVE_AUDIO_UX_PROFILE, "stages": stages}


def _format_wake_status():
    snapshot = wake_word_runtime.get_runtime_wake_word_phrase_settings()
    triggers = list(snapshot.get("arabic_triggers") or [])
    lines = [
        "Wake Word Status",
        f"wake_mode: {snapshot.get('mode')}",
        f"wake_phrase_enabled: {snapshot.get('arabic_enabled')}",
        f"wake_phrase_trigger_count: {len(triggers)}",
        f"wake_phrase_stt_model: {snapshot.get('ar_stt_model')}",
        f"wake_phrase_window_s: {float(snapshot.get('ar_chunk_seconds') or 0.0):.2f}",
        f"wake_phrase_interval_s: {float(snapshot.get('ar_check_interval_seconds') or 0.0):.2f}",
        f"wake_phrase_confirm_hits: {int(snapshot.get('ar_consecutive_hits_required') or 1)}",
        f"wake_phrase_confirm_window_s: {float(snapshot.get('ar_confirm_window_seconds') or 0.0):.2f}",
    ]
    if triggers:
        lines.append("wake_phrase_triggers:")
        for item in triggers[:12]:
            lines.append(f"- {item}")
        if len(triggers) > 12:
            lines.append(f"- ... (+{len(triggers) - 12} more)")
    else:
        lines.append("wake_phrase_triggers: none")
    snapshot["arabic_triggers"] = triggers
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
        speaking = speech_engine.is_speaking()
        enabled = speech_engine.is_enabled()
        stt_profile = _stt_runtime_snapshot()
        audio_ux = _audio_ux_runtime_snapshot()
        lines = [
            "Voice Status",
            f"speech_enabled: {enabled}",
            f"is_speaking: {speaking}",
            f"active_persona: {persona_manager.get_profile()}",
            f"speech_rate: {persona_manager.get_speech_rate()}",
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
            f"wake_mode: {audio_ux['wake_phrase']['mode']}",
            f"wake_phrase_trigger_count: {len(audio_ux['wake_phrase']['arabic_triggers'])}",
            f"wake_barge_in_on_wake: {audio_ux['wake_behavior']['barge_in_interrupt_on_wake']}",
            f"tts_rate_offset: {int(audio_ux['tts_tuning']['rate_offset'])}",
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

    if action == "latency_status":
        message, snapshot = _format_latency_status()
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

    if action == "wake_status":
        message, snapshot = _format_wake_status()
        return True, message, snapshot

    if action == "wake_mode_set":
        requested_mode = args.get("mode", "")
        normalized_mode = _normalize_wake_mode(requested_mode)
        if normalized_mode not in {"english", "arabic", "both"}:
            return False, "Unsupported wake mode. Use: english, arabic, or both.", {}
        active_mode = wake_word_runtime.set_runtime_wake_mode(normalized_mode)
        _mark_audio_ux_custom_profile()
        message, snapshot = _format_wake_status()
        log_action(
            "wake_mode_set",
            "success",
            details={"requested": requested_mode, "active": active_mode},
        )
        return True, message, snapshot

    if action == "wake_triggers_add":
        trigger = str(args.get("trigger") or "").strip()
        if not trigger:
            return False, "Missing wake trigger text. Example: wake triggers add ya jarvis", {}
        added, triggers = wake_word_runtime.add_runtime_wake_trigger(trigger)
        _mark_audio_ux_custom_profile()
        message, snapshot = _format_wake_status()
        log_action(
            "wake_triggers_add",
            "success" if added else "failed",
            details={"trigger": trigger, "count": len(triggers)},
            error=None if added else "duplicate_or_invalid_trigger",
        )
        if added:
            return True, message, snapshot
        return False, "Wake trigger already exists or is invalid.\n" + message, snapshot

    if action == "wake_triggers_remove":
        trigger = str(args.get("trigger") or "").strip()
        if not trigger:
            return False, "Missing wake trigger text. Example: wake triggers remove ya jarvis", {}
        removed, triggers = wake_word_runtime.remove_runtime_wake_trigger(trigger)
        _mark_audio_ux_custom_profile()
        message, snapshot = _format_wake_status()
        log_action(
            "wake_triggers_remove",
            "success" if removed else "failed",
            details={"trigger": trigger, "count": len(triggers)},
            error=None if removed else "trigger_not_found",
        )
        if removed:
            return True, message, snapshot
        return False, "Wake trigger was not found.\n" + message, snapshot

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
            f"Voice quality mode: {active_mode}",
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

    if action == "stt_backend_status":
        message, snapshot = _format_stt_backend_status()
        return True, message, snapshot

    if action == "stt_backend_set":
        requested_backend = args.get("backend", "")
        backend = _normalize_stt_backend_name(requested_backend)
        if not backend:
            return False, "Unsupported STT backend. Use: egyptalk_transformers or faster_whisper.", {}

        backend_info = stt_runtime.get_runtime_stt_backend_info()
        if backend == "egyptalk_transformers" and not bool(backend_info.get("egyptalk_enabled")):
            return (
                False,
                "Egyptian dialect backend is disabled. Set JARVIS_STT_EGYPTALK_ENABLED=true and restart Jarvis.",
                backend_info,
            )

        active_backend = stt_runtime.set_runtime_stt_backend(backend)
        message, snapshot = _format_stt_backend_status()
        log_action(
            "stt_backend_set",
            "success",
            details={"requested": requested_backend, "active": active_backend},
        )

        return True, f"Requested STT backend: {active_backend}\n{message}", snapshot

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
