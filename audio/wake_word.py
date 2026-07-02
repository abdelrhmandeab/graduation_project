"""Unified bilingual wake-word listener.

One custom openWakeWord-compatible ONNX model detects the configured English
and Egyptian-Arabic phrases. Legacy language modes are accepted by the public
runtime API but map to the same unified model.
"""

import pathlib
import time
import wave
from collections import deque

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

from core.config import (
    SAMPLE_RATE,
    WAKE_WORD_AUDIO_GAIN,
    WAKE_WORD_CHUNK_SIZE,
    WAKE_WORD_CONFIRM_FRAMES,
    WAKE_WORD_DEPRECATED_KEYS,
    WAKE_WORD_DETECTION_COOLDOWN_SECONDS,
    WAKE_WORD_INPUT_DEVICE,
    WAKE_WORD_MIN_RMS,
    WAKE_WORD_SCORE_DEBUG,
    WAKE_WORD_SCORE_DEBUG_INTERVAL_SECONDS,
    WAKE_WORD_THRESHOLD,
    WAKE_WORD_UNIFIED_ONNX_PATH,
    WAKE_WORD_USER_SPEAKER_ID,
    WAKE_WORD_USER_SAMPLES_DIR,
)
from core.dialogue_manager import consume_follow_up_wake
from core.logger import get_logger, kv
from core.metrics import stage_timer
from core.shutdown import is_shutdown_requested

log = get_logger("wakeword")

_unified_model = None
_unified_model_path = ""
_last_detection_ts = 0.0
_deprecated_keys_logged = False
_WAKE_SAMPLE_CAPTURE_SECONDS = 2.5
_last_detection_audio: list | None = None

_runtime_wake_word_settings = {
    "threshold": float(WAKE_WORD_THRESHOLD),
    "audio_gain": float(WAKE_WORD_AUDIO_GAIN),
    "detection_cooldown_seconds": float(WAKE_WORD_DETECTION_COOLDOWN_SECONDS),
    "confirm_frames": int(WAKE_WORD_CONFIRM_FRAMES),
    "min_rms": float(WAKE_WORD_MIN_RMS),
}
_runtime_wake_word_phrase_settings = {
    "mode": "unified",
    "unified_onnx_path": str(WAKE_WORD_UNIFIED_ONNX_PATH or "").strip(),
}
_runtime_wake_word_behavior = {
    "ignore_while_speaking": False,
    "barge_in_interrupt_on_wake": True,
}


def _log_deprecated_keys_once() -> None:
    global _deprecated_keys_logged
    if _deprecated_keys_logged or not WAKE_WORD_DEPRECATED_KEYS:
        return
    _deprecated_keys_logged = True
    log.warning(
        "Ignoring legacy wake-word settings now replaced by the unified model: %s",
        ", ".join(WAKE_WORD_DEPRECATED_KEYS),
    )


def _save_wake_activation_sample(audio_chunks) -> None:
    sample_dir = str(WAKE_WORD_USER_SAMPLES_DIR or "").strip()
    if not sample_dir:
        return

    try:
        # Automatic detections are useful for later review but are not trusted
        # positives. Keep them outside the curated enrollment directory.
        directory = pathlib.Path(sample_dir).parent / "auto_captured"
        directory.mkdir(parents=True, exist_ok=True)
        if not audio_chunks:
            return

        audio = np.concatenate(list(audio_chunks), axis=0).astype(np.int16, copy=False).reshape(-1)
        if audio.size == 0:
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"wake_unified_{timestamp}_{int(time.time() * 1000) % 1000:03d}.wav"
        speaker_id = str(WAKE_WORD_USER_SPEAKER_ID or "speaker").strip() or "speaker"
        speaker_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in speaker_id)
        target_dir = directory / speaker_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / filename

        with wave.open(str(target_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(int(SAMPLE_RATE))
            handle.writeframes(audio.tobytes())
    except Exception as exc:
        log.warning("Failed to save wake activation sample: %s", exc)


def _normalize_wake_mode(_value) -> str:
    return "unified"


def get_runtime_wake_word_settings():
    return dict(_runtime_wake_word_settings)


def set_runtime_wake_word_settings(
    *,
    threshold=None,
    audio_gain=None,
    detection_cooldown_seconds=None,
    confirm_frames=None,
    min_rms=None,
):
    if threshold is not None:
        _runtime_wake_word_settings["threshold"] = max(-100.0, min(1.0, float(threshold)))
    if audio_gain is not None:
        _runtime_wake_word_settings["audio_gain"] = max(0.5, min(3.0, float(audio_gain)))
    if detection_cooldown_seconds is not None:
        _runtime_wake_word_settings["detection_cooldown_seconds"] = max(
            0.2,
            min(3.0, float(detection_cooldown_seconds)),
        )
    if confirm_frames is not None:
        _runtime_wake_word_settings["confirm_frames"] = max(1, int(confirm_frames))
    if min_rms is not None:
        _runtime_wake_word_settings["min_rms"] = max(0.0, float(min_rms))
    return get_runtime_wake_word_settings()


def get_runtime_wake_word_phrase_settings():
    return dict(_runtime_wake_word_phrase_settings)


def set_runtime_wake_word_phrase_settings(*, mode=None, unified_onnx_path=None):
    if mode is not None:
        _runtime_wake_word_phrase_settings["mode"] = _normalize_wake_mode(mode)
    if unified_onnx_path is not None:
        _runtime_wake_word_phrase_settings["unified_onnx_path"] = str(unified_onnx_path or "").strip()
    return get_runtime_wake_word_phrase_settings()


def get_runtime_wake_mode() -> str:
    return "unified"


def set_runtime_wake_mode(mode: str) -> str:
    set_runtime_wake_word_phrase_settings(mode=mode)
    return get_runtime_wake_mode()


def get_runtime_wake_word_behavior():
    return dict(_runtime_wake_word_behavior)


def set_runtime_wake_word_behavior(*, ignore_while_speaking=None, barge_in_interrupt_on_wake=None):
    if ignore_while_speaking is not None:
        _runtime_wake_word_behavior["ignore_while_speaking"] = bool(ignore_while_speaking)
    if barge_in_interrupt_on_wake is not None:
        _runtime_wake_word_behavior["barge_in_interrupt_on_wake"] = bool(barge_in_interrupt_on_wake)
    return get_runtime_wake_word_behavior()


def invalidate_model_cache() -> None:
    """Force the next ``_get_unified_model`` call to reload from disk."""
    global _unified_model, _unified_model_path
    _unified_model = None
    _unified_model_path = ""
    log.info("Wake-word model cache invalidated; next listen cycle will reload.")


def get_last_detection_audio() -> list | None:
    """Return the audio chunks captured around the most recent detection."""
    return _last_detection_audio


def _resolve_unified_model_path(model_path: str) -> pathlib.Path:
    candidate = pathlib.Path(str(model_path or "").strip())
    if not str(candidate):
        raise RuntimeError("JARVIS_WAKE_WORD_UNIFIED_ONNX_PATH is empty.")
    if not candidate.is_absolute():
        candidate = pathlib.Path(__file__).resolve().parents[1] / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise RuntimeError(f"Unified wake-word ONNX model was not found: {candidate}")
    return candidate


def _get_unified_model(model_path: str):
    global _unified_model, _unified_model_path

    candidate = _resolve_unified_model_path(model_path)
    candidate_text = str(candidate)
    if _unified_model is not None and _unified_model_path == candidate_text:
        return _unified_model

    try:
        from openwakeword.model import Model
    except Exception as exc:
        raise RuntimeError(
            "openwakeword is unavailable. Install openwakeword in the active environment."
        ) from exc

    try:
        _unified_model = Model(wakeword_models=[candidate_text], inference_framework="onnx")
    except Exception as exc:
        raise RuntimeError(f"Failed to load unified wake-word model: {candidate}") from exc
    _unified_model_path = candidate_text
    log.info("Unified wake-word model loaded: %s", candidate)
    return _unified_model


def preload_runtime_wake_word():
    """Preload the single unified model and validate the configured microphone."""
    if sd is None:
        raise RuntimeError(
            "sounddevice is unavailable. Install sounddevice in the active Python environment."
        ) from _SOUNDDEVICE_IMPORT_ERROR

    _log_deprecated_keys_once()
    input_device = _resolve_input_device()
    model_path = _runtime_wake_word_phrase_settings["unified_onnx_path"]
    _get_unified_model(model_path)
    return {
        "mode": "unified",
        "input_device": input_device if input_device is not None else "default",
        "unified_model_loaded": True,
    }


def _resolve_input_device():
    cfg = WAKE_WORD_INPUT_DEVICE
    if cfg is None or str(cfg).strip() == "":
        return None
    if isinstance(cfg, int) or str(cfg).strip().isdigit():
        return int(cfg)

    name_query = str(cfg).strip().lower()
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Failed to list audio devices: {exc}") from exc

    for idx, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) > 0 and name_query in str(device.get("name", "")).lower():
            return idx

    available = [
        f"{idx}:{device.get('name')}"
        for idx, device in enumerate(devices)
        if int(device.get("max_input_channels", 0)) > 0
    ]
    raise RuntimeError(
        "Configured wake-word input device was not found. "
        f"JARVIS_WAKE_WORD_INPUT_DEVICE={cfg!r}. "
        f"Available input devices: {', '.join(available[:12])}"
    )


def _prediction_score(model, prediction) -> float:
    if not prediction:
        return 0.0
    prediction_key = next(iter(getattr(model, "prediction_buffer", {}) or {}), None)
    score = prediction.get(prediction_key) if prediction_key else None
    if score is None:
        score = next(iter(prediction.values()), 0.0)
    return float(score or 0.0)


def listen_for_wake_word():
    global _last_detection_ts, _last_detection_audio

    if sd is None:
        raise RuntimeError(
            "sounddevice is unavailable. Install sounddevice in the active Python environment."
        ) from _SOUNDDEVICE_IMPORT_ERROR

    _log_deprecated_keys_once()
    runtime = get_runtime_wake_word_settings()
    threshold = float(runtime["threshold"])
    cooldown = float(runtime["detection_cooldown_seconds"])
    confirm_frames = max(1, int(runtime["confirm_frames"]))
    min_rms = max(0.0, float(runtime["min_rms"]))
    model = _get_unified_model(_runtime_wake_word_phrase_settings["unified_onnx_path"])
    input_device = _resolve_input_device()

    log.info(
        "Listening for wake word; say 'Jarvis' / 'يا جارفس' "
        "(device=%s, threshold=%.2f, confirm_frames=%d, min_rms=%.4f).",
        input_device if input_device is not None else "default",
        threshold,
        confirm_frames,
        min_rms,
    )

    last_summary_ts = time.perf_counter()
    inference_durations_ms = []
    summary_max_score = float("-inf")
    summary_max_rms = 0.0
    consecutive_hits = 0

    _PRIME_SECONDS = 3.0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16,
        device=input_device,
        blocksize=WAKE_WORD_CHUNK_SIZE,
    ) as stream:
        prime_chunks = max(1, int(round((_PRIME_SECONDS * SAMPLE_RATE) / float(WAKE_WORD_CHUNK_SIZE))))
        for _ in range(prime_chunks):
            chunk, _ = stream.read(WAKE_WORD_CHUNK_SIZE)
            chunk = np.asarray(chunk).reshape(-1).astype(np.int16, copy=False)
            model.predict(chunk)
        log.debug("Feature extractor primed with %d chunks of live audio.", prime_chunks)

        recent_audio = deque(
            maxlen=max(1, int(round((_WAKE_SAMPLE_CAPTURE_SECONDS * SAMPLE_RATE) / float(WAKE_WORD_CHUNK_SIZE))))
        )
        while True:
            if is_shutdown_requested():
                log.debug("Wake-word listener stopped for shutdown.")
                return "shutdown"
            if consume_follow_up_wake():
                return "follow_up"

            audio_chunk, _ = stream.read(WAKE_WORD_CHUNK_SIZE)
            audio_chunk = np.asarray(audio_chunk).reshape(-1).astype(np.int16, copy=False)
            recent_audio.append(audio_chunk.copy())

            with stage_timer("wake_inference") as inference_timing:
                prediction = model.predict(audio_chunk)
            inference_durations_ms.append(inference_timing.elapsed * 1000.0)
            score = _prediction_score(model, prediction)
            summary_max_score = max(summary_max_score, score)
            rms = float(np.sqrt(np.mean(np.square(audio_chunk.astype(np.float32) / 32768.0))))
            summary_max_rms = max(summary_max_rms, rms)

            now = time.perf_counter()
            debug_interval = 1.0 if WAKE_WORD_SCORE_DEBUG else float(WAKE_WORD_SCORE_DEBUG_INTERVAL_SECONDS)
            if now - last_summary_ts >= debug_interval:
                if inference_durations_ms:
                    durations = np.asarray(inference_durations_ms, dtype=np.float64)
                    kv(
                        "wakeword",
                        inferences=len(inference_durations_ms),
                        p50_ms=f"{float(np.percentile(durations, 50)):.2f}",
                        p95_ms=f"{float(np.percentile(durations, 95)):.2f}",
                        score=f"{score:.4f}",
                        max_score=f"{summary_max_score:.3f}",
                        max_rms=f"{summary_max_rms:.4f}",
                    )
                inference_durations_ms.clear()
                summary_max_score = float("-inf")
                summary_max_rms = 0.0
                last_summary_ts = now

            consecutive_hits = consecutive_hits + 1 if score >= threshold and rms >= min_rms else 0
            if consecutive_hits < confirm_frames:
                continue
            consecutive_hits = 0
            if now - _last_detection_ts < cooldown:
                continue

            _last_detection_ts = now
            _last_detection_audio = list(recent_audio)
            _save_wake_activation_sample(recent_audio)

            from core.runtime_coordinator import RuntimePhase, coordinator
            phase = coordinator.current_phase
            if phase in (RuntimePhase.THINKING, RuntimePhase.SPEAKING):
                coordinator.request_interrupt()
                log.info("Wake interrupt fired (was %s, score=%.3f).", phase.value, score)
                return "wake"
            elif phase == RuntimePhase.EXECUTING_COMMAND:
                log.info("Wake heard but blocked (phase=%s, score=%.3f).", phase.value, score)
                consecutive_hits = 0
                continue
            else:
                log.info("Wake word detected (score=%.3f, rms=%.4f).", score, rms)
                return "wake"
