import pathlib
import time
import wave
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from collections import deque

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

from audio.barge_in import consume_barge_in_wake
from core.dialogue_manager import consume_follow_up_wake
from core.config import (
    BARGE_IN_INTERRUPT_ON_WAKE,
    SAMPLE_RATE,
    WAKE_WORD,
    WAKE_WORD_AR_ONNX_PATH,
    WAKE_WORD_AR_ENABLED,
    WAKE_WORD_AUDIO_GAIN,
    WAKE_WORD_CHUNK_SIZE,
    WAKE_WORD_DETECTION_COOLDOWN_SECONDS,
    WAKE_WORD_IGNORE_WHILE_SPEAKING,
    WAKE_WORD_INPUT_DEVICE,
    WAKE_WORD_MODE,
    WAKE_WORD_SCORE_DEBUG,
    WAKE_WORD_SCORE_DEBUG_INTERVAL_SECONDS,
    WAKE_WORD_THRESHOLD,
    WAKE_WORD_USER_SPEAKER_ID,
    WAKE_WORD_USER_SAMPLES_DIR,
)
from core.logger import logger

_model = None
_arabic_onnx_model = None
_arabic_onnx_model_path = ""
_last_detection_ts = 0.0
_OPENWAKEWORD_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
_runtime_wake_word_settings = {
    "threshold": float(WAKE_WORD_THRESHOLD),
    "audio_gain": float(WAKE_WORD_AUDIO_GAIN),
    "detection_cooldown_seconds": float(WAKE_WORD_DETECTION_COOLDOWN_SECONDS),
}
_runtime_wake_word_phrase_settings = {
    "mode": str(WAKE_WORD_MODE or "both").strip().lower(),
    "arabic_enabled": bool(WAKE_WORD_AR_ENABLED),
    "arabic_onnx_path": str(WAKE_WORD_AR_ONNX_PATH or "").strip(),
}
_runtime_wake_word_behavior = {
    "ignore_while_speaking": bool(WAKE_WORD_IGNORE_WHILE_SPEAKING),
    "barge_in_interrupt_on_wake": bool(BARGE_IN_INTERRUPT_ON_WAKE),
}

_WAKE_SAMPLE_CAPTURE_SECONDS = 2.5


def _save_wake_activation_sample(audio_chunks, wake_source: str) -> None:
    sample_dir = str(WAKE_WORD_USER_SAMPLES_DIR or "").strip()
    if not sample_dir:
        return

    try:
        directory = pathlib.Path(sample_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if not audio_chunks:
            return

        audio = np.concatenate(list(audio_chunks), axis=0).astype(np.int16, copy=False).reshape(-1)
        if audio.size == 0:
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        suffix = "arabic" if str(wake_source or "").strip().lower() == "arabic" else "english"
        filename = f"wake_{suffix}_{timestamp}_{int(time.time() * 1000) % 1000:03d}.wav"
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
        logger.warning("Failed to save wake activation sample: %s", exc)


def _normalize_wake_mode(value) -> str:
    mode = str(value or "both").strip().lower()
    aliases = {
        "en": "english",
        "ar": "arabic",
        "dual": "both",
        "bilingual": "both",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"english", "arabic", "both"}:
        return "both"
    return mode


_runtime_wake_word_phrase_settings["mode"] = _normalize_wake_mode(
    _runtime_wake_word_phrase_settings.get("mode")
)


def get_runtime_wake_word_settings():
    return dict(_runtime_wake_word_settings)


def set_runtime_wake_word_settings(*, threshold=None, audio_gain=None, detection_cooldown_seconds=None):
    if threshold is not None:
        _runtime_wake_word_settings["threshold"] = max(0.05, min(0.95, float(threshold)))
    if audio_gain is not None:
        _runtime_wake_word_settings["audio_gain"] = max(0.5, min(3.0, float(audio_gain)))
    if detection_cooldown_seconds is not None:
        _runtime_wake_word_settings["detection_cooldown_seconds"] = max(
            0.2,
            min(3.0, float(detection_cooldown_seconds)),
        )
    return get_runtime_wake_word_settings()


def get_runtime_wake_word_phrase_settings():
    return dict(_runtime_wake_word_phrase_settings)


def set_runtime_wake_word_phrase_settings(
    *,
    mode=None,
    arabic_enabled=None,
    arabic_onnx_path=None,
):
    if mode is not None:
        _runtime_wake_word_phrase_settings["mode"] = _normalize_wake_mode(mode)
    if arabic_enabled is not None:
        _runtime_wake_word_phrase_settings["arabic_enabled"] = bool(arabic_enabled)
    if arabic_onnx_path is not None:
        _runtime_wake_word_phrase_settings["arabic_onnx_path"] = str(arabic_onnx_path or "").strip()
    return get_runtime_wake_word_phrase_settings()


def get_runtime_wake_mode() -> str:
    return str(_runtime_wake_word_phrase_settings.get("mode") or "both")


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


def _get_arabic_onnx_model(model_path: str):
    """Load an optional custom Arabic wake-word ONNX model.

    The custom model is required when Arabic wake mode is enabled. If loading
    fails we disable the Arabic layer so the assistant can still run.
    """
    global _arabic_onnx_model
    global _arabic_onnx_model_path

    candidate_path = str(model_path or "").strip()
    if not candidate_path:
        return None
    if _arabic_onnx_model is not None and _arabic_onnx_model_path == candidate_path:
        return _arabic_onnx_model

    try:
        import openwakeword
        from openwakeword.model import Model
    except Exception as exc:
        raise RuntimeError(
            "openwakeword is unavailable for the Arabic ONNX wake model."
        ) from exc

    path = pathlib.Path(candidate_path)
    if not path.exists():
        raise RuntimeError(f"Configured Arabic wake-word ONNX model was not found: {candidate_path}")

    _ensure_onnx_resources()
    _arabic_onnx_model = Model(wakeword_models=[str(path)], inference_framework="onnx")
    _arabic_onnx_model_path = candidate_path
    logger.info("Loaded custom Arabic wake-word ONNX model from %s", candidate_path)
    return _arabic_onnx_model


def _download_file(url, target_path):
    pathlib.Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    with open(target_path, "wb") as handle:
        handle.write(data)


def _ensure_onnx_resources():
    try:
        import openwakeword
    except Exception as exc:
        raise RuntimeError(
            "openwakeword is unavailable. Install openwakeword in the active Python environment."
        ) from exc

    package_dir = pathlib.Path(openwakeword.__file__).resolve().parent
    model_dir = package_dir / "resources" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    wakeword_key = WAKE_WORD.replace(" ", "_").strip().lower()
    required = [
        "melspectrogram.onnx",
        "embedding_model.onnx",
        f"{wakeword_key}_v0.1.onnx",
    ]

    for filename in required:
        target = model_dir / filename
        if target.exists():
            continue
        url = f"{_OPENWAKEWORD_RELEASE}/{filename}"
        try:
            _download_file(url, str(target))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download wake-word model resource '{filename}' from {url}."
            ) from exc


def _get_model():
    global _model
    if _model is not None:
        return _model

    try:
        from openwakeword.model import Model
    except Exception as exc:
        raise RuntimeError(
            "openwakeword is unavailable. Install openwakeword in the active environment."
        ) from exc

    try:
        _ensure_onnx_resources()
        _model = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize wake-word model with ONNX resources. "
            "Ensure network access for first-time model download."
        ) from exc
    return _model


def preload_runtime_wake_word():
    """Preload wake-word runtime resources so first listen has no cold start."""
    if sd is None:
        raise RuntimeError(
            "sounddevice is unavailable. Install sounddevice in the active Python environment."
        ) from _SOUNDDEVICE_IMPORT_ERROR

    phrase_runtime = get_runtime_wake_word_phrase_settings()
    wake_mode = _normalize_wake_mode(phrase_runtime.get("mode"))
    english_layer_enabled = wake_mode in {"english", "both"}
    arabic_layer_enabled = bool(phrase_runtime.get("arabic_enabled")) and wake_mode in {
        "english",
        "arabic",
        "both",
    }

    if not english_layer_enabled and not arabic_layer_enabled:
        raise RuntimeError("Wake word mode disabled all wake layers. Enable english, arabic, or both.")

    input_device = _resolve_input_device()

    english_model_loaded = False
    arabic_model_loaded = False
    arabic_onnx_model_loaded = False

    if english_layer_enabled:
        _get_model()
        english_model_loaded = True

    arabic_onnx_path = str(phrase_runtime.get("arabic_onnx_path") or WAKE_WORD_AR_ONNX_PATH or "").strip()
    if arabic_layer_enabled:
        if not arabic_onnx_path:
            if wake_mode == "arabic":
                raise RuntimeError("Arabic wake mode requires JARVIS_WAKE_WORD_AR_ONNX_PATH.")
            logger.warning("Arabic wake-word ONNX path missing; Arabic wake is disabled.")
            arabic_layer_enabled = False
        else:
            _get_arabic_onnx_model(arabic_onnx_path)
            arabic_onnx_model_loaded = True
            arabic_model_loaded = True

    return {
        "mode": wake_mode,
        "input_device": input_device if input_device is not None else "default",
        "english_layer_enabled": bool(english_layer_enabled),
        "arabic_layer_enabled": bool(arabic_layer_enabled),
        "english_model_loaded": bool(english_model_loaded),
        "arabic_onnx_model_loaded": bool(arabic_onnx_model_loaded),
        "arabic_model_loaded": bool(arabic_model_loaded),
    }


def _resolve_input_device():
    cfg = WAKE_WORD_INPUT_DEVICE
    if cfg is None or str(cfg).strip() == "":
        return None
    if isinstance(cfg, int):
        return cfg

    name_query = str(cfg).strip().lower()
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Failed to list audio devices: {exc}") from exc

    for idx, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        if name_query in str(device.get("name", "")).lower():
            return idx

    available = []
    for idx, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) > 0:
            available.append(f"{idx}:{device.get('name')}")
    raise RuntimeError(
        "Configured wake-word input device was not found. "
        f"WAKE_WORD_INPUT_DEVICE={cfg!r}. "
        f"Available input devices: {', '.join(available[:12])}"
    )


def listen_for_wake_word():
    global _last_detection_ts

    if sd is None:
        raise RuntimeError(
            "sounddevice is unavailable. Install sounddevice in the active Python environment."
        ) from _SOUNDDEVICE_IMPORT_ERROR

    runtime = get_runtime_wake_word_settings()
    phrase_runtime = get_runtime_wake_word_phrase_settings()

    wake_threshold = float(runtime["threshold"])
    wake_audio_gain = float(runtime["audio_gain"])
    wake_cooldown = float(runtime["detection_cooldown_seconds"])

    wake_mode = _normalize_wake_mode(phrase_runtime.get("mode"))
    english_layer_enabled = wake_mode in {"english", "both"}
    arabic_layer_enabled = bool(phrase_runtime.get("arabic_enabled")) and wake_mode in {"english", "arabic", "both"}

    if not english_layer_enabled and not arabic_layer_enabled:
        raise RuntimeError("Wake word mode disabled all wake layers. Enable english, arabic, or both.")

    model = _get_model() if english_layer_enabled else None
    arabic_model = None
    if arabic_layer_enabled:
        arabic_onnx_path = str(phrase_runtime.get("arabic_onnx_path") or WAKE_WORD_AR_ONNX_PATH or "").strip()
        if not arabic_onnx_path:
            if wake_mode == "arabic":
                raise RuntimeError("Arabic wake mode requires JARVIS_WAKE_WORD_AR_ONNX_PATH.")
            logger.warning("Arabic wake-word ONNX path missing; Arabic wake is disabled.")
            arabic_layer_enabled = False
        else:
            arabic_model = _get_arabic_onnx_model(arabic_onnx_path)

    input_device = _resolve_input_device()
    print(
        "[WakeWord] Waiting for wake word... "
        f"mode={wake_mode} device={input_device if input_device is not None else 'default'}"
    )

    predictor_executor = None
    if english_layer_enabled and arabic_layer_enabled:
        predictor_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="jarvis-wake-onnx")

    last_debug_ts = 0.0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            device=input_device,
            blocksize=WAKE_WORD_CHUNK_SIZE,
        ) as stream:
            recent_audio = deque(
                maxlen=max(1, int(round((_WAKE_SAMPLE_CAPTURE_SECONDS * SAMPLE_RATE) / float(WAKE_WORD_CHUNK_SIZE)))),
            )
            while True:
                # Exit immediately when the VAD barge-in monitor already captured
                # user speech — no need for a new wake word.
                if consume_barge_in_wake():
                    return "barge_in"
                # Exit immediately when the dialogue manager has opened a
                # follow-up window — the orchestrator will record directly.
                if consume_follow_up_wake():
                    return "follow_up"

                audio_chunk, _ = stream.read(WAKE_WORD_CHUNK_SIZE)
                audio_chunk = np.asarray(audio_chunk).reshape(-1).astype(np.int16, copy=False)
                recent_audio.append(audio_chunk.copy())
                if wake_audio_gain and wake_audio_gain != 1.0:
                    boosted = audio_chunk.astype(np.float32) * wake_audio_gain
                    audio_chunk = np.clip(boosted, -32768, 32767).astype(np.int16, copy=False)

                if english_layer_enabled and arabic_layer_enabled and predictor_executor:
                    futures = {
                        "english": predictor_executor.submit(model.predict, audio_chunk),
                        "arabic": predictor_executor.submit(arabic_model.predict, audio_chunk),
                    }
                    predictions = {}
                    for key, future in futures.items():
                        try:
                            predictions[key] = future.result()
                        except Exception as exc:
                            logger.warning("Wake-word predictor failed (%s): %s", key, exc)
                            predictions[key] = {}
                else:
                    predictions = {}
                    if english_layer_enabled:
                        predictions["english"] = model.predict(audio_chunk)
                    if arabic_layer_enabled:
                        predictions["arabic"] = arabic_model.predict(audio_chunk)

                if english_layer_enabled:
                    prediction = predictions.get("english") or {}
                    score = prediction.get(WAKE_WORD)
                    if score is None and prediction:
                        score = max(prediction.values())

                    if WAKE_WORD_SCORE_DEBUG and score is not None:
                        now = time.perf_counter()
                        if now - last_debug_ts >= float(WAKE_WORD_SCORE_DEBUG_INTERVAL_SECONDS):
                            rms = float(np.sqrt(np.mean(np.square(audio_chunk.astype(np.float32) / 32768.0))))
                            print(
                                f"[WakeWord] score={float(score):.6f} threshold={wake_threshold:.3f} "
                                f"rms={rms:.4f}"
                            )
                            last_debug_ts = now

                    if score is not None and score > wake_threshold:
                        now = time.perf_counter()
                        if now - _last_detection_ts < wake_cooldown:
                            continue
                        _last_detection_ts = now
                        _save_wake_activation_sample(recent_audio, "english")
                        print("[WakeWord] Wake word detected (english).")
                        return "english"

                if arabic_layer_enabled:
                    prediction = predictions.get("arabic") or {}
                    score = max(prediction.values()) if prediction else 0.0
                    if WAKE_WORD_SCORE_DEBUG and score is not None:
                        print(f"[WakeWord][AR] score={float(score):.6f}")
                    if score and float(score) > wake_threshold:
                        now = time.perf_counter()
                        if now - _last_detection_ts < wake_cooldown:
                            continue
                        _last_detection_ts = now
                        _save_wake_activation_sample(recent_audio, "arabic")
                        print("[WakeWord] Wake word detected (arabic).")
                        return "arabic"
    finally:
        if predictor_executor is not None:
            predictor_executor.shutdown(wait=False, cancel_futures=True)
