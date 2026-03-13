import numpy as np
import pathlib
import time
import urllib.request

try:
    import sounddevice as sd
except Exception as exc:
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

from core.config import (
    SAMPLE_RATE,
    WAKE_WORD,
    WAKE_WORD_AUDIO_GAIN,
    WAKE_WORD_CHUNK_SIZE,
    WAKE_WORD_DETECTION_COOLDOWN_SECONDS,
    WAKE_WORD_INPUT_DEVICE,
    WAKE_WORD_SCORE_DEBUG,
    WAKE_WORD_SCORE_DEBUG_INTERVAL_SECONDS,
    WAKE_WORD_THRESHOLD,
)

_model = None
_last_detection_ts = 0.0
_OPENWAKEWORD_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"


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

    model = _get_model()
    input_device = _resolve_input_device()
    print(f"[WakeWord] Waiting for wake word... device={input_device if input_device is not None else 'default'}")

    last_debug_ts = 0.0
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.int16,
        device=input_device,
        blocksize=WAKE_WORD_CHUNK_SIZE,
    ) as stream:
        while True:
            audio_chunk, _ = stream.read(WAKE_WORD_CHUNK_SIZE)
            audio_chunk = np.asarray(audio_chunk).reshape(-1).astype(np.int16, copy=False)
            if WAKE_WORD_AUDIO_GAIN and WAKE_WORD_AUDIO_GAIN != 1.0:
                boosted = audio_chunk.astype(np.float32) * float(WAKE_WORD_AUDIO_GAIN)
                audio_chunk = np.clip(boosted, -32768, 32767).astype(np.int16, copy=False)

            prediction = model.predict(audio_chunk)
            score = prediction.get(WAKE_WORD)
            if score is None and prediction:
                score = max(prediction.values())

            if WAKE_WORD_SCORE_DEBUG and score is not None:
                now = time.perf_counter()
                if now - last_debug_ts >= float(WAKE_WORD_SCORE_DEBUG_INTERVAL_SECONDS):
                    rms = float(np.sqrt(np.mean(np.square(audio_chunk.astype(np.float32) / 32768.0))))
                    print(
                        f"[WakeWord] score={float(score):.6f} threshold={float(WAKE_WORD_THRESHOLD):.3f} "
                        f"rms={rms:.4f}"
                    )
                    last_debug_ts = now

            if score is not None and score > WAKE_WORD_THRESHOLD:
                now = time.perf_counter()
                if now - _last_detection_ts < float(WAKE_WORD_DETECTION_COOLDOWN_SECONDS):
                    continue
                _last_detection_ts = now
                print("[WakeWord] Wake word detected.")
                return
