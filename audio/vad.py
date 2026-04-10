import wave

import numpy as np
from core.config import VAD_ENERGY_THRESHOLD
_ENERGY_FALLBACK_THRESHOLD = max(0.001, float(VAD_ENERGY_THRESHOLD))


def set_energy_fallback_threshold(value):
    global _ENERGY_FALLBACK_THRESHOLD
    _ENERGY_FALLBACK_THRESHOLD = max(0.001, float(value))
    return _ENERGY_FALLBACK_THRESHOLD


def get_energy_fallback_threshold():
    return float(_ENERGY_FALLBACK_THRESHOLD)


def _energy_fallback_is_speech(audio_path):
    try:
        with wave.open(audio_path, "rb") as handle:
            frames = handle.readframes(handle.getnframes())
            sample_width = int(handle.getsampwidth())
            channels = int(handle.getnchannels())
    except Exception:
        return False

    if not frames:
        return False

    if sample_width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        return False

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    energy = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    return energy >= float(_ENERGY_FALLBACK_THRESHOLD)


def is_speech(audio_path):
    return _energy_fallback_is_speech(audio_path)
