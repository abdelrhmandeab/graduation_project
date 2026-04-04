import wave

import numpy as np
from core.config import VAD_ENERGY_THRESHOLD
from core.logger import logger

_SILERO_READY = False
_silero_model = None
_silero_get_speech_timestamps = None
_silero_read_audio = None
_SILERO_ERROR = None
_SILERO_RUNTIME_BROKEN = False
_SILERO_FALLBACK_NOTICE_EMITTED = False
_ENERGY_FALLBACK_THRESHOLD = max(0.001, float(VAD_ENERGY_THRESHOLD))


def _exception_summary(exc, *, max_chars=180):
    message = ""
    for item in getattr(exc, "args", ()):
        if isinstance(item, str) and item.strip():
            message = item.strip()
            break
    if not message:
        message = str(exc or "").strip() or exc.__class__.__name__

    headline = message.splitlines()[0].strip()
    lowered = message.lower()
    if "torchcodec" in lowered or "libtorchcodec" in lowered:
        headline = "torchcodec/torchaudio runtime mismatch"
    elif "torchaudio" in lowered and "codec" in lowered:
        headline = "torchaudio codec backend is unavailable"

    if len(headline) > max_chars:
        return headline[: max_chars - 3] + "..."
    return headline


def set_energy_fallback_threshold(value):
    global _ENERGY_FALLBACK_THRESHOLD
    _ENERGY_FALLBACK_THRESHOLD = max(0.001, float(value))
    return _ENERGY_FALLBACK_THRESHOLD


def get_energy_fallback_threshold():
    return float(_ENERGY_FALLBACK_THRESHOLD)


def _ensure_silero():
    global _SILERO_READY, _silero_model, _silero_get_speech_timestamps, _silero_read_audio, _SILERO_ERROR
    if _SILERO_READY:
        return True
    if _SILERO_ERROR is not None:
        return False

    try:
        import torch
        from silero_vad import get_speech_timestamps, read_audio

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        _silero_model = model
        _silero_get_speech_timestamps = get_speech_timestamps
        _silero_read_audio = read_audio
        _SILERO_READY = True
        return True
    except Exception as exc:
        _SILERO_ERROR = exc
        return False


def _energy_fallback_is_speech(audio_path):
    with wave.open(audio_path, "rb") as handle:
        frames = handle.readframes(handle.getnframes())
        sample_width = handle.getsampwidth()
        channels = handle.getnchannels()
    if sample_width != 2 or not frames:
        return False
    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16, copy=False)
    energy = float(np.sqrt(np.mean(np.square(audio.astype(np.float32) / 32768.0))))
    return energy >= float(_ENERGY_FALLBACK_THRESHOLD)


def is_speech(audio_path):
    global _SILERO_RUNTIME_BROKEN
    global _SILERO_FALLBACK_NOTICE_EMITTED
    silero_ready = _ensure_silero()
    if silero_ready and not _SILERO_RUNTIME_BROKEN:
        try:
            wav = _silero_read_audio(audio_path, sampling_rate=16000)
            timestamps = _silero_get_speech_timestamps(wav, _silero_model)
            return len(timestamps) > 0
        except Exception as exc:
            _SILERO_RUNTIME_BROKEN = True
            if not _SILERO_FALLBACK_NOTICE_EMITTED:
                logger.warning(
                    "Silero VAD runtime failed (%s); switching to energy fallback for this session.",
                    _exception_summary(exc),
                )
                _SILERO_FALLBACK_NOTICE_EMITTED = True
    elif not silero_ready and _SILERO_ERROR is not None and not _SILERO_FALLBACK_NOTICE_EMITTED:
        logger.warning(
            "Silero VAD is unavailable (%s); using energy fallback.",
            _exception_summary(_SILERO_ERROR),
        )
        _SILERO_FALLBACK_NOTICE_EMITTED = True
    return _energy_fallback_is_speech(audio_path)
