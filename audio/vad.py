import wave

import numpy as np
from core.config import VAD_ENERGY_THRESHOLD
from core.logger import logger

_SILERO_READY = False
_silero_model = None
_silero_get_speech_timestamps = None
_silero_torch = None
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


def _resample_audio_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0 or audio.size == 0 or source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 1:
        repeat = max(1, int(round(float(target_rate) / float(source_rate))))
        return np.repeat(audio, repeat).astype(np.float32, copy=False)

    target_len = max(1, int(round(float(audio.shape[0]) * float(target_rate) / float(source_rate))))
    source_positions = np.arange(audio.shape[0], dtype=np.float64)
    target_positions = np.linspace(0.0, float(audio.shape[0] - 1), num=target_len, dtype=np.float64)
    return np.interp(target_positions, source_positions, audio).astype(np.float32, copy=False)


def _read_audio_for_silero(audio_path, target_rate=16000):
    if _silero_torch is None:
        raise RuntimeError("Silero torch runtime is unavailable")

    with wave.open(audio_path, "rb") as handle:
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        sample_rate = int(handle.getframerate())
        frame_count = int(handle.getnframes())
        raw_bytes = handle.readframes(frame_count)

    if sample_width == 1:
        audio = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width for Silero: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate != int(target_rate):
        audio = _resample_audio_linear(audio, sample_rate, int(target_rate))

    return _silero_torch.from_numpy(audio.copy())


def _ensure_silero():
    global _SILERO_READY, _silero_model, _silero_get_speech_timestamps, _silero_torch, _SILERO_ERROR
    if _SILERO_READY:
        return True
    if _SILERO_ERROR is not None:
        return False

    try:
        import torch
        from silero_vad import get_speech_timestamps

        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            verbose=False,
        )
        _silero_model = model
        _silero_get_speech_timestamps = get_speech_timestamps
        _silero_torch = torch
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
            wav = _read_audio_for_silero(audio_path, target_rate=16000)
            timestamps = _silero_get_speech_timestamps(wav, _silero_model, sampling_rate=16000)
            return len(timestamps) > 0
        except Exception as exc:
            _SILERO_RUNTIME_BROKEN = True
            if not _SILERO_FALLBACK_NOTICE_EMITTED:
                logger.info(
                    "Silero VAD runtime failed (%s); switching to energy fallback for this session.",
                    _exception_summary(exc),
                )
                _SILERO_FALLBACK_NOTICE_EMITTED = True
    elif not silero_ready and _SILERO_ERROR is not None and not _SILERO_FALLBACK_NOTICE_EMITTED:
        logger.info(
            "Silero VAD is unavailable (%s); using energy fallback.",
            _exception_summary(_SILERO_ERROR),
        )
        _SILERO_FALLBACK_NOTICE_EMITTED = True
    return _energy_fallback_is_speech(audio_path)
