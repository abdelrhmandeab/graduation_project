import wave

import numpy as np

_SILERO_READY = False
_silero_model = None
_silero_get_speech_timestamps = None
_silero_read_audio = None
_SILERO_ERROR = None


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
    return energy >= 0.01


def is_speech(audio_path):
    if _ensure_silero():
        wav = _silero_read_audio(audio_path, sampling_rate=16000)
        timestamps = _silero_get_speech_timestamps(wav, _silero_model)
        return len(timestamps) > 0
    return _energy_fallback_is_speech(audio_path)
