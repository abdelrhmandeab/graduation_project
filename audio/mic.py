import math
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

try:
    import scipy.io.wavfile as scipy_wav
except Exception:
    scipy_wav = None

from core.config import (
    AUDIO_CHUNK_SIZE,
    MAX_RECORD_DURATION,
    SAMPLE_RATE,
    VAD_ENERGY_THRESHOLD,
    VAD_MIN_SPEECH_SECONDS,
    VAD_PREROLL_SECONDS,
    VAD_SILENCE_SECONDS,
    VAD_START_TIMEOUT_SECONDS,
)


def _seconds_to_chunks(seconds):
    if seconds <= 0:
        return 1
    samples = int(seconds * SAMPLE_RATE)
    return max(1, int(math.ceil(samples / float(AUDIO_CHUNK_SIZE))))


def _chunk_rms(chunk):
    normalized = chunk.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(np.square(normalized))))


def _write_wav_file(filename, sample_rate, audio_int16):
    if scipy_wav is not None:
        scipy_wav.write(filename, sample_rate, audio_int16)
        return

    # Fallback for environments without SciPy.
    with wave.open(filename, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(audio_int16.tobytes())


def record_utterance(
    filename="input.wav",
    max_duration=MAX_RECORD_DURATION,
    energy_threshold=VAD_ENERGY_THRESHOLD,
    silence_seconds=VAD_SILENCE_SECONDS,
    min_speech_seconds=VAD_MIN_SPEECH_SECONDS,
    pre_roll_seconds=VAD_PREROLL_SECONDS,
    start_timeout_seconds=VAD_START_TIMEOUT_SECONDS,
):
    if sd is None:
        raise RuntimeError(
            "sounddevice is unavailable. Install sounddevice in the active Python environment."
        ) from _SOUNDDEVICE_IMPORT_ERROR

    print("[Mic] Listening (VAD)...")
    started_at = time.perf_counter()
    pre_roll = deque(maxlen=_seconds_to_chunks(pre_roll_seconds))
    captured_chunks = []

    speech_detected = False
    silence_chunks = 0
    speech_samples = 0
    max_chunks = _seconds_to_chunks(max_duration)
    silence_chunks_target = _seconds_to_chunks(silence_seconds)
    min_speech_samples = int(max(1, min_speech_seconds * SAMPLE_RATE))
    start_timeout_chunks = _seconds_to_chunks(min(start_timeout_seconds, max_duration))

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            blocksize=AUDIO_CHUNK_SIZE,
        ) as stream:
            for index in range(max_chunks):
                data, _ = stream.read(AUDIO_CHUNK_SIZE)
                chunk = np.asarray(data).reshape(-1).astype(np.int16, copy=False)
                if chunk.size == 0:
                    continue

                rms = _chunk_rms(chunk)
                is_voice = rms >= float(energy_threshold)

                if not speech_detected:
                    pre_roll.append(chunk.copy())
                    if is_voice:
                        speech_detected = True
                        captured_chunks.extend(pre_roll)
                        pre_roll.clear()
                        silence_chunks = 0
                    elif index >= start_timeout_chunks:
                        break

                if speech_detected:
                    captured_chunks.append(chunk.copy())
                    speech_samples += int(chunk.size)
                    if is_voice:
                        silence_chunks = 0
                    else:
                        silence_chunks += 1

                    if speech_samples >= min_speech_samples and silence_chunks >= silence_chunks_target:
                        break
    except Exception as exc:
        raise RuntimeError(f"Microphone recording failed: {exc}") from exc

    elapsed = time.perf_counter() - started_at
    if not speech_detected or not captured_chunks:
        print("[Mic] No speech detected.")
        return {
            "ok": False,
            "speech_detected": False,
            "duration_seconds": elapsed,
            "samples": 0,
        }

    audio = np.concatenate(captured_chunks, axis=0).astype(np.int16, copy=False)
    _write_wav_file(filename, SAMPLE_RATE, audio)
    duration_seconds = float(audio.shape[0]) / float(SAMPLE_RATE)
    print(f"[Mic] Audio captured ({duration_seconds:.2f}s).")
    return {
        "ok": True,
        "speech_detected": True,
        "duration_seconds": duration_seconds,
        "samples": int(audio.shape[0]),
    }


def record_until_silence(filename="input.wav", max_duration=MAX_RECORD_DURATION):
    return record_utterance(filename=filename, max_duration=max_duration)
