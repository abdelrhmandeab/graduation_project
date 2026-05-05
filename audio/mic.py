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

from core.config import (
    AUDIO_CHUNK_SIZE,
    MAX_RECORD_DURATION,
    SAMPLE_RATE,
    VAD_CHAT_SILENCE_SECONDS,
    VAD_COMMAND_SILENCE_SECONDS,
    VAD_ENERGY_THRESHOLD,
    VAD_MIN_SPEECH_SECONDS,
    VAD_PREROLL_SECONDS,
    VAD_SILENCE_SECONDS,
    VAD_START_TIMEOUT_SECONDS,
)
from audio.vad import SileroVAD


_runtime_vad_settings = {
    "energy_threshold": float(VAD_ENERGY_THRESHOLD),
    "command_silence_seconds": float(VAD_COMMAND_SILENCE_SECONDS),
    "chat_silence_seconds": float(VAD_CHAT_SILENCE_SECONDS),
    "silence_seconds": float(VAD_SILENCE_SECONDS),
    "min_speech_seconds": float(VAD_MIN_SPEECH_SECONDS),
    "pre_roll_seconds": float(VAD_PREROLL_SECONDS),
    "start_timeout_seconds": float(VAD_START_TIMEOUT_SECONDS),
    "max_speech_seconds": max(1.5, float(MAX_RECORD_DURATION) * 0.65),
}

_runtime_vad_detector = None


def get_runtime_vad_settings():
    return dict(_runtime_vad_settings)


def set_runtime_vad_settings(
    *,
    energy_threshold=None,
    command_silence_seconds=None,
    chat_silence_seconds=None,
    silence_seconds=None,
    min_speech_seconds=None,
    pre_roll_seconds=None,
    start_timeout_seconds=None,
    max_speech_seconds=None,
):
    if energy_threshold is not None:
        _runtime_vad_settings["energy_threshold"] = max(0.001, float(energy_threshold))
    if command_silence_seconds is not None:
        _runtime_vad_settings["command_silence_seconds"] = max(0.05, float(command_silence_seconds))
    if chat_silence_seconds is not None:
        _runtime_vad_settings["chat_silence_seconds"] = max(0.05, float(chat_silence_seconds))
    if silence_seconds is not None:
        _runtime_vad_settings["silence_seconds"] = max(0.05, float(silence_seconds))
    if min_speech_seconds is not None:
        _runtime_vad_settings["min_speech_seconds"] = max(0.05, float(min_speech_seconds))
    if pre_roll_seconds is not None:
        _runtime_vad_settings["pre_roll_seconds"] = max(0.0, float(pre_roll_seconds))
    if start_timeout_seconds is not None:
        _runtime_vad_settings["start_timeout_seconds"] = max(0.2, float(start_timeout_seconds))
    if max_speech_seconds is not None:
        _runtime_vad_settings["max_speech_seconds"] = max(0.5, float(max_speech_seconds))
    return get_runtime_vad_settings()


def _resolve_vad_mode(vad_mode):
    mode = str(vad_mode or "command").strip().lower()
    if mode in {"chat", "conversation", "dialog", "turn"}:
        return "chat"
    return "command"


def _resolve_silence_seconds(vad_mode, explicit_silence_seconds=None):
    if explicit_silence_seconds is not None:
        return max(0.05, float(explicit_silence_seconds))

    runtime = get_runtime_vad_settings()
    mode = _resolve_vad_mode(vad_mode)
    if mode == "chat":
        return float(runtime.get("chat_silence_seconds") or runtime.get("silence_seconds") or VAD_CHAT_SILENCE_SECONDS)
    return float(runtime.get("command_silence_seconds") or runtime.get("silence_seconds") or VAD_COMMAND_SILENCE_SECONDS)


def _adaptive_silence_seconds(base_seconds, speech_seconds, max_seconds):
    """Scale silence threshold up with accumulated speech so long utterances get more grace."""
    fraction = min(1.0, speech_seconds / 3.0)
    return base_seconds + (max_seconds - base_seconds) * fraction


def _get_runtime_vad_detector():
    global _runtime_vad_detector
    if _runtime_vad_detector is None:
        runtime = get_runtime_vad_settings()
        _runtime_vad_detector = SileroVAD(
            energy_threshold=runtime["energy_threshold"],
        )
    return _runtime_vad_detector


def _seconds_to_chunks(seconds):
    if seconds <= 0:
        return 1
    samples = int(seconds * SAMPLE_RATE)
    return max(1, int(math.ceil(samples / float(AUDIO_CHUNK_SIZE))))


def _chunk_rms(chunk):
    normalized = chunk.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(np.square(normalized))))


def _write_wav_file(filename, sample_rate, audio_int16):
    with wave.open(filename, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(audio_int16.tobytes())


def record_utterance(
    filename="input.wav",
    max_duration=MAX_RECORD_DURATION,
    vad_mode="command",
    energy_threshold=None,
    silence_seconds=None,
    min_speech_seconds=None,
    pre_roll_seconds=None,
    start_timeout_seconds=None,
    max_speech_seconds=None,
):
    if sd is None:
        raise RuntimeError(
            "sounddevice is unavailable. Install sounddevice in the active Python environment."
        ) from _SOUNDDEVICE_IMPORT_ERROR

    print("[Mic] Listening (VAD)...")
    started_at = time.perf_counter()

    runtime = get_runtime_vad_settings()
    if energy_threshold is None:
        energy_threshold = runtime["energy_threshold"]
    if min_speech_seconds is None:
        min_speech_seconds = runtime["min_speech_seconds"]
    if pre_roll_seconds is None:
        pre_roll_seconds = runtime["pre_roll_seconds"]
    if start_timeout_seconds is None:
        start_timeout_seconds = runtime["start_timeout_seconds"]
    if max_speech_seconds is None:
        max_speech_seconds = runtime.get("max_speech_seconds", max_duration)
    base_silence_seconds = _resolve_silence_seconds(vad_mode, silence_seconds)
    max_silence_seconds = max(
        base_silence_seconds,
        float(runtime.get("chat_silence_seconds") or VAD_CHAT_SILENCE_SECONDS),
    )

    vad_detector = _get_runtime_vad_detector()
    if vad_detector is not None:
        vad_detector.reset()

    pre_roll = deque(maxlen=_seconds_to_chunks(pre_roll_seconds))
    captured_chunks = []

    speech_detected = False
    silence_chunks = 0
    speech_samples = 0
    max_chunks = _seconds_to_chunks(max_duration)
    max_speech_chunks = _seconds_to_chunks(min(max_duration, max(0.5, float(max_speech_seconds))))
    min_speech_samples = int(max(1, min_speech_seconds * SAMPLE_RATE))
    start_timeout_chunks = _seconds_to_chunks(min(start_timeout_seconds, max_duration))
    speech_started_index = -1

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
                if vad_detector is not None:
                    is_voice = bool(vad_detector.is_speech(chunk))
                else:
                    is_voice = rms >= float(energy_threshold)

                if not speech_detected:
                    pre_roll.append(chunk.copy())
                    if is_voice:
                        speech_detected = True
                        speech_started_index = index
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

                    silence_target = _seconds_to_chunks(
                        _adaptive_silence_seconds(
                            base_silence_seconds,
                            speech_samples / float(SAMPLE_RATE),
                            max_silence_seconds,
                        )
                    )
                    if speech_samples >= min_speech_samples and silence_chunks >= silence_target:
                        break
                    if speech_started_index >= 0 and (index - speech_started_index + 1) >= max_speech_chunks:
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


def record_until_silence(filename="input.wav", max_duration=MAX_RECORD_DURATION, vad_mode="command"):
    return record_utterance(filename=filename, max_duration=max_duration, vad_mode=vad_mode)
