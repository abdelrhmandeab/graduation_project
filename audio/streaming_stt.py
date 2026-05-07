from __future__ import annotations

import queue
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

import re as _re

from audio.stt import normalize_arabic_post_transcript, transcribe_streaming_with_meta
from audio.vad import SileroVAD
from core.config import (
    AUDIO_CHUNK_SIZE,
    MAX_RECORD_DURATION,
    SAMPLE_RATE,
    VAD_CHAT_SILENCE_SECONDS,
    VAD_COMMAND_SILENCE_SECONDS,
    VAD_MIN_SPEECH_SECONDS,
    VAD_PREROLL_SECONDS,
    VAD_SILENCE_SECONDS,
    VAD_START_TIMEOUT_SECONDS,
)

_ARABIC_CHAR_RE = _re.compile(r"[؀-ۿ]")

# faster-whisper overrides for Arabic streaming sessions.
# beam_size=3  — faster than default 5, sufficient quality for real-time partials.
# vad_filter=False — Silero VAD runs externally; double-VAD causes mis-segmentation.
# language=None  — auto-detect to handle code-switched utterances.
# initial_prompt — primes the model with the wake word to bias toward Arabic.
_ARABIC_STREAMING_WHISPER_KWARGS = {
    "beam_size": 3,
    "vad_filter": False,
    "language": None,
    "initial_prompt": "جارفيس",
}


def _is_arabic_text(text: str) -> bool:
    return bool(_ARABIC_CHAR_RE.search(str(text or "")))


def _seconds_to_chunks(seconds: float) -> int:
    if seconds <= 0:
        return 1
    samples = int(seconds * SAMPLE_RATE)
    return max(1, int(np.ceil(samples / float(AUDIO_CHUNK_SIZE))))


def _chunk_rms(chunk: np.ndarray) -> float:
    normalized = chunk.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(np.square(normalized))))


def _write_wav_file(filename: str, sample_rate: int, audio_int16: np.ndarray) -> None:
    with wave.open(filename, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(audio_int16.tobytes())


def _resolve_silence_seconds(vad_mode: str, explicit_silence_seconds: Optional[float] = None) -> float:
    if explicit_silence_seconds is not None:
        return max(0.05, float(explicit_silence_seconds))

    mode = str(vad_mode or "command").strip().lower()
    if mode in {"chat", "conversation", "dialog", "turn"}:
        return float(VAD_CHAT_SILENCE_SECONDS)
    return float(VAD_COMMAND_SILENCE_SECONDS)


def _adaptive_silence_seconds(base_seconds: float, speech_seconds: float, max_seconds: float) -> float:
    fraction = min(1.0, speech_seconds / 3.0)
    return base_seconds + (max_seconds - base_seconds) * fraction


def _safe_callback(callback: Optional[Callable[..., None]], *args: Any) -> None:
    if callback is None:
        return
    try:
        callback(*args)
    except Exception:
        pass


def _transcribe_buffer(
    chunks: List[np.ndarray],
    filename: str,
    *,
    language_hint: Optional[str],
    on_partial: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    if not chunks:
        return {
            "text": "",
            "confidence": 0.0,
            "language": language_hint or "",
            "backend": "streaming",
            "method": "streaming",
            "fallback_used": False,
        }

    audio = np.concatenate(chunks, axis=0).astype(np.int16, copy=False)
    try:
        from audio.echo_cancel import noise_reducer
        audio = noise_reducer.reduce(audio, SAMPLE_RATE)
    except Exception:
        pass
    _write_wav_file(filename, SAMPLE_RATE, audio)

    # Use Arabic-optimised whisper params for Arabic streaming sessions.
    ar_kwargs = (
        dict(_ARABIC_STREAMING_WHISPER_KWARGS)
        if str(language_hint or "").startswith("ar")
        else None
    )
    result = transcribe_streaming_with_meta(
        filename, on_partial=on_partial, language_hint=language_hint, whisper_kwargs=ar_kwargs
    )
    result["samples"] = int(audio.shape[0])
    result["duration_seconds"] = float(audio.shape[0]) / float(SAMPLE_RATE)
    return result


class StreamingSTT:
    def __init__(
        self,
        *,
        filename: str = "input.wav",
        max_duration: float = MAX_RECORD_DURATION,
        vad_mode: str = "command",
        language_hint: Optional[str] = None,
        silence_seconds: Optional[float] = None,
        min_speech_seconds: Optional[float] = None,
        pre_roll_seconds: Optional[float] = None,
        start_timeout_seconds: Optional[float] = None,
        max_speech_seconds: Optional[float] = None,
        partial_interval_seconds: float = 0.45,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
    ) -> None:
        self.filename = filename
        self.max_duration = float(max_duration)
        self.vad_mode = str(vad_mode or "command")
        self.language_hint = language_hint
        self.silence_seconds = _resolve_silence_seconds(self.vad_mode, silence_seconds)
        self.min_speech_seconds = max(0.05, float(min_speech_seconds or VAD_MIN_SPEECH_SECONDS))
        self.pre_roll_seconds = max(0.0, float(pre_roll_seconds or VAD_PREROLL_SECONDS))
        self.start_timeout_seconds = max(0.2, float(start_timeout_seconds or VAD_START_TIMEOUT_SECONDS))
        self.max_speech_seconds = max(0.5, float(max_speech_seconds or max(1.5, self.max_duration * 0.65)))
        self.partial_interval_seconds = max(0.2, float(partial_interval_seconds))
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self._chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=128)
        self._stop_event = threading.Event()
        self._speech_started = False
        self._vad_detector = SileroVAD()
        self._vad_detector.reset()
        # Arabic partial stability — emit only after 2 consecutive identical windows
        self._ar_pending_partial: str = ""
        self._ar_pending_count: int = 0

    def _audio_callback(self, in_data, frames, time_info, status):  # pragma: no cover - called by sounddevice
        if in_data is None:
            return
        chunk = np.asarray(in_data).reshape(-1).astype(np.int16, copy=False)
        if chunk.size == 0:
            return
        try:
            self._chunk_queue.put_nowait(chunk.copy())
        except queue.Full:
            try:
                _ = self._chunk_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._chunk_queue.put_nowait(chunk.copy())
            except queue.Full:
                pass

    def _transcribe_partial(self, chunks: List[np.ndarray], partial_file: str, last_text: str) -> str:
        if not chunks:
            return last_text
        try:
            result = _transcribe_buffer(
                chunks,
                partial_file,
                language_hint=self.language_hint,
                on_partial=None,
            )
            text = str(result.get("text", "") or "").strip()

            # Normalise Arabic partials: strip tashkeel, normalise alef variants
            if _is_arabic_text(text):
                text = normalize_arabic_post_transcript(text)

            if not text:
                return last_text

            # Arabic stability gate: only emit if same text for 2 consecutive windows.
            # Whisper Arabic partials flicker more than English; this prevents
            # unstable fragments from triggering early intent detection.
            if _is_arabic_text(text):
                if text == self._ar_pending_partial:
                    self._ar_pending_count += 1
                else:
                    self._ar_pending_partial = text
                    self._ar_pending_count = 1
                if self._ar_pending_count >= 2 and text != last_text:
                    _safe_callback(self.on_partial, text)
                    return text
                return last_text

            if text != last_text:
                _safe_callback(self.on_partial, text)
                return text
        except Exception:
            return last_text
        return last_text

    def run(self) -> Dict[str, Any]:
        if sd is None:
            raise RuntimeError(
                "sounddevice is unavailable. Install sounddevice in the active Python environment."
            ) from _SOUNDDEVICE_IMPORT_ERROR

        started_at = time.perf_counter()
        max_chunks = _seconds_to_chunks(self.max_duration)
        start_timeout_chunks = _seconds_to_chunks(min(self.start_timeout_seconds, self.max_duration))
        max_speech_chunks = _seconds_to_chunks(min(self.max_duration, self.max_speech_seconds))
        base_silence_seconds = self.silence_seconds
        max_silence_seconds = max(base_silence_seconds, float(VAD_CHAT_SILENCE_SECONDS))
        min_speech_samples = int(max(1, self.min_speech_seconds * SAMPLE_RATE))
        pre_roll = deque(maxlen=_seconds_to_chunks(self.pre_roll_seconds))
        captured_chunks: List[np.ndarray] = []
        speech_detected = False
        speech_samples = 0
        silence_chunks = 0
        partial_text = ""
        last_partial_emit = 0.0
        speech_started_index = -1

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype=np.int16,
                blocksize=AUDIO_CHUNK_SIZE,
                callback=self._audio_callback,
            ):
                for index in range(max_chunks):
                    if self._stop_event.is_set():
                        break
                    try:
                        chunk = self._chunk_queue.get(timeout=0.1)
                    except queue.Empty:
                        if speech_detected and (time.perf_counter() - started_at) >= self.max_duration:
                            break
                        continue

                    if chunk.size == 0:
                        continue

                    rms = _chunk_rms(chunk)
                    if self._vad_detector is not None:
                        is_voice = bool(self._vad_detector.is_speech_chunk(chunk))
                    else:
                        is_voice = rms >= 0.001

                    if not speech_detected:
                        pre_roll.append(chunk.copy())
                        if is_voice:
                            speech_detected = True
                            speech_started_index = index
                            captured_chunks.extend(pre_roll)
                            pre_roll.clear()
                            silence_chunks = 0
                            _safe_callback(self.on_speech_start)
                        elif index >= start_timeout_chunks:
                            break

                    if not speech_detected:
                        continue

                    captured_chunks.append(chunk.copy())
                    speech_samples += int(chunk.size)
                    if is_voice:
                        silence_chunks = 0
                    else:
                        silence_chunks += 1

                    now = time.perf_counter()
                    should_emit_partial = (
                        speech_samples >= int(0.5 * SAMPLE_RATE)
                        and (now - last_partial_emit) >= self.partial_interval_seconds
                    )
                    if should_emit_partial:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as partial_tmp:
                            partial_text = self._transcribe_partial(captured_chunks, partial_tmp.name, partial_text)
                        last_partial_emit = now

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
        finally:
            self._stop_event.set()

        elapsed = time.perf_counter() - started_at
        if not speech_detected or not captured_chunks:
            return {
                "ok": False,
                "speech_detected": False,
                "duration_seconds": elapsed,
                "samples": 0,
                "text": "",
                "partial_text": partial_text,
            }

        _safe_callback(self.on_speech_end)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            final_result = _transcribe_buffer(
                captured_chunks,
                temp_file.name,
                language_hint=self.language_hint,
                on_partial=self.on_partial,
            )

        final_text = str(final_result.get("text", "") or "").strip()
        final_result.update(
            {
                "ok": True,
                "speech_detected": True,
                "duration_seconds": float(final_result.get("duration_seconds") or 0.0),
                "samples": int(final_result.get("samples") or 0),
                "partial_text": partial_text,
                "text": final_text,
            }
        )
        _safe_callback(self.on_final, dict(final_result))
        return final_result

    def stop(self) -> None:
        self._stop_event.set()


def record_utterance_streaming(
    filename: str = "input.wav",
    max_duration: float = MAX_RECORD_DURATION,
    vad_mode: str = "command",
    language_hint: Optional[str] = None,
    silence_seconds: Optional[float] = None,
    min_speech_seconds: Optional[float] = None,
    pre_roll_seconds: Optional[float] = None,
    start_timeout_seconds: Optional[float] = None,
    max_speech_seconds: Optional[float] = None,
    partial_interval_seconds: float = 0.45,
    on_partial: Optional[Callable[[str], None]] = None,
    on_final: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_speech_start: Optional[Callable[[], None]] = None,
    on_speech_end: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    engine = StreamingSTT(
        filename=filename,
        max_duration=max_duration,
        vad_mode=vad_mode,
        language_hint=language_hint,
        silence_seconds=silence_seconds,
        min_speech_seconds=min_speech_seconds,
        pre_roll_seconds=pre_roll_seconds,
        start_timeout_seconds=start_timeout_seconds,
        max_speech_seconds=max_speech_seconds,
        partial_interval_seconds=partial_interval_seconds,
        on_partial=on_partial,
        on_final=on_final,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
    )
    return engine.run()