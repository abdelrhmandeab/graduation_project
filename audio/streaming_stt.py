from __future__ import annotations

import queue
import tempfile
import threading
import time
import wave
import os
from collections import deque
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

import re as _re

from audio.mic import get_runtime_vad_settings
from audio.stt import (
    normalize_arabic_post_transcript,
    transcribe_partial_with_meta,
    transcribe_backend_direct_with_meta,
    transcribe_streaming_with_meta,
)
from audio.vad import SileroVAD
from core.config import (
    AUDIO_CHUNK_SIZE,
    MAX_RECORD_DURATION,
    SAMPLE_RATE,
    STT_PARTIAL_MIN_SECONDS,
    STT_PARTIAL_INTERVAL_SECONDS,
    STT_PARTIAL_WINDOW_SECONDS,
    VAD_CHAT_SILENCE_SECONDS,
    VAD_COMMAND_SILENCE_SECONDS,
    VAD_ENERGY_THRESHOLD,
    VAD_MIN_SPEECH_SECONDS,
    VAD_PREROLL_SECONDS,
    VAD_SILERO_THRESHOLD,
    VAD_START_TIMEOUT_SECONDS,
)

_ARABIC_CHAR_RE = _re.compile(r"[؀-ۿ]")

# faster-whisper overrides for Arabic streaming sessions.
# beam_size=3  — faster than default 5, sufficient quality for real-time partials.
# vad_filter=False — Silero VAD runs externally; double-VAD causes mis-segmentation.
# initial_prompt — primes the model with the wake word to bias toward Arabic.
_ARABIC_STREAMING_WHISPER_KWARGS = {
    "beam_size": 3,
    "vad_filter": False,
    "initial_prompt": "جارفيس، افتح، اقفل، شغل، وقف، search، play، weather، دلوقتي، عايز، ممكن، من فضلك",
}


def _is_arabic_text(text: str) -> bool:
    return bool(_ARABIC_CHAR_RE.search(str(text or "")))


def _is_mixed_script_text(text: str) -> bool:
    value = str(text or "")
    return _is_arabic_text(value) and any("a" <= ch.lower() <= "z" for ch in value)


_STREAMING_VAD: Optional[SileroVAD] = None
_STREAMING_VAD_LOCK = threading.Lock()


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


def _get_streaming_vad(*, energy_threshold: float, silero_threshold: float) -> SileroVAD:
    global _STREAMING_VAD
    normalized_energy = max(0.001, float(energy_threshold))
    normalized_threshold = max(0.05, min(0.95, float(silero_threshold)))
    with _STREAMING_VAD_LOCK:
        if _STREAMING_VAD is None:
            _STREAMING_VAD = SileroVAD(
                energy_threshold=normalized_energy,
                threshold=normalized_threshold,
            )
        else:
            _STREAMING_VAD.energy_threshold = normalized_energy
            _STREAMING_VAD.threshold = normalized_threshold
        _STREAMING_VAD.reset()
        return _STREAMING_VAD


def _get_shared_streaming_vad() -> SileroVAD:
    return _get_streaming_vad(
        energy_threshold=VAD_ENERGY_THRESHOLD,
        silero_threshold=VAD_SILERO_THRESHOLD,
    )


def prewarm_streaming_vad() -> bool:
    """Initialize the shared streaming VAD before the first microphone turn."""
    try:
        return _get_shared_streaming_vad().is_ready()
    except Exception:
        return False


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
    use_local_only: bool = False,
    whisper_kwargs: Optional[Dict[str, Any]] = None,
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
    _write_wav_file(filename, SAMPLE_RATE, audio)

    # Use Arabic-optimised whisper params only when explicitly in Arabic mode.
    hint = str(language_hint or "").strip().lower()
    ar_kwargs = dict(_ARABIC_STREAMING_WHISPER_KWARGS) if hint.startswith("ar") else None
    if use_local_only:
        result = transcribe_backend_direct_with_meta(
            filename,
            backend="faster_whisper",
            on_partial=on_partial,
            language_hint=language_hint,
            whisper_kwargs=whisper_kwargs or ar_kwargs,
        )
    else:
        result = transcribe_streaming_with_meta(
            filename,
            on_partial=on_partial,
            language_hint=language_hint,
            whisper_kwargs=whisper_kwargs or ar_kwargs,
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
        energy_threshold: Optional[float] = None,
        silero_threshold: Optional[float] = None,
        partial_interval_seconds: float = STT_PARTIAL_INTERVAL_SECONDS,
        enable_partials: bool = False,
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
        self.energy_threshold = max(0.001, float(energy_threshold or VAD_ENERGY_THRESHOLD))
        self.silero_threshold = max(0.05, min(0.95, float(silero_threshold or VAD_SILERO_THRESHOLD)))
        self.partial_interval_seconds = max(0.2, float(partial_interval_seconds))
        self.enable_partials = bool(enable_partials)
        self.on_partial = on_partial
        self.on_final = on_final
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self._chunk_queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=128)
        self._stop_event = threading.Event()
        # Arabic partial stability — emit only after 2 consecutive identical windows
        self._ar_pending_partial: str = ""
        self._ar_pending_count: int = 0

    def _audio_callback(self, in_data, frames, _time_info, status):  # pragma: no cover - called by sounddevice
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

    def _transcribe_partial(self, chunks_snapshot: List[np.ndarray], last_text: str) -> str:
        if not chunks_snapshot or self._stop_event.is_set():
            return last_text
        partial_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                prefix="jarvis_partial_",
                suffix=".wav",
            ) as partial_tmp:
                partial_path = partial_tmp.name
            _write_wav_file(partial_path, SAMPLE_RATE, np.concatenate(chunks_snapshot, axis=0).astype(np.int16, copy=False))
            result = transcribe_partial_with_meta(
                partial_path,
                language_hint=self.language_hint,
                whisper_kwargs={
                    "beam_size": 1,
                    "best_of": 1,
                    "vad_filter": False,
                    "temperature": 0.0,
                },
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
                emit_threshold = 1 if _is_mixed_script_text(text) else 2
                if self._ar_pending_count >= emit_threshold and text != last_text:
                    _safe_callback(self.on_partial, text)
                    return text
                return last_text

            if text != last_text:
                _safe_callback(self.on_partial, text)
                return text
        except Exception:
            return last_text
        finally:
            if partial_path:
                try:
                    os.remove(partial_path)
                except Exception:
                    pass
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
        partial_executor = None
        partial_future = None
        partial_window_chunks = 0
        if self.enable_partials:
            partial_window_chunks = _seconds_to_chunks(float(STT_PARTIAL_WINDOW_SECONDS))
            import concurrent.futures as _cf

            partial_executor = _cf.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="stt-partial",
            )

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
                    is_voice = rms >= float(self.energy_threshold)

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
                    if self.enable_partials:
                        if partial_future is not None and partial_future.done():
                            try:
                                partial_text = partial_future.result()
                            except Exception:
                                pass
                            partial_future = None

                        should_emit_partial = (
                            speech_samples >= int(max(0.0, float(STT_PARTIAL_MIN_SECONDS)) * SAMPLE_RATE)
                            and (now - last_partial_emit) >= self.partial_interval_seconds
                        )
                        if should_emit_partial and partial_executor is not None and partial_future is None:
                            if partial_window_chunks > 0:
                                chunk_view = list(captured_chunks[-partial_window_chunks:])
                            else:
                                chunk_view = list(captured_chunks)
                            partial_future = partial_executor.submit(
                                self._transcribe_partial,
                                chunk_view,
                                partial_text,
                            )
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
            if partial_future is not None:
                try:
                    partial_future.cancel()
                except Exception:
                    pass
            if partial_executor is not None:
                try:
                    partial_executor.shutdown(wait=False)
                except Exception:
                    pass

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
        final_result = _transcribe_buffer(
            captured_chunks,
            self.filename,
            language_hint=self.language_hint,
            on_partial=self.on_partial if self.enable_partials else None,
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
    energy_threshold: Optional[float] = None,
    partial_interval_seconds: float = STT_PARTIAL_INTERVAL_SECONDS,
    enable_partials: bool = False,
    on_partial: Optional[Callable[[str], None]] = None,
    on_final: Optional[Callable[[Dict[str, Any]], None]] = None,
    on_speech_start: Optional[Callable[[], None]] = None,
    on_speech_end: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    runtime = get_runtime_vad_settings()
    if energy_threshold is None:
        energy_threshold = float(runtime.get("energy_threshold") or VAD_ENERGY_THRESHOLD)
    if min_speech_seconds is None:
        min_speech_seconds = float(runtime.get("min_speech_seconds") or VAD_MIN_SPEECH_SECONDS)
    if pre_roll_seconds is None:
        pre_roll_seconds = float(runtime.get("pre_roll_seconds") or VAD_PREROLL_SECONDS)
    if start_timeout_seconds is None:
        start_timeout_seconds = float(runtime.get("start_timeout_seconds") or VAD_START_TIMEOUT_SECONDS)
    if max_speech_seconds is None:
        max_speech_seconds = float(runtime.get("max_speech_seconds") or max(1.5, max_duration * 0.65))
    if silence_seconds is None:
        mode = str(vad_mode or "command").strip().lower()
        if mode in {"chat", "conversation", "dialog", "turn"}:
            silence_seconds = float(
                runtime.get("chat_silence_seconds")
                or runtime.get("silence_seconds")
                or VAD_CHAT_SILENCE_SECONDS
            )
        else:
            silence_seconds = float(
                runtime.get("command_silence_seconds")
                or runtime.get("silence_seconds")
                or VAD_COMMAND_SILENCE_SECONDS
            )
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
        energy_threshold=energy_threshold,
        partial_interval_seconds=partial_interval_seconds,
        enable_partials=enable_partials,
        on_partial=on_partial if enable_partials else None,
        on_final=on_final,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
    )
    return engine.run()
