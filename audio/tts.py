import asyncio
import io
import inspect
import re
import threading
import time
import wave

from core.config import (
    TTS_ARABIC_SPOKEN_DIALECT,
    TTS_DEFAULT_BACKEND,
    TTS_DEFAULT_RATE,
    TTS_EDGE_ARABIC_PITCH,
    TTS_EDGE_ARABIC_RATE,
    TTS_EDGE_ARABIC_VOLUME,
    TTS_EDGE_ARABIC_VOICE,
    TTS_EDGE_ARABIC_VOICE_FALLBACKS,
    TTS_EDGE_MIXED_SCRIPT_CHUNKING,
    TTS_EDGE_MIXED_SCRIPT_MAX_CHUNKS,
    TTS_EDGE_MIXED_SCRIPT_MAX_TEXT_LENGTH,
    TTS_EDGE_RATE,
    TTS_EDGE_VOICE,
    TTS_EGYPTIAN_COLLOQUIAL_REWRITE,
    TTS_ENABLED,
    TTS_QUALITY_MODE,
    TTS_SIMULATED_CHAR_DELAY,
)
from core.logger import logger
from core.metrics import metrics
from core.persona import persona_manager


def _contains_arabic(text):
    for ch in str(text or ""):
        code = ord(ch)
        if (
            0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0x08A0 <= code <= 0x08FF
            or 0xFB50 <= code <= 0xFDFF
            or 0xFE70 <= code <= 0xFEFF
        ):
            return True
    return False


def _contains_latin(text):
    for ch in str(text or ""):
        if "a" <= ch.lower() <= "z":
            return True
    return False


def _count_arabic_letters(text):
    count = 0
    for ch in str(text or ""):
        code = ord(ch)
        if (
            0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0x08A0 <= code <= 0x08FF
            or 0xFB50 <= code <= 0xFDFF
            or 0xFE70 <= code <= 0xFEFF
        ):
            count += 1
    return count


def _count_latin_letters(text):
    count = 0
    for ch in str(text or ""):
        if "a" <= ch.lower() <= "z":
            count += 1
    return count


def _voice_descriptor(voice):
    parts = [
        str(getattr(voice, "id", "") or ""),
        str(getattr(voice, "name", "") or ""),
    ]
    languages = getattr(voice, "languages", None) or []
    for item in languages:
        if isinstance(item, bytes):
            parts.append(item.decode("utf-8", errors="ignore"))
        else:
            parts.append(str(item))
    return " ".join(parts).lower()


_EGYPTIAN_TTS_PHRASE_REPLACEMENTS = (
    ("\u0628\u0627\u0644\u062a\u0623\u0643\u064a\u062f", "\u0627\u0643\u064a\u062f"),
    ("\u064a\u064f\u0641\u0636\u0644", "\u0627\u0644\u0623\u062d\u0633\u0646"),
    ("\u064a\u0641\u0636\u0644", "\u0627\u0644\u0623\u062d\u0633\u0646"),
    ("\u064a\u062d\u062a\u0627\u062c \u0625\u0644\u0649", "\u0645\u062d\u062a\u0627\u062c"),
    ("\u064a\u062d\u062a\u0627\u062c", "\u0645\u062d\u062a\u0627\u062c"),
    ("\u0645\u0646 \u0641\u0636\u0644\u0643", "\u0644\u0648 \u0633\u0645\u062d\u062a"),
    ("\u0645\u0646 \u0641\u0636\u0644\u0643\u060c", "\u0644\u0648 \u0633\u0645\u062d\u062a\u060c"),
    ("\u0644\u0627 \u064a\u0648\u062c\u062f", "\u0645\u0641\u064a\u0634"),
)

_EGYPTIAN_TTS_WORD_REPLACEMENTS = (
    ("\u064a\u0645\u0643\u0646\u0646\u064a", "\u0627\u0642\u062f\u0631"),
    ("\u0623\u0633\u062a\u0637\u064a\u0639", "\u0627\u0642\u062f\u0631"),
    ("\u064a\u0645\u0643\u0646\u0643", "\u062a\u0642\u062f\u0631"),
    ("\u064a\u0645\u0643\u0646\u0643\u0645", "\u062a\u0642\u062f\u0631\u0648\u0627"),
    ("\u0628\u0625\u0645\u0643\u0627\u0646\u0643", "\u062a\u0642\u062f\u0631"),
    ("\u0627\u0644\u0622\u0646", "\u062f\u0644\u0648\u0642\u062a\u064a"),
    ("\u0647\u0630\u0627", "\u062f\u0647"),
    ("\u0647\u0630\u0647", "\u062f\u064a"),
    ("\u0645\u062b\u0644", "\u0632\u064a"),
    ("\u0623\u064a\u0636\u0627", "\u0643\u0645\u0627\u0646"),
    ("\u0644\u064a\u0633", "\u0645\u0634"),
)


def _rewrite_to_egyptian_colloquial(text):
    updated = str(text or "")
    if not updated:
        return updated

    for source, target in _EGYPTIAN_TTS_PHRASE_REPLACEMENTS:
        updated = updated.replace(source, target)

    for source, target in _EGYPTIAN_TTS_WORD_REPLACEMENTS:
        updated = re.sub(rf"(?<!\S){re.escape(source)}(?!\S)", target, updated)

    updated = re.sub(r"\s+", " ", updated).strip()
    return updated


class SpeechEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._process = None
        self._pyttsx3_engine = None
        self._runtime_backend = str(TTS_DEFAULT_BACKEND or "auto").strip().lower()
        self._quality_mode = self._normalize_quality_mode(TTS_QUALITY_MODE)
        self._runtime_rate_offset = 0
        self._runtime_pause_scale = 1.0
        self._last_pyttsx3_voice_id = ""
        self._last_pyttsx3_voice_name = ""
        self._edge_tts_unavailable_logged = False
        self._edge_tts_decode_warning_logged = False
        self._edge_tts_unsupported_voices = set()
        self._last_edge_arabic_voice = ""
        self._enabled = bool(TTS_ENABLED)

    def _normalize_backend(self, backend):
        raw = str(backend or "auto").strip().lower()
        aliases = {
            "edge": "edge_tts",
            "edgetts": "edge_tts",
        }
        resolved = aliases.get(raw, raw)
        allowed = {"auto", "console", "pyttsx3", "edge_tts"}
        if resolved not in allowed:
            return "auto"
        return resolved

    def _normalize_quality_mode(self, mode):
        raw = str(mode or "natural").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "human": "natural",
            "natural_voice": "natural",
            "default": "standard",
            "balanced": "standard",
            "robot": "standard",
            "robotic": "standard",
        }
        normalized = aliases.get(raw, raw)
        if normalized not in {"natural", "standard"}:
            return "natural"
        return normalized

    def get_backend(self):
        with self._lock:
            return self._normalize_backend(self._runtime_backend)

    def set_backend(self, backend):
        with self._lock:
            self._runtime_backend = self._normalize_backend(backend)
            return self._runtime_backend

    def get_quality_mode(self):
        with self._lock:
            return self._normalize_quality_mode(self._quality_mode)

    def set_quality_mode(self, mode):
        with self._lock:
            self._quality_mode = self._normalize_quality_mode(mode)
            return self._quality_mode

    def get_last_pyttsx3_voice(self):
        with self._lock:
            return {
                "id": str(self._last_pyttsx3_voice_id or ""),
                "name": str(self._last_pyttsx3_voice_name or ""),
            }

    def get_tuning_settings(self):
        with self._lock:
            return {
                "rate_offset": int(self._runtime_rate_offset),
                "pause_scale": float(self._runtime_pause_scale),
            }

    def set_tuning_settings(self, *, rate_offset=None, pause_scale=None):
        with self._lock:
            if rate_offset is not None:
                self._runtime_rate_offset = int(max(-60, min(60, int(rate_offset))))
            if pause_scale is not None:
                self._runtime_pause_scale = float(max(0.6, min(1.6, float(pause_scale))))
            return {
                "rate_offset": int(self._runtime_rate_offset),
                "pause_scale": float(self._runtime_pause_scale),
            }

    def set_enabled(self, enabled):
        with self._lock:
            self._enabled = bool(enabled)
        return True, f"Speech {'enabled' if enabled else 'disabled'}."

    def is_enabled(self):
        with self._lock:
            return self._enabled

    def is_speaking(self):
        with self._lock:
            thread = self._thread
        return bool(thread and thread.is_alive())

    def interrupt(self):
        self._stop_event.set()
        with self._lock:
            process = self._process
            pyttsx3_engine = self._pyttsx3_engine
            thread = self._thread
            self._process = None
            self._pyttsx3_engine = None

        if process is not None:
            try:
                process.terminate()
            except Exception:
                pass
        if pyttsx3_engine is not None:
            try:
                pyttsx3_engine.stop()
            except Exception:
                pass
        try:
            import sounddevice as sd  # type: ignore

            sd.stop()
        except Exception:
            pass
        if thread and thread.is_alive():
            thread.join(timeout=2)
        return True

    def speak_async(self, text, language=None):
        if not (text or "").strip():
            return False, "Nothing to speak."

        if not self.is_enabled():
            return False, "Speech output is disabled."

        self.interrupt()
        self._stop_event.clear()

        thread = threading.Thread(
            target=self._run_speech,
            args=(text, language),
            name="jarvis-speech",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()
        return True, "Speech started."

    def _resolve_backend(self):
        return self.get_backend()

    def _prepare_text_for_speech(self, text, *, preferred_language=None):
        normalized = " ".join(str(text or "").split()).strip()
        if not normalized:
            return normalized

        if not self._is_arabic_preferred_text(normalized, preferred_language=preferred_language):
            return normalized

        dialect = str(TTS_ARABIC_SPOKEN_DIALECT or "egyptian").strip().lower()
        if dialect == "egyptian" and bool(TTS_EGYPTIAN_COLLOQUIAL_REWRITE):
            rewritten = _rewrite_to_egyptian_colloquial(normalized)
            if rewritten and rewritten != normalized:
                logger.info("Applied Egyptian colloquial rewrite for Arabic TTS utterance")
            return rewritten or normalized

        return normalized

    def _probe_pyttsx3_environment(self):
        info = {
            "available": False,
            "voice_id": "",
            "voice_name": "",
            "voice_count": 0,
            "error": "",
        }
        try:
            import pyttsx3  # type: ignore
        except Exception as exc:
            info["error"] = str(exc)
            return info

        info["available"] = True
        engine = None
        try:
            engine = pyttsx3.init()
            voice_id = str(engine.getProperty("voice") or "").strip()
            voices = list(engine.getProperty("voices") or [])
            voice_name = ""
            for voice in voices:
                candidate_id = str(getattr(voice, "id", "") or "")
                if candidate_id == voice_id:
                    voice_name = str(getattr(voice, "name", "") or "")
                    break
            info["voice_id"] = voice_id
            info["voice_name"] = voice_name
            info["voice_count"] = len(voices)
        except Exception as exc:
            info["error"] = str(exc)
        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass
        return info

    def _probe_edge_tts_environment(self):
        info = {
            "available": False,
            "voice": str(TTS_EDGE_VOICE or "").strip(),
            "arabic_voice": str(TTS_EDGE_ARABIC_VOICE or "").strip(),
            "arabic_dialect": str(TTS_ARABIC_SPOKEN_DIALECT or "egyptian").strip(),
            "arabic_pitch": str(TTS_EDGE_ARABIC_PITCH or "").strip(),
            "arabic_volume": str(TTS_EDGE_ARABIC_VOLUME or "").strip(),
            "arabic_fallbacks": [
                str(item).strip()
                for item in (TTS_EDGE_ARABIC_VOICE_FALLBACKS or ())
                if str(item).strip()
            ],
            "supports_output_format": False,
            "compressed_decode_available": False,
            "error": "",
        }
        try:
            import edge_tts  # type: ignore
            import numpy  # type: ignore
            import sounddevice  # type: ignore

            _ = edge_tts, numpy, sounddevice

            supports_output_format = bool(self._edge_tts_supports_output_format(edge_tts))
            compressed_decode_available = bool(self._can_decode_edge_compressed_stream())
            info["supports_output_format"] = supports_output_format
            info["compressed_decode_available"] = compressed_decode_available
            info["available"] = bool(supports_output_format or compressed_decode_available)
            if not info["available"]:
                info["error"] = (
                    "edge_tts stream decode unavailable: install soundfile or upgrade edge_tts "
                    "to a version that supports output_format."
                )
        except Exception as exc:
            info["error"] = str(exc)
        return info

    def run_voice_diagnostic(self):
        phrase = "Jarvis voice diagnostic. If you can hear this, text to speech output is working."
        requested_backend = str(self._resolve_backend() or "auto").strip().lower()
        quality_mode = self.get_quality_mode()
        pyttsx3_info = self._probe_pyttsx3_environment()
        edge_info = self._probe_edge_tts_environment()

        active_backend = requested_backend
        if requested_backend == "auto":
            if pyttsx3_info.get("available"):
                active_backend = "pyttsx3"
            elif edge_info.get("available"):
                active_backend = "edge_tts"
            else:
                active_backend = "console"

        if active_backend == "pyttsx3":
            device_label = pyttsx3_info.get("voice_name") or pyttsx3_info.get("voice_id") or "default_system_voice"
        elif active_backend == "edge_tts":
            device_label = edge_info.get("voice") or "edge_tts_voice"
        else:
            device_label = "console_output"

        spoke_ok, spoke_message = self.speak_async(phrase)

        lines = [
            "Voice Diagnostic",
            f"diagnostic_phrase: {phrase}",
            f"speech_enabled: {self.is_enabled()}",
            f"requested_backend: {requested_backend}",
            f"active_backend: {active_backend}",
            f"voice_quality_mode: {quality_mode}",
            f"output_device: {device_label}",
            f"pyttsx3_available: {pyttsx3_info.get('available')}",
            f"pyttsx3_voice_count: {pyttsx3_info.get('voice_count')}",
            f"edge_tts_available: {edge_info.get('available')}",
            f"edge_tts_voice: {edge_info.get('voice')}",
            f"edge_tts_arabic_voice: {edge_info.get('arabic_voice')}",
            f"edge_tts_arabic_dialect: {edge_info.get('arabic_dialect')}",
            f"edge_tts_arabic_pitch: {edge_info.get('arabic_pitch')}",
            f"edge_tts_arabic_volume: {edge_info.get('arabic_volume')}",
            f"edge_tts_arabic_fallbacks: {', '.join(edge_info.get('arabic_fallbacks') or [])}",
            f"edge_tts_supports_output_format: {edge_info.get('supports_output_format')}",
            f"edge_tts_compressed_decode_available: {edge_info.get('compressed_decode_available')}",
            f"speech_attempt: {spoke_message}",
        ]
        if pyttsx3_info.get("error"):
            lines.append(f"pyttsx3_error: {pyttsx3_info.get('error')}")
        if edge_info.get("error"):
            lines.append(f"edge_tts_error: {edge_info.get('error')}")

        meta = {
            "requested_backend": requested_backend,
            "active_backend": active_backend,
            "voice_quality_mode": quality_mode,
            "output_device": device_label,
            "speech_attempt_ok": bool(spoke_ok),
            "pyttsx3_available": bool(pyttsx3_info.get("available")),
            "edge_tts_available": bool(edge_info.get("available")),
            "edge_tts_voice": edge_info.get("voice"),
            "edge_tts_arabic_voice": edge_info.get("arabic_voice"),
            "edge_tts_arabic_dialect": edge_info.get("arabic_dialect"),
            "edge_tts_arabic_pitch": edge_info.get("arabic_pitch"),
            "edge_tts_arabic_volume": edge_info.get("arabic_volume"),
            "edge_tts_arabic_fallbacks": list(edge_info.get("arabic_fallbacks") or []),
            "edge_tts_supports_output_format": bool(edge_info.get("supports_output_format")),
            "edge_tts_compressed_decode_available": bool(edge_info.get("compressed_decode_available")),
        }
        return True, "\n".join(lines), meta

    def _run_speech(self, text, language=None):
        started = time.perf_counter()
        success = True
        backend = str(self._resolve_backend() or "auto").strip().lower()
        quality_mode = self.get_quality_mode()
        style = persona_manager.get_speech_style()
        logger.info("Speech backend=%s quality=%s style=%s", backend, quality_mode, style)
        spoken_text = self._prepare_text_for_speech(text, preferred_language=language)

        try:
            if backend == "auto":
                if self._speak_pyttsx3(spoken_text, preferred_language=language):
                    return
                if self._speak_edge_tts(spoken_text, preferred_language=language):
                    return
                self._speak_console(spoken_text, prefix="TTS fallback")
                return

            if backend == "pyttsx3":
                if self._speak_pyttsx3(spoken_text, preferred_language=language):
                    return
                self._speak_console(spoken_text, prefix="TTS fallback")
                return

            if backend == "edge_tts":
                if self._speak_edge_tts(spoken_text, preferred_language=language):
                    return
                if self._speak_pyttsx3(spoken_text, preferred_language=language):
                    logger.warning("Edge-TTS failed; pyttsx3 fallback succeeded")
                    return
                self._speak_console(spoken_text, prefix="Edge-TTS fallback")
                return

            self._speak_console(spoken_text, prefix="TTS")
        except Exception:
            success = False
            raise
        finally:
            metrics.record_stage("tts", time.perf_counter() - started, success=success)
            with self._lock:
                self._process = None
                self._pyttsx3_engine = None
                if self._thread and self._thread.ident == threading.current_thread().ident:
                    self._thread = None

    def _speak_console(self, text, prefix):
        words = text.split()
        if not words:
            return
        tuning = self.get_tuning_settings()
        delay = max(0.0, float(TTS_SIMULATED_CHAR_DELAY)) * float(tuning.get("pause_scale") or 1.0)
        print(f"[{prefix}]")
        for word in words:
            if self._stop_event.is_set():
                break
            print(word, end=" ", flush=True)
            time.sleep(max(0.01, len(word) * delay))
        print("")

    def _choose_pyttsx3_voice(
        self,
        engine,
        text,
        *,
        prefer_last_voice=False,
        force_english_voice=False,
        preferred_language=None,
    ):
        voices = list(engine.getProperty("voices") or [])
        if not voices:
            return "", "", False

        if prefer_last_voice and self._last_pyttsx3_voice_id:
            for voice in voices:
                candidate_id = str(getattr(voice, "id", "") or "").strip()
                if candidate_id == self._last_pyttsx3_voice_id:
                    return (
                        candidate_id,
                        str(getattr(voice, "name", "") or "").strip(),
                        True,
                    )

        wants_arabic = self._is_arabic_preferred_text(text, preferred_language=preferred_language)
        if force_english_voice:
            wants_arabic = False
        best_voice = None
        best_score = -999999
        best_language_match = False

        for voice in voices:
            descriptor = _voice_descriptor(voice)
            score = 0
            language_match = False

            if wants_arabic:
                if any(token in descriptor for token in ("arabic", "ar-", "ar_", "ar-sa", "ar-eg", "hoda", "naayf", "salma", "tarik")):
                    score += 10
                    language_match = True
                if any(token in descriptor for token in ("english", "en-", "en_", "en-us", "en-gb", "zira", "david")):
                    score -= 2
            else:
                if any(token in descriptor for token in ("english", "en-", "en_", "en-us", "en-gb", "zira", "aria", "jenny", "hazel", "david", "samantha", "mark")):
                    score += 8
                    language_match = True
                if any(token in descriptor for token in ("arabic", "ar-", "ar_", "ar-sa", "ar-eg")):
                    score -= 2

            if any(token in descriptor for token in ("neural", "natural", "zira", "aria", "jenny", "hazel", "samantha", "salma", "hoda")):
                score += 2
            if any(token in descriptor for token in ("female", "woman", "zira", "aria", "jenny", "hazel", "samantha", "salma", "hoda")):
                score += 1
            if "desktop" in descriptor:
                score += 1

            if score > best_score:
                best_score = score
                best_voice = voice
                best_language_match = language_match

        if best_voice is None:
            return "", "", False

        return (
            str(getattr(best_voice, "id", "") or "").strip(),
            str(getattr(best_voice, "name", "") or "").strip(),
            bool(best_language_match),
        )

    def _speak_pyttsx3(
        self,
        text,
        *,
        require_language_match=False,
        prefer_last_voice=False,
        force_english_voice=False,
        preferred_language=None,
    ):
        try:
            import pyttsx3  # type: ignore
        except Exception as exc:
            logger.warning("pyttsx3 unavailable: %s", exc)
            return False

        try:
            engine = pyttsx3.init()
            selected_voice_id, selected_voice_name, language_match = self._choose_pyttsx3_voice(
                engine,
                text,
                prefer_last_voice=prefer_last_voice,
                force_english_voice=force_english_voice,
                preferred_language=preferred_language,
            )
            if require_language_match and not language_match:
                try:
                    engine.stop()
                except Exception:
                    pass
                logger.info("No language-matched system voice found for this utterance; skipping pyttsx3")
                return False

            if selected_voice_id:
                try:
                    engine.setProperty("voice", selected_voice_id)
                except Exception:
                    pass

            rate = int(persona_manager.get_speech_rate() or TTS_DEFAULT_RATE)
            if self.get_quality_mode() == "natural":
                rate = max(145, min(185, rate - 12))

            if self._is_arabic_preferred_text(text, preferred_language=preferred_language):
                rate -= 8

            tuning = self.get_tuning_settings()
            rate += int(tuning.get("rate_offset") or 0)
            rate = max(125, min(220, int(rate)))

            engine.setProperty("rate", int(rate))
            try:
                engine.setProperty("volume", 1.0)
            except Exception:
                pass

            with self._lock:
                self._pyttsx3_engine = engine
                self._last_pyttsx3_voice_id = selected_voice_id
                self._last_pyttsx3_voice_name = selected_voice_name
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception as exc:
            logger.error("pyttsx3 speech failed: %s", exc)
            return False

    def _run_async(self, coroutine):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coroutine)
            finally:
                loop.close()

        return asyncio.run(coroutine)

    def _edge_tts_supports_output_format(self, edge_tts_module):
        try:
            signature = inspect.signature(edge_tts_module.Communicate.__init__)
        except Exception:
            return False
        if "output_format" in signature.parameters:
            return True
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _edge_tts_supports_parameter(self, edge_tts_module, parameter_name):
        try:
            signature = inspect.signature(edge_tts_module.Communicate.__init__)
        except Exception:
            return False
        key = str(parameter_name or "").strip()
        if key in signature.parameters:
            return True
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _can_decode_edge_compressed_stream(self):
        try:
            import soundfile as _sf  # type: ignore

            _ = _sf
            return True
        except Exception:
            return False

    def _decode_edge_audio_bytes(self, audio_bytes):
        payload = bytes(audio_bytes or b"")
        if len(payload) < 8:
            return None

        header = payload[:4]
        if header in {b"RIFF", b"RIFX", b"RF64"}:
            try:
                with wave.open(io.BytesIO(payload), "rb") as handle:
                    sample_rate = int(handle.getframerate() or 0)
                    sample_width = int(handle.getsampwidth() or 0)
                    channels = int(handle.getnchannels() or 1)
                    frame_count = int(handle.getnframes() or 0)
                    frames = handle.readframes(frame_count)
            except Exception:
                return None

            if sample_rate <= 0 or not frames:
                return None

            import numpy as np  # type: ignore

            if sample_width == 1:
                waveform = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                waveform = (waveform - 128.0) / 128.0
            elif sample_width == 2:
                waveform = np.frombuffer(frames, dtype=np.int16)
            elif sample_width == 4:
                waveform = np.frombuffer(frames, dtype=np.int32)
            else:
                return None

            if channels > 1:
                expected = int(waveform.size // channels) * channels
                if expected <= 0:
                    return None
                waveform = waveform[:expected].reshape(-1, channels)

            return sample_rate, waveform

        # Older edge-tts versions stream compressed audio (for example MP3).
        # Decode through soundfile when available.
        try:
            import numpy as np  # type: ignore
            import soundfile as sf  # type: ignore
        except Exception:
            return None

        try:
            waveform, sample_rate = sf.read(io.BytesIO(payload), dtype="float32")
        except Exception:
            return None

        if int(sample_rate or 0) <= 0:
            return None
        samples = np.asarray(waveform)
        if samples.size <= 0:
            return None
        return int(sample_rate), samples

    def _log_edge_tts_decode_warning_once(self, message):
        should_log = False
        with self._lock:
            if not self._edge_tts_decode_warning_logged:
                self._edge_tts_decode_warning_logged = True
                should_log = True
        if should_log:
            logger.warning(message)

    def _is_arabic_only_text(self, text):
        return _contains_arabic(text) and not _contains_latin(text)

    def _is_arabic_preferred_text(self, text, preferred_language=None):
        normalized_language = str(preferred_language or "").strip().lower()
        if normalized_language in {"ar", "arabic"}:
            return True
        if normalized_language in {"en", "english"}:
            return False

        arabic_letters = _count_arabic_letters(text)
        if arabic_letters <= 0:
            return False

        latin_letters = _count_latin_letters(text)
        if latin_letters <= 0:
            return True

        # Keep Arabic voice when Arabic script dominates, even with inline English fragments.
        return arabic_letters >= latin_letters

    def _is_egyptian_edge_voice(self, voice_name):
        return str(voice_name or "").strip().lower().startswith("ar-eg")

    def _is_edge_voice_unavailable_error(self, error_text):
        normalized = str(error_text or "").lower()
        if not normalized or "voice" not in normalized:
            return False
        return any(
            token in normalized
            for token in (
                "invalid",
                "not found",
                "unsupported",
                "unavailable",
                "unknown",
            )
        )

    def _remember_edge_voice_unavailable(self, voice_name):
        voice_key = str(voice_name or "").strip().lower()
        if not voice_key:
            return
        with self._lock:
            self._edge_tts_unsupported_voices.add(voice_key)

    def _edge_tts_voice_candidates(self, normalized_text, *, preferred_language=None):
        configured_voice = str(TTS_EDGE_VOICE or "").strip() or "en-US-AriaNeural"
        if not self._is_arabic_preferred_text(normalized_text, preferred_language=preferred_language):
            return [configured_voice]

        with self._lock:
            last_arabic_voice = str(self._last_edge_arabic_voice or "").strip()
            unsupported = set(self._edge_tts_unsupported_voices)

        candidates = []
        primary_arabic = str(TTS_EDGE_ARABIC_VOICE or "").strip() or "ar-EG-SalmaNeural"
        candidates.append(primary_arabic)

        # Keep English voice path unchanged; only prioritize Arabic-configured voice when explicitly Arabic.
        if configured_voice and configured_voice.lower().startswith("ar-"):
            candidates.append(configured_voice)

        egyptian_fallbacks = []
        non_egyptian_fallbacks = []
        for fallback_voice in (TTS_EDGE_ARABIC_VOICE_FALLBACKS or ()):
            candidate = str(fallback_voice or "").strip()
            if not candidate:
                continue
            if self._is_egyptian_edge_voice(candidate):
                egyptian_fallbacks.append(candidate)
            else:
                non_egyptian_fallbacks.append(candidate)

        candidates.extend(egyptian_fallbacks)

        # Reuse last successful Arabic voice, but do not let non-Egyptian lock the session.
        if last_arabic_voice and self._is_egyptian_edge_voice(last_arabic_voice):
            candidates.append(last_arabic_voice)

        candidates.extend(non_egyptian_fallbacks)
        if last_arabic_voice and not self._is_egyptian_edge_voice(last_arabic_voice):
            candidates.append(last_arabic_voice)

        candidates.append("ar-SA-HamedNeural")

        deduped = []
        seen = set()
        for candidate in candidates:
            voice_name = str(candidate or "").strip()
            if not voice_name:
                continue
            key = voice_name.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(voice_name)

        filtered = [voice for voice in deduped if voice.lower() not in unsupported]
        return filtered or deduped[:1]

    def _edge_tts_text_chunks(self, normalized_text, *, preferred_language=None):
        text = " ".join(str(normalized_text or "").split()).strip()
        if not text:
            return []

        has_arabic = _contains_arabic(text)
        has_latin = _contains_latin(text)
        if not (has_arabic and has_latin):
            return [
                {
                    "text": text,
                    "script": "arabic"
                    if self._is_arabic_preferred_text(text, preferred_language=preferred_language)
                    else "latin",
                }
            ]

        chunks = []
        for token in text.split():
            arabic_letters = _count_arabic_letters(token)
            latin_letters = _count_latin_letters(token)
            if arabic_letters and arabic_letters >= latin_letters:
                token_script = "arabic"
            elif latin_letters:
                token_script = "latin"
            else:
                token_script = "neutral"

            if not chunks:
                chunks.append({"script": token_script, "tokens": [token]})
                continue

            if token_script == "neutral":
                chunks[-1]["tokens"].append(token)
                continue

            if chunks[-1]["script"] == "neutral":
                chunks[-1]["script"] = token_script
                chunks[-1]["tokens"].append(token)
                continue

            if chunks[-1]["script"] == token_script:
                chunks[-1]["tokens"].append(token)
            else:
                chunks.append({"script": token_script, "tokens": [token]})

        default_script = (
            "arabic"
            if self._is_arabic_preferred_text(text, preferred_language=preferred_language)
            else "latin"
        )
        finalized = []
        for chunk in chunks:
            script = str(chunk.get("script") or "").strip().lower()
            if script not in {"arabic", "latin"}:
                script = default_script
            chunk_text = " ".join(chunk.get("tokens") or []).strip()
            if not chunk_text:
                continue
            if script == "latin":
                chunk_text = re.sub(r"(?<!\S)\u0627\u0644(?=[A-Za-z])", "", chunk_text)
                chunk_text = re.sub(r"(?<!\S)\u0648(?=[A-Za-z])", "", chunk_text)
                chunk_text = " ".join(chunk_text.split()).strip()
                if not chunk_text:
                    continue
            if finalized and finalized[-1]["script"] == script:
                finalized[-1]["text"] = f"{finalized[-1]['text']} {chunk_text}".strip()
            else:
                finalized.append({"script": script, "text": chunk_text})

        return finalized or [{"script": default_script, "text": text}]

    def _edge_tts_chunk_audio_profile(self, is_arabic_chunk):
        rate = str(TTS_EDGE_RATE or "+0%").strip() or "+0%"
        pitch = ""
        volume = ""
        if is_arabic_chunk:
            rate = str(TTS_EDGE_ARABIC_RATE or rate).strip() or rate
            pitch = str(TTS_EDGE_ARABIC_PITCH or "").strip()
            volume = str(TTS_EDGE_ARABIC_VOLUME or "").strip()
        return rate, pitch, volume

    def _speak_edge_tts_mixed_chunks(
        self,
        normalized_text,
        edge_tts_module,
        supports_output_format,
        supports_pitch,
        supports_volume,
        can_decode_compressed,
        preferred_language=None,
    ):
        if not supports_output_format and not can_decode_compressed:
            return False

        chunks = self._edge_tts_text_chunks(normalized_text, preferred_language=preferred_language)
        if len(chunks) <= 1:
            return False

        if len(str(normalized_text or "")) > int(TTS_EDGE_MIXED_SCRIPT_MAX_TEXT_LENGTH):
            logger.info(
                "Edge-TTS mixed chunk mode skipped: text_length=%s exceeds max=%s",
                len(str(normalized_text or "")),
                int(TTS_EDGE_MIXED_SCRIPT_MAX_TEXT_LENGTH),
            )
            return False

        if len(chunks) > int(TTS_EDGE_MIXED_SCRIPT_MAX_CHUNKS):
            logger.info(
                "Edge-TTS mixed chunk mode skipped: chunk_count=%s exceeds max=%s",
                len(chunks),
                int(TTS_EDGE_MIXED_SCRIPT_MAX_CHUNKS),
            )
            return False

        logger.info("Edge-TTS mixed-script chunk mode enabled (%s chunks)", len(chunks))

        async def _collect_audio_bytes(chunk_text, voice_name, *, chunk_is_arabic):
            chunk_rate, chunk_pitch, chunk_volume = self._edge_tts_chunk_audio_profile(chunk_is_arabic)
            kwargs = {
                "voice": voice_name,
                "rate": chunk_rate,
            }
            if chunk_is_arabic and supports_pitch and chunk_pitch:
                kwargs["pitch"] = chunk_pitch
            if chunk_is_arabic and supports_volume and chunk_volume:
                kwargs["volume"] = chunk_volume

            if supports_output_format:
                kwargs["output_format"] = "riff-24khz-16bit-mono-pcm"
                speaker = edge_tts_module.Communicate(chunk_text, **kwargs)
            else:
                speaker = edge_tts_module.Communicate(chunk_text, **kwargs)

            collected = []
            async for event in speaker.stream():
                if self._stop_event.is_set():
                    break
                if str(event.get("type") or "").lower() != "audio":
                    continue
                data = event.get("data")
                if data:
                    collected.append(bytes(data))
            return b"".join(collected)

        for chunk in chunks:
            chunk_text = str(chunk.get("text") or "").strip()
            if not chunk_text:
                continue
            chunk_is_arabic = str(chunk.get("script") or "").strip().lower() == "arabic"
            chunk_language = "ar" if chunk_is_arabic else "en"
            chunk_candidates = self._edge_tts_voice_candidates(chunk_text, preferred_language=chunk_language)
            first_voice = chunk_candidates[0] if chunk_candidates else ""
            chunk_ok = False
            chunk_last_error = ""

            for index, voice_name in enumerate(chunk_candidates):
                if index > 0:
                    logger.info("Edge-TTS chunk fallback voice attempt: %s -> %s", first_voice, voice_name)
                try:
                    audio_bytes = self._run_async(
                        _collect_audio_bytes(
                            chunk_text,
                            voice_name,
                            chunk_is_arabic=chunk_is_arabic,
                        )
                    )
                    if self._stop_event.is_set():
                        return False
                    if not audio_bytes:
                        chunk_last_error = f"empty_audio:{voice_name}"
                        continue

                    decoded = self._decode_edge_audio_bytes(audio_bytes)
                    if decoded is None:
                        self._log_edge_tts_decode_warning_once(
                            "Edge-TTS stream decode failed. Install soundfile or upgrade edge_tts for output_format support; using pyttsx3 fallback."
                        )
                        chunk_last_error = f"decode_unavailable:{voice_name}"
                        continue

                    sample_rate, waveform = decoded
                    # Block per chunk to avoid truncated boundaries and missing leading words.
                    played = self._play_waveform(waveform, sample_rate, blocking=True)
                    if not played:
                        chunk_last_error = f"playback_failed:{voice_name}"
                        continue

                    if chunk_is_arabic:
                        with self._lock:
                            self._last_edge_arabic_voice = voice_name
                        logger.info("Edge-TTS Arabic voice selected: %s", voice_name)

                    chunk_ok = True
                    break
                except Exception as exc:
                    chunk_last_error = str(exc)
                    if self._is_edge_voice_unavailable_error(chunk_last_error):
                        self._remember_edge_voice_unavailable(voice_name)
                    logger.warning("Edge-TTS chunk synthesis failed with voice '%s': %s", voice_name, exc)

            if not chunk_ok:
                if chunk_last_error:
                    logger.warning(
                        "Edge-TTS mixed chunk failed after %s voice attempt(s): %s",
                        len(chunk_candidates),
                        chunk_last_error,
                    )
                return False

        return True

    def _normalize_audio_samples(self, waveform):
        import numpy as np  # type: ignore

        samples = np.asarray(waveform)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        if samples.size == 0:
            return None

        if samples.dtype.kind in {"i", "u"}:
            info = np.iinfo(samples.dtype)
            peak_limit = float(max(abs(info.min), info.max)) or 1.0
            normalized = samples.astype(np.float32) / peak_limit
        else:
            normalized = samples.astype(np.float32, copy=False)
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            peak = float(np.max(np.abs(normalized)))
            if peak > 1.0:
                normalized = normalized / peak

        return np.clip(normalized, -1.0, 1.0)

    def _is_effectively_silent(self, samples):
        import numpy as np  # type: ignore

        normalized = np.asarray(samples, dtype=np.float32).reshape(-1)
        if normalized.size == 0:
            return True

        peak = float(np.max(np.abs(normalized)))
        rms = float(np.sqrt(np.mean(np.square(normalized))))
        return peak < 0.003 and rms < 0.0008

    def _play_waveform(self, waveform, sample_rate, *, blocking=False):
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:
            logger.warning("Waveform playback dependency unavailable: %s", exc)
            return False

        samples = self._normalize_audio_samples(waveform)
        if samples is None:
            logger.warning("Waveform playback skipped because synthesized audio was empty")
            return False
        if self._is_effectively_silent(samples):
            logger.warning("Synthesized audio is effectively silent; triggering fallback")
            return False

        playback_rate = max(8000, int(sample_rate or 0))

        try:
            sd.stop()
            if blocking:
                if self._stop_event.is_set():
                    sd.stop()
                    return False
                sd.play(samples, samplerate=playback_rate, blocking=True)
                return True

            sd.play(samples, samplerate=playback_rate, blocking=False)

            expected_seconds = float(samples.shape[0]) / float(max(1, playback_rate))
            playback_deadline = time.perf_counter() + max(1.0, expected_seconds + 2.0)
            while True:
                if self._stop_event.is_set():
                    sd.stop()
                    return False
                if time.perf_counter() >= playback_deadline:
                    logger.warning("TTS playback watchdog reached; forcing stream stop")
                    sd.stop()
                    break
                try:
                    stream = sd.get_stream()
                except Exception:
                    break
                if not stream or not getattr(stream, "active", False):
                    break
                time.sleep(0.05)
            return True
        except Exception as exc:
            logger.error("Waveform playback failed: %s", exc)
            try:
                sd.stop()
            except Exception:
                pass
            return False

    def _speak_edge_tts(self, text, *, preferred_language=None):
        try:
            import edge_tts  # type: ignore
        except Exception as exc:
            should_log = False
            with self._lock:
                if not self._edge_tts_unavailable_logged:
                    self._edge_tts_unavailable_logged = True
                    should_log = True
            if should_log:
                logger.warning("Edge-TTS dependencies unavailable: %s", exc)
            return False

        normalized_text = " ".join(str(text or "").split()).strip()
        if not normalized_text:
            return False

        arabic_preferred = self._is_arabic_preferred_text(normalized_text, preferred_language=preferred_language)
        voice_candidates = self._edge_tts_voice_candidates(normalized_text, preferred_language=preferred_language)
        edge_rate = str(TTS_EDGE_RATE or "+0%").strip() or "+0%"
        edge_pitch = ""
        edge_volume = ""
        if arabic_preferred:
            edge_rate = str(TTS_EDGE_ARABIC_RATE or edge_rate).strip() or edge_rate
            edge_pitch = str(TTS_EDGE_ARABIC_PITCH or "").strip()
            edge_volume = str(TTS_EDGE_ARABIC_VOLUME or "").strip()
        supports_output_format = self._edge_tts_supports_output_format(edge_tts)
        supports_pitch = self._edge_tts_supports_parameter(edge_tts, "pitch")
        supports_volume = self._edge_tts_supports_parameter(edge_tts, "volume")
        can_decode_compressed = self._can_decode_edge_compressed_stream()

        if not supports_output_format and not can_decode_compressed:
            self._log_edge_tts_decode_warning_once(
                "Edge-TTS stream decode unavailable in this environment. Install soundfile or upgrade edge_tts; using pyttsx3 fallback."
            )
            return False

        if bool(TTS_EDGE_MIXED_SCRIPT_CHUNKING):
            mixed_chunk_ok = self._speak_edge_tts_mixed_chunks(
                normalized_text,
                edge_tts,
                supports_output_format,
                supports_pitch,
                supports_volume,
                can_decode_compressed,
                preferred_language=preferred_language,
            )
            if mixed_chunk_ok:
                return True

        async def _collect_audio_bytes(voice_name):
            kwargs = {
                "voice": voice_name,
                "rate": edge_rate,
            }
            if arabic_preferred and supports_pitch and edge_pitch:
                kwargs["pitch"] = edge_pitch
            if arabic_preferred and supports_volume and edge_volume:
                kwargs["volume"] = edge_volume

            if supports_output_format:
                kwargs["output_format"] = "riff-24khz-16bit-mono-pcm"
                speaker = edge_tts.Communicate(normalized_text, **kwargs)
            else:
                speaker = edge_tts.Communicate(normalized_text, **kwargs)

            chunks = []
            async for event in speaker.stream():
                if self._stop_event.is_set():
                    break
                if str(event.get("type") or "").lower() != "audio":
                    continue
                data = event.get("data")
                if data:
                    chunks.append(bytes(data))
            return b"".join(chunks)

        last_error = ""
        first_voice = voice_candidates[0] if voice_candidates else ""

        for index, voice_name in enumerate(voice_candidates):
            if index > 0:
                logger.info("Edge-TTS fallback voice attempt: %s -> %s", first_voice, voice_name)
            try:
                audio_bytes = self._run_async(_collect_audio_bytes(voice_name))
                if self._stop_event.is_set():
                    return False
                if not audio_bytes:
                    last_error = f"empty_audio:{voice_name}"
                    continue

                decoded = self._decode_edge_audio_bytes(audio_bytes)
                if decoded is None:
                    self._log_edge_tts_decode_warning_once(
                        "Edge-TTS stream decode failed. Install soundfile or upgrade edge_tts for output_format support; using pyttsx3 fallback."
                    )
                    last_error = f"decode_unavailable:{voice_name}"
                    continue

                sample_rate, waveform = decoded
                played = self._play_waveform(waveform, sample_rate)
                if not played:
                    last_error = f"playback_failed:{voice_name}"
                    continue

                if arabic_preferred:
                    with self._lock:
                        self._last_edge_arabic_voice = voice_name
                    logger.info("Edge-TTS Arabic voice selected: %s", voice_name)
                return True
            except Exception as exc:
                last_error = str(exc)
                if self._is_edge_voice_unavailable_error(last_error):
                    self._remember_edge_voice_unavailable(voice_name)
                logger.warning("Edge-TTS synthesis failed with voice '%s': %s", voice_name, exc)

        if last_error:
            logger.warning("Edge-TTS synthesis failed after %s voice attempt(s): %s", len(voice_candidates), last_error)
        return False

speech_engine = SpeechEngine()
