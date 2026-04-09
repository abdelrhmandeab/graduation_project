import asyncio
import contextlib
import io
import inspect
import logging
import os
import re
import subprocess
import threading
import time
import warnings

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
    TTS_EXTERNAL_TIMEOUT_SECONDS,
    TTS_FORCE_ENGLISH_VOICE_FOR_ARABIC,
    TTS_HF_MODEL,
    TTS_HF_SAMPLE_RATE,
    TTS_HF_VOICE_PRESET,
    TTS_KOKORO_LANG_CODE,
    TTS_KOKORO_SAMPLE_RATE,
    TTS_KOKORO_VOICE,
    TTS_QUALITY_MODE,
    TTS_SIMULATED_CHAR_DELAY,
    VOICECRAFT_CLI_PATH,
)
from core.logger import logger
from core.metrics import metrics
from core.persona import persona_manager


class _SuppressTransformersGenerationWarnings(logging.Filter):
    _NOISE_SNIPPETS = (
        "Passing `generation_config` together with generation-related arguments",
        "Both `max_new_tokens` (",
        "The attention mask and the pad token id were not set.",
        "The attention mask is not set and cannot be inferred",
        "Setting `pad_token_id` to `eos_token_id`",
    )

    def filter(self, record):
        try:
            message = str(record.getMessage() or "")
        except Exception:
            return True
        return not any(snippet in message for snippet in self._NOISE_SNIPPETS)


_TRANSFORMERS_WARNING_PATTERNS = (
    r".*Passing `generation_config` together with generation-related arguments.*",
    r".*Both `max_new_tokens`.*`max_length`.*",
    r".*The attention mask and the pad token id were not set.*",
    r".*The attention mask is not set and cannot be inferred.*",
)


def _run_hf_quietly(operation):
    with warnings.catch_warnings():
        for pattern in _TRANSFORMERS_WARNING_PATTERNS:
            try:
                warnings.filterwarnings("ignore", message=pattern)
            except Exception:
                pass
        with contextlib.redirect_stderr(io.StringIO()):
            return operation()


def _load_hf_component_with_local_cache(loader, model_id, component_name):
    try:
        return _run_hf_quietly(lambda: loader(local_files_only=True))
    except Exception as local_exc:
        local_text = str(local_exc or "").lower()
        if (
            "local_files_only" in local_text
            or "local cache" in local_text
            or "offline mode" in local_text
            or "couldn't connect" in local_text
        ):
            logger.info(
                "HF %s cache miss for %s; attempting online fetch.",
                str(component_name or "component"),
                str(model_id or "unknown"),
            )
        return _run_hf_quietly(lambda: loader(local_files_only=False))


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
        self._hf_tts_tokenizer = None
        self._hf_tts_model = None
        self._hf_tts_model_id = ""
        self._hf_tts_device = "cpu"
        self._hf_tts_runtime_model = str(TTS_HF_MODEL or "").strip()
        self._hf_tts_runtime_sample_rate = max(0, int(TTS_HF_SAMPLE_RATE))
        self._kokoro_pipeline = None
        self._kokoro_pipeline_lang = ""
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
        self._transformers_warning_filter_installed = False
        self._hf_runtime_noise_configured = False
        self._enabled = bool(TTS_ENABLED)

    def _normalize_backend(self, backend):
        raw = str(backend or "auto").strip().lower()
        aliases = {
            "hf": "huggingface",
            "transformers": "huggingface",
            "edge": "edge_tts",
            "edgetts": "edge_tts",
            "kokoro_tts": "kokoro",
        }
        resolved = aliases.get(raw, raw)
        allowed = {"auto", "console", "pyttsx3", "huggingface", "edge_tts", "kokoro", "voicecraft"}
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

    def get_hf_runtime_settings(self):
        with self._lock:
            return {
                "model": str(self._hf_tts_runtime_model or "").strip(),
                "sample_rate": int(self._hf_tts_runtime_sample_rate),
            }

    def set_hf_runtime_settings(self, *, model=None, sample_rate=None):
        with self._lock:
            model_changed = False
            if model is not None:
                candidate = str(model or "").strip()
                if candidate and candidate != self._hf_tts_runtime_model:
                    self._hf_tts_runtime_model = candidate
                    model_changed = True

            if sample_rate is not None:
                self._hf_tts_runtime_sample_rate = max(0, int(sample_rate))

            if model_changed:
                self._hf_tts_tokenizer = None
                self._hf_tts_model = None
                self._hf_tts_model_id = ""
                self._hf_tts_device = "cpu"

            return {
                "model": str(self._hf_tts_runtime_model or "").strip(),
                "sample_rate": int(self._hf_tts_runtime_sample_rate),
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

    def speak_async(self, text):
        if not (text or "").strip():
            return False, "Nothing to speak."

        if not self.is_enabled():
            return False, "Speech output is disabled."

        self.interrupt()
        self._stop_event.clear()

        thread = threading.Thread(
            target=self._run_speech,
            args=(text,),
            name="jarvis-speech",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()
        return True, "Speech started."

    def _resolve_backend(self):
        clone = persona_manager.get_clone_settings()
        if clone["enabled"]:
            return clone["provider"]
        return self.get_backend()

    def _prepare_text_for_speech(self, text):
        normalized = " ".join(str(text or "").split()).strip()
        if not normalized:
            return normalized

        if not self._is_arabic_preferred_text(normalized):
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

    def _probe_hf_tts_environment(self):
        runtime = self.get_hf_runtime_settings()
        info = {
            "available": False,
            "model": str(runtime.get("model") or "").strip(),
            "voice_preset": str(TTS_HF_VOICE_PRESET or "").strip(),
            "error": "",
        }
        try:
            import sounddevice  # type: ignore
            import torch  # type: ignore
            import transformers  # type: ignore

            _ = sounddevice, torch, transformers
            info["available"] = True
        except Exception as exc:
            info["error"] = str(exc)
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
            "error": "",
        }
        try:
            import edge_tts  # type: ignore
            import numpy  # type: ignore
            import sounddevice  # type: ignore
            from scipy.io import wavfile  # type: ignore

            _ = edge_tts, numpy, sounddevice, wavfile
            info["available"] = True
        except Exception as exc:
            info["error"] = str(exc)
        return info

    def _probe_kokoro_environment(self):
        info = {
            "available": False,
            "voice": str(TTS_KOKORO_VOICE or "").strip(),
            "lang_code": str(TTS_KOKORO_LANG_CODE or "a").strip() or "a",
            "error": "",
        }
        try:
            from kokoro import KPipeline  # type: ignore

            _ = KPipeline
            info["available"] = True
        except Exception as exc:
            info["error"] = str(exc)
        return info

    def run_voice_diagnostic(self):
        phrase = "Jarvis voice diagnostic. If you can hear this, text to speech output is working."
        clone = persona_manager.get_clone_settings()
        requested_backend = str(self._resolve_backend() or "auto").strip().lower()
        quality_mode = self.get_quality_mode()
        pyttsx3_info = self._probe_pyttsx3_environment()
        hf_info = self._probe_hf_tts_environment()
        edge_info = self._probe_edge_tts_environment()
        kokoro_info = self._probe_kokoro_environment()

        active_backend = requested_backend
        if requested_backend == "auto":
            if pyttsx3_info.get("available"):
                active_backend = "pyttsx3"
            elif edge_info.get("available"):
                active_backend = "edge_tts"
            elif kokoro_info.get("available"):
                active_backend = "kokoro"
            elif hf_info.get("available"):
                active_backend = "huggingface"
            else:
                active_backend = "console"

        if active_backend == "pyttsx3":
            device_label = pyttsx3_info.get("voice_name") or pyttsx3_info.get("voice_id") or "default_system_voice"
        elif active_backend in {"hf", "huggingface"}:
            device_label = hf_info.get("model") or "huggingface_tts_model"
        elif active_backend == "edge_tts":
            device_label = edge_info.get("voice") or "edge_tts_voice"
        elif active_backend == "kokoro":
            device_label = kokoro_info.get("voice") or "kokoro_voice"
        elif active_backend == "voicecraft":
            device_label = clone.get("reference_audio") or "reference_audio_not_set"
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
            f"hf_tts_available: {hf_info.get('available')}",
            f"hf_tts_model: {hf_info.get('model')}",
            f"edge_tts_available: {edge_info.get('available')}",
            f"edge_tts_voice: {edge_info.get('voice')}",
            f"edge_tts_arabic_voice: {edge_info.get('arabic_voice')}",
            f"edge_tts_arabic_dialect: {edge_info.get('arabic_dialect')}",
            f"edge_tts_arabic_pitch: {edge_info.get('arabic_pitch')}",
            f"edge_tts_arabic_volume: {edge_info.get('arabic_volume')}",
            f"edge_tts_arabic_fallbacks: {', '.join(edge_info.get('arabic_fallbacks') or [])}",
            f"kokoro_available: {kokoro_info.get('available')}",
            f"kokoro_voice: {kokoro_info.get('voice')}",
            f"speech_attempt: {spoke_message}",
        ]
        if pyttsx3_info.get("error"):
            lines.append(f"pyttsx3_error: {pyttsx3_info.get('error')}")
        if hf_info.get("error"):
            lines.append(f"hf_tts_error: {hf_info.get('error')}")
        if edge_info.get("error"):
            lines.append(f"edge_tts_error: {edge_info.get('error')}")
        if kokoro_info.get("error"):
            lines.append(f"kokoro_error: {kokoro_info.get('error')}")

        meta = {
            "requested_backend": requested_backend,
            "active_backend": active_backend,
            "voice_quality_mode": quality_mode,
            "output_device": device_label,
            "speech_attempt_ok": bool(spoke_ok),
            "pyttsx3_available": bool(pyttsx3_info.get("available")),
            "hf_tts_available": bool(hf_info.get("available")),
            "hf_tts_model": hf_info.get("model"),
            "edge_tts_available": bool(edge_info.get("available")),
            "edge_tts_voice": edge_info.get("voice"),
            "edge_tts_arabic_voice": edge_info.get("arabic_voice"),
            "edge_tts_arabic_dialect": edge_info.get("arabic_dialect"),
            "edge_tts_arabic_pitch": edge_info.get("arabic_pitch"),
            "edge_tts_arabic_volume": edge_info.get("arabic_volume"),
            "edge_tts_arabic_fallbacks": list(edge_info.get("arabic_fallbacks") or []),
            "kokoro_available": bool(kokoro_info.get("available")),
            "kokoro_voice": kokoro_info.get("voice"),
        }
        return True, "\n".join(lines), meta

    def _run_speech(self, text):
        started = time.perf_counter()
        success = True
        backend = str(self._resolve_backend() or "auto").strip().lower()
        quality_mode = self.get_quality_mode()
        clone = persona_manager.get_clone_settings()
        style = persona_manager.get_speech_style()
        logger.info("Speech backend=%s quality=%s style=%s", backend, quality_mode, style)
        spoken_text = self._prepare_text_for_speech(text)

        try:
            if backend == "auto":
                if self._speak_pyttsx3(spoken_text):
                    return
                if self._speak_edge_tts(spoken_text):
                    return
                if self._speak_kokoro(spoken_text):
                    return
                if self._speak_huggingface(spoken_text):
                    return
                self._speak_console(spoken_text, prefix="TTS fallback")
                return

            if backend == "pyttsx3":
                if self._speak_pyttsx3(spoken_text):
                    return
                self._speak_console(spoken_text, prefix="TTS fallback")
                return

            if backend in {"hf", "huggingface"}:
                arabic_preferred = self._is_arabic_preferred_text(spoken_text)
                hf_runtime_model = str((self.get_hf_runtime_settings() or {}).get("model") or "").strip().lower()
                prefer_hf_multilingual_voice = "bark" in hf_runtime_model
                force_english_voice = bool(TTS_FORCE_ENGLISH_VOICE_FOR_ARABIC and arabic_preferred)
                require_language_match = bool(arabic_preferred and not force_english_voice)
                edge_tried_for_arabic = False
                allow_system_tts_before_hf = not (arabic_preferred and not force_english_voice)

                # Arabic clarity is typically better on neural Arabic voices than fallback system voices.
                if arabic_preferred and not prefer_hf_multilingual_voice and not force_english_voice:
                    edge_tried_for_arabic = True
                    if self._speak_edge_tts(spoken_text):
                        logger.info("Arabic speech used Edge-TTS before system/HF-TTS")
                        return

                if (
                    quality_mode == "natural"
                    and allow_system_tts_before_hf
                    and not prefer_hf_multilingual_voice
                    and self._speak_pyttsx3(
                    spoken_text,
                    require_language_match=require_language_match,
                    prefer_last_voice=True,
                    force_english_voice=force_english_voice,
                    )
                ):
                    logger.info("Natural quality mode used system TTS before HF-TTS")
                    return

                if (
                    arabic_preferred
                    and not prefer_hf_multilingual_voice
                    and not edge_tried_for_arabic
                    and self._speak_edge_tts(spoken_text)
                ):
                    logger.info("Arabic speech used Edge-TTS before HF-TTS")
                    return

                if self._speak_huggingface(spoken_text):
                    return
                if not edge_tried_for_arabic and self._speak_edge_tts(spoken_text):
                    logger.warning("HF-TTS failed; Edge-TTS fallback succeeded")
                    return
                if not arabic_preferred and self._speak_pyttsx3(
                    spoken_text,
                    require_language_match=require_language_match,
                    prefer_last_voice=True,
                    force_english_voice=force_english_voice,
                ):
                    logger.warning("HF-TTS failed; pyttsx3 fallback succeeded")
                    return
                self._speak_console(spoken_text, prefix="HF-TTS fallback")
                return

            if backend == "edge_tts":
                if self._speak_edge_tts(spoken_text):
                    return
                if self._speak_pyttsx3(spoken_text):
                    logger.warning("Edge-TTS failed; pyttsx3 fallback succeeded")
                    return
                self._speak_console(spoken_text, prefix="Edge-TTS fallback")
                return

            if backend == "kokoro":
                if self._speak_kokoro(spoken_text):
                    return
                if self._speak_pyttsx3(spoken_text):
                    logger.warning("Kokoro TTS failed; pyttsx3 fallback succeeded")
                    return
                self._speak_console(spoken_text, prefix="Kokoro fallback")
                return

            if backend == "voicecraft":
                if self._speak_clone_backend(spoken_text, provider=backend, reference_audio=clone["reference_audio"]):
                    return
                if self._speak_edge_tts(spoken_text):
                    logger.warning("%s failed; Edge-TTS fallback succeeded", backend)
                    return
                if self._speak_huggingface(spoken_text):
                    logger.warning("%s failed; HF-TTS fallback succeeded", backend)
                    return
                if self._speak_pyttsx3(spoken_text):
                    logger.warning("%s failed; pyttsx3 fallback succeeded", backend)
                    return
                self._speak_console(spoken_text, prefix=f"{backend} fallback")
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

    def _choose_pyttsx3_voice(self, engine, text, *, prefer_last_voice=False, force_english_voice=False):
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

        wants_arabic = self._is_arabic_preferred_text(text)
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

    def _speak_pyttsx3(self, text, *, require_language_match=False, prefer_last_voice=False, force_english_voice=False):
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

            if self._is_arabic_preferred_text(text):
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

    def _decode_edge_audio_bytes(self, audio_bytes):
        header = bytes(audio_bytes[:4])
        if header in {b"RIFF", b"RIFX", b"RF64"}:
            from scipy.io import wavfile  # type: ignore

            sample_rate, waveform = wavfile.read(io.BytesIO(audio_bytes))
            return int(sample_rate), waveform

        try:
            import soundfile as sf  # type: ignore
        except Exception:
            return None

        waveform, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        return int(sample_rate), waveform

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

    def _is_arabic_preferred_text(self, text):
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

    def _edge_tts_voice_candidates(self, normalized_text):
        configured_voice = str(TTS_EDGE_VOICE or "").strip() or "en-US-AriaNeural"
        if not self._is_arabic_preferred_text(normalized_text):
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

    def _edge_tts_text_chunks(self, normalized_text):
        text = " ".join(str(normalized_text or "").split()).strip()
        if not text:
            return []

        has_arabic = _contains_arabic(text)
        has_latin = _contains_latin(text)
        if not (has_arabic and has_latin):
            return [
                {
                    "text": text,
                    "script": "arabic" if self._is_arabic_preferred_text(text) else "latin",
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

        default_script = "arabic" if self._is_arabic_preferred_text(text) else "latin"
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

    def _speak_edge_tts_mixed_chunks(self, normalized_text, edge_tts_module, supports_output_format, supports_pitch, supports_volume):
        chunks = self._edge_tts_text_chunks(normalized_text)
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
            chunk_candidates = self._edge_tts_voice_candidates(chunk_text)
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
                            "Edge-TTS audio decode unavailable for compressed stream; install soundfile or use pyttsx3/HF fallback."
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

    def _install_transformers_generation_warning_filter(self):
        with self._lock:
            if self._transformers_warning_filter_installed:
                return
            self._transformers_warning_filter_installed = True

        for pattern in _TRANSFORMERS_WARNING_PATTERNS:
            try:
                warnings.filterwarnings("ignore", message=pattern)
            except Exception:
                pass

        warning_filter = _SuppressTransformersGenerationWarnings()
        for logger_name in (
            "transformers.generation.utils",
            "transformers.generation.configuration_utils",
        ):
            try:
                generation_logger = logging.getLogger(logger_name)
                generation_logger.addFilter(warning_filter)
                generation_logger.setLevel(logging.ERROR)
            except Exception:
                pass

    def _configure_hf_runtime_noise(self):
        with self._lock:
            if self._hf_runtime_noise_configured:
                return
            self._hf_runtime_noise_configured = True

        try:
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        except Exception:
            pass

        try:
            from huggingface_hub.utils import disable_progress_bars  # type: ignore

            disable_progress_bars()
        except Exception:
            pass

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

    def _speak_edge_tts(self, text):
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

        arabic_preferred = self._is_arabic_preferred_text(normalized_text)
        voice_candidates = self._edge_tts_voice_candidates(normalized_text)
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

        if bool(TTS_EDGE_MIXED_SCRIPT_CHUNKING):
            mixed_chunk_ok = self._speak_edge_tts_mixed_chunks(
                normalized_text,
                edge_tts,
                supports_output_format,
                supports_pitch,
                supports_volume,
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
                        "Edge-TTS audio decode unavailable for compressed stream; install soundfile or use pyttsx3/HF fallback."
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

    def _speak_kokoro(self, text):
        try:
            import numpy as np  # type: ignore
            from kokoro import KPipeline  # type: ignore
        except Exception as exc:
            logger.warning("Kokoro dependencies unavailable: %s", exc)
            return False

        normalized_text = " ".join(str(text or "").split()).strip()
        if not normalized_text:
            return False

        configured_voice = str(TTS_KOKORO_VOICE or "").strip() or "af_heart"
        configured_lang = str(TTS_KOKORO_LANG_CODE or "a").strip() or "a"

        lang_candidates = [configured_lang]
        if self._is_arabic_preferred_text(normalized_text):
            lang_candidates.insert(0, "z")
        if "a" not in lang_candidates:
            lang_candidates.append("a")

        pipeline = None
        selected_lang = ""
        for lang_code in lang_candidates:
            try:
                with self._lock:
                    if self._kokoro_pipeline is not None and self._kokoro_pipeline_lang == lang_code:
                        pipeline = self._kokoro_pipeline
                    else:
                        pipeline = None

                if pipeline is None:
                    pipeline = KPipeline(lang_code=lang_code)
                    with self._lock:
                        self._kokoro_pipeline = pipeline
                        self._kokoro_pipeline_lang = lang_code

                selected_lang = lang_code
                break
            except Exception:
                pipeline = None

        if pipeline is None:
            logger.warning("Kokoro pipeline failed for lang candidates: %s", ", ".join(lang_candidates))
            return False

        try:
            chunks = []
            generator = pipeline(normalized_text, voice=configured_voice, speed=1.0)
            for row in generator:
                if self._stop_event.is_set():
                    return False
                audio = row[2] if isinstance(row, tuple) and len(row) >= 3 else row
                if audio is None:
                    continue
                chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
                if chunk.size:
                    chunks.append(chunk)

            if not chunks:
                logger.warning("Kokoro TTS produced no waveform chunks (lang=%s)", selected_lang or configured_lang)
                return False

            waveform = np.concatenate(chunks)
            sample_rate = max(8000, int(TTS_KOKORO_SAMPLE_RATE or 24000))
            return self._play_waveform(waveform, sample_rate)
        except Exception as exc:
            logger.warning("Kokoro TTS failed: %s", exc)
            return False

    def _speak_huggingface(self, text):
        runtime = self.get_hf_runtime_settings()
        runtime_model_id = str(runtime.get("model") or TTS_HF_MODEL).strip()
        runtime_sample_rate = max(0, int(runtime.get("sample_rate") or 0))
        normalized_text = " ".join(str(text or "").split()).strip()
        if not normalized_text:
            return False

        selected_model_id = runtime_model_id
        model_lower = runtime_model_id.lower()

        if "bark" in model_lower:
            return self._speak_huggingface_bark(normalized_text, selected_model_id, runtime_sample_rate)

        has_arabic = _contains_arabic(normalized_text)
        has_latin = _contains_latin(normalized_text)

        # Avoid HF VITS runtime errors from strong model/text language mismatches.
        if "mms-tts-ara" in model_lower and has_latin and not has_arabic:
            selected_model_id = "facebook/mms-tts-eng"
            logger.warning(
                "HF-TTS auto-selected fallback model '%s' for non-Arabic text",
                selected_model_id,
            )
        elif "mms-tts-eng" in model_lower and has_arabic and not has_latin:
            selected_model_id = "facebook/mms-tts-ara"
            logger.warning(
                "HF-TTS auto-selected fallback model '%s' for Arabic text",
                selected_model_id,
            )

        return self._speak_huggingface_vits(normalized_text, selected_model_id, runtime_sample_rate)

    def _speak_huggingface_vits(self, normalized_text, model_id, runtime_sample_rate):
        try:
            import numpy as np  # type: ignore
            import torch  # type: ignore
            from transformers import AutoTokenizer, VitsModel  # type: ignore
        except Exception as exc:
            logger.warning("HuggingFace VITS dependencies unavailable: %s", exc)
            return False

        self._configure_hf_runtime_noise()

        def _get_hf_components():
            if not model_id:
                raise RuntimeError("HuggingFace TTS model id is empty.")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            with self._lock:
                if (
                    self._hf_tts_model is not None
                    and self._hf_tts_tokenizer is not None
                    and self._hf_tts_model_id == model_id
                    and self._hf_tts_device == device
                ):
                    return self._hf_tts_tokenizer, self._hf_tts_model, self._hf_tts_device

            tokenizer = _load_hf_component_with_local_cache(
                lambda local_files_only=False: AutoTokenizer.from_pretrained(
                    model_id,
                    local_files_only=bool(local_files_only),
                ),
                model_id,
                "TTS tokenizer",
            )
            model = _load_hf_component_with_local_cache(
                lambda local_files_only=False: VitsModel.from_pretrained(
                    model_id,
                    local_files_only=bool(local_files_only),
                ),
                model_id,
                "TTS model",
            ).to(device)
            model.eval()

            with self._lock:
                self._hf_tts_tokenizer = tokenizer
                self._hf_tts_model = model
                self._hf_tts_model_id = model_id
                self._hf_tts_device = device
            return tokenizer, model, device

        try:
            tokenizer, model, device = _get_hf_components()

            inputs = tokenizer(normalized_text, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)

            with torch.no_grad():
                output = model(**inputs)
            waveform = output.waveform.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
            if waveform.size == 0:
                logger.warning("HuggingFace TTS produced empty waveform")
                return False

            sample_rate = int(getattr(model.config, "sampling_rate", 16000) or 16000)
            if runtime_sample_rate > 0:
                sample_rate = runtime_sample_rate

            return self._play_waveform(waveform, sample_rate)
        except Exception as exc:
            logger.error("HuggingFace TTS failed: %s", exc)
            return False

    def _speak_huggingface_bark(self, normalized_text, model_id, runtime_sample_rate):
        try:
            import numpy as np  # type: ignore
            import torch  # type: ignore
            from transformers import AutoProcessor, BarkModel  # type: ignore
        except Exception as exc:
            logger.warning("HuggingFace Bark dependencies unavailable: %s", exc)
            return False

        voice_preset = str(TTS_HF_VOICE_PRESET or "").strip() or "v2/en_speaker_6"
        self._configure_hf_runtime_noise()
        self._install_transformers_generation_warning_filter()

        def _get_hf_components():
            if not model_id:
                raise RuntimeError("HuggingFace Bark model id is empty.")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            with self._lock:
                if (
                    self._hf_tts_model is not None
                    and self._hf_tts_tokenizer is not None
                    and self._hf_tts_model_id == model_id
                    and self._hf_tts_device == device
                ):
                    return self._hf_tts_tokenizer, self._hf_tts_model, self._hf_tts_device

            processor = _load_hf_component_with_local_cache(
                lambda local_files_only=False: AutoProcessor.from_pretrained(
                    model_id,
                    local_files_only=bool(local_files_only),
                ),
                model_id,
                "Bark processor",
            )

            def _load_bark_model(local_files_only=False):
                try:
                    return BarkModel.from_pretrained(
                        model_id,
                        use_safetensors=False,
                        local_files_only=bool(local_files_only),
                    )
                except TypeError:
                    return BarkModel.from_pretrained(
                        model_id,
                        local_files_only=bool(local_files_only),
                    )

            model = _load_hf_component_with_local_cache(
                _load_bark_model,
                model_id,
                "Bark model",
            ).to(device)
            model.eval()

            with self._lock:
                self._hf_tts_tokenizer = processor
                self._hf_tts_model = model
                self._hf_tts_model_id = model_id
                self._hf_tts_device = device
            return processor, model, device

        try:
            processor, model, device = _get_hf_components()

            try:
                inputs = processor(normalized_text, voice_preset=voice_preset, return_tensors="pt")
            except TypeError:
                # Keep compatibility with older processor signatures.
                inputs = processor(normalized_text, return_tensors="pt")
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)

            if isinstance(inputs, dict) and "input_ids" in inputs and "attention_mask" not in inputs:
                input_ids = inputs.get("input_ids")
                if input_ids is not None:
                    inputs["attention_mask"] = torch.ones_like(input_ids)

            with torch.no_grad():
                generated = _run_hf_quietly(lambda: model.generate(**inputs))

            waveform = generated.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)
            if waveform.size == 0:
                logger.warning("HuggingFace Bark produced empty waveform")
                return False

            sample_rate = int(
                getattr(getattr(model, "generation_config", object()), "sample_rate", 0)
                or getattr(model.config, "sample_rate", 0)
                or 24000
            )
            if runtime_sample_rate > 0:
                sample_rate = runtime_sample_rate

            return self._play_waveform(waveform, sample_rate)
        except Exception as exc:
            logger.error("HuggingFace Bark TTS failed: %s", exc)
            return False

    def _speak_clone_backend(self, text, provider, reference_audio):
        cli_path = VOICECRAFT_CLI_PATH
        if not cli_path:
            logger.warning("%s CLI path is not configured.", provider)
            return False
        if provider == "voicecraft" and not reference_audio:
            logger.warning("Voice clone reference audio is not configured.")
            return False

        base_command = [cli_path, "--text", text]
        if reference_audio:
            base_command.extend(["--speaker_wav", reference_audio])

        command_candidates = [base_command]

        last_error = ""
        for command in command_candidates:
            try:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                with self._lock:
                    self._process = process

                try:
                    _stdout, stderr = process.communicate(timeout=TTS_EXTERNAL_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    last_error = "process timed out"
                    continue

                if process.returncode == 0:
                    return True

                last_error = (stderr or "").strip() or f"exit_code={process.returncode}"
            except Exception as exc:
                last_error = str(exc)

        logger.error("%s process failed: %s", provider, last_error)
        return False


speech_engine = SpeechEngine()
