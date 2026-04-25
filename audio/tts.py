import asyncio
import httpx
import io
import inspect
import re
import threading
import time
import wave

from core.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_BASE_URL,
    TTS_ARABIC_SPOKEN_DIALECT,
    TTS_DEFAULT_BACKEND,
    TTS_EDGE_MIXED_SCRIPT_CHUNKING,
    TTS_EDGE_MIXED_SCRIPT_MAX_CHUNKS,
    TTS_EDGE_MIXED_SCRIPT_MAX_TEXT_LENGTH,
    TTS_EDGE_ARABIC_VOICE,
    TTS_EDGE_ARABIC_VOICE_FALLBACKS,
    TTS_EDGE_RATE,
    TTS_EDGE_VOICE,
    TTS_ELEVENLABS_ARABIC_ENABLED,
    TTS_ELEVENLABS_ARABIC_MODEL_ID,
    TTS_ELEVENLABS_ARABIC_VOICE_ID,
    TTS_ELEVENLABS_TIMEOUT_SECONDS,
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


_EGYPTIAN_TTS_PHRASE_REPLACEMENTS = (
    # ── RULE: longer / more-specific phrases must come BEFORE shorter ones ──
    # so "لن أستطيع" fires before "أستطيع", "لماذا" fires before "ماذا", etc.

    # Negated ability (must precede bare ability words)
    ("لن أستطيع", "مش هقدر"),
    ("لن يستطيع", "مش هيقدر"),
    ("لن تستطيع", "مش هتقدر"),
    ("لن أتمكن", "مش هقدر"),
    ("لا أستطيع", "مش قادر"),
    ("لا يستطيع", "مش قادر"),
    ("لا تستطيع", "مش قادرة"),
    ("لا أقدر", "مش قادر"),
    ("لا أعرف", "مش عارف"),
    ("لا أعلم", "مش عارف"),
    # Negated existence (before bare negation)
    ("لا يوجد", "مفيش"),
    ("لا توجد", "مفيش"),
    # Negated past tense
    ("لم يكن", "ماكانش"),
    ("لم تكن", "ماكانتش"),
    ("لم أكن", "ماكنتش"),
    # Question words: compound before simple
    ("كيف ذلك", "إزاي ده"),
    ("من هو", "مين ده"),
    ("من هي", "مين دي"),
    ("لماذا", "ليه"),          # must precede ماذا
    ("لمَ", "ليه"),
    ("ماذا", "إيه"),
    ("متى", "إمتى"),
    ("أين", "فين"),
    # Multi-word expressions (before their shorter components)
    ("بكل تأكيد", "اكيد"),
    ("بالتأكيد", "اكيد"),
    ("علاوة على ذلك", "وكمان"),
    ("بالإضافة إلى ذلك", "وكمان"),
    ("في الوقت الحالي", "دلوقتي"),
    ("في الوقت الراهن", "دلوقتي"),
    ("على سبيل المثال", "مثلاً"),
    ("من الضروري", "لازم"),
    ("يجب عليك", "لازم"),
    ("يجب أن", "لازم"),
    ("ينبغي أن", "المفروض"),
    ("من المفترض", "المفروض"),
    ("بعد ذلك", "بعدين"),
    ("قبل ذلك", "قبل كده"),
    ("في البداية", "في الأول"),
    ("مع ذلك", "مع كده"),
    ("رغم ذلك", "مع كده"),
    ("يحتاج إلى", "محتاج"),
    ("يحتاج الى", "محتاج"),
    ("من فضلك،", "لو سمحت،"),
    ("تمام الأمر", "تمام"),
    ("تم بنجاح", "اتعمل تمام"),
    ("تم الأمر", "خلاص"),
    # Demonstrative compounds (before bare demonstratives)
    ("هذا الأمر", "الموضوع ده"),
    ("هذه الطريقة", "الطريقة دي"),
    ("هذا الشيء", "الحاجة دي"),
    # Relative pronouns
    ("الذي", "اللي"),
    ("التي", "اللي"),
    ("الذين", "اللي"),
    ("اللذان", "اللي"),
    # Connectors (compound before simple)
    ("لأنه", "عشانه"),
    ("لأنها", "عشانها"),
    ("ولكن", "بس"),
    ("لكن", "بس"),
    ("لأن", "عشان"),
    ("كذلك", "كمان"),
    # Verbs (negated before bare)
    ("يُفضَّل", "الأحسن"),
    ("يُفضل", "الأحسن"),
    ("بالطبع", "طبعاً"),
    ("حالياً", "دلوقتي"),
    ("حاليا", "دلوقتي"),
    ("للأسف", "يا ريت"),
    ("أولاً", "الأول"),
    ("ثانياً", "تانياً"),
    ("أخيراً", "في الآخر"),
    ("من فضلك", "لو سمحت"),
)

_EGYPTIAN_TTS_WORD_REPLACEMENTS = (
    # Single-word replacements — applied with word-boundary regex AFTER phrases.
    # Negated forms are handled in the phrase table above; only bare forms here.
    # Ability
    ("يمكنني", "اقدر"),
    ("أستطيع", "اقدر"),
    ("يمكنك", "تقدر"),
    ("يمكنكم", "تقدروا"),
    ("بإمكانك", "تقدر"),
    # Time
    ("الآن", "دلوقتي"),
    # Demonstratives
    ("هذا", "ده"),
    ("هذه", "دي"),
    ("ذلك", "ده"),
    ("تلك", "دي"),
    ("هؤلاء", "دول"),
    ("أولئك", "دول"),
    # Verbs
    ("أذهب", "أروح"),
    ("يذهب", "يروح"),
    ("نذهب", "نروح"),
    ("أريد", "عايز"),
    ("تريد", "عايز"),
    ("يريد", "عايز"),
    ("نريد", "عايزين"),
    ("أعرف", "عارف"),
    ("أعلم", "عارف"),
    ("يقول", "بيقول"),
    ("يعمل", "بيعمل"),
    ("أعمل", "بعمل"),
    ("نعمل", "بنعمل"),
    ("أفكر", "بفكر"),
    ("يفكر", "بيفكر"),
    ("أنظر", "أشوف"),
    ("تنظر", "تشوف"),
    ("ينظر", "يشوف"),
    ("يحتاج", "محتاج"),
    ("يفضل", "الأحسن"),
    # Like / similar
    ("مثل", "زي"),
    # Also / too
    ("أيضاً", "كمان"),
    ("أيضا", "كمان"),
    # Negation (bare)
    ("ليس", "مش"),
    ("لست", "مش"),
    # Common words
    ("جداً", "أوي"),
    ("جدا", "أوي"),
    ("كثيراً", "أوي"),
    ("كثيرا", "أوي"),
    ("سريعاً", "بسرعة"),
    ("سريعا", "بسرعة"),
    ("صحيح", "صح"),
    ("غلط", "غلط"),
    ("كلام", "كلام"),
    ("حسناً", "تمام"),
    ("حسنا", "تمام"),
    ("موافق", "تمام"),
    ("شكراً", "شكراً"),
    ("عفواً", "أهلاً"),
    ("بالطبع", "طبعاً"),
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
        self._queue_thread = None
        self._process = None
        self._runtime_backend = str(TTS_DEFAULT_BACKEND or "auto").strip().lower()
        self._quality_mode = self._normalize_quality_mode(TTS_QUALITY_MODE)
        self._runtime_rate_offset = 0
        self._runtime_pause_scale = 1.0
        self._edge_tts_unavailable_logged = False
        self._edge_tts_decode_warning_logged = False
        self._edge_tts_unsupported_voices = set()
        self._elevenlabs_unavailable_logged = False
        self._enabled = bool(TTS_ENABLED)

    def _normalize_backend(self, backend):
        raw = str(backend or "auto").strip().lower()
        aliases = {
            "edge": "edge_tts",
            "edgetts": "edge_tts",
            "elevenlabs": "hybrid",
            "hybrid_elevenlabs": "hybrid",
        }
        resolved = aliases.get(raw, raw)
        allowed = {"auto", "console", "edge_tts", "hybrid"}
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
            queue_thread = self._queue_thread
        return bool(
            (thread and thread.is_alive())
            or (queue_thread and queue_thread.is_alive())
        )

    def interrupt(self):
        self._stop_event.set()
        with self._lock:
            process = self._process
            thread = self._thread
            queue_thread = self._queue_thread
            self._process = None
            self._thread = None
            self._queue_thread = None

        if process is not None:
            try:
                process.terminate()
            except Exception:
                pass
        try:
            import sounddevice as sd  # type: ignore

            sd.stop()
        except Exception:
            pass
        current_ident = threading.current_thread().ident
        if thread and thread.is_alive() and thread.ident != current_ident:
            thread.join(timeout=2)
        if queue_thread and queue_thread.is_alive() and queue_thread.ident != current_ident:
            queue_thread.join(timeout=2)
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

    def speak_sentence_queue(self, sentences_iterator, language=None):
        if sentences_iterator is None:
            return False, "No sentences provided."

        if not self.is_enabled():
            return False, "Speech output is disabled."

        self.interrupt()
        self._stop_event.clear()

        def _run_sentence_queue():
            try:
                for sentence in sentences_iterator:
                    if self._stop_event.is_set():
                        break
                    utterance = " ".join(str(sentence or "").split()).strip()
                    if not utterance:
                        continue
                    try:
                        self._run_speech(utterance, language=language)
                    except Exception as exc:
                        logger.error("Sentence queue speech failed: %s", exc)
                        if self._stop_event.is_set():
                            break
            finally:
                with self._lock:
                    if (
                        self._queue_thread
                        and self._queue_thread.ident == threading.current_thread().ident
                    ):
                        self._queue_thread = None

        thread = threading.Thread(
            target=_run_sentence_queue,
            name="jarvis-speech-queue",
            daemon=True,
        )
        with self._lock:
            self._queue_thread = thread
        thread.start()
        return True, "Speech queue started."

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

    def _probe_edge_tts_environment(self):
        info = {
            "available": False,
            "voice": str(TTS_EDGE_VOICE or "").strip(),
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

    def _probe_elevenlabs_environment(self):
        api_key = str(ELEVENLABS_API_KEY or "").strip()
        voice_id = str(TTS_ELEVENLABS_ARABIC_VOICE_ID or "").strip()
        model_id = str(TTS_ELEVENLABS_ARABIC_MODEL_ID or "").strip()
        enabled = bool(TTS_ELEVENLABS_ARABIC_ENABLED)
        return {
            "enabled": enabled,
            "api_key_configured": bool(api_key),
            "voice_id": voice_id,
            "model_id": model_id,
            "available_for_arabic": bool(enabled and api_key and voice_id),
        }

    def run_voice_diagnostic(self):
        phrase = "Jarvis voice diagnostic. If you can hear this, text to speech output is working."
        requested_backend = str(self._resolve_backend() or "auto").strip().lower()
        quality_mode = self.get_quality_mode()
        edge_info = self._probe_edge_tts_environment()
        elevenlabs_info = self._probe_elevenlabs_environment()

        active_backend = requested_backend
        if requested_backend == "auto":
            active_backend = "hybrid"

        if active_backend == "edge_tts":
            device_label = edge_info.get("voice") or "edge_tts_voice"
        elif active_backend == "hybrid":
            if elevenlabs_info.get("available_for_arabic"):
                device_label = "elevenlabs_arabic_plus_edge"
            else:
                device_label = edge_info.get("voice") or "hybrid_fallback"
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
            f"edge_tts_available: {edge_info.get('available')}",
            f"edge_tts_voice: {edge_info.get('voice')}",
            f"edge_tts_supports_output_format: {edge_info.get('supports_output_format')}",
            f"edge_tts_compressed_decode_available: {edge_info.get('compressed_decode_available')}",
            f"elevenlabs_arabic_enabled: {elevenlabs_info.get('enabled')}",
            f"elevenlabs_api_key_configured: {elevenlabs_info.get('api_key_configured')}",
            f"elevenlabs_arabic_voice_id: {elevenlabs_info.get('voice_id') or 'not_set'}",
            f"elevenlabs_arabic_model_id: {elevenlabs_info.get('model_id') or 'not_set'}",
            f"elevenlabs_available_for_arabic: {elevenlabs_info.get('available_for_arabic')}",
            f"speech_attempt: {spoke_message}",
        ]
        if edge_info.get("error"):
            lines.append(f"edge_tts_error: {edge_info.get('error')}")

        meta = {
            "requested_backend": requested_backend,
            "active_backend": active_backend,
            "voice_quality_mode": quality_mode,
            "output_device": device_label,
            "speech_attempt_ok": bool(spoke_ok),
            "edge_tts_available": bool(edge_info.get("available")),
            "edge_tts_voice": edge_info.get("voice"),
            "edge_tts_supports_output_format": bool(edge_info.get("supports_output_format")),
            "edge_tts_compressed_decode_available": bool(edge_info.get("compressed_decode_available")),
            "elevenlabs_arabic_enabled": bool(elevenlabs_info.get("enabled")),
            "elevenlabs_api_key_configured": bool(elevenlabs_info.get("api_key_configured")),
            "elevenlabs_arabic_voice_id": elevenlabs_info.get("voice_id") or "",
            "elevenlabs_arabic_model_id": elevenlabs_info.get("model_id") or "",
            "elevenlabs_available_for_arabic": bool(elevenlabs_info.get("available_for_arabic")),
        }
        return True, "\n".join(lines), meta

    def _prewarm_edge_tts(self, *, preferred_language=None):
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

        warmup_text = "Ready."
        voice_candidates = self._edge_tts_voice_candidates(warmup_text, preferred_language="en")
        if not voice_candidates:
            return False

        supports_output_format = self._edge_tts_supports_output_format(edge_tts)
        supports_pitch = self._edge_tts_supports_parameter(edge_tts, "pitch")
        supports_volume = self._edge_tts_supports_parameter(edge_tts, "volume")
        can_decode_compressed = self._can_decode_edge_compressed_stream()
        if not supports_output_format and not can_decode_compressed:
            self._log_edge_tts_decode_warning_once(
                "Edge-TTS stream decode unavailable in this environment. Install soundfile or upgrade edge_tts."
            )
            return False

        edge_rate = str(TTS_EDGE_RATE or "+0%").strip() or "+0%"
        edge_pitch = ""
        edge_volume = ""

        async def _collect_audio_bytes(voice_name):
            kwargs = {
                "voice": voice_name,
                "rate": edge_rate,
            }
            if supports_pitch and edge_pitch:
                kwargs["pitch"] = edge_pitch
            if supports_volume and edge_volume:
                kwargs["volume"] = edge_volume

            if supports_output_format:
                kwargs["output_format"] = "riff-24khz-16bit-mono-pcm"
            speaker = edge_tts.Communicate(warmup_text, **kwargs)

            chunks = []
            async for event in speaker.stream():
                if str(event.get("type") or "").lower() != "audio":
                    continue
                data = event.get("data")
                if data:
                    chunks.append(bytes(data))
            return b"".join(chunks)

        last_error = ""
        first_voice = voice_candidates[0]
        for index, voice_name in enumerate(voice_candidates):
            if index > 0:
                logger.info("Edge-TTS prewarm fallback voice attempt: %s -> %s", first_voice, voice_name)
            try:
                audio_bytes = self._run_async(_collect_audio_bytes(voice_name))
                if not audio_bytes:
                    last_error = f"empty_audio:{voice_name}"
                    continue

                decoded = self._decode_edge_audio_bytes(audio_bytes)
                if decoded is None:
                    last_error = f"decode_unavailable:{voice_name}"
                    continue
                return True
            except Exception as exc:
                last_error = str(exc)
                if self._is_edge_voice_unavailable_error(last_error):
                    self._remember_edge_voice_unavailable(voice_name)

        if last_error:
            logger.debug("Edge-TTS prewarm failed after %s voice attempt(s): %s", len(voice_candidates), last_error)
        return False

    def prewarm(self, *, preferred_language=None):
        backend = str(self._resolve_backend() or "auto").strip().lower()

        if backend == "console":
            return False, "console"

        if backend == "edge_tts":
            if self._prewarm_edge_tts(preferred_language=preferred_language):
                return True, "edge_tts"
            return False, "edge_tts"

        if backend == "hybrid":
            if self._prewarm_edge_tts(preferred_language=preferred_language):
                return True, "edge_tts"
            return False, "hybrid"

        edge_ok = self._prewarm_edge_tts(preferred_language=preferred_language)
        if edge_ok:
            return True, "edge_tts"

        return False, "none"

    def _run_speech(self, text, language=None):
        started = time.perf_counter()
        success = True
        backend = str(self._resolve_backend() or "auto").strip().lower()
        quality_mode = self.get_quality_mode()
        style = persona_manager.get_speech_style()
        logger.info("Speech backend=%s quality=%s style=%s", backend, quality_mode, style)
        spoken_text = self._prepare_text_for_speech(text, preferred_language=language)
        arabic_preferred = self._is_arabic_preferred_text(spoken_text, preferred_language=language)

        try:
            if backend in {"auto", "hybrid"}:
                if arabic_preferred:
                    if self._speak_elevenlabs_arabic(spoken_text):
                        return
                    logger.info("ElevenLabs Arabic TTS unavailable; falling back to edge-tts (ar-EG-SalmaNeural)")
                    if self._speak_edge_tts(spoken_text, preferred_language="ar"):
                        return
                    logger.warning("Edge-TTS Arabic synthesis also failed; using console fallback")
                    self._speak_console(spoken_text, prefix="Arabic TTS fallback")
                    return

                if self._speak_edge_tts(spoken_text, preferred_language="en"):
                    return
                logger.warning("Edge-TTS English synthesis failed")
                self._speak_console(spoken_text, prefix="English TTS fallback")
                return

            if backend == "edge_tts":
                if arabic_preferred:
                    if self._speak_edge_tts(spoken_text, preferred_language="ar"):
                        return
                    self._speak_console(spoken_text, prefix="Edge-TTS Arabic fallback")
                    return

                if self._speak_edge_tts(spoken_text, preferred_language="en"):
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
        with self._lock:
            unsupported = set(self._edge_tts_unsupported_voices)

        # Pick Arabic voices when the text/preferred language calls for Arabic.
        # This is the offline fallback when ElevenLabs is unavailable.
        wants_arabic = self._is_arabic_preferred_text(
            normalized_text, preferred_language=preferred_language,
        )
        if wants_arabic:
            primary_arabic = str(TTS_EDGE_ARABIC_VOICE or "").strip() or "ar-EG-SalmaNeural"
            arabic_fallbacks = [
                str(v or "").strip()
                for v in (TTS_EDGE_ARABIC_VOICE_FALLBACKS or ())
                if str(v or "").strip()
            ]
            candidates = [primary_arabic] + arabic_fallbacks
        else:
            candidates = [configured_voice]

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
        _ = is_arabic_chunk
        rate = str(TTS_EDGE_RATE or "+0%").strip() or "+0%"
        pitch = ""
        volume = ""
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
                            "Edge-TTS stream decode failed. Install soundfile or upgrade edge_tts for output_format support."
                        )
                        chunk_last_error = f"decode_unavailable:{voice_name}"
                        continue

                    sample_rate, waveform = decoded
                    # Block per chunk to avoid truncated boundaries and missing leading words.
                    played = self._play_waveform(waveform, sample_rate, blocking=True)
                    if not played:
                        chunk_last_error = f"playback_failed:{voice_name}"
                        continue

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

    def _speak_elevenlabs_arabic(self, text):
        if not bool(TTS_ELEVENLABS_ARABIC_ENABLED):
            return False

        normalized_text = " ".join(str(text or "").split()).strip()
        if not normalized_text:
            return False

        api_key = str(ELEVENLABS_API_KEY or "").strip()
        voice_id = str(TTS_ELEVENLABS_ARABIC_VOICE_ID or "").strip()
        if not api_key or not voice_id:
            should_log = False
            with self._lock:
                if not self._elevenlabs_unavailable_logged:
                    self._elevenlabs_unavailable_logged = True
                    should_log = True
            if should_log:
                logger.warning(
                    "ElevenLabs Arabic TTS is enabled but not fully configured (missing API key or voice id)."
                )
            return False

        payload = {
            "text": normalized_text,
            "model_id": str(TTS_ELEVENLABS_ARABIC_MODEL_ID or "eleven_multilingual_v2"),
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.75,
            },
        }
        base_url = str(ELEVENLABS_BASE_URL or "https://api.elevenlabs.io").rstrip("/")
        endpoint = f"{base_url}/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }

        try:
            response = httpx.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=float(TTS_ELEVENLABS_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            logger.warning("ElevenLabs Arabic TTS request failed (voice_id=%s): %s", voice_id, exc)
            return False

        if response.status_code >= 400:
            error_preview = (response.text or "").strip().replace("\n", " ")
            if len(error_preview) > 220:
                error_preview = error_preview[:217] + "..."
            logger.warning(
                "ElevenLabs Arabic TTS failed status=%s voice_id=%s detail=%s",
                response.status_code,
                voice_id,
                error_preview,
            )
            return False

        audio_bytes = bytes(response.content or b"")
        if not audio_bytes:
            logger.warning("ElevenLabs Arabic TTS returned empty audio bytes (voice_id=%s)", voice_id)
            return False

        decoded = self._decode_edge_audio_bytes(audio_bytes)
        if decoded is None:
            logger.warning("ElevenLabs Arabic TTS audio decode failed; install soundfile for MP3 decoding support")
            return False

        sample_rate, waveform = decoded
        played = self._play_waveform(waveform, sample_rate)
        if not played:
            return False

        return True

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
        if arabic_preferred:
            logger.info("Edge-TTS Arabic synthesis skipped (English-only policy)")
            return False

        voice_candidates = self._edge_tts_voice_candidates(normalized_text, preferred_language=preferred_language)
        edge_rate = str(TTS_EDGE_RATE or "+0%").strip() or "+0%"
        edge_pitch = ""
        edge_volume = ""
        supports_output_format = self._edge_tts_supports_output_format(edge_tts)
        supports_pitch = self._edge_tts_supports_parameter(edge_tts, "pitch")
        supports_volume = self._edge_tts_supports_parameter(edge_tts, "volume")
        can_decode_compressed = self._can_decode_edge_compressed_stream()

        if not supports_output_format and not can_decode_compressed:
            self._log_edge_tts_decode_warning_once(
                "Edge-TTS stream decode unavailable in this environment. Install soundfile or upgrade edge_tts."
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
            if supports_pitch and edge_pitch:
                kwargs["pitch"] = edge_pitch
            if supports_volume and edge_volume:
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
                        "Edge-TTS stream decode failed. Install soundfile or upgrade edge_tts for output_format support."
                    )
                    last_error = f"decode_unavailable:{voice_name}"
                    continue

                sample_rate, waveform = decoded
                played = self._play_waveform(waveform, sample_rate)
                if not played:
                    last_error = f"playback_failed:{voice_name}"
                    continue
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
