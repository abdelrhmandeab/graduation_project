import subprocess
import threading
import time

from core.config import (
    TTS_DEFAULT_BACKEND,
    TTS_DEFAULT_RATE,
    TTS_ENABLED,
    TTS_EXTERNAL_TIMEOUT_SECONDS,
    TTS_HF_MODEL,
    TTS_HF_SAMPLE_RATE,
    TTS_QUALITY_MODE,
    TTS_SIMULATED_CHAR_DELAY,
    VOICECRAFT_CLI_PATH,
    XTTS_CLI_PATH,
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
        self._runtime_backend = str(TTS_DEFAULT_BACKEND or "auto").strip().lower()
        self._quality_mode = self._normalize_quality_mode(TTS_QUALITY_MODE)
        self._runtime_rate_offset = 0
        self._runtime_pause_scale = 1.0
        self._last_pyttsx3_voice_id = ""
        self._last_pyttsx3_voice_name = ""
        self._enabled = bool(TTS_ENABLED)

    def _normalize_backend(self, backend):
        raw = str(backend or "auto").strip().lower()
        aliases = {
            "hf": "huggingface",
            "transformers": "huggingface",
        }
        resolved = aliases.get(raw, raw)
        allowed = {"auto", "console", "pyttsx3", "huggingface", "xtts", "voicecraft"}
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
            "error": "",
        }
        try:
            import sounddevice  # type: ignore
            import torch  # type: ignore
            from transformers import AutoTokenizer, VitsModel  # type: ignore

            _ = sounddevice, torch, AutoTokenizer, VitsModel
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

        active_backend = requested_backend
        if requested_backend == "auto":
            active_backend = "pyttsx3" if pyttsx3_info.get("available") else "console"

        if active_backend == "pyttsx3":
            device_label = pyttsx3_info.get("voice_name") or pyttsx3_info.get("voice_id") or "default_system_voice"
        elif active_backend in {"hf", "huggingface"}:
            device_label = hf_info.get("model") or "huggingface_tts_model"
        elif active_backend in {"xtts", "voicecraft"}:
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
            f"speech_attempt: {spoke_message}",
        ]
        if pyttsx3_info.get("error"):
            lines.append(f"pyttsx3_error: {pyttsx3_info.get('error')}")
        if hf_info.get("error"):
            lines.append(f"hf_tts_error: {hf_info.get('error')}")

        meta = {
            "requested_backend": requested_backend,
            "active_backend": active_backend,
            "voice_quality_mode": quality_mode,
            "output_device": device_label,
            "speech_attempt_ok": bool(spoke_ok),
            "pyttsx3_available": bool(pyttsx3_info.get("available")),
            "hf_tts_available": bool(hf_info.get("available")),
            "hf_tts_model": hf_info.get("model"),
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

        try:
            if backend in {"auto", "pyttsx3"}:
                if self._speak_pyttsx3(text):
                    return
                self._speak_console(text, prefix="TTS fallback")
                return

            if backend in {"hf", "huggingface"}:
                require_language_match = _contains_arabic(text) and not _contains_latin(text)
                if quality_mode == "natural" and self._speak_pyttsx3(
                    text,
                    require_language_match=require_language_match,
                ):
                    logger.info("Natural quality mode used system TTS before HF-TTS")
                    return
                if self._speak_huggingface(text):
                    return
                if self._speak_pyttsx3(text):
                    logger.warning("HF-TTS failed; pyttsx3 fallback succeeded")
                    return
                self._speak_console(text, prefix="HF-TTS fallback")
                return

            if backend in {"xtts", "voicecraft"}:
                if self._speak_clone_backend(text, provider=backend, reference_audio=clone["reference_audio"]):
                    return
                self._speak_console(text, prefix=f"{backend} fallback")
                return

            self._speak_console(text, prefix="TTS")
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

    def _choose_pyttsx3_voice(self, engine, text):
        voices = list(engine.getProperty("voices") or [])
        if not voices:
            return "", "", False

        wants_arabic = _contains_arabic(text) and not _contains_latin(text)
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

    def _speak_pyttsx3(self, text, *, require_language_match=False):
        try:
            import pyttsx3  # type: ignore
        except Exception as exc:
            logger.warning("pyttsx3 unavailable: %s", exc)
            return False

        try:
            engine = pyttsx3.init()
            selected_voice_id, selected_voice_name, language_match = self._choose_pyttsx3_voice(engine, text)
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

            if _contains_arabic(text) and not _contains_latin(text):
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

    def _speak_huggingface(self, text):
        try:
            import numpy as np  # type: ignore
            import sounddevice as sd  # type: ignore
            import torch  # type: ignore
            from transformers import AutoTokenizer, VitsModel  # type: ignore
        except Exception as exc:
            logger.warning("HuggingFace TTS dependencies unavailable: %s", exc)
            return False

        runtime = self.get_hf_runtime_settings()
        runtime_model_id = str(runtime.get("model") or TTS_HF_MODEL).strip()
        runtime_sample_rate = max(0, int(runtime.get("sample_rate") or 0))
        normalized_text = " ".join(str(text or "").split()).strip()
        if not normalized_text:
            return False

        selected_model_id = runtime_model_id
        model_lower = runtime_model_id.lower()
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

        def _get_hf_components():
            model_id = selected_model_id
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

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = VitsModel.from_pretrained(model_id).to(device)
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

            sd.stop()
            sd.play(waveform, samplerate=sample_rate, blocking=False)
            expected_seconds = float(waveform.shape[0]) / float(max(1, sample_rate))
            playback_deadline = time.perf_counter() + max(1.0, expected_seconds + 2.0)
            while True:
                if self._stop_event.is_set():
                    sd.stop()
                    return False
                if time.perf_counter() >= playback_deadline:
                    logger.warning("HF-TTS playback watchdog reached; forcing stream stop")
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
            logger.error("HuggingFace TTS failed: %s", exc)
            try:
                sd.stop()
            except Exception:
                pass
            return False

    def _speak_clone_backend(self, text, provider, reference_audio):
        cli_path = XTTS_CLI_PATH if provider == "xtts" else VOICECRAFT_CLI_PATH
        if not cli_path:
            logger.warning("%s CLI path is not configured.", provider)
            return False
        if not reference_audio:
            logger.warning("Voice clone reference audio is not configured.")
            return False

        command = [cli_path, "--text", text, "--speaker_wav", reference_audio]
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
                logger.error("%s process timed out.", provider)
                return False

            if process.returncode != 0:
                logger.error("%s process failed: %s", provider, (stderr or "").strip())
                return False
            return True
        except Exception as exc:
            logger.error("Failed to run %s backend: %s", provider, exc)
            return False


speech_engine = SpeechEngine()
