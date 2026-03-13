import subprocess
import threading
import time

from core.config import (
    TTS_DEFAULT_BACKEND,
    TTS_DEFAULT_RATE,
    TTS_ENABLED,
    TTS_EXTERNAL_TIMEOUT_SECONDS,
    TTS_SIMULATED_CHAR_DELAY,
    VOICECRAFT_CLI_PATH,
    XTTS_CLI_PATH,
)
from core.logger import logger
from core.metrics import metrics
from core.persona import persona_manager


class SpeechEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._process = None
        self._pyttsx3_engine = None
        self._enabled = bool(TTS_ENABLED)

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
        return TTS_DEFAULT_BACKEND

    def _run_speech(self, text):
        started = time.perf_counter()
        success = True
        backend = self._resolve_backend()
        clone = persona_manager.get_clone_settings()
        style = persona_manager.get_speech_style()
        logger.info("Speech backend=%s style=%s", backend, style)

        try:
            if backend == "pyttsx3":
                if self._speak_pyttsx3(text):
                    return
                self._speak_console(text, prefix="TTS fallback")
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
        delay = max(0.0, float(TTS_SIMULATED_CHAR_DELAY))
        print(f"[{prefix}]")
        for word in words:
            if self._stop_event.is_set():
                break
            print(word, end=" ", flush=True)
            time.sleep(max(0.01, len(word) * delay))
        print("")

    def _speak_pyttsx3(self, text):
        try:
            import pyttsx3  # type: ignore
        except Exception as exc:
            logger.warning("pyttsx3 unavailable: %s", exc)
            return False

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", int(persona_manager.get_speech_rate() or TTS_DEFAULT_RATE))
            with self._lock:
                self._pyttsx3_engine = engine
            engine.say(text)
            engine.runAndWait()
            return True
        except Exception as exc:
            logger.error("pyttsx3 speech failed: %s", exc)
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
