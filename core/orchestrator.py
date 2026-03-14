import glob
import os
import tempfile
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.mic import record_utterance
from audio.stt import transcribe_streaming
from audio.tts import speech_engine
from audio.wake_word import listen_for_wake_word
from core.command_router import initialize_command_services, route_command
from core.config import (
    BARGE_IN_INTERRUPT_ON_WAKE,
    MAX_RECORD_DURATION,
    REALTIME_BACKPRESSURE_POLL_SECONDS,
    REALTIME_DROP_WHEN_BUSY,
    REALTIME_MAX_PENDING_UTTERANCES,
    WAKE_WORD_IGNORE_WHILE_SPEAKING,
)
from core.logger import logger
from core.metrics import metrics
from core.shutdown import setup_shutdown


_INTERRUPT_COMMANDS = {
    "stop speaking",
    "interrupt speech",
    "be quiet",
    "stop talking",
}


def _is_interrupt_command(text):
    return (text or "").strip().lower() in _INTERRUPT_COMMANDS


def _speech_safe_response(text):
    marker = "\nSources:"
    content = (text or "")
    idx = content.find(marker)
    if idx >= 0:
        content = content[:idx]
    return content.strip()


def _create_utterance_audio_file():
    fd, path = tempfile.mkstemp(prefix="jarvis_utterance_", suffix=".wav")
    try:
        return path
    finally:
        try:
            import os

            os.close(fd)
        except Exception:
            pass


def _safe_remove(path):
    if not path:
        return
    try:
        import os

        if os.path.exists(path):
            os.remove(path)
    except Exception as exc:
        logger.warning("Failed to remove temp audio file %s: %s", path, exc)


def _prune_futures(futures):
    active = []
    for future in futures:
        if future.done():
            try:
                future.result()
            except Exception as exc:
                logger.error("Utterance worker failed: %s", exc)
        else:
            active.append(future)
    return active


def _on_partial_transcript(partial_text):
    if partial_text:
        logger.debug("STT partial: %s", partial_text[-180:])


def _safe_log_text(text, max_chars=220):
    value = " ".join((text or "").split())
    if len(value) > max_chars:
        value = value[: max_chars - 3] + "..."
    try:
        value.encode("cp1252")
        return value
    except Exception:
        return value.encode("unicode_escape", errors="backslashreplace").decode("ascii")


def _run_text_fallback_loop():
    print("Jarvis is running in text fallback mode (no wake-word/audio stack).")
    print("Type 'exit' to stop.")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            return

        route_started = time.perf_counter()
        try:
            response = route_command(text)
            metrics.record_stage("router_text", time.perf_counter() - route_started, success=True)
        except Exception as exc:
            metrics.record_stage("router_text", time.perf_counter() - route_started, success=False)
            logger.error("Text-mode command routing failed: %s", exc)
            response = "Sorry, I had an internal error."

        print(f"Jarvis: {response}")
        if not _is_interrupt_command(text):
            speech_engine.speak_async(_speech_safe_response(response))


def _process_utterance(audio_file, pipeline_started):
    text = ""
    route_success = False
    try:
        stt_started = time.perf_counter()
        text = transcribe_streaming(audio_file, on_partial=_on_partial_transcript)
        metrics.record_stage("stt", time.perf_counter() - stt_started, success=bool(text))
        if not text:
            logger.warning("No valid speech detected")
            return
        logger.info("Transcript: %s", _safe_log_text(text))

        route_started = time.perf_counter()
        try:
            response = route_command(text)
            route_success = True
            metrics.record_stage("router", time.perf_counter() - route_started, success=True)
        except Exception as exc:
            metrics.record_stage("router", time.perf_counter() - route_started, success=False)
            logger.error("Command routing failed: %s", exc)
            response = "Sorry, I had an internal error."

        print(f"Jarvis: {response}")
        if not _is_interrupt_command(text):
            speech_engine.speak_async(_speech_safe_response(response))
    finally:
        metrics.record_stage("pipeline", time.perf_counter() - pipeline_started, success=bool(text) and route_success)
        _safe_remove(audio_file)


def _cleanup_stale_temp_files():
    """Remove leftover jarvis_utterance_*.wav from the temp directory."""
    temp_dir = tempfile.gettempdir()
    pattern = os.path.join(temp_dir, "jarvis_utterance_*.wav")
    removed = 0
    for path in glob.glob(pattern):
        try:
            os.remove(path)
            removed += 1
        except Exception:
            pass
    if removed:
        logger.info("Cleaned up %d stale temp audio file(s).", removed)


def run():
    setup_shutdown()
    _cleanup_stale_temp_files()
    initialize_command_services()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-pipeline")
    in_flight = []
    output_encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf" not in output_encoding:
        logger.warning(
            "Console encoding is %s; non-English text may be garbled. "
            "Use `chcp 65001` and set `PYTHONUTF8=1` before starting Jarvis.",
            output_encoding or "unknown",
        )
    logger.info("Jarvis started")

    try:
        while True:
            in_flight = _prune_futures(in_flight)
            busy = len(in_flight) >= max(1, int(REALTIME_MAX_PENDING_UTTERANCES))
            if busy and REALTIME_DROP_WHEN_BUSY:
                time.sleep(float(REALTIME_BACKPRESSURE_POLL_SECONDS))
                metrics.record_stage("backpressure_wait", float(REALTIME_BACKPRESSURE_POLL_SECONDS), success=True)
                continue
            if WAKE_WORD_IGNORE_WHILE_SPEAKING and speech_engine.is_speaking():
                time.sleep(0.1)
                continue

            wake_started = time.perf_counter()
            try:
                listen_for_wake_word()
                metrics.record_stage("wake_word", time.perf_counter() - wake_started, success=True)
            except RuntimeError as exc:
                metrics.record_stage("wake_word", time.perf_counter() - wake_started, success=False)
                logger.error("Wake word unavailable: %s", exc)
                print("Wake word/audio stack is unavailable in this environment.")
                _run_text_fallback_loop()
                return

            if BARGE_IN_INTERRUPT_ON_WAKE and speech_engine.is_speaking():
                speech_engine.interrupt()
                logger.info("Speech interrupted due to wake-word barge-in.")

            logger.info("Wake word detected")
            pipeline_started = time.perf_counter()

            in_flight = _prune_futures(in_flight)
            busy = len(in_flight) >= max(1, int(REALTIME_MAX_PENDING_UTTERANCES))
            if busy and REALTIME_DROP_WHEN_BUSY:
                logger.warning("Dropping wake event due to pipeline backpressure.")
                metrics.record_stage("backpressure_drop", 0.0, success=False)
                continue

            audio_file = _create_utterance_audio_file()
            record_started = time.perf_counter()
            capture = record_utterance(filename=audio_file, max_duration=MAX_RECORD_DURATION)
            metrics.record_stage(
                "record_audio",
                time.perf_counter() - record_started,
                success=bool(capture.get("speech_detected")),
            )

            if not capture.get("speech_detected"):
                _safe_remove(audio_file)
                continue

            in_flight.append(executor.submit(_process_utterance, audio_file, pipeline_started))
    finally:
        executor.shutdown(wait=False, cancel_futures=False)


if __name__ == "__main__":
    run()
