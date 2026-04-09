import signal
import threading

from audio.tts import speech_engine
from core.logger import logger
from os_control.job_queue import job_queue_service
from os_control.search_index import search_index_service


_shutdown_event = threading.Event()
_cleanup_lock = threading.Lock()
_cleanup_done = False


def is_shutdown_requested():
    return _shutdown_event.is_set()


def reset_shutdown_state_for_tests():
    global _cleanup_done
    _shutdown_event.clear()
    with _cleanup_lock:
        _cleanup_done = False


def perform_shutdown_cleanup():
    global _cleanup_done
    with _cleanup_lock:
        if _cleanup_done:
            return False
        _cleanup_done = True

    try:
        job_queue_service.stop()
    except Exception as exc:
        logger.warning("Job queue shutdown cleanup failed: %s", exc)

    try:
        search_index_service.stop()
    except Exception as exc:
        logger.warning("Search index shutdown cleanup failed: %s", exc)

    try:
        speech_engine.interrupt()
    except Exception as exc:
        logger.warning("Speech engine shutdown cleanup failed: %s", exc)
    return True


def setup_shutdown():
    def handle_exit(sig, frame):
        _ = sig, frame
        if _shutdown_event.is_set():
            return
        logger.info("Graceful shutdown initiated")
        _shutdown_event.set()
        perform_shutdown_cleanup()
        print("\nJarvis shutting down safely.")

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    return _shutdown_event
