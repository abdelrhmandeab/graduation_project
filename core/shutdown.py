import signal
import sys

from audio.tts import speech_engine
from core.logger import logger
from os_control.job_queue import job_queue_service
from os_control.search_index import search_index_service


def setup_shutdown():
    def handle_exit(sig, frame):
        logger.info("Graceful shutdown initiated")
        job_queue_service.stop()
        search_index_service.stop()
        speech_engine.interrupt()
        print("\nJarvis shutting down safely.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
