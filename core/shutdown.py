import signal
import sys
from core.logger import logger

def setup_shutdown():
    def handle_exit(sig, frame):
        logger.info("Graceful shutdown initiated")
        print("\n🛑 Jarvis shutting down safely.")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
