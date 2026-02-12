import logging

logging.basicConfig(
    filename="jarvis.log",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("JARVIS")
