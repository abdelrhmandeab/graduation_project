import json
import logging
import time
from core.config import LOG_FILE

logger = logging.getLogger("jarvis")

if not logger.handlers:
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def log_structured(event, level="info", **fields):
    payload = {
        "event": str(event or "unknown"),
        "timestamp": float(fields.pop("timestamp", time.time())),
    }
    for key, value in fields.items():
        payload[str(key)] = value

    message = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    log_func = getattr(logger, str(level or "info").lower(), logger.info)
    log_func(message)
    return payload
