import json
import logging
import time
from logging.handlers import RotatingFileHandler

from core.config import (
    LOG_CONSOLE_LEVEL,
    LOG_FILE,
    LOG_FILE_LEVEL,
    LOG_ROTATE_BACKUPS,
    LOG_ROTATE_MAX_BYTES,
)


def _level(value, default):
    resolved = logging.getLevelName(str(value or "").strip().upper())
    return resolved if isinstance(resolved, int) else default

logger = logging.getLogger("jarvis")

if not logger.handlers:
    console_level = _level(LOG_CONSOLE_LEVEL, logging.INFO)
    file_level = _level(LOG_FILE_LEVEL, logging.DEBUG)
    logger.setLevel(min(console_level, file_level))
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=max(1, int(LOG_ROTATE_MAX_BYTES)),
        backupCount=max(0, int(LOG_ROTATE_BACKUPS)),
        encoding="utf-8",
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def get_logger(component: str) -> logging.Logger:
    """Return a component-tagged child of the shared Jarvis logger."""
    name = str(component or "general").strip().replace(" ", "_") or "general"
    return logger.getChild(name)


def section(title: str) -> None:
    logger.info("──────── %s ────────", str(title or "").strip())


def kv(component: str, **pairs) -> None:
    width = max((len(str(key)) for key in pairs), default=0)
    message = "  ".join(f"{str(key):<{width}}={value}" for key, value in pairs.items())
    get_logger(component).info(message)


def summary_table(title: str, rows: list[tuple]) -> None:
    section(title)
    normalized = [tuple(str(value) for value in row) for row in rows]
    if not normalized:
        return
    column_count = max(len(row) for row in normalized)
    widths = [
        max((len(row[index]) if index < len(row) else 0) for row in normalized)
        for index in range(column_count)
    ]
    table_logger = get_logger(str(title or "summary").strip().lower().replace(" ", "_"))
    for row in normalized:
        table_logger.info(
            "  ".join(
                (row[index] if index < len(row) else "").ljust(widths[index])
                for index in range(column_count)
            ).rstrip()
        )


def log_structured(event, level="debug", **fields):
    payload = {
        "event": str(event or "unknown"),
        "timestamp": float(fields.pop("timestamp", time.time())),
    }
    for key, value in fields.items():
        payload[str(key)] = value

    message = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    # Structured payloads are intentionally file-detail only; the ``level``
    # argument remains accepted for compatibility with existing callers.
    logger.debug(message)
    return payload
