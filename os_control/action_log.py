import json
import threading
from datetime import datetime, timezone

from core.config import ACTION_LOG_FILE
from core.logger import logger
from os_control.persistence import (
    insert_action_log,
    read_action_logs,
    reseal_action_log_chain,
    verify_action_log_chain,
)

_log_lock = threading.Lock()


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def log_action(action_type, status, details=None, rollback_data=None, error=None):
    entry = {
        "timestamp": _utc_now_iso(),
        "action_type": action_type,
        "status": status,
        "details": details or {},
    }
    if rollback_data is not None:
        entry["rollback_data"] = rollback_data
    if error:
        entry["error"] = str(error)

    persisted = insert_action_log(entry)

    # Keep an append-only text log for quick manual inspection.
    with _log_lock:
        with open(ACTION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(persisted, ensure_ascii=True) + "\n")

    logger.info(
        "Action logged: %s (%s) id=%s",
        action_type,
        status,
        persisted.get("id"),
    )
    return persisted


def read_recent_actions(limit=50):
    try:
        return read_action_logs(limit=limit)
    except Exception as exc:
        logger.error("Failed reading action logs: %s", exc)
        return []


def verify_audit_chain():
    try:
        return verify_action_log_chain()
    except Exception as exc:
        logger.error("Failed verifying action log chain: %s", exc)
        return {
            "ok": False,
            "checked": 0,
            "reason": f"verify_error: {exc}",
        }


def reseal_audit_chain():
    try:
        return reseal_action_log_chain()
    except Exception as exc:
        logger.error("Failed resealing action log chain: %s", exc)
        return {
            "resealed": 0,
            "error": str(exc),
        }
