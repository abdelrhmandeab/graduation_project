import hashlib
import json
import sqlite3
import threading
import time

from core.config import STATE_DB_FILE

_db_lock = threading.Lock()
_initialized = False


def _connect():
    conn = sqlite3.connect(STATE_DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _canonical_action_payload(timestamp, action_type, status, details, rollback_data, error):
    payload = {
        "timestamp": timestamp,
        "action_type": action_type,
        "status": status,
        "details": details or {},
        "rollback_data": rollback_data,
        "error": error,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=True)


def _ensure_db():
    global _initialized
    if _initialized:
        return

    with _db_lock:
        if _initialized:
            return
        with _connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS action_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    rollback_data_json TEXT,
                    error TEXT,
                    prev_hash TEXT,
                    hash TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS confirmations (
                    token TEXT PRIMARY KEY,
                    action_name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rollback_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    undone_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS job_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_retries INTEGER NOT NULL DEFAULT 0,
                    run_at REAL NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    last_error TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_job_queue_status_run_at
                ON job_queue (status, run_at, id)
                """
            )
            conn.commit()
        _initialized = True


def insert_action_log(entry):
    _ensure_db()
    timestamp = entry["timestamp"]
    action_type = entry["action_type"]
    status = entry["status"]
    details = entry.get("details") or {}
    rollback_data = entry.get("rollback_data")
    error = entry.get("error")
    payload_json = _canonical_action_payload(
        timestamp=timestamp,
        action_type=action_type,
        status=status,
        details=details,
        rollback_data=rollback_data,
        error=error,
    )

    with _db_lock, _connect() as conn:
        prev = conn.execute(
            "SELECT hash FROM action_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        prev_hash = prev["hash"] if prev else ""
        digest = hashlib.sha256((prev_hash + payload_json).encode("utf-8")).hexdigest()

        cur = conn.execute(
            """
            INSERT INTO action_logs (
                timestamp, action_type, status, details_json,
                rollback_data_json, error, prev_hash, hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                action_type,
                status,
                json.dumps(details, ensure_ascii=True),
                json.dumps(rollback_data, ensure_ascii=True)
                if rollback_data is not None
                else None,
                error,
                prev_hash,
                digest,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid

    out = dict(entry)
    out["id"] = row_id
    out["hash"] = digest
    out["prev_hash"] = prev_hash
    return out


def verify_action_log_chain():
    _ensure_db()
    with _db_lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, action_type, status, details_json,
                   rollback_data_json, error, prev_hash, hash
            FROM action_logs
            ORDER BY id ASC
            """
        ).fetchall()

    previous_hash = ""
    checked = 0
    for row in rows:
        details = json.loads(row["details_json"] or "{}")
        rollback_data = (
            json.loads(row["rollback_data_json"])
            if row["rollback_data_json"]
            else None
        )
        payload_json = _canonical_action_payload(
            timestamp=row["timestamp"],
            action_type=row["action_type"],
            status=row["status"],
            details=details,
            rollback_data=rollback_data,
            error=row["error"],
        )
        expected_hash = hashlib.sha256((previous_hash + payload_json).encode("utf-8")).hexdigest()

        if (row["prev_hash"] or "") != previous_hash:
            return {
                "ok": False,
                "checked": checked,
                "failed_id": row["id"],
                "reason": "prev_hash_mismatch",
                "expected_prev_hash": previous_hash,
                "actual_prev_hash": row["prev_hash"] or "",
            }
        if row["hash"] != expected_hash:
            return {
                "ok": False,
                "checked": checked,
                "failed_id": row["id"],
                "reason": "hash_mismatch",
                "expected_hash": expected_hash,
                "actual_hash": row["hash"],
            }

        previous_hash = row["hash"]
        checked += 1

    return {
        "ok": True,
        "checked": checked,
        "last_hash": previous_hash,
    }


def reseal_action_log_chain():
    _ensure_db()
    with _db_lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, action_type, status, details_json,
                   rollback_data_json, error
            FROM action_logs
            ORDER BY id ASC
            """
        ).fetchall()

        previous_hash = ""
        resealed = 0
        for row in rows:
            details = json.loads(row["details_json"] or "{}")
            rollback_data = (
                json.loads(row["rollback_data_json"])
                if row["rollback_data_json"]
                else None
            )
            payload_json = _canonical_action_payload(
                timestamp=row["timestamp"],
                action_type=row["action_type"],
                status=row["status"],
                details=details,
                rollback_data=rollback_data,
                error=row["error"],
            )
            digest = hashlib.sha256((previous_hash + payload_json).encode("utf-8")).hexdigest()
            conn.execute(
                """
                UPDATE action_logs
                SET prev_hash = ?, hash = ?
                WHERE id = ?
                """,
                (previous_hash, digest, row["id"]),
            )
            previous_hash = digest
            resealed += 1
        conn.commit()

    return {
        "resealed": resealed,
        "last_hash": previous_hash,
    }


def read_action_logs(limit=50):
    _ensure_db()
    with _db_lock, _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, timestamp, action_type, status, details_json,
                   rollback_data_json, error, prev_hash, hash
            FROM action_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    out = []
    for row in rows:
        out.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "action_type": row["action_type"],
                "status": row["status"],
                "details": json.loads(row["details_json"] or "{}"),
                "rollback_data": json.loads(row["rollback_data_json"])
                if row["rollback_data_json"]
                else None,
                "error": row["error"],
                "prev_hash": row["prev_hash"],
                "hash": row["hash"],
            }
        )
    return out


def store_confirmation(token, action_name, description, payload, created_at, expires_at):
    _ensure_db()
    payload_json = json.dumps(payload, ensure_ascii=True)
    with _db_lock, _connect() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO confirmations
            (token, action_name, description, payload_json, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (token, action_name, description, payload_json, created_at, expires_at),
        )
        conn.commit()


def get_confirmation(token):
    _ensure_db()
    with _db_lock, _connect() as conn:
        row = conn.execute(
            """
            SELECT token, action_name, description, payload_json, created_at, expires_at
            FROM confirmations
            WHERE token = ?
            """,
            (token,),
        ).fetchone()
    if not row:
        return None

    return {
        "token": row["token"],
        "action_name": row["action_name"],
        "description": row["description"],
        "payload": json.loads(row["payload_json"] or "{}"),
        "created_at": row["created_at"],
        "expires_at": row["expires_at"],
    }


def delete_confirmation(token):
    _ensure_db()
    with _db_lock, _connect() as conn:
        cur = conn.execute("DELETE FROM confirmations WHERE token = ?", (token,))
        conn.commit()
        return cur.rowcount


def pop_confirmation(token):
    pending = get_confirmation(token)
    if pending:
        delete_confirmation(token)
    return pending


def cleanup_expired_confirmations(now_ts=None):
    _ensure_db()
    now_ts = time.time() if now_ts is None else now_ts
    with _db_lock, _connect() as conn:
        cur = conn.execute(
            "DELETE FROM confirmations WHERE expires_at < ?",
            (now_ts,),
        )
        conn.commit()
        return cur.rowcount


def count_pending_confirmations():
    _ensure_db()
    now_ts = time.time()
    with _db_lock, _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as count FROM confirmations WHERE expires_at >= ?",
            (now_ts,),
        ).fetchone()
    return int(row["count"]) if row else 0


def push_rollback_action(action_type, payload):
    _ensure_db()
    with _db_lock, _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO rollback_actions (action_type, payload_json, created_at, undone_at)
            VALUES (?, ?, ?, NULL)
            """,
            (action_type, json.dumps(payload, ensure_ascii=True), time.time()),
        )
        conn.commit()
        return cur.lastrowid


def pop_latest_rollback_action():
    _ensure_db()
    with _db_lock, _connect() as conn:
        row = conn.execute(
            """
            SELECT id, action_type, payload_json
            FROM rollback_actions
            WHERE undone_at IS NULL
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None

        conn.execute(
            "UPDATE rollback_actions SET undone_at = ? WHERE id = ?",
            (time.time(), row["id"]),
        )
        conn.commit()

    return {
        "id": row["id"],
        "action_type": row["action_type"],
        "payload": json.loads(row["payload_json"] or "{}"),
    }


def count_pending_rollback_actions():
    _ensure_db()
    with _db_lock, _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS count FROM rollback_actions WHERE undone_at IS NULL"
        ).fetchone()
    return int(row["count"]) if row else 0


def restore_rollback_action(action_id):
    _ensure_db()
    with _db_lock, _connect() as conn:
        conn.execute(
            "UPDATE rollback_actions SET undone_at = NULL WHERE id = ?",
            (action_id,),
        )
        conn.commit()


def _job_row_to_dict(row):
    if not row:
        return None
    return {
        "id": row["id"],
        "command_text": row["command_text"],
        "status": row["status"],
        "attempts": int(row["attempts"]),
        "max_retries": int(row["max_retries"]),
        "run_at": float(row["run_at"]),
        "created_at": float(row["created_at"]),
        "updated_at": float(row["updated_at"]),
        "last_error": row["last_error"],
    }


def create_job(command_text, run_at, max_retries=0):
    _ensure_db()
    now_ts = time.time()
    run_ts = float(run_at)
    with _db_lock, _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO job_queue (
                command_text, status, attempts, max_retries, run_at, created_at, updated_at, last_error
            ) VALUES (?, 'queued', 0, ?, ?, ?, ?, NULL)
            """,
            (str(command_text), int(max_retries), run_ts, now_ts, now_ts),
        )
        conn.commit()
        job_id = cur.lastrowid

    return get_job(job_id)


def get_job(job_id):
    _ensure_db()
    with _db_lock, _connect() as conn:
        row = conn.execute(
            """
            SELECT id, command_text, status, attempts, max_retries, run_at, created_at, updated_at, last_error
            FROM job_queue
            WHERE id = ?
            """,
            (int(job_id),),
        ).fetchone()
    return _job_row_to_dict(row)


def list_jobs(limit=20, status=None):
    _ensure_db()
    limit_value = max(1, int(limit))
    with _db_lock, _connect() as conn:
        if status:
            rows = conn.execute(
                """
                SELECT id, command_text, status, attempts, max_retries, run_at, created_at, updated_at, last_error
                FROM job_queue
                WHERE status = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (str(status), limit_value),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, command_text, status, attempts, max_retries, run_at, created_at, updated_at, last_error
                FROM job_queue
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit_value,),
            ).fetchall()
    return [_job_row_to_dict(row) for row in rows]


def claim_due_job(now_ts=None):
    _ensure_db()
    now_ts = time.time() if now_ts is None else float(now_ts)
    with _db_lock, _connect() as conn:
        row = conn.execute(
            """
            SELECT id, command_text, status, attempts, max_retries, run_at, created_at, updated_at, last_error
            FROM job_queue
            WHERE status = 'queued' AND run_at <= ?
            ORDER BY run_at ASC, id ASC
            LIMIT 1
            """,
            (now_ts,),
        ).fetchone()
        if not row:
            return None

        attempts = int(row["attempts"]) + 1
        updated_at = time.time()
        conn.execute(
            """
            UPDATE job_queue
            SET status = 'running', attempts = ?, updated_at = ?
            WHERE id = ?
            """,
            (attempts, updated_at, row["id"]),
        )
        conn.commit()

        claimed = conn.execute(
            """
            SELECT id, command_text, status, attempts, max_retries, run_at, created_at, updated_at, last_error
            FROM job_queue
            WHERE id = ?
            """,
            (row["id"],),
        ).fetchone()
    return _job_row_to_dict(claimed)


def mark_job_succeeded(job_id):
    _ensure_db()
    with _db_lock, _connect() as conn:
        conn.execute(
            """
            UPDATE job_queue
            SET status = 'succeeded', updated_at = ?, last_error = NULL
            WHERE id = ?
            """,
            (time.time(), int(job_id)),
        )
        conn.commit()
    return get_job(job_id)


def mark_job_failed(job_id, error_message, requeue=False, next_run_at=None):
    _ensure_db()
    now_ts = time.time()
    with _db_lock, _connect() as conn:
        if requeue:
            run_at = float(next_run_at if next_run_at is not None else now_ts)
            conn.execute(
                """
                UPDATE job_queue
                SET status = 'queued', run_at = ?, updated_at = ?, last_error = ?
                WHERE id = ?
                """,
                (run_at, now_ts, str(error_message), int(job_id)),
            )
        else:
            conn.execute(
                """
                UPDATE job_queue
                SET status = 'failed', updated_at = ?, last_error = ?
                WHERE id = ?
                """,
                (now_ts, str(error_message), int(job_id)),
            )
        conn.commit()
    return get_job(job_id)


def cancel_job(job_id):
    _ensure_db()
    with _db_lock, _connect() as conn:
        row = conn.execute(
            "SELECT status FROM job_queue WHERE id = ?",
            (int(job_id),),
        ).fetchone()
        if not row:
            return None
        if row["status"] in {"succeeded", "canceled"}:
            return get_job(job_id)
        conn.execute(
            """
            UPDATE job_queue
            SET status = 'canceled', updated_at = ?
            WHERE id = ?
            """,
            (time.time(), int(job_id)),
        )
        conn.commit()
    return get_job(job_id)


def retry_job(job_id, delay_seconds=0):
    _ensure_db()
    next_run_at = time.time() + max(0, int(delay_seconds))
    with _db_lock, _connect() as conn:
        row = conn.execute(
            "SELECT id FROM job_queue WHERE id = ?",
            (int(job_id),),
        ).fetchone()
        if not row:
            return None
        conn.execute(
            """
            UPDATE job_queue
            SET status = 'queued', attempts = 0, run_at = ?, updated_at = ?, last_error = NULL
            WHERE id = ?
            """,
            (next_run_at, time.time(), int(job_id)),
        )
        conn.commit()
    return get_job(job_id)
