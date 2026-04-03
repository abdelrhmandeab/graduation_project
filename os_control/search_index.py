import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

from core.config import (
    SEARCH_INDEX_DB_FILE,
    SEARCH_INDEX_MAX_RESULTS,
    SEARCH_INDEX_REFRESH_SECONDS,
)
from core.logger import logger
from os_control.adapter_result import failure_result, success_result, to_legacy_pair
from os_control.policy import policy_engine


def _utc_iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


class SearchIndexService:
    def __init__(self):
        self._db_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._db_initialized = False
        self._stop_event = threading.Event()
        self._thread = None
        self._tracked_roots = set()
        self._indexed_roots = {}
        self._refresh_seconds = max(5, int(SEARCH_INDEX_REFRESH_SECONDS))

    def _connect(self):
        conn = sqlite3.connect(SEARCH_INDEX_DB_FILE, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self):
        if self._db_initialized:
            return
        with self._db_lock:
            if self._db_initialized:
                return
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS file_index (
                        path TEXT PRIMARY KEY,
                        root TEXT NOT NULL,
                        name TEXT NOT NULL,
                        parent_dir TEXT NOT NULL,
                        is_dir INTEGER NOT NULL,
                        size_bytes INTEGER NOT NULL,
                        modified_at REAL NOT NULL,
                        indexed_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_file_index_name
                    ON file_index (name, root)
                    """
                )
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_file_index_parent
                    ON file_index (parent_dir, root)
                    """
                )
                conn.commit()
            self._db_initialized = True

    def _is_allowed(self, path):
        allowed, _reason = policy_engine.can_access_path(path, write=False)
        return allowed

    def start_result(self):
        with self._state_lock:
            if self._thread and self._thread.is_alive():
                return failure_result(
                    "Search index worker is already running.",
                    error_code="already_running",
                )
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="jarvis-search-index",
                daemon=True,
            )
            self._thread.start()
            logger.info("Search index worker started")
            return success_result("Search index worker started.", debug_info={"running": True})

    def stop(self):
        with self._state_lock:
            thread = self._thread
            self._thread = None
        self._stop_event.set()
        if thread and thread.is_alive():
            thread.join(timeout=3)
        logger.info("Search index worker stopped")

    def is_running(self):
        with self._state_lock:
            return bool(self._thread and self._thread.is_alive())

    def track_root_result(self, root_path):
        if not root_path:
            return failure_result("No root path was provided.", error_code="invalid_input")
        root = os.path.abspath(os.path.expanduser(str(root_path).strip().strip('"').strip("'")))
        if not os.path.isdir(root):
            return failure_result(f"Directory does not exist: {root}", error_code="not_found")
        if not self._is_allowed(root):
            return failure_result(f"Path blocked by policy: {root}", error_code="policy_blocked")
        with self._state_lock:
            self._tracked_roots.add(root)
        return success_result(root, debug_info={"root": root})

    def tracked_roots(self):
        with self._state_lock:
            return sorted(self._tracked_roots)

    def _index_root(self, root):
        self._ensure_db()
        started = time.time()
        indexed_at = started

        with self._db_lock, self._connect() as conn:
            conn.execute("DELETE FROM file_index WHERE root = ?", (root,))
            rows = []
            for current_root, dirs, files in os.walk(root):
                if self._stop_event.is_set():
                    break
                current_root = os.path.abspath(current_root)
                if not self._is_allowed(current_root):
                    dirs[:] = []
                    continue

                for dirname in dirs:
                    path = os.path.join(current_root, dirname)
                    if not self._is_allowed(path):
                        continue
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    rows.append(
                        (
                            path,
                            root,
                            dirname,
                            current_root,
                            1,
                            int(st.st_size),
                            float(st.st_mtime),
                            indexed_at,
                        )
                    )

                for filename in files:
                    path = os.path.join(current_root, filename)
                    if not self._is_allowed(path):
                        continue
                    try:
                        st = os.stat(path)
                    except OSError:
                        continue
                    rows.append(
                        (
                            path,
                            root,
                            filename,
                            current_root,
                            0,
                            int(st.st_size),
                            float(st.st_mtime),
                            indexed_at,
                        )
                    )

                if len(rows) >= 500:
                    conn.executemany(
                        """
                        INSERT OR REPLACE INTO file_index
                        (path, root, name, parent_dir, is_dir, size_bytes, modified_at, indexed_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        rows,
                    )
                    rows = []

            if rows:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO file_index
                    (path, root, name, parent_dir, is_dir, size_bytes, modified_at, indexed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            conn.commit()

        with self._state_lock:
            self._indexed_roots[root] = indexed_at
        elapsed = time.time() - started
        logger.info("Indexed root: %s (%.2fs)", root, elapsed)

    def refresh_now_result(self, root=None):
        if root:
            tracked = self.track_root_result(root)
            if not tracked.get("success"):
                return tracked
            targets = [tracked.get("user_message")]
        else:
            targets = self.tracked_roots()
            if not targets:
                return failure_result(
                    "No tracked roots. Add one with an indexed search command first.",
                    error_code="invalid_state",
                )

        for item in targets:
            self._index_root(item)
        return success_result(
            f"Refreshed index for {len(targets)} root(s).",
            debug_info={"targets": targets},
        )

    def search(self, query, root=None, limit=SEARCH_INDEX_MAX_RESULTS):
        needle = (query or "").strip()
        if not needle:
            return []

        target_root = None
        if root:
            tracked = self.track_root_result(root)
            if not tracked.get("success"):
                return []
            target_root = tracked.get("user_message")
            with self._state_lock:
                is_indexed = target_root in self._indexed_roots
            if not is_indexed:
                self._index_root(target_root)

        self._ensure_db()
        pattern = f"%{needle.lower()}%"
        with self._db_lock, self._connect() as conn:
            if target_root:
                rows = conn.execute(
                    """
                    SELECT path
                    FROM file_index
                    WHERE root = ? AND lower(name) LIKE ?
                    ORDER BY is_dir DESC, name ASC
                    LIMIT ?
                    """,
                    (target_root, pattern, max(1, int(limit))),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT path
                    FROM file_index
                    WHERE lower(name) LIKE ?
                    ORDER BY is_dir DESC, name ASC
                    LIMIT ?
                    """,
                    (pattern, max(1, int(limit))),
                ).fetchall()
        return [row["path"] for row in rows]

    def status(self):
        with self._state_lock:
            indexed = {root: _utc_iso(ts) for root, ts in self._indexed_roots.items()}
            tracked = sorted(self._tracked_roots)
            running = bool(self._thread and self._thread.is_alive())
        return {
            "running": running,
            "refresh_seconds": self._refresh_seconds,
            "tracked_roots": tracked,
            "indexed_roots": indexed,
        }

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                for root in self.tracked_roots():
                    if self._stop_event.is_set():
                        break
                    self._index_root(root)
            except Exception as exc:
                logger.error("Search index worker failed: %s", exc)
            self._stop_event.wait(self._refresh_seconds)

    # Legacy tuple compatibility
    def start(self):
        return to_legacy_pair(self.start_result())

    def track_root(self, root_path):
        return to_legacy_pair(self.track_root_result(root_path))

    def refresh_now(self, root=None):
        return to_legacy_pair(self.refresh_now_result(root=root))


search_index_service = SearchIndexService()
