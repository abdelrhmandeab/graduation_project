"""Phase 2.8 — SQLite-backed persistence for session memory.

Wraps a tiny ``sqlite3`` schema so ``SessionMemory`` can stop dumping the entire
state to JSON on every mutation. The JSON file remains the canonical legacy
format and is used:

  * as a one-shot import source on first launch (so existing users keep their
    history), and
  * as a manual export target for debugging via :func:`export_to_json`.

Schema
------
``turns``
    append-only conversation rows.

``slots``
    key/value store for the assorted state slots ``SessionMemory`` exposes
    (last_app, language_history, clarification_preferences, ...). Complex
    values are JSON-encoded under the same key.

The store is intentionally schemaless beyond those two tables — it should be
trivial to evolve without migrations.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from typing import Any, Dict, Iterable, List, Optional

from core.config import MEMORY_DB_FILE, MEMORY_FILE
from core.logger import logger

_LEGACY_JSON_IMPORT_MARKER = "__legacy_json_imported__"


class SQLiteMemoryStore:
    """Thread-safe SQLite wrapper used by :class:`SessionMemory`."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = str(db_path or MEMORY_DB_FILE)
        self._lock = threading.RLock()
        self._connection: Optional[sqlite3.Connection] = None
        self._ensure_open()
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def _ensure_open(self) -> sqlite3.Connection:
        with self._lock:
            if self._connection is not None:
                return self._connection
            directory = os.path.dirname(self._db_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            connection = sqlite3.connect(
                self._db_path,
                check_same_thread=False,
                isolation_level=None,  # autocommit; we manage transactions manually
            )
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.row_factory = sqlite3.Row
            self._connection = connection
            return connection

    def _ensure_schema(self) -> None:
        connection = self._ensure_open()
        with self._lock:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    user TEXT,
                    assistant TEXT,
                    language TEXT,
                    intent TEXT
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS slots (
                    name TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at REAL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS turns_by_time ON turns(timestamp)"
            )

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                finally:
                    self._connection = None

    # ------------------------------------------------------------------
    # Slot helpers (key/value)
    # ------------------------------------------------------------------
    def get_slot(self, name: str, default: Any = None) -> Any:
        if not name:
            return default
        connection = self._ensure_open()
        with self._lock:
            row = connection.execute(
                "SELECT value FROM slots WHERE name = ? LIMIT 1", (str(name),)
            ).fetchone()
        if row is None:
            return default
        raw_value = row["value"]
        if raw_value is None:
            return default
        try:
            return json.loads(raw_value)
        except (TypeError, ValueError):
            return default

    def set_slot(self, name: str, value: Any, *, updated_at: Optional[float] = None) -> None:
        if not name:
            return
        connection = self._ensure_open()
        encoded = json.dumps(value, ensure_ascii=False)
        with self._lock:
            connection.execute(
                """
                INSERT INTO slots(name, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (str(name), encoded, float(updated_at or 0.0)),
            )

    def delete_slot(self, name: str) -> None:
        if not name:
            return
        connection = self._ensure_open()
        with self._lock:
            connection.execute("DELETE FROM slots WHERE name = ?", (str(name),))

    def all_slots(self) -> Dict[str, Any]:
        connection = self._ensure_open()
        with self._lock:
            rows = connection.execute(
                "SELECT name, value FROM slots"
            ).fetchall()
        result: Dict[str, Any] = {}
        for row in rows:
            try:
                result[row["name"]] = json.loads(row["value"]) if row["value"] is not None else None
            except (TypeError, ValueError):
                result[row["name"]] = None
        return result

    # ------------------------------------------------------------------
    # Turns helpers (append-only conversation history)
    # ------------------------------------------------------------------
    def append_turn(self, turn: Dict[str, Any]) -> None:
        connection = self._ensure_open()
        with self._lock:
            connection.execute(
                """
                INSERT INTO turns(timestamp, user, assistant, language, intent)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    float(turn.get("timestamp") or 0.0),
                    str(turn.get("user") or ""),
                    str(turn.get("assistant") or ""),
                    str(turn.get("language") or ""),
                    str(turn.get("intent") or ""),
                ),
            )

    def trim_turns(self, max_turns: int) -> None:
        keep = max(0, int(max_turns or 0))
        connection = self._ensure_open()
        with self._lock:
            connection.execute(
                """
                DELETE FROM turns
                WHERE id NOT IN (
                    SELECT id FROM turns ORDER BY id DESC LIMIT ?
                )
                """,
                (keep,),
            )

    def recent_turns(self, limit: int) -> List[Dict[str, Any]]:
        keep = max(1, int(limit or 1))
        connection = self._ensure_open()
        with self._lock:
            rows = connection.execute(
                """
                SELECT timestamp, user, assistant, language, intent
                FROM turns
                ORDER BY id DESC
                LIMIT ?
                """,
                (keep,),
            ).fetchall()
        return [
            {
                "timestamp": float(row["timestamp"] or 0.0),
                "user": row["user"] or "",
                "assistant": row["assistant"] or "",
                "language": row["language"] or "",
                "intent": row["intent"] or "",
            }
            for row in reversed(rows)
        ]

    def replace_turns(self, turns: Iterable[Dict[str, Any]]) -> None:
        connection = self._ensure_open()
        rows = list(turns or [])
        with self._lock:
            connection.execute("BEGIN")
            try:
                connection.execute("DELETE FROM turns")
                for row in rows:
                    connection.execute(
                        """
                        INSERT INTO turns(timestamp, user, assistant, language, intent)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            float(row.get("timestamp") or 0.0),
                            str(row.get("user") or ""),
                            str(row.get("assistant") or ""),
                            str(row.get("language") or ""),
                            str(row.get("intent") or ""),
                        ),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def turn_count(self) -> int:
        connection = self._ensure_open()
        with self._lock:
            row = connection.execute("SELECT COUNT(*) AS n FROM turns").fetchone()
        return int(row["n"] if row is not None else 0)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def reset(self) -> None:
        connection = self._ensure_open()
        with self._lock:
            connection.execute("DELETE FROM turns")
            connection.execute("DELETE FROM slots")

    def import_legacy_json(self, json_path: Optional[str] = None) -> bool:
        """Import a legacy ``jarvis_memory.json`` payload exactly once.

        Returns True if a payload was imported; False otherwise.
        """
        path = str(json_path or MEMORY_FILE)
        if not os.path.exists(path):
            return False

        if self.get_slot(_LEGACY_JSON_IMPORT_MARKER, default=False):
            return False

        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            logger.warning("Could not read legacy memory JSON %s: %s", path, exc)
            return False

        if isinstance(payload, list):
            payload = {"turns": payload}
        if not isinstance(payload, dict):
            return False

        turns = payload.get("turns") if isinstance(payload, dict) else None
        if isinstance(turns, list) and turns:
            try:
                self.replace_turns(turns)
            except Exception as exc:
                logger.warning("Legacy turns import failed: %s", exc)

        slots = payload.get("context_slots") if isinstance(payload, dict) else None
        if isinstance(slots, dict):
            for key, value in slots.items():
                try:
                    self.set_slot(key, value)
                except Exception as exc:
                    logger.debug("Skipped legacy slot %s: %s", key, exc)

        pending = payload.get("pending_clarification") if isinstance(payload, dict) else None
        if isinstance(pending, dict):
            self.set_slot("__pending_clarification__", pending)

        self.set_slot(_LEGACY_JSON_IMPORT_MARKER, True)
        logger.info("Imported legacy session memory from %s", path)
        return True

    def export_to_json(self, json_path: Optional[str] = None) -> str:
        """Dump the current store to a JSON file (debug + manual backup)."""
        path = str(json_path or MEMORY_FILE)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        slots = self.all_slots()
        pending = slots.pop("__pending_clarification__", None)
        legacy_flag = slots.pop(_LEGACY_JSON_IMPORT_MARKER, None)
        _ = legacy_flag  # kept in DB only — not exported

        connection = self._ensure_open()
        with self._lock:
            rows = connection.execute(
                "SELECT timestamp, user, assistant, language, intent FROM turns ORDER BY id"
            ).fetchall()

        payload = {
            "preferred_language": slots.get("preferred_language") or "en",
            "turns": [
                {
                    "timestamp": float(row["timestamp"] or 0.0),
                    "user": row["user"] or "",
                    "assistant": row["assistant"] or "",
                    "language": row["language"] or "",
                    "intent": row["intent"] or "",
                }
                for row in rows
            ],
            "pending_clarification": pending,
            "context_slots": slots,
        }

        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return path


# ---------------------------------------------------------------------------
# Vector memory store (ChromaDB + sentence-transformers)
# ---------------------------------------------------------------------------

class VectorMemoryStore:
    """Semantic long-term memory backed by ChromaDB with local embeddings.

    Stores every LLM_QUERY turn (user question + assistant answer) as an
    embedding so future turns can retrieve the top-N most relevant past
    exchanges to inject into the LLM context.

    Usage:
        store = VectorMemoryStore()
        store.remember("what is machine learning?", "ML teaches computers from data.", language="en")
        results = store.recall("explain deep learning", n=3)
    """

    _CHROMA_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "chroma_memory",
    )
    _COLLECTION_NAME = "jarvis_memory"
    _EMBED_MODEL = "all-MiniLM-L6-v2"

    def __init__(self):
        self._lock = threading.Lock()
        self._client = None
        self._collection = None
        self._embedder = None
        self._ready = False
        self._init_thread = threading.Thread(target=self._init_async, daemon=True)
        self._init_thread.start()

    def _init_async(self) -> None:
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            os.makedirs(self._CHROMA_DIR, exist_ok=True)
            client = chromadb.PersistentClient(path=self._CHROMA_DIR)
            collection = client.get_or_create_collection(
                self._COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            embedder = SentenceTransformer(self._EMBED_MODEL)
            with self._lock:
                self._client = client
                self._collection = collection
                self._embedder = embedder
                self._ready = True
            logger.info("VectorMemoryStore ready (%d entries).", collection.count())
        except Exception as exc:
            logger.warning("VectorMemoryStore init failed (semantic recall disabled): %s", exc)

    def _embed(self, text: str) -> list[float]:
        return self._embedder.encode(text, normalize_embeddings=True).tolist()

    def remember(self, user_text: str, assistant_text: str, *, language: str = "en", intent: str = "LLM_QUERY") -> None:
        """Store a turn embedding. No-op if the store isn't ready."""
        if not self._ready:
            return
        user_text = (user_text or "").strip()
        assistant_text = (assistant_text or "").strip()
        if not user_text:
            return
        combined = f"Q: {user_text}\nA: {assistant_text}" if assistant_text else user_text
        try:
            import time as _time
            doc_id = f"{int(_time.time() * 1000)}_{abs(hash(user_text)) % 100000}"
            embedding = self._embed(combined)
            with self._lock:
                self._collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[combined],
                    metadatas=[{"user": user_text, "assistant": assistant_text, "language": language, "intent": intent}],
                )
        except Exception as exc:
            logger.debug("VectorMemoryStore.remember failed: %s", exc)

    def recall(self, query: str, n: int = 3, *, language: str | None = None) -> list[dict]:
        """Return top-n semantically similar past turns for the given query.

        Each result is {"user": str, "assistant": str, "language": str, "score": float}.
        Returns [] if the store isn't ready or has fewer entries than requested.
        """
        if not self._ready:
            return []
        query = (query or "").strip()
        if not query:
            return []
        try:
            with self._lock:
                count = self._collection.count()
            if count == 0:
                return []
            k = min(n, count)
            embedding = self._embed(query)
            where = {"language": language} if language else None
            with self._lock:
                results = self._collection.query(
                    query_embeddings=[embedding],
                    n_results=k,
                    where=where,
                    include=["metadatas", "distances"],
                )
            memories = []
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            for meta, dist in zip(metadatas, distances):
                score = 1.0 - float(dist)  # cosine distance → similarity
                if score < 0.25:  # skip very dissimilar entries
                    continue
                memories.append({
                    "user": meta.get("user", ""),
                    "assistant": meta.get("assistant", ""),
                    "language": meta.get("language", ""),
                    "score": round(score, 3),
                })
            return memories
        except Exception as exc:
            logger.debug("VectorMemoryStore.recall failed: %s", exc)
            return []

    def count(self) -> int:
        if not self._ready:
            return 0
        try:
            with self._lock:
                return self._collection.count()
        except Exception:
            return 0

    def is_ready(self) -> bool:
        return self._ready


# Module-level singleton — import and use directly.
vector_memory = VectorMemoryStore()
