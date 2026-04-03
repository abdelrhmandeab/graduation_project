import json
import threading
import time

from core.config import (
    MEMORY_ENABLED,
    MEMORY_FILE,
    MEMORY_MAX_CONTEXT_CHARS,
    MEMORY_MAX_TURNS,
)
from core.logger import logger

_SOURCES_MARKER = "\nSources:"
_LOW_VALUE_ASSISTANT_PATTERNS = (
    "i could not run the local model",
    "sorry, i had an internal error",
    "i'm sorry, but i can't assist with that",
    "sorry, but i can't assist with that",
)
_SUPPORTED_LANGUAGES = {"ar", "en"}
_DEFAULT_CLARIFICATION_TTL_SECONDS = 90


def _sanitize_assistant_text(text):
    cleaned = (text or "").strip()
    marker_idx = cleaned.find(_SOURCES_MARKER)
    if marker_idx >= 0:
        cleaned = cleaned[:marker_idx].strip()
    return cleaned


def _is_low_value_assistant_text(text):
    lowered = (text or "").strip().lower()
    if not lowered:
        return True
    return any(pattern in lowered for pattern in _LOW_VALUE_ASSISTANT_PATTERNS)


def _normalize_language_tag(language):
    key = (language or "").strip().lower()
    if key in _SUPPORTED_LANGUAGES:
        return key
    return "en"


class SessionMemory:
    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = bool(MEMORY_ENABLED)
        self._turns = []
        self._preferred_language = "en"
        self._pending_clarification = None
        self._load()

    def _load(self):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if isinstance(payload, list):
                self._turns = payload
                self._preferred_language = "en"
                self._pending_clarification = None
                return

            if isinstance(payload, dict):
                turns = payload.get("turns")
                if isinstance(turns, list):
                    self._turns = turns
                else:
                    self._turns = []
                self._preferred_language = _normalize_language_tag(payload.get("preferred_language", "en"))
                pending = payload.get("pending_clarification")
                self._pending_clarification = pending if isinstance(pending, dict) else None
                return

            self._turns = []
            self._preferred_language = "en"
            self._pending_clarification = None
        except Exception:
            self._turns = []
            self._preferred_language = "en"
            self._pending_clarification = None

    def _save(self):
        payload = {
            "preferred_language": self._preferred_language,
            "turns": self._turns,
            "pending_clarification": self._pending_clarification,
        }
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=True, indent=2)
        except Exception as exc:
            logger.error("Failed writing memory file: %s", exc)

    def set_enabled(self, enabled):
        with self._lock:
            self._enabled = bool(enabled)
        return True, f"Memory {'enabled' if enabled else 'disabled'}."

    def is_enabled(self):
        with self._lock:
            return self._enabled

    def set_preferred_language(self, language):
        with self._lock:
            self._preferred_language = _normalize_language_tag(language)
            self._save()
        return True, f"Language preference set to: {self._preferred_language}"

    def get_preferred_language(self):
        with self._lock:
            return self._preferred_language

    def set_pending_clarification(self, clarification_payload, ttl_seconds=_DEFAULT_CLARIFICATION_TTL_SECONDS):
        payload = dict(clarification_payload or {})
        now_ts = time.time()
        payload.setdefault("created_at", now_ts)
        payload["expires_at"] = now_ts + max(5, int(ttl_seconds))
        with self._lock:
            self._pending_clarification = payload
            self._save()
        return True, "Clarification pending."

    def clear_pending_clarification(self):
        with self._lock:
            self._pending_clarification = None
            self._save()
        return True, "Clarification cleared."

    def get_pending_clarification(self):
        with self._lock:
            pending = dict(self._pending_clarification or {}) if self._pending_clarification else None
            if not pending:
                return None
            expires_at = float(pending.get("expires_at") or 0.0)
            if expires_at and time.time() > expires_at:
                self._pending_clarification = None
                self._save()
                return None
            return pending

    def add_turn(self, user_text, assistant_text):
        if not self.is_enabled():
            return

        user_clean = (user_text or "").strip()
        assistant_clean = _sanitize_assistant_text(assistant_text)
        if _is_low_value_assistant_text(assistant_clean):
            return

        row = {
            "timestamp": time.time(),
            "user": user_clean,
            "assistant": assistant_clean,
        }
        with self._lock:
            self._turns.append(row)
            if len(self._turns) > max(1, int(MEMORY_MAX_TURNS)):
                self._turns = self._turns[-int(MEMORY_MAX_TURNS) :]
            self._save()

    def clear(self):
        with self._lock:
            self._turns = []
            self._pending_clarification = None
            self._save()
        return True, "Memory cleared."

    def recent(self, limit=MEMORY_MAX_TURNS):
        with self._lock:
            return list(self._turns[-max(1, int(limit)) :])

    def build_context(self, max_chars=MEMORY_MAX_CONTEXT_CHARS):
        if not self.is_enabled():
            return ""
        rows = self.recent()
        if not rows:
            return ""

        lines = []
        chars = 0
        for idx, row in enumerate(rows, start=1):
            user_text = (row.get("user") or "").strip()
            assistant_text = _sanitize_assistant_text(row.get("assistant"))
            if _is_low_value_assistant_text(assistant_text):
                continue
            user_line = f"[{idx}] user: {user_text}"
            assistant_line = f"[{idx}] assistant: {assistant_text}"
            chunk = user_line + "\n" + assistant_line
            if chars + len(chunk) > max_chars:
                break
            lines.append(chunk)
            chars += len(chunk)
        return "\n".join(lines)

    def status(self):
        with self._lock:
            count = len(self._turns)
            enabled = self._enabled
            language = self._preferred_language
            has_pending = bool(self._pending_clarification)
        return {
            "enabled": enabled,
            "turn_count": count,
            "max_turns": int(MEMORY_MAX_TURNS),
            "file": MEMORY_FILE,
            "preferred_language": language,
            "pending_clarification": has_pending,
        }


session_memory = SessionMemory()
