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
        self._context_slots = self._default_context_slots()
        self._load()

    def _default_context_slots(self):
        return {
            "last_app": "",
            "last_app_updated_at": 0.0,
            "last_file": "",
            "last_file_updated_at": 0.0,
            "pending_confirmation_token": "",
            "pending_confirmation_updated_at": 0.0,
        }

    def _as_timestamp(self, value):
        try:
            return float(value or 0.0)
        except Exception:
            return 0.0

    def _load_context_slots(self, slots):
        if not isinstance(slots, dict):
            return self._default_context_slots()
        payload = self._default_context_slots()
        payload["last_app"] = str(slots.get("last_app") or "").strip()
        payload["last_app_updated_at"] = self._as_timestamp(slots.get("last_app_updated_at"))
        payload["last_file"] = str(slots.get("last_file") or "").strip()
        payload["last_file_updated_at"] = self._as_timestamp(slots.get("last_file_updated_at"))
        payload["pending_confirmation_token"] = str(slots.get("pending_confirmation_token") or "").strip().lower()
        payload["pending_confirmation_updated_at"] = self._as_timestamp(
            slots.get("pending_confirmation_updated_at")
        )
        return payload

    def _load(self):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if isinstance(payload, list):
                self._turns = payload
                self._preferred_language = "en"
                self._pending_clarification = None
                self._context_slots = self._default_context_slots()
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

                slots = payload.get("context_slots")
                self._context_slots = self._load_context_slots(slots)
                return

            self._turns = []
            self._preferred_language = "en"
            self._pending_clarification = None
            self._context_slots = self._default_context_slots()
        except Exception:
            self._turns = []
            self._preferred_language = "en"
            self._pending_clarification = None
            self._context_slots = self._default_context_slots()

    def _save(self):
        payload = {
            "preferred_language": self._preferred_language,
            "turns": self._turns,
            "pending_clarification": self._pending_clarification,
            "context_slots": self._context_slots,
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

    def set_last_app(self, app_name):
        value = (app_name or "").strip()
        if not value:
            return False, "last_app not updated."
        with self._lock:
            self._context_slots["last_app"] = value
            self._context_slots["last_app_updated_at"] = time.time()
            self._save()
        return True, f"last_app set to: {value}"

    def get_last_app(self):
        with self._lock:
            return str(self._context_slots.get("last_app") or "").strip()

    def get_last_app_timestamp(self):
        with self._lock:
            return self._as_timestamp(self._context_slots.get("last_app_updated_at"))

    def set_last_file(self, file_path):
        value = (file_path or "").strip()
        if not value:
            return False, "last_file not updated."
        with self._lock:
            self._context_slots["last_file"] = value
            self._context_slots["last_file_updated_at"] = time.time()
            self._save()
        return True, f"last_file set to: {value}"

    def get_last_file(self):
        with self._lock:
            return str(self._context_slots.get("last_file") or "").strip()

    def get_last_file_timestamp(self):
        with self._lock:
            return self._as_timestamp(self._context_slots.get("last_file_updated_at"))

    def set_pending_confirmation_token(self, token):
        value = (token or "").strip().lower()
        if not value:
            return False, "pending_confirmation_token not updated."
        with self._lock:
            self._context_slots["pending_confirmation_token"] = value
            self._context_slots["pending_confirmation_updated_at"] = time.time()
            self._save()
        return True, f"pending_confirmation_token set to: {value}"

    def get_pending_confirmation_token(self):
        with self._lock:
            return str(self._context_slots.get("pending_confirmation_token") or "").strip().lower()

    def clear_pending_confirmation_token(self):
        with self._lock:
            self._context_slots["pending_confirmation_token"] = ""
            self._context_slots["pending_confirmation_updated_at"] = 0.0
            self._save()
        return True, "pending_confirmation_token cleared."

    def context_snapshot(self):
        with self._lock:
            return {
                "last_app": str(self._context_slots.get("last_app") or "").strip(),
                "last_app_updated_at": self._as_timestamp(self._context_slots.get("last_app_updated_at")),
                "last_file": str(self._context_slots.get("last_file") or "").strip(),
                "last_file_updated_at": self._as_timestamp(self._context_slots.get("last_file_updated_at")),
                "pending_confirmation_token": str(self._context_slots.get("pending_confirmation_token") or "").strip().lower(),
                "pending_confirmation_updated_at": self._as_timestamp(
                    self._context_slots.get("pending_confirmation_updated_at")
                ),
            }

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
            self._context_slots = self._default_context_slots()
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
            last_app = str(self._context_slots.get("last_app") or "").strip()
            last_app_updated_at = self._as_timestamp(self._context_slots.get("last_app_updated_at"))
            last_file = str(self._context_slots.get("last_file") or "").strip()
            last_file_updated_at = self._as_timestamp(self._context_slots.get("last_file_updated_at"))
            pending_token = str(self._context_slots.get("pending_confirmation_token") or "").strip().lower()
            pending_confirmation_updated_at = self._as_timestamp(
                self._context_slots.get("pending_confirmation_updated_at")
            )
        return {
            "enabled": enabled,
            "turn_count": count,
            "max_turns": int(MEMORY_MAX_TURNS),
            "file": MEMORY_FILE,
            "preferred_language": language,
            "pending_clarification": has_pending,
            "last_app": last_app,
            "last_app_updated_at": last_app_updated_at,
            "last_file": last_file,
            "last_file_updated_at": last_file_updated_at,
            "pending_confirmation_token": pending_token,
            "pending_confirmation_updated_at": pending_confirmation_updated_at,
        }


session_memory = SessionMemory()
