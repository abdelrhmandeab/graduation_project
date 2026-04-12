import json
import re
import threading
import time

from core.config import (
    CLARIFICATION_CORRECTION_WINDOW_SECONDS,
    CLARIFICATION_PREFERENCE_HALF_LIFE_SECONDS,
    CLARIFICATION_PREFERENCE_MIN_SCORE,
    FOLLOWUP_APP_REFERENCE_HALF_LIFE_SECONDS,
    FOLLOWUP_APP_REFERENCE_MAX_AGE_SECONDS,
    FOLLOWUP_FILE_REFERENCE_HALF_LIFE_SECONDS,
    FOLLOWUP_FILE_REFERENCE_MAX_AGE_SECONDS,
    FOLLOWUP_PENDING_CONFIRMATION_HALF_LIFE_SECONDS,
    FOLLOWUP_PENDING_CONFIRMATION_MAX_AGE_SECONDS,
    FOLLOWUP_REFERENCE_MIN_CONFIDENCE,
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
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_DEFAULT_CLARIFICATION_TTL_SECONDS = 90
_MAX_CLARIFICATION_PREFERENCES = 80
_MAX_APP_USAGE_ITEMS = 64
_MAX_LANGUAGE_HISTORY_ITEMS = 8
_MAX_CLARIFICATION_SIGNATURE_TOKENS = 4
_DEFAULT_SLOT_MAX_AGE_SECONDS = 1800
_DEFAULT_SLOT_HALF_LIFE_SECONDS = 900
_RUNTIME_DEFAULT_PREFERRED_LANGUAGE = "en"
_CLARIFICATION_SIGNATURE_STOP_WORDS = {
    "open",
    "close",
    "find",
    "search",
    "file",
    "app",
    "the",
    "this",
    "that",
    "please",
    "now",
    "برنامج",
    "ملف",
    "افتح",
    "اقفل",
    "اقفل",
    "دور",
    "من",
    "فضلك",
}
_SLOT_DECAY_POLICIES = {
    "last_app": {
        "max_age_seconds": max(5, int(FOLLOWUP_APP_REFERENCE_MAX_AGE_SECONDS or 1800)),
        "half_life_seconds": max(5, int(FOLLOWUP_APP_REFERENCE_HALF_LIFE_SECONDS or 900)),
    },
    "previous_app": {
        "max_age_seconds": max(5, int(FOLLOWUP_APP_REFERENCE_MAX_AGE_SECONDS or 1800)),
        "half_life_seconds": max(5, int(FOLLOWUP_APP_REFERENCE_HALF_LIFE_SECONDS or 900)),
    },
    "last_file": {
        "max_age_seconds": max(5, int(FOLLOWUP_FILE_REFERENCE_MAX_AGE_SECONDS or 1800)),
        "half_life_seconds": max(5, int(FOLLOWUP_FILE_REFERENCE_HALF_LIFE_SECONDS or 720)),
    },
    "pending_confirmation_token": {
        "max_age_seconds": max(5, int(FOLLOWUP_PENDING_CONFIRMATION_MAX_AGE_SECONDS or 180)),
        "half_life_seconds": max(5, int(FOLLOWUP_PENDING_CONFIRMATION_HALF_LIFE_SECONDS or 75)),
    },
}


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


def _normalize_intent_tag(intent):
    return str(intent or "").strip().upper()


def _detect_text_language_hint(text, fallback=""):
    value = str(text or "")
    has_ar = bool(_ARABIC_CHAR_RE.search(value))
    has_en = bool(_LATIN_CHAR_RE.search(value))
    if has_ar and not has_en:
        return "ar"
    if has_en and not has_ar:
        return "en"
    key = str(fallback or "").strip().lower()
    if key in _SUPPORTED_LANGUAGES:
        return key
    return ""


def _infer_turn_language(user_text, assistant_text, fallback=""):
    user_lang = _detect_text_language_hint(user_text, fallback="")
    if user_lang in _SUPPORTED_LANGUAGES:
        return user_lang
    assistant_lang = _detect_text_language_hint(assistant_text, fallback="")
    if assistant_lang in _SUPPORTED_LANGUAGES:
        return assistant_lang
    return _detect_text_language_hint("", fallback=fallback)


def _normalize_memory_token(text):
    value = " ".join(str(text or "").lower().split()).strip()
    value = re.sub(r"[^a-z0-9\u0600-\u06FF\s._-]", " ", value)
    return " ".join(value.split())


def _normalize_app_key(app_name):
    value = _normalize_memory_token(app_name)
    if value.startswith("start "):
        value = value[6:].strip()
    if value.endswith(":"):
        value = value[:-1].strip()
    return value


def _clarification_source_signature(text):
    tokens = re.findall(r"[a-z0-9\u0600-\u06FF]+", _normalize_memory_token(text))
    filtered = [tok for tok in tokens if tok and tok not in _CLARIFICATION_SIGNATURE_STOP_WORDS]
    if not filtered:
        filtered = tokens
    return " ".join(filtered[:_MAX_CLARIFICATION_SIGNATURE_TOKENS])


class SessionMemory:
    def __init__(self):
        self._lock = threading.Lock()
        self._enabled = bool(MEMORY_ENABLED)
        self._turns = []
        self._preferred_language = _RUNTIME_DEFAULT_PREFERRED_LANGUAGE
        self._pending_clarification = None
        self._context_slots = self._default_context_slots()
        self._load()

    def _default_context_slots(self):
        return {
            "last_app": "",
            "last_app_updated_at": 0.0,
            "previous_app": "",
            "previous_app_updated_at": 0.0,
            "last_file": "",
            "last_file_updated_at": 0.0,
            "pending_confirmation_token": "",
            "pending_confirmation_updated_at": 0.0,
            "clarification_preferences": {},
            "clarification_preferences_updated_at": 0.0,
            "app_usage": {},
            "app_usage_updated_at": 0.0,
            "stt_profile": "",
            "stt_profile_updated_at": 0.0,
            "audio_ux_profile": "",
            "audio_ux_profile_updated_at": 0.0,
            "response_mode": "default",
            "response_mode_updated_at": 0.0,
            "language_history": [],
            "language_history_updated_at": 0.0,
            "last_clarification_resolution_at": 0.0,
            "last_clarification_resolution_reason": "",
            "last_clarification_resolution_intent": "",
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
        payload["previous_app"] = str(slots.get("previous_app") or "").strip()
        payload["previous_app_updated_at"] = self._as_timestamp(slots.get("previous_app_updated_at"))
        payload["last_file"] = str(slots.get("last_file") or "").strip()
        payload["last_file_updated_at"] = self._as_timestamp(slots.get("last_file_updated_at"))
        payload["pending_confirmation_token"] = str(slots.get("pending_confirmation_token") or "").strip().lower()
        payload["pending_confirmation_updated_at"] = self._as_timestamp(
            slots.get("pending_confirmation_updated_at")
        )
        pref_payload = slots.get("clarification_preferences")
        payload["clarification_preferences"] = dict(pref_payload) if isinstance(pref_payload, dict) else {}
        payload["clarification_preferences_updated_at"] = self._as_timestamp(
            slots.get("clarification_preferences_updated_at")
        )
        app_usage_payload = slots.get("app_usage")
        payload["app_usage"] = dict(app_usage_payload) if isinstance(app_usage_payload, dict) else {}
        payload["app_usage_updated_at"] = self._as_timestamp(slots.get("app_usage_updated_at"))
        payload["stt_profile"] = str(slots.get("stt_profile") or "").strip().lower()
        payload["stt_profile_updated_at"] = self._as_timestamp(slots.get("stt_profile_updated_at"))
        payload["audio_ux_profile"] = str(slots.get("audio_ux_profile") or "").strip().lower()
        payload["audio_ux_profile_updated_at"] = self._as_timestamp(slots.get("audio_ux_profile_updated_at"))
        response_mode = str(slots.get("response_mode") or "default").strip().lower()
        payload["response_mode"] = response_mode if response_mode in {"default", "explain", "concise"} else "default"
        payload["response_mode_updated_at"] = self._as_timestamp(slots.get("response_mode_updated_at"))
        # Language carry-over must not leak across app restarts.
        payload["language_history"] = []
        payload["language_history_updated_at"] = 0.0
        payload["last_clarification_resolution_at"] = self._as_timestamp(
            slots.get("last_clarification_resolution_at")
        )
        payload["last_clarification_resolution_reason"] = str(
            slots.get("last_clarification_resolution_reason") or ""
        ).strip().lower()
        payload["last_clarification_resolution_intent"] = str(
            slots.get("last_clarification_resolution_intent") or ""
        ).strip().upper()
        return payload

    def _load(self):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as handle:
                payload = json.load(handle)

            if isinstance(payload, list):
                self._turns = payload
                # Language is runtime-only and should not be inherited from historical chats.
                self._preferred_language = _RUNTIME_DEFAULT_PREFERRED_LANGUAGE
                self._pending_clarification = None
                self._context_slots = self._default_context_slots()
                return

            if isinstance(payload, dict):
                turns = payload.get("turns")
                if isinstance(turns, list):
                    self._turns = turns
                else:
                    self._turns = []
                # Start each app session with a neutral language state.
                self._preferred_language = _RUNTIME_DEFAULT_PREFERRED_LANGUAGE
                # Do not carry pending clarification from previous runs.
                self._pending_clarification = None

                slots = payload.get("context_slots")
                self._context_slots = self._load_context_slots(slots)
                return

            self._turns = []
            self._preferred_language = _RUNTIME_DEFAULT_PREFERRED_LANGUAGE
            self._pending_clarification = None
            self._context_slots = self._default_context_slots()
        except Exception:
            self._turns = []
            self._preferred_language = _RUNTIME_DEFAULT_PREFERRED_LANGUAGE
            self._pending_clarification = None
            self._context_slots = self._default_context_slots()

    def _save(self):
        payload = {
            # Keep schema backward-compatible without persisting runtime language state.
            "preferred_language": _RUNTIME_DEFAULT_PREFERRED_LANGUAGE,
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
        created_at = float(payload.get("created_at") or now_ts)
        payload["created_at"] = created_at
        explicit_expires_at = float(payload.get("expires_at") or 0.0)
        if explicit_expires_at and explicit_expires_at > now_ts:
            payload["expires_at"] = explicit_expires_at
        else:
            payload["expires_at"] = now_ts + max(5, int(ttl_seconds))
        payload["attempts"] = int(payload.get("attempts") or 0)
        payload["fallback_hint_sent"] = bool(payload.get("fallback_hint_sent"))
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
        now_ts = time.time()
        with self._lock:
            previous_last = str(self._context_slots.get("last_app") or "").strip()
            previous_last_ts = self._as_timestamp(self._context_slots.get("last_app_updated_at"))
            if previous_last and previous_last.lower() != value.lower():
                self._context_slots["previous_app"] = previous_last
                self._context_slots["previous_app_updated_at"] = previous_last_ts or now_ts
            self._context_slots["last_app"] = value
            self._context_slots["last_app_updated_at"] = now_ts
            self._save()
        return True, f"last_app set to: {value}"

    def record_app_usage(self, app_name):
        key = _normalize_app_key(app_name)
        if not key:
            return False, "app_usage not updated."
        now_ts = time.time()
        with self._lock:
            usage = dict(self._context_slots.get("app_usage") or {})
            row = dict(usage.get(key) or {})
            row["count"] = int(row.get("count") or 0) + 1
            row["last_used_at"] = now_ts
            usage[key] = row
            if len(usage) > _MAX_APP_USAGE_ITEMS:
                sorted_items = sorted(
                    usage.items(),
                    key=lambda item: float((item[1] or {}).get("last_used_at") or 0.0),
                    reverse=True,
                )
                usage = dict(sorted_items[:_MAX_APP_USAGE_ITEMS])
            self._context_slots["app_usage"] = usage
            self._context_slots["app_usage_updated_at"] = now_ts
            self._save()
        return True, f"app_usage updated: {key}"

    def get_app_usage_stats(self, app_name):
        key = _normalize_app_key(app_name)
        if not key:
            return {"count": 0, "last_used_at": 0.0}
        with self._lock:
            usage = dict(self._context_slots.get("app_usage") or {})
            row = dict(usage.get(key) or {})
            return {
                "count": int(row.get("count") or 0),
                "last_used_at": self._as_timestamp(row.get("last_used_at")),
            }

    def _clarification_preference_key(self, reason, source_text, language=""):
        reason_key = _normalize_memory_token(reason)
        source_key = _normalize_memory_token(source_text)
        lang_key = _normalize_language_tag(language)
        return f"{reason_key}::{lang_key}::{source_key}"

    def _clarification_preference_score(self, row, now_ts=None):
        payload = dict(row or {})
        now_value = float(now_ts or time.time())
        updated_at = self._as_timestamp(payload.get("updated_at"))
        if updated_at <= 0.0:
            return 0.0

        age_seconds = max(0.0, now_value - updated_at)
        half_life = max(60, int(CLARIFICATION_PREFERENCE_HALF_LIFE_SECONDS or 1209600))
        decay = pow(0.5, age_seconds / float(half_life))

        selection_count = int(payload.get("selection_count") or 0)
        confirmed_count = int(payload.get("confirmed_count") or 0)
        reuse_hits = int(payload.get("reuse_hit_count") or 0)
        reuse_misses = int(payload.get("reuse_miss_count") or 0)

        base = 0.22
        base += min(0.42, selection_count * 0.08)
        base += min(0.22, confirmed_count * 0.06)
        base += min(0.16, reuse_hits * 0.04)
        penalty = min(0.45, reuse_misses * 0.12)

        score = (base - penalty) * decay
        return max(0.0, min(1.0, score))

    def _find_best_clarification_preference(
        self,
        preferences,
        *,
        reason,
        source_text,
        language,
        max_age_seconds,
    ):
        reason_key = _normalize_memory_token(reason)
        language_key = _normalize_language_tag(language)
        source_key = _normalize_memory_token(source_text)
        source_signature = _clarification_source_signature(source_text)
        now_ts = time.time()

        age_limit = max(0, int(max_age_seconds or 0))
        min_score = max(0.0, min(1.0, float(CLARIFICATION_PREFERENCE_MIN_SCORE or 0.34)))

        best_row = None
        best_score = 0.0
        for row in list(preferences.values()):
            payload = dict(row or {})
            if str(payload.get("reason") or "") != reason_key:
                continue
            if _normalize_language_tag(payload.get("language")) != language_key:
                continue

            payload_source = _normalize_memory_token(payload.get("source_text"))
            payload_signature = str(payload.get("source_signature") or "").strip()
            is_exact = bool(source_key and payload_source == source_key)
            is_signature = bool(source_signature and payload_signature and payload_signature == source_signature)
            if not (is_exact or is_signature):
                continue

            updated_at = self._as_timestamp(payload.get("updated_at"))
            if age_limit and updated_at and (now_ts - updated_at) > age_limit:
                continue

            score = self._clarification_preference_score(payload, now_ts=now_ts)
            if is_exact:
                score += 0.08
            if score < min_score:
                continue
            if score <= best_score:
                continue

            payload["preference_score"] = max(0.0, min(1.0, score))
            payload["preference_match"] = "exact" if is_exact else "signature"
            best_row = payload
            best_score = score

        return best_row

    def remember_clarification_choice(self, reason, source_text, option, language=""):
        key = self._clarification_preference_key(reason, source_text, language=language)
        option_payload = dict(option or {})
        if not key or not option_payload:
            return False, "clarification preference not updated."

        now_ts = time.time()
        reason_key = _normalize_memory_token(reason)
        source_key = _normalize_memory_token(source_text)
        source_signature = _clarification_source_signature(source_text)
        language_key = _normalize_language_tag(language)

        with self._lock:
            preferences = dict(self._context_slots.get("clarification_preferences") or {})
            previous = dict(preferences.get(key) or {})
            row = {
                "reason": reason_key,
                "source_text": source_key,
                "source_signature": source_signature,
                "language": language_key,
                "id": str(option_payload.get("id") or "").strip(),
                "intent": str(option_payload.get("intent") or "").strip(),
                "action": str(option_payload.get("action") or "").strip(),
                "args": dict(option_payload.get("args") or {}),
                "selection_count": int(previous.get("selection_count") or 0) + 1,
                "confirmed_count": int(previous.get("confirmed_count") or 0) + 1,
                "reuse_hit_count": int(previous.get("reuse_hit_count") or 0),
                "reuse_miss_count": int(previous.get("reuse_miss_count") or 0),
                "created_at": self._as_timestamp(previous.get("created_at")) or now_ts,
                "updated_at": now_ts,
            }
            preferences[key] = row
            if len(preferences) > _MAX_CLARIFICATION_PREFERENCES:
                sorted_items = sorted(
                    preferences.items(),
                    key=lambda item: float((item[1] or {}).get("updated_at") or 0.0),
                    reverse=True,
                )
                preferences = dict(sorted_items[:_MAX_CLARIFICATION_PREFERENCES])
            self._context_slots["clarification_preferences"] = preferences
            self._context_slots["clarification_preferences_updated_at"] = now_ts
            self._save()
        return True, "clarification preference saved."

    def get_clarification_choice(self, reason, source_text, language="", max_age_seconds=0):
        with self._lock:
            preferences = dict(self._context_slots.get("clarification_preferences") or {})
        if not preferences:
            return None

        return self._find_best_clarification_preference(
            preferences,
            reason=reason,
            source_text=source_text,
            language=language,
            max_age_seconds=max_age_seconds,
        )

    def mark_clarification_reuse_feedback(self, reason, source_text, *, language="", success=False):
        key = self._clarification_preference_key(reason, source_text, language=language)
        if not key:
            return False, "clarification preference feedback skipped."

        with self._lock:
            preferences = dict(self._context_slots.get("clarification_preferences") or {})
            row = dict(preferences.get(key) or {})
            if not row:
                row = self._find_best_clarification_preference(
                    preferences,
                    reason=reason,
                    source_text=source_text,
                    language=language,
                    max_age_seconds=0,
                )
                if not row:
                    return False, "clarification preference feedback skipped."
                key = self._clarification_preference_key(
                    row.get("reason"),
                    row.get("source_text"),
                    language=row.get("language"),
                )
                row = dict(preferences.get(key) or row)

            if success:
                row["reuse_hit_count"] = int(row.get("reuse_hit_count") or 0) + 1
            else:
                row["reuse_miss_count"] = int(row.get("reuse_miss_count") or 0) + 1
            row["updated_at"] = time.time()

            preferences[key] = row
            self._context_slots["clarification_preferences"] = preferences
            self._context_slots["clarification_preferences_updated_at"] = time.time()
            self._save()
        return True, "clarification preference feedback saved."

    def mark_clarification_resolution(self, *, reason="", intent=""):
        with self._lock:
            self._context_slots["last_clarification_resolution_at"] = time.time()
            self._context_slots["last_clarification_resolution_reason"] = _normalize_memory_token(reason)
            self._context_slots["last_clarification_resolution_intent"] = str(intent or "").strip().upper()
            self._save()
        return True, "clarification resolution recorded."

    def recent_clarification_resolution(self, max_age_seconds=0):
        age_limit = max(1, int(max_age_seconds or CLARIFICATION_CORRECTION_WINDOW_SECONDS or 45))
        with self._lock:
            ts = self._as_timestamp(self._context_slots.get("last_clarification_resolution_at"))
            if ts <= 0.0:
                return None
            age = max(0.0, time.time() - ts)
            if age > age_limit:
                return None
            return {
                "timestamp": ts,
                "age_seconds": age,
                "reason": str(self._context_slots.get("last_clarification_resolution_reason") or "").strip(),
                "intent": str(self._context_slots.get("last_clarification_resolution_intent") or "").strip().upper(),
            }

    def clear_clarification_preferences(self):
        with self._lock:
            self._context_slots["clarification_preferences"] = {}
            self._context_slots["clarification_preferences_updated_at"] = 0.0
            self._save()
        return True, "clarification preferences cleared."

    def clear_app_usage(self):
        with self._lock:
            self._context_slots["app_usage"] = {}
            self._context_slots["app_usage_updated_at"] = 0.0
            self._save()
        return True, "app usage cleared."

    def get_last_app(self):
        with self._lock:
            return str(self._context_slots.get("last_app") or "").strip()

    def get_last_app_timestamp(self):
        with self._lock:
            return self._as_timestamp(self._context_slots.get("last_app_updated_at"))

    def get_previous_app(self):
        with self._lock:
            return str(self._context_slots.get("previous_app") or "").strip()

    def get_previous_app_timestamp(self):
        with self._lock:
            return self._as_timestamp(self._context_slots.get("previous_app_updated_at"))

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

    def get_pending_confirmation_timestamp(self):
        with self._lock:
            return self._as_timestamp(self._context_slots.get("pending_confirmation_updated_at"))

    def clear_pending_confirmation_token(self):
        with self._lock:
            self._context_slots["pending_confirmation_token"] = ""
            self._context_slots["pending_confirmation_updated_at"] = 0.0
            self._save()
        return True, "pending_confirmation_token cleared."

    def set_stt_profile(self, profile_name):
        value = (profile_name or "").strip().lower()
        with self._lock:
            self._context_slots["stt_profile"] = value
            self._context_slots["stt_profile_updated_at"] = time.time() if value else 0.0
            self._save()
        return True, f"stt_profile set to: {value or 'default'}"

    def get_stt_profile(self):
        with self._lock:
            return str(self._context_slots.get("stt_profile") or "").strip().lower()

    def get_stt_profile_timestamp(self):
        with self._lock:
            return self._as_timestamp(self._context_slots.get("stt_profile_updated_at"))

    def set_audio_ux_profile(self, profile_name):
        value = (profile_name or "").strip().lower()
        with self._lock:
            self._context_slots["audio_ux_profile"] = value
            self._context_slots["audio_ux_profile_updated_at"] = time.time() if value else 0.0
            self._save()
        return True, f"audio_ux_profile set to: {value or 'custom'}"

    def get_audio_ux_profile(self):
        with self._lock:
            return str(self._context_slots.get("audio_ux_profile") or "").strip().lower()

    def get_audio_ux_profile_timestamp(self):
        with self._lock:
            return self._as_timestamp(self._context_slots.get("audio_ux_profile_updated_at"))

    def set_response_mode(self, mode):
        value = str(mode or "default").strip().lower()
        if value not in {"default", "explain", "concise"}:
            return False, "response_mode must be default|explain|concise"
        with self._lock:
            self._context_slots["response_mode"] = value
            self._context_slots["response_mode_updated_at"] = time.time()
            self._save()
        return True, f"response_mode set to: {value}"

    def get_response_mode(self):
        with self._lock:
            value = str(self._context_slots.get("response_mode") or "default").strip().lower()
            return value if value in {"default", "explain", "concise"} else "default"

    def record_language_turn(self, language):
        value = _normalize_language_tag(language)
        now_ts = time.time()
        with self._lock:
            history = list(self._context_slots.get("language_history") or [])
            history.append(value)
            self._context_slots["language_history"] = history[-_MAX_LANGUAGE_HISTORY_ITEMS:]
            self._context_slots["language_history_updated_at"] = now_ts
            self._save()
        return True, f"language turn recorded: {value}"

    def get_recent_languages(self, limit=4):
        with self._lock:
            history = list(self._context_slots.get("language_history") or [])
        return history[-max(1, int(limit)):]

    def get_language_mix(self, window=6):
        recent = self.get_recent_languages(limit=max(2, int(window or 2)))
        total = len(recent)
        if total <= 0:
            return {
                "total": 0,
                "en_count": 0,
                "ar_count": 0,
                "en_ratio": 0.0,
                "ar_ratio": 0.0,
                "dominant": "mixed",
                "recent": [],
            }

        en_count = sum(1 for item in recent if item == "en")
        ar_count = sum(1 for item in recent if item == "ar")
        en_ratio = float(en_count) / float(total)
        ar_ratio = float(ar_count) / float(total)

        dominant = "mixed"
        if en_ratio > ar_ratio:
            dominant = "en"
        elif ar_ratio > en_ratio:
            dominant = "ar"

        return {
            "total": total,
            "en_count": en_count,
            "ar_count": ar_count,
            "en_ratio": en_ratio,
            "ar_ratio": ar_ratio,
            "dominant": dominant,
            "recent": recent,
        }

    def is_code_switch_active(self, window=4):
        recent = self.get_recent_languages(limit=window)
        if len(recent) < 2:
            return False
        has_en = "en" in recent
        has_ar = "ar" in recent
        if not (has_en and has_ar):
            return False
        return any(recent[idx] != recent[idx - 1] for idx in range(1, len(recent)))

    def slot_confidence(self, slot_name, updated_at=None):
        key = str(slot_name or "").strip().lower()
        policy = dict(_SLOT_DECAY_POLICIES.get(key) or {})
        max_age = max(5, int(policy.get("max_age_seconds") or _DEFAULT_SLOT_MAX_AGE_SECONDS))
        half_life = max(5, int(policy.get("half_life_seconds") or _DEFAULT_SLOT_HALF_LIFE_SECONDS))

        if updated_at is None:
            with self._lock:
                ts = self._as_timestamp(self._context_slots.get(f"{key}_updated_at"))
        else:
            ts = self._as_timestamp(updated_at)

        if ts <= 0.0:
            return 0.0

        age = max(0.0, time.time() - ts)
        if age > max_age:
            return 0.0

        decay = pow(0.5, age / float(half_life))
        confidence = max(0.0, min(1.0, decay))
        return confidence

    def slot_is_fresh(self, slot_name, updated_at=None):
        confidence = self.slot_confidence(slot_name, updated_at=updated_at)
        return confidence >= float(FOLLOWUP_REFERENCE_MIN_CONFIDENCE or 0.2)

    def context_snapshot(self):
        with self._lock:
            return {
                "last_app": str(self._context_slots.get("last_app") or "").strip(),
                "last_app_updated_at": self._as_timestamp(self._context_slots.get("last_app_updated_at")),
                "previous_app": str(self._context_slots.get("previous_app") or "").strip(),
                "previous_app_updated_at": self._as_timestamp(self._context_slots.get("previous_app_updated_at")),
                "last_file": str(self._context_slots.get("last_file") or "").strip(),
                "last_file_updated_at": self._as_timestamp(self._context_slots.get("last_file_updated_at")),
                "pending_confirmation_token": str(self._context_slots.get("pending_confirmation_token") or "").strip().lower(),
                "pending_confirmation_updated_at": self._as_timestamp(
                    self._context_slots.get("pending_confirmation_updated_at")
                ),
                "clarification_preferences_count": len(dict(self._context_slots.get("clarification_preferences") or {})),
                "clarification_preferences_updated_at": self._as_timestamp(
                    self._context_slots.get("clarification_preferences_updated_at")
                ),
                "app_usage_count": len(dict(self._context_slots.get("app_usage") or {})),
                "app_usage_updated_at": self._as_timestamp(self._context_slots.get("app_usage_updated_at")),
                "stt_profile": str(self._context_slots.get("stt_profile") or "").strip().lower(),
                "stt_profile_updated_at": self._as_timestamp(self._context_slots.get("stt_profile_updated_at")),
                "audio_ux_profile": str(self._context_slots.get("audio_ux_profile") or "").strip().lower(),
                "audio_ux_profile_updated_at": self._as_timestamp(self._context_slots.get("audio_ux_profile_updated_at")),
                "response_mode": str(self._context_slots.get("response_mode") or "default").strip().lower(),
                "response_mode_updated_at": self._as_timestamp(self._context_slots.get("response_mode_updated_at")),
                "language_history": list(self._context_slots.get("language_history") or []),
                "language_history_updated_at": self._as_timestamp(self._context_slots.get("language_history_updated_at")),
                "last_clarification_resolution_at": self._as_timestamp(
                    self._context_slots.get("last_clarification_resolution_at")
                ),
                "last_clarification_resolution_reason": str(
                    self._context_slots.get("last_clarification_resolution_reason") or ""
                ).strip(),
                "last_clarification_resolution_intent": str(
                    self._context_slots.get("last_clarification_resolution_intent") or ""
                ).strip().upper(),
            }

    def add_turn(self, user_text, assistant_text, language=None, intent=None):
        if not self.is_enabled():
            return

        user_clean = (user_text or "").strip()
        assistant_clean = _sanitize_assistant_text(assistant_text)
        if _is_low_value_assistant_text(assistant_clean):
            return

        preferred = str(language or "").strip().lower()
        resolved_language = (
            _normalize_language_tag(preferred)
            if preferred in _SUPPORTED_LANGUAGES
            else _infer_turn_language(user_clean, assistant_clean, fallback="")
        )

        row = {
            "timestamp": time.time(),
            "user": user_clean,
            "assistant": assistant_clean,
            "language": resolved_language,
            "intent": _normalize_intent_tag(intent),
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

    def build_context(self, max_chars=MEMORY_MAX_CONTEXT_CHARS, language=None, intents=None):
        if not self.is_enabled():
            return ""
        rows = self.recent()
        if not rows:
            return ""

        target_language = str(language or "").strip().lower()
        if target_language not in _SUPPORTED_LANGUAGES:
            target_language = ""

        allowed_intents = None
        if intents is not None:
            allowed_intents = {
                _normalize_intent_tag(item)
                for item in list(intents or [])
                if _normalize_intent_tag(item)
            }
            if not allowed_intents:
                allowed_intents = None

        lines = []
        chars = 0
        line_number = 0
        for row in rows:
            user_text = (row.get("user") or "").strip()
            assistant_text = _sanitize_assistant_text(row.get("assistant"))
            if _is_low_value_assistant_text(assistant_text):
                continue

            row_intent = _normalize_intent_tag(row.get("intent"))
            if allowed_intents is not None and row_intent not in allowed_intents:
                continue

            row_language = str(row.get("language") or "").strip().lower()
            if row_language not in _SUPPORTED_LANGUAGES:
                row_language = _infer_turn_language(user_text, assistant_text, fallback="")
            if target_language and row_language != target_language:
                continue

            line_number += 1
            user_line = f"[{line_number}] user: {user_text}"
            assistant_line = f"[{line_number}] assistant: {assistant_text}"
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
            previous_app = str(self._context_slots.get("previous_app") or "").strip()
            previous_app_updated_at = self._as_timestamp(self._context_slots.get("previous_app_updated_at"))
            last_file = str(self._context_slots.get("last_file") or "").strip()
            last_file_updated_at = self._as_timestamp(self._context_slots.get("last_file_updated_at"))
            pending_token = str(self._context_slots.get("pending_confirmation_token") or "").strip().lower()
            pending_confirmation_updated_at = self._as_timestamp(
                self._context_slots.get("pending_confirmation_updated_at")
            )
            clarification_preferences_count = len(dict(self._context_slots.get("clarification_preferences") or {}))
            clarification_preferences_updated_at = self._as_timestamp(
                self._context_slots.get("clarification_preferences_updated_at")
            )
            app_usage_count = len(dict(self._context_slots.get("app_usage") or {}))
            app_usage_updated_at = self._as_timestamp(self._context_slots.get("app_usage_updated_at"))
            stt_profile = str(self._context_slots.get("stt_profile") or "").strip().lower()
            stt_profile_updated_at = self._as_timestamp(self._context_slots.get("stt_profile_updated_at"))
            audio_ux_profile = str(self._context_slots.get("audio_ux_profile") or "").strip().lower()
            audio_ux_profile_updated_at = self._as_timestamp(self._context_slots.get("audio_ux_profile_updated_at"))
            response_mode = str(self._context_slots.get("response_mode") or "default").strip().lower()
            response_mode_updated_at = self._as_timestamp(self._context_slots.get("response_mode_updated_at"))
            language_history = list(self._context_slots.get("language_history") or [])
            language_history_updated_at = self._as_timestamp(self._context_slots.get("language_history_updated_at"))
            last_clarification_resolution_at = self._as_timestamp(
                self._context_slots.get("last_clarification_resolution_at")
            )
            last_clarification_resolution_reason = str(
                self._context_slots.get("last_clarification_resolution_reason") or ""
            ).strip()
            last_clarification_resolution_intent = str(
                self._context_slots.get("last_clarification_resolution_intent") or ""
            ).strip().upper()
        language_mix = self.get_language_mix(window=_MAX_LANGUAGE_HISTORY_ITEMS)
        return {
            "enabled": enabled,
            "turn_count": count,
            "max_turns": int(MEMORY_MAX_TURNS),
            "file": MEMORY_FILE,
            "preferred_language": language,
            "pending_clarification": has_pending,
            "last_app": last_app,
            "last_app_updated_at": last_app_updated_at,
            "previous_app": previous_app,
            "previous_app_updated_at": previous_app_updated_at,
            "last_file": last_file,
            "last_file_updated_at": last_file_updated_at,
            "pending_confirmation_token": pending_token,
            "pending_confirmation_updated_at": pending_confirmation_updated_at,
            "clarification_preferences_count": clarification_preferences_count,
            "clarification_preferences_updated_at": clarification_preferences_updated_at,
            "app_usage_count": app_usage_count,
            "app_usage_updated_at": app_usage_updated_at,
            "stt_profile": stt_profile,
            "stt_profile_updated_at": stt_profile_updated_at,
            "audio_ux_profile": audio_ux_profile,
            "audio_ux_profile_updated_at": audio_ux_profile_updated_at,
            "response_mode": response_mode,
            "response_mode_updated_at": response_mode_updated_at,
            "language_history": language_history,
            "language_history_updated_at": language_history_updated_at,
            "language_mix": language_mix,
            "last_clarification_resolution_at": last_clarification_resolution_at,
            "last_clarification_resolution_reason": last_clarification_resolution_reason,
            "last_clarification_resolution_intent": last_clarification_resolution_intent,
        }


session_memory = SessionMemory()
