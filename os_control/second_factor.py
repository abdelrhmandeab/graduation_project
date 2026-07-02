import hashlib
import hmac
import re
import threading
import time

from core.config import (
    CONFIRMATION_LOCKOUT_SECONDS,
    CONFIRMATION_MAX_ATTEMPTS_PER_TOKEN,
    SECOND_FACTOR_LOCKOUT_SECONDS,
    SECOND_FACTOR_MAX_ATTEMPTS_PER_TOKEN,
    SECOND_FACTOR_PASSPHRASE,
    SECOND_FACTOR_PIN,
)

_PIN_HASH = hashlib.sha256(SECOND_FACTOR_PIN.encode("utf-8")).hexdigest()
_PASSPHRASE_HASH = hashlib.sha256(SECOND_FACTOR_PASSPHRASE.encode("utf-8")).hexdigest()
_LOCK = threading.Lock()
_ATTEMPTS = {}
_CONFIRM_ATTEMPTS = {}

# Spoken digit words, English + Egyptian Arabic, for normalizing a PIN
# spoken as words ("one two three four" / "واحد اتنين تلاتة اربعة").
_EN_DIGIT_WORDS = {
    "zero": "0", "oh": "0", "o": "0",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
}
_AR_DIGIT_WORDS = {
    "صفر": "0",
    "واحد": "1", "وحدة": "1",
    "اتنين": "2", "اثنين": "2", "اتنان": "2",
    "تلاتة": "3", "ثلاثة": "3", "تلات": "3",
    "اربعة": "4", "اربعه": "4", "اربع": "4",
    "خمسة": "5", "خمسه": "5", "خمس": "5",
    "ستة": "6", "سته": "6", "ست": "6",
    "سبعة": "7", "سبعه": "7", "سبع": "7",
    "تمانية": "8", "ثمانية": "8", "تمانيه": "8", "تمن": "8",
    "تسعة": "9", "تسعه": "9", "تسع": "9",
}
_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_spoken_pin(text):
    """Normalize a spoken PIN (digits or EN/AR number-words) to a digit string."""
    value = str(text or "").strip().translate(_ARABIC_INDIC_DIGITS)
    if not value:
        return ""

    collapsed = re.sub(r"[\s-]+", "", value)
    if collapsed.isdigit():
        return collapsed

    digits = []
    for token in re.findall(r"[a-zA-Z؀-ۿ]+|\d+", value.lower()):
        if token.isdigit():
            digits.append(token)
            continue
        mapped = _EN_DIGIT_WORDS.get(token) or _AR_DIGIT_WORDS.get(token)
        if mapped is not None:
            digits.append(mapped)
    return "".join(digits)


def _hash(value):
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _token_key(token):
    value = (token or "").strip().lower()
    return value or "__global__"


def _attempt_limits():
    max_attempts = max(1, int(SECOND_FACTOR_MAX_ATTEMPTS_PER_TOKEN or 1))
    lockout_seconds = max(1, int(SECOND_FACTOR_LOCKOUT_SECONDS or 1))
    return max_attempts, lockout_seconds


def _confirmation_attempt_limits():
    max_attempts = max(1, int(CONFIRMATION_MAX_ATTEMPTS_PER_TOKEN or 1))
    lockout_seconds = max(1, int(CONFIRMATION_LOCKOUT_SECONDS or 1))
    return max_attempts, lockout_seconds


def clear_second_factor_attempts(token):
    key = _token_key(token)
    with _LOCK:
        _ATTEMPTS.pop(key, None)


def clear_confirmation_attempts(token):
    key = _token_key(token)
    with _LOCK:
        _CONFIRM_ATTEMPTS.pop(key, None)


def is_confirmation_allowed(token):
    key = _token_key(token)
    now_ts = time.time()
    with _LOCK:
        state = _CONFIRM_ATTEMPTS.setdefault(key, {"failed_attempts": 0, "blocked_until": 0.0})
        blocked_until = float(state.get("blocked_until") or 0.0)
        if blocked_until > now_ts:
            remaining = int(max(1, round(blocked_until - now_ts)))
            return False, f"Too many failed confirmation attempts. Retry in {remaining}s."
    return True, ""


def record_confirmation_attempt(token, success):
    key = _token_key(token)
    now_ts = time.time()
    max_attempts, lockout_seconds = _confirmation_attempt_limits()

    with _LOCK:
        state = _CONFIRM_ATTEMPTS.setdefault(key, {"failed_attempts": 0, "blocked_until": 0.0})
        if success:
            _CONFIRM_ATTEMPTS.pop(key, None)
            return

        failed_attempts = int(state.get("failed_attempts") or 0) + 1
        if failed_attempts >= max_attempts:
            state["failed_attempts"] = 0
            state["blocked_until"] = now_ts + lockout_seconds
            _CONFIRM_ATTEMPTS[key] = state
            return

        state["failed_attempts"] = failed_attempts
        state["blocked_until"] = float(state.get("blocked_until") or 0.0)
        _CONFIRM_ATTEMPTS[key] = state


def verify_second_factor(secret, token=""):
    key = _token_key(token)
    now_ts = time.time()
    max_attempts, lockout_seconds = _attempt_limits()

    with _LOCK:
        state = _ATTEMPTS.setdefault(key, {"failed_attempts": 0, "blocked_until": 0.0})
        if float(state.get("blocked_until") or 0.0) > now_ts:
            remaining = int(max(1, round(float(state["blocked_until"]) - now_ts)))
            return False, f"Too many failed second-factor attempts. Retry in {remaining}s."

    candidate = _hash(secret)
    verified = hmac.compare_digest(candidate, _PIN_HASH) or hmac.compare_digest(candidate, _PASSPHRASE_HASH)

    with _LOCK:
        state = _ATTEMPTS.setdefault(key, {"failed_attempts": 0, "blocked_until": 0.0})
        if verified:
            _ATTEMPTS.pop(key, None)
            return True, ""

        failed_attempts = int(state.get("failed_attempts") or 0) + 1
        if failed_attempts >= max_attempts:
            state["failed_attempts"] = 0
            state["blocked_until"] = now_ts + lockout_seconds
            _ATTEMPTS[key] = state
            return False, f"Too many failed second-factor attempts. Retry in {lockout_seconds}s."

        state["failed_attempts"] = failed_attempts
        state["blocked_until"] = float(state.get("blocked_until") or 0.0)
        _ATTEMPTS[key] = state
    return False, "Second factor verification failed."
