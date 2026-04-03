import hashlib
import hmac
import threading
import time

from core.config import (
    SECOND_FACTOR_LOCKOUT_SECONDS,
    SECOND_FACTOR_MAX_ATTEMPTS_PER_TOKEN,
    SECOND_FACTOR_PASSPHRASE,
    SECOND_FACTOR_PIN,
)

_PIN_HASH = hashlib.sha256(SECOND_FACTOR_PIN.encode("utf-8")).hexdigest()
_PASSPHRASE_HASH = hashlib.sha256(SECOND_FACTOR_PASSPHRASE.encode("utf-8")).hexdigest()
_LOCK = threading.Lock()
_ATTEMPTS = {}


def _hash(value):
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _token_key(token):
    value = (token or "").strip().lower()
    return value or "__global__"


def _attempt_limits():
    max_attempts = max(1, int(SECOND_FACTOR_MAX_ATTEMPTS_PER_TOKEN or 1))
    lockout_seconds = max(1, int(SECOND_FACTOR_LOCKOUT_SECONDS or 1))
    return max_attempts, lockout_seconds


def clear_second_factor_attempts(token):
    key = _token_key(token)
    with _LOCK:
        _ATTEMPTS.pop(key, None)


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
