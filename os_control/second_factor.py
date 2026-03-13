import hashlib
import hmac

from core.config import SECOND_FACTOR_PASSPHRASE, SECOND_FACTOR_PIN

_PIN_HASH = hashlib.sha256(SECOND_FACTOR_PIN.encode("utf-8")).hexdigest()
_PASSPHRASE_HASH = hashlib.sha256(SECOND_FACTOR_PASSPHRASE.encode("utf-8")).hexdigest()


def _hash(value):
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def verify_second_factor(secret):
    candidate = _hash(secret)
    return hmac.compare_digest(candidate, _PIN_HASH) or hmac.compare_digest(candidate, _PASSPHRASE_HASH)
