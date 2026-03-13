import threading

_lock = threading.Lock()
_enabled = False


def is_enabled():
    with _lock:
        return _enabled


def set_enabled(value):
    global _enabled
    with _lock:
        _enabled = bool(value)
        return _enabled
