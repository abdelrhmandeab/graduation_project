"""Clipboard read/write/clear — uses pyperclip with graceful fallback."""

from core.logger import logger

try:
    import pyperclip

    _CLIPBOARD_AVAILABLE = True
except ImportError:
    _CLIPBOARD_AVAILABLE = False


def read_clipboard():
    """Read current clipboard text. Returns content or status message."""
    if not _CLIPBOARD_AVAILABLE:
        logger.warning("pyperclip not installed — clipboard unavailable")
        return "Clipboard access not available (pyperclip not installed)."
    try:
        text = pyperclip.paste()
        if not text:
            return "Clipboard is empty."
        if len(text) > 500:
            return f"Clipboard ({len(text)} chars):\n{text[:500]}..."
        return f"Clipboard:\n{text}"
    except Exception as exc:
        logger.warning("Clipboard read failed: %s", exc)
        return f"Could not read clipboard: {exc}"


def write_clipboard(text):
    """Copy text to clipboard. Returns status message."""
    if not _CLIPBOARD_AVAILABLE:
        logger.warning("pyperclip not installed — clipboard unavailable")
        return "Clipboard access not available (pyperclip not installed)."
    try:
        pyperclip.copy(str(text or ""))
        return f"Copied to clipboard ({len(text)} chars)."
    except Exception as exc:
        logger.warning("Clipboard write failed: %s", exc)
        return f"Could not write to clipboard: {exc}"


def clear_clipboard():
    """Clear the clipboard. Returns status message."""
    if not _CLIPBOARD_AVAILABLE:
        logger.warning("pyperclip not installed — clipboard unavailable")
        return "Clipboard access not available (pyperclip not installed)."
    try:
        pyperclip.copy("")
        return "Clipboard cleared."
    except Exception as exc:
        logger.warning("Clipboard clear failed: %s", exc)
        return f"Could not clear clipboard: {exc}"
