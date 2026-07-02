"""Email draft via Outlook COM — opens compose window, does NOT send."""

import subprocess
import time
import webbrowser

from core.logger import logger


def _is_arabic_language(language):
    return str(language or "").lower().startswith("ar")


def _try_outlook_com(to: str, subject: str, body: str) -> bool:
    """Attempt to open Outlook compose via COM. Returns True on success."""
    import win32com.client

    outlook = win32com.client.Dispatch("Outlook.Application")
    mail = outlook.CreateItem(0)  # olMailItem
    if to:
        mail.To = str(to)
    if subject:
        mail.Subject = str(subject)
    if body:
        mail.Body = str(body)
    mail.Display()
    return True


def draft_email(to="", subject="", body="", language=None):
    """Open a pre-filled Outlook compose window. Safe by design — never sends.

    Strategy: COM first → subprocess launch + retry → Gmail fallback.
    Returns a status message string.
    """
    ar = _is_arabic_language(language)

    try:
        from core.config import EMAIL_PREFER_OUTLOOK
        prefer_outlook = EMAIL_PREFER_OUTLOOK
    except Exception:
        prefer_outlook = True

    if not prefer_outlook:
        # User explicitly disabled Outlook — go straight to Gmail.
        try:
            webbrowser.open("https://mail.google.com/mail/u/0/#compose")
            try:
                from core.logger import log_structured
                log_structured("email_compose_opened", app="Gmail", success=True, language=language)
            except Exception:
                pass
            return "فاتح Gmail..." if ar else "Opening Gmail..."
        except Exception as exc:
            logger.warning("Could not open Gmail: %s", exc)
            return "البريد غير متاح." if ar else "Email unavailable."

    # 1. Try COM directly (Outlook already running).
    try:
        _try_outlook_com(to, subject, body)
        parts = ["فاتح Outlook" if ar else "Opening Outlook"]
        if to:
            parts[0] += f" to {to}"
        if subject:
            parts.append(("الموضوع" if ar else "subject") + f": {subject}")
        try:
            from core.logger import log_structured
            log_structured("email_compose_opened", app="Outlook", success=True, language=language)
        except Exception:
            pass
        return ". ".join(parts) + "."
    except ImportError:
        logger.warning("pywin32 not installed — Outlook COM unavailable")
    except Exception as exc:
        logger.debug("Outlook COM first attempt failed (%s); trying subprocess launch", exc)

    # 2. Launch Outlook via subprocess, then retry COM once after a short wait.
    try:
        import win32com.client  # noqa: F401 — required for retry below

        try:
            subprocess.Popen(["outlook.exe"], shell=True)
        except Exception:
            pass
        time.sleep(1.8)
        try:
            _try_outlook_com(to, subject, body)
            parts = ["فاتح Outlook" if ar else "Opening Outlook"]
            if to:
                parts[0] += f" to {to}"
            if subject:
                parts.append(("الموضوع" if ar else "subject") + f": {subject}")
            try:
                from core.logger import log_structured
                log_structured("email_compose_opened", app="Outlook", success=True, language=language)
            except Exception:
                pass
            return ". ".join(parts) + "."
        except Exception as exc2:
            logger.warning("Outlook COM retry also failed: %s", exc2)
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Outlook subprocess launch failed: %s", exc)

    # 3. Gmail fallback.
    try:
        webbrowser.open("https://mail.google.com/mail/u/0/#compose")
        try:
            from core.logger import log_structured
            log_structured("email_compose_opened", app="Gmail", success=True, language=language)
        except Exception:
            pass
        return "Outlook مش متاح، فاتح Gmail..." if ar else "Outlook not available, opening Gmail..."
    except Exception as exc:
        logger.warning("Could not open Gmail fallback: %s", exc)
        return "البريد غير متاح." if ar else "Email unavailable."
