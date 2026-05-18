"""Email draft via Outlook COM — opens compose window, does NOT send."""

import webbrowser

from core.logger import logger


def _is_arabic_language(language):
    return str(language or "").lower().startswith("ar")


def draft_email(to="", subject="", body="", language=None):
    """Open a pre-filled Outlook compose window. Safe by design — never sends.

    Returns a status message string.
    """
    try:
        import win32com.client

        outlook = win32com.client.Dispatch("Outlook.Application")
        mail = outlook.CreateItem(0)  # olMailItem
        if to:
            mail.To = str(to)
        if subject:
            mail.Subject = str(subject)
        if body:
            mail.Body = str(body)
        mail.Display()  # Opens compose window, does NOT send
        parts = ["فاتح Outlook" if _is_arabic_language(language) else "Opening Outlook"]
        if to:
            parts[0] += f" to {to}"
        if subject:
            parts.append(("الموضوع" if _is_arabic_language(language) else "subject") + f": {subject}")
        try:
            from core.metrics import log_structured

            log_structured("email_compose_opened", app="Outlook", success=True, language=language)
        except Exception:
            pass
        return ". ".join(parts) + "."
    except ImportError:
        logger.warning("pywin32 not installed — Outlook email unavailable")
    except Exception as exc:
        logger.warning("Could not open Outlook: %s", exc)

    try:
        webbrowser.open("https://mail.google.com/mail/u/0/#compose")
        try:
            from core.metrics import log_structured

            log_structured("email_compose_opened", app="Gmail", success=True, language=language)
        except Exception:
            pass
        return "Outlook مش متاح، فاتح Gmail..." if _is_arabic_language(language) else "Outlook not available, opening Gmail..."
    except Exception as exc:
        logger.warning("Could not open Gmail fallback: %s", exc)
        return "Email unavailable." if not _is_arabic_language(language) else "البريد غير متاح."
