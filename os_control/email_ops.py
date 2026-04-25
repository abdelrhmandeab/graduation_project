"""Email draft via Outlook COM — opens compose window, does NOT send."""

from core.logger import logger


def draft_email(to="", subject="", body=""):
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
        parts = ["Email draft opened"]
        if to:
            parts[0] += f" to {to}"
        if subject:
            parts.append(f"subject: {subject}")
        return ". ".join(parts) + "."
    except ImportError:
        logger.warning("pywin32 not installed — Outlook email unavailable")
        return "Email drafting not available (pywin32 not installed)."
    except Exception as exc:
        logger.warning("Could not open Outlook: %s", exc)
        return f"Could not open Outlook: {exc}. Is Outlook installed?"
