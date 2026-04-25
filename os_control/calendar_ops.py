"""Calendar event creation via Outlook COM — opens event window, user confirms."""

from core.logger import logger


def create_calendar_event(subject, start_time, duration_minutes=60):
    """Open an Outlook calendar event window. User must confirm/save manually.

    Args:
        subject: Event title.
        start_time: Start time string, e.g. "2026-04-24 15:00".
        duration_minutes: Duration in minutes (default 60).

    Returns a status message string.
    """
    try:
        import win32com.client

        outlook = win32com.client.Dispatch("Outlook.Application")
        appt = outlook.CreateItem(1)  # olAppointmentItem
        appt.Subject = str(subject or "New Event")
        appt.Start = str(start_time)
        appt.Duration = int(duration_minutes)
        appt.ReminderSet = True
        appt.ReminderMinutesBeforeStart = 15
        appt.Display()  # Opens event window, doesn't save until user confirms
        return f"Calendar event '{subject}' created for {start_time}."
    except ImportError:
        logger.warning("pywin32 not installed — Outlook calendar unavailable")
        return "Calendar events not available (pywin32 not installed)."
    except Exception as exc:
        logger.warning("Could not create calendar event: %s", exc)
        return f"Could not create event: {exc}. Is Outlook installed?"
