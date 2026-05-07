"""Windows reminder system — Task Scheduler (pywin32) primary, threading.Timer fallback.

create_reminder(message, time_str, language)  →  schedule a one-shot reminder
list_reminders(language)                      →  list active Jarvis reminders
cancel_reminder(reminder_id, language)        →  cancel by ID or most recent

Time-string formats understood by _parse_trigger_time:
  English : "at 3pm", "at 3:30 pm", "at 15:00", "in 2 hours", "in 30 minutes",
            "tomorrow at 9", "tomorrow at 9am"
  Arabic  : "الساعة ٣", "الساعة ٣ مساءً", "الساعه ٩ صبح",
            "بعد ساعتين", "بعد ٣٠ دقيقة", "بعد نص ساعة",
            "بكرة الساعة ٩", "بكره الساعة ٩"
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

from core.logger import logger

try:
    import win32com.client
    _WIN32_AVAILABLE = True
except ImportError:
    _WIN32_AVAILABLE = False

try:
    import winsound
    _WINSOUND_AVAILABLE = True
except ImportError:
    _WINSOUND_AVAILABLE = False

_TASK_PREFIX = "Jarvis_Reminder_"
_in_process: dict = {}   # task_name → {timer, message, fires_at}
_lock = threading.Lock()

# Arabic-Indic → ASCII
_AR_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _norm(text: str) -> str:
    return str(text or "").translate(_AR_INDIC).strip().lower()


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

_REL_RE = re.compile(
    r"(?:in|بعد)\s+"
    r"((?:\d+(?:\.\d+)?)|نص|ربع|ساعتين)\s*"
    r"(h(?:ours?|rs?)?|m(?:in(?:utes?|s)?)?|s(?:ec(?:onds?)?)?|"
    r"ساعة|ساعات|ساعه|دقيقة|دقائق|دقايق|ثانية|ثواني)?",
    re.IGNORECASE,
)
_TOMORROW_RE = re.compile(r"tomorrow|بكرة|بكره|بكرا", re.IGNORECASE)
_AR_CLOCK_RE = re.compile(
    r"(?:الساعة|الساعه|ساعه?)\s*(\d+)(?:[:.،,](\d+))?\s*"
    r"(صباحاً|صباحا|صبح|ص|مساءً|مساءا|مساء|م)?",
    re.IGNORECASE,
)
_EN_CLOCK_RE = re.compile(
    r"(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?(?!\d)",
    re.IGNORECASE,
)


def _parse_trigger_time(time_str: str) -> Optional[datetime]:
    """Return a future datetime for the given natural-language time string."""
    now = datetime.now()
    text = _norm(time_str)
    if not text:
        return None

    # --- relative: "in N unit" / "بعد N unit" ---
    rel = _REL_RE.search(text)
    if rel:
        qty_str = rel.group(1) or ""
        unit_str = (rel.group(2) or "").strip().lower()
        if qty_str == "نص":
            qty = 0.5
        elif qty_str == "ربع":
            qty = 0.25
        elif qty_str == "ساعتين":
            qty, unit_str = 2.0, "h"
        else:
            try:
                qty = float(qty_str)
            except ValueError:
                qty = 1.0
        if unit_str in ("h", "hr", "hrs", "hour", "hours", "ساعة", "ساعات", "ساعه"):
            return now + timedelta(hours=qty)
        if unit_str in ("m", "min", "mins", "minute", "minutes", "دقيقة", "دقائق", "دقايق"):
            return now + timedelta(minutes=qty)
        return now + timedelta(seconds=qty)

    # --- tomorrow? ---
    is_tomorrow = bool(_TOMORROW_RE.search(text))
    stripped = _TOMORROW_RE.sub("", text).strip()

    # --- wall-clock time ---
    hour = minute = None
    ampm: Optional[str] = None

    ar = _AR_CLOCK_RE.search(stripped)
    if ar:
        hour = int(ar.group(1))
        minute = int(ar.group(2)) if ar.group(2) else 0
        period = (ar.group(3) or "").strip().lower()
        if period in ("صباحاً", "صباحا", "صبح", "ص"):
            ampm = "am"
        elif period in ("مساءً", "مساءا", "مساء", "م"):
            ampm = "pm"
    else:
        en = _EN_CLOCK_RE.search(stripped)
        if en:
            hour = int(en.group(1))
            minute = int(en.group(2)) if en.group(2) else 0
            ampm = (en.group(3) or "").strip().lower() or None

    if hour is None:
        return None

    # Resolve AM/PM
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    elif ampm is None and 1 <= hour <= 6:
        # Heuristic: ambiguous small hours lean PM in everyday speech
        hour += 12

    hour = hour % 24
    base = now.date() + timedelta(days=1) if is_tomorrow else now.date()
    target = datetime(base.year, base.month, base.day, hour, minute, 0)
    if not is_tomorrow and target <= now:
        target += timedelta(days=1)
    return target


# ---------------------------------------------------------------------------
# Toast notification
# ---------------------------------------------------------------------------

_TOAST_PS = (
    "[Windows.UI.Notifications.ToastNotificationManager,"
    " Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null;"
    " [Windows.Data.Xml.Dom.XmlDocument,"
    " Windows.Data.Xml.Dom.XmlDocument, ContentType = WindowsRuntime] | Out-Null;"
    " $t = [Windows.UI.Notifications.ToastTemplateType]::ToastText02;"
    " $x = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent($t);"
    " $n = $x.GetElementsByTagName('text');"
    " $n.Item(0).AppendChild($x.CreateTextNode('Jarvis Reminder')) | Out-Null;"
    " $n.Item(1).AppendChild($x.CreateTextNode('{MSG}')) | Out-Null;"
    " $toast = [Windows.UI.Notifications.ToastNotification]::new($x);"
    " [Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Jarvis').Show($toast)"
)


def _show_toast(message: str) -> None:
    safe = message.replace("'", "\\'").replace('"', '\\"')
    ps = _TOAST_PS.replace("{MSG}", safe)
    try:
        subprocess.Popen(
            ["powershell", "-NonInteractive", "-WindowStyle", "Hidden", "-Command", ps],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        logger.warning("Reminder toast failed: %s", exc)


# ---------------------------------------------------------------------------
# Fire callback
# ---------------------------------------------------------------------------

def _fire(task_name: str, message: str) -> None:
    logger.info("Reminder fired: %s — %s", task_name, message)
    _show_toast(message)
    if _WINSOUND_AVAILABLE:
        try:
            for _ in range(2):
                winsound.Beep(880, 400)
                time.sleep(0.2)
        except Exception:
            pass
    with _lock:
        _in_process.pop(task_name, None)


# ---------------------------------------------------------------------------
# Task Scheduler (pywin32) helpers
# ---------------------------------------------------------------------------

def _scheduler_create(task_name: str, trigger_time: datetime, message: str) -> bool:
    if not _WIN32_AVAILABLE:
        return False
    try:
        svc = win32com.client.Dispatch("Schedule.Service")
        svc.Connect()
        folder = svc.GetFolder("\\")
        defn = svc.NewTask(0)
        defn.RegistrationInfo.Description = f"Jarvis Reminder: {message}"
        defn.Settings.Enabled = True
        defn.Settings.StartWhenAvailable = True
        defn.Settings.DeleteExpiredTaskAfter = "PT0S"

        trig = defn.Triggers.Create(1)  # TASK_TRIGGER_TIME
        trig.StartBoundary = trigger_time.strftime("%Y-%m-%dT%H:%M:%S")
        trig.Enabled = True
        trig.ExecutionTimeLimit = "PT5M"
        trig.EndBoundary = (trigger_time + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S")

        safe = message.replace('"', '\\"').replace("'", "\\'")
        ps = _TOAST_PS.replace("{MSG}", safe)
        action = defn.Actions.Create(0)  # TASK_ACTION_EXEC
        action.Path = "powershell.exe"
        action.Arguments = f'-NonInteractive -WindowStyle Hidden -Command "{ps}"'

        principal = defn.Principal
        principal.LogonType = 3   # TASK_LOGON_INTERACTIVE_TOKEN
        principal.RunLevel = 0    # TASK_RUNLEVEL_LUA

        folder.RegisterTaskDefinition(task_name, defn, 6, "", "", 3)
        logger.info("Task Scheduler reminder created: %s at %s", task_name, trigger_time)
        return True
    except Exception as exc:
        logger.warning("Task Scheduler create failed: %s", exc)
        return False


def _scheduler_delete(task_name: str) -> bool:
    if not _WIN32_AVAILABLE:
        return False
    try:
        svc = win32com.client.Dispatch("Schedule.Service")
        svc.Connect()
        svc.GetFolder("\\").DeleteTask(task_name, 0)
        return True
    except Exception as exc:
        logger.debug("Task Scheduler delete '%s' failed: %s", task_name, exc)
        return False


def _scheduler_list() -> list:
    """Return list of (name, trigger_iso, description) for Jarvis reminders."""
    if not _WIN32_AVAILABLE:
        return []
    try:
        svc = win32com.client.Dispatch("Schedule.Service")
        svc.Connect()
        tasks = svc.GetFolder("\\").GetTasks(0)
        results = []
        for task in tasks:
            if not task.Name.startswith(_TASK_PREFIX):
                continue
            try:
                d = task.Definition
                trigger_iso = d.Triggers.Item(1).StartBoundary if d.Triggers.Count >= 1 else ""
                desc = d.RegistrationInfo.Description or ""
            except Exception:
                trigger_iso, desc = "", ""
            results.append((task.Name, trigger_iso, desc))
        return results
    except Exception as exc:
        logger.debug("Task Scheduler list failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_reminder(message: str, time_str: str, language: str = "en") -> str:
    """Schedule a reminder. Uses Task Scheduler if pywin32 is available."""
    if not (message and message.strip()):
        return ("محتاج تقولي تذكرني بإيه." if language == "ar"
                else "Please specify what to remind you about.")
    if not (time_str and time_str.strip()):
        return ("محتاج تقولي الوقت." if language == "ar"
                else "Please specify when to set the reminder.")

    trigger = _parse_trigger_time(time_str)
    if trigger is None:
        return (f"ما عرفتش أفسر الوقت: «{time_str}»." if language == "ar"
                else f"Could not parse time: '{time_str}'.")

    if trigger <= datetime.now():
        return ("الوقت ده فات." if language == "ar" else "That time has already passed.")

    task_name = f"{_TASK_PREFIX}{str(uuid.uuid4())[:8]}"
    time_label = trigger.strftime("%H:%M")

    if _scheduler_create(task_name, trigger, message):
        return (f"تمام، هفكرك «{message}» الساعة {time_label}." if language == "ar"
                else f"Reminder set for {time_label}: '{message}'.")

    # Fallback: in-process threading.Timer (won't survive app restart)
    seconds = max(1, int((trigger - datetime.now()).total_seconds()))
    t = threading.Timer(seconds, _fire, args=(task_name, message))
    t.daemon = True
    t.start()
    with _lock:
        _in_process[task_name] = {"timer": t, "message": message, "fires_at": time.time() + seconds}
    logger.info("In-process reminder: %s in %ss", message, seconds)
    note = " (تنبيه: ما يبقاش بعد ما التطبيق يقفل)" if language == "ar" else " (won't survive app restart)"
    return (f"تمام، هفكرك «{message}» الساعة {time_label}.{note}" if language == "ar"
            else f"Reminder set for {time_label}: '{message}'.{note}")


def list_reminders(language: str = "en") -> str:
    """List all active Jarvis reminders."""
    lines = []

    for name, trigger_iso, desc in _scheduler_list():
        short_id = name.replace(_TASK_PREFIX, "")
        time_display = trigger_iso[:16].replace("T", " ") if trigger_iso else "?"
        msg = desc.replace("Jarvis Reminder: ", "")
        lines.append(f"[{short_id}] {time_display} — {msg}")

    with _lock:
        now = time.time()
        for tid, info in list(_in_process.items()):
            remaining = max(0, int(info["fires_at"] - now))
            time_display = f"in {remaining // 60}m {remaining % 60}s" if remaining >= 60 else f"in {remaining}s"
            short_id = tid.replace(_TASK_PREFIX, "")
            lines.append(f"[{short_id}] {time_display} — {info['message']} *")

    if not lines:
        return "مفيش تذكيرات نشطة." if language == "ar" else "No active reminders."
    header = "التذكيرات النشطة:" if language == "ar" else "Active reminders:"
    return header + "\n" + "\n".join(lines)


def cancel_reminder(reminder_id: str = "", language: str = "en") -> str:
    """Cancel a reminder by short ID, or the most recent one if no ID given."""
    cancelled_msg = "تم إلغاء التذكير." if language == "ar" else "Reminder cancelled."
    nothing_msg = "مفيش تذكيرات لإلغائها." if language == "ar" else "No active reminders to cancel."

    full_id = (f"{_TASK_PREFIX}{reminder_id}" if reminder_id and not reminder_id.startswith(_TASK_PREFIX)
               else reminder_id)

    # Check in-process first
    with _lock:
        if full_id and full_id in _in_process:
            _in_process[full_id]["timer"].cancel()
            del _in_process[full_id]
            return cancelled_msg
        if not full_id and _in_process:
            last = list(_in_process.keys())[-1]
            _in_process[last]["timer"].cancel()
            del _in_process[last]
            return cancelled_msg

    # Check Task Scheduler
    tasks = _scheduler_list()
    if full_id:
        if _scheduler_delete(full_id):
            return cancelled_msg
    elif tasks:
        if _scheduler_delete(tasks[-1][0]):
            return cancelled_msg

    return nothing_msg
