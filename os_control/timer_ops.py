"""Timer and alarm system — threading.Timer + winsound.Beep (no external deps)."""

from datetime import datetime, timedelta
import re
import threading
import time

from core.logger import logger

try:
    import winsound

    _WINSOUND_AVAILABLE = True
except ImportError:
    _WINSOUND_AVAILABLE = False

_active_timers = {}  # id -> {thread, label, fires_at}
_lock = threading.Lock()
_ALARM_PREFIX_RE = re.compile(r"^(?:at\s+|time\s+|alarm\s+|for\s+)", re.IGNORECASE)


def _normalize_alarm_time_text(value):
    text = str(value or "").strip().lower()
    if not text:
        return ""

    # Arabic AM/PM shorthands and common phrases.
    text = text.replace("\u0635\u0628\u0627\u062d\u0627", " am")
    text = text.replace("\u0635\u0628\u0627\u062d\u064b\u0627", " am")
    text = text.replace("\u0645\u0633\u0627\u0621", " pm")
    text = text.replace("\u0645\u0633\u0627\u0621\u064b", " pm")
    text = text.replace("\u0635", " am")
    text = text.replace("\u0645", " pm")

    text = text.replace(".", ":")
    text = text.replace("\u2013", "-")
    text = _ALARM_PREFIX_RE.sub("", text)
    text = " ".join(text.split()).strip()
    return text


def _parse_alarm_datetime(alarm_time_text, now=None):
    """Parse alarm time text and return the next matching datetime."""
    now = now or datetime.now()
    text = _normalize_alarm_time_text(alarm_time_text)
    if not text:
        return None

    formats = (
        "%I:%M %p",
        "%I %p",
        "%H:%M",
        "%H",
    )

    parsed = None
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue

    if parsed is None:
        return None

    target = now.replace(
        hour=parsed.hour,
        minute=parsed.minute,
        second=0,
        microsecond=0,
    )
    if target <= now:
        target = target + timedelta(days=1)
    return target


def _speak_timer_alert(label):
    """Announce the timer via Windows SAPI speech (no Jarvis TTS dependency)."""
    try:
        import subprocess

        text = f"{label} is done!" if label and label.lower() != "timer" else "Timer is done!"
        ps_cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Speak('{text}')"
        )
        subprocess.Popen(
            ["powershell", "-NonInteractive", "-Command", ps_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        logger.debug("Timer SAPI speech failed: %s", exc)


def _fire_timer(timer_id, label):
    """Beep and announce via speech when the timer fires."""
    logger.info("Timer fired: %s (%s)", timer_id, label)
    if _WINSOUND_AVAILABLE:
        for _ in range(3):
            try:
                winsound.Beep(1000, 500)
                time.sleep(0.2)
            except Exception:
                break
    _speak_timer_alert(label)
    with _lock:
        _active_timers.pop(timer_id, None)


def set_timer(seconds, label="Timer"):
    """Start a countdown timer. Returns a status message."""
    seconds = max(1, min(86400, int(seconds)))
    timer_id = f"timer_{int(time.time() * 1000)}"

    t = threading.Timer(seconds, _fire_timer, args=(timer_id, label))
    t.daemon = True
    t.start()

    with _lock:
        _active_timers[timer_id] = {
            "thread": t,
            "label": str(label),
            "fires_at": time.time() + seconds,
        }

    if seconds >= 3600:
        hrs, remainder = divmod(seconds, 3600)
        mins = remainder // 60
        human = f"{hrs}h {mins}m" if mins else f"{hrs}h"
    elif seconds >= 60:
        mins, secs = divmod(seconds, 60)
        human = f"{mins}m {secs}s" if secs else f"{mins}m"
    else:
        human = f"{seconds}s"

    logger.info("Timer set: %s for %s (%s)", timer_id, human, label)
    return f"Timer set for {human}."


def set_alarm_at(alarm_time_text, label="Alarm"):
    """Set an alarm for the next occurrence of a wall-clock time."""
    now = datetime.now()
    target = _parse_alarm_datetime(alarm_time_text, now=now)
    if target is None:
        return "Could not parse alarm time. Use formats like '7:30 am', '19:30', or '7 pm'."

    seconds = max(1, int((target - now).total_seconds()))
    timer_status = set_timer(seconds, label=label)
    return f"{timer_status[:-1]} (alarm at {target.strftime('%H:%M')})."


def cancel_timer(timer_id=None):
    """Cancel a specific timer or the most recent one."""
    with _lock:
        if timer_id and timer_id in _active_timers:
            _active_timers[timer_id]["thread"].cancel()
            del _active_timers[timer_id]
            return "Timer cancelled."
        if _active_timers:
            last_id = list(_active_timers.keys())[-1]
            _active_timers[last_id]["thread"].cancel()
            label = _active_timers[last_id]["label"]
            del _active_timers[last_id]
            return f"Timer '{label}' cancelled."
        return "No active timers."


def list_timers():
    """List all active timers with remaining time."""
    with _lock:
        if not _active_timers:
            return "No active timers."
        lines = []
        for tid, info in _active_timers.items():
            remaining = max(0, int(info["fires_at"] - time.time()))
            if remaining >= 60:
                mins, secs = divmod(remaining, 60)
                time_str = f"{mins}m {secs}s"
            else:
                time_str = f"{remaining}s"
            lines.append(f"- {info['label']}: {time_str} remaining")
        return "\n".join(lines)
