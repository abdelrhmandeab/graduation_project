"""Timer and alarm system.

Phase 8 upgrades:
- Persistence: active timers survive Jarvis restarts via %LOCALAPPDATA%/Jarvis/timers.json.
- Named timers: label-based cancel ("cancel the pasta timer").
- TTS fire: uses speech_engine.speak_async when TIMER_FIRE_USE_TTS is True.
- Clock app opt-in: opens ms-clock: when TIMER_OPEN_CLOCK_APP is True.
- winsound.Beep kept as fallback when TTS unavailable.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from core.logger import logger

try:
    from core.config import (
        TIMER_PERSISTENCE_ENABLED,
        TIMER_OPEN_CLOCK_APP,
        TIMER_FIRE_USE_TTS,
    )
except ImportError:
    TIMER_PERSISTENCE_ENABLED = True
    TIMER_OPEN_CLOCK_APP = False
    TIMER_FIRE_USE_TTS = True

try:
    import winsound
    _WINSOUND_AVAILABLE = True
except ImportError:
    _WINSOUND_AVAILABLE = False


# ── persistence file ──────────────────────────────────────────────────────────

def _persistence_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    d = base / "Jarvis"
    d.mkdir(parents=True, exist_ok=True)
    return d / "timers.json"


# ── in-memory state ───────────────────────────────────────────────────────────

_active_timers: dict[str, dict] = {}  # id -> {thread, label, fires_at}
_lock = threading.Lock()


# ── normalization helpers ─────────────────────────────────────────────────────

_ALARM_PREFIX_RE = re.compile(r"^(?:at\s+|time\s+|alarm\s+|for\s+)", re.IGNORECASE)


def _normalize_alarm_time_text(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    text = text.replace("صباحا", " am").replace("صباحًا", " am")
    text = text.replace("مساء", " pm").replace("مساءً", " pm")
    text = text.replace("ص", " am").replace("م", " pm")
    text = text.replace(".", ":").replace("–", "-")
    text = _ALARM_PREFIX_RE.sub("", text)
    return " ".join(text.split()).strip()


def _parse_alarm_datetime(alarm_time_text: str, now: Optional[datetime] = None) -> Optional[datetime]:
    now = now or datetime.now()
    text = _normalize_alarm_time_text(alarm_time_text)
    if not text:
        return None
    for fmt in ("%I:%M %p", "%I %p", "%H:%M", "%H"):
        try:
            parsed = datetime.strptime(text, fmt)
            target = now.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return target
        except ValueError:
            continue
    return None


def _seconds_to_human(seconds: int) -> str:
    if seconds >= 3600:
        hrs, remainder = divmod(seconds, 3600)
        mins = remainder // 60
        return f"{hrs}h {mins}m" if mins else f"{hrs}h"
    if seconds >= 60:
        mins, secs = divmod(seconds, 60)
        return f"{mins}m {secs}s" if secs else f"{mins}m"
    return f"{seconds}s"


def _seconds_to_human_ar(seconds: int) -> str:
    if seconds >= 3600:
        hrs, remainder = divmod(seconds, 3600)
        mins = remainder // 60
        hrs_str = f"{hrs} ساعة" if hrs == 1 else f"{hrs} ساعات"
        if mins:
            mins_str = "دقيقة" if mins == 1 else f"{mins} دقيقة"
            return f"{hrs_str} و{mins_str}"
        return hrs_str
    if seconds >= 60:
        mins, secs = divmod(seconds, 60)
        if mins == 1:
            mins_str = "دقيقة"
        elif mins == 2:
            mins_str = "دقيقتين"
        else:
            mins_str = f"{mins} دقيقة"
        if secs:
            return f"{mins_str} و{secs} ثانية"
        return mins_str
    if seconds == 1:
        return "ثانية"
    if seconds == 2:
        return "ثانيتين"
    return f"{seconds} ثانية"


# ── persistence ───────────────────────────────────────────────────────────────

def _save_timers() -> None:
    if not TIMER_PERSISTENCE_ENABLED:
        return
    try:
        path = _persistence_path()
        with _lock:
            data = {
                tid: {"label": info["label"], "fires_at": info["fires_at"]}
                for tid, info in _active_timers.items()
            }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Timer persist save failed: %s", exc)


def _load_and_rearm_timers() -> None:
    """Called at module init. Re-arms any timers whose fires_at is still in the future."""
    if not TIMER_PERSISTENCE_ENABLED:
        return
    try:
        path = _persistence_path()
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        now = time.time()
        rearmed = 0
        for tid, info in data.items():
            fires_at = float(info.get("fires_at", 0))
            label = str(info.get("label", "Timer"))
            remaining = fires_at - now
            if remaining > 0:
                _rearm_timer(tid, label, remaining, fires_at)
                rearmed += 1
        if rearmed:
            logger.info("Re-armed %d timer(s) from persistence.", rearmed)
        # Overwrite with only still-live timers
        _save_timers()
    except Exception as exc:
        logger.debug("Timer persist load failed: %s", exc)


def _rearm_timer(timer_id: str, label: str, delay_seconds: float, fires_at: float) -> None:
    t = threading.Timer(delay_seconds, _fire_timer, args=(timer_id, label))
    t.daemon = True
    t.start()
    with _lock:
        _active_timers[timer_id] = {
            "thread": t,
            "label": label,
            "fires_at": fires_at,
        }


# ── TTS / beep on fire ────────────────────────────────────────────────────────

def _announce_timer(label: str) -> None:
    text_en = f"{label} timer is done!" if label.lower() not in ("timer", "alarm") else "Timer done!"
    text_ar = f"خلص مؤقت {label}!" if label.lower() not in ("timer", "alarm") else "خلص الوقت!"

    if TIMER_FIRE_USE_TTS:
        try:
            from audio.tts import speech_engine
            # Detect a rough language hint from the label
            has_ar = bool(re.search(r"[؀-ۿ]", label))
            text = text_ar if has_ar else text_en
            speech_engine.speak_async(text)
            return
        except Exception as exc:
            logger.debug("TTS timer announce failed, falling back to SAPI: %s", exc)

    # SAPI fallback
    try:
        ps_cmd = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Speak('{text_en}')"
        )
        subprocess.Popen(
            ["powershell", "-NonInteractive", "-Command", ps_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        logger.debug("Timer SAPI fallback failed: %s", exc)


def _fire_timer(timer_id: str, label: str) -> None:
    logger.info("Timer fired: %s (%s)", timer_id, label)

    if _WINSOUND_AVAILABLE:
        for _ in range(3):
            try:
                winsound.Beep(1000, 500)
                time.sleep(0.2)
            except Exception:
                break

    _announce_timer(label)

    with _lock:
        _active_timers.pop(timer_id, None)
    _save_timers()


# ── public API ────────────────────────────────────────────────────────────────

def set_timer(seconds: int, label: str = "Timer", language: str = "en") -> str:
    """Start a countdown timer. Returns a human-readable status message."""
    seconds = max(1, min(86400, int(seconds)))
    label = str(label or "Timer").strip() or "Timer"
    timer_id = f"timer_{int(time.time() * 1000)}_{id(threading.current_thread())}"
    fires_at = time.time() + seconds

    t = threading.Timer(seconds, _fire_timer, args=(timer_id, label))
    t.daemon = True
    t.start()

    with _lock:
        _active_timers[timer_id] = {
            "thread": t,
            "label": label,
            "fires_at": fires_at,
        }
    _save_timers()

    human = _seconds_to_human(seconds)
    logger.info("Timer set: %s for %s (%s)", timer_id, human, label)
    try:
        from core.logger import log_structured
        log_structured("timer_set", seconds=seconds, human=human, label=label)
    except Exception:
        pass

    is_ar = str(language or "").startswith("ar")
    is_generic = label.lower() in ("timer", "alarm")
    if is_ar:
        human_ar = _seconds_to_human_ar(seconds)
        if is_generic:
            return f"تمام، التايمر اتضبط على {human_ar}."
        return f"تمام، تايمر '{label}' اتضبط على {human_ar}."
    if is_generic:
        return f"Timer set for {human}."
    return f"Timer '{label}' set for {human}."


def set_alarm_at(alarm_time_text: str, label: str = "Alarm", language: str = "en") -> str:
    """Set an alarm for the next occurrence of a wall-clock time."""
    now = datetime.now()
    target = _parse_alarm_datetime(alarm_time_text, now=now)
    is_ar = str(language or "").startswith("ar")
    if target is None:
        return "مش قادر أفهم الوقت ده." if is_ar else "Could not parse alarm time. Use formats like '7:30 am', '19:30', or '7 pm'."

    if TIMER_OPEN_CLOCK_APP:
        try:
            subprocess.Popen(["cmd", "/c", "start", "", "ms-clock:"], shell=False)
        except Exception:
            pass

    seconds = max(1, int((target - now).total_seconds()))
    msg = set_timer(seconds, label=label, language=language)
    time_str = target.strftime('%H:%M')
    if str(language or "").startswith("ar"):
        return f"{msg[:-1]} (الساعة {time_str})."
    return f"{msg[:-1]} (alarm at {time_str})."


def cancel_timer(timer_id: Optional[str] = None, label: Optional[str] = None, language: str = "en") -> str:
    """Cancel a specific timer by id, label, or the most recent one."""
    is_ar = str(language or "").startswith("ar")
    with _lock:
        # Label-based cancel (case-insensitive partial match)
        if label:
            label_lower = label.strip().lower()
            matched = [
                tid for tid, info in _active_timers.items()
                if label_lower in info["label"].lower()
            ]
            if matched:
                tid = matched[-1]
                _active_timers[tid]["thread"].cancel()
                found_label = _active_timers[tid]["label"]
                del _active_timers[tid]
                _save_timers()
                return f"تايمر '{found_label}' اتلغى." if is_ar else f"Timer '{found_label}' cancelled."
            return f"مفيش تايمر اسمه '{label}'." if is_ar else f"No timer found matching '{label}'."

        # Explicit id cancel
        if timer_id and timer_id in _active_timers:
            _active_timers[timer_id]["thread"].cancel()
            del _active_timers[timer_id]
            _save_timers()
            return "التايمر اتلغى." if is_ar else "Timer cancelled."

        # Most recent timer
        if _active_timers:
            last_id = list(_active_timers.keys())[-1]
            _active_timers[last_id]["thread"].cancel()
            found_label = _active_timers[last_id]["label"]
            del _active_timers[last_id]
            _save_timers()
            return f"تايمر '{found_label}' اتلغى." if is_ar else f"Timer '{found_label}' cancelled."

        return "مفيش تايمرات شغالة." if is_ar else "No active timers."


def list_timers(language: str = "en") -> str:
    """List all active timers with remaining time."""
    is_ar = str(language or "").startswith("ar")
    with _lock:
        if not _active_timers:
            return "مفيش تايمرات شغالة." if is_ar else "No active timers."
        lines = []
        for info in _active_timers.values():
            remaining = max(0, int(info["fires_at"] - time.time()))
            if is_ar:
                lines.append(f"- {info['label']}: باقي {_seconds_to_human_ar(remaining)}")
            else:
                lines.append(f"- {info['label']}: {_seconds_to_human(remaining)} remaining")
        return "\n".join(lines)


# ── startup reload ────────────────────────────────────────────────────────────
# Re-arm timers from the last session when this module is first imported.
_load_and_rearm_timers()
