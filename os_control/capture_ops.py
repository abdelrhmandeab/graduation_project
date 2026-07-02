"""Screen capture operations: screenshot and screen recording.

Screenshot: delegates to native_ops.capture_primary_screen_screenshot (ctypes-first).
Screen recording: ffmpeg via gdigrab if available; Game Bar (Win+Alt+R) as fallback.

All paths are saved under user-configured dirs (SCREENSHOT_DIR / SCREENRECORD_DIR).
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from core.config import (
    SCREENSHOT_DIR,
    SCREENRECORD_DIR,
    SCREENRECORD_BACKEND,
    SCREENRECORD_FPS,
)
from core.logger import get_logger

logger = get_logger("oscontrol")

# ── module-level recording state ──────────────────────────────────────────────
_recording_proc: Optional[subprocess.Popen] = None
_recording_path: Optional[str] = None
_recording_lock = threading.Lock()

_IS_WINDOWS = sys.platform == "win32"


# ── helpers ───────────────────────────────────────────────────────────────────

def _expanded(path_str: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path_str)))


def _screenshot_dir() -> Path:
    p = _expanded(SCREENSHOT_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _screenrecord_dir() -> Path:
    p = _expanded(SCREENRECORD_DIR)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ffmpeg_on_path() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            timeout=3,
        )
        return result.returncode == 0
    except Exception:
        return False


def _send_hotkey_gamebar():
    """Win+Alt+R — toggle Game Bar recording."""
    if not _IS_WINDOWS:
        return False
    try:
        import ctypes
        KEYEVENTF_KEYUP = 0x0002
        VK_LWIN  = 0x5B
        VK_MENU  = 0x12
        VK_R     = 0x52
        u32 = ctypes.windll.user32
        for vk in (VK_LWIN, VK_MENU, VK_R):
            u32.keybd_event(vk, 0, 0, 0)
        for vk in reversed((VK_LWIN, VK_MENU, VK_R)):
            u32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
        return True
    except Exception as exc:
        logger.debug("Game Bar hotkey failed: %s", exc)
        return False


def _humanize_path(path: str, language: str = "en") -> str:
    """Return a short spoken description of the save location."""
    try:
        from os_control.path_resolver import humanize_path as _hp
        result = _hp(Path(path))
        return result.get(language, result.get("en", path))
    except Exception:
        pass
    # Fallback: just the filename + parent folder name
    p = Path(path)
    try:
        folder = p.parent.name
        if language == "ar":
            return f"الملف {p.name} في مجلد {folder}."
        return f"{p.name} in {folder}."
    except Exception:
        return str(path)


# ── public API ────────────────────────────────────────────────────────────────

def take_screenshot(language: str = "en") -> tuple[bool, str]:
    """Capture the primary screen.

    Returns (success, spoken_message).
    Saves to SCREENSHOT_DIR.
    """
    try:
        from os_control.native_ops import capture_primary_screen_screenshot
        path = capture_primary_screen_screenshot(output_dir=str(_screenshot_dir()))
        if path:
            location = _humanize_path(path, language)
            if language == "ar":
                msg = f"اتحفظت الصورة في {location}"
            else:
                msg = f"Screenshot saved. {location}"
            logger.info("Screenshot saved: %s", path)
            return True, msg
    except Exception as exc:
        logger.debug("take_screenshot failed: %s", exc)
    return False, ("فشل التصوير." if language == "ar" else "Screenshot failed.")


def is_recording() -> bool:
    """Return True if a screen recording is currently in progress."""
    with _recording_lock:
        if _recording_proc is not None and _recording_proc.poll() is None:
            return True
        return False


def start_recording(language: str = "en") -> tuple[bool, str]:
    """Start screen recording.

    Tries ffmpeg gdigrab first (if backend != 'gamebar' and ffmpeg on PATH).
    Falls back to Game Bar Win+Alt+R hotkey.

    Returns (success, spoken_message).
    """
    global _recording_proc, _recording_path

    with _recording_lock:
        if _recording_proc is not None and _recording_proc.poll() is None:
            msg = "التسجيل شغال خلاص." if language == "ar" else "Recording is already in progress."
            return False, msg

        out_path = str(_screenrecord_dir() / f"jarvis_rec_{_ts()}.mp4")

        use_ffmpeg = SCREENRECORD_BACKEND in ("auto", "ffmpeg") and _ffmpeg_on_path()

        if use_ffmpeg:
            try:
                cmd = [
                    "ffmpeg",
                    "-f", "gdigrab",
                    "-framerate", str(SCREENRECORD_FPS),
                    "-i", "desktop",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "23",
                    "-y",
                    out_path,
                ]
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _recording_proc = proc
                _recording_path = out_path
                logger.info("Screen recording started (ffmpeg): %s", out_path)
                msg = "بدأت التسجيل." if language == "ar" else "Screen recording started."
                return True, msg
            except Exception as exc:
                logger.debug("ffmpeg recording failed: %s", exc)

        # Game Bar fallback
        if _send_hotkey_gamebar():
            _recording_proc = None
            _recording_path = "gamebar"
            logger.info("Screen recording started (Game Bar)")
            msg = "شغّلت تسجيل Game Bar." if language == "ar" else "Game Bar recording started."
            return True, msg

        msg = "مش قادر أبدأ التسجيل." if language == "ar" else "Could not start recording. Make sure ffmpeg is installed or Game Bar is enabled."
        return False, msg


def stop_recording(language: str = "en") -> tuple[bool, str]:
    """Stop an active screen recording.

    For ffmpeg: sends 'q' to stdin gracefully.
    For Game Bar: sends the toggle hotkey again.

    Returns (success, spoken_message).
    """
    global _recording_proc, _recording_path

    with _recording_lock:
        if _recording_proc is None and _recording_path is None:
            msg = "مفيش تسجيل شغال." if language == "ar" else "No recording is in progress."
            return False, msg

        saved_path = _recording_path

        if _recording_proc is not None:
            try:
                if _recording_proc.poll() is None:
                    _recording_proc.stdin.write(b"q")
                    _recording_proc.stdin.flush()
                    _recording_proc.wait(timeout=8)
            except Exception as exc:
                logger.debug("ffmpeg stop failed cleanly: %s", exc)
                try:
                    _recording_proc.terminate()
                except Exception:
                    pass
            _recording_proc = None
            _recording_path = None

            if saved_path and Path(saved_path).exists():
                location = _humanize_path(saved_path, language)
                msg = (f"وقفت التسجيل. اتحفظ في {location}" if language == "ar"
                       else f"Recording stopped. Saved to {location}")
                return True, msg
            msg = "وقفت التسجيل." if language == "ar" else "Recording stopped."
            return True, msg

        if saved_path == "gamebar":
            _recording_path = None
            ok = _send_hotkey_gamebar()
            if ok:
                msg = "وقفت تسجيل Game Bar." if language == "ar" else "Game Bar recording stopped."
                return True, msg
            msg = "حاولت أوقف Game Bar بس مش شغال." if language == "ar" else "Failed to stop Game Bar recording."
            return False, msg

        _recording_proc = None
        _recording_path = None
        msg = "وقفت التسجيل." if language == "ar" else "Recording stopped."
        return True, msg
