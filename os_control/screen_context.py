"""Phase 4 -- describe what is currently visible on screen.

Window mode (default): reads foreground window title + enumerates visible
top-level windows via Win32 API.  No image capture.

Vision mode (SCREEN_DESCRIBE_MODE=vision): captures a screenshot and sends
it to the vision LLM.  Falls back to window mode if capture or LLM fails.
"""
from __future__ import annotations

import ctypes
import logging
from typing import List, Tuple

logger = logging.getLogger("screen_context")

# ---------------------------------------------------------------------------
# Win32 helpers
# ---------------------------------------------------------------------------

_SHELL_CLASS_NAMES = frozenset({
    "Progman", "WorkerW", "SHELLDLL_DefView", "Shell_TrayWnd",
    "DV2ControlHost", "MsgrIMEWindowClass", "SysShadow",
    "tooltips_class32", "EdgeUiInputTopWndClass",
})

_IGNORE_TITLE_FRAGMENTS = frozenset({
    "default ime", "msctfime ui", "gdkhwnd", "gdk screen",
})


def _is_real_window(hwnd: int, user32) -> bool:
    if not user32.IsWindowVisible(hwnd):
        return False
    if user32.GetParent(hwnd):
        return False
    buf_len = user32.GetWindowTextLengthW(hwnd)
    if buf_len == 0 or buf_len > 512:
        return False
    return True


def _get_window_title(hwnd: int, user32) -> str:
    length = user32.GetWindowTextLengthW(hwnd)
    if not length:
        return ""
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value.strip()


def _get_class_name(hwnd: int, user32) -> str:
    buf = ctypes.create_unicode_buffer(256)
    user32.GetClassNameW(hwnd, buf, 256)
    return buf.value.strip()


def _get_process_name(hwnd: int) -> str:
    try:
        import psutil
        kernel32 = ctypes.windll.kernel32
        pid = ctypes.c_ulong(0)
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        proc = psutil.Process(pid.value)
        return proc.name()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_foreground_window() -> Tuple[str, str]:
    """Return (title, process_name) of the current foreground window."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return "", ""
        title = _get_window_title(hwnd, user32)
        process = _get_process_name(hwnd)
        return title, process
    except Exception as exc:
        logger.debug("get_foreground_window failed: %s", exc)
        return "", ""


def list_visible_windows(max_apps: int = 8) -> List[Tuple[str, str]]:
    """Return up to *max_apps* (title, process_name) pairs for visible windows.

    Excludes shell chrome, taskbar, and other non-app windows.
    """
    try:
        user32 = ctypes.windll.user32
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_size_t, ctypes.c_size_t)

        results: List[Tuple[str, str]] = []

        def _callback(hwnd, _lparam):
            if not _is_real_window(hwnd, user32):
                return True
            cls = _get_class_name(hwnd, user32)
            if cls in _SHELL_CLASS_NAMES:
                return True
            title = _get_window_title(hwnd, user32)
            if not title:
                return True
            tl = title.lower()
            if any(frag in tl for frag in _IGNORE_TITLE_FRAGMENTS):
                return True
            process = _get_process_name(hwnd)
            results.append((title, process))
            return True

        user32.EnumWindows(EnumWindowsProc(_callback), 0)
        # Deduplicate by title (keep first occurrence, which tends to be foreground)
        seen: set = set()
        deduped: List[Tuple[str, str]] = []
        for title, proc in results:
            if title not in seen:
                seen.add(title)
                deduped.append((title, proc))
        return deduped[:max_apps]
    except Exception as exc:
        logger.debug("list_visible_windows failed: %s", exc)
        return []


def describe_screen(language: str = "en", max_apps: int = 8) -> str:
    """Return a bilingual natural-language description of the current screen."""
    is_ar = language == "ar"

    fg_title, fg_proc = get_foreground_window()
    windows = list_visible_windows(max_apps=max_apps)

    # Build foreground description
    if fg_title:
        app_label = _friendly_name(fg_title, fg_proc)
        if is_ar:
            fg_line = f"دلوقتي شغّال على «{app_label}»."
        else:
            fg_line = f"You're currently in «{app_label}»."
    else:
        fg_line = "مش شايف شباك مفعّل." if is_ar else "No active window detected."

    # Build open-apps list (exclude foreground to avoid repetition)
    others = [
        _friendly_name(t, p)
        for t, p in windows
        if t != fg_title
    ]

    if not others:
        if is_ar:
            return fg_line + " مفيش تطبيقات تانية مفتوحة."
        return fg_line + " No other apps are open."

    if is_ar:
        if len(others) == 1:
            apps_line = f"وعندك كمان «{others[0]}» مفتوح."
        else:
            items = "» و«".join(others)
            apps_line = f"وعندك كمان «{items}» مفتوحين."
        return f"{fg_line} {apps_line}"
    else:
        if len(others) == 1:
            apps_line = f"You also have «{others[0]}» open."
        else:
            listed = ", ".join(f"«{a}»" for a in others[:-1]) + f", and «{others[-1]}»"
            apps_line = f"You also have {listed} open."
        return f"{fg_line} {apps_line}"


# ---------------------------------------------------------------------------
# Vision mode
# ---------------------------------------------------------------------------

def describe_screen_vision(language: str = "en") -> str:
    """Capture a screenshot and ask the vision LLM to describe it.

    Falls back to window mode on any error.
    """
    try:
        from os_control.native_ops import capture_primary_screen_screenshot
        import os

        screenshot_path = capture_primary_screen_screenshot()
        if not screenshot_path or not os.path.exists(screenshot_path):
            raise RuntimeError("Screenshot capture returned nothing")

        # Read image bytes and send to LLM with vision prompt
        with open(screenshot_path, "rb") as f:
            img_bytes = f.read()

        try:
            from llm.ollama_client import chat_with_image
        except ImportError:
            raise RuntimeError("chat_with_image not available in this LLM backend")

        if language == "ar":
            prompt = "صف الشاشة دي بشكل مختصر بالعامية المصرية. قول ايه التطبيق المفعّل وايه اللي شايفه."
        else:
            prompt = "Briefly describe what is on this screen. Mention the active app and key visible content."

        reply = chat_with_image(prompt, img_bytes, mime_type="image/png")
        if reply:
            return reply.strip()
        raise RuntimeError("Vision LLM returned empty response")

    except Exception as exc:
        logger.debug("describe_screen_vision failed (%s), falling back to window mode", exc)
        return describe_screen(language=language)


# ---------------------------------------------------------------------------
# Entry point (respects SCREEN_DESCRIBE_MODE config)
# ---------------------------------------------------------------------------

def describe_screen_auto(language: str = "en", max_apps: int = 8) -> str:
    """Choose window or vision mode based on SCREEN_DESCRIBE_MODE config."""
    try:
        from core.config import SCREEN_DESCRIBE_MODE
        mode = SCREEN_DESCRIBE_MODE
    except ImportError:
        mode = "window"

    if mode == "vision":
        return describe_screen_vision(language=language)
    return describe_screen(language=language, max_apps=max_apps)


# ---------------------------------------------------------------------------
# Friendly name helper
# ---------------------------------------------------------------------------

_PROC_FRIENDLY: dict[str, str] = {
    "chrome.exe": "Chrome",
    "firefox.exe": "Firefox",
    "msedge.exe": "Edge",
    "brave.exe": "Brave",
    "opera.exe": "Opera",
    "code.exe": "VS Code",
    "code - insiders.exe": "VS Code Insiders",
    "pycharm64.exe": "PyCharm",
    "idea64.exe": "IntelliJ IDEA",
    "slack.exe": "Slack",
    "discord.exe": "Discord",
    "teams.exe": "Microsoft Teams",
    "outlook.exe": "Outlook",
    "winword.exe": "Word",
    "excel.exe": "Excel",
    "powerpnt.exe": "PowerPoint",
    "onenote.exe": "OneNote",
    "notepad.exe": "Notepad",
    "notepad++.exe": "Notepad++",
    "explorer.exe": "File Explorer",
    "cmd.exe": "Command Prompt",
    "powershell.exe": "PowerShell",
    "windowsterminal.exe": "Windows Terminal",
    "spotify.exe": "Spotify",
    "vlc.exe": "VLC",
    "zoom.exe": "Zoom",
    "telegram.exe": "Telegram",
    "whatsapp.exe": "WhatsApp",
    "obsidian.exe": "Obsidian",
    "notion.exe": "Notion",
    "figma.exe": "Figma",
}


def _friendly_name(title: str, process: str) -> str:
    """Return a human-friendly display name for a window."""
    proc_key = process.lower() if process else ""
    friendly_proc = _PROC_FRIENDLY.get(proc_key, "")
    if not friendly_proc:
        return title
    # If process name is known and title already contains it, just use title
    if friendly_proc.lower() in title.lower():
        return title
    # Otherwise prepend the friendly app name
    return f"{friendly_proc} — {title}" if title else friendly_proc
