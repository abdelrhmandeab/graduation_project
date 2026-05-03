"""Native Windows OS helpers for low-latency operations.

This module keeps simple desktop actions off the PowerShell path when possible.
Missing Windows APIs should degrade gracefully to False/None so callers can
fall back to the existing PowerShell bridge.
"""

from __future__ import annotations

import ctypes
import platform
import tempfile
import time
from ctypes import wintypes
from pathlib import Path

from core.logger import logger

_IS_WINDOWS = platform.system().lower() == "windows"
_winmm = ctypes.windll.winmm if _IS_WINDOWS else None
_user32 = ctypes.windll.user32 if _IS_WINDOWS else None
_gdi32 = ctypes.windll.gdi32 if _IS_WINDOWS else None
_powrprof = ctypes.windll.powrprof if _IS_WINDOWS else None

# Preserve the last non-zero volume so mute can restore the user's level.
_last_nonzero_volume_percent = 50


class _BITMAPFILEHEADER(ctypes.Structure):
    _fields_ = [
        ("bfType", wintypes.WORD),
        ("bfSize", wintypes.DWORD),
        ("bfReserved1", wintypes.WORD),
        ("bfReserved2", wintypes.WORD),
        ("bfOffBits", wintypes.DWORD),
    ]


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER)]


def _wave_volume_to_percent(volume_value: int) -> int:
    left = volume_value & 0xFFFF
    right = (volume_value >> 16) & 0xFFFF
    average = (left + right) // 2
    return max(0, min(100, int(round((average / 65535.0) * 100))))


def _percent_to_wave_volume(percent: int) -> int:
    clamped = max(0, min(100, int(percent)))
    channel = int(round((clamped / 100.0) * 65535.0)) & 0xFFFF
    return channel | (channel << 16)


def get_system_volume_percent():
    """Return the current system volume as a 0-100 integer, or None on failure."""
    if not _IS_WINDOWS:
        return None
    try:
        volume_value = wintypes.DWORD()
        result = _winmm.waveOutGetVolume(0, ctypes.byref(volume_value))
        if result != 0:
            return None
        return _wave_volume_to_percent(int(volume_value.value))
    except Exception as exc:
        logger.debug("Native volume read failed: %s", exc)
        return None


def set_system_volume_percent(percent: int):
    """Set the system volume using the Windows waveOut API."""
    global _last_nonzero_volume_percent
    if not _IS_WINDOWS:
        return False
    try:
        level = max(0, min(100, int(percent)))
        if level > 0:
            _last_nonzero_volume_percent = level
        volume_value = _percent_to_wave_volume(level)
        result = _winmm.waveOutSetVolume(0, volume_value)
        return int(result) == 0
    except Exception as exc:
        logger.debug("Native volume set failed: %s", exc)
        return False


def adjust_system_volume_percent(delta: int):
    """Adjust the system volume by delta percent."""
    current = get_system_volume_percent()
    if current is None:
        return False, None
    target = max(0, min(100, int(current + delta)))
    return set_system_volume_percent(target), target


def toggle_system_mute():
    """Mute or restore the system volume.

    Returns:
        tuple[bool, int | None]: (success, resulting_volume_percent)
    """
    global _last_nonzero_volume_percent
    current = get_system_volume_percent()
    if current is None:
        return False, None

    if current > 0:
        _last_nonzero_volume_percent = current
        ok = set_system_volume_percent(0)
        return ok, 0 if ok else None

    restore_level = _last_nonzero_volume_percent or 50
    ok = set_system_volume_percent(restore_level)
    return ok, restore_level if ok else None


def lock_workstation():
    """Lock the current Windows session using the native API."""
    if not _IS_WINDOWS:
        return False
    try:
        return bool(_user32.LockWorkStation())
    except Exception as exc:
        logger.debug("Native lock failed: %s", exc)
        return False


def sleep_system():
    """Put the machine to sleep using the native Windows power API."""
    if not _IS_WINDOWS:
        return False
    try:
        # False, False, False mirrors the existing PowerShell helper.
        return bool(_powrprof.SetSuspendState(False, False, False))
    except Exception as exc:
        logger.debug("Native sleep failed: %s", exc)
        return False


def capture_primary_screen_screenshot(output_dir=None):
    """Capture the primary screen to a PNG-like BMP file.

    The capture is written as a BMP file because it can be produced directly
    with GDI without extra dependencies. Callers may rename or convert it later
    if they need a different format.
    """
    if not _IS_WINDOWS:
        return None

    output_root = Path(output_dir or tempfile.gettempdir())
    output_root.mkdir(parents=True, exist_ok=True)
    target_path = output_root / f"jarvis_shot_{time.strftime('%Y%m%d_%H%M%S')}.bmp"

    try:
        screen_dc = _user32.GetDC(0)
        if not screen_dc:
            return None
        width = int(_user32.GetSystemMetrics(0))
        height = int(_user32.GetSystemMetrics(1))
        if width <= 0 or height <= 0:
            return None

        mem_dc = _gdi32.CreateCompatibleDC(screen_dc)
        if not mem_dc:
            _user32.ReleaseDC(0, screen_dc)
            return None

        bitmap = _gdi32.CreateCompatibleBitmap(screen_dc, width, height)
        if not bitmap:
            _gdi32.DeleteDC(mem_dc)
            _user32.ReleaseDC(0, screen_dc)
            return None

        old_obj = _gdi32.SelectObject(mem_dc, bitmap)
        try:
            if not _gdi32.BitBlt(mem_dc, 0, 0, width, height, screen_dc, 0, 0, 0x00CC0020):
                return None

            bmi = _BITMAPINFO()
            bmi.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
            bmi.bmiHeader.biWidth = width
            bmi.bmiHeader.biHeight = height
            bmi.bmiHeader.biPlanes = 1
            bmi.bmiHeader.biBitCount = 24
            bmi.bmiHeader.biCompression = 0
            bmi.bmiHeader.biSizeImage = ((width * 3 + 3) & ~3) * height

            buffer_size = int(bmi.bmiHeader.biSizeImage)
            buffer = (ctypes.c_ubyte * buffer_size)()

            result = _gdi32.GetDIBits(
                mem_dc,
                bitmap,
                0,
                height,
                ctypes.byref(buffer),
                ctypes.byref(bmi),
                0,
            )
            if result == 0:
                return None

            file_header = _BITMAPFILEHEADER()
            file_header.bfType = 0x4D42
            file_header.bfOffBits = ctypes.sizeof(_BITMAPFILEHEADER) + ctypes.sizeof(_BITMAPINFOHEADER)
            file_header.bfSize = file_header.bfOffBits + buffer_size

            with open(target_path, "wb") as handle:
                handle.write(bytes(file_header))
                handle.write(bytes(bmi.bmiHeader))
                handle.write(buffer)

            return str(target_path)
        finally:
            if old_obj:
                _gdi32.SelectObject(mem_dc, old_obj)
            _gdi32.DeleteObject(bitmap)
            _gdi32.DeleteDC(mem_dc)
            _user32.ReleaseDC(0, screen_dc)
    except Exception as exc:
        logger.debug("Native screenshot failed: %s", exc)
        try:
            if target_path.exists():
                target_path.unlink()
        except Exception:
            pass
        return None
