"""Native Windows OS helpers for low-latency desktop operations.

Each public helper prefers a native Windows API path first and falls back to
PowerShell only when the native path fails.
"""

from __future__ import annotations

import ctypes
import os
import platform
import struct
import subprocess
import time
from ctypes import POINTER, cast, wintypes
from functools import wraps
from pathlib import Path
from typing import Callable, TypeVar

from core.config import POWERSHELL_EXECUTABLE
from core.logger import logger
from os_control.powershell_bridge import run_template

try:
    import comtypes
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    _PYCAW_AVAILABLE = True
except ImportError:
    comtypes = None
    CLSCTX_ALL = None
    AudioUtilities = None
    IAudioEndpointVolume = None
    _PYCAW_AVAILABLE = False

try:
    import wmi as _wmi

    _WMI_AVAILABLE = True
except ImportError:
    _wmi = None
    _WMI_AVAILABLE = False

try:
    import screen_brightness_control as _sbc

    _SBC_AVAILABLE = True
except ImportError:
    _sbc = None
    _SBC_AVAILABLE = False

_IS_WINDOWS = platform.system().lower() == "windows"

if _IS_WINDOWS:
    _user32 = ctypes.windll.user32
    _gdi32 = ctypes.windll.gdi32
    _powrprof = ctypes.windll.powrprof
    _winmm = ctypes.windll.winmm
else:
    _user32 = None
    _gdi32 = None
    _powrprof = None
    _winmm = None

_T = TypeVar("_T")
_last_nonzero_volume_percent = 50


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


def timed_operation(operation_name: str, backend: str):
    """Log how long a backend-specific operation took."""

    def decorator(func: Callable[..., _T]):
        @wraps(func)
        def wrapper(*args, **kwargs):
            started = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                logger.debug("native_ops.%s via %s took %.2f ms", operation_name, backend, elapsed_ms)

        return wrapper

    return decorator


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _powershell_run(script: str):
    result = subprocess.run(
        [POWERSHELL_EXECUTABLE, "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True,
        text=True,
        timeout=20,
        env=os.environ.copy(),
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(stderr or f"PowerShell failed with code {result.returncode}")
    return (result.stdout or "").strip()


def _volume_endpoint():
    if not _IS_WINDOWS or not _PYCAW_AVAILABLE:
        return None

    initialized = False
    try:
        if hasattr(comtypes, "CoInitialize"):
            comtypes.CoInitialize()
            initialized = True
        speakers = AudioUtilities.GetSpeakers()
        interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception as exc:
        logger.debug("pycaw endpoint resolution failed: %s", exc)
        return None
    finally:
        if initialized and hasattr(comtypes, "CoUninitialize"):
            try:
                comtypes.CoUninitialize()
            except Exception:
                pass


def _wave_volume_to_percent(volume_value: int) -> int:
    left = volume_value & 0xFFFF
    right = (volume_value >> 16) & 0xFFFF
    average = (left + right) // 2
    return max(0, min(100, int(round((average / 65535.0) * 100))))


def _percent_to_wave_volume(percent: int) -> int:
    clamped = max(0, min(100, int(percent)))
    channel = int(round((clamped / 100.0) * 65535.0)) & 0xFFFF
    return channel | (channel << 16)


@timed_operation("get_volume", "pycaw")
def _get_volume_pycaw():
    endpoint = _volume_endpoint()
    if endpoint is None:
        return None
    return float(endpoint.GetMasterVolumeLevelScalar())


@timed_operation("set_volume", "pycaw")
def _set_volume_pycaw(level: float):
    endpoint = _volume_endpoint()
    if endpoint is None:
        return False
    endpoint.SetMasterVolumeLevelScalar(float(_clamp(level, 0.0, 1.0)), None)
    return True


@timed_operation("toggle_mute", "pycaw")
def _toggle_mute_pycaw():
    endpoint = _volume_endpoint()
    if endpoint is None:
        return None
    muted = bool(endpoint.GetMute())
    endpoint.SetMute(0 if muted else 1, None)
    return not muted


def _is_muted_pycaw():
    endpoint = _volume_endpoint()
    if endpoint is None:
        return None
    try:
        return bool(endpoint.GetMute())
    except Exception as exc:
        logger.debug("pycaw mute state read failed: %s", exc)
        return None


@timed_operation("get_volume", "ctypes")
def _get_volume_waveout():
    if not _IS_WINDOWS:
        return None
    try:
        volume_value = wintypes.DWORD()
        result = _winmm.waveOutGetVolume(0, ctypes.byref(volume_value))
        if result != 0:
            return None
        return _wave_volume_to_percent(int(volume_value.value)) / 100.0
    except Exception as exc:
        logger.debug("Native waveOut volume read failed: %s", exc)
        return None


@timed_operation("set_volume", "ctypes")
def _set_volume_waveout(level: float):
    global _last_nonzero_volume_percent
    if not _IS_WINDOWS:
        return False
    try:
        percent = int(round(_clamp(level, 0.0, 1.0) * 100.0))
        if percent > 0:
            _last_nonzero_volume_percent = percent
        volume_value = _percent_to_wave_volume(percent)
        result = _winmm.waveOutSetVolume(0, volume_value)
        return int(result) == 0
    except Exception as exc:
        logger.debug("Native waveOut volume set failed: %s", exc)
        return False


@timed_operation("toggle_mute", "ctypes")
def _toggle_mute_waveout():
    current = get_volume()
    if current is None:
        return None
    if current > 0.0:
        return _set_volume_waveout(0.0)
    restore_level = max(0.0, min(1.0, _last_nonzero_volume_percent / 100.0 if _last_nonzero_volume_percent else 0.5))
    return _set_volume_waveout(restore_level)


@timed_operation("set_volume", "powershell")
def _set_volume_powershell(level: float):
    percent = int(round(_clamp(level, 0.0, 1.0) * 100.0))
    ok, error, _output = run_template("volume_set", {"JARVIS_VOLUME_PERCENT": percent})
    if not ok:
        raise RuntimeError(error or "PowerShell volume_set failed")
    return True


@timed_operation("toggle_mute", "powershell")
def _toggle_mute_powershell():
    ok, error, _output = run_template("volume_mute")
    if not ok:
        raise RuntimeError(error or "PowerShell volume_mute failed")
    return True


@timed_operation("get_brightness", "wmi")
def _get_brightness_wmi():
    if not _IS_WINDOWS or not _WMI_AVAILABLE:
        return None
    try:
        client = _wmi.WMI(namespace="wmi")
        monitors = client.WmiMonitorBrightness()
        if not monitors:
            return None
        return int(monitors[0].CurrentBrightness)
    except Exception as exc:
        logger.debug("WMI brightness read failed: %s", exc)
        return None


@timed_operation("set_brightness", "wmi")
def _set_brightness_wmi(level: int):
    if not _IS_WINDOWS or not _WMI_AVAILABLE:
        return False
    try:
        client = _wmi.WMI(namespace="wmi")
        methods = client.WmiMonitorBrightnessMethods()
        if not methods:
            return False
        methods[0].WmiSetBrightness(1, int(_clamp(level, 0, 100)))
        return True
    except Exception as exc:
        logger.debug("WMI brightness set failed: %s", exc)
        return False


@timed_operation("get_brightness", "sbc")
def _get_brightness_sbc():
    if not _SBC_AVAILABLE:
        return None
    try:
        values = _sbc.get_brightness()
        return int(values[0]) if values else None
    except Exception as exc:
        logger.debug("screen-brightness-control read failed: %s", exc)
        return None


@timed_operation("set_brightness", "sbc")
def _set_brightness_sbc(level: int):
    if not _SBC_AVAILABLE:
        return False
    try:
        _sbc.set_brightness(int(_clamp(level, 0, 100)))
        return True
    except Exception as exc:
        logger.debug("screen-brightness-control set failed: %s", exc)
        return False


@timed_operation("get_brightness", "powershell")
def _get_brightness_powershell():
    output = _powershell_run(
        "$value=Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightness | "
        "Select-Object -First 1 -ExpandProperty CurrentBrightness; "
        "if($null -eq $value){ exit 1 }; "
        "Write-Output $value"
    )
    return int(str(output).strip())


@timed_operation("set_brightness", "powershell")
def _set_brightness_powershell(level: int):
    ok, error, _output = run_template("brightness_set", {"JARVIS_BRIGHTNESS_PERCENT": int(_clamp(level, 0, 100))})
    if not ok:
        raise RuntimeError(error or "PowerShell brightness_set failed")
    return True


@timed_operation("lock", "ctypes")
def _lock_workstation_ctypes():
    if not _IS_WINDOWS:
        return False
    try:
        return bool(_user32.LockWorkStation())
    except Exception as exc:
        logger.debug("Native lock failed: %s", exc)
        return False


@timed_operation("lock", "powershell")
def _lock_workstation_powershell():
    ok, error, _output = run_template("lock")
    if not ok:
        raise RuntimeError(error or "PowerShell lock failed")
    return True


@timed_operation("sleep", "ctypes")
def _sleep_system_ctypes():
    if not _IS_WINDOWS:
        return False
    try:
        return bool(_powrprof.SetSuspendState(0, 1, 0))
    except Exception as exc:
        logger.debug("Native sleep failed: %s", exc)
        return False


@timed_operation("sleep", "powershell")
def _sleep_system_powershell():
    ok, error, _output = run_template("sleep")
    if not ok:
        raise RuntimeError(error or "PowerShell sleep failed")
    return True


def _default_screenshot_dir() -> Path:
    return Path.home() / "Pictures" / "Screenshots"


@timed_operation("screenshot", "ctypes")
def _capture_primary_screen_screenshot_ctypes(output_dir: Path):
    if not _IS_WINDOWS:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / f"jarvis_shot_{time.strftime('%Y%m%d_%H%M%S')}.bmp"

    screen_dc = None
    mem_dc = None
    bitmap = None
    old_obj = None
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
            return None

        bitmap = _gdi32.CreateCompatibleBitmap(screen_dc, width, height)
        if not bitmap:
            return None

        old_obj = _gdi32.SelectObject(mem_dc, bitmap)
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
            buffer,
            ctypes.byref(bmi),
            0,
        )
        if result == 0:
            return None

        file_header_size = 14
        info_header_size = ctypes.sizeof(_BITMAPINFOHEADER)
        with open(target_path, "wb") as handle:
            handle.write(struct.pack("<HIHHI", 0x4D42, file_header_size + info_header_size + buffer_size, 0, 0, file_header_size + info_header_size))
            handle.write(
                struct.pack(
                    "<IiiHHIIiiII",
                    info_header_size,
                    width,
                    height,
                    1,
                    24,
                    0,
                    buffer_size,
                    0,
                    0,
                    0,
                    0,
                )
            )
            handle.write(buffer)
        return str(target_path)
    except Exception as exc:
        logger.debug("Native screenshot failed: %s", exc)
        try:
            if target_path.exists():
                target_path.unlink()
        except Exception:
            pass
        return None
    finally:
        try:
            if old_obj:
                _gdi32.SelectObject(mem_dc, old_obj)
        except Exception:
            pass
        try:
            if bitmap:
                _gdi32.DeleteObject(bitmap)
        except Exception:
            pass
        try:
            if mem_dc:
                _gdi32.DeleteDC(mem_dc)
        except Exception:
            pass
        try:
            if screen_dc:
                _user32.ReleaseDC(0, screen_dc)
        except Exception:
            pass


@timed_operation("screenshot", "powershell")
def _capture_primary_screen_screenshot_powershell(output_dir: Path):
    ok, error, output = run_template("screenshot")
    if not ok:
        raise RuntimeError(error or "PowerShell screenshot failed")

    source_path = Path(output) if output else None
    if source_path and source_path.exists() and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        target_path = output_dir / source_path.name
        try:
            if target_path.exists():
                target_path.unlink()
            source_path.replace(target_path)
            return str(target_path)
        except Exception as exc:
            logger.debug("PowerShell screenshot move failed: %s", exc)
            return str(source_path)
    return output or None


def get_volume():
    """Return the current system volume as a 0.0-1.0 float, or None."""
    for getter in (_get_volume_pycaw, _get_volume_waveout):
        try:
            value = getter()
            if value is not None:
                return float(_clamp(float(value), 0.0, 1.0))
        except Exception as exc:
            logger.debug("Volume read fallback failed: %s", exc)
    return None


def set_volume(level: float):
    """Set the system volume to a 0.0-1.0 float level."""
    target = float(_clamp(float(level), 0.0, 1.0))
    for setter in (_set_volume_pycaw, _set_volume_waveout, _set_volume_powershell):
        try:
            if setter(target):
                return True
        except Exception as exc:
            logger.debug("Volume set fallback failed: %s", exc)
    return False


def adjust_volume(delta: float):
    """Adjust the system volume by delta points on the 0.0-1.0 scale."""
    current = get_volume()
    if current is None:
        return False, None
    target = float(_clamp(current + delta, 0.0, 1.0))
    return set_volume(target), int(round(target * 100.0))


def toggle_mute():
    """Toggle the mute state of the system audio."""
    for toggler in (_toggle_mute_pycaw, _toggle_mute_waveout, _toggle_mute_powershell):
        try:
            result = toggler()
            if result is not None:
                return bool(result)
        except Exception as exc:
            logger.debug("Volume mute fallback failed: %s", exc)
    return False


def get_system_volume_percent():
    """Compatibility wrapper returning volume as a 0-100 integer."""
    volume = get_volume()
    if volume is None:
        return None
    return int(round(volume * 100.0))


def set_system_volume_percent(percent: int):
    """Compatibility wrapper returning True on success."""
    return set_volume(int(percent) / 100.0)


def adjust_system_volume_percent(delta: int):
    """Compatibility wrapper returning (success, new_percent)."""
    current = get_system_volume_percent()
    if current is None:
        return False, None
    target = max(0, min(100, int(current + delta)))
    return set_system_volume_percent(target), target


def toggle_system_mute():
    """Compatibility wrapper returning (success, resulting_volume_percent)."""
    success = toggle_mute()
    if not success:
        return False, None
    muted = _is_muted_pycaw()
    if muted is True:
        return True, 0
    current = get_system_volume_percent()
    if current is None:
        return True, None
    if current == 0:
        return True, 0
    return True, current


def get_brightness():
    """Return the current screen brightness as a 0-100 integer, or None."""
    for getter in (_get_brightness_wmi, _get_brightness_sbc, _get_brightness_powershell):
        try:
            value = getter()
            if value is not None:
                return int(_clamp(int(value), 0, 100))
        except Exception as exc:
            logger.debug("Brightness read fallback failed: %s", exc)
    return None


def set_brightness(level: int):
    """Set the screen brightness to a 0-100 integer."""
    target = int(_clamp(int(level), 0, 100))
    for setter in (_set_brightness_wmi, _set_brightness_sbc, _set_brightness_powershell):
        try:
            if setter(target):
                return True
        except Exception as exc:
            logger.debug("Brightness set fallback failed: %s", exc)
    return False


def adjust_brightness(delta: int):
    """Adjust brightness by delta percentage points."""
    current = get_brightness()
    if current is None:
        return False, None
    target = max(0, min(100, int(current + delta)))
    return set_brightness(target), target


def get_system_brightness_percent():
    return get_brightness()


def set_system_brightness_percent(percent: int):
    return set_brightness(percent)


def adjust_system_brightness_percent(delta: int):
    return adjust_brightness(delta)


def lock_workstation():
    """Lock the current Windows session."""
    for locker in (_lock_workstation_ctypes, _lock_workstation_powershell):
        try:
            if locker():
                return True
        except Exception as exc:
            logger.debug("Lock fallback failed: %s", exc)
    return False


def sleep_system():
    """Put the machine to sleep without requiring admin privileges."""
    for sleeper in (_sleep_system_ctypes, _sleep_system_powershell):
        try:
            if sleeper():
                return True
        except Exception as exc:
            logger.debug("Sleep fallback failed: %s", exc)
    return False


def capture_primary_screen_screenshot(output_dir=None):
    """Capture the primary screen to ~/Pictures/Screenshots by default."""
    target_dir = Path(output_dir) if output_dir else _default_screenshot_dir()
    for capturer in (_capture_primary_screen_screenshot_ctypes, _capture_primary_screen_screenshot_powershell):
        try:
            path = capturer(target_dir)
            if path:
                return str(path)
        except Exception as exc:
            logger.debug("Screenshot fallback failed: %s", exc)
    return None
