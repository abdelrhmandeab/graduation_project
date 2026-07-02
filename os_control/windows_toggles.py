"""Windows 11 system toggles: Night Light, Do Not Disturb (Focus Assist),
Energy Saver, and Live Captions.

All implementations are best-effort, no-admin, registry-first with
ms-settings:// URI fallback. Each function returns bool; never raises.
"""
from __future__ import annotations

import ctypes
import struct
import subprocess
import winreg
from typing import Optional

from core.config import (
    TOGGLE_NIGHT_LIGHT_METHOD,
    TOGGLE_DND_METHOD,
    TOGGLE_ENERGY_SAVER_METHOD,
    LIVE_CAPTION_HOTKEY,
)
from core.logger import get_logger

logger = get_logger("oscontrol")

# ──────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────

def _open_settings_uri(uri: str) -> bool:
    """Launch a ms-settings: URI as a best-effort fallback."""
    try:
        subprocess.Popen(["explorer.exe", uri], shell=False)
        return True
    except Exception as exc:
        logger.debug("settings URI %s failed: %s", uri, exc)
        return False


def _reg_get_binary(hive, subkey: str, value: str) -> Optional[bytes]:
    try:
        with winreg.OpenKey(hive, subkey, 0, winreg.KEY_READ) as k:
            data, _ = winreg.QueryValueEx(k, value)
            return bytes(data)
    except Exception:
        return None


def _reg_set_binary(hive, subkey: str, value: str, data: bytes) -> bool:
    try:
        with winreg.OpenKey(hive, subkey, 0, winreg.KEY_WRITE) as k:
            winreg.SetValueEx(k, value, 0, winreg.REG_BINARY, data)
        return True
    except Exception as exc:
        logger.debug("reg set %s failed: %s", value, exc)
        return False


def _reg_get_dword(hive, subkey: str, value: str, default: int = 0) -> int:
    try:
        with winreg.OpenKey(hive, subkey, 0, winreg.KEY_READ) as k:
            data, _ = winreg.QueryValueEx(k, value)
            return int(data)
    except Exception:
        return default


def _reg_set_dword(hive, subkey: str, value: str, data: int) -> bool:
    try:
        with winreg.OpenKey(
            hive, subkey, 0, winreg.KEY_WRITE | winreg.KEY_CREATE_SUB_KEY
        ) as k:
            winreg.SetValueEx(k, value, 0, winreg.REG_DWORD, int(data))
        return True
    except Exception as exc:
        logger.debug("reg dword set %s failed: %s", value, exc)
        return False


def _simulate_hotkey(hotkey_str: str) -> bool:
    """Simulate a hotkey like 'win+ctrl+l' via keybd_event."""
    try:
        user32 = ctypes.windll.user32
        KEYEVENTF_KEYUP = 0x0002

        _VK = {
            "win": 0x5B,    # VK_LWIN
            "ctrl": 0x11,   # VK_CONTROL
            "alt": 0x12,    # VK_MENU
            "shift": 0x10,
        }

        parts = [p.strip().lower() for p in hotkey_str.split("+")]
        vk_codes = []
        for part in parts:
            if part in _VK:
                vk_codes.append(_VK[part])
            elif len(part) == 1:
                vk_codes.append(ord(part.upper()))
            else:
                logger.debug("Unknown hotkey part: %s", part)
                return False

        for vk in vk_codes:
            user32.keybd_event(vk, 0, 0, 0)
        for vk in reversed(vk_codes):
            user32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
        return True
    except Exception as exc:
        logger.debug("Hotkey simulation failed: %s", exc)
        return False


# ──────────────────────────────────────────────────────────
# Night Light
# ──────────────────────────────────────────────────────────
# Windows stores Night Light state as a binary blob in CloudStore.
# The enable byte is at a fixed offset (index 18 in the 43-byte blob).
# Setting it to 0x02 (on) or 0x00 (off) and writing back toggles the feature.
# Source: multiple community reverse-engineering references.

_NL_KEY = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\CloudStore\Store"
    r"\DefaultAccount\Current\default$windows.data.bluelightreduction"
    r".bluelightreductionstate\windows.data.bluelightreduction.bluelightreductionstate"
)
_NL_VALUE = "Data"
_NL_ENABLE_BYTE_INDEX = 18


def _night_light_registry(on: bool) -> bool:
    data = _reg_get_binary(winreg.HKEY_CURRENT_USER, _NL_KEY, _NL_VALUE)
    if data is None or len(data) < _NL_ENABLE_BYTE_INDEX + 1:
        logger.debug("Night light blob not found or too short (len=%s)", len(data) if data else 0)
        return False

    blob = bytearray(data)
    blob[_NL_ENABLE_BYTE_INDEX] = 0x02 if on else 0x00

    if not _reg_set_binary(winreg.HKEY_CURRENT_USER, _NL_KEY, _NL_VALUE, bytes(blob)):
        return False

    # Notify Windows shell to pick up the registry change
    try:
        _SMTO_ABORTIFHUNG = 0x0002
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, "ImmersiveColorSet", _SMTO_ABORTIFHUNG, 100, None)
    except Exception:
        pass

    # Verify the write
    verify = _reg_get_binary(winreg.HKEY_CURRENT_USER, _NL_KEY, _NL_VALUE)
    if verify and len(verify) > _NL_ENABLE_BYTE_INDEX:
        return verify[_NL_ENABLE_BYTE_INDEX] == (0x02 if on else 0x00)
    return True


def set_night_light(on: bool) -> bool:
    if TOGGLE_NIGHT_LIGHT_METHOD in ("auto", "registry"):
        ok = _night_light_registry(on)
        if ok:
            logger.info("Night light -> %s via registry", "on" if on else "off")
            return True
        logger.debug("Night light registry toggle failed, falling back to URI")
    uri = "ms-settings:nightlight"
    logger.info("Opening night light settings URI (manual toggle required)")
    return _open_settings_uri(uri)


# ──────────────────────────────────────────────────────────
# Do Not Disturb / Focus Assist
# ──────────────────────────────────────────────────────────
# Windows 11 DND is controlled by the "QuietHoursSettings" DWORD under
# HKCU\...\Notifications\Settings. Value 0 = off, value 1 = Priority only,
# value 2 = Alarms only (maximum DND).
# We use value 2 for "on" (most restrictive, silences toasts) and 0 for off.

_DND_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Notifications\Settings"
_DND_VALUE = "NOC_GLOBAL_SETTING_TOASTS_ENABLED"
_DND_FOCUS_KEY = r"SOFTWARE\Microsoft\Windows\CurrentVersion\CloudStore\Store\DefaultAccount\Current\default$windows.data.notifications.quiethourssettings\windows.data.notifications.quiethourssettings"
_DND_FOCUS_VALUE = "Data"


def _dnd_registry(on: bool) -> bool:
    # Primary: disable/enable toast notifications globally
    # 0 = toasts disabled (DND on), 1 = toasts enabled (DND off)
    ok = _reg_set_dword(
        winreg.HKEY_CURRENT_USER,
        _DND_KEY,
        _DND_VALUE,
        0 if on else 1,
    )
    if not ok:
        return False

    # Notify shell
    try:
        _SMTO_ABORTIFHUNG = 0x0002
        ctypes.windll.user32.SendMessageTimeoutW(0xFFFF, 0x001A, 0, 0, _SMTO_ABORTIFHUNG, 100, None)
    except Exception:
        pass

    # Verify
    result = _reg_get_dword(winreg.HKEY_CURRENT_USER, _DND_KEY, _DND_VALUE, default=-1)
    expected = 0 if on else 1
    return result == expected


def set_dnd(on: bool) -> bool:
    if TOGGLE_DND_METHOD in ("auto", "registry"):
        ok = _dnd_registry(on)
        if ok:
            logger.info("DND -> %s via registry", "on" if on else "off")
            return True
        logger.debug("DND registry toggle failed, falling back to URI")
    uri = "ms-settings:quiethours"
    logger.info("Opening DND/quiet hours settings URI (manual toggle required)")
    return _open_settings_uri(uri)


# ──────────────────────────────────────────────────────────
# Energy Saver / Battery Saver
# ──────────────────────────────────────────────────────────
# powercfg approach: set the active power scheme to "POWER SAVER" guid
# (a1841308-3541-4fab-bc81-f71556f20b4a) for on, or restore "BALANCED"
# (381b4222-f694-41f0-9685-ff5bb260df2e) for off.
# We first try the registry flag (HKCU battery saver), then powercfg.

_ES_POWERCFG_SAVER_GUID = "a1841308-3541-4fab-bc81-f71556f20b4a"
_ES_POWERCFG_BALANCED_GUID = "381b4222-f694-41f0-9685-ff5bb260df2e"
_ES_BATTERY_SAVER_KEY = r"SYSTEM\CurrentControlSet\Control\Power"
_ES_BATTERY_SAVER_VALUE = "EnergySaverStatus"


def _energy_saver_powercfg(on: bool) -> bool:
    guid = _ES_POWERCFG_SAVER_GUID if on else _ES_POWERCFG_BALANCED_GUID
    try:
        result = subprocess.run(
            ["powercfg", "/setactive", guid],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception as exc:
        logger.debug("powercfg setactive failed: %s", exc)
        return False


def _energy_saver_registry(on: bool) -> bool:
    # EnergySaverStatus: 0 = off, 1 = on
    ok = _reg_set_dword(
        winreg.HKLM,  # HKEY_LOCAL_MACHINE — may need elevation
        _ES_BATTERY_SAVER_KEY,
        _ES_BATTERY_SAVER_VALUE,
        1 if on else 0,
    )
    return ok


def set_energy_saver(on: bool) -> bool:
    method = TOGGLE_ENERGY_SAVER_METHOD

    if method in ("auto", "powercfg"):
        ok = _energy_saver_powercfg(on)
        if ok:
            logger.info("Energy saver -> %s via powercfg", "on" if on else "off")
            return True
        logger.debug("Energy saver powercfg failed")

    if method in ("auto", "registry"):
        ok = _energy_saver_registry(on)
        if ok:
            logger.info("Energy saver -> %s via registry", "on" if on else "off")
            return True
        logger.debug("Energy saver registry failed")

    uri = "ms-settings:batterysaver"
    logger.info("Opening battery saver settings URI (manual toggle required)")
    return _open_settings_uri(uri)


# ──────────────────────────────────────────────────────────
# Live Captions
# ──────────────────────────────────────────────────────────
# Windows 11 Live Captions is toggled by Win+Ctrl+L.
# There is no public API; the system hotkey is the documented reliable path.

def set_live_captions(on: bool) -> bool:
    # The hotkey is a toggle — it turns on if off and off if on.
    # We always simulate it; there is no way to query current state cheaply.
    hotkey = LIVE_CAPTION_HOTKEY
    ok = _simulate_hotkey(hotkey)
    if ok:
        logger.info("Live captions toggled via hotkey %s", hotkey)
        return True
    # Fallback: open settings page
    uri = "ms-settings:easeofaccess-livecaptions"
    logger.info("Opening live captions settings URI (manual toggle required)")
    return _open_settings_uri(uri)
