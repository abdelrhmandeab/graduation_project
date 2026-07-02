"""Windows radio (Wi-Fi / Bluetooth / Airplane mode) control.


Primary path: Windows.Devices.Radios WinRT API via the `winsdk` package.
No admin rights required.

Fallback path: PowerShell scripts — still no admin for Wi-Fi adapter toggling
via netsh, but BT may need elevation on some machines.
"""
from __future__ import annotations

import asyncio
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import winsdk.windows.devices.radios as _wdr_type  # noqa: F401

from core.config import RADIO_BACKEND, AIRPLANE_RESTORE_RADIOS
from core.logger import get_logger

logger = get_logger("oscontrol")

# Module-level snapshot of radio states before airplane mode was engaged,
# so we can restore them on "airplane off".
_pre_airplane_states: dict[str, bool] = {}

# ──────────────────────────────────────────────
# WinRT helpers (winsdk)
# ──────────────────────────────────────────────

def _winrt_available() -> bool:
    if sys.platform != "win32":
        return False
    if RADIO_BACKEND == "powershell":
        return False
    try:
        import winsdk.windows.devices.radios as _wdr  # noqa: F401, type: ignore[import-not-found]
        return True
    except Exception:
        return False


def _run_async(coro):
    """Run an async coroutine synchronously, creating a loop if needed."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


async def _get_radios_async():
    import winsdk.windows.devices.radios as wdr  # type: ignore[import-not-found]
    return await wdr.Radio.get_radios_async()


def _winrt_get_radios() -> list:
    try:
        return list(_run_async(_get_radios_async()))
    except Exception as exc:
        logger.debug("WinRT get_radios failed: %s", exc)
        return []


async def _set_radio_state_async(radio, on: bool):
    import winsdk.windows.devices.radios as wdr  # type: ignore[import-not-found]
    state = wdr.RadioState.ON if on else wdr.RadioState.OFF
    await radio.set_state_async(state)


def _winrt_set_radio(kind_name: str, on: bool) -> bool:
    """Toggle a radio by kind name ('Wi-Fi' or 'Bluetooth')."""
    try:
        import winsdk.windows.devices.radios as wdr  # type: ignore[import-not-found]
        radios = _winrt_get_radios()
        matched = [r for r in radios if r.kind.name.lower() == kind_name.lower()]
        if not matched:
            logger.debug("No WinRT radio found for kind=%s", kind_name)
            return False
        for radio in matched:
            _run_async(_set_radio_state_async(radio, on))
        return True
    except Exception as exc:
        logger.debug("WinRT set_radio(%s, %s) failed: %s", kind_name, on, exc)
        return False


def _winrt_snapshot_radios() -> dict[str, bool]:
    """Return {kind_name: is_on} for all current radios."""
    try:
        import winsdk.windows.devices.radios as wdr  # type: ignore[import-not-found]
        radios = _winrt_get_radios()
        return {r.kind.name: (r.state == wdr.RadioState.ON) for r in radios}
    except Exception as exc:
        logger.debug("WinRT snapshot_radios failed: %s", exc)
        return {}


# ──────────────────────────────────────────────
# PowerShell fallback helpers
# ──────────────────────────────────────────────

def _ps_run(script: str, timeout: int = 8) -> tuple[bool, str]:
    """Run a PowerShell snippet, return (success, output/error)."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        combined = (result.stdout + result.stderr).strip()
        return result.returncode == 0, combined
    except Exception as exc:
        return False, str(exc)


def _ps_wifi(on: bool) -> bool:
    # netsh works without admin for wireless
    action = "connect" if on else "disconnect"
    if on:
        script = (
            "$iface = (netsh wlan show interfaces | Select-String 'Name\\s*:' | "
            "Select-Object -First 1) -replace '.*:\\s*',''; "
            "if ($iface) { netsh wlan $iface }"
        )
        # Simpler: enable the adapter via netsh
        script = (
            "netsh interface set interface name="
            + '"Wi-Fi"'
            + " admin=enabled 2>$null; "
            + "netsh interface set interface name="
            + '"Wireless Network Connection"'
            + " admin=enabled 2>$null; exit 0"
        )
    else:
        script = (
            "netsh interface set interface name="
            + '"Wi-Fi"'
            + " admin=disabled 2>$null; "
            + "netsh interface set interface name="
            + '"Wireless Network Connection"'
            + " admin=disabled 2>$null; exit 0"
        )
    ok, out = _ps_run(script)
    if not ok:
        logger.debug("PS wifi fallback failed: %s", out)
    return ok


def _ps_bluetooth(on: bool) -> bool:
    state = "Enabled" if on else "Disabled"
    action = "enable" if on else "disable"
    # Try PnP device toggle
    script = (
        f"Get-PnpDevice -Class Bluetooth -ErrorAction SilentlyContinue | "
        f"ForEach-Object {{ {action.capitalize()}-PnpDevice -InstanceId $_.InstanceId -Confirm:$false -ErrorAction SilentlyContinue }}; "
        f"exit 0"
    )
    ok, out = _ps_run(script)
    if not ok:
        logger.debug("PS bluetooth fallback failed: %s", out)
    return ok


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def set_radio(kind: str, on: bool) -> bool:
    """Toggle Wi-Fi or Bluetooth.

    kind: 'wifi' | 'bluetooth'
    Returns True on success, False on failure (never raises).
    """
    kind_lower = kind.lower()
    winrt_name = "Wi-Fi" if kind_lower == "wifi" else "Bluetooth"

    if _winrt_available():
        ok = _winrt_set_radio(winrt_name, on)
        if ok:
            logger.info("WinRT radio %s -> %s", kind, "on" if on else "off")
            return True
        logger.debug("WinRT radio toggle failed, falling back to PowerShell")

    # PowerShell fallback
    if kind_lower == "wifi":
        ok = _ps_wifi(on)
    else:
        ok = _ps_bluetooth(on)

    if ok:
        logger.info("PS fallback radio %s -> %s", kind, "on" if on else "off")
    else:
        logger.warning("All radio backends failed for %s -> %s", kind, "on" if on else "off")
    return ok


def set_airplane(on: bool, restore: bool | None = None) -> bool:
    """Toggle airplane mode (all radios off/on).

    When on=True, snapshots current radio states then turns all off.
    When on=False and restore=True (default from config), re-applies snapshot.
    Returns True if at least one radio was toggled successfully.
    """
    global _pre_airplane_states

    if restore is None:
        restore = AIRPLANE_RESTORE_RADIOS

    if on:
        # Snapshot current states
        if _winrt_available():
            _pre_airplane_states = _winrt_snapshot_radios()
            logger.debug("Airplane ON: snapshot=%s", _pre_airplane_states)

        # Turn both radios off in parallel to avoid sequential ~9s PS calls
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_wifi = pool.submit(set_radio, "wifi", False)
            f_bt = pool.submit(set_radio, "bluetooth", False)
            wifi_ok = f_wifi.result()
            bt_ok = f_bt.result()
        return wifi_ok or bt_ok

    else:
        # Airplane OFF
        if restore and _pre_airplane_states:
            items = list(_pre_airplane_states.items())
            _pre_airplane_states.clear()
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    pool.submit(set_radio, "wifi" if "wi-fi" in k.lower() else "bluetooth", v): k
                    for k, v in items
                }
                results = [f.result() for f in futures]
            return any(results)
        else:
            # No snapshot — just turn both on in parallel
            with ThreadPoolExecutor(max_workers=2) as pool:
                f_wifi = pool.submit(set_radio, "wifi", True)
                f_bt = pool.submit(set_radio, "bluetooth", True)
                wifi_ok = f_wifi.result()
                bt_ok = f_bt.result()
            _pre_airplane_states.clear()
            return wifi_ok or bt_ok


def get_radio_states() -> dict[str, bool]:
    """Return current radio states for diagnostics. Returns {} if unavailable."""
    if _winrt_available():
        snapshot = _winrt_snapshot_radios()
        if snapshot:
            return {k.lower(): v for k, v in snapshot.items()}
    return {}
