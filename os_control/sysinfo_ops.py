"""Battery and system status — uses psutil (already a core dependency)."""

import psutil

from core.logger import logger


def get_battery_status():
    """Return human-readable battery status string."""
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            return "No battery detected (desktop PC)."
        pct = battery.percent
        plugged = "charging" if battery.power_plugged else "on battery"
        secs = battery.secsleft
        if secs == psutil.POWER_TIME_UNLIMITED:
            time_str = "fully charged"
        elif secs == psutil.POWER_TIME_UNKNOWN or secs < 0:
            time_str = "calculating..."
        else:
            hrs, mins = divmod(secs // 60, 60)
            time_str = f"{int(hrs)}h {int(mins)}m remaining"
        return f"Battery: {pct}% ({plugged}, {time_str})"
    except Exception as exc:
        logger.warning("Battery status failed: %s", exc)
        return f"Could not read battery status: {exc}"


def get_system_info():
    """Return CPU, RAM, and disk usage summary."""
    try:
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage("C:\\")
        return (
            f"CPU: {cpu}% used\n"
            f"RAM: {ram.percent}% used ({ram.used // (1024 ** 3)}GB / {ram.total // (1024 ** 3)}GB)\n"
            f"Disk C: {disk.percent}% used ({disk.free // (1024 ** 3)}GB free)"
        )
    except Exception as exc:
        logger.warning("System info failed: %s", exc)
        return f"Could not read system info: {exc}"
