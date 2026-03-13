from core.logger import logger
from os_control.action_log import log_action
from os_control.policy import policy_engine
from os_control.powershell_bridge import run_template

KNOWN_APPS = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "cmd": "cmd.exe",
    "powershell": "powershell.exe",
    "explorer": "explorer.exe",
}


def _friendly_open_error(target, error_text):
    lowered = (error_text or "").lower()
    if "cannot find the file specified" in lowered or "not recognized" in lowered:
        return (
            f"I could not find an app or executable named '{target}'. "
            "Try `open app notepad` or use a filesystem command like `open C partition`."
        )
    if "access is denied" in lowered:
        return f"Access denied while trying to open '{target}'."
    return "I could not open that app."


def open_app(app_name):
    if not app_name:
        return False, "No app name provided."
    if not policy_engine.is_command_allowed("app_open"):
        return False, "Application launch is disabled by policy."

    target = KNOWN_APPS.get(app_name.lower(), app_name)

    try:
        ok, error, _output = run_template(
            "open_app",
            env_overrides={"JARVIS_APP_PATH": target},
            timeout_seconds=15,
        )
        if not ok:
            log_action(
                "open_app",
                "failed",
                details={"target": target},
                error=error or "PowerShell template failed",
            )
            return False, _friendly_open_error(target, error or "")

        log_action("open_app", "success", details={"target": target})
        logger.info("Opened app via template PowerShell: %s", target)
        return True, f"Opening {app_name}."
    except Exception as exc:
        log_action("open_app", "failed", details={"target": target}, error=exc)
        logger.error("Open app failed: %s", exc)
        return False, str(exc)
