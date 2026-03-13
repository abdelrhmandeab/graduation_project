import re

from core.config import (
    ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS,
    CONFIRMATION_TIMEOUT_SECONDS,
    SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE,
)
from core.logger import logger
from os_control.action_log import log_action
from os_control.confirmation import confirmation_manager
from os_control.policy import policy_engine
from os_control.powershell_bridge import run_template

SYSTEM_COMMANDS = {
    "shutdown": {
        "template": "shutdown",
        "description": "Shut down this computer",
        "destructive": True,
    },
    "restart": {
        "template": "restart",
        "description": "Restart this computer",
        "destructive": True,
    },
    "sleep": {
        "template": "sleep",
        "description": "Put this computer to sleep",
        "destructive": False,
    },
    "lock": {
        "template": "lock",
        "description": "Lock this computer",
        "destructive": False,
    },
    "logoff": {
        "template": "logoff",
        "description": "Log off current user",
        "destructive": True,
    },
}

ALIASES = {
    "shutdown": "shutdown",
    "shut down": "shutdown",
    "shutdown computer": "shutdown",
    "shut down computer": "shutdown",
    "power off": "shutdown",
    "turn off computer": "shutdown",
    "turn off pc": "shutdown",
    "restart": "restart",
    "restart computer": "restart",
    "restart pc": "restart",
    "reboot": "restart",
    "sleep computer": "sleep",
    "lock computer": "lock",
    "sign out": "logoff",
    "log out": "logoff",
}


def normalize_system_action(text):
    phrase = text.lower().strip()
    if phrase.startswith("system "):
        phrase = phrase[7:].strip()
    phrase = phrase.replace("please ", "")
    phrase = re.sub(r"[^a-z0-9_\s-]", " ", phrase)
    phrase = " ".join(phrase.split())
    if phrase in SYSTEM_COMMANDS:
        return phrase
    return ALIASES.get(phrase)


def is_system_command(text):
    return normalize_system_action(text) is not None


def request_system_command(action_key):
    if action_key not in SYSTEM_COMMANDS:
        return False, "Unsupported system command.", {}

    if not policy_engine.is_command_allowed("system_command"):
        return False, "System commands are disabled by policy.", {}

    cfg = SYSTEM_COMMANDS[action_key]
    require_second_factor = bool(cfg["destructive"] and SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE)

    payload = {
        "kind": "system_command",
        "action_key": action_key,
        "require_second_factor": require_second_factor,
    }
    token = confirmation_manager.create(
        action_name=f"system_{action_key}",
        description=cfg["description"],
        payload=payload,
    )
    log_action(
        "system_command_request",
        "pending",
        details={"action": action_key, "token": token, "second_factor": require_second_factor},
    )

    message = (
        f"Confirmation required: {cfg['description']}. "
        f"Say `confirm {token}`"
    )
    if require_second_factor:
        message += " and provide PIN/passphrase as second factor."
    message += f" within {CONFIRMATION_TIMEOUT_SECONDS} seconds."
    return True, message, {"requires_confirmation": True, "token": token, "second_factor": require_second_factor}


def execute_system_command(action_key):
    if action_key not in SYSTEM_COMMANDS:
        return False, "Unsupported system command."

    cfg = SYSTEM_COMMANDS[action_key]
    if cfg["destructive"] and not ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS:
        msg = (
            "Blocked by configuration. Set ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS=True "
            "in core/config.py to enable this command."
        )
        log_action(
            "system_command",
            "blocked",
            details={"action": action_key, "reason": "destructive_disabled"},
        )
        return False, msg

    ok, error, output = run_template(cfg["template"], timeout_seconds=30)
    if ok:
        log_action(
            "system_command",
            "success",
            details={"action": action_key, "output": output},
        )
        logger.info("Executed system command template: %s", action_key)
        return True, f"Executed system command: {action_key}."

    log_action("system_command", "failed", details={"action": action_key}, error=error)
    return False, f"Execution failed: {error}"
