import re

from core.config import (
    ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS,
    CONFIRMATION_TIMEOUT_SECONDS,
    SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE,
)
from core.logger import logger
from core.response_templates import format_confirmation_prompt
from os_control.action_log import log_action
from os_control.adapter_result import (
    confirmation_result,
    failure_result,
    success_result,
    to_legacy_pair,
)
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
    "\u0627\u0637\u0641\u064a \u0627\u0644\u0643\u0645\u0628\u064a\u0648\u062a\u0631": "shutdown",
    "\u0627\u063a\u0644\u0642 \u0627\u0644\u0643\u0645\u0628\u064a\u0648\u062a\u0631": "shutdown",
    "\u0627\u063a\u0644\u0642 \u0627\u0644\u062c\u0647\u0627\u0632": "shutdown",
    "\u0627\u0639\u0627\u062f\u0629 \u062a\u0634\u063a\u064a\u0644": "restart",
    "\u0627\u0639\u0645\u0644 \u0627\u0639\u0627\u062f\u0629 \u062a\u0634\u063a\u064a\u0644": "restart",
    "\u0646\u0627\u0645 \u0627\u0644\u0643\u0645\u0628\u064a\u0648\u062a\u0631": "sleep",
    "\u0642\u0641\u0644 \u0627\u0644\u0643\u0645\u0628\u064a\u0648\u062a\u0631": "lock",
    "\u0633\u062c\u0644 \u062e\u0631\u0648\u062c": "logoff",
    "\u062a\u0633\u062c\u064a\u0644 \u062e\u0631\u0648\u062c": "logoff",
}

_RETRYABLE_NON_DESTRUCTIVE_ERRORS = ("timed out", "temporarily unavailable")


def normalize_system_action(text):
    phrase = text.lower().strip()
    if phrase.startswith("system "):
        phrase = phrase[7:].strip()
    if phrase.startswith("\u0627\u0644\u0646\u0638\u0627\u0645 "):
        phrase = phrase[len("\u0627\u0644\u0646\u0638\u0627\u0645 ") :].strip()
    phrase = phrase.replace("please ", "")
    phrase = phrase.replace("\u0645\u0646 \u0641\u0636\u0644\u0643 ", "")
    phrase = phrase.replace("\u0644\u0648 \u0633\u0645\u062d\u062a ", "")
    phrase = re.sub(r"[^a-z0-9_\s\-\u0600-\u06FF]", " ", phrase)
    phrase = " ".join(phrase.split())
    if phrase in SYSTEM_COMMANDS:
        return phrase
    return ALIASES.get(phrase)


def is_system_command(text):
    return normalize_system_action(text) is not None


def request_system_command_result(action_key):
    if action_key not in SYSTEM_COMMANDS:
        return failure_result("Unsupported system command.", error_code="unsupported_action")

    if not policy_engine.is_command_allowed("system_command"):
        return failure_result("System commands are disabled by policy.", error_code="policy_blocked")

    cfg = SYSTEM_COMMANDS[action_key]
    require_second_factor = bool(cfg["destructive"] and SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE)
    risk_tier = "high" if cfg["destructive"] else "medium"

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
        details={
            "action": action_key,
            "token": token,
            "second_factor": require_second_factor,
            "risk_tier": risk_tier,
        },
    )

    message = format_confirmation_prompt(
        cfg["description"],
        token,
        risk_tier=risk_tier,
        timeout_seconds=CONFIRMATION_TIMEOUT_SECONDS,
        require_second_factor=require_second_factor,
    )
    return confirmation_result(
        message,
        token=token,
        second_factor=require_second_factor,
        risk_tier=risk_tier,
        debug_info={"action": action_key},
    )


def _run_system_template_with_safe_retry(template_name, destructive):
    attempts = 0
    last_error = ""
    while attempts < (1 if destructive else 2):
        attempts += 1
        ok, error, output = run_template(template_name, timeout_seconds=30)
        if ok:
            return True, "", output, attempts
        last_error = error or "PowerShell template failed"
        if destructive:
            break
        if not any(token in last_error.lower() for token in _RETRYABLE_NON_DESTRUCTIVE_ERRORS):
            break
    return False, last_error, "", attempts


def execute_system_command_result(action_key):
    if action_key not in SYSTEM_COMMANDS:
        return failure_result("Unsupported system command.", error_code="unsupported_action")

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
        return failure_result(msg, error_code="destructive_disabled", debug_info={"action": action_key})

    ok, error, output, attempts = _run_system_template_with_safe_retry(
        cfg["template"],
        destructive=bool(cfg["destructive"]),
    )
    if ok:
        log_action(
            "system_command",
            "success",
            details={"action": action_key, "output": output, "attempts": attempts},
        )
        logger.info("Executed system command template: %s", action_key)
        return success_result(
            f"Executed system command: {action_key}.",
            debug_info={"action": action_key, "attempts": attempts},
            executed_confirmed_action="system_command",
        )

    log_action(
        "system_command",
        "failed",
        details={"action": action_key, "attempts": attempts},
        error=error,
    )
    error_code = "timeout" if "timed out" in (error or "").lower() else "execution_failed"
    return failure_result(
        f"Execution failed: {error}",
        error_code=error_code,
        debug_info={"action": action_key, "attempts": attempts},
    )


def request_system_command(action_key):
    result = request_system_command_result(action_key)
    legacy_success, legacy_message = to_legacy_pair(result)
    legacy_meta = {}
    if isinstance(result, dict):
        for key in ("requires_confirmation", "token", "second_factor", "risk_tier"):
            if key in result:
                legacy_meta[key] = result[key]
    return legacy_success, legacy_message, legacy_meta


def execute_system_command(action_key):
    return to_legacy_pair(execute_system_command_result(action_key))
