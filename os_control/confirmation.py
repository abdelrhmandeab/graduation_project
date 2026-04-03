import secrets
import time

from core.config import CONFIRMATION_TIMEOUT_SECONDS
from core.logger import logger
from os_control.action_log import log_action
from os_control.persistence import (
    cleanup_expired_confirmations,
    count_pending_confirmations,
    delete_confirmation,
    get_confirmation,
    store_confirmation,
)
from os_control.second_factor import clear_second_factor_attempts, verify_second_factor


class ConfirmationManager:
    def __init__(self, timeout_seconds=CONFIRMATION_TIMEOUT_SECONDS):
        self.timeout_seconds = timeout_seconds

    def create(self, action_name, description, payload):
        token = secrets.token_hex(3)
        now_ts = time.time()
        expires_at = now_ts + self.timeout_seconds

        store_confirmation(
            token=token,
            action_name=action_name,
            description=description,
            payload=payload,
            created_at=now_ts,
            expires_at=expires_at,
        )

        logger.info("Confirmation requested for %s (token=%s)", action_name, token)
        return token

    def confirm(self, token):
        cleanup_expired_confirmations()
        pending = get_confirmation(token)
        if not pending:
            clear_second_factor_attempts(token)
            return False, "Confirmation token not found or expired.", None

        if time.time() > pending["expires_at"]:
            delete_confirmation(token)
            clear_second_factor_attempts(token)
            return False, "Confirmation token expired.", None

        payload = pending["payload"] or {}
        if payload.get("require_second_factor"):
            return False, "Second factor required for this action.", payload

        delete_confirmation(token)
        clear_second_factor_attempts(token)
        log_action(
            "confirmation_accepted",
            "success",
            details={"action_name": pending["action_name"], "token": token},
        )
        return True, "Confirmation accepted.", payload

    def confirm_with_second_factor(self, token, second_factor_secret):
        cleanup_expired_confirmations()
        pending = get_confirmation(token)
        if not pending:
            clear_second_factor_attempts(token)
            return False, "Confirmation token not found or expired.", None

        if time.time() > pending["expires_at"]:
            delete_confirmation(token)
            clear_second_factor_attempts(token)
            return False, "Confirmation token expired.", None

        payload = pending["payload"] or {}
        if payload.get("require_second_factor"):
            if not second_factor_secret:
                return False, "Second factor required for this action.", payload
            factor_ok, factor_message = verify_second_factor(second_factor_secret, token=token)
            if not factor_ok:
                log_action(
                    "confirmation_second_factor",
                    "failed",
                    details={"action_name": pending["action_name"], "token": token},
                )
                return False, factor_message or "Second factor verification failed.", payload

            log_action(
                "confirmation_second_factor",
                "success",
                details={"action_name": pending["action_name"], "token": token},
            )

        delete_confirmation(token)
        clear_second_factor_attempts(token)
        log_action(
            "confirmation_accepted",
            "success",
            details={"action_name": pending["action_name"], "token": token},
        )
        return True, "Confirmation accepted.", payload

    def pending_count(self):
        cleanup_expired_confirmations()
        return count_pending_confirmations()

    def cancel(self, token):
        cleanup_expired_confirmations()
        pending = get_confirmation(token)
        if not pending:
            return False, "Confirmation token not found or expired."

        delete_confirmation(token)
        clear_second_factor_attempts(token)
        log_action(
            "confirmation_cancelled",
            "success",
            details={"action_name": pending["action_name"], "token": token},
        )
        return True, "Pending confirmation cancelled."


confirmation_manager = ConfirmationManager()
