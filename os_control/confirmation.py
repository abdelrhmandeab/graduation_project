import secrets
import time

from core.config import CONFIRMATION_TIMEOUT_SECONDS, CONFIRMATION_TOKEN_BYTES
from core.logger import logger
from os_control.action_log import log_action
from os_control.persistence import (
    cleanup_expired_confirmations,
    consume_confirmation,
    count_pending_confirmations,
    delete_confirmation,
    get_confirmation,
    store_confirmation,
)
from os_control.second_factor import (
    clear_confirmation_attempts,
    clear_second_factor_attempts,
    is_confirmation_allowed,
    record_confirmation_attempt,
    verify_second_factor,
)


class ConfirmationManager:
    def __init__(self, timeout_seconds=CONFIRMATION_TIMEOUT_SECONDS):
        self.timeout_seconds = timeout_seconds

    def create(self, action_name, description, payload):
        token = secrets.token_hex(max(4, int(CONFIRMATION_TOKEN_BYTES or 8)))
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

    def _check_confirmation_rate_limit(self, token):
        allowed, message = is_confirmation_allowed(token)
        if allowed:
            return True, ""
        log_action(
            "confirmation_rejected",
            "failed",
            details={"token": token, "reason": "token_rate_limited"},
        )
        return False, message

    def confirm(self, token):
        cleanup_expired_confirmations()
        token = str(token or "").strip().lower()
        rate_ok, rate_message = self._check_confirmation_rate_limit(token)
        if not rate_ok:
            return False, rate_message, None

        pending = get_confirmation(token)
        if not pending:
            clear_second_factor_attempts(token)
            record_confirmation_attempt(token, success=False)
            log_action(
                "confirmation_rejected",
                "failed",
                details={"token": token, "reason": "not_found_or_expired"},
            )
            return False, "Confirmation token not found or expired.", None

        if time.time() > pending["expires_at"]:
            delete_confirmation(token)
            clear_second_factor_attempts(token)
            record_confirmation_attempt(token, success=False)
            log_action(
                "confirmation_rejected",
                "failed",
                details={
                    "action_name": pending["action_name"],
                    "token": token,
                    "reason": "expired",
                },
            )
            return False, "Confirmation token expired.", None

        payload = pending["payload"] or {}
        if payload.get("require_second_factor"):
            record_confirmation_attempt(token, success=False)
            log_action(
                "confirmation_rejected",
                "failed",
                details={
                    "action_name": pending["action_name"],
                    "token": token,
                    "reason": "second_factor_required",
                },
            )
            return False, "Second factor required for this action.", payload

        consumed = consume_confirmation(token)
        if not consumed:
            clear_second_factor_attempts(token)
            record_confirmation_attempt(token, success=False)
            log_action(
                "confirmation_rejected",
                "failed",
                details={"token": token, "reason": "already_confirmed_or_raced"},
            )
            return False, "Confirmation token already used or expired.", None

        clear_second_factor_attempts(token)
        clear_confirmation_attempts(token)
        record_confirmation_attempt(token, success=True)
        log_action(
            "confirmation_accepted",
            "success",
            details={"action_name": consumed["action_name"], "token": token},
        )
        return True, "Confirmation accepted.", consumed["payload"] or {}

    def confirm_with_second_factor(self, token, second_factor_secret):
        cleanup_expired_confirmations()
        token = str(token or "").strip().lower()
        rate_ok, rate_message = self._check_confirmation_rate_limit(token)
        if not rate_ok:
            return False, rate_message, None

        pending = get_confirmation(token)
        if not pending:
            clear_second_factor_attempts(token)
            record_confirmation_attempt(token, success=False)
            log_action(
                "confirmation_rejected",
                "failed",
                details={"token": token, "reason": "not_found_or_expired"},
            )
            return False, "Confirmation token not found or expired.", None

        if time.time() > pending["expires_at"]:
            delete_confirmation(token)
            clear_second_factor_attempts(token)
            record_confirmation_attempt(token, success=False)
            log_action(
                "confirmation_rejected",
                "failed",
                details={
                    "action_name": pending["action_name"],
                    "token": token,
                    "reason": "expired",
                },
            )
            return False, "Confirmation token expired.", None

        payload = pending["payload"] or {}
        if payload.get("require_second_factor"):
            if not second_factor_secret:
                record_confirmation_attempt(token, success=False)
                log_action(
                    "confirmation_rejected",
                    "failed",
                    details={
                        "action_name": pending["action_name"],
                        "token": token,
                        "reason": "second_factor_missing",
                    },
                )
                return False, "Second factor required for this action.", payload
            factor_ok, factor_message = verify_second_factor(second_factor_secret, token=token)
            if not factor_ok:
                log_action(
                    "confirmation_second_factor",
                    "failed",
                    details={"action_name": pending["action_name"], "token": token},
                )
                record_confirmation_attempt(token, success=False)
                log_action(
                    "confirmation_rejected",
                    "failed",
                    details={
                        "action_name": pending["action_name"],
                        "token": token,
                        "reason": "second_factor_failed",
                    },
                )
                return False, factor_message or "Second factor verification failed.", payload

            log_action(
                "confirmation_second_factor",
                "success",
                details={"action_name": pending["action_name"], "token": token},
            )

        consumed = consume_confirmation(token)
        if not consumed:
            clear_second_factor_attempts(token)
            record_confirmation_attempt(token, success=False)
            log_action(
                "confirmation_rejected",
                "failed",
                details={"token": token, "reason": "already_confirmed_or_raced"},
            )
            return False, "Confirmation token already used or expired.", None

        clear_second_factor_attempts(token)
        clear_confirmation_attempts(token)
        record_confirmation_attempt(token, success=True)
        log_action(
            "confirmation_accepted",
            "success",
            details={"action_name": consumed["action_name"], "token": token},
        )
        return True, "Confirmation accepted.", consumed["payload"] or {}

    def pending_count(self):
        cleanup_expired_confirmations()
        return count_pending_confirmations()

    def cancel(self, token):
        cleanup_expired_confirmations()
        token = str(token or "").strip().lower()
        pending = consume_confirmation(token)
        if not pending:
            log_action(
                "confirmation_rejected",
                "failed",
                details={"token": token, "reason": "cancel_not_found_or_expired"},
            )
            return False, "Confirmation token not found or expired."

        clear_second_factor_attempts(token)
        clear_confirmation_attempts(token)
        log_action(
            "confirmation_cancelled",
            "success",
            details={"action_name": pending["action_name"], "token": token},
        )
        return True, "Pending confirmation cancelled."


confirmation_manager = ConfirmationManager()
