import tempfile
import threading
import unittest
from unittest.mock import patch

from core.config import CONFIRMATION_TOKEN_BYTES
from core.command_parser import parse_command
from core.command_router import _execute_confirmed_payload
from os_control.confirmation import confirmation_manager
from os_control.file_ops import request_delete_item
from os_control.risk_policy import validate_risk_policy_coverage
from os_control.second_factor import clear_confirmation_attempts
from os_control.system_ops import request_system_command_result
from os_control.system_ops import SYSTEM_COMMANDS


class Phase4SafetyFrameworkTests(unittest.TestCase):
    def setUp(self):
        for token in ("badbad", "deadbeef", "badbadbad"):
            clear_confirmation_attempts(token)

    def _assert_adapter_contract(self, payload):
        self.assertIsInstance(payload, dict)
        for key in ("success", "user_message", "error_code", "debug_info"):
            self.assertIn(key, payload)

    def test_medium_risk_lock_requires_confirmation(self):
        with patch("os_control.system_ops.policy_engine.is_command_allowed", return_value=True), patch(
            "os_control.system_ops.confirmation_manager.create", return_value="tok-lock"
        ):
            result = request_system_command_result("lock")

        self._assert_adapter_contract(result)
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("requires_confirmation"))
        self.assertEqual(result.get("token"), "tok-lock")
        self.assertEqual(result.get("risk_tier"), "medium")
        self.assertFalse(result.get("second_factor"))

    def test_high_risk_shutdown_requires_second_factor(self):
        with patch("os_control.system_ops.policy_engine.is_command_allowed", return_value=True), patch(
            "os_control.system_ops.confirmation_manager.create", return_value="tok-shutdown"
        ):
            result = request_system_command_result("shutdown")

        self._assert_adapter_contract(result)
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("requires_confirmation"))
        self.assertEqual(result.get("token"), "tok-shutdown")
        self.assertEqual(result.get("risk_tier"), "high")
        self.assertTrue(result.get("second_factor"))

    def test_delete_defaults_to_soft_delete_and_requires_confirmation(self):
        with tempfile.NamedTemporaryFile(prefix="jarvis_phase4_", suffix=".txt", delete=False) as temp_file:
            temp_path = temp_file.name
        try:
            with patch("os_control.file_ops._check_path_policy", return_value=(True, "")), patch(
                "os_control.file_ops.confirmation_manager.create", return_value="tok-delete"
            ):
                result = request_delete_item(temp_path, permanent=False)

            self._assert_adapter_contract(result)
            self.assertTrue(result.get("success"))
            self.assertTrue(result.get("requires_confirmation"))
            self.assertEqual(result.get("token"), "tok-delete")
            self.assertEqual(result.get("risk_tier"), "high")
            self.assertTrue(result.get("second_factor"))
        finally:
            import os

            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_permanent_delete_blocked_by_default(self):
        with tempfile.NamedTemporaryFile(prefix="jarvis_phase4_", suffix=".txt", delete=False) as temp_file:
            temp_path = temp_file.name
        try:
            with patch("os_control.file_ops._check_path_policy", return_value=(True, "")):
                result = request_delete_item(temp_path, permanent=True)

            self._assert_adapter_contract(result)
            self.assertFalse(result.get("success"))
            self.assertEqual(result.get("error_code"), "policy_blocked")
        finally:
            import os

            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_confirmation_rejection_is_audited_for_invalid_token(self):
        with patch("os_control.confirmation.log_action") as log_mock:
            ok, message, payload = confirmation_manager.confirm("badbad")

        self.assertFalse(ok)
        self.assertIsNone(payload)
        self.assertIn("not found", message.lower())
        self.assertTrue(log_mock.called)
        first_call_args = log_mock.call_args_list[0].args
        self.assertEqual(first_call_args[0], "confirmation_rejected")
        self.assertEqual(first_call_args[1], "failed")

    def test_confirmation_token_length_is_hardened(self):
        token = confirmation_manager.create(
            action_name="test_action",
            description="test",
            payload={"kind": "system_command", "action_key": "lock", "require_second_factor": False},
        )
        try:
            self.assertGreaterEqual(len(token), int(CONFIRMATION_TOKEN_BYTES) * 2)
            self.assertRegex(token, r"^[0-9a-f]+$")
        finally:
            confirmation_manager.cancel(token)

    def test_confirmation_rate_limit_blocks_bruteforce_attempts(self):
        token = "badbadbad"
        clear_confirmation_attempts(token)
        with patch("os_control.second_factor._confirmation_attempt_limits", return_value=(2, 120)):
            first_ok, first_message, _ = confirmation_manager.confirm(token)
            second_ok, second_message, _ = confirmation_manager.confirm(token)
            third_ok, third_message, _ = confirmation_manager.confirm(token)

        self.assertFalse(first_ok)
        self.assertIn("not found", first_message.lower())
        self.assertFalse(second_ok)
        self.assertIn("not found", second_message.lower())
        self.assertFalse(third_ok)
        self.assertIn("too many failed confirmation attempts", third_message.lower())

    def test_confirmation_token_is_consumed_once_under_race(self):
        token = confirmation_manager.create(
            action_name="race_test",
            description="race",
            payload={"kind": "system_command", "action_key": "lock", "require_second_factor": False},
        )

        results = []
        lock = threading.Lock()

        def _worker():
            ok, _message, _payload = confirmation_manager.confirm(token)
            with lock:
                results.append(bool(ok))

        threads = [threading.Thread(target=_worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(results.count(True), 1)
        self.assertEqual(results.count(False), 1)

    def test_cancel_missing_token_rejection_is_audited(self):
        with patch("os_control.confirmation.log_action") as log_mock:
            ok, message = confirmation_manager.cancel("deadbeef")

        self.assertFalse(ok)
        self.assertIn("not found", message.lower())
        self.assertTrue(log_mock.called)
        call_args = log_mock.call_args_list[0].args
        self.assertEqual(call_args[0], "confirmation_rejected")
        self.assertEqual(call_args[1], "failed")
        self.assertEqual(log_mock.call_args_list[0].kwargs.get("details", {}).get("reason"), "cancel_not_found_or_expired")

    def test_unsupported_payload_kind_rejection_is_audited(self):
        with patch("core.command_router.log_action") as log_mock:
            ok, message, _meta = _execute_confirmed_payload({"kind": "mystery"})

        self.assertFalse(ok)
        self.assertIn("unsupported", message.lower())
        self.assertTrue(log_mock.called)
        self.assertEqual(log_mock.call_args_list[0].args[0], "confirmation_rejected")
        self.assertEqual(log_mock.call_args_list[0].kwargs.get("details", {}).get("reason"), "unsupported_payload_kind")

    def test_risk_policy_matrix_validation_is_consistent(self):
        report = validate_risk_policy_coverage(
            system_commands=SYSTEM_COMMANDS,
            file_operations={"move_item", "rename_item", "delete_item", "delete_item_permanent"},
            app_operations={"close_app"},
        )
        self.assertTrue(report.get("ok"), msg=str(report.get("errors") or []))

    def test_parser_accepts_long_confirmation_token(self):
        parsed = parse_command("confirm abcdef123456")
        self.assertEqual(parsed.intent, "OS_CONFIRMATION")
        self.assertEqual((parsed.args or {}).get("token"), "abcdef123456")

    def test_permanent_delete_requires_explicit_phrase(self):
        soft = parse_command("delete report.txt")
        permanent = parse_command("delete permanently report.txt")

        self.assertEqual(soft.intent, "OS_FILE_NAVIGATION")
        self.assertEqual(soft.action, "delete_item")
        self.assertEqual(permanent.intent, "OS_FILE_NAVIGATION")
        self.assertEqual(permanent.action, "delete_item_permanent")


if __name__ == "__main__":
    unittest.main()