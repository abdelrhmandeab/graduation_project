import tempfile
import unittest
from unittest.mock import patch

import core.shutdown as shutdown_runtime
from core.shutdown import perform_shutdown_cleanup, setup_shutdown
from os_control.app_ops import close_app_result, request_close_app_result
from os_control.file_ops import (
    create_directory_result,
    get_file_metadata_result,
    list_directory_result,
    request_delete_item,
)
from os_control.system_ops import request_system_command_result
from os_control.system_ops import execute_system_command_result


class Phase3AdapterHardeningTests(unittest.TestCase):
    def _assert_adapter_contract(self, payload):
        self.assertIsInstance(payload, dict)
        for field in ("success", "user_message", "error_code", "debug_info"):
            self.assertIn(field, payload)
        self.assertIsInstance(payload.get("debug_info"), dict)

    def test_list_directory_result_uses_standard_adapter_contract(self):
        result = list_directory_result()

        self._assert_adapter_contract(result)
        self.assertTrue(isinstance(result.get("user_message"), str))

    def test_file_metadata_rejects_control_chars_with_validation_error(self):
        result = get_file_metadata_result("bad\x00path")

        self._assert_adapter_contract(result)
        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("error_code"), "validation_error")

    def test_create_directory_existing_path_returns_already_exists(self):
        with tempfile.TemporaryDirectory(prefix="jarvis_phase3_") as temp_dir:
            result = create_directory_result(temp_dir)

        self._assert_adapter_contract(result)
        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("error_code"), "already_exists")

    def test_close_request_returns_not_found_when_process_not_running(self):
        with patch("os_control.app_ops.policy_engine.is_command_allowed", return_value=True), patch(
            "os_control.app_ops._is_process_running", return_value=False
        ):
            result = request_close_app_result("notepad")

        self._assert_adapter_contract(result)
        self.assertFalse(result.get("success"))
        self.assertEqual(result.get("error_code"), "not_found")
        self.assertNotIn("requires_confirmation", result)

    def test_close_app_safe_retry_on_transient_failure(self):
        resolved_payload = {
            "ok": True,
            "target": "notepad.exe",
            "process_name": "notepad.exe",
            "query": "notepad",
            "resolution_status": "exact",
        }
        with patch("os_control.app_ops.policy_engine.is_command_allowed", return_value=True), patch(
            "os_control.app_ops._resolve_close_target", return_value=resolved_payload
        ), patch(
            "os_control.app_ops.run_template",
            side_effect=[
                (False, "timed out", ""),
                (True, "", "closed"),
            ],
        ):
            result = close_app_result("notepad")

        self._assert_adapter_contract(result)
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("debug_info", {}).get("attempts"), 2)

    def test_system_confirmation_payload_contract(self):
        with patch("os_control.system_ops.policy_engine.is_command_allowed", return_value=True), patch(
            "os_control.system_ops.confirmation_manager.create", return_value="tok-system"
        ) as create_mock:
            result = request_system_command_result("shutdown")

        self._assert_adapter_contract(result)
        self.assertTrue(result.get("success"))
        self.assertTrue(result.get("requires_confirmation"))
        self.assertEqual(result.get("token"), "tok-system")
        self.assertEqual(result.get("risk_tier"), "high")

        payload = dict((create_mock.call_args.kwargs or {}).get("payload") or {})
        self.assertEqual(payload.get("kind"), "system_command")
        self.assertEqual(payload.get("action_key"), "shutdown")
        self.assertIn("command_args", payload)

    def test_system_browser_search_returns_clean_message(self):
        with patch(
            "os_control.system_ops.run_template",
            return_value=(True, "", "browser_search_web=?????"),
        ):
            result = execute_system_command_result("browser_search_web", {"search_query": "iran war"})

        self._assert_adapter_contract(result)
        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("user_message"), "Searching the web for: iran war")

    def test_file_confirmation_payload_contract(self):
        with tempfile.NamedTemporaryFile(prefix="jarvis_phase3_", suffix=".txt", delete=False) as temp_file:
            temp_path = temp_file.name
        try:
            with patch("os_control.file_ops._check_path_policy", return_value=(True, "")), patch(
                "os_control.file_ops.confirmation_manager.create", return_value="tok-file"
            ) as create_mock:
                result = request_delete_item(temp_path, permanent=False)

            self._assert_adapter_contract(result)
            self.assertTrue(result.get("success"))
            self.assertTrue(result.get("requires_confirmation"))
            self.assertEqual(result.get("token"), "tok-file")
            self.assertEqual(result.get("risk_tier"), "high")

            payload = dict((create_mock.call_args.kwargs or {}).get("payload") or {})
            self.assertEqual(payload.get("kind"), "file_operation")
            self.assertEqual(payload.get("operation"), "delete_item")
            self.assertEqual(dict(payload.get("resolved_args") or {}).get("path"), temp_path)
        finally:
            import os

            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_shutdown_cleanup_stops_search_index_worker(self):
        shutdown_runtime.reset_shutdown_state_for_tests()
        with patch("core.shutdown.job_queue_service.stop") as stop_jobs, patch(
            "core.shutdown.search_index_service.stop"
        ) as stop_index, patch("core.shutdown.speech_engine.interrupt") as stop_speech:
            perform_shutdown_cleanup()

        self.assertTrue(stop_jobs.called)
        self.assertTrue(stop_index.called)
        self.assertTrue(stop_speech.called)

    def test_shutdown_signal_handler_sets_event_without_systemexit(self):
        shutdown_runtime.reset_shutdown_state_for_tests()
        handlers = {}

        def _capture_signal(sig, handler):
            handlers[sig] = handler

        with patch("core.shutdown.signal.signal", side_effect=_capture_signal), patch(
            "core.shutdown.perform_shutdown_cleanup"
        ) as cleanup_mock, patch("builtins.print") as print_mock:
            event = setup_shutdown()
            self.assertFalse(event.is_set())

            sigint_handler = handlers.get(shutdown_runtime.signal.SIGINT)
            self.assertIsNotNone(sigint_handler)
            sigint_handler(shutdown_runtime.signal.SIGINT, None)

            self.assertTrue(event.is_set())
            self.assertEqual(cleanup_mock.call_count, 1)
            self.assertGreaterEqual(print_mock.call_count, 1)

    def test_shutdown_cleanup_is_idempotent(self):
        shutdown_runtime.reset_shutdown_state_for_tests()
        with patch("core.shutdown.job_queue_service.stop") as stop_jobs, patch(
            "core.shutdown.search_index_service.stop"
        ) as stop_index, patch("core.shutdown.speech_engine.interrupt") as stop_speech:
            first = perform_shutdown_cleanup()
            second = perform_shutdown_cleanup()

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(stop_jobs.call_count, 1)
        self.assertEqual(stop_index.call_count, 1)
        self.assertEqual(stop_speech.call_count, 1)


if __name__ == "__main__":
    unittest.main()