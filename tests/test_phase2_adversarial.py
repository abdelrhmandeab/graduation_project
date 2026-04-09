import unittest
from unittest.mock import patch

from core.command_parser import ParsedCommand
from core.command_router import route_command
from core.intent_confidence import _entity_clarification_threshold, resolve_clarification_reply
from core.metrics import metrics
from core.session_memory import session_memory


class Phase2AdversarialTests(unittest.TestCase):
    def setUp(self):
        self._previous_language = session_memory.get_preferred_language()
        self._previous_pending = session_memory.get_pending_clarification()
        session_memory.clear_pending_clarification()
        session_memory.clear_clarification_preferences()
        session_memory.clear_app_usage()
        session_memory.set_preferred_language("en")
        metrics.reset()

    def tearDown(self):
        session_memory.clear_pending_clarification()
        if self._previous_pending:
            session_memory.set_pending_clarification(self._previous_pending)
        session_memory.set_preferred_language(self._previous_language)

    def test_adaptive_entity_threshold_changes_by_language(self):
        parsed = ParsedCommand(
            intent="OS_APP_OPEN",
            raw="open app power",
            normalized="open app power",
            action="",
            args={"app_name": "power"},
        )
        en_threshold = _entity_clarification_threshold(parsed, language="en", mixed_language=False)
        ar_threshold = _entity_clarification_threshold(parsed, language="ar", mixed_language=False)
        mixed_threshold = _entity_clarification_threshold(parsed, language="en", mixed_language=True)

        self.assertLess(ar_threshold, en_threshold)
        self.assertGreater(mixed_threshold, en_threshold)

    def test_clarification_reply_accepts_natural_phrase(self):
        payload = {
            "prompt": "pick app",
            "language": "en",
            "options": [
                {
                    "id": "open_app_1",
                    "label": "Google Chrome (chrome.exe)",
                    "reply_tokens": ["chrome", "google chrome", "browser"],
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "chrome.exe"},
                },
                {
                    "id": "open_app_2",
                    "label": "Microsoft Edge (msedge.exe)",
                    "reply_tokens": ["edge", "microsoft edge"],
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "msedge.exe"},
                },
            ],
        }

        result = resolve_clarification_reply("the chrome one", payload)
        self.assertEqual(result.status, "resolved")
        self.assertEqual((result.option or {}).get("id"), "open_app_1")

    def test_clarification_reply_handles_negative_correction(self):
        payload = {
            "prompt": "pick app",
            "language": "en",
            "options": [
                {
                    "id": "open_app_1",
                    "label": "Google Chrome (chrome.exe)",
                    "reply_tokens": ["chrome", "google chrome"],
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "chrome.exe"},
                },
                {
                    "id": "open_app_2",
                    "label": "Microsoft Edge (msedge.exe)",
                    "reply_tokens": ["edge", "microsoft edge"],
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "msedge.exe"},
                },
            ],
        }

        result = resolve_clarification_reply("not chrome", payload)
        self.assertEqual(result.status, "resolved")
        self.assertEqual((result.option or {}).get("id"), "open_app_2")

    def test_unrelated_reply_twice_triggers_fallback_hint(self):
        pending_payload = {
            "reason": "app_name_ambiguous",
            "prompt": "Reply with the number or cancel.",
            "options": [
                {
                    "id": "open_app_1",
                    "label": "Google Chrome (chrome.exe)",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "chrome.exe"},
                    "reply_tokens": ["chrome", "1"],
                },
                {
                    "id": "open_app_2",
                    "label": "Microsoft Edge (msedge.exe)",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "msedge.exe"},
                    "reply_tokens": ["edge", "2"],
                },
            ],
            "source_text": "open app browser",
            "language": "en",
        }
        session_memory.set_pending_clarification(pending_payload)

        first = route_command("I am talking about something else entirely")
        second = route_command("still random unrelated words for this prompt")

        self.assertIn("Reply with", first)
        self.assertIn("direct selection", second)

    def test_saved_clarification_choice_is_reused(self):
        ambiguous = {
            "status": "ambiguous",
            "query": "power",
            "candidates": [
                {
                    "canonical_name": "PowerShell",
                    "executable": "powershell.exe",
                    "matched_alias": "power shell",
                    "score": 0.81,
                },
                {
                    "canonical_name": "PowerPoint",
                    "executable": "powerpnt.exe",
                    "matched_alias": "power point",
                    "score": 0.79,
                },
            ],
        }
        saved_option = {
            "id": "open_app_1",
            "intent": "OS_APP_OPEN",
            "action": "",
            "args": {"app_name": "powershell.exe"},
        }
        session_memory.remember_clarification_choice(
            "app_name_ambiguous",
            "open app power",
            saved_option,
            language="en",
        )

        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.intent_confidence.resolve_app_request", return_value=ambiguous
        ), patch("core.command_router._dispatch", return_value=(True, "Opening powershell.exe.", {"target": "powershell.exe"})):
            response = route_command("open app power")

        pending = session_memory.get_pending_clarification()
        self.assertIsNone(pending)
        self.assertIn("Opening powershell.exe", response)

    def test_language_switch_bypasses_pending_clarification(self):
        pending_payload = {
            "reason": "app_name_ambiguous",
            "prompt": "Reply with the app number or cancel.",
            "options": [
                {
                    "id": "open_app_1",
                    "label": "Google Chrome",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "chrome.exe"},
                    "reply_tokens": ["chrome", "1"],
                }
            ],
            "source_text": "open app browser",
            "language": "en",
        }
        session_memory.set_pending_clarification(pending_payload)

        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False):
            response = route_command("Switch language to Arabic.")

        self.assertEqual(session_memory.get_preferred_language(), "ar")
        self.assertIsNone(session_memory.get_pending_clarification())
        self.assertIn("Preferred language: ar", response)

    def test_observability_report_includes_phase2_clarification_kpis(self):
        metrics.record_command("OS_APP_OPEN", True, 0.05, language="en")
        metrics.record_clarification_event(
            "requested",
            intent="OS_APP_OPEN",
            language="en",
            reason="app_name_ambiguous",
            source_text="power",
            wrong_action_prevented=True,
        )
        metrics.record_clarification_event(
            "resolved",
            intent="OS_APP_OPEN",
            language="en",
            reason="app_name_ambiguous",
            source_text="power",
            retry_count=1,
        )

        report = metrics.format_observability_report()
        self.assertIn("Clarification Metrics:", report)
        self.assertIn("first_retry_success", report)
        self.assertIn("wrong_action_prevented=1", report)
        self.assertIn("power", report)


if __name__ == "__main__":
    unittest.main()
