import unittest
from unittest.mock import patch

from core.command_parser import ParsedCommand
from core.command_router import route_command
from core.intent_confidence import assess_intent_confidence, resolve_clarification_reply
from core.session_memory import session_memory


class Phase2IntentReliabilityTests(unittest.TestCase):
    def setUp(self):
        self._previous_language = session_memory.get_preferred_language()
        self._previous_pending = session_memory.get_pending_clarification()
        session_memory.clear_pending_clarification()
        session_memory.clear_clarification_preferences()
        session_memory.clear_app_usage()
        session_memory.set_preferred_language("en")

    def tearDown(self):
        session_memory.clear_pending_clarification()
        if self._previous_pending:
            session_memory.set_pending_clarification(self._previous_pending)
        session_memory.set_preferred_language(self._previous_language)

    def test_known_app_alias_has_high_entity_confidence(self):
        parsed = ParsedCommand(
            intent="OS_APP_OPEN",
            raw="open app cmd",
            normalized="open app cmd",
            action="",
            args={"app_name": "cmd"},
        )

        assessment = assess_intent_confidence("open app cmd", parsed, language="en")

        self.assertFalse(assessment.should_clarify)
        self.assertGreaterEqual(float(assessment.entity_scores.get("app_name") or 0.0), 0.90)

    def test_unknown_app_target_triggers_low_entity_clarification(self):
        parsed = ParsedCommand(
            intent="OS_APP_OPEN",
            raw="open app qqqnonexistent",
            normalized="open app qqqnonexistent",
            action="",
            args={"app_name": "qqqnonexistent"},
        )

        assessment = assess_intent_confidence("open app qqqnonexistent", parsed, language="en")

        self.assertTrue(assessment.should_clarify)
        self.assertEqual(assessment.reason, "low_entity_confidence")
        self.assertTrue(assessment.prompt)

    def test_short_unclear_llm_query_triggers_clarification(self):
        parsed = ParsedCommand(
            intent="LLM_QUERY",
            raw="أتو السوق",
            normalized="أتو السوق",
            action="",
            args={},
        )

        assessment = assess_intent_confidence("أتو السوق", parsed, language="ar")

        self.assertTrue(assessment.should_clarify)
        self.assertEqual(assessment.reason, "low_confidence_unclear_query")
        self.assertIn("طلب واضح", str(assessment.prompt or ""))

    def test_short_greeting_llm_query_does_not_clarify(self):
        parsed = ParsedCommand(
            intent="LLM_QUERY",
            raw="مرحبا",
            normalized="مرحبا",
            action="",
            args={},
        )

        assessment = assess_intent_confidence("مرحبا", parsed, language="ar")

        self.assertFalse(assessment.should_clarify)

    def test_clarification_reply_accepts_ordinal_words(self):
        payload = {
            "prompt": "choose one",
            "language": "en",
            "options": [
                {"id": "open_app", "label": "Open app", "reply_tokens": ["app"]},
                {"id": "open_path", "label": "Open folder", "reply_tokens": ["folder"]},
                {"id": "cancel", "label": "Cancel", "reply_tokens": ["cancel"]},
            ],
        }

        first_pick = resolve_clarification_reply("first", payload)
        second_pick = resolve_clarification_reply("الثاني", payload)

        self.assertEqual(first_pick.status, "resolved")
        self.assertEqual((first_pick.option or {}).get("id"), "open_app")
        self.assertEqual(second_pick.status, "resolved")
        self.assertEqual((second_pick.option or {}).get("id"), "open_path")

    def test_clarification_reply_arabic_no_thanks_cancels(self):
        payload = {
            "prompt": "اختر واحد",
            "language": "ar",
            "options": [
                {"id": "open_app", "label": "فتح التطبيق", "reply_tokens": ["تطبيق"]},
                {"id": "open_folder", "label": "فتح المجلد", "reply_tokens": ["مجلد"]},
            ],
        }

        result = resolve_clarification_reply("لأ شكر", payload)

        self.assertEqual(result.status, "cancelled")

    def test_route_file_search_multimatch_sets_clarification(self):
        fake_matches = [
            r"C:\\Users\\abdel\\Desktop\\report.txt",
            r"C:\\Users\\abdel\\Documents\\report.txt",
        ]
        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router.search_index_service.start"
        ), patch("core.command_router.search_index_service.search", return_value=[]), patch(
            "core.command_router.find_files", return_value=fake_matches
        ):
            response = route_command("find file report in desktop")

        pending = session_memory.get_pending_clarification()
        self.assertIsNotNone(pending)
        self.assertEqual(pending.get("reason"), "file_search_multiple_matches")
        self.assertGreaterEqual(len(pending.get("options") or []), 2)
        self.assertIn("multiple files", response.lower())

    def test_route_app_ambiguous_sets_clarification(self):
        ambiguous = {
            "status": "ambiguous",
            "query": "power",
            "candidates": [
                {
                    "canonical_name": "PowerShell",
                    "executable": "powershell.exe",
                    "matched_alias": "power shell",
                    "score": 0.80,
                },
                {
                    "canonical_name": "PowerPoint",
                    "executable": "powerpnt.exe",
                    "matched_alias": "power point",
                    "score": 0.76,
                },
            ],
        }
        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.intent_confidence.resolve_app_request", return_value=ambiguous
        ), patch("core.command_router.resolve_app_request", return_value=ambiguous):
            response = route_command("open app power")

        pending = session_memory.get_pending_clarification()
        self.assertIsNotNone(pending)
        self.assertEqual(pending.get("reason"), "app_name_ambiguous")
        self.assertGreaterEqual(len(pending.get("options") or []), 2)
        self.assertIn("multiple app matches", response.lower())

    def test_route_parser_fastpath_skips_nlu_for_clear_open_command(self):
        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", True), patch(
            "core.command_router.NLU_PARSER_FASTPATH_ENABLED", True
        ), patch("core.command_router.classify_with_nlu") as nlu_mock, patch(
            "core.command_router._dispatch",
            return_value=(True, "Opening Firefox.", {}),
        ):
            response = route_command("open firefox")

        self.assertIn("Opening Firefox", response)
        self.assertEqual(nlu_mock.call_count, 0)


if __name__ == "__main__":
    unittest.main()
