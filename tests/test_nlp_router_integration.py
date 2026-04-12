import unittest
from unittest import mock
from types import SimpleNamespace

import core.command_router as command_router
from core.command_parser import ParsedCommand
from core.language_gate import LanguageGateResult


class NlpRouterIntegrationTests(unittest.TestCase):
    def _run_route_with_nlp(self, text, nlp_result):
        parser_candidate = ParsedCommand(
            intent="LLM_QUERY",
            raw=text,
            normalized=" ".join(str(text).lower().split()),
            action="",
            args={},
        )
        captured = {}

        def fake_dispatch(parsed, *args, **kwargs):
            captured["parsed"] = parsed
            return True, "ok", {}

        with mock.patch.object(
            command_router,
            "parse_command",
            return_value=parser_candidate,
        ), mock.patch.object(
            command_router,
            "_classify_keyword_intent",
            return_value=nlp_result,
        ), mock.patch.object(
            command_router,
            "_dispatch",
            side_effect=fake_dispatch,
        ), mock.patch.object(
            command_router,
            "_rewrite_followup_command",
            return_value=(text, {}),
        ), mock.patch.object(
            command_router,
            "_analyze_tone_markers",
            return_value={},
        ), mock.patch.object(
            command_router,
            "_update_short_term_context",
        ), mock.patch.object(
            command_router,
            "_finalize_success_response",
            side_effect=lambda response, *args, **kwargs: response,
        ), mock.patch.object(
            command_router,
            "_should_store_turn",
            return_value=False,
        ), mock.patch.object(
            command_router,
            "assess_intent_confidence",
            return_value=SimpleNamespace(
                should_clarify=False,
                confidence=0.88,
                entity_scores={},
            ),
        ), mock.patch.object(
            command_router,
            "detect_supported_language",
            return_value=LanguageGateResult(
                supported=True,
                language="en",
                normalized_text=text,
                reason="ok",
            ),
        ), mock.patch.object(
            command_router,
            "NLU_INTENT_ROUTING_ENABLED",
            True,
        ), mock.patch.object(
            command_router,
            "NLU_LLM_QUERY_EXTRACTION_ENABLED",
            False,
        ), mock.patch.object(
            command_router.session_memory,
            "get_preferred_language",
            return_value="en",
        ), mock.patch.object(
            command_router.session_memory,
            "set_preferred_language",
        ), mock.patch.object(
            command_router.session_memory,
            "record_language_turn",
        ), mock.patch.object(
            command_router.session_memory,
            "get_pending_clarification",
            return_value=None,
        ), mock.patch.object(
            command_router.session_memory,
            "recent_clarification_resolution",
            return_value=None,
        ), mock.patch.object(
            command_router.session_memory,
            "add_turn",
        ):
            response = command_router.route_command(text)

        return response, captured.get("parsed")

    def test_route_command_maps_open_chrome_intent(self):
        response, parsed = self._run_route_with_nlp(
            "افتح كروم",
            {
                "intent": "open_chrome",
                "confidence": 0.84,
                "matched_keywords": ["افتح", "كروم"],
            },
        )

        self.assertEqual(response, "ok")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.intent, "OS_APP_OPEN")
        self.assertEqual(parsed.args.get("app_name"), "chrome")

    def test_route_command_maps_volume_up_intent(self):
        response, parsed = self._run_route_with_nlp(
            "ارفع الصوت",
            {
                "intent": "volume_up",
                "confidence": 0.80,
                "matched_keywords": ["ارفع", "الصوت"],
            },
        )

        self.assertEqual(response, "ok")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "volume_up")

    def test_route_command_maps_open_google_intent(self):
        response, parsed = self._run_route_with_nlp(
            "open google",
            {
                "intent": "open_google",
                "confidence": 0.79,
                "matched_keywords": ["open", "google"],
            },
        )

        self.assertEqual(response, "ok")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "browser_open_url")
        self.assertEqual(parsed.args.get("url"), "https://www.google.com")

    def test_route_command_maps_wifi_on_intent(self):
        response, parsed = self._run_route_with_nlp(
            "شغل الواي فاي",
            {
                "intent": "wifi_on",
                "confidence": 0.82,
                "matched_keywords": ["شغل", "واي فاي"],
            },
        )

        self.assertEqual(response, "ok")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "wifi_on")

    def test_route_command_maps_wifi_off_intent(self):
        response, parsed = self._run_route_with_nlp(
            "اطفي الواي فاي",
            {
                "intent": "wifi_off",
                "confidence": 0.83,
                "matched_keywords": ["اطفي", "واي فاي"],
            },
        )

        self.assertEqual(response, "ok")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "wifi_off")

    def test_route_command_maps_bluetooth_on_intent(self):
        response, parsed = self._run_route_with_nlp(
            "شغل البلوتوث",
            {
                "intent": "bluetooth_on",
                "confidence": 0.81,
                "matched_keywords": ["شغل", "بلوتوث"],
            },
        )

        self.assertEqual(response, "ok")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "bluetooth_on")

    def test_route_command_maps_bluetooth_off_intent(self):
        response, parsed = self._run_route_with_nlp(
            "اقفل البلوتوث",
            {
                "intent": "bluetooth_off",
                "confidence": 0.81,
                "matched_keywords": ["اقفل", "بلوتوث"],
            },
        )

        self.assertEqual(response, "ok")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "bluetooth_off")

    def test_route_command_maps_screenshot_intent(self):
        response, parsed = self._run_route_with_nlp(
            "خد سكرين شوت",
            {
                "intent": "screenshot",
                "confidence": 0.86,
                "matched_keywords": ["خد", "سكرين شوت"],
            },
        )

        self.assertEqual(response, "ok")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "screenshot")

    def test_route_command_low_confidence_nlp_keeps_llm_query(self):
        response, parsed = self._run_route_with_nlp(
            "open chrome",
            {
                "intent": "open_chrome",
                "confidence": 0.20,
                "matched_keywords": ["open", "chrome"],
            },
        )

        self.assertEqual(response, "ok")
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.intent, "LLM_QUERY")


if __name__ == "__main__":
    unittest.main()
