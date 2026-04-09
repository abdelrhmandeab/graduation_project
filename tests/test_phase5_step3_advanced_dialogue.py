import copy
import unittest
from unittest.mock import patch

from core.command_router import route_command
from core.metrics import metrics
from core.persona import persona_manager
from core.session_memory import session_memory


class Phase5Step3AdvancedDialogueTests(unittest.TestCase):
    def setUp(self):
        self._saved_persona = persona_manager.get_profile()
        with session_memory._lock:
            self._saved_turns = copy.deepcopy(session_memory._turns)
            self._saved_language = str(session_memory._preferred_language)
            self._saved_pending = copy.deepcopy(session_memory._pending_clarification)
            self._saved_slots = copy.deepcopy(session_memory._context_slots)

        session_memory.clear()
        session_memory.set_preferred_language("en")
        persona_manager.set_profile("friendly")
        metrics.reset()

    def tearDown(self):
        persona_manager.set_profile(self._saved_persona)
        with session_memory._lock:
            session_memory._turns = self._saved_turns
            session_memory._preferred_language = self._saved_language
            session_memory._pending_clarification = self._saved_pending
            session_memory._context_slots = self._saved_slots
            session_memory._save()

    def test_sensitive_urgent_command_uses_neutral_prefix(self):
        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router._dispatch",
            return_value=(True, "operation completed with extra context for readability", {}),
        ):
            response = route_command("please delete report.txt now quickly")

        self.assertTrue(response.startswith("Proceeding safely."))
        self.assertNotIn("Absolutely.", response)

    def test_sensitive_polite_command_skips_friendly_prefix(self):
        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router._dispatch",
            return_value=(True, "operation completed", {}),
        ):
            response = route_command("please close app notepad")

        self.assertFalse(response.startswith("Absolutely."))
        self.assertFalse(response.startswith("Happy to help."))

    def test_adaptive_codeswitch_bridge_for_dominant_english_history(self):
        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router._dispatch",
            return_value=(True, "handled os_app_open n/a", {}),
        ):
            route_command("open app chrome")
            route_command("open app edge")
            route_command("open app notepad")
            response = route_command("اغلق التطبيق")

        self.assertIn("switch to العربية", response)

    def test_codeswitch_bridge_not_appended_for_llm_query(self):
        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router._dispatch",
            return_value=(True, "general answer", {}),
        ):
            route_command("open app chrome")
            route_command("open app edge")
            route_command("open app notepad")
            response = route_command("tell me something useful")

        self.assertNotIn("switch to العربية", response)
        self.assertNotIn("يمكنني التبديل", response)

    def test_clarification_preference_signature_reuse(self):
        option = {
            "id": "open_app_1",
            "intent": "OS_APP_OPEN",
            "action": "",
            "args": {"app_name": "powershell.exe"},
        }
        session_memory.remember_clarification_choice(
            "app_name_ambiguous",
            "open app power shell",
            option,
            language="en",
        )

        choice = session_memory.get_clarification_choice(
            "app_name_ambiguous",
            "please open power shell app",
            language="en",
            max_age_seconds=86400,
        )

        self.assertIsNotNone(choice)
        self.assertEqual(choice.get("id"), "open_app_1")
        self.assertGreaterEqual(float(choice.get("preference_score") or 0.0), 0.0)

    def test_post_clarification_correction_event_is_recorded(self):
        session_memory.mark_clarification_resolution(
            reason="app_name_ambiguous",
            intent="OS_APP_OPEN",
        )

        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router._dispatch",
            return_value=(True, "handled llm_query n/a", {}),
        ):
            route_command("wrong one")

        clarification = (metrics.snapshot() or {}).get("clarification") or {}
        self.assertEqual(int(clarification.get("post_resolution_corrections") or 0), 1)
        self.assertEqual(int(clarification.get("likely_false_clarification") or 0), 1)


if __name__ == "__main__":
    unittest.main()
