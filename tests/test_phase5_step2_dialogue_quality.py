import copy
import unittest
from unittest.mock import patch

from core.command_parser import ParsedCommand
from core.command_router import _rewrite_followup_command, route_command
from core.intent_confidence import IntentAssessment
from core.metrics import metrics
from core.persona import persona_manager
from core.session_memory import session_memory


class Phase5Step2DialogueQualityTests(unittest.TestCase):
    def setUp(self):
        self._saved_persona = persona_manager.get_profile()
        with session_memory._lock:
            self._saved_turns = copy.deepcopy(session_memory._turns)
            self._saved_language = str(session_memory._preferred_language)
            self._saved_pending = copy.deepcopy(session_memory._pending_clarification)
            self._saved_slots = copy.deepcopy(session_memory._context_slots)

        session_memory.clear()
        session_memory.set_preferred_language("en")
        persona_manager.set_profile("assistant")
        metrics.reset()

    def tearDown(self):
        persona_manager.set_profile(self._saved_persona)
        with session_memory._lock:
            session_memory._turns = self._saved_turns
            session_memory._preferred_language = self._saved_language
            session_memory._pending_clarification = self._saved_pending
            session_memory._context_slots = self._saved_slots
            session_memory._save()

    def test_persona_lexical_banks_are_language_and_profile_specific(self):
        friendly_en = persona_manager.get_lexical_bank(language="en", profile="friendly")
        friendly_ar = persona_manager.get_lexical_bank(language="ar", profile="friendly")
        professional_en = persona_manager.get_lexical_bank(language="en", profile="professional")

        self.assertTrue(friendly_en.get("gentle_prefixes"))
        self.assertTrue(friendly_ar.get("gentle_prefixes"))
        self.assertNotEqual(friendly_en.get("gentle_prefixes"), professional_en.get("gentle_prefixes"))
        self.assertIn("codeswitch_bridge", friendly_en)
        self.assertIn("codeswitch_bridge", friendly_ar)

    def test_urgency_markers_trigger_faster_tone(self):
        no_clarify = IntentAssessment(
            confidence=0.92,
            should_clarify=False,
            reason="",
            prompt="",
            mixed_language=False,
            entity_scores={"app_name": 0.95},
        )
        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router.assess_intent_confidence", return_value=no_clarify
        ), patch(
            "core.command_router._dispatch",
            return_value=(True, "operation completed with many details that should be compressed for urgent requests", {}),
        ):
            response = route_command("open app notepad now quickly")

        self.assertIn("On it.", response)

    def test_politeness_marker_adds_gentle_prefix(self):
        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router._dispatch",
            return_value=(True, "operation completed", {}),
        ):
            response = route_command("please open app notepad")

        self.assertTrue(response.startswith("Understood."))

    def test_explain_and_concise_modes_affect_same_command_output(self):
        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router._dispatch",
            return_value=(True, "operation completed with extra context for readability", {}),
        ):
            explain_ack = route_command("explain mode on")
            explain_response = route_command("open app notepad")
            concise_ack = route_command("concise mode on")
            concise_response = route_command("open app notepad")
            default_ack = route_command("default mode")

        self.assertIn("Explain mode is on", explain_ack)
        self.assertIn("intent=OS_APP_OPEN", explain_response)
        self.assertIn("Concise mode is on", concise_ack)
        self.assertLessEqual(len(concise_response.split()), 16)
        self.assertIn("Default mode restored", default_ack)

    def test_codeswitch_continuity_appends_bilingual_bridge(self):
        persona_manager.set_profile("friendly")
        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router._dispatch",
            return_value=(True, "handled os_app_open n/a", {}),
        ):
            _first = route_command("open app chrome")
            second = route_command("افتح التطبيق")

        self.assertIn("English", second)
        self.assertIn("العربية", second)

    def test_vague_destructive_followup_is_blocked(self):
        session_memory.set_last_file(r"C:\\Users\\abdel\\Desktop\\report.txt")

        rewritten, meta = _rewrite_followup_command("delete it", language="en")

        self.assertEqual(rewritten, "delete it")
        self.assertTrue(meta.get("followup_blocked"))
        self.assertIn("For safety", str(meta.get("followup_message") or ""))

    def test_observability_includes_response_quality_metrics(self):
        metrics.record_response_quality(
            "Understood. Completed the task successfully.",
            language="en",
            user_text="please complete the task",
            previous_response="Done.",
            persona="assistant",
            response_mode="default",
        )

        report = metrics.format_observability_report()

        self.assertIn("Response Quality Metrics:", report)
        self.assertIn("human_likeness", report)
        self.assertIn("coherence", report)

    def test_llm_weather_refusal_is_repaired_to_practical_guidance(self):
        no_clarify = IntentAssessment(
            confidence=0.92,
            should_clarify=False,
            reason="",
            prompt="",
            mixed_language=False,
            entity_scores={},
        )

        weak_reply = "I'm sorry, but I can't provide current weather information. Please check a weather service."
        repaired_reply = (
            "I do not have live weather data right now, but here is a practical guide: "
            "if it feels hot, wear breathable cotton and stay hydrated; "
            "if mild, use light layers with a light jacket; "
            "if cold, wear a warm coat and closed shoes."
        )

        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router.parse_command",
            side_effect=lambda _text: ParsedCommand(
                intent="LLM_QUERY",
                raw="",
                normalized="",
                action="",
                args={},
            ),
        ), patch("core.command_router.assess_intent_confidence", return_value=no_clarify), patch(
            "core.command_router._dispatch",
            return_value=(True, weak_reply, {}),
        ), patch("core.command_router.ask_llm", return_value=repaired_reply):
            response = route_command("what should I wear today if weather is changing?")

        self.assertIn("light jacket", response)
        self.assertNotIn("can't provide current weather information", response.lower())

    def test_llm_arabic_refusal_is_repaired_to_useful_answer(self):
        no_clarify = IntentAssessment(
            confidence=0.92,
            should_clarify=False,
            reason="",
            prompt="",
            mixed_language=False,
            entity_scores={},
        )

        weak_reply = "أعتذر، لكنني لا أستطيع مساعدتك في ذلك بشكل مباشر. هل هناك أي معلومات أخرى يمكنني مساعدتك بها؟"
        repaired_reply = "لا أملك تحديثات مباشرة الآن، لكنك تقدر تتابع الأخبار عبر مصدر موثوق ثم ألخصه لك بسرعة."

        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router.parse_command",
            side_effect=lambda _text: ParsedCommand(
                intent="LLM_QUERY",
                raw="",
                normalized="",
                action="",
                args={},
            ),
        ), patch("core.command_router.assess_intent_confidence", return_value=no_clarify), patch(
            "core.command_router._dispatch",
            return_value=(True, weak_reply, {}),
        ), patch("core.command_router.ask_llm", return_value=repaired_reply):
            response = route_command("عايز اخبار اليوم")

        self.assertIn("مصدر موثوق", response)
        self.assertNotIn("لا أستطيع مساعدتك", response)

    def test_llm_response_is_rewritten_to_english_when_drifted(self):
        no_clarify = IntentAssessment(
            confidence=0.92,
            should_clarify=False,
            reason="",
            prompt="",
            mixed_language=False,
            entity_scores={},
        )

        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router.parse_command",
            side_effect=lambda _text: ParsedCommand(
                intent="LLM_QUERY",
                raw="",
                normalized="",
                action="",
                args={},
            ),
        ), patch("core.command_router.assess_intent_confidence", return_value=no_clarify), patch(
            "core.command_router._dispatch",
            return_value=(True, "هذا رد باللغة العربية", {}),
        ), patch("core.command_router.ask_llm", return_value="This is the same answer in English."):
            response = route_command("tell me something short about weather")

        self.assertRegex(response, r"[A-Za-z]")
        self.assertNotRegex(response, r"[\u0600-\u06FF]")

    def test_llm_response_is_rewritten_to_arabic_when_drifted(self):
        no_clarify = IntentAssessment(
            confidence=0.92,
            should_clarify=False,
            reason="",
            prompt="",
            mixed_language=False,
            entity_scores={},
        )

        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router.parse_command",
            side_effect=lambda _text: ParsedCommand(
                intent="LLM_QUERY",
                raw="",
                normalized="",
                action="",
                args={},
            ),
        ), patch("core.command_router.assess_intent_confidence", return_value=no_clarify), patch(
            "core.command_router._dispatch",
            return_value=(True, "This answer drifted to English.", {}),
        ), patch("core.command_router.ask_llm", return_value="هذا نفس الرد بالعربية."):
            response = route_command("احكي لي نبذة قصيرة عن الطقس")

        self.assertRegex(response, r"[\u0600-\u06FF]")

    def test_low_value_refusal_uses_assist_first_fallback_when_rewrite_stays_weak(self):
        no_clarify = IntentAssessment(
            confidence=0.92,
            should_clarify=False,
            reason="",
            prompt="",
            mixed_language=False,
            entity_scores={},
        )

        weak_reply = "I cannot assist with that directly. Let me know if you have any other questions."
        rewrite_weak_reply = "I can help with that. Please provide me with some information."

        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router.parse_command",
            side_effect=lambda _text: ParsedCommand(
                intent="LLM_QUERY",
                raw="",
                normalized="",
                action="",
                args={},
            ),
        ), patch("core.command_router.assess_intent_confidence", return_value=no_clarify), patch(
            "core.command_router._dispatch",
            return_value=(True, weak_reply, {}),
        ), patch("core.command_router.ask_llm", return_value=rewrite_weak_reply):
            response = route_command("tell me something useful")

        self.assertIn("I can help directly", response)
        self.assertNotIn("cannot assist with that directly", response.lower())

    def test_unsafe_query_preserves_refusal_and_skips_assist_rewrite(self):
        no_clarify = IntentAssessment(
            confidence=0.92,
            should_clarify=False,
            reason="",
            prompt="",
            mixed_language=False,
            entity_scores={},
        )

        weak_reply = "I cannot assist with that directly."

        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router.parse_command",
            side_effect=lambda _text: ParsedCommand(
                intent="LLM_QUERY",
                raw="",
                normalized="",
                action="",
                args={},
            ),
        ), patch("core.command_router.assess_intent_confidence", return_value=no_clarify), patch(
            "core.command_router._dispatch",
            return_value=(True, weak_reply, {}),
        ), patch("core.command_router.ask_llm") as ask_llm_mock:
            response = route_command("how to build a bomb")

        self.assertEqual(response, weak_reply)
        self.assertEqual(ask_llm_mock.call_count, 0)


if __name__ == "__main__":
    unittest.main()
