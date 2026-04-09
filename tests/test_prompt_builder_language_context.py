import copy
import unittest
from unittest.mock import patch

from core.session_memory import session_memory
from llm.prompt_builder import build_prompt_package


class PromptBuilderLanguageContextTests(unittest.TestCase):
    def setUp(self):
        with session_memory._lock:
            self._saved_turns = copy.deepcopy(session_memory._turns)
            self._saved_language = str(session_memory._preferred_language)
            self._saved_pending = copy.deepcopy(session_memory._pending_clarification)
            self._saved_slots = copy.deepcopy(session_memory._context_slots)

        session_memory.clear()
        session_memory.set_preferred_language("en")

    def tearDown(self):
        with session_memory._lock:
            session_memory._turns = self._saved_turns
            session_memory._preferred_language = self._saved_language
            session_memory._pending_clarification = self._saved_pending
            session_memory._context_slots = self._saved_slots
            session_memory._save()

    def _patch_prompt_dependencies(self):
        return patch("llm.prompt_builder.knowledge_base_service.retrieve_for_prompt", return_value={"context": "", "sources": [], "results": []})

    def test_arabic_prompt_uses_only_arabic_llm_memory(self):
        session_memory.add_turn("open chrome", "Opening Chrome.", language="en", intent="OS_APP_OPEN")
        session_memory.add_turn("Tell me a quick tip", "Use strong passwords.", language="en", intent="LLM_QUERY")
        session_memory.add_turn("احكيلي نصيحة سريعة", "خليك منظم وحدد اهداف يومية.", language="ar", intent="LLM_QUERY")

        with self._patch_prompt_dependencies(), patch("llm.prompt_builder.persona_manager.get_system_prompt", return_value="You are Jarvis."):
            package = build_prompt_package("عايز نصيحة تانية", response_language="ar")

        prompt = str(package.get("prompt") or "")
        self.assertIn("خليك منظم", prompt)
        self.assertNotIn("Use strong passwords", prompt)
        self.assertNotIn("Opening Chrome", prompt)

    def test_english_prompt_uses_only_english_llm_memory(self):
        session_memory.add_turn("open notepad", "Opening Notepad.", language="en", intent="OS_APP_OPEN")
        session_memory.add_turn("احكيلي معلومة", "دي معلومة بالعربي.", language="ar", intent="LLM_QUERY")
        session_memory.add_turn("Tell me one fact", "Water boils at 100C at sea level.", language="en", intent="LLM_QUERY")

        with self._patch_prompt_dependencies(), patch("llm.prompt_builder.persona_manager.get_system_prompt", return_value="You are Jarvis."):
            package = build_prompt_package("Tell me another fact", response_language="en")

        prompt = str(package.get("prompt") or "")
        self.assertIn("Water boils at 100C", prompt)
        self.assertNotIn("دي معلومة بالعربي", prompt)
        self.assertNotIn("Opening Notepad", prompt)


if __name__ == "__main__":
    unittest.main()
