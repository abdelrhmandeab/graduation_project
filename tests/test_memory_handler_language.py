import unittest
from unittest.mock import patch

from core.command_parser import ParsedCommand
from core.handlers import memory


class MemoryHandlerLanguageTests(unittest.TestCase):
    def test_set_language_arabic(self):
        parsed = ParsedCommand(
            intent="MEMORY_COMMAND",
            raw="language arabic",
            normalized="language arabic",
            action="set_language",
            args={"language": "arabic"},
        )

        with patch("core.handlers.memory.session_memory.set_preferred_language", return_value=(True, "ok")), patch(
            "core.handlers.memory.log_action"
        ) as log_action_mock:
            ok, message, meta = memory.handle(parsed)

        self.assertTrue(ok)
        self.assertEqual(message, "Preferred language: ar")
        self.assertEqual(meta.get("preferred_language"), "ar")
        self.assertEqual(log_action_mock.call_count, 1)

    def test_set_language_invalid_value(self):
        parsed = ParsedCommand(
            intent="MEMORY_COMMAND",
            raw="language french",
            normalized="language french",
            action="set_language",
            args={"language": "french"},
        )

        with patch("core.handlers.memory.session_memory.set_preferred_language") as set_language_mock:
            ok, message, meta = memory.handle(parsed)

        self.assertFalse(ok)
        self.assertIn("Unsupported language", message)
        self.assertEqual(meta, {})
        self.assertEqual(set_language_mock.call_count, 0)


if __name__ == "__main__":
    unittest.main()
