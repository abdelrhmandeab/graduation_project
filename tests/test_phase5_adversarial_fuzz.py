import copy
import random
import string
import unittest
from unittest.mock import patch

from core.command_router import _rewrite_followup_command, route_command
from core.intent_confidence import resolve_clarification_reply
from core.session_memory import session_memory


class Phase5AdversarialFuzzTests(unittest.TestCase):
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

    def test_fuzz_unrelated_clarification_replies_do_not_resolve(self):
        payload = {
            "prompt": "pick app",
            "language": "en",
            "options": [
                {
                    "id": "open_app_1",
                    "label": "Google Chrome (chrome.exe)",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "chrome.exe"},
                    "reply_tokens": ["1", "chrome"],
                },
                {
                    "id": "open_app_2",
                    "label": "Microsoft Edge (msedge.exe)",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "msedge.exe"},
                    "reply_tokens": ["2", "edge"],
                },
            ],
        }

        rng = random.Random(42)
        alphabet = "qwxvzjktmnrhs"
        for _ in range(120):
            noise = "".join(rng.choice(alphabet) for _ in range(rng.randint(5, 14)))
            result = resolve_clarification_reply(noise, payload)
            self.assertNotEqual(result.status, "resolved", msg=f"noise resolved unexpectedly: {noise}")

    def test_context_drift_blocks_destructive_followup_when_reference_stale(self):
        session_memory.set_last_file(r"C:\\Users\\abdel\\Desktop\\report.txt")

        with patch("core.command_router._is_fresh_reference", return_value=False):
            rewritten, meta = _rewrite_followup_command("delete the file", language="en")

        self.assertEqual(rewritten, "delete the file")
        self.assertTrue(meta.get("followup_blocked"))
        self.assertIn("too old", str(meta.get("followup_message") or "").lower())

    def test_random_noise_during_clarification_keeps_pending_state(self):
        pending_payload = {
            "reason": "app_name_ambiguous",
            "prompt": "I found multiple app matches.",
            "options": [
                {
                    "id": "open_app_1",
                    "label": "Google Chrome (chrome.exe)",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "chrome.exe"},
                    "reply_tokens": ["1", "chrome"],
                },
                {
                    "id": "open_app_2",
                    "label": "Microsoft Edge (msedge.exe)",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "msedge.exe"},
                    "reply_tokens": ["2", "edge"],
                },
            ],
            "source_text": "open browser",
            "language": "en",
        }
        session_memory.set_pending_clarification(pending_payload)

        rng = random.Random(7)
        first_noise = "".join(rng.choice(string.ascii_lowercase) for _ in range(18))
        second_noise = "".join(rng.choice(string.ascii_lowercase) for _ in range(20))

        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False):
            first = route_command(first_noise)
            second = route_command(second_noise)

        self.assertIn("multiple app matches", first.lower())
        self.assertIn("direct selection", second.lower())
        self.assertIsNotNone(session_memory.get_pending_clarification())


if __name__ == "__main__":
    unittest.main()
