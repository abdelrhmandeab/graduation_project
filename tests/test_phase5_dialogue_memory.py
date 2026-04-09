import copy
import time
import unittest
from unittest.mock import patch

from core.command_router import _build_file_search_runtime_clarification, _rewrite_followup_command, route_command
from core.persona import persona_manager
from core.session_memory import session_memory


class Phase5DialogueMemoryTests(unittest.TestCase):
    def setUp(self):
        self._saved_persona = persona_manager.get_profile()
        with session_memory._lock:
            self._saved_turns = copy.deepcopy(session_memory._turns)
            self._saved_language = str(session_memory._preferred_language)
            self._saved_pending = copy.deepcopy(session_memory._pending_clarification)
            self._saved_slots = copy.deepcopy(session_memory._context_slots)

        session_memory.clear()
        session_memory.set_preferred_language("en")

    def tearDown(self):
        persona_manager.set_profile(self._saved_persona)
        with session_memory._lock:
            session_memory._turns = self._saved_turns
            session_memory._preferred_language = self._saved_language
            session_memory._pending_clarification = self._saved_pending
            session_memory._context_slots = self._saved_slots
            session_memory._save()

    def _route_two_success_turns_with_persona(self, profile, response_text="Done."):
        ok, _msg = persona_manager.set_profile(profile)
        self.assertTrue(ok)
        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router._dispatch", return_value=(True, response_text, {})
        ):
            first = route_command("hello there")
            second = route_command("hello there")
        return first, second

    def test_persona_profiles_include_phase5_targets(self):
        profiles = set(persona_manager.list_profiles())
        self.assertTrue({"professional", "friendly", "brief"}.issubset(profiles))

    def test_explicit_open_last_app_followup(self):
        session_memory.set_last_app("notepad")

        rewritten, meta = _rewrite_followup_command("open the app", language="en")

        self.assertEqual(rewritten, "open app notepad")
        self.assertEqual(meta.get("followup_rewrite"), "open_last_app")

    def test_explicit_open_last_file_followup(self):
        session_memory.set_last_file(r"C:\\Users\\abdel\\Desktop\\notes.txt")

        rewritten, meta = _rewrite_followup_command("open the file", language="en")

        self.assertEqual(rewritten, r"file info C:\\Users\\abdel\\Desktop\\notes.txt")
        self.assertEqual(meta.get("followup_rewrite"), "file_info_last_file")

    def test_arabic_delete_last_file_followup(self):
        session_memory.set_last_file(r"C:\\Users\\abdel\\Desktop\\todo.txt")

        rewritten, meta = _rewrite_followup_command("احذف الملف", language="ar")

        self.assertEqual(rewritten, r"delete C:\\Users\\abdel\\Desktop\\todo.txt")
        self.assertEqual(meta.get("followup_rewrite"), "delete_last_file")

    def test_open_it_uses_most_recent_reference(self):
        session_memory.set_last_app("notepad")
        session_memory.set_last_file(r"C:\\Users\\abdel\\Desktop\\new.txt")

        rewritten, meta = _rewrite_followup_command("open it", language="en")

        self.assertEqual(rewritten, r"file info C:\\Users\\abdel\\Desktop\\new.txt")
        self.assertEqual(meta.get("followup_rewrite"), "file_info_last_file")

    def test_stale_reference_blocks_followup_guessing(self):
        session_memory.set_last_app("notepad")

        with patch("core.command_router._is_fresh_reference", return_value=False):
            rewritten, meta = _rewrite_followup_command("close the app", language="en")

        self.assertEqual(rewritten, "close the app")
        self.assertTrue(meta.get("followup_blocked"))
        self.assertIn("too old", str(meta.get("followup_message") or "").lower())

    def test_open_it_without_context_requests_reference(self):
        rewritten, meta = _rewrite_followup_command("open it", language="en")

        self.assertEqual(rewritten, "open it")
        self.assertTrue(meta.get("followup_blocked"))
        self.assertIn("recent app or file", str(meta.get("followup_message") or "").lower())

    def test_implicit_yes_confirms_pending_token(self):
        session_memory.set_pending_confirmation_token("abc123")

        rewritten, meta = _rewrite_followup_command("yes", language="en")

        self.assertEqual(rewritten, "confirm abc123")
        self.assertEqual(meta.get("followup_rewrite"), "confirmation_implicit_yes")
        self.assertEqual(meta.get("token"), "abc123")

    def test_implicit_yes_with_second_factor_confirms_pending_token(self):
        session_memory.set_pending_confirmation_token("abc123")

        rewritten, meta = _rewrite_followup_command("yes 1234", language="en")

        self.assertEqual(rewritten, "confirm abc123 1234")
        self.assertEqual(meta.get("followup_rewrite"), "confirmation_implicit_yes")
        self.assertEqual(meta.get("token"), "abc123")

    def test_implicit_no_cancels_pending_token(self):
        session_memory.set_pending_confirmation_token("abc123")

        rewritten, meta = _rewrite_followup_command("no", language="en")

        self.assertEqual(rewritten, "no")
        self.assertTrue(meta.get("followup_cancel_confirmation"))
        self.assertEqual(meta.get("followup_rewrite"), "confirmation_implicit_no")
        self.assertEqual(meta.get("token"), "abc123")

    def test_arabic_implicit_yes_confirms_pending_token(self):
        session_memory.set_pending_confirmation_token("abc123")

        rewritten, meta = _rewrite_followup_command("نعم", language="ar")

        self.assertEqual(rewritten, "confirm abc123")
        self.assertEqual(meta.get("followup_rewrite"), "confirmation_implicit_yes")

    def test_yes_without_pending_token_is_not_rewritten(self):
        rewritten, meta = _rewrite_followup_command("yes", language="en")

        self.assertEqual(rewritten, "yes")
        self.assertEqual(meta, {})

    def test_open_it_conflict_between_equally_recent_app_and_file_prompts_disambiguation(self):
        now_ts = time.time()
        with session_memory._lock:
            session_memory._context_slots["last_app"] = "notepad"
            session_memory._context_slots["last_app_updated_at"] = now_ts
            session_memory._context_slots["last_file"] = r"C:\\Users\\abdel\\Desktop\\notes.txt"
            session_memory._context_slots["last_file_updated_at"] = now_ts

        with patch("core.command_router.FOLLOWUP_REFERENCE_CONFLICT_WINDOW_SECONDS", 2.0):
            rewritten, meta = _rewrite_followup_command("open it", language="en")

        self.assertEqual(rewritten, "open it")
        self.assertTrue(meta.get("followup_blocked"))
        self.assertIn("recent app and file", str(meta.get("followup_message") or "").lower())

    def test_open_both_executes_two_recent_targets(self):
        session_memory.set_last_app("notepad")
        session_memory.set_last_app("calc")

        def _dispatch_side_effect(parsed, *args, **kwargs):
            return True, f"{parsed.intent}:{(parsed.args or {}).get('app_name')}", {}

        with patch("core.command_router._dispatch", side_effect=_dispatch_side_effect) as dispatch_mock:
            response = route_command("open both")

        self.assertEqual(dispatch_mock.call_count, 2)
        self.assertIn("OS_APP_OPEN:calc", response)
        self.assertIn("OS_APP_OPEN:notepad", response)

    def test_close_them_executes_two_recent_apps(self):
        session_memory.set_last_app("notepad")
        session_memory.set_last_app("calc")

        def _dispatch_side_effect(parsed, *args, **kwargs):
            return True, f"{parsed.intent}:{(parsed.args or {}).get('app_name')}", {}

        with patch("core.command_router._dispatch", side_effect=_dispatch_side_effect) as dispatch_mock:
            response = route_command("close them")

        self.assertEqual(dispatch_mock.call_count, 2)
        self.assertIn("OS_APP_CLOSE:calc", response)
        self.assertIn("OS_APP_CLOSE:notepad", response)

    def test_session_memory_slot_decay_confidence_is_slot_specific(self):
        stale_ts = time.time() - 1000

        pending_conf = session_memory.slot_confidence("pending_confirmation_token", updated_at=stale_ts)
        app_conf = session_memory.slot_confidence("last_app", updated_at=stale_ts)

        self.assertEqual(pending_conf, 0.0)
        self.assertGreater(app_conf, 0.0)

    def test_runtime_clarification_prompt_includes_confidence_feedback(self):
        prompt, payload = _build_file_search_runtime_clarification(
            "report.txt",
            [
                r"C:\\Users\\abdel\\Desktop\\report.txt",
                r"C:\\Users\\abdel\\Documents\\report.txt",
            ],
        )

        self.assertIn("confident", prompt.lower())
        self.assertTrue(str(payload.get("prompt") or "").startswith("I am"))

    def test_route_clarification_show_more_switches_to_next_page(self):
        _prompt, payload = _build_file_search_runtime_clarification(
            "report.txt",
            [
                r"C:\\Users\\abdel\\Desktop\\report.txt",
                r"C:\\Users\\abdel\\Documents\\report.txt",
                r"C:\\Users\\abdel\\Downloads\\report.txt",
                r"C:\\Users\\abdel\\Music\\report.txt",
                r"C:\\Users\\abdel\\Videos\\report.txt",
                r"C:\\Users\\abdel\\Pictures\\report.txt",
            ],
        )
        session_memory.set_pending_clarification(payload)

        response = route_command("show more")
        pending = session_memory.get_pending_clarification()

        self.assertIn("page 2 of 2", response.lower())
        self.assertEqual(int((pending or {}).get("page_index") or 0), 1)
        self.assertEqual(len((pending or {}).get("options") or []), 1)

    def test_low_confidence_unclear_pending_is_replaced_by_new_substantive_query(self):
        pending_payload = {
            "reason": "low_confidence_unclear_query",
            "prompt": "I did not fully catch that short phrase.",
            "options": [],
            "source_text": "جربز",
            "language": "ar",
            "confidence": 0.36,
        }
        session_memory.set_pending_clarification(pending_payload)

        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router._dispatch",
            return_value=(True, "The ozone layer absorbs harmful UV radiation.", {}),
        ) as dispatch_mock:
            response = route_command("Explain to me how the ozone layer works.")

        self.assertEqual(dispatch_mock.call_count, 1)
        self.assertIn("ozone layer", response.lower())
        self.assertIsNone(session_memory.get_pending_clarification())

    def test_low_confidence_unclear_retry_hint_uses_clear_request_wording(self):
        pending_payload = {
            "reason": "low_confidence_unclear_query",
            "prompt": "I did not fully catch that short phrase.",
            "options": [],
            "source_text": "jrvs",
            "language": "en",
            "confidence": 0.36,
        }
        session_memory.set_pending_clarification(pending_payload)

        with patch("core.command_router.CLARIFICATION_FALLBACK_AFTER_MISSES", 1):
            response = route_command("hmm")

        self.assertIn("clear request", response.lower())
        self.assertNotIn("direct selection", response.lower())

    def test_route_clarification_none_of_these_cancels_pending(self):
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

        response = route_command("none of these")

        self.assertIn("none of these", response.lower())
        self.assertIsNone(session_memory.get_pending_clarification())

    def test_route_clarification_this_one_selects_first_option(self):
        pending_payload = {
            "reason": "open_target_ambiguous",
            "prompt": "Did you mean app or folder?",
            "options": [
                {
                    "id": "open_path",
                    "label": "Open folder 'notes'",
                    "intent": "OS_FILE_NAVIGATION",
                    "action": "list_directory",
                    "args": {"path": "notes"},
                    "reply_tokens": ["1", "folder"],
                },
                {
                    "id": "open_app",
                    "label": "Open application 'notes'",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "notes"},
                    "reply_tokens": ["2", "app"],
                },
            ],
            "source_text": "open notes",
            "language": "en",
        }
        session_memory.set_pending_clarification(pending_payload)

        with patch("core.command_router._dispatch", return_value=(True, "Opening folder notes", {})) as dispatch_mock:
            response = route_command("this one")

        parsed = dispatch_mock.call_args.args[0]
        self.assertEqual(parsed.intent, "OS_FILE_NAVIGATION")
        self.assertIn("Opening folder notes", response)

    def test_route_clarification_fuzzy_name_match_resolves_option(self):
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
                    "reply_tokens": ["1", "chrome", "google chrome"],
                },
                {
                    "id": "open_app_2",
                    "label": "Microsoft Edge (msedge.exe)",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "msedge.exe"},
                    "reply_tokens": ["2", "edge", "microsoft edge"],
                },
            ],
            "source_text": "open browser",
            "language": "en",
        }
        session_memory.set_pending_clarification(pending_payload)

        with patch("core.command_router._dispatch", return_value=(True, "Opening chrome", {})) as dispatch_mock:
            response = route_command("chrmoe")

        parsed = dispatch_mock.call_args.args[0]
        self.assertEqual((parsed.args or {}).get("app_name"), "chrome.exe")
        self.assertIn("Opening chrome", response)

    def test_brief_persona_enforces_word_budget_on_llm_query(self):
        ok, _msg = persona_manager.set_profile("brief")
        self.assertTrue(ok)
        long_response = (
            "This response intentionally contains many words so we can verify that the router "
            "enforces a strict brief persona length limit before returning it to the user."
        )

        with patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False), patch(
            "core.command_router._dispatch", return_value=(True, long_response, {})
        ):
            response = route_command("tell me a summary")

        self.assertTrue(response.endswith("..."))
        self.assertLessEqual(len(response.split()), 16)

    def test_route_clarification_yes_selects_first_binary_option(self):
        pending_payload = {
            "reason": "open_target_ambiguous",
            "prompt": "Did you mean app or folder?",
            "options": [
                {
                    "id": "open_path",
                    "label": "Open folder 'notes'",
                    "intent": "OS_FILE_NAVIGATION",
                    "action": "list_directory",
                    "args": {"path": "notes"},
                    "reply_tokens": ["1", "folder"],
                },
                {
                    "id": "open_app",
                    "label": "Open application 'notes'",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "notes"},
                    "reply_tokens": ["2", "app"],
                },
            ],
            "source_text": "open notes",
            "language": "en",
        }
        session_memory.set_pending_clarification(pending_payload)

        with patch("core.command_router._dispatch", return_value=(True, "Opening folder notes", {})) as dispatch_mock:
            response = route_command("yes")

        parsed = dispatch_mock.call_args.args[0]
        self.assertEqual(parsed.intent, "OS_FILE_NAVIGATION")
        self.assertEqual(parsed.action, "list_directory")
        self.assertIn("Opening folder notes", response)
        self.assertIsNone(session_memory.get_pending_clarification())

    def test_route_clarification_arabic_no_selects_second_binary_option(self):
        pending_payload = {
            "reason": "open_target_ambiguous",
            "prompt": "هل تقصد تطبيق ام مجلد؟",
            "options": [
                {
                    "id": "open_path",
                    "label": "فتح المجلد notes",
                    "intent": "OS_FILE_NAVIGATION",
                    "action": "list_directory",
                    "args": {"path": "notes"},
                    "reply_tokens": ["1", "مجلد"],
                },
                {
                    "id": "open_app",
                    "label": "فتح التطبيق notes",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "notes"},
                    "reply_tokens": ["2", "تطبيق"],
                },
            ],
            "source_text": "افتح notes",
            "language": "ar",
        }
        session_memory.set_pending_clarification(pending_payload)
        session_memory.set_preferred_language("ar")

        with patch("core.command_router._dispatch", return_value=(True, "جار فتح التطبيق", {})) as dispatch_mock:
            response = route_command("لا")

        parsed = dispatch_mock.call_args.args[0]
        self.assertEqual(parsed.intent, "OS_APP_OPEN")
        self.assertIn("جار فتح التطبيق", response)
        self.assertIsNone(session_memory.get_pending_clarification())

    def test_route_clarification_yes_selects_first_app_candidate(self):
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
                {
                    "id": "open_app_3",
                    "label": "Firefox (firefox.exe)",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "firefox.exe"},
                    "reply_tokens": ["3", "firefox"],
                },
            ],
            "source_text": "open browser",
            "language": "en",
        }
        session_memory.set_pending_clarification(pending_payload)

        with patch("core.command_router._dispatch", return_value=(True, "Opening chrome", {})) as dispatch_mock:
            response = route_command("yes")

        parsed = dispatch_mock.call_args.args[0]
        self.assertEqual(parsed.intent, "OS_APP_OPEN")
        self.assertEqual((parsed.args or {}).get("app_name"), "chrome.exe")
        self.assertIn("Opening chrome", response)
        self.assertIsNone(session_memory.get_pending_clarification())

    def test_route_clarification_no_selects_second_file_candidate(self):
        pending_payload = {
            "reason": "file_search_multiple_matches",
            "prompt": "I found multiple files.",
            "options": [
                {
                    "id": "file_match_1",
                    "label": r"C:\\Users\\abdel\\Desktop\\report.txt",
                    "intent": "OS_FILE_NAVIGATION",
                    "action": "file_info",
                    "args": {"path": r"C:\\Users\\abdel\\Desktop\\report.txt"},
                    "reply_tokens": ["1"],
                },
                {
                    "id": "file_match_2",
                    "label": r"C:\\Users\\abdel\\Documents\\report.txt",
                    "intent": "OS_FILE_NAVIGATION",
                    "action": "file_info",
                    "args": {"path": r"C:\\Users\\abdel\\Documents\\report.txt"},
                    "reply_tokens": ["2"],
                },
                {
                    "id": "file_match_3",
                    "label": r"C:\\Users\\abdel\\Downloads\\report.txt",
                    "intent": "OS_FILE_NAVIGATION",
                    "action": "file_info",
                    "args": {"path": r"C:\\Users\\abdel\\Downloads\\report.txt"},
                    "reply_tokens": ["3"],
                },
            ],
            "source_text": "report.txt",
            "language": "en",
        }
        session_memory.set_pending_clarification(pending_payload)

        with patch("core.command_router._dispatch", return_value=(True, "Showing second file", {})) as dispatch_mock:
            response = route_command("no")

        parsed = dispatch_mock.call_args.args[0]
        self.assertEqual(parsed.intent, "OS_FILE_NAVIGATION")
        self.assertEqual(parsed.action, "file_info")
        self.assertEqual(
            (parsed.args or {}).get("path"),
            r"C:\\Users\\abdel\\Documents\\report.txt",
        )
        self.assertIn("Showing second file", response)
        self.assertIsNone(session_memory.get_pending_clarification())

    def test_route_clarification_no_third_selects_third_app_candidate(self):
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
                {
                    "id": "open_app_3",
                    "label": "Firefox (firefox.exe)",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "firefox.exe"},
                    "reply_tokens": ["3", "firefox"],
                },
            ],
            "source_text": "open browser",
            "language": "en",
        }
        session_memory.set_pending_clarification(pending_payload)

        with patch("core.command_router._dispatch", return_value=(True, "Opening firefox", {})) as dispatch_mock:
            response = route_command("no third")

        parsed = dispatch_mock.call_args.args[0]
        self.assertEqual(parsed.intent, "OS_APP_OPEN")
        self.assertEqual((parsed.args or {}).get("app_name"), "firefox.exe")
        self.assertIn("Opening firefox", response)
        self.assertIsNone(session_memory.get_pending_clarification())

    def test_route_clarification_yes_3_selects_third_file_candidate(self):
        pending_payload = {
            "reason": "file_search_multiple_matches",
            "prompt": "I found multiple files.",
            "options": [
                {
                    "id": "file_match_1",
                    "label": r"C:\\Users\\abdel\\Desktop\\report.txt",
                    "intent": "OS_FILE_NAVIGATION",
                    "action": "file_info",
                    "args": {"path": r"C:\\Users\\abdel\\Desktop\\report.txt"},
                    "reply_tokens": ["1"],
                },
                {
                    "id": "file_match_2",
                    "label": r"C:\\Users\\abdel\\Documents\\report.txt",
                    "intent": "OS_FILE_NAVIGATION",
                    "action": "file_info",
                    "args": {"path": r"C:\\Users\\abdel\\Documents\\report.txt"},
                    "reply_tokens": ["2"],
                },
                {
                    "id": "file_match_3",
                    "label": r"C:\\Users\\abdel\\Downloads\\report.txt",
                    "intent": "OS_FILE_NAVIGATION",
                    "action": "file_info",
                    "args": {"path": r"C:\\Users\\abdel\\Downloads\\report.txt"},
                    "reply_tokens": ["3"],
                },
            ],
            "source_text": "report.txt",
            "language": "en",
        }
        session_memory.set_pending_clarification(pending_payload)

        with patch("core.command_router._dispatch", return_value=(True, "Showing third file", {})) as dispatch_mock:
            response = route_command("yes 3")

        parsed = dispatch_mock.call_args.args[0]
        self.assertEqual(parsed.intent, "OS_FILE_NAVIGATION")
        self.assertEqual(parsed.action, "file_info")
        self.assertEqual(
            (parsed.args or {}).get("path"),
            r"C:\\Users\\abdel\\Downloads\\report.txt",
        )
        self.assertIn("Showing third file", response)
        self.assertIsNone(session_memory.get_pending_clarification())

    def test_route_clarification_arabic_no_third_selects_third_option(self):
        pending_payload = {
            "reason": "app_name_ambiguous",
            "prompt": "وجدت عدة تطبيقات.",
            "options": [
                {
                    "id": "open_app_1",
                    "label": "كروم",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "chrome.exe"},
                    "reply_tokens": ["1", "كروم"],
                },
                {
                    "id": "open_app_2",
                    "label": "ايدج",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "msedge.exe"},
                    "reply_tokens": ["2", "ايدج"],
                },
                {
                    "id": "open_app_3",
                    "label": "فايرفوكس",
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": "firefox.exe"},
                    "reply_tokens": ["3", "فايرفوكس"],
                },
            ],
            "source_text": "افتح متصفح",
            "language": "ar",
        }
        session_memory.set_pending_clarification(pending_payload)
        session_memory.set_preferred_language("ar")

        with patch("core.command_router._dispatch", return_value=(True, "فتح فايرفوكس", {})) as dispatch_mock:
            response = route_command("لا الثالث")

        parsed = dispatch_mock.call_args.args[0]
        self.assertEqual(parsed.intent, "OS_APP_OPEN")
        self.assertEqual((parsed.args or {}).get("app_name"), "firefox.exe")
        self.assertIn("فتح فايرفوكس", response)
        self.assertIsNone(session_memory.get_pending_clarification())

    def test_persona_friendly_adds_friendly_prefix_on_repeat(self):
        first, second = self._route_two_success_turns_with_persona("friendly", response_text="Done.")

        self.assertEqual(first, "Done.")
        self.assertNotEqual(second, "Done.")
        self.assertTrue(second.startswith(("Happy to help. ", "Absolutely. ")))
        self.assertIn("Done.", second)

    def test_persona_brief_adds_brief_prefix_on_repeat(self):
        first, second = self._route_two_success_turns_with_persona("brief", response_text="Done.")

        self.assertEqual(first, "Done.")
        self.assertNotEqual(second, "Done.")
        self.assertTrue(second.startswith(("Noted. ", "Okay. ")))
        self.assertIn("Done.", second)

    def test_persona_professional_adds_professional_prefix_on_repeat(self):
        first, second = self._route_two_success_turns_with_persona("professional", response_text="Done.")

        self.assertEqual(first, "Done.")
        self.assertNotEqual(second, "Done.")
        self.assertTrue(second.startswith(("Certainly. ", "Noted. ")))
        self.assertIn("Done.", second)


if __name__ == "__main__":
    unittest.main()