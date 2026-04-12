import unittest
from unittest import mock

import httpx

from core.command_parser import ParsedCommand
import core.command_router as command_router
import llm.ollama_client as ollama_client
import llm.prompt_builder as prompt_builder


class _FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = int(status_code)
        self._payload = dict(payload)
        self.text = str(payload)

    def json(self):
        return dict(self._payload)


class Phase1V3RuntimeTests(unittest.TestCase):
    def test_ask_llm_uses_pinned_qwen_3b_model(self):
        with mock.patch(
            "llm.ollama_client.httpx.post",
            return_value=_FakeResponse(200, {"response": "ok"}),
        ) as mocked_post:
            result = ollama_client.ask_llm("hello")

        self.assertEqual(result, "ok")
        self.assertEqual(mocked_post.call_count, 1)
        call_kwargs = mocked_post.call_args.kwargs
        self.assertEqual(call_kwargs["json"]["model"], "qwen2.5:3b")

    def test_ask_llm_timeout_returns_without_fallback(self):
        with mock.patch(
            "llm.ollama_client.httpx.post",
            side_effect=httpx.TimeoutException("timeout"),
        ) as mocked_post:
            result = ollama_client.ask_llm("hello")

        self.assertIn("timed out", result.lower())
        self.assertEqual(mocked_post.call_count, 1)

    def test_prompt_contains_egyptian_arabic_guidance(self):
        with mock.patch.object(
            prompt_builder.session_memory,
            "build_context",
            return_value="",
        ), mock.patch.object(
            prompt_builder.session_memory,
            "context_snapshot",
            return_value={},
        ), mock.patch.object(
            prompt_builder.knowledge_base_service,
            "retrieve_for_prompt",
            return_value={"context": "", "sources": [], "results": []},
        ):
            package = prompt_builder.build_prompt_package(
                "قل لي اخبار مصر النهارده",
                response_language="ar",
            )

        prompt = package["prompt"]
        self.assertIn("Egyptian Arabic (Masri)", prompt)
        self.assertIn("Reply in Arabic only", prompt)

    def test_nlu_skip_for_llm_query_is_enabled_by_default(self):
        candidate = ParsedCommand(
            intent="LLM_QUERY",
            raw="",
            normalized="",
            action="",
            args={},
        )
        with mock.patch.object(command_router, "NLU_LLM_QUERY_EXTRACTION_ENABLED", False):
            self.assertTrue(command_router._should_skip_nlu_llm_query(candidate))

        with mock.patch.object(command_router, "NLU_LLM_QUERY_EXTRACTION_ENABLED", True):
            self.assertFalse(command_router._should_skip_nlu_llm_query(candidate))

    def test_keyword_nlp_maps_open_youtube_to_system_command(self):
        candidate = ParsedCommand(
            intent="LLM_QUERY",
            raw="يوتيوب",
            normalized="يوتيوب",
            action="",
            args={},
        )
        with mock.patch.object(
            command_router,
            "_classify_keyword_intent",
            return_value={
                "intent": "open_youtube",
                "confidence": 0.91,
                "matched_keywords": ["يوتيوب"],
            },
        ):
            parsed, meta = command_router._try_keyword_nlp_routing("يوتيوب", candidate)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "browser_open_url")
        self.assertEqual(parsed.args.get("url"), "https://www.youtube.com")
        self.assertTrue(meta.get("nlp_used"))
        self.assertTrue(meta.get("nlp_accepted"))

    def test_keyword_nlp_maps_search_and_extracts_query(self):
        candidate = ParsedCommand(
            intent="LLM_QUERY",
            raw="دور على اسعار الذهب",
            normalized="دور على اسعار الذهب",
            action="",
            args={},
        )
        with mock.patch.object(
            command_router,
            "_classify_keyword_intent",
            return_value={
                "intent": "search",
                "confidence": 0.78,
                "matched_keywords": ["دور"],
            },
        ):
            parsed, meta = command_router._try_keyword_nlp_routing("دور على اسعار الذهب", candidate)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "browser_search_web")
        self.assertEqual(parsed.args.get("search_query"), "اسعار الذهب")
        self.assertTrue(meta.get("nlp_accepted"))

    def test_keyword_nlp_unknown_falls_back_to_llm_query(self):
        candidate = ParsedCommand(
            intent="LLM_QUERY",
            raw="asdf qwer",
            normalized="asdf qwer",
            action="",
            args={},
        )
        with mock.patch.object(
            command_router,
            "_classify_keyword_intent",
            return_value={
                "intent": "unknown",
                "confidence": 0.0,
                "matched_keywords": [],
            },
        ):
            parsed, meta = command_router._try_keyword_nlp_routing("asdf qwer", candidate)

        self.assertIsNone(parsed)
        self.assertTrue(meta.get("nlp_used"))
        self.assertFalse(meta.get("nlp_accepted"))


if __name__ == "__main__":
    unittest.main()
