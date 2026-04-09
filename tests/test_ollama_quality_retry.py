import unittest
from unittest.mock import patch

from llm import ollama_client


class _FakeResponse:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = int(status_code)
        self._payload = dict(payload or {})
        self.text = str(text or "")

    def json(self):
        return dict(self._payload)


class OllamaQualityRetryTests(unittest.TestCase):
    def test_low_value_reply_uses_fallback_model(self):
        responses = [
            _FakeResponse(
                200,
                {
                    "response": "Yes, I can help with that. Please provide me with some information.",
                },
            ),
            _FakeResponse(
                200,
                {
                    "response": "Top headlines right now: markets are mixed, energy prices are stable, and regional weather alerts remain active.",
                },
            ),
        ]

        with patch("llm.ollama_client._candidate_models", return_value=["qwen2.5:0.5b", "qwen2.5:1.5b"]), patch(
            "llm.ollama_client.httpx.post", side_effect=responses
        ) as post_mock:
            answer = ollama_client.ask_llm("USER:\nTell me the world news now.\nASSISTANT:")

        self.assertIn("Top headlines", answer)
        self.assertEqual(post_mock.call_count, 2)

    def test_low_value_reply_keeps_best_effort_when_fallback_fails(self):
        responses = [
            _FakeResponse(
                200,
                {
                    "response": "بالطبع، يمكنني القيام بذلك. هل لديك أي أسئلة أخرى؟",
                },
            ),
            _FakeResponse(500, {"error": "internal_error"}, text="internal_error"),
        ]

        with patch("llm.ollama_client._candidate_models", return_value=["qwen2.5:0.5b", "qwen2.5:1.5b"]), patch(
            "llm.ollama_client.httpx.post", side_effect=responses
        ):
            answer = ollama_client.ask_llm("USER:\nاخبار الطقس ايه انهارده\nASSISTANT:")

        self.assertIn("بالطبع", answer)

    def test_weather_refusal_reply_uses_fallback_model(self):
        responses = [
            _FakeResponse(
                200,
                {
                    "response": "I'm sorry, but I can't provide current weather information. Please check a weather service.",
                },
            ),
            _FakeResponse(
                200,
                {
                    "response": "I cannot see live weather right now, but if it is warm wear breathable layers, if mild carry a light jacket, and if cold use a coat.",
                },
            ),
        ]

        with patch("llm.ollama_client._candidate_models", return_value=["qwen2.5:0.5b", "qwen2.5:1.5b"]), patch(
            "llm.ollama_client.httpx.post", side_effect=responses
        ) as post_mock:
            answer = ollama_client.ask_llm("USER:\nwhat should i wear today?\nASSISTANT:")

        self.assertIn("light jacket", answer)
        self.assertEqual(post_mock.call_count, 2)

    def test_low_value_reply_skips_weaker_fallback_model(self):
        responses = [
            _FakeResponse(
                200,
                {
                    "response": "I can help with that. Please provide me with some information.",
                },
            ),
            _FakeResponse(
                200,
                {
                    "response": "This fallback response should not be used.",
                },
            ),
        ]

        with patch("llm.ollama_client._candidate_models", return_value=["qwen2.5:1.5b", "qwen2.5:0.5b"]), patch(
            "llm.ollama_client.httpx.post", side_effect=responses
        ) as post_mock:
            answer = ollama_client.ask_llm("USER:\nTell me the world news now.\nASSISTANT:")

        self.assertIn("Please provide me", answer)
        self.assertEqual(post_mock.call_count, 1)

    def test_intent_extraction_prompt_skips_quality_retry(self):
        responses = [
            _FakeResponse(
                200,
                {
                    "response": '{"intent":"LLM_QUERY","action":"","args":{},"confidence":0.41}',
                },
            ),
            _FakeResponse(200, {"response": "should not be used"}),
        ]

        prompt = "\n".join(
            [
                "SYSTEM:",
                "You are a strict intent extraction engine for a local Windows assistant.",
                "OUTPUT SCHEMA:",
                '{"intent":"...","action":"...","args":{},"confidence":0.0}',
                "ALLOWED INTENTS:",
                "- LLM_QUERY",
                "USER:",
                "turn off wifi",
            ]
        )

        with patch("llm.ollama_client._candidate_models", return_value=["qwen2.5:0.5b", "qwen2.5:1.5b"]), patch(
            "llm.ollama_client.httpx.post", side_effect=responses
        ) as post_mock:
            answer = ollama_client.ask_llm(prompt)

        self.assertIn('"intent"', answer)
        self.assertEqual(post_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
