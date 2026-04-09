import unittest
from unittest.mock import patch

import core.command_classifier as command_classifier


class NluPostNormalizationTests(unittest.TestCase):
    def setUp(self):
        command_classifier.clear_nlu_cache()

    @patch("core.command_classifier.ask_llm")
    def test_schedule_phrase_overrides_wrong_intent(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"screenshot"},"confidence":0.99}'
        )

        result = command_classifier.classify_with_nlu(
            "in 5 minutes mute volume",
            language="en",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("intent"), "JOB_QUEUE_COMMAND")
        self.assertEqual(result.get("action"), "enqueue")
        self.assertEqual(result.get("args", {}).get("delay_seconds"), 300)
        self.assertEqual(result.get("args", {}).get("command_text"), "mute volume")

    @patch("core.command_classifier.ask_llm")
    def test_media_app_launch_overrides_media_control(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"media_play_pause"},"confidence":0.96}'
        )

        result = command_classifier.classify_with_nlu(
            "play music on spotify",
            language="en",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("intent"), "OS_APP_OPEN")
        self.assertEqual(result.get("args", {}).get("app_name"), "spotify")

    @patch("core.command_classifier.ask_llm")
    def test_browser_back_overrides_wrong_seek_action(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"media_seek_backward"},"confidence":0.92}'
        )

        result = command_classifier.classify_with_nlu(
            "go back in the browser",
            language="en",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("intent"), "OS_SYSTEM_COMMAND")
        self.assertEqual(result.get("args", {}).get("action_key"), "browser_back")

    @patch("core.command_classifier.ask_llm")
    def test_spoken_number_normalization(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"volume_set","volume_level":"forty"},"confidence":0.93}'
        )

        result = command_classifier.classify_with_nlu(
            "set volume to forty percent",
            language="en",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("args", {}).get("volume_level"), 40)

    @patch("core.command_classifier.ask_llm")
    def test_cache_reuses_frequent_intent(self, mock_ask):
        if not command_classifier.NLU_INTENT_CACHE_ENABLED:
            self.skipTest("NLU intent cache disabled")

        mock_ask.return_value = (
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"wifi_off"},"confidence":0.95}'
        )

        first = command_classifier.classify_with_nlu("turn off wifi", language="en")
        second = command_classifier.classify_with_nlu("turn off wifi", language="en")

        self.assertTrue(first.get("ok"))
        self.assertTrue(second.get("ok"))
        self.assertFalse(first.get("cache_hit"))
        self.assertTrue(second.get("cache_hit"))
        self.assertEqual(mock_ask.call_count, 1)

        stats = command_classifier.get_nlu_cache_stats()
        self.assertGreaterEqual(int(stats.get("hits") or 0), 1)

    @patch("core.command_classifier.ask_llm")
    def test_dnd_phrase_maps_to_notifications_off(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"wifi_on"},"confidence":0.89}'
        )

        result = command_classifier.classify_with_nlu(
            "turn on do not disturb",
            language="en",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("intent"), "OS_SYSTEM_COMMAND")
        self.assertEqual(result.get("args", {}).get("action_key"), "notifications_off")

    @patch("core.command_classifier.ask_llm")
    def test_app_alias_normalizes_arabic_firefox_name(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"OS_APP_OPEN","action":"","args":{"app_name":"فايرفوكس"},"confidence":0.94}'
        )

        result = command_classifier.classify_with_nlu(
            "افتح فايرفوكس",
            language="ar",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("intent"), "OS_APP_OPEN")
        self.assertEqual(result.get("args", {}).get("app_name"), "firefox")

    @patch("core.command_classifier.ask_llm")
    def test_speech_off_requires_explicit_toggle_phrase(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"VOICE_COMMAND","action":"speech_off","args":{},"confidence":0.95}'
        )

        result = command_classifier.classify_with_nlu(
            "تشهر الوصول على إيراني",
            language="ar",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("intent"), "LLM_QUERY")
        self.assertEqual(result.get("action"), "")

    @patch("core.command_classifier.ask_llm")
    def test_speech_off_allows_explicit_arabic_toggle_phrase(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"VOICE_COMMAND","action":"speech_off","args":{},"confidence":0.95}'
        )

        result = command_classifier.classify_with_nlu(
            "اطفي الصوت",
            language="ar",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("intent"), "VOICE_COMMAND")
        self.assertEqual(result.get("action"), "speech_off")

    @patch("core.command_classifier.ask_llm")
    def test_empty_voice_command_is_downgraded_to_llm_query(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"VOICE_COMMAND","action":"","args":{},"confidence":0.96}'
        )

        result = command_classifier.classify_with_nlu(
            "أخبر عن أخبار الحرب من إيران وأمريكا",
            language="ar",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("intent"), "LLM_QUERY")
        self.assertEqual(result.get("action"), "")
        self.assertEqual(result.get("args"), {})

    @patch("core.command_classifier.ask_llm")
    def test_arabic_informational_query_blocks_hallucinated_volume_set(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"volume_set","volume_level":40},"confidence":0.96}'
        )

        result = command_classifier.classify_with_nlu(
            "أريدك أن تخبرني عن الوصول بين إيران وأمريكا",
            language="ar",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("intent"), "LLM_QUERY")
        self.assertEqual(result.get("action"), "")
        self.assertEqual(result.get("args"), {})

    @patch("core.command_classifier.ask_llm")
    def test_informational_query_blocks_hallucinated_shutdown(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"shutdown"},"confidence":0.97}'
        )

        result = command_classifier.classify_with_nlu(
            "tell me why this app keeps crashing",
            language="en",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("intent"), "LLM_QUERY")
        self.assertEqual(result.get("action"), "")
        self.assertEqual(result.get("args"), {})

    @patch("core.command_classifier.ask_llm")
    def test_unrelated_text_blocks_hallucinated_notifications_toggle(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"notifications_off"},"confidence":0.89}'
        )

        result = command_classifier.classify_with_nlu(
            "طبقة الأوزون مهمة جدا للحياة",
            language="ar",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("intent"), "LLM_QUERY")
        self.assertEqual(result.get("action"), "")
        self.assertEqual(result.get("args"), {})

    @patch("core.command_classifier.ask_llm")
    def test_negated_shutdown_phrase_is_not_executed(self, mock_ask):
        mock_ask.return_value = (
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"shutdown"},"confidence":0.97}'
        )

        result = command_classifier.classify_with_nlu(
            "No no no, don't turn off the PC.",
            language="en",
            use_cache=False,
        )

        self.assertTrue(result.get("ok"))
        self.assertEqual(result.get("intent"), "LLM_QUERY")
        self.assertEqual(result.get("action"), "")
        self.assertEqual(result.get("args"), {})


if __name__ == "__main__":
    unittest.main()
