import unittest

from audio import wake_word


class WakeWordPhase2Tests(unittest.TestCase):
    def setUp(self):
        self._saved_phrase_settings = wake_word.get_runtime_wake_word_phrase_settings()
        wake_word._ar_consecutive_hits = 0
        wake_word._ar_last_hit_ts = 0.0

    def tearDown(self):
        wake_word.set_runtime_wake_word_phrase_settings(
            mode=self._saved_phrase_settings.get("mode"),
            arabic_enabled=self._saved_phrase_settings.get("arabic_enabled"),
            arabic_triggers=self._saved_phrase_settings.get("arabic_triggers"),
            ar_stt_model=self._saved_phrase_settings.get("ar_stt_model"),
            ar_chunk_seconds=self._saved_phrase_settings.get("ar_chunk_seconds"),
            ar_check_interval_seconds=self._saved_phrase_settings.get("ar_check_interval_seconds"),
            ar_consecutive_hits_required=self._saved_phrase_settings.get("ar_consecutive_hits_required"),
            ar_confirm_window_seconds=self._saved_phrase_settings.get("ar_confirm_window_seconds"),
        )
        wake_word._ar_consecutive_hits = 0
        wake_word._ar_last_hit_ts = 0.0

    def test_wake_mode_aliases(self):
        mode = wake_word.set_runtime_wake_mode("ar")
        self.assertEqual(mode, "arabic")
        mode = wake_word.set_runtime_wake_mode("en")
        self.assertEqual(mode, "english")
        mode = wake_word.set_runtime_wake_mode("bilingual")
        self.assertEqual(mode, "both")

    def test_add_and_remove_wake_triggers(self):
        wake_word.set_runtime_wake_word_phrase_settings(arabic_triggers=["jarvis"])

        added, triggers = wake_word.add_runtime_wake_trigger("ya jarvis")
        self.assertTrue(added)
        self.assertIn("ya jarvis", triggers)

        duplicate_added, _ = wake_word.add_runtime_wake_trigger("YA   JARVIS")
        self.assertFalse(duplicate_added)

        removed, triggers_after = wake_word.remove_runtime_wake_trigger("ya jarvis")
        self.assertTrue(removed)
        self.assertNotIn("ya jarvis", triggers_after)

    def test_default_step2_trigger_pack_contains_required_phrases(self):
        snapshot = wake_word.get_runtime_wake_word_phrase_settings()
        triggers = list(snapshot.get("arabic_triggers") or [])
        normalized = {wake_word._normalize_trigger_text(item) for item in triggers}

        required = {
            wake_word._normalize_trigger_text("hey jarvis"),
            wake_word._normalize_trigger_text("hello jarvis"),
            wake_word._normalize_trigger_text("jarvis"),
            wake_word._normalize_trigger_text("اهلا جارفيس"),
            wake_word._normalize_trigger_text("مرحبا جارفيس"),
            wake_word._normalize_trigger_text("يا جارفيس"),
            wake_word._normalize_trigger_text("جارفيس"),
        }
        self.assertTrue(required.issubset(normalized))

    def test_arabic_trigger_match_normalization(self):
        transcript = "\u064a\u064e\u0627 \u062c\u064e\u0627\u0631\u0650\u0641\u064a\u0633"
        triggers = ["\u064a\u0627 \u062c\u0627\u0631\u0641\u064a\u0633"]
        matched = wake_word._match_arabic_trigger(transcript, triggers)
        self.assertEqual(matched, triggers[0])

    def test_multilingual_phrase_match_english_and_arabic(self):
        triggers = ["hello jarvis", "اهلا جارفيس"]

        english = wake_word._match_arabic_trigger("hello jarvis", triggers)
        arabic = wake_word._match_arabic_trigger("اهلا يا جارفيس", triggers)

        self.assertEqual(english, "hello jarvis")
        self.assertEqual(arabic, "اهلا جارفيس")

    def test_single_word_trigger_guard_reduces_false_positives(self):
        triggers = ["jarvis", "جارفيس"]

        long_transcript_match = wake_word._match_arabic_trigger(
            "please search about jarvis architecture in detail",
            triggers,
        )
        short_transcript_match = wake_word._match_arabic_trigger("jarvis", triggers)

        self.assertEqual(long_transcript_match, "")
        self.assertEqual(short_transcript_match, "jarvis")

    def test_consecutive_hit_guard(self):
        first = wake_word._register_arabic_hit(
            hit_detected=True,
            now_ts=10.0,
            required_hits=2,
            confirm_window_seconds=3.0,
        )
        second = wake_word._register_arabic_hit(
            hit_detected=True,
            now_ts=12.0,
            required_hits=2,
            confirm_window_seconds=3.0,
        )
        self.assertFalse(first)
        self.assertTrue(second)

    def test_consecutive_hit_expires_outside_window(self):
        first = wake_word._register_arabic_hit(
            hit_detected=True,
            now_ts=20.0,
            required_hits=2,
            confirm_window_seconds=3.0,
        )
        second = wake_word._register_arabic_hit(
            hit_detected=True,
            now_ts=24.5,
            required_hits=2,
            confirm_window_seconds=3.0,
        )
        self.assertFalse(first)
        self.assertFalse(second)


if __name__ == "__main__":
    unittest.main()
