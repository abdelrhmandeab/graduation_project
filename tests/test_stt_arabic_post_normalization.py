import unittest
from unittest.mock import patch

from audio import stt as stt_runtime


class SttArabicPostNormalizationTests(unittest.TestCase):
    def test_normalize_egyptian_spelling_variants(self):
        raw = "عاوز افتحلي نوتباد دلوقتى واطفي الواى فاى"
        normalized = stt_runtime.normalize_arabic_post_transcript(raw)

        self.assertIn("عايز", normalized)
        self.assertIn("افتح لي", normalized)
        self.assertIn("نوت باد", normalized)
        self.assertIn("دلوقتي", normalized)
        self.assertIn("واطفي", normalized)
        self.assertIn("واي فاي", normalized)

    def test_normalize_noisy_egyptian_news_words(self):
        raw = "أريدك تتلاني عن أصعار الذهب في البورسة"
        normalized = stt_runtime.normalize_arabic_post_transcript(raw)

        self.assertIn("عايزك", normalized)
        self.assertIn("تقولي", normalized)
        self.assertIn("اسعار", normalized)
        self.assertIn("الدهب", normalized)
        self.assertIn("البورصة", normalized)

    def test_non_arabic_text_is_unchanged(self):
        raw = "open notepad quickly"
        normalized = stt_runtime.normalize_arabic_post_transcript(raw)
        self.assertEqual(normalized, raw)

    def test_runtime_applies_arabic_post_normalization(self):
        with patch("audio.stt.STT_ARABIC_POST_NORMALIZATION", True), patch(
            "audio.stt.transcribe_backend_direct_with_meta",
            return_value={"text": "عاوز افتحلي نوتباد دلوقتى", "language": "ar", "language_confidence": 0.9, "backend": "faster_whisper"},
        ):
            payload = stt_runtime.transcribe_streaming_with_meta("dummy.wav")

        text = str((payload or {}).get("text") or "")
        self.assertIn("عايز", text)
        self.assertIn("افتح لي", text)
        self.assertIn("نوت باد", text)
        self.assertIn("دلوقتي", text)


if __name__ == "__main__":
    unittest.main()
