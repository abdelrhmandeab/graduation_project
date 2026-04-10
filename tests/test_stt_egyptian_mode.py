import unittest
from unittest.mock import patch

from audio import stt as stt_runtime
from core import orchestrator


class SttEgyptianDialectModeTests(unittest.TestCase):
    def test_resolve_whisper_language_keeps_auto_and_honors_explicit(self):
        self.assertIsNone(stt_runtime._resolve_whisper_language(None))
        self.assertIsNone(stt_runtime._resolve_whisper_language("auto"))
        self.assertEqual(stt_runtime._resolve_whisper_language("en"), "en")
        self.assertEqual(stt_runtime._resolve_whisper_language("ar"), "ar")

    def test_runtime_coerces_arabic_language_when_text_is_arabic(self):
        with patch(
            "audio.stt.transcribe_backend_direct_with_meta",
            return_value={"text": "أريد أخبار البورسة اليوم", "language": "en", "language_confidence": 0.55, "backend": "faster_whisper"},
        ):
            payload = stt_runtime.transcribe_streaming_with_meta("dummy.wav")

        text = str((payload or {}).get("text") or "")
        language = str((payload or {}).get("language") or "")
        self.assertEqual(language, "ar")
        self.assertIn("عايز", text)
        self.assertIn("البورصة", text)

    def test_orchestrator_keeps_auto_hint_and_uses_single_pass_for_good_arabic(self):
        with patch("core.orchestrator.transcribe_streaming", return_value="عايز اخبار البورصة") as transcribe_mock, patch(
            "core.orchestrator._extract_detected_language_from_stt", return_value="ar"
        ):
            text, language = orchestrator._transcribe_with_auto_then_english_retry("dummy.wav")

        self.assertEqual(text, "عايز اخبار البورصة")
        self.assertEqual(language, "ar")
        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertIsNone(transcribe_mock.call_args.kwargs.get("language_hint"))


if __name__ == "__main__":
    unittest.main()
