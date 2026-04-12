import unittest
from unittest import mock

from stt import dual_transcriber
from stt import stt_engine
from utils import language_detector


class SttRewriteCoreTests(unittest.TestCase):
    def test_language_detector_ar(self):
        self.assertEqual(language_detector.detect_language("اهلا بيك"), "ar")

    def test_language_detector_en(self):
        self.assertEqual(language_detector.detect_language("hello world"), "en")

    def test_language_detector_mixed(self):
        self.assertEqual(language_detector.detect_language("hello يا صاحبي"), "mixed")

    def test_language_detector_unknown(self):
        self.assertEqual(language_detector.detect_language("1234 !!!"), "unknown")

    def test_stt_engine_uses_auto_method_for_strong_text(self):
        with mock.patch.object(stt_engine, "_transcribe_auto", return_value="hello world"):
            result = stt_engine.transcribe("dummy.wav")

        self.assertEqual(result["method"], "auto")
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["text"], "hello world")

    def test_stt_engine_uses_dual_method_for_weak_text(self):
        with mock.patch.object(stt_engine, "_transcribe_auto", return_value=""), mock.patch(
            "stt.dual_transcriber.dual_transcribe",
            return_value=("اهلا", "ar"),
        ):
            result = stt_engine.transcribe("dummy.wav")

        self.assertEqual(result["method"], "dual")
        self.assertEqual(result["language"], "ar")
        self.assertEqual(result["text"], "اهلا")

    def test_dual_transcriber_prefers_higher_quality_text(self):
        with mock.patch.object(
            dual_transcriber,
            "_transcribe_with_language",
            side_effect=["افتح الكاميرا", "open"],
        ):
            text, language = dual_transcriber.dual_transcribe("dummy.wav")

        self.assertEqual(text, "افتح الكاميرا")
        self.assertEqual(language, "ar")


if __name__ == "__main__":
    unittest.main()
