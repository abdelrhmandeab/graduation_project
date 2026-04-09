import copy
import unittest
from unittest.mock import patch

from audio import vad


class VadRuntimeTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            "_SILERO_READY": vad._SILERO_READY,
            "_silero_model": vad._silero_model,
            "_silero_get_speech_timestamps": vad._silero_get_speech_timestamps,
            "_silero_torch": vad._silero_torch,
            "_SILERO_ERROR": vad._SILERO_ERROR,
            "_SILERO_RUNTIME_BROKEN": vad._SILERO_RUNTIME_BROKEN,
            "_SILERO_FALLBACK_NOTICE_EMITTED": vad._SILERO_FALLBACK_NOTICE_EMITTED,
        }

    def tearDown(self):
        vad._SILERO_READY = self._saved["_SILERO_READY"]
        vad._silero_model = self._saved["_silero_model"]
        vad._silero_get_speech_timestamps = self._saved["_silero_get_speech_timestamps"]
        vad._silero_torch = self._saved["_silero_torch"]
        vad._SILERO_ERROR = self._saved["_SILERO_ERROR"]
        vad._SILERO_RUNTIME_BROKEN = self._saved["_SILERO_RUNTIME_BROKEN"]
        vad._SILERO_FALLBACK_NOTICE_EMITTED = self._saved["_SILERO_FALLBACK_NOTICE_EMITTED"]

    def test_is_speech_uses_custom_audio_reader_for_silero(self):
        vad._SILERO_RUNTIME_BROKEN = False
        vad._SILERO_FALLBACK_NOTICE_EMITTED = False
        vad._silero_model = object()

        with patch("audio.vad._ensure_silero", return_value=True), patch(
            "audio.vad._read_audio_for_silero", return_value="wav_tensor"
        ) as read_audio_mock, patch(
            "audio.vad._silero_get_speech_timestamps", return_value=[{"start": 0, "end": 1}]
        ) as speech_timestamps_mock:
            is_speech = vad.is_speech("dummy.wav")

        self.assertTrue(is_speech)
        read_audio_mock.assert_called_once_with("dummy.wav", target_rate=16000)
        speech_timestamps_mock.assert_called_once_with("wav_tensor", vad._silero_model, sampling_rate=16000)

    def test_runtime_failure_switches_to_energy_fallback(self):
        vad._SILERO_RUNTIME_BROKEN = False
        vad._SILERO_FALLBACK_NOTICE_EMITTED = False

        with patch("audio.vad._ensure_silero", return_value=True), patch(
            "audio.vad._read_audio_for_silero", side_effect=RuntimeError("torchcodec mismatch")
        ), patch("audio.vad._energy_fallback_is_speech", return_value=False), patch("audio.vad.logger.info") as info_mock:
            is_speech = vad.is_speech("dummy.wav")

        self.assertFalse(is_speech)
        self.assertTrue(vad._SILERO_RUNTIME_BROKEN)
        self.assertTrue(vad._SILERO_FALLBACK_NOTICE_EMITTED)
        self.assertEqual(info_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
