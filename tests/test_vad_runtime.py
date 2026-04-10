import tempfile
import unittest
import wave
import os

import numpy as np

from audio import vad


class VadRuntimeTests(unittest.TestCase):
    def _write_wave(self, samples):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            path = handle.name
        int16_samples = np.asarray(samples, dtype=np.int16)
        with wave.open(path, "wb") as wav_handle:
            wav_handle.setnchannels(1)
            wav_handle.setsampwidth(2)
            wav_handle.setframerate(16000)
            wav_handle.writeframes(int16_samples.tobytes())
        return path

    def test_threshold_setter_and_getter(self):
        original = vad.get_energy_fallback_threshold()
        try:
            updated = vad.set_energy_fallback_threshold(0.02)
            self.assertAlmostEqual(updated, 0.02, places=6)
            self.assertAlmostEqual(vad.get_energy_fallback_threshold(), 0.02, places=6)
        finally:
            vad.set_energy_fallback_threshold(original)

    def test_is_speech_detects_non_silent_audio(self):
        original = vad.get_energy_fallback_threshold()
        path = ""
        try:
            vad.set_energy_fallback_threshold(0.05)
            # Constant medium amplitude should cross the threshold.
            path = self._write_wave(np.full(16000, 12000, dtype=np.int16))
            self.assertTrue(vad.is_speech(path))
        finally:
            vad.set_energy_fallback_threshold(original)
            if path and os.path.exists(path):
                os.remove(path)

    def test_is_speech_rejects_silence(self):
        original = vad.get_energy_fallback_threshold()
        path = ""
        try:
            vad.set_energy_fallback_threshold(0.01)
            path = self._write_wave(np.zeros(16000, dtype=np.int16))
            self.assertFalse(vad.is_speech(path))
        finally:
            vad.set_energy_fallback_threshold(original)
            if path and os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
