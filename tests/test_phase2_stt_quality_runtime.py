import os
import unittest
from unittest import mock

from audio import stt as stt_runtime
from core.config import MEMORY_FILE
from core.handlers import voice as voice_handler


class Phase2SttQualityRuntimeTests(unittest.TestCase):
    def test_session_memory_file_path_is_absolute(self):
        self.assertTrue(MEMORY_FILE)
        self.assertTrue(str(MEMORY_FILE).lower().endswith("jarvis_memory.json"))
        self.assertTrue(os.path.isabs(str(MEMORY_FILE)))

    def test_initialize_runtime_profiles_applies_auto_stt_when_not_persisted(self):
        with mock.patch.object(voice_handler.session_memory, "get_stt_profile", return_value=""), mock.patch.object(
            voice_handler.session_memory,
            "get_audio_ux_profile",
            return_value="",
        ), mock.patch.object(
            voice_handler,
            "_apply_stt_profile",
            return_value=(True, "", {}),
        ) as apply_stt, mock.patch.object(
            voice_handler,
            "_apply_audio_ux_profile",
            return_value=(True, "", {}),
        ):
            voice_handler._RUNTIME_PROFILES_INITIALIZED = False
            ok, summary = voice_handler.initialize_runtime_profiles(force=True)

        self.assertTrue(ok)
        self.assertIn("default_stt_profile:auto", summary)
        apply_stt.assert_called_once_with("auto", persist=False)

    def test_initialize_runtime_profiles_falls_back_to_auto_on_invalid_profile(self):
        with mock.patch.object(voice_handler.session_memory, "get_stt_profile", return_value="legacy_invalid"), mock.patch.object(
            voice_handler.session_memory,
            "get_audio_ux_profile",
            return_value="",
        ), mock.patch.object(
            voice_handler.session_memory,
            "set_stt_profile",
        ) as set_stt_profile, mock.patch.object(
            voice_handler,
            "_apply_stt_profile",
            side_effect=[
                (False, "Unsupported", {}),
                (True, "", {}),
            ],
        ) as apply_stt, mock.patch.object(
            voice_handler,
            "_apply_audio_ux_profile",
            return_value=(True, "", {}),
        ):
            voice_handler._RUNTIME_PROFILES_INITIALIZED = False
            ok, summary = voice_handler.initialize_runtime_profiles(force=True)

        self.assertTrue(ok)
        self.assertIn("failed_stt_restore", summary)
        self.assertIn("fallback_stt_profile:auto", summary)
        self.assertEqual(apply_stt.call_count, 2)
        apply_stt.assert_has_calls(
            [
                mock.call("legacy_invalid", persist=False),
                mock.call("auto", persist=False),
            ]
        )
        set_stt_profile.assert_called_once_with("")

    def test_transcribe_faster_whisper_with_meta_uses_single_pass_no_retry(self):
        runtime_settings = {
            "beam_size": 4,
            "vad_filter": True,
            "condition_on_previous_text": False,
            "quality_retry_threshold": 0.54,
            "quality_retry_beam_size": 6,
        }
        fake_result = ("اعوزك تقول لي اخبار الحرب", "ar", 0.42)

        with mock.patch.object(stt_runtime, "_get_whisper_model", return_value=object()), mock.patch.object(
            stt_runtime,
            "get_runtime_stt_settings",
            return_value=runtime_settings,
        ), mock.patch.object(
            stt_runtime,
            "_transcribe_once_with_meta",
            return_value=fake_result,
        ) as transcribe_once:
            text, language, confidence = stt_runtime._transcribe_faster_whisper_with_meta(
                "dummy.wav",
                language_hint=None,
            )

        self.assertEqual(text, fake_result[0])
        self.assertEqual(language, "ar")
        self.assertAlmostEqual(confidence, fake_result[2])
        self.assertEqual(transcribe_once.call_count, 1)

    def test_stt_normalizer_is_minimal(self):
        normalized = stt_runtime.normalize_arabic_post_transcript("  اخبار   التكس   في  مصر  ")
        self.assertEqual(normalized, "اخبار التكس في مصر")

    def test_transcribe_backend_direct_uses_egyptalk_when_requested(self):
        with mock.patch.object(
            stt_runtime,
            "_transcribe_egyptalk_transformers_with_meta",
            return_value=("الجو عامل ايه", "ar", 0.85),
        ) as egyptalk_call, mock.patch.object(
            stt_runtime,
            "_transcribe_faster_whisper_with_meta",
        ) as faster_call:
            result = stt_runtime.transcribe_backend_direct_with_meta(
                "dummy.wav",
                backend="egyptalk",
                language_hint="ar",
            )

        self.assertEqual(result["backend"], "egyptalk_transformers")
        self.assertEqual(result["language"], "ar")
        self.assertEqual(egyptalk_call.call_count, 1)
        self.assertEqual(faster_call.call_count, 0)

    def test_transcribe_backend_direct_falls_back_to_faster_on_backend_failure(self):
        with mock.patch.object(
            stt_runtime,
            "_transcribe_egyptalk_transformers_with_meta",
            side_effect=RuntimeError("backend unavailable"),
        ) as egyptalk_call, mock.patch.object(
            stt_runtime,
            "_transcribe_faster_whisper_with_meta",
            return_value=("weather in cairo", "en", 0.73),
        ) as faster_call:
            result = stt_runtime.transcribe_backend_direct_with_meta(
                "dummy.wav",
                backend="egyptalk_transformers",
                language_hint=None,
            )

        self.assertEqual(result["backend"], "faster_whisper")
        self.assertEqual(result["language"], "en")
        self.assertEqual(egyptalk_call.call_count, 1)
        self.assertEqual(faster_call.call_count, 1)

    def test_dual_fallback_gate_triggers_for_weak_arabic_text(self):
        should_fallback = stt_runtime._should_try_egyptalk_fallback(
            "ar",
            0.0,
            text="؟؟",
            language_hint="ar",
        )

        self.assertTrue(should_fallback)

    def test_dual_fallback_gate_skips_for_strong_english_text(self):
        should_fallback = stt_runtime._should_try_egyptalk_fallback(
            "en",
            0.0,
            text="open chrome please",
            language_hint=None,
        )

        self.assertFalse(should_fallback)

    def test_dual_fallback_gate_triggers_for_empty_text(self):
        should_fallback = stt_runtime._should_try_egyptalk_fallback(
            "unknown",
            0.0,
            text="",
            language_hint=None,
        )

        self.assertTrue(should_fallback)

    def test_voice_handler_normalizes_arabic_fast_profile_alias(self):
        self.assertEqual(
            voice_handler._normalize_profile_name("arabic_fast"),
            "arabic_egy",
        )

    def test_voice_handler_normalizes_egyptalk_backend_alias(self):
        self.assertEqual(
            voice_handler._normalize_stt_backend_name("nemo"),
            "egyptalk_transformers",
        )


if __name__ == "__main__":
    unittest.main()
