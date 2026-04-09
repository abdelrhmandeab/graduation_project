import unittest
from unittest.mock import patch

from core.command_parser import ParsedCommand
from core.handlers import voice


class VoiceHandlerProfileTests(unittest.TestCase):
    def _snapshot(self, profile, stt_model="openai/whisper-small"):
        return {
            "profile": profile,
            "persisted_profile": profile,
            "stt_backend": "huggingface",
            "tts_backend": "huggingface",
            "voice_quality_mode": "natural",
            "stt": {"model": stt_model, "mode": "manual"},
            "tts": {"model": "facebook/mms-tts-ara"},
        }

    def test_hf_profile_alias_eg_maps_to_egyptian(self):
        self.assertEqual(voice._normalize_hf_profile_name("eg"), "egyptian")

    def test_apply_hf_egyptian_profile_updates_preferred_language(self):
        with patch("core.handlers.voice.mic_capture.set_runtime_vad_settings"), patch(
            "core.handlers.voice.vad_runtime.set_energy_fallback_threshold"
        ), patch("core.handlers.voice.stt_runtime.set_runtime_stt_backend"), patch(
            "core.handlers.voice.stt_runtime.set_runtime_hf_settings"
        ) as set_runtime_hf_settings_mock, patch(
            "core.handlers.voice.stt_runtime.set_runtime_stt_settings"
        ), patch(
            "core.handlers.voice.speech_engine.set_backend"
        ), patch(
            "core.handlers.voice.speech_engine.set_hf_runtime_settings"
        ), patch(
            "core.handlers.voice.speech_engine.set_quality_mode"
        ), patch(
            "core.handlers.voice.session_memory.set_hf_profile"
        ), patch(
            "core.handlers.voice._hf_speech_runtime_snapshot", return_value=self._snapshot("egyptian")
        ), patch(
            "core.handlers.voice.session_memory.set_preferred_language"
        ) as preferred_language_mock:
            ok, _message, _snapshot = voice._apply_hf_speech_profile("egyptian")

        self.assertTrue(ok)
        preferred_language_mock.assert_called_once_with("ar")
        set_runtime_hf_settings_mock.assert_called_once_with(
            model="openai/whisper-small",
            mode="manual",
            chunk_length_s=12.0,
            batch_size=4,
        )

    def test_apply_hf_english_profile_updates_preferred_language(self):
        with patch("core.handlers.voice.mic_capture.set_runtime_vad_settings"), patch(
            "core.handlers.voice.vad_runtime.set_energy_fallback_threshold"
        ), patch("core.handlers.voice.stt_runtime.set_runtime_stt_backend"), patch(
            "core.handlers.voice.stt_runtime.set_runtime_hf_settings"
        ) as set_runtime_hf_settings_mock, patch(
            "core.handlers.voice.stt_runtime.set_runtime_stt_settings"
        ), patch(
            "core.handlers.voice.speech_engine.set_backend"
        ), patch(
            "core.handlers.voice.speech_engine.set_hf_runtime_settings"
        ), patch(
            "core.handlers.voice.speech_engine.set_quality_mode"
        ), patch(
            "core.handlers.voice.session_memory.set_hf_profile"
        ), patch(
            "core.handlers.voice._hf_speech_runtime_snapshot", return_value=self._snapshot("english")
        ), patch(
            "core.handlers.voice.session_memory.set_preferred_language"
        ) as preferred_language_mock:
            ok, _message, _snapshot = voice._apply_hf_speech_profile("english")

        self.assertTrue(ok)
        preferred_language_mock.assert_called_once_with("en")
        set_runtime_hf_settings_mock.assert_called_once_with(
            model="openai/whisper-small",
            mode="manual",
            chunk_length_s=12.0,
            batch_size=4,
        )

    def test_handle_hf_profile_set_returns_concise_message(self):
        parsed = ParsedCommand(
            intent="VOICE_COMMAND",
            raw="hf profile egyptian",
            normalized="hf profile egyptian",
            action="hf_profile_set",
            args={"profile": "egyptian"},
        )

        with patch("core.handlers.voice.initialize_runtime_profiles"), patch(
            "core.handlers.voice._apply_hf_speech_profile", return_value=(True, "verbose", {"profile": "egyptian"})
        ), patch("core.handlers.voice.session_memory.get_preferred_language", return_value="ar"), patch(
            "core.handlers.voice.log_action"
        ):
            ok, message, _meta = voice.handle(parsed)

        self.assertTrue(ok)
        self.assertIn("HF speech profile set to egyptian.", message)
        self.assertIn("Preferred language: ar.", message)
        self.assertNotIn("hf_stt_model", message)

    def test_handle_stt_backend_set_returns_status(self):
        parsed = ParsedCommand(
            intent="VOICE_COMMAND",
            raw="stt backend huggingface",
            normalized="stt backend huggingface",
            action="stt_backend_set",
            args={"backend": "huggingface"},
        )

        with patch("core.handlers.voice.initialize_runtime_profiles"), patch(
            "core.handlers.voice.stt_runtime.set_runtime_stt_backend", return_value="huggingface"
        ), patch(
            "core.handlers.voice._format_stt_backend_status",
            return_value=("STT Backend Status\nstt_backend: huggingface", {"stt_backend": "huggingface"}),
        ), patch("core.handlers.voice.log_action"):
            ok, message, meta = voice.handle(parsed)

        self.assertTrue(ok)
        self.assertIn("Requested STT backend: huggingface", message)
        self.assertEqual(meta.get("stt_backend"), "huggingface")

    def test_handle_stt_backend_status(self):
        parsed = ParsedCommand(
            intent="VOICE_COMMAND",
            raw="stt backend status",
            normalized="stt backend status",
            action="stt_backend_status",
            args={},
        )

        with patch("core.handlers.voice.initialize_runtime_profiles"), patch(
            "core.handlers.voice._format_stt_backend_status",
            return_value=("STT Backend Status\nstt_backend: faster_whisper", {"stt_backend": "faster_whisper"}),
        ):
            ok, message, meta = voice.handle(parsed)

        self.assertTrue(ok)
        self.assertIn("STT Backend Status", message)
        self.assertEqual(meta.get("stt_backend"), "faster_whisper")


if __name__ == "__main__":
    unittest.main()
