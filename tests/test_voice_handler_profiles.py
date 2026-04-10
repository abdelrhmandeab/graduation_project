import unittest
from unittest.mock import patch

from core.command_parser import ParsedCommand
from core.handlers import voice


class VoiceHandlerProfileTests(unittest.TestCase):
    def test_handle_stt_backend_set_returns_status(self):
        parsed = ParsedCommand(
            intent="VOICE_COMMAND",
            raw="stt backend whisper",
            normalized="stt backend whisper",
            action="stt_backend_set",
            args={"backend": "whisper"},
        )

        with patch("core.handlers.voice.initialize_runtime_profiles"), patch(
            "core.handlers.voice.stt_runtime.set_runtime_stt_backend", return_value="faster_whisper"
        ), patch(
            "core.handlers.voice._format_stt_backend_status",
            return_value=("STT Backend Status\nstt_backend: faster_whisper", {"stt_backend": "faster_whisper"}),
        ), patch("core.handlers.voice.log_action"):
            ok, message, meta = voice.handle(parsed)

        self.assertTrue(ok)
        self.assertIn("Requested STT backend: faster_whisper", message)
        self.assertEqual(meta.get("stt_backend"), "faster_whisper")

    def test_handle_stt_backend_set_rejects_unsupported_backend(self):
        parsed = ParsedCommand(
            intent="VOICE_COMMAND",
            raw="stt backend legacy",
            normalized="stt backend legacy",
            action="stt_backend_set",
            args={"backend": "legacy"},
        )

        with patch("core.handlers.voice.initialize_runtime_profiles"):
            ok, message, _meta = voice.handle(parsed)

        self.assertFalse(ok)
        self.assertIn("Unsupported STT backend", message)

    def test_handle_voice_quality_set_natural(self):
        parsed = ParsedCommand(
            intent="VOICE_COMMAND",
            raw="voice quality natural",
            normalized="voice quality natural",
            action="voice_quality_set",
            args={"mode": "natural"},
        )

        with patch("core.handlers.voice.initialize_runtime_profiles"), patch(
            "core.handlers.voice.speech_engine.set_quality_mode", return_value="natural"
        ), patch("core.handlers.voice.log_action"):
            ok, message, meta = voice.handle(parsed)

        self.assertTrue(ok)
        self.assertIn("Voice quality mode: natural", message)
        self.assertEqual(meta.get("voice_quality_mode"), "natural")

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
