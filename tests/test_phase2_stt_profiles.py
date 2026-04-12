import unittest
from unittest import mock

from audio import stt as stt_runtime
from core.command_parser import parse_command
import core.orchestrator as orchestrator


class Phase2SttProfileTests(unittest.TestCase):
    def test_parser_supports_arabic_egy_profile(self):
        parsed = parse_command("stt profile arabic-egy")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "stt_profile_set")
        self.assertEqual(parsed.args.get("profile"), "arabic_egy")

    def test_parser_supports_code_switched_profile(self):
        parsed = parse_command("stt profile code switched")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "stt_profile_set")
        self.assertEqual(parsed.args.get("profile"), "code_switched")

    def test_parser_supports_auto_profile(self):
        parsed = parse_command("set stt profile to auto")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "stt_profile_set")
        self.assertEqual(parsed.args.get("profile"), "auto")

    def test_parser_supports_arabic_profile_alias(self):
        parsed = parse_command("خلي الاستماع مصري")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "stt_profile_set")
        self.assertEqual(parsed.args.get("profile"), "arabic_egy")

    def test_parser_rejects_formal_arabic_cd_phrase(self):
        parsed = parse_command("اذهب الى C")
        self.assertEqual(parsed.intent, "LLM_QUERY")

    def test_parser_rejects_formal_arabic_close_tab_phrase(self):
        parsed = parse_command("اغلق التبويب")
        self.assertEqual(parsed.intent, "LLM_QUERY")

    def test_parser_supports_egyptalk_backend_keyword(self):
        parsed = parse_command("stt backend egyptalk")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "stt_backend_set")
        self.assertEqual(parsed.args.get("backend"), "egyptalk_transformers")

    def test_parser_supports_egyptalk_backend_regex_alias(self):
        parsed = parse_command("set speech backend to nemo")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "stt_backend_set")
        self.assertEqual(parsed.args.get("backend"), "nemo")

    def test_parser_supports_egyptian_quiet_profile_phrase(self):
        parsed = parse_command("خلي الاستماع هادي")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "stt_profile_set")
        self.assertEqual(parsed.args.get("profile"), "quiet")

    def test_parser_supports_egyptian_voice_quality_phrase(self):
        parsed = parse_command("ظبط جودة الصوت طبيعي")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "voice_quality_set")
        self.assertEqual(parsed.args.get("mode"), "طبيعي")

    def test_parser_supports_egyptian_list_drives_phrase(self):
        parsed = parse_command("وريني الدرايفات")
        self.assertEqual(parsed.intent, "OS_FILE_NAVIGATION")
        self.assertEqual(parsed.action, "list_drives")

    def test_parser_supports_egyptian_pwd_phrase(self):
        parsed = parse_command("احنا فين")
        self.assertEqual(parsed.intent, "OS_FILE_NAVIGATION")
        self.assertEqual(parsed.action, "pwd")

    def test_parser_supports_egyptian_set_language_phrase(self):
        parsed = parse_command("خلي اللغة انجليزي")
        self.assertEqual(parsed.intent, "MEMORY_COMMAND")
        self.assertEqual(parsed.action, "set_language")
        self.assertEqual(parsed.args.get("language"), "en")

    def test_orchestrator_prefers_arabic_hint_for_arabic_wake_phrase(self):
        with mock.patch.object(orchestrator.session_memory, "get_stt_profile", return_value="arabic_egy"):
            self.assertIsNone(orchestrator._resolve_stt_language_hint(wake_source="phrase"))
            self.assertIsNone(orchestrator._resolve_stt_language_hint(wake_source="english"))

    def test_orchestrator_auto_profile_uses_auto_detection(self):
        with mock.patch.object(orchestrator.session_memory, "get_stt_profile", return_value="auto"):
            self.assertIsNone(orchestrator._resolve_stt_language_hint(wake_source="phrase"))
            self.assertIsNone(orchestrator._resolve_stt_language_hint(wake_source="english"))

    def test_orchestrator_empty_profile_defaults_to_auto_detection(self):
        with mock.patch.object(orchestrator.session_memory, "get_stt_profile", return_value=""):
            self.assertIsNone(orchestrator._resolve_stt_language_hint(wake_source="english"))
            self.assertIsNone(orchestrator._resolve_stt_language_hint(wake_source="phrase"))

    def test_orchestrator_code_switched_profile_keeps_phrase_auto(self):
        with mock.patch.object(orchestrator.session_memory, "get_stt_profile", return_value="code_switched"):
            self.assertIsNone(orchestrator._resolve_stt_language_hint(wake_source="phrase"))

    def test_runtime_uses_engine_auto_without_language_forcing(self):
        with mock.patch.object(
            stt_runtime,
            "_engine_transcribe",
            return_value={"text": "اهلا", "language": "ar", "method": "auto"},
        ) as engine_call, mock.patch.object(
            stt_runtime,
            "_is_weak_result",
            return_value=False,
        ):
            result = stt_runtime.transcribe_backend_direct_with_meta(
                "dummy.wav",
                backend="faster_whisper",
                language_hint="ar",
            )

        self.assertEqual(result["language"], "ar")
        engine_call.assert_called_once_with("dummy.wav")


if __name__ == "__main__":
    unittest.main()
