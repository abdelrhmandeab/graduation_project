import unittest

from core.command_parser import parse_command


class ParserHeuristicsTests(unittest.TestCase):
    def test_schedule_english_enqueue(self):
        parsed = parse_command("in 5 minutes mute volume")
        self.assertEqual(parsed.intent, "JOB_QUEUE_COMMAND")
        self.assertEqual(parsed.action, "enqueue")
        self.assertEqual(parsed.args.get("delay_seconds"), 300)
        self.assertEqual(parsed.args.get("command_text"), "mute volume")

    def test_schedule_arabic_enqueue(self):
        parsed = parse_command("بعد 10 دقائق خذ صورة للشاشة")
        self.assertEqual(parsed.intent, "JOB_QUEUE_COMMAND")
        self.assertEqual(parsed.action, "enqueue")
        self.assertEqual(parsed.args.get("delay_seconds"), 600)
        self.assertEqual(parsed.args.get("command_text"), "خذ صورة للشاشة")

    def test_browser_open_url_normalized(self):
        parsed = parse_command("open github.com")
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "browser_open_url")
        self.assertEqual(parsed.args.get("url"), "https://github.com")

    def test_browser_back_not_misread_as_media_seek(self):
        parsed = parse_command("go back in the browser")
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "browser_back")

    def test_browser_search_about_with_filler_prefix(self):
        parsed = parse_command("No, I just want you to search about the war between Iran and the United States.")
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "browser_search_web")
        self.assertIn("war between iran and the united states", parsed.args.get("search_query", "").lower())

    def test_browser_search_arabic_with_web_keyword(self):
        parsed = parse_command("ابحث في الويب عن أخبار الاقتصاد")
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "browser_search_web")
        self.assertEqual(parsed.args.get("search_query"), "أخبار الاقتصاد")

    def test_browser_search_english_with_trailing_online(self):
        parsed = parse_command("search for news about iran online")
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "browser_search_web")
        self.assertEqual(parsed.args.get("search_query"), "news about iran")

    def test_browser_search_arabic_with_trailing_online_word(self):
        parsed = parse_command("ابحث عن الحرب بين ايران وامريكا اونلاين")
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "browser_search_web")
        self.assertEqual(parsed.args.get("search_query"), "الحرب بين ايران وامريكا")

    def test_file_search_preserved_for_explicit_file_phrase(self):
        parsed = parse_command("search file report.docx in desktop")
        self.assertEqual(parsed.intent, "OS_FILE_SEARCH")
        self.assertEqual(parsed.args.get("filename"), "report.docx")

    def test_window_arabic_maximize(self):
        parsed = parse_command("كبر النافذة")
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "window_maximize")

    def test_file_operation_move(self):
        parsed = parse_command("move report.docx to documents")
        self.assertEqual(parsed.intent, "OS_FILE_NAVIGATION")
        self.assertEqual(parsed.action, "move_item")
        self.assertEqual(parsed.args.get("source"), "report.docx")

    def test_wake_mode_command(self):
        parsed = parse_command("wake mode arabic")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "wake_mode_set")
        self.assertEqual(parsed.args.get("mode"), "arabic")

    def test_wake_trigger_add_command(self):
        parsed = parse_command("wake triggers add ya jarvis")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "wake_triggers_add")
        self.assertEqual(parsed.args.get("trigger"), "ya jarvis")

    def test_wake_trigger_remove_command(self):
        parsed = parse_command("wake triggers remove ya jarvis")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "wake_triggers_remove")
        self.assertEqual(parsed.args.get("trigger"), "ya jarvis")

    def test_wake_status_keyword(self):
        parsed = parse_command("wake triggers list")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "wake_status")

    def test_stt_backend_keyword_faster_whisper(self):
        parsed = parse_command("stt backend faster whisper")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "stt_backend_set")
        self.assertEqual(parsed.args.get("backend"), "faster_whisper")

    def test_stt_backend_status_keyword(self):
        parsed = parse_command("stt backend status")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "stt_backend_status")

    def test_latency_mode_fast_keyword(self):
        parsed = parse_command("latency mode fast")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "audio_ux_profile_set")
        self.assertEqual(parsed.args.get("profile"), "responsive")

    def test_latency_mode_regex_balanced(self):
        parsed = parse_command("set speed mode to normal")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "audio_ux_profile_set")
        self.assertEqual(parsed.args.get("profile"), "normal")

    def test_latency_status_keyword(self):
        parsed = parse_command("pipeline latency status")
        self.assertEqual(parsed.intent, "VOICE_COMMAND")
        self.assertEqual(parsed.action, "latency_status")

    def test_language_arabic_keyword_command(self):
        parsed = parse_command("language arabic")
        self.assertEqual(parsed.intent, "MEMORY_COMMAND")
        self.assertEqual(parsed.action, "set_language")
        self.assertEqual(parsed.args.get("language"), "ar")

    def test_language_regex_command(self):
        parsed = parse_command("set language to en")
        self.assertEqual(parsed.intent, "MEMORY_COMMAND")
        self.assertEqual(parsed.action, "set_language")
        self.assertEqual(parsed.args.get("language"), "en")

    def test_language_change_into_phrase(self):
        parsed = parse_command("change the language into arabic")
        self.assertEqual(parsed.intent, "MEMORY_COMMAND")
        self.assertEqual(parsed.action, "set_language")
        self.assertEqual(parsed.args.get("language"), "ar")

    def test_language_switch_phrase_with_trailing_punctuation(self):
        parsed = parse_command("Switch language to Arabic.")
        self.assertEqual(parsed.intent, "MEMORY_COMMAND")
        self.assertEqual(parsed.action, "set_language")
        self.assertEqual(parsed.args.get("language"), "ar")

    def test_wake_benchmark_command(self):
        parsed = parse_command("benchmark wake")
        self.assertEqual(parsed.intent, "BENCHMARK_COMMAND")
        self.assertEqual(parsed.action, "wake_reliability")

    def test_stt_benchmark_command(self):
        parsed = parse_command("benchmark stt")
        self.assertEqual(parsed.intent, "BENCHMARK_COMMAND")
        self.assertEqual(parsed.action, "stt_reliability")

    def test_tts_benchmark_command(self):
        parsed = parse_command("benchmark tts")
        self.assertEqual(parsed.intent, "BENCHMARK_COMMAND")
        self.assertEqual(parsed.action, "tts_quality")

    def test_notifications_off_command(self):
        parsed = parse_command("disable notifications")
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "notifications_off")

    def test_natural_arabic_firefox_open_phrase(self):
        parsed = parse_command("أريد أن أفتح فايرفوكس")
        self.assertEqual(parsed.intent, "OS_APP_OPEN")
        self.assertEqual(parsed.args.get("app_name"), "firefox")

    def test_dnd_off_maps_to_notifications_on(self):
        parsed = parse_command("turn off do not disturb")
        self.assertEqual(parsed.intent, "OS_SYSTEM_COMMAND")
        self.assertEqual(parsed.args.get("action_key"), "notifications_on")

    def test_informational_arabic_phrase_stays_llm_query(self):
        parsed = parse_command("أريدك أن تخبرني عن الوصول بين إيران وأمريكا")
        self.assertEqual(parsed.intent, "LLM_QUERY")


if __name__ == "__main__":
    unittest.main()
