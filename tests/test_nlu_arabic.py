"""Arabic NLU test suite — Task 4.4.

Covers:
  - 20 pure Arabic commands
  - 15 mixed Arabic/English commands
  - 10 Arabic with Arabic-Indic numerals
  - 10 edge cases (dialectal variants, STT artifacts)

Run:  pytest tests/test_nlu_arabic.py -v
Pass: all 55 tests pass; no MSA assumptions in entity extraction.
"""

from __future__ import annotations

import pytest

from core.command_parser import parse_command
from tools.calculator import quick_calc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse(text: str):
    """Return parsed command dict or None."""
    result = parse_command(text)
    if result is None:
        return None
    return {
        "intent": result.intent,
        "action": result.action,
        "args": dict(result.args or {}),
    }


def _calc(text: str):
    """Return quick_calc result string or None."""
    return quick_calc(text)


# ---------------------------------------------------------------------------
# Group 1 — Pure Arabic Commands (20 tests)
# ---------------------------------------------------------------------------

class TestPureArabic:

    def test_open_chrome_arabic(self):
        r = _parse("افتح كروم")
        assert r["intent"] == "OS_APP_OPEN"
        assert r["args"]["app_name"] == "chrome"

    def test_open_notepad_arabic(self):
        r = _parse("افتح المفكرة")
        assert r["intent"] == "OS_APP_OPEN"
        assert r["args"]["app_name"] == "notepad"

    def test_open_calculator_arabic(self):
        r = _parse("افتح الحاسبة")
        assert r["intent"] == "OS_APP_OPEN"
        assert r["args"]["app_name"] == "calculator"

    def test_open_firefox_arabic(self):
        r = _parse("افتح فايرفوكس")
        assert r["intent"] == "OS_APP_OPEN"
        assert r["args"]["app_name"] == "firefox"

    def test_open_spotify_arabic(self):
        r = _parse("شغل سبوتيفاي")
        assert r["intent"] == "OS_APP_OPEN"
        assert r["args"]["app_name"] == "spotify"

    def test_close_chrome_arabic(self):
        r = _parse("سكر كروم")
        assert r["intent"] == "OS_APP_CLOSE"
        assert r["args"]["app_name"] == "chrome"

    def test_close_notepad_arabic(self):
        r = _parse("اقفل المفكرة")
        assert r["intent"] == "OS_APP_CLOSE"
        assert r["args"]["app_name"] == "notepad"

    def test_timer_five_minutes_arabic(self):
        r = _parse("حط تايمر ٥ دقايق")
        assert r["intent"] == "OS_TIMER"
        assert r["args"]["seconds"] == 300

    def test_timer_half_hour_arabic(self):
        r = _parse("حط تايمر نص ساعة")
        assert r["intent"] == "OS_TIMER"
        assert r["args"]["seconds"] == 1800

    def test_timer_quarter_hour_arabic(self):
        r = _parse("حط تايمر ربع ساعة")
        assert r["intent"] == "OS_TIMER"
        assert r["args"]["seconds"] == 900

    def test_volume_up_arabic(self):
        r = _parse("ارفع الصوت")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["action_key"] == "volume_up"

    def test_volume_down_arabic(self):
        r = _parse("اخفض الصوت")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["action_key"] == "volume_down"

    def test_volume_set_arabic(self):
        r = _parse("اضبط الصوت ٥٠")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["action_key"] == "volume_set"
        assert r["args"]["volume_level"] == 50

    def test_brightness_set_arabic(self):
        r = _parse("اضبط السطوع ٨٠")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["action_key"] == "brightness_set"
        assert r["args"]["brightness_level"] == 80

    def test_brightness_down_arabic(self):
        r = _parse("اخفض السطوع")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["action_key"] == "brightness_down"

    def test_search_file_arabic(self):
        r = _parse("ابحث عن report")
        assert r["intent"] == "OS_FILE_SEARCH"
        assert r["args"]["filename"] == "report"

    def test_reminder_clock_arabic(self):
        r = _parse("فكرني الساعة ٣ اكلم احمد")
        assert r["intent"] == "OS_REMINDER"
        assert r["action"] == "create"
        assert "الساعة ٣" in r["args"]["time_str"]
        assert "اكلم احمد" in r["args"]["message"]

    def test_reminder_tomorrow_arabic(self):
        r = _parse("فكرني بكرة الساعة ٩ اصحى")
        assert r["intent"] == "OS_REMINDER"
        assert r["action"] == "create"
        assert "بكرة" in r["args"]["time_str"]

    def test_stop_music_arabic(self):
        r = _parse("وقف الموسيقى")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["action_key"] == "media_stop"

    def test_lock_screen_arabic(self):
        r = _parse("قفل الشاشة")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["action_key"] == "lock"


# ---------------------------------------------------------------------------
# Group 2 — Mixed Arabic/English Commands (15 tests)
# ---------------------------------------------------------------------------

class TestMixedArabicEnglish:

    def test_open_chrome_mixed(self):
        r = _parse("افتح Chrome")
        assert r["intent"] == "OS_APP_OPEN"
        assert r["args"]["app_name"] == "chrome"

    def test_close_notepad_mixed(self):
        r = _parse("سكر Notepad")
        assert r["intent"] == "OS_APP_CLOSE"
        assert r["args"]["app_name"] == "notepad"

    def test_open_spotify_mixed(self):
        r = _parse("شغل Spotify")
        assert r["intent"] == "OS_APP_OPEN"
        assert r["args"]["app_name"] == "spotify"

    def test_open_vlc_mixed(self):
        r = _parse("شغل VLC")
        assert r["intent"] == "OS_APP_OPEN"
        assert r["args"]["app_name"] == "vlc"

    def test_open_files_english_verb(self):
        r = _parse("open الملفات")
        assert r["intent"] == "OS_FILE_NAVIGATION"

    def test_search_multiword_latin_entity(self):
        r = _parse("دور على machine learning report")
        assert r["intent"] == "OS_FILE_SEARCH"
        assert r["args"]["filename"] == "machine learning report"

    def test_volume_up_with_arabic_indic_level(self):
        r = _parse("ارفع الصوت ٢٠")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["action_key"] == "volume_up"
        assert r["args"]["volume_level"] == 20

    def test_timer_arabic_indic_mixed(self):
        r = _parse("حط تايمر ٥ دقايق")
        assert r["intent"] == "OS_TIMER"
        assert r["args"]["seconds"] == 300

    def test_close_chrome_english_verb(self):
        r = _parse("close كروم")
        assert r["intent"] == "OS_APP_CLOSE"
        assert r["args"]["app_name"] == "chrome"

    def test_volume_down_with_level_mixed(self):
        r = _parse("وطي الصوت ٢٠")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["action_key"] == "volume_down"
        assert r["args"]["volume_level"] == 20

    def test_volume_increase_with_level_mixed(self):
        r = _parse("زود الصوت ٣٠")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["action_key"] == "volume_up"
        assert r["args"]["volume_level"] == 30

    def test_search_english_topic_arabic_verb(self):
        r = _parse("دور على Python tutorial")
        assert r["intent"] == "OS_FILE_SEARCH"
        assert "python tutorial" in r["args"]["filename"].lower()

    def test_reminder_arabic_time_then_message(self):
        r = _parse("ذكرني الساعة ٥ مساء اتصل بمحمد")
        assert r["intent"] == "OS_REMINDER"
        assert r["action"] == "create"
        assert "اتصل بمحمد" in r["args"]["message"]

    def test_reminder_relative_arabic(self):
        r = _parse("فكرني بعد ساعتين اكل الدواء")
        assert r["intent"] == "OS_REMINDER"
        assert r["action"] == "create"
        assert "ساعتين" in r["args"]["time_str"]

    def test_open_firefox_mixed(self):
        r = _parse("افتح Firefox")
        assert r["intent"] == "OS_APP_OPEN"
        assert r["args"]["app_name"] == "firefox"


# ---------------------------------------------------------------------------
# Group 3 — Arabic with Arabic-Indic Numerals (10 tests)
# ---------------------------------------------------------------------------

class TestArabicIndicNumerals:

    def test_timer_ten_seconds(self):
        r = _parse("حط تايمر ١٠ ثانية")
        assert r["intent"] == "OS_TIMER"
        assert r["args"]["seconds"] == 10

    def test_timer_two_hours(self):
        r = _parse("حط تايمر ٢ ساعة")
        assert r["intent"] == "OS_TIMER"
        assert r["args"]["seconds"] == 7200

    def test_timer_three_minutes_alt_verb(self):
        r = _parse("اعمل تايمر ٣ دقايق")
        assert r["intent"] == "OS_TIMER"
        assert r["args"]["seconds"] == 180

    def test_volume_up_level_30(self):
        r = _parse("زود الصوت ٣٠")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["volume_level"] == 30

    def test_volume_set_level_50(self):
        r = _parse("اضبط الصوت ٥٠")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["volume_level"] == 50

    def test_brightness_set_level_80(self):
        r = _parse("اضبط السطوع ٨٠")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["brightness_level"] == 80

    def test_calc_percent_arabic_indic(self):
        result = _calc("٢٥ في المية من ٢٠٠")
        assert result == "50"

    def test_calc_square_root_arabic_indic(self):
        result = _calc("الجذر التربيعي من ١٤٤")
        assert result == "12"

    def test_calc_add_arabic_indic(self):
        result = _calc("٥٠ زائد ٣٠")
        assert result == "80"

    def test_calc_subtract_arabic_indic(self):
        result = _calc("١٠٠ ناقص ٢٥")
        assert result == "75"


# ---------------------------------------------------------------------------
# Group 4 — Edge Cases: Dialectal Variants & STT Artifacts (10 tests)
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_open_chrome_alt_verb_shaghal(self):
        # شغّل (with shadda) — alternate "open/run" verb
        r = _parse("شغّل كروم")
        assert r["intent"] == "OS_APP_OPEN"
        assert r["args"]["app_name"] == "chrome"

    def test_close_chrome_aghlag(self):
        # اغلق — alternate close verb
        r = _parse("اغلق كروم")
        assert r["intent"] == "OS_APP_CLOSE"
        assert r["args"]["app_name"] == "chrome"

    def test_timer_half_hour_aamal_verb(self):
        # اعمل instead of حط — different set verb
        r = _parse("اعمل تايمر نص ساعة")
        assert r["intent"] == "OS_TIMER"
        assert r["args"]["seconds"] == 1800

    def test_reminder_thakerni_verb(self):
        # ذكرني — alternate reminder trigger verb (نبهني routes to OS_TIMER alarm)
        r = _parse("ذكرني الساعة ٤ اجتماع")
        assert r["intent"] == "OS_REMINDER"
        assert r["action"] == "create"
        assert "اجتماع" in r["args"]["message"]

    def test_volume_up_rafa3_verb(self):
        # رفع — past-tense form sometimes produced by STT
        r = _parse("رفع الصوت ٢٥")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["action_key"] == "volume_up"
        assert r["args"]["volume_level"] == 25

    def test_calc_percent_what_is_arabic(self):
        # "كام" opening + "في المية" percent phrase
        result = _calc("كام ١٥ في المية من ٢٣٠")
        assert result == "34.5"

    def test_calc_multiply_arabic_indic(self):
        # "في" as multiplication between digits
        result = _calc("١٢ في ١٢")
        assert result == "144"

    def test_calc_divide_arabic_indic(self):
        result = _calc("٢٥٠ قسمة ٥")
        assert result == "50"

    def test_screenshot_arabic(self):
        r = _parse("صورة الشاشة")
        assert r["intent"] == "OS_SYSTEM_COMMAND"
        assert r["args"]["action_key"] == "screenshot"

    def test_open_notepad_nout_bad_alias(self):
        # "نوت باد" — Arabic phonetic spelling of "Notepad" (STT output)
        r = _parse("افتح نوت باد")
        assert r["intent"] == "OS_APP_OPEN"
        assert r["args"]["app_name"] == "notepad"
