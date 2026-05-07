"""ResponseShaper bilingual template tests — 30 cases.

Tests shape() for all major intent/action combinations in both English and
Egyptian Arabic, and get_dialogue() for all slot-fill / error dialogue keys.

All Arabic text is Egyptian colloquial (عامية مصرية), not MSA.
"""

from __future__ import annotations

import pytest

from core.response_shaper import ResponseShaper, DIALOGUE_TEMPLATES


@pytest.fixture
def rs():
    return ResponseShaper()


# ---------------------------------------------------------------------------
# Group 1 — App open / close templates (6 tests)
# ---------------------------------------------------------------------------

class TestAppTemplates:

    def test_open_en(self, rs):
        r = rs.shape("OS_APP_OPEN", "open", {"app_name": "chrome"}, language="en")
        assert "chrome" in r.lower()
        assert r  # non-empty

    def test_open_ar(self, rs):
        r = rs.shape("OS_APP_OPEN", "open", {"app_name": "chrome"}, language="ar")
        assert r  # non-empty, Arabic response

    def test_close_en(self, rs):
        r = rs.shape("OS_APP_CLOSE", "close", {"app_name": "notepad"}, language="en")
        assert "notepad" in r.lower()

    def test_close_ar(self, rs):
        r = rs.shape("OS_APP_CLOSE", "close", {"app_name": "notepad"}, language="ar")
        assert r

    def test_not_found_en(self, rs):
        r = rs.shape("OS_APP_OPEN", "not_found", {"app_name": "xyz123"}, language="en")
        assert r  # has a fallback message

    def test_should_use_template_true(self, rs):
        assert rs.should_use_template("OS_APP_OPEN", "open") is True

    def test_should_use_template_false_for_llm(self, rs):
        assert rs.should_use_template("LLM_QUERY", "") is False


# ---------------------------------------------------------------------------
# Group 2 — System command templates (8 tests)
# ---------------------------------------------------------------------------

class TestSystemCommandTemplates:

    def test_lock_en(self, rs):
        r = rs.shape("OS_SYSTEM_COMMAND", "lock", {"action_key": "lock"}, language="en")
        assert r
        assert "lock" in r.lower() or "computer" in r.lower()

    def test_lock_ar(self, rs):
        r = rs.shape("OS_SYSTEM_COMMAND", "lock", {"action_key": "lock"}, language="ar")
        assert r

    def test_volume_set_with_level_en(self, rs):
        r = rs.shape("OS_SYSTEM_COMMAND", "volume_set",
                     {"action_key": "volume_set", "volume_level": 50}, language="en")
        assert "50" in r

    def test_volume_set_with_level_ar(self, rs):
        r = rs.shape("OS_SYSTEM_COMMAND", "volume_set",
                     {"action_key": "volume_set", "volume_level": 50}, language="ar")
        assert "50" in r

    def test_volume_up_with_level_en(self, rs):
        r = rs.shape("OS_SYSTEM_COMMAND", "volume_up",
                     {"action_key": "volume_up", "volume_level": 20}, language="en")
        assert "20" in r

    def test_screenshot_en(self, rs):
        r = rs.shape("OS_SYSTEM_COMMAND", "screenshot",
                     {"action_key": "screenshot"}, language="en")
        assert r

    def test_screenshot_ar(self, rs):
        r = rs.shape("OS_SYSTEM_COMMAND", "screenshot",
                     {"action_key": "screenshot"}, language="ar")
        assert r

    def test_brightness_set_en(self, rs):
        r = rs.shape("OS_SYSTEM_COMMAND", "brightness_set",
                     {"action_key": "brightness_set", "brightness_level": 80}, language="en")
        assert "80" in r


# ---------------------------------------------------------------------------
# Group 3 — Timer templates (4 tests)
# ---------------------------------------------------------------------------

class TestTimerTemplates:

    def test_timer_set_en(self, rs):
        r = rs.shape("OS_TIMER", "set", {"seconds": 300, "label": "Timer"}, language="en")
        assert r
        assert "5" in r or "min" in r.lower() or "timer" in r.lower()

    def test_timer_set_ar(self, rs):
        r = rs.shape("OS_TIMER", "set", {"seconds": 300, "label": "Timer"}, language="ar")
        assert r

    def test_timer_done_with_label_en(self, rs):
        r = rs.shape("OS_TIMER", "done", {"label": "Pasta timer"}, language="en")
        assert "Pasta timer" in r

    def test_timer_done_without_label_en(self, rs):
        r = rs.shape("OS_TIMER", "done", {}, language="en")
        assert r  # should not crash with missing optional label


# ---------------------------------------------------------------------------
# Group 4 — Reminder + file search templates (4 tests)
# ---------------------------------------------------------------------------

class TestReminderFileTemplates:

    def test_reminder_create_en(self, rs):
        r = rs.shape("OS_REMINDER", "create",
                     {"time_str": "3pm", "message": "call Ahmed"}, language="en")
        assert r
        assert "3pm" in r

    def test_reminder_create_ar(self, rs):
        r = rs.shape("OS_REMINDER", "create",
                     {"time_str": "3pm", "message": "call Ahmed"}, language="ar")
        assert r

    def test_file_not_found_en(self, rs):
        r = rs.shape("OS_FILE_SEARCH", "not_found", {"filename": "missing.docx"}, language="en")
        assert r

    def test_file_not_found_ar(self, rs):
        r = rs.shape("OS_FILE_SEARCH", "not_found", {"filename": "missing.docx"}, language="ar")
        assert r


# ---------------------------------------------------------------------------
# Group 5 — Dialogue template coverage (14 tests: all 14 keys × both languages)
# ---------------------------------------------------------------------------

class TestDialogueTemplates:

    def test_all_keys_have_en_value(self, rs):
        for key in DIALOGUE_TEMPLATES:
            val = rs.get_dialogue(key, "en")
            assert val, f"dialogue key {key!r} returned empty English string"

    def test_all_keys_have_ar_value(self, rs):
        for key in DIALOGUE_TEMPLATES:
            val = rs.get_dialogue(key, "ar")
            assert val, f"dialogue key {key!r} returned empty Arabic string"

    def test_didnt_catch_en(self, rs):
        r = rs.get_dialogue("didnt_catch", "en")
        assert "catch" in r.lower() or "sorry" in r.lower()

    def test_didnt_catch_ar_is_egyptian(self, rs):
        r = rs.get_dialogue("didnt_catch", "ar")
        # Egyptian colloquial: مسمعتش (not فصحى: لم أسمع)
        assert "مسمع" in r or "معلش" in r

    def test_generic_error_en(self, rs):
        r = rs.get_dialogue("generic_error", "en")
        assert "wrong" in r.lower() or "error" in r.lower() or "went" in r.lower()

    def test_generic_error_ar_is_egyptian(self, rs):
        r = rs.get_dialogue("generic_error", "ar")
        assert "مشكلة" in r or "حصل" in r

    def test_slot_missing_app_en(self, rs):
        r = rs.get_dialogue("slot_missing_app", "en")
        assert "app" in r.lower() or "which" in r.lower()

    def test_slot_missing_time_en(self, rs):
        r = rs.get_dialogue("slot_missing_time", "en")
        assert r

    def test_slot_missing_dur_ar(self, rs):
        r = rs.get_dialogue("slot_missing_dur", "ar")
        assert r

    def test_slot_missing_file_en(self, rs):
        r = rs.get_dialogue("slot_missing_file", "en")
        assert "file" in r.lower() or "which" in r.lower()

    def test_no_internet_en(self, rs):
        r = rs.get_dialogue("no_internet", "en")
        assert "internet" in r.lower() or "reach" in r.lower()

    def test_not_supported_ar(self, rs):
        r = rs.get_dialogue("not_supported", "ar")
        assert r

    def test_timeout_en(self, rs):
        r = rs.get_dialogue("timeout", "en")
        assert "long" in r.lower() or "took" in r.lower()

    def test_anything_else_ar(self, rs):
        r = rs.get_dialogue("anything_else", "ar")
        assert r
