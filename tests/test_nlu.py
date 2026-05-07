"""English NLU entity-extraction tests — 30 cases.

Tests parse_command() for intent routing and NLU.understand() for slot
enrichment on English utterances.
"""

from __future__ import annotations

import pytest

from core.command_parser import parse_command
from nlp.nlu import NLU, APP_CONTROL, FILE_OPS, SYSTEM, TIMER, REMINDER


def _p(text: str):
    r = parse_command(text)
    assert r is not None, f"parse_command({text!r}) returned None"
    return r


def _u(text: str, intent: str, lang: str = "en"):
    nlu = NLU()
    return nlu.understand(text, language=lang, intent=intent)


# ---------------------------------------------------------------------------
# Group 1 — App open / close (8 tests)
# ---------------------------------------------------------------------------

class TestAppControl:

    def test_open_chrome(self):
        r = _p("open chrome")
        assert r.intent == "OS_APP_OPEN"
        assert r.args["app_name"] == "chrome"

    def test_open_notepad(self):
        r = _p("open notepad")
        assert r.intent == "OS_APP_OPEN"
        assert r.args["app_name"] == "notepad"

    def test_launch_spotify(self):
        r = _p("launch spotify")
        assert r.intent == "OS_APP_OPEN"
        assert r.args["app_name"] == "spotify"

    def test_close_chrome(self):
        r = _p("close chrome")
        assert r.intent == "OS_APP_CLOSE"
        assert r.args["app_name"] == "chrome"

    def test_close_notepad(self):
        r = _p("close notepad")
        assert r.intent == "OS_APP_CLOSE"
        assert r.args["app_name"] == "notepad"

    def test_nlu_open_chrome_extracts_canonical(self):
        r = _u("open chrome", "OS_APP_OPEN")
        assert r.domain == APP_CONTROL
        assert r.entities.get("app_name") is not None
        assert r.missing_slots == []

    def test_nlu_missing_app_name(self):
        r = _u("open", "OS_APP_OPEN")
        assert "app_name" in r.missing_slots

    def test_open_file_explorer(self):
        r = _p("open file explorer")
        assert r.intent == "OS_APP_OPEN"


# ---------------------------------------------------------------------------
# Group 2 — System commands (8 tests)
# ---------------------------------------------------------------------------

class TestSystemCommands:

    def test_screenshot(self):
        r = _p("take a screenshot")
        assert r.intent == "OS_SYSTEM_COMMAND"
        assert r.args["action_key"] == "screenshot"

    def test_lock(self):
        r = _p("lock the computer")
        assert r.intent == "OS_SYSTEM_COMMAND"
        assert r.args["action_key"] == "lock"

    def test_volume_set(self):
        r = _p("set volume to 70")
        assert r.intent == "OS_SYSTEM_COMMAND"
        assert r.args["action_key"] == "volume_set"
        assert r.args["volume_level"] == 70

    def test_volume_down(self):
        r = _p("decrease volume")
        assert r.intent == "OS_SYSTEM_COMMAND"
        assert r.args["action_key"] == "volume_down"

    def test_brightness_up(self):
        r = _p("increase brightness")
        assert r.intent == "OS_SYSTEM_COMMAND"
        assert r.args["action_key"] == "brightness_up"

    def test_mute(self):
        r = _p("mute")
        assert r.intent == "OS_SYSTEM_COMMAND"
        assert r.args["action_key"] == "volume_mute"

    def test_window_maximize(self):
        r = _p("maximize window")
        assert r.intent == "OS_SYSTEM_COMMAND"
        assert r.args["action_key"] == "window_maximize"

    def test_empty_recycle_bin(self):
        r = _p("empty recycle bin")
        assert r.intent == "OS_SYSTEM_COMMAND"
        assert r.args["action_key"] == "empty_recycle_bin"


# ---------------------------------------------------------------------------
# Group 3 — Timer / reminder (7 tests)
# ---------------------------------------------------------------------------

class TestTimerReminder:

    def test_timer_10_minutes(self):
        r = _p("set timer 10 minutes")
        assert r.intent == "OS_TIMER"
        assert r.args["seconds"] == 600

    def test_timer_5_minutes_alarm(self):
        r = _p("set alarm for 5 minutes")
        assert r.intent == "OS_TIMER"
        assert r.args["seconds"] == 300

    def test_timer_90_seconds(self):
        r = _p("set timer for 90 seconds")
        assert r.intent == "OS_TIMER"
        assert r.args["seconds"] == 90

    def test_reminder_with_message(self):
        r = _p("remind me at 3pm to call John")
        assert r.intent == "OS_REMINDER"
        assert r.action == "create"
        assert "call John" in r.args["message"]

    def test_nlu_timer_extracts_seconds(self):
        r = _u("set timer 10 minutes", "OS_TIMER")
        assert r.domain == TIMER
        assert r.entities.get("seconds") == 600
        assert r.missing_slots == []

    def test_nlu_timer_missing_duration(self):
        r = _u("set a timer", "OS_TIMER")
        assert "seconds" in r.missing_slots

    def test_nlu_reminder_extracts_time_and_message(self):
        r = _u("remind me at 3pm to call Ahmed", "OS_REMINDER")
        assert r.domain == REMINDER
        assert "time_str" in r.entities
        assert "message" in r.entities


# ---------------------------------------------------------------------------
# Group 4 — File search + NLU entity enrichment (7 tests)
# ---------------------------------------------------------------------------

class TestFileSearch:

    def test_search_for_presentation(self):
        r = _p("search for presentation")
        assert r.intent == "OS_FILE_SEARCH"
        assert "presentation" in r.args["filename"]

    def test_find_pdf(self):
        r = _p("find report.pdf")
        assert r.intent == "OS_FILE_SEARCH"
        assert "report" in r.args["filename"].lower()

    def test_nlu_search_strips_verb(self):
        r = _u("search for quarterly report", "OS_FILE_SEARCH")
        assert r.domain == FILE_OPS
        assert "quarterly report" in r.entities.get("filename", "")
        assert r.missing_slots == []

    def test_nlu_find_strips_find_verb(self):
        r = _u("find my presentation", "OS_FILE_SEARCH")
        assert "presentation" in r.entities.get("filename", "")

    def test_nlu_missing_filename(self):
        # Empty utterance → nothing to extract → filename slot missing
        r = _u("", "OS_FILE_SEARCH")
        assert "filename" in r.missing_slots

    def test_nlu_search_extracts_location(self):
        r = _u("ابحث عن report.pdf", "OS_FILE_SEARCH", "ar")
        assert r.entities.get("filename") == "report.pdf"

    def test_nlu_confidence_full_when_no_missing_slots(self):
        r = _u("open chrome", "OS_APP_OPEN")
        assert r.confidence == 1.0
