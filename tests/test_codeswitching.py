"""Mixed Arabic/English codeswitching tests — 20 cases.

Tests normalize_codeswitched() entity extraction and parse_command() routing
on mixed-language utterances spanning common Egyptian Arabic verbs + Latin app
names, filenames, and system targets.
"""

from __future__ import annotations

import pytest

from nlp.codeswitching import convert_arabic_numerals, normalize_codeswitched
from core.command_parser import parse_command


def _cs(text: str) -> dict:
    return normalize_codeswitched(text)


def _p(text: str):
    r = parse_command(text)
    assert r is not None, f"parse_command({text!r}) returned None"
    return r


# ---------------------------------------------------------------------------
# Group 1 — Entity extraction from normalize_codeswitched (10 tests)
# ---------------------------------------------------------------------------

class TestNormalizeCodeswitched:

    def test_arabic_verb_latin_app(self):
        r = _cs("افتح Chrome")
        assert r["language"] == "mixed"
        assert r["verb_intent"] == "open"
        assert r["entity_text"].lower() == "chrome"

    def test_close_verb_latin_app(self):
        r = _cs("سكر Notepad")
        assert r["verb_intent"] == "close"
        assert r["entity_text"].lower() == "notepad"

    def test_english_verb_arabic_target(self):
        r = _cs("close كروم")
        assert r["language"] == "mixed"
        assert r["verb_intent"] == "close"

    def test_launch_spotify_latin(self):
        r = _cs("شغل Spotify")
        assert r["verb_intent"] == "open"
        assert r["entity_text"].lower() == "spotify"

    def test_open_vlc_mixed(self):
        r = _cs("شغل VLC")
        assert r["verb_intent"] == "open"
        assert r["entity_text"].lower() == "vlc"

    def test_multiword_latin_entity(self):
        r = _cs("دور على machine learning report")
        assert r["verb_intent"] == "search"
        assert r["entity_text"].lower() == "machine learning report"

    def test_search_python_tutorial(self):
        r = _cs("دور على Python tutorial")
        assert r["verb_intent"] == "search"
        assert "python" in r["entity_text"].lower()

    def test_increase_volume_arabic(self):
        r = _cs("ارفع الصوت")
        assert r["verb_intent"] == "increase"
        assert "volume" in r["entity_text"].lower() or r["entity_text"] in {"الصوت", "volume"}

    def test_volume_with_level_number(self):
        r = _cs("زود الصوت ٣٠")
        assert r["verb_intent"] == "increase"
        assert 30 in (r.get("numbers") or [])

    def test_decrease_volume_with_level(self):
        r = _cs("وطي الصوت ٢٠")
        assert r["verb_intent"] == "decrease"
        assert 20 in (r.get("numbers") or [])


# ---------------------------------------------------------------------------
# Group 2 — parse_command routing for mixed utterances (10 tests)
# ---------------------------------------------------------------------------

class TestMixedCommandParsing:

    def test_open_chrome_arabic_verb(self):
        r = _p("افتح Chrome")
        assert r.intent == "OS_APP_OPEN"
        assert r.args["app_name"].lower() == "chrome"

    def test_close_notepad_arabic_verb(self):
        r = _p("سكر Notepad")
        assert r.intent == "OS_APP_CLOSE"
        assert r.args["app_name"].lower() == "notepad"

    def test_open_firefox_arabic_verb(self):
        r = _p("افتح Firefox")
        assert r.intent == "OS_APP_OPEN"
        assert r.args["app_name"].lower() == "firefox"

    def test_close_chrome_english_verb(self):
        r = _p("close كروم")
        assert r.intent == "OS_APP_CLOSE"
        assert r.args["app_name"] == "chrome"

    def test_search_multiword_filename(self):
        r = _p("دور على machine learning report")
        assert r.intent == "OS_FILE_SEARCH"
        assert r.args["filename"].lower() == "machine learning report"

    def test_volume_up_with_arabic_level(self):
        r = _p("ارفع الصوت ٢٠")
        assert r.intent == "OS_SYSTEM_COMMAND"
        assert r.args["action_key"] == "volume_up"
        assert r.args.get("volume_level") == 20

    def test_volume_set_arabic_indic(self):
        r = _p("اضبط الصوت ٥٠")
        assert r.intent == "OS_SYSTEM_COMMAND"
        assert r.args["action_key"] == "volume_set"
        assert r.args.get("volume_level") == 50

    def test_volume_increase_zood(self):
        r = _p("زود الصوت ٣٠")
        assert r.intent == "OS_SYSTEM_COMMAND"
        assert r.args["action_key"] == "volume_up"
        assert r.args.get("volume_level") == 30

    def test_convert_arabic_numerals_basic(self):
        assert convert_arabic_numerals("٥٠") == "50"
        assert convert_arabic_numerals("١٢٣") == "123"
        assert convert_arabic_numerals("٠") == "0"

    def test_convert_arabic_numerals_mixed(self):
        result = convert_arabic_numerals("الساعة ٣ مساءً")
        assert "3" in result
        assert "ال" in result  # Arabic text preserved
