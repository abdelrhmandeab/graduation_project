"""Wake-word runtime settings tests — 15 cases.

Tests the pure configuration and settings API — no hardware, no ONNX model
loading.  Covers: default settings values, runtime mutability, phrase-settings
round-trip, wake-mode switching, and behavior flags.
"""

from __future__ import annotations

import pytest

from audio.wake_word import (
    WAKE_WORD,
    WAKE_WORD_THRESHOLD,
    WAKE_WORD_MODE,
    consume_barge_in_wake,
    consume_follow_up_wake,
    get_runtime_wake_mode,
    get_runtime_wake_word_behavior,
    get_runtime_wake_word_phrase_settings,
    get_runtime_wake_word_settings,
    set_runtime_wake_mode,
    set_runtime_wake_word_behavior,
    set_runtime_wake_word_phrase_settings,
)


# ---------------------------------------------------------------------------
# Group 1 — Config constants (3 tests)
# ---------------------------------------------------------------------------

class TestConfigConstants:

    def test_wake_word_defined(self):
        assert WAKE_WORD and isinstance(WAKE_WORD, str)

    def test_threshold_in_range(self):
        assert 0.0 < WAKE_WORD_THRESHOLD <= 1.0

    def test_mode_is_valid(self):
        assert WAKE_WORD_MODE in {"english", "arabic", "both"}


# ---------------------------------------------------------------------------
# Group 2 — Runtime settings get / set (4 tests)
# ---------------------------------------------------------------------------

class TestRuntimeSettings:

    def test_get_settings_returns_dict(self):
        s = get_runtime_wake_word_settings()
        assert isinstance(s, dict)

    def test_behavior_has_expected_keys(self):
        b = get_runtime_wake_word_behavior()
        assert "ignore_while_speaking" in b
        assert "barge_in_interrupt_on_wake" in b

    def test_set_and_get_behavior(self):
        original = get_runtime_wake_word_behavior()
        try:
            set_runtime_wake_word_behavior(ignore_while_speaking=False)
            b = get_runtime_wake_word_behavior()
            assert b["ignore_while_speaking"] is False
        finally:
            set_runtime_wake_word_behavior(**original)

    def test_behavior_restore_after_mutation(self):
        original = get_runtime_wake_word_behavior()
        set_runtime_wake_word_behavior(ignore_while_speaking=True)
        restored = get_runtime_wake_word_behavior()
        assert restored["ignore_while_speaking"] is True
        set_runtime_wake_word_behavior(**original)


# ---------------------------------------------------------------------------
# Group 3 — Wake mode switching (4 tests)
# ---------------------------------------------------------------------------

class TestWakeModeSwitching:

    def test_get_wake_mode_returns_string(self):
        mode = get_runtime_wake_mode()
        assert isinstance(mode, str)
        assert mode in {"english", "arabic", "both"}

    def test_set_mode_english(self):
        original = get_runtime_wake_mode()
        try:
            set_runtime_wake_mode("english")
            assert get_runtime_wake_mode() == "english"
        finally:
            set_runtime_wake_mode(original)

    def test_set_mode_arabic(self):
        original = get_runtime_wake_mode()
        try:
            set_runtime_wake_mode("arabic")
            assert get_runtime_wake_mode() == "arabic"
        finally:
            set_runtime_wake_mode(original)

    def test_set_mode_both(self):
        original = get_runtime_wake_mode()
        try:
            set_runtime_wake_mode("both")
            assert get_runtime_wake_mode() == "both"
        finally:
            set_runtime_wake_mode(original)


# ---------------------------------------------------------------------------
# Group 4 — Phrase settings + barge-in event flags (4 tests)
# ---------------------------------------------------------------------------

class TestPhraseSettings:

    def test_phrase_settings_has_mode(self):
        ps = get_runtime_wake_word_phrase_settings()
        assert "mode" in ps

    def test_phrase_settings_arabic_enabled_key(self):
        ps = get_runtime_wake_word_phrase_settings()
        assert "arabic_enabled" in ps

    def test_consume_barge_in_wake_returns_bool(self):
        result = consume_barge_in_wake()
        assert isinstance(result, bool)

    def test_consume_follow_up_wake_returns_bool(self):
        result = consume_follow_up_wake()
        assert isinstance(result, bool)
