"""Streaming STT pure-function tests — 15 cases.

Tests helper functions that do not require a microphone or Whisper model:
chunk-level RMS computation, seconds-to-chunks conversion, silence-threshold
resolution, adaptive silence scaling, and Arabic post-transcription
normalization.
"""

from __future__ import annotations

import numpy as np
import pytest

from audio.streaming_stt import (
    _adaptive_silence_seconds,
    _chunk_rms,
    _resolve_silence_seconds,
    _seconds_to_chunks,
    normalize_arabic_post_transcript,
    AUDIO_CHUNK_SIZE,
    SAMPLE_RATE,
)


# ---------------------------------------------------------------------------
# Group 1 — _seconds_to_chunks (3 tests)
# ---------------------------------------------------------------------------

class TestSecondsToChunks:

    def test_one_second(self):
        chunks = _seconds_to_chunks(1.0)
        expected = max(1, int(np.ceil(SAMPLE_RATE / AUDIO_CHUNK_SIZE)))
        assert chunks == expected

    def test_zero_seconds_returns_at_least_one(self):
        assert _seconds_to_chunks(0.0) >= 1

    def test_half_second(self):
        assert _seconds_to_chunks(0.5) >= 1


# ---------------------------------------------------------------------------
# Group 2 — _chunk_rms (3 tests)
# ---------------------------------------------------------------------------

class TestChunkRms:

    def test_silence_rms_is_zero(self):
        silence = np.zeros(512, dtype=np.int16)
        assert _chunk_rms(silence) == 0.0

    def test_constant_signal_rms(self):
        # All samples = 16384 → float32 = 0.5 → RMS = 0.5
        chunk = np.full(512, 16384, dtype=np.int16)
        assert _chunk_rms(chunk) == pytest.approx(0.5, rel=1e-4)

    def test_max_amplitude_rms(self):
        chunk = np.full(512, 32767, dtype=np.int16)
        assert _chunk_rms(chunk) == pytest.approx(32767 / 32768, rel=1e-3)


# ---------------------------------------------------------------------------
# Group 3 — _resolve_silence_seconds (3 tests)
# ---------------------------------------------------------------------------

class TestResolveSilenceSeconds:

    def test_command_mode(self):
        s = _resolve_silence_seconds("command")
        assert 0.1 < s < 3.0

    def test_chat_mode_longer_than_command(self):
        command_s = _resolve_silence_seconds("command")
        chat_s = _resolve_silence_seconds("chat")
        assert chat_s > command_s

    def test_explicit_override(self):
        s = _resolve_silence_seconds("command", explicit_silence_seconds=0.8)
        assert s == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Group 4 — _adaptive_silence_seconds (2 tests)
# ---------------------------------------------------------------------------

class TestAdaptiveSilence:

    def test_adapts_between_base_and_max(self):
        base, max_s = 0.5, 1.5
        result = _adaptive_silence_seconds(base, speech_seconds=1.5, max_seconds=max_s)
        assert base <= result <= max_s

    def test_short_speech_stays_near_base(self):
        result = _adaptive_silence_seconds(0.5, speech_seconds=0.1, max_seconds=1.5)
        assert result < 0.6  # barely moved from base


# ---------------------------------------------------------------------------
# Group 5 — normalize_arabic_post_transcript (4 tests)
# ---------------------------------------------------------------------------

class TestNormalizeArabicPostTranscript:

    def test_strips_tashkeel(self):
        result = normalize_arabic_post_transcript("جَارَفِيسَ")
        assert result == "جارفيس"

    def test_normalizes_hamza_above(self):
        result = normalize_arabic_post_transcript("أفتح")
        assert result == "افتح"

    def test_normalizes_hamza_below(self):
        result = normalize_arabic_post_transcript("إفتح")
        assert result == "افتح"

    def test_normalizes_alef_madda(self):
        result = normalize_arabic_post_transcript("آخر")
        assert result == "اخر"
