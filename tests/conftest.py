"""Shared fixtures for the Jarvis test suite.

Run all tests:      pytest tests/ -v --tb=short
Arabic only:        pytest tests/test_nlu_arabic.py tests/test_codeswitching.py -v
Latency only:       pytest tests/test_latency.py -v
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_audio_chunk():
    """512-sample silent int16 audio chunk."""
    return np.zeros(512, dtype=np.int16)


@pytest.fixture
def speech_audio_chunk():
    """512-sample int16 chunk with speech-like amplitude (RMS ≈ 0.24)."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(512) * 8000).astype(np.int16)


@pytest.fixture
def one_second_silence():
    """One second of 16 kHz silent audio."""
    return np.zeros(16000, dtype=np.int16)


@pytest.fixture
def one_second_noise():
    """One second of 16 kHz pink-noise-like audio (RMS ≈ 0.06)."""
    rng = np.random.default_rng(7)
    return (rng.standard_normal(16000) * 2000).astype(np.int16)


# ---------------------------------------------------------------------------
# NLU / parser fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_nlu():
    """NLU instance (no model load — pure regex)."""
    from nlp.nlu import NLU
    return NLU()


@pytest.fixture
def parsed():
    """Return a callable that runs parse_command and asserts non-None."""
    from core.command_parser import parse_command

    def _parse(text: str):
        result = parse_command(text)
        assert result is not None, f"parse_command({text!r}) returned None"
        return result

    return _parse


# ---------------------------------------------------------------------------
# Dialogue / response fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_dm():
    """Fresh DialogueManager (not the module-level singleton)."""
    from core.dialogue_manager import DialogueManager
    return DialogueManager()


@pytest.fixture
def shaper():
    """ResponseShaper instance."""
    from core.response_shaper import ResponseShaper
    return ResponseShaper()


# ---------------------------------------------------------------------------
# Metrics fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def fresh_tracker():
    """Fresh LatencyTracker with no recorded data."""
    from core.metrics import LatencyTracker
    return LatencyTracker()
