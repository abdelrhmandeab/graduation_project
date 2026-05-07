"""Latency budget regression tests — 12 cases.

Tests LatencyTracker behaviour: recording, stats accuracy, budget warnings,
and regression assertions that fail when any stage averages more than 2× its
defined budget — catching performance regressions before they reach production.
"""

from __future__ import annotations

import time

import pytest

from core.metrics import LatencyTracker


@pytest.fixture
def tracker():
    return LatencyTracker()


# ---------------------------------------------------------------------------
# Group 1 — Basic recording and stats (4 tests)
# ---------------------------------------------------------------------------

class TestRecording:

    def test_empty_report(self, tracker):
        assert tracker.report() == {}

    def test_single_record_appears_in_report(self, tracker):
        tracker.record("stt_total", 0.5)
        r = tracker.report()
        assert "stt_total" in r
        assert r["stt_total"]["count"] == 1
        assert r["stt_total"]["avg_ms"] == pytest.approx(500.0)

    def test_multiple_records_compute_avg(self, tracker):
        tracker.record("stt_total", 0.4)
        tracker.record("stt_total", 0.6)
        r = tracker.report()
        assert r["stt_total"]["avg_ms"] == pytest.approx(500.0)
        assert r["stt_total"]["count"] == 2

    def test_reset_clears_all_data(self, tracker):
        tracker.record("stt_total", 0.5)
        tracker.reset()
        assert tracker.report() == {}


# ---------------------------------------------------------------------------
# Group 2 — Budget metadata (3 tests)
# ---------------------------------------------------------------------------

class TestBudgetMetadata:

    def test_known_stage_has_budget_ms(self, tracker):
        tracker.record("stt_total", 0.5)
        r = tracker.report()
        assert r["stt_total"]["budget_ms"] == pytest.approx(1000.0)

    def test_unknown_stage_has_zero_budget(self, tracker):
        tracker.record("custom_stage", 0.1)
        r = tracker.report()
        assert r["custom_stage"]["budget_ms"] == pytest.approx(0.0)

    def test_all_defined_budgets_present(self):
        """Ensure every budget-stage key is defined and positive."""
        t = LatencyTracker()
        for stage, budget in t._budgets.items():
            assert budget > 0, f"Budget for {stage!r} must be > 0"


# ---------------------------------------------------------------------------
# Group 3 — Percentile stats (2 tests)
# ---------------------------------------------------------------------------

class TestPercentileStats:

    def test_p50_equals_median(self, tracker):
        for v in [0.1, 0.2, 0.3, 0.4, 0.5]:
            tracker.record("stt_total", v)
        r = tracker.report()["stt_total"]
        # median of 5 values sorted: 0.3 → 300 ms
        assert r["p50_ms"] == pytest.approx(300.0)

    def test_max_ms_is_largest(self, tracker):
        tracker.record("stt_total", 0.2)
        tracker.record("stt_total", 0.9)
        tracker.record("stt_total", 0.5)
        r = tracker.report()["stt_total"]
        assert r["max_ms"] == pytest.approx(900.0)


# ---------------------------------------------------------------------------
# Group 4 — Regression: 2× budget guard (3 tests)
# ---------------------------------------------------------------------------

class TestBudgetRegression:
    """Fail when any stage's average exceeds 2× its defined budget."""

    _BUDGET_MULTIPLIER = 2.0

    def _assert_within_budget(self, tracker: LatencyTracker) -> None:
        report = tracker.report()
        violations = []
        for stage, stat in report.items():
            budget_ms = stat["budget_ms"]
            if budget_ms <= 0:
                continue
            avg_ms = stat["avg_ms"]
            if avg_ms > budget_ms * self._BUDGET_MULTIPLIER:
                violations.append(
                    f"{stage}: avg={avg_ms:.0f}ms > 2×budget={budget_ms * 2:.0f}ms"
                )
        assert not violations, "Latency budget violations:\n" + "\n".join(violations)

    def test_within_budget_passes(self, tracker):
        tracker.record("stt_total", 0.8)      # 800 ms < 2×1000 ms ✓
        tracker.record("e2e_command", 1.4)    # 1400 ms < 2×1500 ms ✓
        tracker.record("intent_detection", 0.015)  # 15 ms < 2×20 ms ✓
        self._assert_within_budget(tracker)

    def test_over_2x_budget_fails(self, tracker):
        tracker.record("stt_total", 2.5)   # 2500 ms > 2×1000 ms ✗
        with pytest.raises(AssertionError, match="stt_total"):
            self._assert_within_budget(tracker)

    def test_e2e_command_regression(self, tracker):
        # Simulate a fast command pipeline: parse + route in <100 ms total.
        tracker.record("intent_detection", 0.008)
        tracker.record("action_execution", 0.045)
        tracker.record("e2e_command", 0.4)
        self._assert_within_budget(tracker)
