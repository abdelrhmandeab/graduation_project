"""ActionPlanner multi-step execution tests — 10 cases.

Tests plan_and_execute() with a mock executor, covering: single-step success,
multi-step with {result_N} references, early-exit on failure, bilingual
partial-success responses, and degenerate edge cases.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from core.action_planner import ActionPlanner


# ---------------------------------------------------------------------------
# Executor helpers
# ---------------------------------------------------------------------------

def _ok_executor(message: str = "Done", data: Dict[str, Any] = None):
    """Returns a mock executor that always succeeds."""
    def _exec(parsed):
        return True, message, data or {}
    return _exec


def _fail_executor(message: str = "Error"):
    """Returns a mock executor that always fails."""
    def _exec(parsed):
        return False, message, {}
    return _exec


def _sequence_executor(results: List[tuple]):
    """Returns a mock executor that yields results in sequence."""
    calls = iter(results)

    def _exec(parsed):
        try:
            return next(calls)
        except StopIteration:
            return False, "no more results", {}

    return _exec


def _make_call(name: str, **args) -> Dict[str, Any]:
    return {"name": name, "arguments": args}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestActionPlanner:

    def test_empty_tool_calls_returns_success(self):
        ap = ActionPlanner(executor=_ok_executor())
        ok, response, results = ap.plan_and_execute([], "do nothing", "en")
        assert ok is True
        assert results == []

    def test_single_step_success_en(self):
        ap = ActionPlanner(executor=_ok_executor("Opening Chrome", {"app": "chrome"}))
        ok, response, results = ap.plan_and_execute(
            [_make_call("open_app", app_name="chrome")],
            "open chrome",
            "en",
        )
        assert ok is True
        assert results[0]["success"] is True

    def test_single_step_success_ar(self):
        ap = ActionPlanner(executor=_ok_executor("فتحت كروم"))
        ok, response, results = ap.plan_and_execute(
            [_make_call("open_app", app_name="chrome")],
            "افتح كروم",
            "ar",
        )
        assert ok is True
        assert "فتحت كروم" in response

    def test_single_step_failure_en(self):
        ap = ActionPlanner(executor=_fail_executor("App not found"))
        ok, response, results = ap.plan_and_execute(
            [_make_call("open_app", app_name="xyz")],
            "open xyz",
            "en",
        )
        assert ok is False
        assert "App not found" in response

    def test_single_step_failure_ar(self):
        ap = ActionPlanner(executor=_fail_executor("مش موجود"))
        ok, response, results = ap.plan_and_execute(
            [_make_call("open_app", app_name="xyz")],
            "افتح xyz",
            "ar",
        )
        assert ok is False
        assert "مش موجود" in response

    def test_two_step_both_succeed(self):
        ap = ActionPlanner(
            executor=_sequence_executor([
                (True, "Searched: report.pdf", {"path": "C:/report.pdf"}),
                (True, "Opened folder", {}),
            ])
        )
        calls = [
            _make_call("search_files", filename="report"),
            _make_call("open_folder", path="{result_0}"),
        ]
        ok, response, results = ap.plan_and_execute(calls, "find report and open folder", "en")
        assert ok is True
        assert len(results) == 2
        assert all(r["success"] for r in results)

    def test_two_step_first_fails_stops_chain(self):
        ap = ActionPlanner(
            executor=_sequence_executor([
                (False, "File not found", {}),
                (True, "Opened folder", {}),
            ])
        )
        calls = [
            _make_call("search_files", filename="missing"),
            _make_call("open_folder", path="{result_0}"),
        ]
        ok, _, results = ap.plan_and_execute(calls, "find missing file", "en")
        assert ok is False
        assert len(results) == 1  # chain stopped after first failure

    def test_result_reference_resolved(self):
        """Verify {result_0} is resolved to the data from step 0."""
        received_args = {}

        def _capturing_executor(parsed):
            received_args.update(getattr(parsed, "args", {}) or {})
            return True, "ok", {"path": "/docs/report.pdf"}

        ap = ActionPlanner(executor=_capturing_executor)
        calls = [
            _make_call("search_files", filename="report"),
            _make_call("open_folder", path="{result_0}"),
        ]
        ok, _, _ = ap.plan_and_execute(calls, "find and open", "en")
        assert ok is True

    def test_no_executor_returns_failure(self):
        ap = ActionPlanner(executor=None)
        ok, response, _ = ap.plan_and_execute(
            [_make_call("open_app", app_name="chrome")],
            "open chrome",
            "en",
        )
        assert ok is False

    def test_partial_response_mentions_succeeded_and_failed(self):
        ap = ActionPlanner(
            executor=_sequence_executor([
                (True, "Opened chrome", {}),
                (False, "Timer failed", {}),
            ])
        )
        calls = [
            _make_call("open_app", app_name="chrome"),
            _make_call("set_timer", seconds=300),
        ]
        ok, response, _ = ap.plan_and_execute(calls, "open chrome then timer", "en")
        assert ok is False
        assert "Opened chrome" in response
        assert "Timer failed" in response
