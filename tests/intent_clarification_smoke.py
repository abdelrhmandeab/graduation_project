import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.command_classifier import classify_with_confidence
from core.command_router import route_command
from core.session_memory import session_memory
from os_control.app_ops import resolve_app_request


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


@contextmanager
def _workspace_tempdir():
    base = Path(__file__).resolve().parents[1] / ".tmp_tests"
    base.mkdir(parents=True, exist_ok=True)
    temp_path = base / f"case_{uuid.uuid4().hex}"
    temp_path.mkdir(parents=True, exist_ok=True)
    try:
        yield str(temp_path)
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def test_classifier_confidence_for_ambiguous_app_command():
    result = classify_with_confidence("open power", language="en")
    _assert(result["intent"] == "OS_APP_OPEN", f"Unexpected result: {result}")
    _assert(result["should_clarify"], f"Expected clarification for ambiguous app command: {result}")
    _assert(result["reason"] == "app_name_ambiguous", f"Unexpected reason: {result}")
    _assert(result.get("entity_scores"), f"Expected entity scores: {result}")


def test_router_asks_for_app_clarification():
    first = route_command("open power")
    _assert("multiple app matches" in first.lower(), f"Unexpected response: {first}")
    _assert("1)" in first and "2)" in first, f"Unexpected response: {first}")
    route_command("cancel")


def test_router_reprompts_on_unclear_short_reply():
    first = route_command("open power")
    _assert("multiple app matches" in first.lower(), f"Unexpected response: {first}")
    second = route_command("maybe")
    _assert("multiple app matches" in second.lower(), f"Expected clarification re-prompt: {second}")
    route_command("cancel")


def test_router_can_cancel_clarification():
    first = route_command("open power")
    _assert("multiple app matches" in first.lower(), f"Unexpected response: {first}")
    second = route_command("cancel")
    _assert("Clarification cancelled." in second, f"Unexpected cancel response: {second}")


def test_multiple_actions_trigger_clarification():
    response = route_command("open calculator and delete temp.txt")
    _assert("more than one action" in response.lower(), f"Unexpected response: {response}")


def test_memory_status_exposes_pending_clarification_flag():
    route_command("open power")
    status = session_memory.status()
    _assert("pending_clarification" in status, f"Unexpected status payload: {status}")
    _assert(status["pending_clarification"] is True, f"Expected pending clarification True: {status}")
    route_command("cancel")


def test_app_resolver_returns_ambiguous_for_power():
    result = resolve_app_request("power")
    _assert(result["status"] == "ambiguous", f"Expected ambiguous resolution: {result}")
    _assert(len(result["candidates"]) >= 2, f"Expected at least two candidates: {result}")


def test_file_search_multimatch_clarification_and_resolution():
    with _workspace_tempdir() as tmp:
        base = Path(tmp)
        (base / "report_1.txt").write_text("a", encoding="utf-8")
        (base / "report_2.txt").write_text("b", encoding="utf-8")
        route_command(f"go to {tmp}")

        first = route_command("find file report")
        _assert("multiple files" in first.lower(), f"Unexpected response: {first}")
        _assert("1)" in first and "2)" in first, f"Unexpected response: {first}")

        second = route_command("1")
        _assert("Path:" in second, f"Expected file metadata after choosing result: {second}")


if __name__ == "__main__":
    test_classifier_confidence_for_ambiguous_app_command()
    test_router_asks_for_app_clarification()
    test_router_reprompts_on_unclear_short_reply()
    test_router_can_cancel_clarification()
    test_multiple_actions_trigger_clarification()
    test_memory_status_exposes_pending_clarification_flag()
    test_app_resolver_returns_ambiguous_for_power()
    test_file_search_multimatch_clarification_and_resolution()
    print("Intent clarification smoke tests passed.")
