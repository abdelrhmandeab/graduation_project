import json
import re
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.command_router import route_command
from core.config import BENCHMARK_HISTORY_FILE, BENCHMARK_OUTPUT_FILE, RESILIENCE_HISTORY_FILE, RESILIENCE_OUTPUT_FILE
from core.session_memory import session_memory
from os_control.policy import policy_engine


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
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def _extract_token(text):
    match = re.search(r"confirm\s+([0-9a-f]{6})", text, flags=re.IGNORECASE)
    return match.group(1).lower().strip() if match else ""


def test_bilingual_followup_and_high_risk_confirmation_flow():
    policy_engine.set_profile("normal")
    route_command("memory clear")

    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        target = tmp / "phase8_followup.txt"
        target.write_text("phase8-flow", encoding="utf-8")

        inspect_response = route_command("file info phase8_followup.txt")
        _assert("Path:" in inspect_response, f"Unexpected response: {inspect_response}")

        delete_prompt = route_command("احذفه")
        _assert("Confirmation required" in delete_prompt, f"Unexpected response: {delete_prompt}")
        token = _extract_token(delete_prompt)
        _assert(token, f"Token missing in response: {delete_prompt}")

        followup_confirm = route_command("confirm it")
        _assert("Second factor required" in followup_confirm, f"Unexpected response: {followup_confirm}")

        final_confirm = route_command(f"confirm {token} 2468")
        _assert("Deleted" in final_confirm, f"Unexpected response: {final_confirm}")
        _assert(not target.exists(), "Target file should be deleted")


def test_file_search_clarification_resolution_end_to_end():
    policy_engine.set_profile("normal")
    route_command("memory clear")

    with _workspace_tempdir() as tmp:
        (tmp / "report_1.txt").write_text("alpha", encoding="utf-8")
        (tmp / "report_2.txt").write_text("beta", encoding="utf-8")
        route_command(f"go to {tmp}")

        prompt = route_command("find file report")
        _assert("multiple files" in prompt.lower(), f"Unexpected response: {prompt}")
        _assert("1)" in prompt and "2)" in prompt, f"Unexpected response: {prompt}")

        selection = route_command("1")
        _assert("Path:" in selection, f"Unexpected response: {selection}")

        status = session_memory.status()
        _assert(status.get("pending_clarification") is False, f"Unexpected memory status: {status}")


def test_benchmark_observability_and_history_roundtrip():
    policy_engine.set_profile("normal")

    benchmark_history = Path(BENCHMARK_HISTORY_FILE)
    resilience_history = Path(RESILIENCE_HISTORY_FILE)
    if benchmark_history.exists():
        benchmark_history.unlink()
    if resilience_history.exists():
        resilience_history.unlink()

    benchmark_response = route_command("benchmark run")
    _assert("Benchmark Report" in benchmark_response, f"Unexpected response: {benchmark_response}")
    _assert("sla_passed: True" in benchmark_response, f"Unexpected response: {benchmark_response}")

    resilience_response = route_command("resilience demo")
    _assert("Resilience Report" in resilience_response, f"Unexpected response: {resilience_response}")
    _assert("sla_passed: True" in resilience_response, f"Unexpected response: {resilience_response}")

    dashboard = route_command("observability")
    _assert("Observability Dashboard" in dashboard, f"Unexpected response: {dashboard}")
    _assert("Language Metrics:" in dashboard, f"Unexpected response: {dashboard}")

    benchmark_payload = json.loads(Path(BENCHMARK_OUTPUT_FILE).read_text(encoding="utf-8"))
    resilience_payload = json.loads(Path(RESILIENCE_OUTPUT_FILE).read_text(encoding="utf-8"))
    _assert(bool((benchmark_payload.get("sla") or {}).get("passed")), f"Benchmark SLA not passed: {benchmark_payload}")
    _assert(bool((resilience_payload.get("sla") or {}).get("passed")), f"Resilience SLA not passed: {resilience_payload}")

    benchmark_history_payload = json.loads(benchmark_history.read_text(encoding="utf-8"))
    resilience_history_payload = json.loads(resilience_history.read_text(encoding="utf-8"))
    for payload in (benchmark_history_payload, resilience_history_payload):
        _assert(payload.get("runs"), f"Missing runs in history payload: {payload}")
        _assert(payload.get("daily"), f"Missing daily rollup in history payload: {payload}")
        _assert(payload.get("weekly"), f"Missing weekly rollup in history payload: {payload}")


if __name__ == "__main__":
    test_bilingual_followup_and_high_risk_confirmation_flow()
    test_file_search_clarification_resolution_end_to_end()
    test_benchmark_observability_and_history_roundtrip()
    print("Phase 8 end-to-end regression tests passed.")
