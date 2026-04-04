import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.command_router import route_command
from core.doctor import collect_diagnostics, format_diagnostics_report
from core.metrics import metrics
from os_control.policy import policy_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def test_language_and_intent_metrics_snapshot():
    policy_engine.set_profile("normal")
    metrics.reset()

    en_response = route_command("voice status")
    _assert("Voice Status" in en_response, f"Unexpected response: {en_response}")

    ar_response = route_command("حالة الصوت")
    _assert("Voice Status" in ar_response, f"Unexpected response: {ar_response}")

    metrics.record_diagnostic("doctor_startup", True, 0.1)

    snap = metrics.snapshot()
    _assert("languages" in snap, f"Missing language metrics: {snap}")
    _assert("en" in snap["languages"], f"Missing EN language bucket: {snap}")
    _assert("ar" in snap["languages"], f"Missing AR language bucket: {snap}")
    _assert("intent_language" in snap, f"Missing intent-language metrics: {snap}")
    _assert("VOICE_COMMAND" in snap["intent_language"], f"Missing VOICE_COMMAND bucket: {snap}")
    _assert("en" in snap["intent_language"]["VOICE_COMMAND"], f"Missing VOICE_COMMAND[en]: {snap}")
    _assert("ar" in snap["intent_language"]["VOICE_COMMAND"], f"Missing VOICE_COMMAND[ar]: {snap}")
    _assert("diagnostics" in snap and "doctor_startup" in snap["diagnostics"], f"Missing diagnostics metrics: {snap}")

    dashboard = route_command("observability")
    _assert("Language Metrics:" in dashboard, f"Unexpected observability output: {dashboard}")
    _assert("Intent/Language Metrics:" in dashboard, f"Unexpected observability output: {dashboard}")
    _assert("Diagnostics Metrics:" in dashboard, f"Unexpected observability output: {dashboard}")


def test_benchmark_and_resilience_history_rollups():
    policy_engine.set_profile("normal")

    benchmark_history = PROJECT_ROOT / "jarvis_benchmark_history.json"
    resilience_history = PROJECT_ROOT / "jarvis_resilience_history.json"
    if benchmark_history.exists():
        benchmark_history.unlink()
    if resilience_history.exists():
        resilience_history.unlink()

    benchmark_report = route_command("benchmark run")
    _assert("Benchmark Report" in benchmark_report, f"Unexpected response: {benchmark_report}")
    _assert("history_file: jarvis_benchmark_history.json" in benchmark_report, f"Unexpected response: {benchmark_report}")
    _assert(benchmark_history.exists(), "Expected benchmark history file to be created")

    benchmark_payload = json.loads(benchmark_history.read_text(encoding="utf-8"))
    _assert("runs" in benchmark_payload and benchmark_payload["runs"], f"Unexpected benchmark history payload: {benchmark_payload}")
    _assert("daily" in benchmark_payload and benchmark_payload["daily"], f"Unexpected benchmark history payload: {benchmark_payload}")
    _assert("weekly" in benchmark_payload and benchmark_payload["weekly"], f"Unexpected benchmark history payload: {benchmark_payload}")

    resilience_report = route_command("resilience demo")
    _assert("Resilience Report" in resilience_report, f"Unexpected response: {resilience_report}")
    _assert("history_file: jarvis_resilience_history.json" in resilience_report, f"Unexpected response: {resilience_report}")
    _assert(resilience_history.exists(), "Expected resilience history file to be created")

    resilience_payload = json.loads(resilience_history.read_text(encoding="utf-8"))
    _assert("runs" in resilience_payload and resilience_payload["runs"], f"Unexpected resilience history payload: {resilience_payload}")
    _assert("daily" in resilience_payload and resilience_payload["daily"], f"Unexpected resilience history payload: {resilience_payload}")
    _assert("weekly" in resilience_payload and resilience_payload["weekly"], f"Unexpected resilience history payload: {resilience_payload}")


def test_doctor_payload_and_report_format():
    payload = collect_diagnostics(include_model_load_checks=False)
    _assert("check_count" in payload and int(payload["check_count"]) > 0, f"Unexpected doctor payload: {payload}")
    _assert("checks" in payload and isinstance(payload["checks"], list), f"Unexpected doctor payload: {payload}")

    report = format_diagnostics_report(payload)
    _assert("Jarvis Realtime Doctor" in report, f"Unexpected doctor report: {report}")
    _assert("checks:" in report, f"Unexpected doctor report: {report}")


if __name__ == "__main__":
    test_language_and_intent_metrics_snapshot()
    test_benchmark_and_resilience_history_rollups()
    test_doctor_payload_and_report_format()
    print("Phase 7 smoke tests passed.")
