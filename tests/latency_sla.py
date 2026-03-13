import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.command_router import route_command
from core.config import BENCHMARK_OUTPUT_FILE, RESILIENCE_OUTPUT_FILE
from os_control.policy import policy_engine


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def test_benchmark_and_resilience_sla():
    policy_engine.set_profile("normal")

    benchmark_response = route_command("benchmark run")
    _assert("Benchmark Report" in benchmark_response, f"Unexpected response: {benchmark_response}")
    _assert("sla_passed:" in benchmark_response, f"Unexpected response: {benchmark_response}")

    benchmark_payload = json.loads(Path(BENCHMARK_OUTPUT_FILE).read_text(encoding="utf-8"))
    _assert("sla" in benchmark_payload, "Benchmark payload missing SLA block")
    _assert(isinstance(benchmark_payload["sla"].get("checks"), list), "Benchmark SLA checks missing")
    _assert(benchmark_payload["sla"].get("passed") is True, "Benchmark SLA failed")

    resilience_response = route_command("resilience demo")
    _assert("Resilience Report" in resilience_response, f"Unexpected response: {resilience_response}")
    _assert("sla_passed:" in resilience_response, f"Unexpected response: {resilience_response}")

    resilience_payload = json.loads(Path(RESILIENCE_OUTPUT_FILE).read_text(encoding="utf-8"))
    _assert("sla" in resilience_payload, "Resilience payload missing SLA block")
    _assert(isinstance(resilience_payload["sla"].get("checks"), list), "Resilience SLA checks missing")
    _assert(resilience_payload["sla"].get("passed") is True, "Resilience SLA failed")


if __name__ == "__main__":
    test_benchmark_and_resilience_sla()
    print("Latency SLA tests passed.")
