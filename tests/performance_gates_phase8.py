import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.command_router import route_command
from core.config import (
    BENCHMARK_OUTPUT_FILE,
    BENCHMARK_SLA_P95_MS,
    BENCHMARK_SLA_SUCCESS_RATE_MIN,
    RESILIENCE_OUTPUT_FILE,
    RESILIENCE_SLA_P95_MS,
    RESILIENCE_SLA_SUCCESS_RATE_MIN,
)
from os_control.policy import policy_engine


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _percentile(values, percentile):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((percentile / 100.0) * (len(ordered) - 1)))
    return float(ordered[index])


def _is_semantic_success(command, response):
    lowered = (response or "").lower()
    checks = {
        "policy status": ["policy status", "permissions:"],
        "persona status": ["persona status", "active_profile:"],
        "voice status": ["voice status", "speech_enabled:"],
        "memory status": ["memory status", "turn_count:"],
        "show metrics": ["metrics report", "overall success rate"],
        "observability": ["observability dashboard", "command metrics:"],
        "kb status": ["knowledge base status", "file_count:"],
        "kb quality": ["knowledge quality report"],
        "verify audit log": ["audit chain is valid"],
    }
    required_markers = checks.get(command, [])
    return all(marker in lowered for marker in required_markers)


def test_runtime_route_p95_gate():
    policy_engine.set_profile("normal")

    commands = [
        "policy status",
        "persona status",
        "voice status",
        "memory status",
        "show metrics",
        "observability",
        "kb status",
        "kb quality",
        "verify audit log",
    ]

    latencies_ms = []
    success = 0
    for command in commands:
        start = time.perf_counter()
        response = route_command(command)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)
        if _is_semantic_success(command, response):
            success += 1

    success_rate = success / float(len(commands)) if commands else 0.0
    p95_ms = _percentile(latencies_ms, 95)

    _assert(
        success_rate >= float(BENCHMARK_SLA_SUCCESS_RATE_MIN),
        f"Runtime route success rate below gate: success_rate={success_rate:.4f} threshold={BENCHMARK_SLA_SUCCESS_RATE_MIN}",
    )
    _assert(
        p95_ms <= float(BENCHMARK_SLA_P95_MS),
        f"Runtime route p95 latency above gate: p95_ms={p95_ms:.2f} threshold={BENCHMARK_SLA_P95_MS}",
    )


def test_benchmark_and_resilience_sla_gates():
    policy_engine.set_profile("normal")

    benchmark_report = route_command("benchmark run")
    _assert("Benchmark Report" in benchmark_report, f"Unexpected response: {benchmark_report}")
    benchmark_payload = json.loads(Path(BENCHMARK_OUTPUT_FILE).read_text(encoding="utf-8"))
    benchmark_sla = benchmark_payload.get("sla") or {}
    _assert(bool(benchmark_sla.get("passed")), f"Benchmark SLA failed: {benchmark_sla}")

    benchmark_thresholds = benchmark_sla.get("thresholds") or {}
    _assert(
        float(benchmark_thresholds.get("p95_latency_ms") or math.inf) <= float(BENCHMARK_SLA_P95_MS),
        f"Unexpected benchmark p95 threshold: {benchmark_thresholds}",
    )
    _assert(
        float(benchmark_thresholds.get("success_rate_min") or 0.0) >= float(BENCHMARK_SLA_SUCCESS_RATE_MIN),
        f"Unexpected benchmark success threshold: {benchmark_thresholds}",
    )

    resilience_report = route_command("resilience demo")
    _assert("Resilience Report" in resilience_report, f"Unexpected response: {resilience_report}")
    resilience_payload = json.loads(Path(RESILIENCE_OUTPUT_FILE).read_text(encoding="utf-8"))
    resilience_sla = resilience_payload.get("sla") or {}
    _assert(bool(resilience_sla.get("passed")), f"Resilience SLA failed: {resilience_sla}")

    resilience_thresholds = resilience_sla.get("thresholds") or {}
    _assert(
        float(resilience_thresholds.get("p95_latency_ms") or math.inf) <= float(RESILIENCE_SLA_P95_MS),
        f"Unexpected resilience p95 threshold: {resilience_thresholds}",
    )
    _assert(
        float(resilience_thresholds.get("success_rate_min") or 0.0) >= float(RESILIENCE_SLA_SUCCESS_RATE_MIN),
        f"Unexpected resilience success threshold: {resilience_thresholds}",
    )


if __name__ == "__main__":
    test_runtime_route_p95_gate()
    test_benchmark_and_resilience_sla_gates()
    print("Phase 8 performance gate tests passed.")
