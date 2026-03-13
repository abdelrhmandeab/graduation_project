import json
import os
import tempfile
import time
from pathlib import Path

from core.config import (
    BENCHMARK_OUTPUT_FILE,
    BENCHMARK_SLA_P95_MS,
    BENCHMARK_SLA_SUCCESS_RATE_MIN,
    RESILIENCE_OUTPUT_FILE,
    RESILIENCE_SLA_P95_MS,
    RESILIENCE_SLA_SUCCESS_RATE_MIN,
)


def run_quick_benchmark(executor):
    scenarios = [
        ("metrics", "show metrics"),
        ("observability", "observability"),
        ("policy_status", "policy status"),
        ("persona_status", "persona status"),
        ("kb_status", "kb status"),
        ("kb_quality", "kb quality"),
        ("audit_verify", "verify audit log"),
    ]

    payload = _run_scenarios(executor, scenarios)
    payload["sla"] = _evaluate_sla(
        payload,
        p95_limit_ms=float(BENCHMARK_SLA_P95_MS),
        success_rate_min=float(BENCHMARK_SLA_SUCCESS_RATE_MIN),
    )
    _write_json(BENCHMARK_OUTPUT_FILE, payload)
    return payload


def run_resilience_demo(executor):
    scenarios = [
        ("invalid_confirmation_token", _scenario_invalid_confirmation),
        ("policy_block_write", _scenario_policy_block_write),
        ("missing_kb_file", _scenario_missing_kb_file),
        ("speech_interrupt_noop", _scenario_interrupt_no_speech),
        ("batch_rollback_recovery", _scenario_batch_rollback_recovery),
    ]

    results = []
    for name, fn in scenarios:
        started = time.perf_counter()
        ok = False
        details = ""
        error = ""
        try:
            ok, details = fn(executor)
        except Exception as exc:
            error = str(exc)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        results.append(
            {
                "name": name,
                "ok": bool(ok),
                "latency_ms": elapsed_ms,
                "details": details,
                "error": error,
            }
        )

    success_count = sum(1 for row in results if row["ok"])
    payload = {
        "timestamp": time.time(),
        "scenario_count": len(results),
        "success_count": success_count,
        "success_rate": (success_count / len(results)) if results else 0.0,
        "p50_latency_ms": _percentile([row["latency_ms"] for row in results], 50) or 0.0,
        "p95_latency_ms": _percentile([row["latency_ms"] for row in results], 95) or 0.0,
        "results": results,
    }
    payload["sla"] = _evaluate_sla(
        payload,
        p95_limit_ms=float(RESILIENCE_SLA_P95_MS),
        success_rate_min=float(RESILIENCE_SLA_SUCCESS_RATE_MIN),
    )
    _write_json(RESILIENCE_OUTPUT_FILE, payload)
    return payload


def _run_scenarios(executor, scenarios):
    results = []
    for name, command in scenarios:
        started = time.perf_counter()
        output = executor(command)
        elapsed = time.perf_counter() - started
        ok = "internal error" not in (output or "").lower()
        results.append(
            {
                "name": name,
                "command": command,
                "latency_ms": elapsed * 1000.0,
                "ok": ok,
                "output_preview": (output or "")[:240],
            }
        )

    success_count = sum(1 for row in results if row["ok"])
    return {
        "timestamp": time.time(),
        "scenario_count": len(results),
        "success_count": success_count,
        "success_rate": (success_count / len(results)) if results else 0.0,
        "p50_latency_ms": _percentile([row["latency_ms"] for row in results], 50) or 0.0,
        "p95_latency_ms": _percentile([row["latency_ms"] for row in results], 95) or 0.0,
        "results": results,
    }


def _scenario_invalid_confirmation(executor):
    output = executor("confirm abc123")
    ok = "confirmation failed" in output.lower()
    return ok, output[:220]


def _scenario_policy_block_write(executor):
    executor("policy profile strict")
    output = executor("create folder should_be_blocked")
    executor("policy profile normal")
    ok = ("blocked by policy" in output.lower()) or ("read-only mode" in output.lower())
    return ok, output[:220]


def _scenario_missing_kb_file(executor):
    output = executor("kb add c:\\this\\path\\does\\not\\exist\\missing.txt")
    ok = "file not found" in output.lower()
    return ok, output[:220]


def _scenario_interrupt_no_speech(executor):
    output = executor("stop speaking")
    ok = "no active speech" in output.lower() or "interrupted" in output.lower()
    return ok, output[:220]


def _scenario_batch_rollback_recovery(executor):
    test_root = Path(".tmp_tests")
    test_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(test_root)) as tmp:
        tmp_path = Path(tmp)
        folder = tmp_path / "batch_ok"

        executor(f"go to {tmp}")
        executor("batch plan")
        executor("batch add create folder batch_ok")
        executor(r"batch add go to C:\Windows\System32\config")
        output = executor("batch commit")
        executor("batch abort")

        ok = ("batch failed" in output.lower()) and (not folder.exists())
        return ok, output[:220]


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2)


def _evaluate_sla(payload, p95_limit_ms, success_rate_min):
    p95_latency = float(payload.get("p95_latency_ms") or 0.0)
    success_rate = float(payload.get("success_rate") or 0.0)
    checks = [
        {
            "name": "p95_latency_ms",
            "actual": p95_latency,
            "threshold": p95_limit_ms,
            "operator": "<=",
            "passed": p95_latency <= p95_limit_ms,
        },
        {
            "name": "success_rate",
            "actual": success_rate,
            "threshold": success_rate_min,
            "operator": ">=",
            "passed": success_rate >= success_rate_min,
        },
    ]
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "thresholds": {
            "p95_latency_ms": p95_limit_ms,
            "success_rate_min": success_rate_min,
        },
    }


def _percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((p / 100) * (len(ordered) - 1)))
    return float(ordered[index])
