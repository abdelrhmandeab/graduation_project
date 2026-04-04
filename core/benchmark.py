import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from core.config import (
    BENCHMARK_HISTORY_FILE,
    BENCHMARK_HISTORY_MAX_DAILY_POINTS,
    BENCHMARK_HISTORY_MAX_RUNS,
    BENCHMARK_HISTORY_MAX_WEEKLY_POINTS,
    BENCHMARK_OUTPUT_FILE,
    BENCHMARK_SLA_P95_MS,
    BENCHMARK_SLA_SUCCESS_RATE_MIN,
    RESILIENCE_HISTORY_FILE,
    RESILIENCE_OUTPUT_FILE,
    RESILIENCE_SLA_P95_MS,
    RESILIENCE_SLA_SUCCESS_RATE_MIN,
)


_BENCHMARK_EXPECTED_MARKERS = {
    "metrics": ("metrics report", "overall success rate"),
    "observability": ("observability dashboard", "command metrics:"),
    "policy_status": ("policy status", "permissions:"),
    "persona_status": ("persona status", "active_profile:"),
    "kb_status": ("knowledge base status", "file_count:"),
    "kb_quality": ("knowledge quality report", "ok="),
    "audit_verify": ("audit chain is valid",),
}


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
    payload["history"] = _update_history(BENCHMARK_HISTORY_FILE, payload, kind="benchmark")
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
    payload["history"] = _update_history(RESILIENCE_HISTORY_FILE, payload, kind="resilience")
    _write_json(RESILIENCE_OUTPUT_FILE, payload)
    return payload


def _run_scenarios(executor, scenarios):
    results = []
    for name, command in scenarios:
        started = time.perf_counter()
        output = executor(command)
        elapsed = time.perf_counter() - started
        ok = _is_benchmark_result_ok(name, output)
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


def _is_benchmark_result_ok(name, output):
    lowered = (output or "").lower()
    if "internal error" in lowered:
        return False

    markers = _BENCHMARK_EXPECTED_MARKERS.get(name)
    if markers:
        return all(marker in lowered for marker in markers)

    return bool((output or "").strip())


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
    temp_root = Path(".tmp_workspace")
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(temp_root)) as tmp:
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


def _read_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default if default is not None else {}


def _build_run_entry(payload, kind):
    timestamp = float(payload.get("timestamp") or time.time())
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    iso_year, iso_week, _iso_weekday = dt.isocalendar()
    return {
        "timestamp": timestamp,
        "kind": kind,
        "date_utc": dt.strftime("%Y-%m-%d"),
        "week_utc": f"{iso_year}-W{iso_week:02d}",
        "scenario_count": int(payload.get("scenario_count") or 0),
        "success_count": int(payload.get("success_count") or 0),
        "success_rate": float(payload.get("success_rate") or 0.0),
        "p50_latency_ms": float(payload.get("p50_latency_ms") or 0.0),
        "p95_latency_ms": float(payload.get("p95_latency_ms") or 0.0),
        "sla_passed": bool((payload.get("sla") or {}).get("passed")),
    }


def _rollup_runs(runs, key_name, max_points):
    grouped = {}
    for run in runs:
        key = str(run.get(key_name) or "")
        if not key:
            continue
        bucket = grouped.setdefault(
            key,
            {
                "count": 0,
                "scenario_count_total": 0,
                "success_rate_total": 0.0,
                "p95_latency_total": 0.0,
                "max_p95_latency_ms": 0.0,
                "min_success_rate": 1.0,
                "sla_pass_count": 0,
                "last_timestamp": 0.0,
            },
        )
        success_rate = float(run.get("success_rate") or 0.0)
        p95_latency_ms = float(run.get("p95_latency_ms") or 0.0)
        bucket["count"] += 1
        bucket["scenario_count_total"] += int(run.get("scenario_count") or 0)
        bucket["success_rate_total"] += success_rate
        bucket["p95_latency_total"] += p95_latency_ms
        bucket["max_p95_latency_ms"] = max(bucket["max_p95_latency_ms"], p95_latency_ms)
        bucket["min_success_rate"] = min(bucket["min_success_rate"], success_rate)
        if bool(run.get("sla_passed")):
            bucket["sla_pass_count"] += 1
        bucket["last_timestamp"] = max(bucket["last_timestamp"], float(run.get("timestamp") or 0.0))

    rows = []
    for key in sorted(grouped.keys(), reverse=True):
        bucket = grouped[key]
        count = int(bucket["count"])
        rows.append(
            {
                key_name: key,
                "count": count,
                "scenario_count_total": int(bucket["scenario_count_total"]),
                "avg_success_rate": (float(bucket["success_rate_total"]) / float(count)) if count else 0.0,
                "min_success_rate": float(bucket["min_success_rate"]) if count else 0.0,
                "avg_p95_latency_ms": (float(bucket["p95_latency_total"]) / float(count)) if count else 0.0,
                "max_p95_latency_ms": float(bucket["max_p95_latency_ms"]),
                "sla_pass_rate": (float(bucket["sla_pass_count"]) / float(count)) if count else 0.0,
                "last_timestamp": float(bucket["last_timestamp"]),
            }
        )
    return rows[: max(1, int(max_points))]


def _update_history(path, payload, *, kind):
    history = _read_json(path, default={})
    runs = list(history.get("runs") or [])
    runs.append(_build_run_entry(payload, kind=kind))

    max_runs = max(20, int(BENCHMARK_HISTORY_MAX_RUNS))
    if len(runs) > max_runs:
        runs = runs[-max_runs:]

    daily = _rollup_runs(runs, "date_utc", max_points=int(BENCHMARK_HISTORY_MAX_DAILY_POINTS))
    weekly = _rollup_runs(runs, "week_utc", max_points=int(BENCHMARK_HISTORY_MAX_WEEKLY_POINTS))
    latest = runs[-1] if runs else {}

    payload_to_write = {
        "schema": "phase7_history_v1",
        "kind": kind,
        "updated_at": time.time(),
        "latest": latest,
        "runs": runs,
        "daily": daily,
        "weekly": weekly,
    }
    _write_json(path, payload_to_write)

    return {
        "history_file": path,
        "run_count": len(runs),
        "daily_points": len(daily),
        "weekly_points": len(weekly),
        "latest_daily": daily[0] if daily else {},
        "latest_weekly": weekly[0] if weekly else {},
    }


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
