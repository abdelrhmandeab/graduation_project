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
    WAKE_BENCHMARK_HISTORY_FILE,
    WAKE_BENCHMARK_OUTPUT_FILE,
    WAKE_BENCHMARK_HISTORY_SERIES_VERSION,
    WAKE_BENCHMARK_SLA_DETECTION_RATE_MIN,
    WAKE_BENCHMARK_SLA_FALSE_POSITIVE_RATE_MAX,
    WAKE_BENCHMARK_SLA_P95_MS,
    STT_BENCHMARK_OUTPUT_FILE,
    STT_BENCHMARK_HISTORY_FILE,
    STT_BENCHMARK_SLA_P95_MS,
    STT_BENCHMARK_SLA_AVG_WER_MAX,
    STT_BENCHMARK_SLA_AVG_CER_MAX,
    STT_BENCHMARK_SLA_SUCCESS_RATE_MIN,
    STT_BENCHMARK_HISTORY_SERIES_VERSION,
    TTS_BENCHMARK_OUTPUT_FILE,
    TTS_BENCHMARK_HISTORY_FILE,
    TTS_BENCHMARK_SLA_P95_MS,
    TTS_BENCHMARK_SLA_AVG_RTF_MAX,
    TTS_BENCHMARK_SLA_AVG_QUALITY_SCORE_MIN,
    TTS_BENCHMARK_SLA_FALLBACK_RELIABILITY_MIN,
    TTS_BENCHMARK_SLA_SUCCESS_RATE_MIN,
    TTS_BENCHMARK_HISTORY_SERIES_VERSION,
    TTS_MOS_OUTPUT_FILE,
    TTS_MOS_CHECKLIST_MIN_RATINGS,
    TTS_MOS_CHECKLIST_MIN_RATERS,
)
from core.stt_benchmark import run_stt_reliability_scenarios
from core.tts_benchmark import run_tts_quality_scenarios
from core.wake_benchmark import run_wake_reliability_scenarios


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


def run_wake_reliability_benchmark(*, scenarios_per_language=None):
    payload = run_wake_reliability_scenarios(target_per_language=scenarios_per_language)
    payload["sla"] = _evaluate_wake_sla(
        payload,
        p95_limit_ms=float(WAKE_BENCHMARK_SLA_P95_MS),
        detection_rate_min=float(WAKE_BENCHMARK_SLA_DETECTION_RATE_MIN),
        false_positive_rate_max=float(WAKE_BENCHMARK_SLA_FALSE_POSITIVE_RATE_MAX),
    )
    payload["history_series"] = _wake_history_series(payload)
    payload["history"] = _update_history(WAKE_BENCHMARK_HISTORY_FILE, payload, kind="wake")
    _write_json(WAKE_BENCHMARK_OUTPUT_FILE, payload)
    return payload


def run_stt_reliability_benchmark(*, corpus_path=None, mode="auto"):
    payload = run_stt_reliability_scenarios(corpus_path=corpus_path, mode=mode)
    payload["sla"] = _evaluate_stt_sla(
        payload,
        p95_limit_ms=float(STT_BENCHMARK_SLA_P95_MS),
        avg_wer_max=float(STT_BENCHMARK_SLA_AVG_WER_MAX),
        avg_cer_max=float(STT_BENCHMARK_SLA_AVG_CER_MAX),
        success_rate_min=float(STT_BENCHMARK_SLA_SUCCESS_RATE_MIN),
    )
    payload["history_series"] = _stt_history_series(payload)
    payload["history"] = _update_history(STT_BENCHMARK_HISTORY_FILE, payload, kind="stt")
    _write_json(STT_BENCHMARK_OUTPUT_FILE, payload)
    return payload


def run_tts_quality_benchmark(*, corpus_path=None, mode="auto", backend="auto"):
    payload = run_tts_quality_scenarios(corpus_path=corpus_path, mode=mode, backend=backend)
    payload["mos_checklist"] = _evaluate_tts_mos_checklist()
    payload["sla"] = _evaluate_tts_sla(
        payload,
        p95_limit_ms=float(TTS_BENCHMARK_SLA_P95_MS),
        avg_rtf_max=float(TTS_BENCHMARK_SLA_AVG_RTF_MAX),
        avg_quality_score_min=float(TTS_BENCHMARK_SLA_AVG_QUALITY_SCORE_MIN),
        fallback_reliability_min=float(TTS_BENCHMARK_SLA_FALLBACK_RELIABILITY_MIN),
        success_rate_min=float(TTS_BENCHMARK_SLA_SUCCESS_RATE_MIN),
    )
    payload["history_series"] = _tts_history_series(payload)
    payload["history"] = _update_history(TTS_BENCHMARK_HISTORY_FILE, payload, kind="tts")
    _write_json(TTS_BENCHMARK_OUTPUT_FILE, payload)
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
    entry = {
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
    history_series = str(payload.get("history_series") or "").strip()
    if history_series:
        entry["history_series"] = history_series
    return entry


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
    latest_entry = _build_run_entry(payload, kind=kind)
    runs.append(latest_entry)

    dropped_incompatible_runs = 0
    active_series = str(latest_entry.get("history_series") or "").strip()
    if kind in {"wake", "stt", "tts"} and active_series:
        filtered_runs = [row for row in runs if str(row.get("history_series") or "").strip() == active_series]
        dropped_incompatible_runs = max(0, len(runs) - len(filtered_runs))
        runs = filtered_runs

    max_runs = max(20, int(BENCHMARK_HISTORY_MAX_RUNS))
    if len(runs) > max_runs:
        runs = runs[-max_runs:]

    daily = _rollup_runs(runs, "date_utc", max_points=int(BENCHMARK_HISTORY_MAX_DAILY_POINTS))
    weekly = _rollup_runs(runs, "week_utc", max_points=int(BENCHMARK_HISTORY_MAX_WEEKLY_POINTS))
    latest = runs[-1] if runs else {}

    payload_to_write = {
        "schema": "phase7_history_v2",
        "kind": kind,
        "updated_at": time.time(),
        "history_series": active_series,
        "latest": latest,
        "runs": runs,
        "daily": daily,
        "weekly": weekly,
    }
    _write_json(path, payload_to_write)

    return {
        "history_file": path,
        "history_series": active_series,
        "dropped_incompatible_runs": dropped_incompatible_runs,
        "run_count": len(runs),
        "daily_points": len(daily),
        "weekly_points": len(weekly),
        "latest_daily": daily[0] if daily else {},
        "latest_weekly": weekly[0] if weekly else {},
    }


def _wake_history_series(payload):
    pack = dict(payload.get("scenario_pack") or {})
    pack_name = str(pack.get("name") or "wake_reliability_pack").strip().lower()
    pack_version = str(pack.get("version") or "unknown").strip().lower()
    per_language = int(pack.get("target_per_language") or 0)
    return (
        f"wake:{WAKE_BENCHMARK_HISTORY_SERIES_VERSION}:"
        f"{pack_name}:{pack_version}:tpl={per_language}:"
        f"p95<={float(WAKE_BENCHMARK_SLA_P95_MS):.1f}:"
        f"dr>={float(WAKE_BENCHMARK_SLA_DETECTION_RATE_MIN):.2f}:"
        f"fp<={float(WAKE_BENCHMARK_SLA_FALSE_POSITIVE_RATE_MAX):.2f}"
    )


def _stt_history_series(payload):
    corpus = dict(payload.get("corpus") or {})
    corpus_name = str(corpus.get("name") or "stt_reliability_pack").strip().lower()
    corpus_version = str(corpus.get("version") or "unknown").strip().lower()
    corpus_mode = str(corpus.get("mode") or "auto").strip().lower()
    scenario_count = int(payload.get("scenario_count") or 0)
    return (
        f"stt:{STT_BENCHMARK_HISTORY_SERIES_VERSION}:"
        f"{corpus_name}:{corpus_version}:{corpus_mode}:sc={scenario_count}:"
        f"p95<={float(STT_BENCHMARK_SLA_P95_MS):.1f}:"
        f"wer<={float(STT_BENCHMARK_SLA_AVG_WER_MAX):.2f}:"
        f"cer<={float(STT_BENCHMARK_SLA_AVG_CER_MAX):.2f}:"
        f"sr>={float(STT_BENCHMARK_SLA_SUCCESS_RATE_MIN):.2f}"
    )


def _tts_history_series(payload):
    corpus = dict(payload.get("corpus") or {})
    corpus_name = str(corpus.get("name") or "tts_quality_pack").strip().lower()
    corpus_version = str(corpus.get("version") or "unknown").strip().lower()
    corpus_mode = str(corpus.get("mode") or "auto").strip().lower()
    corpus_backend = str(corpus.get("backend") or "auto").strip().lower()
    scenario_count = int(payload.get("scenario_count") or 0)
    return (
        f"tts:{TTS_BENCHMARK_HISTORY_SERIES_VERSION}:"
        f"{corpus_name}:{corpus_version}:{corpus_mode}:{corpus_backend}:sc={scenario_count}:"
        f"p95<={float(TTS_BENCHMARK_SLA_P95_MS):.1f}:"
        f"rtf<={float(TTS_BENCHMARK_SLA_AVG_RTF_MAX):.2f}:"
        f"qs>={float(TTS_BENCHMARK_SLA_AVG_QUALITY_SCORE_MIN):.2f}:"
        f"fr>={float(TTS_BENCHMARK_SLA_FALLBACK_RELIABILITY_MIN):.2f}:"
        f"sr>={float(TTS_BENCHMARK_SLA_SUCCESS_RATE_MIN):.2f}"
    )


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


def _evaluate_stt_sla(payload, p95_limit_ms, avg_wer_max, avg_cer_max, success_rate_min):
    p95_latency = float(payload.get("p95_latency_ms") or 0.0)
    avg_wer = float(payload.get("avg_wer") or 1.0)
    avg_cer = float(payload.get("avg_cer") or 1.0)
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
            "name": "avg_wer",
            "actual": avg_wer,
            "threshold": avg_wer_max,
            "operator": "<=",
            "passed": avg_wer <= avg_wer_max,
        },
        {
            "name": "avg_cer",
            "actual": avg_cer,
            "threshold": avg_cer_max,
            "operator": "<=",
            "passed": avg_cer <= avg_cer_max,
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
            "avg_wer_max": avg_wer_max,
            "avg_cer_max": avg_cer_max,
            "success_rate_min": success_rate_min,
        },
    }


def _evaluate_tts_mos_checklist():
    payload = _read_json(TTS_MOS_OUTPUT_FILE, default={})
    artifact_exists = bool(os.path.exists(TTS_MOS_OUTPUT_FILE))

    rating_count = int((payload or {}).get("rating_count") or 0)
    rater_count = int((payload or {}).get("rater_count") or 0)
    by_backend = dict((payload or {}).get("by_backend") or {})
    by_language = dict((payload or {}).get("by_language") or {})

    checks = [
        {
            "name": "artifact_exists",
            "actual": artifact_exists,
            "threshold": True,
            "operator": "==",
            "passed": artifact_exists,
        },
        {
            "name": "rating_count",
            "actual": rating_count,
            "threshold": int(TTS_MOS_CHECKLIST_MIN_RATINGS),
            "operator": ">=",
            "passed": rating_count >= int(TTS_MOS_CHECKLIST_MIN_RATINGS),
        },
        {
            "name": "rater_count",
            "actual": rater_count,
            "threshold": int(TTS_MOS_CHECKLIST_MIN_RATERS),
            "operator": ">=",
            "passed": rater_count >= int(TTS_MOS_CHECKLIST_MIN_RATERS),
        },
        {
            "name": "backend_breakdown_present",
            "actual": bool(by_backend),
            "threshold": True,
            "operator": "==",
            "passed": bool(by_backend),
        },
        {
            "name": "language_breakdown_present",
            "actual": bool(by_language),
            "threshold": True,
            "operator": "==",
            "passed": bool(by_language),
        },
    ]
    return {
        "passed": all(item.get("passed") for item in checks),
        "checks": checks,
        "source": str((payload or {}).get("source_csv") or ""),
        "rating_count": rating_count,
        "rater_count": rater_count,
    }


def _evaluate_tts_sla(payload, p95_limit_ms, avg_rtf_max, avg_quality_score_min, fallback_reliability_min, success_rate_min):
    p95_latency = float(payload.get("p95_latency_ms") or 0.0)
    avg_rtf = float(payload.get("avg_rtf") or 99.0)
    avg_quality_score = float(payload.get("avg_quality_score") or 0.0)
    fallback_reliability = float(payload.get("fallback_reliability") or 0.0)
    success_rate = float(payload.get("success_rate") or 0.0)
    mos_checklist = dict(payload.get("mos_checklist") or {})
    mos_passed = bool(mos_checklist.get("passed"))
    checks = [
        {
            "name": "p95_latency_ms",
            "actual": p95_latency,
            "threshold": p95_limit_ms,
            "operator": "<=",
            "passed": p95_latency <= p95_limit_ms,
        },
        {
            "name": "avg_rtf",
            "actual": avg_rtf,
            "threshold": avg_rtf_max,
            "operator": "<=",
            "passed": avg_rtf <= avg_rtf_max,
        },
        {
            "name": "avg_quality_score",
            "actual": avg_quality_score,
            "threshold": avg_quality_score_min,
            "operator": ">=",
            "passed": avg_quality_score >= avg_quality_score_min,
        },
        {
            "name": "fallback_reliability",
            "actual": fallback_reliability,
            "threshold": fallback_reliability_min,
            "operator": ">=",
            "passed": fallback_reliability >= fallback_reliability_min,
        },
        {
            "name": "success_rate",
            "actual": success_rate,
            "threshold": success_rate_min,
            "operator": ">=",
            "passed": success_rate >= success_rate_min,
        },
        {
            "name": "mos_checklist",
            "actual": mos_passed,
            "threshold": True,
            "operator": "==",
            "passed": mos_passed,
        },
    ]
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "thresholds": {
            "p95_latency_ms": p95_limit_ms,
            "avg_rtf_max": avg_rtf_max,
            "avg_quality_score_min": avg_quality_score_min,
            "fallback_reliability_min": fallback_reliability_min,
            "success_rate_min": success_rate_min,
        },
    }


def _evaluate_wake_sla(payload, p95_limit_ms, detection_rate_min, false_positive_rate_max):
    p95_latency = float(payload.get("p95_latency_ms") or 0.0)
    detection_rate = float(payload.get("detection_rate") or 0.0)
    false_positive_rate = float(payload.get("false_positive_rate") or 0.0)
    checks = [
        {
            "name": "p95_latency_ms",
            "actual": p95_latency,
            "threshold": p95_limit_ms,
            "operator": "<=",
            "passed": p95_latency <= p95_limit_ms,
        },
        {
            "name": "detection_rate",
            "actual": detection_rate,
            "threshold": detection_rate_min,
            "operator": ">=",
            "passed": detection_rate >= detection_rate_min,
        },
        {
            "name": "false_positive_rate",
            "actual": false_positive_rate,
            "threshold": false_positive_rate_max,
            "operator": "<=",
            "passed": false_positive_rate <= false_positive_rate_max,
        },
    ]
    return {
        "passed": all(item["passed"] for item in checks),
        "checks": checks,
        "thresholds": {
            "p95_latency_ms": p95_limit_ms,
            "detection_rate_min": detection_rate_min,
            "false_positive_rate_max": false_positive_rate_max,
        },
    }


def _percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((p / 100) * (len(ordered) - 1)))
    return float(ordered[index])
