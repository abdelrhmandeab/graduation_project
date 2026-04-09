import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _dig(payload, path, default=None):
    current = payload
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def _parse_timestamp(raw_value):
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        if value > 0:
            return value
        return None

    text = str(raw_value or "").strip()
    if not text:
        return None

    try:
        return float(text)
    except Exception:
        pass

    normalized = text.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except Exception:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def _build_specs(minimums):
    return [
        {
            "name": "wake",
            "path": "jarvis_wake_benchmark.json",
            "timestamp_path": ["timestamp"],
            "count_path": ["scenario_count"],
            "min_count": int(minimums["wake"]),
            "require_sla": True,
        },
        {
            "name": "stt",
            "path": "jarvis_stt_benchmark.json",
            "timestamp_path": ["timestamp"],
            "count_path": ["scenario_count"],
            "min_count": int(minimums["stt"]),
            "require_sla": True,
        },
        {
            "name": "tts",
            "path": "jarvis_tts_benchmark.json",
            "timestamp_path": ["timestamp"],
            "count_path": ["scenario_count"],
            "min_count": int(minimums["tts"]),
            "require_sla": True,
        },
        {
            "name": "phase5_dialogue",
            "path": "jarvis_phase5_dialogue_benchmark.json",
            "timestamp_path": ["timestamp_utc"],
            "count_path": ["summary", "turns_total"],
            "min_count": int(minimums["phase5"]),
            "require_sla": False,
        },
        {
            "name": "phase5_long_horizon",
            "path": "jarvis_phase5_dialogue_long_horizon_benchmark.json",
            "timestamp_path": ["timestamp_utc"],
            "count_path": ["summary", "turns_total"],
            "min_count": int(minimums["phase5_long"]),
            "require_sla": False,
        },
    ]


def evaluate_freshness(*, root_dir, max_age_hours, minimums=None, now_ts=None):
    if minimums is None:
        minimums = {
            "wake": 20,
            "stt": 10,
            "tts": 10,
            "phase5": 8,
            "phase5_long": 20,
        }

    checks = []
    root = Path(root_dir)
    current_ts = float(now_ts if now_ts is not None else time.time())
    max_age_seconds = float(max_age_hours) * 3600.0

    for spec in _build_specs(minimums):
        path = root / spec["path"]
        file_result = {
            "name": spec["name"],
            "path": str(path),
            "exists": path.exists() and path.is_file(),
            "passed": False,
            "issues": [],
        }

        if not file_result["exists"]:
            file_result["issues"].append("missing_artifact")
            checks.append(file_result)
            continue

        payload = _read_json(path)
        if not isinstance(payload, dict):
            file_result["issues"].append("invalid_json")
            checks.append(file_result)
            continue

        timestamp_value = _parse_timestamp(_dig(payload, spec["timestamp_path"]))
        if timestamp_value is None:
            file_result["issues"].append("missing_timestamp")
        else:
            age_seconds = max(0.0, current_ts - float(timestamp_value))
            file_result["age_hours"] = age_seconds / 3600.0
            if age_seconds > max_age_seconds:
                file_result["issues"].append("stale_artifact")

        scenario_count = int(_dig(payload, spec["count_path"], 0) or 0)
        file_result["scenario_count"] = scenario_count
        if scenario_count < int(spec["min_count"]):
            file_result["issues"].append("below_minimum_scenarios")

        if spec["require_sla"]:
            sla_passed = bool(_dig(payload, ["sla", "passed"], False))
            file_result["sla_passed"] = sla_passed
            if not sla_passed:
                file_result["issues"].append("sla_failed")

        if spec["name"] == "tts":
            fallback_reliability = float(payload.get("fallback_reliability") or 0.0)
            file_result["fallback_reliability"] = fallback_reliability
            if fallback_reliability < 0.95:
                file_result["issues"].append("fallback_reliability_too_low")

            mos_passed = bool(_dig(payload, ["mos_checklist", "passed"], False))
            file_result["mos_checklist_passed"] = mos_passed
            if not mos_passed:
                file_result["issues"].append("mos_checklist_failed")

        file_result["passed"] = not file_result["issues"]
        checks.append(file_result)

    passed = all(item.get("passed") for item in checks)
    return {
        "passed": passed,
        "max_age_hours": float(max_age_hours),
        "checks": checks,
    }


def _print_report(report):
    print("Benchmark Freshness Policy")
    print("--------------------------")
    print(f"max_age_hours: {float(report.get('max_age_hours') or 0.0):.1f}")

    for item in list(report.get("checks") or []):
        status = "PASS" if bool(item.get("passed")) else "FAIL"
        issue_text = "none" if not item.get("issues") else ",".join(item.get("issues"))
        print(
            "[{status}] {name} scenarios={scenario_count} issues={issues} path={path}".format(
                status=status,
                name=str(item.get("name") or "unknown"),
                scenario_count=int(item.get("scenario_count") or 0),
                issues=issue_text,
                path=str(item.get("path") or ""),
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Validate benchmark artifact freshness and minimum scenario coverage."
    )
    parser.add_argument("--max-age-hours", type=float, default=168.0, help="Maximum artifact age in hours.")
    parser.add_argument("--wake-min-scenarios", type=int, default=20)
    parser.add_argument("--stt-min-scenarios", type=int, default=10)
    parser.add_argument("--tts-min-scenarios", type=int, default=10)
    parser.add_argument("--phase5-min-turns", type=int, default=8)
    parser.add_argument("--phase5-long-min-turns", type=int, default=20)
    args = parser.parse_args()

    minimums = {
        "wake": max(1, int(args.wake_min_scenarios)),
        "stt": max(1, int(args.stt_min_scenarios)),
        "tts": max(1, int(args.tts_min_scenarios)),
        "phase5": max(1, int(args.phase5_min_turns)),
        "phase5_long": max(1, int(args.phase5_long_min_turns)),
    }

    report = evaluate_freshness(
        root_dir=PROJECT_ROOT,
        max_age_hours=float(args.max_age_hours),
        minimums=minimums,
    )
    _print_report(report)

    if bool(report.get("passed")):
        print("policy_result: passed")
        return 0

    print("policy_result: failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
