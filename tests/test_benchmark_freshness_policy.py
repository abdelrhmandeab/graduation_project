import importlib.util
import json
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory


def _load_freshness_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "check_benchmark_freshness.py"
    spec = importlib.util.spec_from_file_location("check_benchmark_freshness", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class BenchmarkFreshnessPolicyTests(unittest.TestCase):
    def test_policy_passes_for_fresh_artifacts(self):
        module = _load_freshness_module()
        now = time.time()
        now_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            _write_json(
                root / "jarvis_wake_benchmark.json",
                {
                    "timestamp": now,
                    "scenario_count": 24,
                    "sla": {"passed": True},
                },
            )
            _write_json(
                root / "jarvis_stt_benchmark.json",
                {
                    "timestamp": now,
                    "scenario_count": 12,
                    "avg_cer": 0.12,
                    "sla": {"passed": True},
                },
            )
            _write_json(
                root / "jarvis_tts_benchmark.json",
                {
                    "timestamp": now,
                    "scenario_count": 12,
                    "fallback_reliability": 1.0,
                    "mos_checklist": {"passed": True},
                    "sla": {"passed": True},
                },
            )
            _write_json(
                root / "jarvis_phase5_dialogue_benchmark.json",
                {
                    "timestamp_utc": now_iso,
                    "summary": {"turns_total": 10},
                },
            )
            _write_json(
                root / "jarvis_phase5_dialogue_long_horizon_benchmark.json",
                {
                    "timestamp_utc": now_iso,
                    "summary": {"turns_total": 24},
                },
            )

            report = module.evaluate_freshness(
                root_dir=root,
                max_age_hours=48.0,
                now_ts=now,
            )
            self.assertTrue(bool(report.get("passed")), msg=str(report))

    def test_policy_fails_for_stale_or_underfilled_artifacts(self):
        module = _load_freshness_module()
        now = time.time()
        stale = now - (72.0 * 3600.0)

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)

            _write_json(
                root / "jarvis_wake_benchmark.json",
                {
                    "timestamp": stale,
                    "scenario_count": 24,
                    "sla": {"passed": True},
                },
            )
            _write_json(
                root / "jarvis_stt_benchmark.json",
                {
                    "timestamp": now,
                    "scenario_count": 5,
                    "avg_cer": 0.12,
                    "sla": {"passed": True},
                },
            )
            _write_json(
                root / "jarvis_tts_benchmark.json",
                {
                    "timestamp": now,
                    "scenario_count": 12,
                    "fallback_reliability": 0.50,
                    "mos_checklist": {"passed": False},
                    "sla": {"passed": False},
                },
            )
            _write_json(
                root / "jarvis_phase5_dialogue_benchmark.json",
                {
                    "timestamp_utc": "2026-01-01T00:00:00+00:00",
                    "summary": {"turns_total": 4},
                },
            )
            _write_json(
                root / "jarvis_phase5_dialogue_long_horizon_benchmark.json",
                {
                    "timestamp_utc": "2026-01-01T00:00:00+00:00",
                    "summary": {"turns_total": 12},
                },
            )

            report = module.evaluate_freshness(
                root_dir=root,
                max_age_hours=24.0,
                now_ts=now,
            )
            self.assertFalse(bool(report.get("passed")))

            checks = {item.get("name"): item for item in list(report.get("checks") or [])}
            self.assertIn("stale_artifact", list(checks["wake"].get("issues") or []))
            self.assertIn("below_minimum_scenarios", list(checks["stt"].get("issues") or []))
            self.assertIn("fallback_reliability_too_low", list(checks["tts"].get("issues") or []))


if __name__ == "__main__":
    unittest.main()
