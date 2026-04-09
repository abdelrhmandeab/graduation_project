import importlib.util
import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACK_PATH = PROJECT_ROOT / "benchmarks" / "phase5_transcripts.json"
LONG_PACK_PATH = PROJECT_ROOT / "benchmarks" / "phase5_transcripts_long_horizon.json"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_phase5_dialogue.py"


class Phase5BenchmarkPackTests(unittest.TestCase):
    def test_pack_file_has_multi_turn_scenarios(self):
        payload = json.loads(PACK_PATH.read_text(encoding="utf-8"))
        scenarios = list(payload.get("scenarios") or [])

        self.assertGreaterEqual(len(scenarios), 4)
        self.assertTrue(any("اغلق" in " ".join(str(turn.get("user") or "") for turn in s.get("turns") or []) for s in scenarios))
        for scenario in scenarios:
            self.assertGreaterEqual(len(list(scenario.get("turns") or [])), 2)

    def test_runner_produces_structured_report(self):
        spec = importlib.util.spec_from_file_location("phase5_benchmark_runner", str(SCRIPT_PATH))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        pack = module._load_pack(str(PACK_PATH))
        report = module.run_pack(pack)

        self.assertIn("summary", report)
        self.assertIn("details", report)
        self.assertGreater(int((report.get("summary") or {}).get("turns_total") or 0), 0)

    def test_long_horizon_pack_contains_twenty_turn_dialogue(self):
        payload = json.loads(LONG_PACK_PATH.read_text(encoding="utf-8"))
        scenarios = list(payload.get("scenarios") or [])

        self.assertGreaterEqual(len(scenarios), 1)
        self.assertTrue(any(len(list(s.get("turns") or [])) >= 20 for s in scenarios))

    def test_long_horizon_pack_runs_with_high_pass_rate(self):
        spec = importlib.util.spec_from_file_location("phase5_benchmark_runner", str(SCRIPT_PATH))
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)

        pack = module._load_pack(str(LONG_PACK_PATH))
        report = module.run_pack(pack)
        summary = dict(report.get("summary") or {})

        self.assertGreaterEqual(int(summary.get("turns_total") or 0), 20)
        self.assertGreaterEqual(float(summary.get("pass_rate") or 0.0), 0.90)


if __name__ == "__main__":
    unittest.main()
