import unittest

from core.stt_egyptian_benchmark import run_stt_egyptian_benchmark


class SttEgyptianBenchmarkTests(unittest.TestCase):
    def test_egyptian_setup_benchmark_selects_balanced_recommendation(self):
        payload = run_stt_egyptian_benchmark()

        corpus = dict(payload.get("corpus") or {})
        setups = list(payload.get("setups") or [])
        recommendation = dict(payload.get("recommendation") or {})
        baseline = dict(payload.get("baseline_vs_recommended") or {})
        done_gate = dict(payload.get("done_gate") or {})

        self.assertGreaterEqual(int(corpus.get("scenario_count") or 0), 12)
        self.assertGreaterEqual(len(setups), 2)

        self.assertTrue(str(recommendation.get("setup_id") or ""))
        self.assertNotEqual(
            str(baseline.get("baseline_setup") or ""),
            str(baseline.get("recommended_setup") or ""),
        )

        self.assertGreaterEqual(float(baseline.get("wer_gain_abs") or 0.0), 0.03)
        self.assertTrue(bool(recommendation.get("acceptable_latency")))
        self.assertTrue(bool(done_gate.get("quality_clearly_better_than_baseline")))
        self.assertTrue(bool(done_gate.get("latency_acceptable_low_mid_cpu")))
        self.assertTrue(bool(done_gate.get("passed")))

        normalization_gains = [float(item.get("normalization_wer_gain") or 0.0) for item in setups]
        self.assertTrue(any(gain > 0.01 for gain in normalization_gains))


if __name__ == "__main__":
    unittest.main()
