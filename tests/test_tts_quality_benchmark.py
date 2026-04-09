import unittest

from core.benchmark import run_tts_quality_benchmark


class TtsQualityBenchmarkTests(unittest.TestCase):
    def test_tts_quality_benchmark_payload_and_sla(self):
        payload = run_tts_quality_benchmark(mode="mock", backend="auto")

        self.assertGreaterEqual(int(payload.get("scenario_count") or 0), 10)
        self.assertEqual(int(payload.get("evaluated_count") or 0), int(payload.get("scenario_count") or 0))
        self.assertEqual(int(payload.get("real_scenario_count") or 0), 0)
        self.assertGreaterEqual(int(payload.get("mock_scenario_count") or 0), 10)

        self.assertIn("results", payload)
        self.assertIn("sla", payload)
        self.assertIn("history", payload)
        self.assertTrue(str(payload.get("history_series") or "").startswith("tts:"))

        success_rate = float(payload.get("success_rate") or 0.0)
        avg_quality_score = float(payload.get("avg_quality_score") or 0.0)
        avg_rtf = float(payload.get("avg_rtf") or 99.0)
        fallback_reliability = float(payload.get("fallback_reliability") or 0.0)
        p95_latency = float(payload.get("p95_latency_ms") or 0.0)
        mos_checklist = dict(payload.get("mos_checklist") or {})

        self.assertGreaterEqual(success_rate, 0.90)
        self.assertGreaterEqual(avg_quality_score, 0.70)
        self.assertLessEqual(avg_rtf, 1.20)
        self.assertGreaterEqual(fallback_reliability, 0.95)
        self.assertGreater(p95_latency, 0.0)
        self.assertTrue(bool(mos_checklist.get("passed")))
        self.assertTrue(bool((payload.get("sla") or {}).get("passed")))

        languages = dict(payload.get("languages") or {})
        self.assertIn("en", languages)
        self.assertIn("ar", languages)

        latest_daily = dict(((payload.get("history") or {}).get("latest_daily") or {}))
        self.assertGreaterEqual(float(latest_daily.get("sla_pass_rate") or 0.0), 0.99)


if __name__ == "__main__":
    unittest.main()
