import unittest

from core.benchmark import run_stt_reliability_benchmark


class SttReliabilityBenchmarkTests(unittest.TestCase):
    def test_stt_reliability_benchmark_payload_and_sla(self):
        payload = run_stt_reliability_benchmark(mode="mock")

        self.assertGreaterEqual(int(payload.get("scenario_count") or 0), 10)
        self.assertGreaterEqual(int(payload.get("evaluated_count") or 0), 10)
        self.assertEqual(int(payload.get("real_audio_scenario_count") or 0), 0)
        self.assertGreaterEqual(int(payload.get("mock_scenario_count") or 0), 10)

        self.assertIn("results", payload)
        self.assertIn("sla", payload)
        self.assertIn("history", payload)
        self.assertTrue(str(payload.get("history_series") or "").startswith("stt:"))

        success_rate = float(payload.get("success_rate") or 0.0)
        avg_wer = float(payload.get("avg_wer") or 1.0)
        avg_cer = float(payload.get("avg_cer") or 1.0)
        p95_latency = float(payload.get("p95_latency_ms") or 0.0)

        self.assertGreaterEqual(success_rate, 0.90)
        self.assertLessEqual(avg_wer, 0.20)
        self.assertLessEqual(avg_cer, 0.30)
        self.assertGreater(p95_latency, 0.0)
        self.assertTrue(bool((payload.get("sla") or {}).get("passed")))

        languages = dict(payload.get("languages") or {})
        self.assertIn("en", languages)
        self.assertIn("ar", languages)
        self.assertIn("avg_cer", dict(languages.get("en") or {}))
        self.assertIn("avg_cer", dict(languages.get("ar") or {}))

        latest_daily = dict(((payload.get("history") or {}).get("latest_daily") or {}))
        self.assertGreaterEqual(float(latest_daily.get("sla_pass_rate") or 0.0), 0.99)


if __name__ == "__main__":
    unittest.main()
