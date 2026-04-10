import unittest

from core.benchmark import run_wake_reliability_benchmark


class WakeReliabilityBenchmarkTests(unittest.TestCase):
    def test_wake_reliability_benchmark_payload_and_sla(self):
        payload = run_wake_reliability_benchmark()

        self.assertGreaterEqual(int(payload.get("scenario_count") or 0), 20)
        self.assertGreaterEqual(int(payload.get("english_scenario_count") or 0), 10)
        self.assertGreaterEqual(int(payload.get("arabic_scenario_count") or 0), 10)
        self.assertIn("results", payload)
        self.assertIn("sla", payload)

        detection_rate_raw = payload.get("detection_rate")
        false_positive_rate_raw = payload.get("false_positive_rate")
        p95_latency_raw = payload.get("p95_latency_ms")

        detection_rate = float(detection_rate_raw if detection_rate_raw is not None else 0.0)
        false_positive_rate = float(false_positive_rate_raw if false_positive_rate_raw is not None else 1.0)
        p95_latency_ms = float(p95_latency_raw if p95_latency_raw is not None else 0.0)

        self.assertGreaterEqual(detection_rate, 0.95)
        self.assertLessEqual(false_positive_rate, 0.05)
        self.assertGreater(p95_latency_ms, 0.0)
        self.assertTrue(bool((payload.get("sla") or {}).get("passed")))

        names = {str(row.get("name") or "") for row in list(payload.get("results") or [])}
        self.assertTrue(any(name.startswith("english_detection_latency_") for name in names))
        self.assertTrue(any(name.startswith("arabic_detection_latency_") for name in names))

        latest_daily = dict(((payload.get("history") or {}).get("latest_daily") or {}))
        sla_pass_rate = float(latest_daily.get("sla_pass_rate") or 0.0)
        self.assertGreaterEqual(sla_pass_rate, 0.0)
        self.assertLessEqual(sla_pass_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
