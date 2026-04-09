import csv
import tempfile
import unittest
from pathlib import Path

from core.tts_mos import aggregate_mos_scores, generate_mos_template


class TtsMosWorkflowTests(unittest.TestCase):
    def test_generate_mos_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "tts_mos_template.csv"
            result = generate_mos_template(output_path=output_path, backend="edge_tts")

            self.assertTrue(output_path.exists())
            self.assertGreaterEqual(int(result.get("scenario_count") or 0), 10)
            self.assertEqual(str(result.get("backend") or ""), "edge_tts")

    def test_aggregate_mos_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "ratings.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "scenario_id",
                        "language",
                        "backend",
                        "text",
                        "audio_file",
                        "rater_id",
                        "naturalness",
                        "clarity",
                        "pronunciation",
                        "overall",
                        "notes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scenario_id": "en_status_overview",
                        "language": "en",
                        "backend": "edge_tts",
                        "rater_id": "rater_a",
                        "naturalness": "4",
                        "clarity": "5",
                        "pronunciation": "4",
                        "overall": "",
                    }
                )
                writer.writerow(
                    {
                        "scenario_id": "en_status_overview",
                        "language": "en",
                        "backend": "edge_tts",
                        "rater_id": "rater_b",
                        "naturalness": "5",
                        "clarity": "4",
                        "pronunciation": "5",
                        "overall": "",
                    }
                )
                writer.writerow(
                    {
                        "scenario_id": "ar_status_overview",
                        "language": "ar",
                        "backend": "kokoro",
                        "rater_id": "rater_a",
                        "naturalness": "4",
                        "clarity": "4",
                        "pronunciation": "4",
                        "overall": "",
                    }
                )

            payload = aggregate_mos_scores(csv_path=csv_path)
            overall = dict(payload.get("overall") or {})
            by_backend = dict(payload.get("by_backend") or {})

            self.assertEqual(int(payload.get("rating_count") or 0), 3)
            self.assertEqual(int(payload.get("rater_count") or 0), 2)
            self.assertGreater(float(overall.get("mos") or 0.0), 4.0)
            self.assertIn("edge_tts", by_backend)
            self.assertIn("kokoro", by_backend)


if __name__ == "__main__":
    unittest.main()
