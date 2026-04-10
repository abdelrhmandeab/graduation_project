import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.stt_egyptian_benchmark import run_stt_egyptian_benchmark


class SttEgyptianBenchmarkRuntimeABTests(unittest.TestCase):
    def test_runtime_ab_runs_with_audio_cases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            corpus_path = root / "egyptian_runtime_pack.json"
            audio_path = root / "case1.wav"
            audio_path.write_bytes(b"wav")

            corpus = {
                "name": "runtime_pack",
                "version": "test",
                "baseline_setup": "fw_tiny_cpu",
                "latency_budget_ms_low_mid_cpu": 800,
                "setups": [
                    {"id": "fw_tiny_cpu", "label": "fw", "backend": "faster_whisper", "model": "tiny"},
                ],
                "scenarios": [
                    {
                        "name": "case1",
                        "language": "ar",
                        "domain": "daily_speech",
                        "expected_text": "عايز اعرف اخبار البورصة",
                        "audio_file": "case1.wav",
                        "setup_predictions": {
                            "fw_tiny_cpu": {
                                "transcript": "عايز اعرف اخبار البورسة",
                                "latency_ms": 400,
                            }
                        },
                    }
                ],
            }
            corpus_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")

            def fake_transcribe(audio_file, *, backend, on_partial=None, language_hint=None):
                return {
                    "text": "عايز اعرف اخبار البورصة",
                    "language": "ar",
                    "backend": "faster_whisper",
                }

            with patch(
                "core.stt_egyptian_benchmark.stt_runtime.transcribe_backend_direct_with_meta",
                side_effect=fake_transcribe,
            ):
                payload = run_stt_egyptian_benchmark(
                    corpus_path=str(corpus_path),
                    include_runtime_ab=True,
                    runtime_backends="faster_whisper",
                )

        runtime_ab = dict(payload.get("runtime_ab") or {})
        self.assertTrue(runtime_ab.get("enabled"))
        self.assertTrue(runtime_ab.get("executed"))
        self.assertEqual(int(runtime_ab.get("audio_scenario_count") or 0), 1)
        self.assertEqual(len(list(runtime_ab.get("setups") or [])), 1)
        self.assertEqual(runtime_ab.get("recommendation", {}).get("setup_id"), "runtime_faster_whisper")

    def test_runtime_ab_skips_when_no_audio_cases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            corpus_path = root / "egyptian_runtime_pack.json"
            corpus = {
                "name": "runtime_pack",
                "version": "test",
                "baseline_setup": "fw_tiny_cpu",
                "latency_budget_ms_low_mid_cpu": 800,
                "setups": [
                    {"id": "fw_tiny_cpu", "label": "fw", "backend": "faster_whisper", "model": "tiny"},
                ],
                "scenarios": [
                    {
                        "name": "case1",
                        "language": "ar",
                        "domain": "daily_speech",
                        "expected_text": "عايز اعرف اخبار البورصة",
                        "setup_predictions": {
                            "fw_tiny_cpu": {
                                "transcript": "عايز اعرف اخبار البورسة",
                                "latency_ms": 400,
                            }
                        },
                    }
                ],
            }
            corpus_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")

            payload = run_stt_egyptian_benchmark(
                corpus_path=str(corpus_path),
                include_runtime_ab=True,
                runtime_backends="faster_whisper",
            )

        runtime_ab = dict(payload.get("runtime_ab") or {})
        self.assertTrue(runtime_ab.get("enabled"))
        self.assertFalse(runtime_ab.get("executed"))
        self.assertEqual(runtime_ab.get("reason"), "no_audio_scenarios_available")

    def test_runtime_ab_reports_transcription_errors(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            corpus_path = root / "egyptian_runtime_pack.json"
            audio_path = root / "case1.wav"
            audio_path.write_bytes(b"wav")

            corpus = {
                "name": "runtime_pack",
                "version": "test",
                "baseline_setup": "fw_tiny_cpu",
                "latency_budget_ms_low_mid_cpu": 800,
                "setups": [
                    {"id": "fw_tiny_cpu", "label": "fw", "backend": "faster_whisper", "model": "tiny"},
                ],
                "scenarios": [
                    {
                        "name": "case1",
                        "language": "ar",
                        "domain": "daily_speech",
                        "expected_text": "عايز اعرف اخبار البورصة",
                        "audio_file": "case1.wav",
                        "setup_predictions": {
                            "fw_tiny_cpu": {
                                "transcript": "عايز اعرف اخبار البورصة",
                                "latency_ms": 400,
                            }
                        },
                    }
                ],
            }
            corpus_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")

            def fake_transcribe(audio_file, *, backend, on_partial=None, language_hint=None):
                raise RuntimeError("runtime backend unavailable")

            with patch(
                "core.stt_egyptian_benchmark.stt_runtime.transcribe_backend_direct_with_meta",
                side_effect=fake_transcribe,
            ):
                payload = run_stt_egyptian_benchmark(
                    corpus_path=str(corpus_path),
                    include_runtime_ab=True,
                    runtime_backends="faster_whisper",
                )

        runtime_ab = dict(payload.get("runtime_ab") or {})
        setups = list(runtime_ab.get("setups") or [])
        self.assertEqual(len(setups), 1)
        self.assertEqual(int(setups[0].get("error_count") or 0), 1)


if __name__ == "__main__":
    unittest.main()
