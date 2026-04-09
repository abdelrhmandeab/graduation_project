import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.benchmark import run_stt_reliability_benchmark
from core.config import STT_BENCHMARK_CORPUS_FILE, STT_BENCHMARK_OUTPUT_FILE


def main():
    parser = argparse.ArgumentParser(description="Run STT reliability benchmark (WER + latency).")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / STT_BENCHMARK_OUTPUT_FILE),
        help="Output report JSON path.",
    )
    parser.add_argument(
        "--corpus",
        default=str(PROJECT_ROOT / STT_BENCHMARK_CORPUS_FILE),
        help="STT corpus JSON path.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "mock", "real"),
        default="auto",
        help="Execution mode: auto uses real audio when available then mock fallback.",
    )
    args = parser.parse_args()

    payload = run_stt_reliability_benchmark(corpus_path=args.corpus, mode=args.mode)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    corpus = dict(payload.get("corpus") or {})
    print("STT Reliability Benchmark")
    print("-------------------------")
    print(f"corpus: {corpus.get('name', 'unknown')} ({corpus.get('version', 'unknown')})")
    print(f"mode: {corpus.get('mode', 'auto')}")
    print(f"scenarios: {payload.get('scenario_count', 0)}")
    print(f"evaluated: {payload.get('evaluated_count', 0)}")
    print(f"real_audio_scenarios: {payload.get('real_audio_scenario_count', 0)}")
    print(f"mock_scenarios: {payload.get('mock_scenario_count', 0)}")
    print(f"success_rate: {float(payload.get('success_rate') or 0.0):.2%}")
    print(f"avg_wer: {float(payload.get('avg_wer') or 0.0):.4f}")
    print(f"avg_cer: {float(payload.get('avg_cer') or 0.0):.4f}")
    print(f"p95_wer: {float(payload.get('p95_wer') or 0.0):.4f}")
    print(f"p95_cer: {float(payload.get('p95_cer') or 0.0):.4f}")
    print(f"p95_latency_ms: {float(payload.get('p95_latency_ms') or 0.0):.1f}")
    print(f"sla_passed: {bool((payload.get('sla') or {}).get('passed'))}")
    print(f"history_dropped_incompatible_runs: {int(((payload.get('history') or {}).get('dropped_incompatible_runs') or 0))}")
    print(f"report_file: {output_path}")


if __name__ == "__main__":
    main()
