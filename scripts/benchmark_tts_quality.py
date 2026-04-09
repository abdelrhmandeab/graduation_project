import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.benchmark import run_tts_quality_benchmark
from core.config import TTS_BENCHMARK_CORPUS_FILE, TTS_BENCHMARK_OUTPUT_FILE


def main():
    parser = argparse.ArgumentParser(description="Run TTS quality benchmark (latency + objective quality proxies).")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / TTS_BENCHMARK_OUTPUT_FILE),
        help="Output report JSON path.",
    )
    parser.add_argument(
        "--corpus",
        default=str(PROJECT_ROOT / TTS_BENCHMARK_CORPUS_FILE),
        help="TTS corpus JSON path.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "mock", "real"),
        default="auto",
        help="Execution mode: auto prefers real synthesis and falls back to mock metrics.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "huggingface", "edge_tts", "kokoro"),
        default="auto",
        help="Preferred backend for objective synthesis in real/auto modes.",
    )
    args = parser.parse_args()

    payload = run_tts_quality_benchmark(
        corpus_path=args.corpus,
        mode=args.mode,
        backend=args.backend,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    corpus = dict(payload.get("corpus") or {})
    print("TTS Quality Benchmark")
    print("---------------------")
    print(f"corpus: {corpus.get('name', 'unknown')} ({corpus.get('version', 'unknown')})")
    print(f"mode: {corpus.get('mode', 'auto')}")
    print(f"backend: {corpus.get('backend', 'auto')}")
    print(f"scenarios: {payload.get('scenario_count', 0)}")
    print(f"real_scenarios: {payload.get('real_scenario_count', 0)}")
    print(f"mock_scenarios: {payload.get('mock_scenario_count', 0)}")
    print(f"fallback_attempted: {payload.get('fallback_attempted_count', 0)}")
    print(f"fallback_success: {payload.get('fallback_success_count', 0)}")
    print(f"fallback_reliability: {float(payload.get('fallback_reliability') or 0.0):.2%}")
    print(f"success_rate: {float(payload.get('success_rate') or 0.0):.2%}")
    print(f"avg_quality_score: {float(payload.get('avg_quality_score') or 0.0):.4f}")
    print(f"avg_rtf: {float(payload.get('avg_rtf') or 0.0):.4f}")
    print(f"p95_latency_ms: {float(payload.get('p95_latency_ms') or 0.0):.1f}")
    print(f"mos_checklist_passed: {bool((payload.get('mos_checklist') or {}).get('passed'))}")
    print(f"sla_passed: {bool((payload.get('sla') or {}).get('passed'))}")
    print(f"history_dropped_incompatible_runs: {int(((payload.get('history') or {}).get('dropped_incompatible_runs') or 0))}")
    print(f"report_file: {output_path}")


if __name__ == "__main__":
    main()
