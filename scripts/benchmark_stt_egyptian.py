import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import STT_EGYPTIAN_BENCHMARK_CORPUS_FILE, STT_EGYPTIAN_BENCHMARK_OUTPUT_FILE
from core.stt_egyptian_benchmark import run_stt_egyptian_benchmark


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Egyptian Arabic STT setup quality versus latency for low/mid CPU devices."
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / STT_EGYPTIAN_BENCHMARK_OUTPUT_FILE),
        help="Output report JSON path.",
    )
    parser.add_argument(
        "--corpus",
        default=str(PROJECT_ROOT / STT_EGYPTIAN_BENCHMARK_CORPUS_FILE),
        help="Egyptian dialect benchmark corpus path.",
    )
    parser.add_argument(
        "--runtime-ab",
        action="store_true",
        help="Run direct runtime A/B on scenarios that provide audio_file paths.",
    )
    parser.add_argument(
        "--runtime-backends",
        default="faster_whisper",
        help="Comma-separated backends for runtime A/B (faster_whisper).",
    )
    parser.add_argument(
        "--runtime-max-cases",
        type=int,
        default=0,
        help="Limit runtime A/B to the first N audio scenarios (0 means all available).",
    )
    args = parser.parse_args()

    payload = run_stt_egyptian_benchmark(
        corpus_path=args.corpus,
        include_runtime_ab=bool(args.runtime_ab),
        runtime_backends=args.runtime_backends,
        runtime_max_cases=int(args.runtime_max_cases),
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    recommendation = dict(payload.get("recommendation") or {})
    done_gate = dict(payload.get("done_gate") or {})
    baseline = dict(payload.get("baseline_vs_recommended") or {})

    print("Egyptian Arabic STT Benchmark")
    print("-----------------------------")
    print(f"corpus: {((payload.get('corpus') or {}).get('name') or 'unknown')} ({((payload.get('corpus') or {}).get('version') or 'unknown')})")
    print(f"scenario_count: {int(((payload.get('corpus') or {}).get('scenario_count') or 0))}")
    print(f"latency_budget_ms_low_mid_cpu: {float(payload.get('latency_budget_ms_low_mid_cpu') or 0.0):.1f}")
    print(f"baseline_setup: {baseline.get('baseline_setup')}")
    print(f"recommended_setup: {recommendation.get('setup_id')}")
    print(f"recommended_avg_wer: {float(recommendation.get('avg_wer_normalized') or 0.0):.4f}")
    print(f"recommended_p95_latency_ms: {float(recommendation.get('p95_latency_ms') or 0.0):.1f}")
    print(f"wer_gain_abs_vs_baseline: {float(baseline.get('wer_gain_abs') or 0.0):.4f}")
    print(f"wer_gain_rel_vs_baseline: {float(baseline.get('wer_gain_rel') or 0.0):.2%}")
    print(f"done_gate_passed: {bool(done_gate.get('passed'))}")

    runtime_ab = dict(payload.get("runtime_ab") or {})
    if runtime_ab:
        print("runtime_ab_enabled: True")
        print(f"runtime_ab_executed: {bool(runtime_ab.get('executed'))}")
        print(f"runtime_audio_scenario_count: {int(runtime_ab.get('audio_scenario_count') or 0)}")
        print(f"runtime_requested_backends: {','.join(list(runtime_ab.get('requested_backends') or []))}")
        if not bool(runtime_ab.get("executed")):
            print(f"runtime_ab_reason: {runtime_ab.get('reason')}")
        else:
            runtime_reco = dict(runtime_ab.get("recommendation") or {})
            print(f"runtime_recommended_setup: {runtime_reco.get('setup_id')}")
            print(f"runtime_recommended_avg_wer: {float(runtime_reco.get('avg_wer_normalized') or 0.0):.4f}")
            print(f"runtime_recommended_p95_latency_ms: {float(runtime_reco.get('p95_latency_ms') or 0.0):.1f}")

    print(f"report_file: {output_path}")


if __name__ == "__main__":
    main()
