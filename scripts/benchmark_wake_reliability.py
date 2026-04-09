import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.benchmark import run_wake_reliability_benchmark
from core.config import WAKE_BENCHMARK_OUTPUT_FILE


def main():
    parser = argparse.ArgumentParser(description="Run wake reliability integration benchmark.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / WAKE_BENCHMARK_OUTPUT_FILE),
        help="Output report JSON path.",
    )
    parser.add_argument(
        "--scenarios-per-language",
        type=int,
        default=0,
        help="Target deterministic scenarios per language (10-40).",
    )
    args = parser.parse_args()

    payload = run_wake_reliability_benchmark(
        scenarios_per_language=(args.scenarios_per_language or None),
    )
    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Wake Reliability Benchmark")
    print("--------------------------")
    print(f"pack: {((payload.get('scenario_pack') or {}).get('name') or 'unknown')}")
    print(f"target_per_language: {int((payload.get('scenario_pack') or {}).get('target_per_language') or 0)}")
    print(f"scenarios: {payload.get('scenario_count', 0)}")
    print(f"english_scenarios: {payload.get('english_scenario_count', 0)}")
    print(f"arabic_scenarios: {payload.get('arabic_scenario_count', 0)}")
    print(f"scenario_success_rate: {float(payload.get('success_rate') or 0.0):.2%}")
    print(f"detection_rate: {float(payload.get('detection_rate') or 0.0):.2%}")
    print(f"false_positive_rate: {float(payload.get('false_positive_rate') or 0.0):.2%}")
    print(f"p95_latency_ms: {float(payload.get('p95_latency_ms') or 0.0):.1f}")
    print(f"sla_passed: {bool((payload.get('sla') or {}).get('passed'))}")
    print(f"history_dropped_incompatible_runs: {int(((payload.get('history') or {}).get('dropped_incompatible_runs') or 0))}")
    print(f"report_file: {output_path}")


if __name__ == "__main__":
    main()
