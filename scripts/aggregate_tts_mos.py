import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import TTS_MOS_OUTPUT_FILE, TTS_MOS_TEMPLATE_FILE
from core.tts_mos import aggregate_mos_scores, generate_mos_template


def main():
    parser = argparse.ArgumentParser(description="Generate/aggregate TTS MOS workflow artifacts.")
    parser.add_argument(
        "--generate-template",
        action="store_true",
        help="Generate MOS rating template CSV from current TTS corpus.",
    )
    parser.add_argument(
        "--template",
        default=str(PROJECT_ROOT / TTS_MOS_TEMPLATE_FILE),
        help="Template CSV path for generation mode.",
    )
    parser.add_argument(
        "--backend",
        default="auto",
        help="Backend label written to generated template rows.",
    )
    parser.add_argument(
        "--csv",
        default="",
        help="Completed MOS ratings CSV path (required for aggregation mode).",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / TTS_MOS_OUTPUT_FILE),
        help="Aggregated MOS output JSON path.",
    )
    args = parser.parse_args()

    if args.generate_template:
        result = generate_mos_template(
            output_path=args.template,
            backend=args.backend,
        )
        print("TTS MOS template generated")
        print(f"template_file: {result['template_path']}")
        print(f"scenario_count: {result['scenario_count']}")
        print(f"backend: {result['backend']}")
        return

    csv_path = str(args.csv or "").strip()
    if not csv_path:
        raise SystemExit("Provide --csv <ratings_file.csv> or use --generate-template.")

    payload = aggregate_mos_scores(csv_path=csv_path)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("TTS MOS aggregation")
    print("-------------------")
    print(f"source_csv: {payload.get('source_csv')}")
    print(f"rating_count: {payload.get('rating_count')}")
    print(f"rater_count: {payload.get('rater_count')}")
    print(f"overall_mos: {float(((payload.get('overall') or {}).get('mos') or 0.0)):.3f}")
    print(f"output_file: {output_path}")


if __name__ == "__main__":
    main()
