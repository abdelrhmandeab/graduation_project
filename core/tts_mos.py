import csv
import json
import time
from pathlib import Path

from core.config import TTS_BENCHMARK_CORPUS_FILE, TTS_MOS_TEMPLATE_FILE


def _clamp_score(value):
    try:
        score = float(value)
    except Exception:
        return None
    if score < 1.0:
        return 1.0
    if score > 5.0:
        return 5.0
    return score


def _read_tts_scenarios(corpus_path=None):
    path = Path(str(corpus_path or TTS_BENCHMARK_CORPUS_FILE)).resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    scenarios = list((payload or {}).get("scenarios") or [])
    rows = []
    for item in scenarios:
        rows.append(
            {
                "scenario_id": str(item.get("name") or "").strip(),
                "language": str(item.get("language") or "auto").strip().lower(),
                "text": " ".join(str(item.get("text") or "").split()).strip(),
            }
        )
    return rows


def generate_mos_template(*, output_path=None, corpus_path=None, backend="auto"):
    target = Path(str(output_path or TTS_MOS_TEMPLATE_FILE))
    target.parent.mkdir(parents=True, exist_ok=True)

    scenarios = _read_tts_scenarios(corpus_path=corpus_path)
    fieldnames = [
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
    ]

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in scenarios:
            writer.writerow(
                {
                    "scenario_id": row["scenario_id"],
                    "language": row["language"],
                    "backend": str(backend or "auto"),
                    "text": row["text"],
                    "audio_file": "",
                    "rater_id": "",
                    "naturalness": "",
                    "clarity": "",
                    "pronunciation": "",
                    "overall": "",
                    "notes": "",
                }
            )

    return {
        "template_path": str(target),
        "scenario_count": len(scenarios),
        "backend": str(backend or "auto"),
    }


def aggregate_mos_scores(*, csv_path):
    source = Path(str(csv_path)).resolve()
    if not source.exists():
        raise FileNotFoundError(f"MOS CSV not found: {source}")

    rows = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            scenario_id = str(raw.get("scenario_id") or "").strip()
            if not scenario_id:
                continue

            naturalness = _clamp_score(raw.get("naturalness"))
            clarity = _clamp_score(raw.get("clarity"))
            pronunciation = _clamp_score(raw.get("pronunciation"))
            overall = _clamp_score(raw.get("overall"))

            aspect_scores = [score for score in [naturalness, clarity, pronunciation] if score is not None]
            if overall is None and aspect_scores:
                overall = sum(aspect_scores) / float(len(aspect_scores))

            if overall is None:
                continue

            rows.append(
                {
                    "scenario_id": scenario_id,
                    "language": str(raw.get("language") or "auto").strip().lower(),
                    "backend": str(raw.get("backend") or "auto").strip().lower(),
                    "rater_id": str(raw.get("rater_id") or "anonymous").strip() or "anonymous",
                    "overall": float(overall),
                    "naturalness": naturalness,
                    "clarity": clarity,
                    "pronunciation": pronunciation,
                }
            )

    by_scenario = {}
    by_backend = {}
    by_language = {}
    for row in rows:
        by_scenario.setdefault(row["scenario_id"], []).append(row)
        by_backend.setdefault(row["backend"], []).append(row)
        by_language.setdefault(row["language"], []).append(row)

    def _summary(group):
        scores = [float(item.get("overall")) for item in group]
        return {
            "count": len(scores),
            "mos": (sum(scores) / float(len(scores))) if scores else 0.0,
            "min": min(scores) if scores else 0.0,
            "max": max(scores) if scores else 0.0,
        }

    scenario_summary = {
        key: _summary(group)
        for key, group in sorted(by_scenario.items())
    }
    backend_summary = {
        key: _summary(group)
        for key, group in sorted(by_backend.items())
    }
    language_summary = {
        key: _summary(group)
        for key, group in sorted(by_language.items())
    }

    overall = _summary(rows)
    raters = sorted({str(row.get("rater_id") or "anonymous") for row in rows})

    return {
        "timestamp": time.time(),
        "source_csv": str(source),
        "rating_count": len(rows),
        "rater_count": len(raters),
        "raters": raters,
        "overall": overall,
        "by_scenario": scenario_summary,
        "by_backend": backend_summary,
        "by_language": language_summary,
    }
