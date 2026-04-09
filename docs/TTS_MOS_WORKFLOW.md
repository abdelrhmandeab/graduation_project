# TTS MOS Workflow

This workflow provides the human-evaluation complement to the objective TTS benchmark.

## Artifacts

- Template CSV: `benchmarks/tts_mos_template.csv`
- Aggregation script: `scripts/aggregate_tts_mos.py`
- Output JSON: `jarvis_tts_mos.json`

## 1) Generate Rating Template

```powershell
python scripts\aggregate_tts_mos.py --generate-template --backend edge_tts
```

The template includes one row per scenario from `benchmarks/tts_corpus.json`.

## 2) Collect Ratings

For each scenario/audio sample, ask each rater to score:

- `naturalness` (1-5)
- `clarity` (1-5)
- `pronunciation` (1-5)
- optional `overall` (1-5)

If `overall` is blank, aggregation uses the average of the three aspect scores.

## 3) Aggregate MOS

```powershell
python scripts\aggregate_tts_mos.py --csv path\to\ratings.csv --output jarvis_tts_mos.json
```

Output includes:

- overall MOS
- MOS by scenario
- MOS by backend
- MOS by language
- rating/rater counts for auditability

## Notes

- Keep at least 3 raters per scenario for stable MOS trend tracking.
- Keep corpus text fixed between runs when comparing backend versions.
- Use `benchmark tts` objective report and MOS report together before backend promotion.
