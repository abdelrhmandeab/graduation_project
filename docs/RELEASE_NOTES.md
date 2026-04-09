# Release Notes

## Release: Hardening Update (Phase 5 Runtime + Quality Gates)
Date: 2026-04-09

## Summary

This update hardens runtime quality for mixed-language speech, refreshes benchmark and MOS artifacts, and verifies release readiness through compile checks, full test pass, benchmark freshness policy, and startup diagnostics.

## Highlights

- Mixed-script TTS reliability improvements for Arabic-dominant responses that include English fragments.
- Arabic Edge-TTS default profile switched to female Egyptian voice (`ar-EG-SalmaNeural`) with deterministic fallback order.
- TTS quality benchmark + MOS aggregation workflow added to release assets.
- Freshness policy gate enforced for checked-in benchmark artifacts in CI and local validation.

## Included Documentation

- docs/PHASE5_BEHAVIOR_CONTRACT.md
- docs/TTS_MOS_WORKFLOW.md
- docs/RELEASE_NOTES.md (updated)

## Included Scripts

- scripts/benchmark_phase1_intent.py
- scripts/benchmark_phase5_dialogue.py
- scripts/benchmark_tts_quality.py
- scripts/aggregate_tts_mos.py
- scripts/check_benchmark_freshness.py

## Validation Gate

Primary validation commands:

```powershell
python -m compileall -q .
python core/doctor.py
python -W error::ResourceWarning -m unittest discover -s tests -p "test_*.py"
python scripts/check_benchmark_freshness.py --max-age-hours 168
```

Expected outcome:
- syntax validation passes
- doctor report is generated
- test suite passes
- freshness policy returns PASS for all required benchmark families

## Previous Release

Release: Graduation Demo Candidate
Date: 2026-04-04

Key focus:
- Packaging, setup documentation, and demo-readiness assets.
- Archived Phase 8 QA evidence and baseline runtime quality checks.

## Known Constraints

- Supported runtime target is Windows.
- Supported interaction languages are English and Arabic.
- High-risk operations remain policy/confirmation constrained by design.
