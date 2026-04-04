# Release Notes

## Release: Graduation Demo Candidate
Date: 2026-04-04

## Summary

This release finalizes packaging, documentation, and demo-readiness artifacts while preserving the validated Phase 8 quality gate.

## Highlights

- Bilingual operation (English and Arabic) with strict language discipline.
- Safety-first high-risk action flow with confirmation and second factor.
- Adversarial and regression test gates integrated into CI.
- Reproducible setup and troubleshooting documentation for presentation environments.

## Included Documentation

- docs/USER_GUIDE.md
- docs/ADMIN_GUIDE.md
- docs/DEMO_SCRIPT.md
- docs/TROUBLESHOOTING.md
- docs/PHASE8_QA_SIGNOFF.md

## Included Scripts

- scripts/setup_windows.ps1
- scripts/run_phase8_gate.ps1

## Validation Gate

Primary validation command:

```powershell
python tests/phase8_regression.py
```

Expected outcome:
- all suites pass
- timing summary emitted

## Known Constraints

- Supported runtime target is Windows.
- Supported interaction languages are English and Arabic.
- High-risk operations remain policy/confirmation constrained by design.
