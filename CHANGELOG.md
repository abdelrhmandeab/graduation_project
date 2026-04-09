# Changelog

All notable changes to this project are documented in this file.

## 2026-04-09 - Release Hardening and Speech Quality Update

### Added
- Benchmark freshness policy gate and benchmark pack validation coverage.
- TTS MOS workflow artifacts and supporting benchmark scripts.

### Changed
- Mixed-script Arabic/English TTS routing now prefers Arabic voice when Arabic content is dominant.
- Arabic Edge-TTS default voice updated to female Egyptian profile (`ar-EG-SalmaNeural`) with deterministic fallbacks.
- Realtime startup diagnostics and benchmark artifacts refreshed for release readiness.

### Fixed
- Removed high-confidence unused code findings from static dead-code scan (`audio/stt.py`, `core/wake_benchmark.py`).

## 2026-04-04 - Phase 9 Demo Readiness

### Added
- User guide: docs/USER_GUIDE.md
- Admin guide: docs/ADMIN_GUIDE.md
- Demo script: docs/DEMO_SCRIPT.md
- Troubleshooting guide: docs/TROUBLESHOOTING.md
- Windows setup script: scripts/setup_windows.ps1
- Release notes: docs/RELEASE_NOTES.md

### Finalized
- Packaging and documentation assets for reproducible graduation demo.
- Demo-ready bilingual (EN/AR) scenario flow documentation.
- Operational runbooks for setup, diagnostics, and validation.

## 2026-04-04 - Phase 8 QA and Hardening

### Added
- Adversarial safety suite (archived during repository cleanup)
- End-to-end regression suite (archived during repository cleanup)
- Performance gate suite (archived during repository cleanup)
- Aggregated regression gate (archived during repository cleanup)

### Changed
- QA sign-off approved: docs/PHASE8_QA_SIGNOFF.md
- CI now uses runtime quality-check workflow (dependency install + compileall syntax validation)

### Verified
- Historical Phase 8 regression sign-off preserved in archived QA report.
