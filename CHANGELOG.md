# Changelog

All notable changes to this project are documented in this file.

## 2026-04-04 - Phase 9 Demo Readiness

### Added
- User guide: docs/USER_GUIDE.md
- Admin guide: docs/ADMIN_GUIDE.md
- Demo script: docs/DEMO_SCRIPT.md
- Troubleshooting guide: docs/TROUBLESHOOTING.md
- Windows setup script: scripts/setup_windows.ps1
- Smoke helper script: scripts/run_phase8_gate.ps1
- Release notes: docs/RELEASE_NOTES.md

### Finalized
- Packaging and documentation assets for reproducible graduation demo.
- Demo-ready bilingual (EN/AR) scenario flow documentation.
- Operational runbooks for setup, diagnostics, and validation.

## 2026-04-04 - Phase 8 QA and Hardening

### Added
- Adversarial safety suite: tests/adversarial_safety_phase8.py
- End-to-end regression suite: tests/e2e_regression_phase8.py
- Performance gate suite: tests/performance_gates_phase8.py
- Aggregated regression gate: tests/phase8_regression.py

### Changed
- QA sign-off approved: docs/PHASE8_QA_SIGNOFF.md
- Dedicated CI phase8-regression job in .github/workflows/ci.yml

### Verified
- Full phase8 regression pass with SLA gate coverage.
