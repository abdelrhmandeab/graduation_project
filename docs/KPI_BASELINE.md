# KPI Baseline

This document captures the current measurable baseline for the Jarvis project at the start of Phase 0.

## Snapshot Date

- Captured: 2026-04-04
- Source artifacts: `jarvis_benchmark.json`, `jarvis_resilience.json`

## Baseline Metrics

| KPI | Baseline Value | Notes |
|---|---:|---|
| Benchmark scenario success rate | 100% | 7/7 scenarios passed |
| Benchmark p50 latency | 1.937 ms | Current benchmark artifact |
| Benchmark p95 latency | 59.719 ms | Current benchmark artifact |
| Resilience scenario success rate | 100% | 5/5 scenarios passed |
| Resilience p50 latency | 10.587 ms | Current resilience artifact |
| Resilience p95 latency | 103.136 ms | Current resilience artifact |
| Phase 4 regression pass rate | 100% | Historical Phase 4 regression sign-off from 2026-04-04 |

## Baseline Observations

- The safety path is working: invalid confirmation handling, policy blocking, and rollback recovery all passed in the resilience run.
- The language gate is active and blocks unsupported language input before routing.
- Clarification flows are active for ambiguous commands.
- The benchmark output includes a successful audit verification check (`audit_verify` passed).

## Initial Targets

These are starting targets for the next phases, not current results.

| KPI | Target |
|---|---:|
| Command acknowledgement latency p95 | <= 1500 ms |
| Common command completion latency | <= 3000 ms |
| Phase 4 regression success rate | 100% |
| High-risk safety compliance | 100% |
| Unsupported language leakage | 0% |

## How This Baseline Will Be Used

1. Compare future benchmark runs against the values above.
2. Track weekly drift in latency and success rate.
3. Update this file only when a new baseline snapshot is intentionally frozen.