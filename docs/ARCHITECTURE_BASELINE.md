# Architecture Baseline

This document freezes the current repository shape as the Phase 0 baseline for the Jarvis graduation project.

## Snapshot Date

- Captured: 2026-04-04
- Scope: current working tree and baseline runtime behavior
- Target: Windows desktop voice assistant with Arabic and English only

## Module Overview

| Module | Responsibility |
|--------|---------------|
| `audio/` | Mic capture, VAD, wake word detection, STT, TTS |
| `core/` | Orchestrator, command parsing/routing, language gate, confidence, memory, metrics, persona, demo mode |
| `llm/` | Ollama client, prompt construction |
| `os_control/` | File ops, app ops, system ops, policy, confirmations, second factor, audit |
| `tests/` | Smoke, safety, fuzz, latency, regression suites |

## Runtime Data Flow

```
Wake Word / Text Input
  → Audio capture / STT
  → Language gate (Arabic / English only)
  → Command parser
  → Intent confidence + clarification check
  → Router
    → deterministic action handlers
    → confirmation / second-factor flow for risky actions
    → LLM fallback when no deterministic intent matches
  → Metrics + audit logging
  → TTS / console response
```

## Current Runtime Characteristics

- `core/orchestrator.py` provides the live voice loop and text fallback path.
- `core/language_gate.py` rejects unsupported scripts and normalizes Arabic/English input.
- `core/intent_confidence.py` assigns confidence scores and triggers clarification for ambiguous commands.
- `core/session_memory.py` persists preferred language and pending clarification state.
- `os_control/confirmation.py` and the action adapters implement token-based confirmation and optional second factor.
- `core/metrics.py` collects command and stage timing data for reports.

## Key Design Decisions

- Local-first execution remains the default: Whisper, OpenWakeWord, Ollama, and Windows adapters are all used without a required cloud dependency.
- Safety is layered: policy gate, confirmation token, second factor, and audit trail.
- The parser/router remain deterministic for actions; the LLM is a fallback for non-deterministic or non-command text.
- Unsupported languages never proceed to action execution.

## Current Baseline Status

- Phase 0 documentation exists after this update.
- Phase 1 through Phase 4 functionality is implemented and currently regression-tested.
- Phase 5 through Phase 9 remain partially implemented or documentation-complete only.

## Evidence

- Phase 4 regression suite passed on 2026-04-04.
- Current benchmark and resilience artifacts show full scenario success at baseline.
