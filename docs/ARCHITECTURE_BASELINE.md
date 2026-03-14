# Architecture Baseline

> Auto-generated stub. To be filled during Phase 0.

## Module Overview

| Module | Responsibility |
|--------|---------------|
| `audio/` | Mic capture, VAD, wake word detection, STT, TTS |
| `core/` | Orchestrator, command parser/router, config, memory, metrics, personas |
| `llm/` | Ollama client, prompt construction |
| `os_control/` | File ops, app ops, system ops, policy, confirmations, audit |
| `tests/` | Smoke, safety, fuzz, latency, phase-specific test suites |

## Data Flow

```
Wake Word → VAD → Record → STT (Whisper) → Command Parser
  → [Known Intent] → Command Router → Action Handler → Response
  → [Unknown Intent] → LLM (Ollama) → Response
Response → TTS → Speaker
```

## Key Design Decisions

- **Local-first**: All processing on-device (Ollama, Whisper, OpenWakeWord).
- **Safety layered**: Policy engine → confirmation tokens → second-factor → audit trail.
- **Graceful degradation**: Every subsystem has a fallback path.
