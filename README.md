# graduation_project

## Environment Setup

Install core runtime dependencies:

```powershell
python -m pip install numpy scipy sounddevice faster-whisper openwakeword pyttsx3 psutil
```

Optional (only if you want Silero VAD backend instead of fallback):

```powershell
python -m pip install torch silero-vad
```

If audio/wake-word dependencies are missing, `core/orchestrator.py` now falls back to text mode automatically.

## Realtime Run

0. Run a one-shot health check:

```powershell
python core\doctor.py
```

1. Start Ollama service in a separate terminal:

```powershell
ollama serve
```

2. Ensure your model exists (first time only):

```powershell
ollama pull llama3
```

3. Start Jarvis realtime pipeline:

```powershell
python core\orchestrator.py
```

Expected startup for realtime mode:
- `Jarvis started`
- `[WakeWord] Waiting for wake word...`

If wake-word models are missing, they are now downloaded automatically on first run.

## Phase 4 Features

- Voice synthesis abstraction with optional clone providers (`xtts`, `voicecraft`) and safe fallback.
- Interruptible speech output (`stop speaking`).
- Multi-persona profiles (`assistant`, `formal`, `casual`).
- Offline knowledge base with vector retrieval (FAISS when available, local fallback otherwise).
- Hybrid retrieval reranking + prompt-injection-safe context sanitization.
- Incremental KB sync and deduplicated ingestion.
- Persona-to-voice mapping.
- Session memory + observability dashboard + quick benchmark runner.
- Real-time core capture with VAD endpointing and async utterance processing.
- Benchmark and resilience SLA evaluation blocks in JSON reports.

## Phase 4 Commands

- Persona:
  - `persona list`
  - `persona status`
  - `persona set formal`
  - `persona set casual`
  - `assistant mode`
  - `persona voice status`
  - `persona voice clone <profile> on|off`
  - `persona voice provider <profile> xtts|voicecraft`
  - `persona voice reference <profile> <path_to_wav>`
- Voice / Speech:
  - `voice status`
  - `voice clone on`
  - `voice clone off`
  - `voice clone provider xtts`
  - `voice clone provider voicecraft`
  - `voice clone reference <path_to_wav>`
  - `speech on`
  - `speech off`
  - `stop speaking`
- Knowledge base:
  - `kb status`
  - `kb add <path_to_file>`
  - `kb index <path_to_directory>`
  - `kb sync <path_to_directory>`
  - `kb search <query>`
  - `kb quality`
  - `kb retrieval on`
  - `kb retrieval off`
  - `kb clear`
- Memory / observability:
  - `memory status`
  - `memory show`
  - `memory on`
  - `memory off`
  - `memory clear`
  - `observability`
  - `benchmark run`
  - `resilience demo`
- Audit:
  - `verify audit log`
  - `audit reseal`

## Test Commands

Run sequentially:

```powershell
python -m compileall -q .
python tests\phase3_smoke.py
python tests\safety_suite.py
python tests\parser_fuzz.py
python tests\phase3_advanced.py
python tests\phase4_smoke.py
python tests\phase4_exceptional.py
python tests\latency_sla.py
```

## Milestone Freeze

- Baseline snapshot file: `milestones/phase4_stable_snapshot_2026-03-09.json`
