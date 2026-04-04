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

Optional (if you want Hugging Face speech models for both STT and TTS):

```powershell
python -m pip install transformers huggingface-hub sentencepiece
```

Set backends in `.env`:

```env
JARVIS_STT_BACKEND=huggingface
JARVIS_STT_HF_MODEL=openai/whisper-small
JARVIS_STT_HF_MODE=manual
JARVIS_TTS_BACKEND=huggingface
JARVIS_TTS_HF_MODEL=facebook/mms-tts-eng
JARVIS_TTS_QUALITY_MODE=natural
```

This is optional after first setup: you can switch and persist HF speech profiles at runtime with voice commands (no `.env` edit required each time).

Notes:
- First run downloads model files from Hugging Face and may take time.
- For Arabic TTS, try `facebook/mms-tts-ara` for `JARVIS_TTS_HF_MODEL`.
- `JARVIS_TTS_QUALITY_MODE=natural` makes Jarvis prefer tuned system voices before HF-TTS when possible.

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

## Phase 0 Baseline

Current baseline artifacts:

- [Architecture baseline](docs/ARCHITECTURE_BASELINE.md)
- [KPI baseline](docs/KPI_BASELINE.md)
- [Phase 0 milestone snapshot](milestones/phase0_baseline_2026-04-04.json)

Phase 0 progress board:

- Freeze current repository baseline: done
- Document architecture baseline: done
- Capture KPI baseline: done
- Record milestone snapshot: done
- Prepare remaining phase deliverables: completed through Phase 9 docs and setup assets

## Current Features

- Voice synthesis abstraction with optional clone providers (`xtts`, `voicecraft`) and safe fallback.
- Interruptible speech output (`stop speaking`).
- Multi-persona profiles (`assistant`, `formal`, `casual`, `professional`, `friendly`, `brief`).
- Offline knowledge base with vector retrieval (FAISS when available, local fallback otherwise).
- Hybrid retrieval reranking + prompt-injection-safe context sanitization.
- Incremental KB sync and deduplicated ingestion.
- Persona-to-voice mapping.
- Session memory + observability dashboard + quick benchmark runner.
- Real-time core capture with VAD endpointing and async utterance processing.
- Benchmark and resilience SLA evaluation blocks in JSON reports.
- Per-language and per-intent/per-language metrics in observability output.
- Daily and weekly benchmark/resilience rollups in history artifacts.
- Startup + scheduled doctor diagnostics integrated in realtime orchestrator.
- Structured JSON route events in logs for incident analysis.

## Core Commands

- Persona:
  - `persona list`
  - `persona status`
  - `persona set <assistant|formal|casual|professional|friendly|brief>`
  - `assistant mode`
  - `persona voice status`
  - `persona voice clone <profile> on|off`
  - `persona voice provider <profile> xtts|voicecraft`
  - `persona voice reference <profile> <path_to_wav>`
- Voice / Speech:
  - `voice status`
  - `voice diagnostic`
  - `voice clone on`
  - `voice clone off`
  - `voice clone provider xtts`
  - `voice clone provider voicecraft`
  - `voice clone reference <path_to_wav>`
  - `voice quality natural`
  - `voice quality standard`
  - `voice quality status`
  - `audio ux profile balanced`
  - `audio ux profile responsive`
  - `audio ux profile robust`
  - `audio ux profiles`
  - `audio ux status`
  - `set mic threshold to 0.012`
  - `set wake threshold to 0.38`
  - `set wake gain to 1.6`
  - `set pause scale to 0.9`
  - `set rate offset to -8`
  - `stt profile quiet`
  - `stt profile noisy`
  - `stt profile status`
  - selected STT profile is persisted and restored on next startup
  - `hf profile arabic`
  - `hf profile english`
  - `hf profile status`
  - `set hf profile to ar|en`
  - selected HF profile is persisted, restored on next startup, and forces runtime STT/TTS backends to `huggingface`
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

## Validation Commands

Run static runtime validation:

```powershell
python -m compileall -q .
python core\doctor.py
```

CI automation:

- GitHub Actions workflow [Jarvis CI](.github/workflows/ci.yml) runs dependency install and syntax validation on every push/PR to `main` and manual dispatch.

## Milestone Freeze

- Baseline snapshot file: `milestones/phase4_stable_snapshot_2026-03-09.json`
