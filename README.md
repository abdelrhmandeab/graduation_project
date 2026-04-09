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

Optional (if you want cloud/local neural TTS backends beyond system voices):

```powershell
python -m pip install edge-tts kokoro
```

Set backends in `.env`:

```env
JARVIS_STT_BACKEND=huggingface
JARVIS_STT_HF_MODEL=openai/whisper-small
JARVIS_STT_HF_MODE=manual
JARVIS_TTS_BACKEND=huggingface
JARVIS_TTS_HF_MODEL=facebook/mms-tts-eng
JARVIS_TTS_QUALITY_MODE=natural
JARVIS_TTS_EDGE_VOICE=en-US-AriaNeural
JARVIS_TTS_EDGE_RATE=+0%
JARVIS_TTS_EDGE_ARABIC_VOICE=ar-EG-SalmaNeural
JARVIS_TTS_EDGE_ARABIC_VOICE_FALLBACKS=ar-EG-ShakirNeural,ar-SA-HamedNeural
JARVIS_TTS_EDGE_ARABIC_RATE=-4%
JARVIS_TTS_EDGE_ARABIC_PITCH=-8Hz
JARVIS_TTS_EDGE_ARABIC_VOLUME=+4%
JARVIS_TTS_ARABIC_SPOKEN_DIALECT=egyptian
JARVIS_TTS_EGYPTIAN_COLLOQUIAL_REWRITE=true
JARVIS_TTS_KOKORO_VOICE=af_heart
```

This is optional after first setup: you can switch and persist HF speech profiles at runtime with voice commands (no `.env` edit required each time).

Notes:
- First run downloads model files from Hugging Face and may take time.
- For Arabic TTS, try `facebook/mms-tts-ara` for `JARVIS_TTS_HF_MODEL`.
- You can switch to `JARVIS_TTS_BACKEND=edge_tts` or `JARVIS_TTS_BACKEND=kokoro` when those dependencies are installed.
- `JARVIS_TTS_QUALITY_MODE=natural` makes Jarvis prefer tuned system voices before HF-TTS when possible.
- English Edge-TTS remains on `JARVIS_TTS_EDGE_VOICE` (default `en-US-AriaNeural`) with `JARVIS_TTS_EDGE_RATE`.
- Arabic Edge-TTS uses conversational Egyptian profile by default (`ar-EG-SalmaNeural`) and falls back through `JARVIS_TTS_EDGE_ARABIC_VOICE_FALLBACKS`.
- Use `JARVIS_TTS_ARABIC_SPOKEN_DIALECT=egyptian` and `JARVIS_TTS_EGYPTIAN_COLLOQUIAL_REWRITE=true` to make spoken Arabic less formal.
- Fine tune character with `JARVIS_TTS_EDGE_ARABIC_PITCH` and `JARVIS_TTS_EDGE_ARABIC_VOLUME`.
- For low/mid CPU devices, `JARVIS_WHISPER_MODEL=base` is the recommended realtime default for Egyptian Arabic + English balance.
- Egyptian Arabic post-normalization is always enabled for Arabic transcripts.
- Arabic STT runs in Egyptian-dialect mode while English STT keeps the same auto + English-retry behavior.

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

- Voice synthesis abstraction with runtime backends (`pyttsx3`, `huggingface`, `edge_tts`, `kokoro`) plus clone provider (`voicecraft`) and safe fallback.
- Interruptible speech output (`stop speaking`).
- Multi-persona profiles (`assistant`, `formal`, `casual`, `professional`, `friendly`, `brief`).
- Offline knowledge base with vector retrieval (FAISS when available, local fallback otherwise).
- Hybrid retrieval reranking + prompt-injection-safe context sanitization.
- Incremental KB sync and deduplicated ingestion.
- Persona-to-voice mapping.
- Session memory + observability dashboard + quick benchmark runner.
- Explain/concise response mode toggles with shared-command output shaping.
- Urgency/politeness tone adaptation and bilingual code-switch continuity handling.
- Response-quality observability metrics (human-likeness, coherence, lexical diversity).
- Real-time core capture with VAD endpointing and async utterance processing.
- Bilingual wake layer with OpenWakeWord (EN) + short-window Arabic trigger checks.
- Benchmark and resilience SLA evaluation blocks in JSON reports.
- Integration wake reliability benchmark (false positives + detection latency) with SLA checks.
- Integration STT reliability benchmark (WER + CER + latency) and TTS quality benchmark (latency + objective quality proxies + fallback reliability + MOS checklist gate) with SLA checks.
- MOS workflow artifacts for human-rated TTS quality collection and aggregation.
- Per-language and per-intent/per-language metrics in observability output.
- Daily and weekly benchmark/resilience rollups in history artifacts.
- Startup + scheduled doctor diagnostics integrated in realtime orchestrator.
- Structured JSON route events in logs for incident analysis.

## Release Hardening Snapshot (2026-04-09)

- Runtime validation refreshed (compileall, full unittest suite, startup doctor checks).
- Benchmark freshness policy validated for wake, STT, TTS, Phase 5 short-horizon, and Phase 5 long-horizon artifacts.
- Mixed-script TTS reliability hardened with chunk limits and Arabic-preferred voice routing.

## Core Commands

- Persona:
  - `persona list`
  - `persona status`
  - `persona set <assistant|formal|casual|professional|friendly|brief>`
  - `assistant mode`
  - `persona voice status`
  - `persona voice clone <profile> on|off`
  - `persona voice provider <profile> voicecraft`
  - `persona voice reference <profile> <path_to_wav>`
- Voice / Speech:
  - `voice status`
  - `voice diagnostic`
  - `voice clone on`
  - `voice clone off`
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
  - `wake status`
  - `wake mode english`
  - `wake mode arabic`
  - `wake mode both`
  - `wake triggers list`
  - `wake triggers add يا جارفس`
  - `wake triggers remove يا جارفس`
  - `set pause scale to 0.9`
  - `set rate offset to -8`
  - `stt profile quiet`
  - `stt profile noisy`
  - `stt profile status`
  - selected STT profile is persisted and restored on next startup
  - `hf profile egyptian`
  - `hf profile english`
  - `hf profile status`
  - `set hf profile to egyptian|english`
  - selected HF profile is persisted, restored on next startup, and forces runtime STT/TTS backends to `huggingface`
  - `stt backend status`
  - `stt backend faster whisper`
  - `stt backend huggingface`
  - `set speech backend to faster_whisper|huggingface`
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
  - `benchmark wake`
  - `benchmark stt`
  - `benchmark tts`
  - `resilience demo`
  - `explain mode on`
  - `concise mode on`
  - `default mode`
- Audit:
  - `verify audit log`
  - `audit reseal`

## Validation Commands

Run static runtime validation:

```powershell
python -m compileall -q .
python core\doctor.py
```

Run wake reliability benchmark:

```powershell
python scripts\benchmark_wake_reliability.py
```

Run STT reliability benchmark (WER + CER + latency):

```powershell
python scripts\benchmark_stt_reliability.py --mode mock
```

Run Egyptian Arabic STT setup benchmark (quality vs latency, low/mid CPU recommendation):

```powershell
python scripts\benchmark_stt_egyptian.py
```

Run direct runtime A/B for Egyptian STT backends on corpus scenarios that include `audio_file` paths:

```powershell
python scripts\benchmark_stt_egyptian.py --runtime-ab --runtime-backends faster_whisper,huggingface
```

Run the small audio-backed Egyptian corpus pack (ready-to-run sample audio included):

```powershell
python scripts\benchmark_stt_egyptian.py --corpus benchmarks\stt_egyptian_corpus_audio_small.json --runtime-ab --runtime-backends faster_whisper,huggingface --runtime-max-cases 3
```

Run Phase 5 dialogue benchmark packs through safe-runtime routing:

```powershell
python scripts\benchmark_phase5_dialogue.py --pack benchmarks\phase5_transcripts.json --output jarvis_phase5_dialogue_benchmark.json
python scripts\benchmark_phase5_dialogue.py --pack benchmarks\phase5_transcripts_long_horizon.json --output jarvis_phase5_dialogue_long_horizon_benchmark.json
```

Run TTS quality benchmark (latency + objective quality proxies):

```powershell
python scripts\benchmark_tts_quality.py --mode mock --backend auto
```

Generate and aggregate MOS workflow artifacts:

```powershell
python scripts\aggregate_tts_mos.py --generate-template
python scripts\aggregate_tts_mos.py --csv benchmarks\tts_mos_sample_ratings.csv
```

Validate freshness policy (artifact age + minimum scenario counts + benchmark gate status):

```powershell
python scripts\check_benchmark_freshness.py --max-age-hours 168
```

Run the automated test suite:

```powershell
python -W error::ResourceWarning -m unittest discover -s tests -p "test_*.py"
```

CI automation:

- GitHub Actions workflow [Jarvis CI](.github/workflows/ci.yml) runs dependency install, compile checks, benchmark freshness policy validation (checked-in and regenerated artifacts), benchmark artifact regeneration, and unit tests on every push/PR to `main` and manual dispatch.

## Milestone Freeze

- Baseline snapshot file: `milestones/phase4_stable_snapshot_2026-03-09.json`

## Phase 5 Contract

- Behavioral contract and EN/AR examples: [docs/PHASE5_BEHAVIOR_CONTRACT.md](docs/PHASE5_BEHAVIOR_CONTRACT.md)
- Scripted transcript benchmark pack: [benchmarks/phase5_transcripts.json](benchmarks/phase5_transcripts.json)
- Benchmark runner: [scripts/benchmark_phase5_dialogue.py](scripts/benchmark_phase5_dialogue.py)
