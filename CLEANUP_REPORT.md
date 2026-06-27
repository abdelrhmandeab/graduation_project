# Phase 5 Cleanup Report

Date: 2026-06-20

## Removed (reference proof)

- Removed unused imports and local assignments reported by Ruff (`F401`/`F841`). A final `ruff check --select F .` passes.
- Removed private helpers and instance fields with no call/read sites, confirmed by repository-wide reference search and Vulture: `_has_mixed_script`, `_language_codes_from_hint`, `_flush_sentence_buffer`, `_looks_low_quality_transcript`, `_transcript_quality_score`, `_should_skip_nlu_llm_query`, `mark_early_response_spoken`, `_speech_started`, and `_vad_detector`.
- Removed duplicate dictionary keys in `os_control/system_ops.py` and `os_control/temporal_parser.py` reported by Ruff (`F601`).
- Fixed an undefined variable in the multi-tool planning path (`language` is now `language_result.language`) reported by Ruff (`F821`).
- Removed configuration entries with no references anywhere in the project: `INPUT_AUDIO_FILE`, the unused Arabic wake-trigger/STT tuning group, the CPU upgrade test group, real-time rewrite flag, default TTS rate, tool-calling flags, and code-switching flag. Matching example-environment entries were removed.
- Removed `pynput` from runtime requirements after both reference search and Deptry found no use.
- Removed the abandoned `core/command_parser.py.patch.tmp` patch artifact. Runtime partial-audio files remain covered by the startup cleanup patterns `jarvis_utterance_*.wav` and `jarvis_partial_*.wav`.
- Removed unused private training arguments and added `requirements-training.txt` for the training-only SciPy and PyTorch dependencies.

## Needs review

- `anthropic` remains in `requirements.txt`. Claude is still referenced by conditional imports, but `llm/claude_client.py` is currently deleted in the working tree. That deletion predates this cleanup and was not changed. Decide whether to restore the Claude backend or remove its remaining integration and dependency together.
- Vulture findings below 80% confidence were retained when they were public APIs, callbacks, platform-specific Windows/COM paths, or dynamically dispatched handlers. Static reference counts are not reliable deletion proof for those paths.
- `.env.example` now exposes every environment key consumed by `core/config.py` (196/196), with advanced defaults grouped in a dedicated section. Secrets in `.env` were not read into this report or modified.
- `faiss-cpu` and `pypdf` are explicit full-install dependencies because their imports are optional knowledge-base features. Deptry requires the package-to-module mapping `faiss-cpu=faiss`.
- No automated test suite exists in the repository. Static checks and a bounded runtime smoke test are the available automated verification; real English/Arabic microphone wake-word checks remain manual hardware tests.

## Verification

- `python -m ruff check --select F .` — pass
- `python -m vulture . --min-confidence 80` — pass (no findings)
- `python -m compileall -q .` — pass
- Deptry — pass with the documented Faiss module mapping, local training-script classification, training-only dependency classification, and the pending Anthropic exception above
- Bounded `python -u main.py` startup — pass: critical warmup completed in 4.35s, Jarvis reached listening state, greeting/TTS ran, and the startup doctor reported 26/26 checks. The process was intentionally stopped after 45s. Background Ollama prewarm timed out and correctly degraded to first-query loading rather than blocking readiness.

---

# Component 2 — Wake Word & Interrupt Cleanup (Phase 6)

Date: 2026-06-25

## Removed (with reference proof)

### Deleted files
- **`audio/barge_in.py`** — `BargeInMonitor`, `ThinkingPhaseMonitor`, `consume_barge_in_wake`, `set_thinking_phase`, `notify_barge_in_wake`, `is_thinking_interrupted`, `clear_thinking_interrupt`, `start_thinking_monitor`, `stop_thinking_monitor`. Grep confirmed zero remaining imports across the codebase before deletion.

### Removed config keys (core/config.py)
All had zero consumers after Phase 4 refactored their call sites:
- `BARGE_IN_INTERRUPT_ON_WAKE` (was hardcoded `True`)
- `WAKE_WORD_IGNORE_WHILE_SPEAKING`
- `BARGE_IN_VAD_ENABLED`
- `BARGE_IN_VAD_ENERGY_THRESHOLD`
- `BARGE_IN_VAD_MIN_SPEECH_SECONDS`
- `BARGE_IN_VAD_GRACE_SECONDS`
- `BARGE_IN_ENERGY_RATIO`
- `BARGE_IN_COOLDOWN_SECONDS`

### Removed .env / .env.example keys
- `JARVIS_BARGE_IN_VAD_GRACE_SECONDS`, `JARVIS_BARGE_IN_ENERGY_RATIO`, `JARVIS_BARGE_IN_VAD_MIN_SPEECH_SECONDS` (from .env)
- `JARVIS_BARGE_IN_COOLDOWN_SECONDS`, `JARVIS_BARGE_IN_ENERGY_RATIO`, `JARVIS_BARGE_IN_VAD_ENABLED`, `JARVIS_BARGE_IN_VAD_ENERGY_THRESHOLD`, `JARVIS_BARGE_IN_VAD_GRACE_SECONDS`, `JARVIS_BARGE_IN_VAD_MIN_SPEECH_SECONDS` (from .env.example)
- `JARVIS_WAKE_WORD_IGNORE_WHILE_SPEAKING` (from both)

### Removed imports
- `core/orchestrator.py`: removed `BARGE_IN_VAD_ENABLED`, `WAKE_WORD_MODE`, `get_runtime_wake_word_behavior` imports
- `audio/wake_word.py`: removed `consume_barge_in_wake` import from `audio.barge_in`, removed `WAKE_WORD_IGNORE_WHILE_SPEAKING` import
- `audio/tts.py`: removed `BargeInMonitor` import from `audio.barge_in`, removed `BARGE_IN_VAD_ENABLED` import

### Removed code
- `audio/tts.py`: removed `_barge_in_monitor` instance, `_on_barge_in_detected()`, `_start_barge_in_monitor()`, `_stop_barge_in_monitor()`, all `set_tts_rms()` calls, and 4 stale barge-in comments
- `core/orchestrator.py`: removed `ignore_while_speaking` gate in main loop, removed `barge_in` wake-source handler, removed `barge_in_interrupt_on_wake` check

### Phase 6 specific cleanup
- Removed stale barge-in comments from `audio/tts.py` (lines 664, 701, 734, 744)
- Replaced `WAKE_WORD_MODE` import in `core/orchestrator.py` with literal `"unified"` string

## Needs review

- `_runtime_wake_word_behavior["barge_in_interrupt_on_wake"]` in `audio/wake_word.py` and `core/handlers/voice.py` — kept for audio UX profile compatibility. The key is a no-op at runtime (coordinator handles interrupt gating) but the voice handler reads/writes it for status display. Removing it would break voice command profile switching.
- `WAKE_WORD_AUDIO_GAIN` config key — imported and stored in `_runtime_wake_word_settings` dict but no longer applied at inference time (gain augmentation during training made runtime gain unnecessary). Kept for the settings API surface.
- `WAKE_WORD_MODE` in `core/config.py` — hardcoded to `"unified"`. Only consumed by the deprecated-keys warning system. Kept for back-compat warning when users set `JARVIS_WAKE_WORD_MODE` in their `.env`.

---

# Component 3 — STT Cleanup (Phase 6)

Date: 2026-06-26

## Removed (with reference proof)

### STT detector and stale language workaround
- Removed `STT_MIXED_TREAT_AS_ARABIC` from `core/config.py`.
- Removed `JARVIS_STT_MIXED_TREAT_AS_ARABIC` from `.env` and `.env.example`.
- Proof: repo-wide grep found no remaining `STT_MIXED_TREAT_AS_ARABIC` / `JARVIS_STT_MIXED_TREAT_AS_ARABIC` references after removal.
- Confirmed the Phase 2 detector removals remain clean: repo-wide grep found no `STT_LANGUAGE_DETECT`, `JARVIS_STT_LANGUAGE_DETECT`, `_detect_audio_language_with_whisper`, `_read_audio_bytes`, `JARVIS_STT_ELEVENLABS_TIMEOUT_SECONDS`, or `JARVIS_STT_ELEVENLABS_ARABIC_LANG` references.

### Static cleanup from the STT sweep
- `audio/tts.py`: removed two unused locals (`released_thread`, `queue_thread_active`) reported by Ruff `F841`. They were assigned in cleanup code but never read.
- `core/adaptive_wake.py`: removed unused `WAKE_WORD_UNIFIED_ONNX_PATH` import reported by Ruff `F401`.
- `core/orchestrator.py`: extended stale temp cleanup to include `jarvis_stt_probe_*.wav`, matching the Phase 1 tiny-probe temp files.
- `docs/JARVIS_PROJECT_BOOK.md` and `core/orchestrator.py`: updated stale references to the removed Whisper language-detector model; wording now points to the locked-language picker and tiny probe path.

## Needs review (kept)

- `utils/language_detector.py` remains live. It is a text/script language helper used after STT, not the removed Whisper detector model.
- Deptry still reports non-STT/package-metadata issues:
  - `core/adaptive_wake.py` local training-script imports (`train_arabic_wake_model`) are not package dependencies.
  - `torch`/`scipy` are training/runtime optional heavy dependencies and currently classified by Deptry as transitive.
  - `faiss` and `pypdf` are optional knowledge-base imports with package/module naming differences.
  - `anthropic` is still reported as dependency-defined-but-unused by Deptry; Claude backend cleanup/restoration remains a separate decision.
- No dependency was removed in this STT sweep because the findings are outside the STT runtime surface or require packaging-policy decisions.

## Verification

- `python -m vulture audio core --min-confidence 80` — pass (no findings)
- `python -m ruff check --select F audio core` — pass
- `python -m compileall -q audio core main.py` — pass
- `python -m deptry .` — reports the non-STT/package-metadata items listed in Needs review
- Config audit: every `JARVIS_STT_*` / `JARVIS_WHISPER_*` key read by `core/config.py` appears in `.env.example`; no extras found.

---

# Component 5 — LLM Surface Cleanup (Phase 7)

Date: 2026-06-27

## Removed (with reference proof)

### Dead config keys (core/config.py)
All had zero consumers after Phases 1–6:
- `LLM_REALTIME_REWRITE_ENABLED` — the secondary LLM rewrite pass was deleted in Phase 3; this flag defaulted to `False` and was never imported anywhere.
- `STREAM_AR_SOFT_FLUSH_CHARS` / `STREAM_AR_HARD_FLUSH_CHARS` — legacy char-count flush thresholds superseded by Phase 5's word-count `SentenceBuffer`. Never imported after Phase 5.
- `VOICE_NORMALIZER_LOCALE` — defined and validated in config but never imported by any module. Language is resolved per-turn from STT/prompt context.

### Dead .env / .env.example keys
- `JARVIS_LLM_REALTIME_REWRITE_ENABLED` (from `.env` and `.env.example`)
- `JARVIS_STREAM_AR_SOFT_FLUSH_CHARS`, `JARVIS_STREAM_AR_HARD_FLUSH_CHARS` (from `.env.example`)
- `JARVIS_VOICE_NORMALIZER_LOCALE` (from `.env.example`)

### Dead code in llm/prompt_builder.py
- `_FEW_SHOT_EXAMPLES` backward-compat alias — assigned `= _FEW_SHOT_EXAMPLES_FULL` but never referenced. The tier-specific `_fewshot_examples_for_tier()` replaced it.
- `build_prompt()` — thin wrapper around `build_prompt_package()["prompt"]`. Never imported or called externally.
- `build_minimal_prompt()` — dead; `build_prompt_package()` with tier auto-selection replaced it.
- `build_full_prompt()` — dead; same reason.
- `build_prompt_for_tier()` — dead dispatcher; never imported.
- `get_system_prompt_for_model()` — dead; never imported.
- Dead `live_data_rule` first assignment in `build_tool_augmented_prompt()` — the if/else block on lines 445–454 was immediately overwritten by lines 455–459. Removed the dead first assignment.

### Dead code in llm/ollama_client.py
- `_SENTENCE_END_RE` regex — compiled but never referenced. `SentenceBuffer` handles all flush logic since Phase 5.

### Dead code in tools/live_data.py
- `_TOOL_FRAMING` initial values — the dict was initialized with old verbose framing strings that were immediately overwritten two lines later. Consolidated into direct assignment.

### Prompt template cleanup (llm/prompts/*.txt)
- Removed `Name: {name}`, `Language: {lang} only.`, `Response style: {style}.` lines from all three templates (`full_prompt.txt`, `slim_prompt.txt`, `micro_prompt.txt`). These are now injected by the inline language pin (Phase 1) and persona block (Phase 2) in `_build_system_block()`.
- Removed the `{name}`, `{lang}`, `{style}` template format arguments from the rendering call in `_build_system_block()`.
- Simplified `_filter_template_lines()` — removed the "be direct … language" filter since that line no longer appears in templates.
- Simplified inline fallback in `_build_system_block()` — removed duplicate `Name:` / `Response style:` / `Language:` lines that duplicated persona block content.

## Needs review (Awaiting approval)

### Ollama models
The following models are installed in `~/.ollama/models`:

| Model | Size | Status |
|---|---|---|
| `qwen3:4b` | 2.5 GB | **Active** — primary runtime model |
| `qwen2.5:3b` | 1.9 GB | **Unused** — was a CPU upgrade test candidate (`JARVIS_LLM_CPU_UPGRADE_MODEL`). The CPU upgrade test config keys were already removed in Phase 5 cleanup. Safe to delete via `ollama rm qwen2.5:3b` |

### Other observations
- `LLM_RESPONSE_CACHE_TTL_SECONDS` remains as a fallback for the split factual/opinion TTL system. It is still imported and read in `command_router.py`. Kept.
- `_normalize_response_language()` in `prompt_builder.py` is still used internally by all prompt builders. Kept.
- `PERSONA_DEFAULT` in `config.py` is still consumed by `PersonaManager.__init__()`. Kept.
- No prompt template file became fully empty after stripping — all three retain their few-shot examples section and `{ar_rule}` placeholder.

## Verification

- `python -m ast` parse check on all edited files — pass
- Repo-wide grep confirms zero remaining references to removed symbols
- `.env.example` matches `core/config.py` for all `JARVIS_LLM_*`, `JARVIS_PERSONA_*`, `JARVIS_VOICE_NORMALIZER_*`, and `JARVIS_SENTENCE_BUFFER_*` keys
