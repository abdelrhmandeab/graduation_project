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

## Awaiting approval

- Wake-word Phase 2 supersedes `data/openwakeword/jarvis_ar/` and `models/arabic_wake_test_tiny/` with the unified corpus/model. Both legacy folders remain untouched pending explicit approval after English and Arabic manual validation; model artifacts are never deleted automatically.

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

## Awaiting approval

- **`models/arabic_wake_test_tiny/`** — contains `jarvis_ar_custom_test_tiny.onnx` (2,009 bytes) + `jarvis_ar_custom_test_tiny.onnx.data` (2,080,768 bytes). Superseded by `models/jarvis_unified/jarvis_unified.onnx`. Safe to delete.
- **`data/openwakeword/jarvis_ar/`** — empty directory. Old Arabic-only training corpus holder. Safe to delete.
