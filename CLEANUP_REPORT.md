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

- Retained the root parser-tuning utilities `enhance_patterns.py`, `enhance_patterns_v2.py`, `enhance_patterns_v3.py`, and `fix_regex.py`. They have no runtime imports, but project documentation identifies them as maintenance utilities. Delete or archive only with explicit approval.
- No model is an unused deletion candidate. Both files under `models/arabic_wake_test_tiny/` are configured by the local environment and were retained. Model artifacts must never be deleted automatically.

## Verification

- `python -m ruff check --select F .` — pass
- `python -m vulture . --min-confidence 80` — pass (no findings)
- `python -m compileall -q .` — pass
- Deptry — pass with the documented Faiss module mapping, local training-script classification, training-only dependency classification, and the pending Anthropic exception above
- Bounded `python -u main.py` startup — pass: critical warmup completed in 4.35s, Jarvis reached listening state, greeting/TTS ran, and the startup doctor reported 26/26 checks. The process was intentionally stopped after 45s. Background Ollama prewarm timed out and correctly degraded to first-query loading rather than blocking readiness.
