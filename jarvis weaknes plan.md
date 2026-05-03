You are a senior Python engineer working on Jarvis, a local-first bilingual voice assistant.


Requirements:
- Maintain graceful degradation and hardware adaptation principles.
- Prioritize low latency.
- Add clear comments and update docstrings.
- Include before/after metrics (token count, latency) where relevant.
- Do not regress existing behavior on high-tier models.
- Follow existing code style.

# Jarvis Voice Assistant – Detailed Technical Specification
**Production Improvement Plan – Phases 1, 2 & 3**

**Version:** 1.0  
**Status:** Ready for Implementation  
**Target:** Make Jarvis fast, reliable, and maintainable for daily use on Windows (CPU-only to high-end GPU).

---

## Core Philosophy (Must Preserve)

- Latency over perfection
- Graceful degradation (missing optional deps never crash the app)
- Hardware-aware behavior (auto-select models, adaptive prewarm)
- Bilingual support (English + Egyptian Arabic) with code-switching tolerance
- Local-first with optional cloud augmentation

---

## Phase 1: High Impact – Must Do First

### 1. Tiered / Model-Size-Aware Prompts (`llm/prompt_builder.py`)

**Problem:** Current system prompt is too long (~12+ lines) for qwen3:4b and smaller models → drift, repetition, refusal on safe queries.

**Solution:**
- Create two prompt families:
  - `build_minimal_prompt()` — for low/medium tiers (qwen3:0.6b–4b)
  - `build_full_prompt()` — for high tier (qwen3:8b+)
- Minimal prompt: 6–8 lines max + 2–3 high-quality few-shot examples.
- Remove redundant instructions ("never refuse safe questions", "answer directly", "be concise") that the persona already implies.
- Use clear role assignment and delimiters suitable for Qwen/ChatML format.
- Dynamically select based on `hardware_detect.get_current_tier()` or `JARVIS_LLM_MODEL`.

**Implementation Details:**
- Add new functions: `get_system_prompt_for_model(model_name: str) -> str`
- Store prompts in a new `llm/prompts/` directory or as constants with clear versioning.
- Log prompt token count (via `tiktoken` or Ollama token counter) on first use for monitoring.
- Acceptance Criteria:
  - Token count reduction ≥40% on 4B models.
  - Small models show improved coherence and fewer refusals on test set.
  - No regression on 8B+ models.

**Priority:** Highest  
**Effort:** 1–2 days

---

### 2. Cleanup Post-Processing Pipeline (`core/command_router.py`)

**Problem:** 8+ transforms run on every reply. Some still call the LLM → high latency and inconsistency.

**Solution:**
- Keep **only**:
  - Egyptian colloquial rewriter (move to TTS-only path in `audio/tts.py`)
  - Persona length cap (simple truncation with ellipsis)
  - Low-value fallback (e.g., "I didn't understand, can you rephrase?")
- Remove all LLM-based post-processing.
- Make the pipeline linear and configurable via flags.

**Implementation:**
- Refactor response flow to a clean chain.
- Add `post_process_response(text: str, context: dict) -> str` with minimal steps.
- Update orchestrator to bypass unnecessary steps for action intents.

**Acceptance Criteria:**
- Measurable latency reduction for action replies.
- Consistent output style across turns.

**Priority:** High  
**Effort:** 2 days

---

### 3. Arabic Wake Word Replacement (`audio/wake_word.py`)

**Problem:** Current Whisper-tiny rolling window is inaccurate and relatively expensive.

**Solution:**
- Train/integrate a **custom openWakeWord** ONNX model for "جارفيس" + common Egyptian variations ("يا جارفيس", "جارفيس يا", etc.).
- Keep current Whisper method as fallback when custom model is not available.
- Use synthetic + real data for training (follow openWakeWord training notebook).

**Implementation Steps:**
- Add new class `CustomArabicWakeWord` or integrate into existing detector.
- Support loading `.onnx` model via config (`JARVIS_WAKE_WORD_AR_ONNX_PATH`).
- Parallel detection: English ONNX + Arabic custom model.
- Fallback logic if model fails to load.

**Resources:** See openWakeWord training notebook on GitHub.

**Acceptance Criteria:**
- Wake latency ~25–40 ms.
- False accept rate lower than current implementation.
- CPU usage similar to English wake layer.

**Priority:** High  
**Effort:** 3–5 days (includes data collection/training)

---

### 4. Improve Arabic Sentence Boundary Detection (`llm/ollama_client.py`)

**Problem:** Streaming splits mid-clause in Egyptian Arabic due to weak punctuation regex.

**Solution:**
- Expand `_SENTENCE_END_RE` to include Arabic punctuation: `؟ ، . ! ؟ \n`
- Add fallback: flush after ≥6–8 tokens or ~80–100 characters if no punctuation found.
- Make boundary detection Egyptian-aware (common colloquial patterns).

**Implementation:**
- Create `detect_sentence_boundaries(text: str, is_arabic: bool) -> list[str]`
- Buffer tokens intelligently during streaming.
- Test with mixed MSA/colloquial output.

**Acceptance Criteria:**
- TTS speaks full natural clauses without mid-sentence cuts in Egyptian Arabic.

**Priority:** High  
**Effort:** 1 day

---

### 5. Replace PowerShell for Simple Operations (`os_control/`)

**Problem:** PowerShell child process spawn adds 150–400 ms latency for trivial commands.

**Solution:**
- Use `ctypes` + Windows API (`user32.dll`, `gdi32.dll`, etc.) for:
  - Volume control
  - Screen brightness (DDC/CI via monitor API)
  - Lock screen / sleep
  - Screenshot (full or active window)
  - Get foreground window / app name

**Implementation:**
- Create new module `os_control/native_ops.py` or `ctypes_bridge.py`
- Add fallback to PowerShell if ctypes call fails.
- Update `system_ops.py`, `app_ops.py`, etc., to prefer native path.

**Examples:**
- Volume: `waveOutSetVolume`
- Brightness: Monitor configuration API
- Screenshot: `BitBlt` + `CreateCompatibleBitmap`

**Acceptance Criteria:**
- Latency for volume/brightness reduced by ≥200 ms.
- Graceful fallback maintained.

**Priority:** High  
**Effort:** 2–3 days

---

### 6. Reduce Regex Bloat in Command Parser (`core/command_parser.py`)

**Problem:** ~140 hand-written patterns → hard to maintain, misses paraphrases.

**Solution:**
- Shrink to ~40 patterns covering **structural** commands only (file paths, exact system ops, tokens, confirmation phrases).
- Rely more on semantic router + keyword fuzzy for paraphrases.

**Implementation:**
- Categorize patterns: structural vs semantic.
- Deprecate overlapping patterns.
- Add comments explaining why each regex remains.

**Acceptance Criteria:**
- Parser still catches all critical unambiguous commands.
- Semantic tier handles the rest without regression.

**Priority:** High  
**Effort:** 2 days

---

## Phase 2: Quality & Reliability

**7. Live Data Injection Improvements**  
- Add per-tool instructions when injecting weather/search results.  
- Implement domain allowlist + recency scoring in `tools/web_search.py`.

**8. Session Memory Migration**  
- Move to SQLite (`jarvis_memory.db`).  
- Append-only turns table + key-value slots.  
- Keep JSON export for debugging.

**9. Persistent Language Preference**  
- Store last 3 language detections + user preference in memory/SQLite.  
- Use on STT language hinting.

**10. Egyptian Colloquial Rewriter**  
- Improve regex rules in `audio/tts.py`.  
- Optional: lightweight small model rewrite later.

**11. VAD-based Barge-in During TTS**  
- Detect speech while speaking → treat as implicit interrupt.  
- Update orchestrator and TTS playback loop.

---

## Phase 3: Polish & Future-Proofing

- **Knowledge Base**: Add `watchdog` auto-sync + improved hybrid reranking.
- **Testing**: Comprehensive bilingual test suite (200+ cases) with latency tracking.
- **Diagnostics**: Enhance `core/doctor.py` with model + VRAM checks.
- **Prewarm**: Make adaptive based on CPU core count.
- **Security/UX**: Better confirmation phrases + dry-run mode.
- **Observability**: Structured per-turn metrics.
- **Future**: Optional function/tool calling path for complex intents.

---
