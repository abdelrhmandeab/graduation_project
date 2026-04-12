# Jarvis Graduation Project — Detailed Phase-by-Phase Work Plan
**Repository:** `abdelrhmandeab/graduation_project`  
**Date:** 2026-03-14  
**Scope:** Improve the **existing** project into a reliable, human-like **Windows desktop voice assistant** supporting **Arabic and English only**.

> Historical planning artifact: this file captures the original rollout plan. Current operational state is documented in `README.md`, `docs/ARCHITECTURE_BASELINE.md`, `docs/USER_GUIDE.md`, and `docs/ADMIN_GUIDE.md`.

---

## 1) Vision & Success Criteria

### Vision
Build a production-grade local assistant that can:
- understand spoken Arabic/English commands,
- execute Windows desktop operations safely,
- respond naturally like a human assistant,
- remain robust, observable, and testable.

### Primary Success Criteria
1. **Language discipline:** Arabic + English only (no confusion with other languages).
2. **Action reliability:** Core commands execute correctly with high success rate.
3. **Safety first:** Destructive/system-critical operations require explicit confirmation.
4. **Human-like UX:** Context-aware, concise, natural responses in both languages.
5. **Engineering quality:** Modular architecture, tests, metrics, and clear docs.

---

## 2) Current Repository Assets (What We Reuse)

You already have a strong foundation:
- `audio/` — mic, STT, TTS, VAD, wake-word
- `core/` — parser, classifier, router, orchestrator, memory, metrics
- `os_control/` — policy, confirmation, second factor, file/app/system ops
- `llm/` — prompt building + local model client
- validation suites — smoke, safety, latency, fuzz
- persistence/logging artifacts (`jarvis_*.json`, `.db`, `.log`)

**Decision:** We will **not** create a new project. We will evolve this architecture incrementally.

---

## 3) Product Scope (In Scope / Out of Scope)

## In Scope (v1 target)
- Windows desktop operations:
  - open/close apps
  - file search/open/move/rename/delete
  - system actions (volume, screenshot, lock, etc.)
- web search + open URLs
- bilingual conversational interaction (AR/EN only)
- safety, confirmations, and action audit logs

## Out of Scope (for now)
- Linux/macOS support
- multilingual beyond Arabic/English
- fully autonomous unsupervised high-risk actions
- cloud dependency as mandatory runtime

---

## 4) Functional Feature Set (Final Target)

1. **Speech Input Pipeline**
   - Always-on or push-to-talk mode
   - Wake-word optional
   - VAD + noise tolerance

2. **Bilingual Understanding**
   - strict language gate: Arabic/English only
   - intent classification + entity extraction in AR/EN
   - ambiguity resolution dialog

3. **Action Engine**
   - app operations
   - file operations
   - system operations
   - safe command routing + policy checks

4. **Dialogue Quality**
   - human-like responses
   - memory of recent context
   - clarification and follow-up turns

5. **Safety Layer**
   - risk-tier policy
   - confirmation + second factor for high risk
   - rollback-friendly behavior (e.g., recycle bin default)

6. **Reliability & Observability**
   - structured logs
   - metrics + benchmarks
   - health diagnostics

---

## 5) Non-Functional Requirements

- **Latency:** common command acknowledgement < 1.5s, completion usually < 3s
- **Safety:** zero high-risk action execution without policy-compliant confirmation
- **Accuracy:** high intent precision in AR/EN command sets
- **Resilience:** graceful fallback on model/API/action failure
- **Maintainability:** clear modules and test coverage for core flows

---

## 6) Phase-by-Phase Execution Plan

## Phase 0 — Project Baseline & Planning (2–3 days)
### Objectives
- Freeze current baseline.
- Define measurable targets.
- Prepare tracking artifacts.

### Tasks
- Tag current stable snapshot in git.
- Document architecture overview from existing modules.
- Define KPI baseline from current test/benchmark outputs.
- Create project board with phases and deliverables.

### Deliverables
- `docs/ARCHITECTURE_BASELINE.md`
- `docs/KPI_BASELINE.md`
- milestone board with phase tasks

### Exit Criteria
- Baseline agreed and measurable.

---

## Phase 1 — Language Gate & Normalization (Week 1)
### Objectives
Enforce **Arabic/English-only** behavior end-to-end.

### Tasks
- Add strict language detection after STT.
- Route only `ar` / `en`; reject others politely.
- Add normalization rules:
  - Arabic orthographic normalization
  - English cleanup
- Ensure language preference is stored per session.

### Deliverables
- language gating module integration
- bilingual fallback prompts
- tests for non-supported languages

### Exit Criteria
- Unsupported languages never reach action execution.

---

## Phase 2 — Intent & Entity Reliability (Week 2)
### Objectives
Increase command parsing correctness in AR/EN.

### Tasks
- Define canonical intent schema.
- Add entity confidence scoring.
- Build disambiguation logic for multi-match app/file.
- Expand app alias dictionary (Arabic + English names).

### Deliverables
- intent catalog document
- entity extraction improvements
- clarification flow implementation

### Exit Criteria
- Ambiguous requests trigger clarification, not wrong actions.

---

## Phase 3 — Windows Action Adapters Hardening (Week 3)
### Objectives
Make action execution robust and predictable.

### Tasks
- Standardize adapter interface return format:
  - `success`, `user_message`, `error_code`, `debug_info`
- Harden file path validation and existence checks.
- Improve app launch/close resolution.
- Add retries only where safe.

### Deliverables
- stable `os_control` adapter contracts
- improved error mapping and user-safe messages

### Exit Criteria
- core operations stable across repeated test runs.

---

## Phase 4 — Safety & Permission Framework (Week 4)
### Objectives
Guarantee safe handling of risky commands.

### Tasks
- Implement risk tiers:
  - low: open/search
  - medium: close/kill
  - high: delete/system-critical
- Mandatory confirmations for medium/high.
- Second factor for selected high-risk actions.
- Soft delete default; permanent delete requires explicit phrase.

### Deliverables
- policy matrix (`docs/SAFETY_POLICY.md`)
- confirmation-state handling
- destructive-operation audit trail

### Exit Criteria
- no high-risk action executes without required confirmation flow.

---

## Phase 5 — Human-like Dialogue & Memory (Week 5)
### Objectives
Improve conversational quality and naturalness.

### Tasks
- Expand persona styles (professional/friendly/brief).
- Add anti-repetition response templates.
- Track short-term context:
  - last app, last file, pending confirmation
- Improve follow-up understanding ("open it", "امسحه").

### Deliverables
- bilingual response template library
- memory schema update
- clarification and follow-up behavior improvements

### Exit Criteria
- assistant maintains coherent multi-turn conversations.

---

## Phase 6 — Audio UX Optimization (Week 6)
### Objectives
Make voice interaction feel fluid and realistic.

### Tasks
- Tune VAD thresholds.
- Improve wake-word reliability (if enabled).
- Add barge-in support (interrupt assistant speech).
- Tune TTS per language (pace, pauses, emphasis).

### Deliverables
- updated audio config presets
- speech UX test report

### Exit Criteria
- improved responsiveness and fewer missed/false activations.

---

## Phase 7 — Observability, Metrics, and Diagnostics (Week 7)
### Objectives
Measure quality continuously.

### Tasks
- Add per-intent/per-language metrics.
- Add dashboards/reports from benchmark artifacts.
- Integrate doctor checks into startup and scheduled diagnostics.
- Standardize structured logging for incident analysis.

### Deliverables
- `docs/METRICS_AND_SLO.md`
- daily/weekly benchmark output format
- health check runbook

### Exit Criteria
- KPI trends visible and actionable.

---

## Phase 8 — Comprehensive Testing & QA (Week 8)
### Objectives
Prevent regressions before finalization.

### Tasks
- Expand bilingual utterance suites (AR/EN).
- Add adversarial safety tests.
- Add end-to-end regression scenarios.
- Add performance SLA test gates.

### Deliverables
- updated test packs
- QA sign-off report

### Exit Criteria
- all critical test suites pass with agreed thresholds.

---

## Phase 9 — Packaging, Documentation & Demo Readiness (Week 9)
### Objectives
Prepare project for graduation presentation/use.

### Tasks
- Write user/admin guides.
- Add setup scripts and troubleshooting guide.
- Create demo scenario scripts (AR and EN).
- Finalize changelog and architecture docs.

### Deliverables
- `docs/USER_GUIDE.md`
- `docs/ADMIN_GUIDE.md`
- `docs/DEMO_SCRIPT.md`
- release notes

### Exit Criteria
- reproducible setup + smooth demo flow.

---

## 7) Backlog of Added Features & Improvements

## A) Core Intelligence
- [ ] strict AR/EN language gate
- [ ] intent confidence thresholds
- [ ] entity confidence scoring
- [ ] disambiguation dialogue
- [ ] follow-up command resolution

## B) Windows Capabilities
- [ ] robust app alias map
- [ ] file targeting by name/path/type/date
- [ ] safe delete (recycle bin default)
- [ ] clipboard/screenshot/window focus controls
- [ ] web search/open link actions

## C) Safety & Trust
- [ ] risk-tier policy engine
- [ ] mandatory confirmations
- [ ] second-factor for sensitive actions
- [ ] action audit and tamper-evident logs

## D) Human-Like Conversation
- [ ] bilingual natural response templates
- [ ] anti-repetition logic
- [ ] contextual memory slots
- [ ] emotional tone tuning (calm/professional/helpful)

## E) Reliability & Ops
- [ ] unified structured logs
- [ ] benchmark automation
- [ ] latency/error KPIs
- [ ] self-diagnostic startup checks

---

## 8) KPI Framework (Track Weekly)

1. **Intent Accuracy (AR/EN separately)**
2. **Entity Extraction Accuracy**
3. **Action Success Rate**
4. **High-Risk Safety Compliance (must be 100%)**
5. **p50/p95 Command Latency**
6. **Clarification Rate (lower with maturity)**
7. **User Satisfaction (manual rating per session)**

---

## 9) Risk Register & Mitigation

1. **Arabic dialect variability**
   - Mitigation: expand utterance dataset + normalization + clarification loop

2. **Unsafe destructive commands**
   - Mitigation: strict policy + confirmations + second factor + soft delete

3. **Latency spikes**
   - Mitigation: local caching, async execution, lightweight fallback paths

4. **Model hallucination in action routing**
   - Mitigation: deterministic action schema, LLM only for understanding/wording

5. **Windows environment variability**
   - Mitigation: adapter abstraction + robust error handling + compatibility tests

---

## 10) Team/Execution Cadence (Even for Solo Developer)

- **Daily:** code + tests + short benchmark run
- **Twice weekly:** safety regression + latency checks
- **Weekly review:** KPI report + next sprint reprioritization
- **Milestone end:** demo scenario pass in Arabic and English

---

## 11) Definition of Done (Project Completion)

Project is considered complete when:
1. Assistant supports only Arabic/English without leakage.
2. Core Windows commands are reliable and safe.
3. Human-like responses are context-aware and non-robotic.
4. Safety policy is enforced for all risky operations.
5. Test suites + benchmark thresholds pass consistently.
6. Setup/docs/demo flow are complete and reproducible.

---

## 12) Immediate Next Sprint (Start Now)

### Sprint Goal (7 days)
“Establish strict bilingual safety baseline.”

### Must-do tasks
1. Implement AR/EN language gate in STT-to-parser flow.
2. Add risk-tier confirmation enforcement for delete/system actions.
3. Add intent/entity confidence + clarification fallback.
4. Add 50 Arabic + 50 English critical command tests.
5. Generate updated benchmark and safety report.

### Sprint Exit Output
- working bilingual-safe command loop
- measurable improvement report
- clear backlog for Phase 2

---

## 13) Documentation Files to Add

- `docs/ARCHITECTURE_BASELINE.md`
- `docs/INTENT_SCHEMA.md`
- `docs/SAFETY_POLICY.md`
- `docs/METRICS_AND_SLO.md`
- `docs/TEST_PLAN.md`
- `docs/USER_GUIDE.md`
- `docs/DEMO_SCRIPT.md`

---

## 14) Final Note

This plan is intentionally built to **upgrade your current repository in-place**.  
No rewrite, no new project.  
