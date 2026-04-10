# Intent Schema (Current Runtime)

This document defines the canonical intent and entity contract used by parser, classifier, and router for the current repository state.

## Intent Catalog

| Intent | Action (if any) | Description | Example |
|---|---|---|---|
| `OBSERVABILITY_REPORT` | `""` | Show observability dashboard | `observability` |
| `PERSONA_COMMAND` | `status`, `list`, `set` | Persona profile controls | `persona set formal` |
| `VOICE_COMMAND` | `status`, `diagnostic`, `speech_on`, `speech_off`, `interrupt`, `stt_profile_set`, `stt_profile_status`, `stt_backend_set`, `stt_backend_status`, `voice_quality_set`, `voice_quality_status`, `audio_ux_profile_set`, `audio_ux_profiles`, `audio_ux_status`, `audio_ux_mic_threshold_set`, `audio_ux_wake_threshold_set`, `audio_ux_wake_gain_set`, `audio_ux_pause_scale_set`, `audio_ux_rate_offset_set`, `wake_status`, `wake_mode_set`, `wake_triggers_add`, `wake_triggers_remove` | Runtime voice/STT/TTS/audio UX controls | `voice diagnostic` |
| `KNOWLEDGE_BASE_COMMAND` | `status`, `quality`, `clear`, `retrieval_on`, `retrieval_off`, `sync_dir`, `add_file`, `index_dir`, `search` | Local knowledge base operations | `kb search policy` |
| `MEMORY_COMMAND` | `status`, `show`, `clear`, `on`, `off` | Session memory controls | `memory status` |
| `DEMO_MODE` | `on`, `off`, `status` | Demo-mode output wrapper | `demo mode on` |
| `METRICS_REPORT` | `""` | Show metrics report | `show metrics` |
| `AUDIT_LOG_REPORT` | `""` | Show recent audit log rows | `show audit log 20` |
| `AUDIT_VERIFY` | `""` | Verify audit hash chain | `verify audit log` |
| `AUDIT_RESEAL` | `""` | Reseal broken audit chain from current state | `audit reseal` |
| `POLICY_COMMAND` | `status`, `set_profile`, `set_read_only`, `set_permission` | Policy profile and permission controls | `policy profile strict` |
| `BATCH_COMMAND` | `plan`, `add`, `preview`, `status`, `commit`, `abort` | Transaction-like batch execution | `batch add create folder demo` |
| `SEARCH_INDEX_COMMAND` | `status`, `start`, `refresh`, `search` | File search index management | `index refresh in C:\\` |
| `JOB_QUEUE_COMMAND` | `worker_start`, `worker_stop`, `worker_status`, `enqueue`, `status`, `cancel`, `retry`, `list` | Delayed/background job orchestration | `queue job in 10 create folder backup` |
| `OS_APP_OPEN` | `""` | Open app/process | `open app notepad` |
| `OS_APP_CLOSE` | `""` | Close app/process (confirmation-gated) | `close app notepad` |
| `OS_FILE_SEARCH` | `""` | Search files by name under optional path | `find file report.pdf in desktop` |
| `OS_FILE_NAVIGATION` | `pwd`, `cd`, `list_drives`, `list_directory`, `file_info`, `create_directory`, `delete_item`, `delete_item_permanent`, `move_item`, `rename_item` | Navigation and file operations | `rename old.txt to new.txt` |
| `OS_SYSTEM_COMMAND` | `""` | System-level action request (confirmation-gated) | `shutdown computer` |
| `OS_CONFIRMATION` | `""` | Confirm pending tokenized action | `confirm ab12cd 2468` |
| `OS_ROLLBACK` | `""` | Undo latest rollback-supported action | `undo` |
| `LLM_QUERY` | `""` | Non-deterministic conversational fallback | `explain recursion` |

## Entity Schema

| Entity | Type | Used By | Notes |
|---|---|---|---|
| `app_name` | string | `OS_APP_OPEN`, `OS_APP_CLOSE` | Alias/canonical/executable candidate |
| `filename` | string | `OS_FILE_SEARCH` | Partial or full filename |
| `search_path` | string\|null | `OS_FILE_SEARCH` | Optional search root |
| `path` | string | navigation/KB operations | File/folder path |
| `source` | string | `move_item`, `rename_item` | Source path |
| `destination` | string | `move_item` | Destination path |
| `new_name` | string | `rename_item` | New filename |
| `action_key` | enum | `OS_SYSTEM_COMMAND` | `shutdown`, `restart`, `sleep`, `lock`, `logoff` |
| `token` | hex(6) | `OS_CONFIRMATION` | Confirmation token |
| `second_factor` | string\|null | `OS_CONFIRMATION` | PIN/passphrase for high-risk actions |
| `profile` | string | persona/voice profile actions | persona name, `quiet/noisy`, `arabic/english`, or audio UX profile |
| `mode` | string | quality/profile actions | e.g. `natural`, `standard` |
| `trigger` | string | `wake_triggers_add`, `wake_triggers_remove` | Arabic/English wake phrase text |
| `value` | float\|int\|string | tuning actions | threshold, gain, pause scale, rate offset |
| `query` | string | KB/index search | search query |
| `root` | string\|null | index refresh/search | optional index root |
| `command_text` | string | `BATCH_COMMAND`, `JOB_QUEUE_COMMAND` | queued command payload |
| `job_id` | int | `JOB_QUEUE_COMMAND` | target job id |
| `delay_seconds` | int | `JOB_QUEUE_COMMAND` | queue/retry delay |
| `status` | string\|null | `JOB_QUEUE_COMMAND` list | optional status filter |
| `limit` | int | `AUDIT_LOG_REPORT`, `JOB_QUEUE_COMMAND` list | optional limit |
| `enabled` | bool | policy toggles, clone toggles | flag-like controls |
| `permission` | string | `POLICY_COMMAND` | permission key to toggle |

## Confidence Contract

1. Intent confidence score (`0.0` to `1.0`).
2. Entity confidence map per parsed entity.
3. Clarification gate before execution.

Rules:

- If confidence is low and command appears action-oriented, require clarification.
- If entities are weak or ambiguous, lower effective intent confidence.
- If ambiguous app/file matches are detected, require clarification before execution.
- Clarification state is persisted in session memory until resolved/cancelled/expired.

## Entity Confidence Heuristics (Phase 2)

- `OS_APP_OPEN` and `OS_APP_CLOSE` entity confidence uses app-resolution status:
	- exact alias match: `0.98`
	- high-confidence fuzzy match: `0.92`
	- likely fuzzy match: `0.80`
	- ambiguous multi-match: `0.45` (clarify)
	- unresolved target: falls back to conservative low score
- Low-entity-confidence clarification thresholds:
	- app open/close: `0.58`
	- file search: `0.56`
	- file navigation (path/source/destination/new name): `0.56`
	- system command entities: `0.55`
	- queued command payload entities: `0.52`
- Adaptive thresholding (Phase 2.1):
	- thresholds are configurable per intent via environment (`JARVIS_ENTITY_THRESHOLD_*`)
	- language adjustments apply on top of intent thresholds (`EN` baseline, `AR` adjustment)
	- mixed-script inputs add a conservative clarification bonus threshold
- Clarification memory reuse (Phase 2.1):
	- resolved clarification choices are cached in session memory
	- repeated ambiguous app/file prompts can auto-resolve from recent user preference when still valid
- App ranking improvements (Phase 2.1):
	- candidate score now includes alias similarity + exact/prefix boosts
	- runtime signals contribute: app usage frequency, recency, executable availability, running-process state
- Clarification reply parsing improvements (Phase 2.1):
	- accepts natural phrases like `the chrome one`, `the first app`, and Arabic equivalents
	- supports negative correction patterns (for example `not chrome`)
	- after repeated unrelated replies, router emits one-shot guided examples before re-prompt
- Observability additions (Phase 2.1):
	- clarification rate per intent/language
	- clarification success on first retry
	- wrong-action-prevented counter
	- top ambiguous app/file tokens
- Clarification replies accept numeric replies plus ordinal words (for example `first`, `second`, and Arabic equivalents).

## Safety Expectations

- Medium and high-risk actions are confirmation-gated.
- High-risk actions can require second factor.
- Ambiguous app/file requests are clarified before execution.
- Permanent delete must be explicit and can be disabled by configuration.
- Batch and queued execution reject unsupported high-risk command categories.
