# Intent Schema (Phase 2)

This document defines the canonical intent and entity contract used by parser, classifier, and router.

## Intent Catalog

| Intent | Action (if any) | Description | Example |
|---|---|---|---|
| `OS_APP_OPEN` | `""` | Open an app or executable | `open app notepad` |
| `OS_APP_CLOSE` | `""` | Close an app or process (medium risk; confirmation required) | `close app notepad` |
| `OS_FILE_SEARCH` | `""` | Search files by name under a path | `find file report.pdf` |
| `OS_FILE_NAVIGATION` | `pwd` | Show current directory | `current directory` |
| `OS_FILE_NAVIGATION` | `cd` | Change working directory | `go to Desktop` |
| `OS_FILE_NAVIGATION` | `list_drives` | List system drives | `list drives` |
| `OS_FILE_NAVIGATION` | `list_directory` | List files/folders in a directory | `list files in Downloads` |
| `OS_FILE_NAVIGATION` | `file_info` | Show file metadata | `file info notes.txt` |
| `OS_FILE_NAVIGATION` | `create_directory` | Create directory | `create folder demo` |
| `OS_FILE_NAVIGATION` | `delete_item` | Soft delete file/folder (rollback supported) | `delete temp.txt` |
| `OS_FILE_NAVIGATION` | `delete_item_permanent` | Permanent delete (high risk; explicit phrasing required) | `delete permanently temp.txt` |
| `OS_FILE_NAVIGATION` | `move_item` | Move file/folder | `move a.txt to archive\\a.txt` |
| `OS_FILE_NAVIGATION` | `rename_item` | Rename file/folder | `rename old.txt to new.txt` |
| `OS_SYSTEM_COMMAND` | `""` | OS-level command request (confirmation gated) | `shutdown computer` |
| `OS_CONFIRMATION` | `""` | Confirm pending tokenized action | `confirm ab12cd 2468` |
| `OS_ROLLBACK` | `""` | Undo latest rollback-supported action | `undo` |
| `LLM_QUERY` | `""` | Non-deterministic fallback query | `explain recursion` |

## Entity Schema

| Entity | Type | Used By | Notes |
|---|---|---|---|
| `app_name` | string | `OS_APP_OPEN`, `OS_APP_CLOSE` | Alias/canonical/executable candidate |
| `filename` | string | `OS_FILE_SEARCH` | Partial or full filename |
| `search_path` | string\|null | `OS_FILE_SEARCH` | Optional search root |
| `path` | string | file navigation actions | File/folder path |
| `source` | string | `move_item`/`rename_item` | Source path |
| `destination` | string | `move_item` | Destination path |
| `new_name` | string | `rename_item` | New filename only |
| `action_key` | enum | `OS_SYSTEM_COMMAND` | `shutdown`, `restart`, `sleep`, `lock`, `logoff` |
| `token` | hex(6) | `OS_CONFIRMATION` | Confirmation token |
| `second_factor` | string\|null | `OS_CONFIRMATION` | PIN/passphrase for high-risk actions |

## Confidence Contract

1. Intent confidence score (`0.0` to `1.0`).
2. Entity confidence map per parsed entity.
3. Clarification gate before execution.

Rules:

- If confidence is low and command appears action-oriented, require clarification.
- If entities are weak/ambiguous, lower effective intent confidence.
- If ambiguous app/file matches are detected, require clarification before execution.
- Clarification state is persisted in session memory until resolved/cancelled/expired.

## Safety Expectation

- Medium and high-risk actions are confirmation-gated.
- High-risk actions can require second factor.
- Ambiguous app/file requests are clarified before execution.
- Permanent delete must be explicit and can be disabled by configuration.
