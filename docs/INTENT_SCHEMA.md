# Intent Schema

> Auto-generated stub. To be completed during Phase 2.

## Intent Categories

| Intent | Description | Example (EN) | Example (AR) |
|--------|------------|--------------|--------------|
| `OS_FILE_SEARCH` | Find files by name | "find file report.pdf" | TBD |
| `OS_FILE_NAVIGATION` | Navigate/manage filesystem | "go to Desktop" | TBD |
| `OS_APP_OPEN` | Launch application | "open chrome" | TBD |
| `OS_SYSTEM_COMMAND` | System-level action | "shutdown computer" | TBD |
| `LLM_QUERY` | General question (fallback to LLM) | "what is Python?" | TBD |

## Entity Schema

| Entity | Type | Example |
|--------|------|---------|
| `filename` | string | "report.pdf" |
| `app_name` | string | "chrome" |
| `path` | string | "C:\\Users\\Desktop" |
| `action_key` | enum | "shutdown", "restart", "sleep" |
