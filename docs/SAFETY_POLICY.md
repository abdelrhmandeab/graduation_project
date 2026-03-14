# Safety Policy

> Auto-generated stub. To be completed during Phase 4.

## Risk Tiers

| Tier | Actions | Confirmation | Second Factor |
|------|---------|-------------|---------------|
| **Low** | open app, search files, list directory | None | No |
| **Medium** | close app, move/rename files | Verbal confirm | No |
| **High** | delete files, system shutdown/restart/logoff | Token confirm | PIN/passphrase |

## Current Safeguards

- `ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS = False` by default.
- Confirmation tokens expire after 45 seconds.
- Second-factor (PIN or passphrase) required for destructive system commands.
- Policy engine enforces path allowlists and blocklists.
- Audit log with hash chain for tamper evidence.
