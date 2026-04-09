# Safety Policy

This document defines the command safety framework for Jarvis on Windows.

## Risk Tiers

| Tier | Examples | Confirmation | Second Factor |
|---|---|---|---|
| `low` | open app, file search, list directory, metadata, indexed search | Not required | No |
| `medium` | close app, move item, rename item, lock/sleep system command | Required (token confirmation) | No |
| `high` | delete item (soft delete), permanent delete, shutdown/restart/logoff | Required (token confirmation) | Yes (when `SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE=True`) |

## Enforcement Rules

1. High-risk actions must never execute without valid confirmation.
2. Medium-risk actions must require confirmation, but do not require second factor.
3. Soft delete is the default delete behavior.
4. Permanent delete requires explicit phrasing (for example: `delete permanently <path>`) and is blocked unless `ALLOW_PERMANENT_DELETE=True`.
5. Destructive system commands are blocked unless `ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS=True`.
6. Confirmation tokens use configurable entropy (`CONFIRMATION_TOKEN_BYTES`) and parser acceptance supports backward-compatible minimum length.
7. Confirmation attempts are rate-limited per token; repeated failures trigger lockout (`CONFIRMATION_MAX_ATTEMPTS_PER_TOKEN` / `CONFIRMATION_LOCKOUT_SECONDS`).

## Confirmation State Handling

1. Create a confirmation token with expiry (`CONFIRMATION_TIMEOUT_SECONDS`).
2. Store pending confirmation payload in persistence (`confirmations` table).
3. On `confirm <token>`, validate token, expiry, and second factor (if required).
4. Execute only the operation encoded in payload; reject unknown payload kinds.
5. Clear confirmation token after successful acceptance.

## Audit Requirements

Every risky action must emit audit entries for:

1. Request: `*_request` with `risk_tier`, token, and resolved args.
2. Confirmation acceptance/rejection.
3. Final execution outcome (`success`, `failed`, or `blocked`).

The audit log uses a hash chain (`prev_hash` + canonical payload -> `hash`) to support tamper detection.

## Current Config Knobs

- `ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS`
- `ALLOW_PERMANENT_DELETE`
- `SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE`
- `CONFIRMATION_TIMEOUT_SECONDS`
- `CONFIRMATION_TOKEN_BYTES`
- `CONFIRMATION_TOKEN_MIN_HEX_LEN`
- `CONFIRMATION_MAX_ATTEMPTS_PER_TOKEN`
- `CONFIRMATION_LOCKOUT_SECONDS`
- `SECOND_FACTOR_MAX_ATTEMPTS_PER_TOKEN`
- `SECOND_FACTOR_LOCKOUT_SECONDS`
- policy profile permissions (`POLICY_PROFILES` / `POLICY_COMMAND_PERMISSIONS`)
