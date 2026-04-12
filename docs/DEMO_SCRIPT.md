# Demo Script (English + Arabic)

## Demo Goals

Show that Jarvis is:
- bilingual (English and Arabic)
- safe for risky actions
- context-aware in multi-turn flows
- observable and testable

Total suggested duration: 8-12 minutes.

## Pre-demo Checklist

1. Start Ollama.
2. Run doctor check:

```powershell
python core/doctor.py
```

3. Start Jarvis:

```powershell
python core/orchestrator.py
```

4. Keep a terminal ready for backup text input.

## Scenario A: English Flow

### A1. Voice and runtime health
Input:
- voice status
- voice diagnostic

Expected:
- Voice status block appears.
- Diagnostic confirms active backend and playback attempt.

### A2. File search with clarification
Input:
- find file report
- 1

Expected:
- Jarvis asks for clarification when multiple matches exist.
- Selection resolves to a specific file path.

### A3. Follow-up resolution
Input:
- delete it
- confirm it
- confirm <token> <PIN_or_passphrase>

Expected:
- Confirmation is required for deletion.
- Token-based flow is enforced.

### A4. Observability snapshot
Input:
- observability

Expected:
- Command metrics, language metrics, intent/language metrics, stage metrics.

## Scenario B: Arabic Flow

### B1. Arabic command understanding
Input:
- حالة الصوت
- دور على ملف تقرير في المكتب

Expected:
- Arabic intent parsing succeeds.
- Search output appears in expected format.

### B2. Arabic follow-up command
Input:
- امسحه

Expected:
- Jarvis uses recent context and requests safe confirmation.

### B3. Safety refusal behavior
Input:
- امسح الملف نهائي

Expected:
- Permanent delete remains blocked by configuration unless explicitly permitted.

## Scenario C: Safety and Policy Proof

Input:
- delete it
- confirm it
- confirm <token> wrong-pin

Expected:
- Confirmation flow is enforced.
- Incorrect second factor is rejected.

## Scenario D: Runtime Health Verification

Run in terminal:

```powershell
python -m compileall -q .
python core/doctor.py
```

Expected:
- Syntax validation completes without errors.
- Doctor report shows environment readiness details.

## Demo Closing

State final claims:
- Bilingual operation is enforced (AR/EN only).
- High-risk operations require confirmation and second factor.
- Runtime validation is reproducible via compileall + doctor checks and CI syntax gate.
- Setup and operations are documented for reproducible execution.
