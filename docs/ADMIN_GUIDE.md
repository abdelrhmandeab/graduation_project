# Admin Guide

## 1. Deployment Model

Jarvis is designed for local-first Windows deployment.

Components:
- Python runtime
- Local model runtime (Ollama)
- Speech stack (faster-whisper, pyttsx3, edge-tts)

## 2. Initial Provisioning

Run setup script:

```powershell
./scripts/setup_windows.ps1
```

## 3. Environment Configuration

Jarvis reads settings from .env.

Important keys:
- JARVIS_SECOND_FACTOR_PIN
- JARVIS_SECOND_FACTOR_PASSPHRASE
- JARVIS_STT_BACKEND
- JARVIS_TTS_BACKEND
- JARVIS_TTS_QUALITY_MODE

Safety keys:
- JARVIS_SECOND_FACTOR_MAX_ATTEMPTS_PER_TOKEN
- JARVIS_SECOND_FACTOR_LOCKOUT_SECONDS

## 4. Operations and Runbooks

### Start sequence
1. Start Ollama.
2. Run doctor checks.
3. Start orchestrator.

### Doctor checks

```powershell
python core/doctor.py
```

### Baseline validation

```powershell
python -m compileall -q .
python core/doctor.py
```

## 5. CI Gate Behavior

Workflow file:
- .github/workflows/ci.yml

Key behavior:
- Trigger on push and pull request to main.
- Manual trigger supported.
- Dependency install and syntax validation run on every trigger.

## 6. Logs and Runtime State

Runtime logs:
- jarvis.log
- jarvis_actions.log

## 7. Policy and Audit Governance

Commands:
- policy status
- policy profile strict
- verify audit log
- audit reseal

Administrative recommendation:
- Keep strict profile for demos involving destructive operations.
- Verify audit chain after high-risk testing sessions.

## 8. Backup and Recovery

Backup minimum set:
- .env
- jarvis_memory.json
- jarvis_state.db
- jarvis_actions.log
- jarvis.log

Restore by copying files back into project root and rerunning doctor checks.

## 9. Support Matrix

Validated platform:
- Windows

Supported language gate:
- English
- Arabic
