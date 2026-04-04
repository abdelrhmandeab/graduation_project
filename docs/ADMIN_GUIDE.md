# Admin Guide

## 1. Deployment Model

Jarvis is designed for local-first Windows deployment.

Components:
- Python runtime
- Local model runtime (Ollama)
- Optional speech backends (faster-whisper, Hugging Face, pyttsx3)

## 2. Initial Provisioning

Run setup script:

```powershell
./scripts/setup_windows.ps1
```

Optional speech extras:

```powershell
./scripts/setup_windows.ps1 -InstallSpeechExtras
```

## 3. Environment Configuration

Jarvis reads settings from .env.

Important keys:
- JARVIS_SECOND_FACTOR_PIN
- JARVIS_SECOND_FACTOR_PASSPHRASE
- JARVIS_STT_BACKEND
- JARVIS_STT_HF_MODEL
- JARVIS_TTS_BACKEND
- JARVIS_TTS_HF_MODEL

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

## 6. Logs and Artifacts

Runtime logs:
- jarvis.log
- jarvis_actions.log

Benchmark and resilience artifacts:
- jarvis_benchmark.json
- jarvis_resilience.json
- jarvis_benchmark_history.json
- jarvis_resilience_history.json

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
