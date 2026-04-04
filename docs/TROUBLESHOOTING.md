# Troubleshooting Guide

## 1. Ollama is not responding

Symptoms:
- LLM queries fail.

Fix:
1. Start service:

```powershell
ollama serve
```

2. Verify model:

```powershell
ollama pull llama3
```

## 2. No microphone or wake-word capture

Symptoms:
- No voice input is detected.

Fix:
1. Run doctor:

```powershell
python core/doctor.py
```

2. Check sound input device in Windows settings.
3. Use text mode fallback if audio dependencies are unavailable.

## 3. STT returns empty or poor transcript

Fix:
- Switch STT profile:
  - stt profile quiet
  - stt profile noisy
- Check audio UX profile:
  - audio ux profile balanced
  - audio ux profile responsive
  - audio ux profile robust
- If using HF STT, verify model download and backend selection.

## 4. TTS does not play audio

Fix:
- Run voice diagnostic:
  - voice diagnostic
- Try fallback backend:
  - voice quality standard
- Verify pyttsx3 installation and Windows audio output device.

## 5. Confirmation fails unexpectedly

Symptoms:
- Message: token not found or expired.

Fix:
- Confirmation tokens expire by timeout.
- Re-issue the risky command and use the new token.

## 6. Second factor lockout triggered

Symptoms:
- Message: too many failed second-factor attempts.

Fix:
- Wait for lockout window to expire.
- Re-run with correct PIN/passphrase.
- Check .env values for second-factor keys.

## 7. Policy blocks expected operation

Fix:
- Inspect current policy:
  - policy status
- Set expected profile for test/demo scope:
  - policy profile normal
  - policy profile strict

## 8. Regression suite fails

Fix order:
1. Run compile check:

```powershell
python -m compileall -q .
```

2. Run targeted failing suite.
3. Run full Phase 8 gate:

```powershell
python tests/phase8_regression.py
```

4. Review jarvis.log and jarvis_actions.log for root cause.
