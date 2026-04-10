# Jarvis

Jarvis is a local-first Windows voice assistant runtime with this active pipeline:

wake word (OpenWakeWord) -> STT (faster-whisper) -> routing/LLM (Ollama) -> TTS (edge-tts or pyttsx3)

## Environment Setup

Install core runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set values as needed.

Recommended baseline keys:

```env
JARVIS_STT_BACKEND=faster_whisper
JARVIS_WHISPER_MODEL=base
JARVIS_TTS_BACKEND=auto
JARVIS_TTS_QUALITY_MODE=natural
JARVIS_TTS_EDGE_VOICE=en-US-AriaNeural
JARVIS_TTS_EDGE_ARABIC_VOICE=ar-EG-SalmaNeural
JARVIS_WAKE_MODE=both
JARVIS_LLM_MODEL=qwen2.5:1.5b
```

Notes:
- `JARVIS_TTS_BACKEND` supports `edge_tts`, `pyttsx3`, `auto`, and `console`.
- `auto` is recommended for conservative setups: pyttsx3 works offline by default, while edge-tts is used when available.
- Arabic wake triggers and Arabic TTS tuning are enabled by default.
- If audio/wake dependencies are unavailable, orchestrator falls back to text mode.

## Realtime Run

1. Start Ollama in another terminal:

```powershell
ollama serve
```

2. Pull your model once (if needed):

```powershell
ollama pull qwen2.5:1.5b
```

3. Run startup diagnostics:

```powershell
python core\doctor.py
```

4. Start realtime orchestrator:

```powershell
python core\orchestrator.py
```

## Core Commands

Persona:
- `persona list`
- `persona status`
- `persona set <assistant|formal|casual|professional|friendly|brief>`

Voice and latency:
- `voice status`
- `voice diagnostic`
- `voice quality natural`
- `voice quality standard`
- `voice quality status`
- `audio ux profiles`
- `audio ux profile balanced`
- `audio ux profile responsive`
- `audio ux profile robust`
- `audio ux status`
- `set mic threshold to 0.012`
- `set wake threshold to 0.38`
- `set wake gain to 1.6`
- `set pause scale to 0.9`
- `set rate offset to -8`
- `wake status`
- `wake mode english`
- `wake mode arabic`
- `wake mode both`
- `wake triggers list`
- `wake triggers add يا جارفس`
- `wake triggers remove يا جارفس`
- `stt profile quiet`
- `stt profile noisy`
- `stt profile status`
- `stt backend status`
- `stt backend faster whisper`
- `speech on`
- `speech off`
- `stop speaking`

Knowledge base:
- `kb status`
- `kb add <path_to_file>`
- `kb index <path_to_directory>`
- `kb sync <path_to_directory>`
- `kb search <query>`
- `kb quality`
- `kb retrieval on`
- `kb retrieval off`
- `kb clear`

Memory and observability:
- `memory status`
- `memory show`
- `memory on`
- `memory off`
- `memory clear`
- `observability`

Audit and policy:
- `policy status`
- `verify audit log`
- `audit reseal`

## Validation

```powershell
python -m compileall -q .
python core\doctor.py
```

## Related Docs

- [Architecture baseline](docs/ARCHITECTURE_BASELINE.md)
- [Intent schema](docs/INTENT_SCHEMA.md)
- [Safety policy](docs/SAFETY_POLICY.md)
- [User guide](docs/USER_GUIDE.md)
- [Admin guide](docs/ADMIN_GUIDE.md)
