# Jarvis

Jarvis is a local-first Windows voice assistant runtime with this active pipeline:

wake word (OpenWakeWord) -> STT (egyptalk-transformers by default, with faster-whisper fallback) -> routing/LLM (Ollama) -> TTS (edge-tts or pyttsx3)

## Environment Setup

Install core runtime dependencies:

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set values as needed.

Recommended baseline keys:

```env
JARVIS_STT_BACKEND=egyptalk_transformers
JARVIS_STT_EGYPTALK_ENABLED=true
JARVIS_STT_EGYPTALK_MODEL=NAMAA-Space/EgypTalk-ASR-v2
JARVIS_STT_EGYPTALK_CHUNK_SECONDS=18
JARVIS_STT_EGYPTALK_STRIDE_SECONDS=4
JARVIS_WHISPER_MODEL=small
JARVIS_TTS_BACKEND=auto
JARVIS_TTS_QUALITY_MODE=natural
JARVIS_TTS_EDGE_VOICE=en-US-AriaNeural
JARVIS_TTS_EDGE_ARABIC_VOICE=ar-EG-SalmaNeural
JARVIS_WAKE_MODE=both
JARVIS_LLM_MODEL=qwen2.5:3b
JARVIS_LLM_TIMEOUT_SECONDS=30
JARVIS_LLM_OLLAMA_NUM_CTX=2048
JARVIS_LLM_REALTIME_REWRITE_ENABLED=false
JARVIS_NLU_LLM_QUERY_EXTRACTION_ENABLED=false
JARVIS_KB_TOP_K=3
JARVIS_KB_MAX_CONTEXT_CHARS=1400
JARVIS_MEMORY_MAX_CONTEXT_CHARS=900
```

Notes:
- `JARVIS_TTS_BACKEND` supports `edge_tts`, `pyttsx3`, `auto`, and `console`.
- `auto` is recommended for conservative setups: pyttsx3 works offline by default, while edge-tts is used when available.
- `JARVIS_STT_BACKEND` supports `egyptalk_transformers` and `faster_whisper`.
- Local high-quality default is `egyptalk_transformers`; tune chunk/stride for better long-utterance stability.
- `qwen2.5:3b` is the pinned runtime model for the current low-latency Phase 1 profile.
- Qwen models are under the Qwen license. If you need an alternative, use `llama3.2:3b`.
- Arabic wake triggers and Arabic TTS tuning are enabled by default.
- If audio/wake dependencies are unavailable, orchestrator falls back to text mode.

## Realtime Run

1. Start Ollama in another terminal:

```powershell
ollama serve
```

2. Pull your model once (if needed):

```powershell
ollama pull qwen2.5:3b
```

Optional Modelfile (if you prefer model-side context tuning):

```text
FROM qwen2.5:3b
PARAMETER num_ctx 2048
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
- `stt profile arabic-egy`
- `stt profile code-switched`
- `stt profile auto`
- `stt profile status`
- `stt backend status`
- `stt backend faster whisper`
- `stt backend egyptalk`
- `speech on`
- `speech off`
- `stop speaking`

STT profile notes:
- `arabic-egy` biases decoding toward Arabic for Egyptian-Arabic-heavy turns.
- `code-switched` keeps mixed Arabic/English turns in auto language detection.
- `auto` keeps bilingual auto-detection active for mixed Arabic/English turns.
- `stt backend egyptalk` is available by default; disable with `JARVIS_STT_EGYPTALK_ENABLED=false`.

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
