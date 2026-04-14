# Jarvis

Jarvis is a local-first Windows voice assistant focused on production runtime stability.

## Architecture

STT -> NLP/Intent Routing -> TTS

- STT:
  - Arabic: ElevenLabs STT
  - English: local Faster-Whisper (`small`)
  - Fallback: local Faster-Whisper when ElevenLabs STT fails
- NLP:
  - Language gate + parser + intent router
  - Command execution for OS actions
  - LLM fallback through Ollama (`qwen2.5:3b`)
- TTS:
  - Arabic: ElevenLabs TTS
  - English: edge-tts (`en-US-AriaNeural`)

## Setup

1. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

2. Copy config template:

```powershell
copy .env.example .env
```

3. Fill required values in `.env`:
- `ELEVENLABS_API_KEY`
- `JARVIS_TTS_ELEVENLABS_ARABIC_VOICE_ID`

4. Ensure Ollama model is available:

```powershell
ollama serve
ollama pull qwen2.5:3b
```

## Required Environment Variables

Minimum required for production:

```env
JARVIS_LLM_MODEL=qwen2.5:3b
JARVIS_STT_BACKEND=hybrid_elevenlabs
ELEVENLABS_API_KEY=
JARVIS_TTS_BACKEND=hybrid
JARVIS_TTS_ELEVENLABS_ARABIC_VOICE_ID=
```

Recommended performance/runtime variables are documented in `.env.example`.

## Run

Preferred entrypoint:

```powershell
python main.py
```

Legacy equivalent:

```powershell
python -m core.orchestrator
```

## Health Check

```powershell
python core/doctor.py
```
