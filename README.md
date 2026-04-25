# Jarvis

A local-first Windows voice assistant focused on production runtime stability.
Bilingual (English + Egyptian Arabic), runs on any Windows 10/11 PC with 8GB+ RAM —
no GPU required, no admin needed for everyday commands.

## Architecture

```
Wake word → STT → 3-tier intent router → Action / LLM → TTS
```

- **STT**:
  - Arabic: ElevenLabs STT (cloud) with local Faster-Whisper (`small`) fallback
  - English: local Faster-Whisper (`small`)
- **Intent routing** (cascade — earliest tier that hits wins):
  1. **Regex parser** (0 ms) — exact keywords + bilingual patterns
  2. **Semantic router** (~10 ms) — multilingual MiniLM embeddings, handles paraphrases
  3. **Keyword NLP** — fuzzy match (rapidfuzz) for noisy STT
  4. **LLM fallback** — local Ollama (Qwen3 family) for conversational queries
- **Action handlers**: 35+ system commands + timers, clipboard, sysinfo, settings,
  Outlook email/calendar drafts, Windows Search Index, persona, knowledge base.
- **TTS**:
  - Arabic: ElevenLabs TTS, falling back to edge-tts `ar-EG-SalmaNeural` when offline
  - English: edge-tts `en-US-AriaNeural`

## Hardware-aware model selection

At startup, [core/hardware_detect.py](core/hardware_detect.py) reads RAM + GPU and
picks the best Qwen3 model that fits. Override via `JARVIS_LLM_MODEL` in `.env`,
or disable with `JARVIS_LLM_AUTO_SELECT=false`.

| RAM    | GPU | Tier      | Model        | num_ctx | lightweight_ctx |
|--------|-----|-----------|--------------|---------|-----------------|
| 16 GB+ | Yes | `high`    | qwen3:8b     | 8192    | 4096            |
| 12 GB+ | Any | `medium`  | qwen3:4b     | 4096    | 2048            |
| 8 GB   | Yes | `medium`  | qwen3:4b     | 4096    | 2048            |
| 8 GB   | No  | `low`     | qwen3:1.7b   | 2048    | 1024            |
| < 8 GB | Any | `minimal` | qwen3:0.6b   | 1024    | 512             |

Missing models are auto-pulled on first run with streaming progress logs —
no manual `ollama pull` needed.

## Setup

### 1. Install dependencies

Pick the dependency tier that matches your needs:

```powershell
# Default — full primary feature set (recommended)
python -m pip install -r requirements.txt

# Minimal — voice + LLM only, fewer optional features
python -m pip install -r requirements-minimal.txt

# Full — everything including optional FAISS / PDF tools
python -m pip install -r requirements-full.txt
```

### 2. Copy and edit the environment template

```powershell
copy .env.example .env
notepad .env
```

Required keys:
- `ELEVENLABS_API_KEY` — only if you want cloud-quality Arabic STT/TTS.
  Without it, Jarvis falls back to local Whisper + edge-tts Arabic voice automatically.
- `JARVIS_TTS_ELEVENLABS_ARABIC_VOICE_ID` — paired with the API key.

Everything else has a sensible default.

### 3. Start Ollama

```powershell
ollama serve
```

You don't need to manually `ollama pull` anything — the orchestrator auto-pulls
the configured / hardware-recommended model on first launch.

## Run

```powershell
python main.py
```

Wake word: say **"Jarvis"** (English wake-word ONNX model) or **"جارفيس"** (Egyptian
Arabic, Whisper `tiny`-based detection — say it once).

## Health check

Run the doctor to verify all dependencies, audio devices, Ollama, and feature tiers:

```powershell
python core/doctor.py
```

The report flags each optional feature as `available` or `degraded (...)` so you
can see exactly which extras are active without grepping logs.

## Optional dependencies

All of these gracefully degrade if missing — Jarvis still starts:

| Package                    | Feature                                                  |
|----------------------------|----------------------------------------------------------|
| `pyperclip`                | Clipboard read/write                                     |
| `screen-brightness-control`| DDC/CI brightness (no admin)                             |
| `sentence-transformers`    | Semantic intent routing (paraphrase tolerance)           |
| `duckduckgo-search` / `ddgs` | Web search for live data                               |
| `pywin32`                  | Outlook email/calendar drafts + Windows Search Index     |

## Live data

Weather queries hit Open-Meteo (no API key). Search/news/price queries hit
DuckDuckGo (no API key). Results are injected into the LLM prompt as a `LIVE DATA`
section so the model can answer with real, current information instead of an
"I don't have live data" apology.

## Project layout

```
core/         — orchestrator, command parser/router, config, doctor, hardware detect
audio/        — wake-word, STT, TTS, mic, VAD
llm/          — Ollama client, prompt builder
nlp/          — semantic router, fuzzy matcher, keyword engine
os_control/   — timer, clipboard, sysinfo, settings, file/app/system ops, email, calendar
tools/        — weather (Open-Meteo), web_search (DuckDuckGo)
```
