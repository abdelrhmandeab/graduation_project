# Jarvis — Weaknesses-Only Analysis & Production Work Plan
## For Claude Code / Codex execution

---

## WEAKNESS 1: LLM is broken by design — `_PINNED_MODEL` ignores all config

**The actual bug**: In `llm/ollama_client.py`, line `_PINNED_MODEL = "qwen2.5:3b"` and `_resolve_model_name()` ALWAYS returns `_PINNED_MODEL` regardless of what's in `.env`. It even logs a warning saying it's ignoring your config. This means even when you tried `qwen3:4b`, the code silently used `qwen2.5:3b` anyway.

```python
# CURRENT (broken):
def _resolve_model_name():
    configured = str(LLM_MODEL or "").strip()
    if configured and configured.lower() != _PINNED_MODEL:
        logger.warning(
            "Configured LLM model '%s' ignored; runtime is pinned to '%s'.",
            configured, _PINNED_MODEL,
        )
    return _PINNED_MODEL  # ← ALWAYS returns qwen2.5:3b no matter what
```

### Fix (Task 1.1):
```python
# FIXED:
def _resolve_model_name():
    configured = str(LLM_MODEL or "").strip()
    if configured:
        return configured
    return "qwen3:4b"  # sensible default
```

**Files**: `llm/ollama_client.py`
**Priority**: CRITICAL — nothing else matters until this is fixed
**Time**: 5 minutes

---

## WEAKNESS 2: System prompt is 40+ lines — drowns the small model

**The problem**: `build_prompt_package()` in `llm/prompt_builder.py` produces a system prompt that's 40+ lines with sections like `RESPONSE_LANGUAGE_REQUIREMENT`, `SHORT-TERM CONTEXT`, `LOCAL KNOWLEDGE BASE CONTEXT`, persona text, memory context, etc. A 3-4B model's instruction-following degrades after ~500-800 tokens of system text. The model "forgets" to speak Egyptian Arabic by the time it reaches the user query.

**The bloat breakdown**:
- 6 lines: persona prompt (from `persona_manager`)
- 4 lines: "You are Jarvis" repeated identity (already in persona)
- 5 lines: language requirement section (redundant with "Reply in X only")
- 3 lines: safety constraints
- 3 lines: memory context section
- 3 lines: short-term context section (last_app, last_file, pending_confirmation)
- 5 lines: KB context section
- Multiple lines: response mode, code-switch bridges, anti-repetition

**Total**: ~800-1200 tokens of system text before the user query even appears.

### Fix (Task 1.2):
Cut to ≤15 lines. Remove all redundancy. Keep only what a small model can follow.

```python
def build_prompt_package(user_text, response_language="en"):
    query = (user_text or "").strip()
    lang_label = "Arabic (Egyptian dialect)" if response_language == "ar" else "English"

    sections = [
        "SYSTEM:",
        f"You are Jarvis, a voice assistant. Reply in {lang_label} only. Be concise (1-3 sentences).",
        "When Arabic, use Egyptian colloquial (مصري), not formal MSA.",
        "Answer directly. If you lack live data, say so briefly then give practical advice.",
        "",
    ]

    # Few-shot examples (most effective technique for small models)
    sections.extend([
        "Examples:",
        "USER: الجو عامل ازاي؟",
        "ASSISTANT: مش معايا بيانات طقس دلوقتي، بس لو في القاهرة الأيام دي الجو حر — البس خفيف واشرب مية كتير.",
        "USER: what is python?",
        "ASSISTANT: Python is a programming language known for its simple syntax. It's widely used for web development, data science, and automation.",
        "USER: افتحلي كروم",
        "ASSISTANT: تمام، بفتح جوجل كروم دلوقتي.",
        "",
    ])

    # Inject tool context if available (Phase 2)
    # Inject KB context only if present, as ONE line
    kb_package = knowledge_base_service.retrieve_for_prompt(query, top_k=3, max_chars=800)
    if kb_package["context"]:
        sections.append(f"Context: {kb_package['context']}")
        sections.append("")

    sections.extend(["USER:", query, "", "ASSISTANT:"])
    return {"prompt": "\n".join(sections), ...}
```

**Files**: `llm/prompt_builder.py`
**Priority**: CRITICAL
**Time**: 1-2 hours

---

## WEAKNESS 3: Triple LLM call per response = 3x latency

**The problem**: In `_finalize_success_response()` in `core/command_router.py`, every LLM response goes through:
1. `_repair_low_value_llm_response()` — calls `ask_llm()` AGAIN to rewrite
2. `_enforce_llm_response_language()` — calls `ask_llm()` AGAIN to translate
3. `_apply_egyptian_dialect_style()` — calls TTS rewriter (text-level, no LLM)
4. `_apply_persona_length_target()` — truncation
5. `_apply_output_mode()` — mode shaping
6. `_apply_tone_adaptation()` — tone prefixes
7. `_apply_codeswitch_continuity()` — code-switch bridges
8. `_apply_anti_repetition()` — de-duplication

Steps 1 and 2 each call the LLM. So a single user question triggers **3 LLM calls** (original + repair + language enforce). On CPU, each call is 2-5 seconds. Total: 6-15 seconds per response.

### Fix (Task 1.3):
Delete the LLM rewrite calls. With a better model + better prompt, they're unnecessary. If the model output is bad, a second call to the same bad model won't fix it.

```python
def _finalize_success_response(response_text, parsed, language, original_text, tone_meta, *, realtime=False):
    text = str(response_text or "").strip()

    # If truly empty/useless, use the assist-first fallback directly (no LLM call)
    if _looks_low_value_llm_reply(text) and _is_assist_first_safe_request(original_text):
        text = _fallback_assist_first_response(original_text, language) or text

    # Text-level post-processing only (no LLM calls)
    text = _apply_egyptian_dialect_style(text, parsed, language)
    text = _apply_persona_length_target(text, parsed)
    text = _apply_output_mode(text, parsed, language)
    text = _apply_tone_adaptation(text, language, tone_meta, parsed=parsed)
    text = _apply_anti_repetition(text, language)
    _record_response_quality(text, language, original_text)
    return text
```

**Files**: `core/command_router.py`
**Priority**: CRITICAL
**Time**: 30 minutes

---

## WEAKNESS 4: Wake word detection is unreliable

**The problem**: Arabic wake word detection uses `whisper tiny` model to transcribe 1.5-second audio chunks, then string-matches against trigger phrases like "جارفيس". This has multiple issues:
- `whisper tiny` is terrible at Arabic — it mishears "جارفيس" frequently
- Requires 2 consecutive hits within 3 seconds (so you say "Jarvis" twice)
- The English openwakeword model uses a dedicated wake-word ONNX model (much more reliable)
- Arabic path uses general-purpose STT for a wake-word task — wrong tool for the job

**The design flaw**: Using a general-purpose STT model for wake-word detection is like using a sledgehammer to hang a picture. Wake-word detection needs a lightweight, always-on, specialized model — not a full transcription engine.

### Fix (Task 4.1):
**Option A (quick win)**: Train a custom openwakeword model for "جارفيس" / "يا جارفيس" — openwakeword supports custom wake words via their training pipeline. This gives Arabic the same reliability as English detection.

**Option B (medium effort)**: Use `whisper small` instead of `tiny` for Arabic wake detection. `tiny` has terrible Arabic accuracy. `small` is significantly better and still fast enough for 1.5-second chunks on CPU (~200ms inference).

**Option C (production-grade)**: Use `Porcupine` from Picovoice — it's specifically built for wake-word detection, supports Arabic, runs on CPU with <1% usage, and is free for personal use. It's what production assistants actually use.

```python
# Option B — quick improvement (change in config)
JARVIS_WAKE_WORD_AR_STT_MODEL=small  # instead of tiny
JARVIS_WAKE_WORD_AR_CONSECUTIVE_HITS_REQUIRED=1  # instead of 2
JARVIS_WAKE_WORD_AR_CHUNK_SECONDS=2.0  # instead of 1.5
```

**Files**: `audio/wake_word.py`, `.env.example`, `core/config.py`
**Priority**: HIGH
**Time**: Option B = 10 minutes. Option A = 1-2 days. Option C = 1 day.

---

## WEAKNESS 5: No real-time data — weather/news queries hit a dead end

**The problem**: When a user asks about weather, news, or prices, the code detects it via `_looks_weather_or_clothing_query()` and `_looks_news_query()` markers, then returns hardcoded apology strings from `_fallback_assist_first_response()`. The LLM itself has no internet access. The user gets useless responses.

### Fix (Task 2.1-2.3):
Add two free APIs (no keys, no signup, no cost):

```python
# tools/web_search.py
from duckduckgo_search import DDGS

def search_web(query: str, max_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return "\n".join(f"- {r['title']}: {r['body']}" for r in results) if results else ""
    except Exception:
        return ""
```

```python
# tools/weather.py
import httpx

WEATHER_CODES = {0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
                 45: "Foggy", 51: "Light drizzle", 61: "Slight rain", 63: "Moderate rain",
                 65: "Heavy rain", 71: "Slight snow", 80: "Rain showers", 95: "Thunderstorm"}

def get_weather(lat=30.04, lon=31.24, city="Cairo") -> str:
    try:
        r = httpx.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude": lat, "longitude": lon,
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                    "timezone": "auto"}, timeout=5.0)
        d = r.json().get("current", {})
        cond = WEATHER_CODES.get(d.get("weather_code", 0), "Unknown")
        return f"Weather in {city}: {cond}, {d.get('temperature_2m')}°C, humidity {d.get('relative_humidity_2m')}%, wind {d.get('wind_speed_10m')} km/h"
    except Exception:
        return ""
```

**Integration**: In `command_router.py`, before LLM fallback, check if query matches search/weather markers → call tool → inject result into prompt as "LIVE DATA:" section → LLM generates answer WITH data.

**Files**: `tools/web_search.py` (new), `tools/weather.py` (new), `core/command_router.py`, `llm/prompt_builder.py`, `requirements.txt`
**Priority**: CRITICAL
**Time**: 3-4 hours

---

## WEAKNESS 6: Missing everyday commands

**Currently have**: volume, brightness, wifi, bluetooth, screenshot, media controls, browser tabs, window management, file operations, app open/close, shutdown/restart/sleep/lock, notifications/DND.

**Missing** (sorted by user request frequency):

| Command | Implementation | Effort |
|---------|---------------|--------|
| Timer/alarm | `threading.Timer` + `winsound.Beep` (built-in Python) | 2 hours |
| "What's my battery?" | `psutil.sensors_battery()` (already installed) | 30 min |
| "How much RAM am I using?" | `psutil.virtual_memory()` (already installed) | 30 min |
| Clipboard read/write | `pyperclip` (pure Python, no native deps) | 1 hour |
| "Search my files for X" (fast) | Windows Search Index via `ADODB.Connection` + `pywin32` | 2 hours |
| Email draft (Outlook) | `win32com.client` Outlook COM — opens compose, doesn't send | 2 hours |
| Calendar event (Outlook) | `win32com.client` Outlook COM — creates event, user confirms | 2 hours |
| Open Settings pages | `start ms-settings:display`, `ms-settings:network`, etc. | 1 hour |
| Empty Downloads folder | `shutil.rmtree` with confirmation (uses existing safety system) | 30 min |
| Dictation mode | Use existing STT → write to clipboard → paste via `pyautogui` | 2 hours |

### Fix (Tasks 3.1-3.7):
Each command follows the same pattern:
1. Create `os_control/{module}_ops.py` with the implementation
2. Add intent + regex/keyword patterns to `command_parser.py` (EN + AR)
3. Add dispatch in `command_router.py`
4. Every command must: work without admin, work without GPU, work without internet, gracefully fail if optional dependency missing

**Timer example** (no external packages):
```python
# os_control/timer_ops.py
import threading, time, winsound

_active_timers = {}

def set_timer(seconds, label="Timer"):
    timer_id = f"timer_{int(time.time())}"
    def _fire():
        for _ in range(5):
            winsound.Beep(1000, 500)
            time.sleep(0.3)
        _active_timers.pop(timer_id, None)
    t = threading.Timer(seconds, _fire)
    t.daemon = True
    t.start()
    _active_timers[timer_id] = {"thread": t, "label": label, "fires_at": time.time() + seconds}
    return f"Timer set: {label} ({seconds}s)"

def cancel_all():
    for tid in list(_active_timers):
        _active_timers[tid]["thread"].cancel()
    _active_timers.clear()
    return "All timers cancelled."
```

**Battery/system info** (zero new dependencies):
```python
# os_control/sysinfo_ops.py
import psutil

def battery_status():
    b = psutil.sensors_battery()
    if not b: return "No battery detected (desktop PC)."
    status = "charging" if b.power_plugged else "on battery"
    return f"Battery: {b.percent}% ({status})"

def system_info():
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory()
    return f"CPU: {cpu}% | RAM: {ram.percent}% ({ram.used//(1024**3)}GB/{ram.total//(1024**3)}GB)"
```

**Parser patterns** to add (examples):
```python
# In command_parser.py EXACT_KEYWORD_ROUTES:
({"set timer", "timer", "حط تايمر", "تايمر"}, "OS_TIMER", "set_timer"),
({"cancel timer", "stop timer", "الغي التايمر", "وقف التايمر"}, "OS_TIMER", "cancel_timer"),
({"battery", "battery status", "البطارية", "البطارية كام", "الشحن كام"}, "OS_SYSTEM_INFO", "battery"),
({"system info", "ram usage", "cpu usage", "معلومات النظام", "الرام", "المعالج"}, "OS_SYSTEM_INFO", "system_info"),
({"clipboard", "what's in clipboard", "الكليب بورد", "اللي متنسخ"}, "OS_CLIPBOARD", "read"),
```

**Files**: `os_control/timer_ops.py`, `os_control/sysinfo_ops.py`, `os_control/clipboard_ops.py`, `os_control/settings_ops.py`, `core/command_parser.py`, `core/command_router.py`
**Priority**: HIGH
**Time**: 2-3 days for all

---

## WEAKNESS 7: Intent parser misses paraphrased commands

**The problem**: The regex parser in `command_parser.py` requires exact phrase matches. "ممكن تفتحلي البرنامج بتاع النت" (can you open the internet program) doesn't match any pattern for opening Chrome. "Make it louder" doesn't match "volume up". Any natural paraphrase fails and falls through to the LLM — adding 2-5 seconds latency.

**How bad it is**: The existing `EXACT_KEYWORD_ROUTES` table has ~80 entries, and the regex `_PATTERN_ROUTES` has ~50 patterns. Together they cover maybe 40-50% of natural speech variations. The other 50-60% falls through to LLM intent extraction.

### Fix (Task 4.1-4.2):
Add a semantic router using multilingual embeddings. This handles paraphrases, code-switching, and natural speech variations — all in <50ms on CPU.

```python
# nlp/semantic_router.py
from semantic_router import Route, RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

routes = [
    Route(name="OS_APP_OPEN", utterances=[
        "open chrome", "launch notepad", "start excel", "open the browser",
        "افتح كروم", "شغل النوت باد", "ممكن تفتح الوورد",
        "افتحلي البرنامج بتاع النت", "شغللي اكسل",
    ]),
    Route(name="OS_VOLUME", utterances=[
        "turn up the volume", "make it louder", "I can't hear", "lower the sound",
        "ارفع الصوت", "خفض الصوت", "مش سامع", "الصوت عالي",
    ]),
    Route(name="OS_TIMER", utterances=[
        "set a timer", "remind me in 5 minutes", "wake me up at 7",
        "حط تايمر", "فكرني بعد خمس دقايق", "صحيني الساعة سبعة",
    ]),
    # ... routes for every intent family
]

router = RouteLayer(encoder=encoder, routes=routes)
```

**Three-tier cascade** in `command_router.py`:
```
Tier 1: Regex parser (0ms) — exact matches
  ↓ no match
Tier 2: Semantic router (~5-10ms) — paraphrase matching
  ↓ low confidence
Tier 3: LLM fallback (2-5s) — conversational queries only
```

**Model size**: `paraphrase-multilingual-MiniLM-L12-v2` is 90MB on disk, ~180MB in RAM. Loads in 2-3s at startup. Classification is <5ms (dot product).

**Files**: `nlp/semantic_router.py` (new), `core/command_router.py`, `requirements.txt`
**Priority**: HIGH
**Time**: 1-2 days

---


## WEAKNESS 9: No hardware auto-detection or auto-scaling

**The problem**: The project uses the same model and context window whether running on an i5 8th gen with 8GB RAM or an i9 with 64GB + RTX 4090. No adaptation.

### Fix (Task 5.2):
```python
# core/hardware_detect.py
import psutil

def detect_hardware_tier():
    ram_gb = psutil.virtual_memory().total / (1024**3)
    # Check GPU via Ollama API
    gpu_available = _check_ollama_gpu()

    if ram_gb >= 16 and gpu_available:
        return {"model": "qwen3:8b", "num_ctx": 8192, "lightweight_ctx": 4096, "tier": "high"}
    elif ram_gb >= 12:
        return {"model": "qwen3:4b", "num_ctx": 4096, "lightweight_ctx": 2048, "tier": "medium"}
    elif ram_gb >= 8:
        return {"model": "qwen3:1.7b", "num_ctx": 2048, "lightweight_ctx": 1024, "tier": "low"}
    else:
        return {"model": "qwen3:0.6b", "num_ctx": 1024, "lightweight_ctx": 512, "tier": "minimal"}

def _check_ollama_gpu():
    try:
        import httpx
        r = httpx.get("http://localhost:11434/api/ps", timeout=2)
        # Ollama reports GPU usage in response
        return "gpu" in r.text.lower()
    except Exception:
        return False
```

Call at startup in `orchestrator.py`, use to set model and context window. Allow `.env` override.

**Files**: `core/hardware_detect.py` (new), `core/orchestrator.py`, `core/config.py`
**Priority**: MEDIUM
**Time**: 2-3 hours

---

## WEAKNESS 10: Over-engineered post-processing pipeline

**The problem**: `_finalize_success_response()` runs 8 sequential text transformations on every response. Most add no value for a voice assistant and some actively hurt quality:

- `_apply_codeswitch_continuity()` — appends "I can switch to العربية anytime" to random responses
- `_apply_tone_adaptation()` — prepends "On it." or "حاضر." even when it sounds unnatural
- `_apply_output_mode()` — truncates to 18 words in "concise" mode (too aggressive for useful answers)
- `_apply_anti_repetition()` — sometimes mangles correct responses

### Fix (Task 1.4):
Simplify to 3 post-processing steps max:
```python
def _finalize_success_response(response_text, parsed, language, original_text, tone_meta, *, realtime=False):
    text = str(response_text or "").strip()
    if _looks_low_value_llm_reply(text) and _is_assist_first_safe_request(original_text):
        text = _fallback_assist_first_response(original_text, language) or text
    text = _apply_egyptian_dialect_style(text, parsed, language)  # keep: TTS needs this
    text = _apply_persona_length_target(text, parsed)             # keep: prevents rambling
    _record_response_quality(text, language, original_text)
    return text
```

**Files**: `core/command_router.py`
**Priority**: MEDIUM
**Time**: 30 minutes

---

## WEAKNESS 11: Streaming callback quality gate causes response corruption

**The problem**: The `_stream_callback` wrapper in `route_command()` inspects each streamed sentence and replaces "low quality" sentences with fallback text mid-stream. This means the user might hear: "Machine learning is..." → [quality gate triggers] → "مش معايا بيانات دلوقتي" — a completely incoherent switch mid-response.

```python
# CURRENT (in route_command):
def _stream_callback(sentence):
    nonlocal stream_quality_repaired
    shaped = _apply_egyptian_dialect_style(sentence, parsed, language)
    if (not stream_quality_repaired
        and _looks_low_value_llm_reply(shaped)
        and _is_assist_first_safe_request(parsed.raw)):
        fallback = _fallback_assist_first_response(parsed.raw, language)
        # ← Replaces a streamed sentence with completely different text!
```

### Fix (Task 1.5):
Either quality-gate the FULL response AFTER streaming completes, or don't quality-gate streaming at all. Mid-stream replacement is never coherent.

```python
# FIXED: Just stream directly, apply style only
def _stream_callback(sentence):
    shaped = _apply_egyptian_dialect_style(sentence, parsed, language)
    stream_callback(shaped)
```

**Files**: `core/command_router.py`
**Priority**: MEDIUM
**Time**: 15 minutes

---

## WEAKNESS 12: Context window too small for real conversations

**Current**: `LLM_OLLAMA_NUM_CTX=2048`, `LLM_LIGHTWEIGHT_NUM_CTX=1024`

With qwen3:4b, the model supports up to 256K context. 2048 tokens is so small that the system prompt + few-shot examples + user query barely fit. There's almost no room for the actual response.

### Fix:
```env
JARVIS_LLM_OLLAMA_NUM_CTX=4096
JARVIS_LLM_LIGHTWEIGHT_NUM_CTX=2048
```

**Files**: `.env.example`, `core/config.py`
**Priority**: MEDIUM
**Time**: 2 minutes

---

## WEAKNESS 13: Auto-pull missing model at startup

**The problem**: If the user doesn't run `ollama pull qwen3:4b`, the project fails silently.

### Fix:
```python
# In core/orchestrator.py, inside _ensure_ollama_running():
def _ensure_model_pulled(model_name):
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if any(model_name in m for m in models):
            return True
        logger.info("Model '%s' not found locally. Pulling...", model_name)
        print(f"[Jarvis] Downloading model '{model_name}'... (first run only)")
        subprocess.run(["ollama", "pull", model_name], check=True)
        return True
    except Exception as e:
        logger.error("Failed to pull model '%s': %s", model_name, e)
        return False
```

**Files**: `core/orchestrator.py`
**Priority**: MEDIUM
**Time**: 30 minutes

---

## EXECUTION ORDER (copy-paste for Claude Code / Codex)

```
PHASE 1 — LLM Fix (do these FIRST, in order):
  1.1 Fix _PINNED_MODEL in ollama_client.py (5 min)
  1.2 Slim system prompt in prompt_builder.py (1-2 hr)
  1.3 Remove triple-rewrite in command_router.py (30 min)
  1.4 Simplify post-processing pipeline (30 min)
  1.5 Fix streaming quality gate (15 min)

PHASE 2 — Real-time Data:
  2.1 Create tools/weather.py with Open-Meteo (1 hr)
  2.2 Create tools/web_search.py with DuckDuckGo (1 hr)
  2.3 Wire tool results into prompt in command_router.py (1-2 hr)

PHASE 3 — Desktop Commands:
  3.1 Timer/alarm — os_control/timer_ops.py (2 hr)
  3.2 Battery/system info — os_control/sysinfo_ops.py (30 min)
  3.3 Clipboard — os_control/clipboard_ops.py (1 hr)
  3.4 Settings pages — os_control/settings_ops.py (1 hr)
  3.5 Windows Search Index — upgrade os_control/file_ops.py (2 hr)
  3.6 Email draft — os_control/email_ops.py (2 hr)
  3.7 Calendar — os_control/calendar_ops.py (2 hr)

PHASE 4 — Wake Word + Intent:
  4.1 Upgrade Arabic wake model from tiny→small (10 min)
  4.2 Add semantic router — nlp/semantic_router.py (1 day)
  4.3 Wire three-tier cascade in command_router.py (3-4 hr)

PHASE 5 — Production Hardening:
  5.1 Hardware auto-detect — core/hardware_detect.py (2-3 hr)
  5.2 Auto-pull model at startup (30 min)
  5.3 Update .env.example, requirements.txt, README.md (1 hr)
```

---

## NEW PACKAGES (all optional except duckduckgo-search)

```
# Required (core)
duckduckgo-search       # Web search, no API key

# Recommended
pyperclip               # Clipboard
screen-brightness-control  # Better brightness
semantic-router         # Intent classification
sentence-transformers   # Multilingual embeddings

# Optional (specific features)
pywin32                 # Outlook email/calendar, Windows Search Index
```

---

## HARDWARE COMPATIBILITY (after all fixes)

| Spec | LLM | Speed | All features? |
|------|-----|-------|---------------|
| i5 8th gen, 8GB, no GPU | qwen3:1.7b | ~25 tok/s | Yes |
| i5 10th gen, 12GB, no GPU | qwen3:4b | ~12 tok/s | Yes |
| i7 13th gen, 16GB, no GPU | qwen3:4b | ~15-18 tok/s | Yes |
| i7 13th gen, 16GB, GPU | qwen3:8b | ~40-50 tok/s | Yes |
| Any, 8GB+, with/without GPU | Auto-selected | Auto-scaled | Yes |

**Total RAM footprint**: ~3-4 GB on 16GB machine (Python + Whisper + Ollama + semantic router)
