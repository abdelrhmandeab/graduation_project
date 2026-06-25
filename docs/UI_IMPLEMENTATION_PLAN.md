# Jarvis Desktop UI — Full Implementation Plan

## 1. Product vision

A floating animated avatar for Jarvis, a local-first bilingual Windows voice assistant for English and Egyptian Arabic. The avatar lives hidden at the screen edge and **rises when the wake word is spoken**. It listens, processes, responds via TTS, then **sinks back down and disappears**. A persistent **system tray icon** provides access to settings, diagnostics, theme controls, avatar direction selection, and a text chat panel as an alternative to voice. The engine (Python) and UI (Tauri + React) are separate processes connected by a local WebSocket.

The UI implementation scope is intentionally narrow: build the real application overlay, tray panels, and theme system only. Do not build a fake desktop, mock desktop wallpaper, virtual display, or presentation-style environment around the assistant.

### Questions answered / current design brief

- **Idea**: Jarvis is an ambient assistant presence, not a conventional app window. The primary UI is a premium glass avatar that appears only when needed, reflects the assistant state, then vanishes.
- **Platform**: Windows desktop overlay using Tauri + WebView2. The avatar window is frameless, transparent, always on top, and positioned near the bottom-right screen edge. The system tray is the permanent UI anchor.
- **Design identity**: Quiet futurism with iOS-like translucent glass: low-opacity shells, radial rim light, layered rim strokes, subtle cyan caustics, specular highlights, and restrained state glow. Avoid gradient-heavy AI decoration, emoji, decorative dashboards, and noisy sci-fi panels.
- **Avatar directions**: Build four selectable directions before choosing the production identity:
  - Direction A — Aurora glass sphere.
  - Direction B — technical glyph ring.
  - Direction C — glass AI intelligence core.
  - Direction D — friendly companion face tile.
- **State language**: The seven assistant states remain shared across every direction: IDLE/hidden, LISTENING, PROCESSING, RESPONDING, CONFIRMING, EXECUTING, and FOLLOW_UP. Each direction decides where the state color appears.
- **Fidelity**: High-fidelity and interactive application UI. Implement the rise/dismiss animation, state transitions, live transcript placement, Canvas 2D waveform/listening visualization, tray panels, and themes as real app components.
- **Scope**: Focus on the application only: avatar overlay, system tray menu, settings panel, diagnostics panel, text chat/command panel, and theme/avatar-direction controls.
- **Out of scope**: No mock desktop, virtual monitor, fake OS environment, desktop wallpaper composition, or prototype-only display frame.
- **References**: macOS Siri’s ambient orb, Windows Copilot’s system-level presence, Nothing Phone glyph lights, and Iron Man’s Jarvis as a room-like assistant presence — interpreted through the current premium glass identity.
- **Tuning targets**: Rise/dismiss timing and easing, avatar size, glass transparency, glow radius, screen-edge rise distance, transcript positioning, state transitions, tray panel density, and theme contrast.

### Hard constraints

- Runs on **8GB RAM, no GPU**, Windows 10/11.
- UI adds **< 130 MB RAM** at runtime (WebView2 reuse).
- Animations stay on the **GPU-composited fast path** (`transform` + `opacity` only).
- **Canvas 2D** for waveform — no WebGL/Three.js (no GPU fallback risk).
- Engine works without UI. UI live behavior expects the local bridge; disconnected states are handled as real app states, not as a simulated desktop.

---

## 2. Architecture

```
┌──────────────────────────┐                              ┌──────────────────────────┐
│    Python voice engine   │    WebSocket (localhost)      │   Tauri + React app      │
│    (orchestrator.py)     │  ─────── events ──────►      │   (desktop/)             │
│                          │  ◄────── commands ─────      │                          │
│    ui/bridge.py runs as  │      JSON over WS            │   frameless, transparent,│
│    daemon thread inside  │                              │   always-on-top window   │
│    the engine process    │                              │                          │
└──────────────────────────┘                              └──────────────────────────┘
          ▲                                                          ▲
          │                                                          │
   Python process                                            Tauri process
   (main.py)                                          (system WebView2 runtime)
```

### Three layers

| Layer | Role | Language | Process |
|-------|------|----------|---------|
| **Engine** | Voice pipeline, NLU, LLM, OS control (existing, unchanged) | Python | Main process |
| **Bridge** (`ui/bridge.py`) | WebSocket server, event bus, command receiver (~150 LOC) | Python | Daemon thread inside engine |
| **Desktop** (`desktop/`) | Avatar overlay, tray panels, text chat | React + TypeScript | Tauri window (separate OS process) |

---

## 3. Project structure

```
graduation_project/
│
│   # ── Existing Python engine (untouched) ─────────────────────
├── core/
│   ├── orchestrator.py          # main loop — add event emission hooks
│   ├── command_router.py        # route_command — text_command calls this
│   ├── command_parser.py        # parse_command
│   ├── dialogue_manager.py      # register_state_listener() — already exists
│   ├── config.py                # add UI_BRIDGE_PORT, UI_BRIDGE_ENABLED
│   ├── metrics.py               # expose stage timings to bridge
│   ├── doctor.py                # expose health checks to bridge
│   ├── hardware_detect.py
│   ├── session_memory.py
│   ├── knowledge_base.py
│   └── ...
├── audio/
├── llm/
├── nlp/
├── os_control/
├── tools/
├── utils/
│
│   # ── Bridge layer (NEW — the integration seam) ──────────────
├── ui/
│   ├── tray.py                  # existing system tray — add panel launch items
│   ├── bridge.py                # NEW — WebSocket server + event bus
│   └── events.py                # NEW — event type definitions + serialization
│
│   # ── Desktop UI (NEW — entirely separate toolchain) ─────────
├── desktop/
│   ├── package.json             # Node dependencies
│   ├── vite.config.ts           # Vite config
│   ├── tsconfig.json            # TypeScript config
│   ├── tailwind.config.ts       # Tailwind config
│   ├── index.html               # Vite entry
│   │
│   ├── src/
│   │   ├── App.tsx              # root — window routing (avatar vs tray panel)
│   │   ├── main.tsx             # React entry
│   │   │
│   │   ├── components/
│   │   │   ├── avatar/
│   │   │   │   ├── Avatar.tsx           # direction switcher + shared lifecycle
│   │   │   │   ├── AuroraAvatar.tsx     # Direction A — glass sphere + waveform field
│   │   │   │   ├── GlyphAvatar.tsx      # Direction B — luminous technical glyph ring
│   │   │   │   ├── GlassAIAvatar.tsx    # Direction C — frosted glass tile + light core
│   │   │   │   ├── CompanionAvatar.tsx  # Direction D — friendly glass face tile
│   │   │   │   ├── GlassSurface.tsx     # reusable premium iOS-style glass shell/rims
│   │   │   │   ├── Waveform.tsx         # Canvas 2D audio visualizer
│   │   │   │   └── Transcript.tsx       # floating partial + final text near avatar
│   │   │   │
│   │   │   ├── chat/
│   │   │   │   ├── ChatPanel.tsx        # text command alternative to voice
│   │   │   │   ├── ChatBubble.tsx       # single message (user / jarvis)
│   │   │   │   └── ChatInput.tsx        # text input + send button
│   │   │   │
│   │   │   ├── settings/
│   │   │   │   ├── SettingsPanel.tsx     # top-level settings container
│   │   │   │   ├── FeatureFlags.tsx     # toggle switches for 4 feature flags
│   │   │   │   ├── AvatarDirection.tsx  # selects Direction A/B/C/D
│   │   │   │   ├── ThemeSettings.tsx    # premium glass theme controls
│   │   │   │   ├── ModelSelector.tsx    # Qwen3 model tier picker
│   │   │   │   ├── VoiceSettings.tsx    # STT/TTS backend + voice selection
│   │   │   │   ├── AudioSettings.tsx    # VAD thresholds, silence durations
│   │   │   │   └── WakeWordSettings.tsx # mode (en/ar/both), thresholds
│   │   │   │
│   │   │   └── diagnostics/
│   │   │       ├── DiagnosticsPanel.tsx  # top-level diagnostics container
│   │   │       ├── LatencyChart.tsx      # stage timing waterfall
│   │   │       ├── HealthChecks.tsx      # doctor.py 26/26 check list
│   │   │       └── SessionStats.tsx      # commands count, avg latency, uptime
│   │   │
│   │   ├── hooks/
│   │   │   ├── useJarvisSocket.ts       # WebSocket client — connect, reconnect, parse
│   │   │   ├── useAmplitude.ts          # throttled amplitude → waveform data
│   │   │   ├── useTheme.ts             # reads theme from store, applies CSS custom properties
│   │   │   └── useRTL.ts               # per-message RTL/LTR detection
│   │   │
│   │   ├── stores/
│   │   │   └── jarvisStore.ts           # zustand — dialogue state, messages, avatar direction, theme (persisted via zustand persist middleware)
│   │   │
│   │   ├── styles/
│   │   │   ├── globals.css              # Tailwind base + dark theme tokens
│   │   │   ├── themes.css               # premium glass theme variables
│   │   │   └── animations.css           # avatar keyframes (transform + opacity only)
│   │   │
│   │   ├── lib/
│   │   │   ├── protocol.ts             # message type definitions (mirrors events.py)
│   │   │   └── constants.ts            # state colors, animation timings
│   │   │
│   │   └── types/
│   │       └── index.ts                # shared TypeScript interfaces
│   │
│   ├── src-tauri/
│   │   ├── Cargo.toml                  # Tauri Rust dependencies
│   │   ├── tauri.conf.json             # window: frameless, transparent, always-on-top
│   │   ├── src/
│   │   │   └── main.rs                 # Tauri entry + window management commands
│   │   └── icons/                      # app icons for tray + taskbar
│   │
│   ├── tests/
│   │   ├── unit/                       # Vitest unit tests
│   │   │   ├── jarvisStore.test.ts
│   │   │   ├── protocol.test.ts
│   │   │   └── useJarvisSocket.test.ts
│   │   └── e2e/                        # Playwright e2e tests
│   │       ├── avatar.spec.ts
│   │       └── chat.spec.ts
│   │
│   └── .gitignore                      # node_modules, dist, src-tauri/target
│
│   # ── Root files ─────────────────────────────────────────────
├── main.py                     # add bridge startup
├── requirements.txt            # add fastapi, uvicorn, websockets
├── .env.example                # add JARVIS_UI_BRIDGE_* vars
└── .gitignore                  # add desktop/node_modules, desktop/dist, etc.
```

---

## 4. Technology stack

### 4.1 Python side (bridge)

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | >= 0.115.0 | WebSocket endpoint + optional REST health route |
| `uvicorn[standard]` | >= 0.34.0 | ASGI server running in daemon thread |
| `websockets` | >= 15.0 | WebSocket transport (uvicorn dependency) |

Added to `requirements.txt`. No new system-level installs.

### 4.2 Desktop app

| Tool / Library | Version | Purpose |
|----------------|---------|---------|
| **Tauri** | 2.x | Native desktop shell — uses system WebView2, ~3–10 MB bundle |
| **React** | 18.x | UI framework |
| **TypeScript** | 5.x | Type safety |
| **Vite** | 6.x | Dev server + bundler (HMR, fast builds) |
| **Tailwind CSS** | 4.x | Utility-first styling, dark theme |
| **Framer Motion** | 12.x | Declarative animations (restricted to transform + opacity) |
| **Zustand** | 5.x | Lightweight state management (~1 KB) |
| **Vitest** | 3.x | Unit testing (Vite-native) |
| **Playwright** | 1.x | End-to-end testing |

All declared in `desktop/package.json`. Fully isolated from Python dependencies.

### 4.3 Build-time prerequisites

| Tool | Version | Install | Needed for |
|------|---------|---------|------------|
| **Node.js** | >= 18 | `winget install OpenJS.NodeJS.LTS` | npm, Vite, Tauri CLI |
| **Rust** | stable | `winget install Rustlang.Rust.MSVC` | Tauri build (not needed for `npm run dev`) |
| **WebView2** | runtime | Pre-installed on Win11; auto-installed on Win10 | Tauri rendering engine |

### 4.4 VS Code extensions (recommended)

| Extension | ID | Purpose |
|-----------|----|---------|
| Tauri | `tauri-apps.tauri-vscode` | Tauri project support, build tasks |
| ES7+ React snippets | `dsznajder.es7-react-js-snippets` | React component scaffolding |
| Tailwind Intellisense | `bradlc.vscode-tailwindcss` | Tailwind class autocomplete |
| Prettier | `esbenp.prettier-vscode` | Code formatting (TS/TSX) |
| ESLint | `dbaeumer.vscode-eslint` | Linting |
| Auto Rename Tag | `formulahendry.auto-rename-tag` | HTML/JSX tag sync |
| PostCSS Language Support | `csstools.postcss` | Tailwind `@apply` syntax |
| Error Lens | `usernamehw.errorlens` | Inline error display |
| Rust Analyzer | `rust-lang.rust-analyzer` | Tauri Rust backend (optional) |

---

## 5. Bridge protocol

### 5.1 Connection

```
URL:    ws://localhost:{JARVIS_UI_BRIDGE_PORT}/ws
Port:   default 9720 (configurable via .env)
```

Auto-reconnect with exponential backoff (1s → 2s → 4s → max 30s). Heartbeat ping every 15s.

### 5.2 Engine → UI (events)

All messages are JSON with a `type` field:

```jsonc
// State change — drives avatar color + animation
{ "type": "state_changed", "state": "listening", "previous": "idle" }

// Partial transcript — live STT text, emitted ~10 Hz
{ "type": "partial_transcript", "text": "open the...", "language": "en" }

// Final transcript — completed utterance
{ "type": "final_transcript", "text": "open the browser", "language": "en" }

// Response — assistant reply
{ "type": "response", "text": "Opening Chrome.", "language": "en", "intent": "OS_APP_OPEN" }

// Amplitude — mic level for waveform, emitted ~15 Hz
{ "type": "amplitude", "level": 0.73 }

// Metrics — latency stage timings
{ "type": "metrics", "stages": [
    { "name": "stt", "ms": 320 },
    { "name": "parse", "ms": 2 },
    { "name": "route", "ms": 45 },
    { "name": "tts", "ms": 180 }
]}

// Doctor health — periodic health check results
{ "type": "health", "checks": [
    { "name": "ollama", "status": "ok" },
    { "name": "microphone", "status": "ok" },
    { "name": "whisper_model", "status": "degraded", "detail": "model not loaded" }
]}

// Error — runtime errors for UI display
{ "type": "error", "message": "Ollama connection failed", "recoverable": true }

// Config snapshot — current settings (sent on connect + on change)
{ "type": "config", "values": {
    "model": "qwen3:4b",
    "model_tier": "medium",
    "wake_mode": "both",
    "feature_flags": {
        "NUMERIC_PARSING_ENABLED": true,
        "AUTO_APP_DISCOVERY_ENABLED": true,
        "MEDIA_DIRECT_DISPATCH_ENABLED": true,
        "SYSTEM_VOLUME_CONTROL": true
    },
    "stt_backend": "hybrid_elevenlabs",
    "tts_backend": "hybrid",
    "persona": "friendly"
}}
```

### 5.3 UI → Engine (commands)

```jsonc
// Text command — bypasses STT, sent to route_command()
{ "type": "text_command", "text": "open chrome", "language": "en" }

// Mute toggle
{ "type": "mute_toggle", "muted": true }

// Setting update — change a config value at runtime
{ "type": "setting_update", "key": "JARVIS_LLM_MODEL", "value": "qwen3:8b" }

// Feature flag toggle
{ "type": "feature_flag", "flag": "NUMERIC_PARSING_ENABLED", "enabled": false }

// Request config snapshot
{ "type": "config_request" }

// Request health check
{ "type": "health_request" }
```

---

## 6. Avatar behavior

### 6.1 Window configuration (Tauri)

```jsonc
// src-tauri/tauri.conf.json (relevant window settings)
{
  "app": {
    "windows": [{
      "label": "avatar",
      "title": "Jarvis",
      "width": 280,
      "height": 400,
      "x": null,           // positioned programmatically (bottom-right)
      "y": null,
      "decorations": false, // frameless
      "transparent": true,  // see-through background
      "alwaysOnTop": true,
      "skipTaskbar": true,  // no taskbar entry
      "resizable": false,
      "visible": false      // starts hidden, shown on wake word
    }]
  }
}
```

### 6.2 Rise / dismiss flow

```
[hidden]
    │
    ├── wake word detected (state_changed → "listening")
    │
    ▼
[rise animation: translateY(100%) → translateY(0), 300ms ease-out]
    │
    ├── LISTENING  — green state light, amplitude-reactive pulse, waveform visible
    ├── PROCESSING — amber state light, searching/swirling animation
    ├── RESPONDING — blue state light, smooth wave, response text appears
    │
    ├── state_changed → "follow_up"
    │   └── FOLLOW_UP — teal state light, gentle shimmer, stays visible for 10s
    │
    ├── state_changed → "idle" (follow_up expired or no follow-up)
    │
    ▼
[dismiss animation: translateY(0) → translateY(100%), 400ms ease-in]
    │
    ▼
[hidden — window invisible until next wake word]
```

### 6.3 State-driven avatar

Colors inherited from existing `ui/tray.py` state mapping:

| State | Color | Hex | Behavior (direction-interpreted) | Intensity |
|-------|-------|-----|-----------------------------------|-----------|
| IDLE | Grey | `#808080` | Slow breath — calm, minimal motion | Low |
| LISTENING | Green | `#00B400` | Amplitude-reactive pulse + waveform active | High (rAF) |
| PROCESSING | Amber | `#FFC800` | Searching / indeterminate motion | Medium |
| RESPONDING | Blue | `#0078FF` | Smooth rhythmic wave | Medium |
| CONFIRMING | Orange | `#FF8C00` | Attention pulse — draws user focus | Medium-high |
| EXECUTING | Indigo | `#5A5AC8` | Steady glow — no motion | Static |
| FOLLOW_UP | Teal | `#00A078` | Gentle shimmer — open-listening feel | Low |

Each direction component translates these behaviors into its own visual language:
- **Aurora**: sphere scale/opacity pulse, waveform ring amplitude.
- **Glyph**: ring stroke dash-offset rotation, core dot scale, particle orbit speed.
- **Glass AI**: internal core brightness, circuit node pulse pattern, shell rim glow opacity.
- **Companion**: face glow intensity, sparkle animation speed, status dot pulse, inner panel opacity.

The selected direction determines where the state color appears:

- **Direction A — Aurora**: the sphere body and waveform rings inherit the state color.
- **Direction B — Glyph**: the ring stroke, dot-core, and dashed inner ring inherit the state color.
- **Direction C — Glass AI**: the glass shell stays ice-blue; the internal lightbulb core and circuit nodes inherit the state color.
- **Direction D — Companion Face**: the glass shell stays ice-blue; the face glow, sparkle, status dots, and inner panel inherit the state color.

### 6.4 Premium glass treatment

All four directions use the same premium glass language, tuned to feel closer to an iOS-style translucent material than a flat neon mascot.

- **Transparent shell**: shell fill opacity targets ~17%, so the avatar feels airy and see-through rather than milky or plastic.
- **Radial rim light**: a tight top-left radial highlight creates the main light catch, fading naturally across the form.
- **Layered physical edge**: two concentric rim strokes create perceived glass thickness — a bright outer stroke around 1.3px and a softer inner stroke around 0.7px.
- **Cold-glass accent**: electric cyan `#8EEBFF` appears only on edge catches, caustics, and small state accents, not as a full neon fill.
- **Bottom caustic band**: a subtle lower-edge gradient simulates refracted light through the base of the glass.
- **Double specular streak**: each glass tile or sphere can use a broad soft white streak plus a thinner offset streak to suggest multiple studio lights.
- **Corner micro-catch**: square/tile directions add a small bloom ellipse at the top-left corner as the brightest specular point.
- **Physical shadow**: avoid CSS `drop-shadow` on the whole avatar; prefer two static SVG shadow layers — a close soft shadow and a farther cool blue-slate shadow.
- **Face clarity**: Direction D uses slightly thicker eye arcs around 2.9px, wider eye bloom, and tiny reflection dots so the face remains readable on transparent glass.

### 6.5 Animation rules (no-GPU safety)

- **Only animate** `transform` and `opacity` — these are GPU-composited and skip layout/paint.
- **Canvas 2D** for waveform — draw ~60 bars from amplitude data, one `requestAnimationFrame` loop.
- **Frame rate**: 30 fps while IDLE (if visible), 60 fps while LISTENING / RESPONDING.
- **Pause** animation loop when `document.visibilitychange` fires `hidden`.
- **Respect** `prefers-reduced-motion: reduce` — disable all animation, show static avatar.
- **No animated `box-shadow`** — use separate static SVG glow layers and animate only their `opacity`.
- **No animated `filter: blur()`** — it triggers CPU paint on no-GPU machines. Static SVG filters are allowed for glass shadows and specular bloom if their inputs do not animate.

---

## 7. System tray panels

The existing `ui/tray.py` (pystray) tray icon gains new menu items that open Tauri-managed panel windows.

### 7.1 Settings panel

Surfaces real config values from `core/config.py` (currently edited via `.env`):

| Section | Controls | Config keys |
|---------|----------|-------------|
| **Avatar** | Direction selector (Aurora / Glyph / Glass AI / Companion) | UI-local setting in `jarvisStore` |
| **Theme** | Premium glass theme selector, transparency and contrast presets | UI-local setting in `jarvisStore` |
| **Feature flags** | 4 toggle switches | `JARVIS_FEATURE_NUMERIC_PARSING_ENABLED`, `JARVIS_FEATURE_AUTO_APP_DISCOVERY_ENABLED`, `JARVIS_FEATURE_MEDIA_DIRECT_DISPATCH_ENABLED`, `JARVIS_FEATURE_SYSTEM_VOLUME_CONTROL` |
| **Model** | Dropdown (auto / 0.6b / 1.7b / 4b / 8b) | `JARVIS_LLM_MODEL`, `JARVIS_LLM_AUTO_SELECT` |
| **STT** | Backend selector, language hint | `JARVIS_STT_BACKEND`, `JARVIS_STT_LANGUAGE_HINT`, `JARVIS_WHISPER_MODEL` |
| **TTS** | Backend, voice, Arabic dialect | `JARVIS_TTS_BACKEND`, `JARVIS_TTS_EDGE_VOICE`, `JARVIS_TTS_ARABIC_SPOKEN_DIALECT` |
| **Audio / VAD** | Sliders for thresholds | `JARVIS_VAD_ENERGY_THRESHOLD`, `JARVIS_VAD_COMMAND_SILENCE_SECONDS`, `JARVIS_VAD_CHAT_SILENCE_SECONDS`, `JARVIS_MAX_RECORD_DURATION` |
| **Wake word** | Mode selector (en/ar/both), threshold sliders | `JARVIS_WAKE_MODE`, `JARVIS_WAKE_WORD_EN_THRESHOLD`, `JARVIS_WAKE_WORD_AR_THRESHOLD` |
| **Persona** | Style selector | `JARVIS_PERSONA_DEFAULT` (friendly/formal/casual/brief/professional) |
| **Greeting** | Toggle, language, custom text | `JARVIS_GREETING_ENABLED`, `JARVIS_GREETING_LANGUAGE`, `JARVIS_GREETING_TEXT_EN`, `JARVIS_GREETING_TEXT_AR` |

### 7.2 Diagnostics panel

- **Latency waterfall**: bar chart of `core/metrics.py` stage timings (wake → STT → parse → route → TTS).
- **Health checks**: `core/doctor.py` results — 26 checks rendered as a status list (pass/degraded/fail).
- **Session stats**: commands processed, average latency, uptime, current model, RAM usage.

### 7.3 Chat panel (text command mode)

- Scrolling conversation history with **user** and **Jarvis** bubbles.
- Text input at bottom — sends `text_command` via bridge → `route_command()`.
- **Bilingual**: RTL layout flip for Arabic messages (per-message, using existing language detection).
- Works with engine running but microphone muted or unavailable.

---

## 8. Development workflows

| Workflow | Command(s) | What runs |
|----------|-----------|-----------|
| **App UI shell** | `cd desktop && npm run dev` | Vite dev server for the real Tauri/React UI. Shows disconnected state when bridge is not running; live data streams when bridge is active. |
| **Full stack** | `python main.py` then `cd desktop && npm run dev` | Engine + bridge + UI with Vite HMR. |
| **Engine only** | `python main.py` | Voice assistant runs as today. Bridge starts but UI is optional. |
| **UI unit tests** | `cd desktop && npm test` | Vitest — tests stores, hooks, protocol parsing. |
| **UI e2e tests** | `cd desktop && npm run test:e2e` | Playwright — tests avatar rise/dismiss, chat flow. |
| **Type check** | `cd desktop && npm run typecheck` | `tsc --noEmit` across all `.ts`/`.tsx` files. |
| **Lint** | `cd desktop && npm run lint` | ESLint + Prettier. |
| **Production build** | `cd desktop && npm run tauri build` | Bundles React into native `.exe` with Tauri. |
| **Bridge only** | `python -m ui.bridge` | Starts bridge standalone for testing with `wscat`. |

No mock desktop, virtual display, or simulated OS environment is part of the build. Development should happen against the real app windows and the local bridge/disconnected-state handling.

---

## 9. Environment variables (new)

Add to `.env.example`:

```env
# UI Bridge (WebSocket server for desktop UI)
JARVIS_UI_BRIDGE_ENABLED=true
JARVIS_UI_BRIDGE_PORT=9720
JARVIS_UI_BRIDGE_HOST=127.0.0.1
```

Add to `core/config.py`:

```python
UI_BRIDGE_ENABLED = _env_bool("JARVIS_UI_BRIDGE_ENABLED", True)
UI_BRIDGE_PORT = _env_int("JARVIS_UI_BRIDGE_PORT", 9720)
UI_BRIDGE_HOST = _env("JARVIS_UI_BRIDGE_HOST", "127.0.0.1")
```

---

## 10. Integration points with existing code

| File | Change | Risk |
|------|--------|------|
| `core/config.py` | Add `UI_BRIDGE_ENABLED`, `UI_BRIDGE_PORT`, `UI_BRIDGE_HOST` | None — additive |
| `core/orchestrator.py` | Emit events to bridge: `state_changed`, `partial_transcript`, `final_transcript`, `response`, `amplitude` | Low — add calls after existing logic, no behavior change |
| `core/dialogue_manager.py` | None — `register_state_listener()` already exists at line 81 | None |
| `core/metrics.py` | Add public method to export stage timings as dict | Low — read-only accessor |
| `core/doctor.py` | Add public method to export check results as dict | Low — read-only accessor |
| `ui/tray.py` | Add menu items: "Open Chat", "Settings", "Diagnostics" | Low — extends existing menu |
| `main.py` | Start bridge daemon thread before `run()` | Low — conditional on `UI_BRIDGE_ENABLED` |
| `.env.example` | Add 3 new vars (`JARVIS_UI_BRIDGE_*`) | None |
| `requirements.txt` | Add `fastapi`, `uvicorn[standard]`, `websockets` | Low — optional, import-guarded |
| `.gitignore` | Add `desktop/node_modules/`, `desktop/dist/`, `desktop/src-tauri/target/` | None |

---

## 11. Visual system

| Property | Value |
|----------|-------|
| **Theme** | Premium glass dark theme by default (near-black base `#0A0A0F`), with UI-local contrast/transparency presets. |
| **Accent** | Single dynamic accent = avatar's current state color. Secondary glass tint is limited to `#8EEBFF` edge/caustic details. |
| **Surfaces** | Tray panels: `#0A0A0F` at 92% opacity. Chat bubbles: `#1A1A24` (user) / `#12121A` (jarvis). Avatar shells use ~17% translucent mist-blue fill with layered rim strokes. |
| **Text** | Primary `#E8E8EC`, secondary `#8888A0`, muted `#55556A`. |
| **Typography** | UI: Geist Sans (or Outfit). Mono: Geist Mono (or JetBrains Mono). Avoid Inter/Roboto. |
| **Border radius** | Aurora: circle. Glyph: circle/ring. Glass AI + Companion: rounded-square glass tile. Panels: 12px. Buttons: 8px. Chat bubbles: 16px. |
| **Shadows** | None on panels (dark theme). Avatar shadow uses static dual-layer SVG shadow; glow visibility changes via `opacity`. |
| **Density** | Compact — minimal padding, no decorative whitespace. |
| **RTL** | Arabic messages and settings labels flip to RTL via `dir="rtl"` per-element. |
| **Motion** | All via `transform` + `opacity`. No `box-shadow` animation, no `filter: blur()`. |
| **Reduced motion** | `@media (prefers-reduced-motion: reduce)` — static avatar, no transitions. |
| **Display scope** | Render only app UI surfaces: avatar overlay, tray panels, chat, settings, diagnostics, and themes. Do not render a fake desktop, monitor frame, virtual display, or wallpaper scene. |

---

## 12. Implementation phases

### Phase 1 — Bridge (Python) — ~2 days

- [ ] Create `ui/events.py` — event type enum + `to_json()` serialization.
- [ ] Create `ui/bridge.py` — FastAPI app with WebSocket endpoint, client registry, broadcast helper.
- [ ] Subscribe to `dialogue_manager.register_state_listener()` for state events.
- [ ] Add event emission hooks in `orchestrator.py` for: partial transcript, final transcript, response, amplitude.
- [ ] Add `config_request` and `health_request` command handlers.
- [ ] Add `text_command` handler that calls `route_command()` on a worker thread.
- [ ] Add config vars to `core/config.py` and `.env.example`.
- [ ] Wire bridge startup into `main.py` (conditional on `UI_BRIDGE_ENABLED`).
- [ ] Test with `wscat -c ws://localhost:9720/ws` — verify events stream during normal voice use.

**Deliverable**: `python main.py` starts engine + bridge; any WebSocket client receives live events.

### Phase 2 — Avatar (Tauri + React) — ~4 days

- [ ] Scaffold Tauri + React + Vite + TypeScript project in `desktop/`.
- [ ] Configure `tauri.conf.json` for frameless, transparent, always-on-top, skip-taskbar window.
- [ ] Install + configure Tailwind CSS, Framer Motion, Zustand.
- [ ] Build `useJarvisSocket` hook — WebSocket client with auto-reconnect + message parsing.
- [ ] Build `jarvisStore` — Zustand store for dialogue state, messages, amplitude, config.
- [ ] Build `Avatar.tsx` — shared lifecycle wrapper and selected direction switcher.
- [ ] Build shared `GlassSurface.tsx` primitives for shell fill, radial rim light, concentric rim strokes, caustic band, specular streaks, corner micro-catch, and static SVG shadow layers.
- [ ] Build `AuroraAvatar.tsx` — premium glass sphere with state-colored waveform field.
- [ ] Build `GlyphAvatar.tsx` — luminous ring, dot-core, dashed inner ring, and technical matrix details.
- [ ] Build `GlassAIAvatar.tsx` — rounded glass tile with internal lightbulb core, circuit paths, and 16 state-reactive nodes.
- [ ] Build `CompanionAvatar.tsx` — rounded glass tile with inner face panel, thickened eye arcs, eye bloom, reflection dots, sparkle, and status dots.
- [ ] Build `Waveform.tsx` — Canvas 2D bar visualizer driven by amplitude events.
- [ ] Build `Transcript.tsx` — floating text near avatar showing partial → final transcript + response.
- [ ] Implement rise/dismiss animation (`translateY`) triggered by state changes.
- [ ] Build disconnected-state UI for when the local bridge is unavailable.
- [ ] Build `themes.css` for premium glass theme tokens and contrast presets.
- [ ] Add `constants.ts` with state colors and animation timings.

**Deliverable**: wake word triggers the selected avatar direction to rise, animate through states, show transcript, and dismiss.

### Phase 3 — Tray panels — ~3 days

- [ ] Build `ChatPanel.tsx` + `ChatBubble.tsx` + `ChatInput.tsx`.
- [ ] Wire `text_command` from chat input → bridge → `route_command()`.
- [ ] Build `SettingsPanel.tsx` with all sub-components (AvatarDirection, ThemeSettings, FeatureFlags, ModelSelector, VoiceSettings, AudioSettings, WakeWordSettings).
- [ ] Wire `setting_update` and `feature_flag` commands.
- [ ] Build `DiagnosticsPanel.tsx` + `LatencyChart.tsx` + `HealthChecks.tsx` + `SessionStats.tsx`.
- [ ] Implement per-message RTL detection in `useRTL.ts`.
- [ ] Add panel-launching menu items to `ui/tray.py`.
- [ ] Create Tauri multi-window setup (avatar window + panel windows).

**Deliverable**: right-click tray opens settings/diagnostics; text chat works as voice alternative.

### Phase 4 — Polish and packaging — ~2 days

- [ ] Tune rise/dismiss easing curves and timing.
- [ ] Tune glass transparency, rim highlights, caustic band, specular streaks, state glow radius, pulse amplitude, and color transitions.
- [ ] Add `prefers-reduced-motion` support across all animations.
- [ ] Add window positioning logic (bottom-right of primary monitor, above taskbar).
- [ ] Add `document.visibilitychange` frame-rate throttling.
- [ ] Configure Tauri build for single `.exe` (NSIS installer).
- [ ] Add auto-launch: Tauri process starts Python engine as child process.
- [ ] Write unit tests (Vitest) for store, protocol, hooks.
- [ ] Write e2e tests (Playwright) for avatar lifecycle and chat flow.

**Deliverable**: `npm run tauri build` produces `Jarvis-Setup.exe` that installs and runs both engine and UI.

---

## 13. Design directions (4 planned)

Four visual directions will be designed for the avatar + chat experience. All share the same architecture, bridge protocol, state colors, and premium translucent-glass material system above. The directions differ in silhouette, emotional tone, and where the current state color is expressed.

### Direction A — Aurora Form & Silhouette

- Perfect sphere with no edges or corners.
- Best for an ambient, calm AI presence.
- State color lives directly in the orb body, rim glow, and waveform rings.
- Uses Canvas 2D for waveform rings and soft audio-reactive ripples.

### Direction B — Glyph / Technical Ring

- Hollow luminous ring with a dot-core, dashed inner ring, and matrix-like micro details.
- Best for a power-user or technical identity.
- State color lives in strokes, dash segments, core dot, and status particles.
- Keeps the center visually open so it feels like an interface marker rather than a face.

### Direction C — Glass AI / Intelligence Core

- Rounded-square frosted glass tile with an internal lightbulb/intelligence core.
- Best for a product identity that reads as “AI assistant” immediately.
- The shell remains ice-blue and transparent; the internal core, circuit paths, and 16 nodes carry the state color.
- Premium treatment emphasizes layered rim strokes, bottom caustic band, and double specular streaks.

### Direction D — Friendly AI / Companion Face

- Rounded-square frosted glass tile with an inner face panel, smiling eye arcs, small mouth, sparkle, and status dots.
- Best for a warm desktop companion identity.
- The shell remains ice-blue and transparent; the face glow, sparkle, inner panel, and status dots carry the state color.
- Face details must stay minimal and non-human: 2.9px eye arcs, widened eye bloom, tiny reflection dots, and no realistic facial anatomy.

After review, one direction is selected and refined into the production UI.

---

## 14. Appendix: CLI quick-start

### First-time setup

```powershell
# 1. Install Node.js (if not already)
winget install OpenJS.NodeJS.LTS

# 2. Install Rust (only needed for production build, not dev)
winget install Rustlang.Rust.MSVC

# 3. Install Python bridge dependencies
pip install fastapi uvicorn[standard] websockets

# 4. Scaffold desktop app
cd desktop
npm install

# 5. Run in development mode
# Terminal 1: engine + bridge
python main.py

# Terminal 2: UI with hot-reload
cd desktop
npm run dev
```

### Key npm scripts (desktop/package.json)

```jsonc
{
  "scripts": {
    "dev": "vite",                           // dev server + HMR
    "build": "tsc && vite build",            // production web build
    "preview": "vite preview",               // preview production build
    "tauri": "tauri",                         // Tauri CLI passthrough
    "tauri dev": "tauri dev",                // Tauri dev (with native window)
    "tauri build": "tauri build",            // build .exe installer
    "test": "vitest run",                    // unit tests
    "test:watch": "vitest",                  // unit tests in watch mode
    "test:e2e": "playwright test",           // e2e tests
    "typecheck": "tsc --noEmit",             // type check only
    "lint": "eslint src/ --ext .ts,.tsx",    // lint
    "format": "prettier --write src/"        // format
  }
}
```
