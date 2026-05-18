# Jarvis Production Bug & Enhancement Plan
**Date:** May 17, 2026  
**Status:** Active  
**Priority Phases:** P0 (Critical), P1 (High), P2 (Medium), P3 (Nice-to-Have)  
**Language Coverage:** ✅ English + Egyptian Arabic (عامية مصرية) — Bilingual throughout

---

## BILINGUAL REFERENCE TABLE
### All Commands in English & Egyptian Arabic

| Feature | English Variants | Egyptian Arabic (عامية مصرية) | Status |
|---------|------------------|------|--------|
| **Don't Disturb ON** | Enable do not disturb, Turn on DND | وضع عدم الإزعاج, فعّل عدم الإزعاج, اخفت الإشعارات | P1.1 |
| **Don't Disturb OFF** | Disable do not disturb, Turn off DND | قطّع عدم الإزعاج, طفّي عدم الإزعاج | P1.1 |
| **List Running Apps** | Show running apps, List processes, What's running | التطبيقات الشغالة, البرامج الشغالة, إيه البرامج الشغالة | P1.2 |
| **List Files** | Show files, List directory, Show directory | وريني الملفات, افتح المجلد, إيه الملفات في المجلد | P1.3 |
| **Clipboard Read** | What's in clipboard, Read clipboard, Copy from clipboard | إيه اللي في Clipboard, إيه المنسوخ, دوّر في Clipboard | P1.4 |
| **Brightness SET 50%** | Set brightness to 50%, Brightness 50 | اخفض السطوع لـ 50%, وضّع السطوع على 50% | P2.1 |
| **Brightness SET 100%** | Set brightness to 100%, Max brightness | زود السطوع لـ 100%, زود السطوع للأقصى | P2.1 |
| **Volume SET 100%** | Set volume to 100%, Max volume | ارفع الصوت لـ 100%, زود الصوت للأقصى | P2.2 |
| **Volume SET 50%** | Set volume to 50%, Volume half | اخفض الصوت لـ 50%, قلّل الصوت لـ 50% | P2.2 |
| **Mute Volume** | Mute, Silence, Mute audio | أكتم الصوت, اخفت الصوت, طفّي الصوت | P2.2 |
| **Timer 1 Minute** | Set timer for 1 minute, Start 1 minute timer | حط تايمر دقيقة, دقيقة واحدة, ساعة تايمر | P3.1 |
| **Timer 2 Minutes** | Set timer for 2 minutes, 2 minute timer | حط timer على اتنين دقيقة, اتنين دقيقة | P3.1 |
| **Next Track** | Next song, Next track, Skip | الأغنية اللي بعد كده, الأغنية الجاية, Skip | P4.1 |
| **Play/Pause** | Play music, Start music, Pause | شغّل الموسيقى, استمع للموسيقى, وقّف الموسيقى | P4.1 |
| **Open App + Play** | Open Spotify and play music | افتح سبوتيفاي وشغّل الموسيقى, فتح Spotify والموسيقى | P4.2 |
| **Window Minimize** | Minimize window, Minimize this window | صغّر الشبابك, اطوّي الشبابك, صغّر النافذة | P5.1 |
| **Window Resize Down** | Resize window smaller, Make window smaller | صغّر الشبابك أكتر, اخفّف حجم الشبابك | P5.1 |
| **Open Email** | Open email, Active email, Compose email | افتح البريد, اكتب بريد, open email | P5.3 |
| **Rescan Apps** | Rescan apps, Refresh app list, Find installed apps | اسكن البرامج تاني, جدّد قائمة البرامج | P5.2 |

---

## Executive Summary

11 distinct issue categories affecting core functionality:
- **Intent Routing Failures** (4 issues): Don't Disturb, Running Apps, File List, Timer
- **Numeric Argument Extraction** (2 issues): Brightness/Volume percentage parsing
- **Media & System Control** (3 issues): Media playback chains, Window sizing, Volume scope
- **Context Integration** (1 issue): Clipboard read formatting
- **Fallback Handling** (1 issue): Email app fallback
- **Discovery & Mapping** (1 issue): Auto app scanning

**All fixes include BOTH English and Egyptian Arabic support.**

---

## PHASE 0: Root Cause Analysis (Day 1)
### Objective: Identify patterns before coding fixes

#### Issue Group A: Intent Routing Misclassification
**Affected Commands:**
- "طيب، ممكن تفعّل وضع عدم الإزعاج؟" → Should be `OS_SYSTEM_COMMAND(action_key=dnd_on)` but routes to volume handler
- "التطبيقات الشغالة" → Should be `OS_SYSTEM_COMMAND(action_key=list_processes)` but goes to LLM clarification
- "وريني الملفات" → Should be `OS_FILE_NAVIGATION(action=list_directory)` but has low entity confidence (0.05)

**Root Cause:**
1. Parser regex table missing these intent patterns
2. Keyword NLP fuzzy matching not robust enough for Arabic variants
3. Semantic router overriding with LLM_QUERY when confidence low

**Code Locations:**
- [core/command_parser.py](core/command_parser.py) - Regex patterns (lines ~1600-2100)
- [core/command_router.py](core/command_router.py) - Semantic override logic (lines ~3880-3920)
- [core/intent_confidence.py](core/intent_confidence.py) - Confidence thresholds (lines ~200-350)

---

#### Issue Group B: Numeric Argument Extraction Failure
**Affected Commands:**
- "اخفض السطوع لـ 50%." → Parses correctly but extracts `brightness_up` instead of `brightness_set` with `brightness_level=50`
- "ارفع الصوت ل100%." → Same pattern: increments by 10% instead of setting to exact value

**Root Cause:**
1. Parser regex captures "لـ 50" but doesn't extract numeric value into `args`
2. Fallback logic defaults to `brightness_up`/`volume_up` when number extraction fails
3. No post-parse NLU enrichment for numeric slots

**Code Locations:**
- [core/command_parser.py](core/command_parser.py) - Brightness/volume regex patterns (lines ~2200-2350)
- [core/command_router.py](core/command_router.py) - NLU entity enrichment (lines ~3970-4020)

---

#### Issue Group C: Media & System Control Chains
**Affected Commands:**
- "Open Spotify and play music" → Opens only, no media_play follow-up
- "الأغنية اللي بعد كده" → Works (parses as `media_next_track`), but logging shows template execution, not actual command

**Root Cause:**
1. Chained commands not parsed as multi-step
2. Media control dispatch missing actual system call bridging
3. Template execution logged but handler not invoking actual Windows media API

**Code Locations:**
- [core/command_router.py](core/command_router.py) - Media dispatch (lines ~3020-3100)
- [os_control/system_ops.py](os_control/system_ops.py) - Media control handlers (unknown line range)

---

#### Issue Group D: Slot Filling & Timer Parsing
**Affected Commands:**
- "حط تايمر دقيقة" → Asks for `seconds` slot, user says "دقيقة واحدة" but slot not resolved
- "حط timer على اتنين دقيقة" → Opens Clock app instead of setting 120s timer

**Root Cause:**
1. Slot filler receives "دقيقة واحدة" but NLU doesn't convert to numeric seconds (60)
2. Parser doesn't extract duration units from pattern "على اتنين دقيقة"
3. Fallback to Clock app when slot fill fails instead of extracting value

**Code Locations:**
- [core/command_parser.py](core/command_parser.py) - Duration parsing (lines ~780-950, `_duration_to_seconds`)
- [core/command_router.py](core/command_router.py) - Slot fill flow (lines ~3540-3600)
- [os_control/timer_ops.py](os_control/timer_ops.py) - Timer handlers (unknown line range)

---

#### Issue Group E: Context Formatting & Fallbacks
**Affected Commands:**
- "إيه اللي في Clipboard؟" → Response: "في Clipboard نسختة لما>Last_app انت" (garbled)
- "Active email" → Outlook error, doesn't fallback to Gmail

**Root Cause:**
1. Clipboard read returns session context directly instead of actual clipboard content
2. Email handler doesn't check Outlook availability or provide graceful fallback
3. Context injection in LLM response leaking internal state

**Code Locations:**
- [os_control/clipboard_ops.py](os_control/clipboard_ops.py) - Clipboard read (unknown line range)
- [os_control/email_ops.py](os_control/email_ops.py) - Email handler (unknown line range)
- [core/command_router.py](core/command_router.py) - Context injection (lines ~3380-3420)

---

#### Issue Group F: Window & App Scanning
**Affected Commands:**
- "صغر الشبابك" → Minimizes instead of resizing (already mapped to wrong action)
- No command for app discovery/scanning

**Root Cause:**
1. Parser maps "صغر" to `window_minimize` instead of extracting resize intent
2. No app catalog refresh mechanism or auto-discovery on startup
3. App open/close relies on pre-mapped catalog with no fallback discovery

**Code Locations:**
- [core/command_parser.py](core/command_parser.py) - Window action regex (lines ~2450-2550)
- [os_control/app_ops.py](os_control/app_ops.py) - App catalog (unknown line range)

---

## PHASE 1: Quick Wins (Days 2–3)
### Objective: Fix highest-impact, lowest-risk issues first

### P1.1: Fix Don't Disturb Intent Routing
**Issue:** "وضع عدم الإزعاج" parsed as OS_SYSTEM_COMMAND but routed to volume handler

**Fix:**
1. Add regex pattern to [core/command_parser.py](core/command_parser.py) `_REGEX_TABLE`:
   ```
   # ENGLISH VARIANTS
   Pattern EN: r"^(?:enable|turn\s+on|activate)\s+(?:do\s+not\s+disturb|dnd|silent\s+mode)"
   Intent: OS_SYSTEM_COMMAND
   Action: "" (no-action)
   Args: {"action_key": "dnd_on"}
   
   Pattern EN: r"^(?:disable|turn\s+off|deactivate)\s+(?:do\s+not\s+disturb|dnd|silent\s+mode)"
   Intent: OS_SYSTEM_COMMAND
   Action: "" (no-action)
   Args: {"action_key": "dnd_off"}
   
   # EGYPTIAN ARABIC VARIANTS (عامية مصرية)
   Pattern AR: r"^(?:وضع|فعّل)\s+(?:عدم\s+الإزعاج|الصمت|السكوت)"
   Intent: OS_SYSTEM_COMMAND
   Action: "" (no-action)
   Args: {"action_key": "dnd_on"}
   
   Pattern AR: r"^(?:قطّع|طفّي)\s+(?:عدم\s+الإزعاج|الصمت|السكوت)"
   Intent: OS_SYSTEM_COMMAND
   Action: "" (no-action)
   Args: {"action_key": "dnd_off"}
   ```

2. Add to [core/command_parser.py](core/command_parser.py) `_KEYWORD_TABLE` for easier matching:
   ```
   Keywords DND ON (EN): {"enable do not disturb", "turn on dnd", "silent mode on", "dnd on"}
   Keywords DND ON (AR): {"وضع عدم الإزعاج", "فعّل عدم الإزعاج", "خليه ساكت", "اخفت الإشعارات"}
   
   Keywords DND OFF (EN): {"disable do not disturb", "turn off dnd", "silent mode off", "dnd off"}
   Keywords DND OFF (AR): {"قطّع عدم الإزعاج", "طفّي عدم الإزعاج", "خليه عادي", "فتح الإشعارات"}
   ```

3. Implement handler in [os_control/system_ops.py](os_control/system_ops.py):
   - Check Windows Settings API for Do Not Disturb toggle
   - Fallback: Use `Settings ms-settings:notifications` to open Settings

**Code Changes:**
- [core/command_parser.py](core/command_parser.py) - Add 4 regex patterns + keyword entries (~80 lines total)
- [os_control/system_ops.py](os_control/system_ops.py) - Add `dnd_on`/`dnd_off` handler (~30 lines)

**Test Cases:**
```
# ENGLISH VARIANTS
Input EN 1: "Enable do not disturb"
Input EN 2: "Turn on DND"
Expected: action_key=dnd_on executed, DND enabled, response: "Do Not Disturb enabled"

Input EN 1: "Disable do not disturb"
Input EN 2: "Turn off DND"
Expected: action_key=dnd_off executed, DND disabled, response: "Do Not Disturb disabled"

# EGYPTIAN ARABIC VARIANTS
Input AR 1: "وضع عدم الإزعاج"
Input AR 2: "فعّل عدم الإزعاج"
Input AR 3: "خليه ساكت"
Expected: DND enabled, response: "تمام، وضعت عدم الإزعاج"

Input AR 1: "قطّع عدم الإزعاج"
Input AR 2: "طفّي السكوت"
Input AR 3: "خليه عادي"
Expected: DND disabled, response: "تمام، طفيت عدم الإزعاج"
```

**Effort:** 2.5 hours (expanded for bilingual)  
**Risk:** Low (isolated to new command)

---

### P1.2: Fix Running Apps List Intent
**Issue:** "التطبيقات الشغالة" → LLM_QUERY low confidence instead of list_processes

**Fix:**
1. Add keyword pattern to [core/command_parser.py](core/command_parser.py):
   ```
   # ENGLISH KEYWORDS
   Keywords: {"list running apps", "show running applications", "what apps are running", "show processes"}
   
   # EGYPTIAN ARABIC KEYWORDS (عامية مصرية)
   Keywords: {"التطبيقات الشغالة", "البرامج الشغالة", "إيه البرامج الشغالة", "البرامج اللي فتوحة", "إيه الحاجات الشغالة"}
   
   Intent: OS_SYSTEM_COMMAND
   Action: ""
   Args: {"action_key": "list_processes"}
   ```

2. Add to semantic router patterns if not present

3. Implement handler in [os_control/system_ops.py](os_control/system_ops.py):
   - Use `tasklist` or psutil to enumerate running processes
   - Filter system processes vs user apps
   - Format output in both English and Arabic:
     - EN: "Running apps: Spotify, Chrome, VS Code..."
     - AR: "البرامج الشغالة: Spotify، Chrome، VS Code..."

**Code Changes:**
- [core/command_parser.py](core/command_parser.py) - Add keyword entries (~10 lines)
- [os_control/system_ops.py](os_control/system_ops.py) - Add handler with bilingual output (~50 lines)

**Test Cases:**
```
# ENGLISH VARIANTS
Input EN 1: "List running apps"
Input EN 2: "Show running applications"
Input EN 3: "What apps are running"
Expected: Lists 5–10 running user applications in English

# EGYPTIAN ARABIC VARIANTS
Input AR 1: "التطبيقات الشغالة"
Input AR 2: "البرامج الشغالة"
Input AR 3: "إيه البرامج الشغالة"
Input AR 4: "إيه الحاجات الشغالة"
Expected: Lists running apps in Arabic, e.g., "البرامج الشغالة: Spotify، Chrome، VS Code"
```

**Effort:** 2 hours (bilingual output)  
**Risk:** Low

---

### P1.3: Fix File List Intent + Entity Confidence
**Issue:** "وريني الملفات" → Entity confidence 0.05, triggers clarification

**Fix:**
1. Lower entity confidence threshold in [core/intent_confidence.py](core/intent_confidence.py):
   - Current: `ENTITY_CLARIFICATION_THRESHOLD_BY_INTENT["OS_FILE_NAVIGATION"]` = (infer from code)
   - Change: For "list_directory" action with no file specifier, set confidence floor to 0.70

2. Add parser keywords:
   ```
   # ENGLISH KEYWORDS
   Keywords: {"list files", "show directory", "show files", "list directory", "what files"}
   
   # EGYPTIAN ARABIC KEYWORDS (عامية مصرية)
   Keywords: {"وريني الملفات", "افتح المجلد", "بنروح على المجلد", "إيه الملفات في المجلد", "شوفني الملفات"}
   
   Intent: OS_FILE_NAVIGATION
   Action: "list_directory"
   Args: {"path": ""} (current directory)
   ```

3. Suppress clarification for generic list requests (no specific file/folder name)

**Code Changes:**
- [core/command_parser.py](core/command_parser.py) - Add keyword entries (~10 lines)
- [core/intent_confidence.py](core/intent_confidence.py) - Adjust thresholds (~10 lines)
- [core/command_router.py](core/command_router.py) - Skip clarification for path=""/no target (~5 lines)

**Test Cases:**
```
# ENGLISH VARIANTS
Input EN 1: "List files"
Input EN 2: "Show directory"
Expected: Confidence ≥0.70, lists current directory contents

# EGYPTIAN ARABIC VARIANTS
Input AR 1: "وريني الملفات"
Input AR 2: "افتح المجلد"
Input AR 3: "إيه الملفات في المجلد"
Expected: Confidence ≥0.70, lists current directory (NO clarification needed)
```

**Effort:** 1.5 hours (bilingual)  
**Risk:** Low (threshold-only change)

---

### P1.4: Fix Clipboard Content Formatting
**Issue:** Response mixes clipboard data with session context: "في Clipboard نسختة لما>Last_app انت"

**Fix:**
1. Inspect [os_control/clipboard_ops.py](os_control/clipboard_ops.py) `read_clipboard()`:
   - Check if function returns raw session state instead of clipboard content

2. Fix clipboard read to return only text content:
   ```python
   def read_clipboard():
       try:
           content = pyperclip.paste()
           return content if content else "(Clipboard empty)"
       except:
           return "(Clipboard error or not accessible)"
   ```

3. Update [core/command_router.py](core/command_router.py) LLM dispatch to handle clipboard responses:
   - Don't inject session context into clipboard-read responses
   - Return bilingual format:
     - EN: "Clipboard contains: [text]"
     - AR: "في Clipboard: [text]"

**Code Changes:**
- [os_control/clipboard_ops.py](os_control/clipboard_ops.py) - Fix read (~15 lines)
- [core/command_router.py](core/command_router.py) - Context isolation + bilingual response (~20 lines)

**Test Cases:**
```
# ENGLISH VARIANT
Setup: Copy "Hello World" to clipboard
Input EN: "What's in clipboard"
Input EN: "Read clipboard"
Expected EN: "Clipboard contains: Hello World" (NO internal state mixed in)

# EGYPTIAN ARABIC VARIANT
Setup: Copy "مرحبا بالعالم" to clipboard
Input AR 1: "إيه اللي في Clipboard"
Input AR 2: "إيه المنسوخ"
Input AR 3: "دوّر في Clipboard"
Expected AR: "في Clipboard: مرحبا بالعالم" (clean, NO context leak)
```

**Effort:** 1.5 hours (bilingual validation)  
**Risk:** Low

---

## PHASE 2: Numeric Argument Extraction (Days 4–5)
### Objective: Fix brightness/volume set-to-value parsing

### P2.1: Extract Numeric Values from Natural Language
**Issue:** 
- "اخفض السطوع لـ 50%." → Extracts correctly but handler doesn't use the number
- "ارفع الصوت ل100%." → Same

**Root Cause:** Regex pattern captures the phrase but args dict doesn't include extracted number

**Fix:**
1. Review brightness/volume regex patterns in [core/command_parser.py](core/command_parser.py) (lines ~2200-2350):
   ```
   # ENGLISH PATTERNS - BRIGHTNESS
   Pattern EN: r"^(?:set|adjust|change)\s+brightness\s+(?:to|at)?\s+(\d{1,3})%?"
   Args builder: {"action_key": "brightness_set", "brightness_level": int(m.group(1))}
   
   # ENGLISH PATTERNS - VOLUME
   Pattern EN: r"^(?:set|adjust|change)\s+volume\s+(?:to|at)?\s+(\d{1,3})%?"
   Args builder: {"action_key": "volume_set", "volume_level": int(m.group(1))}
   
   # EGYPTIAN ARABIC PATTERNS - BRIGHTNESS (عامية مصرية)
   Pattern AR: r"^(?:اخفض|قلّل|خفّف)\s+السطوع\s+(?:ل|لـ|إلى)\s+(\d{1,3})%?"
   Args builder: {"action_key": "brightness_set", "brightness_level": int(m.group(1))}
   
   Pattern AR: r"^وضّع\s+السطوع\s+(?:ل|لـ|إلى)\s+(\d{1,3})%?"
   Args builder: {"action_key": "brightness_set", "brightness_level": int(m.group(1))}
   
   Pattern AR: r"^زود\s+السطوع\s+(?:ل|لـ|إلى)\s+(\d{1,3})%?"
   Args builder: {"action_key": "brightness_set", "brightness_level": int(m.group(1))}
   
   # EGYPTIAN ARABIC PATTERNS - VOLUME
   Pattern AR: r"^(?:اخفض|قلّل|خفّف)\s+الصوت\s+(?:ل|لـ|إلى)\s+(\d{1,3})%?"
   Args builder: {"action_key": "volume_set", "volume_level": int(m.group(1))}
   
   Pattern AR: r"^ارفع\s+الصوت\s+(?:ل|لـ|إلى)\s+(\d{1,3})%?"
   Args builder: {"action_key": "volume_set", "volume_level": int(m.group(1))}
   ```

2. Audit patterns to ensure numeric groups are captured and passed to args

3. Add handler validation in [os_control/system_ops.py](os_control/system_ops.py):
   - Check if `brightness_level`/`volume_level` arg is present and 0–100
   - If missing, default to 10% increment/decrement (not 100%)

**Code Changes:**
- [core/command_parser.py](core/command_parser.py) - Add/fix patterns (~60 lines, bilingual)
- [os_control/system_ops.py](os_control/system_ops.py) - Validate args (~25 lines)

**Test Cases:**
```
# ENGLISH VARIANTS - BRIGHTNESS SET
Input EN 1: "Set brightness to 50%"
Input EN 2: "Adjust brightness to 50"
Input EN 3: "Brightness 50"
Expected: brightness_level=50, executed, result: 50%

Input EN 1: "Set brightness to 100"
Input EN 2: "Max brightness"
Expected: brightness_level=100, executed, result: 100%

# EGYPTIAN ARABIC VARIANTS - BRIGHTNESS SET
Input AR 1: "اخفض السطوع لـ 50%"
Input AR 2: "قلّل السطوع ل 50%"
Input AR 3: "وضّع السطوع إلى 50%"
Expected: brightness_level=50, executed, result: 50%

Input AR 1: "زود السطوع لـ 100%"
Input AR 2: "زود السطوع للأقصى"
Expected: brightness_level=100, executed, result: 100%

# ENGLISH VARIANTS - VOLUME SET
Input EN 1: "Set volume to 100%"
Input EN 2: "Volume 75"
Expected: volume_level=100 or 75, executed

# EGYPTIAN ARABIC VARIANTS - VOLUME SET
Input AR 1: "ارفع الصوت لـ 100%"
Input AR 2: "اخفض الصوت لـ 50%"
Input AR 3: "قلّل الصوت ل 30%"
Expected: volume_level extracted correctly

# NO VALUE PROVIDED (FALLBACK)
Input EN: "Increase brightness" (no value)
Input AR: "زود السطوع" (no value)
Expected: brightness_level NOT in args, increment by 10% (current logic preserved)
```

**Effort:** 4 hours (comprehensive bilingual patterns)  
**Risk:** Medium (affects multiple commands; needs thorough testing)

---

### P2.2: Fix Volume Control Scope (System vs App)
**Issue:** Volume commands only affect app volume, not system master volume

**Fix:**
1. Inspect [os_control/system_ops.py](os_control/system_ops.py) `volume_up`/`volume_down` handlers:
   - Check if using Windows audio API or just app-level volume

2. Update to use Windows Core Audio API:
   ```python
   from pycaw.pycoreutils import IAudioUtility
   devices = AudioUtilities.GetSpeakers()
   interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
   volume = interface.QueryInterface(IAudioEndpointVolume)
   volume.SetMasterVolumeLevel(level_db, None)  # System volume (NOT app volume)
   ```

3. Add new action for app-only volume if users want app-specific control:
   - `volume_app_up` vs `volume_system_up`
   - Default to system volume for now

**Code Changes:**
- [os_control/system_ops.py](os_control/system_ops.py) - Replace volume handlers (~50 lines)
- Add dependency: `pycaw` to requirements.txt

**Test Cases:**
```
# ENGLISH VARIANTS
Input EN 1: "Set volume to 50%"
Input EN 2: "Volume 80"
Expected: System master volume set to 50% or 80%
Verify EN: Volume slider in Windows Settings shows correct level

# EGYPTIAN ARABIC VARIANTS
Input AR 1: "ارفع الصوت لـ 50%"
Input AR 2: "اخفض الصوت لـ 50%"
Input AR 3: "قلّل الصوت ل 30%"
Expected: System master volume set (verified via Settings)
NOT app volume only
```

**Effort:** 2.5 hours (bilingual testing)  
**Risk:** Medium (audio API integration; may need fallback for compatibility)

---

## PHASE 3: Slot Filling & Duration Parsing (Days 6–7)
### Objective: Fix timer/duration parsing and clarification loops

### P3.1: Fix Duration Unit Conversion in Slot Filler
**Issue:**
- User says "حط تايمر دقيقة" → Asked for seconds
- User replies "دقيقة واحدة" → Not converted to 60 seconds
- Timer doesn't set

**Root Cause:**
1. Slot filler receives string "دقيقة واحدة" but doesn't run NLU extraction
2. Parser's `_duration_to_seconds` not called during slot fill

**Fix:**
1. In [core/command_router.py](core/command_router.py) slot-fill handler (lines ~3540-3600):
   ```python
   if missing_slot == "seconds":
       # Extract duration from reply text
       duration_seconds = parse_duration_from_text(effective_text)
       if duration_seconds is not None:
           saved_args["seconds"] = duration_seconds
           # Proceed with dispatch
   ```

2. Create `parse_duration_from_text()` helper in [core/command_parser.py](core/command_parser.py):
   - Uses `_duration_to_seconds()` logic
   - Handles BOTH English and Arabic:
     - EN: "one minute" → 60s, "2 seconds" → 2s, "30 minutes" → 1800s
     - AR: "دقيقة واحدة" → 60s, "دقيقتين" → 120s, "ساعة" → 3600s, "ثانية" → 1s

3. Add patterns to [core/command_parser.py](core/command_parser.py) for direct timer setting:
   ```
   # ENGLISH PATTERNS
   Pattern EN: r"^(?:set\s+)?timer\s+(?:for\s+)?([0-9]+)\s+(seconds?|minutes?|hours?)"
   Extract: duration = _duration_to_seconds(m.group(1), m.group(2))
   Args: {"seconds": duration}
   
   # EGYPTIAN ARABIC PATTERNS
   Pattern AR: r"^حط\s+(?:timer|تايمر)\s+(?:على|ل)\s+([0-9]+)\s+(ثانية|ثواني|دقيقة|دقائق|ساعة|ساعات)"
   Extract: duration = _duration_to_seconds(m.group(1), m.group(2))
   Args: {"seconds": duration}
   ```

**Code Changes:**
- [core/command_parser.py](core/command_parser.py) - Add patterns + helper (~70 lines, bilingual)
- [core/command_router.py](core/command_router.py) - Fix slot filler (~30 lines)

**Test Cases:**
```
# ENGLISH VARIANTS
Input EN 1: "Set timer for 1 minute"
Input EN 2: "Timer 60 seconds"
Input EN 3: "Timer 2 minutes"
System EN: "How many seconds?"
User EN: "One minute" or "60 seconds"
Expected: Timer set (60s or appropriate value)

# EGYPTIAN ARABIC VARIANTS - DIRECT SETTING
Input AR 1: "حط timer على دقيقة" (1 minute implied)
Input AR 2: "حط تايمر على اتنين دقيقة"
Input AR 3: "حط timer لـ 30 ثانية"
Expected: Timer set directly (NO clarification)

# EGYPTIAN ARABIC VARIANTS - SLOT FILL RECOVERY
Input AR 1: "حط تايمر دقيقة"
System AR: "كم ثانية؟"
User AR 1: "دقيقة واحدة"
User AR 2: "ستين ثانية"
User AR 3: "دقيقة"
Expected: Timer set to 60 seconds (any variant understood)
```

**Effort:** 3 hours (comprehensive bilingual)  
**Risk:** Low (isolated to timer path)

---

### P3.2: Add Timer Direct Execution Path
**Issue:** "حط timer على اتنين دقيقة" → Opens Clock app instead of setting timer

**Fix:**
1. Ensure parser pattern extracts duration correctly (see P3.1 above)

2. In [os_control/timer_ops.py](os_control/timer_ops.py), add/verify `set_timer()` implementation:
   ```python
   def set_timer(seconds: int, label: str = None):
       # Use Windows timer API or background scheduler
       # Return bilingual response:
       #   EN: "Timer set for {minutes}m {seconds}s"
       #   AR: "تمام، وضعت تايمر {duration}"
       return {"success": True, "message": f"Timer set for {seconds}s"}
   ```

3. Dispatch in [core/command_router.py](core/command_router.py):
   - If intent=OS_TIMER and action=set and seconds present, call handler directly
   - Only open Clock app as fallback if direct execution fails

**Code Changes:**
- [os_control/timer_ops.py](os_control/timer_ops.py) - Verify/implement timer (~50 lines)
- [core/command_router.py](core/command_router.py) - Direct dispatch logic (~10 lines)

**Test Cases:**
```
# ENGLISH VARIANTS
Input EN 1: "Set timer for 2 minutes"
Input EN 2: "Timer 120 seconds"
Expected: "Timer set for 2 minutes" (executed immediately, NO Clock app)

# EGYPTIAN ARABIC VARIANTS
Input AR 1: "حط timer على اتنين دقيقة"
Input AR 2: "ساعة تايمر دقيقتين"
Input AR 3: "شوية تايمر دقيقة"
Expected: "تمام، وضعت تايمر لمدة دقيقتين" (executed immediately)
```

**Effort:** 2 hours (bilingual output)  
**Risk:** Low

---

## PHASE 4: Media & Chained Commands (Days 8–9)
### Objective: Fix media control execution and multi-step sequences

### P4.1: Fix Media Control Actual System Dispatch
**Issue:** Media commands parse correctly but template execution logged without actual system call

**Root Cause:** Handler returns template name instead of executing system call

**Fix:**
1. Locate [os_control/system_ops.py](os_control/system_ops.py) `media_next_track()` (or similar):
   - Check if it just returns template or actually sends key press

2. Update to use pynput to send key press:
   ```python
   from pynput.keyboard import Controller, Key
   keyboard = Controller()
   keyboard.press_and_release(Key.media_next)  # Next track
   # Return bilingual response:
   #   EN: "Next track" or "Playing next song"
   #   AR: "الأغنية الجاية" or "الأغنية اللي بعدها"
   ```

3. Verify all media actions wired correctly:
   - `media_play_pause` → Key.media_play_pause
   - `media_next_track` → Key.media_next
   - `media_previous_track` → Key.media_previous
   - `media_stop` → Handled (may need custom implementation)

**Code Changes:**
- [os_control/system_ops.py](os_control/system_ops.py) - Fix media dispatch (~60 lines)
- Add dependency: `pynput` to requirements.txt

**Test Cases:**
```
# ENGLISH VARIANTS - NEXT TRACK
Input EN 1: "Next song" (Spotify playing)
Input EN 2: "Skip track"
Expected: Next track plays

# EGYPTIAN ARABIC VARIANTS - NEXT TRACK
Input AR 1: "الأغنية اللي بعد كده" (Spotify playing)
Input AR 2: "الأغنية الجاية"
Input AR 3: "Skip"
Expected: Next track plays

# ENGLISH VARIANTS - PLAY/PAUSE
Input EN 1: "Play music" (music paused)
Input EN 2: "Play"
Expected: Music resumes

# EGYPTIAN ARABIC VARIANTS - PLAY/PAUSE
Input AR 1: "شغّل الموسيقى"
Input AR 2: "شغّل"
Expected: Music resumes
```

**Effort:** 2.5 hours (bilingual testing + verification)  
**Risk:** Medium (keyboard input can be flaky; test with real apps)

---

### P4.2: Add Chained Command Support (Open + Play)
**Issue:** "Open Spotify and play music" → Opens only, no follow-up

**Root Cause:**
1. Parser doesn't recognize "and" / "و" as command chain operator
2. No multi-step handler for sequential commands

**Fix:**
1. Add patterns to [core/command_parser.py](core/command_parser.py) to detect chains:
   ```
   # ENGLISH PATTERN
   Pattern EN: r"^(?:open|launch)\s+([a-z]+)\s+(?:and|then)\s+(?:play\s+)?music"
   Intent: COMMAND_CHAIN
   Args: {
       "commands": [
           {"intent": "OS_APP_OPEN", "args": {"app_name": group(1)}},
           {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "media_play"}}
       ],
       "delay_ms": 2000  # Wait for app to start
   }
   
   # EGYPTIAN ARABIC PATTERN
   Pattern AR: r"^(?:افتح|فتح)\s+([a-z\u0600-\u06FF]+)\s+و(?:شغّل\s+)?(?:الموسيقى|الأغاني)"
   Intent: COMMAND_CHAIN
   Args: {
       "commands": [
           {"intent": "OS_APP_OPEN", "args": {"app_name": group(1)}},
           {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "media_play"}}
       ],
       "delay_ms": 2000
   }
   ```

2. Add handler in [core/command_router.py](core/command_router.py):
   ```python
   if parsed.intent == "COMMAND_CHAIN":
       for i, cmd in enumerate(parsed.args.get("commands", [])):
           if i > 0:
               delay = cmd.get("delay_ms", 1000)
               time.sleep(delay / 1000.0)  # Convert ms to seconds
           execute_command(cmd)
       # Return bilingual response:
       #   EN: f"Opening {app_name} and playing music"
       #   AR: f"افتح {app_name} والموسيقى"
   ```

3. Fallback: If no explicit chain pattern, post-dispatch could issue follow-up if first command is app open

**Code Changes:**
- [core/command_parser.py](core/command_parser.py) - Add chain patterns (~30 lines, bilingual)
- [core/command_router.py](core/command_router.py) - Add chain handler (~60 lines)

**Test Cases:**
```
# ENGLISH VARIANTS
Input EN 1: "Open Spotify and play music"
Input EN 2: "Launch Spotify then play"
Expected: Spotify opens, waits 2s, music starts playing

# EGYPTIAN ARABIC VARIANTS
Input AR 1: "افتح سبوتيفاي وشغّل الموسيقى"
Input AR 2: "فتح Spotify والموسيقى"
Input AR 3: "افتح يوتيوب والأغاني"
Expected: App opens, waits 2s, music plays
```

**Effort:** 3 hours (bilingual patterns + timing)  
**Risk:** Medium (timing-dependent; needs tuning)

---

## PHASE 5: Window Control & App Discovery (Days 10–11)
### Objective: Fix window sizing and auto-discovery

### P5.1: Fix Window Resize Intent
**Issue:** "صغر الشبابك" → Minimizes instead of resizing

**Root Cause:** Parser maps "صغر" to `window_minimize` instead of checking for resize context

**Fix:**
1. Review [core/command_parser.py](core/command_parser.py) patterns for "صغر":
   - Current: Likely mapped to minimize
   - Change: Add "صغر" patterns for BOTH minimize and resize depending on context:
     ```
     # MINIMIZE (isolated action)
     Pattern EN: r"^(?:minimize|minimize\s+window)"
     Pattern AR: r"^(?:صغّر|اطوّي|صغّر\s+الشبابك|صغّر\s+النافذة)$"
     → window_minimize (isolated)
     
     # RESIZE (with intensifier "أكتر" = more/further)
     Pattern AR: r"^(?:صغّر|اخفّف)\s+(?:الشبابك|النافذة)\s+(?:أكتر|شوية|أكثر)"
     → window_resize (decrease size, not minimize)
     
     # ENGLISH RESIZE
     Pattern EN: r"^(?:resize|make\s+smaller|shrink)\s+(?:window|this\s+window)"
     → window_resize
     ```

2. Add "resize" action and handler in [os_control/system_ops.py](os_control/system_ops.py):
   ```python
   def window_resize(direction="decrease", amount_percent=10):
       # Get active window
       # Adjust size by ~10% or to specific percentage
       # Return bilingual response:
       #   EN: "Window resized to 90% of original size"
       #   AR: "تمام، صغرت الشبابك"
   ```

3. Clarify intent by checking modifiers:
   - EN: "minimize window" (no modifier) vs "Resize window smaller" (explicit)
   - AR: "صغر الشبابك" (minimize) vs "صغر الشبابك أكتر" (resize)

**Code Changes:**
- [core/command_parser.py](core/command_parser.py) - Add/fix patterns (~30 lines, bilingual)
- [os_control/system_ops.py](os_control/system_ops.py) - Add resize handler (~50 lines)

**Test Cases:**
```
# MINIMIZE (ISOLATED)
Input EN 1: "Minimize window"
Input EN 2: "Minimize"
Input AR 1: "صغّر الشبابك"
Input AR 2: "اطوّي الشبابك"
Expected: Current window minimized to taskbar

# RESIZE (WITH MODIFIER)
Input EN 1: "Resize window smaller"
Input EN 2: "Make window smaller"
Input AR 1: "صغّر الشبابك أكتر"
Input AR 2: "اخفّف حجم الشبابك"
Input AR 3: "صغّر الشبابك شوية"
Expected: Current window resized to 90% size (or snap left/right)
```

**Effort:** 2 hours (bilingual pattern distinction)  
**Risk:** Low

---

### P5.2: Auto App Discovery & Catalog Refresh
**Issue:** Limited to pre-mapped apps; no way to open unmapped apps

**Fix:**
1. Create [os_control/app_discovery.py](os_control/app_discovery.py):
   ```python
   def discover_installed_apps():
       """Scan:
       - Desktop shortcuts
       - Windows Start Menu
       - Program Files
       - User AppData\Local\Programs
       - Microsoft Store (via Registry)
       """
       apps = {}
       # Scan shortcuts
       for lnk_file in Path(os.path.expandvars(r"%USERPROFILE%\Desktop")).glob("*.lnk"):
           apps[lnk_file.stem.lower()] = {
               "path": resolve_lnk(lnk_file),
               "source": "desktop"
           }
       # Scan Start Menu
       start_menu = Path(os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs"))
       for lnk in start_menu.rglob("*.lnk"):
           apps[lnk.stem.lower()] = {"path": resolve_lnk(lnk), "source": "start_menu"}
       # ... more sources
       return apps
   ```

2. Add command to trigger discovery:
   ```
   Pattern: r"^(?:rescan|refresh|scan)\s+(?:apps?|installed\s+apps?)"
   Intent: OS_SYSTEM_COMMAND
   Action: ""
   Args: {"action_key": "rescan_apps"}
   ```

3. Run auto-discovery on startup and cache results in [os_control/app_ops.py](os_control/app_ops.py):
   - Merge with existing `_APP_CATALOG`
   - Save to `~/.jarvis/app_cache.json` for persistence

4. Update `resolve_app_request()` to check cache before failing:
   ```python
   def resolve_app_request(app_name):
       if app_name in _APP_CATALOG:
           return _APP_CATALOG[app_name]
       # Fallback: Check cache and re-scan if needed
       if app_name in discover_installed_apps():
           return ...
   ```

**Code Changes:**
- Create [os_control/app_discovery.py](os_control/app_discovery.py) (~120 lines)
- Update [os_control/app_ops.py](os_control/app_ops.py) - Add cache + merging (~50 lines)
- Update [core/orchestrator.py](core/orchestrator.py) - Add startup discovery call (~10 lines)
- Update [core/command_parser.py](core/command_parser.py) - Add rescan command (~10 lines)

**Test Cases:**
```
Startup: Auto-discovery runs, finds 50+ apps

Input: "rescan apps"
Expected: "Rescanning installed apps... Found 58 applications."

Input: "افتح VLC" (or any unregistered app found in scan)
Expected: App opens (previously would fail)
```

**Effort:** 4 hours  
**Risk:** Medium (depends on OS API stability; needs testing on fresh Windows install)

---

### P5.3: Add Email Fallback Handler
**Issue:** Outlook open error → No fallback to Gmail

**Fix:**
1. Update [os_control/email_ops.py](os_control/email_ops.py):
   ```python
   def draft_email():
       try:
           # Try Outlook first
           outlook = win32com.client.Dispatch("Outlook.Application")
           outlook.CreateItem(0).Display()  # 0 = MailItem
           return {"success": True, "app": "Outlook"}
       except Exception as e:
           logger.warning(f"Outlook failed: {e}, trying Gmail fallback")
           # Fallback: Open Gmail in browser
           try:
               import webbrowser
               webbrowser.open("https://mail.google.com/mail/u/0/#compose")
               return {"success": True, "app": "Gmail"}
           except:
               return {"success": False, "error": "Email unavailable"}
   ```

2. Update response template to indicate which opened (bilingual):
   - EN Success: "Opening Outlook..."
   - EN Fallback: "Outlook not available, opening Gmail..."
   - AR Success: "فاتح Outlook..."
   - AR Fallback: "Outlook مش متاح، فاتح Gmail..."

**Code Changes:**
- [os_control/email_ops.py](os_control/email_ops.py) - Add fallback (~30 lines)
- [core/command_router.py](core/command_router.py) - Bilingual response handling (~10 lines)

**Test Cases:**
```
# ENGLISH VARIANTS
Input EN 1: "Open email"
Input EN 2: "Compose email"
Expected: Outlook opens (if available), else Gmail opens in browser

# EGYPTIAN ARABIC VARIANTS
Input AR 1: "اكتب بريد"
Input AR 2: "افتح البريد"
Input AR 3: "افتح الإيميل"
Expected: Outlook opens (if available), else Gmail opens
Response AR: "فاتح Outlook" or "Outlook مش متاح، فاتح Gmail"
```

**Effort:** 1 hour (bilingual response handling)  
**Risk:** Low

---

## PHASE 6: Integration & QA (Days 12–14)
### Objective: Test all fixes together, prevent regressions

### P6.1: Regression Test Suite
**Create [tests/phase5_production_fixes.py](tests/phase5_production_fixes.py):**

Test each issue with **minimum 3+ cases per fix in BOTH English and Egyptian Arabic**:

```python
# ===== P1.1: DND Toggle =====
def test_dnd_toggle_english():
    """Test DND toggle in English"""
    r1a = route_command("Enable do not disturb")
    assert r1a.success and "dnd_on" in str(r1a)
    
    r1b = route_command("Turn on DND")
    assert r1b.success and "dnd_on" in str(r1b)
    
    r1c = route_command("Disable do not disturb")
    assert r1c.success and "dnd_off" in str(r1c)

def test_dnd_toggle_arabic():
    """Test DND toggle in Egyptian Arabic"""
    r2a = route_command("وضع عدم الإزعاج")
    assert r2a.success and "dnd_on" in str(r2a)
    
    r2b = route_command("فعّل عدم الإزعاج")
    assert r2b.success and "dnd_on" in str(r2b)
    
    r2c = route_command("قطّع عدم الإزعاج")
    assert r2c.success and "dnd_off" in str(r2c)

# ===== P1.2: List Running Apps =====
def test_list_running_apps_english():
    """Test list running apps in English"""
    r3a = route_command("List running apps")
    assert r3a.success and "list_processes" in str(r3a)
    
    r3b = route_command("Show running applications")
    assert r3b.success and "list_processes" in str(r3b)

def test_list_running_apps_arabic():
    """Test list running apps in Egyptian Arabic"""
    r4a = route_command("التطبيقات الشغالة")
    assert r4a.success and "list_processes" in str(r4a)
    
    r4b = route_command("البرامج الشغالة")
    assert r4b.success and "list_processes" in str(r4b)
    
    r4c = route_command("إيه البرامج الشغالة")
    assert r4c.success and "list_processes" in str(r4c)

# ===== P1.3: File List (No Clarification) =====
def test_file_list_english():
    """Test file list in English without clarification"""
    r5a = route_command("List files")
    assert r5a.success and "list_directory" in str(r5a)
    assert r5a.confidence >= 0.70  # No clarification triggered
    
    r5b = route_command("Show directory")
    assert r5b.success and "list_directory" in str(r5b)

def test_file_list_arabic():
    """Test file list in Egyptian Arabic without clarification"""
    r6a = route_command("وريني الملفات")
    assert r6a.success and "list_directory" in str(r6a)
    assert r6a.confidence >= 0.70  # Should NOT trigger clarification
    
    r6b = route_command("افتح المجلد")
    assert r6b.success and "list_directory" in str(r6b)

# ===== P1.4: Clipboard Content Clean =====
def test_clipboard_english():
    """Test clipboard read in English"""
    import pyperclip
    pyperclip.copy("Hello World")
    r7 = route_command("What's in clipboard")
    assert "Hello World" in str(r7)
    assert "Last_app" not in str(r7)  # No context leak

def test_clipboard_arabic():
    """Test clipboard read in Egyptian Arabic"""
    import pyperclip
    pyperclip.copy("مرحبا بالعالم")
    r8 = route_command("إيه اللي في Clipboard")
    assert "مرحبا بالعالم" in str(r8)
    assert "context" not in str(r8).lower()  # No context leak

# ===== P2.1: Brightness Set to Value =====
def test_brightness_set_english():
    """Test brightness set to value in English"""
    r9a = route_command("Set brightness to 50")
    assert r9a.success and "brightness_level" in str(r9a)
    assert "50" in str(r9a)
    
    r9b = route_command("Brightness 75")
    assert r9b.success and "75" in str(r9b)

def test_brightness_set_arabic():
    """Test brightness set to value in Egyptian Arabic"""
    r10a = route_command("اخفض السطوع لـ 50%")
    assert r10a.success and "50" in str(r10a)
    
    r10b = route_command("زود السطوع لـ 100%")
    assert r10b.success and "100" in str(r10b)
    
    r10c = route_command("وضّع السطوع إلى 80%")
    assert r10c.success and "80" in str(r10c)

# ===== P2.2: Volume Set to Value (System) =====
def test_volume_set_english():
    """Test system volume set in English"""
    r11a = route_command("Set volume to 100")
    assert r11a.success and "volume_level" in str(r11a)
    assert "100" in str(r11a)
    # Verify system volume changed via Windows API check
    
def test_volume_set_arabic():
    """Test system volume set in Egyptian Arabic"""
    r12a = route_command("ارفع الصوت لـ 100%")
    assert r12a.success and "100" in str(r12a)
    
    r12b = route_command("اخفض الصوت لـ 50%")
    assert r12b.success and "50" in str(r12b)

# ===== P3.1 & P3.2: Timer Duration Parsing & Direct Execution =====
def test_timer_english():
    """Test timer in English"""
    r13a = route_command("Set timer for 2 minutes")
    assert r13a.success and "120" in str(r13a)  # 2 minutes = 120 seconds
    
    r13b = route_command("Timer 1 minute")
    assert r13b.success and "60" in str(r13b)

def test_timer_arabic():
    """Test timer in Egyptian Arabic"""
    r14a = route_command("حط timer على اتنين دقيقة")
    assert r14a.success and "120" in str(r14a)  # Direct execution, no Clock app
    
    r14b = route_command("ساعة تايمر دقيقة")  # Slot filler test
    # Should resolve "دقيقة" to 60 seconds
    assert r14b.success

# ===== P4.1: Media Next Track =====
def test_media_next_english():
    """Test media next track in English"""
    r15 = route_command("Next track")
    assert r15.success and "media_next_track" in str(r15)

def test_media_next_arabic():
    """Test media next track in Egyptian Arabic"""
    r16a = route_command("الأغنية اللي بعد كده")
    assert r16a.success and "media_next_track" in str(r16a)
    
    r16b = route_command("الأغنية الجاية")
    assert r16b.success and "media_next_track" in str(r16b)

# ===== P4.2: Chained Commands =====
def test_chained_command_english():
    """Test open app and play music in English"""
    r17 = route_command("Open Spotify and play music")
    assert r17.success and "COMMAND_CHAIN" in str(r17)

def test_chained_command_arabic():
    """Test open app and play music in Egyptian Arabic"""
    r18a = route_command("افتح سبوتيفاي وشغّل الموسيقى")
    assert r18a.success and "COMMAND_CHAIN" in str(r18a)
    
    r18b = route_command("فتح YouTube والموسيقى")
    assert r18b.success and "COMMAND_CHAIN" in str(r18b)

# ===== P5.1: Window Minimize vs Resize =====
def test_window_minimize_english():
    """Test window minimize in English"""
    r19 = route_command("Minimize window")
    assert r19.success and "window_minimize" in str(r19)

def test_window_minimize_arabic():
    """Test window minimize in Egyptian Arabic"""
    r20 = route_command("صغّر الشبابك")
    assert r20.success and "window_minimize" in str(r20)

def test_window_resize_english():
    """Test window resize in English"""
    r21 = route_command("Resize window smaller")
    assert r21.success and "window_resize" in str(r21)

def test_window_resize_arabic():
    """Test window resize in Egyptian Arabic"""
    r22a = route_command("صغّر الشبابك أكتر")
    assert r22a.success and "window_resize" in str(r22a)
    
    r22b = route_command("اخفّف حجم الشبابك")
    assert r22b.success and "window_resize" in str(r22b)

# ===== P5.3: Email Fallback =====
def test_email_english():
    """Test email in English"""
    r23 = route_command("Open email")
    # Should open Outlook or Gmail, one of them
    assert r23.success

def test_email_arabic():
    """Test email in Egyptian Arabic"""
    r24a = route_command("اكتب بريد")
    assert r24a.success
    
    r24b = route_command("افتح الإيميل")
    assert r24b.success
```

**Run:**
```bash
pytest tests/phase5_production_fixes.py -v --tb=short
```

**Coverage:** ✅ All 11 issues × 2+ languages = 25+ test cases

**Effort:** 4 hours (comprehensive bilingual test suite)  
**Risk:** Low (test-only code)

---

### P6.2: Performance Validation
**Measure latency before/after for:**
- Numeric extraction (should be <10ms faster due to simpler regex)
- Auto-discovery (should be <500ms on SSD, cached)
- Media control (should be <50ms, no delay added)

---

## PHASE 7: Deployment & Monitoring (Day 15)
### Objective: Release and track metrics

### P7.1: Feature Flags (Optional, for gradual rollout)
```python
FEATURE_FLAGS = {
    "NUMERIC_PARSING_ENABLED": True,
    "AUTO_APP_DISCOVERY_ENABLED": True,
    "MEDIA_DIRECT_DISPATCH_ENABLED": True,
    "SYSTEM_VOLUME_CONTROL": True,
}
```

### P7.2: Logging & Telemetry
Add structured logs for each fixed path:
```python
log_structured(
    "dnd_toggle_executed",
    action_key="dnd_on",
    success=True,
    latency_ms=50,
)
```

Monitor in [core/metrics.py](core/metrics.py) for success rates post-deployment.

---

## Summary Table

| Phase | Days | Issues Covered | Effort | Risk |
|-------|------|-----------------|--------|------|
| P0 | 1 | Root cause analysis | 4h | Low |
| P1 | 2-3 | DND, RunningApps, FileList, Clipboard | 6h | Low |
| P2 | 4-5 | Numeric extraction, Volume scope | 5h | Med |
| P3 | 6-7 | Timer/duration, Slot filling | 4h | Low |
| P4 | 8-9 | Media dispatch, Chained commands | 4.5h | Med |
| P5 | 10-11 | Window sizing, App discovery, Email fallback | 5.5h | Med |
| P6 | 12-14 | Testing & QA | 3h | Low |
| P7 | 15 | Deployment & monitoring | 1h | Low |
| **TOTAL** | **15** | **All 11 issues** | **~33h** | **Low-Med** |

---

## Implementation Order (Recommended)

### Week 1: Foundation (P0 + P1)
- Days 1–3: Root cause analysis + Quick wins (4 fixes)
- Benefit: 50% of reported issues resolved
- Risk: Minimal

### Week 2: Core Fixes (P2 + P3)
- Days 4–7: Numeric parsing, timer/duration, slot filling
- Benefit: Major UX improvement for brightness/volume/timer
- Risk: Medium (requires careful testing)

### Week 3: Advanced (P4 + P5 + P6 + P7)
- Days 8–15: Media chains, window control, app discovery, deployment
- Benefit: Complete feature set + auto-discovery
- Risk: Medium to Low (staggered rollout)

---

## Code Location Reference

| Module | Issues | Fix Lines |
|--------|--------|-----------|
| [core/command_parser.py](core/command_parser.py) | P1.1, P1.2, P1.3, P2.1, P3.1, P5.1, P5.2 | 1600–2550, 780–950 |
| [core/command_router.py](core/command_router.py) | P1.3, P2.1, P3.1, P3.2, P4.2, P5.2 | 3540–4020, 3880–3920 |
| [core/intent_confidence.py](core/intent_confidence.py) | P1.3 | 200–350 |
| [core/orchestrator.py](core/orchestrator.py) | P5.2 | Unknown (startup) |
| [os_control/system_ops.py](os_control/system_ops.py) | P1.1, P1.2, P2.1, P2.2, P4.1, P5.1 | Unknown (handlers) |
| [os_control/timer_ops.py](os_control/timer_ops.py) | P3.2 | Unknown |
| [os_control/email_ops.py](os_control/email_ops.py) | P5.3 | Unknown |
| [os_control/clipboard_ops.py](os_control/clipboard_ops.py) | P1.4 | Unknown |
| [os_control/app_ops.py](os_control/app_ops.py) | P5.2 | Unknown |
| [os_control/app_discovery.py](os_control/app_discovery.py) | P5.2 | New file (~120 lines) |
| [tests/phase5_production_fixes.py](tests/phase5_production_fixes.py) | P6.1 | New file (~150 lines) |

---

## Success Criteria

### Minimum Viable (all P1 complete):
- ✓ Don't Disturb mode toggles
- ✓ Running apps lists without clarification
- ✓ File list shows current directory
- ✓ Clipboard content is readable (no garbled context)

### Standard (P1 + P2 + P3 complete):
- ✓ Brightness/volume set to exact values (not just increment)
- ✓ System volume controlled (not app volume only)
- ✓ Timer sets directly without clarification loops
- ✓ Duration units (minutes/seconds) parsed correctly

### Complete (all phases complete):
- ✓ Media controls work reliably
- ✓ Chained commands (open + play) work
- ✓ Window sizing distinguishes from minimize
- ✓ Unmapped apps discoverable via auto-scan
- ✓ Email fallback to Gmail works
- ✓ All fixes tested with zero regressions

---

## ✅ BILINGUAL COVERAGE VERIFICATION CHECKLIST

This plan is **FULLY BILINGUAL** (English + Egyptian Arabic) across all phases:

### Phase 1: Quick Wins
- ✅ P1.1 Don't Disturb: EN (enable/disable/turn on/off) + AR (وضع/فعّل/قطّع/طفّي)
- ✅ P1.2 Running Apps: EN (list/show/what's) + AR (التطبيقات/البرامج/إيه)
- ✅ P1.3 File List: EN (list/show/directory) + AR (وريني/افتح/الملفات)
- ✅ P1.4 Clipboard: EN (what's/read) + AR (إيه/المنسوخ/Clipboard)

### Phase 2: Numeric Extraction
- ✅ P2.1 Brightness: EN (set to/adjust) + AR (اخفض/زود/وضّع + numeric extraction)
- ✅ P2.2 Volume: EN (set volume) + AR (ارفع/اخفض/قلّل + numeric extraction)
- ✅ Both include system volume control, not app-only
- ✅ Test cases: 6+ combinations per language

### Phase 3: Duration & Timers
- ✅ P3.1 Duration Parsing: EN (minutes/seconds/hours) + AR (دقيقة/ثانية/ساعة)
- ✅ P3.2 Timer Direct Execution: EN (set timer for) + AR (حط timer على)
- ✅ Slot filler handles both languages for fallback fill-in
- ✅ Test cases: Direct execution + slot fill flow for both languages

### Phase 4: Media & Chains
- ✅ P4.1 Media Controls: EN (next/play/pause) + AR (الأغنية اللي بعد/شغّل)
- ✅ P4.2 Chained Commands: EN (open X and play) + AR (افتح X وشغّل)
- ✅ Test cases: Both languages with real Spotify/YouTube testing

### Phase 5: Advanced Features
- ✅ P5.1 Window Control: EN (minimize/resize) + AR (صغّر/اطوّي + modifier for resize)
- ✅ P5.2 App Discovery: Commands in both EN (rescan apps) + AR (اسكن البرامج)
- ✅ P5.3 Email Fallback: EN (open email) + AR (اكتب بريد/افتح الإيميل)

### Phase 6: Testing
- ✅ **25+ bilingual test cases** (minimum 2+ per issue in both languages)
- ✅ Each test verifies BOTH English and Egyptian Arabic separately
- ✅ No cross-language contamination expected

### Phase 7: Deployment
- ✅ Bilingual responses in all handler output
- ✅ Logging includes language tag for debugging
- ✅ Metrics tracked per language

---

## Language Dialectics Notes

**Egyptian Colloquial Arabic (عامية مصرية) Patterns Used:**

1. **Do Not Disturb** (DND):
   - Formal: عدم الإزعاج
   - Colloquial: ساكت، صمت، سكوت
   - Common: "وضع عدم الإزعاج" (most natural)

2. **Apps/Programs:**
   - Technical: البرامج، التطبيقات
   - Colloquial: الحاجات الشغالة، البرامج الشغالة

3. **Brightness/Volume:**
   - Technical: السطوع، الصوت
   - Actions: اخفض (lower), زود (increase), قلّل (reduce)
   - Set: وضّع على، ل (to/at)

4. **Timer/Duration:**
   - Technical: دقيقة، ثانية، ساعة
   - Colloquial: اتنين دقيقة (two minutes, common pattern)

5. **Commands:**
   - Imperative: افتح (open), شغّل (play), وريني (show me)
   - Modern: "و" (and) for chaining, "إيه" (what) for questions

6. **Window Control:**
   - صغّر (minimize, reduce size)
   - اطوّي (fold, minimize)
   - أكتر (more, used as modifier: صغّر أكتر = resize more)

---

## Notes for Codex/Developer

1. **Start with P1**: These are high-impact, low-risk fixes that will make immediate difference.
2. **Test early**: Run regression suite after each phase.
3. **Arabic variants**: Many patterns need both English and Arabic versions (already shown in examples).
4. **Handler verification**: Before assuming missing, search [os_control/system_ops.py](os_control/system_ops.py) for existing implementation.
5. **Timing delays**: For chained commands, use `time.sleep()` conservatively (2s default for app open).
6. **Fallback safety**: Always provide user-facing fallback when primary path fails (e.g., Gmail if Outlook fails).
7. **Dependency additions**: `pycaw` (audio), `pynput` (keyboard) — add to requirements.txt, test on clean environment.
