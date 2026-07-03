import glob
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.streaming_stt import prewarm_streaming_vad, record_utterance_streaming
from audio import stt as stt_runtime
from audio.stt import transcribe_streaming
from audio.tts import speech_engine
from audio.vad import is_speech, prewarm_batch_vad
from audio.wake_word import (
    get_last_detection_audio,
    listen_for_wake_word,
    preload_runtime_wake_word,
)
from core.adaptive_wake import (
    record_confirmed as _adaptive_record_confirmed,
    record_false_positive as _adaptive_record_false_positive,
    start_daemon as _adaptive_start_daemon,
    stop_daemon as _adaptive_stop_daemon,
)
from core.command_parser import parse_command
from core.command_router import (
    initialize_command_services,
    route_command,
    inject_precomputed_live_context,
    clear_precomputed_live_context,
    looks_like_live_data_query,
    prime_llm_response_cache_async,
)
from core.knowledge_base import knowledge_base_service
from core.doctor import collect_diagnostics
from core.config import (
    DEMO_MODE,
    EARLY_EXEC_CONFIDENCE_THRESHOLD,
    DOCTOR_INCLUDE_MODEL_LOAD_CHECKS,
    DOCTOR_SCHEDULE_INTERVAL_SECONDS,
    DOCTOR_STARTUP_ASYNC,
    DOCTOR_STARTUP_ENABLED,
    FOLLOWUP_CHIME_ENABLED,
    FOLLOWUP_ENABLED,
    SENSITIVE_CONFIRM_MODE,
    GREETING_ENABLED,
    GREETING_LANGUAGE,
    GREETING_TEXT_AR,
    GREETING_TEXT_EN,
    GREETING_PRESPEAK_SETTLE_MS,
    GREETING_DEVICE_WARMUP,
    LLM_PREWARM_BEFORE_GREETING,
    LLM_AUTO_SELECT_MODEL,
    LLM_MODEL,
    LLM_OLLAMA_AUTOSTART,
    LLM_OLLAMA_AUTOSTART_TIMEOUT_SECONDS,
    LLM_OLLAMA_BASE_URL,
    LLM_OLLAMA_EXECUTABLE,
    LLM_LIGHTWEIGHT_NUM_CTX,
    LLM_OLLAMA_NUM_CTX,
    LLM_TIMEOUT_SECONDS,
    MAX_RECORD_DURATION,
    PREWARM_LLM_BLOCKING,
    PREWARM_SEMANTIC_ROUTER_BLOCKING,
    REALTIME_BACKPRESSURE_POLL_SECONDS,
    REALTIME_DROP_WHEN_BUSY,
    REALTIME_MAX_PENDING_UTTERANCES,
    SPEECH_GUARD_SKIP_NON_RESPONSIVE_PROFILES,
    SEMANTIC_ROUTER_ENABLED,
    STT_PARTIAL_WHISPER_MODEL,
    STT_LANGUAGE_HINT,
    STARTUP_PARSER_NLP_PREWARM_ENABLED,
    STARTUP_BACKGROUND_PREWARM_ENABLED,
    TTS_PREWARM_ENABLED,
    TTS_DEFAULT_BACKEND,
    WHISPER_MODEL,
    KB_AUTO_SYNC_ENABLED,
    APP_SCAN_ON_STARTUP,
    APP_WATCH_STARTMENU,
)
from core.dialogue_manager import DialogueState, dialogue_manager, notify_follow_up_wake
from core.intent_confidence import assess_intent_confidence
from core.logger import get_logger, kv, logger, section
from core.language_gate import detect_supported_language, looks_romanized_arabic
from core.metrics import (
    get_thread_stage_timing,
    latency_tracker,
    log_turn_summary,
    log_warmup_table,
    metrics,
    record_stage_timing,
    reset_thread_stage_timings,
    stage_timer,
)
from core.runtime_coordinator import RuntimePhase, coordinator
from core.session_memory import session_memory
from core.shutdown import perform_shutdown_cleanup, setup_shutdown
from os_control.confirmation import confirmation_manager

try:
    from tools.live_data import gather_live_data as _gather_live_data
except Exception:  # pragma: no cover
    _gather_live_data = None


_INTERRUPT_COMMANDS = {
    "stop speaking",
    "interrupt speech",
    "be quiet",
    "stop talking",
}

_LOW_LATENCY_AUDIO_UX_PROFILES = {"responsive"}
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_LAST_STT_LANGUAGE_CONFIDENCE = 0.0
_OLLAMA_AUTOSTART_PROCESS = None

# Match transcripts that are pure non-speech annotations from ElevenLabs/Whisper
# (e.g. "[صوت انطلاق سيارة]", "[music]", "[laughter]", "(silence)") — these
# should never reach the parser/clarification system. Several common forms.
_STT_ANNOTATION_RE = re.compile(
    r"^\s*[\[\(\<]\s*[^\]\)\>]{0,80}\s*[\]\)\>]\s*\.?\s*$",
    re.UNICODE,
)

# Task 1.3 — Concurrent Pipeline ─────────────────────────────────────────────

# Intents safe to execute on a partial transcript before STT finalises.
# Only simple, non-destructive, instant commands — no LLM round-trip needed and
# the full utterance cannot change the meaning (e.g. no duration needed).
_EARLY_EXECUTABLE_INTENTS = frozenset({
    "OS_APP_OPEN",
    "OS_APP_CLOSE",
    "OS_SYSTEM_COMMAND",
    "VOICE_COMMAND",
})

# OS_SYSTEM_COMMAND actions that must never execute early — irreversible or risky.
_EARLY_EXEC_DANGEROUS_ACTIONS = frozenset({
    "shutdown", "restart", "sleep", "lock", "logoff",
})

# Minimum confidence from assess_intent_confidence before committing to early exec.
# Loaded from config so it can be tuned via JARVIS_EARLY_EXEC_CONFIDENCE_THRESHOLD.
_EARLY_EXEC_CONFIDENCE_THRESHOLD = float(EARLY_EXEC_CONFIDENCE_THRESHOLD)

# Minimum word count in a partial before attempting early intent detection.
_EARLY_INTENT_MIN_WORDS = 3

# Lightweight live-data keyword list for pre-fetch triggering during recording.
_LIVE_PREFETCH_KEYWORDS = frozenset({
    "weather", "temperature", "forecast", "طقس", "حرارة", "درجة",
    "news", "أخبار", "اخبار",
    "price", "stock", "سعر",
    "latest", "current", "today", "النهارده", "دلوقتي",
    "search for", "look up", "ابحث", "ابحث عن",
})


class ConcurrentPipeline:
    """Event-driven pipeline for one utterance.

    Created before ``record_utterance_streaming()`` and kept alive until
    ``_process_utterance()`` finishes.  Two concurrent win-paths:

    1. **Early command execution**: When a partial transcript resolves to a
       high-confidence direct OS command (non-LLM intent) and that intent is
       stable across two consecutive partials, we execute the command
       immediately — before recording even finishes — and speak the response.
       ``_process_utterance`` then skips routing for that utterance.

    2. **Live data pre-fetch**: When live-data keywords appear in a partial we
       submit ``gather_live_data`` to a background thread.  The result is
       injected into the command router thread-local so that ``_fetch_live_tool_context``
       returns the cached result instead of making a new network call, shaving
       ~1 s off LLM responses that need weather / search context.
    """

    def __init__(self, executor, *, language_hint: str = ""):
        self._executor = executor
        self._language_hint = str(language_hint or "")
        self._lock = threading.Lock()

        self._early_executed = False
        self._early_response = ""
        self._early_response_spoken = False  # NEW: Track if we've already spoken the early response
        self._early_intent_str: "str | None" = None
        self._early_timings = {"route": 0.0, "tts": 0.0}
        self._early_execution_future = None
        self._prev_intent = None

        self._live_future = None

    # ── Public interface used by the main loop / _process_utterance ──────────

    def on_partial(self, partial_text: str) -> None:
        """Receive incremental STT text emitted during recording."""
        if not partial_text:
            return
        normalized = " ".join(str(partial_text).split()).strip()
        if len(normalized.split()) < _EARLY_INTENT_MIN_WORDS:
            return

        with self._lock:
            if self._early_executed:
                return

        self._maybe_prefetch_live_data(normalized)
        self._maybe_early_execute(normalized)

    def is_early_executed(self) -> bool:
        with self._lock:
            return self._early_executed

    def get_early_response(self) -> str:
        with self._lock:
            return self._early_response

    def get_live_context(self, timeout: float = 0.15) -> str:
        """Return pre-fetched live data if the future is already resolved."""
        with self._lock:
            future = self._live_future
        if future is None:
            return ""
        try:
            result = future.result(timeout=timeout)
            return str(result or "")
        except Exception:
            return ""

    def get_early_intent_str(self) -> "str | None":
        with self._lock:
            return self._early_intent_str

    def get_early_timings(self) -> dict:
        with self._lock:
            return dict(self._early_timings)

    def is_early_response_spoken(self) -> bool:
        """Check if early response has already been spoken."""
        with self._lock:
            return self._early_response_spoken

    def cancel_early_if_possible(self) -> None:
        """Best-effort cancel. Resets early-executed flag so _process_utterance re-routes."""
        with self._lock:
            future = self._early_execution_future
            self._early_executed = False
            self._early_intent_str = None
            self._early_response_spoken = False
        if future is not None and not future.done():
            future.cancel()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _maybe_prefetch_live_data(self, text: str) -> None:
        if _gather_live_data is None:
            return
        with self._lock:
            if self._live_future is not None:
                return
        text_lower = text.lower()
        if not any(kw in text_lower for kw in _LIVE_PREFETCH_KEYWORDS):
            if not looks_like_live_data_query(text):
                return
        with self._lock:
            if self._live_future is not None:
                return
            try:
                self._live_future = self._executor.submit(
                    _gather_live_data, text, parallel=True
                )
                logger.debug("ConcurrentPipeline: live-data pre-fetch started for '%s'", text[:80])
            except Exception as exc:
                logger.debug("ConcurrentPipeline: live-data pre-fetch submit failed: %s", exc)

    def _maybe_early_execute(self, text: str) -> None:
        try:
            parsed = parse_command(text)
        except Exception:
            return

        intent = str(getattr(parsed, "intent", "") or "")
        if intent not in _EARLY_EXECUTABLE_INTENTS:
            with self._lock:
                self._prev_intent = None
            return

        # Dangerous action guard — never fire early for irreversible OS actions.
        # Check both the parsed.action field and parsed.args['action_key'] because
        # the command parser stores the semantic action key in args for OS_SYSTEM_COMMAND.
        action = str(getattr(parsed, "action", "") or "").strip().lower()
        action_key = str((getattr(parsed, "args", None) or {}).get("action_key") or "").strip().lower()
        if action in _EARLY_EXEC_DANGEROUS_ACTIONS or action_key in _EARLY_EXEC_DANGEROUS_ACTIONS:
            with self._lock:
                self._prev_intent = None
            return

        # High-confidence gate — partial must be unambiguous before we commit.
        try:
            lang = str(self._language_hint or "en").strip().lower()
            if lang not in {"ar", "en"}:
                lang = "en"
            assessment = assess_intent_confidence(text, parsed, language=lang)
            confidence = float(getattr(assessment, "confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        if confidence < _EARLY_EXEC_CONFIDENCE_THRESHOLD:
            with self._lock:
                self._prev_intent = None
            return

        with self._lock:
            prev = self._prev_intent
            self._prev_intent = intent
            if prev != intent:
                # First occurrence — wait for a second consecutive partial with
                # the same intent before committing to early execution.
                return
            if self._early_executed:
                return
            self._early_executed = True  # Reserve — prevents double-execution
            self._early_intent_str = intent

        try:
            future = self._executor.submit(self._run_early_command, text, parsed)
            with self._lock:
                self._early_execution_future = future
        except Exception as exc:
            with self._lock:
                self._early_executed = False
                self._early_intent_str = None
            logger.debug("ConcurrentPipeline: early-execute submit failed: %s", exc)

    def _run_early_command(self, text: str, parsed) -> None:
        """Execute the command from a background thread and speak the response."""
        lang = self._language_hint or session_memory.get_preferred_language() or ""
        try:
            route_started = time.perf_counter()
            response = route_command(
                text,
                detected_language=lang or None,
                realtime=True,
            )
            route_elapsed = time.perf_counter() - route_started
            record_stage_timing("routing", route_elapsed, intent=getattr(parsed, "intent", "unknown"))
            if response:
                safe = _speech_safe_response(response)
                if safe:
                    with stage_timer("tts_first_word", lang=lang or "unknown") as tts_timing:
                        speech_engine.speak_async(safe, language=lang or None)
                with self._lock:
                    self._early_response = response
                    self._early_response_spoken = True  # Mark as spoken to prevent duplicate TTS
                    self._early_timings = {
                        "route": route_elapsed,
                        "tts": tts_timing.elapsed if safe else 0.0,
                    }
                metrics.record_stage("early_execute", 0.0, success=True)
                logger.info(
                    "ConcurrentPipeline: early-executed %s on partial '%s'",
                    getattr(parsed, "intent", "?"),
                    text[:60],
                )
        except Exception as exc:
            with self._lock:
                self._early_executed = False  # Allow _process_utterance to retry
            logger.debug("ConcurrentPipeline: early command failed, will retry in full pipeline: %s", exc)

# ─────────────────────────────────────────────────────────────────────────────


def _is_stt_annotation_only(text):
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return False
    return bool(_STT_ANNOTATION_RE.match(value))


def _resolve_stt_language_hint(*, wake_source=None):
    hint = str(STT_LANGUAGE_HINT or "auto").strip().lower()
    if hint in {"ar", "arabic"}:
        return "ar"
    if hint in {"en", "english"}:
        return "en"

    preferred = str(session_memory.get_preferred_language() or "").strip().lower()
    if preferred in {"ar", "en"}:
        return preferred

    # Default to auto: the STT layer picks and locks ar/en using streaming
    # text, this hint, or a tiny one-second probe before full transcription.
    return "auto"


def _is_interrupt_command(text):
    return (text or "").strip().lower() in _INTERRUPT_COMMANDS


def _speech_safe_response(text):
    marker = "\nSources:"
    content = (text or "")
    idx = content.find(marker)
    if idx >= 0:
        content = content[:idx]
    return content.strip()


def _remaining_after_streamed_sentences(full_text, streamed_sentences):
    normalized_full = " ".join(str(full_text or "").split()).strip()
    if not normalized_full:
        return ""

    normalized_streamed = [
        " ".join(str(sentence or "").split()).strip()
        for sentence in (streamed_sentences or [])
        if str(sentence or "").strip()
    ]
    if not normalized_streamed:
        return normalized_full

    prefix = " ".join(normalized_streamed).strip()
    if not prefix:
        return normalized_full
    if normalized_full.startswith(prefix):
        return normalized_full[len(prefix):].strip()
    return ""


def _create_utterance_audio_file():
    fd, path = tempfile.mkstemp(prefix="jarvis_utterance_", suffix=".wav")
    try:
        return path
    finally:
        try:
            import os

            os.close(fd)
        except Exception:
            pass


def _safe_remove(path):
    if not path:
        return
    try:
        import os

        if os.path.exists(path):
            os.remove(path)
    except Exception as exc:
        logger.warning("Failed to remove temp audio file %s: %s", path, exc)


def _prune_futures(futures):
    active = []
    for future in futures:
        if future.done():
            try:
                future.result()
            except Exception as exc:
                logger.error("Utterance worker failed: %s", exc)
        else:
            active.append(future)
    return active


def _emit_ui_event(event_type, **fields):
    """Best-effort broadcast to connected desktop UI clients via the optional bridge.

    No-ops when the UI bridge isn't running/available; never raises into the
    voice pipeline.
    """
    try:
        from ui.bridge import broadcast_event
        from ui.events import make_event

        broadcast_event(make_event(event_type, **fields))
    except Exception:
        logger.debug("UI event emit failed: type=%s", event_type, exc_info=True)


def _guess_language(text):
    return "ar" if _ARABIC_CHAR_RE.search(str(text or "")) else "en"


def _on_partial_transcript(partial_text):
    if partial_text:
        logger.debug("STT partial: %s", partial_text[-180:])
        _emit_ui_event(
            "partial_transcript",
            text=str(partial_text),
            language=_guess_language(partial_text),
        )


def _safe_log_text(text, max_chars=220):
    value = " ".join((text or "").split())
    if len(value) > max_chars:
        value = value[: max_chars - 3] + "..."
    return value


def _extract_detected_language_from_stt(text):
    global _LAST_STT_LANGUAGE_CONFIDENCE
    stt_meta = stt_runtime.get_last_transcription_meta()
    detected_language = str((stt_meta or {}).get("language") or "").strip().lower()
    if looks_romanized_arabic(text):
        detected_language = "ar"
    try:
        _LAST_STT_LANGUAGE_CONFIDENCE = float((stt_meta or {}).get("language_confidence") or 0.0)
    except (TypeError, ValueError):
        _LAST_STT_LANGUAGE_CONFIDENCE = 0.0
    _LAST_STT_LANGUAGE_CONFIDENCE = max(0.0, min(1.0, _LAST_STT_LANGUAGE_CONFIDENCE))
    if detected_language not in {"ar", "en"} and text:
        detected_language = detect_supported_language(
            text,
            previous_language="",
        ).language
    if detected_language in {"ar", "en"}:
        return detected_language
    return ""


def _transcribe_with_runtime_stt(audio_file, wake_source=None, streaming_text=""):
    global _LAST_STT_LANGUAGE_CONFIDENCE
    text = transcribe_streaming(
        audio_file,
        on_partial=_on_partial_transcript,
        language_hint=_resolve_stt_language_hint(wake_source=wake_source),
        streaming_text=str(streaming_text or ""),
    )
    _LAST_STT_LANGUAGE_CONFIDENCE = 0.0
    detected_language = _extract_detected_language_from_stt(text)
    detected_language_confidence = float(_LAST_STT_LANGUAGE_CONFIDENCE or 0.0)
    _ = detected_language_confidence
    return text, detected_language


def _precompute_post_stt_routing(text, *, detected_language=None):
    normalized_text = " ".join(str(text or "").split()).strip()
    if not normalized_text:
        return None, None

    forced_language = str(detected_language or "").strip().lower()
    if forced_language not in {"ar", "en"}:
        forced_language = ""
    previous_language = forced_language or session_memory.get_preferred_language()

    try:
        def _timed_parse(text):
            with stage_timer("intent_detection"):
                return parse_command(text)

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="jarvis-route-precompute") as executor:
            language_future = executor.submit(
                detect_supported_language,
                normalized_text,
                previous_language=previous_language,
            )
            parser_future = executor.submit(_timed_parse, normalized_text)
            language_result = language_future.result()
            parser_candidate = parser_future.result()
    except Exception as exc:
        logger.debug("Routing precompute failed; falling back to route-time parse: %s", exc)
        return None, None

    gated_text = " ".join(
        str(getattr(language_result, "normalized_text", "") or normalized_text).split()
    ).strip()
    if gated_text and gated_text != normalized_text:
        try:
            parser_candidate = parse_command(gated_text)
        except Exception as exc:
            logger.debug("Routing precompute parser re-run failed: %s", exc)

    return language_result, parser_candidate


_EN_COMPOUND_VERBS = frozenset({
    "open", "close", "find", "search", "tell", "show", "play", "pause", "stop",
    "set", "turn", "check", "get", "look", "create", "delete", "move", "copy",
    "rename", "maximize", "minimize", "snap", "lock", "sleep", "restart",
    "shut", "take", "launch", "start", "run", "navigate", "go", "scroll",
    "what", "who", "how", "when", "where", "why",
})

_AR_COMPOUND_PREFIXES = (
    "افتح", "اغلق", "أغلق", "ابحث", "دور", "خبرني", "قولي", "شغّل", "شغل",
    "وقف", "اعمل", "صور", "نزل", "حمل", "اقفل", "ابدأ", "اطفي", "اطفى",
    "ايه", "مين", "كيف",
)
# Matches waw-conjunction prefix attached directly to a command verb, e.g. "وقولي", "وافتح".
# Stripping the leading "و" from group(1) gives the clean sub-command.
_AR_WAW_COMPOUND_RE = re.compile(
    r'\s+(و(?:' + '|'.join(re.escape(p) for p in _AR_COMPOUND_PREFIXES) + r'))'
)


def _split_compound_utterance(text):
    """Split 'open X and do Y' into ['open X', 'do Y']. Returns [text] if not compound."""
    text = text.strip()
    if not text:
        return [text]

    # Arabic sequential connectors with وـ prefix: وبعدين, وكمان, وبعد كده, وبعد ذلك
    ar_then = re.search(r'\s*و(?:بعدين|كمان|بعد\s+كده|بعد\s+ذلك)\s+', text)
    if ar_then:
        before = text[:ar_then.start()].strip()
        after = text[ar_then.end():].strip()
        if before and len(after.split()) >= 2:
            return [before, after]

    # Arabic standalone sequential: ثم / بعدين / بعد كده / بعد ذلك (no و prefix)
    ar_standalone = re.search(r'\s+(?:ثم|بعدين|بعد\s+كده|بعد\s+ذلك)\s+', text)
    if ar_standalone:
        before = text[:ar_standalone.start()].strip()
        after = text[ar_standalone.end():].strip()
        if before and len(after.split()) >= 1:
            return [before, after]

    # English "and then" / "then"
    en_then = re.search(r'\s+(?:and\s+)?then\s+', text, re.IGNORECASE)
    if en_then:
        before = text[:en_then.start()].strip()
        after = text[en_then.end():].strip()
        if before and len(after.split()) >= 2:
            return [before, after]

    # Arabic waw + command verb: وقولي, وافتح, وابحث, ...
    # The leading "و" is the conjunction; strip it to get the clean sub-command.
    ar_waw = _AR_WAW_COMPOUND_RE.search(text)
    if ar_waw:
        before = text[:ar_waw.start()].strip()
        waw_token = ar_waw.group(1)       # e.g. "وقولي"
        after_verb = waw_token[1:]         # strip "و" → "قولي"
        after_rest = text[ar_waw.end():]  # remaining text after the token
        after = (after_verb + after_rest).strip()
        if before and len(after.split()) >= 1:
            return [before, after]

    # English "and [command-verb]" — split only when the post-and clause starts a new command
    and_match = re.search(r'\s+and\s+', text, re.IGNORECASE)
    if and_match:
        before = text[:and_match.start()].strip()
        after = text[and_match.end():].strip()
        if before and after:
            after_words = after.split()
            first_word = after_words[0].lower() if after_words else ""
            if first_word in _EN_COMPOUND_VERBS and len(after_words) >= 2:
                return [before, after]
            # Handle "I want (you) to X", "I need X", "please X" lead-ins
            filler = re.match(
                r'^(?:i\s+(?:want|need|would\s+like)(?:\s+you)?\s+(?:to\s+)?|please\s+)',
                after, re.IGNORECASE,
            )
            if filler:
                remainder = after[filler.end():].strip().split()
                if remainder and remainder[0].lower() in _EN_COMPOUND_VERBS and len(remainder) >= 2:
                    return [before, after]
            if any(after.startswith(ar_prefix) for ar_prefix in _AR_COMPOUND_PREFIXES):
                return [before, after]

    return [text]


def _run_text_fallback_loop():
    print("Jarvis is running in text fallback mode (no wake-word/audio stack).")
    print("Type 'exit' to stop.")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("")
            return
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            return

        route_started = time.perf_counter()
        try:
            response = route_command(text)
            metrics.record_stage("router_text", time.perf_counter() - route_started, success=True)
        except Exception as exc:
            metrics.record_stage("router_text", time.perf_counter() - route_started, success=False)
            logger.error("Text-mode command routing failed: %s", exc)
            response = "Sorry, I had an internal error."

        print(f"Jarvis: {response}")
        if not _is_interrupt_command(text):
            speech_engine.speak_async(
                _speech_safe_response(response),
                language=session_memory.get_preferred_language(),
            )


def _process_utterance(
    audio_file,
    pipeline_started,
    wake_source=None,
    capture_summary=None,
    pipeline=None,
    turn_timing=None,
):
    text = ""
    route_success = False
    detected_language = ""
    turn_intent = ""
    timing_parts = dict(turn_timing or {})
    reset_thread_stage_timings()
    for stage_name in ("stt", "route", "llm", "tts"):
        timing_parts.setdefault(stage_name, 0.0)
    try:
        active_audio_ux_profile = str(session_memory.get_audio_ux_profile() or "").strip().lower()
        skip_post_capture_guard = active_audio_ux_profile in _LOW_LATENCY_AUDIO_UX_PROFILES
        capture_detected_speech = bool((capture_summary or {}).get("speech_detected"))
        if capture_detected_speech:
            skip_post_capture_guard = True
        elif not skip_post_capture_guard and bool(SPEECH_GUARD_SKIP_NON_RESPONSIVE_PROFILES):
            skip_post_capture_guard = True

        if skip_post_capture_guard:
            # record_utterance already runs mic VAD; skip duplicate file-based guard in fast profile.
            metrics.record_stage("speech_guard", 0.0, success=True)
        else:
            speech_guard_started = time.perf_counter()
            try:
                looks_like_speech = bool(is_speech(audio_file))
            except Exception as exc:
                logger.warning("Speech guard failed; continuing with STT: %s", exc)
                looks_like_speech = True
            metrics.record_stage("speech_guard", time.perf_counter() - speech_guard_started, success=looks_like_speech)
            if not looks_like_speech:
                logger.warning("Captured audio appears to be non-speech noise; skipping STT")
                return

        dialogue_manager.transition(DialogueState.PROCESSING)
        coordinator.set_phase(RuntimePhase.TRANSCRIBING)

        # ── Task 1.1: streaming STT fast-path ────────────────────────────────
        # StreamingSTT transcribes the audio *during* recording and stores the
        # final result in capture_summary["text"].  Reuse it to skip the second
        # full STT pass that used to add 1–2 s of dead time after speech ended.
        stt_started = time.perf_counter()
        _streaming_text = str((capture_summary or {}).get("text", "") or "").strip()
        _streaming_lang = str((capture_summary or {}).get("language", "") or "").strip().lower()

        if _streaming_text:
            text = _streaming_text
            detected_language = _streaming_lang if _streaming_lang in {"ar", "en"} else ""
            if not detected_language:
                # Language field absent or unknown — infer from text content.
                detected_language = _extract_detected_language_from_stt(text)
            _stt_duration = time.perf_counter() - stt_started
            metrics.record_stage("stt", _stt_duration, success=True)
            record_stage_timing("stt_total", _stt_duration, lang=detected_language or "unknown")
            logger.debug("STT fast-path: reusing streaming transcript (skipped redundant STT pass)")
        else:
            # Fallback: no streaming transcript available (e.g. sounddevice unavailable,
            # or capture returned empty text). Run standard batch STT.
            text, detected_language = _transcribe_with_runtime_stt(
                audio_file,
                wake_source=wake_source,
                streaming_text=_streaming_text,
            )
            _stt_duration = time.perf_counter() - stt_started
            metrics.record_stage("stt", _stt_duration, success=bool(text))
            record_stage_timing("stt_total", _stt_duration, lang=detected_language or "unknown")
        timing_parts["stt"] = _stt_duration
        # ─────────────────────────────────────────────────────────────────────

        if detected_language in {"ar", "en"}:
            session_memory.set_preferred_language(detected_language)
            session_memory.record_language_turn(detected_language)
        if not text:
            logger.warning("No valid speech detected")
            return
        # Skip non-speech annotation-only transcripts (e.g. "[صوت انطلاق سيارة]",
        # "[music]") before they reach the parser and trigger a clarification.
        if _is_stt_annotation_only(text):
            logger.info("Skipping non-speech STT annotation: %s", _safe_log_text(text))
            return
        logger.info("Transcript[%s]: %s", detected_language or "unknown", _safe_log_text(text))
        _emit_ui_event(
            "final_transcript",
            text=str(text),
            language=detected_language or _guess_language(text),
        )



        # Demo mode: print intent/confidence overlay to console for presentations.
        if DEMO_MODE:
            try:
                from core.command_parser import parse_command as _pc
                from core.intent_confidence import assess_intent_confidence as _aic
                _dm_parsed = _pc(text)
                _dm_assess = _aic(text, _dm_parsed, language=detected_language or "en")
                _dm_conf = round(float(getattr(_dm_assess, "confidence", 0.0) or 0.0), 2)
                print(
                    f"\n┌─ DEMO ─────────────────────────────────────────────────────────┐\n"
                    f"│  Transcript : {text[:55]:<55} │\n"
                    f"│  Language   : {detected_language or 'unknown':<55} │\n"
                    f"│  Intent     : {_dm_parsed.intent:<55} │\n"
                    f"│  Confidence : {_dm_conf:<55} │\n"
                    f"└────────────────────────────────────────────────────────────────┘",
                    flush=True,
                )
            except Exception:
                pass

        # ── Tasks 1.3 / 1.4: early-execution fast path with mismatch detection ──
        # If ConcurrentPipeline already executed the command from a high-confidence
        # partial, skip full routing.  But if the final transcript resolves to a
        # DIFFERENT intent the early execution was wrong — cancel what we can and
        # fall through so the correct intent is processed.
        if pipeline is not None and pipeline.is_early_executed():
            early_intent = pipeline.get_early_intent_str()
            try:
                _final_parsed_check = parse_command(text)
                final_intent_check = str(getattr(_final_parsed_check, "intent", "") or "")
                turn_intent = final_intent_check
            except Exception:
                final_intent_check = ""
            if early_intent and final_intent_check and early_intent != final_intent_check:
                logger.warning(
                    "ConcurrentPipeline: intent mismatch — early=%s final=%s; "
                    "cancelling early execution and re-routing.",
                    early_intent,
                    final_intent_check,
                )
                pipeline.cancel_early_if_possible()
                # Fall through to full routing below.
            else:
                early_resp = pipeline.get_early_response()
                if early_resp:
                    print(f"Jarvis (early): {early_resp}")
                    _emit_ui_event(
                        "response",
                        text=str(early_resp),
                        language=detected_language or _guess_language(early_resp),
                    )
                logger.info("ConcurrentPipeline: skipping full route — already handled via early execution.")
                timing_parts.update(pipeline.get_early_timings())
                route_success = True
                return
        # ─────────────────────────────────────────────────────────────────────

        precomputed_language_result, precomputed_parser_candidate = _precompute_post_stt_routing(
            text,
            detected_language=detected_language,
        )
        turn_intent = str(getattr(precomputed_parser_candidate, "intent", "") or "")

        tts_language = detected_language or session_memory.get_preferred_language()
        if _ARABIC_CHAR_RE.search(str(text or "")) or looks_romanized_arabic(text):
            tts_language = "ar"
        should_speak_response = not _is_interrupt_command(text)

        dialogue_manager.transition(DialogueState.RESPONDING)

        sub_commands = _split_compound_utterance(text)
        route_started = time.perf_counter()
        is_compound = len(sub_commands) > 1
        coordinator.set_phase(RuntimePhase.ROUTING)

        # ── Task 1.3: inject pre-fetched live context for LLM queries ─────────
        live_context = pipeline.get_live_context() if pipeline is not None else ""
        if live_context:
            inject_precomputed_live_context(live_context)
        # ─────────────────────────────────────────────────────────────────────

        stream_sentence_queue = None
        stream_sentence_done = object()
        streamed_sentences = []
        streaming_tts_enabled = False

        def _stream_sentence(sentence):
            if stream_sentence_queue is None:
                return
            utterance = " ".join(str(sentence or "").split()).strip()
            if not utterance:
                return
            try:
                stream_sentence_queue.put_nowait(utterance)
                streamed_sentences.append(utterance)
            except Exception:
                pass

        def _finish_streaming_tts(final_response=""):
            if stream_sentence_queue is None:
                return False
            if streamed_sentences:
                remainder = _remaining_after_streamed_sentences(
                    final_response,
                    streamed_sentences,
                )
                if remainder:
                    try:
                        stream_sentence_queue.put_nowait(remainder)
                    except Exception:
                        pass
            try:
                stream_sentence_queue.put_nowait(stream_sentence_done)
            except Exception:
                pass
            # Give the queue consumer a short moment to drain before we fall
            # back to the final response path or shutdown cleanup.
            time.sleep(0.01)
            return bool(streamed_sentences)

        if (
            should_speak_response
            and not is_compound
            and str(getattr(precomputed_parser_candidate, "intent", "") or "") == "LLM_QUERY"
        ):
            stream_sentence_queue = queue.Queue()

            def _sentence_iterator():
                while True:
                    sentence = stream_sentence_queue.get()
                    if sentence is stream_sentence_done:
                        return
                    yield sentence

            coordinator.set_phase(RuntimePhase.SPEAKING)
            tts_started_at = time.perf_counter()
            started, _ = speech_engine.speak_sentence_queue(_sentence_iterator(), language=tts_language)
            tts_start_elapsed = time.perf_counter() - tts_started_at
            if started:
                record_stage_timing("tts_first_word", tts_start_elapsed, lang=tts_language or "unknown")
                timing_parts["tts"] = tts_start_elapsed
            streaming_tts_enabled = bool(started)
            if not streaming_tts_enabled:
                stream_sentence_queue = None

        _is_llm_query = str(getattr(precomputed_parser_candidate, "intent", "") or "") == "LLM_QUERY"
        coordinator.set_phase(RuntimePhase.THINKING if _is_llm_query else RuntimePhase.EXECUTING_COMMAND)

        try:
            if is_compound:
                coordinator.set_phase(RuntimePhase.EXECUTING_COMMAND)
                try:
                    all_responses = []
                    for sub_text in sub_commands:
                        sub_response = route_command(
                            sub_text,
                            detected_language=detected_language,
                            realtime=True,
                        )
                        if sub_response:
                            all_responses.append(sub_response)
                            print(f"Jarvis: {sub_response}")
                    response = " ".join(all_responses).strip() or "Done."
                    route_success = True
                    _route_duration = time.perf_counter() - route_started
                    metrics.record_stage("router", _route_duration, success=True)
                    latency_tracker.record("action_execution", _route_duration)
                    latency_tracker.record("e2e_command", time.perf_counter() - pipeline_started)
                except Exception as exc:
                    metrics.record_stage("router", time.perf_counter() - route_started, success=False)
                    logger.error("Compound command routing failed: %s", exc)
                    response = "Sorry, I had an internal error."
            else:
                try:
                    response = route_command(
                        text,
                        detected_language=detected_language,
                        realtime=True,
                        on_sentence=_stream_sentence if streaming_tts_enabled else None,
                        precomputed_language_result=precomputed_language_result,
                        precomputed_parser_candidate=precomputed_parser_candidate,
                    )
                    route_success = True
                    _route_duration = time.perf_counter() - route_started
                    metrics.record_stage("router", _route_duration, success=True)
                    latency_tracker.record("action_execution", _route_duration)
                    latency_tracker.record("e2e_command", time.perf_counter() - pipeline_started)
                except Exception as exc:
                    metrics.record_stage("router", time.perf_counter() - route_started, success=False)
                    logger.error("Command routing failed: %s", exc)
                    response = "Sorry, I had an internal error."
        finally:
            _route_duration = time.perf_counter() - route_started
            timing_parts["route"] = _route_duration
            record_stage_timing("routing", _route_duration, intent=turn_intent or "unknown")
            if turn_intent == "LLM_QUERY":
                timing_parts["llm"] = get_thread_stage_timing("llm_generation")
            if streaming_tts_enabled:
                streaming_tts_enabled = _finish_streaming_tts(response)
            if live_context:
                clear_precomputed_live_context()

        if not is_compound:
            print(f"Jarvis: {response}")
        _emit_ui_event(
            "response",
            text=str(response or ""),
            language=tts_language or detected_language or _guess_language(response),
        )
        interrupted = coordinator.current_phase == RuntimePhase.LISTENING
        if (
            should_speak_response
            and not streaming_tts_enabled
            and not interrupted
            and not (pipeline is not None and pipeline.is_early_response_spoken())
        ):
            coordinator.set_phase(RuntimePhase.SPEAKING)
            safe_response = _speech_safe_response(response)
            with stage_timer("tts_first_word", lang=tts_language or "unknown") as tts_timing:
                speech_engine.speak_async(safe_response, language=tts_language)
            timing_parts["tts"] = tts_timing.elapsed
    finally:
        if not speech_engine.is_speaking():
            coordinator.set_phase(RuntimePhase.IDLE)
        total_elapsed = time.perf_counter() - pipeline_started
        metrics.record_stage("pipeline", total_elapsed, success=bool(text) and route_success)
        if turn_intent == "LLM_QUERY":
            record_stage_timing("e2e_llm_query", total_elapsed, lang=detected_language or "unknown")
        log_turn_summary(
            total_elapsed,
            timing_parts,
            lang=detected_language or "unknown",
            intent=turn_intent or "unknown",
        )
        _safe_remove(audio_file)
        # Open a follow-up window whenever a real utterance was processed.
        # The main thread's listen_for_wake_word() will exit on this signal
        # and offer the user FOLLOWUP_WINDOW_SECONDS to speak without wake word.
        if text and FOLLOWUP_ENABLED:
            if (
                SENSITIVE_CONFIRM_MODE == "pin"
                and session_memory.get_pending_confirmation_token() == "pin_required"
                and confirmation_manager.has_pending_pin_action()
            ):
                # PIN is pending: use CONFIRMING state (30s, no wake word) so
                # the user doesn't have to re-trigger the wake word just to
                # speak the PIN — the follow-up window (10s) is too short once
                # TTS finishes saying "This needs your PIN to continue."
                dialogue_manager.transition(DialogueState.CONFIRMING)
            else:
                dialogue_manager.transition(DialogueState.FOLLOW_UP)
            notify_follow_up_wake()


def _cleanup_stale_temp_files():
    """Remove stale utterance and partial-transcription WAVs created by Jarvis."""
    temp_dir = tempfile.gettempdir()
    removed = 0
    for filename_pattern in ("jarvis_utterance_*.wav", "jarvis_partial_*.wav", "jarvis_stt_probe_*.wav"):
        pattern = os.path.join(temp_dir, filename_pattern)
        for path in glob.glob(pattern):
            try:
                os.remove(path)
                removed += 1
            except Exception:
                pass
    if removed:
        logger.info("Cleaned up %d stale temp audio file(s).", removed)


def _run_doctor_diagnostics(trigger):
    started = time.perf_counter()
    doctor_logger = get_logger("doctor")
    try:
        payload = collect_diagnostics(include_model_load_checks=bool(DOCTOR_INCLUDE_MODEL_LOAD_CHECKS))
        ok = bool(payload.get("ok"))
        metrics.record_diagnostic(f"doctor_{trigger}", ok, time.perf_counter() - started)
        doctor_logger.debug(
            "Doctor diagnostics (%s): %s",
            trigger,
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
        )
        doctor_logger.info(
            "Doctor: %d/%d checks OK (required %d/%d)",
            int(payload.get("ok_count") or 0),
            int(payload.get("check_count") or 0),
            int(payload.get("required_ok_count") or 0),
            int(payload.get("required_check_count") or 0),
        )
        failing = [
            str(row.get("name") or "unknown")
            for row in payload.get("checks", [])
            if not row.get("ok")
        ]
        if failing:
            doctor_logger.warning("Doctor failures: %s", ", ".join(failing) or "unknown")
        return payload
    except Exception as exc:
        metrics.record_diagnostic(f"doctor_{trigger}", False, time.perf_counter() - started)
        doctor_logger.warning("Doctor diagnostics failed for trigger=%s: %s", trigger, exc)
        return {
            "ok": False,
            "error": str(exc),
            "trigger": trigger,
        }


def _preload_stt_model():
    """Warm only the latency-critical STT model during startup."""
    try:
        preload_snapshot = stt_runtime.preload_critical_model()
        logger.debug("Critical STT preload complete: %s", preload_snapshot)
    except Exception as exc:
        logger.warning("STT model preload failed (will load on first use): %s", exc)


def _preload_optional_stt_models():
    """Warm optional partial-transcription model off-path."""
    try:
        preload_snapshot = stt_runtime.preload_optional_models()
        logger.debug("Optional STT preload complete: %s", preload_snapshot)
    except Exception as exc:
        logger.warning("Optional STT preload failed (will load on first use): %s", exc)


def _prewarm_streaming_vad():
    if not prewarm_streaming_vad():
        logger.debug("Streaming VAD prewarm unavailable; it will retry on first recording.")


def _prewarm_batch_vad():
    if not prewarm_batch_vad():
        logger.debug("Batch VAD prewarm unavailable; it will retry on first speech guard.")



def _prewarm_llm():
    """Load the Ollama model without generating a throwaway response."""
    try:
        from llm.ollama_client import prewarm_model

        model_name = prewarm_model(
            timeout_seconds=min(60.0, max(10.0, float(LLM_TIMEOUT_SECONDS))),
        )
        logger.info("LLM model loaded and ready (%s).", model_name)
    except Exception as exc:
        logger.debug("LLM load-only prewarm skipped; first query will load the model: %s", exc)


def _prepare_llm_runtime():
    """Start Ollama, select/verify the runtime model, then warm one request."""
    _ensure_ollama_running()
    _detect_and_set_runtime_model()
    _prewarm_llm()


def _start_knowledge_base_auto_sync():
    if not KB_AUTO_SYNC_ENABLED:
        return
    ok, message = knowledge_base_service.start_auto_sync()
    if ok:
        logger.debug("Knowledge-base auto-sync startup: %s", message)
    else:
        logger.warning("Knowledge-base auto-sync startup skipped: %s", message)





def _ollama_version_endpoint() -> str:
    return f"{str(LLM_OLLAMA_BASE_URL or 'http://localhost:11434').rstrip('/')}/api/version"


def _is_ollama_reachable(timeout_seconds: float = 1.0) -> bool:
    try:
        response = httpx.get(_ollama_version_endpoint(), timeout=max(0.2, float(timeout_seconds)))
    except Exception:
        return False
    return bool(response.status_code == 200)


def _ensure_ollama_running():
    global _OLLAMA_AUTOSTART_PROCESS

    if _is_ollama_reachable(timeout_seconds=1.0):
        logger.info("Ollama already running at %s", str(LLM_OLLAMA_BASE_URL or "http://localhost:11434"))
        return True

    if not bool(LLM_OLLAMA_AUTOSTART):
        logger.warning("Ollama is not reachable and auto-start is disabled.")
        return False

    command = [str(LLM_OLLAMA_EXECUTABLE or "ollama"), "serve"]
    creation_flags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))

    logger.info("Ollama not reachable; starting background server via: %s", " ".join(command))
    try:
        _OLLAMA_AUTOSTART_PROCESS = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creation_flags,
            start_new_session=True,
        )
    except Exception as exc:
        logger.warning("Failed to start Ollama server process: %s", exc)
        return False

    wait_seconds = max(3.0, float(LLM_OLLAMA_AUTOSTART_TIMEOUT_SECONDS or 25.0))
    deadline = time.perf_counter() + wait_seconds
    while time.perf_counter() < deadline:
        if _is_ollama_reachable(timeout_seconds=0.8):
            logger.info("Ollama server is ready at %s", str(LLM_OLLAMA_BASE_URL or "http://localhost:11434"))
            return True
        if _OLLAMA_AUTOSTART_PROCESS is not None and _OLLAMA_AUTOSTART_PROCESS.poll() is not None:
            logger.warning("Ollama server process exited before becoming ready.")
            return False
        time.sleep(0.4)

    logger.warning("Timed out waiting for Ollama server startup after %.1fs", wait_seconds)
    return False


def _preload_wake_word_runtime():
    """Warm wake-word model/device resources before entering wake listening loop."""
    started = time.perf_counter()
    try:
        snapshot = preload_runtime_wake_word()
        metrics.record_stage("wake_word_prewarm", time.perf_counter() - started, success=True)
        get_logger("wakeword").info("Wake-word preload complete: %s", snapshot)
    except Exception as exc:
        metrics.record_stage("wake_word_prewarm", time.perf_counter() - started, success=False)
        logger.warning("Wake-word preload failed (will retry on first listen): %s", exc)


def _prewarm_tts():
    """Warm TTS backend resources so first spoken response avoids cold-start penalty."""
    started = time.perf_counter()
    try:
        preferred_language = session_memory.get_preferred_language()
        warmed, backend = speech_engine.prewarm(preferred_language=preferred_language)
        metrics.record_stage("tts_prewarm", time.perf_counter() - started, success=bool(warmed))
        if warmed:
            logger.debug("TTS prewarmed successfully (%s).", backend)
        else:
            logger.debug("TTS prewarm skipped/unavailable (%s).", backend)
    except Exception as exc:
        metrics.record_stage("tts_prewarm", time.perf_counter() - started, success=False)
        logger.warning("TTS prewarm failed (will initialize on first response): %s", exc)


def _prewarm_parser_nlp():
    """Warm parser and keyword-NLU modules to reduce first-command import/init latency."""
    started = time.perf_counter()
    parser_ready = False
    keyword_nlu_ready = False
    try:
        parse_command("open chrome")
        parse_command("افتح كروم")
        parser_ready = True
    except Exception as exc:
        logger.warning("Parser prewarm failed (will initialize on first command): %s", exc)

    if parser_ready:
        try:
            from nlp.intent_classifier import classify_intent

            classify_intent("open youtube")
            classify_intent("افتح يوتيوب")
            keyword_nlu_ready = True
        except Exception as exc:
            logger.warning("Keyword NLU prewarm skipped/unavailable: %s", exc)

    success = bool(parser_ready)
    metrics.record_stage("parser_nlp_prewarm", time.perf_counter() - started, success=success)
    if parser_ready and keyword_nlu_ready:
        logger.debug("Parser + keyword NLU prewarmed successfully.")
    elif parser_ready:
        logger.debug("Parser prewarmed successfully (keyword NLU unavailable).")


def _prewarm_semantic_router():
    """Load the semantic router embedding model so first classification is instant."""
    started = time.perf_counter()
    try:
        from nlp.semantic_router import prewarm as sr_prewarm
        ok = sr_prewarm()
        metrics.record_stage("semantic_router_prewarm", time.perf_counter() - started, success=ok)
        if ok:
            logger.debug("Semantic router prewarmed successfully.")
        else:
            logger.debug("Semantic router prewarm skipped (unavailable).")
    except Exception as exc:
        metrics.record_stage("semantic_router_prewarm", time.perf_counter() - started, success=False)
        logger.warning("Semantic router prewarm failed (will try on first command): %s", exc)


def _startup_app_scan():
    """Background task: scan installed apps and start the Start Menu watcher."""
    started = time.perf_counter()
    try:
        from os_control.app_ops import _BASE_APP_CATALOG, _apply_app_catalog, refresh_app_catalog
        from os_control.app_scanner import start_startmenu_watch

        count = refresh_app_catalog(force=False)
        get_logger("startup").info("app catalog ready: %d entries", count)

        if APP_WATCH_STARTMENU:
            def _on_catalog_change(new_catalog):
                _apply_app_catalog(new_catalog)

            start_startmenu_watch(_on_catalog_change, base_catalog=_BASE_APP_CATALOG)

        metrics.record_stage("app_scan", time.perf_counter() - started, success=True)
    except Exception as exc:
        metrics.record_stage("app_scan", time.perf_counter() - started, success=False)
        logger.warning("Startup app scan failed: %s", exc)


def _detect_and_set_runtime_model():
    """Detect hardware, select model, ensure it's available in Ollama, and set runtime model."""
    from llm.ollama_client import set_runtime_model
    from core.hardware_detect import DEFAULT_MODEL as HARDWARE_DEFAULT_MODEL, recommend_model_tier

    ollama_url = str(LLM_OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
    configured_model = str(LLM_MODEL or "").strip()
    default_model = str(HARDWARE_DEFAULT_MODEL or "qwen3:4b").strip() or "qwen3:4b"

    tier = None
    selection_reason = "configured"
    model_name = configured_model or default_model
    num_ctx = int(LLM_OLLAMA_NUM_CTX)
    lightweight_num_ctx = int(LLM_LIGHTWEIGHT_NUM_CTX)

    # Treat any non-default configured value as explicit manual override.
    explicit_override = bool(configured_model and configured_model.lower() != default_model.lower())
    if explicit_override:
        selection_reason = "manual_override"
    elif bool(LLM_AUTO_SELECT_MODEL):
        selection_reason = "hardware_auto_select"
        tier = recommend_model_tier(ollama_url)
        model_name = str(tier.get("model") or default_model).strip() or default_model
        num_ctx = int(tier.get("num_ctx") or LLM_OLLAMA_NUM_CTX)
        lightweight_num_ctx = int(tier.get("lightweight_num_ctx") or LLM_LIGHTWEIGHT_NUM_CTX)
    else:
        selection_reason = "auto_select_disabled"

    if selection_reason == "hardware_auto_select" and isinstance(tier, dict):
        logger.info(
            "Hardware auto-select: tier=%s model=%s num_ctx=%d lightweight_num_ctx=%d (RAM=%.1fGB, GPU=%s)",
            str(tier.get("tier") or "unknown"),
            model_name,
            num_ctx,
            lightweight_num_ctx,
            float(tier.get("ram_gb") or 0.0),
            "yes" if bool(tier.get("gpu")) else "no",
        )
    else:
        logger.info(
            "Using model '%s' (reason=%s, num_ctx=%d, lightweight_num_ctx=%d)",
            model_name,
            selection_reason,
            num_ctx,
            lightweight_num_ctx,
        )

    set_runtime_model(
        model_name,
        num_ctx=num_ctx,
        lightweight_num_ctx=lightweight_num_ctx,
        tier=str(tier.get("tier") or "medium") if isinstance(tier, dict) else "medium",
    )
    _ensure_model_available(model_name, ollama_url)


def _ensure_model_available(model_name, ollama_url):
    """Check if model exists in Ollama. If not, pull it (blocking)."""
    try:
        r = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        if r.status_code == 200:
            models = [m.get("name", "") for m in r.json().get("models", [])]
            # Check if model is already available (exact or prefix match)
            if any(model_name in m for m in models):
                logger.debug("Model '%s' is available in Ollama.", model_name)
                return
        logger.info("Model '%s' not found locally, pulling...", model_name)
        _pull_model(model_name)
    except Exception as exc:
        logger.warning("Could not verify model availability: %s", exc)


def _pull_model(model_name):
    """Pull a model from Ollama registry with streaming progress logs.

    Blocks until complete or timeout. Logs at most one progress line per ~5 seconds
    to keep the user informed without spamming the log.
    """
    url = f"{str(LLM_OLLAMA_BASE_URL or 'http://localhost:11434').rstrip('/')}/api/pull"
    last_status = ""
    last_log_at = 0.0
    progress_interval = 5.0  # seconds between progress logs

    try:
        with httpx.stream(
            "POST",
            url,
            json={"name": model_name, "stream": True},
            timeout=httpx.Timeout(connect=10.0, read=900.0, write=10.0, pool=10.0),
        ) as response:
            if response.status_code != 200:
                logger.warning(
                    "Model pull returned status %d for '%s'.",
                    response.status_code, model_name,
                )
                return False

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue

                status = str(payload.get("status") or "").strip()
                if not status:
                    continue

                now = time.time()
                total = payload.get("total")
                completed = payload.get("completed")

                # Always log status transitions (e.g. "pulling manifest" → "downloading")
                status_changed = status != last_status
                throttle_elapsed = (now - last_log_at) >= progress_interval

                if status_changed or throttle_elapsed:
                    if total and completed:
                        try:
                            pct = (float(completed) / float(total)) * 100.0
                            mb_done = float(completed) / (1024 ** 2)
                            mb_total = float(total) / (1024 ** 2)
                            logger.info(
                                "Pulling '%s': %s — %.1f%% (%.1f / %.1f MB)",
                                model_name, status, pct, mb_done, mb_total,
                            )
                        except (TypeError, ValueError):
                            logger.info("Pulling '%s': %s", model_name, status)
                    else:
                        logger.info("Pulling '%s': %s", model_name, status)
                    last_status = status
                    last_log_at = now

                if status.lower() == "success":
                    logger.info("Model '%s' pulled successfully.", model_name)
                    return True
        return True
    except Exception as exc:
        logger.warning("Failed to pull model '%s': %s", model_name, exc)
        return False


def _wait_for_tts_completion(max_wait: float = 12.0) -> None:
    """Poll until TTS finishes or max_wait seconds elapse."""
    deadline = time.perf_counter() + max(0.0, float(max_wait))
    while speech_engine.is_speaking() and time.perf_counter() < deadline:
        time.sleep(0.05)


def _play_follow_up_chime() -> None:
    """Emit a short audio cue telling the user the follow-up window is open.

    Only fires when JARVIS_FOLLOWUP_CHIME_ENABLED=true (off by default).
    Uses winsound on Windows; silently skipped on other platforms or when the
    module is unavailable.
    """
    if not FOLLOWUP_CHIME_ENABLED:
        return
    try:
        import winsound  # Windows-only
        winsound.Beep(880, 150)  # 880 Hz, 150 ms — subtle rising cue
    except Exception:
        pass  # Non-Windows or winsound absent — silent fallback


def _warmup_output_device() -> None:
    """Play ~80 ms of silence at 24000 Hz to open the sounddevice output stream.

    Edge-TTS outputs at 24 kHz. sounddevice opens a new stream on every sd.play()
    call; the first open on Windows takes ~100-200 ms and clips the first audio
    chunk. Playing silence at the same sample rate pre-opens the stream so the
    greeting starts cleanly.
    """
    try:
        import numpy as np
        import sounddevice as sd
        sr = 24000  # Match Edge-TTS output rate
        silence = np.zeros(int(sr * 0.08), dtype=np.float32)
        sd.play(silence, samplerate=sr, blocking=True)
        sd.wait()  # Ensure stream is fully open and flushed before returning
    except Exception:
        pass  # Non-fatal — if sounddevice is absent, skip warmup


def _speak_startup_greeting():
    """Speak the configured bilingual greeting synchronously (blocking).

    Called as the strict last step before the wake loop so the greeting plays
    fully before Jarvis starts listening. Fires after every prewarm thread is
    already launched so there is no audio-device contention.
    """
    if not GREETING_ENABLED or not speech_engine.is_enabled():
        return False

    configured_language = str(GREETING_LANGUAGE or "en").strip().lower()
    if configured_language == "auto":
        language = "ar" if str(STT_LANGUAGE_HINT or "").strip().lower() == "ar" else "en"
    else:
        language = configured_language if configured_language in {"ar", "en"} else "en"

    text = GREETING_TEXT_AR if language == "ar" else GREETING_TEXT_EN
    text = str(text or "").strip()
    if not text:
        return False

    if GREETING_DEVICE_WARMUP:
        _warmup_output_device()

    if GREETING_PRESPEAK_SETTLE_MS > 0:
        time.sleep(GREETING_PRESPEAK_SETTLE_MS / 1000.0)

    try:
        started, _ = speech_engine.speak_async(text, language=language)
        if started:
            # Block until TTS finishes so the wake loop doesn't start mid-greeting.
            _wait_for_tts_completion(max_wait=30.0)
            get_logger("startup").info("Greeting spoken (lang=%s)", language)
        return bool(started)
    except Exception as exc:
        get_logger("startup").warning("Startup greeting failed: %s", exc)
        return False


def _calibrate_baseline_noise():
    """Measure ambient noise floor before any TTS plays; used for echo threshold."""
    logger.debug("Baseline noise calibration skipped (echo cancel disabled for STT debugging).")
    return


def _run_startup_prewarm_blocking():
    section("Startup")
    _calibrate_baseline_noise()

    critical_tasks = [
        ("wake_word", _preload_wake_word_runtime),
        ("stt", _preload_stt_model),
        ("streaming_vad", _prewarm_streaming_vad),
        ("batch_vad", _prewarm_batch_vad),
    ]
    if TTS_PREWARM_ENABLED:
        critical_tasks.append(("tts", _prewarm_tts))

    background_tasks = [("stt_optional", _preload_optional_stt_models)]
    if STARTUP_PARSER_NLP_PREWARM_ENABLED:
        background_tasks.append(("parser_nlp", _prewarm_parser_nlp))
    if SEMANTIC_ROUTER_ENABLED:
        target = critical_tasks if PREWARM_SEMANTIC_ROUTER_BLOCKING else background_tasks
        target.append(("semantic_router", _prewarm_semantic_router))
    # When LLM_PREWARM_BEFORE_GREETING is set, the full LLM runtime setup
    # (Ollama start + model select + model load) is done on the pre-greeting
    # daemon thread launched in run() — don't double-schedule it here.
    if not (LLM_PREWARM_BEFORE_GREETING and not PREWARM_LLM_BLOCKING):
        llm_target = critical_tasks if PREWARM_LLM_BLOCKING else background_tasks
        llm_target.append(("llm", _prepare_llm_runtime))
    if APP_SCAN_ON_STARTUP:
        background_tasks.append(("app_scan", _startup_app_scan))
    if KB_AUTO_SYNC_ENABLED:
        background_tasks.append(("knowledge_base", _start_knowledge_base_auto_sync))
    if DOCTOR_STARTUP_ENABLED and DOCTOR_STARTUP_ASYNC:
        background_tasks.append(("doctor", lambda: _run_doctor_diagnostics("startup")))
    from core.config import NLU_SCHEMA_ENABLED as _NLU_SCHEMA_ENABLED
    if _NLU_SCHEMA_ENABLED:
        def _validate_intent_schema():
            from core.intent_schema import validate_schema_coverage
            validate_schema_coverage()
        background_tasks.append(("intent_schema", _validate_intent_schema))

    if not STARTUP_BACKGROUND_PREWARM_ENABLED:
        critical_tasks.extend(
            task for task in background_tasks if task[0] not in {"knowledge_base", "doctor"}
        )
        background_tasks = []

    cpu_cores = max(1, int(os.cpu_count() or 1))
    allow_sequential_prewarm = cpu_cores <= 4

    logger.info(
        "Critical warmup started (cpu_cores=%d, sequential=%s).",
        cpu_cores,
        allow_sequential_prewarm,
    )
    started = time.perf_counter()
    warmup_rows = []

    def _run_timed_task(task_name, task_fn):
        task_started = time.perf_counter()
        try:
            return task_fn()
        finally:
            warmup_rows.append((task_name, time.perf_counter() - task_started))

    if allow_sequential_prewarm:
        for task_name, task_fn in critical_tasks:
            try:
                _run_timed_task(task_name, task_fn)
            except Exception as exc:
                logger.warning("Critical warmup task '%s' crashed: %s", task_name, exc)
    else:
        with ThreadPoolExecutor(
            max_workers=max(1, len(critical_tasks)),
            thread_name_prefix="jarvis-critical-prewarm",
        ) as prewarm_executor:
            futures = {
                prewarm_executor.submit(_run_timed_task, task_name, task_fn): task_name
                for task_name, task_fn in critical_tasks
            }
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    logger.warning("Critical warmup task '%s' crashed: %s", task_name, exc)

    total_elapsed = time.perf_counter() - started
    task_order = {name: index for index, (name, _) in enumerate(critical_tasks)}
    ordered_rows = sorted(warmup_rows, key=lambda row: task_order[row[0]])
    log_warmup_table(
        [("Task", "Seconds")]
        + [(name, f"{elapsed:.2f}") for name, elapsed in ordered_rows]
        + [("TOTAL", f"{total_elapsed:.2f}")]
    )
    get_logger("startup").info("Critical warmup done in %.2fs — listening", total_elapsed)
    kv(
        "startup",
        stt=WHISPER_MODEL,
        partial=STT_PARTIAL_WHISPER_MODEL,
        wake="unified",
        tts=TTS_DEFAULT_BACKEND,
        router=(
            "loaded"
            if PREWARM_SEMANTIC_ROUTER_BLOCKING or not STARTUP_BACKGROUND_PREWARM_ENABLED
            else "background"
        ),
    )

    if not background_tasks:
        return None
    def _run_background_tasks():
        background_started = time.perf_counter()
        names = [name for name, _ in background_tasks]

        def _run_background_task(task_name, task_fn):
            try:
                task_fn()
            except Exception as exc:
                logger.warning("Background warmup task '%s' crashed: %s", task_name, exc)

        workers = [
            threading.Thread(
                target=_run_background_task,
                args=(task_name, task_fn),
                name=f"jarvis-background-{task_name}",
                daemon=True,
            )
            for task_name, task_fn in background_tasks
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        get_logger("startup").info(
            "Background warmup done in %.2fs (%s)",
            time.perf_counter() - background_started,
            ", ".join(names),
        )

    thread = threading.Thread(
        target=_run_background_tasks,
        name="jarvis-background-prewarm-coordinator",
        daemon=True,
    )
    thread.start()
    return thread


def run():
    section("Jarvis")
    kv(
        "startup",
        model=LLM_MODEL,
        tier="auto" if LLM_AUTO_SELECT_MODEL else "configured",
    )
    kv(
        "startup",
        followup=FOLLOWUP_ENABLED,
        wake_interrupt=True,
    )
    shutdown_event = setup_shutdown()
    _cleanup_stale_temp_files()
    initialize_command_services()
    stt_runtime.set_runtime_stt_settings(language_hint=_resolve_stt_language_hint())

    # Only latency-critical audio components block listening. Optional models,
    # Ollama, knowledge sync, and doctor diagnostics continue in the background.
    _run_startup_prewarm_blocking()

    if not STARTUP_BACKGROUND_PREWARM_ENABLED:
        _start_knowledge_base_auto_sync()
    if DOCTOR_STARTUP_ENABLED and (
        not DOCTOR_STARTUP_ASYNC or not STARTUP_BACKGROUND_PREWARM_ENABLED
    ):
        _run_doctor_diagnostics("startup")

    doctor_interval_seconds = max(0.0, float(DOCTOR_SCHEDULE_INTERVAL_SECONDS))
    next_doctor_run_at = time.time() + doctor_interval_seconds if doctor_interval_seconds > 0 else 0.0

    # max_workers=3: one for _process_utterance, one for early-command execution,
    # one for live-data pre-fetch — all can overlap on a single utterance.
    executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="jarvis-pipeline")
    in_flight = []
    output_encoding = (getattr(sys.stdout, "encoding", "") or "").lower()
    if "utf" not in output_encoding:
        logger.warning(
            "Console encoding is %s; non-English text may be garbled. "
            "Use `chcp 65001` and set `PYTHONUTF8=1` before starting Jarvis.",
            output_encoding or "unknown",
        )
    coordinator.attach_speech_engine(speech_engine)
    from llm.ollama_client import llm_cancel_event
    coordinator.attach_llm_cancel_event(llm_cancel_event)

    # Fire full LLM runtime setup (Ollama + model select + model load) on a
    # daemon thread so it runs concurrently with the greeting audio. The greeting
    # takes ~2-3 s to speak, which hides most of the model cold-start latency.
    # This is the ONLY path that sets up the LLM runtime when
    # LLM_PREWARM_BEFORE_GREETING=true (it's excluded from prewarm task lists).
    if LLM_PREWARM_BEFORE_GREETING and not PREWARM_LLM_BLOCKING:
        _llm_prewarm_thread = threading.Thread(
            target=_prepare_llm_runtime,
            name="jarvis-llm-prewarm-pre-greeting",
            daemon=True,
        )
        _llm_prewarm_thread.start()

    # Greeting is the strict last step before listening — plays fully (blocking)
    # so the wake loop doesn't start while TTS is still holding the mic device.
    _speak_startup_greeting()
    get_logger("startup").info("Jarvis ready — listening.")

    prime_llm_response_cache_async()
    _adaptive_start_daemon()

    try:
        while not shutdown_event.is_set():
            if doctor_interval_seconds > 0 and time.time() >= next_doctor_run_at:
                _run_doctor_diagnostics("scheduled")
                next_doctor_run_at = time.time() + doctor_interval_seconds

            in_flight = _prune_futures(in_flight)
            busy = len(in_flight) >= max(1, int(REALTIME_MAX_PENDING_UTTERANCES))
            if busy and REALTIME_DROP_WHEN_BUSY:
                time.sleep(float(REALTIME_BACKPRESSURE_POLL_SECONDS))
                metrics.record_stage("backpressure_wait", float(REALTIME_BACKPRESSURE_POLL_SECONDS), success=True)
                continue
            if coordinator.current_phase in (RuntimePhase.IDLE, RuntimePhase.LISTENING):
                if speech_engine.is_speaking():
                    coordinator.set_phase(RuntimePhase.SPEAKING)
                else:
                    coordinator.set_phase(RuntimePhase.LISTENING)
            wake_started = time.perf_counter()
            if dialogue_manager.should_skip_wake_word():
                # Already in FOLLOW_UP or CONFIRMING — bypass the wake-word
                # listener entirely.  The dialogue manager's state acts as the
                # gate; no need to run the ONNX wake-word model for this turn.
                wake_source = "follow_up"
                wake_elapsed = 0.0
                metrics.record_stage("wake_word", 0.0, success=True)
            else:
                try:
                    wake_source = listen_for_wake_word()
                    wake_elapsed = time.perf_counter() - wake_started
                    metrics.record_stage("wake_word", wake_elapsed, success=True)
                except RuntimeError as exc:
                    wake_elapsed = time.perf_counter() - wake_started
                    metrics.record_stage("wake_word", wake_elapsed, success=False)
                    logger.error("Wake word unavailable: %s", exc)
                    print("Wake word/audio stack is unavailable in this environment.")
                    _run_text_fallback_loop()
                    return

            if shutdown_event.is_set():
                break

            if wake_source == "follow_up":
                # Follow-up window: the previous _process_utterance signalled us
                # to skip the wake word for this turn.
                dialogue_manager.transition(DialogueState.LISTENING)
                logger.info(
                    "Follow-up window: recording without wake word (turn=%d).",
                    dialogue_manager.conversation_turns,
                )
                metrics.record_stage("follow_up_trigger", 0.0, success=True)
                # Wait for TTS from the previous response to finish so that the
                # mic does not pick up the assistant's own voice as user speech.
                # Clarification questions can be long (the list of options takes
                # time to speak), so use a longer cap when clarification is pending.
                _pending_clarf = session_memory.get_pending_clarification()
                _tts_wait = 30.0 if _pending_clarf else 12.0
                _wait_for_tts_completion(max_wait=_tts_wait)
                # Reset the follow-up window deadline after TTS finishes so the
                # user gets the full window to reply, not a countdown that started
                # while TTS was still speaking.
                if _pending_clarf:
                    dialogue_manager.reset_window()
                # Optional chime: audible cue that the follow-up window is now
                # open and the user may speak without the wake word.
                _play_follow_up_chime()
                remaining = dialogue_manager.time_remaining()
                if remaining < 0.5:
                    logger.info("Follow-up window expired; returning to IDLE.")
                    dialogue_manager.transition(DialogueState.IDLE)
                    continue
            else:
                dialogue_manager.transition(DialogueState.LISTENING)
                logger.info("Wake word detected via %s", wake_source or "unknown")
            pipeline_started = time.perf_counter()
            with stage_timer("wake_to_stt_start", source=wake_source or "unknown") as wake_to_stt_timing:
                in_flight = _prune_futures(in_flight)
                busy = len(in_flight) >= max(1, int(REALTIME_MAX_PENDING_UTTERANCES))
                if busy and REALTIME_DROP_WHEN_BUSY:
                    logger.warning("Dropping wake event due to pipeline backpressure.")
                    metrics.record_stage("backpressure_drop", 0.0, success=False)
                    continue

                audio_file = _create_utterance_audio_file()

                # Task 1.3: create pipeline before recording so partials are
                # processed concurrently while the user is still speaking.
                concurrent_pipeline = ConcurrentPipeline(
                    executor,
                    language_hint=_resolve_stt_language_hint(wake_source=wake_source) or "",
                )

            coordinator.set_phase(RuntimePhase.RECORDING)
            from llm.ollama_client import llm_cancel_event as _llm_cancel
            _llm_cancel.clear()
            record_started = time.perf_counter()

            _partial_latency_recorded = False

            def _pipeline_partial(partial_text):
                nonlocal _partial_latency_recorded
                _on_partial_transcript(partial_text)
                concurrent_pipeline.on_partial(partial_text)
                if not _partial_latency_recorded and (partial_text or "").strip():
                    _partial_latency_recorded = True
                    latency_tracker.record("stt_partial_latency", time.perf_counter() - record_started)

            with stage_timer("recording", source=wake_source or "unknown") as recording_timing:
                capture = record_utterance_streaming(
                    filename=audio_file,
                    max_duration=MAX_RECORD_DURATION,
                    vad_mode="chat" if wake_source == "follow_up" else "command",
                    language_hint=_resolve_stt_language_hint(wake_source=wake_source),
                    start_timeout_seconds=(
                        max(1.0, dialogue_manager.time_remaining())
                        if wake_source == "follow_up"
                        else None
                    ),
                    enable_partials=True,
                    on_partial=_pipeline_partial,
                )
            metrics.record_stage(
                "record_audio",
                recording_timing.elapsed,
                success=bool(capture.get("speech_detected")),
            )

            if shutdown_event.is_set():
                _safe_remove(audio_file)
                break

            _wake_audio = get_last_detection_audio() if wake_source == "wake" else None

            if not capture.get("speech_detected"):
                if _wake_audio and wake_source == "wake":
                    _adaptive_record_false_positive(_wake_audio)
                _safe_remove(audio_file)
                if wake_source == "follow_up":
                    logger.info("No speech in follow-up window; returning to IDLE.")
                    dialogue_manager.transition(DialogueState.IDLE)
                continue

            if _wake_audio and wake_source == "wake":
                _adaptive_record_confirmed(_wake_audio)

            in_flight.append(
                executor.submit(
                    _process_utterance,
                    audio_file,
                    pipeline_started,
                    wake_source,
                    capture,
                    concurrent_pipeline,
                    {
                        "wake": wake_elapsed,
                        "rec": recording_timing.elapsed,
                        "wake_to_stt": wake_to_stt_timing.elapsed,
                    },
                )
            )
    finally:
        with stage_timer("shutdown") as shutdown_timing:
            _adaptive_stop_daemon()
            try:
                knowledge_base_service.stop_auto_sync()
            except Exception:
                pass
            stt_runtime.close_cloud_http_client()
            perform_shutdown_cleanup()
            executor.shutdown(wait=False, cancel_futures=False)
        get_logger("shutdown").info("Shutdown complete in %.2fs", shutdown_timing.elapsed)


if __name__ == "__main__":
    run()
