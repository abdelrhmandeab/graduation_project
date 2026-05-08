import glob
import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.streaming_stt import record_utterance_streaming
from audio import stt as stt_runtime
from audio.stt import transcribe_streaming
from audio.tts import speech_engine
from audio.vad import is_speech
from audio.wake_word import (
    get_runtime_wake_word_behavior,
    listen_for_wake_word,
    preload_runtime_wake_word,
)
from core.command_parser import parse_command
from core.command_router import (
    initialize_command_services,
    route_command,
    inject_precomputed_live_context,
    clear_precomputed_live_context,
    looks_like_live_data_query,
)
from core.knowledge_base import knowledge_base_service
from core.doctor import collect_diagnostics
from core.config import (
    EARLY_EXEC_CONFIDENCE_THRESHOLD,
    DOCTOR_INCLUDE_MODEL_LOAD_CHECKS,
    DOCTOR_SCHEDULE_INTERVAL_SECONDS,
    DOCTOR_STARTUP_ENABLED,
    FOLLOWUP_CHIME_ENABLED,
    FOLLOWUP_ENABLED,
    FOLLOWUP_WINDOW_SECONDS,
    LLM_AUTO_SELECT_MODEL,
    LLM_MODEL,
    LLM_OLLAMA_AUTOSTART,
    LLM_OLLAMA_AUTOSTART_TIMEOUT_SECONDS,
    LLM_OLLAMA_BASE_URL,
    LLM_OLLAMA_EXECUTABLE,
    LLM_LIGHTWEIGHT_NUM_CTX,
    LLM_OLLAMA_NUM_CTX,
    MAX_RECORD_DURATION,
    REALTIME_BACKPRESSURE_POLL_SECONDS,
    REALTIME_DROP_WHEN_BUSY,
    REALTIME_MAX_PENDING_UTTERANCES,
    SPEECH_GUARD_SKIP_NON_RESPONSIVE_PROFILES,
    SEMANTIC_ROUTER_ENABLED,
    STARTUP_PARSER_NLP_PREWARM_ENABLED,
    TTS_PREWARM_ENABLED,
    KB_AUTO_SYNC_ENABLED,
)
from core.dialogue_manager import DialogueState, dialogue_manager, notify_follow_up_wake
from core.intent_confidence import assess_intent_confidence
from core.logger import logger
from core.language_gate import detect_supported_language
from core.metrics import latency_tracker, metrics
from core.session_memory import session_memory
from core.shutdown import perform_shutdown_cleanup, setup_shutdown

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
_TRANSCRIPT_TOKEN_RE = re.compile(r"[A-Za-z0-9\u0600-\u06FF]+")
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
        self._early_intent_str: "str | None" = None
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

    def cancel_early_if_possible(self) -> None:
        """Best-effort cancel. Resets early-executed flag so _process_utterance re-routes."""
        with self._lock:
            future = self._early_execution_future
            self._early_executed = False
            self._early_intent_str = None
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
            response = route_command(
                text,
                detected_language=lang or None,
                realtime=True,
            )
            if response:
                safe = _speech_safe_response(response)
                if safe:
                    speech_engine.speak_async(safe, language=lang or None)
                with self._lock:
                    self._early_response = response
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
    # Prefer auto-detection on every utterance so one Arabic turn does not
    # lock the next turn into Arabic. Keep a narrow fast-path when the wake
    # word itself is Arabic or English and the runtime has not already been
    # forced to a specific language.
    runtime_hint = str(stt_runtime.get_runtime_stt_settings().get("language_hint") or "auto").strip().lower()
    if runtime_hint in {"ar", "arabic", "ar-eg", "ar_eg"}:
        return "ar"
    if runtime_hint in {"en", "english", "en-us", "en_us"}:
        return "en"
    wake_source_value = str(wake_source or "").strip().lower()
    if wake_source_value == "arabic":
        return "ar"
    if wake_source_value == "english":
        return "en"
    return None


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


def _on_partial_transcript(partial_text):
    if partial_text:
        logger.debug("STT partial: %s", partial_text[-180:])


def _safe_log_text(text, max_chars=220):
    value = " ".join((text or "").split())
    if len(value) > max_chars:
        value = value[: max_chars - 3] + "..."
    return value


def _extract_detected_language_from_stt(text):
    global _LAST_STT_LANGUAGE_CONFIDENCE
    stt_meta = stt_runtime.get_last_transcription_meta()
    detected_language = str((stt_meta or {}).get("language") or "").strip().lower()
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


def _looks_low_quality_transcript(text):
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return True

    tokens = _TRANSCRIPT_TOKEN_RE.findall(normalized)
    if not tokens:
        return True

    if len(tokens) == 1 and len(tokens[0]) <= 4:
        return True

    total_alpha = sum(len(token) for token in tokens)
    if len(tokens) <= 2 and total_alpha <= 6:
        return True

    return False


def _transcript_quality_score(text, detected_language, wake_source=None):
    _ = wake_source
    normalized = " ".join(str(text or "").split()).strip()
    if not normalized:
        return -100.0

    lang = str(detected_language or "").strip().lower()
    if lang not in {"ar", "en"}:
        lang = "en"

    parsed = parse_command(normalized)
    assessment = assess_intent_confidence(normalized, parsed, language=lang)

    score = float(assessment.confidence or 0.0) * 100.0
    tokens = _TRANSCRIPT_TOKEN_RE.findall(normalized)
    score += min(12.0, float(sum(len(token) for token in tokens)) / 4.0)

    if bool(getattr(assessment, "should_clarify", False)):
        reason = str(getattr(assessment, "reason", "") or "").strip().lower()
        if reason == "low_confidence_unclear_query":
            score -= 45.0
        elif reason == "low_confidence_action_like_query":
            score -= 22.0
        else:
            score -= 10.0

    if _looks_low_quality_transcript(normalized):
        score -= 18.0

    return score


def _transcribe_with_runtime_stt(audio_file, wake_source=None):
    global _LAST_STT_LANGUAGE_CONFIDENCE
    primary_hint = _resolve_stt_language_hint(wake_source=wake_source)
    text = transcribe_streaming(
        audio_file,
        on_partial=_on_partial_transcript,
        language_hint=primary_hint,
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
            _t = time.perf_counter()
            _r = parse_command(text)
            latency_tracker.record("intent_detection", time.perf_counter() - _t)
            return _r

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


def _process_utterance(audio_file, pipeline_started, wake_source=None, capture_summary=None, pipeline=None):
    text = ""
    route_success = False
    _utterance_process_started = time.perf_counter()
    try:
        active_audio_ux_profile = str(session_memory.get_audio_ux_profile() or "").strip().lower()
        skip_post_capture_guard = active_audio_ux_profile in _LOW_LATENCY_AUDIO_UX_PROFILES
        # If the streaming VAD already confirmed speech during recording, the
        # redundant batch file-based check is both slow and less reliable — skip it.
        if not skip_post_capture_guard and bool((capture_summary or {}).get("speech_detected")):
            skip_post_capture_guard = True

        if skip_post_capture_guard:
            # record_utterance already runs mic VAD; skip duplicate file-based guard.
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

        # ── Task 1.1: streaming STT fast-path ────────────────────────────────
        # StreamingSTT transcribes the audio *during* recording and stores the
        # final result in capture_summary["text"].  Reuse it to skip the second
        # full STT pass that used to add 1–2 s of dead time after speech ended.
        stt_started = time.perf_counter()
        latency_tracker.record("wake_to_stt_start", stt_started - _utterance_process_started)
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
            latency_tracker.record("stt_total", _stt_duration)
            logger.debug("STT fast-path: reusing streaming transcript (skipped redundant STT pass)")
        else:
            # Fallback: no streaming transcript available (e.g. sounddevice unavailable,
            # or capture returned empty text). Run standard batch STT.
            _ = _resolve_stt_language_hint(wake_source=wake_source)
            text, detected_language = _transcribe_with_runtime_stt(
                audio_file,
                wake_source=wake_source,
            )
            _stt_duration = time.perf_counter() - stt_started
            metrics.record_stage("stt", _stt_duration, success=bool(text))
            latency_tracker.record("stt_total", _stt_duration)
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
                logger.info("ConcurrentPipeline: skipping full route — already handled via early execution.")
                route_success = True
                return
        # ─────────────────────────────────────────────────────────────────────

        precomputed_language_result, precomputed_parser_candidate = _precompute_post_stt_routing(
            text,
            detected_language=detected_language,
        )

        # Streaming TTS state: queue sentence chunks immediately as they arrive
        # so playback pipelines naturally without polling for completion.
        tts_language = detected_language or session_memory.get_preferred_language()
        should_speak_response = not _is_interrupt_command(text)
        streamed_sentences = []
        sentence_queue = queue.Queue()
        sentence_queue_started = False

        def _iter_streamed_sentences():
            while True:
                item = sentence_queue.get()
                if item is None:
                    break
                yield item

        if should_speak_response:
            sentence_queue_started, _ = speech_engine.speak_sentence_queue(
                _iter_streamed_sentences(),
                language=tts_language,
            )

        _first_sentence_recorded = False

        def _on_sentence_streamed(sentence):
            nonlocal _first_sentence_recorded
            if not (should_speak_response and sentence_queue_started):
                return
            normalized = _speech_safe_response(sentence)
            normalized = " ".join(str(normalized or "").split()).strip()
            if not normalized:
                return
            streamed_sentences.append(normalized)
            sentence_queue.put(normalized)
            if not _first_sentence_recorded:
                _first_sentence_recorded = True
                latency_tracker.record("llm_first_token", time.perf_counter() - route_started)

        dialogue_manager.transition(DialogueState.RESPONDING)

        sub_commands = _split_compound_utterance(text)
        route_started = time.perf_counter()
        is_compound = len(sub_commands) > 1

        # ── Task 1.3: inject pre-fetched live context for LLM queries ─────────
        live_context = pipeline.get_live_context() if pipeline is not None else ""
        if live_context:
            inject_precomputed_live_context(live_context)
        # ─────────────────────────────────────────────────────────────────────

        try:
            if is_compound:
                # Close the streaming queue immediately — compound path speaks the full response at the end
                sentence_queue.put(None)
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
                        on_sentence=_on_sentence_streamed,
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
            if live_context:
                clear_precomputed_live_context()

        if not is_compound:
            print(f"Jarvis: {response}")
        if should_speak_response:
            safe_response = _speech_safe_response(response)
            if is_compound:
                speech_engine.speak_async(safe_response, language=tts_language)
            elif sentence_queue_started:
                remaining = _remaining_after_streamed_sentences(safe_response, streamed_sentences)
                if remaining:
                    sentence_queue.put(remaining)
                sentence_queue.put(None)
            else:
                # Fallback when queue startup failed.
                speech_engine.speak_async(safe_response, language=tts_language)
    finally:
        metrics.record_stage("pipeline", time.perf_counter() - pipeline_started, success=bool(text) and route_success)
        _safe_remove(audio_file)
        # Open a follow-up window whenever a real utterance was processed.
        # The main thread's listen_for_wake_word() will exit on this signal
        # and offer the user FOLLOWUP_WINDOW_SECONDS to speak without wake word.
        if text and FOLLOWUP_ENABLED:
            dialogue_manager.transition(DialogueState.FOLLOW_UP)
            notify_follow_up_wake()


def _cleanup_stale_temp_files():
    """Remove leftover jarvis_utterance_*.wav from the temp directory."""
    temp_dir = tempfile.gettempdir()
    pattern = os.path.join(temp_dir, "jarvis_utterance_*.wav")
    removed = 0
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
    try:
        payload = collect_diagnostics(include_model_load_checks=bool(DOCTOR_INCLUDE_MODEL_LOAD_CHECKS))
        ok = bool(payload.get("ok"))
        metrics.record_diagnostic(f"doctor_{trigger}", ok, time.perf_counter() - started)
        encoded = json.dumps(payload, ensure_ascii=True)
        if len(encoded) > 2000:
            encoded = encoded[:1997] + "..."
        logger.info("Doctor diagnostics (%s): %s", trigger, encoded)
        if not ok:
            logger.warning("Doctor diagnostics reported failures for trigger=%s", trigger)
        return payload
    except Exception as exc:
        metrics.record_diagnostic(f"doctor_{trigger}", False, time.perf_counter() - started)
        logger.warning("Doctor diagnostics failed for trigger=%s: %s", trigger, exc)
        return {
            "ok": False,
            "error": str(exc),
            "trigger": trigger,
        }


def _preload_stt_model():
    """Warm the active STT runtime backend during startup prewarm."""
    try:
        preload_snapshot = stt_runtime.preload_runtime_models()
        logger.info("STT preload complete: %s", preload_snapshot)
    except Exception as exc:
        logger.warning("STT model preload failed (will load on first use): %s", exc)


def _prewarm_streaming_vad():
    """Pre-load the Silero VAD singleton used by StreamingSTT.

    Without this, the first utterance after wake-word detection pays a
    100–500 ms ONNX model-load penalty before VAD can classify any chunk.
    """
    try:
        from audio.streaming_stt import prewarm_streaming_vad
        ready = prewarm_streaming_vad()
        if ready:
            logger.info("Streaming VAD prewarmed successfully (Silero ONNX ready).")
        else:
            logger.info("Streaming VAD prewarmed (energy-fallback mode; Silero ONNX unavailable).")
    except Exception as exc:
        logger.warning("Streaming VAD prewarm failed (will load on first utterance): %s", exc)


def _prewarm_batch_vad():
    """Pre-load the batch Silero VAD singleton used by the speech guard."""
    try:
        from audio.vad import prewarm_batch_vad
        ready = prewarm_batch_vad()
        if ready:
            logger.info("Batch VAD prewarmed successfully (Silero ONNX ready).")
        else:
            logger.info("Batch VAD prewarmed (energy-fallback mode; Silero ONNX unavailable).")
    except Exception as exc:
        logger.warning("Batch VAD prewarm failed (will load on first speech guard): %s", exc)


def _is_llm_prewarm_failure(response_text):
    text = " ".join(str(response_text or "").strip().lower().split())
    if not text:
        return True

    failure_markers = (
        "timed out",
        "cannot connect to ollama",
        "could not run the local model",
        "internal error",
    )
    return any(marker in text for marker in failure_markers)


def _prewarm_llm():
    """Send a minimal prompt to Ollama so the model is loaded into memory before the user speaks."""
    try:
        from llm.ollama_client import ask_llm
        warmup_response = ask_llm("Hi", num_ctx=64)
        if _is_llm_prewarm_failure(warmup_response):
            raise RuntimeError(warmup_response)
        logger.info("LLM prewarmed successfully.")
    except Exception as exc:
        logger.warning("LLM prewarm failed (will load on first query): %s", exc)


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
        logger.info("Wake-word preload complete: %s", snapshot)
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
            logger.info("TTS prewarmed successfully (%s).", backend)
        else:
            logger.info("TTS prewarm skipped/unavailable (%s).", backend)
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
        logger.info("Parser + keyword NLU prewarmed successfully.")
    elif parser_ready:
        logger.info("Parser prewarmed successfully (keyword NLU unavailable).")


def _prewarm_semantic_router():
    """Load the semantic router embedding model so first classification is instant."""
    started = time.perf_counter()
    try:
        from nlp.semantic_router import prewarm as sr_prewarm
        ok = sr_prewarm()
        metrics.record_stage("semantic_router_prewarm", time.perf_counter() - started, success=ok)
        if ok:
            logger.info("Semantic router prewarmed successfully.")
        else:
            logger.info("Semantic router prewarm skipped (unavailable).")
    except Exception as exc:
        metrics.record_stage("semantic_router_prewarm", time.perf_counter() - started, success=False)
        logger.warning("Semantic router prewarm failed (will try on first command): %s", exc)


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
                logger.info("Model '%s' is available in Ollama.", model_name)
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


def _calibrate_baseline_noise():
    """Measure ambient noise floor before any TTS plays; used for echo threshold."""
    started = time.perf_counter()
    try:
        from audio.echo_cancel import baseline_noise
        from core.config import BASELINE_NOISE_CALIBRATION_SECONDS
        rms = baseline_noise.calibrate(seconds=float(BASELINE_NOISE_CALIBRATION_SECONDS))
        metrics.record_stage("baseline_noise_calibration", time.perf_counter() - started, success=True)
        logger.info("Ambient noise floor calibrated: rms=%.5f", rms)
    except Exception as exc:
        metrics.record_stage("baseline_noise_calibration", time.perf_counter() - started, success=False)
        logger.warning("Ambient noise calibration failed (echo threshold will use ratio-only): %s", exc)


def _run_startup_prewarm_blocking():
    # Measure ambient noise before TTS or LLM prewarm to avoid contamination.
    _calibrate_baseline_noise()

    _ensure_ollama_running()
    _detect_and_set_runtime_model()

    tasks = [
        ("wake_word", _preload_wake_word_runtime),
        ("stt", _preload_stt_model),
        ("streaming_vad", _prewarm_streaming_vad),
        ("batch_vad", _prewarm_batch_vad),
        ("llm", _prewarm_llm),
    ]
    cpu_cores = max(1, int(os.cpu_count() or 1))
    allow_extended_prewarm = cpu_cores >= 6
    allow_heavy_prewarm = cpu_cores >= 10

    if STARTUP_PARSER_NLP_PREWARM_ENABLED and allow_extended_prewarm:
        tasks.append(("parser_nlp", _prewarm_parser_nlp))
    if SEMANTIC_ROUTER_ENABLED and allow_heavy_prewarm:
        tasks.append(("semantic_router", _prewarm_semantic_router))
    if TTS_PREWARM_ENABLED and allow_extended_prewarm:
        tasks.append(("tts", _prewarm_tts))

    if not tasks:
        return

    logger.info(
        "Startup prewarm started (cpu_cores=%d, extended=%s, heavy=%s); waiting before wake-word listening begins.",
        cpu_cores,
        allow_extended_prewarm,
        allow_heavy_prewarm,
    )
    started = time.perf_counter()

    with ThreadPoolExecutor(
        max_workers=max(1, len(tasks)),
        thread_name_prefix="jarvis-startup-prewarm",
    ) as prewarm_executor:
        futures = {
            prewarm_executor.submit(task_fn): task_name
            for task_name, task_fn in tasks
        }
        for future in as_completed(futures):
            task_name = futures[future]
            try:
                future.result()
            except Exception as exc:
                logger.warning("Startup prewarm task '%s' crashed: %s", task_name, exc)

    logger.info(
        "Startup prewarm finished in %.2fs; entering wake-word loop.",
        time.perf_counter() - started,
    )


def run():
    shutdown_event = setup_shutdown()
    _cleanup_stale_temp_files()
    initialize_command_services()
    stt_runtime.set_runtime_stt_settings(language_hint="auto")

    # Block startup until warm-up completes so wake-word listening begins on a fully loaded runtime.
    _run_startup_prewarm_blocking()

    if KB_AUTO_SYNC_ENABLED:
        ok, message = knowledge_base_service.start_auto_sync()
        if ok:
            logger.info("Knowledge-base auto-sync startup: %s", message)
        else:
            logger.warning("Knowledge-base auto-sync startup skipped: %s", message)

    if DOCTOR_STARTUP_ENABLED:
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
    logger.info("Jarvis started")

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
            wake_behavior = get_runtime_wake_word_behavior()
            if wake_behavior.get("ignore_while_speaking") and speech_engine.is_speaking():
                time.sleep(0.1)
                continue

            wake_started = time.perf_counter()
            if dialogue_manager.should_skip_wake_word():
                # Already in FOLLOW_UP or CONFIRMING — bypass the wake-word
                # listener entirely.  The dialogue manager's state acts as the
                # gate; no need to run the ONNX wake-word model for this turn.
                wake_source = "follow_up"
                metrics.record_stage("wake_word", 0.0, success=True)
            else:
                try:
                    wake_source = listen_for_wake_word()
                    metrics.record_stage("wake_word", time.perf_counter() - wake_started, success=True)
                except RuntimeError as exc:
                    metrics.record_stage("wake_word", time.perf_counter() - wake_started, success=False)
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
                _wait_for_tts_completion(max_wait=12.0)
                # Optional chime: audible cue that the follow-up window is now
                # open and the user may speak without the wake word.
                _play_follow_up_chime()
                remaining = dialogue_manager.time_remaining()
                if remaining < 0.5:
                    logger.info("Follow-up window expired; returning to IDLE.")
                    dialogue_manager.transition(DialogueState.IDLE)
                    continue
            elif wake_source == "barge_in":
                # VAD barge-in already interrupted TTS; log and skip to recording.
                dialogue_manager.transition(DialogueState.LISTENING)
                logger.info("VAD barge-in: TTS interrupted, entering listening mode directly.")
                metrics.record_stage("barge_in_interrupt", 0.0, success=True)
            elif wake_behavior.get("barge_in_interrupt_on_wake") and speech_engine.is_speaking():
                dialogue_manager.transition(DialogueState.LISTENING)
                speech_engine.interrupt()
                logger.info("Speech interrupted due to wake-word barge-in.")
                metrics.record_stage("barge_in_interrupt", 0.0, success=True)
            else:
                dialogue_manager.transition(DialogueState.LISTENING)
                logger.info("Wake word detected via %s", wake_source or "unknown")
            pipeline_started = time.perf_counter()

            in_flight = _prune_futures(in_flight)
            busy = len(in_flight) >= max(1, int(REALTIME_MAX_PENDING_UTTERANCES))
            if busy and REALTIME_DROP_WHEN_BUSY:
                logger.warning("Dropping wake event due to pipeline backpressure.")
                metrics.record_stage("backpressure_drop", 0.0, success=False)
                continue

            audio_file = _create_utterance_audio_file()
            record_started = time.perf_counter()

            # Task 1.3: create pipeline before recording so partials are
            # processed concurrently while the user is still speaking.
            concurrent_pipeline = ConcurrentPipeline(
                executor,
                language_hint=_resolve_stt_language_hint(wake_source=wake_source) or "",
            )

            _partial_latency_recorded = False

            def _pipeline_partial(partial_text):
                nonlocal _partial_latency_recorded
                _on_partial_transcript(partial_text)
                concurrent_pipeline.on_partial(partial_text)
                if not _partial_latency_recorded and (partial_text or "").strip():
                    _partial_latency_recorded = True
                    latency_tracker.record("stt_partial_latency", time.perf_counter() - record_started)

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
                on_partial=_pipeline_partial,
            )
            metrics.record_stage(
                "record_audio",
                time.perf_counter() - record_started,
                success=bool(capture.get("speech_detected")),
            )

            if shutdown_event.is_set():
                _safe_remove(audio_file)
                break

            if not capture.get("speech_detected"):
                _safe_remove(audio_file)
                if wake_source == "follow_up":
                    logger.info("No speech in follow-up window; returning to IDLE.")
                    dialogue_manager.transition(DialogueState.IDLE)
                continue

            in_flight.append(
                executor.submit(
                    _process_utterance,
                    audio_file,
                    pipeline_started,
                    wake_source,
                    capture,
                    concurrent_pipeline,
                )
            )
    finally:
        try:
            knowledge_base_service.stop_auto_sync()
        except Exception:
            pass
        perform_shutdown_cleanup()
        executor.shutdown(wait=False, cancel_futures=False)


if __name__ == "__main__":
    run()
