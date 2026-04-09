import glob
import json
import os
import re
import tempfile
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.mic import record_utterance
from audio import stt as stt_runtime
from audio.stt import transcribe_streaming
from audio.tts import speech_engine
from audio.vad import is_speech
from audio.wake_word import get_runtime_wake_word_behavior, listen_for_wake_word
from core.command_parser import parse_command
from core.command_router import initialize_command_services, route_command
from core.doctor import collect_diagnostics
from core.config import (
    DOCTOR_INCLUDE_MODEL_LOAD_CHECKS,
    DOCTOR_SCHEDULE_INTERVAL_SECONDS,
    DOCTOR_STARTUP_ENABLED,
    MAX_RECORD_DURATION,
    REALTIME_BACKPRESSURE_POLL_SECONDS,
    REALTIME_DROP_WHEN_BUSY,
    REALTIME_MAX_PENDING_UTTERANCES,
)
from core.intent_confidence import assess_intent_confidence
from core.logger import logger
from core.language_gate import detect_supported_language
from core.metrics import metrics
from core.session_memory import session_memory
from core.shutdown import perform_shutdown_cleanup, setup_shutdown


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


def _resolve_stt_language_hint(*, wake_source=None):
    _ = wake_source
    # Keep realtime STT in true auto-detect mode for every utterance.
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


def _should_retry_with_english_hint(text, detected_language, wake_source=None, language_confidence=0.0):
    _ = text, detected_language, wake_source, language_confidence
    # Orchestrator-level language retries are disabled intentionally.
    # Language recovery stays inside STT internals only.
    return False


def _transcribe_with_auto_then_english_retry(audio_file, wake_source=None):
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
            speech_engine.speak_async(_speech_safe_response(response))


def _process_utterance(audio_file, pipeline_started, wake_source=None):
    text = ""
    route_success = False
    try:
        active_audio_ux_profile = str(session_memory.get_audio_ux_profile() or "").strip().lower()
        skip_post_capture_guard = active_audio_ux_profile in _LOW_LATENCY_AUDIO_UX_PROFILES

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

        stt_started = time.perf_counter()
        _ = _resolve_stt_language_hint(
            wake_source=wake_source,
        )
        text, detected_language = _transcribe_with_auto_then_english_retry(
            audio_file,
            wake_source=wake_source,
        )
        if detected_language in {"ar", "en"}:
            session_memory.set_preferred_language(detected_language)
            session_memory.record_language_turn(detected_language)
        metrics.record_stage("stt", time.perf_counter() - stt_started, success=bool(text))
        if not text:
            logger.warning("No valid speech detected")
            return
        logger.info("Transcript[%s]: %s", detected_language or "unknown", _safe_log_text(text))

        route_started = time.perf_counter()
        try:
            response = route_command(text, detected_language=detected_language)
            route_success = True
            metrics.record_stage("router", time.perf_counter() - route_started, success=True)
        except Exception as exc:
            metrics.record_stage("router", time.perf_counter() - route_started, success=False)
            logger.error("Command routing failed: %s", exc)
            response = "Sorry, I had an internal error."

        print(f"Jarvis: {response}")
        if not _is_interrupt_command(text):
            speech_engine.speak_async(_speech_safe_response(response))
    finally:
        metrics.record_stage("pipeline", time.perf_counter() - pipeline_started, success=bool(text) and route_success)
        _safe_remove(audio_file)


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


def run():
    shutdown_event = setup_shutdown()
    _cleanup_stale_temp_files()
    initialize_command_services()
    stt_runtime.set_runtime_stt_settings(language_hint="auto")
    if DOCTOR_STARTUP_ENABLED:
        _run_doctor_diagnostics("startup")

    doctor_interval_seconds = max(0.0, float(DOCTOR_SCHEDULE_INTERVAL_SECONDS))
    next_doctor_run_at = time.time() + doctor_interval_seconds if doctor_interval_seconds > 0 else 0.0

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-pipeline")
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

            if wake_behavior.get("barge_in_interrupt_on_wake") and speech_engine.is_speaking():
                speech_engine.interrupt()
                logger.info("Speech interrupted due to wake-word barge-in.")

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
            capture = record_utterance(filename=audio_file, max_duration=MAX_RECORD_DURATION)
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
                continue

            in_flight.append(
                executor.submit(
                    _process_utterance,
                    audio_file,
                    pipeline_started,
                    wake_source,
                )
            )
    finally:
        perform_shutdown_cleanup()
        executor.shutdown(wait=False, cancel_futures=False)


if __name__ == "__main__":
    run()
