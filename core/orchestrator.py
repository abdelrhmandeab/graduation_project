import glob
import json
import os
import queue
import re
import subprocess
import tempfile
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.mic import record_utterance
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
from core.command_router import initialize_command_services, route_command
from core.doctor import collect_diagnostics
from core.config import (
    DOCTOR_INCLUDE_MODEL_LOAD_CHECKS,
    DOCTOR_SCHEDULE_INTERVAL_SECONDS,
    DOCTOR_STARTUP_ENABLED,
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
_OLLAMA_AUTOSTART_PROCESS = None


def _resolve_stt_language_hint(*, wake_source=None):
    _ = wake_source
    # Use the session's preferred language as a soft hint so the Egyptian model
    # is selected immediately for Arabic speakers, avoiding the extra auto-detect
    # pass on every utterance.  Falls back to None (fully automatic) when no
    # language preference has been established yet.
    lang = session_memory.get_preferred_language()
    if lang in {"ar", "en"}:
        return lang
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
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="jarvis-route-precompute") as executor:
            language_future = executor.submit(
                detect_supported_language,
                normalized_text,
                previous_language=previous_language,
            )
            parser_future = executor.submit(parse_command, normalized_text)
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


def _process_utterance(audio_file, pipeline_started, wake_source=None, capture_summary=None):
    text = ""
    route_success = False
    try:
        active_audio_ux_profile = str(session_memory.get_audio_ux_profile() or "").strip().lower()
        skip_post_capture_guard = active_audio_ux_profile in _LOW_LATENCY_AUDIO_UX_PROFILES
        if not skip_post_capture_guard and bool(SPEECH_GUARD_SKIP_NON_RESPONSIVE_PROFILES):
            capture_detected_speech = bool((capture_summary or {}).get("speech_detected"))
            if capture_detected_speech:
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

        stt_started = time.perf_counter()
        _ = _resolve_stt_language_hint(
            wake_source=wake_source,
        )
        text, detected_language = _transcribe_with_runtime_stt(
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

        def _on_sentence_streamed(sentence):
            if not (should_speak_response and sentence_queue_started):
                return
            normalized = _speech_safe_response(sentence)
            normalized = " ".join(str(normalized or "").split()).strip()
            if not normalized:
                return
            streamed_sentences.append(normalized)
            sentence_queue.put(normalized)

        route_started = time.perf_counter()
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
            metrics.record_stage("router", time.perf_counter() - route_started, success=True)
        except Exception as exc:
            metrics.record_stage("router", time.perf_counter() - route_started, success=False)
            logger.error("Command routing failed: %s", exc)
            response = "Sorry, I had an internal error."

        print(f"Jarvis: {response}")
        if should_speak_response:
            safe_response = _speech_safe_response(response)
            if sentence_queue_started:
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


def _run_startup_prewarm_blocking():
    _ensure_ollama_running()
    _detect_and_set_runtime_model()

    tasks = [
        ("wake_word", _preload_wake_word_runtime),
        ("stt", _preload_stt_model),
        ("llm", _prewarm_llm),
    ]
    if STARTUP_PARSER_NLP_PREWARM_ENABLED:
        tasks.append(("parser_nlp", _prewarm_parser_nlp))
    if SEMANTIC_ROUTER_ENABLED:
        tasks.append(("semantic_router", _prewarm_semantic_router))
    if TTS_PREWARM_ENABLED:
        tasks.append(("tts", _prewarm_tts))

    if not tasks:
        return

    logger.info("Startup prewarm started; waiting before wake-word listening begins.")
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
                    capture,
                )
            )
    finally:
        perform_shutdown_cleanup()
        executor.shutdown(wait=False, cancel_futures=False)


if __name__ == "__main__":
    run()
