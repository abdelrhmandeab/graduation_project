import json
import re
import time

import httpx

from core.config import (
    LLM_FALLBACK_MODELS,
    LLM_MAX_RESPONSE_TOKENS,
    LLM_MODEL,
    LLM_OLLAMA_BASE_URL,
    LLM_OLLAMA_NUM_CTX,
    LLM_REPEAT_PENALTY,
    LLM_STOP_TOKENS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT_SECONDS,
    LLM_TOP_P,
)
from core.logger import logger
from core.metrics import metrics, record_stage_timing
from llm.sentence_buffer import SentenceBuffer

_OLLAMA_BASE_URL = str(LLM_OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
_GENERATE_ENDPOINT = f"{_OLLAMA_BASE_URL}/api/generate"

import threading

llm_cancel_event = threading.Event()

# Resolved at startup by set_runtime_model(); falls back to config value.
_runtime_model = None
_runtime_num_ctx = None
_runtime_lightweight_num_ctx = None
_runtime_model_tier = None

# qwen3 family emits <think>...</think> reasoning blocks that burn predict tokens
# and (depending on Ollama version) leak into the response field. Strip them.
_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _is_thinking_mode_model(model_name):
    """Return True for models with internal reasoning that should be suppressed."""
    name = str(model_name or "").strip().lower()
    return name.startswith("qwen3:") or name == "qwen3"


def _strip_thinking_tags(text):
    """Remove <think>...</think> blocks (closed) and orphan opening tags from streamed text."""
    if not text:
        return text
    cleaned = _THINK_TAG_RE.sub("", str(text))
    # Drop everything before an unclosed <think>...EOF (rare partial chunks)
    if "<think>" in cleaned and "</think>" not in cleaned:
        cleaned = cleaned.split("<think>", 1)[0]
    # Drop everything before a stray </think> (model leaked partial reasoning)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    return cleaned.strip()


def _decode_stop_tokens(tokens):
    decoded = []
    for token in tokens or []:
        value = str(token or "")
        if not value:
            continue
        decoded.append(value.replace("\\n", "\n").replace("\\t", "\t"))
    return decoded


def _build_request_payload(model_name, prompt, num_ctx, stream):
    """Construct an Ollama /api/generate payload, suppressing thinking when needed.

    Adds keep_alive=30m so the model stays resident in RAM between user queries
    (default Ollama keep_alive is 5m which causes cold-loads after a brief idle).
    """
    effective_prompt = str(prompt or "")
    payload = {
        "model": model_name,
        "prompt": effective_prompt,
        "stream": bool(stream),
        "keep_alive": "30m",
        "options": {
            "num_ctx": int(num_ctx),
            "temperature": float(LLM_TEMPERATURE),
            "top_p": float(LLM_TOP_P),
            "repeat_penalty": float(LLM_REPEAT_PENALTY),
            "num_predict": int(LLM_MAX_RESPONSE_TOKENS),
            "stop": _decode_stop_tokens(LLM_STOP_TOKENS),
        },
    }
    if _is_thinking_mode_model(model_name):
        # Belt-and-suspenders: top-level think flag (Ollama 0.9+) + prompt suffix
        payload["think"] = False
        if "/no_think" not in effective_prompt:
            payload["prompt"] = effective_prompt.rstrip() + "\n\n/no_think"
    return payload


def set_runtime_model(model_name, num_ctx=None, lightweight_num_ctx=None, tier=None):
    """Called once at startup after hardware detection to lock in runtime LLM settings.
    
    Args:
        model_name: e.g. "qwen3:4b"
        num_ctx: Context window size
        lightweight_num_ctx: Lightweight context size
        tier: Model tier for prompt selection ("minimal", "low", "medium", "high")
    """
    global _runtime_model, _runtime_num_ctx, _runtime_lightweight_num_ctx, _runtime_model_tier
    _runtime_model = str(model_name or "").strip() or None
    if num_ctx is not None:
        _runtime_num_ctx = int(num_ctx)
    if lightweight_num_ctx is not None:
        _runtime_lightweight_num_ctx = int(lightweight_num_ctx)
    if tier is not None:
        _runtime_model_tier = str(tier).strip().lower()
    logger.debug(
        "Runtime LLM model set to '%s' (tier=%s, num_ctx=%s, lightweight_num_ctx=%s)",
        _runtime_model,
        _runtime_model_tier or "auto",
        _runtime_num_ctx,
        _runtime_lightweight_num_ctx,
    )


def get_runtime_num_ctx(default=None):
    """Return runtime-selected num_ctx if available, else fallback default."""
    value = _runtime_num_ctx if _runtime_num_ctx is not None else default
    if value is None:
        value = LLM_OLLAMA_NUM_CTX
    return int(value)


def get_runtime_lightweight_num_ctx(default=None):
    """Return runtime-selected lightweight num_ctx if available, else fallback default."""
    value = _runtime_lightweight_num_ctx if _runtime_lightweight_num_ctx is not None else default
    if value is None:
        value = LLM_OLLAMA_NUM_CTX
    return int(value)


def get_runtime_model_tier(default="medium"):
    """Return the runtime-selected model tier for prompt selection.
    
    Args:
        default: Default tier if not set at runtime (default: "medium")
        
    Returns:
        One of "minimal", "low", "medium", "high"
    """
    if _runtime_model_tier:
        return _runtime_model_tier
    # Infer tier from runtime model name if available
    if _runtime_model:
        from llm.prompt_builder import _get_model_tier
        return _get_model_tier(_runtime_model)
    return str(default).strip().lower() or "medium"


def _resolve_model_name():
    if _runtime_model:
        return _runtime_model
    configured = str(LLM_MODEL or "").strip()
    return configured or "qwen3:4b"


def _resolve_model_candidates(primary_model: str):
    primary = str(primary_model or "").strip()
    ordered = []
    seen = set()

    for candidate in [primary, *list(LLM_FALLBACK_MODELS or ())]:
        name = str(candidate or "").strip()
        if not name:
            continue
        lowered = name.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        ordered.append(name)

    return ordered


def prewarm_model(timeout_seconds=None):
    """Load the active Ollama model into memory without generating text.

    Ollama treats an empty prompt as a load-only request. This avoids spending
    startup time generating a throwaway response and works reliably on slower
    CPU-only machines.
    """
    model_name = _resolve_model_name()
    effective_num_ctx = int(_runtime_lightweight_num_ctx or _runtime_num_ctx or LLM_OLLAMA_NUM_CTX)
    request_timeout = float(timeout_seconds or LLM_TIMEOUT_SECONDS)
    response = httpx.post(
        _GENERATE_ENDPOINT,
        json={
            "model": model_name,
            "prompt": "",
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_ctx": effective_num_ctx,
                "temperature": float(LLM_TEMPERATURE),
                "top_p": float(LLM_TOP_P),
                "repeat_penalty": float(LLM_REPEAT_PENALTY),
                "num_predict": int(LLM_MAX_RESPONSE_TOKENS),
                "stop": _decode_stop_tokens(LLM_STOP_TOKENS),
            },
        },
        timeout=request_timeout,
    )
    response.raise_for_status()
    return model_name


def ask_llm_streaming(prompt, on_sentence=None, num_ctx=None, is_arabic=False, cancel_event=None):
    """Stream tokens from Ollama; call on_sentence(text) at each sentence boundary.

    Returns the complete accumulated response text, or an error string.
    Falls back to non-streaming ask_llm() when on_sentence is None.

    Args:
        is_arabic: When True, uses Arabic-aware sentence splitting (splits on ؟،؛
                   and performs soft/hard char-count flushes for un-punctuated text).
        cancel_event: Optional threading.Event checked after each token chunk.
                      When set, the stream is closed and partial text returned.
    """
    if on_sentence is None:
        return ask_llm(prompt, num_ctx=num_ctx)

    if cancel_event is None:
        cancel_event = llm_cancel_event

    started = time.perf_counter()
    success = False
    model_name = _resolve_model_name()
    accumulated = []
    sb = SentenceBuffer(is_arabic=bool(is_arabic))
    hard_timeout_seconds = max(5.0, float(LLM_TIMEOUT_SECONDS or 30.0))
    hard_timeout_hit = False
    first_token_recorded = False

    def _record_first_token():
        nonlocal first_token_recorded
        if first_token_recorded:
            return
        first_token_recorded = True
        record_stage_timing("llm_first_token", time.perf_counter() - started, model=model_name)

    effective_num_ctx = int(num_ctx or _runtime_num_ctx or LLM_OLLAMA_NUM_CTX)
    payload = _build_request_payload(model_name, prompt, effective_num_ctx, stream=True)
    suppress_thinking = _is_thinking_mode_model(model_name)
    inside_think_block = False
    try:
        with httpx.stream(
            "POST",
            _GENERATE_ENDPOINT,
            json=payload,
            timeout=LLM_TIMEOUT_SECONDS,
        ) as stream_response:
            if stream_response.status_code != 200:
                err = ""
                try:
                    err = stream_response.read().decode("utf-8", errors="replace")
                except Exception:
                    pass
                logger.error(
                    "LLM streaming request failed status=%s model=%s err=%s",
                    stream_response.status_code,
                    model_name,
                    err[:120],
                )
                return "I could not run the local model."

            for raw_line in stream_response.iter_lines():
                if cancel_event is not None and cancel_event.is_set():
                    logger.info("LLM stream cancelled by wake-word interrupt.")
                    break

                elapsed = time.perf_counter() - started
                if elapsed >= hard_timeout_seconds:
                    hard_timeout_hit = True
                    logger.error(
                        "LLM streaming hard-timeout after %.2fs (model=%s)",
                        elapsed,
                        model_name,
                    )
                    break

                line = (raw_line or "").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                token = chunk.get("response") or ""
                if token and suppress_thinking:
                    # Drop tokens that fall inside a <think>...</think> block so
                    # reasoning never reaches TTS or the user-visible transcript.
                    while token:
                        if inside_think_block:
                            close_idx = token.find("</think>")
                            if close_idx == -1:
                                token = ""  # whole chunk is reasoning, skip
                                break
                            token = token[close_idx + len("</think>"):]
                            inside_think_block = False
                        else:
                            open_idx = token.find("<think>")
                            if open_idx == -1:
                                break
                            pre = token[:open_idx]
                            token = token[open_idx + len("<think>"):]
                            inside_think_block = True
                            if pre:
                                _record_first_token()
                                accumulated.append(pre)
                                result = sb.add_token(pre)
                                if result:
                                    on_sentence(result)
                if token:
                    _record_first_token()
                    accumulated.append(token)
                    result = sb.add_token(token)
                    if result:
                        on_sentence(result)
                if chunk.get("done"):
                    break

        if hard_timeout_hit:
            remainder = sb.flush()
            if remainder:
                on_sentence(remainder)
            partial = "".join(accumulated).strip()
            if partial:
                success = True
                return partial
            return "The local model timed out. Try a shorter query."

        # Flush any remaining sentence fragment
        remainder = sb.flush()
        if remainder:
            on_sentence(remainder)

        full_text = "".join(accumulated).strip()
        if full_text:
            success = True
            latency = time.perf_counter() - started
            logger.info("LLM streaming latency: %.2fs (model=%s)", latency, model_name)
            return full_text

        logger.error("LLM streaming returned empty response (model=%s)", model_name)
        return "I could not run the local model."

    except httpx.TimeoutException:
        latency = time.perf_counter() - started
        logger.error(
            "LLM streaming timeout after %.2fs (model=%s)", latency, model_name
        )
        partial = "".join(accumulated).strip()
        if partial:
            success = True
            return partial
        fallback = ask_llm(prompt, num_ctx=num_ctx)
        if fallback and "timed out" not in str(fallback).lower():
            try:
                on_sentence(str(fallback))
            except Exception:
                pass
            success = True
            return str(fallback)
        return "The local model timed out. Try a shorter query."
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama at %s. Is it running?", _OLLAMA_BASE_URL)
        fallback = ask_llm(prompt, num_ctx=num_ctx)
        if fallback and "cannot connect to ollama" not in str(fallback).lower():
            try:
                on_sentence(str(fallback))
            except Exception:
                pass
            success = True
            return str(fallback)
        return "Cannot connect to Ollama. Make sure it is running."
    except Exception as exc:
        logger.error("LLM streaming failed: %s", exc)
        return "Sorry, I had an internal error."
    finally:
        llm_elapsed = time.perf_counter() - started
        metrics.record_stage("llm", llm_elapsed, success=success)
        record_stage_timing("llm_generation", llm_elapsed, model=model_name)


def ask_llm(prompt, num_ctx=None, timeout_seconds=None, allow_fallbacks=True):
    started = time.perf_counter()
    success = False
    try:
        model_name = _resolve_model_name()
        effective_num_ctx = int(num_ctx or _runtime_num_ctx or LLM_OLLAMA_NUM_CTX)
        candidates = _resolve_model_candidates(model_name) if allow_fallbacks else [model_name]
        request_timeout = float(timeout_seconds or LLM_TIMEOUT_SECONDS)
        timeout_seen = False
        connect_seen = False

        for idx, candidate_model in enumerate(candidates):
            payload = _build_request_payload(candidate_model, prompt, effective_num_ctx, stream=False)
            try:
                response = httpx.post(
                    _GENERATE_ENDPOINT,
                    json=payload,
                    timeout=request_timeout,
                )
            except httpx.TimeoutException:
                timeout_seen = True
                logger.error(
                    "LLM timeout after %.2fs (model=%s, timeout=%ss)",
                    time.perf_counter() - started,
                    candidate_model,
                    request_timeout,
                )
                continue
            except httpx.ConnectError:
                connect_seen = True
                logger.error("Cannot connect to Ollama at %s. Is it running?", _OLLAMA_BASE_URL)
                continue

            latency = time.perf_counter() - started
            logger.info("LLM latency: %.2fs (model=%s)", latency, candidate_model)

            if response.status_code == 200:
                data = response.json()
                text = (data.get("response") or "").strip()
                if _is_thinking_mode_model(candidate_model):
                    text = _strip_thinking_tags(text)
                if text:
                    record_stage_timing("llm_first_token", time.perf_counter() - started, model=candidate_model)
                    if idx > 0:
                        logger.warning("LLM fallback used: %s -> %s", model_name, candidate_model)
                    success = True
                    return text
                logger.error("LLM returned an empty response (model=%s)", candidate_model)
                continue

            err_text = ""
            try:
                err_text = response.json().get("error", "")
            except Exception:
                err_text = response.text or ""

            logger.error(
                "LLM request failed with status %s (model=%s): %s",
                response.status_code,
                candidate_model,
                err_text or "unknown_error",
            )

        if connect_seen and not timeout_seen:
            return "Cannot connect to Ollama. Make sure it is running."
        if timeout_seen:
            return "The local model timed out. Try a shorter query."
        return "I could not run the local model."
    except Exception as exc:
        logger.error("LLM failed: %s", exc)
        return "Sorry, I had an internal error."
    finally:
        llm_elapsed = time.perf_counter() - started
        metrics.record_stage("llm", llm_elapsed, success=success)
        record_stage_timing("llm_generation", llm_elapsed, model=_resolve_model_name())
