import json
import re
import time

import httpx

from core.config import (
    LLM_MODEL,
    LLM_OLLAMA_BASE_URL,
    LLM_OLLAMA_NUM_CTX,
    LLM_TIMEOUT_SECONDS,
)
from core.logger import logger
from core.metrics import metrics
from llm.sentence_buffer import SentenceBuffer

_OLLAMA_BASE_URL = str(LLM_OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
_GENERATE_ENDPOINT = f"{_OLLAMA_BASE_URL}/api/generate"

# Resolved at startup by set_runtime_model(); falls back to config value.
_runtime_model = None
_runtime_num_ctx = None
_runtime_lightweight_num_ctx = None
_runtime_model_tier = None  # Track tier for tiered prompt selection

# Sentence boundary characters for streaming sentence detection
_SENTENCE_END_RE = re.compile(r"(?<=[.!?؟\n])\s+|(?<=[.!?؟])$")
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_STREAM_FLUSH_WORDS = 7
_STREAM_FLUSH_CHARS = 90

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
        "options": {"num_ctx": int(num_ctx)},
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
    logger.info(
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


def _flush_sentence_buffer(buf, on_sentence):
    """Emit any complete sentences from buf; return leftover fragment."""
    text = buf.strip()
    if not text:
        return ""

    chunks = detect_sentence_boundaries(text, is_arabic=bool(_ARABIC_CHAR_RE.search(text)))
    if not chunks:
        return buf

    for chunk in chunks:
        if chunk:
            on_sentence(chunk)
    return ""


def detect_sentence_boundaries(text: str, is_arabic: bool) -> list[str]:
    """Split streamed text into speakable chunks.

    Arabic text often arrives with weak punctuation, so we keep normal sentence
    splitting for punctuation and add a conservative forced flush when a chunk is
    clearly long enough to speak naturally but still has no boundary.
    """
    value = str(text or "").strip()
    if not value:
        return []

    parts = [part.strip() for part in _SENTENCE_END_RE.split(value) if part.strip()]
    if len(parts) > 1:
        return parts

    if is_arabic:
        word_count = len(value.split())
        char_count = len(value)
        if word_count >= _STREAM_FLUSH_WORDS or char_count >= _STREAM_FLUSH_CHARS:
            return [value]

    return []


def ask_llm_streaming(prompt, on_sentence=None, num_ctx=None, is_arabic=False):
    """Stream tokens from Ollama; call on_sentence(text) at each sentence boundary.

    Returns the complete accumulated response text, or an error string.
    Falls back to non-streaming ask_llm() when on_sentence is None.

    Args:
        is_arabic: When True, uses Arabic-aware sentence splitting (splits on ؟،؛
                   and performs soft/hard char-count flushes for un-punctuated text).
    """
    if on_sentence is None:
        return ask_llm(prompt, num_ctx=num_ctx)

    started = time.perf_counter()
    success = False
    model_name = _resolve_model_name()
    accumulated = []
    sb = SentenceBuffer(is_arabic=bool(is_arabic))
    hard_timeout_seconds = max(5.0, float(LLM_TIMEOUT_SECONDS or 30.0))
    hard_timeout_hit = False

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
                                accumulated.append(pre)
                                result = sb.add_token(pre)
                                if result:
                                    on_sentence(result)
                if token:
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
        return "The local model timed out. Try a shorter query."
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama at %s. Is it running?", _OLLAMA_BASE_URL)
        return "Cannot connect to Ollama. Make sure it is running."
    except Exception as exc:
        logger.error("LLM streaming failed: %s", exc)
        return "Sorry, I had an internal error."
    finally:
        metrics.record_stage("llm", time.perf_counter() - started, success=success)


def ask_llm(prompt, num_ctx=None):
    started = time.perf_counter()
    success = False
    try:
        model_name = _resolve_model_name()
        effective_num_ctx = int(num_ctx or _runtime_num_ctx or LLM_OLLAMA_NUM_CTX)
        payload = _build_request_payload(model_name, prompt, effective_num_ctx, stream=False)
        try:
            response = httpx.post(
                _GENERATE_ENDPOINT,
                json=payload,
                timeout=LLM_TIMEOUT_SECONDS,
            )
        except httpx.TimeoutException:
            latency = time.perf_counter() - started
            logger.error(
                "LLM timeout after %.2fs (model=%s, timeout=%ss)",
                latency,
                model_name,
                LLM_TIMEOUT_SECONDS,
            )
            return "The local model timed out. Try a shorter query."
        except httpx.ConnectError:
            logger.error("Cannot connect to Ollama at %s. Is it running?", _OLLAMA_BASE_URL)
            return "Cannot connect to Ollama. Make sure it is running."

        latency = time.perf_counter() - started
        logger.info("LLM latency: %.2fs (model=%s)", latency, model_name)

        if response.status_code == 200:
            data = response.json()
            text = (data.get("response") or "").strip()
            if _is_thinking_mode_model(model_name):
                text = _strip_thinking_tags(text)
            if text:
                success = True
                return text

            logger.error("LLM returned an empty response (model=%s)", model_name)
            return "I could not run the local model."

        err_text = ""
        try:
            err_text = response.json().get("error", "")
        except Exception:
            err_text = response.text or ""

        logger.error(
            "LLM request failed with status %s (model=%s): %s",
            response.status_code,
            model_name,
            err_text or "unknown_error",
        )
        return "I could not run the local model."
    except Exception as exc:
        logger.error("LLM failed: %s", exc)
        return "Sorry, I had an internal error."
    finally:
        metrics.record_stage("llm", time.perf_counter() - started, success=success)
