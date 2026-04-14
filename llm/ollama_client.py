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

_OLLAMA_BASE_URL = str(LLM_OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
_GENERATE_ENDPOINT = f"{_OLLAMA_BASE_URL}/api/generate"
_PINNED_MODEL = "qwen2.5:3b"

# Sentence boundary characters for streaming sentence detection
_SENTENCE_END_RE = re.compile(r"(?<=[.!?؟\n])\s+|(?<=[.!?؟])$")


def _resolve_model_name():
    configured = str(LLM_MODEL or "").strip()
    if configured and configured.lower() != _PINNED_MODEL:
        logger.warning(
            "Configured LLM model '%s' ignored; runtime is pinned to '%s'.",
            configured,
            _PINNED_MODEL,
        )
    return _PINNED_MODEL


def _flush_sentence_buffer(buf, on_sentence):
    """Emit any complete sentences from buf; return leftover fragment."""
    text = buf.strip()
    if not text:
        return ""
    parts = _SENTENCE_END_RE.split(text)
    if len(parts) <= 1:
        return buf  # no complete sentence yet
    # Everything except the last fragment is a complete sentence (or empty)
    for part in parts[:-1]:
        sentence = part.strip()
        if sentence:
            on_sentence(sentence)
    return parts[-1]  # leftover fragment


def ask_llm_streaming(prompt, on_sentence=None, num_ctx=None):
    """Stream tokens from Ollama; call on_sentence(text) at each sentence boundary.

    Returns the complete accumulated response text, or an error string.
    Falls back to non-streaming ask_llm() when on_sentence is None.
    """
    if on_sentence is None:
        return ask_llm(prompt, num_ctx=num_ctx)

    started = time.perf_counter()
    success = False
    model_name = _resolve_model_name()
    accumulated = []
    sentence_buf = ""

    try:
        with httpx.stream(
            "POST",
            _GENERATE_ENDPOINT,
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": True,
                "options": {"num_ctx": int(num_ctx or LLM_OLLAMA_NUM_CTX)},
            },
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
                line = (raw_line or "").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:
                    continue
                token = chunk.get("response") or ""
                if token:
                    accumulated.append(token)
                    sentence_buf += token
                    sentence_buf = _flush_sentence_buffer(sentence_buf, on_sentence)
                if chunk.get("done"):
                    break

        # Flush any remaining sentence fragment
        remainder = sentence_buf.strip()
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
        # Return whatever we accumulated before the timeout
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
        try:
            response = httpx.post(
                _GENERATE_ENDPOINT,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": int(num_ctx or LLM_OLLAMA_NUM_CTX),
                    },
                },
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
