import time

import httpx

from core.config import (
    LLM_MODEL,
    LLM_OLLAMA_NUM_CTX,
    LLM_TIMEOUT_SECONDS,
)
from core.logger import logger
from core.metrics import metrics

_OLLAMA_BASE_URL = "http://localhost:11434"
_GENERATE_ENDPOINT = f"{_OLLAMA_BASE_URL}/api/generate"
_PINNED_MODEL = "qwen2.5:3b"

def _resolve_model_name():
    configured = str(LLM_MODEL or "").strip()
    if configured and configured.lower() != _PINNED_MODEL:
        logger.warning(
            "Configured LLM model '%s' ignored; runtime is pinned to '%s'.",
            configured,
            _PINNED_MODEL,
        )
    return _PINNED_MODEL


def ask_llm(prompt):
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
                        "num_ctx": int(LLM_OLLAMA_NUM_CTX),
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
