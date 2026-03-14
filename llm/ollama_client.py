import time

import httpx

from core.config import LLM_FALLBACK_MODELS, LLM_MODEL, LLM_TIMEOUT_SECONDS
from core.logger import logger
from core.metrics import metrics

_OLLAMA_BASE_URL = "http://localhost:11434"
_GENERATE_ENDPOINT = f"{_OLLAMA_BASE_URL}/api/generate"


def _candidate_models():
    ordered = [LLM_MODEL, *(LLM_FALLBACK_MODELS or ())]
    unique = []
    seen = set()
    for model in ordered:
        key = str(model or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def _should_try_fallback(error_text):
    text = (error_text or "").lower()
    return (
        "model requires more system memory" in text
        or "model '" in text and "not found" in text
        or "pull model manifest" in text
        or "manifest does not exist" in text
    )


def ask_llm(prompt):
    started = time.perf_counter()
    success = False
    try:
        models = _candidate_models()
        last_error = ""
        for index, model_name in enumerate(models):
            try:
                response = httpx.post(
                    _GENERATE_ENDPOINT,
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=LLM_TIMEOUT_SECONDS,
                )
            except httpx.TimeoutException:
                latency = time.perf_counter() - started
                last_error = f"timeout_after={LLM_TIMEOUT_SECONDS}s"
                logger.error(
                    "LLM timeout after %.2fs (model=%s, timeout=%ss)",
                    latency,
                    model_name,
                    LLM_TIMEOUT_SECONDS,
                )
                if index < len(models) - 1:
                    logger.warning(
                        "Trying fallback model '%s' after timeout on '%s'.",
                        models[index + 1],
                        model_name,
                    )
                    continue
                return "The local model timed out. Try a shorter query or use a smaller model."
            except httpx.ConnectError:
                logger.error("Cannot connect to Ollama at %s. Is it running?", _OLLAMA_BASE_URL)
                return "Cannot connect to Ollama. Make sure it is running."

            latency = time.perf_counter() - started
            logger.info("LLM latency: %.2fs (model=%s)", latency, model_name)

            if response.status_code == 200:
                data = response.json()
                text = (data.get("response") or "").strip()
                if text:
                    if index > 0:
                        logger.warning("LLM fallback model used: %s", model_name)
                    success = True
                    return text
                last_error = "empty_response"
                continue

            err_text = ""
            try:
                err_text = response.json().get("error", "")
            except Exception:
                err_text = response.text or ""
            last_error = err_text or f"status_code={response.status_code}"

            if index < len(models) - 1 and _should_try_fallback(err_text):
                logger.warning(
                    "LLM model '%s' failed (%s). Trying fallback model '%s'.",
                    model_name,
                    last_error,
                    models[index + 1],
                )
                continue

            logger.error(
                "LLM request failed with status %s (model=%s): %s",
                response.status_code,
                model_name,
                last_error,
            )
            return "I could not run the local model."

        logger.error("LLM failed after trying models %s: %s", models, last_error)
        return "I could not run the local model."
    except Exception as exc:
        logger.error("LLM failed: %s", exc)
        return "Sorry, I had an internal error."
    finally:
        metrics.record_stage("llm", time.perf_counter() - started, success=success)
