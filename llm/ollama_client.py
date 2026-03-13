import subprocess
import time

from core.config import LLM_FALLBACK_MODELS, LLM_MODEL, LLM_TIMEOUT_SECONDS
from core.logger import logger
from core.metrics import metrics


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
                result = subprocess.run(
                    ["ollama", "run", model_name],
                    input=prompt,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=LLM_TIMEOUT_SECONDS,
                )
            except subprocess.TimeoutExpired:
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
            latency = time.perf_counter() - started
            logger.info("LLM latency: %.2fs (model=%s)", latency, model_name)

            if result.returncode == 0:
                response = (result.stdout or "").strip()
                if response:
                    if index > 0:
                        logger.warning("LLM fallback model used: %s", model_name)
                    success = True
                    return response
                last_error = "empty_response"
                continue

            err_text = (result.stderr or "").strip()
            last_error = err_text or f"return_code={result.returncode}"
            if index < len(models) - 1 and _should_try_fallback(err_text):
                logger.warning(
                    "LLM model '%s' failed (%s). Trying fallback model '%s'.",
                    model_name,
                    last_error,
                    models[index + 1],
                )
                continue
            logger.error(
                "LLM command failed with code %s (model=%s): %s",
                result.returncode,
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
