import re
import time

import httpx

from core.config import (
    LLM_CPU_UPGRADE_MAX_LATENCY_SECONDS,
    LLM_CPU_UPGRADE_MODEL,
    LLM_CPU_UPGRADE_TEST_ENABLED,
    LLM_FALLBACK_MODELS,
    LLM_MODEL,
    LLM_TIMEOUT_SECONDS,
)
from core.logger import logger
from core.metrics import metrics

_OLLAMA_BASE_URL = "http://localhost:11434"
_GENERATE_ENDPOINT = f"{_OLLAMA_BASE_URL}/api/generate"

_QUALITY_RETRY_SKIP_PROMPT_MARKERS = (
    "strict intent extraction engine",
    "output schema",
    "allowed intents",
)

_LOW_VALUE_REPLY_MARKERS = (
    "i can help with that",
    "i can certainly help with that",
    "please provide me with some information",
    "do you have any other questions",
    "let me know if you have any other questions",
    "feel free to ask if you have any other questions",
    "let me know what information",
    "i cannot assist with that directly",
    "i can t assist with that directly",
    "i m sorry but i can t provide current weather information",
    "i cannot provide current weather information",
    "can t provide current weather information",
    "cannot provide current weather information",
    "i can t provide live weather",
    "i cannot provide live weather",
    "please check a weather service",
    "check a weather service",
    "i can t provide current news",
    "i cannot provide current news",
    "please check a reliable news source",
    "بالطبع يمكنني القيام بذلك",
    "هل لديك اي اسئلة اخرى",
    "هل هناك اي معلومات اخرى",
    "يمكنني مساعدتك في ذلك",
    "لا استطيع مساعدتك",
    "لا يمكنني مساعدتك",
    "اعتذر لكنني",
    "لا استطيع تقديم معلومات الطقس الحالية",
    "لا يمكنني تقديم معلومات الطقس الحالية",
    "يرجى التحقق من خدمة طقس",
)

_UPGRADE_MODEL_NORMALIZED = str(LLM_CPU_UPGRADE_MODEL or "").strip().lower()
_UPGRADE_MODEL_DISABLED_FOR_QUALITY_RETRY = False


def _normalize_model_name(model_name):
    return str(model_name or "").strip().lower()


def _is_upgrade_model(model_name):
    if not _UPGRADE_MODEL_NORMALIZED:
        return False
    return _normalize_model_name(model_name) == _UPGRADE_MODEL_NORMALIZED


def _candidate_models(*, quality_retry_enabled=False):
    global _UPGRADE_MODEL_DISABLED_FOR_QUALITY_RETRY

    ordered = [LLM_MODEL]

    if quality_retry_enabled and LLM_CPU_UPGRADE_TEST_ENABLED:
        upgrade_model = str(LLM_CPU_UPGRADE_MODEL or "").strip()
        if (
            upgrade_model
            and _normalize_model_name(upgrade_model) != _normalize_model_name(LLM_MODEL)
            and not _UPGRADE_MODEL_DISABLED_FOR_QUALITY_RETRY
        ):
            ordered.append(upgrade_model)

    ordered.extend(LLM_FALLBACK_MODELS or ())
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


def _normalize_quality_text(text):
    raw = str(text or "").strip().lower()
    if not raw:
        return ""
    normalized = (
        raw.replace("أ", "ا")
        .replace("إ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    normalized = re.sub(r"[^\w\s\u0600-\u06FF]", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _is_intent_extraction_prompt(prompt):
    normalized = _normalize_quality_text(prompt)
    if not normalized:
        return False
    return all(marker in normalized for marker in _QUALITY_RETRY_SKIP_PROMPT_MARKERS)


def _looks_low_value_reply(text):
    normalized = _normalize_quality_text(text)
    if not normalized:
        return True
    word_count = len(normalized.split())
    if word_count == 0:
        return True
    return any(marker in normalized for marker in _LOW_VALUE_REPLY_MARKERS) and word_count <= 36


_MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b\b")


def _extract_model_size_billions(model_name):
    text = str(model_name or "").strip().lower()
    if not text:
        return None
    match = _MODEL_SIZE_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def _should_retry_low_value_with_fallback(models, index):
    next_index = int(index) + 1
    if next_index >= len(models):
        return False

    current_size = _extract_model_size_billions(models[index])
    next_size = _extract_model_size_billions(models[next_index])

    # If both sizes are known, only retry when fallback is not weaker.
    if current_size is not None and next_size is not None:
        return next_size >= current_size
    return True


def ask_llm(prompt):
    started = time.perf_counter()
    success = False
    try:
        quality_retry_enabled = not _is_intent_extraction_prompt(prompt)
        models = _candidate_models(quality_retry_enabled=quality_retry_enabled)
        last_error = ""
        best_effort_text = ""
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
                if best_effort_text:
                    logger.warning(
                        "Returning best-effort LLM response after timeout on '%s'.",
                        model_name,
                    )
                    break
                return "The local model timed out. Try a shorter query or use a smaller model."
            except httpx.ConnectError:
                logger.error("Cannot connect to Ollama at %s. Is it running?", _OLLAMA_BASE_URL)
                return "Cannot connect to Ollama. Make sure it is running."

            latency = time.perf_counter() - started
            logger.info("LLM latency: %.2fs (model=%s)", latency, model_name)

            if _is_upgrade_model(model_name) and latency > float(LLM_CPU_UPGRADE_MAX_LATENCY_SECONDS):
                global _UPGRADE_MODEL_DISABLED_FOR_QUALITY_RETRY
                _UPGRADE_MODEL_DISABLED_FOR_QUALITY_RETRY = True
                logger.warning(
                    "LLM upgrade model '%s' exceeded latency budget (%.2fs > %.2fs); disabling quality-retry use.",
                    model_name,
                    latency,
                    float(LLM_CPU_UPGRADE_MAX_LATENCY_SECONDS),
                )

            if response.status_code == 200:
                data = response.json()
                text = (data.get("response") or "").strip()
                if text:
                    if not best_effort_text:
                        best_effort_text = text
                    if quality_retry_enabled and _looks_low_value_reply(text) and index < len(models) - 1:
                        if _should_retry_low_value_with_fallback(models, index):
                            logger.warning(
                                "LLM response from '%s' looked low-value. Trying fallback model '%s'.",
                                model_name,
                                models[index + 1],
                            )
                            continue
                        logger.info(
                            "LLM response from '%s' looked low-value, but skipping weaker fallback '%s'.",
                            model_name,
                            models[index + 1],
                        )
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

            if index < len(models) - 1 and (
                _should_try_fallback(err_text)
                or (quality_retry_enabled and bool(best_effort_text))
            ):
                logger.warning(
                    "LLM model '%s' failed (%s). Trying fallback model '%s'.",
                    model_name,
                    last_error,
                    models[index + 1],
                )
                continue

            if best_effort_text:
                logger.warning(
                    "Returning best-effort LLM response after model '%s' failed (%s).",
                    model_name,
                    last_error,
                )
                break

            logger.error(
                "LLM request failed with status %s (model=%s): %s",
                response.status_code,
                model_name,
                last_error,
            )
            return "I could not run the local model."

        if best_effort_text:
            success = True
            return best_effort_text

        logger.error("LLM failed after trying models %s: %s", models, last_error)
        return "I could not run the local model."
    except Exception as exc:
        logger.error("LLM failed: %s", exc)
        return "Sorry, I had an internal error."
    finally:
        metrics.record_stage("llm", time.perf_counter() - started, success=success)
