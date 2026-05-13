"""Claude API client — streaming and non-streaming LLM calls.

Drop-in replacement for the Ollama backend when LLM_BACKEND=claude is set.
Uses claude-haiku-4-5 by default (fast, ~1s first token) and claude-sonnet-4-6
for quality-sensitive queries when the caller passes model=CLAUDE_QUALITY_MODEL.
"""
from __future__ import annotations

import time

from core.config import (
    ANTHROPIC_API_KEY,
    CLAUDE_DEFAULT_MODEL,
    CLAUDE_MAX_TOKENS_QUESTION,
)
from core.logger import logger
from core.metrics import metrics
from llm.sentence_buffer import SentenceBuffer

_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def ask_claude_streaming(
    system_prompt: str,
    user_text: str,
    on_sentence=None,
    *,
    is_arabic: bool = False,
    max_tokens: int | None = None,
    model: str | None = None,
    prior_messages: list | None = None,
) -> str:
    """Stream tokens from Claude; call on_sentence(text) at each sentence boundary.

    Returns the full accumulated response text, or an error string.
    Falls back to non-streaming ask_claude() when on_sentence is None.
    """
    if on_sentence is None:
        return ask_claude(system_prompt, user_text, max_tokens=max_tokens, model=model)

    started = time.perf_counter()
    success = False
    effective_model = model or CLAUDE_DEFAULT_MODEL
    effective_max_tokens = max_tokens or CLAUDE_MAX_TOKENS_QUESTION
    sb = SentenceBuffer(is_arabic=bool(is_arabic))
    accumulated: list[str] = []

    try:
        import anthropic as _anthropic

        client = _get_client()
        messages_payload = list(prior_messages or []) + [{"role": "user", "content": user_text}]
        with client.messages.stream(
            model=effective_model,
            max_tokens=effective_max_tokens,
            system=system_prompt,
            messages=messages_payload,
        ) as stream:
            for text_chunk in stream.text_stream:
                if not text_chunk:
                    continue
                accumulated.append(text_chunk)
                result = sb.add_token(text_chunk)
                if result:
                    on_sentence(result)

        remainder = sb.flush()
        if remainder:
            on_sentence(remainder)

        full_text = "".join(accumulated).strip()
        if full_text:
            success = True
            latency = time.perf_counter() - started
            logger.info("Claude streaming latency: %.2fs (model=%s)", latency, effective_model)
            return full_text

        logger.error("Claude streaming returned empty response (model=%s)", effective_model)
        return "I could not get a response."

    except _anthropic.AuthenticationError:
        logger.error("Claude API authentication failed — check ANTHROPIC_API_KEY in .env")
        return "API key error. Please check your configuration."
    except _anthropic.RateLimitError:
        logger.warning("Claude API rate limit hit")
        return "I'm temporarily rate limited. Please try again in a moment."
    except Exception as exc:
        logger.error("Claude streaming failed: %s", exc)
        return "Sorry, I had an internal error."
    finally:
        metrics.record_stage("llm", time.perf_counter() - started, success=success)


def ask_claude(
    system_prompt: str,
    user_text: str,
    *,
    max_tokens: int | None = None,
    model: str | None = None,
) -> str:
    """Non-streaming Claude call. Returns response text or an error string."""
    started = time.perf_counter()
    success = False
    effective_model = model or CLAUDE_DEFAULT_MODEL
    effective_max_tokens = max_tokens or CLAUDE_MAX_TOKENS_QUESTION

    try:
        client = _get_client()
        message = client.messages.create(
            model=effective_model,
            max_tokens=effective_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_text}],
        )
        text = "".join(
            block.text for block in message.content if hasattr(block, "text")
        ).strip()
        if text:
            success = True
            return text
        return "I could not get a response."
    except Exception as exc:
        logger.error("Claude non-streaming failed: %s", exc)
        return "Sorry, I had an internal error."
    finally:
        metrics.record_stage("llm", time.perf_counter() - started, success=success)
