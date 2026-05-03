from core.config import KB_MAX_CONTEXT_CHARS, KB_TOP_K, MEMORY_MAX_CONTEXT_CHARS
from core.knowledge_base import knowledge_base_service
from core.logger import logger
from core.persona import persona_manager
from core.session_memory import session_memory


def _normalize_response_language(language):
    value = str(language or "en").strip().lower()
    return "ar" if value == "ar" else "en"


# ---------------------------------------------------------------------------
# Few-shot examples for small models (1.7B–4B): minimal set (2-3 examples)
# Keeps the prompt lean while steering toward Egyptian Arabic and concise English.
# ---------------------------------------------------------------------------
_FEW_SHOT_EXAMPLES_MINIMAL = """\
USER: افتحلي كروم
ASSISTANT: تمام، بفتح جوجل كروم دلوقتي.

USER: what is machine learning?
ASSISTANT: Machine learning teaches computers to learn from data rather than writing explicit rules."""

# ---------------------------------------------------------------------------
# Few-shot examples for large models (8B+): comprehensive set (4+ examples)
# Provides richer context for higher-capacity models.
# ---------------------------------------------------------------------------
_FEW_SHOT_EXAMPLES_FULL = """\
USER: الجو عامل ازاي؟
ASSISTANT: مش معايا بيانات طقس دلوقتي، بس لو في القاهرة الأيام دي الجو حر — البس خفيف واشرب مية كتير.

USER: افتحلي كروم
ASSISTANT: تمام، بفتح جوجل كروم دلوقتي.

USER: what is machine learning?
ASSISTANT: Machine learning is when computers learn patterns from data instead of being explicitly programmed. Think of it like teaching by example rather than writing rules.

USER: احكيلي عن الأهرامات
ASSISTANT: الأهرامات اتبنت من حوالي 4500 سنة في الجيزة. أكبرهم هرم خوفو — ارتفاعه 146 متر وكان أطول مبنى في العالم لمدة 3800 سنة."""

# Backward compat: full examples is the default
_FEW_SHOT_EXAMPLES = _FEW_SHOT_EXAMPLES_FULL

_PROMPT_MEMORY_CONTEXT_MAX_CHARS = 200


def _estimate_token_count(text):
    """Rough estimate of token count (1 token ≈ 4 chars for English, 1-2 for Arabic).
    
    Args:
        text: string to estimate token count for
        
    Returns:
        Approximate token count
    """
    # Rough heuristic: English ~4 chars per token, Arabic ~2 chars per token
    # Count as ~1 Arabic char for every alef-lam-baa type character
    arabic_char_count = sum(1 for c in text if ord(c) > 0x0600 and ord(c) < 0x06FF)
    english_char_count = len(text) - arabic_char_count
    estimated_tokens = (english_char_count / 4.0) + (arabic_char_count / 2.0)
    return int(estimated_tokens)


def _get_model_tier(model_name_or_tier):
    """Infer model tier from model name.
    
    Args:
        model_name_or_tier: e.g. "qwen3:8b" or "high", "qwen3:4b" or "medium"
        
    Returns:
        One of "minimal", "low", "medium", "high"
    """
    model_or_tier = str(model_name_or_tier or "").lower().strip()
    
    if model_or_tier in ("high", "qwen3:8b"):
        return "high"
    elif model_or_tier in ("medium", "qwen3:4b", "qwen3:7b"):
        return "medium"
    elif model_or_tier in ("low", "qwen3:1.7b"):
        return "low"
    elif model_or_tier in ("minimal", "qwen3:0.6b", "qwen3:1b"):
        return "minimal"
    else:
        # Default to medium
        return "medium"


def _build_system_block(response_language, persona_prompt, include_few_shot=True, tier="medium"):
    """Build the system prompt, optimized for the given tier.

    For minimal/low tiers (1.7B–4B): Keep to ~6–8 core lines + minimal few-shot.
    For medium/high tiers (4B+): Include richer context and full few-shot examples.

    Target: Reduce token count ≥40% vs baseline for small models.
    
    Args:
        response_language: "en" or "ar"
        persona_prompt: Optional custom persona text
        include_few_shot: Whether to include few-shot examples
        tier: One of "minimal", "low", "medium", "high"
        
    Returns:
        List of prompt sections
    """
    lang_label = "Arabic (Egyptian dialect)" if response_language == "ar" else "English"
    inferred_tier = _get_model_tier(tier)

    # Core directives (always included)
    sections = [
        "SYSTEM:",
        f"You are Jarvis, a voice assistant. Reply in {lang_label} only. Be concise (1-3 sentences).",
        "Answer directly. If you lack live data, say so briefly then give practical advice.",
    ]

    if response_language == "ar":
        sections.append("Use Egyptian colloquial (تمام، دلوقتي، هعمل كده) — not formal MSA.")

    # Inject persona only if the user explicitly customized it (non-default).
    persona_text = " ".join(str(persona_prompt or "").split()).strip()
    default_persona_marker = "You are Jarvis, a helpful, friendly"
    if persona_text and not persona_text.startswith(default_persona_marker):
        sections.append(persona_text)

    # Few-shot examples: minimal set for small models, full set for large models
    if include_few_shot:
        sections.append("")
        sections.append("Examples:")
        if inferred_tier in ("minimal", "low"):
            sections.append(_FEW_SHOT_EXAMPLES_MINIMAL)
        else:
            sections.append(_FEW_SHOT_EXAMPLES_FULL)

    return sections


def build_prompt_package(user_text, response_language="en", tier="medium"):
    query = (user_text or "").strip()
    response_language = _normalize_response_language(response_language)
    persona_prompt = persona_manager.get_system_prompt()
    memory_context = session_memory.build_context(
        max_chars=min(int(MEMORY_MAX_CONTEXT_CHARS), _PROMPT_MEMORY_CONTEXT_MAX_CHARS),
        language=response_language,
        intents={"LLM_QUERY"},
    )
    context_slots = session_memory.context_snapshot()
    kb_package = knowledge_base_service.retrieve_for_prompt(
        query,
        top_k=KB_TOP_K,
        max_chars=KB_MAX_CONTEXT_CHARS,
    )
    kb_context = kb_package["context"]
    compact_memory_context = " ".join(str(memory_context or "").split()).strip()
    compact_kb_context = " ".join(str(kb_context or "").split()).strip()

    sections = _build_system_block(response_language, persona_prompt, tier=tier)

    if compact_memory_context:
        sections.append(f"MEMORY: {compact_memory_context}")

    # Collapse session context into a single line — small models don't need
    # the multi-bullet structure and it just eats tokens.
    last_app = context_slots.get("last_app") or ""
    last_file = context_slots.get("last_file") or ""
    pending = context_slots.get("pending_confirmation_token") or ""
    context_parts = []
    if last_app:
        context_parts.append(f"last_app={last_app}")
    if last_file:
        context_parts.append(f"last_file={last_file}")
    if pending:
        context_parts.append(f"pending_confirmation={pending}")
    if context_parts:
        sections.append(f"CONTEXT: {', '.join(context_parts)}")

    if compact_kb_context:
        sections.append(f"KNOWLEDGE: {compact_kb_context}")

    sections.extend(["", "USER:", query, "", "ASSISTANT:"])

    prompt_text = "\n".join(sections)
    token_count = _estimate_token_count(prompt_text)
    
    return {
        "prompt": prompt_text,
        "kb_sources": kb_package["sources"],
        "kb_results": kb_package["results"],
        "kb_context_used": bool(compact_kb_context),
        "memory_used": bool(compact_memory_context),
        "token_count": token_count,
        "tier": tier,
    }


def build_prompt(user_text):
    return build_prompt_package(user_text)["prompt"]


def build_lightweight_prompt(user_text, response_language="en", tier="medium"):
    """Minimal prompt for short/simple queries — skips KB retrieval and session memory.
    
    Args:
        user_text: User query
        response_language: "en" or "ar"
        tier: Optional model tier for optimization (default: "medium")
        
    Returns:
        Dict with "prompt" and metadata
    """
    query = (user_text or "").strip()
    response_language = _normalize_response_language(response_language)
    persona_prompt = persona_manager.get_system_prompt()

    sections = _build_system_block(response_language, persona_prompt, tier=tier)
    sections.extend(["", "USER:", query, "", "ASSISTANT:"])

    prompt_text = "\n".join(sections)
    token_count = _estimate_token_count(prompt_text)

    return {
        "prompt": prompt_text,
        "kb_sources": [],
        "kb_results": [],
        "kb_context_used": False,
        "memory_used": False,
        "token_count": token_count,
        "tier": tier,
    }


def build_tool_augmented_prompt(user_text, tool_context, response_language="en", tier="medium"):
    """Prompt with live data (weather, search results) injected before the user query.
    
    Args:
        user_text: User query
        tool_context: Pre-formatted tool results (weather, web search, etc.)
        response_language: "en" or "ar"
        tier: Optional model tier for optimization (default: "medium")
        
    Returns:
        Dict with "prompt" and metadata
    """
    query = (user_text or "").strip()
    response_language = _normalize_response_language(response_language)
    persona_prompt = persona_manager.get_system_prompt()

    sections = _build_system_block(response_language, persona_prompt, tier=tier)
    # Per-tool framing lives inside ``tool_context`` (each block carries its own
    # [WEATHER]/[WEB_SEARCH] header). Here we add the global rule so the model
    # treats the block as authoritative for facts and never invents numbers.
    if response_language == "ar":
        live_data_rule = (
            "بيانات حية (استخدمها كمصدر حقائق ولا تخترع أرقام أو تفاصيل غير موجودة):"
        )
    else:
        live_data_rule = (
            "LIVE DATA (treat as authoritative — quote figures verbatim, "
            "do not invent missing details):"
        )
    sections.extend([
        "",
        live_data_rule,
        str(tool_context or "").strip(),
        "",
        "USER:",
        query,
        "",
        "ASSISTANT:",
    ])

    prompt_text = "\n".join(sections)
    token_count = _estimate_token_count(prompt_text)

    return {
        "prompt": prompt_text,
        "kb_sources": [],
        "kb_results": [],
        "kb_context_used": False,
        "memory_used": False,
        "token_count": token_count,
        "tier": tier,
    }


def build_intent_extraction_prompt(user_text, language="en"):
    query = (user_text or "").strip()
    lang = (language or "en").strip().lower() or "en"
    # Keep this prompt deterministic and schema-locked so routing can trust the output.
    return "\n".join(
        [
            "SYSTEM:",
            "You are a strict intent extraction engine for a local Windows assistant.",
            "Return one JSON object only. No markdown. No explanation.",
            "",
            "OUTPUT SCHEMA:",
            '{"intent":"...","action":"...","args":{},"confidence":0.0}',
            "",
            "ALLOWED INTENTS:",
            "- OS_APP_OPEN",
            "- OS_APP_CLOSE",
            "- OS_FILE_SEARCH",
            "- OS_FILE_NAVIGATION",
            "- OS_SYSTEM_COMMAND",
            "- JOB_QUEUE_COMMAND",
            "- VOICE_COMMAND",
            "- LLM_QUERY",
            "",
            "ACTION/ARGS RULES:",
            "- OS_APP_OPEN / OS_APP_CLOSE: args.app_name",
            "- OS_FILE_SEARCH: args.filename, optional args.search_path",
            (
                "- OS_FILE_NAVIGATION: action one of list_directory, cd, file_info, "
                "create_directory, delete_item, move_item, rename_item; provide required args"
            ),
            (
                "- OS_SYSTEM_COMMAND: args.action_key one of shutdown,restart,sleep,lock,logoff,"
                "volume_up,volume_down,volume_mute,volume_set,brightness_up,brightness_down,"
                "brightness_set,wifi_on,wifi_off,bluetooth_on,bluetooth_off,notifications_on,notifications_off,screenshot,"
                "empty_recycle_bin,list_processes,focus_window,window_maximize,window_minimize,"
                "window_snap_left,window_snap_right,window_next,window_close_active,"
                "media_play_pause,media_next_track,media_previous_track,media_stop,"
                "media_seek_forward,media_seek_backward,browser_new_tab,browser_close_tab,"
                "browser_back,browser_forward,browser_open_url,browser_search_web"
            ),
            "- For volume_set provide args.volume_level (0-100 integer)",
            "- For brightness_set provide args.brightness_level (0-100 integer)",
            "- For focus_window provide args.window_query (window title or app name)",
            "- For media_seek_forward/media_seek_backward provide args.seek_seconds (positive integer)",
            "- For browser_open_url provide args.url",
            "- For browser_search_web provide args.search_query",
            "- JOB_QUEUE_COMMAND: action one of enqueue,status,cancel,retry,list",
            "- For JOB_QUEUE_COMMAND enqueue provide args.command_text and optional args.delay_seconds",
            "- VOICE_COMMAND: action one of interrupt, speech_on, speech_off, status",
            "- If unclear, use intent=LLM_QUERY with confidence below 0.50",
            "",
            "EXAMPLES:",
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"volume_set","volume_level":40},"confidence":0.93}',
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"brightness_set","brightness_level":55},"confidence":0.92}',
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"focus_window","window_query":"chrome"},"confidence":0.9}',
            '{"intent":"OS_SYSTEM_COMMAND","action":"","args":{"action_key":"browser_open_url","url":"https://github.com"},"confidence":0.88}',
            '{"intent":"JOB_QUEUE_COMMAND","action":"enqueue","args":{"command_text":"mute volume","delay_seconds":300},"confidence":0.86}',
            "",
            "CONFIDENCE:",
            "- Number between 0 and 1",
            "- High confidence only when intent and critical entities are explicit",
            "",
            "HARD SAFETY:",
            "- Never invent file paths, app names, or action keys not implied by user text",
            "",
            f"LANGUAGE_HINT: {lang}",
            "",
            "USER:",
            query,
        ]
    )


# ---------------------------------------------------------------------------
# TIERED PROMPT BUILDERS – Phase 1 Production Improvement
# ---------------------------------------------------------------------------

def get_system_prompt_for_model(model_name, response_language="en"):
    """Get the optimal system prompt for a given model.
    
    Dispatches to minimal or full prompt based on model tier.
    For use in orchestrator and other entry points to select the right prompt style.
    
    Args:
        model_name: e.g. "qwen3:1.7b", "qwen3:4b", "qwen3:8b"
        response_language: "en" or "ar"
        
    Returns:
        System prompt text (without user query)
    """
    tier = _get_model_tier(model_name)
    persona_prompt = persona_manager.get_system_prompt()
    sections = _build_system_block(response_language, persona_prompt, include_few_shot=True, tier=tier)
    return "\n".join(sections)


def build_minimal_prompt(user_text, response_language="en"):
    """Build a minimal prompt for small models (1.7B–4B).
    
    Features:
    - 6–8 core system lines (no redundancy)
    - 2–3 few-shot examples (minimal set)
    - No KB or session memory (keep it lean)
    - Target: ≥40% token reduction vs full prompt
    
    Args:
        user_text: User query
        response_language: "en" or "ar"
        
    Returns:
        Dict with "prompt" and metadata
    """
    query = (user_text or "").strip()
    response_language = _normalize_response_language(response_language)
    persona_prompt = persona_manager.get_system_prompt()

    sections = _build_system_block(response_language, persona_prompt, include_few_shot=True, tier="low")
    sections.extend(["", "USER:", query, "", "ASSISTANT:"])

    prompt_text = "\n".join(sections)
    token_count = _estimate_token_count(prompt_text)
    
    logger.debug(
        "build_minimal_prompt: %d tokens, tier=low (for qwen3:0.6b-1.7b)",
        token_count,
    )

    return {
        "prompt": prompt_text,
        "kb_sources": [],
        "kb_results": [],
        "kb_context_used": False,
        "memory_used": False,
        "token_count": token_count,
        "tier": "low",
    }


def build_full_prompt(user_text, response_language="en"):
    """Build a full prompt for large models (8B+).
    
    Features:
    - Rich system prompt with full persona
    - Comprehensive few-shot examples (4+ examples)
    - Optional KB and session memory
    - Full context window utilization
    
    Args:
        user_text: User query
        response_language: "en" or "ar"
        
    Returns:
        Dict with "prompt" and metadata (same structure as build_prompt_package)
    """
    query = (user_text or "").strip()
    response_language = _normalize_response_language(response_language)
    persona_prompt = persona_manager.get_system_prompt()
    
    memory_context = session_memory.build_context(
        max_chars=min(int(MEMORY_MAX_CONTEXT_CHARS), _PROMPT_MEMORY_CONTEXT_MAX_CHARS),
        language=response_language,
        intents={"LLM_QUERY"},
    )
    context_slots = session_memory.context_snapshot()
    kb_package = knowledge_base_service.retrieve_for_prompt(
        query,
        top_k=KB_TOP_K,
        max_chars=KB_MAX_CONTEXT_CHARS,
    )
    kb_context = kb_package["context"]
    compact_memory_context = " ".join(str(memory_context or "").split()).strip()
    compact_kb_context = " ".join(str(kb_context or "").split()).strip()

    sections = _build_system_block(response_language, persona_prompt, include_few_shot=True, tier="high")

    if compact_memory_context:
        sections.append(f"MEMORY: {compact_memory_context}")

    last_app = context_slots.get("last_app") or ""
    last_file = context_slots.get("last_file") or ""
    pending = context_slots.get("pending_confirmation_token") or ""
    context_parts = []
    if last_app:
        context_parts.append(f"last_app={last_app}")
    if last_file:
        context_parts.append(f"last_file={last_file}")
    if pending:
        context_parts.append(f"pending_confirmation={pending}")
    if context_parts:
        sections.append(f"CONTEXT: {', '.join(context_parts)}")

    if compact_kb_context:
        sections.append(f"KNOWLEDGE: {compact_kb_context}")

    sections.extend(["", "USER:", query, "", "ASSISTANT:"])

    prompt_text = "\n".join(sections)
    token_count = _estimate_token_count(prompt_text)
    
    logger.debug(
        "build_full_prompt: %d tokens, tier=high (for qwen3:8b+), kb=%s, memory=%s",
        token_count,
        bool(compact_kb_context),
        bool(compact_memory_context),
    )

    return {
        "prompt": prompt_text,
        "kb_sources": kb_package["sources"],
        "kb_results": kb_package["results"],
        "kb_context_used": bool(compact_kb_context),
        "memory_used": bool(compact_memory_context),
        "token_count": token_count,
        "tier": "high",
    }


def build_prompt_for_tier(user_text, tier="medium", response_language="en"):
    """Build a prompt optimized for a specific model tier.
    
    Main dispatcher for tiered prompt construction. Use this when you know the
    target model tier and want the optimal prompt for it.
    
    Args:
        user_text: User query
        tier: One of "minimal", "low", "medium", "high"
        response_language: "en" or "ar"
        
    Returns:
        Dict with "prompt" and metadata
    """
    inferred_tier = _get_model_tier(tier)
    response_language = _normalize_response_language(response_language)
    
    if inferred_tier in ("minimal", "low"):
        return build_minimal_prompt(user_text, response_language)
    elif inferred_tier == "high":
        return build_full_prompt(user_text, response_language)
    else:
        # Default to build_prompt_package for medium tier
        return build_prompt_package(user_text, response_language, tier="medium")
