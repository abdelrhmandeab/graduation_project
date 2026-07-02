import pathlib

from core.config import (
    KB_MAX_CONTEXT_CHARS,
    KB_TOP_K,
    LLM_CTX_AUTOSIZE,
    LLM_FEWSHOT_MAX,
    LLM_FEWSHOT_MIN,
    LLM_OLLAMA_NUM_CTX,
    LLM_LANG_PIN_ENABLED,
    MEMORY_MAX_CONTEXT_CHARS,
)
from core.knowledge_base import knowledge_base_service
from core.logger import logger
from core.persona import format_persona_block, get_active_persona, persona_manager
from core.session_memory import session_memory

_PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"

_TIER_TO_TEMPLATE = {
    "minimal": "micro_prompt.txt",
    "low":     "micro_prompt.txt",
    "medium":  "slim_prompt.txt",
    "high":    "full_prompt.txt",
}

_first_build_logged = False


def _log_first_prompt_build(token_count, tier, response_language, builder_name):
    global _first_build_logged
    if _first_build_logged:
        return
    logger.info(
        "First prompt build: %d tokens, tier=%s, lang=%s, builder=%s",
        token_count,
        tier,
        response_language,
        builder_name,
    )
    _first_build_logged = True


def _normalize_response_language(language):
    value = str(language or "en").strip().lower()
    return "ar" if value == "ar" else "en"


def _language_pin_rule(response_language):
    if not LLM_LANG_PIN_ENABLED:
        return ""
    if _normalize_response_language(response_language) == "ar":
        return "جاوب بالعامية المصرية فقط. ممنوع تستخدم الفصحى أو أي لغة تانية."
    return "Reply ONLY in English. Never switch to Arabic or any other language."


def _filter_template_lines(lines):
    """Drop older soft language hints once the hard language pin is injected."""
    if not LLM_LANG_PIN_ENABLED:
        return list(lines)
    filtered = []
    for line in lines:
        stripped = str(line or "").strip()
        lowered = stripped.lower()
        if lowered.startswith("language:"):
            continue
        if lowered.startswith("name:"):
            continue
        if lowered.startswith("response style:"):
            continue
        filtered.append(line)
    return filtered


# ---------------------------------------------------------------------------
# Few-shot examples for small models (1.7B–4B): minimal set (2-3 examples)
# Keeps the prompt lean while steering toward Egyptian Arabic and concise English.
# ---------------------------------------------------------------------------
_FEW_SHOT_EXAMPLES_MINIMAL_EN = """\
USER: how can I become a better computer engineer?
ASSISTANT: Start with fundamentals: programming, operating systems, networking, and databases. Then build small real projects every month, read other people's code, and ask for feedback.

USER: what is machine learning?
ASSISTANT: Machine learning is teaching computers to spot patterns from examples, instead of hand-writing every rule."""

_FEW_SHOT_EXAMPLES_MINIMAL_AR = """\
USER: عايز أكون مهندس كمبيوتر شاطر، أعمل إيه؟
ASSISTANT: ركز على الأساسيات الأول — برمجة وداتا ستركتشر وشبكات. بعد كده ابني مشاريع صغيرة حقيقية كل شهر وخلي حد أحسن منك يراجع شغلك.

USER: مصر هتقابل مين في الدور الثاني من كأس العالم؟
ASSISTANT: معلوماتي محدودة بتاريخ معين ومش عندي نتايج لحظية دلوقتي. دور على Google أو beIN Sports عشان تعرف آخر الأخبار.

USER: اشرحلي يعني إيه تعلم آلي
ASSISTANT: ببساطة، الكمبيوتر بيتعلم من أمثلة كتير بدل ما تكتبله كل قاعدة بإيدك. زي فلتر السبام — بيتعلم من إيميلات قديمة إيه اللي سبام وإيه لأ."""

# ---------------------------------------------------------------------------
# Few-shot examples for large models (8B+): comprehensive set (4+ examples)
# Provides richer context for higher-capacity models.
# ---------------------------------------------------------------------------
_FEW_SHOT_EXAMPLES_FULL_EN = """\
USER: how can I become a successful computer engineer?
ASSISTANT: Build strong fundamentals first: coding, data structures, operating systems, networking, and databases. Then pick a track, ship projects, read good code, and practice explaining your decisions clearly.

USER: what's the weather in Cairo?
ASSISTANT: If live weather data is available, use it directly. If not, say you don't have live weather right now and give a useful general suggestion.

USER: explain machine learning simply
ASSISTANT: Machine learning is when software learns patterns from examples. A spam filter, for example, learns from old emails instead of relying only on hand-written rules."""

_FEW_SHOT_EXAMPLES_FULL_AR = """\
USER: عايز أكون مهندس كمبيوتر ناجح، أعمل إيه؟
ASSISTANT: ابدأ بالأساسيات — برمجة وداتا ستركتشر وشبكات. بعد كده اختار مسار يعجبك وابني فيه مشاريع حقيقية، واقرأ كود ناس شاطرة عشان تتعلم منهم.

USER: الجو عامل إزاي في القاهرة؟
ASSISTANT: لو معايا بيانات طقس هقولهالك مباشرة. لو مش معايا دلوقتي، البس خفيف واشرب مية كتير لأن الجو حر الأيام دي.

USER: مصر هتقابل مين في الدور الثاني من كأس العالم؟
ASSISTANT: مش عندي بيانات نتايج لحظية — معلوماتي محدودة بتاريخ معين. دور على beIN Sports أو Google عشان تعرف آخر أخبار البطولة.

USER: إيه التخصصات الموجودة في هندسة الكمبيوتر؟
ASSISTANT: التخصصات الشائعة في هندسة الكمبيوتر: برمجيات (software engineering)، شبكات ونظم موزعة، ذكاء اصطناعي وتعلم آلي، أمن معلومات، نظم مدمجة (embedded systems)، وعلم البيانات. اختار اللي بيشدك حسب اهتمامك.

USER: اشرحلي التعلم الآلي ببساطة
ASSISTANT: ببساطة، البرنامج بيتعلم من أمثلة كتير بدل ما تكتبله كل قاعدة بإيدك. زي فلتر السبام — بيشوف إيميلات قديمة ويتعلم يفرق بين العادي والسبام."""

_PROMPT_MEMORY_CONTEXT_MAX_CHARS = 600


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


def _tier_ctx_ceiling(tier="medium"):
    """Return the runtime context ceiling for this turn.

    The configured/runtime Ollama context values are ceilings now; prompt size
    decides the actual per-call num_ctx.
    """
    try:
        from llm.ollama_client import get_runtime_lightweight_num_ctx, get_runtime_num_ctx

        inferred = _get_model_tier(tier)
        if inferred in ("minimal", "low"):
            return int(get_runtime_lightweight_num_ctx(default=LLM_OLLAMA_NUM_CTX))
        return int(get_runtime_num_ctx(default=LLM_OLLAMA_NUM_CTX))
    except Exception:
        return int(LLM_OLLAMA_NUM_CTX)


def pick_num_ctx(prompt_tokens, tier="medium"):
    """Autosize Ollama num_ctx from prompt token count, capped by model tier."""
    ceiling = max(512, int(_tier_ctx_ceiling(tier)))
    if not LLM_CTX_AUTOSIZE:
        return ceiling

    tokens = max(0, int(prompt_tokens or 0))
    if tokens <= 256:
        selected = 512
    elif tokens <= 512:
        selected = 1024
    elif tokens <= 1024:
        selected = 2048
    elif tokens <= 2048:
        selected = 4096
    else:
        selected = ceiling
    return max(512, min(int(selected), ceiling))


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


def _load_prompt_template(tier: str):
    """Read a .txt template for the given tier; return None on failure."""
    filename = _TIER_TO_TEMPLATE.get(str(tier).lower(), "slim_prompt.txt")
    try:
        return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")
    except Exception:
        return None


def _fewshot_limit_for_tier(tier: str) -> int:
    inferred = _get_model_tier(tier)
    tier_default = 2 if inferred in ("minimal", "low") else 3 if inferred == "medium" else 4
    lower = max(0, int(LLM_FEWSHOT_MIN or 0))
    upper = max(lower, int(LLM_FEWSHOT_MAX or tier_default))
    return min(max(tier_default, lower), upper)


def _split_fewshot_examples(examples: str) -> list[str]:
    blocks = []
    current = []
    for line in str(examples or "").splitlines():
        if line.strip().upper().startswith("USER:") and current:
            blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        block = "\n".join(current).strip()
        if block:
            blocks.append(block)
    return [block for block in blocks if block]


def _fewshot_examples_for_tier(tier: str, response_language: str = "en") -> str:
    lang = _normalize_response_language(response_language)
    if _get_model_tier(tier) in ("minimal", "low"):
        source = _FEW_SHOT_EXAMPLES_MINIMAL_AR if lang == "ar" else _FEW_SHOT_EXAMPLES_MINIMAL_EN
    else:
        source = _FEW_SHOT_EXAMPLES_FULL_AR if lang == "ar" else _FEW_SHOT_EXAMPLES_FULL_EN
    return "\n\n".join(_split_fewshot_examples(source)[: _fewshot_limit_for_tier(tier)])


def _cap_template_examples(rendered: str, tier: str) -> str:
    """Strictly cap template Example blocks without touching the core prompt."""
    lines = str(rendered or "").splitlines()
    capped = []
    seen = 0
    limit = _fewshot_limit_for_tier(tier)
    skipping = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("example "):
            seen += 1
            skipping = seen > limit
        elif stripped.upper().startswith("USER:") and not any(
            ln.strip().lower().startswith("example ") for ln in lines
        ):
            seen += 1
            skipping = seen > limit
        if not skipping:
            capped.append(line)
    return "\n".join(capped).strip()


def get_prompt_tier() -> str:
    """Return the current runtime model tier for prompt selection."""
    from llm.ollama_client import get_runtime_model_tier
    return get_runtime_model_tier(default="medium")


def _answer_quality_contract(response_language: str) -> list[str]:
    if _normalize_response_language(response_language) == "ar":
        return [
            "مهمتك: افهم طلب المستخدم وجاوب عليه هو بالظبط.",
            "اكتب بالعربية بالحروف العربية دايماً. ممنوع تكتب عربي بحروف إنجليزي (romanized) زي 'marhaba' أو 'ahlan'.",
            "جاوب بالعامية المصرية زي ما الناس بتتكلم في الشارع. ممنوع فصحى أو أسلوب رسمي.",
            "لو السؤال عن حاجة بتتغير زي نتايج ماتشات أو أخبار أو أسعار، قول بوضوح 'أنا مش متأكد' أو 'معلوماتي لحد [سنة]، ممكن اتغير' — ممنوع تخترع معلومات.",
            "لو السؤال محتاج بيانات لحظية (طقس، أخبار، نتايج)، استخدم البيانات اللي موجودة في الـ CONTEXT أو قول إنك مش عندك بيانات دلوقتي.",
            "لو في كلمة غلط من الصوت، فهمها من السياق.",
            "ممنوع تكرر السؤال أو تقول كلام عام فاضي. خليك مباشر ومفيد.",
        ]
    return [
        "Task: answer the user's exact request, not a nearby topic.",
        "Always write in the same script as the question. Never romanize Arabic (e.g. never write 'marhaba' — write 'مرحبا').",
        "For advice or explanation questions, give practical concrete steps; do not refuse harmless requests.",
        "If the question is about live or changing data (scores, news, prices), clearly say 'I'm not sure' or 'my data goes up to [year], this may have changed' — never fabricate facts.",
        "If speech-to-text looks slightly wrong, infer the likely meaning from context and ask one clarification only if the meaning is impossible.",
        "Do not repeat the user's question. Do not give generic filler like 'share your goal' when the goal is already present.",
        "Sound like a helpful human: direct, specific, natural, no fluff.",
    ]


def _build_system_block(response_language, include_few_shot=True, tier="medium"):
    """Build the system prompt block for the given tier.

    Primary path: loads a .txt template from llm/prompts/ and substitutes
    {lang} and {ar_rule} placeholders.
    Fallback: inline prompt (used if template files are missing).
    """
    inferred_tier = _get_model_tier(tier)
    language_pin = _language_pin_rule(response_language)
    persona_block = format_persona_block(get_active_persona(), response_language)
    ar_rule = (
        "Use Egyptian colloquial only (تمام، دلوقتي، هعمل كده) — not formal MSA."
        if response_language == "ar"
        else ""
    )

    # --- Inline fallback ---
    sections = [
        "SYSTEM:",
    ]
    if language_pin:
        sections.append(language_pin)
    if persona_block:
        sections.append(persona_block)
    sections.extend(_answer_quality_contract(response_language))
    if ar_rule:
        sections.append(ar_rule)
    if include_few_shot:
        sections.append("")
        sections.append("Examples:")
        sections.append(_fewshot_examples_for_tier(inferred_tier, response_language))
    return sections


def _package_num_ctx(package: dict, query: str = "") -> dict:
    token_count = int(package.get("token_count") or 0)
    tier = str(package.get("tier") or "medium")
    num_ctx = pick_num_ctx(token_count, tier=tier)
    query_words = len(str(query or "").split())
    logger.info("num_ctx auto: query_words=%d tokens=%d tier=%s ctx=%d", query_words, token_count, tier, num_ctx)
    return {**package, "num_ctx": num_ctx}


def build_prompt_package(user_text, response_language="en", tier="medium"):
    query = (user_text or "").strip()
    response_language = _normalize_response_language(response_language)
    # Auto-select tier from runtime when caller uses the default.
    effective_tier = get_prompt_tier() if tier == "medium" else tier
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

    sections = _build_system_block(response_language, tier=effective_tier)

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
    _log_first_prompt_build(token_count, effective_tier, response_language, "build_prompt_package")

    return _package_num_ctx({
        "prompt": prompt_text,
        "kb_sources": kb_package["sources"],
        "kb_results": kb_package["results"],
        "kb_context_used": bool(compact_kb_context),
        "memory_used": bool(compact_memory_context),
        "token_count": token_count,
        "tier": effective_tier,
    }, query)


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
    sections = _build_system_block(response_language, tier=tier)
    sections.extend(["", "USER:", query, "", "ASSISTANT:"])

    prompt_text = "\n".join(sections)
    token_count = _estimate_token_count(prompt_text)
    _log_first_prompt_build(token_count, tier, response_language, "build_lightweight_prompt")

    return _package_num_ctx({
        "prompt": prompt_text,
        "kb_sources": [],
        "kb_results": [],
        "kb_context_used": False,
        "memory_used": False,
        "token_count": token_count,
        "tier": tier,
    }, query)


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
    # Live-data answers need facts quickly; skip few-shot examples and keep the
    # system block lean so weather/news queries spend fewer tokens on prompt setup.
    sections = _build_system_block(response_language, include_few_shot=False, tier=tier)
    live_data_rule = (
        "نتايج بحث ممكن تفيدك. لو بتجاوب على سؤال حقيقي/معلومة محتاجة تحديث "
        "(زي نتيجة ماتش، سعر، خبر، تاريخ، شخص)، استخدم النتايج دي كحقائق. "
        "لو مش مرتبطة بالسؤال (زي نصيحة أو رأي أو كلام عادي)، تجاهلها تماماً "
        "وجاوب من فهمك العادي من غير ما تذكرها:"
        if response_language == "ar"
        else "Search results that may help. If the user is asking a factual or "
        "time-sensitive question (scores, prices, news, history, people), use "
        "these results as facts. If they are not relevant to the question "
        "(e.g. advice, opinions, casual talk), ignore them completely and "
        "answer from your own understanding without mentioning them:"
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
    _log_first_prompt_build(token_count, tier, response_language, "build_tool_augmented_prompt")

    return _package_num_ctx({
        "prompt": prompt_text,
        "kb_sources": [],
        "kb_results": [],
        "kb_context_used": False,
        "memory_used": False,
        "token_count": token_count,
        "tier": tier,
    }, query)


def build_claude_messages(
    user_text,
    response_language="en",
    *,
    use_memory: bool = True,
    use_kb: bool = True,
    tier: str = "medium",
) -> dict:
    """Return {"system": str, "user": str, "token_count": int} for Claude API calls.

    Builds the same context as build_prompt_package() but separated into a
    Claude-style system message and a user message instead of a single string.
    """
    query = (user_text or "").strip()
    response_language = _normalize_response_language(response_language)
    effective_tier = get_prompt_tier() if tier == "medium" else tier

    # System block (strips the "SYSTEM:" header line for Claude's system param)
    system_sections = _build_system_block(response_language, tier=effective_tier)
    # Drop leading "SYSTEM:" label — it's a Ollama-ism, not needed for Claude
    if system_sections and system_sections[0].strip().upper() == "SYSTEM:":
        system_sections = system_sections[1:]

    if use_memory:
        memory_context = session_memory.build_context(
            max_chars=min(int(MEMORY_MAX_CONTEXT_CHARS), _PROMPT_MEMORY_CONTEXT_MAX_CHARS),
            language=response_language,
            intents={"LLM_QUERY"},
        )
        compact_memory = " ".join(str(memory_context or "").split()).strip()
        if compact_memory:
            system_sections.append(f"Memory from previous turns: {compact_memory}")

        # Semantic recall: inject top-3 relevant past exchanges from vector store.
        try:
            semantic_hits = session_memory.recall_semantic(query, n=3, language=response_language)
            if semantic_hits:
                recall_lines = []
                for hit in semantic_hits:
                    u = (hit.get("user") or "").strip()
                    a = (hit.get("assistant") or "").strip()
                    if u and a:
                        recall_lines.append(f"Q: {u}\nA: {a}")
                if recall_lines:
                    system_sections.append(
                        "Relevant past knowledge (from earlier sessions):\n" + "\n---\n".join(recall_lines)
                    )
        except Exception:
            pass

    context_slots = session_memory.context_snapshot()
    last_app = context_slots.get("last_app") or ""
    last_file = context_slots.get("last_file") or ""
    context_parts = []
    if last_app:
        context_parts.append(f"last_app={last_app}")
    if last_file:
        context_parts.append(f"last_file={last_file}")
    if context_parts:
        system_sections.append(f"Session context: {', '.join(context_parts)}")

    kb_sources: list = []
    if use_kb:
        kb_package = knowledge_base_service.retrieve_for_prompt(
            query, top_k=KB_TOP_K, max_chars=KB_MAX_CONTEXT_CHARS
        )
        compact_kb = " ".join(str(kb_package.get("context") or "").split()).strip()
        if compact_kb:
            system_sections.append(f"Reference knowledge: {compact_kb}")
        kb_sources = kb_package.get("sources", [])

    # Length / dialect constraints as a final system instruction
    if response_language == "ar":
        system_sections.append(
            "Keep answers concise (1–3 sentences for commands, up to 4 for questions). "
            "Reply in Egyptian colloquial Arabic (عامية مصرية) only — not formal MSA."
        )
    else:
        system_sections.append(
            "Keep answers concise: 1–2 sentences for commands, up to 3 for questions. "
            "No markdown, no bullet lists — speak naturally."
        )

    system_text = "\n".join(ln for ln in system_sections if ln is not None)
    token_count = _estimate_token_count(system_text + query)

    # Fetch recent conversation turns to pass as prior messages (not as system text).
    # This gives Claude proper multi-turn context instead of injected text blocks.
    prior_messages = session_memory.get_messages_for_claude(
        limit=5, language=response_language
    ) if use_memory else []

    return {
        "system": system_text,
        "user": query,
        "prior_messages": prior_messages,
        "kb_sources": kb_sources,
        "token_count": token_count,
        "tier": effective_tier,
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

