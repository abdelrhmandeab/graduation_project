from core.config import KB_MAX_CONTEXT_CHARS, KB_TOP_K, MEMORY_MAX_CONTEXT_CHARS
from core.knowledge_base import knowledge_base_service
from core.persona import persona_manager
from core.session_memory import session_memory


def _normalize_response_language(language):
    value = str(language or "en").strip().lower()
    return "ar" if value == "ar" else "en"


# ---------------------------------------------------------------------------
# Few-shot examples that steer the model toward Egyptian Arabic and concise
# English answers.  Injected directly into the system prompt so small models
# (1.7B-8B) see concrete examples of the target style.
# ---------------------------------------------------------------------------
_FEW_SHOT_EXAMPLES = """\
USER: الجو عامل ازاي؟
ASSISTANT: مش معايا بيانات طقس دلوقتي، بس لو في القاهرة الأيام دي الجو حر — البس خفيف واشرب مية كتير.

USER: افتحلي كروم
ASSISTANT: تمام، بفتح جوجل كروم دلوقتي.

USER: what is machine learning?
ASSISTANT: Machine learning is when computers learn patterns from data instead of being explicitly programmed. Think of it like teaching by example rather than writing rules.

USER: احكيلي عن الأهرامات
ASSISTANT: الأهرامات اتبنت من حوالي 4500 سنة في الجيزة. أكبرهم هرم خوفو — ارتفاعه 146 متر وكان أطول مبنى في العالم لمدة 3800 سنة."""

_PROMPT_MEMORY_CONTEXT_MAX_CHARS = 200


def _build_system_block(response_language, persona_prompt, include_few_shot=True):
    """Build the slim system prompt.

    Target: ≤10 lines of system text before the few-shot examples. Small models
    (1.7B–4B) lose instruction-following past ~800 system tokens, so every line
    must earn its place. We deliberately drop the long persona prose and keep
    only its one-line essence inline with the core directives.
    """
    lang_label = "Arabic (Egyptian dialect)" if response_language == "ar" else "English"

    # Single, dense system rule line. Persona is implied (it's just "Jarvis").
    sections = [
        "SYSTEM:",
        f"You are Jarvis, a voice assistant. Reply in {lang_label} only. Be concise (1-3 sentences).",
        "Answer directly. If you lack live data, say so briefly then give practical advice.",
        "Never refuse safe questions.",
    ]

    if response_language == "ar":
        sections.append("Use Egyptian colloquial (تمام، دلوقتي، هعمل كده) — not formal MSA.")

    # Inject persona only if the user explicitly customized it (non-default).
    persona_text = " ".join(str(persona_prompt or "").split()).strip()
    default_persona_marker = "You are Jarvis, a helpful, friendly"
    if persona_text and not persona_text.startswith(default_persona_marker):
        sections.append(persona_text)

    if include_few_shot:
        sections.extend(["", "Examples:", _FEW_SHOT_EXAMPLES])

    return sections


def build_prompt_package(user_text, response_language="en"):
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

    sections = _build_system_block(response_language, persona_prompt)

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

    return {
        "prompt": "\n".join(sections),
        "kb_sources": kb_package["sources"],
        "kb_results": kb_package["results"],
        "kb_context_used": bool(compact_kb_context),
        "memory_used": bool(compact_memory_context),
    }


def build_prompt(user_text):
    return build_prompt_package(user_text)["prompt"]


def build_lightweight_prompt(user_text, response_language="en"):
    """Minimal prompt for short/simple queries — skips KB retrieval and session memory."""
    query = (user_text or "").strip()
    response_language = _normalize_response_language(response_language)
    persona_prompt = persona_manager.get_system_prompt()

    sections = _build_system_block(response_language, persona_prompt)
    sections.extend(["", "USER:", query, "", "ASSISTANT:"])

    return {
        "prompt": "\n".join(sections),
        "kb_sources": [],
        "kb_results": [],
        "kb_context_used": False,
        "memory_used": False,
    }


def build_tool_augmented_prompt(user_text, tool_context, response_language="en"):
    """Prompt with live data (weather, search results) injected before the user query."""
    query = (user_text or "").strip()
    response_language = _normalize_response_language(response_language)
    persona_prompt = persona_manager.get_system_prompt()

    sections = _build_system_block(response_language, persona_prompt)
    sections.extend([
        "",
        "LIVE DATA (use this to answer):",
        str(tool_context or ""),
        "",
        "USER:",
        query,
        "",
        "ASSISTANT:",
    ])

    return {
        "prompt": "\n".join(sections),
        "kb_sources": [],
        "kb_results": [],
        "kb_context_used": False,
        "memory_used": False,
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
