import os
import re
import time
from collections import OrderedDict

from core.command_classifier import classify_with_nlu
from core.command_parser import ParsedCommand, parse_command
from core.config import (
    CLARIFICATION_CORRECTION_WINDOW_SECONDS,
    CODE_SWITCH_CONTINUITY_ENABLED,
    CODE_SWITCH_CONTINUITY_WINDOW,
    CODE_SWITCH_DOMINANT_RATIO,
    CLARIFICATION_FALLBACK_AFTER_MISSES,
    CLARIFICATION_PREFERENCE_MAX_AGE_SECONDS,
    FOLLOWUP_DESTRUCTIVE_REFERENCE_MIN_CONFIDENCE,
    FOLLOWUP_DESTRUCTIVE_REQUIRE_EXPLICIT_REFERENCE,
    FOLLOWUP_APP_REFERENCE_HALF_LIFE_SECONDS,
    FOLLOWUP_APP_REFERENCE_MAX_AGE_SECONDS,
    FOLLOWUP_FILE_REFERENCE_HALF_LIFE_SECONDS,
    FOLLOWUP_FILE_REFERENCE_MAX_AGE_SECONDS,
    FOLLOWUP_PENDING_CONFIRMATION_HALF_LIFE_SECONDS,
    FOLLOWUP_PENDING_CONFIRMATION_MAX_AGE_SECONDS,
    FOLLOWUP_REFERENCE_CONFLICT_WINDOW_SECONDS,
    FOLLOWUP_REFERENCE_MAX_AGE_SECONDS,
    FOLLOWUP_REFERENCE_MIN_CONFIDENCE,
    LLM_APPEND_SOURCE_CITATIONS,
    LLM_LIGHTWEIGHT_NUM_CTX,
    LLM_RESPONSE_CACHE_ENABLED,
    LLM_RESPONSE_CACHE_MAX_QUERY_WORDS,
    LLM_RESPONSE_CACHE_MAX_SIZE,
    LLM_RESPONSE_CACHE_TTL_SECONDS,
    LLM_REALTIME_REWRITE_ENABLED,
    NLU_LLM_QUERY_EXTRACTION_ENABLED,
    NLU_PARSER_FASTPATH_CONFIDENCE_FLOOR,
    NLU_PARSER_FASTPATH_ENABLED,
    NLU_INTENT_CONFIDENCE_THRESHOLD,
    NLU_INTENT_ROUTING_ENABLED,
    NLU_INTENT_THRESHOLD_BY_FAMILY,
    PERSONA_LENGTH_TARGET_ENABLED,
    PERSONA_RESPONSE_MAX_WORDS,
    RESPONSE_MODE_FEATURE_ENABLED,
    TONE_SENSITIVE_NEUTRAL_ENABLED,
    TONE_ADAPTATION_ENABLED,
)
from core.demo_mode import is_enabled as is_demo_mode_enabled
from core.demo_mode import set_enabled as set_demo_mode
from core.handlers import audit, batch, file_navigation
from core.handlers import job_queue as job_queue_handler
from core.handlers import knowledge_base, memory, persona, policy, search_index, voice
from core.intent_confidence import (
    assess_intent_confidence,
    build_clarification_payload,
    resolve_clarification_reply,
)
from core.language_gate import UNSUPPORTED_LANGUAGE_MESSAGE, detect_supported_language
from core.logger import logger, log_structured
from core.metrics import metrics
from core.persona import persona_manager
from core.response_templates import anti_repetition_prefixes, detect_language_hint, normalize_language, render_template
from core.session_memory import session_memory
from llm.ollama_client import ask_llm, ask_llm_streaming
from llm.prompt_builder import build_prompt_package, build_lightweight_prompt
try:
    from nlp.intent_classifier import classify_intent as _classify_keyword_intent
except Exception:
    _classify_keyword_intent = None
from os_control.action_log import log_action, read_recent_actions
from os_control.adapter_result import to_router_tuple
from os_control.app_ops import execute_confirmed_app_operation, open_app_result, request_close_app_result, resolve_app_request
from os_control.confirmation import confirmation_manager
from os_control.file_ops import (
    execute_confirmed_file_operation,
    find_files,
    get_current_directory,
    undo_last_action,
)
from os_control.job_queue import job_queue_service
from os_control.policy import policy_engine
from os_control.search_index import search_index_service
from os_control.system_ops import execute_system_command_result, request_system_command_result


_JOB_QUEUE_EXECUTOR_READY = False
_LLM_RESPONSE_CACHE = OrderedDict()
_LLM_RESPONSE_CACHE_STATS = {
    "hits": 0,
    "misses": 0,
    "stores": 0,
    "evictions": 0,
}


def _nlu_threshold_for_intent(intent: str):
    intent_key = str(intent or "").strip().upper()
    family_threshold = NLU_INTENT_THRESHOLD_BY_FAMILY.get(intent_key)
    if family_threshold is None:
        return float(NLU_INTENT_CONFIDENCE_THRESHOLD)
    return float(family_threshold)


_PARSER_FASTPATH_INTENTS = {
    "OS_APP_OPEN",
    "OS_APP_CLOSE",
    "OS_FILE_SEARCH",
    "OS_FILE_NAVIGATION",
    "OS_SYSTEM_COMMAND",
    "JOB_QUEUE_COMMAND",
    "VOICE_COMMAND",
    "MEMORY_COMMAND",
    "PERSONA_COMMAND",
    "POLICY_COMMAND",
    "SEARCH_INDEX_COMMAND",
    "AUDIT_VERIFY",
    "AUDIT_RESEAL",
    "AUDIT_LOG_REPORT",
    "OBSERVABILITY_REPORT",
    "METRICS_REPORT",
    "DEMO_MODE",
    "OS_CONFIRMATION",
    "OS_ROLLBACK",
}


def _select_parser_fastpath_assessment(source_text, parser_candidate, language):
    if not NLU_PARSER_FASTPATH_ENABLED:
        return None

    intent = str(getattr(parser_candidate, "intent", "") or "").strip().upper()
    if not intent or intent == "LLM_QUERY":
        return None
    if intent not in _PARSER_FASTPATH_INTENTS:
        return None

    assessment = assess_intent_confidence(source_text, parser_candidate, language=language)
    if assessment.should_clarify:
        return assessment

    threshold = _nlu_threshold_for_intent(intent)
    confidence = float(assessment.confidence or 0.0)
    confidence_floor = max(0.0, min(1.0, float(NLU_PARSER_FASTPATH_CONFIDENCE_FLOOR or 0.55)))
    fastpath_gate = min(float(threshold), confidence_floor)
    if confidence >= fastpath_gate:
        return assessment
    return None


def _should_skip_nlu_llm_query(parser_candidate):
    intent = str(getattr(parser_candidate, "intent", "") or "").strip().upper()
    return intent == "LLM_QUERY" and not NLU_LLM_QUERY_EXTRACTION_ENABLED


_KEYWORD_NLP_MIN_CONFIDENCE = 0.45
_KEYWORD_NLP_SEARCH_PREFIX_RE = re.compile(
    r"^(?:search|find|look\s+up|google|دور|دوّر|دورلي|دوّرلي|ابحث)(?:\s+(?:for|about|on|in|عن|على|في))?\s+",
    re.IGNORECASE,
)
_KEYWORD_NLP_SEARCH_WEB_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:web|internet|online|الويب|النت)\s*(?:for|about|عن)?\s+",
    re.IGNORECASE,
)
_KEYWORD_NLP_URL_INTENT_MAP = {
    "open_youtube": "https://www.youtube.com",
    "open_google": "https://www.google.com",
}
_KEYWORD_NLP_APP_OPEN_INTENT_MAP = {
    "play_music": "spotify",
    "open_spotify": "spotify",
    "open_chrome": "chrome",
    "open_calculator": "calculator",
}
_KEYWORD_NLP_SYSTEM_ACTION_INTENT_MAP = {
    "volume_up": "volume_up",
    "volume_down": "volume_down",
    "wifi_on": "wifi_on",
    "wifi_off": "wifi_off",
    "bluetooth_on": "bluetooth_on",
    "bluetooth_off": "bluetooth_off",
    "screenshot": "screenshot",
}
_KEYWORD_NLP_SCREENSHOT_EXPLICIT_MARKERS = {
    "screenshot",
    "screen shot",
    "screen",
    "سكرين",
    "سكرينشوت",
    "سكرين شوت",
    "شاشه",
    "شاشة",
    "الشاشه",
    "الشاشة",
    "لقطه شاشه",
    "لقطة شاشة",
    "صوره شاشة",
    "صورة شاشة",
    "صوره للشاشه",
    "صورة للشاشة",
}
_KEYWORD_NLP_INFORMATIONAL_QUERY_MARKERS = {
    "weather",
    "forecast",
    "temperature",
    "news",
    "headline",
    "price",
    "prices",
    "gold price",
    "gold prices",
    "سعر",
    "اسعار",
    "أسعار",
    "ذهب",
    "دهب",
    "النهارده",
    "النهاردة",
    "today",
    "في مصر",
    "in egypt",
    "اخبار",
    "أخبار",
    "الجو",
    "طقس",
}


def _extract_keyword_nlp_search_query(source_text):
    value = " ".join(str(source_text or "").split()).strip()
    if not value:
        return ""

    value = _KEYWORD_NLP_SEARCH_PREFIX_RE.sub("", value, count=1).strip()
    value = _KEYWORD_NLP_SEARCH_WEB_PREFIX_RE.sub("", value, count=1).strip()
    value = value.strip(" .,!?؟،")
    return value


def _has_keyword_nlp_screenshot_marker(source_text, matched_keywords):
    normalized_text = " ".join(str(source_text or "").lower().split()).strip()
    if not normalized_text:
        return False

    if any(marker in normalized_text for marker in _KEYWORD_NLP_SCREENSHOT_EXPLICIT_MARKERS):
        return True

    for keyword in matched_keywords:
        normalized_keyword = " ".join(str(keyword or "").lower().split()).strip()
        if not normalized_keyword:
            continue
        if any(marker in normalized_keyword for marker in _KEYWORD_NLP_SCREENSHOT_EXPLICIT_MARKERS):
            return True
    return False


def _looks_keyword_nlp_informational_query(source_text):
    normalized_text = " ".join(str(source_text or "").lower().split()).strip()
    if not normalized_text:
        return False

    if any(marker in normalized_text for marker in _WEATHER_QUERY_MARKERS):
        return True
    if any(marker in normalized_text for marker in _NEWS_QUERY_MARKERS):
        return True
    return any(marker in normalized_text for marker in _KEYWORD_NLP_INFORMATIONAL_QUERY_MARKERS)


def _map_keyword_nlp_intent_to_command(source_text, nlp_result):
    intent_name = str((nlp_result or {}).get("intent") or "").strip().lower()
    if intent_name in {"", "unknown"}:
        return None

    try:
        confidence = float((nlp_result or {}).get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if confidence < _KEYWORD_NLP_MIN_CONFIDENCE:
        return None

    matched_keywords = list((nlp_result or {}).get("matched_keywords") or [])

    if intent_name == "screenshot" and not _has_keyword_nlp_screenshot_marker(source_text, matched_keywords):
        return None

    normalized = " ".join(str(source_text or "").lower().split()).strip()
    target_url = _KEYWORD_NLP_URL_INTENT_MAP.get(intent_name)
    if target_url:
        return ParsedCommand(
            intent="OS_SYSTEM_COMMAND",
            raw=source_text,
            normalized=normalized,
            args={"action_key": "browser_open_url", "url": target_url},
        )

    app_name = _KEYWORD_NLP_APP_OPEN_INTENT_MAP.get(intent_name)
    if app_name:
        return ParsedCommand(
            intent="OS_APP_OPEN",
            raw=source_text,
            normalized=normalized,
            args={"app_name": app_name},
        )

    action_key = _KEYWORD_NLP_SYSTEM_ACTION_INTENT_MAP.get(intent_name)
    if action_key:
        if _looks_keyword_nlp_informational_query(source_text):
            return None
        return ParsedCommand(
            intent="OS_SYSTEM_COMMAND",
            raw=source_text,
            normalized=normalized,
            args={"action_key": action_key},
        )

    if intent_name == "search":
        query = _extract_keyword_nlp_search_query(source_text)
        if not query:
            return None
        return ParsedCommand(
            intent="OS_SYSTEM_COMMAND",
            raw=source_text,
            normalized=normalized,
            args={"action_key": "browser_search_web", "search_query": query},
        )

    return None


def _try_keyword_nlp_routing(source_text, parser_candidate):
    meta = {
        "nlp_used": False,
        "nlp_accepted": False,
        "nlp_intent": "",
        "nlp_confidence": 0.0,
        "nlp_matched_keywords": [],
    }

    intent = str(getattr(parser_candidate, "intent", "") or "").strip().upper()
    if intent != "LLM_QUERY" or _classify_keyword_intent is None:
        return None, meta

    meta["nlp_used"] = True
    try:
        nlp_result = dict(_classify_keyword_intent(source_text) or {})
    except Exception as exc:
        logger.warning("Keyword NLP routing failed: %s", exc)
        return None, meta

    meta["nlp_intent"] = str(nlp_result.get("intent") or "").strip().lower()
    try:
        meta["nlp_confidence"] = float(nlp_result.get("confidence") or 0.0)
    except (TypeError, ValueError):
        meta["nlp_confidence"] = 0.0
    meta["nlp_matched_keywords"] = list(nlp_result.get("matched_keywords") or [])

    mapped = _map_keyword_nlp_intent_to_command(source_text, nlp_result)
    if mapped is not None:
        meta["nlp_accepted"] = True
    return mapped, meta


_CLARIFICATION_PREVENTED_REASONS = {
    "app_name_ambiguous",
    "app_close_ambiguous",
    "file_search_multiple_matches",
    "open_target_ambiguous",
    "low_entity_confidence",
    "multiple_actions_detected",
}

_POST_CLARIFICATION_CORRECTION_MARKERS_EN = {
    "wrong",
    "not that",
    "no that one",
    "different one",
    "other one",
    "not this",
    "incorrect",
}

_POST_CLARIFICATION_CORRECTION_MARKERS_AR = {
    "غلط",
    "لا هذا",
    "لا ده",
    "مش هذا",
    "مش ده",
    "غيره",
    "غير هذا",
    "خطا",
    "خطأ",
}

_SENSITIVE_SYSTEM_ACTION_KEYS = {
    "shutdown",
    "restart",
    "logoff",
    "sleep",
    "lock",
    "empty_recycle_bin",
}


def _clarification_intent_from_payload(payload):
    options = list((payload or {}).get("options") or [])
    if options:
        intent = str((options[0] or {}).get("intent") or "").strip().upper()
        if intent:
            return intent
    return "INTENT_CLARIFICATION"


def _is_wrong_action_prevented_reason(reason):
    return str(reason or "").strip().lower() in _CLARIFICATION_PREVENTED_REASONS


def _is_sensitive_command(parsed):
    intent = str(getattr(parsed, "intent", "") or "").strip().upper()
    action = str(getattr(parsed, "action", "") or "").strip().lower()
    args = dict(getattr(parsed, "args", {}) or {})

    if intent == "OS_CONFIRMATION":
        return True

    if intent == "OS_APP_CLOSE":
        return True

    if intent == "OS_FILE_NAVIGATION" and action in {
        "delete_item",
        "delete_item_permanent",
        "move_item",
        "rename_item",
    }:
        return True

    if intent == "OS_SYSTEM_COMMAND":
        action_key = str(args.get("action_key") or "").strip().lower()
        if action_key in _SENSITIVE_SYSTEM_ACTION_KEYS:
            return True

    return False


def _looks_like_post_clarification_correction(text, language="en"):
    normalized = _normalize_compact(text)
    if not normalized:
        return False

    markers = (
        _POST_CLARIFICATION_CORRECTION_MARKERS_AR
        if str(language or "").strip().lower() == "ar"
        else _POST_CLARIFICATION_CORRECTION_MARKERS_EN
    )
    if any(marker in normalized for marker in markers):
        return True

    # Keep a tiny language-agnostic fallback for mixed phrases.
    if "wrong" in normalized or "غلط" in normalized:
        return True
    return False


def _find_preferred_clarification_option(payload):
    payload_dict = dict(payload or {})
    reason = str(payload_dict.get("reason") or "").strip()
    source_text = str(payload_dict.get("source_text") or "").strip()
    language = str(payload_dict.get("language") or "en").strip().lower()
    options = list(payload_dict.get("options") or [])
    if not reason or not source_text or not options:
        return None

    preference = session_memory.get_clarification_choice(
        reason,
        source_text,
        language=language,
        max_age_seconds=CLARIFICATION_PREFERENCE_MAX_AGE_SECONDS,
    )
    if not preference:
        return None

    pref_id = str(preference.get("id") or "").strip()
    pref_intent = str(preference.get("intent") or "").strip()
    pref_action = str(preference.get("action") or "").strip()
    pref_args = dict(preference.get("args") or {})

    for option in options:
        option_args = dict(option.get("args") or {})
        if pref_id and str(option.get("id") or "").strip() == pref_id:
            return option
        if pref_intent and str(option.get("intent") or "").strip() != pref_intent:
            continue
        if pref_action and str(option.get("action") or "").strip() != pref_action:
            continue
        if pref_args:
            matches = True
            for key, value in pref_args.items():
                if str(option_args.get(key) or "").strip().lower() != str(value or "").strip().lower():
                    matches = False
                    break
            if matches:
                return option

    return None

_OPEN_FOLLOWUP_TEXTS = {
    "open it",
    "open this",
    "open that",
    "launch it",
    "start it",
    "افتحه",
    "افتحها",
    "افتحه الان",
    "افتحها الان",
    "شغله",
    "شغلها",
}

_CLOSE_FOLLOWUP_TEXTS = {
    "close it",
    "close this",
    "close that",
    "terminate it",
    "kill it",
    "اقفله",
    "اقفلها",
    "اقفلهم",
    "سكره",
    "سكرها",
}

_DELETE_FOLLOWUP_TEXTS = {
    "delete it",
    "delete this",
    "delete that",
    "remove it",
    "remove this",
    "امسحه",
    "امسحها",
    "شيله",
    "شيلها",
}

_DELETE_VAGUE_FOLLOWUP_TEXTS = {
    "delete it",
    "delete this",
    "delete that",
    "remove it",
    "remove this",
    "امسحه",
    "امسحها",
    "شيله",
    "شيلها",
}

_OPEN_LAST_APP_FOLLOWUP_TEXTS = {
    "open the app",
    "open same app",
    "open that app",
    "افتح البرنامج",
    "افتح نفس البرنامج",
}

_OPEN_LAST_FILE_FOLLOWUP_TEXTS = {
    "open the file",
    "open same file",
    "open that file",
    "open this file",
    "افتح الملف",
    "افتح نفس الملف",
}

_CLOSE_LAST_APP_FOLLOWUP_TEXTS = {
    "close the app",
    "close same app",
    "close that app",
    "اقفل البرنامج",
    "سكر البرنامج",
    "سكرلي البرنامج",
}

_DELETE_LAST_FILE_FOLLOWUP_TEXTS = {
    "delete the file",
    "delete same file",
    "delete that file",
    "remove the file",
    "امسح الملف",
    "شيل الملف",
}

_OPEN_BOTH_FOLLOWUP_TEXTS = {
    "open both",
    "open both of them",
    "open them both",
    "افتح الاثنين",
    "افتحهم",
    "افتحهم الاثنين",
}

_CLOSE_BOTH_FOLLOWUP_TEXTS = {
    "close both",
    "close both of them",
    "close them",
    "close them both",
    "اقفل الاثنين",
    "اقفلهم",
    "سكرهم",
}

_YES_CONFIRM_FOLLOWUP_TEXTS = {
    "yes",
    "yes please",
    "ok",
    "okay",
    "go ahead",
    "proceed",
    "do it",
    "confirm it",
    "approve it",
    "نعم",
    "ايوه",
    "أيوه",
    "تمام",
    "اوكي",
    "أوكي",
    "نفذ",
    "نفذه",
    "نفذها",
}

_NO_CANCEL_FOLLOWUP_TEXTS = {
    "no",
    "no thanks",
    "dont",
    "don't",
    "stop",
    "never mind",
    "nevermind",
    "cancel it",
    "لا",
    "لا شكرا",
    "بلاش",
    "لا تنفذ",
    "لا تنفذه",
    "لا تنفذها",
}

_CONFIRM_FOLLOWUP_TEXTS = {
    "confirm",
    "confirm it",
    "confirm this",
    "confirm that",
    "approve",
    "approve it",
    "اكد",
    "أكد",
    "تاكيد",
    "تأكيد",
    "اكده",
    "أكده",
}

_CANCEL_FOLLOWUP_TEXTS = {
    "cancel",
    "cancel it",
    "cancel this",
    "cancel that",
    "abort",
    "abort it",
    "stop it",
    "الغي",
    "الغيها",
    "الغيه",
    "سيبها",
}

_RENAME_IT_TO_RE = re.compile(r"^\s*(?:rename|change\s+name)\s+(?:it|this|that)\s+to\s+(.+)$", re.IGNORECASE)
_MOVE_IT_TO_RE = re.compile(r"^\s*(?:move)\s+(?:it|this|that)\s+to\s+(.+)$", re.IGNORECASE)
_CONFIRM_IT_WITH_FACTOR_RE = re.compile(
    r"^\s*(?:confirm|approve)\s+(?:it|this|that)\s+(.+)$",
    re.IGNORECASE,
)
_AR_RENAME_IT_TO_RE = re.compile(
    r"^\s*(?:غيره|غيرها|سميه|سميها|سمّيه|سمّيها)\s+(?:ل)\s+(.+)$",
    re.IGNORECASE,
)
_AR_MOVE_IT_TO_RE = re.compile(
    r"^\s*(?:انقله|انقلها|حركه|حركها|وديه|وديها)\s+(?:على)\s+(.+)$",
    re.IGNORECASE,
)
_AR_CONFIRM_IT_WITH_FACTOR_RE = re.compile(
    r"^\s*(?:اكدها|أكدها|اكده|أكده|اكد|أكد)\s+(.+)$",
    re.IGNORECASE,
)
_YES_WITH_FACTOR_RE = re.compile(
    r"^\s*(?:yes|ok(?:ay)?|go\s+ahead|proceed|do\s+it)\s+(.+)$",
    re.IGNORECASE,
)
_AR_YES_WITH_FACTOR_RE = re.compile(
    r"^\s*(?:نعم|ايوه|أيوه|تمام|اوكي|أوكي|نفذ|نفذه|نفذها)\s+(.+)$",
    re.IGNORECASE,
)

_URGENT_MARKERS_EN = {
    "now",
    "right now",
    "quickly",
    "asap",
    "immediately",
    "urgent",
}

_URGENT_MARKERS_AR = {
    "الان",
    "حالا",
    "فورا",
    "بسرعة",
    "حالاً",
    "سريعا",
}

_POLITE_MARKERS_EN = {
    "please",
    "kindly",
}

_POLITE_MARKERS_AR = {
    "من فضلك",
    "لو سمحت",
    "رجاء",
    "رجاءا",
}

_RESPONSE_MODE_EXPLAIN_ON_MARKERS = {
    "explain mode",
    "explain mode on",
    "enable explain mode",
    "turn on explain mode",
    "فعل وضع الشرح",
    "شغل وضع الشرح",
    "وضع الشرح",
}

_RESPONSE_MODE_EXPLAIN_OFF_MARKERS = {
    "explain mode off",
    "disable explain mode",
    "turn off explain mode",
    "الغي وضع الشرح",
    "اقفل وضع الشرح",
}

_RESPONSE_MODE_CONCISE_ON_MARKERS = {
    "concise mode",
    "concise mode on",
    "enable concise mode",
    "turn on concise mode",
    "brief mode",
    "short mode",
    "فعل الوضع المختصر",
    "شغل الوضع المختصر",
    "وضع مختصر",
}

_RESPONSE_MODE_CONCISE_OFF_MARKERS = {
    "concise mode off",
    "disable concise mode",
    "turn off concise mode",
    "الغي الوضع المختصر",
    "اقفل الوضع المختصر",
}

_RESPONSE_MODE_DEFAULT_MARKERS = {
    "default mode",
    "normal mode",
    "reset mode",
    "الوضع الافتراضي",
    "الوضع العادي",
}

_UNCLEAR_QUERY_CLARIFICATION_REPLY_TOKENS = {
    "1",
    "2",
    "3",
    "yes",
    "no",
    "cancel",
    "show",
    "more",
    "first",
    "second",
    "third",
    "this",
    "that",
    "نعم",
    "لا",
    "الغي",
    "الاول",
    "الأول",
    "الثاني",
    "الثالث",
    "هذا",
    "هذه",
    "ذاك",
    "more",
}

_UNCLEAR_QUERY_SUBSTANTIVE_MARKERS = {
    "what",
    "why",
    "how",
    "when",
    "where",
    "who",
    "explain",
    "tell me",
    "about",
    "search",
    "open",
    "close",
    "كيف",
    "لماذا",
    "ماذا",
    "اشرح",
    "اشرح لي",
    "اخبرني",
    "خبرني",
    "عن",
    "دور",
    "افتح",
    "اقفل",
}


def _looks_substantive_unclear_query_followup(text):
    normalized = _normalize_compact(text)
    if not normalized:
        return False

    words = re.findall(r"[a-z0-9\u0600-\u06FF]+", normalized)
    if not words:
        return False

    if normalized in _UNCLEAR_QUERY_CLARIFICATION_REPLY_TOKENS:
        return False

    reply_tokens = set(words)
    if reply_tokens and reply_tokens.issubset(_UNCLEAR_QUERY_CLARIFICATION_REPLY_TOKENS):
        return False

    if len(words) >= 5:
        return True

    if len(words) >= 3 and any(marker in normalized for marker in _UNCLEAR_QUERY_SUBSTANTIVE_MARKERS):
        return True

    if "?" in str(text or "") and len(words) >= 3:
        return True

    return False


def _should_bypass_pending_clarification(parsed, pending_payload=None, source_text=""):
    if not parsed:
        return False
    intent = str(getattr(parsed, "intent", "") or "").strip().upper()
    action = str(getattr(parsed, "action", "") or "").strip().lower()
    if intent == "MEMORY_COMMAND" and action == "set_language":
        return True

    pending_reason = _normalize_compact((pending_payload or {}).get("reason") or "")
    if pending_reason != "low_confidence_unclear_query":
        return False

    # For short/noisy unresolved queries, treat a substantive new utterance as a fresh request
    # instead of forcing the user to stay in stale clarification mode.
    if intent and intent != "LLM_QUERY":
        return True

    return _looks_substantive_unclear_query_followup(source_text)

# Maps intents to their required permission key.
_PERMISSION_MAP = {
    "OS_CONFIRMATION": "confirmation",
    "OS_ROLLBACK": "rollback",
    "OS_FILE_SEARCH": "file_search",
    "OS_APP_OPEN": "app_open",
    "OS_APP_CLOSE": "app_close",
    "OS_SYSTEM_COMMAND": "system_command",
    "METRICS_REPORT": "metrics",
    "AUDIT_LOG_REPORT": "audit_log",
    "AUDIT_VERIFY": "audit_log",
    "AUDIT_RESEAL": "audit_log",
    "POLICY_COMMAND": "policy",
    "BATCH_COMMAND": "batch",
    "SEARCH_INDEX_COMMAND": "search_index",
    "JOB_QUEUE_COMMAND": "job_queue",
    "PERSONA_COMMAND": "persona",
    "VOICE_COMMAND": "speech",
    "KNOWLEDGE_BASE_COMMAND": "knowledge_base",
    "MEMORY_COMMAND": "memory",
    "OBSERVABILITY_REPORT": "observability",
}


def _truncate_text(value, max_chars=180):
    text = " ".join(str(value or "").split())
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _normalize_compact(text):
    return " ".join(str(text or "").lower().split()).strip()


def _llm_cache_key(prompt: str, language: str):
    return (_normalize_compact(language or "en"), _normalize_compact(prompt or ""))


def _cache_get_llm_response(prompt: str, language: str):
    if not LLM_RESPONSE_CACHE_ENABLED:
        return None

    now = time.time()
    key = _llm_cache_key(prompt, language)
    entry = _LLM_RESPONSE_CACHE.get(key)
    if not entry:
        _LLM_RESPONSE_CACHE_STATS["misses"] += 1
        return None

    cached_at = float(entry.get("cached_at") or 0.0)
    if cached_at <= 0 or (now - cached_at) > max(1, int(LLM_RESPONSE_CACHE_TTL_SECONDS or 600)):
        _LLM_RESPONSE_CACHE.pop(key, None)
        _LLM_RESPONSE_CACHE_STATS["misses"] += 1
        _LLM_RESPONSE_CACHE_STATS["evictions"] += 1
        return None

    _LLM_RESPONSE_CACHE.move_to_end(key)
    _LLM_RESPONSE_CACHE_STATS["hits"] += 1
    return str(entry.get("value") or "").strip()


def _cache_put_llm_response(prompt: str, language: str, response: str):
    if not LLM_RESPONSE_CACHE_ENABLED:
        return

    value = str(response or "").strip()
    if not value:
        return

    key = _llm_cache_key(prompt, language)
    _LLM_RESPONSE_CACHE[key] = {
        "cached_at": time.time(),
        "value": value,
    }
    _LLM_RESPONSE_CACHE.move_to_end(key)
    _LLM_RESPONSE_CACHE_STATS["stores"] += 1

    max_size = max(16, int(LLM_RESPONSE_CACHE_MAX_SIZE or 256))
    while len(_LLM_RESPONSE_CACHE) > max_size:
        _LLM_RESPONSE_CACHE.popitem(last=False)
        _LLM_RESPONSE_CACHE_STATS["evictions"] += 1


def clear_llm_response_cache():
    _LLM_RESPONSE_CACHE.clear()
    _LLM_RESPONSE_CACHE_STATS.update({"hits": 0, "misses": 0, "stores": 0, "evictions": 0})


def get_llm_response_cache_stats():
    return {
        "enabled": bool(LLM_RESPONSE_CACHE_ENABLED),
        "size": len(_LLM_RESPONSE_CACHE),
        "hits": int(_LLM_RESPONSE_CACHE_STATS["hits"]),
        "misses": int(_LLM_RESPONSE_CACHE_STATS["misses"]),
        "stores": int(_LLM_RESPONSE_CACHE_STATS["stores"]),
        "evictions": int(_LLM_RESPONSE_CACHE_STATS["evictions"]),
        "ttl_seconds": int(LLM_RESPONSE_CACHE_TTL_SECONDS or 600),
        "max_size": int(LLM_RESPONSE_CACHE_MAX_SIZE or 256),
    }


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


_LOW_VALUE_LLM_REPLY_MARKERS = {
    "i can help with that",
    "i can certainly help with that",
    "i am sorry",
    "i m sorry",
    "i m sorry but",
    "i'm sorry",
    "sorry but",
    "please provide me with some information",
    "do you have any other questions",
    "let me know if you have any other questions",
    "i cannot assist with that directly",
    "i can t assist with that directly",
    "i cannot help with that",
    "i can t help with that",
    "unable to help with that",
    "provide current weather information",
    "check a weather service",
    "cannot provide current news",
    "can t provide current news",
    "check a reliable news source",
    "بالطبع يمكنني",
    "هل لديك اي اسئلة اخرى",
    "هل هناك اي معلومات اخرى",
    "مش هقدر اساعدك",
    "مش اقدر اساعدك",
    "اسف بس",
    "مش هقدر اديك معلومات الطقس دلوقتي",
    "مش اقدر اديك معلومات الطقس دلوقتي",
    "اتأكد من خدمة الطقس",
    "شوف خدمة طقس",
    "سابحث عن احدث تحديثات الرصد الجوي",
    "يمكنك متابعة المحادثة",
    "بمجرد وجودها",
    "اسف",
    "آسف",
}

_WEATHER_QUERY_MARKERS = {
    "weather",
    "temperature",
    "forecast",
    "rain",
    "wind",
    "humidity",
    "what is the weather",
    "طقس",
    "الطقس",
    "درجة الحرارة",
    "حرارة",
    "مطر",
    "رياح",
    "رطوبة",
    "تنبؤ",
    "اخبار الجو",
    "أخبار الجو",
    "الجو النهاردة",
    "جو النهاردة",
    "الجو ايه",
    "الجو عامل ايه",
    "حالة الجو",
    "حاله الجو",
}

_CLOTHING_QUERY_MARKERS = {
    "what should i wear",
    "what to wear",
    "wear today",
    "clothes",
    "clothing",
    "jacket",
    "coat",
    "لبس",
    "البس",
    "ألبس",
    "ملابس",
    "جاكيت",
    "معطف",
}

_NEWS_QUERY_MARKERS = {
    "news",
    "headline",
    "headlines",
    "breaking",
    "today news",
    "world news",
    "اخبار",
    "الأخبار",
    "خبر",
    "العناوين",
    "عاجل",
}

_ASSIST_FIRST_REWRITE_BLOCK_MARKERS = {
    "hack",
    "exploit",
    "malware",
    "virus",
    "ransomware",
    "phishing",
    "bomb",
    "weapon",
    "kill",
    "suicide",
    "self harm",
    "self-harm",
    "terror",
    "ارهاب",
    "إرهاب",
    "متفجر",
    "متفجرات",
    "انتحار",
    "ايذاء النفس",
    "إيذاء النفس",
    "قتل",
}


def _looks_low_value_llm_reply(text):
    normalized = _normalize_quality_text(text)
    if not normalized:
        return True
    word_count = len(normalized.split())
    if word_count == 0:
        return True
    has_marker = any(marker in normalized for marker in _LOW_VALUE_LLM_REPLY_MARKERS)
    return has_marker and word_count <= 90


def _looks_weather_or_clothing_query(text):
    normalized = _normalize_quality_text(text)
    if not normalized:
        return False
    if any(marker in normalized for marker in _WEATHER_QUERY_MARKERS):
        return True
    return any(marker in normalized for marker in _CLOTHING_QUERY_MARKERS)


def _looks_news_query(text):
    normalized = _normalize_quality_text(text)
    if not normalized:
        return False
    return any(marker in normalized for marker in _NEWS_QUERY_MARKERS)


def _is_assist_first_safe_request(text):
    normalized = _normalize_quality_text(text)
    if not normalized:
        return False
    return not any(marker in normalized for marker in _ASSIST_FIRST_REWRITE_BLOCK_MARKERS)


def _fallback_assist_first_response(original_text, language):
    target_language = normalize_language(language)
    if _looks_weather_or_clothing_query(original_text):
        if target_language == "ar":
            return (
                "مش معايا بيانات طقس لحظية دلوقتي. "
                "قاعدة سريعة: في الحر البس لبس خفيف واشرب مية، "
                "في الجو المعتدل البس طبقات خفيفة مع جاكيت خفيف، "
                "وفي البرد او الرياح البس معطف دافي وحذاء مقفول."
            )
        return (
            "I do not have live weather data right now. "
            "Quick rule: in hot weather wear breathable light layers and hydrate, "
            "in mild weather use light layers with a light jacket, "
            "and in cold or windy weather wear a warm coat with closed shoes."
        )

    if _looks_news_query(original_text):
        if target_language == "ar":
            return (
                "مش معايا بث اخبار لحظي جوه الجلسة دي. "
                "بس اقدر اساعدك فوراً: ابعت الموضوع والمنطقة والفترة الزمنية، "
                "وهديك ملخص واضح مع نقاط تحقق سريعة للمصادر."
            )
        return (
            "I do not have live news feed access in this session. "
            "I can still help immediately: share the topic, region, and timeframe, "
            "and I will produce a concise summary structure with quick source-check questions."
        )

    if target_language == "ar":
        return (
            "اقدر اساعدك بشكل مباشر. "
            "اكتب هدفك في سطر واحد مع اي قيود مهمة، "
            "وهديك خطوات عملية قصيرة وواضحة."
        )
    return (
        "I can help directly. "
        "Share your exact goal in one line plus any constraints, "
        "and I will give you a concise, practical step-by-step answer."
    )


def _normalize_supported_language_tag(value):
    key = str(value or "").strip().lower()
    if key in {"ar", "arabic"}:
        return "ar"
    if key in {"en", "english"}:
        return "en"
    return ""


def _shorten_to_words(text, max_words=16):
    words = str(text or "").split()
    if len(words) <= max(1, int(max_words or 1)):
        return str(text or "").strip()
    trimmed = " ".join(words[: max(1, int(max_words or 1))]).rstrip(".,;: ")
    return f"{trimmed}..."


def _analyze_tone_markers(text, language="en"):
    normalized = _normalize_compact(text)
    if not normalized:
        return {"urgent": False, "polite": False}

    urgent_markers = _URGENT_MARKERS_AR if str(language).strip().lower() == "ar" else _URGENT_MARKERS_EN
    polite_markers = _POLITE_MARKERS_AR if str(language).strip().lower() == "ar" else _POLITE_MARKERS_EN

    urgent = any(marker in normalized for marker in urgent_markers)
    polite = any(marker in normalized for marker in polite_markers)
    return {"urgent": bool(urgent), "polite": bool(polite)}


def _try_handle_response_mode_toggle(text, language):
    if not RESPONSE_MODE_FEATURE_ENABLED:
        return ""

    normalized = _normalize_compact(text)
    if not normalized:
        return ""

    mode = ""
    if normalized in _RESPONSE_MODE_EXPLAIN_ON_MARKERS:
        mode = "explain"
    elif normalized in _RESPONSE_MODE_EXPLAIN_OFF_MARKERS:
        mode = "default"
    elif normalized in _RESPONSE_MODE_CONCISE_ON_MARKERS:
        mode = "concise"
    elif normalized in _RESPONSE_MODE_CONCISE_OFF_MARKERS:
        mode = "default"
    elif normalized in _RESPONSE_MODE_DEFAULT_MARKERS:
        mode = "default"

    if not mode:
        return ""

    current_mode = session_memory.get_response_mode()
    if current_mode != mode:
        session_memory.set_response_mode(mode)

    if mode == "explain":
        return render_template("response_mode_explain_on", language)
    if mode == "concise":
        return render_template("response_mode_concise_on", language)
    return render_template("response_mode_default_on", language)


def _apply_output_mode(response_text, parsed, language):
    mode = session_memory.get_response_mode()
    text = str(response_text or "").strip()
    if mode == "default" or not text:
        return text

    if mode == "concise":
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
        max_words = 18 if parsed and parsed.intent == "LLM_QUERY" else 14
        return _shorten_to_words(first_line, max_words=max_words)

    if mode == "explain":
        lexical = persona_manager.get_lexical_bank(language=language)
        bridge = str(lexical.get("explain_bridge") or render_template("response_explain_bridge", language)).strip()
        explain_suffix = render_template(
            "response_mode_explain_suffix",
            language,
            bridge=bridge,
            intent=str(getattr(parsed, "intent", "unknown") or "unknown"),
            action=str(getattr(parsed, "action", "") or "n/a"),
        )
        if explain_suffix and explain_suffix not in text:
            separator = " " if text.endswith((".", "!", "?", "؟")) else ". "
            return f"{text}{separator}{explain_suffix}"
    return text


def _apply_tone_adaptation(response_text, language, tone_meta, parsed=None):
    if not TONE_ADAPTATION_ENABLED:
        return response_text
    text = str(response_text or "").strip()
    if not text:
        return text

    tone = dict(tone_meta or {})
    lexical = persona_manager.get_lexical_bank(language=language)
    urgent_prefixes = list(lexical.get("urgent_prefixes") or [])
    gentle_prefixes = list(lexical.get("gentle_prefixes") or [])
    is_sensitive = _is_sensitive_command(parsed)

    if tone.get("urgent"):
        max_words = 18 if is_sensitive else 16
        text = _shorten_to_words(text, max_words=max_words)
        if is_sensitive and TONE_SENSITIVE_NEUTRAL_ENABLED:
            prefix = "سأنفذ بحذر." if str(language).strip().lower() == "ar" else "Proceeding safely."
        else:
            prefix = urgent_prefixes[0] if urgent_prefixes else ""
        if prefix and not _normalize_compact(text).startswith(_normalize_compact(prefix)):
            text = f"{prefix} {text}".strip()
        return text

    if tone.get("polite"):
        if is_sensitive and TONE_SENSITIVE_NEUTRAL_ENABLED:
            return text
        prefix = gentle_prefixes[0] if gentle_prefixes else ""
        if prefix and not _normalize_compact(text).startswith(_normalize_compact(prefix)):
            text = f"{prefix} {text}".strip()
    return text


def _apply_codeswitch_continuity(response_text, language, parsed=None):
    if not CODE_SWITCH_CONTINUITY_ENABLED:
        return response_text
    text = str(response_text or "").strip()
    if not text:
        return text
    if parsed and str(parsed.intent or "").strip().upper() == "LLM_QUERY":
        return text
    continuity_window = max(2, int(CODE_SWITCH_CONTINUITY_WINDOW or 6))
    if not session_memory.is_code_switch_active(window=continuity_window):
        return text

    lexical = persona_manager.get_lexical_bank(language=language)
    mix = session_memory.get_language_mix(window=continuity_window)
    dominant_ratio = max(0.50, min(0.90, float(CODE_SWITCH_DOMINANT_RATIO or 0.70)))
    dominant = str(mix.get("dominant") or "mixed")
    en_ratio = float(mix.get("en_ratio") or 0.0)
    ar_ratio = float(mix.get("ar_ratio") or 0.0)

    if dominant == "en" and en_ratio >= dominant_ratio:
        bridge = "I can switch to العربية anytime if you prefer."
    elif dominant == "ar" and ar_ratio >= dominant_ratio:
        bridge = "ممكن احول لـ English في اي وقت لو تحب."
    else:
        bridge = str(lexical.get("codeswitch_bridge") or "").strip()

    if not bridge or bridge in text:
        return text

    if "\n" in text:
        return f"{text}\n{bridge}"
    return f"{text} {bridge}"


def _record_response_quality(response_text, language, user_text):
    recent = session_memory.recent(limit=1)
    previous_response = ""
    if recent:
        previous_response = str((recent[-1] or {}).get("assistant") or "")
    metrics.record_response_quality(
        response_text,
        language=language,
        user_text=user_text,
        previous_response=previous_response,
        persona=persona_manager.get_profile(),
        response_mode=session_memory.get_response_mode(),
    )


def _apply_egyptian_dialect_style(response_text, parsed, language):
    text = str(response_text or "").strip()
    if not text:
        return text
    if normalize_language(language) != "ar":
        return text
    if not parsed or str(parsed.intent or "").strip().upper() != "LLM_QUERY":
        return text

    try:
        from audio.tts import _rewrite_to_egyptian_colloquial

        rewritten = str(_rewrite_to_egyptian_colloquial(text) or "").strip()
        if rewritten:
            return rewritten
    except Exception:
        pass

    return text


def _repair_low_value_llm_response(response_text, parsed, language, original_text, *, allow_llm_rewrite=True):
    text = str(response_text or "").strip()
    if not text:
        return text
    if not parsed or str(parsed.intent or "").strip().upper() != "LLM_QUERY":
        return text
    if not _looks_low_value_llm_reply(text):
        return text
    if not _is_assist_first_safe_request(original_text):
        return text

    target_language = normalize_language(language)
    target_label = "Arabic" if target_language == "ar" else "English"

    if _looks_weather_or_clothing_query(original_text):
        rewrite_prompt = (
            f"Rewrite the weak draft into a useful {target_label} response for a weather/clothing question.\n"
            "- Give a direct answer first.\n"
            "- If you do not have live weather data, say that briefly in one line only.\n"
            "- Then provide practical clothing guidance for hot/mild/cold conditions.\n"
            "- Keep it concise, concrete, and natural.\n"
            "- Do not end with a refusal-style sentence.\n"
            "- Return only the final answer.\n\n"
            f"User request: {_truncate_text(original_text, max_chars=300)}\n"
            f"Weak draft:\n{text}"
        )
    else:
        rewrite_prompt = (
            f"Rewrite the weak draft into a useful {target_label} answer.\n"
            "- Answer directly with concrete information.\n"
            "- If live data is unavailable, mention that briefly and give practical guidance.\n"
            "- Keep the same topic and intent.\n"
            "- Do not use generic refusal language unless the request is unsafe.\n"
            "- Return only the final answer.\n\n"
            f"User request: {_truncate_text(original_text, max_chars=300)}\n"
            f"Weak draft:\n{text}"
        )

    if allow_llm_rewrite:
        improved = (ask_llm(rewrite_prompt) or "").strip()
        if improved and not _looks_low_value_llm_reply(improved):
            log_structured(
                "route_llm_quality_repair",
                language=target_language,
                response_preview=_truncate_text(improved),
            )
            return improved

    # Hard assist-first rule: for normal safe user requests, never leave a generic
    # dead-end refusal as the final answer.
    assist_first = _fallback_assist_first_response(original_text, target_language)
    if assist_first and not _looks_low_value_llm_reply(assist_first):
        log_structured(
            "route_llm_quality_fallback",
            language=target_language,
            response_preview=_truncate_text(assist_first),
        )
        return assist_first

    return text


def _enforce_llm_response_language(response_text, parsed, language, original_text):
    text = str(response_text or "").strip()
    if not text:
        return text
    if not parsed or str(parsed.intent or "").strip().upper() != "LLM_QUERY":
        return text

    target_language = normalize_language(language)
    detected_language = detect_language_hint(text, fallback=target_language)
    if detected_language == target_language:
        return text

    target_label = "Arabic" if target_language == "ar" else "English"
    rewrite_prompt = (
        f"Rewrite the assistant answer below into {target_label} only.\n"
        "- Keep the exact meaning.\n"
        "- Do not add or remove facts.\n"
        "- Keep similar length and tone.\n"
        "- Return only the rewritten answer.\n\n"
        f"User request: {_truncate_text(original_text, max_chars=260)}\n"
        f"Assistant answer:\n{text}"
    )
    rewritten = (ask_llm(rewrite_prompt) or "").strip()
    if not rewritten:
        return text

    rewritten_language = detect_language_hint(rewritten, fallback=target_language)
    if rewritten_language != target_language:
        return text

    log_structured(
        "route_llm_language_rewrite",
        language=target_language,
        previous_language=detected_language,
        response_preview=_truncate_text(rewritten),
    )
    return rewritten


def _finalize_success_response(response_text, parsed, language, original_text, tone_meta, *, realtime=False):
    text = str(response_text or "").strip()
    allow_llm_rewrite = bool(LLM_REALTIME_REWRITE_ENABLED) or not bool(realtime)
    text = _repair_low_value_llm_response(
        text,
        parsed,
        language,
        original_text,
        allow_llm_rewrite=allow_llm_rewrite,
    )
    if allow_llm_rewrite:
        text = _enforce_llm_response_language(text, parsed, language, original_text)
    text = _apply_egyptian_dialect_style(text, parsed, language)
    text = _apply_persona_length_target(text, parsed)
    text = _apply_output_mode(text, parsed, language)
    text = _apply_tone_adaptation(text, language, tone_meta, parsed=parsed)
    text = _apply_codeswitch_continuity(text, language, parsed=parsed)
    text = _apply_anti_repetition(text, language)
    _record_response_quality(text, language, original_text)
    return text


def _reference_confidence(timestamp, slot_type="generic"):
    ts = float(timestamp or 0.0)
    if ts <= 0.0:
        return 0.0

    slot_key = str(slot_type or "").strip().lower()
    max_age = max(5, int(FOLLOWUP_REFERENCE_MAX_AGE_SECONDS or 1800))
    half_life = max(5, int(FOLLOWUP_REFERENCE_MAX_AGE_SECONDS or 1800) // 2)

    if slot_key == "last_app":
        max_age = max(5, int(FOLLOWUP_APP_REFERENCE_MAX_AGE_SECONDS or max_age))
        half_life = max(5, int(FOLLOWUP_APP_REFERENCE_HALF_LIFE_SECONDS or half_life))
    elif slot_key == "last_file":
        max_age = max(5, int(FOLLOWUP_FILE_REFERENCE_MAX_AGE_SECONDS or max_age))
        half_life = max(5, int(FOLLOWUP_FILE_REFERENCE_HALF_LIFE_SECONDS or half_life))
    elif slot_key in {"pending_confirmation", "pending_confirmation_token"}:
        max_age = max(5, int(FOLLOWUP_PENDING_CONFIRMATION_MAX_AGE_SECONDS or 180))
        half_life = max(5, int(FOLLOWUP_PENDING_CONFIRMATION_HALF_LIFE_SECONDS or 75))

    age = max(0.0, time.time() - ts)
    if age > max_age:
        return 0.0

    confidence = pow(0.5, age / float(half_life))
    return max(0.0, min(1.0, confidence))


def _is_fresh_reference(timestamp, slot_type="generic"):
    if session_memory.slot_is_fresh(slot_type, updated_at=timestamp):
        return True
    confidence = _reference_confidence(timestamp, slot_type=slot_type)
    min_confidence = float(FOLLOWUP_REFERENCE_MIN_CONFIDENCE or 0.2)
    if confidence >= min_confidence:
        return True
    if str(slot_type or "").strip().lower() == "generic" and timestamp:
        # Backward-compatible fallback for any call site that still uses generic slots.
        max_age = max(5, int(FOLLOWUP_REFERENCE_MAX_AGE_SECONDS or 1800))
        return (time.time() - float(timestamp or 0.0)) <= max_age
    return False


def _apply_persona_length_target(response_text, parsed):
    if not PERSONA_LENGTH_TARGET_ENABLED:
        return response_text
    if not response_text:
        return response_text
    if (response_text or "").count("\n") > 0:
        return response_text
    if not parsed or parsed.intent != "LLM_QUERY":
        return response_text

    persona_key = persona_manager.get_profile()
    max_words = int((PERSONA_RESPONSE_MAX_WORDS or {}).get(persona_key) or 0)
    if max_words <= 0:
        return response_text

    words = str(response_text).split()
    if len(words) <= max_words:
        return response_text
    shortened = " ".join(words[:max_words]).rstrip(".,;: ")
    return f"{shortened}..."


def _required_permission(parsed):
    if parsed.intent == "OS_FILE_NAVIGATION":
        if parsed.action in {"create_directory", "delete_item", "delete_item_permanent", "move_item", "rename_item"}:
            return "file_write"
        return "file_navigation"
    return _PERMISSION_MAP.get(parsed.intent)


def _execute_confirmed_payload(payload):
    kind = (payload or {}).get("kind")
    if kind == "system_command":
        action_key = payload.get("action_key")
        command_args = dict(payload.get("command_args") or {})
        return to_router_tuple(execute_system_command_result(action_key, command_args=command_args))
    if kind == "file_operation":
        return to_router_tuple(execute_confirmed_file_operation(payload))
    if kind == "app_operation":
        return to_router_tuple(execute_confirmed_app_operation(payload))
    log_action(
        "confirmation_rejected",
        "failed",
        details={"reason": "unsupported_payload_kind", "kind": str(kind or "")},
    )
    language = session_memory.get_preferred_language()
    return False, render_template("unsupported_confirmation_payload", language), {}


def _format_source_citations(sources):
    if not sources:
        return ""
    lines = ["", "Sources:"]
    seen = set()
    for item in sources:
        key = (item.get("source"), item.get("chunk_index"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {item.get('source')} (chunk {item.get('chunk_index')})")
    return "\n".join(lines)


def _normalize_repetition_text(text):
    return " ".join((text or "").lower().split()).strip()


def _apply_anti_repetition(response_text, language):
    if (response_text or "").count("\n") > 3:
        return response_text

    normalized_response = _normalize_repetition_text(response_text)
    if not normalized_response:
        return response_text

    recent = session_memory.recent(limit=3)
    if not recent:
        return response_text

    last_assistant = _normalize_repetition_text((recent[-1] or {}).get("assistant") or "")
    if normalized_response != last_assistant:
        return response_text

    language_key = detect_language_hint(response_text, fallback=language)
    persona_key = persona_manager.get_profile()
    prefixes = anti_repetition_prefixes(language_key, persona_key)
    if not prefixes:
        return response_text

    prefix = prefixes[len(recent) % len(prefixes)]
    if _normalize_repetition_text(prefix) and normalized_response.startswith(_normalize_repetition_text(prefix)):
        return response_text
    return f"{prefix}{response_text}"


def _should_store_turn(parsed, response_text):
    if not parsed or not response_text:
        return False
    if len(response_text) > 2000 or response_text.count("\n") > 20:
        return False
    if parsed.intent in {
        "METRICS_REPORT",
        "OBSERVABILITY_REPORT",
        "AUDIT_LOG_REPORT",
        "AUDIT_VERIFY",
        "AUDIT_RESEAL",
    }:
        return False
    return True


def _rewrite_followup_command(text, language="en"):
    raw = str(text or "").strip()
    normalized = " ".join(raw.lower().split())
    if not normalized:
        return text, {}

    pending_clarification = session_memory.get_pending_clarification()
    pending_token = session_memory.get_pending_confirmation_token()
    pending_token_ts = session_memory.get_pending_confirmation_timestamp()
    has_fresh_pending_token = bool(pending_token) and _is_fresh_reference(
        pending_token_ts,
        slot_type="pending_confirmation",
    )
    if pending_token and not has_fresh_pending_token:
        session_memory.clear_pending_confirmation_token()
        pending_token = ""

    if normalized in _CANCEL_FOLLOWUP_TEXTS and pending_token and not pending_clarification:
        return raw, {"followup_cancel_confirmation": True, "token": pending_token}

    if normalized in _NO_CANCEL_FOLLOWUP_TEXTS and pending_token and not pending_clarification:
        return raw, {
            "followup_cancel_confirmation": True,
            "followup_rewrite": "confirmation_implicit_no",
            "token": pending_token,
        }

    yes_with_factor_match = _YES_WITH_FACTOR_RE.match(raw) or _AR_YES_WITH_FACTOR_RE.match(raw)
    if yes_with_factor_match:
        if pending_token:
            second_factor = yes_with_factor_match.group(1).strip()
            return (
                f"confirm {pending_token} {second_factor}",
                {"followup_rewrite": "confirmation_implicit_yes", "token": pending_token},
            )
        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_pending_confirmation", language),
        }

    if normalized in _YES_CONFIRM_FOLLOWUP_TEXTS and pending_token:
        return f"confirm {pending_token}", {"followup_rewrite": "confirmation_implicit_yes", "token": pending_token}

    factor_match = _CONFIRM_IT_WITH_FACTOR_RE.match(raw) or _AR_CONFIRM_IT_WITH_FACTOR_RE.match(raw)
    if factor_match:
        if pending_token:
            second_factor = factor_match.group(1).strip()
            return (
                f"confirm {pending_token} {second_factor}",
                {"followup_rewrite": "confirmation", "token": pending_token},
            )
        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_pending_confirmation", language),
        }

    if normalized in _CONFIRM_FOLLOWUP_TEXTS and pending_token:
        return f"confirm {pending_token}", {"followup_rewrite": "confirmation", "token": pending_token}

    if normalized in _CONFIRM_FOLLOWUP_TEXTS and not pending_token:
        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_pending_confirmation", language),
        }

    last_file = session_memory.get_last_file()
    last_file_ts = session_memory.get_last_file_timestamp()
    has_fresh_file = bool(last_file) and _is_fresh_reference(last_file_ts, slot_type="last_file")
    has_stale_file = bool(last_file) and not has_fresh_file

    last_app = session_memory.get_last_app()
    last_app_ts = session_memory.get_last_app_timestamp()
    has_fresh_app = bool(last_app) and _is_fresh_reference(last_app_ts, slot_type="last_app")
    has_stale_app = bool(last_app) and not has_fresh_app
    previous_app = session_memory.get_previous_app()
    previous_app_ts = session_memory.get_previous_app_timestamp()
    has_fresh_previous_app = bool(previous_app) and _is_fresh_reference(previous_app_ts, slot_type="previous_app")
    has_stale_previous_app = bool(previous_app) and not has_fresh_previous_app

    rename_match = _RENAME_IT_TO_RE.match(raw) or _AR_RENAME_IT_TO_RE.match(raw)
    if rename_match:
        if has_fresh_file:
            return (
                f"rename {last_file} to {rename_match.group(1).strip()}",
                {"followup_rewrite": "rename_last_file", "last_file": last_file},
            )
        if has_stale_file:
            return raw, {
                "followup_blocked": True,
                "followup_message": render_template("stale_followup_reference", language),
            }
        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_last_file_rename", language),
        }

    move_match = _MOVE_IT_TO_RE.match(raw) or _AR_MOVE_IT_TO_RE.match(raw)
    if move_match:
        if has_fresh_file:
            return (
                f"move {last_file} to {move_match.group(1).strip()}",
                {"followup_rewrite": "move_last_file", "last_file": last_file},
            )
        if has_stale_file:
            return raw, {
                "followup_blocked": True,
                "followup_message": render_template("stale_followup_reference", language),
            }
        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_last_file_move", language),
        }

    if normalized in _DELETE_FOLLOWUP_TEXTS or normalized in _DELETE_LAST_FILE_FOLLOWUP_TEXTS:
        if FOLLOWUP_DESTRUCTIVE_REQUIRE_EXPLICIT_REFERENCE and normalized in _DELETE_VAGUE_FOLLOWUP_TEXTS:
            return raw, {
                "followup_blocked": True,
                "followup_message": render_template("destructive_followup_requires_explicit_target", language),
            }

        destructive_confidence = session_memory.slot_confidence("last_file", updated_at=last_file_ts)
        destructive_min_confidence = max(
            float(FOLLOWUP_REFERENCE_MIN_CONFIDENCE or 0.2),
            float(FOLLOWUP_DESTRUCTIVE_REFERENCE_MIN_CONFIDENCE or 0.55),
        )
        if has_fresh_file:
            if destructive_confidence < destructive_min_confidence:
                return raw, {
                    "followup_blocked": True,
                    "followup_message": render_template("destructive_followup_low_confidence", language),
                }
            return f"delete {last_file}", {"followup_rewrite": "delete_last_file", "last_file": last_file}
        if has_stale_file:
            return raw, {
                "followup_blocked": True,
                "followup_message": render_template("stale_followup_reference", language),
            }
        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_last_file_delete", language),
        }

    if normalized in _OPEN_LAST_APP_FOLLOWUP_TEXTS:
        if has_fresh_app:
            return f"open app {last_app}", {"followup_rewrite": "open_last_app", "last_app": last_app}
        if has_stale_app:
            return raw, {
                "followup_blocked": True,
                "followup_message": render_template("stale_followup_reference", language),
            }
        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_last_app_open", language),
        }

    if normalized in _OPEN_LAST_FILE_FOLLOWUP_TEXTS:
        if has_fresh_file:
            if os.path.isdir(last_file):
                return f"open {last_file}", {"followup_rewrite": "open_last_file", "last_file": last_file}
            return f"file info {last_file}", {"followup_rewrite": "file_info_last_file", "last_file": last_file}
        if has_stale_file:
            return raw, {
                "followup_blocked": True,
                "followup_message": render_template("stale_followup_reference", language),
            }
        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_followup_reference", language),
        }

    if normalized in _OPEN_BOTH_FOLLOWUP_TEXTS:
        actions = []
        if has_fresh_app:
            actions.append(
                {
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": last_app},
                }
            )
        if has_fresh_previous_app and previous_app.lower() != last_app.lower():
            actions.append(
                {
                    "intent": "OS_APP_OPEN",
                    "action": "",
                    "args": {"app_name": previous_app},
                }
            )
        if len(actions) < 2 and has_fresh_file:
            file_action = "open" if os.path.isdir(last_file) else "file_info"
            actions.append(
                {
                    "intent": "OS_FILE_NAVIGATION",
                    "action": file_action,
                    "args": {"path": last_file},
                }
            )

        if len(actions) >= 2:
            return raw, {
                "followup_rewrite": "open_both_recent_targets",
                "followup_multi_actions": actions[:2],
            }

        if has_stale_app or has_stale_file or has_stale_previous_app:
            return raw, {
                "followup_blocked": True,
                "followup_message": render_template("stale_followup_reference", language),
            }

        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_second_recent_app", language),
        }

    if normalized in _CLOSE_BOTH_FOLLOWUP_TEXTS:
        actions = []
        if has_fresh_app:
            actions.append(
                {
                    "intent": "OS_APP_CLOSE",
                    "action": "",
                    "args": {"app_name": last_app},
                }
            )
        if has_fresh_previous_app and previous_app.lower() != last_app.lower():
            actions.append(
                {
                    "intent": "OS_APP_CLOSE",
                    "action": "",
                    "args": {"app_name": previous_app},
                }
            )

        if len(actions) >= 2:
            return raw, {
                "followup_rewrite": "close_both_recent_apps",
                "followup_multi_actions": actions[:2],
            }

        if has_stale_app or has_stale_previous_app:
            return raw, {
                "followup_blocked": True,
                "followup_message": render_template("stale_followup_reference", language),
            }

        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_second_recent_app", language),
        }

    if normalized in _OPEN_FOLLOWUP_TEXTS:
        candidates = []
        has_stale_reference = bool(has_stale_file or has_stale_app)
        if has_fresh_file and has_fresh_app:
            conflict_window = max(0.0, float(FOLLOWUP_REFERENCE_CONFLICT_WINDOW_SECONDS or 0.0))
            if conflict_window > 0.0 and abs(float(last_file_ts or 0.0) - float(last_app_ts or 0.0)) <= float(conflict_window):
                return raw, {
                    "followup_blocked": True,
                    "followup_message": render_template("followup_reference_conflict", language),
                }
        if has_fresh_file:
            if os.path.isdir(last_file):
                candidates.append((last_file_ts, f"open {last_file}", "open_last_file", {"last_file": last_file}))
            elif os.path.isfile(last_file):
                candidates.append((last_file_ts, f"file info {last_file}", "file_info_last_file", {"last_file": last_file}))
            else:
                candidates.append((last_file_ts, f"file info {last_file}", "file_info_last_file", {"last_file": last_file}))
        if has_fresh_app:
            candidates.append((last_app_ts, f"open app {last_app}", "open_last_app", {"last_app": last_app}))

        if candidates:
            _ts, rewritten, rewrite_name, extra_meta = max(candidates, key=lambda row: row[0])
            meta = {"followup_rewrite": rewrite_name}
            meta.update(extra_meta)
            return rewritten, meta

        if has_stale_reference:
            return raw, {
                "followup_blocked": True,
                "followup_message": render_template("stale_followup_reference", language),
            }

        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_followup_reference", language),
        }

    if normalized in _CLOSE_FOLLOWUP_TEXTS or normalized in _CLOSE_LAST_APP_FOLLOWUP_TEXTS:
        if has_fresh_app:
            return f"close app {last_app}", {"followup_rewrite": "close_last_app", "last_app": last_app}
        if has_stale_app:
            return raw, {
                "followup_blocked": True,
                "followup_message": render_template("stale_followup_reference", language),
            }
        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_last_app_close", language),
        }

    return text, {}


def _update_short_term_context(parsed, success, message, meta):
    token = str(meta.get("token") or "").strip().lower()
    if token:
        session_memory.set_pending_confirmation_token(token)
    elif parsed.intent == "OS_CONFIRMATION" and success:
        session_memory.clear_pending_confirmation_token()
    elif parsed.intent == "OS_CONFIRMATION" and not success:
        lowered_message = str(message or "").lower()
        if "not found or expired" in lowered_message or "token expired" in lowered_message:
            session_memory.clear_pending_confirmation_token()

    if parsed.intent == "OS_FILE_SEARCH" and success and not meta.get("clarification_payload"):
        candidate = str(message or "").strip()
        if candidate and (":\\" in candidate or "/" in candidate):
            session_memory.set_last_file(candidate)

    if parsed.intent in {"OS_APP_OPEN", "OS_APP_CLOSE"} and success:
        app_name = (
            str(meta.get("target") or "").strip()
            or str((parsed.args or {}).get("app_name") or "").strip()
            or str(meta.get("process_name") or "").strip()
        )
        if app_name:
            session_memory.set_last_app(app_name)
            session_memory.record_app_usage(app_name)

    if parsed.intent == "OS_FILE_NAVIGATION" and success:
        action = parsed.action
        args = dict(parsed.args or {})
        path = ""
        if action in {"cd", "list_directory", "file_info", "create_directory", "delete_item", "delete_item_permanent"}:
            path = str(args.get("path") or "").strip()
        elif action in {"move_item", "rename_item"}:
            path = str(args.get("destination") or args.get("source") or "").strip()
        if path:
            session_memory.set_last_file(path)

    if parsed.intent == "OS_CONFIRMATION" and success:
        operation = str(meta.get("operation") or "").strip()
        if operation == "close_app":
            app_name = str(meta.get("target") or meta.get("process_name") or "").strip()
            if app_name:
                session_memory.set_last_app(app_name)
                session_memory.record_app_usage(app_name)
        if operation in {"delete_item", "delete_item_permanent", "move_item", "rename_item", "create_directory", "file_info"}:
            candidate_path = str(meta.get("path") or meta.get("destination") or meta.get("source") or "").strip()
            if candidate_path:
                session_memory.set_last_file(candidate_path)


def _build_paginated_runtime_prompt(header_lines, options_page, *, page_index, total_pages, language):
    lines = [str(line).strip() for line in (header_lines or []) if str(line or "").strip()]
    for index, option in enumerate(options_page, start=1):
        lines.append(f"{index}) {option.get('label')}")

    if total_pages > 1:
        lines.append(
            render_template(
                "clarification_page_indicator",
                language,
                page=int(page_index) + 1,
                total_pages=int(total_pages),
            )
        )
        if page_index < (total_pages - 1):
            lines.append(render_template("reply_with_number_cancel_or_more", language))
        else:
            lines.append(render_template("reply_with_number_or_cancel", language))
    else:
        lines.append(render_template("reply_with_number_or_cancel", language))

    return "\n".join(lines)


def _build_runtime_page_prompts(all_options, header_lines, *, page_size, language):
    size = max(1, int(page_size or 1))
    pages = []
    total_pages = max(1, (len(all_options) + size - 1) // size)
    for page_index in range(total_pages):
        start = page_index * size
        page_options = all_options[start : start + size]
        pages.append(
            _build_paginated_runtime_prompt(
                header_lines,
                page_options,
                page_index=page_index,
                total_pages=total_pages,
                language=language,
            )
        )
    return pages


def _build_app_runtime_clarification(app_query, candidates, *, operation="open"):
    operation_mode = "close" if operation == "close" else "open"
    intent = "OS_APP_CLOSE" if operation_mode == "close" else "OS_APP_OPEN"
    option_prefix = "close_app_runtime" if operation_mode == "close" else "open_app_runtime"
    language = session_memory.get_preferred_language()

    all_options = []
    for index, candidate in enumerate(candidates, start=1):
        canonical = candidate.get("canonical_name") or candidate.get("executable")
        executable = candidate.get("executable")
        matched_alias = candidate.get("matched_alias") or ""
        canonical_tokens = str(canonical).lower().split()
        executable_tokens = str(executable).lower().replace(".exe", "").split()
        alias_tokens = str(matched_alias).lower().split()
        label = f"{canonical} ({executable})"
        all_options.append(
            {
                "id": f"{option_prefix}_{index}",
                "label": label,
                "intent": intent,
                "action": "",
                "args": {"app_name": executable},
                "reply_tokens": [
                    str(index),
                    str(canonical).lower(),
                    str(executable).lower(),
                    str(matched_alias).lower(),
                    "app",
                    "\u062a\u0637\u0628\u064a\u0642",
                    *canonical_tokens,
                    *executable_tokens,
                    *alias_tokens,
                ],
            }
        )

    page_size = 3
    header_lines = [
        render_template("clarification_confidence_line", language, confidence_percent=58),
        render_template(f"app_ambiguous_{operation_mode}_intro", language),
    ]
    page_prompts = _build_runtime_page_prompts(
        all_options,
        header_lines,
        page_size=page_size,
        language=language,
    )
    options = all_options[:page_size]
    prompt = page_prompts[0] if page_prompts else render_template(f"app_ambiguous_{operation_mode}_intro", language)
    payload = {
        "reason": "app_close_ambiguous" if operation_mode == "close" else "app_name_ambiguous",
        "prompt": prompt,
        "options": options,
        "all_options": all_options,
        "page_size": page_size,
        "page_index": 0,
        "page_prompts": page_prompts,
        "prompt_intro": render_template(f"app_ambiguous_{operation_mode}_intro", language),
        "source_text": app_query,
        "language": language,
        "confidence": 0.58,
        "entity_scores": {"app_name": 0.62},
    }
    return prompt, payload


def _build_file_search_runtime_clarification(filename, matches):
    language = session_memory.get_preferred_language()
    all_options = []
    for index, match in enumerate(matches, start=1):
        all_options.append(
            {
                "id": f"file_match_{index}",
                "label": match,
                "intent": "OS_FILE_NAVIGATION",
                "action": "file_info",
                "args": {"path": match},
                "reply_tokens": [str(index), str(match).lower()],
            }
        )

    page_size = 5
    intro = render_template("file_ambiguous_intro", language, filename=filename)
    header_lines = [
        render_template("clarification_confidence_line", language, confidence_percent=60),
        intro,
    ]
    page_prompts = _build_runtime_page_prompts(
        all_options,
        header_lines,
        page_size=page_size,
        language=language,
    )
    options = all_options[:page_size]
    prompt = page_prompts[0] if page_prompts else intro
    payload = {
        "reason": "file_search_multiple_matches",
        "prompt": prompt,
        "options": options,
        "all_options": all_options,
        "page_size": page_size,
        "page_index": 0,
        "page_prompts": page_prompts,
        "prompt_intro": intro,
        "source_text": filename,
        "language": language,
        "confidence": 0.60,
        "entity_scores": {"filename": 0.66},
    }
    return prompt, payload


def _ensure_job_queue_executor():
    global _JOB_QUEUE_EXECUTOR_READY
    if _JOB_QUEUE_EXECUTOR_READY:
        return
    job_queue_service.configure_executor(_execute_job_command)
    _JOB_QUEUE_EXECUTOR_READY = True


def _execute_internal_command_text(command_text):
    parsed = parse_command(command_text)
    if parsed.intent == "OS_FILE_NAVIGATION" and parsed.action in {"delete_item", "delete_item_permanent", "move_item", "rename_item"}:
        return False, "Risky file operations are not allowed in batch commit; run interactively."
    if parsed.intent == "OS_APP_CLOSE":
        return False, "Risky app-close operations are not allowed in batch commit; run interactively."
    success, message, _meta = _dispatch(
        parsed,
        allow_batch=False,
        allow_job_queue=False,
        allow_llm=False,
    )
    return success, message


def _execute_job_command(command_text):
    parsed = parse_command(command_text)
    if parsed.intent in {
        "JOB_QUEUE_COMMAND",
        "BATCH_COMMAND",
        "OS_CONFIRMATION",
        "OS_SYSTEM_COMMAND",
        "VOICE_COMMAND",
        "AUDIT_RESEAL",
        "OS_APP_CLOSE",
    }:
        return False, f"Disallowed command for queued execution: {parsed.intent}"
    if parsed.intent == "OS_FILE_NAVIGATION" and parsed.action in {"delete_item", "delete_item_permanent", "move_item", "rename_item"}:
        return False, "Disallowed command for queued execution: risky file operation"
    success, message, _meta = _dispatch(
        parsed,
        allow_batch=False,
        allow_job_queue=False,
        allow_llm=False,
    )
    return success, message


def _dispatch(parsed, *, allow_batch=True, allow_job_queue=True, allow_llm=True, on_sentence=None):
    logger.info("Command parsed: %s (%s)", parsed.intent, parsed.action or "no-action")
    language = session_memory.get_preferred_language()

    if parsed.intent == "DEMO_MODE":
        if parsed.action == "on":
            set_demo_mode(True)
            return True, "Demo mode enabled.", {}
        if parsed.action == "off":
            set_demo_mode(False)
            return True, "Demo mode disabled.", {}
        enabled = is_demo_mode_enabled()
        return True, f"Demo mode is {'ON' if enabled else 'OFF'}.", {}

    permission_key = _required_permission(parsed)
    if permission_key and not policy_engine.is_command_allowed(permission_key):
        return False, f"Command blocked by policy: {permission_key}", {}

    if parsed.intent == "OS_CONFIRMATION":
        token = parsed.args.get("token")
        second_factor = parsed.args.get("second_factor")
        ok, message, payload = confirmation_manager.confirm_with_second_factor(token, second_factor)
        if not ok:
            if "Second factor required" in message and token:
                return (
                    False,
                    render_template(
                        "confirmation_failed_with_usage",
                        language,
                        message=message,
                        token=token,
                    ),
                    {},
                )
            return False, render_template("confirmation_failed", language, message=message), {}
        return _execute_confirmed_payload(payload)

    if parsed.intent == "OS_ROLLBACK":
        ok, message = undo_last_action()
        return ok, message, {}

    if parsed.intent == "OS_FILE_SEARCH":
        filename = parsed.args.get("filename", "")
        if not filename:
            return False, render_template("missing_filename_search", language), {}
        root = parsed.args.get("search_path") or get_current_directory()
        search_index_service.start()
        indexed_results = search_index_service.search(filename, root=root)
        if indexed_results:
            if len(indexed_results) > 1:
                prompt, payload = _build_file_search_runtime_clarification(filename, indexed_results)
                return True, prompt, {"indexed_search": True, "clarification_payload": payload}
            return True, indexed_results[0], {"indexed_search": True}
        results = find_files(filename, search_path=parsed.args.get("search_path"))
        if len(results) > 1:
            prompt, payload = _build_file_search_runtime_clarification(filename, results)
            return True, prompt, {"indexed_search": False, "clarification_payload": payload}
        message = results[0] if results else render_template("file_not_found", language)
        return True, message, {"indexed_search": False}

    if parsed.intent == "OS_FILE_NAVIGATION":
        return file_navigation.handle(parsed)

    if parsed.intent == "OS_APP_OPEN":
        app_name = parsed.args.get("app_name", "")
        if not app_name:
            return False, render_template("missing_app_name_open", language), {}
        resolution = resolve_app_request(app_name, operation="open")
        if resolution.get("status") == "ambiguous":
            prompt, payload = _build_app_runtime_clarification(
                app_name,
                resolution.get("candidates") or [],
            )
            return True, prompt, {"clarification_payload": payload}
        return to_router_tuple(open_app_result(app_name))

    if parsed.intent == "OS_APP_CLOSE":
        app_name = parsed.args.get("app_name", "")
        if not app_name:
            return False, render_template("missing_app_name_close", language), {}
        resolution = resolve_app_request(app_name, operation="close")
        if resolution.get("status") == "ambiguous":
            prompt, payload = _build_app_runtime_clarification(
                app_name,
                resolution.get("candidates") or [],
                operation="close",
            )
            return True, prompt, {"clarification_payload": payload}
        return to_router_tuple(request_close_app_result(app_name))

    if parsed.intent == "OS_SYSTEM_COMMAND":
        action_key = parsed.args.get("action_key")
        return to_router_tuple(request_system_command_result(action_key, command_args=dict(parsed.args or {})))

    if parsed.intent == "METRICS_REPORT":
        return True, metrics.format_report(), {}

    if parsed.intent == "AUDIT_LOG_REPORT":
        limit = parsed.args.get("limit", 10)
        return True, audit.format_audit_log(limit), {}

    if parsed.intent == "AUDIT_VERIFY":
        return True, audit.format_audit_verify(), {}

    if parsed.intent == "AUDIT_RESEAL":
        return True, audit.format_audit_reseal(), {}

    if parsed.intent == "PERSONA_COMMAND":
        return persona.handle(parsed)

    if parsed.intent == "VOICE_COMMAND":
        return voice.handle(parsed)

    if parsed.intent == "KNOWLEDGE_BASE_COMMAND":
        return knowledge_base.handle(parsed)

    if parsed.intent == "MEMORY_COMMAND":
        return memory.handle(parsed)

    if parsed.intent == "OBSERVABILITY_REPORT":
        return True, metrics.format_observability_report(), {}

    if parsed.intent == "POLICY_COMMAND":
        return policy.handle(parsed)

    if parsed.intent == "BATCH_COMMAND":
        if not allow_batch:
            return False, "Nested batch commands are not allowed.", {}
        return batch.handle(parsed, parse_command, _execute_internal_command_text)

    if parsed.intent == "SEARCH_INDEX_COMMAND":
        return search_index.handle(parsed)

    if parsed.intent == "JOB_QUEUE_COMMAND":
        if not allow_job_queue:
            return False, "Nested job queue commands are not allowed.", {}
        _ensure_job_queue_executor()
        return job_queue_handler.handle(parsed)

    if not allow_llm:
        return False, "LLM fallback is disabled for this execution path.", {}

    # LLM fallback — use lightweight prompt for short, simple queries
    query_words = len((parsed.raw or "").split())
    has_memory_context = False
    has_recent_context_fn = getattr(type(session_memory), "has_recent_context", None)
    if callable(has_recent_context_fn):
        has_memory_context = bool(session_memory.has_recent_context(language=language, intents={"LLM_QUERY"}))
    else:
        has_memory_context = bool(
            session_memory.build_context(max_chars=1, language=language, intents={"LLM_QUERY"})
        )
    use_lightweight = query_words <= 8 and not has_memory_context
    if use_lightweight:
        package = build_lightweight_prompt(parsed.raw, response_language=language)
    else:
        package = build_prompt_package(parsed.raw, response_language=language)

    cache_eligible = (
        bool(LLM_RESPONSE_CACHE_ENABLED)
        and bool(use_lightweight)
        and int(query_words) <= max(1, int(LLM_RESPONSE_CACHE_MAX_QUERY_WORDS or 8))
        and not bool(package.get("kb_context_used"))
        and not bool(package.get("memory_used"))
    )

    cache_hit = False
    response = ""
    stream_callback = on_sentence
    if stream_callback and parsed.intent == "LLM_QUERY":
        stream_quality_repaired = False

        def _stream_callback(sentence):
            nonlocal stream_quality_repaired
            shaped = _apply_egyptian_dialect_style(sentence, parsed, language)
            if (
                not stream_quality_repaired
                and _looks_low_value_llm_reply(shaped)
                and _is_assist_first_safe_request(parsed.raw)
            ):
                fallback = _fallback_assist_first_response(parsed.raw, language)
                fallback = _apply_egyptian_dialect_style(fallback, parsed, language)
                if fallback:
                    shaped = fallback
                    stream_quality_repaired = True
            elif stream_quality_repaired and _looks_low_value_llm_reply(shaped):
                return

            stream_callback(shaped)

    else:
        _stream_callback = stream_callback

    if cache_eligible:
        response = str(_cache_get_llm_response(package["prompt"], language) or "").strip()
        cache_hit = bool(response)
        if cache_hit and _stream_callback:
            try:
                _stream_callback(response)
            except Exception:
                pass

    llm_num_ctx = int(LLM_LIGHTWEIGHT_NUM_CTX) if use_lightweight else None
    if not cache_hit:
        response = (
            ask_llm_streaming(
                package["prompt"],
                on_sentence=_stream_callback,
                num_ctx=llm_num_ctx,
            )
            or ""
        ).strip()
        if LLM_APPEND_SOURCE_CITATIONS and package["kb_sources"]:
            response += _format_source_citations(package["kb_sources"])
        if cache_eligible and response:
            _cache_put_llm_response(package["prompt"], language, response)

    return (
        True,
        response,
        {
            "persona": persona_manager.get_profile(),
            "kb_augmented": package["kb_context_used"],
            "kb_sources": len(package["kb_sources"]),
            "memory_used": package["memory_used"],
            "llm_lightweight": use_lightweight,
            "llm_cache_eligible": cache_eligible,
            "llm_cache_hit": cache_hit,
        },
    )


def _format_demo_output(parsed, success, message, meta):
    if not is_demo_mode_enabled() or parsed.intent == "DEMO_MODE":
        return message

    latest = read_recent_actions(limit=1)
    audit_row = latest[0] if latest else {}

    lines = [
        "[DEMO MODE]",
        "PLAN:",
        f"- intent: {parsed.intent}",
        f"- action: {parsed.action or 'n/a'}",
        f"- args: {parsed.args if parsed.args else '{}'}",
    ]
    if meta.get("language"):
        lines.append(f"- language: {meta.get('language')}")
    if meta.get("intent_confidence") is not None:
        lines.append(f"- intent_confidence: {float(meta.get('intent_confidence')):.2f}")
    if meta.get("nlu_used"):
        nlu_conf = float(meta.get("nlu_confidence") or 0.0)
        nlu_thr = float(meta.get("nlu_threshold") or NLU_INTENT_CONFIDENCE_THRESHOLD)
        nlu_status = "accepted" if meta.get("nlu_accepted") else "fallback"
        cache_tag = "hit" if meta.get("nlu_cache_hit") else "miss"
        lines.append(f"- nlu: {nlu_status} ({nlu_conf:.2f}/{nlu_thr:.2f}) cache={cache_tag}")
    elif meta.get("nlu_fastpath"):
        lines.append("- nlu: parser_fastpath")
    if meta.get("nlp_used"):
        nlp_conf = float(meta.get("nlp_confidence") or 0.0)
        nlp_intent = str(meta.get("nlp_intent") or "unknown")
        nlp_status = "accepted" if meta.get("nlp_accepted") else "fallback"
        lines.append(f"- nlp: {nlp_status} ({nlp_intent} {nlp_conf:.2f})")
    if meta.get("entity_scores"):
        lines.append(f"- entity_scores: {meta.get('entity_scores')}")
    if meta.get("clarification_resolved"):
        lines.append("- clarification: resolved")
    lines.extend(
        [
            "CONFIRM:",
            f"- required: {'yes' if meta.get('requires_confirmation') else 'no'}",
        ]
    )
    if meta.get("token"):
        lines.append(f"- token: {meta.get('token')}")
    if meta.get("second_factor"):
        lines.append("- second_factor: required")
    if meta.get("persona"):
        lines.append(f"- persona: {meta.get('persona')}")
    if meta.get("kb_augmented"):
        lines.append(f"- kb_sources: {meta.get('kb_sources', 0)}")
    if meta.get("memory_used"):
        lines.append("- memory: used")

    lines.extend(
        [
            "EXECUTE:",
            f"- status: {'success' if success else 'failed'}",
            f"- result: {message}",
            "AUDIT:",
        ]
    )
    if audit_row:
        lines.append(f"- id: {audit_row.get('id')}")
        lines.append(f"- action: {audit_row.get('action_type')} ({audit_row.get('status')})")
        lines.append(f"- hash: {audit_row.get('hash')}")
    else:
        lines.append("- no audit row found")
    return "\n".join(lines)


def _execute_followup_multi_actions(actions):
    responses = []
    any_success = False

    for item in list(actions or []):
        parsed = ParsedCommand(
            intent=str(item.get("intent") or "LLM_QUERY"),
            raw="",
            normalized="",
            action=str(item.get("action") or ""),
            args=dict(item.get("args") or {}),
        )
        success, response, dispatch_meta = _dispatch(parsed)
        meta = {"language": session_memory.get_preferred_language()}
        if dispatch_meta:
            meta.update(dispatch_meta)
        _update_short_term_context(parsed, success, response, meta)
        any_success = any_success or bool(success)
        if response:
            responses.append(str(response))

    joined = "\n".join([row for row in responses if row.strip()]).strip()
    if not joined:
        joined = "Multi-action follow-up completed." if any_success else "Multi-action follow-up failed."

    return any_success, joined


def route_command(
    text,
    detected_language=None,
    realtime=False,
    on_sentence=None,
    precomputed_language_result=None,
    precomputed_parser_candidate=None,
):
    original_text = text or ""
    start = time.perf_counter()
    forced_language = _normalize_supported_language_tag(detected_language)

    language_result = precomputed_language_result
    if language_result is None:
        language_result = detect_supported_language(
            original_text,
            previous_language=forced_language or session_memory.get_preferred_language(),
        )
    if forced_language:
        language_result = language_result.__class__(
            supported=True,
            language=forced_language,
            normalized_text=language_result.normalized_text or " ".join(str(original_text or "").split()),
            reason="stt_detected_language",
        )
    if not language_result.supported:
        latency = time.perf_counter() - start
        metrics.record_command("LANGUAGE_GATE_BLOCK", False, latency, language="unsupported")
        log_structured(
            "route_language_gate_block",
            level="warning",
            text=_truncate_text(original_text),
            reason=language_result.reason,
            latency_ms=latency * 1000.0,
        )
        log_action(
            "language_gate_block",
            "blocked",
            details={
                "text": original_text,
                "reason": language_result.reason,
            },
        )
        return UNSUPPORTED_LANGUAGE_MESSAGE

    effective_text = language_result.normalized_text or original_text
    session_memory.set_preferred_language(language_result.language)
    session_memory.record_language_turn(language_result.language)

    mode_toggle_message = _try_handle_response_mode_toggle(effective_text, language_result.language)
    if mode_toggle_message:
        latency = time.perf_counter() - start
        metrics.record_command("RESPONSE_MODE_COMMAND", True, latency, language=language_result.language)
        return mode_toggle_message

    tone_meta = _analyze_tone_markers(original_text, language=language_result.language)

    effective_text, followup_meta = _rewrite_followup_command(
        effective_text,
        language=language_result.language,
    )

    if followup_meta.get("followup_cancel_confirmation"):
        token = str(followup_meta.get("token") or "").strip().lower()
        ok, _cancel_message = confirmation_manager.cancel(token)
        session_memory.clear_pending_confirmation_token()
        if ok:
            return render_template("confirmation_cancelled", language_result.language)
        return render_template("missing_pending_confirmation", language_result.language)

    forced_parsed = None
    pending = session_memory.get_pending_clarification()
    if not pending:
        recent_resolution = session_memory.recent_clarification_resolution(
            max_age_seconds=CLARIFICATION_CORRECTION_WINDOW_SECONDS
        )
        if recent_resolution and _looks_like_post_clarification_correction(
            effective_text,
            language=language_result.language,
        ):
            metrics.record_clarification_event(
                "post_correction",
                intent=recent_resolution.get("intent") or "INTENT_CLARIFICATION",
                language=language_result.language,
                reason=recent_resolution.get("reason") or "clarification_correction",
                source_text=original_text,
            )

    if pending:
        pending_candidate = parse_command(effective_text)
        pending_candidate.raw = original_text
        if _should_bypass_pending_clarification(
            pending_candidate,
            pending_payload=pending,
            source_text=effective_text,
        ):
            session_memory.clear_pending_clarification()
            pending = None
            forced_parsed = pending_candidate

    if pending:
        resolution = resolve_clarification_reply(effective_text, pending)
        pending_reason = str(pending.get("reason") or "")
        pending_source_text = str(pending.get("source_text") or original_text)
        pending_intent = _clarification_intent_from_payload(pending)
        pending_attempts = int(pending.get("attempts") or 0)
        if resolution.status == "cancelled":
            session_memory.clear_pending_clarification()
            latency = time.perf_counter() - start
            metrics.record_command("INTENT_CLARIFICATION", True, latency, language=language_result.language)
            metrics.record_clarification_event(
                "cancelled",
                intent=pending_intent,
                language=language_result.language,
                reason=pending_reason,
                source_text=pending_source_text,
                retry_count=pending_attempts,
            )
            log_structured(
                "route_clarification_cancelled",
                language=language_result.language,
                latency_ms=latency * 1000.0,
                source_text=_truncate_text(pending.get("source_text") or original_text),
            )
            log_action(
                "intent_clarification_cancelled",
                "success",
                details={"source_text": pending.get("source_text"), "language": language_result.language},
            )
            return resolution.message or render_template("clarification_cancelled", language_result.language)

        if resolution.status == "next_page":
            updated_payload = dict(resolution.updated_payload or pending)
            updated_payload["attempts"] = pending_attempts
            updated_payload["fallback_hint_sent"] = bool(pending.get("fallback_hint_sent"))
            session_memory.set_pending_clarification(updated_payload)

            latency = time.perf_counter() - start
            metrics.record_command("INTENT_CLARIFICATION", False, latency, language=language_result.language)
            metrics.record_clarification_event(
                "reprompt",
                intent=pending_intent,
                language=language_result.language,
                reason=pending_reason,
                source_text=pending_source_text,
                retry_count=pending_attempts,
            )
            log_structured(
                "route_clarification_next_page",
                language=language_result.language,
                latency_ms=latency * 1000.0,
                source_text=_truncate_text(pending.get("source_text") or original_text),
                page_index=int(updated_payload.get("page_index") or 0),
            )
            return resolution.message or updated_payload.get("prompt") or pending.get("prompt") or render_template(
                "please_clarify_intent",
                language_result.language,
            )

        if resolution.status == "resolved":
            session_memory.clear_pending_clarification()
            option = resolution.option or {}
            session_memory.remember_clarification_choice(
                pending_reason,
                pending_source_text,
                option,
                language=language_result.language,
            )
            parsed = ParsedCommand(
                intent=option.get("intent", "LLM_QUERY"),
                raw=original_text,
                normalized=" ".join(effective_text.lower().split()).strip(),
                action=option.get("action", ""),
                args=dict(option.get("args") or {}),
            )
            success = False
            response = ""
            meta = {
                "language": language_result.language,
                "intent_confidence": pending.get("confidence"),
                "clarification_resolved": True,
                "entity_scores": pending.get("entity_scores") or {},
            }
            try:
                success, response, dispatch_meta = _dispatch(parsed)
                if dispatch_meta:
                    meta.update(dispatch_meta)
            except Exception as exc:
                logger.error("Command routing failed after clarification: %s", exc)
                response = "Sorry, I had an internal error."
                success = False

            _update_short_term_context(parsed, success, response, meta)
            session_memory.mark_clarification_resolution(reason=pending_reason, intent=parsed.intent)
            latency = time.perf_counter() - start
            metrics.record_command(parsed.intent, success, latency, language=language_result.language)
            metrics.record_clarification_event(
                "resolved" if success else "resolved_failed",
                intent=parsed.intent,
                language=language_result.language,
                reason=pending_reason,
                source_text=pending_source_text,
                retry_count=pending_attempts,
                wrong_action_prevented=_is_wrong_action_prevented_reason(pending_reason),
            )
            log_structured(
                "route_command_result",
                language=language_result.language,
                intent=parsed.intent,
                action=parsed.action or "",
                success=bool(success),
                latency_ms=latency * 1000.0,
                confidence=float(meta.get("intent_confidence") or 0.0),
                clarified=True,
                user_text=_truncate_text(original_text),
                response_preview=_truncate_text(response),
            )
            if success:
                response = _finalize_success_response(
                    response,
                    parsed,
                    language_result.language,
                    original_text,
                    tone_meta,
                    realtime=realtime,
                )
                if _should_store_turn(parsed, response):
                    session_memory.add_turn(
                        original_text,
                        response,
                        language=language_result.language,
                        intent=parsed.intent,
                    )
            return _format_demo_output(parsed, success, response, meta)

        if resolution.status in {"needs_clarification", "not_a_reply"}:
            pending["attempts"] = pending_attempts + 1
            fallback_after = max(1, int(CLARIFICATION_FALLBACK_AFTER_MISSES or 1))
            send_fallback_hint = pending["attempts"] >= fallback_after and not pending.get("fallback_hint_sent")
            pending["fallback_hint_sent"] = bool(pending.get("fallback_hint_sent") or send_fallback_hint)
            session_memory.set_pending_clarification(pending)

            latency = time.perf_counter() - start
            metrics.record_command("INTENT_CLARIFICATION", False, latency, language=language_result.language)
            metrics.record_clarification_event(
                "reprompt",
                intent=pending_intent,
                language=language_result.language,
                reason=pending_reason,
                source_text=pending_source_text,
                retry_count=int(pending.get("attempts") or 0),
            )
            log_structured(
                "route_clarification_reprompt",
                level="warning",
                language=language_result.language,
                latency_ms=latency * 1000.0,
                source_text=_truncate_text(pending.get("source_text") or original_text),
            )

            base_message = (
                resolution.message
                or pending.get("prompt")
                or render_template("please_clarify_intent", language_result.language)
            )
            if send_fallback_hint:
                if pending_reason == "low_confidence_unclear_query":
                    fallback_hint = render_template("clarification_retry_unclear_query", language_result.language)
                else:
                    fallback_hint = render_template("clarification_retry_with_examples", language_result.language)
                return f"{fallback_hint}\n{base_message}" if base_message else fallback_hint
            return base_message

        session_memory.clear_pending_clarification()

    if followup_meta.get("followup_multi_actions"):
        success, response = _execute_followup_multi_actions(followup_meta.get("followup_multi_actions"))
        latency = time.perf_counter() - start
        metrics.record_command("BATCH_COMMAND", success, latency, language=language_result.language)
        if success:
            parsed_for_memory = ParsedCommand(intent="BATCH_COMMAND", raw=original_text, normalized="", action="", args={})
            response = _finalize_success_response(
                response,
                parsed_for_memory,
                language_result.language,
                original_text,
                tone_meta,
                realtime=realtime,
            )
            if _should_store_turn(parsed_for_memory, response):
                session_memory.add_turn(
                    original_text,
                    response,
                    language=language_result.language,
                    intent=parsed_for_memory.intent,
                )
        return response

    if followup_meta.get("followup_blocked"):
        return str(followup_meta.get("followup_message") or "")

    parsed = forced_parsed
    parser_candidate = forced_parsed or precomputed_parser_candidate or parse_command(effective_text)
    parser_candidate.raw = original_text
    parser_fastpath_assessment = None
    nlu_meta = {
        "nlu_used": False,
        "nlu_accepted": False,
        "nlu_confidence": 0.0,
        "nlu_threshold": float(NLU_INTENT_CONFIDENCE_THRESHOLD),
        "nlu_cache_hit": False,
        "nlu_fastpath": False,
        "nlu_skipped_for_llm_query": False,
        "nlp_used": False,
        "nlp_accepted": False,
        "nlp_intent": "",
        "nlp_confidence": 0.0,
        "nlp_matched_keywords": [],
    }

    if parsed is None and NLU_INTENT_ROUTING_ENABLED:
        parser_fastpath_assessment = _select_parser_fastpath_assessment(
            original_text,
            parser_candidate,
            language_result.language,
        )
        if parser_fastpath_assessment is not None:
            parsed = parser_candidate
            nlu_meta["nlu_fastpath"] = True
        elif _should_skip_nlu_llm_query(parser_candidate):
            nlu_meta["nlu_skipped_for_llm_query"] = True
        else:
            nlu_result = classify_with_nlu(effective_text, language=language_result.language)
            nlu_intent = str(nlu_result.get("intent") or "")
            nlu_threshold = _nlu_threshold_for_intent(nlu_intent)
            nlu_meta["nlu_used"] = bool(nlu_result.get("ok"))
            nlu_meta["nlu_confidence"] = float(nlu_result.get("confidence") or 0.0)
            nlu_meta["nlu_threshold"] = float(nlu_threshold)
            nlu_meta["nlu_cache_hit"] = bool(nlu_result.get("cache_hit"))
            if (
                nlu_result.get("ok")
                and nlu_intent != "LLM_QUERY"
                and float(nlu_result.get("confidence") or 0.0) >= float(nlu_threshold)
            ):
                parsed = ParsedCommand(
                    intent=nlu_intent or "LLM_QUERY",
                    raw=original_text,
                    normalized=" ".join(effective_text.lower().split()).strip(),
                    action=str(nlu_result.get("action") or ""),
                    args=dict(nlu_result.get("args") or {}),
                )
                nlu_meta["nlu_accepted"] = True

    if parsed is None:
        keyword_nlp_parsed, keyword_nlp_meta = _try_keyword_nlp_routing(
            original_text,
            parser_candidate,
        )
        nlu_meta.update(keyword_nlp_meta)
        if keyword_nlp_parsed is not None:
            parsed = keyword_nlp_parsed

    if parsed is None:
        parsed = parser_candidate

    assessment = (
        parser_fastpath_assessment
        if nlu_meta.get("nlu_fastpath") and parser_fastpath_assessment is not None
        else assess_intent_confidence(original_text, parsed, language=language_result.language)
    )
    if assessment.should_clarify:
        clarification_payload = build_clarification_payload(
            assessment,
            source_text=original_text,
            language=language_result.language,
        )

        preferred_option = _find_preferred_clarification_option(clarification_payload)
        if preferred_option:
            parsed = ParsedCommand(
                intent=preferred_option.get("intent", "LLM_QUERY"),
                raw=original_text,
                normalized=" ".join(effective_text.lower().split()).strip(),
                action=preferred_option.get("action", ""),
                args=dict(preferred_option.get("args") or {}),
            )
            success = False
            response = ""
            meta = {
                "language": language_result.language,
                "intent_confidence": assessment.confidence,
                "clarification_resolved": True,
                "clarification_preference_used": True,
            }
            meta.update(nlu_meta)
            if assessment.entity_scores:
                meta["entity_scores"] = dict(assessment.entity_scores)
            if followup_meta:
                meta.update(followup_meta)

            try:
                success, response, dispatch_meta = _dispatch(parsed)
                if dispatch_meta:
                    meta.update(dispatch_meta)
            except Exception as exc:
                logger.error("Command routing failed after preference reuse: %s", exc)
                response = "Sorry, I had an internal error."
                success = False

            if meta.get("clarification_payload"):
                clarification_payload = dict(meta.get("clarification_payload") or {})
                session_memory.mark_clarification_reuse_feedback(
                    clarification_payload.get("reason") or assessment.reason,
                    clarification_payload.get("source_text") or original_text,
                    language=language_result.language,
                    success=False,
                )
                session_memory.set_pending_clarification(clarification_payload)
                latency = time.perf_counter() - start
                metrics.record_command("INTENT_CLARIFICATION", False, latency, language=language_result.language)
                metrics.record_clarification_event(
                    "requested",
                    intent=parsed.intent,
                    language=language_result.language,
                    reason=clarification_payload.get("reason") or assessment.reason,
                    source_text=clarification_payload.get("source_text") or original_text,
                    wrong_action_prevented=_is_wrong_action_prevented_reason(
                        clarification_payload.get("reason") or assessment.reason
                    ),
                )
                return clarification_payload.get("prompt") or response

            session_memory.remember_clarification_choice(
                clarification_payload.get("reason"),
                clarification_payload.get("source_text") or original_text,
                preferred_option,
                language=language_result.language,
            )
            session_memory.mark_clarification_reuse_feedback(
                clarification_payload.get("reason"),
                clarification_payload.get("source_text") or original_text,
                language=language_result.language,
                success=bool(success),
            )
            _update_short_term_context(parsed, success, response, meta)
            session_memory.mark_clarification_resolution(
                reason=clarification_payload.get("reason"),
                intent=parsed.intent,
            )
            latency = time.perf_counter() - start
            metrics.record_command(parsed.intent, success, latency, language=language_result.language)
            metrics.record_clarification_event(
                "resolved" if success else "resolved_failed",
                intent=parsed.intent,
                language=language_result.language,
                reason=clarification_payload.get("reason"),
                source_text=clarification_payload.get("source_text") or original_text,
                retry_count=0,
                wrong_action_prevented=_is_wrong_action_prevented_reason(clarification_payload.get("reason")),
            )
            log_structured(
                "route_clarification_preference_reused",
                language=language_result.language,
                intent=parsed.intent,
                action=parsed.action or "",
                success=bool(success),
                latency_ms=latency * 1000.0,
                reason=clarification_payload.get("reason"),
                user_text=_truncate_text(original_text),
                response_preview=_truncate_text(response),
            )
            if success:
                response = _finalize_success_response(
                    response,
                    parsed,
                    language_result.language,
                    original_text,
                    tone_meta,
                    realtime=realtime,
                )
                if _should_store_turn(parsed, response):
                    session_memory.add_turn(
                        original_text,
                        response,
                        language=language_result.language,
                        intent=parsed.intent,
                    )
            return _format_demo_output(parsed, success, response, meta)

        session_memory.set_pending_clarification(clarification_payload)
        latency = time.perf_counter() - start
        metrics.record_command("INTENT_CLARIFICATION", False, latency, language=language_result.language)
        metrics.record_clarification_event(
            "requested",
            intent=parsed.intent,
            language=language_result.language,
            reason=assessment.reason,
            source_text=clarification_payload.get("source_text") or original_text,
            wrong_action_prevented=_is_wrong_action_prevented_reason(assessment.reason),
        )
        log_structured(
            "route_clarification_requested",
            level="warning",
            language=language_result.language,
            intent=parsed.intent,
            action=parsed.action or "",
            confidence=float(assessment.confidence),
            reason=assessment.reason,
            latency_ms=latency * 1000.0,
            user_text=_truncate_text(original_text),
        )
        log_action(
            "intent_clarification_requested",
            "pending",
            details={
                "reason": assessment.reason,
                "intent": parsed.intent,
                "action": parsed.action,
                "confidence": assessment.confidence,
                "mixed_language": assessment.mixed_language,
                "source_text": original_text,
            },
        )
        return assessment.prompt

    success = False
    response = ""
    meta = {
        "language": language_result.language,
        "intent_confidence": assessment.confidence,
    }
    meta.update(nlu_meta)
    if followup_meta:
        meta.update(followup_meta)
    if assessment.entity_scores:
        meta["entity_scores"] = dict(assessment.entity_scores)

    try:
        success, response, dispatch_meta = _dispatch(parsed, on_sentence=on_sentence)
        if dispatch_meta:
            meta.update(dispatch_meta)
            if dispatch_meta.get("clarification_payload"):
                clarification_payload = dispatch_meta["clarification_payload"]
                preferred_option = _find_preferred_clarification_option(clarification_payload)
                if preferred_option:
                    preferred_parsed = ParsedCommand(
                        intent=preferred_option.get("intent", "LLM_QUERY"),
                        raw=original_text,
                        normalized=" ".join(effective_text.lower().split()).strip(),
                        action=preferred_option.get("action", ""),
                        args=dict(preferred_option.get("args") or {}),
                    )
                    success, response, preferred_meta = _dispatch(preferred_parsed, on_sentence=on_sentence)
                    parsed = preferred_parsed
                    meta["clarification_resolved"] = True
                    meta["clarification_preference_used"] = True
                    session_memory.remember_clarification_choice(
                        clarification_payload.get("reason"),
                        clarification_payload.get("source_text") or original_text,
                        preferred_option,
                        language=language_result.language,
                    )
                    if preferred_meta:
                        meta.update(preferred_meta)

                    nested_clarification = dict((preferred_meta or {}).get("clarification_payload") or {})
                    if nested_clarification:
                        session_memory.mark_clarification_reuse_feedback(
                            clarification_payload.get("reason"),
                            clarification_payload.get("source_text") or original_text,
                            language=language_result.language,
                            success=False,
                        )
                        clarification_payload = nested_clarification
                        session_memory.set_pending_clarification(clarification_payload)
                        latency = time.perf_counter() - start
                        metrics.record_command("INTENT_CLARIFICATION", False, latency, language=language_result.language)
                        metrics.record_clarification_event(
                            "requested",
                            intent=parsed.intent,
                            language=language_result.language,
                            reason=clarification_payload.get("reason", "runtime_disambiguation"),
                            source_text=clarification_payload.get("source_text") or original_text,
                            wrong_action_prevented=_is_wrong_action_prevented_reason(
                                clarification_payload.get("reason", "runtime_disambiguation")
                            ),
                        )
                        return clarification_payload.get("prompt") or response

                    session_memory.mark_clarification_reuse_feedback(
                        clarification_payload.get("reason"),
                        clarification_payload.get("source_text") or original_text,
                        language=language_result.language,
                        success=bool(success),
                    )
                    metrics.record_clarification_event(
                        "resolved" if success else "resolved_failed",
                        intent=parsed.intent,
                        language=language_result.language,
                        reason=clarification_payload.get("reason", "runtime_disambiguation"),
                        source_text=clarification_payload.get("source_text") or original_text,
                        retry_count=0,
                        wrong_action_prevented=_is_wrong_action_prevented_reason(
                            clarification_payload.get("reason", "runtime_disambiguation")
                        ),
                    )
                    session_memory.mark_clarification_resolution(
                        reason=clarification_payload.get("reason", "runtime_disambiguation"),
                        intent=parsed.intent,
                    )
                else:
                    session_memory.set_pending_clarification(clarification_payload)
                    latency = time.perf_counter() - start
                    metrics.record_command("INTENT_CLARIFICATION", False, latency, language=language_result.language)
                    metrics.record_clarification_event(
                        "requested",
                        intent=parsed.intent,
                        language=language_result.language,
                        reason=clarification_payload.get("reason", "runtime_disambiguation"),
                        source_text=clarification_payload.get("source_text") or original_text,
                        wrong_action_prevented=_is_wrong_action_prevented_reason(
                            clarification_payload.get("reason", "runtime_disambiguation")
                        ),
                    )
                    log_structured(
                        "route_runtime_clarification_requested",
                        level="warning",
                        language=language_result.language,
                        intent=parsed.intent,
                        action=parsed.action or "",
                        reason=clarification_payload.get("reason", "runtime_disambiguation"),
                        confidence=float(clarification_payload.get("confidence") or 0.0),
                        latency_ms=latency * 1000.0,
                        user_text=_truncate_text(original_text),
                    )
                    log_action(
                        "intent_clarification_requested",
                        "pending",
                        details={
                            "reason": clarification_payload.get("reason", "runtime_disambiguation"),
                            "intent": parsed.intent,
                            "action": parsed.action,
                            "confidence": clarification_payload.get("confidence"),
                            "source_text": original_text,
                        },
                    )
                    return clarification_payload.get("prompt") or response
    except Exception as exc:
        logger.error("Command routing failed: %s", exc)
        response = "Sorry, I had an internal error."
        success = False

    _update_short_term_context(parsed, success, response, meta)
    latency = time.perf_counter() - start
    metrics.record_command(parsed.intent, success, latency, language=language_result.language)
    log_structured(
        "route_command_result",
        language=language_result.language,
        intent=parsed.intent,
        action=parsed.action or "",
        success=bool(success),
        latency_ms=latency * 1000.0,
        confidence=float(meta.get("intent_confidence") or 0.0),
        clarified=bool(meta.get("clarification_resolved")),
        user_text=_truncate_text(original_text),
        response_preview=_truncate_text(response),
    )
    if success:
        response = _finalize_success_response(
            response,
            parsed,
            language_result.language,
            original_text,
            tone_meta,
            realtime=realtime,
        )
        if _should_store_turn(parsed, response):
            session_memory.add_turn(
                original_text,
                response,
                language=language_result.language,
                intent=parsed.intent,
            )
    return _format_demo_output(parsed, success, response, meta)


def initialize_command_services():
    voice.initialize_runtime_profiles()
    _ensure_job_queue_executor()
    job_queue_service.start()
    search_index_service.start()





