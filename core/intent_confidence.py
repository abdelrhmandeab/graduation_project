import difflib
import re
from dataclasses import dataclass, field

from core.config import (
    ENTITY_CLARIFICATION_LANGUAGE_ADJUSTMENT,
    ENTITY_CLARIFICATION_MIXED_LANGUAGE_BONUS,
    ENTITY_CLARIFICATION_THRESHOLD_BY_INTENT,
)
from os_control.app_ops import resolve_app_request
from core.response_templates import render_template


_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
_DRIVE_PATH_RE = re.compile(r"\b[a-z]:\\", flags=re.IGNORECASE)

_OPEN_VERBS = {
    "open",
    "show",
    "browse",
    "access",
    "enter",
    "\u0627\u0641\u062a\u062d",
    "\u0634\u063a\u0644",
    "\u0627\u0639\u0631\u0636",
    "\u0627\u0638\u0647\u0631",
    "\u062a\u0635\u0641\u062d",
}
_DELETE_VERBS = {"delete", "remove", "\u0627\u062d\u0630\u0641", "\u0627\u0645\u0633\u062d", "\u0627\u0632\u0644"}
_MOVE_VERBS = {"move", "\u0627\u0646\u0642\u0644", "\u062d\u0631\u0643"}
_RENAME_VERBS = {"rename", "\u0627\u0639\u062f \u062a\u0633\u0645\u064a\u0629", "\u063a\u064a\u0631 \u0627\u0633\u0645", "\u063a\u064a\u0651\u0631 \u0627\u0633\u0645"}
_SYSTEM_VERBS = {
    "shutdown",
    "restart",
    "sleep",
    "lock",
    "logoff",
    "\u0627\u0637\u0641\u064a",
    "\u0627\u063a\u0644\u0642 \u0627\u0644\u0643\u0645\u0628\u064a\u0648\u062a\u0631",
    "\u0627\u0639\u0627\u062f\u0629 \u062a\u0634\u063a\u064a\u0644",
    "\u0642\u0641\u0644",
    "\u0633\u062c\u0644 \u062e\u0631\u0648\u062c",
}
_FS_HINTS = {
    "drive",
    "partition",
    "folder",
    "directory",
    "file",
    "desktop",
    "downloads",
    "documents",
    "pictures",
    "music",
    "videos",
    "path",
    "\u0642\u0631\u0635",
    "\u0628\u0627\u0631\u062a\u0634\u0646",
    "\u0642\u0633\u0645",
    "\u062f\u0631\u0627\u064a\u0641",
    "\u0645\u062c\u0644\u062f",
    "\u0645\u0644\u0641",
    "\u0645\u0633\u0627\u0631",
    "\u0633\u0637\u062d \u0627\u0644\u0645\u0643\u062a\u0628",
    "\u0627\u0644\u062a\u062d\u0645\u064a\u0644\u0627\u062a",
    "\u0627\u0644\u062a\u0646\u0632\u064a\u0644\u0627\u062a",
    "\u0627\u0644\u0645\u0633\u062a\u0646\u062f\u0627\u062a",
}
_SPECIAL_FOLDER_ALIASES = {
    "desktop",
    "downloads",
    "documents",
    "pictures",
    "music",
    "videos",
    "\u0633\u0637\u062d \u0627\u0644\u0645\u0643\u062a\u0628",
    "\u0627\u0644\u062a\u062d\u0645\u064a\u0644\u0627\u062a",
    "\u0627\u0644\u062a\u0646\u0632\u064a\u0644\u0627\u062a",
    "\u0627\u0644\u0645\u0633\u062a\u0646\u062f\u0627\u062a",
    "\u0627\u0644\u0635\u0648\u0631",
    "\u0627\u0644\u0645\u0648\u0633\u064a\u0642\u0649",
    "\u0627\u0644\u0641\u064a\u062f\u064a\u0648\u0647\u0627\u062a",
}
_APP_EXPLICIT_HINTS = {"open app", "\u0627\u0641\u062a\u062d \u062a\u0637\u0628\u064a\u0642", "\u0634\u063a\u0644 \u062a\u0637\u0628\u064a\u0642"}
_QUESTION_HINTS = {
    "?",
    "\u061f",
    "what",
    "why",
    "how",
    "who",
    "where",
    "when",
    "\u0647\u0644",
    "\u0645\u0627",
    "\u0645\u0627\u0630\u0627",
    "\u0645\u0646",
    "\u0627\u064a\u0646",
    "\u0623\u064a\u0646",
    "\u0645\u062a\u0649",
    "\u0643\u064a\u0641",
    "\u0644\u0645\u0627\u0630\u0627",
}
_GREETING_HINTS = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "\u0645\u0631\u062d\u0628\u0627",
    "\u0623\u0647\u0644\u0627",
    "\u0627\u0647\u0644\u0627",
    "\u0633\u0644\u0627\u0645",
    "\u0634\u0643\u0631\u0627",
    "\u0634\u0643\u0631\u0627\u064b",
}
_CANCEL_REPLY_TOKENS = {
    "cancel",
    "stop",
    "never mind",
    "nevermind",
    "no thanks",
    "no thank you",
    "nah thanks",
    "abort",
    "\u0627\u0644\u063a\u0627\u0621",
    "\u0625\u0644\u063a\u0627\u0621",
    "\u062e\u0644\u0627\u0635",
    "\u0627\u062a\u0631\u0643\u0647",
    "\u0644\u0627 \u0634\u0643\u0631",
    "\u0644\u0623 \u0634\u0643\u0631",
    "\u0644\u0627 \u0634\u0643\u0631\u0627",
    "\u0644\u0623 \u0634\u0643\u0631\u0627",
}
_NEGATION_REPLY_TOKENS = {
    "not",
    "dont",
    "don't",
    "no",
    "nope",
    "isnt",
    "isn't",
    "مش",
    "مو",
    "ليس",
    "لا",
}
_ORDINAL_REPLY_TO_INDEX = {
    "first": 0,
    "1st": 0,
    "one": 0,
    "\u0627\u0644\u0627\u0648\u0644": 0,
    "\u0623\u0648\u0644": 0,
    "\u0627\u0648\u0644": 0,
    "second": 1,
    "2nd": 1,
    "two": 1,
    "\u0627\u0644\u062b\u0627\u0646\u064a": 1,
    "\u062b\u0627\u0646\u064a": 1,
    "\u0627\u0644\u062b\u0627\u0646\u064a\u0629": 1,
    "third": 2,
    "3rd": 2,
    "three": 2,
    "third one": 2,
    "third app": 2,
    "third option": 2,
    "\u0627\u0644\u062b\u0627\u0644\u062b": 2,
    "\u062b\u0627\u0644\u062b": 2,
}
_BINARY_YES_REPLY_TOKENS = {
    "yes",
    "yeah",
    "yep",
    "ok",
    "okay",
    "sure",
    "correct",
    "right",
    "affirmative",
    "نعم",
    "ايوه",
    "أيوه",
    "ايوا",
    "أيوة",
    "تمام",
    "صحيح",
}
_BINARY_NO_REPLY_TOKENS = {
    "no",
    "nope",
    "nah",
    "negative",
    "لا",
    "لأ",
    "مش",
    "مو",
    "غلط",
}
_BINARY_AMBIGUITY_REASONS = {
    "open_target_ambiguous",
    "app_name_ambiguous",
    "app_close_ambiguous",
    "file_search_multiple_matches",
}
_SHOW_MORE_REPLY_TOKENS = {
    "more",
    "show more",
    "next",
    "next page",
    "more options",
    "another page",
    "المزيد",
    "اكتر",
    "أكثر",
    "التالي",
    "صفحة تانية",
    "صفحة ثانية",
}
_NONE_OF_THESE_REPLY_TOKENS = {
    "none",
    "none of these",
    "none of them",
    "neither",
    "no option",
    "ولا واحد",
    "ولا وحدة",
    "ولا شي",
    "لا واحد منهم",
    "لا شيء",
}
_THIS_REFERENCE_TOKENS = {
    "this one",
    "this app",
    "this file",
    "هذا الخيار",
    "هذا التطبيق",
    "هذا الملف",
}
_THAT_REFERENCE_TOKENS = {
    "that one",
    "that app",
    "that file",
    "ذلك الخيار",
    "ذلك التطبيق",
    "ذلك الملف",
}


@dataclass
class ClarificationOption:
    id: str
    label: str
    intent: str
    action: str = ""
    args: dict = field(default_factory=dict)
    reply_tokens: list[str] = field(default_factory=list)


@dataclass
class IntentAssessment:
    confidence: float
    should_clarify: bool
    reason: str = ""
    prompt: str = ""
    options: list[ClarificationOption] = field(default_factory=list)
    mixed_language: bool = False
    entity_scores: dict = field(default_factory=dict)


@dataclass
class ClarificationResolution:
    status: str
    option: dict | None = None
    message: str = ""
    updated_payload: dict | None = None


def _normalized(text):
    return " ".join((text or "").lower().split()).strip()


def _with_confidence_prefix(prompt, confidence, language):
    base_prompt = str(prompt or "").strip()
    if not base_prompt:
        return base_prompt
    percent = int(round(max(0.0, min(1.0, float(confidence or 0.0))) * 100.0))
    confidence_line = render_template(
        "clarification_confidence_line",
        language,
        confidence_percent=percent,
    )
    confidence_line = str(confidence_line or "").strip()
    if not confidence_line:
        return base_prompt
    if base_prompt.startswith(confidence_line):
        return base_prompt
    return f"{confidence_line}\n{base_prompt}"


def _requests_show_more(reply_norm, reply_words):
    if reply_norm in _SHOW_MORE_REPLY_TOKENS:
        return True
    if any(token in _SHOW_MORE_REPLY_TOKENS for token in reply_words):
        return True
    return "show more" in reply_norm or "more options" in reply_norm or "المزيد" in reply_norm


def _requests_none_of_these(reply_norm, reply_words):
    if reply_norm in _NONE_OF_THESE_REPLY_TOKENS:
        return True
    if any(token in _NONE_OF_THESE_REPLY_TOKENS for token in reply_words):
        return True
    return "none of" in reply_norm or "ولا" in reply_norm


def _advance_clarification_page(pending_payload, current_options):
    payload = dict(pending_payload or {})
    all_options = list(payload.get("all_options") or [])
    if not all_options:
        return None, ""

    page_size = max(1, int(payload.get("page_size") or len(current_options or []) or 1))
    page_index = max(0, int(payload.get("page_index") or 0))
    total_pages = max(1, (len(all_options) + page_size - 1) // page_size)
    if page_index + 1 >= total_pages:
        return None, ""

    next_index = page_index + 1
    start = next_index * page_size
    end = start + page_size
    next_options = all_options[start:end]

    page_prompts = list(payload.get("page_prompts") or [])
    if page_prompts and next_index < len(page_prompts):
        next_prompt = str(page_prompts[next_index] or "").strip()
    else:
        language = str(payload.get("language") or "en")
        intro = str(payload.get("prompt_intro") or "").strip()
        lines = [intro] if intro else []
        for idx, option in enumerate(next_options, start=1):
            lines.append(f"{idx}) {option.get('label')}")
        lines.append(render_template("reply_with_number_or_cancel", language))
        next_prompt = "\n".join(lines)

    updated_payload = dict(payload)
    updated_payload["options"] = next_options
    updated_payload["page_index"] = next_index
    updated_payload["prompt"] = next_prompt
    return updated_payload, next_prompt


def _best_fuzzy_option_index(reply_norm, options, option_tokens):
    query = _normalized(reply_norm)
    if len(query) < 3:
        return None

    query = re.sub(r"\b(?:the|option|app|file|please|kindly|show|open|close|choose|one)\b", " ", query)
    query = " ".join(query.split())
    if len(query) < 3:
        return None
    if len(query.split()) > 4:
        return None
    if len(query) > 36:
        return None

    best_idx = None
    best_score = 0.0
    second_score = 0.0
    for idx, option in enumerate(options):
        candidates = []
        label = _normalized(option.get("label", ""))
        if label:
            candidates.append(label)
        for token in option_tokens[idx]:
            if len(token) >= 3 and not token.isdigit():
                candidates.append(token)

        local_best = 0.0
        for candidate in candidates:
            score = difflib.SequenceMatcher(a=query, b=candidate).ratio()
            if score > local_best:
                local_best = score

        if local_best > best_score:
            second_score = best_score
            best_score = local_best
            best_idx = idx
        elif local_best > second_score:
            second_score = local_best

    if best_idx is None:
        return None
    if best_score < 0.80:
        return None
    if (best_score - second_score) < 0.08:
        return None
    return best_idx


def _script_profile(text):
    arabic = 0
    latin = 0
    for char in text or "":
        if _ARABIC_CHAR_RE.match(char):
            arabic += 1
        elif _LATIN_CHAR_RE.match(char):
            latin += 1
    return {"arabic": arabic, "latin": latin, "mixed": arabic > 0 and latin > 0}


def _extract_open_target(raw_text):
    match = re.match(
        r"^(?:open|\u0627\u0641\u062a\u062d|\u0634\u063a\u0644)\s+(.+)$",
        (raw_text or "").strip(),
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(1).strip()
    return ""


def _looks_like_path_or_fs(text):
    lowered = _normalized(text)
    if not lowered:
        return False
    if "\\" in lowered or "/" in lowered:
        return True
    if _DRIVE_PATH_RE.search(lowered):
        return True
    if any(hint in lowered for hint in _FS_HINTS):
        return True
    return False


def _has_explicit_app_intent(text):
    lowered = _normalized(text)
    return any(hint in lowered for hint in _APP_EXPLICIT_HINTS)


def _looks_action_oriented(text):
    lowered = _normalized(text)
    buckets = (_OPEN_VERBS, _DELETE_VERBS, _MOVE_VERBS, _RENAME_VERBS, _SYSTEM_VERBS)
    return any(any(token in lowered for token in bucket) for bucket in buckets)


def _looks_brief_unclear_query(text):
    lowered = _normalized(text)
    if not lowered:
        return False

    words = lowered.split()
    if len(words) > 3:
        return False

    if any(hint in lowered for hint in _QUESTION_HINTS):
        return False

    if lowered in _GREETING_HINTS or any(token in _GREETING_HINTS for token in words):
        return False

    if _looks_action_oriented(lowered):
        return False

    if _looks_like_path_or_fs(lowered):
        return False

    return len(lowered) <= 24


def _multiple_action_categories(text):
    lowered = _normalized(text)
    categories = 0
    for bucket in (_OPEN_VERBS, _DELETE_VERBS, _MOVE_VERBS, _RENAME_VERBS, _SYSTEM_VERBS):
        if any(token in lowered for token in bucket):
            categories += 1
    return categories >= 2


def _entity_score_for_value(value):
    text = str(value or "").strip()
    if not text:
        return 0.0
    lowered = _normalized(text)
    if _DRIVE_PATH_RE.search(lowered) or "\\" in lowered or "/" in lowered:
        return 0.95
    if "." in lowered and len(lowered.split(".")) >= 2:
        return 0.88
    if len(lowered) >= 4:
        return 0.72
    return 0.58


def _app_entity_score(app_name, operation="open"):
    target = str(app_name or "").strip()
    if not target:
        return 0.0

    resolution = resolve_app_request(target, operation=operation)
    status = str(resolution.get("status") or "")
    if status == "exact":
        return 0.98
    if status == "high_confidence":
        return 0.92
    if status == "likely":
        return 0.80
    if status == "ambiguous":
        return 0.45
    return max(0.35, _entity_score_for_value(target) - 0.15)


def _entity_clarification_threshold(parsed, language="en", mixed_language=False):
    base = float(ENTITY_CLARIFICATION_THRESHOLD_BY_INTENT.get(parsed.intent, 0.0) or 0.0)
    if parsed.intent == "JOB_QUEUE_COMMAND" and parsed.action != "enqueue":
        return 0.0
    if parsed.intent == "OS_FILE_NAVIGATION" and parsed.action not in {
        "cd",
        "list_directory",
        "file_info",
        "create_directory",
        "delete_item",
        "delete_item_permanent",
        "move_item",
        "rename_item",
    }:
        base = max(0.0, base - 0.06)

    language_key = str(language or "en").strip().lower()
    adjustment = float(ENTITY_CLARIFICATION_LANGUAGE_ADJUSTMENT.get(language_key, 0.0) or 0.0)
    if mixed_language:
        adjustment += float(ENTITY_CLARIFICATION_MIXED_LANGUAGE_BONUS)
    return min(0.95, max(0.0, base + adjustment))


def _compute_entity_scores(parsed):
    args = dict(parsed.args or {})
    scores = {}
    if parsed.intent in {"OS_APP_OPEN", "OS_APP_CLOSE"}:
        operation = "close" if parsed.intent == "OS_APP_CLOSE" else "open"
        scores["app_name"] = _app_entity_score(args.get("app_name"), operation=operation)
    elif parsed.intent == "OS_FILE_SEARCH":
        scores["filename"] = _entity_score_for_value(args.get("filename"))
        if args.get("search_path"):
            scores["search_path"] = _entity_score_for_value(args.get("search_path"))
    elif parsed.intent == "OS_FILE_NAVIGATION":
        if parsed.action in {"cd", "list_directory", "file_info", "create_directory", "delete_item"}:
            scores["path"] = _entity_score_for_value(args.get("path"))
        elif parsed.action == "move_item":
            scores["source"] = _entity_score_for_value(args.get("source"))
            scores["destination"] = _entity_score_for_value(args.get("destination"))
        elif parsed.action == "rename_item":
            scores["source"] = _entity_score_for_value(args.get("source"))
            scores["new_name"] = _entity_score_for_value(args.get("new_name"))
    elif parsed.intent == "OS_SYSTEM_COMMAND":
        scores["action_key"] = 0.96 if args.get("action_key") else 0.0
        action_key = str(args.get("action_key") or "").strip().lower()
        if action_key == "volume_set":
            scores["volume_level"] = _entity_score_for_value(args.get("volume_level"))
        elif action_key == "brightness_set":
            scores["brightness_level"] = _entity_score_for_value(args.get("brightness_level"))
        elif action_key == "focus_window":
            scores["window_query"] = _entity_score_for_value(args.get("window_query"))
        elif action_key in {"media_seek_forward", "media_seek_backward"}:
            scores["seek_seconds"] = _entity_score_for_value(args.get("seek_seconds"))
        elif action_key == "browser_open_url":
            scores["url"] = _entity_score_for_value(args.get("url"))
        elif action_key == "browser_search_web":
            scores["search_query"] = _entity_score_for_value(args.get("search_query"))
    elif parsed.intent == "JOB_QUEUE_COMMAND":
        if parsed.action == "enqueue":
            scores["command_text"] = _entity_score_for_value(args.get("command_text"))
            if "delay_seconds" in args:
                scores["delay_seconds"] = _entity_score_for_value(args.get("delay_seconds"))
    return scores


def _min_entity_score(entity_scores):
    if not entity_scores:
        return 1.0
    return min(float(v) for v in entity_scores.values())


def _build_open_path_vs_app_disambiguation(target, prefer_app=False, language="en"):
    app_label = render_template("open_target_option_app", language, target=target)
    path_label = render_template("open_target_option_path", language, target=target)
    app_option = ClarificationOption(
        id="open_app",
        label=app_label,
        intent="OS_APP_OPEN",
        action="",
        args={"app_name": target},
        reply_tokens=["1", "app", "application", "program", "\u062a\u0637\u0628\u064a\u0642", "\u0628\u0631\u0646\u0627\u0645\u062c"],
    )
    path_option = ClarificationOption(
        id="open_path",
        label=path_label,
        intent="OS_FILE_NAVIGATION",
        action="list_directory",
        args={"path": target},
        reply_tokens=["2", "folder", "directory", "path", "file", "\u0645\u062c\u0644\u062f", "\u0645\u0644\u0641", "\u0645\u0633\u0627\u0631"],
    )
    options = [app_option, path_option] if prefer_app else [path_option, app_option]
    return options, (
        render_template("open_target_ambiguous_intro", language)
        + "\n"
        f"1) {options[0].label}\n"
        f"2) {options[1].label}\n"
        + render_template("reply_with_1_2_app_folder_or_cancel", language)
    )


def _build_app_candidate_disambiguation(query, candidates, language="en"):
    options = []
    lines = [render_template("app_ambiguous_open_intro", language)]
    for index, candidate in enumerate(candidates[:3], start=1):
        canonical_name = candidate.get("canonical_name") or candidate.get("executable")
        executable = candidate.get("executable")
        label = f"{canonical_name} ({executable})"
        options.append(
            ClarificationOption(
                id=f"open_app_{index}",
                label=label,
                intent="OS_APP_OPEN",
                action="",
                args={"app_name": executable},
                reply_tokens=[
                    str(index),
                    _normalized(canonical_name),
                    _normalized(executable),
                    "app",
                    "\u062a\u0637\u0628\u064a\u0642",
                ],
            )
        )
        lines.append(f"{index}) {label}")
    lines.append(render_template("reply_with_number_or_cancel", language))
    return options, "\n".join(lines)


def assess_intent_confidence(raw_text, parsed, language="en"):
    profile = _script_profile(raw_text)
    mixed_language = profile["mixed"]
    entity_scores = _compute_entity_scores(parsed)
    confidence = 0.82

    if parsed.intent == "LLM_QUERY":
        confidence = 0.36
        if _looks_action_oriented(raw_text):
            return IntentAssessment(
                confidence=confidence,
                should_clarify=True,
                reason="low_confidence_action_like_query",
                prompt=render_template("low_confidence_action_like_query", language),
                mixed_language=mixed_language,
                entity_scores=entity_scores,
            )
        if _looks_brief_unclear_query(raw_text):
            return IntentAssessment(
                confidence=confidence,
                should_clarify=True,
                reason="low_confidence_unclear_query",
                prompt=render_template("low_confidence_unclear_query", language),
                mixed_language=mixed_language,
                entity_scores=entity_scores,
            )
        return IntentAssessment(
            confidence=confidence,
            should_clarify=False,
            mixed_language=mixed_language,
            entity_scores=entity_scores,
        )

    if parsed.intent == "OS_FILE_NAVIGATION":
        if parsed.action in {"create_directory", "file_info"}:
            confidence = 0.92
        elif parsed.action in {"delete_item", "move_item", "rename_item"}:
            confidence = 0.90
        elif parsed.action == "list_directory":
            confidence = 0.78
        else:
            confidence = 0.85
    elif parsed.intent == "OS_APP_OPEN":
        confidence = 0.80
    elif parsed.intent == "OS_APP_CLOSE":
        confidence = 0.80
    elif parsed.intent == "OS_SYSTEM_COMMAND":
        confidence = 0.88
    elif parsed.intent == "OS_FILE_SEARCH":
        confidence = 0.86
    elif parsed.intent == "JOB_QUEUE_COMMAND":
        confidence = 0.84

    if mixed_language:
        confidence -= 0.08

    confidence = min(confidence, max(0.40, _min_entity_score(entity_scores) + 0.10))

    if _multiple_action_categories(raw_text):
        confidence -= 0.18
        return IntentAssessment(
            confidence=max(0.0, confidence),
            should_clarify=True,
            reason="multiple_actions_detected",
            prompt=render_template("multiple_actions_detected", language),
            mixed_language=mixed_language,
            entity_scores=entity_scores,
        )

    if parsed.intent == "OS_APP_OPEN":
        app_name = (parsed.args or {}).get("app_name", "").strip()
        if app_name:
            resolution = resolve_app_request(app_name)
            if resolution.get("status") == "ambiguous":
                options, prompt = _build_app_candidate_disambiguation(
                    app_name,
                    resolution.get("candidates") or [],
                    language=language,
                )
                return IntentAssessment(
                    confidence=0.58,
                    should_clarify=True,
                    reason="app_name_ambiguous",
                    prompt=prompt,
                    options=options,
                    mixed_language=mixed_language,
                    entity_scores=entity_scores,
                )

            looks_fs = _looks_like_path_or_fs(app_name)
            if (
                resolution.get("status") == "none"
                and not _has_explicit_app_intent(raw_text)
                and looks_fs
            ):
                options, prompt = _build_open_path_vs_app_disambiguation(
                    app_name,
                    prefer_app=True,
                    language=language,
                )
                return IntentAssessment(
                    confidence=0.48,
                    should_clarify=True,
                    reason="open_target_ambiguous",
                    prompt=prompt,
                    options=options,
                    mixed_language=mixed_language,
                    entity_scores=entity_scores,
                )

    if parsed.intent == "OS_FILE_NAVIGATION" and parsed.action == "list_directory":
        path_value = ((parsed.args or {}).get("path") or "").strip()
        if not path_value:
            path_value = _extract_open_target(raw_text)
        lowered_path = _normalized(path_value)
        if path_value:
            if lowered_path in _SPECIAL_FOLDER_ALIASES:
                return IntentAssessment(
                    confidence=max(0.70, confidence),
                    should_clarify=False,
                    mixed_language=mixed_language,
                    entity_scores=entity_scores,
                )
            plain_name = (
                "\\" not in path_value
                and "/" not in path_value
                and not _DRIVE_PATH_RE.search(path_value)
                and not any(hint in lowered_path for hint in _FS_HINTS)
            )
            if plain_name:
                options, prompt = _build_open_path_vs_app_disambiguation(
                    path_value,
                    prefer_app=False,
                    language=language,
                )
                return IntentAssessment(
                    confidence=0.50,
                    should_clarify=True,
                    reason="open_target_ambiguous",
                    prompt=prompt,
                    options=options,
                    mixed_language=mixed_language,
                    entity_scores=entity_scores,
                )

    entity_threshold = _entity_clarification_threshold(
        parsed,
        language=language,
        mixed_language=mixed_language,
    )
    if entity_scores and entity_threshold > 0.0:
        min_score = _min_entity_score(entity_scores)
        if min_score < entity_threshold:
            return IntentAssessment(
                confidence=max(0.0, min(confidence, min_score + 0.05)),
                should_clarify=True,
                reason="low_entity_confidence",
                prompt=render_template("low_entity_confidence", language),
                mixed_language=mixed_language,
                entity_scores=entity_scores,
            )

    return IntentAssessment(
        confidence=max(0.0, min(1.0, confidence)),
        should_clarify=False,
        mixed_language=mixed_language,
        entity_scores=entity_scores,
    )


def build_clarification_payload(assessment, source_text, language):
    prompt = _with_confidence_prefix(assessment.prompt, assessment.confidence, language)
    return {
        "reason": assessment.reason,
        "prompt": prompt,
        "options": [
            {
                "id": option.id,
                "label": option.label,
                "intent": option.intent,
                "action": option.action,
                "args": dict(option.args or {}),
                "reply_tokens": list(option.reply_tokens or []),
            }
            for option in assessment.options
        ],
        "source_text": source_text,
        "language": language,
        "confidence": float(assessment.confidence),
        "entity_scores": dict(assessment.entity_scores or {}),
    }


def resolve_clarification_reply(reply_text, pending_payload):
    reply_norm = _normalized(reply_text)
    if not reply_norm:
        return ClarificationResolution(
            status="needs_clarification",
            message=pending_payload.get("prompt") or "",
        )

    language = str((pending_payload or {}).get("language") or "en")
    reply_words = set(re.findall(r"[a-z0-9\u0600-\u06FF]+", reply_norm))

    if any(token in reply_norm for token in _CANCEL_REPLY_TOKENS):
        return ClarificationResolution(
            status="cancelled",
            message=render_template("clarification_cancelled", language),
        )

    if _requests_none_of_these(reply_norm, reply_words):
        return ClarificationResolution(
            status="cancelled",
            message=render_template("clarification_none_selected", language),
        )

    options = list((pending_payload or {}).get("options") or [])
    if not options:
        return ClarificationResolution(status="not_a_reply")

    if _requests_show_more(reply_norm, reply_words):
        updated_payload, next_prompt = _advance_clarification_page(pending_payload, options)
        if updated_payload and next_prompt:
            return ClarificationResolution(
                status="next_page",
                message=next_prompt,
                updated_payload=updated_payload,
            )
        no_more_message = render_template("no_more_options", language)
        base_prompt = str((pending_payload or {}).get("prompt") or "").strip()
        if base_prompt:
            return ClarificationResolution(status="needs_clarification", message=f"{no_more_message}\n{base_prompt}")
        return ClarificationResolution(status="needs_clarification", message=no_more_message)

    all_options = list((pending_payload or {}).get("all_options") or [])
    searchable_options = all_options if all_options else options

    reason = _normalized((pending_payload or {}).get("reason") or "")
    has_explicit_selection_hint = bool(re.search(r"\d", reply_norm))
    if not has_explicit_selection_hint:
        has_explicit_selection_hint = any(word in _ORDINAL_REPLY_TO_INDEX for word in reply_words)

    explicit_number_match = re.search(r"\d+", reply_norm)
    if explicit_number_match:
        index = int(explicit_number_match.group(0)) - 1
        if 0 <= index < len(options):
            return ClarificationResolution(status="resolved", option=options[index])
        if 0 <= index < len(searchable_options):
            return ClarificationResolution(status="resolved", option=searchable_options[index])

    if reason in _BINARY_AMBIGUITY_REASONS and options and not has_explicit_selection_hint:
        if reply_words.intersection(_BINARY_YES_REPLY_TOKENS):
            return ClarificationResolution(status="resolved", option=options[0])
        if reply_words.intersection(_BINARY_NO_REPLY_TOKENS):
            if len(options) >= 2:
                return ClarificationResolution(status="resolved", option=options[1])
            return ClarificationResolution(
                status="needs_clarification",
                message=pending_payload.get("prompt") or "",
            )

    if any(token in reply_norm for token in _THIS_REFERENCE_TOKENS):
        return ClarificationResolution(status="resolved", option=options[0])
    if len(options) >= 2 and any(token in reply_norm for token in _THAT_REFERENCE_TOKENS):
        return ClarificationResolution(status="resolved", option=options[1])

    def _option_tokens(option):
        token_pool = set()
        option_id = _normalized(option.get("id", ""))
        label = _normalized(option.get("label", ""))
        if option_id:
            token_pool.add(option_id)
            token_pool.update(option_id.split())
        if label:
            token_pool.add(label)
            token_pool.update(label.split())
        for token in option.get("reply_tokens") or []:
            token_norm = _normalized(token)
            if token_norm:
                token_pool.add(token_norm)
                token_pool.update(token_norm.split())
        return {tok for tok in token_pool if tok}

    option_tokens = [_option_tokens(option) for option in searchable_options]

    numeric_match = re.match(r"^\d+$", reply_norm)
    if numeric_match:
        index = int(reply_norm) - 1
        if 0 <= index < len(options):
            return ClarificationResolution(status="resolved", option=options[index])
        if 0 <= index < len(searchable_options):
            return ClarificationResolution(status="resolved", option=searchable_options[index])

    ordinal_index = _ORDINAL_REPLY_TO_INDEX.get(reply_norm)
    if ordinal_index is not None and 0 <= ordinal_index < len(options):
        return ClarificationResolution(status="resolved", option=options[ordinal_index])

    for word in reply_words:
        ordinal_index = _ORDINAL_REPLY_TO_INDEX.get(word)
        if ordinal_index is not None and 0 <= ordinal_index < len(options):
            return ClarificationResolution(status="resolved", option=options[ordinal_index])

    phrase_match = re.search(
        r"(?:the\s+)?([a-z0-9\u0600-\u06FF\s._-]+?)\s+(?:one|app|option|\u062a\u0637\u0628\u064a\u0642|\u062e\u064a\u0627\u0631)$",
        reply_norm,
    )
    if phrase_match:
        descriptor = _normalized(phrase_match.group(1))
        if descriptor:
            descriptor_words = set(descriptor.split())
            for idx, tokens in enumerate(option_tokens):
                if descriptor in tokens or descriptor_words.intersection(tokens):
                    return ClarificationResolution(status="resolved", option=searchable_options[idx])

    negation_present = any(token in reply_words for token in _NEGATION_REPLY_TOKENS)
    if negation_present:
        excluded_indexes = set()
        for idx, tokens in enumerate(option_tokens[: len(options)]):
            if any(token in reply_norm for token in tokens):
                excluded_indexes.add(idx)
        if len(options) == 2 and len(excluded_indexes) == 1:
            selected = 1 if 0 in excluded_indexes else 0
            return ClarificationResolution(status="resolved", option=options[selected])

    fuzzy_index = _best_fuzzy_option_index(reply_norm, searchable_options, option_tokens)
    if fuzzy_index is not None and 0 <= fuzzy_index < len(searchable_options):
        return ClarificationResolution(status="resolved", option=searchable_options[fuzzy_index])

    for index, option in enumerate(options, start=1):
        if reply_norm == str(index):
            return ClarificationResolution(status="resolved", option=option)

        option_id = _normalized(option.get("id", ""))
        if option_id and option_id in reply_words:
            return ClarificationResolution(status="resolved", option=option)

        for token in option.get("reply_tokens") or []:
            token_norm = _normalized(token)
            if token_norm and (token_norm in reply_words or token_norm in reply_norm):
                return ClarificationResolution(status="resolved", option=option)

        label_norm = _normalized(option.get("label", ""))
        if label_norm and label_norm in reply_norm:
            return ClarificationResolution(status="resolved", option=option)

    for option in searchable_options:
        option_id = _normalized(option.get("id", ""))
        if option_id and option_id in reply_norm:
            return ClarificationResolution(status="resolved", option=option)

        label_norm = _normalized(option.get("label", ""))
        if label_norm and label_norm in reply_norm:
            return ClarificationResolution(status="resolved", option=option)

        for token in option.get("reply_tokens") or []:
            token_norm = _normalized(token)
            if token_norm and token_norm in reply_norm:
                return ClarificationResolution(status="resolved", option=option)

    if len(reply_words) <= 5:
        return ClarificationResolution(
            status="needs_clarification",
            message=pending_payload.get("prompt") or "",
        )
    return ClarificationResolution(status="not_a_reply")
