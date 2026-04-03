import re
from dataclasses import dataclass, field

from os_control.app_ops import resolve_app_request


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
_CANCEL_REPLY_TOKENS = {
    "cancel",
    "stop",
    "never mind",
    "nevermind",
    "abort",
    "\u0627\u0644\u063a\u0627\u0621",
    "\u0625\u0644\u063a\u0627\u0621",
    "\u062e\u0644\u0627\u0635",
    "\u0627\u062a\u0631\u0643\u0647",
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


def _normalized(text):
    return " ".join((text or "").lower().split()).strip()


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


def _multiple_action_categories(text):
    lowered = _normalized(text)
    categories = 0
    for bucket in (_OPEN_VERBS, _DELETE_VERBS, _MOVE_VERBS, _RENAME_VERBS, _SYSTEM_VERBS):
        if any(token in lowered for token in bucket):
            categories += 1
    return categories >= 2


def _entity_score_for_value(value):
    text = (value or "").strip()
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


def _compute_entity_scores(parsed):
    args = dict(parsed.args or {})
    scores = {}
    if parsed.intent == "OS_APP_OPEN":
        scores["app_name"] = _entity_score_for_value(args.get("app_name"))
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
    return scores


def _min_entity_score(entity_scores):
    if not entity_scores:
        return 1.0
    return min(float(v) for v in entity_scores.values())


def _build_open_path_vs_app_disambiguation(target, prefer_app=False):
    app_option = ClarificationOption(
        id="open_app",
        label=f"Open application '{target}'",
        intent="OS_APP_OPEN",
        action="",
        args={"app_name": target},
        reply_tokens=["1", "app", "application", "program", "\u062a\u0637\u0628\u064a\u0642", "\u0628\u0631\u0646\u0627\u0645\u062c"],
    )
    path_option = ClarificationOption(
        id="open_path",
        label=f"Open folder/path '{target}'",
        intent="OS_FILE_NAVIGATION",
        action="list_directory",
        args={"path": target},
        reply_tokens=["2", "folder", "directory", "path", "file", "\u0645\u062c\u0644\u062f", "\u0645\u0644\u0641", "\u0645\u0633\u0627\u0631"],
    )
    options = [app_option, path_option] if prefer_app else [path_option, app_option]
    return options, (
        "I want to confirm before acting. Did you mean:\n"
        f"1) {options[0].label}\n"
        f"2) {options[1].label}\n"
        "Reply with `1`, `2`, `app`, `folder`, or `cancel`."
    )


def _build_app_candidate_disambiguation(query, candidates):
    options = []
    lines = ["I found multiple app matches. Which one should I open?"]
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
    lines.append("Reply with the number (for example `1`) or `cancel`.")
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
                prompt=(
                    "I am not fully sure which action you want. "
                    "Please rephrase as one clear command (for example: "
                    "`open app notepad`, `find file notes.txt`, or `delete file.txt`)."
                ),
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
    elif parsed.intent == "OS_SYSTEM_COMMAND":
        confidence = 0.88
    elif parsed.intent == "OS_FILE_SEARCH":
        confidence = 0.86

    if mixed_language:
        confidence -= 0.08

    confidence = min(confidence, max(0.40, _min_entity_score(entity_scores) + 0.10))

    if _multiple_action_categories(raw_text):
        confidence -= 0.18
        return IntentAssessment(
            confidence=max(0.0, confidence),
            should_clarify=True,
            reason="multiple_actions_detected",
            prompt=(
                "I detected more than one action in one command. "
                "Please split it into one step at a time."
            ),
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
                options, prompt = _build_open_path_vs_app_disambiguation(app_name, prefer_app=True)
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
                options, prompt = _build_open_path_vs_app_disambiguation(path_value, prefer_app=False)
                return IntentAssessment(
                    confidence=0.50,
                    should_clarify=True,
                    reason="open_target_ambiguous",
                    prompt=prompt,
                    options=options,
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
    return {
        "reason": assessment.reason,
        "prompt": assessment.prompt,
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

    if any(token in reply_norm for token in _CANCEL_REPLY_TOKENS):
        return ClarificationResolution(
            status="cancelled",
            message="Clarification cancelled.",
        )

    options = list((pending_payload or {}).get("options") or [])
    if not options:
        return ClarificationResolution(status="not_a_reply")

    numeric_match = re.match(r"^\d+$", reply_norm)
    if numeric_match:
        index = int(reply_norm) - 1
        if 0 <= index < len(options):
            return ClarificationResolution(status="resolved", option=options[index])

    reply_words = set(reply_norm.split())
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

    if len(reply_words) <= 5:
        return ClarificationResolution(
            status="needs_clarification",
            message=pending_payload.get("prompt") or "",
        )
    return ClarificationResolution(status="not_a_reply")
