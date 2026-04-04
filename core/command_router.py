import os
import re
import time

from core.command_parser import ParsedCommand, parse_command
from core.config import LLM_APPEND_SOURCE_CITATIONS
from core.demo_mode import is_enabled as is_demo_mode_enabled
from core.demo_mode import set_enabled as set_demo_mode
from core.handlers import audit, batch, benchmark, file_navigation
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
from core.response_templates import anti_repetition_prefixes, detect_language_hint, render_template
from core.session_memory import session_memory
from llm.ollama_client import ask_llm
from llm.prompt_builder import build_prompt_package
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
    "اغلقه",
    "اغلقها",
    "اقفله",
    "اقفلها",
    "سكره",
    "سكرها",
}

_DELETE_FOLLOWUP_TEXTS = {
    "delete it",
    "delete this",
    "delete that",
    "remove it",
    "remove this",
    "احذفه",
    "احذفها",
    "امسحه",
    "امسحها",
    "ازله",
    "ازلها",
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
    "الغ",
    "الغاء",
    "إلغاء",
    "الغه",
    "ألغِه",
    "الغها",
    "ألغِها",
}

_RENAME_IT_TO_RE = re.compile(r"^\s*(?:rename|change\s+name)\s+(?:it|this|that)\s+to\s+(.+)$", re.IGNORECASE)
_MOVE_IT_TO_RE = re.compile(r"^\s*(?:move)\s+(?:it|this|that)\s+to\s+(.+)$", re.IGNORECASE)
_CONFIRM_IT_WITH_FACTOR_RE = re.compile(
    r"^\s*(?:confirm|approve)\s+(?:it|this|that)\s+(.+)$",
    re.IGNORECASE,
)
_AR_RENAME_IT_TO_RE = re.compile(
    r"^\s*(?:غيره|غيرها|غير\s+اسمه|غير\s+اسمها)\s+(?:الى|إلى)\s+(.+)$",
    re.IGNORECASE,
)
_AR_MOVE_IT_TO_RE = re.compile(
    r"^\s*(?:انقله|انقلها|حركه|حركها)\s+(?:الى|إلى)\s+(.+)$",
    re.IGNORECASE,
)
_AR_CONFIRM_IT_WITH_FACTOR_RE = re.compile(
    r"^\s*(?:اكدها|أكدها|اكده|أكده|اكد|أكد)\s+(.+)$",
    re.IGNORECASE,
)

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
    "BENCHMARK_COMMAND": "benchmark",
}


def _truncate_text(value, max_chars=180):
    text = " ".join(str(value or "").split())
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


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
        return to_router_tuple(execute_system_command_result(action_key))
    if kind == "file_operation":
        return to_router_tuple(execute_confirmed_file_operation(payload))
    if kind == "app_operation":
        return to_router_tuple(execute_confirmed_app_operation(payload))
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
        "BENCHMARK_COMMAND",
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

    if normalized in _CANCEL_FOLLOWUP_TEXTS and pending_token and not pending_clarification:
        return raw, {"followup_cancel_confirmation": True, "token": pending_token}

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

    rename_match = _RENAME_IT_TO_RE.match(raw) or _AR_RENAME_IT_TO_RE.match(raw)
    if rename_match:
        last_file = session_memory.get_last_file()
        if last_file:
            return (
                f"rename {last_file} to {rename_match.group(1).strip()}",
                {"followup_rewrite": "rename_last_file", "last_file": last_file},
            )
        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_last_file_rename", language),
        }

    move_match = _MOVE_IT_TO_RE.match(raw) or _AR_MOVE_IT_TO_RE.match(raw)
    if move_match:
        last_file = session_memory.get_last_file()
        if last_file:
            return (
                f"move {last_file} to {move_match.group(1).strip()}",
                {"followup_rewrite": "move_last_file", "last_file": last_file},
            )
        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_last_file_move", language),
        }

    if normalized in _DELETE_FOLLOWUP_TEXTS:
        last_file = session_memory.get_last_file()
        if last_file:
            return f"delete {last_file}", {"followup_rewrite": "delete_last_file", "last_file": last_file}
        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_last_file_delete", language),
        }

    if normalized in _OPEN_FOLLOWUP_TEXTS:
        last_file = session_memory.get_last_file()
        last_file_ts = session_memory.get_last_file_timestamp()
        last_app = session_memory.get_last_app()
        last_app_ts = session_memory.get_last_app_timestamp()

        candidates = []
        if last_file:
            if os.path.isdir(last_file):
                candidates.append((last_file_ts, f"open {last_file}", "open_last_file", {"last_file": last_file}))
            elif os.path.isfile(last_file):
                candidates.append((last_file_ts, f"file info {last_file}", "file_info_last_file", {"last_file": last_file}))
            else:
                candidates.append((last_file_ts, f"file info {last_file}", "file_info_last_file", {"last_file": last_file}))
        if last_app:
            candidates.append((last_app_ts, f"open app {last_app}", "open_last_app", {"last_app": last_app}))

        if candidates:
            _ts, rewritten, rewrite_name, extra_meta = max(candidates, key=lambda row: row[0])
            meta = {"followup_rewrite": rewrite_name}
            meta.update(extra_meta)
            return rewritten, meta

        return raw, {
            "followup_blocked": True,
            "followup_message": render_template("missing_last_app_open", language),
        }

    if normalized in _CLOSE_FOLLOWUP_TEXTS:
        last_app = session_memory.get_last_app()
        if last_app:
            return f"close app {last_app}", {"followup_rewrite": "close_last_app", "last_app": last_app}
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
        if operation in {"delete_item", "delete_item_permanent", "move_item", "rename_item", "create_directory", "file_info"}:
            candidate_path = str(meta.get("path") or meta.get("destination") or meta.get("source") or "").strip()
            if candidate_path:
                session_memory.set_last_file(candidate_path)


def _build_app_runtime_clarification(app_query, candidates, *, operation="open"):
    operation_mode = "close" if operation == "close" else "open"
    intent = "OS_APP_CLOSE" if operation_mode == "close" else "OS_APP_OPEN"
    option_prefix = "close_app_runtime" if operation_mode == "close" else "open_app_runtime"
    language = session_memory.get_preferred_language()

    options = []
    lines = [render_template(f"app_ambiguous_{operation_mode}_intro", language)]
    for index, candidate in enumerate(candidates[:3], start=1):
        canonical = candidate.get("canonical_name") or candidate.get("executable")
        executable = candidate.get("executable")
        label = f"{canonical} ({executable})"
        lines.append(f"{index}) {label}")
        options.append(
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
                    "app",
                    "\u062a\u0637\u0628\u064a\u0642",
                ],
            }
        )
    lines.append(render_template("reply_with_number_or_cancel", language))
    prompt = "\n".join(lines)
    payload = {
        "reason": "app_close_ambiguous" if operation_mode == "close" else "app_name_ambiguous",
        "prompt": prompt,
        "options": options,
        "source_text": app_query,
        "language": language,
        "confidence": 0.58,
        "entity_scores": {"app_name": 0.62},
    }
    return prompt, payload


def _build_file_search_runtime_clarification(filename, matches):
    language = session_memory.get_preferred_language()
    options = []
    lines = [render_template("file_ambiguous_intro", language, filename=filename)]
    for index, match in enumerate(matches[:5], start=1):
        lines.append(f"{index}) {match}")
        options.append(
            {
                "id": f"file_match_{index}",
                "label": match,
                "intent": "OS_FILE_NAVIGATION",
                "action": "file_info",
                "args": {"path": match},
                "reply_tokens": [str(index), str(match).lower()],
            }
        )
    lines.append(render_template("reply_with_number_or_cancel", language))
    prompt = "\n".join(lines)
    payload = {
        "reason": "file_search_multiple_matches",
        "prompt": prompt,
        "options": options,
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
        "BENCHMARK_COMMAND",
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


def _dispatch(parsed, *, allow_batch=True, allow_job_queue=True, allow_llm=True):
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
        resolution = resolve_app_request(app_name)
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
        resolution = resolve_app_request(app_name)
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
        return to_router_tuple(request_system_command_result(action_key))

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

    if parsed.intent == "BENCHMARK_COMMAND":
        return benchmark.handle(parsed, route_command)

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

    # LLM fallback
    package = build_prompt_package(parsed.raw)
    response = (ask_llm(package["prompt"]) or "").strip()
    if LLM_APPEND_SOURCE_CITATIONS and package["kb_sources"]:
        response += _format_source_citations(package["kb_sources"])
    return (
        True,
        response,
        {
            "persona": persona_manager.get_profile(),
            "kb_augmented": package["kb_context_used"],
            "kb_sources": len(package["kb_sources"]),
            "memory_used": package["memory_used"],
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


def route_command(text):
    original_text = text or ""
    start = time.perf_counter()

    language_result = detect_supported_language(
        original_text,
        previous_language=session_memory.get_preferred_language(),
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

    pending = session_memory.get_pending_clarification()
    if pending:
        resolution = resolve_clarification_reply(effective_text, pending)
        if resolution.status == "cancelled":
            session_memory.clear_pending_clarification()
            latency = time.perf_counter() - start
            metrics.record_command("INTENT_CLARIFICATION", True, latency, language=language_result.language)
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
            return render_template("clarification_cancelled", language_result.language)

        if resolution.status == "resolved":
            session_memory.clear_pending_clarification()
            option = resolution.option or {}
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
                clarified=True,
                user_text=_truncate_text(original_text),
                response_preview=_truncate_text(response),
            )
            if success:
                response = _apply_anti_repetition(response, language_result.language)
                if _should_store_turn(parsed, response):
                    session_memory.add_turn(original_text, response)
            return _format_demo_output(parsed, success, response, meta)

        if resolution.status == "needs_clarification":
            latency = time.perf_counter() - start
            metrics.record_command("INTENT_CLARIFICATION", False, latency, language=language_result.language)
            log_structured(
                "route_clarification_reprompt",
                level="warning",
                language=language_result.language,
                latency_ms=latency * 1000.0,
                source_text=_truncate_text(pending.get("source_text") or original_text),
            )
            return (
                resolution.message
                or pending.get("prompt")
                or render_template("please_clarify_intent", language_result.language)
            )

        session_memory.clear_pending_clarification()

    if followup_meta.get("followup_blocked"):
        return str(followup_meta.get("followup_message") or "")

    parsed = parse_command(effective_text)
    parsed.raw = original_text
    assessment = assess_intent_confidence(original_text, parsed, language=language_result.language)
    if assessment.should_clarify:
        clarification_payload = build_clarification_payload(
            assessment,
            source_text=original_text,
            language=language_result.language,
        )
        session_memory.set_pending_clarification(clarification_payload)
        latency = time.perf_counter() - start
        metrics.record_command("INTENT_CLARIFICATION", False, latency, language=language_result.language)
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
    meta = {"language": language_result.language, "intent_confidence": assessment.confidence}
    if followup_meta:
        meta.update(followup_meta)
    if assessment.entity_scores:
        meta["entity_scores"] = dict(assessment.entity_scores)

    try:
        success, response, dispatch_meta = _dispatch(parsed)
        if dispatch_meta:
            meta.update(dispatch_meta)
            if dispatch_meta.get("clarification_payload"):
                clarification_payload = dispatch_meta["clarification_payload"]
                session_memory.set_pending_clarification(clarification_payload)
                latency = time.perf_counter() - start
                metrics.record_command("INTENT_CLARIFICATION", False, latency, language=language_result.language)
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
        clarified=False,
        user_text=_truncate_text(original_text),
        response_preview=_truncate_text(response),
    )
    if success:
        response = _apply_anti_repetition(response, language_result.language)
        if _should_store_turn(parsed, response):
            session_memory.add_turn(original_text, response)
    return _format_demo_output(parsed, success, response, meta)


def initialize_command_services():
    voice.initialize_runtime_profiles()
    _ensure_job_queue_executor()
    job_queue_service.start()
    search_index_service.start()





