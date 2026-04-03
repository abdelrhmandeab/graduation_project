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
from core.logger import logger
from core.metrics import metrics
from core.persona import persona_manager
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
    return False, "Unsupported confirmation payload.", {}


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


def _build_app_runtime_clarification(app_query, candidates, *, operation="open"):
    operation_mode = "close" if operation == "close" else "open"
    intent = "OS_APP_CLOSE" if operation_mode == "close" else "OS_APP_OPEN"
    option_prefix = "close_app_runtime" if operation_mode == "close" else "open_app_runtime"

    options = []
    lines = [f"I found multiple app matches. Which one should I {operation_mode}?"]
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
    lines.append("Reply with the number (for example `1`) or `cancel`.")
    prompt = "\n".join(lines)
    payload = {
        "reason": "app_close_ambiguous" if operation_mode == "close" else "app_name_ambiguous",
        "prompt": prompt,
        "options": options,
        "source_text": app_query,
        "language": session_memory.get_preferred_language(),
        "confidence": 0.58,
        "entity_scores": {"app_name": 0.62},
    }
    return prompt, payload


def _build_file_search_runtime_clarification(filename, matches):
    options = []
    lines = [f"I found multiple files for '{filename}'. Which one do you mean?"]
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
    lines.append("Reply with the number (for example `1`) or `cancel`.")
    prompt = "\n".join(lines)
    payload = {
        "reason": "file_search_multiple_matches",
        "prompt": prompt,
        "options": options,
        "source_text": filename,
        "language": session_memory.get_preferred_language(),
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
                    (
                        f"Confirmation failed: {message} "
                        f"Use `confirm {token} <PIN_or_passphrase>`."
                    ),
                    {},
                )
            return False, f"Confirmation failed: {message}", {}
        return _execute_confirmed_payload(payload)

    if parsed.intent == "OS_ROLLBACK":
        ok, message = undo_last_action()
        return ok, message, {}

    if parsed.intent == "OS_FILE_SEARCH":
        filename = parsed.args.get("filename", "")
        if not filename:
            return False, "Please provide a filename to search for.", {}
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
        message = results[0] if results else "File not found."
        return True, message, {"indexed_search": False}

    if parsed.intent == "OS_FILE_NAVIGATION":
        return file_navigation.handle(parsed)

    if parsed.intent == "OS_APP_OPEN":
        app_name = parsed.args.get("app_name", "")
        if not app_name:
            return False, "Please provide an app name to open.", {}
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
            return False, "Please provide an app name to close.", {}
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
    session_memory.add_turn(parsed.raw, response)
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
        metrics.record_command("LANGUAGE_GATE_BLOCK", False, latency)
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

    pending = session_memory.get_pending_clarification()
    if pending:
        resolution = resolve_clarification_reply(effective_text, pending)
        if resolution.status == "cancelled":
            session_memory.clear_pending_clarification()
            latency = time.perf_counter() - start
            metrics.record_command("INTENT_CLARIFICATION", True, latency)
            log_action(
                "intent_clarification_cancelled",
                "success",
                details={"source_text": pending.get("source_text"), "language": language_result.language},
            )
            return resolution.message or "Clarification cancelled."

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

            latency = time.perf_counter() - start
            metrics.record_command(parsed.intent, success, latency)
            return _format_demo_output(parsed, success, response, meta)

        if resolution.status == "needs_clarification":
            latency = time.perf_counter() - start
            metrics.record_command("INTENT_CLARIFICATION", False, latency)
            return resolution.message or pending.get("prompt") or "Please clarify your intent."

        session_memory.clear_pending_clarification()

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
        metrics.record_command("INTENT_CLARIFICATION", False, latency)
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
                metrics.record_command("INTENT_CLARIFICATION", False, latency)
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

    latency = time.perf_counter() - start
    metrics.record_command(parsed.intent, success, latency)
    return _format_demo_output(parsed, success, response, meta)


def initialize_command_services():
    _ensure_job_queue_executor()
    job_queue_service.start()
    search_index_service.start()





