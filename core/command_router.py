import time

from core.command_parser import parse_command
from core.config import LLM_APPEND_SOURCE_CITATIONS
from core.demo_mode import is_enabled as is_demo_mode_enabled
from core.demo_mode import set_enabled as set_demo_mode
from core.handlers import audit, batch, benchmark, file_navigation
from core.handlers import job_queue as job_queue_handler
from core.handlers import knowledge_base, memory, persona, policy, search_index, voice
from core.logger import logger
from core.metrics import metrics
from core.persona import persona_manager
from core.session_memory import session_memory
from llm.ollama_client import ask_llm
from llm.prompt_builder import build_prompt_package
from os_control.action_log import log_action, read_recent_actions
from os_control.app_ops import open_app
from os_control.confirmation import confirmation_manager
from os_control.file_ops import find_files, get_current_directory, undo_last_action
from os_control.job_queue import job_queue_service
from os_control.policy import policy_engine
from os_control.search_index import search_index_service
from os_control.system_ops import execute_system_command, request_system_command


_JOB_QUEUE_EXECUTOR_READY = False

# Maps intents to their required permission key.
_PERMISSION_MAP = {
    "OS_CONFIRMATION": "confirmation",
    "OS_ROLLBACK": "rollback",
    "OS_FILE_SEARCH": "file_search",
    "OS_APP_OPEN": "app_open",
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
        if parsed.action in {"create_directory", "delete_item", "move_item", "rename_item"}:
            return "file_write"
        return "file_navigation"
    return _PERMISSION_MAP.get(parsed.intent)


def _execute_confirmed_payload(payload):
    kind = (payload or {}).get("kind")
    if kind == "system_command":
        action_key = payload.get("action_key")
        ok, message = execute_system_command(action_key)
        return ok, message, {"executed_confirmed_action": kind}
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


def _ensure_job_queue_executor():
    global _JOB_QUEUE_EXECUTOR_READY
    if _JOB_QUEUE_EXECUTOR_READY:
        return
    job_queue_service.configure_executor(_execute_job_command)
    _JOB_QUEUE_EXECUTOR_READY = True


def _execute_internal_command_text(command_text):
    parsed = parse_command(command_text)
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
    }:
        return False, f"Disallowed command for queued execution: {parsed.intent}"
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
            return True, "\n".join(indexed_results), {"indexed_search": True}
        results = find_files(filename, search_path=parsed.args.get("search_path"))
        message = "\n".join(results) if results else "File not found."
        return True, message, {"indexed_search": False}

    if parsed.intent == "OS_FILE_NAVIGATION":
        return file_navigation.handle(parsed)

    if parsed.intent == "OS_APP_OPEN":
        app_name = parsed.args.get("app_name", "")
        if not app_name:
            return False, "Please provide an app name to open.", {}
        ok, message = open_app(app_name)
        return ok, message, {}

    if parsed.intent == "OS_SYSTEM_COMMAND":
        action_key = parsed.args.get("action_key")
        return request_system_command(action_key)

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
        "CONFIRM:",
        f"- required: {'yes' if meta.get('requires_confirmation') else 'no'}",
    ]
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
    parsed = parse_command(text)
    start = time.perf_counter()
    success = False
    response = ""
    meta = {}

    try:
        success, response, meta = _dispatch(parsed)
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
