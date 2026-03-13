import os
import time
from datetime import datetime, timezone

from audio.tts import speech_engine
from core.benchmark import run_quick_benchmark, run_resilience_demo
from core.command_parser import parse_command
from core.config import LLM_APPEND_SOURCE_CITATIONS
from core.demo_mode import is_enabled as is_demo_mode_enabled
from core.demo_mode import set_enabled as set_demo_mode
from core.knowledge_base import knowledge_base_service
from core.logger import logger
from core.metrics import metrics
from core.persona import persona_manager
from core.session_memory import session_memory
from llm.ollama_client import ask_llm
from llm.prompt_builder import build_prompt_package
from os_control.action_log import (
    log_action,
    read_recent_actions,
    reseal_audit_chain,
    verify_audit_chain,
)
from os_control.app_ops import open_app
from os_control.batch_ops import batch_manager
from os_control.confirmation import confirmation_manager
from os_control.file_ops import (
    change_directory,
    create_directory,
    delete_item,
    find_files,
    get_current_directory,
    get_file_metadata,
    list_directory,
    list_drives_win32,
    move_item,
    undo_last_action,
)
from os_control.job_queue import job_queue_service
from os_control.policy import policy_engine
from os_control.search_index import search_index_service
from os_control.system_ops import execute_system_command, request_system_command


_JOB_QUEUE_EXECUTOR_READY = False


def _required_permission(parsed):
    if parsed.intent == "OS_CONFIRMATION":
        return "confirmation"
    if parsed.intent == "OS_ROLLBACK":
        return "rollback"
    if parsed.intent == "OS_FILE_SEARCH":
        return "file_search"
    if parsed.intent == "OS_FILE_NAVIGATION":
        if parsed.action in {"create_directory", "delete_item", "move_item", "rename_item"}:
            return "file_write"
        return "file_navigation"
    if parsed.intent == "OS_APP_OPEN":
        return "app_open"
    if parsed.intent == "OS_SYSTEM_COMMAND":
        return "system_command"
    if parsed.intent == "METRICS_REPORT":
        return "metrics"
    if parsed.intent in {"AUDIT_LOG_REPORT", "AUDIT_VERIFY", "AUDIT_RESEAL"}:
        return "audit_log"
    if parsed.intent == "POLICY_COMMAND":
        return "policy"
    if parsed.intent == "BATCH_COMMAND":
        return "batch"
    if parsed.intent == "SEARCH_INDEX_COMMAND":
        return "search_index"
    if parsed.intent == "JOB_QUEUE_COMMAND":
        return "job_queue"
    if parsed.intent == "PERSONA_COMMAND":
        return "persona"
    if parsed.intent == "VOICE_COMMAND":
        return "speech"
    if parsed.intent == "KNOWLEDGE_BASE_COMMAND":
        return "knowledge_base"
    if parsed.intent == "MEMORY_COMMAND":
        return "memory"
    if parsed.intent == "OBSERVABILITY_REPORT":
        return "observability"
    if parsed.intent == "BENCHMARK_COMMAND":
        return "benchmark"
    return None


def _execute_confirmed_payload(payload):
    kind = (payload or {}).get("kind")
    if kind == "system_command":
        action_key = payload.get("action_key")
        ok, message = execute_system_command(action_key)
        return ok, message, {"executed_confirmed_action": kind}
    return False, "Unsupported confirmation payload.", {}


def _format_audit_log(limit):
    entries = read_recent_actions(limit=limit)
    if not entries:
        return "No audit entries found."

    lines = []
    for row in entries:
        lines.append(
            (
                f"id={row.get('id')} | ts={row.get('timestamp')} | "
                f"type={row.get('action_type')} | status={row.get('status')} | "
                f"hash={row.get('hash')}"
            )
        )
    return "\n".join(lines)


def _format_audit_verify():
    result = verify_audit_chain()
    if result.get("ok"):
        return (
            "Audit chain is valid. "
            f"checked={result.get('checked', 0)} "
            f"last_hash={result.get('last_hash', '')}"
        )

    return (
        "Audit chain verification failed. "
        f"checked={result.get('checked', 0)} "
        f"failed_id={result.get('failed_id', 'n/a')} "
        f"reason={result.get('reason', 'unknown')}"
    )


def _format_audit_reseal():
    result = reseal_audit_chain()
    resealed = int(result.get("resealed", 0))
    if result.get("error"):
        return f"Audit reseal failed: {result.get('error')}"
    verify = verify_audit_chain()
    if verify.get("ok"):
        return (
            f"Audit resealed successfully. rows={resealed} "
            f"checked={verify.get('checked', 0)} last_hash={verify.get('last_hash', '')}"
        )
    return (
        f"Audit resealed rows={resealed}, but verification still failed: "
        f"reason={verify.get('reason', 'unknown')} failed_id={verify.get('failed_id', 'n/a')}"
    )


def _format_policy_status():
    snapshot = policy_engine.status()
    lines = [
        "Policy Status",
        f"profile: {snapshot.get('profile')}",
        f"read_only_mode: {snapshot.get('read_only_mode')}",
        "permissions:",
    ]
    for key in sorted(snapshot.get("permissions", {})):
        lines.append(f"- {key}: {snapshot['permissions'][key]}")
    lines.append("allowed_paths:")
    for path in snapshot.get("allowed_paths", []):
        lines.append(f"- {path}")
    lines.append("blocked_prefixes:")
    for path in snapshot.get("blocked_prefixes", []):
        lines.append(f"- {path}")
    return "\n".join(lines)


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


def _handle_file_navigation(parsed):
    action = parsed.action
    args = parsed.args

    if action == "pwd":
        return True, f"Current directory: {get_current_directory()}", {}
    if action == "cd":
        return (*change_directory(args.get("path", "")), {})
    if action == "list_drives":
        return (*list_drives_win32(), {})
    if action == "list_directory":
        return (*list_directory(args.get("path")), {})
    if action == "file_info":
        return (*get_file_metadata(args.get("path", "")), {})
    if action == "create_directory":
        return (*create_directory(args.get("path", "")), {})
    if action == "delete_item":
        return (*delete_item(args.get("path", "")), {})
    if action == "move_item":
        return (*move_item(args.get("source", ""), args.get("destination", "")), {})
    if action == "rename_item":
        source = args.get("source", "")
        new_name = args.get("new_name", "")
        source_abs = (
            os.path.abspath(source)
            if os.path.isabs(source)
            else os.path.join(get_current_directory(), source)
        )
        destination = os.path.join(os.path.dirname(source_abs), new_name)
        return (*move_item(source, destination), {})

    return False, "Unsupported file navigation command.", {}


def _handle_demo_mode(parsed):
    if parsed.action == "on":
        set_demo_mode(True)
        return True, "Demo mode enabled.", {}
    if parsed.action == "off":
        set_demo_mode(False)
        return True, "Demo mode disabled.", {}
    enabled = is_demo_mode_enabled()
    return True, f"Demo mode is {'ON' if enabled else 'OFF'}.", {}


def _format_persona_status():
    status = persona_manager.status()
    lines = [
        "Persona Status",
        f"active_profile: {status['active_profile']}",
        f"speech_style: {status['speech_style']}",
        f"speech_rate: {status['speech_rate']}",
        f"clone_enabled: {status['clone_enabled']}",
        f"clone_provider: {status['clone_provider']}",
        f"clone_reference_audio: {status['clone_reference_audio'] or 'not_set'}",
        "available_profiles:",
    ]
    for profile in status["available_profiles"]:
        lines.append(f"- {profile}")
    lines.append("profile_voice_map:")
    for profile, voice_cfg in sorted(status.get("voice_profiles", {}).items()):
        lines.append(
            (
                f"- {profile}: clone_enabled={voice_cfg.get('clone_enabled')}, "
                f"clone_provider={voice_cfg.get('clone_provider')}, "
                f"reference_audio={voice_cfg.get('reference_audio') or 'not_set'}"
            )
        )
    return "\n".join(lines)


def _handle_persona_command(parsed):
    action = parsed.action
    args = parsed.args

    if action == "status":
        return True, _format_persona_status(), {}
    if action == "list":
        profiles = persona_manager.list_profiles()
        return True, "Available personas:\n" + "\n".join(f"- {p}" for p in profiles), {}
    if action == "set":
        ok, message = persona_manager.set_profile(args.get("profile", ""))
        log_action(
            "persona_set",
            "success" if ok else "failed",
            details={"profile": args.get("profile")},
            error=None if ok else message,
        )
        return ok, message, {"persona": persona_manager.get_profile()}

    if action == "voice_status":
        voice_map = persona_manager.profile_voice_map()
        lines = ["Persona Voice Status"]
        for profile, voice_cfg in sorted(voice_map.items()):
            lines.append(
                (
                    f"- {profile}: clone_enabled={voice_cfg.get('clone_enabled')}, "
                    f"clone_provider={voice_cfg.get('clone_provider')}, "
                    f"reference_audio={voice_cfg.get('reference_audio') or 'not_set'}"
                )
            )
        return True, "\n".join(lines), {}

    if action == "set_profile_clone_enabled":
        profile = args.get("profile")
        enabled = bool(args.get("enabled"))
        ok, message = persona_manager.set_profile_clone_enabled(profile, enabled)
        log_action(
            "persona_voice_clone_toggle",
            "success" if ok else "failed",
            details={"profile": profile, "enabled": enabled},
            error=None if ok else message,
        )
        return ok, message, {}

    if action == "set_profile_clone_provider":
        profile = args.get("profile")
        provider = args.get("provider")
        ok, message = persona_manager.set_profile_clone_provider(profile, provider)
        log_action(
            "persona_voice_provider",
            "success" if ok else "failed",
            details={"profile": profile, "provider": provider},
            error=None if ok else message,
        )
        return ok, message, {}

    if action == "set_profile_clone_reference":
        profile = args.get("profile")
        path = args.get("path")
        ok, message = persona_manager.set_profile_clone_reference_audio(profile, path)
        log_action(
            "persona_voice_reference",
            "success" if ok else "failed",
            details={"profile": profile, "path": path},
            error=None if ok else message,
        )
        return ok, message, {}

    return False, "Unsupported persona command.", {}


def _handle_voice_command(parsed):
    action = parsed.action
    args = parsed.args

    if action == "status":
        clone = persona_manager.get_clone_settings()
        speaking = speech_engine.is_speaking()
        enabled = speech_engine.is_enabled()
        lines = [
            "Voice Status",
            f"speech_enabled: {enabled}",
            f"is_speaking: {speaking}",
            f"active_persona: {clone.get('profile')}",
            f"speech_rate: {persona_manager.get_speech_rate()}",
            f"clone_enabled: {clone['enabled']}",
            f"clone_provider: {clone['provider']}",
            f"clone_reference_audio: {clone['reference_audio'] or 'not_set'}",
        ]
        return True, "\n".join(lines), {}

    if action == "clone_on":
        ok, message = persona_manager.set_clone_enabled(True)
        log_action("voice_clone_toggle", "success", details={"enabled": True})
        return ok, message, {"voice_clone": True}

    if action == "clone_off":
        ok, message = persona_manager.set_clone_enabled(False)
        log_action("voice_clone_toggle", "success", details={"enabled": False})
        return ok, message, {"voice_clone": False}

    if action == "set_provider":
        provider = args.get("provider", "")
        ok, message = persona_manager.set_clone_provider(provider)
        log_action(
            "voice_clone_provider",
            "success" if ok else "failed",
            details={"provider": provider},
            error=None if ok else message,
        )
        return ok, message, {}

    if action == "set_reference":
        ref_path = args.get("path", "")
        ok, message = persona_manager.set_clone_reference_audio(ref_path)
        log_action(
            "voice_clone_reference",
            "success" if ok else "failed",
            details={"path": ref_path},
            error=None if ok else message,
        )
        return ok, message, {}

    if action == "interrupt":
        if not speech_engine.is_speaking():
            return True, "No active speech to interrupt.", {"speech_interrupted": False}
        speech_engine.interrupt()
        log_action("speech_interrupt", "success")
        return True, "Speech interrupted.", {"speech_interrupted": True}

    if action == "speech_on":
        ok, message = speech_engine.set_enabled(True)
        log_action("speech_toggle", "success", details={"enabled": True})
        return ok, message, {"speech_enabled": True}

    if action == "speech_off":
        ok, message = speech_engine.set_enabled(False)
        log_action("speech_toggle", "success", details={"enabled": False})
        return ok, message, {"speech_enabled": False}

    return False, "Unsupported voice command.", {}


def _handle_knowledge_base_command(parsed):
    action = parsed.action
    args = parsed.args

    if action == "status":
        status = knowledge_base_service.status()
        lines = [
            "Knowledge Base Status",
            f"enabled: {status['enabled']}",
            f"retrieval_enabled: {status['retrieval_enabled']}",
            f"vector_backend: {status['vector_backend']}",
            f"embedding_backend: {status['embedding_backend']}",
            f"embedding_dim: {status['embedding_dim']}",
            f"file_count: {status['file_count']}",
            f"chunk_count: {status['chunk_count']}",
            f"storage_dir: {status['storage_dir']}",
            f"source_state_file: {status['source_state_file']}",
        ]
        return True, "\n".join(lines), {}

    if action == "add_file":
        path = args.get("path", "")
        ok, message, chunk_count = knowledge_base_service.add_document(path)
        log_action(
            "kb_add_file",
            "success" if ok else "failed",
            details={"path": path, "chunk_count": chunk_count},
            error=None if ok else message,
        )
        return ok, message, {"kb_chunks": chunk_count}

    if action == "index_dir":
        path = args.get("path", "")
        ok, message, files_count, chunk_count = knowledge_base_service.index_directory(path)
        log_action(
            "kb_index_dir",
            "success" if ok else "failed",
            details={"path": path, "file_count": files_count, "chunk_count": chunk_count},
            error=None if ok else message,
        )
        return ok, message, {"kb_files": files_count, "kb_chunks": chunk_count}

    if action == "sync_dir":
        path = args.get("path", "")
        ok, message, indexed_files, skipped_files, removed_files = knowledge_base_service.sync_directory(path)
        log_action(
            "kb_sync_dir",
            "success" if ok else "failed",
            details={
                "path": path,
                "indexed_files": indexed_files,
                "skipped_files": skipped_files,
                "removed_files": removed_files,
            },
            error=None if ok else message,
        )
        return (
            ok,
            message,
            {
                "kb_indexed_files": indexed_files,
                "kb_skipped_files": skipped_files,
                "kb_removed_files": removed_files,
            },
        )

    if action == "search":
        query = args.get("query", "")
        results = knowledge_base_service.search(query)
        if not results:
            return True, "No knowledge base results found.", {}

        lines = ["Knowledge Search Results"]
        for item in results:
            snippet = item["text"].replace("\n", " ").strip()
            if len(snippet) > 160:
                snippet = snippet[:160] + "..."
            lines.append(
                (
                    f"- score={item['score']:.3f} | source={item['source']} | "
                    f"chunk={item['chunk_index']} | {snippet}"
                )
            )
        return True, "\n".join(lines), {"kb_results": len(results)}

    if action == "quality":
        report = knowledge_base_service.quality_report()
        if not report.get("ok"):
            return (
                True,
                (
                    "Knowledge Quality Report\n"
                    f"ok=False\nreason={report.get('reason')}\n"
                    f"chunk_count={report.get('status', {}).get('chunk_count', 0)}"
                ),
                {"kb_quality_ok": False},
            )

        status = report.get("status", {})
        lines = [
            "Knowledge Quality Report",
            "ok=True",
            f"semantic_backend_ready={report.get('semantic_backend_ready')}",
            f"embedding_backend={status.get('embedding_backend')}",
            f"vector_backend={status.get('vector_backend')}",
            f"file_count={status.get('file_count')}",
            f"chunk_count={status.get('chunk_count')}",
            f"probes={report.get('probes')}",
            f"hits={report.get('hits')}",
            f"accuracy_at_k={report.get('accuracy_at_k'):.2%}",
        ]
        for probe in report.get("probe_preview", []):
            lines.append(
                (
                    f"- probe='{probe.get('query')}' matched={probe.get('matched')} "
                    f"source={probe.get('expected_source')}"
                )
            )
        return True, "\n".join(lines), {"kb_quality_ok": True, "kb_quality_accuracy": report.get("accuracy_at_k")}

    if action == "clear":
        ok, message = knowledge_base_service.clear()
        log_action("kb_clear", "success" if ok else "failed")
        return ok, message, {}

    if action == "retrieval_on":
        ok, message = knowledge_base_service.set_retrieval_enabled(True)
        log_action("kb_retrieval_toggle", "success", details={"enabled": True})
        return ok, message, {"kb_retrieval": True}

    if action == "retrieval_off":
        ok, message = knowledge_base_service.set_retrieval_enabled(False)
        log_action("kb_retrieval_toggle", "success", details={"enabled": False})
        return ok, message, {"kb_retrieval": False}

    return False, "Unsupported knowledge base command.", {}


def _handle_memory_command(parsed):
    action = parsed.action

    if action == "status":
        status = session_memory.status()
        lines = [
            "Memory Status",
            f"enabled: {status['enabled']}",
            f"turn_count: {status['turn_count']}",
            f"max_turns: {status['max_turns']}",
            f"file: {status['file']}",
        ]
        return True, "\n".join(lines), {}

    if action == "clear":
        ok, message = session_memory.clear()
        log_action("memory_clear", "success" if ok else "failed")
        return ok, message, {}

    if action == "on":
        ok, message = session_memory.set_enabled(True)
        log_action("memory_toggle", "success", details={"enabled": True})
        return ok, message, {}

    if action == "off":
        ok, message = session_memory.set_enabled(False)
        log_action("memory_toggle", "success", details={"enabled": False})
        return ok, message, {}

    if action == "show":
        context = session_memory.build_context()
        if not context:
            return True, "Memory is empty.", {}
        return True, "Recent Memory\n" + context, {}

    return False, "Unsupported memory command.", {}


def _handle_benchmark_command(parsed):
    action = parsed.action
    if action == "run":
        payload = run_quick_benchmark(route_command)
        sla = payload.get("sla") or {}
        lines = [
            "Benchmark Report",
            f"scenarios: {payload['scenario_count']}",
            f"success_rate: {payload['success_rate']:.2%}",
            f"p50_latency_ms: {payload['p50_latency_ms']:.1f}",
            f"p95_latency_ms: {payload['p95_latency_ms']:.1f}",
            f"sla_passed: {sla.get('passed')}",
        ]
        for check in sla.get("checks", []):
            lines.append(
                f"- sla {check.get('name')}: actual={check.get('actual')}, "
                f"expected {check.get('operator')} {check.get('threshold')}, passed={check.get('passed')}"
            )
        for row in payload["results"]:
            lines.append(
                (
                    f"- {row['name']}: ok={row['ok']}, latency_ms={row['latency_ms']:.1f}, "
                    f"command={row['command']}"
                )
            )
        return True, "\n".join(lines), {"benchmark_success_rate": payload["success_rate"]}

    if action == "resilience_demo":
        payload = run_resilience_demo(route_command)
        sla = payload.get("sla") or {}
        lines = [
            "Resilience Report",
            f"scenarios: {payload['scenario_count']}",
            f"success_rate: {payload['success_rate']:.2%}",
            f"p50_latency_ms: {payload['p50_latency_ms']:.1f}",
            f"p95_latency_ms: {payload['p95_latency_ms']:.1f}",
            f"sla_passed: {sla.get('passed')}",
        ]
        for check in sla.get("checks", []):
            lines.append(
                f"- sla {check.get('name')}: actual={check.get('actual')}, "
                f"expected {check.get('operator')} {check.get('threshold')}, passed={check.get('passed')}"
            )
        for row in payload["results"]:
            line = f"- {row['name']}: ok={row['ok']}, latency_ms={row['latency_ms']:.1f}"
            if row.get("error"):
                line += f", error={row['error']}"
            lines.append(line)
        return True, "\n".join(lines), {"resilience_success_rate": payload["success_rate"]}

    return False, "Unsupported benchmark command.", {}


def _handle_policy_command(parsed):
    if parsed.action == "status":
        return True, _format_policy_status(), {}

    if parsed.action == "set_profile":
        ok, message = policy_engine.set_profile(parsed.args.get("profile", ""))
        status = "success" if ok else "failed"
        log_action("policy_set_profile", status, details={"profile": parsed.args.get("profile")})
        return ok, message, {}

    if parsed.action == "set_read_only":
        enabled = bool(parsed.args.get("enabled"))
        policy_engine.set_read_only_mode(enabled)
        log_action("policy_set_read_only", "success", details={"enabled": enabled})
        return True, f"Policy read-only mode set to: {enabled}", {}

    if parsed.action == "set_permission":
        permission = parsed.args.get("permission")
        enabled = bool(parsed.args.get("enabled"))
        if not permission:
            return False, "Permission key is required.", {}
        policy_engine.set_command_permission(permission, enabled)
        log_action(
            "policy_set_permission",
            "success",
            details={"permission": permission, "enabled": enabled},
        )
        return True, f"Permission {permission} set to: {enabled}", {}

    return False, "Unsupported policy command.", {}


def _handle_batch_command(parsed):
    if parsed.action == "plan":
        ok, message = batch_manager.plan()
        return ok, message, {}
    if parsed.action == "add":
        ok, message = batch_manager.add(parsed.args.get("command_text", ""))
        return ok, message, {}
    if parsed.action == "status":
        ok, message = batch_manager.status()
        return ok, message, {}
    if parsed.action == "preview":
        ok, message = batch_manager.preview(parse_command)
        return ok, message, {}
    if parsed.action == "abort":
        ok, message = batch_manager.abort()
        return ok, message, {}
    if parsed.action == "commit":
        ok, message = batch_manager.commit(parse_command, _execute_internal_command_text)
        return ok, message, {}
    return False, "Unsupported batch command.", {}


def _handle_search_index_command(parsed):
    action = parsed.action
    args = parsed.args

    if action == "start":
        ok, message = search_index_service.start()
        return True, message if ok else message, {}

    if action == "status":
        status = search_index_service.status()
        lines = [
            "Search Index Status",
            f"running: {status.get('running')}",
            f"refresh_seconds: {status.get('refresh_seconds')}",
            f"tracked_roots: {len(status.get('tracked_roots', []))}",
        ]
        for root in status.get("tracked_roots", []):
            indexed_at = status.get("indexed_roots", {}).get(root, "never")
            lines.append(f"- {root} | indexed_at={indexed_at}")
        return True, "\n".join(lines), {}

    if action == "refresh":
        root = args.get("root")
        if root is None:
            root = get_current_directory()
        search_index_service.start()
        ok, message = search_index_service.refresh_now(root=root)
        return ok, message, {}

    if action == "search":
        query = args.get("query", "")
        root = args.get("root") or get_current_directory()
        search_index_service.start()
        results = search_index_service.search(query, root=root)
        if not results:
            return True, "No indexed results found.", {}
        return True, "\n".join(results), {"indexed_search": True}

    return False, "Unsupported search index command.", {}


def _ensure_job_queue_executor():
    global _JOB_QUEUE_EXECUTOR_READY
    if _JOB_QUEUE_EXECUTOR_READY:
        return
    job_queue_service.configure_executor(_execute_job_command)
    _JOB_QUEUE_EXECUTOR_READY = True


def _format_timestamp(ts):
    if ts is None:
        return "n/a"
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.isoformat()


def _format_job(job):
    if not job:
        return "Job not found."
    return (
        f"id={job['id']} | status={job['status']} | attempts={job['attempts']} | "
        f"max_retries={job['max_retries']} | run_at={_format_timestamp(job['run_at'])} | "
        f"command={job['command_text']} | last_error={job['last_error'] or 'none'}"
    )


def _handle_job_queue_command(parsed):
    _ensure_job_queue_executor()
    action = parsed.action
    args = parsed.args

    if action == "worker_start":
        _ok, message = job_queue_service.start()
        return True, message, {}
    if action == "worker_stop":
        job_queue_service.stop()
        return True, "Job queue worker stopped.", {}
    if action == "worker_status":
        running = job_queue_service.is_running()
        return True, f"Job queue worker running: {running}", {}

    if action == "enqueue":
        delay = int(args.get("delay_seconds", 0))
        command_text = args.get("command_text", "")
        ok, message, _job = job_queue_service.enqueue(command_text, delay_seconds=delay)
        return ok, message, {}

    if action == "status":
        job = job_queue_service.status(args.get("job_id"))
        if not job:
            return False, "Job not found.", {}
        return True, _format_job(job), {}

    if action == "cancel":
        ok, message, _job = job_queue_service.cancel(args.get("job_id"))
        return ok, message, {}

    if action == "retry":
        ok, message, _job = job_queue_service.retry(
            args.get("job_id"),
            delay_seconds=args.get("delay_seconds", 0),
        )
        return ok, message, {}

    if action == "list":
        jobs = job_queue_service.list(limit=args.get("limit", 10), status=args.get("status"))
        if not jobs:
            return True, "No jobs found.", {}
        return True, "\n".join(_format_job(job) for job in jobs), {}

    return False, "Unsupported job queue command.", {}


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
        return _handle_demo_mode(parsed)

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
        return _handle_file_navigation(parsed)

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
        return True, _format_audit_log(limit), {}

    if parsed.intent == "AUDIT_VERIFY":
        return True, _format_audit_verify(), {}

    if parsed.intent == "AUDIT_RESEAL":
        return True, _format_audit_reseal(), {}

    if parsed.intent == "PERSONA_COMMAND":
        return _handle_persona_command(parsed)

    if parsed.intent == "VOICE_COMMAND":
        return _handle_voice_command(parsed)

    if parsed.intent == "KNOWLEDGE_BASE_COMMAND":
        return _handle_knowledge_base_command(parsed)

    if parsed.intent == "MEMORY_COMMAND":
        return _handle_memory_command(parsed)

    if parsed.intent == "OBSERVABILITY_REPORT":
        return True, metrics.format_observability_report(), {}

    if parsed.intent == "BENCHMARK_COMMAND":
        return _handle_benchmark_command(parsed)

    if parsed.intent == "POLICY_COMMAND":
        return _handle_policy_command(parsed)

    if parsed.intent == "BATCH_COMMAND":
        if not allow_batch:
            return False, "Nested batch commands are not allowed.", {}
        return _handle_batch_command(parsed)

    if parsed.intent == "SEARCH_INDEX_COMMAND":
        return _handle_search_index_command(parsed)

    if parsed.intent == "JOB_QUEUE_COMMAND":
        if not allow_job_queue:
            return False, "Nested job queue commands are not allowed.", {}
        return _handle_job_queue_command(parsed)

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
    audit = latest[0] if latest else {}

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
    if audit:
        lines.append(f"- id: {audit.get('id')}")
        lines.append(f"- action: {audit.get('action_type')} ({audit.get('status')})")
        lines.append(f"- hash: {audit.get('hash')}")
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
