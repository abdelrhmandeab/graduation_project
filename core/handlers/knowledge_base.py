from core.knowledge_base import knowledge_base_service
from os_control.action_log import log_action


def handle(parsed):
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
        auto_sync = status.get("auto_sync") or {}
        lines.extend(
            [
                f"auto_sync_enabled: {auto_sync.get('enabled')}",
                f"auto_sync_running: {auto_sync.get('running')}",
                f"auto_sync_interval_seconds: {auto_sync.get('interval_seconds')}",
                f"auto_sync_roots: {auto_sync.get('roots')}",
            ]
        )
        return True, "\n".join(lines), {}

    if action == "autosync_status":
        status = knowledge_base_service.auto_sync_status()
        lines = [
            "Knowledge Base Auto-Sync Status",
            f"enabled: {status.get('enabled')}",
            f"running: {status.get('running')}",
            f"interval_seconds: {status.get('interval_seconds')}",
            f"roots: {status.get('roots')}",
            f"last_run_ts: {status.get('last_run_ts')}",
            f"last_changes: {status.get('last_changes')}",
            f"last_error: {status.get('last_error') or 'none'}",
        ]
        return True, "\n".join(lines), {"kb_auto_sync": status}

    if action == "autosync_on":
        ok, message = knowledge_base_service.set_auto_sync(True)
        log_action("kb_autosync_on", "success" if ok else "failed")
        return ok, message, {}

    if action == "autosync_off":
        ok, message = knowledge_base_service.set_auto_sync(False)
        log_action("kb_autosync_off", "success" if ok else "failed")
        return ok, message, {}

    if action == "autosync_toggle":
        mode = str(args.get("mode") or "").strip().lower()
        if mode == "on":
            ok, message = knowledge_base_service.set_auto_sync(True)
            log_action("kb_autosync_on", "success" if ok else "failed")
            return ok, message, {}
        if mode == "off":
            ok, message = knowledge_base_service.set_auto_sync(False)
            log_action("kb_autosync_off", "success" if ok else "failed")
            return ok, message, {}
        if mode == "status":
            status = knowledge_base_service.auto_sync_status()
            return True, f"KB auto-sync: enabled={status.get('enabled')} running={status.get('running')}", {
                "kb_auto_sync": status
            }
        return False, "Unsupported autosync mode.", {}

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
