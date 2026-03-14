from os_control.file_ops import get_current_directory
from os_control.search_index import search_index_service


def handle(parsed):
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
