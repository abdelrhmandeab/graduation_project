from os_control.action_log import log_action
from os_control.policy import policy_engine


def _format_status():
    snapshot = policy_engine.status()
    lines = [
        "Policy Status",
        f"profile: {snapshot.get('profile')}",
        f"read_only_mode: {snapshot.get('read_only_mode')}",
        f"dry_run_mode: {snapshot.get('dry_run_mode')}",
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


def handle(parsed):
    if parsed.action == "status":
        return True, _format_status(), {}

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

    if parsed.action == "set_dry_run":
        enabled = bool(parsed.args.get("enabled"))
        policy_engine.set_dry_run_mode(enabled)
        log_action("policy_set_dry_run", "success", details={"enabled": enabled})
        return True, f"Policy dry-run mode set to: {enabled}", {}

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
