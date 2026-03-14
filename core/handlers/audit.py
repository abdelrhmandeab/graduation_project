from os_control.action_log import (
    read_recent_actions,
    reseal_audit_chain,
    verify_audit_chain,
)


def format_audit_log(limit):
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


def format_audit_verify():
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


def format_audit_reseal():
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
