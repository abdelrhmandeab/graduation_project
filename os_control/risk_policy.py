SYSTEM_RISK_OVERRIDES = {
    "lock": "medium",
    "sleep": "medium",
}

FILE_OPERATION_RISK = {
    "move_item": "medium",
    "rename_item": "medium",
    "delete_item": "high",
    "delete_item_permanent": "high",
}

APP_OPERATION_RISK = {
    "close_app": "medium",
}


def risk_tier_for_system(action_key, *, destructive=False, requires_confirmation=False):
    if not action_key:
        return "low"
    override = SYSTEM_RISK_OVERRIDES.get(str(action_key).strip().lower())
    if override:
        return override
    if destructive:
        return "high"
    if requires_confirmation:
        return "medium"
    return "low"


def risk_tier_for_file_operation(operation):
    return FILE_OPERATION_RISK.get(str(operation or "").strip().lower(), "low")


def risk_tier_for_app_operation(operation):
    return APP_OPERATION_RISK.get(str(operation or "").strip().lower(), "low")


def validate_risk_policy_coverage(*, system_commands=None, file_operations=None, app_operations=None):
    errors = []

    if isinstance(system_commands, dict):
        for action_key, cfg in system_commands.items():
            requires_confirmation = bool((cfg or {}).get("requires_confirmation", bool((cfg or {}).get("destructive"))))
            computed = risk_tier_for_system(
                action_key,
                destructive=bool((cfg or {}).get("destructive")),
                requires_confirmation=requires_confirmation,
            )
            if requires_confirmation and computed == "low":
                errors.append(f"system:{action_key}:requires_confirmation_but_low_risk")

    if file_operations:
        for operation in file_operations:
            computed = risk_tier_for_file_operation(operation)
            if str(operation or "").strip() and computed == "low":
                errors.append(f"file:{operation}:missing_risk_mapping")

    if app_operations:
        for operation in app_operations:
            computed = risk_tier_for_app_operation(operation)
            if str(operation or "").strip() and computed == "low":
                errors.append(f"app:{operation}:missing_risk_mapping")

    return {
        "ok": not errors,
        "errors": errors,
    }