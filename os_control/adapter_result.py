from __future__ import annotations

from typing import Any


def build_adapter_result(
    success: bool,
    user_message: str,
    error_code: str = "",
    debug_info: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": bool(success),
        "user_message": str(user_message or ""),
        "error_code": str(error_code or ""),
        "debug_info": dict(debug_info or {}),
    }
    payload.update(extra)
    return payload


def success_result(
    user_message: str,
    debug_info: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return build_adapter_result(
        success=True,
        user_message=user_message,
        error_code="",
        debug_info=debug_info,
        **extra,
    )


def failure_result(
    user_message: str,
    error_code: str = "execution_failed",
    debug_info: dict[str, Any] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    return build_adapter_result(
        success=False,
        user_message=user_message,
        error_code=error_code,
        debug_info=debug_info,
        **extra,
    )


def confirmation_result(
    user_message: str,
    token: str,
    second_factor: bool = False,
    risk_tier: str = "",
    debug_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_adapter_result(
        success=True,
        user_message=user_message,
        error_code="",
        debug_info=debug_info,
        requires_confirmation=True,
        token=token,
        second_factor=bool(second_factor),
        risk_tier=risk_tier,
    )


def to_router_tuple(response: Any) -> tuple[bool, str, dict[str, Any]]:
    if isinstance(response, dict):
        success = bool(response.get("success", False))
        message = str(response.get("user_message", ""))
        meta: dict[str, Any] = {}

        debug_info = response.get("debug_info")
        if isinstance(debug_info, dict):
            meta.update(debug_info)

        for key in (
            "error_code",
            "requires_confirmation",
            "token",
            "second_factor",
            "risk_tier",
            "executed_confirmed_action",
        ):
            if key in response:
                meta[key] = response[key]

        return success, message, meta

    if isinstance(response, tuple):
        if len(response) >= 2:
            success = bool(response[0])
            message = str(response[1])
            meta = response[2] if len(response) > 2 and isinstance(response[2], dict) else {}
            return success, message, dict(meta)
        return False, "Invalid adapter tuple response.", {"error_code": "invalid_adapter_response"}

    return False, "Invalid adapter response.", {"error_code": "invalid_adapter_response"}


def to_legacy_pair(response: Any) -> tuple[bool, str]:
    success, message, _meta = to_router_tuple(response)
    return success, message
