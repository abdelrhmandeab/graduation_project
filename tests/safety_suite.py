import re
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.command_router import route_command
from core.demo_mode import set_enabled as set_demo_mode
from os_control.policy import policy_engine


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


@contextmanager
def _workspace_tempdir():
    base = Path(__file__).resolve().parents[1] / ".tmp_tests"
    base.mkdir(parents=True, exist_ok=True)
    temp_path = base / f"case_{uuid.uuid4().hex}"
    temp_path.mkdir(parents=True, exist_ok=True)
    try:
        yield str(temp_path)
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def test_invalid_confirmation_token():
    response = route_command("confirm abc123")
    _assert("Confirmation failed" in response, f"Unexpected response: {response}")


def test_destructive_command_abuse_blocked():
    response = route_command("shutdown computer")
    _assert("Confirmation required" in response, f"Unexpected response: {response}")

    token_match = re.search(r"confirm\s+([0-9a-f]{6})", response, flags=re.IGNORECASE)
    _assert(token_match is not None, "Token not found in confirmation response")
    token = token_match.group(1)

    bad_factor = route_command(f"confirm {token} 0000")
    _assert("verification failed" in bad_factor.lower(), f"Unexpected response: {bad_factor}")

    confirm_response = route_command(f"confirm {token} 2468")
    _assert(
        "Blocked by configuration" in confirm_response,
        f"Unexpected response: {confirm_response}",
    )


def test_high_risk_file_delete_requires_confirmation():
    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        target = Path(tmp) / "safe_delete.txt"
        target.write_text("x", encoding="utf-8")

        response = route_command("delete safe_delete.txt")
        _assert("Confirmation required" in response, f"Unexpected response: {response}")

        token_match = re.search(r"confirm\s+([0-9a-f]{6})", response, flags=re.IGNORECASE)
        _assert(token_match is not None, "Token not found in delete confirmation response")
        token = token_match.group(1)

        missing_factor = route_command(f"confirm {token}")
        _assert("Second factor required" in missing_factor, f"Unexpected response: {missing_factor}")

        confirm_response = route_command(f"confirm {token} 2468")
        _assert("Deleted" in confirm_response, f"Unexpected response: {confirm_response}")
        _assert(not target.exists(), "File should be deleted after valid confirmation")


def test_medium_risk_close_app_requires_confirmation_without_second_factor():
    response = route_command("close app notepad")
    _assert("Confirmation required" in response, f"Unexpected response: {response}")
    _assert("second factor" not in response.lower(), f"Unexpected response: {response}")

    token_match = re.search(r"confirm\s+([0-9a-f]{6})", response, flags=re.IGNORECASE)
    _assert(token_match is not None, "Token not found in close-app confirmation response")
    token = token_match.group(1)

    confirm_response = route_command(f"confirm {token}")
    _assert("Second factor required" not in confirm_response, f"Unexpected response: {confirm_response}")


def test_permanent_delete_blocked_by_default():
    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        target = Path(tmp) / "danger.txt"
        target.write_text("danger", encoding="utf-8")

        response = route_command("delete permanently danger.txt")
        _assert("Permanent delete is disabled by configuration" in response, f"Unexpected response: {response}")
        _assert(target.exists(), "File should not be deleted when permanent delete is blocked")


def test_path_traversal_blocked():
    with _workspace_tempdir() as tmp:
        response = route_command(f"go to {tmp}")
        _assert("Current directory set to" in response, f"Unexpected response: {response}")

        response = route_command(r"go to C:\Windows\System32\config")
        _assert(
            "Blocked by policy" in response or "Path outside allowlist" in response,
            f"Unexpected response: {response}",
        )


def test_read_only_mode_blocks_write():
    policy_engine.set_read_only_mode(True)
    try:
        with _workspace_tempdir() as tmp:
            route_command(f"go to {tmp}")
            response = route_command("create folder readonly_demo")
            _assert("read-only mode" in response, f"Unexpected response: {response}")
    finally:
        policy_engine.set_read_only_mode(False)


def test_command_permission_blocks_app_open():
    policy_engine.set_command_permission("app_open", False)
    try:
        response = route_command("open app notepad")
        _assert("blocked by policy" in response.lower(), f"Unexpected response: {response}")
    finally:
        policy_engine.set_command_permission("app_open", True)


def test_demo_mode_output():
    try:
        set_demo_mode(True)
        response = route_command("list drives")
        _assert("[DEMO MODE]" in response, "Demo mode response wrapper missing")
        _assert("PLAN:" in response and "AUDIT:" in response, "Demo mode sections missing")
    finally:
        set_demo_mode(False)


if __name__ == "__main__":
    test_invalid_confirmation_token()
    test_destructive_command_abuse_blocked()
    test_high_risk_file_delete_requires_confirmation()
    test_medium_risk_close_app_requires_confirmation_without_second_factor()
    test_permanent_delete_blocked_by_default()
    test_path_traversal_blocked()
    test_read_only_mode_blocks_write()
    test_command_permission_blocks_app_open()
    test_demo_mode_output()
    print("Safety suite passed.")
