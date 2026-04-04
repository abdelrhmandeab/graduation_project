import re
import shutil
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.command_router import route_command
from core.config import SECOND_FACTOR_MAX_ATTEMPTS_PER_TOKEN
from core.session_memory import session_memory
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
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def _extract_token(text):
    match = re.search(r"confirm\s+([0-9a-f]{6})", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).lower().strip()


def test_confirmation_token_replay_is_rejected():
    policy_engine.set_profile("normal")

    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        target = tmp / "replay_case.txt"
        target.write_text("phase8", encoding="utf-8")

        prompt = route_command("delete replay_case.txt")
        _assert("Confirmation required" in prompt, f"Unexpected response: {prompt}")
        token = _extract_token(prompt)
        _assert(token, f"Missing token in response: {prompt}")

        first_confirm = route_command(f"confirm {token} 2468")
        _assert("Deleted" in first_confirm, f"Unexpected response: {first_confirm}")

        replay = route_command(f"confirm {token} 2468")
        _assert("not found or expired" in replay.lower(), f"Unexpected response: {replay}")


def test_confirmation_payload_injection_variants_are_rejected():
    policy_engine.set_profile("normal")

    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        injected_payloads = [
            "2468; shutdown computer",
            "2468 | shutdown computer",
            "2468 && shutdown computer",
        ]

        for index, payload in enumerate(injected_payloads):
            target = tmp / f"injection_case_{index}.txt"
            target.write_text("phase8", encoding="utf-8")

            prompt = route_command(f"delete {target.name}")
            _assert("Confirmation required" in prompt, f"Unexpected response: {prompt}")
            token = _extract_token(prompt)
            _assert(token, f"Missing token in response: {prompt}")

            injected = route_command(f"confirm {token} {payload}")
            lowered = injected.lower()
            _assert(
                "verification failed" in lowered
                or "too many failed second-factor attempts" in lowered,
                f"Unexpected response: {injected}",
            )
            _assert(target.exists(), "File must remain until valid confirmation")

            final_confirm = route_command(f"confirm {token} 2468")
            _assert("Deleted" in final_confirm, f"Unexpected response: {final_confirm}")


def test_unknown_confirmation_token_does_not_execute_pending_action():
    policy_engine.set_profile("normal")

    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        target = tmp / "unknown_token_case.txt"
        target.write_text("phase8", encoding="utf-8")

        prompt = route_command("delete unknown_token_case.txt")
        _assert("Confirmation required" in prompt, f"Unexpected response: {prompt}")
        token = _extract_token(prompt)
        _assert(token, f"Missing token in response: {prompt}")

        wrong_token = "deadbe" if token != "deadbe" else "feedbe"
        wrong_attempt = route_command(f"confirm {wrong_token} 2468")
        _assert("not found or expired" in wrong_attempt.lower(), f"Unexpected response: {wrong_attempt}")
        _assert(target.exists(), "File must remain when wrong token is submitted")

        final_confirm = route_command(f"confirm {token} 2468")
        _assert("Deleted" in final_confirm, f"Unexpected response: {final_confirm}")


def test_second_factor_lockout_window_blocks_immediate_bypass_attempts():
    policy_engine.set_profile("normal")
    route_command("memory clear")

    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        target = tmp / "auth_window_case.txt"
        target.write_text("phase8", encoding="utf-8")

        prompt = route_command("delete auth_window_case.txt")
        _assert("Confirmation required" in prompt, f"Unexpected response: {prompt}")
        token = _extract_token(prompt)
        _assert(token, f"Missing token in response: {prompt}")

        max_attempts = max(1, int(SECOND_FACTOR_MAX_ATTEMPTS_PER_TOKEN))
        last_failure = ""
        for index in range(max_attempts):
            last_failure = route_command(f"confirm {token} wrong-pin-{index}")

        _assert(
            "too many failed second-factor attempts" in last_failure.lower(),
            f"Unexpected response: {last_failure}",
        )

        immediate_token_retry = route_command(f"confirm {token} 2468")
        _assert(
            "too many failed second-factor attempts" in immediate_token_retry.lower(),
            f"Unexpected response: {immediate_token_retry}",
        )

        immediate_followup_retry = route_command("confirm it 2468")
        _assert(
            "too many failed second-factor attempts" in immediate_followup_retry.lower(),
            f"Unexpected response: {immediate_followup_retry}",
        )
        _assert(target.exists(), "Locked confirmation must not execute file deletion")


def test_path_escape_and_blocked_prefix_are_denied():
    policy_engine.set_profile("normal")

    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")

        traversal = route_command(r"go to ..\..\..\Windows\System32\config")
        _assert(
            "Blocked by policy" in traversal
            or "Path outside allowlist" in traversal
            or "Directory does not exist" in traversal,
            f"Unexpected response: {traversal}",
        )

        blocked_write = route_command(r"create folder C:\Windows\System32\config\phase8_blocked")
        _assert("blocked by policy" in blocked_write.lower(), f"Unexpected response: {blocked_write}")


def test_permanent_delete_is_blocked_even_under_pressure_wording():
    policy_engine.set_profile("normal")

    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        target = tmp / "danger.txt"
        target.write_text("danger", encoding="utf-8")

        response = route_command("delete permanently danger.txt")
        _assert("Permanent delete is disabled by configuration" in response, f"Unexpected response: {response}")
        _assert(target.exists(), "Permanent delete should not remove file by default")


def test_clarification_reply_suffix_does_not_execute_extra_actions():
    policy_engine.set_profile("normal")
    route_command("memory clear")

    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        (tmp / "report_1.txt").write_text("alpha", encoding="utf-8")
        (tmp / "report_2.txt").write_text("beta", encoding="utf-8")

        prompt = route_command("find file report")
        _assert("multiple files" in prompt.lower(), f"Unexpected response: {prompt}")

        injected_reply = route_command("1; delete report_2.txt")
        _assert("Path:" in injected_reply, f"Unexpected response: {injected_reply}")

        status = session_memory.status()
        _assert(not bool(status.get("pending_clarification")), f"Unexpected status: {status}")
        _assert((tmp / "report_2.txt").exists(), "Clarification reply injection must not delete files")


def test_clarification_alias_reply_misuse_does_not_execute_injected_action():
    policy_engine.set_profile("normal")
    route_command("memory clear")

    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        sentinel = tmp / "alias_guard.txt"
        sentinel.write_text("phase8", encoding="utf-8")

        prompt = route_command("open power")
        lowered_prompt = prompt.lower()
        _assert("1)" in prompt and "2)" in prompt, f"Unexpected response: {prompt}")
        _assert(
            "which application did you mean" in lowered_prompt
            or "which app did you mean" in lowered_prompt
            or "reply" in lowered_prompt,
            f"Unexpected response: {prompt}",
        )

        injected_reply = route_command("app; delete alias_guard.txt")
        lowered_reply = injected_reply.lower()
        _assert("delete" not in lowered_reply, f"Unexpected response: {injected_reply}")

        status = session_memory.status()
        _assert(not bool(status.get("pending_clarification")), f"Unexpected status: {status}")
        _assert(sentinel.exists(), "Clarification alias misuse must not execute injected file deletion")


if __name__ == "__main__":
    test_confirmation_token_replay_is_rejected()
    test_confirmation_payload_injection_variants_are_rejected()
    test_unknown_confirmation_token_does_not_execute_pending_action()
    test_second_factor_lockout_window_blocks_immediate_bypass_attempts()
    test_path_escape_and_blocked_prefix_are_denied()
    test_permanent_delete_is_blocked_even_under_pressure_wording()
    test_clarification_reply_suffix_does_not_execute_extra_actions()
    test_clarification_alias_reply_misuse_does_not_execute_injected_action()
    print("Adversarial safety Phase 8 tests passed.")
