import re
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.command_router import route_command
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


def test_persona_profiles_phase5():
    policy_engine.set_profile("normal")

    response = route_command("persona set professional")
    _assert("Persona set to: professional" in response, f"Unexpected response: {response}")

    response = route_command("persona set friendly")
    _assert("Persona set to: friendly" in response, f"Unexpected response: {response}")

    response = route_command("persona set brief")
    _assert("Persona set to: brief" in response, f"Unexpected response: {response}")

    status = route_command("persona status")
    _assert("active_profile: brief" in status, f"Unexpected response: {status}")

    route_command("persona set assistant")


def test_followup_delete_and_confirm_it():
    policy_engine.set_profile("normal")
    route_command("memory clear")

    with _workspace_tempdir() as tmp:
        target = tmp / "phase5_followup.txt"
        target.write_text("phase5", encoding="utf-8")

        response = route_command(f"go to {tmp}")
        _assert("Current directory set to" in response, f"Unexpected response: {response}")

        response = route_command("file info phase5_followup.txt")
        _assert("Path:" in response, f"Unexpected response: {response}")

        response = route_command("احذفه")
        _assert("Confirmation required" in response, f"Unexpected response: {response}")

        token_match = re.search(r"confirm\s+([0-9a-f]{6})", response, flags=re.IGNORECASE)
        _assert(token_match is not None, "Token not found in delete confirmation response")
        token = token_match.group(1)

        followup_confirm = route_command("confirm it")
        _assert("Second factor required" in followup_confirm, f"Unexpected response: {followup_confirm}")

        final_confirm = route_command(f"confirm {token} 2468")
        _assert("Deleted" in final_confirm, f"Unexpected response: {final_confirm}")


def test_followup_close_it_and_memory_status():
    policy_engine.set_profile("normal")
    route_command("memory clear")

    response = route_command("open app notepad")
    _assert("Opening" in response or "could not" in response.lower(), f"Unexpected response: {response}")

    response = route_command("close it")
    _assert("Confirmation required" in response, f"Unexpected response: {response}")

    confirm_response = route_command("confirm it")
    _assert("Second factor required" not in confirm_response, f"Unexpected response: {confirm_response}")

    memory_status = route_command("memory status")
    _assert("last_app:" in memory_status, f"Unexpected response: {memory_status}")
    _assert("last_file:" in memory_status, f"Unexpected response: {memory_status}")
    _assert("pending_confirmation_token:" in memory_status, f"Unexpected response: {memory_status}")


def test_followup_rename_move_cancel_chain():
    policy_engine.set_profile("normal")
    route_command("memory clear")

    with _workspace_tempdir() as tmp:
        source = tmp / "chain_source.txt"
        source.write_text("phase5-chain", encoding="utf-8")

        response = route_command(f"go to {tmp}")
        _assert("Current directory set to" in response, f"Unexpected response: {response}")

        response = route_command("file info chain_source.txt")
        _assert("Path:" in response, f"Unexpected response: {response}")

        response = route_command("rename it to chain_renamed.txt")
        _assert("Confirmation required" in response, f"Unexpected response: {response}")

        cancel_response = route_command("cancel it")
        _assert("cancelled" in cancel_response.lower(), f"Unexpected response: {cancel_response}")
        _assert(source.exists(), "Rename should not happen after cancellation")

        response = route_command("rename it to chain_renamed.txt")
        _assert("Confirmation required" in response, f"Unexpected response: {response}")
        confirm_response = route_command("confirm it")
        _assert("Second factor required" not in confirm_response, f"Unexpected response: {confirm_response}")

        renamed = tmp / "chain_renamed.txt"
        _assert(renamed.exists(), "Rename should succeed after confirmation")

        response = route_command(r"move it to moved\chain_final.txt")
        _assert("Confirmation required" in response, f"Unexpected response: {response}")
        confirm_move = route_command("confirm it")
        _assert("Second factor required" not in confirm_move, f"Unexpected response: {confirm_move}")

        moved = tmp / "moved" / "chain_final.txt"
        _assert(moved.exists(), "Move follow-up should use the latest file context")


def test_bilingual_template_library_for_arabic_clarification():
    policy_engine.set_profile("normal")
    route_command("memory clear")

    with _workspace_tempdir() as tmp:
        (tmp / "تقرير_1.txt").write_text("a", encoding="utf-8")
        (tmp / "تقرير_2.txt").write_text("b", encoding="utf-8")

        response = route_command(f"go to {tmp}")
        _assert("Current directory set to" in response, f"Unexpected response: {response}")

        response = route_command("ابحث عن ملف تقرير")
        _assert("وجدت" in response and "اكتب الرقم" in response, f"Unexpected response: {response}")

        cancel_response = route_command("cancel")
        _assert("cancelled" in cancel_response.lower(), f"Unexpected response: {cancel_response}")


def test_deterministic_anti_repetition():
    policy_engine.set_profile("normal")
    route_command("memory clear")
    route_command("memory on")
    route_command("persona set brief")

    first = route_command("current directory")
    second = route_command("current directory")

    _assert(second != first, "Second deterministic response should not be identical")
    _assert(second.endswith(first), f"Expected anti-repetition prefix in second response: {second}")

    route_command("persona set assistant")
    route_command("memory clear")


def test_context_slot_recency_snapshot_and_followup_priority():
    policy_engine.set_profile("normal")
    route_command("memory clear")

    with _workspace_tempdir() as tmp:
        target = tmp / "recency_case.txt"
        target.write_text("phase5-recency", encoding="utf-8")

        session_memory.set_last_file(str(target))
        first_file_ts = session_memory.get_last_file_timestamp()
        _assert(first_file_ts > 0.0, f"Unexpected file timestamp: {first_file_ts}")

        time.sleep(0.02)
        session_memory.set_last_app("imaginary_app_for_recency")
        app_ts = session_memory.get_last_app_timestamp()
        _assert(app_ts > first_file_ts, "App timestamp should be newer than file timestamp")

        app_priority = route_command("open it")
        _assert("Path:" not in app_priority, f"Expected app-priority follow-up: {app_priority}")

        time.sleep(0.02)
        session_memory.set_last_file(str(target))
        file_ts = session_memory.get_last_file_timestamp()
        _assert(file_ts > app_ts, "File timestamp should become newest after update")

        file_priority = route_command("open it")
        _assert("Path:" in file_priority, f"Expected file-priority follow-up: {file_priority}")

        snapshot = session_memory.context_snapshot()
        _assert("last_app_updated_at" in snapshot, f"Missing app timestamp in snapshot: {snapshot}")
        _assert("last_file_updated_at" in snapshot, f"Missing file timestamp in snapshot: {snapshot}")


if __name__ == "__main__":
    test_persona_profiles_phase5()
    test_followup_delete_and_confirm_it()
    test_followup_close_it_and_memory_status()
    test_followup_rename_move_cancel_chain()
    test_bilingual_template_library_for_arabic_clarification()
    test_deterministic_anti_repetition()
    test_context_slot_recency_snapshot_and_followup_priority()
    route_command("memory clear")
    print("Phase 5 smoke tests passed.")