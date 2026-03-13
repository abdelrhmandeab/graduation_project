import re
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.command_router import route_command
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


def test_navigation_and_rollback():
    policy_engine.set_profile("normal")
    with _workspace_tempdir() as tmp:
        response = route_command(f"go to {tmp}")
        _assert("Current directory set to" in response, f"Unexpected response: {response}")

        response = route_command("create folder demo")
        _assert("Created directory" in response, f"Unexpected response: {response}")
        _assert((Path(tmp) / "demo").exists(), "Directory should exist after creation")

        response = route_command("undo")
        _assert("Rollback completed" in response, f"Unexpected response: {response}")
        _assert(not (Path(tmp) / "demo").exists(), "Directory should be removed after rollback")


def test_delete_and_undo():
    policy_engine.set_profile("normal")
    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        target = Path(tmp) / "note.txt"
        target.write_text("hello", encoding="utf-8")

        response = route_command("delete note.txt")
        _assert("Deleted" in response, f"Unexpected response: {response}")
        _assert(not target.exists(), "File should be removed after delete")

        response = route_command("undo")
        _assert("Rollback completed" in response, f"Unexpected response: {response}")
        _assert(target.exists(), "File should be restored after rollback")


def test_system_confirmation_flow():
    response = route_command("shutdown computer")
    _assert("Confirmation required" in response, f"Unexpected response: {response}")

    token_match = re.search(r"confirm\s+([0-9a-f]{6})", response, flags=re.IGNORECASE)
    _assert(token_match is not None, "Confirmation token was not present")
    token = token_match.group(1)

    missing_factor = route_command(f"confirm {token}")
    _assert("Second factor required" in missing_factor, f"Unexpected response: {missing_factor}")

    bad_factor = route_command(f"confirm {token} 0000")
    _assert("verification failed" in bad_factor.lower(), f"Unexpected response: {bad_factor}")

    confirm_response = route_command(f"confirm {token} 2468")
    _assert("Blocked by configuration" in confirm_response, f"Unexpected response: {confirm_response}")


def test_drive_listing():
    response = route_command("list drives")
    _assert("\\" in response or "failed" in response.lower(), f"Unexpected response: {response}")


def test_audit_verify_and_policy():
    response = route_command("verify audit log")
    _assert("Audit chain" in response, f"Unexpected response: {response}")

    response = route_command("policy profile demo")
    _assert("Policy profile set to: demo" in response, f"Unexpected response: {response}")

    response = route_command("policy status")
    _assert("profile: demo" in response, f"Unexpected response: {response}")

    route_command("policy profile normal")


def test_batch_commit():
    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        _assert("initialized" in route_command("batch plan").lower(), "Batch plan failed")
        _assert("Added step 1" in route_command("batch add create folder one"), "Batch add #1 failed")
        _assert("Added step 2" in route_command("batch add create folder two"), "Batch add #2 failed")
        preview = route_command("batch preview")
        _assert("Batch Preview" in preview, f"Unexpected preview: {preview}")
        commit = route_command("batch commit")
        _assert("Batch committed successfully" in commit, f"Unexpected commit: {commit}")
        _assert((Path(tmp) / "one").exists(), "Folder one was not created")
        _assert((Path(tmp) / "two").exists(), "Folder two was not created")


def test_index_and_job_queue():
    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        target = Path(tmp) / "alpha_test.txt"
        target.write_text("demo", encoding="utf-8")

        refresh = route_command(f"index refresh in {tmp}")
        _assert("Refreshed index" in refresh, f"Unexpected response: {refresh}")

        indexed = route_command(f"indexed find alpha in {tmp}")
        _assert(str(target) in indexed, f"Indexed search missing target: {indexed}")

        queue_response = route_command("queue job create folder queued_job")
        match = re.search(r"#(\d+)", queue_response)
        _assert(match is not None, f"Job id not found: {queue_response}")
        job_id = int(match.group(1))

        status_text = ""
        for _ in range(10):
            time.sleep(0.5)
            status_text = route_command(f"job status {job_id}")
            if "status=succeeded" in status_text:
                break
        _assert("status=succeeded" in status_text, f"Job did not finish successfully: {status_text}")
        _assert((Path(tmp) / "queued_job").exists(), "Queued job did not create folder")


if __name__ == "__main__":
    test_navigation_and_rollback()
    test_delete_and_undo()
    test_system_confirmation_flow()
    test_drive_listing()
    test_audit_verify_and_policy()
    test_batch_commit()
    test_index_and_job_queue()
    print("Phase 3 smoke tests passed.")
