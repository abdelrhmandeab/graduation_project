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


def test_batch_transaction_rolls_back_on_failure():
    policy_engine.set_profile("normal")
    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")
        route_command("batch plan")
        route_command("batch add create folder before_one")
        route_command("batch add create folder before_two")
        route_command(r"batch add go to C:\Windows\System32\config")
        response = route_command("batch commit")
        _assert("Batch failed at step 3" in response, f"Unexpected response: {response}")
        _assert(
            not (Path(tmp) / "before_one").exists(),
            "Transactional rollback should remove before_one",
        )
        _assert(
            not (Path(tmp) / "before_two").exists(),
            "Transactional rollback should remove before_two",
        )


def test_job_cancel_and_retry():
    policy_engine.set_profile("normal")
    with _workspace_tempdir() as tmp:
        route_command(f"go to {tmp}")

        queue_response = route_command("queue job in 10 create folder delayed_retry")
        match = re.search(r"#(\d+)", queue_response)
        _assert(match is not None, f"Job id missing: {queue_response}")
        job_id = int(match.group(1))

        cancel_response = route_command(f"job cancel {job_id}")
        _assert("status: canceled" in cancel_response.lower(), f"Unexpected response: {cancel_response}")

        retry_response = route_command(f"job retry {job_id}")
        _assert("re-queued" in retry_response.lower(), f"Unexpected response: {retry_response}")

        status_text = ""
        for _ in range(12):
            time.sleep(0.5)
            status_text = route_command(f"job status {job_id}")
            if "status=succeeded" in status_text:
                break

        _assert("status=succeeded" in status_text, f"Job did not succeed after retry: {status_text}")
        _assert((Path(tmp) / "delayed_retry").exists(), "Retried job did not execute command")


if __name__ == "__main__":
    test_batch_transaction_rolls_back_on_failure()
    test_job_cancel_and_retry()
    print("Phase 3 advanced tests passed.")
