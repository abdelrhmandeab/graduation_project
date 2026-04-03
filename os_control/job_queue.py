import threading
import time

from core.config import JOB_MAX_RETRIES_DEFAULT
from core.logger import logger
from os_control.action_log import log_action
from os_control.adapter_result import failure_result, success_result, to_legacy_pair
from os_control.persistence import (
    cancel_job,
    claim_due_job,
    create_job,
    get_job,
    list_jobs,
    mark_job_failed,
    mark_job_succeeded,
    retry_job,
)


class JobQueueService:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._executor = None
        self._poll_seconds = 1.0

    def configure_executor(self, executor):
        with self._lock:
            self._executor = executor

    def start_result(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return failure_result(
                    "Job queue worker already running.",
                    error_code="already_running",
                )
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="jarvis-job-queue",
                daemon=True,
            )
            self._thread.start()
        logger.info("Job queue worker started")
        return success_result("Job queue worker started.", debug_info={"running": True})

    def stop(self):
        with self._lock:
            thread = self._thread
            self._thread = None
        self._stop_event.set()
        if thread and thread.is_alive():
            thread.join(timeout=3)
        logger.info("Job queue worker stopped")

    def is_running(self):
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def enqueue_result(self, command_text, delay_seconds=0, max_retries=JOB_MAX_RETRIES_DEFAULT):
        command = (command_text or "").strip()
        if not command:
            return failure_result("Job command is empty.", error_code="invalid_input")

        delay = max(0, int(delay_seconds))
        retries = max(0, int(max_retries))
        run_at = time.time() + delay
        job = create_job(command, run_at=run_at, max_retries=retries)
        log_action(
            "job_enqueue",
            "success",
            details={
                "job_id": job["id"],
                "command": command,
                "delay_seconds": delay,
                "max_retries": retries,
            },
        )
        self.start_result()
        return success_result(
            f"Queued job #{job['id']} (runs in {delay}s).",
            debug_info={"job": job},
        )

    def list(self, limit=10, status=None):
        return list_jobs(limit=limit, status=status)

    def status(self, job_id):
        return get_job(job_id)

    def cancel_result(self, job_id):
        job = cancel_job(job_id)
        if not job:
            return failure_result("Job not found.", error_code="not_found")
        log_action(
            "job_cancel",
            "success",
            details={"job_id": job["id"], "status": job["status"]},
        )
        return success_result(
            f"Job #{job['id']} status: {job['status']}.",
            debug_info={"job": job},
        )

    def retry_result(self, job_id, delay_seconds=0):
        job = retry_job(job_id, delay_seconds=delay_seconds)
        if not job:
            return failure_result("Job not found.", error_code="not_found")
        log_action(
            "job_retry",
            "success",
            details={"job_id": job["id"], "delay_seconds": max(0, int(delay_seconds))},
        )
        return success_result(
            f"Job #{job['id']} re-queued.",
            debug_info={"job": job},
        )

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                self._run_next_due_job()
            except Exception as exc:
                logger.error("Job queue worker loop error: %s", exc)
            self._stop_event.wait(self._poll_seconds)

    def _run_next_due_job(self):
        job = claim_due_job()
        if not job:
            return

        with self._lock:
            executor = self._executor
        if not executor:
            mark_job_failed(job["id"], "No executor configured for job queue.", requeue=False)
            return

        try:
            success, message = executor(job["command_text"])
        except Exception as exc:
            success = False
            message = f"Unhandled job exception: {exc}"

        if success:
            mark_job_succeeded(job["id"])
            log_action(
                "job_execute",
                "success",
                details={"job_id": job["id"], "command": job["command_text"]},
            )
            return

        attempts = int(job["attempts"])
        retries = int(job["max_retries"])
        if attempts <= retries:
            backoff_seconds = min(30, 2 ** max(0, attempts - 1))
            next_run_at = time.time() + backoff_seconds
            mark_job_failed(
                job["id"],
                message,
                requeue=True,
                next_run_at=next_run_at,
            )
            log_action(
                "job_execute",
                "retrying",
                details={
                    "job_id": job["id"],
                    "command": job["command_text"],
                    "attempts": attempts,
                    "max_retries": retries,
                    "next_delay_seconds": backoff_seconds,
                },
                error=message,
            )
            return

        mark_job_failed(job["id"], message, requeue=False)
        log_action(
            "job_execute",
            "failed",
            details={
                "job_id": job["id"],
                "command": job["command_text"],
                "attempts": attempts,
                "max_retries": retries,
            },
            error=message,
        )

    # Legacy tuple compatibility
    def start(self):
        return to_legacy_pair(self.start_result())

    def enqueue(self, command_text, delay_seconds=0, max_retries=JOB_MAX_RETRIES_DEFAULT):
        result = self.enqueue_result(command_text, delay_seconds=delay_seconds, max_retries=max_retries)
        ok, message = to_legacy_pair(result)
        job = (result.get("debug_info") or {}).get("job") if isinstance(result, dict) else None
        return ok, message, job

    def cancel(self, job_id):
        result = self.cancel_result(job_id)
        ok, message = to_legacy_pair(result)
        job = (result.get("debug_info") or {}).get("job") if isinstance(result, dict) else None
        return ok, message, job

    def retry(self, job_id, delay_seconds=0):
        result = self.retry_result(job_id, delay_seconds=delay_seconds)
        ok, message = to_legacy_pair(result)
        job = (result.get("debug_info") or {}).get("job") if isinstance(result, dict) else None
        return ok, message, job


job_queue_service = JobQueueService()
