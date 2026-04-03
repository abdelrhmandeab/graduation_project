from datetime import datetime, timezone

from os_control.action_log import log_action
from os_control.adapter_result import to_router_tuple
from os_control.job_queue import job_queue_service


def _format_timestamp(ts):
    if ts is None:
        return "n/a"
    dt = datetime.fromtimestamp(float(ts), tz=timezone.utc)
    return dt.isoformat()


def _format_job(job):
    if not job:
        return "Job not found."
    return (
        f"id={job['id']} | status={job['status']} | attempts={job['attempts']} | "
        f"max_retries={job['max_retries']} | run_at={_format_timestamp(job['run_at'])} | "
        f"command={job['command_text']} | last_error={job['last_error'] or 'none'}"
    )


def handle(parsed):
    action = parsed.action
    args = parsed.args

    if action == "worker_start":
        return to_router_tuple(job_queue_service.start_result())
    if action == "worker_stop":
        job_queue_service.stop()
        return True, "Job queue worker stopped.", {}
    if action == "worker_status":
        running = job_queue_service.is_running()
        return True, f"Job queue worker running: {running}", {}

    if action == "enqueue":
        delay = int(args.get("delay_seconds", 0))
        command_text = args.get("command_text", "")
        return to_router_tuple(job_queue_service.enqueue_result(command_text, delay_seconds=delay))

    if action == "status":
        job = job_queue_service.status(args.get("job_id"))
        if not job:
            return False, "Job not found.", {"error_code": "not_found"}
        return True, _format_job(job), {}

    if action == "cancel":
        return to_router_tuple(job_queue_service.cancel_result(args.get("job_id")))

    if action == "retry":
        return to_router_tuple(
            job_queue_service.retry_result(
                args.get("job_id"),
                delay_seconds=args.get("delay_seconds", 0),
            )
        )

    if action == "list":
        jobs = job_queue_service.list(limit=args.get("limit", 10), status=args.get("status"))
        if not jobs:
            return True, "No jobs found.", {}
        return True, "\n".join(_format_job(job) for job in jobs), {}

    return False, "Unsupported job queue command.", {}
