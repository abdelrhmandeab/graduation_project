import threading

from os_control.action_log import log_action
from os_control.adapter_result import failure_result, success_result, to_legacy_pair
from os_control.file_ops import undo_last_action
from os_control.persistence import count_pending_rollback_actions


DISALLOWED_BATCH_INTENTS = {
    "BATCH_COMMAND",
    "JOB_QUEUE_COMMAND",
    "OS_CONFIRMATION",
    "OS_SYSTEM_COMMAND",
    "OS_APP_CLOSE",
}


class BatchManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._planned_commands = []

    def plan_result(self):
        with self._lock:
            self._planned_commands = []
        log_action("batch_plan", "success", details={"count": 0})
        return success_result(
            "Batch plan initialized. Add commands with `batch add <command>`.",
            debug_info={"count": 0},
        )

    def add_result(self, command_text):
        cmd = (command_text or "").strip()
        if not cmd:
            return failure_result("Batch command is empty.", error_code="invalid_input")

        with self._lock:
            self._planned_commands.append(cmd)
            count = len(self._planned_commands)
        log_action("batch_add", "success", details={"command": cmd, "count": count})
        return success_result(
            f"Added step {count}: {cmd}",
            debug_info={"command": cmd, "count": count},
        )

    def status_result(self):
        with self._lock:
            count = len(self._planned_commands)
        return success_result(
            f"Batch has {count} planned command(s).",
            debug_info={"count": count},
        )

    def preview_result(self, parser):
        with self._lock:
            commands = list(self._planned_commands)
        if not commands:
            return success_result("Batch is empty.", debug_info={"count": 0})

        lines = ["Batch Preview"]
        preview_items = []
        for idx, command_text in enumerate(commands, start=1):
            parsed = parser(command_text)
            rendered = (
                f"{idx}. {command_text} -> {parsed.intent}"
                + (f" ({parsed.action})" if parsed.action else "")
            )
            lines.append(rendered)
            preview_items.append(
                {
                    "step": idx,
                    "command": command_text,
                    "intent": parsed.intent,
                    "action": parsed.action,
                }
            )
        return success_result("\n".join(lines), debug_info={"count": len(commands), "preview": preview_items})

    def abort_result(self):
        with self._lock:
            previous = len(self._planned_commands)
            self._planned_commands = []
        log_action("batch_abort", "success", details={"cleared_count": previous})
        return success_result(
            f"Batch aborted. Cleared {previous} step(s).",
            debug_info={"cleared_count": previous},
        )

    def commit_result(self, parser, executor):
        with self._lock:
            commands = list(self._planned_commands)
            self._planned_commands = []

        if not commands:
            return failure_result("Batch is empty. Add commands first.", error_code="invalid_state")

        for command_text in commands:
            parsed = parser(command_text)
            if parsed.intent in DISALLOWED_BATCH_INTENTS:
                log_action(
                    "batch_commit",
                    "failed",
                    details={"command": command_text, "reason": "disallowed_intent"},
                )
                return failure_result(
                    f"Command not allowed in batch: {command_text}",
                    error_code="policy_blocked",
                    debug_info={"command": command_text, "reason": "disallowed_intent"},
                )

        created_rollback_actions = 0
        previous_rollback_count = count_pending_rollback_actions()
        executed = []

        for index, command_text in enumerate(commands, start=1):
            success, message = executor(command_text)
            if not success:
                rollback_done = 0
                for _ in range(created_rollback_actions):
                    undo_ok, _undo_msg = undo_last_action()
                    if undo_ok:
                        rollback_done += 1

                log_action(
                    "batch_commit",
                    "failed",
                    details={
                        "failed_step": index,
                        "failed_command": command_text,
                        "executed_count": len(executed),
                        "rollback_requested": created_rollback_actions,
                        "rollback_done": rollback_done,
                    },
                    error=message,
                )
                return failure_result(
                    (
                        f"Batch failed at step {index}: {message}. "
                        f"Rolled back {rollback_done}/{created_rollback_actions} action(s)."
                    ),
                    error_code="execution_failed",
                    debug_info={
                        "failed_step": index,
                        "failed_command": command_text,
                        "executed_count": len(executed),
                        "rollback_requested": created_rollback_actions,
                        "rollback_done": rollback_done,
                    },
                )

            executed.append(command_text)
            pending_rollback_count = count_pending_rollback_actions()
            if pending_rollback_count > previous_rollback_count:
                created_rollback_actions += pending_rollback_count - previous_rollback_count
            previous_rollback_count = pending_rollback_count

        log_action(
            "batch_commit",
            "success",
            details={"executed_count": len(executed), "rollback_actions": created_rollback_actions},
        )
        return success_result(
            f"Batch committed successfully. Executed {len(executed)} command(s).",
            debug_info={
                "executed_count": len(executed),
                "rollback_actions": created_rollback_actions,
            },
        )

    # Legacy tuple compatibility
    def plan(self):
        return to_legacy_pair(self.plan_result())

    def add(self, command_text):
        return to_legacy_pair(self.add_result(command_text))

    def status(self):
        return to_legacy_pair(self.status_result())

    def preview(self, parser):
        return to_legacy_pair(self.preview_result(parser))

    def abort(self):
        return to_legacy_pair(self.abort_result())

    def commit(self, parser, executor):
        return to_legacy_pair(self.commit_result(parser, executor))


batch_manager = BatchManager()


