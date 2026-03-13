import threading

from os_control.action_log import log_action
from os_control.file_ops import undo_last_action
from os_control.persistence import count_pending_rollback_actions


DISALLOWED_BATCH_INTENTS = {
    "BATCH_COMMAND",
    "JOB_QUEUE_COMMAND",
    "OS_CONFIRMATION",
    "OS_SYSTEM_COMMAND",
}


class BatchManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._planned_commands = []

    def plan(self):
        with self._lock:
            self._planned_commands = []
        log_action("batch_plan", "success", details={"count": 0})
        return True, "Batch plan initialized. Add commands with `batch add <command>`."

    def add(self, command_text):
        cmd = (command_text or "").strip()
        if not cmd:
            return False, "Batch command is empty."

        with self._lock:
            self._planned_commands.append(cmd)
            count = len(self._planned_commands)
        log_action("batch_add", "success", details={"command": cmd, "count": count})
        return True, f"Added step {count}: {cmd}"

    def status(self):
        with self._lock:
            count = len(self._planned_commands)
        return True, f"Batch has {count} planned command(s)."

    def preview(self, parser):
        with self._lock:
            commands = list(self._planned_commands)
        if not commands:
            return True, "Batch is empty."

        lines = ["Batch Preview"]
        for idx, command_text in enumerate(commands, start=1):
            parsed = parser(command_text)
            lines.append(
                f"{idx}. {command_text} -> {parsed.intent}"
                + (f" ({parsed.action})" if parsed.action else "")
            )
        return True, "\n".join(lines)

    def abort(self):
        with self._lock:
            previous = len(self._planned_commands)
            self._planned_commands = []
        log_action("batch_abort", "success", details={"cleared_count": previous})
        return True, f"Batch aborted. Cleared {previous} step(s)."

    def commit(self, parser, executor):
        with self._lock:
            commands = list(self._planned_commands)
            self._planned_commands = []

        if not commands:
            return False, "Batch is empty. Add commands first."

        for command_text in commands:
            parsed = parser(command_text)
            if parsed.intent in DISALLOWED_BATCH_INTENTS:
                log_action(
                    "batch_commit",
                    "failed",
                    details={"command": command_text, "reason": "disallowed_intent"},
                )
                return False, f"Command not allowed in batch: {command_text}"

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
                return (
                    False,
                    (
                        f"Batch failed at step {index}: {message}. "
                        f"Rolled back {rollback_done}/{created_rollback_actions} action(s)."
                    ),
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
        return True, f"Batch committed successfully. Executed {len(executed)} command(s)."


batch_manager = BatchManager()
