"""Multi-Step Action Planner — Task 3.3.

Executes a list of tool calls produced by ``llm/tool_caller.py`` in dependency
order, resolving ``{result_N}`` placeholders so the output of one step can
flow into the arguments of a later step.

Example multi-step tool call list produced by the LLM:

    [
        {"name": "search_files", "arguments": {"filename": "report"}},
        {"name": "open_folder",  "arguments": {"path": "{result_0}"}},
    ]

Step 0 dispatches ``search_files``; when it succeeds its ``dispatch_meta``
(containing the found ``path``) is stored.  Step 1 resolves ``{result_0}``
against that stored result before dispatching ``open_folder``.

Design notes
------------
* The ``executor`` callable is injected by the caller (``command_router.py``)
  to avoid a circular import.  It must accept a ``ParsedCommand`` and return
  ``(success: bool, message: str, meta: dict)``.
* ``{result_N}`` references are resolved BEFORE ``tool_calls_to_parsed_commands``
  converts a call dict to a ``ParsedCommand``, so the resolved value lands in
  the right field of the dataclass.
* When ``key == "path"`` and the resolved value is a file (not a directory),
  ``os.path.dirname`` is applied automatically, enabling the common
  "find file → open its folder" pattern without the LLM needing to know the
  distinction.
* A failed step stops the chain immediately; the partial-success response
  reports what succeeded and what failed in both English and Egyptian Arabic.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.logger import logger

# Fast prefix check avoids regex overhead on the common no-reference case.
_RESULT_REF_PREFIX = "{result_"


class ActionPlanner:
    def __init__(self, executor: Optional[Callable] = None) -> None:
        # executor: (ParsedCommand) → (bool, str, dict)
        # Injected at call-site; never imported here to prevent circular deps.
        self._executor = executor

    # ── Public API ────────────────────────────────────────────────────────────

    def plan_and_execute(
        self,
        tool_calls: List[Dict[str, Any]],
        user_text: str,
        language: str,
    ) -> Tuple[bool, str, List[Dict[str, Any]]]:
        """Execute tool calls in order, passing results between steps.

        Args:
            tool_calls: raw dicts from ``call_tool_tier`` (name + arguments).
            user_text:  original user utterance; used as ParsedCommand.raw.
            language:   "en" or "ar" — controls response phrasing.

        Returns:
            (success, response_text, result_details)
        """
        results: List[Dict[str, Any]] = []

        for call in tool_calls:
            tool_name = str(call.get("name") or "").strip().lower()
            raw_args = dict(call.get("arguments") or {})

            resolved_args = self._resolve_references(raw_args, results)

            try:
                ok, message, data = self._execute_single(
                    tool_name, resolved_args, user_text
                )
                results.append(
                    {
                        "name": tool_name,
                        "success": ok,
                        "message": str(message or ""),
                        "data": data,
                    }
                )
                if not ok:
                    logger.info(
                        "ActionPlanner: step '%s' failed (%s); stopping chain.",
                        tool_name,
                        message,
                    )
                    return (
                        False,
                        self._build_partial_response(results, language),
                        results,
                    )
            except Exception as exc:
                logger.warning(
                    "ActionPlanner: step '%s' raised an unexpected error: %s",
                    tool_name,
                    exc,
                )
                results.append(
                    {
                        "name": tool_name,
                        "success": False,
                        "message": str(exc),
                        "data": {},
                    }
                )
                return (
                    False,
                    self._build_partial_response(results, language),
                    results,
                )

        return True, self._build_success_response(results, language), results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_references(
        self,
        args: Dict[str, Any],
        previous_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Replace ``{result_N}`` placeholders with data from completed steps.

        When ``key == "path"`` and the resolved value is a file rather than a
        directory, the parent directory is returned instead — enabling the
        "search_files → open_folder" pattern transparently.
        """
        resolved: Dict[str, Any] = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith(_RESULT_REF_PREFIX):
                try:
                    idx = int(value.split("_")[1].rstrip("}"))
                except (IndexError, ValueError):
                    resolved[key] = value
                    continue

                if idx < len(previous_results) and previous_results[idx]["success"]:
                    prev_data = previous_results[idx].get("data") or {}
                    raw_value = (
                        prev_data.get("path")
                        or prev_data.get("result")
                        or str(prev_data)
                    )
                    raw_str = str(raw_value)
                    if key == "path" and raw_str and os.path.isfile(raw_str):
                        resolved[key] = os.path.dirname(raw_str)
                    else:
                        resolved[key] = raw_str
                else:
                    # Reference unavailable — leave placeholder unchanged so
                    # the dispatch layer can report a meaningful error.
                    resolved[key] = value
            else:
                resolved[key] = value
        return resolved

    def _execute_single(
        self,
        tool_name: str,
        args: Dict[str, Any],
        user_text: str,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """Convert one resolved tool-call dict to a ParsedCommand and dispatch."""
        from llm.tool_caller import tool_calls_to_parsed_commands

        parsed_list = tool_calls_to_parsed_commands(
            [{"name": tool_name, "arguments": args}],
            raw_text=str(user_text or ""),
        )
        if not parsed_list:
            return False, f"Unknown tool: {tool_name}", {}

        parsed = parsed_list[0]
        if self._executor is None:
            return False, "ActionPlanner: no executor configured.", {}

        try:
            result = self._executor(parsed)
        except Exception as exc:
            return False, str(exc), {}

        if isinstance(result, tuple) and len(result) >= 2:
            ok = bool(result[0])
            message = str(result[1] or "")
            data = dict(result[2]) if len(result) >= 3 and isinstance(result[2], dict) else {}
        else:
            ok = bool(result)
            message = ""
            data = {}

        logger.debug(
            "ActionPlanner: tool=%s success=%s message=%s",
            tool_name,
            ok,
            message[:80],
        )
        return ok, message, data

    def _build_partial_response(
        self,
        results: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """Bilingual response for partial completion (at least one step failed)."""
        succeeded = [r for r in results if r["success"]]
        failed = [r for r in results if not r["success"]]

        if language == "ar":
            parts: List[str] = []
            done_msgs = " و".join(r["message"] for r in succeeded if r["message"])
            if done_msgs:
                parts.append(f"عملت {done_msgs}")
            fail_msg = failed[-1]["message"] if failed and failed[-1]["message"] else ""
            if fail_msg:
                parts.append(f"بس {fail_msg}")
            return (". ".join(parts) + ".") if parts else "حصل مشكلة."
        else:
            parts = []
            done_msgs = ", ".join(r["message"] for r in succeeded if r["message"])
            if done_msgs:
                parts.append(f"Done: {done_msgs}")
            fail_msg = failed[-1]["message"] if failed and failed[-1]["message"] else ""
            if fail_msg:
                parts.append(f"But: {fail_msg}")
            return " ".join(parts) if parts else "An error occurred."

    def _build_success_response(
        self,
        results: List[Dict[str, Any]],
        language: str,
    ) -> str:
        """Response when all steps completed successfully."""
        messages = [r["message"] for r in results if r.get("message")]
        if not messages:
            return "تمام." if language == "ar" else "Done."
        return "\n".join(messages)
