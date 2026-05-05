from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

import httpx

from core.config import LLM_MODEL, LLM_OLLAMA_BASE_URL, LLM_TIMEOUT_SECONDS
from core.logger import logger

_OLLAMA_BASE_URL = str(LLM_OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
_CHAT_ENDPOINT = f"{_OLLAMA_BASE_URL}/api/chat"


def build_default_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "open_app",
                "description": "Open a desktop application by name or alias.",
                "parameters": {
                    "type": "object",
                    "properties": {"app_name": {"type": "string"}},
                    "required": ["app_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "close_app",
                "description": "Close a desktop application by name or alias.",
                "parameters": {
                    "type": "object",
                    "properties": {"app_name": {"type": "string"}},
                    "required": ["app_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search the local filesystem for a filename or partial match.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {"type": "string"},
                        "search_path": {"type": "string"},
                    },
                    "required": ["filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "open_folder",
                "description": "Open or list a folder path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_timer",
                "description": "Set a timer for a number of seconds.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seconds": {"type": "integer"},
                        "label": {"type": "string"},
                    },
                    "required": ["seconds"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web with the browser.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_volume",
                "description": "Set the system volume percentage.",
                "parameters": {
                    "type": "object",
                    "properties": {"level": {"type": "integer"}},
                    "required": ["level"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "adjust_volume",
                "description": "Increase or decrease the system volume by a delta.",
                "parameters": {
                    "type": "object",
                    "properties": {"delta": {"type": "integer"}},
                    "required": ["delta"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "set_brightness",
                "description": "Set the screen brightness percentage.",
                "parameters": {
                    "type": "object",
                    "properties": {"level": {"type": "integer"}},
                    "required": ["level"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "adjust_brightness",
                "description": "Increase or decrease screen brightness by a delta.",
                "parameters": {
                    "type": "object",
                    "properties": {"delta": {"type": "integer"}},
                    "required": ["delta"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lock_screen",
                "description": "Lock the workstation.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sleep_system",
                "description": "Put the computer to sleep.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "screenshot",
                "description": "Capture a screenshot of the primary screen.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]


def _resolve_model_name(model_name: Optional[str] = None) -> str:
    candidate = str(model_name or LLM_MODEL or "").strip()
    return candidate or "qwen3:4b"


def _normalize_tool_call(call: Dict[str, Any]) -> Dict[str, Any]:
    function = call.get("function") or {}
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except Exception:
            arguments = {"text": arguments}
    return {
        "name": str(function.get("name") or call.get("name") or "").strip(),
        "arguments": dict(arguments or {}),
    }


def call_tool_tier(
    user_text: str,
    *,
    model_name: Optional[str] = None,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    payload = {
        "model": _resolve_model_name(model_name),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a command router. Pick only function calls that help execute "
                    "the user's request. Prefer 1-3 tool calls. Do not answer in prose."
                ),
            },
            {"role": "user", "content": str(user_text or "")},
        ],
        "stream": False,
        "tools": list(tools or build_default_tools()),
        "options": {"temperature": 0.1},
    }

    try:
        response = httpx.post(_CHAT_ENDPOINT, json=payload, timeout=LLM_TIMEOUT_SECONDS)
    except Exception as exc:
        logger.debug("Tool-calling request failed: %s", exc)
        return {"tool_calls": [], "message": "", "error": str(exc)}

    if response.status_code != 200:
        logger.debug("Tool-calling request returned %s: %s", response.status_code, response.text[:200])
        return {"tool_calls": [], "message": "", "error": response.text[:200]}

    try:
        payload_json = response.json()
    except Exception as exc:
        logger.debug("Tool-calling response was not JSON: %s", exc)
        return {"tool_calls": [], "message": "", "error": str(exc)}

    message = payload_json.get("message") or {}
    tool_calls = [_normalize_tool_call(call) for call in (message.get("tool_calls") or [])]
    tool_calls = [call for call in tool_calls if call.get("name")]
    content = str(message.get("content") or "").strip()
    return {"tool_calls": tool_calls, "message": content, "raw": payload_json}


def tool_calls_to_parsed_commands(tool_calls: Iterable[Dict[str, Any]], raw_text: str):
    from core.command_parser import ParsedCommand

    parsed_commands = []
    normalized = str(raw_text or "").strip()
    for call in tool_calls:
        name = str(call.get("name") or "").strip().lower()
        args = dict(call.get("arguments") or {})
        if name == "open_app":
            parsed_commands.append(ParsedCommand("OS_APP_OPEN", raw_text, normalized, args={"app_name": args.get("app_name", "")}))
        elif name == "close_app":
            parsed_commands.append(ParsedCommand("OS_APP_CLOSE", raw_text, normalized, args={"app_name": args.get("app_name", "")}))
        elif name == "search_files":
            parsed_commands.append(
                ParsedCommand(
                    "OS_FILE_SEARCH",
                    raw_text,
                    normalized,
                    args={
                        "filename": args.get("filename", ""),
                        "search_path": args.get("search_path", ""),
                    },
                )
            )
        elif name == "open_folder":
            parsed_commands.append(
                ParsedCommand(
                    "OS_FILE_NAVIGATION",
                    raw_text,
                    normalized,
                    action="list_directory",
                    args={"path": args.get("path", "")},
                )
            )
        elif name == "set_timer":
            parsed_commands.append(
                ParsedCommand(
                    "OS_TIMER",
                    raw_text,
                    normalized,
                    action="set",
                    args={"seconds": args.get("seconds"), "label": args.get("label", "Timer")},
                )
            )
        elif name == "web_search":
            parsed_commands.append(
                ParsedCommand(
                    "OS_SYSTEM_COMMAND",
                    raw_text,
                    normalized,
                    args={"action_key": "browser_search_web", "search_query": args.get("query", "")},
                )
            )
        elif name == "set_volume":
            parsed_commands.append(
                ParsedCommand(
                    "OS_SYSTEM_COMMAND",
                    raw_text,
                    normalized,
                    args={"action_key": "volume_set", "volume_level": args.get("level")},
                )
            )
        elif name == "adjust_volume":
            delta = int(args.get("delta") or 0)
            action_key = "volume_up" if delta >= 0 else "volume_down"
            parsed_commands.append(ParsedCommand("OS_SYSTEM_COMMAND", raw_text, normalized, args={"action_key": action_key, "volume_delta": delta}))
        elif name == "set_brightness":
            parsed_commands.append(
                ParsedCommand(
                    "OS_SYSTEM_COMMAND",
                    raw_text,
                    normalized,
                    args={"action_key": "brightness_set", "brightness_level": args.get("level")},
                )
            )
        elif name == "adjust_brightness":
            delta = int(args.get("delta") or 0)
            action_key = "brightness_up" if delta >= 0 else "brightness_down"
            parsed_commands.append(ParsedCommand("OS_SYSTEM_COMMAND", raw_text, normalized, args={"action_key": action_key, "brightness_delta": delta}))
        elif name == "lock_screen":
            parsed_commands.append(ParsedCommand("OS_SYSTEM_COMMAND", raw_text, normalized, args={"action_key": "lock"}))
        elif name == "sleep_system":
            parsed_commands.append(ParsedCommand("OS_SYSTEM_COMMAND", raw_text, normalized, args={"action_key": "sleep"}))
        elif name == "screenshot":
            parsed_commands.append(ParsedCommand("OS_SYSTEM_COMMAND", raw_text, normalized, args={"action_key": "screenshot"}))

    return parsed_commands
