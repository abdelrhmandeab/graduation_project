"""Event helpers for the Jarvis UI bridge."""

from __future__ import annotations

import json


# Engine -> UI events
EVENT_STATE_CHANGED = "state_changed"
EVENT_PARTIAL_TRANSCRIPT = "partial_transcript"
EVENT_FINAL_TRANSCRIPT = "final_transcript"
EVENT_RESPONSE = "response"
EVENT_AMPLITUDE = "amplitude"
EVENT_METRICS = "metrics"
EVENT_HEALTH = "health"
EVENT_ERROR = "error"
EVENT_CONFIG = "config"

# UI -> engine commands
COMMAND_TEXT = "text_command"
COMMAND_MUTE_TOGGLE = "mute_toggle"
COMMAND_SETTING_UPDATE = "setting_update"
COMMAND_FEATURE_FLAG = "feature_flag"
COMMAND_CONFIG_REQUEST = "config_request"
COMMAND_HEALTH_REQUEST = "health_request"


def make_event(event_type, **fields) -> dict:
    event = {"type": str(event_type or "")}
    event.update(fields)
    return event


def to_json(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False)
