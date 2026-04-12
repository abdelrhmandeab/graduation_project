import re
from difflib import SequenceMatcher

from core.config import (
    ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS,
    CONFIRMATION_TIMEOUT_SECONDS,
    SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE,
)
from core.logger import logger
from core.response_templates import format_confirmation_prompt
from os_control.action_log import log_action
from os_control.adapter_result import (
    confirmation_result,
    failure_result,
    success_result,
    to_legacy_pair,
)
from os_control.confirmation import confirmation_manager
from os_control.policy import policy_engine
from os_control.powershell_bridge import run_template
from os_control.risk_policy import risk_tier_for_system

SYSTEM_COMMANDS = {
    "shutdown": {
        "template": "shutdown",
        "description": "Shut down this computer",
        "destructive": True,
    },
    "restart": {
        "template": "restart",
        "description": "Restart this computer",
        "destructive": True,
    },
    "sleep": {
        "template": "sleep",
        "description": "Put this computer to sleep",
        "destructive": False,
        "requires_confirmation": True,
    },
    "lock": {
        "template": "lock",
        "description": "Lock this computer",
        "destructive": False,
        "requires_confirmation": True,
    },
    "logoff": {
        "template": "logoff",
        "description": "Log off current user",
        "destructive": True,
    },
    "volume_up": {
        "template": "volume_up",
        "description": "Increase system volume",
        "destructive": False,
        "requires_confirmation": False,
    },
    "volume_down": {
        "template": "volume_down",
        "description": "Decrease system volume",
        "destructive": False,
        "requires_confirmation": False,
    },
    "volume_mute": {
        "template": "volume_mute",
        "description": "Toggle mute",
        "destructive": False,
        "requires_confirmation": False,
    },
    "volume_set": {
        "template": "volume_set",
        "description": "Set system volume",
        "destructive": False,
        "requires_confirmation": False,
    },
    "brightness_up": {
        "template": "brightness_up",
        "description": "Increase screen brightness",
        "destructive": False,
        "requires_confirmation": False,
    },
    "brightness_down": {
        "template": "brightness_down",
        "description": "Decrease screen brightness",
        "destructive": False,
        "requires_confirmation": False,
    },
    "brightness_set": {
        "template": "brightness_set",
        "description": "Set screen brightness",
        "destructive": False,
        "requires_confirmation": False,
    },
    "wifi_on": {
        "template": "wifi_on",
        "description": "Enable Wi-Fi",
        "destructive": False,
        "requires_confirmation": False,
    },
    "wifi_off": {
        "template": "wifi_off",
        "description": "Disable Wi-Fi",
        "destructive": False,
        "requires_confirmation": False,
    },
    "bluetooth_on": {
        "template": "bluetooth_on",
        "description": "Enable Bluetooth",
        "destructive": False,
        "requires_confirmation": False,
    },
    "bluetooth_off": {
        "template": "bluetooth_off",
        "description": "Disable Bluetooth",
        "destructive": False,
        "requires_confirmation": False,
    },
    "notifications_on": {
        "template": "notifications_on",
        "description": "Enable desktop notifications",
        "destructive": False,
        "requires_confirmation": False,
    },
    "notifications_off": {
        "template": "notifications_off",
        "description": "Disable desktop notifications",
        "destructive": False,
        "requires_confirmation": False,
    },
    "screenshot": {
        "template": "screenshot",
        "description": "Take a screenshot",
        "destructive": False,
        "requires_confirmation": False,
    },
    "empty_recycle_bin": {
        "template": "empty_recycle_bin",
        "description": "Empty recycle bin",
        "destructive": True,
    },
    "list_processes": {
        "template": "list_processes",
        "description": "Show running processes",
        "destructive": False,
        "requires_confirmation": False,
    },
    "focus_window": {
        "template": "focus_window",
        "description": "Focus a window",
        "destructive": False,
        "requires_confirmation": False,
    },
    "window_maximize": {
        "template": "window_maximize",
        "description": "Maximize active window",
        "destructive": False,
        "requires_confirmation": False,
    },
    "window_minimize": {
        "template": "window_minimize",
        "description": "Minimize active window",
        "destructive": False,
        "requires_confirmation": False,
    },
    "window_snap_left": {
        "template": "window_snap_left",
        "description": "Snap active window to left",
        "destructive": False,
        "requires_confirmation": False,
    },
    "window_snap_right": {
        "template": "window_snap_right",
        "description": "Snap active window to right",
        "destructive": False,
        "requires_confirmation": False,
    },
    "window_next": {
        "template": "window_next",
        "description": "Switch to next window",
        "destructive": False,
        "requires_confirmation": False,
    },
    "window_close_active": {
        "template": "window_close_active",
        "description": "Close active window",
        "destructive": False,
        "requires_confirmation": False,
    },
    "media_play_pause": {
        "template": "media_play_pause",
        "description": "Play or pause media",
        "destructive": False,
        "requires_confirmation": False,
    },
    "media_next_track": {
        "template": "media_next_track",
        "description": "Next media track",
        "destructive": False,
        "requires_confirmation": False,
    },
    "media_previous_track": {
        "template": "media_previous_track",
        "description": "Previous media track",
        "destructive": False,
        "requires_confirmation": False,
    },
    "media_stop": {
        "template": "media_stop",
        "description": "Stop media playback",
        "destructive": False,
        "requires_confirmation": False,
    },
    "media_seek_forward": {
        "template": "media_seek_forward",
        "description": "Seek media forward",
        "destructive": False,
        "requires_confirmation": False,
    },
    "media_seek_backward": {
        "template": "media_seek_backward",
        "description": "Seek media backward",
        "destructive": False,
        "requires_confirmation": False,
    },
    "browser_new_tab": {
        "template": "browser_new_tab",
        "description": "Open browser new tab",
        "destructive": False,
        "requires_confirmation": False,
    },
    "browser_close_tab": {
        "template": "browser_close_tab",
        "description": "Close active browser tab",
        "destructive": False,
        "requires_confirmation": False,
    },
    "browser_back": {
        "template": "browser_back",
        "description": "Browser back",
        "destructive": False,
        "requires_confirmation": False,
    },
    "browser_forward": {
        "template": "browser_forward",
        "description": "Browser forward",
        "destructive": False,
        "requires_confirmation": False,
    },
    "browser_open_url": {
        "template": "browser_open_url",
        "description": "Open website URL",
        "destructive": False,
        "requires_confirmation": False,
    },
    "browser_search_web": {
        "template": "browser_search_web",
        "description": "Search web query",
        "destructive": False,
        "requires_confirmation": False,
    },
}

ALIASES = {
    "shutdown": "shutdown",
    "shut down": "shutdown",
    "shutdown computer": "shutdown",
    "shut down computer": "shutdown",
    "power off": "shutdown",
    "turn off computer": "shutdown",
    "turn off pc": "shutdown",
    "restart": "restart",
    "restart computer": "restart",
    "restart pc": "restart",
    "reboot": "restart",
    "sleep computer": "sleep",
    "lock computer": "lock",
    "sign out": "logoff",
    "log out": "logoff",
    "turn it up": "volume_up",
    "volume up": "volume_up",
    "increase volume": "volume_up",
    "raise volume": "volume_up",
    "turn it down": "volume_down",
    "volume down": "volume_down",
    "decrease volume": "volume_down",
    "lower volume": "volume_down",
    "it's too loud": "volume_down",
    "it is too loud": "volume_down",
    "mute": "volume_mute",
    "mute volume": "volume_mute",
    "dim the screen": "brightness_down",
    "too bright": "brightness_down",
    "my screen is too bright": "brightness_down",
    "lower brightness": "brightness_down",
    "decrease brightness": "brightness_down",
    "increase brightness": "brightness_up",
    "raise brightness": "brightness_up",
    "brighten screen": "brightness_up",
    "turn off wi fi": "wifi_off",
    "turn off wifi": "wifi_off",
    "disable wifi": "wifi_off",
    "turn on wi fi": "wifi_on",
    "turn on wifi": "wifi_on",
    "enable wifi": "wifi_on",
    "turn off bluetooth": "bluetooth_off",
    "disable bluetooth": "bluetooth_off",
    "turn on bluetooth": "bluetooth_on",
    "enable bluetooth": "bluetooth_on",
    "turn on notifications": "notifications_on",
    "enable notifications": "notifications_on",
    "notifications on": "notifications_on",
    "allow notifications": "notifications_on",
    "turn off notifications": "notifications_off",
    "disable notifications": "notifications_off",
    "notifications off": "notifications_off",
    "mute notifications": "notifications_off",
    "silence notifications": "notifications_off",
    "turn on do not disturb": "notifications_off",
    "enable do not disturb": "notifications_off",
    "do not disturb on": "notifications_off",
    "dnd on": "notifications_off",
    "focus assist on": "notifications_off",
    "turn off do not disturb": "notifications_on",
    "disable do not disturb": "notifications_on",
    "do not disturb off": "notifications_on",
    "dnd off": "notifications_on",
    "focus assist off": "notifications_on",
    "take a screenshot": "screenshot",
    "capture the screen": "screenshot",
    "screenshot": "screenshot",
    "empty the trash": "empty_recycle_bin",
    "empty recycle bin": "empty_recycle_bin",
    "what is open right now": "list_processes",
    "show running processes": "list_processes",
    "show processes": "list_processes",
    "maximize window": "window_maximize",
    "maximize this window": "window_maximize",
    "minimize window": "window_minimize",
    "minimize this window": "window_minimize",
    "snap window left": "window_snap_left",
    "snap left": "window_snap_left",
    "snap window right": "window_snap_right",
    "snap right": "window_snap_right",
    "next window": "window_next",
    "switch window": "window_next",
    "close active window": "window_close_active",
    "close this window": "window_close_active",
    "pause media": "media_play_pause",
    "play media": "media_play_pause",
    "resume media": "media_play_pause",
    "next track": "media_next_track",
    "previous track": "media_previous_track",
    "prev track": "media_previous_track",
    "stop media": "media_stop",
    "new tab": "browser_new_tab",
    "open new tab": "browser_new_tab",
    "close tab": "browser_close_tab",
    "go back": "browser_back",
    "browser back": "browser_back",
    "go forward": "browser_forward",
    "browser forward": "browser_forward",
    "\u0627\u0637\u0641\u064a \u0627\u0644\u0643\u0645\u0628\u064a\u0648\u062a\u0631": "shutdown",
    "\u0627\u063a\u0644\u0642 \u0627\u0644\u0643\u0645\u0628\u064a\u0648\u062a\u0631": "shutdown",
    "\u0627\u063a\u0644\u0642 \u0627\u0644\u062c\u0647\u0627\u0632": "shutdown",
    "\u0627\u0639\u0627\u062f\u0629 \u062a\u0634\u063a\u064a\u0644": "restart",
    "\u0627\u0639\u0645\u0644 \u0627\u0639\u0627\u062f\u0629 \u062a\u0634\u063a\u064a\u0644": "restart",
    "\u0646\u0627\u0645 \u0627\u0644\u0643\u0645\u0628\u064a\u0648\u062a\u0631": "sleep",
    "\u0642\u0641\u0644 \u0627\u0644\u0643\u0645\u0628\u064a\u0648\u062a\u0631": "lock",
    "\u0633\u062c\u0644 \u062e\u0631\u0648\u062c": "logoff",
    "\u062a\u0633\u062c\u064a\u0644 \u062e\u0631\u0648\u062c": "logoff",
    "\u0643\u062a\u0645 \u0627\u0644\u0635\u0648\u062a": "volume_mute",
    "\u0627\u062e\u0641\u0636 \u0627\u0644\u0635\u0648\u062a": "volume_down",
    "\u0627\u0631\u0641\u0639 \u0627\u0644\u0635\u0648\u062a": "volume_up",
    "\u0635\u0648\u062a \u0639\u0627\u0644\u064a": "volume_down",
    "\u0627\u062e\u0641\u0636 \u0627\u0644\u0633\u0637\u0648\u0639": "brightness_down",
    "\u0632\u0648\u062f \u0627\u0644\u0633\u0637\u0648\u0639": "brightness_up",
    "\u0633\u0637\u0648\u0639 \u0639\u0627\u0644\u064a": "brightness_down",
    "\u0627\u0641\u0635\u0644 \u0627\u0644\u0627\u0646\u062a\u0631\u0646\u062a": "wifi_off",
    "\u0627\u0637\u0641\u064a \u0627\u0644\u0648\u0627\u064a \u0641\u0627\u064a": "wifi_off",
    "\u0634\u063a\u0644 \u0627\u0644\u0648\u0627\u064a \u0641\u0627\u064a": "wifi_on",
    "\u0627\u0637\u0641\u064a \u0627\u0644\u0628\u0644\u0648\u062a\u0648\u062b": "bluetooth_off",
    "\u0634\u063a\u0644 \u0627\u0644\u0628\u0644\u0648\u062a\u0648\u062b": "bluetooth_on",
    "\u0634\u063a\u0644 \u0627\u0644\u0627\u0634\u0639\u0627\u0631\u0627\u062a": "notifications_on",
    "\u0641\u0639\u0644 \u0627\u0644\u0627\u0634\u0639\u0627\u0631\u0627\u062a": "notifications_on",
    "\u0627\u0637\u0641\u064a \u0627\u0644\u0627\u0634\u0639\u0627\u0631\u0627\u062a": "notifications_off",
    "\u0627\u0642\u0641\u0644 \u0627\u0644\u0627\u0634\u0639\u0627\u0631\u0627\u062a": "notifications_off",
    "\u0627\u0643\u062a\u0645 \u0627\u0644\u0627\u0634\u0639\u0627\u0631\u0627\u062a": "notifications_off",
    "\u0648\u0636\u0639 \u0639\u062f\u0645 \u0627\u0644\u0627\u0632\u0639\u0627\u062c": "notifications_off",
    "\u0648\u0636\u0639 \u0639\u062f\u0645 \u0627\u0644\u0625\u0632\u0639\u0627\u062c": "notifications_off",
    "\u0634\u063a\u0644 \u0648\u0636\u0639 \u0639\u062f\u0645 \u0627\u0644\u0627\u0632\u0639\u0627\u062c": "notifications_off",
    "\u0634\u063a\u0644 \u0648\u0636\u0639 \u0639\u062f\u0645 \u0627\u0644\u0625\u0632\u0639\u0627\u062c": "notifications_off",
    "\u0627\u0642\u0641\u0644 \u0648\u0636\u0639 \u0639\u062f\u0645 \u0627\u0644\u0627\u0632\u0639\u0627\u062c": "notifications_on",
    "\u0627\u0642\u0641\u0644 \u0648\u0636\u0639 \u0639\u062f\u0645 \u0627\u0644\u0625\u0632\u0639\u0627\u062c": "notifications_on",
    "\u062e\u0630 \u0635\u0648\u0631\u0629 \u0644\u0644\u0634\u0627\u0634\u0629": "screenshot",
    "\u0635\u0648\u0631\u0629 \u0634\u0627\u0634\u0629": "screenshot",
    "\u0627\u0641\u0631\u063a \u0633\u0644\u0629 \u0627\u0644\u0645\u062d\u0630\u0648\u0641\u0627\u062a": "empty_recycle_bin",
    "\u0627\u0639\u0631\u0636 \u0627\u0644\u062a\u0637\u0628\u064a\u0642\u0627\u062a \u0627\u0644\u0634\u063a\u0627\u0644\u0629": "list_processes",
    "\u0643\u0628\u0631 \u0627\u0644\u0634\u0628\u0627\u0643": "window_maximize",
    "\u0635\u063a\u0631 \u0627\u0644\u0634\u0628\u0627\u0643": "window_minimize",
    "\u062d\u0631\u0643 \u0627\u0644\u0634\u0628\u0627\u0643 \u064a\u0645\u064a\u0646": "window_snap_right",
    "\u062d\u0631\u0643 \u0627\u0644\u0634\u0628\u0627\u0643 \u0634\u0645\u0627\u0644": "window_snap_left",
    "\u0627\u0644\u0634\u0628\u0627\u0643 \u0627\u0644\u0644\u064a \u0628\u0639\u062f\u0647": "window_next",
    "\u0627\u0642\u0641\u0644 \u0627\u0644\u0634\u0628\u0627\u0643": "window_close_active",
    "\u0633\u0643\u0631 \u0627\u0644\u0634\u0628\u0627\u0643": "window_close_active",
    "\u0634\u063a\u0644 \u0627\u0644\u0645\u0632\u064a\u0643\u0627": "media_play_pause",
    "\u0634\u063a\u0644 \u0627\u0644\u0645\u0632\u064a\u0643\u0647": "media_play_pause",
    "\u0627\u0644\u0627\u063a\u0646\u064a\u0629 \u0627\u0644\u0644\u064a \u0628\u0639\u062f \u0643\u062f\u0647": "media_next_track",
    "\u0627\u0644\u0627\u063a\u0646\u064a\u0647 \u0627\u0644\u0644\u064a \u0628\u0639\u062f \u0643\u062f\u0647": "media_next_track",
    "\u0627\u0644\u0627\u063a\u0646\u064a\u0629 \u0627\u0644\u0644\u064a \u0642\u0628\u0644\u0647\u0627": "media_previous_track",
    "\u0627\u0644\u0627\u063a\u0646\u064a\u0647 \u0627\u0644\u0644\u064a \u0642\u0628\u0644\u0647\u0627": "media_previous_track",
    "\u0648\u0642\u0641 \u0627\u0644\u0645\u064a\u062f\u064a\u0627": "media_stop",
    "\u0648\u0642\u0641 \u0627\u0644\u0645\u0632\u064a\u0643\u0627": "media_stop",
    "\u0648\u0642\u0641 \u0627\u0644\u0645\u0632\u064a\u0643\u0647": "media_stop",
    "\u0627\u0641\u062a\u062d \u062a\u0627\u0628 \u062c\u062f\u064a\u062f": "browser_new_tab",
    "\u0627\u0642\u0641\u0644 \u0627\u0644\u062a\u0627\u0628": "browser_close_tab",
    "\u0633\u0643\u0631 \u0627\u0644\u062a\u0627\u0628": "browser_close_tab",
    "\u0627\u0631\u062c\u0639 \u0648\u0631\u0627": "browser_back",
    "\u0627\u0631\u062c\u0639 \u0644\u0648\u0631\u0627": "browser_back",
    "\u0631\u0648\u062d \u0644\u0642\u062f\u0627\u0645": "browser_forward",
}

_RETRYABLE_NON_DESTRUCTIVE_ERRORS = ("timed out", "temporarily unavailable")
_URL_RE = re.compile(r"^(?:https?://|www\.)[^\s]+$", flags=re.IGNORECASE)
_DURATION_UNIT_SECONDS = {
    "s": 1,
    "sec": 1,
    "secs": 1,
    "second": 1,
    "seconds": 1,
    "\u062b\u0627\u0646\u064a\u0629": 1,
    "\u062b\u0648\u0627\u0646\u064a": 1,
    "m": 60,
    "min": 60,
    "mins": 60,
    "minute": 60,
    "minutes": 60,
    "\u062f\u0642\u064a\u0642\u0629": 60,
    "\u062f\u0642\u0627\u0626\u0642": 60,
    "h": 3600,
    "hr": 3600,
    "hrs": 3600,
    "hour": 3600,
    "hours": 3600,
    "\u0633\u0627\u0639\u0629": 3600,
    "\u0633\u0627\u0639\u0627\u062a": 3600,
}
_NUMBER_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "\u0635\u0641\u0631": 0,
    "\u0648\u0627\u062d\u062f": 1,
    "\u0627\u062b\u0646\u064a\u0646": 2,
    "\u0627\u062b\u0646\u064a\u0646": 2,
    "\u062b\u0644\u0627\u062b\u0629": 3,
    "\u0627\u0631\u0628\u0639\u0629": 4,
    "\u062e\u0645\u0633\u0629": 5,
    "\u0633\u062a\u0629": 6,
    "\u0633\u0628\u0639\u0629": 7,
    "\u062b\u0645\u0627\u0646\u064a\u0629": 8,
    "\u062a\u0633\u0639\u0629": 9,
    "\u0639\u0634\u0631\u0629": 10,
}
_NUMBER_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "\u0639\u0634\u0631\u064a\u0646": 20,
    "\u062b\u0644\u0627\u062b\u064a\u0646": 30,
    "\u0627\u0631\u0628\u0639\u064a\u0646": 40,
    "\u062e\u0645\u0633\u064a\u0646": 50,
    "\u0633\u062a\u064a\u0646": 60,
    "\u0633\u0628\u0639\u064a\u0646": 70,
    "\u062b\u0645\u0627\u0646\u064a\u0646": 80,
    "\u062a\u0633\u0639\u064a\u0646": 90,
}


def _normalize_words(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s\u0600-\u06FF]", " ", text)
    return " ".join(text.split())


def _parse_spoken_int(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(float(value))

    text = _normalize_words(value)
    if not text:
        return None

    digit = re.search(r"\d{1,4}", text)
    if digit:
        return int(digit.group(0))

    tokens = text.split()
    total = 0
    current = 0
    found = False
    for token in tokens:
        if token in {"and", "\u0648"}:
            continue
        if token in _NUMBER_ONES:
            current += _NUMBER_ONES[token]
            found = True
            continue
        if token in _NUMBER_TENS:
            current += _NUMBER_TENS[token]
            found = True
            continue
        if token in {"hundred", "\u0645\u0626\u0629", "\u0645\u0627\u064a\u0629", "\u0645\u064a\u0629"}:
            current = max(1, current) * 100
            found = True
            continue
        if token in {"thousand", "\u0627\u0644\u0641", "\u0623\u0644\u0641"}:
            total += max(1, current) * 1000
            current = 0
            found = True
            continue
    if not found:
        return None
    return total + current


def _parse_duration_seconds(value, unit_hint):
    number = _parse_spoken_int(value)
    if number is None:
        return None
    unit = _normalize_words(unit_hint)
    factor = _DURATION_UNIT_SECONDS.get(unit, 1)
    return max(1, min(3600, int(number * factor)))


def _normalize_url(value):
    candidate = str(value or "").strip().strip("\"").strip("'")
    candidate = re.sub(r"^(?:website|site|url|\u0645\u0648\u0642\u0639|\u0631\u0627\u0628\u0637)\s+", "", candidate, flags=re.IGNORECASE).strip()
    if not candidate:
        return ""
    if _URL_RE.match(candidate):
        if candidate.lower().startswith("www."):
            return f"https://{candidate}"
        return candidate
    if re.match(r"^[a-z0-9.-]+\.[a-z]{2,}(?:/[^\s]*)?$", candidate, flags=re.IGNORECASE):
        return f"https://{candidate}"
    return ""


def _fuzzy_resolve_system_action(phrase):
    if not phrase:
        return None
    words = str(phrase).split()
    if len(words) > 6 or len(str(phrase)) > 64:
        return None
    candidates = list(ALIASES.keys()) + list(SYSTEM_COMMANDS.keys())
    best_score = 0.0
    best_action = None
    for candidate in candidates:
        score = SequenceMatcher(a=phrase, b=candidate).ratio()
        if score > best_score:
            best_score = score
            best_action = ALIASES.get(candidate, candidate)
    if best_score >= 0.90:
        return best_action
    return None


def _parse_percent_value(value):
    if value is None:
        return None
    number = _parse_spoken_int(value)
    if number is None:
        return None
    return max(0, min(100, number))


def _normalize_system_command_args(action_key, command_args):
    args = dict(command_args or {})

    if action_key == "volume_set":
        level = _parse_percent_value(
            args.get("volume_level")
            or args.get("level")
            or args.get("percent")
            or args.get("value")
        )
        if level is None:
            return None, "Volume level is required (0-100)."
        return {"volume_level": level}, ""

    if action_key == "brightness_set":
        level = _parse_percent_value(
            args.get("brightness_level")
            or args.get("level")
            or args.get("percent")
            or args.get("value")
        )
        if level is None:
            return None, "Brightness level is required (0-100)."
        return {"brightness_level": level}, ""

    if action_key == "focus_window":
        query = str(
            args.get("window_query")
            or args.get("window_title")
            or args.get("target")
            or args.get("app_name")
            or ""
        ).strip()
        if not query:
            return None, "Window query is required (for example: Chrome)."
        return {"window_query": query[:120]}, ""

    if action_key in {"media_seek_forward", "media_seek_backward"}:
        seconds = _parse_duration_seconds(
            args.get("seek_seconds")
            or args.get("seconds")
            or args.get("duration")
            or args.get("value")
            or 10,
            args.get("unit") or "seconds",
        )
        if seconds is None:
            return None, "Seek duration is required (for example: 10 seconds)."
        return {"seek_seconds": seconds}, ""

    if action_key == "browser_open_url":
        url = _normalize_url(
            args.get("url")
            or args.get("link")
            or args.get("target")
            or args.get("value")
            or ""
        )
        if not url:
            return None, "URL is required (for example: https://github.com)."
        return {"url": url}, ""

    if action_key == "browser_search_web":
        query = str(
            args.get("search_query")
            or args.get("query")
            or args.get("text")
            or args.get("value")
            or ""
        ).strip()
        if not query:
            return None, "Search query is required (for example: Python asyncio tutorial)."
        return {"search_query": query[:200]}, ""

    return {}, ""


def _template_env_overrides(action_key, normalized_args):
    if action_key == "volume_set":
        return {"JARVIS_VOLUME_PERCENT": int(normalized_args.get("volume_level", 0))}
    if action_key == "brightness_set":
        return {"JARVIS_BRIGHTNESS_PERCENT": int(normalized_args.get("brightness_level", 0))}
    if action_key == "focus_window":
        return {"JARVIS_WINDOW_QUERY": str(normalized_args.get("window_query", ""))}
    if action_key in {"media_seek_forward", "media_seek_backward"}:
        return {"JARVIS_MEDIA_SEEK_SECONDS": int(normalized_args.get("seek_seconds", 10))}
    if action_key == "browser_open_url":
        return {"JARVIS_BROWSER_URL": str(normalized_args.get("url", ""))}
    if action_key == "browser_search_web":
        return {"JARVIS_BROWSER_QUERY": str(normalized_args.get("search_query", ""))}
    return {}


def _render_system_description(action_key, fallback_description, normalized_args):
    if action_key == "volume_set":
        return f"Set system volume to {normalized_args.get('volume_level')}%"
    if action_key == "brightness_set":
        return f"Set screen brightness to {normalized_args.get('brightness_level')}%"
    if action_key == "focus_window":
        return f"Focus window {normalized_args.get('window_query')}"
    if action_key == "media_seek_forward":
        return f"Seek media forward {normalized_args.get('seek_seconds')}s"
    if action_key == "media_seek_backward":
        return f"Seek media backward {normalized_args.get('seek_seconds')}s"
    if action_key == "browser_open_url":
        return f"Open website {normalized_args.get('url')}"
    if action_key == "browser_search_web":
        return f"Search web for {normalized_args.get('search_query')}"
    return fallback_description


def _render_system_success_message(action_key, normalized_args, output):
    if action_key == "browser_search_web":
        query = str((normalized_args or {}).get("search_query") or "").strip()
        return f"Searching the web for: {query}" if query else "Searching the web."

    if action_key == "browser_open_url":
        url = str((normalized_args or {}).get("url") or "").strip()
        return f"Opening website: {url}" if url else "Opening website."

    if action_key == "volume_set":
        level = (normalized_args or {}).get("volume_level")
        if level is not None:
            return f"Volume set to {level}%."

    if action_key == "brightness_set":
        level = (normalized_args or {}).get("brightness_level")
        if level is not None:
            return f"Brightness set to {level}%."

    if action_key in {"media_seek_forward", "media_seek_backward"}:
        seconds = (normalized_args or {}).get("seek_seconds")
        direction = "forward" if action_key == "media_seek_forward" else "backward"
        if seconds is not None:
            return f"Seeked media {direction} by {seconds}s."

    cleaned_output = (output or "").strip()
    if cleaned_output:
        return cleaned_output
    return f"Executed system command: {action_key}."


def normalize_system_action(text):
    phrase = text.lower().strip()
    if phrase.startswith("system "):
        phrase = phrase[7:].strip()
    if phrase.startswith("\u0627\u0644\u0646\u0638\u0627\u0645 "):
        phrase = phrase[len("\u0627\u0644\u0646\u0638\u0627\u0645 ") :].strip()
    phrase = phrase.replace("please ", "")
    phrase = phrase.replace("\u0645\u0646 \u0641\u0636\u0644\u0643 ", "")
    phrase = phrase.replace("\u0644\u0648 \u0633\u0645\u062d\u062a ", "")
    phrase = re.sub(r"[^a-z0-9_\s\-\u0600-\u06FF]", " ", phrase)
    phrase = " ".join(phrase.split())
    if phrase in SYSTEM_COMMANDS:
        return phrase
    direct = ALIASES.get(phrase)
    if direct:
        return direct
    return _fuzzy_resolve_system_action(phrase)


def is_system_command(text):
    return normalize_system_action(text) is not None


def request_system_command_result(action_key, command_args=None):
    if action_key not in SYSTEM_COMMANDS:
        return failure_result("Unsupported system command.", error_code="unsupported_action")

    if not policy_engine.is_command_allowed("system_command"):
        return failure_result("System commands are disabled by policy.", error_code="policy_blocked")

    normalized_args, args_error = _normalize_system_command_args(action_key, command_args)
    if args_error:
        return failure_result(args_error, error_code="invalid_input")

    cfg = SYSTEM_COMMANDS[action_key]
    requires_confirmation = bool(cfg.get("requires_confirmation", cfg["destructive"]))

    if not requires_confirmation:
        return execute_system_command_result(action_key, command_args=normalized_args)

    require_second_factor = bool(cfg["destructive"] and SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE)
    risk_tier = risk_tier_for_system(
        action_key,
        destructive=bool(cfg.get("destructive")),
        requires_confirmation=requires_confirmation,
    )
    description = _render_system_description(action_key, cfg["description"], normalized_args)

    payload = {
        "kind": "system_command",
        "action_key": action_key,
        "command_args": dict(normalized_args or {}),
        "require_second_factor": require_second_factor,
    }
    token = confirmation_manager.create(
        action_name=f"system_{action_key}",
        description=description,
        payload=payload,
    )
    log_action(
        "system_command_request",
        "pending",
        details={
            "action": action_key,
            "token": token,
            "second_factor": require_second_factor,
            "risk_tier": risk_tier,
            "args": dict(normalized_args or {}),
        },
    )

    message = format_confirmation_prompt(
        description,
        token,
        risk_tier=risk_tier,
        timeout_seconds=CONFIRMATION_TIMEOUT_SECONDS,
        require_second_factor=require_second_factor,
    )
    return confirmation_result(
        message,
        token=token,
        second_factor=require_second_factor,
        risk_tier=risk_tier,
        debug_info={"action": action_key, "command_args": dict(normalized_args or {})},
    )


def _run_system_template_with_safe_retry(template_name, destructive, env_overrides=None):
    attempts = 0
    last_error = ""
    while attempts < (1 if destructive else 2):
        attempts += 1
        ok, error, output = run_template(
            template_name,
            env_overrides=dict(env_overrides or {}),
            timeout_seconds=30,
        )
        if ok:
            return True, "", output, attempts
        last_error = error or "PowerShell template failed"
        if destructive:
            break
        if not any(token in last_error.lower() for token in _RETRYABLE_NON_DESTRUCTIVE_ERRORS):
            break
    return False, last_error, "", attempts


def execute_system_command_result(action_key, command_args=None):
    if action_key not in SYSTEM_COMMANDS:
        return failure_result("Unsupported system command.", error_code="unsupported_action")

    normalized_args, args_error = _normalize_system_command_args(action_key, command_args)
    if args_error:
        return failure_result(args_error, error_code="invalid_input")

    cfg = SYSTEM_COMMANDS[action_key]
    if cfg["destructive"] and not ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS:
        msg = (
            "Blocked by configuration. Set ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS=True "
            "in core/config.py to enable this command."
        )
        log_action(
            "system_command",
            "blocked",
            details={"action": action_key, "reason": "destructive_disabled"},
        )
        return failure_result(msg, error_code="destructive_disabled", debug_info={"action": action_key})

    ok, error, output, attempts = _run_system_template_with_safe_retry(
        cfg["template"],
        destructive=bool(cfg["destructive"]),
        env_overrides=_template_env_overrides(action_key, normalized_args),
    )
    if ok:
        message = _render_system_success_message(action_key, normalized_args, output)
        log_action(
            "system_command",
            "success",
            details={
                "action": action_key,
                "args": dict(normalized_args or {}),
                "output": output,
                "attempts": attempts,
            },
        )
        logger.info("Executed system command template: %s", action_key)
        return success_result(
            message,
            debug_info={"action": action_key, "args": dict(normalized_args or {}), "attempts": attempts},
            executed_confirmed_action="system_command",
        )

    log_action(
        "system_command",
        "failed",
        details={"action": action_key, "args": dict(normalized_args or {}), "attempts": attempts},
        error=error,
    )
    error_code = "timeout" if "timed out" in (error or "").lower() else "execution_failed"
    return failure_result(
        f"Execution failed: {error}",
        error_code=error_code,
        debug_info={"action": action_key, "args": dict(normalized_args or {}), "attempts": attempts},
    )


def request_system_command(action_key, command_args=None):
    result = request_system_command_result(action_key, command_args=command_args)
    legacy_success, legacy_message = to_legacy_pair(result)
    legacy_meta = {}
    if isinstance(result, dict):
        for key in ("requires_confirmation", "token", "second_factor", "risk_tier"):
            if key in result:
                legacy_meta[key] = result[key]
    return legacy_success, legacy_message, legacy_meta


def execute_system_command(action_key, command_args=None):
    return to_legacy_pair(execute_system_command_result(action_key, command_args=command_args))
