import ctypes
import re
from difflib import SequenceMatcher
from ctypes import wintypes

from core.config import (
    ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS,
    CONFIRMATION_TIMEOUT_SECONDS,
    SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE,
)
from core.config import FEATURE_FLAGS
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
from os_control.native_ops import (
    adjust_system_volume_percent,
    adjust_system_brightness_percent,
    capture_primary_screen_screenshot,
    lock_workstation,
    set_system_volume_percent,
    set_system_brightness_percent,
    sleep_system,
    toggle_system_mute,
)
from os_control.policy import policy_engine
from os_control.powershell_bridge import run_template
from os_control.risk_policy import risk_tier_for_system
from core.logger import log_structured


_KEYEVENTF_KEYUP = 0x0002
_VK_MEDIA_NEXT_TRACK = 0xB0
_VK_MEDIA_PREV_TRACK = 0xB1
_VK_MEDIA_STOP = 0xB2
_VK_MEDIA_PLAY_PAUSE = 0xB3


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
    "airplane_on": {
        "template": "airplane_on",
        "description": "Enable airplane mode (turn off all radios)",
        "destructive": False,
        "requires_confirmation": False,
    },
    "airplane_off": {
        "template": "airplane_off",
        "description": "Disable airplane mode (restore radios)",
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
    "dnd_on": {
        "template": "dnd_on",
        "description": "Enable Do Not Disturb (Focus Assist)",
        "destructive": False,
        "requires_confirmation": False,
    },
    "dnd_off": {
        "template": "dnd_off",
        "description": "Disable Do Not Disturb (Focus Assist)",
        "destructive": False,
        "requires_confirmation": False,
    },
    "night_light_on": {
        "template": "night_light_on",
        "description": "Enable night light (warm screen)",
        "destructive": False,
        "requires_confirmation": False,
    },
    "night_light_off": {
        "template": "night_light_off",
        "description": "Disable night light",
        "destructive": False,
        "requires_confirmation": False,
    },
    "energy_saver_on": {
        "template": "energy_saver_on",
        "description": "Enable energy saver / battery saver",
        "destructive": False,
        "requires_confirmation": False,
    },
    "energy_saver_off": {
        "template": "energy_saver_off",
        "description": "Disable energy saver / battery saver",
        "destructive": False,
        "requires_confirmation": False,
    },
    "live_caption_on": {
        "template": "live_caption_on",
        "description": "Enable live captions",
        "destructive": False,
        "requires_confirmation": False,
    },
    "live_caption_off": {
        "template": "live_caption_off",
        "description": "Disable live captions",
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
    "window_resize": {
        "template": "window_resize",
        "description": "Resize active window smaller",
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
    "browser_close_named_tab": {
        "template": "browser_close_named_tab",
        "description": "Close a browser tab by title/name (focus then Ctrl+W)",
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
    "clipboard_read": {
        "template": "clipboard_read",
        "description": "Read clipboard contents",
        "destructive": False,
        "requires_confirmation": False,
    },
    "clipboard_write": {
        "template": "clipboard_write",
        "description": "Write text to clipboard",
        "destructive": False,
        "requires_confirmation": False,
    },
    "clipboard_clear": {
        "template": "clipboard_clear",
        "description": "Clear the clipboard",
        "destructive": False,
        "requires_confirmation": False,
    },
    "screen_record_start": {
        "template": "screen_record_start",
        "description": "Start screen recording",
        "destructive": False,
        "requires_confirmation": False,
    },
    "screen_record_stop": {
        "template": "screen_record_stop",
        "description": "Stop screen recording",
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
    # Mixed-language Bluetooth variants that STT commonly produces
    "اقفل bluetooth": "bluetooth_off",
    "اطفي bluetooth": "bluetooth_off",
    "إطفئ bluetooth": "bluetooth_off",
    "إيقاف bluetooth": "bluetooth_off",
    "ايقاف bluetooth": "bluetooth_off",
    "شغل bluetooth": "bluetooth_on",
    "تشغيل bluetooth": "bluetooth_on",
    "فعل bluetooth": "bluetooth_on",
    "اقفل بلوتوث": "bluetooth_off",
    "اطفي بلوتوث": "bluetooth_off",
    "إطفئ بلوتوث": "bluetooth_off",
    "إيقاف بلوتوث": "bluetooth_off",
    "ايقاف بلوتوث": "bluetooth_off",
    "شغل بلوتوث": "bluetooth_on",
    "تشغيل بلوتوث": "bluetooth_on",
    "فعل بلوتوث": "bluetooth_on",
    # Mixed-language Wi-Fi variants
    "اقفل wifi": "wifi_off",
    "اطفي wifi": "wifi_off",
    "إطفئ wifi": "wifi_off",
    "إيقاف wifi": "wifi_off",
    "ايقاف wifi": "wifi_off",
    "شغل wifi": "wifi_on",
    "تشغيل wifi": "wifi_on",
    "فعل wifi": "wifi_on",
    "airplane mode on": "airplane_on",
    "airplane mode off": "airplane_off",
    "flight mode on": "airplane_on",
    "flight mode off": "airplane_off",
    "enable airplane mode": "airplane_on",
    "disable airplane mode": "airplane_off",
    "turn on airplane mode": "airplane_on",
    "turn off airplane mode": "airplane_off",
    "turn notifications on": "notifications_on",
    "turn on notifications": "notifications_on",
    "enable notifications": "notifications_on",
    "notifications on": "notifications_on",
    "allow notifications": "notifications_on",
    "turn off notifications": "notifications_off",
    "disable notifications": "notifications_off",
    "notifications off": "notifications_off",
    "mute notifications": "notifications_off",
    "silence notifications": "notifications_off",
    # Do Not Disturb / Focus Assist (FIXED: was wrongly pointing to notifications)
    "turn on do not disturb": "dnd_on",
    "enable do not disturb": "dnd_on",
    "do not disturb on": "dnd_on",
    "dnd on": "dnd_on",
    "focus assist on": "dnd_on",
    "turn off do not disturb": "dnd_off",
    "disable do not disturb": "dnd_off",
    "do not disturb off": "dnd_off",
    "dnd off": "dnd_off",
    "focus assist off": "dnd_off",
    # Night light
    "turn on night light": "night_light_on",
    "enable night light": "night_light_on",
    "night light on": "night_light_on",
    "turn off night light": "night_light_off",
    "disable night light": "night_light_off",
    "night light off": "night_light_off",
    "warm screen": "night_light_on",
    "cool screen": "night_light_off",
    # Energy saver
    "turn on energy saver": "energy_saver_on",
    "enable energy saver": "energy_saver_on",
    "energy saver on": "energy_saver_on",
    "battery saver on": "energy_saver_on",
    "turn on battery saver": "energy_saver_on",
    "turn off energy saver": "energy_saver_off",
    "disable energy saver": "energy_saver_off",
    "energy saver off": "energy_saver_off",
    "battery saver off": "energy_saver_off",
    "turn off battery saver": "energy_saver_off",
    # Live captions
    "turn on live captions": "live_caption_on",
    "enable live captions": "live_caption_on",
    "live captions on": "live_caption_on",
    "live caption on": "live_caption_on",
    "turn off live captions": "live_caption_off",
    "disable live captions": "live_caption_off",
    "live captions off": "live_caption_off",
    "live caption off": "live_caption_off",
    "take a screenshot": "screenshot",
    "capture the screen": "screenshot",
    "screenshot": "screenshot",
    # Screen recording — English
    "start recording": "screen_record_start",
    "start screen recording": "screen_record_start",
    "record screen": "screen_record_start",
    "record my screen": "screen_record_start",
    "begin recording": "screen_record_start",
    "stop recording": "screen_record_stop",
    "stop screen recording": "screen_record_stop",
    "end recording": "screen_record_stop",
    "finish recording": "screen_record_stop",
    "empty the trash": "empty_recycle_bin",
    "empty recycle bin": "empty_recycle_bin",
    "what is open right now": "list_processes",
    "show running processes": "list_processes",
    "show processes": "list_processes",
    "maximize window": "window_maximize",
    "maximize this window": "window_maximize",
    "minimize window": "window_minimize",
    "minimize this window": "window_minimize",
    "resize window smaller": "window_resize",
    "resize this window smaller": "window_resize",
    "make window smaller": "window_resize",
    "shrink window": "window_resize",
    "shrink this window": "window_resize",
    "window resize smaller": "window_resize",
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
    # Airplane mode \u2014 Arabic aliases
    "\u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0627\u0631\u0627\u0646": "airplane_on",
    "\u0634\u063a\u0644 \u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0627\u0631\u0627\u0646": "airplane_on",
    "\u0637\u064a\u0627\u0631\u0627\u0646": "airplane_on",
    "\u0645\u0648\u062f \u0637\u064a\u0627\u0631\u0627\u0646": "airplane_on",
    "\u0641\u0644\u0627\u064a\u062a \u0645\u0648\u062f": "airplane_on",
    "\u0627\u0644\u063a\u064a \u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0627\u0631\u0627\u0646": "airplane_off",
    "\u0623\u0644\u063a\u064a \u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0627\u0631\u0627\u0646": "airplane_off",
    "\u0627\u0644\u063a\u064a \u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0631\u0627\u0646": "airplane_off",
    "\u0623\u0644\u063a\u064a \u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0631\u0627\u0646": "airplane_off",
    "\u0625\u064a\u0642\u0641 \u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0631\u0627\u0646": "airplane_off",
    "\u0625\u064a\u0642\u0641 \u0627\u0644\u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0631\u0627\u0646": "airplane_off",
    "\u0627\u064a\u0642\u0641 \u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0631\u0627\u0646": "airplane_off",
    "\u0648\u0642\u0641 \u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0631\u0627\u0646": "airplane_off",
    "\u0627\u0648\u0642\u0641 \u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0631\u0627\u0646": "airplane_off",
    "\u0627\u0642\u0641\u0644 \u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0627\u0631\u0627\u0646": "airplane_off",
    "\u0627\u0637\u0641\u064a \u0645\u0648\u062f \u0627\u0644\u0637\u064a\u0627\u0631\u0627\u0646": "airplane_off",
    # airplane on variants with \u0627\u0644\u0637\u064a\u0631\u0627\u0646 (without \u0627\u0646)
    "\u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0631\u0627\u0646": "airplane_on",
    "\u0634\u063a\u0644 \u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0631\u0627\u0646": "airplane_on",
    "\u0641\u0639\u0644 \u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0631\u0627\u0646": "airplane_on",
    "\u0645\u0648\u062f \u0637\u064a\u0631\u0627\u0646": "airplane_on",
    "\u0641\u0644\u0627\u064a\u062a \u0645\u0648\u062f": "airplane_on",
    # STT variant with tatweel/kashida: "\u0634\u063a\u0644 \u0627\u0644\u0640 Airplane mode"
    "\u0634\u063a\u0644 \u0627\u0644 airplane mode": "airplane_on",
    "\u0634\u063a\u0644 \u0627\u0644 airplane": "airplane_on",
    "\u0634\u063a\u0644 \u0627\u0644 \u0637\u064a\u0631\u0627\u0646": "airplane_on",
    "\u0634\u063a\u0644 \u0627\u0644\u0627\u0634\u0639\u0627\u0631\u0627\u062a": "notifications_on",
    "\u0641\u0639\u0644 \u0627\u0644\u0627\u0634\u0639\u0627\u0631\u0627\u062a": "notifications_on",
    "\u0627\u0637\u0641\u064a \u0627\u0644\u0627\u0634\u0639\u0627\u0631\u0627\u062a": "notifications_off",
    "\u0627\u0642\u0641\u0644 \u0627\u0644\u0627\u0634\u0639\u0627\u0631\u0627\u062a": "notifications_off",
    "\u0627\u0643\u062a\u0645 \u0627\u0644\u0627\u0634\u0639\u0627\u0631\u0627\u062a": "notifications_off",
    # DND Arabic aliases (FIXED: were wrongly pointing to notifications)
    "\u0648\u0636\u0639 \u0639\u062f\u0645 \u0627\u0644\u0627\u0632\u0639\u0627\u062c": "dnd_on",
    "\u0648\u0636\u0639 \u0639\u062f\u0645 \u0627\u0644\u0625\u0632\u0639\u0627\u062c": "dnd_on",
    "\u0634\u063a\u0644 \u0648\u0636\u0639 \u0639\u062f\u0645 \u0627\u0644\u0627\u0632\u0639\u0627\u062c": "dnd_on",
    "\u0634\u063a\u0644 \u0648\u0636\u0639 \u0639\u062f\u0645 \u0627\u0644\u0625\u0632\u0639\u0627\u062c": "dnd_on",
    "\u0627\u0642\u0641\u0644 \u0648\u0636\u0639 \u0639\u062f\u0645 \u0627\u0644\u0627\u0632\u0639\u0627\u062c": "dnd_off",
    "\u0627\u0642\u0641\u0644 \u0648\u0636\u0639 \u0639\u062f\u0645 \u0627\u0644\u0625\u0632\u0639\u0627\u062c": "dnd_off",
    "\u0639\u062f\u0645 \u0627\u0644\u0625\u0632\u0639\u0627\u062c": "dnd_on",
    "\u0639\u062f\u0645 \u0627\u0644\u0627\u0632\u0639\u0627\u062c": "dnd_on",
    "\u0627\u0644\u063a\u064a \u0639\u062f\u0645 \u0627\u0644\u0625\u0632\u0639\u0627\u062c": "dnd_off",
    "\u0627\u0644\u063a\u064a \u0639\u062f\u0645 \u0627\u0644\u0627\u0632\u0639\u0627\u062c": "dnd_off",
    # Night light Arabic
    "\u0627\u0644\u0625\u0636\u0627\u0621\u0629 \u0627\u0644\u0644\u064a\u0644\u064a\u0629": "night_light_on",
    "\u0627\u0644\u0648\u0636\u0639 \u0627\u0644\u0644\u064a\u0644\u064a": "night_light_on",
    "\u0634\u063a\u0644 \u0627\u0644\u0625\u0636\u0627\u0621\u0629 \u0627\u0644\u0644\u064a\u0644\u064a\u0629": "night_light_on",
    "\u0634\u063a\u0644 \u0627\u0644\u0648\u0636\u0639 \u0627\u0644\u0644\u064a\u0644\u064a": "night_light_on",
    "\u0627\u0637\u0641\u064a \u0627\u0644\u0625\u0636\u0627\u0621\u0629 \u0627\u0644\u0644\u064a\u0644\u064a\u0629": "night_light_off",
    "\u0627\u0637\u0641\u064a \u0627\u0644\u0648\u0636\u0639 \u0627\u0644\u0644\u064a\u0644\u064a": "night_light_off",
    "\u0646\u0627\u064a\u062a \u0644\u0627\u064a\u062a": "night_light_on",
    "\u0627\u0637\u0641\u064a \u0646\u0627\u064a\u062a \u0644\u0627\u064a\u062a": "night_light_off",
    # Energy saver Arabic
    "\u062a\u0648\u0641\u064a\u0631 \u0627\u0644\u0637\u0627\u0642\u0629": "energy_saver_on",
    "\u0648\u0636\u0639 \u062a\u0648\u0641\u064a\u0631 \u0627\u0644\u0637\u0627\u0642\u0629": "energy_saver_on",
    "\u0634\u063a\u0644 \u062a\u0648\u0641\u064a\u0631 \u0627\u0644\u0637\u0627\u0642\u0629": "energy_saver_on",
    "\u062a\u0648\u0641\u064a\u0631 \u0627\u0644\u0628\u0637\u0627\u0631\u064a\u0629": "energy_saver_on",
    "\u0648\u0636\u0639 \u062a\u0648\u0641\u064a\u0631 \u0627\u0644\u0628\u0637\u0627\u0631\u064a\u0629": "energy_saver_on",
    "\u0627\u0637\u0641\u064a \u062a\u0648\u0641\u064a\u0631 \u0627\u0644\u0637\u0627\u0642\u0629": "energy_saver_off",
    "\u0627\u0644\u063a\u064a \u062a\u0648\u0641\u064a\u0631 \u0627\u0644\u0637\u0627\u0642\u0629": "energy_saver_off",
    "\u0627\u0637\u0641\u064a \u062a\u0648\u0641\u064a\u0631 \u0627\u0644\u0628\u0637\u0627\u0631\u064a\u0629": "energy_saver_off",
    # Live captions Arabic
    "\u0627\u0644\u062a\u0631\u062c\u0645\u0629 \u0627\u0644\u062d\u064a\u0629": "live_caption_on",
    "\u0627\u0644\u0644\u0627\u064a\u0641 \u0643\u0627\u0628\u0634\u0646": "live_caption_on",
    "\u0634\u063a\u0644 \u0627\u0644\u062a\u0631\u062c\u0645\u0629 \u0627\u0644\u062d\u064a\u0629": "live_caption_on",
    "\u0627\u0637\u0641\u064a \u0627\u0644\u062a\u0631\u062c\u0645\u0629 \u0627\u0644\u062d\u064a\u0629": "live_caption_off",
    "\u0627\u0637\u0641\u064a \u0627\u0644\u0644\u0627\u064a\u0641 \u0643\u0627\u0628\u0634\u0646": "live_caption_off",
    "\u062e\u0630 \u0635\u0648\u0631\u0629 \u0644\u0644\u0634\u0627\u0634\u0629": "screenshot",
    "\u0635\u0648\u0631\u0629 \u0634\u0627\u0634\u0629": "screenshot",
    "\u0627\u0641\u0631\u063a \u0633\u0644\u0629 \u0627\u0644\u0645\u062d\u0630\u0648\u0641\u0627\u062a": "empty_recycle_bin",
    "\u0627\u0639\u0631\u0636 \u0627\u0644\u062a\u0637\u0628\u064a\u0642\u0627\u062a \u0627\u0644\u0634\u063a\u0627\u0644\u0629": "list_processes",
    "\u0643\u0628\u0631 \u0627\u0644\u0634\u0628\u0627\u0643": "window_maximize",
    "\u0635\u063a\u0631 \u0627\u0644\u0634\u0628\u0627\u0643": "window_minimize",
    "\u0635\u063a\u0631 \u0627\u0644\u0646\u0627\u0641\u0630\u0629": "window_minimize",
    "\u0627\u0637\u0648\u064a \u0627\u0644\u0634\u0628\u0627\u0643": "window_minimize",
    "\u0627\u0637\u0648\u064a \u0627\u0644\u0646\u0627\u0641\u0630\u0629": "window_minimize",
    "\u0627\u0637\u0648\u0651\u064a \u0627\u0644\u0634\u0628\u0627\u0643": "window_minimize",
    "\u0627\u0637\u0648\u0651\u064a \u0627\u0644\u0646\u0627\u0641\u0630\u0629": "window_minimize",
    "\u0635\u063a\u0631 \u0627\u0644\u0634\u0628\u0627\u0643 \u0627\u0643\u062a\u0631": "window_resize",
    "\u0635\u063a\u0651\u0631 \u0627\u0644\u0634\u0628\u0627\u0643 \u0623\u0643\u062a\u0631": "window_resize",
    "\u0635\u063a\u0631 \u0627\u0644\u0634\u0628\u0627\u0643 \u0634\u0648\u064a\u0629": "window_resize",
    "\u0635\u063a\u0651\u0631 \u0627\u0644\u0634\u0628\u0627\u0643 \u0634\u0648\u064a\u0629": "window_resize",
    "\u0635\u063a\u0631 \u0627\u0644\u0646\u0627\u0641\u0630\u0629 \u0627\u0643\u062a\u0631": "window_resize",
    "\u0635\u063a\u0651\u0631 \u0627\u0644\u0646\u0627\u0641\u0630\u0629 \u0623\u0643\u062a\u0631": "window_resize",
    "\u0627\u062e\u0641\u0651\u0641 \u062d\u062c\u0645 \u0627\u0644\u0634\u0628\u0627\u0643": "window_resize",
    "\u0627\u062e\u0641\u0641 \u062d\u062c\u0645 \u0627\u0644\u0634\u0628\u0627\u0643": "window_resize",
    "\u0627\u062e\u0641\u0651\u0641 \u062d\u062c\u0645 \u0627\u0644\u0646\u0627\u0641\u0630\u0629": "window_resize",
    "\u0627\u062e\u0641\u0641 \u062d\u062c\u0645 \u0627\u0644\u0646\u0627\u0641\u0630\u0629": "window_resize",
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
    # --- Egyptian Arabic dialect additions ---
    # Shutdown
    "\u0627\u0637\u0641\u064a\u0647": "shutdown",
    "\u0627\u0642\u0641\u0644\u0647": "shutdown",
    "\u0648\u0642\u0641\u0647": "shutdown",
    # Restart
    "\u0627\u0639\u064a\u062f\u0647": "restart",
    "\u0627\u0639\u064a\u062f \u062a\u0634\u063a\u064a\u0644\u0647": "restart",
    "\u0631\u064a\u0633\u062a\u0627\u0631\u062a\u0647": "restart",
    # Sleep
    "\u0646\u064a\u0645\u0647": "sleep",
    "\u062e\u0644\u064a\u0647 \u064a\u0646\u0627\u0645": "sleep",
    "\u062d\u0637\u0647 \u064a\u0646\u0627\u0645": "sleep",
    "\u0646\u0648\u0645 \u0627\u0644\u062c\u0647\u0627\u0632": "sleep",
    # Lock
    "\u0642\u0641\u0644 \u0627\u0644\u0634\u0627\u0634\u0629": "lock",
    "\u0642\u0641\u0644\u064a \u0627\u0644\u0634\u0627\u0634\u0629": "lock",
    "\u0642\u0641\u0644\u064a \u0627\u0644\u0643\u0645\u0628\u064a\u0648\u062a\u0631": "lock",
    "\u0644\u0648\u0643 \u0627\u0644\u0634\u0627\u0634\u0629": "lock",
    # Volume
    "\u0627\u0631\u0641\u0639\u0644\u064a \u0627\u0644\u0635\u0648\u062a": "volume_up",
    "\u0632\u0648\u062f\u0647": "volume_up",
    "\u0632\u0648\u062f\u0644\u064a \u0627\u0644\u0635\u0648\u062a": "volume_up",
    "\u0648\u0637\u064a\u0644\u064a \u0627\u0644\u0635\u0648\u062a": "volume_down",
    "\u0648\u0637\u064a\u0647": "volume_down",
    "\u0635\u0648\u062a \u0648\u0627\u0637\u064a": "volume_down",
    "\u0627\u0635\u0645\u062a\u0644\u064a": "volume_mute",
    "\u0627\u0633\u0643\u062a\u0644\u064a": "volume_mute",
    "\u0627\u0644\u0635\u0648\u062a \u0632\u064a\u0627\u062f\u0647": "volume_down",
    # Brightness
    "\u0631\u0641\u0639\u0644\u064a \u0627\u0644\u0633\u0637\u0648\u0639": "brightness_up",
    "\u0632\u0648\u062f\u0644\u064a \u0627\u0644\u0646\u0648\u0631": "brightness_up",
    "\u0646\u0648\u0631 \u0627\u0643\u062a\u0631": "brightness_up",
    "\u0648\u0637\u064a\u0644\u064a \u0627\u0644\u0633\u0637\u0648\u0639": "brightness_down",
    "\u0642\u0644\u0644\u0644\u064a \u0627\u0644\u0646\u0648\u0631": "brightness_down",
    "\u0646\u0648\u0631 \u0627\u0642\u0644": "brightness_down",
    # Windows
    "\u0643\u0628\u0631\u0644\u064a \u0627\u0644\u0634\u0628\u0627\u0643": "window_maximize",
    "\u0643\u0628\u0631\u0647": "window_maximize",
    "\u0635\u063a\u0631\u0644\u064a \u0627\u0644\u0634\u0628\u0627\u0643": "window_minimize",
    "\u0635\u063a\u0631\u0647": "window_minimize",
    "\u0633\u0643\u0631\u0644\u064a \u0627\u0644\u0634\u0628\u0627\u0643": "window_close_active",
    "\u0642\u0641\u0644\u064a \u0627\u0644\u0634\u0628\u0627\u0643": "window_close_active",
    "\u0634\u0628\u0627\u0643 \u062a\u0627\u0646\u064a": "window_next",
    "\u063a\u064a\u0631 \u0627\u0644\u0634\u0628\u0627\u0643": "window_next",
    "\u062d\u0631\u0643 \u064a\u0645\u064a\u0646": "window_snap_right",
    "\u062d\u0631\u0643 \u0634\u0645\u0627\u0644": "window_snap_left",
    # Media
    "\u0627\u0648\u0642\u0641 \u0627\u0644\u0645\u0648\u0632\u064a\u0643\u0627": "media_stop",
    "\u0627\u0648\u0642\u0641 \u0627\u0644\u0645\u0648\u0632\u064a\u0643\u0647": "media_stop",
    "\u0627\u0648\u0642\u0641 \u0627\u0644\u0645\u064a\u0648\u0632\u064a\u0643": "media_stop",
    "\u0627\u0644\u0627\u063a\u0646\u064a\u0629 \u0627\u0644\u062c\u0627\u064a\u0629": "media_next_track",
    "\u0627\u063a\u0646\u064a\u0629 \u062a\u0627\u0646\u064a\u0629": "media_next_track",
    "\u0627\u0644\u0627\u063a\u0646\u064a\u0629 \u0627\u0644\u0644\u064a \u0641\u0627\u062a\u062a": "media_previous_track",
    "\u0631\u062c\u0639\u0644\u064a \u0627\u0644\u0627\u063a\u0646\u064a\u0629": "media_previous_track",
    # Browser
    "\u062a\u0627\u0628 \u062c\u062f\u064a\u062f": "browser_new_tab",
    "\u0633\u0643\u0631\u0644\u064a \u0627\u0644\u062a\u0627\u0628": "browser_close_tab",
    "\u0642\u0641\u0644\u064a \u0627\u0644\u062a\u0627\u0628": "browser_close_tab",
    "\u0631\u062c\u0639\u0644\u064a \u0648\u0631\u0627": "browser_back",
    "\u0631\u0648\u062d\u0644\u064a \u0642\u062f\u0627\u0645": "browser_forward",
    # Screenshot
    "\u0633\u0643\u0631\u064a\u0646 \u0634\u0648\u062a": "screenshot",
    "\u0635\u0648\u0631\u0629 \u0644\u0644\u0634\u0627\u0634\u0629": "screenshot",
    "\u0633\u0643\u0631\u064a\u0646\u0634\u0648\u062a": "screenshot",
    # Screen recording \u2014 Arabic / Egyptian
    "\u0627\u0628\u062f\u0623 \u0627\u0644\u062a\u0633\u062c\u064a\u0644": "screen_record_start",
    "\u0633\u062c\u0651\u0644 \u0627\u0644\u0634\u0627\u0634\u0629": "screen_record_start",
    "\u0633\u062c\u0644 \u0627\u0644\u0634\u0627\u0634\u0629": "screen_record_start",
    "\u0627\u0628\u062f\u0623 \u062a\u0633\u062c\u064a\u0644 \u0627\u0644\u0634\u0627\u0634\u0629": "screen_record_start",
    "\u0634\u063a\u0651\u0644 \u0627\u0644\u062a\u0633\u062c\u064a\u0644": "screen_record_start",
    "\u0634\u063a\u0644 \u0627\u0644\u062a\u0633\u062c\u064a\u0644": "screen_record_start",
    "\u0648\u0642\u0651\u0641 \u0627\u0644\u062a\u0633\u062c\u064a\u0644": "screen_record_stop",
    "\u0648\u0642\u0641 \u0627\u0644\u062a\u0633\u062c\u064a\u0644": "screen_record_stop",
    "\u0627\u0648\u0642\u0641 \u0627\u0644\u062a\u0633\u062c\u064a\u0644": "screen_record_stop",
    "\u062e\u0644\u0651\u0635 \u0627\u0644\u062a\u0633\u062c\u064a\u0644": "screen_record_stop",
    "\u062e\u0644\u0635 \u0627\u0644\u062a\u0633\u062c\u064a\u0644": "screen_record_stop",
    # Other
    "\u0641\u0636\u064a\u0644\u064a \u0633\u0644\u0629 \u0627\u0644\u0645\u062d\u0630\u0648\u0641\u0627\u062a": "empty_recycle_bin",
    "\u0641\u0636\u064a\u0644\u064a \u0627\u0644\u0632\u0628\u0627\u0644\u0647": "empty_recycle_bin",
    "\u062e\u0631\u0648\u062c": "logoff",
    "\u0627\u062e\u0631\u062c \u0645\u0646 \u0627\u0644\u062d\u0633\u0627\u0628": "logoff",
    # Clipboard \u2014 English
    "read clipboard": "clipboard_read",
    "what's in my clipboard": "clipboard_read",
    "show clipboard": "clipboard_read",
    "clipboard read": "clipboard_read",
    "paste content": "clipboard_read",
    "write to clipboard": "clipboard_write",
    "copy to clipboard": "clipboard_write",
    "clipboard write": "clipboard_write",
    "clear clipboard": "clipboard_clear",
    "empty clipboard": "clipboard_clear",
    "clipboard clear": "clipboard_clear",
    # Clipboard \u2014 Arabic / Egyptian dialect
    "\u0627\u0642\u0631\u0623 \u0627\u0644\u0643\u0644\u064a\u0628\u0628\u0648\u0631\u062f": "clipboard_read",
    "\u0627\u0639\u0631\u0636 \u0627\u0644\u0643\u0644\u064a\u0628\u0628\u0648\u0631\u062f": "clipboard_read",
    "\u0627\u064a\u0647 \u0641\u064a \u0627\u0644\u0643\u0644\u064a\u0628\u0628\u0648\u0631\u062f": "clipboard_read",
    "\u0627\u064a\u0647 \u0641\u064a \u0627\u0644\u0643\u0644\u0628\u0648\u0631\u062f": "clipboard_read",
    "\u0627\u0643\u062a\u0628 \u0641\u064a \u0627\u0644\u0643\u0644\u064a\u0628\u0628\u0648\u0631\u062f": "clipboard_write",
    "\u0627\u0646\u0633\u062e \u0644\u0644\u0643\u0644\u064a\u0628\u0628\u0648\u0631\u062f": "clipboard_write",
    "\u0627\u0646\u0633\u062e \u0641\u064a \u0627\u0644\u0643\u0644\u064a\u0628\u0628\u0648\u0631\u062f": "clipboard_write",
    "\u0627\u0645\u0633\u062d \u0627\u0644\u0643\u0644\u064a\u0628\u0628\u0648\u0631\u062f": "clipboard_clear",
    "\u0641\u0636\u064a\u0644\u064a \u0627\u0644\u0643\u0644\u064a\u0628\u0628\u0648\u0631\u062f": "clipboard_clear",
}

_RETRYABLE_NON_DESTRUCTIVE_ERRORS = ("timed out", "temporarily unavailable")
_PERMISSION_DENIED_ERROR_MARKERS = (
    "access is denied",
    "permissiondenied",
    "windows system error 5",
    "requires elevation",
    "requested operation requires elevation",
    "not have sufficient privilege",
)
_NETWORK_RADIO_ACTIONS = {"wifi_on", "wifi_off", "bluetooth_on", "bluetooth_off", "airplane_on", "airplane_off"}
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
        if token.startswith("ال") and len(token) > 2:
            candidate = token[2:]
        else:
            candidate = token
        if token in {"and", "\u0648"}:
            continue
        if candidate in _NUMBER_ONES:
            current += _NUMBER_ONES[candidate]
            found = True
            continue
        if candidate in _NUMBER_TENS:
            current += _NUMBER_TENS[candidate]
            found = True
            continue
        if candidate in {"hundred", "\u0645\u0626\u0629", "\u0645\u0627\u064a\u0629", "\u0645\u064a\u0629"}:
            current = max(1, current) * 100
            found = True
            continue
        if candidate in {"thousand", "\u0627\u0644\u0641", "\u0623\u0644\u0641"}:
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
    text = _normalize_words(value)
    if any(token in text.split() for token in {"نص", "نصف"}):
        return 50
    if "ربع" in text.split():
        return 25
    if any(token in text.split() for token in {"تلت", "ثلث"}):
        return 33

    text = re.sub(r"\b(?:في\s+)?الم(?:ية|ئه|ئة|يه|ائه)\b", "", text)
    text = text.replace("بالمية", "").replace("بالمئة", "").replace("بالمئه", "")
    number = _parse_spoken_int(text)
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

    if action_key == "clipboard_write":
        text = str(
            args.get("text")
            or args.get("content")
            or args.get("value")
            or ""
        )
        if not text.strip():
            return None, "Text to copy is required."
        return {"text": text[:4000]}, ""

    if action_key == "browser_close_named_tab":
        query = str(
            args.get("tab_query")
            or args.get("tab_name")
            or args.get("query")
            or args.get("target")
            or ""
        ).strip().lower()
        if not query:
            return None, "Tab name is required (for example: YouTube)."
        return {"tab_query": query[:120]}, ""

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
    if action_key == "browser_close_named_tab":
        return {"JARVIS_TAB_QUERY": str(normalized_args.get("tab_query", ""))}
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
    if action_key == "browser_close_named_tab":
        return f"Close browser tab matching '{normalized_args.get('tab_query')}'"
    return fallback_description


def _is_arabic_language(language):
    return str(language or "").lower().startswith("ar")


def _send_media_key(vk_code):
    try:
        user32 = ctypes.windll.user32
        user32.keybd_event(vk_code, 0, 0, 0)
        user32.keybd_event(vk_code, 0, _KEYEVENTF_KEYUP, 0)
        return True
    except Exception:
        return False


_VK_CONTROL = 0x11
_VK_MENU = 0x12   # Alt
_VK_SHIFT = 0x10
_VK_LEFT = 0x25
_VK_RIGHT = 0x27
_VK_F4 = 0x73
_VK_T = 0x54
_VK_W = 0x57


def _send_hotkey(*vk_codes):
    """Press and release a sequence of VK codes as a chord (press all, release all reversed)."""
    try:
        user32 = ctypes.windll.user32
        for vk in vk_codes:
            user32.keybd_event(vk, 0, 0, 0)
        for vk in reversed(vk_codes):
            user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)
        return True
    except Exception:
        return False


_BROWSER_PROCESS_NAMES = {"firefox", "chrome", "msedge", "opera", "brave", "vivaldi", "iexplore"}


def _close_named_browser_tab(normalized_args, is_ar):
    """Find a browser window whose title matches *tab_query*, focus it, then send Ctrl+W."""
    import time as _time

    tab_query = str((normalized_args or {}).get("tab_query", "")).strip().lower()
    if not tab_query:
        msg = "مش عارف أقفل إيه — قولي اسم التاب." if is_ar else "I don't know which tab to close — tell me the tab name."
        return False, msg, {}

    if not hasattr(ctypes, "windll"):
        return False, "", {}

    user32 = ctypes.windll.user32

    EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_size_t, ctypes.c_size_t)

    best_hwnd = [0]
    best_score = [0.0]
    best_title = [""]

    def _enum_cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length == 0 or length > 512:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value.strip()
        title_lower = title.lower()
        if not title_lower:
            return True
        # Only consider windows from known browsers (check class name)
        class_buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buf, 256)
        cls = class_buf.value.lower()
        # Firefox: "mozillawindowclass", Chrome/Edge: "chrome_widgetwin_1"
        if not any(b in cls for b in ("mozilla", "chrome", "edge", "opera", "brave")):
            return True
        score = SequenceMatcher(a=tab_query, b=title_lower).ratio()
        if tab_query in title_lower:
            score = max(score, 0.75)
        if score > best_score[0]:
            best_score[0] = score
            best_hwnd[0] = hwnd
            best_title[0] = title
        return True

    user32.EnumWindows(EnumWindowsProc(_enum_cb), 0)

    if best_score[0] >= 0.40 and best_hwnd[0]:
        hwnd = best_hwnd[0]
        if user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        _time.sleep(0.15)  # let the OS switch focus before sending keys
        if _send_hotkey(_VK_CONTROL, _VK_W):
            title = best_title[0]
            msg = f"قفّلت تاب '{title}'." if is_ar else f"Closed tab: {title}"
            return True, msg, {"method": "close_named_tab", "title": title, "score": best_score[0]}

    msg = (f"مش لاقي تاب اسمه '{tab_query}'." if is_ar else f"No browser tab found matching '{tab_query}'.")
    return False, msg, {"method": "close_named_tab", "query": tab_query, "score": best_score[0]}


def _run_native_window_command(action_key, language=None):
    """Native ctypes dispatch for window management and browser tab shortcuts."""
    if not hasattr(ctypes, "windll"):
        return False, "", {}

    is_ar = _is_arabic_language(language)
    user32 = ctypes.windll.user32

    try:
        if action_key == "window_maximize":
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return False, "", {}
            user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
            msg = "كبّرت الشبابك." if is_ar else "Window maximized."
            return True, msg, {"method": "native_window", "sw": "SW_MAXIMIZE"}

        if action_key == "window_minimize":
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return False, "", {}
            user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
            msg = "صغّرت الشبابك." if is_ar else "Window minimized."
            return True, msg, {"method": "native_window", "sw": "SW_MINIMIZE"}

        if action_key == "window_snap_left":
            if _send_hotkey(0x5B, _VK_LEFT):  # Win+Left
                msg = "دفعت الشبابك ناحية اليسار." if is_ar else "Window snapped left."
                return True, msg, {"method": "native_hotkey", "keys": "Win+Left"}
            return False, "", {}

        if action_key == "window_snap_right":
            if _send_hotkey(0x5B, _VK_RIGHT):  # Win+Right
                msg = "دفعت الشبابك ناحية اليمين." if is_ar else "Window snapped right."
                return True, msg, {"method": "native_hotkey", "keys": "Win+Right"}
            return False, "", {}

        if action_key == "window_next":
            if _send_hotkey(_VK_MENU, 0x09):  # Alt+Tab
                msg = "غيّرت للشبابك التانية." if is_ar else "Switched to next window."
                return True, msg, {"method": "native_hotkey", "keys": "Alt+Tab"}
            return False, "", {}

        if action_key == "window_close_active":
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return False, "", {}
            WM_CLOSE = 0x0010
            user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            msg = "اتقفل الشبابك." if is_ar else "Window closed."
            return True, msg, {"method": "native_window", "msg": "WM_CLOSE"}

        if action_key == "browser_new_tab":
            if _send_hotkey(_VK_CONTROL, _VK_T):  # Ctrl+T
                msg = "فتحت تاب جديد." if is_ar else "Opened new browser tab."
                return True, msg, {"method": "native_hotkey", "keys": "Ctrl+T"}
            return False, "", {}

        if action_key == "browser_close_tab":
            if _send_hotkey(_VK_CONTROL, _VK_W):  # Ctrl+W
                msg = "قفّلت التاب." if is_ar else "Closed browser tab."
                return True, msg, {"method": "native_hotkey", "keys": "Ctrl+W"}
            return False, "", {}

        if action_key == "browser_back":
            if _send_hotkey(_VK_MENU, _VK_LEFT):  # Alt+Left
                msg = "رجعت للصفحة السابقة." if is_ar else "Went back."
                return True, msg, {"method": "native_hotkey", "keys": "Alt+Left"}
            return False, "", {}

        if action_key == "browser_forward":
            if _send_hotkey(_VK_MENU, _VK_RIGHT):  # Alt+Right
                msg = "رحت للصفحة الجاية." if is_ar else "Went forward."
                return True, msg, {"method": "native_hotkey", "keys": "Alt+Right"}
            return False, "", {}

    except Exception as exc:
        logger.debug("_run_native_window_command(%s) failed: %s", action_key, exc)
        return False, "", {}

    return False, "", {}


def _resize_active_window(amount_percent=10, language=None):
    if not hasattr(ctypes, "windll"):
        return False, "", {}

    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return False, "", {}

        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return False, "", {}

        width = int(rect.right - rect.left)
        height = int(rect.bottom - rect.top)
        if width <= 0 or height <= 0:
            return False, "", {}

        if hasattr(user32, "IsIconic") and user32.IsIconic(hwnd):
            user32.ShowWindow(hwnd, 9)

        shrink_ratio = max(0.0, min(1.0, float(amount_percent) / 100.0))
        new_width = max(320, int(round(width * (1.0 - shrink_ratio))))
        new_height = max(240, int(round(height * (1.0 - shrink_ratio))))
        if new_width >= width and new_height >= height:
            new_width = max(320, width - 80)
            new_height = max(240, height - 80)

        ok = bool(user32.MoveWindow(hwnd, rect.left, rect.top, new_width, new_height, True))
        if not ok:
            return False, "", {}

        if _is_arabic_language(language):
            message = f"تمام، صغّرت الشبابك لـ {int(round((new_width / width) * 100))}%."
        else:
            message = f"Window resized to {int(round((new_width / width) * 100))}% of original size."
        return True, message, {"method": "native_window", "width": new_width, "height": new_height}
    except Exception as exc:
        logger.debug("Native window resize failed: %s", exc)
        return False, "", {}


def _run_native_media_command(action_key, language=None):
    if action_key == "media_play_pause":
        if _send_media_key(_VK_MEDIA_PLAY_PAUSE):
            return True, ("شغّلت أو وقفت الميديا." if _is_arabic_language(language) else "Toggled media play/pause."), {"method": "native_media"}
        return False, "", {}
    if action_key == "media_next_track":
        if _send_media_key(_VK_MEDIA_NEXT_TRACK):
            return True, ("الأغنية الجاية." if _is_arabic_language(language) else "Skipped to the next track."), {"method": "native_media"}
        return False, "", {}
    if action_key == "media_previous_track":
        if _send_media_key(_VK_MEDIA_PREV_TRACK):
            return True, ("رجعت للأغنية اللي قبلها." if _is_arabic_language(language) else "Went to the previous track."), {"method": "native_media"}
        return False, "", {}
    if action_key == "media_stop":
        if _send_media_key(_VK_MEDIA_STOP):
            return True, ("وقفت تشغيل الميديا." if _is_arabic_language(language) else "Stopped media playback."), {"method": "native_media"}
        return False, "", {}
    return False, "", {}


def _render_system_success_message(action_key, normalized_args, output, language=None):
    is_ar = _is_arabic_language(language)
    if action_key == "browser_search_web":
        query = str((normalized_args or {}).get("search_query") or "").strip()
        if is_ar:
            return f"بدور على: {query}" if query else "بفتح البحث."
        return f"Searching the web for: {query}" if query else "Searching the web."

    if action_key == "browser_open_url":
        url = str((normalized_args or {}).get("url") or "").strip()
        if is_ar:
            return f"بفتح الموقع: {url}" if url else "بفتح الموقع."
        return f"Opening website: {url}" if url else "Opening website."

    if action_key == "volume_set":
        level = (normalized_args or {}).get("volume_level")
        if level is not None:
            return f"Volume set to {level}%."

    if action_key == "brightness_set":
        level = (normalized_args or {}).get("brightness_level")
        if level is not None:
            return f"Brightness set to {level}%."

    if action_key == "media_play_pause":
        return "شغّلت أو وقفت الميديا." if _is_arabic_language(language) else "Toggled media play/pause."

    if action_key == "media_next_track":
        return "الأغنية الجاية." if _is_arabic_language(language) else "Skipped to the next track."

    if action_key == "media_previous_track":
        return "رجعت للأغنية اللي قبلها." if _is_arabic_language(language) else "Went to the previous track."

    if action_key == "media_stop":
        return "وقفت تشغيل الميديا." if _is_arabic_language(language) else "Stopped media playback."

    if action_key in {"media_seek_forward", "media_seek_backward"}:
        seconds = (normalized_args or {}).get("seek_seconds")
        direction = "forward" if action_key == "media_seek_forward" else "backward"
        if seconds is not None:
            return f"Seeked media {direction} by {seconds}s."

    cleaned_output = (output or "").strip()
    if cleaned_output:
        return cleaned_output
    return f"Executed system command: {action_key}."


def _compact_system_error(error_text, max_chars=220):
    lines = [line.strip() for line in str(error_text or "").splitlines() if line.strip()]
    if not lines:
        return "Unknown system execution error."

    for line in lines:
        lowered = line.lower()
        if lowered.startswith("at line:"):
            continue
        if lowered.startswith("+"):
            continue
        if lowered.startswith("categoryinfo"):
            continue
        if lowered.startswith("fullyqualifiederrorid"):
            continue
        if len(line) > max_chars:
            return line[: max_chars - 3] + "..."
        return line

    fallback = lines[0]
    if len(fallback) > max_chars:
        return fallback[: max_chars - 3] + "..."
    return fallback


def _is_permission_denied_error(error_text):
    lowered = str(error_text or "").lower()
    return any(marker in lowered for marker in _PERMISSION_DENIED_ERROR_MARKERS)


def _permission_denied_message(action_key):
    if action_key in _NETWORK_RADIO_ACTIONS:
        return (
            "I need Administrator privileges to change Wi-Fi/Bluetooth state. "
            "Please run Jarvis as Administrator and try again."
        )
    return (
        "Windows denied permission for this action. "
        "Please run Jarvis as Administrator and try again."
    )


# Transliteration map: Latin tech-words that STT produces in mixed phrases
# mapped to the Arabic form used in ALIASES (so "\u0634\u063a\u0644 Bluetooth" \u2192 "\u0634\u063a\u0644 \u0627\u0644\u0628\u0644\u0648\u062a\u0648\u062b").
_LATIN_TO_ARABIC_TECH = {
    "bluetooth": "\u0627\u0644\u0628\u0644\u0648\u062a\u0648\u062b",
    "wifi": "\u0627\u0644\u0648\u0627\u064a \u0641\u0627\u064a",
    "wi-fi": "\u0627\u0644\u0648\u0627\u064a \u0641\u0627\u064a",
    "wi fi": "\u0627\u0644\u0648\u0627\u064a \u0641\u0627\u064a",
    "airplane": "\u0627\u0644\u0637\u064a\u0631\u0627\u0646",
    "airplane mode": "\u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0631\u0627\u0646",
    "flight mode": "\u0648\u0636\u0639 \u0627\u0644\u0637\u064a\u0631\u0627\u0646",
}


def normalize_system_action(text):
    phrase = text.lower().strip()
    if phrase.startswith("system "):
        phrase = phrase[7:].strip()
    if phrase.startswith("\u0627\u0644\u0646\u0638\u0627\u0645 "):
        phrase = phrase[len("\u0627\u0644\u0646\u0638\u0627\u0645 ") :].strip()
    phrase = phrase.replace("please ", "")
    phrase = phrase.replace("\u0645\u0646 \u0641\u0636\u0644\u0643 ", "")
    phrase = phrase.replace("\u0644\u0648 \u0633\u0645\u062d\u062a ", "")

    # Strip tatweel (kashida, U+0640) and definite-article ligatures like "\u0627\u0644\u0640"
    # that STT sometimes inserts in mixed phrases.
    phrase = phrase.replace("\u0640", "")  # tatweel
    phrase = re.sub(r"\u0627\u0644\s+", "\u0627\u0644", phrase)  # "\u0627\u0644 " with space \u2192 "\u0627\u0644"

    # Transliterate mixed-language tech-words (only when Arabic script is present
    # so we don't corrupt pure-English phrases like "turn off wifi").
    # Sort longest-key-first so "airplane mode" matches before "airplane".
    _has_arabic = bool(re.search(r"[\u0600-\u06FF]", phrase))
    if _has_arabic:
        for latin in sorted(_LATIN_TO_ARABIC_TECH, key=len, reverse=True):
            phrase = re.sub(
                r"\b" + re.escape(latin) + r"\b",
                _LATIN_TO_ARABIC_TECH[latin],
                phrase,
                flags=re.IGNORECASE,
            )

    phrase = re.sub(r"[^a-z0-9_\s\-\u0600-\u06FF]", " ", phrase)
    phrase = " ".join(phrase.split())
    if phrase in SYSTEM_COMMANDS:
        return phrase
    direct = ALIASES.get(phrase)
    if direct:
        return direct
    if "volume" in phrase or "الصوت" in phrase or "صوت" in phrase or "الفوليم" in phrase:
        if any(token in phrase for token in (
            "up",
            "raise",
            "increase",
            "turn up",
            "louder",
            "ارفع",
            "زود",
            "زوّد",
            "علي",
            "عالي",
        )):
            return "volume_up"
        if any(token in phrase for token in (
            "down",
            "lower",
            "decrease",
            "turn down",
            "softer",
            "اخفض",
            "خفض",
            "قلل",
            "وط",
            "وطي",
        )):
            return "volume_down"
        if any(token in phrase for token in ("mute", "silent", "كتم", "اسكت")):
            return "volume_mute"

    if "brightness" in phrase or "السطوع" in phrase or "سطوع" in phrase or "اضاءة" in phrase or "الإضاءة" in phrase or "نور" in phrase:
        if any(token in phrase for token in (
            "up",
            "raise",
            "increase",
            "brighten",
            "ارفع",
            "زود",
            "زوّد",
            "علي",
            "اعلي",
            "أعلي",
        )):
            return "brightness_up"
        if any(token in phrase for token in (
            "down",
            "lower",
            "decrease",
            "dim",
            "اخفض",
            "خفض",
            "قلل",
            "وط",
            "وطي",
        )):
            return "brightness_down"
    return _fuzzy_resolve_system_action(phrase)


def is_system_command(text):
    return normalize_system_action(text) is not None


def request_system_command_result(action_key, command_args=None, language=None):
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
        return execute_system_command_result(action_key, command_args=normalized_args, language=language)

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


def _run_native_volume_command(action_key, normalized_args):
    """Try native Windows volume control before falling back to PowerShell.

    Returns:
        tuple[bool, str, dict] - (success, message, debug_info)
    """
    if action_key == "volume_set":
        level = int(normalized_args.get("volume_level", 50))
        ok = set_system_volume_percent(level)
        if ok:
            return True, f"Volume set to {level}%.", {"method": "native_volume", "level": level}
        return False, "", {}

    if action_key == "volume_up":
        ok, new_level = adjust_system_volume_percent(10)
        if ok and new_level is not None:
            return True, f"Volume increased to {new_level}%.", {"method": "native_volume", "level": new_level}
        return False, "", {}

    if action_key == "volume_down":
        ok, new_level = adjust_system_volume_percent(-10)
        if ok and new_level is not None:
            return True, f"Volume decreased to {new_level}%.", {"method": "native_volume", "level": new_level}
        return False, "", {}

    if action_key == "volume_mute":
        ok, new_level = toggle_system_mute()
        if ok:
            if new_level == 0:
                return True, "Volume muted.", {"method": "native_volume", "level": 0}
            return True, f"Volume restored to {new_level}%.", {"method": "native_volume", "level": new_level}
        return False, "", {}

    return False, "", {}


def _run_native_system_command(action_key):
    """Try native Windows APIs for simple system actions.

    Returns:
        tuple[bool, str, dict] - (success, message, debug_info)
    """
    if action_key == "lock":
        if lock_workstation():
            return True, "Locking this computer.", {"method": "native_lock"}
        return False, "", {}

    if action_key == "sleep":
        if sleep_system():
            return True, "Putting this computer to sleep.", {"method": "native_sleep"}
        return False, "", {}

    if action_key == "screenshot":
        path = capture_primary_screen_screenshot()
        if path:
            return True, f"Screenshot saved to {path}", {"method": "native_screenshot", "path": path}
        return False, "", {}

    if action_key == "window_resize":
        return _resize_active_window(amount_percent=10)

    return False, "", {}


def _run_native_windows_toggle(action_key, language=None):
    """Toggle Night Light, DND, Energy Saver, or Live Captions via windows_toggles.

    Returns:
        tuple[bool, str, dict] - (success, message, debug_info)
    """
    try:
        from os_control.windows_toggles import (
            set_night_light,
            set_dnd,
            set_energy_saver,
            set_live_captions,
        )
    except ImportError:
        return False, "", {}

    is_ar = _is_arabic_language(language)

    _TOGGLE_DISPATCH = {
        "night_light_on":   (set_night_light,   True,  "شغّلت الإضاءة الليلية.",      "Night light on."),
        "night_light_off":  (set_night_light,   False, "أطفيت الإضاءة الليلية.",      "Night light off."),
        "dnd_on":           (set_dnd,           True,  "شغّلت وضع عدم الإزعاج.",      "Do not disturb on."),
        "dnd_off":          (set_dnd,           False, "ألغيت وضع عدم الإزعاج.",      "Do not disturb off."),
        "energy_saver_on":  (set_energy_saver,  True,  "شغّلت توفير الطاقة.",          "Energy saver on."),
        "energy_saver_off": (set_energy_saver,  False, "ألغيت توفير الطاقة.",          "Energy saver off."),
        "live_caption_on":  (set_live_captions, True,  "شغّلت الترجمة الحية.",         "Live captions on."),
        "live_caption_off": (set_live_captions, False, "أطفيت الترجمة الحية.",         "Live captions off."),
    }

    if action_key not in _TOGGLE_DISPATCH:
        return False, "", {}

    fn, arg, msg_ar, msg_en = _TOGGLE_DISPATCH[action_key]
    ok = fn(arg)
    if ok:
        msg = msg_ar if is_ar else msg_en
        return True, msg, {"method": "windows_toggles", "action": action_key}
    return False, "", {}


def _run_native_radio_command(action_key, language=None):
    """Toggle Wi-Fi, Bluetooth, or Airplane mode via radio_ops (WinRT-first).

    Returns:
        tuple[bool, str, dict] - (success, message, debug_info)
    """
    try:
        from os_control.radio_ops import set_radio, set_airplane
    except ImportError:
        return False, "", {}

    is_ar = _is_arabic_language(language)

    if action_key == "wifi_on":
        ok = set_radio("wifi", True)
        if ok:
            msg = "شغّلت الواي فاي." if is_ar else "Wi-Fi enabled."
            return True, msg, {"method": "radio_ops", "radio": "wifi", "state": "on"}
        return False, "", {}

    if action_key == "wifi_off":
        ok = set_radio("wifi", False)
        if ok:
            msg = "أطفيت الواي فاي." if is_ar else "Wi-Fi disabled."
            return True, msg, {"method": "radio_ops", "radio": "wifi", "state": "off"}
        return False, "", {}

    if action_key == "bluetooth_on":
        ok = set_radio("bluetooth", True)
        if ok:
            msg = "شغّلت البلوتوث." if is_ar else "Bluetooth enabled."
            return True, msg, {"method": "radio_ops", "radio": "bluetooth", "state": "on"}
        return False, "", {}

    if action_key == "bluetooth_off":
        ok = set_radio("bluetooth", False)
        if ok:
            msg = "أطفيت البلوتوث." if is_ar else "Bluetooth disabled."
            return True, msg, {"method": "radio_ops", "radio": "bluetooth", "state": "off"}
        return False, "", {}

    if action_key == "airplane_on":
        ok = set_airplane(True)
        if ok:
            msg = "شغّلت وضع الطياران." if is_ar else "Airplane mode on. All radios off."
            return True, msg, {"method": "radio_ops", "radio": "all", "state": "airplane_on"}
        return False, "", {}

    if action_key == "airplane_off":
        ok = set_airplane(False)
        if ok:
            msg = "ألغيت وضع الطياران." if is_ar else "Airplane mode off. Radios restored."
            return True, msg, {"method": "radio_ops", "radio": "all", "state": "airplane_off"}
        return False, "", {}

    return False, "", {}


def _run_native_clipboard_command(action_key, normalized_args, language=None):
    try:
        from os_control.clipboard_ops import clear_clipboard, read_clipboard, write_clipboard
        from core.config import CLIPBOARD_READ_MAX_CHARS
    except ImportError:
        return False, "", {}

    is_ar = _is_arabic_language(language)

    if action_key == "clipboard_read":
        content = read_clipboard()
        if content and not content.startswith("Could not") and not content.startswith("Clipboard access"):
            if len(content) > CLIPBOARD_READ_MAX_CHARS:
                content = content[:CLIPBOARD_READ_MAX_CHARS] + "…"
            return True, content, {"method": "clipboard_ops", "length": len(content)}
        return True, content, {"method": "clipboard_ops"}

    if action_key == "clipboard_write":
        text = str((normalized_args or {}).get("text", ""))
        result = write_clipboard(text)
        msg = f"نسخت {len(text)} حرف للكليببورد." if is_ar else result
        return True, msg, {"method": "clipboard_ops", "length": len(text)}

    if action_key == "clipboard_clear":
        result = clear_clipboard()
        msg = "مسحت الكليببورد." if is_ar else result
        return True, msg, {"method": "clipboard_ops"}

    return False, "", {}


def _run_native_capture_command(action_key, language=None):
    """Dispatch screen_record_start / screen_record_stop via capture_ops."""
    try:
        from os_control.capture_ops import start_recording, stop_recording
    except ImportError:
        return False, "", {}

    lang = language or "en"
    if action_key == "screen_record_start":
        ok, msg = start_recording(language=lang)
        return ok, msg, {"method": "capture_ops", "action": action_key}
    if action_key == "screen_record_stop":
        ok, msg = stop_recording(language=lang)
        return ok, msg, {"method": "capture_ops", "action": action_key}
    return False, "", {}


def _run_native_focus_window(normalized_args, language=None):
    if not hasattr(ctypes, "windll"):
        return False, "", {}

    query = str((normalized_args or {}).get("window_query", "")).strip().lower()
    if not query:
        return False, "", {}

    try:
        user32 = ctypes.windll.user32
        is_ar = _is_arabic_language(language)

        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_size_t, ctypes.c_size_t)

        best_hwnd = [0]
        best_score = [0.0]
        best_title = [""]

        def _enum_callback(hwnd, lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0 or length > 512:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value.lower().strip()
            if not title:
                return True
            score = SequenceMatcher(a=query, b=title).ratio()
            if query in title:
                score = max(score, 0.75)
            if score > best_score[0]:
                best_score[0] = score
                best_hwnd[0] = hwnd
                best_title[0] = buf.value.strip()
            return True

        user32.EnumWindows(EnumWindowsProc(_enum_callback), 0)

        if best_score[0] >= 0.55 and best_hwnd[0]:
            hwnd = best_hwnd[0]
            _SW_RESTORE = 9
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, _SW_RESTORE)
            user32.SetForegroundWindow(hwnd)
            title = best_title[0]
            msg = f"فتحت '{title}'." if is_ar else f"Focused window: {title}"
            return True, msg, {"method": "enum_windows", "title": title, "score": best_score[0]}

        msg = ("مش لاقي الشباك ده." if is_ar else f"No window found matching '{query}'.")
        return False, msg, {"method": "enum_windows", "score": best_score[0]}
    except Exception as exc:
        logger.debug("focus_window native failed: %s", exc)
        return False, "", {}


def _run_native_brightness_command(action_key, normalized_args):
    """Try native brightness controls before falling back to PowerShell.

    Returns:
        tuple[bool, str, dict] - (success, message, debug_info)
    """
    if action_key == "brightness_set":
        level = int(normalized_args.get("brightness_level", 50))
        ok = set_system_brightness_percent(level)
        if ok:
            return True, f"Brightness set to {level}%.", {"method": "native_brightness", "level": level}
        return False, "", {}

    if action_key == "brightness_up":
        ok, new_level = adjust_system_brightness_percent(10)
        if ok and new_level is not None:
            return True, f"Brightness increased to {new_level}%.", {"method": "native_brightness", "level": new_level}
        return False, "", {}

    if action_key == "brightness_down":
        ok, new_level = adjust_system_brightness_percent(-10)
        if ok and new_level is not None:
            return True, f"Brightness decreased to {new_level}%.", {"method": "native_brightness", "level": new_level}
        return False, "", {}

    return False, "", {}


def execute_system_command_result(action_key, command_args=None, language=None):
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

    # Native-first volume/brightness control (falls through to PowerShell if unavailable)
    if action_key in {"volume_up", "volume_down", "volume_mute", "volume_set"}:
        if FEATURE_FLAGS.get("SYSTEM_VOLUME_CONTROL", True):
            native_ok, native_msg, native_debug = _run_native_volume_command(action_key, normalized_args)
        else:
            native_ok, native_msg, native_debug = False, "", {}
        if native_ok:
            log_action("system_command", "success", details={"action": action_key, **native_debug})
            log_structured("system_command_executed", action=action_key, success=True, method=native_debug.get("method"), level="info")
            return success_result(native_msg, debug_info={"action": action_key, **native_debug})

    if action_key in {"media_play_pause", "media_next_track", "media_previous_track", "media_stop"}:
        # NEW: Prevent media commands from interfering with TTS playback
        # If TTS is currently speaking, wait a moment for it to finish
        from audio.tts import speech_engine
        import time
        max_wait_time = 0.5  # Maximum time to wait for TTS to finish
        start_time = time.time()
        while speech_engine.is_speaking() and (time.time() - start_time) < max_wait_time:
            time.sleep(0.05)
        
        if FEATURE_FLAGS.get("MEDIA_DIRECT_DISPATCH_ENABLED", True):
            native_ok, native_msg, native_debug = _run_native_media_command(action_key, language=language)
        else:
            native_ok, native_msg, native_debug = False, "", {}
        if native_ok:
            log_action("system_command", "success", details={"action": action_key, **native_debug})
            log_structured("system_command_executed", action=action_key, success=True, method=native_debug.get("method"))
            return success_result(native_msg, debug_info={"action": action_key, **native_debug})

    if action_key == "screenshot":
        try:
            from os_control.capture_ops import take_screenshot
            ss_ok, ss_msg = take_screenshot(language=language or "en")
            if ss_ok:
                log_action("system_command", "success", details={"action": action_key})
                log_structured("system_command_executed", action=action_key, success=True, method="capture_ops")
                return success_result(ss_msg, debug_info={"action": action_key, "method": "capture_ops"})
        except Exception as _ss_exc:
            logger.debug("capture_ops.take_screenshot failed: %s", _ss_exc)
        # fall through to native_ops path below

    if action_key in {"lock", "sleep", "screenshot"}:
        native_ok, native_msg, native_debug = _run_native_system_command(action_key)
        if native_ok:
            log_action("system_command", "success", details={"action": action_key, **native_debug})
            log_structured("system_command_executed", action=action_key, success=True, method=native_debug.get("method"))
            return success_result(native_msg, debug_info={"action": action_key, **native_debug})

    if action_key == "window_resize":
        native_ok, native_msg, native_debug = _run_native_system_command(action_key)
        if native_ok:
            log_action("system_command", "success", details={"action": action_key, **native_debug})
            log_structured("system_command_executed", action=action_key, success=True, method=native_debug.get("method"))
            return success_result(native_msg, debug_info={"action": action_key, **native_debug})
        log_action("system_command", "failed", details={"action": action_key}, error="Native window resize failed")
        log_structured("system_command_executed", action=action_key, success=False, error="native_window_resize_failed")
        return failure_result("Could not resize the active window.", error_code="execution_failed", debug_info={"action": action_key})

    _NATIVE_WINDOW_ACTIONS = {
        "window_maximize", "window_minimize",
        "window_snap_left", "window_snap_right",
        "window_next", "window_close_active",
        "browser_new_tab", "browser_close_tab",
        "browser_back", "browser_forward",
    }
    if action_key in _NATIVE_WINDOW_ACTIONS:
        native_ok, native_msg, native_debug = _run_native_window_command(action_key, language=language)
        if native_ok:
            log_action("system_command", "success", details={"action": action_key, **native_debug})
            log_structured("system_command_executed", action=action_key, success=True, method=native_debug.get("method"))
            return success_result(native_msg, debug_info={"action": action_key, **native_debug})
        log_action("system_command", "failed", details={"action": action_key}, error="native_window_command_failed")
        log_structured("system_command_executed", action=action_key, success=False, error="native_window_command_failed")
        err_msg = "مش قدرت أنفّذ الأمر." if _is_arabic_language(language) else "Could not execute window/browser command."
        return failure_result(err_msg, error_code="execution_failed", debug_info={"action": action_key})

    if action_key == "browser_close_named_tab":
        is_ar = _is_arabic_language(language)
        native_ok, native_msg, native_debug = _close_named_browser_tab(normalized_args, is_ar)
        if native_ok:
            log_action("system_command", "success", details={"action": action_key, **native_debug})
            log_structured("system_command_executed", action=action_key, success=True, method=native_debug.get("method"))
            return success_result(native_msg, debug_info={"action": action_key, **native_debug})
        log_action("system_command", "failed", details={"action": action_key, **native_debug})
        log_structured("system_command_executed", action=action_key, success=False)
        return failure_result(native_msg or "Could not find or close that browser tab.", error_code="not_found", debug_info={"action": action_key, **native_debug})

    if action_key in {"brightness_up", "brightness_down", "brightness_set"}:
        native_ok, native_msg, native_debug = _run_native_brightness_command(action_key, normalized_args)
        if native_ok:
            log_action("system_command", "success", details={"action": action_key, **native_debug})
            log_structured("system_command_executed", action=action_key, success=True, method=native_debug.get("method"))
            return success_result(native_msg, debug_info={"action": action_key, **native_debug})

    _TOGGLE_ACTIONS = {
        "night_light_on", "night_light_off",
        "dnd_on", "dnd_off",
        "energy_saver_on", "energy_saver_off",
        "live_caption_on", "live_caption_off",
    }
    if action_key in _TOGGLE_ACTIONS:
        native_ok, native_msg, native_debug = _run_native_windows_toggle(action_key, language=language)
        if native_ok:
            log_action("system_command", "success", details={"action": action_key, **native_debug})
            log_structured("system_command_executed", action=action_key, success=True, method=native_debug.get("method"))
            return success_result(native_msg, debug_info={"action": action_key, **native_debug})
        # Toggles always have a URI fallback inside windows_toggles — if we got False
        # back it means the URI was opened (user needs to confirm manually) or everything
        # failed. Return a best-effort message rather than a hard failure.
        msg = (
            "فتحت صفحة الإعدادات — غيّر الإعداد يدويًا." if _is_arabic_language(language)
            else "Settings page opened — please toggle manually."
        )
        return success_result(msg, debug_info={"action": action_key, "method": "settings_uri"})

    if action_key in _NETWORK_RADIO_ACTIONS:
        native_ok, native_msg, native_debug = _run_native_radio_command(action_key, language=language)
        if native_ok:
            log_action("system_command", "success", details={"action": action_key, **native_debug})
            log_structured("system_command_executed", action=action_key, success=True, method=native_debug.get("method"))
            return success_result(native_msg, debug_info={"action": action_key, **native_debug})

    if action_key in {"clipboard_read", "clipboard_write", "clipboard_clear"}:
        native_ok, native_msg, native_debug = _run_native_clipboard_command(action_key, normalized_args, language=language)
        log_action("system_command", "success" if native_ok else "failed", details={"action": action_key, **native_debug})
        log_structured("system_command_executed", action=action_key, success=native_ok, method=native_debug.get("method"))
        if native_ok:
            return success_result(native_msg, debug_info={"action": action_key, **native_debug})
        return failure_result("Clipboard operation failed.", error_code="execution_failed", debug_info={"action": action_key})

    if action_key == "focus_window":
        native_ok, native_msg, native_debug = _run_native_focus_window(normalized_args, language=language)
        log_action("system_command", "success" if native_ok else "failed", details={"action": action_key, **native_debug})
        log_structured("system_command_executed", action=action_key, success=native_ok, method=native_debug.get("method"))
        if native_ok:
            return success_result(native_msg, debug_info={"action": action_key, **native_debug})
        return failure_result(native_msg or "No matching window found.", error_code="not_found", debug_info={"action": action_key})

    if action_key in {"screen_record_start", "screen_record_stop"}:
        native_ok, native_msg, native_debug = _run_native_capture_command(action_key, language=language)
        log_action("system_command", "success" if native_ok else "failed", details={"action": action_key, **native_debug})
        log_structured("system_command_executed", action=action_key, success=native_ok, method=native_debug.get("method"))
        if native_ok:
            return success_result(native_msg, debug_info={"action": action_key, **native_debug})
        return failure_result(native_msg or "Recording operation failed.", error_code="execution_failed", debug_info={"action": action_key})

    ok, error, output, attempts = _run_system_template_with_safe_retry(
        cfg["template"],
        destructive=bool(cfg["destructive"]),
        env_overrides=_template_env_overrides(action_key, normalized_args),
    )
    if ok:
        message = _render_system_success_message(action_key, normalized_args, output, language=language)
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
    error_text = str(error or "").strip()
    compact_error = _compact_system_error(error_text)
    if _is_permission_denied_error(error_text):
        return failure_result(
            _permission_denied_message(action_key),
            error_code="permission_denied",
            debug_info={
                "action": action_key,
                "args": dict(normalized_args or {}),
                "attempts": attempts,
                "error_summary": compact_error,
            },
        )

    error_code = "timeout" if "timed out" in error_text.lower() else "execution_failed"
    return failure_result(
        f"Execution failed: {compact_error}",
        error_code=error_code,
        debug_info={
            "action": action_key,
            "args": dict(normalized_args or {}),
            "attempts": attempts,
            "error_summary": compact_error,
        },
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
