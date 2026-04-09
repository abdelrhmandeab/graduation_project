import argparse
from collections import Counter, defaultdict
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.command_classifier import classify_with_nlu
from core.command_parser import parse_command
from core.config import NLU_INTENT_CONFIDENCE_THRESHOLD, NLU_INTENT_THRESHOLD_BY_FAMILY


TARGET_ACCURACY = 0.90

_ARG_ALIAS_CANONICAL = {
    "chrome": "chrome",
    "google chrome": "chrome",
    "chrome window": "chrome",
    "كروم": "chrome",
    "جوجل كروم": "chrome",
    "calculator": "calculator",
    "calc": "calculator",
    "الحاسبة": "calculator",
    "spotify": "spotify",
    "سبوتيفاي": "spotify",
    "vlc": "vlc",
    "youtube music": "youtube music",
    "yt music": "youtube music",
    "يوتيوب ميوزك": "youtube music",
    "notepad": "notepad",
    "نوت باد": "notepad",
    "المفكرة": "notepad",
    "github": "github",
    "github.com": "github.com",
}


_SEED_SCENARIOS = [
    {
        "id": "en_open_notepad_polite",
        "language": "en",
        "utterance": "can you open notepad for me",
        "expected": {"intent": "OS_APP_OPEN"},
    },
    {
        "id": "en_open_calc_natural",
        "language": "en",
        "utterance": "i need calculator now",
        "expected": {"intent": "OS_APP_OPEN"},
    },
    {
        "id": "en_open_text_editor",
        "language": "en",
        "utterance": "launch the text editor",
        "expected": {"intent": "OS_APP_OPEN"},
    },
    {
        "id": "en_music_spotify",
        "language": "en",
        "utterance": "play music on spotify",
        "expected": {"intent": "OS_APP_OPEN"},
    },
    {
        "id": "en_find_file",
        "language": "en",
        "utterance": "find my report.pdf in documents",
        "expected": {"intent": "OS_FILE_SEARCH"},
    },
    {
        "id": "en_volume_set",
        "language": "en",
        "utterance": "set volume to 40 percent",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "volume_set", "volume_level": 40},
        },
    },
    {
        "id": "en_brightness_set",
        "language": "en",
        "utterance": "set brightness to 60",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "brightness_set", "brightness_level": 60},
        },
    },
    {
        "id": "en_focus_window",
        "language": "en",
        "utterance": "focus window chrome",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "focus_window", "window_query": "chrome"},
        },
    },
    {
        "id": "en_focus_switch",
        "language": "en",
        "utterance": "switch to the chrome window",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "focus_window", "window_query": "chrome"},
        },
    },
    {
        "id": "en_wifi_off",
        "language": "en",
        "utterance": "turn off wifi",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "wifi_off"},
        },
    },
    {
        "id": "en_bt_on",
        "language": "en",
        "utterance": "turn on bluetooth",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "bluetooth_on"},
        },
    },
    {
        "id": "en_notifications_off",
        "language": "en",
        "utterance": "disable notifications",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "notifications_off"},
        },
    },
    {
        "id": "en_dnd_on",
        "language": "en",
        "utterance": "turn on do not disturb",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "notifications_off"},
        },
    },
    {
        "id": "en_screenshot",
        "language": "en",
        "utterance": "take a screenshot",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "screenshot"},
        },
    },
    {
        "id": "en_processes",
        "language": "en",
        "utterance": "what is open right now",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "list_processes"},
        },
    },
    {
        "id": "ar_open_notepad",
        "language": "ar",
        "utterance": "افتح نوت باد من فضلك",
        "expected": {"intent": "OS_APP_OPEN"},
    },
    {
        "id": "ar_find_file",
        "language": "ar",
        "utterance": "ابحث عن ملف التقرير في المستندات",
        "expected": {"intent": "OS_FILE_SEARCH"},
    },
    {
        "id": "ar_volume_set",
        "language": "ar",
        "utterance": "اضبط الصوت على 60",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "volume_set", "volume_level": 60},
        },
    },
    {
        "id": "ar_brightness_set",
        "language": "ar",
        "utterance": "اضبط السطوع على 40",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "brightness_set", "brightness_level": 40},
        },
    },
    {
        "id": "ar_focus_window",
        "language": "ar",
        "utterance": "ركز على نافذة كروم",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "focus_window", "window_query": "كروم"},
        },
    },
    {
        "id": "ar_screenshot",
        "language": "ar",
        "utterance": "خذ صورة للشاشة",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "screenshot"},
        },
    },
    {
        "id": "ar_wifi_off",
        "language": "ar",
        "utterance": "افصل الانترنت",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "wifi_off"},
        },
    },
    {
        "id": "ar_bt_on",
        "language": "ar",
        "utterance": "شغل البلوتوث",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "bluetooth_on"},
        },
    },
    {
        "id": "ar_notifications_off",
        "language": "ar",
        "utterance": "اقفل الاشعارات",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "notifications_off"},
        },
    },
    {
        "id": "ar_dnd_on",
        "language": "ar",
        "utterance": "وضع عدم الازعاج",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "notifications_off"},
        },
    },
    {
        "id": "en_window_maximize",
        "language": "en",
        "utterance": "maximize the current window",
        "category": "window",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "window_maximize"},
        },
    },
    {
        "id": "en_window_snap_left",
        "language": "en",
        "utterance": "snap this window to the left",
        "category": "window",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "window_snap_left"},
        },
    },
    {
        "id": "en_media_pause",
        "language": "en",
        "utterance": "pause the music",
        "category": "media",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "media_play_pause"},
        },
    },
    {
        "id": "en_media_next",
        "language": "en",
        "utterance": "next track",
        "category": "media",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "media_next_track"},
        },
    },
    {
        "id": "en_media_seek_forward",
        "language": "en",
        "utterance": "skip forward 15 seconds",
        "category": "media",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "media_seek_forward", "seek_seconds": 15},
        },
    },
    {
        "id": "en_browser_new_tab",
        "language": "en",
        "utterance": "open a new tab",
        "category": "browser",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "browser_new_tab"},
        },
    },
    {
        "id": "en_browser_back",
        "language": "en",
        "utterance": "go back in the browser",
        "category": "browser",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "browser_back"},
        },
    },
    {
        "id": "en_browser_open_url",
        "language": "en",
        "utterance": "open github.com",
        "category": "browser",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "browser_open_url", "url": "https://github.com"},
        },
    },
    {
        "id": "en_browser_search_web",
        "language": "en",
        "utterance": "search the web for python dataclasses",
        "category": "browser",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "browser_search_web", "search_query": "python dataclasses"},
        },
    },
    {
        "id": "en_schedule_in_minutes",
        "language": "en",
        "utterance": "in 5 minutes mute volume",
        "category": "scheduling",
        "expected": {
            "intent": "JOB_QUEUE_COMMAND",
            "action": "enqueue",
            "args": {"delay_seconds": 300, "command_text": "mute volume"},
        },
    },
    {
        "id": "en_schedule_after_seconds",
        "language": "en",
        "utterance": "after 30 seconds take a screenshot",
        "category": "scheduling",
        "expected": {
            "intent": "JOB_QUEUE_COMMAND",
            "action": "enqueue",
            "args": {"delay_seconds": 30, "command_text": "take a screenshot"},
        },
    },
    {
        "id": "en_volume_spoken_number",
        "language": "en",
        "utterance": "set volume to forty percent",
        "category": "robustness",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "volume_set", "volume_level": 40},
        },
    },
    {
        "id": "ar_window_maximize",
        "language": "ar",
        "utterance": "كبر النافذة",
        "category": "window",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "window_maximize"},
        },
    },
    {
        "id": "ar_media_next",
        "language": "ar",
        "utterance": "الاغنية التالية",
        "category": "media",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "media_next_track"},
        },
    },
    {
        "id": "ar_browser_new_tab",
        "language": "ar",
        "utterance": "افتح تبويب جديد",
        "category": "browser",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "browser_new_tab"},
        },
    },
    {
        "id": "ar_schedule_after_minutes",
        "language": "ar",
        "utterance": "بعد 10 دقائق خذ صورة للشاشة",
        "category": "scheduling",
        "expected": {
            "intent": "JOB_QUEUE_COMMAND",
            "action": "enqueue",
            "args": {"delay_seconds": 600, "command_text": "خذ صورة للشاشة"},
        },
    },
    {
        "id": "ar_brightness_spoken_number",
        "language": "ar",
        "utterance": "اضبط السطوع على ستين",
        "category": "robustness",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "brightness_set", "brightness_level": 60},
        },
    },
    {
        "id": "en_file_create_free_phrase",
        "language": "en",
        "utterance": "create a folder called worknotes on desktop",
        "category": "file_ops",
        "expected": {
            "intent": "OS_FILE_NAVIGATION",
            "action": "create_directory",
        },
    },
    {
        "id": "ar_file_delete_free_phrase",
        "language": "ar",
        "utterance": "احذف ملف temp.txt",
        "category": "file_ops",
        "expected": {
            "intent": "OS_FILE_NAVIGATION",
            "action": "delete_item",
        },
    },
]

_EXTRA_SCENARIOS = [
    {
        "id": "en_open_edge",
        "language": "en",
        "utterance": "open microsoft edge",
        "category": "app_open",
        "expected": {"intent": "OS_APP_OPEN"},
    },
    {
        "id": "en_open_chrome_fast",
        "language": "en",
        "utterance": "please launch chrome now",
        "category": "app_open",
        "expected": {"intent": "OS_APP_OPEN"},
    },
    {
        "id": "en_close_notepad",
        "language": "en",
        "utterance": "close notepad",
        "category": "app_close",
        "expected": {"intent": "OS_APP_CLOSE"},
    },
    {
        "id": "en_find_budget_downloads",
        "language": "en",
        "utterance": "find budget.xlsx in downloads",
        "category": "file_search",
        "expected": {"intent": "OS_FILE_SEARCH"},
    },
    {
        "id": "en_find_notes_desktop",
        "language": "en",
        "utterance": "search for notes.txt inside desktop",
        "category": "file_search",
        "expected": {"intent": "OS_FILE_SEARCH"},
    },
    {
        "id": "en_volume_mute",
        "language": "en",
        "utterance": "mute volume",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "volume_mute"}},
    },
    {
        "id": "en_volume_up",
        "language": "en",
        "utterance": "turn it up",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "volume_up"}},
    },
    {
        "id": "en_volume_down",
        "language": "en",
        "utterance": "turn it down",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "volume_down"}},
    },
    {
        "id": "en_brightness_up",
        "language": "en",
        "utterance": "increase brightness",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "brightness_up"}},
    },
    {
        "id": "en_brightness_down",
        "language": "en",
        "utterance": "dim the screen",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "brightness_down"}},
    },
    {
        "id": "en_wifi_on",
        "language": "en",
        "utterance": "enable wifi",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "wifi_on"}},
    },
    {
        "id": "en_bt_off",
        "language": "en",
        "utterance": "disable bluetooth",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "bluetooth_off"}},
    },
    {
        "id": "en_notifications_on",
        "language": "en",
        "utterance": "enable notifications",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "notifications_on"}},
    },
    {
        "id": "en_dnd_off",
        "language": "en",
        "utterance": "turn off do not disturb",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "notifications_on"}},
    },
    {
        "id": "en_window_minimize",
        "language": "en",
        "utterance": "minimize this window",
        "category": "window",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "window_minimize"}},
    },
    {
        "id": "en_window_next",
        "language": "en",
        "utterance": "switch window",
        "category": "window",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "window_next"}},
    },
    {
        "id": "en_window_close_active",
        "language": "en",
        "utterance": "close this window",
        "category": "window",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "window_close_active"}},
    },
    {
        "id": "en_media_previous",
        "language": "en",
        "utterance": "previous track",
        "category": "media",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "media_previous_track"}},
    },
    {
        "id": "en_media_stop",
        "language": "en",
        "utterance": "stop media",
        "category": "media",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "media_stop"}},
    },
    {
        "id": "en_browser_close_tab",
        "language": "en",
        "utterance": "close tab",
        "category": "browser",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "browser_close_tab"}},
    },
    {
        "id": "en_browser_forward",
        "language": "en",
        "utterance": "go forward in browser",
        "category": "browser",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "browser_forward"}},
    },
    {
        "id": "en_browser_search_alt",
        "language": "en",
        "utterance": "google weather tomorrow",
        "category": "browser",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "browser_search_web", "search_query": "weather tomorrow"},
        },
    },
    {
        "id": "en_schedule_remind",
        "language": "en",
        "utterance": "remind me in 20 minutes to lock computer",
        "category": "scheduling",
        "expected": {
            "intent": "JOB_QUEUE_COMMAND",
            "action": "enqueue",
            "args": {"delay_seconds": 1200, "command_text": "lock computer"},
        },
    },
    {
        "id": "en_schedule_after_hour",
        "language": "en",
        "utterance": "after 1 hour turn off wifi",
        "category": "scheduling",
        "expected": {
            "intent": "JOB_QUEUE_COMMAND",
            "action": "enqueue",
            "args": {"delay_seconds": 3600, "command_text": "turn off wifi"},
        },
    },
    {
        "id": "en_file_move_phrase",
        "language": "en",
        "utterance": "move report.docx to documents",
        "category": "file_ops",
        "expected": {"intent": "OS_FILE_NAVIGATION", "action": "move_item"},
    },
    {
        "id": "en_file_rename_phrase",
        "language": "en",
        "utterance": "rename report.docx to report-final.docx",
        "category": "file_ops",
        "expected": {"intent": "OS_FILE_NAVIGATION", "action": "rename_item"},
    },
    {
        "id": "en_file_delete_perm",
        "language": "en",
        "utterance": "delete temp.log permanently",
        "category": "file_ops",
        "expected": {"intent": "OS_FILE_NAVIGATION", "action": "delete_item_permanent"},
    },
    {
        "id": "ar_open_chrome_dialect",
        "language": "ar",
        "utterance": "شغل كروم",
        "category": "app_open",
        "expected": {"intent": "OS_APP_OPEN"},
    },
    {
        "id": "ar_open_calc_dialect",
        "language": "ar",
        "utterance": "افتح الحاسبة بسرعة",
        "category": "app_open",
        "expected": {"intent": "OS_APP_OPEN"},
    },
    {
        "id": "ar_close_notepad",
        "language": "ar",
        "utterance": "اغلق نوت باد",
        "category": "app_close",
        "expected": {"intent": "OS_APP_CLOSE"},
    },
    {
        "id": "ar_find_budget_downloads",
        "language": "ar",
        "utterance": "دور على ملف budget.xlsx في التحميلات",
        "category": "file_search",
        "expected": {"intent": "OS_FILE_SEARCH"},
    },
    {
        "id": "ar_find_notes_desktop",
        "language": "ar",
        "utterance": "ابحث عن notes.txt على سطح المكتب",
        "category": "file_search",
        "expected": {"intent": "OS_FILE_SEARCH"},
    },
    {
        "id": "ar_volume_mute",
        "language": "ar",
        "utterance": "اكتم الصوت",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "volume_mute"}},
    },
    {
        "id": "ar_volume_up",
        "language": "ar",
        "utterance": "ارفع الصوت",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "volume_up"}},
    },
    {
        "id": "ar_volume_down",
        "language": "ar",
        "utterance": "وطي الصوت",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "volume_down"}},
    },
    {
        "id": "ar_brightness_up",
        "language": "ar",
        "utterance": "زود السطوع",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "brightness_up"}},
    },
    {
        "id": "ar_brightness_down",
        "language": "ar",
        "utterance": "قلل الإضاءة",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "brightness_down"}},
    },
    {
        "id": "ar_wifi_on",
        "language": "ar",
        "utterance": "شغل الواي فاي",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "wifi_on"}},
    },
    {
        "id": "ar_bt_off",
        "language": "ar",
        "utterance": "اطفي البلوتوث",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "bluetooth_off"}},
    },
    {
        "id": "ar_notifications_on",
        "language": "ar",
        "utterance": "شغل الاشعارات",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "notifications_on"}},
    },
    {
        "id": "ar_dnd_off",
        "language": "ar",
        "utterance": "اقفل وضع عدم الازعاج",
        "category": "system",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "notifications_on"}},
    },
    {
        "id": "ar_window_minimize",
        "language": "ar",
        "utterance": "صغر النافذة الحالية",
        "category": "window",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "window_minimize"}},
    },
    {
        "id": "ar_window_next",
        "language": "ar",
        "utterance": "روح للنافذة التالية",
        "category": "window",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "window_next"}},
    },
    {
        "id": "ar_window_close_active",
        "language": "ar",
        "utterance": "اغلق النافذة الحالية",
        "category": "window",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "window_close_active"}},
    },
    {
        "id": "ar_media_previous",
        "language": "ar",
        "utterance": "الأغنية اللي قبلها",
        "category": "media",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "media_previous_track"}},
    },
    {
        "id": "ar_media_stop",
        "language": "ar",
        "utterance": "وقف التشغيل",
        "category": "media",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "media_stop"}},
    },
    {
        "id": "ar_browser_close_tab",
        "language": "ar",
        "utterance": "اقفل التبويب",
        "category": "browser",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "browser_close_tab"}},
    },
    {
        "id": "ar_browser_forward",
        "language": "ar",
        "utterance": "اذهب للأمام في المتصفح",
        "category": "browser",
        "expected": {"intent": "OS_SYSTEM_COMMAND", "args": {"action_key": "browser_forward"}},
    },
    {
        "id": "ar_browser_search_alt",
        "language": "ar",
        "utterance": "ابحث في جوجل عن الطقس بكرة",
        "category": "browser",
        "expected": {
            "intent": "OS_SYSTEM_COMMAND",
            "args": {"action_key": "browser_search_web", "search_query": "الطقس بكرة"},
        },
    },
    {
        "id": "ar_schedule_after_seconds",
        "language": "ar",
        "utterance": "بعد 45 ثانية اكتم الصوت",
        "category": "scheduling",
        "expected": {
            "intent": "JOB_QUEUE_COMMAND",
            "action": "enqueue",
            "args": {"delay_seconds": 45, "command_text": "اكتم الصوت"},
        },
    },
    {
        "id": "ar_schedule_after_hours",
        "language": "ar",
        "utterance": "بعد 2 ساعات افصل الانترنت",
        "category": "scheduling",
        "expected": {
            "intent": "JOB_QUEUE_COMMAND",
            "action": "enqueue",
            "args": {"delay_seconds": 7200, "command_text": "افصل الانترنت"},
        },
    },
    {
        "id": "ar_file_move_phrase",
        "language": "ar",
        "utterance": "انقل التقرير.docx الى المستندات",
        "category": "file_ops",
        "expected": {"intent": "OS_FILE_NAVIGATION", "action": "move_item"},
    },
    {
        "id": "ar_file_rename_phrase",
        "language": "ar",
        "utterance": "غير اسم التقرير.docx الى تقرير-نهائي.docx",
        "category": "file_ops",
        "expected": {"intent": "OS_FILE_NAVIGATION", "action": "rename_item"},
    },
    {
        "id": "ar_file_delete_perm",
        "language": "ar",
        "utterance": "احذف الملف temp.log نهائيا",
        "category": "file_ops",
        "expected": {"intent": "OS_FILE_NAVIGATION", "action": "delete_item_permanent"},
    },
]


def _norm_case_text(value):
    return " ".join(str(value or "").strip().lower().split())


def _clone_case(case, case_id, utterance):
    cloned = {
        "id": case_id,
        "language": case.get("language"),
        "utterance": utterance,
        "expected": dict(case.get("expected") or {}),
    }
    if case.get("category"):
        cloned["category"] = case.get("category")
    return cloned


def _noisy_variants(utterance: str, language: str):
    text = str(utterance or "").strip()
    if not text:
        return []

    variants = []
    if language == "en":
        typo_variant = text
        typo_variant = re.sub(r"\bbluetooth\b", "blutooth", typo_variant, flags=re.IGNORECASE)
        typo_variant = re.sub(r"\bbrowser\b", "broser", typo_variant, flags=re.IGNORECASE)
        typo_variant = re.sub(r"\bwifi\b", "wi fi", typo_variant, flags=re.IGNORECASE)
        typo_variant = re.sub(r"\bminutes\b", "mins", typo_variant, flags=re.IGNORECASE)
        variants.extend(
            [
                f"hey jarvis {text}",
                f"{text} please",
                f"uh {text} right now",
                typo_variant,
                re.sub(r"\s+", "  ", text),
                text + " ???",
            ]
        )
    else:
        dialect_variant = text
        dialect_variant = dialect_variant.replace("الانترنت", "النت")
        dialect_variant = dialect_variant.replace("اضبط", "زبط")
        dialect_variant = dialect_variant.replace("افتح", "افتحلي")
        variants.extend(
            [
                f"يا جارفيس {text}",
                f"{text} لو سمحت",
                dialect_variant,
                text.replace("أ", "ا").replace("إ", "ا"),
                re.sub(r"\s+", "  ", text),
                text + " ؟؟",
            ]
        )

    return [v.strip() for v in variants if v and _norm_case_text(v) != _norm_case_text(text)]


def build_scenarios(target_per_language=90, include_noisy_variants=True):
    target = max(80, min(120, int(target_per_language or 90)))
    source_cases = list(_SEED_SCENARIOS) + list(_EXTRA_SCENARIOS)
    grouped = {"en": [], "ar": []}
    seen = {"en": set(), "ar": set()}

    for case in source_cases:
        language = _norm_case_text(case.get("language"))
        if language not in grouped:
            continue
        utter_key = _norm_case_text(case.get("utterance"))
        if not utter_key or utter_key in seen[language]:
            continue
        grouped[language].append(_clone_case(case, case.get("id"), case.get("utterance")))
        seen[language].add(utter_key)

    if include_noisy_variants:
        for language in ("en", "ar"):
            base_cases = list(grouped[language])
            if not base_cases:
                continue

            index = 0
            while len(grouped[language]) < target and index < (target * 12):
                source = base_cases[index % len(base_cases)]
                variants = _noisy_variants(source.get("utterance", ""), language)
                if variants:
                    variant = variants[(index // len(base_cases)) % len(variants)]
                    key = _norm_case_text(variant)
                    if key and key not in seen[language]:
                        new_id = f"{source['id']}_noisy_{len(grouped[language]) + 1}"
                        grouped[language].append(_clone_case(source, new_id, variant))
                        seen[language].add(key)
                index += 1

    for language in ("en", "ar"):
        grouped[language] = grouped[language][:target]

    return grouped["en"] + grouped["ar"]


SCENARIOS = build_scenarios(target_per_language=90, include_noisy_variants=True)


def _norm_text(value):
    return " ".join(str(value or "").strip().lower().split())


def _canonical_alias(value):
    text = _norm_text(value)
    if not text:
        return ""
    direct = _ARG_ALIAS_CANONICAL.get(text)
    if direct:
        return direct
    for alias, canonical in sorted(_ARG_ALIAS_CANONICAL.items(), key=lambda item: len(item[0]), reverse=True):
        if (
            text.startswith(alias + " ")
            or text.endswith(" " + alias)
            or (" " + alias + " ") in (" " + text + " ")
        ):
            return canonical
    return text


def _extract_int(value):
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value or "")
    match = re.search(r"\d+", text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except (TypeError, ValueError):
        return None


def _match_value(expected, predicted):
    if isinstance(expected, (int, float)):
        pred_num = _extract_int(predicted)
        return pred_num is not None and int(expected) == int(pred_num)

    e = _canonical_alias(expected)
    p = _canonical_alias(predicted)
    if not e:
        return not p
    if e == p:
        return True
    if e in p or p in e:
        return True
    return False


def _match_expected(expected, predicted):
    if _norm_text(expected.get("intent")) != _norm_text(predicted.get("intent")):
        return False, "intent_mismatch"

    expected_action = expected.get("action")
    if expected_action and _norm_text(expected_action) != _norm_text(predicted.get("action")):
        return False, "action_mismatch"

    expected_args = dict(expected.get("args") or {})
    predicted_args = dict(predicted.get("args") or {})
    for key, value in expected_args.items():
        if key not in predicted_args:
            return False, f"missing_arg:{key}"
        if not _match_value(value, predicted_args.get(key)):
            return False, f"arg_mismatch:{key}"

    return True, "ok"


def _baseline_prediction(text):
    parsed = parse_command(text)
    return {
        "intent": parsed.intent,
        "action": parsed.action,
        "args": dict(parsed.args or {}),
        "source": "parser",
        "confidence": None,
    }


def _improved_prediction(text, language):
    nlu = classify_with_nlu(text, language=language)
    nlu_conf = float(nlu.get("confidence") or 0.0)
    nlu_intent = str(nlu.get("intent") or "")
    nlu_threshold = float(
        NLU_INTENT_THRESHOLD_BY_FAMILY.get(
            nlu_intent,
            NLU_INTENT_CONFIDENCE_THRESHOLD,
        )
    )
    if (
        nlu.get("ok")
        and nlu_intent != "LLM_QUERY"
        and nlu_conf >= nlu_threshold
    ):
        return {
            "intent": nlu_intent or "LLM_QUERY",
            "action": str(nlu.get("action") or ""),
            "args": dict(nlu.get("args") or {}),
            "source": "nlu",
            "confidence": nlu_conf,
        }

    parsed = parse_command(text)
    return {
        "intent": parsed.intent,
        "action": parsed.action,
        "args": dict(parsed.args or {}),
        "source": "parser_fallback",
        "confidence": nlu_conf,
    }


def _scenario_category(case):
    category = _norm_text(case.get("category"))
    if category:
        return category

    expected = dict(case.get("expected") or {})
    intent = _norm_text(expected.get("intent"))
    if intent == "os_system_command":
        action_key = _norm_text((expected.get("args") or {}).get("action_key"))
        if action_key.startswith("window_"):
            return "window"
        if action_key.startswith("media_"):
            return "media"
        if action_key.startswith("browser_"):
            return "browser"
        return "system"
    if intent == "job_queue_command":
        return "scheduling"
    if intent == "os_file_navigation":
        return "file_ops"
    if intent == "os_file_search":
        return "file_search"
    if intent == "os_app_open":
        return "app_open"
    return intent or "other"


def _increment_bucket(bucket_map, key, ok):
    bucket = bucket_map[key]
    bucket["total"] += 1
    if ok:
        bucket["correct"] += 1


def _finalize_accuracy_buckets(bucket_map):
    finalized = {}
    for key, data in sorted(bucket_map.items()):
        total = int(data.get("total") or 0)
        correct = int(data.get("correct") or 0)
        accuracy = (correct / total) if total else 0.0
        finalized[key] = {
            "correct": correct,
            "total": total,
            "accuracy": round(accuracy, 4),
        }
    return finalized


def _finalize_confusion(confusion_map):
    return {
        expected: dict(sorted(predictions.items(), key=lambda item: item[0]))
        for expected, predictions in sorted(confusion_map.items(), key=lambda item: item[0])
    }


def run_benchmark(limit=None, scenarios=None):
    corpus = list(scenarios or SCENARIOS)
    cases = corpus[: limit or len(corpus)]
    baseline_correct = 0
    improved_correct = 0
    details = []
    by_category_baseline = defaultdict(lambda: {"correct": 0, "total": 0})
    by_category_improved = defaultdict(lambda: {"correct": 0, "total": 0})
    by_language_baseline = defaultdict(lambda: {"correct": 0, "total": 0})
    by_language_improved = defaultdict(lambda: {"correct": 0, "total": 0})
    baseline_confusion = defaultdict(Counter)
    improved_confusion = defaultdict(Counter)
    source_usage = Counter()
    regressions = []
    fixes = []

    for case in cases:
        expected = case["expected"]
        category = _scenario_category(case)
        language = _norm_text(case.get("language") or "unknown")
        baseline = _baseline_prediction(case["utterance"])
        improved = _improved_prediction(case["utterance"], case["language"])
        source_usage[improved.get("source") or "unknown"] += 1

        baseline_ok, baseline_reason = _match_expected(expected, baseline)
        improved_ok, improved_reason = _match_expected(expected, improved)

        if baseline_ok:
            baseline_correct += 1
        if improved_ok:
            improved_correct += 1

        _increment_bucket(by_category_baseline, category, baseline_ok)
        _increment_bucket(by_category_improved, category, improved_ok)
        _increment_bucket(by_language_baseline, language, baseline_ok)
        _increment_bucket(by_language_improved, language, improved_ok)

        expected_intent = _norm_text(expected.get("intent") or "unknown")
        baseline_intent = _norm_text(baseline.get("intent") or "unknown")
        improved_intent = _norm_text(improved.get("intent") or "unknown")
        baseline_confusion[expected_intent][baseline_intent] += 1
        improved_confusion[expected_intent][improved_intent] += 1

        if baseline_ok and not improved_ok:
            regressions.append(
                {
                    "id": case["id"],
                    "category": category,
                    "baseline_reason": baseline_reason,
                    "improved_reason": improved_reason,
                }
            )
        if (not baseline_ok) and improved_ok:
            fixes.append(
                {
                    "id": case["id"],
                    "category": category,
                    "baseline_reason": baseline_reason,
                }
            )

        details.append(
            {
                "id": case["id"],
                "utterance": case["utterance"],
                "language": language,
                "category": category,
                "expected": expected,
                "baseline": {
                    "ok": baseline_ok,
                    "reason": baseline_reason,
                    "prediction": baseline,
                },
                "improved": {
                    "ok": improved_ok,
                    "reason": improved_reason,
                    "prediction": improved,
                },
            }
        )

    total = len(cases)
    baseline_accuracy = (baseline_correct / total) if total else 0.0
    improved_accuracy = (improved_correct / total) if total else 0.0
    parser_fallback_count = int(source_usage.get("parser_fallback") or 0)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_accuracy": TARGET_ACCURACY,
        "nlu_threshold": float(NLU_INTENT_CONFIDENCE_THRESHOLD),
        "nlu_threshold_by_family": {
            k: float(v)
            for k, v in sorted(NLU_INTENT_THRESHOLD_BY_FAMILY.items(), key=lambda item: item[0])
        },
        "total_cases": total,
        "baseline": {
            "correct": baseline_correct,
            "accuracy": round(baseline_accuracy, 4),
        },
        "improved": {
            "correct": improved_correct,
            "accuracy": round(improved_accuracy, 4),
            "meets_target": bool(improved_accuracy >= TARGET_ACCURACY),
            "source_usage": dict(sorted(source_usage.items(), key=lambda item: item[0])),
            "parser_fallback_rate": round((parser_fallback_count / total) if total else 0.0, 4),
        },
        "delta_accuracy": round(improved_accuracy - baseline_accuracy, 4),
        "by_category": {
            "baseline": _finalize_accuracy_buckets(by_category_baseline),
            "improved": _finalize_accuracy_buckets(by_category_improved),
        },
        "by_language": {
            "baseline": _finalize_accuracy_buckets(by_language_baseline),
            "improved": _finalize_accuracy_buckets(by_language_improved),
        },
        "confusion": {
            "baseline": _finalize_confusion(baseline_confusion),
            "improved": _finalize_confusion(improved_confusion),
        },
        "regressions": regressions,
        "fixes": fixes,
        "details": details,
    }


def _print_summary(report):
    total = report["total_cases"]
    baseline = report["baseline"]
    improved = report["improved"]
    print("Phase 1 Intent Benchmark")
    print("------------------------")
    print(f"cases: {total}")
    print(f"nlu_threshold: {report['nlu_threshold']:.2f}")
    threshold_map = dict(report.get("nlu_threshold_by_family") or {})
    if threshold_map:
        threshold_items = ", ".join(
            f"{intent}:{value:.2f}" for intent, value in sorted(threshold_map.items(), key=lambda item: item[0])
        )
        print(f"nlu_threshold_by_family: {threshold_items}")
    print(
        f"baseline: {baseline['correct']}/{total} "
        f"({baseline['accuracy'] * 100:.2f}%)"
    )
    print(
        f"improved: {improved['correct']}/{total} "
        f"({improved['accuracy'] * 100:.2f}%)"
    )
    print(f"delta: {(report['delta_accuracy'] * 100):.2f}%")
    print(f"parser_fallback_rate: {improved.get('parser_fallback_rate', 0.0) * 100:.2f}%")
    source_usage = dict(improved.get("source_usage") or {})
    if source_usage:
        source_items = ", ".join(f"{k}:{v}" for k, v in sorted(source_usage.items(), key=lambda item: item[0]))
        print(f"improved_sources: {source_items}")
    print(f"regressions: {len(report.get('regressions') or [])}")
    print(f"fixes: {len(report.get('fixes') or [])}")

    category_rows = report.get("by_category", {}).get("improved", {})
    if category_rows:
        print("category_accuracy_improved:")
        for category, stats in category_rows.items():
            print(
                f"- {category}: {stats['correct']}/{stats['total']} "
                f"({stats['accuracy'] * 100:.2f}%)"
            )

    language_rows = report.get("by_language", {}).get("improved", {})
    if language_rows:
        print("language_accuracy_improved:")
        for language, stats in language_rows.items():
            print(
                f"- {language}: {stats['correct']}/{stats['total']} "
                f"({stats['accuracy'] * 100:.2f}%)"
            )

    print(
        "target_90_percent: "
        + ("PASS" if improved["meets_target"] else "FAIL")
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark baseline parser vs NLU-improved intent routing.")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "jarvis_phase1_intent_benchmark.json"),
        help="Output JSON report path.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on number of scenarios.")
    parser.add_argument(
        "--target-per-language",
        type=int,
        default=90,
        help="Target scenarios per language (bounded to 80-120).",
    )
    parser.add_argument(
        "--disable-noisy-variants",
        action="store_true",
        help="Disable deterministic noisy/dialect variant generation.",
    )
    args = parser.parse_args()

    scenarios = build_scenarios(
        target_per_language=args.target_per_language,
        include_noisy_variants=not bool(args.disable_noisy_variants),
    )
    report = run_benchmark(limit=(args.limit or None), scenarios=scenarios)
    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    _print_summary(report)
    print(f"report_file: {output_path}")


if __name__ == "__main__":
    main()
