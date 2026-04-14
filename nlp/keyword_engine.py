"""Keyword catalog for bilingual, STT-tolerant intent classification."""

from __future__ import annotations

from typing import Dict, List

INTENTS: Dict[str, Dict[str, List[str]]] = {
    "open_youtube": {
        "actions": [
            "افتح",
            "افتحلي",
            "شغل",
            "شغللي",
            "open",
            "run",
            "launch",
            "start",
        ],
        "targets": [
            "يوتيوب",
            "يوتوب",
            "يوتيب",
            "يوتيو",
            "youtube",
            "you tube",
            "yt",
        ],
    },
    "play_music": {
        "actions": [
            "شغل",
            "شغللي",
            "play",
            "start",
            "resume",
        ],
        "targets": [
            "اغاني",
            "أغاني",
            "اغنيه",
            "أغنية",
            "مزيكا",
            "موسيقى",
            "موسيقي",
            "music",
            "song",
            "songs",
            "playlist",
        ],
    },
    "search": {
        "actions": [
            "دور",
            "دوّر",
            "دورلي",
            "دوّرلي",
            "search",
            "find",
            "look up",
            "google",
            "ابحث",
        ],
        "targets": [],
    },
    "open_chrome": {
        "actions": [
            "افتح",
            "افتحلي",
            "شغل",
            "شغللي",
            "open",
            "launch",
            "start",
            "run",
        ],
        "targets": [
            "كروم",
            "جوجل كروم",
            "chrome",
            "google chrome",
            "chrom",
            "chorme",
        ],
    },
    "open_calculator": {
        "actions": [
            "افتح",
            "افتحلي",
            "شغل",
            "شغللي",
            "open",
            "launch",
            "start",
            "run",
        ],
        "targets": [
            "حاسبة",
            "الحاسبة",
            "اله حاسبة",
            "calculator",
            "calc",
            "calclator",
        ],
    },
    "open_spotify": {
        "actions": [
            "افتح",
            "افتحلي",
            "شغل",
            "شغللي",
            "open",
            "launch",
            "start",
            "run",
        ],
        "targets": [
            "سبوتيفاي",
            "سبوتفي",
            "سبوتيفي",
            "spotify",
            "spotfy",
            "spotifiy",
        ],
    },
    "open_google": {
        "actions": [
            "افتح",
            "افتحلي",
            "open",
            "launch",
            "go to",
            "روح",
            "خش",
        ],
        "targets": [
            "جوجل",
            "google",
            "google.com",
            "جوجل دوت كوم",
        ],
    },
    "volume_up": {
        "actions": [
            "ارفع",
            "زوّد",
            "زود",
            "علي",
            "increase",
            "raise",
            "turn up",
            "volume up",
        ],
        "targets": [
            "الصوت",
            "صوت",
            "volume",
            "sound",
        ],
    },
    "volume_down": {
        "actions": [
            "وطي",
            "قلل",
            "خفّض",
            "خفض",
            "decrease",
            "lower",
            "turn down",
            "volume down",
        ],
        "targets": [
            "الصوت",
            "صوت",
            "volume",
            "sound",
        ],
    },
    "wifi_on": {
        "actions": [
            "شغل",
            "فعّل",
            "فعل",
            "وصل",
            "enable",
            "turn on",
            "start",
        ],
        "targets": [
            "واي فاي",
            "وايفاي",
            "wifi",
            "wi fi",
            "wireless",
            "النت",
        ],
    },
    "wifi_off": {
        "actions": [
            "اطفي",
            "اقفل",
            "افصل",
            "طفي",
            "disable",
            "turn off",
            "stop",
        ],
        "targets": [
            "واي فاي",
            "وايفاي",
            "wifi",
            "wi fi",
            "wireless",
            "النت",
        ],
    },
    "bluetooth_on": {
        "actions": [
            "شغل",
            "فعّل",
            "فعل",
            "enable",
            "turn on",
        ],
        "targets": [
            "بلوتوث",
            "bluetooth",
            "blue tooth",
        ],
    },
    "bluetooth_off": {
        "actions": [
            "اطفي",
            "اقفل",
            "افصل",
            "disable",
            "turn off",
        ],
        "targets": [
            "بلوتوث",
            "bluetooth",
            "blue tooth",
        ],
    },
    "screenshot": {
        "actions": [
            "خد",
            "هات",
            "التقط",
            "لقط",
            "take",
            "capture",
            "grab",
        ],
        "targets": [
            "شاشة",
            "الشاشة",
            "لقطة شاشة",
            "لقطه شاشه",
            "صورة شاشة",
            "صوره شاشه",
            "صورة للشاشة",
            "صوره للشاشه",
            "سكرين شوت",
            "سكرينشوت",
            "screenshot",
            "screen shot",
            "screen",
        ],
    },
}


def get_intents() -> Dict[str, Dict[str, List[str]]]:
    """Return the full intent keyword catalog."""
    return INTENTS


def get_intent_keywords(intent_name: str) -> List[str]:
    """Return all action+target keywords for a single intent."""
    payload = INTENTS.get(intent_name, {})
    actions = list(payload.get("actions", []))
    targets = list(payload.get("targets", []))
    return actions + targets
