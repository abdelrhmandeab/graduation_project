import os
import threading

from core.config import (
    PERSONA_DEFAULT,
    VOICE_CLONE_ENABLED,
    VOICE_CLONE_PROVIDER,
    VOICE_CLONE_REFERENCE_AUDIO,
)

PERSONA_PROFILES = {
    "assistant": {
        "label": "Assistant",
        "system_prompt": (
            "You are JARVIS, a reliable local AI assistant. "
            "Give concise, practical answers and prioritize user safety. "
            "Do not refuse normal harmless requests."
        ),
        "speech_style": "neutral",
        "speech_rate": 175,
    },
    "formal": {
        "label": "Formal",
        "system_prompt": (
            "Respond in a formal and professional tone. "
            "Use precise language and structured explanations."
        ),
        "speech_style": "calm",
        "speech_rate": 160,
    },
    "casual": {
        "label": "Casual",
        "system_prompt": (
            "Respond in a casual, friendly tone while staying clear and accurate."
        ),
        "speech_style": "friendly",
        "speech_rate": 185,
    },
}


class PersonaManager:
    def __init__(self):
        self._lock = threading.Lock()
        default_profile = PERSONA_DEFAULT if PERSONA_DEFAULT in PERSONA_PROFILES else "assistant"
        self._active_profile = default_profile
        self._voice_profiles = {}
        for profile_name in PERSONA_PROFILES:
            self._voice_profiles[profile_name] = {
                "clone_enabled": bool(VOICE_CLONE_ENABLED),
                "clone_provider": VOICE_CLONE_PROVIDER,
                "reference_audio": (VOICE_CLONE_REFERENCE_AUDIO or "").strip(),
            }

    def list_profiles(self):
        return sorted(PERSONA_PROFILES.keys())

    def get_profile(self):
        with self._lock:
            return self._active_profile

    def set_profile(self, name):
        key = (name or "").strip().lower()
        if key not in PERSONA_PROFILES:
            return False, f"Unknown persona: {name}"
        with self._lock:
            self._active_profile = key
        return True, f"Persona set to: {key}"

    def get_system_prompt(self):
        with self._lock:
            profile = self._active_profile
        return PERSONA_PROFILES[profile]["system_prompt"]

    def get_speech_style(self):
        with self._lock:
            profile = self._active_profile
        return PERSONA_PROFILES[profile]["speech_style"]

    def get_speech_rate(self):
        with self._lock:
            profile = self._active_profile
        return int(PERSONA_PROFILES[profile].get("speech_rate", 175))

    def set_clone_enabled(self, enabled):
        with self._lock:
            profile = self._active_profile
            self._voice_profiles[profile]["clone_enabled"] = bool(enabled)
        return True, f"Voice cloning {'enabled' if enabled else 'disabled'}."

    def set_clone_provider(self, provider):
        value = (provider or "").strip().lower()
        if value not in {"xtts", "voicecraft"}:
            return False, "Voice clone provider must be `xtts` or `voicecraft`."
        with self._lock:
            profile = self._active_profile
            self._voice_profiles[profile]["clone_provider"] = value
        return True, f"Voice clone provider set to: {value}"

    def set_clone_reference_audio(self, path):
        if not path:
            return False, "Reference audio path is required."
        resolved = os.path.abspath(os.path.expanduser(path.strip().strip('"').strip("'")))
        if not os.path.isfile(resolved):
            return False, f"Reference audio file does not exist: {resolved}"
        with self._lock:
            profile = self._active_profile
            self._voice_profiles[profile]["reference_audio"] = resolved
        return True, f"Voice clone reference audio set: {resolved}"

    def get_clone_settings(self):
        with self._lock:
            profile = self._active_profile
            selected = dict(self._voice_profiles.get(profile, {}))
            selected["profile"] = profile
            return {
                "enabled": bool(selected.get("clone_enabled", False)),
                "provider": selected.get("clone_provider", VOICE_CLONE_PROVIDER),
                "reference_audio": selected.get("reference_audio", ""),
                "profile": profile,
            }

    def set_profile_clone_enabled(self, profile, enabled):
        key = (profile or "").strip().lower()
        if key not in PERSONA_PROFILES:
            return False, f"Unknown persona profile: {profile}"
        with self._lock:
            self._voice_profiles[key]["clone_enabled"] = bool(enabled)
        return True, f"Persona {key} clone_enabled set to {bool(enabled)}"

    def set_profile_clone_provider(self, profile, provider):
        key = (profile or "").strip().lower()
        value = (provider or "").strip().lower()
        if key not in PERSONA_PROFILES:
            return False, f"Unknown persona profile: {profile}"
        if value not in {"xtts", "voicecraft"}:
            return False, "Voice clone provider must be `xtts` or `voicecraft`."
        with self._lock:
            self._voice_profiles[key]["clone_provider"] = value
        return True, f"Persona {key} clone_provider set to {value}"

    def set_profile_clone_reference_audio(self, profile, path):
        key = (profile or "").strip().lower()
        if key not in PERSONA_PROFILES:
            return False, f"Unknown persona profile: {profile}"
        resolved = os.path.abspath(os.path.expanduser((path or "").strip().strip('"').strip("'")))
        if not os.path.isfile(resolved):
            return False, f"Reference audio file does not exist: {resolved}"
        with self._lock:
            self._voice_profiles[key]["reference_audio"] = resolved
        return True, f"Persona {key} reference audio set: {resolved}"

    def profile_voice_map(self):
        with self._lock:
            return {k: dict(v) for k, v in self._voice_profiles.items()}

    def status(self):
        with self._lock:
            active = self._active_profile
            voice_profiles = {k: dict(v) for k, v in self._voice_profiles.items()}
            active_voice = voice_profiles.get(active, {})

        return {
            "active_profile": active,
            "available_profiles": self.list_profiles(),
            "speech_style": PERSONA_PROFILES[active]["speech_style"],
            "speech_rate": int(PERSONA_PROFILES[active].get("speech_rate", 175)),
            "clone_enabled": bool(active_voice.get("clone_enabled", False)),
            "clone_provider": active_voice.get("clone_provider", VOICE_CLONE_PROVIDER),
            "clone_reference_audio": active_voice.get("reference_audio", ""),
            "voice_profiles": voice_profiles,
        }


persona_manager = PersonaManager()
