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
            "You are Jarvis, a helpful, friendly, and highly capable real-time voice assistant. "
            "Support Arabic and English naturally, keep responses concise and practical, "
            "and prioritize user safety without refusing normal harmless requests."
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
    "professional": {
        "label": "Professional",
        "system_prompt": (
            "Respond in a professional and pragmatic style. "
            "Keep answers actionable, concise, and structured when needed."
        ),
        "speech_style": "calm",
        "speech_rate": 165,
    },
    "friendly": {
        "label": "Friendly",
        "system_prompt": (
            "Respond in a warm and human-like style while remaining accurate and safe. "
            "Prefer short and supportive phrasing."
        ),
        "speech_style": "friendly",
        "speech_rate": 182,
    },
    "brief": {
        "label": "Brief",
        "system_prompt": (
            "Respond with minimal words while preserving correctness and safety. "
            "Avoid long explanations unless explicitly requested."
        ),
        "speech_style": "neutral",
        "speech_rate": 190,
    },
}


PERSONA_LEXICAL_BANKS = {
    "assistant": {
        "en": {
            "gentle_prefixes": ["Understood.", "Got it."],
            "urgent_prefixes": ["On it.", "Right away."],
            "explain_bridge": "Why:",
            "codeswitch_bridge": "يمكنني المتابعة بالعربية أو English.",
        },
        "ar": {
            "gentle_prefixes": ["حاضر.", "تم."],
            "urgent_prefixes": ["حالا.", "جار التنفيذ فورا."],
            "explain_bridge": "السبب:",
            "codeswitch_bridge": "I can continue in English أو العربية.",
        },
    },
    "formal": {
        "en": {
            "gentle_prefixes": ["Certainly.", "Understood."],
            "urgent_prefixes": ["Executing immediately.", "Proceeding now."],
            "explain_bridge": "Rationale:",
            "codeswitch_bridge": "يمكنني المتابعة بالعربية أو English with the same formal tone.",
        },
        "ar": {
            "gentle_prefixes": ["بكل تاكيد.", "مفهوم."],
            "urgent_prefixes": ["سأنفذ فورا.", "جار التنفيذ مباشرة."],
            "explain_bridge": "التبرير:",
            "codeswitch_bridge": "I can continue in English او العربية بنفس الاسلوب الرسمي.",
        },
    },
    "casual": {
        "en": {
            "gentle_prefixes": ["Sure thing.", "No problem."],
            "urgent_prefixes": ["On it now.", "Doing it right now."],
            "explain_bridge": "Quick why:",
            "codeswitch_bridge": "عادي نكمل عربي أو English.",
        },
        "ar": {
            "gentle_prefixes": ["تمام.", "ولا يهمك."],
            "urgent_prefixes": ["حالا.", "تمام.. بنفذ دلوقتي."],
            "explain_bridge": "سبب سريع:",
            "codeswitch_bridge": "We can keep going in English او عربي عادي.",
        },
    },
    "professional": {
        "en": {
            "gentle_prefixes": ["Certainly.", "Noted."],
            "urgent_prefixes": ["Prioritizing now.", "Executing on priority."],
            "explain_bridge": "Execution note:",
            "codeswitch_bridge": "يمكنني الاستمرار بالعربية أو English مع نفس الدقة.",
        },
        "ar": {
            "gentle_prefixes": ["تم.", "مفهوم."],
            "urgent_prefixes": ["تم رفع الاولوية وسأنفذ الان.", "جار التنفيذ على الفور."],
            "explain_bridge": "ملاحظة تنفيذ:",
            "codeswitch_bridge": "I can continue in English او العربية بنفس الدقة.",
        },
    },
    "friendly": {
        "en": {
            "gentle_prefixes": ["Absolutely.", "Happy to help."],
            "urgent_prefixes": ["On it right now.", "You got it, doing it now."],
            "explain_bridge": "Here is why:",
            "codeswitch_bridge": "أكيد.. نقدر نكمل بالعربية أو English.",
        },
        "ar": {
            "gentle_prefixes": ["بكل سرور.", "اكيد."],
            "urgent_prefixes": ["حاضر حالا.", "اكيد.. بنفذ بسرعة."],
            "explain_bridge": "وذلك لان:",
            "codeswitch_bridge": "Sure, we can keep going in English او العربية.",
        },
    },
    "brief": {
        "en": {
            "gentle_prefixes": ["Noted.", "Okay."],
            "urgent_prefixes": ["Now.", "On it."],
            "explain_bridge": "Why:",
            "codeswitch_bridge": "عربي أو English.. as you prefer.",
        },
        "ar": {
            "gentle_prefixes": ["تم.", "حاضر."],
            "urgent_prefixes": ["الان.", "حالا."],
            "explain_bridge": "السبب:",
            "codeswitch_bridge": "English او عربي.. مثل ما تحب.",
        },
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
        if value not in {"voicecraft"}:
            return False, "Voice clone provider must be voicecraft."
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
        if value not in {"voicecraft"}:
            return False, "Voice clone provider must be voicecraft."
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

    def get_lexical_bank(self, language="en", profile=None):
        lang = "ar" if str(language or "").strip().lower() == "ar" else "en"
        with self._lock:
            selected_profile = str(profile or self._active_profile).strip().lower()
        if selected_profile not in PERSONA_PROFILES:
            selected_profile = "assistant"

        profile_banks = PERSONA_LEXICAL_BANKS.get(selected_profile) or PERSONA_LEXICAL_BANKS["assistant"]
        bank = dict(profile_banks.get(lang) or profile_banks.get("en") or {})
        bank["gentle_prefixes"] = list(bank.get("gentle_prefixes") or [])
        bank["urgent_prefixes"] = list(bank.get("urgent_prefixes") or [])
        return bank


persona_manager = PersonaManager()
