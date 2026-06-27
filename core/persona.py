import os
import threading

from core.config import (
    PERSONA_DEFAULT,
    PERSONA_ADDRESSEE_AR,
    PERSONA_ADDRESSEE_EN,
    PERSONA_FORBIDDEN_AR,
    PERSONA_FORBIDDEN_EN,
    PERSONA_NAME,
    PERSONA_PROFILE,
    PERSONA_STYLE_AR,
    PERSONA_STYLE_EN,
    PERSONA_VOICE_LENGTH_AR,
    PERSONA_VOICE_LENGTH_EN,
)


DEFAULT_PERSONAS = {
    "jarvis_classic": {
        "name": "Jarvis",
        "addressee": {"en": "", "ar": ""},
        "style": {
            "en": "calm, dry, useful",
            "ar": "\u0647\u0627\u062f\u064a\u060c \u062e\u0641\u064a\u0641\u060c \u0639\u0645\u0644\u064a",
        },
        "voice_length": {
            "en": "1-3 short sentences",
            "ar": "\u062c\u0645\u0644\u0629 \u0644\u0627\u062a\u0646\u064a\u0646 \u0642\u0635\u064a\u0631\u0629",
        },
        "signature_phrases": {
            "en": ["On it.", "Done.", "Simple enough."],
            "ar": ["\u062a\u0645\u0627\u0645.", "\u062d\u0627\u0636\u0631.", "\u0645\u0639\u0627\u0643."],
        },
        "forbidden": {
            "en": ["As an AI", "as a language model", "I'm just", "I cannot"],
            "ar": ["\u0628\u0635\u0641\u062a\u064a", "\u0643\u0630\u0643\u0627\u0621 \u0635\u0646\u0627\u0639\u064a", "\u064a\u0633\u0639\u062f\u0646\u064a"],
        },
    },
    "jarvis_warm": {
        "name": "Jarvis",
        "addressee": {"en": "", "ar": ""},
        "style": {
            "en": "warm, clear, practical",
            "ar": "\u062f\u0627\u0641\u064a\u060c \u062e\u0641\u064a\u0641\u060c \u0639\u0645\u0644\u064a",
        },
        "voice_length": {
            "en": "1-3 short sentences",
            "ar": "\u062c\u0645\u0644\u0629 \u0644\u0627\u062a\u0646\u064a\u0646 \u0642\u0635\u064a\u0631\u0629",
        },
        "signature_phrases": {
            "en": ["Absolutely.", "I'm with you.", "Easy."],
            "ar": ["\u0623\u0643\u064a\u062f.", "\u062d\u0627\u0636\u0631.", "\u0645\u0639\u0627\u0643."],
        },
        "forbidden": {
            "en": ["As an AI", "as a language model", "I'm just", "I cannot"],
            "ar": ["\u0628\u0635\u0641\u062a\u064a", "\u0643\u0630\u0643\u0627\u0621 \u0635\u0646\u0627\u0639\u064a", "\u064a\u0633\u0639\u062f\u0646\u064a"],
        },
    },
}


def _split_csv_tuple(values):
    return [str(item).strip() for item in values or () if str(item).strip()]


def _env_is_set(key: str) -> bool:
    return key in os.environ


def get_active_persona() -> dict:
    profile = PERSONA_PROFILE if PERSONA_PROFILE in {"jarvis_classic", "jarvis_warm", "custom"} else "jarvis_classic"
    base = DEFAULT_PERSONAS.get(profile) or DEFAULT_PERSONAS["jarvis_classic"]
    persona = {
        "profile": profile,
        "name": base.get("name") or "Jarvis",
        "addressee": dict(base.get("addressee") or {}),
        "style": dict(base.get("style") or {}),
        "voice_length": dict(base.get("voice_length") or {}),
        "signature_phrases": {
            "en": list((base.get("signature_phrases") or {}).get("en") or []),
            "ar": list((base.get("signature_phrases") or {}).get("ar") or []),
        },
        "forbidden": {
            "en": list((base.get("forbidden") or {}).get("en") or []),
            "ar": list((base.get("forbidden") or {}).get("ar") or []),
        },
    }

    # Env overrides apply to all profiles, including custom.
    if _env_is_set("JARVIS_PERSONA_NAME"):
        persona["name"] = PERSONA_NAME or persona["name"]
    if _env_is_set("JARVIS_PERSONA_ADDRESSEE_EN"):
        persona["addressee"]["en"] = PERSONA_ADDRESSEE_EN
    if _env_is_set("JARVIS_PERSONA_ADDRESSEE_AR"):
        persona["addressee"]["ar"] = PERSONA_ADDRESSEE_AR
    if _env_is_set("JARVIS_PERSONA_STYLE_EN"):
        persona["style"]["en"] = PERSONA_STYLE_EN or persona["style"].get("en") or "calm, dry, useful"
    if _env_is_set("JARVIS_PERSONA_STYLE_AR"):
        persona["style"]["ar"] = PERSONA_STYLE_AR or persona["style"].get("ar") or "\u0647\u0627\u062f\u064a\u060c \u062e\u0641\u064a\u0641\u060c \u0639\u0645\u0644\u064a"
    if _env_is_set("JARVIS_PERSONA_VOICE_LENGTH_EN"):
        persona["voice_length"]["en"] = PERSONA_VOICE_LENGTH_EN or persona["voice_length"].get("en") or "1-3 short sentences"
    if _env_is_set("JARVIS_PERSONA_VOICE_LENGTH_AR"):
        persona["voice_length"]["ar"] = PERSONA_VOICE_LENGTH_AR or persona["voice_length"].get("ar") or "\u062c\u0645\u0644\u0629 \u0644\u0627\u062a\u0646\u064a\u0646 \u0642\u0635\u064a\u0631\u0629"
    if _env_is_set("JARVIS_PERSONA_FORBIDDEN_EN") and PERSONA_FORBIDDEN_EN:
        persona["forbidden"]["en"] = _split_csv_tuple(PERSONA_FORBIDDEN_EN)
    if _env_is_set("JARVIS_PERSONA_FORBIDDEN_AR") and PERSONA_FORBIDDEN_AR:
        persona["forbidden"]["ar"] = _split_csv_tuple(PERSONA_FORBIDDEN_AR)
    return persona


def format_persona_block(persona: dict, language: str) -> str:
    lang = "ar" if str(language or "").strip().lower() == "ar" else "en"
    data = persona or get_active_persona()
    name = str(data.get("name") or "Jarvis").strip()
    addressee = str((data.get("addressee") or {}).get(lang) or "").strip()
    style = str((data.get("style") or {}).get(lang) or "").strip()
    length = str((data.get("voice_length") or {}).get(lang) or "").strip()
    forbidden = " | ".join(str(item).strip() for item in (data.get("forbidden") or {}).get(lang, []) if str(item).strip())
    if lang == "ar":
        addressee_part = f" \u0648\u0646\u0627\u062f\u064a \u0627\u0644\u0645\u0633\u062a\u062e\u062f\u0645 {addressee}" if addressee else ""
        return (
            f"\u0627\u0644\u0634\u062e\u0635\u064a\u0629: \u0627\u0633\u0645\u0643 {name}.{addressee_part} "
            f"\u0623\u0633\u0644\u0648\u0628\u0643 {style}. "
            f"\u0637\u0648\u0644 \u0627\u0644\u0631\u062f {length}. "
            f"\u0627\u0648\u0639\u0649 \u062a\u0642\u0648\u0644 {forbidden}."
        )
    addressee_part = f" addressee={addressee}." if addressee else ""
    return f"PERSONA: name={name}.{addressee_part} style={style}. length={length}. avoid={forbidden}."

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
            "explain_bridge": "وده عشان:",
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

    def status(self):
        with self._lock:
            active = self._active_profile

        return {
            "active_profile": active,
            "available_profiles": self.list_profiles(),
            "speech_style": PERSONA_PROFILES[active]["speech_style"],
            "speech_rate": int(PERSONA_PROFILES[active].get("speech_rate", 175)),
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
