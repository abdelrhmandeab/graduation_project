"""Voice-Optimized Response Shaper — Task 2.2 / 4.2.

Two responsibilities:
  1. Action confirmation templates: for known OS intents, returns a short bilingual
     phrase instead of whatever the OS layer produced (which is always English).
     This means "افتح كروم" gets "بفتح كروم." rather than "Opening Google Chrome."

  2. LLM voice constraint: provides a prompt-suffix fragment that is injected into
     the LLM prompt before generation, constraining the model to 1-4 natural spoken
     sentences with no markdown or bullet lists.

The trimmer (`_trim_for_voice`) acts as a post-generation safety net: if the model
ignores the constraint suffix, the trimmer strips markdown and caps sentence count.

Arabic text is Egyptian colloquial (عامية مصرية) throughout — never MSA.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Action templates — (intent, action) → {language: text}
# Placeholders resolved by _render_template from the entities dict.
# Templates that contain {n} or other optional fields return "" when the
# field is missing; the shaper then falls through to the OS layer message.
# ---------------------------------------------------------------------------
ACTION_TEMPLATES: Dict[tuple, Dict[str, str]] = {

    # -----------------------------------------------------------------------
    # APP_CONTROL
    # -----------------------------------------------------------------------
    ("OS_APP_OPEN", "open"): {
        "en": "Opening {app_name}.",
        "ar": "بفتح {app_name}.",
    },
    ("OS_APP_OPEN", ""): {
        "en": "Opening {app_name}.",
        "ar": "بفتح {app_name}.",
    },
    ("OS_APP_OPEN", "not_found"): {
        "en": "I can't find {app_name}.",
        "ar": "مش لاقي {app_name}.",
    },
    ("OS_APP_CLOSE", "close"): {
        "en": "Closing {app_name}.",
        "ar": "بقفل {app_name}.",
    },
    ("OS_APP_CLOSE", ""): {
        "en": "Closing {app_name}.",
        "ar": "بقفل {app_name}.",
    },
    ("OS_APP_CLOSE", "not_found"): {
        "en": "I can't find {app_name}.",
        "ar": "مش لاقي {app_name}.",
    },

    # -----------------------------------------------------------------------
    # TIMER
    # -----------------------------------------------------------------------
    ("OS_TIMER", "set"): {
        "en": "Timer set for {duration}.",
        "ar": "التايمر اتظبط على {duration}.",
    },
    ("OS_TIMER", "set_alarm"): {
        "en": "Alarm set for {alarm_time}.",
        "ar": "المنبه اتظبط على {alarm_time}.",
    },
    ("OS_TIMER", "done"): {
        "en": "Time's up! {label}",
        "ar": "الوقت خلص! {label}",
    },
    ("OS_TIMER", "cancel"): {
        "en": "Timer cancelled.",
        "ar": "التايمر اتلغى.",
    },
    ("OS_TIMER", "list"): {
        "en": "Here are your active timers.",
        "ar": "دي التايمرات الشغالة.",
    },

    # -----------------------------------------------------------------------
    # VOLUME  (action_key variants from command_router dispatch)
    # -----------------------------------------------------------------------
    ("OS_SYSTEM_COMMAND", "volume_set"): {
        "en": "Volume at {n}%.",
        "ar": "الصوت بقى {n}%.",
    },
    # Legacy key kept for backward compat with older dispatch paths
    ("OS_SYSTEM_COMMAND", "set_volume"): {
        "en": "Volume at {n}%.",
        "ar": "الصوت بقى {n}%.",
    },
    ("OS_SYSTEM_COMMAND", "volume_up"): {
        "en": "Volume up to {n}%.",
        "ar": "الصوت زاد لـ {n}%.",
    },
    ("OS_SYSTEM_COMMAND", "volume_down"): {
        "en": "Volume down to {n}%.",
        "ar": "الصوت قل لـ {n}%.",
    },
    ("OS_SYSTEM_COMMAND", "mute"): {
        "en": "Muted.",
        "ar": "الصوت اتكتم.",
    },
    ("OS_SYSTEM_COMMAND", "volume_mute"): {
        "en": "Muted.",
        "ar": "الصوت اتكتم.",
    },
    ("OS_SYSTEM_COMMAND", "unmute"): {
        "en": "Unmuted at {n}%.",
        "ar": "الصوت رجع {n}%.",
    },
    ("OS_SYSTEM_COMMAND", "volume_unmute"): {
        "en": "Unmuted at {n}%.",
        "ar": "الصوت رجع {n}%.",
    },

    # -----------------------------------------------------------------------
    # BRIGHTNESS
    # -----------------------------------------------------------------------
    ("OS_SYSTEM_COMMAND", "brightness_set"): {
        "en": "Brightness at {n}%.",
        "ar": "السطوع بقى {n}%.",
    },
    # Legacy key
    ("OS_SYSTEM_COMMAND", "set_brightness"): {
        "en": "Brightness at {n}%.",
        "ar": "السطوع بقى {n}%.",
    },
    ("OS_SYSTEM_COMMAND", "brightness_up"): {
        "en": "Brighter, now {n}%.",
        "ar": "الشاشة نورت، دلوقتي {n}%.",
    },
    ("OS_SYSTEM_COMMAND", "brightness_down"): {
        "en": "Dimmer, now {n}%.",
        "ar": "الشاشة بهتت، دلوقتي {n}%.",
    },

    # -----------------------------------------------------------------------
    # SYSTEM
    # -----------------------------------------------------------------------
    ("OS_SYSTEM_COMMAND", "lock"): {
        "en": "Locking the computer.",
        "ar": "بقفل الجهاز.",
    },
    ("OS_SYSTEM_COMMAND", "sleep"): {
        "en": "Going to sleep.",
        "ar": "الجهاز هينام.",
    },
    ("OS_SYSTEM_COMMAND", "shutdown_ask"): {
        "en": "Are you sure? Tell me your PIN.",
        "ar": "متأكد؟ قولي الـ PIN.",
    },
    ("OS_SYSTEM_COMMAND", "shutdown"): {
        "en": "Shutting down.",
        "ar": "الجهاز هيقفل.",
    },
    ("OS_SYSTEM_COMMAND", "restart"): {
        "en": "Restarting.",
        "ar": "الجهاز هيعيد التشغيل.",
    },
    ("OS_SYSTEM_COMMAND", "screenshot"): {
        "en": "Screenshot saved.",
        "ar": "الصورة اتحفظت.",
    },
    ("OS_SYSTEM_COMMAND", "media_play"): {
        "en": "Playing.",
        "ar": "بشغل.",
    },
    ("OS_SYSTEM_COMMAND", "media_pause"): {
        "en": "Paused.",
        "ar": "اتوقف.",
    },
    ("OS_SYSTEM_COMMAND", "media_stop"): {
        "en": "Stopped.",
        "ar": "وقف.",
    },
    ("OS_SYSTEM_COMMAND", "media_next"): {
        "en": "Next track.",
        "ar": "الأغنية الجاية.",
    },
    ("OS_SYSTEM_COMMAND", "media_prev"): {
        "en": "Previous track.",
        "ar": "الأغنية اللي فاتت.",
    },
    ("OS_SYSTEM_COMMAND", "browser_search_web"): {
        "en": "Searching for {search_query}.",
        "ar": "بدور على {search_query}.",
    },

    # -----------------------------------------------------------------------
    # MEDIA (dedicated intent variants)
    # -----------------------------------------------------------------------
    ("OS_MEDIA_PLAY", ""): {
        "en": "Playing.",
        "ar": "بشغل.",
    },
    ("OS_MEDIA_CONTROL", "play"): {
        "en": "Playing.",
        "ar": "بشغل.",
    },
    ("OS_MEDIA_CONTROL", "pause"): {
        "en": "Paused.",
        "ar": "اتوقف.",
    },
    ("OS_MEDIA_CONTROL", "stop"): {
        "en": "Stopped.",
        "ar": "وقف.",
    },
    ("OS_MEDIA_CONTROL", "next"): {
        "en": "Next track.",
        "ar": "الأغنية الجاية.",
    },
    ("OS_MEDIA_CONTROL", "previous"): {
        "en": "Previous track.",
        "ar": "الأغنية اللي فاتت.",
    },
    ("OS_MEDIA_CONTROL", "prev"): {
        "en": "Previous track.",
        "ar": "الأغنية اللي فاتت.",
    },
    ("OS_MEDIA_CONTROL", ""): {
        "en": "Playing.",
        "ar": "بشغل.",
    },

    # -----------------------------------------------------------------------
    # FILE OPS
    # -----------------------------------------------------------------------
    ("OS_FILE_SEARCH", ""): {
        "en": "Found {count} file(s). Opening.",
        "ar": "لقيت {count} ملفات. بفتح.",
    },
    ("OS_FILE_SEARCH", "not_found"): {
        "en": "Couldn't find that.",
        "ar": "مش لاقي حاجة.",
    },

    # -----------------------------------------------------------------------
    # SETTINGS
    # -----------------------------------------------------------------------
    ("OS_SETTINGS", "open"): {
        "en": "Opening {setting} settings.",
        "ar": "بفتح إعدادات {setting}.",
    },
    ("OS_SETTINGS", ""): {
        "en": "Opening {setting} settings.",
        "ar": "بفتح إعدادات {setting}.",
    },

    # -----------------------------------------------------------------------
    # REMINDER
    # -----------------------------------------------------------------------
    ("OS_REMINDER", "create"): {
        "en": "I'll remind you at {time_str}.",
        "ar": "هفكرك الساعة {time_str}.",
    },
    ("OS_REMINDER", "cancel"): {
        "en": "Reminder cancelled.",
        "ar": "التذكير اتلغى.",
    },
    ("OS_REMINDER", "list"): {
        "en": "Here are your active reminders.",
        "ar": "دي التذكيرات الشغالة.",
    },

    # -----------------------------------------------------------------------
    # CALCULATOR (quick_calc bypass)
    # -----------------------------------------------------------------------
    ("QUICK_CALC", ""): {
        "en": "The answer is {result}.",
        "ar": "الإجابة {result}.",
    },

    # -----------------------------------------------------------------------
    # CLIPBOARD / SYSINFO
    # -----------------------------------------------------------------------
    ("OS_CLIPBOARD", "copy"): {
        "en": "Copied.",
        "ar": "اتنسخ.",
    },
    ("OS_CLIPBOARD", "paste"): {
        "en": "Pasted.",
        "ar": "اتلصق.",
    },
    ("OS_CLIPBOARD", "clear"): {
        "en": "Clipboard cleared.",
        "ar": "الكليب بورد اتمسح.",
    },

    # -----------------------------------------------------------------------
    # ROLLBACK
    # -----------------------------------------------------------------------
    ("OS_ROLLBACK", ""): {
        "en": "Last action undone.",
        "ar": "الأكشن اللي فات اتعكس.",
    },
}

# ---------------------------------------------------------------------------
# Dialogue templates — standalone short phrases for slot-filling, errors,
# and conversational turn-taking.  Keyed by a simple string name.
# ---------------------------------------------------------------------------
DIALOGUE_TEMPLATES: Dict[str, Dict[str, str]] = {
    # Slot-fill questions
    "slot_missing_app": {
        "en": "Which app?",
        "ar": "أي تطبيق؟",
    },
    "slot_missing_time": {
        "en": "At what time?",
        "ar": "الساعة كام؟",
    },
    "slot_missing_dur": {
        "en": "For how long?",
        "ar": "لمدة قد إيه؟",
    },
    "slot_missing_file": {
        "en": "Which file?",
        "ar": "أي ملف؟",
    },
    "slot_missing_query": {
        "en": "What should I search for?",
        "ar": "أدور على إيه؟",
    },
    # Confirmations
    "confirm_yes_no": {
        "en": "Are you sure?",
        "ar": "متأكد؟",
    },
    "confirm_pin": {
        "en": "Tell me your PIN.",
        "ar": "قولي الـ PIN.",
    },
    # Conversational
    "didnt_catch": {
        "en": "Sorry, I didn't catch that.",
        "ar": "معلش، مسمعتش كويس.",
    },
    "anything_else": {
        "en": "Anything else?",
        "ar": "حاجة تانية؟",
    },
    "how_can_i_help": {
        "en": "How can I help?",
        "ar": "أيه اللي تحتاجه؟",
    },
    # Errors
    "generic_error": {
        "en": "Something went wrong.",
        "ar": "حصل مشكلة.",
    },
    "timeout": {
        "en": "That took too long.",
        "ar": "ده أخد وقت كتير.",
    },
    "no_internet": {
        "en": "Can't reach the internet.",
        "ar": "مش قادر أوصل للنت.",
    },
    "not_supported": {
        "en": "I can't do that yet.",
        "ar": "مش قادر أعمل ده دلوقتي.",
    },
}

# ---------------------------------------------------------------------------
# LLM voice constraint suffixes
# Injected into the prompt before the ASSISTANT: marker.
# ---------------------------------------------------------------------------
VOICE_PROMPT_SUFFIX: Dict[str, str] = {
    "en": "RULE: Answer in 1-2 sentences. No lists, no markdown, no formatting. Speak naturally.",
    "ar": "RULE: جاوب في جملة أو اتنين بالمصري. من غير قوائم أو تنسيق.",
}

CONVERSATIONAL_PROMPT_SUFFIX: Dict[str, str] = {
    "en": "RULE: Keep your answer under 4 sentences. No lists or markdown formatting.",
    "ar": "RULE: خلي إجابتك أقل من ٤ جمل. من غير قوائم أو تنسيق.",
}

# Intents that benefit from a short factual-answer constraint (1-2 sentences)
_FACTUAL_INTENTS = frozenset({"LLM_QUERY"})

# Placeholders that don't abort rendering when missing — filled with "" instead
_OPTIONAL_PLACEHOLDERS = frozenset({"label"})

# Sentence-boundary pattern for voice trimming
_SENT_END_RE = re.compile(r"(?<=[.!?؟])\s+")

# Markdown patterns to strip
_MD_BOLD_RE = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_MD_HEADER_RE = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^\s*[-•*]\s+", re.MULTILINE)
_MD_NUMBERED_RE = re.compile(r"^\s*\d+\.\s+", re.MULTILINE)
_MD_CODE_RE = re.compile(r"`{1,3}[^`]*`{1,3}")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
_MD_HR_RE = re.compile(r"^\s*[-*_]{3,}\s*$", re.MULTILINE)


def _seconds_to_human(seconds: int, language: str = "en") -> str:
    """Convert a duration in seconds to a short human-readable string."""
    secs = max(1, int(seconds))
    if secs >= 3600:
        hrs, rem = divmod(secs, 3600)
        mins = rem // 60
        if language == "ar":
            if mins:
                return f"{hrs} ساعة و{mins} دقيقة"
            return f"{hrs} ساعة"
        return f"{hrs}h {mins}m" if mins else f"{hrs}h"
    if secs >= 60:
        mins, rem_s = divmod(secs, 60)
        if language == "ar":
            if rem_s:
                return f"{mins} دقيقة و{rem_s} ثانية"
            return f"{mins} دقيقة"
        return f"{mins}m {rem_s}s" if rem_s else f"{mins}m"
    if language == "ar":
        return f"{secs} ثانية"
    return f"{secs}s"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-like chunks on sentence-ending punctuation."""
    parts = _SENT_END_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


class ResponseShaper:
    """Bilingual voice-optimized response post-processor."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def shape(
        self,
        intent: str,
        action: str,
        entities: Dict[str, Any],
        language: str,
        llm_response: Optional[str] = None,
    ) -> str:
        """Return a voice-optimized response for the given intent.

        Priority:
          1. Action template (if intent + action has a template entry)
          2. LLM response trimmed for voice
          3. Raw LLM response unchanged
        """
        lang = str(language or "en").strip().lower()[:2]
        if lang not in ("en", "ar"):
            lang = "en"

        template_key = (str(intent or "").strip().upper(), str(action or "").strip().lower())
        template_map = ACTION_TEMPLATES.get(template_key)

        if template_map and lang in template_map:
            rendered = self._render_template(template_map[lang], entities, lang)
            if rendered:
                return rendered

        if llm_response:
            return self._trim_for_voice(str(llm_response), lang)

        return str(llm_response or "")

    def get_dialogue(self, key: str, language: str, **kwargs: Any) -> str:
        """Return a short bilingual dialogue phrase by key.

        Falls back to the English variant when Arabic is missing, then to key.
        """
        lang = str(language or "en").strip().lower()[:2]
        if lang not in ("en", "ar"):
            lang = "en"
        tpl_map = DIALOGUE_TEMPLATES.get(key) or {}
        template = tpl_map.get(lang) or tpl_map.get("en") or key
        try:
            return template.format(**kwargs) if kwargs else template
        except Exception:
            return template

    def should_use_template(self, intent: str, action: str) -> bool:
        """Return True if a template exists for this (intent, action) pair."""
        key = (str(intent or "").strip().upper(), str(action or "").strip().lower())
        return key in ACTION_TEMPLATES

    def get_prompt_suffix(
        self,
        intent: str,
        has_live_data: bool,
        language: str,
    ) -> str:
        """Return a voice constraint rule to inject into the LLM prompt.

        Returns an empty string for non-LLM intents (they don't reach the LLM).
        """
        if str(intent or "").strip().upper() != "LLM_QUERY":
            return ""
        lang = str(language or "en").strip().lower()[:2]
        if lang not in ("en", "ar"):
            lang = "en"
        if has_live_data:
            return VOICE_PROMPT_SUFFIX.get(lang, VOICE_PROMPT_SUFFIX["en"])
        return CONVERSATIONAL_PROMPT_SUFFIX.get(lang, CONVERSATIONAL_PROMPT_SUFFIX["en"])

    def inject_suffix_into_prompt(self, prompt: str, suffix: str) -> str:
        """Insert `suffix` on the line before the ASSISTANT: marker in `prompt`."""
        if not suffix:
            return prompt
        marker = "\nASSISTANT:"
        if marker in prompt:
            return prompt.replace(marker, f"\n{suffix}{marker}", 1)
        return prompt

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_template(
        self,
        template: str,
        entities: Dict[str, Any],
        language: str,
    ) -> str:
        """Fill in a template string from entities dict.

        Recognized placeholder → entity key mappings:
          {app_name}     → entities["app_name"]
          {duration}     → entities["seconds"] / entities["duration_seconds"]
          {alarm_time}   → entities["alarm_time"]
          {n}            → entities["volume_level"] / entities["brightness_level"]
                           / entities["level"] / entities["value"] / entities["n"]
          {count}        → entities["count"] / entities["n_results"]
          {search_query} → entities["search_query"] / entities["query"]
          {setting}      → entities["setting"] / entities["setting_name"]
                           / entities["category"]
          {time_str}     → entities["time_str"] / entities["trigger_time"]
          {result}       → entities["result"]
          {label}        → entities["label"]  (optional — defaults to "")

        Returns empty string if a required placeholder cannot be filled, so the
        caller can fall through to the raw OS message.
        """
        try:
            placeholders = set(re.findall(r"\{(\w+)\}", template))
            values: Dict[str, str] = {}

            for ph in placeholders:
                val = self._resolve_placeholder(ph, entities, language)
                if val is None:
                    if ph in _OPTIONAL_PLACEHOLDERS:
                        values[ph] = ""
                    else:
                        return ""
                else:
                    values[ph] = val

            result = template.format(**values)
            # Collapse double spaces that can appear when optional labels are ""
            return re.sub(r"  +", " ", result).strip()
        except Exception:
            return ""

    def _resolve_placeholder(
        self,
        ph: str,
        entities: Dict[str, Any],
        language: str,
    ) -> Optional[str]:
        """Return the string value for a single placeholder, or None if unresolvable."""
        if ph == "duration":
            secs = entities.get("seconds") or entities.get("duration_seconds")
            if secs is None:
                return None
            return _seconds_to_human(int(secs), language)

        if ph == "n":
            raw = (
                entities.get("volume_level")
                or entities.get("brightness_level")
                or entities.get("level")
                or entities.get("value")
                or entities.get("n")
            )
            if raw is None:
                return None
            return str(raw).strip()

        if ph == "count":
            c = entities.get("count") or entities.get("n_results")
            if c is None:
                return None
            return str(c)

        if ph == "alarm_time":
            t = entities.get("alarm_time")
            if not t:
                return None
            return str(t).strip()

        if ph == "search_query":
            q = entities.get("search_query") or entities.get("query")
            if not q:
                return None
            return str(q).strip()

        if ph == "setting":
            s = entities.get("setting") or entities.get("setting_name") or entities.get("category")
            if not s:
                return None
            return str(s).strip()

        if ph == "time_str":
            t = entities.get("time_str") or entities.get("trigger_time")
            if not t:
                return None
            return str(t).strip()

        if ph == "result":
            r = entities.get("result")
            if r is None:
                return None
            return str(r).strip()

        if ph == "label":
            lbl = entities.get("label") or ""
            return str(lbl).strip()

        # Generic: look up by placeholder name directly in entities
        if ph in entities:
            v = entities[ph]
            if not v and v != 0:
                return None
            return str(v).strip()

        return None

    def _trim_for_voice(self, text: str, language: str, max_sentences: int = 4) -> str:
        """Strip markdown formatting and cap to `max_sentences` sentences."""
        if not text:
            return text

        # Strip markdown
        text = _MD_BOLD_RE.sub(r"\1", text)
        text = _MD_HEADER_RE.sub("", text)
        text = _MD_BULLET_RE.sub("", text)
        text = _MD_NUMBERED_RE.sub("", text)
        text = _MD_CODE_RE.sub("", text)
        text = _MD_LINK_RE.sub(r"\1", text)
        text = _MD_HR_RE.sub("", text)

        # Collapse whitespace while preserving sentence structure
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r" {2,}", " ", text).strip()

        # Cap sentence count
        sentences = _split_sentences(text)
        if len(sentences) > max_sentences:
            text = " ".join(sentences[:max_sentences])
            if not text.endswith((".", "!", "?", "؟")):
                text += "."

        return text


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
response_shaper = ResponseShaper()


def get_dialogue(key: str, language: str = "en", **kwargs: Any) -> str:
    """Module-level shortcut for ResponseShaper.get_dialogue()."""
    return response_shaper.get_dialogue(key, language, **kwargs)
