"""NLU — entity extraction and slot validation layer.

Sits between intent classification and command dispatch.  Given a resolved
intent and the raw utterance, it:
  1. Maps the intent to a semantic domain (APP_CONTROL, TIMER, …)
  2. Extracts entities missing from the regex parser's output
  3. Returns the first unfilled required slot so the orchestrator can ask a
     targeted follow-up question rather than failing silently.

Design constraints:
  - No heavy ML deps — pure regex + catalog lookup, <1 ms per call.
  - Enriches *existing* args rather than replacing them; the regex parser has
    higher precision for cases it already handles.
  - No circular imports: imports only from os_control / nlp / core.logger.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.logger import logger

# ---------------------------------------------------------------------------
# Domain constants
# ---------------------------------------------------------------------------
APP_CONTROL = "APP_CONTROL"
FILE_OPS = "FILE_OPS"
SYSTEM = "SYSTEM"
MEDIA = "MEDIA"
TIMER = "TIMER"
REMINDER = "REMINDER"
SEARCH = "SEARCH"
CHAT = "CHAT"
EMAIL = "EMAIL"
CALENDAR = "CALENDAR"

# ---------------------------------------------------------------------------
# Intent → domain
# ---------------------------------------------------------------------------
_INTENT_TO_DOMAIN: Dict[str, str] = {
    "OS_APP_OPEN": APP_CONTROL,
    "OS_APP_CLOSE": APP_CONTROL,
    "OS_FILE_SEARCH": FILE_OPS,
    "OS_FILE_NAVIGATION": FILE_OPS,
    "OS_SYSTEM_COMMAND": SYSTEM,
    "OS_TIMER": TIMER,
    "OS_MEDIA_PLAY": MEDIA,
    "OS_MEDIA_CONTROL": MEDIA,
    "OS_EMAIL": EMAIL,
    "OS_CALENDAR": CALENDAR,
    "OS_REMINDER": REMINDER,
    "OS_CLIPBOARD": SYSTEM,
    "OS_SYSINFO": SYSTEM,
    "OS_SETTINGS": SYSTEM,
    "VOICE_COMMAND": SYSTEM,
    "LLM_QUERY": CHAT,
}

# ---------------------------------------------------------------------------
# Required slots per intent (checked after extraction)
# Uses the same key names that _dispatch looks up in parsed.args.
# ---------------------------------------------------------------------------
REQUIRED_SLOTS: Dict[str, List[str]] = {
    "OS_APP_OPEN": ["app_name"],
    "OS_APP_CLOSE": ["app_name"],
    "OS_TIMER": ["seconds"],
    "OS_FILE_SEARCH": ["filename"],
    # OS_REMINDER create-action slots are validated by the dispatcher (reminder_ops)
    # because list/cancel actions share the same intent but need no slots.
}

# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------
_DURATION_UNIT_SECONDS: Dict[str, int] = {
    "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1,
    "ثانية": 1, "ثواني": 1,
    "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
    "دقيقة": 60, "دقائق": 60, "دقايق": 60,
    "h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
    "ساعة": 3600, "ساعات": 3600,
}

# Arabic-Indic → ASCII digits  (٠١٢٣٤٥٦٧٨٩ → 0-9)
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Arabic written numbers
_AR_NUMBERS: Dict[str, int] = {
    "صفر": 0, "واحد": 1, "اثنين": 2, "اتنين": 2, "ثلاثة": 3, "تلاتة": 3,
    "اربعة": 4, "اربعه": 4, "خمسة": 5, "خمسه": 5, "ستة": 6, "سته": 6,
    "سبعة": 7, "سبعه": 7, "ثمانية": 8, "تمانية": 8, "تمانيه": 8,
    "تسعة": 9, "تسعه": 9, "عشرة": 10, "عشره": 10, "عشرين": 20,
    "تلاتين": 30, "ثلاثين": 30, "اربعين": 40, "خمسين": 50, "ستين": 60,
    "سبعين": 70, "تمانين": 80, "تسعين": 90, "خمستاشر": 15, "عشراشر": 10,
}

# Egyptian Arabic fractions: "نص" = half, "ربع" = quarter
_AR_FRACTIONS: Dict[str, float] = {"نص": 0.5, "ربع": 0.25}

# Arabic time-unit pattern for re.search
_AR_TIME_UNITS = r"ثانية|ثواني|دقيقة|دقائق|دقايق|ساعة|ساعات"
_EN_TIME_UNITS = r"seconds?|secs?|minutes?|mins?|hours?|hrs?"
_ALL_TIME_UNITS = f"(?:{_EN_TIME_UNITS}|{_AR_TIME_UNITS})"

# Duration regex: "5 minutes", "٥ دقايق", "3.5 hours"
_DURATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(" + _ALL_TIME_UNITS + r")",
    re.IGNORECASE,
)

# Arabic verb prefixes to strip before entity extraction
_AR_ACTION_VERBS = frozenset({
    "افتح", "افتحلي", "ممكن تفتح", "ممكن تشغل",
    "شغل", "شغّل", "شغللي",
    "اقفل", "اغلق", "سكر", "سكّر", "قفل", "اوقف", "وقف",
    "ابحث عن", "دور على", "دوّر على", "ابحث", "دور", "دوّر",
    "حط تايمر", "حطلي تايمر", "اعمل تايمر", "ظبط تايمر",
    "حط", "حطلي", "ظبط", "اعمل", "اضبط", "صحيني", "نبهني", "فكرني",
})

# English verb prefixes to strip
_EN_ACTION_VERBS = frozenset({
    "open", "launch", "start", "run", "close", "quit", "exit", "kill",
    "find", "search for", "look for", "where is", "set a timer", "set timer",
    "set an alarm", "set alarm",
})

# Filler words to strip after the verb
_FILLER_TOKENS = frozenset({
    "the", "a", "an", "app", "application", "program",
    "التطبيق", "البرنامج", "الاب",
})

# Filename prefix patterns to strip
_FILENAME_PREFIX_RE = re.compile(
    r"^(?:find|search\s+for|look\s+for|ابحث\s+عن|دور\s+على|دوّر\s+على)\s+",
    re.IGNORECASE,
)
_FILENAME_FILLER_RE = re.compile(
    r"^(?:the\s+)?(?:file|folder|document|doc|ملف|مجلد)\s+",
    re.IGNORECASE,
)
_FILENAME_LOCATION_SUFFIX_RE = re.compile(
    r"\s+(?:in|inside|on|from|في)\s+\S+$",
    re.IGNORECASE,
)

# Latin token pattern (app names are mostly Latin)
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_\.]*")
_LATIN_STOP = frozenset({
    "the", "a", "an", "to", "for", "and", "or", "in", "on", "at", "of",
    "open", "launch", "start", "close", "quit", "exit", "kill", "run",
    "find", "search", "look", "where", "is", "set", "timer", "alarm",
})

# Arabic location hint
_AR_LOCATION_RE = re.compile(r"في\s+([؀-ۿ\w]+)")


# ---------------------------------------------------------------------------
# App alias lookup cache (populated lazily from os_control.app_ops)
# ---------------------------------------------------------------------------
_APP_ALIAS_CACHE: Optional[List[tuple]] = None


def _get_app_alias_pairs() -> List[tuple]:
    global _APP_ALIAS_CACHE
    if _APP_ALIAS_CACHE is not None:
        return _APP_ALIAS_CACHE
    try:
        from os_control.app_ops import _APP_CATALOG
        pairs: List[tuple] = []
        for entry in _APP_CATALOG.values():
            canonical = str(entry.get("canonical_name") or "").strip()
            if not canonical:
                continue
            for alias in entry.get("aliases", []):
                a = str(alias or "").strip().lower()
                if a:
                    pairs.append((a, canonical))
            pairs.append((canonical.lower(), canonical))
        pairs.sort(key=lambda p: -len(p[0]))
        _APP_ALIAS_CACHE = pairs
    except Exception:
        _APP_ALIAS_CACHE = []
    return _APP_ALIAS_CACHE


# ---------------------------------------------------------------------------
# NLUResult
# ---------------------------------------------------------------------------

@dataclass
class NLUResult:
    domain: str
    intent: str
    entities: Dict[str, Any] = field(default_factory=dict)
    missing_slots: List[str] = field(default_factory=list)
    confidence: float = 0.0
    language: str = ""


# ---------------------------------------------------------------------------
# NLU class
# ---------------------------------------------------------------------------

class NLU:
    """Entity extraction and slot validation for known intents."""

    def understand(
        self,
        text: str,
        language: Optional[str] = None,
        *,
        intent: str = "",
        existing_args: Optional[Dict[str, Any]] = None,
    ) -> NLUResult:
        """Return NLUResult for the given utterance and resolved intent.

        Only enriches slots that are absent in `existing_args`.
        """
        intent_upper = str(intent or "").strip().upper()
        domain = _INTENT_TO_DOMAIN.get(intent_upper, CHAT)
        lang = str(language or "").strip().lower()

        existing = dict(existing_args or {})
        entities = self._extract_entities(text, domain, intent_upper, existing)

        required = list(REQUIRED_SLOTS.get(intent_upper, []))
        missing = [s for s in required if not _truthy(entities.get(s))]

        confidence = 1.0 if not missing else 0.5
        return NLUResult(
            domain=domain,
            intent=intent_upper,
            entities=entities,
            missing_slots=missing,
            confidence=confidence,
            language=lang,
        )

    # ------------------------------------------------------------------
    # Extraction dispatcher
    # ------------------------------------------------------------------

    def _extract_entities(
        self,
        text: str,
        domain: str,
        intent: str,
        existing: Dict[str, Any],
    ) -> Dict[str, Any]:
        result = dict(existing)

        # Codeswitching pre-pass: enrich app_name from mixed-language entity
        if domain == APP_CONTROL and not _truthy(result.get("app_name")) and not _truthy(result.get("app_query")):
            try:
                from nlp.codeswitching import normalize_codeswitched
                cs = normalize_codeswitched(text)
                cs_entity = str((cs or {}).get("entity_text") or "").strip()
                if cs_entity and cs_entity not in {"volume", "brightness", "timer", "files", "folder", "web"}:
                    result["_cs_entity_hint"] = cs_entity
            except Exception:
                pass

        if domain == APP_CONTROL:
            if not _truthy(result.get("app_name")) and not _truthy(result.get("app_query")):
                app = self._extract_app_name(text)
                if not app:
                    app = result.pop("_cs_entity_hint", "") or ""
                else:
                    result.pop("_cs_entity_hint", None)
                if app:
                    result["app_name"] = app
            else:
                result.pop("_cs_entity_hint", None)

        elif domain == TIMER:
            if not _truthy(result.get("seconds")):
                secs = self._extract_duration(text)
                if secs is not None:
                    result["seconds"] = secs
            if not _truthy(result.get("label")):
                label = self._extract_timer_label(text)
                if label:
                    result["label"] = label

        elif domain == FILE_OPS:
            if not _truthy(result.get("filename")):
                fname = self._extract_filename(text)
                if fname:
                    result["filename"] = fname
            if not _truthy(result.get("search_path")):
                loc = self._extract_location(text)
                if loc:
                    result["search_path"] = loc

        elif domain == REMINDER:
            if not _truthy(result.get("time_str")):
                ts = self._extract_time_str(text)
                if ts:
                    result["time_str"] = ts
            if not _truthy(result.get("message")):
                msg = self._extract_reminder_message(text, result.get("time_str", ""))
                if msg:
                    result["message"] = msg

        return result

    # ------------------------------------------------------------------
    # App name
    # ------------------------------------------------------------------

    def _extract_app_name(self, text: str) -> str:
        """Extract app name from mixed-language utterance.

        1. Strip AR/EN action verb prefix
        2. Strip filler tokens (the / app / …)
        3. Try catalog lookup (longest alias match)
        4. Fall back to first meaningful Latin token
        5. Fall back to Arabic entity map (codeswitching module)
        """
        normalized = " ".join(str(text or "").split())

        # Strip AR verbs (longest first to handle multi-word verbs)
        for verb in sorted(_AR_ACTION_VERBS, key=len, reverse=True):
            if normalized.lower().startswith(verb.lower() + " "):
                normalized = normalized[len(verb):].strip()
                break

        # Strip EN verbs
        normalized_lower = normalized.lower()
        for verb in sorted(_EN_ACTION_VERBS, key=len, reverse=True):
            if normalized_lower.startswith(verb + " "):
                normalized = normalized[len(verb):].strip()
                normalized_lower = normalized.lower()
                break

        # Strip leading filler tokens
        parts = normalized.split()
        while parts and parts[0].lower() in _FILLER_TOKENS:
            parts = parts[1:]
        normalized = " ".join(parts)

        if not normalized:
            return ""

        # Catalog lookup (covers Arabic aliases like كروم)
        padded = " " + normalized.lower() + " "
        for alias, canonical in _get_app_alias_pairs():
            if (" " + alias + " ") in padded:
                return canonical

        # Latin token fallback
        latin = [t for t in _LATIN_TOKEN_RE.findall(normalized) if t.lower() not in _LATIN_STOP]
        if latin:
            return " ".join(latin)

        # Arabic entity map (codeswitching module)
        try:
            from nlp.codeswitching import _ARABIC_ENTITY_MAP
            for ar_word, en_name in sorted(_ARABIC_ENTITY_MAP.items(), key=lambda p: -len(p[0])):
                if ar_word in normalized:
                    return en_name
        except Exception:
            pass

        return ""

    # ------------------------------------------------------------------
    # Duration
    # ------------------------------------------------------------------

    def _extract_duration(self, text: str) -> Optional[int]:
        """Return duration in seconds or None.

        Handles:
          - "5 minutes", "3.5 hours"
          - "٥ دقايق" (Arabic-Indic digits)
          - "نص ساعة" / "ربع ساعة" (Egyptian Arabic fractions)
          - "half an hour" / "half a minute"
          - "خمسة دقايق" (Arabic word numbers)
        """
        normalized = str(text or "").translate(_ARABIC_INDIC)

        # Standard "N unit" pattern
        m = _DURATION_RE.search(normalized)
        if m:
            number = float(m.group(1))
            unit = m.group(2).lower().strip()
            factor = _DURATION_UNIT_SECONDS.get(unit, 0)
            if factor:
                return max(1, min(86400, int(number * factor)))

        # Arabic fractions: "نص ساعة", "ربع دقيقة"
        for ar_frac, frac_val in _AR_FRACTIONS.items():
            fm = re.search(ar_frac + r"\s+(" + _ALL_TIME_UNITS + r")", normalized)
            if fm:
                factor = _DURATION_UNIT_SECONDS.get(fm.group(1).lower(), 0)
                if factor:
                    return max(1, int(frac_val * factor))

        # English fractions: "half an hour"
        half_m = re.search(r"\bhalf\s+(?:an?\s+)?(hour|minute|second)", normalized, re.IGNORECASE)
        if half_m:
            factor = _DURATION_UNIT_SECONDS.get(half_m.group(1).lower(), 0)
            if factor:
                return max(1, int(0.5 * factor))

        # Arabic word numbers: "خمسة دقايق"
        for ar_word, val in sorted(_AR_NUMBERS.items(), key=lambda p: -len(p[0])):
            wm = re.search(re.escape(ar_word) + r"\s+(" + _ALL_TIME_UNITS + r")", normalized)
            if wm:
                factor = _DURATION_UNIT_SECONDS.get(wm.group(1).lower(), 0)
                if factor:
                    return max(1, min(86400, int(val * factor)))

        return None

    def _extract_timer_label(self, text: str) -> Optional[str]:
        """Extract optional 'for X' label that isn't a duration phrase."""
        m = re.search(
            r"\bfor\s+(?:the\s+|a\s+)?([\w\s]+?)(?:\s+timer|\s+alarm|$)",
            text,
            re.IGNORECASE,
        )
        if m:
            label = m.group(1).strip()
            # Discard labels that are just duration phrases ("5 minutes")
            if label and not re.match(r"^\d", label) and not re.search(_ALL_TIME_UNITS, label, re.IGNORECASE):
                return label
        return None

    # ------------------------------------------------------------------
    # Reminder time string + message
    # ------------------------------------------------------------------

    # Patterns that match the time portion of a reminder utterance
    _TIME_PATTERNS = [
        # English relative: "in 2 hours", "in 30 minutes"
        re.compile(r"\bin\s+\d+(?:\.\d+)?\s+(?:hours?|hrs?|minutes?|mins?|seconds?|secs?)\b", re.IGNORECASE),
        # English wall-clock: "at 3pm", "at 3:30 pm", "at 15:00"
        re.compile(r"\bat\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", re.IGNORECASE),
        # English with tomorrow
        re.compile(r"\btomorrow\s+at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b", re.IGNORECASE),
        # Arabic relative: "بعد ساعتين", "بعد ٣٠ دقيقة", "بعد نص ساعة"
        re.compile(
            r"بعد\s+(?:\d+(?:\.\d+)?|[٠-٩]+(?:\.[٠-٩]+)?|نص|ربع|ساعتين)\s*"
            r"(?:ثانية|ثواني|دقيقة|دقائق|دقايق|ساعة|ساعات|ساعه)?",
            re.IGNORECASE,
        ),
        # Arabic wall-clock: "الساعة ٣", "الساعة ٣ مساءً", "الساعه ٩ صبح"
        re.compile(
            r"(?:الساعة|الساعه|ساعه?)\s+[\d٠-٩]+(?:[:.،,][\d٠-٩]+)?\s*"
            r"(?:صباحاً|صباحا|صبح|ص|مساءً|مساءا|مساء|م)?",
            re.IGNORECASE,
        ),
        # Arabic with tomorrow: "بكرة الساعة ٩"
        re.compile(
            r"(?:بكرة|بكره|بكرا)\s+(?:الساعة|الساعه|ساعه?)\s+[\d٠-٩]+(?:[:.،,][\d٠-٩]+)?",
            re.IGNORECASE,
        ),
    ]

    def _extract_time_str(self, text: str) -> str:
        """Extract raw time string from a reminder utterance."""
        for pattern in self._TIME_PATTERNS:
            m = pattern.search(text)
            if m:
                return m.group(0).strip()
        return ""

    def _extract_reminder_message(self, text: str, time_str: str) -> str:
        """Extract reminder message by stripping trigger verb and time portion."""
        normalized = " ".join(str(text or "").split())

        # Strip English trigger verbs
        normalized = re.sub(
            r"^(?:remind(?:\s+me)?|set\s+(?:a\s+)?reminder(?:\s+to)?)\s+",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip()
        # Strip Arabic trigger verbs
        normalized = re.sub(
            r"^(?:فكرني|فكّرني|ذكرني|ذكّرني|نبهني|نبّهني)\s+",
            "",
            normalized,
            flags=re.IGNORECASE,
        ).strip()
        # Strip the extracted time portion (if present)
        if time_str:
            normalized = normalized.replace(time_str, "").strip()
        # Strip leading "to " connector
        normalized = re.sub(r"^to\s+", "", normalized, flags=re.IGNORECASE).strip()
        # Strip leading "at/in/by/tomorrow" time words if time was extracted
        if time_str:
            normalized = re.sub(
                r"^(?:at|in|by|tomorrow)\s+\S+(?:\s+(?:am|pm))?\s*",
                "",
                normalized,
                flags=re.IGNORECASE,
            ).strip()
        return normalized

    # ------------------------------------------------------------------
    # Filename / location
    # ------------------------------------------------------------------

    def _extract_filename(self, text: str) -> str:
        """Extract search target from file search utterance."""
        normalized = " ".join(str(text or "").split())
        normalized = _FILENAME_PREFIX_RE.sub("", normalized).strip()
        normalized = _FILENAME_FILLER_RE.sub("", normalized).strip()
        normalized = _FILENAME_LOCATION_SUFFIX_RE.sub("", normalized).strip()
        return normalized

    def _extract_location(self, text: str) -> str:
        """Extract directory hint from file operation utterance."""
        m = re.search(
            r"\b(?:in|inside|on|from)\s+(?:the\s+)?([\w\s]+?)(?:\s+folder|\s+directory|$)",
            text,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        ar_m = _AR_LOCATION_RE.search(text)
        if ar_m:
            return ar_m.group(1).strip()
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truthy(value: Any) -> bool:
    """Return True if `value` is a non-empty, non-None, non-zero value."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


# ---------------------------------------------------------------------------
# Module-level singleton + convenience wrapper
# ---------------------------------------------------------------------------

_nlu = NLU()


def understand(
    text: str,
    language: Optional[str] = None,
    *,
    intent: str = "",
    existing_args: Optional[Dict[str, Any]] = None,
) -> NLUResult:
    """Convenience wrapper around the module-level NLU singleton."""
    return _nlu.understand(text, language, intent=intent, existing_args=existing_args)
