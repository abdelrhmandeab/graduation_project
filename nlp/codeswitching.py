"""Mixed Arabic/English command normalization helpers.

The parser uses this module as a lightweight pre-pass so commands like
"افتح Chrome" or "open الملفات" can be resolved before the regex cascade.
"""

from __future__ import annotations

import re

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]+")
_LATIN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_TOKEN_RE = re.compile(r"[\u0600-\u06FF]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+%?|[^\s]")

_ARABIC_VERB_MAP = {
    "افتح": "open",
    "افتحلي": "open",
    "شغل": "open",
    "شغّل": "open",
    "شغللي": "open",
    "اقفل": "close",
    "سكر": "close",
    "سكّر": "close",
    "وقف": "stop",
    "ابحث": "search",
    "دور": "search",
    "دوّر": "search",
    "احذف": "delete",
}

_ENGLISH_VERB_MAP = {
    "open": "open",
    "launch": "open",
    "start": "open",
    "play": "open",
    "close": "close",
    "quit": "close",
    "stop": "stop",
    "search": "search",
    "find": "search",
}

_ARABIC_ENTITY_MAP = {
    "المتصفح": "browser",
    "المفكرة": "notepad",
    "الملفات": "files",
    "ملفات": "files",
    "المجلد": "folder",
    "مجلد": "folder",
    "الموسيقى": "music",
    "المزيكا": "music",
    "الكروم": "chrome",
    "جوجل كروم": "chrome",
    "السطوع": "brightness",
    "الصوت": "volume",
    "الفوليم": "volume",
    "البحث": "search",
    "الويب": "web",
    "النت": "web",
}

_INTENT_FROM_ENTITY = {
    "browser": "open",
    "chrome": "open",
    "edge": "open",
    "firefox": "open",
    "spotify": "open",
    "vlc": "open",
    "notepad": "open",
    "calculator": "open",
    "files": "search",
    "folder": "open",
    "music": "open",
    "volume": "open",
    "brightness": "open",
    "web": "search",
}


def _script_counts(text: str) -> tuple[int, int]:
    arabic = len(_ARABIC_RE.findall(str(text or "")))
    latin = len(_LATIN_RE.findall(str(text or "")))
    return arabic, latin


def _normalize_token(token: str) -> str:
    return " ".join(str(token or "").lower().split()).strip()


def _extract_first_meaningful_entity(tokens: list[str]) -> str:
    for token in tokens:
        normalized = _normalize_token(token)
        if not normalized:
            continue
        if normalized in _ARABIC_VERB_MAP or normalized in _ENGLISH_VERB_MAP:
            continue
        if normalized in {"and", "و", "then", "بعدها", "وبعدين", "for", "the", "a", "an", "of", "about", "to", "on", "in", "عن", "على", "في", "من", "ل", "لي", "ال"}:
            continue
        return token.strip()
    return ""


def _map_arabic_entity(entity: str) -> str:
    normalized = " ".join(entity.split()).strip()
    return _ARABIC_ENTITY_MAP.get(normalized, normalized)


def normalize_codeswitched(text: str):
    """Return a best-effort language/entity breakdown for mixed commands.

    Returns:
        (intent_lang, entity_lang, entities)
    """
    source = str(text or "").strip()
    if not source:
        return "", "", {}

    tokens = [token for token in _TOKEN_RE.findall(source) if token.strip()]
    if not tokens:
        return "", "", {}

    first_token = _normalize_token(tokens[0])
    intent = _ARABIC_VERB_MAP.get(first_token) or _ENGLISH_VERB_MAP.get(first_token)
    if not intent:
        for token in tokens[:3]:
            normalized = _normalize_token(token)
            intent = _ARABIC_VERB_MAP.get(normalized) or _ENGLISH_VERB_MAP.get(normalized)
            if intent:
                break

    entity = _extract_first_meaningful_entity(tokens[1:] if intent else tokens)
    if not entity and len(tokens) > 1:
        entity = _extract_first_meaningful_entity(tokens)

    entity_lang = ""
    if entity:
        arabic_count, latin_count = _script_counts(entity)
        if arabic_count and latin_count:
            entity_lang = "mixed"
        elif arabic_count:
            entity_lang = "ar"
            entity = _map_arabic_entity(entity)
        elif latin_count:
            entity_lang = "en"

    intent_lang = ""
    if intent:
        arabic_verb = any(_normalize_token(token) in _ARABIC_VERB_MAP for token in tokens[:2])
        english_verb = any(_normalize_token(token) in _ENGLISH_VERB_MAP for token in tokens[:2])
        if arabic_verb and english_verb:
            intent_lang = "mixed"
        elif arabic_verb:
            intent_lang = "ar"
        elif english_verb:
            intent_lang = "en"

    entities = {
        "intent": intent or "",
        "entity": entity or "",
        "normalized_entity": _normalize_token(entity),
        "source_text": source,
    }
    if entity and not intent:
        mapped_intent = _INTENT_FROM_ENTITY.get(_normalize_token(entity))
        if mapped_intent:
            entities["intent"] = mapped_intent

    return intent_lang, entity_lang, entities
