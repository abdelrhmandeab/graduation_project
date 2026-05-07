"""Mixed Arabic/English command normalization helpers.

The parser uses this module as a lightweight pre-pass so commands like
"افتح Chrome" or "open الملفات" can be resolved before the regex cascade.
"""

from __future__ import annotations

import re

_ARABIC_RE = re.compile(r"[؀-ۿ]+")
_LATIN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_TOKEN_RE = re.compile(r"[؀-ۿ]+|[A-Za-z]+(?:'[A-Za-z]+)?|\d+%?|[^\s]")

# Arabic-Indic → ASCII for number extraction
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_ARABIC_VERB_MAP = {
    "افتح": "open",
    "افتحلي": "open",
    "شغل": "open",
    "شغّل": "open",
    "شغللي": "open",
    "اقفل": "close",
    "اغلق": "close",
    "سكر": "close",
    "سكّر": "close",
    "وقف": "stop",
    "ابحث": "search",
    "دور": "search",
    "دوّر": "search",
    "دورلي": "search",
    "احذف": "delete",
    # Volume / brightness control
    "ارفع": "increase",
    "زود": "increase",
    "رفع": "increase",
    "اخفض": "decrease",
    "قلل": "decrease",
    "خفض": "decrease",
    "وطي": "decrease",
    "وطّي": "decrease",
    # Set / assign
    "حط": "set",
    "اضبط": "set",
    "ظبط": "set",
    "اعمل": "set",
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
    "increase": "increase",
    "raise": "increase",
    "decrease": "decrease",
    "lower": "decrease",
    "set": "set",
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
    "صوت": "volume",
    "البحث": "search",
    "الويب": "web",
    "النت": "web",
    # Timer-related Arabic words
    "تايمر": "timer",
    "المؤقت": "timer",
    "مؤقت": "timer",
    # Duration unit words (used to detect timer context)
    "دقيقة": "minute",
    "دقايق": "minute",
    "دقائق": "minute",
    "ثانية": "second",
    "ثواني": "second",
    "ساعة": "hour",
    "ساعات": "hour",
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

# Preposition/filler tokens to skip when extracting entity
_STOPWORDS = {
    "and", "و", "then", "بعدها", "وبعدين",
    "for", "the", "a", "an", "of", "about", "to", "on", "in", "at",
    "عن", "على", "في", "من", "ل", "لي", "ال",
}


def convert_arabic_numerals(text: str) -> str:
    """Translate Arabic-Indic digits (٠١٢…٩) to ASCII digits (012…9)."""
    return str(text or "").translate(_AR_DIGITS)


def _script_counts(text: str) -> tuple[int, int]:
    arabic = len(_ARABIC_RE.findall(str(text or "")))
    latin = len(_LATIN_RE.findall(str(text or "")))
    return arabic, latin


def _normalize_token(token: str) -> str:
    return " ".join(str(token or "").lower().split()).strip()


def _extract_numbers(text: str) -> list[int | float]:
    """Extract all numeric values (Arabic-Indic or ASCII) from text."""
    normalized = convert_arabic_numerals(str(text or ""))
    results = []
    for m in re.finditer(r"\d+(?:\.\d+)?", normalized):
        val_str = m.group(0)
        try:
            val = float(val_str) if "." in val_str else int(val_str)
            results.append(val)
        except ValueError:
            pass
    return results


def _extract_latin_entity(tokens: list[str], skip_verb_token: bool = True) -> str:
    """Collect consecutive Latin-script tokens as one multi-word entity.

    Returns the longest run of Latin tokens that follows the verb, ignoring
    stopwords and prepositions. Falls back to the first single Latin token if
    no run is found.
    """
    runs: list[list[str]] = []
    current_run: list[str] = []

    for tok in tokens:
        normalized = _normalize_token(tok)
        if not normalized:
            continue
        if normalized in _ARABIC_VERB_MAP or normalized in _ENGLISH_VERB_MAP:
            if skip_verb_token:
                if current_run:
                    runs.append(current_run)
                    current_run = []
                continue
        if normalized in _STOPWORDS:
            if current_run:
                runs.append(current_run)
                current_run = []
            continue
        ar_c, lat_c = _script_counts(tok)
        if lat_c and not ar_c:
            current_run.append(tok.strip())
        else:
            if current_run:
                runs.append(current_run)
                current_run = []

    if current_run:
        runs.append(current_run)

    if not runs:
        return ""
    # Return the longest run (favors multi-word entities)
    best = max(runs, key=len)
    return " ".join(best)


def _extract_arabic_entity(tokens: list[str]) -> str:
    """Return first meaningful Arabic token that is not a verb or stopword."""
    for tok in tokens:
        normalized = _normalize_token(tok)
        if not normalized:
            continue
        if normalized in _ARABIC_VERB_MAP:
            continue
        if normalized in _STOPWORDS:
            continue
        ar_c, lat_c = _script_counts(tok)
        if ar_c and not lat_c:
            return tok.strip()
    return ""


def _map_arabic_entity(entity: str) -> str:
    normalized = " ".join(entity.split()).strip()
    return _ARABIC_ENTITY_MAP.get(normalized, normalized)


def normalize_codeswitched(text: str) -> dict:
    """Return a best-effort language/entity breakdown for mixed commands.

    Returns a dict with keys:
      original, arabic_segments, latin_segments,
      detected_verb, verb_intent, entity_text, language, numbers,
      intent, entity, normalized_entity, source_text
    """
    source = str(text or "").strip()
    empty = {
        "original": source,
        "arabic_segments": [],
        "latin_segments": [],
        "detected_verb": "",
        "verb_intent": "",
        "entity_text": "",
        "language": "",
        "numbers": [],
        "intent": "",
        "entity": "",
        "normalized_entity": "",
        "source_text": source,
    }
    if not source:
        return empty

    tokens = [tok for tok in _TOKEN_RE.findall(source) if tok.strip()]
    if not tokens:
        return empty

    arabic_segments = [tok for tok in tokens if _script_counts(tok)[0] and not _script_counts(tok)[1]]
    latin_segments = [tok for tok in tokens if _script_counts(tok)[1] and not _script_counts(tok)[0]]
    numbers = _extract_numbers(source)

    # --- detect verb (first matching token in first 3 positions) ---
    detected_verb = ""
    verb_intent = ""
    verb_idx = -1
    for i, tok in enumerate(tokens[:4]):
        norm = _normalize_token(tok)
        vi = _ARABIC_VERB_MAP.get(norm) or _ENGLISH_VERB_MAP.get(norm)
        if vi:
            detected_verb = tok.strip()
            verb_intent = vi
            verb_idx = i
            break

    # --- determine language ---
    ar_count = len(arabic_segments)
    lat_count = len(latin_segments)
    if ar_count and lat_count:
        language = "mixed"
    elif ar_count:
        language = "ar"
    elif lat_count:
        language = "en"
    else:
        language = ""

    # --- extract entity ---
    entity_text = ""
    rest_tokens = tokens[verb_idx + 1:] if verb_idx >= 0 else tokens

    if ar_count and lat_count:
        # Mixed: prefer multi-word Latin run after the verb
        entity_text = _extract_latin_entity(rest_tokens, skip_verb_token=False)
        if not entity_text:
            # Fallback to Arabic entity
            ar_ent = _extract_arabic_entity(rest_tokens)
            if ar_ent:
                entity_text = _map_arabic_entity(ar_ent)
    elif lat_count:
        entity_text = _extract_latin_entity(rest_tokens, skip_verb_token=False)
        if not entity_text and verb_idx < 0:
            entity_text = _extract_latin_entity(tokens, skip_verb_token=True)
    elif ar_count:
        ar_ent = _extract_arabic_entity(rest_tokens)
        if ar_ent:
            entity_text = _map_arabic_entity(ar_ent)

    entity = entity_text
    if not verb_intent and entity:
        mapped = _INTENT_FROM_ENTITY.get(_normalize_token(entity))
        if mapped:
            verb_intent = mapped

    result = {
        "original": source,
        "arabic_segments": arabic_segments,
        "latin_segments": latin_segments,
        "detected_verb": detected_verb,
        "verb_intent": verb_intent,
        "entity_text": entity_text,
        "language": language,
        "numbers": numbers,
        # Legacy compat keys used by _try_codeswitched_command
        "intent": verb_intent,
        "entity": entity,
        "normalized_entity": _normalize_token(entity),
        "source_text": source,
    }
    return result
