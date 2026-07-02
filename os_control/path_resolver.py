"""Path resolver — canonical location aliases + path humanizer.

Single source of truth for:
- KNOWN_FOLDERS: shell special-folder paths (Desktop, Downloads, …)
- DRIVE_ALIASES: spoken drive/partition phrases → root path
- resolve_location(spoken) → Path | None
- humanize_path(path) → {"en": str, "ar": str}

All other modules that need folder aliases or friendly path display should
import from here instead of defining their own tables.
"""
from __future__ import annotations

import ctypes
import os
import re
import string
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Known shell folders — resolved once at import time
# ---------------------------------------------------------------------------

def _sh_known_folder(guid: str) -> Optional[Path]:
    """Return the Windows shell folder for a KNOWNFOLDERID GUID string, or None."""
    if os.name != "nt":
        return None
    try:
        from ctypes import wintypes
        _SHGetKnownFolderPath = ctypes.windll.shell32.SHGetKnownFolderPath
        _SHGetKnownFolderPath.argtypes = [
            ctypes.c_char_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_wchar_p),
        ]
        _SHGetKnownFolderPath.restype = ctypes.HRESULT
        # Parse GUID string into bytes
        import uuid
        guid_bytes = uuid.UUID(guid).bytes_le
        buf = ctypes.c_wchar_p()
        hr = _SHGetKnownFolderPath(guid_bytes, 0, None, ctypes.byref(buf))
        if hr == 0 and buf.value:
            return Path(buf.value)
    except Exception:
        pass
    return None


def _expand(name: str) -> Path:
    return Path(os.path.expanduser(f"~/{name}"))


# Windows KNOWNFOLDERIDs for common shell folders.
_FOLDER_GUIDS = {
    "Desktop":   "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
    "Downloads": "{374DE290-123F-4565-9164-39C4925E467B}",
    "Documents": "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}",
    "Pictures":  "{33E28130-4E1E-4676-835A-98395C3BC3BB}",
    "Music":     "{4BD8D571-6D19-48D3-BE97-422220080E43}",
    "Videos":    "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}",
}

KNOWN_FOLDERS: dict[str, Path] = {}
for _name, _guid in _FOLDER_GUIDS.items():
    _resolved = _sh_known_folder(_guid) or _expand(_name)
    KNOWN_FOLDERS[_name] = _resolved

# Reverse map: absolute path string → friendly name (longest match wins later).
_KNOWN_FOLDER_PATHS: dict[str, str] = {
    str(p).lower(): name for name, p in KNOWN_FOLDERS.items()
}

# Friendly names in Arabic (Egyptian colloquial) matching KNOWN_FOLDERS keys.
_FOLDER_AR: dict[str, str] = {
    "Desktop":   "سطح المكتب",
    "Downloads": "التحميلات",
    "Documents": "المستندات",
    "Pictures":  "الصور",
    "Music":     "الموسيقى",
    "Videos":    "الفيديوهات",
}


# ---------------------------------------------------------------------------
# Spoken folder aliases → canonical name used in KNOWN_FOLDERS
# ---------------------------------------------------------------------------

# These are the authoritative aliases imported by command_parser and
# intent_confidence — both used to import their own local copies, now centralised.
FOLDER_ALIASES: dict[str, str] = {
    # English
    "desktop": "Desktop",
    "downloads": "Downloads",
    "download": "Downloads",
    "documents": "Documents",
    "document": "Documents",
    "docs": "Documents",
    "pictures": "Pictures",
    "picture": "Pictures",
    "photos": "Pictures",
    "photo": "Pictures",
    "music": "Music",
    "videos": "Videos",
    "video": "Videos",
    # Arabic
    "سطح المكتب": "Desktop",
    "المكتب": "Desktop",
    # Phonetic spellings STT produces for "Desktop" in Egyptian Arabic
    "ديسكتوب": "Desktop",
    "ديسك توب": "Desktop",
    "الديسكتوب": "Desktop",
    "الديسك توب": "Desktop",
    "التحميلات": "Downloads",
    "التنزيلات": "Downloads",
    "تحميلات": "Downloads",
    "داونلودز": "Downloads",
    "المستندات": "Documents",
    "مستندات": "Documents",
    "الصور": "Pictures",
    "صور": "Pictures",
    "الموسيقى": "Music",
    "موسيقى": "Music",
    "الفيديوهات": "Videos",
    "فيديوهات": "Videos",
    "فيديو": "Videos",
}

# Set form used by intent_confidence for membership tests.
FOLDER_ALIAS_SET: frozenset[str] = frozenset(FOLDER_ALIASES)

# Richer search-path alias map used by command_parser (includes المكتب → Desktop).
SEARCH_PATH_ALIASES: dict[str, str] = {
    **FOLDER_ALIASES,
    "المكتب": "Desktop",
}


# ---------------------------------------------------------------------------
# Drive / partition aliases
# ---------------------------------------------------------------------------

# Egyptian-Arabic spoken letter names for the drive letters users actually
# say out loud ("قرص دي" / "قرص د" for D, "قرص سي" for C, …). Both the
# phonetic transliteration and the matching Arabic alphabet letter are
# accepted since speakers use either depending on habit.
_AR_LETTER_NAMES: dict[str, str] = {
    "a": "اي", "b": "بي", "c": "سي", "d": "دي", "e": "اي", "f": "اف",
    "g": "جي", "h": "اتش", "i": "اي", "j": "جاي", "k": "كي", "l": "ال",
    "m": "ام", "n": "ان", "o": "او", "p": "بي", "q": "كيو", "r": "ار",
    "s": "اس", "t": "تي", "u": "يو", "v": "في", "w": "دبليو", "x": "اكس",
    "y": "واي", "z": "زد",
}
# Direct Arabic-alphabet equivalents for the common drive letters (C/D/E)
# that Egyptian speakers commonly substitute by sound.
_AR_ALPHABET_EQUIV: dict[str, str] = {
    "c": "س", "d": "د", "e": "اي",
}


def _build_drive_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        lc = letter.lower()
        # English: "C drive", "drive C", "C partition", "partition C", "C:"
        for pat in (f"{lc} drive", f"drive {lc}", f"{lc} partition",
                    f"partition {lc}", f"{lc}:"):
            aliases[pat] = root
        # Arabic: "قرص ج", "باتشن ج", "قرص C"
        for pat in (f"قرص {lc}", f"باتشن {lc}", f"قسم {lc}",
                    f"درايف {lc}", f"بارتشن {lc}"):
            aliases[pat] = root
        # Arabic spoken letter name: "قرص دي", "قرص سي"
        ar_name = _AR_LETTER_NAMES.get(lc)
        if ar_name:
            for prefix in ("قرص", "باتشن", "قسم", "درايف", "بارتشن"):
                aliases[f"{prefix} {ar_name}"] = root
        # Arabic alphabet equivalent: "قرص د" for D, "قرص س" for C
        ar_equiv = _AR_ALPHABET_EQUIV.get(lc)
        if ar_equiv:
            for prefix in ("قرص", "باتشن", "قسم", "درايف", "بارتشن"):
                aliases[f"{prefix} {ar_equiv}"] = root
    return aliases

DRIVE_ALIASES: dict[str, str] = _build_drive_aliases()

# "This PC" equivalents that mean "enumerate all drives".
_THIS_PC_PHRASES: frozenset[str] = frozenset({
    "this pc", "my computer", "my pc", "computer",
    "الكمبيوتر", "جهازي", "هذا الكمبيوتر",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_AR_ARTICLE_RE = re.compile(r"^(?:الـ|لل|ال|لـ|ل)\s*", re.IGNORECASE)
_EN_ARTICLE_RE = re.compile(r"^(?:the|my)\s+", re.IGNORECASE)


def resolve_location(spoken: str) -> Optional[Path]:
    """Resolve a spoken location phrase to an absolute Path.

    Returns None when the phrase is not a recognisable location so callers
    can fall back to treating it as a literal path.
    """
    if not spoken:
        return None
    key = spoken.strip().lower()

    # 1. Known folder alias (Desktop, Downloads, …)
    folder_name = FOLDER_ALIASES.get(key)
    if folder_name:
        return KNOWN_FOLDERS.get(folder_name)

    # Strip Arabic definite-article prefix and retry ("الـ desktop" → "desktop").
    stripped = _AR_ARTICLE_RE.sub("", key).strip()
    if stripped != key:
        folder_name = FOLDER_ALIASES.get(stripped)
        if folder_name:
            return KNOWN_FOLDERS.get(folder_name)
        key = stripped

    # Strip English article prefix and retry ("the desktop" → "desktop").
    stripped_en = _EN_ARTICLE_RE.sub("", key).strip()
    if stripped_en != key:
        folder_name = FOLDER_ALIASES.get(stripped_en)
        if folder_name:
            return KNOWN_FOLDERS.get(folder_name)
        key = stripped_en

    # 2. Drive / partition alias ("C drive", "قرص د")
    drive_root = DRIVE_ALIASES.get(key)
    if drive_root:
        return Path(drive_root)

    # 3. Bare drive letter with optional colon/slash ("C", "C:", "D:\\")
    bare = re.match(r"^([a-z]):?\\?$", key, re.IGNORECASE)
    if bare:
        return Path(f"{bare.group(1).upper()}:\\")

    # 4. "This PC" — return user home as a reasonable default for listing
    if key in _THIS_PC_PHRASES:
        return Path(os.path.expanduser("~"))

    # 5. Absolute path that exists — pass through as-is
    candidate = Path(spoken.strip().strip('"').strip("'"))
    if candidate.is_absolute() and candidate.exists():
        return candidate

    return None


def humanize_path(path: "str | Path") -> dict[str, str]:
    """Return a human-readable EN + AR description for a file/directory path.

    Strips the user-profile prefix and replaces known-folder roots with
    their friendly names.  Keeps only the filename + at most the last two
    parent folder names so responses stay short enough to speak aloud.

    Returns {"en": "...", "ar": "..."}.
    """
    p = Path(path)
    name = p.name or str(p)

    # Find the longest matching known-folder root.
    folder_label: Optional[str] = None
    remaining_parts: list[str] = []
    parts = list(p.parts)

    for length in range(len(parts), 0, -1):
        prefix = Path(*parts[:length])
        prefix_str = str(prefix).lower()
        if prefix_str in _KNOWN_FOLDER_PATHS:
            folder_label = _KNOWN_FOLDER_PATHS[prefix_str]
            remaining_parts = parts[length:]
            break

    # Fallback: strip the user home prefix.
    if folder_label is None:
        home = Path(os.path.expanduser("~"))
        try:
            rel = p.relative_to(home)
            remaining_parts = list(rel.parts)
        except ValueError:
            remaining_parts = list(p.parts)

    # Cap to at most 2 intermediate folder names + filename.
    intermediate = [part for part in remaining_parts[:-1] if part] if remaining_parts else []
    intermediate = intermediate[-2:]  # keep last two at most

    # Build friendly description.
    if folder_label and intermediate:
        folder_ar = _FOLDER_AR.get(folder_label, folder_label)
        en = f"{name} in {folder_label}, folder {'/'.join(intermediate)}"
        ar = f"ملف {name} في {folder_ar}، مجلد {'/'.join(intermediate)}"
    elif folder_label:
        folder_ar = _FOLDER_AR.get(folder_label, folder_label)
        en = f"{name} in {folder_label}"
        ar = f"ملف {name} في {folder_ar}"
    elif intermediate:
        en = f"{name} in folder {'/'.join(intermediate)}"
        ar = f"ملف {name} في مجلد {'/'.join(intermediate)}"
    else:
        en = name
        ar = name

    return {"en": en, "ar": ar}
