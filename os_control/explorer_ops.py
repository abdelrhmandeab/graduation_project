"""Phase 5 -- open paths and reveal files in Windows Explorer.

Two entry points:
  open_in_explorer(path, language)   -- opens a folder (or the parent of a
                                        file) in Explorer
  reveal_in_explorer(path, language) -- opens Explorer and selects the item
                                        (`explorer /select,<path>`)

Both resolve known folder aliases (Desktop, Downloads, …) via path_resolver
and fall back gracefully when the path does not exist.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from core.logger import logger
from os_control.path_resolver import KNOWN_FOLDERS, resolve_location


# ---------------------------------------------------------------------------
# Location-qualifier stripping
# ---------------------------------------------------------------------------

# Matches "X في Y", "X in Y", "X from Y", "X من Y" — same as parser's
# _TRAILING_LOCATION_RE but used here independently.
_LOCATION_SUFFIX_RE = re.compile(
    r"^(.*?)\s+(?:in|من|في|from|inside|بداخل|داخل)\s+(.+)$",
    re.IGNORECASE | re.UNICODE,
)

# Arabic definite article prefix: "الـ", "ال", "ـ" (tatweel leftover)
_AR_ARTICLE_RE = re.compile(r"^(?:الـ|ال|ـ)", re.UNICODE)

# File/folder filler words that STT inserts before the actual name
_FILE_FILLER_RE = re.compile(
    r"^(?:الملف|المجلد|الملفات|ملف|مجلد|file|folder|the\s+file|the\s+folder)\s+",
    re.IGNORECASE | re.UNICODE,
)

# Trailing punctuation STT sometimes appends
_TRAIL_PUNCT_RE = re.compile(r"[.,،؟?!]+$")

# Map spoken folder names to KNOWN_FOLDERS keys (lowercase spoken → key)
_FOLDER_SPOKEN_MAP: dict[str, str] = {
    "desktop": "Desktop",
    "downloads": "Downloads",
    "download": "Downloads",
    "documents": "Documents",
    "document": "Documents",
    "pictures": "Pictures",
    "music": "Music",
    "videos": "Videos",
    "داونلود": "Downloads",
    "داونلودز": "Downloads",
    "التحميلات": "Downloads",
    "تحميلات": "Downloads",
    "سطح المكتب": "Desktop",
    "المستندات": "Documents",
    "الصور": "Pictures",
    "الموسيقى": "Music",
    "الفيديوهات": "Videos",
    "مستندات": "Documents",
    "صور": "Pictures",
}


def _spoken_to_folder(name: str) -> Path | None:
    """Map a spoken folder name to a real Path, or None."""
    cleaned = _TRAIL_PUNCT_RE.sub("", name.strip()).strip()
    # Strip Arabic article
    no_article = _AR_ARTICLE_RE.sub("", cleaned).strip()
    for candidate in (cleaned.lower(), no_article.lower()):
        key = _FOLDER_SPOKEN_MAP.get(candidate)
        if key and key in KNOWN_FOLDERS:
            return KNOWN_FOLDERS[key]
    # Try resolve_location (handles all aliases)
    path = resolve_location(no_article or cleaned)
    if path and path.exists():
        return path
    return None


def _fuzzy_find(filename: str, search_root: Path, *, recursive: bool = False) -> Path | None:
    """Case-insensitive filename search within *search_root*.

    When *recursive* is True, walks all subdirectories after failing the
    shallow pass.  Capped at 5 000 entries to stay fast.
    """
    try:
        target = filename.lower()
        stem = re.sub(r"\.[a-zA-Z0-9]{1,5}$", "", target)
        has_ext = target != stem  # filename had an extension
        # When user says "CV PDF" (space, no dot), last token may be the extension
        # and first tokens form the name stem, e.g. "cv pdf" → name="cv", ext="pdf"
        ext_hint: str = ""
        name_stem: str = stem
        if not has_ext and " " in stem:
            parts = stem.split()
            _ext_like = {"pdf", "doc", "docx", "txt", "xlsx", "pptx", "mp4", "mp3",
                         "png", "jpg", "jpeg", "zip", "py", "js", "ts", "md", "csv"}
            if parts[-1] in _ext_like:
                ext_hint = parts[-1]
                name_stem = " ".join(parts[:-1])

        def _matches(name_lower: str) -> bool:
            if name_lower == target:
                return True
            if len(stem) >= 3:
                if name_lower.startswith(stem):
                    return True
                if not has_ext and stem in name_lower:
                    return True
            # "CV PDF" → match files whose stem starts with "cv" and ext is "pdf"
            if ext_hint and name_stem:
                file_stem = re.sub(r"\.[a-zA-Z0-9]{1,5}$", "", name_lower)
                file_ext = name_lower[len(file_stem):].lstrip(".")
                if file_ext == ext_hint and (
                    file_stem == name_stem
                    or file_stem.startswith(name_stem)
                    or (len(name_stem) >= 2 and name_stem in file_stem)
                ):
                    return True
            return False

        # Shallow pass (files only)
        shallow_files = [e for e in search_root.iterdir() if e.is_file()]
        for entry in shallow_files:
            if _matches(entry.name.lower()):
                return entry

        if recursive:
            _SKIP_DIRS = {
                "node_modules", ".git", ".svn", "__pycache__", ".venv", "venv",
                ".tox", "dist", "build", ".mypy_cache", ".pytest_cache",
            }
            count = 0
            for dirpath, dirs, file_names in os.walk(search_root, topdown=True):
                # Prune noisy dirs in-place so os.walk skips them entirely
                dirs[:] = [d for d in dirs if d.lower() not in _SKIP_DIRS and not d.startswith(".")]
                for fname in file_names:
                    count += 1
                    if count > 5000:
                        return None
                    if _matches(fname.lower()):
                        return Path(dirpath) / fname
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_path(raw: str) -> Path | None:
    """Resolve *raw* to an absolute Path, stripping location qualifiers."""
    raw = _TRAIL_PUNCT_RE.sub("", raw.strip()).strip()
    if not raw:
        return None

    # Split off trailing location qualifier ("X في Downloads" → name="X", folder="Downloads")
    filename: str = raw
    folder_hint: Path | None = None
    loc_match = _LOCATION_SUFFIX_RE.match(raw)
    if loc_match:
        filename = loc_match.group(1).strip()
        folder_hint = _spoken_to_folder(loc_match.group(2).strip())

    # Strip file/folder filler words ("الملف X" → "X", "file X" → "X")
    filename = _FILE_FILLER_RE.sub("", filename).strip()
    # Strip Arabic article ("الـ downloads" → "downloads", "ـ downloads" → "downloads")
    filename_clean = _AR_ARTICLE_RE.sub("", filename).strip()
    filename_clean = _TRAIL_PUNCT_RE.sub("", filename_clean).strip()

    # 1. If folder_hint given, search there first
    if folder_hint:
        # Maybe they gave us just a folder name (e.g. "وريني مكان الـ downloads")
        if folder_hint.exists() and not filename_clean:
            return folder_hint
        if filename_clean:
            exact = folder_hint / filename_clean
            if exact.exists():
                return exact
            fuzzy = _fuzzy_find(filename_clean, folder_hint)
            if fuzzy:
                return fuzzy

    # 2. Maybe the whole thing is a known folder alias
    folder_as_alias = _spoken_to_folder(filename_clean) or _spoken_to_folder(filename)
    if folder_as_alias:
        return folder_as_alias

    # 3. Known folder aliases via path_resolver
    alias = resolve_location(filename_clean) or resolve_location(filename)
    if alias and alias.exists():
        return alias

    # 4. Absolute / relative path
    for name in (filename_clean, filename):
        expanded = Path(os.path.expandvars(os.path.expanduser(name)))
        if expanded.exists():
            return expanded

    # 5. Search all known folder roots (bare filename without location hint)
    for folder_path in KNOWN_FOLDERS.values():
        exact = folder_path / filename_clean
        if exact.exists():
            return exact
        if filename_clean != filename:
            exact2 = folder_path / filename
            if exact2.exists():
                return exact2

    # 6. Fuzzy match (shallow) across known folder roots
    if filename_clean:
        search_roots = [folder_hint] if folder_hint else list(KNOWN_FOLDERS.values())
        for root in search_roots:
            if root:
                hit = _fuzzy_find(filename_clean, root)
                if hit:
                    return hit

    # 7. Recursive fuzzy search when shallow passes all failed
    if filename_clean:
        search_roots = [folder_hint] if folder_hint else list(KNOWN_FOLDERS.values())
        for root in search_roots:
            if root:
                hit = _fuzzy_find(filename_clean, root, recursive=True)
                if hit:
                    return hit

    return None


def _launch_explorer(args: list[str]) -> bool:
    """Run explorer.exe with *args*.  Returns True on successful launch."""
    try:
        subprocess.Popen(["explorer.exe"] + args)
        return True
    except Exception as exc:
        logger.debug("explorer_ops: Popen failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def open_in_explorer(raw_path: str, language: str = "en") -> tuple[bool, str, dict]:
    """Open *raw_path* (folder, or parent of a file) in File Explorer.

    Returns (success, spoken_message, meta).
    """
    is_ar = language == "ar"
    path = _resolve_path(raw_path.strip()) if raw_path else None

    if path is None:
        # Unknown path — try opening the raw string directly; Explorer will
        # show an error dialog itself if the path is invalid.
        if raw_path.strip():
            _launch_explorer([raw_path.strip()])
            if is_ar:
                msg = f"بفتحلك المستكشف على «{raw_path.strip()}»."
            else:
                msg = f"Opening Explorer at «{raw_path.strip()}»."
            return True, msg, {"method": "explorer_raw", "path": raw_path}
        if is_ar:
            return False, "مش عارف المسار ده.", {}
        return False, "I don't recognise that path.", {}

    # If path is a file, open its parent folder
    target = path if path.is_dir() else path.parent

    ok = _launch_explorer([str(target)])
    if not ok:
        if is_ar:
            return False, "معرفتش أفتح المستكشف.", {}
        return False, "Couldn't open File Explorer.", {}

    display = path.name or str(target)
    if is_ar:
        msg = f"فتحت المستكشف على «{display}»."
    else:
        msg = f"Opened Explorer at «{display}»."
    return True, msg, {"method": "explorer_open", "path": str(target)}


def reveal_in_explorer(raw_path: str, language: str = "en") -> tuple[bool, str, dict]:
    """Open File Explorer with *raw_path* selected/highlighted.

    Returns (success, spoken_message, meta).
    """
    is_ar = language == "ar"
    path = _resolve_path(raw_path.strip()) if raw_path else None

    if path is None:
        if is_ar:
            return False, "مش لاقي الملف أو المجلد ده.", {}
        return False, "I can't find that file or folder.", {}

    # If it's a folder, just open it rather than /select (which would select
    # the folder itself inside its parent, not show its contents).
    if path.is_dir():
        return open_in_explorer(raw_path, language=language)

    ok = _launch_explorer(["/select,", str(path)])
    if not ok:
        if is_ar:
            return False, "معرفتش أفتح المستكشف.", {}
        return False, "Couldn't open File Explorer.", {}

    if is_ar:
        msg = f"وريتلك «{path.name}» في المستكشف."
    else:
        msg = f"Revealed «{path.name}» in Explorer."
    return True, msg, {"method": "explorer_reveal", "path": str(path)}


def open_file(raw_path: str, language: str = "en") -> tuple[bool, str, dict]:
    """Open *raw_path* with its default application (like double-clicking it).

    For folders, opens File Explorer instead.
    Returns (success, spoken_message, meta).
    """
    is_ar = language == "ar"
    path = _resolve_path(raw_path.strip()) if raw_path else None

    if path is None:
        if is_ar:
            return False, "مش لاقي الملف أو المجلد ده.", {}
        return False, "I can't find that file or folder.", {}

    if path.is_dir():
        return open_in_explorer(raw_path, language=language)

    try:
        os.startfile(str(path))
    except Exception as exc:
        logger.debug("open_file startfile failed for %s: %s", path, exc)
        if is_ar:
            return False, "معرفتش أفتح الملف ده.", {}
        return False, "Couldn't open that file.", {}

    if is_ar:
        msg = f"فتحت «{path.name}»."
    else:
        msg = f"Opened «{path.name}»."
    return True, msg, {"method": "startfile", "path": str(path)}
