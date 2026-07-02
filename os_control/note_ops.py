"""Note-taking operations — save dictated text as .txt files on the Desktop."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from core.config import NOTE_BASENAME, NOTE_DIR
from core.logger import logger


def _resolve_note_dir() -> Path:
    """Return the directory where notes are saved (Desktop by default)."""
    from os_control.path_resolver import KNOWN_FOLDERS, resolve_location

    # Try the configured alias first (e.g. "Desktop", "Documents")
    resolved = resolve_location(NOTE_DIR)
    if resolved and resolved.exists():
        return resolved

    # Fall back to KNOWN_FOLDERS lookup by canonical name
    canonical = NOTE_DIR.strip().title()
    if canonical in KNOWN_FOLDERS:
        p = KNOWN_FOLDERS[canonical]
        if p.exists():
            return p

    # Last resort: ~/Desktop
    fallback = Path.home() / "Desktop"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def next_note_name(dir_path: Path, basename: str) -> str:
    """Return the next auto-numbered note name, e.g. 'note 3'."""
    pattern = re.compile(
        r"^" + re.escape(basename) + r"\s+(\d+)\.txt$",
        re.IGNORECASE,
    )
    max_n = 0
    try:
        for entry in dir_path.iterdir():
            m = pattern.match(entry.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
    except Exception:
        pass
    return f"{basename} {max_n + 1}"


def save_note(text: str, name: Optional[str] = None, language: str = "en") -> str:
    """Write *text* to a .txt file in the note directory.

    If *name* is None, auto-numbers as 'note 1', 'note 2', …
    Returns a bilingual confirmation string.
    """
    is_ar = str(language or "").strip().lower().startswith("ar")
    dir_path = _resolve_note_dir()

    if name:
        # Strip any user-supplied .txt suffix to avoid double extension
        stem = re.sub(r"\.txt$", "", name.strip(), flags=re.IGNORECASE)
    else:
        stem = next_note_name(dir_path, NOTE_BASENAME)

    file_path = dir_path / f"{stem}.txt"

    try:
        file_path.write_text(str(text or "").strip(), encoding="utf-8")
        logger.info("Note saved: %s", file_path)
    except Exception as exc:
        logger.error("Failed to save note to %s: %s", file_path, exc)
        if is_ar:
            return "مش قادر أحفظ النوتة دي."
        return "Couldn't save the note."

    if is_ar:
        return f"اتحفظت باسم {stem} على الـ Desktop."
    return f"Saved as '{stem}' on your Desktop."
