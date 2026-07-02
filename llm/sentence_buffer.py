"""Streaming sentence accumulator for TTS chunking.

Phase 5: language-aware streaming boundaries.

English:
  - Soft flush when there is punctuation and at least EN_SOFT_WORDS.
  - Hard flush when the buffer reaches EN_HARD_WORDS.

Egyptian Arabic:
  - Boundary chars include . ! ? ؟ ، ؛ and newline.
  - Soft flush when there is a boundary and at least AR_SOFT_WORDS.
  - Hard flush at AR_HARD_WORDS, but connector endings are held.
"""

from __future__ import annotations

import re
from typing import Optional

from core.config import (
    SENTENCE_BUFFER_AR_HARD_WORDS,
    SENTENCE_BUFFER_AR_SOFT_WORDS,
    SENTENCE_BUFFER_EN_HARD_WORDS,
    SENTENCE_BUFFER_EN_SOFT_WORDS,
    SENTENCE_BUFFER_HOLD_CONNECTORS,
)

_EN_BOUNDARY_CHARS = {".", "!", "?", "\n"}
_AR_BOUNDARY_CHARS = {".", "!", "?", "؟", "،", "؛", "\n"}
_CONNECTORS_EN = {"and", "then"}
_CONNECTORS_AR = {"و", "ف", "ثم"}
_WORD_RE = re.compile(r"\S+")


class SentenceBuffer:
    """Token-by-token accumulator that emits complete speakable chunks."""

    def __init__(
        self,
        is_arabic: bool = False,
        soft_flush_chars: Optional[int] = None,
        hard_flush_chars: Optional[int] = None,
        *,
        en_soft_words: Optional[int] = None,
        en_hard_words: Optional[int] = None,
        ar_soft_words: Optional[int] = None,
        ar_hard_words: Optional[int] = None,
        hold_connectors: Optional[bool] = None,
    ):
        self.is_arabic = bool(is_arabic)
        self.en_soft_words = int(en_soft_words or SENTENCE_BUFFER_EN_SOFT_WORDS)
        self.en_hard_words = int(en_hard_words or SENTENCE_BUFFER_EN_HARD_WORDS)
        self.ar_soft_words = int(ar_soft_words or SENTENCE_BUFFER_AR_SOFT_WORDS)
        self.ar_hard_words = int(ar_hard_words or SENTENCE_BUFFER_AR_HARD_WORDS)
        self.hold_connectors = bool(SENTENCE_BUFFER_HOLD_CONNECTORS if hold_connectors is None else hold_connectors)
        # Backward-compatible parameters are accepted but no longer drive flushes.
        _ = soft_flush_chars, hard_flush_chars
        self._buf = ""

    def add_token(self, token: str) -> Optional[str]:
        if not token:
            return None
        self._buf += str(token)
        return self._check()

    def flush(self) -> str:
        result = self._buf.strip()
        self._buf = ""
        return result

    def _check(self) -> Optional[str]:
        if not self._buf.strip():
            return None
        return self._check_arabic() if self.is_arabic else self._check_english()

    def _check_arabic(self) -> Optional[str]:
        return self._check_language(
            boundary_chars=_AR_BOUNDARY_CHARS,
            soft_words=self.ar_soft_words,
            hard_words=self.ar_hard_words,
            connectors=_CONNECTORS_AR,
        )

    def _check_english(self) -> Optional[str]:
        return self._check_language(
            boundary_chars=_EN_BOUNDARY_CHARS,
            soft_words=self.en_soft_words,
            hard_words=self.en_hard_words,
            connectors=_CONNECTORS_EN,
        )

    def _check_language(self, *, boundary_chars: set[str], soft_words: int, hard_words: int, connectors: set[str]) -> Optional[str]:
        text = self._buf
        word_count = self._word_count(text)
        boundary_pos = self._last_boundary_pos(text, boundary_chars)

        if boundary_pos >= 0 and word_count >= soft_words:
            candidate = text[: boundary_pos + 1].strip()
            if candidate and not self._ends_with_connector(candidate, connectors):
                self._buf = text[boundary_pos + 1 :].lstrip()
                return candidate

        if word_count >= hard_words:
            if self.hold_connectors and self._ends_with_connector(text, connectors):
                return None
            cut = self._hard_cut_pos(text, hard_words)
            candidate = text[:cut].strip() if cut > 0 else ""
            if self.hold_connectors and candidate and self._ends_with_connector(candidate, connectors):
                next_cut = self._hard_cut_pos(text, hard_words + 1)
                if next_cut > cut:
                    cut = next_cut
            if cut > 0:
                candidate = text[:cut].strip()
                if candidate and not self._is_single_word(candidate):
                    self._buf = text[cut:].lstrip()
                    return candidate

        return None

    @staticmethod
    def _word_count(text: str) -> int:
        return len(_WORD_RE.findall(str(text or "")))

    @staticmethod
    def _last_boundary_pos(text: str, boundary_chars: set[str]) -> int:
        best = -1
        for index, char in enumerate(str(text or "")):
            if char in boundary_chars:
                best = index
        return best

    @staticmethod
    def _hard_cut_pos(text: str, hard_words: int) -> int:
        matches = list(_WORD_RE.finditer(str(text or "")))
        if len(matches) < hard_words:
            return -1
        return matches[hard_words - 1].end()

    @staticmethod
    def _ends_with_connector(text: str, connectors: set[str]) -> bool:
        words = _WORD_RE.findall(str(text or "").strip())
        if not words:
            return False
        last = words[-1].strip(".,!?؟،؛:;-").lower()
        return last in connectors

    @staticmethod
    def _is_single_word(text: str) -> bool:
        return SentenceBuffer._word_count(text) <= 1
