"""Streaming sentence accumulator for TTS chunking.

Arabic mode: flushes on ؟ . ! \n; soft-splits at connectors (و/ف/ثم)
after `soft_flush_chars`; hard-cuts at `hard_flush_chars` regardless.
English mode: flushes on . ! ? \\n (same as prior regex behaviour).
"""
import re
from typing import Optional

try:
    from core.config import STREAM_AR_SOFT_FLUSH_CHARS as _CFG_SOFT, STREAM_AR_HARD_FLUSH_CHARS as _CFG_HARD
except Exception:
    _CFG_SOFT, _CFG_HARD = 50, 100

# Arabic sentence-enders and newline. Commas stay inside the current chunk to
# avoid restarting TTS playback at mid-sentence pauses.
_AR_PUNCT_RE = re.compile(r"[.!?؟\n]")
# English sentence-end: punctuation + whitespace, or punctuation at string end
_EN_SENT_END_RE = re.compile(r"[.!?؟\n]\s+|[.!?؟]$")
# Arabic connectors flanked by spaces — preferred soft-split points
_AR_CONNECTOR_RE = re.compile(r"(?<= )(?:ثم|و|ف)(?= )")


class SentenceBuffer:
    """Token-by-token accumulator that emits complete speakable sentences.

    Usage:
        buf = SentenceBuffer(is_arabic=True)
        for token in stream:
            sentence = buf.add_token(token)
            if sentence:
                speak(sentence)
        remainder = buf.flush()
        if remainder:
            speak(remainder)
    """

    def __init__(
        self,
        is_arabic: bool = False,
        soft_flush_chars: Optional[int] = None,
        hard_flush_chars: Optional[int] = None,
    ):
        self.is_arabic = is_arabic
        self.soft_flush_chars = int(soft_flush_chars if soft_flush_chars is not None else _CFG_SOFT)
        self.hard_flush_chars = int(hard_flush_chars if hard_flush_chars is not None else _CFG_HARD)
        self._buf = ""

    def add_token(self, token: str) -> Optional[str]:
        """Append *token* and return a flushed sentence if a boundary is found."""
        if not token:
            return None
        self._buf += token
        return self._check()

    def flush(self) -> str:
        """Force-flush whatever remains in the buffer."""
        result = self._buf.strip()
        self._buf = ""
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check(self) -> Optional[str]:
        if not self._buf.strip():
            return None
        return self._check_arabic() if self.is_arabic else self._check_english()

    def _check_arabic(self) -> Optional[str]:
        # Punctuation boundary — emit up to and including the mark
        m = _AR_PUNCT_RE.search(self._buf)
        if m:
            sentence = self._buf[: m.end()].strip()
            self._buf = self._buf[m.end() :].lstrip()
            return sentence or None

        text = self._buf.strip()
        length = len(text)

        # Soft flush: split at last Arabic connector or space
        if length >= self.soft_flush_chars:
            pos = self._soft_split_pos(text)
            if pos > 0:
                sentence = text[:pos].strip()
                if self._is_single_word(sentence):
                    return None
                self._buf = text[pos:].lstrip()
                return sentence or None

        # Hard flush: cut unconditionally
        if length >= self.hard_flush_chars:
            if self._is_single_word(text):
                return None
            self._buf = ""
            return text

        return None

    def _soft_split_pos(self, text: str) -> int:
        """Return the rightmost connector-word boundary, falling back to last space."""
        best = -1
        for m in _AR_CONNECTOR_RE.finditer(text):
            if m.start() > 0:
                best = m.start()
        if best > 0:
            return best
        pos = text.rfind(" ")
        return pos if pos > 0 else -1

    def _check_english(self) -> Optional[str]:
        m = _EN_SENT_END_RE.search(self._buf)
        if m:
            sentence = self._buf[: m.end()].strip()
            if self._is_single_word(sentence):
                return None
            self._buf = self._buf[m.end() :].lstrip()
            return sentence or None
        return None

    @staticmethod
    def _is_single_word(text: str) -> bool:
        words = [part for part in str(text or "").split() if part]
        return len(words) <= 1
