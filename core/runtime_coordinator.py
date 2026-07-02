"""Runtime phase coordinator for wake-word interrupt gating.

Tracks the current execution phase so the wake-word listener can decide
whether an interrupt is allowed (THINKING, SPEAKING) or blocked
(EXECUTING_COMMAND, RECORDING, TRANSCRIBING, ROUTING).
"""

from __future__ import annotations

import enum
import threading

from core.logger import get_logger

log = get_logger("coordinator")


class RuntimePhase(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    ROUTING = "routing"
    EXECUTING_COMMAND = "executing_command"
    THINKING = "thinking"
    SPEAKING = "speaking"


_INTERRUPTIBLE = frozenset({RuntimePhase.THINKING, RuntimePhase.SPEAKING})


def _play_ack_tone() -> None:
    try:
        from core.config import (
            WAKE_INTERRUPT_ACK_SOUND,
            WAKE_INTERRUPT_ACK_FREQ_HZ,
            WAKE_INTERRUPT_ACK_DURATION_MS,
        )
        if not WAKE_INTERRUPT_ACK_SOUND:
            return
        from audio.cues import play_cue
        play_cue(int(WAKE_INTERRUPT_ACK_FREQ_HZ), int(WAKE_INTERRUPT_ACK_DURATION_MS))
    except Exception:
        pass


def _play_blocked_tone() -> None:
    try:
        from core.config import (
            WAKE_INTERRUPT_BLOCKED_TONE_ENABLED,
            WAKE_INTERRUPT_BLOCKED_TONE_FREQ_HZ,
            WAKE_INTERRUPT_BLOCKED_TONE_DURATION_MS,
        )
        if not WAKE_INTERRUPT_BLOCKED_TONE_ENABLED:
            return
        from audio.cues import play_cue
        play_cue(int(WAKE_INTERRUPT_BLOCKED_TONE_FREQ_HZ), int(WAKE_INTERRUPT_BLOCKED_TONE_DURATION_MS))
    except Exception:
        pass


class RuntimeCoordinator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phase = RuntimePhase.IDLE
        self._speech_engine = None
        self._llm_cancel_event: threading.Event | None = None

    @property
    def current_phase(self) -> RuntimePhase:
        return self._phase

    def set_phase(self, phase: RuntimePhase) -> None:
        with self._lock:
            self._phase = phase

    def attach_speech_engine(self, engine) -> None:
        self._speech_engine = engine

    def attach_llm_cancel_event(self, event: threading.Event) -> None:
        self._llm_cancel_event = event

    def can_interrupt(self) -> bool:
        return self._phase in _INTERRUPTIBLE

    def request_interrupt(self) -> bool:
        with self._lock:
            if self._phase not in _INTERRUPTIBLE:
                log.info(
                    "Interrupt blocked (phase=%s).",
                    self._phase.value,
                )
                _play_blocked_tone()
                return False

            if self._speech_engine is not None:
                try:
                    self._speech_engine.interrupt()
                except Exception as exc:
                    log.warning("Failed to interrupt TTS: %s", exc)

            if self._llm_cancel_event is not None:
                self._llm_cancel_event.set()

            self._phase = RuntimePhase.LISTENING
            log.info("Interrupt accepted — TTS/LLM cancelled, ready to listen.")
            _play_ack_tone()
            return True


coordinator = RuntimeCoordinator()
