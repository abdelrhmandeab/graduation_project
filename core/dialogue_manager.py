"""Dialogue State Machine — Task 3.1.

Tracks the conversational state of a single Jarvis session so the main loop
can decide whether to require a wake word or to listen directly for a follow-up.

States
------
IDLE        Waiting for wake word.
LISTENING   Capturing user speech (after wake or follow-up).
PROCESSING  STT + NLU + routing in progress.
CONFIRMING  Asked user yes/no or PIN — waiting for answer (up to 30 s).
EXECUTING   Running the action (reserved for future slot-filling integration).
RESPONDING  TTS speaking the response.
FOLLOW_UP   10-second window after response — user may speak without wake word.

Key design decisions
--------------------
* FOLLOW_UP and CONFIRMING auto-expire when their timeout elapses; the next
  call to ``dialogue_manager.state`` observes the expiry and transitions to
  IDLE automatically.
* ``should_skip_wake_word()`` is the only public gate the orchestrator needs
  to check; it respects both the FOLLOW_UP and CONFIRMING states and the
  JARVIS_FOLLOWUP_ENABLED config flag.
* The module-level ``_follow_up_wake_event`` lets the background pipeline
  thread (where _process_utterance runs) poke the wake-word listener in the
  main thread so it exits immediately and hands control back to the follow-up
  VAD path — the same pattern used by audio/barge_in.py.
"""

from __future__ import annotations

import threading
import time
from enum import Enum

from core.config import FOLLOWUP_ENABLED, FOLLOWUP_WINDOW_SECONDS
from core.logger import logger


class DialogueState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    CONFIRMING = "confirming"
    EXECUTING = "executing"
    RESPONDING = "responding"
    FOLLOW_UP = "follow_up"


# Module-level event: set by _process_utterance (background thread) when a
# follow-up window opens.  listen_for_wake_word() checks it every chunk and
# returns "follow_up" as soon as it fires — no destructive interrupt needed.
_follow_up_wake_event = threading.Event()


def notify_follow_up_wake() -> None:
    """Signal that a follow-up window has opened; wake-word loop should exit."""
    _follow_up_wake_event.set()


def consume_follow_up_wake() -> bool:
    """Return True and clear the flag if a follow-up wake signal is pending."""
    if _follow_up_wake_event.is_set():
        _follow_up_wake_event.clear()
        return True
    return False


class DialogueManager:
    def __init__(self) -> None:
        self._state = DialogueState.IDLE
        self._state_entered_at = time.monotonic()
        self._follow_up_timeout = max(1.0, float(FOLLOWUP_WINDOW_SECONDS))
        self._confirming_timeout = 30.0
        self._pending_action = None
        self._missing_slots: dict = {}
        self._conversation_turns = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> DialogueState:
        """Current state with auto-expiry for FOLLOW_UP and CONFIRMING."""
        with self._lock:
            now = time.monotonic()
            if self._state == DialogueState.FOLLOW_UP:
                if now - self._state_entered_at > self._follow_up_timeout:
                    logger.debug("Dialogue: FOLLOW_UP expired → IDLE")
                    self._state = DialogueState.IDLE
                    self._conversation_turns = 0
            elif self._state == DialogueState.CONFIRMING:
                if now - self._state_entered_at > self._confirming_timeout:
                    logger.debug("Dialogue: CONFIRMING expired → IDLE")
                    self._state = DialogueState.IDLE
                    self._pending_action = None
            return self._state

    def time_remaining(self) -> float:
        """Seconds left in the current timed state (FOLLOW_UP or CONFIRMING)."""
        with self._lock:
            now = time.monotonic()
            if self._state == DialogueState.FOLLOW_UP:
                return max(0.0, self._follow_up_timeout - (now - self._state_entered_at))
            if self._state == DialogueState.CONFIRMING:
                return max(0.0, self._confirming_timeout - (now - self._state_entered_at))
        return 0.0

    def transition(self, new_state: DialogueState, **kwargs) -> None:
        with self._lock:
            old = self._state
            self._state = new_state
            self._state_entered_at = time.monotonic()
            if new_state == DialogueState.CONFIRMING:
                self._pending_action = kwargs.get("pending_action")
            if new_state == DialogueState.FOLLOW_UP:
                self._conversation_turns += 1
            if new_state == DialogueState.IDLE:
                self._conversation_turns = 0
                self._pending_action = None
                self._missing_slots = {}
                # Discard any pending follow-up wake signal so a stale event
                # cannot cause listen_for_wake_word() to spuriously return
                # "follow_up" after the window has already expired.
                _follow_up_wake_event.clear()
        logger.debug("Dialogue: %s → %s", old.value, new_state.value)

    def should_skip_wake_word(self) -> bool:
        """True when the user may speak without the wake word."""
        if not FOLLOWUP_ENABLED:
            return False
        return self.state in {DialogueState.FOLLOW_UP, DialogueState.CONFIRMING}

    def set_missing_slots(self, intent: str, slots: list) -> None:
        """Called when NLU detects missing required parameters for an intent."""
        with self._lock:
            self._missing_slots = {"intent": intent, "slots": list(slots)}
        self.transition(DialogueState.CONFIRMING)

    def get_missing_slots(self) -> dict:
        with self._lock:
            return dict(self._missing_slots)

    def get_pending_action(self):
        with self._lock:
            return self._pending_action

    @property
    def conversation_turns(self) -> int:
        """Number of follow-up turns in the current conversation."""
        with self._lock:
            return self._conversation_turns


dialogue_manager = DialogueManager()
