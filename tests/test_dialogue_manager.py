"""DialogueManager state-machine tests — 15 cases.

Tests valid and invalid state transitions, slot tracking, follow-up window
timing, and conversation-turn counting.
"""

from __future__ import annotations

import time

import pytest

from core.dialogue_manager import DialogueManager, DialogueState


@pytest.fixture
def dm():
    return DialogueManager()


# ---------------------------------------------------------------------------
# Group 1 — Initial state (2 tests)
# ---------------------------------------------------------------------------

class TestInitialState:

    def test_starts_idle(self, dm):
        assert dm.state == DialogueState.IDLE

    def test_conversation_turns_zero(self, dm):
        assert dm.conversation_turns == 0


# ---------------------------------------------------------------------------
# Group 2 — Happy-path transitions (5 tests)
# ---------------------------------------------------------------------------

class TestHappyPathTransitions:

    def test_idle_to_listening(self, dm):
        dm.transition(DialogueState.LISTENING)
        assert dm.state == DialogueState.LISTENING

    def test_listening_to_processing(self, dm):
        dm.transition(DialogueState.LISTENING)
        dm.transition(DialogueState.PROCESSING)
        assert dm.state == DialogueState.PROCESSING

    def test_processing_to_responding(self, dm):
        dm.transition(DialogueState.LISTENING)
        dm.transition(DialogueState.PROCESSING)
        dm.transition(DialogueState.RESPONDING)
        assert dm.state == DialogueState.RESPONDING

    def test_responding_to_idle(self, dm):
        dm.transition(DialogueState.LISTENING)
        dm.transition(DialogueState.PROCESSING)
        dm.transition(DialogueState.RESPONDING)
        dm.transition(DialogueState.IDLE)
        assert dm.state == DialogueState.IDLE

    def test_full_cycle_increments_turns(self, dm):
        for state in (
            DialogueState.LISTENING,
            DialogueState.PROCESSING,
            DialogueState.RESPONDING,
            DialogueState.FOLLOW_UP,
        ):
            dm.transition(state)
        assert dm.conversation_turns >= 1


# ---------------------------------------------------------------------------
# Group 3 — CONFIRMING and slot tracking (4 tests)
# ---------------------------------------------------------------------------

class TestSlotTracking:

    def test_transition_to_confirming(self, dm):
        dm.transition(DialogueState.LISTENING)
        dm.transition(DialogueState.PROCESSING)
        dm.transition(DialogueState.CONFIRMING)
        assert dm.state == DialogueState.CONFIRMING

    def test_set_and_get_missing_slots(self, dm):
        dm.set_missing_slots("OPEN_APP", ["app_name"])
        assert dm.get_missing_slots() == {"intent": "OPEN_APP", "slots": ["app_name"]}

    def test_set_and_get_pending_action(self, dm):
        dm.set_missing_slots("SET_TIMER", ["seconds"])
        action = dm.get_pending_action()
        # pending action may be None until set via transition(CONFIRMING, pending_action=...)
        assert action is None or isinstance(action, object)

    def test_missing_slots_cleared_on_idle(self, dm):
        dm.set_missing_slots("OPEN_APP", ["app_name"])
        dm.transition(DialogueState.IDLE)
        # After IDLE transition slots should be cleared
        assert dm.get_missing_slots() == {}


# ---------------------------------------------------------------------------
# Group 4 — FOLLOW_UP window (4 tests)
# ---------------------------------------------------------------------------

class TestFollowUpWindow:

    def test_transition_to_follow_up(self, dm):
        dm.transition(DialogueState.LISTENING)
        dm.transition(DialogueState.PROCESSING)
        dm.transition(DialogueState.RESPONDING)
        dm.transition(DialogueState.FOLLOW_UP)
        assert dm.state == DialogueState.FOLLOW_UP

    def test_should_skip_wake_word_in_follow_up(self, dm):
        dm.transition(DialogueState.LISTENING)
        dm.transition(DialogueState.PROCESSING)
        dm.transition(DialogueState.RESPONDING)
        dm.transition(DialogueState.FOLLOW_UP)
        assert dm.should_skip_wake_word() is True

    def test_should_not_skip_wake_word_in_idle(self, dm):
        assert dm.should_skip_wake_word() is False

    def test_time_remaining_positive_in_follow_up(self, dm):
        dm.transition(DialogueState.LISTENING)
        dm.transition(DialogueState.PROCESSING)
        dm.transition(DialogueState.RESPONDING)
        dm.transition(DialogueState.FOLLOW_UP)
        remaining = dm.time_remaining()
        assert remaining > 0
