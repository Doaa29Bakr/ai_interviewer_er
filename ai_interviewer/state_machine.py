"""
Interview State Machine
=======================

Manages the interview lifecycle through four stages:

    INTRO  →  ASK  →  FOLLOWUP  →  CLOSE

- INTRO    : Greet the candidate, set context, explain the process.
- ASK      : Pose the main interview question for the current round.
- FOLLOWUP : Probe deeper based on the candidate's answer.
- CLOSE    : Wrap up, thank the candidate, and summarize.

Transitions
-----------
    INTRO    → ASK                (after introduction is complete)
    ASK      → FOLLOWUP           (after candidate answers a question)
    FOLLOWUP → ASK                (loop back for the next question)
    FOLLOWUP → CLOSE              (no more questions — end interview)
    CLOSE    → (terminal state)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------

class InterviewState(str, Enum):
    """Enumeration of the interview stages."""

    INTRO    = "INTRO"
    WARMUP   = "WARMUP"
    ASK      = "ASK"
    FOLLOWUP = "FOLLOWUP"
    CLOSE    = "CLOSE"


# ---------------------------------------------------------------------------
# Transition table
# ---------------------------------------------------------------------------

# Maps each state to the set of states it is allowed to transition into.
TRANSITIONS: dict[InterviewState, set[InterviewState]] = {
    InterviewState.INTRO:    {InterviewState.WARMUP},
    InterviewState.WARMUP:   {InterviewState.ASK},
    InterviewState.ASK:      {InterviewState.FOLLOWUP},
    InterviewState.FOLLOWUP: {InterviewState.ASK, InterviewState.CLOSE},
    InterviewState.CLOSE:    set(),                       # terminal — no exits
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class InvalidTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""

    def __init__(self, current: InterviewState, target: InterviewState) -> None:
        self.current = current
        self.target = target
        allowed = TRANSITIONS.get(current, set())
        super().__init__(
            f"Cannot transition from {current.value} → {target.value}. "
            f"Allowed transitions from {current.value}: "
            f"{', '.join(s.value for s in allowed) or '(none — terminal state)'}."
        )


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

class InterviewStateMachine:
    """
    Drives an interview through the  INTRO → ASK → FOLLOWUP → CLOSE  lifecycle.

    Usage
    -----
    >>> sm = InterviewStateMachine()
    >>> sm.state
    <InterviewState.INTRO: 'INTRO'>

    >>> sm.transition_to(InterviewState.ASK)
    <InterviewState.ASK: 'ASK'>

    >>> sm.transition_to(InterviewState.FOLLOWUP)
    >>> sm.transition_to(InterviewState.ASK)       # loop back for next question
    >>> sm.transition_to(InterviewState.FOLLOWUP)
    >>> sm.transition_to(InterviewState.CLOSE)      # end the interview

    >>> sm.is_terminal
    True
    """

    def __init__(self) -> None:
        self._state: InterviewState = InterviewState.INTRO
        self._history: list[InterviewState] = [InterviewState.INTRO]
        self._context: dict[str, Any] = {}
        self._hooks: dict[InterviewState, list[Callable[[InterviewState, InterviewState], None]]] = {
            s: [] for s in InterviewState
        }

    # -- Properties ---------------------------------------------------------

    @property
    def state(self) -> InterviewState:
        """Return the current interview state."""
        return self._state

    @property
    def history(self) -> list[InterviewState]:
        """Return the ordered list of states visited (including the current one)."""
        return list(self._history)

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` if the interview has reached the CLOSE state."""
        return self._state == InterviewState.CLOSE

    @property
    def context(self) -> dict[str, Any]:
        """Mutable bag of key/value data carried across states."""
        return self._context

    @property
    def questions_asked(self) -> int:
        """Return how many ASK states have been entered so far."""
        return self._history.count(InterviewState.ASK)

    @property
    def followups_asked(self) -> int:
        """Return how many FOLLOWUP states have been entered so far."""
        return self._history.count(InterviewState.FOLLOWUP)

    # -- Allowed transitions ------------------------------------------------

    def allowed_transitions(self) -> set[InterviewState]:
        """Return the set of states reachable from the current state."""
        return set(TRANSITIONS.get(self._state, set()))

    def can_transition_to(self, target: InterviewState) -> bool:
        """Check whether transitioning to *target* is legal."""
        return target in TRANSITIONS.get(self._state, set())

    # -- Transition ---------------------------------------------------------

    def transition_to(self, target: InterviewState) -> InterviewState:
        """
        Move the state machine to *target*.

        Parameters
        ----------
        target : InterviewState
            The desired next state.

        Returns
        -------
        InterviewState
            The new current state (equal to *target*).

        Raises
        ------
        InvalidTransitionError
            If the transition is not allowed by the transition table.
        """
        if not self.can_transition_to(target):
            raise InvalidTransitionError(self._state, target)

        previous = self._state
        self._state = target
        self._history.append(target)

        # Fire any registered hooks for the new state.
        for hook in self._hooks.get(target, []):
            hook(previous, target)

        return self._state

    # -- Hooks / Callbacks --------------------------------------------------

    def on_enter(
        self,
        state: InterviewState,
        callback: Callable[[InterviewState, InterviewState], None],
    ) -> None:
        """
        Register a *callback* that fires every time *state* is entered.

        The callback receives ``(previous_state, new_state)``.
        """
        self._hooks[state].append(callback)

    # -- Convenience shortcuts ----------------------------------------------

    def start_warmup(self) -> InterviewState:
        """Shortcut: INTRO → WARMUP."""
        return self.transition_to(InterviewState.WARMUP)

    def start_asking(self) -> InterviewState:
        """Shortcut: WARMUP → ASK."""
        return self.transition_to(InterviewState.ASK)

    def follow_up(self) -> InterviewState:
        """Shortcut: ASK → FOLLOWUP."""
        return self.transition_to(InterviewState.FOLLOWUP)

    def next_question(self) -> InterviewState:
        """Shortcut: FOLLOWUP → ASK (loop)."""
        return self.transition_to(InterviewState.ASK)

    def close(self) -> InterviewState:
        """Shortcut: FOLLOWUP → CLOSE."""
        return self.transition_to(InterviewState.CLOSE)

    # -- Serialisation helpers ----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the machine's snapshot to a plain dict."""
        return {
            "current_state": self._state.value,
            "history": [s.value for s in self._history],
            "questions_asked": self.questions_asked,
            "followups_asked": self.followups_asked,
            "is_terminal": self.is_terminal,
            "context": self._context,
        }

    # -- Dunder helpers -----------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"InterviewStateMachine(state={self._state.value!r}, "
            f"questions={self.questions_asked}, "
            f"followups={self.followups_asked})"
        )

    def __str__(self) -> str:
        flow = " → ".join(s.value for s in InterviewState)
        return f"[{flow}]  current={self._state.value}"
