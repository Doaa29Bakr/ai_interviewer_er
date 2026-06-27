"""
Conversation History Manager
=============================

Manages the multi-turn ``messages[]`` array sent to the Groq API (Llama).

The Interviewer agent needs **full context** of every previous exchange
so it can:
- Reference what the candidate said earlier
- Avoid repeating questions
- Build natural transitions between topics
- Generate context-aware follow-ups

This module provides:

1. ``ConversationHistory``  — Builds & maintains the Groq ``messages[]`` array.
2. ``FollowUpDecision``     — Decides whether a candidate answer needs probing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from prompts import (
    EVALUATOR_SYSTEM_PROMPT,
    INTERVIEWER_SYSTEM_PROMPT,
    build_interviewer_user_prompt,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Message types (Groq / OpenAI format)
# ═══════════════════════════════════════════════════════════════════════════

class Role(str, Enum):
    """Chat-completion message roles."""
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in the conversation."""
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to the ``{"role": ..., "content": ...}`` dict Groq expects."""
        return {"role": self.role.value, "content": self.content}


# ═══════════════════════════════════════════════════════════════════════════
#  1.  CONVERSATION  HISTORY  MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class ConversationHistory:
    """
    Builds and maintains the full ``messages[]`` array for Groq API calls.

    Lifecycle
    ---------
    1. ``__init__``  → Sets the system prompt.
    2. ``add_user_message``  → Appends a user-role turn (state instruction / candidate answer).
    3. ``add_assistant_message``  → Appends the interviewer's response.
    4. ``get_messages``  → Returns the full ``messages[]`` ready for Groq.

    The system prompt is always ``messages[0]``, followed by alternating
    user / assistant turns — exactly what Llama expects.

    Example
    -------
    >>> history = ConversationHistory(system_prompt=INTERVIEWER_SYSTEM_PROMPT)
    >>>
    >>> # INTRO state — first turn
    >>> intro_prompt = build_interviewer_user_prompt("INTRO", candidate_name="Sara", ...)
    >>> history.add_user_message(intro_prompt)
    >>> messages = history.get_messages()   # → send to Groq
    >>> intro_response = call_groq(messages)
    >>> history.add_assistant_message(intro_response)
    >>>
    >>> # ASK state — second turn (history carries forward)
    >>> ask_prompt = build_interviewer_user_prompt("ASK", question_text="...", ...)
    >>> history.add_user_message(ask_prompt)
    >>> messages = history.get_messages()   # includes INTRO context!
    >>> ask_response = call_groq(messages)
    >>> history.add_assistant_message(ask_response)
    """

    def __init__(self, system_prompt: str = INTERVIEWER_SYSTEM_PROMPT) -> None:
        self._system = Message(role=Role.SYSTEM, content=system_prompt)
        self._turns: list[Message] = []

    # -- Core API -----------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        """Append a user-role message (state instruction or candidate answer)."""
        self._turns.append(Message(role=Role.USER, content=content))

    def add_interviewer_directive(self, directive: str) -> None:
        """
        Inject a brief instruction for the interviewer.
        Wrapped in [DIRECTIVE] tags so the LLM knows it's not the candidate.
        """
        self._turns.append(Message(
            role=Role.USER,
            content=f"[INTERVIEWER DIRECTIVE]\n{directive}",
        ))

    def add_assistant_message(self, content: str) -> None:
        """Append the interviewer's (assistant) response."""
        self._turns.append(Message(role=Role.ASSISTANT, content=content))

    def get_messages(self) -> list[dict[str, str]]:
        """
        Return the full ``messages[]`` array for the Groq API.
        Automatically trims history if it exceeds 12 turns.
        """
        # Auto-compress context if it gets too long to avoid LLM confusion
        if len(self._turns) > 12:
            self.trim_to_last_n_turns(12)
            
        return [self._system.to_dict()] + [m.to_dict() for m in self._turns]

    # -- Candidate answer injection -----------------------------------------

    def add_candidate_answer(self, candidate_answer: str) -> None:
        """
        Inject the candidate's raw answer as a user-role message.

        This is the candidate speaking — separate from the state-instruction
        messages that drive the interviewer.
        """
        self._turns.append(Message(
            role=Role.USER,
            content=candidate_answer,
        ))

    # -- State-aware prompt injection ---------------------------------------

    def add_state_prompt(self, state: str, **kwargs: Any) -> None:
        """
        Build the user prompt for the given interview state and append it.

        Combines ``build_interviewer_user_prompt`` with ``add_user_message``
        in one call.

        Parameters
        ----------
        state : str
            One of "INTRO", "ASK", "FOLLOWUP", "CLOSE".
        **kwargs
            Template variables (candidate_name, question_text, etc.).
        """
        prompt = build_interviewer_user_prompt(state, **kwargs)
        self.add_interviewer_directive(prompt)

    # -- Context window management ------------------------------------------

    @property
    def turn_count(self) -> int:
        """Total number of turns (user + assistant), excluding the system prompt."""
        return len(self._turns)

    @property
    def total_chars(self) -> int:
        """Approximate character count across all messages (for token estimation)."""
        system_chars = len(self._system.content)
        turn_chars = sum(len(m.content) for m in self._turns)
        return system_chars + turn_chars

    @property
    def estimated_tokens(self) -> int:
        """Rough token estimate (1 token ≈ 4 chars for English text)."""
        return self.total_chars // 4

    def trim_to_last_n_turns(self, n: int) -> None:
        """
        Keep only the last *n* turns to stay within the context window.

        The system prompt is always preserved. Useful if the interview
        gets very long and approaches the Llama context limit.

        Parameters
        ----------
        n : int
            Number of recent turns to retain.
        """
        if len(self._turns) > n:
            self._turns = self._turns[-n:]

    def get_summary_of_trimmed(self) -> str:
        """
        Generate a text summary of the conversation so far.

        Useful for injecting into the system prompt or a user message
        when you need to trim history but keep context.
        """
        lines: list[str] = []
        for i, msg in enumerate(self._turns):
            role_label = "Interviewer" if msg.role == Role.ASSISTANT else "Candidate/System"
            # Truncate each turn to first 200 chars for the summary
            snippet = msg.content[:200].replace("\n", " ")
            if len(msg.content) > 200:
                snippet += "…"
            lines.append(f"Turn {i+1} ({role_label}): {snippet}")
        return "\n".join(lines)

    # -- Reset / Clear ------------------------------------------------------

    def reset(self, new_system_prompt: Optional[str] = None) -> None:
        """Clear all turns. Optionally replace the system prompt."""
        self._turns.clear()
        if new_system_prompt is not None:
            self._system = Message(role=Role.SYSTEM, content=new_system_prompt)

    # -- Serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise the full conversation state."""
        return {
            "system_prompt": self._system.content,
            "turns": [m.to_dict() for m in self._turns],
            "turn_count": self.turn_count,
            "estimated_tokens": self.estimated_tokens,
        }

    def __repr__(self) -> str:
        return (
            f"ConversationHistory(turns={self.turn_count}, "
            f"~{self.estimated_tokens} tokens)"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  2.  FOLLOW-UP  DECISION  LOGIC
# ═══════════════════════════════════════════════════════════════════════════

# -- Thresholds & config ---------------------------------------------------

# If the evaluator's overall score is below this, trigger a follow-up.
FOLLOWUP_SCORE_THRESHOLD: float = 5.0

# If the candidate's answer has fewer words than this, it's too short → follow-up.
FOLLOWUP_MIN_WORDS: int = 15

# Maximum follow-ups allowed per question (to prevent infinite loops).
MAX_FOLLOWUPS_PER_QUESTION: int = 2


@dataclass
class FollowUpDecision:
    """
    The result of checking whether the candidate's answer needs a follow-up.

    Attributes
    ----------
    needs_followup : bool
        True if the system should transition to FOLLOWUP instead of next ASK.
    reason : str
        Human-readable explanation of why (for logging / debugging).
    trigger : str
        Which check triggered the follow-up ("score", "too_short", "vague", "none").
    """
    needs_followup: bool
    reason: str
    trigger: str  # "score" | "too_short" | "vague" | "none"


def decide_followup(
    candidate_answer: str,
    evaluator_score: Optional[float] = None,
    followups_so_far: int = 0,
    score_threshold: float = FOLLOWUP_SCORE_THRESHOLD,
    min_words: int = FOLLOWUP_MIN_WORDS,
    max_followups: int = MAX_FOLLOWUPS_PER_QUESTION,
) -> FollowUpDecision:
    """
    Decide whether a candidate's answer needs a follow-up probe.

    The decision is based on three checks (in priority order):

    1. **Max follow-ups reached** → stop (prevent infinite loops).
    2. **Answer too short** → follow-up (likely vague / incomplete).
    3. **Evaluator score too low** → follow-up (substance is lacking).

    Parameters
    ----------
    candidate_answer : str
        The candidate's raw response text.
    evaluator_score : float | None
        The Evaluator's overall score (0–10). If None, only rule-based
        checks are used (word count, etc.).
    followups_so_far : int
        How many follow-ups have already been asked for this question.
    score_threshold : float
        Score below which a follow-up is triggered (default: 5.0).
    min_words : int
        Minimum word count — answers shorter than this trigger follow-up.
    max_followups : int
        Maximum follow-ups per question before moving on.

    Returns
    -------
    FollowUpDecision
        Whether to follow up, and why.

    Examples
    --------
    >>> decide_followup("Yes.", evaluator_score=3.0)
    FollowUpDecision(needs_followup=True, reason='...', trigger='too_short')

    >>> decide_followup("A detailed multi-paragraph answer...", evaluator_score=7.5)
    FollowUpDecision(needs_followup=False, reason='...', trigger='none')
    """
    word_count = len(candidate_answer.strip().split())

    # ── Check 1: Max follow-ups reached — STOP ──────────────────────────
    if followups_so_far >= max_followups:
        return FollowUpDecision(
            needs_followup=False,
            reason=(
                f"Already asked {followups_so_far} follow-up(s) for this question "
                f"(max: {max_followups}). Moving to next question."
            ),
            trigger="none",
        )

    # ── Check 2: Answer too short — FOLLOW UP ───────────────────────────
    if word_count < min_words:
        return FollowUpDecision(
            needs_followup=True,
            reason=(
                f"Answer is only {word_count} words (minimum: {min_words}). "
                f"The response is too brief — probing for more detail."
            ),
            trigger="too_short",
        )

    # ── Check 3: Evaluator score too low — FOLLOW UP ────────────────────
    if evaluator_score is not None and evaluator_score < score_threshold:
        return FollowUpDecision(
            needs_followup=True,
            reason=(
                f"Evaluator score is {evaluator_score}/10 "
                f"(threshold: {score_threshold}). "
                f"The answer lacks substance — probing deeper."
            ),
            trigger="score",
        )

    # ── All checks passed — NO FOLLOW UP ────────────────────────────────
    return FollowUpDecision(
        needs_followup=False,
        reason=(
            f"Answer is {word_count} words"
            + (f", score {evaluator_score}/10" if evaluator_score is not None else "")
            + ". No follow-up needed."
        ),
        trigger="none",
    )


# ═══════════════════════════════════════════════════════════════════════════
#  3.  EVALUATOR  CONVERSATION  (separate context — no history bleed)
# ═══════════════════════════════════════════════════════════════════════════

def build_evaluator_messages(
    question_text: str,
    candidate_answer: str,
    golden_answer: str,
    difficulty: str = "medium",
    topic: str = "general",
) -> list[dict[str, str]]:
    """
    Build a fresh ``messages[]`` array for a single Evaluator call.

    The Evaluator is **stateless** — it does NOT carry conversation history.
    Each evaluation is an independent call with:
    - System prompt (grading rules)
    - User prompt (question + candidate answer + golden answer)

    Parameters
    ----------
    question_text : str
        The interview question.
    candidate_answer : str
        What the candidate said.
    golden_answer : str
        The ideal answer from the Planner's JSON.
    difficulty : str
        Question difficulty level.
    topic : str
        Topic area.

    Returns
    -------
    list[dict[str, str]]
        Ready-to-send ``messages[]`` for Groq.
    """
    from prompts import build_evaluator_user_prompt

    user_prompt = build_evaluator_user_prompt(
        question_text=question_text,
        candidate_answer=candidate_answer,
        golden_answer=golden_answer,
        difficulty=difficulty,
        topic=topic,
    )

    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user",   "content": user_prompt},
    ]
