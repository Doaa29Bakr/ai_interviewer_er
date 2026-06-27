"""
Data Models — Pydantic
======================

Core domain models for the AI interviewer system:

- **Candidate**  : The person being interviewed.
- **Score**      : A rubric-based score for a single answer.
- **Answer**     : A candidate's response to one question (with optional score).
- **Interview**  : The top-level session tying everything together.

All models integrate with the ``InterviewState`` enum defined in
``state_machine.py``.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from state_machine import InterviewState


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DifficultyLevel(str, Enum):
    """Question / interview difficulty tier."""

    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class InterviewType(str, Enum):
    """The kind of interview being conducted."""

    TECHNICAL  = "technical"
    BEHAVIORAL = "behavioral"
    SYSTEM_DESIGN = "system_design"
    GENERAL    = "general"


# ---------------------------------------------------------------------------
# Candidate
# ---------------------------------------------------------------------------

class Candidate(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    name: str
    email: Optional[str] = None
    role: str = "software engineer"
    experience_years: int = 0
    skills: list[str] = Field(default_factory=list)
    resume_summary: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- helpers ------------------------------------------------------------

    @property
    def skill_tags(self) -> str:
        """Comma-separated skill list for prompt injection."""
        return ", ".join(self.skills) if self.skills else "not specified"

    def __str__(self) -> str:
        return f"{self.name} - {self.role} ({self.experience_years}y exp)"


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

class ScoreDimension(BaseModel):
    """A single rubric dimension (e.g. 'clarity', 'depth')."""

    name: str
    score: float = Field(..., ge=0.0, le=10.0, description="Score from 0–10")
    feedback: str = ""


class Score(BaseModel):
    """
    Evaluation result for a single answer.

    Attributes
    ----------
    overall : float
        Aggregated score (0–10).
    dimensions : list[ScoreDimension]
        Per-rubric breakdown (e.g. clarity, depth, accuracy).
    key_points_covered : list[str]
        Golden answer key points the candidate addressed.
    key_points_missed : list[str]
        Golden answer key points the candidate did NOT address.
    strengths : list[str]
        Notable strengths observed in the answer.
    improvements : list[str]
        Areas where the candidate can improve.
    needs_followup : bool
        Whether the Evaluator thinks this answer needs a follow-up probe.
    followup_reason : str
        Why the Evaluator flagged (or didn't flag) for follow-up.
    notes : str
        Free-form evaluator notes.
    """

    overall: float = Field(..., ge=0.0, le=10.0, description="Overall score 0–10")
    dimensions: list[ScoreDimension] = Field(default_factory=list)
    key_points_covered: list[str] = Field(default_factory=list)
    key_points_missed: list[str] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    needs_followup: bool = False
    followup_reason: str = ""
    notes: str = ""

    # -- helpers ------------------------------------------------------------

    @property
    def passed(self) -> bool:
        """Quick check: did the candidate score ≥ 6?"""
        return self.overall >= 6.0

    @property
    def dimension_average(self) -> float:
        """Mean of all dimension scores (returns 0.0 if none)."""
        if not self.dimensions:
            return 0.0
        return round(sum(d.score for d in self.dimensions) / len(self.dimensions), 2)

    def summary_line(self) -> str:
        """One-liner summary for logs / CLI output."""
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return f"{status}  overall={self.overall}/10  dims={self.dimension_average}/10"


# ---------------------------------------------------------------------------
# Answer
# ---------------------------------------------------------------------------

class Answer(BaseModel):

    id: str = Field(default_factory=lambda: uuid4().hex)
    question: str
    question_index: int = 1
    answer_text: str
    is_followup: bool = False
    parent_answer_id: Optional[str] = None
    golden_answer: Optional[str] = None
    score: Optional[Score] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- helpers ------------------------------------------------------------

    @property
    def is_scored(self) -> bool:
        """Return True if this answer has been graded."""
        return self.score is not None

    @property
    def word_count(self) -> int:
        """Rough word count of the candidate's response."""
        return len(self.answer_text.split())

    def __str__(self) -> str:
        kind = "Follow-up" if self.is_followup else "Question"
        scored = f" [{self.score.overall}/10]" if self.score else ""
        return f"{kind} #{self.question_index}{scored}: {self.question[:60]}..."


# ---------------------------------------------------------------------------
# Interview
# ---------------------------------------------------------------------------

class Interview(BaseModel):
    """
    Top-level session model that ties a Candidate, a sequence of Answers,
    and the current InterviewState together.

    Attributes
    ----------
    id : str
        Unique session identifier (auto-generated UUID).
    candidate : Candidate
        The candidate being interviewed.
    interview_type : InterviewType
        Category of interview (technical, behavioral, etc.).
    difficulty : DifficultyLevel
        Overall difficulty tier.
    topic : str
        The subject-matter focus (e.g. "Python backend", "leadership").
    current_state : InterviewState
        Mirror of the state machine's current state.
    answers : list[Answer]
        Ordered list of all Q&A pairs captured so far.
    max_questions : int
        The maximum number of main questions to ask before closing.
    started_at : datetime
        Session start timestamp.
    ended_at : datetime | None
        Session end timestamp (set when state reaches CLOSE).
    summary : str | None
        AI-generated wrap-up summary (populated at CLOSE).
    overall_score : Score | None
        Aggregated evaluation across all answers.
    metadata : dict[str, Any]
        Extra data (model config, session flags, etc.).
    """

    id: str = Field(default_factory=lambda: uuid4().hex)
    candidate: Candidate
    interview_type: InterviewType = InterviewType.TECHNICAL
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM
    topic: str = "general software engineering"
    current_state: InterviewState = InterviewState.INTRO
    answers: list[Answer] = Field(default_factory=list)
    max_questions: int = 5
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None
    overall_score: Optional[Score] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # -- Computed properties ------------------------------------------------

    @property
    def questions_asked(self) -> int:
        """Number of main (non-follow-up) questions answered so far."""
        return sum(1 for a in self.answers if not a.is_followup)

    @property
    def followups_asked(self) -> int:
        """Number of follow-up probes answered so far."""
        return sum(1 for a in self.answers if a.is_followup)

    @property
    def is_complete(self) -> bool:
        """True if the interview has reached the CLOSE state."""
        return self.current_state == InterviewState.CLOSE

    @property
    def has_remaining_questions(self) -> bool:
        """True if we haven't hit the question cap yet."""
        return self.questions_asked < self.max_questions

    @property
    def average_score(self) -> float | None:
        """Mean overall score across all graded answers, or None."""
        scored = [a.score.overall for a in self.answers if a.score]
        return round(sum(scored) / len(scored), 2) if scored else None

    @property
    def duration_minutes(self) -> float | None:
        """Total interview duration in minutes (None if still running)."""
        if self.ended_at is None:
            return None
        delta = self.ended_at - self.started_at
        return round(delta.total_seconds() / 60, 1)

    # -- Mutators -----------------------------------------------------------

    def add_answer(self, answer: Answer) -> None:
        """Append an answer and auto-set its question_index."""
        answer.question_index = len(self.answers) + 1
        self.answers.append(answer)

    def close_interview(self, summary: str | None = None) -> None:
        """Mark the interview as closed."""
        self.current_state = InterviewState.CLOSE
        self.ended_at = datetime.utcnow()
        if summary:
            self.summary = summary

    # -- Serialisation ------------------------------------------------------

    def to_report(self) -> dict[str, Any]:
        """Build a lightweight report dict suitable for JSON export."""
        return {
            "interview_id": self.id,
            "candidate": self.candidate.name,
            "role": self.candidate.role,
            "type": self.interview_type.value,
            "difficulty": self.difficulty.value,
            "topic": self.topic,
            "questions_asked": self.questions_asked,
            "followups_asked": self.followups_asked,
            "average_score": self.average_score,
            "duration_minutes": self.duration_minutes,
            "is_complete": self.is_complete,
            "summary": self.summary,
        }

    # -- Dunder helpers -----------------------------------------------------

    def __str__(self) -> str:
        return (
            f"Interview({self.candidate.name} | {self.interview_type.value} | "
            f"state={self.current_state.value} | "
            f"Q={self.questions_asked}/{self.max_questions})"
        )
