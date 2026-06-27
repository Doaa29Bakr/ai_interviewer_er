"""
Interview Orchestrator
======================

The core engine that drives the interview loop. It wraps:

- ``InterviewStateMachine``  (state transitions)
- ``ConversationHistory``    (multi-turn context for Groq)
- Groq LLM calls            (interviewer only)

The orchestrator is **session-scoped**: one instance per interview session.
The WebSocket handler creates it, then calls ``start()`` once and
``handle_answer()`` for every candidate response.

Data contract
-------------
Input:  An ``InterviewPlan`` from the Planner (questions + golden answers).
Output: ``OrchestratorResponse`` objects the WebSocket handler forwards to
        the frontend (text for TTS, state info, metadata).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from config import get_key, get_key_float, get_key_int

from groq import Groq

from state_machine import InterviewStateMachine, InterviewState
from models import (
    Candidate,
    Interview,
    Answer,
    DifficultyLevel,
)
from conversation import (
    ConversationHistory,
)
import random
from intent_classifier import classify_intent

logger = logging.getLogger(__name__)

WARMUP_QUESTIONS = [
    "Tell me about yourself.",
    "Can you describe your previous experience?",
    "What projects are you most proud of?",
    "Why are you interested in this role?"
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GROQ_CHAT_MODEL = get_key("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
GROQ_CHAT_TEMPERATURE = get_key_float("GROQ_CHAT_TEMPERATURE", "0.5")
GROQ_MAX_TOKENS = get_key_int("GROQ_MAX_TOKENS", "1024")

# Level -> DifficultyLevel mapping for the planner's level field
LEVEL_TO_DIFFICULTY = {
    "Junior": "easy",
    "Mid": "medium",
    "Senior": "hard",
}


# ---------------------------------------------------------------------------
# Response object (what the WebSocket handler receives)
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorResponse:
    text: str
    state: str
    is_complete: bool = False
    question_index: int = 0
    total_questions: int = 0
    evaluation: Optional[dict[str, Any]] = None
    followup_triggered: bool = False
    is_clarification: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for WebSocket JSON transmission."""
        return {
            "text": self.text,
            "state": self.state,
            "is_complete": self.is_complete,
            "question_index": self.question_index,
            "total_questions": self.total_questions,
            "evaluation": self.evaluation,
            "followup_triggered": self.followup_triggered,
            "is_clarification": self.is_clarification,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class InterviewOrchestrator:
    def __init__(self, plan: Any) -> None:
        self._plan = plan
        self._session_id = plan.session_id

        # -- Extract candidate info from the plan -------------------------
        self._candidate = Candidate(
            name=plan.candidate_name,
            role=plan.job_role,
            skills=[],
        )

        # -- Map planner level to difficulty ------------------------------
        difficulty_str = LEVEL_TO_DIFFICULTY.get(plan.level, "medium")

        # -- Build the questions list (Planner format -> Interviewer format)
        self._questions: list[dict[str, Any]] = []
        for q in plan.questions:
            self._questions.append({
                "question_id": getattr(q, "question_id", ""),
                "question_text": q.question,
                "golden_answer": q.golden_answer,
                "difficulty": difficulty_str,
                "topic": q.skill,
                "time_limit_seconds": getattr(q, "time_limit_seconds", 120),
            })

        # Default time limit for non-technical phases (warmup, intro)
        self._default_time_limit: int = 120

        # -- Initialize components ----------------------------------------
        self._sm = InterviewStateMachine()
        self._history = ConversationHistory()
        self._interview = Interview(
            candidate=self._candidate,
            topic=plan.job_role,
            max_questions=len(self._questions),
            difficulty=DifficultyLevel(difficulty_str),
        )

        # -- Tracking state -----------------------------------------------
        self._current_q_index: int = 0
        self._current_question: Optional[dict] = None
        self._last_candidate_answer: str = ""
        self._started_at: datetime = datetime.now(timezone.utc)
        
        # Transcript format to be saved to Redis directly
        self._transcript: list[dict] = []

        # -- Groq client --------------------------------------------------
        self._client = Groq(api_key=get_key("GROQ_API_KEY"))

        logger.info(
            f"Orchestrator initialized | session={self._session_id} "
            f"candidate={plan.candidate_name} questions={len(self._questions)}"
        )

    # ===================================================================
    #  PUBLIC API
    # ===================================================================

    async def start(self) -> OrchestratorResponse:
        logger.info(f"[{self._session_id}] Starting interview (INTRO)")

        self._history.add_state_prompt(
            "INTRO",
            candidate_name=self._candidate.name,
            candidate_role=self._candidate.role,
            experience_years=self._candidate.experience_years,
            skills=self._candidate.skill_tags,
            interview_type=self._interview.interview_type.value,
            topic=self._interview.topic,
            max_questions=self._interview.max_questions,
            duration_limit=getattr(self._plan, "duration_limit", 30),
        )

        intro_text = await self._call_interviewer()
        
        self._transcript.append({
            "type_of_question": "intro",
            "interviewer": intro_text
        })

        return OrchestratorResponse(
            text=intro_text,
            state=self._sm.state.value,
            question_index=0,
            total_questions=len(self._questions),
        )

    async def handle_answer(self, candidate_answer: str) -> OrchestratorResponse:
        self._last_candidate_answer = candidate_answer
        
        if self._transcript:
            self._transcript[-1]["candidate"] = candidate_answer

        self._history.add_candidate_answer(candidate_answer)

        current_state = self._sm.state

        if current_state == InterviewState.INTRO:
            return await self._transition_to_warmup()
            
        if current_state == InterviewState.WARMUP:
            return await self._handle_warmup_answer(candidate_answer)

        if current_state == InterviewState.ASK:
            return await self._handle_main_answer(candidate_answer)

        # Should not reach here
        logger.warning(f"[{self._session_id}] Unexpected state: {current_state}")
        return OrchestratorResponse(
            text="Something went wrong. Let me wrap up.",
            state=current_state.value,
            is_complete=True,
        )

    async def force_timeout(self) -> OrchestratorResponse:
        """Forcefully transition to CLOSE due to timeout."""
        self._sm._state = InterviewState.CLOSE
        self._sm._history.append(InterviewState.CLOSE)
        self._interview.current_state = self._sm.state

        self._history.add_state_prompt("TIMEOUT_CLOSE")
        close_text = await self._call_interviewer()
        
        self._transcript.append({
            "type_of_question": "close",
            "interviewer": close_text
        })
        
        self._interview.close_interview(summary=close_text)

        logger.info(f"[{self._session_id}] TIMEOUT CLOSE triggered")

        return OrchestratorResponse(
            text=close_text,
            state=self._sm.state.value,
            is_complete=True,
            question_index=0,
            total_questions=len(self._questions),
            metadata={"report": self.get_report()},
        )

    async def force_early_close(self) -> None:
        """Forcefully transition to CLOSE and save data without generating a spoken response."""
        if self._sm.state != InterviewState.CLOSE:
            self._sm._state = InterviewState.CLOSE
            self._sm._history.append(InterviewState.CLOSE)
            self._interview.current_state = self._sm.state

        self._interview.close_interview(summary="Interview was ended early by the candidate.")
        logger.info(f"[{self._session_id}] EARLY CLOSE triggered")

    def get_report(self) -> dict[str, Any]:
        """
        Return the final transcript report to be saved in Redis.
        """
        return {
            "session_id":     self._session_id,
            "candidate_name": self._plan.candidate_name,
            "job_role":       self._plan.job_role,
            "level":          self._plan.level,
            "transcript":     self._transcript
        }

    @property
    def is_complete(self) -> bool:
        return self._sm.is_terminal

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def current_state(self) -> str:
        return self._sm.state.value

    @property
    def current_question_time_limit(self) -> int:
        """Return the time limit (in seconds) for the current question."""
        if self._current_question:
            return self._current_question.get("time_limit_seconds", self._default_time_limit)
        return self._default_time_limit

    # ===================================================================
    #  CLARIFICATION HANDLING
    # ===================================================================

    async def get_candidate_intent(self, candidate_text: str) -> str:
        """
        Detect the candidate's intent using the LLM classifier.
        """
        if self._sm.state not in (InterviewState.ASK, InterviewState.FOLLOWUP):
            return "technical_answer"

        # Build conversation history string
        # We only need the last few exchanges to give context
        messages = self._history.get_messages()
        history_str = ""
        for msg in messages[-6:]:  # Last 3 turns
            if msg["role"] in ("assistant", "user"):
                speaker = "Interviewer" if msg["role"] == "assistant" else "Candidate"
                history_str += f"{speaker}: {msg['content']}\n"

        q = self._current_question or {}
        question_text = q.get("question_text", "the current question")

        result = await classify_intent(
            conversation_history=history_str.strip(),
            current_question=question_text,
            candidate_message=candidate_text,
        )

        return result.get("intent", "unknown")

    async def handle_clarification(self, candidate_text: str) -> OrchestratorResponse:
        """
        Respond to a clarification request WITHOUT advancing the state machine.

        The interviewer rephrases / elaborates on the current question.
        """
        q = self._current_question or {}
        question_text = q.get("question_text", "the current question")

        self._history.add_candidate_answer(candidate_text)
        self._history.add_state_prompt(
            "CLARIFY",
            question_text=question_text,
            candidate_answer=candidate_text,
        )

        clarify_text = await self._call_interviewer()

        logger.info(
            f"[{self._session_id}] CLARIFICATION requested — "
            f"candidate said: {candidate_text[:60]}..."
        )

        return OrchestratorResponse(
            text=clarify_text,
            state=self._sm.state.value,
            is_clarification=True,
            question_index=self._current_q_index + 1,
            total_questions=len(self._questions),
        )

    async def handle_answer_seeking(self, candidate_text: str) -> OrchestratorResponse:
        """
        Respond to an answer seeking request WITHOUT advancing the state machine.

        The interviewer refuses to give the answer and encourages the candidate.
        """
        q = self._current_question or {}
        question_text = q.get("question_text", "the current question")

        self._history.add_candidate_answer(candidate_text)
        self._history.add_state_prompt(
            "ANSWER_SEEKING",
            question_text=question_text,
            candidate_answer=candidate_text,
        )

        response_text = await self._call_interviewer()

        logger.info(
            f"[{self._session_id}] ANSWER_SEEKING requested — "
            f"candidate said: {candidate_text[:60]}..."
        )

        return OrchestratorResponse(
            text=response_text,
            state=self._sm.state.value,
            is_clarification=True,
            question_index=self._current_q_index + 1,
            total_questions=len(self._questions),
        )

    async def handle_off_topic(self, candidate_text: str) -> OrchestratorResponse:
        """
        Respond to an off-topic message WITHOUT advancing the state machine.

        The interviewer politely redirects the candidate back to the current question.
        """
        q = self._current_question or {}
        question_text = q.get("question_text", "the current question")

        self._history.add_candidate_answer(candidate_text)
        self._history.add_state_prompt(
            "OFF_TOPIC",
            question_text=question_text,
            candidate_answer=candidate_text,
        )

        response_text = await self._call_interviewer()

        logger.info(
            f"[{self._session_id}] OFF_TOPIC detected — "
            f"candidate said: {candidate_text[:60]}..."
        )

        return OrchestratorResponse(
            text=response_text,
            state=self._sm.state.value,
            is_clarification=True,
            question_index=self._current_q_index + 1,
            total_questions=len(self._questions),
        )

    # ===================================================================
    #  PRIVATE — State Handlers
    # ===================================================================

    async def _transition_to_warmup(self) -> OrchestratorResponse:
        """Transition to WARMUP state and pose the warmup question."""
        self._sm.start_warmup()
        self._interview.current_state = self._sm.state

        question_text = random.choice(WARMUP_QUESTIONS)
        self._history.add_state_prompt("WARMUP", question_text=question_text)
        warmup_text = await self._call_interviewer()

        self._current_question = {
            "question_text": question_text,
            "golden_answer": None,
            "topic": "warmup",
            "difficulty": "easy"
        }
        
        self._transcript.append({
            "type_of_question": "warmup",
            "interviewer": warmup_text
        })

        logger.info(f"[{self._session_id}] WARMUP Q: {question_text}")

        return OrchestratorResponse(
            text=warmup_text,
            state=self._sm.state.value,
            question_index=0,
            total_questions=len(self._questions),
        )

    async def _handle_warmup_answer(self, candidate_answer: str) -> OrchestratorResponse:
        q = self._current_question
        answer = Answer(
            question=q["question_text"],
            answer_text=candidate_answer,
            golden_answer=None,
            is_followup=False,
            score=None,
        )
        self._interview.add_answer(answer)
        return await self._transition_to_ask()

    async def _transition_to_ask(self) -> OrchestratorResponse:
        self._sm.start_asking() if self._sm.state == InterviewState.WARMUP else self._sm.next_question()
        self._interview.current_state = self._sm.state

        self._current_question = self._questions[self._current_q_index]
        q = self._current_question
        previous = self._interview.answers[-1].answer_text if self._interview.answers else "(first question)"

        self._history.add_state_prompt(
            "ASK",
            question_text=q["question_text"],
            question_index=self._current_q_index + 1,
            max_questions=len(self._questions),
            question_topic=q["topic"],
            question_difficulty=q["difficulty"],
            previous_answer=previous,
        )

        ask_text = await self._call_interviewer()
        
        self._transcript.append({
            "type_of_question": "technical",
            "core_question": q["question_text"],
            "golden_answer": q.get("golden_answer"),
            "interviewer": ask_text
        })

        logger.info(f"[{self._session_id}] ASK Q#{self._current_q_index + 1}: {q['question_text'][:50]}...")

        return OrchestratorResponse(
            text=ask_text,
            state=self._sm.state.value,
            question_index=self._current_q_index + 1,
            total_questions=len(self._questions),
        )

    async def _handle_main_answer(self, candidate_answer: str) -> OrchestratorResponse:
        q = self._current_question
        answer = Answer(
            question=q["question_text"],
            answer_text=candidate_answer,
            golden_answer=q.get("golden_answer"),
            is_followup=False,
            score=None,
        )
        self._interview.add_answer(answer)
        return await self._advance_to_next_or_close()

    async def _advance_to_next_or_close(self) -> OrchestratorResponse:
        if self._sm.state == InterviewState.ASK:
            self._sm.follow_up()

        self._current_q_index += 1

        if self._current_q_index < len(self._questions):
            return await self._transition_to_ask_from_followup()
        else:
            return await self._do_close()

    async def _transition_to_ask_from_followup(self) -> OrchestratorResponse:
        self._sm.next_question()
        self._interview.current_state = self._sm.state

        self._current_question = self._questions[self._current_q_index]
        q = self._current_question
        previous = self._interview.answers[-1].answer_text if self._interview.answers else "(first question)"

        self._history.add_state_prompt(
            "ASK",
            question_text=q["question_text"],
            question_index=self._current_q_index + 1,
            max_questions=len(self._questions),
            question_topic=q["topic"],
            question_difficulty=q["difficulty"],
            previous_answer=previous,
        )

        ask_text = await self._call_interviewer()
        
        self._transcript.append({
            "type_of_question": "technical",
            "core_question": q["question_text"],
            "golden_answer": q.get("golden_answer"),
            "interviewer": ask_text
        })

        logger.info(f"[{self._session_id}] ASK Q#{self._current_q_index + 1}: {q['question_text'][:50]}...")

        return OrchestratorResponse(
            text=ask_text,
            state=self._sm.state.value,
            question_index=self._current_q_index + 1,
            total_questions=len(self._questions),
        )

    async def _do_close(self) -> OrchestratorResponse:
        self._sm.close()
        self._interview.current_state = self._sm.state

        duration = round((datetime.now(timezone.utc) - self._started_at).total_seconds() / 60, 1)

        self._history.add_state_prompt(
            "CLOSE",
            candidate_name=self._candidate.name,
            questions_asked=self._interview.questions_asked,
            followups_asked=self._interview.followups_asked,
            duration=str(duration),
        )

        close_text = await self._call_interviewer()
        
        self._transcript.append({
            "type_of_question": "close",
            "interviewer": close_text
        })

        self._interview.close_interview(summary=close_text)
        logger.info(f"[{self._session_id}] CLOSE")

        return OrchestratorResponse(
            text=close_text,
            state=self._sm.state.value,
            is_complete=True,
            question_index=0,
            total_questions=len(self._questions),
            metadata={"report": self.get_report()},
        )

    # ===================================================================
    #  PRIVATE — LLM Calls
    # ===================================================================

    async def _call_interviewer(self) -> str:
        messages = self._history.get_messages()

        try:
            response = self._client.chat.completions.create(
                model=GROQ_CHAT_MODEL,
                messages=messages,
                temperature=GROQ_CHAT_TEMPERATURE,
                max_tokens=GROQ_MAX_TOKENS,
            )
            text = response.choices[0].message.content.strip()
            self._history.add_assistant_message(text)
            return text

        except Exception as exc:
            logger.error(f"[{self._session_id}] Interviewer LLM call failed: {exc}")
            fallback = "I appreciate your response. Let me move on to the next point."
            self._history.add_assistant_message(fallback)
            return fallback

    def __repr__(self) -> str:
        return (
            f"InterviewOrchestrator("
            f"session={self._session_id}, "
            f"state={self._sm.state.value}, "
            f"q={self._current_q_index + 1}/{len(self._questions)})"
        )
