"""
Session Manager
================

Stores interview sessions in memory, keyed by ``session_id``.

Lifecycle
---------
1. **Prepare phase** (.NET calls ``POST /api/interview/prepare``):
   - The Planner generates an ``InterviewPlan``.
   - ``create_session(plan)`` stores the plan and returns the ``session_id``.
   - Session status: ``PENDING`` (waiting for candidate to connect).

2. **Connect phase** (React opens WebSocket ``/ws/interview/{session_id}``):
   - ``activate_session(session_id)`` creates an ``InterviewOrchestrator``
     from the stored plan and marks the session as ``ACTIVE``.

3. **Interview phase** (WebSocket messages flowing):
   - ``get_orchestrator(session_id)`` returns the live orchestrator.
   - The WebSocket handler calls ``orch.start()`` and ``orch.handle_answer()``.

4. **Completion** (interview ends):
   - ``complete_session(session_id)`` marks it as ``COMPLETED`` and
     stores the final report.

5. **Cleanup**:
   - ``delete_session(session_id)`` removes all data.
   - ``cleanup_stale_sessions(max_age_hours)`` removes old sessions.

Storage
-------
In-memory ``dict`` for now. If you need persistence across restarts or
multi-worker support, swap the dict for Redis (the API stays the same).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from orchestrator import InterviewOrchestrator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session Status
# ---------------------------------------------------------------------------

class SessionStatus(str, Enum):
    """Tracks where a session is in its lifecycle."""
    PENDING   = "pending"    # Plan created, waiting for WebSocket connection
    ACTIVE    = "active"     # WebSocket connected, interview in progress
    COMPLETED = "completed"  # Interview finished, report available
    EXPIRED   = "expired"    # Session timed out without completing


# ---------------------------------------------------------------------------
# Session Data
# ---------------------------------------------------------------------------

@dataclass
class SessionData:
    """
    All data associated with a single interview session.

    Attributes
    ----------
    session_id : str
        Unique identifier (from the Planner or generated).
    plan : Any
        The InterviewPlan object from the Planner.
    status : SessionStatus
        Current lifecycle status.
    orchestrator : InterviewOrchestrator | None
        The live orchestrator (created when WebSocket connects).
    report : dict | None
        Final interview report (set when interview completes).
    created_at : datetime
        When the session was created (UTC).
    activated_at : datetime | None
        When the WebSocket connected (UTC).
    completed_at : datetime | None
        When the interview ended (UTC).
    metadata : dict
        Extra data (candidate info, frontend details, etc.).
    """
    session_id: str
    plan: Any
    status: SessionStatus = SessionStatus.PENDING
    orchestrator: Optional[InterviewOrchestrator] = None
    report: Optional[dict[str, Any]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    activated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def age_hours(self) -> float:
        """Hours since session was created."""
        delta = datetime.now(timezone.utc) - self.created_at
        return delta.total_seconds() / 3600

    @property
    def candidate_name(self) -> str:
        return getattr(self.plan, "candidate_name", "Unknown")

    @property
    def job_role(self) -> str:
        return getattr(self.plan, "job_role", "Unknown")

    def to_summary(self) -> dict[str, Any]:
        """Return a serialisable summary (safe for API responses)."""
        return {
            "session_id": self.session_id,
            "status": self.status.value,
            "candidate_name": self.candidate_name,
            "job_role": self.job_role,
            "total_questions": len(self.plan.questions) if self.plan else 0,
            "created_at": self.created_at.isoformat(),
            "activated_at": self.activated_at.isoformat() if self.activated_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "age_hours": round(self.age_hours, 2),
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SessionNotFoundError(Exception):
    """Raised when a session_id does not exist."""
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        super().__init__(f"Session not found: {session_id}")


class SessionStateError(Exception):
    """Raised when a session is in the wrong state for the requested action."""
    def __init__(self, session_id: str, expected: str, actual: str) -> None:
        self.session_id = session_id
        super().__init__(
            f"Session {session_id} is '{actual}', expected '{expected}'"
        )


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    In-memory store for interview sessions.

    Thread-safety note: This is designed for a single-process async server
    (FastAPI with uvicorn). If you scale to multiple workers, replace
    ``_sessions`` with Redis.

    Usage
    -----
    >>> manager = SessionManager()
    >>>
    >>> # 1. Planner creates the plan
    >>> sid = manager.create_session(plan)
    >>>
    >>> # 2. WebSocket connects
    >>> orch = manager.activate_session(sid)
    >>>
    >>> # 3. During interview
    >>> orch = manager.get_orchestrator(sid)
    >>> resp = await orch.handle_answer("...")
    >>>
    >>> # 4. Interview ends
    >>> manager.complete_session(sid, report={...})
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionData] = {}

    # -- Create -------------------------------------------------------------

    def create_session(self, plan: Any) -> str:
        """
        Store a new interview plan and return its session_id.

        Parameters
        ----------
        plan : InterviewPlan
            The Planner's output. Must have a ``session_id`` attribute.

        Returns
        -------
        str
            The session_id.

        Raises
        ------
        ValueError
            If a session with this ID already exists.
        """
        session_id = plan.session_id

        if session_id in self._sessions:
            raise ValueError(f"Session already exists: {session_id}")

        self._sessions[session_id] = SessionData(
            session_id=session_id,
            plan=plan,
        )

        logger.info(
            f"Session created | id={session_id} "
            f"candidate={plan.candidate_name} "
            f"questions={len(plan.questions)}"
        )

        return session_id

    # -- Activate -----------------------------------------------------------

    def activate_session(self, session_id: str) -> InterviewOrchestrator:
        """
        Create an InterviewOrchestrator for this session.

        Called when the WebSocket connects. The session must be in PENDING state.

        Parameters
        ----------
        session_id : str
            The session to activate.

        Returns
        -------
        InterviewOrchestrator
            The ready-to-use orchestrator.

        Raises
        ------
        SessionNotFoundError
            If the session_id doesn't exist.
        SessionStateError
            If the session is not in PENDING state.
        """
        session = self._get_session(session_id)

        if session.status != SessionStatus.PENDING:
            raise SessionStateError(
                session_id, "pending", session.status.value
            )

        # Create the orchestrator from the stored plan
        orchestrator = InterviewOrchestrator(plan=session.plan)

        session.orchestrator = orchestrator
        session.status = SessionStatus.ACTIVE
        session.activated_at = datetime.now(timezone.utc)

        logger.info(f"Session activated | id={session_id}")

        return orchestrator

    # -- Get orchestrator ---------------------------------------------------

    def get_orchestrator(self, session_id: str) -> InterviewOrchestrator:
        """
        Retrieve the live orchestrator for an active session.

        Parameters
        ----------
        session_id : str
            The session ID.

        Returns
        -------
        InterviewOrchestrator

        Raises
        ------
        SessionNotFoundError
            If the session_id doesn't exist.
        SessionStateError
            If the session is not ACTIVE.
        """
        session = self._get_session(session_id)

        if session.status != SessionStatus.ACTIVE:
            raise SessionStateError(
                session_id, "active", session.status.value
            )

        if session.orchestrator is None:
            raise SessionStateError(
                session_id, "active (with orchestrator)", "active (no orchestrator)"
            )

        return session.orchestrator

    # -- Complete -----------------------------------------------------------

    def complete_session(
        self, session_id: str, report: Optional[dict] = None
    ) -> None:
        """
        Mark a session as completed and store the final report.

        Parameters
        ----------
        session_id : str
            The session to complete.
        report : dict | None
            The final interview report from the orchestrator.
        """
        session = self._get_session(session_id)

        session.status = SessionStatus.COMPLETED
        session.completed_at = datetime.now(timezone.utc)
        session.report = report

        logger.info(f"Session completed | id={session_id}")

    # -- Get session info ---------------------------------------------------

    def get_session_info(self, session_id: str) -> dict[str, Any]:
        """Return a summary of the session (safe for API responses)."""
        return self._get_session(session_id).to_summary()

    def get_session_report(self, session_id: str) -> Optional[dict]:
        """Return the final report, or None if not yet complete."""
        session = self._get_session(session_id)
        return session.report

    def get_session_status(self, session_id: str) -> str:
        """Return the current status string."""
        return self._get_session(session_id).status.value

    # -- List / Query -------------------------------------------------------

    def list_sessions(
        self, status: Optional[SessionStatus] = None
    ) -> list[dict[str, Any]]:
        """
        List all sessions, optionally filtered by status.

        Returns a list of session summaries.
        """
        sessions = self._sessions.values()
        if status is not None:
            sessions = [s for s in sessions if s.status == status]
        return [s.to_summary() for s in sessions]

    @property
    def active_count(self) -> int:
        """Number of currently active interview sessions."""
        return sum(
            1 for s in self._sessions.values()
            if s.status == SessionStatus.ACTIVE
        )

    @property
    def total_count(self) -> int:
        """Total number of sessions (all statuses)."""
        return len(self._sessions)

    # -- Delete / Cleanup ---------------------------------------------------

    def delete_session(self, session_id: str) -> None:
        """Remove a session and all its data."""
        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)

        del self._sessions[session_id]
        logger.info(f"Session deleted | id={session_id}")

    def cleanup_stale_sessions(self, max_age_hours: float = 24.0) -> int:
        """
        Remove sessions older than ``max_age_hours``.

        Returns the number of sessions removed.
        """
        stale = [
            sid for sid, s in self._sessions.items()
            if s.age_hours > max_age_hours
        ]

        for sid in stale:
            del self._sessions[sid]

        if stale:
            logger.info(f"Cleaned up {len(stale)} stale sessions")

        return len(stale)

    # -- Private ------------------------------------------------------------

    def _get_session(self, session_id: str) -> SessionData:
        """Look up a session or raise SessionNotFoundError."""
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        return session

    def __repr__(self) -> str:
        return (
            f"SessionManager("
            f"total={self.total_count}, "
            f"active={self.active_count})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# Import this instance from anywhere:
#   from session import session_manager

session_manager = SessionManager()
