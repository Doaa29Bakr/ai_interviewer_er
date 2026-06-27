"""
Evaluator Pipeline
==================

Orchestrates the full post-interview evaluation flow:

1. Connect to Redis and fetch the session transcript (key: interview_report:{session_id})
2. Extract only the "technical" question entries from the transcript
3. Validate that each technical entry has a candidate answer and a golden answer
4. Call evaluate_answer() for each valid technical Q&A pair
5. Aggregate all results into a structured evaluation report

Usage
-----
    from evaluator.pipeline import run_evaluation

    report = run_evaluation("session-uuid-here")
"""

import json
import logging
import sys
import os
from typing import Any

# Allow importing config.py from the parent project directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import redis
from config import get_key
from evaluator.agent import evaluate_answer, generate_overall_summary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------

def _get_redis_client() -> redis.Redis:
    """Build and return a Redis client using the project's config."""
    url = get_key("reddis_url") or get_key("REDIS_URL")
    if not url:
        raise RuntimeError(
            "No Redis URL found. "
            "Set 'REDIS_URL' in api_keys.json or as an environment variable."
        )
    return redis.from_url(url, decode_responses=True)


def _fetch_transcript(session_id: str) -> tuple[list[dict], dict]:
    """
    Fetch the session transcript and metadata from Redis.

    The orchestrator saves the report under the key:
        interview_report:{session_id}

    The report schema is:
        {
            "session_id":     "...",
            "candidate_name": "...",
            "job_role":       "...",
            "level":          "...",
            "transcript":     [ {...}, {...}, ... ]
        }

    Returns
    -------
    tuple[list[dict], dict]
        (transcript list, metadata dict with candidate_name/job_role/level).

    Raises
    ------
    KeyError
        If the session is not found in Redis.
    ValueError
        If the stored data is malformed.
    """
    client = _get_redis_client()
    redis_key = f"interview_report:{session_id}"

    raw = client.get(redis_key)
    if raw is None:
        raise KeyError(
            f"No transcript found in Redis for session '{session_id}'. "
            "The interview may not have finished yet, or the key has expired."
        )

    try:
        report = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in Redis for key '{redis_key}': {exc}") from exc

    transcript = report.get("transcript")
    if not isinstance(transcript, list):
        raise ValueError(
            f"Expected 'transcript' to be a list in the stored report, "
            f"got {type(transcript).__name__}."
        )

    metadata = {
        "candidate_name": report.get("candidate_name", "Unknown"),
        "job_role":       report.get("job_role", "Unknown"),
        "level":          report.get("level", "Unknown"),
    }

    return transcript, metadata


# ---------------------------------------------------------------------------
# Transcript filtering
# ---------------------------------------------------------------------------

def _extract_technical_questions(transcript: list[dict]) -> list[dict]:
    """
    Filter the transcript to only technical question entries.

    A valid technical entry must have:
    - type_of_question == "technical"
    - core_question       (the actual question text)
    - golden_answer       (the reference answer; must not be None/empty)
    - candidate           (the candidate's spoken answer; must not be None/empty)

    Entries missing golden_answer or candidate answer are skipped with a warning.

    Returns
    -------
    list[dict]
        List of valid technical entries ready for evaluation.
    """
    technical = []

    for i, entry in enumerate(transcript):
        if entry.get("type_of_question") != "technical":
            continue

        question = entry.get("core_question", "").strip()
        golden   = entry.get("golden_answer", "")
        candidate = entry.get("candidate", "")

        if not question:
            logger.warning(f"Transcript entry #{i}: skipping technical entry with no question text.")
            continue

        if not golden:
            logger.warning(
                f"Transcript entry #{i} ('{question[:50]}...'): "
                "skipping — no golden_answer available."
            )
            continue

        if not candidate:
            logger.warning(
                f"Transcript entry #{i} ('{question[:50]}...'): "
                "skipping — candidate did not answer (empty transcript)."
            )
            continue

        technical.append({
            "question":       question,
            "golden_answer":  golden,
            "candidate_answer": candidate,
        })

    return technical


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_evaluation(session_id: str) -> dict[str, Any]:
    """
    Run the full evaluation pipeline for a completed interview session.

    Steps
    -----
    1. Fetch transcript from Redis
    2. Extract valid technical Q&A pairs
    3. Evaluate each pair with the LLM evaluator agent
    4. Return aggregated report as a dict (ready to serialize as JSON)

    Parameters
    ----------
    session_id : str
        The interview session ID used as the Redis key.

    Returns
    -------
    dict with keys:
        session_id         : str
        total_questions    : int   — number of technical questions in transcript
        evaluated_count    : int   — number successfully evaluated
        skipped_count      : int   — number skipped (missing data)
        average_score      : float | None
        evaluations        : list[dict]  — one result dict per evaluated question
        errors             : list[str]   — any per-question error messages

    Raises
    ------
    KeyError   — session not found in Redis
    ValueError — malformed stored data
    RuntimeError — Redis / Groq connection failure
    """
    logger.info(f"[Evaluator] Starting evaluation for session: {session_id}")

    # Step 1 — Fetch transcript and metadata
    transcript, meta = _fetch_transcript(session_id)
    candidate_name = meta["candidate_name"]
    job_role       = meta["job_role"]
    level          = meta["level"]
    logger.info(f"[Evaluator] Fetched transcript with {len(transcript)} total entries.")

    # Step 2 — Filter to technical questions only
    technical_qs = _extract_technical_questions(transcript)
    total_technical = len(technical_qs)
    skipped_count = sum(
        1 for e in transcript
        if e.get("type_of_question") == "technical"
    ) - total_technical

    logger.info(
        f"[Evaluator] Found {total_technical} evaluatable technical questions "
        f"({skipped_count} skipped due to missing data)."
    )

    if total_technical == 0:
        logger.warning(f"[Evaluator] No technical questions to evaluate for session {session_id}.")
        return {
            "session_id":      session_id,
            "candidate_name":  candidate_name,
            "job_role":        job_role,
            "level":           level,
            "average_score":   None,
            "overall_summary": None,
            "evaluations":     [],
        }

    # Step 3 — Evaluate each question
    evaluations: list[dict] = []
    errors: list[str] = []

    for idx, item in enumerate(technical_qs, start=1):
        logger.info(
            f"[Evaluator] Evaluating Q{idx}/{total_technical}: "
            f"{item['question'][:60]}..."
        )
        try:
            result = evaluate_answer(
                question=item["question"],
                golden_answer=item["golden_answer"],
                candidate_answer=item["candidate_answer"],
            )
            evaluations.append(result)
        except Exception as exc:
            msg = f"Q{idx} evaluation failed: {exc}"
            logger.error(f"[Evaluator] {msg}")
            errors.append(msg)

    # Step 4 — Aggregate
    scores = [e.get("score", 0) for e in evaluations if isinstance(e.get("score"), (int, float))]
    average_score = round(sum(scores) / len(scores), 2) if scores else None

    # Step 5 — Generate overall summary
    overall_summary: str | None = None
    if evaluations:
        try:
            overall_summary = generate_overall_summary(
                candidate_name=candidate_name,
                job_role=job_role,
                level=level,
                evaluations=evaluations,
            )
            logger.info(f"[Evaluator] Overall summary generated.")
        except Exception as exc:
            logger.error(f"[Evaluator] Failed to generate overall summary: {exc}")
            errors.append(f"Overall summary generation failed: {exc}")

    logger.info(
        f"[Evaluator] Done. evaluated={len(evaluations)}, "
        f"errors={len(errors)}, avg_score={average_score}"
    )

    return {
        "session_id":      session_id,
        "candidate_name":  candidate_name,
        "job_role":        job_role,
        "level":           level,
        "average_score":   average_score,
        "overall_summary": overall_summary,
        "evaluations":     evaluations,
    }
