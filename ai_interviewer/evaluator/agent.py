"""
Evaluator Agent
===============

Expert technical interviewer that scores a single Q&A pair.

The candidate's answer comes from a speech transcript, so it may contain
STT errors. The agent evaluates based on intended meaning, not surface form.

Usage
-----
    from evaluator.agent import evaluate_answer

    result = evaluate_answer(
        question="Explain the difference between a list and a tuple in Python.",
        golden_answer="Lists are mutable, tuples are immutable ...",
        candidate_answer="uh lists can be changed tuples cannot ...",
    )
    # result is a dict with: question, score, covered_requirements, missing_requirements
"""

import json
import sys
import os

# Allow importing config.py from the parent project directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groq import Groq
from config import get_key


# ---------------------------------------------------------------------------
# Groq client — key pulled from api_keys.json / environment, never hardcoded
# ---------------------------------------------------------------------------

def _build_client() -> Groq:
    api_key = get_key("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Add it to api_keys.json or set the GROQ_API_KEY environment variable."
        )
    return Groq(api_key=api_key)


# ---------------------------------------------------------------------------
# Model config — can be overridden in api_keys.json as EVALUATOR_MODEL
# ---------------------------------------------------------------------------

EVALUATOR_MODEL = get_key("EVALUATOR_MODEL", "openai/gpt-oss-120b")


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are an expert technical interviewer.

The candidate's answer comes from an automatic speech transcript.
The transcript may contain spelling mistakes, grammar mistakes,
punctuation errors, repeated words, and speech-to-text errors.

IMPORTANT:
- Evaluate based on the intended meaning.
- Ignore minor transcription mistakes.
- Infer the key requirements from the golden answer.
- Compare the candidate's answer against those inferred requirements.
- Assign a score between 0 and 100 based on how well the candidate answered.

Return ONLY valid JSON with EXACTLY the following schema:

{
    "question": "",
    "score": 0,
    "covered_requirements": [],
    "missing_requirements": []
}

Do not add any other fields.
Do not return markdown.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_answer(
    question: str,
    golden_answer: str,
    candidate_answer: str,
) -> dict:
    """
    Evaluate a single technical Q&A pair.

    Parameters
    ----------
    question : str
        The interview question that was asked.
    golden_answer : str
        The ideal / reference answer used as a rubric.
    candidate_answer : str
        The candidate's raw speech transcript response.

    Returns
    -------
    dict
        Keys: question, score (0-100), covered_requirements, missing_requirements
    """
    client = _build_client()

    user_prompt = f"""
Question:
{question}

Golden Answer:
{golden_answer}

Candidate Answer (Transcript):
{candidate_answer}
"""

    response = client.chat.completions.create(
        model=EVALUATOR_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Overall performance summary
# ---------------------------------------------------------------------------

_SUMMARY_SYSTEM_PROMPT = """\
You are a senior technical hiring manager writing a final performance summary \
for a candidate after a technical interview.

You will be given:
- The job role the candidate applied for and the experience level expected.
- A list of evaluated questions with scores, covered requirements, and missing requirements.

Write a very brief, high-level summary (1-2 sentences maximum). 
State whether the performance was overall good, average, or poor for the role, and briefly mention if they answered most questions technically correct or struggled. 
Do not include specific details about individual questions or requirements.

Return ONLY the plain-text paragraph. No JSON. No headers. No bullet points.\
"""


def generate_overall_summary(
    candidate_name: str,
    job_role: str,
    level: str,
    evaluations: list[dict],
) -> str:
    """
    Generate a role-aware holistic performance summary using an LLM.

    Parameters
    ----------
    candidate_name : str
        The candidate's full name.
    job_role : str
        The job role the candidate applied for.
    level : str
        Experience level (e.g. Junior, Mid, Senior).
    evaluations : list[dict]
        List of per-question evaluation dicts produced by evaluate_answer().
        Each dict contains: question, score, covered_requirements, missing_requirements.

    Returns
    -------
    str
        A plain-text paragraph summarising overall performance.
    """
    client = _build_client()

    # Build a compact summary of each question for the LLM context
    qa_lines: list[str] = []
    for i, ev in enumerate(evaluations, start=1):
        covered = ", ".join(ev.get("covered_requirements", [])) or "none"
        missing = ", ".join(ev.get("missing_requirements", [])) or "none"
        qa_lines.append(
            f"Q{i}: {ev.get('question', '')}\n"
            f"  Score: {ev.get('score', 'N/A')}/100\n"
            f"  Covered: {covered}\n"
            f"  Missing: {missing}"
        )

    user_prompt = (
        f"Candidate: {candidate_name}\n"
        f"Applied for: {level} {job_role}\n\n"
        + "\n\n".join(qa_lines)
    )

    response = client.chat.completions.create(
        model=EVALUATOR_MODEL,
        messages=[
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()
