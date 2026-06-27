"""
Interactive Interview - YOU are the candidate
===============================================

Runs the full interview pipeline interactively:
- The Interviewer (Llama via Groq) asks you questions
- YOU type your answers
- The Evaluator scores your answers against golden answers
- Follow-up probes happen automatically if your answer needs more depth
"""

import json
import os
import sys
from datetime import datetime

from groq import Groq

sys.path.insert(0, os.path.dirname(__file__))

from config import get_key

from state_machine import InterviewStateMachine, InterviewState
from models import Candidate, Interview, Answer, Score, ScoreDimension
from prompts import (
    INTERVIEWER_SYSTEM_PROMPT,
    build_interviewer_user_prompt,
)
from conversation import (
    ConversationHistory,
    decide_followup,
    build_evaluator_messages,
)

# ===========================================================================
#  CONFIG
# ===========================================================================

GROQ_MODEL = "llama-3.3-70b-versatile"

PLANNER_OUTPUT = [
    { "session_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890", "candidate_name": "Ahmed Hassan", "job_role": "Machine Learning Engineer", "level": "Junior", "assessment_strategy": { "strengths": [ "python", "deep_learning", "classical_ml", "data preprocessing and feature engineering", "model_evaluation", "statistics_mathematics" ], "gaps": [ "mlops", "nlp", "computer vision" ], "focus_skills": [ { "skill": "python", "importance": 10, "reason": "Must-have skill. Candidate has some exposure; depth to be probed. Importance: 10/10." }, { "skill": "classical_ml", "importance": 9, "reason": "Must-have skill. Candidate has some exposure; depth to be probed. Importance: 9/10." }, { "skill": "data preprocessing and feature engineering", "importance": 9, "reason": "Must-have skill. Candidate has some exposure; depth to be probed. Importance: 9/10." }, { "skill": "mlops", "importance": 5, "reason": "Gap in candidate's background; present in JD competencies. Importance: 5/10." }, { "skill": "deep_learning", "importance": 7, "reason": "Relevant competency for a Junior deep_learning role. Importance: 7/10." }, { "skill": "statistics_mathematics", "importance": 6, "reason": "Relevant competency for a Junior statistics_mathematics role. Importance: 6/10." } ] }, "questions": [ { "question": "How do you write unit tests for a machine learning model or data pipeline in Python using pytest?", "golden_answer": "You write deterministic tests that check the structural properties of your system. Using pytest, I would assert that the data preprocessing function outputs DataFrames of the expected shape and lacking NaNs. For models, I would test that the forward pass outputs the correct tensor dimensions, the model successfully overfits on a single batch (proving gradients flow), and the loss strictly decreases during dummy training steps.", "skill": "python", "rationale": "As a Junior Machine Learning Engineer with a strong Python background, starting with a warmup question on unit testing in Python helps build confidence and assesses a fundamental skill." }, { "question": "Which classical ML algorithms require feature scaling and which do not?", "golden_answer": "Distance-based algorithms (KNN, K-Means, SVM) and gradient-based algorithms (Linear/Logistic Regression) require scaling to ensure gradients converge smoothly. Tree-based algorithms (Decision Trees, Random Forests, Gradient Boosting) do not require scaling as they split based on ordinal inequalities.", "skill": "classical_ml", "rationale": "Given the candidate's background in classical ML, this question assesses their understanding of a critical preprocessing step, aligning with their strengths." }, { "question": "You are building a model to predict retail store sales. You have a 'Date' column. What feature engineering steps would you perform on this?", "golden_answer": "A raw Date column must be transformed. I would extract temporal features: cyclical features (day_of_week, month) to capture seasonality, binary flags (is_weekend, is_holiday), and absolute time features. For cyclical features like 'month' (1-12), I would apply sine and cosine transformations so the model understands that December and January are chronologically adjacent.", "skill": "data preprocessing and feature engineering", "rationale": "This question evaluates the candidate's ability to apply feature engineering techniques, a key aspect of their strengths, to a practical problem." }, { "question": "Explain how *args and kwargs are utilized when structuring complex machine learning pipelines in Python.", "golden_answer": "*args passes a variable number of non-keyworded arguments, while kwargs passes variable keyword arguments as a dictionary. In ML pipelines, kwargs is incredibly useful for passing arbitrary hyperparameter configurations down a deep stack of wrapper functions or base classes, allowing you to initialize an underlying Scikit-Learn or PyTorch estimator without explicitly defining every possible parameter in the parent function.", "skill": "python", "rationale": "As a Junior with a strong Python foundation, understanding how to leverage *args and kwargs is crucial for building flexible and reusable ML pipelines." }, { "question": "Describe how you would evaluate and mitigate bias in an automated ML pipeline.", "golden_answer": "Bias evaluation should be a mandatory automated step in the CI/CD pipeline. I would slice the validation dataset by sensitive attributes (e.g., race, gender, age) and calculate fairness metrics like demographic parity or equalized odds across these groups. If bias is detected, the pipeline fails the deployment. Mitigation involves resampling the training data, applying fairness-aware algorithms, or re-weighting the loss function.", "skill": "mlops", "rationale": "Given the candidate's gap in MLOps, this question directly addresses their need to understand bias evaluation and mitigation, a critical aspect of model deployment and maintenance." }, { "question": "What is the core idea behind the self-attention mechanism in modern deep learning models?", "golden_answer": "Self-attention computes a representation of a sequence by relating different positions of a single sequence to compute a weighted sum of the inputs. For each token, it generates three vectors: Query, Key, and Value. The attention score between two tokens is calculated by taking the dot product of the first token's Query with the second token's Key. This score is passed through a Softmax function to determine the weight, which is then multiplied by the Value vector. This allows the model to learn dynamically which parts of the sequence are most relevant to each specific word.", "skill": "deep_learning", "rationale": "As the candidate has experience with deep learning, probing their understanding of a key concept like self-attention helps assess their depth of knowledge in this area." }, { "question": "Why is the Central Limit Theorem critical for machine learning and statistics?", "golden_answer": "It guarantees that the distribution of sample means will be approximately normal given a large enough sample size, regardless of the underlying population distribution. This allows us to use normal-distribution-based statistics and hypothesis tests on almost any dataset.", "skill": "statistics_mathematics", "rationale": "Understanding the Central Limit Theorem is fundamental for any machine learning or statistics work, making it a crucial question for a Junior candidate to demonstrate their grasp of statistical basics." }, { "question": "Can you explain conditional probability using a practical machine learning example?", "golden_answer": "Conditional probability measures the likelihood of an event occurring given that another event has already occurred. In a spam filter, it is the probability that an email is spam given that it contains the word lottery.", "skill": "statistics_mathematics", "rationale": "As a closing question, explaining conditional probability with a practical example allows the candidate to reflect on and demonstrate their understanding of a foundational statistical concept in a real-world context, showcasing their ability to apply theoretical knowledge to practical problems." } ], "total_questions": 8}  
]


# ===========================================================================
#  HELPERS
# ===========================================================================

client = Groq(api_key=get_key("GROQ_API_KEY"))


def call_groq(messages, temperature=0.7):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


def call_evaluator(question, candidate_answer):
    messages = build_evaluator_messages(
        question_text=question["question_text"],
        candidate_answer=candidate_answer,
        golden_answer=question["golden_answer"],
        difficulty=question["difficulty"],
        topic=question["topic"],
    )
    raw = call_groq(messages, temperature=0.2)
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def parse_score(eval_result):
    return Score(**{
        "overall": eval_result.get("overall", 0),
        "dimensions": [
            ScoreDimension(**d) for d in eval_result.get("dimensions", [])
        ],
        "key_points_covered": eval_result.get("key_points_covered", []),
        "key_points_missed": eval_result.get("key_points_missed", []),
        "strengths": eval_result.get("strengths", []),
        "improvements": eval_result.get("improvements", []),
        "needs_followup": eval_result.get("needs_followup", False),
        "followup_reason": eval_result.get("followup_reason", ""),
        "notes": eval_result.get("notes", ""),
    })


def sep(title=""):
    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
        print('=' * 60)


def get_user_input():
    """Get multi-line input from the user. Empty line submits."""
    print("\n>> Your answer (press Enter twice to submit):")
    lines = []
    while True:
        line = input()
        if line == "" and lines:
            break
        lines.append(line)
    return "\n".join(lines).strip()


# ===========================================================================
#  INTERACTIVE INTERVIEW
# ===========================================================================

def main():
    sep("AI INTERVIEWER - INTERACTIVE MODE")
    print("  You are the candidate. Answer the questions!")
    print("  Type your answer, then press Enter twice to submit.")
    print(f"  Model: {GROQ_MODEL}")
    print(f"  Questions: {len(PLANNER_OUTPUT)}")

    # -- Get candidate info ------------------------------------------------
    print("\n-- Before we start, a few details --")
    name = input("Your name: ").strip() or "Candidate"
    role = input("Role you're applying for: ").strip() or "Software Engineer"
    exp_str = input("Years of experience: ").strip()
    exp = int(exp_str) if exp_str.isdigit() else 0
    skills_raw = input("Your skills (comma-separated): ").strip()
    skills = [s.strip() for s in skills_raw.split(",") if s.strip()] if skills_raw else []

    candidate = Candidate(
        name=name,
        role=role,
        experience_years=exp,
        skills=skills,
    )

    interview = Interview(
        candidate=candidate,
        topic="Python fundamentals",
        max_questions=len(PLANNER_OUTPUT),
    )

    sm = InterviewStateMachine()
    history = ConversationHistory()

    # -- INTRO -------------------------------------------------------------
    sep("INTERVIEW STARTING...")
    print("(Connecting to interviewer...)\n")

    history.add_state_prompt(
        "INTRO",
        candidate_name=candidate.name,
        candidate_role=candidate.role,
        experience_years=candidate.experience_years,
        skills=candidate.skill_tags,
        interview_type=interview.interview_type.value,
        topic=interview.topic,
        max_questions=interview.max_questions,
    )

    intro = call_groq(history.get_messages())
    history.add_assistant_message(intro)

    sep("INTERVIEWER")
    print(f"\n{intro}")

    # Wait for candidate to acknowledge
    print("\n>> Press Enter to start the interview...")
    input()

    sm.start_asking()
    interview.current_state = sm.state

    # -- QUESTION LOOP -----------------------------------------------------
    all_scores = []

    for i, question in enumerate(PLANNER_OUTPUT):
        q_num = i + 1
        followups_for_q = 0

        # -- ASK -----------------------------------------------------------
        sep(f"QUESTION {q_num} of {interview.max_questions}")

        previous = interview.answers[-1].answer_text if interview.answers else "(first question)"

        history.add_state_prompt(
            "ASK",
            question_text=question["question_text"],
            question_index=q_num,
            max_questions=interview.max_questions,
            question_topic=question["topic"],
            question_difficulty=question["difficulty"],
            previous_answer=previous,
        )

        ask_resp = call_groq(history.get_messages())
        history.add_assistant_message(ask_resp)

        print(f"\n{ask_resp}")

        # -- GET YOUR ANSWER -----------------------------------------------
        candidate_answer = get_user_input()
        history.add_candidate_answer(candidate_answer)

        # -- EVALUATE (silently) -------------------------------------------
        print("\n(Evaluating your answer...)")
        try:
            eval_result = call_evaluator(question, candidate_answer)
            score = parse_score(eval_result)
        except Exception as e:
            print(f"(Evaluator error: {e} - skipping scoring)")
            eval_result = {"overall": 5.0, "needs_followup": False}
            score = Score(overall=5.0)

        answer = Answer(
            question=question["question_text"],
            answer_text=candidate_answer,
            is_followup=False,
            score=score,
        )
        interview.add_answer(answer)
        all_scores.append(score)

        # -- FOLLOW-UP DECISION --------------------------------------------
        decision = decide_followup(
            candidate_answer=candidate_answer,
            evaluator_score=eval_result.get("overall"),
            followups_so_far=followups_for_q,
        )
        llm_wants = eval_result.get("needs_followup", False)
        should_followup = decision.needs_followup or llm_wants

        # -- FOLLOWUP LOOP -------------------------------------------------
        while should_followup and followups_for_q < 2:
            # Only transition if not already in FOLLOWUP state
            if sm.state != InterviewState.FOLLOWUP:
                sm.follow_up()
            interview.current_state = sm.state

            sep(f"FOLLOW-UP (Q{q_num})")

            history.add_state_prompt(
                "FOLLOWUP",
                question_text=question["question_text"],
                candidate_answer=candidate_answer,
            )

            followup_resp = call_groq(history.get_messages())
            history.add_assistant_message(followup_resp)

            print(f"\n{followup_resp}")

            # Get follow-up answer
            candidate_answer = get_user_input()
            history.add_candidate_answer(candidate_answer)

            followups_for_q += 1

            # Record follow-up answer
            fu_answer = Answer(
                question=followup_resp,
                answer_text=candidate_answer,
                is_followup=True,
                parent_answer_id=answer.id,
            )
            interview.add_answer(fu_answer)

            # Re-evaluate
            print("\n(Evaluating follow-up...)")
            try:
                eval_result = call_evaluator(question, candidate_answer)
            except Exception:
                eval_result = {"overall": 5.0, "needs_followup": False}

            decision = decide_followup(
                candidate_answer=candidate_answer,
                evaluator_score=eval_result.get("overall"),
                followups_so_far=followups_for_q,
            )
            llm_wants = eval_result.get("needs_followup", False)
            should_followup = decision.needs_followup or llm_wants

        # -- Transition to next question or close --------------------------
        if sm.state == InterviewState.FOLLOWUP:
            if q_num < len(PLANNER_OUTPUT):
                sm.next_question()
            else:
                sm.close()
        elif sm.state == InterviewState.ASK:
            sm.follow_up()
            if q_num < len(PLANNER_OUTPUT):
                sm.next_question()
            else:
                sm.close()

        interview.current_state = sm.state

    # -- CLOSE -------------------------------------------------------------
    sep("INTERVIEW CLOSING")

    duration = round((datetime.utcnow() - interview.started_at).total_seconds() / 60, 1)

    history.add_state_prompt(
        "CLOSE",
        candidate_name=candidate.name,
        questions_asked=interview.questions_asked,
        followups_asked=interview.followups_asked,
        duration=str(duration),
    )

    close_resp = call_groq(history.get_messages())
    history.add_assistant_message(close_resp)

    print(f"\n{close_resp}")

    interview.close_interview(summary=close_resp)

    # -- SCORECARD ---------------------------------------------------------
    sep("YOUR SCORECARD")

    for j, ans in enumerate(interview.answers):
        if ans.score and not ans.is_followup:
            print(f"\n  Q{ans.question_index}: {ans.question[:50]}...")
            print(f"     Score: {ans.score.overall}/10")
            if ans.score.key_points_covered:
                print(f"     Covered: {', '.join(ans.score.key_points_covered[:3])}")
            if ans.score.key_points_missed:
                print(f"     Missed:  {', '.join(ans.score.key_points_missed[:3])}")
            if ans.score.strengths:
                print(f"     Strengths:    {', '.join(ans.score.strengths[:2])}")
            if ans.score.improvements:
                print(f"     To improve:   {', '.join(ans.score.improvements[:2])}")

    avg = interview.average_score
    print(f"\n  {'=' * 40}")
    print(f"  Average Score: {avg}/10" if avg else "  Average Score: N/A")
    print(f"  Questions:     {interview.questions_asked}")
    print(f"  Follow-ups:    {interview.followups_asked}")
    print(f"  Duration:      {duration} min")
    print(f"  Verdict:       {'PASS' if avg and avg >= 6.0 else 'NEEDS IMPROVEMENT'}")
    sep("END")


if __name__ == "__main__":
    main()
