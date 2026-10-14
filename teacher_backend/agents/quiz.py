"""Quiz Agent -- generates a multiple-choice quiz with an answer key for the teacher."""

from .. import llm_client


def _clean(questions: list) -> list:
    """Small models sometimes return malformed items; keep only usable ones."""
    good = []
    for q in questions if isinstance(questions, list) else []:
        opts = q.get("options") if isinstance(q, dict) else None
        idx = q.get("correct_index") if isinstance(q, dict) else None
        if q.get("question") and isinstance(opts, list) and len(opts) >= 2 \
                and isinstance(idx, int) and 0 <= idx < len(opts):
            good.append({
                "question": q["question"],
                "options": opts,
                "correct_index": idx,
                "explanation": q.get("explanation", ""),
            })
    return good


def quiz_node(state: dict) -> dict:
    n = max(1, min(int(state.get("num_questions") or 5), 10))
    level = state.get("level", "beginner")
    language = state.get("language", "English")
    system = (
        f"You write multiple-choice quizzes for teachers, in {language}. "
        "Respond with ONLY valid JSON, no markdown fences, no extra text."
    )
    user = (
        f'Write exactly {n} multiple-choice questions on "{state.get("topic_title", "")}" '
        f"for {level} learners, based only on these notes:\n{(state.get('context') or state.get('topic_notes', ''))}\n"
        "Each question has 4 options and one correct answer. Return this exact JSON shape:\n"
        '{"questions":[{"question":"...","options":["...","...","...","..."],'
        '"correct_index":0,"explanation":"..."}]}'
    )
    try:
        questions = _clean(llm_client.call_llm_json(system, user).get("questions"))
    except Exception as e:
        return {"error": f"Could not generate a quiz: {e}"}
    if not questions:
        return {"error": "The model returned no usable questions. Please try again."}
    return {"quiz": {"questions": questions}}
