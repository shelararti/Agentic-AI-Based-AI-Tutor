"""Coding Agent -- generates a coding exercise with a rubric and a sample solution."""

from .. import llm_client


def coding_node(state: dict) -> dict:
    level = state.get("level", "beginner")
    language = state.get("language", "English")
    system = (
        f"You design short hands-on coding exercises for teachers, written in {language}. "
        "Respond with ONLY valid JSON, no markdown fences, no extra text."
    )
    user = (
        f'Create one coding exercise on "{state.get("topic_title", "")}" for {level} learners, '
        f"based on: {(state.get('context') or state.get('topic_notes', ''))}\n"
        "Return this exact JSON shape:\n"
        '{"task":"clear description of what to write","rubric":["criterion 1","criterion 2",'
        '"criterion 3"],"solution":"a short sample solution"}'
    )
    try:
        data = llm_client.call_llm_json(system, user)
    except Exception as e:
        return {"error": f"Could not generate a coding task: {e}"}
    if not data.get("task"):
        return {"error": "The model returned no task. Please try again."}
    rubric = data.get("rubric")
    return {"code_task": {
        "task": data["task"],
        "rubric": rubric if isinstance(rubric, list) else [],
        "solution": data.get("solution", ""),
    }}
