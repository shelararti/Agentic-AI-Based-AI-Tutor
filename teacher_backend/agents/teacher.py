"""Teacher Agent -- turns a teacher's notes into ready-to-use lesson material."""

from .. import llm_client


def teacher_node(state: dict) -> dict:
    level = state.get("level", "beginner")
    language = state.get("language", "English")
    system = (
        f"You help teachers in under-resourced schools prepare lessons. Write in simple, "
        f"clear {language}, suitable for {level} learners, using everyday examples. "
        f"Stay faithful to the notes provided; do not invent facts beyond them. "
        f"If the notes do not cover something, say so instead of guessing."
    )
    user = (
        f"Subject: {state.get('subject_title', '')}\nTopic: {state.get('topic_title', '')}\n\n"
        f"Teacher's notes (numbered excerpts):\n{(state.get('context') or state.get('topic_notes', ''))}\n\n"
        "Write a lesson plan with these sections, each with a short heading:\n"
        "1. Learning goals (2-3 bullets)\n"
        "2. Explanation (a few short paragraphs the teacher can read out)\n"
        "3. Worked example\n"
        "4. Key points to remember (3-5 bullets)\n"
        "5. Questions to ask the class (3 questions)"
    )
    try:
        return {"lesson": llm_client.call_llm(system, user)}
    except RuntimeError as e:
        return {"error": str(e)}
