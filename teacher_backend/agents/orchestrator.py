"""Orchestrator -- resolves the topic the teacher picked and routes to the
right generator (teach / quiz / code), via the retriever."""

from .. import knowledge_base

ROUTES = {"teach": "teacher_agent", "quiz": "quiz_agent", "code": "coding_agent"}


def orchestrator_node(state: dict) -> dict:
    if state.get("intent") not in ROUTES:
        return {"error": f"Unknown intent: {state.get('intent')!r}"}

    topic = knowledge_base.get_topic(state.get("topic_id") or "")
    if topic is None:
        return {"error": f"Unknown topic_id: {state.get('topic_id')!r}"}

    subject = knowledge_base.get_subject(topic.get("subject_id"))
    return {
        "topic_id": topic["id"],
        "subject_id": topic.get("subject_id"),
        "topic_title": topic["title"],
        "topic_notes": topic["notes"],
        "subject_title": subject["title"] if subject else "the course",
    }


def route_after_orchestrator(state: dict) -> str:
    return "end" if state.get("error") else "retriever"
