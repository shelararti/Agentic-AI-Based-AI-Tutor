"""Retriever -- the RAG step. Finds the note chunks most relevant to the chosen
topic and puts them in state['context'] for the generator agents. Falls back to
the topic's own notes if retrieval returns nothing, so generation never breaks."""

from .. import config, knowledge_base, observability, vector_store

ROUTES = {"teach": "teacher_agent", "quiz": "quiz_agent", "code": "coding_agent"}


def retriever_node(state: dict) -> dict:
    subject = knowledge_base.get_subject(state.get("subject_id") or "")
    notes = state.get("topic_notes", "")
    title = state.get("topic_title", "")
    result = {"method": "none", "hits": []}
    with observability.span("retrieve", topic=title) as meta:
        if subject and config.RAG_ENABLED:
            try:
                result = vector_store.retrieve(subject, f"{title}. {notes[:200]}", prefer_topic_id=state.get("topic_id"))
            except Exception as e:      # never let retrieval take down generation
                meta["error"] = str(e)
        meta.update(method=result["method"], hits=len(result["hits"]))

    if result["hits"]:
        return {
            "context": vector_store.format_context(result["hits"]),
            "retrieval_method": result["method"],
            "sources": [{"topic": h["topic_title"], "score": h["score"], "snippet": h["text"][:140]}
                        for h in result["hits"]],
        }
    return {"context": notes, "retrieval_method": "full-notes",
            "sources": [{"topic": title, "score": None, "snippet": notes[:140]}]}


def route_after_retriever(state: dict) -> str:
    return ROUTES[state["intent"]]
