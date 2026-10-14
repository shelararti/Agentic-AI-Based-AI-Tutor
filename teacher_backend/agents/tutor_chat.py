"""Tutor chat -- the conversational side of "Teach".

The learner asks questions one after another and the tutor explains each one,
like a teacher in a classroom: short, simple, grounded in the notes for the
chosen topic, and aware of the conversation so far (follow-ups like "why?" or
"give another example" work). No tool loop: one retrieval + one model call per
turn keeps it fast on a small local model.
"""

from .. import config, knowledge_base, llm_client, observability, vector_store

SYSTEM = """You are a patient, friendly teacher having a one-to-one conversation with a student about "{topic}" ({subject}).
Explain in simple {language}, suitable for {level} learners, with everyday examples.
Rules:
- Answer the student's latest question directly, in a short paragraph or two. Do not write a full lesson.
- Use only the notes below. If the notes do not cover the question, say so honestly and offer what the notes do say.
- Build on the earlier conversation; do not repeat yourself.
- End with ONE short follow-up question to check understanding or invite the next question.

Notes (numbered excerpts):
{context}"""

OPENING = ("Start the lesson: greet the student in one line, explain the main idea of this topic in a short, "
           "simple way, then ask what they would like to know more about.")


def run(topic_id: str, message: str, history: list, language: str, level: str) -> dict:
    topic = knowledge_base.get_topic(topic_id)
    if topic is None:
        raise KeyError(topic_id)
    subject = knowledge_base.get_subject(topic.get("subject_id"))
    question = (message or "").strip()
    query = f"{topic['title']}. {question}" if question else topic["title"]

    hits = []
    with observability.span("tutor_retrieve", topic=topic["title"]):
        if subject and config.RAG_ENABLED:
            try:
                hits = vector_store.retrieve(subject, query, prefer_topic_id=topic["id"])["hits"]
            except Exception:           # retrieval problems must not stop the conversation
                hits = []
    context = vector_store.format_context(hits) if hits else topic["notes"][:config.RAG_CONTEXT_CHARS]

    messages = [{"role": "system", "content": SYSTEM.format(
        topic=topic["title"], subject=subject["title"] if subject else "the course",
        language=language, level=level, context=context)}]
    messages += [{"role": m["role"], "content": m["text"]} for m in history[-config.ASK_HISTORY_TURNS * 2:]]
    messages.append({"role": "user", "content": question or OPENING})

    reply = llm_client.call_llm_messages(messages)
    return {"reply": reply, "topic_title": topic["title"],
            "sources": sorted({h["topic_title"] for h in hits}) or [topic["title"]]}
