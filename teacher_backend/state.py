"""Shared state passed through the Teacher graph."""

from typing import Optional, TypedDict


class TeacherState(TypedDict, total=False):
    # input
    intent: str            # "teach" | "quiz" | "code"
    level: str             # beginner | intermediate | advanced
    language: str          # output language, e.g. "English", "Hindi", "Marathi"
    num_questions: int     # quiz length
    subject_id: Optional[str]
    topic_id: Optional[str]

    # set by orchestrator
    topic_title: str
    topic_notes: str
    subject_title: str

    # set by retriever (RAG)
    context: str
    retrieval_method: str
    sources: list

    # outputs
    lesson: str
    quiz: dict             # {"questions": [...]}
    code_task: dict        # {"task", "rubric", "solution"}

    grounding: dict        # {"score", "label"} from the guardrail node

    error: Optional[str]
