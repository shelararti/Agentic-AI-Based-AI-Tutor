"""
FastAPI layer for the Teacher module.

Flow: a teacher picks a built-in subject, uploads a PDF of notes, or pastes
notes -> topics are built -> the teacher clicks Teach / Quiz / Code on a
topic and the agent graph generates the material.
"""

import re
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, knowledge_base, llm_client, memory, observability, pdf_ingest, vector_store
from .agents import ask_agent, tutor_chat
from .graph import teacher_graph

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
LANGUAGES = config.LANGUAGES
MAX_PDF_BYTES = 10 * 1024 * 1024

app = FastAPI(title="Teacher Assistant")


class GenerateRequest(BaseModel):
    kind: str                      # "teach" | "quiz" | "code"
    topic_id: str
    level: str = "beginner"
    language: str = "English"
    num_questions: int = 5


class NotesRequest(BaseModel):
    title: str
    text: str


class AskTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=2000)


class AskRequest(BaseModel):
    subject_id: str
    question: str = Field(min_length=1, max_length=1000)
    language: str = "English"
    history: list[AskTurn] = Field(default_factory=list, max_length=12)


class TutorRequest(BaseModel):
    topic_id: str
    message: str = Field(default="", max_length=1000)    # empty = the tutor opens the lesson
    level: str = "beginner"
    language: str = "English"
    history: list[AskTurn] = Field(default_factory=list, max_length=24)


def _subject_summary(s: dict) -> dict:
    return {
        "id": s["id"],
        "title": s["title"],
        "description": s.get("description", ""),
        "source": s.get("source", "builtin"),
        "topic_count": len(s["topics"]),
    }


def _register_subject(title: str, description: str, build_topics) -> dict:
    """Pick a unique id, build (topics, full_text) via build_topics(subject_id), register."""
    base = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") or "notes"
    subject_id, n = base, 2
    while knowledge_base.get_subject(subject_id):
        subject_id, n = f"{base}-{n}", n + 1
    try:
        topics, corpus = build_topics(subject_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    vector_store.drop(subject_id)
    subject = knowledge_base.add_subject(subject_id, title, description, topics, source="uploaded", corpus=corpus)
    return {
        "subject": _subject_summary(subject),
        "topics": [{"id": t["id"], "title": t["title"]} for t in topics],
    }


def _pdf_build(data: bytes, subject_id: str):
    text = pdf_ingest.extract_text(data)
    if not text.strip():
        raise ValueError("Could not find any selectable text in this PDF -- it may be "
                         "scanned images rather than real text.")
    return pdf_ingest.text_to_topics(text, subject_id), text


@app.get("/api/subjects")
def list_subjects():
    return [_subject_summary(s) for s in knowledge_base.all_subjects()]


@app.get("/api/languages")
def list_languages():
    return LANGUAGES


@app.post("/api/subjects/upload")
async def upload_notes_pdf(file: UploadFile = File(...), title: Optional[str] = Form(None)):
    filename = file.filename or "notes.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Please upload a PDF file.")
    data = await file.read()
    if len(data) > MAX_PDF_BYTES:
        raise HTTPException(400, "PDF is too large (limit 10 MB).")
    subject_title = (title or Path(filename).stem or "Uploaded notes").strip() or "Uploaded notes"
    return _register_subject(
        subject_title, f"Uploaded from {filename}.",
        lambda sid: _pdf_build(data, sid),
    )


@app.post("/api/subjects/from-notes")
def notes_to_subject(req: NotesRequest):
    title = req.title.strip() or "My notes"
    return _register_subject(
        title, "Pasted notes.", lambda sid: (pdf_ingest.text_to_topics(req.text, sid), req.text)
    )


@app.get("/api/topics")
def list_topics(subject_id: str):
    if not knowledge_base.get_subject(subject_id):
        raise HTTPException(404, f"Unknown subject_id: {subject_id}")
    return [{"id": t["id"], "title": t["title"]} for t in knowledge_base.all_topics(subject_id)]


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.language not in LANGUAGES:
        raise HTTPException(400, f"Language must be one of {LANGUAGES}")
    with observability.span("generate", kind=req.kind, topic=req.topic_id, language=req.language) as meta:
        result = teacher_graph.invoke({
            "intent": req.kind,
            "topic_id": req.topic_id,
            "level": req.level,
            "language": req.language,
            "num_questions": req.num_questions,
        })
        meta["retrieval"] = result.get("retrieval_method")
        meta["error"] = result.get("error")
    if result.get("error"):
        raise HTTPException(400, result["error"])

    out = {"kind": req.kind, "topic_title": result["topic_title"],
           "sources": result.get("sources", []), "retrieval": result.get("retrieval_method"),
           "grounding": result.get("grounding")}
    if req.kind == "teach":
        out["lesson"] = result["lesson"]
    elif req.kind == "quiz":
        out["questions"] = result["quiz"]["questions"]
    else:
        out.update(result["code_task"])
    return out


@app.post("/api/ask")
def ask(req: AskRequest):
    """ReAct agent over one subject's notes (tools: search_notes, list_topics, calculator, remember)."""
    if req.language not in LANGUAGES:
        raise HTTPException(400, f"Language must be one of {LANGUAGES}")
    subject = knowledge_base.get_subject(req.subject_id)
    if subject is None:
        raise HTTPException(404, f"Unknown subject_id: {req.subject_id}")
    history = [{"role": t.role, "text": t.text} for t in req.history]
    with observability.span("ask", subject=req.subject_id, language=req.language):
        try:
            return ask_agent.run(subject, req.question.strip(), req.language, history)
        except RuntimeError as e:
            raise HTTPException(503, str(e))


@app.post("/api/teach-chat")
def teach_chat(req: TutorRequest):
    """Conversational teaching: the student asks, the tutor explains, turn after turn."""
    if req.language not in LANGUAGES:
        raise HTTPException(400, f"Language must be one of {LANGUAGES}")
    history = [{"role": t.role, "text": t.text} for t in req.history]
    with observability.span("teach_chat", topic=req.topic_id, language=req.language):
        try:
            return tutor_chat.run(req.topic_id, req.message, history, req.language, req.level)
        except KeyError:
            raise HTTPException(404, f"Unknown topic_id: {req.topic_id}")
        except RuntimeError as e:
            raise HTTPException(503, str(e))


@app.get("/api/memory")
def get_memory():
    return {"facts": memory.load()}


@app.delete("/api/memory")
def clear_memory():
    memory.clear()
    return {"facts": []}


@app.get("/api/traces")
def get_traces(limit: int = 50):
    return observability.recent(max(1, min(limit, 200)))


@app.get("/api/system")
def system_info():
    """What is running under the hood (shown in the page footer)."""
    return {
        "llm": llm_client.backend_info(),
        "embeddings": {"model": config.EMBED_MODEL, "faiss": vector_store.faiss_available()},
        "observability": observability.status(),
    }


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
