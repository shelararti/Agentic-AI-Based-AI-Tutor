# Teacher Assistant (Module 1)

A standalone prototype for teachers in under-resourced schools. A teacher picks a
built-in subject, uploads a PDF of notes, or pastes notes; the app splits them
into topics and, per topic, generates:

- a **lesson plan** (goals, explanation, worked example, key points, class questions)
- a **quiz** with answer key
- a **coding exercise** with rubric and solution

in English, Hindi or Marathi, at beginner / intermediate / advanced level.
Runs on a local Ollama model, so it works without a cloud account.

Syllabus tech (PEEC411C) used here: **RAG** (embeddings + FAISS), **LangGraph** multi-agent
graph, a **ReAct tool-using agent** with short/long-term **memory**, **LiteLLM**, **Langfuse**
observability, local LLMs, and prompt-/hallucination-evaluation. See `SYLLABUS_MAP.md` for the
topic-by-topic mapping and oral Q&A.

- **Talk with your teacher** — under Teach, a conversation mode: the student asks questions one after another and the teacher explains each, grounded in the topic's notes and aware of the chat so far (`/api/teach-chat`)
- **Ask your notes** — a ReAct agent that searches the notes, does arithmetic, and remembers facts about you
- **Grounded output** — each lesson/quiz shows the sources retrieved and a grounding check; weakly supported quiz questions are flagged

## Run
```bash
pip install -r requirements.txt
ollama serve && ollama pull qwen3:1.7b && ollama pull nomic-embed-text   # chat model + embedding model
python -m uvicorn teacher_backend.api:app --port 8000     # run from this folder
# open http://localhost:8000
python -m teacher_backend.test_core                       # RAG / ReAct / grounding / memory tests, no Ollama or web framework needed
python -m teacher_backend.test_teacher                    # + API tests (mocked LLM)
python -m teacher_backend.evaluate --subject <id>         # RAG on vs off, needs Ollama running
```
Settings (env vars):

| Variable | Meaning |
|---|---|
| `TUTOR_MODEL_NAME` | local chat model (default `qwen3:1.7b`) |
| `TUTOR_EMBED_MODEL` | embedding model (default `nomic-embed-text`) |
| `TUTOR_LLM_BACKEND` | `auto` (LiteLLM if installed) / `litellm` / `ollama` |
| `TUTOR_LITELLM_MODEL` | any LiteLLM model string, e.g. `gemini/gemini-1.5-flash` (plus that provider's API key) |
| `TUTOR_RAG=0` | disable retrieval (paste whole notes) — for A/B comparison |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | turn on Langfuse tracing |
| `TUTOR_OLLAMA_URL`, `TUTOR_UPLOADS_DIR` | Ollama endpoint, storage folder |

Without `litellm`, `faiss-cpu` or Langfuse installed the app still runs (direct Ollama, NumPy search, local traces at `/api/traces`).

## Layout
- `api.py` — FastAPI routes (`/api/subjects`, `/api/generate`, `/api/ask`, `/api/memory`, `/api/traces`, `/api/system`) and the page
- `graph.py`, `agents/` — LangGraph: Orchestrator → Retriever → Teacher | Quiz | Coding → Grounding check
- `agents/tutor_chat.py` (conversational teaching), `agents/ask_agent.py`, `tools.py`, `memory.py` — ReAct agent, its tools, long-term memory
- `vector_store.py` — chunking, embeddings, FAISS/NumPy/keyword search; `grounding.py` — hallucination heuristic
- `observability.py` — trace spans + Langfuse switch; `evaluate.py` — A/B evaluation
- `knowledge_base.py`, `pdf_ingest.py` — subjects/topics from built-ins, PDFs, or pasted text
- `llm_client.py` — the single place that talks to the model (LiteLLM → Ollama, embeddings, retries, JSON extraction)
- `frontend/` — three modules (Teach / Quiz / Code) sharing a notes + topic sidebar; plain HTML/JS/CSS, no CDN libraries (fonts load from Google Fonts if online)

## Known prototype limits
The grounding score is a lexical heuristic (English output only), not a fact-checker. The ReAct agent
depends on a small model following a text format and falls back to treating its reply as the answer.
Small local model, so output quality varies (especially Hindi/Marathi); quiz
and code outputs are validated but not guaranteed correct, so review before
class. Uploaded subjects are stored as JSON files, not a database.
