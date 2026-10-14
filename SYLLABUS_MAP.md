# PEEC411C Agentic AI & LLM — where each topic lives in this project

Project: **Teacher Assistant** — an *AI copilot for teachers* (Unit 5, Application 2). It turns a
teacher's notes into lesson plans, quizzes and coding exercises, and answers questions about the notes.

Status key: ✅ implemented and used · 🟡 partly / simplified · 💬 not implemented here — be ready to discuss only · ➖ belongs to another module (Peer: Whisper/video)

## Unit 1 — Foundations of LLMs  (theory; the project *uses* these)
| Topic | In the project |
|---|---|
| Transformers, causal LM, pre-training objectives | 💬 The model we run, `qwen3:1.7b`, is a decoder-only causal LM (next-token prediction). `nomic-embed-text` is an encoder-style embedding model. |
| Tokenization (BPE) | 💬 Qwen3 uses a byte-level BPE tokenizer. Visible effect: Hindi/Marathi cost more tokens and a ~2B model handles them worse. |
| Models: GPT, BERT, LLaMA | 💬 Swap models with one env var via LiteLLM (see below). |
| Metrics & evaluation of LLMs | 🟡 `evaluate.py` measures success rate, latency, quiz usability and grounding. |

## Unit 2 — Prompt engineering and RAG
| Topic | In the project |
|---|---|
| Zero-shot, role / system prompts | ✅ Every agent: `agents/teacher.py`, `quiz.py`, `coding.py` (role + audience + language in the system prompt). |
| Structured output | ✅ Quiz/code prompts specify an exact JSON shape; `llm_client.call_llm_json` retries with a stricter reminder and `extract_json` repairs fenced/wrapped JSON. |
| Few-shot / chain-of-thought | 💬 Not used (small model + latency). The ReAct *Thought* line in `ask_agent.py` is a form of CoT. |
| In-context learning | ✅ Retrieved note excerpts are placed in the prompt (`context`). |
| **RAG** | ✅ `vector_store.py` (chunk → embed → search) + `agents/retriever.py` as a LangGraph node. |
| **Embeddings + semantic search, FAISS** | ✅ `llm_client.embed` (Ollama `nomic-embed-text`), FAISS `IndexFlatIP` over normalized vectors = cosine similarity; NumPy fallback; keyword fallback if embeddings are down. Pinecone: 💬 hosted alternative to FAISS, not used. |
| Evaluation of prompts | ✅ `python -m teacher_backend.evaluate` runs RAG **on vs off** and compares grounding. |
| **Hallucination mitigation** | ✅ Four layers: (1) retrieval grounds the prompt, (2) "do not invent facts / say if not covered" instruction, (3) `agents/grounding_check.py` guardrail node scores output against the notes, (4) the UI flags quiz questions the notes don't support and lists the sources used. |
| Multimodal prompting, CLIP, Whisper, Gemini/GPT multimodal | ➖ Whisper lives in the Peer module (lecture video → transcript). Not in the Teacher module. |

## Unit 3 — Agentic AI systems
| Topic | In the project |
|---|---|
| Intro to agentic AI; multi-agent systems | ✅ `graph.py`: Orchestrator → Retriever → Teacher \| Quiz \| Coding → Grounding check, as a **LangGraph** state graph with conditional edges. |
| LLM as reasoning engine | ✅ In the Ask agent the model decides which tool to call. 🟡 The orchestrator is deliberately *deterministic* (routes on the button the teacher pressed) — cheaper and more reliable than an LLM router on a 1B model. |
| **Tool-using agents** | ✅ `agents/ask_agent.py` + `tools.py`: `search_notes` (RAG), `list_topics`, `calculator` (safe AST evaluator, no `eval`), `remember`. |
| **ReAct** | ✅ Thought → Action → Observation loop, max 4 steps, forced final answer if it runs out; the UI shows the steps ("How I found this"). |
| **Memory (short / long term)** | ✅ Short-term: last 6 chat turns sent with each question. Long-term: `memory.py` JSON file of facts the agent saves with the `remember` tool and re-reads in later sessions. |
| Frameworks: LangChain | 🟡 We use **LangGraph** (from the LangChain team) for the agent graph, not LangChain chains. AutoGPT / CrewAI: 💬 alternatives, not used. |
| Tree-of-Thoughts, multimodal agents | 💬 Not implemented. |

## Unit 4 — LLM APIs for application development
| Topic | In the project |
|---|---|
| API response handling | ✅ `llm_client.py`: timeouts, one error type (`RuntimeError`) for every failure, empty-reply check, JSON extraction + retries. |
| Context management | ✅ Retrieved text capped (`RAG_CONTEXT_CHARS`), top-k chunks, chat history capped (`ASK_HISTORY_TURNS`), per-topic notes capped. |
| Local LLMs | ✅ Ollama, default `qwen3:1.7b` — runs offline on a school laptop. |
| **LiteLLM** | ✅ `llm_client.py` calls `litellm.completion` / `litellm.embedding` when installed. Switch provider with env vars only, e.g. `TUTOR_LITELLM_MODEL=gemini/gemini-1.5-flash` + the provider's API key. Falls back to direct Ollama if LiteLLM is absent. |
| **Langfuse** | ✅ Set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` (+ `LANGFUSE_HOST`) and LiteLLM's Langfuse callback logs every generation. Independently, `observability.py` keeps a local trace log (request → retrieve → LLM spans with latency/tokens) at `GET /api/traces`. |
| Best practices | ✅ Single LLM access point, validation of model output, graceful degradation (RAG → keyword → full notes; LiteLLM → direct), input limits, no `eval`, secrets only via env vars. |

## Unit 5 — Applications
| Topic | In the project |
|---|---|
| AI Copilot / Intelligent Assistant | ✅ This project: a copilot for teachers in under-resourced schools. |
| Customer-support agent, video summarizer, image captioning, speech-to-text, media analysis | ➖/💬 Peer module covers lecture-video → transcript (Whisper). Others not built. |

## Likely oral questions — short answers
1. **What is RAG and why did you use it?** The model answers from retrieved excerpts of the teacher's notes instead of memory, which cuts hallucination and keeps prompts small for a 1B model with a tiny context window.
2. **How does retrieval work?** Notes are split into ~700-character chunks, each embedded by `nomic-embed-text`; the topic title is embedded as the query; FAISS returns the nearest chunks by cosine similarity (inner product on normalized vectors). Chunks from the chosen topic get a small score boost.
3. **Why FAISS and not Pinecone?** Local, free, no network — fits the offline rural-school constraint. Pinecone is a managed service; same idea, different store.
4. **What if embeddings are unavailable?** It degrades to BM25-style keyword scoring, then to the topic's full notes; generation never fails because retrieval did. The response states which method ran.
5. **How do you detect hallucination?** A guardrail node measures what fraction of the output's content words (stems) appear in the retrieved notes. It's a heuristic, not a fact-checker; quiz questions below 50% are flagged. Only for English output.
6. **What is ReAct?** The model alternates reasoning ("Thought"), a tool call ("Action"), and the result ("Observation") until it can answer. Ours has four tools and a 4-step cap.
7. **Why is the orchestrator not an LLM?** The user already chose the action; deterministic routing is faster and never misroutes. An LLM router would only be needed for free-text requests.
8. **Short-term vs long-term memory?** Short-term = recent conversation turns in the prompt. Long-term = facts persisted to disk and injected into every future prompt as background data.
9. **What does LiteLLM give you?** One call signature for 100+ providers; changing from local Ollama to Gemini/OpenAI is an environment variable, and its callbacks feed Langfuse.
10. **What does Langfuse give you?** Observability: per-call prompt, output, tokens, latency and cost, grouped by trace — useful to debug prompts and compare models.
11. **How is the calculator tool safe?** It parses the expression to an AST and allows only numbers and arithmetic operators; no names, calls or attributes; exponent and length are capped.
12. **Limits / what would you improve?** 1B model is weak at Hindi/Marathi and at ReAct formatting (hence the fallbacks); grounding score is lexical; no auth; embeddings re-computed per subject on first use. Next: a bigger local model, an LLM-as-judge evaluation, multimodal (scanned-PDF OCR, images) notes.
