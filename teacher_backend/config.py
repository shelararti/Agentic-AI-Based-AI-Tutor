"""
Central configuration for the Teacher Assistant.
Keeping this in one place means swapping models, endpoints, or storage
locations later doesn't require touching agent logic.
"""

import os
from pathlib import Path

OLLAMA_URL = os.environ.get("TUTOR_OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL_NAME = os.environ.get("TUTOR_MODEL_NAME", "qwen3:1.7b")

LEVELS = ["beginner", "intermediate", "advanced"]
LANGUAGES = ["English", "Hindi", "Marathi"]

# Resolve storage paths relative to this package, not the launch directory.
BASE_DIR = Path(__file__).resolve().parent

# Where uploaded / pasted notes are saved as extra subjects (survive restarts).
UPLOADS_DIR = os.environ.get("TUTOR_UPLOADS_DIR", str(BASE_DIR / "tutor_uploads"))

LLM_TIMEOUT_SECONDS = 120
JSON_CALL_ATTEMPTS = 3

# ---------------------------------------------------------------------------
# LLM access: LiteLLM (provider-agnostic) with a direct-Ollama fallback
# ---------------------------------------------------------------------------
# "auto"    -> use LiteLLM if it is installed, else call Ollama directly
# "litellm" -> require LiteLLM        "ollama" -> always call Ollama directly
LLM_BACKEND = os.environ.get("TUTOR_LLM_BACKEND", "auto")
# Any LiteLLM model string works, e.g. "gemini/gemini-1.5-flash" or "gpt-4o-mini"
# (set the provider's API key env var). Default is the same local model.
LITELLM_MODEL = os.environ.get("TUTOR_LITELLM_MODEL", f"ollama_chat/{MODEL_NAME}")
LITELLM_API_BASE = os.environ.get("TUTOR_LITELLM_API_BASE", "http://localhost:11434")

# ---------------------------------------------------------------------------
# RAG (embeddings + vector search)
# ---------------------------------------------------------------------------
EMBED_MODEL = os.environ.get("TUTOR_EMBED_MODEL", "nomic-embed-text")   # ollama pull nomic-embed-text
RAG_CHUNK_CHARS = 700        # retrieval chunks are smaller than topics
RAG_TOP_K = 4
RAG_ENABLED = os.environ.get("TUTOR_RAG", "1") != "0"   # 0 = paste whole topic notes (for A/B evaluation)
RAG_CONTEXT_CHARS = 3000     # cap on retrieved text put into a prompt
RAG_SAME_TOPIC_BOOST = 0.05  # nudge results toward the topic the teacher picked

# ---------------------------------------------------------------------------
# Observability (local trace log always on; Langfuse if keys are set)
# ---------------------------------------------------------------------------
# Langfuse is enabled by setting LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
# (and LANGFUSE_HOST for self-hosted) -- see observability.py.
TRACE_BUFFER = 200

# ---------------------------------------------------------------------------
# Ask agent (ReAct)
# ---------------------------------------------------------------------------
ASK_MAX_STEPS = 4
ASK_HISTORY_TURNS = 6         # short-term memory: recent turns sent to the model
MEMORY_MAX_FACTS = 20         # long-term memory size cap
MEMORY_MAX_FACT_CHARS = 200


def memory_path() -> str:
    return os.path.join(UPLOADS_DIR, "teacher_memory.json")
