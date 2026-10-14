"""
Observability for the LLM pipeline.

Two layers:
1. A local trace log that is ALWAYS on: every request becomes a trace and every
   retrieval / LLM call inside it becomes a span with latency and metadata.
   Visible at GET /api/traces (handy for a live demo, no account needed).
2. Langfuse, if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set: LiteLLM's
   Langfuse callback is switched on, so every LLM generation (prompt, output,
   tokens, latency) shows up in the Langfuse dashboard grouped by trace_id.
"""

import contextvars
import os
import time
import uuid
from collections import deque
from contextlib import contextmanager

from . import config

_TRACES = deque(maxlen=config.TRACE_BUFFER)     # finished spans, newest last
_trace_id = contextvars.ContextVar("trace_id", default=None)
_langfuse_state = {"enabled": False, "error": None}


def langfuse_configured() -> bool:
    return bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))


def setup_langfuse() -> None:
    """Turn on LiteLLM's Langfuse callback. Safe to call when not configured."""
    if _langfuse_state["enabled"] or not langfuse_configured():
        return
    try:
        import litellm
        litellm.success_callback = list({*litellm.success_callback, "langfuse"})
        litellm.failure_callback = list({*litellm.failure_callback, "langfuse"})
        _langfuse_state["enabled"] = True
    except Exception as e:  # litellm/langfuse missing or incompatible: keep running
        _langfuse_state["error"] = str(e)


def status() -> dict:
    return {"local_traces": True, "langfuse_configured": langfuse_configured(),
            "langfuse_enabled": _langfuse_state["enabled"], "langfuse_error": _langfuse_state["error"]}


def current_trace_id():
    return _trace_id.get()


@contextmanager
def span(name: str, **meta):
    """Time a block. The yielded dict can be updated with more metadata.
    The outermost span in a request starts a new trace."""
    started_trace = _trace_id.get() is None
    token = _trace_id.set(uuid.uuid4().hex[:12]) if started_trace else None
    record = {"trace_id": _trace_id.get(), "name": name, "root": started_trace,
              "started": time.time(), "meta": dict(meta), "error": None}
    t0 = time.perf_counter()
    try:
        yield record["meta"]
    except Exception as e:
        record["error"] = str(e)[:300]
        raise
    finally:
        record["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        _TRACES.append(record)
        if token is not None:
            _trace_id.reset(token)


def recent(limit: int = 50) -> list:
    return list(_TRACES)[-limit:][::-1]


def clear() -> None:
    _TRACES.clear()
