"""
Long-term memory for the Ask agent: short facts the teacher wants remembered
across sessions ("my class is Grade 7", "no projector"). Stored in a small JSON
file and injected into the agent's prompt as background data. Short-term memory
is the recent chat history the browser sends with each question.
"""

import json
import os
import threading

from . import config

_lock = threading.Lock()


def load() -> list:
    try:
        with open(config.memory_path()) as f:
            facts = json.load(f)
        return [str(x) for x in facts if isinstance(x, str)][-config.MEMORY_MAX_FACTS:]
    except (OSError, ValueError):
        return []


def _save(facts: list) -> None:
    try:
        os.makedirs(config.UPLOADS_DIR, exist_ok=True)
        with open(config.memory_path(), "w") as f:
            json.dump(facts, f)
    except OSError:
        pass


def add(fact: str) -> bool:
    fact = " ".join((fact or "").split())[: config.MEMORY_MAX_FACT_CHARS]
    if len(fact) < 3:
        return False
    with _lock:
        facts = load()
        if any(fact.lower() == f.lower() for f in facts):
            return True
        facts = (facts + [fact])[-config.MEMORY_MAX_FACTS:]
        _save(facts)
    return True


def clear() -> None:
    with _lock:
        _save([])
