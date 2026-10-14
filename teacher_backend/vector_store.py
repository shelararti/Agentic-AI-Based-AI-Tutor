"""
Retrieval for RAG: chunk a subject's notes, embed the chunks, search them.

Search order of preference (the app degrades instead of breaking):
  1. FAISS inner-product index over normalized embeddings (if `faiss-cpu` is installed)
  2. plain NumPy cosine similarity over the same embeddings
  3. keyword (BM25-style) scoring, if the embedding model isn't reachable
`retrieve()` reports which method was used so the UI/traces can show it.

Embeddings of uploaded subjects are cached in <uploads>/<subject_id>.embeddings.json
so a restart doesn't re-embed everything. Built-in subjects are indexed lazily
in memory on first use.
"""

import hashlib
import json
import math
import os
import re
import time
from collections import Counter

import numpy as np

from . import config, llm_client, observability, pdf_ingest

try:  # optional accelerator
    import faiss
except ImportError:
    faiss = None

_TOKEN_RE = re.compile(r"[^\s.,;:!?()\[\]{}\"'“”‘’।|/\\\-]+")
_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+")
_EMBED_BATCH = 32
_RETRY_EMBED_AFTER_S = 60

_INDEXES: dict = {}


def tokenize(text: str) -> list:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def faiss_available() -> bool:
    return faiss is not None


# ---- chunking ---------------------------------------------------------------
def _split_big(text: str, size: int) -> list:
    """Split one oversized paragraph on sentence boundaries."""
    out, cur = [], ""
    for sent in _SENT_SPLIT.split(text):
        if cur and len(cur) + len(sent) > size:
            out.append(cur.strip())
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        out.append(cur.strip())
    return out


def _chunk(body: str) -> list:
    size = config.RAG_CHUNK_CHARS
    chunks = []
    for c in pdf_ingest._chunk_long_text(body, size):
        chunks.extend(_split_big(c, size) if len(c) > size * 1.5 else [c])
    return [c for c in chunks if len(c.strip()) > 40]


def chunks_for(subject: dict) -> list:
    """[{id, topic_id, topic_title, text}] for a subject. Uploaded subjects keep
    their full text (`corpus`) so retrieval covers more than the topic notes."""
    topics = subject.get("topics", [])
    by_title = {t["title"]: t["id"] for t in topics}

    def topic_for(title: str):
        if title in by_title:
            return by_title[title]
        return next((t["id"] for t in topics if t["title"].startswith(title)), None)

    raw = []   # (topic_id, topic_title, text)
    if subject.get("corpus"):
        for title, lines in pdf_ingest._split_into_sections(subject["corpus"]):
            heading = title or "Overview"
            for c in _chunk("\n".join(lines)):
                raw.append((topic_for(heading), heading, c))
    else:
        for t in topics:
            for c in _chunk(t.get("notes", "")):
                raw.append((t["id"], t["title"], c))
    return [{"id": i, "topic_id": tid, "topic_title": ttl, "text": txt} for i, (tid, ttl, txt) in enumerate(raw)]


# ---- index build / cache -----------------------------------------------------
def _cache_path(subject_id: str) -> str:
    safe = "".join(c for c in subject_id if c.isalnum() or c in "-_") or "subject"
    return os.path.join(config.UPLOADS_DIR, f"{safe}.embeddings.json")


def _fingerprint(chunks: list) -> str:
    h = hashlib.md5(config.EMBED_MODEL.encode())
    for c in chunks:
        h.update(c["text"].encode("utf-8"))
    return h.hexdigest()


def _normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


def _load_cache(subject: dict, chunks: list):
    if subject.get("source") != "uploaded":
        return None
    try:
        with open(_cache_path(subject["id"])) as f:
            data = json.load(f)
        if data.get("fingerprint") == _fingerprint(chunks) and len(data["vectors"]) == len(chunks):
            return np.array(data["vectors"], dtype=np.float32)
    except (OSError, ValueError, KeyError):
        pass
    return None


def _save_cache(subject: dict, chunks: list, vectors: np.ndarray) -> None:
    if subject.get("source") != "uploaded":
        return
    try:
        os.makedirs(config.UPLOADS_DIR, exist_ok=True)
        with open(_cache_path(subject["id"]), "w") as f:
            json.dump({"fingerprint": _fingerprint(chunks), "model": config.EMBED_MODEL,
                       "vectors": vectors.tolist()}, f)
    except OSError:
        pass


class Index:
    def __init__(self, subject: dict):
        self.chunks = chunks_for(subject)
        self.vectors = None
        self.faiss = None
        self.embed_error = None
        self.built_at = time.time()
        self.tokens = [Counter(tokenize(c["text"])) for c in self.chunks]
        df = Counter()
        for t in self.tokens:
            df.update(t.keys())
        self.df = df
        if not self.chunks:
            return
        vectors = _load_cache(subject, self.chunks)
        if vectors is None:
            try:
                rows = []
                for i in range(0, len(self.chunks), _EMBED_BATCH):
                    rows += llm_client.embed([c["text"] for c in self.chunks[i:i + _EMBED_BATCH]])
                vectors = _normalize(np.array(rows, dtype=np.float32))
                _save_cache(subject, self.chunks, vectors)
            except (RuntimeError, ValueError) as e:
                self.embed_error = str(e)
                return
        self.vectors = vectors
        if faiss is not None:
            self.faiss = faiss.IndexFlatIP(vectors.shape[1])
            self.faiss.add(vectors)

    @property
    def method(self) -> str:
        if self.vectors is None:
            return "keyword"
        return "faiss" if self.faiss is not None else "numpy-cosine"


def _get_index(subject: dict) -> Index:
    idx = _INDEXES.get(subject["id"])
    stale = idx is not None and idx.vectors is None and idx.chunks \
        and time.time() - idx.built_at > _RETRY_EMBED_AFTER_S
    if idx is None or stale:
        with observability.span("build_index", subject=subject["id"]) as meta:
            idx = Index(subject)
            meta.update(chunks=len(idx.chunks), method=idx.method, embed_error=idx.embed_error)
        _INDEXES[subject["id"]] = idx
    return idx


def drop(subject_id: str) -> None:
    _INDEXES.pop(subject_id, None)
    try:
        os.remove(_cache_path(subject_id))
    except OSError:
        pass


# ---- search ------------------------------------------------------------------
def _keyword_scores(idx: Index, query: str) -> np.ndarray:
    n = max(len(idx.chunks), 1)
    q = set(tokenize(query))
    scores = np.zeros(len(idx.chunks), dtype=np.float32)
    for i, tf in enumerate(idx.tokens):
        total = sum(tf.values()) or 1
        for term in q:
            if term in tf:
                idf = math.log(1 + (n - idx.df[term] + 0.5) / (idx.df[term] + 0.5))
                scores[i] += idf * (tf[term] * 2.2) / (tf[term] + 1.2 * (0.25 + 0.75 * total / 100))
    top = scores.max() if len(scores) else 0
    return scores / top if top > 0 else scores     # scale to 0..1 so the topic boost is comparable


def retrieve(subject: dict, query: str, k: int = None, prefer_topic_id: str = None) -> dict:
    """-> {"method": str, "hits": [{chunk_id, topic_id, topic_title, text, score}]}"""
    k = k or config.RAG_TOP_K
    idx = _get_index(subject)
    if not idx.chunks:
        return {"method": "none", "hits": []}

    method, scores = idx.method, None
    if idx.vectors is not None:
        try:
            q = _normalize(np.array(llm_client.embed([query]), dtype=np.float32))
            if idx.faiss is not None:
                d, i = idx.faiss.search(q, len(idx.chunks))
                scores = np.zeros(len(idx.chunks), dtype=np.float32)
                scores[i[0]] = d[0]
            else:
                scores = (idx.vectors @ q[0]).astype(np.float32)
        except (RuntimeError, ValueError):
            method, scores = "keyword", None       # embedding service went away: degrade
    if scores is None:
        method, scores = "keyword", _keyword_scores(idx, query)

    if prefer_topic_id:
        scores = scores + np.array(
            [config.RAG_SAME_TOPIC_BOOST if c["topic_id"] == prefer_topic_id else 0 for c in idx.chunks],
            dtype=np.float32)
    order = np.argsort(-scores)[:k]
    hits = [dict(chunk_id=idx.chunks[j]["id"], topic_id=idx.chunks[j]["topic_id"],
                 topic_title=idx.chunks[j]["topic_title"], text=idx.chunks[j]["text"],
                 score=round(float(scores[j]), 3))
            for j in order if scores[j] > 0]
    return {"method": method, "hits": hits}


def format_context(hits: list) -> str:
    """Numbered, capped text block to put in a prompt."""
    parts, used = [], 0
    for n, h in enumerate(hits, start=1):
        piece = f"[{n}] ({h['topic_title']}) {h['text']}"
        if used + len(piece) > config.RAG_CONTEXT_CHARS and parts:
            break
        parts.append(piece)
        used += len(piece)
    return "\n\n".join(parts)
