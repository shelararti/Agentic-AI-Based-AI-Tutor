"""
Evaluate the generation pipeline on a subject: success rate, latency, how well
the output is grounded in the notes, and quiz quality -- with RAG on vs. off
(an A/B comparison of the retrieval step).

    python -m teacher_backend.evaluate --subject agentic-ai --kinds teach,quiz --topics 3

Needs Ollama running (it makes real model calls). English output only, because
the grounding metric compares words with the notes.
"""

import argparse
import statistics
import time

from . import config, knowledge_base
from .graph import teacher_graph


def run_once(topic_id: str, kind: str, num_questions: int) -> dict:
    t0 = time.perf_counter()
    r = teacher_graph.invoke({"intent": kind, "topic_id": topic_id, "level": "beginner",
                              "language": "English", "num_questions": num_questions})
    row = {"ok": not r.get("error"), "seconds": round(time.perf_counter() - t0, 1),
           "retrieval": r.get("retrieval_method"), "grounding": (r.get("grounding") or {}).get("score")}
    if kind == "quiz" and row["ok"]:
        qs = r["quiz"]["questions"]
        row["usable_ratio"] = round(len(qs) / num_questions, 2)
        row["flagged"] = sum(1 for q in qs if q.get("grounded") is False)
    return row


def summarize(rows: list) -> dict:
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(statistics.mean(vals), 2) if vals else None
    return {"runs": len(rows), "success_rate": round(sum(r["ok"] for r in rows) / len(rows), 2),
            "avg_seconds": mean("seconds"), "avg_grounding": mean("grounding"),
            "avg_quiz_usable": mean("usable_ratio"), "flagged_questions": sum(r.get("flagged", 0) for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="subject id (see GET /api/subjects)")
    ap.add_argument("--kinds", default="teach,quiz", help="comma list of: teach,quiz,code")
    ap.add_argument("--topics", type=int, default=3, help="how many topics to test")
    ap.add_argument("--questions", type=int, default=5)
    args = ap.parse_args()

    subject = knowledge_base.get_subject(args.subject)
    if not subject:
        raise SystemExit(f"Unknown subject {args.subject!r}. Known: {[s['id'] for s in knowledge_base.all_subjects()]}")
    topics = subject["topics"][: args.topics]
    kinds = [k for k in args.kinds.split(",") if k in ("teach", "quiz", "code")]

    results = {}
    for label, enabled in (("RAG on", True), ("RAG off", False)):
        config.RAG_ENABLED = enabled
        rows = [run_once(t["id"], k, args.questions) for t in topics for k in kinds]
        results[label] = summarize(rows)
        print(f"{label:8s}", results[label])
    on, off = results["RAG on"]["avg_grounding"], results["RAG off"]["avg_grounding"]
    if on is not None and off is not None:
        print(f"\nGrounding with RAG vs without: {on} vs {off} (higher = output stays closer to your notes)")


if __name__ == "__main__":
    main()
