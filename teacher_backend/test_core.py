"""Core-logic tests (RAG, grounding, ReAct, memory, tracing, LLM backend). LLM and
embeddings are mocked, and no web framework is needed.
Run from the project root:  python -m teacher_backend.test_core"""

import hashlib
import tempfile
from unittest.mock import patch

import numpy as np

from teacher_backend import config, grounding, knowledge_base, llm_client, memory, observability, pdf_ingest, tools, vector_store
from teacher_backend.agents import ask_agent, grounding_check, retriever

NOTES = ("Photosynthesis\n" + "Plants make food using sunlight, water and carbon dioxide inside chloroplasts. " * 8
         + "\n\nRespiration\n" + "Cells release energy from glucose using oxygen in the mitochondria. " * 8
         + "\n\nCell division\n" + "Mitosis copies a cell into two identical daughter cells with the same chromosomes. " * 8)


def fake_embed(texts):
    """Deterministic bag-of-words hashing embedding: similar words -> similar vectors."""
    out = []
    for t in texts:
        v = np.zeros(64)
        for w in vector_store.tokenize(t):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        out.append(v.tolist())
    return out


def make_subject(tmp):
    config.UPLOADS_DIR = tmp
    topics = pdf_ingest.text_to_topics(NOTES, "bio")
    return knowledge_base.add_subject("bio", "Biology", "x", topics, corpus=NOTES), topics


def test_retrieval():
    with tempfile.TemporaryDirectory() as tmp:
        subject, topics = make_subject(tmp)
        vector_store.drop("bio")
        with patch("teacher_backend.llm_client.embed", side_effect=fake_embed):
            r = vector_store.retrieve(subject, "how do cells get energy from oxygen", k=2)
            assert r["method"] in ("faiss", "numpy-cosine"), r["method"]
            assert r["hits"][0]["topic_title"] == "Respiration", [h["topic_title"] for h in r["hits"]]
            assert "[1] (Respiration)" in vector_store.format_context(r["hits"])
            # cache written for uploaded subjects, and reused (no embed calls) on a fresh index
            vector_store._INDEXES.clear()
        with patch("teacher_backend.llm_client.embed", side_effect=AssertionError("cache not used")) as m:
            idx = vector_store.Index(subject)
            assert idx.vectors is not None
        # Embeddings down -> keyword fallback still finds the right section
        vector_store.drop("bio")
        with patch("teacher_backend.llm_client.embed", side_effect=RuntimeError("no model")):
            r = vector_store.retrieve(subject, "mitosis daughter cells", k=2)
            assert r["method"] == "keyword" and r["hits"][0]["topic_title"] == "Cell division", r
        # Embeddings that die mid-flight (index built, query embed fails) also degrade
        vector_store.drop("bio")
        with patch("teacher_backend.llm_client.embed", side_effect=fake_embed):
            vector_store.retrieve(subject, "x y z", k=1)
        with patch("teacher_backend.llm_client.embed", side_effect=RuntimeError("gone")):
            assert vector_store.retrieve(subject, "mitosis cells", k=1)["method"] == "keyword"
        vector_store.drop("bio")


def test_retriever_and_grounding():
    with tempfile.TemporaryDirectory() as tmp:
        subject, topics = make_subject(tmp)
        vector_store.drop("bio")
        t = next(x for x in topics if x["title"] == "Photosynthesis")
        state = {"intent": "teach", "topic_id": t["id"], "subject_id": "bio", "topic_title": t["title"],
                 "topic_notes": t["notes"], "language": "English"}
        with patch("teacher_backend.llm_client.embed", side_effect=fake_embed):
            out = retriever.retriever_node(state)
        assert out["retrieval_method"] != "full-notes" and out["sources"] and out["context"]
        assert retriever.route_after_retriever({"intent": "quiz"}) == "quiz_agent"

        # Unknown subject -> falls back to the topic's own notes instead of failing
        out2 = retriever.retriever_node({**state, "subject_id": "nope"})
        assert out2["retrieval_method"] == "full-notes" and out2["context"] == t["notes"]

        ctx = out["context"]
        good = "Plants make food from sunlight, water and carbon dioxide in chloroplasts."
        bad = "Volcanoes erupt molten magma through tectonic fissures producing basalt."
        assert grounding.support(good, ctx) >= 0.6 and grounding.support(bad, ctx) < 0.2
        assert grounding.support("hi", ctx) is None

        quiz = {"questions": [
            {"question": "What do plants use to make food?", "options": ["sunlight and water", "magma", "x", "y"], "correct_index": 0},
            {"question": "What erupts from volcanoes producing basalt?", "options": ["molten magma", "a", "b", "c"], "correct_index": 0}]}
        res = grounding_check.grounding_node({"language": "English", "quiz": quiz, "context": ctx})
        assert [q["grounded"] for q in res["quiz"]["questions"]] == [True, False], res
        assert grounding_check.grounding_node({"language": "Hindi", "lesson": "x", "context": ctx})["grounding"]["score"] is None
        assert grounding_check.grounding_node({"error": "boom"}) == {}


def test_tools_and_memory():
    assert tools.calculator("(12*3)/4") == "9.0" or tools.calculator("(12*3)/4") == "9"
    assert tools.calculator("2^3") == "8"
    for evil in ("__import__('os').system('ls')", "open('x')", "2**9999", "1/0", "a+1", ""):
        assert tools.calculator(evil).startswith("Error"), evil
    with tempfile.TemporaryDirectory() as tmp:
        config.UPLOADS_DIR = tmp
        assert memory.add("Teaches Grade 7") and memory.add("teaches grade 7")   # duplicate ignored
        assert memory.load() == ["Teaches Grade 7"]
        assert not memory.add("hi")
        for i in range(30):
            memory.add(f"fact number {i}")
        assert len(memory.load()) == config.MEMORY_MAX_FACTS
        memory.clear()
        assert memory.load() == []


def test_react_agent():
    with tempfile.TemporaryDirectory() as tmp:
        subject, _ = make_subject(tmp)
        vector_store.drop("bio")
        script = iter([
            "Thought: I should look this up.\nAction: search_notes\nAction Input: mitochondria energy\nObservation: made up",
            "Thought: And remember the class.\nAction: remember\nAction Input: Teaches Grade 7",
            "Thought: Now I can answer.\nFinal Answer: Cells get energy from glucose using oxygen in mitochondria.",
        ])
        seen = []

        def fake_llm(messages, timeout=0):
            seen.append([dict(m) for m in messages])
            return next(script)

        with patch("teacher_backend.llm_client.embed", side_effect=fake_embed), \
             patch("teacher_backend.llm_client.call_llm_messages", side_effect=fake_llm):
            r = ask_agent.run(subject, "How do cells get energy?", "English", [{"role": "user", "text": "hello"}, {"role": "assistant", "text": "hi"}])
        assert "mitochondria" in r["answer"] and [s["action"] for s in r["steps"]] == ["search_notes", "remember"], r
        assert "Respiration" in r["sources"] and memory.load() == ["Teaches Grade 7"]
        assert "made up" not in str(seen[1])                       # hallucinated Observation was dropped
        assert seen[0][1]["content"] == "hello" and seen[0][-1]["content"] == "How do cells get energy?"   # short-term memory
        # long-term memory is injected into the next conversation's system prompt
        with patch("teacher_backend.llm_client.call_llm_messages", return_value="Final Answer: ok") as m:
            ask_agent.run(subject, "q", "English", [])
        assert "Teaches Grade 7" in m.call_args[0][0][0]["content"]

        # model ignores the format -> its text is the answer; unknown tool is reported; step cap forces an answer
        with patch("teacher_backend.llm_client.call_llm_messages", return_value="Just plain text."):
            assert ask_agent.run(subject, "q", "English", [])["answer"] == "Just plain text."
        loop = iter(["Action: teleport\nAction Input: x"] * config.ASK_MAX_STEPS + ["Final Answer: gave up"])
        with patch("teacher_backend.llm_client.call_llm_messages", side_effect=lambda m, timeout=0: next(loop)):
            r = ask_agent.run(subject, "q", "English", [])
        assert r["answer"] == "gave up" and len(r["steps"]) == config.ASK_MAX_STEPS and "Unknown tool" in r["steps"][0]["observation"]
        memory.clear()
        vector_store.drop("bio")


def test_llm_backend_and_tracing():
    observability.clear()
    assert llm_client.strip_thinking("<think>hmm</think>\nAnswer").strip() == "Answer"
    assert llm_client.strip_thinking("<think>unfinished").strip() == ""
    with patch.object(config, "LLM_BACKEND", "ollama"):
        assert llm_client.backend_info()["backend"] == "ollama (direct)"
        with patch("teacher_backend.llm_client._complete_ollama", return_value=("hello ", {"prompt": 3, "completion": 1})):
            with observability.span("request") as meta:
                assert llm_client.call_llm("s", "u") == "hello"
    spans = observability.recent()
    assert {s["name"] for s in spans} == {"request", "llm_call"}
    assert len({s["trace_id"] for s in spans}) == 1                 # child shares the request's trace
    assert next(s for s in spans if s["name"] == "llm_call")["meta"]["tokens"]["completion"] == 1
    with patch.object(config, "LLM_BACKEND", "litellm"):            # litellm required but missing -> clear error
        try:
            llm_client._litellm()
        except RuntimeError as e:
            assert "litellm" in str(e)
        except ImportError:
            pass
    assert observability.status()["local_traces"] is True


def run():
    for t in (test_retrieval, test_retriever_and_grounding, test_tools_and_memory, test_react_agent, test_llm_backend_and_tracing):
        t()
    print("Core tests passed (RAG, grounding, ReAct, memory, tracing).")


if __name__ == "__main__":
    run()
