"""Smoke test with the LLM mocked (no Ollama needed).
Run from the project root:  python -m teacher_backend.test_teacher"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from teacher_backend import config, observability
from teacher_backend.api import app
from teacher_backend.test_core import fake_embed

client = TestClient(app)

NOTES = ("Photosynthesis\n" + "Plants make food using sunlight, water and carbon dioxide. " * 12
         + "\n\nRespiration\n" + "Cells release energy from food using oxygen. " * 15)


def fake_text(system, user, timeout=120):
    return "1. Learning goals\n- Understand the topic\n2. Explanation\n..."


def fake_json(system, user, attempts=3):
    if "multiple-choice" in system:
        return {"questions": [
            {"question": "Q1?", "options": ["a", "b", "c", "d"], "correct_index": 2, "explanation": "x"},
            {"question": "bad", "options": ["a"], "correct_index": 5},   # must be filtered out
        ]}
    return {"task": "Write a loop.", "rubric": ["runs", "clear"], "solution": "for i in x: pass"}


def run():
    from teacher_backend import test_core
    test_core.run()
    import tempfile
    config.UPLOADS_DIR = tempfile.mkdtemp()
    ask_script = iter(["Thought: look\nAction: search_notes\nAction Input: sunlight",
                       "Thought: done\nFinal Answer: Plants use sunlight."])
    with patch("teacher_backend.llm_client.call_llm", side_effect=fake_text), \
         patch("teacher_backend.llm_client.call_llm_json", side_effect=fake_json), \
         patch("teacher_backend.llm_client.embed", side_effect=fake_embed), \
         patch("teacher_backend.llm_client.call_llm_messages", side_effect=lambda m, timeout=0: next(ask_script)):

        r = client.post("/api/subjects/from-notes", json={"title": "Biology", "text": NOTES})
        assert r.status_code == 200, r.text
        topics = r.json()["topics"]
        assert len(topics) >= 2, topics
        tid = topics[0]["id"]

        lesson = client.post("/api/generate", json={"kind": "teach", "topic_id": tid}).json()
        assert "Learning goals" in lesson["lesson"]
        assert lesson["sources"] and lesson["retrieval"] in ("faiss", "numpy-cosine", "keyword")   # RAG ran
        assert "score" in lesson["grounding"]

        quiz = client.post("/api/generate", json={"kind": "quiz", "topic_id": tid}).json()
        assert len(quiz["questions"]) == 1 and quiz["questions"][0]["correct_index"] == 2

        code = client.post("/api/generate", json={"kind": "code", "topic_id": tid}).json()
        assert code["task"] and code["solution"]

        subject_id = r.json()["subject"]["id"]
        ask = client.post("/api/ask", json={"subject_id": subject_id, "question": "What do plants use?",
                                            "history": [{"role": "user", "text": "hi"}]}).json()
        assert ask["answer"] == "Plants use sunlight." and ask["steps"][0]["action"] == "search_notes"
        assert client.post("/api/ask", json={"subject_id": "nope", "question": "x"}).status_code == 404
        assert client.post("/api/ask", json={"subject_id": subject_id, "question": ""}).status_code == 422
        with patch("teacher_backend.llm_client.call_llm_messages", return_value="Plants make food from light. Why?") as m:
            t = client.post("/api/teach-chat", json={"topic_id": tid, "message": "How do plants eat?",
                                                    "history": [{"role": "assistant", "text": "Hi!"}]})
            assert t.status_code == 200 and t.json()["reply"].startswith("Plants"), t.text
            sent = m.call_args[0][0]
            assert sent[-1] == {"role": "user", "content": "How do plants eat?"} and sent[1]["content"] == "Hi!"
            assert client.post("/api/teach-chat", json={"topic_id": tid}).status_code == 200   # opening turn
        assert client.post("/api/teach-chat", json={"topic_id": "nope"}).status_code == 404
        assert client.get("/api/memory").json() == {"facts": []}
        assert client.delete("/api/memory").status_code == 200
        names = {t["name"] for t in client.get("/api/traces").json()}
        assert {"generate", "retrieve", "ask"} <= names, names
        system = client.get("/api/system").json()
        assert system["llm"]["model"] and system["embeddings"]["model"]

        assert client.post("/api/generate", json={"kind": "teach", "topic_id": "nope"}).status_code == 400
        assert client.post("/api/generate", json={"kind": "dance", "topic_id": tid}).status_code == 400
        assert client.post("/api/generate", json={"kind": "teach", "topic_id": tid, "language": "Klingon"}).status_code == 400
        assert client.post("/api/subjects/from-notes", json={"title": "x", "text": "too short"}).status_code == 400
        assert client.post("/api/subjects/upload", files={"file": ("x.txt", b"hi")}).status_code == 400
        assert client.get("/api/subjects").status_code == 200
        assert client.get("/api/languages").json() == config.LANGUAGES
        assert client.get("/").status_code == 200
        assert client.get("/static/app.js").status_code == 200

    print("Teacher API tests passed.")


if __name__ == "__main__":
    run()
