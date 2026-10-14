"""Guardrail node: after generation, score how well the output is supported by
the retrieved notes and flag weakly supported quiz questions."""

from .. import grounding


def grounding_node(state: dict) -> dict:
    if state.get("error"):
        return {}
    if state.get("language", "English") != "English":
        return {"grounding": {"score": None, "label": "check is only available for English output"}}

    context = state.get("context", "")
    if state.get("lesson"):
        s = grounding.support(state["lesson"], context)
        return {"grounding": {"score": s, "label": grounding.label(s)}}

    if state.get("quiz"):
        questions, scores = [], []
        for q in state["quiz"]["questions"]:
            s = grounding.support(f"{q['question']} {q['options'][q['correct_index']]}", context)
            scores.append(s)
            questions.append({**q, "grounded": s is None or s >= grounding.GROUNDED_AT})
        known = [s for s in scores if s is not None]
        overall = round(sum(known) / len(known), 2) if known else None
        return {"quiz": {"questions": questions}, "grounding": {"score": overall, "label": grounding.label(overall)}}
    return {}
