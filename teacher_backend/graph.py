"""Teacher agent graph:

  Orchestrator -> Retriever (RAG) -> Teacher | Quiz | Coding -> Grounding check -> END
"""

from langgraph.graph import StateGraph, START, END

from .state import TeacherState
from .agents.orchestrator import orchestrator_node, route_after_orchestrator
from .agents.retriever import retriever_node, route_after_retriever
from .agents.teacher import teacher_node
from .agents.quiz import quiz_node
from .agents.coding import coding_node
from .agents.grounding_check import grounding_node

GENERATORS = ("teacher_agent", "quiz_agent", "coding_agent")


def build_graph():
    b = StateGraph(TeacherState)
    b.add_node("orchestrator", orchestrator_node)
    b.add_node("retriever", retriever_node)
    b.add_node("teacher_agent", teacher_node)
    b.add_node("quiz_agent", quiz_node)
    b.add_node("coding_agent", coding_node)
    b.add_node("grounding_check", grounding_node)

    b.add_edge(START, "orchestrator")
    b.add_conditional_edges("orchestrator", route_after_orchestrator, {"retriever": "retriever", "end": END})
    b.add_conditional_edges("retriever", route_after_retriever, {g: g for g in GENERATORS})
    for node in GENERATORS:
        b.add_edge(node, "grounding_check")
    b.add_edge("grounding_check", END)
    return b.compile()


teacher_graph = build_graph()
