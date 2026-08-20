"""
Assemble the LangGraph agent.

Without human review:
    START → planner → researcher → critic ─┬─(gaps + under limit)→ researcher  (loop)
                                           └─(complete OR limit)──→ synthesizer → END

With human review (build_graph(review=True)):
    START → planner → human_review ─┬─(approve)→ researcher → … → synthesizer → END
                ▲                   ├─(revise)──→ planner   (re-plan with feedback)
                └───────────────────┘
                                    └─(cancel)──→ END

Two decision points make this an agent rather than a pipeline: the critic's
self-assessment loop (machine reflection) and the optional human gate on the plan.
Interrupts need a checkpointer so the run can be suspended and resumed.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from config import settings
from agent.nodes.critic import critic_node
from agent.nodes.human_review import human_review_node
from agent.nodes.planner import planner_node
from agent.nodes.researcher import researcher_node
from agent.nodes.synthesizer import synthesizer_node
from agent.state import AgentState


def route_after_critic(state: AgentState) -> str:
    """Decide the next node after the critic runs.

    Loop back to the researcher only if the critic found important gaps AND we haven't
    hit the loop cap; otherwise finish. The cap guarantees termination.
    """
    if state.get("complete") or state.get("loops", 0) >= settings.max_critic_loops:
        return "done"
    if state.get("followups"):
        return "research_more"
    return "done"


def route_after_review(state: AgentState) -> str:
    """Route on the human's decision about the plan."""
    if state.get("cancelled"):
        return "cancel"
    if state.get("human_feedback"):
        return "revise"
    return "approve"


def build_graph(review: bool = False):
    """Build and compile the agent graph.

    review=True inserts the human-in-the-loop plan gate (and a checkpointer, which
    LangGraph needs in order to suspend and resume at an interrupt).
    """
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")

    if review:
        graph.add_node("human_review", human_review_node)
        graph.add_edge("planner", "human_review")
        graph.add_conditional_edges(
            "human_review",
            route_after_review,
            {"approve": "researcher", "revise": "planner", "cancel": END},
        )
    else:
        graph.add_edge("planner", "researcher")

    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"research_more": "researcher", "done": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)

    if not review:
        return graph.compile()

    # A checkpointer persists state between the interrupt and the resume. Our state
    # holds Pydantic models, so the serializer must be told they're allowed to be
    # restored from a checkpoint (otherwise LangGraph warns and will later refuse).
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("agent.state", "ResearchPlan"),
            ("agent.state", "SubQuestion"),
            ("agent.state", "CriticVerdict"),
        ]
    )
    return graph.compile(checkpointer=MemorySaver(serde=serde))
