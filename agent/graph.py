"""
Assemble the LangGraph agent.

    START → planner → researcher → critic ─┬─(gaps + under limit)→ researcher  (loop)
                                           └─(complete OR limit)──→ END

The conditional edge after the critic is what makes this a real agent: it *decides*
whether to research more or stop, based on the critic's verdict and a loop guard.
(The Synthesizer will sit on the exit edge in the next milestone; for now "done" ends.)
"""

from langgraph.graph import END, START, StateGraph

from config import settings
from agent.nodes.critic import critic_node
from agent.nodes.planner import planner_node
from agent.nodes.researcher import researcher_node
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


def build_graph():
    """Build and compile the agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"research_more": "researcher", "done": END},
    )

    return graph.compile()
