"""
Assemble the LangGraph agent.

For this first milestone the graph is just: START → planner → END. As we add the
Researcher, Critic (with its loop-back), and Synthesizer nodes, they slot into this
same graph — that incremental, inspectable structure is the point of using LangGraph
instead of a hand-rolled loop.
"""

from langgraph.graph import END, START, StateGraph

from agent.nodes.planner import planner_node
from agent.nodes.researcher import researcher_node
from agent.state import AgentState


def build_graph():
    """Build and compile the agent graph."""
    graph = StateGraph(AgentState)

    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", END)

    return graph.compile()
