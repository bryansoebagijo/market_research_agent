"""
Human-in-the-loop node — pause and let a person approve the research plan.

This sits between the planner and the researcher, which is the moment where human
input is worth the most: the plan decides what the agent will spend its API calls and
time on, so catching a wrong angle here is much cheaper than after the research runs.

It uses LangGraph's `interrupt()`, which suspends the graph mid-run and hands control
back to the caller. The run's state is saved by the checkpointer, so the caller can
show the plan to a person, collect a decision, and resume exactly where it left off by
passing a `Command(resume=...)`.

The decision the caller sends back is a dict:
    {"action": "approve"}                          -> research the plan as-is
    {"action": "revise", "feedback": "focus on …"} -> re-plan with that guidance
    {"action": "cancel"}                           -> stop without researching
"""

from langgraph.types import interrupt

from agent.state import AgentState


def human_review_node(state: AgentState) -> dict:
    """Suspend the run and wait for a human decision on the plan."""
    plan = state["plan"]

    # Everything in this payload is shown to the human by the caller (main.py).
    decision = interrupt(
        {
            "kind": "plan_review",
            "question": state["question"],
            "rationale": plan.rationale,
            "tickers": plan.tickers,
            "subquestions": [
                {"question": sq.question, "focus": sq.focus, "tickers": sq.tickers}
                for sq in plan.subquestions
            ],
        }
    )

    # `decision` is whatever the caller resumed with.
    if not isinstance(decision, dict):
        decision = {"action": "approve"}
    action = decision.get("action", "approve")

    if action == "cancel":
        return {"cancelled": True}
    if action == "revise":
        # Feedback goes back to the planner, which re-plans with it in mind.
        return {"human_feedback": decision.get("feedback", "").strip()}
    return {"human_feedback": ""}  # approved: clear any earlier feedback
