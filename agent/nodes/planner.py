"""
Planner node — the first step of the agent.

It reads the user's markets question and decomposes it into a small set of focused,
answerable sub-questions, each tagged with the kind of data that answers it (price,
technicals, fundamentals, news, macro). Later nodes use those tags to pick the right
tool. Breaking a broad question into targeted sub-questions is what lets the agent
research thoroughly instead of answering in one vague shot.
"""

from config import settings
from agent.llm import get_llm
from agent.state import AgentState, ResearchPlan


PLANNER_SYSTEM = (
    "You are the planning step of a markets research analyst for crypto and US "
    "stocks. Given a user's question, break it into at most {max_q} focused, "
    "answerable sub-questions. Identify any tickers (e.g. BTC, ETH, NVDA, AAPL). "
    "Tag each sub-question with the data type that answers it: price, technicals, "
    "fundamentals (valuation ratios like P/E and PEG), earnings (quarterly results: "
    "EPS vs estimates, revenue growth, margins — STOCKS ONLY, never crypto), news, or "
    "macro, and list the specific ticker(s) each sub-question concerns (empty for "
    "macro). Cover the angles a careful analyst would: recent price/trend, technical "
    "setup, and for stocks BOTH valuation fundamentals AND recent quarterly earnings "
    "performance, plus news/catalysts and broader market context. Be specific and "
    "avoid overlap."
)


def planner_node(state: AgentState) -> dict:
    """Produce a structured ResearchPlan from the user's question."""
    question = state["question"]

    # with_structured_output makes Gemini return a validated ResearchPlan object
    # (not free text we'd have to parse), so downstream nodes get clean fields.
    llm = get_llm().with_structured_output(ResearchPlan)

    system = PLANNER_SYSTEM.format(max_q=settings.max_subquestions)

    # If a human reviewed a previous plan and asked for changes, fold that guidance in.
    feedback = state.get("human_feedback", "")
    prompt = f"Research question:\n{question}"
    if feedback:
        prompt += (
            f"\n\nA human reviewed your previous plan and asked for these changes:\n"
            f"{feedback}\n\nProduce a revised plan that follows this guidance."
        )

    plan: ResearchPlan = llm.invoke([("system", system), ("human", prompt)])

    # Trim to the configured maximum in case the model over-produces.
    plan.subquestions = plan.subquestions[: settings.max_subquestions]
    return {"plan": plan, "loops": 0}
