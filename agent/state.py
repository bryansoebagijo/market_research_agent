"""
The shared state that flows through the graph.

Every node reads from and writes to this single object. LangGraph threads it from
node to node, so the Planner's output becomes the Researcher's input, and so on.
Keeping all state in one typed place is what makes the agent's run inspectable.
"""

from operator import add
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """One focused research sub-question the planner produced."""

    question: str = Field(description="A specific, answerable research question.")
    focus: str = Field(
        description="What kind of data answers it: 'price', 'technicals', "
        "'fundamentals', 'news', or 'macro'."
    )


class ResearchPlan(BaseModel):
    """The planner's structured decomposition of the user's question."""

    tickers: list[str] = Field(
        default_factory=list,
        description="Crypto or stock symbols mentioned or implied (e.g. BTC, NVDA).",
    )
    subquestions: list[SubQuestion] = Field(
        description="3-5 focused sub-questions that together answer the user's ask."
    )
    rationale: str = Field(description="One sentence on the overall research angle.")


class Evidence(TypedDict):
    """A single finding gathered by the researcher, with where it came from."""

    subquestion: str
    finding: str
    source: str


class AgentState(TypedDict, total=False):
    """The full state object threaded through the graph."""

    question: str                 # the user's original ask
    plan: ResearchPlan            # produced by the planner
    # `add` reducer: researcher nodes append evidence instead of overwriting it.
    evidence: Annotated[list[Evidence], add]
    critique: str                 # the critic's assessment
    loops: int                    # how many times the critic has sent work back
    report: str                   # the final synthesized answer
