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
        "'fundamentals' (valuation ratios), 'earnings' (quarterly results: EPS vs "
        "estimates, revenue growth, margins), 'news', or 'macro'."
    )
    tickers: list[str] = Field(
        default_factory=list,
        description="The specific ticker(s) this sub-question concerns (a subset of "
        "the plan's tickers). Leave empty for general/macro questions.",
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


class CriticVerdict(BaseModel):
    """The critic's judgement of whether the evidence is good enough."""

    complete: bool = Field(
        description="True if the evidence sufficiently answers the user's question."
    )
    assessment: str = Field(
        description="One or two sentences on coverage quality and any important gaps."
    )
    followups: list[SubQuestion] = Field(
        default_factory=list,
        description="If not complete, up to 2 targeted follow-up sub-questions "
        "(with focus + tickers) to fill the biggest gaps.",
    )


class AnalystView(BaseModel):
    """A professional-style read on what the evidence implies — informational only."""

    stance: str = Field(
        description="One of: Accumulate, Hold, Trim, Avoid, Watch. Per ticker if the "
        "question covers several, e.g. 'BTC: Watch | NVDA: Hold'."
    )
    conviction: str = Field(description="Low, Medium, or High — how strong the evidence is.")
    horizon: str = Field(description="Time frame this view applies to, e.g. 'weeks to months'.")
    reasoning: list[str] = Field(
        description="2-4 short points justifying the stance, each grounded in the evidence."
    )
    what_would_change_it: list[str] = Field(
        description="1-3 concrete conditions that would invalidate or flip this view."
    )
    key_risks: list[str] = Field(description="1-3 main risks to this view.")


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
    complete: bool                # critic says coverage is sufficient
    followups: list[SubQuestion]  # critic's targeted gap questions (drives the loop)
    loops: int                    # how many critic passes have run (loop guard)
    report: str                   # the final synthesized answer
    analyst_view: AnalystView     # professional-style read on what the evidence implies
    # --- human-in-the-loop ---
    human_feedback: str           # guidance from a human, fed back into the planner
    cancelled: bool               # human stopped the run at plan review
