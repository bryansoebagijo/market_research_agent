"""
The shared state that flows through the graph.

Every node reads from and writes to this single object. LangGraph threads it from
node to node, so the Planner's output becomes the Researcher's input, and so on.
Keeping all state in one typed place is what makes the agent's run inspectable.
"""

from operator import add
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field


def merge_metrics(left: dict, right: dict) -> dict:
    """Reducer for `metrics`: merge per-ticker dicts instead of overwriting them.

    A second research pass (after the critic loops back) should add to what the first
    pass measured, not replace it.
    """
    merged = {**(left or {})}
    for ticker, values in (right or {}).items():
        merged[ticker] = {**merged.get(ticker, {}), **values}
    return merged


class SubQuestion(BaseModel):
    """One focused research sub-question the planner produced."""

    question: str = Field(description="A specific, answerable research question.")
    focus: str = Field(
        description="What kind of data answers it: 'price', 'technicals', "
        "'fundamentals' (valuation ratios), 'earnings' (quarterly results: EPS vs "
        "estimates, revenue growth, margins), 'flows' (US spot ETF demand — BTC and "
        "ETH only), 'news', or 'macro'."
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


class Trigger(BaseModel):
    """A machine-checkable condition that would invalidate the thesis.

    Prose invalidation conditions read well but can't be monitored. Forcing a measurable
    threshold alongside them is what makes a thesis trackable — and it's better
    discipline anyway: "margins might compress" becomes "operating margin below 50%".
    """

    metric: str = Field(
        description="One of: rsi, price, revenue_growth_3m, tvl_growth_3m, "
        "revenue_to_tvl, revenue_yoy, eps_yoy, operating_margin, net_margin. "
        "Use only metrics relevant to the asset type — the *_yoy and *_margin metrics "
        "are stocks-only, revenue_growth_3m/tvl_growth_3m/revenue_to_tvl crypto-only."
    )
    operator: str = Field(description="Either '<' or '>'.")
    threshold: float = Field(description="The numeric level that would trip it.")
    description: str = Field(description="Plain-language statement of the condition.")


class AnalystView(BaseModel):
    """A long-horizon value-investor read on the evidence — informational only."""

    stance: str = Field(
        description="One of: Accumulate, Hold, Trim, Avoid, Watch. Per ticker if the "
        "question covers several, e.g. 'BTC: Watch | NVDA: Hold'."
    )
    conviction: str = Field(description="Low, Medium, or High — how strong the evidence is.")
    horizon: str = Field(
        description="Time frame this view applies to. Default to years (3-5+) unless the "
        "question explicitly asks about a shorter period."
    )
    quality: str = Field(
        description="The business's durability and earnings power in 1-2 sentences: "
        "margins, returns on equity, growth consistency. For crypto, assess the network "
        "and adoption instead and say plainly that company-quality metrics don't apply."
    )
    valuation: str = Field(
        description="Whether the current price looks attractive relative to earnings "
        "power and growth (P/E, forward P/E, PEG, growth rate), in 1-2 sentences — and "
        "whether there is any margin of safety. For crypto, say that intrinsic-value "
        "methods do not apply and explain what you are judging instead."
    )
    reasoning: list[str] = Field(
        description="2-4 short points justifying the stance, each grounded in the evidence."
    )
    what_would_change_it: list[str] = Field(
        description="1-3 concrete conditions that would invalidate or flip this view."
    )
    triggers: list[Trigger] = Field(
        default_factory=list,
        description="Up to 3 of those conditions expressed as measurable thresholds, so "
        "they can be monitored automatically. Only include a condition here if it can "
        "honestly be reduced to a number — leave qualitative ones out rather than "
        "inventing a threshold for them.",
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
    # Structured numbers behind the findings, keyed by ticker, for charting in the UI.
    metrics: Annotated[dict, merge_metrics]
    critique: str                 # the critic's assessment
    complete: bool                # critic says coverage is sufficient
    followups: list[SubQuestion]  # critic's targeted gap questions (drives the loop)
    loops: int                    # how many critic passes have run (loop guard)
    report: str                   # the final synthesized answer
    analyst_view: AnalystView     # professional-style read on what the evidence implies
    # --- human-in-the-loop ---
    human_feedback: str           # guidance from a human, fed back into the planner
    cancelled: bool               # human stopped the run at plan review
