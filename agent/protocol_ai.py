"""
LLM explanation of *why* a protocol's revenue or TVL moved.

The numbers say what happened; only outside context says why. So this pairs the computed
history with a live web search about the protocol and asks the model to explain the
recent move — with a hard rule that it must say when the evidence doesn't actually
contain a cause, rather than inventing a plausible-sounding one. A confident wrong reason
is worse than an honest "the data doesn't say".

Kept out of market_data.py so the dashboard's core panels stay LLM-free and instant.
"""

from pydantic import BaseModel, Field

from agent.llm import get_llm
from agent.tools import web_search


class ProtocolExplanation(BaseModel):
    """The model's read on a protocol's recent trajectory."""

    assessment: str = Field(
        description="2-3 sentences on whether this looks like a healthy protocol, based "
        "on the revenue and TVL trends and how they relate to each other."
    )
    drivers: list[str] = Field(
        description="1-3 specific reasons the last 3 months moved as they did, each "
        "grounded in the news provided. If the news does not explain the move, say so "
        "explicitly in one item instead of speculating."
    )
    watch_next: list[str] = Field(
        description="1-2 concrete things that would confirm or reverse the current trend."
    )


SYSTEM = (
    "You analyse DeFi protocols for an investor dashboard. You are given a protocol's "
    "revenue and TVL trends over 3, 6 and 12 months, plus recent web results about it.\n"
    "Rules:\n"
    "- Judge health from the RELATIONSHIP between revenue and TVL, not either alone. "
    "Revenue up on falling TVL means capital working harder; TVL up on falling revenue "
    "means fee compression. Say which is happening.\n"
    "- Explain the last 3 months using ONLY the news provided. Never invent a cause. If "
    "the search results don't explain the move, say plainly that the available news "
    "doesn't account for it — that is a valid and useful answer.\n"
    "- Be concrete and unsentimental. No price predictions, no advice.\n"
)


def explain_protocol(name: str, symbol: str, metrics: dict) -> ProtocolExplanation | None:
    """Explain a protocol's recent revenue/TVL trajectory. None if the model call fails."""

    def fmt(value, suffix="%"):
        return f"{value:+.1f}{suffix}" if isinstance(value, (int, float)) else "n/a"

    trend_block = (
        f"Protocol: {name} ({symbol})\n"
        f"Revenue change — 3m {fmt(metrics.get('revenue_3m'))}, "
        f"6m {fmt(metrics.get('revenue_6m'))}, 12m {fmt(metrics.get('revenue_12m'))}\n"
        f"TVL change — 3m {fmt(metrics.get('tvl_3m'))}, "
        f"6m {fmt(metrics.get('tvl_6m'))}, 12m {fmt(metrics.get('tvl_12m'))}\n"
        f"Latest monthly revenue: {metrics.get('latest_revenue')}\n"
        f"Latest TVL: {metrics.get('latest_tvl')}\n"
        f"Annualised revenue/TVL: {fmt(metrics.get('revenue_to_tvl_pct'))}"
    )

    news = web_search(f"{name} protocol TVL revenue fees news last 3 months")
    news_block = news.get("text", "(no results)")

    try:
        llm = get_llm().with_structured_output(ProtocolExplanation)
        return llm.invoke(
            [
                ("system", SYSTEM),
                ("human", f"{trend_block}\n\nRecent web results:\n{news_block}"),
            ]
        )
    except Exception:  # noqa: BLE001 - the panel still shows the rule-based verdict
        return None
