"""
LLM enrichment for news headlines — kept separate from `market_data.py`, which is
deliberately LLM-free so the dashboard's core panels stay instant.

An important limitation shapes the design: RSS gives us a headline and sometimes a
one-line lede, never the article body. So the model is told to work strictly from what
it's given and NOT to invent specifics — no numbers, names, or causes that aren't in the
source. Where a feed supplies no lede, the honest output is a plain restatement plus a
market-impact read, not a fabricated précis.

Everything is done in ONE batched call for the whole panel, so enriching five headlines
costs one request rather than five.
"""

from pydantic import BaseModel, Field

from agent.llm import get_llm


class HeadlineTake(BaseModel):
    """The model's read on a single headline."""

    impact: str = Field(description="Bullish, Bearish, or Neutral for the market.")
    summary: str = Field(
        description="One sentence, max ~25 words, stating what the story says and why it "
        "matters. Use ONLY the provided title and lede — never invent figures, names, "
        "or causes."
    )


class HeadlineTakes(BaseModel):
    """One take per headline, in the order the headlines were given."""

    takes: list[HeadlineTake] = Field(description="Same order and count as the input.")


SYSTEM = (
    "You summarise market news for a dashboard. For each numbered item you are given a "
    "headline and sometimes a short lede — you do NOT have the article body.\n"
    "Rules:\n"
    "- Work ONLY from the text provided. Never invent figures, names, dates or causes. "
    "If the item is just a headline, restate its substance plainly rather than "
    "elaborating.\n"
    "- One sentence each, around 25 words, plain language, no hype.\n"
    "- Also judge the likely market impact: Bullish, Bearish, or Neutral. Use Neutral "
    "when the story is informational or the direction genuinely isn't clear.\n"
    "- Return exactly one take per item, in the same order."
)


def enrich_headlines(items: list[dict]) -> list[dict]:
    """Add `ai_summary` and `impact` to each headline item.

    Returns the items unchanged if the model call fails — the dashboard must still
    render when the LLM is unavailable or out of quota.
    """
    if not items:
        return items

    lines = []
    for i, item in enumerate(items, start=1):
        lede = item.get("summary") or ""
        lines.append(
            f"{i}. HEADLINE: {item['title']}"
            + (f"\n   LEDE: {lede}" if lede else "\n   (no lede provided)")
        )

    try:
        llm = get_llm().with_structured_output(HeadlineTakes)
        result: HeadlineTakes = llm.invoke(
            [("system", SYSTEM), ("human", "\n".join(lines))]
        )
        takes = result.takes
    except Exception as exc:  # noqa: BLE001 - never let this break the panel
        for item in items:
            item["ai_error"] = str(exc)[:100]
        return items

    for i, item in enumerate(items):
        if i < len(takes):
            item["ai_summary"] = takes[i].summary.strip()
            item["impact"] = takes[i].impact.strip()
    return items
