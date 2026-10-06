"""
Researcher node — gathers evidence for each sub-question.

For every sub-question the planner produced, this node:
  1. routes to the right tool(s) based on the sub-question's `focus` tag and tickers
     (price/technicals → market snapshot; fundamentals → Alpha Vantage; news/macro →
     web search),
  2. hands the raw tool output to the LLM, which distills it into a short factual
     `finding`, and
  3. appends an Evidence entry (finding + sources) to the shared state.

Because the state's `evidence` field uses an `add` reducer, a second research pass
(after the critic loops back) APPENDS to these findings instead of replacing them.
"""

from pydantic import BaseModel, Field

from agent.llm import get_llm
from agent.state import AgentState, Evidence, SubQuestion
from agent.tools import (
    ETF_TYPES,
    classify_ticker,
    crypto_protocol_fundamentals,
    crypto_snapshot,
    crypto_technicals,
    etf_flows,
    fetch_news,
    stock_earnings,
    stock_fundamentals,
    stock_snapshot,
    stock_technicals,
    web_search,
)


SUMMARY_SYSTEM = (
    "You are the research step of a markets analyst. You are given several numbered "
    "sub-questions, each with raw data pulled from live tools. For EACH sub-question, "
    "write a concise, factual finding (1-3 sentences) using ONLY that sub-question's "
    "data. If the data says something is unavailable, say so plainly rather than "
    "guessing. Neutral tone, no advice. Return exactly one finding per sub-question, "
    "in the same order."
)


class BatchFindings(BaseModel):
    """One finding per sub-question, in the order the sub-questions were given."""

    findings: list[str] = Field(
        description="One concise factual finding per sub-question, same order."
    )


def _gather(
    subq: SubQuestion, plan_tickers: list[str], metrics: dict
) -> list[tuple[str, str]]:
    """Run the appropriate tools for one sub-question. Returns (text, source) pairs.

    Any structured numbers a tool returns are collected into `metrics[ticker]` so the UI
    can chart them — the same fetch serves both the LLM's text findings and the charts.
    """
    focus = subq.focus.lower()
    # Prefer the sub-question's own tickers; fall back to the plan's.
    tickers = subq.tickers or plan_tickers
    results: list[tuple[str, str]] = []

    def record(ticker: str, out: dict) -> None:
        """Keep a tool's text/source for the LLM and stash its numbers for charting."""
        results.append((out["text"], out["source"]))
        if out.get("data"):
            metrics.setdefault(ticker.upper(), {}).update(out["data"])

    for t in tickers:
        kind = classify_ticker(t)
        if focus == "price":
            record(t, crypto_snapshot(t) if kind == "crypto" else stock_snapshot(t))
        elif focus == "technicals":
            record(t, crypto_technicals(t) if kind == "crypto" else stock_technicals(t))
        elif focus == "fundamentals" and kind == "stock":
            record(t, stock_fundamentals(t))
        elif focus == "fundamentals" and kind == "crypto":
            # Crypto's analogue: protocol revenue and TVL for DeFi tokens.
            record(t, crypto_protocol_fundamentals(t))
            # BTC and ETH have no protocol revenue, so the call above correctly reports
            # that revenue-based valuation doesn't apply — and leaves nothing to weigh.
            # Spot ETF demand is the fundamental they do have, so it's added here rather
            # than requiring the planner to have thought to ask for it separately. Only
            # for those two: appending "AAVE has no US spot ETF" to every DeFi token's
            # fundamentals would be noise, not an answer.
            if t.upper() in ETF_TYPES:
                record(t, etf_flows(t))
        elif focus == "flows":
            record(t, etf_flows(t))
        elif focus == "earnings" and kind == "stock":
            record(t, stock_earnings(t))

    # News: pull dated headlines from real outlets (RSS), split by market type, so the
    # findings are actual recent news rather than undated web snippets.
    if focus == "news":
        for kind in {classify_ticker(t) for t in tickers} or {"crypto", "stock"}:
            relevant = [t for t in tickers if classify_ticker(t) == kind]
            n = fetch_news(kind, ",".join(sorted(relevant)))
            results.append((n["text"], n["source"]))

    # Macro, news follow-up context, or anything a market tool couldn't cover → web.
    if focus in ("news", "macro") or not results:
        w = web_search(subq.question)
        results.append((w["text"], w["source"]))

    return results


def researcher_node(state: AgentState) -> dict:
    """Produce an Evidence entry for each target sub-question.

    First pass: the plan's sub-questions. On a loop-back, the critic's follow-up
    sub-questions instead — so a second pass only fills the gaps rather than redoing
    everything. New evidence is appended (the state's `add` reducer).
    """
    plan = state["plan"]

    # On a loop-back the critic put targeted follow-ups in state; use those if present.
    targets = state.get("followups") or plan.subquestions

    # 1. Gather raw tool data for every sub-question (no LLM yet — just APIs).
    blocks: list[str] = []
    sources_per: list[str] = []
    metrics: dict = {}
    for i, sq in enumerate(targets, start=1):
        raw = _gather(sq, plan.tickers, metrics)
        raw_text = "\n".join(f"[{src}] {text}" for text, src in raw)
        blocks.append(f"Sub-question {i}: {sq.question}\nData:\n{raw_text}")
        sources_per.append(", ".join(sorted({src for _, src in raw})))

    # 2. ONE LLM call turns all the raw data into one finding per sub-question.
    llm = get_llm().with_structured_output(BatchFindings)
    result: BatchFindings = llm.invoke(
        [("system", SUMMARY_SYSTEM), ("human", "\n\n".join(blocks))]
    )
    findings = result.findings

    # 3. Pair findings back to sub-questions (guard against a length mismatch).
    evidence: list[Evidence] = []
    for i, sq in enumerate(targets):
        finding = findings[i].strip() if i < len(findings) else "(no finding generated)"
        evidence.append(
            Evidence(subquestion=sq.question, finding=finding, source=sources_per[i])
        )

    return {"evidence": evidence, "metrics": metrics}
