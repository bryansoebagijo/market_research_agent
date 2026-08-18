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

from agent.llm import get_llm, to_text
from agent.state import AgentState, Evidence, SubQuestion
from agent.tools import (
    classify_ticker,
    crypto_snapshot,
    stock_fundamentals,
    stock_snapshot,
    web_search,
)


SUMMARY_SYSTEM = (
    "You are the research step of a markets analyst. You are given a sub-question and "
    "raw data pulled from live tools. Write a concise, factual finding (1-3 sentences) "
    "that answers the sub-question using ONLY the data provided. If the data says "
    "something is unavailable, say so plainly rather than guessing. Neutral tone, no "
    "advice."
)


def _gather(subq: SubQuestion, plan_tickers: list[str]) -> list[tuple[str, str]]:
    """Run the appropriate tools for one sub-question. Returns (text, source) pairs."""
    focus = subq.focus.lower()
    # Prefer the sub-question's own tickers; fall back to the plan's.
    tickers = subq.tickers or plan_tickers
    results: list[tuple[str, str]] = []

    for t in tickers:
        kind = classify_ticker(t)
        if focus in ("price", "technicals"):
            snap = crypto_snapshot(t) if kind == "crypto" else stock_snapshot(t)
            results.append((snap["text"], snap["source"]))
        elif focus == "fundamentals" and kind == "stock":
            f = stock_fundamentals(t)
            results.append((f["text"], f["source"]))

    # News, macro, or anything a market tool couldn't cover → web search.
    if focus in ("news", "macro") or not results:
        w = web_search(subq.question)
        results.append((w["text"], w["source"]))

    return results


def researcher_node(state: AgentState) -> dict:
    """Produce an Evidence entry for every sub-question in the plan."""
    plan = state["plan"]
    llm = get_llm()
    evidence: list[Evidence] = []

    for sq in plan.subquestions:
        raw = _gather(sq, plan.tickers)
        raw_block = "\n".join(f"[{src}] {text}" for text, src in raw)

        msg = llm.invoke(
            [
                ("system", SUMMARY_SYSTEM),
                ("human", f"Sub-question: {sq.question}\n\nRaw data:\n{raw_block}"),
            ]
        )
        finding = to_text(msg).strip()
        sources = ", ".join(sorted({src for _, src in raw}))
        evidence.append(
            Evidence(subquestion=sq.question, finding=finding, source=sources)
        )

    return {"evidence": evidence}
