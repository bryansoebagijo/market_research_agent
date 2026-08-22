"""
Streamlit UI for the markets research agent.

    streamlit run streamlit_app.py

Why a UI at all: a run makes several live API calls plus a few LLM calls and takes a
minute or two, so this streams the graph node-by-node and shows what the agent is doing
as it happens, instead of leaving you staring at a blank terminal. It also turns the
human-in-the-loop plan gate into approve / revise / cancel buttons.

The app calls the agent directly (it's Python too) — there's no separate backend.
"""

import uuid

import pandas as pd
import streamlit as st
from langgraph.types import Command

from agent.graph import build_graph

st.set_page_config(
    page_title="Markets Research Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",  # keep the review toggle visible on load
)

# Friendly labels for each graph node, shown while it runs.
NODE_LABELS = {
    "planner": "Planning the research",
    "human_review": "Waiting for your approval",
    "researcher": "Gathering live data, news and technicals",
    "critic": "Critiquing coverage",
    "synthesizer": "Writing the report",
    "analyst": "Forming the analyst view",
}

STANCE_COLOURS = {
    "accumulate": "🟢", "buy": "🟢",
    "hold": "🟡", "watch": "🔵",
    "trim": "🟠", "reduce": "🟠",
    "avoid": "🔴", "sell": "🔴",
}


@st.cache_resource
def get_agent(review: bool):
    """Compile the graph once per mode and reuse it across reruns."""
    return build_graph(review=review, checkpointer=True)


def field(obj, name, default=None):
    """Read a field whether `obj` is a Pydantic model or a plain dict.

    State restored from a checkpoint can come back as dicts, so rendering must not
    assume attribute access.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def stance_icon(stance: str) -> str:
    """Pick an icon from the first stance word we recognise."""
    low = stance.lower()
    for word, icon in STANCE_COLOURS.items():
        if word in low:
            return icon
    return "⚪"


def run_stream(agent, payload, config, status) -> dict | None:
    """Advance the graph, reporting each node as it completes.

    Returns the interrupt payload if the run paused for human review, else None.
    """
    for chunk in agent.stream(payload, config=config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                return update[0].value if isinstance(update, tuple) else update
            label = NODE_LABELS.get(node, node)
            status.write(f"✅ {label}")
    return None


def render_plan(plan) -> None:
    st.subheader("Research plan")
    st.caption(field(plan, "rationale", ""))
    cols = st.columns([1, 4])
    cols[0].metric("Tickers", ", ".join(field(plan, "tickers", []) or []) or "—")
    with cols[1]:
        for i, sq in enumerate(field(plan, "subquestions", []) or [], start=1):
            tickers = field(sq, "tickers", []) or []
            tk = f" · {', '.join(tickers)}" if tickers else ""
            st.markdown(
                f"**{i}. `{field(sq, 'focus', '?')}`{tk}** — {field(sq, 'question', '')}"
            )


def render_charts(metrics: dict) -> None:
    """Chart the structured numbers the tools collected, one tab per ticker.

    These are the same fetches that produced the written findings, so charting costs no
    extra API calls — it just shows the numbers instead of describing them.
    """
    tickers = [t for t, m in metrics.items() if m]
    if not tickers:
        return

    st.subheader("Charts")
    for ticker, tab in zip(tickers, st.tabs(tickers)):
        m = metrics[ticker]
        with tab:
            # --- price history + where price sits vs its moving averages ---
            closes = m.get("closes")
            if closes:
                df = pd.DataFrame({"close": closes})
                st.caption(f"{ticker} — recent closes ({len(closes)} periods)")
                st.line_chart(df, height=220)

            cols = st.columns(3)
            if m.get("rsi") is not None:
                rsi = m["rsi"]
                zone = "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"
                cols[0].metric("RSI(14)", f"{rsi:.0f}", zone)
                cols[0].progress(min(max(rsi / 100, 0.0), 1.0))
            if m.get("ema13") and m.get("ema21"):
                gap = (m["ema13"] / m["ema21"] - 1) * 100
                cols[1].metric("EMA13 vs EMA21", f"{gap:+.2f}%",
                               "bullish" if gap > 0 else "bearish")
            if m.get("change_pct") is not None:
                cols[2].metric("Change today", f"{m['change_pct']:+.2f}%")

            # --- position within the 52-week range ---
            low, high, price = m.get("week52_low"), m.get("week52_high"), m.get("price")
            if low and high and price and high > low:
                pos = (price - low) / (high - low)
                st.caption(
                    f"52-week range — ${low:,.2f} … **${price:,.2f}** … ${high:,.2f} "
                    f"({pos * 100:.0f}% of the way up)"
                )
                st.progress(min(max(pos, 0.0), 1.0))

            # --- multi-year EPS: the track record a long-horizon view rests on ---
            years = m.get("eps_years")
            if years:
                df = pd.DataFrame(years).set_index("year")[["eps"]]
                st.caption(f"{ticker} — annual EPS (multi-year track record)")
                st.bar_chart(df, height=220)

            # --- EPS: reported vs estimated, by quarter ---
            eps = m.get("eps_quarters")
            if eps:
                df = pd.DataFrame(eps).set_index("quarter")[["reported", "estimated"]]
                st.caption(f"{ticker} — quarterly EPS: reported vs estimated")
                st.bar_chart(df, height=240)
                beats = sum(1 for q in eps if q["reported"] > q["estimated"])
                st.caption(f"Beat estimates in **{beats} of {len(eps)}** quarters shown.")

            # --- revenue trend and margins ---
            rev = m.get("revenue_quarters")
            if rev:
                df = pd.DataFrame(rev).set_index("quarter")
                st.caption(f"{ticker} — quarterly revenue")
                st.bar_chart(df[["revenue"]], height=240)
                margins = [c for c in ("gross_margin", "operating_margin") if c in df]
                if margins and df[margins].notna().any().any():
                    st.caption("Margins (%)")
                    st.line_chart(df[margins], height=200)


def render_results(state: dict) -> None:
    """Render evidence, critic verdict, report and analyst view."""
    plan = state.get("plan")
    if plan:
        render_plan(plan)

    render_charts(state.get("metrics") or {})

    evidence = state.get("evidence", [])
    if evidence:
        with st.expander(f"Evidence gathered ({len(evidence)} findings)", expanded=False):
            for i, ev in enumerate(evidence, start=1):
                st.markdown(f"**{i}. {ev['subquestion']}**")
                st.write(ev["finding"])
                st.caption(f"Source: {ev['source']}")
                st.divider()

    c1, c2 = st.columns(2)
    c1.metric("Research passes", state.get("loops", 1))
    c2.metric("Coverage sufficient?", "Yes" if state.get("complete") else "Capped")
    if state.get("critique"):
        st.caption(f"Critic: {state['critique']}")

    if state.get("report"):
        st.subheader("Report")
        st.markdown(state["report"])

    view = state.get("analyst_view")
    if view:
        stance = field(view, "stance", "—")
        st.subheader("Analyst view")
        st.caption("Long-horizon value-investor perspective")
        a, b, c = st.columns(3)
        a.metric("Stance", f"{stance_icon(stance)} {stance}")
        b.metric("Conviction", field(view, "conviction", "—"))
        c.metric("Horizon", field(view, "horizon", "—"))

        q, v = st.columns(2)
        if field(view, "quality"):
            q.markdown(f"**Business quality**\n\n{field(view, 'quality')}")
        if field(view, "valuation"):
            v.markdown(f"**Valuation & margin of safety**\n\n{field(view, 'valuation')}")

        for heading, key in (
            ("**Why**", "reasoning"),
            ("**What would change this view**", "what_would_change_it"),
            ("**Key risks**", "key_risks"),
        ):
            st.markdown(heading)
            for p in field(view, key, []) or []:
                st.markdown(f"- {p}")
        st.info(
            "Analyst view: an informational interpretation of the evidence, produced by "
            "a language model with no validated track record. Not financial advice, not "
            "a prediction, and not personalised to your situation."
        )


# ---------- Sidebar ----------
with st.sidebar:
    st.title("📈 Markets Research Agent")
    st.caption(
        "Plans research, gathers live prices, technicals, fundamentals and news, "
        "critiques its own coverage, then reports with citations."
    )
    review = st.toggle(
        "Review the plan before researching",
        value=False,
        help="Pause after planning so you can approve, revise, or cancel.",
    )
    st.divider()
    st.caption(
        "Sources: CoinGecko · Twelve Data · Alpha Vantage · RSS news · web search.\n\n"
        "A run takes 1–3 minutes and uses free-tier API quota."
    )
    if st.button("Clear session"):
        for key in ("thread", "interrupt", "state", "question"):
            st.session_state.pop(key, None)
        st.rerun()

# ---------- Main ----------
st.header("Ask about crypto or US stocks")

question = st.text_input(
    "Your question",
    placeholder="e.g. Is NVDA expensive right now given its fundamentals and technicals?",
    label_visibility="collapsed",
)
go = st.button("Research", type="primary", disabled=not question.strip())

if go:
    # Fresh thread per run so checkpoints don't collide.
    st.session_state.thread = str(uuid.uuid4())
    st.session_state.question = question
    st.session_state.pop("interrupt", None)
    st.session_state.pop("state", None)

    agent = get_agent(review)
    config = {"configurable": {"thread_id": st.session_state.thread}}
    with st.status("Researching…", expanded=True) as status:
        payload = run_stream(agent, {"question": question}, config, status)
        if payload:
            st.session_state.interrupt = payload
            status.update(label="Waiting for your approval", state="complete")
        else:
            st.session_state.state = agent.get_state(config).values
            status.update(label="Done", state="complete")

# --- Human-in-the-loop gate ---
if st.session_state.get("interrupt"):
    payload = st.session_state.interrupt
    st.warning("Review the plan before the agent spends time and API calls on research.")
    st.caption(payload.get("rationale", ""))
    st.markdown(f"**Tickers:** {', '.join(payload.get('tickers') or []) or '—'}")
    for i, sq in enumerate(payload.get("subquestions", []), start=1):
        tk = f" · {', '.join(sq['tickers'])}" if sq.get("tickers") else ""
        st.markdown(f"**{i}. `{sq['focus']}`{tk}** — {sq['question']}")

    feedback = st.text_input(
        "Optional: what should the plan do differently?",
        placeholder="e.g. focus only on BTC and ignore macro",
    )
    a, r, c = st.columns(3)
    decision = None
    if a.button("✅ Approve", type="primary"):
        decision = {"action": "approve"}
    if r.button("✏️ Revise", disabled=not feedback.strip()):
        decision = {"action": "revise", "feedback": feedback.strip()}
    if c.button("✖️ Cancel"):
        decision = {"action": "cancel"}

    if decision:
        agent = get_agent(True)
        config = {"configurable": {"thread_id": st.session_state.thread}}
        st.session_state.pop("interrupt", None)
        with st.status("Working…", expanded=True) as status:
            payload = run_stream(agent, Command(resume=decision), config, status)
            state = agent.get_state(config).values
            if payload:  # revised -> paused again for another review
                st.session_state.interrupt = payload
                status.update(label="Re-planned — review again", state="complete")
            elif state.get("cancelled"):
                status.update(label="Cancelled", state="complete")
                st.session_state.state = {}
                st.info("Cancelled — no research was run.")
            else:
                st.session_state.state = state
                status.update(label="Done", state="complete")
        st.rerun()

# --- Results ---
if st.session_state.get("state"):
    render_results(st.session_state.state)
