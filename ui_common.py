"""
Shared pieces for the dashboard pages, so Crypto and US Stocks look and behave alike.

Caching policy lives here rather than in the data layer: dashboard panels are wrapped in
`st.cache_data` with a short TTL so the pages feel instant and repeated visits don't
hammer the free-tier APIs.
"""

import uuid

import streamlit as st

from agent import market_data as md
from agent.graph import build_graph

# How long each kind of panel stays cached. News and rankings move slowly; per-asset
# detail is refreshed a bit more eagerly because that's what people click into.
TTL_NEWS = 600
TTL_RANKINGS = 300
TTL_DETAIL = 180


# ---------- cached data accessors ----------

@st.cache_data(ttl=TTL_NEWS, show_spinner=False)
def cached_headlines(kind: str, limit: int = 5):
    return md.latest_headlines(kind, limit)


@st.cache_data(ttl=TTL_RANKINGS, show_spinner=False)
def cached_open_interest(limit: int = 5):
    return md.top_open_interest(limit)


@st.cache_data(ttl=TTL_RANKINGS, show_spinner=False)
def cached_trending_coins(limit: int = 7):
    return md.trending_coins(limit)


@st.cache_data(ttl=TTL_RANKINGS, show_spinner=False)
def cached_trending_stocks(limit: int = 7):
    return md.trending_stocks(limit)


@st.cache_data(ttl=TTL_DETAIL, show_spinner=False)
def cached_coin_overview(symbol: str):
    return md.coin_overview(symbol)


@st.cache_data(ttl=TTL_DETAIL, show_spinner=False)
def cached_stock_overview(symbol: str, with_fundamentals: bool = True):
    return md.stock_overview(symbol, with_fundamentals)


# ---------- formatting ----------

def money(value, decimals: int = 2) -> str:
    """Compact currency: $68.97B, $8.35M, $1,234.56."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "—"
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(n) >= cutoff:
            return f"${n / cutoff:.2f}{suffix}"
    return f"${n:,.{decimals}f}"


def pct(value) -> str:
    try:
        return f"{float(value):+.2f}%"
    except (TypeError, ValueError):
        return "—"


# ---------- shared panels ----------

def render_news(kind: str, limit: int = 5) -> None:
    """Top headlines panel."""
    for item in cached_headlines(kind, limit):
        date = item["date"] or "undated"
        if item["link"]:
            st.markdown(f"**[{item['title']}]({item['link']})**")
        else:
            st.markdown(f"**{item['title']}**")
        st.caption(f"{date} · {item['source']}")


def render_watchlist_editor(kind: str, label: str, current: list[str]) -> None:
    """Add/remove controls for a watchlist. Callers re-read the list after this runs."""
    from agent import watchlist

    with st.form(f"add_{kind}", clear_on_submit=True):
        col1, col2 = st.columns([3, 1])
        new = col1.text_input(
            f"Add {label}", placeholder="e.g. BTC" if kind == "crypto" else "e.g. AAPL",
            label_visibility="collapsed",
        )
        if col2.form_submit_button("Add", use_container_width=True) and new.strip():
            watchlist.add(kind, new)
            st.rerun()

    if current:
        cols = st.columns(min(len(current), 4))
        for i, sym in enumerate(current):
            if cols[i % len(cols)].button(f"✕ {sym}", key=f"rm_{kind}_{sym}"):
                watchlist.remove(kind, sym)
                st.rerun()


def render_signal(tech: dict) -> None:
    """The instant rule-based read, clearly labelled as not the analyst's view."""
    trend, note = md.quick_signal(tech)
    icon = {"Uptrend": "🟢", "Downtrend": "🔴"}.get(trend, "🟡")
    st.markdown(f"**{icon} {trend}** — {note}")
    st.caption("Rule-based indicator read, not the analyst's view. Use Analyze for that.")


def run_agent_analysis(question: str) -> dict:
    """Run the full agent for an on-demand recommendation, streaming progress."""
    # A checkpointer lets us stream progress and then read the final state back.
    agent = build_graph(checkpointer=True)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    labels = {
        "planner": "Planning the research",
        "researcher": "Gathering data, news and technicals",
        "critic": "Critiquing coverage",
        "synthesizer": "Writing the report",
        "analyst": "Forming the analyst view",
    }
    with st.status("Analyzing… (about a minute)", expanded=True) as status:
        for chunk in agent.stream({"question": question}, config=config,
                                  stream_mode="updates"):
            for node in chunk:
                if node in labels:
                    status.write(f"✅ {labels[node]}")
        status.update(label="Analysis complete", state="complete")
    return agent.get_state(config).values
