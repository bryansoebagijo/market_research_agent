"""Crypto dashboard: news, leverage, trending, and a watchlist with per-coin detail."""

import pandas as pd
import streamlit as st

from agent import watchlist
from ui_common import (
    cached_coin_overview,
    cached_open_interest,
    cached_top_protocols,
    cached_trending_coins,
    money,
    pct,
    render_news,
    render_signal,
    render_watchlist_editor,
    run_agent_analysis,
)

st.set_page_config(page_title="Crypto · Markets Dashboard", page_icon="🪙", layout="wide")
st.title("🪙 Crypto")

# ---------------- top row: news + leverage ----------------
left, right = st.columns([3, 2])

with left:
    head, toggle = st.columns([2, 1])
    head.subheader("Top news")
    # One batched LLM call per refresh window, so this is cheap — but it's still quota,
    # so it stays opt-in.
    ai_news = toggle.toggle(
        "AI summaries", value=True, key="crypto_ai_news",
        help="Adds a one-line summary and a market-impact tag. One LLM call per refresh.",
    )
    render_news("crypto", 5, ai=ai_news)

with right:
    st.subheader("Biggest open interest")
    st.caption("Top 10 coins by leverage outstanding, summed across derivatives venues.")
    try:
        rows = cached_open_interest(10)
        if rows:
            df = pd.DataFrame([
                {
                    "Coin": r["symbol"],
                    "Open interest": money(r["open_interest"]),
                    "24h volume": money(r["volume_24h"]),
                    "24h": pct(r["change_24h"]) if r["change_24h"] is not None else "—",
                }
                for r in rows
            ])
            st.dataframe(df, hide_index=True, use_container_width=True)
            chart = pd.DataFrame(
                {"open interest ($B)": [r["open_interest"] / 1e9 for r in rows]},
                index=[r["symbol"] for r in rows],
            )
            st.bar_chart(chart, height=260)
        else:
            st.info("Open-interest data unavailable right now.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Couldn't load open interest: {exc}")

st.divider()

# ---------------- top projects: revenue and TVL ----------------
st.subheader("Top projects")
st.caption(
    "Revenue is fees actually earned by the protocol (DefiLlama). Many high-revenue "
    "projects hold no TVL — a stablecoin issuer or trading bot earns without locking "
    "value — so the two rankings deliberately differ."
)

rev_tab, tvl_tab = st.tabs(["By revenue (24h)", "By TVL"])


def _protocol_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Project": r["name"],
            "Category": r["category"],
            "Revenue 24h": money(r["revenue_24h"]) if r["revenue_24h"] else "—",
            "Revenue 30d": money(r["revenue_30d"]) if r["revenue_30d"] else "—",
            "TVL": money(r["tvl"]) if r["tvl"] else "n/a",
        }
        for r in rows
    ])


with rev_tab:
    try:
        rows = cached_top_protocols("revenue_24h", 5)
        if rows:
            st.dataframe(_protocol_table(rows), hide_index=True, use_container_width=True)
            st.bar_chart(
                pd.DataFrame(
                    {"revenue 24h ($M)": [r["revenue_24h"] / 1e6 for r in rows]},
                    index=[r["name"] for r in rows],
                ),
                height=220,
            )
        else:
            st.info("Revenue data unavailable right now.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Couldn't load protocol revenue: {exc}")

with tvl_tab:
    try:
        rows = cached_top_protocols("tvl", 5)
        if rows:
            st.dataframe(_protocol_table(rows), hide_index=True, use_container_width=True)
            st.bar_chart(
                pd.DataFrame(
                    {"TVL ($B)": [r["tvl"] / 1e9 for r in rows]},
                    index=[r["name"] for r in rows],
                ),
                height=220,
            )
        else:
            st.info("TVL data unavailable right now.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Couldn't load protocol TVL: {exc}")

st.divider()

# ---------------- trending ----------------
st.subheader("Trending")
try:
    trend = cached_trending_coins(7)
    if trend:
        cols = st.columns(len(trend))
        for col, c in zip(cols, trend):
            rank = f"#{c['rank']}" if c.get("rank") else "unranked"
            col.metric(c["symbol"], rank, help=c.get("name"))
            if col.button("Add", key=f"addtrend_{c['symbol']}", use_container_width=True):
                watchlist.add("crypto", c["symbol"])
                st.rerun()
    else:
        st.info("Trending list unavailable right now.")
except Exception as exc:  # noqa: BLE001
    st.warning(f"Couldn't load trending coins: {exc}")

st.divider()

# ---------------- watchlist ----------------
st.subheader("My watchlist")
coins = watchlist.get("crypto")
render_watchlist_editor("crypto", "coin", coins)

if not coins:
    st.info("Add a coin above (e.g. BTC, ETH, SOL) to track it here.")
    st.stop()

selected = st.radio(
    "Select a coin", coins, horizontal=True, label_visibility="collapsed",
    key="crypto_selected",
)

with st.spinner(f"Loading {selected}…"):
    data = cached_coin_overview(selected)

if data.get("error"):
    st.warning(f"Some data unavailable: {data['error']}")

# --- headline metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    data.get("name") or selected,
    money(data.get("price")),
    pct(data.get("change_24h")) if data.get("change_24h") is not None else None,
)
c2.metric("Market cap", money(data.get("market_cap")))
c3.metric("24h volume", money(data.get("volume_24h")))
tvl = data.get("tvl")
if tvl:
    c4.metric("TVL", money(tvl["tvl"]), help=f"DeFi TVL on {tvl['name']} (DefiLlama)")
else:
    c4.metric("TVL", "n/a", help="TVL is a DeFi metric — it doesn't apply to every coin.")

# --- performance + indicators ---
p1, p2, p3 = st.columns(3)
p1.metric("7d", pct(data.get("change_7d")))
p2.metric("30d", pct(data.get("change_30d")))
p3.metric("From ATH", pct(data.get("ath_change")))

tech = data.get("technicals") or {}
if tech:
    render_signal(tech)
    t1, t2, t3 = st.columns(3)
    if tech.get("rsi") is not None:
        rsi = tech["rsi"]
        zone = "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"
        t1.metric("RSI(14)", f"{rsi:.0f}", zone)
    if tech.get("ema13") and tech.get("ema21"):
        gap = (tech["ema13"] / tech["ema21"] - 1) * 100
        t2.metric("EMA13 vs EMA21", f"{gap:+.2f}%")
    t3.metric("Price", money(tech.get("price")))
    if tech.get("closes"):
        st.caption("Recent daily closes")
        st.line_chart(pd.DataFrame({"close": tech["closes"]}), height=220)
else:
    st.info(data.get("technicals_text") or "Indicators unavailable for this coin.")

# --- on-demand full analysis ---
st.divider()
st.caption(
    "The full analyst view runs the research agent — about a minute, and it uses "
    "free-tier API quota, so it only runs when you ask."
)
if st.button(f"🔍 Analyze {selected} (buy / hold / sell view)", type="primary"):
    result = run_agent_analysis(
        f"Would {selected} be a good long-term investment at current prices? "
        "Consider price trend, technicals, news and macro."
    )
    st.session_state[f"analysis_{selected}"] = result

result = st.session_state.get(f"analysis_{selected}")
if result:
    view = result.get("analyst_view")
    if view:
        get = (lambda k: view.get(k) if isinstance(view, dict) else getattr(view, k, None))
        st.subheader("Analyst view")
        a, b, c = st.columns(3)
        a.metric("Stance", get("stance") or "—")
        b.metric("Conviction", get("conviction") or "—")
        c.metric("Horizon", get("horizon") or "—")
        if get("quality"):
            st.markdown(f"**Quality / network** — {get('quality')}")
        if get("valuation"):
            st.markdown(f"**Valuation** — {get('valuation')}")
        for heading, key in (("Why", "reasoning"),
                             ("What would change this view", "what_would_change_it"),
                             ("Key risks", "key_risks")):
            items = get(key) or []
            if items:
                st.markdown(f"**{heading}**")
                for p in items:
                    st.markdown(f"- {p}")
    if result.get("report"):
        with st.expander("Full research report"):
            st.markdown(result["report"])
    st.info(
        "Informational only — produced by a language model with no validated track "
        "record. Not financial advice."
    )
