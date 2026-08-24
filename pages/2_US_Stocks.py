"""US stocks dashboard: news, trending tickers, and a watchlist with per-stock detail."""

import pandas as pd
import streamlit as st

from agent import watchlist
from ui_common import (
    cached_stock_growth,
    cached_stock_overview,
    cached_trending_stocks,
    money,
    pct,
    render_news,
    render_signal,
    render_stock_financials,
    render_watchlist_editor,
    run_agent_analysis,
)

st.set_page_config(page_title="US Stocks · Markets Dashboard", page_icon="🏦", layout="wide")
st.title("🏦 US Stocks")

# ---------------- top row: news + trending ----------------
left, right = st.columns([3, 2])

with left:
    head, toggle = st.columns([2, 1])
    head.subheader("Top news")
    ai_news = toggle.toggle(
        "AI summaries", value=True, key="stock_ai_news",
        help="Adds a one-line summary and a market-impact tag. One LLM call per refresh.",
    )
    render_news("stock", 5, ai=ai_news)

with right:
    st.subheader("Trending tickers")
    st.caption("Most-searched US equities right now (Yahoo Finance).")
    try:
        tickers = cached_trending_stocks(8)
        if tickers:
            for row_start in range(0, len(tickers), 2):
                cols = st.columns(2)
                for col, sym in zip(cols, tickers[row_start:row_start + 2]):
                    col.markdown(f"**{sym}**")
                    if col.button("Add to watchlist", key=f"addtrend_{sym}",
                                  use_container_width=True):
                        watchlist.add("stocks", sym)
                        st.rerun()
        else:
            st.info("No equities in the trending feed right now (it skews to crypto pairs).")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Couldn't load trending tickers: {exc}")

st.divider()

# ---------------- watchlist ----------------
st.subheader("My watchlist")
stocks = watchlist.get("stocks")
render_watchlist_editor("stocks", "ticker", stocks)

if not stocks:
    st.info("Add a ticker above (e.g. AAPL, MSFT, NVDA) to track it here.")
    st.stop()

# ---------------- growth ranking across the watchlist ----------------
with st.expander("📈 Growth ranking — revenue & earnings, year over year", expanded=False):
    st.caption(
        "Your watchlist ranked by year-over-year revenue growth, with EPS growth and "
        "the change in operating margin. Year-over-year rather than quarter-on-quarter "
        "because most businesses are seasonal.\n\n"
        "Unlike the crypto page this covers your watchlist rather than the whole market: "
        "Alpha Vantage is a per-symbol API on a ~25 requests/day free tier, so a "
        f"market-wide scan isn't possible. Costs 2 requests per ticker "
        f"({len(stocks) * 2} for your {len(stocks)}), then caches for hours."
    )
    if st.button("Load growth data", key="load_stock_growth"):
        st.session_state["stock_growth_loaded"] = True

    if st.session_state.get("stock_growth_loaded"):
        with st.spinner("Fetching quarterly results…"):
            try:
                grows = cached_stock_growth(tuple(stocks))
            except Exception as exc:  # noqa: BLE001
                grows = []
                st.warning(f"Couldn't load growth data: {exc}")
        if grows:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Ticker": g["symbol"],
                        "Quarter": g.get("quarter", "—"),
                        "Revenue": money(g.get("revenue_latest")),
                        "Revenue YoY": f"{g['revenue_yoy']:+.1f}%" if g.get("revenue_yoy") is not None else "—",
                        "EPS YoY": f"{g['eps_yoy']:+.1f}%" if g.get("eps_yoy") is not None else "—",
                        "Op margin": f"{g['operating_margin']:.1f}%" if g.get("operating_margin") is not None else "—",
                        "Margin change": f"{g['margin_change_pp']:+.1f}pp" if g.get("margin_change_pp") is not None else "—",
                    }
                    for g in grows
                ]),
                hide_index=True, use_container_width=True,
            )
            chart_rows = [g for g in grows if g.get("revenue_yoy") is not None]
            if chart_rows:
                st.bar_chart(
                    pd.DataFrame(
                        {
                            "revenue YoY %": [g["revenue_yoy"] for g in chart_rows],
                            "EPS YoY %": [g.get("eps_yoy") or 0 for g in chart_rows],
                        },
                        index=[g["symbol"] for g in chart_rows],
                    ),
                    height=240,
                )
            st.caption(
                "Margin change is in percentage points. Expanding margins alongside "
                "revenue growth is the healthier combination; revenue growing while "
                "margins compress can mean growth is being bought."
            )
        elif st.session_state.get("stock_growth_loaded"):
            st.info("No quarterly data available for these tickers (or the API is rate limited).")
    else:
        st.info("Not loaded — uses Alpha Vantage quota, so it's on demand.")

selected = st.radio(
    "Select a stock", stocks, horizontal=True, label_visibility="collapsed",
    key="stock_selected",
)

# Fundamentals cost an Alpha Vantage call (~25/day), so make them opt-in per view.
show_fundamentals = st.checkbox("Include fundamentals (uses Alpha Vantage quota)", value=True)

with st.spinner(f"Loading {selected}…"):
    data = cached_stock_overview(selected, show_fundamentals)

# --- headline metrics ---
c1, c2, c3 = st.columns(3)
c1.metric(
    selected,
    money(data.get("price")),
    pct(data.get("change_pct")) if data.get("change_pct") is not None else None,
)
low, high, price = data.get("week52_low"), data.get("week52_high"), data.get("price")
c2.metric("52-week low", money(low))
c3.metric("52-week high", money(high))

if low and high and price and high > low:
    pos = (price - low) / (high - low)
    st.caption(f"Position in 52-week range — {pos * 100:.0f}% of the way up")
    st.progress(min(max(pos, 0.0), 1.0))

# --- indicators ---
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
    st.info(data.get("technicals_text") or "Indicators unavailable for this ticker.")

if show_fundamentals and data.get("fundamentals_text"):
    st.markdown("**Fundamentals**")
    st.write(data["fundamentals_text"])

# --- quarterly financial statements ---
with st.expander(f"📊 {selected} quarterly financials — revenue, profit, margins"):
    st.caption(
        "From Alpha Vantage's quarterly income statements. Costs 2 requests against the "
        "~25/day free tier the first time, then caches for hours."
    )
    if st.button("Load financials", key=f"load_fin_{selected}"):
        st.session_state[f"fin_{selected}"] = True
    if st.session_state.get(f"fin_{selected}"):
        render_stock_financials(selected)
    else:
        st.info("Not loaded — uses Alpha Vantage quota, so it's on demand.")

# --- on-demand full analysis ---
st.divider()
st.caption(
    "The full analyst view runs the research agent — about a minute, and it uses "
    "free-tier API quota, so it only runs when you ask."
)
if st.button(f"🔍 Analyze {selected} (buy / hold / sell view)", type="primary"):
    result = run_agent_analysis(
        f"Would {selected} be a good long-term investment at current prices? "
        "Consider valuation fundamentals, recent quarterly earnings, technicals and news."
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
            st.markdown(f"**Business quality** — {get('quality')}")
        if get("valuation"):
            st.markdown(f"**Valuation & margin of safety** — {get('valuation')}")
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
