"""Crypto dashboard: news, leverage, trending, and a watchlist with per-coin detail."""

import pandas as pd
import streamlit as st

from agent import watchlist
from ui_common import (
    cached_coin_overview,
    cached_growing_protocols,
    cached_open_interest,
    cached_top_protocols,
    cached_trending_coins,
    cached_trending_narratives,
    money,
    pct,
    render_growth_charts,
    render_news,
    render_protocol_history,
    render_signal,
    render_watchlist_editor,
    render_watchlist_revenue,
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

rev_tab, tvl_tab, growth_tab = st.tabs(
    ["By revenue (24h)", "By TVL", "Growing month-over-month"]
)


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


def _selected_row(rows: list[dict], key: str):
    """Render a selectable table and return the row the user clicked, if any."""
    event = st.dataframe(
        _protocol_table(rows),
        hide_index=True,
        use_container_width=True,
        on_select="rerun",
        selection_mode="single-row",
        key=key,
    )
    picked = (event.selection.rows or []) if hasattr(event, "selection") else []
    return rows[picked[0]] if picked else None


with rev_tab:
    try:
        rows = cached_top_protocols("revenue_24h", 5)
        if rows:
            st.caption("Click a row to see that project's monthly history.")
            chosen = _selected_row(rows, "rev_table")
            if chosen:
                st.divider()
                render_protocol_history(chosen["name"], chosen.get("slug", ""), "revenue")
        else:
            st.info(
                "No revenue data right now — DefiLlama's fees endpoint is intermittently "
                "slow and this ranking degrades rather than blocking the page. The TVL "
                "tab still works; try again in a few minutes."
            )
    except Exception as exc:  # noqa: BLE001
        st.warning(
            f"Couldn't load protocol revenue ({type(exc).__name__}). DefiLlama's fees "
            "endpoint is often slow; the TVL tab is unaffected."
        )

with tvl_tab:
    try:
        rows = cached_top_protocols("tvl", 5)
        if rows:
            st.caption("Click a row to see that project's monthly history.")
            chosen = _selected_row(rows, "tvl_table")
            if chosen:
                st.divider()
                render_protocol_history(chosen["name"], chosen.get("slug", ""), "tvl")
        else:
            st.info("TVL data unavailable right now.")
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Couldn't load protocol TVL: {exc}")

with growth_tab:
    st.caption(
        "Projects where **both** revenue and TVL grew over the last 30 days, ranked by "
        "revenue growth. Revenue compares the last 30 days with the 30 before it. "
        "Only projects above $50k monthly revenue **in both periods** and $10M TVL "
        "qualify — otherwise a protocol going from $200 to $2,000 shows +900% and "
        "crowds out real trends."
    )
    if st.button("Load growth data", key="load_growth"):
        st.session_state["growth_loaded"] = True

    if st.session_state.get("growth_loaded"):
        with st.spinner("Fetching 30-day TVL history per project… (~15s, then cached)"):
            try:
                rows = cached_growing_protocols(5)
            except Exception as exc:  # noqa: BLE001
                rows = []
                st.warning(f"Couldn't load growth data: {exc}")
        if rows:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Project": r["name"],
                        "Category": r["category"],
                        "Revenue 30d": money(r["revenue_30d"]),
                        "Revenue growth": f"{r['revenue_growth_30d']:+.1f}%",
                        "TVL": money(r["tvl"]),
                        "TVL growth": f"{r['tvl_growth_30d']:+.1f}%",
                    }
                    for r in rows
                ]),
                hide_index=True, use_container_width=True,
            )
            st.bar_chart(
                pd.DataFrame(
                    {
                        "revenue growth %": [r["revenue_growth_30d"] for r in rows],
                        "TVL growth %": [r["tvl_growth_30d"] for r in rows],
                    },
                    index=[r["name"] for r in rows],
                ),
                height=240,
            )
            st.divider()
            render_growth_charts(rows)
        elif st.session_state.get("growth_loaded"):
            st.info("No projects currently meet both growth conditions at these sizes.")
    else:
        st.info(
            "Not loaded — this one needs a request per project for monthly TVL history, "
            "so it's on demand to keep the page fast."
        )

st.divider()

# ---------------- trending narratives ----------------
st.subheader("Trending narratives")
st.caption(
    "Hottest sectors by 24h change in **sector** market cap, so a narrative only ranks "
    "when the whole basket moves — not one coin. Categories under $500M are excluded, "
    "or micro-cap sectors posting +800% on a few thousand dollars drown out real trends."
)
try:
    narratives = cached_trending_narratives(5)
    if narratives:
        ncols = st.columns(len(narratives))
        for idx, (col, n) in enumerate(zip(ncols, narratives)):
            col.metric(
                n["name"][:22],
                pct(n["change_24h"]),
                help=f"Sector market cap {money(n['market_cap'])} · "
                     f"24h volume {money(n['volume_24h'])}",
            )
            # Tickers rather than coin names, each one addable to the watchlist.
            names = n.get("top_coin_names") or []
            for j, ticker in enumerate(n.get("top_coins") or []):
                full = names[j] if j < len(names) else ticker
                if col.button(f"+ {ticker}", key=f"narr_{idx}_{j}_{ticker}",
                              help=f"{full} — add to watchlist",
                              use_container_width=True):
                    watchlist.add("crypto", ticker)
                    st.rerun()
        st.bar_chart(
            pd.DataFrame(
                {"24h sector market cap change %": [n["change_24h"] for n in narratives]},
                index=[n["name"][:26] for n in narratives],
            ),
            height=220,
        )
    else:
        st.info("Narrative data unavailable right now.")
except Exception as exc:  # noqa: BLE001
    st.warning(f"Couldn't load trending narratives: {exc}")

st.divider()

# ---------------- trending coins ----------------
st.subheader("Trending coins")
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

# --- protocol revenue across the whole watchlist ---
with st.expander("📈 Protocol revenue — last 12 months across your watchlist"):
    st.caption(
        "Monthly revenue for each watchlist coin backed by a fee-earning protocol. "
        "Costs a couple of requests per coin, so it loads on demand and caches."
    )
    if st.button("Load revenue history", key="load_wl_revenue"):
        st.session_state["wl_revenue_loaded"] = True
    if st.session_state.get("wl_revenue_loaded"):
        render_watchlist_revenue(coins)
    else:
        st.info("Not loaded — fetches per-protocol history, so it's on demand.")

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
