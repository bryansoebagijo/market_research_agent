"""
Shared pieces for the dashboard pages, so Crypto and US Stocks look and behave alike.

Caching policy lives here rather than in the data layer: dashboard panels are wrapped in
`st.cache_data` with a short TTL so the pages feel instant and repeated visits don't
hammer the free-tier APIs.
"""

import uuid

import altair as alt
import pandas as pd
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


@st.cache_data(ttl=TTL_NEWS, show_spinner=False)
def cached_headlines_ai(kind: str, limit: int = 5):
    """Headlines plus an LLM summary and market-impact tag.

    Cached on the same TTL as the raw feed, so the whole panel costs one LLM call per
    refresh window rather than one per headline per page view.
    """
    from agent.news_ai import enrich_headlines

    return enrich_headlines(list(md.latest_headlines(kind, limit)))


@st.cache_data(ttl=TTL_RANKINGS, show_spinner=False)
def cached_open_interest(limit: int = 5):
    return md.top_open_interest(limit)


@st.cache_data(ttl=TTL_RANKINGS, show_spinner=False)
def cached_trending_coins(limit: int = 7):
    return md.trending_coins(limit)


@st.cache_data(ttl=TTL_RANKINGS, show_spinner=False)
def cached_top_protocols(by: str = "revenue_24h", limit: int = 5):
    return md.top_protocols(by, limit)


# Monthly TVL growth needs a request per protocol, so this takes ~15s on a cold cache.
# It's a monthly metric that barely moves, so it's cached far longer and loaded on
# demand rather than blocking every page view.
@st.cache_data(ttl=1800, show_spinner=False)
def cached_growing_protocols(limit: int = 5):
    return md.top_growing_protocols(limit)


# Quarterly results change a few times a year, and each symbol costs two Alpha Vantage
# requests against a ~25/day free tier, so this is cached for hours and loaded on demand.
@st.cache_data(ttl=21600, show_spinner=False)
def cached_stock_growth(symbols: tuple[str, ...]):
    return md.stock_growth_ranking(list(symbols))


@st.cache_data(ttl=1800, show_spinner=False)
def cached_protocol_history(slug: str, months: int = 12):
    return md.protocol_history(slug, months)


# Quarterly statements change a few times a year, so this is cached for hours.
@st.cache_data(ttl=21600, show_spinner=False)
def cached_stock_financials(symbol: str):
    return md.stock_financials(symbol)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_watchlist_revenue(symbols: tuple[str, ...], months: int = 12):
    return md.watchlist_revenue_history(list(symbols), months)


def render_watchlist_revenue(symbols: list[str]) -> None:
    """Twelve months of protocol revenue for every watchlist coin that earns any."""
    with st.spinner("Fetching protocol revenue history…"):
        result = cached_watchlist_revenue(tuple(symbols), 12)

    series, meta = result.get("series") or {}, result.get("meta") or {}
    without = result.get("without_revenue") or []

    if not series:
        st.info(
            "None of your watchlist coins are backed by a fee-earning protocol, so "
            "there's no revenue to chart. That's expected for monetary coins like BTC "
            "or DOGE — they have no protocol income by design."
        )
        return

    # Absolute revenue, all coins on one chart.
    frame = pd.DataFrame(
        {sym: {row["month"]: row["revenue"] for row in rows} for sym, rows in series.items()}
    ).sort_index()
    frame.index.name = "month"
    st.caption("Monthly protocol revenue (last 12 months)")
    multi_line_chart(frame, "Revenue")

    # Indexed to 100 at the start, so a small protocol's growth is visible next to a
    # large one — absolute lines alone would flatten everything under the biggest.
    if len(series) > 1:
        indexed = frame.copy()
        for col in indexed.columns:
            base = indexed[col].dropna()
            if not base.empty and base.iloc[0]:
                indexed[col] = indexed[col] / base.iloc[0] * 100
        st.caption("Same data indexed to 100 at the start — relative growth")
        multi_line_chart(indexed, "Indexed (start = 100)", money_axis=False)

    st.dataframe(
        pd.DataFrame([
            {
                "Coin": sym,
                "Protocol": m["protocol"],
                "Latest month": money(m["latest"]),
                "12-month change": f"{m['growth_pct']:+.1f}%" if m["growth_pct"] is not None else "—",
                "Period": f"{m['from']} → {m['to']}",
            }
            for sym, m in meta.items()
        ]),
        hide_index=True, use_container_width=True,
    )

    if without:
        st.caption(
            f"No protocol revenue for: {', '.join(without)} — monetary or non-protocol "
            "assets, so there is nothing to measure rather than nothing to show."
        )


def render_stock_financials(symbol: str) -> None:
    """Quarterly revenue, profit and margin charts from the income statement."""
    with st.spinner(f"Loading {symbol} quarterly financials…"):
        fin = cached_stock_financials(symbol)

    quarters = fin.get("quarters") or []
    if not quarters:
        st.info(
            f"No quarterly statements returned for {symbol} — either Alpha Vantage has "
            "no coverage or the daily request limit was reached."
        )
        return

    df = pd.DataFrame(quarters).set_index("quarter")
    df.index.name = "quarter"

    st.caption(f"{symbol} — quarterly revenue (last {len(quarters)} quarters)")
    month_line_chart(quarters, "revenue", "Revenue",
                     period_field="quarter", period_fmt="%Y-%m-%d")
    first, last = quarters[0]["revenue"], quarters[-1]["revenue"]
    if first:
        st.caption(
            f"{quarters[0]['quarter']} → {quarters[-1]['quarter']}: "
            f"{money(first)} → {money(last)} ({(last / first - 1) * 100:+.1f}%)"
        )

    profit_cols = [c for c in ("gross_profit", "operating_income", "net_income") if c in df]
    if profit_cols and df[profit_cols].notna().any().any():
        st.caption(f"{symbol} — quarterly profit")
        multi_line_chart(df[profit_cols], "Profit", period_fmt="%Y-%m-%d")

    margin_cols = [c for c in ("gross_margin", "operating_margin", "net_margin") if c in df]
    if margin_cols and df[margin_cols].notna().any().any():
        st.caption(f"{symbol} — margins (%)")
        multi_line_chart(df[margin_cols], "Margin (%)", period_fmt="%Y-%m-%d",
                         money_axis=False)
        st.caption(
            "Margins matter more than the absolute figures: revenue rising while margins "
            "compress means growth is being bought rather than earned."
        )


def render_protocol_history(name: str, slug: str, primary: str = "revenue") -> None:
    """Monthly revenue and TVL history for one protocol, primary metric first."""
    if not slug:
        st.info(f"No DefiLlama slug for {name}, so history isn't available.")
        return

    with st.spinner(f"Loading {name} history…"):
        hist = cached_protocol_history(slug, 12)

    rev = hist.get("revenue_monthly") or []
    tvl = hist.get("tvl_monthly") or []
    if not rev and not tvl:
        st.info(f"No monthly history returned for {name}.")
        return

    def rev_chart():
        if rev:
            st.caption(f"{name} — monthly revenue (last 12 months, current month excluded)")
            month_line_chart(rev, "revenue", "Revenue")
            first, last = rev[0]["revenue"], rev[-1]["revenue"]
            if first:
                st.caption(
                    f"{rev[0]['month']} → {rev[-1]['month']}: "
                    f"{money(first)} → {money(last)} ({(last / first - 1) * 100:+.1f}%)"
                )
        else:
            st.info("No revenue history for this project.")

    def tvl_chart():
        if tvl:
            st.caption(f"{name} — monthly TVL (month-end level)")
            month_line_chart(tvl, "tvl", "TVL")
            first, last = tvl[0]["tvl"], tvl[-1]["tvl"]
            if first:
                st.caption(
                    f"{tvl[0]['month']} → {tvl[-1]['month']}: "
                    f"{money(first)} → {money(last)} ({(last / first - 1) * 100:+.1f}%)"
                )
        else:
            st.info("No TVL history for this project (not every project locks value).")

    if primary == "revenue":
        rev_chart()
        tvl_chart()
    else:
        tvl_chart()
        rev_chart()


@st.cache_data(ttl=TTL_RANKINGS, show_spinner=False)
def cached_trending_stocks(limit: int = 7):
    return md.trending_stocks(limit)


@st.cache_data(ttl=TTL_RANKINGS, show_spinner=False)
def cached_trending_narratives(limit: int = 5):
    return md.trending_narratives(limit)


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


# ---------- charts ----------

def _to_dates(df: pd.DataFrame, column: str, fmt: str | None = None) -> pd.DataFrame:
    """Parse a period column into real dates.

    Kept as datetimes rather than pre-formatted strings so the axis stays in
    chronological order — "Apr-2026" before "Jan-2026" is what string sorting gives.
    """
    out = df.copy()
    out["_date"] = pd.to_datetime(out[column], format=fmt, errors="coerce")
    return out.dropna(subset=["_date"])


def month_line_chart(
    rows: list[dict],
    value_field: str,
    label: str,
    period_field: str = "month",
    period_fmt: str | None = "%Y-%m",
    money_axis: bool = True,
) -> None:
    """Single-series monthly line chart with readable dates and formatted tooltips."""
    if not rows:
        return
    df = _to_dates(pd.DataFrame(rows), period_field, period_fmt)
    if df.empty:
        return

    value_format = "$,.0f" if money_axis else ",.2f"
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("_date:T", title=None,
                    axis=alt.Axis(format="%b-%Y", labelAngle=-45, tickCount=len(df))),
            y=alt.Y(f"{value_field}:Q", title=label,
                    axis=alt.Axis(format="$~s" if money_axis else "~s")),
            tooltip=[
                alt.Tooltip("_date:T", title="Month", format="%b %Y"),
                alt.Tooltip(f"{value_field}:Q", title=label, format=value_format),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


def multi_line_chart(
    frame: pd.DataFrame,
    label: str,
    period_fmt: str | None = "%Y-%m",
    money_axis: bool = True,
) -> None:
    """Multi-series line chart from a wide frame indexed by period."""
    if frame.empty:
        return
    long = frame.reset_index().melt(
        id_vars=frame.index.name or "index", var_name="series", value_name="value"
    )
    period_col = frame.index.name or "index"
    long = _to_dates(long, period_col, period_fmt).dropna(subset=["value"])
    if long.empty:
        return

    value_format = "$,.0f" if money_axis else ",.1f"
    chart = (
        alt.Chart(long)
        .mark_line(point=True)
        .encode(
            x=alt.X("_date:T", title=None, axis=alt.Axis(format="%b-%Y", labelAngle=-45)),
            y=alt.Y("value:Q", title=label,
                    axis=alt.Axis(format="$~s" if money_axis else "~s")),
            color=alt.Color("series:N", title=None),
            tooltip=[
                alt.Tooltip("_date:T", title="Month", format="%b %Y"),
                alt.Tooltip("series:N", title=""),
                alt.Tooltip("value:Q", title=label, format=value_format),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, use_container_width=True)


# ---------- shared panels ----------

IMPACT_ICONS = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}


def render_news(kind: str, limit: int = 5, ai: bool = False) -> None:
    """Top headlines as cards: thumbnail, headline, summary, source and date.

    Feeds are inconsistent — some carry an image but no summary, some the reverse — so
    each card lays out around whatever that item actually has, and never invents text.
    With ai=True, one batched LLM call adds a uniform summary and a market-impact tag.
    """
    items = cached_headlines_ai(kind, limit) if ai else cached_headlines(kind, limit)

    if ai and items and items[0].get("ai_error"):
        st.caption(f"AI summaries unavailable ({items[0]['ai_error']}) — showing feed text.")

    for i, item in enumerate(items):
        if item.get("image"):
            img_col, text_col = st.columns([1, 4])
            try:
                img_col.image(item["image"], use_container_width=True)
            except Exception:  # noqa: BLE001 - a dead image URL must not break the page
                img_col.caption("🖼️")
        else:
            text_col = st.container()

        with text_col:
            impact = (item.get("impact") or "").lower()
            badge = f"{IMPACT_ICONS.get(impact, '')} " if impact else ""
            if item["link"]:
                st.markdown(f"{badge}**[{item['title']}]({item['link']})**")
            else:
                st.markdown(f"{badge}**{item['title']}**")

            # Prefer the LLM's uniform summary; fall back to the feed's own lede.
            body = item.get("ai_summary") or item.get("summary")
            if body:
                st.write(body)

            # Assets the story touches, each clickable to add to the watchlist.
            assets = item.get("assets") or []
            if assets:
                label_col, *asset_cols = st.columns([2] + [1] * len(assets))
                label_col.caption("Affects:")
                for col, sym in zip(asset_cols, assets):
                    if col.button(f"+ {sym}", key=f"news_{kind}_{i}_{sym}",
                                  help=f"Add {sym} to your {kind} watchlist"):
                        from agent import watchlist

                        watchlist.add("crypto" if kind == "crypto" else "stocks", sym)
                        st.rerun()

            meta = f"{item['date'] or 'undated'} · {item['source']}"
            if item.get("ai_summary"):
                meta += " · AI summary"
                if impact:
                    meta += f" · impact: {item['impact']}"
            st.caption(meta)
        if i < len(items) - 1:
            st.divider()


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
