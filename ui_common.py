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
def cached_protocol_history(slug: str, months: int = 12, include_partial: bool = False):
    return md.protocol_history(slug, months, include_partial)


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

    # Rendered as markdown rather than a dataframe so the changes can be coloured.
    st.markdown("**Revenue change by coin**")
    for sym, m in meta.items():
        twelve = coloured_pct(m["growth_pct"]) if m["growth_pct"] is not None else "—"
        six_rows = [r for r in series[sym] if r.get("revenue") is not None]
        six = "—"
        if len(six_rows) > 6:
            start, end = six_rows[-7]["revenue"], six_rows[-1]["revenue"]
            if start:
                six = coloured_pct((end / start - 1) * 100)
        st.markdown(
            f"**{sym}** ({m['protocol']}) — latest {money(m['latest'])} · "
            f"6-month {six} · 12-month {twelve}"
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
    # Four quarters is the equivalent recent window for statements.
    note = change_note(quarters, "revenue", months=4)
    if note:
        st.markdown(note.replace("Last 4 months", "Last 4 quarters"))

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
            note = change_note(rev, "revenue")
            if note:
                st.markdown(note)
        else:
            st.info("No revenue history for this project.")

    def tvl_chart():
        if tvl:
            st.caption(f"{name} — monthly TVL (month-end level)")
            month_line_chart(tvl, "tvl", "TVL")
            note = change_note(tvl, "tvl")
            if note:
                st.markdown(note)
        else:
            st.info("No TVL history for this project (not every project locks value).")

    render_protocol_verdict(name, slug, hist)

    if primary == "revenue":
        rev_chart()
        tvl_chart()
    else:
        tvl_chart()
        rev_chart()


def render_growth_charts(rows: list[dict]) -> None:
    """Monthly revenue trends for the fastest-growing projects.

    A single growth percentage says how far a project moved but not how: steady climb,
    one spike, or a recovery from a collapse all look identical as "+179%". These charts
    show the shape behind the number.
    """
    if not rows:
        return

    histories: dict[str, list[dict]] = {}
    partial_month = None
    with st.spinner("Loading revenue history for each project…"):
        for r in rows:
            slug = r.get("slug")
            if not slug:
                continue
            # The current month is kept here: these projects rank as growing BECAUSE of
            # the last few weeks, so excluding it would hide the very move being shown.
            hist = cached_protocol_history(slug, 12, True)
            revenue = hist.get("revenue_monthly") or []
            if revenue:
                histories[r["name"]] = revenue
                partial_month = partial_month or hist.get("partial_month")

    if not histories:
        st.info("No monthly revenue history available for these projects.")
        return

    if partial_month:
        st.caption(
            f"The final point ({partial_month}) is **month-to-date**, not a complete "
            "month — it's included because the recent surge is what put these projects "
            "in this ranking."
        )

    # Indexed rather than absolute: these projects differ by orders of magnitude, so
    # absolute lines would flatten every smaller one against the largest.
    frame = pd.DataFrame(
        {name: {row["month"]: row["revenue"] for row in rev} for name, rev in histories.items()}
    ).sort_index()
    frame.index.name = "month"
    indexed = frame.copy()
    for col in indexed.columns:
        base = indexed[col].dropna()
        if not base.empty and base.iloc[0]:
            indexed[col] = indexed[col] / base.iloc[0] * 100

    st.markdown("**Revenue trend, indexed to 100 at the start of the window**")
    multi_line_chart(indexed, "Indexed revenue", money_axis=False)

    st.markdown("**Each project's monthly revenue**")
    for name, revenue in histories.items():
        with st.expander(name):
            month_line_chart(revenue, "revenue", "Revenue")
            note = change_note(revenue, "revenue")
            if note:
                st.markdown(note)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_token_multiples(symbol: str):
    return md.token_multiples(symbol)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_multiples_universe():
    return md._multiples_universe()


def render_token_multiples(symbol: str) -> None:
    """Market cap against annualised protocol revenue, next to a peer median."""
    m = cached_token_multiples(symbol)
    if not m.get("available"):
        st.caption(
            f"No revenue multiple for {symbol.upper()}: it either earns no protocol fees "
            "or its market cap prices a whole ecosystem (an L1 or exchange token) rather "
            "than a claim on one fee line, which makes the ratio meaningless."
        )
        return

    st.markdown("**Valuation multiple**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Market cap / revenue", f"{m['ps']:.1f}x",
              help="Annualised from the last 30 days of protocol revenue — crypto's "
                   "equivalent of price-to-sales.")
    c2.metric(f"{m['peer_basis'][:16]} median", f"{m['peer_median_ps']:.1f}x",
              help=f"Median across {m['peer_count']} comparable tokens.")
    if m.get("mcap_to_tvl") is not None:
        c3.metric("Market cap / TVL", f"{m['mcap_to_tvl']:.2f}x")

    if m.get("vs_peers_pct") is not None and not m["implausible"]:
        direction = "above" if m["vs_peers_pct"] > 0 else "below"
        st.markdown(
            f"Trading **{abs(m['vs_peers_pct']):.0f}% {direction}** the "
            f"{m['peer_basis']} median — cheaper than "
            f"**{m['cheaper_than']:.0f}%** of its peers."
        )

    if m["implausible"]:
        st.warning(
            f"{m['ps']:.0f}x is high enough that the market cap is almost certainly "
            "pricing something well beyond these fees — treat the ratio as noise here, "
            "not as a valuation."
        )

    st.caption(
        f"Revenue summed across {m['product_count']} protocol(s) carrying this ticker, "
        f"annualised: {money(m['annual_revenue'])}. **The caveat that matters:** many "
        "tokens are governance-only and have no claim on the fees being counted, so a "
        "low multiple is a reason to check the token's rights, not a conclusion."
    )


@st.cache_data(ttl=900, show_spinner=False)
def cached_total3():
    """The stored TOTAL3 series, or None — never builds, since building takes minutes."""
    from agent.total3 import read_cache

    return read_cache()


def _forecast_verdict() -> dict | None:
    """The measured answer to 'does a model beat the random walk?', if it's been run."""
    import json

    from config import ROOT_DIR

    path = ROOT_DIR / "eval" / "total3_forecast.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def render_total3() -> None:
    """TOTAL3 with a volatility cone — a range, not a prediction."""
    from agent import total3 as t3

    st.subheader("TOTAL3 — crypto market cap excluding BTC and ETH")
    data = cached_total3()

    if data is None:
        st.caption(
            "CoinGecko's market-cap history is a paid endpoint, so this series is rebuilt "
            "by summing the 50 largest non-BTC/ETH coins. The free tier sustains about "
            "five requests a minute, so building it takes roughly 11 minutes — then it's "
            "cached for a day."
        )
        if st.button("Build the TOTAL3 series (~11 min)", key="build_total3"):
            status = st.status("Fetching market-cap history…", expanded=True)
            t3.load_series(progress=lambda i, n, sym:
                           status.update(label=f"Fetching {i}/{n} — {sym}"))
            status.update(label="Series built", state="complete")
            cached_total3.clear()
            st.rerun()
        return

    series = data["series"]
    levels = [p["total3"] for p in series]
    live = data.get("live") or {}
    change = (levels[-1] / levels[0] - 1) * 100 if levels[0] else None

    c1, c2, c3 = st.columns(3)
    c1.metric("TOTAL3 now", money(live.get("total3") or levels[-1]))
    c2.metric("Past year", f"{change:+.1f}%" if change is not None else "—")
    c3.metric("BTC dominance", f"{live.get('btc_dominance', 0):.1f}%",
              help="Rising dominance mechanically squeezes everything in TOTAL3.")

    # The panel forecasts whatever horizon the evaluation declared primary, so the two
    # can't drift apart.
    forecast = _forecast_verdict()
    horizon = int((forecast or {}).get("primary_horizon", 3))
    cone = t3.volatility_cone(levels, horizon_days=horizon)

    # Volatility always uses the full series; only the drawn history is shortened. A
    # 30-day cone against a year of history renders as a sliver, which hides the forward
    # range this panel exists to show.
    window = st.radio("History shown", ["1 month", "3 months", "1 year"], horizontal=True,
                      label_visibility="collapsed", key="total3_window")
    cutoff = {"1 month": 30, "3 months": 90}.get(window)
    shown = series[-cutoff:] if cutoff else series
    history = pd.DataFrame({
        "date": pd.to_datetime([p["date"] for p in shown]),
        "TOTAL3": [p["total3"] for p in shown],
    })
    last_date = history["date"].iloc[-1]
    projection = pd.DataFrame([
        {"date": last_date + pd.Timedelta(days=b["day"]), **b} for b in cone["bands"]
    ])

    base = alt.Chart(projection)
    chart = alt.layer(
        base.mark_area(opacity=0.12, color="#4c78a8").encode(
            x=alt.X("date:T", title=None), y=alt.Y("lo2:Q", title=None), y2="hi2:Q"),
        base.mark_area(opacity=0.22, color="#4c78a8").encode(
            x="date:T", y="lo1:Q", y2="hi1:Q"),
        base.mark_line(strokeDash=[4, 4], color="#4c78a8").encode(x="date:T", y="mid:Q"),
        alt.Chart(history).mark_line(color="#e45756").encode(
            x=alt.X("date:T", title=None),
            # zero=False: the movement sits between $0.7T and $1T, so a zero-based axis
            # squeezes all of it into the top third of the chart.
            y=alt.Y("TOTAL3:Q", title="TOTAL3", axis=alt.Axis(format="$~s"),
                    scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip("date:T", title="Date", format="%b %d, %Y"),
                     alt.Tooltip("TOTAL3:Q", title="TOTAL3", format="$,.0f")]),
    ).properties(height=320)
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        f"Shaded: where TOTAL3 sits at 1σ and 2σ over the next {cone['horizon_days']} days "
        f"if it keeps the volatility of the last {cone['window_days']} days "
        f"({cone['annualised_vol'] * 100:.0f}% annualised). **The dashed centre is flat on "
        "purpose** — it is the random walk, which beat every model tested below."
    )

    # A percentage for each direction, shown with the base rate beside it: a probability
    # only carries information insofar as it differs from how often the market rose anyway.
    from agent import total3_model as t3_model

    prediction = t3_model.predict_now(levels, horizon_days=horizon)
    primary = (forecast or {}).get("horizons", {}).get(str(horizon)) or {}
    direction = primary.get("direction_model") or {}

    if prediction.get("available"):
        st.markdown(f"**Direction over the next {horizon} days**")
        d1, d2, d3 = st.columns(3)
        d1.metric("Model says up", f"{prediction['probability_up']:.0%}")
        d2.metric("Model says down", f"{prediction['probability_down']:.0%}")
        d3.metric("Base rate up", f"{prediction['base_rate_up']:.0%}",
                  help="How often TOTAL3 actually rose over this horizon in the past year. "
                       "The model has to beat this to be worth anything.")

        if direction:
            st.markdown(("🟢 " if direction.get("beats_base_rate") else "🔴 ")
                        + direction.get("verdict", ""))
            # Bins under 5 cases are too small to read anything into.
            bins = [b for b in (direction.get("calibration") or []) if b.get("n", 0) >= 5]
            worst = max(bins, key=lambda b: abs(b["predicted"] - b["actual"]), default=None)
            if worst:
                st.caption(
                    f"Calibration: when it said {worst['predicted']:.0%}, TOTAL3 rose "
                    f"{worst['actual']:.0%} of the time (n={worst['n']}). Accuracy alone "
                    f"({direction.get('accuracy', 0):.1%}) would have hidden that, which is "
                    "why the probability is graded on Brier score against the base rate."
                )

    if primary:
        skill = (primary.get("ridge_model") or {}).get("skill_vs_naive")
        st.markdown(("🟢 " if primary.get("beats_baseline") else "🔴 ")
                    + f"**Magnitude:** {primary.get('verdict', '')}")
        st.caption(
            f"Measured walk-forward over {primary.get('independent_windows')} independent "
            f"windows. Skill {skill:+.3f} — negative means its squared error is worse than "
            "assuming no change, which is why the cone's centre is flat."
            if isinstance(skill, (int, float)) else
            f"Measured walk-forward over {primary.get('independent_windows')} independent "
            "windows."
        )

    st.caption(
        f"Reconstructed from {data['n_coins_used']} coins covering "
        f"{data['coverage_pct']:.0f}% of TOTAL3, chained so constituents entering or "
        f"leaving don't fake a jump, and calibrated to the live level. "
        f"{data['stable_share_pct']:.0f}% of it is stablecoins, which damps its "
        "volatility. One year of history — a single market regime."
    )


TONE_ICONS = {"green": "🟢", "orange": "🟠", "red": "🔴", "gray": "⚪"}


@st.cache_data(ttl=1800, show_spinner=False)
def cached_protocol_explanation(name: str, symbol: str, metrics: dict):
    from agent.protocol_ai import explain_protocol

    return explain_protocol(name, symbol, metrics)


def render_protocol_verdict(name: str, symbol: str, history: dict) -> None:
    """Rule-based health verdict, with an optional LLM explanation of the recent move."""
    health = md.protocol_health(history)
    if health.get("verdict") == "Not enough data":
        return

    st.markdown(
        f"### {TONE_ICONS.get(health['tone'], '')} {health['verdict']}"
    )
    st.write(health["reason"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue · 3m", coloured_pct_plain(health.get("revenue_3m")))
    c2.metric("TVL · 3m", coloured_pct_plain(health.get("tvl_3m")))
    if health.get("revenue_to_tvl_pct") is not None:
        c3.metric("Revenue / TVL", f"{health['revenue_to_tvl_pct']:.2f}%",
                  help="Annualised revenue as a share of TVL — how hard the locked "
                       "capital works.")

    st.caption(
        f"6-month revenue {coloured_pct(health.get('revenue_6m'))} · "
        f"12-month {coloured_pct(health.get('revenue_12m'))}  |  "
        f"6-month TVL {coloured_pct(health.get('tvl_6m'))} · "
        f"12-month {coloured_pct(health.get('tvl_12m'))}"
    )

    # The "why" needs outside context, so it's an LLM call and stays on demand.
    key = f"explain_{symbol}"
    if st.button(f"Explain why {name} moved", key=f"btn_{key}"):
        st.session_state[key] = True
    if st.session_state.get(key):
        with st.spinner("Searching for what drove the move…"):
            explanation = cached_protocol_explanation(name, symbol, dict(health))
        if explanation is None:
            st.info("Couldn't generate an explanation (model unavailable or out of quota).")
        else:
            st.markdown(f"**Assessment** — {explanation.assessment}")
            st.markdown("**What drove the last 3 months**")
            for d in explanation.drivers:
                st.markdown(f"- {d}")
            st.markdown("**What to watch next**")
            for w in explanation.watch_next:
                st.markdown(f"- {w}")
            st.caption(
                "Explanation generated by a language model from public search results; "
                "it may be incomplete. Informational only, not advice."
            )


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


def coloured_pct(value) -> str:
    """A signed percentage in Streamlit markdown colour — green up, red down."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "—"
    colour = "green" if n > 0 else "red" if n < 0 else "gray"
    return f":{colour}[{n:+.1f}%]"


def coloured_pct_plain(value) -> str:
    """Signed percentage without markdown colour, for st.metric (which renders plain)."""
    try:
        return f"{float(value):+.1f}%"
    except (TypeError, ValueError):
        return "—"


def change_note(rows: list[dict], value_key: str, months: int = 6) -> str | None:
    """Markdown summarising the change over the last `months` periods.

    Recent direction is what a reader actually wants: a year-long decline that turned
    around three months ago reads as "down 52%" over the full window, which hides the
    turn. Falls back to the whole series when it's shorter than the window.
    """
    points = [r for r in rows if r.get(value_key) is not None]
    if len(points) < 2:
        return None

    start = points[-(months + 1)] if len(points) > months else points[0]
    end = points[-1]
    first, last = start[value_key], end[value_key]
    if not first:
        return None

    span = len(points) - 1 if len(points) <= months else months
    period_key = "month" if "month" in end else "quarter"
    return (
        f"Last {span} months: {start[period_key]} → {end[period_key]} · "
        f"{money(first)} → {money(last)} ({coloured_pct((last / first - 1) * 100)})"
    )


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
    x_enc = alt.X("_date:T", title=None,
                  axis=alt.Axis(format="%b-%Y", labelAngle=-45, tickCount=len(df)))
    y_enc = alt.Y(f"{value_field}:Q", title=label,
                  axis=alt.Axis(format="$~s" if money_axis else "~s"))

    base = alt.Chart(df)
    line = base.mark_line(point=True).encode(x=x_enc, y=y_enc)

    # Hovering the line itself means hitting a few pixels exactly. This invisible rule
    # spans the full height at each x, so the tooltip fires from anywhere vertically and
    # snaps to the nearest month.
    hover = alt.selection_point(nearest=True, on="pointerover",
                                fields=["_date"], empty=False)
    hover_layer = (
        base.mark_rule(color="gray")
        .encode(
            x=x_enc,
            opacity=alt.condition(hover, alt.value(0.35), alt.value(0)),
            tooltip=[
                alt.Tooltip("_date:T", title="Month", format="%b %Y"),
                alt.Tooltip(f"{value_field}:Q", title=label, format=value_format),
            ],
        )
        .add_params(hover)
    )
    st.altair_chart((line + hover_layer).properties(height=280),
                    use_container_width=True)


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
    x_enc = alt.X("_date:T", title=None, axis=alt.Axis(format="%b-%Y", labelAngle=-45))

    base = alt.Chart(long)

    names = [str(c) for c in frame.columns]

    # Clicking a legend entry removes that series; clicking again restores it.
    # empty=False means nothing is selected initially, so everything shows, and a click
    # selects just that series — which the inverted filter then drops. (Binding a legend
    # the conventional way does the opposite: it isolates the clicked series.)
    # Filtering rather than dimming matters here, because removing the rows lets the
    # y-axis rescale — the usual reason for hiding a dominant line is to see the rest.
    # The colour domain is pinned so the legend keeps every entry even once its data is
    # filtered out, otherwise a hidden series could never be clicked back.
    hidden = alt.selection_point(fields=["series"], bind="legend", empty=False)
    line = (
        base.mark_line(point=True)
        .encode(
            x=x_enc,
            y=alt.Y("value:Q", title=label,
                    axis=alt.Axis(format="$~s" if money_axis else "~s")),
            color=alt.Color("series:N", title=None,
                            scale=alt.Scale(domain=names)),
        )
        .add_params(hidden)
        .transform_filter(~hidden)
    )

    # Pivoting the series into columns lets one hover show every line's value at that
    # month, rather than only whichever line the cursor happens to be touching. The same
    # filter is applied first so a hidden series doesn't linger in the tooltip.
    hover = alt.selection_point(nearest=True, on="pointerover",
                                fields=["_date"], empty=False)
    hover_layer = (
        base.transform_filter(~hidden)
        .transform_pivot("series", value="value", groupby=["_date"])
        .mark_rule(color="gray")
        .encode(
            x=x_enc,
            opacity=alt.condition(hover, alt.value(0.35), alt.value(0)),
            tooltip=[alt.Tooltip("_date:T", title="Month", format="%b %Y")]
            + [alt.Tooltip(n, type="quantitative", format=value_format) for n in names],
        )
        .add_params(hover)
    )
    st.altair_chart((line + hover_layer).properties(height=280),
                    use_container_width=True)
    if len(names) > 1:
        st.caption(
            "Click a name in the legend to hide that line (the axis rescales to what's "
            "left); click it again to bring it back."
        )


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


def render_save_thesis(symbol: str, kind: str, view) -> None:
    """Offer to save an analyst view as a monitorable thesis."""
    from agent import theses

    def field(name, default=None):
        return view.get(name, default) if isinstance(view, dict) else getattr(view, name, default)

    triggers = field("triggers", []) or []
    st.divider()
    left, right = st.columns([3, 1])
    if triggers:
        left.caption(
            f"{len(triggers)} of these conditions can be measured, so they can be "
            "monitored automatically on the Theses page."
        )
    else:
        left.caption(
            "This view's conditions are qualitative, so it'll be saved for your own "
            "review rather than automatic checking."
        )
    if right.button("📌 Save as thesis", key=f"savethesis_{symbol}",
                    use_container_width=True):
        theses.save_thesis(symbol, kind, view)
        st.success(f"Saved — {symbol} is now tracked on the Theses page.")


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
