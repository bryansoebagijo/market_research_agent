"""Portfolio: what you actually hold, what it's worth, and where the risk sits."""

import pandas as pd
import streamlit as st

from agent import market_data as md, portfolio
from ui_common import coloured_pct, money

st.set_page_config(page_title="Portfolio · Markets Dashboard", page_icon="💼", layout="wide")
st.title("💼 Portfolio")


@st.cache_data(ttl=180, show_spinner=False)
def cached_prices(symbols: tuple[str, ...]):
    return md.price_many(list(symbols))


@st.cache_data(ttl=900, show_spinner=False)
def cached_correlation(symbols: tuple[str, ...]):
    return md.correlation_matrix(list(symbols))


positions = portfolio.list_positions()

# ---------------- add a position ----------------
with st.expander("➕ Add a position", expanded=not positions):
    with st.form("add_position", clear_on_submit=True):
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
        symbol = c1.text_input("Ticker", placeholder="AAVE or NVDA")
        kind = c2.selectbox("Type", ["crypto", "stocks"])
        quantity = c3.number_input("Quantity", min_value=0.0, step=1.0, format="%.6f")
        entry = c4.number_input("Entry price ($)", min_value=0.0, step=1.0, format="%.6f")
        note = st.text_input("Note (optional)", placeholder="Why you took this position")
        if st.form_submit_button("Add position", type="primary"):
            if symbol.strip() and quantity > 0 and entry > 0:
                portfolio.add_position(symbol, kind, quantity, entry, note=note)
                st.rerun()
            else:
                st.warning("Ticker, quantity and entry price are all required.")

if not positions:
    st.info(
        "No positions yet. Add one above — a watchlist tracks what you're watching, "
        "this tracks what you actually own, so every figure below is in real money."
    )
    st.stop()

# ---------------- value the book ----------------
symbols = tuple(sorted({p["symbol"] for p in positions}))
with st.spinner("Pricing positions…"):
    prices = cached_prices(symbols)
valued = portfolio.value_positions(positions, prices)

t1, t2, t3, t4 = st.columns(4)
t1.metric("Market value", money(valued["total_value"]))
t2.metric("Cost basis", money(valued["total_cost"]))
t3.metric(
    "Unrealised P&L",
    money(valued["total_pnl"]),
    f"{valued['total_pnl_pct']:+.2f}%" if valued["total_pnl_pct"] is not None else None,
)
t4.metric("Positions", len([r for r in valued["rows"] if r["value"] is not None]))

if valued["unpriced"]:
    st.warning(
        f"No current price for {', '.join(valued['unpriced'])} — excluded from totals. "
        "Check the ticker, or the data provider may be rate limited."
    )

# ---------------- positions table ----------------
st.subheader("Holdings")
st.dataframe(
    pd.DataFrame([
        {
            "Ticker": r["symbol"],
            "Type": r["kind"],
            "Qty": f"{r['quantity']:g}",
            "Entry": money(r["entry_price"]),
            "Price": money(r["price"]) if r["price"] is not None else "—",
            "Value": money(r["value"]) if r["value"] is not None else "—",
            "P&L": money(r["pnl"]) if r["pnl"] is not None else "—",
            "P&L %": f"{r['pnl_pct']:+.1f}%" if r["pnl_pct"] is not None else "—",
            "Weight": f"{r['weight_pct']:.1f}%" if r["weight_pct"] is not None else "—",
            "Opened": r.get("opened", ""),
        }
        for r in valued["rows"]
    ]),
    hide_index=True, use_container_width=True,
)

# Coloured P&L per position — a table can't render colour, so the movers are repeated
# here where green and red actually carry meaning.
movers = [r for r in valued["rows"] if r["pnl_pct"] is not None]
if movers:
    best = max(movers, key=lambda r: r["pnl_pct"])
    worst = min(movers, key=lambda r: r["pnl_pct"])
    m1, m2 = st.columns(2)
    m1.markdown(f"**Best** — {best['symbol']} {coloured_pct(best['pnl_pct'])} "
                f"({money(best['pnl'])})")
    m2.markdown(f"**Worst** — {worst['symbol']} {coloured_pct(worst['pnl_pct'])} "
                f"({money(worst['pnl'])})")

# ---------------- allocation ----------------
weighted = [r for r in valued["rows"] if r.get("weight_pct")]
if weighted:
    st.subheader("Allocation")
    st.bar_chart(
        pd.DataFrame(
            {"weight %": [r["weight_pct"] for r in weighted]},
            index=[r["symbol"] for r in weighted],
        ),
        height=240,
    )
    for note in portfolio.concentration_notes(valued):
        st.warning(note)

# ---------------- correlation ----------------
st.subheader("Correlation")
st.caption(
    "Daily-return correlation between holdings. Position count flatters a portfolio: "
    "holdings that move together are one bet wearing several hats, and that only shows "
    "up here."
)
if st.button("Check correlation"):
    st.session_state["corr_loaded"] = True

if st.session_state.get("corr_loaded"):
    with st.spinner("Comparing return histories…"):
        corr = cached_correlation(symbols)
    if len(corr["names"]) < 2:
        st.info("Need at least two positions with enough price history to compare.")
    else:
        st.dataframe(
            pd.DataFrame(corr["matrix"]).round(2),
            use_container_width=True,
        )
        strong = [p for p in corr["pairs"] if abs(p[2]) >= 0.7]
        if strong:
            for a, b, v in strong[:5]:
                st.warning(
                    f"{a} and {b} move together ({v:+.2f}) — sizing them separately "
                    "does not diversify as much as it appears."
                )
        else:
            st.success("No pair is strongly correlated (all below 0.7).")

# ---------------- manage ----------------
with st.expander("Manage positions"):
    for r in valued["rows"]:
        cols = st.columns([3, 1])
        label = (f"{r['symbol']} · {r['quantity']:g} @ {money(r['entry_price'])}"
                 f" · opened {r.get('opened','')}")
        cols[0].markdown(label + (f"  \n_{r['note']}_" if r.get("note") else ""))
        if cols[1].button("Remove", key=f"rm_{r['id']}"):
            portfolio.remove_position(r["id"])
            st.rerun()
