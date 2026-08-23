"""
Dashboard data: fast, structured market pulls for the Crypto and US Stocks pages.

This is deliberately separate from `tools.py`. Those tools return prose for the LLM to
read; these functions return structured rows for tables and charts, and they're built to
feel instant — no LLM anywhere, only keyless or already-configured APIs.

Nothing here caches on its own; the Streamlit pages wrap these in `st.cache_data` with a
TTL, so the caching policy lives with the UI rather than being baked in.
"""

from collections import defaultdict

import httpx

from agent.tools import (
    CRYPTO_FEEDS,
    STOCK_FEEDS,
    _get,
    _num,
    compute_ema,
    compute_rsi,
    crypto_technicals,
    stock_fundamentals,
    stock_snapshot,
    stock_technicals,
    _coingecko_id,
    _entry_date,
)

HEADERS = {"User-Agent": "markets-research-agent"}


# ---------------------------------------------------------------- news

def latest_headlines(kind: str, limit: int = 5) -> list[dict]:
    """Most recent headlines for 'crypto' or 'stock', newest first, across feeds."""
    import feedparser

    feeds = CRYPTO_FEEDS if kind == "crypto" else STOCK_FEEDS
    items: list[dict] = []
    for name, url in feeds.items():
        try:
            for e in feedparser.parse(url).entries[:10]:
                title = (getattr(e, "title", "") or "").strip()
                if title:
                    items.append({
                        "title": title,
                        "source": name,
                        "date": _entry_date(e),
                        "link": getattr(e, "link", "") or "",
                    })
        except Exception:  # noqa: BLE001 - one bad feed shouldn't empty the panel
            continue
    # Undated entries sort last; otherwise newest first.
    items.sort(key=lambda i: i["date"] or "", reverse=True)
    return items[:limit]


# ---------------------------------------------------------------- crypto

def top_open_interest(limit: int = 5) -> list[dict]:
    """Coins ranked by total open interest across every derivatives venue.

    CoinGecko's /derivatives lists ~25k individual contracts; summing `open_interest`
    per underlying index gives a market-wide view of where leverage actually sits.
    """
    rows = _get("https://api.coingecko.com/api/v3/derivatives")
    if not isinstance(rows, list):
        return []
    oi_by_coin: dict[str, float] = defaultdict(float)
    vol_by_coin: dict[str, float] = defaultdict(float)
    change: dict[str, list[float]] = defaultdict(list)
    for c in rows:
        idx, oi = c.get("index_id"), c.get("open_interest")
        if not idx or not oi:
            continue
        oi_by_coin[idx] += float(oi)
        vol_by_coin[idx] += float(c.get("volume_24h") or 0)
        pct = _num(c.get("price_percentage_change_24h"))
        if pct is not None:
            change[idx].append(pct)
    top = sorted(oi_by_coin.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    return [
        {
            "symbol": sym,
            "open_interest": oi,
            "volume_24h": vol_by_coin[sym],
            # Average across venues — individual contracts can diverge slightly.
            "change_24h": (sum(change[sym]) / len(change[sym])) if change[sym] else None,
        }
        for sym, oi in top
    ]


def trending_coins(limit: int = 7) -> list[dict]:
    """CoinGecko's search-popularity trending list."""
    d = _get("https://api.coingecko.com/api/v3/search/trending")
    coins = d.get("coins", []) if isinstance(d, dict) else []
    out = []
    for c in coins[:limit]:
        item = c.get("item", {})
        out.append({
            "symbol": (item.get("symbol") or "").upper(),
            "name": item.get("name"),
            "rank": item.get("market_cap_rank"),
            "id": item.get("id"),
        })
    return out


def coin_tvl(symbol: str) -> dict | None:
    """Total value locked for a chain whose native token is `symbol`, if any.

    TVL is a DeFi concept: it exists for smart-contract chains and protocols, not for
    every coin (Bitcoin has none). Returns None when the concept doesn't apply.
    """
    chains = httpx.get("https://api.llama.fi/v2/chains", timeout=25, headers=HEADERS).json()
    if not isinstance(chains, list):
        return None
    sym = symbol.upper()
    for c in chains:
        if (c.get("tokenSymbol") or "").upper() == sym:
            return {"name": c.get("name"), "tvl": c.get("tvl")}
    return None


def coin_overview(symbol: str) -> dict:
    """Everything the crypto detail panel shows for one coin."""
    out: dict = {"symbol": symbol.upper()}
    try:
        coin_id = _coingecko_id(symbol)
        if coin_id:
            markets = _get(
                "https://api.coingecko.com/api/v3/coins/markets",
                {"vs_currency": "usd", "ids": coin_id,
                 "price_change_percentage": "24h,7d,30d"},
            )
            if markets:
                m = markets[0]
                out.update({
                    "name": m.get("name"),
                    "price": m.get("current_price"),
                    "market_cap": m.get("market_cap"),
                    "volume_24h": m.get("total_volume"),
                    "change_24h": m.get("price_change_percentage_24h_in_currency"),
                    "change_7d": m.get("price_change_percentage_7d_in_currency"),
                    "change_30d": m.get("price_change_percentage_30d_in_currency"),
                    "ath_change": m.get("ath_change_percentage"),
                })
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)[:120]

    tech = crypto_technicals(symbol)
    out["technicals"] = tech.get("data") or {}
    out["technicals_text"] = tech.get("text", "")

    try:
        out["tvl"] = coin_tvl(symbol)
    except Exception:  # noqa: BLE001 - TVL is a nice-to-have, never fatal
        out["tvl"] = None
    return out


# ---------------------------------------------------------------- stocks

# Yahoo's "trending US" list mixes in crypto and FX pairs; these suffixes mark them.
_NON_EQUITY_SUFFIXES = ("-USD", "-EUR", "-CAD", "-GBP", "=X", "=F", "^")


def trending_stocks(limit: int = 7) -> list[str]:
    """Yahoo's trending tickers, filtered down to actual US equities."""
    d = httpx.get(
        "https://query1.finance.yahoo.com/v1/finance/trending/US",
        timeout=20, headers=HEADERS,
    ).json()
    results = (d.get("finance", {}).get("result") or [{}])[0]
    symbols = [q.get("symbol", "") for q in results.get("quotes", [])]
    equities = [
        s for s in symbols
        if s and not any(s.endswith(sfx) or s.startswith("^") for sfx in _NON_EQUITY_SUFFIXES)
    ]
    return equities[:limit]


def stock_overview(symbol: str, with_fundamentals: bool = True) -> dict:
    """Everything the stock detail panel shows for one ticker."""
    out: dict = {"symbol": symbol.upper()}

    snap = stock_snapshot(symbol)
    out["quote_text"] = snap.get("text", "")
    out.update(snap.get("data") or {})

    tech = stock_technicals(symbol)
    out["technicals"] = tech.get("data") or {}
    out["technicals_text"] = tech.get("text", "")

    if with_fundamentals:
        f = stock_fundamentals(symbol)
        out["fundamentals_text"] = f.get("text", "")
    return out


# ---------------------------------------------------------------- shared

def quick_signal(tech: dict) -> tuple[str, str]:
    """A fast, rule-based read of the technicals — no LLM, so it's instant.

    This is explicitly NOT the analyst's view: it only describes what the indicators
    say right now, so the dashboard can show something useful before anyone spends a
    minute (and API quota) on the full agent run.
    """
    rsi, ema13, ema21, price = (
        tech.get("rsi"), tech.get("ema13"), tech.get("ema21"), tech.get("price")
    )
    if rsi is None or ema13 is None or ema21 is None or price is None:
        return "No data", "Indicators unavailable."

    if ema13 > ema21 and price > ema13:
        trend = "Uptrend"
    elif ema13 < ema21 and price < ema13:
        trend = "Downtrend"
    else:
        trend = "No clear trend"

    if rsi >= 70:
        note = f"{trend}, but RSI {rsi:.0f} is overbought — stretched."
    elif rsi <= 30:
        note = f"{trend}, RSI {rsi:.0f} is oversold — washed out."
    else:
        note = f"{trend}, RSI {rsi:.0f} is neutral."
    return trend, note
