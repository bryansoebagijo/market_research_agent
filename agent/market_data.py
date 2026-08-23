"""
Dashboard data: fast, structured market pulls for the Crypto and US Stocks pages.

This is deliberately separate from `tools.py`. Those tools return prose for the LLM to
read; these functions return structured rows for tables and charts, and they're built to
feel instant — no LLM anywhere, only keyless or already-configured APIs.

Nothing here caches on its own; the Streamlit pages wrap these in `st.cache_data` with a
TTL, so the caching policy lives with the UI rather than being baked in.
"""

import re
from collections import defaultdict
from datetime import datetime, timezone

import httpx

from agent.tools import (
    CRYPTO_FEEDS,
    STOCK_FEEDS,
    _get,
    _num,
    compute_ema,
    compute_rsi,
    crypto_technicals,
    stock_earnings,
    stock_fundamentals,
    stock_snapshot,
    stock_technicals,
    _coingecko_id,
    _entry_date,
)

HEADERS = {"User-Agent": "markets-research-agent"}


# ---------------------------------------------------------------- news

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean_summary(entry, max_chars: int = 260) -> str:
    """Plain-text summary from a feed entry, HTML stripped and trimmed.

    Feeds disagree about this: some populate `summary`, some `description`, some
    neither. Returns "" rather than inventing text when a feed provides none.
    """
    raw = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
    text = _WS_RE.sub(" ", _TAG_RE.sub(" ", raw)).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "…"
    return text


def _entry_image(entry) -> str:
    """Best available image URL for a feed entry, or "" if it has none.

    Feeds expose images in several different places, so try each in turn:
    media:content, media:thumbnail, an image enclosure, then an <img> in the HTML.
    """
    for key in ("media_content", "media_thumbnail"):
        media = getattr(entry, key, None)
        if media:
            url = (media[0] or {}).get("url")
            if url:
                return url
    for link in getattr(entry, "links", []) or []:
        if (link.get("type") or "").startswith("image") and link.get("href"):
            return link["href"]
    html = getattr(entry, "summary", "") or getattr(entry, "description", "") or ""
    match = re.search(r'<img[^>]+src=["\']([^"\']+)', html)
    return match.group(1) if match else ""


def latest_headlines(kind: str, limit: int = 5) -> list[dict]:
    """Most recent headlines for 'crypto' or 'stock', newest first, across feeds.

    Each item carries a summary and an image where the feed supplies them; both are
    empty strings otherwise, so the UI can lay out around what's actually available.
    """
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
                        "summary": _clean_summary(e),
                        "image": _entry_image(e),
                    })
        except Exception:  # noqa: BLE001 - one bad feed shouldn't empty the panel
            continue

    # Undated entries sort last; otherwise newest first.
    items.sort(key=lambda i: i["date"] or "", reverse=True)

    # Cap items per outlet so the panel spans several sources. The most prolific feed
    # would otherwise fill every slot — a worse read, and it hides the outlets that
    # actually supply summaries.
    per_source = max(1, limit // 2)
    picked, counts = [], defaultdict(int)
    for item in items:
        if counts[item["source"]] < per_source:
            picked.append(item)
            counts[item["source"]] += 1
        if len(picked) == limit:
            break
    # If a cap left us short (few sources live), top up with the freshest remainder.
    if len(picked) < limit:
        chosen = {(i["source"], i["title"]) for i in picked}
        picked += [i for i in items if (i["source"], i["title"]) not in chosen][
            : limit - len(picked)
        ]
    return picked[:limit]


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


def _defillama_protocols() -> list[dict]:
    """Every DefiLlama protocol, joined with its 24h/30d revenue where it earns any.

    Revenue and TVL come from two different endpoints whose names don't always agree
    ("Circle USDC" vs "Circle"), so join on DefiLlama's own ids first and fall back to
    slug, then name. Note that plenty of high-revenue protocols legitimately have no
    TVL — a stablecoin issuer or a trading bot earns fees without locking value.
    """
    fees = httpx.get(
        "https://api.llama.fi/overview/fees?excludeTotalDataChart=true&dataType=dailyRevenue",
        timeout=45, headers=HEADERS,
    ).json()
    protocols = httpx.get("https://api.llama.fi/protocols", timeout=60, headers=HEADERS).json()
    if not isinstance(protocols, list):
        return []

    by_id, by_slug, by_name = {}, {}, {}
    for p in protocols:
        tvl = p.get("tvl")
        if p.get("id"):
            by_id[str(p["id"])] = tvl
        if p.get("slug"):
            by_slug[str(p["slug"]).lower()] = tvl
        if p.get("name"):
            by_name[str(p["name"]).lower()] = tvl

    def tvl_for(entry: dict):
        return (
            by_id.get(str(entry.get("defillamaId")))
            or by_slug.get(str(entry.get("slug", "")).lower())
            or by_name.get(str(entry.get("name", "")).lower())
        )

    rows: list[dict] = []
    seen: set[str] = set()
    for e in (fees.get("protocols") or []) if isinstance(fees, dict) else []:
        name = e.get("displayName") or e.get("name")
        if not name or name in seen:
            continue
        seen.add(name)
        rows.append({
            "name": name,
            "slug": e.get("slug") or "",
            "category": e.get("category") or "—",
            "revenue_24h": _num(e.get("total24h")),
            "revenue_30d": _num(e.get("total30d")),
            # The 30 days *before* the last 30 — the baseline for month-on-month growth.
            "revenue_prev_30d": _num(e.get("total60dto30d")),
            "tvl": tvl_for(e),
            "logo": e.get("logo") or "",
        })

    # Protocols with TVL but no fee data still belong in the TVL ranking.
    for p in protocols:
        name = p.get("name")
        if name and name not in seen and p.get("tvl"):
            rows.append({
                "name": name,
                "slug": p.get("slug") or "",
                "category": p.get("category") or "—",
                "revenue_24h": None,
                "revenue_30d": None,
                "revenue_prev_30d": None,
                "tvl": p.get("tvl"),
                "logo": p.get("logo") or "",
            })
    return rows


def _tvl_change_30d(slug: str) -> float | None:
    """Percentage change in a protocol's TVL over the last 30 days.

    DefiLlama's bulk /protocols feed only exposes 1h/1d/7d changes, so a monthly figure
    needs the protocol's own history endpoint — one request each, which is why callers
    only ask for a shortlist.
    """
    if not slug:
        return None
    try:
        d = httpx.get(f"https://api.llama.fi/protocol/{slug}", timeout=40, headers=HEADERS).json()
        series = d.get("tvl") or []
        if len(series) < 32:
            return None
        now = series[-1].get("totalLiquidityUSD")
        past = series[-31].get("totalLiquidityUSD")
        if not now or not past:
            return None
        return (now / past - 1) * 100
    except Exception:  # noqa: BLE001 - a missing history shouldn't drop the whole panel
        return None


# Floors that keep the growth ranking meaningful: a protocol earning $200 last month and
# $2,000 this month is up 900%, which is noise rather than a trend worth showing.
MIN_REVENUE_30D = 50_000
MIN_TVL = 10_000_000


def top_growing_protocols(limit: int = 5, candidates: int = 10) -> list[dict]:
    """Projects whose revenue AND TVL are both growing month over month.

    Revenue growth comes from the bulk fees feed (last 30 days vs the 30 before it).
    TVL growth needs a per-protocol request, so only the strongest revenue growers of
    meaningful size are checked, and the result keeps just those growing on both.
    """
    rows = []
    for r in _defillama_protocols():
        if r.get("category") in _NON_PROTOCOL_CATEGORIES:
            continue
        cur, prev, tvl = r.get("revenue_30d"), r.get("revenue_prev_30d"), r.get("tvl")
        if not cur or not prev or prev <= 0 or not tvl:
            continue
        # BOTH periods must be meaningful. Requiring only the current month lets a
        # protocol that earned a few hundred dollars last month show +40,000% growth,
        # which crowds out real trends.
        if cur < MIN_REVENUE_30D or prev < MIN_REVENUE_30D or tvl < MIN_TVL:
            continue
        growth = (cur / prev - 1) * 100
        if growth <= 0:
            continue
        rows.append({**r, "revenue_growth_30d": growth})

    rows.sort(key=lambda r: r["revenue_growth_30d"], reverse=True)

    out = []
    for r in rows[:candidates]:
        tvl_growth = _tvl_change_30d(r["slug"])
        if tvl_growth is not None and tvl_growth > 0:
            out.append({**r, "tvl_growth_30d": tvl_growth})
        if len(out) == limit:
            break
    return out


# Centralised exchanges report custodial reserves as "TVL", which dwarfs and crowds out
# actual protocol TVL. They aren't DeFi projects, so they're excluded from the rankings.
_NON_PROTOCOL_CATEGORIES = {"CEX", "Chain"}


def top_protocols(by: str = "revenue_24h", limit: int = 5) -> list[dict]:
    """Top DeFi projects ranked by 24h revenue or by TVL (exchanges excluded)."""
    rows = [
        r for r in _defillama_protocols()
        if r.get(by) is not None and r.get("category") not in _NON_PROTOCOL_CATEGORIES
    ]
    rows.sort(key=lambda r: r[by], reverse=True)
    return rows[:limit]


def _month_key(timestamp) -> str | None:
    """'YYYY-MM' for a unix timestamp."""
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m")
    except (TypeError, ValueError, OSError):
        return None


def protocol_history(slug: str, months: int = 12) -> dict:
    """Monthly TVL and revenue history for one protocol.

    The two series aggregate differently and mixing that up would be misleading:
      * TVL is a level (a stock), so a month's value is its LAST reading.
      * Revenue is a flow, so a month's value is the SUM of that month's days.

    The current, partial month is dropped from revenue — a month that's three days old
    would otherwise look like a collapse next to complete months.
    """
    out: dict = {"slug": slug, "tvl_monthly": [], "revenue_monthly": []}
    if not slug:
        return out
    this_month = datetime.now(tz=timezone.utc).strftime("%Y-%m")

    # --- TVL: last reading of each month ---
    try:
        d = httpx.get(f"https://api.llama.fi/protocol/{slug}", timeout=45, headers=HEADERS).json()
        last_of_month: dict[str, float] = {}
        for point in d.get("tvl") or []:
            key = _month_key(point.get("date"))
            value = point.get("totalLiquidityUSD")
            if key and value is not None:
                last_of_month[key] = float(value)  # later points overwrite earlier ones
        out["tvl_monthly"] = [
            {"month": k, "tvl": v} for k, v in sorted(last_of_month.items())
        ][-months:]
    except Exception as exc:  # noqa: BLE001
        out["tvl_error"] = str(exc)[:100]

    # --- Revenue: sum of each month's days ---
    try:
        d = httpx.get(
            f"https://api.llama.fi/summary/fees/{slug}?dataType=dailyRevenue",
            timeout=45, headers=HEADERS,
        ).json()
        totals: dict[str, float] = defaultdict(float)
        for point in d.get("totalDataChart") or []:
            if not isinstance(point, list) or len(point) < 2:
                continue
            key = _month_key(point[0])
            if key and point[1] is not None:
                totals[key] += float(point[1])
        totals.pop(this_month, None)  # drop the incomplete current month
        out["revenue_monthly"] = [
            {"month": k, "revenue": v} for k, v in sorted(totals.items())
        ][-months:]
    except Exception as exc:  # noqa: BLE001
        out["revenue_error"] = str(exc)[:100]

    return out


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


def stock_growth(symbol: str) -> dict | None:
    """Year-over-year revenue and earnings growth for one stock, plus margin direction.

    Derived entirely from the data `stock_earnings` already fetches, so ranking a
    watchlist costs no more API calls than looking those stocks up individually.
    Year-over-year (this quarter vs the same quarter last year) rather than
    quarter-on-quarter, because most businesses are seasonal.
    """
    data = (stock_earnings(symbol) or {}).get("data") or {}
    quarters = data.get("revenue_quarters") or []   # oldest first
    years = data.get("eps_years") or []             # oldest first

    out: dict = {"symbol": symbol.upper()}

    if len(quarters) >= 5:
        latest, year_ago = quarters[-1], quarters[-5]
        rev_now, rev_then = latest.get("revenue"), year_ago.get("revenue")
        if rev_now and rev_then:
            out["revenue_latest"] = rev_now
            out["revenue_yoy"] = (rev_now / rev_then - 1) * 100
            out["quarter"] = latest.get("quarter")
        # Margin direction in percentage points — expanding or compressing.
        m_now, m_then = latest.get("operating_margin"), year_ago.get("operating_margin")
        if m_now is not None and m_then is not None:
            out["operating_margin"] = m_now
            out["margin_change_pp"] = m_now - m_then

    if len(years) >= 2:
        eps_now, eps_prev = years[-1].get("eps"), years[-2].get("eps")
        if eps_now is not None and eps_prev:
            out["eps_latest"] = eps_now
            out["eps_yoy"] = (eps_now / eps_prev - 1) * 100 if eps_prev > 0 else None

    return out if "revenue_yoy" in out or "eps_yoy" in out else None


def stock_growth_ranking(symbols: list[str]) -> list[dict]:
    """Rank a set of tickers by year-over-year revenue growth.

    Scoped to a supplied list rather than the whole market on purpose: Alpha Vantage is
    a per-symbol API on a ~25 requests/day free tier, so a market-wide scan isn't
    possible the way it is for DeFi protocols.
    """
    rows = []
    for sym in symbols:
        try:
            g = stock_growth(sym)
        except Exception:  # noqa: BLE001 - one bad symbol shouldn't drop the table
            g = None
        if g:
            rows.append(g)
    rows.sort(key=lambda r: r.get("revenue_yoy") if r.get("revenue_yoy") is not None else -1e9,
              reverse=True)
    return rows


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
