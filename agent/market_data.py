"""
Dashboard data: fast, structured market pulls for the Crypto and US Stocks pages.

This is deliberately separate from `tools.py`. Those tools return prose for the LLM to
read; these functions return structured rows for tables and charts, and they're built to
feel instant — no LLM anywhere, only keyless or already-configured APIs.

Nothing here caches on its own; the Streamlit pages wrap these in `st.cache_data` with a
TTL, so the caching policy lives with the UI rather than being baked in.
"""

import re
import time
from collections import defaultdict
from datetime import datetime, timezone

import httpx

from agent.tools import (
    CRYPTO_FEEDS,
    STOCK_FEEDS,
    _get,
    _num,
    classify_ticker,
    compute_ema,
    compute_rsi,
    ETF_TYPES,
    crypto_technicals,
    etf_flow_data as _etf_flow_data,
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


# ---------------------------------------------- US spot ETF flows

# The fetching and parsing live in agent/tools.py, because market_data imports FROM tools
# and the agent's own ETF tool needs the same parsed series. Re-exported here so the
# dashboard keeps importing it from the module it already uses.
etf_flows = _etf_flow_data


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


# Narrative ranking floor. Without one the list fills with micro-cap categories posting
# +800% on a few thousand dollars of market cap — technically the biggest movers, but
# noise rather than a narrative anyone is actually trading.
MIN_NARRATIVE_MARKET_CAP = 500_000_000


_COIN_SYMBOLS: dict = {"at": 0.0, "map": None}


def _coin_id_to_symbol() -> dict[str, str]:
    """CoinGecko coin id -> ticker, for the whole coin universe.

    Category endpoints identify their top coins by id ("ether-fi"), not ticker, so one
    cached lookup of the full list turns those into the symbols people recognise
    (ETHFI). ~1.2MB and 18k coins, so it's fetched once an hour rather than per call.
    """
    now = time.time()
    if _COIN_SYMBOLS["map"] is not None and now - _COIN_SYMBOLS["at"] < 3600:
        return _COIN_SYMBOLS["map"]
    try:
        rows = _get("https://api.coingecko.com/api/v3/coins/list")
        mapping = {
            c["id"]: (c.get("symbol") or "").upper()
            for c in rows if isinstance(c, dict) and c.get("id")
        }
    except Exception:  # noqa: BLE001 - fall back to showing ids
        mapping = {}
    _COIN_SYMBOLS.update({"at": now, "map": mapping})
    return mapping


def trending_narratives(limit: int = 5, min_market_cap: float = MIN_NARRATIVE_MARKET_CAP) -> list[dict]:
    """Hottest crypto sectors right now, ranked by 24h change in sector market cap.

    CoinGecko groups coins into categories — Restaking, AI Agents, RWA and so on — which
    is as close as the free data gets to the "narratives" traders talk about. Sector
    market-cap change is used rather than a single coin's move, so a narrative only
    ranks when the whole basket is moving.
    """
    rows = _get("https://api.coingecko.com/api/v3/coins/categories")
    if not isinstance(rows, list):
        return []
    candidates = [
        c for c in rows
        if (c.get("market_cap") or 0) >= min_market_cap
        and c.get("market_cap_change_24h") is not None
    ]
    candidates.sort(key=lambda c: c["market_cap_change_24h"], reverse=True)

    symbols = _coin_id_to_symbol()
    out = []
    for c in candidates[:limit]:
        # top_3_coins_id holds ids ("ether-fi"); fall back to parsing the image URLs in
        # top_3_coins when the id list is absent.
        coin_ids = c.get("top_3_coins_id") or []
        if not coin_ids:
            coin_ids = [
                url.split("/")[-1].split(".")[0]
                for url in (c.get("top_3_coins") or [])
            ]
        tickers, names = [], []
        for cid in coin_ids[:3]:
            ticker = symbols.get(str(cid))
            names.append(str(cid).replace("-", " ").title())
            tickers.append(ticker or str(cid).replace("-", " ").title())
        out.append({
            "name": c.get("name"),
            "change_24h": c.get("market_cap_change_24h"),
            "market_cap": c.get("market_cap"),
            "volume_24h": c.get("volume_24h"),
            "top_coins": tickers,
            "top_coin_names": names,
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


# The DefiLlama join is several megabytes and every ranking needs it, so the result is
# memoised for the process. Without this a single page load repeated the whole fetch
# once per tab.
_PROTOCOL_CACHE: dict = {"at": 0.0, "rows": None}
# Held for a while because the fetch is expensive and the data moves slowly; a
# successful pull shouldn't be thrown away just because DefiLlama is having a bad hour.
_PROTOCOL_CACHE_SECONDS = 900


def _defillama_protocols() -> list[dict]:
    """Every DefiLlama protocol, joined with its 24h/30d revenue where it earns any.

    Revenue and TVL come from two different endpoints whose names don't always agree
    ("Circle USDC" vs "Circle"), so join on DefiLlama's own ids first and fall back to
    slug, then name. Note that plenty of high-revenue protocols legitimately have no
    TVL — a stablecoin issuer or a trading bot earns fees without locking value.
    """
    now = time.time()
    if _PROTOCOL_CACHE["rows"] is not None and now - _PROTOCOL_CACHE["at"] < _PROTOCOL_CACHE_SECONDS:
        return _PROTOCOL_CACHE["rows"]

    # TVL first: /protocols is fast and reliable, and it alone powers the TVL ranking.
    protocols = httpx.get("https://api.llama.fi/protocols", timeout=60, headers=HEADERS).json()
    if not isinstance(protocols, list):
        return []

    # Revenue is fetched separately and allowed to fail. The fees endpoint is often slow
    # (observed at 60-80s) and a timeout there shouldn't take the TVL ranking down with
    # it — the revenue columns just read "n/a" instead.
    # excludeTotalDataChartBreakdown matters: without it the response carries a per-chain
    # daily history for every protocol, 21MB instead of 3.6MB.
    fees: dict = {}
    try:
        fees = httpx.get(
            "https://api.llama.fi/overview/fees",
            params={
                "excludeTotalDataChart": "true",
                "excludeTotalDataChartBreakdown": "true",
                "dataType": "dailyRevenue",
            },
            # Short on purpose. A healthy DefiLlama answers in 1-2s; when it's degraded
            # it takes 60-80s, and making the page wait that long to end up degrading
            # anyway is worse than giving up quickly and showing TVL.
            timeout=20, headers=HEADERS,
        ).json()
    except Exception:  # noqa: BLE001 - degrade to TVL-only rather than failing outright
        fees = {}

    # Keyed three ways because the two endpoints don't always agree on a name. Each
    # holds (tvl, token symbol) — the symbol is what lets a protocol's revenue be
    # compared against its token's market cap.
    by_id, by_slug, by_name = {}, {}, {}
    for p in protocols:
        pair = (p.get("tvl"), (p.get("symbol") or "").upper())
        if p.get("id"):
            by_id[str(p["id"])] = pair
        if p.get("slug"):
            by_slug[str(p["slug"]).lower()] = pair
        if p.get("name"):
            by_name[str(p["name"]).lower()] = pair

    def pair_for(entry: dict):
        return (
            by_id.get(str(entry.get("defillamaId")))
            or by_slug.get(str(entry.get("slug", "")).lower())
            or by_name.get(str(entry.get("name", "")).lower())
            or (None, "")
        )

    def tvl_for(entry: dict):
        return pair_for(entry)[0]

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
            "symbol": pair_for(e)[1],
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

    _PROTOCOL_CACHE.update({"at": now, "rows": rows})
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


def protocol_history(slug: str, months: int = 12, include_partial: bool = False) -> dict:
    """Monthly TVL and revenue history for one protocol.

    The two series aggregate differently and mixing that up would be misleading:
      * TVL is a level (a stock), so a month's value is its LAST reading.
      * Revenue is a flow, so a month's value is the SUM of that month's days.

    By default the current, partial month is dropped from revenue — a month three days
    old would otherwise look like a collapse next to complete ones. Set include_partial
    when the point IS the recent move: a growth view that hides the current month hides
    the growth. The month kept is reported back as `partial_month` so callers can label
    it rather than presenting month-to-date as a finished month.
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
        if include_partial:
            if this_month in totals:
                out["partial_month"] = this_month
        else:
            totals.pop(this_month, None)  # drop the incomplete current month
        out["revenue_monthly"] = [
            {"month": k, "revenue": v} for k, v in sorted(totals.items())
        ][-months:]
    except Exception as exc:  # noqa: BLE001
        out["revenue_error"] = str(exc)[:100]

    return out


_MULTIPLES_CACHE: dict = {"at": 0.0, "rows": None}

# Categories where a token's market cap simply isn't a claim on the fee line DefiLlama
# reports. An L1's value is a monetary/ecosystem claim, not a multiple of one chain's
# revenue, and pricing it that way would be nonsense dressed up as analysis.
MULTIPLE_EXCLUDED_CATEGORIES = {"Chain", "CEX"}

# A multiple on a tiny revenue base is arithmetic, not information.
MIN_ANNUALISED_REVENUE = 5_000_000

# Past this, the market cap is almost certainly pricing something far beyond the fees
# being measured (an exchange token, an ecosystem bet), so the multiple gets a caveat
# rather than being presented as a valuation.
IMPLAUSIBLE_MULTIPLE = 500


def _multiples_universe() -> list[dict]:
    """Every protocol token whose revenue can be honestly compared to its market cap.

    Two traps make the naive version misleading, and both showed up in the data:

    * DefiLlama lists sub-products separately, each carrying the parent token's ticker.
      Valuing Jupiter's whole market cap against its Lend product alone gave 294,755x;
      summing every product under a ticker gives 9.7x. So revenue is aggregated per
      token, not per protocol entry.
    * Chain and CEX tokens price an entire ecosystem, so dividing by one protocol's
      fees produced 2,100x for SOL. Those categories are excluded outright.
    """
    now = time.time()
    if _MULTIPLES_CACHE["rows"] is not None and now - _MULTIPLES_CACHE["at"] < 1800:
        return _MULTIPLES_CACHE["rows"]

    per_token: dict[str, dict] = {}
    for p in _defillama_protocols():
        sym = (p.get("symbol") or "").upper()
        revenue = p.get("revenue_30d") or 0
        if sym in ("", "-") or not revenue:
            continue
        if p.get("category") in MULTIPLE_EXCLUDED_CATEGORIES:
            continue
        entry = per_token.setdefault(
            sym, {"symbol": sym, "revenue_30d": 0.0, "tvl": 0.0,
                  "by_category": {}, "products": []}
        )
        entry["revenue_30d"] += revenue
        entry["tvl"] += p.get("tvl") or 0
        entry["products"].append(p.get("name") or "")
        cat = p.get("category") or "—"
        entry["by_category"][cat] = entry["by_category"].get(cat, 0) + revenue

    try:
        markets = _get(
            "https://api.coingecko.com/api/v3/coins/markets",
            {"vs_currency": "usd", "order": "market_cap_desc",
             "per_page": 250, "page": 1},
        )
    except Exception:  # noqa: BLE001
        return []
    caps = {
        (m.get("symbol") or "").upper(): m.get("market_cap")
        for m in (markets if isinstance(markets, list) else [])
        if m.get("market_cap")
    }

    rows = []
    for sym, e in per_token.items():
        cap = caps.get(sym)
        annual = e["revenue_30d"] * 12
        if not cap or annual < MIN_ANNUALISED_REVENUE:
            continue
        rows.append({
            "symbol": sym,
            "market_cap": cap,
            "annual_revenue": annual,
            "tvl": e["tvl"] or None,
            # The category earning most of this token's revenue is the fair peer group.
            "category": max(e["by_category"], key=e["by_category"].get),
            "product_count": len(e["products"]),
            "ps": cap / annual,
            "mcap_to_tvl": (cap / e["tvl"]) if e["tvl"] else None,
        })
    _MULTIPLES_CACHE.update({"at": now, "rows": rows})
    return rows


def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2


def token_multiples(symbol: str) -> dict:
    """Crypto's price-to-sales: market cap against annualised protocol revenue.

    Returned with a peer median, because a bare multiple carries no information — 34x
    only means something once you know comparable protocols trade at 20x. Peers are the
    same DefiLlama category where at least three exist, since lending and DEX tokens
    aren't priced alike; otherwise it falls back to all protocol tokens and says so.
    """
    sym = symbol.upper()
    universe = _multiples_universe()
    me = next((r for r in universe if r["symbol"] == sym), None)
    if me is None:
        return {"symbol": sym, "available": False}

    peers = [r for r in universe if r["category"] == me["category"] and r["symbol"] != sym]
    basis = me["category"]
    if len(peers) < 2:
        peers = [r for r in universe if r["symbol"] != sym]
        basis = "all protocol tokens"

    peer_ps = _median([r["ps"] for r in peers])
    return {
        **me,
        "available": True,
        "peer_basis": basis,
        "peer_count": len(peers),
        "peer_median_ps": peer_ps,
        "vs_peers_pct": ((me["ps"] / peer_ps - 1) * 100) if peer_ps else None,
        "cheaper_than": (
            sum(1 for r in peers if r["ps"] > me["ps"]) / len(peers) * 100
            if peers else None
        ),
        "implausible": me["ps"] > IMPLAUSIBLE_MULTIPLE,
    }


def coin_price_history(symbol: str, days: int = 365) -> list[dict]:
    """Daily closes with their dates, oldest first — enough history to read weekly swings.

    The technicals tool fetches 95 days, which is ample for daily indicators but only
    about 13 weekly bars: too few swings to call a weekly structure. This asks for the
    free tier's full year instead.
    """
    coin_id = _coingecko_id(symbol)
    if not coin_id:
        return []
    chart = _get(
        f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
        {"vs_currency": "usd", "days": days, "interval": "daily"},
    )
    points = chart.get("prices", []) if isinstance(chart, dict) else []
    return [
        {
            "date": datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d"),
            "close": float(price),
        }
        for ms, price in points if price
    ]


def current_metrics(symbol: str) -> dict:
    """Current values for every metric a thesis trigger can reference.

    Mostly built from the already-cached technicals, protocol health and growth calls, so
    monitoring a thesis costs little beyond what viewing the asset would. The exception is
    the ETF flow pair, which is one extra request for BTC and ETH — worth it because the
    analyst's invalidation conditions for those two are usually about flows, and a
    condition nothing can read is one the monitor has to hand back to you as prose.
    """
    kind = classify_ticker(symbol)
    out: dict = {}

    if symbol.upper() in ETF_TYPES:
        try:
            # days=1 trims only the returned daily series; the windows below are computed
            # from the full history regardless, so this keeps the payload small.
            flows = _etf_flow_data(symbol, days=1)
            if flows.get("flow_counted") is not None:
                # Millions, matching the threshold scale the analyst is asked to use.
                out["etf_flow_20d"] = flows["flow_counted"] / 1e6
            out["etf_sessions_positive"] = flows["sessions_positive"]
        except Exception:  # noqa: BLE001 - a flow outage shouldn't cost the other checks
            pass

    tech = (crypto_technicals(symbol) if kind == "crypto" else stock_technicals(symbol))
    data = tech.get("data") or {}
    if data.get("rsi") is not None:
        out["rsi"] = data["rsi"]
    if data.get("price") is not None:
        out["price"] = data["price"]

    if kind == "crypto":
        resolved = None
        try:
            resolved = coin_protocol_slug(symbol)
        except Exception:  # noqa: BLE001
            resolved = None
        if resolved and resolved[0]:
            health = protocol_health(protocol_history(resolved[0], 12))
            for src, dest in (("revenue_3m", "revenue_growth_3m"),
                              ("tvl_3m", "tvl_growth_3m"),
                              ("revenue_to_tvl_pct", "revenue_to_tvl")):
                if health.get(src) is not None:
                    out[dest] = health[src]
    else:
        growth = stock_growth(symbol) or {}
        for src, dest in (("revenue_yoy", "revenue_yoy"),
                          ("eps_yoy", "eps_yoy"),
                          ("operating_margin", "operating_margin")):
            if growth.get(src) is not None:
                out[dest] = growth[src]
        fin = stock_financials(symbol)
        quarters = fin.get("quarters") or []
        if quarters and quarters[-1].get("net_margin") is not None:
            out["net_margin"] = quarters[-1]["net_margin"]
    return out


def price_many(symbols: list[str]) -> dict[str, float]:
    """Current prices for a mixed list of crypto and stock tickers.

    Crypto is fetched in ONE batched CoinGecko call rather than per coin, because that
    endpoint is the slow one — pricing a portfolio shouldn't cost a request per holding.
    Stocks go through the existing cached per-symbol quote.
    """
    prices: dict[str, float] = {}
    crypto = [s.upper() for s in symbols if classify_ticker(s) == "crypto"]
    stocks = [s.upper() for s in symbols if classify_ticker(s) != "crypto"]

    if crypto:
        ids: dict[str, str] = {}
        for sym in crypto:
            try:
                coin_id = _coingecko_id(sym)
            except Exception:  # noqa: BLE001
                coin_id = None
            if coin_id:
                ids[coin_id] = sym
        if ids:
            try:
                rows = _get(
                    "https://api.coingecko.com/api/v3/coins/markets",
                    {"vs_currency": "usd", "ids": ",".join(ids)},
                )
                for m in rows if isinstance(rows, list) else []:
                    sym = ids.get(m.get("id"))
                    if sym and m.get("current_price") is not None:
                        prices[sym] = float(m["current_price"])
            except Exception:  # noqa: BLE001 - unpriced holdings are reported, not fatal
                pass

    for sym in stocks:
        try:
            data = stock_snapshot(sym).get("data") or {}
            if data.get("price") is not None:
                prices[sym] = float(data["price"])
        except Exception:  # noqa: BLE001
            continue
    return prices


def daily_returns(symbol: str) -> list[float]:
    """Daily returns from the close series the technicals already fetch (cached)."""
    kind = classify_ticker(symbol)
    tech = (crypto_technicals(symbol) if kind == "crypto" else stock_technicals(symbol))
    closes = (tech.get("data") or {}).get("closes") or []
    return [
        closes[i] / closes[i - 1] - 1
        for i in range(1, len(closes))
        if closes[i - 1]
    ]


def correlation_matrix(symbols: list[str]) -> dict:
    """Pairwise correlation of daily returns between holdings.

    This is the risk most retail dashboards never surface: five positions that move
    together are one bet wearing five hats, and position count alone hides that.
    """
    series = {s.upper(): daily_returns(s) for s in symbols}
    series = {s: r for s, r in series.items() if len(r) >= 20}
    names = list(series)

    def corr(a: list[float], b: list[float]) -> float | None:
        n = min(len(a), len(b))
        if n < 20:
            return None
        a, b = a[-n:], b[-n:]
        ma, mb = sum(a) / n, sum(b) / n
        va = sum((x - ma) ** 2 for x in a)
        vb = sum((y - mb) ** 2 for y in b)
        if va <= 0 or vb <= 0:
            return None
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        return cov / (va ** 0.5 * vb ** 0.5)

    matrix = {
        x: {y: (1.0 if x == y else corr(series[x], series[y])) for y in names}
        for x in names
    }
    pairs = [
        (x, y, matrix[x][y])
        for i, x in enumerate(names) for y in names[i + 1:]
        if matrix[x][y] is not None
    ]
    pairs.sort(key=lambda p: abs(p[2]), reverse=True)
    return {"names": names, "matrix": matrix, "pairs": pairs}


def _window_change(rows: list[dict], key: str, periods: int) -> float | None:
    """Percentage change over the last `periods` entries of a monthly series."""
    points = [r for r in rows if r.get(key) is not None]
    if len(points) < 2:
        return None
    start = points[-(periods + 1)] if len(points) > periods else points[0]
    first, last = start[key], points[-1][key]
    return ((last / first - 1) * 100) if first else None


def protocol_health(history: dict) -> dict:
    """Assess a protocol from its revenue and TVL history alone.

    The verdict comes from the RELATIONSHIP between the two, which is more telling than
    either on its own: revenue rising while TVL falls means capital is working harder,
    while TVL rising as revenue falls is fee compression — more money parked, less
    earned on it. Deliberately rule-based, so it's instant and always says the same
    thing about the same numbers.
    """
    rev = history.get("revenue_monthly") or []
    tvl = history.get("tvl_monthly") or []

    metrics = {
        "revenue_3m": _window_change(rev, "revenue", 3),
        "revenue_6m": _window_change(rev, "revenue", 6),
        "revenue_12m": _window_change(rev, "revenue", 12),
        "tvl_3m": _window_change(tvl, "tvl", 3),
        "tvl_6m": _window_change(tvl, "tvl", 6),
        "tvl_12m": _window_change(tvl, "tvl", 12),
        "latest_revenue": rev[-1]["revenue"] if rev else None,
        "latest_tvl": tvl[-1]["tvl"] if tvl else None,
    }

    # How hard the locked capital works, annualised from the latest month.
    if metrics["latest_revenue"] and metrics["latest_tvl"]:
        metrics["revenue_to_tvl_pct"] = (
            metrics["latest_revenue"] * 12 / metrics["latest_tvl"] * 100
        )

    r3, t3 = metrics["revenue_3m"], metrics["tvl_3m"]
    if r3 is None and t3 is None:
        metrics.update(verdict="Not enough data", tone="gray",
                       reason="No revenue or TVL history to assess.")
        return metrics

    r_up, t_up = (r3 or 0) > 0, (t3 or 0) > 0
    if r_up and t_up:
        verdict, tone = "Strengthening", "green"
        reason = "Revenue and TVL both grew over the last 3 months."
    elif r_up and not t_up:
        verdict, tone = "Earning more from less", "green"
        reason = ("Revenue grew while TVL fell — the protocol is earning more from a "
                  "smaller capital base, which is efficiency rather than decline.")
    elif not r_up and t_up:
        verdict, tone = "Fee compression", "orange"
        reason = ("TVL grew but revenue fell — more capital is parked in the protocol "
                  "while it earns less on it, so fees or activity per dollar are shrinking.")
    else:
        verdict, tone = "Contracting", "red"
        reason = "Revenue and TVL both fell over the last 3 months."

    metrics.update(verdict=verdict, tone=tone, reason=reason)
    return metrics


def coin_protocol_slug(symbol: str) -> tuple[str, str] | None:
    """Resolve a coin ticker to its DefiLlama protocol as (slug, display name).

    Several protocols can share a ticker (v2 and v3 of the same product), so the
    largest by TVL wins. Returns None for coins that aren't fee-earning protocols.
    """
    rows = _get("https://api.llama.fi/protocols")
    if not isinstance(rows, list):
        return None
    sym = symbol.upper()
    matches = [p for p in rows if (p.get("symbol") or "").upper() == sym and (p.get("tvl") or 0) > 0]
    if not matches:
        return None
    matches.sort(key=lambda p: p.get("tvl") or 0, reverse=True)
    best = matches[0]
    return (best.get("slug") or "", best.get("name") or sym)


def watchlist_revenue_history(symbols: list[str], months: int = 12) -> dict:
    """Twelve months of protocol revenue for each watchlist coin that earns any.

    Splits the watchlist in two: coins backed by a fee-earning protocol get a monthly
    revenue series, and the rest are reported as having none. That distinction is the
    point — a coin with no protocol revenue isn't failing to grow, it simply has no
    revenue to measure.
    """
    series: dict[str, list[dict]] = {}
    meta: dict[str, dict] = {}
    without: list[str] = []

    for sym in symbols:
        try:
            resolved = coin_protocol_slug(sym)
        except Exception:  # noqa: BLE001
            resolved = None
        if not resolved or not resolved[0]:
            without.append(sym.upper())
            continue

        slug, name = resolved
        hist = protocol_history(slug, months)
        revenue = hist.get("revenue_monthly") or []
        if not revenue:
            without.append(sym.upper())
            continue

        series[sym.upper()] = revenue
        first, last = revenue[0]["revenue"], revenue[-1]["revenue"]
        meta[sym.upper()] = {
            "protocol": name,
            "latest": last,
            "growth_pct": ((last / first - 1) * 100) if first else None,
            "from": revenue[0]["month"],
            "to": revenue[-1]["month"],
        }
    return {"series": series, "meta": meta, "without_revenue": without}


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


def stock_financials(symbol: str) -> dict:
    """Quarterly income-statement series for charting: revenue, profit, and margins.

    Reads the same `stock_earnings` payload the analyst uses, so viewing the charts
    costs no additional Alpha Vantage requests beyond that cached call.
    """
    payload = stock_earnings(symbol) or {}
    data = payload.get("data") or {}
    return {
        "symbol": symbol.upper(),
        "quarters": data.get("revenue_quarters") or [],   # oldest first
        "eps_quarters": data.get("eps_quarters") or [],
        "eps_years": data.get("eps_years") or [],
        "text": payload.get("text", ""),
    }


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
