"""
TOTAL3 — total crypto market cap excluding BTC and ETH — rebuilt from per-coin data.

CoinGecko's market-cap chart endpoint is paid, so the series is reconstructed by summing
the market caps of the largest non-BTC/ETH coins. Three things make that defensible
rather than hand-wavy:

  * Constituents change. Summing today's top coins straight across a year would silently
    delete the ones that collapsed and retroactively include the ones that soared. So the
    index is CHAINED: each day's move is measured only across coins present on both that
    day and the one before, then multiplied through. A coin entering or leaving moves the
    level only by returns it actually earned.
  * It is a proxy, not the published index. Summed top-250 alts come to roughly 107% of
    CoinGecko's own TOTAL3, because its headline total isn't a plain sum of coin market
    caps. The chained series is therefore calibrated so its last point equals the live
    TOTAL3: the shape is what a model needs, and the level stays honest.
  * The free tier sustains about five requests a minute (measured: five in a row at 13s
    apart all succeeded, while 1.2s apart returned 429 on the fourth). Fetching is paced
    and cached to disk for a day, so rebuilding is a deliberate ~11-minute act rather than
    something a page refresh can trigger.

Chaining reduces survivorship bias but cannot remove it: the constituent list still comes
from today's rankings, so a coin that died before today never appears at all. Any model
trained on this series inherits that, and the history is one year — a single regime.
"""

import json
import time
from datetime import datetime, timezone

import httpx

from config import ROOT_DIR

API = "https://api.coingecko.com/api/v3"
HEADERS = {"User-Agent": "markets-research-agent"}
CACHE = ROOT_DIR / "data" / "total3_series.json"

# Measured, not guessed: 1.2s apart gets 429s, 13s apart does not.
REQUEST_PACING_SECONDS = 13.0
DEFAULT_COINS = 50          # ~96% of TOTAL3, ~11 minutes to build
CACHE_MAX_AGE_HOURS = 24

EXCLUDED = {"bitcoin", "ethereum"}

# TOTAL3 includes stablecoins by definition, so they stay in. Their share is reported
# instead, because a large stable block damps the index's volatility and means part of
# any move is stablecoin supply rather than altcoin risk appetite.
STABLE_SYMBOLS = {
    "USDT", "USDC", "DAI", "USDE", "FDUSD", "USDS", "PYUSD", "TUSD", "USDD", "USD1",
    "BUSD", "GUSD", "LUSD", "FRAX", "USDP", "RLUSD",
}


def _get(path: str, params: dict | None = None, tries: int = 3):
    """One paced GET, retrying a rate-limit refusal rather than losing the whole build."""
    for attempt in range(tries):
        response = httpx.get(f"{API}{path}", params=params, timeout=60, headers=HEADERS)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429:
            wait = float(response.headers.get("retry-after") or 60)
            time.sleep(wait * (attempt + 1))
            continue
        response.raise_for_status()
    raise RuntimeError(f"CoinGecko kept rate-limiting {path} after {tries} attempts")


def live_total3() -> dict:
    """TOTAL3 right now, straight from CoinGecko's global figures (one request)."""
    data = _get("/global")["data"]
    total = data["total_market_cap"]["usd"]
    btc = data["market_cap_percentage"].get("btc", 0.0)
    eth = data["market_cap_percentage"].get("eth", 0.0)
    return {
        "total3": total * (100 - btc - eth) / 100,
        "total": total,
        "btc_dominance": btc,
        "eth_dominance": eth,
        "asof": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def top_alts(n: int = DEFAULT_COINS) -> list[dict]:
    """The n largest coins that aren't BTC or ETH, with their current market caps."""
    rows = _get("/coins/markets", {"vs_currency": "usd", "order": "market_cap_desc",
                                   "per_page": 250, "page": 1})
    alts = [
        {"id": r["id"], "symbol": (r.get("symbol") or "").upper(),
         "market_cap": r.get("market_cap") or 0}
        for r in rows
        if r["id"] not in EXCLUDED and (r.get("market_cap") or 0) > 0
    ]
    return alts[:n]


def _daily_caps(coin_id: str) -> dict[str, float]:
    """A coin's market cap by date over the past year (the free tier's whole window)."""
    payload = _get(f"/coins/{coin_id}/market_chart",
                   {"vs_currency": "usd", "days": 365, "interval": "daily"})
    caps: dict[str, float] = {}
    for milliseconds, value in payload.get("market_caps", []):
        if not value:
            continue
        day = datetime.fromtimestamp(milliseconds / 1000, timezone.utc).strftime("%Y-%m-%d")
        caps[day] = float(value)
    return caps


def chain_index(caps_by_coin: dict[str, dict[str, float]]) -> list[tuple[str, float]]:
    """Chain daily moves across coins present on both days, starting the index at 1.0.

    This is what keeps a changing constituent list from inventing jumps: a coin appearing
    for the first time contributes nothing on its first day, only from its next move on.
    """
    days = sorted({day for caps in caps_by_coin.values() for day in caps})
    if not days:
        return []

    level = 1.0
    series = [(days[0], level)]
    for previous, current in zip(days, days[1:]):
        before = after = 0.0
        for caps in caps_by_coin.values():
            if previous in caps and current in caps:
                before += caps[previous]
                after += caps[current]
        if before > 0:
            level *= after / before
        series.append((current, level))
    return series


def build_series(n_coins: int = DEFAULT_COINS, progress=None) -> dict:
    """Rebuild the TOTAL3 proxy series. Slow and paced — roughly 11 minutes for 50 coins."""
    alts = top_alts(n_coins)
    time.sleep(REQUEST_PACING_SECONDS)

    caps_by_coin: dict[str, dict[str, float]] = {}
    failed: list[str] = []
    for index, coin in enumerate(alts, start=1):
        if progress:
            progress(index, len(alts), coin["symbol"])
        try:
            caps = _daily_caps(coin["id"])
            if caps:
                caps_by_coin[coin["id"]] = caps
            else:
                failed.append(coin["symbol"])
        except Exception:  # noqa: BLE001 - one missing coin shouldn't lose the build
            failed.append(coin["symbol"])
        if index < len(alts):
            time.sleep(REQUEST_PACING_SECONDS)

    chained = chain_index(caps_by_coin)
    if not chained:
        raise RuntimeError("No market-cap history could be fetched.")

    # Calibrate the level to the published TOTAL3, keeping the chained shape.
    live = live_total3()
    scale = live["total3"] / chained[-1][1]
    series = [{"date": day, "total3": level * scale} for day, level in chained]

    stable_cap = sum(c["market_cap"] for c in alts if c["symbol"] in STABLE_SYMBOLS)
    alt_cap = sum(c["market_cap"] for c in alts)
    lengths = [len(c) for c in caps_by_coin.values()]

    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_coins_requested": n_coins,
        "n_coins_used": len(caps_by_coin),
        "failed": failed,
        "coverage_pct": (alt_cap / live["total3"] * 100) if live["total3"] else None,
        "stable_share_pct": (stable_cap / alt_cap * 100) if alt_cap else None,
        "shortest_history_days": min(lengths) if lengths else 0,
        "longest_history_days": max(lengths) if lengths else 0,
        "live": live,
        "series": series,
    }


def load_series(n_coins: int = DEFAULT_COINS, force: bool = False, progress=None) -> dict:
    """The cached series, rebuilt only when it's missing, stale or explicitly forced."""
    if not force and CACHE.exists():
        try:
            cached = json.loads(CACHE.read_text(encoding="utf-8"))
            built = datetime.fromisoformat(cached["built_at"])
            age_hours = (datetime.now(timezone.utc) - built).total_seconds() / 3600
            if age_hours < CACHE_MAX_AGE_HOURS and cached.get("series"):
                cached["age_hours"] = round(age_hours, 1)
                return cached
        except (json.JSONDecodeError, OSError, KeyError, ValueError):
            pass

    data = build_series(n_coins, progress=progress)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    data["age_hours"] = 0.0
    return data


def read_cache() -> dict | None:
    """The cached series if there is one, never triggering an 11-minute rebuild."""
    if not CACHE.exists():
        return None
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not data.get("series"):
        return None
    try:
        built = datetime.fromisoformat(data["built_at"])
        data["age_hours"] = round(
            (datetime.now(timezone.utc) - built).total_seconds() / 3600, 1)
    except (KeyError, ValueError):
        data["age_hours"] = None
    return data


VOL_WINDOW_DAYS = 90


def volatility_cone(levels, horizon_days: int = 30, sigmas=(1, 2),
                    window: int = VOL_WINDOW_DAYS) -> dict:
    """Where TOTAL3 sits at 1σ and 2σ if it keeps behaving as it recently has.

    This is not a forecast, and the centre line is deliberately flat. eval/total3_forecast
    measured a ridge model and momentum persistence against the random walk at 7 and 30
    days, and the random walk — predicting no change — beat both. So the honest centre is
    today's level, and the cone is only the dispersion its own recent volatility implies
    around it. Width grows with the square root of time, which assumes returns are
    independent day to day; real markets cluster their volatility, so a calm stretch
    understates the tails and a violent one overstates them.
    """
    import math

    closes = [float(x) for x in levels if x and x > 0]
    if len(closes) < 3:
        return {"bands": [], "daily_vol": None}
    window = min(window, len(closes) - 1)

    recent = closes[-(window + 1):]
    returns = [math.log(recent[i] / recent[i - 1]) for i in range(1, len(recent))]
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / max(1, len(returns) - 1)
    daily = math.sqrt(variance)

    last = closes[-1]
    bands = []
    for day in range(horizon_days + 1):
        spread = daily * math.sqrt(day)
        row = {"day": day, "mid": last}
        for k in sigmas:
            row[f"lo{k}"] = last * math.exp(-k * spread)
            row[f"hi{k}"] = last * math.exp(k * spread)
        bands.append(row)

    return {
        "daily_vol": daily,
        "annualised_vol": daily * math.sqrt(365),
        "window_days": window,
        "last_level": last,
        "horizon_days": horizon_days,
        "bands": bands,
    }
