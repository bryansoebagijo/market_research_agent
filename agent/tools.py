"""
The live-data tools the researcher calls — the agent's connection to the real world.

Each tool fetches current data from a free source and returns a small dict:
    {"text": "<concise factual summary>", "source": "<where it came from>"}

Results are cached per-process (lru_cache) so asking about the same ticker in several
sub-questions only hits the API once per run — important for rate-limited free tiers.
"""

import time
from functools import lru_cache

import httpx

from config import settings

# Common crypto tickers, so we can tell "BTC" (crypto) from "NVDA" (stock) without an
# API call. Anything not here is treated as a stock.
COMMON_CRYPTO = {
    "BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", "DOT", "MATIC", "AVAX",
    "LINK", "LTC", "TRX", "SHIB", "ATOM", "UNI", "XLM", "BCH", "NEAR", "APT",
    "ARB", "OP", "TON", "SUI", "PEPE", "ICP", "FIL", "HBAR", "INJ", "RNDR",
}


def classify_ticker(symbol: str) -> str:
    """'crypto' or 'stock' — decides which market tool to use for a ticker."""
    return "crypto" if symbol.upper() in COMMON_CRYPTO else "stock"


def _get(url: str, params: dict | None = None, timeout: float = 20.0) -> dict | list:
    """Small JSON GET helper with a User-Agent (some APIs reject blank ones)."""
    r = httpx.get(url, params=params, timeout=timeout, headers={"User-Agent": "markets-research-agent"})
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=64)
def crypto_snapshot(symbol: str) -> dict:
    """Current price + 24h/7d/30d change for a crypto, from CoinGecko (keyless)."""
    src = "CoinGecko"
    try:
        found = _get("https://api.coingecko.com/api/v3/search", {"query": symbol})
        coins = found.get("coins", []) if isinstance(found, dict) else []
        match = next((c for c in coins if c["symbol"].upper() == symbol.upper()), coins[0] if coins else None)
        if not match:
            return {"text": f"No CoinGecko match for {symbol}.", "source": src}
        time.sleep(1.2)  # respect CoinGecko's aggressive free-tier rate limit
        markets = _get(
            "https://api.coingecko.com/api/v3/coins/markets",
            {"vs_currency": "usd", "ids": match["id"], "price_change_percentage": "24h,7d,30d"},
        )
        if not markets:
            return {"text": f"No market data for {symbol}.", "source": src}
        d = markets[0]
        text = (
            f"{symbol.upper()} ({d.get('name')}): ${d.get('current_price'):,} | "
            f"24h {d.get('price_change_percentage_24h_in_currency') or 0:+.1f}% | "
            f"7d {d.get('price_change_percentage_7d_in_currency') or 0:+.1f}% | "
            f"30d {d.get('price_change_percentage_30d_in_currency') or 0:+.1f}%"
        )
        return {"text": text, "source": f"{src} (live)"}
    except Exception as exc:  # noqa: BLE001 - degrade to a text finding, never crash the node
        return {"text": f"Crypto data unavailable for {symbol} ({exc}).", "source": src}


@lru_cache(maxsize=64)
def stock_snapshot(symbol: str) -> dict:
    """Current price + daily change + 52-week range for a stock, from Twelve Data."""
    src = "Twelve Data"
    if not settings.twelve_data_api_key:
        return {"text": f"{symbol}: stock price unavailable (no TWELVE_DATA_API_KEY set).", "source": src}
    try:
        d = _get(
            "https://api.twelvedata.com/quote",
            {"symbol": symbol, "apikey": settings.twelve_data_api_key},
        )
        if not isinstance(d, dict) or d.get("status") == "error":
            return {"text": f"{symbol}: {d.get('message', 'no data') if isinstance(d, dict) else 'no data'}", "source": src}
        wk = d.get("fifty_two_week", {}) or {}
        text = (
            f"{symbol.upper()} ({d.get('name')}): ${d.get('close')} | "
            f"today {d.get('percent_change')}% | "
            f"52w {wk.get('low')}–{wk.get('high')}"
        )
        time.sleep(1.5)  # Twelve Data free tier ~8 req/min
        return {"text": text, "source": f"{src} (live)"}
    except Exception as exc:  # noqa: BLE001
        return {"text": f"Stock data unavailable for {symbol} ({exc}).", "source": src}


@lru_cache(maxsize=64)
def stock_fundamentals(symbol: str) -> dict:
    """Key fundamentals (P/E, market cap, margin, sector) from Alpha Vantage."""
    src = "Alpha Vantage"
    if not settings.alphavantage_api_key:
        return {"text": f"{symbol}: fundamentals unavailable (no ALPHAVANTAGE_API_KEY set).", "source": src}
    try:
        d = _get(
            "https://www.alphavantage.co/query",
            {"function": "OVERVIEW", "symbol": symbol, "apikey": settings.alphavantage_api_key},
        )
        if not isinstance(d, dict) or "Symbol" not in d:
            return {"text": f"{symbol}: fundamentals unavailable (rate limited or no data).", "source": src}
        text = (
            f"{symbol.upper()} fundamentals — P/E {d.get('PERatio', 'n/a')}, "
            f"market cap {d.get('MarketCapitalization', 'n/a')}, "
            f"profit margin {d.get('ProfitMargin', 'n/a')}, sector {d.get('Sector', 'n/a')}"
        )
        return {"text": text, "source": src}
    except Exception as exc:  # noqa: BLE001
        return {"text": f"Fundamentals unavailable for {symbol} ({exc}).", "source": src}


def web_search(query: str, k: int = 5) -> dict:
    """Recent web results (news, macro context) via keyless DuckDuckGo search."""
    src = "DuckDuckGo web search"
    try:
        from ddgs import DDGS  # imported here so a missing dep doesn't break the module

        with DDGS() as ddg:
            results = list(ddg.text(query, max_results=k))
        if not results:
            return {"text": f"No web results for: {query}", "source": src}
        lines = [f"- {r.get('title', '')}: {(r.get('body') or '')[:180]}" for r in results]
        return {"text": "\n".join(lines), "source": src}
    except Exception as exc:  # noqa: BLE001
        return {"text": f"Web search unavailable ({exc}).", "source": src}
