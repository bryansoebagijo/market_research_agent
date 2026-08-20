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


# --- Technical indicators (same approach as the Telegram bot: Wilder's RSI, EMA) ---

def compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """RSI with Wilder's smoothing. Needs > period closes."""
    if len(closes) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):  # Wilder's smoothing over the rest
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def compute_ema(values: list[float], period: int) -> float | None:
    """Exponential moving average, seeded with an SMA of the first `period` values."""
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def _trend_read(closes: list[float]) -> str:
    """Turn a chronological close series into a plain-language technical read."""
    price = closes[-1]
    ema13, ema21, rsi = compute_ema(closes, 13), compute_ema(closes, 21), compute_rsi(closes, 14)
    if ema13 is None or ema21 is None or rsi is None:
        return "insufficient price history to compute indicators."
    if ema13 > ema21 and price > ema13 and price > ema21:
        momentum = "bullish (EMA13 > EMA21, price above both)"
    elif ema13 < ema21 and price < ema13 and price < ema21:
        momentum = "bearish (EMA13 < EMA21, price below both)"
    else:
        momentum = "mixed / no clear trend"
    rsi_note = "overbought" if rsi >= 70 else "oversold" if rsi <= 30 else "neutral"
    return (
        f"RSI(14) {rsi:.0f} ({rsi_note}); EMA13 {ema13:,.2f} vs EMA21 {ema21:,.2f}; "
        f"momentum {momentum}"
    )


@lru_cache(maxsize=64)
def _coingecko_id(symbol: str) -> str | None:
    """Resolve a ticker like 'BTC' to a CoinGecko coin id (cached)."""
    found = _get("https://api.coingecko.com/api/v3/search", {"query": symbol})
    coins = found.get("coins", []) if isinstance(found, dict) else []
    match = next((c for c in coins if c["symbol"].upper() == symbol.upper()), coins[0] if coins else None)
    return match["id"] if match else None


@lru_cache(maxsize=64)
def crypto_snapshot(symbol: str) -> dict:
    """Current price + 24h/7d/30d change for a crypto, from CoinGecko (keyless)."""
    src = "CoinGecko"
    try:
        coin_id = _coingecko_id(symbol)
        if not coin_id:
            return {"text": f"No CoinGecko match for {symbol}.", "source": src}
        time.sleep(1.2)  # respect CoinGecko's aggressive free-tier rate limit
        markets = _get(
            "https://api.coingecko.com/api/v3/coins/markets",
            {"vs_currency": "usd", "ids": coin_id, "price_change_percentage": "24h,7d,30d"},
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
def crypto_technicals(symbol: str) -> dict:
    """RSI(14) + EMA13/EMA21 momentum for a crypto, computed from 95 daily closes."""
    src = "CoinGecko"
    try:
        coin_id = _coingecko_id(symbol)
        if not coin_id:
            return {"text": f"No CoinGecko match for {symbol}.", "source": src}
        time.sleep(1.2)
        chart = _get(
            f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
            {"vs_currency": "usd", "days": 95, "interval": "daily"},
        )
        closes = [p[1] for p in chart.get("prices", [])] if isinstance(chart, dict) else []
        if len(closes) < 22:
            return {"text": f"{symbol}: not enough history for indicators.", "source": src}
        return {"text": f"{symbol.upper()} technicals — {_trend_read(closes)}", "source": f"{src} (computed)"}
    except Exception as exc:  # noqa: BLE001
        return {"text": f"Crypto technicals unavailable for {symbol} ({exc}).", "source": src}


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
def stock_technicals(symbol: str) -> dict:
    """RSI(14) + EMA13/EMA21 momentum for a stock, computed from 60 daily closes."""
    src = "Twelve Data"
    if not settings.twelve_data_api_key:
        return {"text": f"{symbol}: technicals unavailable (no TWELVE_DATA_API_KEY set).", "source": src}
    try:
        d = _get(
            "https://api.twelvedata.com/time_series",
            {"symbol": symbol, "interval": "1day", "outputsize": 60, "order": "ASC",
             "apikey": settings.twelve_data_api_key},
        )
        if not isinstance(d, dict) or d.get("status") == "error":
            return {"text": f"{symbol}: {d.get('message', 'no data') if isinstance(d, dict) else 'no data'}", "source": src}
        closes = [float(v["close"]) for v in d.get("values", [])]  # ASC = oldest first
        if len(closes) < 22:
            return {"text": f"{symbol}: not enough history for indicators.", "source": src}
        time.sleep(1.5)
        return {"text": f"{symbol.upper()} technicals — {_trend_read(closes)}", "source": f"{src} (computed)"}
    except Exception as exc:  # noqa: BLE001
        return {"text": f"Stock technicals unavailable for {symbol} ({exc}).", "source": src}


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


# --- News headlines from RSS feeds (dated, from real outlets) ---

CRYPTO_FEEDS = {
    "CoinDesk": "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "Cointelegraph": "https://cointelegraph.com/rss",
    "Decrypt": "https://decrypt.co/feed",
}

STOCK_FEEDS = {
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
}

# Ticker -> extra words to match in a headline (a headline rarely says "NVDA").
TICKER_ALIASES = {
    "BTC": ["bitcoin"], "ETH": ["ethereum", "ether"], "SOL": ["solana"],
    "XRP": ["ripple"], "DOGE": ["dogecoin"], "ADA": ["cardano"],
    "NVDA": ["nvidia"], "AAPL": ["apple"], "MSFT": ["microsoft"],
    "AMZN": ["amazon"], "GOOGL": ["google", "alphabet"], "META": ["meta", "facebook"],
    "TSLA": ["tesla"],
}


def _entry_date(entry) -> str:
    """Best-effort published date for a feed entry (empty string if absent)."""
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed:
        return time.strftime("%Y-%m-%d", parsed)
    return ""


@lru_cache(maxsize=32)
def fetch_news(kind: str, tickers_key: str = "", per_feed: int = 6) -> dict:
    """Recent headlines from crypto or stock RSS feeds.

    `kind` is 'crypto' or 'stock'. `tickers_key` is a comma-joined ticker string
    (cacheable); when given, headlines mentioning those tickers/names are preferred,
    with general market headlines used as filler so there's always context.
    """
    import feedparser  # imported lazily so a missing dep can't break the module

    feeds = CRYPTO_FEEDS if kind == "crypto" else STOCK_FEEDS
    tickers = [t.strip().upper() for t in tickers_key.split(",") if t.strip()]
    # Words that make a headline "relevant" to the tickers we care about.
    wanted = {t.lower() for t in tickers}
    for t in tickers:
        wanted.update(TICKER_ALIASES.get(t, []))

    matched: list[str] = []
    general: list[str] = []
    used_sources: list[str] = []

    # For stocks, Yahoo publishes a per-ticker feed — the reliable way to get news
    # about a specific company, since general top-stories feeds often won't mention it.
    feeds = dict(feeds)
    if kind == "stock":
        for t in tickers:
            feeds[f"Yahoo {t}"] = (
                f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={t}&region=US&lang=en-US"
            )

    for name, url in feeds.items():
        try:
            parsed = feedparser.parse(url)
            entries = parsed.entries[:per_feed]
            if entries:
                used_sources.append(name)
            for e in entries:
                title = (getattr(e, "title", "") or "").strip()
                if not title:
                    continue
                date = _entry_date(e)
                line = f"- ({date or 'undated'}, {name}) {title}"
                # Entries from a per-ticker feed are about that ticker by definition,
                # even when the headline text doesn't spell the name out.
                ticker_feed = name.startswith("Yahoo ") and name != "Yahoo Finance"
                if ticker_feed or (wanted and any(w in title.lower() for w in wanted)):
                    matched.append(line)
                else:
                    general.append(line)
        except Exception:  # noqa: BLE001 - one bad feed shouldn't sink the rest
            continue

    if not matched and not general:
        return {"text": "No headlines could be fetched from the news feeds.", "source": "RSS news feeds"}

    lines = matched[:8]
    if len(lines) < 6:  # top up with general market headlines for context
        lines += general[: 6 - len(lines)]

    header = (
        f"Headlines mentioning {', '.join(tickers)} (plus general market headlines):"
        if matched else "Recent market headlines (no ticker-specific matches found):"
    )
    src = f"RSS: {', '.join(used_sources)}" if used_sources else "RSS news feeds"
    return {"text": header + "\n" + "\n".join(lines), "source": src}


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
