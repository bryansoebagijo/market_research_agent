"""
A structured read on whether a coin looks worth owning — from the data, not a model.

The dashboard already holds the pieces: market structure on two timeframes, momentum,
protocol revenue and TVL, and a valuation multiple against peers. What it lacked was a
single place that weighs them together and says what they add up to.

Deliberately rule-based. Every factor below scores -1, 0 or +1 by a stated test, the
weights are fixed, and the total maps to a band. So the same data always produces the same
read, the reasoning is visible rather than asserted, and it costs nothing to run — which
is what lets it sit on the page instead of behind a button.

Two weighting decisions carry most of the meaning:

  * THE WEEKLY OUTRANKS THE DAILY, 2 to 1. A daily uptrend inside a broken weekly is a
    bounce in a deteriorating trend, and treating both as equal is how timeframe
    confusion starts.
  * A CHoCH FLIPS THE SIGN OF ITS TREND. A change of character in a downtrend means the
    pattern making lower highs has failed — a potential turn up. In an uptrend it means
    the opposite. Reading the trend label without the event would get both backwards.

This is an assessment of evidence, not advice, and it forecasts nothing. Structure
describes what price has already done; revenue and TVL describe what a protocol has
already earned.
"""

from agent.market_data import (
    coin_price_history,
    coin_protocol_slug,
    current_metrics,
    token_multiples,
)
from agent.structure import read_both

WEEKLY_WEIGHT = 2
DAILY_WEIGHT = 1

# Above this the coin is overbought enough that entry timing is poor even when the
# structure is sound; below it, the opposite.
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

BANDS = [
    (4, "Constructive", "green"),
    (2, "Leaning constructive", "green"),
    (-1, "Mixed", "orange"),
    (-3, "Cautious", "red"),
    (-99, "Unfavourable", "red"),
]


def _structure_signal(read: dict) -> tuple[int, str]:
    """Turn one timeframe's trend and last event into a single -1/0/+1 signal."""
    if not read.get("available"):
        return 0, "structure unreadable"

    trend, event = read["trend"], read.get("event")
    score = {"bullish": 1, "bearish": -1}.get(trend, 0)

    if event == "CHoCH":
        # The pattern that defined the trend has failed, so the signal inverts.
        score = -score
        note = (f"{trend} but changed character — the pattern has failed"
                if score else f"{trend}, changed character")
    elif event == "BOS":
        note = f"{trend}, and it broke structure in that direction"
    elif event == "breakout":
        score, note = 1, "ranging, and broke out above the range"
    elif event == "breakdown":
        score, note = -1, "ranging, and broke down below the range"
    else:
        note = trend

    leg = read.get("current_leg") or {}
    if abs(leg.get("change_pct", 0)) >= 15:
        note += (f" — note the current leg is unconfirmed, {leg['change_pct']:+.0f}% "
                 f"since {leg['from_date']}")
    return score, note


def assess_coin(symbol: str) -> dict:
    """Weigh structure, momentum, protocol earnings and valuation into one read."""
    symbol = symbol.upper()
    factors: list[dict] = []

    def add(name: str, score: int, weight: int, detail: str) -> None:
        factors.append({"name": name, "score": score, "weight": weight,
                        "points": score * weight, "detail": detail})

    # Each source is fetched separately and allowed to fail on its own. CoinGecko's free
    # tier allows about five requests a minute and one assessment makes several, so a 429
    # is routine — it should cost the factor it feeds, not the whole read.
    def _safely(fetch, default):
        try:
            return fetch()
        except Exception:  # noqa: BLE001
            return default

    unavailable: list[str] = []

    history = _safely(lambda: coin_price_history(symbol, 365), [])
    structure = read_both(history) if history else None

    if structure:
        weekly_score, weekly_note = _structure_signal(structure["weekly"])
        daily_score, daily_note = _structure_signal(structure["daily"])
        add("Weekly structure", weekly_score, WEEKLY_WEIGHT, weekly_note)
        add("Daily structure", daily_score, DAILY_WEIGHT, daily_note)
    else:
        unavailable.append("price history")

    metrics = _safely(lambda: current_metrics(symbol), {})
    if not metrics:
        unavailable.append("momentum")

    rsi = metrics.get("rsi")
    if rsi is not None:
        if rsi >= RSI_OVERBOUGHT:
            add("Momentum (RSI)", -1, 1, f"RSI {rsi:.0f} — overbought, a poor entry even "
                                         "if the trend is sound")
        elif rsi <= RSI_OVERSOLD:
            add("Momentum (RSI)", 1, 1, f"RSI {rsi:.0f} — oversold")
        else:
            add("Momentum (RSI)", 0, 1, f"RSI {rsi:.0f} — neutral")

    # Protocol earnings, for the tokens that have any. The relationship between the two
    # series carries more than either alone.
    revenue, tvl = metrics.get("revenue_growth_3m"), metrics.get("tvl_growth_3m")
    if revenue is not None and tvl is not None:
        if revenue > 0 and tvl > 0:
            add("Protocol earnings", 1, 1,
                f"revenue {revenue:+.0f}% and TVL {tvl:+.0f}% over 3 months — both growing")
        elif revenue > 0:
            add("Protocol earnings", 1, 1,
                f"revenue {revenue:+.0f}% on TVL {tvl:+.0f}% — earning more from less")
        elif tvl > 0:
            add("Protocol earnings", -1, 1,
                f"TVL {tvl:+.0f}% but revenue {revenue:+.0f}% — fee compression")
        else:
            add("Protocol earnings", -1, 1,
                f"revenue {revenue:+.0f}% and TVL {tvl:+.0f}% — both shrinking")
    elif _safely(lambda: coin_protocol_slug(symbol), None) is None:
        factors.append({"name": "Protocol earnings", "score": None, "weight": 0,
                        "points": 0,
                        "detail": "no protocol revenue to measure — a monetary coin isn't "
                                  "failing this test, it simply doesn't take it"})

    multiples = _safely(lambda: token_multiples(symbol), {})
    if multiples.get("available") and not multiples.get("implausible"):
        cheaper = multiples.get("cheaper_than")
        ps, median = multiples["ps"], multiples["peer_median_ps"]
        if cheaper is not None and cheaper >= 60:
            add("Valuation", 1, 1, f"{ps:.0f}x revenue vs a {median:.0f}x peer median — "
                                   f"cheaper than {cheaper:.0f}% of peers")
        elif cheaper is not None and cheaper <= 25:
            add("Valuation", -1, 1, f"{ps:.0f}x revenue vs a {median:.0f}x peer median — "
                                    f"dearer than {100 - cheaper:.0f}% of peers")
        else:
            add("Valuation", 0, 1, f"{ps:.0f}x revenue, around the {median:.0f}x peer median")

    total = sum(f["points"] for f in factors)
    possible = sum(f["weight"] for f in factors if f["score"] is not None)
    verdict, tone = next((label, colour) for cutoff, label, colour in BANDS if total >= cutoff)

    return {
        "symbol": symbol,
        "factors": factors,
        "score": total,
        "possible": possible,
        "verdict": verdict,
        "tone": tone,
        "structure": structure,
        "unavailable": unavailable,
        "summary": structure["summary"] if structure else "No price history available.",
    }
