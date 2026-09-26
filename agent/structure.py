"""
Market structure: is the trend intact, or has it just changed character?

Price action traders read a trend through its swings. An uptrend makes higher highs and
higher lows and is intact while that holds. Two events matter:

  * BOS — break of structure. Price takes out the last swing high in an uptrend (or the
    last swing low in a downtrend). The trend continued.
  * CHoCH — change of character. Price breaks the last higher-LOW in an uptrend, or the
    last lower-HIGH in a downtrend. The pattern that defined the trend has failed, which
    is the earliest structural warning that it may be turning.

Swings come from a ZIGZAG rather than from bar-by-bar pivots, and that choice is what
makes the read usable. Detecting highs and lows independently produces two problems seen
immediately on real data: 51 "swings" in a year of daily closes, nearly all of them
noise, and a most-recent low sitting *above* the most-recent high because the two were
months apart and never interleaved. A zigzag fixes both — it alternates high, low, high
by construction, and ignores any move smaller than a percentage threshold, so a swing
means a real leg rather than a wiggle.

Everything here is deterministic: the same prices always give the same read, no model
involved.

Two honest limits:

  * CoinGecko's free tier returns daily CLOSES, not intraday highs and lows. Swings are
    therefore close-based: a wick that pierced a level and closed back inside does not
    register. That makes this read more conservative than a candle chart, since it needs
    a close beyond a level rather than a touch.
  * Structure describes what price has already done. It says a trend is intact or broken;
    it does not say what happens next. The final leg is also still unconfirmed — price
    has not yet reversed enough to fix that pivot in place.
"""

# Crypto legs are large, so a swing has to move this far to count as one rather than as
# noise. Weekly is looser because a weekly bar already absorbs a week of movement.
DAILY_SWING_PCT = 5.0
WEEKLY_SWING_PCT = 8.0
DAYS_PER_WEEK = 7


def to_weekly(series: list[dict], days: int = DAYS_PER_WEEK) -> list[dict]:
    """Group daily points into weekly bars, anchored so the last bar ends on the newest day.

    Each bar carries the week's last close plus the highest and lowest close within it.
    Those aren't true highs and lows — only closes are available — but across a week they
    approximate the range closely enough for swing detection.
    """
    if not series:
        return []
    bars = []
    for end in range(len(series), 0, -days):
        chunk = series[max(0, end - days):end]
        if not chunk:
            continue
        closes = [p["close"] for p in chunk]
        bars.append({
            "date": chunk[-1]["date"],
            "close": closes[-1],
            "high": max(closes),
            "low": min(closes),
            "bars": len(chunk),
        })
    return list(reversed(bars))


def zigzag(values: list[float], threshold_pct: float) -> list[dict]:
    """Alternating swing highs and lows, ignoring any reversal smaller than the threshold.

    Walks forward holding the running extreme. A move of threshold_pct against that
    extreme confirms it as a pivot and flips the direction, which guarantees the output
    alternates high, low, high — so "the last high" and "the last low" are always parts
    of the same recent sequence and can be compared meaningfully.
    """
    if len(values) < 3 or threshold_pct <= 0:
        return []

    pivots: list[dict] = []
    extreme_index, extreme_value = 0, values[0]
    direction = 0  # 0 not yet established, 1 rising, -1 falling

    for i in range(1, len(values)):
        value = values[i]
        if not extreme_value:
            continue
        change_pct = (value - extreme_value) / extreme_value * 100

        if direction >= 0 and change_pct <= -threshold_pct:
            pivots.append({"index": extreme_index, "value": extreme_value, "kind": "high"})
            direction, extreme_index, extreme_value = -1, i, value
        elif direction <= 0 and change_pct >= threshold_pct:
            pivots.append({"index": extreme_index, "value": extreme_value, "kind": "low"})
            direction, extreme_index, extreme_value = 1, i, value
        elif direction >= 0 and value > extreme_value:
            extreme_index, extreme_value = i, value
        elif direction <= 0 and value < extreme_value:
            extreme_index, extreme_value = i, value

    return pivots


def _trend_from(highs: list[dict], lows: list[dict]) -> str:
    """Bullish needs a higher high AND a higher low; bearish the mirror; else ranging."""
    if len(highs) < 2 or len(lows) < 2:
        return "unclear"
    higher_high = highs[-1]["value"] > highs[-2]["value"]
    higher_low = lows[-1]["value"] > lows[-2]["value"]
    if higher_high and higher_low:
        return "bullish"
    if not higher_high and not higher_low:
        return "bearish"
    return "ranging"


def analyse(series: list[dict], threshold_pct: float) -> dict:
    """Read one timeframe's structure: its trend, the last event, and the levels involved.

    `series` is {date, close} points, oldest first.
    """
    if len(series) < 5:
        return {"available": False, "reason": "not enough history"}

    closes = [p["close"] for p in series]
    pivots = zigzag(closes, threshold_pct)
    highs = [p for p in pivots if p["kind"] == "high"]
    lows = [p for p in pivots if p["kind"] == "low"]

    if len(highs) < 2 or len(lows) < 2:
        return {"available": False,
                "reason": f"only {len(highs)} swing high(s) and {len(lows)} low(s) at a "
                          f"{threshold_pct:g}% threshold — no structure to read yet"}

    trend = _trend_from(highs, lows)
    last_high, last_low = highs[-1], lows[-1]
    price = closes[-1]

    # The event is judged on the latest close against the levels the trend was built on.
    event = event_note = None
    if trend == "bullish":
        if price < last_low["value"]:
            event, event_note = "CHoCH", (
                f"closed below the last higher-low ({last_low['value']:,.4g} on "
                f"{series[last_low['index']]['date']}) — the uptrend's pattern has failed")
        elif price > last_high["value"]:
            event, event_note = "BOS", (
                f"closed above the last swing high ({last_high['value']:,.4g} on "
                f"{series[last_high['index']]['date']}) — the uptrend continued")
    elif trend == "bearish":
        if price > last_high["value"]:
            event, event_note = "CHoCH", (
                f"closed above the last lower-high ({last_high['value']:,.4g} on "
                f"{series[last_high['index']]['date']}) — the downtrend's pattern has failed")
        elif price < last_low["value"]:
            event, event_note = "BOS", (
                f"closed below the last swing low ({last_low['value']:,.4g} on "
                f"{series[last_low['index']]['date']}) — the downtrend continued")
    else:
        if price > last_high["value"]:
            event, event_note = "breakout", f"closed above the range high ({last_high['value']:,.4g})"
        elif price < last_low["value"]:
            event, event_note = "breakdown", f"closed below the range low ({last_low['value']:,.4g})"

    # The leg since the last confirmed pivot has no pivot of its own yet: price hasn't
    # reversed far enough to set one. Reporting it matters, because a zigzag's last
    # confirmed swings can be months old while price has run a long way past them.
    last_pivot = pivots[-1]
    leg_pct = ((price - last_pivot["value"]) / last_pivot["value"] * 100
               if last_pivot["value"] else 0.0)
    current_leg = {
        "direction": "up" if leg_pct >= 0 else "down",
        "change_pct": leg_pct,
        "from_value": last_pivot["value"],
        "from_date": series[last_pivot["index"]]["date"],
        "bars": len(series) - 1 - last_pivot["index"],
    }

    return {
        "available": True,
        "trend": trend,
        "event": event,
        "event_note": event_note,
        "current_leg": current_leg,
        "price": price,
        "last_swing_high": last_high["value"],
        "last_swing_high_date": series[last_high["index"]]["date"],
        "last_swing_low": last_low["value"],
        "last_swing_low_date": series[last_low["index"]]["date"],
        "swings": len(pivots),
        "threshold_pct": threshold_pct,
        "bars": len(series),
    }


def _describe(timeframe: str, read: dict) -> str:
    if not read.get("available"):
        return f"{timeframe}: unreadable"
    return f"{timeframe}: {read['trend']}" + (f" ({read['event']})" if read.get("event") else "")


def read_both(daily: list[dict], daily_pct: float = DAILY_SWING_PCT,
              weekly_pct: float = WEEKLY_SWING_PCT) -> dict:
    """Structure on the weekly and the daily, and what the two together imply.

    The weekly is the senior timeframe: it sets the context the daily move happens inside.
    A daily uptrend against a broken weekly is a bounce within a deteriorating trend, and
    calling that simply "bullish" is where timeframe confusion starts.
    """
    weekly = analyse(to_weekly(daily), weekly_pct)
    day = analyse(daily, daily_pct)

    if not (weekly.get("available") and day.get("available")):
        alignment = "unknown"
        summary = "Not enough history to read structure on both timeframes."
    elif weekly["trend"] == day["trend"]:
        alignment = "aligned"
        summary = f"Weekly and daily agree — both {weekly['trend']}."
    else:
        alignment = "conflicted"
        summary = (f"Weekly is {weekly['trend']} while the daily is {day['trend']}. The "
                   "weekly is the senior timeframe, so the daily move sits inside it.")

    if weekly.get("event") == "CHoCH":
        summary += " The weekly changed character, which is the more serious of the two."
    elif day.get("event") == "CHoCH":
        summary += " The daily changed character — an early warning, not yet a weekly one."

    return {
        "weekly": weekly,
        "daily": day,
        "alignment": alignment,
        "summary": summary,
        "headline": f"{_describe('Weekly', weekly)} | {_describe('Daily', day)}",
    }
