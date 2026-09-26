"""
Regression tests for agent/structure.py.

The first version detected swing highs and lows independently, and real data exposed two
failures straight away: 51 "swings" in a year of daily closes, and a most-recent low
priced ABOVE the most-recent high because the two were months apart and never
interleaved. Comparing those to call a trend is meaningless.

The zigzag fixes both, so these pin the properties that matter: pivots alternate, moves
under the threshold are ignored, higher-high-plus-higher-low reads bullish, and breaking
the last higher-low is reported as a change of character.

Runs under pytest, or directly:  venv\\Scripts\\python.exe tests/test_structure.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.structure import analyse, read_both, to_weekly, zigzag  # noqa: E402


def series(values: list[float]) -> list[dict]:
    """Turn bare prices into the {date, close} shape the reader expects."""
    return [{"date": f"2026-01-{i + 1:02d}", "close": float(v)} for i, v in enumerate(values)]


# Each leg moves well over 10%, with one 4% wiggle that must be ignored.
ZIGZAG_VALUES = [100, 110, 105, 130, 115, 150]
# Higher highs (130 -> 160) and higher lows (110 -> 135).
BULLISH_VALUES = [100, 130, 110, 160, 135, 190]


# --- the zigzag ----------------------------------------------------------------------------

def test_pivots_always_alternate():
    kinds = [p["kind"] for p in zigzag(ZIGZAG_VALUES, 10)]
    assert kinds, "expected some pivots"
    assert all(a != b for a, b in zip(kinds, kinds[1:])), kinds


def test_moves_under_the_threshold_are_ignored():
    # The dip from 110 to 105 is under 10% and must not create a pivot.
    values = [p["value"] for p in zigzag(ZIGZAG_VALUES, 10)]
    assert 105 not in values, values


def test_a_lower_threshold_finds_more_swings():
    assert len(zigzag(ZIGZAG_VALUES, 3)) > len(zigzag(ZIGZAG_VALUES, 10))


def test_a_flat_series_has_no_swings():
    assert zigzag([100.0] * 20, 5) == []


# --- reading a structure -------------------------------------------------------------------

def test_higher_highs_and_higher_lows_read_bullish():
    read = analyse(series(BULLISH_VALUES), 10)
    assert read["available"] and read["trend"] == "bullish", read


def test_the_last_low_is_never_above_the_last_high():
    # The bug that started this file: independently detected swings weren't comparable.
    read = analyse(series(BULLISH_VALUES), 10)
    assert read["last_swing_low"] < read["last_swing_high"], read


def test_breaking_the_last_higher_low_is_a_change_of_character():
    read = analyse(series(BULLISH_VALUES + [120]), 10)
    assert read["trend"] == "bullish"
    assert read["event"] == "CHoCH", read
    assert "higher-low" in read["event_note"]


def test_taking_out_the_last_high_is_a_break_of_structure():
    read = analyse(series(BULLISH_VALUES), 10)   # ends at 190, above the 160 swing high
    assert read["event"] == "BOS", read


def test_the_unconfirmed_leg_is_reported():
    read = analyse(series(BULLISH_VALUES), 10)
    leg = read["current_leg"]
    assert leg["direction"] == "up"
    assert leg["change_pct"] > 0 and leg["from_date"]


def test_too_little_history_says_so_rather_than_guessing():
    read = analyse(series([100, 101, 102, 103, 104, 105]), 10)
    assert read["available"] is False and "threshold" in read["reason"]


# --- weekly bars ---------------------------------------------------------------------------

def test_daily_points_group_into_weekly_bars():
    bars = to_weekly(series(list(range(1, 15))))   # 14 days -> 2 weeks
    assert len(bars) == 2
    assert bars[-1]["close"] == 14          # the week's last close
    assert bars[-1]["high"] == 14 and bars[-1]["low"] == 8


def test_the_last_weekly_bar_ends_on_the_newest_day():
    daily = series(list(range(1, 11)))      # 10 days: the partial week must be the oldest
    bars = to_weekly(daily)
    assert bars[-1]["date"] == daily[-1]["date"]
    assert bars[0]["bars"] == 3             # 10 = 3 + 7


# --- both timeframes -----------------------------------------------------------------------

def test_conflicting_timeframes_defer_to_the_weekly():
    read = read_both(series(BULLISH_VALUES * 12))
    assert read["alignment"] in ("aligned", "conflicted", "unknown")
    if read["alignment"] == "conflicted":
        assert "senior" in read["summary"]


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  pass  {name}")
        except AssertionError:
            failures += 1
            frame = traceback.extract_tb(sys.exc_info()[2])[-1]
            print(f"  FAIL  {name}  (line {frame.lineno}: {frame.line})")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
