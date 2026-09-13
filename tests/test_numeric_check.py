"""
Regression tests for eval/numeric_check.py.

The checker is the harness's model-free faithfulness signal, which only helps if its
parser is right. Its first real run was not: "$500 by Friday" parsed as $500 billion and
"5 months" as five million, so a correct report was flagged while the LLM judge got it
right. These cases pin that down so it can't quietly come back.

Runs under pytest, or directly:  venv\\Scripts\\python.exe tests/test_numeric_check.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.numeric_check import _values, check_numbers  # noqa: E402


# --- the bug: a scale suffix must not steal the first letter of the next word -------------

def test_price_target_followed_by_a_b_word_is_not_billions():
    assert _values("Will SOL reach $500 by the end of this week?") == [500.0]


def test_price_target_followed_by_a_t_word_is_not_trillions():
    assert _values("hit $500 this week") == [500.0]


def test_prose_is_not_read_as_scaled_figures():
    assert _values("over the last 5 months") == []
    assert _values("revenue grew 3 times") == []
    assert _values("needs 60 more days") == []


# --- formats that must keep working --------------------------------------------------------

def test_attached_and_spelled_out_scales_normalise_to_the_same_value():
    assert _values("market cap $1.94B") == [1.94e9]
    assert _values("market cap 1.94 billion") == [1.94e9]
    assert _values("about 5m users") == [5e6]


def test_percentages_and_multiples():
    assert _values("trades at 34.4x") == [34.4]
    assert _values("revenue grew +105.9% YoY") == [105.9]


def test_matching_is_on_magnitude_so_sign_is_dropped():
    # A documented blind spot: prose carries direction in words, not signs.
    assert _values("fell -4.2%") == [4.2]


def test_currency_marks_a_small_figure_as_data():
    assert _values("a $50 target") == [50.0]


def test_prose_integers_and_years_are_skipped():
    assert _values("3 of the last 4 quarters, top 5 coins") == []
    assert _values("guidance for fiscal 2026") == []


def test_comma_formatted_numbers_count_as_data():
    assert _values("2,026 active users") == [2026.0]


# --- check_numbers -------------------------------------------------------------------------

EVIDENCE = [{
    "subquestion": "NVDA valuation",
    "finding": "market cap $5.45T, P/E 28.57, revenue grew 105.9% YoY, net margin 62.0%",
    "source": "alphavantage",
}]


def test_a_faithful_report_that_rounds_still_traces_fully():
    report = "NVDA is worth about 5.45 trillion, trades near 28.6x earnings, revenue up 105.9%."
    assert check_numbers(report, EVIDENCE)["numeric_traceability"] == 1.0


def test_invented_figures_are_flagged_with_their_context():
    report = "NVDA carries a market cap of $9.1T and net margins of 41%."
    result = check_numbers(report, EVIDENCE)
    assert result["numeric_traceability"] == 0.0
    assert result["unverified_count"] == 2
    assert "9.1T" in result["unverified"][0]["context"]


def test_the_question_counts_as_a_source():
    # Repeating the user's own figure back to them is not an invented claim.
    report = "Reaching $500 this week would need a far larger move than recent volatility."
    assert check_numbers(report, EVIDENCE)["unverified_count"] == 1
    traced = check_numbers(report, EVIDENCE, question="Will SOL reach $500 by Friday?")
    assert traced["numeric_traceability"] == 1.0


def test_analyst_view_figures_are_checked_too():
    view = {"quality": "Net margin of 62%.", "valuation": "About 29x earnings.",
            "reasoning": ["Revenue up 250% on the year."]}
    result = check_numbers("", EVIDENCE, analyst_view=view)
    assert result["claims"] == 3
    assert result["unverified_count"] == 1


def test_no_numbers_means_no_score_rather_than_a_perfect_one():
    result = check_numbers("Dogecoin has no revenue, earnings or P/E ratio.", EVIDENCE)
    assert result["numeric_traceability"] is None


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
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
