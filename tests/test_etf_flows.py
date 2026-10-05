"""
Regression tests for the US spot ETF flow reader in agent/market_data.py.

The source has three properties that are easy to get wrong and that the panel depends on:

  * Rows are TRADING sessions, not calendar days. Summing "the last 5" must mean five
    sessions; treating the series as daily would silently turn a week into nine days.
  * SoSoValue reports failures in the response BODY with HTTP 200, so a status check alone
    would let an error render as an empty chart instead of a message.
  * Numbers arrive as long decimal STRINGS inside a {value, lastUpdateDate, status}
    wrapper on the metrics endpoint, but as plain floats on the history endpoint.

These pin all three, plus the rule that the headline figure and the chart are read from
the same series — the two endpoints carry slightly different cumulative totals, and taking
one from each would put a number beside a chart that disagrees with it.

No network: the transport is replaced with recorded-shape payloads.

Runs under pytest, or directly:  venv\\Scripts\\python.exe tests/test_etf_flows.py
"""

import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import market_data as md  # noqa: E402

# Ten trading sessions: note 3-4 Jan and 10-11 Jan are a weekend and simply absent.
SESSIONS = [
    ("2026-01-01", 100.0), ("2026-01-02", -50.0), ("2026-01-05", 200.0),
    ("2026-01-06", 10.0), ("2026-01-07", -20.0), ("2026-01-08", 300.0),
    ("2026-01-09", 40.0), ("2026-01-12", -5.0), ("2026-01-13", 80.0),
    ("2026-01-14", 25.0),
]

HISTORY = [
    {"date": d, "totalNetInflow": f, "totalNetAssets": 1_000.0 + i,
     "cumNetInflow": 500.0 + i, "totalValueTraded": 9.0}
    for i, (d, f) in enumerate(SESSIONS)
]

METRICS = {
    "totalNetAssetsPercentage": {"value": "0.06426528", "status": "1"},
    "totalTokenHoldings": {"value": "1291147.89233952", "status": "1"},
    # Deliberately disagrees with the history endpoint's cumulative figure.
    "cumNetInflow": {"value": "99999.0", "status": "1"},
    "list": [
        {"ticker": "SMALL", "institute": "Minor", "fee": {"value": "0.00250000"},
         "netAssets": {"value": "100.0"}, "dailyNetInflow": {"value": "5.0"},
         "cumNetInflow": {"value": "50.0"}},
        {"ticker": "BIG", "institute": "Major ", "fee": {"value": "0.00150000"},
         "netAssets": {"value": "900.0"}, "dailyNetInflow": {"value": "-7.0"},
         "cumNetInflow": {"value": "-400.0"}},
    ],
}


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


@contextmanager
def patched(body_for):
    """Swap httpx.post for a stub returning `body_for(endpoint)`, then put it back.

    `md.httpx` is the shared httpx module, so patching it without restoring would leak
    into every other test in the same pytest process.
    """
    original = md.httpx.post

    def post(url, json=None, timeout=None, headers=None):  # noqa: A002
        return FakeResponse(body_for(url.rsplit("/", 1)[-1]))

    md.httpx.post = post
    try:
        yield
    finally:
        md.httpx.post = original


def ok(endpoint):
    data = HISTORY if endpoint == "historicalInflowChart" else METRICS
    return {"code": 0, "msg": None, "data": data}


def load(asset="BTC", days=90):
    with patched(ok):
        return md.etf_flows(asset, days)


# --- windows are counted in sessions -------------------------------------------------------

def test_five_session_window_sums_five_rows():
    # The last five sessions are 8th, 9th, 12th, 13th, 14th Jan -> 300+40-5+80+25.
    assert load()["flow_5d"] == 440.0


def test_twenty_session_window_is_none_when_history_is_shorter():
    # Ten sessions can't answer a twenty-session question; None says so, 0 would lie.
    assert load()["flow_20d"] is None


def test_window_ignores_the_calendar_gap():
    # A weekend sits inside the last five sessions. If the reader were date-based rather
    # than row-based, this window would silently cover a different span.
    assert load()["flow_5d"] != sum(f for _, f in SESSIONS[-7:])


# --- headline and chart come from one source ------------------------------------------------

def test_latest_flow_matches_the_last_charted_row():
    data = load()
    assert data["latest_flow"] == data["daily"][-1]["net_flow"] == 25.0


def test_cumulative_comes_from_history_not_metrics():
    # METRICS carries 99999.0; taking it would contradict the charted series.
    assert load()["cum_flow"] == 500.0 + len(SESSIONS) - 1


def test_as_of_is_the_newest_session():
    assert load()["as_of"] == "2026-01-14"


def test_rows_are_sorted_oldest_first():
    dates = [r["date"] for r in load()["daily"]]
    assert dates == sorted(dates)


def test_days_argument_trims_the_history():
    assert len(load(days=3)["daily"]) == 3


# --- the {value,...} wrapper and string numbers ----------------------------------------------

def test_wrapped_string_numbers_are_parsed():
    data = load()
    assert abs(data["share_of_market_cap"] - 0.06426528) < 1e-9
    assert abs(data["token_holdings"] - 1291147.89233952) < 1e-6


def test_funds_are_parsed_and_sorted_by_size():
    funds = load()["funds"]
    assert [f["ticker"] for f in funds] == ["BIG", "SMALL"]
    assert funds[0]["net_flow"] == -7.0
    assert funds[0]["cum_flow"] == -400.0


def test_issuer_whitespace_is_stripped():
    # The live feed pads some issuer names ("BlackRock ").
    assert load()["funds"][0]["institute"] == "Major"


# --- failures arrive with HTTP 200 -----------------------------------------------------------

def test_body_level_error_raises():
    with patched(lambda ep: {"code": "401", "msg": "API key missing.", "data": None}):
        try:
            md.etf_flows("BTC")
        except RuntimeError as exc:
            assert "API key missing" in str(exc)
        else:
            raise AssertionError("a body-level error must raise, not return empty data")


def test_a_broken_fund_list_still_returns_the_series():
    # The breakdown is a bonus; losing it must not cost the chart.
    def body_for(endpoint):
        if endpoint == "currentEtfDataMetrics":
            return {"code": "500", "msg": "boom", "data": None}
        return {"code": 0, "data": HISTORY}

    with patched(body_for):
        data = md.etf_flows("BTC")
    assert data["funds"] == []
    assert len(data["daily"]) == len(SESSIONS)
    assert data["latest_flow"] == 25.0


def test_unknown_asset_is_rejected():
    # Rejected before any request, so this needs no stub.
    try:
        md.etf_flows("DOGE")
    except ValueError:
        pass
    else:
        raise AssertionError("only BTC and ETH have US spot ETFs")


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
