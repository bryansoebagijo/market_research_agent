"""
Tests for grading a run later from its stored outputs (eval/evaluate.py).

Two failures drove these. The first full run lost its judge scores to a spent quota and,
because only scores were stored, grading it would have meant running all 20 agents again.
Then the first --rejudge pass graded nothing, because a per-minute limit was mistaken for
a spent day and the pass gave up on a refusal that clears in under a minute.

So these pin: telling a daily cap apart from a per-minute one, waiting out the latter,
stopping on the former, grading only what's still ungraded, and never treating an
unfinished run as a baseline.

Runs under pytest, or directly:  venv\\Scripts\\python.exe tests/test_eval_rejudge.py
"""

import json
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval.evaluate as evaluate  # noqa: E402

# A day's allowance, gone: names the per-day quota explicitly.
DAILY_TEXT = (
    "Error calling model 'gemini-3.6-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "Quota exceeded for metric: generate_content_free_tier_requests, limit: 20 "
    "'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier' 'retryDelay': '12s'"
)
# This minute's allowance, gone: small limit, and it clears in under a minute.
RATE_TEXT = (
    "Error calling model 'gemini-3.6-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
    "Quota exceeded for metric: generate_content_free_tier_requests, limit: 5, "
    "model: gemini-3.6-flash. Please retry in 49.35s."
)


@contextmanager
def scripted_judge(responses: list):
    """Stand in for the judge: each call takes the next response, raising it if it's an error."""
    calls = []
    original = evaluate.judge_run

    def judge(question, evidence, report, view):
        calls.append(question)
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    evaluate.judge_run = judge
    try:
        yield calls
    finally:
        evaluate.judge_run = original


def scores(value: float = 1.0):
    return SimpleNamespace(report_faithfulness=value, report_relevance=value,
                           analyst_groundedness=value, reason="ok")


def failed_run(n: int):
    """Rows as a refused run left them, plus the outputs stored for each."""
    rows = [{"question": f"q{i}", "judge_error": DAILY_TEXT, "judge_quota_exhausted": True}
            for i in range(n)]
    outputs = [{"question": f"q{i}", "evidence": [], "report": "r", "analyst_view": None}
               for i in range(n)]
    return rows, outputs


# --- telling the two refusals apart -------------------------------------------------------

def test_a_days_allowance_is_recognised_as_daily():
    assert evaluate.quota_scope(DAILY_TEXT) == "daily"


def test_a_per_minute_limit_is_not_mistaken_for_a_spent_day():
    assert evaluate.quota_scope(RATE_TEXT) == "rate"


def test_a_small_limit_alone_reads_as_per_minute():
    assert evaluate.quota_scope("429 RESOURCE_EXHAUSTED ... limit: 5") == "rate"


def test_other_failures_are_neither():
    assert evaluate.quota_scope("ValueError: could not parse the judge's JSON") is None
    assert evaluate.quota_scope("503 UNAVAILABLE: high demand") is None


def test_retry_delay_is_read_from_either_wording():
    assert evaluate._retry_after(RATE_TEXT) == 49.35
    assert evaluate._retry_after("'retryDelay': '12s'") == 12.0
    assert evaluate._retry_after("no delay mentioned") is None


# --- the gate ------------------------------------------------------------------------------

def test_the_gate_closes_after_two_daily_refusals():
    gate = evaluate.JudgeGate()
    assert gate.record({"judge_quota_exhausted": True}) is True
    assert gate.record({"judge_quota_exhausted": True}) is False


def test_a_per_minute_refusal_never_closes_the_gate():
    gate = evaluate.JudgeGate()
    for _ in range(5):
        assert gate.record({"judge_rate_limited": True}) is True


def test_a_success_resets_the_gate():
    gate = evaluate.JudgeGate()
    gate.record({"judge_quota_exhausted": True})
    gate.record({"report_faithfulness": 1, "report_relevance": 1, "analyst_groundedness": 1})
    assert gate.record({"judge_quota_exhausted": True}) is True


# --- grading stored outputs ----------------------------------------------------------------

def test_rejudge_grades_only_the_rows_still_missing_scores():
    rows, outputs = failed_run(3)
    rows[0] = {"question": "q0", "report_faithfulness": 0.9, "report_relevance": 0.9,
               "analyst_groundedness": 0.9}
    with scripted_judge([scores(), scores()]) as calls:
        result = evaluate.rejudge_rows(rows, outputs)
    assert calls == ["q1", "q2"]
    assert result["graded"] == 2 and result["remaining"] == 0
    assert rows[0]["report_faithfulness"] == 0.9


def test_a_successful_grade_clears_the_old_failure():
    rows, outputs = failed_run(1)
    with scripted_judge([scores()]):
        evaluate.rejudge_rows(rows, outputs)
    assert "judge_error" not in rows[0] and "judge_quota_exhausted" not in rows[0]
    assert rows[0]["report_faithfulness"] == 1.0


def test_a_per_minute_limit_is_waited_out_and_the_row_retried():
    rows, outputs = failed_run(1)
    waits = []
    with scripted_judge([RuntimeError(RATE_TEXT), scores()]) as calls:
        result = evaluate.rejudge_rows(rows, outputs, sleeper=waits.append)
    assert calls == ["q0", "q0"]          # same row, tried again
    assert result["graded"] == 1
    assert result["rate_limit_waits"] == 1
    assert waits and waits[0] >= 49       # waited at least what the provider asked
    assert result["stopped_for_quota"] is False


def test_a_spent_day_stops_the_pass_instead_of_failing_every_case():
    rows, outputs = failed_run(5)
    refused = RuntimeError(DAILY_TEXT)
    with scripted_judge([scores(), refused, refused, scores(), scores()]) as calls:
        result = evaluate.rejudge_rows(rows, outputs, sleeper=lambda _s: None)
    assert calls == ["q0", "q1", "q2"]
    assert result["stopped_for_quota"] is True
    assert result["graded"] == 1 and result["remaining"] == 4


def test_a_non_quota_error_is_recorded_and_the_pass_carries_on():
    rows, outputs = failed_run(2)
    with scripted_judge([ValueError("bad json"), scores()]) as calls:
        result = evaluate.rejudge_rows(rows, outputs, sleeper=lambda _s: None)
    assert calls == ["q0", "q1"]
    assert "bad json" in rows[0]["judge_error"]
    assert "judge_quota_exhausted" not in rows[0]
    assert result["stopped_for_quota"] is False


def test_a_row_without_a_stored_output_is_skipped_rather_than_crashing():
    rows, outputs = failed_run(2)
    with scripted_judge([scores()]) as calls:
        result = evaluate.rejudge_rows(rows, outputs[:1], sleeper=lambda _s: None)
    assert calls == ["q0"]
    assert result["missing_output"] == 1


def test_a_busy_server_is_recognised_as_transient_not_quota():
    assert evaluate._is_transient_error(
        "503 UNAVAILABLE. This model is currently experiencing high demand.")
    assert not evaluate._is_transient_error(DAILY_TEXT)
    assert not evaluate._is_transient_error(RATE_TEXT)


def test_a_transient_failure_is_waited_out_and_the_row_retried():
    rows, outputs = failed_run(1)
    waits = []
    busy = RuntimeError("503 UNAVAILABLE. This model is currently experiencing high demand.")
    with scripted_judge([busy, scores()]) as calls:
        result = evaluate.rejudge_rows(rows, outputs, sleeper=waits.append,
                                       transient_wait=7)
    assert calls == ["q0", "q0"]          # same row, tried again
    assert result["graded"] == 1
    assert waits == [7]
    assert result["stopped_for_quota"] is False


# --- unfinished runs -----------------------------------------------------------------------

def test_an_unfinished_run_is_never_used_as_a_baseline():
    original = evaluate.HISTORY_DIR
    with tempfile.TemporaryDirectory() as tmp:
        evaluate.HISTORY_DIR = Path(tmp)
        try:
            rows = [{"question": "q0"}, {"question": "q1"}]
            (Path(tmp) / "20260914-090000.json").write_text(
                json.dumps({"rows": rows, "in_progress": True}), encoding="utf-8")
            current = Path(tmp) / "20260915-090000.json"
            current.write_text(json.dumps({"rows": rows}), encoding="utf-8")
            assert evaluate.previous_run(current) is None
        finally:
            evaluate.HISTORY_DIR = original


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
