"""
Regression tests for how evaluation runs are compared (eval/evaluate.py).

A run's deltas only mean something against a run over the same cases. The first version
compared against whatever ran last, so a full 20-case run following a 4-case honesty run
would have printed deltas that measured the change in dataset rather than in the agent.
These pin the fix: a baseline has to cover exactly the same questions.

Runs under pytest, or directly:  venv\\Scripts\\python.exe tests/test_eval_history.py
"""

import json
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import eval.evaluate as evaluate  # noqa: E402

FULL = [f"question {i}" for i in range(20)]
HONESTY = ["question 4", "question 6", "question 7", "question 14"]


@contextmanager
def history_dir():
    """Point the harness at a throwaway history directory, restoring the real one after."""
    original = evaluate.HISTORY_DIR
    with tempfile.TemporaryDirectory() as tmp:
        evaluate.HISTORY_DIR = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            evaluate.HISTORY_DIR = original


def _run(directory: Path, name: str, questions: list[str]) -> Path:
    path = directory / f"{name}.json"
    path.write_text(json.dumps({
        "timestamp": name,
        "cases": len(questions),
        "summary": {},
        "rows": [{"question": q} for q in questions],
    }), encoding="utf-8")
    return path


def test_a_run_over_different_cases_is_not_a_baseline():
    with history_dir() as h:
        _run(h, "20260913-224124", HONESTY)
        current = _run(h, "20260914-100000", FULL)
        assert evaluate.previous_run(current) is None


def test_the_latest_run_over_the_same_cases_is_the_baseline():
    with history_dir() as h:
        _run(h, "20260910-090000", FULL)
        _run(h, "20260912-090000", FULL)
        _run(h, "20260913-090000", HONESTY)  # newer, but a different case set
        current = _run(h, "20260914-090000", FULL)
        prev = evaluate.previous_run(current)
        assert prev is not None and prev["timestamp"] == "20260912-090000"


def test_a_filtered_run_compares_against_the_last_run_filtered_alike():
    with history_dir() as h:
        _run(h, "20260913-090000", HONESTY)
        _run(h, "20260914-090000", FULL)
        current = _run(h, "20260915-090000", HONESTY)
        prev = evaluate.previous_run(current)
        assert prev is not None and prev["timestamp"] == "20260913-090000"


def test_question_order_does_not_matter():
    with history_dir() as h:
        _run(h, "20260913-090000", list(reversed(HONESTY)))
        current = _run(h, "20260914-090000", HONESTY)
        assert evaluate.previous_run(current) is not None


def test_no_history_means_no_baseline():
    with history_dir() as h:
        current = _run(h, "20260914-090000", FULL)
        assert evaluate.previous_run(current) is None


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
