"""
Evaluation harness for the research agent.

Two layers, because they answer different questions:

TRAJECTORY (deterministic, no judge calls) — did the agent *behave* correctly?
  - ticker recall     : did the planner detect the tickers the question is about?
  - focus coverage    : did the plan cover the expected angles (technicals, news, …)?
  - evidence density  : findings produced per sub-question
  - data availability : share of findings backed by real data rather than "unavailable"
  - source diversity  : how many distinct data sources the run actually used
  - critic passes     : how many research rounds it took
  - completeness      : did it produce both a report and an analyst view?

QUALITY (LLM-as-judge) — was the *output* any good?
  - report faithfulness / relevance, and analyst groundedness (see judge.py)

Usage:
    python -m eval.evaluate                 # trajectory + judge on the whole dataset
    python -m eval.evaluate --no-judge      # trajectory only (fewer API calls)
    python -m eval.evaluate --limit 1       # just the first question

NOTE ON COST: every question runs the full agent (~4-5 LLM calls) plus one judge call.
The free tier is limited — use --limit while iterating.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

from config import ROOT_DIR, settings
from agent.graph import build_graph
from eval.judge import judge_run

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

UNAVAILABLE_MARKERS = ("unavailable", "not provided", "no data", "not available")


def _is_real_data(finding: str) -> bool:
    """A finding counts as 'real data' unless it says the data couldn't be obtained."""
    low = finding.lower()
    return not any(m in low for m in UNAVAILABLE_MARKERS)


def evaluate_one(item: dict, judge: bool) -> dict:
    """Run the agent on one question and score the run."""
    question = item["question"]
    agent = build_graph()
    result = agent.invoke({"question": question})

    plan = result.get("plan")
    evidence = result.get("evidence", [])
    report = result.get("report", "")
    view = result.get("analyst_view")

    # --- trajectory metrics ---
    expected_tickers = [t.upper() for t in item.get("expected_tickers", [])]
    found_tickers = [t.upper() for t in (plan.tickers if plan else [])]
    ticker_recall = (
        sum(1 for t in expected_tickers if t in found_tickers) / len(expected_tickers)
        if expected_tickers else None
    )

    expected_focus = [f.lower() for f in item.get("expected_focus", [])]
    plan_focus = {sq.focus.lower() for sq in plan.subquestions} if plan else set()
    focus_coverage = (
        sum(1 for f in expected_focus if f in plan_focus) / len(expected_focus)
        if expected_focus else None
    )

    n_subq = len(plan.subquestions) if plan else 0
    real = sum(1 for e in evidence if _is_real_data(e["finding"]))
    sources = {s.strip() for e in evidence for s in e["source"].split(",") if s.strip()}

    row = {
        "question": question,
        "ticker_recall": ticker_recall,
        "focus_coverage": focus_coverage,
        "evidence_count": len(evidence),
        "subquestions": n_subq,
        "data_availability": round(real / len(evidence), 2) if evidence else 0.0,
        "source_diversity": len(sources),
        "critic_passes": result.get("loops", 0),
        "has_report": bool(report),
        "has_analyst_view": view is not None,
    }

    # --- quality metrics (judge) ---
    if judge:
        try:
            scores = judge_run(question, evidence, report, view)
            row.update(
                {
                    "report_faithfulness": scores.report_faithfulness,
                    "report_relevance": scores.report_relevance,
                    "analyst_groundedness": scores.analyst_groundedness,
                    "judge_reason": scores.reason,
                }
            )
        except Exception as exc:  # noqa: BLE001 - one failed grade shouldn't sink the run
            row["judge_error"] = str(exc)[:120]
    return row


def _avg(rows: list[dict], key: str):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else None


def print_report(rows: list[dict], judged: bool) -> None:
    line = "=" * 72
    print("\n" + line)
    print("AGENT EVALUATION REPORT")
    print(line)
    print(f"Questions evaluated : {len(rows)}")
    print("\n--- Trajectory (did it behave correctly?) ---")
    print(f"Ticker recall       : {_avg(rows, 'ticker_recall')}   (found the right tickers)")
    print(f"Focus coverage      : {_avg(rows, 'focus_coverage')}   (planned the expected angles)")
    print(f"Data availability   : {_avg(rows, 'data_availability')}   (findings with real data)")
    print(f"Source diversity    : {_avg(rows, 'source_diversity')}   (distinct sources per run)")
    print(f"Critic passes       : {_avg(rows, 'critic_passes')}   (research rounds)")
    print(f"Produced report     : {sum(r['has_report'] for r in rows)}/{len(rows)}")
    print(f"Produced analyst    : {sum(r['has_analyst_view'] for r in rows)}/{len(rows)}")

    if judged:
        print("\n--- Quality (LLM-as-judge) ---")
        print(f"Report faithfulness : {_avg(rows, 'report_faithfulness')}   (claims supported by evidence)")
        print(f"Report relevance    : {_avg(rows, 'report_relevance')}   (answers the question)")
        print(f"Analyst groundedness: {_avg(rows, 'analyst_groundedness')}   (stance follows the evidence)")
        errors = [r for r in rows if "judge_error" in r]
        if errors:
            print(f"Judge errors        : {len(errors)} (excluded from averages)")

    print("\n" + "-" * 72)
    for r in rows:
        print(f"\nQ: {r['question'][:64]}")
        print(f"   tickers {r['ticker_recall']} | focus {r['focus_coverage']} | "
              f"evidence {r['evidence_count']} | real-data {r['data_availability']} | "
              f"sources {r['source_diversity']} | passes {r['critic_passes']}")
        if "report_faithfulness" in r:
            print(f"   judge: faith {r['report_faithfulness']} | rel {r['report_relevance']} | "
                  f"analyst {r['analyst_groundedness']}")
            print(f"   {r.get('judge_reason', '')[:100]}")
        if "judge_error" in r:
            print(f"   judge error: {r['judge_error']}")
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the research agent.")
    parser.add_argument("--dataset", default=str(ROOT_DIR / "eval" / "dataset.json"))
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge scoring.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions.")
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.limit:
        dataset = dataset[: args.limit]

    judge = not args.no_judge
    rows = []
    for i, item in enumerate(dataset, start=1):
        print(f"[{i}/{len(dataset)}] running: {item['question'][:60]} …")
        rows.append(evaluate_one(item, judge))
        if judge and i < len(dataset):
            time.sleep(settings.judge_delay_seconds)  # pace against free-tier limits

    print_report(rows, judged=judge)


if __name__ == "__main__":
    main()
