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
from eval.judge import judge_provider, judge_run
from eval.numeric_check import check_numbers

HISTORY_DIR = ROOT_DIR / "eval" / "history"

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
        "case_type": item.get("case_type", "—"),
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

    # --- deterministic faithfulness (no model, no cost, always runs) ---
    row.update(check_numbers(report, evidence, view, question=question))

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


def save_run(rows: list[dict], judged: bool) -> Path:
    """Append this run to the history so scores can be compared, not just read once."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    model, independent = judge_provider()
    path = HISTORY_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent_model": settings.gemini_model,
        "judge_model": model if judged else None,
        "judge_independent": independent if judged else None,
        "cases": len(rows),
        "summary": {k: _avg(rows, k) for k in (
            "ticker_recall", "focus_coverage", "data_availability", "source_diversity",
            "critic_passes", "numeric_traceability", "report_faithfulness",
            "report_relevance", "analyst_groundedness",
        )},
        "rows": rows,
    }, indent=2), encoding="utf-8")
    return path


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def previous_run(exclude: Path) -> dict | None:
    """The most recent earlier run over the SAME cases as the run saved at `exclude`.

    Averages from different case sets can't be compared: a 20-case run measured against a
    4-case honesty run would print deltas that track the change in dataset, not in the
    agent. So only a run with exactly the same questions counts as a baseline — which also
    lets a filtered run (--case-type honesty) compare against the last run filtered alike.
    """
    if not HISTORY_DIR.exists():
        return None
    current = _load(exclude) or {}
    questions = {r.get("question") for r in current.get("rows", [])}
    for path in sorted(HISTORY_DIR.glob("*.json"), reverse=True):  # names sort by time
        if path == exclude:
            continue
        run = _load(path)
        if run and {r.get("question") for r in run.get("rows", [])} == questions:
            return run
    return None


def _delta(now, before) -> str:
    """A signed change against the previous run, or blank when not comparable."""
    if not isinstance(now, (int, float)) or not isinstance(before, (int, float)):
        return ""
    diff = now - before
    if abs(diff) < 0.0005:
        return "   (unchanged)"
    return f"   ({diff:+.3f} vs last run)"


def print_report(rows: list[dict], judged: bool, prev: dict | None = None) -> None:
    line = "=" * 72
    before = (prev or {}).get("summary", {})

    def d(key):
        return _delta(_avg(rows, key), before.get(key))

    print("\n" + line)
    print("AGENT EVALUATION REPORT")
    print(line)
    print(f"Questions evaluated : {len(rows)}")
    if prev:
        print(f"Compared against    : {prev.get('timestamp', 'previous run')} "
              f"(the same {prev.get('cases', '?')} cases)")
    else:
        print("Compared against    : nothing yet — no earlier run covered exactly these "
              "cases, so deltas start from the next one")
    print("\n--- Trajectory (did it behave correctly?) ---")
    print(f"Ticker recall       : {_avg(rows, 'ticker_recall')}   (found the right tickers){d('ticker_recall')}")
    print(f"Focus coverage      : {_avg(rows, 'focus_coverage')}   (planned the expected angles){d('focus_coverage')}")
    print(f"Data availability   : {_avg(rows, 'data_availability')}   (findings with real data){d('data_availability')}")
    print(f"Source diversity    : {_avg(rows, 'source_diversity')}   (distinct sources per run){d('source_diversity')}")
    print(f"Critic passes       : {_avg(rows, 'critic_passes')}   (research rounds){d('critic_passes')}")
    print(f"Produced report     : {sum(r['has_report'] for r in rows)}/{len(rows)}")
    print(f"Produced analyst    : {sum(r['has_analyst_view'] for r in rows)}/{len(rows)}")

    # Deterministic first: it needs no model, so it's the one number here that can't be
    # talked into a good score.
    print("\n--- Faithfulness (deterministic, no model involved) ---")
    print(f"Numeric traceability: {_avg(rows, 'numeric_traceability')}   "
          f"(figures found in the evidence){d('numeric_traceability')}")
    flagged = [r for r in rows if r.get("unverified_count")]
    print(f"Cases with untraced : {len(flagged)}/{len(rows)}   "
          "(may be derived rather than invented — worth reading)")

    if judged:
        model, independent = judge_provider()
        print(f"\n--- Quality (LLM-as-judge: {model}) ---")
        if not independent:
            print("  ! Same model family as the agent, so this grade is NOT")
            print("    provider-independent. Set ANTHROPIC_API_KEY or OPENAI_API_KEY")
            print("    to switch the judge to a different provider.")
        print(f"Report faithfulness : {_avg(rows, 'report_faithfulness')}   (claims supported by evidence){d('report_faithfulness')}")
        print(f"Report relevance    : {_avg(rows, 'report_relevance')}   (answers the question){d('report_relevance')}")
        print(f"Analyst groundedness: {_avg(rows, 'analyst_groundedness')}   (stance follows the evidence){d('analyst_groundedness')}")
        errors = [r for r in rows if "judge_error" in r]
        if errors:
            print(f"Judge errors        : {len(errors)} (excluded from averages)")

    # Honesty cases are the ones worth eyeballing: the failure mode they test is a
    # confident answer where the right move was to decline.
    honesty = [r for r in rows if str(r.get("case_type", "")).startswith("honesty_")]
    if honesty:
        print("\n--- Honesty cases (should decline, not confabulate) ---")
        for r in honesty:
            bits = [f"real-data {r['data_availability']}"]
            if r.get("numeric_traceability") is not None:
                bits.append(f"traceability {r['numeric_traceability']}")
            if "report_faithfulness" in r:
                bits.append(f"faith {r['report_faithfulness']}")
            print(f"  {r['case_type']:28} {' | '.join(bits)}")

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
        # Untraced figures, shown where they appeared: a derived number and an invented
        # one look identical as bare values, and only the surrounding text tells them apart.
        for u in r.get("unverified") or []:
            print(f"   untraced {u['value']:g}: {u['context']}")
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the research agent.")
    parser.add_argument("--dataset", default=str(ROOT_DIR / "eval" / "dataset.json"))
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge scoring.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions.")
    parser.add_argument(
        "--case-type", default=None,
        help="Only cases whose case_type starts with this (e.g. 'honesty' for the "
             "decline-rather-than-confabulate cases). A full run costs ~120 LLM calls, "
             "so this is how you check one behaviour without spending a day's quota.",
    )
    args = parser.parse_args()

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.case_type:
        dataset = [d for d in dataset
                   if str(d.get("case_type", "")).startswith(args.case_type)]
        if not dataset:
            sys.exit(f"No cases with case_type starting '{args.case_type}'.")
    if args.limit:
        dataset = dataset[: args.limit]

    judge = not args.no_judge
    rows, skipped = [], []
    consecutive_failures = 0
    for i, item in enumerate(dataset, start=1):
        print(f"[{i}/{len(dataset)}] running: {item['question'][:60]} …")
        try:
            rows.append(evaluate_one(item, judge))
            consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001 - one failed case shouldn't lose the rest
            # Results are only saved at the end, so an unhandled error here used to throw
            # away every finished case. A single failure now skips just that case; two in a
            # row almost always means the daily quota ran out, and carrying on would only
            # fail every remaining case the same way.
            skipped.append((item["question"], f"{type(exc).__name__}: {str(exc)[:100]}"))
            print(f"    ! failed: {skipped[-1][1]}")
            consecutive_failures += 1
            if consecutive_failures >= 2:
                print("    ! two failures in a row — most likely quota; stopping early")
                skipped += [(d["question"], "not run") for d in dataset[i:]]
                break
        if judge and i < len(dataset):
            time.sleep(settings.judge_delay_seconds)  # pace against free-tier limits

    if not rows:
        sys.exit("No cases completed — nothing to save.")

    path = save_run(rows, judged=judge)
    print_report(rows, judged=judge, prev=previous_run(exclude=path))
    if skipped:
        # A partial run covers fewer questions, so it can never become the baseline for
        # a full one.
        print(f"\nIncomplete run: {len(skipped)} case(s) not scored")
        for question, reason in skipped:
            print(f"  - {question[:60]}  ({reason})")
    print(f"\nRun saved to {path.relative_to(ROOT_DIR)} — the next run compares against it.")


if __name__ == "__main__":
    main()
