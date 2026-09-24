"""
Evaluation harness for the research agent.

Three layers, because they fail in different ways:

TRAJECTORY (deterministic) — did the agent *behave* correctly?
  - ticker recall     : did the planner detect the tickers the question is about?
  - focus coverage    : did the plan cover the expected angles (technicals, news, …)?
  - data availability : share of findings backed by real data rather than "unavailable"
  - source diversity  : how many distinct data sources the run actually used
  - critic passes     : how many research rounds it took
  - completeness      : did it produce both a report and an analyst view?

NUMERIC TRACEABILITY (deterministic, see numeric_check.py) — does every figure in the
output appear in the evidence the agent gathered?

QUALITY (LLM-as-judge, see judge.py) — report faithfulness and relevance, and analyst
groundedness.

A run writes two files as it goes, after every case, so a killed session keeps what
finished:
  eval/history/<run>.json  scores; committed, and compared against later runs
  eval/outputs/<run>.json  what each case produced; gitignored, kept so the judge can
                           grade the run later without running the agent again

Usage:
    python -m eval.evaluate                      # agent + deterministic checks + judge
    python -m eval.evaluate --no-judge           # skip the judge now, grade later
    python -m eval.evaluate --case-type honesty  # one family of cases
    python -m eval.evaluate --limit 1            # first N cases while iterating
    python -m eval.evaluate --rejudge latest     # grade stored outputs; safe to re-run

NOTE ON COST: each case runs the agent (~4-5 LLM calls) plus one judge call, and the
judge model's free tier allows about 20 requests a day per project — one full run, shared
with anything else on the same key. --rejudge can spread grading over several days.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from config import ROOT_DIR, settings
from agent.graph import build_graph
from eval.judge import judge_provider, judge_run
from eval.numeric_check import check_numbers

HISTORY_DIR = ROOT_DIR / "eval" / "history"
OUTPUTS_DIR = ROOT_DIR / "eval" / "outputs"

SUMMARY_KEYS = (
    "ticker_recall", "focus_coverage", "data_availability", "source_diversity",
    "critic_passes", "numeric_traceability", "report_faithfulness",
    "report_relevance", "analyst_groundedness",
)
JUDGE_SCORE_KEYS = ("report_faithfulness", "report_relevance", "analyst_groundedness")
JUDGE_STATE_KEYS = ("judge_error", "judge_quota_exhausted", "judge_skipped")

# How providers say the quota is spent: Google, then Anthropic, then OpenAI.
QUOTA_MARKERS = ("RESOURCE_EXHAUSTED", "rate_limit", "RateLimitError", "insufficient_quota")

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

UNAVAILABLE_MARKERS = ("unavailable", "not provided", "no data", "not available")


def _is_real_data(finding: str) -> bool:
    """A finding counts as 'real data' unless it says the data couldn't be obtained."""
    low = finding.lower()
    return not any(m in low for m in UNAVAILABLE_MARKERS)


def _is_quota_error(error) -> bool:
    """True when a provider is refusing on quota, as opposed to any other failure."""
    text = str(error)
    return any(m in text for m in QUOTA_MARKERS) or re.search(r"\b429\b", text) is not None


def _retry_after(error) -> float | None:
    """How many seconds the provider asked us to wait, when it said."""
    text = str(error)
    match = (re.search(r"retry in ([\d.]+)s", text)
             or re.search(r"'retryDelay':\s*'(\d+)s'", text))
    return float(match.group(1)) if match else None


def quota_scope(error) -> str | None:
    """Whether a refusal spends the day's allowance or only this minute's.

    The two need opposite responses and look identical unless you read the limit and the
    retry delay. The judge model allows 5 requests a minute and 20 a day on the free tier,
    and a burst of retries trips the per-minute one in seconds. Waiting under a minute
    clears that; nothing clears a spent day.
    """
    if not _is_quota_error(error):
        return None
    text = str(error)
    if "PerDay" in text:
        return "daily"
    delay = _retry_after(error)
    if delay is not None:
        # A wait measured in seconds is a per-minute window; a day's cap won't clear soon.
        return "rate" if delay <= 300 else "daily"
    limit = re.search(r"limit:\s*(\d+)", text)
    if limit:
        return "rate" if int(limit.group(1)) <= 10 else "daily"
    return "daily"


def _is_judged(row: dict) -> bool:
    return all(key in row for key in JUDGE_SCORE_KEYS)


class JudgeGate:
    """Stops calling the judge once the DAY's allowance is gone.

    Only a daily refusal closes this. A per-minute refusal is transient, and treating the
    two alike abandoned an entire pass over a limit that clears in under a minute — which
    is exactly what happened the first time --rejudge ran.
    """

    def __init__(self, limit: int = 2):
        self.limit = limit
        self.streak = 0
        self.open = True

    def record(self, row: dict) -> bool:
        if row.get("judge_quota_exhausted"):
            self.streak += 1
        elif _is_judged(row):
            self.streak = 0
        if self.streak >= self.limit:
            self.open = False
        return self.open


def _judge_into(row: dict, question: str, evidence: list, report: str, view) -> None:
    """Grade one case into its row, replacing any judge failure recorded there before."""
    for key in (*JUDGE_STATE_KEYS, "judge_rate_limited", "judge_retry_after"):
        row.pop(key, None)
    try:
        scores = judge_run(question, evidence, report, view)
    except Exception as exc:  # noqa: BLE001 - one failed grade shouldn't sink the run
        # Long enough to keep the provider's own quota details. Truncating at 300 cut the
        # message off just before the limit and retry delay — the one part that tells a
        # daily cap apart from a per-minute one.
        row["judge_error"] = str(exc)[:900]
        scope = quota_scope(exc)
        if scope == "daily":
            row["judge_quota_exhausted"] = True
        elif scope == "rate":
            row["judge_rate_limited"] = True
            row["judge_retry_after"] = _retry_after(exc)
        return
    row.update({
        "report_faithfulness": scores.report_faithfulness,
        "report_relevance": scores.report_relevance,
        "analyst_groundedness": scores.analyst_groundedness,
        "judge_reason": scores.reason,
        # Kept per row: a run graded across several days could meet more than one judge.
        "judge_model": judge_provider()[0],
    })


def evaluate_one(item: dict, judge: bool) -> tuple[dict, dict]:
    """Run the agent on one question; return its scored row and what it produced."""
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

    # What the judge needs, kept so the case can be graded later without the agent.
    output = {
        "question": question,
        "evidence": evidence,
        "report": report,
        "analyst_view": view.model_dump() if hasattr(view, "model_dump") else view,
    }

    # --- quality metrics (judge) ---
    if judge:
        _judge_into(row, question, evidence, report, view)
    else:
        row["judge_skipped"] = "not requested"
    return row, output


def _avg(rows: list[dict], key: str):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else None


def write_run(run_id: str, rows: list[dict], judged: bool, *, in_progress: bool = False,
              started: str | None = None) -> Path:
    """Write a run's scores to history. Called after every case, not only at the end."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    model, independent = judge_provider()
    payload = {
        "timestamp": started or time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent_model": settings.gemini_model,
        "judge_model": model if judged else None,
        "judge_independent": independent if judged else None,
        "cases": len(rows),
        "summary": {k: _avg(rows, k) for k in SUMMARY_KEYS},
        "rows": rows,
    }
    if in_progress:
        payload["in_progress"] = True
    path = HISTORY_DIR / f"{run_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def write_outputs(run_id: str, outputs: list[dict]) -> Path:
    """Keep what each case produced, so the judge can grade it on a later day."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"{run_id}.json"
    path.write_text(json.dumps(outputs, indent=2, default=str), encoding="utf-8")
    return path


def _load(path: Path) -> dict | list | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def previous_run(exclude: Path) -> dict | None:
    """The most recent earlier run over the SAME cases as the run saved at `exclude`.

    Averages from different case sets can't be compared: a 20-case run measured against a
    4-case honesty run would print deltas that track the change in dataset, not in the
    agent. So only a finished run with exactly the same questions counts as a baseline —
    which also lets a filtered run (--case-type honesty) compare against the last run
    filtered alike. A run still marked in progress was cut short and is never a baseline.
    """
    if not HISTORY_DIR.exists():
        return None
    current = _load(exclude) or {}
    questions = {r.get("question") for r in current.get("rows", [])}
    for path in sorted(HISTORY_DIR.glob("*.json"), reverse=True):  # names sort by time
        if path == exclude:
            continue
        run = _load(path)
        if not run or run.get("in_progress"):
            continue
        if {r.get("question") for r in run.get("rows", [])} == questions:
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


def print_report(rows: list[dict], judged: bool, prev: dict | None = None,
                 run_id: str | None = None) -> None:
    line = "=" * 72
    before = (prev or {}).get("summary", {})
    rejudge_cmd = f"python -m eval.evaluate --rejudge {run_id or '<run>'}"

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
        pending = [r for r in rows if not _is_judged(r)]
        if pending:
            print(f"Not yet judged      : {len(pending)}/{len(rows)}   (grade later: {rejudge_cmd})")
    else:
        print(f"\nNot judged in this run — grade it later with: {rejudge_cmd}")

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
            print(f"   judge error: {r['judge_error'][:120]}")
        if r.get("judge_skipped") not in (None, "not requested"):
            print(f"   judge skipped: {r['judge_skipped']}")
        # Untraced figures, shown where they appeared: a derived number and an invented
        # one look identical as bare values, and only the surrounding text tells them apart.
        for u in r.get("unverified") or []:
            print(f"   untraced {u['value']:g}: {u['context']}")
    print(line)


def _is_transient_error(error) -> bool:
    """True for a provider saying "busy, try again" rather than "no".

    Gemini's free tier answers 503 UNAVAILABLE — "this model is currently experiencing
    high demand" — under load, and that was 13 of 15 judge failures on one pass and 15 of
    16 on the next. Unlike a quota refusal it clears on its own, so it's worth waiting out
    rather than abandoning the case.
    """
    text = str(error)
    return any(marker in text for marker in
               ("503", "UNAVAILABLE", "high demand", "overloaded"))


def rejudge_rows(rows: list[dict], outputs: list[dict], *, delay: float = 0.0,
                 on_progress=None, sleeper=time.sleep, rate_retries: int = 3,
                 transient_wait: float = 20.0) -> dict:
    """Grade every row that has no judge scores yet, from the outputs stored for it.

    Resumable by design, and it separates the two ways a provider says no. A spent day
    stops the pass, so running it again after the reset carries on from here. A per-minute
    limit is waited out and the same row retried — the first version treated both as fatal
    and graded nothing at all over a refusal that clears in under a minute.

    `sleeper` is injected so the waiting can be tested without actually waiting.
    """
    by_question = {o["question"]: o for o in outputs}
    gate = JudgeGate()
    graded = missing_output = rate_limit_waits = transient_waits = 0
    pending = [r for r in rows if not _is_judged(r)]

    for n, row in enumerate(pending, start=1):
        out = by_question.get(row["question"])
        if out is None:
            missing_output += 1
            continue

        for attempt in range(rate_retries + 1):
            _judge_into(row, out["question"], out["evidence"], out["report"],
                        out["analyst_view"])
            if attempt == rate_retries:
                break
            if row.get("judge_rate_limited"):
                # Wait exactly as long as the provider asked, plus a second of headroom.
                rate_limit_waits += 1
                sleeper((row.get("judge_retry_after") or 60) + 1)
                continue
            if _is_transient_error(row.get("judge_error")):
                # "High demand" clears on its own; the case is fine, the server is busy.
                transient_waits += 1
                sleeper(transient_wait)
                continue
            break

        if _is_judged(row):
            graded += 1
        if on_progress:
            on_progress(row)
        if not gate.record(row):
            break
        if delay and n < len(pending):
            sleeper(delay)

    return {
        "graded": graded,
        "remaining": sum(1 for r in rows if not _is_judged(r)),
        "stopped_for_quota": not gate.open,
        "missing_output": missing_output,
        "rate_limit_waits": rate_limit_waits,
        "transient_waits": transient_waits,
    }


def rejudge(run_ref: str) -> None:
    """Grade a saved run's stored outputs with the judge, resuming where it last stopped."""
    if run_ref == "latest":
        stored = sorted(OUTPUTS_DIR.glob("*.json")) if OUTPUTS_DIR.exists() else []
        if not stored:
            sys.exit("No stored outputs yet. Outputs are saved during a run, so run the eval "
                     "first.")
        run_id = stored[-1].stem
    else:
        run_id = run_ref

    history_path = HISTORY_DIR / f"{run_id}.json"
    run, outputs = _load(history_path), _load(OUTPUTS_DIR / f"{run_id}.json")
    if run is None:
        sys.exit(f"No history for run {run_id}.")
    if outputs is None:
        sys.exit(f"Run {run_id} has no stored outputs: it predates output saving, so it can "
                 "only be judged by running the agent again.")

    rows = run["rows"]
    todo = sum(1 for r in rows if not _is_judged(r))
    print(f"Re-judging run {run_id}: {todo} of {len(rows)} case(s) still need a judge score")

    def save(_row=None):
        write_run(run_id, rows, True, started=run.get("timestamp"),
                  in_progress=bool(run.get("in_progress")))

    result = {"graded": 0, "remaining": todo, "stopped_for_quota": False, "missing_output": 0}
    if todo:
        # Saved after every graded case, so an interrupted pass keeps its progress.
        # A judge-only pass has no agent call between judgements, so it needs the
        # per-minute interval rather than the mid-run delay.
        result = rejudge_rows(rows, outputs, delay=settings.judge_min_interval_seconds,
                              on_progress=save)
        save()

    print_report(rows, judged=True, prev=previous_run(exclude=history_path), run_id=run_id)
    print(f"\nGraded {result['graded']} this pass; {result['remaining']} still ungraded.")
    if result.get("rate_limit_waits"):
        print(f"Waited out the per-minute limit {result['rate_limit_waits']} time(s) — "
              "that one clears in under a minute, unlike the daily cap.")
    if result["stopped_for_quota"]:
        print("Stopped early because the day's judge allowance is spent. Run the same "
              "command again after it resets (midnight Pacific) to finish.")
    if result["missing_output"]:
        print(f"{result['missing_output']} case(s) had no stored output and stayed ungraded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the research agent.")
    parser.add_argument("--dataset", default=str(ROOT_DIR / "eval" / "dataset.json"))
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the judge now; grade the run later with --rejudge.")
    parser.add_argument("--limit", type=int, default=None, help="Only the first N questions.")
    parser.add_argument(
        "--case-type", default=None,
        help="Only cases whose case_type starts with this (e.g. 'honesty' for the "
             "decline-rather-than-confabulate cases). A full run costs ~120 LLM calls, "
             "so this is how you check one behaviour without spending a day's quota.",
    )
    parser.add_argument(
        "--rejudge", metavar="RUN", default=None,
        help="Grade a saved run's stored outputs instead of running the agent: a run id "
             "such as 20260915-140512, or 'latest'. Cases already graded are skipped, so "
             "it's safe to run again after the quota resets.",
    )
    args = parser.parse_args()

    if args.rejudge:
        rejudge(args.rejudge)
        return

    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    if args.case_type:
        dataset = [d for d in dataset
                   if str(d.get("case_type", "")).startswith(args.case_type)]
        if not dataset:
            sys.exit(f"No cases with case_type starting '{args.case_type}'.")
    if args.limit:
        dataset = dataset[: args.limit]

    judge_requested = not args.no_judge
    run_id = time.strftime("%Y%m%d-%H%M%S")
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    rows, outputs, skipped = [], [], []
    gate = JudgeGate()
    consecutive_failures = 0
    for i, item in enumerate(dataset, start=1):
        print(f"[{i}/{len(dataset)}] running: {item['question'][:60]} …")
        judge_now = judge_requested and gate.open
        try:
            row, output = evaluate_one(item, judge_now)
        except Exception as exc:  # noqa: BLE001 - one failed case shouldn't lose the rest
            # A single failure skips just that case; two in a row almost always means the
            # agent's own quota ran out, and carrying on would fail every remaining case.
            skipped.append((item["question"], f"{type(exc).__name__}: {str(exc)[:100]}"))
            print(f"    ! failed: {skipped[-1][1]}")
            consecutive_failures += 1
            if consecutive_failures >= 2:
                print("    ! two failures in a row — most likely quota; stopping early")
                skipped += [(d["question"], "not run") for d in dataset[i:]]
                break
            continue

        consecutive_failures = 0
        if judge_requested and not judge_now:
            row["judge_skipped"] = "quota spent earlier in this run"
        rows.append(row)
        outputs.append(output)
        # Saved after every case, so a killed session keeps everything that finished.
        write_run(run_id, rows, judge_requested, started=started, in_progress=True)
        write_outputs(run_id, outputs)
        if judge_now and not gate.record(row):
            print("    ! judge quota spent — finishing the run without it; grade the rest "
                  f"later with: python -m eval.evaluate --rejudge {run_id}")
        if judge_now and i < len(dataset):
            time.sleep(settings.judge_delay_seconds)  # pace against free-tier limits

    if not rows:
        sys.exit("No cases completed — nothing to save.")

    path = write_run(run_id, rows, judge_requested, started=started)
    print_report(rows, judged=judge_requested, prev=previous_run(exclude=path), run_id=run_id)
    if skipped:
        # A partial run covers fewer questions, so it can never become the baseline for
        # a full one.
        print(f"\nIncomplete run: {len(skipped)} case(s) not scored")
        for question, reason in skipped:
            print(f"  - {question[:60]}  ({reason})")
    print(f"\nRun saved to {path.relative_to(ROOT_DIR)} — later runs over the same cases "
          "compare against it.")


if __name__ == "__main__":
    main()
