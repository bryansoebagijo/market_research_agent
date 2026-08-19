"""
CLI entry point for the Markets Research Analyst agent.

    python main.py "Should I be cautious on NVDA and BTC this week?"

For now it runs the planner and prints the research plan. As more nodes are added,
the same command will run the full research → critique → report pipeline.
"""

import sys

from agent.graph import build_graph

# Windows consoles default to cp1252, which can't encode characters the LLM may emit
# (arrows, curly quotes, etc.). Force UTF-8 so printing findings never crashes.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 - older Pythons / non-reconfigurable streams
    pass


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python main.py "your markets question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    agent = build_graph()

    print(f"\nQuestion: {question}\n")
    print("Planning and researching … (this makes several API + LLM calls)\n")

    result = agent.invoke({"question": question})
    plan = result["plan"]

    print("=" * 70)
    print("RESEARCH PLAN")
    print("=" * 70)
    print(f"Tickers  : {', '.join(plan.tickers) or '(none detected)'}")
    print(f"Angle    : {plan.rationale}\n")
    for i, sq in enumerate(plan.subquestions, start=1):
        tk = f" ({', '.join(sq.tickers)})" if sq.tickers else ""
        print(f"  {i}. [{sq.focus}]{tk} {sq.question}")

    print("\n" + "=" * 70)
    print("EVIDENCE GATHERED")
    print("=" * 70)
    for i, ev in enumerate(result.get("evidence", []), start=1):
        print(f"\n[{i}] {ev['subquestion']}")
        print(f"    -> {ev['finding']}")
        print(f"    source: {ev['source']}")

    print("\n" + "=" * 70)
    print("CRITIC")
    print("=" * 70)
    print(f"Research passes : {result.get('loops', 1)}")
    print(f"Coverage OK?    : {result.get('complete')}")
    print(f"Assessment      : {result.get('critique', '(none)')}")

    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)
    print(result.get("report", "(no report generated)"))
    print("=" * 70)


if __name__ == "__main__":
    main()
