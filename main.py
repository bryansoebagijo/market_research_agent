"""
CLI entry point for the Markets Research Analyst agent.

    python main.py "Should I be cautious on NVDA and BTC this week?"

For now it runs the planner and prints the research plan. As more nodes are added,
the same command will run the full research → critique → report pipeline.
"""

import sys

from agent.graph import build_graph


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python main.py "your markets question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    agent = build_graph()

    print(f"\nQuestion: {question}\n")
    print("Planning research …\n")

    result = agent.invoke({"question": question})
    plan = result["plan"]

    print("=" * 66)
    print("RESEARCH PLAN")
    print("=" * 66)
    print(f"Tickers  : {', '.join(plan.tickers) or '(none detected)'}")
    print(f"Angle    : {plan.rationale}\n")
    for i, sq in enumerate(plan.subquestions, start=1):
        print(f"  {i}. [{sq.focus}] {sq.question}")
    print("=" * 66)


if __name__ == "__main__":
    main()
