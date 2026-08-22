"""
CLI entry point for the Markets Research Analyst agent.

    python main.py "Is now a risky time to hold NVDA and BTC?"
    python main.py --review "Is now a risky time to hold NVDA and BTC?"

With --review the agent pauses after planning and asks you to approve, revise, or
cancel the research plan before it spends any time or API calls on research.
"""

import argparse
import sys
import uuid

from langgraph.types import Command

from agent.graph import build_graph

# Windows consoles default to cp1252, which can't encode characters the LLM may emit
# (arrows, curly quotes, etc.). Force UTF-8 so printing findings never crashes.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001 - older Pythons / non-reconfigurable streams
    pass

LINE = "=" * 70


def print_plan(payload: dict) -> None:
    """Show the proposed research plan to the human reviewer."""
    print("\n" + LINE)
    print("PROPOSED RESEARCH PLAN — your approval needed")
    print(LINE)
    print(f"Tickers : {', '.join(payload.get('tickers') or []) or '(none detected)'}")
    print(f"Angle   : {payload.get('rationale', '')}\n")
    for i, sq in enumerate(payload.get("subquestions", []), start=1):
        tk = f" ({', '.join(sq['tickers'])})" if sq.get("tickers") else ""
        print(f"  {i}. [{sq['focus']}]{tk} {sq['question']}")
    print(LINE)


def ask_decision() -> dict:
    """Prompt the human for approve / revise / cancel."""
    while True:
        choice = input(
            "\n[a] approve and research   [r] revise with feedback   [c] cancel\n> "
        ).strip().lower()
        if choice in ("a", "approve", ""):
            return {"action": "approve"}
        if choice in ("r", "revise"):
            feedback = input("What should the plan do differently?\n> ").strip()
            if feedback:
                return {"action": "revise", "feedback": feedback}
            print("No feedback given — please type what to change.")
            continue
        if choice in ("c", "cancel"):
            return {"action": "cancel"}
        print("Please enter a, r, or c.")


def print_results(result: dict) -> None:
    """Print the plan, evidence, critic verdict, and final report."""
    plan = result["plan"]
    print("\n" + LINE)
    print("RESEARCH PLAN")
    print(LINE)
    print(f"Tickers  : {', '.join(plan.tickers) or '(none detected)'}")
    print(f"Angle    : {plan.rationale}\n")
    for i, sq in enumerate(plan.subquestions, start=1):
        tk = f" ({', '.join(sq.tickers)})" if sq.tickers else ""
        print(f"  {i}. [{sq.focus}]{tk} {sq.question}")

    print("\n" + LINE)
    print("EVIDENCE GATHERED")
    print(LINE)
    for i, ev in enumerate(result.get("evidence", []), start=1):
        print(f"\n[{i}] {ev['subquestion']}")
        print(f"    -> {ev['finding']}")
        print(f"    source: {ev['source']}")

    print("\n" + LINE)
    print("CRITIC")
    print(LINE)
    print(f"Research passes : {result.get('loops', 1)}")
    print(f"Coverage OK?    : {result.get('complete')}")
    print(f"Assessment      : {result.get('critique', '(none)')}")

    print("\n" + LINE)
    print("REPORT")
    print(LINE)
    print(result.get("report", "(no report generated)"))

    view = result.get("analyst_view")
    if view:
        from agent.nodes.analyst import DISCLAIMER as ANALYST_DISCLAIMER

        # State restored from a checkpoint (--review) can arrive as a dict, so read
        # fields in a way that works for both a model and a dict.
        def f(name, default=""):
            return view.get(name, default) if isinstance(view, dict) else getattr(view, name, default)

        print("\n" + LINE)
        print("ANALYST VIEW (professional-style interpretation)")
        print(LINE)
        print(f"Stance     : {f('stance')}")
        print(f"Conviction : {f('conviction')}   |   Horizon: {f('horizon')}")
        for heading, key in (
            ("Why:", "reasoning"),
            ("What would change this view:", "what_would_change_it"),
            ("Key risks:", "key_risks"),
        ):
            print(f"\n{heading}")
            for point in f(key, []) or []:
                print(f"  - {point}")
        print(ANALYST_DISCLAIMER)
    print(LINE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Markets research analyst agent.")
    parser.add_argument("question", nargs="+", help="Your markets question.")
    parser.add_argument(
        "--review",
        action="store_true",
        help="Pause for human approval of the research plan before researching.",
    )
    args = parser.parse_args()
    question = " ".join(args.question)

    agent = build_graph(review=args.review)
    # Each run needs its own thread id so the checkpointer can save/resume its state.
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print(f"\nQuestion: {question}\n")
    print("Planning …\n" if args.review else "Planning and researching …\n")

    result = agent.invoke({"question": question}, config=config)

    # While the graph is suspended at an interrupt, show the payload and resume.
    while "__interrupt__" in result:
        print_plan(result["__interrupt__"][0].value)
        result = agent.invoke(Command(resume=ask_decision()), config=config)
        if result.get("cancelled"):
            print("\nCancelled — no research was run.")
            return
        if "__interrupt__" in result:
            print("\nRe-planned with your feedback.")

    print_results(result)


if __name__ == "__main__":
    main()
