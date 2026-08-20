"""
Analyst node — a professional-investor read on what the evidence implies.

The synthesizer answers "what does the data say?". This node goes one step further and
answers "how would an experienced investor interpret it?" — a stance (Accumulate / Hold
/ Trim / Avoid / Watch), how strong the conviction is, over what horizon, why, what
would change the view, and the main risks.

Two deliberate design choices, both for honesty:

  * It must state INVALIDATION conditions. A view you can't disprove is a vibe, not
    analysis; naming what would flip it keeps the output falsifiable and useful.
  * It gives no position sizing, no price targets, and no personalised instruction.
    This is a structured reading of the gathered evidence, not advice — an LLM persona
    has no validated edge, and the disclaimer says so.
"""

from agent.llm import get_llm
from agent.state import AgentState, AnalystView


ANALYST_SYSTEM = (
    "You are an experienced portfolio manager reviewing a research brief. You have seen "
    "many cycles, you are sceptical of hype, and you weigh evidence rather than "
    "narrative. Read the research report and evidence, then give your professional read.\n"
    "\n"
    "Rules:\n"
    "- Choose a stance from: Accumulate, Hold, Trim, Avoid, Watch. If several assets are "
    "covered, give a stance per asset.\n"
    "- Ground every point in the evidence provided. Never invent data.\n"
    "- Weigh conflicting signals openly (e.g. strong fundamentals but overbought "
    "technicals) rather than cherry-picking one side.\n"
    "- Set conviction honestly: Low when data is thin, conflicting, or missing.\n"
    "- Always name concrete conditions that would change your view.\n"
    "- Do NOT give position sizes, price targets, or personalised instructions, and do "
    "not address the reader's own portfolio. This is an interpretation, not advice."
)

DISCLAIMER = (
    "\n(Analyst view: an informational interpretation of the evidence above, produced by "
    "a language model with no validated track record. It is not financial advice, not a "
    "prediction, and not personalised to your situation or risk tolerance.)"
)


def analyst_node(state: AgentState) -> dict:
    """Produce a structured professional-style view from the report + evidence."""
    evidence = state.get("evidence", [])
    ev_block = "\n".join(
        f"- {e['subquestion']}\n  {e['finding']}" for e in evidence
    ) or "(no evidence gathered)"

    llm = get_llm().with_structured_output(AnalystView)
    view: AnalystView = llm.invoke(
        [
            ("system", ANALYST_SYSTEM),
            (
                "human",
                f"Question:\n{state['question']}\n\n"
                f"Research report:\n{state.get('report', '(none)')}\n\n"
                f"Underlying evidence:\n{ev_block}",
            ),
        ]
    )
    return {"analyst_view": view}
