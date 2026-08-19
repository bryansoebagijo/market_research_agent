"""
Critic node — the reflection step that makes this an agent, not a pipeline.

It looks at the evidence gathered so far and judges whether it's enough to answer the
user's question. If important gaps remain, it says so and proposes targeted follow-up
sub-questions; the graph's conditional edge then loops back to the researcher to fill
them. If coverage is good enough, it passes through.

A loop guard (`loops`, capped by MAX_CRITIC_LOOPS) guarantees the loop terminates — the
critic increments it every pass, and the router forces an exit once it hits the cap.
"""

from agent.llm import get_llm
from agent.state import AgentState, CriticVerdict


CRITIC_SYSTEM = (
    "You are the critic step of a markets research analyst. Given the user's question "
    "and the evidence gathered so far, judge whether coverage is sufficient to write a "
    "balanced, honest answer. Set complete=true if it is. If an IMPORTANT angle is "
    "missing, unanswered, or contradictory, set complete=false and propose up to 2 "
    "targeted follow-up sub-questions (each with a focus tag and relevant tickers) that "
    "would fill the biggest gaps. Be pragmatic: perfect data is impossible and some "
    "'unavailable' findings are acceptable — only loop back for genuinely important "
    "gaps, not minor polish."
)


def critic_node(state: AgentState) -> dict:
    """Assess the evidence; decide whether to loop back or finish."""
    llm = get_llm().with_structured_output(CriticVerdict)

    evidence = state.get("evidence", [])
    ev_block = "\n".join(
        f"- Q: {e['subquestion']}\n  finding: {e['finding']}" for e in evidence
    ) or "(no evidence gathered)"

    verdict: CriticVerdict = llm.invoke(
        [
            ("system", CRITIC_SYSTEM),
            (
                "human",
                f"User question:\n{state['question']}\n\nEvidence gathered:\n{ev_block}",
            ),
        ]
    )

    return {
        "critique": verdict.assessment,
        "complete": verdict.complete,
        "followups": verdict.followups,
        "loops": state.get("loops", 0) + 1,
    }
