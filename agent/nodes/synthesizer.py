"""
Synthesizer node — the final step that writes the report.

It takes the user's question and all the evidence the researcher gathered (each finding
tagged with its sources) and composes a balanced, honest answer that cites where each
claim came from. A fixed disclaimer is appended in code (not left to the model), so the
"not financial advice" framing is always present regardless of what the model writes.
"""

from agent.llm import get_llm, to_text
from agent.state import AgentState


# Appended in code so it's always present, exactly, on every report.
DISCLAIMER = (
    "\n\n---\nThis is best-effort research, not financial advice. It surfaces data and "
    "context; it does not predict prices or recommend buying or selling. Data may be "
    "delayed, incomplete, or unavailable, and any figures should be double-checked."
)

SYNTH_SYSTEM = (
    "You are the synthesis step of a markets research analyst. Using ONLY the evidence "
    "provided, write a balanced, honest answer to the user's question. Structure it as:\n"
    "1) a short bottom-line summary of the risk picture (2-3 sentences, balanced — never "
    "a buy/sell recommendation or price prediction);\n"
    "2) a few bullet points grouping the key findings, and after each point cite the "
    "source(s) it came from in parentheses, e.g. (CoinGecko), (web search);\n"
    "3) a brief 'What's uncertain or missing' note listing gaps in the data.\n"
    "Be neutral and factual. Do not invent data that isn't in the evidence."
)


def synthesizer_node(state: AgentState) -> dict:
    """Compose the final cited report from the gathered evidence."""
    evidence = state.get("evidence", [])
    ev_block = "\n".join(
        f"- {e['subquestion']}\n  finding: {e['finding']}\n  sources: {e['source']}"
        for e in evidence
    ) or "(no evidence was gathered)"

    llm = get_llm()
    msg = llm.invoke(
        [
            ("system", SYNTH_SYSTEM),
            (
                "human",
                f"User question:\n{state['question']}\n\nEvidence:\n{ev_block}",
            ),
        ]
    )
    report = to_text(msg).strip() + DISCLAIMER
    return {"report": report}
