"""
LLM-as-judge: score the agent's *output quality*, not just whether it ran.

Trajectory metrics (in evaluate.py) tell you the agent took the right steps. They can't
tell you whether the report is actually faithful to the evidence — only reading it can.
So a separate model reads the evidence, the report, and the analyst view, and scores:

  * report_faithfulness  — is every claim in the report supported by the evidence?
                           (the hallucination detector)
  * report_relevance     — does the report actually answer the question asked?
  * analyst_groundedness — does the analyst's stance follow from the evidence, and does
                           it acknowledge conflicting signals rather than cherry-pick?

The judge deliberately uses a DIFFERENT model from the one that produced the output, so
it isn't grading itself with the same blind spots. Both are Gemini here (that's what's
free), so the independence is partial — a different provider would be stronger.
"""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from config import settings


class JudgeScores(BaseModel):
    """Quality scores from the judge, each 0.0-1.0."""

    report_faithfulness: float = Field(
        description="1.0 = every claim in the report is supported by the evidence; "
        "0.0 = largely unsupported or invented."
    )
    report_relevance: float = Field(
        description="1.0 = fully answers the question asked; 0.0 = off-topic."
    )
    analyst_groundedness: float = Field(
        description="1.0 = the stance follows from the evidence and acknowledges "
        "conflicting signals; 0.0 = unsupported or one-sided."
    )
    reason: str = Field(description="One short sentence explaining the scores.")


JUDGE_SYSTEM = (
    "You are a strict, impartial evaluator of a markets research agent's output. You "
    "grade ONLY what you are given, on a 0.0-1.0 scale. Penalise any claim in the report "
    "or analyst view that is not supported by the evidence, and penalise an analyst "
    "stance that ignores conflicting signals present in the evidence. Note that findings "
    "which honestly say data was 'unavailable' are GOOD (honest), not a fault."
)


@lru_cache(maxsize=1)
def _judge_llm():
    """The judge model — different from the agent's generation model on purpose."""
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is not set; the judge cannot run.")
    return ChatGoogleGenerativeAI(
        model=settings.judge_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
    ).with_structured_output(JudgeScores)


def judge_run(question: str, evidence: list, report: str, analyst_view) -> JudgeScores:
    """Score one agent run. Raises on API failure so the caller can handle it."""
    ev_block = "\n".join(
        f"- {e['subquestion']}\n  {e['finding']} [sources: {e['source']}]" for e in evidence
    ) or "(no evidence gathered)"

    view_block = "(no analyst view)"
    if analyst_view is not None:
        view_block = (
            f"stance: {analyst_view.stance}\n"
            f"conviction: {analyst_view.conviction}\n"
            f"reasoning: {'; '.join(analyst_view.reasoning)}\n"
            f"invalidation: {'; '.join(analyst_view.what_would_change_it)}"
        )

    return _judge_llm().invoke(
        [
            ("system", JUDGE_SYSTEM),
            (
                "human",
                f"QUESTION:\n{question}\n\n"
                f"EVIDENCE THE AGENT GATHERED:\n{ev_block}\n\n"
                f"REPORT IT WROTE:\n{report}\n\n"
                f"ANALYST VIEW IT PRODUCED:\n{view_block}",
            ),
        ]
    )
