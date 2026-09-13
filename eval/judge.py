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
it isn't grading itself with the same blind spots. Set ANTHROPIC_API_KEY or
OPENAI_API_KEY and it becomes genuinely provider-independent; with neither, it falls
back to a different Gemini model and the evaluation report says so, because a
same-family grade shouldn't be presented as an independent one.

An LLM judge is also not the only check here: eval/numeric_check.py verifies every
figure in the report against the gathered evidence deterministically, with no model
involved at all. That's the more trustworthy faithfulness signal of the two — it just
can't assess anything that isn't a number.
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


def judge_provider() -> tuple[str, bool]:
    """(judge model name, whether it comes from a different provider than the agent).

    Exposed so the evaluation report can state how independent the grade actually is.
    A same-family judge still catches plenty, but claiming independence it doesn't have
    would overstate the numbers.
    """
    if settings.anthropic_api_key:
        return settings.anthropic_judge_model, True
    if settings.openai_api_key:
        return settings.openai_judge_model, True
    return settings.judge_model, False


@lru_cache(maxsize=1)
def _judge_llm():
    """The judge model, preferring a different PROVIDER to the agent's.

    Anthropic or OpenAI is used whenever a key for either is set; otherwise this falls
    back to a different Gemini model. The fallback shares a model family with the agent
    and therefore some of its blind spots — which is why judge_provider() reports it.
    """
    if settings.anthropic_api_key:
        from langchain_anthropic import ChatAnthropic  # optional, judge-only dependency

        return ChatAnthropic(
            model=settings.anthropic_judge_model,
            api_key=settings.anthropic_api_key,
            temperature=0.0,
        ).with_structured_output(JudgeScores)

    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI  # optional, judge-only dependency

        return ChatOpenAI(
            model=settings.openai_judge_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        ).with_structured_output(JudgeScores)

    if not settings.gemini_api_key:
        raise RuntimeError(
            "No judge key set — need ANTHROPIC_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY."
        )
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
