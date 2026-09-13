"""
Deterministic check: is every number in the report traceable to the evidence?

The LLM judge shares a model family with the agent it grades, so it shares that model's
blind spots, and a model asked "is this faithful?" can be talked into yes. Numbers are
the part of faithfulness that needs no judgement: a figure either appears in the
evidence or it doesn't. So this check is repeatable — same text, same score, no model,
no cost — and can't be argued into a good grade.

Repeatable is not the same as right, though: it is only as good as its parser. Its first
real run read "$500 by Friday" as $500 *billion*, because the "b" of "by" matched as a
scale suffix. The judge scored that report 1.0 and was correct; this check flagged it
and was not. The same bug was quietly turning "5 months" into five million. Abbreviated
scales now have to be attached to the number ("$1.94B").

Results are "unverified", never "hallucinated". A figure missing from the evidence may be
legitimately derived — a difference, a ratio — so each one comes back with the text
around it, for a person to read and decide. A bare 60.0 in a results file can't be
judged; "margins comfortably above 60%" can.

Known blind spot: matching is on magnitude only, because prose carries direction in words
("fell 31.7%") rather than signs. A report that turns a decline into a rise will still
trace. That error is the judge's to catch.
"""

import re

# Scale words and suffixes, so "$1.94B" in the evidence matches "1.94 billion" in prose.
SCALES = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "trillion": 1e12,
}

NUMBER = re.compile(
    r"""
    (?<![\w.])                                   # not the tail of a word or a decimal
    (?P<cur>\$)?\s?
    (?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)
    (?:
        \s*(?P<word>thousand|million|billion|trillion)\b     # "1.94 billion"
      | (?P<abbr>mn|bn|k|m|b|t)\b                            # "$1.94B" - attached only,
    )?                                                       # so "$500 by" isn't $500B
    (?P<unit>\s*%|x\b)?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Unmarked whole numbers are usually prose rather than data — "3 of the last 4
# quarters", "top 5", "in 2026" — and counting them buries the real findings. A currency
# sign, scale or unit marks a figure as data however small it is ("$50").
MIN_BARE_INTEGER = 100
YEARS = range(1900, 2101)


def _scan(text: str):
    """Yield (value, start, end) for each numeric claim, normalised to one magnitude."""
    for m in NUMBER.finditer(text or ""):
        raw = m.group("num")
        try:
            value = float(raw.replace(",", ""))
        except ValueError:
            continue
        scale = (m.group("word") or m.group("abbr") or "").lower()
        if scale:
            value *= SCALES[scale]
        marked = bool(scale or m.group("unit") or m.group("cur"))
        whole = "," not in raw and "." not in raw
        if not marked and whole and (value < MIN_BARE_INTEGER or int(value) in YEARS):
            continue
        yield value, m.start(), m.end()


def _values(text: str) -> list[float]:
    return [value for value, _, _ in _scan(text)]


def _context(text: str, start: int, end: int, radius: int = 45) -> str:
    """The text around a figure, so an unverified number can be read where it appeared."""
    lo, hi = max(0, start - radius), min(len(text), end + radius)
    snippet = " ".join(text[lo:hi].split())
    return ("…" if lo else "") + snippet + ("…" if hi < len(text) else "")


def _matches(target: float, pool: list[float], tolerance: float = 0.02) -> bool:
    """True when the evidence carries this figure, allowing for rounding in prose.

    Reports round ("34.4x" becomes "34x", "$1.94B" becomes "$1.9B"), so an exact
    comparison would flag good writing as unfaithful. 2% relative tolerance covers
    normal rounding without letting a genuinely different number through.
    """
    for candidate in pool:
        if candidate == target:
            return True
        scale = max(abs(target), abs(candidate))
        if scale and abs(candidate - target) / scale <= tolerance:
            return True
    return False


def check_numbers(report: str, evidence: list, analyst_view=None, question: str = "") -> dict:
    """Share of numeric claims in the output that appear in the evidence or the question.

    The question counts as a source: echoing the user's own "$500 target" back to them
    isn't a figure the agent made up.
    """
    pool = _values(question)
    for e in evidence or []:
        pool += _values(e.get("finding", "")) + _values(e.get("subquestion", ""))

    texts = [report or ""]
    if analyst_view is not None:
        def field(name):
            return (analyst_view.get(name) if isinstance(analyst_view, dict)
                    else getattr(analyst_view, name, None))
        texts += [field("quality") or "", field("valuation") or ""]
        texts += list(field("reasoning") or [])

    claims = [
        (value, text, start, end)
        for text in texts
        for value, start, end in _scan(text)
    ]
    if not claims:
        return {"numeric_traceability": None, "claims": 0,
                "unverified_count": 0, "unverified": []}

    unverified = [c for c in claims if not _matches(c[0], pool)]
    return {
        "numeric_traceability": round(1 - len(unverified) / len(claims), 3),
        "claims": len(claims),
        "unverified_count": len(unverified),
        # A sample with context is enough to investigate; the full list is noise.
        "unverified": [
            {"value": round(value, 4), "context": _context(text, start, end)}
            for value, text, start, end in unverified[:8]
        ],
    }
