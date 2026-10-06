"""
Thesis monitoring — saving a view and watching for what would break it.

The analyst already states what would change its mind; until now that was printed once
and forgotten. Saving it turns a one-off opinion into something that can be checked
later, which is how a view actually gets held: decide what would prove you wrong, then
watch only for that instead of re-reacting to every price move.

Conditions arrive in two forms. Measurable ones carry a metric, operator and threshold
and are evaluated deterministically here — no model call, so the answer is always the
same for the same numbers. Genuinely qualitative ones ("V4 ships without incident") are
kept as prose and surfaced for the reader to judge, never guessed at.
"""

import json
import uuid
from datetime import date, datetime

from config import ROOT_DIR

STORE = ROOT_DIR / "data" / "theses.json"

# A view built on 30-day data stops meaning much after a month or so.
STALE_AFTER_DAYS = 30

# Metric -> how it's labelled and formatted in the UI.
METRIC_LABELS = {
    "rsi": ("RSI(14)", "{:.0f}"),
    "price": ("Price", "${:,.2f}"),
    "revenue_growth_3m": ("Revenue growth (3m)", "{:+.1f}%"),
    "tvl_growth_3m": ("TVL growth (3m)", "{:+.1f}%"),
    "revenue_to_tvl": ("Revenue / TVL", "{:.2f}%"),
    "revenue_yoy": ("Revenue YoY", "{:+.1f}%"),
    "eps_yoy": ("EPS YoY", "{:+.1f}%"),
    "operating_margin": ("Operating margin", "{:.1f}%"),
    "net_margin": ("Net margin", "{:.1f}%"),
    # Flows are carried in millions so a threshold stays human-scale, in line with every
    # other metric here. Raw dollars would ask the model to emit -500000000, which is
    # easy to get wrong by an order of magnitude and impossible to eyeball on the card.
    "etf_flow_20d": ("US spot ETF net flow (20 sessions)", "{:+,.0f}M USD"),
    "etf_sessions_positive": ("ETF buying sessions (of 20)", "{:.0f}"),
}


def _load() -> list[dict]:
    if not STORE.exists():
        return []
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _save(rows: list[dict]) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def list_theses() -> list[dict]:
    """Saved theses, newest first."""
    return sorted(_load(), key=lambda t: t.get("formed", ""), reverse=True)


def save_thesis(symbol: str, kind: str, view) -> dict:
    """Store an analyst view as a monitorable thesis. Replaces any existing one."""

    def field(name, default=None):
        return view.get(name, default) if isinstance(view, dict) else getattr(view, name, default)

    def as_dict(t):
        return t if isinstance(t, dict) else t.model_dump()

    thesis = {
        "id": uuid.uuid4().hex[:8],
        "symbol": symbol.upper(),
        "kind": kind,
        "formed": date.today().isoformat(),
        "stance": field("stance", ""),
        "conviction": field("conviction", ""),
        "horizon": field("horizon", ""),
        "quality": field("quality", ""),
        "valuation": field("valuation", ""),
        "reasoning": list(field("reasoning", []) or []),
        "conditions": list(field("what_would_change_it", []) or []),
        "triggers": [as_dict(t) for t in (field("triggers", []) or [])],
        "key_risks": list(field("key_risks", []) or []),
    }
    rows = [t for t in _load() if t["symbol"] != thesis["symbol"]]
    rows.append(thesis)
    _save(rows)
    return thesis


def remove_thesis(thesis_id: str) -> None:
    _save([t for t in _load() if t.get("id") != thesis_id])


def age_days(thesis: dict) -> int | None:
    try:
        formed = datetime.fromisoformat(thesis["formed"]).date()
    except (KeyError, ValueError):
        return None
    return (date.today() - formed).days


def evaluate(thesis: dict, metrics: dict) -> dict:
    """Check a thesis's triggers against current metrics.

    A trigger fires when its condition is MET — the condition describes what would break
    the thesis, so a met condition means the view no longer holds. Metrics that can't be
    read are reported as unknown rather than quietly treated as fine.
    """
    checks = []
    for trig in thesis.get("triggers", []):
        metric = trig.get("metric")
        current = metrics.get(metric)
        label, fmt = METRIC_LABELS.get(metric, (metric, "{}"))

        if current is None:
            status = "unknown"
        else:
            op = trig.get("operator", "<")
            threshold = trig.get("threshold")
            try:
                status = "triggered" if (
                    current < threshold if op == "<" else current > threshold
                ) else "holding"
            except TypeError:
                status = "unknown"

        checks.append({
            "description": trig.get("description", ""),
            "metric": metric,
            "label": label,
            "operator": trig.get("operator", "<"),
            "threshold": trig.get("threshold"),
            "current": current,
            "current_text": fmt.format(current) if current is not None else "no data",
            "threshold_text": fmt.format(trig["threshold"]) if trig.get("threshold") is not None else "—",
            "status": status,
        })

    # Prose conditions with no measurable form still matter; they're for the reader.
    measurable = {t.get("description", "") for t in thesis.get("triggers", [])}
    manual = [c for c in thesis.get("conditions", []) if c not in measurable]

    triggered = [c for c in checks if c["status"] == "triggered"]
    holding = [c for c in checks if c["status"] == "holding"]
    age = age_days(thesis)

    # "Intact" has to mean something was actually verified. With no checks, or with
    # every check unreadable, the honest answer is that nothing has been confirmed —
    # claiming the thesis holds would be asserting more than the data supports.
    if triggered:
        status = "broken"
    elif holding:
        status = "intact"
    else:
        status = "unverified"

    return {
        "checks": checks,
        "manual": manual,
        "triggered": triggered,
        "holding": holding,
        "unknown": [c for c in checks if c["status"] == "unknown"],
        "age_days": age,
        "stale": age is not None and age > STALE_AFTER_DAYS,
        "status": status,
    }
