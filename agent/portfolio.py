"""
Positions and P&L — what turns a watchlist into a portfolio.

A watchlist says what you're interested in; a position says what you actually own, at
what price, and in what size. That difference is what lets the rest of the dashboard
answer the question a professional asks first each morning: how am I doing, and why.

Positions live in the same JSON store as the watchlists, under their own key, so the
two features stay independent and the file format stays readable.
"""

import json
import uuid
from datetime import date

from config import ROOT_DIR

STORE = ROOT_DIR / "data" / "positions.json"


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


def list_positions() -> list[dict]:
    """Every open position, newest first."""
    return sorted(_load(), key=lambda p: p.get("opened", ""), reverse=True)


def add_position(symbol: str, kind: str, quantity: float, entry_price: float,
                 opened: str | None = None, note: str = "") -> list[dict]:
    """Record a holding. `kind` is 'crypto' or 'stocks'."""
    rows = _load()
    rows.append({
        "id": uuid.uuid4().hex[:8],
        "symbol": symbol.strip().upper(),
        "kind": kind,
        "quantity": float(quantity),
        "entry_price": float(entry_price),
        "opened": opened or date.today().isoformat(),
        "note": note.strip(),
    })
    _save(rows)
    return rows


def remove_position(position_id: str) -> list[dict]:
    rows = [p for p in _load() if p.get("id") != position_id]
    _save(rows)
    return rows


# ---------------------------------------------------------------- P&L

def value_positions(positions: list[dict], prices: dict[str, float]) -> dict:
    """Turn positions plus current prices into P&L, weights and concentration.

    Weight is computed against total market value rather than cost, so it reflects the
    exposure you actually carry now — a winner that doubled is a bigger bet today than
    it was on the day you bought it.
    """
    rows: list[dict] = []
    total_value = 0.0
    total_cost = 0.0

    for p in positions:
        price = prices.get(p["symbol"].upper())
        cost = p["quantity"] * p["entry_price"]
        value = p["quantity"] * price if price is not None else None
        pnl = (value - cost) if value is not None else None

        rows.append({
            **p,
            "price": price,
            "cost": cost,
            "value": value,
            "pnl": pnl,
            "pnl_pct": (pnl / cost * 100) if (pnl is not None and cost) else None,
        })
        total_cost += cost
        if value is not None:
            total_value += value

    for r in rows:
        r["weight_pct"] = (r["value"] / total_value * 100) if (r["value"] and total_value) else None

    priced = [r for r in rows if r["value"] is not None]
    rows.sort(key=lambda r: r["value"] if r["value"] is not None else -1, reverse=True)

    total_pnl = total_value - sum(r["cost"] for r in priced) if priced else 0.0
    cost_of_priced = sum(r["cost"] for r in priced)

    return {
        "rows": rows,
        "total_value": total_value,
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "total_pnl_pct": (total_pnl / cost_of_priced * 100) if cost_of_priced else None,
        "unpriced": [r["symbol"] for r in rows if r["value"] is None],
    }


# A single position past this share of the book is worth flagging: it means the
# portfolio's outcome is mostly one decision, whatever the other holdings do.
CONCENTRATION_WARN_PCT = 35.0


def concentration_notes(valued: dict) -> list[str]:
    """Plain-language warnings about how the book is distributed."""
    notes: list[str] = []
    rows = [r for r in valued["rows"] if r.get("weight_pct")]
    if not rows:
        return notes

    top = max(rows, key=lambda r: r["weight_pct"])
    if top["weight_pct"] >= CONCENTRATION_WARN_PCT:
        notes.append(
            f"{top['symbol']} is {top['weight_pct']:.0f}% of the book — the portfolio's "
            f"outcome mostly rides on this one position."
        )

    by_kind: dict[str, float] = {}
    for r in rows:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + r["weight_pct"]
    for kind, share in by_kind.items():
        if share >= 80 and len(by_kind) > 1:
            notes.append(f"{share:.0f}% of the book is in {kind}.")

    if len(rows) < 3:
        notes.append(
            f"Only {len(rows)} priced position(s) — concentration and correlation "
            "figures are indicative at this size."
        )
    return notes
