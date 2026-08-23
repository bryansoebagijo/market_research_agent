"""
Watchlist storage — the small bit of state the dashboard needs to remember.

Kept in a plain JSON file next to the project so a watchlist survives restarts without
introducing a database. Two lists: 'crypto' and 'stocks'.
"""

import json
from pathlib import Path

from config import ROOT_DIR

STORE = ROOT_DIR / "data" / "watchlists.json"
KINDS = ("crypto", "stocks")


def _load() -> dict:
    if not STORE.exists():
        return {k: [] for k in KINDS}
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {k: [] for k in KINDS}
    return {k: list(data.get(k, [])) for k in KINDS}


def _save(data: dict) -> None:
    STORE.parent.mkdir(parents=True, exist_ok=True)
    STORE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get(kind: str) -> list[str]:
    """The saved tickers for 'crypto' or 'stocks'."""
    return _load().get(kind, [])


def add(kind: str, symbol: str) -> list[str]:
    """Add a ticker (uppercased, no duplicates). Returns the updated list."""
    symbol = symbol.strip().upper()
    data = _load()
    if symbol and symbol not in data[kind]:
        data[kind].append(symbol)
        _save(data)
    return data[kind]


def remove(kind: str, symbol: str) -> list[str]:
    """Remove a ticker. Returns the updated list."""
    data = _load()
    data[kind] = [s for s in data[kind] if s != symbol.strip().upper()]
    _save(data)
    return data[kind]
