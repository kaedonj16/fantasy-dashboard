"""Trade-calculator player value math shared by the server eval and the JS preview.

``SCORING_MULTS`` and ``player_trade_value`` must stay in lockstep with
``SCORING_MULTS`` / ``getPlayerValue`` in static/app.js. tests/test_scoring_mult_parity.py
and tests/test_trade_value_parity.py fail if they drift.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional


SCORING_MULTS = {
    "ppr": {"QB": 1.00, "RB": 1.00, "WR": 1.00, "TE": 1.00},
    "half": {"QB": 1.00, "RB": 1.06, "WR": 0.97, "TE": 0.94},
    "std": {"QB": 1.00, "RB": 1.13, "WR": 0.93, "TE": 0.87},
}


def player_trade_value(
    player: Mapping,
    *,
    league_type: str = "1qb",
    league_size: int = 10,
    scoring_format: str = "ppr",
    scoring_type: str = "dynasty",
    te_premium: float = 0.0,
) -> float:
    """Per-player value used by ``/api/trade-eval`` and the live trade preview."""
    fmt = (scoring_format or "ppr").strip().lower()
    scoring_mults = SCORING_MULTS.get(fmt, SCORING_MULTS["ppr"])
    lt = (league_type or "1qb").strip().lower()
    st = (scoring_type or "dynasty").strip().lower()
    try:
        size = int(league_size or 10)
    except (TypeError, ValueError):
        size = 10
    try:
        tep = float(te_premium or 0)
    except (TypeError, ValueError):
        tep = 0.0

    def _n(v) -> float:
        try:
            return float(v or 0)
        except (TypeError, ValueError):
            return 0.0

    if st == "redraft":
        if lt == "sf":
            val = _n(player.get("redraft_value_sf") or player.get("redraft_value_1qb"))
        else:
            val = _n(player.get("redraft_value_1qb"))
    elif lt == "sf":
        size_key = "sf_value" if size == 10 else f"sf_value_{size}"
        val = _n(player.get(size_key) or player.get("sf_value") or player.get("value"))
    else:
        size_key = "value" if size == 10 else f"value_{size}"
        val = _n(player.get(size_key) or player.get("value"))

    pos = str(player.get("position") or "").upper()
    mult = scoring_mults.get(pos, 1.0)
    if tep and pos == "TE":
        mult *= (1 + tep * 0.20)
    return math.floor(val * mult * 10 + 0.5) / 10


def fairness_label(net_delta: float) -> str:
    """Same ±150 bands Trade Outcome uses for now-vs-then net delta."""
    try:
        delta = float(net_delta or 0)
    except (TypeError, ValueError):
        delta = 0.0
    if delta > 150:
        return "strong_win"
    if delta < -150:
        return "strong_loss"
    return "fair"
