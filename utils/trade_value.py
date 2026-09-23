"""Trade-calculator player value math shared by the server eval and the JS preview.

``SCORING_MULTS`` and ``player_trade_value`` must stay in lockstep with
``SCORING_MULTS`` / ``getPlayerValue`` in static/app.js. tests/test_scoring_mult_parity.py
and tests/test_trade_value_parity.py fail if they drift.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

SUPPORTED_LEAGUE_SIZES = (8, 10, 12, 14)


def snap_league_size(n) -> int:
    """Nearest supported value-table size (8 / 10 / 12 / 14)."""
    try:
        size = int(n or 10)
    except (TypeError, ValueError):
        return 10
    if size in SUPPORTED_LEAGUE_SIZES:
        return size
    return min(SUPPORTED_LEAGUE_SIZES, key=lambda s: abs(s - size))


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
    size = snap_league_size(league_size)
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
        # Redraft is size-invariant. The 10-team base columns are the
        # FantasyCalc-ratio board the player modal shows. Size-bucketed
        # redraft_*_{8,12,14} columns are a WLS overlay and can invert
        # Superflex (elite QBs priced like 1QB). League-size controls are
        # disabled in redraft for the same reason. Ranked surfaces
        # (My Leagues / Teams) pin league_size=10 so they match the modal.
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


def fair_value_band(baseline: float, floor: float = 25.0) -> float:
    """Shared "fair trade" band: the max value delta still called fair.

    Continuous in the baseline (the larger side's total value): a flat 7%
    with a ``floor`` minimum. This replaced the old tiered 5%/7%/10% bands,
    which had a discontinuity at the 600 threshold (band 41.9 at baseline
    599, dropping to 30.0 at 600) so near-identical trades flipped verdicts.

    Used by the trade calculator (api_trade_eval) and the trade outcome
    analyzer (api_trade_outcome) so both surfaces apply the same definition
    of fair and can't give contradictory verdicts.
    """
    try:
        b = max(float(baseline or 0.0), 1.0)
    except (TypeError, ValueError):
        b = 1.0
    try:
        f = max(float(floor or 0.0), 0.0)
    except (TypeError, ValueError):
        f = 25.0
    return max(b * 0.07, f)


def fairness_label(net_delta: float, baseline: Optional[float] = None) -> str:
    """Classify a trade's net value delta using the shared fair band.

    ``baseline`` is the larger side's total value; when omitted the band
    falls back to the 25.0 floor so the label still works for delta-only
    callers.
    """
    try:
        delta = float(net_delta or 0)
    except (TypeError, ValueError):
        delta = 0.0
    band = fair_value_band(baseline if baseline is not None else 0.0)
    if delta > band:
        return "strong_win"
    if delta < -band:
        return "strong_loss"
    return "fair"
