"""Approximate injury return planner (roadmap R07).

Never claims medical certainty. ESPN return dates are preferred; otherwise we
fall back to status-class duration bands from the waiver model.
"""
from __future__ import annotations

from typing import Any, Optional

from utils.waiver_score import INJURY_DURATION_WEEKS

# Dynasty value thresholds for stash vs drop (display values, not SF).
_VALUE_STASH = 80.0
_VALUE_HOLD = 40.0


def status_weeks_band(status: Optional[str]) -> Optional[float]:
    """Fixed duration band for a Sleeper/ESPN injury status class."""
    st = str(status or "").strip().upper()
    if not st or st in ("ACTIVE", "ACT", "HEALTHY", ""):
        return None
    # Common aliases
    if st in ("O",):
        st = "OUT"
    if st in ("D",):
        st = "DOUBTFUL"
    if st in ("Q", "GTD"):
        st = "QUESTIONABLE"
    if st in ("SUS", "SUSPENDED"):
        st = "SUSP"
    return INJURY_DURATION_WEEKS.get(st)


def resolve_weeks_out(
    *,
    status: Optional[str] = None,
    espn_weeks: Optional[float] = None,
) -> tuple[Optional[float], str]:
    """Return (weeks, source) with source ``espn`` | ``status`` | ``none``."""
    try:
        if espn_weeks is not None and float(espn_weeks) >= 0:
            return float(espn_weeks), "espn"
    except (TypeError, ValueError):
        pass
    band = status_weeks_band(status)
    if band is not None:
        return float(band), "status"
    return None, "none"


def injury_plan(
    *,
    status: Optional[str] = None,
    espn_weeks: Optional[float] = None,
    player_value: Optional[float] = None,
    has_open_ir_slot: bool = False,
) -> Optional[dict[str, Any]]:
    """Heuristic stash / drop / IR verdict for an injured player.

    Returns ``None`` when the player is not injured. Always sets
    ``approximate: True`` and hedging copy in ``reason``.
    """
    st = str(status or "").strip()
    weeks, source = resolve_weeks_out(status=st, espn_weeks=espn_weeks)
    if weeks is None and not st:
        return None
    if not st and weeks is None:
        return None
    # Healthy / no meaningful designation
    if st and st.upper() in ("ACTIVE", "ACT", "HEALTHY") and weeks is None:
        return None

    try:
        val = float(player_value) if player_value is not None else None
    except (TypeError, ValueError):
        val = None

    if weeks is not None and weeks <= 1.0:
        verdict = "Monitor"
        reason = "Listed return is soon (approx) — check inactive reports before lock."
    elif weeks is not None and weeks <= 3.0:
        if val is not None and val >= _VALUE_HOLD:
            verdict = "Stash"
            reason = "Short absence (~%.0f wk approx) and enough roster value to hold." % weeks
        else:
            verdict = "Drop candidate"
            reason = "Short absence but limited stash value — free the spot if you need it."
    elif weeks is not None and weeks <= 6.0:
        if has_open_ir_slot:
            verdict = "IR"
            reason = "Multi-week absence (~%.0f wk approx) — use an IR slot if available." % weeks
        elif val is not None and val >= _VALUE_STASH:
            verdict = "Stash"
            reason = "Longer absence (~%.0f wk approx) but high value — stash if you can." % weeks
        else:
            verdict = "Drop candidate"
            reason = "Longer absence (~%.0f wk approx) without IR room — lean drop unless deep bench." % weeks
    else:
        # IR / PUP / unknown long
        if has_open_ir_slot or (st and st.upper() in ("IR", "PUP", "NFI")):
            verdict = "IR"
            reason = "Extended absence (approx) — IR if the league allows; otherwise stash only if elite."
        elif val is not None and val >= _VALUE_STASH:
            verdict = "Stash"
            reason = "Extended absence (approx) but elite value — hold through the window if possible."
        else:
            verdict = "Drop candidate"
            reason = "Extended absence (approx) with limited stash value."

    if weeks is not None and weeks < 1:
        weeks_label = "~this week"
    elif weeks is not None:
        weeks_label = f"~{weeks:.0f} wk" if weeks >= 1 else f"~{weeks:.1f} wk"
    else:
        weeks_label = "unknown window"

    return {
        "verdict": verdict,
        "reason": reason,
        "weeks_out": weeks,
        "weeks_label": weeks_label,
        "source": source,
        "approximate": True,
        "status": st or None,
    }
