"""Buy/sell trade-window classification.

Pure logic for the Season Hub advisor: given a team's playoff odds, roster-age
standing, and the weeks remaining before the league trade deadline, decide
whether the team should be buying (contender consolidating), selling
(rebuilder cashing vets), or holding, and pick the trade partners on the
opposite side of the market.
"""
from typing import List, Optional

# Playoff-odds thresholds for the verdict. Between them is a genuine coin
# flip where pushing someone to buy or sell would be false confidence.
BUY_THRESHOLD = 65.0
SELL_THRESHOLD = 35.0

# A deadline this close makes the verdict urgent.
URGENT_WEEKS = 3


def trade_window_verdict(
    playoff_pct: float,
    weeks_to_deadline: Optional[int] = None,
    age_rank: Optional[int] = None,
    n_teams: Optional[int] = None,
) -> dict:
    """Classify a team's trade posture.

    Args:
        playoff_pct: 0-100 playoff probability.
        weeks_to_deadline: whole weeks until the trade deadline; None when the
            league has no usable deadline.
        age_rank: 1 = oldest core in the league (optional flavor signal).
        n_teams: league size, required for age_rank to mean anything.

    Returns {"verdict": "buy"|"sell"|"hold", "urgent": bool, "modifier": str}.
    modifier is "" or a refinement: "all_in" (buying with an old core, the
    window is now), "youth" (selling with a young core, rebuild is on
    schedule), "aging_bubble" (holding with an old core on the playoff bubble,
    the riskiest place to sit).
    """
    pct = float(playoff_pct or 0.0)
    if pct >= BUY_THRESHOLD:
        verdict = "buy"
    elif pct <= SELL_THRESHOLD:
        verdict = "sell"
    else:
        verdict = "hold"

    urgent = weeks_to_deadline is not None and 0 <= int(weeks_to_deadline) <= URGENT_WEEKS

    modifier = ""
    if age_rank and n_teams and n_teams >= 4:
        old_third = age_rank <= max(1, round(n_teams / 3))
        young_third = age_rank > n_teams - max(1, round(n_teams / 3))
        if verdict == "buy" and old_third:
            modifier = "all_in"
        elif verdict == "sell" and young_third:
            modifier = "youth"
        elif verdict == "hold" and old_third:
            modifier = "aging_bubble"

    return {"verdict": verdict, "urgent": urgent, "modifier": modifier}


def trade_partners(teams: List[dict], verdict: str, limit: int = 3) -> List[str]:
    """Names of the best trade partners on the opposite side of the market.

    teams: [{"name", "playoff_pct", "is_viewer"}]. Buyers should call the
    clearest sellers (lowest playoff odds) and vice versa; holders get no
    partner list. The viewer is always excluded.
    """
    if verdict not in ("buy", "sell"):
        return []
    pool = [
        t for t in teams or []
        if not t.get("is_viewer") and t.get("name")
    ]
    if verdict == "buy":
        pool = [t for t in pool if float(t.get("playoff_pct") or 0) <= SELL_THRESHOLD]
        pool.sort(key=lambda t: float(t.get("playoff_pct") or 0))
    else:
        pool = [t for t in pool if float(t.get("playoff_pct") or 0) >= BUY_THRESHOLD]
        pool.sort(key=lambda t: -float(t.get("playoff_pct") or 0))
    return [str(t["name"]) for t in pool[: max(0, int(limit))]]
