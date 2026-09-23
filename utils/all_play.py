"""All-play / luck analysis for standings.

Pure computation so it can be unit-tested without the app. Given each finalized
week's scores per team, computes:

  - all-play record: your W-L if you had played every other team every week
    (immune to who your actual opponent was), and the derived all-play win %.
  - expected wins: all_play_pct * games played -- how many wins your scoring
    "deserved" against an average schedule.
  - luck delta: actual wins minus expected wins. Positive = luckier than your
    scoring warranted; negative = unlucky.
  - expected seed: standings rank by all-play (1 = best), vs the real seed.

Ties within a week are split (0.5 win / 0.5 loss vs each equal-scoring team),
so all-play wins can be fractional.
"""
import math
from typing import Dict


def all_play_analysis(
    weekly_scores: Dict[int, Dict[str, float]],
    actual_wins: Dict[str, float],
) -> Dict[str, dict]:
    """
    Args:
        weekly_scores: {week: {team: score}} for finalized weeks only.
        actual_wins: {team: actual head-to-head wins so far} (ties count 0.5).

    Returns {team: {
        all_play_wins, all_play_losses, all_play_pct, games,
        expected_wins, actual_wins, luck_delta, expected_seed, actual_rank
    }} for every team seen. Empty dict when there are no weeks.
    """
    # Normalize defensively.  Callers normally do this while constructing the
    # weekly map, but this public helper should never compare NaN/inf (whose
    # ordering semantics would silently turn bad input into losses or ties).
    valid_weeks: Dict[int, Dict[str, float]] = {}
    for week, scores in (weekly_scores or {}).items():
        clean = {}
        for team, score in (scores or {}).items():
            try:
                value = float(score)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(value):
                clean[team] = value
        # All-play has no meaning without an opponent comparison.
        if len(clean) >= 2:
            valid_weeks[week] = clean

    # Collect the full team set across valid weeks (a team missing from one
    # week, e.g. a bye in odd leagues, is simply not scored that week).
    teams = set()
    for wk in valid_weeks.values():
        teams.update(wk.keys())
    if not teams:
        return {}

    ap_wins = {t: 0.0 for t in teams}
    ap_losses = {t: 0.0 for t in teams}
    weeks_played = {t: 0 for t in teams}

    for wk in valid_weeks.values():
        rows = list(wk.items())
        for t, s in rows:
            weeks_played[t] += 1
            for u, s2 in rows:
                if u == t:
                    continue
                if s > s2:
                    ap_wins[t] += 1.0
                elif s < s2:
                    ap_losses[t] += 1.0
                else:  # tie: split
                    ap_wins[t] += 0.5
                    ap_losses[t] += 0.5

    out: Dict[str, dict] = {}
    for t in teams:
        w, l = ap_wins[t], ap_losses[t]
        total = w + l
        pct = (w / total) if total > 0 else 0.0
        games = weeks_played[t]
        exp_w = pct * games
        try:
            act_w = float(actual_wins[t])
            if not math.isfinite(act_w):
                raise ValueError("non-finite actual wins")
        except (KeyError, TypeError, ValueError, OverflowError):
            act_w = None
        out[t] = {
            "all_play_wins": round(w, 1),
            "all_play_losses": round(l, 1),
            "all_play_pct": round(pct, 4),
            "games": games,
            "expected_wins": round(exp_w, 1),
            "actual_wins": act_w,
            "luck_delta": round(act_w - exp_w, 1) if act_w is not None else None,
        }

    # Expected seed: rank by all-play pct (desc), tie-broken by all-play wins.
    order = sorted(out.keys(), key=lambda t: (-out[t]["all_play_pct"], -out[t]["all_play_wins"]))
    for i, t in enumerate(order):
        out[t]["expected_seed"] = i + 1

    # Actual seed: rank by actual wins (desc). Only meaningful as a comparison
    # point; the caller usually already has the real standings order.
    order_actual = sorted(
        (t for t in out if out[t]["actual_wins"] is not None),
        key=lambda t: -out[t]["actual_wins"],
    )
    for i, t in enumerate(order_actual):
        out[t]["actual_rank"] = i + 1

    return out


def luck_label(luck_delta: float, threshold: float = 1.0) -> str:
    """'Lucky' / 'Unlucky' / '' from a luck delta, with a neutral dead zone."""
    if luck_delta >= threshold:
        return "Lucky"
    if luck_delta <= -threshold:
        return "Unlucky"
    return ""
