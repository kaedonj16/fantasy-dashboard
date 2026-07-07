"""Per-team season-in-review summary.

Pure computation (no pandas/app imports) so it unit-tests cleanly. Given one
team's finalized weekly rows plus a few league-context numbers, it derives the
headline facts for a "Season in Review" card: record, scoring, best/worst week,
longest win streak, and the luck read (all-play record, luck delta, expected vs
actual seed) that the all-play analysis already produced.
"""
from __future__ import annotations

from typing import Dict, List, Optional


def season_review(
    weekly: List[dict],
    all_play_entry: Optional[dict] = None,
    finish_rank: Optional[int] = None,
    num_teams: Optional[int] = None,
    pf_rank: Optional[int] = None,
) -> Dict:
    """
    Args:
        weekly: finalized weekly rows for ONE team, each {week, points, win},
            where win is 1 (win), 0 (loss) or 0.5 (tie).
        all_play_entry: this team's entry from all_play_analysis (optional).
        finish_rank: the team's actual standings rank (1 = first).
        num_teams: league size.
        pf_rank: the team's rank by points-for (1 = most points).

    Returns {} when there are no finalized weeks, else a dict of review facts.
    """
    rows = [w for w in (weekly or []) if w.get("points") is not None]
    games = len(rows)
    if games == 0:
        return {}

    pts = [float(w.get("points") or 0) for w in rows]
    wins = sum(1 for w in rows if float(w.get("win") or 0) >= 1)
    ties = sum(1 for w in rows if 0 < float(w.get("win") or 0) < 1)
    losses = games - wins - ties

    # Best / worst scoring week.
    best = max(rows, key=lambda w: float(w.get("points") or 0))
    worst = min(rows, key=lambda w: float(w.get("points") or 0))

    # Longest win streak across the season (in week order).
    ordered = sorted(rows, key=lambda w: int(w.get("week") or 0))
    longest = cur = 0
    for w in ordered:
        if float(w.get("win") or 0) >= 1:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0

    out: Dict = {
        "games": games,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "record": f"{wins}-{losses}" + (f"-{ties}" if ties else ""),
        "points_for": round(sum(pts), 1),
        "avg_points": round(sum(pts) / games, 1),
        "best_week": {"week": int(best.get("week") or 0), "points": round(float(best.get("points") or 0), 1)},
        "worst_week": {"week": int(worst.get("week") or 0), "points": round(float(worst.get("points") or 0), 1)},
        "longest_win_streak": longest,
        "finish_rank": finish_rank,
        "num_teams": num_teams,
        "pf_rank": pf_rank,
    }

    if all_play_entry:
        out["all_play_record"] = (
            f"{all_play_entry.get('all_play_wins', 0):.0f}-{all_play_entry.get('all_play_losses', 0):.0f}"
        )
        out["luck_delta"] = all_play_entry.get("luck_delta")
        out["expected_seed"] = all_play_entry.get("expected_seed")

    return out
