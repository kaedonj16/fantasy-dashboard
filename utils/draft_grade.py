"""Pure draft-grade scoring helpers.

Extracted from app.py so the (Python-side mirror of the client's) draft-grade
math can be unit-tested without the pandas/DB stack. Every function here is pure
— no IO, no globals beyond stdlib ``math``.

``clamp01`` is a generic 0..1 clamp used across app.py; it lives here (its
heaviest caller) and is re-imported into app.py under its original name.
"""
from __future__ import annotations

import math
from typing import Optional


def clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def dr_grade_letter(s: float) -> str:
    """Mirror gradeLetter(): 0-100 score -> letter with +/- bands."""
    if s >= 90: return "A+"
    if s >= 85: return "A"
    if s >= 80: return "A-"
    if s >= 75: return "B+"
    if s >= 70: return "B"
    if s >= 65: return "B-"
    if s >= 60: return "C+"
    if s >= 55: return "C"
    if s >= 50: return "C-"
    if s >= 40: return "D"
    return "F"


def dr_letter_to_score(letter: str) -> int:
    """Mirror letterToScore(): canonical 0-100 for a coarse team letter (rookie)."""
    return {"A+": 92, "A": 87, "B": 70, "C": 55, "D": 43, "F": 20, "N/A": 55}.get(letter, 55)


def dr_slot_eligible(slot: str, pos: str) -> bool:
    pos = (pos or "").upper()
    if slot == "FLEX": return pos in ("RB", "WR", "TE")
    if slot == "SF":   return pos in ("QB", "RB", "WR", "TE")
    return slot == pos


def dr_lineup_score(p: dict) -> float:
    """Mirror lineupScore(): projected PPG, else value scaled into a ppg-like range."""
    ppg = p.get("ppg")
    if ppg is not None:
        return float(ppg)
    v = p.get("val")
    return (float(v) if v is not None else 0.0) / 1000.0


def dr_optimal_lineup(players: "list[dict]", slots: "list[str]") -> "set[str]":
    """Mirror optimalLineup(): fill the most restrictive slots first with the
    highest-lineupScore eligible player. Returns the set of starter player ids."""
    flex = {"SF": 3, "FLEX": 2}
    order = sorted(
        [{"slot": s, "i": i} for i, s in enumerate(slots)],
        key=lambda o: (flex.get(o["slot"], 1), o["i"]),
    )
    used: set = set()
    starter_ids: set = set()
    for o in order:
        best, best_score = -1, float("-inf")
        for j, pl in enumerate(players):
            if j in used:
                continue
            if not dr_slot_eligible(o["slot"], str(pl.get("pos") or "")):
                continue
            sc = dr_lineup_score(pl)
            if sc > best_score:
                best_score, best = sc, j
        if best >= 0:
            used.add(best)
            starter_ids.add(str(players[best].get("id")))
    return starter_ids


def dr_avg_top_n(arr: "list[float]", n: int) -> float:
    if not arr or n <= 0:
        return 0.0
    s = sorted(arr, reverse=True)[:n]
    return sum(s) / len(s) if s else 0.0


def dr_team_grade_score(
    picks: "list[dict]", *, slots: "list[str]", targets: dict, num_teams: int,
    draft_type: str, league_ppg_list: "list[float]", league_val_list: "list[float]",
) -> Optional[float]:
    """Mirror gradePicks() (startup/redraft branch) -> raw 0-100 composite.
    `picks` items: {id, pos, ps, pn, val, ppg}. Returns None if not gradeable."""
    if not picks:
        return None
    starter_ids = dr_optimal_lineup(picks, slots)
    # Each starter occupies exactly one slot, so filled slots == starters chosen.
    coverage = (len(starter_ids) / len(slots)) if slots else 0.0

    # 1) Starter quality: round-weighted (1/round^0.60) avg PS of starters.
    w_sum, w_tot = 0.0, 0.0
    avg_ps_vals = [p["ps"] for p in picks if p.get("ps") is not None]
    avg_ps = (sum(avg_ps_vals) / len(avg_ps_vals)) if avg_ps_vals else None
    for x in picks:
        if str(x.get("id")) not in starter_ids or x.get("ps") is None:
            continue
        rnd = max(1, math.ceil((x.get("pn") or 1) / max(num_teams, 1)))
        wt = 1.0 / (rnd ** 0.60)
        w_sum += x["ps"] * wt
        w_tot += wt
    starter_avg_ps = (w_sum / w_tot) if w_tot > 0 else avg_ps
    # Half-up rounding (floor(x+0.5)) to match the JS shared composite exactly.
    value_pts = math.floor(clamp01((starter_avg_ps or 0) / 100) * 35 + 0.5) if starter_avg_ps is not None else 17

    # 2) Starting-lineup strength vs a league-average team.
    starter_arr = [p for p in picks if str(p.get("id")) in starter_ids]
    n_start = max(num_teams, 1) * len(slots)
    my_ppgs = [p["ppg"] for p in starter_arr if p.get("ppg") is not None]
    ppg_ratio = None
    if len(my_ppgs) >= max(2, math.floor(len(starter_arr) * 0.5)):
        my_ppg_avg = sum(my_ppgs) / len(my_ppgs)
        league_ppg_avg = dr_avg_top_n(league_ppg_list, n_start)
        if league_ppg_avg > 0:
            ppg_ratio = my_ppg_avg / league_ppg_avg
    my_val_avg = (sum((p.get("val") or 0) for p in starter_arr) / len(starter_arr)) if starter_arr else 0.0
    league_val_avg = dr_avg_top_n(league_val_list, n_start)
    value_ratio = (my_val_avg / league_val_avg) if league_val_avg > 0 else None
    if draft_type == "redraft":
        strength_ratio = ppg_ratio if ppg_ratio is not None else (value_ratio if value_ratio is not None else 0.80)
    else:
        if ppg_ratio is not None and value_ratio is not None:
            strength_ratio = 0.6 * ppg_ratio + 0.4 * value_ratio
        else:
            strength_ratio = ppg_ratio if ppg_ratio is not None else (value_ratio if value_ratio is not None else 0.80)
    starter_pts = math.floor(clamp01((strength_ratio - 0.80) / 0.40) * 35 + 0.5)

    # 3) Construction: coverage + balance + efficiency.
    counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
    for p in picks:
        pos = str(p.get("pos") or "").upper()
        if pos in counts:
            counts[pos] += 1
    bsum, useful_picks, graded_picks = 0.0, 0, 0
    for pos in ("QB", "RB", "WR", "TE"):
        t = targets.get(pos, 0) or 0
        bsum += (min(counts[pos], t) / t) if t else 0.0
        cap = t + 1
        useful_picks += min(counts[pos], cap)
        graded_picks += counts[pos]
    efficiency = (useful_picks / graded_picks) if graded_picks > 0 else 1.0
    construction_raw = clamp01(0.45 * coverage + 0.30 * (bsum / 4) + 0.25 * efficiency)
    ramp = min(1.0, len(picks) / 8)
    balance_pts = math.floor(((1 - ramp) * 0.85 + ramp * construction_raw) * 30 + 0.5)

    return float(value_pts + starter_pts + balance_pts)


def dr_apply_field_curve(scores: "list[float]", rounds_done: int = 99) -> "list[float]":
    """Mirror of static/draft_grade_curve.js `curveFieldScores`. This is a
    deliberate cross-runtime copy (browser draft room vs Python server); the two
    are pinned identical by tests/test_draft_grade_curve_parity.py, so any change
    to one fails CI until the other matches. Do not edit this without editing the
    JS (and vice versa).

    Curve raw composites against the field so real separation reads on a
    B-anchored scale. ``rounds_done`` drives early-draft damping; the Teams page
    only grades completed drafts, so it defaults to full spread. Needs >=3 teams.
    """
    n = len(scores)
    if n < 3:
        return list(scores)
    mean = sum(scores) / n
    variance = sum((s - mean) ** 2 for s in scores) / n
    eff_std = max(math.sqrt(variance), 8)
    # ANCHOR 74 -> 70, PTS 11 -> 9, both from the letter-calibration backtest.
    # At anchor 74 the field average was a B, so the top THIRD of every league
    # landed in A-range (measured ~31% of teams) - too generous for "A = elite".
    # Anchoring the average at a low B (70) reserves A-range for ~the best 1-2
    # teams per league (~15%). PTS 9 keeps the spread modest to match the weak
    # measured signal (grade separation only lightly predicts real success).
    ANCHOR, PTS = 70, 9
    ramp = max(0.0, min(1.0, (rounds_done or 0) / 6))
    pts_eff = PTS * (0.5 + 0.5 * ramp)
    out = []
    for raw in scores:
        z = (raw - mean) / eff_std
        curved = ANCHOR + z * pts_eff
        curved = min(curved, raw + 8)             # can't out-curve the raw composite
        if curved >= 85 and raw < 80:             # A band needs real raw quality
            curved = 84
        curved = max(0.0, min(100.0, curved))
        out.append(float(math.floor(curved + 0.5)))  # round-half-up, matching JS
    return out
