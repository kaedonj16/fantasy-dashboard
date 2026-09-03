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


def dr_rookie_team_score(pick_letters: "list[str]") -> Optional[float]:
    """Smooth 0-100 rookie team score: the MEAN of each pick's canonical letter
    score, instead of averaging letters into a single coarse team letter and then
    bucketing that back to a score.

    The old path (average letter -> letterToScore) snapped whole classes to a
    handful of values (A=87, B=70, ...) and rounded mixed classes up — an [A, B]
    class scored a full A (87). Averaging the per-pick canonical scores keeps the
    same anchors (an all-B class is still 70) but grades mixed classes on a
    continuous scale ([A, B] -> 78.5 -> B+), so real differences in a rookie haul
    read through. Per-pick letters still come from the BPA/ADP-diff grader.

    Mirrored in static/draft_room.js (rookie branch of gradePicks); keep the two
    in lock-step. Returns None when there are no gradeable picks.
    """
    scores = [dr_letter_to_score(L) for L in pick_letters if L and L != "N/A"]
    if not scores:
        return None
    return sum(scores) / len(scores)


def dr_slot_eligible(slot: str, pos: str) -> bool:
    pos = (pos or "").upper()
    if pos == "PK":
        pos = "K"
    if pos in ("D/ST", "DST", "D-ST"):
        pos = "DEF"
    if slot == "FLEX": return pos in ("RB", "WR", "TE")
    if slot == "RB_WR": return pos in ("RB", "WR")
    if slot == "WR_TE": return pos in ("WR", "TE")
    if slot == "RB_TE": return pos in ("RB", "TE")
    if slot == "SF":   return pos in ("QB", "RB", "WR", "TE")
    return slot == pos


_FLEX_COVERS = {
    "RB": {"FLEX", "RB_WR", "RB_TE"},
    "WR": {"FLEX", "RB_WR", "WR_TE"},
    "TE": {"FLEX", "WR_TE", "RB_TE"},
}


def _has_flex_for(slots, pos: str) -> bool:
    """True when this lineup has a flex slot that can start ``pos``."""
    from utils.lineup_slots import canonicalize_slot
    wanted = _FLEX_COVERS.get((pos or "").upper(), {"FLEX"})
    return any(canonicalize_slot(s) in wanted for s in (slots or []))


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
    flex = {"SF": 3, "FLEX": 2, "RB_WR": 1.5, "WR_TE": 1.5, "RB_TE": 1.5}
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


def dr_league_lineup_avg(
    players: "list[dict]", slots: "list[str]", num_teams: int, metric: str,
) -> Optional[float]:
    """Average a metric across a roster-valid league-wide starting field.

    A global ``top N`` baseline is position-blind: projected PPG, in particular,
    fills most of that imaginary field with quarterbacks. Build ``num_teams``
    copies of the real lineup instead, then optimize on the requested metric.
    Players missing that metric are excluded rather than counted as zero.

    Live grades prefer ``dr_peer_starter_avg`` (this draft's actual lineups).
    This helper remains the fallback when the caller has a player pool but no
    per-team pick lists (offline backtests, single-team unit tests).
    """
    eligible = []
    for i, player in enumerate(players or []):
        value = player.get(metric)
        if value is None:
            continue
        eligible.append({
            "id": f"league-{i}", "pos": player.get("pos"),
            "ppg": float(value),
        })
    if not eligible or not slots:
        return None
    league_slots = list(slots) * max(int(num_teams or 1), 1)
    selected = dr_optimal_lineup(eligible, league_slots)
    values = [p["ppg"] for p in eligible if str(p["id"]) in selected]
    return (sum(values) / len(values)) if values else None


def dr_starter_metric_avg(
    picks: "list[dict]", slots: "list[str]", metric: str,
) -> Optional[float]:
    """Average ``metric`` (``ppg`` or ``val``) across one team's optimal lineup."""
    if not picks or not slots:
        return None
    starter_ids = dr_optimal_lineup(picks, slots)
    values = []
    for p in picks:
        if str(p.get("id")) not in starter_ids:
            continue
        value = p.get(metric)
        if value is None:
            continue
        values.append(float(value))
    if not values:
        return None
    return sum(values) / len(values)


def dr_peer_starter_avg(
    teams: "list[list[dict]]", slots: "list[str]", metric: str,
) -> Optional[float]:
    """Mean of each team's starter-metric average. Skip teams with no values."""
    avgs = []
    for picks in teams or []:
        avg = dr_starter_metric_avg(picks, slots, metric)
        if avg is not None:
            avgs.append(avg)
    if not avgs:
        return None
    return sum(avgs) / len(avgs)


def dr_weighted_pick_score(
    picks: "list[dict]", slots: "list[str]", num_teams: int, *,
    sf: Optional[bool] = None, tep: float = 0.0,
) -> Optional[float]:
    """Role- and round-weighted pick-score average used by the Value bar."""
    if not picks:
        return None
    starter_ids = dr_optimal_lineup(picks, slots)
    is_sf = ("SF" in (slots or [])) if sf is None else bool(sf)
    bench_by_pos = {p: [] for p in ("QB", "RB", "WR", "TE")}
    for p in picks:
        pos = str(p.get("pos") or "").upper()
        if pos in bench_by_pos and str(p.get("id")) not in starter_ids:
            bench_by_pos[pos].append(p)

    def _lineup_score(p):
        return float(p.get("ppg") if p.get("ppg") is not None else (p.get("val") or 0) / 1000)

    for arr in bench_by_pos.values():
        arr.sort(key=_lineup_score, reverse=True)

    def _bench_utility(p):
        pos = str(p.get("pos") or "").upper()
        arr = bench_by_pos.get(pos, [])
        idx = arr.index(p) if p in arr else -1
        if pos == "QB":
            return (0.78 if is_sf else 0.32) if idx == 0 else (0.55 if is_sf else 0.12)
        if pos == "TE":
            return (0.72 if tep > 0 else 0.32) if idx == 0 else (0.48 if tep > 0 else 0.16)
        if pos == "RB":
            return 0.82 if idx == 0 else 0.68
        if pos == "WR":
            return 0.78 if idx == 0 else 0.64
        return 0.0

    def _role(p):
        if str(p.get("id")) in starter_ids:
            return "starter"
        pos = str(p.get("pos") or "").upper()
        arr = bench_by_pos.get(pos, [])
        idx = arr.index(p) if p in arr else -1
        if pos in ("RB", "WR"):
            if idx == 0:
                return "primary"
            if idx == 1 and _has_flex_for(slots, pos):
                return "primary"
            return "fringe"
        return "primary" if idx == 0 else "fringe"

    w_sum, w_tot = 0.0, 0.0
    for x in picks:
        if x.get("ps") is None or str(x.get("pos") or "").upper() in {"K", "DEF", "DST", "D/ST"}:
            continue
        rnd = max(1, math.ceil((x.get("pn") or 1) / max(int(num_teams or 1), 1)))
        role = _role(x)
        role_w = 1.0 if role == "starter" else 0.55 if role == "primary" else 0.18
        utility = 1.0 if role == "starter" else _bench_utility(x)
        wt = (1.0 / ((1 + (rnd - 1) / 5) ** 0.85)) * role_w * (0.55 + 0.45 * utility)
        w_sum += float(x["ps"]) * wt
        w_tot += wt
    if w_tot > 0:
        return w_sum / w_tot
    avg_ps_vals = [float(p["ps"]) for p in picks if p.get("ps") is not None]
    if not avg_ps_vals:
        return None
    return sum(avg_ps_vals) / len(avg_ps_vals)


def dr_peer_value_ps(
    teams: "list[list[dict]]", slots: "list[str]", num_teams: int, *,
    sf: Optional[bool] = None, tep: float = 0.0,
) -> Optional[float]:
    """Mean of each team's weighted pick-score average. Skip teams with no PS."""
    avgs = []
    for picks in teams or []:
        avg = dr_weighted_pick_score(picks, slots, num_teams, sf=sf, tep=tep)
        if avg is not None:
            avgs.append(avg)
    if not avgs:
        return None
    return sum(avgs) / len(avgs)


def dr_resolve_strength_baseline(
    slots: "list[str]", num_teams: int, metric: str, *,
    league_teams: Optional["list[list[dict]]"] = None,
    peer_avg: Optional[float] = None,
    league_players: Optional["list[dict]"] = None,
    league_list: Optional["list[float]"] = None,
) -> Optional[float]:
    """League-average starter baseline for the Starters bar.

    Prefer this draft's actual lineups (explicit peer average, then the mean
    of each team's optimal lineup). Fall back to a roster-valid field from the
    player pool, then a position-blind top-N list.
    """
    if peer_avg is not None and peer_avg > 0:
        return float(peer_avg)
    if league_teams:
        avg = dr_peer_starter_avg(league_teams, slots, metric)
        if avg is not None and avg > 0:
            return avg
    field = dr_league_lineup_avg(league_players, slots, num_teams, metric)
    if field is not None and field > 0:
        return field
    n_start = max(int(num_teams or 1), 1) * len(slots or [])
    top = dr_avg_top_n(league_list or [], n_start)
    return top if top > 0 else None


# Value / Starters / Construction point caps. Startup stays process-heavy (ADP
# value + conventional roster shape). Redraft is outcome-heavy: starting-lineup
# PPG is what the playoff-odds sim ranks teams on, and a 35/25/40 split let
# pick-score value invert that ranking (worst-graded team, 3rd-highest odds).
DR_SPLIT_STARTUP = (35.0, 25.0, 40.0)
DR_SPLIT_REDRAFT = (20.0, 50.0, 30.0)
# Construction mix: coverage / positional balance / extra-pick efficiency.
# Redraft leans on filled starting slots (empty slots score 0 in the odds sim);
# extra bench bodies are depth, not a grade penalty.
DR_CONSTRUCTION_STARTUP = (0.45, 0.30, 0.25)
DR_CONSTRUCTION_REDRAFT = (0.70, 0.20, 0.10)


def dr_grade_split(draft_type: str) -> tuple[float, float, float]:
    """Shipped Value/Starters/Construction caps for ``draft_type``."""
    return DR_SPLIT_REDRAFT if draft_type == "redraft" else DR_SPLIT_STARTUP


def dr_construction_mix(draft_type: str) -> tuple[float, float, float]:
    """Coverage / balance / efficiency weights for construction_raw."""
    return DR_CONSTRUCTION_REDRAFT if draft_type == "redraft" else DR_CONSTRUCTION_STARTUP


def dr_team_grade_score(
    picks: "list[dict]", *, slots: "list[str]", targets: dict, num_teams: int,
    draft_type: str, league_ppg_list: "list[float]", league_val_list: "list[float]",
    league_players: Optional["list[dict]"] = None,
    league_teams: Optional["list[list[dict]]"] = None,
    peer_starter_ppg: Optional[float] = None,
    peer_starter_val: Optional[float] = None,
    peer_value_ps: Optional[float] = None,
    value_weight: Optional[float] = None, starter_weight: Optional[float] = None,
    balance_weight: Optional[float] = None,
    sf: Optional[bool] = None, tep: float = 0.0,
) -> Optional[float]:
    """Mirror gradePicks() (startup/redraft branch) -> raw 0-100 composite.
    `picks` items: {id, pos, ps, pn, val, ppg}. Returns None if not gradeable.

    Value and Starters compare this team to this draft's teams when
    ``league_teams`` or ``peer_*`` is provided. The player-pool field and the
    absolute 0-100 pick-score scale are only fallbacks.

    value/starter/balance_weight are the point caps for the three components.
    ``None`` (the default) picks the shipped split for ``draft_type``:
    startup 35/25/40, redraft 20/50/30. The backtest overrides these to sweep
    further; the JS mirror uses the same per-type split, so parity holds when
    they're left as defaults."""
    if not picks:
        return None
    split_v, split_s, split_b = dr_grade_split(draft_type)
    if value_weight is None:
        value_weight = split_v
    if starter_weight is None:
        starter_weight = split_s
    if balance_weight is None:
        balance_weight = split_b
    starter_ids = dr_optimal_lineup(picks, slots)
    # Each starter occupies exactly one slot, so filled slots == starters chosen.
    coverage = (len(starter_ids) / len(slots)) if slots else 0.0
    is_sf = ("SF" in slots) if sf is None else bool(sf)
    bench_by_pos = {p: [] for p in ("QB", "RB", "WR", "TE")}
    for p in picks:
        pos = str(p.get("pos") or "").upper()
        if pos in bench_by_pos and str(p.get("id")) not in starter_ids:
            bench_by_pos[pos].append(p)
    def _lineup_score(p):
        return float(p.get("ppg") if p.get("ppg") is not None else (p.get("val") or 0) / 1000)
    for arr in bench_by_pos.values():
        arr.sort(key=_lineup_score, reverse=True)
    def _bench_utility(p):
        pos = str(p.get("pos") or "").upper()
        arr = bench_by_pos.get(pos, [])
        idx = arr.index(p) if p in arr else -1
        if pos == "QB": return (0.78 if is_sf else 0.32) if idx == 0 else (0.55 if is_sf else 0.12)
        if pos == "TE": return (0.72 if tep > 0 else 0.32) if idx == 0 else (0.48 if tep > 0 else 0.16)
        if pos == "RB": return 0.82 if idx == 0 else 0.68
        if pos == "WR": return 0.78 if idx == 0 else 0.64
        return 0.0
    def _role(p):
        if str(p.get("id")) in starter_ids:
            return "starter"
        pos = str(p.get("pos") or "").upper()
        arr = bench_by_pos.get(pos, [])
        idx = arr.index(p) if p in arr else -1
        # RB3/WR4 (first bench) are primary cover. A second RB/WR is still
        # primary when a flex that can start them exists — injury/bye path.
        if pos in ("RB", "WR"):
            if idx == 0:
                return "primary"
            if idx == 1 and _has_flex_for(slots, pos):
                return "primary"
            return "fringe"
        return "primary" if idx == 0 else "fringe"

    # 1) Pick-score value vs this league's average (same 80–120% band as Starters).
    starter_avg_ps = dr_weighted_pick_score(
        picks, slots, num_teams, sf=is_sf, tep=tep,
    )
    league_ps_avg = peer_value_ps
    if (league_ps_avg is None or league_ps_avg <= 0) and league_teams:
        league_ps_avg = dr_peer_value_ps(
            league_teams, slots, num_teams, sf=is_sf, tep=tep,
        )
    if starter_avg_ps is None:
        value_pts = math.floor(value_weight / 2)
    elif league_ps_avg is not None and league_ps_avg > 0:
        value_pts = math.floor(
            clamp01((starter_avg_ps / league_ps_avg - 0.80) / 0.40) * value_weight + 0.5
        )
    else:
        value_pts = math.floor(clamp01(starter_avg_ps / 100) * value_weight + 0.5)

    # 2) Starting-lineup strength vs this league's average starting lineup.
    starter_arr = [p for p in picks if str(p.get("id")) in starter_ids]
    my_ppgs = [p["ppg"] for p in starter_arr if p.get("ppg") is not None]
    ppg_ratio = None
    if len(my_ppgs) >= max(2, math.floor(len(starter_arr) * 0.5)):
        my_ppg_avg = sum(my_ppgs) / len(my_ppgs)
        league_ppg_avg = dr_resolve_strength_baseline(
            slots, num_teams, "ppg", league_teams=league_teams,
            peer_avg=peer_starter_ppg, league_players=league_players,
            league_list=league_ppg_list,
        )
        if league_ppg_avg is not None and league_ppg_avg > 0:
            ppg_ratio = my_ppg_avg / league_ppg_avg
            # Redraft playoff odds sum every starting slot (empty = 0). Scale
            # the filled-starter average by coverage so a finished stars-and-
            # scrubs roster with holes doesn't outrank a complete one on mean
            # PPG alone. Only apply once the team has had enough picks to fill
            # those slots — mid-draft every roster has holes, and raw coverage
            # (2/8 at the start of round 3) zeros the 50-pt starter term and
            # prints F for the whole league.
            if draft_type == "redraft" and slots and len(picks) >= len(slots):
                ppg_ratio *= coverage
    my_val_avg = (sum((p.get("val") or 0) for p in starter_arr) / len(starter_arr)) if starter_arr else 0.0
    league_val_avg = dr_resolve_strength_baseline(
        slots, num_teams, "val", league_teams=league_teams,
        peer_avg=peer_starter_val, league_players=league_players,
        league_list=league_val_list,
    )
    value_ratio = (my_val_avg / league_val_avg) if league_val_avg and league_val_avg > 0 else None
    if draft_type == "redraft":
        strength_ratio = ppg_ratio if ppg_ratio is not None else (value_ratio if value_ratio is not None else 0.80)
    else:
        if ppg_ratio is not None and value_ratio is not None:
            strength_ratio = 0.6 * ppg_ratio + 0.4 * value_ratio
        else:
            strength_ratio = ppg_ratio if ppg_ratio is not None else (value_ratio if value_ratio is not None else 0.80)
    starter_pts = math.floor(clamp01((strength_ratio - 0.80) / 0.40) * starter_weight + 0.5)

    # 3) Construction: coverage + functional cover + efficient bench use.
    counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
    for p in picks:
        pos = str(p.get("pos") or "").upper()
        if pos in counts:
            counts[pos] += 1
    bench = [p for p in picks if str(p.get("id")) not in starter_ids and str(p.get("pos") or "").upper() in counts]
    utility_vals = [_bench_utility(p) for p in bench]
    efficiency = sum(utility_vals) / len(utility_vals) if utility_vals else 1.0
    primary = [p for p in bench if _role(p) == "primary"]
    functional_depth = sum(_bench_utility(p) for p in primary) / len(primary) if primary else 0.0
    if draft_type == "redraft":
        construction_raw = clamp01(0.45 * coverage + 0.35 * functional_depth + 0.20 * efficiency)
    else:
        bsum = useful_picks = graded_picks = 0.0
        for pos in ("QB", "RB", "WR", "TE"):
            t = targets.get(pos, 0) or 0
            bsum += (min(counts[pos], t) / t) if t else 0.0
            useful_picks += min(counts[pos], t + 1)
            graded_picks += counts[pos]
        cov_w, bal_w, eff_w = dr_construction_mix(draft_type)
        construction_raw = clamp01(cov_w * coverage + bal_w * (bsum / 4) + eff_w * (useful_picks / graded_picks if graded_picks else 1.0))
    ramp = min(1.0, len(picks) / 8)
    balance_pts = math.floor(((1 - ramp) * 0.85 + ramp * construction_raw) * balance_weight + 0.5)

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
    # ANCHOR 74 -> 68, PTS 11 -> 9, from the letter-calibration backtest. At 74
    # the top THIRD of every league landed in A-range (~31% of teams) - too
    # generous for "A = elite". Anchoring the average at a B- reserves A-range for
    # ~the best 1 team per league (~10-15%); the best drafter still earns an A-.
    # PTS 9 keeps the spread modest to match the weak measured signal.
    ANCHOR, PTS = 68, 9
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
