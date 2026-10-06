"""Consolidated utils module: draft.

draft grades, pick scores/slots, keeper values

Merged from: utils/draft_grade.py, utils/pick_score.py, utils/pick_slots.py, utils/keeper_value.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# --- imports carried over from utils/utils.py ---
import glob
import json
import os
import re
import threading as _threading
import requests
import time
import traceback
import uuid
from contextlib import contextmanager as _contextmanager
from bs4 import BeautifulSoup
from collections import OrderedDict as _OrderedDict, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, Callable, List, Iterable, TYPE_CHECKING


# ======================================================================
# From utils/draft_grade.py
# ======================================================================

"""Pure draft-grade scoring helpers.

Extracted from app.py so the (Python-side mirror of the client's) draft-grade
math can be unit-tested without the pandas/DB stack. Every function here is pure
— no IO, no globals beyond stdlib ``math``.

``clamp01`` is a generic 0..1 clamp used across app.py; it lives here (its
heaviest caller) and is re-imported into app.py under its original name.
"""

import math


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
    from utils.lineups import canonicalize_slot
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


# ======================================================================
# From utils/pick_score.py
# ======================================================================

"""Pure draft pick-score computation.

Extracted from app.py so the pick-scoring engine can be unit-tested without the
pandas/DB stack. ``compute_pick_score`` blends DB dynasty value, ADP, tier,
positional need, youth, momentum and production into a 0-100 score, mirroring
the client-side Draft Room. Pure — reuses ``clamp01`` from utils.draft_grade.
"""



# Component weights per draft type (approximately normalized within each row).
# Rookie momentum (rank_change_7d) was down-weighted 0.06 -> 0.03 after a
# 509-team backtest (data_building/run_draft_backtest.py): cutting it raised the
# grade-vs-rookie-season correlation monotonically (r +0.220 -> +0.233), i.e. a
# 7-day ranking blip is noise for predicting a full season. The freed 0.03 went
# to value/adp, the levers that held up in the same sweep.
#
# Rookie & startup youth set to the validated 0.10 winner. Tuned modestly on 60
# leagues, then CONFIRMED at 10x scale on 600 leagues with multi-year outcomes:
#   startup multi-year (5,882 teams, base r +0.091): youth-0.10 #1 (+0.106), ppg/tier up
#   rookie  multi-year (9,725 teams, base r +0.082): youth-0.10 #1 (+0.095), tier/ppg up
# Both monotonic and replicated across independent samples. Youth double-counts
# (a young player already carries high dynasty value), while tier (scarcity) and
# ppg (production) predict multi-year success and were under-weighted. Rookie
# youth 0.16 -> 0.10 (freed mass to tier/ppg); startup youth 0.05 -> 0.10
# (taken from the 0.30 ADP term). Need weights are unchanged.
#
# Redraft's prior 0.30 explicit ADP weight also leaked ADP into 65% of the 0.24
# value component (~45.6% effective market influence). Value is now DB-only,
# explicit ADP is 0.18 early and decays by round, and the freed mass moves to
# VOR, production, tier, model value, and roster fit. The existing 1,573-team
# run established the direction (ADP down; PPG/tier up); CI parity tests pin the
# shipped JS/Python implementation. Re-running the external evaluation requires
# its league portfolio, DATABASE_URL, and completed-season outcome data.
PS_WEIGHTS = {
    "rookie":  {"vor": 0.06, "value": 0.20, "adp": 0.30, "tier": 0.18, "need": 0.05, "youth": 0.10, "mom": 0.03, "ppg": 0.13},
    "redraft": {"vor": 0.15, "value": 0.25, "adp": 0.18, "tier": 0.12, "need": 0.09, "youth": 0.00, "mom": 0.03, "ppg": 0.22},
    "startup": {"vor": 0.07, "value": 0.24, "adp": 0.25, "tier": 0.15, "need": 0.09, "youth": 0.10, "mom": 0.03, "ppg": 0.12},
}
PS_AGE_PEAKS = {"RB": 24, "WR": 27, "TE": 27, "QB": 29}


def ps_tier_of(value: float, thresholds: list):
    """Tier number for a dynasty value given gap-significance thresholds (1 = elite)."""
    if not thresholds:
        return None
    for i, thr in enumerate(thresholds):
        if value >= thr:
            return i + 1
    return len(thresholds) + 1


def starter_counts(counts: dict) -> dict:
    """Mirror of static/pick_score.js `starterCounts`; pinned by the parity test.
    Effective starters per position from roster slot counts (SF split half to QB,
    FLEX split half each to RB/WR), so the server's VOR/PPG replacement levels
    match the draft room's computeReplacement instead of a hardcoded guess."""
    c = counts or {}

    def n(k):
        try:
            return float(c.get(k) or 0)
        except (TypeError, ValueError):
            return 0.0

    return {
        "QB": n("QB") + n("SF") * 0.5,
        "RB": n("RB") + n("FLEX") * 0.5 + n("RB_WR") * 0.5 + n("RB_TE") * 0.5,
        "WR": n("WR") + n("FLEX") * 0.5 + n("RB_WR") * 0.5 + n("WR_TE") * 0.5,
        "TE": n("TE") + n("WR_TE") * 0.5 + n("RB_TE") * 0.5,
    }


def empirical_slot_allocation(players: list, slots: list, num_teams: int = 12,
                              metric: str = "value") -> dict:
    """Infer replacement demand by filling the league's actual starting field.

    FLEX/SF shares are outcomes, not fixed 50/50 assumptions: the best available
    eligible player fills each slot. Scarce dedicated slots are filled before
    flexible ones, and the result is returned as starters per fantasy team.
    """
    aliases = {
        "SUPER_FLEX": "SF", "SUPERFLEX": "SF", "SFLEX": "SF", "OP": "SF",
        "QB_RB_WR_TE": "SF", "Q_RB_WR_TE": "SF",
        "QB/RB/WR/TE": "SF", "QB/WR/RB/TE": "SF",
        "WRRBTE_FLEX": "FLEX", "RB_WR_TE": "FLEX",
        "RB/WR/TE": "FLEX", "WR/RB/TE": "FLEX", "W/R/T": "FLEX",
        "WRRB_FLEX": "RB_WR", "RB_WR_FLEX": "RB_WR", "RBWR_FLEX": "RB_WR",
        "RB/WR": "RB_WR", "WR/RB": "RB_WR", "W/R": "RB_WR",
        "REC_FLEX": "WR_TE", "WRTE_FLEX": "WR_TE",
        "WR/TE": "WR_TE", "W/T": "WR_TE",
        "RB/TE": "RB_TE", "R/T": "RB_TE",
    }
    normalized = [aliases.get(str(slot).upper(), str(slot).upper()) for slot in (slots or [])]
    if not normalized:
        normalized = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]
    eligibility = {
        "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"},
        "RB_WR": {"RB", "WR"}, "WR_TE": {"WR", "TE"}, "RB_TE": {"RB", "TE"},
        "FLEX": {"RB", "WR", "TE"}, "SF": {"QB", "RB", "WR", "TE"},
    }
    pool = []
    for i, player in enumerate(players or []):
        pos = str(player.get("position") or player.get("pos") or "").upper()
        try:
            score = float(player.get(metric) or 0)
        except (TypeError, ValueError):
            continue
        if pos in {"QB", "RB", "WR", "TE"}:
            pool.append((score, i, pos))
    # Dedicated slots first; SF last because it has the broadest eligibility.
    slot_order = sorted(normalized * max(1, int(num_teams or 1)),
                        key=lambda s: len(eligibility.get(s, set())))
    used, selected = set(), {p: 0 for p in ("QB", "RB", "WR", "TE")}
    for slot in slot_order:
        allowed = eligibility.get(slot, set())
        options = (item for item in pool if item[1] not in used and item[2] in allowed)
        best = max(options, default=None)
        if best is not None:
            used.add(best[1])
            selected[best[2]] += 1
    teams = max(1, int(num_teams or 1))
    return {pos: selected[pos] / teams for pos in selected}


def compute_pick_score(*, pos, value, vor, tier, age, rank_change_7d,
                       avg_pick, pick_no, max_val, draft_type, is_sf,
                       need_raw, qb_count, total_picks=None, num_teams=None,
                       ppg_norm=None, ppr=1.0, tep=0.0, pass_td=4.0, is_tier_cliff=False,
                       starter_slots=None,
                       weights=None,
                       depth_slope=None, depth_floor=None) -> int:
    """Mirror of static/pick_score.js `computePickScore`; the two are pinned
    identical by tests/test_pick_score_parity.py. This kernel is pure pick
    QUALITY only: live-draft timing (survival to the next pick, redraft
    handcuff insurance, late-round upside) lives entirely in the Draft Room's
    decision layer (static/draft_board_core.js ``decisionScore``), NOT here.
    Do not add wait/survive/handcuff/upside parameters. Do not edit this
    without editing the JS (and vice versa)."""
    pos = (pos or "").upper()
    # DB-sourced numbers arrive as decimal.Decimal; coerce to float so they mix
    # with the float weights below (Decimal * float raises TypeError).
    value = float(value) if value is not None else 0.0
    vor = float(vor) if vor is not None else None
    age = float(age) if age is not None else None
    rank_change_7d = float(rank_change_7d) if rank_change_7d is not None else None
    avg_pick = float(avg_pick) if avg_pick is not None else None
    max_val = float(max_val) if max_val is not None else 0.0
    need_raw = float(need_raw) if need_raw is not None else 0.0
    total_picks = float(total_picks) if total_picks is not None else 0.0
    db_value_norm = clamp01(value / max_val) if max_val and max_val > 0 else 0.0
    # Model/database value is independent of the explicit market component.
    # A near-zero value also caps selected-only ADP below.
    adp_untrusted = db_value_norm < 0.05
    value_norm = db_value_norm
    # Scarcity residual: share of this player's own value that sits above
    # replacement. Same-position players with similar value get nearly the same
    # residual (VOR no longer re-ranks Bijan vs Gibbs). Scarce positions — low
    # replacement relative to value — score higher. Do not divide VOR by the
    # global max value; that just copies value_norm.
    if vor is not None and value > 0:
        vor_norm = clamp01(max(vor, 0.0) / value)
    elif vor is not None:
        vor_norm = 0.0
    else:
        vor_norm = value_norm * 0.8

    # ADP component: proportional gap so a 2-pick fall from ADP 2 == a 10-pick
    # fall from ADP 20, with an elite-ADP floor for top-8 players.
    if avg_pick is not None:
        gap = pick_no - avg_pick
        rel = gap / max(avg_pick, 1.5)
        if rel >= 0.5:
            adp_val = 1.0
        elif rel >= -0.3:
            adp_val = 0.5 + rel
        else:
            adp_val = max(0.0, 0.2 + rel * 0.25)
        if avg_pick <= 8:
            adp_val = max(adp_val, clamp01(0.5 + (8 - avg_pick) / 16))
        if adp_untrusted:
            adp_val = min(adp_val, 0.5)
    else:
        adp_val = 0.5

    tier_score = clamp01((10 - min(tier, 9)) / 9) if tier else value_norm
    # Tier-cliff boost: position scarcity when this player's tier is drying up
    # (<=2 left in the bucket). Mirrors the Draft Room's isTierCliff() bump.
    if is_tier_cliff:
        tier_score = clamp01(tier_score + 0.15)

    # Need ramps in a touch earlier than before (/10 vs /12) so roster
    # construction starts to matter before the mid rounds, not only after.
    need_ramp = clamp01((pick_no - 1) / 10.0)
    need = (1 - need_ramp) * 0.5 + need_ramp * need_raw

    youth = 0.5
    if age is not None and pos in ("RB", "WR", "TE", "QB"):
        peak = PS_AGE_PEAKS.get(pos, 27)
        youth = clamp01((peak - age + 4) / 8)

    mom = clamp01((rank_change_7d or 0) / 20 + 0.5)

    # Production: position-normalized PPG. Missing data falls back to value_norm
    # so a player isn't penalized for absent projections (mirrors the Draft Room).
    ppg_n = ppg_norm if ppg_norm is not None else value_norm

    # ``weights`` lets the backtest harness sweep alternate weight tables
    # (data_building/draft_grade_backtest.py) without touching the shipped
    # PS_WEIGHTS. Defaults to the live table, so the parity test is unaffected.
    w = weights if weights is not None else PS_WEIGHTS.get(draft_type, PS_WEIGHTS["startup"])
    if draft_type == "redraft" and weights is None:
        teams0 = max(1, int(num_teams or 12))
        round0 = (int(pick_no) - 1) // teams0 + 1 if pick_no else 1
        adp_factor = max(0.15, 1.0 - max(0, round0 - 2) * 0.075)
        w = dict(w)
        freed = w["adp"] * (1.0 - adp_factor)
        base = sum(w[k] for k in ("vor", "value", "tier", "need", "ppg"))
        w["adp"] *= adp_factor
        for key in ("vor", "value", "tier", "need", "ppg"):
            w[key] += freed * w[key] / base
    s = (w["vor"] * vor_norm + w["value"] * value_norm + w["adp"] * adp_val
         + w["tier"] * tier_score + w["need"] * need + w["youth"] * youth
         + w["mom"] * mom + w.get("ppg", 0.0) * ppg_n)

    # QB overfill (1QB only): a second QB only carries real opportunity cost in
    # the early rounds. By the late rounds a backup QB is a normal pick, so the
    # penalty tapers out (mirrors the Draft Room).
    if not is_sf and pos == "QB":
        _teams = int(num_teams) if num_teams else 12
        _round = (int(pick_no) - 1) // max(_teams, 1) + 1 if pick_no else 1
        _qc = qb_count or 0
        _pen = 1.0
        if _qc >= 1:
            # Overfill: a 2nd+ QB in 1QB only carries real opportunity cost early;
            # by the late rounds a backup QB is a normal pick, so it tapers out.
            if _round <= 3:
                _pen = 0.30
            elif _round <= 6:
                _pen = 0.60
            elif _round <= 9:
                _pen = 0.85
            else:
                _pen = 1.0
            if _qc >= 2:
                _pen *= 0.7
        elif draft_type == "redraft":
            # First QB, redraft only: QB is streamable in 1QB, so a startable QB
            # shouldn't leapfrog clear starters / ADP-fallers in the early-mid
            # rounds. Tapers to none by the late rounds (grabbing a QB is fine
            # then). Dynasty/startup keep full QB value - young QBs are long-term
            # assets, not a streamable weekly slot.
            if _round <= 3:
                _pen = 0.82
            elif _round <= 6:
                _pen = 0.90
            elif _round <= 9:
                _pen = 0.97
            else:
                _pen = 1.0
        s *= _pen

    # Redundancy: a pick at a skill position already stocked to its realistic
    # depth target (need_raw == 0) is a bench body at a full spot while other
    # starting needs may remain - the opportunity cost the old score ignored
    # (it just dropped the need bonus). Penalize it, hardest at a true
    # single-starter slot (1-TE, or any position whose dedicated starter count
    # is <= 1) and in the early rounds; bench depth in the late rounds is
    # normal, so the penalty tapers out. QB overfill has its own rule above.
    # Need math itself is unchanged — only the single-vs-multi classification.
    if need_raw <= 0 and pos in ("RB", "WR", "TE"):
        _teams2 = int(num_teams) if num_teams else 12
        _rd2 = (int(pick_no) - 1) // max(_teams2, 1) + 1 if pick_no else 1
        if starter_slots is not None:
            try:
                _single = float(starter_slots) <= 1
            except (TypeError, ValueError):
                _single = pos == "TE"
        else:
            _single = pos == "TE"
        if _rd2 <= 3:
            _rp = 0.55 if _single else 0.82
        elif _rd2 <= 6:
            _rp = 0.72 if _single else 0.90
        elif _rd2 <= 9:
            _rp = 0.86 if _single else 0.96
        else:
            _rp = 1.0
        s *= _rp

    # Scoring-format adjustments: shift toward the build the league's scoring
    # rewards. Mirrors the Draft Room's scoringCfg() multipliers exactly.
    if tep and tep > 0 and pos == "TE":
        s *= (1 + 0.12 * tep)
    if pos == "QB" and pass_td is not None and float(pass_td) >= 6:
        s *= 1.06
    if pos in ("WR", "TE"):
        if ppr is not None and ppr >= 1:
            s *= 1.02
    elif pos == "RB" and ppr is not None and ppr <= 0:
        s *= 1.03

    # Depth normalization: re-anchor the 0-100 scale to what's achievable at this
    # pick slot so late-round picks aren't unfairly buried (mirrors the Draft Room).
    # slope/floor default to the shipped 0.44/0.40; the backtest can override them
    # (compute_pick_score is the sweep target) to tune the by-round flatness. A
    # diagnostic showed an at-ADP "par" pick declines ~35 pts across a draft, so
    # the by-round curve isn't flat — BUT the depth-slope sweep on 1,573 redraft
    # teams found 0.44 is the OPTIMAL slope (correlation-vs-outcome falls
    # monotonically as the slope steepens: .44 +0.150 -> .80 +0.119). The dip is
    # real signal (mid-round picks ARE lower value); flattening it only adds noise.
    # So 0.44 stays. Overrides are grading/analysis-only; the JS mirror uses the
    # shipped 0.44/0.40, so parity holds when they're left as defaults.
    if total_picks and total_picks > 1 and pick_no:
        _slope = 0.44 if depth_slope is None else float(depth_slope)
        _floor = 0.40 if depth_floor is None else float(depth_floor)
        _depth = min(0.98, (float(pick_no) - 1) / float(total_picks))
        _par = max(_floor, 1.0 - _depth * _slope)
        s = s / _par

    # Display relabel (monotonic): everything above is the backtested ranking and
    # is left untouched. This only stretches the near-ceiling band so the best
    # pick's 0-100 number differentiates instead of clustering at ~97. Scores
    # under the knee (~85) are unchanged; above it the curve is steeper than 1:1,
    # so a truly elite pick pulls toward 100 while a merely-good "best available"
    # reads lower. Monotonic => never changes ranking, only the label. Keep
    # identical to static/pick_score.js.
    d = clamp01(s)
    _knee = 0.85
    if d > _knee:
        _t = (d - _knee) / (1.0 - _knee)
        d = _knee + _t * _t * (1.0 - _knee)
    return int(math.floor(d * 100 + 0.5))  # round-half-up, matching JS


# ======================================================================
# From utils/pick_slots.py
# ======================================================================

"""Pure rookie-pick slot logic: bracket parsing, draft-order computation, and
pick display labels.

Extracted from app.py so the ordering rules can be unit-tested without the
pandas/DB stack. The draft order is the reverse of final overall standings:
non-playoff teams first (worst regular-season record gets slot 1), then
playoff teams ordered by playoff finish (earliest eliminated first, champion
last).
"""
from typing import Set, Tuple


def placements_from_bracket(winners_bracket: list) -> Tuple[Set[int], Dict[int, int]]:
    """Parse a Sleeper winners bracket into playoff participation and finish.

    Returns (playoff_roster_ids, {roster_id: final_placement}). Sleeper sets
    "p" on the decisive matchup for each placement: the winner takes placement
    p, the loser p + 1. Roster ids appear as direct integers in the t1/t2/w/l
    fields; anything else (TBD references like {"w": ...} dicts) is ignored.
    """
    playoff_rids: Set[int] = set()
    placements: Dict[int, int] = {}
    for m in winners_bracket or []:
        if not isinstance(m, dict):
            continue
        for key in ("t1", "t2", "w", "l"):
            v = m.get(key)
            if isinstance(v, int) and v > 0:
                playoff_rids.add(v)
        p = m.get("p")
        if p is None:
            continue
        try:
            p = int(p)
        except (TypeError, ValueError):
            continue
        w = m.get("w")
        l = m.get("l")
        if isinstance(w, int) and w > 0:
            placements[w] = p
        if isinstance(l, int) and l > 0:
            placements[l] = p + 1
    return playoff_rids, placements


def compute_pick_slots(
    reg_ranks: Dict[int, int],
    playoff_rids: Set[int],
    playoff_placements: Dict[int, int],
) -> Dict[int, int]:
    """Rookie draft slots from regular-season ranks and playoff finishes.

    Non-playoff teams get slots 1..k ordered worst-to-best regular season;
    playoff teams get the remaining slots ordered worst-to-best playoff finish
    (champion picks last). Returns {} when there are no playoff placements to
    anchor the order (callers fall back to regular-season-only slots).
    """
    if not playoff_placements:
        return {}

    non_playoff = sorted(
        ((rid, rank) for rid, rank in reg_ranks.items() if rid not in playoff_rids),
        key=lambda x: x[1],  # highest rank number = worst record
        reverse=True,
    )
    playoff_ordered = sorted(
        playoff_placements.items(),
        key=lambda x: x[1],  # highest placement number = worst finish
        reverse=True,
    )

    slot_map: Dict[int, int] = {}
    slot = 1
    for rid, _ in non_playoff:
        slot_map[rid] = slot
        slot += 1
    for rid, _ in playoff_ordered:
        slot_map[rid] = slot
        slot += 1
    return slot_map


def slots_from_regular_season(reg_ranks: Dict[int, int], total_teams: Optional[int] = None) -> Dict[int, int]:
    """Fallback draft order from regular-season standings only: the worst
    record (highest rank number) picks first."""
    total = total_teams or len(reg_ranks)
    return {rid: total - rank + 1 for rid, rank in reg_ranks.items()}


def pick_label(year: int, rnd: int, exact_slot: Optional[int] = None) -> str:
    """Display label for a rookie pick: "2026 1.03" when the exact slot is
    known, "2026 1st (Mid)" otherwise, "Pick" when year/round are missing."""
    if not year or not rnd:
        return "Pick"
    if exact_slot is not None:
        return f"{year} {rnd}.{exact_slot:02d}"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")
    return f"{year} {rnd}{suffix} (Mid)"


def avg_pick_value_for_round(by_id: dict, season: int, rnd: int) -> float:
    """Average model value of all picks matching season + round prefix.
    Pick value keys look like "2026_1_03"."""
    prefix = f"{season}_{rnd}_"
    vals = [v for k, v in by_id.items() if k.startswith(prefix)]
    return (sum(vals) / len(vals)) if vals else 0.0




def pick_value_from_table(tbl: dict, year: int, rnd: int, slot: Optional[int] = None,
                          num_teams: Optional[int] = None) -> float:
    """Resolve a pick's value from a pick-value table with graceful fallback.

    Lookup order: exact slot key ("2027_1_03") -> slot bucket key
    ("2027_1_early") -> average of all slot keys in the round -> bare round
    key ("2027_1") -> 0.0. Mirrors the hierarchy used by the trade-intel
    resolver so a pick is never silently collapsed to a flat number when
    finer-grained data exists.
    """
    tbl = tbl or {}

    def _pos(v) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return v if v > 0 else 0.0

    if slot:
        v = _pos(tbl.get(f"{year}_{rnd}_{int(slot):02d}") or tbl.get(f"{year}_{rnd}_{int(slot)}"))
        if v:
            return v
        if num_teams:
            v = _pos(tbl.get(f"{year}_{rnd}_{bucket_for_slot(int(slot), int(num_teams))}"))
            if v:
                return v
    slot_vals = [
        _pos(v) for k, v in tbl.items()
        if k.startswith(f"{year}_{rnd}_") and k.split("_")[-1].isdigit() and _pos(v)
    ]
    if slot_vals:
        return round(sum(slot_vals) / len(slot_vals), 1)
    v = _pos(tbl.get(f"{year}_{rnd}_mid"))
    if v:
        return v
    return _pos(tbl.get(f"{year}_{rnd}"))


def is_pick_asset_id(asset_id) -> bool:
    """A draft-pick asset id looks like '2026_1_01' or '2026_1_early'
    (year_round_slotOrBucket). Player ids are bare numeric Sleeper ids."""
    parts = str(asset_id or "").split("_")
    if len(parts) < 2:
        return False
    yr = parts[0]
    return len(yr) == 4 and yr.isdigit() and parts[1].isdigit()


def parse_pick_asset(pick_id) -> Optional[dict]:
    """Parse a pick asset id into its parts plus a display name.

    Returns {"season", "round", "slot" (int or None), "slot_raw" (the third
    segment verbatim, e.g. "01" or "early"), "bucket" ("Early"/"Mid"/"Late" or
    None), "name"} or None when the id is not a recognizable pick.
    Names: "2026 1.03" (exact slot), "2026 1st (Early)" (bucket),
    "2026 1st" (round only).
    """
    if not is_pick_asset_id(pick_id):
        return None
    parts = str(pick_id).split("_")
    try:
        yr = int(parts[0])
        rnd = int(parts[1])
    except (ValueError, IndexError):
        return None
    third = parts[2] if len(parts) >= 3 else ""
    sfx = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")
    bkt = {"early": "Early", "mid": "Mid", "late": "Late"}.get(third.lower())
    if bkt:
        name = f"{yr} {rnd}{sfx} ({bkt})"
    elif third.isdigit():
        name = f"{yr} {rnd}.{int(third):02d}"
    else:
        name = f"{yr} {rnd}{sfx}"
    return {
        "season": yr,
        "round": rnd,
        "slot": int(third) if third.isdigit() else None,
        "slot_raw": third,
        "bucket": bkt,
        "name": name,
    }


# ======================================================================
# From utils/keeper_value.py
# ======================================================================

"""Keeper-league decision math.

Pure, dependency-free scoring for "who should I keep?" in a keeper league. The
page layer (dashboard_services/pages/keeper_page.py) feeds this real roster,
draft, ADP and value data; everything here is deterministic and unit-tested so
the numbers a manager acts on are trustworthy.

Core idea — **surplus**:

    A keeper is worth it when the pick it costs is *later* than where the player
    would be drafted on the open market. Surplus is that gap, in rounds:

        surplus = keeper_cost_round - market_round

    A player who drafts in round 2 (market) but only costs a round-10 keeper
    slot is +8 rounds of surplus — a slam-dunk keep. A player who costs a round
    earlier than his market price is negative surplus — let him go back in the
    draft and take him (or someone better) there.

Keeper cost is league-configurable (see ``KeeperRules``); market round comes
from redraft ADP. Recommendations use a diminishing pick-value curve so equal
round savings at the top and bottom of the draft are not treated as equivalent.
"""

from dataclasses import dataclass
from math import ceil, sqrt
from typing import Sequence

# Verdict tiers, most-to-least keepable.
KEEP = "keep"
TOSS = "toss"
PASS = "pass"


@dataclass(frozen=True)
class KeeperRules:
    """A league's keeper cost rules.

    league_size:     teams in the league (rounds ≈ overall_pick / league_size).
    num_rounds:      draft rounds; cost is clamped into [1, num_rounds].
    round_offset:    shift applied to the drafted round. 0 = keep at the round
                     drafted; -1 = one round *earlier* (more expensive); +1 =
                     one round later (cheaper).
    escalation:      rounds the cost climbs (gets earlier / more expensive) for
                     each year the player has already been kept.
    undrafted_round: keeper cost for a player who wasn't drafted (waiver/FA add).
                     Defaults to the last round. Ignored when ``last_round_cost``
                     is True (every keeper costs the last round).
    last_round_cost: when True, every keeper costs the last draft round
                     regardless of where they were drafted last year. Common in
                     leagues that spend "your last pick" instead of the prior
                     drafted round. Escalation still applies (multi-year keeps
                     move earlier).
    keep_at / pass_at: surplus thresholds for the KEEP / PASS verdict tiers.
    one_per_round:   when True, no two kept players may share a cost round (the
                     common "you only own one pick per round" rule); the optimizer
                     bumps duplicates to the nearest open round and re-prices, so
                     the surplus total reflects the real, legal cost.
    """
    league_size: int = 12
    num_rounds: int = 15
    round_offset: int = 0
    escalation: int = 1
    undrafted_round: Optional[int] = None
    last_round_cost: bool = False
    keep_at: int = 2      # surplus >= keep_at  -> KEEP
    pass_at: int = 0      # surplus <  pass_at  -> PASS  (between the two -> TOSS)
    one_per_round: bool = False


def market_round(adp_overall: Optional[float], league_size: int) -> Optional[int]:
    """Round a player is expected to be drafted in, from his overall redraft ADP.

    ``adp_overall`` is a 1-based overall pick/rank (1 = the consensus #1 pick).
    Returns None when ADP is unknown (player off the draftable board)."""
    if not adp_overall or adp_overall <= 0 or league_size <= 0:
        return None
    return ceil(float(adp_overall) / league_size)


def adjust_adp_for_keepers(
    adp: Optional[float],
    kept_adps: Sequence[Optional[float]],
) -> Optional[float]:
    """Compress ADP after keepers leave the draft pool.

    Redraft ADP assumes the full player pool. When keepers are set, those
    players are gone — everyone behind them slides up. A player with ADP 24
    and 10 keepers at ADP ≤ 24 is effectively the 14th player available.

    ``kept_adps`` should be the raw ADPs of *other* kept players (do not include
    the player being adjusted, or he will subtract himself). Keepers with
    unknown ADP are ignored. Returns None when ``adp`` is None/invalid.
    """
    if adp is None:
        return None
    try:
        raw = float(adp)
    except (TypeError, ValueError):
        return None
    if raw <= 0:
        return None
    n = 0
    for k in kept_adps or ():
        if k is None:
            continue
        try:
            kv = float(k)
        except (TypeError, ValueError):
            continue
        if kv > 0 and kv <= raw:
            n += 1
    return max(1.0, raw - n)


def pick_value(overall_pick: float) -> float:
    """Relative value of a draft selection on a diminishing-return curve.

    The exact unit is intentionally abstract; only differences are compared.
    Unlike raw rounds, this recognizes that moving from pick 30 to pick 6 is
    much more valuable than moving from pick 174 to pick 150.
    """
    return 100.0 / sqrt(max(float(overall_pick), 1.0))


def keeper_surplus_value(adp_overall: Optional[float], cost_round: int,
                         league_size: int) -> Optional[float]:
    """Value gained by keeping a player at ``cost_round`` instead of ADP.

    A round cost is represented by its midpoint because leagues do not expose
    the manager's future slot when keeper decisions are made.
    """
    if not adp_overall or adp_overall <= 0 or league_size <= 0:
        return None
    cost_pick = (max(int(cost_round), 1) - 0.5) * league_size + 0.5
    return pick_value(adp_overall) - pick_value(cost_pick)


def keeper_cost_round(
    drafted_round: Optional[int],
    years_kept: int,
    rules: KeeperRules,
) -> int:
    """The draft round it costs to keep this player next season.

    ``drafted_round`` is the round he was drafted (None = undrafted / waiver add,
    which costs ``rules.undrafted_round`` or the last round). When
    ``rules.last_round_cost`` is set, every keeper starts at the last round
    instead (drafted round / undrafted override ignored). Escalation makes a
    long-held keeper progressively more expensive (an earlier round). Always
    clamped into a real round [1, num_rounds]."""
    last = max(1, int(rules.num_rounds))
    if rules.last_round_cost:
        base = last
    elif drafted_round is None:
        base = rules.undrafted_round if rules.undrafted_round is not None else last
    else:
        base = int(drafted_round) + int(rules.round_offset)
    cost = base - max(0, int(years_kept)) * int(rules.escalation)
    return max(1, min(last, cost))


def verdict(surplus: Optional[int], rules: KeeperRules) -> str:
    """KEEP / TOSS / PASS from a surplus (rounds gained)."""
    if surplus is None:
        return PASS
    if surplus >= rules.keep_at:
        return KEEP
    if surplus < rules.pass_at:
        return PASS
    return TOSS


@dataclass
class KeeperCandidate:
    """One rostered player evaluated as a keeper."""
    player_id: str
    name: str
    position: str
    drafted_round: Optional[int]      # None = undrafted / waiver add
    years_kept: int
    adp_overall: Optional[float]      # redraft ADP (overall rank), None if off-board
    value: float = 0.0                # redraft value, for tie-breaks / display
    # ── derived (filled by analyze) ───────────────────────────────────────
    cost_round: int = 0
    market_round: Optional[int] = None
    surplus: Optional[int] = None
    surplus_value: Optional[float] = None  # nonlinear value of the picks saved
    verdict: str = PASS
    keep: bool = False                # chosen by the optimizer


def analyze(candidate: KeeperCandidate, rules: KeeperRules) -> KeeperCandidate:
    """Fill a candidate's cost, market round, surplus and verdict in place."""
    candidate.cost_round = keeper_cost_round(
        candidate.drafted_round, candidate.years_kept, rules
    )
    candidate.market_round = market_round(candidate.adp_overall, rules.league_size)
    if candidate.market_round is None:
        # No market price: treat as no surplus (undraftable player).
        candidate.surplus = None
    else:
        candidate.surplus = candidate.cost_round - candidate.market_round
    candidate.surplus_value = keeper_surplus_value(
        candidate.adp_overall, candidate.cost_round, rules.league_size
    )
    candidate.verdict = verdict(candidate.surplus, rules)
    return candidate


def _sort_key(c: KeeperCandidate):
    # Rank by the value of picks saved, not raw round count. Early-round gains
    # are materially more valuable than identical gains in the late rounds.
    eligible = c.surplus is not None and c.surplus > 0
    s = c.surplus_value if c.surplus_value is not None else -9999
    return (not eligible, -s, -(c.value or 0.0))


def _optimize_unique_rounds(candidates: Sequence[KeeperCandidate], rules: KeeperRules,
                            limit: int) -> None:
    """Jointly choose keepers and unique cost rounds via dynamic programming.

    A duplicate keeper may move to an earlier (more expensive) unused round.
    Considering selection and assignment together avoids the old failure mode
    where a post-selection bump made a chosen keeper worse than an omitted one.
    """
    # (chosen count, occupied-round bitmask) -> (score, [(candidate index, round)])
    states = {(0, 0): (0.0, [])}
    for idx, candidate in enumerate(candidates):
        if candidate.surplus is None or candidate.surplus <= 0:
            continue
        updated = dict(states)
        for (count, mask), (score, assignments) in states.items():
            if count >= limit:
                continue
            # A collision can only make a keeper costlier, never award a later
            # pick. Consider every legal earlier round so the global optimum is
            # not dependent on candidate iteration order.
            for assigned_round in range(candidate.cost_round, 0, -1):
                bit = 1 << (assigned_round - 1)
                if mask & bit:
                    continue
                gain = keeper_surplus_value(
                    candidate.adp_overall, assigned_round, rules.league_size
                )
                if gain is None or gain <= 0:
                    continue
                key = (count + 1, mask | bit)
                proposed = score + gain
                if key not in updated or proposed > updated[key][0]:
                    updated[key] = (proposed, assignments + [(idx, assigned_round)])
        states = updated

    best = max(states.values(), key=lambda item: (item[0], len(item[1])))
    chosen = {idx: assigned_round for idx, assigned_round in best[1]}
    for idx, candidate in enumerate(candidates):
        candidate.keep = idx in chosen
        if not candidate.keep:
            continue
        candidate.cost_round = chosen[idx]
        candidate.surplus = candidate.cost_round - candidate.market_round
        candidate.surplus_value = keeper_surplus_value(
            candidate.adp_overall, candidate.cost_round, rules.league_size
        )
        candidate.verdict = verdict(candidate.surplus, rules)


def evaluate(
    candidates: Sequence[KeeperCandidate],
    rules: KeeperRules,
    limit: Optional[int] = None,
) -> List[KeeperCandidate]:
    """Analyze candidates and mark the value-optimal ``limit`` keepers.

    With independent costs this ranks by nonlinear pick-value surplus. With
    one-per-round enabled, dynamic programming jointly selects players and
    assigns legal cost rounds. Returns a new ranked list; inputs are mutated.
    """
    ranked = sorted((analyze(c, rules) for c in candidates), key=_sort_key)
    n = len(ranked) if limit is None else max(0, int(limit))
    if rules.one_per_round:
        _optimize_unique_rounds(ranked, rules, n)
    else:
        eligible = [c for c in ranked if c.surplus is not None and c.surplus > 0]
        selected = {id(c) for c in eligible[:n]}
        for c in ranked:
            c.keep = id(c) in selected
    return ranked


def total_surplus(candidates: Sequence[KeeperCandidate]) -> int:
    """Sum of surplus across the currently-kept candidates."""
    return sum(c.surplus or 0 for c in candidates if c.keep)


def cost_collisions(candidates: Sequence[KeeperCandidate]) -> dict:
    """Cost rounds shared by more than one *kept* candidate.

    Many keeper leagues forbid keeping two players at the same draft-round cost
    (you only own one pick per round) and bump duplicates to adjacent rounds.
    Returns {round: [player_id, ...]} for each round with a conflict, so the UI
    can warn rather than silently mis-price. Empty when there's no clash."""
    by_round: dict = {}
    for c in candidates:
        if c.keep:
            by_round.setdefault(c.cost_round, []).append(c.player_id)
    return {rd: ids for rd, ids in by_round.items() if len(ids) > 1}


def resolve_cost_collisions(candidates: Sequence[KeeperCandidate], rules: KeeperRules) -> List[KeeperCandidate]:
    """Give every *kept* candidate a unique cost round (one pick per round).

    Greedy selection can land two keepers on the same cost round; leagues that
    enforce one-pick-per-round bump a duplicate to a neighbouring round. This
    keeps the strongest claim (highest surplus, then value) on the contested
    round and bumps the others to the nearest open round — **earlier (costlier)
    preferred** on a tie, since a bumped keeper should never get *cheaper* (that
    would inflate the surplus the tool is trying to price honestly). Each bumped
    candidate's ``cost_round``, ``surplus`` and ``verdict`` are recomputed from
    the resolved round, in place. Returns the same list.

    This re-prices the selected set; it does not re-optimise which players are
    kept (a rare case where a bump turns a marginal keep negative is surfaced by
    the now-accurate surplus rather than silently reshuffled)."""
    last = max(1, int(rules.num_rounds))
    kept = [c for c in candidates if c.keep]
    # Strongest claim first, so it holds its natural round and weaker ones move.
    order = sorted(kept, key=lambda c: (-(c.surplus if c.surplus is not None else -9999),
                                        -(c.value or 0.0)))
    taken: set = set()
    for c in order:
        r = c.cost_round
        if 1 <= r <= last and r not in taken:
            taken.add(r)
            continue
        placed = None
        for d in range(1, last):
            for cand in (r - d, r + d):   # earlier first, then later, widening out
                if 1 <= cand <= last and cand not in taken:
                    placed = cand
                    break
            if placed is not None:
                break
        if placed is None:
            placed = r   # degenerate: more keepers than rounds — leave it clashing
        c.cost_round = placed
        taken.add(placed)
        if c.market_round is not None:
            c.surplus = c.cost_round - c.market_round
            c.surplus_value = keeper_surplus_value(
                c.adp_overall, c.cost_round, rules.league_size
            )
            c.verdict = verdict(c.surplus, rules)
    return list(candidates)


def project_league_keepers(
    rosters: dict,
    rules: KeeperRules,
    limit: Optional[int],
) -> dict:
    """Project each team's likely keepers for draft-board planning.

    ``rosters`` maps a team key -> that team's list of KeeperCandidate. Every
    team is assumed to keep its value-optimal set under ``limit`` (the same
    surplus optimizer used for the viewer). Returns team key -> list of kept
    player_ids.

    This is a *projection*: real keeper intentions aren't published before a
    draft, so the caller should surface these as editable estimates, not fact —
    except for the viewer's own team, whose selections are known."""
    out: dict = {}
    for team, cands in (rosters or {}).items():
        ranked = evaluate(cands, rules, limit=limit)
        out[team] = [c.player_id for c in ranked if c.keep]
    return out


# ======================================================================
# From utils/utils.py (split per consolidation map)
# ======================================================================

# --- utils/utils.py L1884 ---
def bucket_for_slot(slot: int, num_teams: int = 10) -> str:
    """
    Map a pick number within the round (1..N) into 'early'/'mid'/'late'.
    Tuned for 10-team by default.
    """
    if slot <= 0:
        return "late"

    if num_teams == 10:
        if 1 <= slot <= 3:
            return "early"
        elif 4 <= slot <= 6:
            return "mid"
        else:
            return "late"

    # Generic fallback: split into thirds
    third = max(1, num_teams // 3)
    if slot <= third:
        return "early"
    elif slot <= 2 * third:
        return "mid"
    else:
        return "late"
