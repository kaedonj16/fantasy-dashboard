"""Lineup Lab payload builder (Start/Sit tab, Lab mode).

Builds the per-week data the browser needs to run 2,000 client-side
lineup simulations: your starters + eligible bench (with shared scoring
profiles), the matchup opponent as a team-level distribution, and the
same-team correlation pairs for the Gaussian copula.

The Lab never re-projects; it re-simulates around Sleeper's projections
(the mean). Profiles change uncertainty, skew and dud risk, not the mean.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Injury rates shared with the season sim (no numpy in this module's import
# chain, so the lint shard stays happy).
from data_building.injury_rates import expected_injury_loss_per_week as _inj_expected_injury_loss
from data_building.injury_rates import injury_onset_rate as _inj_onset_rate

# Mirror the serious-injury set from utils/waiver_score.py (PR #2075) without
# importing the whole scoring module here.
_SERIOUS_INJURY = frozenset(
    {"IR", "PUP", "NFI", "SUSP", "SUS", "OUT", "DOUBTFUL", "NA"}
)

_N_SIMS = 2000


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result else default


def _slot_counts(roster_positions: List[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for slot in roster_positions or []:
        name = str(slot or "").upper()
        if not name or name in ("BN", "IR", "TAXI"):
            continue
        counts[name] = counts.get(name, 0) + 1
    return counts


def _slot_eligible_positions(slot: str) -> frozenset:
    try:
        from utils.lineup_slots import slot_eligible_positions
        return frozenset(slot_eligible_positions(slot))
    except Exception:
        base = str(slot or "").upper()
        if base in ("QB",):
            return frozenset({"QB"})
        if base in ("RB",):
            return frozenset({"RB"})
        if base in ("WR",):
            return frozenset({"WR"})
        if base in ("TE",):
            return frozenset({"TE"})
        if base in ("K",):
            return frozenset({"K"})
        if base in ("DEF",):
            return frozenset({"DEF"})
        if base in ("FLEX",):
            return frozenset({"RB", "WR", "TE"})
        if base in ("WRRB_FLEX", "WR/RB"):
            return frozenset({"RB", "WR"})
        if base in ("REC_FLEX", "WR/TE"):
            return frozenset({"WR", "TE"})
        if base in ("SUPER_FLEX", "OP"):
            return frozenset({"QB", "RB", "WR", "TE"})
        return frozenset({"QB", "RB", "WR", "TE", "K", "DEF"})


def _player_pos(players_index: dict, pid: str) -> str:
    meta = players_index.get(str(pid)) or {}
    pos = str(meta.get("position") or meta.get("pos") or "").upper()
    if pos == "DST":
        pos = "DEF"
    return pos


def _player_name(players_index: dict, pid: str) -> str:
    meta = players_index.get(str(pid)) or {}
    name = meta.get("full_name") or meta.get("name") or ""
    if name:
        return str(name)
    team = str(meta.get("team") or "").upper()
    if _player_pos(players_index, pid) == "DEF" and team:
        return f"{team} DST"
    return f"Player {pid}"


def _player_team(players_index: dict, pid: str) -> str:
    meta = players_index.get(str(pid)) or {}
    return str(meta.get("team") or "").upper()


def _is_seriously_hurt(players_index: dict, pid: str) -> bool:
    meta = players_index.get(str(pid)) or {}
    status = str(meta.get("injury_status") or "").upper()
    return status in _SERIOUS_INJURY


def _projection_lookup(scoring: dict, season: int, week: int, ctx: dict):
    """Return a pid -> projected points function (league scoring)."""
    raw_week_map: dict = {}
    wk_proj: dict = {}
    try:
        from utils.utils import load_week_projection as _lwp
        raw_week_map = _lwp(int(season), int(week)) or {}
    except Exception:
        raw_week_map = {}
    try:
        proj_by_week = ctx.get("proj_by_week")
        if not proj_by_week:
            from dashboard_services.service import build_projections_by_week
            proj_by_week = build_projections_by_week(season, 18, scoring)
        wk_proj = ((proj_by_week or {}).get(week) or {}).get("projections") or {}
    except Exception:
        wk_proj = {}
    try:
        from utils.fantasy_scoring import weekly_projection_points as _wpp
    except Exception:
        _wpp = None  # type: ignore

    def _lookup(pid: str, pos: str = "") -> float:
        if _wpp is not None:
            try:
                pts = _wpp(raw_week_map, pid, scoring, pos)
            except Exception:
                pts = None
            if pts is not None:
                return round(_safe_float(pts), 1)
        v = wk_proj.get(pid)
        if v is None:
            v = wk_proj.get(str(pid))
        return round(_safe_float(v), 1)

    return _lookup


def _week_team_set(season: int, week: int) -> set:
    """NFL teams playing in the given week (for BYE detection)."""
    try:
        from utils.utils import load_week_sched as _lsched
        sched = _lsched(season, week) or []
        teams = set()
        for game in sched:
            home = str(game.get("home") or "").upper()
            away = str(game.get("away") or "").upper()
            if home:
                teams.add(home)
            if away:
                teams.add(away)
        return teams
    except Exception:
        return set()


def _on_bye(players_index: dict, pid: str, week_teams: set) -> bool:
    if not week_teams:
        return False
    pos = _player_pos(players_index, pid)
    team = _player_team(players_index, pid)
    if pos == "DEF":
        # Sleeper defenses are keyed by team abbrev-ish ids; name fallback.
        meta = players_index.get(str(pid)) or {}
        team = team or str(meta.get("full_name") or "").upper().split(" ")[0]
    return bool(team) and team not in week_teams


def _optimal_fallback_lineup(
    candidate_pids: List[str],
    proj_fn,
    players_index: dict,
    slot_counts: Dict[str, int],
) -> List[tuple]:
    """Greedy optimal lineup by projection when real starters are unknown."""
    scored = []
    for pid in candidate_pids:
        pos = _player_pos(players_index, pid)
        scored.append((_safe_float(proj_fn(pid, pos)), pid, pos))
    scored.sort(reverse=True)
    used: set = set()
    lineup: List[tuple] = []
    # Expand slot counts into an ordered slot list (fixed slots first).
    order = ["QB", "RB", "WR", "TE", "FLEX", "WRRB_FLEX", "REC_FLEX",
             "SUPER_FLEX", "OP", "K", "DEF"]
    slots: List[str] = []
    for name in order:
        slots.extend([name] * slot_counts.get(name, 0))
    for name in slot_counts:
        if name not in order:
            slots.extend([name] * slot_counts[name])
    for slot in slots:
        eligible = _slot_eligible_positions(slot)
        pick = next(
            (s for s in scored if s[1] not in used and s[2] in eligible), None
        )
        if pick:
            used.add(pick[1])
            lineup.append((slot, pick[1]))
    return lineup


def _team_std_from_profiles(
    pids: List[str], profiles: Dict[str, dict], pairs: Dict[tuple, float]
) -> float:
    var = 0.0
    sigmas = {}
    for pid in pids:
        s = _safe_float((profiles.get(pid) or {}).get("std"), 8.0)
        sigmas[pid] = s
        var += s * s
    for (a, b), rho in pairs.items():
        if a in sigmas and b in sigmas:
            var += 2.0 * rho * sigmas[a] * sigmas[b]
    return max(math.sqrt(max(var, 0.0)), 8.0)


def _lab_tags(profile: dict) -> List[dict]:
    """Display-only context chips derived from profile factors."""
    factors = (profile or {}).get("factors") or {}
    tags: List[dict] = []
    if factors.get("unrealized_ay"):
        tags.append({"kind": "due", "label": "Due"})
    td_share = _safe_float(factors.get("td_share"), 0.0)
    if td_share >= 0.30:
        tags.append({"kind": "td", "label": "TD-dependent"})
    if factors.get("questionable"):
        tags.append({"kind": "q", "label": "Questionable"})
    if factors.get("teammate_out"):
        tags.append({"kind": "boost", "label": "Teammate out"})
    if factors.get("qb_out"):
        tags.append({"kind": "qb", "label": "QB out"})
    elif factors.get("new_qb"):
        tags.append({"kind": "qb", "label": "New QB"})
    return tags


def _profile_payload(profile: dict) -> dict:
    return {
        "mean": round(_safe_float(profile.get("mean")), 1),
        "std": round(_safe_float(profile.get("std")), 2),
        "skew_alpha": round(_safe_float(profile.get("skew_alpha")), 2),
        "dud_risk": round(_safe_float(profile.get("dud_risk")), 3),
    }


def build_lineup_lab_payload(
    *,
    ctx: dict,
    league_id: str,
    viewer_roster_id: Any,
    season: int,
    week: int,
    scoring_settings: Optional[dict] = None,
) -> dict:
    """Build the Lab payload for one viewer roster and week.

    Raises LookupError when the roster/matchup cannot be resolved.
    """
    scoring = scoring_settings or ctx.get("raw_scoring_settings") or ctx.get("scoring_settings") or {}
    players_index = ctx.get("players_index") or {}
    rosters = ctx.get("rosters") or []
    viewer_roster = next(
        (r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)),
        None,
    )
    if not viewer_roster:
        raise LookupError("viewer roster not found")

    reserve_set = {str(p) for p in (viewer_roster.get("reserve") or [])}
    taxi_set = {str(p) for p in (viewer_roster.get("taxi") or [])}
    roster_pids = [
        str(pid) for pid in (viewer_roster.get("players") or [])
        if str(pid) not in reserve_set and str(pid) not in taxi_set
    ]

    # ── Matchup: real starters + opponent ────────────────────────────────
    starters: List[str] = []
    opponent_roster_id: Any = None
    # One matchup fetch, reused for both sides below. A second fetch doubles
    # the chance the opponent silently goes missing (which used to render a
    # fake 50/50); the payload now flags it instead (see opp_missing).
    matchups: List[dict] = []
    try:
        from dashboard_services.api import get_matchups
        matchups = get_matchups(str(league_id), int(week)) or []
    except Exception:
        logger.debug("lineup-lab: matchup fetch failed", exc_info=True)
    mine = next(
        (m for m in matchups if str(m.get("roster_id")) == str(viewer_roster_id)),
        None,
    )
    if mine:
        starters = [
            str(p) for p in (mine.get("starters") or [])
            if p and str(p) != "0"
        ]
        mid = mine.get("matchup_id")
        opp = next(
            (m for m in matchups
             if m.get("matchup_id") == mid
             and str(m.get("roster_id")) != str(viewer_roster_id)),
            None,
        )
        if opp:
            opponent_roster_id = opp.get("roster_id")

    proj_fn = _projection_lookup(scoring, season, week, ctx)
    roster_positions = ctx.get("roster_positions") or []
    slot_counts = _slot_counts(roster_positions)

    if not starters:
        # Offseason / week not yet set: fall back to optimal by projection.
        starters = [pid for _, pid in _optimal_fallback_lineup(
            roster_pids, proj_fn, players_index, slot_counts)]

    # Map each starter to its slot for eligibility (order of roster_positions).
    slot_order: List[str] = []
    for slot in roster_positions or []:
        name = str(slot or "").upper()
        if name and name not in ("BN", "IR", "TAXI"):
            slot_order.append(name)
    starter_slots: Dict[str, str] = {}
    for i, pid in enumerate(starters):
        starter_slots[pid] = slot_order[i] if i < len(slot_order) else ""

    week_teams = _week_team_set(season, week)
    bench_all = [p for p in roster_pids if p not in set(starters)]

    def _bench_available(pid: str) -> bool:
        if _is_seriously_hurt(players_index, pid):
            return False
        if _on_bye(players_index, pid, week_teams):
            return False
        return True

    bench_avail = [p for p in bench_all if _bench_available(p)]

    # ── Opponent starters (reuses the matchup fetch above) ───────────────
    # opp_missing is stamped on the payload so the browser can show an
    # explicit retry state instead of a fake 50/50 with no opponent.
    opp_starters: List[str] = []
    opp_name = "Opponent"
    opp_missing = True
    if opponent_roster_id is not None:
        opp_entry = next(
            (m for m in matchups if str(m.get("roster_id")) == str(opponent_roster_id)),
            None,
        )
        if opp_entry:
            opp_starters = [
                str(p) for p in (opp_entry.get("starters") or [])
                if p and str(p) != "0"
            ]
        if opp_starters:
            opp_missing = False
        try:
            users = ctx.get("users") or []
            rosters_by_id = {str(r.get("roster_id")): r for r in rosters}
            oroster = rosters_by_id.get(str(opponent_roster_id)) or {}
            owner_id = oroster.get("owner_id")
            user = next((u for u in users if str(u.get("user_id")) == str(owner_id)), None)
            if user:
                opp_name = str(user.get("display_name") or user.get("username") or opp_name)
            elif oroster.get("metadata", {}).get("team_name"):
                opp_name = str(oroster["metadata"]["team_name"])
        except Exception:
            pass

    # ── Profiles + correlations (shared model) ───────────────────────────
    all_pids = list(dict.fromkeys(starters + bench_avail + opp_starters))
    requests = []
    for pid in all_pids:
        pos = _player_pos(players_index, pid)
        requests.append({
            "player_id": pid,
            "pos": pos,
            "mean": proj_fn(pid, pos),
        })
    try:
        from data_building.player_distributions import (
            build_profiles as _build_profiles,
            correlation_pairs as _correlation_pairs,
        )
        profiles = _build_profiles(requests, int(season), int(week))
    except Exception:
        logger.debug("lineup-lab: profile build failed", exc_info=True)
        profiles = {}
        _correlation_pairs = None  # type: ignore
    pairs: Dict[tuple, float] = {}
    if _correlation_pairs is not None:
        try:
            pairs = _correlation_pairs(list(dict.fromkeys(starters + bench_avail)), int(season))
        except Exception:
            pairs = {}

    def _prof(pid: str) -> dict:
        pos = _player_pos(players_index, pid)
        mean = proj_fn(pid, pos)
        return profiles.get(pid) or {
            "player_id": pid, "pos": pos, "mean": mean,
            "std": 2.0 + 0.42 * mean, "skew_alpha": 2.0, "dud_risk": 0.0,
            "n_games": 0.0, "factors": {"baseline_only": True},
        }

    # ── Matchup label per player (vs/away) ───────────────────────────────
    matchup_labels: Dict[str, str] = {}
    try:
        from utils.utils import load_week_sched as _lsched2
        sched = _lsched2(season, week) or []
        for game in sched:
            home = str(game.get("home") or "").upper()
            away = str(game.get("away") or "").upper()
            if home and away:
                matchup_labels[home] = f"vs {away}"
                matchup_labels[away] = f"@ {home}"
    except Exception:
        pass

    def _entry(pid: str, slot: str, bench_for_slot: Optional[List[dict]] = None) -> dict:
        pos = _player_pos(players_index, pid)
        prof = _prof(pid)
        mean = _safe_float(prof.get("mean"))
        std = _safe_float(prof.get("std"))
        team = _player_team(players_index, pid)
        entry = {
            "player_id": pid,
            "name": _player_name(players_index, pid),
            "pos": pos,
            "slot": slot,
            "proj": round(mean, 1),
            "floor": round(max(0.0, mean - 1.28 * std), 1),
            "ceiling": round(mean + 1.28 * std, 1),
            "matchup": matchup_labels.get(team, ""),
            "tags": _lab_tags(prof),
            "profile": _profile_payload(prof),
        }
        if bench_for_slot is not None:
            entry["bench"] = bench_for_slot
        return entry

    # Bench entries with per-slot eligibility.
    bench_entries: Dict[str, dict] = {}
    for pid in bench_avail:
        bench_entries[pid] = _entry(pid, "BN")

    lineup = []
    for pid in starters:
        slot = starter_slots.get(pid, "")
        eligible = _slot_eligible_positions(slot) if slot else frozenset()
        bench_for_slot = [
            bench_entries[p] for p in bench_avail
            if not eligible or _player_pos(players_index, p) in eligible
        ]
        # Sort bench by projection desc for the default view.
        bench_for_slot.sort(key=lambda e: e["proj"], reverse=True)
        lineup.append(_entry(pid, slot, bench_for_slot))

    # ── Opponent team distribution ───────────────────────────────────────
    opp_mean = 0.0
    opp_std = 15.0
    opp_injury_adj = 0.0
    if opp_starters:
        opp_profiles = {pid: _prof(pid) for pid in opp_starters}
        opp_mean = round(sum(_safe_float(p.get("mean")) for p in opp_profiles.values()), 1)
        opp_pairs = {(a, b): r for (a, b), r in pairs.items()
                     if a in opp_profiles and b in opp_profiles}
        opp_std = round(_team_std_from_profiles(opp_starters, opp_profiles, opp_pairs), 1)
        # The browser applies per-player in-game injury draws to YOUR side only
        # (the opponent ships as a team-level aggregate with no per-player
        # draws). Haircut the opponent mean by the expected injury loss so the
        # win probability stays centered — same rates as the season engine
        # (data_building/injury_rates.py), replacement at the waiver-wire
        # fallback since we have no visibility into their bench.
        opp_injury_adj = round(sum(
            _inj_expected_injury_loss(_safe_float(p.get("mean")), p.get("pos"))
            for p in opp_profiles.values()
        ), 1)
        opp_mean = round(max(opp_mean - opp_injury_adj, 0.0), 1)

    corr_out = {}
    for (a, b), rho in pairs.items():
        corr_out[f"{a}:{b}"] = round(float(rho), 3)

    return {
        "week": int(week),
        "n_sims": _N_SIMS,
        # Per-position single-game injury onset rates, sourced from
        # data_building/injury_rates.py (the same research-backed table as the
        # season sim). The browser draws in-game injuries off these.
        "injury_onset": {
            pos: round(_inj_onset_rate(pos), 4)
            for pos in ("QB", "RB", "WR", "TE", "K", "DEF")
        },
        "you": {
            "roster_id": viewer_roster_id,
            "lineup": lineup,
        },
        "opponent": {
            "name": opp_name,
            "roster_id": opponent_roster_id,
            "mean": opp_mean,
            "std": opp_std,
            "missing": opp_missing,
            "injury_adj": opp_injury_adj,
        },
        "corr": corr_out,
    }
