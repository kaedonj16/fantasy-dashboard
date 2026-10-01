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


def _assign_starter_slots(
    starters: List[str],
    claimed_slots: Dict[str, str],
    slot_order: List[str],
    players_index: dict,
) -> Dict[str, str]:
    """Resolve the slot each starter actually occupies.

    Providers that know the real slots publish a parallel ``starters_slots``
    list (Yahoo reports ``selected_position`` per player); those claims win
    when they are legal for the player's position and within the league's
    slot capacity. Any starter left over is seated by position eligibility,
    most restrictive slots first, so a flex-eligible player cannot steal a
    dedicated slot from the player who needs it. Pairing starters with
    slots by raw list index is never safe: a provider whose starter order
    differs from ``roster_positions`` order seats a WR in an RB slot (and
    a K in the TE slot), and every per-slot bench pool built from the
    wrong slot is wrong too.
    """
    capacity: Dict[str, int] = {}
    for slot in slot_order:
        capacity[slot] = capacity.get(slot, 0) + 1
    assigned: Dict[str, str] = {}
    used: Dict[str, int] = {}
    for pid in starters:
        slot = str(claimed_slots.get(pid) or "").upper()
        if not slot or pid in assigned:
            continue
        if used.get(slot, 0) >= capacity.get(slot, 0):
            continue
        pos = _player_pos(players_index, pid)
        if pos and pos not in _slot_eligible_positions(slot):
            continue
        assigned[pid] = slot
        used[slot] = used.get(slot, 0) + 1
    remaining: List[str] = []
    for slot in dict.fromkeys(slot_order):
        remaining.extend([slot] * (capacity[slot] - used.get(slot, 0)))
    remaining.sort(key=lambda s: len(_slot_eligible_positions(s)))
    pool = [pid for pid in starters if pid not in assigned]
    for slot in remaining:
        eligible = _slot_eligible_positions(slot)
        pick = next(
            (p for p in pool if _player_pos(players_index, p) in eligible),
            None,
        )
        if pick is None:
            continue
        assigned[pick] = slot
        pool.remove(pick)
    for pid in starters:
        assigned.setdefault(pid, "")
    return assigned


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
    platform: str = "sleeper",
    guest_pids: Optional[List[str]] = None,
) -> dict:
    """Build the Lab payload for one viewer roster and week.

    Raises LookupError when the roster/matchup cannot be resolved.

    Guests are non-roster players (free agents or another team's) the
    viewer wants to test as hypothetical starters. They are returned under
    ``payload["guests"]`` as bench-shaped entries built by the same
    profile machinery as roster players; the browser seats them into
    eligible slots' bench pools. Guest state is view-only: guests are
    never written into the lineup, the bench, or any roster data. A guest
    whose data cannot support a simulation (unknown player, bye week,
    serious injury, no projection) is flagged ``available: False`` with an
    explicit reason instead of invented samples.
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

    # Guests: anyone on the viewer's roster (starters, bench, IR, taxi) is
    # already in the Lab through the normal pools, so they are not guests.
    _viewer_all_pids = {str(p) for p in (viewer_roster.get("players") or [])}
    guest_pids_clean: List[str] = []
    for _gp in (guest_pids or []):
        _gp = str(_gp)
        if _gp and _gp not in _viewer_all_pids and _gp not in guest_pids_clean:
            guest_pids_clean.append(_gp)

    # ── Matchup: real starters + opponent ────────────────────────────────
    starters: List[str] = []
    matchup_slots: Dict[str, str] = {}
    opponent_roster_id: Any = None
    # One matchup fetch, reused for both sides below. A second fetch doubles
    # the chance the opponent silently goes missing (which used to render a
    # fake 50/50); the payload now flags it instead (see opp_missing).
    matchups: List[dict] = []
    try:
        # Canonical platform path: provider adapters normalize every
        # platform's matchups to the same Sleeper-shaped rows. The legacy
        # dashboard_services.api.get_matchups is Sleeper-only and returns
        # [] for a Yahoo/ESPN league id, which silently dropped the
        # opponent for every non-Sleeper league.
        from dashboard_services.platform_api import get_matchups
        matchups = get_matchups(
            platform, str(league_id), int(week), int(season)) or []
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
        # Parallel real slots when the provider publishes them (Yahoo):
        # starters arrive in the provider's own order there, so the slot
        # list is the only record of who sits where.
        raw_slots = mine.get("starters_slots") or []
        matchup_slots = {
            str(p): str(s)
            for p, s in zip(mine.get("starters") or [], raw_slots)
            if p and str(p) != "0" and s
        }
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

    fallback_slots: Dict[str, str] = {}
    if not starters:
        # Offseason / week not yet set: fall back to optimal by projection.
        fallback = _optimal_fallback_lineup(
            roster_pids, proj_fn, players_index, slot_counts)
        starters = [pid for _, pid in fallback]
        fallback_slots = {pid: slot for slot, pid in fallback}

    # Map each starter to the slot it actually occupies (see
    # _assign_starter_slots): provider-claimed slots first, then position
    # eligibility. Never a raw index zip against roster_positions.
    slot_order: List[str] = []
    for slot in roster_positions or []:
        name = str(slot or "").upper()
        if name and name not in ("BN", "IR", "TAXI"):
            slot_order.append(name)
    starter_slots = _assign_starter_slots(
        starters, fallback_slots or matchup_slots, slot_order, players_index)
    # Display in slot order regardless of the provider's starter order.
    _slot_rank = {slot: i for i, slot in enumerate(slot_order)}
    starters.sort(
        key=lambda pid: _slot_rank.get(starter_slots.get(pid, ""), len(slot_order)))

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
    opp_entry: Optional[dict] = None
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
    # Guests with a resolvable position join the same profile build (and
    # the same-team correlation pool) as roster players.
    _guest_known = [p for p in guest_pids_clean if _player_pos(players_index, p)]
    all_pids = list(dict.fromkeys(
        starters + bench_avail + opp_starters + _guest_known))
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
            pairs = _correlation_pairs(list(dict.fromkeys(
                starters + bench_avail + _guest_known)), int(season))
        except Exception:
            pairs = {}

    # Per-player usage context for the row meta line: the season average of
    # the position's key usage stat (QB snap %, RB touches, WR/TE targets),
    # the same source and convention as the Start/Sit cards. Display only;
    # missing data simply omits the stat.
    usage_trends: Dict[str, dict] = {}
    try:
        from data_building.weekly_metrics import get_usage_trends
        usage_trends = get_usage_trends(int(season)) or {}
    except Exception:
        logger.debug("lineup-lab: usage trends unavailable", exc_info=True)
        usage_trends = {}

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
    sched: list = []
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

    # ── Live state (current week only) ───────────────────────────────────
    # Once games start, the matchup entries carry real per-player points
    # (players_points). A player whose game is final locks at their actual
    # points in the browser sim; a player mid-game is flagged live for
    # display only (the sim has no partial-game model). The final+points
    # double gate keeps every pre-game and past-week payload unchanged.
    live_by_pid: Dict[str, dict] = {}
    opp_live_points: Optional[float] = None
    mine_points = (mine or {}).get("players_points") or {}
    opp_points = (opp_entry or {}).get("players_points") or {}
    _is_current_week = False
    try:
        _is_current_week = int(week) == int(ctx.get("current_week") or 0)
    except (TypeError, ValueError):
        _is_current_week = False
    if _is_current_week and (mine_points or opp_points):
        try:
            from utils.utils import build_games_by_team as _bgbt
            from utils.utils import lookup_team_map as _ltm
            _games_by_team = _bgbt(sched or [])
        except Exception:
            _games_by_team = {}
        if _games_by_team:
            def _live_state(pid: str, points_map: dict) -> Optional[dict]:
                team = _player_team(players_index, pid)
                if _player_pos(players_index, pid) == "DEF" and not team:
                    meta = players_index.get(str(pid)) or {}
                    team = str(meta.get("full_name") or "").upper().split(" ")[0]
                if not team:
                    return None
                game = _ltm(_games_by_team, team)
                if not game:
                    return None
                status = game.get("status")
                # The time-window status can call a long game final early;
                # when the feed carries an explicit game status it wins.
                raw_game = game.get("game") or {}
                tank_code = str(raw_game.get("gameStatusCode") or "").strip()
                tank_text = str(raw_game.get("gameStatus") or "").lower()
                if tank_code == "2" or "final" in tank_text or "completed" in tank_text:
                    status = "post"
                elif tank_code == "1" or "in progress" in tank_text or "live" in tank_text:
                    status = "in"
                pts: Optional[float] = None
                raw_pts = points_map.get(pid)
                if raw_pts is None:
                    raw_pts = points_map.get(str(pid))
                if raw_pts is not None:
                    try:
                        pts = round(float(raw_pts), 1)
                    except (TypeError, ValueError):
                        pts = None
                if status == "post":
                    if pts is None:
                        return None
                    return {"status": "final", "points": pts}
                if status == "in":
                    state: Dict[str, Any] = {"status": "live"}
                    if pts is not None:
                        state["points"] = pts
                    return state
                return None

            for pid in list(dict.fromkeys(starters + bench_avail)):
                st = _live_state(pid, mine_points)
                if st:
                    live_by_pid[pid] = st
            if opp_starters:
                opp_states = [_live_state(pid, opp_points) for pid in opp_starters]
                if all(s is not None and s["status"] == "final" for s in opp_states):
                    opp_live_points = round(
                        sum(s["points"] for s in opp_states if s), 1)

    def _entry(pid: str, slot: str, bench_for_slot: Optional[List[dict]] = None,
               eligible_positions: Optional[frozenset] = None) -> dict:
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
            "usage_stat": (usage_trends.get(pid) or {}).get("stat"),
            "usage_avg": (usage_trends.get(pid) or {}).get("season_avg"),
        }
        live = live_by_pid.get(pid)
        if live:
            entry["live"] = live
        if bench_for_slot is not None:
            entry["bench"] = bench_for_slot
        if eligible_positions is not None:
            # Per-slot position eligibility, so the client can re-seat a
            # demoted starter onto every bench they qualify for after a swap.
            entry["eligible"] = sorted(eligible_positions)
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
        lineup.append(_entry(pid, slot, bench_for_slot, eligible))

    # ── Guests (hypothetical starters, view state only) ──────────────────
    def _owner_label(pid: str) -> str:
        for r in rosters:
            if str(pid) in {str(x) for x in (r.get("players") or [])}:
                owner_id = r.get("owner_id")
                user = next(
                    (u for u in (ctx.get("users") or [])
                     if str(u.get("user_id")) == str(owner_id)), None)
                if user:
                    return (str(user.get("display_name")
                                or user.get("username") or "")
                            or str((r.get("metadata") or {}).get("team_name") or "")
                            or "Another team")
                return (str((r.get("metadata") or {}).get("team_name") or "")
                        or "Another team")
        return "Free Agent"

    guests_out: List[dict] = []
    for pid in guest_pids_clean:
        entry = _entry(pid, "BN")
        entry["guest"] = True
        entry["owner_label"] = _owner_label(pid)
        _gstatus = str(
            (players_index.get(str(pid)) or {}).get("injury_status") or ""
        ).upper()
        if pid not in players_index and not _player_pos(players_index, pid):
            entry["available"] = False
            entry["unavailable_reason"] = "Player not found in the player pool"
        elif _on_bye(players_index, pid, week_teams):
            entry["available"] = False
            entry["unavailable_reason"] = f"On bye in Week {int(week)}"
        elif _gstatus in _SERIOUS_INJURY:
            entry["available"] = False
            entry["unavailable_reason"] = f"Unavailable ({_gstatus.title()})"
        elif not (entry.get("proj") or 0) > 0:
            entry["available"] = False
            entry["unavailable_reason"] = f"No projection for Week {int(week)}"
        else:
            entry["available"] = True
            entry["unavailable_reason"] = None
        guests_out.append(entry)

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

    payload = {
        "week": int(week),
        "n_sims": _N_SIMS,
        "guests": guests_out,
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
    # Live keys exist only when live state does, so pre-game and past-week
    # payloads stay byte-identical to before. The opponent locks only when
    # their whole lineup is final (their side ships as an aggregate).
    if opp_live_points is not None:
        payload["opponent"]["live_points"] = opp_live_points
    if live_by_pid or opp_live_points is not None:
        payload["live"] = True
    return payload
