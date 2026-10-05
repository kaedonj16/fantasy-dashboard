"""Consolidated utils module: start_sit.

start/sit inputs, scoring, game conditions, QB situation

Merged from: utils/start_sit_context.py, utils/start_sit_score.py, utils/game_conditions.py, utils/qb_situation.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# ======================================================================
# From utils/start_sit_context.py
# ======================================================================

"""Pure adapters from existing weekly/team caches into Start/Sit inputs."""

from statistics import mean, pstdev

from utils.nfl import normalize_nfl_team

# Display-only absence notes (teammate / opponent injuries). These never feed
# the start/sit score: Sleeper's projections already redistribute opportunity
# when a teammate is out, so scoring them again would double-count.
_ABSENCE_SKILL_POS = ("QB", "RB", "WR", "TE")
_ABSENCE_DEF_POS = ("DL", "DE", "DT", "NT", "EDGE", "LB", "OLB", "ILB", "MLB",
                    "DB", "CB", "S", "FS", "SS")
# Sleeper gives offensive linemen no depth data at all (depth_chart_order is
# None for starters and practice-squadders alike), so "starting" linemen are
# identified by snap share instead (see starting_lineman_pids).
_OL_POSITIONS = frozenset({"OL", "T", "G", "C", "OT", "OG"})
_ABSENCE_STATUSES = frozenset({"IR", "PUP", "NFI", "SUSP", "SUS", "OUT",
                               "DOUBTFUL", "NA"})
_ABSENCE_STATUS_LABEL = {
    "OUT": "Out", "DOUBTFUL": "Doubtful", "IR": "IR", "PUP": "PUP",
    "NFI": "NFI", "SUSP": "Suspended", "SUS": "Suspended", "NA": "Out",
}


def _absence_entry(pid: str, p: dict, *, productive: bool = False,
                   starting_lineman: bool = False) -> dict | None:
    """One display note for a seriously-hurt player, or None when not notable."""
    if not isinstance(p, dict):
        return None
    status = str(p.get("injury_status") or p.get("status") or "").strip().upper()
    if status not in _ABSENCE_STATUSES:
        return None
    pos = str(p.get("position") or p.get("pos") or "").strip().upper()
    name = str(p.get("full_name") or p.get("name") or "").strip()
    if not name:
        return None
    body = str(p.get("injury_body_part") or "").strip()
    label = _ABSENCE_STATUS_LABEL.get(status, status.title())
    if pos in _OL_POSITIONS:
        # The "OL · " tag rides inside the parenthetical so the existing
        # renderers (which split the text at " (") show it in the muted
        # status part: "Tyler Smith (OL · IR, Thumb)".
        text = f"{name} (OL · {label}{', ' + body if body else ''})"
    else:
        text = f"{name} ({label}{', ' + body if body else ''})"
    try:
        order = p.get("depth_chart_order")
        order = float(order) if order is not None else None
    except (TypeError, ValueError):
        order = None
    # Importance gate: only absences a viewer would actually care about.
    # Skill positions: a proven producer (pooled weekly points) always counts;
    # otherwise depth order must say starter/immediate backup. Depth order for
    # skill spots is polluted by the injury itself (a hurt starter slides
    # down), which is why the production signal exists. QBs are stricter:
    # only the starter matters. Defense depth order is per-slot and clean
    # (deep/IR players are None), so starter-or-backup is the whole test and
    # production never applies.
    if pos in _ABSENCE_SKILL_POS:
        if not productive:
            if pos == "QB":
                if order != 1:
                    return None
            elif order is None or order > 2:
                return None
    elif pos in _ABSENCE_DEF_POS:
        if order is None or order > 2:
            return None
    elif pos in _OL_POSITIONS:
        # No depth signal exists for linemen; only a snap-share-identified
        # starter counts (callers pass the set via build_absence_index).
        if not starting_lineman:
            return None
    return {"pid": str(pid), "name": name, "pos": pos, "status": status,
            "text": text, "depth_order": order}


def productive_pids_from_weekly_points(*weekly_maps, min_games: int = 4,
                                       min_ppg: float = 6.0) -> set:
    """Player ids with a real production track record, pooled across seasons.

    Each map is ``{pid: [weekly fantasy points]}`` (one entry per game with a
    stat line). A player qualifies with at least ``min_games`` pooled games
    averaging at least ``min_ppg``. Pure and defensive: junk input is skipped,
    never raised on.
    """
    pooled: dict = {}
    for weekly in weekly_maps:
        if not isinstance(weekly, dict):
            continue
        for pid, pts in weekly.items():
            if not isinstance(pts, (list, tuple)):
                continue
            try:
                vals = [float(v) for v in pts]
            except (TypeError, ValueError):
                continue
            if vals:
                pooled.setdefault(str(pid), []).extend(vals)
    out = set()
    for pid, vals in pooled.items():
        if len(vals) >= min_games and sum(vals) / len(vals) >= min_ppg:
            out.add(pid)
    return out


def starting_lineman_pids(snap_totals, positions_by_pid, *,
                          min_games: int = 2, min_share: float = 0.5) -> set:
    """Ids of starting offensive linemen, identified by snap share.

    ``snap_totals`` is a list of per-season maps
    ``{pid: (off_snaps, team_snaps, games)}`` pooled across seasons;
    ``positions_by_pid`` maps pid -> Sleeper position. A lineman qualifies
    with at least ``min_games`` pooled games and a pooled offensive snap
    share of at least ``min_share`` (healthy starters sit near 100%, and
    hurt/backup linemen have no snap line at all). Sleeper carries no depth
    data for linemen, so this is the only "starting" signal. Pure and
    defensive: junk input is skipped, never raised on.
    """
    pooled: dict = {}
    if isinstance(snap_totals, (list, tuple)):
        for season_map in snap_totals:
            if not isinstance(season_map, dict):
                continue
            for pid, totals in season_map.items():
                if not isinstance(totals, (list, tuple)) or len(totals) < 3:
                    continue
                try:
                    off, team_snaps, games = (float(totals[0]),
                                              float(totals[1]),
                                              float(totals[2]))
                except (TypeError, ValueError):
                    continue
                acc = pooled.setdefault(str(pid), [0.0, 0.0, 0.0])
                acc[0] += off
                acc[1] += team_snaps
                acc[2] += games
    positions = positions_by_pid if isinstance(positions_by_pid, dict) else {}
    out = set()
    for pid, (off, team_snaps, games) in pooled.items():
        if games < min_games or team_snaps <= 0:
            continue
        if off / team_snaps < min_share:
            continue
        pos = str(positions.get(pid) or "").strip().upper()
        if pos in _OL_POSITIONS:
            out.add(pid)
    return out


def build_absence_index(full_players: dict, *, productive_pids=None,
                        starting_linemen=None) -> dict:
    """Precompute per-team serious-injury lists from a Sleeper players map.

    Returns ``{TEAM: {"skill": [...], "defense": [...], "line": [...]}}``
    where each entry is the ``_absence_entry`` dict above. One pass over the
    map so per-row lookups stay cheap. ``productive_pids`` (ids from
    ``productive_pids_from_weekly_points``) lets a proven producer count as
    notable even when his depth-chart order slid because of the injury.
    ``starting_linemen`` (ids from ``starting_lineman_pids``) is the only
    way an offensive lineman counts: Sleeper gives linemen no depth data.
    """
    productive_ids = {str(pid) for pid in productive_pids} if productive_pids else set()
    lineman_ids = {str(pid) for pid in starting_linemen} if starting_linemen else set()
    index: dict = {}
    for pid, p in (full_players or {}).items():
        if not isinstance(p, dict):
            continue
        team = str(p.get("team") or "").strip().upper()
        if not team:
            continue
        entry = _absence_entry(pid, p, productive=str(pid) in productive_ids,
                               starting_lineman=str(pid) in lineman_ids)
        if entry is None:
            continue
        bucket = index.setdefault(team, {"skill": [], "defense": [], "line": []})
        if entry["pos"] in _ABSENCE_SKILL_POS:
            bucket["skill"].append(entry)
        elif entry["pos"] in _ABSENCE_DEF_POS:
            bucket["defense"].append(entry)
        elif entry["pos"] in _OL_POSITIONS:
            bucket["line"].append(entry)
    return index


def absence_notes(index: dict, team: str, opponent: str, *,
                  exclude_pid: str | None = None, max_defense: int = 3,
                  max_teammates: int = 4, max_linemen: int = 2) -> dict:
    """Display-only absence notes for one player's game.

    ``teammates``: seriously-hurt skill-position teammates (QB/RB/WR/TE),
    excluding the player themselves, capped at ``max_teammates``; then
    seriously-hurt starting offensive linemen (snap-share identified),
    sorted by name under their own ``max_linemen`` cap so they never
    consume skill slots. ``opponents``: seriously-hurt defenders on the
    opposing team, capped at ``max_defense`` so the note never gets noisy.
    Empty lists when nothing is notable.
    """
    team = str(team or "").strip().upper()
    opponent = str(opponent or "").strip().upper()
    exclude = str(exclude_pid) if exclude_pid is not None else None

    def _sort_key(e: dict):
        order = e.get("depth_order")
        return (order is None, order if order is not None else 0.0,
                e.get("name") or "")

    teammates = []
    if team:
        for e in sorted((index or {}).get(team, {}).get("skill", []),
                        key=_sort_key):
            if exclude is not None and e.get("pid") == exclude:
                continue
            teammates.append(e)
        teammates = teammates[:max(0, int(max_teammates))]
        linemen = []
        for e in sorted((index or {}).get(team, {}).get("line", []) or [],
                        key=lambda e: e.get("name") or ""):
            if exclude is not None and e.get("pid") == exclude:
                continue
            linemen.append(e)
        teammates = teammates + linemen[:max(0, int(max_linemen))]
    opponents = []
    if opponent:
        for e in sorted((index or {}).get(opponent, {}).get("defense", []),
                        key=_sort_key)[:max(0, int(max_defense))]:
            opponents.append(e)
    return {"teammates": teammates, "opponents": opponents}


def expected_plays_context(team_rows: dict, team: str, opponent: str, nfl_avg) -> dict:
    """Blend actual offense plays with opponent plays faced; no invented pace."""
    team = normalize_nfl_team(team)
    opponent = normalize_nfl_team(opponent)
    try:
        avg = float(nfl_avg)
    except (TypeError, ValueError):
        return {}
    own = (team_rows or {}).get(team) or {}
    opp = (team_rows or {}).get(opponent) or {}
    try:
        offense = float(own["off_plays_pg"])
        allowed = float(opp.get("plays_faced_l4_pg") or opp["plays_faced_pg"])
    except (KeyError, TypeError, ValueError):
        return {}
    # Shrink both observations toward league average, then blend evenly. This
    # guards against one anomalous recent game while retaining real possession.
    expected = mean((0.7 * offense + 0.3 * avg, 0.7 * allowed + 0.3 * avg))
    return {"expected_team_plays": round(expected, 1), "league_average_plays": round(avg, 1),
            "source": "team_play_volume"}


def role_confidence_from_trend(trend: dict) -> float | None:
    """0..1 recent role stability, preserving confirmed promotions."""
    series = trend.get("series") if isinstance(trend, dict) else None
    if not isinstance(series, list) or len(series) < 2:
        return None
    try:
        values = [float(v) for v in series[-3:]]
    except (TypeError, ValueError):
        return None
    level = max(1.0, mean(values))
    stability = max(0.0, 1.0 - pstdev(values) / level)
    # A promoted player with two consecutive elevated readings is not punished
    # for the old low baseline that made them interesting in the first place.
    if len(values) >= 2 and values[-1] >= values[-2] >= mean(series[:-2] or values):
        stability = max(stability, 0.75)
    return round(min(1.0, stability), 3)


# ======================================================================
# From utils/start_sit_score.py
# ======================================================================

"""Single start/sit ranking formula: projection times capped multipliers.

The START badges, optimal-lineup banner, and Compare card all rank on this
score so they cannot contradict each other. Kept as a pure function so the
math is unit-tested without Flask.

Weekly projections already bake in the opponent, so defensive matchup rank is
*not* re-multiplied into the score (that used to double-count). Matchup stays on
the row as a chip only. Weather and Vegas *are* applied here because those
signals are usually missing from raw projection feeds.

Offensive-line quality is only *partly* reflected in projection feeds, so it is
applied as a deliberately small residual (±4%), not a full multiplier — the same
double-count caution as matchup. Callers pass the position-relevant 0-100 index
(run block for RB, pass block for QB/WR/TE); 50 is league-average and neutral.
"""

from typing import Optional

# Notable weather → position multipliers (only hurts; never boosts).
# Wind hurts pass game / kickers most; RBs are mostly insulated.
_WEATHER_MULT = {
    "wind": {"QB": 0.92, "WR": 0.94, "TE": 0.94, "K": 0.90, "RB": 0.99, "DEF": 0.97},
    "precip": {"QB": 0.95, "WR": 0.95, "TE": 0.96, "K": 0.93, "RB": 0.98, "DEF": 0.96},
    "cold": {"QB": 0.97, "WR": 0.97, "TE": 0.97, "K": 0.94, "RB": 0.98, "DEF": 0.97},
}
_WEATHER_DEFAULT = {"QB": 0.96, "WR": 0.96, "TE": 0.96, "K": 0.94, "RB": 0.98, "DEF": 0.97}


def _neutral_factors(proj: float) -> dict:
    return {
        "proj": proj,
        "form": 1.0,
        "matchup": 1.0,
        "usage": 1.0,
        "avail": 1.0,
        "vegas": 1.0,
        "floor": 1.0,
        "weather": 1.0,
        "oline": 1.0,
        "expected_plays": 1.0,
        "role": 1.0,
        "def_injuries": 1.0,
    }


def _weather_mult(weather_kind: Optional[str], position: Optional[str]) -> float:
    if not weather_kind:
        return 1.0
    kind = str(weather_kind).lower().strip()
    # Open-Meteo tag uses kind "weather" as a generic fallback; treat like light cold.
    if kind == "weather":
        kind = "cold"
    pos = (position or "").upper().strip()
    table = _WEATHER_MULT.get(kind) or _WEATHER_DEFAULT
    return float(table.get(pos) or table.get("WR") or 0.96)


def _lerp_clamped(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Linear interpolation of x in [x0, x1] → [y0, y1], clamped at the ends."""
    if x <= x0:
        return y0
    if x >= x1:
        return y1
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def _vegas_mult(implied_total: float, position: Optional[str]) -> float:
    """Position-aware Vegas nudge from implied team total.

    Piecewise-linear: ramps from the low-total haircut up to neutral over
    17→20 and from neutral up to the high-total boost over 24→27. The old
    step function jumped ~8% between 17.0 and 17.1 implied total, so a
    0.1-point Vegas line move could flip a start/sit call; now the nudge is
    continuous in the total.
    """
    pos = (position or "").upper().strip()
    pass_catcher = pos in ("QB", "WR", "TE")
    if pass_catcher:
        low, high = 0.92, 1.05
    elif pos == "RB":
        low, high = 0.96, 1.02
    elif pos == "K":
        low, high = 0.94, 1.03
    else:
        low, high = 0.94, 1.04
    if implied_total < 20.0:
        return _lerp_clamped(implied_total, 17.0, 20.0, low, 1.0)
    if implied_total > 24.0:
        return _lerp_clamped(implied_total, 24.0, 27.0, 1.0, high)
    return 1.0


def bottom_teams_by_implied_total(game_conditions: dict, n: int = 8) -> set:
    """Team abbreviations of the ``n`` teams with the lowest implied totals.

    Pure rank-based rule behind the "Low team total" demotion chip. Teams with
    no game (bye weeks have no ``game_conditions`` entry) or a missing/None
    implied total are excluded from the ranking. Ties are broken by team
    abbreviation so the result is deterministic. When fewer than ``n`` teams
    have totals, every ranked team is returned.
    """
    ranked = []
    for team, cond in (game_conditions or {}).items():
        try:
            total = float((cond or {}).get("implied_total"))
        except (TypeError, ValueError):
            continue
        ranked.append((total, str(team)))
    ranked.sort(key=lambda t: (t[0], t[1]))
    return {team for _, team in ranked[:max(n, 0)]}


def compute_start_score(
    proj_pts: float,
    *,
    on_bye: bool = False,
    recent_ppg: float = 0.0,
    season_ppg: float = 0.0,
    def_rank: Optional[float] = None,
    def_total: Optional[int] = None,
    usage_delta: Optional[float] = None,
    usage_season_avg: Optional[float] = None,
    injury_status: Optional[str] = None,
    implied_total: Optional[float] = None,
    bust_rate: Optional[float] = None,
    weather_kind: Optional[str] = None,
    position: Optional[str] = None,
    oline_index: Optional[float] = None,
    apply_matchup: bool = False,
    expected_team_plays: Optional[float] = None,
    league_average_plays: Optional[float] = None,
    role_confidence: Optional[float] = None,
    defensive_injury_impact: Optional[float] = None,
    low_total_team: bool = False,
) -> tuple[float, dict, Optional[str]]:
    """Return ``(score, score_factors, demotion)``.

    Factors: proj, form, matchup, usage, avail, vegas, floor, weather, oline.
    Bye and OUT/IR zero the score. Non-projection signals are capped multipliers.

    ``oline_index`` is the player's position-relevant 0-100 O-line rating (run
    block for RB, pass block for QB/WR/TE, composite otherwise); 50 is neutral.
    It is applied as a small ±4% residual because projections already reflect
    line quality in part. Pass None to leave the factor neutral.

    ``apply_matchup`` defaults to False because weekly projections already
    reflect the opponent. Pass True only for matchup-neutral projection feeds.
    ``def_rank`` / ``def_total`` are still accepted so callers can pass them
    without branching; they only affect the score when ``apply_matchup`` is True.

    ``low_total_team`` marks the player's team as one of the week's bottom-8
    implied-total teams (see ``bottom_teams_by_implied_total``). It only gates
    the "low_total" demotion label; the Vegas score multiplier is unchanged.
    """
    form = mu = usage = avail = vegas = floor = weather = oline = plays = role = def_inj = 1.0
    demotion = None
    try:
        proj = float(proj_pts or 0)
    except (TypeError, ValueError):
        proj = 0.0
    if on_bye:
        return 0.0, _neutral_factors(proj), "bye"

    try:
        recent = float(recent_ppg or 0)
    except (TypeError, ValueError):
        recent = 0.0
    try:
        season = float(season_ppg or 0)
    except (TypeError, ValueError):
        season = 0.0
    if recent > 0 and season > 0:
        # Mild form nudge — projections often already react to hot/cold streaks.
        form = min(1.08, max(0.92, recent / season))

    if apply_matchup and def_rank and def_total and def_total > 1:
        ease = (float(def_total) - float(def_rank)) / (float(def_total) - 1)
        # Residual only (±3%): even matchup-neutral feeds shouldn't swing hard.
        mu = 0.97 + ease * 0.06

    if usage_delta is not None and usage_season_avg:
        rel = float(usage_delta) / max(float(usage_season_avg), 1.0)
        usage = min(1.05, max(0.95, 1.0 + rel * 0.25))

    status = (injury_status or "").upper()
    if any(k in status for k in ("OUT", "IR", "SUSP", "DOUBT", "PUP", "DNP")):
        avail = 0.0
        demotion = "out"
    elif "QUESTION" in status or status in ("GTD", "Q"):
        # Soft residual only (−5%). Questionable players usually play, starting
        # them is normally fine, and weekly projections often already bake in
        # some risk — a 15% haircut was flipping too many start/sit calls.
        avail = 0.95
        demotion = "questionable"

    if implied_total is not None:
        try:
            imp = float(implied_total)
        except (TypeError, ValueError):
            imp = None
        if imp is not None:
            vegas = _vegas_mult(imp, position)
    # Rank-based label only: the "Low team total" demotion fires for players on
    # the week's bottom-8 implied-total teams, regardless of the absolute
    # total. The vegas score multiplier above is untouched.
    if low_total_team:
        demotion = demotion or "low_total"

    if bust_rate is not None:
        try:
            floor = min(1.10, max(0.90, 1.0 + (0.5 - float(bust_rate)) * 0.4))
        except (TypeError, ValueError):
            floor = 1.0

    weather = _weather_mult(weather_kind, position)
    if weather < 1.0:
        demotion = demotion or "weather"

    if oline_index is not None:
        try:
            e = max(0.0, min(100.0, float(oline_index))) / 100.0
            # 0.96 (worst line) .. 1.04 (best line); ~1.0 at league-average (50).
            oline = 0.96 + e * 0.08
            if oline < 1.0:
                demotion = demotion or "oline"
        except (TypeError, ValueError):
            oline = 1.0

    if expected_team_plays is not None and league_average_plays:
        try:
            # Pace affects opportunity, but feeds and projections are correlated;
            # apply only half the relative delta and cap the residual at ±5%.
            relative = float(expected_team_plays) / float(league_average_plays) - 1.0
            plays = min(1.05, max(0.95, 1.0 + relative * 0.5))
            if plays < 1.0:
                demotion = demotion or "low_play_volume"
        except (TypeError, ValueError, ZeroDivisionError):
            plays = 1.0
    if role_confidence is not None:
        try:
            confidence = min(1.0, max(0.0, float(role_confidence)))
            # Confidence widens/narrows the range more than it moves the mean.
            role = 0.97 + 0.03 * confidence
            if confidence < 0.45:
                demotion = demotion or "volatile_role"
        except (TypeError, ValueError):
            role = 1.0
    if defensive_injury_impact is not None:
        try:
            # Caller supplies a quality-weighted 0..1 impact; absence is neutral.
            def_inj = 1.0 + 0.03 * min(1.0, max(0.0, float(defensive_injury_impact)))
        except (TypeError, ValueError):
            def_inj = 1.0

    score = proj * form * mu * usage * avail * vegas * floor * weather * oline * plays * role * def_inj
    return score, {
        "proj": proj,
        "form": round(form, 3),
        "matchup": round(mu, 3),
        "usage": round(usage, 3),
        "avail": round(avail, 3),
        "vegas": round(vegas, 3),
        "floor": round(floor, 3),
        "weather": round(weather, 3),
        "oline": round(oline, 3),
        "expected_plays": round(plays, 3),
        "role": round(role, 3),
        "def_injuries": round(def_inj, 3),
    }, demotion


def likely_range(expected: float, role_confidence: Optional[float] = None) -> tuple[float, float]:
    """Conservative display range; confidence changes width, never fake mean precision."""
    try:
        mean = max(0.0, float(expected))
        confidence = 0.5 if role_confidence is None else min(1.0, max(0.0, float(role_confidence)))
    except (TypeError, ValueError):
        return (0.0, 0.0)
    width = mean * (0.22 + (1.0 - confidence) * 0.18)
    return round(max(0.0, mean - width), 1), round(mean + width, 1)


# ======================================================================
# From utils/game_conditions.py
# ======================================================================

"""Live game-condition signals for start/sit: Vegas totals and weather.

Two enrichments layered on top of the static venue tags (utils.nfl_stadiums):

  * **Vegas implied team total** - from Tank01's betting-odds endpoint (the same
    RapidAPI key the app already uses for projections). The game total plus a
    team's spread give its *implied team total*, the single best one-number read
    on how much scoring the market expects from that team this week.

  * **Weather** - from Open-Meteo (a keyless public forecast API), looked up by
    stadium coordinates for the game date. Only meaningful for outdoor venues, so
    domes are skipped. We surface a tag only when conditions are actually
    notable (strong wind, hard cold, real precipitation) - a mild, dry day gets
    no chip.

Design rules that mirror the rest of the app:
  * The pure parsing / math / tagging helpers take plain data and are fully unit
    tested; the network fetchers wrap them with a short on-disk + in-memory TTL
    cache and *never raise* - any failure degrades to None so the caller falls
    back to the static dome/cold tags.
  * Forecasts firm up only a few days out, and betting totals move all week, so
    both are cached briefly and re-fetched, not stored long term.
"""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pure helpers (no network) - unit tested directly
# ---------------------------------------------------------------------------

def implied_team_total(game_total: float, team_spread: float) -> Optional[float]:
    """Implied points for a team given the game total and that team's spread.

    A favorite laying 6 in a 44.5-point game is implied for
    44.5/2 - (-6)/2 = 22.25 + 3 = 25.25. Returns None on bad inputs.
    """
    try:
        gt = float(game_total)
        sp = float(team_spread)
    except (TypeError, ValueError):
        return None
    if gt <= 0:
        return None
    return round(gt / 2.0 - sp / 2.0, 1)


def total_tag(implied: Optional[float]) -> Optional[dict]:
    """Chip for an implied team total, or None if unremarkable/unknown.

    Thresholds are league-average anchored (~22-23 implied is a normal team
    total): >= 26 is a strong spot, <= 18 is a dud, the wide middle is unmarked.
    """
    if implied is None:
        return None
    label = f"{implied:g} implied"
    if implied >= 26:
        return {"label": label, "kind": "high", "note": "High team total (Vegas)"}
    if implied <= 18:
        return {"label": label, "kind": "low", "note": "Low team total (Vegas)"}
    return {"label": label, "kind": "mid", "note": "Vegas implied team total"}


def weather_tag(
    dome: bool,
    temp_f: Optional[float],
    wind_mph: Optional[float],
    precip_pct: Optional[float],
) -> Optional[dict]:
    """Notable-weather chip for an outdoor game, or None.

    Domes and benign conditions return None. Priority: wind (most impactful on
    passing/kicking) > precipitation > hard cold. Thresholds are the points where
    fantasy production is actually affected.
    """
    if dome:
        return None
    parts = []
    kind = "weather"
    if wind_mph is not None and wind_mph >= 15:
        parts.append(f"{round(wind_mph)} mph wind")
        kind = "wind"
    if precip_pct is not None and precip_pct >= 60:
        parts.append("rain/snow")
        if kind != "wind":
            kind = "precip"
    if temp_f is not None and temp_f <= 25:
        parts.append(f"{round(temp_f)}°")
        if kind == "weather":
            kind = "cold"
    if not parts:
        return None
    return {"label": " · ".join(parts), "kind": kind, "note": "Notable weather"}


def parse_tank01_odds(body) -> dict:
    """Parse a Tank01 getNFLBettingOdds `body` into per-team totals/spreads.

    Tank01 keys the body by gameID; each game carries team abbreviations and,
    either at the top level or under a sportsbook sub-dict, a total and home/away
    spreads. We read defensively across the field names Tank01 has used
    (totalUnder/totalOver/total, homeTeamSpread/awayTeamSpread) and take the first
    sportsbook when a `sportsBookOdds` list/dict is present.

    Returns ``{TEAM: {"total": float, "spread": float, "implied": float}}``.
    """
    out: dict = {}
    if not isinstance(body, dict):
        return out
    for _gid, game in body.items():
        if not isinstance(game, dict):
            continue
        home = _norm(game.get("homeTeam") or game.get("home") or game.get("teamAbvHome"))
        away = _norm(game.get("awayTeam") or game.get("away") or game.get("teamAbvAway"))
        if not home or not away:
            continue
        odds = _first_book(game)
        total = _to_float(
            odds.get("totalOver") or odds.get("total") or odds.get("totalUnder")
        )
        home_sp = _to_float(odds.get("homeTeamSpread") or odds.get("homeSpread"))
        away_sp = _to_float(odds.get("awayTeamSpread") or odds.get("awaySpread"))
        # If only one spread is present, the other is its negation.
        if home_sp is None and away_sp is not None:
            home_sp = -away_sp
        if away_sp is None and home_sp is not None:
            away_sp = -home_sp
        if total is None:
            continue
        for team, sp in ((home, home_sp), (away, away_sp)):
            if sp is None:
                continue
            out[team] = {
                "total": total,
                "spread": sp,
                "implied": implied_team_total(total, sp),
            }
    return out


def parse_open_meteo_daily(payload, index: int = 0) -> Optional[dict]:
    """Extract {temp_f, wind_mph, precip_pct} from an Open-Meteo daily forecast.

    ``index`` selects the day offset in the returned arrays. Returns None if the
    payload is missing the expected fields.
    """
    if not isinstance(payload, dict):
        return None
    daily = payload.get("daily") or {}
    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    winds = daily.get("wind_speed_10m_max") or []
    precip = daily.get("precipitation_probability_max") or []
    if index >= len(highs) or index >= len(winds):
        return None
    hi = _to_float(highs[index]) if index < len(highs) else None
    lo = _to_float(lows[index]) if index < len(lows) else None
    # Game-relevant temp: afternoon/evening runs closer to the daily high; use a
    # high-low blend so a frigid low doesn't overstate a mild-afternoon game.
    temp = None
    if hi is not None and lo is not None:
        temp = round(0.65 * hi + 0.35 * lo, 1)
    elif hi is not None:
        temp = hi
    return {
        "temp_f": temp,
        "wind_mph": _to_float(winds[index]) if index < len(winds) else None,
        "precip_pct": _to_float(precip[index]) if index < len(precip) else None,
    }


# ---------------------------------------------------------------------------
# small internal utils
# ---------------------------------------------------------------------------

def _to_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm(team) -> str:
    from utils.nfl import normalize_team
    return normalize_team(team) if team else ""


def _first_book(game: dict) -> dict:
    """Return the odds dict for a game, unwrapping a sportsBookOdds list/dict."""
    sbo = game.get("sportsBookOdds")
    if isinstance(sbo, list) and sbo and isinstance(sbo[0], dict):
        return sbo[0]
    if isinstance(sbo, dict) and sbo:
        first = next(iter(sbo.values()))
        if isinstance(first, dict):
            return first
    return game  # odds live at the top level


# ---------------------------------------------------------------------------
# Network fetchers (cached, never raise)
# ---------------------------------------------------------------------------

_ODDS_CACHE: dict = {}
_ODDS_TTL = 60 * 60          # betting lines move all week; refresh hourly
_WEATHER_CACHE: dict = {}
_WEATHER_TTL = 60 * 60 * 3   # forecasts change slowly; refresh every few hours


def fetch_week_odds(season: int, week: int, game_dates: "list[str]") -> dict:
    """Vegas implied team totals via SportsGameOdds.

    Returns ``{TEAM: {"implied": float}}`` keyed by canonical team abbreviation,
    or ``{}`` when the provider is unconfigured/unreachable. Never raises.
    The SGO client caches API responses on disk for 1h, so repeated page loads
    within the hour do not hit the provider again.
    """
    try:
        from dashboard_services.market_intelligence.client import SportsGameOddsClient
        from dashboard_services.market_intelligence.team import build_team_environments
    except Exception:
        logger.debug("[game_conditions] market_intelligence import failed", exc_info=True)
        return {}
    try:
        client = SportsGameOddsClient()
        if not client.configured:
            return {}
        dates = sorted(d for d in (game_dates or []) if d)
        if not dates:
            return {}
        from datetime import datetime, timedelta, timezone
        start = datetime.strptime(dates[0], "%Y%m%d").replace(tzinfo=timezone.utc) - timedelta(days=1)
        end = datetime.strptime(dates[-1], "%Y%m%d").replace(tzinfo=timezone.utc) + timedelta(days=2)
        events = list(client.iter_nfl_events(
            starts_after=start.isoformat(),
            starts_before=end.isoformat(),
        ))
        envs = build_team_environments(events) or {}
        return {
            team: {"implied": float(data["implied_points"])}
            for team, data in envs.items()
            if data.get("implied_points") is not None
        }
    except Exception:
        logger.debug("[game_conditions] SportsGameOdds odds fetch failed", exc_info=True)
        return {}

def fetch_game_weather(lat: float, lon: float, game_date: str, today: "Optional[str]" = None) -> Optional[dict]:
    """Weather for a stadium on a game date via Open-Meteo. Never raises.

    ``game_date`` / ``today`` are YYYYMMDD strings; the day offset picks the right
    entry from the daily forecast. Returns parsed {temp_f, wind_mph, precip_pct}
    or None (past date, out of forecast range, or any failure).
    """
    ck = (round(float(lat), 3), round(float(lon), 3), str(game_date))
    hit = _WEATHER_CACHE.get(ck)
    if hit and time.time() - hit[0] < _WEATHER_TTL:
        return hit[1]
    offset = _day_offset(game_date, today)
    if offset is None or offset < 0 or offset > 15:
        return None  # only within Open-Meteo's ~16-day forecast horizon
    parsed = None
    try:
        import requests
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,wind_speed_10m_max,precipitation_probability_max",
            "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
            "forecast_days": min(offset + 1, 16), "timezone": "America/New_York",
        }
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            parsed = parse_open_meteo_daily(resp.json(), offset)
    except Exception:
        logger.debug("[game_conditions] weather fetch failed", exc_info=True)
        return None
    _WEATHER_CACHE[ck] = (time.time(), parsed)
    return parsed


def build_week_conditions(
    season: int,
    week: int,
    week_games: "list[tuple]",
    *,
    today: "Optional[str]" = None,
    fetch_weather: bool = True,
) -> dict:
    """Per-team Vegas + weather conditions for a week's games. Never raises.

    ``week_games`` is a list of ``(home_team, away_team, game_date)`` tuples
    (game_date = YYYYMMDD). Returns ``{TEAM: {"implied_total": float|None,
    "weather": tag|None}}`` for every team with a game. Weather is looked up once
    per outdoor venue (domes skipped) and shared by both teams in the game;
    lookups run concurrently and are cached, so a warmed request is instant.
    """
    from utils.nfl import game_environment, normalize_team, stadium_coords

    out: dict = {}
    try:
        game_dates = [gd for _h, _a, gd in week_games if gd]
        odds = fetch_week_odds(season, week, game_dates)

        # One weather lookup per distinct outdoor home venue.
        venue_weather: dict = {}
        if fetch_weather:
            jobs = {}
            for home, _away, gd in week_games:
                hn = normalize_team(home)
                if hn in jobs:
                    continue
                env = game_environment(hn)
                coords = stadium_coords(hn)
                if env and not env.get("dome") and coords:
                    jobs[hn] = (coords[0], coords[1], gd)
            if jobs:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as ex:
                    futs = {
                        ex.submit(fetch_game_weather, lat, lon, gd, today): hn
                        for hn, (lat, lon, gd) in jobs.items()
                    }
                    for fut in futs:
                        hn = futs[fut]
                        try:
                            venue_weather[hn] = fut.result()
                        except Exception:
                            venue_weather[hn] = None

        for home, away, _gd in week_games:
            hn, an = normalize_team(home), normalize_team(away)
            wx = venue_weather.get(hn)  # both teams share the host venue's weather
            wx_tag = None
            if wx:
                wx_tag = weather_tag(False, wx.get("temp_f"), wx.get("wind_mph"), wx.get("precip_pct"))
            for team in (hn, an):
                if not team:
                    continue
                out[team] = {
                    "implied_total": (odds.get(team) or {}).get("implied"),
                    "weather": wx_tag,
                }
    except Exception:
        logger.debug("[game_conditions] build_week_conditions failed", exc_info=True)
        return out
    return out


def _day_offset(game_date: "Optional[str]", today: "Optional[str]") -> Optional[int]:
    """Whole-day offset between two YYYYMMDD strings (game_date - today)."""
    from datetime import datetime
    if not game_date:
        return None
    try:
        gd = datetime.strptime(str(game_date), "%Y%m%d").date()
        if today:
            td = datetime.strptime(str(today), "%Y%m%d").date()
        else:
            td = datetime.now().date()
        return (gd - td).days
    except (ValueError, TypeError):
        return None


# ======================================================================
# From utils/qb_situation.py
# ======================================================================

"""Backup/third-string QB signal for start/sit rows.

Reads the live Sleeper QB depth chart for a team and finds the effective
starter: the lowest depth_chart_order QB expected to play (same "will play"
rule the waiver depth-chart logic uses). When that is the QB2 or deeper, the
team's pass-catchers are catching passes from a backup, which is the signal
this chip carries.

Pure stdlib + utils.waiver_score so it stays importable without Flask.
"""


from utils.waivers import _will_play


def qb_situation_chip(team: Optional[str], depth_index: dict,
                      full_players: dict) -> Optional[dict]:
    """Return a ``{"label", "kind", "note"}`` chip dict, or None.

    Returns None when the QB1 is starting, when the chart is missing or
    incomplete, or on any error, so callers degrade to no chip instead of a
    wrong one.
    """
    if not team or not depth_index:
        return None
    try:
        group = (depth_index or {}).get((str(team).upper(), "QB")) or []
        ranked = []
        for g in group:
            try:
                order = int(g.get("depth_order"))
            except (TypeError, ValueError):
                continue
            ranked.append((order, g))
        if not ranked:
            return None
        ranked.sort(key=lambda t: t[0])
        starter = next((g for _, g in ranked if _will_play(g.get("status"))), None)
        if not starter:
            return None
        order = next(o for o, g in ranked if g is starter)
        if order <= 1:
            return None
        pid = str(starter.get("pid") or "")
        pdata = (full_players or {}).get(pid) or {}
        name = (pdata.get("full_name") or "").strip()
        if order == 2:
            return {"label": "Backup QB", "kind": "qb2",
                    "note": f"{name or 'The backup'} (QB2) is starting for "
                            f"{str(team).upper()} with the starter out"}
        return {"label": "3rd-string QB", "kind": "qb3",
                "note": f"{name or 'A third-stringer'} (QB{order}) is starting for "
                        f"{str(team).upper()} with the top QBs out"}
    except Exception:
        return None
