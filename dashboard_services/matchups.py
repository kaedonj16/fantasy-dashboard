from __future__ import annotations

import logging
import html
import json
from datetime import datetime, date
from itertools import zip_longest
from pathlib import Path
from typing import Dict, List, Any, Optional

from dashboard_services.api import avatar_from_users, team_avatar, get_nfl_scores_for_date, build_team_game_lookup, \
    get_league_settings
from dashboard_services.platform_api import (
    get_matchups,
    get_users,
    get_rosters,
    get_bracket
)
from utils.utils import (
    write_json,
    load_week_schedule,
    load_teams_index,
    load_week_stats,
    normalize_name,
    from_players_map,
    game_has_started,
    lookup_team_map,
    team_abbr_keys,
    canon_team,
    canonical_teams_index,
    box_score_line_is_trusted,
    player_week_stat_entry,
    overlay_idp_and_k_stats_from_sleeper,
    load_sleeper_week_stats,
)
from utils.matchup_schedule import lineup_from_roster, _starters_look_like_full_roster
from utils.week_proj import week_proj_map_from_bundles as _week_proj_map_from_bundles

STATUS_NOT_STARTED = "not_started"
STATUS_IN_PROGRESS = "in_progress"
STATUS_FINAL = "final"

logger = logging.getLogger(__name__)


def _week_stats_for_slide(season, w) -> dict:
    """Week stats for a matchup slide, with a lazy K/IDP/DEF overlay.

    Completed weeks that were cached before the Sleeper stats-file lookup was
    fixed hold only QB/RB/WR/TE lines, so every kicker and defense rendered as
    "Stats unavailable" and skill players missing from Footballguys had no line
    at all. Those on-disk snapshots are only rebuilt for the live week, so past
    weeks would stay broken until a manual backfill. Overlay the K/IDP/DEF (and
    any missing skill) lines from the week's Sleeper snapshot in memory when they
    are absent -- the underlying index and Sleeper reads are mtime-cached, so
    this stays cheap across the several slides on a page and is never written
    back to disk.
    """
    week_stats = load_week_stats(season, w) or {}
    if not week_stats:
        return week_stats
    has_k_or_idp = any(
        isinstance(v, dict) and (v.get("K") or v.get("IDP"))
        for v in week_stats.values()
    )
    if has_k_or_idp:
        return week_stats
    try:
        overlay_idp_and_k_stats_from_sleeper(
            league_week_stats=week_stats,
            season=int(season),
            week=int(w),
            teams_index=load_teams_index() or {},
        )
    except Exception:
        logger.info(
            "[matchups] K/IDP/DEF overlay skipped for season=%s week=%s",
            season, w, exc_info=True,
        )
    return week_stats

# NFL regulation clock: 4 quarters of 15 minutes.
_QUARTERS = 4
_QUARTER_MINUTES = 15.0
_REGULATION_MINUTES = _QUARTERS * _QUARTER_MINUTES  # 60


def _parse_quarter(period_raw: Any) -> Optional[int]:
    """Parse a game's current quarter. Returns 1-4 for regulation, 5+ for OT,
    or None when the period can't be read."""
    s = str(period_raw or "").strip().upper()
    if not s:
        return None
    if s.startswith("OT") or "OVERTIME" in s:
        return _QUARTERS + 1
    if "HALF" in s:  # "Halftime" / "Half" -> end of the 2nd quarter
        return 2
    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits)
    except ValueError:
        return None


def _parse_clock_minutes(clock_raw: Any) -> Optional[float]:
    """Minutes (as a float) left on the game clock in the current quarter, from a
    'MM:SS' string. Returns None when the clock can't be parsed."""
    s = str(clock_raw or "").strip().upper()
    if not s:
        return None
    if "HALF" in s or "END" in s:
        # Between-period text ("Halftime", "End of 2nd"): no time on the clock.
        return 0.0
    if ":" not in s:
        return None
    mm, _, ss = s.partition(":")
    try:
        minutes = int(mm)
        seconds = int("".join(ch for ch in ss if ch.isdigit()) or 0)
    except ValueError:
        return None
    return minutes + seconds / 60.0


def game_fraction_remaining(game: Optional[dict]) -> Optional[float]:
    """Fraction of NFL regulation time left in a game, in [0.0, 1.0].

    1.0 = not yet kicked off, 0.0 = final. Used to blend a player's live actual
    points with their remaining pregame projection into a live projected final:
    ``live_final = actual + pregame_proj * fraction_remaining``.

    Returns None when the game is live but its clock/quarter can't be read, so
    callers can fall back to prior behaviour instead of guessing.
    """
    if not isinstance(game, dict):
        return None
    code = str(game.get("gameStatusCode") or "").strip()
    if code == "2":  # final
        return 0.0
    if code == "0":  # scheduled / not started
        return 1.0

    # Live (code "1"), or unknown code but possibly carrying clock data.
    ls = game.get("lineScore") if isinstance(game.get("lineScore"), dict) else {}
    period = _parse_quarter(ls.get("period") or ls.get("quarter"))
    clock = _parse_clock_minutes(game.get("gameClock"))

    if period is None:
        # Live game with no usable quarter signal: don't guess.
        return None
    if period > _QUARTERS:
        # Overtime: regulation is spent; treat as essentially over.
        return 0.02

    mins_left_in_quarter = clock if clock is not None else _QUARTER_MINUTES
    mins_left_in_quarter = max(0.0, min(_QUARTER_MINUTES, mins_left_in_quarter))
    remaining = mins_left_in_quarter + (_QUARTERS - period) * _QUARTER_MINUTES
    return max(0.0, min(1.0, remaining / _REGULATION_MINUTES))


def resolve_team_game(
    nfl: Any,
    team_game_lookup: Optional[dict],
    team_schedule_lookup: Optional[dict] = None,
) -> Optional[dict]:
    """Find an NFL team's game dict, preferring the live scoreboard lookup and
    falling back to the static week schedule."""
    if not nfl:
        return None
    tc = str(nfl).upper()
    game = None
    if team_game_lookup:
        game = lookup_team_map(team_game_lookup, tc)
    if game is None and team_schedule_lookup:
        game = lookup_team_map(team_schedule_lookup, tc)
    return game


# --- Live projected-finish model -------------------------------------------
# A player's live projected final = points already banked + expected remaining.
# The naive estimate for "remaining" is the pregame projection scaled by the
# fraction of the game left (a pure clock model). That ignores how the player is
# actually doing: a back getting no touches still projects his full pregame rate,
# and a receiver on a tear projects as if he'd cooled off. We sharpen it by
# blending that pregame-rate estimate with the player's *observed* in-game
# scoring pace, trusting the observed pace more as more of the game is played
# (more snaps seen -> more signal). Early on it stays pregame-dominated so one
# fluke play doesn't send the number flying.
_PACE_WEIGHT_EXP = 2.0     # >1 slows how fast observed pace overtakes pregame
_PACE_MIN_ELAPSED = 0.05   # ignore pace until ~3 game-minutes have been played
# Skill positions accumulate points continuously enough to extrapolate; kicker /
# defense / IDP scoring is too lumpy (one FG, one pick-six) to read as a "pace".
_PACE_POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


def live_final_from_frac(
    actual: float,
    pregame_proj: float,
    frac: Any,
    pos: str = "",
) -> float:
    """Live projected final points, given the fraction of regulation remaining.

    Blends the pregame-rate estimate of remaining production with the player's
    observed in-game fantasy pace, weighting pace more heavily the further the
    game has progressed. Skill positions only; K/DEF/IDP fall back to the pure
    pregame-rate (clock) estimate, and an unreadable fraction returns the
    pregame projection unchanged.
    """
    try:
        r = float(frac)
    except (TypeError, ValueError):
        return float(pregame_proj or 0.0)
    r = max(0.0, min(1.0, r))
    actual = float(actual or 0.0)
    pregame_proj = float(pregame_proj or 0.0)

    pregame_remaining = pregame_proj * r          # rest of game at pregame rate
    elapsed = 1.0 - r
    if pos.upper() not in _PACE_POSITIONS or elapsed <= _PACE_MIN_ELAPSED:
        return actual + pregame_remaining

    pace_remaining = actual * (r / elapsed)       # rest of game at observed rate
    w = min(1.0, elapsed ** _PACE_WEIGHT_EXP)     # trust pace more as game plays
    remaining = (1.0 - w) * pregame_remaining + w * pace_remaining
    return actual + remaining


def live_projected_final(
    actual: float,
    pregame_proj: float,
    game: Optional[dict],
    *,
    pos: str = "",
) -> float:
    """Live projected final for a player, resolving game progress from ``game``.

    Falls back to the pregame projection when progress is unknown, and to the
    actual once the game is final (fraction 0)."""
    frac = game_fraction_remaining(game)
    if frac is None:
        return float(pregame_proj or 0.0)
    return live_final_from_frac(actual, pregame_proj, frac, pos)


def make_frac_lookup(
    team_game_lookup: Optional[dict],
    team_schedule_lookup: Optional[dict] = None,
):
    """Build a ``fn(starter_dict) -> Optional[float]`` that returns the fraction
    of regulation left in that starter's NFL game (None when undeterminable).
    Used to make live totals and win probability track games as they play."""

    def _frac(p: Optional[dict]) -> Optional[float]:
        game = resolve_team_game(
            (p or {}).get("nfl"), team_game_lookup, team_schedule_lookup,
        )
        return game_fraction_remaining(game)

    return _frac


def _proj_value_for_pid(
    week_proj_map: Dict[str, Any],
    pid: Any,
    *,
    raw_week_map: Optional[Dict[str, Any]] = None,
    scoring_settings: Optional[Dict[str, Any]] = None,
    pos: str = "",
) -> float:
    """Resolve one player's weekly projection, with Scout-style raw fallback."""
    if pid is None:
        return 0.0
    key = str(pid)
    if key in week_proj_map:
        try:
            return float(week_proj_map.get(key) or 0.0)
        except (TypeError, ValueError):
            pass
    if pid in week_proj_map and pid != key:
        try:
            return float(week_proj_map.get(pid) or 0.0)
        except (TypeError, ValueError):
            pass
    if raw_week_map:
        try:
            from utils.fantasy_scoring import weekly_projection_points
            pts = weekly_projection_points(
                raw_week_map, key, scoring_settings or {}, pos or "",
            )
            if pts is not None:
                return float(pts)
        except Exception:
            logging.getLogger(__name__).debug(
                "suppressed projection fallback", exc_info=True,
            )
    return 0.0


def _allow_live_game_indicators(viewed_season) -> bool:
    """Live pulse/dot only during regular season or playoffs for the current NFL year."""
    from dashboard_services.api import get_nfl_state

    state = get_nfl_state() or {}
    nfl_season = int(state.get("season") or 0)
    season_type = (state.get("season_type") or "").lower()
    try:
        viewed = int(viewed_season or 0)
    except (TypeError, ValueError):
        return False
    if not nfl_season or viewed != nfl_season:
        return False
    return season_type in ("reg", "post")


def _synthetic_week_matchups(rosters: List[dict], week: int) -> List[dict]:
    from utils.matchup_schedule import synthetic_week_matchups
    return synthetic_week_matchups(rosters, week)


def build_matchup_preview(
        league_id: str,
        week: int,
        roster_map: Dict[str, str],
        players_map: Dict[str, Dict[str, str]],
        season: str,
        platform: str
) -> List[dict]:
    try:
        mlist = get_matchups(platform, league_id, week, season) or []
    except Exception as e:
        # 404s are expected for future weeks - log at debug level without traceback
        # Other errors get full warning with traceback for investigation
        is_404 = getattr(e, 'response', None) and getattr(e.response, 'status_code', None) == 404
        logger = logging.getLogger(__name__)
        if is_404:
            logger.debug(
                "get_matchups 404 (future week) platform=%s league=%s week=%s; synthesizing",
                platform, league_id, week,
            )
        else:
            logger.warning(
                "get_matchups failed platform=%s league=%s week=%s; synthesizing",
                platform, league_id, week, exc_info=True,
            )
        mlist = []

    # Pre-fetch users/rosters once instead of per team
    users = get_users(platform, league_id, season) or []
    rosters = get_rosters(platform, league_id, season) or []

    # Dynasty leagues often keep a pending rookie draft after the NFL season
    # starts. Some platforms still return an empty matchup feed for Week 1+
    # until that draft finishes. Synthesize a deterministic round-robin so the
    # Season Hub Matchups tab is never blank once the league is in-season.
    # Yahoo publishes the full-season scoreboard; inventing opponents here is
    # how Yahoo pairings drifted from the real Yahoo matchup page.
    if not mlist and rosters and str(platform or "").lower() != "yahoo":
        mlist = _synthetic_week_matchups(rosters, week)
        if not mlist:
            return []
    elif not mlist:
        return []

    # Pull league settings to find playoff start week
    settings = get_league_settings() or {}
    playoff_week_start = int(settings.get("playoff_week_start") or 0)

    # Brackets (always defined)
    winners_bracket: List[dict] = []
    losers_bracket: List[dict] = []
    if playoff_week_start and week >= playoff_week_start:
        winners_bracket = get_bracket(platform, league_id, "winners", season) or []
        losers_bracket = get_bracket(platform, league_id, "losers", season) or []

    # figure out league size from rosters / roster_map
    if rosters:
        num_teams = len({str(r.get("roster_id")) for r in rosters})
    else:
        num_teams = len(roster_map) if roster_map else 0

    # expected number of head-to-head games in regular season
    expected_matchups = max(1, num_teams // 2) if num_teams else None

    # Precompute maps for fast lookup
    owner_id_by_rid: Dict[str, Optional[str]] = {}
    record_by_rid: Dict[str, tuple[int, int]] = {}
    for r in rosters:
        rid_str = str(r.get("roster_id"))
        owner_id_by_rid[rid_str] = r.get("owner_id")
        r_settings = r.get("settings") or {}
        record_by_rid[rid_str] = (
            r_settings.get("wins", 0),
            r_settings.get("losses", 0),
        )

    # Standings rank (1 = best) for the matchup header meta, e.g.
    # "2-0 • @hoodiekj1 (#2)". Sorted by wins, then points-for; teams tied
    # on both share the rank (competition ranking).
    def _roster_sort_key(r: dict) -> tuple:
        s = r.get("settings") or {}
        fpts = float(s.get("fpts", 0) or 0) + float(s.get("fpts_decimal", 0) or 0) / 100.0
        return (-int(s.get("wins", 0) or 0), -fpts)

    rank_by_rid: Dict[str, int] = {}
    _prev_key: Optional[tuple] = None
    for _i, _r in enumerate(sorted(rosters, key=_roster_sort_key), start=1):
        _key = _roster_sort_key(_r)
        if _key != _prev_key:
            _rank = _i
            _prev_key = _key
        rank_by_rid[str(_r.get("roster_id"))] = _rank

    username_by_owner: Dict[str, Optional[str]] = {
        u["user_id"]: u.get("display_name") for u in users if "user_id" in u
    }
    users_by_rid: Dict[str, dict] = {
        str(u.get("roster_id")): u for u in users if u.get("roster_id") is not None
    }

    avatar_cache: Dict[str, Any] = {}
    roster_by_owner: Dict[Optional[str], dict] = {r.get("owner_id"): r for r in rosters}
    roster_by_rid: Dict[str, dict] = {str(r.get("roster_id")): r for r in rosters}

    def get_avatar_for_rid(rid: str) -> Any:
        if rid not in avatar_cache:
            avatar_cache[rid] = team_avatar(platform, roster_by_rid.get(rid), users)
        return avatar_cache[rid]

    def _to_int(x) -> Optional[int]:
        try:
            return int(x)
        except (TypeError, ValueError):
            return None


    def _pinfo(pid: str, pts_map: Dict[str, float]) -> dict:
        base = from_players_map(pid, players_map)
        pts = pts_map.get(pid) if pts_map else None
        return {
            "pid": pid,
            "name": base["name"],
            "pos": base["pos"],
            "nfl": base["nfl"],
            "pts": pts,
        }

    def _team_block_from_match_row(row: dict) -> dict:
        rid = str(row.get("roster_id"))
        starters_raw = [s for s in (row.get("starters") or []) if s]
        # Fleaflicker uses "0" for empty/unresolved boxscore slots. Those are
        # placeholders, not real lineup data; exclude them from the historical
        # check so unresolved boxscores don't masquerade as valid lineups.
        real_starters = [s for s in starters_raw if str(s) != "0"]
        # Keep whether the provider supplied a real weekly lineup before the
        # display-only current-roster fallback below. Historical consumers must
        # never mistake today's roster for the lineup started in an earlier week.
        lineup_is_historical = bool(real_starters) and not _starters_look_like_full_roster(
            real_starters, row.get("players") or []
        )
        starter_set = {str(s) for s in starters_raw}
        all_players = [str(p) for p in (row.get("players") or []) if p]
        bench_raw = [p for p in all_players if p not in starter_set]
        pts_map = {str(k): v for k, v in (row.get("players_points") or {}).items()}

        roster = roster_by_rid.get(rid) or {}
        if not roster:
            oid = owner_id_by_rid.get(rid)
            if oid:
                roster = next((r for r in rosters if str(r.get("owner_id")) == str(oid)), {})
        # A provider's week-specific player list is authoritative.  Never
        # replace an empty/malformed historical assignment with today's roster:
        # that silently turns traded players and bench slots into starters.
        if _starters_look_like_full_roster(starters_raw, all_players):
            starters_raw = []
            starter_set = set()
            bench_raw = list(all_players)
        need_roster_lineup = not starters_raw and not all_players
        if need_roster_lineup and roster:
            starters_raw, bench_raw = lineup_from_roster(roster)
            all_players = [str(p) for p in (roster.get("players") or []) if p] or all_players

        s_infos: List[dict] = [_pinfo(str(pid), pts_map) for pid in starters_raw]
        b_infos: List[dict] = [_pinfo(str(pid), pts_map) for pid in bench_raw]
        pts_total = float(row["points"]) if isinstance(row.get("points"), (int, float)) else None
        proj_total = None
        raw_proj = row.get("projected_points")
        if isinstance(raw_proj, (int, float)):
            proj_total = float(raw_proj)

        wins, losses = record_by_rid.get(rid, (0, 0))
        owner_id = owner_id_by_rid.get(rid)
        user = users_by_rid.get(rid) or (
            next((u for u in users if u.get("user_id") == owner_id), None) if owner_id else None
        )
        username = (user or {}).get("display_name")

        return {
            "name": roster_map.get(rid, f"Roster {rid}"),
            "roster_id": rid,
            "owner_id": owner_id,
            "starters": s_infos,
            "lineup_is_historical": lineup_is_historical,
            "bench": b_infos,
            "pts_total": pts_total,
            "proj_total": proj_total,
            "avatar": get_avatar_for_rid(rid),
            "record": f"{wins}-{losses}",
            "rank": rank_by_rid.get(rid),
            "username": username,
        }

    def _team_block_tbd(rid: Optional[int | str]) -> dict:
        rid_str = str(rid) if rid is not None else None
        name = "TBD"
        record = "-"

        if rid_str and rid_str in roster_map:
            name = roster_map[rid_str]
        if rid_str and rid_str in record_by_rid:
            w, l = record_by_rid[rid_str]
            record = f"{w}-{l}"

        owner_id = owner_id_by_rid.get(rid_str)
        user = users_by_rid.get(rid_str) if rid_str else None
        return {
            "name": name,
            "roster_id": rid_str,
            "owner_id": owner_id,
            "starters": [],
            "pts_total": None,
            "avatar": get_avatar_for_rid(rid_str) if rid_str else None,
            "record": record,
            "rank": rank_by_rid.get(rid_str) if rid_str else None,
            "username": (user or {}).get("display_name") if user else (
                username_by_owner.get(owner_id) if owner_id else None
            ),
        }

    # ------------------------------------------------------------------
    # PLAYOFF BRANCH – winners + losers bracket
    # ------------------------------------------------------------------
    is_playoff_week = (
            bool(playoff_week_start)
            and week >= playoff_week_start
            and (winners_bracket or losers_bracket)
    )

    if is_playoff_week:
        # Map roster_id -> matchup row for this fantasy week
        by_rid: Dict[str, dict] = {}
        for row in mlist:
            rid_str = str(row.get("roster_id"))
            if rid_str not in by_rid:
                by_rid[rid_str] = row

        all_brackets = list(winners_bracket) + list(losers_bracket)

        # Determine which bracket rounds exist, and map fantasy week offset to a round.
        rounds_present = sorted({r for r in (_to_int(b.get("r")) for b in all_brackets) if r is not None})
        if not rounds_present:
            # Brackets exist but malformed; fall back to regular season grouping
            is_playoff_week = False
        else:
            week_offset = max(0, week - playoff_week_start)  # 0 for first playoff week
            idx = min(week_offset, len(rounds_present) - 1)
            current_round = rounds_present[idx]

            # result_by_match[m] = {"w": roster_id or None, "l": roster_id or None}
            result_by_match: Dict[int, Dict[str, Optional[int]]] = {}

            for b in all_brackets:
                mid = _to_int(b.get("m"))
                if mid is None:
                    continue

                entry = result_by_match.setdefault(mid, {"w": None, "l": None})

                w_team = _to_int(b.get("w"))
                l_team = _to_int(b.get("l"))

                if w_team is not None:
                    entry["w"] = w_team
                if l_team is not None:
                    entry["l"] = l_team

            def _resolve_slot(b: dict, slot_key: str, from_key: str) -> Optional[int]:
                """
                Resolve t1 / t2 from either a direct value or a from-spec:
                  - tX: direct roster id
                  - tX_from: {"w": match_no} or {"l": match_no}
                """
                direct = _to_int(b.get(slot_key))
                if direct is not None:
                    return direct

                from_spec = b.get(from_key)
                if not isinstance(from_spec, dict) or not from_spec:
                    return None

                if "w" in from_spec:
                    prev_m = _to_int(from_spec.get("w"))
                    if prev_m is None:
                        return None
                    return result_by_match.get(prev_m, {}).get("w")

                if "l" in from_spec:
                    prev_m = _to_int(from_spec.get("l"))
                    if prev_m is None:
                        return None
                    return result_by_match.get(prev_m, {}).get("l")

                return None

            def _build_round_matchups(bracket_list: List[dict]) -> List[dict]:
                out_matches: List[dict] = []
                for b in bracket_list:
                    if _to_int(b.get("r")) != current_round:
                        continue

                    mid = b.get("m")  # keep original (can be str/int)
                    t1_rid = _resolve_slot(b, "t1", "t1_from")
                    t2_rid = _resolve_slot(b, "t2", "t2_from")

                    left_row = by_rid.get(str(t1_rid)) if t1_rid is not None else None
                    right_row = by_rid.get(str(t2_rid)) if t2_rid is not None else None

                    left = _team_block_from_match_row(left_row) if left_row is not None else _team_block_tbd(t1_rid)
                    right = _team_block_from_match_row(right_row) if right_row is not None else _team_block_tbd(t2_rid)

                    out_matches.append({
                        "matchup_id": mid,
                        "left": left,
                        "right": right,
                    })
                return out_matches

            playoff_out: List[dict] = []
            playoff_out.extend(_build_round_matchups(winners_bracket))
            playoff_out.extend(_build_round_matchups(losers_bracket))

            # In playoffs we want all games we can render. No capping.
            return playoff_out

    # ------------------------------------------------------------------
    # REGULAR SEASON BRANCH – existing logic
    # ------------------------------------------------------------------
    by_mid: Dict[Any, List[dict]] = {}
    for m in mlist:
        mid = m.get("matchup_id")
        by_mid.setdefault(mid, []).append(m)

    out: List[dict] = []
    for mid, rows in by_mid.items():
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: str(r.get("roster_id")))
        left = _team_block_from_match_row(rows_sorted[0])
        right = (
            _team_block_from_match_row(rows_sorted[1])
            if len(rows_sorted) > 1
            else {
                "name": "TBD",
                "avatar": None,
                "starters": [],
                "pts_total": None,
                "record": "-",
                "username": None,
            }
        )
        out.append({"matchup_id": mid, "left": left, "right": right})

    if expected_matchups is not None:
        return out[:expected_matchups]
    return out


def _identity_part(value: Any) -> Optional[str]:
    """Normalize a provider id while rejecting absent/invalid id values."""
    if value is None or isinstance(value, bool):
        return None
    value = str(value).strip()
    if not value or value.lower() in {"none", "null", "nan", "undefined"}:
        return None
    return value


def normalized_matchup_identity(matchup: Optional[dict]) -> Optional[str]:
    """Return one canonical identity for a displayed matchup or GOTW record.

    Provider matchup ids are preferred.  Recap selections carry ``roster_ids``
    while rendered matchups carry ``left``/``right``; the order-independent
    team-pair key is the common fallback for all supported providers.
    """
    if not isinstance(matchup, dict):
        return None
    matchup_id = _identity_part(matchup.get("matchup_id"))
    if matchup_id is not None:
        return f"matchup:{matchup_id}"

    roster_ids = matchup.get("roster_ids")
    if not isinstance(roster_ids, (list, tuple)):
        roster_ids = [
            (matchup.get("left") or {}).get("roster_id"),
            (matchup.get("right") or {}).get("roster_id"),
        ]
    team_ids = [_identity_part(value) for value in roster_ids]
    if len(team_ids) != 2 or any(value is None for value in team_ids):
        return None
    if team_ids[0] == team_ids[1]:
        return None
    return "teams:" + ":".join(sorted(team_ids))


def gotw_identity_for_context(
        selection: Optional[dict], *, loaded: bool, platform: str,
        league_id: str, season: Any, week: int,
) -> Optional[str]:
    """Normalize a GOTW only after it is loaded and belongs to this view."""
    if not loaded or not isinstance(selection, dict):
        return None
    expected = {
        "platform": _identity_part(platform),
        "league_id": _identity_part(league_id),
        "season": _identity_part(season),
        "target_week": _identity_part(week),
    }
    for field, value in expected.items():
        actual = _identity_part(selection.get(field))
        if value is None or actual is None or actual.lower() != value.lower():
            return None
    return normalized_matchup_identity(selection)


def matchup_gotw_flags(matchups: List[dict], gotw_key: Optional[str]) -> List[bool]:
    """Return row-specific flags, defensively allowing at most one badge."""
    claimed = False
    flags = []
    for matchup in matchups:
        matchup_key = normalized_matchup_identity(matchup)
        is_game_of_week = bool(
            not claimed and gotw_key and matchup_key and gotw_key == matchup_key
        )
        flags.append(is_game_of_week)
        claimed = claimed or is_game_of_week
    return flags


def matchup_matches_gotw(matchup: dict, selection: Optional[dict]) -> bool:
    """Compatibility helper for callers that already hold a scoped selection."""
    gotw_key = normalized_matchup_identity(selection)
    matchup_key = normalized_matchup_identity(matchup)
    return bool(gotw_key and matchup_key and gotw_key == matchup_key)


def render_matchup_carousel_weeks(
        slides_by_week: dict[int, str],
        dashboard: bool,
        active_week: Optional[int] = None,
        title_href: Optional[str] = None,
        gotw_selection: Optional[dict] = None,
) -> str:
    """
    Render a single matchup carousel card.

    slides_by_week: {week: "<div class='m-slide'>...</div>..."}
    active_week: which week's slides to show inside the track.
    title_href: when set, the "Matchup Preview" heading links to the full page.
    """
    from html import escape as html_escape

    if not slides_by_week:
        slides_html = "<div class='m-empty'>No matchups</div>"
    else:
        # pick active week if given, else first key
        if active_week is None:
            active_week = sorted(slides_by_week.keys())[0]
        slides_html = slides_by_week.get(active_week) or "<div class='m-empty'>No matchups</div>"

    central = "central" if dashboard else ""
    compact_cls = " matchup-carousel--compact" if dashboard else ""
    # On the Weekly Hub (non-dashboard) let the carousel fill its grid column so
    # it stretches to the stats sidebar instead of leaving a gap beside it.
    style = ""

    if title_href:
        title_html = (
            "<h2>"
            f'<a class="os-section-title-link" href="{html_escape(title_href)}">'
            "Matchup Preview"
            ' <span class="os-section-title-arrow" aria-hidden="true">&rarr;</span></a></h2>'
        )
    else:
        title_html = "<h2>Matchup Preview</h2>"

    week_context_html = ""
    if active_week is not None and not dashboard:
        week_context_html = f"<div class='m-week-context'><span>Week {int(active_week)}</span></div>"

    return f"""
      <div class="card matchup-carousel {central}{compact_cls}" data-section="matchups" style="{style} margin-bottom:30px;">
        <div class="m-nav">
          <div>{title_html}{week_context_html}</div>
          <div class="m-controls">
            <button class="m-btn m-btn-prev" type="button">‹ Prev</button>
            <button class="m-btn m-btn-next" type="button">Next ›</button>
          </div>
        </div>
        <div class="m-carousel">
          <div class="m-track">
            {slides_html}
          </div>
        </div>
      </div>
    """


def team_live_totals(
        team: dict,
        status_by_pid: dict[str, str],
        projections: dict[str, float],
        *,
        proj_lookup=None,
        frac_lookup=None,
) -> tuple[float, float]:
    """
    actual_total:
        sum of all actual points for starters (p['pts'])
    live_proj_total:
        - players not started      -> use pregame projection
        - players in progress       -> live projected final
                                       (banked points + remaining projection)
        - players final             -> use actual

    ``frac_lookup(starter) -> Optional[float]`` supplies the fraction of that
    player's game still to play; when given, in-progress starters project their
    finish instead of freezing at their current points, so the team total climbs
    with the games. When it's absent (or can't read a game), in-progress
    starters fall back to their locked actual, the prior behaviour.

    Compact dashboard slides never render starter rows, so this total is the
    only projected number the user sees. Yahoo scoreboard rows often have no
    Sleeper-mapped starter projs; fall back to Yahoo ``proj_total``.
    """
    actual_total = 0.0
    live_proj_total = 0.0
    any_locked = False
    missing_locked = False

    starters = team.get("starters") or []
    projections = projections or {}

    for p in starters:
        pid = p.get("pid")

        status = status_by_pid.get(pid, STATUS_NOT_STARTED)
        if pid is not None and status == STATUS_NOT_STARTED:
            status = status_by_pid.get(str(pid), status)
        raw_actual = p.get("pts")
        actual_available = isinstance(raw_actual, (int, float)) and not isinstance(raw_actual, bool)
        actual = float(raw_actual) if actual_available else 0.0
        if actual_available:
            actual_total += actual
        elif status in (STATUS_FINAL, STATUS_IN_PROGRESS):
            missing_locked = True
        if proj_lookup:
            try:
                proj_val = float(proj_lookup(pid, p.get("pos") or "") or 0.0)
            except TypeError:
                proj_val = float(proj_lookup(pid) or 0.0)
        else:
            proj_val = _proj_value_for_pid(projections, pid)

        # Use == for strings, not `is`
        if status == STATUS_FINAL:
            any_locked = True
            live_proj_total += actual
        elif status == STATUS_IN_PROGRESS:
            any_locked = True
            frac = frac_lookup(p) if frac_lookup else None
            if frac is None:
                live_proj_total += actual
            else:
                live_proj_total += live_final_from_frac(
                    actual, proj_val, frac, p.get("pos") or "",
                )
        else:
            live_proj_total += proj_val

    if missing_locked:
        # Yahoo's scoreboard totals/projections are authoritative and remain
        # useful when one player-stat batch is partial. Never manufacture the
        # missing player's contribution as zero.
        try:
            if team.get("pts_total") is not None:
                actual_total = float(team["pts_total"])
        except (TypeError, ValueError):
            pass
        try:
            if team.get("proj_total") is not None:
                live_proj_total = float(team["proj_total"])
        except (TypeError, ValueError):
            pass
    elif live_proj_total == 0.0 and not any_locked:
        fallback = team.get("proj_total")
        try:
            if fallback is not None and float(fallback) > 0:
                live_proj_total = float(fallback)
        except (TypeError, ValueError):
            pass

    return actual_total, live_proj_total


def compute_win_prob(
        left: dict,
        right: dict,
        status_by_pid: dict[str, str],
        proj_map: dict[str, float],
        *,
        frac_lookup=None,
) -> float:
    """
    Returns left team win probability (0.0–1.0) based on locked scores
    and projected remaining points modelled as normal distributions.
    Variance per pending player: sigma = max(0.4 * projection, 4.0), and
    each team's pending variance is floored at (TEAM_CV_FLOOR * remaining
    projection)^2 so a full lineup's spread reflects real weekly team-score
    dispersion (CV ~0.24) instead of the far tighter figure the independent
    per-player sum produces.

    ``frac_lookup(starter) -> Optional[float]`` supplies the fraction of that
    player's game left to play. When given, an in-progress player banks the
    points already scored and carries only the remaining slice of their
    projection as pending, with variance that shrinks toward zero as the game
    ends. Without it, in-progress players are treated as fully locked at their
    current points, the prior behaviour.
    """
    from math import erf

    # Coefficient of variation a team's remaining points are assumed to carry.
    # ~0.24 matches observed weekly team-score dispersion; see the pending
    # variance floor in _stats.
    TEAM_CV_FLOOR = 0.24

    def _stats(team: dict):
        locked = 0.0
        pend_proj = 0.0
        pend_var = 0.0
        for p in (team.get("starters") or []):
            pid = p.get("pid")
            raw_actual = p.get("pts")
            if (
                status_by_pid.get(pid, status_by_pid.get(str(pid), STATUS_NOT_STARTED))
                in (STATUS_FINAL, STATUS_IN_PROGRESS)
                and not isinstance(raw_actual, (int, float))
            ):
                return None
            actual = float(raw_actual) if isinstance(raw_actual, (int, float)) else 0.0
            status = status_by_pid.get(pid, STATUS_NOT_STARTED)
            if pid is not None and status == STATUS_NOT_STARTED:
                status = status_by_pid.get(str(pid), status)
            proj = _proj_value_for_pid(proj_map, pid)
            if status == STATUS_FINAL:
                locked += actual
            elif status == STATUS_IN_PROGRESS:
                frac = frac_lookup(p) if frac_lookup else None
                if frac is None:
                    locked += actual
                else:
                    # Bank the points scored; carry the sharpened remaining
                    # (pace-blended) projection as the pending, uncertain slice.
                    locked += actual
                    final = live_final_from_frac(actual, proj, frac, p.get("pos") or "")
                    remaining_proj = max(0.0, final - actual)
                    pend_proj += remaining_proj
                    sigma = max(0.4 * remaining_proj, 4.0 * frac)
                    pend_var += sigma * sigma
            else:
                pend_proj += proj
                sigma = max(0.4 * proj, 4.0)
                pend_var += sigma * sigma
        if pend_proj == 0.0 and locked == 0.0:
            try:
                fallback = float(team.get("proj_total") or 0.0)
            except (TypeError, ValueError):
                fallback = 0.0
            if fallback > 0:
                pend_proj = fallback
                sigma = max(0.4 * fallback, 4.0)
                pend_var = sigma * sigma
        # Team-level variance floor. Summing independent per-player variances
        # dilutes a full lineup's spread by ~1/sqrt(n starters): nine starters
        # projected to 130 came out at sigma ~17 (CV ~0.13), when real weekly
        # team scores run CV ~0.20-0.25. That undershoot pushed the win bar to
        # 1%/99%. Floor the pending variance at (TEAM_CV_FLOOR * remaining
        # projection)^2 so the distribution matches observed dispersion. It
        # scales with what is left to play, so it fades to zero as games finish
        # and near-final blowouts still read decisively.
        floor_var = (TEAM_CV_FLOOR * pend_proj) ** 2
        if floor_var > pend_var:
            pend_var = floor_var
        return locked, pend_proj, pend_var

    left_stats = _stats(left)
    right_stats = _stats(right)
    if left_stats is None or right_stats is None:
        return None
    l_lock, l_pend, l_var = left_stats
    r_lock, r_pend, r_var = right_stats
    l_total = l_lock + l_pend
    r_total = r_lock + r_pend
    combined_var = l_var + r_var

    if combined_var < 1e-6:
        if l_total > r_total:
            return 1.0
        if r_total > l_total:
            return 0.0
        return 0.5

    z = (l_total - r_total) / (combined_var ** 0.5 * 2 ** 0.5)
    return max(0.01, min(0.99, 0.5 * (1 + erf(z))))


def compute_team_projections_for_weeks(
        matchups_by_week: dict[int, list[dict]],
        statuses_by_week: dict[int, dict],
        projections_by_week: dict[int, dict],
        roster_map: dict[str, str],  # roster_id -> owner name
) -> dict[tuple[int, str], float]:
    """
    Returns {(week, roster_id): live_proj_total}

    Assumes:
      - statuses_by_week: {week: {"statuses": {pid: status_str}}}
      - projections_by_week: {week: {"projections": {pid: proj_val}}}
      - matchups_by_week: {week: [ { "left": {...}, "right": {...} }, ... ]}
    """
    proj_by_roster: dict[tuple[int, str], float] = {}

    # reverse: owner display name -> roster_id (fallback if roster_id missing)
    owner_to_rid = {owner: rid for rid, owner in roster_map.items()}

    for week, matchups in matchups_by_week.items():
        # per-week statuses
        week_status_bundle = statuses_by_week.get(week) or {}
        week_status_by_pid = (week_status_bundle.get("statuses") or {}) if isinstance(week_status_bundle, dict) else {}

        # per-week projections
        if isinstance(projections_by_week, dict):
            week_proj_map = _week_proj_map_from_bundles(projections_by_week, week)
        else:
            # fallback – treat as already a flat {pid: proj_val}
            week_proj_map = projections_by_week or {}

        for m in matchups:
            for side in ("left", "right"):
                team = m.get(side) or {}
                rid = team.get("roster_id")

                # fallback if roster_id not in the team obj
                if rid is None:
                    rid = owner_to_rid.get(team.get("name", ""))

                if rid is None:
                    continue

                _, live_proj_total = team_live_totals(
                    team,
                    week_status_by_pid,
                    week_proj_map,
                )
                proj_by_roster[(week, str(rid))] = live_proj_total

    return proj_by_roster


def build_team_schedule_lookup(games: List[dict]) -> Dict[str, dict]:
    """
    Given a list of game dicts from Tank01 getNFLGamesForWeek,
    build a lookup: team_abv -> that week's game dict.

    Each team appears at most once per week, so mapping is safe.
    """
    lookup: Dict[str, dict] = {}
    for g in games:
        home = (g["home"] or "").upper()
        away = (g["away"] or "").upper()

        if home:
            for key in team_abbr_keys(home):
                lookup[key] = g
        if away:
            for key in team_abbr_keys(away):
                lookup[key] = g

    return lookup


def parse_game_datetime(game_time_str: str) -> datetime:
    """
    Convert Tank01 game date/time into a real datetime object.

    game_date: "20251204"
    game_time_str: "8:15p" or "1:00a" (Tank01 style)
    """
    time_str = game_time_str.strip().lower()

    # Add missing "m"
    if time_str.endswith("a") or time_str.endswith("p"):
        time_str += "m"  # "8:15p" → "8:15pm"

    dt = datetime.strptime(f"{time_str}", "%I:%M%p")
    return dt


def has_any_stats(stats: Dict[str, Any]) -> bool:
    """
    Returns True if at least one numeric stat is non-zero.
    """
    for v in stats.values():
        if isinstance(v, (int, float)) and v != 0:
            return True
    return False


def format_player_stats(
        teams_stats: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]],
        team: str,
        pos: str,
        player: str,
) -> Optional[str]:
    """
    Returns a compact stat line with no player name.
    Supports: QB/RB/WR/TE, K (incl PK), IDP, and combined DEF/DST.
    """
    defensive_positions = {
        "DL", "DE", "DT",
        "EDGE",
        "LB", "ILB", "OLB",
        "DB", "CB", "S", "FS", "SS",
    }

    pos_norm = (pos or "").strip().upper()

    # ---------- helpers ----------
    def phrase(v: int | float, singular: str, plural: str) -> str:
        v_int = int(v)
        return f"{v_int} {singular if v_int == 1 else plural}"

    def first_key(d: Dict[str, Any], *keys: str, default: int | float = 0):
        for k in keys:
            if k in d and d.get(k) is not None:
                return d.get(k)
        return default

    def sum_numeric_fields(objs: Dict[str, Any]) -> Dict[str, float]:
        """
        objs: {name_key: {stat: val, ...}, ...}
        returns a single dict with all numeric stats summed.
        """
        combined: Dict[str, float] = {}
        for _, st in (objs or {}).items():
            if not isinstance(st, dict):
                continue
            for k, v in st.items():
                if isinstance(v, (int, float)):
                    combined[k] = combined.get(k, 0.0) + float(v)
        return combined

    def fmt_dst_line(combined: Dict[str, Any]) -> str:
        parts: list[str] = []

        # Common DST/defense keys across various feeds
        sack = first_key(combined, "sack", "sacks", "def_sack", "def_sacks", "idp_sack", "idp_sacks", default=0)
        ints = first_key(combined, "int", "ints", "def_int", "def_ints", "idp_int", default=0)
        ff = first_key(combined, "ff", "forced_fum", "forced_fumbles", "def_ff", "idp_ff", "idp_forced_fum", default=0)
        fr = first_key(combined, "fum_rec", "fumble_recovery", "fumble_recoveries", "def_fr", "idp_fum_rec", default=0)
        td = first_key(combined, "def_td", "dst_td", "td", "tds", "def_tds", "idp_td", default=0)

        pa = first_key(combined, "pts_allow", "points_allowed", "def_pts_allow", "dst_pa", default=0)
        ya = first_key(combined, "yds_allow", "yards_allowed", "def_yds_allow", "dst_ya", default=0)

        # Points allowed is meaningful even at zero (a shutout), so it shows
        # whenever the feed reports it. Everything else is a tally, so a 0 means
        # it did not happen -- drop it rather than pad the line with empties.
        if any(k in combined for k in ("pts_allow", "points_allowed", "def_pts_allow", "dst_pa")):
            parts.append(f"{int(pa)} pa")
        if sack: parts.append(phrase(sack, "sack", "sacks"))
        if ints: parts.append(phrase(ints, "int", "ints"))
        if fr: parts.append(phrase(fr, "fr", "fr"))
        if td: parts.append(phrase(td, "td", "tds"))
        if ff: parts.append(phrase(ff, "ff", "ff"))
        if ya: parts.append(f"{int(ya)} ya")

        return ", ".join(parts)

    # ---------- pick lookup bucket ----------
    if pos_norm == "PK":
        lookup_pos = "K"
    elif pos_norm in ("DEF", "DST", "D/ST"):
        lookup_pos = "DEF"
    else:
        lookup_pos = "IDP" if pos_norm in defensive_positions or pos_norm == "IDP" else pos_norm

    teams_stats = teams_stats or {}
    team_data = lookup_team_map(teams_stats, team) or {}

    parts: list[str] = []

    def add(v, singular: str, plural: str | None = None) -> None:
        """Append ``"N label"`` only when the value is truthy. A 0 means the
        event did not happen, so it is left off the line entirely."""
        if v:
            parts.append(phrase(v, singular, plural if plural is not None else singular))

    # ---------- DEF/DST combined branch ----------
    if lookup_pos == "DEF":
        if not team_data:
            return None
        if isinstance(team_data.get("IDP"), dict) and team_data.get("IDP"):
            combined = sum_numeric_fields(team_data["IDP"])
        else:
            return None

        if not has_any_stats(combined):
            return None

        return fmt_dst_line(combined)

    # ---------- normal per-player lookup ----------
    # Use the same canonical, historical-team-aware resolver used by the raw
    # entry/trust path; formatting must not have a weaker identity lookup.
    player_stats = player_week_stat_entry(teams_stats, team, lookup_pos, player)

    if not player_stats or not has_any_stats(player_stats):
        return None

    # ---------------- QB / RB / WR / TE ----------------
    if lookup_pos == "QB":
        cmp = first_key(player_stats, "pass_cmp", "pass_comp", "cmp", "completions", default=0)
        att = first_key(player_stats, "pass_att", "att", "attempts", default=0)
        py = player_stats.get("pass_yds", 0)
        ptd = player_stats.get("pass_td", 0)
        ints = player_stats.get("int", 0)
        ra = player_stats.get("rush_att", 0)
        ry = player_stats.get("rush_yds", 0)
        rtd = player_stats.get("rush_td", 0)

        if att or cmp: parts.append(f"{int(cmp)}/{int(att)} cmp/att")
        add(py, "yd", "yds")
        add(ptd, "td", "tds")
        add(ints, "int", "ints")
        add(ra, "car", "car")
        add(ry, "rush yd", "rush yds")
        add(rtd, "rush td", "rush tds")

    elif lookup_pos in {"RB", "WR", "TE"}:
        ra = player_stats.get("rush_att", 0)
        ry = player_stats.get("rush_yds", 0)
        rtd = player_stats.get("rush_td", 0)
        rec = player_stats.get("rec", 0)
        tgt = first_key(player_stats, "tgt", "targets", "rec_tgt", default=0)
        rec_yds = player_stats.get("rec_yds", 0)
        rec_td = player_stats.get("rec_td", 0)

        if lookup_pos == "RB":
            add(ra, "car", "car")
            add(ry, "rush yd", "rush yds")
            add(rtd, "rush td", "rush tds")
        add(rec, "rec", "rec")
        add(tgt, "tgt", "tgt")
        add(rec_yds, "rec yd", "rec yds")
        add(rec_td, "rec td", "rec tds")
        if lookup_pos in {"WR", "TE"}:
            add(ra, "car", "car")
            add(ry, "rush yd", "rush yds")
            add(rtd, "rush td", "rush tds")

    # ---------------- K / PK ----------------
    elif lookup_pos == "K":
        fg_m = first_key(player_stats, "fgm", "fg_made", "field_goals_made", default=0)
        fg_a = first_key(player_stats, "fga", "fg_att", "field_goals_attempted", default=0)
        xp_m = first_key(player_stats, "xpm", "xp_made", "pat_made", "extra_points_made", default=0)
        xp_a = first_key(player_stats, "xpa", "xp_att", "pat_att", "extra_points_attempted", default=0)
        fg_long = first_key(player_stats, "fg_long", "fg_longest", "fg_lng", "lng", default=0)

        if fg_a:
            parts.append(f"{int(fg_m)}/{int(fg_a)} fg")
        elif fg_m:
            parts.append(phrase(fg_m, "fg", "fg"))

        if xp_a:
            parts.append(f"{int(xp_m)}/{int(xp_a)} xp")
        elif xp_m:
            parts.append(phrase(xp_m, "xp", "xp"))

        if fg_long: parts.append(f"long {int(fg_long)}")

    # ---------------- IDP ----------------
    elif lookup_pos == "IDP":
        tkl = player_stats.get("idp_tkl", 0)
        tkl_solo = player_stats.get("idp_tkl_solo", 0)
        tkl_ast = player_stats.get("idp_tkl_ast", 0)
        qb_hit = player_stats.get("idp_qb_hit", 0)
        ff = player_stats.get("idp_ff", 0) or player_stats.get("idp_forced_fum", 0)
        sack = player_stats.get("idp_sack") or player_stats.get("idp_sk") or player_stats.get("idp_sacks") or 0
        int_def = player_stats.get("idp_int", 0)
        pd = player_stats.get("idp_pd", 0) or player_stats.get("idp_pass_def", 0)

        if tkl:
            parts.append(phrase(tkl, "tkl", "tkl"))
            breakdown_bits = []
            if tkl_solo: breakdown_bits.append(phrase(tkl_solo, "solo", "solo"))
            if tkl_ast: breakdown_bits.append(phrase(tkl_ast, "ast", "ast"))
            if breakdown_bits:
                parts[-1] += f" ({', '.join(breakdown_bits)})"

        add(sack, "sack", "sacks")
        add(ff, "ff", "ff")
        add(qb_hit, "qb hit", "qb hits")
        add(int_def, "int", "ints")
        add(pd, "pd", "pd")

    # ---------------- fallback ----------------
    else:
        for k, v in player_stats.items():
            if isinstance(v, int) and v != 0:
                parts.append(f"{k}={v}")

    if not parts:
        return None
    return ", ".join(parts)


# Sleeper's per-player feed keys differ slightly from the Footballguys weekly
# scrape that format_player_stats reads. Map them so a Sleeper line can reuse
# the exact same formatter (lowercase labels, zero-suppression, and all).
_SLEEPER_TO_WEEKSTATS = {
    "pass_cmp": "pass_cmp", "pass_att": "pass_att", "pass_yd": "pass_yds",
    "pass_td": "pass_td", "pass_int": "int",
    "rush_att": "rush_att", "rush_yd": "rush_yds", "rush_td": "rush_td",
    "rec": "rec", "rec_tgt": "tgt", "rec_yd": "rec_yds", "rec_td": "rec_td",
    "fum_lost": "fum_lost",
}


def _sleeper_skill_stat_line(season, w, pid, pos, name, team_code) -> Optional[str]:
    """A QB/RB/WR/TE box-score line built from the Sleeper per-player feed.

    Footballguys can miss a player entirely (rookies, mid-week adds), leaving a
    starter who clearly played with no line. Sleeper's feed -- the same source
    the player-modal game log reads -- has everyone, keyed by the starter's own
    pid, so this fills the gap. The Sleeper line is this week's real data (it is
    what the points are scored from), never a stale leftover.
    """
    if pos not in ("QB", "RB", "WR", "TE") or not pid or not team_code:
        return None
    try:
        week = load_sleeper_week_stats(season, w)
    except Exception:
        return None
    row = week.get(str(pid))
    if not isinstance(row, dict):
        return None
    mapped: Dict[str, Any] = {}
    for src, dst in _SLEEPER_TO_WEEKSTATS.items():
        v = row.get(src)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            mapped[dst] = v
    if not mapped:
        return None
    # Reuse the real formatter by handing it a one-player, one-team snapshot in
    # the shape it already understands.
    synthetic = {team_code: {pos: {normalize_name(name): mapped}}}
    return format_player_stats(synthetic, team_code, pos, name)


def build_offense_rankings(teams_index: dict) -> dict:
    """
    Returns a dictionary ranking all teams by offensive metrics:
      - rush_yds_rank  (higher rush_yds_pg = better, rank 1 is best)
      - pass_yds_rank  (higher pass_yds_pg = better, rank 1 is best)
      - total_off_rank (combined yards + TDs, rank 1 is best)

    teams_index example:
      {
        "ARI": {
          "rush_yds_pg": 100.6,
          "pass_yds_pg": 236.7,
          "rush_td_pg": 0.75,
          "pass_td_pg": 1.58,
          ...
        },
        ...
      }

    Output:
      {
        "ARI": {
          "rush_yds_rank": 14,
          "pass_yds_rank": 10,
          "total_off_rank": 8,
        },
        "ATL": {...},
        ...
      }
    """

    TD_WEIGHT = 40.0  # treat 1 TD per game ~ 40 yards; tweak if desired

    rush_list = []
    pass_list = []
    total_list = []

    teams_index = canonical_teams_index(teams_index)

    for abbr, info in teams_index.items():
        rush_yds = info.get("rush_yds_pg")
        pass_yds = info.get("pass_yds_pg")
        rush_td = info.get("rush_td_pg")
        pass_td = info.get("pass_td_pg")

        # rushing yards list
        if rush_yds is not None:
            rush_list.append((abbr, float(rush_yds)))

        # passing yards list
        if pass_yds is not None:
            pass_list.append((abbr, float(pass_yds)))

        # total offense list (need at least both yardage numbers)
        if rush_yds is not None and pass_yds is not None:
            r_y = float(rush_yds)
            p_y = float(pass_yds)
            r_td = float(rush_td) if rush_td is not None else 0.0
            p_td = float(pass_td) if pass_td is not None else 0.0

            total_yards = r_y + p_y
            tds_pg = r_td + p_td
            total_score = total_yards + TD_WEIGHT * tds_pg

            total_list.append((abbr, total_score))

    # Sort: higher is better for offense
    rush_sorted = sorted(rush_list, key=lambda x: x[1], reverse=True)
    pass_sorted = sorted(pass_list, key=lambda x: x[1], reverse=True)
    total_sorted = sorted(total_list, key=lambda x: x[1], reverse=True)

    rankings = {abbr: {} for abbr in teams_index.keys()}

    for rank, (abbr, _) in enumerate(rush_sorted, start=1):
        rankings[abbr]["rush_yds_rank"] = rank

    for rank, (abbr, _) in enumerate(pass_sorted, start=1):
        rankings[abbr]["pass_yds_rank"] = rank

    for rank, (abbr, _) in enumerate(total_sorted, start=1):
        rankings[abbr]["total_off_rank"] = rank

    return rankings


def render_matchup_slide(
        season: str,
        m: dict,
        w: int,
        proj_week: int,
        status_by_pid: dict[str, str],
        projections: dict[str, float],
        players: dict,
        teams: dict,
        team_game_lookup: dict,
        fpts_against: Optional[dict] = None,
        viewer_roster_id: Optional[str] = None,
        compact: bool = False,
        scoring_settings: Optional[dict] = None,
        is_gotw: bool = False,
        gotw_selection: Optional[dict] = None,
        roster_positions: Optional[List[str]] = None,
) -> str:
    """One slide with rows like:
       [Left Name] [Left Pts/Proj] [Right Pts/Proj] [Right Name]

    roster_positions: the league's lineup slots (e.g. QB/RB/WR/TE/FLEX/
    SUPER_FLEX/K/DEF/BN). Provider starter lists follow the same order, so
    row i pairs with the i-th non-bench slot and the centre chip names the
    *slot* (FLEX/SF) instead of copying one player's position.

    viewer_roster_id: when the viewer's own team wins this (current, finalized)
    week, the slide plays the bigger "final whistle" takeover instead of the
    small matchup-win pop.

    compact: dashboard slides render only m-head + m-win-bar (no starter body).
    """
    proj = w > proj_week
    completed_week = not proj
    # A week strictly before the current/last-final week (proj_week) is fully
    # finalized: its box scores come from that week's own snapshot file
    # (load_week_stats(season, w)), which is season+week specific and cannot be
    # a "last year's Week N" leftover. The stale-leftover trust gate only needs
    # to guard the live/most-recent week, so past weeks show their real lines
    # for every position (K/DEF/IDP included) once the game has started.
    past_week = w < proj_week
    compact = bool(compact)
    allow_live = _allow_live_game_indicators(season)

    # Heavy stuff: do once per call. Compact slides skip week stats / schedule.
    _fpts_pos_cache: dict = {}
    if compact:
        offense_ranks = {}
        _fpts_data: dict = {}
        week_stats = {}
        team_schedule_lookup = {}
    else:
        teams_index = load_teams_index()
        offense_ranks = build_offense_rankings(teams_index)
        _fpts_data = fpts_against or {}
        week_stats = _week_stats_for_slide(season, w)
        team_schedule_lookup = build_team_schedule_lookup(load_week_schedule(season, w))

    # Live game progress: lets in-progress starters project their finish (banked
    # points + remaining projection) instead of freezing at their current score,
    # so team totals and the win bar track the games as they play.
    _frac_lookup = make_frac_lookup(team_game_lookup, team_schedule_lookup)

    def _get_fpts_rank(team: str, pos: str):
        if not _fpts_data:
            return None, 0.0
        if pos not in _fpts_pos_cache:
            vals = [(t, _fpts_data.get(t, {}).get(pos, 0)) for t in _fpts_data]
            vals.sort(key=lambda x: x[1], reverse=True)
            _fpts_pos_cache[pos] = {t: i + 1 for i, (t, _) in enumerate(vals)}
        rank = _fpts_pos_cache.get(pos, {}).get(team)
        fpts_val = float(_fpts_data.get(team, {}).get(pos, 0))
        return rank, fpts_val

    # Projections for this week (dict {pid: proj_val})
    week_proj_map = _week_proj_map_from_bundles(projections, w)
    # Lazy raw-file fallback (same source Scout uses) when the bundle is empty
    # or a starter pid is missing -- common for Yahoo where scoreboard rows omit
    # lineups and the first paint races projection hydration.
    _raw_week_proj: Optional[Dict[str, Any]] = None

    def _raw_week_map() -> Dict[str, Any]:
        nonlocal _raw_week_proj
        if _raw_week_proj is None:
            try:
                from utils.utils import load_week_projection
                _raw_week_proj = load_week_projection(int(season), int(w)) or {}
            except Exception:
                _raw_week_proj = {}
        return _raw_week_proj

    def _pid_proj(pid: Any, pos: str = "") -> float:
        mapped = _proj_value_for_pid(
            week_proj_map, pid,
            raw_week_map=None, scoring_settings=scoring_settings, pos=pos,
        )
        if mapped != 0.0 or (pid is not None and str(pid) in week_proj_map):
            return mapped
        # Bundle miss (or explicit absence): try the raw weekly file once.
        return _proj_value_for_pid(
            week_proj_map, pid,
            raw_week_map=_raw_week_map(),
            scoring_settings=scoring_settings,
            pos=pos,
        )

    # Cache NFL score lookups per date
    score_cache: dict[str, dict] = {}

    def get_team_game_from_scores(game_date_str: str, team_abv: str) -> Optional[dict]:
        """
        Lazily fetch scores for a given date once, then reuse for all players.
        """
        if not game_date_str:
            return None
        if game_date_str not in score_cache:
            scores_body = get_nfl_scores_for_date(game_date_str)
            score_cache[game_date_str] = build_team_game_lookup(scores_body) if scores_body else {}
        return score_cache[game_date_str].get(team_abv)

    today_str = date.today().strftime("%Y%m%d")
    now_dt = datetime.now()

    def _score_html(t, proj_mode: bool) -> tuple[str, bool]:
        """Returns (html, has_live_proj). has_live_proj=True means actual+proj stacked."""
        if not proj_mode:
            points = f"{t['pts_total']:.2f}" if isinstance(t.get("pts_total"), (int, float)) else "-"
            return f"<span class='num'>{points}</span>", False
        lineup_actual, live_proj_total = team_live_totals(
            t, status_by_pid, week_proj_map,
            proj_lookup=_pid_proj, frac_lookup=_frac_lookup,
        )

        def _trend_arrow(live) -> str:
            """Up/down arrow comparing the live projected final to the pregame
            projection, so the header shows whether the team is beating or
            missing its number in real time."""
            pregame = t.get("proj_total")
            if not isinstance(pregame, (int, float)) or not isinstance(live, (int, float)):
                return ""
            if live > pregame + 0.05:
                return ("<span class='mb-trend mb-trend-up' "
                        "aria-label='projection trending up'>&#9650;</span>")
            if live < pregame - 0.05:
                return ("<span class='mb-trend mb-trend-down' "
                        "aria-label='projection trending down'>&#9660;</span>")
            return "<span class='mb-trend mb-trend-flat' aria-hidden='true'>&#9644;</span>"

        any_started = any(
            status_by_pid.get(p.get("pid"), STATUS_NOT_STARTED) in (STATUS_IN_PROGRESS, STATUS_FINAL)
            for p in (t.get("starters") or [])
        )
        if not any_started:
            return (f"<span class='num m-proj-only'>{live_proj_total:.1f}</span>"
                    f"{_trend_arrow(live_proj_total)}"), False
        # Provider scoreboard totals remain authoritative.  A partially mapped
        # Yahoo lineup may enrich rows but must never zero the matchup header.
        actual_total = t.get("pts_total")
        if not isinstance(actual_total, (int, float)):
            starters = t.get("starters") or []
            lineup_complete = bool(starters) and all(
                isinstance(p.get("pts"), (int, float)) for p in starters
            )
            actual_total = lineup_actual if lineup_complete else 0.0
        return (f"<span class='num'>{actual_total:.1f}</span>"
                f"<span class='proj'>{live_proj_total:.1f}"
                f"{_trend_arrow(live_proj_total)}</span>"), True

    def _team_col(t, side: str) -> str:
        rid = t.get('roster_id', '')
        name = t['name']
        record = t.get('record', '0-0')
        username = t.get('username') or ''
        ava = t.get("avatar") or ""
        img_html = f"<img class='avatar m-av' src='{ava}' alt='' loading='lazy' decoding='async' onerror=\"this.style.display='none'\">" if ava else ""
        name_el = f"<div class='m-team-name team-clickable' style='cursor:pointer;' data-roster-id='{rid}' data-team-name='{name}'>{name}</div>"
        rank = t.get('rank')
        rank_txt = f" (#{rank})" if rank else ""
        if side == 'left':
            meta = f"<div class='m-team-meta'>{record} &bull; @{username}{rank_txt}</div>"
            return f"<div class='m-team-col m-col-left'>{img_html}{name_el}{meta}</div>"
        else:
            meta = f"<div class='m-team-meta'>@{username}{rank_txt} &bull; {record}</div>"
            return f"<div class='m-team-col m-col-right'>{img_html}{name_el}{meta}</div>"


    def format_team_game_line(team_abv: str, game: dict, pos: str, side: str) -> str:
        if not team_abv or not game:
            return ""

        home = canon_team(game.get("home")) or str(game.get("home") or "").upper()
        away = canon_team(game.get("away")) or str(game.get("away") or "").upper()
        t_up = (canon_team(team_abv) or team_abv).upper()
        home_keys = set(team_abbr_keys(home))
        away_keys = set(team_abbr_keys(away))
        if t_up not in home_keys and t_up not in away_keys:
            return ""

        is_home = t_up in home_keys
        opp = away if is_home else home
        status_code = str(game.get("gameStatusCode") or "0")  # '0' scheduled, '1' live, '2' final
        game_date = str(game.get("gameDate") or game.get("gameID", "")[:8])  # '20251204'
        game_time = str(game.get("gameTime") or "")  # '8:15p'

        # quick status correction by date
        if game_date < today_str:
            status_code = "2"
        elif game_date == today_str and game_time:
            try:
                # parse_game_datetime parses only the clock time, so its date
                # defaults to 1900 -- anchor it to today before comparing to now,
                # otherwise the kickoff is always "in the past" and a game later
                # today is never corrected back to scheduled.
                kickoff = parse_game_datetime(game_time).replace(
                    year=now_dt.year, month=now_dt.month, day=now_dt.day
                )
                if kickoff > now_dt:
                    # future kick within same date – treat as scheduled
                    status_code = "0"
            except ValueError:
                logging.getLogger(__name__).debug("suppressed exception", exc_info=True)

        if status_code == "0":
            dow = ""
            if game_date:
                try:
                    dt = datetime.strptime(game_date, "%Y%m%d")
                    dow = dt.strftime("%a")
                except ValueError:
                    logging.getLogger(__name__).debug("suppressed exception", exc_info=True)

            display_time = game_time
            if display_time.endswith("p"):
                display_time = display_time[:-1] + " pm"
            elif display_time.endswith("a"):
                display_time = display_time[:-1] + " am"

            off_ranks = offense_ranks.get(opp, {})

            suffix = ""
            if pos in ("QB", "WR", "TE", "RB", "K"):
                fpts_pos = pos
                opp_rank, fpts_val = _get_fpts_rank(opp, fpts_pos)
                if opp_rank is not None:
                    suffix = f" (#{opp_rank} / {fpts_val:.1f})"
            elif pos == "DEF":
                opp_rank = off_ranks.get("total_off_rank")
                if opp_rank is not None:
                    suffix = f" (#{opp_rank})"
            opponent = ("@ " + opp) if not is_home else ("vs " + opp)
            # Separate semantic pieces so narrow matchup columns can wrap at
            # useful boundaries instead of clipping one long metadata string.
            return (f"<span class='m-game-kickoff'>{html.escape(' '.join(x for x in [dow, display_time] if x))}</span> "
                    f"<span class='m-game-opponent'>{html.escape(opponent)}</span> "
                    f"<span class='m-game-rank'>{html.escape(suffix.strip())}</span>").strip()

        # For live/final, pull from scores API once per date
        game_date_std = game_date  # already YYYYMMDD
        score_str = ""
        score_game = get_team_game_from_scores(game_date_std, team_abv)

        if score_game:
            if is_home:
                my_pts = score_game.get("homePts")
                opp_pts = score_game.get("awayPts")
            else:
                my_pts = score_game.get("awayPts")
                opp_pts = score_game.get("homePts")

            if my_pts is not None and opp_pts is not None:
                score_str = f"{my_pts}-{opp_pts}"

        if status_code == "1":
            line_score = game.get("lineScore") or {}
            period = line_score.get("period", "")
            clock = game.get("gameClock", "")
            prefix = "@ " + opp if not is_home else "vs " + opp
            live_clock = " ".join(x for x in [period, clock] if x).strip()
            rest = " ".join(x for x in [score_str, prefix] if x).strip()

            if not allow_live:
                return " ".join(x for x in [score_str, live_clock, prefix] if x).strip()

            # Live dot glued to the game clock as one non-wrapping unit, so it
            # sits with the clock instead of orphaning onto its own line once the
            # box-score text is allowed to wrap.
            live_unit = f"<span class='mb-live'><span class='live-dot'></span>{live_clock}</span>".strip()
            return f"{live_unit} {rest}".strip() if rest else live_unit

        if status_code == "2":
            prefix = "@ " + opp if not is_home else "vs " + opp
            if score_str:
                return f"Final {prefix} {score_str}"
            return "Final"

        return ""

    def player_bits(
            p,
            side: str,
            left_side: bool,
    ):
        if not p:
            # Must match the 9-tuple success path: info, pos, actual, proj, bye,
            # not_started, stats, nfl, pid. Empty starter slots (zip_longest fill)
            # hit this branch and used to 500 the dashboard on league switch.
            return "", "", 0.0, None, False, False, None, "", ""

        pid = p.get("pid")
        name = p.get("name", "")
        nfl = p.get("nfl", "")
        pos = p.get("pos")
        if pos not in ["QB", "RB", "WR", "TE", "K", "DEF"]:
            pos = "IDP"

        if pos == "IDP":
            team_stats = lookup_team_map(week_stats or {}, nfl) or {}
            pos_data = team_stats.get(pos, {})
            player_stats = pos_data.get(normalize_name(name), {})
            actual = player_stats.get('pts_idp', 0.0)
        else:
            actual = p.get("pts")

        proj_val = _pid_proj(pid, pos)
        is_bye = False

        player_index = players.get(pid) or teams.get(pid)
        if player_index:
            if proj_val == 0.0 and player_index.get("byeWeek") == w:
                is_bye = True

        status = status_by_pid.get(pid)
        if status is None:
            for alt in team_abbr_keys(str(pid or "")):
                if alt in status_by_pid:
                    status = status_by_pid[alt]
                    break
        if status is None:
            # Historical status maps are commonly absent.  The selected week,
            # not today's player status, decides whether normalized matchup
            # points are final and whether its cached weekly box score is shown.
            status = STATUS_FINAL if completed_week else STATUS_NOT_STARTED

        if status == "BYE":
            is_bye = True

        # Resolve this player's NFL game once, up front, so the live-projection
        # blend below and the game/stats lines further down share it.
        game = resolve_team_game(nfl, team_game_lookup, team_schedule_lookup)

        # decide what to show
        is_not_started = False
        if is_bye:
            display_actual = 0.0
            display_proj = None
        elif status == STATUS_NOT_STARTED:
            is_not_started = True
            display_actual = 0.0
            display_proj = proj_val
        elif status == STATUS_IN_PROGRESS:
            # Live projected finish: banked points + the slice of the pregame
            # projection still to come. Falls back to the pregame projection
            # when the game clock can't be read.
            display_actual = actual
            display_proj = (
                live_projected_final(float(actual), proj_val, game, pos=pos)
                if actual is not None else None
            )
        elif status == STATUS_FINAL:
            display_actual = actual
            display_proj = None
        else:
            display_actual = 0.0 if actual is None else actual
            display_proj = proj_val

        # game / stats
        game_line = ""
        stats = None
        if nfl:
            team_code = str(nfl).upper()
            lookup_name = name
            if game:
                game_line = format_team_game_line(team_code, game, pos, side)

            stats = format_player_stats(
                week_stats,
                team_code,
                pos,
                lookup_name,
            )

        # Only surface a box-score line once the player's game has actually
        # started. week_stats can still carry last year's Wk 1 line (Footballguys
        # keeps it until the new season plays), and WAS/WSH alias misses used to
        # mark Commanders as FINAL so the old "hide if not_started" gate never
        # fired. Keep the schedule game_line; drop the stat line until kickoff.
        #
        # Prefer the schedule row when we have one. Fall back to pid status only
        # when schedule lookup missed. Even after kickoff, hide Footballguys
        # leftovers unless Tank01 confirms live/final or the line was Tank-overlaid.
        raw_stat_entry = None
        if stats and nfl:
            raw_stat_entry = player_week_stat_entry(
                week_stats, str(nfl).upper(), pos, lookup_name if nfl else name,
            )
        if is_bye:
            stats = None
        elif game is not None:
            if not game_has_started(game):
                stats = None
            elif not past_week and not box_score_line_is_trusted(game, raw_stat_entry):
                # Tank01's cached schedule can lag "Final" for a game that has
                # clearly already been played (game_has_started already treats
                # calendar-past as started). Rather than blanket-hiding a
                # completed game's real stats until the code catches up, trust
                # the line anyway when its implied fantasy points line up with
                # Sleeper's authoritative live/final total -- a genuine match
                # means this is this week's box score, not a stale leftover.
                rescued = False
                if (
                    pos in ("QB", "RB", "WR", "TE")
                    and isinstance(raw_stat_entry, dict)
                    and scoring_settings and "rec" in scoring_settings
                ):
                    live_pts = p.get("pts")
                    if isinstance(live_pts, (int, float)) and not isinstance(live_pts, bool):
                        try:
                            from utils.fantasy_scoring import week_stats_line_points
                            implied = week_stats_line_points(raw_stat_entry, scoring_settings, pos)
                        except Exception:
                            implied = None
                        if implied is not None:
                            tol = max(4.0, 0.4 * max(abs(implied), abs(float(live_pts))))
                            rescued = abs(implied - float(live_pts)) <= tol
                if not rescued:
                    stats = None
        elif is_not_started:
            stats = None

        # Stale box-score guard. The points (p['pts']) are Sleeper's authoritative
        # live total; the box-score line is a separate Footballguys/Tank feed.
        # Footballguys republishes a prior week/season until its logs flip, so an
        # in-progress game can surface last week's full stat line beside this
        # week's real, low points. Tank-overlaid lines (_src=tank) are built from
        # the current game and trusted as-is; a non-Tank line whose implied score
        # is far from the points shown is stale -- hide it so stats match scoring.
        if (
            stats
            and pos in ("QB", "RB", "WR", "TE")
            and isinstance(raw_stat_entry, dict)
            and raw_stat_entry.get("_src") != "tank"
            and status == STATUS_IN_PROGRESS
            and scoring_settings and "rec" in scoring_settings
        ):
            live_pts = p.get("pts")
            if isinstance(live_pts, (int, float)) and not isinstance(live_pts, bool):
                try:
                    from utils.fantasy_scoring import week_stats_line_points
                    implied = week_stats_line_points(raw_stat_entry, scoring_settings, pos)
                except Exception:
                    implied = None
                if implied is not None:
                    tol = max(4.0, 0.4 * max(abs(implied), abs(float(live_pts))))
                    if abs(implied - float(live_pts)) > tol:
                        stats = None

        # Gap fill: a skill starter whose game has started but who has no line
        # (Footballguys missed them, or a stale leftover was just dropped) still
        # gets a box score from the Sleeper per-player feed, keyed by their pid.
        if stats is None and not is_bye and pid and nfl and pos in ("QB", "RB", "WR", "TE"):
            _played = (
                game_has_started(game) if game is not None
                else status in (STATUS_FINAL, STATUS_IN_PROGRESS)
            )
            if _played:
                stats = _sleeper_skill_stat_line(
                    season, w, pid, pos, name, str(nfl).upper(),
                )

        # Sub-line under the name: "NO • WR" (team + the player's *real*
        # position). On mobile CSS stacks it beneath the name; on desktop it
        # sits inline after the name. The centre rail chip names the lineup
        # *slot* (FLEX/SF), so the real position stays visible here.
        _pos_label_inline = "D/ST" if pos in ("DEF", "DST") else (pos or "")
        _team_txt = str(nfl or "").strip()
        if _team_txt and _pos_label_inline:
            meta_content = f"{_team_txt} \u2022 {_pos_label_inline}"
        else:
            meta_content = _team_txt or _pos_label_inline
        meta_content = html.escape(meta_content)

        # Add clickable attributes
        _safe_name = html.escape(name or "")
        _safe_pid = html.escape(str(pid or ""))
        clickable_attrs = (
            f" class='pname player-clickable' style='cursor:pointer;'"
            f" data-player-id='{_safe_pid}' data-player-name='{_safe_name}'"
            if pid else " class='pname'"
        )

        stats_inline = f"<span class='meta m-cell-stats mb-stat'>{stats}</span>" if stats else ""
        if status == STATUS_FINAL and not stats and raw_stat_entry is None and not is_bye:
            stats_inline = (
                "<span class='meta m-cell-stats mb-stat m-cell-stats--unavailable'>"
                "Stats unavailable</span>"
            )
        team_span = f"<span class='meta p-team mb-team'>{meta_content}</span>" if meta_content else ""

        # Compact board cell: name on top with a TEAM • POS sub-line, then the
        # game line, then the box score. The position badge is NOT inline here
        # -- it moves to the shared centre rail (assembled in the row loop) so
        # both players in a slot read against one coloured chip. Left/right
        # mirroring is handled in CSS off the parent .mb-cell-r, so the markup
        # is identical either side.
        bye_cls = " mb-info--bye" if is_bye else ""
        info_html = (
            f"<div class='mb-info{bye_cls}'>"
            "<div class='p-name-line mb-nameline'>"
            f"<span{clickable_attrs}>{_safe_name}</span>{team_span}"
            "</div>"
            f"<span class='meta p-game-line mb-game'>{game_line}</span>"
            f"{stats_inline}"
            "</div>"
        )

        return (
            info_html,
            pos,
            (float(display_actual) if display_actual is not None else None),
            display_proj,
            is_bye,
            is_not_started,
            (stats if stats else None),
            str(nfl or "").upper(),
            str(pid or ""),
        )

    rows_html: List[str] = []

    def _score_box(actual_val, proj_val, is_bye: bool, more: bool, not_started: bool) -> str:
        """Compact score cell: big actual on top, small live projection under it.
        The projection-only (not-started) state keeps ``m-proj-only`` so the
        existing tests and styling still find it."""
        if is_bye:
            return "<div class='mb-score'><span class='mb-a mb-bye'>BYE</span></div>"
        if not_started:
            pv = proj_val if isinstance(proj_val, (int, float)) else 0.0
            return (
                "<div class='mb-score'>"
                f"<span class='mb-a mb-proj-only m-proj-only'>{pv:.1f}</span>"
                "</div>"
            )
        if actual_val is None:
            return (
                "<div class='mb-score'>"
                "<span class='mb-a mb-na' title='Weekly fantasy points unavailable'>&mdash;</span>"
                "</div>"
            )
        a_cls = "mb-a" + (" mb-win" if more else "")
        proj_line = (
            f"<span class='mb-pj'>{proj_val:.1f}</span>"
            if isinstance(proj_val, (int, float)) else ""
        )
        return f"<div class='mb-score'><span class='{a_cls}'>{actual_val:.1f}</span>{proj_line}</div>"

    _starter_pairs = () if compact else zip_longest(
            m["left"].get("starters", []),
            m["right"].get("starters", []),
            fillvalue=None,
    )
    # Lineup slot order: provider starter lists follow roster_positions order,
    # so row i pairs with the i-th non-bench slot. Short/missing slot data
    # falls back to the old player-position chip for that row.
    _slot_order = [
        str(s).upper() for s in (roster_positions or []) if str(s).upper() != "BN"
    ]

    def _slot_chip(slot: str, left_p: str, right_p: str) -> tuple:
        """(css_class, label) for the centre-rail chip: the *lineup slot*.

        FLEX and SUPER_FLEX rows read FLEX / SF even though the two players
        in them are different positions; ordinary slots read QB/RB/WR/TE/K/
        D/ST. Unknown or missing slot data falls back to a player's real
        position (the old behaviour).
        """
        s = (slot or "").upper()
        if s == "FLEX":
            return "FLEX", "FLEX"
        if s == "SUPER_FLEX":
            return "SF", "SF"
        if s in ("QB", "RB", "WR", "TE", "K"):
            return s, s
        if s in ("DEF", "DST"):
            return "DEF", "D/ST"
        p = left_p or right_p or ""
        if p in ("DEF", "DST"):
            return "DEF", "D/ST"
        return p, p

    for _row_idx, (L, R) in enumerate(_starter_pairs):
        (left_info, left_pos, left_actual, left_proj, left_is_bye,
         left_not_started, _left_stats, left_nfl, left_pid) = player_bits(L, "left", True)
        (right_info, right_pos, right_actual, right_proj, right_is_bye,
         right_not_started, _right_stats, right_nfl, right_pid) = player_bits(R, "right", False)

        la = 0.0 if left_is_bye else left_actual
        ra = 0.0 if right_is_bye else right_actual

        left_more = la is not None and ra is not None and la > ra
        right_more = la is not None and ra is not None and ra > la

        left_sb = _score_box(left_actual, left_proj, left_is_bye, left_more, left_not_started)
        right_sb = _score_box(right_actual, right_proj, right_is_bye, right_more, right_not_started)

        # Shared centre rail: one coloured chip per *lineup slot*. A FLEX row
        # reads FLEX and a superflex row reads SF even though the two players
        # in it are different positions; ordinary slots read QB/RB/WR/TE/K/
        # D/ST. Reuses the site-wide .pos-badge colours (+ FLEX teal, SF blue).
        _slot = _slot_order[_row_idx] if _row_idx < len(_slot_order) else ""
        _chip_class, _chip_label = _slot_chip(_slot, left_pos, right_pos)
        pos_chip = (
            f"<span class='pos-badge {html.escape(_chip_class)}'>{html.escape(_chip_label)}</span>"
            if _chip_class else ""
        )

        # Live drive-bar mounts -- one per player's cell, tagged with the NFL
        # team. The client fills a bar under whichever player's team currently
        # has the ball (from the shared live game feed); inert until then and
        # only for started games.
        left_fld = f"<div class='mb-fld' data-team='{html.escape(left_nfl)}'></div>"
        right_fld = f"<div class='mb-fld' data-team='{html.escape(right_nfl)}'></div>"

        rows_html.append(
            "<div class=\"mb-row\">"
            f"<div class=\"mb-cell mb-cell-l\"><div class=\"mb-cell-row\">{left_info}{left_sb}</div>{left_fld}</div>"
            f"<div class=\"mb-pos\">{pos_chip}</div>"
            f"<div class=\"mb-cell mb-cell-r\"><div class=\"mb-cell-row\">{right_info}{right_sb}</div>{right_fld}</div>"
            "</div>"
        )

    # Win probability: only for live/projection weeks (skip completed weeks)
    win_bar_html = ""
    if proj:
        l_prob = compute_win_prob(
            m["left"], m["right"], status_by_pid, week_proj_map,
            frac_lookup=_frac_lookup,
        )
        if l_prob is not None:
            lp = round(l_prob * 100)
            rp = 100 - lp
            l_leading = l_prob >= 0.5
            win_green = "#22c55e"
            lose_fade = "rgba(148,163,184,0.35)"
            l_col = win_green if l_leading else "var(--text-muted)"
            r_col = win_green if not l_leading else "var(--text-muted)"
            l_bar = win_green if l_leading else lose_fade
            r_bar = win_green if not l_leading else lose_fade
            track_bg = f"linear-gradient(to right,{l_bar} {lp}%,{r_bar} {lp}%)"
            _wp_lname = str(m['left'].get('name') or 'left team').replace('"', '')
            _wp_rname = str(m['right'].get('name') or 'right team').replace('"', '')
            win_bar_html = f"""<div class="m-win-bar" role="img" aria-label="Win probability: {_wp_lname} {lp} percent, {_wp_rname} {rp} percent">
  <span class="m-wp-pct" style="color:{l_col};">{lp}%</span>
  <div class="m-wp-track" style="background:{track_bg};"></div>
  <span class="m-wp-pct" style="color:{r_col};text-align:right;">{rp}%</span>
</div>"""

    l_score, l_live = _score_html(m['left'], proj)
    r_score, r_live = _score_html(m['right'], proj)
    proj_class = " has-proj" if (l_live or r_live) else ""

    # Matchup-win moment: on a completed (non-projection) week, the winning side's
    # score pops with a green glow and its team column lifts when the slide first
    # scrolls into view. Skipped for live/projection weeks and for ties.
    win_attr = ""
    if not proj:
        _lp, _rp = m['left'].get('pts_total'), m['right'].get('pts_total')
        if isinstance(_lp, (int, float)) and isinstance(_rp, (int, float)) and _lp != _rp:
            _won = "left" if _lp > _rp else "right"
            win_attr = f' data-br-moment="matchupwin" data-mo-win="{_won}"'
            # Live final whistle: the viewer's own team winning the current week
            # gets a full-slide takeover with confetti, not just the score pop.
            if viewer_roster_id and w == proj_week:
                _won_rid = str(m[_won].get("roster_id"))
                if _won_rid == str(viewer_roster_id):
                    win_attr = (
                        f' data-br-moment="whistle" data-mo-win="{_won}"'
                        ' data-br-confetti="green" data-br-confetti-delay="250"'
                    )

    h2h = m.get("h2h") or {}
    h2h_l = h2h.get("left_wins", 0)
    h2h_r = h2h.get("right_wins", 0)
    h2h_html = ""
    if h2h_l + h2h_r > 0:
        h2h_html = (
            f"<div class='m-h2h'>H2H this season: "
            f"<b>{h2h_l}</b>–<b>{h2h_r}</b></div>"
        )

    # All-time rivalry line (client fills it from /api/rivalry so the heavy
    # multi-season scan stays lazy and off the initial render). Needs both
    # managers' user ids; skipped in compact slides and when either is missing.
    _riv_a = (m.get("left") or {}).get("owner_id")
    _riv_b = (m.get("right") or {}).get("owner_id")
    _riv_ln = str((m.get("left") or {}).get("name") or "")
    _riv_rn = str((m.get("right") or {}).get("name") or "")
    if (not compact) and _riv_a and _riv_b:
        h2h_html += (
            f"<div class='m-rivalry' data-riv-a=\"{html.escape(str(_riv_a), quote=True)}\" "
            f"data-riv-b=\"{html.escape(str(_riv_b), quote=True)}\" "
            f"data-riv-lname=\"{html.escape(_riv_ln, quote=True)}\" "
            f"data-riv-rname=\"{html.escape(_riv_rn, quote=True)}\" hidden></div>"
        )

    slide_cls = "m-slide m-slide--compact" if compact else "m-slide"
    body_html = ""
    if not compact:
        body_html = f"""
      <div class="m-body">
        <div class="m-combo">
          {''.join(rows_html)}
        </div>
      </div>"""

    gotw_badge_html = ""
    if is_gotw:
        _gotw_why = str((gotw_selection or {}).get("why") or "").strip()
        _gotw_reasons = [str(r).strip() for r in ((gotw_selection or {}).get("reasons") or []) if str(r).strip()]
        _info_html = ""
        if _gotw_why:
            _reasons_li = "".join(f"<li>{html.escape(r)}</li>" for r in _gotw_reasons[:2])
            _info_html = (
                "<button type='button' class='m-gotw-info' aria-label='Why this is the game of the week' "
                "aria-expanded='false' title='Why this is the game of the week'>i</button>"
                "<div class='m-gotw-pop' role='tooltip' hidden>"
                f"<div class='m-gotw-pop-why'>{html.escape(_gotw_why)}</div>"
                + (f"<ul class='m-gotw-pop-reasons'>{_reasons_li}</ul>" if _reasons_li else "")
                + "</div>"
            )
        _mobile_line = f"<div class='m-gotw-why-mobile'>{html.escape(_gotw_why)}</div>" if _gotw_why else ""
        gotw_badge_html = (
            "<div class='m-head-badges'><span class='m-gotw-badge'>"
            "<i class='fa-solid fa-fire' aria-hidden='true'></i>GOTW</span>"
            f"{_info_html}</div>{_mobile_line}"
        )

    # Matchup Moments: the client (brInitMatchupMoments) filters the shared live
    # play feed to this slide's starters and paints a big-play strip. Starter
    # pids are emitted so an arbitrary (Prev/Next) matchup can be filtered, and
    # the mount stays hidden until the client has something live to show.
    def _starter_pids(team_block) -> str:
        pids = [str(s.get("pid")) for s in (team_block.get("starters") or [])
                if s and s.get("pid")]
        return html.escape(",".join(pids))

    moments_attrs = ""
    moments_mount = ""
    if not compact:
        moments_attrs = (
            f" data-mb-left-pids='{_starter_pids(m['left'])}'"
            f" data-mb-right-pids='{_starter_pids(m['right'])}'"
        )
        moments_mount = "<div class=\"mb-moments\" hidden></div>"

    return f"""
    <div class="{slide_cls}"{win_attr}{moments_attrs}>
      <div class="m-head">
        {gotw_badge_html}
        <div class="m-head-row">
          {_team_col(m['left'], 'left')}
          <div class="m-scoreboard{proj_class}">
            <div class="m-score-val m-score-l">{l_score}</div>
            <div class="m-vs">vs</div>
            <div class="m-score-val m-score-r">{r_score}</div>
          </div>
          {_team_col(m['right'], 'right')}
        </div>
        {h2h_html}
      </div>
      {win_bar_html}
      {moments_mount}
      {body_html}
    </div>
    """
