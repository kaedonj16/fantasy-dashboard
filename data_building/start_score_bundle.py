"""Daily precomputed start/sit score inputs.

``compute_start_score`` (utils.start_sit_score) is a pure function of per-player
inputs that are almost entirely league-independent: weekly projection, recent
and season form, usage trend, Vegas implied total, weather, O-line quality,
expected plays, role confidence, and consistency. Only the *fantasy-point
scoring* of projections and actuals varies by league, and that collapses to one
of the twelve canonical projection variants (``ppr | half_ppr | std | tep |
tep_half | tep_std | 6pt_ppr | 6pt_half | 6pt_std | 6pt_tep | 6pt_tep_half |
6pt_tep_std``) already used by the projection cache
(``data_building.fetch_projections.PROJ_VARIANTS``,
``utils.proj_variant.pick_proj_variant``).

This module builds one input bundle per (season, week, player, variant) in the
daily cron. Request paths (``/api/start-sit-options``) then become a DB lookup
plus a live injury overlay plus the pure scoring function, instead of
re-reading multi-MB stat files, rebuilding the season usage map, and
re-scoring every week per request.

Exactness contract: a league uses the precomputed bundles only when its scoring
settings are *exactly* covered by one canonical variant
(``variant_for_exact_scoring``); anything exotic (fractional PPR, point-per-
first-down, custom yardage rates, ...) falls back to the live per-request
computation, so no league ever gets approximate numbers silently.

Injury status is deliberately NOT trusted from the bundle: Q tags change
intraday, so the request always overlays the live status from the player map
before calling ``compute_start_score``. The bundle stores the batch-time status
for reference only.

The bundle's ``proj_pts`` gets the same treatment at load time (see
``_apply_live_projection_overlay`` below): projections move intraday too —
when a starter is ruled out, Sleeper redistributes his volume to his
teammates within hours — so the loader swaps in the live Sleeper projection
for the week, cached for a few minutes, and silently keeps the batch value
whenever the live feed is unavailable or has no line for the player.

Form/usage inputs follow the 2026-10 rating rework: only PLAYED weeks count
(participation: ``gp`` >= 1 or an offensive snap count present; a
participated week with no production counts as a 0.0 game), recent form is the last 3 played weeks,
and usage is the absolute per-game delta of opportunities (RB) or targets
(WR/TE). ``derive_start_sit_inputs`` is the one derivation behind both this
bundle and the Start/Sit page's live fallback.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical scoring per projection variant.
#
# Chosen so pick_proj_variant(VARIANT_SCORING[v]) == v for every variant (the
# TE-premium bonus 0.5 matches the projection file's TEP_BONUS). A league only
# uses the fast path when its own settings match one of these exactly on the
# variant dimensions (rec / pass_td / bonus_rec_te) and are otherwise standard.
# ---------------------------------------------------------------------------
VARIANT_SCORING: Dict[str, Dict[str, float]] = {
    "ppr":          {"rec": 1.0, "pass_td": 4.0, "bonus_rec_te": 0.0},
    "half_ppr":     {"rec": 0.5, "pass_td": 4.0, "bonus_rec_te": 0.0},
    "std":          {"rec": 0.0, "pass_td": 4.0, "bonus_rec_te": 0.0},
    "tep":          {"rec": 1.0, "pass_td": 4.0, "bonus_rec_te": 0.5},
    "tep_half":     {"rec": 0.5, "pass_td": 4.0, "bonus_rec_te": 0.5},
    "tep_std":      {"rec": 0.0, "pass_td": 4.0, "bonus_rec_te": 0.5},
    "6pt_ppr":      {"rec": 1.0, "pass_td": 6.0, "bonus_rec_te": 0.0},
    "6pt_half":     {"rec": 0.5, "pass_td": 6.0, "bonus_rec_te": 0.0},
    "6pt_std":      {"rec": 0.0, "pass_td": 6.0, "bonus_rec_te": 0.0},
    "6pt_tep":      {"rec": 1.0, "pass_td": 6.0, "bonus_rec_te": 0.5},
    "6pt_tep_half": {"rec": 0.5, "pass_td": 6.0, "bonus_rec_te": 0.5},
    "6pt_tep_std":  {"rec": 0.0, "pass_td": 6.0, "bonus_rec_te": 0.5},
}

# Fantasy positions the Start/Sit advisor ranks.
_BUNDLE_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")

#: Injury-exit cleansing (Kaedon, 2026-11-27): a played week whose
#: snap share falls under this line, where the player then missed his
#: team's next game, is a week he left hurt. It drops out of the
#: played-week sample the rating's windows are derived from (see
#: derive_start_sit_inputs).
INJURY_EXIT_SNAP_SHARE = 0.25

#: Contested-room gate (Kaedon, 2026-11-27): once the room lead has
#: missed at least this many consecutive team games while the
#: non-lead played them, the replacement held a real role for a
#: stretch -- the room is contested and the lead filter is not
#: derived for the pair (see derive_start_sit_inputs).
CONTESTED_ROOM_MIN_LEAD_ABSENCE = 3

_TABLE_READY = False


def _num(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def variant_for_exact_scoring(scoring_settings: Optional[dict]) -> Optional[str]:
    """Return the precompute variant covering this league's scoring, or None.

    None means the league's scoring is exotic relative to the twelve canonical
    variants (fractional reception points, non-4/6 pass-TD value, a TE premium
    other than 0.5, custom yardage rates, milestone/first-down bonuses, ...).
    Callers must fall back to live per-request scoring for those leagues so
    values stay exact.
    """
    from utils.fantasy_scoring import _DEFAULT_RATES
    from utils.proj_variant import pick_proj_variant

    s = scoring_settings or {}
    # The exactness gate still requires "rec" explicitly: a missing rate now
    # defaults to PPR (1.0) consistently across the picker and the stat
    # scorer, but an exact-match fast path must not guess which canonical
    # variant a partial settings dict meant.
    if "rec" not in s:
        return None
    variant = pick_proj_variant(s)
    canon = VARIANT_SCORING.get(variant)
    if not canon:
        return None
    if _num(s.get("rec"), 1.0) != canon["rec"]:
        return None
    if _num(s.get("pass_td"), 4.0) != canon["pass_td"]:
        return None
    if _num(s.get("bonus_rec_te"), 0.0) != canon["bonus_rec_te"]:
        return None
    for key, standard_rate in _DEFAULT_RATES.items():
        if key in ("rec", "pass_td"):
            # Variant dimensions, already checked exactly against the canonical
            # variant above (rec / pass_td / bonus_rec_te).
            continue
        if key in s and _num(s.get(key), standard_rate) != float(standard_rate):
            return None
    for key, value in s.items():
        if key == "bonus_rec_te":
            continue
        if not (str(key).startswith("bonus_") or key in ("pass_fd", "rush_fd", "rec_fd")):
            continue
        try:
            if float(value or 0) != 0:
                return None
        except (TypeError, ValueError):
            return None
    return variant


# ---------------------------------------------------------------------------
# Postgres table
# ---------------------------------------------------------------------------

def init_start_score_db() -> None:
    """Create the bundle table if needed. Never raises."""
    global _TABLE_READY
    if _TABLE_READY:
        return
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS start_score_bundle (
                    season    INTEGER NOT NULL,
                    week      INTEGER NOT NULL,
                    player_id TEXT    NOT NULL,
                    variant   TEXT    NOT NULL,
                    bundle    JSONB   NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (season, week, player_id, variant)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_ssb_lookup "
                "ON start_score_bundle (season, week, variant)"
            )
        _TABLE_READY = True
    except Exception:
        logger.debug("start_score_bundle init failed", exc_info=True)


def _sleeper_stats_week_num(path: str) -> int:
    m = re.search(r"_w(\d+)", os.path.basename(str(path)))
    try:
        return int(m.group(1)) if m else -1
    except (TypeError, ValueError):
        return -1


def _stat_files(season: int) -> List[str]:
    from utils.paths import CACHE_DIR
    pattern = os.path.join(CACHE_DIR, "sleeper_stats", f"sleeper_stats_s{int(season)}_w*.json")
    return sorted(glob.glob(pattern), key=_sleeper_stats_week_num)


def _variant_week_points(season: int, scoring: dict):
    """Score one season's weekly stat files under canonical variant scoring.

    Returns ``(weekstat, weekly)`` mirroring the two app.py loaders the
    Start/Sit page uses:

    - ``weekstat``: ``{pid: {"sum":, "n":, "last4": [...]}}`` — all files in
      glob order, ``> 0`` filter, ``week_stat_points`` scorer with no TE flag
      (the page calls ``_season_weekstat_points(season, scoring)`` without a
      position, so TE premium never applies here).
    - ``weekly``: ``{pid: [pts...]}`` in completed-week order, ``score_stats``
      scorer *with* the stat line's position (the page's
      ``_load_season_weekly_points``), no ``> 0`` filter — every appearance
      counts as a game, but did-not-play rows (bookkeeping-only Sleeper rows
      for active players with zero snaps) are skipped, mirroring the page.

    Never raises; returns empty maps on failure.
    """
    from utils.fantasy_scoring import (
        score_stats, stat_row_is_appearance, week_stat_points,
    )
    from utils.season_qualification import qualification_policy

    weekstat: Dict[str, dict] = {}
    weekly: Dict[str, list] = {}
    try:
        completed = set(qualification_policy(int(season)).completed_weeks or ())
    except Exception:
        completed = set()
    try:
        files = _stat_files(season)
        last_idx = list(range(max(0, len(files) - 4), len(files)))
        for widx, wf in enumerate(files):
            try:
                with open(wf) as f:
                    wdata = json.load(f)
            except Exception:
                continue
            if not isinstance(wdata, dict):
                continue
            wnum = _sleeper_stats_week_num(wf)
            in_last4 = widx in last_idx
            for pid, st in wdata.items():
                if not isinstance(st, dict):
                    continue
                pid = str(pid)
                pos = str(st.get("position") or st.get("pos") or "")
                try:
                    pts_ws = float(week_stat_points(st, scoring, "") or 0)
                except Exception:
                    pts_ws = 0.0
                if pts_ws > 0:
                    rec = weekstat.setdefault(pid, {"sum": 0.0, "n": 0, "last4": {}})
                    rec["sum"] += pts_ws
                    rec["n"] += 1
                    if in_last4:
                        rec["last4"][widx] = pts_ws
                if wnum in completed and stat_row_is_appearance(st):
                    try:
                        pts_w = float(score_stats(st, scoring, pos))
                    except Exception:
                        continue
                    weekly.setdefault(pid, []).append(pts_w)
        # Normalize last4 to a dense 4-list in file order (None = no line).
        for rec in weekstat.values():
            by_idx = rec.pop("last4")
            rec["last4"] = [by_idx.get(wi) for wi in last_idx]
            while len(rec["last4"]) < 4:
                rec["last4"].append(None)
    except Exception:
        logger.debug("start_score_bundle weekly scoring failed", exc_info=True)
    return weekstat, weekly


_DERIVE_CACHE: Dict = {}
_DERIVE_CACHE_MAX = 16


def _expected_column_for_scoring(scoring_settings: Optional[dict]) -> Optional[str]:
    """Weekly expected-points column matching ``scoring_settings``.

    The site's expected-points model stores three totals per
    player-week (full PPR / half PPR / standard), so the column is
    picked by the reception rate. Returns None for a non-4-point
    passing-TD rate: the model prices passing TDs at 4, so under a
    6-point rate its expectation is not like-for-like with the
    projection and the projection cap must not fire on it (the cap
    was validated on the 4-point configuration).
    """
    s = scoring_settings or {}
    if _num(s.get("pass_td"), 4.0) != 4.0:
        return None
    rec = _num(s.get("rec"), 1.0)
    if rec >= 0.75:
        return "expected_ppr"
    if rec >= 0.25:
        return "expected_half_ppr"
    return "expected_standard"


def _load_weekly_expected_map(season: int, column: Optional[str]) -> Dict[str, dict]:
    """Bulk-read one expected-points column of the weekly table.

    Returns {player_id: {week: expected}} from
    ``player_weekly_advanced_metrics`` -- the table the daily cron's
    nflverse sync fills from the site's expected-points model
    (``build_expected_points_both``) ahead of the bundle build, and
    the same stored series the player pages read. ``column`` only
    ever comes from :func:`_expected_column_for_scoring`'s fixed
    literals. Never raises: {} on any failure (no cap evidence).
    """
    if not column:
        return {}
    try:
        from data_building.advanced_metrics import (
            init_weekly_advanced_metrics_db)
        from dashboard_services.db import get_conn

        init_weekly_advanced_metrics_db()
        with get_conn() as conn:
            rows = conn.execute(
                f"SELECT player_id, week, {column} "
                "FROM player_weekly_advanced_metrics WHERE season = %s",
                (int(season),),
            ).fetchall()
        out: Dict[str, dict] = {}
        for r in rows or []:
            if r[column] is None:
                continue
            out.setdefault(str(r["player_id"]), {})[int(r["week"])] = (
                float(r[column]))
        return out
    except Exception:
        logger.debug("start_score_bundle weekly expected read failed",
                     exc_info=True)
        return {}


def derive_start_sit_inputs(season: int, scoring_settings: dict) -> Dict[str, dict]:
    """Per-player form/usage inputs for the start/sit rating, derived from
    the cached weekly stat files under ``scoring_settings``.

    A week counts as PLAYED when the player actually participated:
    ``gp`` >= 1 or an offensive snap count (``off_snp``) present. A
    participated week with no production — Sleeper emits no points line
    (``pts_ppr``) for it at all — is a real 0.0 game: it scores 0.0 with
    ``week_stat_points`` and enters every average as 0.0, with 0 usage.
    Rows with neither participation marker are did-not-play /
    bookkeeping rows and never enter any average. Points are scored with
    ``week_stat_points`` (the same scorer the Start/Sit page's live path
    uses), recent form is the last 3 played weeks, and the usage delta
    is the absolute per-game change of the position's volume metric (see
    ``utils.start_sit.form_usage_inputs``).
    The derivation also carries ``last4_ppg`` (last 4 played weeks):
    the rating's form-blend input (``compute_start_score`` blends it
    into the base at 35% on a full window), and the value the compare
    rows labeled "L4 PPG" display. It also carries ``snap_trend``:
    the role-trend input (snap-share change over the last 4 played
    weeks, see ``utils.start_sit.snap_share_trend``), derived on the
    same participation rule. RB/WR/TE
    entries for non-lead players also carry ``lead_pid`` /
    ``last4_ppg_lead``: the room lead (highest season usage per played
    game at his team + position) and the player's last-4 average over
    only the weeks that lead also played — the handcuff form filter
    the ranked pools apply when the lead is active (see
    ``utils.start_sit.lead_adjusted_last4_ppg``). The same entries
    carry the shared-games versions of the remaining rating inputs
    (``season_ppg_lead`` / ``recent_ppg_lead`` / ``usage_delta_lead``
    / ``snap_trend_lead``) for the projection blend and input swaps
    the pools apply alongside it (see
    ``utils.start_sit.lead_adjusted_score_inputs``). Entries also
    carry ``absence_weeks``: consecutive non-bye weeks missed
    immediately before the newest stat-file week -- the
    first-game-back discount's input (bye weeks never count, and no
    value is derived when a bye cannot be ruled out for the gap).

    Two gates clean the played-week sample before windows are taken
    from it (Kaedon, 2026-11-27). An INJURY EXIT -- a played week
    with a snap share under 25% where the player then missed his
    team's next game -- is a week he left hurt: it is not evidence
    of his form or his role, so it enters no form window, no snap
    trend and not the projection cap's last-4 window. The absence
    count is the one reader that keeps the raw weeks (the exit week
    is a game played, not a game missed). And the lead filter is
    derived only for an uncontested room: when the room lead missed
    3+ consecutive team games while the non-lead played them, the
    replacement's form over that stretch is his own, not borrowed
    from the lead's absence, so no ``lead_pid`` / ``*_lead`` values
    are derived for the pair and both players score on raw inputs.

    This is the single derivation the bundle build and the Start/Sit page's
    live fallback share, so both paths feed ``compute_start_score`` identical
    inputs. Cached per (season, scoring signature, newest stat-file mtime).
    Never raises: {} on failure.
    """
    import hashlib

    try:
        files = sorted(_stat_files(int(season)), key=_week_num)
    except Exception:
        return {}
    newest = 0.0
    for path in files:
        try:
            newest = max(newest, os.path.getmtime(path))
        except OSError:
            pass
    sig = hashlib.md5(
        json.dumps(scoring_settings or {}, sort_keys=True, default=str).encode()
    ).hexdigest()[:10]
    key = (int(season), sig, newest)
    hit = _DERIVE_CACHE.get(key)
    if hit is not None:
        return hit
    out: Dict[str, dict] = {}
    try:
        from utils.fantasy_scoring import week_stat_points
        from utils.start_sit_score import (
            form_usage_inputs, snap_share_trend, usage_value_from_stats)
        from utils.utils import load_players_index

        try:
            index = load_players_index() or {}
        except Exception:
            index = {}
        positions = {
            str(pid): str((meta or {}).get("pos") or (meta or {}).get("position") or "")
            for pid, meta in index.items()
        }
        teams = {
            str(pid): str((meta or {}).get("team") or "").upper()
            for pid, meta in index.items()
        }
        played_points: Dict[str, list] = {}
        played_weeks: Dict[str, list] = {}
        # Teams with at least one played row per week, on the same
        # participation rule as played_weeks: the bye evidence the
        # first-game-back absence count below is built from (a week
        # where a team shows no played row is a week that team did
        # not play -- its bye, or indistinguishable from one).
        team_played: Dict[int, set] = {}
        # (week, team) pairs where the team's offense demonstrably
        # took the field -- a QB/RB/WR/TE recorded offensive snaps.
        # The two 2026-11-27 gates below read team calendars built
        # from this, not from team_played: a bye week still carries
        # stray bookkeeping rows (defenders marked gp=1 with no
        # snaps, an occasional line row) that put the team in
        # team_played without the team having played a game.
        team_game_seen: set = set()
        played_usage: Dict[str, list] = {}
        played_shares: Dict[str, list] = {}
        played_share_weeks: Dict[str, list] = {}
        played_recs: Dict[str, list] = {}
        played_pos: Dict[str, str] = {}
        for path in files:
            wnum = _week_num(path)
            try:
                with open(path) as fh:
                    payload = json.load(fh)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            for pid, st in payload.items():
                if not isinstance(st, dict):
                    continue
                # Snap shares for the role-trend factor, on the trend's
                # own played rule (game participation), independent of
                # the points line below. A played week whose share
                # cannot be computed does not enter the window.
                try:
                    _gp = st.get("gp")
                    if ((_gp is not None and float(_gp) >= 1)
                            or "off_snp" in st):
                        _off = st.get("off_snp")
                        _tm = st.get("tm_off_snp")
                        if _off is not None and _tm:
                            played_shares.setdefault(str(pid), []).append(
                                float(_off) / float(_tm))
                            played_share_weeks.setdefault(
                                str(pid), []).append(wnum)
                except (TypeError, ValueError):
                    pass
                # Played-week rule (participation): gp >= 1 or an
                # offensive snap count present. A participated week with
                # no points line scores 0.0 below and counts as a game;
                # a row with neither marker is a did-not-play row.
                try:
                    _played_week = float(st.get("gp") or 0) >= 1 \
                        or st.get("off_snp") is not None
                except (TypeError, ValueError):
                    _played_week = st.get("off_snp") is not None
                if not _played_week:
                    continue  # did not play (no gp, no snaps)
                pid = str(pid)
                _team = teams.get(pid) or ""
                if _team:
                    team_played.setdefault(wnum, set()).add(_team)
                try:
                    pts = float(week_stat_points(st, scoring_settings, ""))
                except Exception:
                    continue
                pos = positions.get(pid) or str(st.get("position") or st.get("pos") or "")
                if _team and pos.upper() in ("QB", "RB", "WR", "TE"):
                    try:
                        _off = float(st.get("off_snp") or 0)
                    except (TypeError, ValueError):
                        _off = 0.0
                    if _off > 0:
                        team_game_seen.add((wnum, _team))
                played_points.setdefault(pid, []).append(pts)
                played_weeks.setdefault(pid, []).append(wnum)
                played_usage.setdefault(pid, []).append(
                    usage_value_from_stats(st, pos))
                try:
                    _rec = float(st.get("rec") or 0)
                except (TypeError, ValueError):
                    _rec = 0.0
                played_recs.setdefault(pid, []).append(_rec)
                played_pos[pid] = pos
        # Team game calendars: the weeks each team's offense
        # actually took the field (see ``team_game_seen``). The two
        # gates below read them -- a week his team did not play (its
        # bye) is on nobody's calendar, so it neither breaks nor
        # extends a run of missed team games, and the game after an
        # injury exit is the next game the team really played.
        team_weeks: Dict[str, list] = {}
        for _w, _t in team_game_seen:
            team_weeks.setdefault(_t, []).append(_w)
        for _cal in team_weeks.values():
            _cal.sort()
        # Injury-exit weeks (Kaedon, 2026-11-27): a played week with
        # a snap share under INJURY_EXIT_SNAP_SHARE where the player
        # then missed his team's next game is a week he left hurt
        # (David Montgomery's Week 5: 13.6% of the snaps, then out
        # four games -- yet the week sat in his form and snap-trend
        # windows and scored him as a cold, shrinking player the week
        # after a 15.4-point return). The week drops out of every
        # window derived from the played lists below: the form
        # inputs, the projection cap's last-4 window and the snap
        # trend. The absence count further down is the exception --
        # it reads the RAW played weeks, because the exit week is a
        # game he played, not one he missed. A week whose share
        # cannot be computed, or with no later team game in the data
        # to check the absence against, is never an exit: no
        # evidence, no exclusion.
        exit_weeks: Dict[str, set] = {}
        for _pid, _weeks in played_weeks.items():
            _share_by_week = dict(zip(
                played_share_weeks.get(_pid) or [],
                played_shares.get(_pid) or []))
            _cal = team_weeks.get(teams.get(_pid) or "") or []
            _played_set = set(_weeks)
            _exits = set()
            for _w in _weeks:
                _share = _share_by_week.get(_w)
                if _share is None or _share >= INJURY_EXIT_SNAP_SHARE:
                    continue
                _next = next((tw for tw in _cal if tw > _w), None)
                if _next is not None and _next not in _played_set:
                    _exits.add(_w)
            if _exits:
                exit_weeks[_pid] = _exits

        def _kept(_pid):
            _exits = exit_weeks.get(_pid)
            _weeks = played_weeks.get(_pid) or []
            if not _exits:
                return list(range(len(_weeks)))
            return [i for i, w in enumerate(_weeks) if w not in _exits]

        out = {}
        for pid, pts in played_points.items():
            _keep = _kept(pid)
            _usage = played_usage.get(pid) or []
            out[pid] = form_usage_inputs(
                [pts[i] for i in _keep],
                [_usage[i] if i < len(_usage) else None for i in _keep])
        # Projection-cap evidence (Kaedon, 2026-11-20): each player's
        # expected PPG over his last 4 played weeks from the site's
        # expected-points model, read from the weekly advanced-metrics
        # table the daily nflverse sync fills (see
        # _load_weekly_expected_map). The window is the same played
        # weeks the rating's other inputs use: a played week with no
        # expected row counts 0.0 (he was on the field with no
        # opportunity), and a player whose window carries no expected
        # row at all gets no value -- no evidence, no cap. Under a
        # TE-premium scoring the premium on his window receptions is
        # added, so the expectation stays like-for-like with the
        # projection the cap compares it against. The scoring paths
        # apply the cap via utils.start_sit.cap_projection; the raw
        # projection is never rewritten anywhere.
        try:
            _exp_col = _expected_column_for_scoring(scoring_settings)
            _exp_map = (_load_weekly_expected_map(season, _exp_col)
                        if _exp_col else {})
            if _exp_map:
                _tep_bonus = _num(
                    (scoring_settings or {}).get("bonus_rec_te"), 0.0)
                for _pid, _entry in out.items():
                    _weeks = played_weeks.get(_pid) or []
                    if not _weeks:
                        continue
                    # The cap reads the same cleansed last-4 played
                    # list as the form windows (injury exits out).
                    _keep = _kept(_pid)
                    _win = [_weeks[i] for i in _keep][-4:]
                    if not _win:
                        continue
                    _rows = _exp_map.get(_pid) or {}
                    if not any(w in _rows for w in _win):
                        continue
                    _exp = (sum(_rows.get(w, 0.0) for w in _win)
                            / len(_win))
                    if (_tep_bonus
                            and (played_pos.get(_pid) or "").upper() == "TE"):
                        _all_recs = played_recs.get(_pid) or []
                        _recs = [_all_recs[i] for i in _keep
                                 if i < len(_all_recs)][-len(_win):]
                        if _recs:
                            _exp += _tep_bonus * (sum(_recs) / len(_recs))
                    _entry["exp_last4_ppg"] = _exp
        except Exception:
            logger.debug("start_score_bundle exp derivation failed",
                         exc_info=True)
        # First-game-back absence (Kaedon, 2026-11-20): consecutive
        # weeks missed immediately before the newest stat-file week,
        # counting only weeks his team actually played. A week where
        # his team shows no played row is that team's bye (or a week
        # the data cannot distinguish from one) and never counts --
        # a bye is not an injury absence. No team on the index, or a
        # last played game in the newest week, derives no value: no
        # evidence of a layoff, no discount (the scoring paths shade
        # the projection via utils.start_sit
        # .return_adjusted_projection). A player with no played game
        # at all is not in ``out`` and gets nothing either: a debut
        # is not a return.
        try:
            _file_weeks = sorted({_week_num(p) for p in files})
            _last_week = _file_weeks[-1] if _file_weeks else 0
            if _last_week:
                for _pid, _entry in out.items():
                    _weeks = played_weeks.get(_pid) or []
                    if not _weeks or _weeks[-1] >= _last_week:
                        continue
                    _team = teams.get(_pid) or ""
                    if not _team:
                        continue
                    _missed = sum(
                        1 for _w in range(_weeks[-1] + 1,
                                           _last_week + 1)
                        if _team in team_played.get(_w, ()))
                    if _missed >= 1:
                        _entry["absence_weeks"] = _missed
        except Exception:
            logger.debug("start_score_bundle absence derivation failed",
                         exc_info=True)
        # Room leads + lead-filtered form (Kaedon, 2026-11-19): a
        # handcuff's last-4 must not be built on the starter's absence.
        # (2026-11-27: the contested-room gate inside the loop below
        # withholds the filter entirely when the lead's absence ran
        # 3+ team games with the non-lead playing -- a long-term
        # replacement's form is his own.)
        # For RB/WR/TE, group each team's room from the players index
        # and name the lead as the teammate with the highest season
        # usage per played game (opportunities for RB, targets for
        # WR/TE -- the same usage metric derived above, so no new data
        # source enters the pipeline; the lead must have a 2+ game
        # usage sample so a one-week stand-in never becomes the lead).
        # Every other measurable player in the room gets ``lead_pid``
        # and ``last4_ppg_lead``: his last-4 average over only the
        # weeks the lead also played (fallbacks in
        # utils.start_sit.lead_filtered_last4_ppg). Whether the
        # filtered value REPLACES the raw one is a current-week
        # decision -- the lead must be active this week -- made by the
        # scoring paths via utils.start_sit.lead_adjusted_last4_ppg.
        try:
            from utils.start_sit_score import lead_filtered_last4_ppg

            rooms: Dict = {}
            for _pid in played_points:
                _pos = (positions.get(_pid) or "").upper()
                _team = teams.get(_pid) or ""
                if _pos in ("RB", "WR", "TE") and _team:
                    rooms.setdefault((_team, _pos), []).append(_pid)
            for (_room_team, _room_pos), _room_pids in rooms.items():
                if len(_room_pids) < 2:
                    continue
                _usage_pg: Dict[str, tuple] = {}
                for _pid in _room_pids:
                    _u = [u for u in (played_usage.get(_pid) or [])
                          if u is not None]
                    if _u:
                        _usage_pg[_pid] = (sum(_u) / len(_u), sum(_u),
                                           len(_u))
                _eligible = [p for p, v in _usage_pg.items() if v[2] >= 2]
                if not _eligible:
                    continue
                _lead = max(
                    _eligible,
                    key=lambda p: (_usage_pg[p][0], _usage_pg[p][1], p))
                _lead_weeks = set(played_weeks.get(_lead) or [])
                for _pid in _room_pids:
                    if _pid == _lead or _pid not in _usage_pg:
                        continue
                    if _usage_pg[_lead][0] <= _usage_pg[_pid][0]:
                        continue  # co-lead on usage: no filter
                    # Contested-room gate (Kaedon, 2026-11-27): the
                    # filter prices the non-lead's opportunity as
                    # borrowed from the lead, which stops being true
                    # once the lead missed a real stretch while the
                    # non-lead held the job. Woody Marks ran Houston's
                    # backfield for four games while David Montgomery
                    # was out, yet the filter priced him off only the
                    # five early games they shared -- and named
                    # Montgomery the lead on season usage built from
                    # Week 1 and his return game. When the lead missed
                    # CONTESTED_ROOM_MIN_LEAD_ABSENCE or more
                    # consecutive team games that the non-lead played,
                    # the room is contested: no filter is derived for
                    # the pair and both score on their raw inputs. A
                    # one- or two-game fill-in (Emmett Johnson for
                    # Kenneth Walker III, two games) stays filtered.
                    _cal = team_weeks.get(_room_team) or []
                    _pid_weeks = set(played_weeks.get(_pid) or [])
                    _run = 0
                    _contested = False
                    for _tw in _cal:
                        if _tw in _pid_weeks and _tw not in _lead_weeks:
                            _run += 1
                            if _run >= CONTESTED_ROOM_MIN_LEAD_ABSENCE:
                                _contested = True
                                break
                        else:
                            _run = 0
                    if _contested:
                        continue
                    _entry = out.get(_pid)
                    if _entry is None:
                        continue
                    _entry["lead_pid"] = _lead
                    _entry["last4_ppg_lead"] = lead_filtered_last4_ppg(
                        played_points.get(_pid), played_weeks.get(_pid),
                        _lead_weeks)
                    # The rest of the rating's inputs over the same
                    # shared games (Kaedon, 2026-11-19): what the
                    # player scores per game alongside his lead, his
                    # recent form and usage delta in those games, and
                    # his snap trend across them. The scoring paths
                    # swap these in (and blend the projection toward
                    # season_ppg_lead) only while the lead is active --
                    # see utils.start_sit.lead_adjusted_score_inputs.
                    # No shared game at all: nothing is derived and
                    # the raw inputs stand, as with the last-4 above.
                    _p_weeks = played_weeks.get(_pid) or []
                    _shared_idx = [i for i, w in enumerate(_p_weeks)
                                   if w in _lead_weeks]
                    if _shared_idx:
                        _p_usage = played_usage.get(_pid) or []
                        _lead_inputs = form_usage_inputs(
                            [played_points[_pid][i] for i in _shared_idx],
                            [_p_usage[i] if i < len(_p_usage) else None
                             for i in _shared_idx])
                        _entry["season_ppg_lead"] = _lead_inputs[
                            "season_ppg"]
                        _entry["recent_ppg_lead"] = _lead_inputs[
                            "recent_ppg"]
                        _entry["usage_delta_lead"] = _lead_inputs[
                            "usage_delta"]
                        _s_weeks = played_share_weeks.get(_pid) or []
                        _entry["snap_trend_lead"] = snap_share_trend(
                            [s for s, w in zip(
                                played_shares.get(_pid) or [], _s_weeks)
                             if w in _lead_weeks])
        except Exception:
            logger.debug("start_score_bundle lead-form derivation failed",
                         exc_info=True)
        for _pid, _entry in out.items():
            _exits = exit_weeks.get(_pid)
            if not _exits:
                _entry["snap_trend"] = snap_share_trend(
                    played_shares.get(_pid))
            else:
                # The trend window is cleansed like the form windows:
                # an injury-exit week is not role evidence.
                _entry["snap_trend"] = snap_share_trend(
                    [s for s, w in zip(played_shares.get(_pid) or [],
                                       played_share_weeks.get(_pid) or [])
                     if w not in _exits])
    except Exception:
        logger.debug("start_score_bundle input derivation failed", exc_info=True)
        out = {}
    if len(_DERIVE_CACHE) >= _DERIVE_CACHE_MAX:
        _DERIVE_CACHE.clear()
    _DERIVE_CACHE[key] = out
    return out


def _week_num(path: str) -> int:
    m = re.search(r"_w(\d+)\.json$", str(path))
    return int(m.group(1)) if m else 0


def _oline_index_for(season: int, ratings: dict, team: str, pos: str) -> Optional[float]:
    """Position-relevant 0-100 O-line index, mirroring app._oline_for_player."""
    try:
        from data_building.oline_ratings import _norm_team
        team = _norm_team(team)
    except Exception:
        team = (team or "").upper().strip()
    if not team:
        return None
    row = (ratings or {}).get(team)
    if not row:
        return None
    pos = (pos or "").upper().strip()
    key = ("run_block" if pos == "RB"
           else "pass_block" if pos in ("QB", "WR", "TE")
           else "composite")
    try:
        return float(row.get(key)) if row.get(key) is not None else None
    except (TypeError, ValueError):
        return None


def _load_oline_ratings(season: int) -> dict:
    # Database first (the weekly cron persists here; its container's cache/
    # files are ephemeral), then the flat file, then the newest built season.
    try:
        from dashboard_services.oline_store import (
            load_oline_ratings as _db_load_oline,
            newest_oline_season as _db_newest_oline,
        )

        row = _db_load_oline(int(season))
        if row and row.get("ratings"):
            return row["ratings"]
        newest = _db_newest_oline()
        if newest is not None and newest != int(season):
            row = _db_load_oline(newest)
            if row and row.get("ratings"):
                return row["ratings"]
    except Exception:
        pass
    try:
        path = os.path.join("cache", f"oline_ratings_s{int(season)}.json")
        if os.path.exists(path):
            with open(path) as f:
                return (json.load(f) or {}).get("ratings") or {}
    except Exception:
        pass
    # Fall back to the newest built season (same policy as the page).
    try:
        newest = None
        for fn in os.listdir("cache"):
            if fn.startswith("oline_ratings_s") and fn.endswith(".json"):
                try:
                    yr = int(fn[len("oline_ratings_s"):-len(".json")])
                except ValueError:
                    continue
                newest = yr if newest is None else max(newest, yr)
        if newest is not None:
            with open(os.path.join("cache", f"oline_ratings_s{newest}.json")) as f:
                return (json.load(f) or {}).get("ratings") or {}
    except Exception:
        pass
    return {}

# ---------------------------------------------------------------------------
# Batch build
# ---------------------------------------------------------------------------

def build_start_score_bundles(season: int = None, week: int = None) -> dict:
    """Build one input bundle per (player, variant) for the current week.

    Gathers every ``compute_start_score`` input the Start/Sit page uses, once,
    and upserts it into ``start_score_bundle``. Runs in the daily cron after
    the weekly projections step. Prunes bundles for older (season, week) so the
    table only ever holds the current week. Never raises; returns a summary.
    """
    from dashboard_services.api import get_nfl_state
    from dashboard_services.db import get_conn
    from dashboard_services.team_play_volume import load_team_play_volume
    from data_building.fetch_projections import PROJ_VARIANTS
    from data_building.weekly_metrics import get_usage_trends
    from utils.consistency import BLEND_FULL_SEASON, blended_consistency_profile
    from utils.fantasy_scoring import weekly_projection_points
    from utils.game_conditions import build_week_conditions
    from utils.league_scoring import stamp_scoring_aliases
    from utils.start_sit_context import expected_plays_context, role_confidence_from_trend
    from utils.start_sit_score import bottom_teams_by_implied_total
    from utils.utils import load_players_index, load_week_projection, load_week_sched

    t0 = time.time()
    summary: dict = {"ok": False, "season": season, "week": week}
    try:
        state = get_nfl_state() or {}
        season = int(season or state.get("season") or 0)
        week = int(week or state.get("week") or 0)
        if not season or not week:
            summary["error"] = "no season/week"
            return summary
        summary["season"], summary["week"] = season, week

        players_index = load_players_index() or {}

        # ── Schedule: opponent map + (home, away, gameDate) for conditions ──
        opponent_map: Dict[str, str] = {}
        week_games: list = []
        try:
            for game in (load_week_sched(season, week) or []):
                home = str(game.get("home") or "").upper()
                away = str(game.get("away") or "").upper()
                if home and away:
                    opponent_map[home] = away
                    opponent_map[away] = home
                    week_games.append((home, away, str(game.get("gameDate") or "")))
        except Exception:
            logger.debug("start_score_bundle schedule failed", exc_info=True)

        # ── League-independent inputs, loaded once ──────────────────────────
        try:
            usage_trends = get_usage_trends(season) or {}
        except Exception:
            usage_trends = {}
        try:
            game_conditions = build_week_conditions(season, week, week_games) if week_games else {}
        except Exception:
            game_conditions = {}
        # Rank-based "Low team total" demotion: bottom-8 implied team totals,
        # computed once per batch. Bye-week teams have no conditions entry and
        # can never be in the set.
        low_total_teams = bottom_teams_by_implied_total(game_conditions)
        try:
            _tpv_blob = load_team_play_volume(season) or {}
            team_play_volume = _tpv_blob.get("teams") or {}
            tpv_nfl_avg = _tpv_blob.get("nfl_avg_plays_faced_pg")
        except Exception:
            team_play_volume, tpv_nfl_avg = {}, None
        oline_ratings = _load_oline_ratings(season)
        try:
            week_proj_map = load_week_projection(season, week) or {}
        except Exception:
            week_proj_map = {}

        # ── Player universe: index fantasy positions + projection entries ───
        universe: Dict[str, dict] = {}
        for pid, meta in players_index.items():
            pos = str((meta or {}).get("pos") or (meta or {}).get("position") or "").upper()
            if pos in _BUNDLE_POSITIONS:
                universe[str(pid)] = {
                    "pos": pos,
                    "team": str((meta or {}).get("team") or "").upper(),
                    "injury_status": str((meta or {}).get("injury_status")
                                         or (meta or {}).get("status") or ""),
                }
        for pid, entry in (week_proj_map or {}).items():
            pid = str(pid)
            if pid not in universe and isinstance(entry, dict):
                meta = players_index.get(pid) or {}
                pos = str(meta.get("pos") or meta.get("position") or "").upper()
                if pos in _BUNDLE_POSITIONS:
                    universe[pid] = {
                        "pos": pos,
                        "team": str(meta.get("team") or "").upper(),
                        "injury_status": str(meta.get("injury_status")
                                             or meta.get("status") or ""),
                    }

        # ── Per-variant scoring-dependent inputs ────────────────────────────
        variants = [v for v in PROJ_VARIANTS if v in VARIANT_SCORING]
        per_variant: dict = {}
        for variant in variants:
            scoring = stamp_scoring_aliases(VARIANT_SCORING[variant])
            weekstat, weekly = _variant_week_points(season, scoring)
            # Prior-season crossfade for consistency (same BLEND_FULL_SEASON
            # gate as the page: only while the current sample is short).
            prior_weekly: Dict[str, list] = {}
            try:
                if weekly:
                    for py in (season - 1, season - 2):
                        _, pweekly = _variant_week_points(py, scoring)
                        if pweekly and max((len(v) for v in pweekly.values()), default=0) >= 3:
                            prior_weekly = pweekly
                            break
            except Exception:
                prior_weekly = {}
            per_variant[variant] = {
                "scoring": scoring, "weekstat": weekstat, "weekly": weekly,
                "prior_weekly": prior_weekly,
                # Played-week form/usage inputs for the rating (shared with
                # the Start/Sit page's live fallback derivation).
                "derived": derive_start_sit_inputs(season, scoring),
            }

        # ── Assemble bundles ────────────────────────────────────────────────
        rows: list = []
        for pid, u in universe.items():
            pos = u["pos"]
            team = u["team"]
            opponent = opponent_map.get(team, "")
            on_bye = bool(opponent_map) and team not in opponent_map
            ut = usage_trends.get(pid) or {}
            cond = (game_conditions.get(team) or {}) if not on_bye else {}
            implied_total = cond.get("implied_total")
            wx = cond.get("weather") or {}
            weather_kind = wx.get("kind") if isinstance(wx, dict) else None
            # The tag label already carries the specifics ("22 mph wind");
            # the bundle keeps it so rows can show it without a live lookup.
            weather_label = wx.get("label") if isinstance(wx, dict) else None
            oline_index = _oline_index_for(season, oline_ratings, team, pos)
            pace = expected_plays_context(team_play_volume, team, opponent, tpv_nfl_avg) or {}
            role_conf = role_confidence_from_trend(ut)
            for variant in variants:
                pv = per_variant[variant]
                scoring = pv["scoring"]
                # Projection: this week's Sleeper line only, scored with the
                # variant's settings (same weekly_projection_points selection
                # the page uses: week map + player id). A player absent from
                # the week file, or present with an explicit 0, has no line
                # this week (bye, doubtful/out, inactive) -- Sleeper shows
                # 0.0 for them. Never substitute season PPG here: that gave
                # zeroed-out players their full projection and the start
                # nod over healthy players. Season PPG stays on the bundle
                # as its own field (season_ppg below) for form context.
                proj_pts = 0.0
                try:
                    _wk_pts = weekly_projection_points(
                        week_proj_map, pid, scoring, pos)
                    if _wk_pts is not None:
                        proj_pts = float(_wk_pts)
                except Exception:
                    proj_pts = 0.0
                _inp = (pv.get("derived") or {}).get(pid) or {}
                season_ppg = round(float(_inp.get("season_ppg") or 0.0), 1)
                recent_ppg = round(float(_inp.get("recent_ppg") or 0.0), 1)
                last4_ppg = round(float(_inp.get("last4_ppg") or 0.0), 1)
                bust_rate = None
                try:
                    cur = pv["weekly"].get(pid) or []
                    prior = pv["prior_weekly"].get(pid) or []
                    cons = None
                    if cur or prior:
                        cons = blended_consistency_profile(cur, prior, pos)
                    if cons and not cons.get("small_sample"):
                        bust_rate = cons.get("bust_rate")
                except Exception:
                    bust_rate = None
                bundle = {
                    "pos": pos,
                    "team": team,
                    "opponent": opponent or None,
                    "on_bye": bool(on_bye),
                    "proj_pts": round(proj_pts, 1),
                    "recent_ppg": recent_ppg,
                    "last4_ppg": last4_ppg,
                    # Lead-filtered form candidate + the lead it was
                    # filtered on (2026-11-19 handcuff rule): the
                    # ranked pools swap this in for last4_ppg when the
                    # lead is active this week. See utils.start_sit.
                    "last4_ppg_lead": (
                        round(float(_inp["last4_ppg_lead"]), 1)
                        if _inp.get("last4_ppg_lead") is not None else None),
                    "lead_pid": _inp.get("lead_pid") or None,
                    # Shared-games versions of the remaining rating
                    # inputs (same handcuff rule, extended): the pools
                    # blend the projection toward season_ppg_lead and
                    # swap the rest in while the lead is active. See
                    # utils.start_sit.lead_adjusted_score_inputs.
                    "season_ppg_lead": (
                        round(float(_inp["season_ppg_lead"]), 1)
                        if _inp.get("season_ppg_lead") is not None
                        else None),
                    "recent_ppg_lead": (
                        round(float(_inp["recent_ppg_lead"]), 1)
                        if _inp.get("recent_ppg_lead") is not None
                        else None),
                    "usage_delta_lead": (
                        round(float(_inp["usage_delta_lead"]), 1)
                        if _inp.get("usage_delta_lead") is not None
                        else None),
                    "snap_trend_lead": (
                        round(float(_inp["snap_trend_lead"]), 4)
                        if _inp.get("snap_trend_lead") is not None
                        else None),
                    # Projection-cap evidence (2026-11-20): expected
                    # PPG over the last 4 played weeks from the site's
                    # expected-points model. The ranked pools cap an
                    # outlier projection against it before scoring
                    # (utils.start_sit.cap_projection); scoring-
                    # internal only -- proj_pts stays raw.
                    "exp_last4_ppg": (
                        round(float(_inp["exp_last4_ppg"]), 2)
                        if _inp.get("exp_last4_ppg") is not None
                        else None),
                    # First-game-back absence (2026-11-20):
                    # consecutive non-bye weeks missed immediately
                    # before this week. The ranked pools shade the
                    # scoring projection by it after the cap
                    # (utils.start_sit.return_adjusted_projection);
                    # scoring-internal only -- proj_pts stays raw.
                    "absence_weeks": (
                        int(_inp["absence_weeks"])
                        if _inp.get("absence_weeks") is not None
                        else None),
                    "season_ppg": season_ppg,
                    "bust_rate": bust_rate,
                    # Rating inputs from the played-week derivation (NOT the
                    # weekly_metrics trend, which still feeds role confidence
                    # above): absolute per-game delta of opportunities (RB)
                    # or targets (WR/TE); None for QB/K/DEF.
                    "usage_delta": (
                        round(float(_inp["usage_delta"]), 1)
                        if _inp.get("usage_delta") is not None else None),
                    "usage_season_avg": (
                        round(float(_inp["usage_season_avg"]), 1)
                        if _inp.get("usage_season_avg") is not None else None),
                    "played_games": int(_inp.get("played_games") or 0),
                    # Role trend from the same derivation (snap-share
                    # change over the last 4 played weeks); None when
                    # the window is short — the rating treats it as
                    # neutral.
                    "snap_trend": (
                        round(float(_inp["snap_trend"]), 4)
                        if _inp.get("snap_trend") is not None else None),
                    "implied_total": implied_total,
                    "low_total_team": team in low_total_teams,
                    "weather_kind": weather_kind,
                    "weather_label": weather_label,
                    "oline_index": oline_index,
                    "expected_team_plays": pace.get("expected_team_plays"),
                    "league_average_plays": pace.get("league_average_plays"),
                    "role_confidence": role_conf,
                    # Batch-time status for reference; requests always overlay
                    # the live status before scoring.
                    "injury_status": u.get("injury_status") or None,
                }
                rows.append((season, week, pid, variant, json.dumps(bundle)))

        init_start_score_db()
        with get_conn() as conn:
            # Prune older weeks first so the table only holds the current one.
            conn.execute(
                "DELETE FROM start_score_bundle WHERE NOT (season = %s AND week = %s)",
                (season, week),
            )
            # psycopg3 only exposes executemany on cursors, not connections.
            with conn.cursor() as cur:
                for i in range(0, len(rows), 500):
                    chunk = rows[i:i + 500]
                    cur.executemany(
                        """
                        INSERT INTO start_score_bundle
                            (season, week, player_id, variant, bundle, updated_at)
                        VALUES (%s, %s, %s, %s, %s::jsonb, now())
                        ON CONFLICT (season, week, player_id, variant)
                        DO UPDATE SET bundle = EXCLUDED.bundle, updated_at = now()
                        """,
                        chunk,
                    )
        summary.update({
            "ok": True,
            "players": len(universe),
            "variants": len(variants),
            "rows": len(rows),
            "seconds": round(time.time() - t0, 1),
        })
        logger.info("start_score_bundle built: %s", summary)
        return summary
    except Exception as exc:
        logger.warning("build_start_score_bundles failed", exc_info=True)
        summary["error"] = str(exc)[:200]
        return summary


# ---------------------------------------------------------------------------
# Request-time read path
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Live projection overlay
#
# The mirror of the live injury-status overlay the request paths apply: the
# batch freezes proj_pts at build time, but Sleeper revises its lines
# intraday — most sharply when a player is ruled out and his volume moves to
# his teammates (2026-11-07: Skattebo Out moved Singletary 8.7 -> 13.9 while
# the site kept scoring the frozen 8.7). One Sleeper fetch per (season, week)
# is cached in-process; every bundle load then swaps each player's proj_pts
# for the live value under the bundle variant's scoring — the same
# weekly_projection_points selection the batch build itself uses, so the
# number is identical in kind, just fresher. Fully fail-soft: any fetch or
# scoring failure keeps the batch values, and a player with no live line
# (absent from the feed, or a present-but-zero line that is indistinguishable
# from a partial feed) keeps his batch value too — exclusion for ruled-out
# players stays with the live STATUS overlay, never an invented zero here.
# ---------------------------------------------------------------------------
_LIVE_PROJ_TTL_SEC = 12 * 60        # fresh map: at most one fetch per window
_LIVE_PROJ_EMPTY_TTL_SEC = 60       # failed/empty fetch: retry again soon
_LIVE_PROJ_CACHE: Dict[tuple, tuple] = {}   # (season, week) -> (fetched_at, map)
_LIVE_PROJ_LOCK = threading.Lock()


def _fetch_live_projection_map(season: int, week: int) -> dict:
    """Sleeper's live weekly projections, {pid: multi-variant entry}.

    The codebase's own Sleeper fetch (utils.projections). Never raises:
    {} on any failure so callers fall back to the batch projections.
    """
    try:
        from utils.projections import fetch_week_projections
        return fetch_week_projections(int(season), int(week)) or {}
    except Exception:
        logger.debug("live projection fetch failed", exc_info=True)
        return {}


def live_projection_map(season: int, week: int) -> dict:
    """TTL-cached ``_fetch_live_projection_map`` (12 min; 1 min when empty)."""
    key = (int(season), int(week))
    now = time.monotonic()
    with _LIVE_PROJ_LOCK:
        hit = _LIVE_PROJ_CACHE.get(key)
        if hit is not None:
            fetched_at, cached = hit
            ttl = _LIVE_PROJ_TTL_SEC if cached else _LIVE_PROJ_EMPTY_TTL_SEC
            if now - fetched_at < ttl:
                return cached
    fresh = _fetch_live_projection_map(int(season), int(week))
    with _LIVE_PROJ_LOCK:
        _LIVE_PROJ_CACHE[key] = (time.monotonic(), fresh)
    return fresh


def _apply_live_projection_overlay(bundles: Dict[str, dict], season: int,
                                   week: int, variant: str) -> Dict[str, dict]:
    """Swap each bundle's proj_pts for the live Sleeper value. Never raises.

    Returns the input untouched when the variant is unknown, the live map is
    empty, or anything fails; otherwise returns a dict whose swapped bundles
    are copies (batch dicts are never mutated in place).
    """
    try:
        if not bundles or variant not in VARIANT_SCORING:
            return bundles
        live_map = live_projection_map(season, week)
        if not live_map:
            return bundles
        from utils.fantasy_scoring import weekly_projection_points
        from utils.league_scoring import stamp_scoring_aliases
        scoring = stamp_scoring_aliases(VARIANT_SCORING[variant])
        out: Dict[str, dict] = {}
        for pid, bundle in bundles.items():
            if not isinstance(bundle, dict):
                out[pid] = bundle
                continue
            pts = None
            try:
                pts = weekly_projection_points(
                    live_map, pid, scoring,
                    str(bundle.get("pos") or "").upper())
            except Exception:
                pts = None
            if pts is not None and float(pts) > 0:
                nb = dict(bundle)
                nb["proj_pts"] = round(float(pts), 1)
                out[pid] = nb
            else:
                out[pid] = bundle
        return out
    except Exception:
        logger.debug("live projection overlay failed", exc_info=True)
        return bundles


def load_start_score_bundles(season: int, week: int, variant: str,
                             player_ids: list) -> Dict[str, dict]:
    """{player_id: bundle} for one (season, week, variant). Never raises.

    Returns {} when the table is missing, the batch hasn't run yet, or the DB
    is unreachable — callers fall back to live per-request computation.
    proj_pts is overlaid with the live Sleeper projection before returning
    (see ``_apply_live_projection_overlay``); every other field is the
    batch's, and injury status still comes from the callers' live overlay.
    """
    if not player_ids or not variant:
        return {}
    try:
        from dashboard_services.db import get_conn
        pids = [str(p) for p in player_ids]
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT player_id, bundle FROM start_score_bundle
                WHERE season = %s AND week = %s AND variant = %s
                  AND player_id = ANY(%s)
                """,
                (int(season), int(week), str(variant), pids),
            ).fetchall()
        out: Dict[str, dict] = {}
        for r in rows or []:
            pid = str(r["player_id"])
            bundle = r["bundle"]
            if isinstance(bundle, str):
                try:
                    bundle = json.loads(bundle)
                except Exception:
                    continue
            if isinstance(bundle, dict):
                out[pid] = bundle
        return _apply_live_projection_overlay(out, season, week, variant)
    except Exception:
        logger.debug("load_start_score_bundles failed", exc_info=True)
        return {}


def load_start_score_bundles_for_scoring(
        season: int, week: int, scoring_settings: Optional[dict],
        player_ids: list) -> Dict[str, dict]:
    """{player_id: bundle} exact under ``scoring_settings``. Never raises.

    When the scoring is exactly one canonical variant these are that
    variant's bundles untouched. For any other scoring (milestone
    bonuses, 2pt conversions, fractional rates, ...) the nearest
    variant's bundles supply the league-independent inputs (Vegas,
    weather, O-line, pace, role confidence, bye, batch injury status)
    while the scoring-dependent fields — proj_pts, recent/season PPG,
    usage, played games, bust rate — are re-derived under the league's
    own settings with the same derivations the Start/Sit page's live
    path and the batch build share (``weekly_projection_points``,
    ``derive_start_sit_inputs``, blended consistency over weekly points
    scored with the league's settings, including the prior-season
    crossfade). Returns {} when no bundles exist at all, so callers keep
    their no-bundle fallback instead of inventing numbers.

    This is what lets league-wide start/sit ranking (the pool behind
    ``start_score_league_index``) rank on the full start score for
    every league: previously a non-canonical league's pool silently
    fell back to raw projections, so its printed rank/index ignored
    the factor-adjusted score its own modal displays.
    """
    if not player_ids:
        return {}
    try:
        variant = variant_for_exact_scoring(scoring_settings)
        if variant:
            return load_start_score_bundles(season, week, variant, player_ids)
        from utils.proj_variant import pick_proj_variant

        base_variant = pick_proj_variant(scoring_settings or {})
        if not base_variant:
            return {}
        base = load_start_score_bundles(season, week, base_variant, player_ids)
        if not base:
            return {}
        from utils.consistency import blended_consistency_profile
        from utils.fantasy_scoring import weekly_projection_points
        from utils.utils import load_week_projection

        scoring = scoring_settings or {}
        season = int(season)
        try:
            week_proj_map = load_week_projection(season, int(week)) or {}
        except Exception:
            week_proj_map = {}
        derived = derive_start_sit_inputs(season, scoring)
        _, weekly = _variant_week_points(season, scoring)
        # Prior-season crossfade for consistency, mirroring the batch
        # build: first of season-1 / season-2 with a real sample.
        prior_weekly: Dict[str, list] = {}
        try:
            if weekly:
                for py in (season - 1, season - 2):
                    _, pweekly = _variant_week_points(py, scoring)
                    if pweekly and max(
                            (len(v) for v in pweekly.values()), default=0) >= 3:
                        prior_weekly = pweekly
                        break
        except Exception:
            prior_weekly = {}
        out: Dict[str, dict] = {}
        for pid, bundle in base.items():
            if not isinstance(bundle, dict):
                continue
            pid = str(pid)
            nb = dict(bundle)
            pos = str(nb.get("pos") or "").upper()
            proj_pts = 0.0
            try:
                _wk_pts = weekly_projection_points(
                    week_proj_map, pid, scoring, pos)
                if _wk_pts is not None:
                    proj_pts = float(_wk_pts)
            except Exception:
                proj_pts = 0.0
            nb["proj_pts"] = round(proj_pts, 1)
            _inp = derived.get(pid) or {}
            nb["recent_ppg"] = round(float(_inp.get("recent_ppg") or 0.0), 1)
            nb["last4_ppg"] = round(float(_inp.get("last4_ppg") or 0.0), 1)
            # Lead-filtered form rides the re-derivation too, so a
            # non-canonical league's pool applies the same handcuff
            # rule as the canonical bundles (see utils.start_sit).
            nb["last4_ppg_lead"] = (
                round(float(_inp["last4_ppg_lead"]), 1)
                if _inp.get("last4_ppg_lead") is not None else None)
            # Projection-cap evidence rides the re-derivation too, so
            # non-canonical variants score the same cap as the
            # canonical bundles (utils.start_sit.cap_projection).
            nb["exp_last4_ppg"] = (
                round(float(_inp["exp_last4_ppg"]), 2)
                if _inp.get("exp_last4_ppg") is not None else None)
            # First-game-back absence rides the re-derivation too,
            # so non-canonical variants shade a returner's projection
            # the same way (utils.start_sit
            # .return_adjusted_projection).
            nb["absence_weeks"] = (
                int(_inp["absence_weeks"])
                if _inp.get("absence_weeks") is not None else None)
            nb["lead_pid"] = _inp.get("lead_pid") or bundle.get("lead_pid")
            nb["season_ppg"] = round(float(_inp.get("season_ppg") or 0.0), 1)
            nb["usage_delta"] = (
                round(float(_inp["usage_delta"]), 1)
                if _inp.get("usage_delta") is not None else None)
            nb["usage_season_avg"] = (
                round(float(_inp["usage_season_avg"]), 1)
                if _inp.get("usage_season_avg") is not None else None)
            nb["played_games"] = int(_inp.get("played_games") or 0)
            nb["snap_trend"] = (
                round(float(_inp["snap_trend"]), 4)
                if _inp.get("snap_trend") is not None else None)
            bust_rate = None
            try:
                cur = weekly.get(pid) or []
                prior = prior_weekly.get(pid) or []
                if cur or prior:
                    cons = blended_consistency_profile(cur, prior, pos)
                    if cons and not cons.get("small_sample"):
                        bust_rate = cons.get("bust_rate")
            except Exception:
                bust_rate = None
            nb["bust_rate"] = bust_rate
            out[pid] = nb
        return out
    except Exception:
        logger.debug("load_start_score_bundles_for_scoring failed",
                     exc_info=True)
        return {}


def start_score_bundles_ready(season: int, week: int, variant: str,
                              player_ids: list) -> bool:
    """True when every requested player has a bundle for (season, week, variant)."""
    if not player_ids:
        return False
    try:
        from dashboard_services.db import get_conn
        pids = [str(p) for p in player_ids]
        with get_conn() as conn:
            n = conn.execute(
                """
                SELECT COUNT(*) AS n FROM start_score_bundle
                WHERE season = %s AND week = %s AND variant = %s
                  AND player_id = ANY(%s)
                """,
                (int(season), int(week), str(variant), pids),
            ).fetchone()
        count = int((n or {}).get("n") or 0)
        return count >= len(set(pids))
    except Exception:
        logger.debug("start_score_bundles_ready check failed", exc_info=True)
        return False
