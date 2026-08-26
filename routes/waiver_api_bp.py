"""Waiver wire API endpoints.

Extracted from app.py to shrink the monolith. League-context helpers still live
in app.py and are reached via the lazy shims below so importing this blueprint
at start-up stays free of a circular import.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime

from flask import Blueprint, jsonify, request

from dashboard_services.api import get_nfl_state
from dashboard_services.service import age_from_bday
from utils.lineup_slots import starter_need_counts as _starter_need_counts
from utils.validation import safe_int as _safe_int
from utils.value_helpers import apply_te_premium, te_premium_from_settings
from utils.waiver_score import (
    WAIVER_PRIME_MAX as _WAIVER_PRIME_MAX,
    WEIGHTS,
    adaptive_trend_thresholds as _adaptive_trend_thresholds,
    build_depth_index as _build_depth_index,
    depth_analysis_for_player as _depth_analysis_for_player,
    need_multiplier as _need_multiplier,
    positional_need_scores as _positional_need_scores,
    replacement_levels as _replacement_levels,
    scarcity_multiplier as _scarcity_multiplier,
    strip_bye_weeks as _strip_bye_weeks,
    waiver_pickup_score as _waiver_pickup_score,
    waiver_signal as _waiver_signal,
    weeks_out_from_projections as _weeks_out_from_projections,
)

logger = logging.getLogger(__name__)

waiver_api_bp = Blueprint("waiver_api", __name__)


# ── Lazy shims to app.py internals (resolved at request time) ─────────────────

def _api_err(*args, **kwargs):
    from app import _api_err as _fn
    return _fn(*args, **kwargs)


def get_league_ctx_from_cache(*args, **kwargs):
    from app import get_league_ctx_from_cache as _fn
    return _fn(*args, **kwargs)


def get_model_value_table_cached(*args, **kwargs):
    from app import get_model_value_table_cached as _fn
    return _fn(*args, **kwargs)


def _waiver_value_keys(*args, **kwargs):
    from app import _waiver_value_keys as _fn
    return _fn(*args, **kwargs)


def get_players_global(*args, **kwargs):
    from app import get_players_global as _fn
    return _fn(*args, **kwargs)


def get_players_index_global(*args, **kwargs):
    from app import get_players_index_global as _fn
    return _fn(*args, **kwargs)


def get_viewer_session_for_league(*args, **kwargs):
    from app import get_viewer_session_for_league as _fn
    return _fn(*args, **kwargs)


def _matchup_rank_table(*args, **kwargs):
    from app import _matchup_rank_table as _fn
    return _fn(*args, **kwargs)


@waiver_api_bp.route("/api/waiver-candidates")
def api_waiver_candidates():
    """
    Returns scored waiver wire candidates for a league.
    Query params: platform, league_id, season, position (optional filter)
    """
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    season = int(request.args.get("season") or datetime.now().year)
    position_filter = (request.args.get("position") or "").strip().upper()

    if not league_id:
        return jsonify({"error": "league_id required"}), 400

    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
    except Exception as e:
        return _api_err("Request failed", e)

    rosters = ctx.get("rosters") or []
    rostered_ids = {
        str(pid)
        for r in rosters
        for pid in (r.get("players") or [])
    }

    # Detect whether the rookie draft has happened by checking if any
    # current-year rookie Sleeper ID is already on a roster
    _rookie_sids_wv: set[str] = set()
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class as _grc_wv
        from dashboard_services.db import get_conn as _gc_wv
        _ry_wv = _grc_wv()
        with _gc_wv() as _cc_wv:
            _rr_wv = _cc_wv.execute(
                "SELECT sleeper_id FROM rookie_prospects WHERE draft_class_year = %s AND sleeper_id IS NOT NULL",
                (_ry_wv,),
            ).fetchall()
        _rookie_sids_wv = {str(r["sleeper_id"]) for r in _rr_wv if r["sleeper_id"]}
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
    _rookie_draft_done_wv = bool(_rookie_sids_wv and any(sid in rostered_ids for sid in _rookie_sids_wv))

    players_index = ctx.get("players_index") or {}
    model_value_table = list(get_model_value_table_cached() or [])

    _rp_wv = ctx.get("roster_positions") or []
    # Pick the value column that matches this league's format (redraft vs
    # dynasty, 1QB vs Superflex) — shared with the offseason/Season-Hub card so
    # both waiver surfaces rank and display off identical values.
    _vf_wv, _vfb_wv = _waiver_value_keys(ctx)
    # Auto-apply the league's TE premium, exactly like the value column: a no-op
    # for non-TE-premium leagues / non-TEs.
    _tep_wv = te_premium_from_settings(ctx.get("scoring_settings"))

    candidates = []
    for row in model_value_table:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("id") or "")
        pos = str(row.get("position") or row.get("pos") or "").upper()
        team = str(row.get("team") or players_index.get(pid, {}).get("team") or "").strip().upper()
        if not pid or pid in rostered_ids:
            continue
        if pid in _rookie_sids_wv and not _rookie_draft_done_wv:
            continue
        if team in ("", "FA", "FREE AGENT"):
            continue
        if pos not in {"QB", "RB", "WR", "TE"}:
            continue
        try:
            val = float(row.get(_vf_wv) or row.get(_vfb_wv) or 0.0)
        except Exception:
            val = 0.0
        val = apply_te_premium(val, pos, _tep_wv)
        # Floor out near-zero-value noise (see _build_waiver_targets_rows): a
        # negligible-value free agent only surfaces on a trend/age bonus.
        if val < WEIGHTS.min_value:
            continue
        pmeta_wv = players_index.get(pid, {})
        precise_age_wv = age_from_bday(pmeta_wv.get("bDay"))
        try:
            age = precise_age_wv if precise_age_wv is not None else float(row.get("age") or 0)
        except Exception:
            age = 0.0
        player_name = (
                row.get("name")
                or pmeta_wv.get("name")
                or f"Player {pid}"
        )
        candidates.append({
            "player_id": pid,
            "name": player_name,
            "position": pos,
            "team": row.get("team") or players_index.get(pid, {}).get("team") or "",
            "value": val,
            "age": age,
            "pos_rank_label": row.get("pos_rank_label") or "",
            "rank_change_7d": row.get("rank_change_7d"),
        })

    # Breakout scores that align with the Breakout Engine page (same season
    # resolution + eligibility gate), so the "Breakout" waiver signal never tags
    # a player the engine wouldn't call a breakout.
    waiver_breakout: dict = {}
    try:
        _db_url = os.getenv("DATABASE_URL", "").strip()
        if _db_url and not any(t in _db_url for t in ("USER", "PASSWORD", "HOST")):
            from dashboard_services.breakout_api import aligned_breakout_scores as _abs
            waiver_breakout = _abs(
                [c["player_id"] for c in candidates[:100]],
                int(season) if season else None,
            )
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    # Weekly usage trends: recent role growth is the strongest waiver signal.
    # Usage trends are a live in-season signal: last-3-week average vs season
    # average. In the offseason the only data is last season's final weeks, which
    # reads as if it were current activity ("Usage Spike / +6 touches" in July),
    # so we hide usage entirely until real games return. Value/breakout signals
    # ("Breakout", "Trending Up" from value-rank movement) still show.
    usage_trends: dict = {}
    try:
        from data_building.weekly_metrics import get_usage_trends
        _nfl_wv = get_nfl_state() or {}
        if str(_nfl_wv.get("season_type") or "").lower() != "off":
            usage_trends = get_usage_trends(int(_nfl_wv.get("season") or season))
    except Exception:
        usage_trends = {}

    # Depth-chart injury vacancies: an injured player ahead of a candidate on
    # the same team+position frees up the role directly. The reduced players_index
    # cache lacks depth_chart_order / injury_status, so pull the full Sleeper
    # players feed (get_players_global) and index it once per request.
    _depth_idx_wv: dict = {}
    _full_players_wv: dict = {}
    try:
        _full_players_wv = get_players_global() or {}
        _depth_idx_wv = _build_depth_index(_full_players_wv)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    # Recent PPR points-per-game as a rest-of-season production proxy (#5), also
    # reused to size injury vacancies by the vacated role's production (#7).
    _ppg_by_pid_wv: dict = {}
    try:
        from utils.utils import load_usage_table
        for _u in (load_usage_table() or []):
            if isinstance(_u, dict):
                _upid = str(_u.get("player_id") or _u.get("id") or "")
                if _upid:
                    _ppg_by_pid_wv[_upid] = _u.get("ppr_ppg")
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    # Season projected PPG — the *healthy* production of a role, used to value an
    # injury vacancy (the injured player projects ~0 while hurt, so their own
    # recent ppg understates the role).
    _season_ppg_wv: dict = {}
    try:
        from data_building.fetch_projections import fetch_sleeper_season_projections
        _nfl_pj = get_nfl_state() or {}
        _pj_season = int(_nfl_pj.get("season") or season)
        for _pid, _row in (fetch_sleeper_season_projections(_pj_season, "ppr") or {}).items():
            if isinstance(_row, dict) and _row.get("ppg") is not None:
                _season_ppg_wv[str(_pid)] = _row.get("ppg")
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    # Upcoming weekly projections — a player projected for ~0 points across the
    # next N weeks IS the projection provider's read on how long they're out, so
    # count the leading zero-run to get the injury timeline directly (#: weeks
    # out from projections rather than guessing from the injury label).
    _future_week_projs_wv: list = []
    _future_week_teams_wv: list = []  # parallel: set of teams playing each week
    try:
        from utils.utils import load_week_projection, load_week_schedule
        _nfl_wk = get_nfl_state() or {}
        _proj_season = int(_nfl_wk.get("season") or season)
        _cur_week = int(_nfl_wk.get("week") or _nfl_wk.get("display_week") or 1)
        _horizon = int(round(float(WEIGHTS.injury_horizon_weeks))) + 2
        for _w in range(_cur_week, _cur_week + _horizon):
            _wp = load_week_projection(_proj_season, _w) or {}
            _future_week_projs_wv.append(_wp if isinstance(_wp, dict) else {})
            _teams = set()
            for _g in (load_week_schedule(_proj_season, _w) or []):
                if isinstance(_g, dict):
                    for _side in ("away", "home"):
                        _t = str(_g.get(_side) or "").upper()
                        if _t:
                            _teams.add(_t)
            _future_week_teams_wv.append(_teams)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    def _week_pts_wv(week_map, pid):
        """Numeric weekly projection for pid, scored for the league (Sleeper's
        published total for plain leagues, recompute for custom); None when the
        player is absent from the map."""
        from utils.fantasy_scoring import weekly_projection_points
        _pos = str(
            (_full_players_wv.get(str(pid)) or players_index.get(str(pid)) or {}).get("pos") or ""
        ).upper()
        return weekly_projection_points(
            week_map, pid, ctx.get("raw_scoring_settings"), _pos
        )

    def _weeks_out_wv(pid):
        """Projection-derived weeks-out for an injured player, or None if the
        weekly projections aren't available. Bye weeks (team not scheduled) are
        stripped first so a bye isn't mistaken for a missed game."""
        if not _future_week_projs_wv:
            return None
        _pm = _full_players_wv.get(str(pid)) or players_index.get(str(pid)) or {}
        _team = str(_pm.get("team") or "").upper()
        series = [_week_pts_wv(_wm, pid) for _wm in _future_week_projs_wv]
        plays = [
            (_team in _tset) if (_tset and _team) else True
            for _tset in _future_week_teams_wv
        ]
        series = _strip_bye_weeks(series, plays)
        if all(x is None for x in series):
            return None
        return _weeks_out_from_projections(series)

    def _forward_ppg_wv(pid):
        """Mean of a player's own upcoming (non-bye) weekly projections — a
        forward-looking production estimate (#1), better than backward ppg."""
        if not _future_week_projs_wv:
            return None
        _pm = _full_players_wv.get(str(pid)) or players_index.get(str(pid)) or {}
        _team = str(_pm.get("team") or "").upper()
        vals = []
        for _i, _wm in enumerate(_future_week_projs_wv):
            _tset = _future_week_teams_wv[_i] if _i < len(_future_week_teams_wv) else set()
            if _tset and _team and _team not in _tset:
                continue  # bye
            _p = _week_pts_wv(_wm, pid)
            if _p is not None:
                vals.append(max(0.0, _p))
        return (sum(vals) / len(vals)) if vals else None

    # Upcoming schedule ease per position (#3): a soft slate nudges a candidate up.
    _matchup_by_pos_wv: dict = {}
    try:
        for _mp in ("QB", "RB", "WR", "TE"):
            _rmap, _mtot, _minfo, _mz = _matchup_rank_table(season, _mp)
            _matchup_by_pos_wv[_mp] = (_rmap or {}, int(_mtot or 0))
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    # Positional scarcity / value-over-replacement (#4): value above the position's
    # replacement level is worth more where the drop-off is steeper.
    _repl_by_pos_wv: dict = {}
    try:
        _vals_by_pos = {}
        for _row in model_value_table:
            if isinstance(_row, dict):
                _rpos = str(_row.get("position") or _row.get("pos") or "").upper()
                if _rpos in ("QB", "RB", "WR", "TE"):
                    try:
                        _vals_by_pos.setdefault(_rpos, []).append(
                            float(_row.get(_vf_wv) or _row.get(_vfb_wv) or 0.0)
                        )
                    except (TypeError, ValueError):
                        pass
        _teams_n = len(rosters) or 12
        _repl_cutoffs = {"QB": _teams_n * 2, "RB": _teams_n * 3,
                         "WR": _teams_n * 3, "TE": _teams_n * 1}
        _repl_by_pos_wv = _replacement_levels(_vals_by_pos, _repl_cutoffs)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    # Viewer roster need (#4): a position the viewer is short on is worth more.
    _need_scores_wv: dict = {}
    try:
        _starter_reqs = _starter_need_counts(_rp_wv, extra_depth=1)
        _vsess = get_viewer_session_for_league(ctx.get("users") or [], rosters)
        _vrid = str((_vsess or {}).get("viewer_roster_id") or "")
        if _vrid:
            _vros = next((r for r in rosters if str(r.get("roster_id")) == _vrid), None)
            if _vros:
                _counts = {}
                for _pid in (_vros.get("players") or []):
                    _pp = _full_players_wv.get(str(_pid)) or players_index.get(str(_pid)) or {}
                    _ppos = str(_pp.get("position") or _pp.get("pos") or "").upper()
                    if _ppos in ("QB", "RB", "WR", "TE"):
                        _counts[_ppos] = _counts.get(_ppos, 0) + 1
                _need_scores_wv = _positional_need_scores(_counts, _starter_reqs)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    _now_ms_wv = time.time() * 1000.0

    def _freshness_wv(injured_pids):
        # Proxy for injury recency via Sleeper's last_modified (ms): a role that
        # went stale weeks ago is largely already absorbed by whoever replaced it.
        lms = [
            _full_players_wv.get(str(p), {}).get("last_modified")
            for p in (injured_pids or [])
        ]
        lms = [x for x in lms if isinstance(x, (int, float)) and x > 0]
        if not lms:
            return 1.0
        days_old = (_now_ms_wv - max(lms)) / 86400000.0
        return max(0.6, min(1.0, 1.0 - max(0.0, days_old - 10.0) / 60.0))

    # Attach every signal the shared scorer/labeler in utils.waiver_score reads,
    # so ranking + badge reflect: usage spikes, depth-chart injury vacancies
    # (proximity/volume/freshness weighted), the candidate's own health, roster
    # need, and rest-of-season production — not just static dynasty value.
    for c in candidates:
        # Safe defaults so a failed signal-join for one candidate can't 500 the
        # whole response (or the sort / result-building below that read these).
        c.setdefault("usage_delta", None)
        c.setdefault("usage_stat", None)
        c["injured_ahead"] = c.get("injured_ahead") or []
        c.setdefault("healthy_ahead", None)
        c["vacated"] = c.get("vacated") or []
        c.setdefault("injury_freshness", None)
        c.setdefault("self_status", "")
        c.setdefault("own_proj_ppg", None)
        c.setdefault("ros_ppg", None)
        c.setdefault("need_mult", 1.0)
        c.setdefault("schedule_ease_rank", None)
        c.setdefault("schedule_total", 0)
        c.setdefault("scarcity_mult", 1.0)
        c.setdefault("handcuff_upside", 0.0)
        try:
            ut = usage_trends.get(c["player_id"]) or {}
            c["usage_delta"] = ut.get("delta")
            c["usage_stat"] = ut.get("stat")

            _da = _depth_analysis_for_player(c["player_id"], _full_players_wv, _depth_idx_wv)
            _inj_pids = _da.get("injured_pids_ahead") or []
            c["injured_ahead"] = _da.get("injured_ahead") or []
            c["healthy_ahead"] = _da.get("healthy_ahead") or 0

            # Handcuff upside (#8): the immediate backup to a healthy, high-usage
            # starter is a valuable stash — if that starter goes down the role, and
            # the fantasy points, transfer wholesale. Only the direct #2 (exactly
            # one healthy body ahead) to an elite lead back earns it, scaled by that
            # starter's ROS production. RB-only: no other position concentrates a
            # role into a single handcuff the way a bell-cow backfield does.
            _hc = 0.0
            if c["position"] == "RB" and c["healthy_ahead"] == 1:
                _sp = (_da.get("healthy_pids_ahead") or [None])[0]
                if _sp is not None:
                    _sppg = _forward_ppg_wv(_sp) or _ppg_by_pid_wv.get(_sp) or 0.0
                    try:
                        _sppg = float(_sppg or 0.0)
                    except (TypeError, ValueError):
                        _sppg = 0.0
                    # ~14 ppg lead back → starts earning; ~19+ ppg bell-cow → full.
                    _hc = max(0.0, min(1.0, (_sppg - 14.0) / 5.0))
            c["handcuff_upside"] = _hc
            # Value each vacancy from the injured player's role production (healthy
            # season projected ppg, else recent ppg) and, crucially, from how long
            # they're projected to be out (leading zero-run in the weekly
            # projections) rather than a fixed guess by injury label.
            _vac = _da.get("vacated") or []
            for _v in _vac:
                _vpid = str(_v.get("pid"))
                _v["proj_ppg"] = _season_ppg_wv.get(_vpid) or _ppg_by_pid_wv.get(_vpid)
                try:
                    from dashboard_services.injury_return import weeks_out_for_player as _espn_wo
                    _espn_weeks = _espn_wo(_vpid)
                except Exception:
                    _espn_weeks = None
                if _espn_weeks is not None:
                    _v["weeks_out"] = _espn_weeks
                    _v["return_source"] = "espn"
                else:
                    _v["weeks_out"] = _weeks_out_wv(_vpid)
                    _v["return_source"] = "status"
            c["vacated"] = _vac
            c["injury_freshness"] = _freshness_wv(_inj_pids)

            _self = _full_players_wv.get(c["player_id"]) or {}
            c["self_status"] = _self.get("injury_status") or _self.get("status") or ""

            # Forward projected ppg for this candidate (#1) — used for production
            # and, via the transfer guard (#2), to fade injury upside taken.
            _fwd_ppg = _forward_ppg_wv(c["player_id"])
            c["own_proj_ppg"] = _fwd_ppg
            c["ros_ppg"] = _fwd_ppg if _fwd_ppg is not None else _ppg_by_pid_wv.get(c["player_id"])

            c["need_mult"] = _need_multiplier(c["position"], _need_scores_wv)

            # Upcoming schedule ease (#3).
            _team_up = str(_self.get("team") or players_index.get(c["player_id"], {}).get("team")
                           or c.get("team") or "").upper()
            _rmap_s, _tot_s = _matchup_by_pos_wv.get(c["position"], ({}, 0))
            c["schedule_ease_rank"] = _rmap_s.get(_team_up)
            c["schedule_total"] = _tot_s

            # Positional scarcity (#4).
            c["scarcity_mult"] = _scarcity_multiplier(c["position"], c["value"], _repl_by_pos_wv)
        except Exception:
            logger.exception("[waiver-candidates] signal join failed for %s", c.get("player_id"))
            continue

    def _safe_pickup_score(c):
        try:
            return _waiver_pickup_score(c, waiver_breakout, _WAIVER_PRIME_MAX)
        except Exception:
            logger.exception("[waiver-candidates] pickup score failed for %s", c.get("player_id"))
            return 0.0

    candidates.sort(key=_safe_pickup_score, reverse=True)
    if position_filter and position_filter in {"QB", "RB", "WR", "TE"}:
        candidates = [c for c in candidates if c["position"] == position_filter]

    result = []
    _shown = candidates[:30]
    # One bulk local read. SportsGameOdds ingestion is scheduled separately and
    # never runs in this request path.
    try:
        from dashboard_services.market_intelligence.repository import attach_weekly_signals as _attach_mi_wv
        from dashboard_services.market_intelligence.signals import market_opportunity as _mi_opp
        _mi_state_wv = get_nfl_state() or {}
        _mi_week_wv = int(_mi_state_wv.get("week") or _mi_state_wv.get("display_week") or 1)
        _attach_mi_wv(_shown, int(_mi_state_wv.get("season") or season), _mi_week_wv,
                      site_key="ros_ppg", scoring_settings=ctx.get("raw_scoring_settings") or {})
    except Exception:
        logger.debug("market intelligence unavailable for waivers", exc_info=True)
    from utils.model_confidence import confidence_from_inputs
    # Badge trend relative to the shown set so the (trend-sorted) list doesn't
    # read "Rising Fast" on every row.
    _fast_thr, _up_thr = _adaptive_trend_thresholds([c.get("rank_change_7d") for c in _shown])

    # Suggested FAAB bid band (% of budget). Only the top targets warrant real
    # money; a player who fills the viewer's own roster need is nudged up. A band
    # rather than a number because league budgets differ ($100 / $1000 / rolling);
    # the % reads the same regardless. Gated client-side on the league using FAAB.
    # FAAB detection. Sleeper's waiver_type == 2 is NOT sufficient on its own —
    # rolling / waiver-priority leagues report the same code, so keying on it alone
    # showed a bid % to non-FAAB leagues. A genuine FAAB league always carries a
    # positive waiver_budget, so require both. ESPN uses an explicit acquisition
    # budget. Fail closed when FAAB can't be confirmed so a non-FAAB league never
    # sees a bid % it can't use.
    _wv_settings = (ctx.get("league") or {}).get("settings") or {}
    _faab_enabled = (
        # Sleeper FAAB: waiver_type flagged AND a real budget to spend.
            (_safe_int(_wv_settings.get("waiver_type"), -1) == 2
             and _safe_int(_wv_settings.get("waiver_budget"), 0) > 0)
            or _safe_int(_wv_settings.get("acquisition_budget"), 0) > 0  # ESPN FAAB
    )
    _wv_scores = [_safe_pickup_score(c) for c in _shown]
    _wv_smin = min(_wv_scores) if _wv_scores else 0.0
    _wv_srng = ((max(_wv_scores) - _wv_smin) if _wv_scores else 1.0) or 1.0

    def _faab_band(_c):
        # Waiver-wire targets, not premium trade pieces — keep bids modest so the
        # single best add tops out around the low-20s% and typical adds sit in the
        # low single digits, matching how much of a budget these fliers are worth.
        _t = (_safe_pickup_score(_c) - _wv_smin) / _wv_srng  # 0..1 within shown set
        _center = 1.0 + (_t ** 1.7) * 16.0  # top target ~17%, tapers fast
        _center *= 1.0 + min(max((_c.get("need_mult") or 1.0) - 1.0, 0.0), 0.25)  # fills your need → bid up
        # Elite-handcuff premium: the direct backup to a healthy stud starter is
        # worth a real speculative bid beyond his standalone score — a top bell-cow
        # handcuff lands ~10-18% even with a low pickup score of his own.
        _center += float(_c.get("handcuff_upside") or 0.0) * 13.0
        return max(0, int(round(_center * 0.72))), min(50, int(round(_center * 1.12)) + 1)

    # ── Add/drop pairing: for each target, the best player on the viewer's own
    # roster to cut to make room. Prefer thinning a position where the viewer is
    # deep (esp. the add's own position); only ever suggest a drop that's a value
    # downgrade from the add, so the pairing is always a genuine upgrade. Needs
    # the viewer's roster, passed as ?rid= (the shared page cache means the client
    # supplies it, matching the window._viewerRid personalization pattern).
    _rid = (request.args.get("rid") or "").strip()
    _mvt_by_id = {str(r.get("id")): r for r in model_value_table
                  if isinstance(r, dict) and r.get("id")}
    _KEEP = {"QB": 2, "RB": 5, "WR": 6, "TE": 2}  # keep-depth before a spot is "spare"

    def _roster_val(pid: str) -> float:
        row = _mvt_by_id.get(pid) or {}
        try:
            return apply_te_premium(
                float(row.get(_vf_wv) or row.get("value") or 0.0),
                str(row.get("position") or "").upper(), _tep_wv)
        except Exception:
            return 0.0

    _drop_pool: list = []
    _pos_counts: dict = {}
    _viewer_roster = next((r for r in rosters if str(r.get("roster_id")) == _rid), None) if _rid else None
    if _viewer_roster:
        for pid in (_viewer_roster.get("players") or []):
            pid = str(pid)
            row = _mvt_by_id.get(pid) or {}
            meta = players_index.get(pid, {})
            pos = str(row.get("position") or meta.get("pos") or "").upper()
            name = row.get("name") or meta.get("name") or f"Player {pid}"
            _pos_counts[pos] = _pos_counts.get(pos, 0) + 1
            # Never suggest cutting a current-class rookie — those are usually
            # deliberate stashes (dynasty picks / upside bench holds), not spare
            # parts. Keep them in the position counts (they do take a roster spot)
            # but out of the droppable pool.
            if pid in _rookie_sids_wv:
                continue
            _drop_pool.append({"player_id": pid, "name": name, "position": pos,
                               "value": _roster_val(pid)})
        _drop_pool.sort(key=lambda d: d["value"])  # weakest first

    def _drop_for(_c):
        if not _drop_pool:
            return None
        add_val = _c.get("value") or 0.0
        add_pos = _c.get("position") or ""
        elig = [d for d in _drop_pool if d["value"] < add_val]
        if not elig:
            return None  # everyone you'd cut is worth more than the add — hold
        same_pos = [d for d in elig
                    if d["position"] == add_pos and _pos_counts.get(add_pos, 0) > _KEEP.get(add_pos, 3)]
        deep = [d for d in elig if _pos_counts.get(d["position"], 0) > _KEEP.get(d["position"], 3)]
        pick = (same_pos or deep or elig)[0]
        return {"name": pick["name"], "position": pick["position"], "value": round(pick["value"])}

    _adds_by_id = {}
    try:
        for row in _sleeper_trending_adds(limit=50) or []:
            pid = str(row.get("player_id") or "")
            if pid:
                _adds_by_id[pid] = int(row.get("count") or 0)
    except Exception:
        _adds_by_id = {}

    for c in _shown:
        try:
            sig_cls, sig_label = _waiver_signal(
                c, waiver_breakout, _WAIVER_PRIME_MAX,
                fast_thr=_fast_thr, up_thr=_up_thr,
            )
            bscore = waiver_breakout.get(c["player_id"], 0.0)
            ut = usage_trends.get(c["player_id"]) or {}
            _flo, _fhi = _faab_band(c)
            _confidence_inputs = sum([
                c.get("ros_ppg") is not None,
                c.get("usage_delta") is not None,
                c.get("schedule_ease_rank") is not None,
                c.get("age") is not None,
                bool(c.get("team")),
                c.get("value") is not None,
            ])
            _confidence = confidence_from_inputs(_confidence_inputs, 6)
            _market_signal = c.get("market_signal") or {}
            _market_opp = _mi_opp(c.get("market_projection"), c.get("ros_ppg"),
                                  _market_signal.get("confidence"),
                                  c.get("rostered_pct") or 0) if _market_signal else None
            result.append({
                "player_id": c["player_id"],
                "name": c["name"],
                "position": c["position"],
                "team": c["team"],
                "value": c["value"],
                "age": c["age"],
                "pos_rank_label": c["pos_rank_label"],
                "rank_change_7d": c["rank_change_7d"],
                "breakout_score": bscore,
                "signal": sig_label,
                "signal_class": sig_cls,
                "composite_score": _safe_pickup_score(c),
                "usage_delta": ut.get("delta"),
                "usage_stat": ut.get("stat"),
                "usage_series": ut.get("series"),
                "injured_ahead": len(c.get("injured_ahead") or []),
                "healthy_ahead": c.get("healthy_ahead"),
                "ros_ppg": round(c["ros_ppg"], 1) if c.get("ros_ppg") is not None else None,
                "roster_need": round((c.get("need_mult") or 1.0) - 1.0, 3),
                "scarcity": round((c.get("scarcity_mult") or 1.0) - 1.0, 3),
                "schedule_ease_rank": c.get("schedule_ease_rank"),
                "faab_low": _flo,
                "faab_high": _fhi,
                "drop": _drop_for(c),
                "confidence": _confidence,
                "market_projection": c.get("market_projection"),
                "market_opportunity": _market_opp,
                "rostered_pct": c.get("rostered_pct"),
                "adds_48h": _adds_by_id.get(str(c["player_id"])),
            })
        except Exception:
            logger.exception("[waiver-candidates] result row failed for %s", c.get("player_id"))
            continue

    return jsonify({"candidates": result, "total": len(result), "faab_enabled": _faab_enabled})


_TRENDING_ADDS_CACHE: dict = {}
_TRENDING_ADDS_TTL = 1800  # 30 min; a league-wide signal shared by all viewers


def _sleeper_trending_adds(limit: int = 25, lookback_hours: int = 48) -> list:
    """Sleeper's league-wide most-added players (add count over a lookback
    window). Cached 30 min; this is a global signal, not per-league, so every
    viewer shares one fetch. Best-effort: any failure returns []."""
    key = (limit, lookback_hours)
    hit = _TRENDING_ADDS_CACHE.get(key)
    if hit and (time.time() - hit[0]) < _TRENDING_ADDS_TTL:
        return hit[1]
    try:
        from dashboard_services.api import fetch_json
        data = fetch_json(
            f"/players/nfl/trending/add?lookback_hours={lookback_hours}&limit={limit}"
        )
        result = data if isinstance(data, list) else []
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        result = []
    if result:
        _TRENDING_ADDS_CACHE[key] = (time.time(), result)
    return result


@waiver_api_bp.route("/api/trending-adds")
def api_trending_adds():
    """Trending waiver adds across all Sleeper leagues, filtered to players the
    viewer's league can still add (not already rostered), with our value + pos
    rank joined on. Feeds the 'Trending across leagues' strip on Waivers."""
    platform = (request.args.get("platform") or "sleeper").strip().lower()
    league_id = (request.args.get("league_id") or "").strip()
    season = int(request.args.get("season") or datetime.now().year)
    if not league_id:
        return jsonify({"trending": []})

    # Trending is a Sleeper signal; other platforms just get an empty strip.
    if platform != "sleeper":
        return jsonify({"trending": []})

    trend = _sleeper_trending_adds()
    if not trend:
        return jsonify({"trending": []})

    try:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
    except Exception:
        ctx = {}
    rostered_ids = {
        str(pid)
        for r in (ctx.get("rosters") or [])
        for pid in (r.get("players") or [])
    }
    players_index = ctx.get("players_index") or get_players_index_global() or {}
    _mvt = {str(r.get("id")): r for r in (get_model_value_table_cached() or [])
            if isinstance(r, dict) and r.get("id")}

    out = []
    for row in trend:
        pid = str(row.get("player_id") or "")
        if not pid or pid in rostered_ids:
            continue  # can't add what you already own
        meta = players_index.get(pid) or {}
        val_row = _mvt.get(pid) or {}
        # Skip players we can't resolve in our index anywhere (neither the league
        # players index nor the value table). These render as a nameless
        # "Player 12345" card, which is noise — drop them entirely.
        if not meta and not val_row:
            continue
        pos = str(val_row.get("position") or meta.get("pos") or "").upper()
        if pos == "DEF":
            name = f"{(meta.get('team') or pid)} D/ST"
        else:
            name = val_row.get("name") or meta.get("name") or f"Player {pid}"
        out.append({
            "player_id": pid,
            "name": name,
            "position": pos,
            "team": (val_row.get("team") or meta.get("team") or "").upper(),
            "value": round(float(val_row.get("value") or 0)),
            "pos_rank_label": val_row.get("pos_rank_label") or "",
            "adds": int(row.get("count") or 0),
        })
        if len(out) >= 12:
            break
    return jsonify({"trending": out})
