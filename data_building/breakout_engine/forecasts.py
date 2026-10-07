"""Live forecasts and track-record aggregation for breakout calls.

Two read-only views over the breakout engines' persisted calls:

* **Forecasts** project how an *open* call is tracking before its outcome
  window closes. They are computed live at read time from the same weekly
  rows the grader will eventually use, are NEVER stored, and are NEVER fed
  into any hit rate. A forecast is a band ("Tracking to hit" / "Borderline"
  / "Tracking to miss") plus the numbers driving it, so it can never be
  mistaken for a grade: grades come only from ``weekly_grading`` (weekly
  calls) over a complete outcome window.

* **Track record** aggregates *finished* grades only: weekly hit rates per
  classification for the current scoring version (via the grader's own
  ``summarize_grade_rows``, so the 10-graded floor and rate math match the
  grader exactly), plus calibration band rates by breakout score and by
  confidence over the same pooled calls (via ``calibration``'s shared
  helpers), plus the season engine's grades by phase once that table
  exists. The season grades table is written by the season grading path;
  until it exists (or has rows) the season section reports an explicit
  pending state, never an error and never a fabricated rate.

Weekly forecast banding deliberately reuses the grader's own machinery
(``resolve_baseline`` / ``window_stats`` / ``classify_outcome`` and its
threshold constants) on the partial outcome window, so a forecast can never
disagree with the grader about what the finished numbers will mean: the
band is the interim verdict the grader's rules produce on the games played
so far. Preseason (season engine) forecasts reuse the backtest hit
definition from ``backtest_multitask.get_breakout_pids``: a valid prior
season (>= 6 games, >= 4.0 PPG) sets the target at prior PPG x 1.15 with a
7.0 PPG floor; without a valid prior the target is 10.0 PPG. The forecast
PPG blends banked production with rest-of-season projected PPG, weighted
by games.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dashboard_services.db import get_conn
from data_building.breakout_engine import calibration
from data_building.breakout_engine import weekly_grading as wg

logger = logging.getLogger(__name__)

# Forecast bands (stable keys; labels are the user-facing wording).
BAND_TRACKING_HIT = "tracking_to_hit"
BAND_BORDERLINE = "borderline"
BAND_TRACKING_MISS = "tracking_to_miss"
BAND_LABELS = {
    BAND_TRACKING_HIT: "Tracking to hit",
    BAND_BORDERLINE: "Borderline",
    BAND_TRACKING_MISS: "Tracking to miss",
}

# Season-engine phases that are genuine pre-season calls (the in-season
# phase re-scores weekly and is not a preseason call). Mirrors the phase
# keys in breakout_engine.config.PHASE_WEIGHTS minus "in_season".
PRESEASON_PHASES = ("offseason", "post_free_agency", "post_draft", "preseason")

# Grades table written by the season-engine grading path. Read defensively:
# the table may not exist yet, and its arrival must light this section up
# without any code change here.
SEASON_GRADES_TABLE = "season_breakout_grades"

# Regular-season length used for rest-of-season projection weeks.
FINAL_REGULATION_WEEK = 18

# Preseason banding: a forecast within this fraction of the target (but
# under it) is Borderline rather than Tracking to miss.
BORDERLINE_FRACTION = 0.90

# Backtest hit-target constants (backtest_multitask.get_breakout_pids).
PRIOR_MIN_GAMES = 6
PRIOR_MIN_PPG = 4.0
TARGET_GROWTH = 1.15
TARGET_FLOOR_PPG = 7.0
NO_PRIOR_TARGET_PPG = 10.0

TOP_GRADES_LIMIT = 5
TOP_FORECAST_LIMIT = 3


# =============================================================================
# weekly forecasts (pure)
# =============================================================================

def weekly_forecast(
    call: Dict[str, Any],
    weekly_rows: Sequence[Dict[str, Any]],
    through_week: int,
    prior_rows: Optional[Sequence[Dict[str, Any]]] = None,
    cohort_ppg: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Live forecast for one open weekly call. Pure.

    Returns None when the call's outcome window is already complete (the
    call is the grader's business then, not a forecast's). Otherwise a dict
    whose ``band`` is set only when at least one outcome game has been
    played and a role baseline exists; the zero-game and no-baseline cases
    return explicit states with no band. ``prior_rows`` is the player's
    prior-season series, forwarded to the grader's baseline resolution so
    a prior-season-baseline call forecasts against the same filled PPG
    baseline it will eventually grade against.
    """
    call_week = int(call.get("as_of_week") or 0)
    through = int(through_week)
    if wg.is_mature(call_week, through):
        return None
    window = [call_week + i for i in range(1, wg.OUTCOME_WEEKS + 1)]
    elapsed = [w for w in window if w <= through]
    elapsed_set = set(elapsed)
    rows = list(weekly_rows)
    outcome_rows = [r for r in rows
                    if wg._num(r.get("week")) is not None
                    and int(r["week"]) in elapsed_set]
    outcome = wg.window_stats(outcome_rows)
    base: Dict[str, Any] = {
        "kind": "weekly",
        "call_week": call_week,
        "weeks_in": len(elapsed),
        "outcome_weeks": wg.OUTCOME_WEEKS,
        "games": outcome["games"],
        "band": None,
        "band_label": None,
    }
    if not outcome["games"]:
        base.update({
            "state": "no_games",
            "basis": f"No games yet in the {wg.OUTCOME_WEEKS} weeks after "
                     f"the Week {call_week} call",
        })
        return base
    baseline, _source = wg.resolve_baseline(call, rows, prior_rows,
                                             cohort_ppg)
    verdict = wg.classify_outcome(baseline, outcome,
                                  cohort=(_source == "cohort"))
    base.update({
        "ppg_delta": wg._round(verdict.get("ppg_delta")),
        "opp_delta": wg._round(verdict.get("opp_delta")),
        "snap_delta": wg._round(verdict.get("snap_delta")),
    })
    if verdict.get("grade") == wg.GRADE_UNGRADED:
        # classify_outcome is ungraded here only when no role baseline
        # exists (games > 0 was handled above).
        base.update({
            "state": "no_baseline",
            "basis": "No role baseline to project from yet",
        })
        return base
    band = {
        wg.GRADE_HIT: BAND_TRACKING_HIT,
        wg.GRADE_MISS: BAND_TRACKING_MISS,
    }.get(verdict.get("grade"), BAND_BORDERLINE)
    games = outcome["games"]
    basis = (f"{len(elapsed)} of {wg.OUTCOME_WEEKS} weeks in, "
             f"{games} game{'s' if games != 1 else ''} played")
    if _source == "cohort":
        basis += (f" vs a {baseline['ppr_ppg']:.1f} PPG typical-rookie "
                  f"{call.get('position') or 'player'} baseline")
    base.update({
        "state": "forecast",
        "band": band,
        "band_label": BAND_LABELS[band],
        "basis": basis,
    })
    return base


# =============================================================================
# preseason (season engine) forecasts (pure math)
# =============================================================================

def season_hit_target(
    prior_ppg: Optional[float],
    prior_games: int,
) -> Tuple[float, bool]:
    """(target PPG, has_valid_prior) per the backtest breakout definition:
    a valid prior season (>= 6 games at >= 4.0 PPG) sets the bar at
    prior x 1.15 with a 7.0 PPG floor; otherwise the bar is 10.0 PPG."""
    if (prior_ppg is not None and int(prior_games or 0) >= PRIOR_MIN_GAMES
            and float(prior_ppg) >= PRIOR_MIN_PPG):
        return round(max(float(prior_ppg) * TARGET_GROWTH, TARGET_FLOOR_PPG), 2), True
    return NO_PRIOR_TARGET_PPG, False


def blend_forecast_ppg(
    current_ppg: Optional[float],
    games: int,
    ros_ppg: Optional[float],
    ros_games: int,
) -> Optional[float]:
    """Games-weighted blend of banked PPG and rest-of-season projected PPG.
    Falls back to banked pace alone when no ROS projection exists."""
    if current_ppg is None:
        return None
    if ros_ppg is not None and int(ros_games) > 0:
        total_games = int(games) + int(ros_games)
        return round((float(current_ppg) * int(games)
                      + float(ros_ppg) * int(ros_games)) / total_games, 2)
    return round(float(current_ppg), 2)


def preseason_forecast(
    *,
    games: int,
    current_ppg: Optional[float],
    prior_ppg: Optional[float],
    prior_games: int,
    ros_ppg: Optional[float] = None,
    ros_games: int = 0,
) -> Dict[str, Any]:
    """Live forecast for one preseason (season engine) call. Pure.

    Zero games played (or no PPG recorded yet) returns an explicit pending
    state with no band. Otherwise the band compares the forecast PPG to the
    call's hit target: at or above target is Tracking to hit, within 10%
    under target is Borderline, below that is Tracking to miss.
    """
    target, has_prior = season_hit_target(prior_ppg, prior_games)
    base: Dict[str, Any] = {
        "kind": "preseason",
        "games": int(games),
        "current_ppg": wg._round(current_ppg),
        "prior_ppg": wg._round(prior_ppg),
        "prior_games": int(prior_games or 0),
        "target_ppg": target,
        "has_prior": has_prior,
        "forecast_ppg": None,
        "ros_ppg": wg._round(ros_ppg) if ros_ppg is not None else None,
        "ros_games": int(ros_games),
        "ros_available": bool(ros_ppg is not None and int(ros_games) > 0),
        "band": None,
        "band_label": None,
    }
    if not games or current_ppg is None:
        base.update({
            "state": "no_games",
            "basis": "No games played yet this season",
        })
        return base
    forecast_ppg = blend_forecast_ppg(current_ppg, games, ros_ppg, ros_games)
    if forecast_ppg >= target:
        band = BAND_TRACKING_HIT
    elif forecast_ppg >= target * BORDERLINE_FRACTION:
        band = BAND_BORDERLINE
    else:
        band = BAND_TRACKING_MISS
    basis = (f"Forecast {forecast_ppg:.1f} PPG vs a {target:.1f} PPG target, "
             f"{int(games)} game{'s' if games != 1 else ''} in")
    if not base["ros_available"]:
        basis += ". Pace only, no rest of season projection available"
    base.update({
        "state": "forecast",
        "band": band,
        "band_label": BAND_LABELS[band],
        "forecast_ppg": forecast_ppg,
        "basis": basis,
    })
    return base


# =============================================================================
# bulk loaders
# =============================================================================

def load_preseason_calls(season: int) -> List[Dict[str, Any]]:
    """The final pre-season snapshot per player for a season: the latest
    as_of_date whose phase is a pre-season phase. One row per player."""
    placeholders = ", ".join(["%s"] * len(PRESEASON_PHASES))
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT ON (player_id) * FROM breakout_opportunity_scores "
            f"WHERE season = %s AND phase IN ({placeholders}) "
            f"ORDER BY player_id, as_of_date DESC, calculated_at DESC",
            [int(season), *PRESEASON_PHASES],
        ).fetchall()
    return [dict(r) for r in rows]


def load_ros_projections(
    season: int,
    after_week: int,
    player_ids: Sequence[str],
    positions: Dict[str, Optional[str]],
) -> Dict[str, Tuple[float, int]]:
    """{player_id: (ros_ppg, projected_weeks)} from cached week projections.

    Reads the on-disk week projection cache only (never triggers a fetch):
    a week with no cached projection is skipped, as is a player absent
    from a week's map (bye / no projection). The per-week points are
    Sleeper's own published PPR totals for the standard PPR shape.
    """
    from utils.utils import _read_week_projection_file
    from utils.fantasy_scoring import weekly_projection_points

    scoring = {"rec": 1.0}
    wanted = [str(p) for p in player_ids]
    totals: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for week in range(int(after_week) + 1, FINAL_REGULATION_WEEK + 1):
        week_map = _read_week_projection_file(int(season), week)
        if not week_map:
            continue
        for pid in wanted:
            pts = weekly_projection_points(
                week_map, pid, scoring, positions.get(pid) or "")
            if pts is None:
                continue
            totals[pid] = totals.get(pid, 0.0) + float(pts)
            counts[pid] = counts.get(pid, 0) + 1
    return {pid: (round(totals[pid] / counts[pid], 2), counts[pid])
            for pid in totals}


def weekly_forecasts_for_calls(
    season: int,
    calls: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """{player_id: forecast} for the open calls among ``calls`` (raw
    weekly_breakout_scores rows). One bulk series load for the season,
    plus the prior-season series only when an open call's baseline PPG
    needs the prior-season fill, plus the cohort aggregate only when an
    open call has no baseline at all."""
    through = wg.default_through_week(season)
    if through is None:
        return {}
    open_calls = [c for c in calls
                  if not wg.is_mature(int(c.get("as_of_week") or 0), through)]
    if not open_calls:
        return {}
    series = wg.load_season_series(season, through)
    prior_series: Dict[str, List[Dict[str, Any]]] = {}
    if any(wg.call_needs_prior_ppg(c) for c in open_calls):
        prior_series = wg.load_prior_season_series(season)
    cohort: Dict[str, float] = {}
    if any(wg.call_needs_cohort_baseline(c) for c in open_calls):
        cohort = wg.load_cohort_baselines(season)
    out: Dict[str, Dict[str, Any]] = {}
    for call in open_calls:
        pid = str(call.get("player_id") or "")
        forecast = weekly_forecast(
            call, series.get(pid, []), through, prior_series.get(pid),
            cohort.get(str(call.get("position") or "")))
        if forecast is not None:
            out[pid] = forecast
    return out


def weekly_backtest_forecasts(
    season: int,
    limit: Optional[int] = None,
) -> Tuple[List[int], Dict[str, Dict[str, Any]]]:
    """Open-call forecasts for the season's reconstructed (backtest) weeks.

    Returns ``(weeks, views)``: ``weeks`` are the reconstructed weeks whose
    runs were actually loaded, ascending; ``views`` maps player_id to the
    call/forecast view for the player's most recent reconstructed call
    (weeks are walked oldest first, so the later week's forecast wins the
    merge). Each week's stored rows are capped to the top ``limit`` by
    breakout score, the same surfaced-candidates cap the boards use.
    Mature calls drop out via :func:`weekly_forecasts_for_calls` itself:
    once a call's outcome window completes it is the grader's business
    and leaves this outlook on its own. Each week fails soft on its own,
    and with no reconstructions the result is ``([], {})``.
    """
    from data_building.breakout_engine import weekly_store

    included: List[int] = []
    views: Dict[str, Dict[str, Any]] = {}
    for week in sorted(load_reconstructed_weeks(season)):
        try:
            run = weekly_store.get_reconstructed_run(season, week)
            if not run:
                continue
            # load_run_score_rows returns best score first.
            rows = weekly_store.load_run_score_rows(run["id"])
        except Exception:
            logger.warning(
                "forecasts: backtest week %s run load failed", week,
                exc_info=True)
            continue
        included.append(week)
        if limit:
            rows = rows[: int(limit)]
        try:
            calls_forecasts = weekly_forecasts_for_calls(season, rows)
        except Exception:
            logger.warning(
                "forecasts: backtest week %s forecast failed", week,
                exc_info=True)
            continue
        by_pid = {str(r.get("player_id") or ""): r for r in rows}
        for pid, fc in calls_forecasts.items():
            row = by_pid.get(pid) or {}
            views[pid] = {
                "player_id": pid,
                "player_name": row.get("player_name"),
                "classification": row.get("classification"),
                "breakout_score": wg._num(row.get("breakout_score")),
                "call_week": week,
                "forecast": fc,
            }
    return included, views


def preseason_forecasts_for_season(
    season: int,
    player_ids: Optional[Sequence[Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """{player_id: forecast} for the season's final pre-season calls.

    Bulk loads only: the snapshot query, the season and prior-season weekly
    series, and the cached ROS projection files.
    """
    calls = load_preseason_calls(season)
    if player_ids is not None:
        wanted = {str(p) for p in player_ids}
        calls = [c for c in calls if str(c.get("player_id") or "") in wanted]
    if not calls:
        return {}
    through = wg.default_through_week(season)
    series = wg.load_season_series(season, through) if through else {}
    prior_series = (wg.load_season_series(int(season) - 1, FINAL_REGULATION_WEEK)
                    if through else {})
    positions = {str(c.get("player_id") or ""): c.get("position") for c in calls}
    ros = (load_ros_projections(season, through, list(positions), positions)
           if through else {})
    out: Dict[str, Dict[str, Any]] = {}
    for call in calls:
        pid = str(call.get("player_id") or "")
        actual = wg.window_stats(series.get(pid, []))
        prior = wg.window_stats(prior_series.get(pid, []))
        ros_ppg, ros_games = ros.get(pid, (None, 0))
        forecast = preseason_forecast(
            games=actual["games"],
            current_ppg=actual["ppr_ppg"],
            prior_ppg=prior["ppr_ppg"],
            prior_games=prior["games"],
            ros_ppg=ros_ppg,
            ros_games=ros_games,
        )
        forecast.update({
            "player_name": call.get("player_name"),
            "phase": call.get("phase"),
            "breakout_score": wg._num(call.get("breakout_opportunity_score")),
        })
        out[pid] = forecast
    return out


# =============================================================================
# track record (finished grades only; forecasts never enter these numbers)
# =============================================================================

def load_weekly_grade_rows(season: int) -> List[Dict[str, Any]]:
    """Finished weekly grades for the season under the CURRENT scoring
    version only (older-version calls are never blended into this rate)."""
    from data_building.breakout_engine.weekly_breakout import SCORING_VERSION
    wg.init_weekly_breakout_grades_db()
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM {wg.GRADES_TABLE} "
            f"WHERE season = %s AND scoring_version = %s",
            (int(season), SCORING_VERSION),
        ).fetchall()
    return [dict(r) for r in rows]


def load_reconstructed_weeks(season: int) -> set:
    """Weeks whose current-version calls are v6 reconstructions: completed
    runs flagged ``detail.reconstructed`` under the current scoring version.

    A (season, week, scoring_version) maps to exactly one run, so grade
    rows are attributable exactly by their ``as_of_week``. Fails soft to
    an empty set: with no reconstruction metadata, no row is flagged as
    reconstructed.
    """
    from data_building.breakout_engine.weekly_breakout import SCORING_VERSION
    from data_building.breakout_engine.weekly_store import WEEKLY_RUNS_TABLE
    try:
        with get_conn() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT as_of_week FROM {WEEKLY_RUNS_TABLE} "
                f"WHERE season = %s AND scoring_version = %s "
                f"AND status = 'completed' "
                f"AND COALESCE(detail->>'reconstructed', 'false') = 'true'",
                (int(season), SCORING_VERSION),
            ).fetchall()
        return {int(r["as_of_week"]) for r in rows
                if r.get("as_of_week") is not None}
    except Exception:
        logger.warning("forecasts: reconstructed-week read failed",
                       exc_info=True)
        return set()


def _grade_row_view(row: Dict[str, Any],
                    reconstructed: bool = False) -> Dict[str, Any]:
    return {
        "player_id": str(row.get("player_id") or ""),
        "player_name": row.get("player_name"),
        "classification": row.get("classification"),
        "call_week": row.get("as_of_week"),
        "breakout_score": wg._num(row.get("breakout_score")),
        "outcome_games": row.get("outcome_games"),
        "ppg_delta": wg._num(row.get("ppg_delta")),
        "opp_delta": wg._num(row.get("opp_delta")),
        "snap_delta": wg._num(row.get("snap_delta")),
        "reconstructed": bool(reconstructed),
    }


def top_grade_rows(
    rows: Sequence[Dict[str, Any]],
    grade: str,
    limit: int = TOP_GRADES_LIMIT,
    reconstructed_weeks: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """The biggest graded hits / misses from grade rows. Pure.

    Hits rank by PPG delta descending. Misses rank by PPG delta ascending
    (the deepest production collapse first), falling back to opportunity
    delta when no PPG delta was recorded. Rows missing the ranking delta
    sort last, never first. When ``reconstructed_weeks`` is given, each
    view is flagged so the UI can tag backtest rows.
    """
    recon = reconstructed_weeks or set()
    picked = [
        _grade_row_view(r, reconstructed=r.get("as_of_week") in recon)
        for r in rows
        if r.get("grade") == grade
        and str(r.get("classification") or "") not in ("watchlist", "monitored")
    ]
    if grade == wg.GRADE_HIT:
        picked.sort(key=lambda v: (v["ppg_delta"] is None,
                                   -(v["ppg_delta"] or 0.0)))
    else:
        picked.sort(key=lambda v: (
            v["ppg_delta"] is None,
            v["ppg_delta"] if v["ppg_delta"] is not None else 0.0,
            v["opp_delta"] if v["opp_delta"] is not None else 0.0,
        ))
    return picked[: int(limit)]


def weekly_track_record(season: int) -> Dict[str, Any]:
    """Weekly hit rates per classification + biggest hits/misses, from
    finished grades only. Rates come from the grader's own summarizer, so
    the 10-graded floor applies exactly as it does for threshold review.

    Live and reconstructed (backtest) calls pool into ONE record: the
    overall and by-classification summaries cover every current-version
    graded call, whichever week it came from. Biggest hits/misses span
    both sets, with reconstructed rows flagged per row.
    """
    from data_building.breakout_engine.weekly_breakout import SCORING_VERSION
    try:
        rows = load_weekly_grade_rows(season)
    except Exception:
        logger.warning("forecasts: weekly grade read failed", exc_info=True)
        rows = []
    recon_weeks = load_reconstructed_weeks(season)
    summary = wg.summarize_grade_rows(rows)
    # Calibration bands over the same pooled rows, via the shared
    # calibration helpers so the rail and the calibration CLI can never
    # disagree. Bands surface only once something has graded (with zero
    # graded calls the block is its slim pending line and nothing else),
    # and a band with no calls at all is omitted, exactly like a
    # classification with no calls.
    band_summary = calibration.summarize_bands(rows)
    if summary["overall"]["graded"] > 0:
        score_bands = [
            _band_view(entry) for entry in band_summary["score_bands"]
            if entry["calls"] > 0
        ]
        confidence_bands = [
            _band_view(entry) for entry in band_summary["confidence_bands"]
            if entry["calls"] > 0
        ]
    else:
        score_bands = []
        confidence_bands = []
    return {
        "scoring_version": SCORING_VERSION,
        "min_sample": summary["min_sample"],
        "overall": summary["overall"],
        "groups": [
            {"classification": key, **bucket}
            for key, bucket in summary["by_classification"].items()
        ],
        "score_bands": score_bands,
        "confidence_bands": confidence_bands,
        "by_week": _weekly_rates_by_week(rows),
        "hits": top_grade_rows(rows, wg.GRADE_HIT,
                               reconstructed_weeks=recon_weeks),
        "misses": top_grade_rows(rows, wg.GRADE_MISS,
                                 reconstructed_weeks=recon_weeks),
    }


def _weekly_rates_by_week(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Hit rates grouped by call week, newest week first.

    Each entry carries the week number, call counts, and the graded hit
    rate (None until the 10-graded floor is met). Watchlist/monitored
    rows are excluded, matching the hits/misses filter: only actual
    breakout calls appear in the track record. Pure.
    """
    by_week: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("classification") or "") in ("watchlist", "monitored"):
            continue
        # Surfaced = breakout score >= 30 (product decision). A missing
        # score passes (old rows predate the threshold).
        _bs = row.get("breakout_score")
        if _bs is not None:
            try:
                if float(_bs) < 30:
                    continue
            except (TypeError, ValueError):
                pass
        try:
            wk = int(row.get("as_of_week") or 0)
        except (TypeError, ValueError):
            continue
        if wk <= 0:
            continue
        by_week.setdefault(wk, []).append(row)
    out = []
    for wk in sorted(by_week, reverse=True):
        bucket = wg._rate_bucket(by_week[wk], wg.MIN_SUMMARY_SAMPLE)
        out.append({"week": wk, **bucket})
    return out


def _band_view(entry: Dict[str, Any]) -> Dict[str, Any]:
    """A calibration band entry in the rail's group-row shape: the band
    label rides as ``label`` and the bucket fields pass through."""
    return {
        "label": entry["band"],
        "calls": entry["calls"],
        "graded": entry["graded"],
        "hit": entry["hit"],
        "partial": entry["partial"],
        "miss": entry["miss"],
        "ungraded": entry["ungraded"],
        "hit_rate": entry["hit_rate"],
        "partial_rate": entry["partial_rate"],
        "miss_rate": entry["miss_rate"],
    }


def season_track_record(season: int) -> Dict[str, Any]:
    """Season-engine hit rates by phase, from the season grades table.

    Fails soft to an explicit pending payload when the table is absent or
    empty: the section lights up on its own once season grading has run.
    When a call has both an early and a final grade row, the final verdict
    is the one counted.
    """
    pending: Dict[str, Any] = {"available": False, "groups": [], "overall": None}
    try:
        with get_conn() as conn:
            found = conn.execute(
                "SELECT 1 AS x FROM information_schema.tables "
                "WHERE table_name = %s",
                (SEASON_GRADES_TABLE,),
            ).fetchone()
        if not found:
            return pending
        with get_conn() as conn:
            rows = [dict(r) for r in conn.execute(
                f"SELECT * FROM {SEASON_GRADES_TABLE} WHERE season = %s",
                (int(season),),
            ).fetchall()]
    except Exception:
        logger.warning("forecasts: season grade read failed", exc_info=True)
        return pending
    if not rows:
        return pending
    best: Dict[Tuple[str, str], Tuple[int, Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("player_id") or ""), str(row.get("as_of_date") or ""))
        rank = 1 if str(row.get("grading_stage") or "") == "final" else 0
        if key not in best or rank > best[key][0]:
            best[key] = (rank, row)
    deduped = [row for _rank, row in best.values()]
    by_phase: Dict[str, List[Dict[str, Any]]] = {}
    for row in deduped:
        by_phase.setdefault(str(row.get("phase") or "unknown"), []).append(row)
    return {
        "available": True,
        "groups": [
            {"phase": phase, **wg._rate_bucket(group, wg.MIN_SUMMARY_SAMPLE)}
            for phase, group in sorted(by_phase.items())
        ],
        "overall": wg._rate_bucket(deduped, wg.MIN_SUMMARY_SAMPLE),
    }
