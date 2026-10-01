"""
Realized-outcome grading for the SEASON breakout engine's preseason calls.

PR #2169's :mod:`weekly_grading` closed the feedback loop for the in-season
weekly board, but the season engine's calls - the preseason breakout list -
were never graded against what actually happened. This module is that second
grader, built to the same conventions:

* :func:`grade_season_breakouts` loads, for each player-season, the FINAL
  pre-season snapshot from ``breakout_opportunity_scores`` (the latest
  ``as_of_date`` row whose phase is one of the pre-season phases in
  ``config.PHASE_WEIGHTS`` - offseason / post_free_agency / post_draft /
  preseason; ``in_season`` snapshots are later revisions, not the call as it
  stood entering the season) and grades it against realized PPR production
  from ``player_weekly_metrics`` (the weekly grader's outcome source). The
  stored ``season`` is the season being predicted (see build_historical_
  scores: rows are saved with ``season = prediction_season + 1``, "the
  season being predicted"), so the outcome window is that season and the
  baseline is the season before it.
* Grades are written in TWO stages per call, each immutable once written:
  ``early`` (outcome weeks 1-8, written once week 8 data exists, flagged
  provisional - a preseason call is a season-long prediction, so this is an
  early read, not a verdict of record) and ``final`` (outcome weeks 1
  through the last stored week, written once the season is complete: the
  latest stored week has reached 17, the point past which no regular-season
  outcome can still change the picture). UNIQUE is on call identity +
  stage, so the final grade never overwrites the early one.
* :func:`summarize_grades` aggregates hit / partial / miss rates by phase,
  by season, and by predicted-score band, with the weekly grader's honesty
  rule: rates are None below ``MIN_SUMMARY_SAMPLE`` graded calls in a group;
  counts are always real.

Hit definition - REUSED VERBATIM from the season engine's own backtest,
``backtest_multitask.get_breakout_pids`` (the same label ``model_health``
evaluates the shipped model's stored predictions against):

* the outcome sample must reach the backtest's games floor (10 games);
* a prior baseline "exists" when the player's prior season had >= 6 games
  at >= 4.0 PPR PPG; with a baseline, a hit needs outcome PPG >= prior PPG
  x 1.15 AND outcome PPG >= 7.0;
* with no meaningful prior baseline (rookie / scratch), a hit needs
  outcome PPG >= 10.0.

The four-way verdict maps that binary label onto the weekly grader's
hit / partial / miss / ungraded scale:

* hit     = the backtest bar above, cleared on a full backtest sample.
* partial = the PPG bar cleared but the season was cut short (games
  between the stage floor and the backtest's 10), or production improved
  without clearing the bar, or a no-baseline player landed between the
  7.0 floor and the 10.0 scratch bar.
* miss    = with a baseline: outcome PPG at or below the prior season's
  PPG (no improvement at all). Without a baseline: outcome PPG below 7.0.
* ungraded = no games in the outcome window (absence is not a failed
  call), too few games for the stage's sample floor (early: 4 games in
  weeks 1-8; final: 8 games), or no recorded PPG.

At the ``early`` stage the 10-game backtest floor is unreachable (weeks
1-8 hold at most 8 games), so the stage floor stands in for it and every
early grade is flagged provisional; the PPG bar itself is unchanged.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from dashboard_services.db import get_conn
from data_building.breakout_engine.config import (
    BREAKOUT_SCORES_TABLE,
    PHASE_WEIGHTS,
)
from data_building.breakout_engine.weekly_grading import (
    _num,
    _rate_bucket,
    _round,
    default_through_week,
    window_stats,
)

GRADES_TABLE = "season_breakout_grades"
GRADING_VERSION = "season-grading-v1"

# The call that counts is the final snapshot taken before the season:
# every configured phase except in_season (config.PHASE_WEIGHTS keys are
# offseason, post_free_agency, post_draft, preseason, in_season).
PRESEASON_PHASES: Tuple[str, ...] = tuple(
    phase for phase in PHASE_WEIGHTS if phase != "in_season"
)

STAGE_EARLY = "early"
STAGE_FINAL = "final"
STAGES = (STAGE_EARLY, STAGE_FINAL)

# Stage windows / gates. Early reads cover weeks 1-8 and open once week 8
# data exists. Final grades cover the whole season and open once the
# latest stored week reaches 17 (season effectively complete).
EARLY_WINDOW_END = 8
SEASON_COMPLETE_MIN_WEEK = 17

# Hit-definition thresholds, verbatim from backtest_multitask
# .get_breakout_pids (do not retune here; the backtest owns them).
BACKTEST_MIN_GAMES = 10      # outcome games the label itself requires
PRIOR_MIN_GAMES = 6          # prior season games for a baseline to exist
PRIOR_MIN_PPG = 4.0          # prior season PPG for a baseline to exist
GROWTH_MULTIPLE = 1.15       # outcome PPG >= prior PPG x this
FLOOR_PPG = 7.0              # absolute outcome PPG floor with a baseline
SCRATCH_PPG = 10.0           # outcome PPG bar with no baseline

# Per-stage minimum outcome games before a call is gradeable at all
# (below: ungraded - the sample cannot carry a verdict).
EARLY_MIN_GAMES = 4
FINAL_MIN_GAMES = 8

# Rates in summarize_grades are None below this many *graded* calls in a
# group (counts are always reported). Mirrors weekly_grading.
MIN_SUMMARY_SAMPLE = 10

GRADE_HIT = "hit"
GRADE_PARTIAL = "partial"
GRADE_MISS = "miss"
GRADE_UNGRADED = "ungraded"

_INIT_DONE = False


def init_season_breakout_grades_db() -> None:
    """Create the grades table and indexes if absent. Idempotent; safe to
    call on every run (acts as the migration for existing databases)."""
    global _INIT_DONE
    if _INIT_DONE:
        return
    with get_conn() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {GRADES_TABLE} (
                id               SERIAL PRIMARY KEY,
                player_id        VARCHAR(50) NOT NULL,
                player_name      VARCHAR(255),
                season           INTEGER NOT NULL,
                as_of_date       DATE NOT NULL,
                phase            VARCHAR(20),
                grading_stage    VARCHAR(10) NOT NULL,
                grading_version  VARCHAR(40) NOT NULL,
                provisional      BOOLEAN DEFAULT FALSE,
                breakout_score   NUMERIC,
                confidence       NUMERIC,
                hit_probability  NUMERIC,
                projected_role_tag VARCHAR(100),
                grade            VARCHAR(10) NOT NULL,
                outcome_weeks    INTEGER[],
                outcome_games    INTEGER,
                baseline_season  INTEGER,
                baseline_games   INTEGER,
                baseline_ppg     NUMERIC,
                outcome_ppg      NUMERIC,
                ppg_delta        NUMERIC,
                detail           JSONB,
                graded_at        TIMESTAMP DEFAULT NOW(),
                UNIQUE (player_id, season, as_of_date, grading_stage)
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_sbg_season_grade "
            f"ON {GRADES_TABLE} (season, grade)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_sbg_phase "
            f"ON {GRADES_TABLE} (phase)"
        )
    _INIT_DONE = True


# =============================================================================
# pure helpers
# =============================================================================

def select_final_preseason_calls(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """The gradeable call per (player_id, season): the latest as_of_date
    snapshot whose phase is a pre-season phase. IN_SEASON snapshots are
    later revisions of the prediction, not the preseason call, and a row
    without a recognizable pre-season phase cannot be proven to be one.
    Pure; the DB loader applies the same rule in SQL."""
    best: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in rows:
        if str(row.get("phase") or "") not in PRESEASON_PHASES:
            continue
        key = (str(row.get("player_id") or ""), int(row.get("season") or 0))
        current = best.get(key)
        if current is None or _date_key(row) >= _date_key(current):
            best[key] = row
    return [best[key] for key in sorted(best)]


def _date_key(row: Dict[str, Any]) -> str:
    return str(row.get("as_of_date") or "")[:10]


def stage_eligible(stage: str, through_week: Optional[int]) -> bool:
    """Whether a stage's outcome window has closed given the latest stored
    week of the season. Early opens at week 8; final opens once the season
    is complete (latest stored week >= 17)."""
    if through_week is None:
        return False
    if stage == STAGE_EARLY:
        return int(through_week) >= EARLY_WINDOW_END
    if stage == STAGE_FINAL:
        return int(through_week) >= SEASON_COMPLETE_MIN_WEEK
    return False


def stage_window(stage: str) -> Tuple[int, Optional[int]]:
    """(first_week, last_week) of a stage's outcome window; last_week None
    means through the end of the stored season (final stage)."""
    if stage == STAGE_EARLY:
        return (1, EARLY_WINDOW_END)
    return (1, None)


def ppg_bar_cleared(
    prior: Dict[str, Any],
    outcome_ppg: Optional[float],
) -> Tuple[bool, bool]:
    """(bar_cleared, prior_valid) under the backtest's hit definition.

    ``prior`` is a window_stats dict over the player's prior season;
    ``outcome_ppg`` is the realized PPG over the stage window. The
    thresholds are backtest_multitask.get_breakout_pids verbatim; the
    10-game label floor is applied by the caller, not here, so the early
    stage can reuse the bar with its own sample floor.
    """
    if outcome_ppg is None:
        return False, False
    prior_games = int(prior.get("games") or 0)
    prior_ppg = prior.get("ppr_ppg")
    prior_valid = (
        prior_games >= PRIOR_MIN_GAMES
        and prior_ppg is not None
        and float(prior_ppg) >= PRIOR_MIN_PPG
    )
    if prior_valid:
        return (float(outcome_ppg) >= float(prior_ppg) * GROWTH_MULTIPLE
                and float(outcome_ppg) >= FLOOR_PPG), True
    return float(outcome_ppg) >= SCRATCH_PPG, False


def classify_season_outcome(
    prior: Dict[str, Any],
    outcome: Dict[str, Any],
    stage: str,
) -> Dict[str, Any]:
    """Classify one realized outcome. Pure; see the module docstring for
    the verdict rules. ``prior`` / ``outcome`` are window_stats dicts."""
    games = int(outcome.get("games") or 0)
    outcome_ppg = outcome.get("ppr_ppg")
    if games == 0:
        return {"grade": GRADE_UNGRADED, "reason": "no_outcome_games",
                "prior_valid": False, "bar_cleared": False}
    if outcome_ppg is None:
        return {"grade": GRADE_UNGRADED, "reason": "no_outcome_ppg",
                "prior_valid": False, "bar_cleared": False}
    stage_floor = EARLY_MIN_GAMES if stage == STAGE_EARLY else FINAL_MIN_GAMES
    if games < stage_floor:
        return {"grade": GRADE_UNGRADED, "reason": "insufficient_games",
                "prior_valid": False, "bar_cleared": False}

    bar_cleared, prior_valid = ppg_bar_cleared(prior, outcome_ppg)
    prior_ppg = prior.get("ppr_ppg")

    if bar_cleared:
        if games >= BACKTEST_MIN_GAMES or stage == STAGE_EARLY:
            # Final stage: the backtest's own 10-game sample. Early stage:
            # unreachable by construction in weeks 1-8, so the stage floor
            # stands in and the grade is provisional (see stage flag).
            reason = ("ppg_bar_cleared" if stage == STAGE_FINAL
                      else "ppg_bar_cleared_provisional")
            return {"grade": GRADE_HIT, "reason": reason,
                    "prior_valid": prior_valid, "bar_cleared": True}
        return {"grade": GRADE_PARTIAL,
                "reason": "ppg_bar_cleared_below_backtest_games_floor",
                "prior_valid": prior_valid, "bar_cleared": True}
    if prior_valid and prior_ppg is not None and float(outcome_ppg) <= float(prior_ppg):
        return {"grade": GRADE_MISS, "reason": "no_improvement_over_prior",
                "prior_valid": True, "bar_cleared": False}
    if not prior_valid and float(outcome_ppg) < FLOOR_PPG:
        return {"grade": GRADE_MISS, "reason": "below_breakout_floor_no_prior",
                "prior_valid": False, "bar_cleared": False}
    reason = ("improved_below_breakout_bar" if prior_valid
              else "established_below_scratch_bar")
    return {"grade": GRADE_PARTIAL, "reason": reason,
            "prior_valid": prior_valid, "bar_cleared": False}


def grade_season_call(
    call: Dict[str, Any],
    prior_rows: Sequence[Dict[str, Any]],
    outcome_rows: Sequence[Dict[str, Any]],
    stage: str,
    finish: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Grade one preseason call at one stage. Pure.

    ``prior_rows`` is the player's full player_weekly_metrics series for
    the season before the call's season; ``outcome_rows`` the full series
    for the call's season (the stage window is applied here). ``finish``
    is an optional :func:`compute_position_finishes` entry for the player,
    recorded as descriptive detail only - never an input to the verdict.
    """
    season = int(call.get("season") or 0)
    first_week, last_week = stage_window(stage)
    window_rows = [
        r for r in outcome_rows
        if _num(r.get("week")) is not None
        and first_week <= int(r["week"])
        and (last_week is None or int(r["week"]) <= last_week)
    ]
    prior = window_stats(list(prior_rows))
    outcome = window_stats(window_rows)
    verdict = classify_season_outcome(prior, outcome, stage)
    baseline_ppg = prior.get("ppr_ppg")
    outcome_ppg = outcome.get("ppr_ppg")
    ppg_delta = (None if baseline_ppg is None or outcome_ppg is None
                 else float(outcome_ppg) - float(baseline_ppg))
    finish = finish or {}
    return {
        "player_id": str(call.get("player_id") or ""),
        "player_name": call.get("player_name"),
        "season": season,
        "as_of_date": call.get("as_of_date"),
        "phase": call.get("phase"),
        "grading_stage": stage,
        "grading_version": GRADING_VERSION,
        "provisional": stage == STAGE_EARLY,
        "breakout_score": _num(call.get("breakout_opportunity_score")),
        "confidence": _num(call.get("confidence_score")),
        "hit_probability": _num(call.get("hit_probability")),
        "projected_role_tag": call.get("projected_role_tag"),
        "grade": verdict["grade"],
        "outcome_weeks": sorted({int(r["week"]) for r in window_rows}),
        "outcome_games": outcome["games"],
        "baseline_season": season - 1,
        "baseline_games": prior["games"],
        "baseline_ppg": _round(baseline_ppg),
        "outcome_ppg": _round(outcome_ppg),
        "ppg_delta": _round(ppg_delta),
        "detail": {
            "stage": stage,
            "provisional": stage == STAGE_EARLY,
            "reason": verdict["reason"],
            "prior_valid": verdict["prior_valid"],
            "bar_cleared": verdict["bar_cleared"],
            "position": call.get("position"),
            "team": call.get("team"),
            "outcome_position_rank": finish.get("position_rank"),
            "top_n_finish": finish.get("top_n_finish"),
            "thresholds": {
                "growth_multiple": GROWTH_MULTIPLE,
                "floor_ppg": FLOOR_PPG,
                "scratch_ppg": SCRATCH_PPG,
                "prior_min_games": PRIOR_MIN_GAMES,
                "prior_min_ppg": PRIOR_MIN_PPG,
                "backtest_min_games": BACKTEST_MIN_GAMES,
                "stage_min_games": (EARLY_MIN_GAMES if stage == STAGE_EARLY
                                    else FINAL_MIN_GAMES),
            },
        },
    }


def compute_position_finishes(
    series_by_player: Dict[str, List[Dict[str, Any]]],
    stage: str,
) -> Dict[str, Dict[str, Any]]:
    """player_id -> {position_rank, top_n_finish} over a stage window.

    Descriptive context for grades: every player with at least one game in
    the window is ranked by total PPR within his position (position from
    his weekly rows, the outcome source) and the top-N cutoffs mirror
    backtest_multitask.get_top12_pids (top 6 QB/TE, top 12 elsewhere).
    Unlike that label, no 10-game eligibility filter is applied: this is a
    finish line for the window as it stands, not the hit label itself.
    Pure.
    """
    first_week, last_week = stage_window(stage)
    totals: Dict[str, List[Any]] = {}
    for player_id, rows in series_by_player.items():
        games = 0
        total = 0.0
        position = ""
        for row in rows:
            week = _num(row.get("week"))
            if week is None or int(week) < first_week:
                continue
            if last_week is not None and int(week) > last_week:
                continue
            games += 1
            total += float(_num(row.get("ppr_pts")) or 0.0)
            position = position or str(row.get("position") or "")
        if games:
            totals.setdefault(position, []).append((str(player_id), total))
    finishes: Dict[str, Dict[str, Any]] = {}
    for position, players in totals.items():
        players.sort(key=lambda item: item[1], reverse=True)
        cutoff = 6 if position in ("QB", "TE") else 12
        for rank, (player_id, _total) in enumerate(players, start=1):
            finishes[player_id] = {
                "position_rank": rank,
                "top_n_finish": rank <= cutoff,
            }
    return finishes


def call_key(call: Dict[str, Any], stage: str) -> Tuple[str, int, str, str]:
    """The identity a grade is keyed by: the call's own unique key plus
    the grading stage (early and final grades coexist)."""
    return (
        str(call.get("player_id") or ""),
        int(call.get("season") or 0),
        _date_key(call),
        str(stage or ""),
    )


def score_band(score: Any) -> str:
    """Predicted-score band for threshold review (the live storage floor
    is 60, so the bands split the stored range)."""
    value = _num(score)
    if value is None:
        return "unknown"
    if value < 60:
        return "<60"
    if value < 70:
        return "60-69"
    if value < 80:
        return "70-79"
    if value < 90:
        return "80-89"
    return "90+"


# =============================================================================
# summary (pure aggregation + DB accessor)
# =============================================================================

def summarize_grade_rows(
    rows: Sequence[Dict[str, Any]],
    min_sample: int = MIN_SUMMARY_SAMPLE,
) -> Dict[str, Any]:
    """Hit / partial / miss rates overall, by phase, by season, and by
    predicted-score band. Rates are over graded calls only (ungraded calls
    are excluded from the denominator) and are None for any group with
    fewer than ``min_sample`` graded calls - counts are always real, never
    suppressed. Same rule as weekly_grading.summarize_grade_rows."""
    rows = list(rows)
    by_phase: Dict[str, List[Dict[str, Any]]] = {}
    by_season: Dict[str, List[Dict[str, Any]]] = {}
    by_band: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_phase.setdefault(str(row.get("phase") or "unknown"), []).append(row)
        by_season.setdefault(str(row.get("season") or "unknown"), []).append(row)
        by_band.setdefault(score_band(row.get("breakout_score")), []).append(row)
    return {
        "min_sample": int(min_sample),
        "overall": _rate_bucket(rows, min_sample),
        "by_phase": {
            key: _rate_bucket(group, min_sample)
            for key, group in sorted(by_phase.items())
        },
        "by_season": {
            key: _rate_bucket(group, min_sample)
            for key, group in sorted(by_season.items())
        },
        "by_score_band": {
            key: _rate_bucket(group, min_sample)
            for key, group in sorted(by_band.items())
        },
    }


def summarize_grades(
    season: Optional[int] = None,
    stage: str = STAGE_FINAL,
    min_sample: int = MIN_SUMMARY_SAMPLE,
) -> Dict[str, Any]:
    """DB accessor over persisted grades; see :func:`summarize_grade_rows`.
    Defaults to final grades (the verdict of record); pass stage='early'
    for the provisional reads."""
    init_season_breakout_grades_db()
    query = f"SELECT * FROM {GRADES_TABLE} WHERE grading_stage = %s"
    params: List[Any] = [stage]
    if season is not None:
        query += " AND season = %s"
        params.append(int(season))
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return summarize_grade_rows([dict(r) for r in rows], min_sample=min_sample)


# =============================================================================
# DB layer: loaders, saver, orchestration
# =============================================================================

def load_final_preseason_calls(season: int) -> List[Dict[str, Any]]:
    """The final pre-season snapshot per player for a season - the same
    selection :func:`select_final_preseason_calls` makes, in SQL."""
    placeholders = ", ".join(["%s"] * len(PRESEASON_PHASES))
    with get_conn() as conn:
        rows = conn.execute(
            f"""
            SELECT DISTINCT ON (player_id)
                player_id, player_name, season, as_of_date, team, position,
                phase, breakout_opportunity_score, confidence_score,
                hit_probability, projected_role_tag
            FROM {BREAKOUT_SCORES_TABLE}
            WHERE season = %s AND phase IN ({placeholders})
            ORDER BY player_id, as_of_date DESC, calculated_at DESC
            """,
            (int(season), *PRESEASON_PHASES),
        ).fetchall()
    return [dict(r) for r in rows]


def load_call_seasons() -> List[int]:
    """Every season with stored season-engine calls, ascending - the
    backfill frontier for the CLI."""
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT DISTINCT season FROM {BREAKOUT_SCORES_TABLE} "
            f"ORDER BY season"
        ).fetchall()
    return [int(r["season"]) for r in rows if r.get("season") is not None]


def load_existing_grade_keys(
    season: int,
) -> Set[Tuple[str, int, str, str]]:
    """(call key, stage) pairs that already have a grade - the
    idempotency frontier."""
    init_season_breakout_grades_db()
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT player_id, season, as_of_date, grading_stage "
            f"FROM {GRADES_TABLE} WHERE season = %s",
            (int(season),),
        ).fetchall()
    return {
        (str(r["player_id"]), int(r["season"]), _date_key(r),
         str(r["grading_stage"]))
        for r in rows
    }


def load_season_series(
    season: int, through_week: int
) -> Dict[str, List[Dict[str, Any]]]:
    """player_id -> weekly_metrics rows through ``through_week``. The same
    loader the weekly grader reads, so grading sees exactly its source."""
    from data_building.weekly_metrics import get_weekly_series_by_player
    return get_weekly_series_by_player(int(season), int(through_week))


def save_grade_rows(grades: List[Dict[str, Any]]) -> int:
    """Insert grades idempotently. An existing grade for the same call
    and stage is never duplicated and never rewritten (grades are
    immutable history, and an early grade never overwrites a final one).
    Returns the number of rows actually inserted."""
    if not grades:
        return 0
    init_season_breakout_grades_db()
    cols = [
        "player_id", "player_name", "season", "as_of_date", "phase",
        "grading_stage", "grading_version", "provisional",
        "breakout_score", "confidence", "hit_probability",
        "projected_role_tag", "grade", "outcome_weeks", "outcome_games",
        "baseline_season", "baseline_games", "baseline_ppg", "outcome_ppg",
        "ppg_delta", "detail",
    ]
    placeholders = ", ".join(
        f"%({column})s::jsonb" if column == "detail" else f"%({column})s"
        for column in cols
    )
    inserted = 0
    with get_conn() as conn:
        for grade in grades:
            row = dict(grade)
            row["detail"] = json.dumps(grade.get("detail") or {})
            result = conn.execute(
                f"INSERT INTO {GRADES_TABLE} ({', '.join(cols)}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT (player_id, season, as_of_date, grading_stage) "
                f"DO NOTHING RETURNING id",
                row,
            ).fetchone()
            if result:
                inserted += 1
    return inserted


def grade_season_breakouts(
    season: int,
    *,
    through_week: Optional[int] = None,
    stages: Sequence[str] = STAGES,
) -> Dict[str, Any]:
    """Grade every stage-ready, not-yet-graded preseason call for a season.

    A stage is graded once its outcome window has closed (see
    :func:`stage_eligible`); ``through_week`` pins that frontier and by
    default is the latest stored week in player_weekly_metrics.
    Already-graded (call, stage) pairs are skipped before any math, and
    the weekly series is only loaded when something is pending, so repeat
    runs are cheap no-ops. Returns a run summary dict.
    """
    init_season_breakout_grades_db()
    through = (int(through_week) if through_week is not None
               else default_through_week(season))
    summary: Dict[str, Any] = {
        "season": int(season),
        "through_week": through,
        "grading_version": GRADING_VERSION,
        "calls_considered": 0,
        "skipped_stage_gate": 0,
        "already_graded": 0,
        "graded": 0,
        "inserted": 0,
        "by_grade": {},
        "by_stage": {},
    }
    if through is None:
        summary["status"] = "skipped"
        summary["reason"] = "no weekly metrics for season"
        return summary

    calls = load_final_preseason_calls(season)
    existing = load_existing_grade_keys(season)
    summary["calls_considered"] = len(calls)

    pending: List[Tuple[Dict[str, Any], str]] = []
    for call in calls:
        for stage in stages:
            if call_key(call, stage) in existing:
                summary["already_graded"] += 1
                continue
            if not stage_eligible(stage, through):
                summary["skipped_stage_gate"] += 1
                continue
            pending.append((call, stage))

    if pending:
        outcome_series = load_season_series(season, through)
        prior_series = load_season_series(season - 1, 99)
        finishes = {
            stage: compute_position_finishes(outcome_series, stage)
            for stage in {stage for _call, stage in pending}
        }
        grades = [
            grade_season_call(
                call,
                prior_series.get(str(call.get("player_id") or ""), []),
                outcome_series.get(str(call.get("player_id") or ""), []),
                stage,
                finish=finishes[stage].get(str(call.get("player_id") or "")),
            )
            for call, stage in pending
        ]
        summary["inserted"] = save_grade_rows(grades)
        summary["graded"] = len(grades)
        for grade in grades:
            key = grade["grade"]
            summary["by_grade"][key] = summary["by_grade"].get(key, 0) + 1
            stage_key = grade["grading_stage"]
            summary["by_stage"][stage_key] = (
                summary["by_stage"].get(stage_key, 0) + 1
            )
    summary["status"] = "completed"
    return summary
