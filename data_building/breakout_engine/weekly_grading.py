"""
Realized-outcome grading for in-season weekly breakout calls.

The weekly engine (``weekly_breakout`` / ``weekly_store``) persists every call
it makes, but nothing ever checked those calls against what actually happened,
so the score thresholds (WATCHLIST_MIN_SCORE / EMERGING_MIN_SCORE) were never
tuned from live outcomes. This module closes that loop:

* :func:`grade_weekly_breakouts` loads persisted calls that have matured
  (at least ``OUTCOME_WEEKS`` subsequent weeks of data exist - three NFL
  weeks is the ~21-day horizon the call needs to prove itself over), computes
  the realized outcome over the 3 weeks AFTER the call week from
  ``player_weekly_metrics`` (the same source the engine reads), and persists
  one grade per call.
* :func:`summarize_grades` aggregates grades into hit / partial / miss rates
  grouped by classification and by scoring version, for threshold review.

Grading is idempotent within a grading version: a call's grade is keyed
by the call's identity (player_id, season, as_of_week, scoring_version)
and re-running under the same GRADING_VERSION never duplicates or
rewrites a grade. A new GRADING_VERSION is a new verdict on the same
history, not new history: stale-version grades stop counting as the
idempotency frontier (see :func:`load_existing_grade_keys`) and the
saver overwrites them in place (see :func:`save_grade_rows`), so a
grading-logic change regrades every stored call on the next run.

Like the scorer, the outcome math is pure: :func:`grade_call` and
:func:`classify_outcome` operate on plain row dicts and never touch the DB.
All DB access is confined to the loader / saver functions at the bottom.

Outcome thresholds (change vs the call's stored baseline; see
:func:`classify_outcome`). Since weekly-grading-v5 the verdict is
production-first: the production delta decides, and the role leg is
still computed and recorded on every grade (the report header and the
receipts print it) but only names the partial inside the flat band.

* production rose: PPR PPG delta >= +2.0
* production collapsed: PPR PPG delta <= -2.0
* hit    = production rose AND the outcome PPG itself reached the
  position's startable floor (HIT_PPG_FLOOR: QB 20, RB 10, WR 10,
  TE 8 PPR PPG), on at least 2 outcome games. The floor exists because
  a delta alone can lie: +2.25 PPG off a 1.8 baseline lands at 4.05
  PPG, which is not a breakout anyone can start. A hit-pace outcome
  on a single game is a partial (single_game_sample): one game cannot
  prove a breakout. The role no longer gates a hit: production that
  rose to a startable level broke out even if the role mix shifted
  underneath it (production_rose_despite_role_reverted is a hit now,
  not the v4 partial).
* miss   = production collapsed, whatever the role did (a player who
  kept the snaps but stopped producing did not break out).
* partial = anything in between: production rose but landed below the
  floor (production_rose_below_floor), or production inside the flat
  band (-2.0 < delta < +2.0) on any role. A reverted role no longer
  turns flat production into a miss; inside the band the role only
  names the partial (role_held_without_production_rise /
  production_flat_role_reverted / role_partially_retained).
* role state is still classified exactly as before - held:
  opportunity delta >= +2.0 per game, OR snap-share delta >= +8.0
  percentage points with opportunities not falling (opportunity
  delta >= 0, or unmeasured; snaps alone cannot carry the role when
  the player is actually being used less); reverted: every available
  role delta back at baseline noise (opportunity <= +0.5 per game AND
  snap share <= +3.0 points); otherwise mixed - and stored on the
  grade as role_state.
* when no production delta can be measured (a prior-season baseline
  whose prior-season rows were not supplied), the pre-v5 role-only
  verdicts apply unchanged, with their partial ceiling: role reverted
  is a miss (role_reverted_to_baseline), role held or mixed is a
  partial, and a hit still requires a measured production rise.
* injured = the outcome window was wiped out by injury: fewer than 2
  games played while the player carried an IR / PUP / NFI designation
  (any outcome week) or an OUT / DOUBTFUL designation (>= 2 of the 3
  outcome weeks). QUESTIONABLE never voids. Injury is a terminal grade
  (excluded from hit-rate denominators) - an injured player did not
  fail the call, the call simply cannot be evaluated. With >= 2 games
  played the call had its shot and grades normally on those games, no
  matter the designation.
* ungraded = the player had no games in the outcome window for any
  other reason (bye-stacked, healthy scratch: absence is not a failed
  call) or no role baseline exists to grade against.

Injury designations come from weekly snapshots of the Sleeper players
feed (see :func:`snapshot_injury_statuses`), because a designation is
only meaningful if it was recorded DURING the outcome week. No
backfill: calls whose outcome weeks predate snapshotting grade under
the normal rules.

Baseline precedence, per field: the call's stored evidence first
(``fantasy.baseline_ppg``, ``signals.snap_share.baseline``, and for RBs
``signals.carry_opportunity_pg.baseline`` - the engine's carries+targets
composite). Opportunity for other positions has no unit-consistent stored
composite, so it (and any other missing field) is computed from the weekly
rows over the call's stored ``baseline_weeks``; when those are absent, the
up-to-3 weeks immediately before the call week. Prior-season baselines
(``baseline_source == 'prior_season'``) never fall back to current-season
weeks - those weeks are the call's *recent* window, not its baseline.
Their stored evidence carries no PPG (the engine's prior-season pseudo-row
has no fantasy points), so when the caller supplies the player's
prior-season rows, the baseline PPG is filled from them instead: the mean
PPR PPG over that prior season, registered as a computed field. Without
prior-season rows such calls grade with no PPG delta and cap at partial
when the role held (a hit requires a measured production rise).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from dashboard_services.db import get_conn
from data_building.breakout_engine.weekly_breakout import display_classification
from data_building.breakout_engine.weekly_store import (
    WEEKLY_SCORES_TABLE,
    get_serving_run,
    init_weekly_breakout_db,
)

GRADES_TABLE = "weekly_breakout_grades"
INJURY_SNAPSHOTS_TABLE = "player_injury_snapshots"
GRADING_VERSION = "weekly-grading-v5"

logger = logging.getLogger(__name__)

# Cohort baselines for initial-role (no-baseline) calls: a rookie is
# measured against the typical rookie year at his position, not against
# a personal history he does not have. The cohort is the six seasons
# before the call season; a debut season counts with >= COHORT_MIN_GAMES
# games, and a position publishes a baseline with >= COHORT_MIN_PLAYERS
# such player-seasons. Below either floor there is no cohort baseline
# and the call stays honestly ungraded / pending.
COHORT_LOOKBACK_SEASONS = 6
COHORT_MIN_GAMES = 6
COHORT_MIN_PLAYERS = 15
COHORT_POSITIONS = ("QB", "RB", "WR", "TE")

# A call is graded over the 3 weeks after the call week, and only once those
# 3 subsequent weeks exist in the data (3 NFL weeks ~= 21 days).
OUTCOME_WEEKS = 3
BASELINE_FALLBACK_WEEKS = 3

# Outcome thresholds (deltas vs the call's baseline). See module docstring.
PPG_HIT_DELTA = 2.0          # PPR points per game
PPG_MISS_DELTA = -2.0        # PPR points per game: a collapse this deep is
                             # a miss on its own (production-aware miss)
OPP_HELD_DELTA = 2.0         # carries+targets per game
SNAP_HELD_DELTA = 8.0        # snap-share percentage points
OPP_REVERTED_DELTA = 0.5     # carries+targets per game
SNAP_REVERTED_DELTA = 3.0    # snap-share percentage points
HIT_MIN_GAMES = 2            # a hit needs at least this many outcome games;
                             # a 1-game hit-pace outcome caps at partial
# Production floor for a hit (weekly-grading-v5): the outcome PPG must
# itself be startable at the position, not just up from baseline - a
# delta off a near-zero baseline is not a breakout (the Chris Bell
# shape: +2.25 PPG landing at 4.05). The values mirror _MIN_COMP_PPG
# in data_building/breakout_engine/projections.py, the comps layer's
# usable-production levels; kept as a literal here so this module
# never imports the projections layer.
HIT_PPG_FLOOR = {"QB": 20.0, "RB": 10.0, "WR": 10.0, "TE": 8.0}
HIT_PPG_FLOOR_DEFAULT = 10.0


def hit_ppg_floor(position: Any) -> float:
    """The outcome-PPG floor a hit must clear at ``position``."""
    return HIT_PPG_FLOOR.get(str(position or "").strip().upper(),
                             HIT_PPG_FLOOR_DEFAULT)

# Rates in summarize_grades are None below this many *graded* calls in a
# group (counts are always reported). Small samples produce noise, not signal.
MIN_SUMMARY_SAMPLE = 10

GRADE_HIT = "hit"
GRADE_PARTIAL = "partial"
GRADE_MISS = "miss"
GRADE_UNGRADED = "ungraded"
GRADE_INJURED = "injured"

_INIT_DONE = False
_SNAPSHOT_INIT_DONE = False


def init_weekly_breakout_grades_db() -> None:
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
                as_of_week       INTEGER NOT NULL,
                as_of_date       DATE,
                scoring_version  VARCHAR(40) NOT NULL,
                grading_version  VARCHAR(40) NOT NULL,
                classification   VARCHAR(30),
                breakout_score   NUMERIC,
                confidence       NUMERIC,
                grade            VARCHAR(10) NOT NULL,
                outcome_weeks    INTEGER[],
                outcome_games    INTEGER,
                baseline_source  VARCHAR(20),
                baseline_ppg     NUMERIC,
                outcome_ppg      NUMERIC,
                ppg_delta        NUMERIC,
                baseline_opp_pg  NUMERIC,
                outcome_opp_pg   NUMERIC,
                opp_delta        NUMERIC,
                baseline_snap_pct NUMERIC,
                outcome_snap_pct NUMERIC,
                snap_delta       NUMERIC,
                detail           JSONB,
                graded_at        TIMESTAMP DEFAULT NOW(),
                UNIQUE (player_id, season, as_of_week, scoring_version)
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_wbg_season_grade "
            f"ON {GRADES_TABLE} (season, grade)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_wbg_classification "
            f"ON {GRADES_TABLE} (classification)"
        )
    _INIT_DONE = True


def init_injury_snapshots_db() -> None:
    """Create the weekly injury-designation snapshot table if absent.
    Idempotent; safe to call on every run (acts as the migration for
    existing databases)."""
    global _SNAPSHOT_INIT_DONE
    if _SNAPSHOT_INIT_DONE:
        return
    with get_conn() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {INJURY_SNAPSHOTS_TABLE} (
                player_id     TEXT NOT NULL,
                season        INTEGER NOT NULL,
                week          INTEGER NOT NULL,
                injury_status TEXT,
                status        TEXT,
                captured_at   TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (player_id, season, week)
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_injsnap_season_week "
            f"ON {INJURY_SNAPSHOTS_TABLE} (season, week)"
        )
    _SNAPSHOT_INIT_DONE = True


def snapshot_injury_statuses(season: int, week: int) -> int:
    """Capture one row per player of the live Sleeper injury designations
    for the current NFL week. Upsert: the latest snapshot within a week
    wins (captured_at refreshes). Fail-soft by design - a missing feed
    or a DB error logs and returns 0, and must never break the pipeline
    that calls this. Returns the number of rows written."""
    init_injury_snapshots_db()
    try:
        from dashboard_services.api import get_nfl_players
        feed = get_nfl_players() or {}
    except Exception:
        logger.warning(
            "weekly grading: injury snapshot skipped (players feed unavailable)",
            exc_info=True)
        return 0
    rows = []
    for pid, p in feed.items():
        if not isinstance(p, dict):
            continue
        rows.append((
            str(pid),
            int(season),
            int(week),
            p.get("injury_status") or None,
            p.get("status") or None,
        ))
    if not rows:
        return 0
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    f"INSERT INTO {INJURY_SNAPSHOTS_TABLE} "
                    f"(player_id, season, week, injury_status, status) "
                    f"VALUES (%s, %s, %s, %s, %s) "
                    f"ON CONFLICT (player_id, season, week) DO UPDATE SET "
                    f"injury_status = EXCLUDED.injury_status, "
                    f"status = EXCLUDED.status, "
                    f"captured_at = NOW()",
                    rows,
                )
    except Exception:
        logger.warning(
            "weekly grading: injury snapshot write failed for season %s week %s",
            season, week, exc_info=True)
        return 0
    return len(rows)


# =============================================================================
# pure helpers
# =============================================================================

def _num(value: Any) -> Optional[float]:
    """Float or None. Mirrors the scorer's rule: missing stays missing,
    never a fabricated zero."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _mean(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def is_mature(call_week: int, through_week: int) -> bool:
    """A call matures once OUTCOME_WEEKS subsequent weeks exist in the data."""
    return int(through_week) - int(call_week) >= OUTCOME_WEEKS


def window_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-game realized stats over one window of weekly_metrics rows.

    A row's presence means the player was active with usage that week (the
    weekly builder drops inactive weeks), so ``games`` is simply the row
    count. Opportunity is carries + targets per game, matching the engine's
    definition. Snap share / PPG average over the games where they were
    recorded; a missing value stays missing rather than reading as zero.
    """
    rows = list(rows)
    opp_values: List[Optional[float]] = []
    for row in rows:
        carries = _num(row.get("carries"))
        targets = _num(row.get("targets"))
        if carries is None and targets is None:
            opp_values.append(None)
        else:
            opp_values.append((carries or 0.0) + (targets or 0.0))
    return {
        "games": len(rows),
        "ppr_ppg": _mean([_num(r.get("ppr_pts")) for r in rows]),
        "opp_pg": _mean(opp_values),
        "snap_pct": _mean([_num(r.get("snap_pct")) for r in rows]),
    }


def _parse_evidence(call: Dict[str, Any]) -> Dict[str, Any]:
    evidence = call.get("evidence")
    if isinstance(evidence, dict):
        return evidence
    if isinstance(evidence, str):
        try:
            parsed = json.loads(evidence)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def stored_baseline(call: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """The baseline the engine itself scored against, from the call's stored
    evidence. Fields are None when the engine did not store them (e.g. PPG
    for prior-season baselines, whose pseudo-row carries no fantasy points;
    opportunity for non-RB positions, whose stored signals have no
    carries+targets composite)."""
    evidence = _parse_evidence(call)
    fantasy = evidence.get("fantasy") or {}
    signals = evidence.get("signals") or {}
    snap_signal = signals.get("snap_share") or {}
    opp_signal = signals.get("carry_opportunity_pg") or {}
    return {
        "ppr_ppg": _num(fantasy.get("baseline_ppg")),
        "snap_pct": _num(snap_signal.get("baseline")),
        "opp_pg": _num(opp_signal.get("baseline")),
    }


def baseline_window_weeks(call: Dict[str, Any]) -> List[int]:
    """Weeks whose weekly rows reconstruct the call's baseline: the stored
    baseline_weeks when present, else the up-to-BASELINE_FALLBACK_WEEKS weeks
    immediately before the call week. Empty for prior-season baselines -
    current-season weeks are never a substitute for a prior-season baseline."""
    stored = call.get("baseline_weeks") or []
    weeks = [int(w) for w in stored if _num(w) is not None]
    if weeks:
        return weeks
    if call.get("baseline_source") == "prior_season":
        return []
    call_week = int(call.get("as_of_week") or 0)
    return [w for w in range(call_week - BASELINE_FALLBACK_WEEKS, call_week) if w >= 1]


def call_needs_prior_ppg(call: Dict[str, Any]) -> bool:
    """True when the call's baseline PPG can only come from prior-season
    rows: a prior-season-baseline call whose stored evidence carries no
    PPG (the engine's prior-season pseudo-row has no fantasy points).
    Bulk loaders use this to decide whether a prior-season series load
    is worth doing at all."""
    return (
        call.get("baseline_source") == "prior_season"
        and stored_baseline(call).get("ppr_ppg") is None
    )


def resolve_baseline(
    call: Dict[str, Any],
    weekly_rows: Sequence[Dict[str, Any]],
    prior_rows: Optional[Sequence[Dict[str, Any]]] = None,
    cohort_ppg: Optional[float] = None,
) -> Tuple[Dict[str, Optional[float]], str]:
    """(baseline stats, source) for a call: stored evidence per field first,
    computed-from-weekly-rows for any field the evidence lacks.

    One exception to the weekly-rows fallback: a prior-season-baseline
    call whose stored PPG is missing takes its baseline PPG from
    ``prior_rows`` (the player's prior-season weekly rows, when the
    caller supplies them) as that season's mean PPR PPG. Current-season
    weeks are never promoted into a prior-season baseline. The filled
    PPG registers as a computed field for source bookkeeping.

    A second fill, for initial-role calls with no baseline at all:
    ``cohort_ppg`` (the position's mean rookie-year PPG, see
    :func:`load_cohort_baselines`) supplies the PPG baseline and the
    source is ``cohort``. Role fields stay None: there is no role to
    hold relative to, so cohort baselines grade on production alone
    (see :func:`classify_outcome`).

    Source is ``stored`` when every resolved field came from the evidence,
    ``computed`` when every field came from weekly rows, ``mixed`` for a
    blend, and ``none`` when nothing could be resolved.
    """
    stored = stored_baseline(call)
    weeks = set(baseline_window_weeks(call))
    window_rows = [r for r in weekly_rows if _num(r.get("week")) is not None
                   and int(r["week"]) in weeks]
    computed = window_stats(window_rows) if window_rows else {
        "games": 0, "ppr_ppg": None, "opp_pg": None, "snap_pct": None,
    }
    prior_ppg: Optional[float] = None
    if prior_rows and call.get("baseline_source") == "prior_season":
        prior_ppg = window_stats(list(prior_rows)).get("ppr_ppg")
    baseline: Dict[str, Optional[float]] = {}
    origins: Set[str] = set()
    for field in ("ppr_ppg", "snap_pct", "opp_pg"):
        if stored.get(field) is not None:
            baseline[field] = stored[field]
            origins.add("stored")
        elif computed.get(field) is not None:
            baseline[field] = computed[field]
            origins.add("computed")
        elif field == "ppr_ppg" and prior_ppg is not None:
            baseline[field] = prior_ppg
            origins.add("computed")
        else:
            baseline[field] = None
    if not origins and cohort_ppg is not None \
            and call_needs_cohort_baseline(call):
        baseline["ppr_ppg"] = float(cohort_ppg)
        origins.add("cohort")
    if not origins:
        source = "none"
    elif origins == {"stored"}:
        source = "stored"
    elif origins == {"computed"}:
        source = "computed"
    elif origins == {"cohort"}:
        source = "cohort"
    else:
        source = "mixed"
    return baseline, source


def classify_outcome(
    baseline: Dict[str, Optional[float]],
    outcome: Dict[str, Any],
    cohort: bool = False,
    position: Optional[str] = None,
) -> Dict[str, Any]:
    """Classify one realized outcome. Pure; see the module docstring for the
    thresholds. ``baseline`` carries ppr_ppg / opp_pg / snap_pct (any may be
    None); ``outcome`` is a :func:`window_stats` dict.

    ``cohort`` marks a cohort baseline (an initial-role call measured
    against the typical rookie year at its position). There is no role
    to hold relative to, so the verdict is production-only: a hit at
    +PPG_HIT_DELTA over the cohort, a miss at -PPG_HIT_DELTA under it,
    partial in between. The startable floor does not apply to cohort
    grades: the cohort mean is already a position-typical level, so a
    rookie graded against it is not the near-zero-baseline case the
    floor exists for.

    ``position`` selects the HIT_PPG_FLOOR a non-cohort hit must clear
    (the outcome PPG itself, per game played); an unknown position
    takes HIT_PPG_FLOOR_DEFAULT."""
    if not outcome.get("games"):
        return {"grade": GRADE_UNGRADED, "role_state": "unknown",
                "reason": "no_outcome_games"}
    if cohort:
        ppg_delta = _delta(outcome.get("ppr_ppg"), baseline.get("ppr_ppg"))
        if ppg_delta is None:
            return {"grade": GRADE_UNGRADED, "role_state": "cohort",
                    "reason": "no_cohort_comparison",
                    "opp_delta": None, "snap_delta": None, "ppg_delta": None}
        if ppg_delta >= PPG_HIT_DELTA:
            grade, reason = GRADE_HIT, "production_above_cohort"
        elif ppg_delta <= PPG_MISS_DELTA:
            grade, reason = GRADE_MISS, "production_below_cohort"
        else:
            grade, reason = GRADE_PARTIAL, "production_near_cohort"
        if grade == GRADE_HIT and int(outcome.get("games") or 0) < HIT_MIN_GAMES:
            grade, reason = GRADE_PARTIAL, "single_game_sample"
        return {"grade": grade, "role_state": "cohort", "reason": reason,
                "opp_delta": None, "snap_delta": None, "ppg_delta": ppg_delta}
    opp_delta = _delta(outcome.get("opp_pg"), baseline.get("opp_pg"))
    snap_delta = _delta(outcome.get("snap_pct"), baseline.get("snap_pct"))
    ppg_delta = _delta(outcome.get("ppr_ppg"), baseline.get("ppr_ppg"))
    if opp_delta is None and snap_delta is None:
        return {"grade": GRADE_UNGRADED, "role_state": "unknown",
                "reason": "no_role_baseline",
                "opp_delta": None, "snap_delta": None, "ppg_delta": ppg_delta}

    # Role held has two legs. The touches leg stands on its own. The
    # snap leg is guarded: snaps only count as a held role when touches
    # did not fall, because a player can be on the field more while
    # being used less (or while a teammate's absence inflates his snaps).
    # The role state is still computed and recorded exactly as before -
    # the report header and receipts print it - but since v5 it no
    # longer decides the verdict outside the flat band.
    role_held = (
        (opp_delta is not None and opp_delta >= OPP_HELD_DELTA)
        or (snap_delta is not None and snap_delta >= SNAP_HELD_DELTA
            and (opp_delta is None or opp_delta >= 0))
    )
    role_reverted = (
        (opp_delta is None or opp_delta <= OPP_REVERTED_DELTA)
        and (snap_delta is None or snap_delta <= SNAP_REVERTED_DELTA)
    )
    if role_held:
        role_state = "held"
    elif role_reverted:
        role_state = "reverted"
    else:
        role_state = "mixed"
    production_rose = ppg_delta is not None and ppg_delta >= PPG_HIT_DELTA
    production_collapsed = ppg_delta is not None and ppg_delta <= PPG_MISS_DELTA
    outcome_ppg = _num(outcome.get("ppr_ppg"))
    if ppg_delta is None:
        # No measured production (a prior-season baseline whose
        # prior-season rows were not supplied): the pre-v5 role-only
        # verdicts, with their partial ceiling - a hit has always
        # required a measured production rise.
        if role_reverted:
            grade, reason = GRADE_MISS, "role_reverted_to_baseline"
        elif role_held:
            grade, reason = GRADE_PARTIAL, "role_held_without_production_rise"
        else:
            grade, reason = GRADE_PARTIAL, "role_partially_retained"
    elif production_collapsed:
        # Production fell off a cliff: a miss whatever the role did.
        # Keeping the snaps is not a breakout when the points disappear.
        grade, reason = GRADE_MISS, "production_collapsed"
    elif production_rose:
        if outcome_ppg is not None and outcome_ppg < hit_ppg_floor(position):
            # The rise is real but landed below a startable level: a
            # delta off a near-zero baseline is not a breakout.
            grade, reason = GRADE_PARTIAL, "production_rose_below_floor"
        elif role_held:
            grade, reason = GRADE_HIT, "role_held_and_production_rose"
        elif role_reverted:
            # The role went away but the player produced anyway, at a
            # startable level (efficiency, touchdowns): production
            # decides - that is a breakout, not the v4 partial.
            grade, reason = GRADE_HIT, "production_rose_despite_role_reverted"
        else:
            grade, reason = GRADE_HIT, "production_rose"
    elif role_reverted:
        # Flat production on a reverted role: inside the flat band the
        # role only names the partial - it is no longer a miss (the
        # v4 role_reverted_to_baseline miss is retired).
        grade, reason = GRADE_PARTIAL, "production_flat_role_reverted"
    elif role_held:
        grade, reason = GRADE_PARTIAL, "role_held_without_production_rise"
    else:
        grade, reason = GRADE_PARTIAL, "role_partially_retained"
    if grade == GRADE_HIT and int(outcome.get("games") or 0) < HIT_MIN_GAMES:
        # One game at hit pace is a promising start, not a proven
        # breakout: the verdict caps at partial until a second game
        # exists. Role state is preserved; only the grade is capped.
        grade, reason = GRADE_PARTIAL, "single_game_sample"
    return {"grade": grade, "role_state": role_state, "reason": reason,
            "opp_delta": opp_delta, "snap_delta": snap_delta, "ppg_delta": ppg_delta}


def _delta(outcome_value: Optional[float], baseline_value: Optional[float]) -> Optional[float]:
    if outcome_value is None or baseline_value is None:
        return None
    return float(outcome_value) - float(baseline_value)


# Injury designations, kept LOCAL on purpose: dashboard_services.injuries
# imports pandas, and this grading module must stay importable on the
# pandas-less CI shard. These mirror the canonical sets there
# (INJURY_STATUSES / INJURY_SHORT_STATUSES) - if that module ever adds a
# new season-voiding designation, mirror it here too.
# Designations that void an outcome window outright (any single week).
_SEVERE_INJURY_DESIGNATIONS = {"IR", "PUP", "NFI"}
# Designations that void only when they wipe most of the window.
_ABSENT_INJURY_DESIGNATIONS = {"OUT", "DOUBTFUL"}
_SHORT_INJURY_TO_LONG = {"Q": "QUESTIONABLE", "D": "DOUBTFUL",
                         "O": "OUT", "IR": "IR"}
_SHORT_INJURY_FORMS = set(_SHORT_INJURY_TO_LONG)


def canon_injury_status(value: Any) -> str:
    """Normalize a raw Sleeper designation to its canonical long form
    (upper-cased; short forms like 'Q'/'D'/'O' expanded). Unknown or
    non-injury values ('Active', '', None) normalize to ''."""
    if value is None:
        return ""
    v = str(value).strip().upper()
    if not v or v in ("ACTIVE", "ACT", "NONE", "-"):
        return ""
    if v in _SHORT_INJURY_FORMS:
        v = _SHORT_INJURY_TO_LONG.get(v, v)
    return v


def classify_injury_window(
    outcome_games: int,
    injury_by_week: Dict[int, Dict[str, Any]],
    outcome_weeks: Sequence[int],
) -> Dict[str, Any]:
    """Decide whether injury voids a call's outcome window. Pure.

    ``injury_by_week`` maps week -> {"injury_status": ..., "status": ...}
    from the weekly snapshots; weeks with no snapshot simply do not
    vote. Returns {"injured": bool, "reason": str|None,
    "designation": str|None, "weeks": [...]}.

    Rule (the 2-games rule): with >= 2 games played the call had its
    shot - it grades normally on those games regardless of designation.
    Below 2 games the window is voided ("injured") when any outcome week
    carries an IR / PUP / NFI designation, or when OUT / DOUBTFUL covers
    >= 2 of the 3 outcome weeks. QUESTIONABLE never voids.
    """
    result: Dict[str, Any] = {"injured": False, "reason": None,
                              "designation": None, "weeks": []}
    if int(outcome_games or 0) >= 2:
        return result
    severe = _SEVERE_INJURY_DESIGNATIONS
    absent = _ABSENT_INJURY_DESIGNATIONS
    out_weeks: List[int] = []
    out_designation: Optional[str] = None
    for week in outcome_weeks:
        snap = (injury_by_week or {}).get(int(week)) or {}
        for field in ("status", "injury_status"):
            designation = canon_injury_status(snap.get(field))
            if not designation:
                continue
            if designation in severe:
                result.update({
                    "injured": True,
                    "reason": "on_ir_during_window",
                    "designation": designation,
                    "weeks": [int(week)],
                })
                return result
            if designation in absent:
                if int(week) not in out_weeks:
                    out_weeks.append(int(week))
                if out_designation is None:
                    out_designation = designation
    if len(out_weeks) >= 2:
        result.update({
            "injured": True,
            "reason": "out_during_window",
            "designation": out_designation,
            "weeks": sorted(out_weeks),
        })
    return result


def grade_call(
    call: Dict[str, Any],
    weekly_rows: Sequence[Dict[str, Any]],
    prior_rows: Optional[Sequence[Dict[str, Any]]] = None,
    cohort_ppg: Optional[float] = None,
    injury_by_week: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Grade one persisted call against the player's weekly rows. Pure.

    ``weekly_rows`` is the player's full-season player_weekly_metrics series
    (any order); only the baseline window and the OUTCOME_WEEKS weeks after
    the call week are read. ``prior_rows`` is the player's prior-season
    series, used only to fill the baseline PPG of a prior-season-baseline
    call whose stored evidence has none (see :func:`resolve_baseline`);
    omitting it preserves the historical behavior exactly. ``cohort_ppg``
    is the position's mean rookie-year PPG, used only to fill the baseline
    of an initial-role call that has no baseline at all; omitting it also
    preserves the historical behavior exactly. ``injury_by_week`` maps
    week -> {"injury_status": ..., "status": ...} from the weekly injury
    snapshots; when injury voids the outcome window the grade is
    GRADE_INJURED (terminal) instead of a hit/partial/miss verdict.
    """
    call_week = int(call.get("as_of_week") or 0)
    outcome_weeks = [call_week + i for i in range(1, OUTCOME_WEEKS + 1)]
    outcome_week_set = set(outcome_weeks)
    rows = [r for r in weekly_rows if _num(r.get("week")) is not None]
    outcome_rows = [r for r in rows if int(r["week"]) in outcome_week_set]
    outcome = window_stats(outcome_rows)
    injury = classify_injury_window(outcome["games"],
                                    injury_by_week or {}, outcome_weeks)
    baseline, baseline_source = resolve_baseline(call, rows, prior_rows,
                                                 cohort_ppg)
    if injury["injured"]:
        verdict = {
            "grade": GRADE_INJURED,
            "role_state": "injured",
            "reason": injury["reason"],
            "opp_delta": None,
            "snap_delta": None,
            "ppg_delta": None,
            "injury": {
                "designation": injury["designation"],
                "weeks": injury["weeks"],
            },
        }
    else:
        verdict = classify_outcome(baseline, outcome,
                                   cohort=(baseline_source == "cohort"),
                                   position=call.get("position"))
    detail: Dict[str, Any] = {
        "role_state": verdict["role_state"],
        "reason": verdict["reason"],
    }
    if verdict.get("injury"):
        detail["injury"] = verdict["injury"]
    detail["thresholds"] = {
        "ppg_hit_delta": PPG_HIT_DELTA,
        "ppg_miss_delta": PPG_MISS_DELTA,
        "cohort_miss_delta": -PPG_HIT_DELTA,
        "opp_held_delta": OPP_HELD_DELTA,
        "snap_held_delta": SNAP_HELD_DELTA,
        "opp_reverted_delta": OPP_REVERTED_DELTA,
        "snap_reverted_delta": SNAP_REVERTED_DELTA,
        "hit_min_games": HIT_MIN_GAMES,
        "hit_ppg_floor": hit_ppg_floor(call.get("position")),
    }
    return {
        "player_id": str(call.get("player_id") or ""),
        "player_name": call.get("player_name"),
        "season": int(call.get("season") or 0),
        "as_of_week": call_week,
        "as_of_date": call.get("as_of_date"),
        "scoring_version": call.get("scoring_version"),
        "grading_version": GRADING_VERSION,
        "classification": call.get("classification"),
        "breakout_score": _num(call.get("breakout_score")),
        "confidence": _num(call.get("confidence")),
        "grade": verdict["grade"],
        "outcome_weeks": outcome_weeks,
        "outcome_games": outcome["games"],
        "baseline_source": baseline_source,
        "baseline_ppg": _round(baseline.get("ppr_ppg")),
        "outcome_ppg": _round(outcome.get("ppr_ppg")),
        "ppg_delta": _round(verdict.get("ppg_delta")),
        "baseline_opp_pg": _round(baseline.get("opp_pg")),
        "outcome_opp_pg": _round(outcome.get("opp_pg")),
        "opp_delta": _round(verdict.get("opp_delta")),
        "baseline_snap_pct": _round(baseline.get("snap_pct")),
        "outcome_snap_pct": _round(outcome.get("snap_pct")),
        "snap_delta": _round(verdict.get("snap_delta")),
        "detail": detail,
    }


def call_key(call: Dict[str, Any]) -> Tuple[str, int, int, str]:
    """The identity a grade is keyed by - the call's own unique key."""
    return (
        str(call.get("player_id") or ""),
        int(call.get("season") or 0),
        int(call.get("as_of_week") or 0),
        str(call.get("scoring_version") or ""),
    )


# =============================================================================
# summary (pure aggregation + DB accessor)
# =============================================================================

def _rate_bucket(rows: List[Dict[str, Any]], min_sample: int) -> Dict[str, Any]:
    counts = {GRADE_HIT: 0, GRADE_PARTIAL: 0, GRADE_MISS: 0,
              GRADE_UNGRADED: 0, GRADE_INJURED: 0}
    for row in rows:
        grade = str(row.get("grade") or "")
        if grade in counts:
            counts[grade] += 1
    graded = counts[GRADE_HIT] + counts[GRADE_PARTIAL] + counts[GRADE_MISS]
    enough = graded >= int(min_sample)

    def _rate(n: int) -> Optional[float]:
        if not enough or graded == 0:
            return None
        return round(n / graded, 4)

    # Forecast rate: blends finished grades with the live forecast of
    # open calls, using the same band the player card shows. Graded calls
    # count their actual outcome (hit=1, partial=0.5, miss=0); open
    # (ungraded) calls count their current band (tracking_to_hit=1,
    # borderline=0.5, tracking_to_miss=0), attached as ``forecast_value``
    # by the forecasts module. Rows with no grade and no band (no games
    # yet, no baseline, injured) contribute nothing.
    values = []
    for row in rows:
        grade = str(row.get("grade") or "")
        if grade == GRADE_HIT:
            values.append(1.0)
        elif grade == GRADE_PARTIAL:
            values.append(0.5)
        elif grade == GRADE_MISS:
            values.append(0.0)
        elif grade == GRADE_UNGRADED:
            fv = row.get("forecast_value")
            try:
                fv = float(fv) if fv is not None else None
            except (TypeError, ValueError):
                fv = None
            if fv is not None:
                values.append(fv)
    forecast_rate = (round(sum(values) / len(values), 4)
                     if values else None)

    return {
        "calls": len(rows),
        "graded": graded,
        "hit": counts[GRADE_HIT],
        "partial": counts[GRADE_PARTIAL],
        "miss": counts[GRADE_MISS],
        "ungraded": counts[GRADE_UNGRADED],
        "injured": counts[GRADE_INJURED],
        "hit_rate": _rate(counts[GRADE_HIT]),
        "partial_rate": _rate(counts[GRADE_PARTIAL]),
        "miss_rate": _rate(counts[GRADE_MISS]),
        "forecast_rate": forecast_rate,
    }


def _score(v: Any) -> Optional[float]:
    """Safe float conversion for breakout scores, None on missing/unparseable.

    A missing score means "no score to filter on" (old rows, test fixtures):
    callers treat None as passing the surfaced threshold.
    """
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def is_surfaced_row(row: Dict[str, Any]) -> bool:
    """True when a grade row represents a call that was surfaced to the
    board: not watchlist/monitored, and breakout score >= 30.

    A missing/unparseable score passes (old rows and test fixtures
    predate the threshold). This is the single shared definition of
    "surfaced" used by the sidebar Track Record and the Wednesday
    email report, so the two can never disagree on which calls count.
    """
    if display_classification(row.get("classification"),
                             row.get("breakout_score")) in ("watchlist",
                                                           "monitored"):
        return False
    s = _score(row.get("breakout_score"))
    return s is None or s >= 30


# The surfaced board floor, shared with the board in breakout_api
# (BREAKOUT_SURFACED_MIN_SCORE). Kept as a literal here so this module
# never imports the API layer.
SURFACED_MIN_SCORE = 30.0
SURFACED_REPEAT_WINDOW_WEEKS = 3


def _row_evidence(row: Dict[str, Any]) -> Dict[str, Any]:
    """A score row's evidence dict, parsed defensively (the DB returns
    JSON text in some drivers, a dict in others)."""
    evidence = (row or {}).get("evidence") or {}
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except (ValueError, TypeError):
            return {}
    return evidence if isinstance(evidence, dict) else {}


def _evidence_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def _is_pool_eligible_row(row: Dict[str, Any]) -> bool:
    """True when a serving-run score row belongs to the board pool the
    surfaced board is drawn from, mirroring
    ``get_weekly_breakout_candidates`` in breakout_api: breakout score
    at or above the surfaced floor, in the main-board-eligible or
    early-watch groups, and past the initial-role discovery gate."""
    score = _score((row or {}).get("breakout_score"))
    if score is None or score < SURFACED_MIN_SCORE:
        return False
    evidence = _row_evidence(row)
    if not (_evidence_truthy(evidence.get("main_board_eligible"))
            or str(row.get("classification") or "") == "early_watch"):
        return False
    if evidence.get("score_basis") == "initial_role":
        from data_building.breakout_engine.weekly_breakout import (
            INITIAL_ROLE_DISCOVERY_MIN,
        )
        subscores = evidence.get("subscores") or {}
        current_role = _score(subscores.get(
            "current_role_score", evidence.get("current_role_score")))
        if current_role is None or current_role < INITIAL_ROLE_DISCOVERY_MIN:
            return False
    return True


def surfaced_player_ids_by_week(season: int) -> Dict[int, Set[str]]:
    """{week: {player_id, ...}}: the players actually surfaced on the
    board each week, the one canonical surfaced definition.

    Weeks are walked ascending over completed weeks; only each week's
    SERVING run counts (a high score in some other completed run for
    the same week was never shown to anyone). A pool-eligible player
    (see :func:`_is_pool_eligible_row`) is surfaced in a week only if
    they were not surfaced in any of the previous
    ``SURFACED_REPEAT_WINDOW_WEEKS`` weeks, matching the board's
    no-repeat rule: a player surfaced in week W is not a new call again
    in weeks W+1 through W+3. Fails soft to ``{}`` on any error; an
    absent week means "could not determine", a present week with an
    empty set means nobody surfaced that week.
    """
    try:
        from data_building.breakout_engine import weekly_store

        completed = weekly_store.list_completed_weeks(int(season)) or []
        weeks = sorted({
            int(r["as_of_week"]) for r in completed
            if isinstance(r, dict) and r.get("as_of_week") is not None
        })
        out: Dict[int, Set[str]] = {}
        for week in weeks:
            run = weekly_store.get_serving_run(int(season), week)
            if not run or run.get("id") is None:
                out[week] = set()
                continue
            surfaced: Set[str] = set()
            for row in weekly_store.load_run_score_rows(run["id"]) or []:
                if not isinstance(row, dict) or not _is_pool_eligible_row(row):
                    continue
                pid = str(row.get("player_id") or "")
                if not pid:
                    continue
                if any(pid in out.get(w, ())
                       for w in range(week - SURFACED_REPEAT_WINDOW_WEEKS,
                                      week)):
                    continue
                surfaced.add(pid)
            out[week] = surfaced
        return out
    except Exception:
        logger.warning("weekly_grading: surfaced-by-week lookup failed",
                       exc_info=True)
        return {}


def summarize_grade_rows(
    rows: Sequence[Dict[str, Any]],
    min_sample: int = MIN_SUMMARY_SAMPLE,
) -> Dict[str, Any]:
    """Hit / partial / miss rates overall, by classification, and by scoring
    version. Rates are over graded calls only (hit + partial + miss;
    ungraded and injured calls are excluded from the denominator - injured
    calls are counted separately for transparency) and are None for any
    group with fewer than ``min_sample`` graded calls - counts are always
    real, never suppressed. Classification groups use the display label:
    stored "watchlist" calls scored under the watchlist floor aggregate as
    "monitored"."""
    rows = list(rows)
    by_classification: Dict[str, List[Dict[str, Any]]] = {}
    by_version: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_classification.setdefault(
            display_classification(row.get("classification"),
                                   row.get("breakout_score")), []).append(row)
        by_version.setdefault(str(row.get("scoring_version") or "unknown"), []).append(row)
    # Overall counts surfaced calls only: exclude watchlist/monitored, which
    # were never surfaced to the board. by_classification keeps them grouped
    # separately for transparency.
    surfaced_rows = [r for r in rows if is_surfaced_row(r)]
    return {
        "min_sample": int(min_sample),
        "overall": _rate_bucket(surfaced_rows, min_sample),
        "by_classification": {
            key: _rate_bucket(group, min_sample)
            for key, group in sorted(by_classification.items())
        },
        "by_scoring_version": {
            key: _rate_bucket(group, min_sample)
            for key, group in sorted(by_version.items())
        },
    }


def summarize_grades(
    season: Optional[int] = None,
    min_sample: int = MIN_SUMMARY_SAMPLE,
) -> Dict[str, Any]:
    """DB accessor over persisted grades; see :func:`summarize_grade_rows`."""
    init_weekly_breakout_grades_db()
    query = f"SELECT * FROM {GRADES_TABLE}"
    params: List[Any] = []
    if season is not None:
        query += " WHERE season = %s"
        params.append(int(season))
    with get_conn() as conn:
        rows = conn.execute(query, params).fetchall()
    return summarize_grade_rows([dict(r) for r in rows], min_sample=min_sample)


# =============================================================================
# DB layer: loaders, saver, orchestration
# =============================================================================

def load_calls(season: int) -> List[Dict[str, Any]]:
    """Every persisted weekly call for a season (one row per call).

    Only calls from the serving run of each week are loaded: the same
    run the board displays via ``get_serving_run`` (non-reconstructed
    runs win over reconstructed ones). This keeps the grader aligned
    with what was actually surfaced.
    """
    init_weekly_breakout_db()
    with get_conn() as conn:
        weeks = conn.execute(
            f"SELECT DISTINCT as_of_week FROM {WEEKLY_SCORES_TABLE} "
            f"WHERE season = %s ORDER BY as_of_week",
            (int(season),),
        ).fetchall()
        serving_run_ids = []
        for w in weeks:
            run = get_serving_run(int(season), int(w["as_of_week"]))
            if run and run.get("id") is not None:
                serving_run_ids.append(int(run["id"]))
        if not serving_run_ids:
            return []
        placeholders = ", ".join(["%s"] * len(serving_run_ids))
        rows = conn.execute(
            f"SELECT * FROM {WEEKLY_SCORES_TABLE} "
            f"WHERE run_id IN ({placeholders}) "
            f"ORDER BY as_of_week, player_id",
            tuple(serving_run_ids),
        ).fetchall()
    return [dict(r) for r in rows]


def load_existing_grade_keys(season: int) -> Set[Tuple[str, int, int, str]]:
    """Call keys that already have a real grade - the idempotency frontier.

    The frontier is per grading version: only rows graded under the
    current GRADING_VERSION count. A grade written by an older grading
    version is a stale verdict under old logic, so it does not block;
    the call is regraded and the saver overwrites the row in place.

    Ungraded rows are not grades (they record 'could not grade yet') and
    never block a future grading attempt under any version, so a call
    that was ungraded for lack of a baseline becomes gradeable once a
    baseline exists. Every other current-version grade (hit, partial,
    miss, injured) is terminal: it is written once and never retried."""
    init_weekly_breakout_grades_db()
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT player_id, season, as_of_week, scoring_version "
            f"FROM {GRADES_TABLE} WHERE season = %s AND grade <> %s "
            f"AND grading_version = %s",
            (int(season), GRADE_UNGRADED, GRADING_VERSION),
        ).fetchall()
    return {call_key(dict(r)) for r in rows}


def default_through_week(season: int) -> Optional[int]:
    """Latest week present in player_weekly_metrics for the season - the
    newest week outcomes can be measured through."""
    from data_building.weekly_metrics import init_weekly_metrics_db
    init_weekly_metrics_db()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT MAX(week) AS w FROM player_weekly_metrics WHERE season = %s",
            (int(season),),
        ).fetchone()
    return int(row["w"]) if row and row.get("w") is not None else None


def load_season_series(season: int, through_week: int) -> Dict[str, List[Dict[str, Any]]]:
    """player_id -> weekly_metrics rows through ``through_week``. The same
    loader the engine reads, so grading sees exactly the engine's source."""
    from data_building.weekly_metrics import get_weekly_series_by_player
    return get_weekly_series_by_player(int(season), int(through_week))


# Whole-season frontier for prior-season series loads: a prior-season
# baseline PPG is a full-season mean, so the load is never week-capped.
PRIOR_SEASON_THROUGH_WEEK = 18


def load_prior_season_series(season: int) -> Dict[str, List[Dict[str, Any]]]:
    """player_id -> every weekly_metrics row of the season BEFORE
    ``season``. Supplies the baseline PPG fill for prior-season-baseline
    calls (see :func:`resolve_baseline`). Bulk callers load this only
    when a call actually needs the fill; a failed or empty load degrades
    to no fill, never to a fabricated baseline."""
    try:
        return load_season_series(int(season) - 1, PRIOR_SEASON_THROUGH_WEEK)
    except Exception:
        logger.warning(
            "weekly grading: prior-season series load failed for season %s",
            season, exc_info=True)
        return {}


def call_needs_cohort_baseline(call: Dict[str, Any]) -> bool:
    """True when a cohort baseline is the only baseline available: an
    initial-role call (stored baseline_source 'none') whose evidence
    carries no PPG of its own. Bulk loaders use this to decide whether
    the cohort aggregate query is worth running at all."""
    return (
        call.get("baseline_source") in (None, "none")
        and stored_baseline(call).get("ppr_ppg") is None
    )


def load_cohort_baselines(season: int) -> Dict[str, float]:
    """position -> mean rookie-year PPR PPG over the COHORT_LOOKBACK_SEASONS
    seasons before ``season``.

    A rookie year is the player's debut season in player_weekly_metrics
    (first season with any weekly row); per-player PPG is the mean ppr_pts
    over that debut season, the same per-game definition window_stats
    uses for outcomes. Debut seasons with fewer than COHORT_MIN_GAMES
    games do not count, and a position publishes only with at least
    COHORT_MIN_PLAYERS qualifying player-seasons. Returns {} on any DB
    error: callers treat a missing position as 'no cohort baseline',
    never as a zero baseline."""
    season = int(season)
    start = season - COHORT_LOOKBACK_SEASONS
    try:
        from data_building.weekly_metrics import init_weekly_metrics_db
        init_weekly_metrics_db()
        with get_conn() as conn:
            rows = conn.execute(
                """
                WITH debut AS (
                    SELECT player_id, MIN(season) AS rookie_season
                    FROM player_weekly_metrics
                    GROUP BY player_id
                ),
                rookie_ppg AS (
                    SELECT MAX(m.position) AS position,
                           AVG(m.ppr_pts) AS ppg
                    FROM player_weekly_metrics m
                    JOIN debut d ON d.player_id = m.player_id
                                AND d.rookie_season = m.season
                    WHERE m.season BETWEEN %s AND %s
                      AND m.position IN ('QB', 'RB', 'WR', 'TE')
                    GROUP BY m.player_id, d.rookie_season
                    HAVING COUNT(*) >= %s
                )
                SELECT position, AVG(ppg) AS cohort_ppg
                FROM rookie_ppg
                GROUP BY position
                HAVING COUNT(*) >= %s
                """,
                (start, season - 1, COHORT_MIN_GAMES, COHORT_MIN_PLAYERS),
            ).fetchall()
    except Exception:
        logger.warning(
            "weekly grading: cohort baseline load failed for season %s",
            season, exc_info=True)
        return {}
    return {
        str(r["position"]): float(r["cohort_ppg"])
        for r in (rows or [])
        if r.get("position") and r.get("cohort_ppg") is not None
    }


def load_injury_weeks(
    season: int,
    player_ids: Sequence[str],
    weeks: Sequence[int],
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """player_id -> {week -> {"injury_status", "status"}} from the weekly
    injury snapshots. One bulk query for every pending call's outcome
    window. A failed or empty load degrades to {} (no injury evidence),
    never to a fabricated designation."""
    ids = sorted({str(p) for p in player_ids if p})
    wks = sorted({int(w) for w in weeks if _num(w) is not None})
    if not ids or not wks:
        return {}
    try:
        init_injury_snapshots_db()
        with get_conn() as conn:
            rows = conn.execute(
                f"SELECT player_id, week, injury_status, status "
                f"FROM {INJURY_SNAPSHOTS_TABLE} "
                f"WHERE season = %s AND player_id = ANY(%s) AND week = ANY(%s)",
                (int(season), ids, wks),
            ).fetchall()
    except Exception:
        logger.warning(
            "weekly grading: injury snapshot load failed for season %s",
            season, exc_info=True)
        return {}
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in (rows or []):
        d = dict(r)
        out.setdefault(str(d.get("player_id")), {})[int(d.get("week") or 0)] = {
            "injury_status": d.get("injury_status"),
            "status": d.get("status"),
        }
    return out


def save_grade_rows(grades: List[Dict[str, Any]]) -> int:
    """Insert grades idempotently, with a version-guarded overwrite.

    A grade is immutable history WITHIN its grading version: when the
    stored row for a call already carries the incoming grading_version,
    the conflict is a true no-op (no update, no returned id), so
    same-version reruns never rewrite a grade. When the stored row
    carries an OLDER grading_version, it is a stale verdict under old
    logic: the conflict updates every grade column in place from the
    incoming row, guarded by the version mismatch, so a grading-logic
    bump regrades history instead of being silently blocked by it.
    Returns the number of rows actually inserted or regraded."""
    if not grades:
        return 0
    init_weekly_breakout_grades_db()
    cols = [
        "player_id", "player_name", "season", "as_of_week", "as_of_date",
        "scoring_version", "grading_version", "classification",
        "breakout_score", "confidence", "grade", "outcome_weeks",
        "outcome_games", "baseline_source", "baseline_ppg", "outcome_ppg",
        "ppg_delta", "baseline_opp_pg", "outcome_opp_pg", "opp_delta",
        "baseline_snap_pct", "outcome_snap_pct", "snap_delta", "detail",
    ]
    placeholders = ", ".join(
        f"%({column})s::jsonb" if column == "detail" else f"%({column})s"
        for column in cols
    )
    key_cols = {"player_id", "season", "as_of_week", "scoring_version"}
    updates = ", ".join(
        f"{column} = EXCLUDED.{column}" for column in cols
        if column not in key_cols
    )
    inserted = 0
    with get_conn() as conn:
        for grade in grades:
            row = dict(grade)
            row["detail"] = json.dumps(grade.get("detail") or {})
            result = conn.execute(
                f"INSERT INTO {GRADES_TABLE} ({', '.join(cols)}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT (player_id, season, as_of_week, scoring_version) "
                f"DO UPDATE SET {updates} "
                f"WHERE {GRADES_TABLE}.grading_version <> EXCLUDED.grading_version "
                f"RETURNING id",
                row,
            ).fetchone()
            if result:
                inserted += 1
    return inserted


def grade_weekly_breakouts(
    season: int,
    *,
    through_week: Optional[int] = None,
) -> Dict[str, Any]:
    """Grade every matured, not-yet-graded call for a season.

    A call is graded when OUTCOME_WEEKS subsequent weeks exist in
    player_weekly_metrics (``through_week`` pins that frontier; by default
    the latest stored week). Already-graded calls are skipped before any
    math, so repeat runs are cheap no-ops. Returns a run summary dict.
    """
    init_weekly_breakout_db()
    init_weekly_breakout_grades_db()
    through = (int(through_week) if through_week is not None
               else default_through_week(season))
    summary: Dict[str, Any] = {
        "season": int(season),
        "through_week": through,
        "grading_version": GRADING_VERSION,
        "calls_considered": 0,
        "skipped_immature": 0,
        "already_graded": 0,
        "graded": 0,
        "inserted": 0,
        "by_grade": {},
    }
    # Injury-designation snapshot: grading needs each player's designation
    # DURING each outcome week, so capture this week's Sleeper designations
    # now, on every run (even when nothing is gradeable yet - future calls
    # need the history). Best-effort and fail-soft: the snapshot must never
    # break grading (or the scoring run that calls it).
    try:
        from dashboard_services.api import get_nfl_state
        nfl_state = get_nfl_state() or {}
        snap_week = int(nfl_state.get("week") or 0)
        if snap_week >= 1:
            summary["injury_snapshot"] = {
                "week": snap_week,
                "rows": snapshot_injury_statuses(season, snap_week),
            }
        else:
            summary["injury_snapshot"] = {"status": "skipped",
                                          "reason": "no nfl week in state"}
    except Exception as exc:
        summary["injury_snapshot"] = {"status": "skipped",
                                      "reason": str(exc)}
    if through is None:
        summary["status"] = "skipped"
        summary["reason"] = "no weekly metrics for season"
        return summary

    calls = load_calls(season)
    existing = load_existing_grade_keys(season)
    summary["calls_considered"] = len(calls)
    pending = []
    for call in calls:
        if call_key(call) in existing:
            summary["already_graded"] += 1
            continue
        if not is_mature(int(call.get("as_of_week") or 0), through):
            summary["skipped_immature"] += 1
            continue
        pending.append(call)

    if pending:
        series = load_season_series(season, through)
        # Prior-season series only when a pending call's baseline PPG can
        # come from nowhere else; most runs never pay for the extra load.
        prior_series: Dict[str, List[Dict[str, Any]]] = {}
        if any(call_needs_prior_ppg(call) for call in pending):
            prior_series = load_prior_season_series(season)
        # Cohort baselines only when a pending call has no baseline at
        # all; most runs never pay for the aggregate query.
        cohort: Dict[str, float] = {}
        if any(call_needs_cohort_baseline(call) for call in pending):
            cohort = load_cohort_baselines(season)
        # Injury snapshots for every pending call's outcome window, in one
        # bulk query; each call then gets its own player's week map.
        outcome_weeks_all = sorted({
            int(call.get("as_of_week") or 0) + i
            for call in pending for i in range(1, OUTCOME_WEEKS + 1)
        })
        injury_weeks = load_injury_weeks(
            season,
            [str(call.get("player_id") or "") for call in pending],
            outcome_weeks_all,
        )
        grades = [
            grade_call(
                call,
                series.get(str(call.get("player_id") or ""), []),
                prior_series.get(str(call.get("player_id") or "")),
                cohort.get(str(call.get("position") or "")),
                injury_by_week=injury_weeks.get(
                    str(call.get("player_id") or ""), {}),
            )
            for call in pending
        ]
        summary["inserted"] = save_grade_rows(grades)
        summary["graded"] = len(grades)
        for grade in grades:
            key = grade["grade"]
            summary["by_grade"][key] = summary["by_grade"].get(key, 0) + 1
    summary["status"] = "completed"
    return summary
