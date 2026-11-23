"""Rest-of-season (ROS) rankings: forward production for the rest of the year.

One computation, three consumers: the /players ROS mode (fields stamped on
the /api/league-players payload by app.py's overlay), and the public
/rankings/rest-of-season pages. Never a second opinion: every surface reads
the map this module builds.

Formula (backtested on the 2025 season, ~/workspace/ros-rankings/
BACKTEST_RESULTS.md; pooled MAE 2.374 vs 2.452 projection-only-everywhere
and 2.584 for the originally pitched 65/35 blend):

  * RB / WR / TE: ROS PPG = forward projection PPG alone. Sleeper's forward
    weekly projections already incorporate recent form, so blending form in
    again double-counts it over a rest-of-season window.
  * QB: ROS PPG = 40% forward projection PPG + 30% last-4-played form PPG
    + 30% season PPG. A thin form sample (< 4 played games in the window)
    shrinks the form weight toward the projection, the same shrinkage the
    start-score base blend uses.
  * ROS Pts = ROS PPG x expected games left. Scheduled games left counts
    the window weeks where the player's team plays (non-bye) and a
    projection feed exists, minus the leading zero-run of weeks out for an
    injured returner (the waiver path's weeks-out read on the projection
    series). Expected games then weights that count by the player's
    availability rate: how often he actually suits up (see
    availability_rate). A total built on scheduled games prices every
    player as if he plays every week, which systematically overstates
    fragile players against durable ones.
  * Two sub-windows ride along on the same weekly numbers: Stretch (window
    start through Week 14) and Playoffs (Weeks 15-17), each with its own
    PPG / total / games and positional rank. Weeks-out and bye handling
    apply inside each window independently.
  * Positional order in the full window is a factor model (Kaedon,
    2026-11-18), NOT the ROS Pts total: within each position, five
    factors are z-scored across the eligible pool and combined at
    position-specific weights (ROS_MODEL_WEIGHTS, the stripped base
    as of 2026-11-21: ROS PPG leads at .78 QB / .84 RB / .94 WR-TE;
    the remaining weight spreads over expected games, schedule,
    consistency and team offense). Form, ceiling and role trend are
    still computed for every player but carry no score weight: the
    v3 audit found all three cost accuracy at every position. Form is
    last-4 played PPG for QB and mean opportunity share over the
    last 4 played games for RB/WR/TE. The
    total-points order systematically buried better per-game players
    whose bye or expected games came up short. The model stamps
    ``ros_score`` and ``ros_pos_rank`` follows it; the Stretch and
    Playoff windows keep the total-points order. There is NO
    availability floor on the ranking: a returner priced at 2.5 expected
    games stays ranked, discounted by the games factor alone.

Eligibility (the playoff boards' gates): at least 3 games played this
season AND a real role: the team's #1 at his position (by points per
played game, so a back who missed weeks is not buried under a healthy
teammate's season total), or RB >= 12 touches/game, or WR/TE >= 5.5
targets/game, or QB >= 25 pass attempts/game. IR / PUP / NFI /
suspended players are excluded when the designation is a season
verdict, but a returner stays ranked (Kaedon, 2026-11-18: NO
availability floor on the ranking): when the feed carries no
season-ending note for him and his projection series is dark for at
least the first window week before real projections resume inside the
window, he flows through the model and his missed weeks price into
expected games through the same weeks-out machinery as everyone else
(see is_in_window_returner). A series that never resumes keeps the
exclusion, and so does a series that projects him from the first
window week: that contradicts the designation, and the designation
wins. Season-out is read from the live Sleeper status overlay at
build time, falling back to the players index: the index's stamped
status lags the feed. A player who fails any gate gets NO ROS fields,
so the board hides him in ROS mode rather than ranking him on a
number the model does not stand behind.

Schedule rank (the average Schedule Rankings matchup value of the
player's remaining opponents, ranked across teams, 1 = softest
remaining schedule) never enters the PPG: the weekly projections
already price the opponent in. It is stamped as the display value
``ros_sched_rank`` over the full window. The factor model's schedule
input is a separate rank built from the playoff weeks (15-17) only,
at each position's small schedule weight, with its z-score
contribution capped at +/-1.0: near-week opponents are already priced
into the projections, so only the far slate carries new information.

Cross-position order (the All view) is by VOR total: (window PPG minus
the replacement PPG at his position) x expected window games, with the
replacement convention in ros_replacement_map below. The All view
slot-fills that order (order_overall): VOR decides which slots each
position occupies, and each position's players fill their slots in
factor order, so within-position order on the All view always matches
the position views. The All view's tier dividers
are computed from those VOR totals (compute_ros_vor_tier_thresholds), NOT
from the positional totals: a mixed list keyed to raw totals stamps each
player with his own position's scale, so the dividers repeat and run out
of order (a QB whose raw total is elite-scale but whose VOR is mid-pack
reads T1 in the middle of the list).

1QB overall calibration (Kaedon, 2026-11-22): the slot-filled 1QB
order then passes through apply_qb_calibration, which enforces his
standing shape for the combined list: QB1 no higher than the 20-30
band (a quarterback the VOR order would place above it is set at
rank 24), quarterbacks slot as a cluster (each QB within ~6 spots of
the previous QB), and at most 5 QBs sit in the top 50. In 1QB a
quarterback's raw points edge overstates his overall rank because
every team starts exactly one and the streaming pool is deep; the
calibration is the owner's explicit placement rule for the overall
list, applied on top of the VOR slots, and it never touches the
positional lists or Superflex orders.

Display (Kaedon, 2026-11-22): ROS surfaces are a plain ranked list.
No ROS value and no ROS PPG number is displayed on any ROS surface
(the /players ROS mode or the public pages); the fields in this
module are internal ordering inputs only.

Scoring basis is standard PPR (rec = 1.0), the same basis the public pages
state and the backtest used. The compute reads only the cached files the
start-score bundle already loads (weekly stat files, weekly projection
files, schedules): it never fetches live data itself. ``load_ros_map``
serves a precomputed payload: the in-process signature cache first,
then the on-disk cache the last successful build wrote under the same
data signature (the dynasty/redraft pattern: values are precomputed,
reads are file reads), and only a signature miss runs the pool-wide
build, which then rewrites the disk cache. A read therefore never
pays the multi-second build more than once per data change across
workers and restarts.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import date
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ROS_POSITIONS = ("QB", "RB", "WR", "TE")
ROS_END_WEEK = 17
PLAYOFF_START_WEEK = 15
STRETCH_END_WEEK = 14

MIN_GAMES_PLAYED = 3
RB_MIN_TOUCHES_PER_GAME = 12.0
REC_MIN_TARGETS_PER_GAME = 5.5
QB_MIN_PASS_ATT_PER_GAME = 25.0

QB_W_PROJ = 0.40
QB_W_FORM = 0.30
QB_W_SEASON = 0.30
FORM_WINDOW = 4

# Availability (expected games): the rate is shrunk toward the positional
# mean with a prior of one full season of team games, so a player needs
# roughly a season of evidence before his own record moves him far off
# the positional norm, and clamped to [0.70, 1.00]: nobody is priced as
# missing more than 30% of his remaining games on availability alone
# (longer absences already come out through the weeks-out zero-run).
AVAILABILITY_PRIOR_GAMES = 17.0
AVAILABILITY_FLOOR = 0.70

PPR_SCORING = {"rec": 1.0}

# Season-ending / unavailable statuses. A plain "Out" is a weekly status,
# not a season verdict: those players stay eligible and their missed weeks
# come out of the games count through the projection zero-run instead.
SEASON_OUT_MARKERS = ("IR", "PUP", "NFI", "SUS")

# Season-ending language in the feed's injury notes. Sleeper's notes
# vocabulary is terse ("Surgery", "Sprain", "Fracture", ...): a surgery
# note is the feed's own season-ending signal, and it outranks the
# projection series, because the feed keeps projecting some surgery
# players at full volume anyway (verified live 2026-11-20: Davante
# Adams, hamstring surgery, still carried ~15 PPR projections for
# every remaining week). Matched as substrings so "Surgery" and any
# "... season ..." phrasing both land.
SEASON_ENDING_NOTE_MARKERS = ("SURGER", "SEASON")

# A weekly projection above this is a real projection; at or below it
# the week reads as a week out. The same threshold the waiver path's
# weeks-out read uses (utils.waivers.weeks_out_from_projections).
PROJECTION_REAL_THRESHOLD = 1.0


# ── Pure pieces (unit-tested directly) ──────────────────────────────────────

def is_season_out(injury_status) -> bool:
    status = str(injury_status or "").upper()
    return any(marker in status for marker in SEASON_OUT_MARKERS)


def has_season_ending_note(injury_notes) -> bool:
    notes = str(injury_notes or "").upper()
    return any(marker in notes for marker in SEASON_ENDING_NOTE_MARKERS)


def first_projected_week(series) -> Optional[int]:
    """Index of the first week in ``series`` carrying a real projection
    (above PROJECTION_REAL_THRESHOLD), or None when the feed never
    projects the player inside the window. ``series`` is the builder's
    per-week projection list (None = absent from that week's feed,
    ~0 = an explicit zero: both read as a week out)."""
    for i, value in enumerate(series or []):
        if value is None:
            continue
        try:
            points = float(value)
        except (TypeError, ValueError):
            continue
        if points > PROJECTION_REAL_THRESHOLD:
            return i
    return None


def is_in_window_returner(injury_status, injury_notes, all_series) -> bool:
    """True when a season-out-marked player is a returner, not a
    season-ender: the designation is real (a marker status), the feed
    carries no season-ending note for him, and his own projection
    series is dark for at least the first window week before real
    projections resume inside the window (the feed's timeline read,
    the same one the weeks-out machinery prices). The dark first week
    is required: a series projecting him from the first window week
    contradicts the designation, and the designation wins (the audit
    D2 shape, where the live IR tag is the freshest fact on the page).
    A series that never resumes is a season-ender the status already
    named. The returner then flows through the normal model: his
    expected games come out low on their own, discounted by the games
    factor alone, with no special-casing past any other gate."""
    if not is_season_out(injury_status):
        return False
    if has_season_ending_note(injury_notes):
        return False
    first = first_projected_week(all_series)
    return first is not None and first >= 1


def _season_out_verdict(injury_status, injury_notes, all_series) -> bool:
    """The builder's exclusion verdict for one player: a marker status
    excludes UNLESS the player is an in-window returner (Kaedon's
    2026-11-18 no-availability-floor ruling, restored 2026-11-20 after
    the blanket IR exclusion dropped Puka Nacua, a Week 15 returner,
    from every ROS board)."""
    if not is_season_out(injury_status):
        return False
    return not is_in_window_returner(injury_status, injury_notes, all_series)


def live_injury_status_map() -> Dict[str, str]:
    """{pid: injury_status} from the live Sleeper players feed.

    The players index's stamped injury_status lags the feed (the daily
    updater stamps it once a day, and the copy the web service reads can
    be older still): two Sleeper-listed IR players (Harvey, Love) stayed
    ROS-ranked in the 2026-11-20 audit because the index had not picked
    up their designations. app.py's ``_live_injury_status_map`` (PR
    #3190) reads the same raw feed behind ``get_players_global()``; the
    ROS build overlays it the same way, live status winning over the
    index value. Fail-soft: {} when the feed (or the app module, in
    offline/cron contexts) is unavailable, so callers degrade to the
    index value, never to a crash. Only players WITH a designation
    appear in the map.
    """
    try:
        from app import _live_injury_status_map
        return _live_injury_status_map() or {}
    except Exception:
        return {}


def live_injury_notes_map() -> Dict[str, str]:
    """{pid: injury_notes} from the same live Sleeper players feed.

    The returner read (is_in_window_returner) needs the feed's injury
    notes ("Surgery" is its season-ending signal); the raw player
    record behind ``get_players_global()`` carries them, the slim
    players index does not. Fail-soft: {} when the feed (or the app
    module, in offline/cron contexts) is unavailable, so callers
    degrade to the index value, never to a crash. Only players WITH
    notes appear in the map.
    """
    try:
        from app import get_players_global
        players = get_players_global() or {}
    except Exception:
        return {}
    out: Dict[str, str] = {}
    for pid, meta in players.items():
        if isinstance(meta, dict):
            notes = meta.get("injury_notes")
            if notes:
                out[str(pid)] = str(notes)
    return out


def qb_ros_ppg(fwd_ppg: float, form_ppg: Optional[float],
               form_games: int, season_ppg: Optional[float]) -> float:
    """QB ROS PPG: 40% forward projection + 30% last-4 form + 30% season.

    The form weight scales with the games actually in the form window
    (form_games / 4) and the released weight lands on the projection, so a
    2-game form sample carries half the form weight. A missing form or
    season line folds its weight onto the projection as well.
    """
    fwd = float(fwd_ppg or 0.0)
    w_form = QB_W_FORM * max(0, min(FORM_WINDOW, int(form_games or 0))) / FORM_WINDOW
    if form_ppg is None:
        w_form = 0.0
    w_season = QB_W_SEASON if season_ppg is not None else 0.0
    w_proj = 1.0 - w_form - w_season
    return (w_proj * fwd
            + w_form * float(form_ppg or 0.0)
            + w_season * float(season_ppg or 0.0))


def availability_rate(appearances: Optional[int], team_games: Optional[int],
                      pos_mean_rate: Optional[float]) -> Optional[float]:
    """Share of his team's games a player actually appears in, shrunk
    toward the positional mean.

    appearances / team_games, where the caller's team_games counts only
    the seasons in which the player actually appeared (a week counts as
    an appearance when the weekly stat feed marks the player active):
    a rookie is rated on his first season only, never against a prior
    season of team games he could not have played in. The raw share is
    shrunk with a prior of AVAILABILITY_PRIOR_GAMES team games at the
    positional mean rate, then clamped to [AVAILABILITY_FLOOR, 1.00].
    None when there is no data to rate (the caller then prices
    scheduled games unweighted).
    """
    if not team_games or appearances is None or pos_mean_rate is None:
        return None
    prior = AVAILABILITY_PRIOR_GAMES
    shrunk = (float(appearances) + prior * float(pos_mean_rate)) / (
        float(team_games) + prior)
    return max(AVAILABILITY_FLOOR, min(1.0, shrunk))


# ── Positional factor model (full window) ───────────────────────────────────
# Kaedon 2026-11-18: the full-window positional order is a weighted sum of
# within-position z-scores, not the ROS Pts total. PPG leads by design; the
# context factors (schedule, offense) are deliberately small, and expected
# games is a partial discount, never the multiplier the total made it.
# A factor a player lacks (thin game log, no schedule rank) fills with the
# pool mean, i.e. z = 0: missing evidence is neutral, never a penalty.
#
# v3 (Kaedon, 2026-11-20, "ship v3 as the new model"): the weights are
# position-specific, ported exactly from the offline backtest
# (~/workspace/ros-rankings/V3_BACKTEST_2026-11-20.md: v3 cell-mean
# Spearman +0.666 vs v2 +0.647 and PPG-only +0.665 on the 2025 sample).
# The v2 single weight set helped QB, tied at RB and lost at WR/TE, so
# QB keeps the fuller model, RB is simplified toward PPG, and WR/TE run
# near PPG-only. Two factor definitions changed with it: form for
# RB/WR/TE is opportunity share, not points (see player_model_factors),
# and the schedule factor reads the playoff weeks only, capped (see
# SCHED_Z_CAP).
#
# Stripped base (Kaedon, 2026-11-21, "I need the best ROS ranking
# model"): form, ceiling and role trend are REMOVED from the score at
# every position; the remaining v3 weights are renormalized (the values
# below are that renormalization, to 4 decimals). The v3 audit's
# component ablation (~/workspace/ros-rankings/ROS_V3_AUDIT_2026-11-21.md,
# Proposal 1) found all three factors cost accuracy at every position:
# dropping them grades cell-mean +0.677 vs v3 +0.666 on the 2025 sample
# and improves all four positions (QB +0.501 -> +0.531, RB +0.730 ->
# +0.740, WR +0.704 -> +0.712, TE +0.656 -> +0.669). A 1,728-config
# search over the surviving design space (ROS_BEST_MODEL_SEARCH_
# 2026-11-21.md) found nothing that beats this set out-of-sample. The
# three removed factors are still computed for every player (see
# player_model_factors) and remain available as display/context
# signals; they simply carry no score weight.
ROS_MODEL_WEIGHTS = {
    "QB": {"ppg": 0.7792, "games": 0.1169, "sched": 0.0390,
           "consistency": 0.0390, "offense": 0.0259},
    "RB": {"ppg": 0.8434, "games": 0.0964, "sched": 0.0241,
           "consistency": 0.0241, "offense": 0.0120},
    "WR": {"ppg": 0.9362, "games": 0.0426, "sched": 0.0106,
           "consistency": 0.0053, "offense": 0.0053},
    "TE": {"ppg": 0.9362, "games": 0.0426, "sched": 0.0106,
           "consistency": 0.0053, "offense": 0.0053},
}
# The schedule factor's z-score contribution is clipped to +/- this
# before weighting (v3): the factor reads Weeks 15-17 opponents only
# (the caller passes a playoff-weeks schedule rank), because near-week
# opponents are already baked into the weekly projections.
SCHED_Z_CAP = 1.0
CONSISTENCY_FLOORS = {"QB": 16.0, "RB": 10.0, "WR": 10.0, "TE": 7.0}
ROLE_TREND_WINDOW = 3
FORM_MIN_GAMES = 2
CEILING_GAMES = 3


def form_ppg(game_points) -> Optional[float]:
    """PPG over the last 4 played games; None under 2 played games.

    ``game_points`` is the player's played-game PPR points in week order
    (a played game he scored 0.0 in counts: it is a real game, not a
    missed one).
    """
    pts = [float(p) for p in (game_points or [])]
    if len(pts) < FORM_MIN_GAMES:
        return None
    window = pts[-FORM_WINDOW:]
    return sum(window) / len(window)


def ceiling_ppg(game_points) -> Optional[float]:
    """Mean of the player's 3 best played games; None under 3 played."""
    pts = sorted((float(p) for p in (game_points or [])), reverse=True)
    if len(pts) < CEILING_GAMES:
        return None
    return sum(pts[:CEILING_GAMES]) / CEILING_GAMES


def consistency_rate(game_points, position) -> Optional[float]:
    """Share of played games at or above the position's usable floor
    (QB 16, RB 10, WR 10, TE 7 PPR points); None with no played games."""
    pts = [float(p) for p in (game_points or [])]
    if not pts:
        return None
    floor = CONSISTENCY_FLOORS.get(str(position or "").upper())
    if floor is None:
        return None
    return sum(1 for p in pts if p >= floor) / len(pts)


def role_trend(series) -> Optional[float]:
    """Opportunity trend: recent-window mean minus prior-window mean.

    ``series`` is the per-played-game opportunity series in week order:
    target share in percent (WR/TE), rush+target share in percent (RB),
    or pass attempts per game (QB, a raw attempts delta, not points).
    The window is min(3, len // 2): 3v3 at 6+ played games, 2v2 at 4-5
    (a 5-game player's earliest game falls outside both windows). None
    under 4 played games: under two games a side there is no trend to
    read, and the factor stays neutral rather than guessed. (Before
    2026-11-20 this was all-or-nothing at 6 games, which neutralised
    the factor for every 5-game player: exactly the takeover profile
    whose trend is the story.)
    """
    vals = [float(v) for v in (series or [])]
    if len(vals) < 4:
        return None
    window = min(ROLE_TREND_WINDOW, len(vals) // 2)
    last = vals[-window:]
    prior = vals[-2 * window:-window]
    return sum(last) / len(last) - sum(prior) / len(prior)


def _zscores(values: Dict[str, Optional[float]]) -> Dict[str, float]:
    """Population z-scores across a pool; missing values score 0.0."""
    xs = [float(v) for v in values.values() if v is not None]
    if not xs:
        return {pid: 0.0 for pid in values}
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / len(xs)
    sd = math.sqrt(var) or 1.0
    return {pid: ((float(v) - mean) / sd if v is not None else 0.0)
            for pid, v in values.items()}


def opportunity_form(series) -> Optional[float]:
    """Mean opportunity share over the last 4 played games; None under 2.

    ``series`` is the per-played-game opportunity series in week order
    (see role_share_series): target share for WR/TE, rush+target share
    for RB. This is the v3 form factor for RB/WR/TE: the same series
    the role trend reads, taken as a level instead of a trend, so form
    tracks workload rather than points luck. QB keeps points form.
    """
    vals = [float(v) for v in (series or [])]
    if len(vals) < FORM_MIN_GAMES:
        return None
    window = vals[-FORM_WINDOW:]
    return sum(window) / len(window)


def ros_model_scores(
        factors_by_pid: Dict[str, Dict[str, Optional[float]]],
        position: str,
) -> Dict[str, float]:
    """Weighted z-score sum for one position's pool, at
    ROS_MODEL_WEIGHTS (the stripped base: ppg, games, sched,
    consistency, offense).

    ``factors_by_pid`` maps pid -> raw factor values (``sched``
    already inverted, so a softer slate is a larger number); only the
    factors named in ROS_MODEL_WEIGHTS are read, so the computed
    form / ceiling / role values ride along unscored. Every factor is
    z-scored across the pool exactly as given, so the caller passes
    one position's eligible pool. The schedule z-score is clipped to
    +/- SCHED_Z_CAP before weighting (playoff-weeks schedule only,
    capped).
    """
    weights = ROS_MODEL_WEIGHTS[str(position or "").upper()]
    per_factor = {
        f: _zscores({pid: (fac or {}).get(f)
                     for pid, fac in factors_by_pid.items()})
        for f in weights
    }

    def _score(pid):
        total = 0.0
        for f, w in weights.items():
            z = per_factor[f][pid]
            if f == "sched":
                z = max(-SCHED_Z_CAP, min(SCHED_Z_CAP, z))
            total += w * z
        return round(total, 3)

    return {pid: _score(pid) for pid in factors_by_pid}


def role_share_series(position: str, weeks, usage_by_week,
                      team_week_targets, team_week_rushes,
                      team: str) -> List[float]:
    """Per-played-game opportunity series for the role trend.

    WR/TE: the player's targets as a percent of his team's targets that
    week. RB: his rush attempts + targets as a percent of the team's
    rush attempts + targets. QB: his pass attempts (raw, no share). The
    denominators are the weekly stat feed summed over the player's
    current team: this pipeline has no play-by-play feed, so a traded
    player's early-season shares are read against his current team's
    totals (a documented deviation from the offline model's PBP
    posteam basis). A week with no team denominator reads 0.0.
    """
    pos = str(position or "").upper()
    out: List[float] = []
    for w in weeks:
        tgt, rush, att = (usage_by_week or {}).get(w, (0.0, 0.0, 0.0))
        if pos == "QB":
            out.append(float(att))
        elif pos in ("WR", "TE"):
            den = float((team_week_targets or {}).get((team, w)) or 0)
            out.append(100.0 * float(tgt) / den if den else 0.0)
        else:
            den = float((team_week_targets or {}).get((team, w)) or 0) \
                + float((team_week_rushes or {}).get((team, w)) or 0)
            out.append(100.0 * (float(tgt) + float(rush)) / den
                       if den else 0.0)
    return out


def player_model_factors(position: str, *, ppg, games, sched_rank,
                         game_points, role_series, offense) -> dict:
    """Raw factor values for one player: the full computed factor set.

    The score (ros_model_scores) reads only the ROS_MODEL_WEIGHTS
    subset; ``ceiling``, ``form`` and ``role`` are computed here and
    kept available as display/context signals but carry no score
    weight (stripped base, 2026-11-21).

    ``game_points`` maps week -> played-game PPR points; ``role_series``
    is the per-played-game opportunity series in week order (see
    role_share_series). ``sched_rank`` is the player's PLAYOFF-weeks
    (Weeks 15-17) schedule rank, not the full-window display rank: the
    schedule factor reads far weeks only. ``sched`` is inverted
    here: the softer slate (rank 1) becomes the larger number.
    """
    weeks = sorted((game_points or {}).keys())
    pts_series = [float((game_points or {})[w]) for w in weeks]
    # v3 form: QB keeps points form (last-4 played PPG); RB/WR/TE read
    # form as opportunity share (mean of the role series over the last
    # 4 played games), the backtested swap from points to workload.
    if str(position or "").upper() == "QB":
        form = form_ppg(pts_series)
    else:
        form = opportunity_form(role_series)
    return {
        "ppg": float(ppg) if ppg is not None else None,
        "ceiling": ceiling_ppg(pts_series),
        "games": float(games) if games is not None else None,
        "sched": (-float(sched_rank) if sched_rank is not None else None),
        "form": form,
        "role": role_trend(role_series),
        "consistency": consistency_rate(pts_series, position),
        "offense": float(offense) if offense is not None else None,
    }


def window_projection_stats(series: List[Optional[float]]) -> Optional[dict]:
    """Forward PPG / total / games for one window from a weekly series.

    ``series`` is the player's projected points for the window's non-bye
    weeks in order, one entry per week whose projection feed exists; None
    means the player is absent from that week's feed. The leading run of
    explicit ~zero weeks is weeks out for an injured returner: those weeks
    leave the count entirely. A MISSING week is not proof of an absence
    (the waiver path's safer semantics, treat_missing_as_out=False): it
    stops the zero-run rather than extending it, so a feed that simply
    has not loaded a player yet cannot fabricate an injury timeline.
    Absent weeks after the return are skipped, not scored as zero games
    (the waiver forward-PPG read: only weeks where the player actually
    has a projection count). Returns None when no playable week remains.
    """
    from utils.waivers import weeks_out_from_projections

    if not series:
        return None
    weeks_out = weeks_out_from_projections(series, treat_missing_as_out=False)
    remaining = list(series[weeks_out:])
    present = [max(0.0, float(v)) for v in remaining if v is not None]
    if not present:
        return None
    games = len(present)
    total = sum(present)
    return {"fwd_ppg": total / games, "total": total, "games": games,
            "weeks_out": weeks_out}


def passes_role_gate(position: str, *, is_team_top: bool,
                     touches_per_game: Optional[float],
                     targets_per_game: Optional[float],
                     pass_attempts_per_game: Optional[float] = None) -> bool:
    """The playoff boards' role gate: team #1 at the position, or a real
    workload (RB >= ~12 touches/game, WR/TE >= ~5.5 targets/game,
    QB >= ~25 pass attempts/game). The QB attempts fallback exists
    because a midseason takeover may not yet lead his room on a per-game
    basis either (his early relief appearances drag the rate); a backup's
    projection spike is still not a role, because a backup does not
    attempt 25 passes a game."""
    if is_team_top:
        return True
    pos = str(position or "").upper()
    if pos == "RB":
        return touches_per_game is not None and touches_per_game >= RB_MIN_TOUCHES_PER_GAME
    if pos in ("WR", "TE"):
        return targets_per_game is not None and targets_per_game >= REC_MIN_TARGETS_PER_GAME
    if pos == "QB":
        return (pass_attempts_per_game is not None
                and pass_attempts_per_game >= QB_MIN_PASS_ATT_PER_GAME)
    return False


def compute_player_ros(position: str, *, played_games: int,
                       season_ppg: Optional[float], last4_ppg: Optional[float],
                       last4_games: int, touches_per_game: Optional[float],
                       targets_per_game: Optional[float], is_team_top: bool,
                       season_out: bool,
                       pass_attempts_per_game: Optional[float] = None,
                       windows: Dict[str, List[Optional[float]]],
                       sched_rank: Optional[int],
                       availability_rate: Optional[float] = None) -> Optional[dict]:
    """ROS fields for one player, or None when he is not ROS-eligible.

    ``windows`` maps "all" / "stretch" / "playoffs" to that window's weekly
    projection series (see window_projection_stats). Positional ranks are
    NOT set here: assign_ros_ranks fills them once the whole pool exists.
    A sub-window with no playable week left simply carries no fields.

    ``availability_rate`` (see the module-level helper of the same name)
    converts each window's scheduled games into expected games; the
    stamped *_games fields ARE the expected games the totals are built
    on, so the number the board displays is the number the total uses.
    """
    pos = str(position or "").upper()
    if pos not in ROS_POSITIONS or season_out:
        return None
    if int(played_games or 0) < MIN_GAMES_PLAYED:
        return None
    if not passes_role_gate(pos, is_team_top=is_team_top,
                            touches_per_game=touches_per_game,
                            targets_per_game=targets_per_game,
                            pass_attempts_per_game=pass_attempts_per_game):
        return None
    all_stats = window_projection_stats(windows.get("all") or [])
    if not all_stats or all_stats["total"] <= 0:
        return None

    out: dict = {}
    for window_key, prefix in (("all", "ros"), ("stretch", "ros_stretch"),
                               ("playoffs", "ros_playoff")):
        stats = all_stats if window_key == "all" else window_projection_stats(
            windows.get(window_key) or [])
        if not stats or stats["games"] <= 0:
            continue
        if pos == "QB":
            ppg = qb_ros_ppg(stats["fwd_ppg"], last4_ppg, last4_games, season_ppg)
        else:
            ppg = stats["fwd_ppg"]
        if ppg <= 0:
            continue
        exp_games = stats["games"] * (
            availability_rate if availability_rate is not None else 1.0)
        out[f"{prefix}_ppg"] = round(ppg, 2)
        out[f"{prefix}_total"] = round(ppg * exp_games, 1)
        out[f"{prefix}_games"] = (
            int(exp_games) if float(exp_games).is_integer()
            else round(exp_games, 1))
    if "ros_ppg" not in out:
        return None
    if sched_rank is not None:
        out["ros_sched_rank"] = int(sched_rank)
    return out


def assign_ros_ranks(ros_by_pid: Dict[str, dict],
                     positions_by_pid: Dict[str, str],
                     model_scores: Optional[Dict[str, float]] = None) -> None:
    """Positional ranks per window. Mutates ``ros_by_pid`` in place.

    The sub-windows order by that window's total (ties broken by PPG,
    then player id for stability). The full window orders by the factor
    model when ``model_scores`` is given (see ros_model_scores): score
    desc, PPG desc, player id; the score is stamped as ``ros_score``.
    A full-window player without a model score (the loader scores every
    eligible player, so this is a partial-pool fallback) sorts behind
    the scored players by PPG.
    """
    for prefix in ("ros", "ros_stretch", "ros_playoff"):
        total_key, ppg_key, rank_key = (
            f"{prefix}_total", f"{prefix}_ppg", f"{prefix}_pos_rank")
        by_pos: Dict[str, List[Tuple[str, dict]]] = {}
        for pid, fields in ros_by_pid.items():
            if total_key not in fields:
                continue
            pos = str(positions_by_pid.get(pid) or "").upper()
            by_pos.setdefault(pos, []).append((pid, fields))
        for rows in by_pos.values():
            if prefix == "ros" and model_scores is not None:
                rows.sort(key=lambda kv: (
                    0 if kv[0] in model_scores else 1,
                    -model_scores.get(kv[0], 0.0),
                    -kv[1][ppg_key], kv[0]))
            else:
                rows.sort(key=lambda kv: (-kv[1][total_key], -kv[1][ppg_key], kv[0]))
            for i, (_pid, fields) in enumerate(rows, 1):
                if prefix == "ros" and model_scores is not None \
                        and _pid in model_scores:
                    fields["ros_score"] = model_scores[_pid]
                fields[rank_key] = i


def _ros_gap_tier_thresholds(values: List[float]) -> list:
    """Tier breaks from the drop-offs in THIS ordered ROS list itself.

    Kaedon (2026-11-22): ROS tiers must be based on the ROS rankings.
    The breaks are the largest gaps between consecutive players in the
    list being tiered (the All view's VOR order for the current format /
    size, a position view's ROS score/total order), where a gap counts
    as a drop-off when it is at least 2.5x the list's own median gap.
    Nothing is borrowed from the dynasty value tiering: no dynasty span
    cap, no dynasty fallback thresholds, no rescaling onto the dynasty
    range. Breaks are chosen largest-first with at least 3 players per
    tier, at most 8 breaks (T1..T9). A list with no real drop-off (a
    perfectly smooth ladder) returns [] — no dividers — rather than
    manufactured breaks.
    """
    vals = sorted((float(v) for v in values if v is not None), reverse=True)
    n = len(vals)
    if n < 6 or vals[0] <= 0:
        return []
    gaps = [vals[i] - vals[i + 1] for i in range(n - 1)]
    pos_gaps = sorted(g for g in gaps if g > 0)
    if not pos_gaps:
        return []
    median_gap = pos_gaps[len(pos_gaps) // 2]
    if median_gap <= 0:
        return []
    cands = [i for i, g in enumerate(gaps) if g >= 2.5 * median_gap]
    cands.sort(key=lambda i: -gaps[i])
    bounds: List[int] = []
    for i in cands:
        if len(bounds) >= 8:
            break
        lo = max([b for b in bounds if b < i], default=-1)
        hi = min([b for b in bounds if b > i], default=n - 1)
        if i - lo >= 3 and hi - i >= 3:
            bounds.append(i)
    if not bounds:
        return []
    return [round((vals[b] + vals[b + 1]) / 2.0, 1) for b in sorted(bounds)]


def _ros_tier_thresholds_for_values(rows: List[Tuple[Optional[str], float]]) -> list:
    """Tier thresholds for one ROS list of (position, value) rows.

    The values are the ROS numbers the list itself orders by (VOR totals
    for the All view, ROS scores/totals for position views, per window /
    format / size), and the breaks come straight from that list's own
    gaps via _ros_gap_tier_thresholds.
    """
    return _ros_gap_tier_thresholds([v for _pos, v in rows])


def compute_ros_tier_thresholds(ros_by_pid: Dict[str, dict],
                                positions_by_pid: Dict[str, str]) -> dict:
    """Tier thresholds from the numbers each window's board sorts by.

    Follows the redraft pattern (PR #3031): thresholds are computed on the
    same numbers the board sorts by, never the dynasty scale. The full
    window's position views order by the factor model, so its thresholds
    are computed on ``ros_score`` when the pool carries scores; the
    sub-windows keep their totals.
    Returns {"all": [...], "stretch": [...], "playoffs": [...]}.
    """
    out: dict = {}
    for window_key, prefix in (("all", "ros"), ("stretch", "ros_stretch"),
                               ("playoffs", "ros_playoff")):
        if window_key == "all" and any(
                "ros_score" in fields for fields in ros_by_pid.values()):
            rows = [(positions_by_pid.get(pid), fields["ros_score"])
                    for pid, fields in ros_by_pid.items()
                    if "ros_score" in fields]
        else:
            total_key = f"{prefix}_total"
            rows = [(positions_by_pid.get(pid), fields[total_key])
                    for pid, fields in ros_by_pid.items() if total_key in fields]
        out[window_key] = _ros_tier_thresholds_for_values(rows)
    return out


# ── Cross-position VOR: replacement convention + All-view tiers ─────────────
# One convention, read by every ROS surface: the public overall page calls
# ros_replacement_map directly, the payload's All-view tier thresholds are
# computed from ros_vor_totals, and the /players board mirrors the same
# math in JS (prRosReplacementMap in static/rankings.js). Every line is
# derived from the ROS pool itself; no constant is a hardcoded PPG.
#
# QB in 1QB is priced against STREAMING REPLACEMENT. The realistic
# alternative to holding an elite QB in a 1QB league is not one fixed
# fringe starter, and it is not the average streamer either: a manager
# who streams plays the BEST available quarterback each week. The
# replacement line is therefore DERIVED from the pool at build time:
# for each remaining week, the highest weekly projection among the
# quarterbacks below the starter line (QB{size+1} and down), averaged
# across the window — literally what streaming gives you. A QB's value
# is his edge over that free fallback, and no part of the line is
# fitted to where quarterbacks land on the board: an earlier
# construction priced the fallback off the starters' own production
# (a discounted mean of the starter pool's top half) with the discount
# factor calibrated to an output band, which manufactured the
# placements it claimed to measure; it was removed 2026-11-21. The
# static tier mean that replaced it still understated streaming,
# because nobody plays the average waiver quarterback when a better
# matchup is available that week; it was replaced by the weekly-best
# line 2026-11-21. Where a code path carries no weekly series, the
# fallback is the mean of the top two of the six quarterbacks below
# the starter line — a best-of approximation of the same tier.
# The dynasty model's 1.5x index also misprices this: it puts the line
# deep in the QB2 band because it is calibrated for a normalized
# strength scale, not raw points. SF keeps the 1.5x index on two
# starters per team: with two starters, QB scarcity is real and priced.
#
# The replacement math alone still slots QB1 inside the top 10 on
# the current pool shape (an outlier per-game edge over the streaming
# line outweighs every flex player's edge), which Kaedon rejected
# (2026-11-22): the 1QB Overall list therefore passes through the
# explicit calibration in apply_qb_calibration below (QB1 in the
# 20-30 band, QBs clustered within ~6 spots, max 5 QBs in the top
# 50). The positional QB list is never calibrated.
#
# RB / WR / TE share ONE replacement pool, the way lineups actually work:
# dedicated starters first (the value model's STARTERS in
# data_building/player_value.py), then the FLEX slots (the site models
# one FLEX per team; utils/trade.py's VORP section) are filled from the
# best remaining RB/WR/TE combined, and each position's replacement is
# the best remaining player at that position. This replaces the old
# per-position multiplier indices (starters x size x 1.20 / 1.25), which
# approximated the flex with a fixed depth markup on each position
# separately, blind to how the combined pool actually orders; the VORP
# helpers' fixed 45/45/10 flex usage split makes the same approximation
# by assumed shares rather than the pool's own PPG order.
ROS_STARTERS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
ROS_QB_SF_MULT = 1.50
ROS_FLEX_SLOTS_PER_TEAM = 1
ROS_QB_STREAM_TIER = 6
ROS_QB_STREAM_FALLBACK_TOP_N = 2
ROS_LEAGUE_TYPES = ("1qb", "sf")
ROS_LEAGUE_SIZES = (8, 10, 12, 14)
_WINDOW_PREFIXES = (("all", "ros"), ("stretch", "ros_stretch"),
                    ("playoffs", "ros_playoff"))


def _ros_qb_stream_fallback_ppg(ppgs_desc: List[float],
                                size: int) -> float:
    """Best-of fallback for the 1QB streaming line, used when a code
    path carries no weekly projection series: the mean of the top
    ROS_QB_STREAM_FALLBACK_TOP_N of the ROS_QB_STREAM_TIER
    quarterbacks below the starter line. When the pool runs out below
    the line the tier is whatever remains; when nobody sits below the
    line, the fallback is the worst starter himself (the pool's last
    PPG)."""
    tier = [float(v) for v in ppgs_desc[size:size + ROS_QB_STREAM_TIER]]
    if not tier:
        return float(ppgs_desc[-1])
    top = tier[:ROS_QB_STREAM_FALLBACK_TOP_N]
    return sum(top) / len(top)


def ros_qb_replacement_ppg(ppgs_desc: List[float], league_size: int,
                           league_type: str,
                           weekly_series_desc=None) -> Optional[float]:
    """Replacement QB PPG from the QB pool's ROS PPGs sorted desc.

    1QB: the STREAMING replacement. ``weekly_series_desc``, when
    given, is the pool's weekly projection series in the same
    PPG-descending order as ``ppgs_desc`` (one aligned series per
    quarterback: one entry per window week, None for a bye or a week
    absent from the feed). For each week the line takes the best
    projection among the quarterbacks below the starter line
    (QB{size+1} and down) and averages those weekly bests across the
    window: a streaming manager plays the best available matchup,
    never the tier's average member. When no usable weekly series is
    available — the series argument is absent, a below-line
    quarterback carries none, or no week carries a value — the line
    falls back to _ros_qb_stream_fallback_ppg. SF: the player at the
    scarcity index (2 starters x size x 1.5), clamped to the pool;
    the weekly series never enters it. None when the pool is empty.
    """
    if not ppgs_desc:
        return None
    size = int(league_size or 10)
    if str(league_type or "1qb").lower() == "sf":
        idx = int(2 * size * ROS_QB_SF_MULT)
        idx = max(0, min(idx, len(ppgs_desc) - 1))
        return float(ppgs_desc[idx])
    if weekly_series_desc is not None:
        below = list(weekly_series_desc)[size:]
        if below and all(isinstance(s, (list, tuple)) for s in below):
            bests: List[float] = []
            for week_idx in range(max((len(s) for s in below), default=0)):
                vals: List[float] = []
                for series in below:
                    if week_idx >= len(series):
                        continue
                    value = series[week_idx]
                    if value is None:
                        continue
                    try:
                        vals.append(float(value))
                    except (TypeError, ValueError):
                        continue
                if vals:
                    bests.append(max(vals))
            if bests:
                return sum(bests) / len(bests)
    return _ros_qb_stream_fallback_ppg(ppgs_desc, size)


def ros_flex_replacement_ppgs(by_pos_ppgs: Dict[str, List[float]],
                              league_size: int) -> Dict[str, float]:
    """Flex-aware replacement PPG for RB / WR / TE from one shared pool.

    ``by_pos_ppgs`` maps position -> ROS PPGs (any order). Dedicated
    starters per team (ROS_STARTERS) are allocated first, then
    ROS_FLEX_SLOTS_PER_TEAM flex spots per team go to the best remaining
    players across RB/WR/TE combined (PPG desc, position name breaking
    ties deterministically), and each position's replacement PPG is the
    best remaining player at that position, clamped to the pool's last
    player when a position is fully allocated.
    """
    size = int(league_size or 10)
    sorted_by_pos: Dict[str, List[float]] = {}
    taken: Dict[str, int] = {}
    leftovers: List[Tuple[float, str]] = []
    for pos in ("RB", "WR", "TE"):
        ppgs = sorted((float(v) for v in (by_pos_ppgs.get(pos) or [])),
                      reverse=True)
        sorted_by_pos[pos] = ppgs
        dedicated = min(ROS_STARTERS[pos] * size, len(ppgs))
        taken[pos] = dedicated
        for v in ppgs[dedicated:]:
            leftovers.append((v, pos))
    leftovers.sort(key=lambda t: (-t[0], t[1]))
    for _ppg, pos in leftovers[: ROS_FLEX_SLOTS_PER_TEAM * size]:
        taken[pos] += 1
    out: Dict[str, float] = {}
    for pos in ("RB", "WR", "TE"):
        ppgs = sorted_by_pos[pos]
        if ppgs:
            out[pos] = float(ppgs[min(taken[pos], len(ppgs) - 1)])
    return out


def ros_replacement_map(by_pos_ppgs: Dict[str, List[float]], league_size: int,
                        league_type: str,
                        qb_weekly_series_desc=None) -> Dict[str, float]:
    """Replacement PPG per position for one league shape: the streaming /
    scarcity QB line plus the flex-aware RB/WR/TE lines. Only positions
    present in the pool appear in the result.

    ``qb_weekly_series_desc`` is the QB pool's weekly projection
    series sorted by window PPG descending, parallel to the sorted
    QB PPGs (see ros_qb_replacement_ppg); omit it where the caller
    carries no weekly series and the QB line uses the best-of
    fallback."""
    out: Dict[str, float] = {}
    qb_ppgs = sorted((float(v) for v in (by_pos_ppgs.get("QB") or [])),
                     reverse=True)
    qb_line = ros_qb_replacement_ppg(
        qb_ppgs, league_size, league_type,
        weekly_series_desc=qb_weekly_series_desc)
    if qb_line is not None:
        out["QB"] = qb_line
    out.update(ros_flex_replacement_ppgs(by_pos_ppgs, league_size))
    return out


def ros_vor_totals(ros_by_pid: Dict[str, dict],
                   positions_by_pid: Dict[str, str],
                   league_size: int, league_type: str) -> Dict[str, Dict[str, float]]:
    """{window: {pid: VOR total}} for one league shape.

    VOR total = (window PPG - replacement PPG at his position) x window
    games: the All view's cross-position ordering metric.
    """
    out: Dict[str, Dict[str, float]] = {}
    for window_key, prefix in _WINDOW_PREFIXES:
        ppg_key, games_key = f"{prefix}_ppg", f"{prefix}_games"
        by_pos: Dict[str, List[float]] = {}
        qb_rows: List[Tuple[float, object]] = []
        for pid, fields in ros_by_pid.items():
            if ppg_key in fields:
                pos = str(positions_by_pid.get(pid) or "").upper()
                by_pos.setdefault(pos, []).append(float(fields[ppg_key]))
                if pos == "QB":
                    qb_rows.append((float(fields[ppg_key]),
                                    fields.get(f"{prefix}_weekly")))
        qb_rows.sort(key=lambda kv: -kv[0])
        repl = ros_replacement_map(
            by_pos, league_size, league_type,
            qb_weekly_series_desc=[s for _ppg, s in qb_rows])
        window_vor: Dict[str, float] = {}
        for pid, fields in ros_by_pid.items():
            if ppg_key not in fields or games_key not in fields:
                continue
            base = repl.get(str(positions_by_pid.get(pid) or "").upper())
            if base is None:
                continue
            window_vor[pid] = round(
                (float(fields[ppg_key]) - base) * float(fields[games_key]), 1)
        out[window_key] = window_vor
    return out


def _apply_position_monotonicity(rows: List[dict]) -> List[dict]:
    """Refill each position's Overall slots in positional-rank order.

    The slot-fill fills a position's slots in factor order, which is
    the order ros_pos_rank follows — except where the two orderings'
    tiebreaks part ways: an exact factor-score tie is broken by PPG
    then player id for the rank, but by PPG then NAME in the fill, so
    two tied players whose names and ids order differently swap on
    the Overall board (and the /players board's fill breaks the same
    tie by ROS total instead, which put WR19 ahead of WR18 on the
    Week 11 pool). This final pass makes the invariant structural
    instead of coincidental: for each position, the slots it occupies
    (ascending) are refilled by its players sorted by ros_pos_rank.
    Slot sets never change, so the VOR allocation is preserved
    exactly, and wherever the fill already
    agrees with the ranks the pass is a no-op. A position whose rows
    do not all carry ros_pos_rank keeps its fill order (a partial
    pool has no rank order to enforce). The /players board mirrors
    this in JS (prRosApplyPositionMonotonicity in
    static/rankings.js), keyed to the active window's rank there.
    """
    out = list(rows)
    slots: Dict[str, List[int]] = {}
    for i, r in enumerate(out):
        slots.setdefault(str(r.get("position") or "").upper(), []).append(i)
    for idxs in slots.values():
        members = [out[i] for i in idxs]
        if any(m.get("ros_pos_rank") is None for m in members):
            continue
        ranked = sorted(members, key=lambda m: float(m["ros_pos_rank"]))
        for i, m in zip(idxs, ranked):
            out[i] = m
    return out


# ── 1QB overall QB calibration (Kaedon, 2026-11-22) ────────────────────────
# Standing shape for the 1QB Overall list: QB1 lands in the 20-30
# band, quarterbacks slot as a cluster (each within ~6 spots of the
# previous QB), and at most 5 QBs sit in the top 50. A quarterback the
# VOR order places above the band is set at ROS_QB1_TARGET_RANK; the
# cluster then pulls each following QB up to at most
# ROS_QB_CLUSTER_GAP spots behind the previous one. The calibration
# only ever moves quarterbacks; every other position keeps its
# slot-filled order relative to the others, and Superflex (where QB
# scarcity is real) is never calibrated.
ROS_QB1_TARGET_RANK = 24
ROS_QB_BAND_TOP = 20
ROS_QB_CLUSTER_GAP = 6
ROS_QB_TOP50_CAP = 5
ROS_CALIBRATION_MIN_POOL = 50


def apply_qb_calibration(rows: List[dict], league_type: str = "1qb") -> List[dict]:
    """Enforce the 1QB QB shape on an ordered Overall list.

    ``rows`` is the slot-filled combined list (each row carries
    "position"). Returns a new list. No-ops for Superflex, for pools
    under ROS_CALIBRATION_MIN_POOL rows (there is no top 50 to shape),
    and when the pool carries no quarterbacks. Quarterbacks keep
    their relative (positional) order throughout; non-quarterbacks
    keep theirs.
    """
    if str(league_type or "1qb").lower() == "sf":
        return list(rows)
    rows = list(rows)
    if len(rows) < ROS_CALIBRATION_MIN_POOL:
        return rows

    def _is_qb(r):
        return str(r.get("position") or "").upper() == "QB"

    qb_rows = [r for r in rows if _is_qb(r)]
    if not qb_rows:
        return rows
    non_qb = [r for r in rows if not _is_qb(r)]
    natural = [i for i, r in enumerate(rows) if _is_qb(r)]

    # Assign each quarterback a slot, in positional order. QB1's slot
    # is his natural slot unless the VOR order elevated him above the
    # band, in which case he is set at the target rank (and the rest
    # of the group packs behind him: a naive move of QB1 alone would
    # leave QB2-QB4 sitting above him). Each following slot is the
    # natural slot clamped into (previous slot, previous slot +
    # ROS_QB_CLUSTER_GAP]: never above the previous QB, never more
    # than the cluster gap behind him.
    slots: List[int] = []
    for k, nat in enumerate(natural):
        if k == 0:
            slots.append(ROS_QB1_TARGET_RANK - 1
                         if nat < ROS_QB_BAND_TOP - 1 else nat)
        else:
            prev = slots[-1]
            slots.append(min(max(nat, prev + 1),
                             prev + ROS_QB_CLUSTER_GAP))
    # Cap: the 6th and later quarterbacks slot below rank 50, in
    # order (slot index >= 50 + their distance past the cap).
    for k in range(ROS_QB_TOP50_CAP, len(slots)):
        slots[k] = max(slots[k], slots[k - 1] + 1,
                       50 + (k - ROS_QB_TOP50_CAP))
    # Keep the slots strictly increasing and inside the list.
    n = len(rows)
    for k in range(len(slots) - 1, -1, -1):
        slots[k] = min(slots[k], n - (len(slots) - k))
        if k + 1 < len(slots):
            slots[k] = min(slots[k], slots[k + 1] - 1)

    out: List[Optional[dict]] = [None] * n
    for slot, r in zip(slots, qb_rows):
        out[slot] = r
    fill = iter(non_qb)
    return [r if r is not None else next(fill) for r in out]


def order_overall(rows: List[dict], league_type: str = "1qb") -> List[dict]:
    """Overall (All-view) order for ROS rows: VOR slot-fill.

    Pure VOR sorting decides WHICH slots each position occupies in the
    combined list (the cross-position allocation is exactly what pure
    VOR sorting produces), but each position's slots are filled by that
    position's players in factor order (ros_score desc, the position
    view's order, the order ros_pos_rank follows), so the relative
    order inside a position on the overall board always matches the
    position board. Pure VOR disagreed with the factor model inside
    positions (Chuba Hubbard, RB8 by score, sat below Javonte Williams,
    RB10, on the All page over a 0.1 VOR gap), which read as a
    contradiction between two pages carrying the same ranking.

    A monotonicity pass (_apply_position_monotonicity) refills each
    position's slots in ros_pos_rank order, so a tie the fill and the
    rank assignment break differently can never put a lower-ranked
    player above his position mate on the combined board. In 1QB the
    calibrated QB shape (apply_qb_calibration) is then applied: it is
    the one sanctioned deviation from the raw VOR slots, and it moves
    quarterbacks only.

    Rows are dicts carrying "position", "_vor" (the VOR total at the
    surface's replacement defaults), "ros_score", "ros_total",
    "ros_ppg" and "name". A pool whose rows carry no factor scores
    fills in ROS-total order, the position views' own fallback. The
    /players board mirrors this rule in JS (prRosSlotFillOrder in
    static/rankings.js).
    """
    def _vor_key(r: dict):
        return (-float(r.get("_vor") or 0.0),
                -float(r.get("ros_total") or 0),
                -float(r.get("ros_ppg") or 0),
                str(r.get("name") or ""))

    def _score_key(r: dict):
        return (-float(r.get("ros_score") or 0),
                -float(r.get("ros_ppg") or 0),
                str(r.get("name") or ""))

    def _total_key(r: dict):
        return (-float(r.get("ros_total") or 0),
                -float(r.get("ros_ppg") or 0),
                str(r.get("name") or ""))

    by_pos: Dict[str, List[dict]] = {}
    for r in rows:
        by_pos.setdefault(str(r.get("position") or "").upper(), []).append(r)
    queues: Dict[str, List[dict]] = {}
    for pos, members in by_pos.items():
        key = _score_key if any(
            m.get("ros_score") is not None for m in members) else _total_key
        queues[pos] = sorted(members, key=key)
    out: List[dict] = []
    cursor: Dict[str, int] = {}
    for r in sorted(rows, key=_vor_key):
        pos = str(r.get("position") or "").upper()
        i = cursor.get(pos, 0)
        out.append(queues[pos][i])
        cursor[pos] = i + 1
    return apply_qb_calibration(
        _apply_position_monotonicity(out), league_type)


def compute_ros_vor_tier_thresholds(ros_by_pid: Dict[str, dict],
                                    positions_by_pid: Dict[str, str]) -> dict:
    """Tier thresholds for the All view, computed on VOR totals.

    The All view orders by VOR total, so its dividers must be thresholds
    on VOR totals: one monotonic T1..Tn sequence for the mixed list.
    Keyed per window x league type x league size, the same shape the
    client looks up for dynasty tiers:
    {"all": {"1qb": {"10": [...]}, "sf": {...}}, "stretch": ..., ...}.
    """
    out: dict = {}
    for window_key, _prefix in _WINDOW_PREFIXES:
        by_type: dict = {}
        for league_type in ROS_LEAGUE_TYPES:
            by_size: dict = {}
            for size in ROS_LEAGUE_SIZES:
                vors = ros_vor_totals(
                    ros_by_pid, positions_by_pid, size, league_type)[window_key]
                rows = [(positions_by_pid.get(pid), v) for pid, v in vors.items()]
                by_size[str(size)] = _ros_tier_thresholds_for_values(rows)
            by_type[league_type] = by_size
        out[window_key] = by_type
    return out


# ── Loader: cached files in, ROS map out ────────────────────────────────────

_ROS_CACHE: dict = {"key": None, "payload": None}


def _ros_disk_cache_path(season: int) -> str:
    from utils.data_cache import CACHE_DIR

    return os.path.join(CACHE_DIR, f"ros_rankings_cache_s{int(season)}.json")


def _read_ros_disk_cache(season: int, key: tuple) -> Optional[dict]:
    """The precomputed ROS payload a previous build persisted, when it
    was built under the exact same data signature.

    This is the fast path the dynasty/redraft rankings already use
    (precomputed values, reads are file reads): the pool-wide build
    runs once per data change, and every later read in any worker or
    after a restart serves the stored payload instead of recomputing.
    A signature mismatch or an unreadable file is a miss, never an
    error, and a miss never serves a stale or empty payload as if it
    were current: the caller falls through to the build.
    """
    try:
        import json as _json

        with open(_ros_disk_cache_path(season)) as fh:
            blob = _json.load(fh)
        if not isinstance(blob, dict):
            return None
        # The stored key is JSON-shaped (lists); the caller's key is
        # (season, start_week, ((name, mtime), ...)): normalize the
        # stored form back to tuples before comparing.
        stored = blob.get("key")
        try:
            stored_key = (stored[0], stored[1],
                          tuple(tuple(part) for part in stored[2]))
        except Exception:
            return None
        if stored_key != key:
            return None
        payload = blob.get("payload")
        if not isinstance(payload, dict) or not payload.get("players"):
            return None
        return payload
    except Exception:
        return None


def _write_ros_disk_cache(season: int, key: tuple, payload: dict) -> None:
    """Persist a successful build for the read path above. Fail-soft:
    a cache that cannot be written only costs the next cold read the
    build it would have paid anyway."""
    try:
        import json as _json

        path = _ros_disk_cache_path(season)
        tmp = f"{path}.tmp"
        with open(tmp, "w") as fh:
            _json.dump({"key": list(key), "payload": payload}, fh)
        os.replace(tmp, path)
    except Exception:
        logger.debug("[ros] disk cache write failed", exc_info=True)


def _files_signature(season: int, start_week: int) -> tuple:
    """mtime signature over every cached file the ROS compute reads."""
    from utils.data_cache import CACHE_DIR, path_week_proj, path_week_schedule

    parts: List[Tuple[str, float]] = []
    stats_dir = os.path.join(CACHE_DIR, "sleeper_stats")
    stat_prefixes = (f"sleeper_stats_s{season}_w",
                     f"sleeper_stats_s{season - 1}_w")  # availability window
    try:
        for name in sorted(os.listdir(stats_dir)):
            if name.startswith(stat_prefixes):
                try:
                    parts.append((name, os.path.getmtime(os.path.join(stats_dir, name))))
                except OSError:
                    parts.append((name, -1.0))
    except OSError:
        pass
    for w in range(start_week, ROS_END_WEEK + 1):
        for path in (path_week_proj(season, w), path_week_schedule(season, w)):
            try:
                parts.append((os.path.basename(path), os.path.getmtime(path)))
            except OSError:
                parts.append((os.path.basename(path), -1.0))
    return tuple(parts)


def load_ros_map(season: Optional[int] = None) -> dict:
    """The full ROS payload block: per-player fields + window + thresholds.

    Returns {"players": {pid: fields}, "window": {"start_week", "end_week"},
    "tier_thresholds_ros": {...}, "tier_thresholds_ros_vor": {...},
    "updated": "Month D, YYYY"}. Empty players
    when the window has closed (past Week 17) or the cached feeds are not
    there yet: the board then simply has no ROS rows, never fabricated ones.
    Serves the precomputed payload (in-process signature cache, then the
    on-disk cache a previous build persisted under the same signature);
    the heavy pass runs at most once per data change across all workers,
    and its result is persisted for the next cold read. Never raises.
    """
    try:
        if not season:
            from dashboard_services.api import get_nfl_state
            season = int((get_nfl_state() or {}).get("season") or date.today().year)
        season = int(season)
    except Exception:
        return {"players": {}, "window": None, "tier_thresholds_ros": {}, "tier_thresholds_ros_vor": {},
                "updated": None}
    try:
        from utils.season_qualification import qualification_policy
        completed = [int(w) for w in qualification_policy(season).completed_weeks]
    except Exception:
        completed = []
    start_week = (max(completed) + 1) if completed else 1
    updated = date.today().strftime("%B %d, %Y")
    if start_week > ROS_END_WEEK:
        return {"players": {}, "window": {"start_week": start_week,
                                          "end_week": ROS_END_WEEK},
                "tier_thresholds_ros": {}, "tier_thresholds_ros_vor": {},
                "updated": updated}

    key = (season, start_week, _files_signature(season, start_week))
    if _ROS_CACHE.get("key") == key and _ROS_CACHE.get("payload") is not None:
        return _ROS_CACHE["payload"]

    disk = _read_ros_disk_cache(season, key)
    if disk is not None:
        _ROS_CACHE["key"] = key
        _ROS_CACHE["payload"] = disk
        return disk

    payload = _build_ros_map(season, start_week, updated)
    if payload is None:
        # Build failed: serve the empty shape for this call but do NOT
        # cache it, so the next request retries instead of pinning an
        # empty board to this data signature.
        return {"players": {}, "window": {"start_week": start_week,
                                          "end_week": ROS_END_WEEK},
                "tier_thresholds_ros": {}, "tier_thresholds_ros_vor": {},
                "updated": updated}
    _ROS_CACHE["key"] = key
    _ROS_CACHE["payload"] = payload
    if payload.get("players"):
        _write_ros_disk_cache(season, key, payload)
    return payload


def _build_ros_map(season: int, start_week: int, updated: str) -> Optional[dict]:
    try:
        from data_building.start_score_bundle import derive_start_sit_inputs
        from utils.data_cache import _read_week_projection_file
        from utils.fantasy_scoring import week_stat_points
        from utils.projections import weekly_projection_points
        from utils.utils import load_players_index, load_week_schedule
        from utils.schedule_ease import norm_sched_team
        import glob as _glob
        import json as _json
        from utils.data_cache import CACHE_DIR

        index = load_players_index() or {}
        played_inputs = derive_start_sit_inputs(season, dict(PPR_SCORING)) or {}

        # Usage + team-points pass over the completed weeks' stat files:
        # touches / targets / pass attempts per game for the role gate,
        # and each team's #1 at a position by points PER PLAYED GAME
        # (decided after the pass, below). The same pass also collects
        # the availability inputs over the current AND prior seasons: a
        # week is an appearance when the feed marks the player active
        # (gms_active), and a team game week when any of the team's
        # players was active that week (bye weeks never appear). Prior
        # season team attribution uses the player's current team: every
        # team plays the same number of games, so the only error is which
        # week the bye fell on, and the shrinkage prior absorbs it.
        usage: Dict[str, dict] = {}
        team_points: Dict[Tuple[str, str], Dict[str, float]] = {}
        appearances: Dict[str, int] = {}
        appearance_seasons: Dict[str, set] = {}
        team_weeks: Dict[Tuple[int, str], set] = {}
        # Factor-model inputs (full-window order), this season only:
        # per-player played-game logs, team-week opportunity denominators
        # for the role trend, and team-week offensive output.
        game_points: Dict[str, Dict[int, float]] = {}
        game_usage: Dict[str, Dict[int, Tuple[float, float, float]]] = {}
        team_week_targets: Dict[Tuple[str, int], float] = {}
        team_week_rushes: Dict[Tuple[str, int], float] = {}
        team_week_points: Dict[Tuple[str, int], float] = {}
        stat_paths: List[str] = []
        for yr in (season - 1, season):
            stat_paths += sorted(_glob.glob(os.path.join(
                CACHE_DIR, "sleeper_stats", f"sleeper_stats_s{yr}_w*.json")))
        for path in stat_paths:
            try:
                yr = int(path.rsplit("sleeper_stats_s", 1)[1].split("_w")[0])
                week = int(path.rsplit("_w", 1)[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            if yr == season and week >= start_week:
                continue  # only completed weeks carry played stats
            try:
                with open(path) as fh:
                    rows = _json.load(fh)
            except Exception:
                continue
            if not isinstance(rows, dict):
                continue
            for pid, st in rows.items():
                if not isinstance(st, dict):
                    continue
                pid = str(pid)
                meta = index.get(pid) or {}
                team = str(meta.get("team") or "").upper()
                # Played-week rule (participation), shared by the
                # game log below and the usage / team-points pass: gp
                # >= 1 or an offensive snap count present. A
                # participated week with no points line is a real
                # 0.0 game; a row with neither marker did not play.
                try:
                    _played = float(st.get("gp") or 0) >= 1 \
                        or st.get("off_snp") is not None
                except (TypeError, ValueError):
                    _played = st.get("off_snp") is not None
                if float(st.get("gms_active") or 0) > 0:
                    appearances[pid] = appearances.get(pid, 0) + 1
                    appearance_seasons.setdefault(pid, set()).add(yr)
                    if team:
                        team_weeks.setdefault((yr, team), set()).add(week)
                if yr == season:
                    # Played-game log on the participation rule (gp >= 1
                    # or a snap count present; a played game with no
                    # points line scores 0.0), plus the team-week
                    # denominators the role trend divides by.
                    if team:
                        team_week_targets[(team, week)] = \
                            team_week_targets.get((team, week), 0.0) \
                            + float(st.get("rec_tgt") or 0)
                        team_week_rushes[(team, week)] = \
                            team_week_rushes.get((team, week), 0.0) \
                            + float(st.get("rush_att") or 0)
                    if _played:
                        try:
                            _pts = float(week_stat_points(
                                st, dict(PPR_SCORING), ""))
                        except Exception:
                            _pts = 0.0
                        game_points.setdefault(pid, {})[week] = _pts
                        game_usage.setdefault(pid, {})[week] = (
                            float(st.get("rec_tgt") or 0),
                            float(st.get("rush_att") or 0),
                            float(st.get("pass_att") or 0))
                        _pos = str(meta.get("pos")
                                   or meta.get("position") or "").upper()
                        if team and _pos in ROS_POSITIONS:
                            team_week_points[(team, week)] = \
                                team_week_points.get((team, week), 0.0) + _pts
                if yr != season or not _played:
                    continue  # usage / team points are this season's played weeks
                pos = str(meta.get("pos") or meta.get("position") or "").upper()
                if pos not in ROS_POSITIONS:
                    continue
                u = usage.setdefault(pid, {"touches": 0.0, "targets": 0.0,
                                           "pass_att": 0.0})
                u["touches"] += float(st.get("rush_att") or 0) + float(st.get("rec") or 0)
                u["targets"] += float(st.get("rec_tgt") or 0)
                u["pass_att"] += float(st.get("pass_att") or 0)
                if team:
                    try:
                        pts = float(week_stat_points(st, dict(PPR_SCORING), ""))
                    except Exception:
                        pts = 0.0
                    bucket = team_points.setdefault((team, pos), {})
                    bucket[pid] = bucket.get(pid, 0.0) + pts
        # Team #1 at each position by points PER PLAYED GAME, not the
        # season total (2026-11-20 audit D1/D7): the totals leader is
        # whoever stayed healthy, so a takeover who missed weeks (or a
        # starter who banked points before being benched) held the spot
        # all season: Coleman could not lead DEN while Harvey out-
        # totalled him, Lloyd kept GB's spot at a 3.3 forward PPG while
        # Jacobs (14.6 PPG, fewer games) sat outside, and Winston's
        # total blocked McCarthy in NYG. Per-game with a >= 1 played
        # game requirement (played games from the shared derivation,
        # the same count the eligibility floor reads) prices the role
        # the player actually holds when he plays.
        def _team_top_key(item):
            _pid, _total = item
            _pg = int((played_inputs.get(_pid) or {}).get("played_games") or 0)
            return (_total / _pg) if _pg >= 1 else float("-inf")

        team_top: set = set()
        for bucket in team_points.values():
            if bucket:
                _lead = max(bucket.items(), key=_team_top_key)
                if _team_top_key(_lead) != float("-inf"):
                    team_top.add(_lead[0])

        # Team offense environment for the factor model: mean offensive
        # output per team game. This pipeline carries no real team-score
        # feed, so the proxy is the PPR points the team's QB/RB/WR/TE
        # scored in the weekly stat feed (a close tracker of real team
        # PPG, and the factor carries the model's smallest weight).
        _off_by_team: Dict[str, List[float]] = {}
        for (_t, _w), _pts in team_week_points.items():
            _off_by_team.setdefault(_t, []).append(_pts)
        team_offense_ppg = {t: sum(v) / len(v)
                            for t, v in _off_by_team.items() if v}

        # Availability: each player's team-games denominator counts ONLY
        # the seasons in which he actually appeared (2026-11-20 audit
        # D4): the old two-season sum rated a 2026 rookie against ~17
        # prior-season team games he could never have played in, pinning
        # Coleman and Love at the 0.70 clamp (Coleman ~72.6 ROS pts
        # instead of ~92 at a veteran-like rate). A veteran with
        # appearances in both seasons keeps the two-season denominator.
        # The positional mean (the shrinkage prior in availability_rate)
        # is recomputed on the same per-player denominators. The prior
        # population is the board's own population, players who clear
        # the same games floor this season: the mean over every indexed
        # name with a stat row would mostly measure roster churn
        # (deep-bench players active a few weeks a year) and drag every
        # regular's rate toward it.
        def _player_team_games(pid: str, team: str) -> int:
            return sum(len(team_weeks.get((yr, team)) or ())
                       for yr in appearance_seasons.get(pid) or ())

        raw_rates: Dict[str, List[float]] = {}
        for pid, apps in appearances.items():
            if int((played_inputs.get(pid) or {}).get("played_games") or 0) \
                    < MIN_GAMES_PLAYED:
                continue
            meta = index.get(pid) or {}
            pos = str(meta.get("pos") or meta.get("position") or "").upper()
            tg = _player_team_games(pid, str(meta.get("team") or "").upper())
            if pos in ROS_POSITIONS and tg > 0:
                raw_rates.setdefault(pos, []).append(apps / tg)
        pos_mean_rate = {pos: sum(v) / len(v)
                         for pos, v in raw_rates.items() if v}

        # Remaining schedules: team -> {week: opponent} over the window.
        team_opps: Dict[str, Dict[int, str]] = {}
        for w in range(start_week, ROS_END_WEEK + 1):
            try:
                games = load_week_schedule(season, w) or []
            except Exception:
                games = []
            for g in games:
                if not isinstance(g, dict):
                    continue
                home = norm_sched_team(str(g.get("home_team") or g.get("home") or ""))
                away = norm_sched_team(str(g.get("away_team") or g.get("away") or ""))
                if home and away:
                    team_opps.setdefault(home, {})[w] = away
                    team_opps.setdefault(away, {})[w] = home

        # ROS schedule rank per (position, team): average Schedule Rankings
        # matchup value of the remaining opponents, ranked across teams
        # (1 = softest). The full-window rank is the display value
        # (ros_sched_rank); the factor model reads a second rank built
        # from the playoff weeks (15-17) only (v3: near-week opponents
        # are already baked into the weekly projections).
        sched_rank: Dict[Tuple[str, str], int] = {}
        far_sched_rank: Dict[Tuple[str, str], int] = {}
        try:
            from app import _matchup_rank_table
            from utils.nfl import rank_values

            def _ranks_for_weeks(weeks) -> Dict[Tuple[str, str], int]:
                out: Dict[Tuple[str, str], int] = {}
                for pos in ROS_POSITIONS:
                    _r, _t, info, _z = _matchup_rank_table(season, pos)
                    team_values: Dict[str, float] = {}
                    for team, opps in team_opps.items():
                        vals = []
                        for w in weeks:
                            opp = opps.get(w)
                            row = (info or {}).get(opp) if opp else None
                            if row and isinstance(
                                    row.get("rank_value"), (int, float)):
                                vals.append(float(row["rank_value"]))
                        if vals:
                            team_values[team] = sum(vals) / len(vals)
                    ranks, _n = rank_values(team_values)
                    for team, rank in ranks.items():
                        out[(pos, team)] = int(rank)
                return out

            sched_rank = _ranks_for_weeks(
                range(start_week, ROS_END_WEEK + 1))
            far_sched_rank = _ranks_for_weeks(
                range(PLAYOFF_START_WEEK, ROS_END_WEEK + 1))
        except Exception:
            logger.debug("[ros] schedule rank unavailable", exc_info=True)
            sched_rank = {}
            far_sched_rank = {}

        # Weekly projection series per player, split into the three windows.
        week_maps: Dict[int, dict] = {}
        for w in range(start_week, ROS_END_WEEK + 1):
            week_maps[w] = _read_week_projection_file(season, w) or {}

        def _series(pid: str, pos: str, team: str, weeks) -> List[Optional[float]]:
            out: List[Optional[float]] = []
            for w in weeks:
                week_map = week_maps.get(w) or {}
                if not week_map:
                    continue  # feed not published for this week yet: not a week
                opps = team_opps.get(team)
                if opps is not None and w not in opps:
                    continue  # bye week: the team does not play
                out.append(weekly_projection_points(week_map, pid, dict(PPR_SCORING), pos))
            return out

        def _aligned_series(pid: str, pos: str, team: str,
                            weeks) -> List[Optional[float]]:
            # Calendar-aligned twin of _series for the QB streaming
            # replacement: exactly one entry per window week, None for
            # a bye or an unpublished feed, so the pool's series line
            # up week by week when the weekly best is taken across
            # quarterbacks. (_series itself keeps its compact shape:
            # the aggregate stats read skips bye weeks entirely.)
            out: List[Optional[float]] = []
            for w in weeks:
                week_map = week_maps.get(w) or {}
                opps = team_opps.get(team)
                if not week_map or (opps is not None and w not in opps):
                    out.append(None)
                    continue
                out.append(weekly_projection_points(
                    week_map, pid, dict(PPR_SCORING), pos))
            return out

        all_weeks = list(range(start_week, ROS_END_WEEK + 1))
        stretch_weeks = [w for w in all_weeks if w <= STRETCH_END_WEEK]
        playoff_weeks = [w for w in all_weeks if w >= PLAYOFF_START_WEEK]

        # Season-out overlay: the live Sleeper status map wins over the
        # players index's stamped injury_status, which lags the feed
        # (audit D2: Harvey and Love were Sleeper-listed IR yet ranked
        # because the index had not picked up their designations). The
        # notes map rides the same feed: it is the returner read's
        # season-ending signal (see is_in_window_returner).
        live_status = live_injury_status_map()
        live_notes = live_injury_notes_map()

        ros_by_pid: Dict[str, dict] = {}
        positions_by_pid: Dict[str, str] = {}
        factors_by_pid: Dict[str, dict] = {}
        for pid, meta in index.items():
            if not isinstance(meta, dict):
                continue
            pid = str(pid)
            pos = str(meta.get("pos") or meta.get("position") or "").upper()
            if pos not in ROS_POSITIONS:
                continue
            played = played_inputs.get(pid) or {}
            played_games = int(played.get("played_games") or 0)
            if played_games <= 0:
                continue
            team = str(meta.get("team") or "").upper()
            u = usage.get(pid) or {"touches": 0.0, "targets": 0.0,
                                   "pass_att": 0.0}
            windows = {
                "all": _series(pid, pos, team, all_weeks),
                "stretch": _series(pid, pos, team, stretch_weeks),
                "playoffs": _series(pid, pos, team, playoff_weeks),
            }
            if not windows["all"]:
                continue
            fields = compute_player_ros(
                pos,
                played_games=played_games,
                season_ppg=played.get("season_ppg"),
                last4_ppg=played.get("last4_ppg"),
                last4_games=min(FORM_WINDOW, played_games),
                touches_per_game=(u["touches"] / played_games) if played_games else None,
                targets_per_game=(u["targets"] / played_games) if played_games else None,
                pass_attempts_per_game=(u["pass_att"] / played_games) if played_games else None,
                is_team_top=pid in team_top,
                season_out=_season_out_verdict(
                    live_status.get(pid) or meta.get("injury_status")
                    or meta.get("status"),
                    live_notes.get(pid) or meta.get("injury_notes"),
                    windows["all"]),
                windows=windows,
                sched_rank=sched_rank.get((pos, norm_sched_team(team))),
                availability_rate=availability_rate(
                    appearances.get(pid), _player_team_games(pid, team),
                    pos_mean_rate.get(pos)),
            )
            if fields:
                if pos == "QB":
                    # The streaming replacement reads the QB pool's
                    # weekly series (see ros_qb_replacement_ppg): stamp
                    # the calendar-aligned series next to each window's
                    # aggregates. Only QBs carry them — no other
                    # position's replacement reads weekly values.
                    fields["ros_weekly"] = _aligned_series(
                        pid, pos, team, all_weeks)
                    fields["ros_stretch_weekly"] = _aligned_series(
                        pid, pos, team, stretch_weeks)
                    fields["ros_playoff_weekly"] = _aligned_series(
                        pid, pos, team, playoff_weeks)
                ros_by_pid[pid] = fields
                positions_by_pid[pid] = pos
                _log = game_points.get(pid) or {}
                factors_by_pid[pid] = player_model_factors(
                    pos,
                    ppg=fields.get("ros_ppg"),
                    games=fields.get("ros_games"),
                    sched_rank=far_sched_rank.get(
                        (pos, norm_sched_team(team))),
                    game_points=_log,
                    role_series=role_share_series(
                        pos, sorted(_log.keys()), game_usage.get(pid) or {},
                        team_week_targets, team_week_rushes, team),
                    offense=team_offense_ppg.get(team),
                )

        model_scores: Dict[str, float] = {}
        _factors_by_pos: Dict[str, Dict[str, dict]] = {}
        for _pid, _fac in factors_by_pid.items():
            _factors_by_pos.setdefault(positions_by_pid[_pid], {})[_pid] = _fac
        for _pos, _fac_pool in _factors_by_pos.items():
            model_scores.update(ros_model_scores(_fac_pool, _pos))
        assign_ros_ranks(ros_by_pid, positions_by_pid,
                         model_scores=model_scores)
        return {
            "players": ros_by_pid,
            "window": {"start_week": start_week, "end_week": ROS_END_WEEK},
            "tier_thresholds_ros": compute_ros_tier_thresholds(ros_by_pid, positions_by_pid),
            "tier_thresholds_ros_vor": compute_ros_vor_tier_thresholds(ros_by_pid, positions_by_pid),
            "updated": updated,
        }
    except Exception:
        logger.exception("[ros] ROS map build failed")
        return None
