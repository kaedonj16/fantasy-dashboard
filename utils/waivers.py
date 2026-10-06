"""Consolidated utils module: waivers.

waiver scoring, lineup evaluation, big-game detector, streaming

Merged from: utils/waiver_score.py, utils/waiver_lineup.py, utils/waiver_big_game.py, utils/streaming_targets.py.
Old import paths keep working via compatibility shims.
"""
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations
from __future__ import annotations


# ======================================================================
# From utils/waiver_score.py
# ======================================================================

"""Pure waiver-pickup scoring and signal classification.

Extracted from app.py so the ranking model can be unit-tested without the
pandas/DB stack, and shared by both waiver surfaces (the /api/waiver-candidates
endpoint and the offseason dashboard card) so they rank and label identically.

Design goals for a *waiver target* list (vs. a plain dynasty-value list):

  * Value informs the ranking but must not dominate it. A saturating curve
    compresses the gap between a 1500-value veteran and a 250-value breakout so
    that opportunity signals can lift an emerging player above a static one.
  * Rest-of-season projected production matters for in-season pickups, so it is
    blended in alongside dynasty value (#5).
  * Opportunity signals — an injured player ahead on the depth chart, a recent
    usage spike, a high breakout score — are what make a free agent worth adding
    *now*. They feed the score directly, but because they are correlated (all
    proxy "the role is opening up") they are combined with diminishing returns
    rather than plain addition, so one real event isn't triple-counted (#6).
  * The injury signal only credits a candidate who is actually next in line: a
    healthy body still ahead of them dampens it, and a candidate who is himself
    hurt is discounted (#1, #2). Stale injuries and low-volume roles are
    down-weighted (#3, #7).
  * The ranking is roster-aware: a position of real need to the viewer is worth
    more (#4).
  * Age is a smooth curve: ascending-young players are rewarded progressively
    and past-prime players decay (bounded), rather than a hard cliff at prime.

WEIGHTS below is the single calibration surface — the backtest harness
(scripts/backtest_waiver_targets.py, #8) tunes these against realized
production rather than leaving them hand-picked.
"""

import re
from dataclasses import dataclass, replace as _dc_replace

from utils.draft import clamp01 as _clamp01

# Age past which each position starts losing the age bonus (peak dynasty window).
WAIVER_PRIME_MAX = {"QB": 33, "RB": 26, "WR": 28, "TE": 29}

# Minimum last-3-week-vs-season rise, per stat, to count as a usage spike. A
# candidate whose delta hits its stat's threshold has a usage ratio of 1.0.
USAGE_SPIKE_MIN = {"snap_pct": 8.0, "touches": 3.0, "targets": 2.0}

# NFL injury/roster statuses that vacate opportunity for the players behind them
# on the depth chart, weighted by how likely the injured player is to miss time
# (and thus how much of the role opens up). Even a QUESTIONABLE tag ahead is a
# real bump — the backup is one setback from the job — so it scores, just less
# than a confirmed absence.
VACANCY_SEVERITY = {
    "IR": 1.0, "PUP": 1.0, "NFI": 1.0, "SUSP": 0.9, "OUT": 0.85,
    "DOUBTFUL": 0.5, "QUESTIONABLE": 0.3,
}

# Severity at/above which the vacancy is treated as a genuine "role is open now"
# (confirmed/likely absence -> "Next Man Up"); below it is a softer bump.
VACANCY_STRONG = 0.5

# Expected weeks the role stays open, by the injured player's status — the
# injury "timeline". Sleeper doesn't publish per-player return dates, so this is
# modeled by injury class: IR/PUP/NFI are multi-week (NFL IR = min 4 games), a
# plain OUT is week-to-week (~1 game), and Doubtful/Questionable are this-week
# calls. Combined with the vacated role's projected points, this turns "someone
# ahead is hurt" into "how many points does the role free up over the window".
INJURY_DURATION_WEEKS = {
    "IR": 6.0, "PUP": 6.0, "NFI": 6.0, "SUSP": 4.0,
    "OUT": 1.0, "DOUBTFUL": 0.8, "QUESTIONABLE": 0.4,
}

# Statuses that mean a player is NOT expected on the field, so they no longer
# block the depth chart for the player behind them.
_OUT_STATUSES = {"IR", "PUP", "NFI", "SUSP", "SUS", "OUT", "DOUBTFUL", "NA"}

# Designations that mean "do not present this player as a clean pickup".
# Sleeper emits title-case ("Out", "Questionable") and "SUS" for suspensions;
# compare upper-cased. QUESTIONABLE is deliberately NOT here: a game-time call
# is still addable, just flagged.
SERIOUS_INJURY_STATUSES = {"IR", "PUP", "NFI", "SUSP", "SUS", "OUT", "DOUBTFUL", "NA"}


def is_seriously_hurt(status) -> bool:
    """True when a designation means the player must never be a clean add."""
    return str(status or "").strip().upper() in SERIOUS_INJURY_STATUSES


def waiver_injury_note(status, body_part=None, weeks_out=None) -> str | None:
    """Short human-readable injury flag for a waiver candidate, or None.

    QUESTIONABLE renders as "Game-time call" (the Start/Sit convention);
    anything else renders designation-first with body part / return timeline
    when known. Never invents: missing inputs are simply omitted.
    """
    s = str(status or "").strip()
    if not s:
        return None
    su = s.upper()
    if su == "QUESTIONABLE":
        note = "Game-time call"
        if body_part:
            note += f" ({body_part})"
        return note
    label = su if len(su) <= 3 else s
    parts = [label]
    if body_part:
        parts.append(str(body_part))
    try:
        w = float(weeks_out) if weeks_out is not None else None
    except (TypeError, ValueError):
        w = None
    if w is not None and w > 0:
        parts.append(f"~{int(round(w))} wks")
    return " \u00b7 ".join(parts)


def drop_seriously_hurt(candidates, *, dynasty: bool = False) -> list:
    """Remove seriously-hurt players from a ranked waiver candidate list.

    Redraft: a player who is OUT / DOUBTFUL / IR / PUP / NFI / NA / SUS is
    never a pickup recommendation, so drop them. Dynasty keeps them (an IR
    stash is a real move there) -- callers must render the injury flag instead.
    Reads each candidate's ``self_status`` (the Sleeper designation).
    """
    if dynasty:
        return list(candidates or [])
    return [c for c in (candidates or [])
            if not is_seriously_hurt((c or {}).get("self_status"))]


@dataclass(frozen=True)
class WaiverWeights:
    """Every tunable constant in the model, in one place, so #8 (backtest
    calibration) has a single surface to fit instead of magic numbers scattered
    through the code."""
    # Saturating value curve: VALUE_MAX * v / (v + VALUE_HALF).
    value_max: float = 120.0
    value_half: float = 500.0
    # Minimum dynasty value a candidate must clear to be a *target* at all.
    # Regular starters sit at 500-1500+, so free agents cluster low; below this
    # floor a player has essentially no dynasty relevance and only surfaced by
    # riding a trend/age bonus (e.g. a value-0 player badged "Rising Fast").
    # The waiver surfaces filter on this so that noise can't reach the list.
    min_value: float = 25.0
    # Rest-of-season projection: projected PPG * proj_per_ppg, capped.
    proj_per_ppg: float = 4.0
    proj_max: float = 60.0
    # Opportunity components (pre-combine caps).
    injury_max: float = 55.0
    # Injury vacancy is scored from expected vacated fantasy points over a
    # forward window: sev * min(weeks_out, horizon) * projected_ppg, times this.
    injury_pts_per_vacated_ppg: float = 1.2
    injury_horizon_weeks: float = 4.0
    injury_fallback_ppg: float = 9.0  # used when the injured player's proj is unknown
    # Near-term weeks matter more (you can always drop the player later), so each
    # future week of a vacancy is discounted by this per week (#8).
    injury_week_decay: float = 0.85
    usage_per_ratio: float = 30.0
    usage_max: float = 50.0
    breakout_per: float = 0.5
    breakout_max: float = 45.0
    # Unexpected big game (from the shared detector) as an opportunity signal.
    # Scored from surprise * sustainability so a fluky watchlist game barely
    # moves the needle while a sustainable priority discovery is a real bump.
    big_game_scale: float = 95.0
    big_game_max: float = 40.0
    # Diminishing-returns weights when combining correlated opportunity signals.
    # Big game, usage, breakout and injury all proxy "the role is opening up", so
    # they are combined with decay (not summed) — a player who shows up on several
    # is not quadruple-counted (#6).
    opp_second: float = 0.5
    opp_third: float = 0.25
    opp_fourth: float = 0.125
    # Weekly rank trend. A waiver target should be justified by real value +
    # opportunity, not by a single noisy 7-day rank swing, so the raw-trend term
    # is capped well below the value / opportunity terms (a big riser still gets
    # a meaningful, but not dominating, bump).
    trend_up_per: float = 2.5
    trend_up_max: float = 25.0
    trend_down_per: float = 1.5
    trend_down_floor: float = -15.0
    # Age curve.
    age_base: float = 22.0
    age_youth_per: float = 2.0
    age_youth_max: float = 36.0
    age_decay_per: float = 7.0
    age_floor: float = -22.0
    # Roster need: up to +need_max_bonus (fraction) for a high-need position.
    need_max_bonus: float = 0.25
    # Weekly rank trend is a single noisy window; shrink it toward zero (#7).
    trend_shrink: float = 0.2
    # rank_change_7d is *overall* rank movement, which is dense (noisy) for deep
    # players. Discount the trend by depth: a move at positional rank D counts
    # like D / (D + trend_depth_ref) less. The badge thresholds scale with depth
    # too, so a deep player needs a proportionally bigger move to "rise".
    trend_depth_ref: float = 24.0
    trend_fast_frac: float = 0.5   # "Rising Fast" needs >= max(floor, frac * pos_depth)
    trend_up_frac: float = 0.2     # "Trending Up" needs >= max(floor, frac * pos_depth)
    trend_fast_floor: float = 8.0
    trend_up_floor: float = 3.0
    # A waiver list is sorted by a trend-weighted score, so *everything* shown is
    # a riser — labeling them all "Rising Fast" is useless. Badge relative to the
    # displayed set: only movers at/above these percentiles of the shown pool
    # earn the trend badges.
    trend_fast_pct: float = 0.70
    trend_up_pct: float = 0.40
    # Upcoming schedule ease: up to this many points for a soft slate (#3).
    schedule_bonus_max: float = 12.0
    # Positional scarcity: up to +scarcity_max_bonus (fraction) for a player well
    # above replacement level at a scarce position (#4).
    scarcity_max_bonus: float = 0.20


WEIGHTS = WaiverWeights()


def _pos_depth(c: dict):
    """Positional rank depth for a candidate, e.g. 89 for a WR89. Read from
    ``pos_rank`` if present, else parsed from the trailing number of
    ``pos_rank_label``. Used to discount noisy deep-player rank movement."""
    pr = c.get("pos_rank")
    try:
        if pr:
            return int(pr)
    except (TypeError, ValueError):
        pass
    m = re.search(r"(\d+)\s*$", str(c.get("pos_rank_label") or ""))
    return int(m.group(1)) if m else None


def _discounted_weeks(weeks: float, decay: float) -> float:
    """Sum of geometrically-decayed week weights: week 0 counts 1.0, week 1
    counts ``decay``, week 2 ``decay**2`` ... so near-term weeks of a vacancy
    matter more than distant ones (#8). Handles fractional final weeks."""
    total = 0.0
    i = 0
    rem = max(0.0, float(weeks))
    while rem > 1e-9 and i < 64:
        step = 1.0 if rem >= 1.0 else rem
        total += step * (decay ** i)
        rem -= step
        i += 1
    return total


# ---------------------------------------------------------------------------
# Value / projection
# ---------------------------------------------------------------------------

def value_component(val, w: WaiverWeights = WEIGHTS) -> float:
    """Saturating value contribution.

    w.value_max * v / (v + w.value_half): concave, so value stays monotonic but
    its gaps compress, keeping a high-value free agent attractive without letting
    static value bury emerging players.
    """
    try:
        v = max(0.0, float(val or 0))
    except (TypeError, ValueError):
        return 0.0
    return w.value_max * v / (v + w.value_half)


def projection_component(ros_ppg, w: WaiverWeights = WEIGHTS) -> float:
    """Rest-of-season projected points contribution (#5). 0 when no projection."""
    try:
        ppg = max(0.0, float(ros_ppg or 0))
    except (TypeError, ValueError):
        return 0.0
    return min(ppg * w.proj_per_ppg, w.proj_max)


# ---------------------------------------------------------------------------
# Usage spike
# ---------------------------------------------------------------------------

def usage_ratio(stat, delta) -> float:
    """Usage-spike magnitude as a multiple of the stat's spike threshold.

    Returns 0.0 when there is no usage data. A player exactly at the threshold
    scores 1.0; twice the threshold scores 2.0.
    """
    if not stat or delta is None:
        return 0.0
    thr = USAGE_SPIKE_MIN.get(stat, 3.0)
    if thr <= 0:
        return 0.0
    try:
        return max(0.0, float(delta) / thr)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Depth chart / injuries
# ---------------------------------------------------------------------------

def build_depth_index(full_players: dict) -> dict:
    """Group a Sleeper players map by (team, position) for depth-chart lookups.

    Returns ``{(TEAM, POS): [{"pid", "depth_order", "status"}, ...]}`` where
    ``status`` is the player's injury_status (falling back to roster status).
    """
    idx: dict = {}
    for pid, p in (full_players or {}).items():
        if not isinstance(p, dict):
            continue
        team = str(p.get("team") or "").upper()
        pos = str(p.get("position") or "").upper()
        if not team or not pos:
            continue
        idx.setdefault((team, pos), []).append({
            "pid": str(pid),
            "depth_order": p.get("depth_chart_order"),
            "status": p.get("injury_status") or p.get("status") or "",
        })
    return idx


def _will_play(status) -> bool:
    """Whether a player at this status is expected to take the field (and thus
    still blocks the depth chart for the player behind them). QUESTIONABLE
    players usually play, so they still block — even though they also contribute
    a soft vacancy for the backup."""
    return str(status or "").upper() not in _OUT_STATUSES


def depth_analysis(candidate_order, teammates) -> dict:
    """Analyze the depth chart ahead of a candidate.

    ``teammates`` is an iterable of ``{"depth_order", "status", "pid"?}`` for the
    same team + position (excluding the candidate). Returns:

      * injured_ahead:      vacating injury statuses of players ranked ahead
      * injured_pids_ahead: their pids (for vacated-volume lookup, #7)
      * healthy_ahead:      count of will-play players still ahead (blockers, #1)

    A falsy ``candidate_order`` is treated as deep, so any injured starter ahead
    still counts.
    """
    mine = candidate_order or 99
    injured: list = []
    injured_pids: list = []
    vacated: list = []
    healthy_ahead = 0
    healthy_pairs: list = []   # (depth_order, pid) of will-play blockers ahead
    for t in teammates:
        o = t.get("depth_order") or 99
        if o >= mine:
            continue
        st = str(t.get("status") or "").upper()
        if st in VACANCY_SEVERITY:
            injured.append(st)
            vacated.append({"status": st, "pid": t.get("pid"), "proj_ppg": t.get("proj_ppg")})
            if t.get("pid") is not None:
                injured_pids.append(t.get("pid"))
        if _will_play(st):
            healthy_ahead += 1
            if t.get("pid") is not None:
                healthy_pairs.append((o, t.get("pid")))
    # pids of healthy blockers ahead, nearest first — [0] is the starter this
    # candidate directly backs up (handcuff-upside lookup, #8).
    healthy_pids_ahead = [pid for _o, pid in sorted(healthy_pairs, key=lambda p: p[0])]
    return {
        "injured_ahead": injured,          # statuses (badge / severity)
        "injured_pids_ahead": injured_pids,
        "vacated": vacated,                # [{status, pid, proj_ppg}] for scoring
        "healthy_ahead": healthy_ahead,
        "healthy_pids_ahead": healthy_pids_ahead,
    }


def depth_analysis_for_player(pid, full_players: dict, depth_index: dict) -> dict:
    """Convenience wrapper: depth_analysis for ``pid`` on its own depth chart."""
    fp = full_players or {}
    p = fp.get(pid) or fp.get(str(pid)) or {}
    team = str(p.get("team") or "").upper()
    pos = str(p.get("position") or "").upper()
    if not team or not pos:
        return {"injured_ahead": [], "injured_pids_ahead": [], "vacated": [],
                "healthy_ahead": 0, "healthy_pids_ahead": []}
    group = (depth_index or {}).get((team, pos)) or []
    teammates = [g for g in group if g.get("pid") != str(pid)]
    return depth_analysis(p.get("depth_chart_order"), teammates)


def injured_ahead(depth_order, teammates) -> list:
    """Back-compat helper: just the vacating statuses ahead (statuses only)."""
    return depth_analysis(depth_order, teammates)["injured_ahead"]


def injured_ahead_for_player(pid, full_players: dict, depth_index: dict) -> list:
    """Back-compat helper: vacating statuses ahead of ``pid`` (statuses only)."""
    return depth_analysis_for_player(pid, full_players, depth_index)["injured_ahead"]


def _proximity_weight(healthy_ahead: int) -> float:
    """How much an injury ahead actually helps, given healthy blockers remain (#1).

    0 healthy blockers -> candidate is next up (full credit); each remaining
    healthy body ahead sharply discounts the benefit; 3+ -> effectively none.
    """
    return {0: 1.0, 1: 0.55, 2: 0.2}.get(healthy_ahead, 0.0)


def strip_bye_weeks(weekly_projs, plays_this_week) -> list:
    """Drop a player's bye week(s) from their upcoming-projection series.

    A bye projects ~0 just like an injury, so counting it would both overstate
    the timeline and break/extend the streak wrongly. ``plays_this_week`` is a
    parallel sequence of booleans (True/None = the team is scheduled that week,
    False = bye); False entries are removed so the zero-run reflects games
    actually missed, not byes.
    """
    projs = list(weekly_projs or [])
    out = []
    for i, p in enumerate(projs):
        plays = plays_this_week[i] if (plays_this_week and i < len(plays_this_week)) else True
        if plays is False:
            continue
        out.append(p)
    return out


def weeks_out_from_projections(weekly_projs, zero_threshold: float = 1.0,
                               treat_missing_as_out: bool = True) -> int:
    """Derive weeks-out from the leading run of ~zero weekly projections.

    Projection providers zero out a player's weekly points for every week they're
    expected to miss, so the number of consecutive at-or-below-threshold weeks
    starting now is a direct read on the injury timeline — far better than
    guessing from the injury label. ``weekly_projs`` is this player's projected
    points for the upcoming weeks, in order (week now, +1, +2, ...).

    ``treat_missing_as_out`` controls what a ``None`` (missing) entry means. When
    True (default, back-compat) a missing week counts as out — the historical
    behavior for providers that omit an inactive player. When False (item 9) a
    missing projection is *unknown*: the run STOPS there rather than being treated
    as a confirmed zero-point week, so a player who simply hasn't been loaded into
    the feed can't fabricate or extend an inferred absence. An *explicit* zero
    (present but ~0) always counts either way.
    """
    n = 0
    for p in (weekly_projs or []):
        if p is None:
            if treat_missing_as_out:
                n += 1
                continue
            break  # missing => unknown; do not extend an inferred absence
        try:
            v = float(p)
        except (TypeError, ValueError):
            # Unparseable => unknown, same treatment as missing.
            if treat_missing_as_out:
                n += 1
                continue
            break
        if v <= zero_threshold:
            n += 1
        else:
            break
    return n


# Week-status vocabulary the UI/scoring must keep distinct (item 9): a missing
# projection is NOT an explicit zero, a bye is NOT an absence, and an explicit
# zero alone is NOT proof of injury.
WEEK_MISSING = "missing"        # no projection loaded — unknown
WEEK_ZERO = "zero"              # explicitly projected ~0 (inactive / not in plans)
WEEK_BYE = "bye"                # team not scheduled
WEEK_OUT = "out"               # confirmed absence (injury/return info)
WEEK_PLAYING = "playing"        # projected to play


def classify_projection_week(proj, *, present: bool = True, bye: bool = False,
                             confirmed_out: bool = False,
                             zero_threshold: float = 1.0) -> str:
    """Label one upcoming week with provenance, keeping the five states distinct.

    ``confirmed_out`` comes from reliable injury/return information (not inferred
    from a zero projection). ``present`` is False when the player is absent from
    the projection feed for that week (missing, not zero)."""
    if bye:
        return WEEK_BYE
    if confirmed_out:
        return WEEK_OUT
    if not present or proj is None:
        return WEEK_MISSING
    try:
        v = float(proj)
    except (TypeError, ValueError):
        return WEEK_MISSING
    return WEEK_ZERO if v <= zero_threshold else WEEK_PLAYING


def return_timeline(week_labels, *, return_week_override=None) -> dict:
    """Summarize an upcoming-week label series into a structured, honest timeline.

    ``week_labels`` is the output of :func:`classify_projection_week` per week,
    week now first. Returns weeks_out (confirmed absence run, stopping at the
    first unknown/playing week), the basis, and coverage so the caller can show
    uncertainty rather than a false-precise return date. A leading run of BYE
    weeks is skipped (a bye isn't a missed game); the run ends at the first
    MISSING (unknown) or PLAYING week.
    """
    labels = list(week_labels or [])
    weeks_out = 0
    unknown_from = None
    basis = "none"
    for i, lab in enumerate(labels):
        if lab == WEEK_BYE:
            continue
        if lab in (WEEK_OUT, WEEK_ZERO):
            weeks_out += 1
            basis = "confirmed" if lab == WEEK_OUT else "projection_zero"
        elif lab == WEEK_MISSING:
            unknown_from = i
            break
        else:  # WEEK_PLAYING
            break
    covered = sum(1 for lab in labels if lab != WEEK_MISSING)
    missing = sum(1 for lab in labels if lab == WEEK_MISSING)
    if return_week_override is not None:
        weeks_out = int(return_week_override)
        basis = "return_date"
    return {
        "weeks_out": weeks_out,
        "basis": basis,
        "weeks_covered": covered,
        "weeks_missing": missing,
        "unknown_from": unknown_from,
        "estimated": basis in ("projection_zero",),  # projection-derived => estimate, not certain
    }


def expected_vacated_points(vacated, horizon_weeks: float = None,
                            fallback_ppg: float = None,
                            w: WaiverWeights = WEIGHTS) -> float:
    """Expected fantasy points a candidate inherits from injuries ahead.

    ``vacated`` items may be plain status strings or dicts with ``status`` and,
    optionally, ``proj_ppg`` (the vacated role's healthy production) and
    ``weeks_out`` (a projection-derived timeline — how many upcoming weeks the
    player is projected for ~zero). For each injured player ahead:

        likelihood * min(weeks_out, horizon) * projected_ppg

    Timeline source: when ``weeks_out`` is present and positive it is
    authoritative (the projections literally show the player out that long) and
    likelihood is ~1.0; otherwise the injury *class* supplies both an estimated
    duration (INJURY_DURATION_WEEKS) and a likelihood (VACANCY_SEVERITY). PPG
    falls back to a startable baseline when unknown. Summed across everyone
    injured ahead.
    """
    if horizon_weeks is None:
        horizon_weeks = w.injury_horizon_weeks
    if fallback_ppg is None:
        fallback_ppg = w.injury_fallback_ppg
    total = 0.0
    for item in (vacated or []):
        if isinstance(item, dict):
            st = str(item.get("status") or "").upper()
            ppg = item.get("proj_ppg")
            weeks_override = item.get("weeks_out")
        else:
            st = str(item or "").upper()
            ppg = None
            weeks_override = None
        sev = VACANCY_SEVERITY.get(st, 0.0)

        # Projection-derived timeline is authoritative when it shows the player
        # out; otherwise fall back to the injury-class estimate. (A projection
        # that shows them playing never zeroes out a confirmed injury — it just
        # doesn't extend it — so real injuries aren't dropped on projection quirks.)
        try:
            wo = float(weeks_override) if weeks_override is not None else 0.0
        except (TypeError, ValueError):
            wo = 0.0
        if wo > 0:
            weeks = min(wo, float(horizon_weeks))
            likelihood = 1.0
        elif sev > 0:
            weeks = min(INJURY_DURATION_WEEKS.get(st, 1.0), float(horizon_weeks))
            likelihood = sev
        else:
            continue

        try:
            ppg_v = float(ppg) if ppg is not None else float(fallback_ppg)
        except (TypeError, ValueError):
            ppg_v = float(fallback_ppg)
        # Discount later weeks — near-term opportunity is worth more (#8).
        total += likelihood * _discounted_weeks(weeks, w.injury_week_decay) * max(0.0, ppg_v)
    return total


def depth_chart_vacancy_score(vacated, healthy_ahead: int = 0,
                              volume_weight: float = 1.0, freshness: float = 1.0,
                              w: WaiverWeights = WEIGHTS) -> float:
    """Points for injured players sitting ahead on the depth chart (0 .. injury_max).

    Scored from the *expected vacated fantasy points* (likelihood × timeline ×
    projected production), so a season-ending injury to a high-scoring role
    ahead dwarfs a one-week absence — then scaled by:

      * proximity (#1): healthy players still ahead dampen it,
      * freshness (#3): a stale injury whose role has already transferred is
        worth less, and
      * volume_weight: optional extra nudge (kept for back-compat; defaults 1.0).

    ``vacated`` accepts status strings or ``{status, proj_ppg}`` dicts.
    """
    ev = expected_vacated_points(vacated, w=w)
    if ev <= 0:
        return 0.0
    base = min(ev * w.injury_pts_per_vacated_ppg, w.injury_max)
    scaled = base * _proximity_weight(healthy_ahead) * float(volume_weight) * float(freshness)
    return max(0.0, scaled)


def self_injury_multiplier(status) -> float:
    """Discount for a candidate who is himself hurt (#2). A confirmed-out backup
    is not a pickup this week, so his whole score is zeroed; softer statuses
    scale down."""
    s = str(status or "").upper()
    if s in {"IR", "PUP", "NFI", "SUSP", "SUS", "NA", "OUT"}:
        return 0.0
    if s == "DOUBTFUL":
        return 0.35
    if s == "QUESTIONABLE":
        return 0.85
    return 1.0


# ---------------------------------------------------------------------------
# Trend / schedule / scarcity
# ---------------------------------------------------------------------------

def blended_trend(windows, w: WaiverWeights = WEIGHTS) -> float:
    """Blend one or more rank-change windows into a single, noise-shrunk trend (#7).

    ``windows`` maps a window label to its rank change (e.g. {"7d": 6, "14d": 4}).
    A single window is inherently noisy, so the blend is shrunk toward zero; the
    more windows corroborate, the less it is shrunk. Missing/None windows are
    ignored, so this improves automatically once longer windows exist in the data.
    """
    vals = []
    for v in (windows or {}).values():
        if v is None:
            continue
        try:
            vals.append(float(v))
        except (TypeError, ValueError):
            continue
    if not vals:
        return 0.0
    avg = sum(vals) / len(vals)
    shrink = w.trend_shrink / max(1, len(vals))
    return avg * (1.0 - shrink)


def adaptive_trend_thresholds(rank_changes, w: WaiverWeights = WEIGHTS) -> "tuple[float, float]":
    """Trend-badge thresholds derived from the *displayed* candidates' rank moves.

    Because the waiver list is sorted by a trend-weighted score, every shown
    player is a riser; a fixed threshold labels them all "Rising Fast". Instead,
    reserve "Rising Fast" for the strongest movers in the shown set (>= the
    trend_fast_pct percentile) and "Trending Up" for the next tier, so the badges
    actually differentiate. Falls back to the fixed floors when there aren't
    enough positive movers to form a distribution.
    """
    pos = sorted(float(x) for x in (rank_changes or []) if x is not None and float(x) > 0)
    if len(pos) < 5:
        return (w.trend_fast_floor, w.trend_up_floor)

    def _q(p):
        return pos[min(len(pos) - 1, int(p * len(pos)))]

    return (max(w.trend_fast_floor, _q(w.trend_fast_pct)),
            max(w.trend_up_floor, _q(w.trend_up_pct)))


def schedule_bonus(ease_rank, total_teams, w: WaiverWeights = WEIGHTS) -> float:
    """Bonus/penalty for the upcoming schedule (#3).

    ``ease_rank`` is the position's matchup rank (1 = easiest slate), out of
    ``total_teams``. Easiest slates earn up to +schedule_bonus_max/2, the hardest
    lose the same, and a median schedule is neutral.
    """
    try:
        rank = float(ease_rank)
        total = float(total_teams)
    except (TypeError, ValueError):
        return 0.0
    if not rank or total < 2:
        return 0.0
    pct = 1.0 - (rank - 1.0) / (total - 1.0)   # 1.0 easiest ... 0.0 hardest
    return w.schedule_bonus_max * (pct - 0.5)


def schedule_urgency(ease_rank, total_teams: int = 32) -> str | None:
    """Short claim urgency when the upcoming slate is among the hardest.

    Returns None when schedule data is missing or the matchup is average/easy —
    we only surface urgency when claiming *before* a rough stretch matters.
    """
    try:
        rank = float(ease_rank)
        total = float(total_teams or 32)
    except (TypeError, ValueError):
        return None
    if not rank or total < 4:
        return None
    pct = (rank - 1.0) / (total - 1.0)  # 0 easiest … 1 hardest
    if pct >= 0.75:
        return "Claim before tough stretch"
    if pct >= 0.60:
        return "Schedule turns harder soon"
    return None


def roster_needs_drop(active_count: int, roster_slots: int) -> bool:
    """True when the active roster is at/over capacity (a drop is required to add)."""
    try:
        n = int(active_count)
        slots = int(roster_slots)
    except (TypeError, ValueError):
        return False
    return slots > 0 and n >= slots


def pick_waiver_push_candidate(
    value_tbl: list,
    rostered: set,
    *,
    min_value: float = 500.0,
    players: dict | None = None,
) -> dict | None:
    """Top free-agent for the hourly waiver push (R05.4).

    Prefers higher dynasty value among skill-position players who still have an
    NFL team. Players with a serious injury designation are never picked: a
    push telling a manager to add a player who is OUT or on IR is worse than
    no push. ``players`` is the Sleeper players feed ({pid: meta}); when it is
    unavailable the injury screen is skipped rather than failing the push.
    Returns ``{name, position, value, player_id}`` or None.
    """
    available = []
    rostered_ids = {str(p) for p in (rostered or set())}
    for p in value_tbl or []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "")
        if not pid or pid in rostered_ids:
            continue
        pos = str(p.get("position") or "").upper()
        if pos not in ("QB", "RB", "WR", "TE"):
            continue
        team = str(p.get("team") or "").strip().upper()
        if team in ("", "FA", "FREE AGENT", "NONE"):
            continue
        if players:
            meta = players.get(pid) or players.get(str(pid)) or {}
            if is_seriously_hurt(meta.get("injury_status") or meta.get("status")):
                continue
        try:
            val = float(p.get("value") or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val < float(min_value):
            continue
        available.append({
            "player_id": pid,
            "name": p.get("name") or p.get("full_name") or "A top player",
            "position": pos,
            "value": val,
        })
    if not available:
        return None
    available.sort(key=lambda row: row["value"], reverse=True)
    return available[0]


def waiver_push_copy(candidate: dict) -> tuple[str, str]:
    """Title + body for the waiver-of-the-week push."""
    name = str((candidate or {}).get("name") or "A top player").strip() or "A top player"
    pos = str((candidate or {}).get("position") or "").strip()
    label = f"{name} ({pos})" if pos else name
    title = "Waiver of the week"
    body = (
        f"{label} leads available adds in your league. "
        "Open Waivers for FAAB bands, drop suggestions, and Start/Sit."
    )
    return title, body


# Absolute reference band mapping a composite pickup score onto bid intensity,
# independent of the displayed waiver list (#4). Calibrated to
# waiver_pickup_score: players near the min_value floor sit around FAAB_SCORE_LOW
# while strong every-week adds land near FAAB_SCORE_HIGH. Because these are fixed,
# changing a position filter or paging the list can't move the same player's bid.
FAAB_SCORE_LOW = 45.0
FAAB_SCORE_HIGH = 190.0

# Season-phase intensity multipliers: early-season adds have a whole season to
# pay off (bid up a touch); late-season fliers rarely do (bid down) unless they
# are immediate help, which the score already reflects.
_FAAB_PHASE_MULT = {"early": 1.1, "mid": 1.0, "late": 0.85, "playoffs": 0.8}


def faab_intensity(pickup_score, *, need_mult: float = 1.0,
                   handcuff_upside: float = 0.0, role_duration: float = 1.0,
                   season_phase: str = "mid") -> float:
    """0..1 bid intensity from an ABSOLUTE score reference (never the displayed
    list, #4), nudged by roster need, handcuff upside, expected role duration,
    and season timing.

    ``role_duration`` is a 0..1 estimate of how long the add stays useful (a
    one-week injury fill is low; a player earning a lasting role is high) so a
    short-term plug doesn't command a season-long price.
    """
    try:
        score = float(pickup_score)
    except (TypeError, ValueError):
        return 0.0
    t = _clamp01((score - FAAB_SCORE_LOW) / (FAAB_SCORE_HIGH - FAAB_SCORE_LOW))
    try:
        need = float(need_mult or 1.0)
    except (TypeError, ValueError):
        need = 1.0
    t *= 1.0 + min(max(need - 1.0, 0.0), 0.25)
    try:
        cuff = max(0.0, float(handcuff_upside or 0.0))
    except (TypeError, ValueError):
        cuff = 0.0
    t += cuff * 0.15
    try:
        dur = _clamp01(float(role_duration))
    except (TypeError, ValueError):
        dur = 1.0
    # A short role caps how aggressive the bid gets (0.6..1.0 of intensity).
    t *= 0.6 + 0.4 * dur
    t *= _FAAB_PHASE_MULT.get(str(season_phase or "mid").lower(), 1.0)
    return _clamp01(t)


def _faab_pct_bands(intensity: float) -> "tuple[int, int, int]":
    """Low / target / stretch as % of the budget denominator, from 0..1 intensity.
    Modest by design (waiver fliers, not auction pieces): target tops out ~26%."""
    center = 1.0 + (intensity ** 1.6) * 25.0
    low = max(0, int(round(center * 0.7)))
    target = max(low, int(round(center)))
    high = max(target + 1, min(50, int(round(center * 1.15)) + 1))
    return low, target, high


def _faab_rationale(intensity: float, need: float, cuff: float,
                    role_duration: float) -> str:
    bits = []
    if intensity >= 0.7:
        bits.append("top target on your wire")
    elif intensity >= 0.4:
        bits.append("solid add vs the market")
    else:
        bits.append("speculative flier")
    if need > 1.05:
        bits.append("fills a roster need")
    if cuff >= 0.35:
        bits.append("handcuff upside")
    if role_duration <= 0.4:
        bits.append("short-term role — keep the bid modest")
    return "; ".join(bits)


def faab_bid_bands(pickup_score: float, score_min: float = 0.0,
                   score_range: float = 1.0, need_mult: float = 1.0,
                   handcuff_upside: float = 0.0, role_duration: float = 1.0,
                   season_phase: str = "mid") -> dict:
    """Low / target / stretch FAAB **% of budget** for a waiver target.

    List-independent (#4): the bid comes from the player's own absolute score and
    context, so ``score_min`` / ``score_range`` are accepted only for backward
    compatibility and are ignored — filtering or paging the list never changes a
    player's suggested bid. Always returns ints suitable for UI chips.
    """
    try:
        score = float(pickup_score)
    except (TypeError, ValueError):
        return {"faab_low": 0, "faab_target": 1, "faab_high": 2,
                "faab_rationale": "Baseline flier bid"}
    try:
        need = float(need_mult or 1.0)
    except (TypeError, ValueError):
        need = 1.0
    try:
        cuff = max(0.0, float(handcuff_upside or 0.0))
    except (TypeError, ValueError):
        cuff = 0.0
    try:
        dur = _clamp01(float(role_duration))
    except (TypeError, ValueError):
        dur = 1.0
    intensity = faab_intensity(score, need_mult=need, handcuff_upside=cuff,
                               role_duration=dur, season_phase=season_phase)
    low, target, high = _faab_pct_bands(intensity)
    return {
        "faab_low": low,
        "faab_target": target,
        "faab_high": high,
        "faab_rationale": _faab_rationale(intensity, need, cuff, dur),
    }


def faab_recommendation(pickup_score, *, budget_total=None, budget_remaining=None,
                        waiver_type: str = "faab", season_phase: str = "mid",
                        need_mult: float = 1.0, handcuff_upside: float = 0.0,
                        role_duration: float = 1.0) -> dict:
    """List-independent FAAB / waiver-priority claim guidance (#4).

    Returns a dict describing how hard to bid, with the percentage denominator
    made explicit and dollar amounts only when a real budget is known:

      * ``mode``            — "faab" or "waiver_priority".
      * ``pct_low/target/high`` and ``pct_denominator`` — % of *remaining* budget
        when a remaining figure is supplied, otherwise % of the *season* budget.
        The denominator is named so the UI never shows an ambiguous "%".
      * ``low/target/high`` and ``budget_basis`` — dollar amounts, capped at the
        remaining budget, only when a total budget is available. When it isn't,
        these are ``None`` and only clearly-labeled percentages are shown (no
        fabricated dollars).
      * ``claim_guidance`` — qualitative advice for waiver-priority leagues that
        don't use FAAB at all.
      * ``rationale`` and ``heuristic: True`` — these are heuristic estimates, not
        validated market prices; a weak wire never auto-creates an expensive bid.
    """
    intensity = faab_intensity(pickup_score, need_mult=need_mult,
                               handcuff_upside=handcuff_upside,
                               role_duration=role_duration, season_phase=season_phase)
    pct_low, pct_target, pct_high = _faab_pct_bands(intensity)
    try:
        need = float(need_mult or 1.0)
    except (TypeError, ValueError):
        need = 1.0
    try:
        cuff = max(0.0, float(handcuff_upside or 0.0))
    except (TypeError, ValueError):
        cuff = 0.0
    rationale = _faab_rationale(intensity, need, cuff, _clamp01(float(role_duration)))

    if str(waiver_type or "").lower() in ("priority", "waiver_priority", "rolling_priority"):
        if intensity >= 0.7:
            claim = "Use your top waiver claim"
        elif intensity >= 0.4:
            claim = "Worth a mid-priority claim"
        else:
            claim = "Only if it costs a low claim"
        return {
            "mode": "waiver_priority",
            "pct_low": None, "pct_target": None, "pct_high": None,
            "pct_denominator": None,
            "low": None, "target": None, "high": None, "budget_basis": None,
            "claim_guidance": claim,
            "rationale": rationale,
            "heuristic": True,
        }

    # Dollar denominator: prefer remaining budget (what you can actually spend),
    # else the season budget. Name whichever we used.
    denom = None
    denom_label = None
    if budget_remaining is not None:
        try:
            denom = max(0.0, float(budget_remaining))
            denom_label = "remaining_budget"
        except (TypeError, ValueError):
            denom = None
    if denom is None and budget_total is not None:
        try:
            denom = max(0.0, float(budget_total))
            denom_label = "season_budget"
        except (TypeError, ValueError):
            denom = None

    dollars = {"low": None, "target": None, "high": None}
    if denom is not None:
        cap = denom
        dollars = {
            "low": min(int(round(pct_low / 100.0 * denom)), int(cap)),
            "target": min(int(round(pct_target / 100.0 * denom)), int(cap)),
            "high": min(int(round(pct_high / 100.0 * denom)), int(cap)),
        }
        # Preserve ordering after the remaining-budget cap.
        dollars["target"] = max(dollars["low"], dollars["target"])
        dollars["high"] = max(dollars["target"], dollars["high"])

    return {
        "mode": "faab",
        "pct_low": pct_low, "pct_target": pct_target, "pct_high": pct_high,
        "pct_denominator": denom_label or "season_budget",
        "low": dollars["low"], "target": dollars["target"], "high": dollars["high"],
        "budget_basis": denom_label,
        "claim_guidance": None,
        "rationale": rationale,
        "heuristic": True,
    }


def replacement_levels(values_by_pos: dict, cutoffs: dict) -> dict:
    """Replacement-level value per position: the value at the position's roster
    cutoff rank (#4). ``values_by_pos``: {pos: [values...]}; ``cutoffs``: {pos:
    rank}. Positions with no cutoff or no values are omitted.
    """
    out: dict = {}
    for pos, vals in (values_by_pos or {}).items():
        cut = int((cutoffs or {}).get(pos, 0) or 0)
        if cut <= 0 or not vals:
            continue
        sv = sorted((float(v) for v in vals if v is not None), reverse=True)
        if not sv:
            continue
        idx = min(cut, len(sv)) - 1
        out[pos] = sv[max(0, idx)]
    return out


def scarcity_multiplier(position, value, replacement_by_pos: dict,
                        w: WaiverWeights = WEIGHTS) -> float:
    """Multiplier (1 .. 1+scarcity_max_bonus) rewarding value above the position's
    replacement level (#4) — the same nominal value is worth more at a scarce
    position where the drop-off past the starters is steeper."""
    repl = (replacement_by_pos or {}).get(position)
    try:
        v = float(value or 0)
        repl = float(repl) if repl is not None else 0.0
    except (TypeError, ValueError):
        return 1.0
    if repl <= 0:
        return 1.0
    edge = _clamp01((v - repl) / repl)   # fraction above replacement, capped +100%
    return 1.0 + w.scarcity_max_bonus * edge


# ---------------------------------------------------------------------------
# Composite score + signal
# ---------------------------------------------------------------------------

def waiver_pickup_score(c: dict, waiver_breakout: dict,
                        prime_max: dict = WAIVER_PRIME_MAX,
                        w: WaiverWeights = WEIGHTS) -> float:
    """Composite waiver-pickup score.

    ``c`` is a candidate dict. Recognized keys: value, age, position,
    rank_change_7d, player_id, and (all optional, default to a no-op when
    absent) ros_ppg, own_proj_ppg, trend_windows, usage_stat, usage_delta,
    injured_ahead, vacated, healthy_ahead, vacated_volume_weight,
    injury_freshness, need_mult, scarcity_mult, self_status,
    schedule_ease_rank, schedule_total.
    """
    # League roster membership is authoritative.  Callers may omit this key for
    # legacy/offseason pools, but an explicitly unavailable player can never be
    # promoted by model signals.
    if c.get("available") is False or c.get("is_available") is False:
        return 0.0
    try:
        val = float(c.get("value") or 0)
    except (TypeError, ValueError):
        val = 0.0
    age = c.get("age") or 0
    pos = c.get("position")
    bscore = waiver_breakout.get(c.get("player_id"), 0) or 0
    prime = prime_max.get(pos, 28)

    # Base worth: saturating dynasty value + forward projected production (#1/#5).
    value_pts = value_component(val, w)
    proj_pts = projection_component(c.get("ros_ppg"), w)

    # --- Opportunity signals (correlated -> combined with diminishing returns) --
    # Prefer the projection-aware `vacated` list (status + projected PPG); fall
    # back to bare `injured_ahead` statuses (which use a baseline PPG) so callers
    # that don't supply projections still work.
    injury_pts = depth_chart_vacancy_score(
        c.get("vacated") if c.get("vacated") is not None else c.get("injured_ahead"),
        healthy_ahead=int(c.get("healthy_ahead") or 0),
        volume_weight=float(c.get("vacated_volume_weight") or 1.0),
        freshness=float(c.get("injury_freshness") or 1.0),
        w=w,
    )
    # Role-transfer guard (#2): if the candidate's own forward projection already
    # reflects the vacated role (they've taken over), the injury upside is priced
    # in — fade it so we don't double-count. Full credit only for un-inherited
    # opportunity (own projection still ~0).
    own_ppg = c.get("own_proj_ppg")
    if own_ppg is not None:
        role_ppg = max((float(v.get("proj_ppg") or 0)
                        for v in (c.get("vacated") or []) if isinstance(v, dict)), default=0.0)
        if role_ppg > 0:
            try:
                injury_pts *= _clamp01(1.0 - float(own_ppg) / role_ppg)
            except (TypeError, ValueError):
                pass
    usage_pts = min(usage_ratio(c.get("usage_stat"), c.get("usage_delta")) * w.usage_per_ratio,
                    w.usage_max)
    breakout_pts = min(bscore * w.breakout_per, w.breakout_max)
    # Unexpected big game folded in as a fourth (correlated) opportunity signal.
    try:
        big_game_pts = min(max(0.0, float(c.get("big_game_pts") or 0)), w.big_game_max)
    except (TypeError, ValueError):
        big_game_pts = 0.0
    opp = sorted([injury_pts, usage_pts, breakout_pts, big_game_pts], reverse=True)
    opportunity_pts = (opp[0] + w.opp_second * opp[1]
                       + w.opp_third * opp[2] + w.opp_fourth * opp[3])  # (#6)

    # Weekly rank trend, blended across available windows and noise-shrunk (#7),
    # then discounted by positional depth so a deep player's dense (noisy) overall
    # rank swings don't inflate the score.
    rank_chg = blended_trend(c.get("trend_windows") or {"7d": c.get("rank_change_7d")}, w)
    _depth = _pos_depth(c)
    if _depth and _depth > 0:
        rank_chg *= w.trend_depth_ref / (_depth + w.trend_depth_ref)
    if rank_chg > 0:
        trend_pts = min(rank_chg * w.trend_up_per, w.trend_up_max)
    else:
        trend_pts = max(rank_chg * w.trend_down_per, w.trend_down_floor)

    # Upcoming schedule ease (#3): a soft slate is a small nudge, a brutal one a
    # small penalty.
    sched_pts = schedule_bonus(c.get("schedule_ease_rank"), c.get("schedule_total"), w)

    # Age: smooth youth reward / past-prime decay, both bounded.
    if not age:
        age_pts = 0.0
    else:
        gap = prime - age  # + = younger than prime
        if gap >= 0:
            age_pts = min(w.age_base + gap * w.age_youth_per, w.age_youth_max)
        else:
            age_pts = max(w.age_base + gap * w.age_decay_per, w.age_floor)

    raw = value_pts + proj_pts + opportunity_pts + trend_pts + sched_pts + age_pts

    # Distinct opportunity-surprise overlay. It deliberately rewards earned
    # volume and sustainability, not touchdowns; correlated breakout/usage
    # signals remain in the diminishing-returns bucket above.
    surprise = c.get("waiver_surprise") or {}
    if isinstance(surprise, dict):
        usage_surprise = _clamp01(float(surprise.get("unexpected_usage") or 0))
        sustainable = _clamp01(float(surprise.get("sustainability") or 0))
        role_change = _clamp01(float(surprise.get("role_change") or 0))
        surprise_bonus = 18.0 * (0.55 * usage_surprise + 0.45 * role_change) * sustainable
        low_volume_tds = max(0.0, float(surprise.get("unsustainable_production") or 0))
        temporary = max(0.0, float(surprise.get("temporary_role") or 0))
        raw += surprise_bonus - min(15.0, 10.0 * low_volume_tds + 8.0 * temporary)

    # Roster-aware (#4a): a position of real need to the viewer is worth more.
    raw *= float(c.get("need_mult") or 1.0)
    # Positional scarcity (#4b): value above replacement is worth more at a
    # scarce position.
    raw *= float(c.get("scarcity_mult") or 1.0)
    # Candidate's own health (#2): a hurt backup isn't this week's add.
    raw *= self_injury_multiplier(c.get("self_status"))
    return raw


def waiver_signal(c: dict, waiver_breakout: dict,
                  prime_max: dict = WAIVER_PRIME_MAX,
                  w: WaiverWeights = WEIGHTS,
                  fast_thr=None, up_thr=None) -> "tuple[str, str]":
    """Return (badge_class, label) describing why a candidate is interesting.

    Shared by both waiver surfaces. Branches that read data a surface doesn't
    provide (usage, depth chart) are simply no-ops there. ``fast_thr``/``up_thr``
    are pool-relative trend thresholds (see adaptive_trend_thresholds); when
    given they combine with the depth-based bar so a player must be both a
    meaningful mover for their depth AND a top mover in the shown set to "rise".
    """
    rank_chg = c.get("rank_change_7d") or 0
    age = c.get("age") or 0
    pos = c.get("position")
    bscore = waiver_breakout.get(c.get("player_id"), 0) or 0
    prime = prime_max.get(pos, 28)
    healthy_ahead = int(c.get("healthy_ahead") or 0)
    try:
        val = float(c.get("value") or 0)
    except (TypeError, ValueError):
        val = 0.0

    # rank_change_7d is overall-rank movement, which is dense/noisy for deep
    # players — a WR89 drifts 8+ overall spots on nothing. Require a move that
    # scales with positional depth so "Rising Fast" stays meaningful.
    _depth = _pos_depth(c)
    _fast_thr = max(w.trend_fast_floor, w.trend_fast_frac * _depth) if _depth else w.trend_fast_floor
    _up_thr = max(w.trend_up_floor, w.trend_up_frac * _depth) if _depth else w.trend_up_floor
    # Combine with the pool-relative bar so a trend-sorted list doesn't label
    # everything "Rising Fast": a player must clear both bars.
    if fast_thr is not None:
        _fast_thr = max(_fast_thr, float(fast_thr))
    if up_thr is not None:
        _up_thr = max(_up_thr, float(up_thr))

    # A candidate who is himself out isn't a "target" — label the reason.
    if is_seriously_hurt(c.get("self_status")):
        return ("signal-aging", "Injured")

    inj_sev = max((VACANCY_SEVERITY.get(str(s).upper(), 0.0)
                   for s in (c.get("injured_ahead") or [])), default=0.0)

    # A confirmed/likely absence ahead — and the candidate genuinely next in line
    # (no healthy body still blocking) — is the most actionable signal.
    if inj_sev >= VACANCY_STRONG and healthy_ahead == 0:
        return ("signal-injury", "Next Man Up")
    if usage_ratio(c.get("usage_stat"), c.get("usage_delta")) >= 1.0:
        return ("signal-usage", "Usage Spike")
    if bscore >= 55:
        return ("signal-breakout", "Breakout")
    if rank_chg >= _fast_thr:
        return ("signal-rising", "Rising Fast")
    if rank_chg >= _up_thr:
        return ("signal-rising", "Trending Up")
    # A softer injury bump: a vacancy exists but the candidate isn't cleanly next
    # up (a healthy body remains) or the injury is only Questionable.
    if inj_sev > 0 and healthy_ahead <= 1:
        return ("signal-injury-soft", "Bumped Up")
    if age and age < prime - 2 and val >= 300:
        return ("signal-value", "Value Play")
    # Streamer: a startable weekly projection into a favorable upcoming matchup.
    # This is the useful label for the productive-but-unexciting wire add (esp. an
    # aging vet or a bye/injury fill-in) — actionable *this week*, not a dynasty
    # sell. "Sell Window" is intentionally gone: you can't sell a free agent.
    try:
        _ros = float(c.get("ros_ppg")) if c.get("ros_ppg") is not None else None
    except (TypeError, ValueError):
        _ros = None
    _ease = c.get("schedule_ease_rank")
    _tot = c.get("schedule_total")
    _favorable = False
    try:
        if _ease and _tot and float(_tot) > 1:
            _favorable = (float(_ease) - 1.0) / (float(_tot) - 1.0) <= 0.4
    except (TypeError, ValueError):
        _favorable = False
    if _ros is not None and _ros >= 9.0 and _favorable:
        return ("signal-usage", "Streamer")
    return ("signal-hold", "Available")


# ---------------------------------------------------------------------------
# Roster need (#4) — pure helper the API feeds from the viewer's roster
# ---------------------------------------------------------------------------

def positional_need_scores(roster_counts: dict, starter_reqs: dict) -> dict:
    """Map each position to a 0..1 need score for the viewer.

    ``roster_counts``: how many players the viewer rosters at each position.
    ``starter_reqs``: how many that position ideally fills (starters + a little
    depth). Need rises as the viewer falls short of the requirement.
    """
    out: dict = {}
    for pos, req in (starter_reqs or {}).items():
        req = float(req or 0)
        if req <= 0:
            continue
        have = float((roster_counts or {}).get(pos, 0))
        out[pos] = _clamp01((req - have) / req)
    return out


def need_multiplier(position, need_scores: dict, w: WaiverWeights = WEIGHTS) -> float:
    """Convert a position's 0..1 need score into a score multiplier (1 .. 1+bonus)."""
    n = (need_scores or {}).get(position)
    if n is None:
        return 1.0
    return 1.0 + w.need_max_bonus * _clamp01(n)


# ---------------------------------------------------------------------------
# Recommendation horizons (#2)
# ---------------------------------------------------------------------------

# The three horizons the surfaces offer. Each meaningfully reweights the model:
# immediate help leans on forward production and mutes age / long-term value;
# a stash leans on value/age/upside and mutes this-week injury opportunity.
HORIZONS = ("this_week", "four_week", "stash")


def horizon_weights(horizon, base: WaiverWeights = WEIGHTS, *,
                    dynasty: bool = False) -> WaiverWeights:
    """Return a WaiverWeights tuned for the selected horizon (#2).

    * ``this_week``  — redraft immediate help: forward projection dominates, and
      direct age / long-term value bonuses are minimized.
    * ``four_week``  — the balanced default.
    * ``stash``      — value, youth, and role upside matter; this-week injury
      opportunity is de-emphasized (a short-term vacancy doesn't make a stash),
      and in dynasty leagues age / value are weighted even more.
    """
    h = str(horizon or "four_week").lower()
    if h == "this_week":
        return _dc_replace(
            base,
            value_max=base.value_max * 0.45,
            proj_per_ppg=base.proj_per_ppg * 1.4,
            proj_max=base.proj_max * 1.4,
            age_youth_max=6.0, age_youth_per=0.5,
            age_decay_per=1.5, age_floor=-4.0,
        )
    if h == "stash":
        return _dc_replace(
            base,
            value_max=base.value_max * (1.6 if dynasty else 1.2),
            proj_per_ppg=base.proj_per_ppg * 0.5,
            proj_max=base.proj_max * 0.5,
            age_youth_max=base.age_youth_max * (1.5 if dynasty else 1.1),
            age_youth_per=base.age_youth_per * (1.4 if dynasty else 1.0),
            injury_max=base.injury_max * 0.5,   # short-term vacancy ≠ a stash reason
        )
    return base


# ---------------------------------------------------------------------------
# Candidate discovery floor (#5)
# ---------------------------------------------------------------------------

# Engine breakout score at/above which a low-value player has a *credible*
# opportunity signal worth admitting to the pool (matches the "Breakout" badge).
BREAKOUT_FLOOR = 55.0


def credible_opportunity(c: dict, waiver_breakout: dict = None,
                         w: WaiverWeights = WEIGHTS) -> bool:
    """Does the candidate have a *real* opportunity signal (not just age or noisy
    rank movement)? Used to let promising low-value players bypass the static
    value floor (#5) without admitting every deep player on noise.

    Credible signals: a confirmed usage spike, a confirmed injury vacancy with
    the candidate genuinely next in line, an engine breakout at/above the floor,
    or an explicit verified role-change flag. Age and raw rank_change_7d are
    deliberately excluded — they are the noisy signals the floor exists to block.
    """
    waiver_breakout = waiver_breakout or {}
    # Real usage spike (last-3 vs season at/above the stat threshold).
    if usage_ratio(c.get("usage_stat"), c.get("usage_delta")) >= 1.0:
        return True
    # Confirmed / likely absence ahead, candidate next in line.
    inj_sev = max((VACANCY_SEVERITY.get(str(s).upper(), 0.0)
                   for s in (c.get("injured_ahead") or [])), default=0.0)
    if inj_sev >= VACANCY_STRONG and int(c.get("healthy_ahead") or 0) == 0:
        return True
    # Engine breakout.
    try:
        if float(waiver_breakout.get(c.get("player_id"), 0) or 0) >= BREAKOUT_FLOOR:
            return True
    except (TypeError, ValueError):
        pass
    # Explicit verified role change / big-game discovery flag set by the caller.
    if c.get("verified_role_change") or c.get("big_game_priority"):
        return True
    return False


def passes_candidate_floor(c: dict, waiver_breakout: dict = None,
                           min_value: float = None,
                           w: WaiverWeights = WEIGHTS) -> bool:
    """True when a candidate clears the value floor OR carries a credible
    opportunity signal (#5). This is the single admission gate the discovery
    layer should apply, so a real breakout/vacancy/usage story surfaces even at
    low static value, while age/rank-noise-only players stay filtered out."""
    floor = w.min_value if min_value is None else float(min_value)
    try:
        val = float(c.get("value") or 0)
    except (TypeError, ValueError):
        val = 0.0
    if val >= floor:
        return True
    return credible_opportunity(c, waiver_breakout, w)


# ======================================================================
# From utils/waiver_lineup.py
# ======================================================================

"""Roster-aware waiver evaluation: does adding this player actually improve the
viewer's *starting lineup*, and if the roster is full, who should come off?

This replaces player-count "roster need" as the primary personalization signal
(item 1). Instead of asking "do you have few RBs?", it asks the question that
matters: "does your best legal lineup score more with this player in it — this
week and over the next few weeks — and what does it cost you to make room?"

Everything is pure and reuses the shared optimal-lineup solver
(``utils.optimal_lineup.compute_optimal_lineup``), so league scoring, position
eligibility, FLEX / Superflex / TE-premium, and roster limits are honored
exactly the way Start/Sit and the optimal-lineup page honor them. Fantasy points
are supplied already-scored by the caller; a bye or an injured player simply
projects ~0, so they fall out of the optimal lineup on their own. Bench
production is never counted as a starting-lineup gain — the delta is computed on
optimal *starter* totals only.

The evaluator returns one of five explicit outcomes (item 3):

  * ``add``            — start-worthy now; you have an open slot, no drop needed.
  * ``add_drop``       — start-worthy now; roster full, cut a specific player.
  * ``stash``          — no lineup gain now, but worth a bench spot (upside).
  * ``hold``           — no worthwhile transaction; your roster is better as is.
  * ``cannot_evaluate``— not enough data to judge a safe move.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from utils.lineups import canonicalize_slot, slot_eligible_positions
from utils.lineups import compute_optimal_lineup

# A starting-lineup gain below this many projected points is treated as noise —
# not a real upgrade (item 3: never call something an upgrade on static value
# alone; require an actual lineup improvement).
MIN_MEANINGFUL_GAIN = 0.5


@dataclass
class PickupEvaluation:
    outcome: str
    week_gain: float                    # this-week optimal-starter point gain
    horizon_gain: float                 # summed gain over covered future weeks
    horizon_weeks_covered: int
    horizon_weeks_missing: int
    starts_now: bool
    drop_pid: Optional[str] = None      # stable id of the suggested cut (or None)
    replaces_pid: Optional[str] = None  # starter the pickup benches (may != drop)
    drop_points: Optional[float] = None # dropped player's own projected points (opportunity cost)
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "outcome": self.outcome,
            "week_gain": round(self.week_gain, 2),
            "horizon_gain": round(self.horizon_gain, 2),
            "horizon_weeks_covered": self.horizon_weeks_covered,
            "horizon_weeks_missing": self.horizon_weeks_missing,
            "starts_now": self.starts_now,
            "drop_pid": self.drop_pid,
            "replaces_pid": self.replaces_pid,
            "drop_points": (round(self.drop_points, 2) if self.drop_points is not None else None),
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Lineup helpers (reuse the shared solver; add lock-aware pre-assignment)
# ---------------------------------------------------------------------------

def _slot_counts(roster_positions: Iterable) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in roster_positions or []:
        c = canonicalize_slot(s)
        if c:
            counts[c] = counts.get(c, 0) + 1
    return counts


# Assign a locked starter to the most restrictive eligible slot first, so a
# locked QB doesn't consume a FLEX while a QB slot sits open.
_ASSIGN_ORDER = ["QB", "RB", "WR", "TE", "K", "DEF",
                 "RB_WR", "WR_TE", "RB_TE", "FLEX", "SUPER_FLEX"]


def _assign_locked(locked: List[str], positions: Dict[str, str],
                   slot_counts: Dict[str, int]) -> "Optional[Dict[str, int]]":
    """Consume slots for locked starters; return the remaining slot counts, or
    None if a locked player can't be legally placed (caller then ignores locks)."""
    remaining = dict(slot_counts)
    for pid in locked:
        pos = str(positions.get(pid) or "").upper()
        placed = False
        for slot in _ASSIGN_ORDER:
            if remaining.get(slot, 0) <= 0:
                continue
            if pos in slot_eligible_positions(slot):
                remaining[slot] -= 1
                placed = True
                break
        if not placed:
            return None
    return remaining


def _counts_to_positions(counts: Dict[str, int]) -> List[str]:
    out: List[str] = []
    for slot, n in counts.items():
        out.extend([slot] * int(n))
    return out


def optimal_starters_and_points(pts_map: Dict[str, float],
                                positions: Dict[str, str],
                                roster_positions: Iterable,
                                pids: Iterable,
                                locked_starter_pids: Optional[Iterable] = None
                                ) -> "tuple[set, float]":
    """Best legal lineup over ``pids`` and its total. Locked starters (already
    played / lineup-locked) are forced into the lineup first so the optimizer
    can't bench them; the rest fill the remaining slots optimally."""
    pids = [str(p) for p in pids]
    locked = [str(p) for p in (locked_starter_pids or []) if str(p) in pids]
    if not locked:
        return compute_optimal_lineup(pts_map, positions, list(roster_positions), pids)

    counts = _slot_counts(roster_positions)
    remaining = _assign_locked(locked, positions, counts)
    if remaining is None:
        # Locks can't be honored legally; fall back to a plain optimal lineup.
        return compute_optimal_lineup(pts_map, positions, list(roster_positions), pids)
    rest = [p for p in pids if p not in set(locked)]
    rest_set, rest_pts = compute_optimal_lineup(
        pts_map, positions, _counts_to_positions(remaining), rest)
    starters = set(locked) | rest_set
    total = sum(float(pts_map.get(p) or 0) for p in starters)
    return starters, round(total, 2)


def _starter_slots(roster_positions: Iterable) -> int:
    """Number of real starting slots (excludes bench / IR / taxi)."""
    n = 0
    for s in roster_positions or []:
        c = canonicalize_slot(s)
        if c and c not in ("BN", "IR", "TAXI"):
            n += 1
    return n


# ---------------------------------------------------------------------------
# Single-week pickup evaluation
# ---------------------------------------------------------------------------

def evaluate_pickup_week(candidate_pid: str,
                         candidate_pos: str,
                         roster_pids: Iterable,
                         pts_map: Dict[str, float],
                         positions: Dict[str, str],
                         roster_positions: Iterable,
                         *,
                         droppable_pids: Optional[Iterable] = None,
                         locked_starter_pids: Optional[Iterable] = None,
                         ) -> dict:
    """One week's lineup math for adding ``candidate_pid`` (+ best drop if full).

    Returns a dict with: ``before``, ``after_add`` (add without drop, when a slot
    is open), ``best_after`` (best achievable including a drop), ``drop_pid``,
    ``drop_points``, ``starts_now``, ``replaces_pid``, and ``roster_full``.
    """
    candidate_pid = str(candidate_pid)
    roster_pids = [str(p) for p in roster_pids]
    positions = dict(positions)
    positions.setdefault(candidate_pid, str(candidate_pos or "").upper())
    pts_map = dict(pts_map)

    starter_slots = _starter_slots(roster_positions)
    roster_full = len(roster_pids) >= starter_slots + _bench_slots(roster_positions) \
        if _bench_slots(roster_positions) is not None else len(roster_pids) >= starter_slots

    starters_before, before = optimal_starters_and_points(
        pts_map, positions, roster_positions, roster_pids, locked_starter_pids)

    # Add without dropping (only legal when the roster isn't full).
    after_add = None
    starters_add: set = set()
    if not roster_full:
        starters_add, after_add = optimal_starters_and_points(
            pts_map, positions, roster_positions, roster_pids + [candidate_pid],
            locked_starter_pids)

    # Best add-with-drop over the eligible drop pool.
    drop_pids = [str(p) for p in (droppable_pids
                                  if droppable_pids is not None else roster_pids)]
    locked = {str(p) for p in (locked_starter_pids or [])}
    drop_pids = [p for p in drop_pids if p != candidate_pid and p not in locked]

    best_drop = None
    best_after_drop = None
    best_starters_drop: set = set()
    for d in drop_pids:
        post = [p for p in roster_pids if p != d] + [candidate_pid]
        s_after, pts_after = optimal_starters_and_points(
            pts_map, positions, roster_positions, post,
            [p for p in (locked_starter_pids or []) if p != d])
        # Prefer the highest resulting lineup; tie-break by cutting the least
        # productive player (lowest own projected points).
        key = (pts_after, -float(pts_map.get(d) or 0))
        if best_after_drop is None or key > (best_after_drop, -float(pts_map.get(best_drop) or 0)):
            best_after_drop = pts_after
            best_drop = d
            best_starters_drop = s_after

    # Choose the representative "after" lineup for who-starts / who's-replaced.
    if not roster_full and after_add is not None:
        best_after = after_add
        starters_after = starters_add
        drop_pid = None
    else:
        best_after = best_after_drop
        starters_after = best_starters_drop
        drop_pid = best_drop

    starts_now = candidate_pid in (starters_after or set())
    replaces_pid = None
    if starts_now:
        # A starter who was in the lineup before but isn't now, preferring one who
        # is NOT the dropped player (the pickup bumped a different weak starter);
        # if only the dropped starter left, the pickup takes the dropped slot.
        pushed = [p for p in starters_before
                  if p != candidate_pid and p not in (starters_after or set())]
        non_drop = [p for p in pushed if p != drop_pid]
        if non_drop:
            replaces_pid = non_drop[0]
        elif pushed:
            replaces_pid = pushed[0]

    return {
        "before": before,
        "after_add": after_add,
        "best_after": best_after,
        "drop_pid": drop_pid,
        "drop_points": (float(pts_map.get(drop_pid) or 0) if drop_pid else None),
        "starts_now": starts_now,
        "replaces_pid": replaces_pid,
        "roster_full": roster_full,
    }


def _bench_slots(roster_positions: Iterable) -> Optional[int]:
    n = 0
    seen = False
    for s in roster_positions or []:
        if canonicalize_slot(s) == "BN":
            n += 1
            seen = True
    return n if seen else None


# ---------------------------------------------------------------------------
# Multi-week evaluation + outcome classification (items 1, 2, 3)
# ---------------------------------------------------------------------------

def evaluate_pickup(candidate_pid: str,
                    candidate_pos: str,
                    roster_pids: Iterable,
                    weekly: List[dict],
                    positions: Dict[str, str],
                    roster_positions: Iterable,
                    *,
                    droppable_pids: Optional[Iterable] = None,
                    locked_starter_pids: Optional[Iterable] = None,
                    speculative_upside: float = 0.0,
                    ) -> PickupEvaluation:
    """Evaluate a pickup across one or more weeks and classify the outcome.

    ``weekly`` is a list of per-week dicts, week 0 first::

        {"pts_map": {pid: proj_pts}, "covered": bool, "bye": bool}

    ``covered=False`` marks a week with no usable projections; it is reported as
    missing coverage rather than silently scored as zero (item 2). Week 0 drives
    the this-week decision; the covered weeks are summed for the horizon gain.

    ``speculative_upside`` (0..1) lets a promising player surface as a ``stash``
    even with no current lineup gain, without ever claiming an improvement it
    doesn't produce (item 1: keep speculative upside separate).
    """
    weekly = list(weekly or [])
    if not weekly or not str(candidate_pid):
        return PickupEvaluation("cannot_evaluate", 0.0, 0.0, 0, 0, False,
                                detail={"reason": "no_weekly_data"})

    covered = [w for w in weekly if w.get("covered", True) and not w.get("bye")]
    missing = [w for w in weekly if not w.get("covered", True)]

    wk0 = weekly[0]
    if not wk0.get("pts_map"):
        return PickupEvaluation("cannot_evaluate", 0.0, 0.0, 0, len(missing), False,
                                detail={"reason": "no_week0_projection"})

    wk0_res = evaluate_pickup_week(
        candidate_pid, candidate_pos, roster_pids, wk0.get("pts_map") or {},
        positions, roster_positions,
        droppable_pids=droppable_pids, locked_starter_pids=locked_starter_pids)

    if wk0_res["best_after"] is None:
        return PickupEvaluation("cannot_evaluate", 0.0, 0.0, 0, len(missing),
                                False, detail={"reason": "no_drop_or_slot"})

    week_gain = float(wk0_res["best_after"]) - float(wk0_res["before"])
    drop_pid = wk0_res["drop_pid"]

    # Horizon: re-evaluate each covered week using the SAME drop chosen for week 0
    # (a consistent transaction), so we measure the real multi-week effect rather
    # than cherry-picking a different cut each week.
    horizon_gain = 0.0
    weeks_covered = 0
    for w in covered:
        pm = w.get("pts_map") or {}
        if not pm:
            continue
        res = evaluate_pickup_week(
            candidate_pid, candidate_pos, roster_pids, pm, positions, roster_positions,
            droppable_pids=[drop_pid] if drop_pid else droppable_pids,
            locked_starter_pids=locked_starter_pids)
        if res["best_after"] is None:
            continue
        horizon_gain += float(res["best_after"]) - float(res["before"])
        weeks_covered += 1

    starts_now = bool(wk0_res["starts_now"])
    outcome = _classify_outcome(week_gain, horizon_gain, starts_now,
                                wk0_res["roster_full"], drop_pid,
                                speculative_upside, wk0_res["drop_points"])

    return PickupEvaluation(
        outcome=outcome,
        week_gain=max(0.0, week_gain) if outcome in ("add", "add_drop") else week_gain,
        horizon_gain=horizon_gain,
        horizon_weeks_covered=weeks_covered,
        horizon_weeks_missing=len(missing),
        starts_now=starts_now,
        drop_pid=drop_pid if outcome == "add_drop" else None,
        replaces_pid=wk0_res["replaces_pid"] if starts_now else None,
        drop_points=wk0_res["drop_points"] if outcome == "add_drop" else None,
        detail={
            "before": wk0_res["before"],
            "after_add": wk0_res["after_add"],
            "best_after": wk0_res["best_after"],
            "roster_full": wk0_res["roster_full"],
        },
    )


def _classify_outcome(week_gain: float, horizon_gain: float, starts_now: bool,
                      roster_full: bool, drop_pid: Optional[str],
                      speculative_upside: float,
                      drop_points: Optional[float]) -> str:
    """Map the lineup math onto one of the five explicit outcomes (item 3)."""
    meaningful = (week_gain >= MIN_MEANINGFUL_GAIN
                  or horizon_gain >= MIN_MEANINGFUL_GAIN)
    if meaningful and starts_now:
        if not roster_full:
            return "add"
        if drop_pid is not None:
            return "add_drop"
        # Roster full but no legal/eligible drop found.
        return "cannot_evaluate"
    # No lineup gain now. A promising player can still be a bench stash.
    if speculative_upside >= 0.5:
        # Only if we can make room (open slot, or a droppable spare exists).
        if not roster_full or drop_pid is not None:
            return "stash"
    return "hold"


# ======================================================================
# From utils/waiver_big_game.py
# ======================================================================

"""Shared "unexpected big game" detector and sustainability classifier.

A player who erupts for a big week is only a *waiver* story if two separate
things are true, and this module keeps them separate on purpose (they answer
different questions and must not be blurred into one number):

  1. **Performance surprise** — did the game genuinely exceed what was expected
     of the player *going in*? This is a league-scored comparison of realized
     fantasy points against a saved pregame projection (preferred) or, failing
     that, a clearly-labeled historical baseline. It rewards both *relative*
     surprise (beating expectation) and *absolute* production (so a 2-ppg player
     popping for 12 is not treated as an eight-alarm breakout).

  2. **Role sustainability** — will the usage that produced it continue? This is
     a raw role/usage read (snap share, targets, target share, routes, carries,
     touches, red-zone / goal-line work, and whether a teammate injury opened
     the door) plus cautions when the fantasy day leaned on touchdowns, freak
     efficiency, or one explosive play.

The two feed an explainable **classification**:

  * ``priority``    — strong surprise, credible sustainable role.
  * ``speculative`` — encouraging, but the role evidence is thin or uncertain.
  * ``watchlist``   — surprising box score with little sign the role sticks
                      (the classic fluky-touchdown line).

Everything here is pure and league-scoring-agnostic on the role side: fantasy
points must already be scored for the league by the caller (item 6), while the
usage features are raw counts/shares that mean the same in any league. The
module never invents data — a missing usage feature reads as *role unconfirmed*,
never as zero opportunity (item 8) — and never emits a probability: the scores
are bounded 0..1 confidences, labeled as such, not calibrated likelihoods.

League-specific availability and roster fit are deliberately *not* computed
here; the waiver route layers those on so the shared detector can be reused by
the digest, dashboard, and notifications without dragging a league in.
"""

from dataclasses import dataclass
from typing import Mapping



# ---------------------------------------------------------------------------
# Tunables (one surface, mirroring utils.waiver_score.WEIGHTS)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PositionProductionThresholds:
    """Fantasy-point landmarks for a notable, strong, and elite game."""
    floor: float
    strong: float
    elite: float


POSITION_PRODUCTION_THRESHOLDS: Mapping[str, PositionProductionThresholds] = {
    "QB": PositionProductionThresholds(20.0, 28.0, 35.0),
    "RB": PositionProductionThresholds(13.0, 20.0, 28.0),
    "WR": PositionProductionThresholds(13.0, 20.0, 28.0),
    "TE": PositionProductionThresholds(10.0, 15.0, 22.0),
}
DEFAULT_PRODUCTION_THRESHOLDS = PositionProductionThresholds(13.0, 20.0, 28.0)


def production_thresholds(position: Optional[str]) -> PositionProductionThresholds:
    """Return the configured scale, safely using the skill-position default."""
    return POSITION_PRODUCTION_THRESHOLDS.get(
        str(position or "").strip().upper(), DEFAULT_PRODUCTION_THRESHOLDS)


@dataclass(frozen=True)
class BigGameConfig:
    # Shrinkage added to the expectation when computing relative surprise, so a
    # tiny baseline can't manufacture an enormous ratio (a 2->18 game is a real
    # surprise, but not 8x a 12->28 game). Points.
    surprise_shrink_pts: float = 8.0
    # Relative-surprise points that map to a full 1.0 relative component.
    surprise_full_ratio: float = 1.6
    # Absolute production gate: a game below `abs_floor` league points isn't a
    # "big game" no matter how far it beat a microscopic projection; `abs_ceiling`
    # is a genuinely elite day.
    # Legacy defaults retained for callers constructing custom configurations;
    # normal assessment uses POSITION_PRODUCTION_THRESHOLDS above.
    abs_floor: float = 13.0
    abs_ceiling: float = 28.0
    # Weight of the absolute vs relative component in the surprise blend.
    abs_weight: float = 0.45
    # Usage-sustainability: snap-share delta (fraction, 0..1) that reads as a full
    # role jump, and target/route/touch deltas that read as meaningful growth.
    snap_share_full_delta: float = 0.25
    target_share_full_delta: float = 0.12
    routes_full_delta: float = 12.0
    touches_full_delta: float = 8.0
    targets_full_delta: float = 5.0
    # Fraction of fantasy points from TDs above which the day is TD-dependent.
    td_dependence_frac: float = 0.5
    # Fraction of receiving/rushing yards from the single longest play above which
    # the day leaned on one explosive gain.
    explosive_play_frac: float = 0.45
    # Yards per touch / per route above which efficiency is unsustainable.
    yards_per_touch_hot: float = 9.0
    yards_per_route_hot: float = 3.2
    # Games of history at/under which a baseline is "uncertain" (rookies, returns).
    thin_history_games: int = 3
    # Classification thresholds on the (surprise, sustainability) plane.
    priority_surprise: float = 0.55
    priority_sustain: float = 0.5
    speculative_surprise: float = 0.35
    # A strongly sustainable role can surface a discovery even when the single-game
    # surprise was only moderate (the "fewer points, real usage" case), provided
    # the game cleared a minimum surprise and a minimum absolute production.
    role_surface_min: float = 0.2
    role_surface_abs: float = 0.1
    # Sustainability credited to a confirmed teammate-vacancy alone (no usage data
    # yet), bounded because the starter may return.
    vacancy_sustain_max: float = 0.6


CONFIG = BigGameConfig()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GameContext:
    """One player's single-game performance and the context needed to judge it.

    Fantasy-point fields (``actual_points``, ``pregame_projection``,
    ``baseline_ppg``, ``position_baseline_ppg``) must already be scored for the
    target league by the caller. Usage fields are raw and league-agnostic; every
    one is optional and ``None`` means *unknown / not yet reported* — never zero.
    """
    player_id: str
    position: str
    season: int
    week: int

    # League-scored fantasy points.
    actual_points: Optional[float] = None
    # Saved pregame projection snapshot — the only honest expectation. When
    # absent we fall back to the historical baseline and say so.
    pregame_projection: Optional[float] = None
    projection_source: Optional[str] = None      # e.g. "sleeper", "espn"
    projection_saved_at: Optional[str] = None     # ISO ts; None => no snapshot

    # Established historical baseline (labeled), used only when no snapshot.
    baseline_ppg: Optional[float] = None
    baseline_source: Optional[str] = None          # "season_avg" | "trailing4" | "career"
    baseline_games: Optional[int] = None            # sample size behind the baseline
    # Startable/replacement production at the position (position-relative read).
    position_baseline_ppg: Optional[float] = None

    # Raw usage (all optional; None => unconfirmed).
    snap_share: Optional[float] = None              # 0..1 this game
    snap_share_prev: Optional[float] = None         # 0..1 prior baseline
    routes: Optional[float] = None
    routes_prev: Optional[float] = None
    targets: Optional[float] = None
    targets_prev: Optional[float] = None
    target_share: Optional[float] = None            # 0..1
    target_share_prev: Optional[float] = None
    carries: Optional[float] = None
    carries_prev: Optional[float] = None
    pass_attempts: Optional[float] = None
    pass_attempts_prev: Optional[float] = None
    touches: Optional[float] = None
    touches_prev: Optional[float] = None
    redzone_touches: Optional[float] = None
    goalline_touches: Optional[float] = None

    # Efficiency / explosiveness context.
    touchdowns: Optional[float] = None
    td_points: Optional[float] = None               # league points from TDs, if known
    total_yards: Optional[float] = None
    longest_play_yards: Optional[float] = None
    yards_per_touch: Optional[float] = None
    yards_per_route: Optional[float] = None

    # Opportunity context.
    teammate_out: bool = False                       # a starter ahead is confirmed out
    teammate_out_share: Optional[float] = None       # role fraction plausibly freed (committee-aware)

    # Uncertainty flags.
    is_rookie: bool = False
    returning_from_injury: bool = False
    limited_history: bool = False

    # Lifecycle (item 8): "final" | "in_progress".
    status: str = "final"


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BigGameAssessment:
    player_id: str
    season: int
    week: int
    category: str                       # priority | speculative | watchlist | none
    performance_surprise: float          # 0..1 confidence (NOT a probability)
    role_sustainability: float           # 0..1 confidence
    absolute_score: float                # 0..1 how big the raw box score was
    expectation: Optional[float]         # pts the surprise was measured against
    expectation_basis: str               # "pregame_projection" | "baseline:<src>" | "position" | "none"
    role_confirmed: bool                 # did we have real usage data?
    cautions: tuple = ()                 # e.g. ("td_dependent", "one_big_play")
    factors: tuple = ()                  # human-readable evidence, ranked
    status: str = "final"                # "final" | "in_progress" (provisional)

    def to_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "season": self.season,
            "week": self.week,
            "category": self.category,
            "performance_surprise": round(self.performance_surprise, 3),
            "role_sustainability": round(self.role_sustainability, 3),
            "absolute_score": round(self.absolute_score, 3),
            "expectation": (round(self.expectation, 1) if self.expectation is not None else None),
            "expectation_basis": self.expectation_basis,
            "role_confirmed": self.role_confirmed,
            "cautions": list(self.cautions),
            "factors": list(self.factors),
            "status": self.status,
        }


# ---------------------------------------------------------------------------
# Expectation resolution (item 6: snapshots first, labeled baseline otherwise)
# ---------------------------------------------------------------------------

def resolve_expectation(g: GameContext) -> "tuple[Optional[float], str, bool]":
    """Return (expectation_points, basis_label, is_uncertain).

    A *saved pregame projection* is authoritative — it's the only thing that was
    truly "expected" before the game. Absent one, we use a clearly-labeled
    historical baseline, and absent that, the position baseline. Rookies, players
    with thin history, and injury returns are flagged uncertain so callers never
    treat an invented baseline as established.
    """
    uncertain = bool(g.is_rookie or g.returning_from_injury or g.limited_history)
    if g.pregame_projection is not None and g.projection_saved_at:
        return (float(g.pregame_projection), "pregame_projection", uncertain)
    if g.baseline_ppg is not None:
        games = g.baseline_games
        if games is not None and games <= CONFIG.thin_history_games:
            uncertain = True
        src = g.baseline_source or "baseline"
        return (float(g.baseline_ppg), f"baseline:{src}", uncertain)
    if g.position_baseline_ppg is not None:
        return (float(g.position_baseline_ppg), "position", True)
    return (None, "none", True)


# ---------------------------------------------------------------------------
# Performance surprise (relative + absolute, so tiny baselines don't explode)
# ---------------------------------------------------------------------------

def absolute_component(actual: Optional[float], cfg: BigGameConfig = CONFIG,
                       position: Optional[str] = None) -> float:
    """0..1 for how big the raw box score was, independent of expectation."""
    if actual is None:
        return 0.0
    scale = production_thresholds(position) if position else PositionProductionThresholds(
        cfg.abs_floor, (cfg.abs_floor + cfg.abs_ceiling) / 2.0, cfg.abs_ceiling)
    points = float(actual)
    if points <= scale.floor:
        return 0.0
    # The explicit "strong" landmark matters: 15 TE points is a much more
    # meaningful absolute performance than 15 QB points.
    if points <= scale.strong:
        return 0.6 * (points - scale.floor) / max(1e-6, scale.strong - scale.floor)
    return _clamp01(0.6 + 0.4 * (points - scale.strong) /
                    max(1e-6, scale.elite - scale.strong))


def relative_component(actual: Optional[float], expectation: Optional[float],
                       cfg: BigGameConfig = CONFIG,
                       position: Optional[str] = None) -> float:
    """0..1 for how far the game beat expectation, shrunk so a microscopic
    expectation can't produce an absurd ratio. Falls back to the absolute read
    when there's no expectation at all."""
    if actual is None:
        return 0.0
    if expectation is None:
        return absolute_component(actual, cfg, position)
    over = float(actual) - float(expectation)
    if over <= 0:
        return 0.0
    ratio = over / (max(0.0, float(expectation)) + cfg.surprise_shrink_pts)
    return _clamp01(ratio / cfg.surprise_full_ratio)


def performance_surprise(g: GameContext, cfg: BigGameConfig = CONFIG) -> "tuple[float, float]":
    """Blend relative and absolute surprise into a single 0..1 confidence.

    Both matter: a huge *relative* jump off a tiny base still needs real absolute
    production to count, and a merely-good absolute day that was fully expected is
    no surprise. Returns ``(surprise, absolute_score)``.
    """
    rel = relative_component(g.actual_points, _resolve_pts(g), cfg, g.position)
    ab = absolute_component(g.actual_points, cfg, g.position)
    # Blend relative surprise with absolute production: a huge *ratio* off a tiny
    # box score (2->14) lands below a big absolute day (18->31), while the
    # shrinkage in `relative_component` keeps a real low-baseline breakout in play.
    surprise = _clamp01((1.0 - cfg.abs_weight) * rel + cfg.abs_weight * ab)
    return surprise, ab


def _resolve_pts(g: GameContext) -> Optional[float]:
    exp, _basis, _unc = resolve_expectation(g)
    return exp


# ---------------------------------------------------------------------------
# Role sustainability (raw usage; missing => unconfirmed, never zero)
# ---------------------------------------------------------------------------

def _delta_component(cur, prev, full_delta) -> Optional[float]:
    """0..1 growth signal for one usage stat, or None when either side is
    unknown (role unconfirmed on that axis — never scored as zero growth)."""
    if cur is None or prev is None:
        return None
    try:
        d = float(cur) - float(prev)
    except (TypeError, ValueError):
        return None
    if d <= 0:
        return 0.0
    return _clamp01(d / full_delta)


def _level_component(cur, full_level) -> Optional[float]:
    """0..1 absolute-level signal (e.g. an 80% snap share is a real role even
    with no prior to compare), or None when unknown."""
    if cur is None:
        return None
    try:
        return _clamp01(float(cur) / full_level)
    except (TypeError, ValueError):
        return None


def role_sustainability(g: GameContext, cfg: BigGameConfig = CONFIG) -> "tuple[float, bool, tuple, tuple]":
    """Return ``(sustainability, role_confirmed, cautions, factors)``.

    Sustainability rises with real, *rising* opportunity (snap share, targets,
    target share, routes, touches) and dedicated high-value work (red zone /
    goal line). It's discounted — not inflated — by cautions: a TD-dependent day,
    one explosive play, or unsustainable efficiency. When no usage data is
    available at all the role is *unconfirmed* (item 8): sustainability stays low
    and the caller should not claim a lasting role.
    """
    cfg = cfg or CONFIG
    growth: list = []
    factors: list = []

    pos = str(g.position or "").upper()
    # An every-down snap rate is ordinary for a starting QB.  It is neither a
    # role level nor a role confirmation; QB opportunity must move materially.
    snap_growth = (_delta_component(g.snap_share, g.snap_share_prev, cfg.snap_share_full_delta)
                   if pos != "QB" else None)
    snap_level = _level_component(g.snap_share, 0.85) if pos != "QB" else None
    tgt_share_growth = _delta_component(g.target_share, g.target_share_prev, cfg.target_share_full_delta)
    tgt_growth = _delta_component(g.targets, g.targets_prev, cfg.targets_full_delta)
    route_growth = _delta_component(g.routes, g.routes_prev, cfg.routes_full_delta)
    touch_growth = _delta_component(g.touches, g.touches_prev, cfg.touches_full_delta)
    carry_growth = _delta_component(g.carries, g.carries_prev, cfg.touches_full_delta)
    pass_growth = _delta_component(g.pass_attempts, g.pass_attempts_prev, 12.0)

    def _note(val, label):
        if val is not None and val > 0.15:
            factors.append((val, label))

    _note(snap_growth, _fmt_delta("snap share", g.snap_share_prev, g.snap_share, pct=True))
    _note(tgt_share_growth, _fmt_delta("target share", g.target_share_prev, g.target_share, pct=True))
    _note(tgt_growth, _fmt_delta("targets", g.targets_prev, g.targets))
    _note(route_growth, _fmt_delta("routes", g.routes_prev, g.routes))
    _note(touch_growth, _fmt_delta("touches", g.touches_prev, g.touches))
    _note(carry_growth, _fmt_delta("carries", g.carries_prev, g.carries))
    _note(pass_growth, _fmt_delta("pass attempts", g.pass_attempts_prev, g.pass_attempts))

    if pos == "QB":
        relevant = (pass_growth, carry_growth)
    elif pos == "RB":
        relevant = (snap_growth, tgt_growth, touch_growth, carry_growth)
    elif pos == "TE":
        relevant = (snap_growth, tgt_share_growth, tgt_growth, route_growth)
    else:  # WR and unknown skill positions
        relevant = (snap_growth, tgt_share_growth, tgt_growth, route_growth)
    for c in relevant:
        if c is not None and c > 0:
            growth.append(c)

    # High absolute role even without a prior comparison (e.g. an 85% snap share
    # the first week a starter is out) is itself a sustainability signal.
    level_signals = []
    if pos != "QB" and snap_level is not None:
        level_signals.append(snap_level)
    if pos in ("WR", "TE"):
        target_level = _level_component(g.target_share, 0.28)
        if target_level is not None:
            level_signals.append(target_level)

    high_value_work = []
    if pos == "RB" and g.redzone_touches and g.redzone_touches >= 2:
        factors.append((0.5, f"{int(g.redzone_touches)} red-zone touches"))
        high_value_work.append(min(1.0, float(g.redzone_touches) / 4.0))
    if pos == "RB" and g.goalline_touches and g.goalline_touches >= 1:
        factors.append((0.55, f"{int(g.goalline_touches)} goal-line touches"))
        high_value_work.append(min(1.0, float(g.goalline_touches) / 2.0))

    # Opportunity from a confirmed teammate absence, committee-adjusted: we do NOT
    # assume the whole workload transfers to one player (item 6). It's bounded
    # because the starter may return, and counts as opportunity even before any
    # usage data lands.
    opp = 0.0
    if g.teammate_out:
        share = g.teammate_out_share
        if share is None:
            share = 0.5  # conservative committee split when unknown
        opp = _clamp01(float(share))
        factors.append((0.4 + 0.3 * opp, "starter ahead ruled out"))
    opp_sustain = cfg.vacancy_sustain_max * opp

    # A real, rising or high-level usage read confirms the role; a confirmed
    # vacancy is opportunity even without usage data yet.
    role_confirmed = bool(growth) or bool(level_signals) or bool(high_value_work)
    if pos == "QB" and g.teammate_out:
        role_confirmed = True

    base_signals = growth + level_signals + high_value_work
    base = max(base_signals) if base_signals else 0.0
    # Corroboration: a second independent rising signal lifts confidence with
    # diminishing returns (mirrors the opportunity-combine in waiver_score, so we
    # don't double-count correlated usage stats).
    corrob = sorted(base_signals, reverse=True)
    combined = base + (0.35 * corrob[1] if len(corrob) > 1 else 0.0)
    sustain = _clamp01(max(combined, opp_sustain))

    # Cautions reduce sustainability.
    cautions: list = []
    td_frac = _td_fraction(g)
    if td_frac is not None and td_frac >= cfg.td_dependence_frac:
        cautions.append("td_dependent")
        sustain *= 0.55
    exp_frac = _explosive_fraction(g)
    if exp_frac is not None and exp_frac >= cfg.explosive_play_frac:
        cautions.append("one_big_play")
        sustain *= 0.6
    if _hot_efficiency(g, cfg):
        cautions.append("hot_efficiency")
        sustain *= 0.7
    if not role_confirmed:
        cautions.append("role_unconfirmed")
        # No usage data: a confirmed vacancy still carries (bounded) opportunity,
        # but with nothing at all the role stays near-zero (never negative).
        if opp <= 0:
            sustain = min(sustain, 0.2)
    if g.is_rookie or g.returning_from_injury or g.limited_history:
        cautions.append("uncertain_baseline")

    factors.sort(key=lambda t: t[0], reverse=True)
    factor_labels = tuple(lbl for _s, lbl in factors)
    return _clamp01(sustain), role_confirmed, tuple(cautions), factor_labels


def _td_fraction(g: GameContext) -> Optional[float]:
    if g.actual_points is None or g.actual_points <= 0:
        return None
    if g.td_points is not None:
        return _clamp01(float(g.td_points) / float(g.actual_points))
    if g.touchdowns is not None:
        # Assume ~6 fantasy pts per TD (offensive skill) when explicit TD points
        # aren't provided; clamp so it stays a fraction.
        return _clamp01(float(g.touchdowns) * 6.0 / float(g.actual_points))
    return None


def _explosive_fraction(g: GameContext) -> Optional[float]:
    if not g.total_yards or g.longest_play_yards is None:
        return None
    try:
        return _clamp01(float(g.longest_play_yards) / float(g.total_yards))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _hot_efficiency(g: GameContext, cfg: BigGameConfig) -> bool:
    if g.yards_per_touch is not None and g.yards_per_touch >= cfg.yards_per_touch_hot:
        return True
    if g.yards_per_route is not None and g.yards_per_route >= cfg.yards_per_route_hot:
        return True
    return False


def _fmt_delta(label, prev, cur, pct=False) -> str:
    """Concise evidence string like 'snap share 41%->78%' or 'targets 3->9'."""
    def _f(x):
        if x is None:
            return "?"
        if pct:
            return f"{round(float(x) * 100)}%"
        return f"{round(float(x))}"
    return f"{label} {_f(prev)}->{_f(cur)}"


# ---------------------------------------------------------------------------
# Classification (explainable categories, item 7)
# ---------------------------------------------------------------------------

def classify(surprise: float, sustainability: float, absolute: float,
             cfg: BigGameConfig = CONFIG, role_confirmed: bool = True,
             uncertain_expectation: bool = False) -> str:
    """Map the (surprise, sustainability) plane onto an explainable category.

    priority    — the game was a real surprise AND the role looks like it sticks.
    speculative — encouraging surprise with moderate/uncertain role evidence.
    watchlist   — a surprising box score whose role evidence is thin (fluky).
    none        — not surprising enough to surface.
    """
    # A credible, sustainable role can surface a discovery even when the single
    # game surprise was only moderate ("fewer points, real usage"), as long as
    # the game cleared a floor of surprise and real production.
    role_surface = (surprise >= cfg.role_surface_min
                    and sustainability >= cfg.priority_sustain
                    and absolute >= cfg.role_surface_abs)
    if surprise < cfg.speculative_surprise and not role_surface:
        return "none"
    if (role_confirmed and sustainability >= cfg.priority_sustain
            and (surprise >= cfg.priority_surprise or role_surface)):
        return "priority"
    if surprise >= cfg.speculative_surprise:
        return "speculative" if sustainability >= 0.3 else "watchlist"
    return "watchlist"


def assess_big_game(g: GameContext, cfg: BigGameConfig = CONFIG) -> BigGameAssessment:
    """Full assessment for one player-game. Pure; safe on partial data."""
    cfg = cfg or CONFIG
    exp, basis, uncertain = resolve_expectation(g)
    surprise, absolute = performance_surprise(g, cfg)
    sustain, role_confirmed, cautions, role_factors = role_sustainability(g, cfg)
    category = classify(surprise, sustain, absolute, cfg, role_confirmed, uncertain)

    factors: list = []
    if g.actual_points is not None:
        if exp is not None:
            factors.append(f"{round(g.actual_points, 1)} pts vs {round(exp, 1)} expected "
                           f"({_basis_phrase(basis)})")
        elif absolute > 0:
            factors.append(f"{round(g.actual_points, 1)} points")
    factors.extend(role_factors)
    if "td_dependent" in cautions:
        factors.append("leaned on touchdowns")
    if "one_big_play" in cautions:
        factors.append("one long play drove the yardage")
    if "hot_efficiency" in cautions:
        factors.append("efficiency unlikely to hold")
    if "role_unconfirmed" in cautions:
        factors.append("Usage not yet confirmed")

    return BigGameAssessment(
        player_id=g.player_id,
        season=g.season,
        week=g.week,
        category=category,
        performance_surprise=surprise,
        role_sustainability=sustain,
        absolute_score=absolute,
        expectation=exp,
        expectation_basis=basis,
        role_confirmed=role_confirmed,
        cautions=cautions,
        factors=tuple(factors),
        status=g.status or "final",
    )


def _basis_phrase(basis: str) -> str:
    if basis == "pregame_projection":
        return "pregame projection"
    if basis.startswith("baseline:"):
        return f"{basis.split(':', 1)[1]} baseline"
    if basis == "position":
        return "positional baseline"
    return "no baseline"


# ---------------------------------------------------------------------------
# Idempotent lifecycle (item 8: live -> final -> stat-correction, no dupes)
# ---------------------------------------------------------------------------

def discovery_key(player_id, season, week) -> str:
    """Stable identity for one player-game discovery, so re-running during a game,
    after final stats, and after stat corrections updates one row rather than
    creating duplicates."""
    return f"{season}:{int(week)}:{player_id}"


# Rank of lifecycle states — a later/more-final state supersedes an earlier one.
_STATUS_RANK = {"in_progress": 0, "final": 1, "corrected": 2}


def merge_assessment(existing: Optional[BigGameAssessment],
                     incoming: BigGameAssessment) -> BigGameAssessment:
    """Idempotently fold a new assessment into an existing discovery for the same
    player-game. The more-final status wins; an equal-or-newer status refreshes
    the numbers in place. Never produces a duplicate — callers key on
    ``discovery_key`` and store the merged result."""
    if existing is None:
        return incoming
    if discovery_key(existing.player_id, existing.season, existing.week) != \
       discovery_key(incoming.player_id, incoming.season, incoming.week):
        return incoming
    old_rank = _STATUS_RANK.get(existing.status, 1)
    new_rank = _STATUS_RANK.get(incoming.status, 1)
    if new_rank < old_rank:
        # A late live-scored update arriving after final stats: keep the final one.
        return existing
    return incoming


# ======================================================================
# From utils/streaming_targets.py
# ======================================================================

"""Shared D/ST + K streaming rankers.

Single home for the matchup-based streaming logic behind /api/streaming-options
and the K / D/ST tabs of /api/waiver-candidates. Rankings:

  * defenses: free-agent D/STs sorted by how weak the offense they face is
    (opponent Vegas implied total, ascending);
  * kickers: free-agent Ks on teams playing this week, one per team, sorted by
    their own team's Vegas implied total (descending).

Everything is pure with respect to the passed league context; schedule and
Vegas data flow through the same cached loaders the old app.py endpoint used.
Never raises: missing schedule/Vegas data degrades to empty lists so callers
fall back gracefully (e.g. offseason, or a league that starts no K/DST).
"""

import logging

logger = logging.getLogger(__name__)

# FAAB-scale composite attached to each row (``stream_score``), so waiver-candidate
# K/DST rows can size FAAB bids on the same absolute scale as skill-position rows
# (FAAB_SCORE_LOW=45 .. FAAB_SCORE_HIGH=190 in utils.waiver_score). A genuinely
# elite streamer (bottom-3 opposing offense / top-3 team total) lands well above
# the waiver-floor score; a no-Vegas-data row sits mid-pack.
_STREAM_SCORE_FLOOR = 50.0
_STREAM_SCORE_CAP = 155.0
_STREAM_SCORE_NODATA = 75.0


def stream_score(implied, *, lower_is_better: bool) -> float:
    """Map a Vegas implied total onto the waiver composite scale.

    ``lower_is_better`` for defenses (a low *opponent* total is good), False for
    kickers (a high *own* total is good). Missing data returns a mid-pack score
    rather than zero so an unpriced game doesn't bury the row.
    """
    try:
        v = float(implied)
    except (TypeError, ValueError):
        return _STREAM_SCORE_NODATA
    if lower_is_better:
        # Opp implied 26 -> replacement-level streamer, 14 -> elite.
        s = _STREAM_SCORE_FLOOR + (26.0 - v) * 8.0
    else:
        # Own implied 17 -> replacement-level streamer, 30 -> elite.
        s = _STREAM_SCORE_FLOOR + (v - 17.0) * 8.0
    return round(max(_STREAM_SCORE_FLOOR - 10.0, min(_STREAM_SCORE_CAP, s)), 1)


def streaming_targets(ctx, season, current_week=None, players_index=None, limit=8):
    """Ranked free-agent defenses and kickers for the current week.

    ``ctx`` is the league context (rosters, roster_positions, ...),
    ``players_index`` maps player ids to {pos, team, name} (falls back to the
    ctx's own index). Returns ``{"defense": [...], "kicker": [...],
    "in_season": bool, "uses_def": bool, "uses_k": bool}``. Defense rows carry
    ``opp_implied``; kicker rows carry ``own_implied``; every row carries
    ``stream_score``.
    """
    try:
        return _streaming_targets(ctx, season, current_week, players_index, limit)
    except Exception:
        logger.debug("streaming_targets failed", exc_info=True)
        return {"defense": [], "kicker": [], "in_season": False,
                "uses_def": False, "uses_k": False}


def _streaming_targets(ctx, season, current_week, players_index, limit):
    ctx = ctx or {}
    if current_week is None:
        current_week = int(ctx.get("current_week") or 0)
    if current_week < 1 or ctx.get("offseason_mode"):
        return {"defense": [], "kicker": [], "in_season": False,
                "uses_def": False, "uses_k": False}

    # League's started positions gate which streamers are relevant at all.
    rpos = [str(s).upper() for s in (ctx.get("roster_positions") or [])]
    uses_def = any(s in ("DEF", "DST", "D/ST") for s in rpos)
    uses_k = "K" in rpos
    if not (uses_def or uses_k):
        return {"defense": [], "kicker": [], "in_season": True,
                "uses_def": False, "uses_k": False}

    # ── Schedule → opponent map + games for the Vegas lookup ──────────────────
    opponent_map: dict = {}
    week_games: list = []
    teams: set = set()
    try:
        from utils.data_cache import load_week_sched
        for g in (load_week_sched(season, current_week) or []):
            home = str(g.get("home") or "").upper()
            away = str(g.get("away") or "").upper()
            if home and away:
                opponent_map[home] = away
                opponent_map[away] = home
                week_games.append((home, away, str(g.get("gameDate") or "")))
                teams.add(home)
                teams.add(away)
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
    if not teams:
        return {"defense": [], "kicker": [], "in_season": True,
                "uses_def": uses_def, "uses_k": uses_k}

    conditions: dict = {}
    try:
        from utils.start_sit import build_week_conditions
        conditions = build_week_conditions(season, current_week, week_games) or {}
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    def _implied(team):
        v = (conditions.get(team) or {}).get("implied_total")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def _matchup(team):
        opp = opponent_map.get(team)
        if not opp:
            return "", None
        # home_team_of not tracked here; label as "vs OPP" (venue is secondary).
        return f"vs {opp}", opp

    rostered = {
        str(pid)
        for r in (ctx.get("rosters") or [])
        for pid in (r.get("players") or [])
    }
    players_index = players_index if players_index is not None else (ctx.get("players_index") or {})

    # ── Defenses: one per team, pid == team abbr in Sleeper. Best matchup is the
    # weakest opposing offense (lowest opponent implied total). ────────────────
    defense = []
    if uses_def:
        rows = []
        for t in teams:
            if t in rostered:
                continue
            label, opp = _matchup(t)
            opp_imp = _implied(opp)
            rows.append({
                "player_id": t, "name": f"{t} D/ST", "position": "DEF", "team": t,
                "opponent": opp, "matchup": label, "opp_implied": opp_imp,
                "stream_score": stream_score(opp_imp, lower_is_better=True),
            })
        rows.sort(key=lambda d: (d["opp_implied"] is None,
                                 d["opp_implied"] if d["opp_implied"] is not None else 99.0))
        defense = rows[:limit]

    # ── Kickers: free-agent Ks on teams playing this week, ranked by their own
    # implied total (more team scoring → more FGs/XPs). One per team. ──────────
    kicker = []
    if uses_k:
        cand = []
        for pid, meta in (players_index or {}).items():
            if str((meta or {}).get("pos") or "").upper() != "K":
                continue
            pid = str(pid)
            t = str((meta or {}).get("team") or "").upper()
            if not t or t not in teams or pid in rostered:
                continue
            cand.append((pid, (meta or {}).get("name") or f"Player {pid}", t, _implied(t)))
        cand.sort(key=lambda x: (x[3] is None, -(x[3] if x[3] is not None else 0.0)))
        seen_team = set()
        for pid, name, t, imp in cand:
            if t in seen_team:
                continue
            seen_team.add(t)
            label, opp = _matchup(t)
            kicker.append({
                "player_id": pid, "name": name, "position": "K", "team": t,
                "opponent": opp, "matchup": label, "own_implied": imp,
                "stream_score": stream_score(imp, lower_is_better=False),
            })
            if len(kicker) >= limit:
                break

    return {"defense": defense, "kicker": kicker, "in_season": True,
            "uses_def": uses_def, "uses_k": uses_k}
