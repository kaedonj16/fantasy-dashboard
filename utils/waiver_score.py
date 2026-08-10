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
from __future__ import annotations

import re
from dataclasses import dataclass

from utils.draft_grade import clamp01 as _clamp01

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
_OUT_STATUSES = {"IR", "PUP", "NFI", "SUSP", "OUT", "DOUBTFUL"}


@dataclass(frozen=True)
class WaiverWeights:
    """Every tunable constant in the model, in one place, so #8 (backtest
    calibration) has a single surface to fit instead of magic numbers scattered
    through the code."""
    # Saturating value curve: VALUE_MAX * v / (v + VALUE_HALF).
    value_max: float = 120.0
    value_half: float = 500.0
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
    # Diminishing-returns weights when combining correlated opportunity signals.
    opp_second: float = 0.5
    opp_third: float = 0.25
    # Weekly rank trend.
    trend_up_per: float = 3.5
    trend_up_max: float = 45.0
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


def weeks_out_from_projections(weekly_projs, zero_threshold: float = 1.0) -> int:
    """Derive weeks-out from the leading run of ~zero weekly projections.

    Projection providers zero out a player's weekly points for every week they're
    expected to miss, so the number of consecutive at-or-below-threshold weeks
    starting now is a direct read on the injury timeline — far better than
    guessing from the injury label. ``weekly_projs`` is this player's projected
    points for the upcoming weeks, in order (week now, +1, +2, ...).
    """
    n = 0
    for p in (weekly_projs or []):
        try:
            v = float(p) if p is not None else 0.0
        except (TypeError, ValueError):
            v = 0.0
        if v <= zero_threshold:
            n += 1
        else:
            break
    return n


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
    if s in {"IR", "PUP", "NFI", "SUSP", "OUT"}:
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
    opp = sorted([injury_pts, usage_pts, breakout_pts], reverse=True)
    opportunity_pts = opp[0] + w.opp_second * opp[1] + w.opp_third * opp[2]  # (#6)

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
    if str(c.get("self_status") or "").upper() in {"IR", "PUP", "NFI", "SUSP", "OUT"}:
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
    if age and age > prime + 2:
        return ("signal-aging", "Sell Window")
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
