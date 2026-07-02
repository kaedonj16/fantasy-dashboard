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

from dataclasses import dataclass

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
    injury_pts_per_vacated_ppg: float = 1.0
    injury_horizon_weeks: float = 4.0
    injury_fallback_ppg: float = 9.0  # used when the injured player's proj is unknown
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


WEIGHTS = WaiverWeights()


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


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
    return {
        "injured_ahead": injured,          # statuses (badge / severity)
        "injured_pids_ahead": injured_pids,
        "vacated": vacated,                # [{status, pid, proj_ppg}] for scoring
        "healthy_ahead": healthy_ahead,
    }


def depth_analysis_for_player(pid, full_players: dict, depth_index: dict) -> dict:
    """Convenience wrapper: depth_analysis for ``pid`` on its own depth chart."""
    fp = full_players or {}
    p = fp.get(pid) or fp.get(str(pid)) or {}
    team = str(p.get("team") or "").upper()
    pos = str(p.get("position") or "").upper()
    if not team or not pos:
        return {"injured_ahead": [], "injured_pids_ahead": [], "healthy_ahead": 0}
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
        total += likelihood * weeks * max(0.0, ppg_v)
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
# Composite score + signal
# ---------------------------------------------------------------------------

def waiver_pickup_score(c: dict, waiver_breakout: dict,
                        prime_max: dict = WAIVER_PRIME_MAX,
                        w: WaiverWeights = WEIGHTS) -> float:
    """Composite waiver-pickup score.

    ``c`` is a candidate dict. Recognized keys: value, age, position,
    rank_change_7d, player_id, and (all optional, default to a no-op when
    absent) ros_ppg, usage_stat, usage_delta, injured_ahead, healthy_ahead,
    vacated_volume_weight, injury_freshness, need_mult, self_status.
    """
    try:
        val = float(c.get("value") or 0)
    except (TypeError, ValueError):
        val = 0.0
    age = c.get("age") or 0
    pos = c.get("position")
    rank_chg = c.get("rank_change_7d") or 0
    bscore = waiver_breakout.get(c.get("player_id"), 0) or 0
    prime = prime_max.get(pos, 28)

    # Base worth: saturating dynasty value + rest-of-season projection (#5).
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
    usage_pts = min(usage_ratio(c.get("usage_stat"), c.get("usage_delta")) * w.usage_per_ratio,
                    w.usage_max)
    breakout_pts = min(bscore * w.breakout_per, w.breakout_max)
    opp = sorted([injury_pts, usage_pts, breakout_pts], reverse=True)
    opportunity_pts = opp[0] + w.opp_second * opp[1] + w.opp_third * opp[2]  # (#6)

    # Weekly rank trend: reward risers, mildly penalize players falling away.
    if rank_chg > 0:
        trend_pts = min(rank_chg * w.trend_up_per, w.trend_up_max)
    else:
        trend_pts = max(rank_chg * w.trend_down_per, w.trend_down_floor)

    # Age: smooth youth reward / past-prime decay, both bounded.
    if not age:
        age_pts = 0.0
    else:
        gap = prime - age  # + = younger than prime
        if gap >= 0:
            age_pts = min(w.age_base + gap * w.age_youth_per, w.age_youth_max)
        else:
            age_pts = max(w.age_base + gap * w.age_decay_per, w.age_floor)

    raw = value_pts + proj_pts + opportunity_pts + trend_pts + age_pts

    # Roster-aware (#4): a position of real need to the viewer is worth more.
    raw *= float(c.get("need_mult") or 1.0)
    # Candidate's own health (#2): a hurt backup isn't this week's add.
    raw *= self_injury_multiplier(c.get("self_status"))
    return raw


def waiver_signal(c: dict, waiver_breakout: dict,
                  prime_max: dict = WAIVER_PRIME_MAX) -> "tuple[str, str]":
    """Return (badge_class, label) describing why a candidate is interesting.

    Shared by both waiver surfaces. Branches that read data a surface doesn't
    provide (usage, depth chart) are simply no-ops there.
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
    if rank_chg >= 8:
        return ("signal-rising", "Rising Fast")
    if rank_chg >= 3:
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
