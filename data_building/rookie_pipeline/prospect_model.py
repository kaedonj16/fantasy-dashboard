"""
Prospect evaluation model - position-aware, multi-factor scoring.

Each component score is 0-100.  Final prospect_score is a weighted sum.

Model version: 3.13 (see MODEL_VERSION)
  v1.x: baseline components below.
  v2.0: adds recruiting-pedigree component (247 Composite via CFBD),
        WEPA opponent-adjusted efficiency blend (QB/RB), and SP+/Team Talent
        inputs to the competition adjustment.  Per Kaedon's rule, grading
        changes are version-bumped, never silently tweaked: the version is
        persisted per row in rookie_rankings.model_version.
  v3.4: two-mode grading. COMPLETE mode is v3.3. IN-SEASON mode (the run
        supplies the active season year and the player's latest season is
        that year at 1-11 stored games) grades on prior complete seasons
        plus the partial season at per-game rates with paced counting
        gates; untested athleticism is missing data (weight redistributed),
        never the projected 67.0 floor; confidence is discounted by the
        partial sample. Also: durability only counts seasons where the
        player held a real role (usage thresholds), so backup seasons no
        longer read as injury seasons in either mode; the WR/TE red-zone
        proxy thresholds are recalibrated to the recorded TD-rate
        distribution (they could never fire at the old 5.5-10.0 levels).
  v3.4.1: data repair, no formula change. Stored season rows that
        were persisted without a dominator_rating (team totals absent
        from the fetch map at build time, then frozen by the
        existence-only fetch gate) are derived from their own stored
        inputs at write time and via scripts/backfill_stored_dominators.py;
        breakout components rise for players whose early dominators
        were missing. Grades move only for affected players.
  v3.5: grading-input DEFINITION change. dominator_rating for WR/TE
        is now receiving share - 0.65 * (receiving yards / team pass
        yards) + 0.35 * (receiving TDs / team pass TDs) - instead of
        share of total offense, which structurally taxed pass
        catchers (their denominator included team rushing production)
        and sat on a different basis from this model's own breakout
        anchors (0.35 saturation, 0.4877 recorded best). RB dominator
        is unchanged (scrimmage share of total offense); QB unchanged.
        Stored WR/TE rows are recomputed by
        scripts/recompute_dominators.py (migration 051 adds the
        team_pass_tds denominator); no weights or thresholds change
        in this version.
  v3.6: de-saturation. Top anchors that sat AT recorded-best marks
        (so real seasons pinned whole legs at exactly 100 and the
        grade stopped separating players on them) are widened to
        recorded-best-plus-margin, all absolute anchors: competition
        splits SEC (1.00) from Big Ten (0.97) and its scale anchor
        moves 1.00 -> 1.03; breakout dominator saturation multiples
        WR 1.4 -> 1.7, RB 1.8 -> 2.0, TE 1.8 -> 2.4; WR production
        anchors rec yds/g 100 -> 120 and pass share 0.35 -> 0.40;
        WR efficiency anchors yds/rec 17.0 -> 19.5 and market share
        0.30 -> 0.34; TE production TDs/g anchor 0.5 -> 0.65. The TE
        elite bonuses (x1.18 / x1.08) now require a completed season
        or 8+ games played, so a short in-progress season cannot
        clear them unpaced. No weights change.
        Round-2 audit mechanics (same release): a pick is priced once
        by the pick curve - the round multiplier (whose round-2 x1.08
        bonus made pick 33 worth more than pick 32) and the separate
        Day-3 penalty (a third charge on picks 97+) are removed, so
        capital value is non-increasing in pick by construction; and
        talent_score now holds the translation layer identical across
        the main and talent runs (same pick context, only the capital
        LEG excluded), so capital_delta prices the leg alone instead
        of silently including pick-gated translation (up to +9 for a
        top-10 TE).
        Round-2 audit, full pass (same release; the audit note is
        ~/workspace/prospect-runs/grading-audit-round2-2026-11-23.md):
        F1 seed-only capital (num_mocks_used == 0, not an actual pick)
        counts as missing and redistributes; F13 the confidence
        capital points require the same verified capital. F2
        in-progress seasons shrink toward the completed-season
        baseline by min(1, GP/12) in production, efficiency and
        utilization. F3 the translation layer keeps only the WR
        contested-catch trait (no leg reads it from season data);
        every pick-gated, share, dominator, rec-yds and age branch is
        deleted, translation no longer reads the pick at all, and QB
        rushing is priced once in production (blend share 15% -> 25%)
        instead of a flat translation add. F5 market share leaves
        efficiency at WR/RB/TE (WR gains a YAC component). F6 QB
        production prices volume only; the YPA/comp%/TD:INT trio
        lives in efficiency alone. F7 the transfer multiplier and
        scheme-inflation discount are deleted; competition and
        environment carry team context. F8 the crowded-room
        multiplier applies to production only. F9 growth is priced
        once in breakout (totals-growth switch and ascending bonus
        deleted). F10 missing recruiting returns None and
        redistributes. F11 the age floor becomes a declining tail
        (5 points/year below the raw-20 point, floor 10). F15 the
        complete-mode athleticism floor (imputed 67.0) is deleted;
        redistribution carries untested players in both modes. F16
        the QB experience add-on prices from the redistributed
        weights, and the sub-4-GP production divisor cliff becomes
        max(GP, 4) plus a shrink toward neutral. F12 stale comments
        corrected. No blend weights change in any of it.
  v3.7: capital-free grades. Per Kaedon's directive ("I want
        something that doesn't handle the draft capital, because
        this is meant to be my own rankings"), draft capital -
        actual, projected, consensus or seed - plays NO role in the
        grade. The grade is a pure talent grade: the site's own
        rankings, validated on correlation to NFL production, not
        on draft position. The capital leg is excluded for every
        prospect through the model's existing missing-leg
        redistribution (the path that already handled a prospect
        with no pick), so each 3.7 grade is exactly the 3.6 talent
        grade of the same profile and the capital weight spreads
        proportionally over the remaining legs. Every other leg is
        unchanged from 3.6 at full strength - production,
        efficiency, breakout, competition, age, athleticism,
        recruiting (a talent signal, not capital), utilization,
        durability, environment. Consensus/mock-draft data goes
        with capital: the confidence score no longer awards
        capital-presence points, the payload's capital component
        is always None and capital_delta is always None (there is
        no capital effect to report), and talent_score is simply
        the grade itself. score_prospect still ACCEPTS the
        draft_capital argument for call compatibility with the
        pipeline and the backtest; it never reads it.
        F12 weights fold (2026-11-24, same version - 3.7 had not
        shipped): the blend weights are the F12 fitted set, not
        the design weights the paragraph above describes. Per-leg
        correlations to cumulative NFL production over the dumped
        2016-2025 gate population, blended 50/50 with the design
        weights, LOCO-CV checked, then validated by a full gate
        run with the set injected as an override
        (gate_v37_weights.log: QB .393, RB .411, WR .369, TE .407
        against talent-only bars .361/.397/.332/.365 - all four
        cleared). The set is baked into POSITION_WEIGHTS in
        effective form, so the shipped model reproduces the gated
        grades with no override, and a 3.7 grade is the 3.6 talent
        legs under the fitted weights: legs unchanged, blend
        re-weighted. Full provenance sits above POSITION_WEIGHTS.

Component weights (F12 fitted set; exact per-position values and
provenance above POSITION_WEIGHTS):
    production_score             ~19-24%  college production volume
    efficiency_score             ~12-19%  per-attempt / per-target efficiency
    age_score                    ~14-20%  age-adjusted production; youth premium
    breakout_profile_score       ~7-15%   early-career dominance trajectory
    athleticism_score            ~8-15%   combine / speed score / RAS
    utilization_score            ~5-14%   opportunity share
    competition_score            ~4-10%   conference + opponent quality (+SP+ SOS)
    recruiting_score             ~3.2-3.5% 247 Composite pedigree (stars/rating)
    environment_adjustment       ~0-4%    team scheme / usage context
    experience_score (QB only)   ~4.4%    college starting experience
    durability_score             ~0-0.6%  games missed, injury history
    draft capital                0%       NOT PRICED since v3.7 and no longer
                                          carried in the tables; the capital
                                          pop in score_prospect remains as a
                                          guard for foreign overrides
    ──────────────────────────────────
    Total                        100 %

Draft capital was priced by the position pick curve through v3.6
(v3.6 F4 removed the day-tier round multipliers and the separate
Day-3 penalty that stacked on top of it). Since v3.7 the curve is
not consulted at all: no pick, from any source, enters the grade.

Position-specific adjustments are baked into each scorer to handle the
different stat profiles of QB / RB / WR / TE.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional


# Grading-model version. Bump on ANY change to scoring logic or weights.
MODEL_VERSION = "3.13"


def active_season_year_for_run(today=None) -> Optional[int]:
    """The college season in progress on the run date, else None.

    The season runs from Aug 25 (the durability function's own season-
    start convention) through mid-January (the national championship
    window): Aug 25 to Dec 31 of year Y is season Y, and Jan 1 to Jan 20
    of year Y is still season Y-1. Outside that window no season is in
    progress and every profile grades in COMPLETE mode.

    Callers pass the result to score_prospect / score_all_prospects as
    the active season year. It must come from the RUN CALENDAR, never
    from a player's draft class: deriving it per player would flip
    historical final seasons of 11 or fewer games into in-season mode
    and silently move the backtest (round 14 of the iteration log).
    """
    from datetime import date as _d
    today = today or _d.today()
    if today.month >= 8:
        return today.year if today >= _d(today.year, 8, 25) else None
    if today.month == 1 and today.day <= 20:
        return today.year - 1
    return None


def _season_games_elapsed(season_year, today) -> Optional[int]:
    """Team games a college programme has played by `today` in season
    `season_year` - a conservative floor from the run calendar, not a
    schedule lookup.

    v3.9: the F2 in-season shrink keyed only on the PLAYER's games
    played, so a player who missed games was discounted as if the
    season itself were half-unplayed: on a late-November run a 9-game
    player was shrunk as a 9/12 sample with five games of regression
    still to come, when one regular-season game remained. The shrink
    prices regression risk over the games REMAINING, and games
    remaining are bounded by the calendar, not by the player's
    availability. Profiles store no team-games field, so the calendar
    is the only honest source: the season starts Aug 25 (this module's
    own season-start convention, see active_season_year_for_run),
    teams play roughly weekly, and one bye week is banked, giving
    weeks since Aug 25 minus one, clamped to [0, 12]. The banked bye
    keeps the floor at or below what every programme has had the
    chance to play, so it can only understate season progress, never
    overstate it. Returns None when the run date is unknown (the
    caller pinned none), which keeps the pre-3.9 GP-only shrink.
    """
    if season_year is None or today is None:
        return None
    from datetime import date as _d, datetime as _dt
    if isinstance(today, _dt):
        today = today.date()
    start = _d(int(season_year), 8, 25)
    days = (today - start).days
    if days < 0:
        return 0
    return max(0, min(12, days // 7 - 1))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _shrink_partial(raw: float, baseline: float, gp, elapsed=None) -> float:
    """v3.6 (round-2 audit F2): shrink an in-progress season's leg score
    toward a baseline by the sample-size factor min(1, GP/12) - the
    idiom the model already uses for its in-season pacing and its
    confidence discount. A 6-game partial season's rate stats are
    halfway between evidence and prior; before this, sample size was
    priced nowhere in the grade itself (only inside production's
    counting-stat gates), so a hot 6-game partial season graded at
    full weight. At 12+ games the shrink is 1.0 and the raw score
    stands.

    v3.9: when the run calendar is pinned, the sample is the LARGER of
    the player's games and the season's elapsed team games
    (_season_games_elapsed): the regression the shrink guards against
    can only happen in games still to be played, and a player who
    missed games in a nearly finished season has almost none left.
    With elapsed=None the GP-only factor stands, byte-identical."""
    g = _safe(gp, 12)
    if elapsed is not None:
        g = max(g, float(elapsed))
    f = min(1.0, max(g, 0.0) / 12.0)
    return baseline + f * (raw - baseline)


def _safe(v, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _scale(raw: float, lo: float, hi: float) -> float:
    """Linear map [lo, hi] → [0, 100], clipped."""
    if hi <= lo:
        return 50.0
    return _clip((raw - lo) / (hi - lo) * 100.0)


_COUNTING_KEYS = (
    "receiving_yards", "rush_yards", "pass_yards",
    "receptions", "rush_attempts", "pass_attempts",
)


def season_has_counting_stats(season: Dict) -> bool:
    """True when a season row carries any real counting volume.

    v3.8 (under-rating audit M1): the eval pipeline writes placeholder
    rows for the draft year (e.g. season=2027) with EVERY stat field
    NULL. Those rows are not seasons: legs that read them score
    0-filled dicts (production 0.0, template efficiency from friendly
    rate defaults) instead of reporting the leg missing. Callers drop
    them at profile assembly; score_prospect drops them again so every
    entry point is safe.
    """
    return any(_safe(season.get(k)) > 0 for k in _COUNTING_KEYS)


# Position-relevant counting volume per leg family (v3.8 M1). A leg
# whose read seasons carry none of this volume has no measurement to
# make and returns None (missing-leg redistribution) instead of
# scoring an empty dict.
_PROD_VOLUME_KEYS: Dict[str, tuple] = {
    "QB": ("pass_yards", "pass_attempts", "rush_yards", "rush_attempts"),
    "RB": ("rush_yards", "rush_attempts", "receiving_yards",
           "receptions", "targets"),
    "WR": ("receiving_yards", "receptions", "targets"),
    "TE": ("receiving_yards", "receptions", "targets"),
}


def _has_volume(seasons: List[Dict], keys: tuple) -> bool:
    return any(_safe(s.get(k)) > 0 for s in seasons for k in keys)


def _has_recorded_volume(seasons: List[Dict], keys: tuple) -> bool:
    """v3.8 (TE fix, post-gate): True when any of the leg's counting
    fields is RECORDED (present, not None) in any season - including
    recorded zeros. A recorded zero is a measurement (a college QB
    converted to TE really did catch zero passes; his receiving legs
    price at zero, they are not missing). The M1 None-exits fire only
    when the fields were never recorded at all - absent data, the
    FCS / never-loaded hole the exits were built for. The first
    build's exits keyed on volume > 0 and so converted measured
    zeros into missing legs wholesale (the Swoopes class: all four
    performance legs None, grade floated to the 50s-60s on context).
    """
    return any(s.get(k) is not None for s in seasons for k in keys)


def _is_role_season(season: Dict, pos: str) -> bool:
    """True when the player held a real role that season (v3.8 M2,
    strengthened per Kaedon 2026-11-24: the grade is a future
    outlook, not a career rank - "not performing young in college
    shouldn't hurt them" when they had no role yet).

    Mechanical, from the opportunity data profiles already carry. A
    season is a ROLE season when any of the position's opportunity
    signals clears a small floor (a fraction of a starter's season):
      QB: pass_attempts >= 100, or rush_attempts >= 80 (a featured
          wildcat/dual-threat role is a role)
      RB: rush_attempts + receptions >= 50
      WR: targets >= 30, or receptions >= 15, or receiving_yards >= 250
      TE: targets >= 20, or receptions >= 10, or receiving_yards >= 150
    (targets are absent from many historical profiles, so the
    reception/yardage signals carry the rule there.) One override:
    for RB/WR/TE, a season whose dominator_rating is under 0.04
    while team totals are on file is NOT a role season even when a
    raw count clears the floor - that is garbage-time accumulation
    on a real offense, not a role.
    """
    pos = pos.upper()
    if pos == "QB":
        return (_safe(season.get("pass_attempts")) >= 100
                or _safe(season.get("rush_attempts")) >= 80)
    if pos == "RB":
        featured = (_safe(season.get("rush_attempts"))
                    + _safe(season.get("receptions"))) >= 50
    elif pos == "WR":
        featured = (_safe(season.get("targets")) >= 30
                    or _safe(season.get("receptions")) >= 15
                    or _safe(season.get("receiving_yards")) >= 250)
    elif pos == "TE":
        featured = (_safe(season.get("targets")) >= 20
                    or _safe(season.get("receptions")) >= 10
                    or _safe(season.get("receiving_yards")) >= 150)
    else:
        return True
    if not featured:
        return False
    dom = season.get("dominator_rating")
    if dom is not None and _safe(dom) < 0.04 and (
            _safe(season.get("team_pass_yards")) > 0
            or _safe(season.get("team_total_yards")) > 0):
        return False
    return True


def _role_gated_seasons(seasons: List[Dict], pos: str) -> List[Dict]:
    """The seasons the performance legs may read (v3.8 M2 role gate).

    Seasons BEFORE the player's first role season are excused: he
    had not earned/been given the role yet, so those seasons are not
    performance evidence and must not drag the blends. The excuse is
    directional - once featured, every later season counts in full,
    including a role loss or a decline. A player who was never
    featured keeps every season (he grades on what exists; the gate
    excuses nothing and boosts nothing). Applied by score_prospect
    to the production and utilization reads, and thereby to the
    career aggregates inside them (the M3 receiving-share credit
    reads the production leg's seasons). Breakout is NOT gated: it
    is a change measure whose origin is the pre-role baseline (see
    the call site in score_prospect).
    """
    featured_years = [_safe(s.get("season")) for s in seasons
                      if _is_role_season(s, pos)]
    if not featured_years:
        return list(seasons)
    first = min(featured_years)
    if first <= 0:
        return list(seasons)
    return [s for s in seasons if _safe(s.get("season")) >= first]


def _rate(season: Dict, stored_key: str, num_key: str,
          den_key: str) -> Optional[float]:
    """A per-play rate: the stored rate when present, else derived
    from the season's counting stats, else None (no measurement).

    v3.8 (M1): the friendly defaults this replaces (YPA 7.0, comp
    62%, TD:INT 2.0, YPC 4.25, YPR 7.0/9.0/10.0) were scored as if
    measured whenever a stored rate was absent; a placeholder or
    fragment season read as a league-average efficiency season.
    """
    stored = season.get(stored_key)
    if stored is not None:
        return _safe(stored)
    if season.get(num_key) is None:
        return None  # numerator missing: no derivation, not a 0.0 rate
    den = _safe(season.get(den_key))
    if den > 0:
        return _safe(season.get(num_key)) / den
    return None


def _latest_season(seasons: List[Dict]) -> Optional[Dict]:
    if not seasons:
        return None
    # Prefer the latest season with actual stat data; the eval pipeline writes
    # placeholder rows for the draft year (e.g. 2027) that must not shadow the
    # real latest college season (e.g. 2026).
    with_data = [s for s in seasons if season_has_counting_stats(s)]
    pool = with_data or seasons
    return max(pool, key=lambda s: _safe(s.get("season"), 0))


# ─────────────────────────────────────────────────────────────────────────────
# Conference quality table
# Power 5 / big non-conf → higher competition score
# ─────────────────────────────────────────────────────────────────────────────

CONF_QUALITY: Dict[str, float] = {
    # Power 2 - abbreviations and full names (CFBD returns full names)
    "SEC":                       1.00,
    "Southeastern":              1.00,   # "Southeastern Conference"
    # v3.6: Big Ten split below the SEC (was 1.00, identical to the SEC):
    # with the competition scale's top anchor also at 1.00, every career
    # SEC/Big Ten player pinned the leg at exactly 100.
    "Big Ten":                   0.97,
    "Big 10":                    0.97,

    # Upper Power Tier
    "Big 12":                    0.90,
    "Big Twelve":                0.90,
    "ACC":                       0.88,
    "Atlantic Coast":            0.88,   # "Atlantic Coast Conference"

    # Legacy Pac (if still used)
    "Pac-12":                    0.89,
    "Pac 12":                    0.89,
    "Pacific-12":                0.89,

    # Power 4 (2024+ Big 12 expansion / new branding)
    "Big East":                  0.82,

    # Elite Independent
    "Notre Dame":                0.94,

    # Top G5
    "American":                  0.78,
    "American Athletic":         0.78,

    # Mid G5
    "Mountain West":             0.70,
    "Sun Belt":                  0.66,

    # Lower G5
    "MAC":                       0.60,
    "Mid-American":              0.60,
    "CUSA":                      0.56,
    "Conference USA":            0.56,

    # Independents (non-ND)
    "BYU":                       0.84,
    "Army":                      0.68,
    "Liberty":                   0.66,
    "UMass":                     0.60,
    "New Mexico State":          0.60,

    # Fallback bucket
    "FBS Independents":          0.70,
    "FBS Independent":           0.70,

    # FCS
    "FCS":                       0.48,
}

DEFAULT_CONF_QUALITY = 0.72   # unknown ≈ neutral, not penalised like a weak G5

# Competition leg top anchor (the _scale hi for conference quality).
# 1.03 through 3.8: set ABOVE the best real conference (SEC 1.00) as
# a v3.6 de-saturation, which capped the leg's maximum for any
# player at 94.83 - 100 unreachable by construction. v3.9 (ceiling
# redesign): the best real conference earns the leg's top (1.00);
# separation still comes from the quality table below it.
_COMPETITION_SCALE_HI = 1.00

# ─────────────────────────────────────────────────────────────────────────────
# Sagarin team-caliber dominator adjustment
# Applied multiplicatively to the WR/TE pass-share metric.
# Alabama 2020 (predictor ~99) → +6.47% cap.
# Non-D1 / unrated           → -9.3% floor.
# ─────────────────────────────────────────────────────────────────────────────
_SAGARIN_FBS_AVG = 75.0           # approximate mean Sagarin predictor for FBS teams
_SAGARIN_SCALE   = 0.0647 / 24.0  # ≈ 0.00270 per rating point above avg
                                   # (99 − 75) × 0.00270 = 0.0648 → capped at +6.47%

# ─────────────────────────────────────────────────────────────────────────────
# Scheme-inflation penalties
# High-volume / spread systems where college WR stats are poor NFL predictors.
# Applied as a multiplier to production AND efficiency scores for WRs only.
# Tennessee (Heupel air-raid): very high ADOT, schemed targets, historically
#   weak NFL translation for receivers outside the #1 option.
# Ole Miss (Kiffin spread): similar volume inflation at lower severity.
# ─────────────────────────────────────────────────────────────────────────────
# v3.6 (round-2 audit F7): SCHEME_INFLATION_SYSTEMS and its discount
# helper are deleted with their call sites. The two-team hard-coded
# discount was one of four team-context pricings; competition and
# environment are the surviving channels.


def _conf_quality(conference: Optional[str]) -> float:
    if not conference:
        return DEFAULT_CONF_QUALITY
    conf_lower = conference.lower()
    for k, v in CONF_QUALITY.items():
        if k.lower() in conf_lower:
            return v
    return DEFAULT_CONF_QUALITY


def _sagarin_dom_adj(rating: Optional[float], conference: Optional[str] = None) -> float:
    """
    Convert a Sagarin CFB predictor rating into a multiplicative dominator
    adjustment for WR/TE pass-share scoring.

    Bounds (user-calibrated):
        None + FCS/non-D1 conference → -9.3%  floor
        None + FBS conference        →  0.0%  neutral (Sagarin fetch failed)
        FBS average (~75)            →  0.0%
        Alabama 2020 (~99)           → +6.47% cap

    The conference-aware fallback prevents historical backtest years from
    incorrectly penalising every FBS player when the Sagarin page for that
    season year is unavailable.  _conf_quality() returns 0.48 for FCS and
    higher for all FBS conferences, so 0.50 is a clean dividing line.
    """
    if rating is None:
        if _conf_quality(conference) >= 0.50:
            return 0.0    # FBS team - Sagarin unavailable, apply no adjustment
        return -0.093     # Confirmed FCS / non-D1
    raw = (rating - _SAGARIN_FBS_AVG) * _SAGARIN_SCALE
    return max(-0.093, min(0.0647, raw))


# ─────────────────────────────────────────────────────────────────────────────
# Component scorers
# ─────────────────────────────────────────────────────────────────────────────

def _score_production_season(season: Dict, pos: str, skip_sagarin: bool = False,
                             _partial: bool = False) -> float:
    """
    Compute the raw production score (pre-transfer-penalty) for a single season.
    Returns the weighted component score (0-100).

    v3.4 (round 14): _partial marks the in-progress season in in-season
    mode. Counting-stat gates inside the season score (red-zone yardage
    gates, QB dual-threat TD gates) are paced by min(1, GP/12) so a partial
    season is not held to full-season counting thresholds. All rate legs
    are untouched.

    v3.6 (round-2 audit F16): sub-4-game seasons no longer divide by
    12 (the old cliff: 200 yards in 3 recorded games rated 16.7/g,
    in 4 games 50/g). The divisor is max(GP, 4) and the resulting
    score shrinks toward the leg neutral (52.0) by GP/4, so a tiny
    recorded sample reads as weak evidence, not as a rate.
    """
    _prod = _score_production_season_raw(season, pos, skip_sagarin, _partial)
    _gp_raw = _safe(season.get("games_played"), 12)
    if _gp_raw < 4:
        _prod = 52.0 + (_prod - 52.0) * (_gp_raw / 4.0)
    # v3.11 round 10 (discrepancy audit P1): in per-season mode
    # the crowded-room multiplier prices HERE, on each season's
    # own depth chart, instead of once on the latest season
    # after the blend (see DEPTH_CFG). A partial (in-progress)
    # season is exempt - its room is still forming.
    if DEPTH_CFG.get("per_season") and not _partial:
        _prod = _clip(_prod * _depth_multiplier_for_season(season, pos))
    # v3.11 round 12 (Phase B / discrepancy audit P1, Kaedon's
    # ruling 2026-11-25): the measured room price (ROOM_CFG)
    # applies here, on each season's own room aggregates. A
    # partial (in-progress) season is exempt - its room is
    # still forming - and a season carrying no aggregates
    # prices at 1.0 (fail closed, never a penalty).
    if not _partial:
        _prod = _clip(_prod * _room_multiplier_for_season(season, pos))
    return _prod


def _score_production_season_raw(season: Dict, pos: str, skip_sagarin: bool = False,
                                 _partial: bool = False) -> float:
    # Guard against bad games-played values (0/1) that can explode per-game rates
    # and incorrectly saturate production to 100.
    gp_raw = _safe(season.get("games_played"), 12)
    gp = max(gp_raw, 4)  # v3.6 (F16): was "gp_raw if gp_raw >= 4 else 12"
    _pace = min(1.0, gp / 12.0) if _partial else 1.0

    if pos == "WR":
        rec_yds_pg    = _safe(season.get("receiving_yards")) / gp
        rec_tds_pg    = _safe(season.get("receiving_tds"))   / gp
        dom           = _safe(season.get("dominator_rating"))
        total_rec_yds = _safe(season.get("receiving_yards"))
        total_rec_tds = _safe(season.get("receiving_tds"))
        team_pass_yds = _safe(season.get("team_pass_yards"))
        sag_adj       = 0.0 if skip_sagarin else _sagarin_dom_adj(
                            season.get("sagarin_team_rating"), season.get("conference"))

        # Pass-share dominator: rec yards as % of team passing yards, adjusted
        # for team caliber via Sagarin predictor rating.
        # Falls back to legacy total-offense dominator if team_pass_yards absent.
        if team_pass_yds > 0:
            pass_share = (total_rec_yds / team_pass_yds) * (1 + sag_adj)
            dom_score  = _scale(pass_share, 0.08, 0.40)  # v3.6: 0.35 -> 0.40 (recorded-best-plus-margin)
        else:
            dom_score  = _scale(dom * (1 + sag_adj), 0.08, 0.40)

        prod = (
            _scale(rec_yds_pg, 32,  120) * 0.40 +  # v3.6: top anchor 100 -> 120 (recorded-best-plus-margin)
            _scale(rec_tds_pg, 0.25, 0.9) * 0.30 +
            dom_score                      * 0.30
        )
        # YAC bonus: dynamic after-catch ability is a strong NFL translation signal
        yac = _safe(season.get("yards_after_catch_per_reception"))
        if yac >= 7.0:
            prod = _clip(prod * 1.10)
        elif yac >= 5.5:
            prod = _clip(prod * 1.05)
        # Red zone proxy: TDs per 100 receiving yards - rewards goal-line
        # separators. v3.4 recalibration: the old thresholds (8.0 / 5.5)
        # sat 3-6x above the best recorded mark, so the bonus could NEVER
        # fire. Across the recorded fixture + panel seasons at 200+
        # receiving yards, the rate runs 0.00-1.62 with a median near
        # 0.93; the new tiers sit at the recorded distribution's top
        # quarter (1.15) and elite edge (1.5: Pitts 2020 scored 1.56,
        # Green 2025 1.62).
        if total_rec_yds >= 200 * _pace:
            rz_rate = total_rec_tds / total_rec_yds * 100
            if rz_rate >= 1.5:   prod = _clip(prod * 1.06)
            elif rz_rate >= 1.15: prod = _clip(prod * 1.03)
        # v3.0: G5 production discount REMOVED. Competition quality is already
        # captured by calc_competition_score, so the additional production cut
        # (up to -25%) was double-jeopardy. One mechanism, not two.
        pass
        return prod

    elif pos == "RB":
        rush_yds_pg = _safe(season.get("rush_yards"))     / gp
        rec_yds_pg  = _safe(season.get("receiving_yards")) / gp
        all_yds_pg  = (
            _safe(season.get("rush_yards")) + _safe(season.get("receiving_yards"))
        ) / gp
        tds_pg      = (
            _safe(season.get("rush_tds")) + _safe(season.get("receiving_tds"))
        ) / gp
        dom         = _safe(season.get("dominator_rating"))
        # v3.8 (M1): yards per carry from the counting stats when the
        # stored rate is absent, never a scored 0.0 for a missing rate.
        _ypc = _rate(season, "yds_per_carry", "rush_yards", "rush_attempts")
        ypc         = _ypc if _ypc is not None else 0.0

        prod = (
            _scale(rush_yds_pg, 20,  120) * 0.18 +
            _scale(all_yds_pg,  30,  150) * 0.17 +
            _scale(tds_pg,       0.4,  1.6) * 0.25 +
            _scale(dom,          0.12, 0.50) * 0.15 +  # ceiling lowered: 0.45+ dom is elite
            _scale(ypc,          3.5,  7.0) * 0.25     # ceiling lowered: 7.0 YPC is generational
        )
        # v3.2 V8: genuine pass-catching profiles move 0.05 of weight from
        # the rush anchor to the all-purpose anchor.
        _rec_share_v8 = rec_yds_pg / max(all_yds_pg, 1.0)
        if _rec_share_v8 >= 0.20:
            prod += (_scale(all_yds_pg, 30, 150) - _scale(rush_yds_pg, 20, 120)) * 0.05
        # Receiving tiers: require meaningful rec share so dedicated pass-catchers
        # (Coleman ~34%) are rewarded differently from incidental receivers (Johnson ~20%).
        # Use an additive multiplier budget (single final clip) to reduce 100-point
        # ceiling effects from stacked multiplicative boosts.
        mult = 1.0
        rec_share = rec_yds_pg / max(all_yds_pg, 1.0)
        if rec_yds_pg >= 20 and rec_share >= 0.28:
            mult += 0.08   # dedicated pass-catcher
        elif rec_yds_pg >= 20 and rec_share >= 0.22:
            mult += 0.06   # strong receiving back
        elif rec_yds_pg >= 15:
            mult += 0.03   # incidental receiver
        if ypc >= 6.0:
            mult += 0.03
        if dom >= 0.30:
            mult += 0.05
        # Red zone proxy: TDs per 100 total yards.
        # Realistic benchmarks for RBs: 1.0–2.5 range for elite scorers.
        total_yds = _safe(season.get("rush_yards")) + _safe(season.get("receiving_yards"))
        total_tds = _safe(season.get("rush_tds"))   + _safe(season.get("receiving_tds"))
        if total_yds >= 300 * _pace:
            rz_rate = total_tds / total_yds * 100
            if rz_rate >= 2.0:
                mult += 0.03
            elif rz_rate >= 1.0:
                mult += 0.015

        mult = min(mult, 1.20)  # slightly higher cap to allow elite all-around profiles
        return _clip(prod * mult)

    elif pos == "QB":
        pass_yds_pg    = _safe(season.get("pass_yards")) / gp
        tds_pg         = _safe(season.get("pass_tds"))   / gp
        comp_pct       = _safe(season.get("completion_pct"), 60.0)
        ypa            = _safe(season.get("yds_per_attempt"), 7.0)
        td_int         = _safe(season.get("td_int_ratio"),    2.0)
        rush_yds_pg    = _safe(season.get("rush_yards")) / gp
        rush_tds_season = _safe(season.get("rush_tds"))

        # Rushing QBs: elite college rushers (Lamar, Kyler, Jalen Hurts) are
        # systematically penalised by pure passing metrics. Rushing is a direct
        # fantasy component in the NFL, not just a tie-breaker.
        # v3.6 (round-2 audit F6): production prices VOLUME only (pass
        # yards/game and pass TDs/game, equal halves). The rate trio
        # (YPA, completion%, TD:INT) used to be priced here AND in
        # efficiency - about 28% of a QB's grade paid the same three
        # numbers twice. Efficiency is the QB's rate leg (the model is
        # efficiency-first at QB); it keeps the trio alone now.
        pass_comp = (
            _scale(pass_yds_pg, 150, 285) * 0.50 +  # v3.2 V5 anchors
            _scale(tds_pg,        1.5,  2.8) * 0.50
        )
        rush_comp = _scale(rush_yds_pg, 10.0, 85.0)  # 0–100; Lamar ~90+ yd/g → near 100
        # v3.11 round 8 (QB_PROD_CFG): the rush-TD rate can
        # enrich the rushing component - goal-line rushing is
        # workload the yards rate alone understates.
        _rtd_w = float(QB_PROD_CFG.get("rush_td_weight", 0.0) or 0.0)
        if _rtd_w > 0.0:
            _rtd_lo, _rtd_hi = QB_PROD_CFG.get(
                "rush_td_anchors", (0.1, 1.0))
            rush_comp = ((1.0 - _rtd_w) * rush_comp + _rtd_w
                         * _scale(rush_tds_season / gp,
                                  float(_rtd_lo), float(_rtd_hi)))

        # v3.2 R7-A1: max(pass, blend). A perfect passing season is a
        # 100 production season whether or not the QB also ran.
        # v3.6 (F6): the blend share rises 15% -> 25%: rushing is priced
        # HERE, once, on its anchored scale. (Translation's old flat
        # +3.5/+1.5 rushing add is deleted; a flat add cannot be
        # reproduced through an anchored leg at production's weight,
        # and one anchored price is the audit's requirement.)
        _rshare = float(QB_PROD_CFG.get("rush_share", 0.25))
        prod = _clip(max(pass_comp,
                         pass_comp * (1.0 - _rshare)
                         + rush_comp * _rshare))
        # v3.11 round 8 (QB_PROD_CFG offense_max): the
        # total-offense price - the same season's output read
        # split-blind (see the cfg). A third honest read under
        # the same max() idiom, not a replacement: the passing
        # price stands for pocket passers because the anchors
        # are percentile-matched to theirs.
        if QB_PROD_CFG.get("offense_max"):
            _oy_lo, _oy_hi = QB_PROD_CFG.get(
                "offense_yds", (195.0, 350.0))
            _ot_lo, _ot_hi = QB_PROD_CFG.get(
                "offense_tds", (1.7, 3.3))
            _tot_yds_pg = (_safe(season.get("pass_yards"))
                           + _safe(season.get("rush_yards"))) / gp
            _tot_tds_pg = (_safe(season.get("pass_tds"))
                           + _safe(season.get("rush_tds"))) / gp
            _off_comp = (
                _scale(_tot_yds_pg, float(_oy_lo),
                       float(_oy_hi)) * 0.50
                + _scale(_tot_tds_pg, float(_ot_lo),
                         float(_ot_hi)) * 0.50)
            prod = _clip(max(prod, _off_comp))

        # Extra multiplier for elite rushing + TDs (dual-threat upside)
        # v3.4: in-season, the season TD total is paced to a 12-game
        # equivalent before it is compared to the counting thresholds.
        _rush_tds_cmp = rush_tds_season / _pace if _pace > 0 else rush_tds_season
        if rush_yds_pg >= 70 and _rush_tds_cmp >= 8:
            prod = _clip(prod * 1.06)
        elif rush_yds_pg >= 50 and _rush_tds_cmp >= 5:
            prod = _clip(prod * 1.03)
        return prod

    elif pos == "TE":
        rec_yds_pg    = _safe(season.get("receiving_yards")) / gp
        rec_tds_pg    = _safe(season.get("receiving_tds"))   / gp
        dom           = _safe(season.get("dominator_rating"))
        rec_pg        = _safe(season.get("receptions"))       / gp
        total_rec_yds = _safe(season.get("receiving_yards"))
        total_rec_tds = _safe(season.get("receiving_tds"))
        team_pass_yds = _safe(season.get("team_pass_yards"))
        sag_adj       = 0.0 if skip_sagarin else _sagarin_dom_adj(
                            season.get("sagarin_team_rating"), season.get("conference"))

        # Pass-share dominator for TEs - same logic as WR, adjusted scale for elite TEs
        # (Elite TEs like Bowers can command WR-level share of passing yards).
        if team_pass_yds > 0:
            pass_share = (total_rec_yds / team_pass_yds) * (1 + sag_adj)
            dom_score  = _scale(pass_share, 0.04, 0.40)  # v3.2 R7-B: beyond McBride's 0.36 record share
        else:
            dom_score  = _scale(dom * (1 + sag_adj), 0.05, 0.25)  # Increased upper bound from 0.22

        # v3.13 (Phase 7, lever L2): the yards and TD terms are
        # ALL-PURPOSE - receiving plus rushing - the mirror of
        # the RB production leg's all-purpose term. A TE's
        # rushing usage (Bowers' 19 career carries / 5 TD,
        # Warren's 2024) was invisible to a receiving-only read.
        # The elite gates below still read receiving-only rates:
        # they are receiving-elite gates, not all-purpose ones.
        prod = (
            _scale((total_rec_yds + _safe(season.get("rush_yards"))) / gp, 20,  100) * 0.27 +  # v3.2 R7-B: beyond Pitts' 96.3/g record pace
            _scale((total_rec_tds + _safe(season.get("rush_tds"))) / gp, 0.12, 0.65) * 0.25 +  # v3.6: top anchor 0.5 -> 0.65 (was pinning ordinary hot seasons)
            dom_score                      * 0.35 +  # pass share: the stable TE signal
            _scale(rec_pg,     1.0,  8.0) * 0.13    # v3.2 R7-B: beyond McBride's 7.5/g
        )
        # v3.6: the elite bonuses are RATE gates, and unlike the counting
        # gates above they were never paced - a short in-progress season
        # (e.g. 6 games) could clear both at a hot per-game rate. They now
        # require a completed season, or 8+ games played in-progress.
        _elite_ok = (not _partial) or (gp_raw >= 8)
        # Elite TE bonus: players like Brock Bowers who produce at WR levels get extra credit
        # v3.13 (Phase 4, the OR-gate): the two elite multipliers
        # fire on the RATE path OR the share path. Jointly gated,
        # they paid only the share shape: Pitts' 2020 - the best
        # per-game TE season on the frame (96.3 yds/g) - earned
        # no elite bonus at all because Florida's share was
        # spread across a loaded room (his dom_score priced
        # 35.8, under both bars), while a .30-share hybrid fired
        # both. Each path's thresholds are unchanged; either
        # elite signal now suffices. See prospect-runs/
        # rb-ceiling-audit-2026-11-25.md S5.
        if _elite_ok and (rec_yds_pg >= 55 or dom_score >= 47):  # v3.2 R7-B: same physical share, rescaled
            prod = _clip(prod * 1.18)  # 18% bonus for WR-level production (lowered thresholds)

        # Special bonus for historically elite TEs (multiple major award winners)
        # This captures players like Brock Bowers who won best college TE twice
        if _elite_ok and (rec_yds_pg >= 50 or dom_score >= 42):  # v3.2 R7-B: same physical share, rescaled
            prod = _clip(prod * 1.08)  # Additional 8% bonus for elite production
        # YAC bonus: move TEs who create after the catch are more NFL-translatable
        te_yac = _safe(season.get("yards_after_catch_per_reception"))
        if te_yac >= 5.0:
            prod = _clip(prod * 1.08)
        elif te_yac >= 3.5:
            prod = _clip(prod * 1.04)
        # Red zone proxy: TE goal-line usage is extremely valuable in
        # NFL. v3.4 recalibration: same defect and same recorded
        # distribution as the WR proxy (old thresholds 10.0 / 7.0 could
        # never fire; recorded rates top out at 1.62).
        if total_rec_yds >= 200 * _pace:
            rz_rate = total_rec_tds / total_rec_yds * 100
            if rz_rate >= 1.5:
                prod = _clip(prod * 1.05)
            elif rz_rate >= 1.15:
                prod = _clip(prod * 1.02)
        return prod

    return 52.0


def _eval_metric_value(eval_metrics: Optional[Dict], name: str, min_confidence: float = 0.0):
    """
    Safely extract a metric value from an eval_metrics dict.

    eval_metrics has the shape: {metric_name: {value: ..., confidence: ..., ...}}
    Returns None if metric absent, None value, or confidence below threshold.
    """
    if not eval_metrics:
        return None
    entry = eval_metrics.get(name)
    if not isinstance(entry, dict):
        return None
    confidence = entry.get("confidence") or 0.0
    if confidence < min_confidence:
        return None
    return entry.get("value")


def _eval_metric_percent(
    eval_metrics: Optional[Dict],
    name: str,
    min_confidence: float = 0.0,
) -> Optional[float]:
    """
    Read an eval metric and normalize to 0-100 percent scale.
    Accepts either [0,1] rates or already-scaled percentages.
    """
    val = _eval_metric_value(eval_metrics, name, min_confidence=min_confidence)
    if val is None:
        return None
    f = _safe(val, default=0.0)
    return f * 100.0 if 0.0 <= f <= 1.0 else f


def _rb_receiving_share_credit(seasons: List[Dict]) -> float:
    """v3.8 (under-rating audit M3): bounded receiving-share credit
    inside the RB production leg.

    The audit found the model's only RB receiving channel was a YAC
    proxy capped at +/-10% of production, while realized residuals ran
    monotone in career receiving share (low tercile -5.4, high tercile
    +8.3 percentile points). The gated probe priced the share directly:
    credit = 10 * clip((share - 0.20) / 0.20, -1, +1) production
    points on the CAREER share (receiving yards / scrimmage yards),
    monotone gains at every cap under both share definitions, top of
    the board unhurt. Returns 0.0 when there is no scrimmage volume
    to take a share of.

    The share is computed over the seasons the production leg
    receives; score_prospect role-gates that list (v3.8 M2), so
    pre-role seasons - where a not-yet-featured back's share is
    mechanically near zero - do not dilute the career share.
    """
    rec = sum(_safe(s.get("receiving_yards")) for s in seasons)
    scrim = sum(_safe(s.get("rush_yards")) + _safe(s.get("receiving_yards"))
                for s in seasons)
    if scrim <= 0:
        return 0.0
    share = rec / scrim
    return 10.0 * max(-1.0, min(1.0, (share - 0.20) / 0.20))


def _production_zero_verdict(score: float, read_seasons: List[Dict],
                             pos: str,
                             skip_sagarin: bool = False,
                             all_seasons: Optional[List[Dict]] = None,
                             draft_year: Optional[int] = None) -> Optional[float]:
    """v3.8 (under-rating audit M1, class L2; TE fix, post-gate):
    verdict on an exact 0.0 production score.

    A 0.0 with games_played known is a measurement and stands. A 0.0
    with GP missing on every season the leg read is ambiguous: it is
    either the L2 artifact (the defaulted 12-game divisor meeting
    the leg's hard rate floors - real 1,500-yard passers stored
    exact 0.0) or a genuinely negligible resume priced through the
    same defaulted divisor (a tight end with 8 career catches reads
    0.0 at ANY divisor). The first build converted every such zero
    to None; at TE, where volumes are small and historical profiles
    systematically lack GP, that deleted measured zeros wholesale
    and redistribution floated no-resume profiles to mid-board
    (Kevin Rader 44.5 -> 65.6 on a 126-yard career), costing TE
    Spearman .065. The verdict separates the two: re-score the read
    seasons at GP=4, the most generous divisor the model ever uses
    (the F16 minimum sample), and rebuild the leg's blend. The zero
    is a divisor artifact only when the same evidence prices at or
    above the model's neutral (52.0) at that divisor - the recorded
    evidence describes a real season, so the zero is arithmetically
    impossible for it and the leg is missing (None). Anything less
    and the zero is divisor-robust - a thin resume is a thin resume
    at every plausible divisor - and it stands as a measurement.
    (An earlier 5.0 bar failed exactly here: a 95-yard season with
    one touchdown prices ~9 at GP=4, so sub-neutral evidence was
    still being erased as "missing".)

    Truncation: a zero from a file that ENDS two or more seasons
    before the player's draft class is missing data by construction
    - the store holds nothing from the seasons that decide a grade
    (Dallas Goedert's file stops in 2016 for the 2018 class: his
    senior season, the one that made him a second-round pick, was
    never loaded; Carson Wentz's file is a single 2014 cameo for
    the 2016 class). No divisor question arises: whatever those
    seasons held, this leg did not measure them, so the leg is None
    regardless of the recheck. A complete file that merely ends the
    season before the draft (the normal shape) is untouched.
    """
    if score != 0.0 or not read_seasons:
        return score
    if draft_year and all_seasons:
        _years = [s.get("season") for s in all_seasons
                  if s.get("season") is not None]
        if _years and draft_year - max(_years) >= 2:
            return None
    if any(s.get("games_played") is not None for s in read_seasons):
        return score
    rescored = []
    for s in read_seasons:
        s4 = dict(s)
        s4["games_played"] = 4
        rescored.append(_score_production_season(s4, pos, skip_sagarin))
    if len(rescored) == 1:
        recheck = rescored[0]
    else:
        recheck = max(rescored) * 0.90 + min(rescored) * 0.10
    return score if recheck < 52.0 else None


# v3.11 revision (Kaedon 2026-11-24, the Jameson Williams case):
# production prices an EVIDENCE BASE, not only a peak. The blend
# above takes 90% of the best season, so a profile with exactly
# one featured season carried the same production leg as a
# profile that sustained the level - one monster year at Alabama
# priced as the decade's #2 receiver while a two-year elite
# (Harrison Jr.) priced lower on softer legs elsewhere. The cap
# limits how much of the leg's SPIKE (the part above
# spike_level) a single featured season can carry:
#   capped = spike_level + (prod - spike_level) * spike_keep
# A second featured season releases the cap in full - sustained
# excellence is the evidence. So does a single featured season
# played at an elite-underclassman age (anchored breakout age
# <= release_age, the breakout leg's own youth zone): a
# 19-year-old dominating is its own second evidence, which is
# what keeps the Chase shape (one rotational year, one monster
# year at 19, opt-out) whole. Nothing here scores a buried or
# pre-role season AGAINST anyone - the role gate already excused
# those, and the cap only declines to fully credit an
# uncorroborated spike (Kaedon's future-outlook doctrine: early
# low-role seasons are not evidence against a prospect; one
# featured season is simply less evidence FOR him). Shipped
# dose: a single featured season carries production to 75 plus
# a quarter of the spike above it. On the recorded population
# the cap binds on 14 of 2,265 profiles - one-year air-raid
# QBs and one-season wonders whose NFL outcomes agreed with
# the haircut - and the gate improves at every position.
PROD_EVIDENCE_CFG = {
    "enabled": True,
    "spike_level": 75.0,
    "spike_keep": 0.25,
    "release_age": 20.0,
    # v3.11 round 9 (discrepancy audit P2/P4): the two-season
    # release used to fire on ANY second featured season, so a
    # committee year (Abanikanda's 98-carry 2021, Hyatt's
    # 21-catch 2021) fully released a one-season spike. When
    # positive, release requires the SECOND-best featured
    # season to itself price at or above this level - a spike
    # is corroborated by a second season that was itself
    # good, not merely present. 0.0 = presence release (the
    # pre-round-9 behavior). Swept in round9_lab.
    "release_min_price": 0.0,
    # v3.12 (RB-2b change 1): at RB the release is corroboration-
    # priced. Round 9 shipped the machinery at dose 0.0 because a
    # priced release taxed WR/TE profiles whose second featured
    # season was legitimately modest (Chase / Nabers / Engram);
    # inside an RB-only architecture that objection does not
    # exist, and the RB evidence is one-sided: the cap's presence
    # release is what let one-season spikes (Abanikanda's 2022,
    # corroborated only by a 2021 that prices 28.50) print at
    # full value. An RB spike releases only when the second-best
    # featured season itself prices >= 55 (Javonte 73.55 and
    # Mixon 70.36 still release; Abanikanda 28.50 does not).
    # Other positions keep the global 0.0 presence release,
    # byte-identical.
    # v3.13 (Phase 7, lever L4): TE takes the same corroboration-
    # priced release at 55. The RB precedent's objection at
    # WR/TE was that a priced release taxed files whose second
    # season was legitimately modest; the TE census says the
    # presence release is instead a free pass for one-year
    # files - the profiles it alone releases price their
    # second-best season at 43.30 / 42.61 / 36.35 (Duarte /
    # Engram / Loveland class). Labelled a tightening: it
    # removes presence-release from spike files; corroborated
    # files (Bowers' second-best 68.17) are untouched. WR keeps
    # the presence release. See prospect-runs/
    # bowers-mccaffrey-residuals-2026-11-25.md (L4).
    "release_min_price_by_pos": {"RB": 55.0, "TE": 55.0},
}


def _production_evidence_cap(prod: float, seasons: List[Dict],
                             pos: str, age: Optional[float],
                             draft_year: Optional[int],
                             skip_sagarin: bool = False) -> float:
    """Apply the PROD_EVIDENCE_CFG cap to a finished production
    score. `seasons` is the role-gated list the leg read."""
    if not PROD_EVIDENCE_CFG.get("enabled") or prod is None:
        return prod
    level = float(PROD_EVIDENCE_CFG.get("spike_level", 85.0))
    if prod <= level:
        return prod
    featured = [s for s in seasons if _is_role_season(s, pos)]
    if len(featured) >= 2:
        _rmp = float((PROD_EVIDENCE_CFG.get(
            "release_min_price_by_pos") or {}).get(
                pos, PROD_EVIDENCE_CFG.get("release_min_price", 0.0)))
        if _rmp <= 0:
            return prod
        _prices = sorted(
            (_score_production_season(s, pos, skip_sagarin)
             for s in featured),
            reverse=True)
        if len(_prices) >= 2 and _prices[1] >= _rmp:
            return prod
    if len(featured) == 1 and age is not None:
        # Youth release: the single featured season was played
        # at an elite-underclassman age (draft-anchored, the
        # breakout leg's convention; legacy anchor when no
        # draft year is supplied).
        _s_year = _safe(featured[0].get("season"), 0)
        if _s_year > 0:
            if draft_year:
                _age_in = age - (draft_year - _s_year) + 0.45
            else:
                _last = max(_safe(s.get("season"), 0) for s in seasons)
                _age_in = age - (_last - _s_year)
            if _age_in <= float(PROD_EVIDENCE_CFG.get(
                    "release_age", 20.0)):
                return prod
    keep = float(PROD_EVIDENCE_CFG.get("spike_keep", 0.5))
    return _clip(level + (prod - level) * keep)


# v3.11 round 5 (Kaedon 2026-11-24: "fix the double injury
# count"): an availability season is charged ONCE. Production
# prices what a player did WHEN he played (per-game rates);
# durability prices WHETHER he played. Before this round, a
# featured season cut short by injury still entered the
# production blend at the latest-season share - its depressed
# price dragging the leg - while the durability leg priced the
# same missed games a second time. The blend now drops the
# latest-season share when ALL of these hold:
#   - the latest featured (role-gated) season is an
#     availability season under this config (a missing GP is
#     not evidence of absence and never fires);
#   - it is not the player's best priced season in the blend -
#     a short season that IS the peak (Pitts' 8-game 2020) is
#     his best evidence and is never deleted;
#   - it is not the in-progress season (the F2 shrink already
#     prices a partial season against the best completed
#     baseline; a mid-season GP is a calendar fact, not an
#     availability signal).
# Production then prices off the best season alone, and
# durability remains the single availability charge. The
# threshold was swept on the GP frame (absolute GP <= 8 / 9 /
# 10 and a team-share mode, GP < 75% of team games): every
# definition gated within +/- .002 of the round-4 control -
# the double charge the old frame suggested was already
# mostly priced away by honest per-game denominators, so the
# ship is the conservative absolute line with the smallest
# blast radius and the closest gate, not the line that
# reaches any one name. The share mode is the purer
# definition (a 9-of-9 COVID season is not an absence) but
# season rows carry no team-games field anywhere in the live
# pipeline, so it would ship as a no-op; it stays in the
# config for the day that field exists. See the round-5 note
# (S13 of the ceiling-attainability record).
PROD_AVAILABILITY_CFG = {
    "enabled": True,
    "mode": "gp",
    "gp_max": 8,
    "share_max": 0.75,
}


# v3.11 round 6 (step C1): the production blend's peak-season
# KEY. The blend prices seasons per game, but the completed-
# season path selected its anchor season on RAW YARDS - so a
# longer season won the anchor slot on schedule volume and
# then priced worse per game than a shorter, better season
# that went unused (the Bowers case, S13.1: 942 yds / 15 GP
# prices 68.17 and anchors at the 0.9 share; 882 / 14 GP
# prices 83.05 and is never read). The shipped key is the
# v3.4 in-season RATE key (totals / max(GP, 12)) with a price
# tie-break, "rate_price": the anchor is the player's best
# per-game season, and seasons whose rates sit within the
# tie band of the best rate - measurement-identical rates -
# are separated by their full production PRICE, because the
# price sees the terms the yardage rate cannot (TD rate,
# dominator). The round-6 sweep (S14) rejected the pure keys
# on their own evidence: raw yards buries the best season
# under schedule volume; the pure price canonizes 4-6 game
# seasons (a 4-game TE season anchoring at 0.9, +35 grade)
# and costs RB Poor 5.2 -> 6.9; the pure rate hair-flips
# near-tied seasons whose prices differ by ~50 (the rate is
# TD-blind); and a GP floor on the anchor - the first fix
# tried - deletes seasons both other keys agree are the
# player's best evidence (the Fryfogle case), against the
# model's own never-delete-the-best-season doctrine. The
# in-season path is untouched: it already reads the rate key.
PROD_PEAK_CFG = {
    "by": "rate_price",
    "tie_band": 0.01,
}


# v3.11 round 8 (discrepancy audit P3 / S11.4): the QB
# production season price. Through round 7 the price was the
# passing volume price with a bounded rushing garnish:
# max(pass, 0.75 * pass + 0.25 * rush_rate), so a season's
# rushing output could only ever lift the price toward the
# passing read, never stand as production in its own right -
# a dual-threat season whose passing volume was ordinary
# priced as ordinary production no matter how much offense
# ran through the QB's legs (the Josh Allen / Lamar Jackson
# defect family: production 2.41 / 42.87 against outcome
# ranks #1 / #7; NFL.com missed them too at 6.40, our miss
# was the same direction at roughly triple the depth).
# The total-offense price below is the third honest read of
# the same season: total yards per game and total TDs per
# game, split-blind - the season produced what it produced,
# however the yards were divided between arm and legs. Its
# anchors sit at the same percentile positions of the
# recorded QB role-season distribution as the passing
# anchors occupy in theirs (pass 150/285 ~ p15/p82 of pass
# yards per game; total 195/350 ~ p15/p85 of total yards
# per game; TDs likewise), so a pure passing season prices
# the same under either read and the max() form (the R7-A1
# idiom: a perfect season is a 100 under any honest read)
# leaves pocket passers untouched. rush_td_weight enriches
# the rushing component with the rush-TD rate (a workload
# read: attempts are not in this leg, the TD rate is the
# goal-line workload signal); 0.0 keeps the pure yards read.
#
# SHIPPED DOSE (round 8, S16): rush_share 0.55. The dose
# sweep on the GP frame: share .25 (the old garnish) ->
# .35 -> .45 -> .50 -> .55 improves QB Pearson at every
# step (.3951 -> .3998 -> .4044 -> .4066 -> .4087) with
# Spearman flat-to-up (.4454 -> .4491) and the QB Poor
# tier and pool median byte-identical; .65 buys +.004
# more Pearson at the price of the round's one Poor
# crossing (Joshua Dobbs 67.94 -> 70.48 in the bottom
# outcome quintile), so the sweep stops at .55, the last
# clean dose. At .55 the blend candidate prices a season's
# rushing output as the near-equal partner of its passing
# output while the max() form still guarantees a pure
# passing season its full passing price. Lamar Jackson
# +7.70 and Jalen Hurts +2.15 on the frame as carried;
# the two flagship names' remaining distance is a frame
# data artifact, not dose (see S16.2: their stored GP
# denominators are corrupt - Allen's 2016 reads GP 27 for
# a 14-game season - and on corrected denominators Lamar
# prices 80.81 and Allen's production reads an honest
# 39.61; the GP repair is data-debt work, not model work).
# HELD, NOT SHIPPED: offense_max (the total-offense third
# read) posted the round's best raw gate (P .4099 /
# S .4548 at share .25, Hurts +3.65, Dak Prescott +2.41)
# but crossed Dobbs into the Poor tier at every dose -
# the failure mode this round's brief bars - so it stays
# behind its flag as Kaedon's ruling. rush_td_weight was
# inert at every dose (the TD rate never binds beyond the
# yards read inside a max() at these shares) and stays 0.
QB_PROD_CFG = {
    "rush_share": 0.55,
    "offense_max": False,
    "offense_yds": (195.0, 350.0),
    "offense_tds": (1.7, 3.3),
    "rush_td_weight": 0.0,
    "rush_td_anchors": (0.1, 1.0),
}


def _production_peak_by_price(seasons, pos, skip_sagarin, default):
    """(peak_season, best_price): the season whose production
    PRICE is highest, first such season on ties. Falls back to
    (default, None) when no season prices."""
    _bs, _bv = None, None
    for _s in seasons:
        _v = _score_production_season(_s, pos, skip_sagarin)
        if _bv is None or _v > _bv:
            _bs, _bv = _s, _v
    if _bs is None:
        return default, None
    return _bs, _bv


def _is_availability_season(season: Dict) -> bool:
    """True when the season is an availability season under
    PROD_AVAILABILITY_CFG: a featured season the player
    missed a real share of. Missing GP never fires - absence
    of evidence is not evidence of absence."""
    if not PROD_AVAILABILITY_CFG.get("enabled"):
        return False
    gp = season.get("games_played")
    if gp is None:
        return False
    if PROD_AVAILABILITY_CFG.get("mode") == "share":
        tg = season.get("team_games")
        if tg is None or _safe(tg) <= 0:
            return False
        return _safe(gp) / _safe(tg) < float(
            PROD_AVAILABILITY_CFG.get("share_max", 0.75))
    return _safe(gp) <= float(PROD_AVAILABILITY_CFG.get("gp_max", 10))


def calc_production_score(
    seasons: List[Dict],
    position: str,
    eval_metrics: Optional[Dict] = None,
    skip_sagarin: bool = False,
    _inseason_year: Optional[int] = None,
    _draft_year: Optional[int] = None,
    _season_elapsed: Optional[int] = None,
    _age: Optional[float] = None,
) -> Optional[float]:
    """
    Per-game production vs position-specific elite thresholds.
    Uses a blend of the best season and latest season to capture both
    peak value and recent performance.

    Transfer penalty: Players who transfer to weaker conferences get
    production discounted, as stats may be inflated by weaker competition.

    eval_metrics (optional): If the evaluation pipeline produced yprr/yac_per_att
    metrics, these are used to apply a confidence-weighted production adjustment:
    - WR/TE: yprr proxy ≥ 1.8 boosts score; < 1.2 deflates slightly (±15% max)
    - RB:    yac_per_att proxy adjusts production ±10%
    The adjustment is capped intentionally to prevent proxies from dominating.
    """
    if not seasons:
        return 52.0  # pre-draft neutral: unknown ≠ bad

    pos = position.upper()

    # v3.8 (M1): seasons exist but none of the position-relevant
    # counting fields was ever recorded (placeholder / fragment
    # rows) - the leg has nothing to measure. Return None so
    # score_prospect redistributes the weight instead of scoring
    # absent data as a real season of zeros. Recorded zeros are
    # measurements and score (see _has_recorded_volume).
    if pos in _PROD_VOLUME_KEYS and not _has_recorded_volume(
            seasons, _PROD_VOLUME_KEYS[pos]):
        return None

    ls  = _latest_season(seasons) or {}
    gp  = max(_safe(ls.get("games_played"), 12), 1)

    # v3.6 (round-2 audit F7): the transfer penalty/bonus is deleted.
    # Team context was priced four ways (competition leg, environment
    # leg, scheme discount, this conference-delta multiplier on
    # production, up to -35%/+12%). Two channels remain: the
    # competition leg and the environment leg.

    if pos not in ("WR", "RB", "QB", "TE"):
        return 40.0

    # With only one season there is nothing to blend - just score it directly.
    if len(seasons) == 1:
        _single = _score_production_season(
            ls, pos, skip_sagarin,
            _partial=(_inseason_year is not None
                      and ls.get("season") == _inseason_year))
        if _inseason_year is not None and ls.get("season") == _inseason_year:
            # v3.6 (F2): a lone in-progress season shrinks toward the
            # leg's neutral point - there is no completed baseline.
            _single = _shrink_partial(_single, 52.0,
                                      _safe(ls.get("games_played"), 12),
                                      elapsed=_season_elapsed)
        if pos == "RB":
            _single = _clip(_single + _rb_receiving_share_credit(seasons))
        _single = _production_evidence_cap(_single, seasons, pos,
                                           _age, _draft_year,
                                           skip_sagarin)
        return _production_zero_verdict(_clip(_single), [ls], pos,
                                        skip_sagarin, all_seasons=seasons,
                                        draft_year=_draft_year)

    # Score latest season and best individual season; blend to reward peak while
    # still weighting recent output (NFL analysts evaluate both)
    latest_score = _score_production_season(
        ls, pos, skip_sagarin,
        _partial=(_inseason_year is not None
                  and ls.get("season") == _inseason_year))

    # Find best season by primary volume metric per position
    _PEAK_KEY = {"WR": "receiving_yards", "RB": "rush_yards", "QB": "pass_yards", "TE": "receiving_yards"}
    peak_key = _PEAK_KEY[pos]
    best_score = None
    # v3.4: the rate key - the 12-game-equivalent rate (totals /
    # max(GP, 12)). A partial season's calendar-bound totals
    # cannot lose the peak slot to a longer complete season on
    # volume alone, and a short season is never inflated: the
    # denominator floors at 12, so a short season wins only when
    # its raw total already beats the longer seasons' totals.
    def _peak_key_fn(s):
        _g = _safe(s.get("games_played"), 12)
        _g = _g if _g >= 4 else 12
        return _safe(s.get(peak_key)) / max(_g, 12)
    if _inseason_year is not None:
        peak_season = max(seasons, key=_peak_key_fn, default=ls)
    elif PROD_PEAK_CFG.get("by") == "rate":
        # v3.11 round 6 (step C1): the completed-season path
        # takes the v3.4 rate key (see PROD_PEAK_CFG).
        peak_season = max(seasons, key=_peak_key_fn, default=ls)
    elif PROD_PEAK_CFG.get("by") == "rate_price":
        # v3.11 round 6 (step C1): the rate key with a price
        # tie-break - among seasons whose rate is within the
        # tie band of the best rate, the highest full PRICE
        # anchors (see PROD_PEAK_CFG).
        _rates = {id(_s): _peak_key_fn(_s) for _s in seasons}
        _max_rate = max(_rates.values(), default=0.0)
        if _max_rate > 0:
            _band = float(PROD_PEAK_CFG.get("tie_band", 0.025))
            _tied = [_s for _s in seasons
                     if _rates[id(_s)] >= _max_rate * (1.0 - _band)]
            peak_season, best_score = _production_peak_by_price(
                _tied, pos, skip_sagarin, ls)
        else:
            peak_season = max(seasons, key=_peak_key_fn, default=ls)
    elif PROD_PEAK_CFG.get("by") == "price":
        # v3.11 round 6 (step C1, PROD_PEAK_CFG): the anchor is
        # the season with the highest production PRICE, not the
        # highest raw yardage (see the cfg note).
        peak_season, best_score = _production_peak_by_price(
            seasons, pos, skip_sagarin, ls)
    else:
        peak_season = max(seasons, key=lambda s: _safe(s.get(peak_key)), default=ls)
    if best_score is None:
        best_score = _score_production_season(
            peak_season, pos, skip_sagarin,
            _partial=(_inseason_year is not None
                      and peak_season.get("season") == _inseason_year))

    if _inseason_year is not None:
        # v3.6 (round-2 audit F2): whichever blend input is the
        # in-progress season shrinks toward the player's best COMPLETED
        # season score (the leg's neutral 52.0 when none exists) by
        # min(1, GP/12). The 3.6 TE-bonus gate priced sample size for
        # one bonus; this generalizes the same idea to the blend.
        _completed = [s for s in seasons
                      if s.get("season") != _inseason_year]
        _baseline = max(
            (_score_production_season(s, pos, skip_sagarin)
             for s in _completed),
            default=52.0)
        if ls.get("season") == _inseason_year:
            latest_score = _shrink_partial(
                latest_score, _baseline, _safe(ls.get("games_played"), 12),
                elapsed=_season_elapsed)
        if peak_season.get("season") == _inseason_year:
            best_score = _shrink_partial(
                best_score, _baseline,
                _safe(peak_season.get("games_played"), 12),
                elapsed=_season_elapsed)

    # v3.8 (under-rating audit M2): the blend leans harder on the best
    # completed season - 90% on whichever is higher (recent or peak),
    # 10% on the other (was 85/15). The audit's under-rated cohort was
    # priced down by opportunity-split recent seasons (committees,
    # JUCO detours, injury years) even when a full-workload peak was
    # on file; best-season evidence is the leg's own antidote. The
    # role gate in score_prospect (seasons before the first real role
    # are excused from this leg entirely) is the strengthened form of
    # the same principle, added on Kaedon's 2026-11-24 directive.
    #
    # v3.11 round 5: the availability drop (PROD_AVAILABILITY_CFG).
    # A short latest season that is not the best priced season
    # leaves the blend entirely - production prices the best
    # season alone, and durability alone prices the missed
    # games (one absence, one charge).
    if (latest_score < best_score
            and _is_availability_season(ls)
            and not (_inseason_year is not None
                     and ls.get("season") == _inseason_year)):
        prod = best_score
    else:
        prod = max(latest_score, best_score) * 0.90 + min(latest_score, best_score) * 0.10

    # ── Eval-metric blending ─────────────────────────────────────────────────
    # When evaluation pipeline metrics are available, apply a confidence-weighted
    # adjustment.  Each adjustment is intentionally small (±10–15%) so that low-
    # confidence proxies don't override the primary stat-based score.
    pos = position.upper()
    if eval_metrics and pos in ("WR", "TE"):
        yprr = _eval_metric_value(eval_metrics, "yprr", min_confidence=0.30)
        if yprr is not None:
            # 1.8 yprr ≈ strong college producer; 1.2 ≈ below average
            yprr_adj = min(1.15, max(0.85, float(yprr) / 1.8))
            confidence = (eval_metrics.get("yprr") or {}).get("confidence", 0.35)
            blend = 0.80 + 0.20 * confidence  # max 20% weight from this proxy
            prod = _clip(prod * (blend + (1.0 - blend) * yprr_adj))

    elif eval_metrics and pos == "RB":
        yac = _eval_metric_value(eval_metrics, "yac_per_att", min_confidence=0.35)
        if yac is not None:
            # 2.5 yac ≈ average RB; 4.0 ≈ elite contact runner
            yac_adj = min(1.10, max(0.90, float(yac) / 2.5))
            confidence = (eval_metrics.get("yac_per_att") or {}).get("confidence", 0.40)
            blend = 0.85 + 0.15 * confidence
            prod = _clip(prod * (blend + (1.0 - blend) * yac_adj))

    # Position-specific advanced metric bonuses from rookie source data.
    if eval_metrics and pos in ("WR", "TE"):
        ccr = _eval_metric_percent(eval_metrics, "contested_catch_rate", min_confidence=0.45)
        if ccr is None:
            # Fallback to Reception Perception when PFF contested_catch_rate is absent.
            # RP uses a 38-70 scale vs PFF's 0-100; map RP onto the PFF scale.
            ccr_rp = _eval_metric_value(eval_metrics, "contested_catch_rate_rp", min_confidence=0.65)
            if ccr_rp is not None:
                # Linear map: RP 38 -> PFF 30, RP 70 -> PFF 85
                ccr = 30.0 + (float(ccr_rp) - 38.0) * (55.0 / 32.0)
        if ccr is not None:
            prod = _clip(prod + _clip((ccr - 45.0) * 0.10, -4.0, 4.0))

        adot = _eval_metric_value(eval_metrics, "avg_depth_of_target", min_confidence=0.45)
        if adot is not None:
            adot_bonus = _clip((float(adot) - 8.0) * 0.6, -3.0, 4.0)
            # High ADOT in an air-raid system with poor contested-catch ability is
            # scheme-generated volume, not a separation skill signal.
            if float(adot) > 14.0 and ccr is not None and ccr < 50.0:
                adot_bonus = min(adot_bonus, 0.0)
            prod = _clip(prod + adot_bonus)

    elif eval_metrics and pos == "RB":
        # elusive_rating dropped: was PFF-only, no free replacement.

        breakaway = _eval_metric_percent(eval_metrics, "explosive_run_rate", min_confidence=0.40)
        if breakaway is not None:
            # Scale-based and symmetric: 0 at 20% (average), +2 at 40% elite, -1.5 at 8%.
            breakaway_delta = _clip((breakaway - 20.0) / 20.0, -1.0, 1.0)
            prod = _clip(prod + (breakaway_delta * 2.0))

    elif eval_metrics and pos == "QB":
        # pff_passing_grade is now sourced from CFBD PPA (free replacement for PFF).
        # See rookie_sources.py for the PPA-to-grade mapping.
        pff_pass = _eval_metric_value(eval_metrics, "pff_passing_grade", min_confidence=0.45)
        if pff_pass is not None:
            prod = _clip(prod + _clip((float(pff_pass) - 70.0) * 0.20, -4.0, 7.0))

        # big_time_throw_rate dropped: was PFF-only, no free replacement.

    # v3.6 (round-2 audit F7): the scheme-inflation discount is deleted
    # here and in efficiency (team context consolidation: competition
    # leg + environment leg are the two surviving channels).

    # v3.8 (M3): the bounded RB receiving-share credit lands on the
    # finished leg, exactly as gated (career share, cap 10).
    if pos == "RB":
        prod = _clip(prod + _rb_receiving_share_credit(seasons))

    # v3.11 revision: the evidence-base cap prices the finished
    # leg (see PROD_EVIDENCE_CFG) - one featured season cannot
    # carry the full spike.
    prod = _production_evidence_cap(prod, seasons, pos,
                                    _age, _draft_year, skip_sagarin)

    return _production_zero_verdict(prod, [ls, peak_season], pos,
                                    skip_sagarin, all_seasons=seasons,
                                    draft_year=_draft_year)


def _qb_efficiency_of(season: Dict) -> Optional[float]:
    """One season's QB efficiency from its rate evidence (v3.8 M1).

    Each component is the stored rate when present, else derived from
    the season's counting stats: YPA = pass yards / attempts,
    completion % = completions / attempts, TD:INT = pass TDs /
    interceptions (a season with TDs and zero interceptions reads at
    the scale's top anchor - the ratio is unbounded, not missing).
    Present components combine on the leg's 45/30/25 weights,
    renormalized over what is present; a season with no rate evidence
    at all returns None. The friendly defaults this replaces (YPA
    7.0 / comp 62% / TD:INT 2.0) scored an unmeasured season as a
    league-average one.
    """
    parts = []
    ypa = _rate(season, "yds_per_attempt", "pass_yards", "pass_attempts")
    if ypa is not None:
        parts.append((_scale(ypa, 6.5, 9.5), 0.45))
    cpct = season.get("completion_pct")
    if cpct is None and season.get("completions") is not None:
        att = _safe(season.get("pass_attempts"))
        cpct = (_safe(season.get("completions")) / att * 100.0) if att > 0 else None
    if cpct is not None:
        parts.append((_scale(_safe(cpct), 58.0, 72.0), 0.30))
    td_int = season.get("td_int_ratio")
    if (td_int is None and season.get("pass_tds") is not None
            and season.get("interceptions") is not None):
        ints = _safe(season.get("interceptions"))
        tds = _safe(season.get("pass_tds"))
        if ints > 0:
            td_int = tds / ints
        elif tds > 0:
            td_int = 6.0
    if td_int is not None:
        parts.append((_scale(_safe(td_int), 1.5, 6.0), 0.25))
    if not parts:
        return None
    wsum = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / wsum


def _te_efficiency_of(season: Dict) -> Optional[float]:
    """One season's TE efficiency from its rate evidence (v3.11
    round 3). Yards per reception on the leg's 7.5-14.0 scale,
    blended 0.625/0.375 with the catch-rate score when the season
    carries valid targets (targets below receptions is a pipeline
    failure, not data - v3.2 - and reads as missing); without
    targets the leg is YPR alone (v3.2/v3.6 F5). A season with no
    YPR evidence at all returns None."""
    ypr = _rate(season, "yds_per_reception", "receiving_yards",
                "receptions")
    if ypr is None:
        return None
    targets = _safe(season.get("targets"))
    recs = _safe(season.get("receptions"))
    if targets < recs:
        targets = 0.0
    if targets > 0:
        catch_rate_score = _scale(recs / targets, 0.60, 0.85)
        return (_scale(ypr, 7.5, 14.0) * 0.625
                + catch_rate_score * 0.375)
    return _scale(ypr, 7.5, 14.0)


# v3.11 round 4 (Phase B): the RB efficiency leg's rush-quality
# composite. Round 2's diagnosis (§8.5 of the ceiling note) was
# that the blend is no longer the RB binding constraint - the
# YPC-only leg content is: it cannot see the home-run trait
# (>=20-yd rate) or the stuff rate, the one signal orthogonal to
# YPC (r = -.27 pooled). When a season carries play-by-play rush
# rates that pass the coverage gate, the pair's YPC part is
# replaced by rush_q: the YPC scale plus the >=20 rate, the >=12
# rate, and the inverted team-contexted stuff rate (the player's
# stuff rate minus his teammates' att-weighted baseline, so a
# bad line is not priced as a bad back), at the weights below,
# renormalized over the terms the season can price. Anchors are
# the recorded population's pooled quantiles (SportsDataverse
# ESPN build, >=100-carry rusher-seasons 2015-2023, n=1,337),
# calibrated once and fixed: exp12/exp20 at the probe note's
# p10/p90 (0.060/0.165, 0.015/0.080); stuff_ctx at the rebuilt
# population's p10/p90 (-0.0735/+0.0422 - the contexted
# distribution is centered below zero because the median
# workhorse is stuffed less than the teammate mix he is
# compared against).
RB_EFF_CFG = {
    "enabled": True,
    # Swept on the GP frame (§10 of the ceiling note): the
    # stuff term carries the composite's new information (it is
    # the one signal orthogonal to YPC), and the stuff-heaviest
    # variant gated best on both correlations with the joint-
    # best Poor rate. The >=20 weight stays at the probe
    # sketch's 0.20 - heavier variants bought Barkley's leg
    # points at the price of the gate and of Bijan, which is
    # fitting weights to names, and the round does not do that.
    "w_ypc": 0.40,
    "w_exp20": 0.20,
    "w_exp12": 0.10,
    "w_stuff": 0.30,
    "exp12_lo": 0.060, "exp12_hi": 0.165,
    "exp20_lo": 0.015, "exp20_hi": 0.080,
    "stuff_ctx_lo": -0.0735, "stuff_ctx_hi": 0.0422,
    "min_team_games": 11,
    "min_att_ratio": 0.90,
    "min_pbp_att": 100,
}


# v3.11 round 7: the pair form is the pre-rebuild leg - YPC
# and YPR only, no stuff/explosive content - and on the 502
# recorded RB seasons that carry BOTH forms it overstates the
# rush-quality composite by a pooled margin that grows with
# the print (median delta pair minus composite: +0.2 below
# 60, +4.5 at 60-75, +8.9 at 75-85, +9.6 at 85+). A season
# whose play-by-play rates pass the coverage gate prices on
# the composite; a season with no rates at all prices on the
# pair - and the PBP build ends at 2023, so exactly the
# seasons that price the current top of the board can be
# pair-only (Javonte's 2020, Harvey's 2024). At peak share
# 0.74 the form gap alone was worth about two grade points
# up there: pair 100.0 / 96.0 prints sat peak-resident while
# composite-priced peers printed 70-77 on the same trait.
# The pair form's excess over the floor is therefore
# discounted by the pooled share wherever the pair prices -
# the blend's per-season read and the all-blocked fallback
# alike - mapping the four pooled bands to within ~2 points
# of their composite medians (100 -> 86.0, 91 -> 80.2,
# 80 -> 73.0, 67 -> 64.6). Composite prints are untouched,
# and the leg-level volume shrink still applies on top.
EFF_FORM_PARITY_CFG = {
    "enabled": True,
    "floor": 60.0,
    "discount": 0.35,
}


# v3.13 (Phase 6c): RB efficiency is priced WITHIN a workload
# band. On the absolute scale the leg's top was a short-sample
# artifact: every raw >= 95 season in the recorded pool sits on
# <= 182 opportunities and was volume-shrunk, while the best
# raw season on >= 250 opportunities prices 87.56 - so the
# leg's p99 print was 85.95 and a workhorse's elite efficiency
# could not price elite. A plain stretch of the scale died
# twice on incidence (it paid the capped short-sample holders
# more than the ceiling). Banding prices each season against
# seasons of comparable volume instead: the reference set is
# every frozen-pool season within +/- 75 opportunities, the
# window's [p85, p99] maps linearly onto [85, 100], and a
# workload ramp caps top pay on short samples (90 at <= 175
# opportunities, rising linearly to 100 at 275) - the ramp and
# the band-relative pricing together replace the retired
# volume shrink's noise protection (EFF_VOLUME_SHRINK_CFG
# ships disabled). The pool is the FROZEN recorded population
# below (2016-2024 RB seasons priced by _rb_efficiency_of,
# (opportunities, raw value) pairs sorted by opportunities,
# n = 767): pinned constants, never recomputed per class.
# Measured on the amended design base: RB p99 chimera 97.17
# -> 97.47, gate .4677/.4685 -> .4684/.4691, the capped
# holders pay (Javonte -0.08, Vaughn -1.26) and the leg's top
# becomes a workhorse honor roll. See prospect-runs/
# rb-efficiency-banding-2026-11-25.md.
RB_EFF_BAND_TABLE = (
    (1.0, 0.0), (1.0, 0.0), (1.0, 42.857142857142854), (2.0, 76.71428571428571),
    (3.0, 0.0), (3.0, 0.0), (3.0, 76.71428571428571), (3.0, 86.0),
    (3.0, 86.0), (3.0, 86.0), (4.0, 0.0), (4.0, 74.39285714285714),
    (4.0, 86.0), (5.0, 25.71428571428571), (6.0, 86.0), (7.0, 70.12142857142857),
    (8.0, 57.142857142857146), (8.0, 63.89999999999999), (8.0, 67.42857142857143), (8.0, 86.0),
    (8.0, 86.0), (9.0, 0.0), (9.0, 0.0), (9.0, 61.20714285714286),
    (10.0, 76.72357142857143), (11.0, 86.0), (12.0, 35.714285714285715), (12.0, 67.42857142857143),
    (13.0, 8.571428571428571), (13.0, 30.0), (13.0, 80.24285714285715), (14.0, 72.07142857142857),
    (14.0, 78.66428571428573), (15.0, 80.42857142857143), (15.0, 86.0), (16.0, 50.571428571428555),
    (17.0, 53.714285714285715), (20.0, 86.0), (23.0, 68.63571428571429), (26.0, 17.82857142857143),
    (26.0, 62.07538880595061), (28.0, 67.8), (29.0, 82.21142857142857), (49.0, 73.0),
    (54.0, 33.31428571428571), (57.0, 86.0), (90.0, 74.18857142857142), (95.0, 22.64285714285713),
    (101.0, 14.942271062271065), (104.0, 59.765402847145715), (107.0, 50.99442281595469), (108.0, 25.792464384008873),
    (108.0, 52.89501831501831), (109.0, 49.452607533178785), (111.0, 29.719897298962714), (111.0, 33.116311663414464),
    (111.0, 55.18927483811168), (111.0, 61.382692575160966), (111.0, 74.75714285714285), (112.0, 88.67367906643416),
    (113.0, 27.957292857544505), (113.0, 36.42789462648248), (113.0, 36.94797449464115), (113.0, 69.64146098180817),
    (114.0, 17.924009387672164), (114.0, 36.98291848551518), (114.0, 51.8729735625334), (114.0, 78.21700412389241),
    (115.0, 31.139873167087927), (115.0, 34.51068344323314), (116.0, 19.81345744842996), (116.0, 30.662370978960407),
    (116.0, 39.008891327209554), (116.0, 47.80107284931751), (117.0, 31.473814473225403), (117.0, 46.117948717948714),
    (117.0, 51.399763301730964), (118.0, 20.90683400030165), (119.0, 53.47855417393031), (119.0, 78.04395604395603),
    (120.0, 26.065642567858763), (120.0, 41.667203907203906), (120.0, 46.93316525361881), (120.0, 80.17384750805556),
    (120.0, 92.90647544545848), (121.0, 15.512365842196015), (121.0, 47.933239415609286), (121.0, 88.93702927032126),
    (122.0, 22.843470679374995), (122.0, 53.51035235090547), (122.0, 55.51415274682982), (122.0, 76.67206992269494),
    (122.0, 76.84320409150598), (122.0, 81.83821145130506), (123.0, 48.91428571428571), (123.0, 52.554273273895916),
    (124.0, 28.701679345370994), (124.0, 53.87383595335187), (124.0, 55.33660470635795), (124.0, 61.74462391765188),
    (124.0, 70.52258024890696), (125.0, 11.1743501734637), (125.0, 16.91572248458269), (125.0, 19.162564297743263),
    (125.0, 32.679365079365084), (125.0, 55.14883380311953), (125.0, 59.6414384991955), (125.0, 93.73734015912062),
    (126.0, 48.78246642246642), (126.0, 77.4376130649668), (126.0, 92.23390790806522), (127.0, 10.842857142857138),
    (127.0, 36.5416567247829), (127.0, 38.43506099078915), (127.0, 39.99129505716107), (127.0, 52.55831963356596),
    (127.0, 55.28976693121639), (127.0, 57.47402271885419), (127.0, 78.40071005917159), (128.0, 33.83691340106434),
    (128.0, 45.175369249975994), (128.0, 56.96468553031184), (129.0, 56.11426938740524), (129.0, 60.437256726830796),
    (129.0, 67.01579587208515), (129.0, 90.44737138777197), (131.0, 22.62309082967293), (131.0, 28.027653256413295),
    (131.0, 35.26683503630872), (131.0, 63.971545634152235), (131.0, 69.73071496272277), (132.0, 53.39999999999999),
    (132.0, 69.07214285714285), (132.0, 70.99564624988835), (133.0, 34.76182859895901), (133.0, 40.467950799849426),
    (133.0, 52.50497114860747), (133.0, 53.16249131834266), (134.0, 27.015852911133805), (134.0, 30.45227201573272),
    (134.0, 40.10751047882088), (134.0, 51.957142857142856), (134.0, 52.643856082647176), (134.0, 64.4042654991578),
    (135.0, 34.01774200910845), (135.0, 52.32721581941094), (137.0, 23.590613906529885), (137.0, 54.65791896115161),
    (137.0, 56.216025641025645), (137.0, 57.411765705791524), (138.0, 44.57511429139336), (138.0, 77.70572482832637),
    (138.0, 77.83149227584009), (139.0, 15.913896274688904), (139.0, 42.5599881174541), (139.0, 59.44492510005042),
    (139.0, 65.57959713073002), (140.0, 36.85109754443087), (140.0, 44.61635789669013), (140.0, 44.992616229799275),
    (140.0, 45.808618048618044), (140.0, 57.181611714608934), (140.0, 66.78548483795834), (140.0, 72.794292224902),
    (140.0, 75.05764291122007), (140.0, 75.1739380127351), (141.0, 32.029683511144185), (141.0, 41.60490280498122),
    (141.0, 64.3023617067745), (141.0, 70.40701789283915), (141.0, 73.03800022868165), (142.0, 29.27179559539678),
    (142.0, 87.73248469888532), (143.0, 20.895312387143374), (143.0, 29.88973184576238), (143.0, 45.728571428571435),
    (143.0, 63.684069312228175), (144.0, 21.31871528735538), (144.0, 25.955266892812144), (144.0, 36.972667304605515),
    (144.0, 42.67722575495312), (144.0, 70.96613863704835), (144.0, 78.3170750237417), (144.0, 89.04191677175284),
    (144.0, 92.00108985376758), (145.0, 23.987323474952504), (145.0, 25.930256410256412), (145.0, 37.88735486439743),
    (145.0, 64.76936454849499), (146.0, 23.92464969487441), (146.0, 26.761314064102404), (146.0, 33.70846581641621),
    (146.0, 37.20147894823661), (146.0, 42.29175247281668), (146.0, 47.81704517704517), (146.0, 48.799240954087566),
    (146.0, 58.10595147992392), (146.0, 78.82536279105837), (147.0, 41.18987072243321), (147.0, 49.36948628246813),
    (147.0, 67.45344428150457), (147.0, 68.67649834859915), (147.0, 79.93738101187661), (148.0, 26.83517773379272),
    (148.0, 37.03699749362558), (148.0, 40.089029215389345), (148.0, 47.5953632838394), (148.0, 75.45644097837646),
    (149.0, 23.5919069166128), (149.0, 31.919463440350604), (149.0, 46.782240215924425), (149.0, 54.88070393374741),
    (149.0, 62.44323256825848), (149.0, 63.44898134756052), (149.0, 63.66005794323146), (149.0, 67.05755351681957),
    (149.0, 80.6700198301381), (150.0, 34.17301587301588), (150.0, 41.15476611768804), (150.0, 59.92235758373319),
    (150.0, 71.15230769230769), (150.0, 78.36067108362363), (150.0, 79.0), (151.0, 32.69567438178205),
    (151.0, 36.08565940119906), (151.0, 39.40843788562836), (151.0, 40.83430470763804), (151.0, 41.59803982076717),
    (151.0, 43.79070057725093), (151.0, 80.73236987913424), (151.0, 95.37815245294702), (152.0, 45.0428897028897),
    (152.0, 46.63678382686017), (152.0, 54.86055579744521), (152.0, 77.09484737484738), (153.0, 12.929409580158296),
    (153.0, 13.654421393742995), (154.0, 27.72544310445603), (154.0, 37.9164251163778), (154.0, 52.916884389019216),
    (154.0, 69.13143145391787), (154.0, 78.63487580251625), (154.0, 79.07365488018334), (154.0, 91.83941508756504),
    (155.0, 44.30862741913341), (155.0, 47.4972453321058), (156.0, 39.0502865775016), (156.0, 45.761640318072764),
    (156.0, 56.21284567849453), (157.0, 31.553488273457276), (157.0, 34.20563241526928), (157.0, 46.904829481576975),
    (157.0, 59.90609253942892), (158.0, 14.87925081362135), (158.0, 17.632195000278994), (158.0, 30.404246897381228),
    (158.0, 34.07613635282294), (158.0, 35.64912821960112), (158.0, 54.603701546305636), (158.0, 58.09837895595436),
    (158.0, 62.98187279639155), (158.0, 67.86426264593148), (158.0, 70.31446136146792), (158.0, 70.73399382594789),
    (158.0, 72.1138349749803), (158.0, 75.80013449066061), (159.0, 14.979855448081235), (159.0, 58.67575692544778),
    (159.0, 61.47684167684167), (159.0, 68.17460262755884), (159.0, 71.8372140451567), (160.0, 37.08141365467208),
    (160.0, 52.633853348759715), (161.0, 48.01428571428571), (161.0, 55.8615360468388), (161.0, 64.30220070972786),
    (162.0, 49.59854204236226), (162.0, 65.18123381296341), (162.0, 69.37780951040946), (162.0, 78.51908472410513),
    (162.0, 80.56527784437392), (163.0, 27.159294100610847), (163.0, 31.760135058444), (163.0, 38.71448083066338),
    (163.0, 47.914285714285704), (164.0, 28.526733166091745), (164.0, 55.995751237936254), (164.0, 65.11596684017736),
    (164.0, 67.4883041180423), (164.0, 98.29184673004897), (165.0, 26.733944225082546), (165.0, 29.13020295020295),
    (165.0, 41.62049960794991), (165.0, 41.74514375433357), (165.0, 55.92220306901376), (165.0, 62.08430319532625),
    (165.0, 66.58094204617439), (165.0, 67.8570128567029), (165.0, 79.26441874202848), (165.0, 80.81520350056604),
    (166.0, 42.67485769437047), (166.0, 54.81854136295832), (166.0, 63.51749077628715), (167.0, 30.201116686548644),
    (167.0, 32.68475524475524), (167.0, 51.76841978885278), (167.0, 77.27697510553833), (168.0, 43.138083394694),
    (168.0, 48.890144875429655), (168.0, 49.2028887275214), (168.0, 55.240425013686746), (168.0, 61.62984521985251),
    (169.0, 36.95315982263351), (169.0, 38.545206115719246), (169.0, 38.70287437126579), (169.0, 42.85903520069027),
    (169.0, 53.33280199315556), (169.0, 60.703555445802635), (169.0, 61.01465927771281), (170.0, 13.832632018397137),
    (170.0, 34.62588522588523), (170.0, 36.046448892071034), (170.0, 40.04031473333691), (170.0, 51.05860926442857),
    (170.0, 51.6748542002815), (170.0, 98.25599869549798), (171.0, 38.81658189314543), (171.0, 45.02951299152115),
    (171.0, 48.607245500332084), (172.0, 33.53780632319548), (172.0, 58.700271615209545), (172.0, 82.88587867417135),
    (173.0, 41.67929516262526), (173.0, 47.37928464313944), (174.0, 22.491599511599514), (174.0, 55.82733422090712),
    (175.0, 48.47565724991872), (175.0, 57.250098515716495), (175.0, 65.21727047146402), (176.0, 30.223226943036796),
    (176.0, 38.853713382039615), (176.0, 48.399629320860235), (176.0, 49.66188986520757), (176.0, 57.103756052461364),
    (176.0, 60.8791994996873), (176.0, 65.7644635504826), (176.0, 69.29985871920796), (176.0, 69.42428118697782),
    (176.0, 70.66834014505042), (176.0, 90.94451304498716), (177.0, 14.78005311737443), (177.0, 31.769640385453897),
    (177.0, 32.054062554548324), (177.0, 35.212056340783526), (177.0, 36.018363858363855), (177.0, 48.368521852063694),
    (177.0, 53.400010950379766), (177.0, 60.993072177007576), (177.0, 75.18779084094109), (178.0, 29.49232517474612),
    (178.0, 38.56306775023279), (178.0, 66.97714125290767), (178.0, 80.00904635777417), (179.0, 8.631377709645703),
    (179.0, 42.947604716723575), (179.0, 48.95602852785672), (179.0, 49.557433485302184), (179.0, 65.83780469753137),
    (179.0, 95.01191447140589), (180.0, 30.159729737308027), (180.0, 34.87011015413813), (180.0, 46.73436136012356),
    (180.0, 50.412949696613815), (181.0, 31.417092756324728), (181.0, 45.73111112967833), (181.0, 68.8386530738338),
    (181.0, 74.96765452269628), (181.0, 75.37292714008358), (182.0, 41.84037735143346), (182.0, 50.08747343068632),
    (182.0, 50.69323206371382), (182.0, 73.27375035221189), (182.0, 78.01220338591078), (182.0, 95.49406986714651),
    (183.0, 14.592109560530844), (183.0, 24.864078221754998), (183.0, 42.57815911499841), (183.0, 62.84266490413923),
    (183.0, 64.56875326005118), (184.0, 15.74789729080612), (184.0, 16.614014714362842), (184.0, 39.22558324861258),
    (184.0, 51.8021333877059), (184.0, 56.01393958423946), (184.0, 67.10404721677902), (185.0, 0.0),
    (185.0, 17.866366479049407), (185.0, 22.939654278305962), (186.0, 90.60638642204509), (187.0, 25.577424114347192),
    (187.0, 51.34777391689483), (187.0, 53.270362705574456), (188.0, 35.06239362314834), (188.0, 72.91457715783896),
    (189.0, 28.041001970141323), (189.0, 34.5401439939371), (189.0, 35.53020958993193), (190.0, 15.420985306828001),
    (190.0, 52.21810556746844), (191.0, 12.657142857142855), (191.0, 34.29894993894993), (191.0, 41.66357324471413),
    (191.0, 53.9938441074754), (191.0, 65.8958885287073), (192.0, 42.08688644374706), (192.0, 52.59867021867022),
    (192.0, 70.15360598139173), (192.0, 84.73645098035246), (193.0, 28.578354216661918), (193.0, 32.46631429371499),
    (193.0, 53.275859953999486), (193.0, 60.70737912687675), (194.0, 31.777763413057524), (194.0, 53.45118274259267),
    (194.0, 54.879074133251166), (194.0, 61.116624101384325), (194.0, 73.78239314539377), (194.0, 78.26569677513497),
    (195.0, 46.29920334336842), (195.0, 47.60449894054159), (195.0, 59.600422579627875), (195.0, 69.22647050581477),
    (195.0, 70.21608807729405), (195.0, 75.58852249265729), (196.0, 38.41427757427757), (196.0, 39.70249439880507),
    (196.0, 44.52020335635113), (196.0, 56.147869229741595), (196.0, 80.06515519182881), (197.0, 19.47109318836776),
    (197.0, 67.5084636843902), (198.0, 4.714285714285713), (198.0, 38.891983022887786), (198.0, 40.171114974351305),
    (198.0, 45.552202980265804), (198.0, 49.763059245443216), (199.0, 22.73639908114484), (199.0, 29.233602737239973),
    (199.0, 32.284510026453184), (199.0, 34.79955666848294), (199.0, 57.10729896703572), (200.0, 3.385047233468286),
    (200.0, 36.066988670244484), (200.0, 42.01162066004171), (201.0, 40.28020623177336), (201.0, 59.21012769529624),
    (202.0, 50.908000068072994), (202.0, 51.89129676841998), (202.0, 58.96198365736828), (202.0, 63.47731566031318),
    (202.0, 80.66549116519704), (203.0, 36.13563723632689), (203.0, 49.581130464760534), (203.0, 56.950115053050126),
    (204.0, 34.563170507403335), (204.0, 34.65463495096923), (204.0, 41.54662926578322), (204.0, 43.46447767300046),
    (204.0, 45.70450256986928), (204.0, 46.24336974546376), (204.0, 55.67468339808883), (204.0, 72.00169733880577),
    (205.0, 36.27289934832964), (205.0, 40.36625392220102), (205.0, 41.12673666076112), (205.0, 52.78072105619984),
    (205.0, 55.25171051394117), (205.0, 66.32005123506725), (205.0, 70.94548177644718), (206.0, 55.774893277154966),
    (206.0, 62.22221735235386), (206.0, 71.86137100994245), (207.0, 23.549120775835217), (207.0, 35.434094867428776),
    (207.0, 36.622099384452326), (207.0, 61.1010528204041), (207.0, 68.53573287404136), (207.0, 73.90859236002093),
    (208.0, 32.249089190489286), (208.0, 47.45765536401226), (208.0, 53.49939567474824), (208.0, 63.23243072616789),
    (208.0, 86.0), (209.0, 34.97311905820576), (209.0, 39.030399655121506), (209.0, 47.2495906533686),
    (209.0, 57.08169663365894), (210.0, 41.4766449527923), (210.0, 49.19755086361827), (210.0, 54.30214944304887),
    (211.0, 61.61675690166231), (211.0, 61.93916108888341), (211.0, 82.06455540759048), (211.0, 85.51254776384626),
    (211.0, 87.56062297693494), (212.0, 33.51599839244254), (212.0, 63.06086854094073), (212.0, 69.4798939220048),
    (213.0, 66.19785526809149), (214.0, 66.35557782858413), (215.0, 48.67520868534824), (215.0, 60.410871351718754),
    (215.0, 61.74232763180874), (215.0, 68.93071881858398), (216.0, 34.46490411938385), (216.0, 64.11436650801342),
    (216.0, 69.99338096190556), (216.0, 76.42857142857143), (217.0, 43.36305695648824), (217.0, 64.76531917520683),
    (217.0, 68.93843751876537), (218.0, 50.249072308723605), (218.0, 68.2855555606087), (218.0, 73.80090873665341),
    (219.0, 41.2134611487803), (219.0, 46.869204681802984), (219.0, 61.432252208980415), (220.0, 42.30415110716633),
    (220.0, 73.81045577484268), (220.0, 73.86367346938775), (221.0, 32.81029296106998), (221.0, 60.43716742491159),
    (222.0, 11.304255045313301), (223.0, 39.86526938715298), (223.0, 49.680554621704914), (223.0, 64.78659403920408),
    (224.0, 16.7213579013579), (224.0, 76.81868092691622), (225.0, 52.9121414569297), (226.0, 24.97723949809056),
    (226.0, 27.6129894895724), (226.0, 30.943441486395905), (226.0, 47.59349153393605), (226.0, 62.0377941613918),
    (226.0, 77.777323808771), (227.0, 42.32423405654175), (227.0, 61.87598389339072), (228.0, 59.566842892248516),
    (228.0, 66.92177650509835), (228.0, 79.19917352176127), (229.0, 47.26151368227124), (229.0, 54.9694312465886),
    (230.0, 23.051068344436217), (230.0, 47.92123621524233), (230.0, 52.83560044905917), (230.0, 60.67966386940058),
    (231.0, 56.925877912273855), (232.0, 31.67931446830324), (232.0, 37.897279469714526), (232.0, 55.577451477604946),
    (232.0, 69.19008470265072), (232.0, 69.44833178943585), (233.0, 85.55388235204495), (235.0, 24.06528844389419),
    (235.0, 42.64714846063331), (235.0, 48.97400179519735), (235.0, 53.52354859846834), (235.0, 56.480424504620856),
    (235.0, 66.36754620491604), (236.0, 47.47078465788104), (236.0, 50.8681000159261), (236.0, 62.49755096561379),
    (236.0, 63.0061615339868), (236.0, 69.17425958129839), (237.0, 28.52384922886795), (237.0, 72.24280142917999),
    (238.0, 23.80116247922416), (238.0, 43.94742620576407), (238.0, 77.91132939454758), (239.0, 20.155441505728053),
    (239.0, 36.60653079114618), (239.0, 77.44785714285713), (240.0, 38.97860314184936), (240.0, 39.859636884854275),
    (240.0, 48.80432453942298), (240.0, 52.22638668517443), (240.0, 53.11825350968823), (240.0, 55.0556186718439),
    (240.0, 60.76728570863342), (241.0, 60.8959855781751), (242.0, 48.06506106837389), (242.0, 52.45384562590753),
    (243.0, 40.699264807983404), (243.0, 48.06497173918177), (243.0, 55.935210120383175), (243.0, 58.85650748005889),
    (243.0, 63.95395563238259), (244.0, 38.60885033095559), (244.0, 42.58959146128038), (244.0, 83.4247709951667),
    (245.0, 53.60492261774313), (245.0, 73.25150661201876), (246.0, 18.669959214956357), (246.0, 33.26552754435107),
    (246.0, 51.60262219940517), (248.0, 36.389865578933275), (248.0, 49.756378059222044), (248.0, 52.559884492829035),
    (249.0, 48.273296249510594), (250.0, 48.43329065744386), (250.0, 76.96429562946416), (251.0, 67.10344658163555),
    (252.0, 32.90539131831654), (252.0, 51.75489100055438), (252.0, 63.84789799963501), (252.0, 76.8257142857143),
    (252.0, 81.76567493757113), (252.0, 82.81329822755389), (253.0, 37.65185572075287), (253.0, 71.15714285714286),
    (254.0, 14.630874044806633), (254.0, 66.1731338540977), (255.0, 35.95161363875771), (255.0, 39.394266274312685),
    (255.0, 51.45670450446467), (256.0, 29.686663686307245), (256.0, 61.27376792738431), (257.0, 43.851228858962294),
    (257.0, 70.22525404420142), (258.0, 55.59263457019645), (258.0, 55.71335921402765), (258.0, 85.20483961554358),
    (259.0, 25.491075756742596), (259.0, 48.33375465871836), (259.0, 75.353398375268), (260.0, 25.770697975399134),
    (260.0, 57.52772801465268), (261.0, 43.88133520181921), (261.0, 48.79944127236996), (262.0, 58.57871022848602),
    (262.0, 64.21554810016296), (263.0, 40.55129573784306), (263.0, 71.59255919011709), (264.0, 44.5180315817368),
    (264.0, 57.945663095621256), (264.0, 60.48302860260307), (264.0, 69.16722622023077), (267.0, 41.82486512770041),
    (267.0, 52.15922984299318), (268.0, 40.32812420211481), (268.0, 47.11844895607335), (268.0, 54.148477803838915),
    (268.0, 72.02866445235773), (269.0, 60.081617170673724), (269.0, 60.95275737396889), (269.0, 65.73192282214238),
    (270.0, 25.93064896366539), (270.0, 52.30240737108521), (270.0, 55.68742335866312), (270.0, 83.18682745421509),
    (271.0, 27.583390589723308), (271.0, 42.524866444417), (271.0, 66.8858653899816), (272.0, 11.519049571289802),
    (272.0, 45.13083461209967), (272.0, 58.72528367162535), (273.0, 59.62322407299496), (273.0, 74.57125978745499),
    (274.0, 42.56853333785257), (274.0, 75.16721596364394), (276.0, 34.12675576040087), (276.0, 48.72412698412698),
    (276.0, 49.98956871023808), (276.0, 57.459158649530025), (276.0, 64.57445849142287), (277.0, 45.908992175140035),
    (277.0, 62.93235365941248), (277.0, 73.07504319760997), (278.0, 52.15413625526508), (278.0, 66.0641473713149),
    (278.0, 87.563378846727), (280.0, 15.303808168038172), (280.0, 40.93490934803514), (280.0, 45.31298396158954),
    (281.0, 33.94985701707983), (281.0, 53.314580590820285), (282.0, 29.34055660015246), (282.0, 50.74319633331137),
    (282.0, 55.67794266569006), (282.0, 77.04857142857144), (283.0, 73.83582336723074), (284.0, 50.50845269672856),
    (284.0, 54.25852022971149), (284.0, 55.73090941177345), (287.0, 29.783859890109888), (288.0, 44.361756975016476),
    (288.0, 63.06652121203493), (289.0, 50.96327601325607), (289.0, 60.06413664472633), (290.0, 69.55451363451363),
    (291.0, 48.273765521239866), (292.0, 46.393815527939786), (293.0, 17.857328162034037), (294.0, 42.27342408475341),
    (294.0, 60.98028636365714), (295.0, 28.47673431392397), (295.0, 85.83223775984952), (296.0, 46.54013513701791),
    (296.0, 59.87351644177832), (297.0, 56.34141475432005), (298.0, 46.53122426815712), (298.0, 65.45228951097565),
    (298.0, 81.35804149614725), (298.0, 84.07902216719117), (299.0, 41.667549883496775), (300.0, 59.916620725985275),
    (300.0, 70.25886941274791), (301.0, 27.7602473593512), (301.0, 59.48444461825244), (301.0, 71.70500591168413),
    (302.0, 43.13393380617927), (303.0, 59.736892659956226), (304.0, 55.295345198092406), (304.0, 63.40040161813738),
    (305.0, 44.72411076103384), (305.0, 50.97361786777516), (306.0, 39.96813986439708), (308.0, 65.77913500633677),
    (308.0, 77.24377310665571), (309.0, 39.2624619595388), (309.0, 86.58952444583295), (313.0, 39.392838207118714),
    (314.0, 62.38081102107462), (315.0, 60.472686487031226), (316.0, 62.78978541642901), (319.0, 56.042408238004754),
    (319.0, 80.73264718667006), (320.0, 31.438464770084295), (320.0, 38.226913553965076), (320.0, 76.00245971177486),
    (321.0, 65.30830770305273), (321.0, 67.69230219231497), (322.0, 42.49944384654697), (324.0, 71.33077133825877),
    (324.0, 73.07028824655157), (325.0, 71.98768178157349), (326.0, 44.960534612704464), (327.0, 42.127553786612744),
    (329.0, 30.23685346116498), (330.0, 79.49328569818067), (331.0, 27.76894410518146), (331.0, 66.18090753252523),
    (333.0, 36.84685369518013), (333.0, 45.130674534468746), (334.0, 42.922848637005956), (335.0, 40.58615619170122),
    (335.0, 42.129451501058014), (339.0, 41.07059038428561), (346.0, 65.66300580751792), (351.0, 66.33077688358638),
    (351.0, 66.94063481149973), (355.0, 50.15321448738699), (357.0, 53.661267720877454), (381.0, 52.62501647103264),
    (382.0, 76.66000331957599), (397.0, 57.22211174347771), (406.0, 61.60817081652524),
)
_RB_EFF_BAND_OPPS = tuple(n for n, _ in RB_EFF_BAND_TABLE)


def _rb_opportunities(season: Dict) -> float:
    """A season's workload for banding: rush attempts +
    receptions (the volume shrink's opportunity definition)."""
    return (_safe(season.get("rush_attempts"))
            + _safe(season.get("receptions")))


def _rb_eff_band(value: float, n_opp: float) -> float:
    """Re-price one raw RB efficiency season value within its
    workload window (Phase 6c, variant V2 exactly). A season
    with no recorded opportunities, or a window too thin to
    price against (< 10 frozen seasons - never on the recorded
    frame, whose thinnest window is 17), stands as priced."""
    import bisect as _bisect
    if value is None or n_opp <= 0:
        return value
    lo = _bisect.bisect_left(_RB_EFF_BAND_OPPS, n_opp - 75.0)
    hi = _bisect.bisect_right(_RB_EFF_BAND_OPPS, n_opp + 75.0)
    window = sorted(v for _, v in RB_EFF_BAND_TABLE[lo:hi])
    if len(window) < 10:
        return value
    q85 = _pool_percentile(window, 85.0)
    q99 = _pool_percentile(window, 99.0)
    if value <= q85:
        banded = value
    else:
        banded = min(
            85.0 + (value - q85) / max(q99 - q85, 1e-9) * 15.0,
            100.0)
    if n_opp <= 175.0:
        cap = 90.0
    elif n_opp < 275.0:
        cap = 90.0 + (n_opp - 175.0) / 100.0 * 10.0
    else:
        cap = 100.0
    return min(banded, cap)


def _rb_pair_value(season: Dict) -> Optional[float]:
    """One season's box-score RB efficiency - the pre-rebuild
    leg's per-season read: YPC scale (3.5-7.0) and YPR scale
    (5.0-12.0) renormalized 0.70/0.30 over whichever rate is
    measurable; None when neither is (v3.8 M1). Round 7: the
    returned value carries the pair form's parity discount
    (EFF_FORM_PARITY_CFG) - this function is the pair value's
    single choke point, so the blend and the all-blocked
    fallback price the same read."""
    _ypc = _rate(season, "yds_per_carry", "rush_yards",
                 "rush_attempts")
    _ypr = _rate(season, "yds_per_reception", "receiving_yards",
                 "receptions")
    if _ypc is None and _ypr is None:
        return None
    _parts = []
    if _ypc is not None:
        _parts.append((_scale(_ypc, 3.5, 7.0), 0.70))
    if _ypr is not None:
        _parts.append((_scale(_ypr, 5.0, 12.0), 0.30))
    _w = sum(w for _, w in _parts)
    _v = sum(s * w for s, w in _parts) / _w
    # Round 7 (EFF_FORM_PARITY_CFG): the pair form's excess
    # over the floor is discounted by the pooled pair-vs-
    # composite share - see the cfg block. Applied here, at
    # the pair value's single choke point, so every consumer
    # (the blend's candidates and the all-blocked fallback)
    # prices the same discounted read.
    if EFF_FORM_PARITY_CFG.get("enabled") and _v is not None:
        _fl = float(EFF_FORM_PARITY_CFG.get("floor", 60.0))
        if _v > _fl:
            _v -= (_v - _fl) * float(
                EFF_FORM_PARITY_CFG.get("discount", 0.35))
    return _v


def _rb_efficiency_of_raw(season: Dict):
    """One season's RB efficiency and the form that priced it:
    (value, "composite") when the season carries play-by-play
    rush rates passing the coverage gate; (value, "ypc") for the
    box-score pair read; (None, "blocked") when the season
    carries rates that FAIL the gate - it sits out the blend in
    both forms rather than price on partial data; (None, None)
    when the season has no rate evidence at all.

    Coverage gate (hard doctrine): the team's season must show
    >= min_team_games games in the PBP file AND the file's
    reconstructed attempts must reach min_att_ratio of the
    season's stored attempts AND the reconstructed sample must
    reach min_pbp_att carries - the anchors were calibrated on
    >=100-carry seasons, and a 23-carry sample's rates (Trayone
    Gray's 2015: exp12 above the top anchor, stuff_ctx below
    the bottom one) are noise wearing the calibration's
    clothes; round 2's volume doctrine applied at the gate.
    Missing gate inputs fail closed -
    the ESPN feed's whole-game holes (Bijan's Rice 2021 / UTSA
    2022) are exactly what the gate exists for. Never score
    missing as zero: a season with no rates at all takes the
    pair read, and a profile whose candidates all fail the gate
    falls back to the pre-rebuild leg wholesale (the branch
    owns that fallback)."""
    pair = _rb_pair_value(season)
    if season.get("pbp_rush_att") is None \
            or not RB_EFF_CFG.get("enabled"):
        return pair, ("ypc" if pair is not None else None)
    _tg = season.get("pbp_team_games")
    _stored_att = _safe(season.get("rush_attempts"))
    _pbp_att = _safe(season.get("pbp_rush_att"))
    if _tg is None or _tg < RB_EFF_CFG["min_team_games"] \
            or _stored_att <= 0 \
            or _pbp_att < RB_EFF_CFG["min_att_ratio"] * _stored_att \
            or _pbp_att < float(RB_EFF_CFG.get("min_pbp_att", 0)):
        return None, "blocked"
    _terms = []
    _ypc = _rate(season, "yds_per_carry", "rush_yards",
                 "rush_attempts")
    if _ypc is not None:
        _terms.append((_scale(_ypc, 3.5, 7.0),
                       float(RB_EFF_CFG["w_ypc"])))
    if season.get("pbp_exp20_rate") is not None:
        _terms.append((_scale(_safe(season.get("pbp_exp20_rate")),
                              float(RB_EFF_CFG["exp20_lo"]),
                              float(RB_EFF_CFG["exp20_hi"])),
                       float(RB_EFF_CFG["w_exp20"])))
    if season.get("pbp_exp12_rate") is not None:
        _terms.append((_scale(_safe(season.get("pbp_exp12_rate")),
                              float(RB_EFF_CFG["exp12_lo"]),
                              float(RB_EFF_CFG["exp12_hi"])),
                       float(RB_EFF_CFG["w_exp12"])))
    _ctx = season.get("pbp_stuff_ctx")
    if _ctx is not None:
        _terms.append((100.0 - _scale(
            _safe(_ctx), float(RB_EFF_CFG["stuff_ctx_lo"]),
            float(RB_EFF_CFG["stuff_ctx_hi"])),
            float(RB_EFF_CFG["w_stuff"])))
    if not _terms:
        return pair, ("ypc" if pair is not None else None)
    _tw = sum(w for _, w in _terms)
    rush_q = sum(s * w for s, w in _terms) / _tw
    _ypr = _rate(season, "yds_per_reception", "receiving_yards",
                 "receptions")
    if _ypr is None:
        return rush_q, "composite"
    return (rush_q * 0.70 + _scale(_ypr, 5.0, 12.0) * 0.30), \
        "composite"


def _rb_efficiency_of(season: Dict):
    """One season's RB efficiency, workload-banded (v3.13
    Phase 6c): the raw read from _rb_efficiency_of_raw -
    composite or parity-discounted pair - re-priced within
    its workload window by _rb_eff_band. The pair read inside
    the raw computation is NOT banded (it is an input to the
    composite form, not a priced season); the one direct
    pair consumer, the all-blocked fallback in
    calc_efficiency_score, bands its own read. Form is
    unchanged."""
    value, form = _rb_efficiency_of_raw(season)
    if value is None:
        return value, form
    return _rb_eff_band(value, _rb_opportunities(season)), form


# v3.11 round 2: volume shrinkage on the RB efficiency leg. The
# leg prices the latest season's per-play rates on an absolute
# scale whose top a short season can saturate: Javonte Williams'
# 100.0 rests on 157 carries in an 11-game season, while Saquon
# Barkley's 76.1 rests on a 671-carry career - the leg could not
# tell a proven rate from a sampled one. Above `level`, the leg
# credits only the evidence-backed share of the excess:
#   shrunk = level + (raw - level) * n / (n + k)
# with n the opportunities (rush attempts + receptions) in the
# priced season. Below the level the print is taken at face
# value - the shrink never invents signal in either direction,
# it only declines to fully credit an uncorroborated spike (the
# production evidence cap's doctrine, with the keep factor set
# by the sample instead of a constant). It fires only when the
# priced season carries recorded volume: a stored rate with no
# counting stats behind it is evidence the leg already accepts
# (v3.8 M1), and absence of volume data is not a small sample.
EFF_VOLUME_SHRINK_CFG = {
    # v3.13 (Phase 6c): DISABLED - retired in favor of
    # workload-banded season pricing (see RB_EFF_BAND_TABLE /
    # _rb_eff_band). The shrink's noise protection is replaced
    # by the band-relative pricing itself (a short-sample
    # season must be extreme among short-sample seasons) plus
    # the ramp cap on top pay. The dict and the trailing
    # application block stay for shape; the flag is the ship.
    "enabled": False,
    "level": 85.0,
    "k": 100.0,
}


def calc_efficiency_score(
    seasons: List[Dict],
    position: str,
    eval_metrics: Optional[Dict] = None,
    wepa: Optional[Dict] = None,
    _inseason_year: Optional[int] = None,
    _season_elapsed: Optional[int] = None,
) -> Optional[float]:
    """
    Per-attempt / per-target efficiency.  Rewards quality over quantity.

    Uses the latest season as the primary signal.  When multiple seasons are
    available a consistency bonus (±5) is applied: sustained high efficiency
    across seasons is more predictive than a single-year peak.

    eval_metrics (optional): When available and confidence is sufficient:
    - QB: nfl_passer_rating (computed from box score) supplements efficiency
    - WR/TE: tprr proxy supplements yds_per_reception via a soft blend

    wepa (optional, model v2.0): {(season, wepa_type): {"adj_efficiency_score": 0-100}}.
    CFBD's opponent-adjusted efficiency percentile for the latest season is
    blended 25/75 into the box-score efficiency.  Only QB (passing) and RB
    (rushing) have WEPA coverage - WR/TE skip silently.
    """
    if not seasons:
        return 52.0  # pre-draft neutral: unknown ≠ bad

    pos = position.upper()

    # v3.8 (M1): each branch below returns None when the season it
    # reads carries no rate evidence - neither a stored rate nor
    # counting stats to derive one from. (A stored rate alone IS
    # evidence: leg fixtures and some sources carry rates without
    # counting stats, and they measure fine.)
    ls  = _latest_season(seasons) or {}

    if pos == "WR":
        _ypr_wr = _rate(ls, "yds_per_reception", "receiving_yards",
                        "receptions")
        if _ypr_wr is None:
            return None
        ypr = _ypr_wr
        # v3.6 (round-2 audit F5): market share leaves WR efficiency.
        # Share was priced in four legs (production dominator,
        # efficiency, utilization, breakout) - about 18% of the WR
        # grade. Efficiency now prices per-play quality only: yards
        # per reception (anchors widened in the 3.6 de-saturation:
        # 17.0 -> 19.5 top), plus yards after catch per reception when
        # the season carries it, at 70/30. Without YAC data the leg is
        # yards per reception alone.
        _yac_wr = _safe(ls.get("yards_after_catch_per_reception"))
        if _yac_wr:
            eff = (_scale(ypr, 8.5, 19.5) * 0.70
                   + _scale(_yac_wr, 2.5, 7.5) * 0.30)
        else:
            eff = _scale(ypr, 8.5, 19.5)

        # Real PFF YPRR - centered adjustment: 2.0 = neutral, 2.8 = +8, 1.2 = -8
        yprr = _eval_metric_value(eval_metrics, "yprr", min_confidence=0.75)
        if yprr is not None:
            yprr_score = _scale(float(yprr), 1.2, 2.8)
            eff = _clip(eff + (yprr_score - 50.0) * 0.16)

        # grades_pass_route dropped: was PFF-only, no free replacement.

        # Success rate vs. press coverage
        press_sr = _eval_metric_value(eval_metrics, "success_rate_vs_press", min_confidence=0.70)
        if press_sr is not None:
            press_score = _scale(float(press_sr), 55.0, 85.0)
            eff = _clip(eff + (press_score - 50.0) * 0.06)

        # Success rate vs. man coverage - strongest coverage-type predictor for NFL transition
        man_sr = _eval_metric_value(eval_metrics, "success_rate_vs_man", min_confidence=0.70)
        if man_sr is not None:
            man_score = _scale(float(man_sr), 45.0, 75.0)
            eff = _clip(eff + (man_score - 50.0) * 0.10)

        # Success rate vs. zone coverage
        zone_sr = _eval_metric_value(eval_metrics, "success_rate_vs_zone", min_confidence=0.70)
        if zone_sr is not None:
            zone_score = _scale(float(zone_sr), 52.0, 82.0)
            eff = _clip(eff + (zone_score - 50.0) * 0.08)

        # Contested catch rate - hands + body catching in traffic; physical ceiling signal
        contested_rp = _eval_metric_value(eval_metrics, "contested_catch_rate_rp", min_confidence=0.65)
        if contested_rp is not None:
            contested_score = _scale(float(contested_rp), 38.0, 70.0)
            eff = _clip(eff + (contested_score - 50.0) * 0.08)

        # Tackle break rate - YAC/elusiveness after the catch
        tbr = _eval_metric_value(eval_metrics, "tackle_break_rate", min_confidence=0.65)
        if tbr is not None:
            tbr_score = _scale(float(tbr), 8.0, 28.0)
            eff = _clip(eff + (tbr_score - 50.0) * 0.06)

        # Route target rate from RP - centered adjustment: 30% = neutral, 42% = +6, 18% = -6
        rtr = _eval_metric_value(eval_metrics, "route_target_rate", min_confidence=0.75)
        if rtr is not None:
            tprr_equiv = float(rtr) / 100.0
            rtr_score  = _scale(tprr_equiv, 0.18, 0.42)
            eff = _clip(eff + (rtr_score - 50.0) * 0.12)
        else:
            # Tprr proxy - lower priority fallback when RP data absent
            tprr = _eval_metric_value(eval_metrics, "tprr", min_confidence=0.30)
            if tprr is not None:
                tprr_score = _scale(float(tprr), 0.18, 0.42)
                tprr_conf  = (eval_metrics.get("tprr") or {}).get("confidence", 0.35)
                eff = eff * (1.0 - 0.08 * tprr_conf) + tprr_score * (0.08 * tprr_conf)

    elif pos == "RB":
        # v3.11 round 4: the leg prices per featured season and
        # blends 0.90 x higher + 0.10 x lower - production's
        # treatment - instead of reading the latest season alone
        # (the Baugh defect: two featured seasons priced off a
        # lone 4.98-YPC season at 33.61, a 23-point crater under
        # his next-worst leg). Candidates are the latest season
        # and the peak season (by rush yards, production's RB
        # peak key); each prices via _rb_efficiency_of - the
        # rush-quality composite when its PBP rates pass the
        # coverage gate, the box-score pair otherwise. A
        # candidate whose rates FAIL the gate is blocked and
        # contributes no value in either form; if every candidate
        # is blocked, the leg falls back to the pre-rebuild read
        # exactly (the latest season's pair value, with the
        # trailing adjustments below untouched). The branch owns
        # RB's in-season treatment in production's form - an
        # in-progress blend input shrinks toward the best
        # completed season's read (52.0 when none exists) before
        # blending, a lone in-progress season toward 52.0 - so
        # the trailing F2 block skips RB, as it does QB and TE.
        _ls_val, _ls_form = _rb_efficiency_of(ls)
        if _inseason_year is not None:
            # Production's in-season peak key: 12-game-
            # equivalent rate, so a partial season's totals
            # cannot win or lose the peak slot on volume alone.
            def _rb_peak_fn(s):
                _g = _safe(s.get("games_played"), 12)
                _g = _g if _g >= 4 else 12
                return _safe(s.get("rush_yards")) / max(_g, 12)
            _peak_rb = max(seasons, key=_rb_peak_fn)
        else:
            _peak_rb = max(
                seasons, key=lambda s: _safe(s.get("rush_yards")),
                default=ls)
        if _peak_rb is not ls:
            _pk_val, _pk_form = _rb_efficiency_of(_peak_rb)
        else:
            _pk_val, _pk_form = _ls_val, _ls_form
        _vals = [v for v in (_ls_val, _pk_val) if v is not None]
        if not _vals:
            if "blocked" in (_ls_form, _pk_form):
                # Every candidate carried rates and failed the
                # gate: the pair read, verbatim (with its
                # round-7 parity discount, which lives inside
                # _rb_pair_value).
                # v3.13 (Phase 6c): this direct pair read is a
                # priced season, so it bands like one (the pair
                # read inside _rb_efficiency_of_raw does not -
                # there it is an input, not a price).
                eff = _rb_pair_value(ls)
                if eff is not None:
                    eff = _rb_eff_band(eff, _rb_opportunities(ls))
                if eff is None:
                    return None
            else:
                return None
        else:
            if _inseason_year is not None:
                _completed_vals = [
                    v for v, _f in
                    (_rb_efficiency_of(s) for s in seasons
                     if s.get("season") != _inseason_year)
                    if v is not None]
                _base_rb = (max(_completed_vals) if _completed_vals
                            else 52.0)
                if ls.get("season") == _inseason_year \
                        and _ls_val is not None:
                    _ls_val = _shrink_partial(
                        _ls_val, _base_rb,
                        _safe(ls.get("games_played"), 12),
                        elapsed=_season_elapsed)
                    if _peak_rb is ls:
                        _pk_val = _ls_val
                elif _peak_rb.get("season") == _inseason_year \
                        and _pk_val is not None:
                    _pk_val = _shrink_partial(
                        _pk_val, _base_rb,
                        _safe(_peak_rb.get("games_played"), 12),
                        elapsed=_season_elapsed)
                _vals = [v for v in (_ls_val, _pk_val)
                         if v is not None]
            eff = (max(_vals) * 0.90 + min(_vals) * 0.10
                   if len(_vals) == 2 else _vals[0])
        # grades_pass_route dropped: was PFF-only, no free replacement.

    elif pos == "QB":
        # v3.8 (M1): the rate trio comes from _qb_efficiency_of - stored
        # rates when present, counting-stat derivations otherwise, and
        # None when a season carries no rate evidence at all (the
        # friendly defaults YPA 7.0 / comp 62% / TD:INT 2.0 are gone).
        eff = _qb_efficiency_of(ls)

        # v3.2 Q4: one bad partial season should not overwrite a full elite
        # season. Peak blend: 85% best (by pass yards) with 15% latest.
        # (Comment corrected in v3.6, round-2 audit F12: it long claimed
        # a 70/30 latest/best blend the code never performed.)
        if len(seasons) >= 2:
            if _inseason_year is not None:
                # v3.4: same 12-game-equivalent rate key as production,
                # so the peak slot is not decided by raw totals mid-season.
                def _qb_peak_fn(s):
                    _g = _safe(s.get("games_played"), 12)
                    _g = _g if _g >= 4 else 12
                    return _safe(s.get("pass_yards")) / max(_g, 12)
                _best_qb = max(seasons, key=_qb_peak_fn)
            else:
                _best_qb = max(seasons, key=lambda s: _safe(s.get("pass_yards")))
            if _best_qb is not ls:
                _b_eff = _qb_efficiency_of(_best_qb)
                if eff is None:
                    eff = _b_eff
                elif _b_eff is not None:
                    # v3.6 (round-2 audit F2): whichever side of the blend
                    # is the in-progress season shrinks toward the other
                    # (completed) side by min(1, GP/12) before blending.
                    # At most one side can be the in-progress season.
                    if _inseason_year is not None:
                        if ls.get("season") == _inseason_year:
                            eff = _shrink_partial(
                                eff, _b_eff, _safe(ls.get("games_played"), 12),
                                elapsed=_season_elapsed)
                        elif _best_qb.get("season") == _inseason_year:
                            _b_eff = _shrink_partial(
                                _b_eff, eff,
                                _safe(_best_qb.get("games_played"), 12),
                                elapsed=_season_elapsed)
                    eff = eff * 0.15 + _b_eff * 0.85  # v3.2 R7-A2: peak blend matches production (85% peak / 15% latest)
                elif (_inseason_year is not None
                        and ls.get("season") == _inseason_year):
                    # Only the in-progress season carries rate evidence:
                    # shrink it toward the leg neutral, the lone-season
                    # doctrine production already uses.
                    eff = _shrink_partial(
                        eff, 52.0, _safe(ls.get("games_played"), 12),
                        elapsed=_season_elapsed)
        if eff is None:
            return None

    elif pos == "TE":
        # v3.11 round 3: the leg prices the best season the way
        # production does - 0.90 x higher + 0.10 x lower of the
        # latest season vs the peak season (by receiving yards,
        # production's TE peak key) - not the latest season alone.
        # A TE's final college season is systematically his least
        # representative efficiency sample: the offense schemes
        # him as a safety valve, and late injuries land there
        # (Bowers 2023: 12.75 YPR after 15.75 / 14.95 priced the
        # whole leg at 79.57). Per-season values come from
        # _te_efficiency_of; a season with no rate evidence is
        # skipped, and the leg is missing only when NO season
        # carries any (the QB branch's M1 form). The tprr
        # supplement is player-level and still applies once, to
        # the blended leg, exactly as before.
        eff = _te_efficiency_of(ls)
        if len(seasons) >= 2:
            if _inseason_year is not None:
                # Production's in-season peak key: 12-game-
                # equivalent rate, so a partial season's totals
                # cannot win or lose the peak slot on volume alone.
                def _te_peak_fn(s):
                    _g = _safe(s.get("games_played"), 12)
                    _g = _g if _g >= 4 else 12
                    return _safe(s.get("receiving_yards")) / max(_g, 12)
                _peak_te = max(seasons, key=_te_peak_fn)
            else:
                _peak_te = max(
                    seasons,
                    key=lambda s: _safe(s.get("receiving_yards")))
            _peak_eff = (_te_efficiency_of(_peak_te)
                         if _peak_te is not ls else eff)
            if _inseason_year is not None:
                # v3.6 (F2) doctrine in production's form: an
                # in-progress blend input shrinks toward the best
                # completed season's read (the leg neutral 52.0
                # when none exists) before blending. The branch
                # owns TE's in-season treatment, as QB's does, so
                # the trailing F2 block below skips TE.
                _completed_eff = [
                    v for v in (_te_efficiency_of(s) for s in seasons
                                if s.get("season") != _inseason_year)
                    if v is not None]
                _base_te = (max(_completed_eff) if _completed_eff
                            else 52.0)
                if ls.get("season") == _inseason_year \
                        and eff is not None:
                    eff = _shrink_partial(
                        eff, _base_te,
                        _safe(ls.get("games_played"), 12),
                        elapsed=_season_elapsed)
                    if _peak_te is ls:
                        _peak_eff = eff
                elif _peak_te.get("season") == _inseason_year \
                        and _peak_eff is not None:
                    _peak_eff = _shrink_partial(
                        _peak_eff, _base_te,
                        _safe(_peak_te.get("games_played"), 12),
                        elapsed=_season_elapsed)
            if eff is None:
                eff = _peak_eff
            elif _peak_eff is not None:
                eff = (max(eff, _peak_eff) * 0.90
                       + min(eff, _peak_eff) * 0.10)
        elif _inseason_year is not None \
                and ls.get("season") == _inseason_year \
                and eff is not None:
            # Lone in-progress season: shrink toward the leg
            # neutral - no completed baseline exists.
            eff = _shrink_partial(
                eff, 52.0, _safe(ls.get("games_played"), 12),
                elapsed=_season_elapsed)
        if eff is None:
            return None
        # Supplement with tprr proxy when available
        tprr = _eval_metric_value(eval_metrics, "tprr", min_confidence=0.30)
        if tprr is not None:
            tprr_score = _scale(float(tprr), 0.22, 0.48)
            tprr_conf  = (eval_metrics.get("tprr") or {}).get("confidence", 0.35)
            eff = eff * (1.0 - 0.08 * tprr_conf) + tprr_score * (0.08 * tprr_conf)

        # grades_pass_route dropped: was PFF-only, no free replacement.

    else:
        return 52.0

    # Multi-season consistency bonus: ±5 points based on whether the key
    # efficiency metric held up across seasons (sustained efficiency > one-year wonder)
    if len(seasons) >= 2:
        _KEY = {"WR": "yds_per_reception", "RB": "yds_per_carry",
                "QB": "yds_per_attempt",   "TE": "yds_per_reception"}
        key = _KEY.get(pos)
        if key:
            vals = [_safe(s.get(key)) for s in seasons if s.get(key) is not None]
            if len(vals) >= 2:
                avg_val = sum(vals) / len(vals)
                latest_val = _safe(ls.get(key))
                # Bonus if latest ≈ or exceeds multi-year average; penalty if big drop
                consistency = (latest_val - avg_val) / max(avg_val, 0.1)
                eff = _clip(eff + _clip(consistency * 10, -5.0, 5.0))

    # Advanced efficiency adjustments from evaluation metrics.
    if eval_metrics and pos in ("WR", "TE"):
        # drop_rate dropped: was PFF-only, no free replacement.

        yac_rec = _eval_metric_value(eval_metrics, "yac_per_att", min_confidence=0.45)
        if yac_rec is not None:
            eff = _clip(eff + _clip((float(yac_rec) - 5.0) * 1.6, -4.0, 6.0))

    elif eval_metrics and pos == "RB":
        # pff_rushing_grade is now sourced from CFBD PPA (free replacement for PFF).
        pff_rush = _eval_metric_value(eval_metrics, "pff_rushing_grade", min_confidence=0.45)
        if pff_rush is not None:
            eff = _clip(eff + _clip((float(pff_rush) - 68.0) * 0.22, -5.0, 7.0))

    elif eval_metrics and pos == "QB":
        # adjusted_comp_pct dropped: was PFF-only, no free replacement.
        # nfl_passer_rating is now computed from box-score data (free replacement).
        psr = _eval_metric_value(eval_metrics, "nfl_passer_rating", min_confidence=0.45)
        if psr is not None:
            eff = _clip(eff + _clip((float(psr) - 85.0) * 0.18, -4.0, 6.0))

        # pressure_to_sack_rate dropped: was PFF-only, no free replacement.

    # v3.6 (round-2 audit F7): scheme-inflation discount deleted (see
    # calc_production_score); competition + environment carry context.

    if pos not in ("QB", "TE", "RB") and _inseason_year is not None \
            and ls.get("season") == _inseason_year:
        # v3.6 (round-2 audit F2): WR efficiency reads the latest
        # season only, so an in-progress latest season shrinks toward
        # the efficiency of the completed seasons alone (the leg's
        # neutral 52.0 when there are none) by min(1, GP/12). QB, TE
        # and RB handle it inside their peak-blend branches above.
        # The recursion runs
        # without eval metrics or WEPA so the baseline is the plain
        # completed-season read.
        _completed = [s for s in seasons
                      if s.get("season") != _inseason_year]
        _base_eff = (calc_efficiency_score(_completed, pos)
                     if _completed else 52.0)
        if _base_eff is None:
            # v3.8: the completed seasons carry no rate evidence
            # either - shrink toward the leg neutral.
            _base_eff = 52.0
        eff = _shrink_partial(eff, _base_eff,
                              _safe(ls.get("games_played"), 12),
                              elapsed=_season_elapsed)

    # Model v2.0: opponent-adjusted WEPA blend.  wepa_score is a 0-100
    # percentile within the season/position cohort - a conservative 25% blend
    # so box-score efficiency stays primary and WEPA only nudges.
    if wepa:
        _WEPA_TYPE = {"QB": "passing", "RB": "rushing"}
        wtype = _WEPA_TYPE.get(pos)
        if wtype:
            rec = wepa.get((ls.get("season", 0), wtype)) or {}
            wscore = rec.get("adj_efficiency_score")
            if wscore is not None:
                try:
                    eff = _clip(eff * 0.75 + float(wscore) * 0.25)
                except (TypeError, ValueError):
                    pass

    # v3.11 round 2 (EFF_VOLUME_SHRINK_CFG): an elite RB rate
    # print must be backed by elite evidence volume - shrink the
    # excess over the cfg level by the priced season's share of
    # proof. No recorded volume in the priced season, no shrink.
    # An in-progress priced season is skipped: the F2 shrink
    # above already prices its partiality, and the two evidence
    # questions stay orthogonal.
    if (pos == "RB" and EFF_VOLUME_SHRINK_CFG.get("enabled")
            and eff is not None
            and not (_inseason_year is not None
                     and ls.get("season") == _inseason_year)):
        _n = (_safe(ls.get("rush_attempts"))
              + _safe(ls.get("receptions")))
        _lvl = float(EFF_VOLUME_SHRINK_CFG.get("level", 85.0))
        if _n > 0 and eff > _lvl:
            _k = float(EFF_VOLUME_SHRINK_CFG.get("k", 100.0))
            eff = _lvl + (eff - _lvl) * (_n / (_n + _k))

    return _clip(eff)


# Typical draft-class age by position (age at start of NFL rookie year).
# Updated to reflect modern college football (COVID year, grad transfers, etc.)
_TYPICAL_AGE = {"QB": 23.0, "RB": 22.5, "WR": 22.5, "TE": 23.0}
_AGE_ELITE   = {"QB": 21.0, "RB": 20.5, "WR": 21.0, "TE": 21.0}  # Raised from 20.5 - elite TEs can be slightly older
_AGE_WORST   = {"QB": 27.5, "RB": 25.0, "WR": 25.5, "TE": 26.0}   # QB more lenient - development timelines vary widely


def calc_age_score(age: Optional[float], draft_year: int, position: str) -> float:
    """
    Younger prospects earn a premium.  Age is evaluated relative to draft class.
    A 20-year-old RB producing at a high level is worth more than a 23-year-old.
    """
    if age is None:
        return 50.0  # neutral default
    # Convert decimal to float if needed
    age = float(age)
    pos = position.upper()
    elite = _AGE_ELITE.get(pos, 21.5)
    worst = _AGE_WORST.get(pos, 26.0)
    # Invert: lower age → higher score.  _scale maps [elite, worst] → [100, 0].
    # v3.2 R7-A6: cap 100 (was 95): the elite anchor IS the 100 anchor.
    score = _scale(worst - age, 0.0, worst - elite)
    # v3.6 (round-2 audit F11): the old flat floor at 20 collapsed the
    # entire old tail - every WR aged 25.5+ scored identically. Below
    # the raw-20 point the score now declines in AGE space at 5
    # points per year down to a floor of 10 (a 24.8-year-old WR reads
    # 19, a 25.5 reads 15.5, a 26.5 reads 10.5), so the tail separates
    # without moving the calibrated mid-scale at all. Note the
    # documented overlap: the breakout leg's adjusted-breakout-age
    # term (20% of that leg) is a second, dominance-gated age curve
    # by design - it prices WHEN a player broke out, this leg prices
    # his age at the draft.
    # v3.8 (M4, GATE-REJECTED in the post-build re-gate and reverted
    # before ship): the under-rating audit's residual frame showed
    # players aged 23+ under-rated on average (+9.5 mean residual),
    # and the first 3.8 build halved the per-year slope beyond 23.0
    # on that evidence. The re-gate arbitrated against it: with every
    # other fix held constant, the flattened slope COST correlation
    # at three positions (QB Pearson -.011, TE Spearman -.016, WR
    # Spearman -.013) - a level-bias read on a subset does not
    # survive as a blanket re-ranking of every older prospect, most
    # of whom are older because they needed the years. The 3.7 slope
    # stands; M4 is recorded here as tried and rejected so the next
    # audit does not re-propose it without new evidence.
    if score < 20.0:
        _a20 = worst - 0.2 * (worst - elite)
        score = max(10.0, 20.0 - (age - _a20) * 5.0)
    return _clip(score, 10, 100)


DOM_CFG = {"curve_power": 0.65}


def _dom_ratio_score(ratio: float, pos: str) -> float:
    """The breakout leg's dominator curve: a dominator ratio (vs the
    position threshold) priced on the leg's own scale. Above the
    threshold the curve saturates at the v3.6 multiples (WR 1.7x,
    RB 2.0x, TE 2.4x - v3.3 F3 introduced the saturation, v3.6
    raised the multiples to recorded-best-plus-margin so the curve
    stops pinning at marks real breakout seasons actually reach);
    below it, it ramps linearly through 60 at the threshold. Shared
    by the peak-dominator term and the v3.9 arrival term so both
    price level on one scale.

    DOM_CFG["curve_power"] (v3.10: 0.65; the shipped value through
    3.9 was 1.0, the linear segment) bends the above-threshold
    segment: a power below 1 steepens the slope just above the
    threshold, where real breakouts cluster, and flattens it
    toward saturation, where one generational share is much like
    another. Under the linear segment a 1.23x-threshold season
    priced 73 while the run from there to saturation was worth
    more than the run from ordinary to elite; at 0.65 that season
    prices 79 and a 1.5x season 92 (was 89), with the threshold
    (60) and saturation (100) anchors unchanged."""
    if ratio > 1:
        _sat = {"WR": 1.7, "RB": 2.0, "TE": 2.4}.get(pos, 1.8)
        _x = min((ratio - 1) / (_sat - 1), 1)
        _p = DOM_CFG.get("curve_power", 1.0)
        if _p != 1.0:
            _x = _x ** _p
        return min(100, 60 + 40 * _x)
    return min(100, max(0, (ratio - 1) * 50 + 60))


# v3.9 (arrive-elite fix): the trajectory term prices RISE - per-game
# growth from the first season to the peak. A player who ARRIVES
# elite has no rise to price: his debut is at (or near) his peak, so
# the growth scale reads his flat-elite path at ~25 and the leg
# lands mediocre for exactly the profiles it should crown (Bowers'
# breakout read 55 with a freshman season at his career level).
# The arrival term prices the other shape of the same evidence: the
# LEVEL of the first featured season, on the leg's own dominator
# curve. traj takes the better of the two - a riser keeps his
# growth price (his debut level was ordinary), an arriver is priced
# on the level he arrived at. When the debut season IS the peak,
# the arrival credit is discounted by how much of the debut's
# per-game rate he held in his latest season, so a one-year wonder
# cannot bank the full price. QB is untouched: the QB breakout
# weight is popped before blending (v3.2 V6c), so the leg is never
# priced for QBs.
BREAKOUT_CFG = {"arrival": True,
                "age_anchor_draft": True,
                "gate_lo": 0.65,
                "gate_span": 0.4}


def calc_breakout_score(seasons: List[Dict], age: Optional[float], position: str,
                        _inseason_year: Optional[int] = None,
                        _draft_year: Optional[int] = None) -> Optional[float]:
    """
    Rewards early-career dominance and upward trajectory.

    Key signals:
    - Was the best season achieved at ≤20 (WR/RB/TE) or ≤21 (QB)?
    - Did production increase year-over-year?
    - Is dominator_rating above the breakout threshold?
    """
    if not seasons or len(seasons) < 1:
        return 52.0

    pos = position.upper()

    # v3.8 (M1): none of the position-relevant counting fields was
    # ever recorded - no dominance or trajectory to measure. None,
    # not a priced zero. Recorded zeros measure (see
    # _has_recorded_volume).
    if pos in _PROD_VOLUME_KEYS and not _has_recorded_volume(
            seasons, _PROD_VOLUME_KEYS[pos]):
        return None

    sorted_s = sorted(seasons, key=lambda s: _safe(s.get("season"), 0))
    ls = sorted_s[-1]  # most recent season

    # Dominator breakout threshold by position.
    # QB dominator_rating is receiving-based and not meaningful for QBs - use neutral.
    dom_thresh = {"WR": 0.25, "RB": 0.275, "TE": 0.12}
    # v3.2 R7-A4: peak-season dominator (TE: pass share, per T5).
    def _dom_of(_s):
        if pos == "TE":
            _tpy = _safe(_s.get("team_pass_yards"))
            if _tpy > 0:
                return _safe(_s.get("receiving_yards")) / _tpy
        return _safe(_s.get("dominator_rating"))
    _peak_s = max(sorted_s, key=_dom_of) if sorted_s else ls
    dom = _dom_of(_peak_s)
    thresh = dom_thresh.get(pos)

    if pos == "QB":
        # QB dominator proxy: pass yards as share of team total yards
        # A QB generating ≥60% of team total yards is an alpha
        pass_yds = _safe(ls.get("pass_yards"))
        team_yds = _safe(ls.get("team_total_yards"))
        if team_yds > 0:
            qb_share = pass_yds / team_yds
            dom_score = _scale(qb_share, 0.40, 0.70)
        else:
            # v3.8 (M1, pool class P6): team totals missing means the
            # share cannot be computed. The old fallback priced raw
            # pass yards per game on a 200-380 scale, which floored
            # every run-first QB at a literal 0.0 (seven pool QBs sat
            # at breakout 0.0, all dual-threat). The leg is missing,
            # not zero: return None and let the weight redistribute.
            return None
    elif thresh is None:
        dom_score = 50.0  # unknown position
    elif dom > 0:
        dom_score = _dom_ratio_score(dom / thresh, pos)
    else:
        dom_score = 30.0

    # Trajectory INTO the breakout season (per-game rates, so an
    # injury-shortened final season cannot erase the breakout).
    traj_score = 50.0
    if len(sorted_s) >= 2:
        def _pg(_s):
            _y = _safe(_s.get("receiving_yards", 0)) + _safe(_s.get("rush_yards", 0)) + _safe(_s.get("pass_yards", 0)) * 0.5
            return _y / max(_safe(_s.get("games_played"), 1), 1)
        if _peak_s is sorted_s[0]:
            # Broke out immediately: trajectory is whether he SUSTAINED it.
            _a, _b = _peak_s, sorted_s[-1]
        else:
            _a, _b = sorted_s[0], _peak_s
        if _pg(_a) > 0:
            # v3.6 (round-2 audit F9): trajectory is scored on PER-GAME
            # growth only. The v3.3 totals-growth switch is dropped: it
            # manufactured growth from games played (a 4-game first
            # season into a 13-game peak at the same rate read as
            # ~+200% growth and pinned traj at 100) - a games-played
            # artifact, not development. Per-game growth is the single
            # growth price in this leg (see also the deleted ascending
            # bonus below).
            _growth = (_pg(_b) - _pg(_a)) / _pg(_a)
            traj_score = _clip(_scale(_growth, -0.20, 0.60))

        # v3.9 arrival term (see BREAKOUT_CFG): the better of the
        # rise price and the arrival price. The arrival season is
        # the first FEATURED season (a cameo debut is the growth
        # origin, not an arrival); its dominator level is priced on
        # the leg's own curve. When that season is also the peak,
        # the credit is discounted by the share of its per-game
        # rate held in the latest season (floor 0.5: a real decline
        # halves it, an injury-shortened rate season cannot erase
        # it - _pg is a rate, not a total).
        if BREAKOUT_CFG.get("arrival") and thresh is not None:
            _feat = next((s for s in sorted_s
                          if _is_role_season(s, pos)), sorted_s[0])
            _fdom = _dom_of(_feat)
            if _fdom > 0:
                _arrival = _dom_ratio_score(_fdom / thresh, pos)
                if _peak_s is _feat and _pg(_feat) > 0:
                    _arrival *= _clip(
                        _pg(sorted_s[-1]) / _pg(_feat), 0.5, 1.0)
                traj_score = max(traj_score, _arrival)

    # Age-in-season anchor, shared by the adjusted-breakout-age term
    # and the elite underclassman floor. The pre-3.10 anchor was
    # `age` (the age at the draft) minus the whole seasons between
    # the target season and the LAST season on file. That silently
    # charged a player his draft age for a season played years
    # earlier whenever gap years followed it (opt-out, injury,
    # COVID): Chase's 1,780-yard 2019 priced as a 21-year-old's
    # season (his draft age) because 2020 is missing from his file,
    # and the underclassman floor never fired for a season he
    # played at 19. v3.10 ships the draft anchor (BREAKOUT_CFG
    # age_anchor_draft, needs _draft_year): the age he actually was
    # in that season - draft age less the years to the draft, plus
    # 0.45 back from the late-April draft to the mid-November
    # season midpoint. The legacy path stands only for callers
    # that cannot supply a draft year. The dominance multiplier
    # below was evaluated for softening in the same pass and left
    # at its v3.2 shape: on a correctly anchored age it was worth
    # ~1 leg point for the index case and nothing measurable at
    # the gate - the tax was the anchor, not the gate.
    def _age_in_season(_s_year, _current_year):
        if BREAKOUT_CFG.get("age_anchor_draft") and _draft_year:
            return age - (_draft_year - _s_year) + 0.45
        return age - (_current_year - _s_year)

    # Adjusted breakout age - continuous scale calibrated by position.
    # Uses the age the player was in the FIRST season they hit the dom threshold,
    # not their current age.  A WR who dominated at 18 is a generational rarity;
    # a WR who first broke out at 22 as a senior is ordinary.
    #
    # Scale anchors (WR/RB/TE):  ≤19 → 30 pts,  20 → 20,  21 → 10,  22 → 0,  23+ → -10
    # Scale anchors (QB):        ≤19 → 25 pts,  20 → 18,  21 → 10,  22 → 0,  23+ → -8
    adj_breakout_score = 0.0
    if age is not None:
        # Find age at first breakout season
        breakout_age = age  # default to current age
        if thresh is not None and sorted_s:
            current_year = _safe(sorted_s[-1].get("season"), 0)
            _crossed = False
            for s in sorted_s:
                s_year = _safe(s.get("season"), 0)
                if _dom_of(s) >= thresh and s_year > 0 and current_year > 0:
                    breakout_age = _age_in_season(s_year, current_year)
                    _crossed = True
                    break
            if not _crossed and current_year > 0 and _safe(_peak_s.get("season"), 0) > 0:
                breakout_age = _age_in_season(
                    _safe(_peak_s.get("season"), 0), current_year)

        if pos == "QB":
            # QBs develop later: age 18.5→100, age 23.5→0
            # Inverted: lower age = higher score, so compute (max_age - age) / range
            raw = (23.5 - breakout_age) / (23.5 - 18.5) * 100.0
            adj_breakout_score = _clip(raw, 0.0, 100.0)
        else:
            # Skill positions (WR/RB/TE): age 18.5→100, age 23.0→0
            raw = (23.0 - breakout_age) / (23.0 - 18.5) * 100.0
            adj_breakout_score = _clip(raw, 0.0, 100.0)

        # Dominance quality gate: the early-age bonus should reflect HOW dominant
        # the player was, not merely that they cleared the threshold.  A player
        # barely above threshold (dom ≈ thresh) should not receive the same age
        # credit as a player who doubled the threshold.
        # Improved scale: dom = thresh → 50% credit; dom = 1.5×thresh → 100% credit.
        # The old scale (0% at threshold) was too harsh, especially for in-progress
        # seasons where DOM can be volatile with small samples.
        if thresh is not None and thresh > 0 and pos != "QB":
            ratio = dom / thresh
            _g_lo = BREAKOUT_CFG.get("gate_lo", 0.65)
            _g_span = BREAKOUT_CFG.get("gate_span", 0.4)
            if ratio >= 1.0:
                # v3.2 V6b: 1.0x -> 0.65, 1.4x -> 1.0 (the shipped
                # gate_lo / gate_span defaults)
                dominance_factor = min(
                    1.0, _g_lo + (ratio - 1.0) / _g_span * (1.0 - _g_lo))
            else:
                # v3.2 V6b: smooth ramp below threshold, no knife edge
                dominance_factor = _g_lo * ratio * ratio
            adj_breakout_score *= dominance_factor

    score = dom_score * 0.45 + traj_score * 0.35 + adj_breakout_score * 0.20

    # Elite underclassman floor: a freshman/sophomore (age <= 20 in that
    # season) with elite raw production has proven breakout ability regardless
    # of what the weighted components say. Don't let trajectory noise from
    # in-progress seasons or DOM thresholds on stacked teams drag down a
    # genuinely elite early-career season.
    # WR/TE: 1,200+ receiving yards; RB: 1,200+ rush yards; QB: 3,500+ pass yards
    if age is not None and sorted_s:
        current_year = _safe(sorted_s[-1].get("season"), 0)
        for s in sorted_s:
            s_year = _safe(s.get("season"), 0)
            if s_year <= 0 or current_year <= 0:
                continue
            s_age = _age_in_season(s_year, current_year)
            if s_age > 20:
                continue
            # v3.4: for the in-progress season, the yardage gates are
            # counting thresholds, so the season's yards are paced to a
            # 12-game equivalent before they are compared (and before the
            # floor value is computed from them).
            _gp_r = _safe(s.get("games_played"), 0)
            _paced = (
                _inseason_year is not None and s_year == _inseason_year
                and 0 < _gp_r < 12
            )
            def _y_of(key, _s=s):
                _v = _safe(_s.get(key, 0))
                return _v * (12.0 / _gp_r) if _paced else _v
            # v3.2 V6a: ramp the floor (70 at the low end, 85 at the old cliff)
            _floor = None
            if pos in ("WR", "TE"):
                _y = _y_of("receiving_yards")
                if _y >= 1000:
                    _floor = 70.0 + min(1.0, (_y - 1000.0) / 200.0) * 15.0
            elif pos == "RB":
                _y = _y_of("rush_yards")
                if _y >= 1000:
                    _floor = 70.0 + min(1.0, (_y - 1000.0) / 200.0) * 15.0
            elif pos == "QB":
                _y = _y_of("pass_yards")
                if _y >= 3000:
                    _floor = 70.0 + min(1.0, (_y - 3000.0) / 500.0) * 15.0
            if _floor is not None:
                score = max(score, _floor)
                break

    # v3.6 (round-2 audit F9): the v3.1 ascending-trajectory bonus
    # (+15 at >=50% latest-vs-prior per-game growth) is DELETED. When
    # the peak season is the latest season, traj_score already prices
    # exactly that growth; the bonus priced it a second time inside
    # the same leg. Growth is priced once now, by traj_score. The
    # elite-underclassman floor above is a level floor, not a growth
    # price, and stays.

    return _clip(score)


# Position-specific weights for athleticism metrics.
# Reflects NFL scouting priorities: speed matters most for WR, explosiveness for RB,
# overall athleticism (RAS) for QB, catching radius (vertical) for TE.
# three_cone / shuttle = agility/change-of-direction, important for WR routes and RB open-field.
_ATH_WEIGHTS: Dict[str, Dict[str, float]] = {
    "WR": {"forty": 0.35, "vertical": 0.20, "broad": 0.15, "ras": 0.15,
           "speed_score": 0.25, "three_cone": 0.10, "shuttle": 0.05},
    "RB": {"speed_score": 0.30, "broad": 0.25, "forty": 0.15, "vertical": 0.10,
           "ras": 0.15, "three_cone": 0.05},
    "QB": {"ras": 0.40, "forty": 0.35, "vertical": 0.15, "broad": 0.10},
    "TE": {"vertical": 0.30, "ras": 0.25, "broad": 0.20, "forty": 0.15,
           "three_cone": 0.10},
}

# Maximum athleticism score when metric coverage is sparse.
# Prevents a single elite metric (e.g. one 4.28 40-time) from yielding a top score
# when we have no idea about the rest of the athletic profile.
_ATH_DATA_CAPS = {1: 74, 2: 82, 3: 96}   # n_metrics_present → cap


# ── Knot-anchored combine subscales (v3.3) ─────────────────────────────────
# Every combine leg except RAS is scored against recorded marks at
# performance tiers of drafted players at the position (nflverse combine
# data, drafted only, classes 2000-2025): piecewise-linear through the tier
# marks below, listed in tier order worst / p1 / p5 / p10 / p25 / p50 / p75 /
# p90 / p97 / p99 / best and priced by _KNOT_SCORES. The marks are fixed
# recorded constants, so the scale stays absolute across years. RAS has no
# drafted-tester distribution in that source and stays on its linear
# 4.0-10.0 scale.
_KNOT_SCORES = [0, 5, 12, 22, 38, 56, 77, 88, 93, 96, 100]

_KNOT_MARKS = {
    "QB": {
        "forty": [5.37, 5.266, 5.102, 5.004, 4.9, 4.81, 4.69, 4.58, 4.511, 4.411, 4.33],
        "vertical": [21.5, 24.51, 26.5, 27.05, 29.5, 31.5, 33.875, 36.0, 37.5, 38.5, 40.5],
        "broad": [96.0, 99.0, 100.0, 103.0, 107.0, 112.0, 116.0, 121.0, 125.0, 127.0, 129.0],
        "speed": [53.39, 57.64, 67.61, 70.65, 77.27, 84.53, 92.7, 99.65, 110.59, 118.15, 126.88],
        "cone": [7.8, 7.713, 7.517, 7.408, 7.22, 7.11, 6.96, 6.856, 6.762, 6.729, 6.65],
        "shuttle": [4.78, 4.654, 4.55, 4.52, 4.42, 4.295, 4.18, 4.09, 4.026, 3.952, 3.87],
    },
    "RB": {
        "forty": [4.83, 4.758, 4.67, 4.644, 4.58, 4.51, 4.45, 4.39, 4.355, 4.33, 4.24],
        "vertical": [26.5, 28.325, 30.0, 30.5, 33.0, 35.0, 37.0, 39.5, 41.0, 41.5, 43.0],
        "broad": [103.0, 106.53, 111.0, 113.0, 117.0, 120.0, 123.75, 127.0, 130.0, 132.94, 137.0],
        "speed": [82.9, 83.92, 88.18, 91.64, 97.16, 103.49, 109.23, 116.11, 121.84, 124.46, 129.55],
        "cone": [7.68, 7.556, 7.39, 7.27, 7.15, 7.01, 6.89, 6.81, 6.7, 6.654, 6.5],
        "shuttle": [4.67, 4.62, 4.508, 4.44, 4.348, 4.235, 4.143, 4.07, 3.995, 3.935, 3.93],
    },
    "WR": {
        "forty": [4.79, 4.711, 4.62, 4.59, 4.54, 4.47, 4.41, 4.35, 4.31, 4.279, 4.21],
        "vertical": [26.0, 28.5, 31.0, 32.5, 34.0, 36.0, 38.0, 39.5, 41.5, 42.575, 45.0],
        "broad": [105.0, 110.0, 114.0, 115.0, 118.0, 122.0, 126.0, 130.0, 134.0, 136.0, 140.0],
        "speed": [73.34, 78.76, 84.92, 87.59, 94.17, 100.27, 106.84, 112.51, 118.97, 122.78, 133.5],
        "cone": [7.54, 7.409, 7.31, 7.19, 7.07, 6.94, 6.8, 6.675, 6.612, 6.561, 6.5],
        "shuttle": [4.77, 4.57, 4.45, 4.39, 4.31, 4.2, 4.11, 4.02, 3.946, 3.886, 3.73],
    },
    "TE": {
        "forty": [5.09, 5.023, 4.93, 4.885, 4.81, 4.73, 4.64, 4.545, 4.481, 4.417, 4.37],
        "vertical": [26.5, 27.22, 29.0, 30.0, 31.5, 33.5, 35.5, 37.5, 39.5, 41.78, 43.5],
        "broad": [102.0, 104.68, 109.0, 110.0, 113.0, 116.0, 120.5, 124.0, 129.0, 131.66, 134.0],
        "speed": [78.07, 83.05, 87.04, 89.81, 94.91, 102.1, 108.03, 116.0, 122.7, 127.54, 138.03],
        "cone": [7.95, 7.652, 7.511, 7.44, 7.302, 7.12, 6.99, 6.88, 6.81, 6.76, 6.72],
        "shuttle": [4.82, 4.698, 4.57, 4.53, 4.45, 4.34, 4.25, 4.145, 4.09, 4.011, 3.98],
    },
}


def _knot_score(value, pos, leg):
    """Piecewise-linear score for one combine mark against the recorded
    drafted-tester tier marks at the position. Marks pair with scores in
    performance order, so lower-is-better legs (forty, cone, shuttle) price
    correctly once the pairs are sorted by mark value. Returns None when
    the position/leg has no recorded tier table."""
    marks = _KNOT_MARKS.get(pos, {}).get(leg)
    if not marks:
        return None
    pts = sorted(zip(marks, _KNOT_SCORES))
    if value <= pts[0][0]:
        return float(pts[0][1])
    if value >= pts[-1][0]:
        return float(pts[-1][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return float(y1)
            return y0 + (y1 - y0) * (value - x0) / (x1 - x0)
    return float(pts[-1][1])


def calc_athleticism_score(athleticism: Dict[str, Any], position: str) -> float:
    """
    Combine / pro-day metrics. Returns None when data is missing and
    score_prospect redistributes the weight. (Docstring corrected in
    v3.6, round-2 audit F12: it long promised a "positional median
    (55)" fallback the code never had.)

    RAS (Relative Athletic Score) is the most reliable single signal (0-10).
    Uses position-specific metric weights (e.g., 40-time matters more for WR,
    vertical for TE, speed score for RB).
    """
    if not athleticism:
        return None

    pos = position.upper()

    # Compute individual metric scores (each 0-100)
    metric_scores: Dict[str, float] = {}

    # RAS (0-10 → scale to 0-100)
    ras = athleticism.get("ras_score")
    if ras is not None:
        metric_scores["ras"] = _scale(_safe(ras), 4.0, 10.0)

    # 40-yard dash: knot-anchored to recorded drafted-tester tiers at the
    # position (v3.3). Faster = higher score via the tier pairing.
    forty_raw = athleticism.get("forty_yard")
    if forty_raw:
        _s = _knot_score(_safe(forty_raw), pos, "forty")
        if _s is not None:
            metric_scores["forty"] = _s

    # Vertical jump
    vert = athleticism.get("vertical_inches")
    if vert:
        _s = _knot_score(_safe(vert), pos, "vertical")
        if _s is not None:
            metric_scores["vertical"] = _s

    # Broad jump
    broad = athleticism.get("broad_jump_in")
    if broad:
        _s = _knot_score(_safe(broad), pos, "broad")
        if _s is not None:
            metric_scores["broad"] = _s

    # Speed score = weight * (40^4)^-1 * normalisation constant
    weight_lbs = athleticism.get("weight_lbs")
    if forty_raw and weight_lbs:
        ss_raw = (_safe(weight_lbs) * 200) / (_safe(forty_raw) ** 4)
        _s = _knot_score(ss_raw, pos, "speed")
        if _s is not None:
            metric_scores["speed_score"] = _s

    # Agility / change-of-direction (three_cone and short_shuttle).
    # Lower time = higher score. v3.3 F0: the old _scale calls passed
    # (hi, lo) inverted, which made _scale return a constant 50.0 for every
    # player; the knot tiers price these legs correctly.
    three_cone = athleticism.get("three_cone")
    if three_cone:
        _s = _knot_score(_safe(three_cone), pos, "cone")
        if _s is not None:
            metric_scores["three_cone"] = _s

    shuttle = athleticism.get("short_shuttle")
    if shuttle:
        _s = _knot_score(_safe(shuttle), pos, "shuttle")
        if _s is not None:
            metric_scores["shuttle"] = _s

    if not metric_scores:
        return None

    # Weighted average using position-specific weights
    pos_weights = _ATH_WEIGHTS.get(pos, {})
    if pos_weights:
        weighted_sum = 0.0
        total_weight = 0.0
        for metric, score in metric_scores.items():
            w = pos_weights.get(metric, 0.10)  # small default weight for unlisted metrics
            weighted_sum += score * w
            total_weight += w
        if total_weight > 0:
            raw = _clip(weighted_sum / total_weight)
        else:
            raw = _clip(sum(metric_scores.values()) / len(metric_scores))
    else:
        raw = _clip(sum(metric_scores.values()) / len(metric_scores))

    # Apply data completeness cap: a single outstanding metric should not yield
    # an elite athleticism score when the rest of the profile is unknown.
    n = len(metric_scores)
    cap = _ATH_DATA_CAPS.get(n, 100)   # 4+ metrics → no cap
    return _clip(raw, 0.0, float(cap))


def calc_utilization_score(seasons: List[Dict], position: str,
                           _inseason_year: Optional[int] = None,
                           _season_elapsed: Optional[int] = None) -> Optional[float]:
    """
    Volume of opportunity - how much of the team's offence ran through this player.

    Distinct from production (yards/TDs) and efficiency (per-touch quality):
    utilization measures raw opportunity share, which is highly predictive of
    early NFL role (a high-target WR translates to a high-target role).

    WR / TE: targets per game  (if targets unavailable, receptions per game as proxy)
    RB:      rush attempts per game + targets per game
    QB:      pass attempts per game (workload / usage signal)

    v3.6 (round-2 audit F2): the leg reads the latest season only, so
    when that season is in progress its score shrinks toward the
    completed-seasons read (the leg's neutral 50.0 when there are
    none) by min(1, GP/12), like production and efficiency.
    """
    if _inseason_year is not None and seasons:
        ls = _latest_season(seasons) or {}
        if ls.get("season") == _inseason_year:
            # F2 contract: the in-season raw is the CURRENT
            # season's own read, shrunk toward the completed
            # read. Round 6 step B deliberately does not blend
            # the raw here - the completed seasons already enter
            # through the shrink baseline (itself a step-B read),
            # and blending the raw too would price them twice.
            raw = _utilization_of_season(ls, position)
            if raw is None:
                return None
            _completed = [s for s in seasons
                          if s.get("season") != _inseason_year]
            _base = (_utilization_raw(_completed, position)
                     if _completed else 50.0)
            if _base is None:
                # v3.8: completed seasons carry no opportunity
                # evidence either - shrink toward the leg neutral.
                _base = 50.0
            return _clip(_shrink_partial(
                raw, _base, _safe(ls.get("games_played"), 12),
                elapsed=_season_elapsed))
    raw = _utilization_raw(seasons, position)
    if raw is None:
        return None
    return raw


# v3.11 round 6 (step B): utilization was the one heavy leg
# still reading the LATEST season alone, while production and
# (post-round-4) efficiency price best+latest. A hurt or split
# final season therefore printed a recorded 0.0 at full weight -
# Lacy's hurt 2026 (7.1 carries/g, under the scale's 8.0 floor,
# after a 306-carry 2025) and Hardy's corrected 2026 both read
# exactly 0.0 - and the mirror image maxed a compiler whose
# latest season was his workload peak, his committee seasons
# invisible to the leg (the Badie shape, S12.5). The leg now
# prices production's own form: hi_share on the higher of the
# latest-season read and the best-season read (best by the leg's
# own volume key, the latest season itself eligible - a player
# whose latest IS his workload peak reads exactly as before),
# the rest on the lower. Shipped at production's own 0.9/0.1 in
# round 6 (S14): the S12.5 sensitivity's Poor-tier cost (5.2 ->
# 6.9) ran through utilization's peak residency and does not
# materialize once step A removes that channel - Poor is flat
# at every swept dose (0.7/0.8/0.9), the gate cannot separate
# the doses, and 0.9 is the dose carrying the class evidence
# (Lacy's zero repaired, S12.5). Composition with the F2
# in-season shrink: the blend prices completed-profile reads
# and the shrink's completed-seasons baseline; the in-season
# raw stays the current season's own read (calc_utilization_
# score), so a completed season is never priced twice.
UTIL_BLEND_CFG = {
    "enabled": True,
    "hi_share": 0.9,
}

# v3.11 round 9 (discrepancy audit P2): the utilization leg's
# receptions-proxy top anchors sat at ~p93 (WR) / ~p96 (TE)
# of the recorded full-season distribution, so 118 of 1,034
# WR profiles and 16 TE profiles printed EXACTLY 100.0 - the
# leg stopped discriminating across its whole top decile,
# and short-season sprints (a 5-game 7.6 rec/g TE season)
# printed the same 100 as a full elite season. The *_hi
# anchors recalibrate to ~p98-99 of the recorded GP>=10
# distribution (targets anchors move proportionally for the
# live path, which the gate frame cannot price); the
# sample_denom shrink pulls a completed season's read
# toward the leg neutral (50) in proportion to its games
# when GP < sample_denom (0.0 = off). Swept in round9_lab.
UTIL_CFG = {
    "wr_rec_hi": 6.5,
    "wr_tgt_hi": 10.0,
    "te_rec_hi": 5.2,
    "te_tgt_hi": 8.0,
    "sample_denom": 0.0,
}


def _utilization_volume_key(season: Dict, pos: str) -> float:
    """The leg's own volume key for best-season selection: the
    counting stat the leg's scale is built on."""
    if pos == "RB":
        return _safe(season.get("rush_attempts"))
    if pos == "QB":
        return _safe(season.get("pass_attempts"))
    _t = season.get("targets")
    if _t is not None:
        return _safe(_t)
    return _safe(season.get("receptions"))


def _util_sample_shrink(read: Optional[float],
                        season: Dict) -> Optional[float]:
    """Round 9: a completed season's utilization read shrinks
    toward the leg neutral (50) in proportion to its games
    when UTIL_CFG sample_denom is positive and GP < denom.
    Applied on the completed-profile path (_utilization_raw)
    only; the in-progress season's read keeps its own F2
    treatment in calc_utilization_score."""
    if read is None:
        return None
    _den = float(UTIL_CFG.get("sample_denom", 0.0))
    if _den <= 0:
        return read
    _gp = _safe(season.get("games_played"), 0)
    if _gp <= 0 or _gp >= _den:
        return read
    return _clip(50.0 + (float(read) - 50.0) * (_gp / _den))


def _utilization_raw(seasons: List[Dict], position: str) -> Optional[float]:
    if not seasons:
        return 50.0

    pos = position.upper()
    ls = _latest_season(seasons) or {}
    latest = _util_sample_shrink(_utilization_of_season(ls, position), ls)
    if latest is None:
        # The leg's missing-evidence rule keys on the latest
        # season (v3.8 M1): no recorded opportunity there means
        # the leg is missing, however the earlier seasons read.
        return None
    if not UTIL_BLEND_CFG.get("enabled"):
        return latest
    _best_read = None
    _best_key = None
    for _s in seasons:
        _r = _util_sample_shrink(_utilization_of_season(_s, position), _s)
        if _r is None:
            continue
        _k = _utilization_volume_key(_s, pos)
        if _best_key is None or _k > _best_key:
            _best_key, _best_read = _k, _r
    if _best_read is None or _best_read == latest:
        return latest
    _hi = float(UTIL_BLEND_CFG.get("hi_share", 0.9))
    return _clip(_hi * max(latest, _best_read)
                 + (1.0 - _hi) * min(latest, _best_read))


def _utilization_of_season(season: Dict, position: str) -> Optional[float]:
    pos = position.upper()
    ls = season

    # v3.8 (M1): this season records no opportunity stats at
    # all (placeholder / fragment row - the fields are absent, not
    # zero) - the season carries no utilization measurement.
    # Recorded zeros measure (see _has_recorded_volume).
    if pos in ("WR", "TE"):
        if ls.get("targets") is None and ls.get("receptions") is None:
            return None
    elif pos == "RB":
        if (ls.get("rush_attempts") is None
                and ls.get("targets") is None
                and ls.get("receptions") is None):
            return None
    elif pos == "QB":
        if ls.get("pass_attempts") is None:
            return None

    gp  = max(_safe(ls.get("games_played"), 12), 1)

    if pos == "WR":
        targets = _safe(ls.get("targets"))
        if targets > 0:
            # Elite WR: 9+ targets/game; strong: 6+; average: 4
            return _clip(_scale(targets / gp, 2.5,
                                float(UTIL_CFG["wr_tgt_hi"])))
        # v3.8 (Becker fix, post-gate): targets missing - receptions
        # are the proxy, read on a RECEPTIONS scale, not the targets
        # scale. The RB branch below has done this since v3.3 (rec
        # 0.5-3.5 vs targets 1.0-5.0); WR/TE never got the treatment,
        # so a proxy read fell under the targets floor and stored a
        # manufactured 0.0 (a 2.4 rec/g season read 0.0 against the
        # 2.5 targets floor). The rec scale is the targets scale
        # compressed by the typical catch rate (~0.65): a reception
        # count is a deflated target count, and the scale, not the
        # player, should absorb the deflation.
        rpg = _safe(ls.get("receptions")) / gp
        return _clip(_scale(rpg, 1.6, float(UTIL_CFG["wr_rec_hi"])))

    elif pos == "TE":
        targets = _safe(ls.get("targets"))
        if targets > 0:
            # Elite TE: 7+ targets/game; strong: 4+; average: 2.5
            return _clip(_scale(targets / gp, 1.5,
                                float(UTIL_CFG["te_tgt_hi"])))
        # Receptions proxy on a receptions scale (see WR above).
        rpg = _safe(ls.get("receptions")) / gp
        return _clip(_scale(rpg, 1.0, float(UTIL_CFG["te_rec_hi"])))

    elif pos == "RB":
        carries  = _safe(ls.get("rush_attempts")) / gp
        rec_tgts = _safe(ls.get("targets"))
        _tgt_present = rec_tgts > 0
        rec_pg   = (rec_tgts / gp) if _tgt_present else (_safe(ls.get("receptions")) / gp)
        # Combines rush volume + receiving involvement
        # Elite: 20+ carries + 4+ targets; average: 12 carries + 2 targets
        rush_util = _scale(carries, 8.0, 22.0)
        # v3.3 V-C: when targets are missing, receptions per game is read on
        # a receptions scale (0.5 to 3.5), not the targets scale (1.0 to
        # 5.0): 1.6 receptions/g is not 1.6 targets/g, and the old read
        # under-credited every back graded without target data.
        recv_util = _scale(rec_pg,  1.0,  5.0) if _tgt_present else _scale(rec_pg, 0.5, 3.5)
        # Dynasty: pass-catching usage is more stable and more valued than rush volume
        return _clip(rush_util * 0.55 + recv_util * 0.45)

    elif pos == "QB":
        att_pg = _safe(ls.get("pass_attempts")) / gp
        # Elite: 35+ attempts/game; average: 25; game-manager: 18
        return _clip(_scale(att_pg, 16.0, 40.0))

    return 50.0


def calc_recruiting_score(recruiting: Optional[Dict[str, Any]]) -> Optional[float]:
    """
    Recruiting-pedigree signal (model v2.0).

    247 Composite rating → 0-100.  Recruiting pedigree is one of the strongest
    known predictors of NFL success, largely orthogonal to college production
    (it measures talent before the college scheme touches it).

    v3.6 (round-2 audit F10): missing data returns None, not a flat
    50.0. Recruiting was the lone leg that imputed a value and kept
    its full weight: a complete unknown scored exactly what a known
    0.87-composite 3-star scores, and half the live board has no
    composite at all. score_prospect now redistributes the weight
    when this returns None, like every other missing leg. A KNOWN
    0.87 composite still maps to 50 - that is a measurement, not a
    missing value.
    """
    if not recruiting:
        return None
    rating = recruiting.get("composite_rating")
    if rating is not None:
        try:
            # Composite runs ~0.78 (2-star) to ~1.00 (elite 5-star).  The
            # median draft prospect (~0.87, high 3-star) maps to neutral 50,
            # so known-average pedigree never scores below unknown.
            return _clip(_scale(float(rating), 0.74, 1.00))
        except (TypeError, ValueError):
            pass
    stars = recruiting.get("stars")
    if stars is not None:
        try:
            # Aligned to the composite scale: 5 (typical 0.985) -> 94,
            # 4 (0.93) -> 73, 3 (0.87) -> 50, 2 (0.80) -> 23.
            return {5: 94.0, 4: 73.0, 3: 50.0, 2: 23.0}.get(int(stars), 50.0)
        except (TypeError, ValueError):
            pass
    return None


def calc_competition_score(
    seasons: List[Dict],
    team_context: Optional[Dict] = None,
) -> Optional[float]:
    """
    Conference quality + implied opponent strength.
    Recent seasons are weighted more heavily to reflect current competition level.
    Transfer players who upgraded conferences get credit for their most recent context.

    team_context (optional, model v2.0): {(school_lower, season): {sp_sos, ...}}.
    When SP+ strength-of-schedule is available for the latest season's team it
    is blended 50/50 with the conference-quality estimate - a direct measure
    of who they actually played rather than who shares their conference.

    v3.8 (under-rating audit M1, pool class P5): when NO schedule data
    exists at all - no season carries a conference, the team is not
    Notre Dame, and no SP+ SOS is on file - the leg returns None
    (missing; weight redistributes). The old path scored the
    unknown-conference default quality, which is how 163 pool rows
    came to sit on one identical competition value: a default wearing
    a measurement's clothes.
    """
    if not seasons:
        return None

    _has_schedule = any(
        (s.get("conference") or "").strip()
        or "notre dame" in (s.get("team", "") or "").lower()
        for s in seasons
    )
    if not _has_schedule and team_context:
        _lat = max(seasons, key=lambda s: s.get("season", 0))
        _key = ((_lat.get("team") or "").lower().strip(),
                _lat.get("season", 0))
        _has_schedule = (team_context.get(_key) or {}).get("sp_sos") is not None
    if not _has_schedule:
        return None

    # Sort seasons by year (most recent first)
    sorted_seasons = sorted(seasons, key=lambda s: s.get("season", 0), reverse=True)

    total_weight = 0.0
    weighted_quality = 0.0

    for i, season in enumerate(sorted_seasons):
        conf = season.get("conference", "")
        team = (season.get("team", "") or "").lower()

        # Notre Dame gets a flat quality bonus regardless of conference label
        if "notre dame" in team:
            quality = 0.94
        else:
            quality = _conf_quality(conf)

        # Recency weights: 1.0, 0.8, 0.6, 0.4, 0.2 for up to 5 seasons
        weight = max(0.2, 1.0 - i * 0.2)
        weighted_quality += quality * weight
        total_weight += weight

    avg_quality = weighted_quality / total_weight if total_weight > 0 else DEFAULT_CONF_QUALITY

    # Model v2.0: blend in SP+ SOS when available for the latest season's team.
    if team_context and sorted_seasons:
        latest = sorted_seasons[0]
        team_key = ((latest.get("team") or "").lower().strip(), latest.get("season", 0))
        ctx = team_context.get(team_key) or {}
        sos = ctx.get("sp_sos")
        if sos is not None:
            try:
                # SP+ SOS typically spans roughly -12 (cupcake) to +12 (gauntlet)
                sos_quality = _scale(float(sos), -12.0, 12.0) / 100.0
                avg_quality = avg_quality * 0.5 + sos_quality * 0.5
            except (TypeError, ValueError):
                pass

    # v3.6 set the top anchor to 1.03 (above the SEC table value), so a
    # career SEC player landed ~95 and a career Big Ten player ~90
    # instead of both collapsing to exactly 100. v3.9 re-anchors to
    # 1.00 (see _COMPETITION_SCALE_HI): the SEC career earns 100 and
    # the Big Ten career 94.55 - the separation v3.6 wanted, without
    # an unreachable top.
    return _clip(_scale(avg_quality, 0.45, _COMPETITION_SCALE_HI))


def calc_environment_adjustment(seasons: List[Dict], position: str) -> float:
    """
    Adjusts for team usage patterns.

    - WR/TE: yards-based pass share (team_pass_yards / team_total_yards).
      Captures how much of the offense's actual production came through the air,
      not just how often they called pass plays. Falls back to attempt-based
      team_pass_rate when yard totals are unavailable.
    - For RBs, high rush rate inflates volume - slight discount applied.
    - Uses a recency-weighted average across all seasons so transferred players
      aren't locked to their latest team's scheme.
    """
    if not seasons:
        return 50.0

    pos = position.upper()

    # v3.2 R7-A5: no environment inputs (or QB, whose environment the
    # function does not model) -> missing data, weight redistributes.
    _has_env = any(
        (_safe(s.get("team_pass_yards")) > 0 and _safe(s.get("team_total_yards")) > 0)
        or s.get("team_pass_rate") is not None
        for s in seasons
    )
    if not _has_env or pos == "QB":
        return None

    # Recency-weighted pass share: most recent season weight 1.0, decaying by 0.2
    sorted_seasons = sorted(seasons, key=lambda s: s.get("season", 0), reverse=True)
    weighted_sum = 0.0
    total_weight = 0.0
    for i, s in enumerate(sorted_seasons):
        team_pass_yds  = _safe(s.get("team_pass_yards"))
        team_total_yds = _safe(s.get("team_total_yards"))
        if team_pass_yds > 0 and team_total_yds > 0:
            pr = team_pass_yds / team_total_yds   # yards-based pass share
        else:
            pr = _safe(s.get("team_pass_rate"), 0.55)  # fallback: attempt-based rate
        w  = max(0.2, 1.0 - i * 0.2)
        weighted_sum += pr * w
        total_weight += w
    pass_rate = weighted_sum / total_weight if total_weight > 0 else 0.55

    if pos in ("WR", "TE"):
        # Yards-based pass share: 40% = run-heavy floor, 72% = Air Raid ceiling
        base = _scale(pass_rate, 0.40, 0.72)
    elif pos == "RB":
        # RBs benefit from balanced / run-heavy but it's easier to produce in bad offences
        # Slight penalty for very pass-heavy (≥65%) teams since it reduces rush attempts
        if pass_rate > 0.62:
            base = _scale(1.0 - pass_rate, 0.30, 0.55)
        else:
            base = _scale(pass_rate, 0.35, 0.62) * 0.5 + 50 * 0.5
    else:
        base = 50.0

    return _clip(base)


def _season_has_role(season: Dict, position: Optional[str]) -> bool:
    """
    v3.4: did the player hold a real role in this season?

    Stored games_played counts APPEARANCES, so a backup's 2-game season
    and an injured starter's 2-game season look identical on GP alone.
    Usage separates them: a season only counts as missed time when the
    player was actually in the rotation. Thresholds sit in the empty
    bands of the recorded distributions (panel + fixtures):
    - QB pass attempts >= 75: recorded backup seasons top out at 41
      (Mestemaker 2024; Sayin 2024 is 12, Moore 2024 is 8); the lightest
      recorded starter season is 99 (Moore's 4-game 2026). ~3 games of
      starting volume.
    - RB touches (carries + receptions) >= 60: recorded mop-up/rotation
      partials top out at 43 (Lacy 2026); recorded starter partials run
      110+ (Baugh 2026: 117). A starter lost at midseason runs ~90-120.
    - WR targets >= 20 or receptions >= 12; TE targets >= 15 or
      receptions >= 10: recorded non-role seasons top out at 9
      receptions (Evans 2024, Reynolds 2024); the lightest recorded role
      seasons are 13 (Green 2024). Targets are OR-ed with receptions
      because the targets feed is a separate step that can lag.
    With no position given, any position's test passing is enough (the
    usage fields are position-specific in practice).
    """
    att = _safe(season.get("pass_attempts"))
    touches = _safe(season.get("rush_attempts")) + _safe(season.get("receptions"))
    tgt = season.get("targets")
    tgt = _safe(tgt) if tgt is not None else 0.0
    rec = _safe(season.get("receptions"))
    pos = (position or "").upper()
    if pos == "QB":
        return att >= 75
    if pos == "RB":
        return touches >= 60
    if pos == "WR":
        return tgt >= 20 or rec >= 12
    if pos == "TE":
        return tgt >= 15 or rec >= 10
    return att >= 75 or touches >= 60 or tgt >= 20 or rec >= 12


def calc_durability_score(seasons: List[Dict],
                          position: Optional[str] = None,
                          _inseason_year: Optional[int] = None,
                          _today=None) -> Optional[float]:
    """
    Durability: how often the player is available when the team plays.

    Core principle: games_played / games_available.
    - 12+ games in a completed season = full season (100%). The extra games
      beyond 12 are just playoff runs, not extra durability credit.
    - In-progress season: compare against current week. If the player has
      played every game so far, that's 100% - don't discount for the season
      being unfinished.

    Sophistication beyond raw percentage:
    - Recency weighting: recent seasons predict future availability better
    - Missed-games penalty: a completed season under 9 games suggests injury
      risk. Scaled: 8 games = small ding, 4 games = major red flag.
    - In-progress seasons never trigger penalties, only contribute positively
      to the average if the player has been available.

    v3.4 role rule (both modes): a season under the 12-game full-season
    bar counts only when the player held a real role in it (see
    _season_has_role). Backup and mop-up seasons are excluded entirely,
    neither as availability nor as missed time; before this, Sayin's
    4-game backup season priced 44.07 and Mestemaker's 2-game backup
    season priced 6.01. A 12+ game season is availability-proven by
    itself and always counts, whatever the usage (a blocking TE who
    dressed 16 times was available 16 times). If no counted season has
    GP data, durability is missing and its weight redistributes. (A
    season-ending injury in a player's first 1-2 starts is
    indistinguishable from backup work in stored data and is forgiven
    with it; that is the accepted cost of the rule.)

    v3.4 in-season variant: the in-progress season is NOT blanket-
    excluded. It counts when it is a role season, with absences measured
    against a bye-adjusted calendar floor (current week minus 2: FBS
    teams carry at most 2 byes, so a shortfall beyond that is certainly
    real missed time, while a starter who has played every team game is
    never charged for byes). Non-role partial seasons stay excluded,
    like any non-role season. The partial season never triggers the
    completed-season missed-games penalty.

    Returns None if no GP data (weight redistributed, like athleticism).

    ``_today`` pins the calendar for tests and replays (the same override
    prospect_model_v4 exposes as ``as_of``); production leaves it None and
    reads the real date. The in-progress denominator is the current week,
    so an unpinned caller's result legitimately moves as weeks pass.
    """
    if not seasons:
        return None

    from datetime import date as _d
    today = _today or _d.today()
    current_year = today.year
    season_start = _d(current_year, 8, 25)
    current_week = 1 if today < season_start else min((today - season_start).days // 7 + 1, 16)

    FULL_SEASON_GAMES = 12  # 12+ = full season, no bonus for playoff extras

    sorted_s = sorted(seasons, key=lambda s: s.get("season", 0))
    scored = []  # (pct, is_completed, gp)

    for s in sorted_s:
        gp = s.get("games_played")
        if gp is None:
            continue
        gp = _safe(gp)
        if gp < FULL_SEASON_GAMES and not _season_has_role(s, position):
            continue
        season_year = s.get("season", 0)
        is_partial = (_inseason_year is not None
                      and season_year == _inseason_year)

        if season_year == current_year:
            # In-progress: available = weeks played so far. In-season
            # mode measures absences against the bye-adjusted floor so
            # a healthy starter is not charged for his team's byes.
            if is_partial:
                denom = max(current_week - 2, 1)
            else:
                denom = current_week
            pct = min(gp / denom, 1.0) if denom > 0 else 1.0
            scored.append((pct, False, gp))
        elif season_year < current_year:
            # Completed: 12+ games = 100%, scale below that
            pct = min(gp / FULL_SEASON_GAMES, 1.0)
            scored.append((pct, not is_partial, gp))

    if not scored:
        return None

    # Weighted average: recent seasons count more
    n = len(scored)
    weighted_sum = 0.0
    total_weight = 0.0
    for rank, (pct, _, _) in enumerate(scored):
        w = 0.5 + 0.5 * (rank / max(n - 1, 1))  # 0.5 to 1.0
        weighted_sum += pct * w * 100.0
        total_weight += w
    base = weighted_sum / total_weight

    # Penalty for missed games in completed seasons only.
    # 9+ games: no penalty. Below that, increasingly harsh.
    # 8 games -> -5, 6 games -> -15, 4 games -> -30, 2 games -> -45
    penalty = 0.0
    for pct, is_completed, gp in scored:
        if not is_completed:
            continue
        if gp < 9:
            # Quadratic penalty: missing more games hurts disproportionately
            missed = 9 - gp
            penalty = max(penalty, missed * missed * 1.2)

    return _clip(base - penalty)



# ─────────────────────────────────────────────────────────────────────────────
# Fantasy translation bonus
# ─────────────────────────────────────────────────────────────────────────────

# Position-level adjustment reflecting long-run fantasy value scarcity.
# TE premium is negative because TE translation to fantasy is hardest.
POSITION_FANTASY_MULT: Dict[str, float] = {
    "WR": 1.00,
    "RB": 1.00,
    "QB": 0.90,   # QBs are less valued in 1QB dynasty
    "TE": 0.95,   # Increased from 0.85 - elite TEs should be much closer to WR level
}
POSITION_FANTASY_MULT_SF: Dict[str, float] = {
    "WR": 1.00,
    "RB": 0.98,
    "QB": 1.20,   # QBs get big SF premium
    "TE": 0.90,
}


# ─────────────────────────────────────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────────────────────────────────────

# Position-specific blend weights, stored in EFFECTIVE form: these
# are the weights score_prospect prices after its capital pop - what
# a prospect's grade is blended from when every leg is present.
#
# Provenance (F12 fitted weights, folded in 2026-11-24; model 3.7
# had not shipped, so the version does not bump). Per position, each
# leg's Pearson against cumulative NFL PPR production was measured
# over the dumped 2016-2025 gate population (gate_rows_v3.json,
# 1,945 scored profiles), |r|-normalized, and blended 50/50 with the
# prior design weights (~/workspace/prospect-runs/
# fit_weights_offline.py -> fitted_weights.json). The blend was
# validated out of sample twice: leave-one-class-out CV, then a
# full backtest gate run with the set injected as
# position_weights_override (~/workspace/prospect-runs/
# gate_v37_weights.log), which cleared 3.5's talent-only bar at
# every position - QB .393 vs bar .361, RB .411 vs .397, WR .369
# vs .332, TE .407 vs .365. The set is baked in here so the shipped
# model reproduces the gated grades with NO override;
# tests/test_prospect_v37.py pins the effective weights and the
# override/table equivalence over the gate population.
#
# Two structural notes. (1) There is no draft_capital entry: the fit
# is capital-free by construction, and score_prospect's unconditional
# capital pop (which spreads a foreign override's capital entry over
# the talent legs) is a no-op on these tables. (2) Recruiting is a
# fitted per-position leg like any other (~0.032-0.035). The v2.0
# import-time carve (scale every leg by 0.95, then set recruiting
# to a flat 0.05) is retired: it operated on pre-fit design tables,
# and applying it to the fitted set would move every weight off the
# gated values.
# v3.9 (ceiling redesign): the final blend. Through 3.8 the grade was
# a plain weighted MEAN of the priced legs, and the mean is a hard
# ceiling: a generational profile whose legs read 100/100/97/94/93
# averages to ~93 no matter how the weights are fitted, so the best
# prospect of a decade printed 92-94 while the scale promises the
# high 90s are attainable. The blend keeps the weighted mean as its
# base and adds ONE peak term: a share of the grade is priced on the
# prospect's best legs instead of his average leg, so a profile that
# is elite at what it does best separates from a profile that is
# merely good everywhere.
#
#   grade = (1 - alpha) * weighted_mean(all priced legs)
#         + alpha * weighted_mean(top `peak_k` priced legs)
#
# peak_k counts legs, ranked by score; their weights renormalize
# inside the peak term. Two guards keep the peak honest:
# - Production is the price of admission: the peak set always
#   carries the production leg, so a profile with no production (a
#   recorded 0.0 next to one spike leg - the under-rating audit's
#   P1 hard-zero class) cannot be priced on its spikes alone. In
#   the offline sweep, unanchored peak emphasis lifted that class
#   up to +16 points; anchored, its mean lift (+1.2) sits BELOW the
#   healthy pool's (+3.3).
# - A profile under the multi-None anchor (no measured performance
#   core, v3.8) earns no peak term at all: its grade is the anchored
#   mean, because context legs alone are not a peak.
# - QBs take a smaller peak share (0.30 vs 0.60): college QB volume
#   is the least portable signal in the model - a pass-happy offense
#   hands a bust three 100 legs - and at the full share the
#   Poor-outcome QB tier's >=70 rate tripled (10% -> 30%); at 0.30
#   it is unchanged from the mean blend while the QB gate still
#   improves.
#
# Nothing here rescales the output: the same formula prices every
# profile, the 99.9 clamp still binds, and 100 stays impossible
# (the peak term is itself a weighted mean of legs <= 100, blended
# with a mean below it for any real profile). Offline gate vs 3.8
# (gate_rows_v3 substrate): QB +.004/+.003, RB +.005/+.005,
# WR -.000/+.001, TE +.007/-.009 - the TE Spearman whisker is the
# one documented exception (within noise at n=196; TE profiles
# price few legs, so a 4-leg peak reorders their middle; TE
# Pearson improves and the TE tier table is undamaged). A
# blend-aware re-fit of the position weights was run and REJECTED:
# leave-one-class-out CV scored the fitted set at or below the
# shipped F12 weights at every position under this blend.
#
# v3.11 (ceiling attainability - Kaedon 2026-11-24: "no one can
# get a 99 right now", and "a prospect can be generational while
# checking some boxes and leaving some unchecked"). The 3.10
# audit (ceiling-attainability-2026-11-24.md) showed leg ceilings
# are mostly NOT the bind - production, utilization, efficiency,
# age and competition all print 100 somewhere in the recorded
# population; the bind is co-occurrence taxed by the mean term:
# no real profile holds the maxima simultaneously, and the 40%
# mean share charged a generational profile full price for its
# one mediocre box. Two changes, both inside the blend:
# (1) the peak share rises 0.60 -> 0.74 (QB keeps its 0.30 guard
# share - raising it re-opens the QB Poor-tier wound for +0.3 on
# the best QB, measured); (2) the TRIMMED MEAN: a profile that
# has earned forgiveness drops its single weakest substantive
# box from the mean term. Forgiveness is gated on demonstrated
# spiky greatness, all four conditions absolute and uniform:
# the profile prices >= 5 legs; its best leg is >= 90 (an elite
# box exists); its production-anchored peak set prices >= 88
# (the elite boxes are plural - one spike leg in a pass-happy
# offense earns nothing); and the dropped leg is a genuinely
# weak box (value <= 75) carrying real weight (>= 0.02 - sub-2%
# legs are context adjustments, not boxes). Production is never
# droppable (price of admission, as in the peak set), and the
# trim never applies under the multi-None anchor. The gates are
# what make the trim safe: ungated, it floated fragments
# (+0.55 mean / +5.51 max) and lifted the Anderson case +2.2;
# gated, fragment deltas match the alpha-only lift, Anderson is
# byte-identical at 71.62, and the QB/WR Poor-tier >=70 rates
# are flat. Offline gate vs 3.10: Pearson IMPROVES at QB/RB/WR
# (+.013/+.002/+.001), TE -.002; Spearman within -.005
# everywhere; median R2-4 +0.75. The stored ceiling moves where
# the evidence said it honestly could (Chase 97.84, Nabers
# 96.41, Stroud 94.96) and does not move where it could not
# (best stored RB 92.32 - no stored back pairs a >=95 peak with
# a >=85 mean, and the only blends that print one price three
# legs and nothing else; see the note's detonation run).
BLEND_CFG = {
    "peak_alpha": 0.74,
    "peak_alpha_by_pos": {"QB": 0.30},
    "peak_k": 4,
    # v3.11 round 6 (step A): peak-set ELIGIBILITY. The peak set is
    # chosen by value rank, weight-blind, so a leg whose value is a
    # workload/share read gets paid TWICE at the 0.74 peak share -
    # once in the mean, once in the peak - while the rate legs that
    # check it sit at residual weight (the Tyler Badie anatomy,
    # S12: production + utilization at .76 combined effective
    # weight, his efficiency at .046). Legs named here keep their
    # full MEAN weight; they simply cannot occupy a peak slot.
    # Utilization is a share of team opportunity - how much
    # football ran through the player, not how well he played it -
    # and production already carries the volume evidence into the
    # peak set as the price of admission. Shipped in round 6 on
    # the S14 sweep: the gate improves at QB and RB, WR/TE are
    # byte-identical (utilization never reaches their peak sets),
    # and the compiler's double-pay closes (Badie 93.95 -> 87.33).
    "peak_exclude": ("utilization",),
    # v3.11 round 8: per-position additions to peak_exclude.
    # The peak set is chosen by value rank, so a leg that is
    # high BY CONSTRUCTION for most of a position's pool
    # occupies its slots mechanically and evicts the measured
    # performance legs the peak term exists to price. At QB
    # the context legs saturate: experience (pool median 92.5,
    # a third of profiles at the 100 cap, outcome r ~ +.09)
    # is peak-resident in 90.6% of priced QB blends and
    # competition in 89.7% (round-8 residency census), while
    # competition's real QB work is already done by the
    # discount on the output legs. Both keep their full MEAN
    # weight here; they simply cannot occupy a QB peak slot.
    # SWEPT AND NOT SHIPPED (round 8, S16.3): excluding
    # experience at QB gains +.004 P / +.008 S with Poor
    # flat, but it taxes the dual-threat family the round
    # exists to lift (Allen -1.20, Lamar -3.58 - their
    # experience 100s were peak-resident and propping them)
    # and Herbert -1.02; excluding both legs costs Pearson
    # (-.005) for Spearman (+.013) and taxes Allen -3.06 /
    # Lamar -4.10. The residency is real, but the binding
    # problem is the experience leg's CONTENT (a saturating
    # bounty, S11.4), not its peak address; the exclusion
    # stays available behind this empty default as
    # Kaedon's ruling.
    "peak_exclude_by_pos": {},
    # v3.12 (RB-2b hygiene, peak-composition commission S7.4): a
    # peak slot must carry at least this much weight. The peak set
    # is chosen by value rank, weight-blind, so a leg priced at a
    # token weight (RB durability, ~0.006-0.007) holds a top-4
    # slot in 70.7% of RB profiles and renormalizes the peak term
    # around a value its own weight says is decoration. The floor
    # is the trim's own minimum weight (0.02): a leg too light to
    # be forgiven is too light to anchor the peak. RB-scoped -
    # the census that proved the defect is an RB census, and the
    # other positions' peak sets are untouched (byte-identical).
    # Measured on gate_rows_v8 (RB-2b + p40 base): RB gate
    # .4567/.4531 -> .4682/.4651, Poor >=70 6.9 -> 1.7, fixtures
    # byte-exact. This is hygiene, NOT the Mixon fix: the
    # commission killed every composition mechanism as a Mixon
    # lever (design note S7); Mixon is a named residual.
    # v3.13 (Phase 7, lever L6): the same floor extends to TE.
    # The census is the same defect: 245 of 446 TE profiles
    # hold durability (redistributed weight ~0.0088) in their
    # peak set - a featherweight leg anchoring the peak term.
    # It is the largest single TE gate improvement measured in
    # the residuals lab (P .3970 -> .4133, S .3719 -> .3855
    # standalone, Poor flat). See prospect-runs/
    # bowers-mccaffrey-residuals-2026-11-25.md (L6).
    "peak_min_weight_by_pos": {"RB": 0.02, "TE": 0.02},
    # v3.11 round 6 (step C2): production's admission to the peak
    # set - SWEPT AND KEPT AT "always" (S14). The alternatives
    # ("never", and "margin" at 5/10/15/20 - production admitted
    # only when within the margin of the leg it would evict, the
    # Bowers S13.1 shape) all fail the same way: at a 0.74 peak
    # share the forced slot is not a niche anti-spike guard, it
    # is the scale's level control for the whole pool - median
    # R2-4 rises +6.7 to +7.8, the Poor-tier >=70 rates explode
    # (TE 2.6 -> 25.6-33.3, RB 5.2 -> 6.9-17.2, WR 12.9 ->
    # 17.6-18.8), RB/WR Pearson fall .02-.04, and fragments are
    # crowned (max +25.5). Bowers' eviction cost (+4.2 under
    # "never") is real, and it is the price of the level control;
    # his round-6 lift comes from the production anchor (step
    # C1) instead. The trim gate prices the production-ANCHORED
    # set by definition (trim_min_peak), whatever this mode says.
    "peak_force": "always",
    "peak_force_margin": 10.0,
    # Trimmed mean (v3.11, see above): drop the single lowest
    # priced leg from the MEAN term when the profile has earned
    # it. The peak term always prices the full leg set.
    "trim_lowest": True,
    # Competition stays trimmable. Excluding it was tested in
    # the 3.11 revision (the Fannin case): his weakest box by
    # value is competition (60 at 4.3% weight), but by weighted
    # drag it is athleticism (61.8 at 13.1%) - barring the
    # competition drop just redirected his forgiveness to the
    # heavier leg and RAISED him (94.25 -> 95.13). The level
    # price belongs in the discount, not in trim bookkeeping.
    "trim_exclude": ("production",),
    "trim_min_legs": 5,
    "trim_min_top": 90.0,
    "trim_min_weight": 0.02,
    "trim_drop_ceiling": 75.0,
    # v3.11 round 7: forgiveness has a depth limit. The trim
    # drops a WEAK box (40-75: a mediocre combine, a small-
    # school competition print, one down rate season) - it
    # was built for a spiky profile's softest strength. A leg
    # under 40 is not a weak box, it is a failed trait, and
    # deleting it from the mean is erasure, not forgiveness:
    # the census of shipped-code drops is 75 pool-wide, 15 of
    # them under 40, and the tail is where the doctrine
    # breaks - a 24-year-old draftee's age leg (19.4) dropped
    # whole, so the oldest prospect on the board paid no age
    # price at all in either blend term. Legs under the floor
    # are simply ineligible for the drop; the next-weakest
    # eligible leg takes it instead. Gate in isolation (S15):
    # RB byte-identical, QB improves (+.009 P / +.015 S - the
    # forgiven sub-40 QB craters were noise), WR/TE flat,
    # and no settled name moves (Engram keeps his 67.2 age
    # drop; Fannin's 60 competition drop stands).
    "trim_drop_floor": 40.0,
    "trim_min_peak": 88.0,
}


# v3.9 competition discount (Kaedon 2026-11-24: "competition
# should matter more" - the case was a Tarleton State (FCS) QB
# priced #9 in his class on a 93.9 production leg). The production
# and efficiency legs price raw per-game output on absolute scales,
# so a gaudy line against FCS defenses entered the blend at face
# value while the additive competition leg (~4-10% of the weight,
# and droppable from the peak set entirely) could not pull it
# back. The discount prices the two absolute-output legs against
# the level they were achieved at:
#   factor = 1 - 0.35 * max(0, (65 - competition) / 65)
# applied to the leg VALUES before weights, redistribution or
# blending - the discounted values are the legs (payload,
# percentiles and blend all read the same numbers). It is a slope,
# not a cliff: competition at or above 65 (the G5/P4 boundary on
# this leg's scale) pays nothing, and the price deepens linearly
# to a 35% maximum at the scale's floor. Utilization is EXEMPT: a
# share of team opportunity is a share at any level, and exempting
# it preserved the small-school signal (the Cooper Kupp shape)
# at no cost to the mechanism's effect on the case that prompted
# it. Offline gate vs 3.8 (with the breakout arrival term): the
# discount IMPROVES Pearson at all four positions (+.011 to
# +.017) and Spearman at RB/WR; QB and TE Spearman move -.007
# (within noise at n=106/196, mid-tier shuffles, Pearson improves
# at both). The small-school cohort's mean grade moves -2.0 with
# its within-cohort outcome correlation IMPROVING (.304 -> .316):
# on the drafted record, level-inflated volume was noise, not
# signal. The peak-guard alternative (shrinking the peak share
# for low-competition profiles) was tested and rejected: it moved
# the case player less than a point while taxing QB Spearman
# twice as much. A plain competition-weight doubling was also
# rejected (it lifts big-school busts - QB Poor-tier >=70 rate
# 19% -> 29% - and floats fragments, max +10.8).
COMPETITION_CFG = {
    "prod_discount": 0.35,
    "discount_floor": 65.0,
    "discount_utilization": False,
    # v3.11 round 10 (pattern-5 sub-item): the competition
    # toll currently lands on production AND efficiency.
    # Setting this False prices the toll once (production
    # only) - swept in round10_lab, default keeps the 3.9
    # both-legs ship.
    "discount_efficiency": True,
    # v3.11 revision: the depth curve bends in the boundary
    # band only. The linear depth spent almost nothing just
    # under the boundary (a MAC line at competition 60 paid
    # 2.7%, so a record small-school season priced nearly
    # whole - the Harold Fannin Jr. case, TE max of the pool)
    # while the deep tail paid the most. Power < 1 re-prices
    # the band [discount_band, discount_floor] on a concave
    # curve CONTINUOUS with the original linear depth at the
    # band's lower edge: below the band nothing changes at
    # all (the Anderson case at 47 and the Kupp shape at 49
    # price byte-identically, and the small-school cohort's
    # outcome correlation is unmoved), and no level is ever
    # refunded - a whole-curve power bend was tested first
    # and rejected because it pays for the boundary band by
    # refunding the deep tail, which landed on exactly the
    # small-school busts the discount exists to price (two
    # new Poor-tier crossers). The toll belongs at the
    # boundary: raw counting stats just below the P4/G5 line
    # are the ones nearly indistinguishable from P4 stats.
    # At the shipped 0.3, competition 60 pays ~5.8% (was
    # 2.7%) and competition 55 pays ~7.2% (was 5.4%).
    "discount_band": 50.0,
    "discount_power": 0.3,
}


# v3.11 round 2: the trimmed mean forgives ABSENCE, not measured
# weakness (see the call site in score_prospect). When a profile
# carries at least one role (featured) season, the listed legs
# were priced from full-evidence featured seasons, so a low
# value is a measurement and the leg is blocked from the trim
# drop for that profile. The block scopes to UTILIZATION only,
# on evidence: utilization is the one performance leg that
# aggregates the whole featured body of work (a share of team
# opportunity across every role season), so its low value is
# always a full-evidence measurement - the split backfield is
# the case. Efficiency prices the latest season alone (a
# single-season read: a down year can be injury or scheme, not
# a verdict on the body of work) and breakout is a change
# measure whose low value often means "arrived early", the
# absence of a rise rather than measured weakness. Blocking
# those two as well was swept (round-2 lab A2): it moved
# Justin Jefferson -1.29 and Rondale Moore -3.09 with no
# face-validity gain, disturbing the WR grading Kaedon ruled
# good. Utilization-only (A1) changes exactly five RB drops
# pool-wide and no QB/WR/TE grade at all.
TRIM_MEASURED_CFG = {
    "enabled": True,
    "legs": ("utilization",),
}


# v3.11 round 2: production level ceiling. The competition
# discount (above) tolls the production leg by a percentage, but
# a percentage of a level-inflated spike is still a spike: an FCS
# back whose raw production prints ~97 keeps ~89 after the ~9%
# toll at competition 49 and tops his class (the Eli Gillman
# case: Montana, competition 49.09, the 2027 class RB1 at 79.93).
# Below the ceiling's competition band the production leg's
# PRINTABLE value is capped outright, the cap rising linearly
# from cap_floor at comp_lo to an effective 100 at comp_hi:
#   cap(comp) = cap_floor + (comp - comp_lo) / (comp_hi - comp_lo)
#               * (100 - cap_floor)     for comp_lo < comp < comp_hi
#   cap(comp) = cap_floor               for comp <= comp_lo
#   no cap                              for comp >= comp_hi
# The cap lands BEFORE the discount, so the toll still prices
# the capped value. Scoped to RB: at competition ~49 the Kupp
# fixture's raw production (~96.5) is indistinguishable from
# Gillman's (~97) - a level-only ceiling cannot separate them,
# and the recorded evidence says small-school WR production
# translates (the Kupp shape is the model's own guardrail)
# while lead-back FCS rushing volume is the demonstrated
# failure. The ceiling's dose is set so the small-school
# cohort's outcome correlation - the arbiter for all level
# pricing in this model - does not degrade.
PROD_CEILING_CFG = {
    "enabled": True,
    "positions": ("RB",),
    "cap_floor": 82.0,
    "comp_lo": 45.0,
    "comp_hi": 60.0,
}


def _production_level_cap(competition: float) -> Optional[float]:
    """The PROD_CEILING_CFG cap for a competition leg value, or
    None when the level prices production at face value."""
    if competition is None:
        return None
    _lo = float(PROD_CEILING_CFG.get("comp_lo", 45.0))
    _hi = float(PROD_CEILING_CFG.get("comp_hi", 60.0))
    _cap0 = float(PROD_CEILING_CFG.get("cap_floor", 82.0))
    if competition >= _hi:
        return None
    if competition <= _lo or _hi <= _lo:
        return _cap0
    return _cap0 + (competition - _lo) / (_hi - _lo) * (100.0 - _cap0)


# ------------------------------------------------------------------
# v3.12 RB-2b (Kaedon 2026-11-25: "the rb grading needs
# reimagined because all other pos are good"). RB-scoped changes
# hanging off the shared pipeline; QB/WR/TE are byte-identical by
# construction (every change below keys on pos == "RB" or an
# RB-only cfg entry). Design + prototype evidence:
# prospect-runs/rb-reimagination-design-2026-11-25.md (S1-S7).
# Change 1 (corroboration-priced release) lives in
# PROD_EVIDENCE_CFG; change 6 (peak min-weight) in BLEND_CFG.
# The blend machinery itself is untouched (peak share 0.74,
# gated trim, forced production slot - the level machinery two
# rounds proved load-bearing).
#
# Change 3: athleticism re-anchored to the RB population. The
# shipped scale compresses the combine's information (Barkley's
# 4.40 / 41-inch print priced 73.49, a mid-pack value for the
# best athlete in the pool). At RB the leg's value becomes its
# percentile among priced RBs - the display-percentile doctrine
# turned inward - mapped 5 + 95 x pct, at the same 0.0936
# weight. The pool is the FROZEN recorded population below
# (2016-2024 drafted RBs, shipped-scale values, n = 159): a
# pinned constant table, never recomputed per class, so a live
# grade cannot drift with the class being graded.
RB_ATH_POOL = (
    18.47, 22.0, 22.8, 27.22, 27.54, 28.0, 28.79, 29.06,
    29.23, 30.52, 30.7, 30.76, 30.86, 31.33, 32.71, 32.74,
    34.32, 34.33, 34.41, 34.63, 35.89, 36.17, 37.58, 39.2,
    39.6, 40.52, 40.57, 40.89, 40.91, 41.09, 41.35, 41.57,
    41.79, 41.9, 42.12, 42.63, 42.66, 43.14, 43.14, 43.14,
    43.69, 44.9, 45.39, 45.5, 45.63, 45.71, 45.95, 46.1,
    46.61, 46.65, 47.09, 47.45, 47.75, 49.09, 49.16, 49.49,
    49.96, 50.68, 51.96, 52.33, 52.36, 52.43, 53.2, 53.65,
    54.06, 54.74, 55.17, 55.62, 55.66, 56.86, 57.01, 57.12,
    57.35, 57.7, 57.91, 58.03, 58.19, 58.2, 58.3, 58.37,
    59.0, 59.16, 59.54, 59.6, 59.73, 59.95, 60.1, 60.18,
    60.39, 60.55, 60.97, 61.05, 61.34, 61.68, 62.1, 62.1,
    62.73, 62.8, 62.98, 63.73, 64.31, 64.31, 64.57, 64.59,
    64.61, 64.69, 64.85, 64.85, 65.04, 65.54, 65.95, 66.01,
    66.88, 67.5, 67.58, 67.74, 67.89, 67.94, 67.94, 68.01,
    68.12, 68.42, 68.66, 68.68, 68.75, 69.03, 69.08, 69.31,
    69.8, 70.16, 70.41, 70.6, 70.86, 71.46, 71.5, 71.79,
    72.0, 72.29, 72.82, 73.43, 73.49, 73.64, 73.64, 73.75,
    74.28, 74.4, 74.92, 75.4, 75.4, 75.83, 75.93, 76.93,
    78.18, 78.52, 78.98, 81.01, 81.06, 82.25, 83.47,
)


def _rb_ath_reanchor(value):
    """Map a shipped-scale RB athleticism value to its frozen-
    pool percentile on the 5-100 re-anchored scale."""
    import bisect as _bisect
    pct = _bisect.bisect_right(RB_ATH_POOL, float(value)) / len(RB_ATH_POOL)
    return 5.0 + 95.0 * pct


def _pool_percentile(sorted_vals, q):
    """Linear-interpolation percentile (numpy default method)."""
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    rank = (float(q) / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    return (sorted_vals[lo] * (1.0 - (rank - lo))
            + sorted_vals[hi] * (rank - lo))


# Change 5: a missing RB athleticism leg prices at the pool's
# 40th percentile ON THE RE-ANCHORED SCALE (43.36) instead of
# redistributing its weight - missing must never price better
# than a tested print. Hygiene, not a fix: the Mixon lab
# (design note S6) proved the redistribution premium is worth
# ~0.9 points on the named case and killed the theory that it
# drives his grade; the fill ships because a free leg is wrong
# in principle and the gate improves slightly with it.
RB_ATH_FILL_P40 = _pool_percentile(
    sorted(_rb_ath_reanchor(v) for v in RB_ATH_POOL), 40.0)

# Change 2: receiving as a direct leg, weight 0.08, funded by
# utilization -0.04 / age -0.02 / breakout -0.02 (the legs the
# evidence indicts: workload double-pay and the age amplifier).
# The leg prices 0.5 x best featured-season receiving yards per
# game (scale 4-42) + 0.5 x career receiving share of
# all-purpose yards over featured seasons (scale 0.04-0.32),
# from the counting stats the pipeline already stores.
# v3.13 (Phase 6, R4): the mix is reweighted 0.65 rate /
# 0.35 share. The share term's denominator is the back's own
# all-purpose yards, so at 50/50 it taxed the dual-threat
# shape for also being a high-volume rusher (McCaffrey's leg
# read 84.85: a saturated 100 rate term halved by a 69.7
# share term whose denominator was his own rushing). The
# rate term is the purer receiving-skill signal; the share
# term double-counts rushing volume already priced in
# production and utilization. Gate-clean in the residuals
# lab (P +0.0000 / S -0.0019, Poor flat). See prospect-runs/
# bowers-mccaffrey-residuals-2026-11-25.md (R4).
#
# ASYMMETRIC by Kaedon's pure-rusher doctrine (2026-11-25: "a
# prospect can be great as just a pure rusher though"): the leg
# PAYS when strong and goes QUIET when weak. Below
# RB_RECEIVING_MIN_PRICE the leg is simply absent from the
# priced list - its weight is never charged - so a pure rusher
# grades exactly as if the leg did not exist (Henry is the
# named proof: 76.91 flat vs 78.12 asymmetric in the lab).
# Missing receiving data = leg absent, never zero.
RB_RECEIVING_WEIGHT = 0.08
RB_RECEIVING_MIN_PRICE = 50.0
RB2B_FUND = {"utilization": -0.04, "age": -0.02, "breakout": -0.02}


def _rb_receiving_leg(seasons):
    """The RB-2b receiving leg value, or None when the profile
    carries no featured-season receiving evidence."""
    feat = _role_gated_seasons(seasons or [], "RB")
    if not feat:
        return None
    best_rg = 0.0
    tot_rec = 0.0
    tot_all = 0.0
    for s in feat:
        gp = max(_safe(s.get("games_played"), 12), 1)
        rec = _safe(s.get("receiving_yards"))
        best_rg = max(best_rg, rec / gp)
        tot_rec += rec
        tot_all += rec + _safe(s.get("rush_yards"))
    if tot_all <= 0:
        return None
    share = tot_rec / tot_all
    return (0.65 * _scale(best_rg, 4.0, 42.0)
            + 0.35 * _scale(share, 0.04, 0.32))


def _rb2b_priced_list(priced, seasons, base_weights):
    """Apply the RB-2b priced-list transforms to the shipped
    list: funding deltas on the legs present, the p40 fill when
    athleticism is missing, and the asymmetric receiving leg.
    Mirrors the validated lab construction order exactly."""
    out = []
    for (name, w, v) in priced:
        if name in RB2B_FUND:
            w = max(w + RB2B_FUND[name], 0.0)
        out.append((name, w, v))
    if not any(n == "athleticism" for n, _, _ in out):
        out.append(("athleticism",
                    float(base_weights.get("athleticism", 0.0936)),
                    RB_ATH_FILL_P40))
    rec = _rb_receiving_leg(seasons)
    if rec is not None and rec >= RB_RECEIVING_MIN_PRICE:
        out.append(("receiving", RB_RECEIVING_WEIGHT, rec))
    return out


# ------------------------------------------------------------------
# v3.13: the ceiling-equalization program (Kaedon's ceiling
# doctrine, 2026-11-25: "I just want 97+ obtainable for all
# positions") plus the TE completion package. The transforms
# below are priced-list transforms over PINNED frozen-pool
# constants (the 2016-2024 recorded population on
# gate_rows_v8, never recomputed per class - the RB_ATH_POOL
# precedent), position-scoped so a position's grades change
# only where a phase names them. The source-level pieces
# live in their own blocks: Phase 4's OR-gate and Phase 7's
# L2 in _score_production_season_raw, L4 in
# PROD_EVIDENCE_CFG, L1 in score_prospect's weight assembly,
# L6 in BLEND_CFG, Phase 6c's banding at _rb_eff_band.
# Design + sweep evidence: prospect-runs/
# rb-ceiling-audit-2026-11-25.md (Phases 1-5),
# bowers-mccaffrey-residuals-2026-11-25.md (Phases 6-7),
# rb-97-attainability-2026-11-25.md (Phase 6b - the
# attainability acceptance shipped as a test),
# rb-efficiency-banding-2026-11-25.md (Phase 6c).
#
# Phase 1 (RB breakout): the leg's pool max was 90.21 - its
# three components have near-mutually-exclusive maxima, so
# the best breakout season in the pool's history priced
# under 91 while the same leg pays 98.82 at WR and 100 at
# TE. A p95-anchored stretch re-bases the thin top only:
# below p95 the leg is untouched (the median-anchored form
# was killed in the audit - it refunded the spike family
# the corroboration machinery had just priced).
RB_BREAKOUT_STRETCH3_KNOTS = (85.0, 86.926255, 90.206349)
# Phase 6 (RB age, lever G1): age was a linear leg in a
# percentile world - every 0.1 yr past the 20.5 anchor cost
# 2.2 points uniformly, while the re-anchored legs pay by
# rank. The RB age value becomes its frozen-pool percentile
# (n = 481), the same 5 + 95 x pct form as the athleticism
# re-anchors. It is a broad youth lift, gate-improving in
# the residuals lab (P +0.0028 / S +0.0058 standalone).
RB_AGE_POOL = (
    10.0, 12.700000000000014, 14.850000000000012, 15.75000000000001, 15.950000000000006, 16.100000000000012, 16.450000000000014, 17.05,
    17.350000000000012, 17.45000000000001, 17.550000000000008, 17.600000000000016, 17.600000000000016, 17.80000000000001, 17.85, 18.200000000000003,
    18.25000000000001, 18.25000000000001, 18.3, 18.3, 18.35000000000001, 18.35000000000001, 18.4, 18.4,
    18.600000000000012, 18.600000000000012, 18.650000000000002, 18.70000000000001, 18.70000000000001, 18.70000000000001, 18.850000000000016, 18.850000000000016,
    18.900000000000006, 18.900000000000006, 18.900000000000006, 19.000000000000004, 19.05000000000001, 19.15000000000001, 19.2, 19.2,
    19.400000000000013, 19.55, 19.65, 19.700000000000006, 19.750000000000014, 19.750000000000014, 19.800000000000004, 20.0,
    20.222222222222225, 20.66666666666666, 21.111111111111093, 21.111111111111093, 21.111111111111093, 21.33333333333335, 21.77777777777779, 22.44444444444448,
    23.111111111111093, 23.33333333333335, 23.33333333333335, 23.555555555555525, 23.777777777777782, 24.22222222222222, 24.666666666666654, 24.666666666666654,
    25.333333333333346, 25.333333333333346, 25.333333333333346, 25.555555555555525, 25.777777777777782, 26.00000000000004, 26.222222222222214, 26.44444444444447,
    26.66666666666665, 27.333333333333343, 27.333333333333343, 27.55555555555552, 27.77777777777778, 28.222222222222214, 28.222222222222214, 28.222222222222214,
    29.111111111111082, 29.555555555555518, 30.22222222222221, 30.22222222222221, 30.444444444444468, 30.444444444444468, 30.444444444444468, 30.666666666666643,
    30.666666666666643, 31.11111111111108, 31.333333333333336, 31.555555555555593, 32.00000000000003, 32.00000000000003, 32.22222222222221, 32.22222222222221,
    32.22222222222221, 32.444444444444464, 32.444444444444464, 32.444444444444464, 32.444444444444464, 32.66666666666664, 32.66666666666664, 32.8888888888889,
    33.11111111111107, 33.11111111111107, 33.33333333333333, 33.777777777777764, 34.00000000000002, 34.00000000000002, 34.22222222222221, 34.66666666666664,
    34.8888888888889, 34.8888888888889, 35.33333333333333, 35.33333333333333, 35.33333333333333, 35.33333333333333, 35.33333333333333, 35.555555555555586,
    35.555555555555586, 35.777777777777764, 36.00000000000002, 36.00000000000002, 36.00000000000002, 36.2222222222222, 36.2222222222222, 36.2222222222222,
    36.44444444444446, 36.44444444444446, 36.666666666666636, 36.88888888888889, 36.88888888888889, 37.11111111111115, 37.11111111111115, 37.33333333333333,
    37.555555555555586, 37.555555555555586, 37.555555555555586, 37.77777777777776, 37.77777777777776, 38.000000000000014, 38.000000000000014, 38.22222222222219,
    38.666666666666636, 39.33333333333333, 39.33333333333333, 39.33333333333333, 39.33333333333333, 39.555555555555586, 39.77777777777776, 39.77777777777776,
    39.77777777777776, 40.22222222222219, 40.22222222222219, 40.44444444444445, 40.44444444444445, 40.66666666666663, 40.66666666666663, 40.888888888888886,
    40.888888888888886, 41.11111111111114, 41.11111111111114, 41.11111111111114, 41.11111111111114, 41.33333333333332, 41.55555555555558, 41.55555555555558,
    41.55555555555558, 41.55555555555558, 42.222222222222186, 42.222222222222186, 42.222222222222186, 42.44444444444444, 42.44444444444444, 42.44444444444444,
    42.44444444444444, 43.55555555555558, 43.55555555555558, 43.77777777777776, 43.77777777777776, 43.77777777777776, 44.000000000000014, 44.222222222222186,
    44.44444444444444, 44.44444444444444, 44.6666666666667, 44.88888888888888, 44.88888888888888, 44.88888888888888, 45.55555555555557, 45.55555555555557,
    45.77777777777775, 45.77777777777775, 46.00000000000001, 46.222222222222186, 46.222222222222186, 46.222222222222186, 46.222222222222186, 46.44444444444444,
    46.6666666666667, 46.6666666666667, 46.6666666666667, 46.88888888888887, 46.88888888888887, 46.88888888888887, 47.11111111111113, 47.11111111111113,
    47.33333333333331, 47.555555555555564, 47.555555555555564, 48.44444444444444, 48.6666666666667, 48.6666666666667, 48.88888888888887, 48.88888888888887,
    49.11111111111113, 49.33333333333331, 49.33333333333331, 49.77777777777774, 49.77777777777774, 50.22222222222226, 50.22222222222226, 50.22222222222226,
    51.33333333333331, 51.33333333333331, 51.33333333333331, 51.33333333333331, 51.33333333333331, 51.33333333333331, 51.77777777777774, 51.77777777777774,
    52.0, 52.0, 52.0, 52.44444444444443, 52.44444444444443, 52.44444444444443, 52.44444444444443, 52.666666666666686,
    52.666666666666686, 52.666666666666686, 52.666666666666686, 52.888888888888864, 52.888888888888864, 52.888888888888864, 53.11111111111112, 53.11111111111112,
    53.11111111111112, 53.11111111111112, 53.11111111111112, 53.3333333333333, 53.777777777777814, 53.777777777777814, 53.99999999999999, 53.99999999999999,
    53.99999999999999, 54.22222222222225, 54.22222222222225, 54.44444444444443, 54.44444444444443, 54.44444444444443, 54.888888888888864, 54.888888888888864,
    55.11111111111112, 55.11111111111112, 55.3333333333333, 55.3333333333333, 55.55555555555556, 55.55555555555556, 55.55555555555556, 55.777777777777814,
    55.99999999999999, 55.99999999999999, 55.99999999999999, 55.99999999999999, 56.22222222222225, 56.44444444444443, 56.44444444444443, 56.666666666666686,
    56.666666666666686, 56.666666666666686, 56.88888888888886, 57.111111111111114, 57.111111111111114, 57.33333333333329, 57.33333333333329, 57.55555555555555,
    57.55555555555555, 57.55555555555555, 57.77777777777781, 57.77777777777781, 57.77777777777781, 57.77777777777781, 58.22222222222224, 58.66666666666668,
    59.111111111111114, 59.33333333333337, 59.33333333333337, 59.55555555555555, 59.55555555555555, 59.55555555555555, 59.77777777777781, 59.77777777777781,
    59.77777777777781, 59.77777777777781, 59.77777777777781, 59.999999999999986, 60.22222222222224, 60.66666666666668, 60.88888888888886, 60.88888888888886,
    60.88888888888886, 61.111111111111114, 61.111111111111114, 61.33333333333337, 61.55555555555554, 61.55555555555554, 61.55555555555554, 61.55555555555554,
    61.55555555555554, 61.55555555555554, 61.55555555555554, 62.222222222222236, 62.222222222222236, 62.222222222222236, 62.222222222222236, 62.444444444444414,
    62.444444444444414, 62.66666666666667, 62.66666666666667, 62.88888888888885, 62.88888888888885, 62.88888888888885, 63.11111111111111, 63.11111111111111,
    63.333333333333364, 63.99999999999998, 63.99999999999998, 64.22222222222224, 64.44444444444441, 64.88888888888893, 65.11111111111111, 65.33333333333337,
    65.33333333333337, 65.55555555555554, 65.7777777777778, 65.7777777777778, 65.99999999999997, 65.99999999999997, 66.22222222222223, 66.22222222222223,
    66.4444444444444, 66.66666666666666, 66.88888888888891, 67.1111111111111, 67.1111111111111, 67.1111111111111, 67.1111111111111, 67.33333333333336,
    67.33333333333336, 67.99999999999997, 68.22222222222223, 68.22222222222223, 68.88888888888893, 69.1111111111111, 69.33333333333336, 69.33333333333336,
    69.55555555555554, 69.7777777777778, 69.7777777777778, 69.99999999999997, 69.99999999999997, 70.22222222222223, 70.22222222222223, 70.44444444444449,
    70.66666666666666, 71.11111111111109, 71.11111111111109, 71.11111111111109, 71.99999999999996, 71.99999999999996, 72.22222222222221, 72.22222222222221,
    72.66666666666666, 72.66666666666666, 72.88888888888891, 73.1111111111111, 73.1111111111111, 73.33333333333336, 73.33333333333336, 73.55555555555553,
    73.77777777777779, 73.77777777777779, 73.77777777777779, 73.77777777777779, 73.99999999999997, 74.22222222222223, 74.22222222222223, 74.22222222222223,
    74.22222222222223, 74.22222222222223, 74.22222222222223, 74.44444444444449, 74.66666666666666, 75.11111111111109, 75.33333333333334, 75.33333333333334,
    75.55555555555551, 76.22222222222221, 76.22222222222221, 76.44444444444447, 77.11111111111109, 77.33333333333334, 77.55555555555553, 77.77777777777779,
    77.77777777777779, 78.00000000000004, 78.00000000000004, 78.00000000000004, 78.22222222222221, 78.22222222222221, 78.22222222222221, 78.44444444444447,
    78.66666666666666, 78.88888888888891, 78.88888888888891, 78.88888888888891, 78.88888888888891, 79.11111111111109, 79.11111111111109, 79.33333333333334,
    80.00000000000003, 80.00000000000003, 80.8888888888889, 81.33333333333333, 81.33333333333333, 81.55555555555559, 81.77777777777777, 82.22222222222221,
    82.66666666666664, 82.66666666666664, 82.8888888888889, 83.11111111111109, 83.77777777777777, 83.77777777777777, 84.44444444444446, 84.44444444444446,
    85.33333333333333, 85.77777777777776, 86.00000000000001, 86.66666666666664, 87.33333333333333, 87.33333333333333, 87.33333333333333, 88.00000000000003,
    88.00000000000003, 88.2222222222222, 88.2222222222222, 88.66666666666663, 88.88888888888889, 88.88888888888889, 91.11111111111114, 91.55555555555559,
    91.77777777777776, 92.2222222222222, 92.44444444444446, 92.44444444444446, 92.88888888888889, 93.55555555555557, 94.22222222222219, 94.22222222222219,
    94.44444444444444, 94.88888888888887, 95.11111111111113, 95.33333333333331, 95.33333333333331, 96.88888888888889, 97.55555555555557, 98.88888888888887,
    100.0,
)
# Phase 2 (WR + TE athleticism): the 3.12 RB percentile
# re-anchor extends to the same defect at WR (shipped-scale
# max 86.81, n = 239) and TE (max 85.03, n = 103).
WR_ATH_POOL = (
    8.942307692307713, 13.150904159132024, 15.145927601809976, 16.421245421245438, 18.190476190476236, 20.326916221033894, 20.794871794871796, 23.14971368294154,
    23.917647058823576, 23.960784313725522, 26.90980392156862, 27.222222222222214, 29.304761904761932, 29.32947118241235, 29.424064171122964, 29.714285714285758,
    30.254010695187162, 31.57142857142856, 31.71672903672904, 33.8000801282051, 34.370833333333344, 34.624999999999964, 34.642857142857146, 34.76319758672699,
    34.94999999999997, 35.02787878787876, 35.357142857142854, 35.876470588235264, 36.1833333333333, 36.38095238095242, 36.385684903332006, 36.54441473559124,
    36.666666666666636, 36.76954579300074, 36.791666666666615, 37.38823529411766, 37.431372549019606, 38.433333333333366, 38.84375428493069, 39.45555555555553,
    39.90196078431371, 39.9901960784314, 40.063725490196056, 40.571428571428505, 40.97549019607839, 41.24999999999999, 41.27450980392161, 41.488095238095234,
    42.30090497737553, 42.416666666666615, 42.66458929517754, 42.668431372549016, 42.80392156862742, 43.01960784313725, 43.316346153846176, 43.588235294117624,
    43.92156862745098, 44.917857142857166, 46.119047619047585, 46.59999999999996, 46.75595238095232, 47.59374999999997, 48.213030303030266, 48.285714285714235,
    48.46078431372548, 48.61428571428573, 48.9285714285714, 48.95238095238095, 49.035714285714285, 49.19960784313723, 49.366666666666674, 49.65490196078427,
    49.8371040723982, 49.88536953242834, 49.94799808035101, 50.13235294117642, 50.226190476190496, 50.3401275195392, 50.53571428571423, 50.53571428571426,
    50.74999999999997, 51.16262626262621, 51.1845238095238, 51.3774509803922, 51.64479638009054, 51.71568627450976, 51.822222222222166, 51.90431372549013,
    51.91215686274512, 52.270889894419284, 52.31770833333329, 52.83861236802412, 52.99999999999994, 53.42857142857127, 53.42857142857127, 53.5357142857142,
    53.996503496503465, 54.04901960784317, 54.26470588235285, 54.357142857142804, 54.8751885369532, 55.261245674740486, 55.3333333333333, 55.46256684491981,
    55.666666666666586, 55.67857142857142, 56.093750000000036, 56.77489177489183, 57.06674208144789, 57.15336134453778, 57.658930481283356, 57.869696969696925,
    57.931372549019656, 57.9362745098039, 57.968910558875926, 58.051515151515105, 58.16001645413408, 58.161363636363646, 58.398098633392685, 59.44523809523807,
    59.500000000000036, 59.705882352941146, 59.916666666666686, 59.975034822672235, 60.01029411764706, 60.11666666666668, 60.14354175236526, 60.23750000000003,
    60.274999999999935, 60.27499999999999, 60.31631016042783, 60.34999999999999, 60.45458823529409, 60.6210916799152, 60.67571644042228, 60.7625,
    60.90123033403085, 61.22499999999996, 61.24083710407235, 61.42470588235292, 61.43484162895927, 61.74999999999999, 61.8426470588235, 62.055392156862744,
    62.07598039215687, 62.18749999999999, 62.26127450980394, 62.56510695187161, 63.252861377108985, 63.351143790849676, 63.57760180995478, 63.70598633392751,
    63.74166666666666, 63.74830659536544, 64.14006535947712, 64.21666666666664, 64.23111111111109, 64.64171122994654, 64.66323529411764, 64.725,
    64.72500000000002, 65.12499999999997, 65.28499999999998, 65.31862745098039, 65.41333618686562, 65.90726381461674, 65.98750000000001, 65.99166666666665,
    66.15938235294118, 66.22083333333327, 66.23709150326799, 66.62083333333335, 66.84166666666667, 67.04293582887702, 67.18986928104573, 67.22303921568626,
    67.40833333333335, 67.58749999999993, 67.64588076311603, 67.90537634408601, 67.97499999999998, 68.29999999999998, 68.30813953488371, 68.40000000000003,
    68.50000000000003, 68.53989441930618, 68.82499999999997, 68.96960784313724, 69.21249999999998, 69.37083333333327, 69.45000000000005, 69.8,
    69.8, 69.8, 69.8, 69.86307189542484, 69.86831550802138, 69.91666666666666, 69.9458333333333, 69.9825721153846,
    70.14999999999999, 70.56363636363636, 70.57435252175739, 70.64543740573153, 71.3850515463917, 71.43406120023765, 71.47500000000002, 71.99919786096251,
    72.39754901960788, 72.63749999999995, 72.72845651080948, 72.87708333333326, 73.37500000000001, 73.86617647058819, 74.0671568627451, 74.54426847662143,
    74.61250000000001, 74.92584670231729, 75.13079322638144, 75.2796875, 75.4, 75.63333333333334, 75.76166666666667, 76.74509803921569,
    77.01612903225805, 77.22083523331813, 77.54086961795029, 77.64999999999996, 78.28027699645345, 78.54999999999997, 79.0321428571428, 80.53377329192547,
    80.55357142857142, 80.9821428571428, 81.73214285714282, 82.30535714285712, 83.69666666666666, 85.70059019857001, 86.80535714285713,
)
TE_ATH_POOL = (
    17.03587545018151, 19.073703713764438, 20.19266837169648, 26.697371190166944, 28.742081447963805, 31.30048600636837, 32.70996695612078, 33.73333333333324,
    36.43254847619174, 37.84615384615383, 38.18034906270202, 38.723550958845074, 39.48993730727477, 40.45444261620735, 40.655196465938154, 40.923076923076934,
    41.19848778672309, 42.36322990734757, 42.49999999999995, 42.59397039448191, 43.28819709702062, 44.00000000000001, 44.626696832579206, 46.64508457845379,
    47.0000000000001, 47.553544494720974, 48.536915465486885, 49.8823529411765, 50.07692307692308, 50.77777777777783, 50.976190476190474, 54.35,
    55.11778417660771, 55.166666666666664, 55.17435897435897, 55.55218311100664, 55.886231415643216, 55.958796105854944, 56.30769230769229, 56.33333333333333,
    58.751131221719476, 59.01680672268906, 59.06167851756088, 59.22577371029384, 59.78129713423834, 60.412450980392165, 60.61830303030301, 61.309132044426185,
    61.63, 61.83110516934048, 61.89544303797469, 61.94444444444442, 61.96875, 62.1, 62.12601102941178, 62.426666666666605,
    62.735294117647086, 62.86492673992676, 63.05700341756421, 63.184252022487314, 63.89666666666668, 64.41223776223777, 65.37820512820512, 65.78514328808448,
    66.30080213903744, 66.3793363499246, 66.49705882352941, 66.50949901491697, 66.54204374057316, 67.08602048106694, 67.2470310391363, 67.4210407239819,
    68.35988304093569, 68.36414027149321, 68.87074142156864, 69.45908748114631, 69.75027890238415, 69.8, 69.8, 69.8,
    69.8, 70.7081447963801, 71.5123076923077, 72.05388728918139, 73.13421052631578, 73.27105263157895, 74.14981570512822, 74.38717948717951,
    74.92239583333335, 75.44048582995953, 75.50763673890609, 75.85540865384615, 75.93421052631578, 76.51376253287313, 76.5596279537456, 77.67304359243698,
    77.90894736842105, 77.9729034123771, 78.70225563909776, 79.36684210526317, 82.67937237394959, 83.06859394475497, 85.02522648114272,
)
# Phase 3 (QB): narrow compression of the two legs whose
# saturation is a subsidy rather than a measurement -
# experience (median 90.00; a third of QBs print >= 99.5)
# and competition (p99 = 100, where its real work is
# already done by the production/efficiency discount).
# Piecewise through (median -> median), (p99 -> 95),
# (max -> 97). The broad all-saturated form was killed in
# the audit: it inverted the discrepancy (Burrow 93.57,
# the QB ceiling down to 95.00, below every other
# position).
QB_EXPERIENCE_COMPRESS_KNOTS = (90.0, 100.0, 100.0)
QB_COMPETITION_COMPRESS_KNOTS = (80.0, 100.0, 100.0)
# Phase 3-TE (production stretch): TE production had the
# pool's lowest median and a thin top. The stretch's knots
# are pinned from the AMENDED distribution - the leg
# values after the OR-gate, the all-purpose terms and the
# corroboration-priced release have repriced the pool:
# (median -> median), (p99 -> 97), (max -> 99).
TE_PROD_STRETCH2_KNOTS = (35.231364, 85.096352, 94.268064)
# Phase 7 (L3, sustainment - the Mackey lever): a credit on
# the TE production leg for the elite-from-the-first-
# featured-season-and-held shape. Each featured season is
# priced on production's own RATE terms only (yds/g and
# TDs/g renormalized; share is the team-context part and is
# excluded). X = min(first-featured rate, second-best
# rate); the credit is lambda x max(0, X - 60), capped at
# +8 production points, added after the stretch. On the
# recorded pool it pays exactly 6 profiles (Bowers the
# maximum, +4.00) and zero to late bloomers, one-year
# comets and late explosions, by construction.
TE_SUSTAIN_CFG = {
    "lambda": 0.25,
    "threshold": 60.0,
    "cap": 8.0,
}


def _pct_reanchor(sorted_pool, value):
    """5 + 95 x pct against a pinned frozen pool (the 3.12
    RB athleticism form, generalized to the 3.13 pools)."""
    import bisect as _bisect
    pct = _bisect.bisect_right(
        sorted_pool, float(value)) / len(sorted_pool)
    return 5.0 + 95.0 * pct


def _thin_top_map(value, knots, top_targets):
    """The labs' stretch/compress shape, one form for all
    three: at or below the first knot the value stands;
    between knot 1 and knot 2 it maps linearly onto
    (knot 1 -> target 1); above knot 2 it maps linearly onto
    (target 1 -> target 2) at the pool max."""
    k0, k1, k2 = (float(knots[0]), float(knots[1]),
                  float(knots[2]))
    t1, t2 = float(top_targets[0]), float(top_targets[1])
    v = float(value)
    if v <= k0:
        return v
    if v <= k1:
        return k0 + (v - k0) * (t1 - k0) / max(k1 - k0, 1e-9)
    return t1 + (v - k1) * (t2 - t1) / max(k2 - k1, 1e-9)


def _te_rate_price(season: Dict) -> float:
    """A TE season's RATE price for the sustainment credit:
    the production leg's two rate terms (yds/g at .27, TDs/g
    at .25) renormalized to 0-100. Receiving terms - the
    sustainment shape is a receiving shape; rushing is
    priced by L2 inside the leg itself."""
    gp = max(_safe(season.get("games_played"), 12), 4)
    ypg = _safe(season.get("receiving_yards")) / gp
    tpg = _safe(season.get("receiving_tds")) / gp
    return (_scale(ypg, 20, 100) * 0.27
            + _scale(tpg, 0.12, 0.65) * 0.25) / 0.52


def _te_sustainment_credit(seasons) -> float:
    """The L3 credit in production points (0..cap) for a TE
    profile's season list."""
    feat = [s for s in sorted(
        (s for s in (seasons or [])
         if season_has_counting_stats(s)),
        key=lambda s: s.get("season", 0))
        if _is_role_season(s, "TE")]
    if len(feat) < 2:
        return 0.0
    rates = [_te_rate_price(s) for s in feat]
    second = sorted(rates, reverse=True)[1]
    x = min(rates[0], second)
    return min(float(TE_SUSTAIN_CFG["cap"]),
               float(TE_SUSTAIN_CFG["lambda"]) * max(
                   0.0, x - float(TE_SUSTAIN_CFG["threshold"])))


def _v313_priced_list(priced, pos, seasons):
    """Apply the v3.13 priced-list transforms: Phase 1 (RB
    breakout stretch), Phase 6/G1 (RB age percentile),
    Phase 2 (WR/TE athleticism percentile), Phase 3-TE
    (production stretch, then the L3 sustainment credit,
    clipped at 100 - the labs' construction order), Phase 3
    (QB experience + competition compression). Weights are
    never touched here; positions and legs a phase does not
    name pass through byte-identical."""
    out = []
    _sus = (_te_sustainment_credit(seasons)
            if pos == "TE" else 0.0)
    for (name, w, v) in priced:
        if pos == "RB":
            if name == "breakout":
                v = _thin_top_map(
                    v, RB_BREAKOUT_STRETCH3_KNOTS, (97.0, 99.0))
            elif name == "age":
                v = _pct_reanchor(RB_AGE_POOL, v)
        elif pos == "WR":
            if name == "athleticism":
                v = _pct_reanchor(WR_ATH_POOL, v)
        elif pos == "TE":
            if name == "athleticism":
                v = _pct_reanchor(TE_ATH_POOL, v)
            elif name == "production":
                v = _thin_top_map(
                    v, TE_PROD_STRETCH2_KNOTS, (97.0, 99.0))
                if _sus:
                    v = min(100.0, v + _sus)
        elif pos == "QB":
            if name == "experience":
                v = _thin_top_map(
                    v, QB_EXPERIENCE_COMPRESS_KNOTS, (95.0, 97.0))
            elif name == "competition":
                v = _thin_top_map(
                    v, QB_COMPETITION_COMPRESS_KNOTS, (95.0, 97.0))
        out.append((name, w, v))
    return out


# Phase 5 (classification): position assignment follows the
# position played in COLLEGE, not the draft-projected
# position. The stored position comes from draft/consensus
# sources, which record the position the NFL drafted the
# player to convert to; the model grades the player he was
# in college. The table records the verified college
# position where the two differ, keyed on (name, school,
# draft class year) so it can only ever name the player it
# was verified for. One entry: Thomas Duarte was a UCLA
# wide receiver whom Miami drafted as a TE conversion
# project - priced as a TE, a 231-lb slot receiver's file
# ran through TE anchors to the top of the position
# (95.00); priced at his college position, the identical
# file grades 74.47. Kaedon's directive (2026-11-25: "just
# like drop duarte") is implemented as reclassification,
# not deletion, and never as a draft-capital judgment -
# grades are capital-free, so his 7th-round pick plays no
# role here.
COLLEGE_POSITION_OVERRIDES = {
    ("thomas duarte", "ucla", 2016): "WR",
}


def _college_position_for(prospect):
    """The verified college position for a prospect whose
    stored position is a draft projection, else None."""
    name = str(prospect.get("name") or "").strip().lower()
    school = str(prospect.get("school") or "").strip().lower()
    try:
        year = int(prospect.get("draft_class_year") or 0)
    except (TypeError, ValueError):
        year = 0
    return COLLEGE_POSITION_OVERRIDES.get((name, school, year))


def _peak_set(priced, force_override=None, pos=None):
    """The peak set for a priced leg list [(name, weight, value)]:
    the top `peak_k` legs by value among the peak-ELIGIBLE legs
    (BLEND_CFG peak_exclude names legs that keep their mean weight
    but cannot occupy a peak slot), with the production leg's
    admission per BLEND_CFG peak_force ("always" / "never" /
    "margin" - see the cfg). force_override pins the admission
    mode for this call: the trim gate passes "always", because
    its threshold is defined on the production-ANCHORED peak
    (see trim_min_peak in BLEND_CFG) whatever admission mode the
    peak term itself ships."""
    k = max(1, int(BLEND_CFG.get("peak_k", 4)))
    _excl = set(BLEND_CFG.get("peak_exclude") or ())
    if pos is not None:
        _excl |= set((BLEND_CFG.get("peak_exclude_by_pos") or {})
                     .get(pos) or ())
    _cand = [t for t in priced if t[0] not in _excl]
    _min_w = None
    if pos is not None:
        _min_w = (BLEND_CFG.get("peak_min_weight_by_pos") or {}).get(pos)
    if _min_w:
        _cand = [t for t in _cand if t[1] >= float(_min_w)]
    top = sorted(_cand, key=lambda t: t[2], reverse=True)[:k]
    prod_leg = next((t for t in priced if t[0] == "production"), None)
    if prod_leg is not None and prod_leg not in top:
        _force = force_override or BLEND_CFG.get("peak_force",
                                                  "always")
        _admit = _force == "always"
        if _force == "margin" and top:
            _admit = prod_leg[2] >= top[-1][2] - float(
                BLEND_CFG.get("peak_force_margin", 0.0) or 0.0)
        if _admit:
            top = top[:-1] + [prod_leg] if top else [prod_leg]
    return top


def _blend_grade(priced, pos=None, allow_peak=True, trim_block=()):
    """Blend the priced legs [(name, weight, value), ...] into the
    pre-adjustment grade. Weights renormalize over the legs present,
    which is exactly the post-redistribution total (~1.0).

    allow_peak=False prices the plain mean: the caller passes it
    when the multi-None anchor fired (no measured performance core),
    because a profile with no measured performance has no peak to
    price - its context legs alone are not a peak.

    trim_block names legs that are ineligible for the trimmed-mean
    drop for THIS profile on top of BLEND_CFG's trim_exclude (see
    TRIM_MEASURED_CFG in score_prospect: measured performance legs
    are not forgivable)."""
    if not priced:
        return 0.0
    total_w = sum(w for _, w, _ in priced)
    if total_w <= 0:
        return 0.0
    mean_legs = priced
    _trim_ok = (BLEND_CFG.get("trim_lowest") and allow_peak
                and len(priced) >= int(BLEND_CFG.get("trim_min_legs", 5)))
    if _trim_ok:
        _excl = set(BLEND_CFG.get("trim_exclude") or ()) | set(trim_block)
        _min_w = float(BLEND_CFG.get("trim_min_weight", 0.0) or 0.0)
        _floor = float(BLEND_CFG.get("trim_drop_floor", 0.0) or 0.0)
        _elig = [t for t in priced
                 if t[0] not in _excl and t[1] >= _min_w
                 and t[2] >= _floor]
        _min_top = BLEND_CFG.get("trim_min_top")
        if _min_top is not None and max(
                v for _, _, v in priced) < float(_min_top):
            _elig = []
        _drop = min(_elig, key=lambda t: t[2]) if _elig else None
        _ceil = BLEND_CFG.get("trim_drop_ceiling")
        if _drop is not None and _ceil is not None \
                and _drop[2] > float(_ceil):
            _drop = None
        _min_peak = BLEND_CFG.get("trim_min_peak")
        if _drop is not None and _min_peak is not None:
            _top = _peak_set(priced, force_override="always", pos=pos)
            _tw = sum(w for _, w, _ in _top)
            _pk = (sum(w * v for _, w, v in _top) / _tw
                   if _tw > 0 else 0.0)
            if _pk < float(_min_peak):
                _drop = None
        if _drop is not None:
            mean_legs = [t for t in priced if t is not _drop]
    _mean_w = sum(w for _, w, _ in mean_legs)
    mean = (sum(w * v for _, w, v in mean_legs) / _mean_w
            if _mean_w > 0 else 0.0)
    alpha = float((BLEND_CFG.get("peak_alpha_by_pos") or {}).get(
        pos, BLEND_CFG.get("peak_alpha", 0.0)) or 0.0)
    if alpha <= 0.0 or not allow_peak:
        return mean
    # The peak set: top peak_k eligible legs by value, with
    # production's admission per BLEND_CFG (see _peak_set).
    top = _peak_set(priced, pos=pos)
    top_w = sum(w for _, w, _ in top)
    if top_w <= 0:
        return mean
    peak = sum(w * v for _, w, v in top) / top_w
    return (1.0 - alpha) * mean + alpha * peak


POSITION_WEIGHTS = {
    "QB": {
        "production": 0.1869,
        "utilization": 0.0537,
        "efficiency": 0.1891,
        "age": 0.2012,
        "breakout": 0.0705,
        "athleticism": 0.1473,
        "competition": 0.0613,
        "environment": 0.0126,
        "durability": 0.0,
        "experience": 0.0442,
        "recruiting": 0.0332,
    },
    "RB": {
        "production": 0.2305,
        "utilization": 0.1367,
        "efficiency": 0.1433,
        "age": 0.1587,
        "breakout": 0.146,
        "athleticism": 0.0936,
        "competition": 0.0472,
        "environment": 0.0044,
        "durability": 0.0063,
        "recruiting": 0.0332,
    },
    "WR": {
        "production": 0.2382,
        "utilization": 0.1094,
        "efficiency": 0.1162,
        "age": 0.1415,
        "breakout": 0.1369,
        "athleticism": 0.0802,
        "competition": 0.1037,
        "environment": 0.0393,
        "durability": 0.0,
        "recruiting": 0.0345,
    },
    "TE": {
        "production": 0.2189,
        "utilization": 0.1332,
        "efficiency": 0.1187,
        "age": 0.1637,
        "breakout": 0.1363,
        "athleticism": 0.1311,
        "competition": 0.0432,
        "environment": 0.0164,
        "durability": 0.0062,
        "recruiting": 0.0324,
    },
}

# Validate that all position weights sum to 1.0 (the fitted values
# are rounded to 4 decimal places, so allow their rounding slack).
for pos, weights in POSITION_WEIGHTS.items():
    assert abs(sum(weights.values()) - 1.0) < 0.001, f"{pos} weights must sum to 1.0, got {sum(weights.values())}"


# Enhanced evaluation functions
# 

from typing import Tuple

def calc_loaded_roster_adjustment(
    team: str,
    position: str,
    season: int,
    production_score: float,
    market_share: float,
    ypc: float = 0.0,
) -> float:
    # Values are (room_size, talent_tier).
    # talent_tier 1 = Day-2/3 starter above the prospect (minimal suppression signal)
    # talent_tier 2 = clear first-round prospect, established alpha role
    # talent_tier 3 = elite/top-15 pick prospect with dominant alpha role
    loaded_rosters: Dict[str, Dict[str, Dict[int, Tuple[int, int]]]] = {
        "Ohio State": {
            "WR": {
                2021: (2, 3),  # Olave (#10) + Wilson (#20) - co-equal first-round alphas
                2022: (2, 3),  # JSN (Biletnikoff, consensus WR1 prospect) + Egbuka (first-round)
                2023: (2, 3),  # MHJ (consensus WR1, top-5 overall) + Egbuka (first-round)
                2024: (3, 3),  # Smith (generational) + Egbuka (#19) + Tate (#4) - three first-rounders
                2025: (3, 3),  # Smith (projected #1 WR overall) + Tate (#4 overall pick)
            },
        },
        "Alabama": {
            "WR": {
                2020: (3, 3),  # Smith (Heisman/#10) + Waddle (#4) + Metchie (first-round)
                2021: (2, 3),  # Jameson Williams (#12 overall, dominant alpha 1,572 yds) + Metchie (#38)
                2022: (2, 1),  # Burton (6th round) + Holden (UDFA) - no first-round grade
                2025: (2, 2),  # Ryan Williams (first-round grade, WR2-3 in class, clear alpha)
            },
            "RB": {
                2020: (2, 3),  # Najee Harris (Doak Walker, #24 overall, dominant alpha)
                2021: (2, 1),  # Brian Robinson (3rd round/#98, clear alpha but Day 2-3 grade)
            },
        },
        "Georgia": {
            "TE": {
                2021: (2, 3),  # Brock Bowers (92.1 PFF freshman record, #13 overall eventual pick)
                2022: (2, 3),  # Bowers (established #1 TE in class) + Washington (Day 2/#93)
            },
            "WR": {
                2024: (2, 1),  # Lovett (Day 2-3) + Bell (undrafted) - no first-round grade
                2025: (2, 1),  # Bell (undrafted) + Young (4th round) - below first-round threshold
            },
        },
        "USC": {
            "WR": {
                2022: (2, 2),  # Jordan Addison (Biletnikoff, #23 overall, unambiguous alpha)
                2024: (2, 1),  # Branch (Day 2, no clear alpha) + Lemon (emerging, not yet first-round)
                2025: (2, 2),  # Lemon (#20 overall, Biletnikoff, clear alpha) + Lane (Day 2)
            },
        },
        "LSU": {
            "WR": {
                2019: (3, 3),  # Chase (#5) + Jefferson (#22) + Marshall (2nd round) - legendary room
                2022: (2, 1),  # Boutte (grade cratered Day 3 due to injuries) + Nabers/Thomas emerging sophs
                2023: (2, 3),  # Nabers (#6 overall) + Brian Thomas (#23 overall) - co-equal elite alphas
                2024: (2, 1),  # Kyren Lacy (Day 2) + Anderson - post-elite-era room
                2025: (2, 1),  # Aaron Anderson (Day 2) - no first-round grade alpha
            },
        },
        "Texas": {
            "WR": {
                2023: (2, 2),  # Worthy (#28) + Mitchell (#52) - two first/2nd-round picks, three-way split
                2024: (2, 2),  # Golden (#23, clear alpha by season end) + Bond (Day 2)
                2025: (2, 1),  # Wingo (unproven) + Moore (Day 3) - no first-round grade
            },
        },
        "Oregon": {
            "WR": {
                2024: (2, 1),  # Johnson (Day 2-3) + Stewart (not yet first-round grade) - three-way split
                2025: (2, 1),  # Stewart injured entire season; Moore (5-star freshman) unproven
            }
        },
        "Notre Dame": {
            "RB": {
                2024: (2, 3),  # Love (#3 overall, dominant alpha) + Price (#32 despite never starting)
            },
        },
    }

    team = (team or "").strip()
    position = (position or "").strip().upper()

    if not team or not position:
        return 1.0

    try:
        production_score = float(production_score)
    except (TypeError, ValueError):
        production_score = 0.0

    try:
        market_share = float(market_share)
    except (TypeError, ValueError):
        market_share = 0.0

    production_score = max(0.0, min(production_score, 100.0))
    market_share = max(0.0, min(market_share, 1.0))

    entry = (
        loaded_rosters.get(team, {})
        .get(position, {})
        .get(season)
    )

    if not entry:
        return 1.0

    room_size, talent_tier = entry

    if room_size < 2:
        return 1.0

    # Only backups benefit from the crowded-room bonus - a player who dominates
    # their team's market share is the alpha, not the one being suppressed.
    # Threshold: >= 0.28 market share means they're the clear lead back/receiver;
    # their stats are face-value, not opportunity-constrained.
    if market_share >= 0.28:
        return 1.0

    if room_size >= 4:
        base_bonus = 0.12
    elif room_size == 3:
        base_bonus = 0.10
    elif position == "RB":
        base_bonus = 0.10   # committee RBs are more opportunity-limited than shared WR rooms
    else:
        base_bonus = 0.06

    # ms_factor: inverted so lower market share (more suppressed) = bigger bonus.
    # A backup with ms=0.10 is heavily blocked; ms=0.25 is getting decent volume
    # despite a loaded room - both deserve a bonus, but the more blocked player more so.
    if market_share <= 0.12:
        ms_factor = 1.00   # heavily suppressed backup
    elif market_share <= 0.18:
        ms_factor = 0.80   # clear backup role
    elif market_share <= 0.22:
        ms_factor = 0.60   # committee / split role
    else:
        ms_factor = 0.40   # decent volume but still not the alpha

    if production_score >= 85:
        prod_factor = 1.00
    elif production_score >= 75:
        prod_factor = 0.80
    elif production_score >= 65:
        prod_factor = 0.55
    elif production_score >= 50:
        prod_factor = 0.30
    elif position == "RB" and ypc >= 5.5:
        # Committee RB with elite YPC: volume suppressed by opportunity, not talent.
        # Per-carry efficiency is the quality signal - use it instead of production_score.
        prod_factor = 0.65
    else:
        prod_factor = 0.10

    # Scale base_bonus by how elite the blocking talent was.
    # Tier 3 (generational alpha above) warrants a 55% larger bonus than the baseline.
    tier_scale = {1: 1.00, 2: 1.25, 3: 1.55}.get(talent_tier, 1.00)
    base_bonus = base_bonus * tier_scale

    realized_bonus = base_bonus * ((ms_factor * 0.6) + (prod_factor * 0.4))
    multiplier = 1.0 + realized_bonus

    return min(multiplier, 1.18)


# v3.11 round 12 (Phase B / discrepancy audit P1): the
# measured room price. When a season's top roommate out-
# produced the player per game, the season's production
# price is multiplied by 1 + cap * strength, where strength
# is the roommate's per-game line scaled on the position's
# own per-game anchors - the price the room actually charged
# him, measured from the room_* aggregates the Phase B build
# put on the season row. "replace_depth" retires the
# depth-chart adjustment (calc_depth_chart_adjustment
# returns neutral): the measured price is the ONLY room
# price, so nobody is paid twice for the same room, and a
# season with no aggregates gets no adjustment at all.
# Swept in round12_lab (frame v8) at caps .10/.15/.20;
# Kaedon accepted cap .15 WITH its two named exceptions
# (Jefferson +3.44, TE perfect fixture 99.28) on 2026-11-25.
ROOM_CFG = {
    "cap": 0.15,
    "replace_depth": True,
}


def _room_multiplier_for_season(season: Dict, pos: str) -> float:
    """One season's measured room price from its room_*
    aggregates (position-group totals + the top roommate's
    line). 1.0 when the season carries no aggregates, when
    the player out-produced his top roommate per game, or
    for positions the room does not price (QB)."""
    cap = ROOM_CFG.get("cap") or 0.0
    if not cap:
        return 1.0
    rg = season.get("room_games")
    if not rg:
        return 1.0
    gp = max(_safe(season.get("games_played"), 12), 4)
    if pos == "RB":
        own = (_safe(season.get("rush_yards"))
               + _safe(season.get("receiving_yards"))) / gp
        mate = (season.get("room_mate_all_yds") or 0) / rg
        if mate <= own or mate <= 0:
            return 1.0
        strength = _scale(mate, 30, 150)
    elif pos in ("WR", "TE"):
        own = _safe(season.get("receiving_yards")) / gp
        mate = (season.get("room_mate_rec_yds") or 0) / rg
        if mate <= own or mate <= 0:
            return 1.0
        strength = _scale(mate, 32, 120) if pos == "WR" \
            else _scale(mate, 20, 100)
    else:
        return 1.0
    return 1.0 + cap * strength / 100.0


# v3.11 round 10 (discrepancy audit P1): where the
# crowded-room multiplier lives. "per_season" False = the
# legacy placement (one multiplier on the finished leg,
# from the LATEST season's depth chart only). True = the
# multiplier prices inside each season's production price
# (see _score_production_season), so a career spent behind
# an alpha is credited in every season it happened, and
# the post-hoc latest-only multiplier retires. Swept in
# round10_lab.
DEPTH_CFG = {
    "per_season": False,
}


def _depth_multiplier_for_season(ls: Dict, position: str) -> float:
    """One season's crowded-room multiplier from its own
    depth chart (the tier table calc_depth_chart_adjustment
    has always used). 1.0 when the season carries no depth
    data, lists the player first, or his volume in the room
    was too thin to evidence suppression."""
    depth_rank = ls.get("depth_rank")
    group_size = ls.get("position_group_size")

    if depth_rank is None or group_size is None:
        return 1.0
    if depth_rank == 1 or group_size < 2:
        return 1.0

    pos = position.upper()
    gp = max(_safe(ls.get("games_played"), 12), 1)

    # Thresholds (yds/game) that represent "still producing meaningfully"
    # despite not being the WR1/RB1.
    if pos in ("WR", "TE"):
        vol = _safe(ls.get("receiving_yards")) / gp
        tiers = {2: [(70, 0.12), (50, 0.08), (35, 0.05)],
                 3: [(60, 0.15), (40, 0.10)]}
    elif pos == "RB":
        vol = _safe(ls.get("rush_yards")) / gp
        tiers = {2: [(80, 0.12), (60, 0.08), (40, 0.05)],
                 3: [(60, 0.15), (40, 0.10)]}
    else:
        return 1.0

    rank_tiers = tiers.get(min(depth_rank, 3), [])
    base_bonus = 0.0
    for threshold, bonus in rank_tiers:
        if vol >= threshold:
            base_bonus = bonus
            break

    if base_bonus == 0.0:
        return 1.0

    # Scale up slightly for larger rooms - harder to carve out volume with more competition
    group_factor = min(1.0, group_size / 4.0)
    realized_bonus = base_bonus * (0.60 + 0.40 * group_factor)
    return min(1.0 + realized_bonus, 1.20)


def calc_depth_chart_adjustment(seasons: List[Dict], position: str) -> float:
    """
    Data-driven crowded-room bonus based on CFBD-derived depth chart rank.

    A WR2/RB2 who still puts up meaningful volume despite playing behind an
    alpha shows elite underlying talent suppressed by opportunity, not a talent
    deficit.  This generalises calc_loaded_roster_adjustment() to all teams
    automatically, rather than requiring a manually curated list.

    depth_rank=1  → no adjustment (face-value production)
    depth_rank=2+ → bonus proportional to production volume and group size

    Returns a multiplier in [1.0, 1.20].  Returns 1.0 when depth data is absent
    (e.g. seed-only prospects), allowing the manual loaded_roster fallback to apply.
    """
    # v3.11 round 12 (R2-replace, Kaedon's ruling 2026-11-25):
    # RETIRED while ROOM_CFG replaces it - the measured room
    # price is the only room price now. Returning neutral here
    # is what lets the curated loaded-roster fallback in
    # score_prospect (which keys off a 1.0 from this function)
    # apply exactly as it always did for depth-absent profiles.
    # The tier table itself (_depth_multiplier_for_season)
    # stays on record but no longer reaches a grade here.
    if ROOM_CFG.get("replace_depth"):
        return 1.0
    if not seasons:
        return 1.0

    ls = _latest_season(seasons)
    if not ls:
        return 1.0

    return _depth_multiplier_for_season(ls, position)


def calc_experience_score(seasons: List[Dict],
                          position: str) -> Optional[float]:
    """
    Experience metric for quarterbacks to mitigate misses like Trey Lance.

    Args:
        seasons: List of season data
        position: Player position

    Returns:
        Experience score (0-100), or None for a QB with no
        games-started / games-played evidence on any season.
    """
    if position != "QB":
        return 0.0

    # v3.11 round 3 (M1 doctrine): a QB with NO games evidence on
    # any season has an UNMEASURED leg, not a measured zero - the
    # old code priced him 0.0, the model's last score-missing-as-
    # zero violation (every other leg returns None on no volume).
    # A recorded 0 (appeared but never started, or recorded zero
    # starts) is evidence and still prices 0.0. Live profiles
    # carry GP, so this binds only on GP-missing profiles.
    if not any(s.get("games_started") is not None
               or s.get("games_played") is not None
               for s in seasons):
        return None

    # Calculate total games started across all seasons.
    # Fall back to games_played when games_started is unavailable (common for
    # pre-draft profiles that only have box-score game counts, not start data).
    games_started = sum(_safe(s.get("games_started", 0)) for s in seasons)
    if games_started == 0:
        games_started = sum(_safe(s.get("games_played", 0)) for s in seasons)

    # Normalize to 0-100 scale (40 games = full experience)
    experience_ratio = min(games_started / 40.0, 1.0)
    return experience_ratio * 100.0


def calc_late_round_upside(draft_capital: Optional[Dict], seasons: List[Dict], position: str) -> float:
    """
    Retired in v3.7 (round-3 audit): always 0.0.

    Through v3.6 this was a payload-only diagnostic keyed to the
    projected round (75-100 for a round-4+ projection with an elite
    dominator or YPRR). It never fed the grade - the add-on was
    removed in v3.2 R7, and score_prospect has passed None since
    v3.7 - but the function itself still READ projected_round from
    any dict it was handed, leaving a live capital read inside the
    grading module one call-site change away from re-entering the
    payload. A capital-free model does not handle draft capital, so
    the read is gone: the capital argument is accepted for signature
    compatibility and ignored, whatever it carries.
    """
    return 0.0



def calc_interaction_features(production_score: float, efficiency_score: float, 
                             athleticism_score: float, draft_capital_score: float) -> Dict[str, float]:
    """
    Interaction features between key metrics to capture nuanced signals.
    
    Args:
        production_score: Production component score
        efficiency_score: Efficiency component score
        athleticism_score: Athleticism component score
        draft_capital_score: Draft capital component score
    
    Returns:
        Dictionary of interaction feature scores
    """
    # Production-efficiency interaction (high production + high efficiency)
    prod_eff_interaction = (production_score * efficiency_score) / 100.0
    
    # Athleticism-draft capital interaction (elite athlete + high draft capital)
    ath_dc_interaction = (athleticism_score * draft_capital_score) / 100.0
    
    # Production-athleticism interaction (elite producer + elite athlete)
    prod_ath_interaction = (production_score * athleticism_score) / 100.0
    
    # Triple interaction (all three elite)
    triple_interaction = (production_score * efficiency_score * athleticism_score) / 10000.0
    
    return {
        "production_efficiency_interaction": prod_eff_interaction,
        "athleticism_draft_capital_interaction": ath_dc_interaction,
        "production_athleticism_interaction": prod_ath_interaction,
        "triple_interaction": triple_interaction,
    }


def calc_translation_adjustment(
    prospect: Dict[str, Any],
    position: str,
    draft_capital: Optional[Dict[str, Any]],
    production_score: float,
    efficiency_score: float,
    age_score: float,
) -> float:
    """
    Post-model adjustment for signals no leg reads.

    v3.6 (round-2 audit F3): the layer used to re-price facts already
    inside the legs: every pick-gated branch (WR guardrails and boosts,
    RB boosts, the TE +9/+8/+5/+3/+2 pick tiers, the TE pick<=64
    penalty) duplicated the capital leg; the WR YPRR/market-share/
    rec-yds branches duplicated production and efficiency; the TE
    low-volume, rec-yds and target-share penalties duplicated
    production and utilization; the TE age penalty duplicated the age
    leg; the QB flat rushing bonus duplicated production's rushing
    component (now priced there once, audit F6); the WR YAC penalty
    duplicated production's season YAC multiplier. All removed.

    What remains is the one signal no leg reads from season data: the
    WR contested-catch trait (the legs price contested catches only
    through eval_metrics, a separate pipeline input). Signature and
    call contract are unchanged; the function no longer reads the
    projected pick at all, so the talent-only run and the main run see
    byte-identical translation by construction and capital_delta
    prices the capital leg alone.
    """
    if not prospect.get("seasons"):
        return 0.0

    latest = _latest_season(prospect["seasons"]) or {}
    adj = 0.0

    if position == "WR":
        contested = _safe(latest.get("contested_catch_rate"), 0.0)
        # Classic WR false-positive trait: wins volume without winning
        # contested balls. drop_rate check removed earlier: was
        # PFF-only, no free replacement.
        if contested > 0 and contested < 45.0:
            adj -= 1.5

    # Keep the adjustment bounded; this is a corrective signal, not the
    # main model. (The old -14/+9 bound was sized to the deleted TE
    # tiers; retained as a defensive clamp.)
    return max(-14.0, min(9.0, adj))


def score_prospect(
    prospect: Dict[str, Any],
    draft_capital: Optional[Dict[str, Any]] = None,
    skip_sagarin: bool = False,
    position_weights_override: Optional[Dict[str, Dict[str, float]]] = None,
    _active_season_year: Optional[int] = None,
    _today=None,
) -> Dict[str, Any]:
    """
    Run all component scorers and produce a final prospect_score.

    Args:
        prospect:      Normalised prospect dict from ingestion.py
        draft_capital: Consensus dict from mock_draft_consensus.py.
            Accepted for call compatibility (the pipeline and the
            backtest pass it positionally) and IGNORED since v3.7:
            grades are capital-free, so the grade is byte-identical
            whatever this dict carries - or whether it is passed
            at all.

    Returns:
        Dict with all component scores + final prospect_score + reasons.
    """
    pos      = (prospect.get("position") or "WR").upper()
    # v3.13 (Phase 5): the college position governs - a
    # prospect whose stored position is a draft-projected
    # conversion is graded at the position he played in
    # college (see COLLEGE_POSITION_OVERRIDES).
    _college_pos = _college_position_for(prospect)
    if _college_pos:
        pos = _college_pos
    # v3.8 (M1): placeholder rows (a season row with every counting
    # stat NULL - the eval pipeline's draft-year rows) are not
    # seasons. They are dropped at profile assembly in the DB loader
    # (pipeline.load_prospects_from_db) and again here, so every
    # entry point is safe; a prospect left with no real season takes
    # the empty-list neutral paths and is flagged
    # data_quality="truncated" on the scored output (its own
    # percentile panel hides and pool builders exclude flagged rows).
    seasons  = [s for s in (prospect.get("seasons") or [])
                if season_has_counting_stats(s)]
    _data_quality = "ok" if seasons else "truncated"
    # v3.8 (M2 role gate): the LEVEL legs (production, utilization -
    # and the career aggregates inside them) read only the seasons
    # from the player's first real role onward. See
    # _role_gated_seasons. Breakout is the exception, below: it is a
    # CHANGE measure, and its origin is the pre-role baseline by
    # definition.
    role_seasons = _role_gated_seasons(seasons, pos)
    raw_age  = prospect.get("age")
    age      = float(raw_age) if raw_age is not None else None
    ath      = prospect.get("athleticism") or {}
    dy       = int(prospect.get("draft_class_year") or 2026)

    # _eval_metrics is attached by the pipeline orchestration (pipeline.py) when
    # the evaluation pipeline has run first.  Shape: {metric_name: metric_payload}.
    eval_metrics: Optional[Dict] = prospect.get("_eval_metrics") or None

    # v3.4: two-mode grading (round 14 of the iteration log). A player is
    # IN-SEASON iff the run supplies an active season year, his latest
    # season IS that year, and its stored games played show a partial
    # season (1 to 11 games; 12 is the model's own FULL_SEASON_GAMES
    # constant, and a missing GP is not evidence of partiality). Otherwise
    # COMPLETE mode: every in-season branch below is gated on _inseason,
    # and with no active year supplied the grading is the complete-mode
    # path end to end (this is what keeps the backtest on final profiles
    # reproducible).
    _ls_mode = _latest_season(seasons) or {}
    _gp_mode = _ls_mode.get("games_played")
    _inseason = (
        _active_season_year is not None
        and _ls_mode.get("season") == _active_season_year
        and _gp_mode is not None
        and 1 <= _safe(_gp_mode) <= 11
    )
    _inseason_year = int(_active_season_year) if _inseason else None
    # v3.9: the F2 shrink's season-progress floor, from the run
    # calendar the caller pinned (_today). None when no date was
    # pinned - the pre-3.9 GP-only shrink then stands in every leg.
    _season_elapsed = (_season_games_elapsed(_inseason_year, _today)
                       if _inseason else None)

    production_score    = calc_production_score(role_seasons, pos, eval_metrics=eval_metrics,
                                                skip_sagarin=skip_sagarin,
                                                _inseason_year=_inseason_year,
                                                _draft_year=dy,
                                                _season_elapsed=_season_elapsed,
                                                _age=age)
    
    # Apply crowded-room bonus: prefer data-driven depth rank when available,
    # fall back to manually curated loaded_roster list when depth data is absent.
    ls = _latest_season(seasons) or {}
    team = ls.get("team", "")
    season = ls.get("season", 0)
    market_share = _safe(ls.get("market_share_yards"), 0.15)

    depth_chart_adjustment = calc_depth_chart_adjustment(seasons, pos)

    if DEPTH_CFG.get("per_season"):
        # Round 10: the room multiplier already priced inside
        # each season (see _score_production_season); applying
        # the latest-season multiplier again here would count
        # the room twice. The curated fallback still serves
        # profiles with no depth data on ANY season.
        if any(s.get("depth_rank") is not None for s in seasons):
            loaded_roster_adjustment = 1.0
        else:
            loaded_roster_adjustment = calc_loaded_roster_adjustment(
                team, pos, season, production_score, market_share,
                ypc=_safe(ls.get("yds_per_carry")),
            )
    elif depth_chart_adjustment != 1.0:
        # CFBD depth rank data available - use data-driven adjustment
        loaded_roster_adjustment = depth_chart_adjustment
    else:
        # No depth data (seed-only) - fall back to manually curated list
        loaded_roster_adjustment = calc_loaded_roster_adjustment(
            team, pos, season, production_score, market_share,
            ypc=_safe(ls.get("yds_per_carry")),
        )
    if production_score is not None:
        production_score = _clip(production_score * loaded_roster_adjustment)

    utilization_score   = calc_utilization_score(
        role_seasons, pos, _inseason_year=_inseason_year,
        _season_elapsed=_season_elapsed)
    # v3.6 (round-2 audit F8): the crowded-room multiplier applies to
    # PRODUCTION ONLY (above). It used to multiply utilization in full
    # and echo into efficiency (40% of excess) and breakout (35%), so
    # one room signal moved four legs at once. Utilization,
    # efficiency and breakout now stand on their own reads.

    efficiency_score    = calc_efficiency_score(
        seasons, pos, eval_metrics=eval_metrics,
        wepa=prospect.get("wepa"),
        _inseason_year=_inseason_year,
        _season_elapsed=_season_elapsed,
    )

    age_score           = calc_age_score(age, dy, pos)
    # v3.8 (TE fix, post-gate): breakout reads the UNGATED season
    # list. The role gate excuses pre-role seasons as level evidence,
    # but breakout's trajectory component is growth FROM the first
    # season on file and its age component keys on the first
    # threshold crossing - both are defined against the pre-role
    # baseline. Gating the leg deleted the rise it exists to price:
    # elite breakouts whose first season was a cameo (Trey McBride,
    # 8 catches as a freshman) lost exactly 35.0 points (trajectory
    # 100 -> 0 at its 0.35 weight) and TE Spearman fell .384 -> .319.
    # A pre-role season cannot drag this leg's level components (the
    # dominator term is a max over seasons; the age term keys on the
    # first CROSSING), so the future-outlook principle is kept: the
    # early season serves only as the growth origin, never as a
    # dragged average. The leg's M1 volume None-exit still applies.
    breakout_score      = calc_breakout_score(seasons, age, pos,
                                              _inseason_year=_inseason_year,
                                              _draft_year=dy)

    athleticism_score   = calc_athleticism_score(ath, pos)
    # v3.3 three-zone athleticism transform (class-wide; supersedes the
    # v3.2 R7-A3 tested-QB-only compression, which is removed so nothing
    # double-applies). Testing is a red-flag check, not a gate: the raw
    # component stands at full strength at and below 60, the competent-to-
    # very-good range is compressed (slope 0.70 from 60 to 85), and
    # exceptional testing keeps a bounded toe above 85 (slope 0.90 from the
    # 77.5 intercept, so a perfect raw 100 lands at 91.0).
    if athleticism_score is not None:
        _r = athleticism_score
        if _r <= 60.0:
            pass
        elif _r <= 85.0:
            athleticism_score = 60.0 + (_r - 60.0) * 0.70
        else:
            athleticism_score = 77.5 + (_r - 85.0) * 0.90
    # v3.12 (RB-2b change 3): at RB the transformed value is then
    # re-anchored to the frozen RB pool's percentile (see
    # RB_ATH_POOL): the leg's value becomes where the print sits
    # among priced RBs, so an elite print prices elite (Barkley's
    # 73.49 -> 89.25) instead of mid-pack. RB only; the payload
    # and the blend read the same re-anchored value, like the
    # competition discount's contract.
    if pos == "RB" and athleticism_score is not None:
        athleticism_score = _rb_ath_reanchor(athleticism_score)
    # v3.6 (round-2 audit F15): the complete-mode athleticism floor is
    # DELETED. It imputed a constant 67.0 to untested players with
    # production >= 90 and breakout >= 75 - the one place the model
    # invented a component value instead of redistributing, against
    # its own missing-data doctrine, and it created a combine-day
    # discontinuity (the imputed 67 replaced by a real score, better
    # or worse, the day testing posts). v3.4 already suppressed the
    # floor in-season; now redistribution carries untested players in
    # both modes. _athleticism_projected stays in the payload, always
    # False: nothing is projected anymore.
    _athleticism_projected = False
    competition_score   = calc_competition_score(
        seasons, team_context=prospect.get("team_context"),
    )
    recruiting_score    = calc_recruiting_score(prospect.get("recruiting"))
    environment_score   = calc_environment_adjustment(seasons, pos)
    durability_score    = calc_durability_score(seasons, position=pos,
                                                _inseason_year=_inseason_year,
                                                _today=_today)

    # v3.11 round 2 (PROD_CEILING_CFG): below the ceiling's
    # competition band the production leg's printable value is
    # capped before the discount prices it - a percentage toll
    # on a level-inflated spike leaves a spike (see the cfg).
    if (PROD_CEILING_CFG.get("enabled")
            and pos in (PROD_CEILING_CFG.get("positions") or ())
            and production_score is not None
            and competition_score is not None):
        _p_cap = _production_level_cap(competition_score)
        if _p_cap is not None and production_score > _p_cap:
            production_score = _p_cap

    # v3.9 competition discount (see COMPETITION_CFG): the
    # absolute-output legs are priced against the level they were
    # achieved at, before weights, redistribution or blending see
    # them - the discounted values are the legs (payload,
    # percentiles and blend all read the same numbers).
    # Utilization is exempt unless the cfg arms it (a share is a
    # share at any level).
    if (float(COMPETITION_CFG.get("prod_discount", 0.0) or 0.0) > 0.0
            and competition_score is not None):
        _c_floor = float(COMPETITION_CFG.get("discount_floor", 70.0))
        _c_gap = max(0.0, (_c_floor - competition_score) / _c_floor)
        _c_pow = float(COMPETITION_CFG.get("discount_power", 1.0) or 1.0)
        if _c_pow != 1.0 and _c_gap > 0.0:
            # Boundary-band bend (see COMPETITION_CFG): concave
            # inside [discount_band, discount_floor], continuous
            # with the linear depth at the band's lower edge,
            # the original slope untouched below it.
            _c_band = float(COMPETITION_CFG.get(
                "discount_band", 50.0))
            _c_bgap = max(0.0, (_c_floor - _c_band) / _c_floor)
            if 0.0 < _c_bgap and _c_gap < _c_bgap:
                _c_gap = _c_bgap * (_c_gap / _c_bgap) ** _c_pow
        _c_factor = 1.0 - float(COMPETITION_CFG["prod_discount"]) * _c_gap
        if production_score is not None:
            production_score = _clip(production_score * _c_factor)
        if utilization_score is not None and COMPETITION_CFG.get(
                "discount_utilization", True):
            utilization_score = _clip(utilization_score * _c_factor)
        if efficiency_score is not None and COMPETITION_CFG.get(
                "discount_efficiency", True):
            efficiency_score = _clip(efficiency_score * _c_factor)

    # v3.7: there is no capital leg. The draft_capital argument is
    # accepted for call compatibility and never read: no pick (actual,
    # projected, consensus or seed) and no mock-draft field enters the
    # grade. Through v3.6 the leg was priced by the position pick curve
    # behind projected_draft_capital_score, with provenance gating
    # (v3.6 F1: seed-only picks counted as missing) and the curve
    # standing alone (v3.6 F4). All of that is moot now: below, where
    # the weights are built, the capital entry is excluded for EVERY
    # prospect through the same missing-leg redistribution the model
    # already used for a prospect with no pick, so the capital weight
    # spreads proportionally over the talent legs. (At 3.7's first
    # build that made every grade exactly the 3.6 talent grade; the
    # F12 weights fold then re-weighted the blend - the shipped
    # tables carry no capital entry, so the pop below is a no-op on
    # the default path and the grades are the gated fitted set's.)

    #  Calculate new enhancement features
    experience_score = calc_experience_score(seasons, pos)
    # calc_late_round_upside's only input was the projected round; with
    # no capital concept it is always 0.0. It never fed the grade (the
    # add-on was removed in v3.2 R7); the payload field is kept.
    late_round_upside = calc_late_round_upside(None, seasons, pos)
    # The athleticism x draft-capital interaction feature is likewise
    # payload-only (it never fed the grade); its capital input is 0.0
    # because no capital score exists anymore.
    interaction_features = calc_interaction_features(
        production_score or 0.0, efficiency_score or 0.0,
        athleticism_score or 0.0, 0.0
    )

    # Get position-specific weights (optionally overridden by calibrated weights)
    weights_source = position_weights_override or POSITION_WEIGHTS
    pos_weights = dict(weights_source.get(pos, POSITION_WEIGHTS["WR"]))

    # v3.7: the pre-v3.7 actual-pick weight boost (+0.06 to the capital
    # weight post-draft) is gone with the leg itself; there is no
    # capital weight left to boost.

    # Neutralize missing athleticism: redistribute its weight to other
    # components so a lack of combine data doesn't drag the score down.
    # Keep as None (not 0.0) so the DB stores NULL and the UI hides it.
    _weights = dict(pos_weights)
    _athleticism_missing = athleticism_score is None
    if _athleticism_missing:
        ath_w = _weights.pop("athleticism", 0.0)
        remaining = sum(_weights.values())
        if remaining > 0 and ath_w > 0:
            for k in _weights:
                _weights[k] *= (1.0 + ath_w / remaining)
        # athleticism_score stays None -> DB NULL -> UI hides the bar

    # v3.2 V6c: QB breakout carries no signal; exclude and redistribute.
    if pos == "QB" and "breakout" in _weights:
        bk_w = _weights.pop("breakout", 0.0)
        if bk_w > 0:
            _weights["production"] = _weights.get("production", 0.0) + bk_w / 2.0
            _weights["efficiency"] = _weights.get("efficiency", 0.0) + bk_w / 2.0

    # v3.2 V2/V7: missing utilization and missing age are missing data,
    # not neutral scores. A stored targets value below receptions is a
    # pipeline failure (audit section 0), so it counts as missing too.
    _ls_v32 = _latest_season(seasons) or {}
    _tgt_v32 = _safe(_ls_v32.get("targets"))
    _rec_v32 = _safe(_ls_v32.get("receptions"))
    # v3.13 (Phase 7, lever L1): the gate no longer fires at
    # TE. Historical target data is absent pool-wide, so at TE
    # the gate fired for EVERY profile - the utilization leg
    # priced for 0 of 446 TE profiles even though
    # calc_utilization_score computes it from the receptions
    # proxy and the payload displays it (Bowers' payload shows
    # 97.62 for a leg that was never in his grade). A displayed
    # leg must be priced. The gate keeps firing at WR, where
    # the targets data question is a separate decision. See
    # prospect-runs/bowers-mccaffrey-residuals-2026-11-25.md
    # (L1).
    _util_missing = (
        pos in ("WR",)
        and (_tgt_v32 <= 0 or _tgt_v32 < _rec_v32)
    )
    if _util_missing:
        util_w = _weights.pop("utilization", 0.0)
        remaining = sum(_weights.values())
        if remaining > 0 and util_w > 0:
            for k in _weights:
                _weights[k] *= (1.0 + util_w / remaining)
    _age_missing_v32 = age is None
    if _age_missing_v32:
        age_w = _weights.pop("age", 0.0)
        remaining = sum(_weights.values())
        if remaining > 0 and age_w > 0:
            for k in _weights:
                _weights[k] *= (1.0 + age_w / remaining)

    # v3.1: Neutralize missing durability the same way. A fake 60.0 is
    # worse than honest NULL - it drags down durable players and inflates
    # injury-prone ones. Redistribute the weight when GP data is absent.
    _durability_missing = durability_score is None
    if _durability_missing:
        dur_w = _weights.pop("durability", 0.0)
        remaining = sum(_weights.values())
        if remaining > 0 and dur_w > 0:
            for k in _weights:
                _weights[k] *= (1.0 + dur_w / remaining)
        # durability_score stays None -> DB NULL -> UI hides the bar

    # v3.2 R7-A5: missing environment redistributes like durability.
    if environment_score is None:
        env_w = _weights.pop("environment", 0.0)
        remaining = sum(_weights.values())
        if remaining > 0 and env_w > 0:
            for k in _weights:
                _weights[k] *= (1.0 + env_w / remaining)

    # v3.6 (round-2 audit F10): missing recruiting redistributes too.
    # It was the lone exception, imputing a flat 50.0 at full weight.
    if recruiting_score is None:
        rec_w = _weights.pop("recruiting", 0.0)
        remaining = sum(_weights.values())
        if remaining > 0 and rec_w > 0:
            for k in _weights:
                _weights[k] *= (1.0 + rec_w / remaining)

    # v3.2: missing draft capital redistributes the same way. The remaining
    # weights renormalize to the total they carried with capital included,
    # so the grade stays on the same scale with capital simply absent.
    # v3.7: this exclusion is now UNCONDITIONAL - capital is "missing"
    # for every prospect by design, which is exactly the redistribution
    # above. It also covers foreign overrides that still carry a
    # draft_capital entry (a legacy calibrated set, say): the key is
    # popped and spread the same way. The shipped POSITION_WEIGHTS
    # carry no capital entry at all (F12 fitted set), so on the
    # default path the pop is a no-op.
    dc_w = _weights.pop("draft_capital", 0.0)
    remaining = sum(_weights.values())
    if remaining > 0 and dc_w > 0:
        for k in _weights:
            _weights[k] *= (1.0 + dc_w / remaining)

    # v3.8 (M1): the season legs are Optional now. A leg that found
    # no measurable evidence in the profile's seasons (placeholder /
    # fragment rows, missing team totals, absent schedule data)
    # returns None and its weight redistributes exactly like
    # athleticism / durability / recruiting above. The leg stays None
    # in the payload -> DB NULL -> UI hides it.
    def _pop_missing(_name, _missing):
        if not _missing:
            return
        _w = _weights.pop(_name, 0.0)
        _rem = sum(_weights.values())
        if _rem > 0 and _w > 0:
            for _k in _weights:
                _weights[_k] *= (1.0 + _w / _rem)

    # v3.8 (TE fix, post-gate): the multi-None anchor. When three or
    # more of the four PERFORMANCE legs (production, utilization,
    # efficiency, breakout) are missing, the profile carries no
    # measured performance at all, and full redistribution prices the
    # player entirely on context (age, competition, recruiting) -
    # a QB-convert tight end with zero career catches floated to
    # 69.4 on offense/recruiting context alone. The model's own
    # doctrine for an empty profile is neutral (an empty season list
    # prices production at 52.0), so the missing performance core is
    # anchored at neutral in the grade sum instead of popped: the
    # legs stay None in the payload (UI hides them), but their
    # weight prices at 52.0. One or two missing legs still
    # redistribute - partial evidence is real evidence.
    _perf_missing = sum(1 for _v in (production_score, utilization_score,
                                     efficiency_score, breakout_score)
                        if _v is None)
    _anchor_perf = _perf_missing >= 3

    if not (_anchor_perf and production_score is None):
        _pop_missing("production", production_score is None)
    if not (_anchor_perf and utilization_score is None):
        _pop_missing("utilization", utilization_score is None)
    if not (_anchor_perf and efficiency_score is None):
        _pop_missing("efficiency", efficiency_score is None)
    if not (_anchor_perf and breakout_score is None):
        _pop_missing("breakout", breakout_score is None)
    _pop_missing("competition", competition_score is None)

    # Base prospect score with position-specific weights.
    # v3.0 R6: Use _weights (athleticism redistribution applied) so the
    # redistributed weight actually takes effect. Previously the sum used
    # pos_weights, silently dropping the redistribution and capping the
    # max score when athleticism was NULL.
    # Anchored legs (multi-None rule above) price at neutral in the
    # sum; every other None prices at 0.0 under a popped weight.
    _sum_production  = 52.0 if (_anchor_perf and production_score is None)  else production_score
    _sum_utilization = 52.0 if (_anchor_perf and utilization_score is None) else utilization_score
    _sum_efficiency  = 52.0 if (_anchor_perf and efficiency_score is None)  else efficiency_score
    _sum_breakout    = 52.0 if (_anchor_perf and breakout_score is None)    else breakout_score

    # v3.9 (ceiling redesign): the final grade is blended from the
    # priced leg list by _blend_grade (see BLEND_CFG above
    # POSITION_WEIGHTS). Every leg that survived the missing-leg
    # redistribution above enters with its redistributed weight and
    # its priced value (anchored performance legs at neutral 52.0);
    # a leg whose weight was popped is absent, exactly as the old
    # explicit sum priced it at 0.0 under a zero weight. The QB
    # experience leg is priced in the same list (v3.6 F16: from the
    # redistributed weights, like every other component).
    _priced = []
    for _leg_name, _leg_value in (
        ("production", _sum_production),
        ("utilization", _sum_utilization),
        ("efficiency", _sum_efficiency),
        ("age", age_score),
        ("breakout", _sum_breakout),
        ("athleticism", athleticism_score),
        ("competition", competition_score),
        ("environment", environment_score),
        ("durability", durability_score),
        # v3.7: no draft-capital term - the leg does not exist.
        ("recruiting", recruiting_score),
    ):
        _leg_w = _weights.get(_leg_name, 0.0)
        if _leg_w > 0 and _leg_value is not None:
            _priced.append((_leg_name, _leg_w, float(_leg_value)))
    if (pos == "QB" and experience_score is not None
            and _weights.get("experience", 0.0) > 0):
        _priced.append(("experience", _weights["experience"],
                        float(experience_score)))

    # v3.12 (RB-2b changes 2 + 5): the RB priced list gets the
    # funding deltas, the p40 missing-athleticism fill and the
    # asymmetric receiving leg (see _rb2b_priced_list). The
    # payload legs are untouched: a filled or absent leg still
    # reads None in the payload, so the UI hides what was never
    # measured - the fill is a blend treatment, not a claim.
    if pos == "RB":
        _priced = _rb2b_priced_list(_priced, seasons, pos_weights)

    # v3.13: the ceiling-equalization priced-list transforms
    # (see _v313_priced_list). Like the RB-2b treatments
    # above, these are blend-stage repricings over pinned
    # frozen pools; the payload legs keep their computed
    # values.
    _priced = _v313_priced_list(_priced, pos, seasons)

    # v3.11 round 2 (TRIM_MEASURED_CFG): the trim forgives absence,
    # not measured weakness. When the profile carries at least one
    # role (featured) season, utilization was priced from the full
    # featured body of work - a low value there is a measurement
    # (a split backfield, a demotion), not an unchecked box, so it
    # is blocked from the trim drop for this profile. A profile
    # with no featured season priced everything from scraps;
    # nothing was ever measured in a real role, and forgiveness
    # keeps its original meaning there. Every other leg stays
    # droppable either way (see the cfg for the scoping evidence).
    # The case that forced the rule: Javonte Williams' utilization
    # 43.7 - three featured UNC seasons in which his own coaches
    # split the backfield with Michael Carter - was his trim
    # drop, and forgiving it crowned a very good back the
    # decade's RB max (92.32) over Barkley and McCaffrey.
    _trim_block = ()
    if TRIM_MEASURED_CFG.get("enabled"):
        if any(_is_role_season(_s, pos) for _s in seasons):
            _trim_block = tuple(TRIM_MEASURED_CFG.get("legs") or ())
    # v3.12 (RB-2b change 4): at RB, efficiency is trim-protected
    # outright. The gated trim pool-normally deletes the one
    # skeptical leg at RB - efficiency was the trim drop for
    # Abanikanda (69.35 -> contributed 0.00), Hall and Barkley -
    # so the designated skeptic was systematically the leg the
    # blend forgave. Measured performance legs were already
    # untrimmable under TRIM_MEASURED_CFG when a role season
    # exists; at RB the protection is unconditional, so the
    # skeptic is paid in the mean always.
    if pos == "RB":
        _trim_block = tuple(_trim_block) + ("efficiency",)

    prospect_score = _blend_grade(_priced, pos,
                                  allow_peak=not _anchor_perf,
                                  trim_block=_trim_block)

    # v3.2 R7: no late-round upside add-on. The final grade is the
    # weighted component sum plus the bounded translation adjustment only.

    # v3.2 R7: benchmark boosts are NOT applied. They double-count inputs
    # already inside the components (draft capital, dominator, breakout,
    # production, age, market share, competition, athleticism) and then add
    # a flat +6 benchmark layer on top. An absolute scale cannot carry a
    # second, multiplicative opinion about the same facts.
    benchmark_boosts = {"total_boost": 0.0}

    # Position-specific translation adjustment from historical miss archetypes.
    # Applied late (after global scaling) so the correction magnitude is preserved.
    translation_adjustment = calc_translation_adjustment(
        prospect=prospect,
        position=pos,
        # v3.6 (F3) deleted every pick-gated branch from translation;
        # v3.7 passes nothing at all - score_prospect forwards no
        # capital data anywhere.
        draft_capital=None,
        production_score=production_score,
        efficiency_score=efficiency_score,
        age_score=age_score,
    )
    prospect_score += translation_adjustment

    # v3.2 R7: no thin-sample final multiplier. Sample reliability is
    # reported by confidence_score; it does not rescale the grade.

    # Final clamp to the documented 0-100 grade scale. The weighted sum
    # is bounded by construction, but the additive translation_adjustment
    # is applied afterwards, so a profile at the top of the scale could
    # otherwise finish above 100, breaking the grade scale and the
    # fantasy-scaled overall_score derived from it.
    prospect_score = _clip(prospect_score, 0.0, 99.9)  # v3.2 V9: no 100s

    prospect_score = round(prospect_score, 2)

    # v3.2-v3.6 carried a transparent-capital pair here: talent_score
    # (this profile re-graded with capital excluded) and capital_delta
    # (the leg's price). v3.7 makes the exclusion the only path, so
    # the grade IS the talent grade: talent_score is the grade itself
    # and capital_delta is None - permanently, for every prospect,
    # because there is no capital effect to report. The old
    # missing-capital contract (talent == grade, delta None) is now
    # simply the contract.
    talent_score = prospect_score
    capital_delta = None

    # Quality-weighted confidence score.
    # Each data source is weighted by how much it improves scoring accuracy.
    # v3.7: the pre-v3.7 capital-presence points (+10 for at least one
    # mock behind the pick, +5 more at 5+ mocks; gated to verified
    # capital by v3.6 F13) are removed - consensus/mock data goes with
    # capital, so confidence now certifies only the talent evidence:
    # seasons, dominator, games, targets, age and athleticism.
    ls_check = _latest_season(seasons) or {}
    conf_points = 0.0
    if seasons:
        conf_points += 15.0  # any season data
        if len(seasons) >= 2:
            conf_points += 10.0  # multi-season data (more reliable trends)
        if ls_check.get("dominator_rating"):
            conf_points += 10.0  # dominator rating available (key metric)
        if ls_check.get("games_played"):
            conf_points += 5.0   # exact games played (not default 12)
        if ls_check.get("targets") is not None:
            conf_points += 5.0   # target data (WR/TE evaluation quality)
    if age is not None:
        conf_points += 15.0  # age known (critical for age_score reliability)
    if ath:
        conf_points += 15.0  # any athleticism data
        if len(ath) >= 4:
            conf_points += 10.0  # full combine profile
    confidence_score = round(_clip(conf_points), 1)
    # v3.4: in-season confidence reflects the thinner evidence. The
    # partial season's sample discount is 20 points at the limit, scaled
    # by the same min(1, GP/12) idiom the model uses elsewhere: a 6-game
    # latest season costs 10 points, an 11-game season costs about 2.
    if _inseason:
        _gp_conf = _safe(_gp_mode, 12)
        confidence_score = round(_clip(
            confidence_score
            - round(20.0 * (1.0 - min(1.0, _gp_conf / 12.0)))), 1)

    fantasy_translation = round(
        POSITION_FANTASY_MULT.get(pos, 1.0) * prospect_score, 2
    )

    # v3.2 archetype: scored NFL-outlook label from the isolated archetype
    # module (v2 rule set). Label only; it never feeds back into the grade.
    try:
        from data_building.rookie_pipeline.prospect_archetypes import classify_archetype
        _arch_profile = {
            "weight_lbs": ath.get("weight_lbs") or prospect.get("weight_lbs"),
            "forty_yard": ath.get("forty_yard"),
            # v3.7: no projected_pick - the labeler receives no pick
            # either, so labels are capital-free like the grade. (The
            # archetype module reads no pick at all: it is not a fit
            # input, gate, or presence feature, and no production
            # caller passes one.)
            "age": age,
        }
        _archetype = classify_archetype(
            pos, seasons, production_score=production_score,
            profile=_arch_profile)
    except Exception:
        _archetype = {"archetype": None, "archetype_receipt": None,
                      "archetype_confidence": None}

    # Human-readable reasons (v3.7: no capital bullet exists to build)
    reasons = _build_reasons(
        prospect, pos, seasons,
        production_score, utilization_score, efficiency_score, age_score,
        breakout_score, athleticism_score, competition_score,
        environment_score, durability_score,
    )

    return {
        "player_id":                    prospect["player_id"],
        "draft_class_year":             dy,
        # v3.8 (M1): missing legs stay None (DB NULL, UI hides them),
        # the convention athleticism / durability / recruiting set.
        "production_score":             None if production_score is None else round(production_score, 2),
        "utilization_score":            None if utilization_score is None else round(utilization_score, 2),
        "efficiency_score":             None if efficiency_score is None else round(efficiency_score, 2),
        "age_score":                    round(age_score, 2),
        "breakout_profile_score":       None if breakout_score is None else round(breakout_score, 2),
        "athleticism_score":            None if athleticism_score is None else round(athleticism_score, 2),
        "athleticism_projected":        _athleticism_projected,
        "competition_score":            None if competition_score is None else round(competition_score, 2),
        # v3.8 (M1): "truncated" when no season with counting stats
        # survived the placeholder filter - the profile's grades rest
        # on neutral paths, and the reference-pool quarantine keys
        # off the same word.
        "data_quality":                 _data_quality,
        "environment_adjustment":       None if environment_score is None else round(environment_score, 2),
        "durability_score":             None if durability_score is None else round(durability_score, 2),
        "recruiting_score":             None if recruiting_score is None else round(recruiting_score, 2),
        "model_version":                MODEL_VERSION,
        "in_season":                    _inseason,
        "grading_mode":                 "in_season" if _inseason else "complete",
        # v3.7: always None - there is no capital component. The key
        # is kept so the payload shape (and its consumers) is stable.
        "projected_draft_capital_score":None,
        "talent_score":                 talent_score,
        "capital_delta":                capital_delta,
        "archetype":                    _archetype.get("archetype"),
        "archetype_receipt":            _archetype.get("archetype_receipt"),
        "archetype_confidence":         _archetype.get("archetype_confidence"),
        "fantasy_translation_score":    round(fantasy_translation, 2),
        "confidence_score":             confidence_score,
        "prospect_score":               prospect_score,
        "key_reasons":                  reasons,
        "experience_score":             None if experience_score is None else round(experience_score, 2),
        "late_round_upside":             round(late_round_upside, 2),
        "translation_adjustment":        round(translation_adjustment, 2),
        "loaded_roster_adjustment":      round(loaded_roster_adjustment, 3),
        "depth_chart_adjustment":        round(depth_chart_adjustment, 3),
        "depth_rank":                    ls.get("depth_rank"),
        "position_group_size":           ls.get("position_group_size"),
        "production_efficiency_interaction": round(interaction_features["production_efficiency_interaction"], 2),
        "athleticism_draft_capital_interaction": round(interaction_features["athleticism_draft_capital_interaction"], 2),
        "production_athleticism_interaction": round(interaction_features["production_athleticism_interaction"], 2),
        "benchmark_boosts":             benchmark_boosts,
        "total_benchmark_boost":        round(benchmark_boosts.get("total_boost", 0.0), 3),
        "triple_interaction":           round(interaction_features["triple_interaction"], 2),
    }


def _build_reasons(
    prospect, pos, seasons,
    prod, util, eff, age_sc, break_sc, ath_sc, comp_sc,
    env_sc, dur_sc
) -> str:
    """Build a bullet-point string summarising the prospect's strengths/flags."""
    bullets: List[str] = []
    ls    = _latest_season(seasons) or {}
    age   = prospect.get("age")
    gp    = max(_safe(ls.get("games_played"), 12), 1)

    # ── Production ────────────────────────────────────────────────────────────
    # v3.8: a missing leg (None) earns no bullet - never describe
    # unmeasured production as limited production.
    if prod is not None:
        if prod >= 75:
            bullets.append("Elite production profile - dominant volume for their position")
        elif prod >= 55:
            bullets.append("Solid production numbers with room to grow at the NFL level")
        else:
            bullets.append("Limited production volume - may need time to develop")

    # ── Utilization ───────────────────────────────────────────────────────────
    if pos in ("WR", "TE"):
        tpg = (_safe(ls.get("targets")) or _safe(ls.get("receptions"))) / gp
        if util is not None and util >= 70:
            bullets.append(f"High target share ({tpg:.1f} tgts/game) - clear featured receiver")
        elif util is not None and util <= 35 and seasons:
            bullets.append(f"Limited target volume ({tpg:.1f} tgts/game) - role player or committee")
    elif pos == "RB":
        carries = _safe(ls.get("rush_attempts")) / gp
        if util is not None and util >= 70:
            bullets.append(f"Workhorse RB usage ({carries:.1f} carries/game)")
        elif util is not None and util <= 35:
            bullets.append(f"Limited rush volume ({carries:.1f} carries/game) - situational role")
    elif pos == "QB":
        att_pg = _safe(ls.get("pass_attempts")) / gp
        if util is not None and util >= 70:
            bullets.append(f"High-volume passing attack ({att_pg:.0f} att/game)")

    # ── Efficiency ────────────────────────────────────────────────────────────
    if eff is not None:
        if eff >= 75:
            bullets.append("High efficiency metrics stand out")
        elif eff <= 35:
            bullets.append("Below-average efficiency - volume stats may overstate true impact")

    # ── Advanced metrics ──────────────────────────────────────────────────────
    if pos in ("WR", "TE"):
        adv: List[str] = []

        ccr = ls.get("contested_catch_rate")
        if ccr is not None:
            pct = float(ccr)
            if pct >= 80:
                adv.append(f"{pct:.0f}% contested catch rate - elite ball-winner in traffic")
            elif pct >= 65:
                adv.append(f"{pct:.0f}% contested catch rate - reliable in contested situations")
            elif pct < 40:
                adv.append(f"{pct:.0f}% contested catch rate - struggles in jump-ball situations")

        # drop_rate bullets removed: was PFF-only, no free replacement.

        yac = ls.get("yards_after_catch_per_reception")
        if yac is not None:
            yac = float(yac)
            thresh     = 5.5 if pos == "WR" else 4.0
            low_thresh = 2.5 if pos == "WR" else 1.8
            if yac >= thresh:
                adv.append(f"{yac:.1f} YAC/reception - dynamic after the catch")
            elif yac <= low_thresh:
                adv.append(f"{yac:.1f} YAC/reception - limited after-catch production")

        adot = ls.get("avg_depth_of_target")
        if adot is not None and pos == "WR":
            adot = float(adot)
            if adot >= 14.0:
                adv.append(f"{adot:.1f}-yd avg depth of target - true deep threat")
            elif adot <= 6.0:
                adv.append(f"{adot:.1f}-yd aDOT - short-area route specialist")

        # grades_offense bullets removed: was PFF-only, no free replacement.

        if pos == "WR":
            sr = ls.get("slot_rate")
            if sr is not None and float(sr) >= 0.65:
                adv.append(f"{float(sr):.0f}% slot rate - primary slot receiver")

        if pos == "TE":
            ir = ls.get("inline_rate")
            if ir is not None:
                ir = float(ir)
                if ir >= 0.60:
                    adv.append(f"{ir:.0f}% inline rate - traditional in-line TE")
                elif ir <= 0.20:
                    adv.append(f"{ir:.0f}% inline rate - move TE / receives in space")

        bullets.extend(adv[:3])

    elif pos == "RB":
        adv = []

        # grades_offense bullets removed: was PFF-only, no free replacement.
        # elusive_rating bullets removed: was PFF-only, no free replacement.

        bp = ls.get("breakaway_percentage")
        if bp is not None:
            bpct = float(bp)
            if bpct >= 18.0:
                adv.append(f"{bpct:.0f}% breakaway run rate - consistent big-play threat")
            elif bpct >= 12.0:
                adv.append(f"{bpct:.0f}% breakaway run rate - shows explosive burst")

        bullets.extend(adv[:3])

    elif pos == "QB":
        adv = []

        # pff_passing_grade is now sourced from CFBD PPA; reword bullet accordingly.
        pff_pass = ls.get("pff_passing_grade")
        if pff_pass is not None:
            pff_pass = float(pff_pass)
            if pff_pass >= 85.0:
                adv.append(f"Passing efficiency grade {pff_pass:.1f} - elite passer grade")
            elif pff_pass >= 75.0:
                adv.append(f"Passing efficiency grade {pff_pass:.1f} - above-average")

        # adjusted_completion_rate bullets removed: was PFF-only, no free replacement.
        # big_time_throw_rate bullets removed: was PFF-only, no free replacement.

        adot = ls.get("avg_depth_of_target")
        if adot is not None:
            adot = float(adot)
            if adot >= 10.0:
                adv.append(f"{adot:.1f}-yd avg depth of target - attacks full field vertically")
            elif adot <= 6.0:
                adv.append(f"{adot:.1f}-yd aDOT - relies heavily on short/underneath game")

        bullets.extend(adv[:3])

    # ── Dominator rating ──────────────────────────────────────────────────────
    dom = _safe(ls.get("dominator_rating"))
    if dom >= 0.35 and pos in ("WR", "RB"):
        bullets.append(f"Team dominator rating {dom:.0%} - commanded an outsized share of team production")
    elif dom >= 0.20 and pos == "TE":
        bullets.append(f"Strong target share for a TE - {dom:.0%} team dominator")

    # ── Red zone proxy ────────────────────────────────────────────────────────
    if pos in ("WR", "TE"):
        total_yds = _safe(ls.get("receiving_yards"))
        total_tds = _safe(ls.get("receiving_tds"))
        if total_yds >= 200:
            rz_rate = total_tds / total_yds * 100
            if rz_rate >= 7.0:
                bullets.append(f"High TD rate ({rz_rate:.1f} TDs/100 yds) - dangerous in red zone")
    elif pos == "RB":
        total_yds = _safe(ls.get("rush_yards")) + _safe(ls.get("receiving_yards"))
        total_tds = _safe(ls.get("rush_tds"))   + _safe(ls.get("receiving_tds"))
        if total_yds >= 300:
            rz_rate = total_tds / total_yds * 100
            if rz_rate >= 5.5:
                bullets.append(f"High TD rate ({rz_rate:.1f} TDs/100 yds) - goal-line threat")

    # ── Age / adjusted breakout age ───────────────────────────────────────────
    if age_sc >= 80:
        bullets.append(f"Exceptional age-adjusted production - elite output very young ({age:.1f} yrs)")
    elif age_sc < 40 and age is not None:
        bullets.append(f"Age concern: {age:.1f} yrs is older than typical for this position")

    if break_sc is not None and break_sc >= 75:
        bullets.append("Clear breakout trajectory - early-career dominance at a young age")
    elif break_sc is not None and break_sc >= 60:
        bullets.append("Solid breakout profile - consistent improvement season-over-season")

    # ── Athleticism ───────────────────────────────────────────────────────────
    ath      = prospect.get("athleticism", {}) or {}
    ras      = ath.get("ras_score")
    forty    = ath.get("forty_yard")
    tc       = ath.get("three_cone")
    shuttle  = ath.get("short_shuttle")
    ath_msgs = []
    if ras is not None and _safe(ras) >= 9.0:
        ath_msgs.append(f"RAS {_safe(ras):.1f}/10")
    if forty is not None and _safe(forty) <= 4.35 and pos in ("WR", "RB"):
        ath_msgs.append(f"{_safe(forty):.2f}s 40-yd")
    if tc is not None and _safe(tc) <= 6.55 and pos in ("WR", "RB"):
        ath_msgs.append(f"{_safe(tc):.2f}s 3-cone (elite agility)")
    if ath_msgs:
        bullets.append(f"Elite athleticism: {', '.join(ath_msgs)}")
    elif ath_sc is not None and ath_sc <= 35 and ath:
        bullets.append("Below-average athleticism profile - will need to win on scheme/technique")

    # ── Competition ───────────────────────────────────────────────────────────
    conf = ls.get("conference", "")
    if comp_sc is not None and comp_sc >= 70:
        bullets.append(f"Production came against quality competition")
    elif comp_sc is not None and comp_sc <= 40:
        bullets.append(f"Played at lower competition level ({conf}) - discount applied")

    # ── Transfer history ──────────────────────────────────────────────────────
    if prospect.get("transfer_history"):
        th = prospect["transfer_history"]
        if isinstance(th, list) and len(th) >= 1:
            bullets.append(f"Transfer: {' → '.join(th)} - context matters for production evaluation")
        elif isinstance(th, str):
            bullets.append(f"Transfer history: {th}")

    # v3.7: no Draft capital bullet - reasons describe talent evidence
    # only, like the grade itself.

    # ── Durability ────────────────────────────────────────────────────────────
    if dur_sc is not None and dur_sc <= 40:
        recent_gp = _safe(ls.get("games_played"))
        if recent_gp and recent_gp < 10:
            bullets.append(f"Durability concern: only {int(recent_gp)} games in most recent season")
        else:
            bullets.append("Durability concern: missed significant games in college career")

    # ── Early declare ─────────────────────────────────────────────────────────
    if prospect.get("early_declare"):
        bullets.append("Early declarant - chose to enter draft before exhausting eligibility")

    return "\n".join(f"• {b}" for b in bullets)


# ─────────────────────────────────────────────────────────────────────────────
# Batch scorer
# ─────────────────────────────────────────────────────────────────────────────

def score_all_prospects(
    prospects: List[Dict[str, Any]],
    consensus_map: Optional[Dict[str, Dict]] = None,
    skip_sagarin: bool = False,
    position_weights_override: Optional[Dict[str, Dict[str, float]]] = None,
    active_season_year: Optional[int] = None,
    _today=None,
) -> List[Dict[str, Any]]:
    """
    Score a list of prospects and add overall_rank / position_rank.

    Args:
        prospects:          List of normalised prospect dicts
        consensus_map:      {player_id: consensus_dict}
        active_season_year: The college season in progress on the run
            date (see active_season_year_for_run), enabling v3.4
            in-season mode per player. None (the default, and what the
            backtest passes) grades every profile in COMPLETE mode.
        _today:             The run date, pinned by the pipeline so
            the v3.9 in-season shrink can price season progress
            (_season_games_elapsed). None keeps the pre-3.9 GP-only
            shrink; the backtest passes None.

    Returns:
        List of score dicts, sorted by prospect_score descending.
    """
    if consensus_map is None:
        consensus_map = {}

    scores = []
    for p in prospects:
        dc = consensus_map.get(p["player_id"])
        scores.append(
            score_prospect(
                p,
                dc,
                skip_sagarin=skip_sagarin,
                position_weights_override=position_weights_override,
                _active_season_year=active_season_year,
                _today=_today,
            )
        )

    # Sort overall
    scores.sort(key=lambda x: x["prospect_score"], reverse=True)
    for i, s in enumerate(scores):
        s["overall_rank"] = i + 1

    # Position ranks
    pos_counters: Dict[str, int] = {}
    for s in sorted(scores, key=lambda x: x["prospect_score"], reverse=True):
        # Need position from prospect list
        pid  = s["player_id"]
        pos  = next((p["position"].upper() for p in prospects if p["player_id"] == pid), "UNK")
        pos_counters[pos] = pos_counters.get(pos, 0) + 1
        s["position_rank"] = pos_counters[pos]

    # Calculate tiers based on prospect_score
    for s in scores:
        score = s["prospect_score"]
        if score >= 85:
            s["tier"] = 1
            s["tier_label"] = "Elite Prospect"
        elif score >= 72:
            s["tier"] = 2
            s["tier_label"] = "Top Prospect"
        elif score >= 60:
            s["tier"] = 3
            s["tier_label"] = "Day-2 Upside"
        elif score >= 44:
            s["tier"] = 4
            s["tier_label"] = "Developmental"
        elif score >= 33:
            s["tier"] = 5
            s["tier_label"] = "Deep Flier"
        else:
            s["tier"] = 6
            s["tier_label"] = "Low Priority"

    return scores
