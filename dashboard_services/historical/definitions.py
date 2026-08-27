"""Pure historical-analytics definitions for slim CI.

Tier boundaries, age-as-of-date math, age-bucket maps, draft-capital buckets,
confidence labels, empirical-Bayes smoothing, and bust definitions live here
so they cannot drift across builders, APIs, or the UI. This module must stay
dependency-free (no pandas, Flask, or I/O).
"""
from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Mapping, Optional, Tuple

# Reliable warehouse backbone. Matches NGS availability and the existing
# 2016 gate in nflverse_metrics. Earlier seasons may exist per-stat (PFR
# snaps) but must not be advertised as a uniform "2012+" panel.
RELIABLE_SEASON_FLOOR = 2016
NGS_SEASON_FLOOR = 2016
FTN_SEASON_FLOOR = 2022
# Breakout engine SNAP_COUNT_SEASONS starts at 2022. Earlier snap_share on
# 2018–2022 usage_rows is kept when present; rates that need uniform snap %
# filter to this floor.
SNAP_RELIABLE_FLOOR = 2022

# Age is computed as of this calendar date of the NFL season so buckets
# stay comparable across players and years (not "age today").
SEASON_START_MONTH = 9
SEASON_START_DAY = 1

SKILL_POSITIONS: Tuple[str, ...] = ("QB", "RB", "WR", "TE")

# Display / ranking scoring formats. Column prefixes on warehouse rows.
SCORING_FORMATS: Tuple[str, ...] = ("ppr", "half_ppr", "standard")

# Points column on a canonical player-season row → format.
POINTS_COLUMNS: Mapping[str, str] = {
    "ppr": "ppr_points",
    "half_ppr": "half_ppr_points",
    "standard": "standard_points",
}
PPG_COLUMNS: Mapping[str, str] = {
    "ppr": "ppr_ppg",
    "half_ppr": "half_ppr_ppg",
    "standard": "standard_ppg",
}

# Single source of truth for finish-tier flags. Rank 1 is best.
# A player finishing exactly at the cutoff (e.g. RB12) is inside the tier.
# top_5 is required for "previous RB1 → top-5" repeat rates (Phase 2).
TIER_CUTOFFS: Mapping[str, int] = {
    "top_3": 3,
    "top_5": 5,
    "top_6": 6,
    "top_12": 12,
    "top_24": 24,
    "top_36": 36,
}

# Positional labels: RB1 = ranks 1–12, RB2 = 13–24, WR1 = 1–12, …
# Width is the same at every skill position; the label prefix is the position.
POSITION_TIER_WIDTH = 12

# UI convenience and Phase 4 matching. Exact age is stored; bins are used
# as a matching dimension when present (missing age is omitted, not age 0).
# Inclusive integer age on both ends; None means open.
# RB  <=22 / 23-24 / 25-26 / 27-28 / 29-30 / 31+
# WR  <=22 / 23-24 / 25-27 / 28-30 / 31-32 / 33+
# TE  <=23 / 24-25 / 26-28 / 29-31 / 32+
# QB  <=23 / 24-26 / 27-29 / 30-32 / 33-35 / 36+
_AgeBound = Tuple[Optional[int], Optional[int], str]
AGE_BUCKETS: Mapping[str, Tuple[_AgeBound, ...]] = {
    "RB": (
        (None, 22, "<=22"),
        (23, 24, "23-24"),
        (25, 26, "25-26"),
        (27, 28, "27-28"),
        (29, 30, "29-30"),
        (31, None, "31+"),
    ),
    "WR": (
        (None, 22, "<=22"),
        (23, 24, "23-24"),
        (25, 27, "25-27"),
        (28, 30, "28-30"),
        (31, 32, "31-32"),
        (33, None, "33+"),
    ),
    "TE": (
        (None, 23, "<=23"),
        (24, 25, "24-25"),
        (26, 28, "26-28"),
        (29, 31, "29-31"),
        (32, None, "32+"),
    ),
    "QB": (
        (None, 23, "<=23"),
        (24, 26, "24-26"),
        (27, 29, "27-29"),
        (30, 32, "30-32"),
        (33, 35, "33-35"),
        (36, None, "36+"),
    ),
}

# NFL draft capital. Missing round is unknown (None), never silently undrafted.
DRAFT_CAPITAL_ROUND_1 = "round_1"
DRAFT_CAPITAL_DAY_2 = "day_2"
DRAFT_CAPITAL_DAY_3 = "day_3"
DRAFT_CAPITAL_UNDRAFTED = "undrafted"
DRAFT_CAPITAL_ORDER: Tuple[str, ...] = (
    DRAFT_CAPITAL_ROUND_1,
    DRAFT_CAPITAL_DAY_2,
    DRAFT_CAPITAL_DAY_3,
    DRAFT_CAPITAL_UNDRAFTED,
)

# Career stage from completed seasons before this year (0 = rookie year).
# Missing years_experience is None — never mapped to rookie.
CAREER_STAGE_ROOKIE = "rookie"
CAREER_STAGE_YEAR_2 = "year_2"
CAREER_STAGE_YEAR_3 = "year_3"
CAREER_STAGE_YEAR_4 = "year_4"
CAREER_STAGE_YEAR_5 = "year_5"
CAREER_STAGE_YEAR_6_PLUS = "year_6_plus"
CAREER_STAGE_ORDER: Tuple[str, ...] = (
    CAREER_STAGE_ROOKIE,
    CAREER_STAGE_YEAR_2,
    CAREER_STAGE_YEAR_3,
    CAREER_STAGE_YEAR_4,
    CAREER_STAGE_YEAR_5,
    CAREER_STAGE_YEAR_6_PLUS,
)

# Last-season positional finish as a matching feature. Inclusive lo, exclusive
# hi. ``none`` is *only* for rookies (exp 0) with no prior finish — veterans
# with a missing prior are omitted from this dimension, not labeled ``none``.
PRIOR_FINISH_NONE = "none"
PRIOR_FINISH_BUCKETS: Tuple[Tuple[Optional[int], Optional[int], str], ...] = (
    (1, 6, "top_5"),
    (6, 13, "top_12"),
    (13, 25, "top_24"),
    (25, 37, "top_36"),
    (37, None, "outside_36"),
)
PRIOR_FINISH_ORDER: Tuple[str, ...] = (
    PRIOR_FINISH_NONE,
    "top_5",
    "top_12",
    "top_24",
    "top_36",
    "outside_36",
)

# Comp matching dimensions, coarsest → finest. Relaxation drops from the
# right (finest first) when a cell is below MIN_COMP_CELL_N.
COMP_DIMENSION_ORDER: Tuple[str, ...] = (
    "position",
    "career_stage",
    "draft_capital",
    "prior_finish",
    "age_bucket",
    "target_share",
    "snap_pct",
)
COMP_RELAXATION_ORDER: Tuple[str, ...] = (
    "target_share",
    "snap_pct",
    "age_bucket",
    "draft_capital",
    "career_stage",
    "prior_finish",
)
COMP_BOARD_TIERS: Tuple[str, ...] = ("top_5", "top_12", "top_24")
MIN_COMP_CELL_N = 15
NAMED_EXAMPLES_PER_CELL = 3

# Sleeper reports 999 for undrafted / unpriced. That is missing, never ADP 999.
SLEEPER_UNDRAFTED_ADP = 999.0

# Redraft PPR 1QB source order for a single historical ADP on a player-season.
# Exact PPR (Sleeper) before MFL (compatible 1QB PPR) before generic mixed boards.
# Superflex / TEP historical ADP is not in this list — it does not exist in
# free sources (Yahoo has no season axis; ESPN is mixed; Sleeper 2QB is a
# compatible proxy stored separately, not blended into 1QB hit rates).
ADP_SOURCE_PREFERENCE: Tuple[str, ...] = ("sleeper", "mfl", "espn", "yahoo")

# 12-team overall pick buckets. Inclusive lo, exclusive hi.
ADP_OVERALL_BUCKETS: Tuple[Tuple[Optional[float], Optional[float], str], ...] = (
    (1.0, 13.0, "round_1"),
    (13.0, 25.0, "round_2"),
    (25.0, 37.0, "round_3"),
    (37.0, 49.0, "round_4"),
    (49.0, 61.0, "round_5"),
    (61.0, 85.0, "rounds_6_7"),
    (85.0, 121.0, "rounds_8_10"),
    (121.0, None, "rounds_11_plus"),
)

# Phase 7 live-signal comparison. These are descriptive labels, not a blended
# ranking score. PPG is never converted into a fake P(top-12).
SIGNAL_PROB_ALIGN_DELTA = 0.10
SIGNAL_RANK_ALIGN_SPOTS = 6
SIGNAL_HISTORY_SKEPTICAL_P = 0.25
SIGNAL_HISTORY_BULLISH_P = 0.50
SIGNAL_BOARD_TIER = "top_12"

# Inclusive lo, exclusive hi (None = open). UI convenience only.
# Prior-usage hit rates bin previous-season values; missing is not a bucket.
_RateBound = Tuple[Optional[float], Optional[float], str]
TARGET_SHARE_BUCKETS: Tuple[_RateBound, ...] = (
    (None, 0.10, "<10%"),
    (0.10, 0.15, "10-15%"),
    (0.15, 0.20, "15-20%"),
    (0.20, 0.25, "20-25%"),
    (0.25, None, "25%+"),
)
SNAP_PCT_BUCKETS: Tuple[_RateBound, ...] = (
    (None, 0.40, "<40%"),
    (0.40, 0.60, "40-60%"),
    (0.60, 0.80, "60-80%"),
    (0.80, None, "80%+"),
)
ADOT_BUCKETS: Tuple[_RateBound, ...] = (
    (None, 7.0, "<7"),
    (7.0, 10.0, "7-10"),
    (10.0, 13.0, "10-13"),
    (13.0, None, "13+"),
)
RYOE_BUCKETS: Tuple[_RateBound, ...] = (
    (None, 0.0, "below expected"),
    (0.0, None, "at/above expected"),
)

# Overlay actuals. Names match nflverse_metrics. Missing stays None.
EFFICIENCY_FIELDS: Tuple[str, ...] = (
    "adot",
    "air_yards",
    "ngs_avg_separation",
    "ngs_avg_cushion",
    "ngs_created_separation",
    "ngs_avg_intended_air_yards",
    "ngs_pct_share_intended_air_yards",
    "ngs_avg_yac",
    "ngs_avg_yac_above_expectation",
    "ngs_catch_pct",
    "ngs_rush_yards_over_expected_per_att",
    "ngs_rush_efficiency",
    "ngs_cpoe",
    "ngs_avg_time_to_throw",
    "drop_rate",
    "contested_catch_rate",
    "receiving_epa_per_target",
    "rushing_epa_per_att",
    "passing_epa",
    "racr",
    "pacr",
)

# Sample-size confidence. Inspected against the Phase-1 warehouse; tune later
# if real bucket sizes cluster elsewhere. n < 15 is always low.
#   <15 low / 15–39 moderate / 40–99 good / 100+ strong
CONFIDENCE_LOW = "low"
CONFIDENCE_MODERATE = "moderate"
CONFIDENCE_GOOD = "good"
CONFIDENCE_STRONG = "strong"
CONFIDENCE_THRESHOLDS: Tuple[Tuple[int, str], ...] = (
    (15, CONFIDENCE_LOW),
    (40, CONFIDENCE_MODERATE),
    (100, CONFIDENCE_GOOD),
)

# Absolute bust: an early-round profile that finished outside the listed
# positional rank. Works with zero ADP. ADP-relative bust is defined only
# when a preseason ADP snapshot exists for that player-season (Phase 5+).
ABSOLUTE_BUST_OUTSIDE: Mapping[str, int] = {
    "QB": 24,
    "RB": 24,
    "WR": 36,
    "TE": 12,
}

# Empirical-Bayes default prior strength. Documented so smoothed rates are
# reconstructable: adjusted = (successes + prior_successes) / (n + prior_n).
# The prior rate is the broader position-level rate; prior_n is a small
# pseudo-count so tiny samples shrink toward that rate instead of looking exact.
DEFAULT_BAYES_PRIOR_N = 10


def parse_birth_date(value: Any) -> Optional[date]:
    """Parse a birth date. Accepts date/datetime, ISO ``YYYY-MM-DD``, and
    Sleeper/players_index ``M/D/YYYY`` (also ``MM-DD-YYYY``)."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return None
    text = text.split("T", 1)[0].split(" ", 1)[0]
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m-%d-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def age_as_of_season_start(
    birth_date: Any,
    season: Any,
    *,
    month: int = SEASON_START_MONTH,
    day: int = SEASON_START_DAY,
) -> Optional[float]:
    """Age in years as of ``month/day`` of ``season``, truncated to 1 decimal.

    Truncation (not rounding) matches ``age_from_bday`` so a player three days
    short of a birthday does not jump a year. Returns None when the birth date
    or season is missing/unparseable — never 0.0.
    """
    dob = parse_birth_date(birth_date)
    try:
        year = int(season)
    except (TypeError, ValueError):
        return None
    if dob is None:
        return None
    try:
        as_of = date(year, int(month), int(day))
    except ValueError:
        return None
    days = (as_of - dob).days
    if days < 0:
        return None
    return math.floor((days / 365.25) * 10) / 10


def age_bucket(position: Any, age: Any) -> Optional[str]:
    """Map an exact age onto the position's UI convenience bucket.

    Uses the integer floor of age so 22.9 is still ``<=22``. Missing age or
    an unknown position returns None, not a fake bucket.
    """
    pos = str(position or "").upper()
    bounds = AGE_BUCKETS.get(pos)
    if not bounds:
        return None
    try:
        years = int(math.floor(float(age)))
    except (TypeError, ValueError):
        return None
    if years < 0 or years > 80:
        return None
    for lo, hi, label in bounds:
        if lo is not None and years < lo:
            continue
        if hi is not None and years > hi:
            continue
        return label
    return None


def integer_age(age: Any) -> Optional[int]:
    """Floor of exact age for age-curve bins. Missing/unparseable → None."""
    try:
        years = int(math.floor(float(age)))
    except (TypeError, ValueError):
        return None
    if years < 0 or years > 80:
        return None
    return years


def prior_finish_bucket(
    previous_season_finish: Any,
    *,
    years_experience: Any = None,
) -> Optional[str]:
    """Bucket last season's positional finish for matching.

    Rookies (``years_experience == 0``) with no prior finish → ``none``.
    Veterans with a missing prior → None (dimension omitted, not a fake
    last-place or undrafted-style bucket). Rank 13 is ``top_24``, not ``top_12``.
    """
    finish = _optional_int(previous_season_finish)
    if finish is None:
        exp = _optional_int(years_experience)
        if exp == 0:
            return PRIOR_FINISH_NONE
        return None
    if finish < 1:
        return None
    return value_bucket(finish, PRIOR_FINISH_BUCKETS)


def career_stage(years_experience: Any) -> Optional[str]:
    """Map completed seasons-before-this-year onto a career-stage label.

    0 → ``rookie``, 1 → ``year_2``, …, ≥5 → ``year_6_plus``. Missing
    experience is None, never a fake rookie.
    """
    exp = _optional_int(years_experience)
    if exp is None or exp < 0:
        return None
    if exp == 0:
        return CAREER_STAGE_ROOKIE
    if exp == 1:
        return CAREER_STAGE_YEAR_2
    if exp == 2:
        return CAREER_STAGE_YEAR_3
    if exp == 3:
        return CAREER_STAGE_YEAR_4
    if exp == 4:
        return CAREER_STAGE_YEAR_5
    return CAREER_STAGE_YEAR_6_PLUS


def value_bucket(value: Any, bounds: Any) -> Optional[str]:
    """Map a numeric value onto ``(lo, hi, label)`` bins.

    ``lo`` inclusive, ``hi`` exclusive, ``None`` open. Missing/unparseable
    value returns None — never the first bucket.
    """
    v = _optional_float(value)
    if v is None:
        return None
    for lo, hi, label in bounds:
        if lo is not None and v < lo:
            continue
        if hi is not None and v >= hi:
            continue
        return str(label)
    return None


def years_experience_before_season(
    season: Any,
    draft_year: Any = None,
    *,
    first_season: Any = None,
) -> Optional[int]:
    """Completed NFL seasons *before* ``season``.

    Rookie year (``season == draft_year``) is 0. Missing both draft year and
    first observed season returns None — never a invented 0 for a veteran.
    """
    try:
        s = int(season)
    except (TypeError, ValueError):
        return None
    origin = None
    for raw in (draft_year, first_season):
        if raw is None or raw == "":
            continue
        try:
            origin = int(raw)
            break
        except (TypeError, ValueError):
            continue
    if origin is None:
        return None
    exp = s - origin
    if exp < 0:
        return None
    return exp


def draft_capital_bucket(
    draft_round: Any,
    draft_pick: Any = None,
    *,
    undrafted: bool = False,
) -> Optional[str]:
    """``round_1`` / ``day_2`` / ``day_3`` / ``undrafted``.

    Unknown (missing round, not flagged undrafted) is None. Never infer
    undrafted from a missing field — that would look like a real UDFA.
    """
    if undrafted:
        return DRAFT_CAPITAL_UNDRAFTED
    rnd = _optional_int(draft_round)
    if rnd is None:
        pick = _optional_int(draft_pick)
        if pick is None:
            return None
        rnd = min(7, max(1, (pick - 1) // 32 + 1))
    if rnd <= 0:
        return DRAFT_CAPITAL_UNDRAFTED
    if rnd == 1:
        return DRAFT_CAPITAL_ROUND_1
    if rnd in (2, 3):
        return DRAFT_CAPITAL_DAY_2
    if rnd >= 4:
        return DRAFT_CAPITAL_DAY_3
    return None


def positional_tier_label(position: Any, positional_finish: Any) -> Optional[str]:
    """``RB1`` for ranks 1–12, ``RB2`` for 13–24, etc. None if unranked."""
    pos = str(position or "").upper()
    if pos not in SKILL_POSITIONS:
        return None
    finish = _optional_int(positional_finish)
    if finish is None or finish < 1:
        return None
    band = (finish - 1) // POSITION_TIER_WIDTH + 1
    return f"{pos}{band}"


def tier_flags(positional_finish: Any) -> dict:
    """Boolean flags for each cutoff in ``TIER_CUTOFFS``.

    Unranked players get False for every flag (they did not finish top-N),
    which is distinct from a missing finish used as a *feature* — callers
    that need "unknown" should check the finish itself.
    """
    finish = _optional_int(positional_finish)
    flags = {}
    for name, cutoff in TIER_CUTOFFS.items():
        flags[name] = finish is not None and finish <= cutoff
    return flags


def confidence_label(sample_size: Any) -> Optional[str]:
    """``low`` / ``moderate`` / ``good`` / ``strong``. None if n is missing."""
    n = _optional_int(sample_size)
    if n is None or n < 0:
        return None
    if n < CONFIDENCE_THRESHOLDS[0][0]:
        return CONFIDENCE_LOW
    if n < CONFIDENCE_THRESHOLDS[1][0]:
        return CONFIDENCE_MODERATE
    if n < CONFIDENCE_THRESHOLDS[2][0]:
        return CONFIDENCE_GOOD
    return CONFIDENCE_STRONG


def empirical_bayes(
    successes: Any,
    n: Any,
    prior_successes: Any,
    prior_n: Any = DEFAULT_BAYES_PRIOR_N,
) -> Optional[float]:
    """``(successes + prior_successes) / (n + prior_n)``. None if denom is 0."""
    s = _optional_float(successes)
    nn = _optional_float(n)
    ps = _optional_float(prior_successes)
    pn = _optional_float(prior_n)
    if s is None or nn is None or ps is None or pn is None:
        return None
    denom = nn + pn
    if denom <= 0:
        return None
    return (s + ps) / denom


def normalize_adp(value: Any) -> Optional[float]:
    """Overall ADP pick. ``None`` when missing, non-positive, or Sleeper 999."""
    adp = _optional_float(value)
    if adp is None or adp <= 0:
        return None
    if adp >= SLEEPER_UNDRAFTED_ADP:
        return None
    return adp


def adp_overall_bucket(adp: Any) -> Optional[str]:
    """Map overall ADP onto a 12-team round bucket. Missing → None."""
    return value_bucket(normalize_adp(adp), ADP_OVERALL_BUCKETS)


def is_adp_relative_bust(
    adp_positional: Any,
    positional_finish: Any,
    *,
    cutoff: int = 12,
) -> Optional[bool]:
    """True when drafted inside positional ADP top-N and finished outside.

    None when ADP or finish is missing — not a fake bust. Players drafted
    outside the ADP cutoff are not in this cohort (also None).
    """
    adp_rank = _optional_int(adp_positional)
    finish = _optional_int(positional_finish)
    if adp_rank is None or finish is None:
        return None
    if adp_rank < 1 or cutoff < 1:
        return None
    if adp_rank > cutoff:
        return None
    return finish > cutoff


def source_map_is_usable(
    adp_map: Any,
    *,
    min_players: int = 40,
    max_first: float = 8.0,
    max_twelfth: float = 36.0,
) -> bool:
    """Reject boards that are clearly not a preseason draft (e.g. ESPN 170-wall)."""
    if not isinstance(adp_map, dict):
        return False
    vals = []
    for raw in adp_map.values():
        adp = normalize_adp(raw)
        if adp is not None:
            vals.append(adp)
    vals.sort()
    if len(vals) < min_players:
        return False
    if vals[0] > max_first:
        return False
    twelfth = vals[11] if len(vals) > 11 else vals[-1]
    if twelfth > max_twelfth:
        return False
    return True


def is_absolute_bust(position: Any, positional_finish: Any) -> Optional[bool]:
    """True when the player finished outside the position's absolute-bust bar.

    None when finish or position is missing — we do not call a missing
    season a bust.
    """
    pos = str(position or "").upper()
    cutoff = ABSOLUTE_BUST_OUTSIDE.get(pos)
    finish = _optional_int(positional_finish)
    if cutoff is None or finish is None:
        return None
    return finish > cutoff


def display_percent(rate: Any) -> Optional[int]:
    """Whole-percent display. None stays None; never emit 0 for missing."""
    value = _optional_float(rate)
    if value is None:
        return None
    return int(round(value * 100.0))


def _optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if f != f:  # NaN
            return None
        return int(f)


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return f
