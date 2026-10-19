"""
Prospect model v4 - FOUNDATION SKELETON (not production).

Governing spec: ~/workspace/prospect-model-v4-design.md (Kaedon, 2026-10-17).
The v2.0/v3.x scoring architecture is scrapped; this module is the clean
rebuild. MODEL_VERSION is "4.0-dev" so nothing can mistake its output for
a production grade: the live model remains prospect_model.py (v3.3),
untouched by this module.

What this skeleton implements (the design's architecture):

- Scale semantics (design section 1): the grade is an honest valuation
  anchored to real draft quality. 92+ elite prospect, 80-92 great,
  70-80 the average drafted prospect (second-to-fourth-round caliber),
  below 70 day three / fringe. The bands are the scale's DEFINITION,
  calibrated once against the recorded drafted population; no player
  is ever forced into one.
- Every season-stat leg is priced on RECORDED DISTRIBUTIONS: knot
  anchors loaded from v4_reference.json (produced on Render by
  scripts/derive_v4_reference.py from CFBD data for drafted classes
  2016-2024). A leg whose anchor table is absent is MISSING data: it
  returns None and its weight redistributes over the present legs.
  Nothing is ever defaulted to a plausible number (her standing rule).
- ONE season-selection rule (requirement 1): every component prices
  the same two seasons, chosen once: PEAK = the complete season with
  the highest primary per-game rate for the position; LATEST = the
  most recent season with data. Season sub-scores blend 85% peak /
  15% latest, tempered by a per-season reliability factor. Production
  and efficiency can never price different seasons again (round 16:
  Mensah was priced on his 5-game Miami season in production and his
  500-attempt Duke season in efficiency).
- Supporting cast is a first-class input (requirement 2): a bounded
  adjustment inside QB efficiency (charged for an elite receiving
  corps, credited for a weak one) and inside receiver/RB production
  (charged for elite QB play). Cast indices are computed from the
  same run's graded teammates by score_all_v4; when cast is unknown
  the adjustment is simply absent (it carries no standalone weight,
  so there is nothing to redistribute) and the receipt says so.
- Volume / sample honesty (requirement 4): all season legs are
  per-game rates or ratios; the blend reliability factor is
  min(1, GP/12), and QB efficiency additionally tempers by
  min(1, attempts/300), so a 133-attempt season is not the same
  evidence as a 500-attempt season (round 16).
- Recruiting (requirement 5): priced from the 247 composite via its
  recorded distribution wherever the composite exists; the stars
  table is a fallback that imputes the typical composite for the
  star band and is flagged in the receipt; unknown pedigree is
  MISSING (redistribute), never the same number as a documented
  average pedigree (round 16: 3-star and walk-on both printed 50.0).
- Durability (requirement 6): role-aware, thresholds from PR #2691's
  fix. A season under the 12-game bar counts only when the player
  held a real role in it; backup seasons are not injury seasons.
- Lifecycle states (requirement 7): in_season / pre_combine /
  post_combine, selected per player by data state. The scale never
  changes; the evidence set does. Untested athleticism is missing
  (redistributed) while testing is merely not-yet-available; only a
  settled post-cycle absence keeps the rounds 10-11 floor (67.0).
- Capital (requirement 9): a separate layer over a pure talent
  grade. talent_score never moves when the capital input changes;
  the final grade blends capital at a stated per-position share and
  the receipt prints talent, capital score, and capital_delta.

Deliberate skeleton simplifications (named, to be settled in the
calibration follow-on, not silently): leg weights below are stated
starting weights, not backtest-calibrated; the capital layer uses
the consensus map's precomputed capital score with no round
multipliers or day-3 penalties (v3.3's capital machinery is part of
the scrapped architecture); utilization and QB experience are not
separate legs (opportunity lives in production's share legs, and
attempts live in the efficiency reliability factor); TE catch rate
is not an efficiency sub-leg because no recorded targets
distribution exists (round 15). The knot score vector is unified on
the round 15 pre-registered vector for every leg, athleticism
included; the athleticism MARKS are the recorded drafted-tester
tiers carried over from v3.3, and the three-zone transform is
carried over unchanged.
"""

from __future__ import annotations

import json
import math
import os
from datetime import date as _date
from typing import Any, Dict, List, Optional, Tuple

try:  # same-package import (tests, pipeline)
    from .prospect_model import _conf_quality, DEFAULT_CONF_QUALITY
except ImportError:  # direct module import
    from data_building.rookie_pipeline.prospect_model import (  # noqa: F401
        _conf_quality, DEFAULT_CONF_QUALITY,
    )

MODEL_VERSION = "4.0-dev"

# ─────────────────────────────────────────────────────────────────────
# Scale semantics (design section 1)
# ─────────────────────────────────────────────────────────────────────

BANDS: List[Tuple[float, str, str]] = [
    (92.0, "elite", "Elite Prospect"),
    (80.0, "great", "Great Prospect"),
    (70.0, "average", "Average Prospect"),
    (0.0, "day3", "Day 3 / Fringe"),
]


def band_for(score: Optional[float]) -> Tuple[Optional[str], Optional[str]]:
    """(band_key, band_label) for a grade; (None, None) when ungraded."""
    if score is None:
        return None, None
    for floor, key, label in BANDS:
        if score >= floor:
            return key, label
    return "day3", "Day 3 / Fringe"


# ─────────────────────────────────────────────────────────────────────
# Knot machinery (round 15 pre-registered design)
# ─────────────────────────────────────────────────────────────────────

KNOT_TIERS = ["worst", "p1", "p5", "p10", "p25", "p50",
              "p75", "p90", "p97", "p99", "best"]
KNOT_SCORES = [0, 5, 12, 22, 38, 56, 74, 85, 92, 95, 100]
KNOT_PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 97, 99, 100]
DISPLAY_PERCENTILES = [0, 1, 5, 10, 25, 50, 75, 90, 95, 97, 99, 100]

FULL_SEASON_GAMES = 12     # the model's own full-season constant
MIN_SEASON_GP = 4          # round 15 pool guard (the model's GP guard)
PEAK_BLEND = 0.85          # the one blend rule: 85% peak / 15% latest
UNTESTED_FLOOR = 67.0      # rounds 10-11 settled-absence floor
CAST_ADJ_K = 0.08          # cast adjustment slope (bounded, provisional)
CAST_ADJ_CAP = 5.0         # cast adjustment bound (eval-adjustment style)
INSEASON_CONFIDENCE_CAP = 20   # round 14 confidence discount cap


def knot_score(value: Optional[float], marks: Optional[List[float]],
               scores: Optional[List[float]] = None) -> Optional[float]:
    """Piecewise-linear score for one raw value against recorded marks.

    `marks` are the recorded distribution's tier marks in PERFORMANCE
    order (worst to best) and pair with `scores` ascending. Pricing
    happens in a performance coordinate (the raw value, negated when
    a lower raw value is better, e.g. age or forty time), so both
    orientations share one interpolation.

    Tied marks are recorded truth (discrete stats mass at one value):
    a value sitting exactly on a tied mark earns the HIGHEST tier
    that mark represents, because the value reaches every tier the
    mass spans. A value beyond the best recorded mark earns the top
    score; beyond the worst, the bottom one. Returns None when the
    value or a usable table is missing: no table, no score, never an
    invented one.
    """
    if value is None or not marks or len(marks) < 2:
        return None
    scores = scores or KNOT_SCORES
    if len(marks) != len(scores):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    pairs = [(float(m), float(s)) for m, s in zip(marks, scores)]
    higher_better = pairs[-1][0] >= pairs[0][0]
    sign = 1.0 if higher_better else -1.0
    best: Dict[float, float] = {}
    for m, s in pairs:
        t = sign * m
        if t not in best or s > best[t]:
            best[t] = s
    pts = sorted(best.items())
    tv = sign * v
    if tv <= pts[0][0]:
        return float(pts[0][1])
    if tv >= pts[-1][0]:
        return float(pts[-1][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= tv <= x1:
            return y0 + (y1 - y0) * (tv - x0) / (x1 - x0)
    return float(pts[-1][1])


def percentile_of(value: Optional[float],
                  table: Optional[List[List[float]]]) -> Optional[float]:
    """Percentile (0-100) of `value` in a recorded anchor table.

    `table` is [[percentile, mark], ...] from the reference artifact.
    Linear interpolation between recorded marks; a value on a tied
    mark earns the highest percentile that mark represents (the same
    convention as knot_score); outside the recorded range the
    percentile clamps to 0 / 100.
    """
    if value is None or not table or len(table) < 2:
        return None
    best: Dict[float, float] = {}
    for p, m in table:
        m = float(m)
        if m not in best or float(p) > best[m]:
            best[m] = float(p)
    pts = sorted(best.items())
    v = float(value)
    if v < pts[0][0]:
        return 0.0
    if v >= pts[-1][0]:
        return float(pts[-1][1])
    if v == pts[0][0]:
        return float(pts[0][1])
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= v <= x1:
            return y0 + (y1 - y0) * (v - x0) / (x1 - x0)
    return float(pts[-1][1])


# ─────────────────────────────────────────────────────────────────────
# Reference artifact
# ─────────────────────────────────────────────────────────────────────

_REFERENCE_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "v4_reference.json")


def load_reference(path: Optional[str] = None) -> Dict[str, Any]:
    """Load a v4 reference artifact; {} when absent (every anchor leg
    is then missing data, by design). The fixture sample committed for
    tests is v4_reference_fixture_sample.json; its provenance block
    marks it as fixture data, and receipts carry that marker so a
    fixture-anchored grade can never pose as a recorded one."""
    p = path or _REFERENCE_DEFAULT
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _anchors(reference: Dict[str, Any], pos: str, leg: str):
    return (reference.get("anchors") or {}).get(pos, {}).get(leg)


def _athleticism_marks(reference: Dict[str, Any], pos: str, metric: str):
    return (reference.get("athleticism") or {}).get(pos, {}).get(metric)


def _percentile_table(reference: Dict[str, Any], pos: str, component: str):
    return (reference.get("percentiles") or {}).get(pos, {}).get(component)


# ─────────────────────────────────────────────────────────────────────
# Small helpers (mirrors of the v3.3 conventions)
# ─────────────────────────────────────────────────────────────────────

def _safe(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _has_data(s: Dict) -> bool:
    return any(_safe(s.get(k)) > 0 for k in (
        "receiving_yards", "rush_yards", "pass_yards",
        "receptions", "rush_attempts", "pass_attempts"))


def _latest_season(seasons: List[Dict]) -> Optional[Dict]:
    if not seasons:
        return None
    pool = [s for s in seasons if _has_data(s)] or seasons
    return max(pool, key=lambda s: _safe(s.get("season"), 0))


# ─────────────────────────────────────────────────────────────────────
# Season metrics: the one extraction every leg and the reference
# derivation share, so a leg input means the same thing in the model
# and in the recorded distribution it is priced against.
# ─────────────────────────────────────────────────────────────────────

def season_metrics(season: Dict, position: str) -> Dict[str, Optional[float]]:
    """Per-game rates, ratios, and shares for one season dict.

    Accepts both the pipeline's CFBD season format (ingestion
    ._build_cfbd_season) and the panel-fixture format. A metric is
    None when its inputs are absent: missing stays missing.
    """
    gp_raw = season.get("games_played")
    gp = _safe(gp_raw) if gp_raw is not None else None
    if gp is not None and gp <= 0:
        gp = None

    def rate(total: float) -> Optional[float]:
        return (total / gp) if (gp and total) else (
            0.0 if (gp and total == 0) else None)

    rec = _safe(season.get("receptions"))
    rec_yds = _safe(season.get("receiving_yards"))
    rec_tds = _safe(season.get("receiving_tds"))
    rush_att = _safe(season.get("rush_attempts"))
    rush_yds = _safe(season.get("rush_yards"))
    rush_tds = _safe(season.get("rush_tds"))
    pass_att = _safe(season.get("pass_attempts"))
    pass_yds = _safe(season.get("pass_yards"))
    pass_tds = _safe(season.get("pass_tds"))
    team_pass_yds = _safe(season.get("team_pass_yards"))
    team_total_yds = _safe(season.get("team_total_yards"))

    comp_pct = season.get("completion_pct")
    if comp_pct is None and pass_att > 0 and season.get("completions") is not None:
        comp_pct = 100.0 * _safe(season.get("completions")) / pass_att
    td_int = season.get("td_int_ratio")
    if td_int is None and pass_tds > 0:
        td_int = pass_tds / max(_safe(season.get("interceptions")), 1.0)
    ypa = season.get("yds_per_attempt")
    if ypa is None and pass_att > 0:
        ypa = pass_yds / pass_att
    ypr = season.get("yds_per_reception")
    if ypr is None and rec > 0:
        ypr = rec_yds / rec
    ypc = season.get("yds_per_carry")
    if ypc is None and rush_att > 0:
        ypc = rush_yds / rush_att
    dom = season.get("dominator_rating")
    mshare = season.get("market_share_yards")

    return {
        "games_played": gp,
        "rec_yds_pg": rate(rec_yds),
        "rec_tds_pg": rate(rec_tds),
        "rec_pg": rate(rec),
        "pass_share": (rec_yds / team_pass_yds) if team_pass_yds > 0 else None,
        "dominator_rating": _safe(dom) if dom is not None else None,
        "market_share_yards": _safe(mshare) if mshare is not None else None,
        "ypr": _safe(ypr) if ypr is not None else None,
        "rush_yds_pg": rate(rush_yds),
        "scrimmage_yds_pg": rate(rush_yds + rec_yds),
        "tds_pg": rate(rush_tds + rec_tds),
        "ypc": _safe(ypc) if ypc is not None else None,
        "pass_yds_pg": rate(pass_yds),
        "pass_tds_pg": rate(pass_tds),
        "completion_pct": _safe(comp_pct) if comp_pct is not None else None,
        "ypa": _safe(ypa) if ypa is not None else None,
        "td_int": _safe(td_int) if td_int is not None else None,
        "qb_share": (pass_yds / team_total_yds) if team_total_yds > 0 else None,
        "pass_attempts": pass_att if pass_att > 0 else None,
        "touches": (rush_att + rec) if (rush_att + rec) > 0 else None,
    }


def _primary_rate(season: Dict, pos: str) -> Optional[float]:
    """The position's primary per-game production rate: the single
    definition the season-selection rule peaks on."""
    m = season_metrics(season, pos)
    if pos == "QB":
        if m["pass_yds_pg"] is not None:
            return m["pass_yds_pg"]
        gp = m["games_played"]
        if gp:
            return (_safe(season.get("pass_yards"))
                    + _safe(season.get("rush_yards"))) / gp
        return None
    if pos == "RB":
        return m["scrimmage_yds_pg"]
    return m["rec_yds_pg"]


def select_priced_seasons(seasons: List[Dict], pos: str,
                          partial_year: Optional[int] = None) -> Dict[str, Any]:
    """THE season-selection rule (requirement 1). Every component
    prices the seasons chosen here, and only these:

    - LATEST: the most recent season with data.
    - PEAK: among COMPLETE seasons (never the in-progress partial),
      the one with the highest primary per-game rate; the peak pool
      requires GP >= MIN_SEASON_GP when such a season exists (the
      model's own GP guard, round 15).
    - When no complete season exists, PEAK falls back to LATEST (a
      true freshman's partial season is all the evidence there is).
    """
    data = [s for s in seasons if _has_data(s)]
    if not data:
        return {"peak": None, "latest": None, "partial": False}
    latest = max(data, key=lambda s: _safe(s.get("season"), 0))
    is_partial = (partial_year is not None
                  and _safe(latest.get("season"), 0) == float(partial_year)
                  and 1 <= _safe(latest.get("games_played")) <= 11)
    complete = [s for s in data if s is not latest] if is_partial else list(data)
    pool = [s for s in complete
            if _safe(s.get("games_played")) >= MIN_SEASON_GP] or complete
    peak = None
    best = None
    for s in pool:
        r = _primary_rate(s, pos)
        if r is not None and (best is None or r > best):
            best, peak = r, s
    if peak is None:
        peak = latest
    return {"peak": peak, "latest": latest, "partial": is_partial}


# ─────────────────────────────────────────────────────────────────────
# Leg machinery: price sub-legs per priced season, blend by the one
# rule, redistribute inside the leg when a sub-leg has no anchor or
# no input.
# ─────────────────────────────────────────────────────────────────────

def _season_reliability(season: Dict, pos: str, leg: str) -> float:
    """Sample reliability of one season's evidence: min(1, GP/12),
    and for QB efficiency also min(1, attempts/300), so efficiency
    earned on 133 attempts counts for less than on 500 (round 16)."""
    gp = _safe(season.get("games_played"))
    if gp <= 0:
        return 0.5
    r = min(1.0, gp / FULL_SEASON_GAMES)
    if leg == "efficiency" and pos == "QB":
        att = _safe(season.get("pass_attempts"))
        if att > 0:
            r *= min(1.0, att / 300.0)
    return r


def _price_season_sub_legs(metrics: Dict[str, Optional[float]],
                           specs: List[Tuple[str, float]],
                           reference: Dict[str, Any],
                           pos: str) -> Optional[float]:
    """Weighted mean of one season's sub-leg knot scores over the
    sub-legs that have BOTH an input value and a recorded anchor.
    None when nothing in the season can be priced."""
    total_w = 0.0
    acc = 0.0
    for key, w in specs:
        score = knot_score(metrics.get(key), _anchors(reference, pos, key))
        if score is not None:
            acc += score * w
            total_w += w
    if total_w <= 0:
        return None
    return acc / total_w


def _blend_priced(sel: Dict[str, Any], per_season: Dict[int, Optional[float]],
                  pos: str, leg: str) -> Optional[float]:
    """85% peak / 15% latest, tempered by reliability. One season
    priced alone stands on its own; two missing scores stay missing."""
    peak, latest = sel.get("peak"), sel.get("latest")
    if peak is None:
        return None
    p_score = per_season.get(id(peak))
    if latest is None or latest is peak:
        return p_score
    l_score = per_season.get(id(latest))
    if p_score is None:
        return l_score
    if l_score is None:
        return p_score
    w_peak = PEAK_BLEND * _season_reliability(peak, pos, leg)
    w_latest = (1.0 - PEAK_BLEND) * _season_reliability(latest, pos, leg)
    if w_peak + w_latest <= 0:
        return p_score
    return (p_score * w_peak + l_score * w_latest) / (w_peak + w_latest)


def _rate_leg(sel: Dict[str, Any], specs: List[Tuple[str, float]],
              reference: Dict[str, Any], pos: str,
              leg: str) -> Optional[float]:
    """A season-stat leg (production / efficiency): price the peak
    and latest seasons' sub-legs and blend by the one rule."""
    if sel.get("peak") is None:
        return None
    seasons = {id(sel["peak"]): sel["peak"]}
    if sel.get("latest") is not None:
        seasons[id(sel["latest"])] = sel["latest"]
    per_season = {
        sid: _price_season_sub_legs(season_metrics(s, pos), specs, reference, pos)
        for sid, s in seasons.items()
    }
    return _blend_priced(sel, per_season, pos, leg)


# Sub-leg specs (metric, weight within the leg). Starting weights for
# the skeleton; the calibration follow-on arbitrates them against the
# backtest. A sub-leg with no recorded anchor in the reference simply
# does not fire (requirement 8 is enforced at reference-build time by
# reporting every table's n and firing coverage).
PRODUCTION_SPECS: Dict[str, List[Tuple[str, float]]] = {
    "QB": [("pass_yds_pg", 0.30), ("pass_tds_pg", 0.20),
           ("completion_pct", 0.15), ("ypa", 0.15), ("td_int", 0.10),
           ("rush_yds_pg", 0.05), ("qb_share", 0.05)],
    "RB": [("rush_yds_pg", 0.30), ("scrimmage_yds_pg", 0.25),
           ("tds_pg", 0.15), ("dominator_rating", 0.15), ("ypc", 0.10),
           ("market_share_yards", 0.05)],
    "WR": [("rec_yds_pg", 0.35), ("rec_tds_pg", 0.15),
           ("pass_share", 0.20), ("dominator_rating", 0.15),
           ("market_share_yards", 0.15)],
    "TE": [("rec_yds_pg", 0.30), ("rec_tds_pg", 0.15),
           ("pass_share", 0.20), ("dominator_rating", 0.15),
           ("rec_pg", 0.10), ("market_share_yards", 0.10)],
}

EFFICIENCY_SPECS: Dict[str, List[Tuple[str, float]]] = {
    # QB efficiency-first (the 2026-10-16 ruling): YPA carries the leg.
    "QB": [("ypa", 0.40), ("completion_pct", 0.30), ("td_int", 0.30)],
    "RB": [("ypc", 0.45), ("ypr", 0.30), ("market_share_yards", 0.25)],
    "WR": [("ypr", 0.55), ("market_share_yards", 0.45)],
    # No catch-rate sub-leg: college targets are not an official stat
    # and no recorded distribution exists for them (round 15).
    "TE": [("ypr", 0.50), ("market_share_yards", 0.50)],
}


def breakout_peak_value(seasons: List[Dict],
                        pos: str) -> Tuple[Optional[float], Optional[Dict]]:
    """The breakout peak: max dominator (WR/RB), max pass share (TE),
    max team pass share carried (QB), over seasons with GP >= 1.
    A recorded 0.0 dominator is missing data, not a zero (round 15)."""
    best_val, best_season = None, None
    for s in seasons:
        if _safe(s.get("games_played")) < 1:
            continue
        m = season_metrics(s, pos)
        if pos == "TE":
            v = m["pass_share"]
        elif pos == "QB":
            v = m["qb_share"]
        else:
            v = m["dominator_rating"]
            if v is not None and v <= 0:
                v = None
        if v is not None and (best_val is None or v > best_val):
            best_val, best_season = v, s
    return best_val, best_season


def _traj_total(s: Dict) -> float:
    return (_safe(s.get("receiving_yards")) + _safe(s.get("rush_yards"))
            + 0.5 * _safe(s.get("pass_yards")))


def traj_growth(seasons: List[Dict]) -> Optional[float]:
    """Trajectory growth, first season to peak season, using the
    model's own statistic (round 15): totals growth when the peak
    season's per-game rate holds at least 90% of the first season's
    (the V-A totals switch), per-game growth otherwise."""
    data = [s for s in seasons if _has_data(s)
            and _safe(s.get("games_played")) >= 1]
    if len(data) < 2:
        return None
    data = sorted(data, key=lambda s: _safe(s.get("season"), 0))
    first = data[0]
    # Peak for trajectory = the season with the highest primary total
    # rate, position-agnostic (scrimmage + half passing per game).
    def pg(s: Dict) -> float:
        return _traj_total(s) / max(_safe(s.get("games_played")), 1.0)
    peak = max(data, key=pg)
    if peak is first:
        return None
    if pg(peak) >= 0.9 * pg(first) and _traj_total(first) > 0:
        return (_traj_total(peak) - _traj_total(first)) / _traj_total(first)
    if pg(first) > 0:
        return (pg(peak) - pg(first)) / pg(first)
    return None


def calc_breakout_v4(seasons: List[Dict], pos: str,
                     reference: Dict[str, Any]) -> Optional[float]:
    """Breakout = 70% peak-share knot + 30% trajectory-growth knot,
    over whichever sub-legs can be priced."""
    parts: List[Tuple[float, float]] = []
    peak_val, _ = breakout_peak_value(seasons, pos)
    peak_score = knot_score(peak_val, _anchors(reference, pos, "breakout_dom"))
    if peak_score is not None:
        parts.append((peak_score, 0.70))
    growth = traj_growth(seasons)
    growth_score = knot_score(growth, _anchors(reference, pos, "traj_growth"))
    if growth_score is not None:
        parts.append((growth_score, 0.30))
    if not parts:
        return None
    return sum(s * w for s, w in parts) / sum(w for _, w in parts)


def calc_age_v4(age: Optional[float], pos: str,
                reference: Dict[str, Any]) -> Optional[float]:
    """Age priced against the recorded age-at-draft distribution at
    the position. The reference stores age marks in performance order
    (oldest = worst), so the shared knot helper prices youth up."""
    if age is None:
        return None
    return knot_score(float(age), _anchors(reference, pos, "age_years"))


# Stars fallback: the typical 247 composite for the band (v3.3's
# documented anchors). Used ONLY to impute a composite for knot
# pricing when the composite itself is missing; flagged in receipts.
STARS_COMPOSITE_PROXY = {5: 0.985, 4: 0.93, 3: 0.87, 2: 0.80}


def calc_recruiting_v4(recruiting: Optional[Dict],
                       pos: str, reference: Dict[str, Any]
                       ) -> Tuple[Optional[float], Dict[str, Any]]:
    """(score, receipt). Composite first; stars imputation second;
    unknown pedigree is missing, never a neutral 50 (requirement 5)."""
    if not recruiting:
        return None, {"source": None}
    rating = recruiting.get("composite_rating")
    if rating is not None:
        return (knot_score(_safe(rating),
                           _anchors(reference, pos, "recruiting_composite")),
                {"source": "composite", "value": _safe(rating)})
    stars = recruiting.get("stars")
    if stars is not None and int(stars) in STARS_COMPOSITE_PROXY:
        proxy = STARS_COMPOSITE_PROXY[int(stars)]
        return (knot_score(proxy,
                           _anchors(reference, pos, "recruiting_composite")),
                {"source": "stars_imputed", "value": proxy})
    return None, {"source": None}


def competition_index(seasons: List[Dict],
                      team_context: Optional[Dict] = None) -> Optional[float]:
    """Raw competition index (0-1): recency-weighted conference
    quality, blended 50/50 with SP+ strength of schedule when the
    pipeline's team context carries it for the latest season. The
    index is knot-priced against its recorded distribution; the
    index itself is context, not a score."""
    if not seasons:
        return None
    ordered = sorted(seasons, key=lambda s: _safe(s.get("season"), 0),
                     reverse=True)
    total_w = 0.0
    acc = 0.0
    for i, s in enumerate(ordered):
        team = (s.get("team") or "").lower()
        quality = 0.94 if "notre dame" in team else _conf_quality(
            s.get("conference"))
        w = max(0.2, 1.0 - i * 0.2)
        acc += quality * w
        total_w += w
    if total_w <= 0:
        return None
    idx = acc / total_w
    if team_context and ordered:
        latest = ordered[0]
        key = ((latest.get("team") or "").lower().strip(),
               latest.get("season", 0))
        ctx = team_context.get(key) or {}
        sos = ctx.get("sp_sos")
        if sos is not None:
            sos_q = _clip((_safe(sos) + 12.0) / 24.0 * 100.0) / 100.0
            idx = idx * 0.5 + sos_q * 0.5
    return idx


def calc_competition_v4(seasons: List[Dict], pos: str,
                        reference: Dict[str, Any],
                        team_context: Optional[Dict] = None
                        ) -> Optional[float]:
    idx = competition_index(seasons, team_context)
    if idx is None:
        return None
    return knot_score(idx, _anchors(reference, pos, "competition_index"))


# ─────────────────────────────────────────────────────────────────────
# Athleticism: recorded drafted-tester knot marks (carried from
# v3.3's tables via the reference artifact), the three-zone
# transform carried over unchanged, and the lifecycle rule for
# missing testing.
# ─────────────────────────────────────────────────────────────────────

ATHLETICISM_WEIGHTS: Dict[str, Dict[str, float]] = {
    "QB": {"forty": 0.40, "speed": 0.20, "vertical": 0.15,
           "broad": 0.15, "cone": 0.05, "shuttle": 0.05},
    "RB": {"forty": 0.30, "speed": 0.30, "vertical": 0.10,
           "broad": 0.10, "cone": 0.10, "shuttle": 0.10},
    "WR": {"forty": 0.40, "speed": 0.20, "vertical": 0.15,
           "broad": 0.10, "cone": 0.075, "shuttle": 0.075},
    "TE": {"forty": 0.30, "speed": 0.20, "vertical": 0.20,
           "broad": 0.15, "cone": 0.075, "shuttle": 0.075},
}


def athleticism_transform(raw: float) -> float:
    """Three-zone treatment (rounds 10-12, carried over): testing is
    a red-flag check, not a gate. Identity at and below 60, slope
    0.70 from 60 to 85, slope 0.90 above 85 off the 77.5 intercept,
    so a perfect raw 100 lands at 91.0."""
    if raw <= 60.0:
        return raw
    if raw <= 85.0:
        return 60.0 + (raw - 60.0) * 0.70
    return 77.5 + (raw - 85.0) * 0.90


def _athleticism_raw(athleticism: Dict, pos: str,
                     reference: Dict[str, Any]) -> Optional[float]:
    """Raw knot-priced athleticism from whatever testing exists.
    None when no recorded mark can be priced."""
    if not athleticism:
        return None
    metric_values: Dict[str, Optional[float]] = {
        "forty": athleticism.get("forty_yard"),
        "vertical": athleticism.get("vertical_inches"),
        "broad": athleticism.get("broad_jump_in"),
        "cone": athleticism.get("three_cone"),
        "shuttle": athleticism.get("short_shuttle"),
        "speed": None,
    }
    forty = athleticism.get("forty_yard")
    weight = athleticism.get("weight_lbs")
    if forty and weight:
        metric_values["speed"] = (_safe(weight) * 200.0) / (_safe(forty) ** 4)
    weights = ATHLETICISM_WEIGHTS.get(pos, ATHLETICISM_WEIGHTS["WR"])
    total_w = 0.0
    acc = 0.0
    for metric, w in weights.items():
        v = metric_values.get(metric)
        if v is None:
            continue
        score = knot_score(_safe(v), _athleticism_marks(reference, pos, metric))
        if score is not None:
            acc += score * w
            total_w += w
    if total_w <= 0:
        return None
    return acc / total_w


def calc_athleticism_v4(athleticism: Dict, pos: str,
                        reference: Dict[str, Any],
                        state: str) -> Tuple[Optional[float], Dict[str, Any]]:
    """Testing priced when it exists (any state). When it does not:
    in-season and pre-combine it is not-yet-available, so the leg is
    MISSING and redistributes; post-combine a settled absence keeps
    the rounds 10-11 floor."""
    raw = _athleticism_raw(athleticism, pos, reference)
    if raw is not None:
        return _clip(athleticism_transform(raw)), {"source": "tested"}
    if state == "post_combine":
        return UNTESTED_FLOOR, {"source": "settled_absence_floor"}
    return None, {"source": "not_yet_available"}


# ─────────────────────────────────────────────────────────────────────
# Durability: role-aware (requirement 6; thresholds from PR #2691,
# derived there from the recorded backup/starter season bands).
# ─────────────────────────────────────────────────────────────────────

def season_has_role(season: Dict, position: Optional[str]) -> bool:
    """Did the player hold a real role this season? Stored
    games_played counts appearances, so usage separates a backup's
    cameo from an injured starter's short season:
    QB >= 75 pass attempts, RB >= 60 touches, WR >= 20 targets or
    >= 12 receptions, TE >= 15 targets or >= 10 receptions."""
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


def _current_week(as_of: _date) -> int:
    season_start = _date(as_of.year, 8, 25)
    if as_of < season_start:
        return 1
    return min((as_of - season_start).days // 7 + 1, 16)


def calc_durability_v4(seasons: List[Dict], pos: str,
                       partial_year: Optional[int] = None,
                       as_of: Optional[_date] = None) -> Optional[float]:
    """Availability over COUNTED seasons only. A sub-12-game season
    counts only with a real role; a 12+ game season always counts.
    The in-progress season counts when it is a role season, with
    absences measured against a bye-adjusted floor (current week
    minus 2), and never triggers the completed-season penalty.
    Returns None when no season can be counted (missing data)."""
    if not seasons:
        return None
    as_of = as_of or _date.today()
    week = _current_week(as_of)
    scored: List[Tuple[float, bool, float]] = []  # (pct, is_completed, gp)
    for s in sorted(seasons, key=lambda s: _safe(s.get("season"), 0)):
        gp_raw = s.get("games_played")
        if gp_raw is None:
            continue
        gp = _safe(gp_raw)
        if gp < FULL_SEASON_GAMES and not season_has_role(s, pos):
            continue
        year = int(_safe(s.get("season"), 0))
        is_partial = partial_year is not None and year == partial_year
        if is_partial:
            denom = max(week - 2, 1)
            scored.append((min(gp / denom, 1.0), False, gp))
        elif year == as_of.year:
            scored.append((min(gp / max(week, 1), 1.0), False, gp))
        else:
            scored.append((min(gp / FULL_SEASON_GAMES, 1.0), True, gp))
    if not scored:
        return None
    n = len(scored)
    total_w = 0.0
    acc = 0.0
    for rank, (pct, _done, _gp) in enumerate(scored):
        w = 0.5 + 0.5 * (rank / max(n - 1, 1))
        acc += pct * w * 100.0
        total_w += w
    base = acc / total_w if total_w > 0 else 0.0
    penalty = 0.0
    for _pct, is_completed, gp in scored:
        if is_completed and gp < 9:
            missed = 9 - gp
            penalty = max(penalty, missed * missed * 1.2)
    return _clip(base - penalty)


# ─────────────────────────────────────────────────────────────────────
# Lifecycle state (requirement 7)
# ─────────────────────────────────────────────────────────────────────

def _has_testing(athleticism: Optional[Dict]) -> bool:
    if not athleticism:
        return False
    return any(athleticism.get(k) for k in (
        "forty_yard", "vertical_inches", "broad_jump_in",
        "three_cone", "short_shuttle", "ras_score"))


def detect_lifecycle_state(prospect: Dict[str, Any],
                           active_season_year: Optional[int] = None,
                           cycle_complete: bool = False) -> str:
    """in_season / pre_combine / post_combine, by data state.

    Testing enters whenever it exists for the player, so any
    recorded testing means post_combine. A partial latest season in
    the run's active year means in_season. A caller that knows the
    player's cycle is over (a backtest of a finished class) passes
    cycle_complete=True: no testing then is a settled absence.
    Otherwise an untested player is pre_combine: his testing is
    not-yet-available, never a settled absence."""
    if _has_testing(prospect.get("athleticism")):
        return "post_combine"
    seasons = prospect.get("seasons") or []
    latest = _latest_season(seasons)
    if (active_season_year is not None and latest is not None
            and int(_safe(latest.get("season"), 0)) == int(active_season_year)
            and 1 <= _safe(latest.get("games_played")) <= 11):
        return "in_season"
    if cycle_complete:
        return "post_combine"
    return "pre_combine"


# ─────────────────────────────────────────────────────────────────────
# Weights and the capital layer
# ─────────────────────────────────────────────────────────────────────

# Talent-frame leg weights (skeleton starting weights; the
# calibration follow-on arbitrates them against the backtest).
# QB is efficiency-first per the 2026-10-16 ruling. TE carries real
# traits weight (athleticism + age + efficiency = 0.42) alongside
# production, per requirement 10.
LEG_WEIGHTS: Dict[str, Dict[str, float]] = {
    "QB": {"production": 0.30, "efficiency": 0.35, "breakout": 0.00,
           "age": 0.08, "athleticism": 0.10, "competition": 0.09,
           "recruiting": 0.05, "durability": 0.03},
    "RB": {"production": 0.32, "efficiency": 0.14, "breakout": 0.16,
           "age": 0.09, "athleticism": 0.13, "competition": 0.05,
           "recruiting": 0.05, "durability": 0.06},
    "WR": {"production": 0.32, "efficiency": 0.16, "breakout": 0.16,
           "age": 0.08, "athleticism": 0.10, "competition": 0.08,
           "recruiting": 0.05, "durability": 0.05},
    "TE": {"production": 0.28, "efficiency": 0.16, "breakout": 0.12,
           "age": 0.12, "athleticism": 0.14, "competition": 0.06,
           "recruiting": 0.05, "durability": 0.07},
}
for _pos, _w in LEG_WEIGHTS.items():
    assert abs(sum(_w.values()) - 1.0) < 0.001, f"{_pos} weights must sum to 1.0"

# Capital's share of the FINAL grade when a pick exists (the talent
# grade itself is capital-free; see requirement 9).
CAPITAL_WEIGHT = {"QB": 0.26, "RB": 0.26, "WR": 0.29, "TE": 0.24}

LEG_KEYS = ["production", "efficiency", "breakout", "age",
            "athleticism", "competition", "recruiting", "durability"]


def _capital_score(draft_capital: Optional[Dict], pos: str) -> Optional[float]:
    """The capital layer's score: the consensus map's precomputed
    position-adjusted score when present; otherwise derived from the
    projected pick via the existing consensus helper. None when no
    pick exists anywhere (capital is then missing, and the final
    grade is the pure talent grade)."""
    if not draft_capital or draft_capital.get("projected_pick") is None:
        return None
    pre = draft_capital.get("projected_draft_capital_score")
    if pre is not None:
        return _clip(_safe(pre))
    try:
        from .mock_draft_consensus import pick_to_draft_capital_score
    except ImportError:
        from data_building.rookie_pipeline.mock_draft_consensus import (
            pick_to_draft_capital_score)
    return _clip(pick_to_draft_capital_score(
        int(draft_capital["projected_pick"]), pos))


# ─────────────────────────────────────────────────────────────────────
# Supporting cast (requirement 2)
# ─────────────────────────────────────────────────────────────────────

def cast_adjustment(cast_quality: Optional[float]) -> float:
    """Bounded charge/credit for context: an index of 50 is the
    population middle, so a QB throwing to an elite corps (index 90)
    is charged, and a receiver producing with a poor QB (index 10)
    is credited, in the bounded style of the model's eval
    adjustments. Zero when cast is unknown."""
    if cast_quality is None:
        return 0.0
    return max(-CAST_ADJ_CAP,
               min(CAST_ADJ_CAP, (50.0 - float(cast_quality)) * CAST_ADJ_K))


def compute_cast_context(prospect: Dict[str, Any],
                         graded: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cast indices for one prospect from the SAME RUN's graded
    teammates (pass-1 talent grades in `graded`, keyed by player_id
    in each result's receipt).

    QB: his receiving corps = teammates at RB/WR/TE who shared a
    team-season with him, weighted by their receiving yards in the
    shared season. Receiver/RB: his QB = the QB teammate from a
    shared team-season (attempts-weighted when several). Returns {}
    when no teammate link exists (cast unknown)."""
    pos = (prospect.get("position") or "").upper()
    my_seasons = {
        ((s.get("team") or "").lower().strip(), int(_safe(s.get("season"), 0))): s
        for s in (prospect.get("seasons") or [])
    }
    if not my_seasons:
        return {}
    by_id = {r["player_id"]: r for r in graded}
    prospects_by_id = {r["player_id"]: r.get("_prospect") for r in graded}

    def shared_season(other: Dict) -> Optional[Dict]:
        for s in (other.get("seasons") or []):
            key = ((s.get("team") or "").lower().strip(),
                   int(_safe(s.get("season"), 0)))
            if key in my_seasons:
                return s
        return None

    if pos == "QB":
        acc = 0.0
        total_w = 0.0
        for pid, result in by_id.items():
            other = prospects_by_id.get(pid)
            if not other or pid == prospect.get("player_id"):
                continue
            if (other.get("position") or "").upper() not in ("RB", "WR", "TE"):
                continue
            shared = shared_season(other)
            if shared is None or result.get("talent_score") is None:
                continue
            w = _safe(shared.get("receiving_yards")) or _safe(
                shared.get("receptions")) or 1.0
            acc += float(result["talent_score"]) * w
            total_w += w
        if total_w > 0:
            return {"qb_cast_quality": acc / total_w}
        return {}
    # Receiver / RB: the quality of his QB play.
    acc = 0.0
    total_w = 0.0
    for pid, result in by_id.items():
        other = prospects_by_id.get(pid)
        if not other or pid == prospect.get("player_id"):
            continue
        if (other.get("position") or "").upper() != "QB":
            continue
        shared = shared_season(other)
        if shared is None or result.get("talent_score") is None:
            continue
        w = _safe(shared.get("pass_attempts")) or 1.0
        acc += float(result["talent_score"]) * w
        total_w += w
    if total_w > 0:
        return {"receiver_qb_quality": acc / total_w}
    return {}


# ─────────────────────────────────────────────────────────────────────
# Main scoring
# ─────────────────────────────────────────────────────────────────────

def score_prospect_v4(
    prospect: Dict[str, Any],
    draft_capital: Optional[Dict[str, Any]] = None,
    reference: Optional[Dict[str, Any]] = None,
    active_season_year: Optional[int] = None,
    cycle_complete: bool = False,
    cast_context: Optional[Dict[str, Any]] = None,
    as_of: Optional[_date] = None,
) -> Dict[str, Any]:
    """Grade one prospect under the v4 architecture.

    reference: a v4 reference artifact (see load_reference). When
    None, the default artifact path is tried; when no artifact
    exists, every anchor-priced leg is missing and redistributes.
    cast_context: {"qb_cast_quality": x} for a QB or
    {"receiver_qb_quality": x} for a receiver/RB, 0-100 indices
    where 50 is the population middle (see compute_cast_context).
    """
    if reference is None:
        reference = load_reference()
    pos = (prospect.get("position") or "WR").upper()
    seasons = prospect.get("seasons") or []
    raw_age = prospect.get("age")
    age = float(raw_age) if raw_age is not None else None
    state = detect_lifecycle_state(prospect, active_season_year, cycle_complete)
    partial_year = active_season_year if state == "in_season" else None

    sel = select_priced_seasons(seasons, pos, partial_year=partial_year)

    production = _rate_leg(sel, PRODUCTION_SPECS.get(pos, PRODUCTION_SPECS["WR"]),
                           reference, pos, "production")
    efficiency = _rate_leg(sel, EFFICIENCY_SPECS.get(pos, EFFICIENCY_SPECS["WR"]),
                           reference, pos, "efficiency")
    breakout = calc_breakout_v4(seasons, pos, reference)
    age_score = calc_age_v4(age, pos, reference)
    athleticism, ath_receipt = calc_athleticism_v4(
        prospect.get("athleticism") or {}, pos, reference, state)
    competition = calc_competition_v4(
        seasons, pos, reference, team_context=prospect.get("team_context"))
    recruiting, recruiting_receipt = calc_recruiting_v4(
        prospect.get("recruiting"), pos, reference)
    durability = calc_durability_v4(seasons, pos, partial_year=partial_year,
                                    as_of=as_of)

    # Supporting cast: bounded adjustment inside the priced legs.
    cast_receipt: Dict[str, Any] = {"status": "missing"}
    cast_context = cast_context or {}
    if pos == "QB" and cast_context.get("qb_cast_quality") is not None:
        adj = cast_adjustment(cast_context["qb_cast_quality"])
        if efficiency is not None:
            efficiency = _clip(efficiency + adj)
        cast_receipt = {"qb_cast_quality": round(float(
            cast_context["qb_cast_quality"]), 2), "efficiency_adj": round(adj, 2)}
    elif pos != "QB" and cast_context.get("receiver_qb_quality") is not None:
        adj = cast_adjustment(cast_context["receiver_qb_quality"])
        if production is not None:
            production = _clip(production + adj)
        cast_receipt = {"receiver_qb_quality": round(float(
            cast_context["receiver_qb_quality"]), 2),
            "production_adj": round(adj, 2)}

    leg_scores: Dict[str, Optional[float]] = {
        "production": production, "efficiency": efficiency,
        "breakout": breakout, "age": age_score,
        "athleticism": athleticism, "competition": competition,
        "recruiting": recruiting, "durability": durability,
    }

    # Talent: weighted blend over PRESENT legs only. A missing leg's
    # weight redistributes proportionally (renormalization), so the
    # talent grade stays on the same scale with the leg simply absent.
    weights = dict(LEG_WEIGHTS.get(pos, LEG_WEIGHTS["WR"]))
    present = {k: v for k, v in leg_scores.items()
               if v is not None and weights.get(k, 0.0) > 0}
    missing = [k for k in LEG_KEYS
               if weights.get(k, 0.0) > 0 and leg_scores.get(k) is None]
    talent: Optional[float] = None
    weights_used: Dict[str, float] = {}
    if present:
        total_w = sum(weights[k] for k in present)
        talent = sum(float(v) * weights[k] for k, v in present.items()) / total_w
        weights_used = {k: weights[k] / total_w for k in present}

    # Capital: a separate layer over the pure talent grade.
    cap_score = _capital_score(draft_capital, pos)
    grade = talent
    capital_delta = 0.0
    if talent is not None and cap_score is not None:
        cap_w = CAPITAL_WEIGHT.get(pos, CAPITAL_WEIGHT["WR"])
        grade = (1.0 - cap_w) * talent + cap_w * cap_score
        capital_delta = grade - talent

    # Confidence: thinner evidence reads thinner. The in-season
    # discount is the round 14 rule, capped at 20.
    confidence = 100.0
    if state == "in_season" and sel.get("latest") is not None:
        gp = _safe(sel["latest"].get("games_played"))
        if gp > 0:
            confidence -= round(INSEASON_CONFIDENCE_CAP * (1.0 - gp / 12.0))
    confidence -= 5.0 * len(missing)
    confidence = int(_clip(confidence, 5.0, 100.0))

    # Display percentiles of the leg scores against the recorded
    # population of the CURRENT model's component scores (the
    # reference's percentile tables), for receipts and surfaces.
    display_percentiles: Dict[str, Optional[float]] = {}
    for leg, component in (("production", "production_score"),
                           ("efficiency", "efficiency_score"),
                           ("breakout", "breakout_profile_score"),
                           ("athleticism", "athleticism_score")):
        if leg_scores.get(leg) is not None:
            display_percentiles[leg] = percentile_of(
                leg_scores[leg], _percentile_table(reference, pos, component))

    band_key, band_label = band_for(grade)
    provenance = reference.get("provenance") or {}

    result: Dict[str, Any] = {
        "player_id": prospect.get("player_id"),
        "name": prospect.get("name"),
        "position": pos,
        "model_version": MODEL_VERSION,
        "lifecycle_state": state,
        "in_season": state == "in_season",
        "prospect_score": None if grade is None else round(grade, 2),
        "talent_score": None if talent is None else round(talent, 2),
        "capital_delta": round(capital_delta, 2),
        "projected_draft_capital_score": (
            None if cap_score is None else round(cap_score, 2)),
        "band": band_key,
        "band_label": band_label,
        "production_score": _round_or_none(leg_scores["production"]),
        "efficiency_score": _round_or_none(leg_scores["efficiency"]),
        "breakout_profile_score": _round_or_none(leg_scores["breakout"]),
        "age_score": _round_or_none(leg_scores["age"]),
        "athleticism_score": _round_or_none(leg_scores["athleticism"]),
        "competition_score": _round_or_none(leg_scores["competition"]),
        "recruiting_score": _round_or_none(leg_scores["recruiting"]),
        "durability_score": _round_or_none(leg_scores["durability"]),
        "weights_used": {k: round(v, 4) for k, v in weights_used.items()},
        "legs_missing": missing,
        "seasons_priced": {
            "peak": (sel["peak"] or {}).get("season"),
            "latest": (sel["latest"] or {}).get("season"),
        },
        "cast": cast_receipt,
        "athleticism_receipt": ath_receipt,
        "recruiting_receipt": recruiting_receipt,
        "display_percentiles": display_percentiles,
        "confidence_score": confidence,
        "reference_kind": ("fixture_sample" if provenance.get("fixture_sample")
                           else ("recorded" if provenance else "none")),
        "reasons": [],
    }
    result["reasons"] = _build_reasons_v4(result, leg_scores)
    return result


def _round_or_none(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(float(v), 2)


def _build_reasons_v4(result: Dict[str, Any],
                      leg_scores: Dict[str, Optional[float]]) -> List[str]:
    reasons: List[str] = []
    if result["band_label"]:
        reasons.append(f"{result['band_label']} band "
                       f"({result['lifecycle_state'].replace('_', '-')})")
    scored = [(k, v) for k, v in leg_scores.items() if v is not None]
    if scored:
        top = max(scored, key=lambda kv: kv[1])
        low = min(scored, key=lambda kv: kv[1])
        reasons.append(f"Strongest leg: {top[0]} ({top[1]:.1f})")
        if low[0] != top[0]:
            reasons.append(f"Weakest leg: {low[0]} ({low[1]:.1f})")
    if result["legs_missing"]:
        reasons.append("Missing legs (weight redistributed): "
                       + ", ".join(result["legs_missing"]))
    if result["projected_draft_capital_score"] is not None:
        reasons.append(
            f"Talent {result['talent_score']:.2f}; capital "
            f"{result['capital_delta']:+.2f} to {result['prospect_score']:.2f}")
    if result["cast"].get("status") != "missing":
        reasons.append(f"Supporting cast priced: {result['cast']}")
    if result["reference_kind"] == "fixture_sample":
        reasons.append("Anchored to FIXTURE SAMPLE reference data "
                       "(machinery demonstration, not recorded data)")
    return reasons


def score_all_v4(
    prospects: List[Dict[str, Any]],
    consensus_map: Optional[Dict[str, Dict]] = None,
    reference: Optional[Dict[str, Any]] = None,
    active_season_year: Optional[int] = None,
    cycle_complete: bool = False,
    as_of: Optional[_date] = None,
) -> List[Dict[str, Any]]:
    """Grade a run of prospects. Two passes: pass 1 grades everyone
    cast-free; cast indices are then computed from the same run's
    graded teammates, and pass 2 re-grades with cast priced. Adds
    overall_rank / position_rank, sorted by grade descending."""
    if reference is None:
        reference = load_reference()
    consensus_map = consensus_map or {}

    def grade_all(cast_by_id: Dict[str, Dict[str, Any]]):
        out = []
        for p in prospects:
            r = score_prospect_v4(
                p, consensus_map.get(p.get("player_id")), reference,
                active_season_year=active_season_year,
                cycle_complete=cycle_complete,
                cast_context=cast_by_id.get(p.get("player_id")),
                as_of=as_of)
            r["_prospect"] = p
            out.append(r)
        return out

    first_pass = grade_all({})
    cast_by_id: Dict[str, Dict[str, Any]] = {}
    for p in prospects:
        ctx = compute_cast_context(p, first_pass)
        if ctx:
            cast_by_id[p.get("player_id")] = ctx
    results = grade_all(cast_by_id) if cast_by_id else first_pass
    for r in results:
        r.pop("_prospect", None)

    results.sort(key=lambda x: (x["prospect_score"] is None,
                                -(x["prospect_score"] or 0.0)))
    pos_counters: Dict[str, int] = {}
    for i, r in enumerate(results):
        r["overall_rank"] = i + 1
        pos_counters[r["position"]] = pos_counters.get(r["position"], 0) + 1
        r["position_rank"] = pos_counters[r["position"]]
    return results
