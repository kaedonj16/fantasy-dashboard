"""
Expected Fantasy Points (xFP) from open nflverse play-by-play.

This is a *descriptive* expected-points model in the mold of the public
Fantasy Points Data / PFF "Expected Points" work: every opportunity (target,
carry, dropback) is assigned an expected fantasy value from its context,
independent of whether it actually worked out. Summed per player you get xFP —
what a league-average player would have scored on that exact workload — and

    points over expected = actual − xFP

A negative number is "points left on the board": elite usage that did not (yet)
convert into fantasy points, which historically is a positive-regression signal.

Why this is the "legit" version and not WOPR with extra steps:
  * Receiving/receiving-yards expectation uses nflverse's own per-play modeled
    values — completion probability ``cp`` and expected YAC ``xyac_mean_yardage``
    — so the yards/catch component is a trained model, not a hand bucket.
  * Touchdown equity (the part nflverse gives no per-play probability for) is an
    empirical league-wide rate bucketed by field position (``yardline_100``) and
    a deep-shot flag, so a target at the 3 and a 40-yard shot both carry their
    real scoring equity, and goal-line carries carry theirs.

Everything derives from the same PBP the rest of ``nflverse_metrics`` already
pulls, so ``actual`` and ``expected`` are computed on the same basis and
reconcile. Scoring is stored for all three reception formats (PPR / Half / Std);
only the per-reception bonus differs between them.

nfl_data_py is an optional dependency; the season/weekly builders degrade to {}
if it (or the data) is unavailable, mirroring the other builders in this package.
The pure math (buckets, table lookups, point combination) needs no pandas and is
unit-tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --- Scoring constants (full-PPR family; only the reception bonus varies) ------
# These mirror the constants the Advanced Metrics page already bakes into its
# client-side fpts_per_target (1.0 rec / 0.1 yd / 6 TD). Passing uses the common
# 0.04 yd / 4 TD / -1 INT defaults. Centralised here so a format change is one
# edit and both the expected and actual sides move together.
REC_YARD_PT = 0.1
REC_TD_PT = 6.0
RUSH_YARD_PT = 0.1
RUSH_TD_PT = 6.0
PASS_YARD_PT = 0.04
PASS_TD_PT = 4.0
INT_PT = -1.0
RECEPTION_PT: Dict[str, float] = {"ppr": 1.0, "half": 0.5, "standard": 0.0}
FORMATS: Tuple[str, ...] = ("ppr", "half", "standard")

# A target with air yards at or beyond this is a "deep shot" for TD-equity
# bucketing — deep targets carry TD equity that field position alone misses.
DEEP_THRESHOLD = 20.0

# yardline_100 is the distance to the opponent's goal line (1 = goal to go,
# 99 = backed up). Finer buckets near the goal line where scoring equity is
# concentrated, coarser out in the field of play.
_YARDLINE_EDGES = (3, 5, 10, 20, 40, 60, 80)


def yardline_bucket(yardline_100: Optional[float]) -> int:
    """Map a yardline_100 to a coarse bucket index (0 = closest to goal).

    None / out-of-range values fall into the deepest field-position bucket so a
    missing value never inflates TD equity.
    """
    if yardline_100 is None:
        return len(_YARDLINE_EDGES)
    try:
        y = float(yardline_100)
    except (TypeError, ValueError):
        return len(_YARDLINE_EDGES)
    for i, edge in enumerate(_YARDLINE_EDGES):
        if y < edge:
            return i
    return len(_YARDLINE_EDGES)


# Air-yards buckets used only for the *fallback* completion / YAC tables (when a
# play is missing nflverse's cp / xyac). Behind the line of scrimmage, short,
# intermediate, deep, bomb.
_AIRYARDS_EDGES = (0, 5, 10, 15, 20, 30)


def airyards_bucket(air_yards: Optional[float]) -> int:
    """Map air yards to a coarse bucket index for the fallback tables."""
    if air_yards is None:
        return 1  # treat unknown depth as a short target, not behind the LOS
    try:
        a = float(air_yards)
    except (TypeError, ValueError):
        return 1
    for i, edge in enumerate(_AIRYARDS_EDGES):
        if a < edge:
            return i
    return len(_AIRYARDS_EDGES)


def is_deep(air_yards: Optional[float]) -> bool:
    try:
        return air_yards is not None and float(air_yards) >= DEEP_THRESHOLD
    except (TypeError, ValueError):
        return False


# --- Empirical lookup tables ---------------------------------------------------


@dataclass
class ExpectedPointsTables:
    """League-wide empirical rates, bucketed, with global fallbacks.

    Every ``*_for`` accessor returns the bucket value when the bucket was
    populated, otherwise the global mean, otherwise 0.0 — so a sparsely-sampled
    bucket can never emit a wild rate.
    """

    rec_td_prob: Dict[Tuple[int, bool], float] = field(default_factory=dict)
    rush_td_prob: Dict[int, float] = field(default_factory=dict)
    rush_yds_mean: Dict[int, float] = field(default_factory=dict)
    pass_td_prob: Dict[int, float] = field(default_factory=dict)
    comp_prob: Dict[int, float] = field(default_factory=dict)
    yac_mean: Dict[int, float] = field(default_factory=dict)
    int_rate: float = 0.0

    rec_td_prob_global: float = 0.0
    rush_td_prob_global: float = 0.0
    rush_yds_mean_global: float = 0.0
    pass_td_prob_global: float = 0.0
    comp_prob_global: float = 0.65
    yac_mean_global: float = 5.0

    def rec_td_prob_for(self, yardline_100, air_yards) -> float:
        key = (yardline_bucket(yardline_100), is_deep(air_yards))
        v = self.rec_td_prob.get(key)
        return v if v is not None else self.rec_td_prob_global

    def rush_td_prob_for(self, yardline_100) -> float:
        v = self.rush_td_prob.get(yardline_bucket(yardline_100))
        return v if v is not None else self.rush_td_prob_global

    def rush_yds_mean_for(self, yardline_100) -> float:
        v = self.rush_yds_mean.get(yardline_bucket(yardline_100))
        return v if v is not None else self.rush_yds_mean_global

    def pass_td_prob_for(self, yardline_100) -> float:
        v = self.pass_td_prob.get(yardline_bucket(yardline_100))
        return v if v is not None else self.pass_td_prob_global

    def comp_prob_for(self, air_yards) -> float:
        v = self.comp_prob.get(airyards_bucket(air_yards))
        return v if v is not None else self.comp_prob_global

    def yac_mean_for(self, air_yards) -> float:
        v = self.yac_mean.get(airyards_bucket(air_yards))
        return v if v is not None else self.yac_mean_global


# --- Per-opportunity accumulation (pure) --------------------------------------


def new_components() -> Dict[str, float]:
    """A zeroed expected/actual component accumulator for one player (or week)."""
    return {
        # expected
        "x_receptions": 0.0, "x_rec_yards": 0.0, "x_rec_td": 0.0,
        "x_carries": 0.0, "x_rush_yards": 0.0, "x_rush_td": 0.0,
        "x_pass_att": 0.0, "x_pass_yards": 0.0, "x_pass_td": 0.0, "x_int": 0.0,
        # actual (same basis, so deltas reconcile)
        "a_receptions": 0.0, "a_rec_yards": 0.0, "a_rec_td": 0.0,
        "a_carries": 0.0, "a_rush_yards": 0.0, "a_rush_td": 0.0,
        "a_pass_att": 0.0, "a_pass_yards": 0.0, "a_pass_td": 0.0, "a_int": 0.0,
    }


def add_target(comp: Dict[str, float], tables: ExpectedPointsTables,
               air_yards: Optional[float], yardline_100: Optional[float],
               *, cp: Optional[float] = None, xyac: Optional[float] = None) -> None:
    """Accumulate one receiving target's expected value.

    Expected receiving yards = P(catch) × (air yards + expected YAC), using
    nflverse's per-play cp / xyac when present and the empirical fallback tables
    otherwise. TD equity is the empirical rec-TD rate for the field position and
    deep-shot class.
    """
    catch = cp if cp is not None else tables.comp_prob_for(air_yards)
    yac = xyac if xyac is not None else tables.yac_mean_for(air_yards)
    ay = float(air_yards) if air_yards is not None else 0.0
    comp["x_receptions"] += catch
    comp["x_rec_yards"] += catch * (ay + yac)
    comp["x_rec_td"] += tables.rec_td_prob_for(yardline_100, air_yards)


def add_carry(comp: Dict[str, float], tables: ExpectedPointsTables,
              yardline_100: Optional[float]) -> None:
    """Accumulate one carry's expected value (field-position mean yards + TD)."""
    comp["x_carries"] += 1.0
    comp["x_rush_yards"] += tables.rush_yds_mean_for(yardline_100)
    comp["x_rush_td"] += tables.rush_td_prob_for(yardline_100)


def add_pass_attempt(comp: Dict[str, float], tables: ExpectedPointsTables,
                     air_yards: Optional[float], yardline_100: Optional[float],
                     *, cp: Optional[float] = None,
                     xyac: Optional[float] = None) -> None:
    """Accumulate one dropback's expected passing value (for the QB)."""
    catch = cp if cp is not None else tables.comp_prob_for(air_yards)
    yac = xyac if xyac is not None else tables.yac_mean_for(air_yards)
    ay = float(air_yards) if air_yards is not None else 0.0
    comp["x_pass_att"] += 1.0
    comp["x_pass_yards"] += catch * (ay + yac)
    comp["x_pass_td"] += tables.pass_td_prob_for(yardline_100)
    comp["x_int"] += tables.int_rate


def expected_points(comp: Dict[str, float], fmt: str) -> float:
    """Total expected fantasy points for a scoring format."""
    rec_bonus = RECEPTION_PT[fmt]
    return (
        comp["x_rec_yards"] * REC_YARD_PT + comp["x_rec_td"] * REC_TD_PT
        + comp["x_receptions"] * rec_bonus
        + comp["x_rush_yards"] * RUSH_YARD_PT + comp["x_rush_td"] * RUSH_TD_PT
        + comp["x_pass_yards"] * PASS_YARD_PT + comp["x_pass_td"] * PASS_TD_PT
        + comp["x_int"] * INT_PT
    )


def actual_points(comp: Dict[str, float], fmt: str) -> float:
    """Total actual fantasy points reconstructed from the same PBP opportunities."""
    rec_bonus = RECEPTION_PT[fmt]
    return (
        comp["a_rec_yards"] * REC_YARD_PT + comp["a_rec_td"] * REC_TD_PT
        + comp["a_receptions"] * rec_bonus
        + comp["a_rush_yards"] * RUSH_YARD_PT + comp["a_rush_td"] * RUSH_TD_PT
        + comp["a_pass_yards"] * PASS_YARD_PT + comp["a_pass_td"] * PASS_TD_PT
        + comp["a_int"] * INT_PT
    )


# xFP columns. Both the season snapshot (a season total) and each weekly row (that
# week's total) use the SAME six column names: they are totals, so a selected week
# range sums them the way the UI sums the other advanced-metric totals
# (receiving_epa, yards_after_catch, ...), and the full-season sum matches the
# season snapshot value. A negative *_over_expected is fantasy points left on the
# board (elite opportunity that did not convert).
XFP_COLS: Tuple[str, ...] = (
    "expected_ppr", "expected_half_ppr", "expected_standard",
    "ppr_over_expected", "half_ppr_over_expected", "standard_over_expected",
)
_FMT_SUFFIX = {"ppr": "ppr", "half": "half_ppr", "standard": "standard"}


def _total_columns_from_components(comp: Dict[str, float]) -> Dict[str, float]:
    """Expected total + over-expected total for a component accumulator."""
    out: Dict[str, float] = {}
    for fmt in FORMATS:
        suf = _FMT_SUFFIX[fmt]
        exp = expected_points(comp, fmt)
        act = actual_points(comp, fmt)
        out[f"expected_{suf}"] = round(exp, 2)
        out[f"{suf}_over_expected"] = round(act - exp, 2)
    return out


def season_columns_from_components(comp: Dict[str, float]) -> Dict[str, float]:
    """Season-total expected + over-expected columns for one player's season."""
    return _total_columns_from_components(comp)


def weekly_columns_from_components(comp: Dict[str, float]) -> Dict[str, float]:
    """That week's expected total + over-expected total for one player-week."""
    return _total_columns_from_components(comp)


# --- PBP-backed builders (pandas / nfl_data_py) -------------------------------

_PBP_COLUMNS = [
    "game_id", "play_id", "week", "season_type", "play_type", "yardline_100",
    "air_yards", "cp", "xyac_mean_yardage",
    "pass_attempt", "rush_attempt", "complete_pass", "interception",
    "yards_gained", "passing_yards", "rushing_yards", "receiving_yards",
    "pass_touchdown", "rush_touchdown",
    "passer_player_id", "rusher_player_id", "receiver_player_id",
]


def _load_pbp(season: int):
    """Return a REG-season pbp DataFrame with the xFP columns, or None."""
    try:
        import nfl_data_py as nfl  # optional dependency
        pbp = nfl.import_pbp_data([season], columns=_PBP_COLUMNS, downcast=True)
    except Exception as e:  # pragma: no cover - exercised only without the dep
        print(f"[expected_points] pbp unavailable for {season} ({e})")
        return None
    if pbp is None or pbp.empty:
        return None
    pbp = pbp[pbp["season_type"] == "REG"]
    return pbp if not pbp.empty else None


def _num(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f  # drop NaN


def build_tables_from_pbp(pbp) -> ExpectedPointsTables:
    """Build the league-wide empirical rate tables from a season's pbp.

    In-sample (same season) is deliberate: this is a *descriptive* expected-points
    measure, and each league-wide bucket holds thousands of plays, so there is no
    small-sample instability to guard against by holding out a training season.
    """
    from collections import defaultdict

    rec_td_n: Dict[Tuple[int, bool], float] = defaultdict(float)
    rec_td_k: Dict[Tuple[int, bool], float] = defaultdict(float)
    rush_td_n: Dict[int, float] = defaultdict(float)
    rush_td_k: Dict[int, float] = defaultdict(float)
    rush_yd_sum: Dict[int, float] = defaultdict(float)
    pass_td_n: Dict[int, float] = defaultdict(float)
    pass_td_k: Dict[int, float] = defaultdict(float)
    comp_n: Dict[int, float] = defaultdict(float)
    comp_k: Dict[int, float] = defaultdict(float)
    yac_sum: Dict[int, float] = defaultdict(float)
    yac_n: Dict[int, float] = defaultdict(float)

    rec_td_total = rec_td_ct = 0.0
    rush_td_total = rush_td_ct = rush_yd_total = 0.0
    pass_td_total = pass_td_ct = 0.0
    comp_total = comp_ct = yac_total = yac_ct = 0.0
    int_total = dropback_ct = 0.0

    for r in pbp.itertuples(index=False):
        d = r._asdict()
        ptype = str(d.get("play_type") or "")
        yl = _num(d.get("yardline_100"))
        ay = _num(d.get("air_yards"))

        is_pass = bool(_num(d.get("pass_attempt")) or 0) or ptype == "pass"
        is_rush = bool(_num(d.get("rush_attempt")) or 0) or ptype == "run"

        if is_pass and d.get("receiver_player_id") is not None:
            yb = yardline_bucket(yl)
            deep = is_deep(ay)
            rec_td_n[(yb, deep)] += 1.0
            rec_td_ct += 1.0
            td = _num(d.get("pass_touchdown")) or 0.0
            rec_td_k[(yb, deep)] += td
            rec_td_total += td
            # completion + YAC fallback tables (keyed by air-yards bucket)
            ab = airyards_bucket(ay)
            comp_n[ab] += 1.0
            comp_ct += 1.0
            complete = _num(d.get("complete_pass")) or 0.0
            comp_k[ab] += complete
            comp_total += complete
            # observed YAC from completions to seed the fallback YAC table
            if complete:
                yac_obs = None
                yg = _num(d.get("yards_gained"))
                if yg is not None and ay is not None:
                    yac_obs = yg - ay
                if yac_obs is not None:
                    yac_sum[ab] += yac_obs
                    yac_n[ab] += 1.0
                    yac_total += yac_obs
                    yac_ct += 1.0

        if is_pass and d.get("passer_player_id") is not None:
            pb = yardline_bucket(yl)
            pass_td_n[pb] += 1.0
            pass_td_ct += 1.0
            ptd = _num(d.get("pass_touchdown")) or 0.0
            pass_td_k[pb] += ptd
            pass_td_total += ptd
            int_total += _num(d.get("interception")) or 0.0
            dropback_ct += 1.0

        if is_rush and d.get("rusher_player_id") is not None:
            rb = yardline_bucket(yl)
            rush_td_n[rb] += 1.0
            rush_td_ct += 1.0
            rtd = _num(d.get("rush_touchdown")) or 0.0
            rush_td_k[rb] += rtd
            rush_td_total += rtd
            ry = _num(d.get("rushing_yards")) or 0.0
            rush_yd_sum[rb] += ry
            rush_yd_total += ry

    def _rate(k, n):
        return {b: (k[b] / n[b]) for b in n if n[b] > 0}

    def _mean(s, n):
        return {b: (s[b] / n[b]) for b in n if n[b] > 0}

    return ExpectedPointsTables(
        rec_td_prob=_rate(rec_td_k, rec_td_n),
        rush_td_prob=_rate(rush_td_k, rush_td_n),
        rush_yds_mean=_mean(rush_yd_sum, rush_td_n),
        pass_td_prob=_rate(pass_td_k, pass_td_n),
        comp_prob=_rate(comp_k, comp_n),
        yac_mean=_mean(yac_sum, yac_n),
        int_rate=(int_total / dropback_ct) if dropback_ct > 0 else 0.0,
        rec_td_prob_global=(rec_td_total / rec_td_ct) if rec_td_ct > 0 else 0.0,
        rush_td_prob_global=(rush_td_total / rush_td_ct) if rush_td_ct > 0 else 0.0,
        rush_yds_mean_global=(rush_yd_total / rush_td_ct) if rush_td_ct > 0 else 0.0,
        pass_td_prob_global=(pass_td_total / pass_td_ct) if pass_td_ct > 0 else 0.0,
        comp_prob_global=(comp_total / comp_ct) if comp_ct > 0 else 0.65,
        yac_mean_global=(yac_total / yac_ct) if yac_ct > 0 else 5.0,
    )


def _accumulate_player_weeks(pbp, tables: ExpectedPointsTables):
    """Return {gsis_id: {week: components}} accumulating expected + actual.

    A player appears under a week only for the weeks they had an opportunity, so
    the number of weeks is the games-played denominator for per-game columns.
    """
    from collections import defaultdict

    by_player: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(dict)

    def _bucket(gsis, week) -> Optional[Dict[str, float]]:
        if gsis is None:
            return None
        try:
            wk = int(week)
        except (TypeError, ValueError):
            return None
        weeks = by_player[str(gsis).strip()]
        if wk not in weeks:
            weeks[wk] = new_components()
        return weeks[wk]

    for r in pbp.itertuples(index=False):
        d = r._asdict()
        ptype = str(d.get("play_type") or "")
        week = d.get("week")
        yl = _num(d.get("yardline_100"))
        ay = _num(d.get("air_yards"))
        cp = _num(d.get("cp"))
        xyac = _num(d.get("xyac_mean_yardage"))
        is_pass = bool(_num(d.get("pass_attempt")) or 0) or ptype == "pass"
        is_rush = bool(_num(d.get("rush_attempt")) or 0) or ptype == "run"

        rec_id = d.get("receiver_player_id")
        if is_pass and rec_id is not None:
            comp = _bucket(rec_id, week)
            if comp is not None:
                add_target(comp, tables, ay, yl, cp=cp, xyac=xyac)
                complete = _num(d.get("complete_pass")) or 0.0
                comp["a_receptions"] += complete
                if complete:
                    rec_yds = _num(d.get("receiving_yards"))
                    if rec_yds is None:
                        rec_yds = _num(d.get("yards_gained")) or 0.0
                    comp["a_rec_yards"] += rec_yds
                    comp["a_rec_td"] += _num(d.get("pass_touchdown")) or 0.0

        pas_id = d.get("passer_player_id")
        if is_pass and pas_id is not None:
            comp = _bucket(pas_id, week)
            if comp is not None:
                add_pass_attempt(comp, tables, ay, yl, cp=cp, xyac=xyac)
                comp["a_pass_att"] += _num(d.get("pass_attempt")) or 0.0
                comp["a_pass_yards"] += _num(d.get("passing_yards")) or 0.0
                comp["a_pass_td"] += _num(d.get("pass_touchdown")) or 0.0
                comp["a_int"] += _num(d.get("interception")) or 0.0

        rush_id = d.get("rusher_player_id")
        if is_rush and rush_id is not None:
            comp = _bucket(rush_id, week)
            if comp is not None:
                add_carry(comp, tables, yl)
                comp["a_carries"] += 1.0
                comp["a_rush_yards"] += _num(d.get("rushing_yards")) or 0.0
                comp["a_rush_td"] += _num(d.get("rush_touchdown")) or 0.0

    return by_player


def _merge_components(weeks: Dict[int, Dict[str, float]]) -> Dict[str, float]:
    """Sum a player's per-week components into a season total."""
    total = new_components()
    for comp in weeks.values():
        for k, v in comp.items():
            total[k] += v
    return total


def build_expected_points_for_season(season: int) -> Dict[str, Dict[str, float]]:
    """Return {sleeper_id: {season per-game xFP columns}} for a season."""
    return build_expected_points_both(season)[0]


def build_expected_points_weekly_for_season(
    season: int,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    """Return {(sleeper_id, week): {that week's xFP total columns}} for a season."""
    return build_expected_points_both(season)[1]


def build_expected_points_both(
    season: int,
) -> Tuple[Dict[str, Dict[str, float]],
           Dict[Tuple[str, int], Dict[str, float]]]:
    """Build season (per-game) and weekly (per-week totals) maps in one PBP pass.

    Returns ``(season_map, weekly_map)``. The sync script uses this so a season's
    ~50k play rows are loaded, table-fit, and accumulated once rather than twice.
    Either map is {} when nfl_data_py / the data is unavailable.
    """
    pbp = _load_pbp(season)
    if pbp is None:
        return {}, {}
    from data_building.external_data.nflverse_metrics import _gsis_to_sleeper
    tables = build_tables_from_pbp(pbp)
    by_player = _accumulate_player_weeks(pbp, tables)
    crosswalk = _gsis_to_sleeper()

    season_map: Dict[str, Dict[str, float]] = {}
    weekly_map: Dict[Tuple[str, int], Dict[str, float]] = {}
    for gsis, weeks in by_player.items():
        pid = crosswalk.get(gsis)
        if not pid:
            continue
        season_cols = season_columns_from_components(_merge_components(weeks))
        if season_cols:
            season_map[pid] = season_cols
        for wk, comp in weeks.items():
            week_cols = weekly_columns_from_components(comp)
            if week_cols:
                weekly_map[(pid, wk)] = week_cols
    return season_map, weekly_map
