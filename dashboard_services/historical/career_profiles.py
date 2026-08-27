"""Career-stage, prior-elite, breakout, and draft-capital rates (pure).

Breakout labeling copies the existing engine's constants and comparison
(``prior is None or prior > 13``, this season ≤ 12). That is *not* the
same as first-time elite (never previously top-12 and this season ≤ 12).
Both are kept.

Draft-capital rates exclude missing capital; they are never labeled UDFA.
Missing years_experience is not mapped to rookie.

Descriptive only — these rates do not enter ranking or Pick Score.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

from dashboard_services.historical.age_curves import build_age_curves
from dashboard_services.historical.definitions import (
    CAREER_STAGE_ORDER,
    DEFAULT_BAYES_PRIOR_N,
    DRAFT_CAPITAL_ORDER,
    FTN_SEASON_FLOOR,
    RELIABLE_SEASON_FLOOR,
    SKILL_POSITIONS,
    SNAP_RELIABLE_FLOOR,
    TIER_CUTOFFS,
    age_bucket,
    career_stage,
    is_absolute_bust,
    _optional_int,
)
from dashboard_services.historical.usage import build_prior_usage_rates
from dashboard_services.historical.comps import build_comp_aggregates
from dashboard_services.historical.adp import build_adp_hit_rates
from dashboard_services.historical.finish_rates import (
    cohort_hit_rate,
    filter_era,
    filter_position,
    is_tier_hit,
    make_rate,
    positional_finish,
    position_baseline,
    season_bounds,
)

# Copied from data_building/breakout_engine/backtest_breakout_model.py.
# Do not "fix" `>` to `>=`. Rank 13 is *not* a non-starter under the engine.
# Slim tests string-assert the engine source still has these lines.
BREAKOUT_RANK_THRESHOLD = 12
PRIOR_NON_STARTER_RANK = 13

# Player-level cumulative windows. "by year 2" = exp 0 or 1 (rookie + year 2).
CUMULATIVE_WINDOWS: tuple[dict, ...] = (
    {"id": "top12_as_rookie", "tier": "top_12", "min_exp": 0, "max_exp": 0},
    {"id": "top12_by_year_2", "tier": "top_12", "min_exp": 0, "max_exp": 1},
    {"id": "top12_in_years_2_4", "tier": "top_12", "min_exp": 1, "max_exp": 3},
)


def was_engine_non_starter(
    prior_rank: Any,
    *,
    prior_non_starter: int = PRIOR_NON_STARTER_RANK,
) -> bool:
    """Engine: no prior rank, or prior rank strictly greater than 13."""
    rank = _optional_int(prior_rank)
    return rank is None or rank > prior_non_starter


def is_engine_breakout(
    prior_rank: Any,
    this_finish: Any,
    *,
    top_n: int = BREAKOUT_RANK_THRESHOLD,
    prior_non_starter: int = PRIOR_NON_STARTER_RANK,
) -> bool:
    """Engine breakout: was a non-starter AND this season finished top-N."""
    finish = _optional_int(this_finish)
    broke_out = finish is not None and finish <= top_n
    return was_engine_non_starter(prior_rank, prior_non_starter=prior_non_starter) and broke_out


def is_first_time_elite(
    previously_top12: Any,
    this_finish: Any,
    *,
    cutoff: int = BREAKOUT_RANK_THRESHOLD,
) -> bool:
    """Never previously top-12 AND this season finished top-12."""
    finish = _optional_int(this_finish)
    if finish is None or finish > cutoff:
        return False
    return not bool(previously_top12)


def _prior_rank(row: Mapping[str, Any]) -> Optional[int]:
    return _optional_int(row.get("previous_season_finish"))


def _previously_top12(row: Mapping[str, Any]) -> Optional[bool]:
    flag = row.get("previously_top12")
    if flag is not None:
        return bool(flag)
    candidate = row.get("first_time_top12_candidate")
    if candidate is not None:
        return not bool(candidate)
    count = _optional_int(row.get("prior_top12_count"))
    if count is None:
        return None
    return count > 0


def _first_time_candidate(row: Mapping[str, Any]) -> Optional[bool]:
    flag = row.get("first_time_top12_candidate")
    if flag is not None:
        return bool(flag)
    prev = _previously_top12(row)
    if prev is None:
        return None
    return not prev


def _player_id(row: Mapping[str, Any]) -> str:
    return str(row.get("sleeper_id") or row.get("player_id") or "").strip()


def _group_players(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for row in rows:
        pid = _player_id(row)
        if not pid:
            continue
        by.setdefault(pid, []).append(dict(row))
    return by


def _max_season(rows: Sequence[Mapping[str, Any]]) -> Optional[int]:
    bounds = season_bounds(rows)
    if not bounds:
        return None
    return bounds[1]


# ---------------------------------------------------------------------------
# Career stage
# ---------------------------------------------------------------------------

def rows_with_known_stage(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    out = []
    for row in rows:
        if career_stage(row.get("years_experience")) is None:
            continue
        out.append(dict(row))
    return out


def build_stage_rates(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
    tier: str = "top_12",
) -> dict:
    """P(tier | career stage, position). Missing exp excluded, not rookie."""
    out: dict[str, dict] = {}
    for pos in SKILL_POSITIONS:
        pos_rows = filter_position(rows, pos)
        known = rows_with_known_stage(pos_rows)
        baseline = position_baseline(known, pos, tier=tier, scoring=scoring)
        prior = baseline.get("raw_rate")
        stages = {}
        for label in CAREER_STAGE_ORDER:
            at_stage = [r for r in known if career_stage(r.get("years_experience")) == label]
            stages[label] = cohort_hit_rate(
                at_stage, tier=tier, scoring=scoring, prior_rate=prior
            )
        out[pos] = {
            "position": pos,
            "tier": tier,
            "n_known_stage": len(known),
            "n_missing_exp_excluded": len(pos_rows) - len(known),
            "baseline": baseline,
            "by_stage": stages,
        }
    return out


# ---------------------------------------------------------------------------
# Repeat / breakout (season-level)
# ---------------------------------------------------------------------------

def _prev_tier_cohort(
    rows: Sequence[Mapping[str, Any]],
    *,
    from_tier: str,
) -> list[dict]:
    cutoff = TIER_CUTOFFS[from_tier]
    out = []
    for row in rows:
        prior = _prior_rank(row)
        if prior is None:
            continue
        if prior <= cutoff:
            out.append(dict(row))
    return out


def build_repeat_and_breakout_rates(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
) -> dict:
    """Prev-elite repeats, 2+ prior elite, engine breakout, first-time elite."""
    out: dict[str, dict] = {}
    for pos in SKILL_POSITIONS:
        pos_rows = filter_position(rows, pos)
        baseline = position_baseline(pos_rows, pos, scoring=scoring)
        prior_rate = baseline.get("raw_rate")

        prev_top12 = _prev_tier_cohort(pos_rows, from_tier="top_12")
        two_plus = []
        for row in pos_rows:
            count = _optional_int(row.get("prior_top12_count"))
            if count is None or count < 2:
                continue
            two_plus.append(row)

        non_starters = [r for r in pos_rows if was_engine_non_starter(_prior_rank(r))]
        first_cands = []
        for row in pos_rows:
            cand = _first_time_candidate(row)
            if cand is True:
                first_cands.append(row)

        def _engine_hit(row: Mapping[str, Any]) -> bool:
            return is_engine_breakout(_prior_rank(row), positional_finish(row, scoring))

        def _first_hit(row: Mapping[str, Any]) -> bool:
            prev = _previously_top12(row)
            if prev is None:
                return False
            return is_first_time_elite(prev, positional_finish(row, scoring))

        out[pos] = {
            "position": pos,
            "baseline": baseline,
            "prev_top12_to_top12": cohort_hit_rate(
                prev_top12, tier="top_12", scoring=scoring, prior_rate=prior_rate
            ),
            "prev_top12_to_top5": cohort_hit_rate(
                prev_top12, tier="top_5", scoring=scoring, prior_rate=prior_rate
            ),
            "two_plus_prior_top12_to_top12": cohort_hit_rate(
                two_plus, tier="top_12", scoring=scoring, prior_rate=prior_rate
            ),
            "engine_breakout_among_non_starters": cohort_hit_rate(
                non_starters, scoring=scoring, prior_rate=prior_rate, hit_pred=_engine_hit
            ),
            "first_time_elite_among_candidates": cohort_hit_rate(
                first_cands, scoring=scoring, prior_rate=prior_rate, hit_pred=_first_hit
            ),
            "n_prev_top12": len(prev_top12),
            "n_two_plus_prior_top12": len(two_plus),
            "n_engine_non_starters": len(non_starters),
            "n_first_time_candidates": len(first_cands),
        }
    return out


# ---------------------------------------------------------------------------
# Draft capital (descriptive)
# ---------------------------------------------------------------------------

def rows_with_known_capital(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    out = []
    for row in rows:
        bucket = row.get("draft_capital_bucket")
        if bucket not in DRAFT_CAPITAL_ORDER:
            continue
        out.append(dict(row))
    return out


def _capital_season_rates(
    rows: Sequence[Mapping[str, Any]],
    position: str,
    *,
    scoring: str,
    prior_rate: Optional[float],
) -> dict:
    known = rows_with_known_capital(rows)
    by_capital = {}
    for bucket in DRAFT_CAPITAL_ORDER:
        at = [r for r in known if r.get("draft_capital_bucket") == bucket]
        by_capital[bucket] = {
            "top_12": cohort_hit_rate(at, tier="top_12", scoring=scoring, prior_rate=prior_rate),
            "top_5": cohort_hit_rate(at, tier="top_5", scoring=scoring, prior_rate=prior_rate),
            "top_24": cohort_hit_rate(at, tier="top_24", scoring=scoring, prior_rate=prior_rate),
            "absolute_bust": _bust_rate(at, position, prior_rate=prior_rate),
        }
    return by_capital


def _bust_rate(
    rows: Sequence[Mapping[str, Any]],
    position: str,
    *,
    prior_rate: Optional[float],
) -> dict:
    known = []
    for row in rows:
        flag = is_absolute_bust(position, positional_finish(row))
        if flag is None:
            continue
        known.append(row)
    return cohort_hit_rate(
        known,
        prior_rate=prior_rate,
        hit_pred=lambda r: is_absolute_bust(position, positional_finish(r)) is True,
    )


def _stage_by_capital(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str,
    prior_rate: Optional[float],
    tier: str = "top_12",
) -> dict:
    """position already filtered. Emit only cells with n > 0."""
    known = [
        r for r in rows_with_known_capital(rows)
        if career_stage(r.get("years_experience")) is not None
    ]
    out: dict[str, dict] = {}
    for stage in CAREER_STAGE_ORDER:
        cap_map: dict[str, dict] = {}
        for bucket in DRAFT_CAPITAL_ORDER:
            cell = [
                r for r in known
                if career_stage(r.get("years_experience")) == stage
                and r.get("draft_capital_bucket") == bucket
            ]
            if not cell:
                continue
            cap_map[bucket] = cohort_hit_rate(
                cell, tier=tier, scoring=scoring, prior_rate=prior_rate
            )
        if cap_map:
            out[stage] = cap_map
    return out


def _stage_capital_age_bucket(
    rows: Sequence[Mapping[str, Any]],
    position: str,
    *,
    scoring: str,
    prior_rate: Optional[float],
    tier: str = "top_12",
) -> dict:
    """Cross of stage × capital × UI age bucket; skip empty / missing age."""
    known = []
    for row in rows_with_known_capital(rows):
        if career_stage(row.get("years_experience")) is None:
            continue
        if age_bucket(position, row.get("age")) is None:
            continue
        known.append(row)
    out: dict[str, dict] = {}
    for stage in CAREER_STAGE_ORDER:
        by_cap: dict[str, dict] = {}
        for bucket in DRAFT_CAPITAL_ORDER:
            by_age: dict[str, dict] = {}
            for row in known:
                if career_stage(row.get("years_experience")) != stage:
                    continue
                if row.get("draft_capital_bucket") != bucket:
                    continue
                label = age_bucket(position, row.get("age"))
                by_age.setdefault(label, []).append(row)
            cap_out = {}
            for label, cell in by_age.items():
                if not cell:
                    continue
                cap_out[label] = cohort_hit_rate(
                    cell, tier=tier, scoring=scoring, prior_rate=prior_rate
                )
            if cap_out:
                by_cap[bucket] = cap_out
        if by_cap:
            out[stage] = by_cap
    return out


def _player_draft_year(career: Sequence[Mapping[str, Any]]) -> Optional[int]:
    for row in career:
        year = _optional_int(row.get("draft_year"))
        if year is not None:
            return year
    return None


def _player_capital(career: Sequence[Mapping[str, Any]]) -> Optional[str]:
    for row in career:
        bucket = row.get("draft_capital_bucket")
        if bucket in DRAFT_CAPITAL_ORDER:
            return str(bucket)
    return None


def _cumulative_hit_rate(
    rows: Sequence[Mapping[str, Any]],
    *,
    position: str,
    capital: Optional[str],
    min_exp: int,
    max_exp: int,
    tier: str,
    scoring: str,
    prior_rate: Optional[float],
    max_season: Optional[int],
) -> dict:
    """Player-level P(hit in any appeared season with exp in [min, max]).

    Qualifying players have a known ``draft_year`` such that the window has
    closed (``draft_year + max_exp <= max_season``). Missing capital is
    excluded when ``capital`` is set. Missing exp on a season cannot count
    as a hit; a closed window with no hit is a miss.
    """
    pos_rows = filter_position(rows, position)
    if capital is not None:
        pos_rows = [r for r in pos_rows if r.get("draft_capital_bucket") == capital]
    players = _group_players(pos_rows)
    qualified_n = 0
    hits = 0
    for _pid, career in players.items():
        draft_year = _player_draft_year(career)
        if draft_year is None or max_season is None:
            continue
        if draft_year + max_exp > max_season:
            continue
        if capital is not None and _player_capital(career) != capital:
            continue
        qualified_n += 1
        for row in career:
            exp = _optional_int(row.get("years_experience"))
            if exp is None or exp < min_exp or exp > max_exp:
                continue
            if is_tier_hit(row, tier=tier, scoring=scoring):
                hits += 1
                break
    rate = make_rate(hits, qualified_n, prior_rate=prior_rate, seasons=pos_rows)
    rate["n_players"] = qualified_n
    return rate


def build_draft_capital_rates(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
) -> dict:
    max_season = _max_season(rows)
    out: dict[str, dict] = {}
    for pos in SKILL_POSITIONS:
        pos_rows = filter_position(rows, pos)
        known = rows_with_known_capital(pos_rows)
        baseline = position_baseline(known, pos, scoring=scoring)
        prior = baseline.get("raw_rate")
        cumulative: dict[str, dict] = {}
        for window in CUMULATIVE_WINDOWS:
            by_cap = {}
            for bucket in DRAFT_CAPITAL_ORDER:
                by_cap[bucket] = _cumulative_hit_rate(
                    rows,
                    position=pos,
                    capital=bucket,
                    min_exp=window["min_exp"],
                    max_exp=window["max_exp"],
                    tier=window["tier"],
                    scoring=scoring,
                    prior_rate=prior,
                    max_season=max_season,
                )
            cumulative[window["id"]] = {
                "min_exp": window["min_exp"],
                "max_exp": window["max_exp"],
                "tier": window["tier"],
                "by_capital": by_cap,
            }
        out[pos] = {
            "position": pos,
            "n_known_capital": len(known),
            "n_missing_capital_excluded": len(pos_rows) - len(known),
            "baseline": baseline,
            "season_level_by_capital": _capital_season_rates(
                pos_rows, pos, scoring=scoring, prior_rate=prior
            ),
            "stage_by_capital": _stage_by_capital(
                pos_rows, scoring=scoring, prior_rate=prior
            ),
            "stage_capital_age_bucket": _stage_capital_age_bucket(
                pos_rows, pos, scoring=scoring, prior_rate=prior
            ),
            "cumulative": cumulative,
        }
    return out


# ---------------------------------------------------------------------------
# Assemble JSON payload
# ---------------------------------------------------------------------------

def assemble_profile_aggregates(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
    season_from: int = RELIABLE_SEASON_FLOOR,
    season_to: Optional[int] = None,
) -> dict:
    """Pure aggregator: warehouse records → small JSON-ready dict.

    PPR-primary. ADP hit rates are descriptive market stats (Phase 5–6), not
    ranking inputs and not comp-matching features. No projections.
    """
    era = filter_era(rows, season_from, season_to)
    bounds = season_bounds(era)
    return {
        "schema_version": 1,
        "phase": 6,
        "scoring": scoring,
        "era_floor": season_from,
        "season_range": bounds,
        "n_player_seasons": len(era),
        "descriptive_only": True,
        "definitions": {
            "breakout_rank_threshold": BREAKOUT_RANK_THRESHOLD,
            "prior_non_starter_rank": PRIOR_NON_STARTER_RANK,
            "engine_breakout": (
                "previous_season_finish is None or > 13, AND this season "
                "positional finish <= 12 (copied from the breakout engine; "
                "rank 13 is not a non-starter)"
            ),
            "first_time_elite": (
                "not previously_top12 AND this season positional finish <= 12"
            ),
            "prior_rank_source": (
                "Phase 1 previous_season_finish (last observed prior season, "
                "not a calendar year-1 join)"
            ),
            "prime_window_method": (
                "smoothed P(top-12 | integer age, position) >= known-age "
                "position baseline and n >= 15; longest consecutive run"
            ),
            "bayes_prior_n": DEFAULT_BAYES_PRIOR_N,
            "confidence": {
                "low": "<15",
                "moderate": "15-39",
                "good": "40-99",
                "strong": "100+",
            },
            "missing_age": "omitted from age curves only",
            "missing_exp": "omitted from career-stage cohorts, not labeled rookie",
            "missing_capital": "omitted from draft-capital cohorts, not labeled UDFA",
            "missing_usage": "omitted from prior-usage cohorts, not bucketed as 0",
            "missing_comp_dimension": (
                "omitted from matching; not 0 / UDFA / last-place. "
                "Rookie prior_finish is the explicit label none"
            ),
            "comps": (
                "P(this-season hit | pre-season profile). Matching uses "
                "position, career stage, draft capital, prior finish, age "
                "bucket, previous-season usage. Same-season actuals, ADP, "
                "and projections are not features. Tiny cells relax in "
                "COMP_RELAXATION_ORDER and shrink toward the position "
                "baseline. Named comps exclude the query player. Rates are "
                "pooled historical, not walk-forward. Request path reads "
                "precomputed JSON leaves; no parquet scan, no 031_* table"
            ),
            "missing_adp": "omitted from ADP cohorts; Sleeper 999 is missing, not pick 999",
            "adp": (
                "P(this-season hit | preseason redraft PPR 1QB ADP). "
                "Source order sleeper → mfl → espn → yahoo. Superflex/TEP "
                "historical ADP is not claimed. Not a comp feature and not a "
                "ranking input. Frozen snapshots are not overwritten by cron"
            ),
            "adp_in_comps": False,
            "adp_in_ranking": False,
            "snap_reliable_floor": SNAP_RELIABLE_FLOOR,
            "ftn_floor": FTN_SEASON_FLOOR,
            "prior_usage": (
                "P(this-season hit | previous-season usage bucket); "
                "same-season NGS/snaps are outcomes, not features"
            ),
            "no_adp": False,
            "no_projections": True,
        },
        "age_curves": build_age_curves(
            era, scoring=scoring, season_from=season_from, season_to=season_to
        ),
        "career_stages": build_stage_rates(era, scoring=scoring),
        "repeat_and_breakout": build_repeat_and_breakout_rates(era, scoring=scoring),
        "draft_capital": build_draft_capital_rates(era, scoring=scoring),
        "prior_usage": build_prior_usage_rates(era, scoring=scoring),
        "comps": build_comp_aggregates(era, scoring=scoring),
        "adp": build_adp_hit_rates(era, scoring=scoring),
    }
