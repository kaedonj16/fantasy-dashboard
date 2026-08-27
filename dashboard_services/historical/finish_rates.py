"""Cohort hit rates with sample size, smoothing, and confidence (pure).

Every rate carries ``sample_size``, ``season_range``, ``raw_rate``,
``smoothed_rate`` (empirical Bayes toward a broader prior), and a
confidence label. Empty cohorts keep ``raw_rate=None`` — never a fake 0%.
A real 0% (n>0, 0 successes) is allowed.

This module must stay dependency-free (no pandas, Flask, or I/O).
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from dashboard_services.historical.definitions import (
    DEFAULT_BAYES_PRIOR_N,
    RELIABLE_SEASON_FLOOR,
    SKILL_POSITIONS,
    TIER_CUTOFFS,
    confidence_label,
    display_percent,
    empirical_bayes,
    _optional_int,
)

HitPred = Callable[[Mapping[str, Any]], bool]


def _round_rate(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def season_bounds(rows: Iterable[Mapping[str, Any]]) -> Optional[list[int]]:
    """``[min_season, max_season]`` among rows with a parseable season."""
    seasons = []
    for row in rows:
        season = _optional_int(row.get("season"))
        if season is not None:
            seasons.append(season)
    if not seasons:
        return None
    return [min(seasons), max(seasons)]


def filter_era(
    rows: Iterable[Mapping[str, Any]],
    season_from: int = RELIABLE_SEASON_FLOOR,
    season_to: Optional[int] = None,
) -> list[dict]:
    """Keep rows in ``[season_from, season_to]``. Missing season is dropped."""
    out: list[dict] = []
    for row in rows:
        season = _optional_int(row.get("season"))
        if season is None or season < season_from:
            continue
        if season_to is not None and season > season_to:
            continue
        out.append(dict(row))
    return out


def filter_position(rows: Iterable[Mapping[str, Any]], position: Any) -> list[dict]:
    pos = str(position or "").upper()
    if pos not in SKILL_POSITIONS:
        return []
    return [dict(r) for r in rows if str(r.get("position") or "").upper() == pos]


def positional_finish(row: Mapping[str, Any], scoring: str = "ppr") -> Optional[int]:
    return _optional_int(row.get(f"{scoring}_positional_finish"))


def is_tier_hit(
    row: Mapping[str, Any],
    *,
    tier: str = "top_12",
    scoring: str = "ppr",
) -> bool:
    """True when the season finish is inside ``tier``. Unranked is a miss."""
    cutoff = TIER_CUTOFFS.get(tier)
    if cutoff is None:
        raise ValueError(f"unknown tier {tier!r}")
    finish = positional_finish(row, scoring)
    if finish is None:
        return False
    return finish <= cutoff


def make_rate(
    successes: Any,
    n: Any,
    *,
    prior_rate: Optional[float] = None,
    prior_n: int = DEFAULT_BAYES_PRIOR_N,
    seasons: Optional[Iterable[Mapping[str, Any]]] = None,
) -> dict:
    """Build the standard rate record.

    When ``prior_rate`` is omitted, ``smoothed_rate`` equals ``raw_rate``
    (the cohort *is* the prior). Empty n → rates stay None, not 0.
    """
    sample = _optional_int(n) or 0
    if sample < 0:
        sample = 0
    hits = _optional_int(successes) or 0
    if hits < 0:
        hits = 0
    raw = (hits / sample) if sample > 0 else None
    if sample > 0 and prior_rate is not None:
        prior_successes = float(prior_rate) * float(prior_n)
        smoothed = empirical_bayes(hits, sample, prior_successes, prior_n)
    else:
        smoothed = raw
    display_src = smoothed if smoothed is not None else raw
    return {
        "sample_size": sample,
        "successes": hits,
        "raw_rate": _round_rate(raw),
        "smoothed_rate": _round_rate(smoothed),
        "confidence": confidence_label(sample),
        "season_range": season_bounds(seasons or []),
        "display_pct": display_percent(display_src),
    }


def make_share(
    part: Any,
    whole: Any,
    *,
    seasons: Optional[Iterable[Mapping[str, Any]]] = None,
) -> dict:
    """Composition share (e.g. % of RB1 seasons at an age). Not a hit rate.

    No Bayes shrink — a share of a finite hit population is not a binomial
    rate toward a position baseline. ``raw_rate`` aliases ``share`` so the
    on-disk record still has the standard keys; ``smoothed_rate`` equals
    ``raw_rate``.
    """
    count = _optional_int(part) or 0
    total = _optional_int(whole) or 0
    if count < 0:
        count = 0
    if total < 0:
        total = 0
    share = (count / total) if total > 0 else None
    return {
        "count": count,
        "total": total,
        "share": _round_rate(share),
        "sample_size": total,
        "successes": count,
        "raw_rate": _round_rate(share),
        "smoothed_rate": _round_rate(share),
        "confidence": confidence_label(total),
        "season_range": season_bounds(seasons or []),
        "display_pct": display_percent(share),
        "kind": "distribution",
    }


def cohort_hit_rate(
    rows: Sequence[Mapping[str, Any]],
    *,
    tier: str = "top_12",
    scoring: str = "ppr",
    prior_rate: Optional[float] = None,
    hit_pred: Optional[HitPred] = None,
) -> dict:
    """P(hit | rows). ``hit_pred`` overrides the default tier flag."""
    if hit_pred is None:
        successes = sum(1 for row in rows if is_tier_hit(row, tier=tier, scoring=scoring))
    else:
        successes = sum(1 for row in rows if hit_pred(row))
    return make_rate(successes, len(rows), prior_rate=prior_rate, seasons=rows)


def position_baseline(
    rows: Sequence[Mapping[str, Any]],
    position: Any,
    *,
    tier: str = "top_12",
    scoring: str = "ppr",
) -> dict:
    """P(tier | position) among the given rows (no further age/capital filter)."""
    return cohort_hit_rate(
        filter_position(rows, position),
        tier=tier,
        scoring=scoring,
        prior_rate=None,
    )
