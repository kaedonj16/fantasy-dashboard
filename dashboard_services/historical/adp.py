"""Historical redraft-PPR ADP features and hit rates (pure).

ADP is a **preseason market** signal, not a matching feature for comps and not
a ranking input. Missing / Sleeper-999 / non-positive values are omitted,
never bucketed as 0. Superflex and TEP historical ADP are not claimed.

This module must stay dependency-free (no pandas, Flask, nfl_data_py, or I/O).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashboard_services.historical.definitions import (
    ADP_OVERALL_BUCKETS,
    ADP_SOURCE_PREFERENCE,
    PRIOR_FINISH_BUCKETS,
    SKILL_POSITIONS,
    adp_overall_bucket,
    is_adp_relative_bust,
    normalize_adp,
    value_bucket,
    _optional_int,
)
from dashboard_services.historical.finish_rates import (
    cohort_hit_rate,
    filter_era,
    filter_position,
    is_tier_hit,
    make_rate,
    make_share,
    positional_finish,
    position_baseline,
)
from dashboard_services.historical.finishes import competition_ranks

ADP_FEATURE_FIELDS: tuple[str, ...] = (
    "adp_overall",
    "adp_source",
    "adp_bucket",
    "adp_positional",
    "adp_positional_bucket",
)


def resolve_player_adp(source_values: Mapping[str, Any]) -> tuple[Optional[float], Optional[str]]:
    """Pick one overall ADP from per-source values using ``ADP_SOURCE_PREFERENCE``."""
    for source in ADP_SOURCE_PREFERENCE:
        adp = normalize_adp(source_values.get(source))
        if adp is not None:
            return adp, source
    return None, None


def attach_adp_features(
    rows: Sequence[Mapping[str, Any]],
    adp_by_season: Mapping[Any, Mapping[str, Mapping[str, Any]]],
) -> list[dict]:
    """Copy preseason ADP onto player-season rows. Does not read finishes.

    ``adp_by_season[season][source][sleeper_id] -> overall pick``.
    """
    attached: list[dict] = []
    for row in rows:
        rec = dict(row)
        season = _optional_int(row.get("season"))
        pid = str(row.get("sleeper_id") or "")
        source_vals: dict[str, Any] = {}
        if season is not None and pid:
            season_maps = adp_by_season.get(season) or adp_by_season.get(str(season)) or {}
            for source, mapping in (season_maps or {}).items():
                if isinstance(mapping, Mapping):
                    source_vals[str(source)] = mapping.get(pid)
        adp, source = resolve_player_adp(source_vals)
        rec["adp_overall"] = adp
        rec["adp_source"] = source
        rec["adp_bucket"] = adp_overall_bucket(adp)
        attached.append(rec)
    return assign_positional_adp(attached)


def assign_positional_adp(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Competition-rank ADP within season × position. Missing ADP stays None."""
    groups: dict[tuple, list[int]] = {}
    for i, row in enumerate(rows):
        season = _optional_int(row.get("season"))
        pos = str(row.get("position") or "").upper()
        if season is None or pos not in SKILL_POSITIONS:
            continue
        groups.setdefault((season, pos), []).append(i)
    out = [dict(row) for row in rows]
    for idxs in groups.values():
        values = [normalize_adp(out[i].get("adp_overall")) for i in idxs]
        # Lower pick is better; competition_ranks ranks high-to-low, so negate.
        ranks = competition_ranks(
            [(-v if v is not None else None) for v in values]
        )
        for i, rank in zip(idxs, ranks):
            out[i]["adp_positional"] = rank
            out[i]["adp_positional_bucket"] = (
                value_bucket(rank, PRIOR_FINISH_BUCKETS) if rank is not None else None
            )
    for rec in out:
        rec.setdefault("adp_positional", None)
        rec.setdefault("adp_positional_bucket", None)
    return out


def _adp_window_pair(
    rows: Sequence[Mapping[str, Any]],
    position: Any,
    field: str,
    bucket_label: str,
    buckets: Sequence[tuple],
    *,
    tier: str = "top_12",
    scoring: str = "ppr",
) -> dict:
    pos_rows = filter_position(rows, position)
    known = [r for r in pos_rows if value_bucket(r.get(field), buckets) is not None]
    hits = [r for r in known if is_tier_hit(r, tier=tier, scoring=scoring)]
    in_bucket = [r for r in known if value_bucket(r.get(field), buckets) == bucket_label]
    hits_in = [r for r in in_bucket if is_tier_hit(r, tier=tier, scoring=scoring)]
    baseline = position_baseline(known, position, tier=tier, scoring=scoring)
    return {
        "position": str(position or "").upper(),
        "field": field,
        "bucket": bucket_label,
        "n_known": len(known),
        "n_hits": len(hits),
        "distribution": make_share(len(hits_in), len(hits), seasons=hits),
        "conditional": cohort_hit_rate(
            in_bucket, tier=tier, scoring=scoring, prior_rate=baseline.get("raw_rate")
        ),
    }


def _bust_rate(rows: Sequence[Mapping[str, Any]], *, cutoff: int = 12) -> dict:
    known = []
    for row in rows:
        flag = is_adp_relative_bust(row.get("adp_positional"), positional_finish(row), cutoff=cutoff)
        if flag is None:
            continue
        known.append(row)
    return cohort_hit_rate(
        known,
        hit_pred=lambda r: is_adp_relative_bust(
            r.get("adp_positional"), positional_finish(r), cutoff=cutoff
        )
        is True,
    )


def build_adp_hit_rates(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
    tier: str = "top_12",
) -> dict:
    """P(hit | preseason ADP) plus distribution of hits across ADP buckets."""
    era = filter_era(rows)
    by_position: dict[str, dict] = {}
    coverage: dict[str, dict] = {}
    for row in era:
        season = _optional_int(row.get("season"))
        if season is None:
            continue
        cov = coverage.setdefault(str(season), {"n": 0, "n_with_adp": 0, "by_source": {}})
        cov["n"] += 1
        if normalize_adp(row.get("adp_overall")) is not None:
            cov["n_with_adp"] += 1
            src = str(row.get("adp_source") or "unknown")
            cov["by_source"][src] = cov["by_source"].get(src, 0) + 1

    for pos in SKILL_POSITIONS:
        pos_rows = filter_position(era, pos)
        known = [r for r in pos_rows if normalize_adp(r.get("adp_overall")) is not None]
        baseline = position_baseline(known, pos, tier=tier, scoring=scoring)
        by_overall = {}
        for _lo, _hi, label in ADP_OVERALL_BUCKETS:
            by_overall[label] = _adp_window_pair(
                era, pos, "adp_overall", label, ADP_OVERALL_BUCKETS,
                tier=tier, scoring=scoring,
            )
        by_pos_adp = {}
        for _lo, _hi, label in PRIOR_FINISH_BUCKETS:
            by_pos_adp[label] = _adp_window_pair(
                era, pos, "adp_positional", label, PRIOR_FINISH_BUCKETS,
                tier=tier, scoring=scoring,
            )
        by_position[pos] = {
            "position": pos,
            "n_with_adp": len(known),
            "n_missing_excluded": len(pos_rows) - len(known),
            "baseline": baseline,
            "by_overall_bucket": by_overall,
            "by_positional_bucket": by_pos_adp,
            "adp_relative_bust": _bust_rate(known),
        }
    return {
        "axis": "redraft",
        "scoring": scoring,
        "qb_format": "1qb",
        "source_preference": list(ADP_SOURCE_PREFERENCE),
        "hit_tier": tier,
        "sf_tep_historical": False,
        "pooled_historical": True,
        "descriptive_only": True,
        "by_position": by_position,
        "coverage_by_season": coverage,
    }
