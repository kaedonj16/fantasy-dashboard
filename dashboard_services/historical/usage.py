"""Previous-season usage/efficiency overlay and rates (pure).

Same-season NGS/FTN/snap values are *outcomes* of that season (like points).
Hit rates that claim to be predictive use only **previous-season** usage.
Missing values are skipped, never bucketed as 0. Estimated snaps from
touches are not used.

This module must stay dependency-free (no pandas, Flask, nfl_data_py, or I/O).
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashboard_services.historical.definitions import (
    ADOT_BUCKETS,
    CARRIES_BUCKETS,
    COMP_BOARD_TIERS,
    EFFICIENCY_FIELDS,
    FTN_SEASON_FLOOR,
    GAMES_BUCKETS,
    NGS_SEASON_FLOOR,
    PASS_ATTEMPTS_BUCKETS,
    RECEPTIONS_BUCKETS,
    RELIABLE_SEASON_FLOOR,
    RYOE_BUCKETS,
    SKILL_POSITIONS,
    SNAP_PCT_BUCKETS,
    SNAP_RELIABLE_FLOOR,
    TARGETS_BUCKETS,
    TARGET_SHARE_BUCKETS,
    TOUCHES_BUCKETS,
    value_bucket,
    _optional_float,
    _optional_int,
)
from dashboard_services.historical.finish_rates import (
    cohort_hit_rate,
    filter_era,
    filter_position,
    is_tier_hit,
    make_share,
    position_baseline,
)

# Copied onto season S from the last observed season < S (leakage-safe).
PRIOR_USAGE_SOURCE_FIELDS: tuple[str, ...] = (
    "target_share",
    "snap_pct",
    "targets",
    "carries",
    "receptions",
    "games",
    "red_zone_targets",
    "red_zone_carries",
    "passing_attempts",
    "adot",
    "air_yards",
    "ngs_avg_separation",
    "ngs_created_separation",
    "ngs_avg_yac_above_expectation",
    "ngs_rush_yards_over_expected_per_att",
    "ngs_cpoe",
    "drop_rate",
    "contested_catch_rate",
    "receiving_epa_per_target",
)
VOLUME_COUNT_FIELDS: tuple[str, ...] = (
    "targets",
    "carries",
    "receptions",
    "games",
    "red_zone_targets",
    "red_zone_carries",
    "passing_attempts",
)
# Last-year outcome → preseason "previous_season_*" for live Trends matching.
PRESEASON_VOLUME_FROM_OUTCOME: tuple[tuple[str, str], ...] = (
    ("previous_season_carries", "carries"),
    ("previous_season_receptions", "receptions"),
    ("previous_season_targets", "targets"),
    ("previous_season_games", "games"),
    ("previous_season_passing_attempts", "passing_attempts"),
)
VOLUME_USAGE_IDS: frozenset[str] = frozenset({
    "touches",
    "carries",
    "receptions",
    "targets",
    "games",
    "pass_attempts",
})

OVERLAY_FIELDS: tuple[str, ...] = EFFICIENCY_FIELDS + ("snap_pct", "snaps")


def _created_separation(sep: Any, cushion: Any) -> Optional[float]:
    s = _optional_float(sep)
    c = _optional_float(cushion)
    if s is None or c is None:
        return None
    return round(s - c, 2)


def normalize_snap_pct(value: Any) -> Optional[float]:
    """0–1 snap rate. 0–100 inputs are scaled. Negative → None."""
    pct = _optional_float(value)
    if pct is None:
        return None
    if pct > 1.5:
        pct = pct / 100.0
    if pct < 0:
        return None
    if pct > 1.0:
        pct = 1.0
    return pct


def overlay_value_is_usable(field: str, value: Any, row: Mapping[str, Any]) -> bool:
    """Reject fake zeros: 0% snaps with real volume is not a real observation."""
    number = _optional_float(value)
    if number is None:
        return False
    if field == "snap_pct":
        pct = normalize_snap_pct(number)
        if pct is None:
            return False
        touches = 0.0
        for key in ("targets", "carries", "passing_attempts"):
            part = _optional_float(row.get(key))
            if part is not None:
                touches += part
        games = _optional_float(row.get("games")) or 0.0
        if pct == 0.0 and games >= 1 and touches > 20:
            return False
        return True
    if field == "snaps" and number <= 0:
        return False
    return True


def apply_efficiency_overlay(
    row: Mapping[str, Any],
    overlay: Optional[Mapping[str, Any]],
) -> dict:
    """Fill missing efficiency/snap fields from an overlay. Never overwrite.

    Does not write ``avg_*`` live-valuation columns. 0% snaps with real
    volume are left missing. ``air_yards`` may be derived as adot × targets
    when total air yards are absent but aDOT is known.
    """
    out = dict(row)
    overlay = overlay or {}
    for field in OVERLAY_FIELDS:
        if out.get(field) is not None:
            continue
        raw = overlay.get(field)
        if field == "adot" and raw is None:
            raw = overlay.get("avg_depth_of_target") or overlay.get("ngs_avg_intended_air_yards")
        if field == "snap_pct":
            raw = normalize_snap_pct(raw)
        if not overlay_value_is_usable(field, raw, out):
            continue
        if field == "snap_pct":
            out[field] = normalize_snap_pct(raw)
        else:
            out[field] = _optional_float(raw)
    if out.get("ngs_created_separation") is None:
        created = _created_separation(
            out.get("ngs_avg_separation") or overlay.get("ngs_avg_separation"),
            out.get("ngs_avg_cushion") or overlay.get("ngs_avg_cushion"),
        )
        if created is not None:
            out["ngs_created_separation"] = created
    if out.get("adot") is None:
        adot = _optional_float(
            overlay.get("avg_depth_of_target") or overlay.get("ngs_avg_intended_air_yards")
        )
        if adot is not None:
            out["adot"] = adot
    if out.get("air_yards") is None:
        adot = _optional_float(out.get("adot"))
        targets = _optional_float(out.get("targets"))
        if adot is not None and targets is not None and targets > 0:
            out["air_yards"] = round(adot * targets, 1)
    return out


def prior_usage_features(prev: Optional[Mapping[str, Any]]) -> dict:
    """Leakage-safe previous-season usage. Missing prev → all None, not 0."""
    out: dict[str, Any] = {
        "previous_season_year": None,
        "previous_season_touches": None,
    }
    for field in PRIOR_USAGE_SOURCE_FIELDS:
        out[f"previous_season_{field}"] = None
    if not prev:
        return out
    out["previous_season_year"] = _optional_int(prev.get("season"))
    for field in PRIOR_USAGE_SOURCE_FIELDS:
        value = prev.get(field)
        if field == "snap_pct":
            out[f"previous_season_{field}"] = normalize_snap_pct(value)
        else:
            out[f"previous_season_{field}"] = (
                _optional_int(value) if field in VOLUME_COUNT_FIELDS
                else _optional_float(value)
            )
            if field in VOLUME_COUNT_FIELDS and out[f"previous_season_{field}"] is None:
                out[f"previous_season_{field}"] = _optional_float(value)
    stamp_derived_prior_volume(out)
    return out


def stamp_derived_prior_volume(row: dict[str, Any]) -> dict[str, Any]:
    """Touches = last-year carries + receptions. Missing both stays unset."""
    carries = _optional_float(row.get("previous_season_carries"))
    recs = _optional_float(row.get("previous_season_receptions"))
    if carries is None and recs is None:
        return row
    row["previous_season_touches"] = int(round((carries or 0.0) + (recs or 0.0)))
    return row


def last_season_volume_from_outcome(row: Mapping[str, Any]) -> dict[str, Any]:
    """Map this season's totals onto next-preseason previous_season_* fields."""
    out: dict[str, Any] = {}
    for dest, src in PRESEASON_VOLUME_FROM_OUTCOME:
        val = _optional_int(row.get(src))
        if val is None:
            val = _optional_float(row.get(src))
        if val is not None:
            out[dest] = val
    stamp_derived_prior_volume(out)
    return out


def attach_prior_usage_features(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    """Attach previous-season usage using only seasons strictly before each row.

    Scoring-independent. Safe to call after ``attach_prior_career_features``.
    """
    by_player: dict[str, list] = {}
    passthrough = []
    for row in rows:
        pid = str(row.get("sleeper_id") or row.get("player_id") or "")
        if not pid:
            passthrough.append(dict(row))
            continue
        by_player.setdefault(pid, []).append(dict(row))
    out: list[dict] = []
    for career in by_player.values():
        ordered = sorted(career, key=lambda r: int(r.get("season") or 0))
        for i, row in enumerate(ordered):
            prior = [
                r for r in ordered[:i]
                if int(r.get("season") or 0) < int(row.get("season") or 0)
            ]
            featured = dict(row)
            featured.update(prior_usage_features(prior[-1] if prior else None))
            out.append(featured)
    out.extend(passthrough)
    out.sort(key=lambda r: (int(r.get("season") or 0), str(r.get("sleeper_id") or "")))
    return out


# ---------------------------------------------------------------------------
# Prior-usage hit rates (previous season → this-season finish)
# ---------------------------------------------------------------------------

USAGE_RATE_SPECS: tuple[dict, ...] = (
    {
        "id": "target_share",
        "field": "previous_season_target_share",
        "buckets": TARGET_SHARE_BUCKETS,
        "min_prior_season": NGS_SEASON_FLOOR,
        "positions": ("WR", "RB", "TE"),
    },
    {
        "id": "snap_pct",
        "field": "previous_season_snap_pct",
        "buckets": SNAP_PCT_BUCKETS,
        "min_prior_season": SNAP_RELIABLE_FLOOR,
        "positions": SKILL_POSITIONS,
    },
    {
        "id": "adot",
        "field": "previous_season_adot",
        "buckets": ADOT_BUCKETS,
        "min_prior_season": NGS_SEASON_FLOOR,
        "positions": ("WR", "TE", "RB"),
    },
    {
        "id": "ryoe",
        "field": "previous_season_ngs_rush_yards_over_expected_per_att",
        "buckets": RYOE_BUCKETS,
        "min_prior_season": NGS_SEASON_FLOOR,
        "positions": ("RB",),
    },
    {
        "id": "drop_rate",
        "field": "previous_season_drop_rate",
        "buckets": ((None, 5.0, "<5%"), (5.0, 10.0, "5-10%"), (10.0, None, "10%+")),
        "min_prior_season": FTN_SEASON_FLOOR,
        "positions": ("WR", "TE"),
    },
    {
        "id": "touches",
        "field": "previous_season_touches",
        "buckets": TOUCHES_BUCKETS,
        "min_prior_season": RELIABLE_SEASON_FLOOR,
        "positions": ("RB",),
    },
    {
        "id": "carries",
        "field": "previous_season_carries",
        "buckets": CARRIES_BUCKETS,
        "min_prior_season": RELIABLE_SEASON_FLOOR,
        "positions": ("RB",),
    },
    {
        "id": "receptions",
        "field": "previous_season_receptions",
        "buckets": RECEPTIONS_BUCKETS,
        "min_prior_season": RELIABLE_SEASON_FLOOR,
        "positions": ("WR", "TE"),
    },
    {
        "id": "targets",
        "field": "previous_season_targets",
        "buckets": TARGETS_BUCKETS,
        "min_prior_season": RELIABLE_SEASON_FLOOR,
        "positions": ("WR", "TE"),
    },
    {
        "id": "games",
        "field": "previous_season_games",
        "buckets": GAMES_BUCKETS,
        "min_prior_season": RELIABLE_SEASON_FLOOR,
        "positions": SKILL_POSITIONS,
    },
    {
        "id": "pass_attempts",
        "field": "previous_season_passing_attempts",
        "buckets": PASS_ATTEMPTS_BUCKETS,
        "min_prior_season": RELIABLE_SEASON_FLOOR,
        "positions": ("QB",),
    },
)


def prior_usage_window_pair(
    rows: Sequence[Mapping[str, Any]],
    position: Any,
    field: str,
    bucket_label: str,
    buckets: Sequence[tuple],
    *,
    tier: str = "top_12",
    scoring: str = "ppr",
    min_prior_season: Optional[int] = None,
) -> dict:
    """Distribution of hits vs conditional P(hit | previous-usage bucket)."""
    pos_rows = filter_position(rows, position)
    known = []
    for row in pos_rows:
        if value_bucket(row.get(field), buckets) is None:
            continue
        prior_year = _optional_int(row.get("previous_season_year"))
        if min_prior_season is not None and (prior_year is None or prior_year < min_prior_season):
            continue
        known.append(row)
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


def build_prior_usage_rates(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
    tier: str = "top_12",
    season_from: int = NGS_SEASON_FLOOR,
    ids: Optional[Sequence[str]] = None,
) -> dict:
    era = [stamp_derived_prior_volume(row) for row in filter_era(rows, season_from)]
    wanted = set(ids) if ids is not None else None
    out: dict[str, dict] = {}
    for spec in USAGE_RATE_SPECS:
        if wanted is not None and spec["id"] not in wanted:
            continue
        metric = {
            "field": spec["field"],
            "min_prior_season": spec["min_prior_season"],
            "by_position": {},
        }
        for pos in spec["positions"]:
            pos_rows = filter_position(era, pos)
            known = []
            for row in pos_rows:
                if value_bucket(row.get(spec["field"]), spec["buckets"]) is None:
                    continue
                prior_year = _optional_int(row.get("previous_season_year"))
                if prior_year is None or prior_year < spec["min_prior_season"]:
                    continue
                known.append(row)
            hits = [r for r in known if is_tier_hit(r, tier=tier, scoring=scoring)]
            baseline = position_baseline(known, pos, tier=tier, scoring=scoring)
            by_bucket = {}
            for _lo, _hi, label in spec["buckets"]:
                by_bucket[label] = prior_usage_window_pair(
                    era,
                    pos,
                    spec["field"],
                    label,
                    spec["buckets"],
                    tier=tier,
                    scoring=scoring,
                    min_prior_season=spec["min_prior_season"],
                )
            metric["by_position"][pos] = {
                "n_known": len(known),
                "n_missing_excluded": len(pos_rows) - len(known),
                "baseline": baseline,
                "n_hits": len(hits),
                "by_bucket": by_bucket,
            }
        out[spec["id"]] = metric
    return out


def _volume_only(rates: Mapping[str, Any]) -> dict[str, Any]:
    return {key: rates[key] for key in VOLUME_USAGE_IDS if key in rates}


def build_usage_volume_overlay(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
) -> dict[str, Any]:
    """Sidecar prior-usage rates + live last-year totals. Does not rewrite aggregates."""
    prepared = [stamp_derived_prior_volume(dict(row)) for row in rows]
    prior_usage = _volume_only(
        build_prior_usage_rates(prepared, scoring=scoring, ids=tuple(VOLUME_USAGE_IDS))
    )
    prior_usage_by_tier = {
        tier: _volume_only(
            build_prior_usage_rates(
                prepared, scoring=scoring, tier=tier, ids=tuple(VOLUME_USAGE_IDS)
            )
        )
        for tier in COMP_BOARD_TIERS
    }
    latest: dict[str, dict] = {}
    for row in prepared:
        pid = str(row.get("sleeper_id") or row.get("player_id") or "")
        if not pid:
            continue
        season = _optional_int(row.get("season")) or -1
        prev = latest.get(pid)
        if prev is None or season >= (_optional_int(prev.get("season")) or -1):
            latest[pid] = row
    preseason_volume: dict[str, dict] = {}
    for pid, row in latest.items():
        extra = last_season_volume_from_outcome(row)
        if extra:
            preseason_volume[pid] = extra
    return {
        "prior_usage": prior_usage,
        "prior_usage_by_tier": prior_usage_by_tier,
        "preseason_volume": preseason_volume,
        "n_players": len(preseason_volume),
    }
