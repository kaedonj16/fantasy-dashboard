"""Comparable-player matching and smoothed board probabilities (pure).

Matching uses **pre-season** fields only: position, career stage, draft
capital, prior-finish bucket, age bucket, previous-season usage. Same-season
actuals, ADP, and projections are not features.

Missing dimensions are omitted (not 0 / UDFA / last-place). Tiny cells
relax in ``COMP_RELAXATION_ORDER`` and shrink toward the position baseline
via empirical Bayes. Named comps exclude the query player.

Cell rates are **pooled historical** (all warehouse seasons), not
walk-forward. They are descriptive and do not enter ranking or Pick Score.

Request paths look up precomputed leaves in
``historical_profile_aggregates.json``. This module does not read parquet.
"""
from __future__ import annotations

from typing import Any, Iterable, Iterator, Mapping, Optional, Sequence

from dashboard_services.historical.definitions import (
    CAREER_STAGE_ORDER,
    COMP_BOARD_TIERS,
    COMP_DIMENSION_ORDER,
    COMP_RELAXATION_ORDER,
    DEFAULT_BAYES_PRIOR_N,
    DRAFT_CAPITAL_ORDER,
    MIN_COMP_CELL_N,
    NAMED_EXAMPLES_PER_CELL,
    PRIOR_FINISH_ORDER,
    SKILL_POSITIONS,
    SNAP_PCT_BUCKETS,
    SNAP_RELIABLE_FLOOR,
    TARGET_SHARE_BUCKETS,
    age_bucket,
    career_stage,
    prior_finish_bucket,
    value_bucket,
    _optional_float,
    _optional_int,
)
from dashboard_services.historical.finish_rates import (
    filter_era,
    filter_position,
    is_tier_hit,
    make_rate,
    positional_finish,
    position_baseline,
    season_bounds,
)

# Pre-season matching fields. Same-season volume / points / NGS are outcomes.
COMP_FEATURE_FIELDS: tuple[str, ...] = COMP_DIMENSION_ORDER


def extract_comp_query(row: Mapping[str, Any]) -> dict[str, str]:
    """Pre-season matching key. Missing dimensions are omitted, never faked.

    Accepts a warehouse row (raw fields) or an already-bucketed key. Same-season
    ``target_share`` / ``snap_pct`` / finishes are ignored.
    """
    pos = str(row.get("position") or "").upper()
    if pos not in SKILL_POSITIONS:
        return {}
    feats: dict[str, str] = {"position": pos}

    stage = career_stage(row.get("years_experience"))
    if stage is None:
        raw_stage = row.get("career_stage")
        if raw_stage in CAREER_STAGE_ORDER:
            stage = str(raw_stage)
    if stage is not None:
        feats["career_stage"] = stage

    cap = row.get("draft_capital_bucket")
    if cap not in DRAFT_CAPITAL_ORDER:
        cap = row.get("draft_capital")
    if cap in DRAFT_CAPITAL_ORDER:
        feats["draft_capital"] = str(cap)

    prior = prior_finish_bucket(
        row.get("previous_season_finish"),
        years_experience=row.get("years_experience"),
    )
    if prior is None:
        raw_prior = row.get("prior_finish")
        if raw_prior in PRIOR_FINISH_ORDER:
            prior = str(raw_prior)
    if prior is not None:
        feats["prior_finish"] = prior

    bucket = age_bucket(pos, row.get("age"))
    if bucket is None:
        raw_age = row.get("age_bucket")
        if isinstance(raw_age, str) and raw_age:
            bucket = raw_age
    if bucket is not None:
        feats["age_bucket"] = bucket

    if pos != "QB":
        share = value_bucket(row.get("previous_season_target_share"), TARGET_SHARE_BUCKETS)
        if share is None:
            raw_share = row.get("target_share")
            # Only accept a bucket label, never a same-season rate (0–1 float).
            if isinstance(raw_share, str) and raw_share:
                share = raw_share
        if share is not None:
            feats["target_share"] = share

    prior_year = _optional_int(row.get("previous_season_year"))
    snap = None
    if prior_year is not None and prior_year >= SNAP_RELIABLE_FLOOR:
        snap = value_bucket(row.get("previous_season_snap_pct"), SNAP_PCT_BUCKETS)
    if snap is None:
        raw_snap = row.get("snap_pct")
        if isinstance(raw_snap, str) and raw_snap:
            snap = raw_snap
    if snap is not None:
        feats["snap_pct"] = snap

    return feats


def cell_id(key: Mapping[str, Any]) -> str:
    """Stable id for a matching key. Only present dimensions are included."""
    parts = []
    for dim in COMP_DIMENSION_ORDER:
        value = key.get(dim)
        if value is not None and value != "":
            parts.append(f"{dim}={value}")
    return "|".join(parts)


def key_matches(leaf_key: Mapping[str, Any], required: Mapping[str, Any]) -> bool:
    """True when ``leaf_key`` agrees on every dimension in ``required``.

    Extra leaf dimensions are allowed (a finer cell still matches a coarser
    query). A missing leaf value does not match a required value.
    """
    for dim, value in required.items():
        if dim not in COMP_DIMENSION_ORDER:
            continue
        if value is None or value == "":
            continue
        if leaf_key.get(dim) != value:
            return False
    return True


def iter_relaxed_keys(
    feats: Mapping[str, Any],
) -> Iterator[tuple[dict[str, str], list[str]]]:
    """Yield (active_key, dropped_dims) from most specific to position-only."""
    active = {
        dim: str(feats[dim])
        for dim in COMP_DIMENSION_ORDER
        if feats.get(dim) is not None and feats.get(dim) != ""
    }
    dropped: list[str] = []
    yield dict(active), list(dropped)
    for dim in COMP_RELAXATION_ORDER:
        if dim in active and dim != "position":
            del active[dim]
            dropped.append(dim)
            yield dict(active), list(dropped)


def _example_record(row: Mapping[str, Any]) -> dict[str, Any]:
    pts = _optional_float(row.get("ppr_points"))
    return {
        "sleeper_id": str(row.get("sleeper_id") or ""),
        "name": row.get("name"),
        "season": _optional_int(row.get("season")),
        "positional_finish": positional_finish(row),
        "ppr_points": round(pts, 1) if pts is not None else None,
    }


def pick_named_examples(
    rows: Sequence[Mapping[str, Any]],
    *,
    limit: int = NAMED_EXAMPLES_PER_CELL,
    exclude_sleeper_id: Any = None,
) -> list[dict]:
    """A few notable seasons from a cell. Outcomes here are the comps' results."""
    skip = str(exclude_sleeper_id or "")
    ranked = sorted(
        rows,
        key=lambda r: (
            0 if is_tier_hit(r, tier="top_12") else 1,
            -(_optional_float(r.get("ppr_points")) or 0.0),
        ),
    )
    out: list[dict] = []
    seen: set[str] = set()
    for row in ranked:
        sid = str(row.get("sleeper_id") or "")
        if not sid or sid == skip or sid in seen:
            continue
        seen.add(sid)
        out.append(_example_record(row))
        if len(out) >= limit:
            break
    return out


def _tiebreak(query: Mapping[str, Any], cand: Mapping[str, Any]) -> tuple:
    q_age = _optional_float(query.get("age"))
    c_age = _optional_float(cand.get("age"))
    if q_age is not None and c_age is not None:
        age_d = abs(q_age - c_age)
    else:
        age_d = 100.0
    q_fin = _optional_int(query.get("previous_season_finish"))
    c_fin = _optional_int(cand.get("previous_season_finish"))
    if q_fin is not None and c_fin is not None:
        fin_d = abs(q_fin - c_fin)
    else:
        fin_d = 100
    pts = _optional_float(cand.get("ppr_points")) or 0.0
    season = _optional_int(cand.get("season")) or 0
    return (age_d, fin_d, -pts, -season)


def match_comps(
    query: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    *,
    limit: int = 8,
    as_of_season: Optional[int] = None,
) -> list[dict]:
    """Named comparable player-seasons for a query row.

    Excludes the query player. ``as_of_season`` (default: query season) drops
    later seasons so a historical query cannot use the future. Same-season
    other players are allowed — their matching features are still pre-season.

    Relaxation fills the list when the exact cell is tiny; it does not invent
    players. This primitive is for tests and in-memory rebuilds. Request paths
    should use ``lookup_board_probabilities`` on the precomputed JSON.
    """
    qf = extract_comp_query(query)
    if "position" not in qf:
        return []
    qid = str(query.get("sleeper_id") or "")
    qseason = _optional_int(query.get("season"))
    as_of = as_of_season if as_of_season is not None else qseason

    candidates: list[tuple[dict, dict[str, str]]] = []
    for row in pool:
        sid = str(row.get("sleeper_id") or "")
        if qid and sid == qid:
            continue
        if as_of is not None:
            season = _optional_int(row.get("season"))
            if season is None or season > as_of:
                continue
        cf = extract_comp_query(row)
        if cf.get("position") != qf["position"]:
            continue
        candidates.append((dict(row), cf))

    selected: list[dict] = []
    seen: set[tuple] = set()
    for active, dropped in iter_relaxed_keys(qf):
        batch = [
            (row, cf)
            for row, cf in candidates
            if (row.get("sleeper_id"), row.get("season")) not in seen
            and key_matches(cf, active)
        ]
        batch.sort(key=lambda pair: _tiebreak(query, pair[0]))
        for row, _cf in batch:
            ident = (row.get("sleeper_id"), row.get("season"))
            seen.add(ident)
            rec = _example_record(row)
            rec["matched_key"] = dict(active)
            rec["dropped"] = list(dropped)
            selected.append(rec)
            if len(selected) >= limit:
                return selected
    return selected


def _successes(rows: Sequence[Mapping[str, Any]], scoring: str) -> dict[str, int]:
    return {
        tier: sum(1 for row in rows if is_tier_hit(row, tier=tier, scoring=scoring))
        for tier in COMP_BOARD_TIERS
    }


def build_comp_leaves(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
    examples_per_cell: int = NAMED_EXAMPLES_PER_CELL,
) -> list[dict]:
    """Finest-grain cells: one leaf per unique present-dimension signature."""
    groups: dict[str, list[Mapping[str, Any]]] = {}
    keys: dict[str, dict[str, str]] = {}
    for row in rows:
        feats = extract_comp_query(row)
        if "position" not in feats:
            continue
        cid = cell_id(feats)
        groups.setdefault(cid, []).append(row)
        keys[cid] = feats
    leaves = []
    for cid in sorted(keys):
        group = groups[cid]
        bounds = season_bounds(group)
        leaves.append({
            "id": cid,
            "key": keys[cid],
            "n": len(group),
            "successes": _successes(group, scoring),
            "season_range": bounds,
            "examples": pick_named_examples(group, limit=examples_per_cell)
            if len(group) >= 2
            else pick_named_examples(group, limit=1),
        })
    return leaves


def _merge_season_range(leaves: Iterable[Mapping[str, Any]]) -> Optional[list[int]]:
    lo: Optional[int] = None
    hi: Optional[int] = None
    for leaf in leaves:
        bounds = leaf.get("season_range") or []
        if len(bounds) < 2:
            continue
        a, b = _optional_int(bounds[0]), _optional_int(bounds[1])
        if a is None or b is None:
            continue
        lo = a if lo is None else min(lo, a)
        hi = b if hi is None else max(hi, b)
    if lo is None or hi is None:
        return None
    return [lo, hi]


def pool_leaves(
    leaves: Sequence[Mapping[str, Any]],
    *,
    baselines: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict]:
    """Sum leaf successes/n and Bayes-smooth toward the position baseline."""
    n = sum(int(leaf.get("n") or 0) for leaf in leaves)
    dummy = []
    bounds = _merge_season_range(leaves)
    if bounds:
        dummy = [{"season": bounds[0]}, {"season": bounds[1]}]
    rates = {}
    for tier in COMP_BOARD_TIERS:
        hits = sum(int((leaf.get("successes") or {}).get(tier) or 0) for leaf in leaves)
        prior = (baselines.get(tier) or {}).get("raw_rate")
        rate = make_rate(hits, n, prior_rate=prior, seasons=dummy)
        rate["kind"] = "conditional"
        rates[tier] = rate
    return rates


def merge_examples(
    leaves: Sequence[Mapping[str, Any]],
    *,
    limit: int = NAMED_EXAMPLES_PER_CELL,
    exclude_sleeper_id: Any = None,
) -> list[dict]:
    skip = str(exclude_sleeper_id or "")
    ranked: list[dict] = []
    for leaf in leaves:
        for ex in leaf.get("examples") or []:
            if skip and str(ex.get("sleeper_id") or "") == skip:
                continue
            ranked.append(dict(ex))
    ranked.sort(
        key=lambda ex: (
            0 if (_optional_int(ex.get("positional_finish")) or 999) <= 12 else 1,
            -(_optional_float(ex.get("ppr_points")) or 0.0),
        )
    )
    out: list[dict] = []
    seen: set[str] = set()
    for ex in ranked:
        sid = str(ex.get("sleeper_id") or "")
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(ex)
        if len(out) >= limit:
            break
    return out


def lookup_board_probabilities(
    query: Mapping[str, Any],
    comps_payload: Mapping[str, Any],
    *,
    min_n: int = MIN_COMP_CELL_N,
    exclude_sleeper_id: Any = None,
) -> dict:
    """Smoothed P(hit) from precomputed leaves. Never scans parquet.

    Walks ``COMP_RELAXATION_ORDER`` until the pooled cell reaches ``min_n``
    (or only position remains). Empty / unknown position → rates stay None.
    """
    feats = extract_comp_query(query)
    pos = feats.get("position")
    empty_rates = {
        tier: make_rate(0, 0)
        for tier in COMP_BOARD_TIERS
    }
    for rate in empty_rates.values():
        rate["kind"] = "conditional"
    if not pos:
        return {
            "position": None,
            "key_used": {},
            "dropped": [],
            "fallback": False,
            "n": 0,
            "rates": empty_rates,
            "examples": [],
            "kind": "conditional",
        }
    by_pos = (comps_payload.get("by_position") or {}).get(pos) or {}
    leaves = by_pos.get("leaves") or comps_payload.get("leaves") or []
    baselines = by_pos.get("baseline") or {}
    skip = exclude_sleeper_id if exclude_sleeper_id is not None else query.get("sleeper_id")

    last: Optional[tuple] = None
    for active, dropped in iter_relaxed_keys(feats):
        matching = [leaf for leaf in leaves if key_matches(leaf.get("key") or {}, active)]
        n = sum(int(leaf.get("n") or 0) for leaf in matching)
        last = (active, dropped, matching, n)
        if n >= min_n:
            break
    assert last is not None
    active, dropped, matching, n = last
    rates = pool_leaves(matching, baselines=baselines)
    return {
        "position": pos,
        "key_used": dict(active),
        "dropped": list(dropped),
        "fallback": bool(dropped),
        "n": n,
        "rates": rates,
        "examples": merge_examples(matching, exclude_sleeper_id=skip),
        "kind": "conditional",
        "min_n": min_n,
        "bayes_prior_n": DEFAULT_BAYES_PRIOR_N,
    }


def build_comp_aggregates(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
) -> dict:
    """Warehouse records → compact JSON section for board lookup."""
    era = filter_era(rows)
    by_position: dict[str, dict] = {}
    all_leaves: list[dict] = []
    for pos in SKILL_POSITIONS:
        pos_rows = filter_position(era, pos)
        pos_leaves = build_comp_leaves(pos_rows, scoring=scoring)
        all_leaves.extend(pos_leaves)
        baseline = {
            tier: position_baseline(pos_rows, pos, tier=tier, scoring=scoring)
            for tier in COMP_BOARD_TIERS
        }
        for rate in baseline.values():
            rate["kind"] = "conditional"
        by_position[pos] = {
            "position": pos,
            "n_rows": len(pos_rows),
            "n_leaves": len(pos_leaves),
            "baseline": baseline,
            "leaves": pos_leaves,
        }
    return {
        "min_cell_n": MIN_COMP_CELL_N,
        "relaxation_order": list(COMP_RELAXATION_ORDER),
        "dimensions": list(COMP_DIMENSION_ORDER),
        "board_tiers": list(COMP_BOARD_TIERS),
        "pooled_historical": True,
        "walk_forward": False,
        "descriptive_only": True,
        "n_leaves": len(all_leaves),
        "by_position": by_position,
    }
