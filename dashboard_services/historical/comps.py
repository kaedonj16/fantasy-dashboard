"""Comparable-player matching and smoothed board probabilities (pure).

Matching uses **pre-season** fields only: position, career stage, draft
capital, prior-finish bucket, age bucket, previous-season usage. Same-season
actuals, ADP, and projections are not features.

Missing dimensions are omitted (not 0 / UDFA / last-place). Tiny cells
relax in ``COMP_RELAXATION_ORDER``. When the Hist path keeps an exact cell
below ``MIN_COMP_CELL_N``, rates shrink toward a parent cell that prefers
last-year finish and age (``PARENT_MIN_N``), not every appeared player and
not declining vets mixed into a young RB1. The oldest open-ended age band
is the reverse: a 2-season 32+ cell is often one veteran repeating, so Hist
displays the broader veteran top-5 parent instead of 2/2 = 100%. Walk-forward
still uses ``MIN_COMP_CELL_N`` then the position prior. Named comps exclude
the query player.

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
    PARENT_KEEP_WEIGHT,
    PARENT_MIN_N,
    PARENT_RELAXATION_ORDERS,
    PRIOR_FINISH_ORDER,
    SKILL_POSITIONS,
    SNAP_PCT_BUCKETS,
    SNAP_RELIABLE_FLOOR,
    TARGET_SHARE_BUCKETS,
    age_bucket,
    career_stage,
    draft_capital_bucket,
    is_oldest_age_bucket,
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
    if cap not in DRAFT_CAPITAL_ORDER:
        cap = draft_capital_bucket(
            row.get("draft_round") or row.get("nfl_draft_round"),
            row.get("draft_pick") or row.get("nfl_draft_pick"),
            undrafted=bool(row.get("undrafted")),
        )
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
    order: Sequence[str] = COMP_RELAXATION_ORDER,
) -> Iterator[tuple[dict[str, str], list[str]]]:
    """Yield (active_key, dropped_dims) from most specific to position-only."""
    active = {
        dim: str(feats[dim])
        for dim in COMP_DIMENSION_ORDER
        if feats.get(dim) is not None and feats.get(dim) != ""
    }
    dropped: list[str] = []
    yield dict(active), list(dropped)
    for dim in order:
        if dim in active and dim != "position":
            del active[dim]
            dropped.append(dim)
            yield dict(active), list(dropped)


def _leaf_match_n(
    leaves: Sequence[Mapping[str, Any]],
    active: Mapping[str, str],
) -> tuple[list[Mapping[str, Any]], int]:
    matching = [leaf for leaf in leaves if key_matches(leaf.get("key") or {}, active)]
    n = sum(int(leaf.get("n") or 0) for leaf in matching)
    return matching, n


def _parent_keep_score(active: Mapping[str, str], *, weights: Optional[Mapping[str, int]] = None) -> int:
    table = weights if isinstance(weights, Mapping) else PARENT_KEEP_WEIGHT
    return sum(int(table.get(dim) or 0) for dim in active if dim != "position")


def _parent_prior_cell(
    feats: Mapping[str, Any],
    leaves: Sequence[Mapping[str, Any]],
    chosen_active: Mapping[str, str],
    baselines: Mapping[str, Mapping[str, Any]],
) -> tuple[Optional[dict[str, str]], Optional[int], Mapping[str, Mapping[str, Any]], list]:
    """Pick a Bayes prior cell that keeps last-year finish and age when it can.

    Young stars keep age so a 24-year-old RB1 is not mixed with declining
    year-6+ backs. The oldest open-ended age band is the opposite problem:
    32+ last-year top-5 TEs are often one player repeating (Kelce). Those
    queries require n >= 15 and ignore age/capital so the prior is other
    veteran TEs who were top-5 last year, not a 2/2 self-comp.
    """
    pos = str(feats.get("position") or chosen_active.get("position") or "")
    oldest = is_oldest_age_bucket(pos, feats.get("age_bucket"))
    min_n = MIN_COMP_CELL_N if oldest else PARENT_MIN_N
    weights: dict[str, int] = dict(PARENT_KEEP_WEIGHT)
    if oldest:
        weights["age_bucket"] = 0
        weights["draft_capital"] = 0
    chosen_id = tuple(sorted((chosen_active or {}).items()))
    best: Optional[tuple[int, int, dict[str, str], list]] = None
    seen: set[tuple] = set()
    for order in PARENT_RELAXATION_ORDERS:
        for active, _dropped in iter_relaxed_keys(feats, order=order):
            matching, n = _leaf_match_n(leaves, active)
            if n < min_n:
                continue
            key_id = tuple(sorted(active.items()))
            if key_id == chosen_id or key_id in seen:
                continue
            seen.add(key_id)
            score = _parent_keep_score(active, weights=weights)
            cand = (score, n, dict(active), matching)
            if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand
    if best is None:
        return None, None, baselines, []
    _score, prior_n, prior_key, matching = best
    return prior_key, prior_n, pool_leaves(matching, baselines=baselines), matching


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
            "examples": (
                []
                if examples_per_cell <= 0
                else pick_named_examples(
                    group,
                    limit=examples_per_cell if len(group) >= 2 else min(1, examples_per_cell),
                )
            ),
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

    Exact cells below ``MIN_COMP_CELL_N`` shrink toward a parent that
    prefers last-year finish and age (``PARENT_MIN_N``), not every
    last-year top-5 including declining vets. The oldest age band is the
    reverse: a 2-season 32+ cell is often one veteran repeating, so Hist
    displays the broader veteran top-5 parent instead of 2/2 = 100%.
    """
    feats = extract_comp_query(query)
    pos = feats.get("position")
    empty_rates = {
        tier: make_rate(0, 0)
        for tier in COMP_BOARD_TIERS
    }
    for rate in empty_rates.values():
        rate["kind"] = "conditional"
    empty = {
        "position": pos,
        "key_used": {},
        "dropped": [],
        "fallback": False,
        "n": 0,
        "rates": empty_rates,
        "examples": [],
        "kind": "conditional",
        "prior_source": "position_baseline",
        "prior_key": {},
        "prior_n": None,
        "profile_key": {},
        "exact_n": 0,
    }
    if not pos:
        empty["position"] = None
        return empty
    by_pos = (comps_payload.get("by_position") or {}).get(pos) or {}
    leaves = by_pos.get("leaves") or comps_payload.get("leaves") or []
    baselines = by_pos.get("baseline") or {}
    skip = exclude_sleeper_id if exclude_sleeper_id is not None else query.get("sleeper_id")

    steps: list[tuple[dict[str, str], list[str], list, int]] = []
    for active, dropped in iter_relaxed_keys(feats):
        matching, n = _leaf_match_n(leaves, active)
        steps.append((dict(active), list(dropped), matching, n))
    chosen = None
    for step in steps:
        if step[3] >= min_n:
            chosen = step
            break
    if chosen is None:
        chosen = steps[-1]
    active, dropped, matching, n = chosen
    profile_key = dict(active)
    exact_n = n
    prior_baselines = baselines
    prior_source = "position_baseline"
    prior_key: dict[str, str] = {}
    prior_n: Optional[int] = None
    if 0 < n < MIN_COMP_CELL_N:
        found_key, found_n, found_rates, found_matching = _parent_prior_cell(
            feats, leaves, active, baselines,
        )
        if found_key:
            prior_baselines = found_rates
            prior_source = "parent_cell"
            prior_key = dict(found_key)
            prior_n = found_n
            if (
                is_oldest_age_bucket(pos, feats.get("age_bucket"))
                and n < PARENT_MIN_N
                and found_n
            ):
                # Do not headline a 2/2 self-repeat at 32+/31+. Show the
                # broader veteran top-5 parent; keep the exact profile.
                active = dict(found_key)
                matching = found_matching
                n = int(found_n)
                dropped = [
                    dim for dim in COMP_RELAXATION_ORDER
                    if dim in feats and dim not in active
                ]
                rates = found_rates
                prior_source = "parent_displayed"
            else:
                rates = pool_leaves(matching, baselines=prior_baselines)
        else:
            rates = pool_leaves(matching, baselines=prior_baselines)
    else:
        rates = pool_leaves(matching, baselines=prior_baselines)
    return {
        "position": pos,
        "key_used": dict(active),
        "profile_key": profile_key,
        "dropped": list(dropped),
        "fallback": bool(dropped),
        "n": n,
        "rates": rates,
        "examples": merge_examples(matching, exclude_sleeper_id=skip),
        "kind": "conditional",
        "min_n": min_n,
        "bayes_prior_n": DEFAULT_BAYES_PRIOR_N,
        "prior_source": prior_source,
        "prior_key": prior_key,
        "prior_n": prior_n,
        "exact_n": exact_n,
    }


def build_comp_aggregates(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
    include_named: bool = True,
) -> dict:
    """Warehouse records → compact JSON section for board lookup.

    ``include_named=False`` skips example players (walk-forward rebuilds).
    Live board JSON keeps named comps on.
    """
    era = filter_era(rows)
    by_position: dict[str, dict] = {}
    all_leaves: list[dict] = []
    examples_per_cell = NAMED_EXAMPLES_PER_CELL if include_named else 0
    for pos in SKILL_POSITIONS:
        pos_rows = filter_position(era, pos)
        pos_leaves = build_comp_leaves(
            pos_rows, scoring=scoring, examples_per_cell=examples_per_cell
        )
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
        "named_examples": include_named,
        "descriptive_only": True,
        "n_leaves": len(all_leaves),
        "by_position": by_position,
    }
