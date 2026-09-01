"""True multi-factor historical cohorts from a compact observation index (pure).

Selected Trends buckets are combined with the same OR-within-group /
AND-across-group predicates as Scout. Hit rates are counted from actual
matching player-seasons. Intersections are never estimated by multiplying
marginal probabilities.

The request path reads a precomputed compact index on the aggregates JSON.
It does not scan parquet, send the warehouse to the browser, or enter
ranking / Pick Score / VOR / Draft Grade.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from dashboard_services.historical.definitions import (
    COMP_BOARD_TIERS,
    DEFAULT_BAYES_PRIOR_N,
    EDGE_RANK_PRIOR_N,
    MARKET_ADP_MIN_N,
    MARKET_ADP_MIN_SHARE,
    NAMED_CLOSEST_EXAMPLES,
    SIGNAL_BOARD_TIER,
    SKILL_POSITIONS,
    SNAP_RELIABLE_FLOOR,
    TIER_CUTOFFS,
    TRAJECTORY_SNAP_DOWN,
    TRAJECTORY_SNAP_UP,
    TRAJECTORY_TARGET_SHARE_DOWN,
    TRAJECTORY_TARGET_SHARE_UP,
    TRAJECTORY_WORKLOAD_DOWN,
    TRAJECTORY_WORKLOAD_UP,
    confidence_label,
    display_percent,
    normalize_adp,
    ranking_adjusted_rate,
    wilson_interval,
    _optional_float,
    _optional_int,
)
from dashboard_services.historical.filters import (
    canonical_filter_key,
    extract_trend_features,
    matches_filter_groups,
    matches_trend_filter,
    trajectory_buckets,
)
from dashboard_services.historical.finish_rates import (
    make_rate,
    positional_finish,
)
from dashboard_services.historical.signals import probability_from_rate

_COHORT_CACHE: dict[tuple, dict] = {}
_CACHE_VERSION: Any = None
_CACHE_MAX = 256

CLOSEST_EXAMPLE_LIMIT = NAMED_CLOSEST_EXAMPLES

HIT_TIER_LABELS: dict[str, str] = {
    "top_5": "Top 5",
    "top_12": "Top 12",
    "top_24": "Top 24",
    "miss": "Outside top 24",
}

FINISH_TIER_COPY: dict[str, str] = {
    "QB": (
        "Top 5 is the league-winner line. Top 12 is a weekly starter. "
        "Top 24 is the streaming line."
    ),
    "RB": (
        "Top 5 is the league-winner line. Top 12 is a starter. "
        "Top 24 is the flex line."
    ),
    "WR": (
        "Top 5 is the league-winner line. Top 12 is a WR1. "
        "Top 24 is the flex line."
    ),
    "TE": (
        "Top 5 is the league-winner line. Top 12 is a weekly starter. "
        "Top 24 is a deep TE2, not a flex line."
    ),
}

CONFIDENCE_SHORT: dict[str, str] = {
    "low": "small",
    "moderate": "moderate",
    "good": "solid",
    "strong": "large",
}

TRAJECTORY_CONTRACT: dict[str, Any] = {
    "kind": "pre_outcome_yoy",
    "leakage": False,
    "requires_consecutive_prior_seasons": True,
    "for_season_s": "only seasons S-2 and S-1; never season S actuals",
    "snap_pct_floor": SNAP_RELIABLE_FLOOR,
    "buckets": {
        "target_share_change": [TRAJECTORY_TARGET_SHARE_UP, TRAJECTORY_TARGET_SHARE_DOWN],
        "snap_pct_change": [TRAJECTORY_SNAP_UP, TRAJECTORY_SNAP_DOWN],
        "workload_change": [TRAJECTORY_WORKLOAD_UP, TRAJECTORY_WORKLOAD_DOWN],
    },
    "missing": "omitted, never 0",
}


def _round_rate(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def _pts(value: Optional[float]) -> Optional[int]:
    pct = display_percent(value)
    return pct


def _edge_pts(rate: Optional[float], baseline: Optional[float]) -> Optional[int]:
    if rate is None or baseline is None:
        return None
    return int(round((float(rate) - float(baseline)) * 100.0))


def _successes_n(rec: Mapping[str, Any]) -> tuple[Optional[int], Optional[int]]:
    n = _optional_int(rec.get("sample_size") if rec.get("sample_size") is not None else rec.get("n"))
    hits = _optional_int(rec.get("successes"))
    if hits is None and n is not None and rec.get("raw_rate") is not None:
        raw = _optional_float(rec.get("raw_rate"))
        if raw is not None:
            hits = int(round(raw * n))
    return hits, n


def edge_bundle(
    rec: Mapping[str, Any],
    baseline_rate: Optional[float],
) -> dict[str, Any]:
    """raw / adjusted / ranking edges versus a positional baseline.

    ``adjusted_rate`` uses the ranking prior (k=30) so small samples shrink
    harder than the table's k=10 ``smoothed_rate``. ``smoothed_rate`` on
    ``rec`` is left unchanged.
    """
    hits, n = _successes_n(rec)
    raw = _optional_float(rec.get("raw_rate"))
    smoothed = _optional_float(rec.get("smoothed_rate"))
    ranking = ranking_adjusted_rate(hits, n, baseline_rate) if hits is not None else None
    ci_lo, ci_hi = wilson_interval(hits, n)
    return {
        "raw_rate": _round_rate(raw),
        "smoothed_rate": _round_rate(smoothed),
        "adjusted_rate": _round_rate(ranking),
        "sample_size": n,
        "successes": hits,
        "baseline_rate": _round_rate(baseline_rate),
        "raw_edge": _round_rate((raw - baseline_rate) if raw is not None and baseline_rate is not None else None),
        "adjusted_edge": _round_rate(
            (ranking - baseline_rate) if ranking is not None and baseline_rate is not None else None
        ),
        "raw_edge_pts": _edge_pts(raw, baseline_rate),
        "adjusted_edge_pts": _edge_pts(ranking, baseline_rate),
        "confidence": rec.get("confidence") or confidence_label(n),
        "ci_low": _round_rate(ci_lo),
        "ci_high": _round_rate(ci_hi),
        "ci_low_pct": display_percent(ci_lo),
        "ci_high_pct": display_percent(ci_hi),
        "ranking_prior_n": EDGE_RANK_PRIOR_N,
        "display_prior_n": DEFAULT_BAYES_PRIOR_N,
    }


def attach_row_edges(row: dict, rec: Mapping[str, Any], baseline_rate: Optional[float]) -> dict:
    """Stamp ranking fields onto a Trends table row. Display pct is unchanged."""
    bundle = edge_bundle(rec, baseline_rate)
    if bundle.get("adjusted_edge_pts") is not None:
        row["ranking_edge"] = bundle["adjusted_edge_pts"]
        row["adjusted_edge"] = bundle["adjusted_edge_pts"]
        row["raw_edge"] = bundle.get("raw_edge_pts")
        row["adjusted_rate"] = bundle.get("adjusted_rate")
        row["raw_rate"] = bundle.get("raw_rate")
        row["baseline_rate"] = bundle.get("baseline_rate")
    if bundle.get("ci_low_pct") is not None:
        row["ci_low"] = bundle["ci_low_pct"]
        row["ci_high"] = bundle["ci_high_pct"]
    return row


def _obs_pid(row: Mapping[str, Any]) -> str:
    return str(row.get("sleeper_id") or row.get("player_id") or row.get("pid") or "").strip()


def _compact_observation(row: Mapping[str, Any], extra_feats: Optional[Mapping[str, Any]] = None) -> Optional[dict]:
    pos = str(row.get("position") or "").upper()
    if pos not in SKILL_POSITIONS:
        return None
    pid = _obs_pid(row)
    season = _optional_int(row.get("season"))
    if not pid or season is None:
        return None
    seed = dict(row)
    if extra_feats:
        seed.update(dict(extra_feats))
    feats = extract_trend_features(seed)
    if not feats:
        return None
    finish = positional_finish(row)
    adp = normalize_adp(row.get("adp_overall") or row.get("adp"))
    rec: dict[str, Any] = {
        "pid": pid,
        "name": row.get("name"),
        "season": season,
        "pos": pos,
        "feats": feats,
    }
    if finish is not None:
        rec["finish"] = finish
    if adp is not None:
        rec["adp"] = round(float(adp), 1)
        bucket = row.get("adp_bucket")
        rec["adp_bucket"] = str(bucket) if bucket else None
        if not rec["adp_bucket"]:
            from dashboard_services.historical.definitions import adp_overall_bucket
            rec["adp_bucket"] = adp_overall_bucket(adp)
        if not rec["adp_bucket"]:
            rec.pop("adp_bucket", None)
    return rec


def build_cohort_index(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Compact player-season index for request-time AND/OR cohort matching.

    Trajectory buckets use only consecutive seasons strictly before the
    outcome year. This is not a parquet scan at request time — cron rebuilds
    the index into the aggregates JSON.
    """
    from dashboard_services.historical.offense import (
        prior_offense_rank_for,
        team_offense_lookup_from_rows,
    )

    offense_ranks, offense_teams = team_offense_lookup_from_rows(rows)
    by_player: dict[str, list[dict]] = {}
    for row in rows:
        pid = _obs_pid(row)
        if not pid:
            continue
        by_player.setdefault(pid, []).append(dict(row))
    observations: list[dict] = []
    n_traj = 0
    for pid, career in by_player.items():
        ordered = sorted(career, key=lambda r: _optional_int(r.get("season")) or 0)
        for i, row in enumerate(ordered):
            prior = [
                r for r in ordered[:i]
                if (_optional_int(r.get("season")) or 0) < (_optional_int(row.get("season")) or 0)
            ]
            extra: dict[str, Any] = {}
            if len(prior) >= 2:
                extra.update(
                    trajectory_buckets(
                        prior[-2],
                        prior[-1],
                        position=row.get("position"),
                        current_season=row.get("season"),
                    )
                )
            season = _optional_int(row.get("season"))
            team = None
            if season is not None:
                team = (offense_teams.get(pid) or {}).get(str(season))
            if not team:
                from dashboard_services.historical.definitions import normalize_team_abbr

                team = normalize_team_abbr(row.get("team"))
            rank = prior_offense_rank_for(offense_ranks, team, season)
            if rank is not None:
                extra["prior_offense_rank"] = rank
            rec = _compact_observation(row, extra_feats=extra or None)
            if rec is None:
                continue
            if extra:
                n_traj += 1
            observations.append(rec)
    return {
        "kind": "player_season",
        "descriptive_only": True,
        "not_in_ranking": True,
        "not_in_pick_score": True,
        "n": len(observations),
        "n_with_trajectory": n_traj,
        "board_tiers": list(COMP_BOARD_TIERS),
        "trajectory": TRAJECTORY_CONTRACT,
        "observations": observations,
        "trajectory_rates": build_trajectory_rates(observations),
    }


def observations_of(aggregates: Mapping[str, Any]) -> list[dict]:
    index = aggregates.get("cohort_index") if isinstance(aggregates.get("cohort_index"), Mapping) else {}
    rows = index.get("observations") if isinstance(index, Mapping) else None
    if isinstance(rows, list):
        return rows
    return []


def _tier_hit(obs: Mapping[str, Any], tier: str) -> bool:
    cutoff = TIER_CUTOFFS.get(tier)
    if cutoff is None:
        return False
    finish = _optional_int(obs.get("finish"))
    if finish is None:
        return False
    return finish <= cutoff


def _baseline_rate(aggregates: Mapping[str, Any], pos: str, tier: str) -> Optional[float]:
    comps = ((aggregates.get("comps") or {}).get("by_position") or {}).get(pos) or {}
    block = (comps.get("baseline") or {}).get(tier) if isinstance(comps, Mapping) else None
    if isinstance(block, Mapping) and block.get("raw_rate") is not None:
        return _optional_float(block.get("raw_rate"))
    if tier == "top_12":
        age_block = (aggregates.get("age_curves") or {}).get(pos) or {}
        baseline = age_block.get("baseline") if isinstance(age_block.get("baseline"), Mapping) else {}
        if isinstance(baseline, Mapping):
            return _optional_float(baseline.get("raw_rate"))
    dedicated = ((aggregates.get("age_curves_by_tier") or {}).get(tier) or {}).get(pos) or {}
    base = dedicated.get("baseline") if isinstance(dedicated, Mapping) else None
    if isinstance(base, Mapping):
        return _optional_float(base.get("raw_rate"))
    return None


def _adp_bucket_rate(
    aggregates: Mapping[str, Any],
    pos: str,
    bucket: Any,
    *,
    tier: str = SIGNAL_BOARD_TIER,
) -> Optional[float]:
    if not bucket or tier != SIGNAL_BOARD_TIER:
        return None
    node = ((aggregates.get("adp") or {}).get("by_position") or {}).get(pos) or {}
    pair = (node.get("by_overall_bucket") or {}).get(str(bucket)) if isinstance(node, Mapping) else None
    cond = (pair.get("conditional") or {}) if isinstance(pair, Mapping) else {}
    return probability_from_rate(cond)


def _rate_block(
    matched: Sequence[Mapping[str, Any]],
    *,
    tier: str,
    baseline_rate: Optional[float],
) -> dict:
    n = len(matched)
    hits = sum(1 for obs in matched if _tier_hit(obs, tier))
    rec = make_rate(hits, n, prior_rate=baseline_rate)
    rec["kind"] = "player_season"
    bundle = edge_bundle(rec, baseline_rate)
    rec.update({
        "adjusted_rate": bundle.get("adjusted_rate"),
        "raw_edge": bundle.get("raw_edge"),
        "adjusted_edge": bundle.get("adjusted_edge"),
        "raw_edge_pts": bundle.get("raw_edge_pts"),
        "adjusted_edge_pts": bundle.get("adjusted_edge_pts"),
        "baseline_rate": bundle.get("baseline_rate"),
        "ci_low": bundle.get("ci_low"),
        "ci_high": bundle.get("ci_high"),
        "ci_low_pct": bundle.get("ci_low_pct"),
        "ci_high_pct": bundle.get("ci_high_pct"),
        "ranking_prior_n": EDGE_RANK_PRIOR_N,
        "display_pct": display_percent(rec.get("raw_rate")),
        "adjusted_pct": display_percent(bundle.get("adjusted_rate")),
        "baseline_pct": display_percent(baseline_rate),
        "confidence_short": CONFIDENCE_SHORT.get(str(rec.get("confidence") or ""), rec.get("confidence")),
    })
    # Display the *observed* combined rate (actual matched rows). Adjusted is
    # the shrinkage used for edges, not a substitute for n=84 → 31%.
    rec["display_pct"] = display_percent(rec.get("raw_rate"))
    return rec


def _market_adjusted(
    matched: Sequence[Mapping[str, Any]],
    *,
    pos: str,
    aggregates: Mapping[str, Any],
    observed_rate: Optional[float],
    tier: str,
) -> dict:
    """Observed hit rate vs average ADP-bucket expected P. top_12 only.

    Existing ``adp`` leaves are P(top-12 | overall bucket). Other finish
    lines would require a second market system — we omit them instead.
    """
    empty = {
        "observed_rate": _round_rate(observed_rate),
        "expected_market_rate": None,
        "market_adjusted_edge": None,
        "market_adjusted_edge_pts": None,
        "n_with_adp": 0,
        "coverage": None,
        "unknown_reason": None,
        "tier": tier,
    }
    if tier != SIGNAL_BOARD_TIER:
        empty["unknown_reason"] = "market_rates_are_top_12_only"
        return empty
    expected: list[float] = []
    for obs in matched:
        bucket = obs.get("adp_bucket")
        p = _adp_bucket_rate(aggregates, pos, bucket, tier=tier)
        if p is not None:
            expected.append(float(p))
    n_adp = len(expected)
    empty["n_with_adp"] = n_adp
    share = (n_adp / len(matched)) if matched else 0.0
    empty["coverage"] = _round_rate(share) if matched else None
    if n_adp < MARKET_ADP_MIN_N or share < MARKET_ADP_MIN_SHARE:
        empty["unknown_reason"] = "insufficient_historical_adp"
        return empty
    if observed_rate is None:
        empty["unknown_reason"] = "empty_cohort"
        return empty
    exp = sum(expected) / n_adp
    empty["expected_market_rate"] = _round_rate(exp)
    empty["expected_market_pct"] = display_percent(exp)
    empty["observed_pct"] = display_percent(observed_rate)
    delta = float(observed_rate) - exp
    empty["market_adjusted_edge"] = _round_rate(delta)
    empty["market_adjusted_edge_pts"] = _edge_pts(observed_rate, exp)
    empty["unknown_reason"] = None
    return empty


_COMPACT_CAPITAL_LABELS = {
    "round_1": "Round 1",
    "day_2": "Day 2",
    "day_3": "Day 3",
    "undrafted": "Undrafted",
}
_USAGE_TRAIT_UNITS = {
    "target_share": "targets",
    "snap_pct": "snaps",
    "touches": "touches",
    "carries": "carries",
    "receptions": "receptions",
    "targets": "targets",
    "games": "games",
    "pass_attempts": "pass attempts",
    "ryoe": "RYOE",
}
_EXAMPLE_TRAIT_ORDER = (
    "career_stage",
    "draft_capital",
    "age_bucket",
    "prior_finish",
    "prior_offense_rank",
    "projected_offense_rank",
    "target_share",
    "snap_pct",
    "adot",
    "ryoe",
    "target_share_change",
    "snap_pct_change",
    "workload_change",
)


def _human_feat_label(dim: str, value: Any) -> str:
    """Readable bucket copy for closest-example tags. Never leak underscored keys."""
    if value is None or value == "":
        return ""
    text = str(value)
    if dim == "draft_capital":
        compact = _COMPACT_CAPITAL_LABELS.get(text)
        if compact:
            return compact
    from dashboard_services.historical.board import format_comp_bucket_value

    label = format_comp_bucket_value(dim, value) or text
    if "_" in label:
        label = label.replace("_", " ").strip()
    return label


def _bare_trait_value(text: str, *prefixes: str) -> str:
    """Strip a leading NFL / Label: prefix so we can reattach a short label."""
    out = str(text or "").strip()
    changed = True
    while out and changed:
        changed = False
        low = out.lower()
        if low.startswith("nfl "):
            out = out[4:].strip()
            changed = True
            continue
        for prefix in prefixes:
            token = str(prefix or "").strip().rstrip(":").lower()
            if not token:
                continue
            if low.startswith(token + ":"):
                out = out.split(":", 1)[1].strip()
                changed = True
                break
            if low.startswith(token + " "):
                out = out[len(token) + 1:].strip()
                changed = True
                break
    return out


def _example_trait_phrase(
    dim: str,
    value: Any,
    feats: Optional[Mapping[str, Any]] = None,
) -> str:
    """One labeled tag: Exp: Year 4 · Draft: Round 1 · Age: 23-24 · Last Year: Top 5."""
    key = str(dim or "")
    if key == "nfl_draft_pick":
        from dashboard_services.historical.definitions import trends_round1_pick_range

        band = trends_round1_pick_range(value)
        if band:
            name = _bare_trait_value(band[1], "Draft")
            return f"Draft: {name}" if name else ""
        cap = (feats or {}).get("draft_capital")
        if cap:
            return _example_trait_phrase("draft_capital", cap, feats)
        return ""
    raw = _human_feat_label(key, value)
    if not raw:
        return ""
    if key == "career_stage":
        val = _bare_trait_value(raw, "Exp")
        low = val.lower()
        if low == "rookie":
            return "Exp: Rookie"
        if low.startswith("year "):
            return f"Exp: Year {val[5:].strip()}"
        return f"Exp: {val}"
    if key == "draft_capital":
        val = _bare_trait_value(raw, "Draft")
        return f"Draft: {val}" if val else ""
    if key in ("age_bucket", "age"):
        val = _bare_trait_value(raw, "Age")
        return f"Age: {val}" if val else ""
    if key == "prior_finish":
        val = _bare_trait_value(raw, "Last Year")
        return f"Last Year: {val}" if val else ""
    if key in ("prior_offense_rank", "prior_offense_rank_bucket", "offense"):
        from dashboard_services.historical.definitions import trends_offense_range

        band = trends_offense_range(value)
        name = band[1] if band else _bare_trait_value(raw, "Offense")
        if str(value) == "top_10":
            name = "Top 10"
        return f"Offense: {name} last year" if name else ""
    if key in ("projected_offense_rank", "projected_offense_rank_bucket", "projected_offense"):
        from dashboard_services.historical.definitions import trends_offense_range

        band = trends_offense_range(value)
        name = band[1] if band else _bare_trait_value(raw, "Offense")
        if str(value) == "top_10":
            name = "Top 10"
        return f"Offense: {name} projected" if name else ""
    if key == "prior_elite":
        return raw
    if key == "adot":
        from dashboard_services.historical.board import format_adot_bucket_label

        labeled = _bare_trait_value(format_adot_bucket_label(value) or raw, "aDOT", "Last Year")
        return f"aDOT: {labeled}" if labeled else ""
    unit = _USAGE_TRAIT_UNITS.get(key)
    if unit:
        val = _bare_trait_value(raw, unit, "Last Year")
        label = unit[0].upper() + unit[1:] if unit != "RYOE" else "RYOE"
        return f"{label}: {val}" if val else ""
    if key.endswith("_change"):
        metric = key[: -len("_change")].replace("_", " ")
        val = _bare_trait_value(raw, metric)
        return f"{metric[0].upper() + metric[1:]}: {val}" if val else ""
    return raw


def _example_traits(feats: Mapping[str, Any], filters: Sequence[Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()

    def add(dim: str, value: Any) -> None:
        phrase = _example_trait_phrase(dim, value, feats)
        if not phrase or phrase in seen:
            return
        seen.add(phrase)
        out.append(phrase)

    for spec in filters or ():
        if not isinstance(spec, Mapping):
            continue
        if not matches_trend_filter(feats, spec):
            continue
        field = str(spec.get("field") or spec.get("group") or "")
        value = feats.get(field) if field and feats.get(field) not in (None, "") else spec.get("eq")
        if field and value not in (None, ""):
            add(field, value)
            continue
        label = str(spec.get("label") or "").strip()
        if not label:
            continue
        group = str(spec.get("group") or field)
        phrase = _example_trait_phrase(group, label, feats) or label.replace("_", " ").strip()
        if phrase and phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
    if out:
        return out
    for key in _EXAMPLE_TRAIT_ORDER:
        val = feats.get(key)
        if val is None or val == "":
            continue
        add(key, val)
        if len(out) >= 4:
            break
    return out


def example_finish_hit(finish: Any) -> dict[str, Any]:
    """Best board finish line this example hit. Missing finish stays unknown."""
    rank = _optional_int(finish)
    hits = {tier: False for tier in COMP_BOARD_TIERS}
    if rank is None or rank < 1:
        return {"hit_tier": None, "hit_label": None, "hits": hits}
    for tier in COMP_BOARD_TIERS:
        cut = TIER_CUTOFFS.get(tier)
        if cut is not None and rank <= int(cut):
            hits[str(tier)] = True
    if hits.get("top_5"):
        tier = "top_5"
    elif hits.get("top_12"):
        tier = "top_12"
    elif hits.get("top_24"):
        tier = "top_24"
    else:
        tier = "miss"
    return {
        "hit_tier": tier,
        "hit_label": HIT_TIER_LABELS.get(tier, tier),
        "hits": hits,
    }


def closest_examples(
    matched: Sequence[Mapping[str, Any]],
    *,
    query_feats: Optional[Mapping[str, Any]] = None,
    query_age: Any = None,
    filters: Optional[Sequence[Mapping[str, Any]]] = None,
    exclude_pid: Any = None,
    limit: int = CLOSEST_EXAMPLE_LIMIT,
) -> list[dict]:
    """5-8 named player-seasons from the matched set. Not the full cohort."""
    skip = str(exclude_pid or "")
    qf = query_feats if isinstance(query_feats, Mapping) else {}
    q_age = _optional_float(query_age if query_age is not None else qf.get("age"))
    ranked: list[tuple] = []
    for obs in matched:
        pid = str(obs.get("pid") or "")
        if not pid or pid == skip:
            continue
        feats = obs.get("feats") if isinstance(obs.get("feats"), Mapping) else {}
        overlap = 0
        if qf:
            overlap = sum(
                1
                for key, val in qf.items()
                if key not in ("position", "age") and val is not None and feats.get(key) == val
            )
        c_age = _optional_float(feats.get("age"))
        if q_age is not None and c_age is not None:
            age_d = abs(q_age - c_age)
        else:
            age_d = 100.0
        finish = _optional_int(obs.get("finish")) or 999
        ranked.append((
            -overlap,
            0 if finish <= 12 else 1,
            age_d,
            finish,
            -(_optional_int(obs.get("season")) or 0),
            obs,
        ))
    ranked.sort(key=lambda t: t[:-1])
    out: list[dict] = []
    seen: set[str] = set()
    for item in ranked:
        obs = item[-1]
        pid = str(obs.get("pid") or "")
        if pid in seen:
            continue
        seen.add(pid)
        feats = obs.get("feats") if isinstance(obs.get("feats"), Mapping) else {}
        rec = {
            "sleeper_id": pid,
            "name": obs.get("name") or pid,
            "season": obs.get("season"),
            "positional_finish": obs.get("finish"),
            "adp": obs.get("adp"),
            "adp_bucket": obs.get("adp_bucket"),
            "traits": _example_traits(feats, filters or ()),
        }
        rec.update(example_finish_hit(obs.get("finish")))
        out.append(rec)
        if len(out) >= limit:
            break
    return out


def examples_summary(examples: Sequence[Mapping[str, Any]]) -> dict:
    n = len(examples)
    top5 = sum(1 for ex in examples if (_optional_int(ex.get("positional_finish")) or 999) <= TIER_CUTOFFS["top_5"])
    top12 = sum(1 for ex in examples if (_optional_int(ex.get("positional_finish")) or 999) <= TIER_CUTOFFS["top_12"])
    top24 = sum(1 for ex in examples if (_optional_int(ex.get("positional_finish")) or 999) <= TIER_CUTOFFS["top_24"])
    return {
        "n": n,
        "top_5": top5,
        "top_12": top12,
        "top_24": top24,
        "label": (
            f"{top5}/{n} Top-5 · {top12}/{n} Top-12 · {top24}/{n} Top-24"
            if n else None
        ),
    }


def _selected_filters_payload(filters: Sequence[Mapping[str, Any]], pos: str) -> list[dict]:
    out = []
    for spec in filters or []:
        if not isinstance(spec, Mapping):
            continue
        rec = {
            "group": spec.get("group") or spec.get("field"),
            "field": spec.get("field"),
            "label": spec.get("label") or spec.get("eq") or spec.get("field"),
        }
        for key in ("eq", "in", "gte", "lte", "between", "null_as"):
            if key in spec:
                rec[key] = spec[key]
        out.append(rec)
    return out


def evaluate_cohort(
    aggregates: Mapping[str, Any],
    *,
    position: Any,
    filters: Sequence[Mapping[str, Any]],
    tier: str = SIGNAL_BOARD_TIER,
    data_version: Any = None,
) -> dict:
    """Combined historical hit rates for the exact selected filter intersection.

    Counts matching observation rows. Does not multiply single-bucket rates.
    """
    pos = str(position or "").upper()
    wanted_tier = str(tier or SIGNAL_BOARD_TIER)
    if wanted_tier not in COMP_BOARD_TIERS:
        wanted_tier = SIGNAL_BOARD_TIER
    selected = [dict(f) for f in (filters or []) if isinstance(f, Mapping)]
    cache_key = (data_version, pos, wanted_tier, canonical_filter_key(selected))
    global _CACHE_VERSION
    if data_version != _CACHE_VERSION:
        _COHORT_CACHE.clear()
        _CACHE_VERSION = data_version
    cached = _COHORT_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    empty = {
        "available": False,
        "descriptive_only": True,
        "not_in_ranking": True,
        "not_in_pick_score": True,
        "kind": "player_season",
        "position": pos if pos in SKILL_POSITIONS else None,
        "tier": wanted_tier,
        "filters": _selected_filters_payload(selected, pos),
        "sample_size": 0,
        "n_players": 0,
        "unknown_reason": None,
    }
    if pos not in SKILL_POSITIONS:
        empty["unknown_reason"] = "no_position"
        return empty
    if not selected:
        empty["unknown_reason"] = "no_filters"
        empty["available"] = True
        empty["headline"] = "Tap historical buckets to build a profile."
        _store_cache(cache_key, empty)
        return empty
    pool = observations_of(aggregates)
    if not pool:
        empty["unknown_reason"] = "cohort_index_missing"
        _store_cache(cache_key, empty)
        return empty

    matched: list[dict] = []
    for obs in pool:
        if str(obs.get("pos") or "").upper() != pos:
            continue
        feats = obs.get("feats") if isinstance(obs.get("feats"), Mapping) else {}
        if not matches_filter_groups(feats, selected):
            continue
        matched.append(obs)

    n = len(matched)
    players = {str(obs.get("pid") or "") for obs in matched if obs.get("pid")}
    baseline = _baseline_rate(aggregates, pos, wanted_tier)
    rates = {
        t: _rate_block(matched, tier=t, baseline_rate=_baseline_rate(aggregates, pos, t))
        for t in COMP_BOARD_TIERS
    }
    lead = rates.get(wanted_tier) or {}
    market = _market_adjusted(
        matched,
        pos=pos,
        aggregates=aggregates,
        observed_rate=lead.get("raw_rate"),
        tier=wanted_tier,
    )
    examples = closest_examples(matched, filters=selected, limit=CLOSEST_EXAMPLE_LIMIT)
    payload = {
        "available": True,
        "descriptive_only": True,
        "not_in_ranking": True,
        "not_in_pick_score": True,
        "kind": "player_season",
        "position": pos,
        "tier": wanted_tier,
        "filters": _selected_filters_payload(selected, pos),
        "sample_size": n,
        "n_players": len(players),
        "successes": lead.get("successes"),
        "raw_rate": lead.get("raw_rate"),
        "adjusted_rate": lead.get("adjusted_rate"),
        "baseline_rate": lead.get("baseline_rate"),
        "raw_edge": lead.get("raw_edge"),
        "adjusted_edge": lead.get("adjusted_edge"),
        "raw_edge_pts": lead.get("raw_edge_pts"),
        "adjusted_edge_pts": lead.get("adjusted_edge_pts"),
        "display_pct": lead.get("display_pct"),
        "adjusted_pct": lead.get("adjusted_pct"),
        "baseline_pct": lead.get("baseline_pct"),
        "ci_low": lead.get("ci_low"),
        "ci_high": lead.get("ci_high"),
        "ci_low_pct": lead.get("ci_low_pct"),
        "ci_high_pct": lead.get("ci_high_pct"),
        "confidence": lead.get("confidence"),
        "confidence_short": lead.get("confidence_short"),
        "rates": rates,
        "market": market,
        "examples": examples,
        "examples_summary": examples_summary(examples),
        "finish_tier_copy": FINISH_TIER_COPY.get(pos),
        "bayes_prior_n": DEFAULT_BAYES_PRIOR_N,
        "ranking_prior_n": EDGE_RANK_PRIOR_N,
        "unknown_reason": None if n else "empty_cohort",
    }
    _store_cache(cache_key, payload)
    return payload


def _store_cache(key: tuple, payload: Mapping[str, Any]) -> None:
    if len(_COHORT_CACHE) >= _CACHE_MAX:
        # Drop an arbitrary old entry. Cohort keys are small; this is a cap.
        _COHORT_CACHE.pop(next(iter(_COHORT_CACHE)), None)
    _COHORT_CACHE[key] = dict(payload)


def reset_cohort_cache() -> None:
    _COHORT_CACHE.clear()


def closest_examples_for_query(
    query: Mapping[str, Any],
    aggregates: Mapping[str, Any],
    *,
    limit: int = CLOSEST_EXAMPLE_LIMIT,
    exclude_pid: Any = None,
) -> list[dict]:
    """Named seasons closest to one live player's preseason profile."""
    feats = extract_trend_features(query)
    pos = str(feats.get("position") or query.get("position") or "").upper()
    if pos not in SKILL_POSITIONS:
        return []
    skip = str(exclude_pid if exclude_pid is not None else query.get("sleeper_id") or query.get("id") or "")
    pool = [
        obs for obs in observations_of(aggregates)
        if str(obs.get("pos") or "").upper() == pos
        and str(obs.get("pid") or "") != skip
    ]
    if not pool:
        return []
    # Require position; prefer rows that share the query's present buckets.
    return closest_examples(
        pool,
        query_feats=feats,
        query_age=query.get("age") or feats.get("age"),
        exclude_pid=skip,
        limit=limit,
    )


def preseason_trajectory_fields(
    career: Sequence[Mapping[str, Any]],
    *,
    upcoming_season: Optional[int] = None,
) -> dict[str, str]:
    """Stamp YoY buckets onto a live preseason profile. Last two observed seasons only."""
    ordered = sorted(
        (r for r in career if isinstance(r, Mapping)),
        key=lambda r: _optional_int(r.get("season")) or 0,
    )
    if len(ordered) < 2:
        return {}
    last = ordered[-1]
    prev = ordered[-2]
    current = upcoming_season
    if current is None:
        last_s = _optional_int(last.get("season"))
        current = (last_s + 1) if last_s is not None else None
    return trajectory_buckets(
        prev,
        last,
        position=last.get("position"),
        current_season=current,
    )


def build_trajectory_rates(observations: Sequence[Mapping[str, Any]]) -> dict:
    """Player-season hit rates for leakage-safe YoY buckets. Empty cells omitted."""
    metrics = (
        (
            "target_share_change",
            (TRAJECTORY_TARGET_SHARE_UP, TRAJECTORY_TARGET_SHARE_DOWN),
        ),
        (
            "snap_pct_change",
            (TRAJECTORY_SNAP_UP, TRAJECTORY_SNAP_DOWN),
        ),
        (
            "workload_change",
            (TRAJECTORY_WORKLOAD_UP, TRAJECTORY_WORKLOAD_DOWN),
        ),
    )
    out: dict[str, dict] = {}
    for pos in SKILL_POSITIONS:
        pos_obs = [o for o in observations if str(o.get("pos") or "").upper() == pos]
        if not pos_obs:
            continue
        n_all = len(pos_obs)
        hits_all = sum(1 for o in pos_obs if _tier_hit(o, "top_12"))
        baseline = make_rate(hits_all, n_all)
        pos_block: dict[str, dict] = {}
        for metric, labels in metrics:
            known = [o for o in pos_obs if (o.get("feats") or {}).get(metric)]
            if not known:
                continue
            by_bucket = {}
            for label in labels:
                cell = [
                    o for o in known
                    if (o.get("feats") or {}).get(metric) == label
                ]
                if not cell:
                    continue
                rates = {
                    tier: make_rate(
                        sum(1 for o in cell if _tier_hit(o, tier)),
                        len(cell),
                        prior_rate=_optional_float(baseline.get("raw_rate")),
                    )
                    for tier in COMP_BOARD_TIERS
                }
                by_bucket[label] = {
                    "n": len(cell),
                    "top_12": rates["top_12"],
                    "by_tier": rates,
                }
            if by_bucket:
                pos_block[metric] = {
                    "n_known": len(known),
                    "n_missing_excluded": n_all - len(known),
                    "baseline": baseline,
                    "by_bucket": by_bucket,
                }
        if pos_block:
            out[pos] = pos_block
    return out
