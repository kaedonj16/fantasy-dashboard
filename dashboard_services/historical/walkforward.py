"""Walk-forward comparison of History P vs Market P vs position baseline.

Reuses Phase 4 comps and Phase 5–6 ADP lookups. Train on seasons **< S**,
test on season **S**. Ground truth is warehouse positional finishes — not
the breakout engine's usage-points proxy, and not a second BreakoutEngine.

Live board comps stay pooled historical. This module is the honesty check
for whether Hist may enter Pick Score. Missing probabilities are skipped,
never faked as 0.

This module must stay dependency-free (no pandas, Flask, sklearn,
nfl_data_py, or I/O).
"""
from __future__ import annotations

import bisect
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from dashboard_services.historical.adp import build_adp_hit_rates
from dashboard_services.historical.career_profiles import (
    is_engine_breakout,
    is_first_time_elite,
    is_league_winner,
    is_league_winner_smash,
)
from dashboard_services.historical.comps import (
    build_comp_aggregates,
    lookup_board_probabilities,
)
from dashboard_services.historical.definitions import (
    PICK_SCORE_MIN_AUC_MARGIN_VS_MARKET,
    PICK_SCORE_MIN_HIST_AUC,
    PICK_SCORE_MIN_SCORED_PER_SEASON,
    PICK_SCORE_MIN_SCORED_TOTAL,
    PICK_SCORE_MIN_TEST_SEASONS,
    PICK_SCORE_PRIMARY_LABEL,
    RELIABLE_SEASON_FLOOR,
    WALKFORWARD_TEST_SEASONS,
    _optional_int,
)
from dashboard_services.historical.finish_rates import (
    filter_era,
    is_tier_hit,
    positional_finish,
    season_bounds,
)
from dashboard_services.historical.signals import (
    lookup_market_probability,
    probability_from_rate,
)

LabelFn = Callable[[Mapping[str, Any], str], bool]


def roc_auc(
    scores: Sequence[Optional[float]],
    labels: Sequence[Any],
) -> Optional[float]:
    """Mann–Whitney AUC. Ties count 0.5. Missing scores are skipped.

    Returns None when there are no positives or no negatives among scored
    rows — never a fake 0.5.
    """
    pos: list[float] = []
    neg: list[float] = []
    for score, label in zip(scores, labels):
        if score is None:
            continue
        if label:
            pos.append(float(score))
        else:
            neg.append(float(score))
    if not pos or not neg:
        return None
    neg_sorted = sorted(neg)
    wins = 0.0
    for value in pos:
        lo = bisect.bisect_left(neg_sorted, value)
        hi = bisect.bisect_right(neg_sorted, value)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(pos) * len(neg))


def brier_score(
    scores: Sequence[Optional[float]],
    labels: Sequence[Any],
) -> Optional[float]:
    """Mean squared error of probabilities. Missing scores are skipped."""
    total = 0.0
    n = 0
    for score, label in zip(scores, labels):
        if score is None:
            continue
        y = 1.0 if label else 0.0
        err = float(score) - y
        total += err * err
        n += 1
    if n == 0:
        return None
    return total / n


def split_train_test(
    rows: Iterable[Mapping[str, Any]],
    test_season: int,
) -> tuple[list[dict], list[dict]]:
    """Train = seasons strictly before ``test_season``. Test = that season."""
    train: list[dict] = []
    test: list[dict] = []
    for row in rows:
        season = _optional_int(row.get("season"))
        if season is None:
            continue
        if season < test_season:
            train.append(dict(row))
        elif season == test_season:
            test.append(dict(row))
    return train, test


def _y_top_12(row: Mapping[str, Any], scoring: str) -> bool:
    return is_tier_hit(row, tier="top_12", scoring=scoring)


def _y_league_winner(row: Mapping[str, Any], scoring: str) -> bool:
    return is_league_winner(positional_finish(row, scoring))


def _y_league_winner_smash(row: Mapping[str, Any], scoring: str) -> bool:
    return is_league_winner_smash(
        row.get("previous_season_finish"),
        positional_finish(row, scoring),
    )


def _y_engine_breakout(row: Mapping[str, Any], scoring: str) -> bool:
    return is_engine_breakout(
        row.get("previous_season_finish"),
        positional_finish(row, scoring),
    )


def _y_first_time_elite(row: Mapping[str, Any], scoring: str) -> bool:
    prev = row.get("previously_top12")
    if prev is None:
        count = _optional_int(row.get("prior_top12_count"))
        if count is not None:
            prev = count > 0
        elif row.get("first_time_top12_candidate") is not None:
            prev = not bool(row.get("first_time_top12_candidate"))
        else:
            prev = False
    return is_first_time_elite(prev, positional_finish(row, scoring))


# Each label is scored with History/Market P at ``tier``. League-winner uses
# P(top-5); the board-facing Hist column is P(top-12).
WALKFORWARD_LABELS: tuple[dict[str, Any], ...] = (
    {"id": "top_12", "tier": "top_12", "label": _y_top_12},
    {"id": "league_winner", "tier": "top_5", "label": _y_league_winner},
    {
        "id": "league_winner_smash",
        "tier": "top_5",
        "label": _y_league_winner_smash,
    },
    {"id": "engine_breakout", "tier": "top_12", "label": _y_engine_breakout},
    {"id": "first_time_elite", "tier": "top_12", "label": _y_first_time_elite},
)


def _round_metric(value: Optional[float], ndigits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), ndigits)


def _metrics(
    y: Sequence[bool],
    hist: Sequence[Optional[float]],
    market: Sequence[Optional[float]],
    baseline: Sequence[Optional[float]],
) -> dict:
    n = len(y)
    n_pos = sum(1 for flag in y if flag)
    n_hist = sum(1 for p in hist if p is not None)
    n_market = sum(1 for p in market if p is not None)
    n_base = sum(1 for p in baseline if p is not None)
    n_both = sum(
        1 for h, m in zip(hist, market) if h is not None and m is not None
    )
    residual: list[Optional[float]] = []
    for h, m in zip(hist, market):
        if h is None or m is None:
            residual.append(None)
        else:
            residual.append(float(h) - float(m))
    return {
        "n_test": n,
        "n_pos": n_pos,
        "n_hist": n_hist,
        "n_market": n_market,
        "n_baseline": n_base,
        "n_both": n_both,
        "hist_auc": _round_metric(roc_auc(hist, y)),
        "market_auc": _round_metric(roc_auc(market, y)),
        "baseline_auc": _round_metric(roc_auc(baseline, y)),
        "hist_brier": _round_metric(brier_score(hist, y)),
        "market_brier": _round_metric(brier_score(market, y)),
        "baseline_brier": _round_metric(brier_score(baseline, y)),
        "residual_auc": _round_metric(roc_auc(residual, y)),
    }


def _p_from_comps(looked: Mapping[str, Any], tier: str) -> Optional[float]:
    rate = (looked.get("rates") or {}).get(tier) or {}
    return probability_from_rate(rate)


def _baseline_from_comps(
    comps: Mapping[str, Any],
    position: Any,
    tier: str,
) -> Optional[float]:
    pos = str(position or "").upper()
    by_pos = (comps.get("by_position") or {}).get(pos) or {}
    rate = (by_pos.get("baseline") or {}).get(tier) or {}
    return probability_from_rate(rate)


def score_fold(
    test_rows: Sequence[Mapping[str, Any]],
    comps: Mapping[str, Any],
    adp_by_tier: Mapping[str, Mapping[str, Any]],
    *,
    scoring: str = "ppr",
) -> dict[str, dict]:
    """Score one test season. Does not rebuild comps/ADP (caller does)."""
    prepared: list[dict] = []
    for row in test_rows:
        looked = lookup_board_probabilities(row, comps)
        hist = {
            "top_5": _p_from_comps(looked, "top_5"),
            "top_12": _p_from_comps(looked, "top_12"),
        }
        market = {
            tier: lookup_market_probability(
                row, {"adp": adp_by_tier[tier]}, tier=tier
            ).get("p_top_12")
            for tier in ("top_5", "top_12")
            if tier in adp_by_tier
        }
        pos = row.get("position")
        baseline = {
            "top_5": _baseline_from_comps(comps, pos, "top_5"),
            "top_12": _baseline_from_comps(comps, pos, "top_12"),
        }
        prepared.append(
            {
                "row": row,
                "hist": hist,
                "market": market,
                "baseline": baseline,
            }
        )

    out: dict[str, dict] = {}
    for spec in WALKFORWARD_LABELS:
        tier = spec["tier"]
        fn: LabelFn = spec["label"]
        y = [fn(item["row"], scoring) for item in prepared]
        hist = [item["hist"].get(tier) for item in prepared]
        market = [item["market"].get(tier) for item in prepared]
        baseline = [item["baseline"].get(tier) for item in prepared]
        block = _metrics(y, hist, market, baseline)
        block["tier"] = tier
        out[spec["id"]] = block
    return out


def pick_score_gate_constants() -> dict:
    return {
        "primary_label": PICK_SCORE_PRIMARY_LABEL,
        "min_test_seasons": PICK_SCORE_MIN_TEST_SEASONS,
        "min_scored_per_season": PICK_SCORE_MIN_SCORED_PER_SEASON,
        "min_scored_total": PICK_SCORE_MIN_SCORED_TOTAL,
        "min_hist_auc": PICK_SCORE_MIN_HIST_AUC,
        "min_auc_margin_vs_market": PICK_SCORE_MIN_AUC_MARGIN_VS_MARKET,
    }


def evaluate_pick_score_gate(
    folds: Sequence[Mapping[str, Any]],
    *,
    primary_label: str = PICK_SCORE_PRIMARY_LABEL,
) -> dict:
    """Return whether Hist may enter Pick Score. Conservative: beat market."""
    gate = pick_score_gate_constants()
    qualifying: list[int] = []
    n_hist_total = 0
    n_both_total = 0
    for fold in folds:
        season = _optional_int(fold.get("test_season"))
        block = (fold.get("by_label") or {}).get(primary_label) or {}
        n_hist = int(block.get("n_hist") or 0)
        n_both = int(block.get("n_both") or 0)
        n_hist_total += n_hist
        n_both_total += n_both
        hist_auc = block.get("hist_auc")
        market_auc = block.get("market_auc")
        if n_both < PICK_SCORE_MIN_SCORED_PER_SEASON:
            continue
        if hist_auc is None or market_auc is None:
            continue
        if float(hist_auc) < PICK_SCORE_MIN_HIST_AUC:
            continue
        if float(hist_auc) < float(market_auc) + PICK_SCORE_MIN_AUC_MARGIN_VS_MARKET:
            continue
        if season is not None:
            qualifying.append(season)
    validated = (
        len(qualifying) >= PICK_SCORE_MIN_TEST_SEASONS
        and n_hist_total >= PICK_SCORE_MIN_SCORED_TOTAL
    )
    if validated:
        reason = (
            f"history P beat market P on {primary_label} in "
            f"{len(qualifying)} test seasons {qualifying}"
        )
    else:
        reason = (
            f"gate failed: {len(qualifying)}/{PICK_SCORE_MIN_TEST_SEASONS} "
            f"test seasons beat market AUC by "
            f"{PICK_SCORE_MIN_AUC_MARGIN_VS_MARKET} with hist_auc>="
            f"{PICK_SCORE_MIN_HIST_AUC}; n_hist={n_hist_total} "
            f"(need {PICK_SCORE_MIN_SCORED_TOTAL}). Pick Score unchanged."
        )
    return {
        "validated": validated,
        "reason": reason,
        "primary_label": primary_label,
        "qualifying_seasons": qualifying,
        "n_hist_total": n_hist_total,
        "n_both_total": n_both_total,
        "in_live_ranking": False,
        "gate": gate,
    }


def _pool_folds(folds: Sequence[Mapping[str, Any]]) -> dict[str, dict]:
    """Sum n_* across folds; leave AUC/Brier as None (not a pooled ranking).

    Per-season metrics are the walk-forward result. A concatenated AUC would
    mix incomparable season calibrations.
    """
    ids = [spec["id"] for spec in WALKFORWARD_LABELS]
    out: dict[str, dict] = {}
    for label_id in ids:
        n_test = n_pos = n_hist = n_market = n_base = n_both = 0
        seasons_with_auc = 0
        hist_aucs: list[float] = []
        market_aucs: list[float] = []
        for fold in folds:
            block = (fold.get("by_label") or {}).get(label_id) or {}
            n_test += int(block.get("n_test") or 0)
            n_pos += int(block.get("n_pos") or 0)
            n_hist += int(block.get("n_hist") or 0)
            n_market += int(block.get("n_market") or 0)
            n_base += int(block.get("n_baseline") or 0)
            n_both += int(block.get("n_both") or 0)
            if block.get("hist_auc") is not None:
                hist_aucs.append(float(block["hist_auc"]))
            if block.get("market_auc") is not None:
                market_aucs.append(float(block["market_auc"]))
            if block.get("hist_auc") is not None or block.get("market_auc") is not None:
                seasons_with_auc += 1
        out[label_id] = {
            "n_test": n_test,
            "n_pos": n_pos,
            "n_hist": n_hist,
            "n_market": n_market,
            "n_baseline": n_base,
            "n_both": n_both,
            "n_seasons_with_auc": seasons_with_auc,
            "mean_hist_auc": _round_metric(
                (sum(hist_aucs) / len(hist_aucs)) if hist_aucs else None
            ),
            "mean_market_auc": _round_metric(
                (sum(market_aucs) / len(market_aucs)) if market_aucs else None
            ),
            "pooled_auc_not_computed": True,
        }
    return out


def walkforward_contract() -> dict:
    return {
        "method": "walk_forward",
        "train": "seasons < S",
        "test": "season == S",
        "outcomes": "warehouse positional finishes",
        "not_usage_proxy": True,
        "not_a_second_engine": True,
        "live_comps_stay_pooled": True,
        "missing_p": "skipped, never 0",
        "request_path": "precomputed JSON; no parquet scan, no new Postgres table",
        "pick_score_in_live_ranking": False,
    }


def run_walk_forward(
    rows: Sequence[Mapping[str, Any]],
    *,
    scoring: str = "ppr",
    test_seasons: Optional[Sequence[int]] = None,
    season_from: int = RELIABLE_SEASON_FLOOR,
    season_to: Optional[int] = None,
) -> dict:
    """Rebuild comps + ADP on seasons < S; score season S. No parquet I/O."""
    era = filter_era(rows, season_from, season_to)
    seasons = list(test_seasons) if test_seasons is not None else list(
        WALKFORWARD_TEST_SEASONS
    )
    folds: list[dict] = []
    for test_season in seasons:
        train, test = split_train_test(era, test_season)
        if not train or not test:
            continue
        comps = build_comp_aggregates(
            train, scoring=scoring, include_named=False
        )
        adp_by_tier = {
            "top_12": build_adp_hit_rates(train, scoring=scoring, tier="top_12"),
            "top_5": build_adp_hit_rates(train, scoring=scoring, tier="top_5"),
        }
        by_label = score_fold(test, comps, adp_by_tier, scoring=scoring)
        folds.append({
            "test_season": test_season,
            "n_train": len(train),
            "n_test": len(test),
            "train_season_range": season_bounds(train),
            "by_label": by_label,
        })
    pick_score = evaluate_pick_score_gate(folds)
    return {
        **walkforward_contract(),
        "scoring": scoring,
        "test_seasons_requested": seasons,
        "n_folds": len(folds),
        "folds": folds,
        "pooled": _pool_folds(folds),
        "pick_score": pick_score,
        "descriptive_only": not pick_score["validated"],
    }
