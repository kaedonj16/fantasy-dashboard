"""
Train the ML prospect model on historical draft classes.

Uses the backtest infrastructure to fetch college stats (CFBD),
combine data, and actual NFL outcomes for each draft class.  Trains
one XGBoost model per position (WR, RB, QB, TE) and saves to
models/ml_prospect_models.pkl.

Usage
-----
    # Train on default years (2016-2023 — enough NFL data for all classes)
    python scripts/train_ml_model.py

    # Train on specific years
    python scripts/train_ml_model.py --years 2018 2019 2020 2021 2022 2023

    # Show feature importance after training
    python scripts/train_ml_model.py --show-importance

    # Evaluate with leave-one-year-out cross-validation
    python scripts/train_ml_model.py --cv
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.metrics import mean_absolute_error

# Backtest data-loading utilities (reuse — no need to rewrite)
from scripts.backtest_prospect_model import (
    _load_draft_class,
    _load_combine_athleticism,
    _load_cfbd_college_stats,
    _build_prospect_dicts,
    _build_nfl_ppr_per_player,
    _run_draft_class_backtest,
)
from data_building.rookie_pipeline.ml_model import (
    MLProspectScorer,
    POSITIONS,
    _MODEL_PATH,
    extract_features,
)

# Default training years: use classes with at least 2 complete NFL seasons
DEFAULT_TRAIN_YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]


def build_training_rows(draft_years: List[int]) -> List[Dict[str, Any]]:
    """
    For each draft year, load the full data (CFBD + combine + NFL outcomes)
    and return a list of training rows — one per draftee with NFL data.

    Each row contains:
        position, ppr_avg, seasons_avail,
        _prospect   : full prospect dict (for ML feature extraction)
        _consensus  : consensus/pick dict
    """
    all_rows: List[Dict[str, Any]] = []
    current_year = 2026

    for dy in draft_years:
        print(f"\n[train]  Loading {dy} draft class…")

        draft_class = _load_draft_class(dy)
        if not draft_class:
            continue
        print(f"[train]    {len(draft_class)} skill-position draftees")

        athleticism  = _load_combine_athleticism(dy)
        college_stats = _load_cfbd_college_stats(dy, draft_class)
        prospects, consensus_map = _build_prospect_dicts(
            draft_class, athleticism, dy, college_stats
        )

        n_cfbd = sum(1 for p in prospects if p.get("seasons"))
        print(f"[train]    {n_cfbd}/{len(prospects)} have CFBD stats")

        gsis_ids = [p["gsis_id"] for p in draft_class if p.get("gsis_id")]
        nfl_lookback = min(4, current_year - dy)
        nfl_ppr = _build_nfl_ppr_per_player(gsis_ids, dy, nfl_lookback)

        p_by_id = {p["player_id"]: p for p in prospects}
        dc_map  = {
            p["gsis_id"] or p["name"].lower().replace(" ", "-"): p
            for p in draft_class
        }

        included = 0
        for p in prospects:
            pid  = p["player_id"]
            dc   = dc_map.get(pid, {})
            gid  = dc.get("gsis_id", "")
            ppr  = nfl_ppr.get(gid, {})
            ppr_cum = ppr.get("ppr_cum", 0.0)
            sa   = ppr.get("seasons_available", 0)

            if ppr_cum <= 0:
                continue  # no NFL data → can't use as training target

            ppr_avg = ppr_cum / max(sa, 1)
            cons    = consensus_map.get(pid, {})

            all_rows.append({
                "draft_year":   dy,
                "name":         p.get("name", pid),
                "position":     p.get("position", "WR"),
                "ppr_avg":      ppr_avg,
                "ppr_cum":      ppr_cum,
                "seasons_avail":sa,
                "_prospect":    p,
                "_consensus":   cons,
            })
            included += 1

        print(f"[train]    {included}/{len(prospects)} included in training set")

    print(f"\n[train]  Total training rows: {len(all_rows)}")
    for pos in POSITIONS:
        n = sum(1 for r in all_rows if r["position"] == pos)
        print(f"[train]    {pos}: {n}")
    return all_rows


def evaluate_cv(training_rows: List[Dict[str, Any]]) -> None:
    """
    Leave-one-year-out cross-validation.
    Trains on all years except one, evaluates on the held-out year.
    Reports MAE and Pearson-r for each fold.
    """
    by_year: Dict[int, List] = {}
    for r in training_rows:
        by_year.setdefault(r["draft_year"], []).append(r)

    years = sorted(by_year.keys())
    print(f"\n{'='*70}")
    print("  LEAVE-ONE-YEAR-OUT CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"  {'Year':<6} {'Pos':<5} {'N':>4}  {'MAE':>7}  {'Pearson-r':>10}")
    print(f"  {'-'*6} {'-'*5} {'-'*4}  {'-'*7}  {'-'*10}")

    for held_year in years:
        train_rows = [r for r in training_rows if r["draft_year"] != held_year]
        test_rows  = [r for r in training_rows if r["draft_year"] == held_year]
        if len(test_rows) < 5:
            continue

        scorer = MLProspectScorer()
        scorer.fit(train_rows, verbose=False)

        for pos in POSITIONS:
            test_pos = [r for r in test_rows if r["position"] == pos]
            if len(test_pos) < 3:
                continue

            preds, actuals = [], []
            for r in test_pos:
                model = scorer.models[pos]
                if not model.trained:
                    continue
                feats = extract_features(r["_prospect"], r["_consensus"])
                raw = float(model.model.predict(
                    model.imputer.transform(feats)
                )[0])
                preds.append(raw)
                actuals.append(r["ppr_avg"])

            if len(preds) < 3:
                continue

            mae = mean_absolute_error(actuals, preds)
            r_val = _pearson(preds, actuals)
            r_str = f"{r_val:+.3f}" if not math.isnan(r_val) else "   n/a"
            print(f"  {held_year:<6} {pos:<5} {len(preds):>4}  {mae:>7.1f}  {r_str:>10}")


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs) / n)
    sy = math.sqrt(sum((y - my) ** 2 for y in ys) / n)
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy * n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ML prospect model")
    parser.add_argument(
        "--years", nargs="+", type=int, default=DEFAULT_TRAIN_YEARS,
        help=f"Draft years to train on (default: {DEFAULT_TRAIN_YEARS})",
    )
    parser.add_argument(
        "--show-importance", action="store_true",
        help="Print full feature importance table after training",
    )
    parser.add_argument(
        "--cv", action="store_true",
        help="Run leave-one-year-out cross-validation",
    )
    parser.add_argument(
        "--output", default=_MODEL_PATH,
        help=f"Where to save the trained model (default: {_MODEL_PATH})",
    )
    args = parser.parse_args()

    print(f"Training ML prospect model on years: {args.years}")
    training_rows = build_training_rows(args.years)

    if not training_rows:
        print("[train] No training data found. Check CFBD_API_KEY and network.")
        sys.exit(1)

    if args.cv:
        evaluate_cv(training_rows)

    print("\n[train] Fitting final models on all data…")
    scorer = MLProspectScorer()
    scorer.fit(training_rows, verbose=True)

    if args.show_importance:
        for pos in POSITIONS:
            model = scorer.models[pos]
            if not model.trained:
                continue
            print(f"\n  Feature importance — {pos}:")
            for name, imp in model.feature_importance():
                bar = "█" * int(imp * 200)
                print(f"    {name:<22} {imp:.4f}  {bar}")

    scorer.save(args.output)
    print(f"\n[train] Done. Run 'python scripts/backtest_prospect_model.py' to verify.")


if __name__ == "__main__":
    main()
