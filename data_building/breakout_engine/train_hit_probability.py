"""Fit the breakout hit-probability model from historical outcomes.

Replaces the hand-smoothed empirical curve in multitask_predictions.calculate_
hit_probability with a per-position logistic regression on the component scores,
fit against actual top-N finishes (top-6 QB/TE, top-12 RB/WR — the same "hit"
definition the backtest uses).

Training data
-------------
For each source season S:
  - features  = component scores for each candidate (from the DB
                breakout_opportunity_scores table, or a --scores-json dir written
                by build_historical_scores --output-json)
  - label     = 1 if that player finished top-N at his position in season S+1
                (from cache/player_history/usage_rows_{S+1}.json), else 0

The fitted per-position coefficients (on z-scored features) are written to
hit_probability_model.json, which calculate_hit_probability loads at inference —
no sklearn needed at runtime. Absent that file, inference falls back to the curve,
so this changes nothing until you review the metrics and commit the JSON.

Usage
-----
    # Against the DB (Render shell, DATABASE_URL set):
    python -m data_building.breakout_engine.train_hit_probability \
        --seasons 2021 2022 2023

    # Against pre-exported score JSONs (offline):
    python -m data_building.breakout_engine.train_hit_probability \
        --seasons 2021 2022 2023 --scores-json cache/breakout_scores
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
load_dotenv()

# breakout imports pull in openai transitively; mock so this stays standalone.
from unittest.mock import MagicMock as _MagicMock
for _mod in ("openai", "openai.types", "openai.types.chat"):
    sys.modules.setdefault(_mod, _MagicMock())

from data_building.breakout_engine.backtest_multitask import (  # noqa: E402
    load_actual_outcomes, load_position_map, get_top12_pids,
    load_breakout_scores_from_db, load_breakout_scores_from_json,
)
from data_building.breakout_engine import multitask_predictions as MP  # noqa: E402

MODEL_PATH = Path(__file__).resolve().parent / "hit_probability_model.json"

# Score-row key -> model feature name. Mirrors what calculate_hit_probability reads.
_FEATURE_KEYS = {
    "breakout_score":        ("breakout_opportunity_score", "breakout_score"),
    "readiness_score":       ("player_readiness_score", "readiness_score"),
    "confidence_score":      ("confidence_score",),
    "opportunity_score":     ("opportunity_opened_score", "opportunity_score"),
    "role_trajectory_score": ("role_trajectory_score",),
}
FEATURES = list(MP._HIT_MODEL_FEATURES)


def _feat(row: dict, name: str) -> float:
    for k in _FEATURE_KEYS[name]:
        v = row.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _load_scores(season: int, scores_json: str | None) -> list[dict]:
    if scores_json:
        return load_breakout_scores_from_json(season, scores_json)
    return load_breakout_scores_from_db(season)


def assemble(seasons, scores_json):
    """Return rows of {position, feats..., y} across all source seasons."""
    data = []
    for s in seasons:
        try:
            scores = _load_scores(s, scores_json)
        except Exception as e:
            print(f"[train] season {s}: no scores ({e}) — skipping")
            continue
        outcomes = load_actual_outcomes(s + 1)
        pos_map = load_position_map(s + 1, scores_json)
        top = get_top12_pids(outcomes, pos_map)
        n0 = len(data)
        for r in scores:
            pid = str(r.get("player_id") or r.get("id") or "")
            pos = str(r.get("position") or pos_map.get(pid, "")).upper()
            if not pid or pos not in ("QB", "RB", "WR", "TE"):
                continue
            row = {"position": pos, "y": 1 if pid in top else 0}
            for f in FEATURES:
                row[f] = _feat(r, f)
            data.append(row)
        print(f"[train] season {s}->{s+1}: {len(data)-n0} candidates, "
              f"{sum(1 for d in data[n0:] if d['y'])} hits")
    return data


def _fit_block(rows):
    """Fit a standardized logistic on `rows`; return the JSON block + metrics."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score, brier_score_loss

    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r["y"] for r in rows], dtype=int)
    if len(y) < 40 or y.sum() < 5 or y.sum() == len(y):
        return None
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=1000)
    # Honest metrics via out-of-fold predictions.
    try:
        oof = cross_val_predict(clf, Xs, y, cv=5, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, oof))
        brier = float(brier_score_loss(y, oof))
    except Exception:
        auc = brier = float("nan")
    clf.fit(Xs, y)
    return {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "n": int(len(y)), "hits": int(y.sum()),
        "auc": round(auc, 4), "brier": round(brier, 4),
    }


def _curve_metrics(rows):
    """AUC/Brier of the current empirical curve on the same rows (baseline)."""
    from sklearn.metrics import roc_auc_score, brier_score_loss
    # Force the curve path regardless of any existing model file.
    MP._load_hit_model.cache_clear()
    _orig = MP._load_hit_model
    MP._load_hit_model = lambda: None
    try:
        preds, ys = [], []
        for r in rows:
            preds.append(MP.calculate_hit_probability(
                r["breakout_score"], r["readiness_score"], r["confidence_score"],
                r["position"], r["opportunity_score"], r["role_trajectory_score"]))
            ys.append(r["y"])
    finally:
        MP._load_hit_model = _orig
        MP._load_hit_model.cache_clear()
    try:
        return round(float(roc_auc_score(ys, preds)), 4), round(float(brier_score_loss(ys, preds)), 4)
    except Exception:
        return float("nan"), float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", required=True,
                    help="source seasons S (labels come from S+1)")
    ap.add_argument("--scores-json", default=None,
                    help="dir of breakout_scores_{S}.json (else read the DB)")
    ap.add_argument("--out", default=str(MODEL_PATH))
    ap.add_argument("--write", action="store_true",
                    help="write the model JSON (otherwise dry-run: report only)")
    args = ap.parse_args()

    rows = assemble(args.seasons, args.scores_json)
    if not rows:
        print("[train] no data assembled — check --seasons / DB / --scores-json")
        return
    print(f"[train] total: {len(rows)} candidate-seasons, {sum(r['y'] for r in rows)} hits\n")

    positions = {}
    for pos in ("QB", "RB", "WR", "TE"):
        blk = _fit_block([r for r in rows if r["position"] == pos])
        if blk:
            positions[pos] = blk
    g = _fit_block(rows)
    if g:
        positions["_global"] = g

    base_auc, base_brier = _curve_metrics(rows)
    print("  Model vs current curve (higher AUC / lower Brier = better):")
    print(f"    {'block':<8} {'n':>5} {'hits':>5} {'AUC':>7} {'Brier':>7}")
    for k, b in positions.items():
        print(f"    {k:<8} {b['n']:>5} {b['hits']:>5} {b['auc']:>7} {b['brier']:>7}")
    print(f"    {'curve':<8} {len(rows):>5} {sum(r['y'] for r in rows):>5} {base_auc:>7} {base_brier:>7}  (baseline)")

    model = {
        "version": 1,
        "trained_at": date.today().isoformat(),
        "seasons": args.seasons,
        "features": FEATURES,
        "hit_def": "top-6 QB/TE, top-12 RB/WR by total PPR, >=10 games",
        "positions": positions,
    }
    if args.write:
        Path(args.out).write_text(json.dumps(model, indent=2), encoding="utf-8")
        print(f"\n[train] wrote {args.out} — review the AUC/Brier above, then commit it.")
    else:
        print("\n[train] dry-run (no file written). Re-run with --write to save the model.")


if __name__ == "__main__":
    main()
