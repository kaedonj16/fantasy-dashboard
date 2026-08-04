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


def load_outcomes_parquet(season: int) -> dict:
    """Clean per-player outcomes for `season` from the committed player_history
    parquets (sleeper_id, position, games, ppr_ppg) — the usage_rows JSONs are in
    inconsistent formats (2023 has no position/ppr), which silently zeroed labels.
    Returns {} if no parquet covers the season, so assemble() can fall back."""
    import pandas as pd
    candidates = [
        Path("cache/player_history") / f"player_history_{season}.parquet",
        Path("cache/player_history") / "player_history_all.parquet",
    ]
    for path in candidates:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "season" in df.columns:
            df = df[df["season"] == season]
        if df.empty:
            continue
        out = {}
        for _, r in df.iterrows():
            pid = str(r.get("sleeper_id") or "")
            if not pid or pid == "nan":
                continue
            games = int(r.get("games") or 0)
            ppg = float(r.get("ppr_ppg") or 0)
            out[pid] = {
                "total_ppr": round(ppg * games, 1), "ppr_ppg": ppg, "games": games,
                "position": str(r.get("position") or ""), "name": r.get("name", ""),
            }
        if out:
            return out
    return {}


def _index_position_map() -> dict:
    """sleeper_id -> position from the players index. The committed outcome files
    (usage_rows JSON and player_history parquet) have no position column, so
    get_top12_pids needs this to bucket finishers by position."""
    try:
        from utils.utils import load_players_index
        idx = load_players_index() or {}
        return {str(k): str((v or {}).get("pos") or (v or {}).get("position") or "").upper()
                for k, v in idx.items()}
    except Exception as e:
        print(f"[train] players-index position map unavailable: {e}")
        return {}


def _breakout_hits(outcomes: dict, prior: dict, growth: float,
                   min_ppg: float, scratch_ppg: float) -> set:
    """Relative 'breakout' hits: a meaningful PPG jump over the player's OWN prior
    season (≥10 games in the outcome season). Configurable growth multiple + PPG
    floor (the floor stops tiny baselines like 2→5 PPG counting as breakouts).
    Players with no real prior must clear scratch_ppg on their own."""
    hits = set()
    for pid, o in outcomes.items():
        if o["games"] < 10:
            continue
        actual = o["ppr_ppg"]
        pr = prior.get(pid, {})
        p_ppg, p_games = float(pr.get("ppr_ppg") or 0), int(pr.get("games") or 0)
        if p_games >= 6 and p_ppg >= 4.0:
            if actual >= p_ppg * growth and actual >= min_ppg:
                hits.add(pid)
        elif actual >= scratch_ppg:
            hits.add(pid)
    return hits


def assemble(seasons, scores_json, target="top_n", growth=1.15,
             min_ppg=7.0, scratch_ppg=10.0):
    """Return rows of {position, feats..., y} across all source seasons.

    target: 'top_n'    -> label = top-12 RB/WR, top-6 QB/TE finish (absolute)
            'breakout' -> label = >=`growth`x prior PPG (+min_ppg floor), relative
    """
    data = []
    _pos_idx = _index_position_map()
    for s in seasons:
        try:
            scores = _load_scores(s, scores_json)
        except Exception as e:
            print(f"[train] season {s}: no scores ({e}) — skipping")
            continue
        outcomes = load_outcomes_parquet(s + 1)
        if not outcomes:
            try:
                outcomes = load_actual_outcomes(s + 1)  # JSON fallback
            except FileNotFoundError:
                print(f"[train] season {s+1}: no outcomes (parquet or JSON) — skipping")
                continue
        # Outcome files carry no position — bucket finishers via the players index
        # (merged with any build_historical_scores position file if present).
        pos_map = dict(_pos_idx)
        pos_map.update(load_position_map(s + 1, scores_json) or {})
        if target == "breakout":
            prior = load_outcomes_parquet(s)  # the candidate's own prior season
            top = _breakout_hits(outcomes, prior, growth, min_ppg, scratch_ppg)
        else:
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
    # NOTE: no class_weight balancing. The displayed value is a *probability*, so
    # calibration matters as much as ranking; balancing upweights the rare
    # positives and pushes probs toward ~0.5 (great AUC, terrible Brier / wildly
    # overstated on-screen %). Plain logistic keeps the base rate honest.
    clf = LogisticRegression(C=0.5, max_iter=1000)
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
    ap.add_argument("--target", choices=("top_n", "breakout"), default="top_n",
                    help="label: 'top_n' (top-12 RB/WR, top-6 QB/TE, absolute) or "
                         "'breakout' (relative PPG growth)")
    ap.add_argument("--growth", type=float, default=1.15,
                    help="breakout target: min PPG multiple over prior season (default 1.15 = +15%%)")
    ap.add_argument("--min-ppg", type=float, default=7.0,
                    help="breakout target: absolute PPG floor for a hit (default 7.0)")
    ap.add_argument("--scratch-ppg", type=float, default=10.0,
                    help="breakout target: PPG bar for players with no real prior (default 10.0)")
    args = ap.parse_args()

    rows = assemble(args.seasons, args.scores_json, target=args.target,
                    growth=args.growth, min_ppg=args.min_ppg, scratch_ppg=args.scratch_ppg)
    if not rows:
        print("[train] no data assembled — check --seasons / DB / --scores-json")
        return
    _hit_def = ("top-6 QB/TE, top-12 RB/WR by total PPR, >=10 games" if args.target == "top_n"
                else f">= {args.growth:g}x prior PPG and >= {args.min_ppg:g} PPG "
                     f"(or >= {args.scratch_ppg:g} from scratch), >=10 games")
    print(f"[train] target={args.target}  ({_hit_def})")
    print(f"[train] total: {len(rows)} candidate-seasons, {sum(r['y'] for r in rows)} hits "
          f"({100*sum(r['y'] for r in rows)/max(len(rows),1):.1f}% hit rate)\n")

    g = _fit_block(rows)
    positions = {}
    if g:
        positions["_global"] = g
    # Keep a per-position block ONLY when it actually beats the pooled global
    # model on out-of-fold AUC. Thin positions (few hits) otherwise fit noise and
    # would drag their group below the global fallback, which inference prefers.
    g_auc = (g or {}).get("auc", 0.0)
    dropped = []
    for pos in ("QB", "RB", "WR", "TE"):
        blk = _fit_block([r for r in rows if r["position"] == pos])
        if blk and blk["auc"] > g_auc:
            positions[pos] = blk
        elif blk:
            dropped.append(f"{pos}(AUC {blk['auc']:.3f}<=global {g_auc:.3f}, hits={blk['hits']})")

    base_auc, base_brier = _curve_metrics(rows)
    print("  Kept blocks vs current curve (higher AUC / lower Brier = better):")
    print(f"    {'block':<8} {'n':>5} {'hits':>5} {'AUC':>7} {'Brier':>7}")
    for k, b in positions.items():
        print(f"    {k:<8} {b['n']:>5} {b['hits']:>5} {b['auc']:>7} {b['brier']:>7}")
    print(f"    {'curve':<8} {len(rows):>5} {sum(r['y'] for r in rows):>5} {base_auc:>7} {base_brier:>7}  (baseline)")
    if dropped:
        print(f"  dropped (fall back to _global): {', '.join(dropped)}")

    model = {
        "version": 1,
        "trained_at": date.today().isoformat(),
        "seasons": args.seasons,
        "features": FEATURES,
        "target": args.target,
        "hit_def": _hit_def,
        "positions": positions,
    }
    if args.write:
        Path(args.out).write_text(json.dumps(model, indent=2), encoding="utf-8")
        print(f"\n[train] wrote {args.out} — review the AUC/Brier above, then commit it.")
    else:
        print("\n[train] dry-run (no file written). Re-run with --write to save the model.")


if __name__ == "__main__":
    main()
