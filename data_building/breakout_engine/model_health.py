"""Breakout-model health metrics + a champion/challenger gate metric.

This turns the ad-hoc backtest analysis into a repeatable, importable tool:

  * `load_eval_rows` / `pooled_metrics` — reusable, programmatic backtest metrics
    (breakout AUC, Precision@K, Brier) across seasons, from a --scores-json dir
    or the DB, using the SAME breakout label definition as backtest_multitask.
  * `gate_metric` — a single composite score used to decide whether a challenger
    model may replace the incumbent (see retrain_guarded.py).
  * CLI: a one-command model-health report + an optional blend-weight sweep, so
    the yearly "is the model still sound / has enough data accrued?" check is
    `python -m data_building.breakout_engine.model_health --scores-json DIR
     --seasons 2022 2023 2024`.

Nothing here runs at inference time; it's an offline analysis/ops tool and may
use sklearn (already a dev dependency of train_hit_probability).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data_building.breakout_engine import backtest_multitask as _bt
from data_building.breakout_engine.multitask_predictions import _hit_prob_from_model, _load_hit_model

FEATURES = ("opportunity_opened_score", "competition_removed_score",
            "team_environment_score", "player_readiness_score",
            "role_trajectory_score", "confidence_score")


def load_eval_rows(season: int, scores_json: Optional[str], min_score: float = 30.0) -> list[dict]:
    """Matched candidate rows for a prediction `season`, with features, the shipped
    prob/score, the actual outcome, and the breakout label (from season+1)."""
    if scores_json:
        cands = _bt.load_breakout_scores_from_json(season, scores_json, min_score)
    else:
        cands = _bt.load_breakout_scores_from_db(season, min_score)
    outcomes = _bt.load_actual_outcomes(season + 1)
    source = _bt.load_source_stats(season)
    breakouts = _bt.get_breakout_pids(outcomes, source)
    committed = _load_hit_model()   # for rows that don't carry a stored prob (e.g. DB path)

    rows: list[dict] = []
    for c in cands:
        pid = c.get("player_id")
        if pid not in outcomes:            # only players we can grade
            continue
        pos = c.get("position") or "WR"
        feats = {f: float(c.get(f) or 0) for f in FEATURES}
        # Prefer the stored hit_probability; if a row doesn't carry one (some DB
        # rows / older exports), compute it from the committed model so the
        # "shipped prob" report reflects the real model, not zeros.
        raw = c.get("hit_probability")
        if raw is not None:
            prob = float(raw)
        elif committed is not None:
            p = _hit_prob_from_model(committed, pos, feats)
            prob = float(p) if p is not None else 0.0
        else:
            prob = 0.0
        rows.append({
            "pid": pid,
            "pos": pos,
            "feats": feats,
            "score": float(c.get("breakout_opportunity_score") or 0) / 100.0,
            "prob": prob,
            "hit": 1 if pid in breakouts else 0,
            "prev_ppg": float(source.get(pid, {}).get("ppr_ppg") or 0),
            "actual_ppg": float(outcomes[pid]["ppr_ppg"]),
        })
    return rows


def prob_under_model(row: dict, model: Optional[dict]) -> float:
    """P(breakout) for a row under `model`. Falls back to the row's shipped prob
    for curve positions / missing blocks, so two models are compared only where
    they actually differ."""
    if model is None:
        return row["prob"]
    if row["pos"] in (model.get("curve_positions") or []):
        return row["prob"]
    p = _hit_prob_from_model(model, row["pos"], row["feats"])
    return row["prob"] if p is None else float(p)


def _precision_at_k(scores, hits, k: int) -> float:
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    if not order:
        return 0.0
    return sum(hits[i] for i in order) / len(order)


def _auc(scores, hits) -> Optional[float]:
    from sklearn.metrics import roc_auc_score
    if len(set(hits)) < 2:
        return None
    return float(roc_auc_score(hits, scores))


def _brier(probs, hits) -> float:
    return sum((p - y) ** 2 for p, y in zip(probs, hits)) / max(1, len(hits))


def pooled_metrics(
    rows_by_season: dict[int, list[dict]],
    score_fn: Callable[[dict], float],
    prob_fn: Optional[Callable[[dict], float]] = None,
) -> dict:
    """Metrics for a ranking `score_fn` (higher = more likely to break out),
    pooled across seasons. `prob_fn` (if given) is the calibrated probability used
    for Brier; defaults to score_fn."""
    prob_fn = prob_fn or score_fn
    all_scores, all_probs, all_hits, p10s, p20s = [], [], [], [], []
    per_season = {}
    for season, rows in rows_by_season.items():
        if not rows:
            continue
        s = [score_fn(r) for r in rows]
        p = [prob_fn(r) for r in rows]
        h = [r["hit"] for r in rows]
        all_scores += s; all_probs += p; all_hits += h
        p10 = _precision_at_k(s, h, 10)
        p20 = _precision_at_k(s, h, 20)
        p10s.append(p10); p20s.append(p20)
        per_season[season] = {"n": len(rows), "hits": sum(h),
                              "p_at_10": p10, "p_at_20": p20}
    return {
        "auc": _auc(all_scores, all_hits),
        "brier": _brier(all_probs, all_hits),
        "mean_p_at_10": sum(p10s) / len(p10s) if p10s else 0.0,
        "mean_p_at_20": sum(p20s) / len(p20s) if p20s else 0.0,
        "n": len(all_hits),
        "hits": sum(all_hits),
        "per_season": per_season,
    }


def gate_metric(metrics: dict) -> float:
    """Single number a challenger must beat to be adopted: discrimination (AUC)
    weighted with top-of-board precision (Precision@10). Both are in [0,1] and
    both are ranking-quality measures, which is what the board is judged on."""
    auc = metrics.get("auc")
    auc = 0.5 if auc is None else auc
    return 0.6 * auc + 0.4 * metrics.get("mean_p_at_10", 0.0)


def blend_metric_fn(weight: float) -> Callable[[dict], float]:
    """Ranking = weight*score + (1-weight)*shipped-prob (the live board's blend)."""
    return lambda r: weight * r["score"] + (1.0 - weight) * r["prob"]


# --------------------------------------------------------------------------- CLI

def _fmt(x, pct=False):
    if x is None:
        return "  n/a"
    return f"{x*100:.1f}%" if pct else f"{x:.3f}"


def _report(rows_by_season, model=None, model_label="shipped prob"):
    def score_fn(r):
        return prob_under_model(r, model)
    m = pooled_metrics(rows_by_season, score_fn)
    print(f"\n=== Model health ({model_label}) ===")
    print(f"pool: N={m['n']} candidates, {m['hits']} breakouts "
          f"({m['hits']/max(1,m['n'])*100:.0f}% base rate)")
    print(f"{'season':>8}{'N':>6}{'breakouts':>11}{'P@10':>8}{'P@20':>8}")
    for s, d in sorted(m["per_season"].items()):
        print(f"{s:>8}{d['n']:>6}{d['hits']:>11}{_fmt(d['p_at_10'], True):>8}{_fmt(d['p_at_20'], True):>8}")
    print(f"{'POOLED':>8}{'':>6}{'':>11}{_fmt(m['mean_p_at_10'], True):>8}{_fmt(m['mean_p_at_20'], True):>8}")
    print(f"  breakout AUC: {_fmt(m['auc'])}   Brier: {_fmt(m['brier'])}   gate: {gate_metric(m):.3f}")
    return m


def _sweep(rows_by_season):
    print("\n=== Blend-weight sweep (ranking = w*score + (1-w)*prob) ===")
    print(f"{'w':>5}{'AUC':>8}{'meanP@10':>10}")
    best = None
    for wi in range(0, 21):
        w = wi / 20.0
        m = pooled_metrics(rows_by_season, blend_metric_fn(w))
        g = gate_metric(m)
        if best is None or g > best[1]:
            best = (w, g, m["auc"], m["mean_p_at_10"])
        print(f"{w:>5.2f}{_fmt(m['auc']):>8}{_fmt(m['mean_p_at_10'], True):>10}")
    print(f"\nBest gate at w={best[0]:.2f} (AUC {_fmt(best[2])}, meanP@10 {_fmt(best[3], True)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seasons", type=int, nargs="+", required=True,
                    help="prediction seasons to evaluate (labels come from season+1)")
    ap.add_argument("--scores-json", default=None,
                    help="dir of breakout_scores_{S}.json (else read the DB)")
    ap.add_argument("--min-score", type=float, default=45.0,
                    help="min breakout score to include (default 45 ~ the board-quality "
                         "pool users actually see; lower to evaluate the wider pool)")
    ap.add_argument("--model", default=None,
                    help="score under this model JSON instead of the shipped prob "
                         "(for comparing a candidate model)")
    ap.add_argument("--sweep", action="store_true", help="also run the blend-weight sweep")
    args = ap.parse_args()

    rows_by_season = {}
    for s in args.seasons:
        try:
            rows_by_season[s] = load_eval_rows(s, args.scores_json, args.min_score)
        except FileNotFoundError as e:
            print(f"[health] season {s}: {e} — skipping")
    if not any(rows_by_season.values()):
        print("[health] no evaluable seasons — check --seasons / --scores-json / DB")
        return

    model = None
    label = "shipped prob"
    if args.model:
        import json
        model = json.loads(Path(args.model).read_text(encoding="utf-8"))
        label = f"model {Path(args.model).name}"
    _report(rows_by_season, model, label)
    if args.sweep:
        _sweep(rows_by_season)


if __name__ == "__main__":
    main()
