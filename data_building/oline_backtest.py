"""
Backtest for the nflverse-derived O-line ratings.

The point of this file is honesty: piling metrics onto a rating makes it look
sophisticated but can make it *worse*, and you can't tell which without a test.
This measures two things that a real O-line signal should have:

  1. OUT-OF-SAMPLE PREDICTIVENESS -- ratings built from the first half of a
     season should predict the SAME team's protection/run outcomes in the second
     half. If pass-block rating from weeks 1..k correlates with low pressure
     allowed in weeks k+1.., the metric is capturing something that persists,
     not noise. Reported as Spearman rho (rank correlation; robust to outliers).

  2. YEAR-OVER-YEAR STABILITY -- a unit's rating in year Y vs year Y+1. Lines
     change (draft, free agency, injury) so this won't be huge, but a rating
     with ~zero year-over-year signal is measuring randomness.

It also reports the split-half correlation of the raw inputs (pressure rate,
sack rate, line yards) as a ceiling: the composite can't be more stable than
its ingredients.

Everything is computed from open nflverse play-by-play, no licensed data.

Run:  python -m data_building.oline_backtest [season ...]
      python -m data_building.oline_backtest 2022 2023 2024
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from utils.paths import CACHE_DIR
from data_building.oline_ratings import (
    _prep_season, _load_pbp_year, _season_run_metrics, _season_pass_metrics,
    _regress_to_prior, _residualize, _percentile_index,
    RUN_PRIOR_K, PASS_PRIOR_K, PRESSURE_WEIGHT, SACK_WEIGHT,
    PASS_BLOCK_WEIGHT, RUN_BLOCK_WEIGHT,
)


def _spearman(pairs):
    """Rank correlation for [(x, y), ...]. Returns (rho, n)."""
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    n = len(pairs)
    if n < 4:
        return None, n

    def _ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0   # average rank for ties (1-based)
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = _ranks([x for x, _ in pairs])
    ry = _ranks([y for _, y in pairs])
    mx = sum(rx) / n
    my = sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:
        return None, n
    return sxy / (sxx ** 0.5 * syy ** 0.5), n


def _week_bounds(d, pd):
    wk = pd.to_numeric(d["week"], errors="coerce")
    return wk


def _ratings_from_frame(cur, pd):
    """Full rating pipeline on a single already-filtered frame, no prior blend.

    Used inside the backtest where each split is treated standalone. Returns
    {team: {pass_block, run_block, composite, pressure_rate, sack_rate, line_yards}}.
    """
    run_adj, run_n, run_league, run_detail = _season_run_metrics(cur)
    run_final = _regress_to_prior(run_adj, run_n, {}, run_league, RUN_PRIOR_K)
    cp = _season_pass_metrics(cur)
    press_final = _regress_to_prior(cp["press_adj"], cp["n"], None, cp["press_league"], PASS_PRIOR_K)
    sack_final = _regress_to_prior(cp["sack_adj"], cp["n"], None, cp["sack_league"], PASS_PRIOR_K)
    press_resid = _residualize(press_final, cp["ttt"])
    sack_resid = _residualize(sack_final, cp["ttt"])
    run_index = _percentile_index(run_final, higher_is_better=True)
    press_index = _percentile_index(press_resid, higher_is_better=False)
    sack_index = _percentile_index(sack_resid, higher_is_better=False)
    out = {}
    for t in set(run_final) | set(press_final):
        p_idx, s_idx, r_idx = press_index.get(t), sack_index.get(t), run_index.get(t)
        pass_idx = None
        if p_idx is not None and s_idx is not None:
            pass_idx = PRESSURE_WEIGHT * p_idx + SACK_WEIGHT * s_idx
        comp = None
        if pass_idx is not None and r_idx is not None:
            comp = PASS_BLOCK_WEIGHT * pass_idx + RUN_BLOCK_WEIGHT * r_idx
        rd = run_detail.get(t, {})
        pdd = cp["detail"].get(t, {})
        out[t] = {
            "pass_block": pass_idx, "run_block": r_idx, "composite": comp,
            "pressure_rate": pdd.get("pressure_rate"),
            "sack_rate": pdd.get("sack_rate"),
            "line_yards": rd.get("line_yards"),
            "n_pass": cp["n"].get(t, 0), "n_rush": run_n.get(t, 0),
        }
    return out, cp["pressure_source"]


def run_backtest(seasons, split_week=9, save=True):
    import pandas as pd
    try:
        import nfl_data_py as nfl
    except Exception:
        nfl = None

    # Accumulators for pooled correlations across seasons.
    pred_pass, pred_sack, pred_run = [], [], []   # (rating_1sthalf, outcome_2ndhalf)
    stab_pass_in, stab_sack_in, stab_ly_in = [], [], []
    yoy = {}          # season -> {team: composite} for year-over-year
    pressure_sources = set()

    for season in seasons:
        raw = _prep_season(_load_pbp_year(season, pd, nfl), pd, season)
        if raw is None or raw.empty:
            print(f"[oline_backtest] {season}: no data, skipping")
            continue
        wk = _week_bounds(raw, pd)
        first = raw[wk <= split_week]
        second = raw[wk > split_week]
        if first.empty or second.empty:
            print(f"[oline_backtest] {season}: not enough weeks around split {split_week}")
            continue

        rate1, src = _ratings_from_frame(first, pd)
        pressure_sources.add(src)
        # Second-half OUTCOMES (raw rates, not indices) computed directly.
        _, _, _, run2 = _season_run_metrics(second)
        cp2 = _season_pass_metrics(second)
        out2_pass = cp2["detail"]

        for t, r1 in rate1.items():
            o2p = out2_pass.get(t, {})
            o2r = run2.get(t, {})
            if r1.get("pass_block") is not None and o2p.get("pressure_rate") is not None:
                pred_pass.append((r1["pass_block"], o2p["pressure_rate"]))
            if r1.get("pass_block") is not None and o2p.get("sack_rate") is not None:
                pred_sack.append((r1["pass_block"], o2p["sack_rate"]))
            if r1.get("run_block") is not None and o2r.get("line_yards") is not None:
                pred_run.append((r1["run_block"], o2r["line_yards"]))
        # split-half stability of the raw inputs
        cp1 = _season_pass_metrics(first)["detail"]
        _, _, _, run1 = _season_run_metrics(first)
        for t in set(cp1) & set(out2_pass):
            if cp1[t].get("pressure_rate") is not None and out2_pass[t].get("pressure_rate") is not None:
                stab_pass_in.append((cp1[t]["pressure_rate"], out2_pass[t]["pressure_rate"]))
            if cp1[t].get("sack_rate") is not None and out2_pass[t].get("sack_rate") is not None:
                stab_sack_in.append((cp1[t]["sack_rate"], out2_pass[t]["sack_rate"]))
        for t in set(run1) & set(run2):
            if run1[t].get("line_yards") is not None and run2[t].get("line_yards") is not None:
                stab_ly_in.append((run1[t]["line_yards"], run2[t]["line_yards"]))

        # full-season composite for year-over-year
        full, _ = _ratings_from_frame(raw, pd)
        yoy[season] = {t: v["composite"] for t, v in full.items() if v["composite"] is not None}

    # year-over-year pairs
    yoy_pairs = []
    for s in seasons:
        if s in yoy and (s + 1) in yoy:
            for t in set(yoy[s]) & set(yoy[s + 1]):
                yoy_pairs.append((yoy[s][t], yoy[s + 1][t]))

    def _rep(label, pairs, note=""):
        rho, n = _spearman(pairs)
        rho_s = f"{rho:+.3f}" if rho is not None else "  n/a"
        print(f"  {label:<46} rho={rho_s}  (n={n}) {note}")
        return {"rho": rho, "n": n}

    print(f"\n[oline_backtest] seasons={list(seasons)} split_week={split_week} "
          f"pressure_source={'/'.join(sorted(pressure_sources)) or 'n/a'}")
    print("PREDICTIVE (1st-half rating -> 2nd-half outcome; negative is good for pass):")
    res = {"predictive": {}, "stability": {}, "yoy": {}}
    res["predictive"]["pass_block_vs_future_pressure"] = _rep(
        "pass_block -> future pressure rate", pred_pass, "(want negative)")
    res["predictive"]["pass_block_vs_future_sack"] = _rep(
        "pass_block -> future sack rate", pred_sack, "(want negative)")
    res["predictive"]["run_block_vs_future_line_yards"] = _rep(
        "run_block -> future line yards", pred_run, "(want positive)")
    print("SPLIT-HALF INPUT STABILITY (ceiling on the composite):")
    res["stability"]["pressure_rate"] = _rep("pressure rate 1st vs 2nd half", stab_pass_in, "(want positive)")
    res["stability"]["sack_rate"] = _rep("sack rate 1st vs 2nd half", stab_sack_in, "(want positive)")
    res["stability"]["line_yards"] = _rep("line yards 1st vs 2nd half", stab_ly_in, "(want positive)")
    print("YEAR-OVER-YEAR COMPOSITE STABILITY:")
    res["yoy"]["composite"] = _rep("composite year Y vs Y+1", yoy_pairs, "(want positive)")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seasons": list(seasons), "split_week": split_week,
        "pressure_source": "/".join(sorted(pressure_sources)),
        "results": res,
    }
    if save:
        os.makedirs(str(CACHE_DIR), exist_ok=True)
        path = os.path.join(str(CACHE_DIR), "oline_backtest.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n[oline_backtest] report -> {path}")
    return report


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    seasons = args or [2022, 2023, 2024]
    run_backtest(seasons)
