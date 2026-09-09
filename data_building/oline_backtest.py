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
    grade_indices, _apply_talent_to_priors,
    RUN_PRIOR_K, PASS_PRIOR_K, PRESSURE_WEIGHT, SACK_WEIGHT,
    PASS_BLOCK_WEIGHT, RUN_BLOCK_WEIGHT, RUN_LY_WEIGHT, RUN_SUCCESS_WEIGHT,
    TALENT_PRIOR_W, TALENT_W_CONTINUITY, TALENT_W_DRAFT, TALENT_W_VETERAN,
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
    run_ly_adj, run_su_adj, run_n, run_league, run_detail = _season_run_metrics(cur)
    run_ly_final = _regress_to_prior(run_ly_adj, run_n, {}, run_league, RUN_PRIOR_K)
    su_league = (sum(run_su_adj.values()) / len(run_su_adj)) if run_su_adj else 0.0
    run_su_final = _regress_to_prior(run_su_adj, run_n, {}, su_league, RUN_PRIOR_K) if run_su_adj else {}
    cp = _season_pass_metrics(cur)
    press_final = _regress_to_prior(cp["press_adj"], cp["n"], None, cp["press_league"], PASS_PRIOR_K)
    sack_final = _regress_to_prior(cp["sack_adj"], cp["n"], None, cp["sack_league"], PASS_PRIOR_K)
    press_resid = _residualize(press_final, cp["ttt"])
    sack_resid = _residualize(sack_final, cp["ttt"])
    ly_index = _percentile_index(run_ly_final, higher_is_better=True)
    su_index = _percentile_index(run_su_final, higher_is_better=True) if run_su_final else {}
    run_index = {}
    for t in set(ly_index) | set(su_index):
        ly_i, su_i = ly_index.get(t), su_index.get(t)
        if ly_i is not None and su_i is not None:
            run_index[t] = RUN_LY_WEIGHT * ly_i + RUN_SUCCESS_WEIGHT * su_i
        elif ly_i is not None:
            run_index[t] = ly_i
        elif su_i is not None:
            run_index[t] = su_i
    press_index = _percentile_index(press_resid, higher_is_better=False)
    sack_index = _percentile_index(sack_resid, higher_is_better=False)
    out = {}
    for t in set(run_ly_final) | set(press_final):
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


def _components_from_frame(cur, pd):
    """Per-team scaled component indices (press, sack, run) for one frame.

    Same pipeline as _ratings_from_frame but returns the pieces before the
    pass-block blend, so the sweep can recombine them at different weights
    without recomputing the expensive opponent adjustment each time.
    """
    run_ly_adj, run_su_adj, run_n, run_league, _ = _season_run_metrics(cur)
    run_ly_final = _regress_to_prior(run_ly_adj, run_n, {}, run_league, RUN_PRIOR_K)
    su_league = (sum(run_su_adj.values()) / len(run_su_adj)) if run_su_adj else 0.0
    run_su_final = _regress_to_prior(run_su_adj, run_n, {}, su_league, RUN_PRIOR_K) if run_su_adj else {}
    cp = _season_pass_metrics(cur)
    press_final = _regress_to_prior(cp["press_adj"], cp["n"], None, cp["press_league"], PASS_PRIOR_K)
    sack_final = _regress_to_prior(cp["sack_adj"], cp["n"], None, cp["sack_league"], PASS_PRIOR_K)
    press_index = _percentile_index(_residualize(press_final, cp["ttt"]), higher_is_better=False)
    sack_index = _percentile_index(_residualize(sack_final, cp["ttt"]), higher_is_better=False)
    ly_index = _percentile_index(run_ly_final, higher_is_better=True)
    su_index = _percentile_index(run_su_final, higher_is_better=True) if run_su_final else {}
    run_index = {}
    for t in set(ly_index) | set(su_index):
        ly_i, su_i = ly_index.get(t), su_index.get(t)
        if ly_i is not None and su_i is not None:
            run_index[t] = RUN_LY_WEIGHT * ly_i + RUN_SUCCESS_WEIGHT * su_i
        elif ly_i is not None:
            run_index[t] = ly_i
        elif su_i is not None:
            run_index[t] = su_i
    comp = {}
    for t in set(press_index) | set(sack_index) | set(run_index):
        comp[t] = {"press": press_index.get(t), "sack": sack_index.get(t),
                   "run": run_index.get(t)}
    return comp


def sweep_pressure_weight(seasons, split_week=9, grid=None, save=True):
    """Empirically pick the pressure-vs-sack weight for pass_block.

    For each candidate weight w, pass_block = w*pressure_index + (1-w)*sack_index.
    We score each w by how well that pass_block (from the first half of a season)
    predicts the SAME team's second-half pressure and sack rates out of sample,
    pooled across seasons. Lower future pressure/sacks = better line, so a good
    weight makes both rank correlations strongly negative; the reported score is
    the mean of the two negated rhos (higher = better).

    This is the honest answer to "is pressure really worth more than sacks?" --
    it replaces the 0.65 guess with a measured value.
    """
    import pandas as pd
    try:
        import nfl_data_py as nfl
    except Exception:
        nfl = None

    grid = grid or [round(i / 10, 1) for i in range(11)]
    # Gather first-half components and second-half outcomes once per season.
    seasons_data = []
    for season in seasons:
        raw = _prep_season(_load_pbp_year(season, pd, nfl), pd, season)
        if raw is None or raw.empty:
            continue
        wk = _week_bounds(raw, pd)
        first, second = raw[wk <= split_week], raw[wk > split_week]
        if first.empty or second.empty:
            continue
        comp1 = _components_from_frame(first, pd)
        out2 = _season_pass_metrics(second)["detail"]
        seasons_data.append((comp1, out2))
    if not seasons_data:
        print("[oline_backtest] sweep: no usable seasons")
        return {}

    results = []
    for w in grid:
        pairs_p, pairs_s = [], []
        for comp1, out2 in seasons_data:
            for t, c in comp1.items():
                if c["press"] is None or c["sack"] is None:
                    continue
                pb = w * c["press"] + (1 - w) * c["sack"]
                o = out2.get(t, {})
                if o.get("pressure_rate") is not None:
                    pairs_p.append((pb, o["pressure_rate"]))
                if o.get("sack_rate") is not None:
                    pairs_s.append((pb, o["sack_rate"]))
        rho_p, _ = _spearman(pairs_p)
        rho_s, _ = _spearman(pairs_s)
        if rho_p is None or rho_s is None:
            continue
        score = -(rho_p + rho_s) / 2.0   # want both negative -> higher score better
        results.append({"w_pressure": w, "rho_future_pressure": round(rho_p, 3),
                        "rho_future_sack": round(rho_s, 3), "score": round(score, 4)})

    results.sort(key=lambda r: r["score"], reverse=True)
    print(f"\n[oline_backtest] pressure-weight sweep (seasons={list(seasons)}, split={split_week})")
    print(f"  {'w_press':>8} {'rho_press':>10} {'rho_sack':>10} {'score':>8}")
    for r in results:
        star = "  <- best" if r is results[0] else ""
        print(f"  {r['w_pressure']:>8} {r['rho_future_pressure']:>10} "
              f"{r['rho_future_sack']:>10} {r['score']:>8}{star}")
    best = results[0] if results else {}
    report = {"generated_at": datetime.now(timezone.utc).isoformat(),
              "seasons": list(seasons), "split_week": split_week,
              "grid": results, "best": best}
    if save and best:
        path = os.path.join(str(CACHE_DIR), "oline_weight_sweep.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[oline_backtest] sweep report -> {path}")
        print(f"[oline_backtest] best pressure weight = {best.get('w_pressure')}")
    return report


def _early_rest_frames(raw, pd, early_week):
    wk = _week_bounds(raw, pd)
    early, rest = raw[wk <= early_week], raw[wk > early_week]
    return early, rest


def _blocks_from_indices(g):
    out = {}
    for t in set(g["pass_index"]) | set(g["run_index"]):
        out[t] = {
            "pass_block": g["pass_index"].get(t),
            "run_block": g["run_index"].get(t),
        }
    return out


def sweep_talent_prior(seasons, early_week=4, save=True,
                       weights=None, mixes=None):
    """Sweep the talent-prior scale / mix on early-season out-of-sample prediction.

    Ratings are built from weeks 1..early_week of season Y (plus last season's
    realized prior, optionally shifted by the offseason roster/draft residual)
    and scored against weeks early_week+1.. of the SAME season's raw pass-block
    (pressure, sacks) and run-block (line yards, success) outcomes.

    Weight 0 is the current last-season prior — the baseline to beat. If nothing
    beats it, do not ship `use_talent_prior=True` by default.

    Grid (documented here so a later re-sweep can reuse it):
      weights: 0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0
      mixes: equal 1/3s; 0.2/0.4/0.4 (less continuity); 0/0.5/0.5 (no continuity);
             0.5/0.25/0.25 (continuity-heavy).
    """
    import pandas as pd
    try:
        import nfl_data_py as nfl
    except Exception:
        nfl = None
    from data_building.oline_talent_prior import (
        load_talent_inputs, compute_oline_talent_prior,
    )

    weights = weights if weights is not None else [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]
    mixes = mixes if mixes is not None else [
        {"name": "equal", "continuity": 1 / 3, "draft": 1 / 3, "veteran": 1 / 3},
        {"name": "less_continuity", "continuity": 0.2, "draft": 0.4, "veteran": 0.4},
        {"name": "draft_vet_only", "continuity": 0.0, "draft": 0.5, "veteran": 0.5},
        {"name": "continuity_heavy", "continuity": 0.5, "draft": 0.25, "veteran": 0.25},
    ]

    pbp_cache = {}

    def load_year(year):
        if year not in pbp_cache:
            pbp_cache[year] = _load_pbp_year(year, pd, nfl)
        return pbp_cache[year]

    seasons_data = []
    for season in seasons:
        raw = _prep_season(load_year(season), pd, season)
        if raw is None or raw.empty:
            print(f"[oline_backtest] talent sweep {season}: no current pbp")
            continue
        prior_raw = _prep_season(load_year(season - 1), pd, season - 1)
        early, rest = _early_rest_frames(raw, pd, early_week)
        if early.empty or rest.empty:
            print(f"[oline_backtest] talent sweep {season}: not enough weeks around {early_week}")
            continue
        run_ly_adj, run_su_adj, run_n, run_league, _ = _season_run_metrics(early)
        run_ly_prior, run_su_prior = {}, {}
        if prior_raw is not None and not prior_raw.empty:
            run_ly_prior, run_su_prior, _, _, _ = _season_run_metrics(prior_raw)
        cur_pass = _season_pass_metrics(early)
        prior_pass = (
            _season_pass_metrics(prior_raw)
            if prior_raw is not None and not prior_raw.empty else None
        )
        su_league = (sum(run_su_adj.values()) / len(run_su_adj)) if run_su_adj else 0.0
        rest_pass = _season_pass_metrics(rest)["detail"]
        _, _, _, _, rest_run = _season_run_metrics(rest)

        inputs = load_talent_inputs(season, pd, nfl)
        talent_by_mix = {}
        for mix in mixes:
            pack = compute_oline_talent_prior(
                season, pd, nfl,
                w_continuity=mix["continuity"],
                w_draft=mix["draft"],
                w_veteran=mix["veteran"],
                inputs=inputs,
            )
            talent_by_mix[mix["name"]] = pack
        seasons_data.append({
            "season": season,
            "run_ly_adj": run_ly_adj, "run_su_adj": run_su_adj, "run_n": run_n,
            "run_league": run_league, "su_league": su_league,
            "run_ly_prior": run_ly_prior, "run_su_prior": run_su_prior,
            "cur_pass": cur_pass, "prior_pass": prior_pass,
            "rest_pass": rest_pass, "rest_run": rest_run,
            "talent_by_mix": talent_by_mix,
        })
        n_talent = sum(1 for p in talent_by_mix.values() if p)
        print(f"[oline_backtest] talent sweep {season}: early/rest ready, "
              f"talent_mixes_with_data={n_talent}/{len(mixes)}")

    if not seasons_data:
        print("[oline_backtest] talent sweep: no usable seasons")
        return {}

    results = []
    for mix in mixes:
        for w in weights:
            # Pool pairs across seasons (32 teams x N seasons), same pattern
            # as sweep_pressure_weight.
            all_p, all_s, all_ly, all_su = [], [], [], []
            for sd in seasons_data:
                pack = sd["talent_by_mix"].get(mix["name"])
                scores = (pack or {}).get("scores") if w else None
                if w and scores:
                    ly_p, su_p, pass_p = _apply_talent_to_priors(
                        sd["run_ly_prior"], sd["run_su_prior"], sd["prior_pass"],
                        scores, w, sd["run_league"], sd["su_league"], sd["cur_pass"])
                else:
                    ly_p, su_p, pass_p = sd["run_ly_prior"], sd["run_su_prior"], sd["prior_pass"]
                g = grade_indices(
                    sd["run_ly_adj"], sd["run_n"], ly_p, sd["run_league"],
                    sd["run_su_adj"], su_p, sd["cur_pass"], pass_p, round_to=None)
                blocks = _blocks_from_indices(g)
                for t, b in blocks.items():
                    pb, rb = b.get("pass_block"), b.get("run_block")
                    o, r = sd["rest_pass"].get(t, {}), sd["rest_run"].get(t, {})
                    if pb is not None and o.get("pressure_rate") is not None:
                        all_p.append((pb, o["pressure_rate"]))
                    if pb is not None and o.get("sack_rate") is not None:
                        all_s.append((pb, o["sack_rate"]))
                    if rb is not None and r.get("line_yards") is not None:
                        all_ly.append((rb, r["line_yards"]))
                    if rb is not None and r.get("success_rate") is not None:
                        all_su.append((rb, r["success_rate"]))
            rho_p, n_p = _spearman(all_p)
            rho_s, n_s = _spearman(all_s)
            rho_ly, n_ly = _spearman(all_ly)
            rho_su, n_su = _spearman(all_su)
            aligned = []
            if rho_p is not None:
                aligned.append(-rho_p)
            if rho_s is not None:
                aligned.append(-rho_s)
            if rho_ly is not None:
                aligned.append(rho_ly)
            if rho_su is not None:
                aligned.append(rho_su)
            score = (sum(aligned) / len(aligned)) if aligned else None
            results.append({
                "mix": mix["name"],
                "w_continuity": mix["continuity"],
                "w_draft": mix["draft"],
                "w_veteran": mix["veteran"],
                "talent_w": w,
                "rho_future_pressure": None if rho_p is None else round(rho_p, 3),
                "rho_future_sack": None if rho_s is None else round(rho_s, 3),
                "rho_future_line_yards": None if rho_ly is None else round(rho_ly, 3),
                "rho_future_success": None if rho_su is None else round(rho_su, 3),
                "n": n_p,
                "score": None if score is None else round(score, 4),
            })

    results = [r for r in results if r["score"] is not None]
    results.sort(key=lambda r: r["score"], reverse=True)
    baseline = [r for r in results if r["talent_w"] == 0]
    baseline_score = baseline[0]["score"] if baseline else None
    print(f"\n[oline_backtest] talent-prior sweep "
          f"(seasons={list(seasons)}, early_week={early_week})")
    print(f"  {'mix':<18} {'w':>5} {'rho_pr':>8} {'rho_sk':>8} "
          f"{'rho_ly':>8} {'rho_su':>8} {'score':>8}")
    for r in results:
        star = ""
        if r is results[0]:
            star = "  <- best"
        if r["talent_w"] == 0 and r["mix"] == mixes[0]["name"]:
            star += "  (baseline last-season prior)"
        print(f"  {r['mix']:<18} {r['talent_w']:>5} "
              f"{r['rho_future_pressure']!s:>8} {r['rho_future_sack']!s:>8} "
              f"{r['rho_future_line_yards']!s:>8} {r['rho_future_success']!s:>8} "
              f"{r['score']:>8}{star}")
    best = results[0] if results else {}
    beats = (
        best.get("score") is not None
        and baseline_score is not None
        and best.get("talent_w", 0) > 0
        and best["score"] > baseline_score
    )
    print(f"[oline_backtest] baseline (w=0) score={baseline_score}")
    print(f"[oline_backtest] best={best.get('mix')} w={best.get('talent_w')} "
          f"score={best.get('score')} beats_baseline={beats}")
    print(f"[oline_backtest] shipped constants: w={TALENT_PRIOR_W} "
          f"continuity={TALENT_W_CONTINUITY} draft={TALENT_W_DRAFT} "
          f"veteran={TALENT_W_VETERAN}")
    if not beats:
        print("[oline_backtest] talent prior did NOT beat last-season prior; "
              "leave use_talent_prior default off")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seasons": list(seasons),
        "early_week": early_week,
        "grid": results,
        "best": best,
        "baseline_score": baseline_score,
        "beats_baseline": beats,
    }
    if save and results:
        path = os.path.join(str(CACHE_DIR), "oline_talent_prior_sweep.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[oline_backtest] talent sweep report -> {path}")
    return report


def run_talent_prior_backtest(seasons, early_week=4, save=True):
    """Convenience: sweep, then evaluate the tuned constants vs w=0."""
    return sweep_talent_prior(seasons, early_week=early_week, save=save)


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
        _, _, _, _, run2 = _season_run_metrics(second)
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
        _, _, _, _, run1 = _season_run_metrics(first)
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
    raw = sys.argv[1:]
    flags = {a for a in raw if a.startswith("-")}
    args = [int(a) for a in raw if a.isdigit()]
    seasons = args or [2022, 2023, 2024, 2025]
    if "--talent-prior" in flags or "--sweep-talent" in flags:
        sweep_talent_prior(seasons)
    else:
        run_backtest(seasons)
