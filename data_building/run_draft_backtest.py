"""Run the pick-score weight backtest against real Sleeper leagues.

This is the "real call" for data_building/draft_grade_backtest.py: it wires a
DB-backed ``value_fn`` from the app's own enriched player pool
(``_build_league_players_payload`` — the exact value/ADP/PPG/tier source the live
draft-grader scores against), pulls each league's completed draft + final
standings, and reports how well the shipped weights predict real success plus a
sweep of candidate weight tables.

It needs what the app needs: importable ``app`` (Flask), a reachable
``DATABASE_URL``, and Sleeper network access. Offline it will simply find no
valuations / no drafts and report an empty sample — it never fabricates data.

Usage:
    python -m data_building.run_draft_backtest \
        --league 123456789012345678 987654321098765432 \
        --season 2024 [--sf] [--draft-type startup] [--method spearman]

    # Sweep the shipped table's single-lever nudges (default) or a custom grid
    # by editing WEIGHT_CANDIDATES in draft_grade_backtest.py.
"""
from __future__ import annotations

import argparse
import sys
from typing import Optional

from utils.pick_score import ps_tier_of
from utils.tier_thresholds import compute_tier_thresholds
from data_building.draft_grade_backtest import (
    WEIGHT_CANDIDATES,
    correlate_grades_to_finish,
    load_sleeper_samples,
    sweep,
)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def build_value_fn(draft_type: str, is_sf: bool, num_teams: int):
    """Build a ``value_fn(pick) -> pick-score inputs`` from the live player pool.

    Mirrors the field selection the draft-grader uses (_lp_adp/_lp_ppg): value /
    sf_value, proj_ppg||ppg, per-draft-type ADP, age, rank_change_7d. Derives the
    per-pick VOR (value over positional replacement), tier (drop-based
    thresholds) and ppg_norm (replacement->0, elite->1) from the pool itself, so
    the inputs match what /api/draft-grades feeds compute_pick_score.
    """
    # Import inside the function so `--help` works without booting Flask/DB.
    try:
        from app import _build_league_players_payload  # noqa: WPS433
        payload = _build_league_players_payload(kdef=False) or {}
    except Exception as e:  # no DB/app offline -> empty pool, value_fn returns None
        print(f"[warn] could not build player pool ({e}); no valuations available.",
              file=sys.stderr)
        payload = {}
    players = payload.get("players") or []
    pool = {str(p.get("id")): p for p in players if p.get("id") is not None}

    val_key = "sf_value" if is_sf else "value"

    def _val(d) -> float:
        return float(d.get(val_key) or d.get("value") or 0)

    def _adp(d) -> Optional[float]:
        if draft_type == "rookie":
            a = d.get("sf_rookie_avg_pick") if is_sf else d.get("rookie_avg_pick")
        elif draft_type == "redraft":
            a = d.get("sf_redraft_avg_pick") if is_sf else d.get("redraft_avg_pick")
        else:
            a = d.get("sf_avg_pick") if is_sf else d.get("avg_pick")
        try:
            return float(a) if a is not None else None
        except (TypeError, ValueError):
            return None

    def _ppg(d) -> Optional[float]:
        v = d.get("proj_ppg")
        if v is None:
            v = d.get("ppg")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # Effective starters per position anchor the replacement index (SF splits
    # QB, FLEX splits RB/WR) — matching the grader's _ps_starter_counts intent.
    starters = ({"QB": 1.5, "RB": 2.5, "WR": 3.0, "TE": 1.0} if is_sf
                else {"QB": 1.0, "RB": 2.5, "WR": 3.0, "TE": 1.0})

    by_pos_val: dict[str, list] = {"QB": [], "RB": [], "WR": [], "TE": []}
    by_pos_ppg: dict[str, list] = {"QB": [], "RB": [], "WR": [], "TE": []}
    for d in pool.values():
        pos = str(d.get("position") or "").upper()
        if pos in by_pos_val:
            by_pos_val[pos].append(_val(d))
            pv = _ppg(d)
            if pv is not None and pv > 0:
                by_pos_ppg[pos].append(pv)

    repl_val: dict[str, float] = {}
    ppg_scale: dict[str, dict] = {}
    for pos in by_pos_val:
        arr = sorted(by_pos_val[pos], reverse=True)
        if arr:
            idx = max(0, min(int(round(num_teams * starters.get(pos, 1))) - 1, len(arr) - 1))
            repl_val[pos] = arr[idx]
        parr = sorted(by_pos_ppg[pos], reverse=True)
        if parr:
            idx = max(0, min(int(round(num_teams * starters.get(pos, 1))) - 1, len(parr) - 1))
            topn = max(1, min(3, len(parr)))
            ppg_scale[pos] = {"repl": parr[idx], "elite": sum(parr[:topn]) / topn}

    max_val = max((_val(d) for d in pool.values()), default=0.0) or 1.0
    lt = "sf" if is_sf else "1qb"
    thresholds = compute_tier_thresholds(
        [{"position": d.get("position"), "value": _val(d),
          "sf_value": float(d.get("sf_value") or _val(d))} for d in pool.values()],
        league_type=lt, league_size=num_teams,
    )

    def value_fn(pick) -> Optional[dict]:
        pid = str(pick.get("player_id") or "")
        d = pool.get(pid)
        if not d:
            return None
        pos = (pick.get("position") or str(d.get("position") or "")).upper()
        if pos not in by_pos_val:
            return None
        value = _val(d)
        vor = max(0.0, value - repl_val.get(pos, 0.0))
        ppg_norm = None
        pv = _ppg(d)
        sc = ppg_scale.get(pos)
        if pv is not None and sc and sc["elite"] > sc["repl"]:
            ppg_norm = _clamp01((pv - sc["repl"]) / (sc["elite"] - sc["repl"]))
        return {
            "value": value, "vor": vor, "tier": ps_tier_of(value, thresholds),
            "age": d.get("age"),
            "rank_change_7d": d.get("rank_change_7d"),
            "avg_pick": _adp(d), "max_val": max_val, "ppg_norm": ppg_norm,
        }

    return value_fn


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", nargs="+", required=True, metavar="LEAGUE_ID",
                    help="One or more Sleeper league IDs (completed seasons).")
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--sf", action="store_true", help="Superflex valuation.")
    ap.add_argument("--draft-type", default="startup", choices=["startup", "redraft", "rookie"])
    ap.add_argument("--num-teams", type=int, default=12)
    ap.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    args = ap.parse_args(argv)

    value_fn = build_value_fn(args.draft_type, args.sf, args.num_teams)
    samples = load_sleeper_samples(
        args.league, args.season, value_fn=value_fn,
        draft_type=args.draft_type, is_sf=args.sf, num_teams=args.num_teams,
    )
    if not samples:
        print("No gradeable teams found — check league IDs, network, and DATABASE_URL.")
        return 1

    n_picks = sum(len(s.picks) for s in samples)
    print(f"Loaded {len(samples)} teams ({n_picks} graded picks) "
          f"across {len(set(s.meta.get('league_id') for s in samples))} leagues.\n")

    base_r = correlate_grades_to_finish(samples, method=args.method)
    print(f"Shipped weights: {args.method} r = "
          f"{'n/a' if base_r is None else f'{base_r:+.3f}'} "
          f"(grade vs season points-for; higher = grades track success)\n")

    print("Candidate sweep (best predictor first):")
    for label, _w, r in sweep(samples, WEIGHT_CANDIDATES, method=args.method):
        print(f"  {'   n/a' if r is None else f'{r:+.3f}'}  {label}")
    print("\nA nudge that beats 'base' by a clear margin is a real signal to fold "
          "into PS_WEIGHTS (and its JS mirror). One league-season is noisy — run "
          "several before changing weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
