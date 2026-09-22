"""Week-1 replay with real prior-season baselines.

This deliberately complements the Week 4-14 walk-forward backtest: it audits
the production failure mode where provider-default rows, true rookies, and
unmatched veterans were previously conflated. It reads committed caches only.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from data_building.breakout_engine.backtest_weekly_breakout import load_cached_season_series
from data_building.breakout_engine.weekly_breakout import (
    score_player, WATCHLIST_MIN_SCORE, INITIAL_ROLE_DISCOVERY_MIN,
)
from data_building.breakout_engine.weekly_runner import _prior_baseline_map


def _pct(values: List[float], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[int(fraction * (len(ordered) - 1))], 1)


def run_week1_replay(season: int) -> Dict[str, Any]:
    series, meta = load_cached_season_series(season)
    prior = _prior_baseline_map(season)
    from utils.utils import load_players_index
    player_index = load_players_index() or {}
    groups: Dict[str, List[Dict[str, Any]]] = {
        "established_valid_prior": [], "true_rookie_no_history": [],
        "veteran_missing_match": [], "matched_unusable_prior": [],
        "partially_valid_prior": [],
    }
    for pid, rows in series.items():
        week1 = [row for row in rows if int(row["week"]) == 1]
        if not week1:
            continue
        pm = player_index.get(str(pid)) or {}
        baseline = prior.get(str(pid))
        status = (baseline or {}).get("history_status")
        draft_year = pm.get("draft_year") or pm.get("draft_yr")
        rookie = str(draft_year or "") == str(season)
        if status == "usable":
            group = "established_valid_prior"
        elif status == "partial":
            group = "partially_valid_prior"
        elif baseline:
            group = "matched_unusable_prior"
        elif rookie:
            group = "true_rookie_no_history"
        else:
            group = "veteran_missing_match"
        player = {"player_id": pid, "player_name": pm.get("name"),
                  "position": meta.get(pid, {}).get("position"),
                  "team": meta.get(pid, {}).get("team"), "season": season,
                  "draft_year": draft_year, "draft_round": pm.get("draft_round"),
                  "years_exp": 0 if rookie else pm.get("years_exp")}
        scored = score_player(player, week1, prior_baseline=baseline, cutoff_week=1)
        groups[group].append(scored)

    report = {"season": season, "week": 1, "watchlist_min": WATCHLIST_MIN_SCORE,
              "groups": {}}
    for name, rows in groups.items():
        scores = [float(row["breakout_score"]) for row in rows]
        surfaced = [row for row in rows if row["breakout_score"] >= WATCHLIST_MIN_SCORE
                    and (row["score_basis"] != "initial_role" or
                         row["current_role_score"] >= INITIAL_ROLE_DISCOVERY_MIN)]
        report["groups"][name] = {
            "evaluated": len(rows), "surfaced": len(surfaced),
            "baseline_coverage_pct": round(100 * sum(r["baseline_source"] != "none" for r in rows)
                                           / len(rows), 1) if rows else None,
            "median": _pct(scores, .50), "p90": _pct(scores, .90),
            "p95": _pct(scores, .95), "max": max(scores) if scores else None,
            "exact_100": sum(score == 100 for score in scores),
            "exact_100_pct": round(100 * sum(score == 100 for score in scores) / len(scores), 2)
            if scores else 0.0,
            "top_candidates": [{"player_id": row["player_id"], "name": row["player_name"],
                                "position": row["position"], "score": row["breakout_score"],
                                "basis": row["score_basis"],
                                "support": row["supporting_signals"],
                                "reason": (row["reasons"] or [None])[0]}
                               for row in sorted(rows, key=lambda r: r["ranking_score"], reverse=True)[:5]],
            "obvious_false_positives": [row["player_name"] or row["player_id"] for row in surfaced
                                        if row["role_change_score"] is not None
                                        and row["role_change_score"] < WATCHLIST_MIN_SCORE],
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run_week1_replay(args.season)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
