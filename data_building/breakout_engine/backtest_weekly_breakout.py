"""
Chronological backtest for the weekly breakout engine.

Walks a season week by week. At each cutoff week W it scores every candidate on
data available THROUGH W only (strict as-of, via score_player), takes the top-N
flagged candidates, then looks ahead 2-4 games to ask two questions:

    1. Did the flagged player SUSTAIN the increased usage? (kept at least half the
       usage gain over the lookahead window)
    2. Did fantasy usefulness IMPROVE? (lookahead PPG >= the pre-flag baseline PPG)

A pick "hits" when both hold. The model is compared against two deliberately
simple baselines, so we can see whether the engine adds anything over them:

    - recent-points: rank by last-2-game PPG (chasing box scores)
    - usage-growth-only: rank by raw recent-vs-baseline usage delta, ignoring the
      resulting level, classification, confidence, and the fantasy-spike guard

Reported per method: two precisions among the top-N - precision(usage) for
sustaining the usage rise, and precision(useful) for the flagged player actually
reaching a startable PPR PPG over the lookahead - plus coverage of the week's
true sustained risers, false-positive rate, and average lead time (games between
the flag and the player's first elevated-production game). The two precisions
matter because a raw usage-delta ranker optimizes precision(usage) almost by
construction (the ground truth IS a usage rise); precision(useful) is the read
that tracks product value, and is where workload-awareness and the spike guard
earn their place.

The evaluation core (``run_backtest``) is pure - it takes an in-memory
{player_id: [weekly rows]} map - so it is unit tested without a database. The DB
loader and CLI sit on top.

IMPORTANT on honesty: this measures only what the available historical
player_weekly_metrics support. It does not tune thresholds to hit a target
candidate count, and it keeps the tuning window (``--tune-through``) separate
from the evaluation window when one is supplied. Do not quote numbers this has
not actually produced on real data.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from data_building.breakout_engine.weekly_breakout import (
    score_player, split_windows, _mean_present, WATCHLIST_MIN_SCORE,
)


def _key_stat(position: str) -> str:
    return "snap_pct" if position == "QB" else (
        "target_share" if position in ("WR", "TE") else "snaps")


def _window_avgs(rows: List[Dict], position: str) -> Tuple[Optional[float], Optional[float]]:
    """(baseline_avg, recent_avg) of the key usage stat over the non-overlapping
    windows for these active rows."""
    active = [r for r in rows if _positive_usage(r)]
    recent, baseline = split_windows(active)
    key = _key_stat(position)
    r_avg, _ = _mean_present(recent, key)
    b_avg, _ = _mean_present(baseline, key)
    return b_avg, r_avg


def _positive_usage(r: Dict) -> bool:
    for k in ("snaps", "targets", "carries", "pass_att"):
        try:
            if float(r.get(k) or 0) > 0:
                return True
        except (TypeError, ValueError):
            continue
    # snap_pct present but no raw counts still counts as an active game
    return r.get("snap_pct") is not None


def _future_active(rows: List[Dict], cutoff_week: int, horizon: int) -> List[Dict]:
    fut = [r for r in rows if cutoff_week < int(r["week"]) <= cutoff_week + horizon]
    return [r for r in fut if _positive_usage(r)]


# A "true" outcome is a REAL usage increase that stayed elevated and turned into
# improved fantasy usefulness. Ratios (not absolute deltas) so the same rule
# applies to snap %, target share, and snap counts without a unit mismatch.
_INCREASE_RATIO = 1.25      # recent usage must be >=25% above baseline to count as a rise
_SUSTAIN_FRACTION = 0.5     # keep >=half the rise over the lookahead
_USEFULNESS_RATIO = 1.15    # lookahead PPG >=15% above the pre-flag baseline PPG
_ABS_USAGE_FLOOR = 8.0      # min key-stat level when baseline is ~0 (rookie/debut)

# Absolute fantasy-relevance (roughly startable-flex PPR PPG) per position. Used
# for the SECONDARY outcome metric: did a flagged player actually become
# fantasy-useful over the lookahead, regardless of usage mechanics? This matches
# product value more directly than the usage-sustain metric, which a raw
# usage-delta ranker optimizes by construction.
_USEFUL_PPG = {"QB": 16.0, "RB": 12.0, "WR": 11.0, "TE": 9.0}


def _became_useful(rows: List[Dict], position: str, cutoff_week: int, horizon: int) -> Optional[bool]:
    """Did the player average a fantasy-relevant PPR PPG over the lookahead?
    None when there is no lookahead data."""
    fut = _future_active(rows, cutoff_week, horizon)
    if not fut:
        return None
    fut_ppg, _ = _mean_present(fut, "ppr_pts")
    if fut_ppg is None:
        return False
    return fut_ppg >= _USEFUL_PPG.get((position or "").upper(), 11.0)


def _did_sustain(rows: List[Dict], position: str, cutoff_week: int, horizon: int) -> Optional[bool]:
    """Ground-truth outcome for one player at week W.

    True only when (a) there was a genuine usage INCREASE by week W (recent window
    meaningfully above baseline), (b) the player kept at least half of that
    increase over the next `horizon` games, and (c) fantasy usefulness improved
    over the pre-flag baseline. A flat player or a one-week TD spike is therefore
    NOT a hit - which is the whole point of separating this engine from
    box-score chasing. None when there is no lookahead data to judge.
    """
    fut = _future_active(rows, cutoff_week, horizon)
    if not fut:
        return None
    through = [r for r in rows if int(r["week"]) <= cutoff_week]
    b_avg, r_avg = _window_avgs(through, position)
    key = _key_stat(position)
    fut_avg, _ = _mean_present(fut, key)
    if fut_avg is None or r_avg is None:
        return None
    base = b_avg if b_avg is not None else 0.0

    # (a) was there a real increase?
    if base > 1e-6:
        increased = r_avg >= base * _INCREASE_RATIO
    else:
        increased = r_avg >= _ABS_USAGE_FLOOR
    if not increased:
        return False

    # (b) sustained: kept >= half the rise
    gain = r_avg - base
    retained_ok = fut_avg >= base + _SUSTAIN_FRACTION * gain

    # (c) usefulness improved vs the baseline window's PPG
    recent, baseline = split_windows([r for r in through if _positive_usage(r)])
    base_ppg, _ = _mean_present(baseline, "ppr_pts")
    fut_ppg, _ = _mean_present(fut, "ppr_pts")
    if fut_ppg is None:
        return False
    if base_ppg and base_ppg > 1e-6:
        useful_ok = fut_ppg >= base_ppg * _USEFULNESS_RATIO
    else:
        useful_ok = fut_ppg > 0

    return bool(retained_ok and useful_ok)


def _lead_time(rows: List[Dict], cutoff_week: int, horizon: int) -> Optional[int]:
    """Games from the flag week to the player's first elevated-production game in
    the lookahead (PPG above his pre-flag recent PPG). None if never."""
    through_active = [r for r in rows if int(r["week"]) <= cutoff_week and _positive_usage(r)]
    recent, _ = split_windows(through_active)
    ref_ppg, _ = _mean_present(recent, "ppr_pts")
    ref = ref_ppg if ref_ppg is not None else 0.0
    fut = sorted(_future_active(rows, cutoff_week, horizon), key=lambda r: int(r["week"]))
    for i, r in enumerate(fut, 1):
        try:
            if float(r.get("ppr_pts") or 0) > ref:
                return i
        except (TypeError, ValueError):
            continue
    return None


def _retained_rest_of_season(rows: List[Dict], position: str,
                             cutoff_week: int) -> Optional[bool]:
    """Did at least half of the detected role gain persist over all later games?"""
    future = [row for row in rows if int(row["week"]) > cutoff_week and _positive_usage(row)]
    if not future:
        return None
    through = [row for row in rows if int(row["week"]) <= cutoff_week and _positive_usage(row)]
    baseline, recent = _window_avgs(through, position)
    key = _key_stat(position)
    future_avg, _ = _mean_present(future, key)
    if recent is None or future_avg is None:
        return None
    if baseline is None:
        return future_avg >= recent * .65
    gain = recent - baseline
    if gain <= 0:
        return False
    return future_avg >= baseline + .5 * gain


# =============================================================================
# baseline pickers
# =============================================================================

def _recent_ppg(rows: List[Dict], cutoff_week: int) -> float:
    through = [r for r in rows if int(r["week"]) <= cutoff_week and _positive_usage(r)]
    recent, _ = split_windows(through)
    v, _ = _mean_present(recent, "ppr_pts")
    return v or 0.0


def _usage_delta(rows: List[Dict], position: str, cutoff_week: int) -> float:
    through = [r for r in rows if int(r["week"]) <= cutoff_week]
    b, r = _window_avgs(through, position)
    if r is None:
        return 0.0
    return r - (b or 0.0)


# =============================================================================
# pure backtest core
# =============================================================================

def run_backtest(
    series_by_player: Dict[str, List[Dict]],
    meta_by_player: Dict[str, Dict[str, Any]],
    *,
    eval_weeks: List[int],
    top_n: int = 15,
    horizon: int = 3,
) -> Dict[str, Any]:
    """Evaluate model vs baselines across ``eval_weeks``. Pure. See module doc."""
    methods = ("model", "recent_points", "usage_growth")
    agg = {m: {"picks": 0, "hits": 0, "useful_picks": 0, "useful_hits": 0,
               "ros_picks": 0, "ros_hits": 0, "lead_times": []} for m in methods}
    coverage_num = {m: 0 for m in methods}
    coverage_den = 0

    for W in eval_weeks:
        # Universe: players with at least one active game through W.
        universe: List[str] = []
        model_scored: List[Tuple[str, float]] = []
        recent_scored: List[Tuple[str, float]] = []
        usage_scored: List[Tuple[str, float]] = []
        sustain_by_pid: Dict[str, Optional[bool]] = {}
        useful_by_pid: Dict[str, Optional[bool]] = {}
        ros_by_pid: Dict[str, Optional[bool]] = {}

        for pid, rows in series_by_player.items():
            meta = meta_by_player.get(pid, {})
            pos = (meta.get("position") or "").upper()
            through = [r for r in rows if int(r["week"]) <= W and _positive_usage(r)]
            if not through:
                continue
            universe.append(pid)
            sustain_by_pid[pid] = _did_sustain(rows, pos, W, horizon)
            useful_by_pid[pid] = _became_useful(rows, pos, W, horizon)
            ros_by_pid[pid] = _retained_rest_of_season(rows, pos, W)

            res = score_player(
                {"player_id": pid, "position": pos, "team": meta.get("team")},
                rows, cutoff_week=W,
            )
            if res["classification"] in ("emerging_breakout", "temporary_opportunity"):
                # Selected ranking formula: magnitude × (0.75 + 0.25 confidence).
                # Confidence can move ordering by at most 25%; it cannot turn a
                # weak signal into a candidate because classification happens
                # first on the unadjusted final score and role-quality gates.
                model_scored.append((pid, res.get("ranking_score", res["breakout_score"])))
            recent_scored.append((pid, _recent_ppg(rows, W)))
            usage_scored.append((pid, _usage_delta(rows, pos, W)))

        # Ground truth for coverage: players who actually sustained (had lookahead
        # data and hit) this week.
        true_risers = {pid for pid, s in sustain_by_pid.items() if s is True}
        coverage_den += len(true_risers)

        picks = {
            "model": [p for p, _ in sorted(model_scored, key=lambda t: t[1], reverse=True)[:top_n]],
            "recent_points": [p for p, _ in sorted(recent_scored, key=lambda t: t[1], reverse=True)[:top_n]],
            "usage_growth": [p for p, _ in sorted(usage_scored, key=lambda t: t[1], reverse=True)[:top_n]],
        }
        for m, plist in picks.items():
            for pid in plist:
                s = sustain_by_pid.get(pid)
                if s is not None:  # scorable on the usage-sustain metric
                    agg[m]["picks"] += 1
                    if s:
                        agg[m]["hits"] += 1
                        lt = _lead_time(series_by_player[pid], W, horizon)
                        if lt is not None:
                            agg[m]["lead_times"].append(lt)
                u = useful_by_pid.get(pid)
                if u is not None:  # scorable on the fantasy-usefulness metric
                    agg[m]["useful_picks"] += 1
                    if u:
                        agg[m]["useful_hits"] += 1
                ros = ros_by_pid.get(pid)
                if ros is not None:
                    agg[m]["ros_picks"] += 1
                    if ros:
                        agg[m]["ros_hits"] += 1
            coverage_num[m] += len(set(plist) & true_risers)

    report = {"eval_weeks": eval_weeks, "top_n": top_n, "horizon": horizon, "methods": {}}
    for m in methods:
        picks = agg[m]["picks"]
        hits = agg[m]["hits"]
        lts = agg[m]["lead_times"]
        u_picks = agg[m]["useful_picks"]
        u_hits = agg[m]["useful_hits"]
        ros_picks = agg[m]["ros_picks"]
        ros_hits = agg[m]["ros_hits"]
        ordered_lts = sorted(lts)
        report["methods"][m] = {
            "picks_evaluated": picks,
            "hits": hits,
            "precision": round(hits / picks, 3) if picks else None,
            "precision_useful": round(u_hits / u_picks, 3) if u_picks else None,
            "rest_of_season_role_retention": round(ros_hits / ros_picks, 3) if ros_picks else None,
            "false_positive_rate": round((picks - hits) / picks, 3) if picks else None,
            "coverage": round(coverage_num[m] / coverage_den, 3) if coverage_den else None,
            "avg_lead_time_games": round(sum(lts) / len(lts), 2) if lts else None,
            "median_lead_time_games": ordered_lts[len(ordered_lts) // 2] if ordered_lts else None,
        }
    return report


_BUCKETS = ((0, 49), (50, 59), (60, 69), (70, 79), (80, 89), (90, 100))


def build_evaluation_records(series_by_player, meta_by_player, *, season,
                             eval_weeks, horizons=(1, 3, 6)) -> List[Dict[str, Any]]:
    """Leakage-safe current-version rows suitable for calibration and replay."""
    records = []
    for week in eval_weeks:
        for pid, rows in series_by_player.items():
            meta = meta_by_player.get(pid, {})
            pos = str(meta.get("position") or "").upper()
            through = [r for r in rows if int(r["week"]) <= week and _positive_usage(r)]
            if len(through) < 2:
                continue
            scored = score_player({"player_id": pid, "position": pos,
                                   "team": meta.get("team")}, rows, cutoff_week=week)
            current = through[-1]
            row = {
                "season": int(season), "week": int(week), "player_id": pid,
                "position": pos, "team": meta.get("team"),
                "score": scored["breakout_score"], "components": scored.get("components") or {},
                "signals": scored.get("signals") or {},
                "inputs": {k: current.get(k) for k in (
                    "snap_pct", "routes", "targets", "carries", "red_zone_opportunities", "ppr_pts")},
                "baselines": {"recent_points": _recent_ppg(rows, week),
                              "usage_growth": _usage_delta(rows, pos, week)},
                "outcomes": {},
            }
            for horizon in horizons:
                future = _future_active(rows, week, horizon)
                future_ppg, _ = _mean_present(future, "ppr_pts")
                row["outcomes"][str(horizon)] = {
                    "role_persisted": _did_sustain(rows, pos, week, horizon),
                    "fantasy_hit": _became_useful(rows, pos, week, horizon),
                    "fantasy_ppg": round(future_ppg, 2) if future_ppg is not None else None,
                }
            records.append(row)
    return records


def calibration_report(records: List[Dict[str, Any]], horizon: int = 3) -> dict:
    """Score-bucket and position results; rates stay None for empty buckets."""
    def summarize(rows):
        outcomes = [r["outcomes"].get(str(horizon), {}) for r in rows]
        scorable_role = [o["role_persisted"] for o in outcomes if o.get("role_persisted") is not None]
        scorable_hit = [o["fantasy_hit"] for o in outcomes if o.get("fantasy_hit") is not None]
        ppg = [o["fantasy_ppg"] for o in outcomes if o.get("fantasy_ppg") is not None]
        return {"sample_size": len(rows),
                "role_persistence_rate": round(sum(scorable_role) / len(scorable_role), 3) if scorable_role else None,
                "fantasy_hit_rate": round(sum(scorable_hit) / len(scorable_hit), 3) if scorable_hit else None,
                "average_future_ppg": round(sum(ppg) / len(ppg), 2) if ppg else None}
    buckets = {}
    for lo, hi in _BUCKETS:
        buckets[f"{lo}-{hi}" if hi < 100 else "90+"] = summarize(
            [r for r in records if lo <= float(r.get("score") or 0) <= hi])
    positions = {p: summarize([r for r in records if r.get("position") == p])
                 for p in ("QB", "RB", "WR", "TE")}
    coverage = {}
    for signal in ("routes_pg", "high_value_opportunities_pg"):
        coverage[signal] = round(sum(bool((r.get("signals") or {}).get(signal, {}).get("available"))
                                     for r in records) / len(records), 3) if records else None
    return {"horizon": horizon, "sample_size": len(records), "buckets": buckets,
            "positions": positions, "optional_signal_coverage": coverage}


# =============================================================================
# DB loader + CLI
# =============================================================================

def load_season_series(season: int) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """Load {pid: [weekly rows]} and {pid: meta} for a season from
    player_weekly_metrics. DB-backed; used only by the CLI."""
    from dashboard_services.db import get_conn
    from data_building.weekly_metrics import init_weekly_metrics_db
    init_weekly_metrics_db()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT player_id, season, week, position, snap_pct, snaps, team_snaps,
                   targets, receptions, carries, touches, target_share, ppr_pts, pass_att
            FROM player_weekly_metrics WHERE season = %s ORDER BY player_id, week
            """,
            (int(season),),
        ).fetchall()
    series: Dict[str, List[Dict]] = {}
    meta: Dict[str, Dict] = {}
    for r in rows:
        d = dict(r)
        pid = str(d["player_id"])
        series.setdefault(pid, []).append(d)
        meta.setdefault(pid, {"position": d.get("position"), "team": None})
    return series, meta


def load_cached_season_series(season: int) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict]]:
    """Reproduce the supported weekly subset from committed Sleeper caches.

    This makes measurement runnable in CI/development without production DB
    credentials. It intentionally reports routes/red-zone usage as unavailable.
    """
    from utils.utils import load_players_index
    players = load_players_index() or {}
    series, meta = {}, {}
    pattern = os.path.join("cache", "sleeper_stats", f"sleeper_stats_s{int(season)}_w*.json")
    for path in sorted(glob.glob(pattern), key=lambda p: int(p.rsplit("_w", 1)[1].split(".")[0])):
        week = int(path.rsplit("_w", 1)[1].split(".")[0])
        with open(path, encoding="utf-8") as fh:
            stats = json.load(fh) or {}
        team_targets = {}
        for pid, st in stats.items():
            pm = players.get(str(pid)) or {}
            team = pm.get("team")
            if team:
                team_targets[team] = team_targets.get(team, 0.0) + float(st.get("rec_tgt") or 0)
        for pid, st in stats.items():
            pm = players.get(str(pid)) or {}
            pos = str(pm.get("pos") or "").upper()
            if pos not in _USEFUL_PPG:
                continue
            snaps, team_snaps = float(st.get("off_snp") or 0), float(st.get("tm_off_snp") or 0)
            targets, carries, pass_att = (float(st.get("rec_tgt") or 0),
                                           float(st.get("rush_att") or 0),
                                           float(st.get("pass_att") or 0))
            if not any((snaps, targets, carries, pass_att)):
                continue
            team = pm.get("team")
            row = {"season": int(season), "week": week, "position": pos,
                   "snaps": snaps, "team_snaps": team_snaps,
                   "snap_pct": round(100 * snaps / team_snaps, 1) if team_snaps else None,
                   "targets": targets, "carries": carries, "pass_att": pass_att,
                   "touches": carries + float(st.get("rec") or 0),
                   "target_share": round(100 * targets / team_targets[team], 1)
                   if team and team_targets.get(team) else None,
                   "ppr_pts": float(st.get("pts_ppr") or 0)}
            series.setdefault(str(pid), []).append(row)
            meta.setdefault(str(pid), {"position": pos, "team": team,
                                        "name": pm.get("name")})
    return series, meta


def main() -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="Weekly breakout chronological backtest")
    ap.add_argument("--season", type=int)
    ap.add_argument("--start-season", type=int)
    ap.add_argument("--end-season", type=int)
    ap.add_argument("--output", help="optional structured JSON report path")
    ap.add_argument("--source", choices=("auto", "db", "cache"), default="auto")
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--horizon", type=int, default=3, help="lookahead games (2-4)")
    ap.add_argument("--eval-from", type=int, default=4, help="first cutoff week to evaluate")
    ap.add_argument("--eval-to", type=int, default=15, help="last cutoff week to evaluate")
    ap.add_argument("--tune-through", type=int, default=None,
                    help="if set, weeks <= this are reserved for tuning and NOT evaluated")
    args = ap.parse_args()

    if args.season:
        seasons = [args.season]
    elif args.start_season and args.end_season:
        seasons = list(range(args.start_season, args.end_season + 1))
    else:
        ap.error("provide --season or --start-season and --end-season")
    start = args.eval_from
    if args.tune_through:
        start = max(start, args.tune_through + 1)
    eval_weeks = list(range(start, args.eval_to + 1))

    records, season_reports = [], {}
    for season in seasons:
        if args.source == "cache":
            series, meta = load_cached_season_series(season)
        else:
            try:
                series, meta = load_season_series(season)
            except Exception:
                if args.source == "db":
                    raise
                series, meta = load_cached_season_series(season)
        season_reports[str(season)] = run_backtest(series, meta, eval_weeks=eval_weeks,
                                                   top_n=args.top_n, horizon=args.horizon)
        records.extend(build_evaluation_records(series, meta, season=season, eval_weeks=eval_weeks))
    report = {"seasons": season_reports, "calibration": calibration_report(records, args.horizon),
              "records": records}
    print(f"=== Weekly breakout backtest: seasons {seasons[0]}-{seasons[-1]} ===")
    print(f"eval weeks {eval_weeks} | top_n={args.top_n} | horizon={args.horizon}")
    combined = season_reports[str(seasons[-1])]
    for m, mr in combined["methods"].items():
        print(f"  {m:16s} precision(usage)={mr['precision']} "
              f"precision(useful)={mr['precision_useful']} "
              f"coverage={mr['coverage']} fp_rate={mr['false_positive_rate']} "
              f"lead={mr['avg_lead_time_games']} (n={mr['picks_evaluated']})")
    print("  precision(usage): flagged player sustained the usage rise; "
          "precision(useful): flagged player reached startable PPR PPG over the lookahead.")
    print(json.dumps(report["calibration"], indent=2, sort_keys=True))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    return report


if __name__ == "__main__":
    main()
