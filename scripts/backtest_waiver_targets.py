#!/usr/bin/env python3
"""
Calibration backtest for the waiver-target ranking model (utils.waiver_score).

The score in utils.waiver_score is built from hand-tuned weights
(utils.waiver_score.WEIGHTS). This harness answers the only question that
matters: do the players the model ranks highly actually out-produce over the
following weeks? It scores the waiver-eligible pool as it looked at week W and
compares that ranking to realized fantasy points over weeks W+1..W+horizon,
reporting:

  * Spearman rank correlation (score vs. realized points),
  * precision@k (share of the model's top-k that finished in the actual top-k),
  * lift over a value-only baseline (does the opportunity/injury/need signal
    add anything beyond static dynasty value?).

With --sweep it re-runs while scaling one weight up/down so you can see which
direction improves the metrics, i.e. tune WEIGHTS with evidence instead of
guesswork (#8).

Usage:
    python scripts/backtest_waiver_targets.py --season 2024 --weeks 4-13 --horizon 4
    python scripts/backtest_waiver_targets.py --season 2024 --weeks 4-13 --sweep injury_max

Because realized weekly points and historical value snapshots come from the
app's data stores (pandas / DB), run this in an environment where `app` and
`data_building` import — not the offline unit-test sandbox.
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

# Allow running as a plain script by putting the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.waiver_score import WEIGHTS, WaiverWeights, waiver_pickup_score, value_component


# ---------------------------------------------------------------------------
# Metrics (self-contained — no scipy/numpy dependency)
# ---------------------------------------------------------------------------

def _rank(xs: List[float]) -> List[float]:
    """Average-rank of each element (ties share the mean rank)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: List[float], b: List[float]) -> float:
    """Spearman rank correlation of two equal-length sequences."""
    n = len(a)
    if n < 2:
        return 0.0
    ra, rb = _rank(a), _rank(b)
    mean = (n + 1) / 2.0
    num = sum((ra[i] - mean) * (rb[i] - mean) for i in range(n))
    da = sum((r - mean) ** 2 for r in ra) ** 0.5
    db = sum((r - mean) ** 2 for r in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def precision_at_k(scores: List[float], actuals: List[float], k: int) -> float:
    """Share of the top-k by score that are also in the top-k by realized points."""
    if not scores or k <= 0:
        return 0.0
    top_score = {i for i, _ in sorted(enumerate(scores), key=lambda t: t[1], reverse=True)[:k]}
    top_actual = {i for i, _ in sorted(enumerate(actuals), key=lambda t: t[1], reverse=True)[:k]}
    return len(top_score & top_actual) / float(min(k, len(scores)))


# ---------------------------------------------------------------------------
# Data interface (pluggable so the harness stays testable / env-agnostic)
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class WeekSnapshot:
    """The inputs needed to score + grade one week's waiver pool."""
    candidates: List[dict]                 # scorer-ready candidate dicts at week W
    breakout: Dict[str, float]             # player_id -> breakout score at week W
    realized_points: Dict[str, float]      # player_id -> total points over W+1..W+horizon


_CORE_POS = ("QB", "RB", "WR", "TE")


def _fetch(cur, sql: str, params=()) -> List[dict]:
    """Run a query and return rows as dicts, whether the cursor yields dict rows
    (RealDictCursor) or plain tuples."""
    cur.execute(sql, params)
    rows = cur.fetchall()
    if not rows:
        return []
    if isinstance(rows[0], dict):
        return [dict(r) for r in rows]
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


def _week_anchor_date(season: int, week: int) -> "_dt.date":
    """The calendar date to read week-W values 'as of' — the day after week W's
    last game. Uses the cached schedule when available, else estimates from the
    NFL season start (Thursday after Labor Day)."""
    try:
        from utils.utils import load_week_schedule
        dates = []
        for g in (load_week_schedule(int(season), int(week)) or []):
            gd = str(g.get("gameDate") or "") if isinstance(g, dict) else ""
            if len(gd) == 8 and gd.isdigit():
                dates.append(_dt.date(int(gd[:4]), int(gd[4:6]), int(gd[6:8])))
        if dates:
            return max(dates) + _dt.timedelta(days=1)
    except Exception:
        pass
    sep1 = _dt.date(int(season), 9, 1)
    labor_day = sep1 + _dt.timedelta(days=(7 - sep1.weekday()) % 7)   # first Monday of Sept
    week1_thu = labor_day + _dt.timedelta(days=3)                     # Thu after Labor Day
    return week1_thu + _dt.timedelta(days=(int(week) - 1) * 7 + 4)    # ~Monday after week W


def _snapshot_date_on_or_before(cur, d: "_dt.date"):
    rows = _fetch(
        cur,
        "SELECT DISTINCT as_of_date FROM player_value_history "
        "WHERE as_of_date <= %s AND source = 'model' ORDER BY as_of_date DESC LIMIT 1",
        (d,),
    )
    return rows[0]["as_of_date"] if rows else None


def _load_value_snapshot(cur, snap_date) -> Dict[str, dict]:
    """{player_id: {value, position, pos_rank_label}} for one value-history date,
    limited to core offensive positions."""
    rows = _fetch(
        cur,
        "SELECT player_id, position, value FROM player_value_history "
        "WHERE as_of_date = %s AND source = 'model' AND position IN ('QB','RB','WR','TE')",
        (snap_date,),
    )
    out: Dict[str, dict] = {}
    for r in rows:
        try:
            val = float(r.get("value") or 0)
        except (TypeError, ValueError):
            val = 0.0
        if val < 5:
            continue
        out[str(r["player_id"])] = {"value": val, "position": str(r.get("position") or "").upper()}
    # Positional rank labels (WR12, RB27 …) for the depth-aware trend logic.
    by_pos: Dict[str, list] = {}
    for pid, v in out.items():
        by_pos.setdefault(v["position"], []).append((pid, v["value"]))
    for pos, lst in by_pos.items():
        lst.sort(key=lambda t: t[1], reverse=True)
        for i, (pid, _) in enumerate(lst):
            out[pid]["pos_rank_label"] = f"{pos}{i + 1}"
    return out


def _player_overall_ranks(vals: Dict[str, dict]) -> Dict[str, int]:
    """Player-only overall rank by value (1 = highest), matching how the app
    derives rank_change_7d."""
    ordered = sorted(vals.items(), key=lambda kv: kv[1]["value"], reverse=True)
    return {pid: i + 1 for i, (pid, _) in enumerate(ordered)}


def _default_snapshot_loader(season: int, week: int, horizon: int) -> Optional[WeekSnapshot]:
    """Build a WeekSnapshot from the app's historical stores.

    * candidates      <- player_value_history at the snapshot on/before week W
                         (value + positional rank label + rank_change_7d vs the
                         snapshot ~7 days earlier). No lookahead.
    * breakout        <- breakout_opportunity_scores, latest per player as of the
                         anchor date ({} when none exist historically).
    * realized_points <- SUM(player_weekly_metrics.ppr_pts) over weeks W+1..W+H.

    Note: injury/usage/projection enrichment and age aren't reconstructable from
    the historical stores, so this validates the value + trend + breakout core of
    the model (its ranking vs. a value-only baseline), not those live-only signals.
    """
    try:
        from dashboard_services.db import get_conn
    except Exception as exc:
        print(f"[backtest] DB unavailable: {exc}")
        return None

    try:
        anchor = _week_anchor_date(season, week)
        prior = anchor - _dt.timedelta(days=7)
        with get_conn() as conn:
            with conn.cursor() as cur:
                snap = _snapshot_date_on_or_before(cur, anchor)
                if not snap:
                    print(f"[backtest] wk{week}: no value_history snapshot on/before {anchor}")
                    return None
                cur_vals = _load_value_snapshot(cur, snap)
                prior_snap = _snapshot_date_on_or_before(cur, prior)
                prior_vals = _load_value_snapshot(cur, prior_snap) if prior_snap else {}

                realized_rows = _fetch(
                    cur,
                    "SELECT player_id, SUM(ppr_pts) AS pts FROM player_weekly_metrics "
                    "WHERE season = %s AND week > %s AND week <= %s GROUP BY player_id",
                    (int(season), int(week), int(week) + int(horizon)),
                )
                realized = {}
                for r in realized_rows:
                    try:
                        realized[str(r["player_id"])] = float(r.get("pts") or 0)
                    except (TypeError, ValueError):
                        pass

                breakout: Dict[str, float] = {}
                try:
                    b_rows = _fetch(
                        cur,
                        "SELECT DISTINCT ON (player_id) player_id, breakout_opportunity_score "
                        "FROM breakout_opportunity_scores WHERE as_of_date <= %s "
                        "ORDER BY player_id, as_of_date DESC",
                        (anchor,),
                    )
                    for r in b_rows:
                        s = r.get("breakout_opportunity_score")
                        if s is not None:
                            breakout[str(r["player_id"])] = float(s)
                except Exception:
                    breakout = {}   # table may not exist / no history

        if not cur_vals or not realized:
            print(f"[backtest] wk{week}: values={len(cur_vals)} realized={len(realized)} — skipping")
            return None

        cur_rank = _player_overall_ranks(cur_vals)
        prior_rank = _player_overall_ranks(prior_vals)
        candidates = []
        for pid, v in cur_vals.items():
            rc = (prior_rank[pid] - cur_rank[pid]) if (pid in prior_rank and pid in cur_rank) else None
            candidates.append({
                "player_id": pid,
                "position": v["position"],
                "value": v["value"],
                "pos_rank_label": v.get("pos_rank_label"),
                "rank_change_7d": rc,
            })
        return WeekSnapshot(candidates=candidates, breakout=breakout, realized_points=realized)
    except Exception as exc:  # pragma: no cover - env dependent
        print(f"[backtest] wk{week} load failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(snapshots: List[WeekSnapshot], weights: WaiverWeights, k: int = 15) -> dict:
    """Aggregate Spearman / precision@k / value-only lift across week snapshots."""
    sp_model: List[float] = []
    sp_value: List[float] = []
    p_model: List[float] = []
    p_value: List[float] = []

    for snap in snapshots:
        cands = [c for c in snap.candidates if c.get("player_id") in snap.realized_points]
        if len(cands) < k:
            continue
        actual = [snap.realized_points[c["player_id"]] for c in cands]
        model = [waiver_pickup_score(c, snap.breakout, w=weights) for c in cands]
        valonly = [value_component(c.get("value"), weights) for c in cands]

        sp_model.append(spearman(model, actual))
        sp_value.append(spearman(valonly, actual))
        p_model.append(precision_at_k(model, actual, k))
        p_value.append(precision_at_k(valonly, actual, k))

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "weeks": len(sp_model),
        "spearman_model": _avg(sp_model),
        "spearman_value_only": _avg(sp_value),
        "precision_at_k_model": _avg(p_model),
        "precision_at_k_value_only": _avg(p_value),
        "k": k,
    }


def _parse_weeks(spec: str) -> List[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def _print_report(label: str, m: dict) -> None:
    print(f"\n=== {label} ({m['weeks']} weeks, k={m['k']}) ===")
    print(f"  Spearman   model={m['spearman_model']:+.3f}   "
          f"value-only={m['spearman_value_only']:+.3f}   "
          f"lift={m['spearman_model'] - m['spearman_value_only']:+.3f}")
    print(f"  Precision  model={m['precision_at_k_model']:.3f}   "
          f"value-only={m['precision_at_k_value_only']:.3f}   "
          f"lift={m['precision_at_k_model'] - m['precision_at_k_value_only']:+.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--weeks", default="4-13", help="e.g. 4-13 or 4,6,8")
    ap.add_argument("--horizon", type=int, default=4, help="weeks of realized points to grade against")
    ap.add_argument("--k", type=int, default=15, help="top-k for precision")
    ap.add_argument("--sweep", default=None,
                    help="a WaiverWeights field to scale by 0.5x/1x/1.5x and compare")
    ap.add_argument("--loader", default=None,
                    help="dotted path to a custom snapshot loader(season, week, horizon)")
    args = ap.parse_args()

    loader: Callable = _default_snapshot_loader
    if args.loader:
        mod, _, fn = args.loader.rpartition(".")
        loader = getattr(__import__(mod, fromlist=[fn]), fn)

    snaps: List[WeekSnapshot] = []
    for wk in _parse_weeks(args.weeks):
        snap = loader(args.season, wk, args.horizon)
        if snap:
            snaps.append(snap)

    if not snaps:
        print("\nNo week snapshots loaded — wire _default_snapshot_loader() (or pass "
              "--loader) to your historical value + weekly-stats stores, then re-run.")
        return 1

    if not args.sweep:
        _print_report("waiver-target model", evaluate(snaps, WEIGHTS, args.k))
        return 0

    if args.sweep not in {f.name for f in dataclasses.fields(WaiverWeights)}:
        print(f"Unknown weight '{args.sweep}'. Options: "
              f"{', '.join(f.name for f in dataclasses.fields(WaiverWeights))}")
        return 2
    base = getattr(WEIGHTS, args.sweep)
    for factor in (0.5, 1.0, 1.5):
        w = dataclasses.replace(WEIGHTS, **{args.sweep: base * factor})
        _print_report(f"{args.sweep} x{factor} (= {base * factor:g})", evaluate(snaps, w, args.k))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
