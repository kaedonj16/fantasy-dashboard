#!/usr/bin/env python3
"""
Backtest: does "opp plays faced" actually predict fantasy volume/points?

The Start/Sit advisor shows how many offensive plays a player's NFL opponent
defense faces per game (see data_building/team_play_volume.py). It is display
-only context, deliberately kept out of the score. This harness answers the
question that decides whether the stat is worth trusting, and which *form* of it
to trust:

  * Does a defense's prior plays-faced correlate with the fantasy points a
    player scores against it next week?
  * Is the POSITION-RELEVANT split better than total scrimmage plays?
      - RB volume should track the opponent's RUSH plays faced.
      - QB/WR/TE volume should track the opponent's PASS plays faced.
  * Does a NEUTRAL-SCRIPT filter (drop garbage time by win probability, the same
    idea data_building/oline_ratings.py uses) predict better than raw counts?

Method (no leakage): for each player-week in week W, the opponent's plays-faced
is computed only from that opponent's games in weeks < W of the same season, so
nothing from week W leaks into its own predictor. Spearman rank correlation is
reported per position group and per metric variant, pooled across the requested
seasons. Spearman (not Pearson) because we only care about monotonic ranking,
which is how the stat is used.

Both inputs come from open nflverse feeds via nfl_data_py:
  * import_weekly_data  -> player-week fantasy_points_ppr + position + opponent
  * import_pbp_data     -> per-play posteam/defteam/play_type/wp for the counts

Because those pulls need pandas / nfl_data_py and network access, run this in an
environment where those import -- NOT the offline unit-test sandbox. It does not
import the Flask app.

Usage:
    python scripts/backtest_opp_plays_faced.py --seasons 2023,2024
    python scripts/backtest_opp_plays_faced.py --seasons 2022,2023,2024 --neutral-wp 0.10,0.90
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

# Allow running as a plain script by putting the repo root on the path.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.schedule_ease import norm_sched_team as _norm
except Exception:  # pragma: no cover - repo helper should always import
    def _norm(t):
        return (str(t) or "").upper().strip()

_SCRIMMAGE = frozenset({"pass", "run"})
_POS_GROUPS = ("QB", "RB", "WR", "TE")
# Which faced-split should, in theory, best predict each position's volume.
_EXPECTED_BASIS = {"QB": "pass", "RB": "rush", "WR": "pass", "TE": "pass"}


def _f(v):
    try:
        if v is None:
            return None
        x = float(v)
        return x if x == x else None  # drop NaN
    except (TypeError, ValueError):
        return None


def _rank(vals):
    """Fractional (tie-averaged) ranks, so Spearman handles ties correctly."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rho via Pearson on ranks. None when the sample is too small."""
    n = len(xs)
    if n < 5:
        return None
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return (num / den) if den else None


def _load(seasons, wp_lo, wp_hi):
    """Return (weekly_rows, faced) for the requested seasons.

    weekly_rows: list of (season, week, position, opp_team, ppr_points).
    faced: {(season, team, week): {pass, rush, npass, nrush}} defensive counts
           (n* = neutral-script: win probability inside [wp_lo, wp_hi]).
    """
    import nfl_data_py as nfl

    # ---- Player-week fantasy points / position / opponent ----
    wk = nfl.import_weekly_data(list(seasons))
    if "season_type" in wk:
        wk = wk[wk["season_type"].astype(str).str.upper() == "REG"]
    cols = {c: wk[c].tolist() if c in wk else None for c in
            ("season", "week", "position", "opponent_team", "fantasy_points_ppr")}
    weekly_rows = []
    n = len(wk)
    for i in range(n):
        pos = str((cols["position"] or [""] * n)[i] or "").upper()
        if pos not in _POS_GROUPS:
            continue
        opp = _norm((cols["opponent_team"] or [""] * n)[i])
        pts = _f((cols["fantasy_points_ppr"] or [None] * n)[i])
        sn = _f((cols["season"] or [None] * n)[i])
        w = _f((cols["week"] or [None] * n)[i])
        if not opp or pts is None or sn is None or w is None:
            continue
        weekly_rows.append((int(sn), int(w), pos, opp, pts))

    # ---- Per (season, team, week) defensive plays faced ----
    pbp = nfl.import_pbp_data(
        list(seasons),
        columns=["season", "week", "season_type", "defteam", "play_type", "wp"],
        downcast=True,
    )
    if "season_type" in pbp:
        pbp = pbp[pbp["season_type"].astype(str).str.upper() == "REG"]
    pcols = {c: pbp[c].tolist() if c in pbp else None for c in
             ("season", "week", "defteam", "play_type", "wp")}
    faced = defaultdict(lambda: {"pass": 0, "rush": 0, "npass": 0, "nrush": 0})
    m = len(pbp)
    for i in range(m):
        pt = str((pcols["play_type"] or [""] * m)[i] or "").lower()
        if pt not in _SCRIMMAGE:
            continue
        d = _norm((pcols["defteam"] or [""] * m)[i])
        sn = _f((pcols["season"] or [None] * m)[i])
        w = _f((pcols["week"] or [None] * m)[i])
        if not d or d.lower() == "nan" or sn is None or w is None:
            continue
        key = (int(sn), d, int(w))
        b = faced[key]
        is_pass = pt == "pass"
        b["pass" if is_pass else "rush"] += 1
        wp = _f((pcols["wp"] or [None] * m)[i])
        if wp is not None and wp_lo <= wp <= wp_hi:
            b["npass" if is_pass else "nrush"] += 1
    return weekly_rows, faced


def run(seasons, wp_lo, wp_hi, min_prior_games):
    weekly_rows, faced = _load(seasons, wp_lo, wp_hi)

    # Pre-index defensive weeks per (season, team) for fast prior-week sums.
    weeks_by_team = defaultdict(list)  # (season, team) -> [week, ...]
    for (sn, team, w) in faced:
        weeks_by_team[(sn, team)].append(w)

    # variant -> position -> (metric_values, points_values)
    variants = ("total_raw", "pass_raw", "rush_raw",
                "total_neutral", "pass_neutral", "rush_neutral")
    data = {v: {p: ([], []) for p in _POS_GROUPS} for v in variants}

    for (sn, w, pos, opp, pts) in weekly_rows:
        prior = [pw for pw in weeks_by_team.get((sn, opp), []) if pw < w]
        games = len(prior)
        if games < min_prior_games:
            continue
        acc = {"pass": 0, "rush": 0, "npass": 0, "nrush": 0}
        for pw in prior:
            b = faced[(sn, opp, pw)]
            for k in acc:
                acc[k] += b[k]
        pass_pg = acc["pass"] / games
        rush_pg = acc["rush"] / games
        npass_pg = acc["npass"] / games
        nrush_pg = acc["nrush"] / games
        vals = {
            "total_raw": pass_pg + rush_pg,
            "pass_raw": pass_pg,
            "rush_raw": rush_pg,
            "total_neutral": npass_pg + nrush_pg,
            "pass_neutral": npass_pg,
            "rush_neutral": nrush_pg,
        }
        for v in variants:
            xs, ys = data[v][pos]
            xs.append(vals[v])
            ys.append(pts)

    # ---- Report ----
    print(f"\nopp plays faced -> next-week PPR  (Spearman rho)")
    print(f"seasons={','.join(str(s) for s in seasons)}  "
          f"neutral WP window=[{wp_lo},{wp_hi}]  min prior games={min_prior_games}\n")
    header = f"{'pos':<4}{'n':>6}  " + "".join(f"{v:>15}" for v in variants)
    print(header)
    print("-" * len(header))
    for pos in _POS_GROUPS:
        n = len(data["total_raw"][pos][1])
        cells = []
        for v in variants:
            xs, ys = data[v][pos]
            rho = spearman(xs, ys)
            cells.append("   n/a" if rho is None else f"{rho:+.3f}")
        row = f"{pos:<4}{n:>6}  " + "".join(f"{c:>15}" for c in cells)
        exp = _EXPECTED_BASIS[pos]
        print(row + f"   (expect {exp} to lead)")
    print("\nReading it: for RB the rush_* columns should beat pass_*/total; for")
    print("QB/WR/TE the pass_* columns should. Compare *_raw vs *_neutral to see")
    print("whether dropping garbage time helps. Weak/negative rho across the board")
    print("means the stat is noise and shouldn't be promoted beyond context.\n")


def _parse_seasons(s):
    return [int(x) for x in str(s).split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Backtest opp plays faced vs next-week PPR")
    ap.add_argument("--seasons", default="2023,2024",
                    help="comma-separated seasons, e.g. 2022,2023,2024")
    ap.add_argument("--neutral-wp", default="0.05,0.95",
                    help="win-probability window kept for the neutral-script variant")
    ap.add_argument("--min-prior-games", type=int, default=2,
                    help="require this many prior defensive games before scoring a matchup")
    args = ap.parse_args()

    seasons = _parse_seasons(args.seasons)
    lo, hi = (float(x) for x in str(args.neutral_wp).split(","))
    try:
        run(seasons, lo, hi, args.min_prior_games)
    except ImportError as e:
        print(f"[backtest_opp_plays_faced] nfl_data_py / pandas required: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
