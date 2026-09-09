"""
Offensive-line unit ratings derived entirely from open nflverse play-by-play.

Why this exists
---------------
Fantasy matchup work needs an O-line signal, but the sharp commercial grades
(PFF, etc.) are licensed and can't be redistributed or baked into a shipped
product even when hidden inside a composite. Everything here is computed from
public nflverse play-by-play, so the output is ours to display, cite, and cache.

The tradeoff, stated plainly: even nflverse's charted pressure flag is a
play-level signal, not a per-blocker grade, so this can't attribute a pressure
to a specific lineman or fully separate scheme (quick game, max protect, chip
help) from line talent. These are directional *unit* ratings and tiers, not
lineman grades. For fantasy matchup swings that's the right altitude.

Methodology
-----------
Two sub-ratings, each opponent-adjusted and regressed for sample size, then
blended into a composite.

PASS BLOCK -- a blend of pressure rate allowed (nflverse `was_pressure`, a
    charted flag, 2022+) and sack rate allowed. A backtest sweep set the split
    at 40% pressure / 60% sack (sacks proved the more transferable out-of-sample
    signal; see oline_backtest.sweep_pressure_weight). Both inputs are:
      * opponent-adjusted by an alternating-means model (each team's effect net
        of the pass rushes it faced), iterated to convergence;
      * residualised against the QB's average time to throw, so a line isn't
        punished for a QB who holds the ball (the Hurts/scramble problem);
      * regressed toward a prior (last season's adjusted value, else league
        mean) by sample size, so early-season small samples don't spike.
    Where `was_pressure` is unavailable (pre-2022) the QB-hit rate stands in.

RUN BLOCK -- Football Outsiders "Line Yards" (formula is public): each carry's
    yardage is weighted by distance, crediting the line for the yards it
    plausibly created and discounting the long runs that are mostly the back:
        < 0 yds (stuffed) -> 1.20x    0-4 yds -> 1.00x
        5-10 yds          -> 0.50x    11+ yds -> 0.00x
    Garbage time (win prob outside [0.05, 0.95]) and kneels are dropped, the
    same opponent-adjust + prior-regression is applied, and a descriptive
    interior/tackle/end split is exposed alongside the graded overall number.
    We also track stuffed rate (runs at or behind the LOS).

SCALING -- each team's adjusted-and-regressed metric is turned into a 0-100
    league percentile (100 = best line), the honest scale for a cross-sectional
    ranking (a z-score scale pins half a 32-team league at the 0/100 clamp).
    Composite = 55% pass block + 45% run block; tune with oline_backtest.py.

Output
------
cache/oline_ratings_s{season}.json:
    {
      "season": 2025, "through_week": 6, "generated_at": "...",
      "seasons_used": [2024, 2025], "n_run_plays": ..., "n_pass_plays": ...,
      "pressure_source": "was_pressure",
      "ratings": {
        "PHI": {
          "pass_block": 78.4, "run_block": 66.1, "composite": 72.9,
          "pressure_rate": 28.1, "sack_rate": 3.9, "qb_hit_rate": 14.2,
          "avg_time_to_throw": 2.71,
          "line_yards": 4.55, "stuffed_rate": 15.1,
          "run_block_interior": 61.0, "run_block_tackle": 70.0, "run_block_end": 55.0,
          "n_rush": 178, "n_pass": 241
        }, ...
      }
    }

Run directly:  python -m data_building.oline_ratings [season] [through_week]
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone

from utils.paths import CACHE_DIR

EXCLUDE_WEEKS = {18}            # modern analog of the old week-17 "rest" week
MIN_TEAM_PLAYS = 30            # below this a team's sample is too thin to rate at all
PASS_BLOCK_WEIGHT = 0.55
RUN_BLOCK_WEIGHT = 0.45
# Regression-to-prior strength, in play units: the number of current-season
# plays at which a team is weighted 50/50 against its prior. Higher = more
# conservative early in the season.
RUN_PRIOR_K = 100.0
PASS_PRIOR_K = 120.0
OPP_ADJUST_ITERS = 8
# Garbage-time filter: drop plays where the game is already decided, so blocking
# in blowouts (backups, prevent fronts) doesn't distort the rating.
WP_LOW, WP_HIGH = 0.05, 0.95
# Pass block sub-weights, chosen by oline_backtest.sweep_pressure_weight on
# 2022-2024: 0.4 maximises out-of-sample prediction of future pressure+sacks
# (a broad, flat optimum across 0.3-0.6). Sacks turn out to be the more
# transferable signal, so they carry the larger share -- the opposite of the
# initial guess. Re-run the sweep and update these two numbers if it moves.
PRESSURE_WEIGHT = 0.4
SACK_WEIGHT = 0.6

# Same alias table matchup_ratings uses, so the two caches key on identical codes.
_TEAM_ALIAS = {
    "JAC": "JAX", "LA": "LAR", "STL": "LAR", "OAK": "LV", "SD": "LAC",
    "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
}

# Columns the builder actually reads. nfl_data_py returns the full frame anyway;
# the direct-parquet fallback reads everything too, so this is documentation.
_PBP_COLS = [
    "season", "week", "season_type", "posteam", "defteam",
    "rush_attempt", "pass_attempt", "qb_dropback",
    "rushing_yards", "yards_gained",
    "sack", "qb_hit", "was_pressure", "time_to_throw",
    "qb_kneel", "qb_spike", "run_location", "run_gap",
    "wp", "score_differential",
]

_NFLVERSE_PBP_URLS = (
    "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{year}.parquet",
)


def _norm_team(t) -> str:
    t = (str(t) or "").upper().strip()
    return _TEAM_ALIAS.get(t, t)


def out_path(season: int) -> str:
    return os.path.join(str(CACHE_DIR), f"oline_ratings_s{season}.json")


def _f(v):
    try:
        if v is None:
            return None
        f = float(v)
        return f if f == f else None   # drop NaN
    except (TypeError, ValueError):
        return None


def _line_yards(yards: float) -> float:
    """Football Outsiders per-carry line-yards weighting."""
    if yards < 0:
        return 1.20 * yards
    if yards <= 4:
        return 1.00 * yards
    if yards <= 10:
        return 4.0 + 0.50 * (yards - 4)   # full credit for 0-4, half beyond
    return 4.0 + 0.50 * 6.0               # 11+ capped: line gets no long-run credit


def _gap_of(run_location, run_gap) -> str | None:
    """Bucket a carry into interior / tackle / end for the descriptive split."""
    g = (str(run_gap) or "").lower()
    loc = (str(run_location) or "").lower()
    if g == "guard" or loc == "middle":
        return "interior"
    if g == "tackle":
        return "tackle"
    if g == "end":
        return "end"
    return None


def _alt_adjust(triples, iters=OPP_ADJUST_ITERS):
    """Opponent-adjust via an additive alternating-means model.

    `triples` is an iterable of (offense, defense, value). We fit
        value ~= league + off_effect[o] + def_effect[d]
    by alternating conditional means (a couple of Gauss-Seidel passes). Returns
    ({offense: league + off_effect}, n_by_offense, league_mean). The returned
    offense value is that team's value adjusted for the strength of the units it
    faced -- exactly the strength-of-schedule idea matchup_ratings.py uses.
    """
    rows = [(o, d, v) for (o, d, v) in triples if o and d and v is not None]
    if not rows:
        return {}, {}, 0.0
    league = sum(v for _, _, v in rows) / len(rows)
    off_by = defaultdict(list)
    def_by = defaultdict(list)
    for i, (o, d, v) in enumerate(rows):
        off_by[o].append(i)
        def_by[d].append(i)
    vals = [v for _, _, v in rows]
    defs = [d for _, d, _ in rows]
    offs = [o for o, _, _ in rows]
    off_eff = defaultdict(float)
    def_eff = defaultdict(float)
    for _ in range(iters):
        for o, idxs in off_by.items():
            off_eff[o] = sum(vals[i] - league - def_eff[defs[i]] for i in idxs) / len(idxs)
        for d, idxs in def_by.items():
            def_eff[d] = sum(vals[i] - league - off_eff[offs[i]] for i in idxs) / len(idxs)
    adj = {o: league + off_eff[o] for o in off_by}
    n = {o: len(idxs) for o, idxs in off_by.items()}
    return adj, n, league


def _regress_to_prior(cur_adj, n_cur, prior_adj, league_cur, K):
    """Shrink each team's current value toward a prior (last year, else league).

    Unifies small-sample shrinkage (#3) and prior-season blending (#6): early in
    the year n_cur is small so the prior dominates; late in the year the current
    season takes over.
    """
    out = {}
    for t, cur in cur_adj.items():
        prior = prior_adj.get(t) if prior_adj else None
        if prior is None:
            prior = league_cur
        n = n_cur.get(t, 0)
        out[t] = (n * cur + K * prior) / (n + K) if (n + K) > 0 else cur
    return out


def _residualize(value_by_team, x_by_team):
    """Remove the linear effect of x (e.g. time to throw) from value.

    Regress value on x across teams and return residuals {team: value - fit}.
    Falls back to demeaned value if x has no usable spread. A team whose QB holds
    the ball longer is *expected* to allow more pressure; the residual isolates
    the part that isn't explained by that, i.e. the line's own contribution.
    """
    pairs = [(x_by_team[t], value_by_team[t]) for t in value_by_team
             if t in x_by_team and x_by_team[t] is not None]
    n = len(pairs)
    vmean = sum(value_by_team.values()) / len(value_by_team) if value_by_team else 0.0
    if n < 5:
        return {t: v - vmean for t, v in value_by_team.items()}
    xbar = sum(x for x, _ in pairs) / n
    ybar = sum(y for _, y in pairs) / n
    sxx = sum((x - xbar) ** 2 for x, _ in pairs)
    if sxx <= 0:
        return {t: v - vmean for t, v in value_by_team.items()}
    sxy = sum((x - xbar) * (y - ybar) for x, y in pairs)
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    out = {}
    for t, v in value_by_team.items():
        x = x_by_team.get(t)
        fit = (intercept + slope * x) if x is not None else ybar
        out[t] = v - fit
    return out


def _percentile_index(pairs, higher_is_better=True):
    """Map {team: value} -> {team: 0-100} by percentile rank across the league.

    Percentile (not a raw z-score) is the right scale for a cross-sectional
    ranking: with ~32 teams a z-score scale pins half the league at the 0/100
    clamp, while percentile keeps every team distinct. 100 = best line.
    """
    items = [(t, v) for t, v in pairs.items() if v is not None]
    if not items:
        return {}
    items.sort(key=lambda kv: kv[1], reverse=not higher_is_better)
    n = len(items)
    return {t: (round((i / (n - 1)) * 100.0, 1) if n > 1 else 50.0)
            for i, (t, _v) in enumerate(items)}


def _load_pbp_year(year, pd, nfl=None):
    """Return a play-by-play DataFrame for `year`, or None."""
    if nfl is not None:
        try:
            d = nfl.import_pbp_data([year], downcast=True)
            if d is not None and not d.empty:
                return d
        except Exception as e:
            print(f"[oline_ratings] nfl_data_py pbp {year} failed ({e}); trying nflverse direct")
    for url in _NFLVERSE_PBP_URLS:
        u = url.format(year=year)
        try:
            d = pd.read_parquet(u)
            if d is not None and not d.empty:
                print(f"[oline_ratings] {year} via {u}")
                return d
        except Exception as e:
            print(f"[oline_ratings] {year} {u.rsplit('/', 1)[-1]} -> {e}")
    return None


def _prep_season(d, pd, season, through_week=None):
    """Filter one raw pbp frame to regular-season, in-window rows."""
    if d is None or d.empty:
        return None
    if "season_type" in d:
        d = d[d["season_type"].astype(str).str.upper() == "REG"]
    if "week" in d:
        wk = pd.to_numeric(d["week"], errors="coerce")
        d = d[~wk.isin(EXCLUDE_WEEKS)]
        if through_week is not None and "season" in d:
            sn = pd.to_numeric(d["season"], errors="coerce")
            d = d[~((sn == season) & (wk > through_week))]
    return d


def _season_run_metrics(d):
    """Return per-team run metrics for one season frame.

    -> ({team: adj_line_yards}, {team: n}, league_ly,
        {team: {overall, interior, tackle, end, stuffed, n}})
    """
    triples = []          # (off, def, line_yards) for opponent adjustment
    raw = defaultdict(lambda: {"ly": 0.0, "stuff": 0.0, "n": 0,
                               "interior": [0.0, 0], "tackle": [0.0, 0], "end": [0.0, 0]})
    cols = {c: (d[c].tolist() if c in d else None) for c in
            ("posteam", "defteam", "rushing_yards", "yards_gained",
             "rush_attempt", "qb_kneel", "run_location", "run_gap", "wp")}
    n = len(d)
    for i in range(n):
        if cols["rush_attempt"] is None or _f(cols["rush_attempt"][i]) != 1:
            continue
        if cols["qb_kneel"] and _f(cols["qb_kneel"][i]) == 1:
            continue
        wp = _f(cols["wp"][i]) if cols["wp"] else None
        if wp is not None and not (WP_LOW <= wp <= WP_HIGH):
            continue  # garbage time
        o = _norm_team(cols["posteam"][i]) if cols["posteam"] else ""
        dd = _norm_team(cols["defteam"][i]) if cols["defteam"] else ""
        y = _f(cols["rushing_yards"][i]) if cols["rushing_yards"] else None
        if y is None and cols["yards_gained"]:
            y = _f(cols["yards_gained"][i])
        if not o or not dd or y is None:
            continue
        ly = _line_yards(y)
        triples.append((o, dd, ly))
        b = raw[o]
        b["ly"] += ly
        b["stuff"] += 1.0 if y <= 0 else 0.0
        b["n"] += 1
        gap = _gap_of(cols["run_location"][i] if cols["run_location"] else None,
                      cols["run_gap"][i] if cols["run_gap"] else None)
        if gap:
            b[gap][0] += ly
            b[gap][1] += 1
    adj, ncount, league = _alt_adjust(triples)
    detail = {}
    for t, b in raw.items():
        row = {"n": b["n"]}
        if b["n"] > 0:
            row["line_yards"] = round(b["ly"] / b["n"], 3)
            row["stuffed_rate"] = round(b["stuff"] / b["n"] * 100.0, 1)
        for g in ("interior", "tackle", "end"):
            s, c = b[g]
            row[g] = (s / c) if c else None
        detail[t] = row
    return adj, ncount, league, detail


def _season_pass_metrics(d):
    """Return per-team pass-protection metrics for one season frame.

    -> dict with opponent-adjusted pressure & sack values, plus context.
    Uses `was_pressure` when populated; otherwise falls back to `qb_hit`.
    """
    cols = {c: (d[c].tolist() if c in d else None) for c in
            ("posteam", "defteam", "qb_dropback", "pass_attempt",
             "sack", "qb_hit", "was_pressure", "time_to_throw", "qb_spike", "wp")}
    n = len(d)
    have_pressure = False
    if cols["was_pressure"] is not None:
        have_pressure = sum(1 for v in cols["was_pressure"] if _f(v) is not None) > 0.5 * max(1, n) * 0.1

    press_triples, sack_triples = [], []
    ttt = defaultdict(lambda: [0.0, 0])       # team -> [sum time_to_throw, count]
    raw = defaultdict(lambda: {"press": 0.0, "sack": 0.0, "hit": 0.0, "n": 0})
    for i in range(n):
        db = _f(cols["qb_dropback"][i]) if cols["qb_dropback"] else None
        pa = _f(cols["pass_attempt"][i]) if cols["pass_attempt"] else None
        sk = _f(cols["sack"][i]) if cols["sack"] else 0.0
        is_db = (db == 1) or (pa == 1) or (sk == 1)
        if not is_db:
            continue
        if cols["qb_spike"] and _f(cols["qb_spike"][i]) == 1:
            continue
        wp = _f(cols["wp"][i]) if cols["wp"] else None
        if wp is not None and not (WP_LOW <= wp <= WP_HIGH):
            continue
        o = _norm_team(cols["posteam"][i]) if cols["posteam"] else ""
        dd = _norm_team(cols["defteam"][i]) if cols["defteam"] else ""
        if not o or not dd:
            continue
        b = raw[o]
        b["n"] += 1
        s = 1.0 if sk else 0.0
        h = 1.0 if (cols["qb_hit"] and _f(cols["qb_hit"][i])) else 0.0
        b["sack"] += s
        b["hit"] += h
        sack_triples.append((o, dd, s))
        if have_pressure:
            p = _f(cols["was_pressure"][i])
            p = 1.0 if (p and p >= 1) else 0.0
            b["press"] += p
            press_triples.append((o, dd, p))
        else:
            b["press"] += h
            press_triples.append((o, dd, h))
        t2 = _f(cols["time_to_throw"][i]) if cols["time_to_throw"] else None
        if t2 is not None:
            ttt[o][0] += t2
            ttt[o][1] += 1

    press_adj, ncount, press_league = _alt_adjust(press_triples)
    sack_adj, _, sack_league = _alt_adjust(sack_triples)
    ttt_avg = {t: (v[0] / v[1]) for t, v in ttt.items() if v[1] > 0}
    detail = {}
    for t, b in raw.items():
        if b["n"] == 0:
            continue
        detail[t] = {
            "n": b["n"],
            "pressure_rate": round(b["press"] / b["n"] * 100.0, 1),
            "sack_rate": round(b["sack"] / b["n"] * 100.0, 1),
            "qb_hit_rate": round(b["hit"] / b["n"] * 100.0, 1),
            "avg_time_to_throw": round(ttt_avg[t], 3) if t in ttt_avg else None,
        }
    return {
        "press_adj": press_adj, "press_league": press_league,
        "sack_adj": sack_adj, "sack_league": sack_league,
        "n": ncount, "ttt": ttt_avg, "detail": detail,
        "pressure_source": "was_pressure" if have_pressure else "qb_hit",
    }


def build_oline_ratings(season: int, through_week: int | None = None, save: bool = True) -> dict:
    """Compute and (optionally) cache opponent-adjusted, regressed O-line ratings."""
    import pandas as pd
    try:
        import nfl_data_py as nfl
    except Exception:
        nfl = None

    cur_raw = _prep_season(_load_pbp_year(season, pd, nfl), pd, season, through_week)
    if cur_raw is None or cur_raw.empty:
        print(f"[oline_ratings] no current-season pbp for {season}")
        return {}
    prior_raw = _prep_season(_load_pbp_year(season - 1, pd, nfl), pd, season - 1)

    # --- RUN ---
    run_adj, run_n, run_league, run_detail = _season_run_metrics(cur_raw)
    run_prior = {}
    if prior_raw is not None and not prior_raw.empty:
        run_prior, _, _, _ = _season_run_metrics(prior_raw)
    run_final = _regress_to_prior(run_adj, run_n, run_prior, run_league, RUN_PRIOR_K)

    # --- PASS ---
    cur_pass = _season_pass_metrics(cur_raw)
    prior_pass = _season_pass_metrics(prior_raw) if (prior_raw is not None and not prior_raw.empty) else None
    press_final = _regress_to_prior(
        cur_pass["press_adj"], cur_pass["n"],
        prior_pass["press_adj"] if prior_pass else None,
        cur_pass["press_league"], PASS_PRIOR_K)
    sack_final = _regress_to_prior(
        cur_pass["sack_adj"], cur_pass["n"],
        prior_pass["sack_adj"] if prior_pass else None,
        cur_pass["sack_league"], PASS_PRIOR_K)

    # Residualise pressure & sacks against time to throw (isolate the line).
    press_resid = _residualize(press_final, cur_pass["ttt"])
    sack_resid = _residualize(sack_final, cur_pass["ttt"])

    # --- SCALE ---
    run_index = _percentile_index(run_final, higher_is_better=True)   # more line yards = better
    press_index = _percentile_index(press_resid, higher_is_better=False)  # less pressure = better
    sack_index = _percentile_index(sack_resid, higher_is_better=False)

    # Descriptive per-gap run indices (opponent-unadjusted; label as secondary).
    gap_index = {}
    for g in ("interior", "tackle", "end"):
        gv = {t: run_detail[t][g] for t in run_detail if run_detail[t].get(g) is not None}
        gap_index[g] = _percentile_index(gv, higher_is_better=True)

    # --- ASSEMBLE ---
    teams = set(run_final) | set(press_final)
    ratings = {}
    for t in teams:
        rd = run_detail.get(t, {})
        pd_ = cur_pass["detail"].get(t, {})
        n_rush = run_n.get(t, rd.get("n", 0))
        n_pass = cur_pass["n"].get(t, pd_.get("n", 0))
        if n_rush < MIN_TEAM_PLAYS and n_pass < MIN_TEAM_PLAYS:
            continue
        run_idx = run_index.get(t)
        p_idx = press_index.get(t)
        s_idx = sack_index.get(t)
        pass_idx = None
        if p_idx is not None and s_idx is not None:
            pass_idx = round(PRESSURE_WEIGHT * p_idx + SACK_WEIGHT * s_idx, 1)
        elif p_idx is not None:
            pass_idx = p_idx
        elif s_idx is not None:
            pass_idx = s_idx

        row = {"n_rush": n_rush, "n_pass": n_pass}
        if "line_yards" in rd:
            row["line_yards"] = rd["line_yards"]
            row["stuffed_rate"] = rd["stuffed_rate"]
        for k in ("pressure_rate", "sack_rate", "qb_hit_rate", "avg_time_to_throw"):
            if pd_.get(k) is not None:
                row[k] = pd_[k]
        if run_idx is not None:
            row["run_block"] = run_idx
        if pass_idx is not None:
            row["pass_block"] = pass_idx
        for g in ("interior", "tackle", "end"):
            gi = gap_index[g].get(t)
            if gi is not None:
                row[f"run_block_{g}"] = gi
        if run_idx is not None and pass_idx is not None:
            row["composite"] = round(PASS_BLOCK_WEIGHT * pass_idx + RUN_BLOCK_WEIGHT * run_idx, 1)
        elif run_idx is not None:
            row["composite"] = run_idx
        elif pass_idx is not None:
            row["composite"] = pass_idx
        ratings[t] = row

    out = {
        "season": season,
        "through_week": through_week,
        "seasons_used": sorted({season} | ({season - 1} if run_prior or prior_pass else set())),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pressure_source": cur_pass["pressure_source"],
        "n_run_plays": int(sum(run_n.values())),
        "n_pass_plays": int(sum(cur_pass["n"].values())),
        "weights": {
            "pass_block": PASS_BLOCK_WEIGHT, "run_block": RUN_BLOCK_WEIGHT,
            "pressure": PRESSURE_WEIGHT, "sack": SACK_WEIGHT,
            "run_prior_k": RUN_PRIOR_K, "pass_prior_k": PASS_PRIOR_K,
        },
        "ratings": ratings,
    }
    if save:
        os.makedirs(str(CACHE_DIR), exist_ok=True)
        tmp = out_path(season) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, out_path(season))
    return out


if __name__ == "__main__":
    import sys
    yr = int(sys.argv[1]) if len(sys.argv) > 1 else datetime.now().year
    tw = int(sys.argv[2]) if len(sys.argv) > 2 else None
    res = build_oline_ratings(yr, tw)
    r = res.get("ratings", {})
    print(f"[oline_ratings] season={yr} teams_rated={len(r)} "
          f"pressure_source={res.get('pressure_source')} "
          f"run_plays={res.get('n_run_plays')} pass_plays={res.get('n_pass_plays')} "
          f"-> {out_path(yr)}")
    rows = sorted(r.items(), key=lambda kv: kv[1].get("composite", 0), reverse=True)
    for t, row in rows:
        print(f"  {t:>3} comp={row.get('composite'):>5}  pass={row.get('pass_block'):>5} "
              f"run={row.get('run_block'):>5}  press%={row.get('pressure_rate')} "
              f"sack%={row.get('sack_rate')} ttt={row.get('avg_time_to_throw')}")
