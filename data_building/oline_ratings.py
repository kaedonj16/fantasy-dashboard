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

Optional talent prior (off by default)
--------------------------------------
The grade is lagging: the only prior is last season's opponent-adjusted value,
blended in by `_regress_to_prior` (RUN_PRIOR_K / PASS_PRIOR_K). That machinery
stays. An optional *projection* — snap-weighted returning-OL continuity, draft
capital added to the line, and snap-weighted veteran additions/losses — can
nudge that prior term so week-1 grades reflect offseason roster change instead
of last year's results alone. It does **not** replace the realized-results
pipeline, and it does not keep a side-channel influence after current-season
sample takes over: the adjustment lives inside the prior, so the existing
n_cur/K blend decays it to ~0 by mid-season.

Provenance is open nflverse only (PFR snap counts, weekly/seasonal rosters,
nflverse players file as a PFR↔GSIS id join, draft_picks, Chase Stuart's
public expected-AV chart). No PFF or other licensed grade is read, even as
a hidden input. Coaching/scheme change is omitted — there is no clean
redistributable encoding of OC / OL-coach turnover. See
`oline_talent_prior.py` and `oline_backtest.sweep_talent_prior`. A 2022-2025
weeks 1-4 backtest did not find a meaningful lift over last season's prior,
so the flag stays off.

This is a projection, not a measurement. `use_talent_prior=False` (default)
leaves the results-based composite byte-for-byte as it was. When the flag is
on, the main `pass_block` / `run_block` / `composite` keys use the
talent-shifted prior, and the unshifted measurement is stored alongside as
`pass_block_realized` / `run_block_realized` / `composite_realized`, with a
top-level `talent_prior` object describing sources, weights, and per-team
components. Do not treat those two as interchangeable.

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
      *     residualised against the QB's average time to throw, so a line isn't
        punished for a QB who holds the ball (the Hurts/scramble problem);
      * optionally residualised against average pass-rushers faced
        (`PASS_RUSHERS_RESID`; off until the pipeline sweep says otherwise);
      * regressed toward a prior (last season's adjusted value, optionally
        mixed with Y-2 via `PRIOR_Y2_WEIGHT`, else league mean) by sample
        size, so early-season small samples don't spike.
      * optionally shrunk toward league mean when last-year OL snaps are
        Out/IR this week (`AVAILABILITY_SHRINK`).
    Where `was_pressure` is unavailable (pre-2022) the QB-hit rate stands in.
    Current-season point estimates can be recency-weighted (`RECENCY_HALF_LIFE`);
    n_cur stays a raw play count so late-season grades are not pulled back
    toward last year.

RUN BLOCK -- a blend of Football Outsiders "Line Yards" and rush success rate.
    Line yards weight each carry by distance, crediting the line for yards it
    plausibly created and discounting long runs that are mostly the back:
        < 0 yds (stuffed) -> 1.20x    0-4 yds -> 1.00x
        5-10 yds          -> 0.50x    11+ yds -> 0.00x
    Success rate (nflverse `success`) captures early-down efficiency the FO
    curve alone misses. A 2023-2025 split-half sweep settled on 55% line yards /
    45% success (see RUN_LY_WEIGHT); both inputs are opponent-adjusted and
    prior-regressed like pass block. An optional opponent-adjusted stuffed-rate
    term (`RUN_STUFF_WEIGHT`) can join that blend; 0 keeps the two-way mix.
    QB scrambles and designed QB keeps are
    dropped so a mobile QB doesn't inflate (or deflate) the line's grade.
    Garbage time (win prob outside [0.05, 0.95]) and kneels are also dropped.
    A descriptive interior/tackle/end split and stuffed rate are exposed
    alongside the graded overall number.

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

    With --talent-prior / use_talent_prior=True the primary grade keys are the
    projection and a labeled `talent_prior` object is added; the unshifted
    measurement is stored as pass_block_realized / run_block_realized /
    composite_realized. Default (flag off) JSON is unchanged.

Run directly:  python -m data_building.oline_ratings [season] [through_week] [--talent-prior]
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
# Run-block sub-weights, chosen by a 2023-2025 first-half -> second-half sweep
# of opponent-adjusted components (QB runs excluded). 0.55 line-yards /
# 0.45 success maximises mean Spearman vs future line yards AND future success;
# pure line-yards scores ~0.05 rho worse. Re-sweep if the mix drifts.
RUN_LY_WEIGHT = 0.55
RUN_SUCCESS_WEIGHT = 0.45
# Recency half-life in weeks for the *current-season point estimate*.
# 0 = every play equal (legacy). n_cur for prior-regression stays raw play
# counts so late-season grades are not shrunk back toward last year.
# Swept in oline_backtest.sweep_results_pipeline; 0 until that sweep says else.
RECENCY_HALF_LIFE = 0.0
# Blend of Y-2 into the last-season prior: prior = (1-w)*Y-1 + w*Y-2.
# 0 = last season only (legacy). Swept in sweep_results_pipeline.
PRIOR_Y2_WEIGHT = 0.0
# Extra run-block component: opponent-adjusted stuffed rate (lower is better).
# 0 = line-yards + success only (legacy). Remaining weight stays on LY/success
# in their current ratio. Swept in sweep_results_pipeline.
RUN_STUFF_WEIGHT = 0.0
# After time-to-throw residualisation, also residualise pressure/sacks against
# average pass-rushers faced (nflverse `number_of_pass_rushers`). Isolates the
# line from extra-rusher looks. Swept in sweep_results_pipeline.
PASS_RUSHERS_RESID = False
# When a last-year OL snap-share is Out/IR this week, shrink that team's
# last-season prior toward league mean by the missing snap share. This is
# "the prior unit isn't on the field", not a talent projection. Swept in
# sweep_results_pipeline; off until the sweep says otherwise.
AVAILABILITY_SHRINK = False
# Multiplier on RUN_PRIOR_K / PASS_PRIOR_K. 1.0 = shipped K (legacy).
# Swept in sweep_results_pipeline.
K_MULT = 1.0
# Talent-prior scale: last-year cross-sectional SDs of shift per 1 SD of the
# roster/draft residual. Swept in oline_backtest.sweep_talent_prior on
# 2022-2025, weeks 1-4 ratings -> rest-of-season (weeks 5-17) pressure / sack /
# line-yards / success (n=128 team-seasons). Equal mix, W=0.45 was the
# nominal peak at score 0.3476 vs last-season-prior 0.3474 — a tie, not a
# win (pressure rho improved ~0.03; sacks and run-blocking did not).
# FLAG STAYS OFF. Any larger W monotonically hurt. Re-sweep if the mix drifts.
TALENT_PRIOR_W = 0.45
TALENT_W_CONTINUITY = 1.0 / 3.0
TALENT_W_DRAFT = 1.0 / 3.0
TALENT_W_VETERAN = 1.0 / 3.0

# Same alias table matchup_ratings uses, so the two caches key on identical codes.
# PFR codes (GNB/KAN/...) appear on snap-count dumps feeding the talent prior.
_TEAM_ALIAS = {
    "JAC": "JAX", "LA": "LAR", "STL": "LAR", "OAK": "LV", "SD": "LAC",
    "WSH": "WAS", "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU",
    "GNB": "GB", "GBP": "GB", "KAN": "KC", "KCC": "KC", "NWE": "NE", "NEP": "NE",
    "NOR": "NO", "SFO": "SF", "TAM": "TB", "TBB": "TB", "LVR": "LV",
    "SDG": "LAC", "RAM": "LAR", "OTI": "TEN", "RAI": "LV",
}

# Columns the builder actually reads. nfl_data_py returns the full frame anyway;
# the direct-parquet fallback reads everything too, so this is documentation.
_PBP_COLS = [
    "season", "week", "season_type", "posteam", "defteam",
    "rush_attempt", "pass_attempt", "qb_dropback",
    "rushing_yards", "yards_gained",
    "sack", "qb_hit", "was_pressure", "time_to_throw",
    "qb_kneel", "qb_spike", "qb_scramble", "run_location", "run_gap",
    "success", "passer_player_id", "rusher_player_id",
    "wp", "score_differential",
    "number_of_pass_rushers",
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


def _recency_weight(week, end_week, half_life):
    """Exponential decay: a play `half_life` weeks old counts half as much.

    `half_life <= 0` means unweighted (every play = 1). Used only for the
    current-season point estimate, not for n_cur.
    """
    if not half_life or half_life <= 0:
        return 1.0
    try:
        lag = max(0.0, float(end_week) - float(week))
    except (TypeError, ValueError):
        return 1.0
    return 0.5 ** (lag / float(half_life))


def _frame_end_week(d):
    if d is None or "week" not in getattr(d, "columns", []):
        return None
    try:
        wk = [w for w in (_f(v) for v in d["week"].tolist()) if w is not None]
        return max(wk) if wk else None
    except Exception:
        return None


def _alt_adjust(triples, iters=OPP_ADJUST_ITERS):
    """Opponent-adjust via an additive alternating-means model.

    `triples` is an iterable of (offense, defense, value) or
    (offense, defense, value, weight). We fit
        value ~= league + off_effect[o] + def_effect[d]
    by alternating conditional means (a couple of Gauss-Seidel passes). Returns
    ({offense: league + off_effect}, n_by_offense, league_mean). n is the
    unweighted play count (for prior regression). The returned offense value is
    that team's value adjusted for the strength of the units it faced --
    exactly the strength-of-schedule idea matchup_ratings.py uses.
    """
    rows = []
    for item in triples:
        if item is None:
            continue
        if len(item) == 3:
            o, d, v = item
            w = 1.0
        else:
            o, d, v, w = item[0], item[1], item[2], item[3]
        if o and d and v is not None and w is not None and w > 0:
            rows.append((o, d, v, float(w)))
    if not rows:
        return {}, {}, 0.0
    wsum = sum(w for _, _, _, w in rows)
    league = sum(v * w for _, _, v, w in rows) / wsum
    off_by = defaultdict(list)
    def_by = defaultdict(list)
    for i, (o, d, v, w) in enumerate(rows):
        off_by[o].append(i)
        def_by[d].append(i)
    vals = [v for _, _, v, _ in rows]
    wts = [w for _, _, _, w in rows]
    defs = [d for _, d, _, _ in rows]
    offs = [o for o, _, _, _ in rows]
    off_eff = defaultdict(float)
    def_eff = defaultdict(float)
    for _ in range(iters):
        for o, idxs in off_by.items():
            sw = sum(wts[i] for i in idxs)
            off_eff[o] = sum(wts[i] * (vals[i] - league - def_eff[defs[i]]) for i in idxs) / sw
        for d, idxs in def_by.items():
            sw = sum(wts[i] for i in idxs)
            def_eff[d] = sum(wts[i] * (vals[i] - league - off_eff[offs[i]]) for i in idxs) / sw
    adj = {o: league + off_eff[o] for o in off_by}
    n = {o: len(idxs) for o, idxs in off_by.items()}
    return adj, n, league


def _mix_prior(y1, y2, w2):
    """prior = (1-w2)*year-1 + w2*year-2. Missing years fall back to the other."""
    y1, y2 = dict(y1 or {}), dict(y2 or {})
    if not w2:
        return y1
    out = {}
    for t in set(y1) | set(y2):
        a, b = y1.get(t), y2.get(t)
        if a is not None and b is not None:
            out[t] = (1.0 - w2) * a + w2 * b
        elif a is not None:
            out[t] = a
        elif b is not None:
            out[t] = b
    return out


def _apply_availability_shrink(prior_adj, avail_scale, league):
    """prior' = s*prior + (1-s)*league, s = fraction of last-year OL snaps active."""
    prior_adj = dict(prior_adj or {})
    if not prior_adj or not avail_scale:
        return prior_adj
    out = {}
    for t, p in prior_adj.items():
        s = avail_scale.get(t)
        if s is None:
            out[t] = p
            continue
        try:
            s = min(1.0, max(0.0, float(s)))
        except (TypeError, ValueError):
            out[t] = p
            continue
        out[t] = s * p + (1.0 - s) * league
    return out


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


def _is_qb_run(qb_scramble, passer_player_id, rusher_player_id) -> bool:
    """True for scrambles and designed QB keeps — not an O-line grading play.

    Mobile QBs inflate (Hurts/Allen keepers) or muddy line-yards; dropping them
    isolates the blocking unit. Identification: charted scramble flag, or the
    passer and rusher being the same player (designed QB run).
    """
    try:
        if qb_scramble is not None and float(qb_scramble) == 1:
            return True
    except (TypeError, ValueError):
        pass
    if passer_player_id is None or rusher_player_id is None:
        return False
    # pandas NaN != NaN; require both present and equal.
    if passer_player_id != passer_player_id or rusher_player_id != rusher_player_id:
        return False
    return passer_player_id == rusher_player_id


def _season_run_metrics(d, recency_half_life=None):
    """Return per-team run metrics for one season frame.

    -> ({team: adj_line_yards}, {team: adj_success}, {team: n}, league_ly,
        {team: {overall, interior, tackle, end, stuffed, success, n}},
        {team: adj_stuffed})

    QB scrambles / designed keeps are excluded so the grade reflects the line,
    not the quarterback's legs. `success` may be missing on older frames; the
    success dict is empty in that case and callers fall back to line-yards only.
    Recency weights (if half_life > 0) apply to the opponent-adjusted point
    estimate only; n is still the raw play count.
    """
    ly_triples = []
    success_triples = []
    stuff_triples = []
    raw = defaultdict(lambda: {"ly": 0.0, "stuff": 0.0, "success": 0.0, "n_success": 0, "n": 0,
                               "interior": [0.0, 0], "tackle": [0.0, 0], "end": [0.0, 0]})
    cols = {c: (d[c].tolist() if c in d else None) for c in
            ("posteam", "defteam", "rushing_yards", "yards_gained",
             "rush_attempt", "qb_kneel", "qb_scramble", "run_location", "run_gap",
             "wp", "success", "passer_player_id", "rusher_player_id", "week")}
    n = len(d)
    end_week = _frame_end_week(d)
    hl = recency_half_life
    if hl is None:
        hl = RECENCY_HALF_LIFE
    for i in range(n):
        if cols["rush_attempt"] is None or _f(cols["rush_attempt"][i]) != 1:
            continue
        if cols["qb_kneel"] and _f(cols["qb_kneel"][i]) == 1:
            continue
        if _is_qb_run(
            cols["qb_scramble"][i] if cols["qb_scramble"] else None,
            cols["passer_player_id"][i] if cols["passer_player_id"] else None,
            cols["rusher_player_id"][i] if cols["rusher_player_id"] else None,
        ):
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
        w = _recency_weight(cols["week"][i] if cols["week"] else end_week, end_week, hl)
        ly_triples.append((o, dd, ly, w))
        stuff = 1.0 if y <= 0 else 0.0
        stuff_triples.append((o, dd, stuff, w))
        b = raw[o]
        b["ly"] += ly
        b["stuff"] += stuff
        b["n"] += 1
        if cols["success"] is not None:
            su = _f(cols["success"][i])
            if su is not None:
                success_triples.append((o, dd, su, w))
                b["success"] += su
                b["n_success"] += 1
        gap = _gap_of(cols["run_location"][i] if cols["run_location"] else None,
                      cols["run_gap"][i] if cols["run_gap"] else None)
        if gap:
            b[gap][0] += ly
            b[gap][1] += 1
    ly_adj, ncount, league = _alt_adjust(ly_triples)
    success_adj, _, _ = _alt_adjust(success_triples) if success_triples else ({}, {}, 0.0)
    stuff_adj, _, _ = _alt_adjust(stuff_triples) if stuff_triples else ({}, {}, 0.0)
    detail = {}
    for t, b in raw.items():
        row = {"n": b["n"]}
        if b["n"] > 0:
            row["line_yards"] = round(b["ly"] / b["n"], 3)
            row["stuffed_rate"] = round(b["stuff"] / b["n"] * 100.0, 1)
        if b["n_success"] > 0:
            row["success_rate"] = round(b["success"] / b["n_success"] * 100.0, 1)
        for g in ("interior", "tackle", "end"):
            s, c = b[g]
            row[g] = (s / c) if c else None
        detail[t] = row
    return ly_adj, success_adj, ncount, league, detail, stuff_adj


def _season_pass_metrics(d, recency_half_life=None):
    """Return per-team pass-protection metrics for one season frame.

    -> dict with opponent-adjusted pressure & sack values, plus context.
    Uses `was_pressure` when populated; otherwise falls back to `qb_hit`.
    Recency weights (if half_life > 0) apply to the opponent-adjusted point
    estimate and to TTT / rushers averages; n is still the raw play count.
    """
    cols = {c: (d[c].tolist() if c in d else None) for c in
            ("posteam", "defteam", "qb_dropback", "pass_attempt",
             "sack", "qb_hit", "was_pressure", "time_to_throw", "qb_spike", "wp",
             "week", "number_of_pass_rushers")}
    n = len(d)
    have_pressure = False
    if cols["was_pressure"] is not None:
        have_pressure = sum(1 for v in cols["was_pressure"] if _f(v) is not None) > 0.5 * max(1, n) * 0.1

    press_triples, sack_triples = [], []
    ttt = defaultdict(lambda: [0.0, 0.0])       # team -> [sum ttt*w, sum w]
    rushers = defaultdict(lambda: [0.0, 0.0])
    raw = defaultdict(lambda: {"press": 0.0, "sack": 0.0, "hit": 0.0, "n": 0})
    end_week = _frame_end_week(d)
    hl = recency_half_life
    if hl is None:
        hl = RECENCY_HALF_LIFE
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
        w = _recency_weight(cols["week"][i] if cols["week"] else end_week, end_week, hl)
        b = raw[o]
        b["n"] += 1
        s = 1.0 if sk else 0.0
        h = 1.0 if (cols["qb_hit"] and _f(cols["qb_hit"][i])) else 0.0
        b["sack"] += s
        b["hit"] += h
        sack_triples.append((o, dd, s, w))
        if have_pressure:
            p = _f(cols["was_pressure"][i])
            p = 1.0 if (p and p >= 1) else 0.0
            b["press"] += p
            press_triples.append((o, dd, p, w))
        else:
            b["press"] += h
            press_triples.append((o, dd, h, w))
        t2 = _f(cols["time_to_throw"][i]) if cols["time_to_throw"] else None
        if t2 is not None:
            ttt[o][0] += t2 * w
            ttt[o][1] += w
        ru = _f(cols["number_of_pass_rushers"][i]) if cols["number_of_pass_rushers"] else None
        if ru is not None:
            rushers[o][0] += ru * w
            rushers[o][1] += w

    press_adj, ncount, press_league = _alt_adjust(press_triples)
    sack_adj, _, sack_league = _alt_adjust(sack_triples)
    ttt_avg = {t: (v[0] / v[1]) for t, v in ttt.items() if v[1] > 0}
    rushers_avg = {t: (v[0] / v[1]) for t, v in rushers.items() if v[1] > 0}
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
        "n": ncount, "ttt": ttt_avg, "rushers": rushers_avg, "detail": detail,
        "pressure_source": "was_pressure" if have_pressure else "qb_hit",
    }


def _blend_run_index(ly_index, su_index, stuff_index=None, stuff_weight=None, round_to=1):
    """Blend line-yards + success (+ optional stuffed) percentiles.

    Stuffed weight `w` comes off the top; the remaining 1-w stays on LY/success
    in their current ratio. A team missing stuffed falls back to LY/success.
    """
    if stuff_weight is None:
        stuff_weight = RUN_STUFF_WEIGHT
    stuff_index = stuff_index or {}
    use_stuff = bool(stuff_weight) and bool(stuff_index)
    remaining = (1.0 - float(stuff_weight)) if use_stuff else 1.0
    ly_w = remaining * RUN_LY_WEIGHT
    su_w = remaining * RUN_SUCCESS_WEIGHT
    st_w = float(stuff_weight) if use_stuff else 0.0
    run_index = {}
    teams = set(ly_index) | set(su_index) | (set(stuff_index) if use_stuff else set())
    for t in teams:
        ly_i, su_i, st_i = ly_index.get(t), su_index.get(t), stuff_index.get(t)
        parts, wsum = [], 0.0
        if ly_i is not None and ly_w:
            parts.append(ly_w * ly_i)
            wsum += ly_w
        if su_i is not None and su_w:
            parts.append(su_w * su_i)
            wsum += su_w
        if st_i is not None and st_w:
            parts.append(st_w * st_i)
            wsum += st_w
        if not parts or wsum <= 0:
            continue
        v = sum(parts) / wsum
        run_index[t] = round(v, round_to) if round_to is not None else v
    return run_index


def _blend_pass_index(press_index, sack_index, round_to=1):
    pass_index = {}
    for t in set(press_index) | set(sack_index):
        p_idx, s_idx = press_index.get(t), sack_index.get(t)
        if p_idx is not None and s_idx is not None:
            v = PRESSURE_WEIGHT * p_idx + SACK_WEIGHT * s_idx
            pass_index[t] = round(v, round_to) if round_to is not None else v
        elif p_idx is not None:
            pass_index[t] = p_idx
        elif s_idx is not None:
            pass_index[t] = s_idx
    return pass_index


def _maybe_resid_rushers(value_by_team, rushers, enabled):
    """Second residualisation vs average pass-rushers faced. No-op if sparse."""
    if not enabled or not rushers or not value_by_team:
        return value_by_team
    usable = sum(1 for t in value_by_team
                 if t in rushers and rushers[t] is not None)
    if usable < 5:
        return value_by_team
    return _residualize(value_by_team, rushers)


def grade_indices(
    run_ly_adj, run_n, run_ly_prior, run_league,
    run_su_adj, run_su_prior,
    cur_pass, prior_pass,
    round_to=1,
    run_stuff_adj=None, run_stuff_prior=None,
    stuff_weight=None, rushers_resid=None, k_mult=None,
):
    """Regress to prior, residualise pass vs time-to-throw, percentile-scale.

    Shared by the builder and the backtest so a talent-shifted prior and the
    unshifted last-season prior take the same path through `_regress_to_prior`.
    Optional kwargs default to the module constants (legacy = stuffed weight 0,
    no rushers residual, K_MULT=1).
    """
    if stuff_weight is None:
        stuff_weight = RUN_STUFF_WEIGHT
    if rushers_resid is None:
        rushers_resid = PASS_RUSHERS_RESID
    km = 1.0 if k_mult is None else float(k_mult)
    run_k = RUN_PRIOR_K * km
    pass_k = PASS_PRIOR_K * km
    run_ly_final = _regress_to_prior(run_ly_adj, run_n, run_ly_prior, run_league, run_k)
    su_league = (sum(run_su_adj.values()) / len(run_su_adj)) if run_su_adj else 0.0
    run_su_final = (
        _regress_to_prior(run_su_adj, run_n, run_su_prior, su_league, run_k)
        if run_su_adj else {}
    )
    run_stuff_adj = run_stuff_adj or {}
    st_league = (sum(run_stuff_adj.values()) / len(run_stuff_adj)) if run_stuff_adj else 0.0
    run_st_final = (
        _regress_to_prior(run_stuff_adj, run_n, run_stuff_prior or {}, st_league, run_k)
        if run_stuff_adj else {}
    )
    prior_press = prior_pass["press_adj"] if prior_pass else None
    prior_sack = prior_pass["sack_adj"] if prior_pass else None
    press_final = _regress_to_prior(
        cur_pass["press_adj"], cur_pass["n"], prior_press,
        cur_pass["press_league"], pass_k)
    sack_final = _regress_to_prior(
        cur_pass["sack_adj"], cur_pass["n"], prior_sack,
        cur_pass["sack_league"], pass_k)
    press_resid = _residualize(press_final, cur_pass["ttt"])
    sack_resid = _residualize(sack_final, cur_pass["ttt"])
    rushers = cur_pass.get("rushers") or {}
    press_resid = _maybe_resid_rushers(press_resid, rushers, rushers_resid)
    sack_resid = _maybe_resid_rushers(sack_resid, rushers, rushers_resid)
    ly_index = _percentile_index(run_ly_final, higher_is_better=True)
    su_index = _percentile_index(run_su_final, higher_is_better=True) if run_su_final else {}
    st_index = (
        _percentile_index(run_st_final, higher_is_better=False) if run_st_final else {}
    )
    run_index = _blend_run_index(
        ly_index, su_index, stuff_index=st_index,
        stuff_weight=stuff_weight, round_to=round_to)
    press_index = _percentile_index(press_resid, higher_is_better=False)
    sack_index = _percentile_index(sack_resid, higher_is_better=False)
    pass_index = _blend_pass_index(press_index, sack_index, round_to=round_to)
    return {
        "run_index": run_index,
        "pass_index": pass_index,
        "press_index": press_index,
        "sack_index": sack_index,
        "run_ly_final": run_ly_final,
        "press_final": press_final,
        "sack_final": sack_final,
        "su_league": su_league,
        "st_league": st_league,
    }


def _apply_talent_to_priors(
    run_ly_prior, run_su_prior, prior_pass, talent_scores, weight,
    run_league, su_league, cur_pass,
    run_stuff_prior=None, st_league=0.0,
):
    """Shift last-season priors by the talent residual. Empty scores -> no-op."""
    from data_building.oline_talent_prior import shift_prior
    ly = shift_prior(run_ly_prior, talent_scores, weight, higher_is_better=True,
                     league=run_league)
    su = shift_prior(run_su_prior, talent_scores, weight, higher_is_better=True,
                     league=su_league)
    st = shift_prior(run_stuff_prior, talent_scores, weight, higher_is_better=False,
                     league=st_league)
    if not prior_pass:
        return ly, su, None, st
    shifted = dict(prior_pass)
    shifted["press_adj"] = shift_prior(
        prior_pass.get("press_adj"), talent_scores, weight,
        higher_is_better=False, league=cur_pass.get("press_league"))
    shifted["sack_adj"] = shift_prior(
        prior_pass.get("sack_adj"), talent_scores, weight,
        higher_is_better=False, league=cur_pass.get("sack_league"))
    return ly, su, shifted, st


def _load_availability_scale(season, through_week, pd, nfl, inputs=None):
    """Last-year OL snap share still available this week. Missing data -> {}."""
    if inputs is not None:
        return dict(inputs.get("scale") or {})
    try:
        from data_building.oline_talent_prior import compute_availability_scale
        return compute_availability_scale(season, through_week, pd, nfl) or {}
    except Exception as e:
        print(f"[oline_ratings] availability scale failed ({e}); skipping shrink")
        return {}


def _shrink_priors(run_ly, run_su, run_st, prior_pass, scale, run_league, su_league, st_league, cur_pass):
    if not scale:
        return run_ly, run_su, run_st, prior_pass
    run_ly = _apply_availability_shrink(run_ly, scale, run_league)
    run_su = _apply_availability_shrink(run_su, scale, su_league)
    run_st = _apply_availability_shrink(run_st, scale, st_league)
    if prior_pass:
        prior_pass = dict(prior_pass)
        prior_pass["press_adj"] = _apply_availability_shrink(
            prior_pass.get("press_adj"), scale, cur_pass.get("press_league", 0.0))
        prior_pass["sack_adj"] = _apply_availability_shrink(
            prior_pass.get("sack_adj"), scale, cur_pass.get("sack_league", 0.0))
    return run_ly, run_su, run_st, prior_pass


def build_oline_ratings(
    season: int,
    through_week: int | None = None,
    save: bool = True,
    use_talent_prior: bool = False,
    talent_weight: float | None = None,
    talent_inputs=None,
    recency_half_life=None,
    prior_y2_weight=None,
    run_stuff_weight=None,
    pass_rushers_resid=None,
    availability_shrink=None,
    availability_inputs=None,
    k_mult=None,
) -> dict:
    """Compute and (optionally) cache opponent-adjusted, regressed O-line ratings.

    `use_talent_prior` is off by default. When on, last-season priors are
    nudged by the open-data roster/draft residual before `_regress_to_prior`;
    the unshifted measurement is stored under `*_realized` keys. Missing
    roster/draft/snap data falls back to the unshifted prior (today's
    behavior). `talent_inputs` lets tests / the backtest inject already-parsed
    frames without hitting the network.

    Pipeline knobs (`recency_half_life`, `prior_y2_weight`, `run_stuff_weight`,
    `pass_rushers_resid`, `availability_shrink`, `k_mult`) default to the
    module constants. Those stay at the legacy values until
    `oline_backtest.sweep_results_pipeline` finds a real lift.
    """
    import pandas as pd
    try:
        import nfl_data_py as nfl
    except Exception:
        nfl = None

    recency = RECENCY_HALF_LIFE if recency_half_life is None else recency_half_life
    y2w = PRIOR_Y2_WEIGHT if prior_y2_weight is None else prior_y2_weight
    stuff_w = RUN_STUFF_WEIGHT if run_stuff_weight is None else run_stuff_weight
    rush_resid = PASS_RUSHERS_RESID if pass_rushers_resid is None else pass_rushers_resid
    avail_on = AVAILABILITY_SHRINK if availability_shrink is None else availability_shrink
    km = K_MULT if k_mult is None else k_mult

    cur_raw = _prep_season(_load_pbp_year(season, pd, nfl), pd, season, through_week)
    if cur_raw is None or cur_raw.empty:
        print(f"[oline_ratings] no current-season pbp for {season}")
        return {}
    prior_raw = _prep_season(_load_pbp_year(season - 1, pd, nfl), pd, season - 1)
    y2_raw = None
    if y2w:
        y2_raw = _prep_season(_load_pbp_year(season - 2, pd, nfl), pd, season - 2)

    # --- RUN ---
    # Recency applies to the current-season point estimate only.
    run_ly_adj, run_su_adj, run_n, run_league, run_detail, run_st_adj = (
        _season_run_metrics(cur_raw, recency_half_life=recency))
    run_ly_prior, run_su_prior, run_st_prior = {}, {}, {}
    if prior_raw is not None and not prior_raw.empty:
        run_ly_prior, run_su_prior, _, _, _, run_st_prior = (
            _season_run_metrics(prior_raw, recency_half_life=0.0))
    if y2w and y2_raw is not None and not y2_raw.empty:
        ly2, su2, _, _, _, st2 = _season_run_metrics(y2_raw, recency_half_life=0.0)
        run_ly_prior = _mix_prior(run_ly_prior, ly2, y2w)
        run_su_prior = _mix_prior(run_su_prior, su2, y2w)
        run_st_prior = _mix_prior(run_st_prior, st2, y2w)
    su_league = (sum(run_su_adj.values()) / len(run_su_adj)) if run_su_adj else 0.0
    st_league = (sum(run_st_adj.values()) / len(run_st_adj)) if run_st_adj else 0.0

    # --- PASS ---
    cur_pass = _season_pass_metrics(cur_raw, recency_half_life=recency)
    prior_pass = (
        _season_pass_metrics(prior_raw, recency_half_life=0.0)
        if (prior_raw is not None and not prior_raw.empty) else None
    )
    if y2w and y2_raw is not None and not y2_raw.empty:
        y2_pass = _season_pass_metrics(y2_raw, recency_half_life=0.0)
        if prior_pass:
            mixed = dict(prior_pass)
            mixed["press_adj"] = _mix_prior(prior_pass.get("press_adj"), y2_pass.get("press_adj"), y2w)
            mixed["sack_adj"] = _mix_prior(prior_pass.get("sack_adj"), y2_pass.get("sack_adj"), y2w)
            prior_pass = mixed
        else:
            prior_pass = y2_pass

    if avail_on:
        if availability_inputs is not None:
            scale = dict(availability_inputs.get("scale") or {})
        else:
            scale = _load_availability_scale(season, through_week, pd, nfl)
        run_ly_prior, run_su_prior, run_st_prior, prior_pass = _shrink_priors(
            run_ly_prior, run_su_prior, run_st_prior, prior_pass, scale,
            run_league, su_league, st_league, cur_pass)

    grade_kw = dict(
        run_stuff_adj=run_st_adj, run_stuff_prior=run_st_prior,
        stuff_weight=stuff_w, rushers_resid=rush_resid, k_mult=km,
    )
    realized = grade_indices(
        run_ly_adj, run_n, run_ly_prior, run_league,
        run_su_adj, run_su_prior, cur_pass, prior_pass, round_to=1, **grade_kw)

    talent_pack = None
    graded = realized
    tw = TALENT_PRIOR_W if talent_weight is None else talent_weight
    if use_talent_prior:
        from data_building.oline_talent_prior import compute_oline_talent_prior
        talent_pack = compute_oline_talent_prior(
            season, pd, nfl,
            w_continuity=TALENT_W_CONTINUITY,
            w_draft=TALENT_W_DRAFT,
            w_veteran=TALENT_W_VETERAN,
            inputs=talent_inputs,
        )
        if talent_pack and talent_pack.get("scores") and tw:
            ly_p, su_p, pass_p, st_p = _apply_talent_to_priors(
                run_ly_prior, run_su_prior, prior_pass,
                talent_pack["scores"], tw,
                run_league, su_league, cur_pass,
                run_stuff_prior=run_st_prior, st_league=st_league)
            graded = grade_indices(
                run_ly_adj, run_n, ly_p, run_league,
                run_su_adj, su_p, cur_pass, pass_p, round_to=1,
                run_stuff_adj=run_st_adj, run_stuff_prior=st_p,
                stuff_weight=stuff_w, rushers_resid=rush_resid, k_mult=km)
        else:
            # Missing roster/draft/snap data: identical to the flag-off path.
            talent_pack = None
            graded = realized

    run_index = graded["run_index"]
    pass_index = graded["pass_index"]
    run_ly_final = graded["run_ly_final"]
    press_final = graded["press_final"]

    # Descriptive per-gap run indices (opponent-unadjusted; label as secondary).
    gap_index = {}
    for g in ("interior", "tackle", "end"):
        gv = {t: run_detail[t][g] for t in run_detail if run_detail[t].get(g) is not None}
        gap_index[g] = _percentile_index(gv, higher_is_better=True)

    # --- ASSEMBLE ---
    teams = set(run_ly_final) | set(press_final)
    realized_run = realized["run_index"]
    realized_pass = realized["pass_index"]
    label_realized = bool(talent_pack) and graded is not realized
    ratings = {}
    for t in teams:
        rd = run_detail.get(t, {})
        pd_ = cur_pass["detail"].get(t, {})
        n_rush = run_n.get(t, rd.get("n", 0))
        n_pass = cur_pass["n"].get(t, pd_.get("n", 0))
        if n_rush < MIN_TEAM_PLAYS and n_pass < MIN_TEAM_PLAYS:
            continue
        run_idx = run_index.get(t)
        pass_idx = pass_index.get(t)

        row = {"n_rush": n_rush, "n_pass": n_pass}
        if "line_yards" in rd:
            row["line_yards"] = rd["line_yards"]
            row["stuffed_rate"] = rd["stuffed_rate"]
        if "success_rate" in rd:
            row["success_rate"] = rd["success_rate"]
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
        if label_realized:
            # Keep the results-based measurement distinguishable from the
            # talent-shifted projection sitting in the primary keys.
            rr, rp = realized_run.get(t), realized_pass.get(t)
            if rr is not None:
                row["run_block_realized"] = rr
            if rp is not None:
                row["pass_block_realized"] = rp
            if rr is not None and rp is not None:
                row["composite_realized"] = round(
                    PASS_BLOCK_WEIGHT * rp + RUN_BLOCK_WEIGHT * rr, 1)
            elif rr is not None:
                row["composite_realized"] = rr
            elif rp is not None:
                row["composite_realized"] = rp
        ratings[t] = row

    out = {
        "season": season,
        "through_week": through_week,
        "seasons_used": sorted(
            {season}
            | ({season - 1} if run_ly_prior or prior_pass else set())
            | ({season - 2} if y2w else set())
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pressure_source": cur_pass["pressure_source"],
        "n_run_plays": int(sum(run_n.values())),
        "n_pass_plays": int(sum(cur_pass["n"].values())),
        "weights": {
            "pass_block": PASS_BLOCK_WEIGHT, "run_block": RUN_BLOCK_WEIGHT,
            "pressure": PRESSURE_WEIGHT, "sack": SACK_WEIGHT,
            "run_ly": RUN_LY_WEIGHT, "run_success": RUN_SUCCESS_WEIGHT,
            "run_stuff": stuff_w,
            "run_prior_k": RUN_PRIOR_K, "pass_prior_k": PASS_PRIOR_K,
            "k_mult": km,
            "recency_half_life": recency,
            "prior_y2_weight": y2w,
            "pass_rushers_resid": bool(rush_resid),
            "availability_shrink": bool(avail_on),
        },
        "ratings": ratings,
    }
    if label_realized:
        meta = talent_pack.get("meta") or {}
        out["weights"]["talent_prior_w"] = tw
        out["weights"]["talent_w_continuity"] = TALENT_W_CONTINUITY
        out["weights"]["talent_w_draft"] = TALENT_W_DRAFT
        out["weights"]["talent_w_veteran"] = TALENT_W_VETERAN
        out["talent_prior"] = {
            "enabled": True,
            "applied_to_ratings": True,
            "kind": "projection",
            "note": (
                "Offseason roster/draft residual applied to the last-season "
                "prior only. Decays with n_cur via RUN_PRIOR_K / PASS_PRIOR_K. "
                "Not a measurement; see *_realized for the results-based grade."
            ),
            "sources": meta.get("sources"),
            "omitted": meta.get("omitted"),
            "roster_source": meta.get("roster_source"),
            "pick_chart": meta.get("pick_chart"),
            "weights_used": talent_pack.get("weights_used"),
            "teams": talent_pack.get("detail"),
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
    argv = sys.argv[1:]
    flags = {a for a in argv if a.startswith("-")}
    args = [a for a in argv if not a.startswith("-")]
    yr = int(args[0]) if args else datetime.now().year
    tw = int(args[1]) if len(args) > 1 else None
    use_tp = "--talent-prior" in flags
    res = build_oline_ratings(yr, tw, use_talent_prior=use_tp)
    r = res.get("ratings", {})
    print(f"[oline_ratings] season={yr} teams_rated={len(r)} "
          f"pressure_source={res.get('pressure_source')} "
          f"talent_prior={bool(res.get('talent_prior'))} "
          f"run_plays={res.get('n_run_plays')} pass_plays={res.get('n_pass_plays')} "
          f"-> {out_path(yr)}")
    rows = sorted(r.items(), key=lambda kv: kv[1].get("composite", 0), reverse=True)
    for t, row in rows:
        extra = ""
        if row.get("composite_realized") is not None:
            extra = f"  realized={row.get('composite_realized')}"
        print(f"  {t:>3} comp={row.get('composite'):>5}  pass={row.get('pass_block'):>5} "
              f"run={row.get('run_block'):>5}  press%={row.get('pressure_rate')} "
              f"sack%={row.get('sack_rate')} ttt={row.get('avg_time_to_throw')}{extra}")
