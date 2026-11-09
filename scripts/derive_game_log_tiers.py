#!/usr/bin/env python3
"""Derive the game-log wash tier cutoffs from the real weekly-stats pool.

The digestible game log colors stat cells by where a game sits in the
distribution of real player games at the position (the same percentile
spirit as utils/nfl.py's sched_rank_color, which colors Schedule Rankings
by quartile). This script is the provenance for the _GL_WASH constants in
static/player_modal.js: rerun it against a fuller cache and the tables
can be refreshed without hand-tuning. `--js` prints the JS block itself.

Pool: Sleeper weekly stats cache files (the same files the game log
itself reads), seasons 2025 (weeks 1-18) + 2026 (weeks present), rows
filtered with the site's appearance rule (a row carrying only bookkeeping
keys -- gms_active / pos_rank_* / tm_*_snp -- is a did-not-play, not a
game). Involvement floors keep pure garbage-time rows out of a position's
pool:

  QB: pass_att >= 10          RB: rush_att + rec >= 8 (touches)
  WR: rec_tgt + rush_att >= 3 TE: rec_tgt + rush_att >= 3

Rate stats add their own sample floors inside the base pool: rush Y/A
needs 8+ carries (3+ for QB/WR/TE), YPT needs 3+ targets, pass Y/A uses
the QB base pool. SNP% needs off_snp/tm_off_snp on the row; Car% / TS%
divide by the TEAM_<abbr> totals line in the same week file (player team
from players_index.json, so a mid-season trade can misattribute a week --
accepted noise for a percentile table).

Cutoffs: step 1 at the pool's p90, step 2 at p75, step 3 at p25, step 4
at p10. Rate / share stats use plain quantiles (linear interpolation)
on both tails. Counting stats take their good tail from quantiles but
their bad tail from each value's mid-percentile ((games below + half
the games equal) / pool): with a fifth of the pool massed at zero, a
raw q10 lands inside the zero mass and would call a 0-yard game a
bottom-decile game when it is really a ~20th-percentile one (a
0-catch game, rarer, really is bottom-decile -- the mid-percentile
sees the difference, the quantile cannot). Touchdown and red-zone
columns are top-tail only (a zero there is a normal game, not a bad
one) and use mid-percentiles on the good tail for the same reason.
Interceptions invert (fewer is better): mid-percentiles read the
other way (step 1 at p10, step 2 at p25, step 3 at p75, step 4 at p90).

Two judgment calls, both about stats whose bad tail is not a bad game:
a QB's or WR/TE's rushing counts are bonus production, so they take the
top-tail treatment too (a 0-carry game for a pocket QB is not a red
game); and QB SNP% gets no table at all -- the QB snap distribution is
degenerate (100% or benched), so its percentiles carry no signal.
Counting-stat cutoffs at or below zero are dropped (a threshold of 0 on
a count would tier "did anything at all"), except the bad tail of a
position's core volume stats, where 0 targets on an 8+-touch day really
is a bottom-decile game. A stat whose pool has fewer than 30 qualifying
games gets no table (its cells render plain).

xFP is NOT derived here (its weekly pool lives in the Postgres advanced
store, not this cache) and neither are BTK / YBC (PFR advstats weekly
CSVs cover only 2026 weeks 1-3 locally) -- the renderer uses Kaedon's
fixed xFP bands and leaves BTK/YBC unwashed rather than faking a pool.

Usage: python3 scripts/derive_game_log_tiers.py [--cache-dir DIR] [--js]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

BOOKKEEPING_KEYS = frozenset({
    "gms_active",
    "pos_rank_ppr", "pos_rank_half_ppr", "pos_rank_std",
    "tm_off_snp", "tm_def_snp", "tm_st_snp",
})

SEASONS = (2025, 2026)
POSITIONS = ("QB", "RB", "WR", "TE")

# Stats whose columns the game log actually renders per position, and the
# only ones emitted into the JS tables.
RENDERED = {
    "QB": ("patt", "pcmp", "pyd", "pypa", "ptd", "pint",
           "car", "ryd", "rypa", "rtd"),
    "RB": ("car", "ryd", "rypa", "rtd", "tar", "rec", "recyd", "rectd",
           "ypt", "carpct", "tspct", "rzcar", "rztar", "snppct"),
    "WR": ("tar", "rec", "recyd", "rectd", "ypt", "tspct", "rztar",
           "car", "ryd", "rypa", "rtd", "snppct"),
    "TE": ("tar", "rec", "recyd", "rectd", "ypt", "tspct", "rztar",
           "car", "ryd", "rypa", "rtd", "snppct"),
}
# Top-tail-only stats (no bad tail): touchdowns + red-zone everywhere,
# plus the rushing counts of positions whose rushing is bonus production.
TOP_TAIL = {"ptd", "rtd", "rectd", "rzcar", "rztar"}
BONUS_TOP_TAIL = {
    "QB": {"car", "ryd", "rtd", "rzcar"},
    "WR": {"car", "ryd", "rtd", "rzcar"},
    "TE": {"car", "ryd", "rtd", "rzcar"},
}
COUNTS = {"patt", "pcmp", "pyd", "car", "ryd", "tar", "rec", "recyd",
          "ptd", "rtd", "rectd", "rzcar", "rztar", "pint"}


def is_appearance(row) -> bool:
    if not isinstance(row, dict):
        return False
    return any(v for k, v in row.items() if k not in BOOKKEEPING_KEYS)


def quantile(sorted_vals, q):
    """Linear-interpolation quantile, numpy's default method."""
    n = len(sorted_vals)
    if n == 1:
        return float(sorted_vals[0])
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _round(v, nd):
    return None if v is None else round(v, nd)


def quantile_cutoffs(vals, top_tail_only, nd, midpct_bad_tail=False):
    s = sorted(vals)
    q = lambda p: quantile(s, p)  # noqa: E731
    if top_tail_only:
        return [_round(q(0.90), nd), _round(q(0.75), nd), None, None]
    if midpct_bad_tail:
        bad = midpct_cutoffs(vals, top_tail_only=False, nd=nd)
        return [_round(q(0.90), nd), _round(q(0.75), nd), bad[2], bad[3]]
    return [_round(q(0.90), nd), _round(q(0.75), nd),
            _round(q(0.25), nd), _round(q(0.10), nd)]


def midpct_cutoffs(vals, top_tail_only, nd):
    """Thresholds from each distinct value's mid-percentile:
    (games below + half the games equal) / pool * 100."""
    s = sorted(vals)
    n = len(s)
    mp = {}
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        mp[s[i]] = (i + 0.5 * (j - i)) / n * 100
        i = j
    items = sorted(mp.items())
    t1 = min((v for v, p in items if p >= 90), default=None)
    t2 = min((v for v, p in items if p >= 75), default=None)
    if top_tail_only:
        return [_round(t1, nd), _round(t2, nd), None, None]
    t3 = max((v for v, p in items if p <= 25), default=None)
    t4 = max((v for v, p in items if p <= 10), default=None)
    return [_round(t1, nd), _round(t2, nd), _round(t3, nd), _round(t4, nd)]


def inverted_cutoffs(vals, nd):
    """Interceptions: fewer is better. Steps from mid-percentiles read in
    reverse: step 1 <= p10 value, step 2 <= p25, step 3 >= p75, step 4 >= p90."""
    s = sorted(vals)
    n = len(s)
    mp = {}
    i = 0
    while i < n:
        j = i
        while j < n and s[j] == s[i]:
            j += 1
        mp[s[i]] = (i + 0.5 * (j - i)) / n * 100
        i = j
    items = sorted(mp.items())
    t1 = max((v for v, p in items if p <= 10), default=None)
    t2 = max((v for v, p in items if p <= 25), default=None)
    t3 = min((v for v, p in items if p >= 75), default=None)
    t4 = min((v for v, p in items if p >= 90), default=None)
    return [_round(t1, nd), _round(t2, nd), _round(t3, nd), _round(t4, nd)]


def build_pools(cache):
    with open(os.path.join(cache, "players_index.json")) as f:
        index = json.load(f)
    pos_of = {str(pid): (m.get("pos") or "").upper()
              for pid, m in index.items() if isinstance(m, dict)}
    team_of = {str(pid): (m.get("team") or "").upper()
               for pid, m in index.items() if isinstance(m, dict)}

    pools = {p: {} for p in POSITIONS}

    def add(pos, stat, v):
        if v is not None:
            pools[pos].setdefault(stat, []).append(float(v))

    files = []
    for season in SEASONS:
        files += sorted(glob.glob(os.path.join(
            cache, "sleeper_stats", f"sleeper_stats_s{season}_w*.json")))
    if not files:
        raise SystemExit(f"no sleeper week files under {cache}/sleeper_stats")

    for path in files:
        with open(path) as f:
            week = json.load(f) or {}
        for pid, row in week.items():
            pid = str(pid)
            pos = pos_of.get(pid)
            if pos not in POSITIONS or not is_appearance(row):
                continue
            g = lambda k: row.get(k) or 0  # noqa: E731 (missing = 0 in this feed)
            team_line = week.get(f"TEAM_{team_of.get(pid, '')}") or {}
            if pos == "QB":
                if g("pass_att") < 10:
                    continue
            elif pos == "RB":
                if g("rush_att") + g("rec") < 8:
                    continue
            elif g("rec_tgt") + g("rush_att") < 3:  # WR / TE
                continue

            add(pos, "patt", g("pass_att"))
            add(pos, "pcmp", g("pass_cmp"))
            add(pos, "pyd", g("pass_yd"))
            if g("pass_att") >= 10:
                add(pos, "pypa", g("pass_yd") / g("pass_att"))
            add(pos, "ptd", g("pass_td"))
            add(pos, "pint", g("pass_int"))
            add(pos, "car", g("rush_att"))
            add(pos, "ryd", g("rush_yd"))
            if g("rush_att") >= (8 if pos == "RB" else 3):
                add(pos, "rypa", g("rush_yd") / g("rush_att"))
            add(pos, "rtd", g("rush_td"))
            add(pos, "tar", g("rec_tgt"))
            add(pos, "rec", g("rec"))
            add(pos, "recyd", g("rec_yd"))
            add(pos, "rectd", g("rec_td"))
            if g("rec_tgt") >= 3:
                add(pos, "ypt", g("rec_yd") / g("rec_tgt"))
            if row.get("off_snp") and row.get("tm_off_snp"):
                add(pos, "snppct", row["off_snp"] / row["tm_off_snp"] * 100)
            if team_line.get("rush_att"):
                add(pos, "carpct", g("rush_att") / team_line["rush_att"] * 100)
            if team_line.get("pass_att"):
                add(pos, "tspct", g("rec_tgt") / team_line["pass_att"] * 100)
            add(pos, "rzcar", g("rush_rz_att"))
            add(pos, "rztar", g("rec_rz_tgt"))
    return pools


def derive(pools):
    out = {}
    for pos in POSITIONS:
        out[pos] = {}
        for stat in RENDERED[pos]:
            vals = pools[pos].get(stat) or []
            if len(vals) < 30:
                continue  # too thin to tier honestly; cells render plain
            nd = 1 if stat in COUNTS or stat in ("snppct", "carpct", "tspct") else 2
            top_tail = stat in TOP_TAIL or stat in BONUS_TOP_TAIL.get(pos, set())
            if stat == "pint":
                c = inverted_cutoffs(vals, nd)
            elif stat in TOP_TAIL:
                c = midpct_cutoffs(vals, top_tail_only=True, nd=nd)
            else:
                c = quantile_cutoffs(vals, top_tail_only=top_tail, nd=nd,
                                     midpct_bad_tail=stat in COUNTS)
            if stat in COUNTS:
                # A good-tail cutoff at/below 0 tiers "did anything at
                # all"; a bad-tail cutoff below 0 is unreachable noise.
                if c[0] is not None and c[0] <= 0:
                    c[0] = None
                if c[1] is not None and c[1] <= 0:
                    c[1] = None
                if c[3] is not None and c[3] < 0:
                    c[3] = None
            if all(v is None for v in c):
                continue
            out[pos][stat] = {"n": len(vals), "cutoffs": c}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache"))
    ap.add_argument("--js", action="store_true",
                    help="print the _GL_WASH JS block instead of JSON")
    args = ap.parse_args()

    tables = derive(build_pools(args.cache_dir))
    if not args.js:
        print(json.dumps(tables, indent=1))
        return 0
    # JS tables are hand-checked into player_modal.js; pool sizes ride
    # along in the comment there, not in the emitted literal.
    def _fmt(v):
        if v is None:
            return "null"
        return str(int(v)) if float(v) == int(v) else repr(v)

    print("const _GL_WASH = {")
    for pos in POSITIONS:
        entries = ", ".join(
            f"{stat}: [{', '.join(_fmt(v) for v in t['cutoffs'])}]"
            for stat, t in tables[pos].items())
        print(f"  {pos}: {{ {entries} }},")
    print("};")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
