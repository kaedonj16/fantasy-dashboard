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
    # Mixed portfolio of current leagues: --history reaches prior completed
    # seasons, --auto-type grades each on its own 1QB/SF + startup/redraft/rookie
    # basis (detected from Sleeper). No need to sort leagues by format yourself.
    python -m data_building.run_draft_backtest \
        --league <id1> <id2> ... --season 2026 --history --auto-type

    # Or force one basis for all leagues, listing completed-season IDs directly:
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

from utils.draft_grade import clamp01 as _clamp01
from utils.pick_score import PS_WEIGHTS, ps_tier_of
from utils.tier_thresholds import compute_tier_thresholds
from data_building.draft_grade_backtest import (
    calibration_bins,
    candidate_grid,
    correlate_grades_to_finish,
    letter_calibration,
    load_sleeper_samples,
    load_multiyear_samples,
    pick_score_by_round,
    sweep,
)


def _fmt(r) -> str:
    return "   n/a" if r is None else f"{r:+.3f}"


def _report_group(title: str, samples, method: str, seed_type=None, top: int = 8) -> None:
    """Print a group's shipped-weight correlation + a sweep seeded from the right
    table. ``seed_type`` picks the PS_WEIGHTS row to nudge around (None -> startup,
    for the mixed 'ALL' view where no single per-type table applies)."""
    n = len(samples)
    if n < 8:
        print(f"== {title}: {n} teams — too few to trust, skipping.\n")
        return
    base_r = correlate_grades_to_finish(samples, method=method)
    print(f"== {title}: {n} teams — shipped {method} r = {_fmt(base_r)}")
    seed = PS_WEIGHTS.get(seed_type or "startup", PS_WEIGHTS["startup"])
    ranked = sweep(samples, candidate_grid(seed), method=method)
    for label, _w, r in ranked[:top]:
        print(f"     {_fmt(r)}  {label}")

    # Grade-quintile calibration: is the raw grade SCALE monotonic with success?
    bins = calibration_bins(samples, n_bins=5)
    if bins:
        print("     grade quintile -> mean outcome (want it to climb):")
        for b in bins:
            print(f"       Q{b['bin']} (n={b['n']:>3}): outcome {b['outcome_mean']:.2f}")

    # Letter calibration: does the actual A-F letter (real curve, per league)
    # track outcomes? If A rows don't beat B/C, shift the curve anchor/bands.
    letters = letter_calibration(samples)
    if letters:
        print("     letter grade -> mean outcome (should fall A -> F):")
        for L in letters:
            print(f"       {L['letter']:>2} (n={L['n']:>3}): outcome {L['outcome_mean']:.2f}")

    # Per-round pick-score scale: should be roughly FLAT if depth-normalization
    # is calibrated. A slope means late rounds are under/over-scored.
    rounds = pick_score_by_round(samples)
    if rounds:
        print("     avg pick score by round (want it ~flat):")
        for R in rounds:
            bar = "#" * int(round(R["score_mean"] / 4))
            print(f"       R{R['round']:>2} (n={R['n']:>4}): {R['score_mean']:5.1f}  {bar}")
    print()


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


def make_value_fn_factory(num_teams: int):
    """A cached ``factory(is_sf, draft_type) -> value_fn`` for mixed portfolios.

    Each (is_sf, draft_type) combo builds its own valuation basis (value column,
    positional replacement, ADP field, tier thresholds) once, reusing the app's
    memoized player pool underneath. Lets one backtest run correctly grade a mix
    of 1QB/SF and startup/redraft/rookie leagues.
    """
    cache: dict = {}

    def factory(is_sf: bool, draft_type: str):
        key = (bool(is_sf), str(draft_type))
        if key not in cache:
            cache[key] = build_value_fn(draft_type, is_sf, num_teams)
        return cache[key]

    return factory


def _run_multiyear(args, dtype: str, decay: float) -> int:
    """Backtest one draft type against a multi-year outcome (sustained finish)."""
    factory = make_value_fn_factory(args.num_teams)
    samples = []
    for anchor in args.league:
        samples.extend(load_multiyear_samples(
            anchor, args.season, value_fn_factory=factory, draft_types=(dtype,),
            num_teams=args.num_teams, decay=decay))
    if not samples:
        print(f"No {dtype} drafts with enough forward history found. Needs current "
              f"league ID(s) whose previous_league_id chain includes a {dtype} season "
              "with >=2 later completed seasons — and a reachable app/network/DB.")
        return 1
    avg_seasons = sum(s.meta.get("seasons", 0) for s in samples) / len(samples)
    print(f"Loaded {len(samples)} {dtype} teams "
          f"(avg {avg_seasons:.1f} seasons of forward outcome each) "
          f"across {len(set(s.meta.get('league_id') for s in samples))} {dtype} drafts.\n")
    _report_group(f"{dtype} vs multi-year finish", samples, args.method, seed_type=dtype)
    print("Outcome = decayed mean of each manager's per-season finish from the draft "
          "year forward (champion 1.0, last 0.0). A nudge that beats 'base' here is a "
          f"{dtype}-weight signal that same-season points-for couldn't see.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--league", nargs="+", required=True, metavar="LEAGUE_ID",
                    help="One or more Sleeper league IDs. With --history, pass your "
                         "CURRENT league ID(s) and prior seasons are auto-discovered.")
    ap.add_argument("--season", type=int, required=True,
                    help="Season of the --league IDs (the anchor for --history).")
    ap.add_argument("--history", action="store_true",
                    help="Walk previous_league_id from each --league ID and backtest "
                         "every PRIOR completed season (the anchor season itself is "
                         "excluded - it has no final standings yet).")
    ap.add_argument("--auto-type", action="store_true",
                    help="Detect each league's format (1QB/SF and startup/redraft/"
                         "rookie) from Sleeper and grade it on the right basis. Use "
                         "this for a MIXED portfolio; it overrides --sf/--draft-type.")
    ap.add_argument("--sf", action="store_true",
                    help="Force Superflex valuation for all leagues (ignored with --auto-type).")
    ap.add_argument("--draft-type", default="startup", choices=["startup", "redraft", "rookie"],
                    help="Force draft type for all leagues (ignored with --auto-type).")
    ap.add_argument("--num-teams", type=int, default=12)
    ap.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    ap.add_argument("--startup-multiyear", action="store_true",
                    help="Grade STARTUP drafts against a multi-year outcome (how each "
                         "manager's team finished across the draft season and every "
                         "later one) instead of noisy same-season points-for. Pass your "
                         "CURRENT --league ID(s); walks the full history itself.")
    ap.add_argument("--rookie-multiyear", action="store_true",
                    help="Same multi-year outcome, for ROOKIE drafts - a rookie class "
                         "pays off in years 2-3, so year-1 points is the wrong yardstick "
                         "(uses a higher forward decay so later seasons count more).")
    args = ap.parse_args(argv)

    if args.startup_multiyear:
        return _run_multiyear(args, "startup", decay=0.75)
    if args.rookie_multiyear:
        return _run_multiyear(args, "rookie", decay=0.9)

    league_ids = list(args.league)
    if args.history:
        try:
            from dashboard_services.api import build_league_history_map
        except Exception as e:
            print(f"--history needs a reachable app/network ({e}).", file=sys.stderr)
            return 1
        discovered: list[str] = []
        for anchor in args.league:
            hist = build_league_history_map("sleeper", anchor, args.season) or {}
            # Prior completed seasons only - drop the anchor (in-progress) season.
            for yr, lid in sorted(hist.items()):
                if int(yr) < int(args.season):
                    discovered.append(str(lid))
        # Dedup, preserve order.
        league_ids = list(dict.fromkeys(discovered))
        if not league_ids:
            print("No prior completed seasons found via previous_league_id. "
                  "This may be a first-year league, or the chain isn't set.")
            return 1
        print(f"--history expanded {len(args.league)} current league(s) -> "
              f"{len(league_ids)} prior completed season(s).\n")

    if args.auto_type:
        samples = load_sleeper_samples(
            league_ids, args.season, value_fn_factory=make_value_fn_factory(args.num_teams),
            draft_type=args.draft_type, is_sf=args.sf, num_teams=args.num_teams,
            auto_detect=True,
        )
    else:
        value_fn = build_value_fn(args.draft_type, args.sf, args.num_teams)
        samples = load_sleeper_samples(
            league_ids, args.season, value_fn=value_fn,
            draft_type=args.draft_type, is_sf=args.sf, num_teams=args.num_teams,
        )
    if not samples:
        print("No gradeable teams found — check league IDs, network, and DATABASE_URL.")
        return 1

    n_picks = sum(len(s.picks) for s in samples)
    print(f"Loaded {len(samples)} teams ({n_picks} graded picks) "
          f"across {len(set(s.meta.get('league_id') for s in samples))} leagues.")
    if args.auto_type:
        sf_n = sum(1 for s in samples if s.meta.get("is_sf"))
        types = {}
        for s in samples:
            types[s.meta.get("draft_type")] = types.get(s.meta.get("draft_type"), 0) + 1
        type_str = ", ".join(f"{k}:{v}" for k, v in sorted(types.items()))
        print(f"Auto-detected: {sf_n} SF / {len(samples) - sf_n} 1QB teams; types [{type_str}].")
    print()

    _report_group("ALL leagues (mixed types share one uniform table — see note)",
                  samples, args.method, seed_type=None)

    # Per-draft-type breakdown: each type uses a different shipped weight table
    # (rookie leans on youth, startup doesn't), so a single uniform sweep across a
    # mixed pool grades against a handicapped baseline. Sweeping WITHIN a type,
    # seeded from THAT type's shipped table, is the apples-to-apples comparison
    # that can actually justify a weight change.
    by_type: dict = {}
    for s in samples:
        by_type.setdefault(s.meta.get("draft_type") or "?", []).append(s)
    if len(by_type) > 1:
        for dtype in sorted(by_type):
            _report_group(f"{dtype} leagues only", by_type[dtype], args.method, seed_type=dtype)

    print("Read: a nudge only counts if it beats that group's 'base' by a clear, "
          "consistent margin. 'base' IS the shipped table for that type, so "
          "base-at-top means the current weights are already best. Fold a real "
          "winner into PS_WEIGHTS AND static/pick_score.js (parity-pinned).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
