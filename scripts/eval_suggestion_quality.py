#!/usr/bin/env python3
"""Evaluate the archetype trade-suggestion engine on synthetic leagues.

The consolidate / distribute / need changes are principled but unproven, so this
harness generates a spread of random leagues (loaded teams, one-stud teams, thin
teams, balanced teams), runs every archetype, and reports:

  - how often each archetype produces suggestions at all (coverage),
  - the net win-probability impact of the suggested trades (from the analytical
    model, so no sim state is needed), and
  - how often each archetype's own guardrails hold (invariant pass rate):
      consolidate never ships a lone stud, always trades up past its send
      headliner, and never bundles rocks; distribute always returns several
      usable pieces for one stud.

Run:
    python -m scripts.eval_suggestion_quality [n_leagues] [seed]

This is a model check, not a real-outcome backtest (we have no historical trade
ground truth) - it verifies the engine does what each strategy claims across a
broad input space, and quantifies the impact it's suggesting.
"""
from __future__ import annotations

import random
import statistics
import sys
from typing import Dict, List, Tuple

from dashboard_services import archetype_engine as ae

_POS = ["QB", "RB", "WR", "TE"]
_ARCHES = ["contending", "rebuilding", "consolidate", "distribute"]
# Mirror the engine's lone-stud rule so the eval can check it independently.
_LONE_STUD_GAP = 1.5
_UPGRADE = 1.18
_PIECE_FRAC = 0.10


def _mk(pid: str, name: str, pos: str, val: float, age: int) -> dict:
    return {"id": pid, "name": name, "position": pos, "team": "FA",
            "age": age, "value": round(val, 1), "sf_value": round(val, 1),
            "redraft_value_1qb": round(val * random.uniform(0.6, 1.1), 1),
            "redraft_value_sf": round(val * random.uniform(0.6, 1.1), 1),
            "pos_rank_label": f"{pos}1", "rank_change_7d": 0}


def _team_players(rng: random.Random, tid: int, profile: str) -> List[dict]:
    """Build one roster with a value profile: 'loaded' (several studs), 'lone'
    (one big stud + depth), 'thin' (all mid/low), or 'balanced'."""
    out: List[dict] = []
    def add(pos, val):
        out.append(_mk(f"t{tid}_{pos}{len(out)}", f"T{tid} {pos}{len(out)}", pos, val, rng.randint(21, 31)))
    add("QB", rng.uniform(400, 2500))
    counts = {"RB": rng.randint(3, 5), "WR": rng.randint(3, 6), "TE": rng.randint(1, 2)}
    if profile == "loaded":
        for _ in range(2): add(rng.choice(["RB", "WR"]), rng.uniform(3000, 8000))
    elif profile == "lone":
        add(rng.choice(["RB", "WR"]), rng.uniform(3500, 8000))  # the cornerstone
    for pos, n in counts.items():
        for _ in range(n):
            if profile == "thin":
                add(pos, rng.uniform(150, 900))
            elif profile == "balanced":
                add(pos, rng.uniform(400, 2600))
            else:
                add(pos, rng.uniform(150, 1600))
    return out


def random_league(rng: random.Random, num_teams: int = 12) -> Tuple[dict, str]:
    """A random league; returns (ctx, viewer_roster_id). The viewer (roster 1)
    gets a randomly chosen profile so the slate covers all the cases."""
    profiles = ["loaded", "lone", "thin", "balanced"]
    table: List[dict] = []
    rosters: List[dict] = []
    roster_map: Dict[int, str] = {}
    for tid in range(1, num_teams + 1):
        prof = rng.choice(profiles) if tid == 1 else rng.choice(profiles + ["balanced", "balanced"])
        players = _team_players(rng, tid, prof)
        table.extend(players)
        rosters.append({"roster_id": tid, "players": [p["id"] for p in players]})
        roster_map[tid] = f"Team {tid}"
    picks = {str(tid): [{"season": "2026", "round": r, "original_roster_id": tid}]
             for tid in range(1, num_teams + 1) for r in (1, 2)}
    ctx = {
        "rosters": rosters, "roster_map": roster_map,
        "standings_map": {r["roster_id"]: rng.randint(1, num_teams) for r in rosters},
        "model_value_table": table, "picks_by_roster": picks,
        "settings": {"playoff_week_start": 15},
    }
    return ctx, "1"


def _lone_stud_pid(ctx: dict, viewer_rid: str) -> str | None:
    vals = {str(p["id"]): float(p["value"]) for p in ctx["model_value_table"]}
    roster = next(r for r in ctx["rosters"] if str(r["roster_id"]) == viewer_rid)
    ranked = sorted((str(p) for p in roster["players"]), key=lambda p: -vals.get(p, 0))
    if not ranked:
        return None
    top, nxt = vals.get(ranked[0], 0), (vals.get(ranked[1], 0) if len(ranked) > 1 else 0)
    return ranked[0] if top > 0 and (nxt <= 0 or top >= nxt * _LONE_STUD_GAP) else None


def _check_invariants(archetype: str, sugg: List[dict], ctx: dict, viewer_rid: str) -> List[str]:
    """Return a list of guardrail-violation strings (empty = all held)."""
    v: List[str] = []
    lone = _lone_stud_pid(ctx, viewer_rid)
    for s in sugg:
        send = s.get("suggested_send") or []
        if s.get("direction") == "acquire" and not send:
            v.append(f"{s.get('name')}: acquire with empty send")
        ap = s.get("acceptance_pct")
        if ap is not None and not (5 <= ap <= 90):
            v.append(f"{s.get('name')}: acceptance {ap} out of [5,90]")
        players = [a for a in send if not a.get("is_pick") and a.get("position") != "PICK"]
        if archetype == "consolidate":
            if lone and any(str(a.get("player_id")) == lone for a in players):
                v.append(f"{s.get('name')}: ships the lone stud")
            if len(send) < 2:
                v.append(f"{s.get('name')}: consolidate sent <2 assets")
            tgt = float(s.get("value") or 0)
            hv = max((float(a.get("value") or 0) for a in players), default=0)
            if hv > 0 and tgt < hv * _UPGRADE - 1e-6:
                v.append(f"{s.get('name')}: target {tgt} not > headliner {hv}*{_UPGRADE}")
            for a in players:
                if tgt > 0 and float(a.get("value") or 0) < tgt * _PIECE_FRAC - 1e-6:
                    v.append(f"{s.get('name')}: rock piece {a.get('name')} ({a.get('value')})")
        if archetype == "distribute":
            if len(send) != 1:
                v.append(f"{s.get('name')}: distribute sent {len(send)} (want 1 stud)")
            recv = s.get("suggested_receive") or []
            if not any(not a.get("is_pick") and a.get("position") != "PICK" for a in recv):
                v.append(f"{s.get('name')}: distribute return has no usable player")
    return v


def ensure_offline() -> None:
    """Stub the Sleeper HTTP layer before the engine lazily imports app (whose
    init would otherwise make live calls), so the eval runs offline on the
    analytical path. Idempotent; a no-op if fetch_json is already stubbed."""
    import dashboard_services.api as api
    if getattr(api.fetch_json, "_eval_stubbed", False):
        return

    def _fake(path, timeout=25, retries=3):
        if path == "/state/nfl":
            return {"season": "2026", "week": 0, "leg": 0, "season_type": "off",
                    "display_week": 1, "season_start_date": "2026-09-10"}
        return {}
    _fake._eval_stubbed = True
    api.fetch_json = _fake

    # Neutralize the heavy per-call side work that would otherwise hit the network
    # or DB on every suggestion run - none of it affects the guardrail invariants
    # or the analytical net-WP the eval measures (values come from the synthetic
    # ctx; pick slot labels are cosmetic).
    import app
    app.build_historical_pick_slot_map = lambda **k: {}
    import dashboard_services.player_value_history as _pvh
    _pvh.load_current_values_from_db = lambda: []
    # build_ppg_map fetches FantasyPros projections over the network; without it
    # the engine falls back to value-based lineup scoring, which is what we want
    # for synthetic rosters anyway (and avoids a ~19s-per-call network stall).
    import data_building.simulate_playoff_odds as _spo
    _spo.build_ppg_map = lambda *a, **k: ({}, {})


def evaluate(n_leagues: int = 50, seed: int = 0, num_teams: int = 12) -> dict:
    ensure_offline()
    rng = random.Random(seed)
    report = {a: {"leagues_with_sugg": 0, "n_sugg": 0, "net_wpd": [],
                  "violations": 0, "viol_samples": []} for a in _ARCHES}
    # One fixed league id so the engine's per-id side work (pick-slot map, any
    # value-DB lookups) happens once and is reused; the random ctx we pass in is
    # what actually varies each iteration.
    lid, key = "evalfixed", "sleeper:evalfixed:2026"
    for i in range(n_leagues):
        ctx, vrid = random_league(rng, num_teams)
        for arch in _ARCHES:
            ae._RESULT_CACHE.clear()
            ae._SIM_CACHE[key] = {"sim_state": None, "base_odds": {}, "ts": 9e18}
            try:
                out = ae.get_archetype_suggestions(
                    archetype=arch, platform="sleeper", league_id=lid,
                    season=2026, viewer_roster_id=vrid, league_type="1qb",
                    league_size=num_teams, ctx=ctx)
            finally:
                ae._SIM_CACHE.pop(key, None)
            sugg = out.get("suggestions") or []
            r = report[arch]
            if sugg:
                r["leagues_with_sugg"] += 1
            r["n_sugg"] += len(sugg)
            r["net_wpd"].extend(float(s.get("net_win_prob_delta") or 0) for s in sugg)
            viols = _check_invariants(arch, sugg, ctx, vrid)
            r["violations"] += len(viols)
            if viols and len(r["viol_samples"]) < 5:
                r["viol_samples"].extend(viols[:5])
    report["_meta"] = {"n_leagues": n_leagues, "seed": seed, "num_teams": num_teams}
    return report


def _fmt(report: dict) -> str:
    m = report["_meta"]
    lines = [f"Suggestion-quality eval  |  {m['n_leagues']} leagues x {m['num_teams']} teams  (seed {m['seed']})",
             f"{'archetype':<13}{'coverage':>10}{'#sugg':>7}{'mean netWP%':>13}{'%>=0':>7}{'viol':>6}"]
    for a in _ARCHES:
        r = report[a]
        cov = f"{r['leagues_with_sugg']}/{m['n_leagues']}"
        wp = r["net_wpd"]
        mean = (statistics.mean(wp) * 100) if wp else 0.0
        pos = (100 * sum(1 for x in wp if x >= 0) / len(wp)) if wp else 0.0
        lines.append(f"{a:<13}{cov:>10}{r['n_sugg']:>7}{mean:>13.2f}{pos:>6.0f}%{r['violations']:>6}")
    total_viol = sum(report[a]["violations"] for a in _ARCHES)
    lines.append("")
    lines.append("guardrails: " + ("ALL HELD" if total_viol == 0 else f"{total_viol} VIOLATIONS"))
    for a in _ARCHES:
        for s in report[a]["viol_samples"]:
            lines.append(f"  [{a}] {s}")
    return "\n".join(lines)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    print(_fmt(evaluate(n_leagues=n, seed=seed)))


if __name__ == "__main__":
    main()
