#!/usr/bin/env python3
"""Deep-dive the Yahoo/ESPN/MFL global ADP fetch + crosswalk, from a networked box.

The audit script (audit_adp_sources.py --live) reports the *symptoms* (raw vs
mapped counts); this prints the *evidence* needed to fix a broken fetch or
crosswalk: the raw upstream response shape and concrete id/name comparisons.

Safe + read-only. No credentials. Run on a host with outbound access (Render):

    python scripts/diagnose_global_adp.py --season 2026
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _trunc(obj, n=1200):
    s = json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + f"... (+{len(s)-n} chars)"


def diagnose_yahoo(season: int):
    print("\n" + "=" * 70 + "\nYAHOO\n" + "=" * 70)
    from dashboard_services.providers import global_adp as G
    url = G._YAHOO_URL.format(start=0, count=10)
    print("URL:", url)
    try:
        payload = G._get_json(url)
    except Exception as exc:
        print("  fetch FAILED:", type(exc).__name__, exc)
        return
    fc = (payload or {}).get("fantasy_content") or {}
    print("  top-level keys:", list((payload or {}).keys()))
    print("  fantasy_content keys:", list(fc.keys()))
    game = fc.get("game")
    print("  game type:", type(game).__name__)
    if isinstance(game, list):
        for i, c in enumerate(game):
            print(f"    game[{i}] keys:", list(c.keys()) if isinstance(c, dict) else type(c).__name__)
    elif isinstance(game, dict):
        print("    game keys:", list(game.keys()))
    n = sum(1 for _ in G._iter_players_blocks(payload))
    print("  _iter_players_blocks yielded:", n)
    # Show the raw players container so we can see the real nesting.
    print("  raw payload sample:", _trunc(payload, 1600))


def diagnose_espn(season: int):
    print("\n" + "=" * 70 + "\nESPN\n" + "=" * 70)
    from dashboard_services.providers import global_adp as G
    xwalk = G.espn_id_to_canonical()
    print("  espn_id crosswalk size:", len(xwalk))
    try:
        payload = G._get_json(
            G._ESPN_URL.format(season=int(season)),
            headers={"X-Fantasy-Filter": G._espn_filter(20)})
    except Exception as exc:
        print("  fetch FAILED:", type(exc).__name__, exc)
        return
    players = (payload or {}).get("players") or []
    if isinstance(players, dict):
        players = list(players.values())
    print("  players returned:", len(players))
    print("  --- top 12 response players: id / name / in-crosswalk? ---")
    for entry in players[:12]:
        if not isinstance(entry, dict):
            continue
        p = entry.get("player") if isinstance(entry.get("player"), dict) else entry
        eid = p.get("id")
        nm = p.get("fullName") or p.get("name")
        adp = (p.get("ownership") or {}).get("averageDraftPosition")
        hit = str(eid) in xwalk
        print(f"    id={str(eid):>10}  hit={hit!s:5}  adp={adp}  {nm}")
    # How does the crosswalk key space look vs the response id space?
    sample_keys = list(xwalk.keys())[:5]
    print("  sample crosswalk espn_id keys:", sample_keys)
    resp_ids = [str((e.get("player") or e).get("id"))
                for e in players[:50] if isinstance(e, dict)]
    print("  sample response ids:", resp_ids[:5])
    overlap = sum(1 for i in resp_ids if i in xwalk)
    print(f"  of first 50 response ids, {overlap} are in the crosswalk")


def diagnose_mfl(season: int):
    print("\n" + "=" * 70 + "\nMFL\n" + "=" * 70)
    from dashboard_services.providers import global_adp as G
    xwalk = G.mfl_id_to_canonical(int(season))
    print("  mfl_id crosswalk size:", len(xwalk))
    rows = G._mfl_player_rows(int(season))
    print("  player export rows:", len(rows))
    print("  --- sample raw MFL names (should be 'Last, First') ---")
    for p in rows[:6]:
        print(f"    id={p.get('id')}  name={p.get('name')!r}  pos={p.get('position')}")
    r = G.fetch_mfl_adp(int(season), is_ppr=1, fcount=12, is_mock=0)
    print("  fetch_mfl_adp mapped:", r["mapped_count"], "raw:", r["raw_count"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--only", choices=["yahoo", "espn", "mfl"], default=None)
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    if args.only in (None, "yahoo"):
        diagnose_yahoo(args.season)
    if args.only in (None, "espn"):
        diagnose_espn(args.season)
    if args.only in (None, "mfl"):
        diagnose_mfl(args.season)


if __name__ == "__main__":
    main()
