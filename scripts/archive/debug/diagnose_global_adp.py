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

    # Split the 63-of-350 question: crosswalk miss vs. no average_pick yet.
    xwalk = G.yahoo_id_to_canonical()
    print("  yahoo_id crosswalk size:", len(xwalk))
    total = has_ap = has_hit = mapped = 0
    miss_with_ap = []   # yahoo has an ADP but our crosswalk can't map it (FIXABLE)
    hit_no_ap = []      # we can map it but Yahoo has no ADP yet (not our bug)
    top_rows = []
    start = 0
    page = 25
    while start < 350:
        pg = G._get_json(G._YAHOO_URL.format(start=start, count=page))
        got = 0
        for entry in G._iter_players_blocks(pg):
            got += 1
            total += 1
            flat = G._flatten_yahoo(entry)
            ap = G._yahoo_avg_pick(flat)
            yid = flat.get("player_id")
            nm = flat.get("name")
            nm = nm.get("full") if isinstance(nm, dict) else nm
            hit = str(yid) in xwalk
            if ap is not None:
                has_ap += 1
            if hit:
                has_hit += 1
            if ap is not None and hit:
                mapped += 1
            if ap is not None and not hit and len(miss_with_ap) < 12:
                miss_with_ap.append(f"{nm} (yid={yid}, adp={ap})")
            if hit and ap is None and len(hit_no_ap) < 8:
                hit_no_ap.append(f"{nm} (yid={yid})")
            if len(top_rows) < 12:
                top_rows.append(f"    yid={str(yid):>7}  hit={hit!s:5}  adp={ap}  {nm}")
        if got < page:
            break
        start += page
    print(f"  scanned {total} raw players:")
    print(f"    have average_pick > 0 : {has_ap}")
    print(f"    in yahoo crosswalk    : {has_hit}")
    print(f"    mapped by id only     : {mapped}")
    # Actual fetch result — includes the name/position fallback for the recent
    # players Sleeper's yahoo_id lags on, so this should exceed 'mapped by id only'.
    fr = G.fetch_yahoo_global_adp(int(season))
    print(f"    fetch mapped (id+name): {fr['mapped_count']} of raw {fr['raw_count']}")
    print("  --- top 12: yahoo_id / hit / adp / name ---")
    print("\n".join(top_rows))
    print("  --- has ADP but NOT in crosswalk (fixable crosswalk gap) ---")
    print("    " + ("; ".join(miss_with_ap) if miss_with_ap else "(none)"))
    print("  --- in crosswalk but no ADP yet (Yahoo hasn't priced them) ---")
    print("    " + ("; ".join(hit_no_ap) if hit_no_ap else "(none)"))

    # Can we merge players_index for Yahoo like we did for ESPN? Show its id keys.
    try:
        from utils.utils import load_players_index
        idx = load_players_index() or {}
        sample = next(iter(idx.values()), {})
        yahoo_keys = [k for k in sample.keys() if "yahoo" in k.lower()]
        n_with_yahoo = sum(1 for v in idx.values()
                           if any("yahoo" in str(k).lower() and v.get(k) for k in v))
        print("  players_index sample keys:", list(sample.keys())[:20])
        print("  players_index yahoo-ish keys on sample:", yahoo_keys,
              "| entries with a yahoo id:", n_with_yahoo, "/", len(idx))
    except Exception as exc:
        print("  players_index inspect failed:", type(exc).__name__, exc)


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
