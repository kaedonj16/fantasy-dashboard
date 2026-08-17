#!/usr/bin/env python3
"""Explore Pinnacle's guest API for season-long NFL player over/unders.

The reachability probe (scripts/probe_odds_sources.py) showed Pinnacle's guest
API is the one free source the Render datacenter can reach with live data (HTTP
200 JSON), while every real sportsbook (DraftKings/FanDuel/BetMGM/Caesars) is
datacenter-IP-blocked. This script confirms Pinnacle actually carries season-long
player props and dumps a real sample so a parser can be written against the true
shape (same loop we used for DraftKings).

Run on Render:

    python scripts/probe_pinnacle_nfl.py

It discovers the American-Football sport id and the NFL league, pulls the
league's matchups + straight markets, buckets the "special" matchups, and prints
a handful that look like season-long player over/unders (Regular Season +
passing/rushing/receiving yards/TDs/receptions), with their market prices/points.
"""
from __future__ import annotations

import json
import re
import sys

_BASE = "https://guest.api.arcadia.pinnacle.com/0.1"
_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json",
    # Well-known public guest key used by pinnacle.com's own front end.
    "X-API-Key": "CmX2KcMrXuFmNg6YFbmTxE0y9CIrOi0R",
    "Referer": "https://www.pinnacle.com/",
}
_STAT_RE = re.compile(
    r"pass(?:ing)?|rush(?:ing)?|receiv(?:ing)?|reception|yard|touchdown|\btds?\b",
    re.I,
)
_SEASON_RE = re.compile(r"regular season|season|2025|2026|full season", re.I)


def _get(session, path, **params):
    resp = session.get(f"{_BASE}{path}", headers=_HEADERS, params=params or None, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _find_nfl(session):
    """Return (sport_id, league_id) for the NFL, discovered not hardcoded."""
    sports = _get(session, "/sports")
    football_ids = [s.get("id") for s in sports
                    if "football" in str(s.get("name", "")).lower()
                    and "american" in str(s.get("name", "")).lower()]
    if not football_ids:  # fall back to any "football"
        football_ids = [s.get("id") for s in sports
                        if "football" in str(s.get("name", "")).lower()]
    for sport_id in football_ids:
        leagues = _get(session, f"/sports/{sport_id}/leagues")
        for lg in leagues:
            if str(lg.get("name", "")).strip().upper() == "NFL":
                return sport_id, lg.get("id")
    return (football_ids[0] if football_ids else None), None


def _text(matchup) -> str:
    special = matchup.get("special") or {}
    parts = " ".join(str(p.get("name", "")) for p in (matchup.get("participants") or []))
    return " ".join(str(x) for x in (
        special.get("description", ""), special.get("category", ""),
        matchup.get("units", ""), parts,
    ))


def main() -> int:
    import requests
    session = requests.Session()

    sport_id, league_id = _find_nfl(session)
    print(f"American Football sport id={sport_id}  NFL league id={league_id}\n")
    if not league_id:
        print("Could not locate the NFL league; dumping sports/leagues for debugging.")
        print(json.dumps(_get(session, "/sports")[:20], indent=2)[:2000])
        return 1

    matchups = _get(session, f"/leagues/{league_id}/matchups")
    markets = _get(session, f"/leagues/{league_id}/markets/straight")
    print(f"matchups={len(matchups)}  straight markets={len(markets)}")

    specials = [m for m in matchups if str(m.get("type")) == "special"]
    print(f"special matchups={len(specials)}")
    cats = {}
    for m in specials:
        cat = str((m.get("special") or {}).get("category") or "?")
        cats[cat] = cats.get(cat, 0) + 1
    print("special categories:", json.dumps(cats, indent=2), "\n")

    # Season-long player over/unders: a special whose text mentions a stat and
    # reads season-scoped.
    season_props = [m for m in specials
                    if _STAT_RE.search(_text(m)) and _SEASON_RE.search(_text(m))]
    print(f"season-long player-prop candidates={len(season_props)}\n")

    markets_by_matchup = {}
    for mk in markets:
        markets_by_matchup.setdefault(str(mk.get("matchupId")), []).append(mk)

    for m in season_props[:8]:
        print("=" * 70)
        print(json.dumps({
            "id": m.get("id"), "type": m.get("type"), "units": m.get("units"),
            "special": m.get("special"),
            "participants": [{k: p.get(k) for k in ("id", "name", "alignment")}
                             for p in (m.get("participants") or [])],
        }, indent=2))
        mks = markets_by_matchup.get(str(m.get("id")), [])
        print(f"-- {len(mks)} market(s):")
        print(json.dumps(mks[:4], indent=2)[:1500])

    if not season_props:
        print("No season-long player props matched the filter. Sample of the "
              "first 6 special matchups so we can see what Pinnacle DOES carry:")
        for m in specials[:6]:
            print("-", json.dumps({"units": m.get("units"), "special": m.get("special"),
                                    "participants": [p.get("name") for p in (m.get("participants") or [])]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
