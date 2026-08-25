"""Cross-league Redzone "My Leagues" portfolio helpers.

The live fetch still lives in app.py (it needs matchup/roster providers and the
Flask session). This module is the platform-agnostic bit: which leagues to
include, and which roster in a league is the viewer's. Pure Python so the
Sleeper-only vs account-portfolio decision is unit-testable without Flask.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


MAX_USER_LEAGUES = 12


def match_viewer_roster(
    rosters: List[dict],
    *,
    team_id: Optional[str] = None,
    owner_id: Optional[str] = None,
) -> Optional[dict]:
    """Pick the viewer's roster from a league's roster list.

    Stored ``team_id`` (the league-scoped roster id on the account) wins.
    Otherwise match ``owner_id`` (Sleeper user id, ESPN/Yahoo owner guid).
    """
    tid = str(team_id or "")
    if tid:
        hit = next(
            (r for r in (rosters or []) if str(r.get("roster_id") or "") == tid),
            None,
        )
        if hit:
            return hit
    oid = str(owner_id or "")
    if oid:
        return next(
            (r for r in (rosters or []) if str(r.get("owner_id") or "") == oid),
            None,
        )
    return None


def portfolio_from_account_leagues(
    saved: List[dict],
    *,
    season: int,
    cap: int = MAX_USER_LEAGUES,
) -> List[Dict[str, Any]]:
    """Normalize ``list_user_leagues`` / ``resolve_account_leagues`` rows.

    ESPN season ids roll forward each year; a stale saved season is bumped to
    ``season`` so we collect the current league rather than last year's.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in saved or []:
        plat = str(row.get("platform") or "").lower()
        lid = str(row.get("league_id") or "")
        if not plat or not lid:
            continue
        key = (plat, lid)
        if key in seen:
            continue
        seen.add(key)
        try:
            lg_season = int(row.get("season") or season or 0)
        except (TypeError, ValueError):
            lg_season = int(season or 0)
        if plat == "espn" and season and lg_season and lg_season < int(season):
            lg_season = int(season)
        out.append({
            "platform": plat,
            "league_id": lid,
            "name": row.get("name") or "",
            "season": lg_season or int(season or 0),
            "team_id": str(row.get("team_id") or ""),
        })
        if len(out) >= cap:
            break
    return out


def portfolio_from_sleeper_leagues(
    leagues_raw: List[dict],
    *,
    season: int,
    cap: int = MAX_USER_LEAGUES,
) -> List[Dict[str, Any]]:
    """Normalize Sleeper ``get_sleeper_user_leagues`` rows into the same shape."""
    out: List[Dict[str, Any]] = []
    for i, lg in enumerate(leagues_raw or []):
        lid = str(lg.get("league_id") or "")
        if not lid:
            continue
        out.append({
            "platform": "sleeper",
            "league_id": lid,
            "name": lg.get("name") or f"League {i + 1}",
            "season": int(season or 0),
            "team_id": "",
        })
        if len(out) >= cap:
            break
    return out
