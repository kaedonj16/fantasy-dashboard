"""Cross-league player availability.

For one player, report where they stand in every league the signed-in
viewer belongs to: rostered (and by whom, in which slot) or available.
Powers the player modal's Availability section and the nav search
dropdown's availability line.

Data comes from the cached league contexts (``ctx["rosters"]``), the
same source Waivers builds its rostered set from, so player ids are the
canonical cross-platform ids the rest of the site uses. A league whose
context is not cached reports ``unknown`` - never a guessed status -
because the cached layer cannot distinguish a free agent from a player
on waivers, and an unloaded league must not pose as either.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

_PLATFORM_LABELS = {
    "sleeper": "Sleeper",
    "espn": "ESPN",
    "yahoo": "Yahoo",
    "mfl": "MFL",
    "fleaflicker": "Fleaflicker",
}


def player_slot(roster: dict, player_id: str) -> Optional[str]:
    """The slot a player occupies on one roster, or None if not rostered.

    Mirrors the roster shape every platform adapter produces:
    ``players`` is the full roster, with ``starters`` / ``reserve`` /
    ``taxi`` subsets naming the special slots; anything else is Bench.
    """
    pid = str(player_id)
    players = {str(p) for p in (roster.get("players") or [])}
    if pid not in players:
        return None
    if pid in {str(p) for p in (roster.get("starters") or [])}:
        return "Starter"
    if pid in {str(p) for p in (roster.get("reserve") or [])}:
        return "IR"
    if pid in {str(p) for p in (roster.get("taxi") or [])}:
        return "Taxi"
    return "Bench"


def _scoring_label(scoring_settings: Any) -> Optional[str]:
    if not isinstance(scoring_settings, dict):
        return None
    rec = scoring_settings.get("rec")
    try:
        rec = float(rec)
    except (TypeError, ValueError):
        return None
    if rec >= 1.0:
        return "PPR"
    if rec == 0.5:
        return "Half PPR"
    if rec == 0.0:
        return "Standard"
    return None


def _is_dynasty(league: Any) -> bool:
    """Dynasty only when the provider says so (Sleeper settings.type 2)."""
    if not isinstance(league, dict):
        return False
    settings = league.get("settings") if isinstance(league.get("settings"), dict) else {}
    return settings.get("type") == 2


def league_sub(platform: str, ctx: dict) -> str:
    """The compact format line under a league name, from verified fields
    only: ``Sleeper - 10-Team - PPR - Superflex``. Unknown pieces are
    omitted rather than defaulted."""
    parts = [_PLATFORM_LABELS.get(str(platform).lower(), str(platform).title())]
    league = ctx.get("league") if isinstance(ctx.get("league"), dict) else {}
    size = 0
    try:
        size = int(ctx.get("total_rosters") or 0)
    except (TypeError, ValueError):
        size = 0
    if not size:
        try:
            from utils.league import fields_from_provider_league

            size = int(fields_from_provider_league(league).get("size") or 0)
        except Exception:
            size = 0
    if not size:
        size = len(ctx.get("rosters") or [])
    if size >= 2:
        parts.append(f"{size}-Team")
    scoring = _scoring_label(ctx.get("scoring_settings"))
    if scoring:
        parts.append(scoring)
    try:
        from utils.league import is_sf_from_league

        settings = league.get("settings") if isinstance(league.get("settings"), dict) else {}
        if is_sf_from_league(ctx.get("roster_positions") or league.get("roster_positions"), settings):
            parts.append("Superflex")
    except Exception:
        pass
    if _is_dynasty(league):
        parts.append("Dynasty")
    return " - ".join(parts)


def _owner_name(ctx: dict, roster_id: Any) -> str:
    roster_map = ctx.get("roster_map") or {}
    name = roster_map.get(roster_id) or roster_map.get(str(roster_id))
    if name:
        return str(name)
    return f"Roster {roster_id}"


def availability_in_league(ctx: dict, player_id: str) -> dict:
    """One player's status in one league context.

    ``unknown`` when the context carries no rosters (not cached yet), so
    callers can label the league honestly instead of guessing.
    """
    rosters = ctx.get("rosters") or []
    if not rosters:
        return {"status": "unknown", "owner": None, "slot": None, "mine": False}
    viewer = ctx.get("viewer") or {}
    viewer_rid = viewer.get("viewer_roster_id")
    for roster in rosters:
        slot = player_slot(roster, player_id)
        if slot is None:
            continue
        rid = roster.get("roster_id")
        mine = viewer_rid is not None and str(rid) == str(viewer_rid)
        return {
            "status": "rostered",
            "owner": _owner_name(ctx, rid),
            "slot": slot,
            "mine": mine,
        }
    return {"status": "available", "owner": None, "slot": None, "mine": False}


def build_availability(
    leagues: Iterable[dict],
    player_ids: Iterable[str],
    ctx_getter: Callable[[str, str, int], Optional[dict]],
) -> dict:
    """Availability for each player across the viewer's leagues.

    ``leagues`` rows carry platform / league_id / season / name (the
    resolve_my_leagues shape). ``ctx_getter`` returns a cached league
    context or an empty/None value when the league is not loaded; it is
    called once per league no matter how many players are requested.
    Returns ``{player_id: {"summary": ..., "leagues": [...]}}``.
    """
    pids = [str(p) for p in player_ids]
    league_rows = [lg for lg in leagues if lg.get("league_id")]
    contexts: list[tuple[dict, Optional[dict]]] = []
    for lg in league_rows:
        platform = str(lg.get("platform") or "sleeper").lower()
        try:
            season = int(lg.get("season") or 0)
        except (TypeError, ValueError):
            season = 0
        try:
            ctx = ctx_getter(platform, str(lg["league_id"]), season) if season else None
        except Exception:
            ctx = None
        contexts.append((lg, ctx if isinstance(ctx, dict) else None))

    out: dict[str, dict] = {}
    for pid in pids:
        rows: list[dict] = []
        available = rostered = mine_count = checked = 0
        for lg, ctx in contexts:
            platform = str(lg.get("platform") or "sleeper").lower()
            league_id = str(lg.get("league_id"))
            try:
                season = int(lg.get("season") or 0)
            except (TypeError, ValueError):
                season = 0
            base = {
                "platform": platform,
                "league_id": league_id,
                "season": season,
            }
            if ctx:
                base["name"] = str(
                    (ctx.get("league") or {}).get("name") or lg.get("name") or "League"
                )
                base["sub"] = league_sub(platform, ctx)
                status = availability_in_league(ctx, pid)
            else:
                base["name"] = str(lg.get("name") or "League")
                base["sub"] = _PLATFORM_LABELS.get(platform, platform.title())
                status = {"status": "unknown", "owner": None, "slot": None, "mine": False}
            if status["status"] == "available":
                available += 1
                checked += 1
                base["href"] = f"/{platform}/{season}/{league_id}/waivers"
            elif status["status"] == "rostered":
                rostered += 1
                checked += 1
                if status["mine"]:
                    mine_count += 1
                base["href"] = f"/{platform}/{season}/{league_id}/teams"
            else:
                base["href"] = None
            rows.append({**base, **status})
        out[pid] = {
            "summary": {
                "total": len(contexts),
                "checked": checked,
                "available": available,
                "rostered": rostered,
                "mine": mine_count,
            },
            "leagues": rows,
        }
    return out
