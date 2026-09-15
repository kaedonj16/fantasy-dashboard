"""Account-scoped, asynchronously requested portfolio card summaries."""
from __future__ import annotations

import threading
from datetime import datetime, timezone

_CACHE: dict[tuple[int, str, str, int], dict] = {}
_LOCK = threading.Lock()


def cache_key(account_id, platform, league_id, season):
    # account_id is deliberately part of the key: viewer/team data must never
    # cross an account boundary even for the same public league.
    return (int(account_id), str(platform).lower(), str(league_id), int(season))


def get_cached_summary(account_id, platform, league_id, season):
    with _LOCK:
        value = _CACHE.get(cache_key(account_id, platform, league_id, season))
        return dict(value) if value else None


def build_league_summary(account_id, membership, context_loader):
    """Build only identity/record/standing fields needed by the fast card."""
    platform = str(membership.get("platform") or "sleeper").lower()
    league_id = str(membership.get("league_id") or "")
    season = int(membership.get("season") or 0)
    base = {"platform": platform, "league_id": league_id, "season": season,
            "name": membership.get("name") or "Unknown"}
    if membership.get("connection_status") == "reauth_required":
        return {**base, "state": "reconnect_required", "message": "Reconnect provider"}
    ctx = context_loader(platform, league_id, season)
    rosters = ctx.get("rosters") or []
    league = ctx.get("league") or {}
    latest = ctx.get("latest_draft") if isinstance(ctx.get("latest_draft"), dict) else {}
    from utils.league_payload import draft_start_ms, startup_draft_phase
    phase = startup_draft_phase(league, latest, rosters)
    if phase != "drafted":
        return {**base, "name": league.get("name") or base["name"], "state": "predraft",
                "message": "Drafting now" if phase == "drafting" else "Draft not started",
                "draft_phase": phase, "draft_start_ms": draft_start_ms(league, latest)}
    from dashboard_services.accounts import resolve_account_viewer_for_league
    viewer = resolve_account_viewer_for_league(
        int(account_id), platform, league_id, season, ctx.get("users") or [], rosters,
    )
    rid = str((viewer or {}).get("viewer_roster_id") or "")
    roster = next((r for r in rosters if str(r.get("roster_id") or "") == rid), None)
    if not roster:
        return {**base, "name": league.get("name") or base["name"],
                "state": "team_not_linked", "message": "Team not linked yet"}
    from dashboard_services.ai.context_builders import portfolio_record_and_rank
    from dashboard_services.display_names import team_label_from_user
    wins, losses, ties, _pf, rank = portfolio_record_and_rank(ctx, rid, roster)
    owner_id = str(roster.get("owner_id") or "")
    owner = next((u for u in (ctx.get("users") or [])
                  if str(u.get("user_id") or "") == owner_id), None)
    total = int(ctx.get("total_rosters") or len(rosters) or 0)
    refreshed = datetime.now(timezone.utc).isoformat()
    result = {**base, "name": league.get("name") or base["name"], "state": "ready",
              "team_name": team_label_from_user(owner, roster, fallback=""),
              "wins": wins, "losses": losses, "ties": ties,
              "record": f"{wins}-{losses}" + (f"-{ties}" if ties else ""),
              "rank": rank, "total_teams": total, "refreshed_at": refreshed}
    with _LOCK:
        _CACHE[cache_key(account_id, platform, league_id, season)] = dict(result)
    return result
