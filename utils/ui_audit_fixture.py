"""Deterministic in-season league fixture for local UI/UX audits.

When enabled (``UI_AUDIT=1``), league id ``ui-audit`` resolves to a rich
mid-season Sleeper dynasty context built from committed player/value caches —
no live Sleeper HTTP. Use ``/ui-audit`` as the walkthrough hub.
"""
from __future__ import annotations

import os
import random
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd

UI_AUDIT_LEAGUE_ID = "ui-audit"
_DEFAULT_PLATFORM = "sleeper"
_DEFAULT_SEASON = 2026

_TEAM_NAMES = [f"Audit Team {chr(65 + i)}" for i in range(10)]
_RIDS = [str(i + 1) for i in range(10)]
_UIDS = [f"audit_u{i + 1}" for i in range(10)]

_INSEASON_NFL_STATE = {
    "season": "2026",
    "week": 11,
    "leg": 11,
    "season_type": "regular",
    "display_week": 11,
    "season_start_date": "2026-09-10",
}

_CTX_CACHE: dict[tuple[str, str, int], dict] = {}


def ui_audit_enabled() -> bool:
    flag = (os.environ.get("UI_AUDIT") or "").strip().lower()
    return flag in ("1", "true", "yes", "on")


def is_ui_audit_league(league_id: Optional[str]) -> bool:
    return str(league_id or "").strip().lower() == UI_AUDIT_LEAGUE_ID


def install_ui_audit_hooks() -> None:
    """Patch Sleeper HTTP + NFL state when UI audit mode is on."""
    if not ui_audit_enabled():
        return
    import dashboard_services.api as api

    _real_fetch = api.fetch_json

    def _audit_fetch_json(path, timeout=25, retries=3):
        if path == "/state/nfl":
            return dict(_INSEASON_NFL_STATE)
        if path.startswith("/league/") and path.endswith("/transactions"):
            return []
        if path.startswith("/league/") and "/matchups/" in path:
            return []
        if path.startswith("/league/"):
            return {}
        if path.startswith("/user/"):
            return {}
        return _real_fetch(path, timeout=timeout, retries=retries)

    api.fetch_json = _audit_fetch_json
    api.get_nfl_games_for_week_raw = lambda *a, **k: []
    api.get_nfl_scores_for_date = lambda *a, **k: {}


def _build_df_weekly() -> pd.DataFrame:
    rng = random.Random(7)
    rows: list[dict] = []
    for week in range(1, 11):
        order = list(range(10))
        rng.shuffle(order)
        for mid, i in enumerate(range(0, 10, 2), start=1):
            a, b = order[i], order[i + 1]
            sa = round(rng.uniform(85, 165), 2)
            sb = round(rng.uniform(85, 165), 2)
            for me, opp, mine, theirs in ((a, b, sa, sb), (b, a, sb, sa)):
                rows.append({
                    "owner": _TEAM_NAMES[me],
                    "roster_id": _RIDS[me],
                    "week": week,
                    "points": mine,
                    "points_against": theirs,
                    "finalized": True,
                    "avatar": "",
                    "matchup_id": mid,
                    "opponent": _TEAM_NAMES[opp],
                })
    return pd.DataFrame(rows)


def _pick_roster_player_ids(model_rows: List[dict], roster_idx: int, per_roster: int = 16) -> List[str]:
    skill = [
        r for r in model_rows
        if str(r.get("position") or r.get("pos") or "").upper() in ("QB", "RB", "WR", "TE")
        and r.get("id") is not None
    ]
    if not skill:
        return []
    start = (roster_idx * per_roster) % max(1, len(skill) - per_roster)
    chunk = skill[start:start + per_roster]
    if len(chunk) < per_roster:
        chunk = (chunk + skill[: per_roster - len(chunk)])[:per_roster]
    return [str(r["id"]) for r in chunk]


def build_ui_audit_league_context(
    platform: str = _DEFAULT_PLATFORM,
    league_id: str = UI_AUDIT_LEAGUE_ID,
    season: int = _DEFAULT_SEASON,
) -> dict:
    """Return a league ctx dict shaped like ``build_league_context()``."""
    key = (platform, league_id, int(season))
    cached = _CTX_CACHE.get(key)
    if cached is not None:
        return cached

    import app as appmod
    from utils.league_payload import build_roster_map
    from dashboard_services.service import build_standings_map, finalize_team_stats

    players = appmod.get_players_global()
    players_index = appmod.load_players_index()
    teams_index = appmod.load_teams_index()
    players_map = appmod.get_players_map(players)
    model_value_table = list(appmod.get_model_value_table_cached() or [])

    df_weekly = _build_df_weekly()
    avatar_map = {o: "" for o in _TEAM_NAMES}
    team_stats = finalize_team_stats(
        df_weekly[df_weekly["finalized"]],
        avatar_map,
        {},
        [],
        10,
    )

    users = [
        {
            "user_id": uid,
            "display_name": name,
            "metadata": {"team_name": name, "avatar": ""},
        }
        for uid, name in zip(_UIDS, _TEAM_NAMES)
    ]
    rosters = []
    for idx, (rid, uid, name) in enumerate(zip(_RIDS, _UIDS, _TEAM_NAMES)):
        pids = _pick_roster_player_ids(model_value_table, idx)
        if not team_stats.empty and name in team_stats["owner"].values:
            row = team_stats.loc[team_stats["owner"] == name].iloc[0]
            wins = int(row.get("Wins") or 5)
            losses = int(row.get("Losses") or 5)
        else:
            wins, losses = 5, 5
        rosters.append({
            "roster_id": rid,
            "owner_id": uid,
            "players": pids,
            "starters": pids[:9] if len(pids) >= 9 else pids,
            "settings": {"wins": wins, "losses": losses, "ties": 0},
            "metadata": {"record": f"{wins}-{losses}"},
        })

    roster_map = build_roster_map(users, rosters)
    standings_map = (
        build_standings_map(team_stats, roster_map)
        if team_stats is not None and not team_stats.empty
        else {}
    )

    league_settings = {
        "playoff_week_start": 15,
        "playoff_teams": 6,
        "type": 2,
        "waiver_type": 0,
        "trade_deadline": 12,
    }
    roster_positions = [
        "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "SUPER_FLEX",
        "BN", "BN", "BN", "BN", "BN", "BN",
    ]
    league = {
        "name": "UI Audit Dynasty",
        "league_id": UI_AUDIT_LEAGUE_ID,
        "total_rosters": 10,
        "status": "in_season",
        "season": str(season),
        "sport": "nfl",
        "scoring_settings": {
            "rec": 1.0,
            "pass_td": 4.0,
            "pass_yd": 0.04,
            "rush_yd": 0.1,
            "rush_td": 6.0,
            "rec_yd": 0.1,
            "rec_td": 6.0,
        },
        "roster_positions": roster_positions,
        "settings": league_settings,
    }

    ctx = {
        "platform": platform,
        "league": league,
        "league_id": league_id,
        "resolved_league_id": league_id,
        "season": season,
        "rosters": rosters,
        "users": users,
        "traded": [],
        "current_season": season,
        "current_week": 11,
        "current_leg": 11,
        "season_type": "regular",
        "season_complete": False,
        "weeks": 18,
        "players": players,
        "players_map": players_map,
        "players_index": players_index,
        "teams_index": teams_index,
        "df_weekly": df_weekly,
        "team_stats": team_stats,
        "roster_map": roster_map,
        "injury_df": pd.DataFrame(),
        "activity_df": pd.DataFrame(columns=["kind", "week", "ts", "data"]),
        "standings_map": standings_map,
        "picks_by_roster": {},
        "team_game_lookup": {},
        "model_value_table": model_value_table,
        "scoring_settings": dict(league["scoring_settings"]),
        "raw_scoring_settings": dict(league["scoring_settings"]),
        "roster_positions": roster_positions,
        "league_settings": league_settings,
        "total_rosters": 10,
        "mode": "in_season",
        "offseason_mode": False,
        "drafts": [],
        "latest_draft": None,
        "rookie_rankings": appmod._load_rookie_rankings_for_ctx(),
        "viewer": {},
    }
    _CTX_CACHE[key] = ctx
    return ctx


def bootstrap_viewer_session(session) -> None:
    """Seed Flask session so league pages render as a signed-in team owner."""
    session["viewer_username"] = "ui_audit_user"
    session["viewer_user_id"] = _UIDS[0]
    session["viewer_roster_id"] = _RIDS[0]
    session["viewer_team_name"] = _TEAM_NAMES[0]
    session["last_league_id"] = UI_AUDIT_LEAGUE_ID
    session["last_platform"] = _DEFAULT_PLATFORM
    session["last_season"] = _DEFAULT_SEASON
    session["viewer_platform"] = _DEFAULT_PLATFORM


# ── Route catalog for the /ui-audit hub ──────────────────────────────────────

PUBLIC_PAGES = [
    ("/", "Home"),
    ("/pricing", "Pricing"),
    ("/trade", "Trade calculator (guest)"),
    ("/trade-database", "Trade database"),
    ("/trade-intel", "Trade intel"),
    ("/players", "Player rankings"),
    ("/rankings/dynasty", "Dynasty rankings"),
    ("/dynasty-trade-value-chart", "Value chart"),
    ("/top-movers", "Top movers"),
    ("/compare", "Compare"),
    ("/breakouts", "Breakouts (guest)"),
    ("/prospects", "Prospects"),
    ("/draft", "Draft room (guest)"),
    ("/draft/cheat-sheet", "Cheat sheet"),
    ("/keeper", "Keeper assistant"),
    ("/guides", "Guides index"),
    ("/faq", "FAQ"),
    ("/glossary", "Glossary"),
    ("/about", "About"),
    ("/portfolio", "My Leagues (needs session)"),
    ("/watchlist", "Watchlist (needs session)"),
]

LEAGUE_PAGES = [
    "dashboard",
    "standings",
    "teams",
    "weekly",
    "activity",
    "awards",
    "history",
    "graphs",
    "recap",
    "waivers",
    "trade",
    "trade-intel",
    "trade-database",
    "draft",
    "draft/cheat-sheet",
    "metrics",
    "schedule",
    "breakouts",
    "prospects",
    "league_health",
    "scout",
    "optimal",
    "redzone",
]

SPECIAL_QUERY = {
    "graphs": "?tour=1",
    "history": "?tour=1",
    "redzone": "?demo=1",
}


def league_page_href(page: str, *, platform: str = _DEFAULT_PLATFORM, season: int = _DEFAULT_SEASON) -> str:
    tab_pages = {"scout": "scout", "optimal": "optimal"}
    if page in tab_pages:
        return f"/{platform}/{season}/{UI_AUDIT_LEAGUE_ID}/weekly?tab={tab_pages[page]}"
    q = SPECIAL_QUERY.get(page, "")
    return f"/{platform}/{season}/{UI_AUDIT_LEAGUE_ID}/{page}{q}"


def all_audit_hrefs() -> List[tuple[str, str]]:
    out = list(PUBLIC_PAGES)
    for page in LEAGUE_PAGES:
        label = page.replace("/", " / ").title()
        out.append((league_page_href(page), f"League · {label}"))
    return out
