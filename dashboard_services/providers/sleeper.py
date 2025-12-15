from __future__ import annotations

from typing import List

# import your existing Sleeper functions here
from dashboard_services.api import (
    get_league as sleeper_get_league,
    get_users as sleeper_get_users,
    get_rosters as sleeper_get_rosters,
    get_matchups as sleeper_get_matchups,
)
from .base import LeagueInfo, MatchupRow, Provider, RosterInfo, UserInfo


class SleeperProvider(Provider):
    def get_league(self, league_id: str, season: int) -> LeagueInfo:
        l = sleeper_get_league(league_id)
        return LeagueInfo(league_id=str(league_id), season=int(season), name=l.get("name") or "Sleeper League")

    def get_users(self, league_id: str, season: int) -> List[UserInfo]:
        users = sleeper_get_users(league_id)
        return [
            UserInfo(
                user_id=str(u["user_id"]),
                display_name=u.get("display_name") or "",
                avatar=u.get("avatar"),
                metadata=u.get("metadata") or {},
            )
            for u in users
        ]

    def get_rosters(self, league_id: str, season: int) -> List[RosterInfo]:
        rosters = sleeper_get_rosters(league_id)
        out: List[RosterInfo] = []
        for r in rosters:
            out.append(RosterInfo(
                roster_id=str(r["roster_id"]),
                owner_id=str(r.get("owner_id")) if r.get("owner_id") is not None else None,
                players=[str(p) for p in (r.get("players") or [])],
                starters=[str(p) for p in (r.get("starters") or [])],
                reserve=[str(p) for p in (r.get("reserve") or [])],
                metadata=r.get("metadata") or {},
                settings=r.get("settings") or {},
            ))
        return out

    def get_matchups(self, league_id: str, season: int, week: int) -> List[MatchupRow]:
        m = sleeper_get_matchups(league_id, week)
        out: List[MatchupRow] = []
        for row in m or []:
            out.append(MatchupRow(
                matchup_id=int(row.get("matchup_id") or 0),
                roster_id=str(row.get("roster_id")),
                points=float(row.get("points") or 0.0),
                players=[str(p) for p in (row.get("players") or [])],
                starters=[str(p) for p in (row.get("starters") or [])],
                starters_points=[float(x) for x in (row.get("starters_points") or [])],
                players_points={str(k): float(v) for k, v in (row.get("players_points") or {}).items()},
            ))
        return out
