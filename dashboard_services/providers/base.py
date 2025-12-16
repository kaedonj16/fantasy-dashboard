from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class LeagueInfo:
    league_id: str
    season: int
    name: str


@dataclass
class UserInfo:
    user_id: str
    display_name: str
    avatar: Optional[str]  # URL or None
    metadata: Dict[str, Any]


@dataclass
class RosterInfo:
    roster_id: str
    owner_id: Optional[str]
    players: List[str]  # canonical player ids (your app’s ids)
    starters: List[str]  # canonical
    reserve: List[str]  # canonical
    metadata: Dict[str, Any]
    settings: Dict[str, Any]


@dataclass
class MatchupRow:
    matchup_id: int
    roster_id: str
    points: float
    players: List[str]  # canonical ids
    starters: List[str]  # canonical ids
    starters_points: List[float]
    players_points: Dict[str, float]  # canonical id -> pts


class Provider(Protocol):
    def get_league(self, league_id: str, season: int) -> LeagueInfo: ...

    def get_users(self, league_id: str, season: int) -> List[UserInfo]: ...

    def get_rosters(self, league_id: str, season: int) -> List[RosterInfo]: ...

    def get_matchups(self, league_id: str, season: int, week: int) -> List[MatchupRow]: ...
