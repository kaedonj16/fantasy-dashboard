"""Stable provider contract, metadata, capabilities, and public errors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Protocol

LEAGUE = "league"
USERS = "users"
ROSTERS = "rosters"
STARTERS = "starters"
MATCHUPS = "matchups"
STANDINGS = "standings"
TRANSACTIONS = "transactions"
TRADES = "trades"
DRAFTS = "drafts"
DRAFT_RESULTS = "draft_results"
TRADED_PICKS = "traded_picks"
FUTURE_PICKS = "future_picks"
BRACKET = "bracket"
HISTORY = "history"
SCORING_SETTINGS = "scoring_settings"
ROSTER_SETTINGS = "roster_settings"


class ProviderError(Exception):
    """Safe base error for fantasy data providers."""


class ProviderNotFoundError(ProviderError):
    pass


class ProviderAuthenticationError(ProviderError):
    pass


class LeagueNotFoundError(ProviderError):
    pass


class ProviderUnavailableError(ProviderError):
    pass


class UnsupportedCapabilityError(ProviderError):
    pass


@dataclass(frozen=True)
class ProviderMetadata:
    key: str
    display_name: str
    auth_type: str
    enabled: bool = True
    capabilities: FrozenSet[str] = field(default_factory=frozenset)


class FantasyProvider(Protocol):
    metadata: ProviderMetadata

    def supports(self, capability: str) -> bool: ...
    def get_league(self, league_id: str, season: int) -> dict[str, Any]: ...
    def get_users(self, league_id: str, season: int) -> list[dict[str, Any]]: ...
    def get_rosters(self, league_id: str, season: int) -> list[dict[str, Any]]: ...
    def get_matchups(self, league_id: str, season: int, week: int) -> list[dict[str, Any]]: ...


class ProviderAdapter:
    metadata: ProviderMetadata

    def supports(self, capability: str) -> bool:
        return capability in self.metadata.capabilities

    def _unsupported(self, capability: str):
        raise UnsupportedCapabilityError(
            f"{self.metadata.display_name} does not support {capability}."
        )


# Legacy typed wrappers retained for callers of providers.sleeper.
@dataclass
class LeagueInfo:
    league_id: str
    season: int
    name: str


@dataclass
class UserInfo:
    user_id: str
    display_name: str
    avatar: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class RosterInfo:
    roster_id: str
    owner_id: Optional[str]
    players: List[str]
    starters: List[str]
    reserve: List[str]
    metadata: Dict[str, Any]
    settings: Dict[str, Any]


@dataclass
class MatchupRow:
    matchup_id: int
    roster_id: str
    points: float
    players: List[str]
    starters: List[str]
    starters_points: List[float]
    players_points: Dict[str, float]


Provider = FantasyProvider
