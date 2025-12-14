from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# Sleeper implementation (unchanged)
from dashboard_services import api as sleeper_api

# ESPN implementation (new)
from dashboard_services import espn_api


# -----------------------------
# Provider base
# -----------------------------

class ProviderError(Exception):
    pass


class BaseProvider:
    name: str = "base"

    # ---- league context ----
    def get_league(self, league_id: str, season: Optional[int] = None) -> Dict[str, Any]:
        raise NotImplementedError

    def get_scoring_settings(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_effective_scoring_settings(self) -> Dict[str, float]:
        raise NotImplementedError

    def get_roster_positions(self) -> List[str]:
        raise NotImplementedError

    def get_league_settings(self) -> Dict[str, Any]:
        raise NotImplementedError

    def get_total_rosters(self) -> int:
        raise NotImplementedError

    # ---- league data ----
    def get_users(self, league_id: str, season: Optional[int] = None) -> List[dict]:
        raise NotImplementedError

    def get_rosters(self, league_id: str, season: Optional[int] = None) -> List[dict]:
        raise NotImplementedError

    def get_matchups(self, league_id: str, week: int, season: Optional[int] = None) -> List[dict]:
        raise NotImplementedError

    def get_transactions(self, league_id: str, week: int, season: Optional[int] = None) -> List[dict]:
        raise NotImplementedError

    def get_bracket(self, league_id: str, bracket: str) -> List[dict]:
        raise NotImplementedError

    def get_traded_picks(self, league_id: str) -> List[dict]:
        raise NotImplementedError


# -----------------------------
# Sleeper Provider (unchanged)
# -----------------------------

class SleeperProvider(BaseProvider):
    name = "sleeper"

    def get_league(self, league_id: str, season: Optional[int] = None) -> Dict[str, Any]:
        return sleeper_api.get_league(league_id)

    def get_scoring_settings(self) -> Dict[str, Any]:
        return sleeper_api.get_scoring_settings()

    def get_effective_scoring_settings(self) -> Dict[str, float]:
        return sleeper_api.get_effective_scoring_settings()

    def get_roster_positions(self) -> List[str]:
        return sleeper_api.get_roster_positions()

    def get_league_settings(self) -> Dict[str, Any]:
        return sleeper_api.get_league_settings()

    def get_total_rosters(self) -> int:
        return sleeper_api.get_total_rosters()

    def get_users(self, league_id: str, season: Optional[int] = None) -> List[dict]:
        return sleeper_api.get_users(league_id)

    def get_rosters(self, league_id: str, season: Optional[int] = None) -> List[dict]:
        return sleeper_api.get_rosters(league_id)

    def get_matchups(self, league_id: str, week: int, season: Optional[int] = None) -> List[dict]:
        return sleeper_api.get_matchups(league_id, week)

    def get_transactions(self, league_id: str, week: int, season: Optional[int] = None) -> List[dict]:
        return sleeper_api.get_transactions(league_id, week)

    def get_bracket(self, league_id: str, bracket: str, season: Optional[int] = None) -> List[dict]:
        return sleeper_api.get_bracket(league_id, bracket)

    def get_traded_picks(self, league_id: str, season: Optional[int] = None) -> List[dict]:
        return sleeper_api.get_traded_picks(league_id)


# -----------------------------
# ESPN Provider (IMPLEMENTED)
# -----------------------------

class ESPNProvider(BaseProvider):
    name = "espn"

    def get_league(self, league_id: str, season: Optional[int] = None) -> Dict[str, Any]:
        if season is None:
            raise ProviderError("ESPN requires season")
        return espn_api.get_league(season, league_id)

    def get_scoring_settings(self) -> Dict[str, Any]:
        # ESPN scoring is embedded in league settings; not normalized yet
        return {}

    def get_effective_scoring_settings(self) -> Dict[str, float]:
        # Safe fallback so projections don’t explode
        return sleeper_api.get_effective_scoring_settings()

    def get_roster_positions(self) -> List[str]:
        return []

    def get_league_settings(self) -> Dict[str, Any]:
        return {}

    def get_total_rosters(self) -> int:
        return 0

    def get_users(self, league_id: str, season: Optional[int] = None) -> List[dict]:
        if season is None:
            raise ProviderError("ESPN requires season")
        return espn_api.get_users(season, league_id)

    def get_rosters(self, league_id: str, season: Optional[int] = None) -> List[dict]:

        if season is None:
            raise ProviderError("ESPN requires season")
        return espn_api.get_rosters(season, league_id)

    def get_matchups(self, league_id: str, week: int, season: Optional[int] = None) -> List[dict]:
        if season is None:
            raise ProviderError("ESPN requires season")
        return espn_api.get_matchups(season, league_id, week)

    def get_transactions(self, league_id: str, week: int, season: Optional[int] = None) -> List[dict]:
        return []

    def get_bracket(self, league_id: str, bracket: str, season: Optional[int] = None) -> List[dict]:
        return []

    def get_traded_picks(self, league_id: str, season: Optional[int] = None) -> List[dict]:
        return []


# -----------------------------
# Provider selection
# -----------------------------

_SLEEPER = SleeperProvider()
_ESPN = ESPNProvider()

def get_provider(platform: Optional[str]) -> BaseProvider:
    p = (platform or "sleeper").strip().lower()
    if p in ("sleeper", "slp"):
        return _SLEEPER
    if p in ("espn", "espnff"):
        return _ESPN
    raise ProviderError(f"Unknown platform '{platform}'")


# -----------------------------
# Public API (used by app)
# -----------------------------

def avatar_from_users(users: List[dict], owner_id: Optional[str]) -> Optional[str]:
    # ESPN avatars not wired yet; Sleeper behavior preserved
    return sleeper_api.avatar_from_users(users, owner_id)


def get_league(league_id: str, platform: str = "sleeper", season: Optional[int] = None) -> Dict[str, Any]:
    return get_provider(platform).get_league(league_id, season=season)


def get_scoring_settings(platform: str = "sleeper") -> Dict[str, Any]:
    return get_provider(platform).get_scoring_settings()


def get_effective_scoring_settings(platform: str = "sleeper") -> Dict[str, float]:
    return get_provider(platform).get_effective_scoring_settings()


def get_roster_positions(platform: str = "sleeper") -> List[str]:
    return get_provider(platform).get_roster_positions()


def get_league_settings(platform: str = "sleeper") -> Dict[str, Any]:
    return get_provider(platform).get_league_settings()


def get_total_rosters(platform: str = "sleeper") -> int:
    return get_provider(platform).get_total_rosters()


def get_users(league_id: str, platform: str = "sleeper", season: Optional[int] = None) -> List[dict]:
    return get_provider(platform).get_users(league_id, season=season)


def get_rosters(league_id: str, platform: str = "sleeper", season: Optional[int] = None) -> List[dict]:
    return get_provider(platform).get_rosters(league_id, season=season)


def get_matchups(
    league_id: str,
    week: int,
    platform: str = "sleeper",
    season: Optional[int] = None,
) -> List[dict]:
    return get_provider(platform).get_matchups(league_id, week, season=season)


def get_transactions(
    league_id: str,
    week: int,
    platform: str = "sleeper",
    season: Optional[int] = None,
) -> List[dict]:
    return get_provider(platform).get_transactions(league_id, week, season=season)


def get_bracket(league_id: str, bracket: str, platform: str = "sleeper", season: Optional[int] = None,) -> List[dict]:
    return get_provider(platform).get_bracket(league_id, bracket, season=season)


def get_traded_picks(league_id: str, platform: str = "sleeper") -> List[dict]:
    return get_provider(platform).get_traded_picks(league_id)


# -----------------------------
# Platform-agnostic NFL/Tank01
# -----------------------------

def get_nfl_state() -> dict:
    return sleeper_api.get_nfl_state()


def get_nfl_players() -> dict:
    return sleeper_api.get_nfl_players()


def get_nfl_scores_for_date(game_date: str) -> dict:
    return sleeper_api.get_nfl_scores_for_date(game_date)


def build_team_game_lookup(scores_body: dict) -> dict[str, dict]:
    return sleeper_api.build_team_game_lookup(scores_body)


def fetch_tank_boxscore(game_id: str, session=None) -> dict:
    return sleeper_api.fetch_tank_boxscore(game_id, session=session)


def get_nfl_games_for_week_raw(week: int, season: int, season_type: str = "reg") -> list[dict]:
    return sleeper_api.get_nfl_games_for_week_raw(
        week=week, season=season, season_type=season_type
    )


def get_tank01_player_gamelogs(
    tank_player_id: str, season: Optional[int] = None
) -> List[Dict[str, Any]]:
    return sleeper_api.get_tank01_player_gamelogs(
        tank_player_id=tank_player_id, season=season
    )


def fetch_team_game_logs_html(team_abv: str, season: int) -> str:
    return sleeper_api.fetch_team_game_logs_html(team_abv=team_abv, season=season)
