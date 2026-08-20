"""Low-risk adapters around the established Sleeper, ESPN, and Yahoo modules."""
from __future__ import annotations

from .base import *

_COMMON = frozenset({LEAGUE, USERS, ROSTERS, STARTERS, MATCHUPS, STANDINGS,
                     TRANSACTIONS, TRADES, DRAFTS, DRAFT_RESULTS, HISTORY,
                     SCORING_SETTINGS, ROSTER_SETTINGS})


class SleeperProvider(ProviderAdapter):
    metadata = ProviderMetadata("sleeper", "Sleeper", "username", capabilities=
        _COMMON | frozenset({TRADED_PICKS, FUTURE_PICKS, BRACKET}))

    def get_league(self, league_id, season):
        from dashboard_services.api import get_league
        return get_league(league_id)
    def get_users(self, league_id, season):
        from dashboard_services.api import get_users
        return get_users(league_id)
    def get_rosters(self, league_id, season):
        from dashboard_services.api import get_rosters
        return get_rosters(league_id)
    def get_matchups(self, league_id, season, week):
        from dashboard_services.api import get_matchups
        return get_matchups(league_id, week)
    def get_traded_picks(self, league_id, season):
        from dashboard_services.api import get_traded_picks
        return get_traded_picks(league_id)
    def get_bracket(self, league_id, season, kind):
        from dashboard_services.api import get_bracket
        return get_bracket(league_id, kind)
    def get_drafts(self, league_id, season):
        from dashboard_services.api import get_drafts
        return get_drafts(league_id)
    def get_transactions(self, league_id, season, week):
        from dashboard_services.api import get_transactions
        return get_transactions(league_id, week) or []
    def get_league_globals(self, league_id, season):
        return None  # get_league retains Sleeper's historical global side effect.


class ESPNProvider(ProviderAdapter):
    metadata = ProviderMetadata("espn", "ESPN", "league_id", capabilities=_COMMON | frozenset({BRACKET}))
    def _api(self):
        from . import espn_api
        return espn_api
    def get_league(self, league_id, season): return self._api().get_league(season, league_id)
    def get_users(self, league_id, season): return self._api().get_users(season, league_id)
    def get_rosters(self, league_id, season): return self._api().get_rosters(season, league_id)
    def get_matchups(self, league_id, season, week): return self._api().get_matchups(season, league_id, week)
    def get_traded_picks(self, league_id, season): return []
    def get_bracket(self, league_id, season, kind): return self._api().espn_get_bracket_like(league_id=league_id, season=season, kind=kind)
    def get_drafts(self, league_id, season): return self._api().get_drafts(season, league_id)
    def get_transactions(self, league_id, season, week): return self._api().get_transactions(season, league_id, week)
    def get_league_globals(self, league_id, season): return self._api().get_league_globals(season, league_id)


class YahooProvider(ProviderAdapter):
    metadata = ProviderMetadata("yahoo", "Yahoo", "oauth", capabilities=_COMMON | frozenset({BRACKET}))
    def _call(self, name, league_id, season, *args):
        from . import yahoo_api
        from dashboard_services.platform_api import _yahoo_token
        return getattr(yahoo_api, name)(season, league_id, *args, _yahoo_token(league_id, season))
    def get_league(self, league_id, season): return self._call("get_league", league_id, season)
    def get_users(self, league_id, season): return self._call("get_users", league_id, season)
    def get_rosters(self, league_id, season): return self._call("get_rosters", league_id, season)
    def get_matchups(self, league_id, season, week): return self._call("get_matchups", league_id, season, week)
    def get_traded_picks(self, league_id, season): return []
    def get_bracket(self, league_id, season, kind):
        from . import yahoo_api
        from dashboard_services.platform_api import _yahoo_token
        return yahoo_api.get_bracket_like(league_id, season, kind, _yahoo_token(league_id, season))
    def get_drafts(self, league_id, season): return self._call("get_drafts", league_id, season)
    def get_transactions(self, league_id, season, week): return self._call("get_transactions", league_id, season, week)
    def get_league_globals(self, league_id, season): return self._call("get_league_globals", league_id, season)
