"""ESPN mSettings trade deadline maps onto league_settings for Season Hub."""
from types import SimpleNamespace

from dashboard_services.providers import espn_api


def test_espn_trade_deadline_date_mapped_to_ts(monkeypatch):
    # Mid-November-ish deadline as epoch ms (ESPN's usual unit).
    deadline_ms = 1_762_905_600_000

    class Request:
        def league_get(self, params):
            return {
                "settings": {
                    "scoringSettings": {"scoringItems": []},
                    "rosterSettings": {"lineupSlotCounts": {}},
                    "tradeSettings": {"deadlineDate": deadline_ms},
                }
            }

    league = SimpleNamespace(
        settings=SimpleNamespace(
            scoring_type="ppr",
            position_slot_counts={},
            playoff_team_count=6,
            reg_season_count=14,
        ),
        teams=[object()] * 10,
        espn_request=Request(),
    )
    monkeypatch.setattr(espn_api, "_league", lambda season, league_id: league)
    out = espn_api.get_league_globals(2026, "12345")
    assert out["league_settings"]["trade_deadline_ts"] == int(deadline_ms / 1000)
    assert out["league_settings"]["type"] == 0


def test_espn_missing_trade_deadline_omits_key(monkeypatch):
    class Request:
        def league_get(self, params):
            return {
                "settings": {
                    "scoringSettings": {"scoringItems": []},
                    "rosterSettings": {"lineupSlotCounts": {}},
                }
            }

    league = SimpleNamespace(
        settings=SimpleNamespace(
            scoring_type="ppr",
            position_slot_counts={},
            playoff_team_count=4,
            reg_season_count=14,
        ),
        teams=[object()] * 8,
        espn_request=Request(),
    )
    monkeypatch.setattr(espn_api, "_league", lambda season, league_id: league)
    out = espn_api.get_league_globals(2026, "12345")
    assert "trade_deadline_ts" not in out["league_settings"]
