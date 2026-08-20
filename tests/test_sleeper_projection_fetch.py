from utils import utils


class _Response:
    status_code = 200

    def json(self):
        return [{
            "player_id": "9758",
            "stats": {"pass_yd": 250, "pass_td": 1, "pass_int": 1},
            "pts_ppr": 14.57,
            "pts_half_ppr": 14.57,
            "pts_std": 14.57,
        }]


def test_fetch_preserves_sleeper_totals_outside_stats(monkeypatch):
    monkeypatch.setattr(utils.requests, "get", lambda *args, **kwargs: _Response())
    monkeypatch.setattr(utils, "load_players_index", lambda: {"9758": {"pos": "QB"}})

    result = utils.fetch_week_from_sleeper(2026, 1)

    assert result["9758"]["raw_stats"]["pts_ppr"] == 14.57
    assert result["9758"]["ppr"] == 12.0
