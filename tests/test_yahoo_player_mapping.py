"""Yahoo Phase 2 - exact player mapping via the yahoo_id crosswalk.

Skips cleanly when the full app stack isn't installed (matches the other
integration-style tests); runs in CI.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("bs4")
yahoo_api = pytest.importorskip("dashboard_services.providers.yahoo_api")


def test_flatten_dict_form():
    meta, sel = yahoo_api._flatten_yahoo_player({"player_id": "5", "name": {"full": "X"}})
    assert meta["player_id"] == "5"
    assert sel is None


def test_flatten_positional_list_form():
    rp = [
        [{"player_key": "nfl.p.5"}, {"player_id": "5"},
         {"name": {"full": "Patrick Mahomes"}}, {"editorial_team_abbr": "KC"}],
        {"selected_position": {"position": "BN"}},
    ]
    meta, sel = yahoo_api._flatten_yahoo_player(rp)
    assert meta["player_id"] == "5"
    assert meta["name"]["full"] == "Patrick Mahomes"
    assert meta["editorial_team_abbr"] == "KC"
    assert sel == "BN"


def test_crosswalk_built_from_feed(monkeypatch):
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_players",
                        lambda: {"11111": {"yahoo_id": "5"}, "22222": {"yahoo_id": 9}, "333": {}})
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    xwalk = yahoo_api._yahoo_id_to_canonical()
    assert xwalk == {"5": "11111", "9": "22222"}   # numeric yahoo_id coerced to str; no-id skipped
    yahoo_api._yahoo_id_to_canonical.cache_clear()


def test_resolve_prefers_exact_yahoo_id(monkeypatch):
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_players", lambda: {"11111": {"yahoo_id": "5"}})
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    # A wrong name/team still resolves correctly because the id is exact.
    assert yahoo_api._resolve_player("Wrong Name", "RB", "ZZZ", yahoo_id="5") == "11111"
    yahoo_api._yahoo_id_to_canonical.cache_clear()


def test_resolve_falls_back_to_name_when_id_unknown(monkeypatch):
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_players", lambda: {})   # empty crosswalk
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    yahoo_api._name_pos_to_canonical.cache_clear()
    monkeypatch.setattr(yahoo_api, "load_players_index",
                        lambda: {"99": {"name": "Bijan Robinson", "pos": "RB", "team": "ATL"}})
    assert yahoo_api._resolve_player("Bijan Robinson", "RB", "ATL", yahoo_id="404") == "99"
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    yahoo_api._name_pos_to_canonical.cache_clear()
