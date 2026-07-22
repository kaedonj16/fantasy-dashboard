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


# ── draft_analysis ADP ───────────────────────────────────────────────────────

def test_draft_analysis_block_dict_form():
    entry = [[{"player_id": "5"}], {"draft_analysis": {"average_pick": "3.4"}}]
    assert yahoo_api._draft_analysis_block(entry) == {"average_pick": "3.4"}


def test_draft_analysis_block_list_form():
    entry = [[{"player_id": "5"}],
             {"draft_analysis": [{"average_pick": "3.4"}, {"average_round": "1.2"}]}]
    da = yahoo_api._draft_analysis_block(entry)
    assert da["average_pick"] == "3.4" and da["average_round"] == "1.2"


def test_draft_analysis_block_missing():
    assert yahoo_api._draft_analysis_block([[{"player_id": "5"}]]) is None


def _players_page(players):
    """Wrap player entries in Yahoo's nested players-collection shape."""
    block = {str(i): {"player": p} for i, p in enumerate(players)}
    block["count"] = len(players)
    return {"fantasy_content": {"league": [{}, {"players": block}]}}


def test_get_draft_analysis_adp_maps_and_paginates(monkeypatch):
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_players",
                        lambda: {"111": {"yahoo_id": "5"}, "222": {"yahoo_id": "6"}})
    yahoo_api._yahoo_id_to_canonical.cache_clear()

    def entry(pid, ap):
        return [[{"player_id": pid}, {"name": {"full": "P"}}, {"editorial_team_abbr": "KC"},
                 {"display_position": "RB"}],
                {"draft_analysis": {"average_pick": ap}}]

    # First page full (25) so the loop continues; second page short so it stops.
    page1 = [entry("5", "3.3")] + [entry("nomatch", "9.9")] * 24
    page2 = [entry("6", "12.1")]

    calls = {"n": 0}

    def fake_get(token, path, params=None):
        calls["n"] += 1
        return _players_page(page1 if "start=0;" in path else page2)

    monkeypatch.setattr(yahoo_api, "_yahoo_get", fake_get)
    adp = yahoo_api.get_draft_analysis_adp(2026, "12345", "tok", max_players=50)
    yahoo_api._yahoo_id_to_canonical.cache_clear()

    assert adp["111"] == 3.3     # mapped via yahoo_id 5
    assert adp["222"] == 12.1    # picked up from page 2 -> pagination worked
    assert calls["n"] == 2       # exactly two pages fetched


def test_get_draft_analysis_adp_skips_zero_and_unmapped(monkeypatch):
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_players", lambda: {"111": {"yahoo_id": "5"}})
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    yahoo_api._name_pos_to_canonical.cache_clear()
    monkeypatch.setattr(yahoo_api, "load_players_index", lambda: {})

    page = [
        [[{"player_id": "5"}, {"name": {"full": "P"}}], {"draft_analysis": {"average_pick": "0"}}],
        [[{"player_id": "999"}, {"name": {"full": "Z"}}], {"draft_analysis": {"average_pick": "4.0"}}],
    ]
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda t, p, params=None: _players_page(page))
    adp = yahoo_api.get_draft_analysis_adp(2026, "12345", "tok", max_players=25)
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    yahoo_api._name_pos_to_canonical.cache_clear()
    assert adp == {}   # zero-pick dropped; unmapped id dropped
