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


def test_flatten_list_wrapped_selected_position():
    """Per-team roster resource wraps selected_position in single-element lists."""
    rp = [
        [{"player_id": "5"}, {"name": {"full": "Patrick Mahomes"}}, {"editorial_team_abbr": "KC"}],
        [{"selected_position": [{"position": "QB"}]}],
    ]
    meta, sel = yahoo_api._flatten_yahoo_player(rp)
    assert meta["player_id"] == "5"
    assert sel == "QB"


def test_flatten_triple_nested_player_from_team_roster():
    rp = [[
        [{"player_id": "9"}, {"name": {"full": "Travis Kelce"}}, {"editorial_team_abbr": "KC"}],
        [{"selected_position": [{"position": "TE"}]}],
    ]]
    meta, sel = yahoo_api._flatten_yahoo_player(rp)
    assert meta["player_id"] == "9"
    assert sel == "TE"


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


# ── draft results (keeper drafted-round) ─────────────────────────────────────

def _draft_results_payload(results):
    block = {str(i): {"draft_result": r} for i, r in enumerate(results)}
    block["count"] = len(results)
    return {"fantasy_content": {"league": [{}, {"draft_results": block}]}}


def test_get_draft_results_maps_player_key_to_round(monkeypatch):
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_players",
                        lambda: {"111": {"yahoo_id": "5"}, "222": {"yahoo_id": "6"}})
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    payload = _draft_results_payload([
        {"round": "1", "pick": "1", "player_key": "nfl.p.5"},
        {"round": "7", "pick": "80", "player_key": "nfl.p.6"},
        {"round": "3", "pick": "30", "player_key": "nfl.p.999"},   # unmapped -> skipped
        {"pick": "5", "player_key": "nfl.p.5"},                    # no round -> skipped
    ])
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda t, p, params=None: payload)
    out = yahoo_api.get_draft_results(2026, "12345", "tok")
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    assert out == {"111": 1, "222": 7}   # mapped by yahoo_id; unmapped/round-less dropped


def test_league_key_for_season_uses_that_years_game_key(monkeypatch):
    yahoo_api._season_key_map.clear()
    monkeypatch.setattr(
        yahoo_api, "_nfl_game_keys",
        lambda token: [(2026, "461"), (2025, "449")],
    )
    assert yahoo_api._league_key_for_season("123", 2025, "tok") == "449.l.123"
    assert yahoo_api._league_key_for_season("123", 2026, "tok") == "461.l.123"


def test_yahoo_league_exists_for_season(monkeypatch):
    yahoo_api._season_key_map.clear()
    monkeypatch.setattr(
        yahoo_api, "_nfl_game_keys",
        lambda token: [(2026, "461"), (2025, "449")],
    )

    def fake_get(token, path, params=None):
        if "449.l.123" in path:
            return {"fantasy_content": {"league": [{"name": "Dynasty"}]}}
        raise RuntimeError("missing")

    monkeypatch.setattr(yahoo_api, "_yahoo_get", fake_get)
    assert yahoo_api.yahoo_league_exists_for_season("tok", "123", 2025) is True
    assert yahoo_api.yahoo_league_exists_for_season("tok", "123", 2024) is False


# ── teams / roster collection indexing (Yahoo is 0-based) ────────────────────

def _teams_payload(team_entries):
    """Yahoo league/teams shape: teams keyed 0..n-1 plus count."""
    block = {str(i): {"team": t} for i, t in enumerate(team_entries)}
    block["count"] = len(team_entries)
    return {"fantasy_content": {"league": [{}, {"teams": block}]}}


def test_extract_teams_uses_zero_based_keys():
    teams = [
        [{"team_id": "1", "name": "Alpha"}],
        [{"team_id": "2", "name": "Beta"}],
    ]
    out = yahoo_api._extract_teams(_teams_payload(teams))
    assert len(out) == 2
    assert yahoo_api._team_attr(out[0], "name") == "Alpha"
    assert yahoo_api._team_attr(out[1], "name") == "Beta"


def test_extract_teams_finds_teams_on_league_index_zero():
    """Some Yahoo sub-resource payloads attach teams to league[0], not league[1]."""
    teams = [[{"team_id": "1", "name": "Only"}]]
    block = {str(i): {"team": t} for i, t in enumerate(teams)}
    block["count"] = 1
    payload = {
        "fantasy_content": {
            "league": [
                {"league_key": "449.l.99", "name": "Test", "teams": block},
            ]
        }
    }
    out = yahoo_api._extract_teams(payload)
    assert len(out) == 1
    assert yahoo_api._team_attr(out[0], "name") == "Only"


def test_get_rosters_uses_team_key_when_team_id_missing(monkeypatch):
    team = [[
        {"team_key": "449.l.99.t.7"},
        {"name": "Key Only"},
        {"managers": [{"manager": {"guid": "g7"}}]},
    ]]
    payload = _teams_payload([team])
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda *a, **k: payload)
    monkeypatch.setattr(yahoo_api, "_league_key_for_season", lambda *a, **k: "449.l.99")
    rosters = yahoo_api.get_rosters(2026, "99", "tok")
    assert len(rosters) == 1
    assert rosters[0]["roster_id"] == 7


def test_diagnose_league_reports_parse_counts(monkeypatch):
    teams = [_realistic_yahoo_team(i, f"T{i}", with_roster=True) for i in range(1, 3)]
    payload = _teams_payload(teams)
    meta_payload = {"fantasy_content": {"league": [{"name": "L", "num_teams": "2", "draft_status": "postdraft"}]}}
    paths = {
        "league/449.l.99/teams;out=roster,stats,standings": payload,
        "league/449.l.99/teams": payload,
        "league/449.l.99": meta_payload,
    }
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_players", lambda: {"11111": {"yahoo_id": "5"}})
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda tok, path, params=None: paths[path])
    monkeypatch.setattr(yahoo_api, "_league_key_for_season", lambda *a, **k: "449.l.99")
    report = yahoo_api.diagnose_league(2026, "99", "tok")
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    assert report["ok"] is True
    assert report["extracted_team_count"] == 2
    assert report["parsed_rosters_count"] == 2
    assert report["teams"][0]["resolved_players"] == 1


def test_extract_roster_players_uses_zero_based_keys():
    roster = {
        "roster": {
            "players": {
                "0": {"player": [[{"player_id": "5"}, {"name": {"full": "First"}}]]},
                "1": {"player": [[{"player_id": "6"}, {"name": {"full": "Second"}}]]},
                "count": 2,
            }
        }
    }
    team_data = [roster]
    out = yahoo_api._extract_roster_players(team_data)
    assert len(out) == 2
    meta0, _ = yahoo_api._flatten_yahoo_player(out[0])
    meta1, _ = yahoo_api._flatten_yahoo_player(out[1])
    assert meta0["player_id"] == "5"
    assert meta1["player_id"] == "6"


def _realistic_yahoo_team(team_id, name, *, with_roster=False):
    """Yahoo team shape when ``;out=roster,stats,standings`` is requested."""
    meta = [
        {"team_key": f"449.l.99.t.{team_id}"},
        {"team_id": str(team_id)},
        {"name": name},
        {"managers": [{"manager": {"guid": f"g{team_id}", "nickname": f"Owner {team_id}"}}]},
    ]
    parts = [meta]
    if with_roster:
        parts.append({
            "team_standings": {
                "outcome_totals": {"wins": "1", "losses": "0", "ties": "0"},
                "points_for": "120.5",
                "points_against": "99.1",
            }
        })
        parts.append({
            "roster": {
                "players": {
                    "0": {"player": [[
                        {"player_id": "5"},
                        {"name": {"full": "Patrick Mahomes"}},
                        {"display_position": "QB"},
                        {"editorial_team_abbr": "KC"},
                    ]]},
                    "count": 1,
                }
            }
        })
    return parts


def test_team_attr_reads_nested_metadata_with_subresources():
    team = _realistic_yahoo_team(3, "Champions", with_roster=True)
    assert yahoo_api._team_attr(team, "team_id") == "3"
    assert yahoo_api._team_attr(team, "name") == "Champions"
    standings = yahoo_api._team_attr(team, "team_standings") or {}
    assert standings.get("points_for") == "120.5"


def test_get_users_returns_distinct_roster_ids(monkeypatch):
    teams = [_realistic_yahoo_team(i, f"Team {i}") for i in range(1, 4)]
    payload = _teams_payload(teams)
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda *a, **k: payload)
    monkeypatch.setattr(yahoo_api, "_league_key_for_season", lambda *a, **k: "449.l.99")
    users = yahoo_api.get_users(2026, "99", "tok")
    assert len(users) == 3
    assert {u["roster_id"] for u in users} == {1, 2, 3}
    assert users[0]["metadata"]["team_name"] == "Team 1"


def test_get_rosters_maps_nested_team_players(monkeypatch):
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_players", lambda: {"11111": {"yahoo_id": "5"}})
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    team = _realistic_yahoo_team(7, "Nested Roster Team", with_roster=True)
    payload = _teams_payload([team])
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda *a, **k: payload)
    monkeypatch.setattr(yahoo_api, "_league_key_for_season", lambda *a, **k: "449.l.99")
    rosters = yahoo_api.get_rosters(2026, "99", "tok")
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    assert len(rosters) == 1
    assert rosters[0]["roster_id"] == 7
    assert rosters[0]["players"] == ["11111"]
    assert rosters[0]["settings"]["wins"] == 1
    assert rosters[0]["settings"]["fpts"] == 120


def test_get_rosters_prefetches_team_resource_when_bulk_roster_is_empty(monkeypatch):
    """Bulk teams;out=roster often omits players — hydrate from team/{key}/roster."""
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_players", lambda: {"11111": {"yahoo_id": "5"}})
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    bulk_team = _realistic_yahoo_team(1, "Shell Only", with_roster=True)
    bulk_team[2]["roster"] = {"coverage_type": "week", "week": "1"}
    player_entry = [[
        {"player_id": "5"}, {"name": {"full": "Patrick Mahomes"}},
        {"display_position": "QB"}, {"editorial_team_abbr": "KC"},
    ], [{"selected_position": [{"position": "QB"}]}]]
    team_roster_response = {
        "fantasy_content": {
            "team": [
                [{"team_key": "449.l.99.t.1"}],
                {"roster": {"players": {"0": {"player": player_entry}, "count": 1}}},
            ]
        }
    }

    def fake_get(tok, path, params=None):
        if path.startswith("team/"):
            return team_roster_response
        if "teams" in path:
            return _teams_payload([bulk_team])
        if path == "league/449.l.99":
            return {"fantasy_content": {"league": [{"current_week": "1"}]}}
        raise KeyError(path)

    monkeypatch.setattr(yahoo_api, "_yahoo_get", fake_get)
    monkeypatch.setattr(yahoo_api, "_league_key_for_season", lambda *a, **k: "449.l.99")
    rosters = yahoo_api.get_rosters(2026, "99", "tok")
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    assert len(rosters) == 1
    assert rosters[0]["players"] == ["11111"]


def _yahoo_team_with_list_standings(team_id, name, *, with_roster=False):
    """Yahoo sometimes wraps ``team_standings`` (and managers) in single-element lists."""
    meta = [
        {"team_key": f"449.l.99.t.{team_id}"},
        {"team_id": str(team_id)},
        {"name": name},
        {"managers": [[{"manager": [{"guid": f"g{team_id}", "nickname": f"Owner {team_id}"}]}]]},
    ]
    parts = [meta]
    if with_roster:
        parts.append({
            "team_standings": [{
                "outcome_totals": [{"wins": "2", "losses": "1", "ties": "0"}],
                "points_for": "250.3",
                "points_against": "210.0",
            }]
        })
        parts.append({
            "roster": {
                "players": {
                    "0": {"player": [[
                        {"player_id": "5"},
                        {"name": {"full": "Patrick Mahomes"}},
                        {"display_position": "QB"},
                        {"editorial_team_abbr": "KC"},
                    ]]},
                    "count": 1,
                }
            }
        })
    return parts


def test_get_rosters_handles_list_wrapped_standings(monkeypatch):
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_players", lambda: {"11111": {"yahoo_id": "5"}})
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    team = _yahoo_team_with_list_standings(4, "List Standings", with_roster=True)
    payload = _teams_payload([team])
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda *a, **k: payload)
    monkeypatch.setattr(yahoo_api, "_league_key_for_season", lambda *a, **k: "449.l.99")
    rosters = yahoo_api.get_rosters(2026, "99", "tok")
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    assert rosters[0]["settings"]["wins"] == 2
    assert rosters[0]["settings"]["losses"] == 1
    assert rosters[0]["settings"]["fpts"] == 250


def test_diagnose_league_ok_with_list_wrapped_standings(monkeypatch):
    teams = [_yahoo_team_with_list_standings(i, f"T{i}", with_roster=True) for i in range(1, 3)]
    payload = _teams_payload(teams)
    meta_payload = {"fantasy_content": {"league": [{"name": "L", "num_teams": "2", "draft_status": "postdraft"}]}}
    paths = {
        "league/449.l.99/teams;out=roster,stats,standings": payload,
        "league/449.l.99/teams": payload,
        "league/449.l.99": meta_payload,
    }
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_players", lambda: {"11111": {"yahoo_id": "5"}})
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    monkeypatch.setattr(yahoo_api, "_yahoo_get", lambda tok, path, params=None: paths[path])
    monkeypatch.setattr(yahoo_api, "_league_key_for_season", lambda *a, **k: "449.l.99")
    report = yahoo_api.diagnose_league(2026, "99", "tok")
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    assert report["ok"] is True
    assert report["extracted_team_count"] == 2
    assert report["teams"][0]["wins"] == "2"
    assert report["teams"][0]["points_for"] == "250.3"
