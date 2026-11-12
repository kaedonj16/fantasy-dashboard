from datetime import datetime, timedelta, timezone

import pytest

pytest.importorskip("flask")
pd = pytest.importorskip("pandas")


def _activity_row(kind, minutes_ago, data):
    return {
        "kind": kind,
        "week": 4,
        "ts": datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
        "data": data,
    }


def test_since_last_visit_excludes_viewers_own_transactions(monkeypatch):
    import app as appmod

    # Keep this endpoint test offline. The application's first-request hook
    # otherwise starts the daily data job and may wait on the Sleeper API in CI.
    monkeypatch.setattr(
        appmod, "daily_completed", datetime.now(appmod.EASTERN).date(), raising=False
    )
    ctx = {
        "rosters": [],
        "activity_df": pd.DataFrame([
            _activity_row("waiver", 1, {"rid": "7", "name": "Mine", "adds": [{"name": "A"}]}),
            _activity_row("trade", 2, {"teams": [{"rid": "7", "name": "Mine"}, {"rid": "8", "name": "Them"}]}),
            _activity_row("waiver", 3, {"rid": "9", "name": "Other", "adds": [{"name": "B"}]}),
        ]),
    }
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *args: ctx)
    appmod.app.config["TESTING"] = True

    with appmod.app.test_client() as client:
        response = client.get(
            "/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id=7&since=1"
        )

    payload = response.get_json()
    assert payload["trades"] == 0
    assert payload["waivers"] == 1
    assert [item["text"] for item in payload["items"]] == ["Other added B"]


def test_signed_in_visit_uses_account_baseline(monkeypatch):
    import app as appmod
    import dashboard_services.accounts as accounts

    monkeypatch.setattr(
        appmod, "daily_completed", datetime.now(appmod.EASTERN).date(), raising=False
    )
    previous_visit = datetime.now(timezone.utc) - timedelta(minutes=5)
    ctx = {
        "rosters": [],
        "activity_df": pd.DataFrame([
            _activity_row("waiver", 10, {"rid": "9", "name": "Old", "adds": [{"name": "A"}]}),
        ]),
    }
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *args: ctx)
    monkeypatch.setattr(
        accounts,
        "consume_league_visit",
        lambda *args: {"last_visit_at": previous_visit, "roster_id": "7", "roster_snapshot": []},
    )
    appmod.app.config["TESTING"] = True

    with appmod.app.test_client() as client:
        with client.session_transaction() as signed_in:
            signed_in["account_id"] = 42
        response = client.get(
            "/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id=7&since=1"
        )

    payload = response.get_json()
    assert payload["account_scoped"] is True
    assert payload["previous_roster"] == []
    assert payload["items"] == []


def test_slv_value_map_cached_across_calls(monkeypatch):
    import app as appmod

    monkeypatch.setattr(
        appmod, "daily_completed", datetime.now(appmod.EASTERN).date(), raising=False
    )
    table = [{"id": "p1", "value": 123.456, "sf_value": 200.0}]
    monkeypatch.setattr(appmod, "get_model_value_table_cached", lambda *a, **k: table)
    monkeypatch.setattr(appmod, "get_players_index_global", lambda *a, **k: {
        "p1": {"full_name": "Player One", "position": "RB", "injury_status": "Questionable"},
    })
    appmod._SLV_VMAP_CACHE["key"] = None
    appmod._SLV_VMAP_CACHE["map"] = None

    m1 = appmod._slv_value_map()
    m2 = appmod._slv_value_map()
    assert m1 is m2  # built once for the same underlying table

    ctx = {"rosters": [{"roster_id": 7, "players": ["p1"]}], "activity_df": pd.DataFrame([])}
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *args: ctx)
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as client:
        r1 = client.get("/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id=7")
        r2 = client.get("/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id=7")
    assert r1.get_json()["roster"] == r2.get_json()["roster"] == [
        {"pid": "p1", "name": "Player One", "pos": "RB", "team": "",
         "headshot": "", "value": 123.5, "injury": "Questionable"}
    ]


def test_since_last_visit_items_carry_assets_and_baseline(monkeypatch):
    import app as appmod

    monkeypatch.setattr(
        appmod, "daily_completed", datetime.now(appmod.EASTERN).date(), raising=False
    )
    ctx = {
        "rosters": [],
        "activity_df": pd.DataFrame([
            _activity_row("trade", 2, {
                "teams": [
                    {"roster_id": 8, "name": "Them",
                     "gets": [{"name": "Bijan Robinson", "pos": "RB", "pid": "11"}],
                     "sends": [{"name": "Brock Bowers", "pos": "TE", "pid": "22"}]},
                    {"roster_id": 9, "name": "Other",
                     "gets": [{"name": "Brock Bowers", "pos": "TE", "pid": "22"}],
                     "sends": [{"name": "Bijan Robinson", "pos": "RB", "pid": "11"}]},
                ],
                "draft_picks": [
                    {"season": 2027, "round": 1, "owner_id": 8, "previous_owner_id": 9},
                ],
            }),
            _activity_row("waiver", 3, {
                "rid": "9", "name": "Other",
                "adds": [{"name": "Tyler Allgeier", "pos": "RB", "pid": "33"}],
                "drops": [{"name": "Zonovan Knight", "pos": "RB", "pid": "44"}],
            }),
        ]),
    }
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *args: ctx)
    appmod.app.config["TESTING"] = True

    with appmod.app.test_client() as client:
        response = client.get(
            "/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id=7&since=1"
        )

    payload = response.get_json()
    assert payload["since_ms"] == 1
    trade, waiver = payload["items"]
    # The summary text is unchanged for older consumers.
    assert trade["text"] == "Trade: Them and Other"
    assert waiver["text"] == "Other added Tyler Allgeier"
    them, other = trade["teams"]
    assert them["gets"][0] == {"name": "Bijan Robinson", "pos": "RB", "pid": "11"}
    # The pick lands in the receiving team's gets, labelled by year + round.
    assert them["gets"][1]["name"] == "2027 1st"
    assert them["gets"][1]["pick"] is True
    assert other["sends"][0]["name"] == "Bijan Robinson"
    assert waiver["team"] == "Other"
    assert waiver["adds"] == [{"name": "Tyler Allgeier", "pos": "RB", "pid": "33"}]
    assert waiver["drops"] == [{"name": "Zonovan Knight", "pos": "RB", "pid": "44"}]


def test_roster_snapshot_carries_team_and_headshot(monkeypatch):
    import app as appmod

    monkeypatch.setattr(
        appmod, "daily_completed", datetime.now(appmod.EASTERN).date(), raising=False
    )
    monkeypatch.setattr(appmod, "get_model_value_table_cached", lambda *a, **k: [
        {"id": "p1", "value": 500.0, "sf_value": 600.0},
    ])
    monkeypatch.setattr(appmod, "get_players_index_global", lambda *a, **k: {
        "p1": {"full_name": "Player One", "position": "WR", "team": "CIN",
               "espnHeadshot": "https://img.example/p1.png", "injury_status": ""},
    })
    appmod._SLV_VMAP_CACHE["key"] = None
    appmod._SLV_VMAP_CACHE["map"] = None
    ctx = {"rosters": [{"roster_id": 7, "players": ["p1"]}], "activity_df": pd.DataFrame([])}
    monkeypatch.setattr(appmod, "get_league_ctx_from_cache", lambda *args: ctx)
    appmod.app.config["TESTING"] = True

    with appmod.app.test_client() as client:
        response = client.get(
            "/api/since-last-visit?platform=sleeper&league_id=L&season=2026&roster_id=7"
        )

    (row,) = response.get_json()["roster"]
    assert row["team"] == "CIN"
    assert row["headshot"] == "https://img.example/p1.png"


def test_slv_wrap_stacks_in_front_of_mobile_dock():
    # The digest is a modal: on phones its bottom sheet (and the Got it
    # footer) must paint over the bottom dock (--z-chrome) and the assistant
    # launcher, not slide underneath them. It shares the modal tier with the
    # player/team modals, which are appended to the body after the wrap and
    # so still open on top of the digest.
    import re
    from pathlib import Path

    css = (Path(__file__).resolve().parent.parent / "static" / "dashboard.css").read_text()

    def _z(var):
        m = re.search(rf"--{var}:\s*(\d+)", css)
        assert m, f"--{var} not defined"
        return int(m.group(1))

    assert _z("z-modal") > _z("z-chrome")
    block = re.search(r"\.slv-wrap \{([^}]*)\}", css)
    assert block, ".slv-wrap rule missing"
    assert "z-index: var(--z-modal)" in block.group(1)


def test_slv_card_neutralizes_os_card_spacing():
    # The card element also carries .os-card, whose margin-bottom lifted
    # the phone bottom sheet off the bottom of the screen (a strip of
    # page showed under it) and whose padding added dead space inside
    # the sheet under the footer. .slv-card must zero both; the
    # head/body/foot own the card's spacing.
    import re
    from pathlib import Path

    css = (Path(__file__).resolve().parent.parent / "static" / "dashboard.css").read_text()
    block = re.search(r"\.slv-card \{([^}]*)\}", css)
    assert block, ".slv-card rule missing"
    assert "margin: 0" in block.group(1)
    assert "padding: 0" in block.group(1)
