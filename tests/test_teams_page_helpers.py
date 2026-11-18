from dashboard_services.pages import teams_page


def test_teams_page_uses_shared_pick_slot_helpers():
    table = {"2027_1_03": 850}

    assert teams_page._pk_pick_label(2027, 1, 3) == "2027 1.03"
    assert teams_page._pk_pick_value_from_table(table, 2027, 1, 3, 12) == 850


def test_dw_headshot_url_builds_from_espnid_for_rookie_entries():
    # Rookie-class index entries carry an espnID but no stored espnHeadshot.
    meta = {"espnID": "4432784"}
    assert teams_page._dw_headshot_url(meta, 128) == (
        "https://a.espncdn.com/combiner/i?img=/i/headshots/nfl/players/full/"
        "4432784.png&w=128&scale=crop&cquality=100"
    )


def test_dw_headshot_url_prefers_stored_headshot():
    meta = {
        "espnID": "4432784",
        "espnHeadshot": "https://a.espncdn.com/i/headshots/nfl/players/full/999.png",
    }
    url = teams_page._dw_headshot_url(meta, 128)
    assert "999.png" in url
    assert "4432784" not in url


def test_dw_headshot_url_empty_without_any_id():
    assert teams_page._dw_headshot_url({}, 128) == ""
    assert teams_page._dw_headshot_url(None, 128) == ""
    assert teams_page._dw_headshot_url({"espnID": "  "}, 128) == ""
