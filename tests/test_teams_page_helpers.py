from dashboard_services.pages import teams_page


def test_teams_page_uses_shared_pick_slot_helpers():
    table = {"2027_1_03": 850}

    assert teams_page._pk_pick_label(2027, 1, 3) == "2027 1.03"
    assert teams_page._pk_pick_value_from_table(table, 2027, 1, 3, 12) == 850
