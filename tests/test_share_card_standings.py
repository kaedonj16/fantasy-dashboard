"""Share card must tolerate production standings_map seed ints.

Production ``build_standings_map`` returns ``{roster_id: seed:int}``. Treating
those values as record dicts 500'd ``/share-card`` with:
AttributeError: 'int' object has no attribute 'get'.
"""
from dashboard_services.ai.context_builders import share_card_header_fields


def _roster(**overrides):
    base = {
        "roster_id": 1,
        "owner_id": "u1",
        "players": [],
        "metadata": {"team_name": "Gridiron Goats"},
        "settings": {
            "wins": 3, "losses": 1, "ties": 0,
            "fpts": 412, "fpts_decimal": 50,
            "fpts_against": 380, "fpts_against_decimal": 20,
        },
    }
    base.update(overrides)
    return base


def test_share_card_header_accepts_seed_int_standings_map():
    roster = _roster()
    ctx = {
        "standings_map": {1: 2},  # seed int — production shape
        "roster_map": {"1": "Gridiron Goats"},
        "rosters": [roster],
    }
    user = {"user_id": "u1", "display_name": "Alex", "username": "alex"}
    team_name, record, pf, pa = share_card_header_fields(
        ctx, roster, "1", owner_name="Alex", user=user,
    )
    assert team_name == "Gridiron Goats"
    assert record == "3–1"
    assert pf == 412.5
    assert pa == 380.2


def test_share_card_header_accepts_string_key_seed_int():
    roster = _roster()
    ctx = {
        "standings_map": {"1": 1},
        "roster_map": {1: "Goats"},
        "rosters": [roster],
    }
    team_name, record, pf, pa = share_card_header_fields(
        ctx, roster, "1", owner_name="Alex",
    )
    assert team_name == "Gridiron Goats"
    assert record == "3–1"
    assert pf == 412.5


def test_share_card_header_accepts_dict_standings_map():
    roster = _roster(settings={})
    ctx = {
        "standings_map": {
            "1": {
                "team_name": "Fixture FC",
                "wins": 5, "losses": 2, "ties": 0,
                "pf": 800.4, "pa": 710.1,
            },
        },
        "rosters": [roster],
    }
    team_name, record, pf, pa = share_card_header_fields(
        ctx, roster, "1", owner_name="Alex",
    )
    assert team_name == "Fixture FC"
    assert record == "5–2"
    assert pf == 800.4
    assert pa == 710.1


def test_share_card_header_falls_back_to_owner_name():
    roster = {"roster_id": 1, "settings": {"wins": 0, "losses": 0}}
    ctx = {"standings_map": {1: 4}, "roster_map": {}, "rosters": [roster]}
    team_name, record, pf, pa = share_card_header_fields(
        ctx, roster, "1", owner_name="Alex",
    )
    assert team_name == "Alex"
    assert record == "0–0"
    assert pf == 0.0
    assert pa == 0.0
