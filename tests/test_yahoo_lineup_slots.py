"""Yahoo starters must come back in slot order with their real slots.

Regression (Kaedon's Yahoo league, 2026-10-01): Yahoo returns roster rows
in its own display order, not in roster_positions order, and
``_split_yahoo_lineup`` discarded each player's ``selected_position``
after bucketing bench/IR. Index-based consumers (the Lineup Lab) then
paired starters with slots by list position: a WR landed in an RB slot
and a K in the TE slot, and every per-slot swap pool was wrong.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("bs4")
yahoo_api = pytest.importorskip("dashboard_services.providers.yahoo_api")


# Canonical pid -> (yahoo_id, display_position, nfl team)
ROSTER = {
    "101": ("1", "QB", "NE"),
    "102": ("2", "RB", "IND"),
    "103": ("3", "WR", "KC"),
    "104": ("4", "TE", "ARI"),
    "105": ("5", "RB", "TB"),
    "106": ("6", "K", "LAR"),
    "107": ("7", "WR", "CAR"),
    "108": ("8", "WR", "CIN"),
    "110": ("10", "RB", "DET"),
    "111": ("11", "TE", "KC"),
    "112": ("12", "RB", "SF"),
}
POS = {pid: spec[1] for pid, spec in ROSTER.items()}
POS["106"] = "K"
POS["BUF"] = "DEF"

# Yahoo's return order in the bug report: QB, RB, WR, TE, RB, K, then the
# flex WR, the second WR, DEF, bench, IR. NOT roster_positions order.
YAHOO_ORDER = [
    ("101", "QB"), ("102", "RB"), ("103", "WR"), ("104", "TE"),
    ("105", "RB"), ("106", "K"), ("107", "W/R/T"), ("108", "WR"),
    ("BUF", "DEF"), ("110", "BN"), ("111", "BN"), ("112", "IR+"),
]


def _row(pid, slot):
    if pid == "BUF":
        yid, pos, team, name = "9", "DEF", "Buf", "Buffalo Bills"
    else:
        yid, pos, team = ROSTER[pid]
        name = f"Player {pid}"
    return [[
        {"player_id": yid}, {"name": {"full": name}},
        {"display_position": pos}, {"editorial_team_abbr": team},
    ], {"selected_position": {"position": slot}}]


@pytest.fixture
def crosswalk(monkeypatch):
    import dashboard_services.api as api
    monkeypatch.setattr(api, "get_nfl_players", lambda: {
        pid: {"yahoo_id": spec[0]} for pid, spec in ROSTER.items()
    })
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    yield
    yahoo_api._yahoo_id_to_canonical.cache_clear()


def test_split_orders_starters_by_actual_slot(crosswalk):
    raw = [_row(pid, slot) for pid, slot in YAHOO_ORDER]
    players, starters, slots, reserve = yahoo_api._split_yahoo_lineup_detailed(raw)
    # Roster order is Yahoo's; only the starter list is re-seated.
    assert players == [pid for pid, _slot in YAHOO_ORDER]
    assert starters == ["101", "102", "105", "103", "108",
                        "104", "107", "106", "BUF"]
    assert slots == ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
    assert reserve == ["112"]
    # Every published slot is legal for the player sitting in it.
    from data_building.lineup_lab import _slot_eligible_positions
    for pid, slot in zip(starters, slots):
        assert POS[pid] in _slot_eligible_positions(slot), (pid, slot)


def test_split_wrapper_returns_same_starter_order(crosswalk):
    raw = [_row(pid, slot) for pid, slot in YAHOO_ORDER]
    _players, starters, _reserve = yahoo_api._split_yahoo_lineup(raw)
    assert starters == ["101", "102", "105", "103", "108",
                        "104", "107", "106", "BUF"]


def test_split_all_bn_fallback_has_no_claimed_slots(monkeypatch):
    import dashboard_services.api as api
    ids = {str(10000 + i): {"yahoo_id": str(i)} for i in range(12)}
    monkeypatch.setattr(api, "get_nfl_players", lambda: ids)
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    rows = []
    for i in range(12):
        rows.append([[
            {"player_id": str(i)}, {"name": {"full": f"Player {i}"}},
            {"display_position": "RB"}, {"editorial_team_abbr": "KC"},
        ], {"selected_position": {"position": "BN"}}])
    players, starters, slots, reserve = yahoo_api._split_yahoo_lineup_detailed(rows)
    yahoo_api._yahoo_id_to_canonical.cache_clear()
    assert starters == players[:9]
    assert slots == [""] * 9
    assert reserve == []
