"""Lineup Lab must seat starters in their real slots for Yahoo leagues.

Regression (2026-10-01): Yahoo returns starters in its own display order
(QB, RB, WR, TE, RB, K, ...) rather than roster_positions order, and the
Lab paired starters with slots by list index. A WR was seated in an RB
slot, a TE in a WR slot, and a K in the TE slot; because each starter's
swap bench is filtered by the slot's eligibility, the K's "TE" slot
offered only TEs and most slots had the wrong (or no) lineup actions.
"""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

import data_building.lineup_lab as lab_mod
import data_building.player_distributions as pd_mod
import data_building.weekly_metrics as wm_mod
import utils.fantasy_scoring as fs_mod
import utils.utils as utils_mod
import dashboard_services.platform_api as platform_api_mod


PROJ = {
    "1": 18.6, "2": 20.1, "3": 11.6, "4": 13.8, "5": 12.6, "6": 8.0,
    "7": 12.3, "8": 13.0, "9": 7.0, "10": 10.0, "11": 9.5, "12": 6.0,
    "13": 5.0, "20": 20.0, "21": 15.0,
}
POS = {
    "1": "QB", "2": "RB", "3": "WR", "4": "TE", "5": "RB", "6": "K",
    "7": "WR", "8": "WR", "9": "DEF", "10": "RB", "11": "WR", "12": "TE",
    "13": "K", "20": "QB", "21": "RB",
}
TEAM = {pid: "BUF" for pid in PROJ}

VIEWER_PLAYERS = [str(i) for i in range(1, 14)]
# Yahoo's return order from the bug report: QB, RB, WR, TE, RB, K, then
# the flex WR, the second WR, and DEF. Not roster_positions order.
YAHOO_STARTERS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
YAHOO_SLOTS = ["QB", "RB", "WR", "TE", "RB", "K", "FLEX", "WR", "DEF"]
ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K",
                    "DEF", "BN", "BN", "BN", "BN"]


def _ctx():
    return {
        "current_week": 5,
        "rosters": [
            {"roster_id": 1, "owner_id": "GUID_A", "league_id": "99",
             "metadata": {"team_name": "My Yahoo Team"},
             "players": list(VIEWER_PLAYERS), "reserve": [], "taxi": None},
            {"roster_id": 4, "owner_id": "GUID_B", "league_id": "99",
             "metadata": {"team_name": "Yahoo Rivals"},
             "players": ["20", "21"], "reserve": [], "taxi": None},
        ],
        "users": [{"user_id": "GUID_B", "display_name": "Rival Manager",
                   "username": "Rival Manager"}],
        "players_index": {
            pid: {"position": POS[pid], "pos": POS[pid], "team": TEAM[pid],
                  "full_name": f"Player {pid}", "injury_status": ""}
            for pid in PROJ
        },
        "players": {},
        "roster_positions": list(ROSTER_POSITIONS),
        "raw_scoring_settings": {},
        "scoring_settings": {},
    }


def _profiles(requests, season, week):
    out = {}
    for req in requests:
        pid = str(req["player_id"])
        mean = float(req["mean"])
        out[pid] = {
            "player_id": pid, "pos": req["pos"], "mean": mean,
            "std": round(2.0 + 0.42 * mean, 2), "skew_alpha": 2.0,
            "dud_risk": 0.0, "n_games": 3.0, "factors": {},
        }
    return out


@pytest.fixture
def lab_mocks(monkeypatch):
    monkeypatch.setattr(pd_mod, "build_profiles", _profiles)
    monkeypatch.setattr(pd_mod, "correlation_pairs",
                        lambda pids, season, ctx=None: {})
    monkeypatch.setattr(wm_mod, "get_usage_trends", lambda season: {})
    monkeypatch.setattr(utils_mod, "load_week_projection",
                        lambda season, week: {})
    monkeypatch.setattr(utils_mod, "load_week_sched",
                        lambda season, week: [{"home": "BUF", "away": "MIA"}])
    monkeypatch.setattr(fs_mod, "weekly_projection_points",
                        lambda raw, pid, scoring, pos="": PROJ.get(str(pid)))


def _matchups(with_slots):
    mine = {"roster_id": 1, "matchup_id": 1, "starters": list(YAHOO_STARTERS),
            "players": list(VIEWER_PLAYERS), "players_points": {}}
    if with_slots:
        mine["starters_slots"] = list(YAHOO_SLOTS)
    return [
        mine,
        {"roster_id": 4, "matchup_id": 1, "starters": ["20", "21"],
         "players": ["20", "21"], "players_points": {}},
    ]


def _build(monkeypatch, with_slots):
    monkeypatch.setattr(
        platform_api_mod, "get_matchups",
        lambda platform, league_id, week, season, **kw: _matchups(with_slots))
    return lab_mod.build_lineup_lab_payload(
        ctx=_ctx(), league_id="99", viewer_roster_id=1,
        season=2026, week=5, platform="yahoo")


def _assert_legal_seating(lineup):
    by_pid = {e["player_id"]: e for e in lineup}
    for entry in lineup:
        eligible = lab_mod._slot_eligible_positions(entry["slot"])
        assert entry["pos"] in eligible, (entry["player_id"], entry["slot"])
    # The exact mis-seatings from the bug report.
    assert by_pid["3"]["slot"] == "WR"   # WR was seated at RB
    assert by_pid["4"]["slot"] == "TE"   # TE was seated at WR
    assert by_pid["5"]["slot"] == "RB"   # RB was seated at WR
    assert by_pid["6"]["slot"] == "K"    # K was seated at TE
    # The two remaining WRs split the WR and FLEX slots; which one sits
    # at FLEX is only knowable from the provider's claimed slots.
    assert {by_pid["7"]["slot"], by_pid["8"]["slot"]} == {"WR", "FLEX"}
    return by_pid


def test_lab_uses_provider_slots_for_yahoo_order(lab_mocks, monkeypatch):
    data = _build(monkeypatch, with_slots=True)
    lineup = data["you"]["lineup"]
    by_pid = _assert_legal_seating(lineup)
    assert by_pid["7"]["slot"] == "FLEX"  # Yahoo says Coker is the flex
    # Display follows slot order, not Yahoo's return order.
    assert [e["player_id"] for e in lineup] == [
        "1", "2", "5", "3", "8", "4", "7", "6", "9"]
    # Swap pools (the lineup actions) follow the real slot: the K's pool
    # holds the bench K only, the TE's pool holds the bench TE.
    k_bench = [b["player_id"] for b in by_pid["6"]["bench"]]
    assert k_bench == ["13"]
    te_bench = [b["player_id"] for b in by_pid["4"]["bench"]]
    assert te_bench == ["12"]
    rb_bench = [b["player_id"] for b in by_pid["2"]["bench"]]
    assert rb_bench == ["10"]
    flex_bench = {b["player_id"] for b in by_pid["7"]["bench"]}
    assert flex_bench == {"10", "11", "12"}


def test_lab_assigns_by_eligibility_without_provider_slots(lab_mocks, monkeypatch):
    # Legacy/cached matchup rows carry no starters_slots; seating must
    # still be legal for an out-of-order starter list.
    data = _build(monkeypatch, with_slots=False)
    _assert_legal_seating(data["you"]["lineup"])


def test_lab_sleeper_slot_order_unchanged(lab_mocks, monkeypatch):
    # Sleeper starters already arrive in slot order; the assignment must
    # reproduce the historical index pairing exactly.
    ordered = ["1", "2", "5", "3", "8", "4", "7", "6", "9"]

    def _sleeper_matchups(platform, league_id, week, season, **kw):
        return [
            {"roster_id": 1, "matchup_id": 1, "starters": list(ordered),
             "players": list(VIEWER_PLAYERS), "players_points": {}},
            {"roster_id": 4, "matchup_id": 1, "starters": ["20", "21"],
             "players": ["20", "21"], "players_points": {}},
        ]

    monkeypatch.setattr(platform_api_mod, "get_matchups", _sleeper_matchups)
    data = lab_mod.build_lineup_lab_payload(
        ctx=_ctx(), league_id="99", viewer_roster_id=1,
        season=2026, week=5, platform="sleeper")
    lineup = data["you"]["lineup"]
    assert [e["player_id"] for e in lineup] == ordered
    assert [e["slot"] for e in lineup] == [
        "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]
