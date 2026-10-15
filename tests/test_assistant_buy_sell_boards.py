"""Buy-low / sell-high candidate boards: pool-wide buy/sell questions.

"Who should I buy low?" names no player, so the single-player
buy_sell_hold path declined it with the leaked internal string "That is
not available: no resolved player." (live bug, 2026-10-15). These tests
pin the fix: pool-wide buy/sell phrasings route to leaderboard boards
(board_kind buy_low / sell_high) in BOTH routers, the evidence builder
screens the site's value movers by whether production supports the move,
the deterministic board composer renders them, named-player controls
stay buy_sell_hold, and an unresolvable player question declines in a
natural sentence that asks which player they mean.
Everything external is monkeypatched at the evidence module's lazy shims.
"""
import pytest

pytest.importorskip("openai")
pytestmark = pytest.mark.integration

from dashboard_services.assistant import compose as compose_mod
from dashboard_services.assistant import evidence
from dashboard_services.assistant.router import _keyword_route


def _bundle(**over):
    bundle = {"platform": "sleeper", "league_id": "L1", "season": 2026,
              "league_name": "Blackedraw", "team_name": "Couch", "week": 7,
              "ctx": {"raw_scoring_settings": {"rec": 1.0},
                      "rosters": [], "roster_positions": []},
              "roster": {"players": [], "starters": []},
              "roster_player_ids": set()}
    bundle.update(over)
    return bundle


# -- router keyword fallback ------------------------------------------------

@pytest.mark.parametrize("message,kind", [
    ("Who should I buy low?", "buy_low"),
    ("Who should I sell high?", "sell_high"),
    ("Buy-low candidates", "buy_low"),
    ("Sell-high candidates", "sell_high"),
    ("Who is a buy low right now?", "buy_low"),
    ("Who is undervalued right now?", "buy_low"),
    ("Who is overvalued right now?", "sell_high"),
    ("Who should I buy?", "buy_low"),
    ("Top 5 buy-low candidates", "buy_low"),
])
def test_keyword_pool_buy_sell_is_a_value_board(message, kind):
    r = _keyword_route(message)
    assert r["intent"] == "leaderboard"
    assert r["board_kind"] == kind


@pytest.mark.parametrize("message", [
    "Is Bijan Robinson a buy low?",
    "Should I buy Bijan Robinson?",
    "Should I sell Bijan Robinson?",
    "Buy, sell, or hold on Bijan Robinson?",
])
def test_keyword_named_player_stays_buy_sell_hold(message):
    r = _keyword_route(message)
    assert r["intent"] == "buy_sell_hold"


# -- evidence builder ---------------------------------------------------------

def _mover(pid, name, pos, delta, old, new):
    return {"player_id": pid, "name": name, "position": pos, "team": "T",
            "delta": delta, "old_value": old, "new_value": new}


MOVERS = {
    "used_days": 7,
    "risers": [
        _mover("r1", "Riser One", "RB", 440.0, 1540.0, 1980.0),
        _mover("r2", "Riser Two", "WR", 365.0, 3045.0, 3410.0),
    ],
    "fallers": [
        _mover("f1", "Faller One", "WR", -450.0, 3660.0, 3210.0),
        _mover("f2", "Faller Two", "WR", -500.0, 2900.0, 2400.0),
        _mover("f3", "Faller Three", "RB", -300.0, 2200.0, 1900.0),
        _mover("f4", "Faller Four", "TE", -200.0, 1800.0, 1600.0),
    ],
}

# ppg / expected_ppr_per_game per player (the season line). The weekly
# series below is the recent window the v2 logic actually judges:
# f2's opportunity collapsed with his production (a falling knife, not a
# buy-low), r2's rise is backed by real opportunity growth (earned, not
# a sell-high), r1's recent surplus is TD-propped, f3 has no production
# line at all (unverifiable, never a candidate).
PRODUCTION = {
    "f1": {"ppg": 13.8, "expected_ppr_per_game": 12.9, "games": 6},
    "f2": {"ppg": 8.0, "expected_ppr_per_game": 10.5, "games": 6},
    "f3": {},
    "f4": {"ppg": 12.0, "expected_ppr_per_game": 12.0, "games": 5},
    "r1": {"ppg": 9.8, "expected_ppr_per_game": 11.6, "games": 6},
    "r2": {"ppg": 14.1, "expected_ppr_per_game": 12.4, "games": 6},
}


def _series(rows):
    """Weekly adv-series rows from (week, xppg, over, td_over, tgts, carries)."""
    return [{"week": w, "expected_ppr": x, "ppr_over_expected": o,
             "td_over_expected": td, "w_targets": t, "w_carries": c,
             "w_pass_att": None} for (w, x, o, td, t, c) in rows]


SERIES = {
    # Role rising (targets 8-9 early, 11-13 lately), production lagging
    # the opportunity line over the last 4: the regression buy-low.
    "f1": _series([(1, 15.0, 3.2, 0.0, 8, 0), (2, 14.6, 3.6, 0.0, 9, 0),
                   (3, 14.5, -2.6, 0.0, 11, 0), (4, 13.8, -2.2, 0.0, 12, 0),
                   (5, 13.6, -2.3, 0.0, 12, 0), (6, 13.7, -2.1, 0.0, 13, 0)]),
    # Targets fell from 12 to 6-7 while production fell under the line:
    # production AND opportunity down together, the falling knife.
    "f2": _series([(1, 11.0, -1.0, 0.0, 12, 0), (2, 11.0, -1.5, 0.0, 12, 0),
                   (3, 11.0, -3.5, 0.0, 7, 0), (4, 11.0, -4.0, 0.0, 7, 0),
                   (5, 11.0, -4.5, 0.0, 6, 0), (6, 11.0, -5.0, 0.0, 6, 0)]),
    "f3": [],
    # Flat opportunity, production a touch under the line lately.
    "f4": _series([(1, 12.0, 1.0, 0.0, 8, 0), (2, 12.0, 0.5, 0.0, 8, 0),
                   (3, 12.0, -0.5, 0.0, 8, 0), (4, 12.0, -0.5, 0.0, 8, 0),
                   (5, 12.0, -0.5, 0.0, 8, 0), (6, 12.0, -0.5, 0.0, 8, 0)]),
    # Recent surplus built on touchdowns over expectation (2.1 across
    # the last 4), opportunity flat: the TD-propped sell-high.
    "r1": _series([(1, 10.9, -2.0, 0.0, 0, 12), (2, 10.9, -1.5, 0.0, 0, 12),
                   (3, 10.9, 1.2, 0.2, 0, 12), (4, 10.9, 1.8, 0.5, 0, 12),
                   (5, 10.9, 1.9, 0.6, 0, 12), (6, 10.9, 1.7, 0.8, 0, 12)]),
    # Targets up from 8 to 13-14 with production over the line and no TD
    # inflation: the rise is earned by opportunity growth.
    "r2": _series([(1, 12.4, 1.0, 0.0, 8, 0), (2, 12.4, 1.2, 0.0, 8, 0),
                   (3, 12.4, 1.7, 0.0, 13, 0), (4, 12.4, 1.7, 0.1, 13, 0),
                   (5, 12.4, 1.8, 0.0, 14, 0), (6, 12.4, 1.9, 0.1, 14, 0)]),
}


@pytest.fixture()
def value_env(monkeypatch):
    calls = []

    def fake_movers(**kwargs):
        calls.append(kwargs)
        return dict(MOVERS)

    monkeypatch.setattr(evidence, "get_top_movers", fake_movers)
    monkeypatch.setattr(evidence, "get_model_value_table_cached", list)
    monkeypatch.setattr(
        evidence, "_metrics_for",
        lambda pid, season=None: (dict(PRODUCTION.get(pid, {})), []))
    monkeypatch.setattr(
        evidence, "get_player_weekly_adv_series",
        lambda pid, season=None: [dict(r) for r in SERIES.get(pid, [])])
    monkeypatch.setattr(evidence, "_index", lambda: {})
    monkeypatch.setattr(evidence, "_upcoming_schedule",
                        lambda bundle, team, count=4: [])
    return calls


def test_buy_low_board_screens_fallers_by_production(value_env):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "buy_low"})
    assert pack["available"] is True
    assert pack["label"] == "Buy-low candidates"
    # f2 dropped (falling knife: production AND opportunity fell),
    # f3 dropped (no production line).
    assert [r["name"] for r in pack["board"]] == ["Faller One", "Faller Four"]
    top = pack["board"][0]
    assert top["delta"] == -450.0 and top["value"] == 3210.0
    assert top["baseline_value"] == 3660.0
    assert top["ppg"] == 13.8 and top["expected_ppg"] == 12.9
    assert top["over_expected"] == 0.9
    # The recent window is the judged line now: 11.6 PPG vs 13.9 expected
    # over the last 4, with opportunity up (ratio over 1), and the
    # mispricing score, not the raw delta, sets the order.
    assert top["recent_ppg"] == 11.6 and top["recent_expected_ppg"] == 13.9
    assert top["recent_over_expected"] == -2.3
    assert top["opp_ratio"] == 1.11
    assert top["mechanism"] == (
        "role rising while production lags the opportunity")
    assert top["score"] > pack["board"][1]["score"]
    assert pack["pool_size"] == 2 and pack["movers_examined"] == 4
    assert "2 of 4 fallers qualify" in pack["qualification_text"]
    # The movers call mirrors the /api/value-movers route's arguments.
    call = value_env[0]
    assert call["days"] == 7 and call["min_baseline_value"] == 10
    assert call["min_current_value"] == 20.0 and call["league_type"] == "1qb"
    assert call["current_values"] == {}


def test_sell_high_board_screens_risers_by_production(value_env):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "sell_high"})
    assert pack["available"] is True
    assert pack["label"] == "Sell-high candidates"
    # r2's rise is earned (opportunity up 16% behind production over the
    # line): not a sell-high. r1's recent surplus is TD-propped.
    assert [r["name"] for r in pack["board"]] == ["Riser One"]
    top = pack["board"][0]
    assert top["delta"] == 440.0 and top["over_expected"] == -1.8
    assert top["recent_over_expected"] == 1.65
    assert top["td_over_expected_recent"] == 2.1
    assert top["mechanism"] == (
        "scoring propped by touchdowns over expectation")
    assert "1 of 2 risers qualify" in pack["qualification_text"]


def test_value_board_position_filter(value_env):
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "buy_low", "position": "TE"})
    assert [r["name"] for r in pack["board"]] == ["Faller Four"]


def test_value_board_no_candidates_is_no_data(value_env, monkeypatch):
    monkeypatch.setattr(evidence, "_metrics_for",
                        lambda pid, season=None: ({}, []))
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "buy_low"})
    assert pack["available"] is False
    assert pack["missing"] == "no buy-low candidates right now"


# -- v2 mispricing controls ----------------------------------------------------

def _custom_env(monkeypatch, movers, production, series, index=None):
    monkeypatch.setattr(evidence, "get_top_movers",
                        lambda **kwargs: dict(movers))
    monkeypatch.setattr(evidence, "get_model_value_table_cached", list)
    monkeypatch.setattr(
        evidence, "_metrics_for",
        lambda pid, season=None: (dict(production.get(pid, {})), []))
    monkeypatch.setattr(
        evidence, "get_player_weekly_adv_series",
        lambda pid, season=None: [dict(r) for r in series.get(pid, [])])
    monkeypatch.setattr(evidence, "_index", lambda: dict(index or {}))
    monkeypatch.setattr(evidence, "_upcoming_schedule",
                        lambda bundle, team, count=4: [])


def test_falling_knife_excluded_despite_discount_and_peers(monkeypatch):
    # The knife fell 40% and sits far under his peer median; the peer
    # gap must NOT rescue him, because his opportunity collapsed too.
    movers = {"used_days": 7, "risers": [], "fallers": [
        _mover("k1", "Knife Back", "RB", -800.0, 2000.0, 1200.0),
        _mover("s1", "Steady Back", "RB", -150.0, 2100.0, 1950.0),
        _mover("p1", "Peer Back A", "RB", -60.0, 2600.0, 2540.0),
        _mover("p2", "Peer Back B", "RB", -55.0, 2500.0, 2445.0)]}
    production = {
        "k1": {"ppg": 9.0, "expected_ppr_per_game": 12.0, "games": 6},
        "s1": {"ppg": 12.5, "expected_ppr_per_game": 12.4, "games": 6},
        "p1": {"ppg": 12.6, "expected_ppr_per_game": 12.4, "games": 6},
        "p2": {"ppg": 12.4, "expected_ppr_per_game": 12.3, "games": 6}}
    flat = [(w, 12.4, 0.1, 0.0, 0, 14) for w in range(1, 7)]
    series = {
        # Carries 16 a game early, 8 lately; production fell under the line.
        "k1": _series([(1, 12.0, 1.0, 0.0, 0, 16), (2, 12.0, 0.5, 0.0, 0, 16),
                       (3, 12.0, -2.5, 0.0, 0, 8), (4, 12.0, -3.0, 0.0, 0, 8),
                       (5, 12.0, -3.5, 0.0, 0, 8), (6, 12.0, -4.0, 0.0, 0, 8)]),
        "s1": _series(flat), "p1": _series(flat), "p2": _series(flat)}
    _custom_env(monkeypatch, movers, production, series)
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "buy_low"})
    names = [r["name"] for r in pack["board"]]
    assert "Knife Back" not in names
    assert names[0] == "Steady Back"


def test_peer_gap_headlines_buy_low_and_names_anchor(monkeypatch):
    # The candidate produces and works like two far pricier peers; the
    # peer-median gap, not his modest 10% dip, puts him on top.
    movers = {"used_days": 7, "risers": [], "fallers": [
        _mover("c1", "Peer Match Wideout", "WR", -265.0, 2665.0, 2400.0),
        _mover("d1", "Deep Dip Wideout", "WR", -600.0, 3000.0, 2400.0),
        _mover("a1", "Star Wideout", "WR", -100.0, 5000.0, 4900.0),
        _mover("a2", "Pricey Wideout", "WR", -90.0, 4390.0, 4300.0)]}
    line = {"ppg": 16.5, "expected_ppr_per_game": 16.4, "games": 6}
    production = {"c1": dict(line), "d1": {"ppg": 9.0,
                                          "expected_ppr_per_game": 10.0,
                                          "games": 6},
                  "a1": dict(line), "a2": dict(line)}
    rich = [(w, 16.4, 0.2, 0.0, 11, 0) for w in range(1, 7)]
    series = {"c1": _series(rich), "a1": _series(rich), "a2": _series(rich),
              "d1": _series([(w, 10.0, -1.0, 0.0, 6, 0) for w in range(1, 7)])}
    _custom_env(monkeypatch, movers, production, series)
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "buy_low"})
    top = pack["board"][0]
    assert top["name"] == "Peer Match Wideout"
    assert top["peer_median_value"] == 4600.0
    assert top["peer_anchor_name"] == "Star Wideout"
    assert top["peer_anchor_gap_pct"] == 51.0


def test_opportunity_rising_faller_outranks_flat_faller(monkeypatch):
    movers = {"used_days": 7, "risers": [], "fallers": [
        _mover("rr", "Rising Role Back", "RB", -240.0, 2000.0, 1760.0),
        _mover("ff", "Flat Faller Back", "RB", -280.0, 2000.0, 1720.0)]}
    production = {
        "rr": {"ppg": 12.0, "expected_ppr_per_game": 12.6, "games": 6},
        "ff": {"ppg": 12.2, "expected_ppr_per_game": 12.1, "games": 6}}
    series = {
        # Opportunity up from 9 to 13 lately, production under the line.
        "rr": _series([(1, 13.0, 0.5, 0.0, 0, 9), (2, 13.0, 0.5, 0.0, 0, 9),
                       (3, 13.0, -3.2, 0.0, 0, 13), (4, 13.0, -3.2, 0.0, 0, 13),
                       (5, 13.0, -3.3, 0.0, 0, 13), (6, 13.0, -3.3, 0.0, 0, 13)]),
        # Flat opportunity, production right at the line.
        "ff": _series([(w, 12.1, 0.1, 0.0, 0, 12) for w in range(1, 7)])}
    _custom_env(monkeypatch, movers, production, series)
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "buy_low"})
    assert [r["name"] for r in pack["board"]] == [
        "Rising Role Back", "Flat Faller Back"]
    assert pack["board"][0]["mechanism"] == (
        "role rising while production lags the opportunity")


def test_season_ending_injury_never_a_buy_low(monkeypatch):
    movers = {"used_days": 7, "risers": [], "fallers": [
        _mover("h1", "Hurt Star", "WR", -900.0, 4000.0, 3100.0),
        _mover("s1", "Steady Wideout", "WR", -150.0, 2100.0, 1950.0)]}
    line = {"ppg": 15.0, "expected_ppr_per_game": 14.0, "games": 6}
    production = {"h1": dict(line), "s1": dict(line)}
    rich = [(w, 14.0, 1.0, 0.0, 11, 0) for w in range(1, 7)]
    series = {"h1": _series(rich), "s1": _series(rich)}
    index = {"h1": {"age": 27, "injury_status": "IR"}}
    _custom_env(monkeypatch, movers, production, series, index=index)
    pack, _tools = evidence.build_leaderboard(
        _bundle(), {"board_kind": "buy_low"})
    assert [r["name"] for r in pack["board"]] == ["Steady Wideout"]


# -- compose -------------------------------------------------------------------

def _value_pack():
    return {"type": "leaderboard", "available": True,
            "league": {"platform": "sleeper", "league_id": "L1",
                       "season": 2026, "league_name": "Blackedraw",
                       "team_name": "Couch"},
            "week": 7, "board_kind": "buy_low", "label": "Buy-low candidates",
            "direction": "fallers", "value_days": 7, "pool_size": 2,
            "movers_examined": 4, "board_count": 2, "requested_n": 10,
            "position_filter": None, "team_filter": None,
            "qualification_text": (
                "Value movement over the last 7 days from the site's value "
                "history; value fell while production held at or above "
                "expectation (PPG vs expected PPG). 2 of 4 fallers qualify."),
            "board": [
                {"rank": 1, "player_id": "f1", "name": "Faller One",
                 "team": "T1", "position": "WR", "value": 3210.0, "games": 6,
                 "on_my_roster": False, "delta": -450.0,
                 "baseline_value": 3660.0, "ppg": 13.8,
                 "expected_ppg": 12.9, "over_expected": 0.9},
                {"rank": 2, "player_id": "f4", "name": "Faller Four",
                 "team": "T4", "position": "TE", "value": 1600.0, "games": 5,
                 "on_my_roster": False, "delta": -200.0,
                 "baseline_value": 1800.0, "ppg": 12.0,
                 "expected_ppg": 12.0, "over_expected": 0.0}]}


def _canned(text):
    def fake(system, payload, schema, name):
        return {"verdict": "MODEL VERDICT", "text": text, "receipts": [],
                "caveat": None, "followups": [], "act_link": None}
    return fake


def test_compose_buy_low_board(monkeypatch):
    monkeypatch.setattr(compose_mod, "call_model_json",
                        _canned("A quiet intro that names nobody."))
    ans = compose_mod.compose_answer("leaderboard", "Who should I buy low?",
                                     _value_pack())
    assert ans["verdict"] == "Faller One is the top buy-low"
    assert len(ans["receipts"]) == 2  # one per board row
    assert ans["receipts"][0] == {
        "label": "1. Faller One",
        "value": "value -450 (now 3210); 13.8 PPG (+0.9 vs expected)"}
    assert ("1. Faller One (T1, WR): value -450 to 3210 (was 3660), "
            "13.8 PPG vs 12.9 expected") in ans["text"]
    assert ans["act_link"] == {"href": "/top-movers",
                               "label": "View on Top Movers"}


def test_compose_value_board_recent_line_mechanism_and_peer(monkeypatch):
    monkeypatch.setattr(compose_mod, "call_model_json",
                        _canned("A quiet intro that names nobody."))
    pack = _value_pack()
    pack["board"][0].update({
        "recent_ppg": 11.6, "recent_expected_ppg": 13.9,
        "recent_over_expected": -2.3, "recent_games": 4,
        "mechanism": "role rising while production lags the opportunity",
        "peer_median_value": 4600.0, "peer_anchor_name": "Star Wideout",
        "peer_anchor_value": 4900.0, "peer_anchor_gap_pct": 51.0})
    ans = compose_mod.compose_answer("leaderboard", "Who should I buy low?",
                                     pack)
    assert ("1. Faller One (T1, WR): value -450 to 3210 (was 3660), "
            "11.6 PPG vs 13.9 expected over the last 4; "
            "role rising while production lags the opportunity; "
            "producing like Star Wideout over the last 4, "
            "priced 51% below him") in ans["text"]
    assert ans["receipts"][0] == {
        "label": "1. Faller One",
        "value": ("value -450 (now 3210); 11.6 PPG last 4 "
                  "(-2.3 vs expected); peer median 4600")}


# -- the decline string ---------------------------------------------------------

@pytest.mark.parametrize("intent,missing", [
    ("buy_sell_hold", "no resolved player"),
    ("player", "no resolved player"),
    ("start_sit", "no resolved players"),
    ("breakout_history", "no resolved player"),
])
def test_unresolved_player_decline_asks_who(intent, missing):
    ans = compose_mod.no_data_answer(
        {"type": intent, "available": False, "missing": missing})
    assert "no resolved player" not in ans["text"]
    assert "not available" not in ans["text"]
    assert "which player" in ans["text"]
    assert ans["verdict"] is None


def test_unresolved_buy_sell_hold_decline_via_compose_answer():
    pack = {"type": "buy_sell_hold", "available": False,
            "missing": "no resolved player", "league": {}}
    ans = compose_mod.compose_answer("buy_sell_hold", "Buy, sell, or hold?",
                                     pack)
    assert "buy, sell, or hold call" in ans["text"]
    assert "no resolved player" not in ans["text"]
