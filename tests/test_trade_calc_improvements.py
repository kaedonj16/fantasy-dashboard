"""Trade Calculator improvements: Fit components, deterministic counter,
unvalued honesty, and the Playoff Impact lineup-after block.

Contracts pinned here:

- fit.value_balance is a pure function of the eval's own diff vs the
  larger side (10 even, 0 at a 40% gap), present for any graded
  two-sided trade with no league context at all.
- An ungraded trade (any unvalued asset) gets NO fit components and NO
  counter: its totals exclude real assets, so a score would be fiction.
  This extends the can't-grade gate from test_trade_eval_ungraded_veto.
- fit.acceptance is the archetype engine's own estimate (the suggestions
  engine's logistic on the depth-adjusted ratio, nudged by partner
  needs/surplus), labeled on the Front Office scale; fit.roster_fit is
  built from the depth model's outputs (starter-gap needs filled, depth
  warnings charged). Both require the premium league context, like the
  analyst block they sit beside.
- counter_suggestion names ONE asset whose calibrated value best closes
  the viewer's gap when the viewer is light beyond the fair band: a
  future pick from the calibrated table (dynasty only) or a player from
  the partner's roster. Within the band, or in redraft with no partner
  pool, there is no counter.
- _lineup_after_block prices the optimal lineup with
  power_score.starter_lineup_points over the sim's week maps, diffs the
  exact optimal assignment for enters/leaves, and reports the headline
  room's rank change from the Teams page ranking only when it moves.
"""
from __future__ import annotations

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


TABLE = [
    {"id": "1", "name": "Alpha WR", "position": "WR", "team": "NYJ",
     "value": 1000.0, "redraft_value_1qb": 1000.0},
    {"id": "2", "name": "Beta RB", "position": "RB", "team": "DEN",
     "value": 600.0, "redraft_value_1qb": 600.0},
    {"id": "3", "name": "Gamma WR", "position": "WR", "team": "GB",
     "value": 1000.0, "redraft_value_1qb": 1000.0},
    {"id": "4", "name": "Delta RB", "position": "RB", "team": "KC",
     "value": 300.0, "redraft_value_1qb": 300.0},
]

PICK_TABLE = {
    "2027_1_mid": 1500.0,
    "2027_2_mid": 610.0,
    "2027_3_mid": 120.0,
}


@pytest.fixture
def env(monkeypatch):
    import app as appmod
    import dashboard_services.player_value_history as pvh

    monkeypatch.setattr(
        pvh, "load_current_values_from_db",
        lambda: [dict(r) for r in TABLE],
    )
    monkeypatch.setattr(
        appmod, "load_pick_value_table", lambda **k: dict(PICK_TABLE)
    )
    monkeypatch.setattr(appmod, "_tier_thresholds_cached", lambda *a, **k: [])
    # The handler's id->row map is cached on the value source's refresh
    # timestamp, which the patched loader never advances (same pattern
    # as test_trade_analyst_rank_parity.py).
    monkeypatch.setattr(appmod, "_TRADE_EVAL_IDMAP_CACHE", {})
    return appmod


@pytest.fixture
def client(env, offline_client):
    return offline_client


def _post(client, **payload):
    payload.setdefault("season", 2026)
    payload.setdefault("league_type", "1qb")
    payload.setdefault("league_size", 10)
    payload.setdefault("scoring_format", "ppr")
    payload.setdefault("scoring_type", "dynasty")
    res = client.post("/api/trade-eval", json=payload)
    assert res.status_code == 200, res.get_data(as_text=True)
    return res.get_json()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def test_fit_value_balance_anchors(env):
    f = env._fit_value_balance
    assert f(0, 1000) == 10
    assert f(400, 1000) == 0
    assert f(200, 1000) == 5
    assert f(1000, 1000) == 0  # clamped, never negative


def test_fit_acceptance_label_scale(env):
    f = env._fit_acceptance_label
    assert f(60) == "Likely"
    assert f(90) == "Likely"
    assert f(59) == "Possible"
    assert f(40) == "Possible"
    assert f(39) == "Long shot"
    assert f(5) == "Long shot"
    assert f(None) is None


def test_fit_playoff_score(env):
    f = env._fit_playoff_score
    assert f({"playoff_pct": 0.0, "avg_ppg": 0.0}) == 5
    assert f({"playoff_pct": 0.1, "avg_ppg": 0.6}) == 6
    assert f({"playoff_pct": 5.0, "avg_ppg": 2.0}) == 10  # clamped
    assert f({"playoff_pct": -6.0, "avg_ppg": -2.0}) == 0  # clamped
    assert f(None) is None
    assert f({}) == 5


def test_pick_display_name(env):
    f = env._pick_display_name
    assert f("2027_2_mid") == "2027 2nd (Mid)"
    assert f("2028_1_early") == "2028 1st (Early)"
    assert f("2027_3_late") == "2027 3rd (Late)"
    assert f("2027_1_04") == "2027 1.04"


# ---------------------------------------------------------------------------
# Value balance: no league context needed
# ---------------------------------------------------------------------------

def test_value_balance_even_trade(client):
    data = _post(client, side_a_players=["1"], side_b_players=["3"])
    assert data["graded"] is True
    assert data["fit"]["value_balance"] == 10
    # No league context: roster components stay hidden, never defaulted.
    assert data["fit"]["roster_fit"] is None
    assert data["fit"]["acceptance"] is None
    assert data["fit"]["acceptance_rate"] is None
    assert data["fit"]["acceptance_label"] is None
    assert data["counter_suggestion"] is None


def test_value_balance_matches_response_numbers(client):
    data = _post(client, side_a_players=["1"], side_b_players=["2"])
    a = data["side_a"]["effective_total"]
    b = data["side_b"]["effective_total"]
    baseline = max(a, b, 1.0)
    pct = abs(a - b) / baseline
    expected = int(max(0, min(10, round(10 * (1 - pct / 0.40)))))
    assert data["fit"]["value_balance"] == expected


# ---------------------------------------------------------------------------
# Ungraded: the can't-grade gate extends to Fit and the counter
# ---------------------------------------------------------------------------

def test_ungraded_trade_gets_no_fit_or_counter(client):
    # Viewer is massively light AND one asset is unknown: still no fit,
    # no counter - the totals exclude a real asset.
    data = _post(client, side_a_players=["4"], side_b_players=["1", "999"])
    assert data["graded"] is False
    assert data["fit"] == {
        "value_balance": None,
        "roster_fit": None,
        "acceptance": None,
        "acceptance_rate": None,
        "acceptance_label": None,
    }
    assert data["counter_suggestion"] is None


# ---------------------------------------------------------------------------
# Counter: deterministic, from the calibrated pick table
# ---------------------------------------------------------------------------

def test_counter_names_pick_that_closes_the_gap(client):
    # Viewer gets 300, gives 1000: gap 700, fair band 70. The 2027 2nd
    # at 610 is the only candidate inside [0.4x, 1.6x] of the gap (the
    # 1st at 1500 overshoots the window, the 3rd at 120 undershoots).
    data = _post(
        client, viewer_side="a",
        side_a_players=["4"], side_b_players=["1"],
    )
    s = data["counter_suggestion"]
    assert s is not None
    assert s["type"] == "pick"
    assert s["id"].startswith("2027_2_")
    assert s["value"] == 610.0
    assert s["add_to_side"] == "a"
    gap = data["side_b"]["effective_total"] - data["side_a"]["effective_total"]
    assert s["gap"] == round(gap, 1)


def test_counter_none_within_fair_band(client):
    data = _post(client, side_a_players=["1"], side_b_players=["3"])
    assert data["counter_suggestion"] is None


def test_counter_none_in_redraft_without_partner(client):
    # Redraft: picks are not tradable and no partner roster is loaded,
    # so even a lopsided deal gets no counter here.
    data = _post(
        client, viewer_side="a", scoring_type="redraft",
        side_a_players=["4"], side_b_players=["1"],
    )
    assert data["graded"] is True
    assert data["counter_suggestion"] is None


# ---------------------------------------------------------------------------
# Acceptance + roster fit from the premium league context
# ---------------------------------------------------------------------------

CTX_TABLE = [
    {"id": "qb1", "name": "Viewer QB", "position": "QB", "team": "BUF", "value": 800.0},
    {"id": "rb1", "name": "Viewer RB1", "position": "RB", "team": "MIA", "value": 800.0},
    {"id": "rb2", "name": "Viewer RB2", "position": "RB", "team": "MIA", "value": 60.0},
    {"id": "rb3", "name": "Viewer RB3", "position": "RB", "team": "MIA", "value": 50.0},
    {"id": "wr1", "name": "Viewer WR1", "position": "WR", "team": "CIN", "value": 800.0},
    {"id": "wr2", "name": "Viewer WR2", "position": "WR", "team": "CIN", "value": 750.0},
    {"id": "wr3", "name": "Viewer WR3", "position": "WR", "team": "CIN", "value": 700.0},
    {"id": "te1", "name": "Viewer TE", "position": "TE", "team": "KC", "value": 700.0},
    {"id": "pqb", "name": "Partner QB", "position": "QB", "team": "NYJ", "value": 700.0},
    {"id": "prb", "name": "Partner RB", "position": "RB", "team": "NYJ", "value": 700.0},
    {"id": "pwr", "name": "Partner WR", "position": "WR", "team": "NYJ", "value": 300.0},
    {"id": "pwr2", "name": "Partner WR2", "position": "WR", "team": "NYJ", "value": 650.0},
    {"id": "pte", "name": "Partner TE", "position": "TE", "team": "NYJ", "value": 600.0},
]

CTX_ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF",
                        "BN", "BN", "BN"]


def _fit_ctx():
    rosters = [
        {"roster_id": 1, "players": ["qb1", "rb1", "rb2", "rb3",
                                     "wr1", "wr2", "wr3", "te1"]},
        {"roster_id": 2, "players": ["pqb", "prb", "pwr", "pwr2", "pte"]},
    ] + [{"roster_id": rid, "players": []} for rid in range(3, 11)]
    return {
        "rosters": rosters,
        "roster_positions": CTX_ROSTER_POSITIONS,
        "model_value_table": [dict(r) for r in CTX_TABLE],
    }


@pytest.fixture
def ctx_env(env, monkeypatch):
    import dashboard_services.player_value_history as pvh

    monkeypatch.setattr(
        pvh, "load_current_values_from_db",
        lambda: [dict(r) for r in CTX_TABLE],
    )
    monkeypatch.setattr(env, "has_premium_for_viewer", lambda *a, **k: True)
    monkeypatch.setattr(
        env, "get_league_ctx_from_cache", lambda **k: _fit_ctx())
    monkeypatch.setattr(env, "_TRADE_EVAL_IDMAP_CACHE", {})
    return env


def test_acceptance_and_roster_fit_from_context(ctx_env, client):
    # Viewer sends their ONLY startable RB (800) for a 300 WR.
    # Roster fit: neutral 5, no need filled (WR is covered), one danger
    # depth warning (zero startable RBs left for 2 RB slots) at -3 -> 2.
    # Acceptance: the partner receives 800 for 300, deep into Likely.
    data = _post(
        client, league_id="L1", viewer_roster_id="1",
        opponent_roster_id="2", viewer_side="a", skip_ai=True,
        side_a_players=["pwr"], side_b_players=["rb1"],
    )
    assert data["graded"] is True
    assert data["fit"]["roster_fit"] == 2
    rate = data["fit"]["acceptance_rate"]
    assert rate is not None and rate >= 60
    assert data["fit"]["acceptance_label"] == "Likely"
    assert data["fit"]["acceptance"] == int(max(0, min(10, round(rate / 10.0))))


def test_fit_context_components_hidden_without_premium(env, client, monkeypatch):
    # Same league payload, but the viewer is not premium: the roster
    # components stay hidden (value balance still shows).
    monkeypatch.setattr(env, "has_premium_for_viewer", lambda *a, **k: False)
    monkeypatch.setattr(
        env, "get_league_ctx_from_cache", lambda **k: _fit_ctx())
    import dashboard_services.player_value_history as pvh
    monkeypatch.setattr(
        pvh, "load_current_values_from_db",
        lambda: [dict(r) for r in CTX_TABLE],
    )
    data = _post(
        client, league_id="L1", viewer_roster_id="1",
        opponent_roster_id="2", viewer_side="a", skip_ai=True,
        side_a_players=["pwr"], side_b_players=["rb1"],
    )
    assert data["fit"]["value_balance"] is not None
    assert data["fit"]["roster_fit"] is None
    assert data["fit"]["acceptance"] is None


# ---------------------------------------------------------------------------
# Lineup-after block (Playoff Impact)
# ---------------------------------------------------------------------------

def _sim_state():
    ppg = {
        "qb1": 20.0, "rb1": 18.0, "rb2": 12.0, "rb3": 6.0,
        "wr1": 9.0, "wr2": 8.0, "wr3": 5.0, "te1": 10.0, "wrS": 22.0,
    }
    pos = {
        "qb1": "QB", "rb1": "RB", "rb2": "RB", "rb3": "RB",
        "wr1": "WR", "wr2": "WR", "wr3": "WR", "te1": "TE", "wrS": "WR",
    }
    ppg_map = {pid: {"ppg": v, "pos": pos[pid]} for pid, v in ppg.items()}
    return {
        "ppg_map": ppg_map,
        "pos_map": dict(pos),
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX",
                             "BN", "BN"],
        "roster_pid_map": {7: ["qb1", "rb1", "rb2", "rb3",
                               "wr1", "wr2", "wr3", "te1"]},
        "week_ppg_maps": {9: dict(ppg_map), 10: dict(ppg_map)},
    }


def test_lineup_after_block(env, monkeypatch):
    import utils.utils as uu

    names = {
        "qb1": "QB One", "rb1": "RB One", "rb2": "RB Two", "rb3": "RB Three",
        "wr1": "WR One", "wr2": "WR Two", "wr3": "WR Three",
        "te1": "TE One", "wrS": "Star WR",
    }
    monkeypatch.setattr(
        uu, "load_players_index",
        lambda: {pid: {"name": n} for pid, n in names.items()},
    )
    out = env._lineup_after_block(_sim_state(), 7, ["rb1"], ["wrS"])
    assert out is not None
    # Before: 20 + 18+12 + 9+8 + 10 + FLEX rb3 6 = 83
    # After:  20 + 12+6 + 22+9 + 10 + FLEX wr2 8 = 87
    assert out["ppg_before"] == 83.0
    assert out["ppg_after"] == 87.0
    assert out["enters"] == [
        {"id": "wrS", "name": "Star WR", "slot": "WR1"}]
    assert out["leaves"] == [
        {"id": "rb1", "name": "RB One", "slot": "RB1"}]
    # No league ctx: no room rank, never invented.
    assert out["room"] is None


def test_lineup_after_block_room_rank(env, monkeypatch):
    import utils.utils as uu

    monkeypatch.setattr(uu, "load_players_index", lambda: {})
    table = [
        {"id": "wr1", "position": "WR", "value": 100.0},
        {"id": "wr2", "position": "WR", "value": 90.0},
        {"id": "wrS", "position": "WR", "value": 950.0},
        {"id": "x1", "position": "WR", "value": 500.0},
        {"id": "x2", "position": "WR", "value": 480.0},
        {"id": "y1", "position": "WR", "value": 300.0},
        {"id": "y2", "position": "WR", "value": 280.0},
    ]
    ctx = {
        "rosters": [
            {"roster_id": 7, "players": ["qb1", "rb1", "rb2", "rb3",
                                         "wr1", "wr2", "wr3", "te1"]},
            {"roster_id": 8, "players": ["wrS", "x1", "x2"]},
            {"roster_id": 9, "players": ["y1", "y2"]},
        ],
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
        "model_value_table": table,
    }
    out = env._lineup_after_block(_sim_state(), 7, ["rb1"], ["wrS"], ctx)
    assert out is not None
    # Viewer WR room is last (3rd) before, first after adding the 950 WR.
    assert out["room"] == {"position": "WR", "rank_before": 3, "rank_after": 1}


def test_lineup_after_block_unavailable(env):
    assert env._lineup_after_block(None, 7, [], []) is None
    assert env._lineup_after_block({}, 7, [], []) is None


