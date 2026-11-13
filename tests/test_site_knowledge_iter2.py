"""Iteration 2 (Nov 12 Fleaflicker check) pins for the shared pieces:

- ``gate_package`` (items 1+2): a designated keeper can never be on
  the give side of a surfaced suggestion, and a package that orphans
  the partner's QB2 (the Goodfellas/Caleb case: Maye is a keeper,
  Ward and Mendoza carry no projection in a supplied projections
  layer, so neither is startable per the #3020 orphan guard) fails
  unless the repair returns a startable QB inside the fair band.
  The shapes the site pitched live — Kyren alone (76%) and Watson
  alone (81%) — must not survive as pitched.
- ``positional_room_ranks`` (item 7b): one shared computation behind
  the My Leagues card, Front Office grades, and league grades.
- Lineup Lab ``_projection_lookup`` (item 5): the shared
  ``proj_by_week`` map wins over the Lab's old raw-file rescore (JSN
  read 23.8 in the Lab vs 16.7 in the Hub for non-PPR league 92916).
"""
import pytest

pytest.importorskip("pandas")

from dashboard_services import trade_partner_checks as tpc
from dashboard_services.trade_partner_checks import gate_package

SF_REQ = {"QB": 2, "RB": 2, "WR": 2, "TE": 1}
BARS = {"QB": 400.0, "RB": 300.0, "WR": 300.0, "TE": 180.0}


def _p(pid, pos, value):
    return {"player_id": pid, "id": pid, "position": pos,
            "value": float(value), "name": pid}


# Goodfellas-shaped partner: Caleb + Maye (keeper) startable; Ward and
# Mendoza clear the QB value bar but have NO projection in the
# supplied layer, so the orphan guard keeps them non-startable.
_PARTNER = [
    _p("caleb", "QB", 800), _p("maye", "QB", 850),
    _p("ward", "QB", 500), _p("mendoza", "QB", 480),
    _p("p-rb1", "RB", 600), _p("p-rb2", "RB", 400),
    _p("p-wr1", "WR", 700), _p("p-wr2", "WR", 500),
    _p("p-te1", "TE", 300),
]
_VIEWER = [
    _p("lock", "QB", 700), _p("mccarthy", "QB", 430),
    _p("cousins", "QB", 500),
    _p("kyren", "RB", 780), _p("v-rb2", "RB", 500),
    _p("watson", "WR", 560), _p("jsn", "WR", 900), _p("v-wr3", "WR", 500),
    _p("kincaid", "TE", 350),
]
# Sparse QB-only layer at starter magnitudes, exactly the shape of
# the live Week 9 map in test_trade_partner_checks: non-QB
# startability falls back to the value bars; a QB missing from the
# layer fails closed. The viewer carries three startable QBs (as the
# real room did), so McCarthy — the cheapest spare — is repairable
# under the same spare rule the engines' gates use.
_PROJ = {"maye": 23.0, "caleb": 20.6, "mccarthy": 17.9, "lock": 19.9,
         "cousins": 18.5}


def _gate(package, receive, **kw):
    return gate_package(
        partner_players=_PARTNER, viewer_players=_VIEWER,
        receive=receive, package=package,
        requirements=SF_REQ, value_bars=BARS,
        projections=_PROJ, **kw)


def test_kyren_alone_for_caleb_fails_even_with_repair():
    # Kyren alone orphans the partner QB2; the only repair (McCarthy)
    # pushes the give past the 1.40 fair band, so nothing surfaces.
    ok, pkg, _ev = _gate([_p("kyren", "RB", 780)], [_p("caleb", "QB", 800)])
    assert ok is False
    assert [p["player_id"] for p in pkg] == ["kyren"]


def test_watson_alone_is_never_surfaced_as_pitched():
    # Watson alone orphans the QB2 as well; the gate may only pass it
    # with the QB repair attached (the feed-verified workable shape).
    ok, pkg, ev = _gate([_p("watson", "WR", 560)], [_p("caleb", "QB", 800)])
    assert ok is True
    assert [p["player_id"] for p in pkg] == ["watson", "mccarthy"]
    assert ev is not None and ev["viable"] is True


def test_watson_alone_fails_when_the_repair_qb_is_blocked():
    ok, _pkg, _ev = _gate(
        [_p("watson", "WR", 560)], [_p("caleb", "QB", 800)],
        blocked_repair_ids={"mccarthy"})
    assert ok is False


def test_watson_plus_mccarthy_passes_as_pitched():
    ok, pkg, ev = _gate(
        [_p("watson", "WR", 560), _p("mccarthy", "QB", 430)],
        [_p("caleb", "QB", 800)])
    assert ok is True
    assert [p["player_id"] for p in pkg] == ["watson", "mccarthy"]
    assert ev["viable"] is True


def test_designated_keeper_can_never_be_sent():
    # JSN is a designated keeper: the package fails outright even
    # though the same shape without him is viable, and even when the
    # evaluation itself would pass.
    ok, _pkg, ev = _gate(
        [_p("jsn", "WR", 900), _p("mccarthy", "QB", 430)],
        [_p("caleb", "QB", 800)],
        locked_ids={"jsn"})
    assert ok is False
    assert ev is None


def test_gate_fails_open_only_on_evaluation_error(monkeypatch):
    def _boom(**kw):
        raise RuntimeError("no state")

    monkeypatch.setattr(tpc, "evaluate_package", _boom)
    ok, pkg, ev = _gate([_p("watson", "WR", 560)], [_p("caleb", "QB", 800)])
    assert ok is True and ev is None
    assert [p["player_id"] for p in pkg] == ["watson"]


# --- positional_room_ranks (item 7b) --------------------------------


def _room_ctx():
    def row(pid, pos, val):
        return {"id": pid, "player_id": pid, "name": pid, "position": pos,
                "value": val, "sf_value": val,
                "redraft_value_1qb": val, "redraft_value_sf": val}

    table = [
        row("q1", "QB", 100), row("q2", "QB", 90),
        row("q3", "QB", 95), row("q4", "QB", 10),
        row("q5", "QB", 50), row("q6", "QB", 40),
        row("r1", "RB", 80), row("r2", "RB", 70),
        row("w1", "WR", 85), row("t1", "TE", 60),
    ]
    return {
        "model_value_table": table,
        "rosters": [
            {"roster_id": 1, "players": ["q1", "q2", "r1", "w1", "t1"]},
            {"roster_id": 2, "players": ["q3", "q4", "r2"]},
            {"roster_id": 3, "players": ["q5", "q6"]},
        ],
        "roster_positions": ["QB", "RB", "WR", "TE", "SUPER_FLEX", "BN"],
        "players_map": {},
        "league": {"settings": {}},
    }


def test_positional_room_ranks_shared_computation():
    from dashboard_services.ai.context_builders import positional_room_ranks

    out = positional_room_ranks(_room_ctx())
    assert out["of"] == 3
    # Superflex QB room = top 2: viewer 100+90=190 beats 95+10 and 50+40.
    # (ranks are position-major: {pos: {roster_id_str: rank}}, the
    # shape every consumer reads.)
    assert out["ranks"]["QB"]["1"] == 1
    assert out["ranks"]["QB"]["2"] == 2
    assert out["ranks"]["QB"]["3"] == 3
    # Deterministic: a second consumer reading the same helper agrees.
    again = positional_room_ranks(_room_ctx())
    assert again["ranks"] == out["ranks"]


# --- Lineup Lab projection source (item 5) ---------------------------


def test_lineup_lab_prefers_shared_proj_by_week(monkeypatch):
    import data_building.lineup_lab as lab
    import utils.utils as utils_mod

    # Raw weekly file: JSN with a reception-heavy line that rescoring
    # under a PPR-leaning dict inflates well past the Hub number.
    monkeypatch.setattr(
        utils_mod, "load_week_projection",
        lambda season, week: {"jsn": {"rec": 12, "rec_yd": 140}})
    ctx = {"proj_by_week": {10: {"projections": {"jsn": 16.7}}}}
    lookup = lab._projection_lookup({"rec": 1.0}, 2026, 10, ctx)
    assert lookup("jsn", "WR") == 16.7


def test_lineup_lab_raw_rescore_only_fills_absent_players(monkeypatch):
    import data_building.lineup_lab as lab
    import utils.fantasy_scoring as fs_mod
    import utils.utils as utils_mod

    monkeypatch.setattr(
        utils_mod, "load_week_projection",
        lambda season, week: {"other": {"rec": 5, "rec_yd": 60}})
    # The rescore is stubbed (as in tests/test_lineup_lab.py) so the
    # pin is about WHICH source answers, not the scoring math.
    monkeypatch.setattr(
        fs_mod, "weekly_projection_points",
        lambda raw, pid, scoring, pos="": {"other": 11.0}.get(str(pid)))
    ctx = {"proj_by_week": {10: {"projections": {"jsn": 16.7}}}}
    lookup = lab._projection_lookup({"rec": 1.0}, 2026, 10, ctx)
    assert lookup("jsn", "WR") == 16.7
    assert lookup("other", "WR") == 11.0
    assert lookup("ghost", "WR") == 0.0
