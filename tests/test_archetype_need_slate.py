"""Viewer positional holes must show up on the Contending slate.

A WR-factory last at QB and TE used to see a Strategy tab full of flex RBs/WRs.
1QB scarcity down-weights QB/TE, availability prefers a rival's surplus RB3,
and the old slate cap (3 per position, no reservation) let those two effects
crowd the holes out. These tests pin the Teams-page rank helpers, the reserved
slate, and the full Contending path on that roster shape.

Pure helpers + the impl (network stubbed) so this runs without flask/pandas.
"""
import sys
import types

import pytest

from dashboard_services import archetype_engine as ae


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    fake = types.ModuleType("dashboard_services.api")
    fake.fetch_json = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake)
    ae._RESULT_CACHE.clear()
    yield
    ae._RESULT_CACHE.clear()


def _P(pid, name, pos, val, age=26, team="FA"):
    return {"id": pid, "name": name, "position": pos, "team": team, "age": age,
            "value": val, "sf_value": val,
            "redraft_value_1qb": val * 0.8, "redraft_value_sf": val * 0.8,
            "pos_rank_label": f"{pos}1", "rank_change_7d": 0}


def _info(pid, name, pos, val):
    return {"name": name, "position": pos, "value": val}


# ── Rank / multiplier helpers ─────────────────────────────────────────────────

def test_viewer_pos_ranks_match_teams_style_holes():
    """Last-place QB/TE on a WR-factory read as last, not 'average'."""
    values = {
        "v_wr": _info("v_wr", "WR", "WR", 900),
        "v_rb": _info("v_rb", "RB", "RB", 700),
        "v_te": _info("v_te", "TE", "TE", 40),
        "v_qb": _info("v_qb", "QB", "QB", 12),
        "r_wr": _info("r_wr", "WR", "WR", 300),
        "r_rb": _info("r_rb", "RB", "RB", 300),
        "r_te": _info("r_te", "TE", "TE", 400),
        "r_qb": _info("r_qb", "QB", "QB", 420),
        "s_wr": _info("s_wr", "WR", "WR", 280),
        "s_rb": _info("s_rb", "RB", "RB", 280),
        "s_te": _info("s_te", "TE", "TE", 360),
        "s_qb": _info("s_qb", "QB", "QB", 380),
    }
    rosters = [
        {"roster_id": 1, "players": ["v_wr", "v_rb", "v_te", "v_qb"]},
        {"roster_id": 2, "players": ["r_wr", "r_rb", "r_te", "r_qb"]},
        {"roster_id": 3, "players": ["s_wr", "s_rb", "s_te", "s_qb"]},
    ]
    ranks = ae._viewer_pos_league_ranks(rosters, values, "1", [], "1qb")
    assert ranks["QB"] == 3
    assert ranks["TE"] == 3
    assert ranks["WR"] == 1
    assert ranks["RB"] == 1


def test_need_multiplier_boosts_last_place_and_dampens_first():
    assert ae._need_multiplier(1, 12) == pytest.approx(ae._NEED_MULT_BEST)
    assert ae._need_multiplier(12, 12) == pytest.approx(ae._NEED_MULT_WORST)
    # Mid-pack is near neutral.
    assert 0.95 < ae._need_multiplier(6, 12) < 1.20
    assert ae._need_multiplier(None, 12) == 1.0
    assert ae._need_multiplier(1, 1) == 1.0


def test_needed_positions_are_bottom_35_percent_worst_first():
    # 12-team: cutoff 4 → ranks 9-12 are holes.
    ranks = {"QB": 12, "TE": 11, "RB": 3, "WR": 2}
    assert ae._needed_positions(ranks, 12) == ["QB", "TE"]
    # Balanced top-half roster: no reservation.
    assert ae._needed_positions({"QB": 2, "RB": 3, "WR": 4, "TE": 5}, 12) == []
    # 4-team field: only last place (rank 4) is in the cutoff of 1.
    assert ae._needed_positions({"QB": 4, "TE": 4, "RB": 1, "WR": 1}, 4) == ["QB", "TE"]


def _tgt(pid, pos, owner, score):
    return (score, {
        "player_id": pid, "position": pos, "owner_roster_id": owner, "name": pid,
    })


def test_select_varied_slate_reserves_worst_positions():
    """Without a reservation, 3 RB + 3 WR fill 6 of 8 slots and TE never lands.
    With QB/TE marked as needed, both appear from the best-scored player there."""
    scored = [
        _tgt("rb1", "RB", "2", 9), _tgt("rb2", "RB", "3", 8.5), _tgt("rb3", "RB", "4", 8),
        _tgt("wr1", "WR", "2", 8.8), _tgt("wr2", "WR", "3", 8.4), _tgt("wr3", "WR", "4", 8.1),
        _tgt("qb1", "QB", "2", 4.0),
        _tgt("te1", "TE", "3", 3.5),
        _tgt("qb2", "QB", "3", 3.0),
        _tgt("te2", "TE", "4", 2.8),
    ]
    # 6 slots = the old "3 RB + 3 WR fill the board" shape the user saw.
    plain = ae._select_varied_slate(scored, needed_positions=[], max_targets=6)
    assert {t["position"] for t in plain} == {"RB", "WR"}
    assert [t["position"] for t in plain].count("RB") == 3
    assert [t["position"] for t in plain].count("WR") == 3

    reserved = ae._select_varied_slate(
        scored, needed_positions=["QB", "TE"], max_targets=6)
    pos = {t["position"] for t in reserved}
    assert "QB" in pos
    assert "TE" in pos
    # Best-scored hole-fillers, not the leftovers.
    assert "qb1" in {t["player_id"] for t in reserved}
    assert "te1" in {t["player_id"] for t in reserved}


# ── Full Contending path on the reported roster shape ─────────────────────────

def _ctx_wr_factory_last_at_qb_te():
    """Viewer is a WR/RB factory last at QB and TE. Rivals hold startable
    QB/TE upgrades plus a pile of RB/WR depth that used to flood the slate."""
    table = [
        _P("v_wr1", "Viewer WR1", "WR", 900, 25),
        _P("v_wr2", "Viewer WR2", "WR", 820, 26),
        _P("v_wr3", "Viewer WR3", "WR", 700, 24),
        _P("v_wr4", "Viewer WR4", "WR", 420, 27),
        _P("v_rb1", "Viewer RB1", "RB", 750, 25),
        _P("v_rb2", "Viewer RB2", "RB", 640, 26),
        _P("v_rb3", "Viewer RB3", "RB", 380, 28),
        _P("v_te", "Scrub TE", "TE", 40, 30),
        _P("v_qb1", "Scrub QB1", "QB", 12, 33),
        _P("v_qb2", "Scrub QB2", "QB", 8, 31),
        _P("qb_upgrade", "Upgrade QB", "QB", 420, 27, "SF"),
        _P("r2_rb1", "Rival RB1", "RB", 680, 25, "DET"),
        _P("r2_rb2", "Rival RB2", "RB", 520, 26, "DET"),
        _P("r2_wr1", "Rival WR1", "WR", 600, 25, "DET"),
        _P("te_upgrade", "Upgrade TE", "TE", 390, 26, "CIN"),
        _P("r3_rb1", "Rival3 RB1", "RB", 660, 24, "CIN"),
        _P("r3_wr1", "Rival3 WR1", "WR", 640, 25, "CIN"),
        _P("r3_wr2", "Rival3 WR2", "WR", 500, 27, "CIN"),
        _P("r4_rb1", "Rival4 RB1", "RB", 700, 24, "BUF"),
        _P("r4_rb2", "Rival4 RB2", "RB", 540, 26, "BUF"),
        _P("r4_wr1", "Rival4 WR1", "WR", 620, 25, "BUF"),
        _P("r4_wr2", "Rival4 WR2", "WR", 480, 28, "BUF"),
        _P("r4_qb", "Rival4 QB", "QB", 200, 29, "BUF"),
        _P("r5_qb", "T5 QB", "QB", 350, 27),
        _P("r5_te", "T5 TE", "TE", 280, 26),
        _P("r5_wr", "T5 WR", "WR", 300, 27),
        _P("r6_qb", "T6 QB", "QB", 340, 28),
        _P("r6_te", "T6 TE", "TE", 260, 27),
        _P("r6_rb", "T6 RB", "RB", 300, 27),
    ]
    return {
        "rosters": [
            {"roster_id": 1, "players": [
                "v_wr1", "v_wr2", "v_wr3", "v_wr4",
                "v_rb1", "v_rb2", "v_rb3", "v_te", "v_qb1", "v_qb2",
            ]},
            {"roster_id": 2, "players": ["qb_upgrade", "r2_rb1", "r2_rb2", "r2_wr1"]},
            {"roster_id": 3, "players": ["te_upgrade", "r3_rb1", "r3_wr1", "r3_wr2"]},
            {"roster_id": 4, "players": ["r4_rb1", "r4_rb2", "r4_wr1", "r4_wr2", "r4_qb"]},
            {"roster_id": 5, "players": ["r5_qb", "r5_te", "r5_wr"]},
            {"roster_id": 6, "players": ["r6_qb", "r6_te", "r6_rb"]},
        ],
        "roster_map": {1: "Viewer", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"},
        "standings_map": {1: 3, 2: 1, 3: 2, 4: 4, 5: 5, 6: 6},
        "model_value_table": table,
        "picks_by_roster": {},
        "settings": {"playoff_week_start": 15},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX"],
    }


def test_contending_surfaces_last_place_qb_and_te():
    """The reported Teams-page case: #last QB and #last TE must appear as
    Contending acquire targets, not get buried under flex RB/WR depth."""
    out = ae._get_archetype_suggestions_impl(
        archetype="contending", platform="sleeper", league_id="holes",
        season=2026, viewer_roster_id="1", league_type="1qb", league_size=6,
        ctx=_ctx_wr_factory_last_at_qb_te(),
    )
    sugg = out["suggestions"]
    assert sugg, "expected contending suggestions"
    positions = {s["position"] for s in sugg}
    names = {s["name"] for s in sugg}
    assert "QB" in positions, names
    assert "TE" in positions, names
    # The startable upgrades, not a leftover backup, should be the ones reserved.
    assert "Upgrade QB" in names or any(s["position"] == "QB" for s in sugg)
    assert "Upgrade TE" in names or any(s["position"] == "TE" for s in sugg)
    for s in sugg:
        if s["position"] in ("QB", "TE"):
            assert s.get("suggested_send"), f"{s['name']} surfaced with an empty send"
