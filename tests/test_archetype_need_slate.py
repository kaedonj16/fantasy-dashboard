"""Contending suggestions follow the win-prob rise, with a mild hole nudge.

A last-place QB/TE should compete fairly (1QB scarcity must not bury a real
upgrade), but a WR-factory can still get its best rise from another stud
WR/RB. Hard-reserved hole slots and a first-place penalty were the wrong
fix — they jumped a low-rise TE over a high-rise flex add.

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


# ── Rank / multiplier / rise-first scoring ────────────────────────────────────

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


def test_need_multiplier_boosts_holes_only():
    """Top-half rooms stay 1.0 — a strength is never taxed. Last place gets
    the small tiebreaker, not a 1.45 hammer."""
    assert ae._need_multiplier(1, 12) == 1.0
    assert ae._need_multiplier(6, 12) == 1.0
    assert ae._need_multiplier(12, 12) == pytest.approx(ae._NEED_MULT_WORST)
    assert 1.0 < ae._need_multiplier(10, 12) < ae._NEED_MULT_WORST
    assert ae._need_multiplier(None, 12) == 1.0
    assert ae._need_multiplier(1, 1) == 1.0


def test_contending_ranks_by_rise_not_by_hole():
    """A +6% weekly add at a stacked WR/RB room outranks a +0.5% last-place
    TE fill. A real hole-fill (+7%) still beats a flat add at a strength."""
    strength = ae._acquire_rank_score(0.70, 0.06, 1.00, 1.00, "contending")
    token_hole = ae._acquire_rank_score(0.70, 0.005, 1.00, ae._NEED_MULT_WORST, "contending")
    assert strength > token_hole

    real_hole = ae._acquire_rank_score(0.55, 0.07, 0.75, ae._NEED_MULT_WORST, "contending")
    flat_strength = ae._acquire_rank_score(0.80, 0.01, 1.25, 1.00, "contending")
    assert real_hole > flat_strength

    # Same rise: the last-place nudge breaks the tie toward the hole.
    hole_tie = ae._acquire_rank_score(0.60, 0.04, 1.00, ae._NEED_MULT_WORST, "contending")
    strength_tie = ae._acquire_rank_score(0.60, 0.04, 1.00, 1.00, "contending")
    assert hole_tie > strength_tie


def _tgt(pid, pos, owner, score):
    return (score, {
        "player_id": pid, "position": pos, "owner_roster_id": owner, "name": pid,
    })


def test_select_varied_slate_follows_score_not_holes():
    """The slate is score order + variety caps. A low-score TE/QB does not
    jump a high-score RB/WR just because the room is a hole."""
    scored = [
        _tgt("rb1", "RB", "2", 9), _tgt("rb2", "RB", "3", 8.5), _tgt("rb3", "RB", "4", 8),
        _tgt("wr1", "WR", "2", 8.8), _tgt("wr2", "WR", "3", 8.4), _tgt("wr3", "WR", "4", 8.1),
        _tgt("qb1", "QB", "2", 4.0),
        _tgt("te1", "TE", "3", 3.5),
    ]
    top = ae._select_varied_slate(scored, max_targets=6)
    assert {t["position"] for t in top} == {"RB", "WR"}
    assert [t["player_id"] for t in top][:2] == ["rb1", "wr1"]

    # With room for 8, the next-best (QB, then TE) still land — on merit.
    full = ae._select_varied_slate(scored, max_targets=8)
    assert [t["player_id"] for t in full][-2:] == ["qb1", "te1"]


# ── Full Contending path on the reported roster shape ─────────────────────────

def _ctx_wr_factory_last_at_qb_te():
    """Viewer is a WR/RB factory last at QB and TE. Rivals hold startable
    QB/TE upgrades (real dedicated-slot rises) plus RB/WR depth that can
    also raise a flex lineup."""
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


def test_contending_keeps_strength_rises_and_real_hole_fills():
    """A WR-factory last at QB/TE still sees high-rise RB/WR adds, and the
    dedicated-slot QB/TE upgrades (scrub → startable) compete on impact
    rather than being reserved or buried."""
    out = ae._get_archetype_suggestions_impl(
        archetype="contending", platform="sleeper", league_id="holes",
        season=2026, viewer_roster_id="1", league_type="1qb", league_size=6,
        ctx=_ctx_wr_factory_last_at_qb_te(),
    )
    sugg = out["suggestions"]
    assert sugg, "expected contending suggestions"
    positions = {s["position"] for s in sugg}
    names = {s["name"] for s in sugg}
    # Heavy rooms can still produce the best rise — they must stay on the slate.
    assert positions & {"RB", "WR"}, names
    # Replacing a 12-value QB / 40-value TE is a real dedicated-slot rise, so
    # those upgrades should compete their way on (not be scarcity-buried).
    assert "QB" in positions, names
    assert "TE" in positions, names
    for s in sugg:
        if s["position"] in ("QB", "TE"):
            assert s.get("suggested_send"), f"{s['name']} surfaced with an empty send"
