"""Iteration 2, items 1+2 at the Trade Hub surface: the archetype
suggestion engine must run every package through the shared partner
gate with keeper legality.

- A designated keeper (JSN in the live league) can never appear on
  the give side of any suggestion, in any archetype.
- Keeper format with UNKNOWN designations fails closed: no packages
  are pitched at all.
- Caleb Williams packages to a Goodfellas-shaped partner (Maye
  keeper + Ward/Mendoza unprojected) must return a startable QB:
  the live pitches (Kyren alone, Watson alone) orphan the partner's
  QB2 and must not surface as pitched.
- Surfaced rows carry the checked-evaluation stamps; acceptance is
  restamped from the checked evaluation, not the pre-check estimate.

Harness mirrors tests/test_archetype_suggestions_e2e.py (offline,
hand-built ctx, result cache cleared per test).
"""
import pytest

pytest.importorskip("pandas")

from dashboard_services import archetype_engine as ae


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    import dashboard_services.api as api

    def _fake_fetch_json(path, timeout=25, retries=3):
        if path == "/state/nfl":
            return {"season": "2026", "week": 0, "leg": 0,
                    "season_type": "off", "display_week": 1,
                    "season_start_date": "2026-09-10"}
        return {}

    monkeypatch.setattr(api, "fetch_json", _fake_fetch_json)


@pytest.fixture(autouse=True)
def _clear_result_cache():
    ae._RESULT_CACHE.clear()
    yield
    ae._RESULT_CACHE.clear()


def _p(pid, name, pos, val, age, team="FA"):
    return {"id": pid, "name": name, "position": pos, "team": team,
            "age": age, "value": val, "sf_value": val,
            "redraft_value_1qb": val * 0.8, "redraft_value_sf": val * 0.8,
            "pos_rank_label": f"{pos}1", "rank_change_7d": 0}


def _seed_ctx(keepers=None, max_keepers=None):
    table = [
        _p("r2_rb", "Stud RB", "RB", 1000, 23, "DET"),
        _p("r3_wr", "Stud WR", "WR", 950, 24, "CIN"),
        _p("r4_qb", "Stud QB", "QB", 700, 25, "BUF"),
        _p("r2_wr", "Rival WR2", "WR", 300, 27, "DET"),
        _p("r3_rb", "Rival RB2", "RB", 280, 28, "CIN"),
        _p("r4_te", "Rival TE2", "TE", 260, 26, "BUF"),
        _p("r2_young", "Young WR", "WR", 450, 22, "DET"),
        _p("r3_young", "Young RB", "RB", 420, 21, "CIN"),
        _p("v_wr1", "Viewer WR1", "WR", 560, 26, "PHI"),
        _p("v_rb1", "Viewer RB1", "RB", 480, 27, "PHI"),
        _p("v_wr2", "Viewer WR2", "WR", 300, 29, "PHI"),
        _p("v_rb2", "Viewer RB2", "RB", 220, 30, "PHI"),
        _p("v_te1", "Viewer TE1", "TE", 240, 28, "PHI"),
        _p("v_stud", "Viewer Stud", "WR", 900, 29, "PHI"),
        _p("v_qb1", "Viewer QB1", "QB", 260, 31, "PHI"),
    ]
    rosters = [
        {"roster_id": 1, "players": ["v_wr1", "v_rb1", "v_wr2", "v_rb2",
                                     "v_te1", "v_stud", "v_qb1"]},
        {"roster_id": 2, "players": ["r2_rb", "r2_wr", "r2_young"]},
        {"roster_id": 3, "players": ["r3_wr", "r3_rb", "r3_young"]},
        {"roster_id": 4, "players": ["r4_qb", "r4_te"]},
    ]
    if keepers:
        for roster in rosters:
            roster["keepers"] = list(keepers.get(roster["roster_id"], []))
    ctx = {
        "rosters": rosters,
        "roster_map": {1: "Viewer", 2: "Team Two", 3: "Team Three", 4: "Team Four"},
        "standings_map": {1: 6, 2: 1, 3: 2, 4: 9},
        "model_value_table": table,
        "picks_by_roster": {},
        "settings": {"playoff_week_start": 15},
    }
    if max_keepers:
        ctx["league_settings"] = {"max_keepers": max_keepers}
    return ctx


def _run(ctx, archetype, league_type="1qb"):
    return ae.get_archetype_suggestions(
        archetype=archetype, platform="sleeper", league_id="testlg",
        season=2026, viewer_roster_id="1", league_type=league_type,
        league_size=10, ctx=ctx,
    )


def _send_ids(sugg):
    return [str(a.get("player_id") or a.get("id") or "")
            for a in (sugg.get("suggested_send") or [])
            if isinstance(a, dict) and not a.get("is_pick")]


@pytest.mark.parametrize("archetype",
                         ["consolidate", "contending", "rebuilding", "distribute"])
def test_designated_keeper_never_sent(archetype):
    ctx = _seed_ctx(keepers={1: ["v_stud"]}, max_keepers=5)
    out = _run(ctx, archetype)
    for s in out.get("suggestions") or []:
        assert "v_stud" not in _send_ids(s), (
            f"{archetype} pitched designated keeper v_stud: {s}")


@pytest.mark.parametrize("archetype", ["contending", "distribute"])
def test_keeper_format_unknown_designations_fails_closed(archetype):
    # Keeper limit known, but no roster carries designations and no
    # rules record exists: legality is unverifiable, so no package
    # may be pitched in either direction.
    ctx = _seed_ctx(max_keepers=5)
    out = _run(ctx, archetype)
    assert out.get("suggestions") == []


def test_rows_carry_checked_evaluation_stamps():
    out = _run(_seed_ctx(), "contending")
    assert out.get("suggestions"), "seed ctx should still surface rows"
    for s in out["suggestions"]:
        # Every surfaced row passed the gate (partner_checked) and its
        # acceptance carries the basis of the model that produced it
        # (the shared conditioned acceptance fed by the checked
        # evaluation, or the documented pre-check fallback).
        assert s.get("partner_checked") is True
        assert s.get("acceptance_basis")


def _caleb_ctx():
    table = [
        # Goodfellas (roster 2): Maye (keeper) + Caleb startable;
        # Ward and Mendoza carry value but no projection.
        _p("maye", "Drake Maye", "QB", 867, 24, "NE"),
        _p("caleb", "Caleb Williams", "QB", 805, 25, "CHI"),
        _p("ward", "Cam Ward", "QB", 502, 24, "TEN"),
        _p("mendoza", "Fernando Mendoza", "QB", 419, 23, "LV"),
        _p("g_rb1", "G RB1", "RB", 600, 26, "NE"),
        _p("g_rb2", "G RB2", "RB", 450, 27, "NE"),
        _p("g_wr1", "G WR1", "WR", 650, 26, "NE"),
        _p("g_wr2", "G WR2", "WR", 500, 28, "NE"),
        _p("g_te1", "G TE1", "TE", 320, 27, "NE"),
        # Viewer (roster 1)
        _p("lock", "Drew Lock", "QB", 720, 29, "SEA"),
        _p("mccarthy", "J.J. McCarthy", "QB", 613, 23, "MIN"),
        _p("cousins", "Kirk Cousins", "QB", 640, 38, "ATL"),
        _p("kyren", "Kyren Williams", "RB", 780, 26, "LAR"),
        _p("v_rb2", "Viewer RB2", "RB", 500, 27, "PHI"),
        _p("watson", "Christian Watson", "WR", 615, 27, "GB"),
        _p("jsn", "Jaxon Smith-Njigba", "WR", 900, 24, "SEA"),
        _p("v_wr3", "Viewer WR3", "WR", 500, 26, "PHI"),
        _p("kincaid", "Dalton Kincaid", "TE", 350, 27, "BUF"),
        # Filler rosters
        _p("x_rb", "X RB", "RB", 700, 25, "DET"),
        _p("x_wr", "X WR", "WR", 700, 25, "DET"),
        _p("y_qb", "Y QB", "QB", 700, 26, "BUF"),
        _p("y_te", "Y TE", "TE", 300, 26, "BUF"),
    ]
    rosters = [
        {"roster_id": 1, "players": ["lock", "mccarthy", "cousins", "kyren",
                                     "v_rb2", "watson", "jsn", "v_wr3",
                                     "kincaid"],
         "keepers": ["jsn"]},
        {"roster_id": 2, "players": ["maye", "caleb", "ward", "mendoza",
                                     "g_rb1", "g_rb2", "g_wr1", "g_wr2",
                                     "g_te1"],
         "keepers": ["maye"]},
        {"roster_id": 3, "players": ["x_rb", "x_wr"], "keepers": []},
        {"roster_id": 4, "players": ["y_qb", "y_te"], "keepers": []},
    ]
    return {
        "rosters": rosters,
        "roster_map": {1: "Viewer", 2: "Goodfellas", 3: "Team Three",
                       4: "Team Four"},
        "standings_map": {1: 6, 2: 1, 3: 2, 4: 9},
        "model_value_table": table,
        "picks_by_roster": {},
        "settings": {"playoff_week_start": 15},
        "league_settings": {"max_keepers": 5},
        # Flat projections layer: Ward and Mendoza are absent, so the
        # orphan guard keeps them non-startable.
        "projections": {"maye": 23.0, "caleb": 20.6, "lock": 19.9,
                        "mccarthy": 17.9, "cousins": 18.5},
    }


@pytest.mark.parametrize("archetype", ["contending", "consolidate"])
def test_caleb_packages_must_return_a_startable_qb(archetype):
    out = _run(_caleb_ctx(), archetype, league_type="superflex")
    viewer_qbs = {"lock", "mccarthy", "cousins"}
    for s in out.get("suggestions") or []:
        assert "jsn" not in _send_ids(s)  # designated keeper, again
        if str(s.get("player_id") or "") != "caleb":
            continue
        send = set(_send_ids(s))
        assert send & viewer_qbs, (
            f"Caleb pitched without a QB returning: {sorted(send)}")
        assert s.get("partner_checked") is True
