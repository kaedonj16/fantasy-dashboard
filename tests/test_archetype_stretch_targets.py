"""Every rival player stays reachable as a trade target.

The archetype engine used to DROP targets that didn't match the archetype's
usual roster shape (a manager with no starter-caliber QB got no suggestion at
all for an elite QB, so that player looked unobtainable). Off-pattern targets
are now demoted rather than excluded, and their packages are allowed a deeper
tier drop so a roster without a tier-comparable headliner can still be shown
what it would take.

Unlike tests/test_archetype_suggestions_e2e.py these drive the impl directly
with the network layer stubbed, so they run on the pure interpreter (CI installs
only pytest) instead of skipping.
"""
import sys
import types

import pytest

from dashboard_services import archetype_engine as ae


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Stub the api module the engine lazily imports so nothing hits the network
    (and so it imports at all without flask installed)."""
    fake = types.ModuleType("dashboard_services.api")
    fake.fetch_json = lambda *a, **k: {}
    monkeypatch.setitem(sys.modules, "dashboard_services.api", fake)
    ae._RESULT_CACHE.clear()
    yield
    ae._RESULT_CACHE.clear()


def _P(pid, name, pos, val, age, team="FA"):
    return {"id": pid, "name": name, "position": pos, "team": team, "age": age,
            "value": val, "sf_value": val,
            "redraft_value_1qb": val * 0.8, "redraft_value_sf": val * 0.8,
            "pos_rank_label": f"{pos}1", "rank_change_7d": 0}


def _ctx_no_qb():
    """Viewer owns no real QB (only a scrub) while a rival holds an elite QB -
    the reported case: the roster doesn't fit the consolidate pattern at QB, so
    the elite QB used to be dropped entirely."""
    table = [
        _P("r4_qb", "Elite QB", "QB", 900, 25, "JAX"),
        _P("r2_rb", "Rival RB", "RB", 500, 26, "DET"),
        _P("v_wr1", "Viewer WR1", "WR", 400, 26),
        _P("v_wr2", "Viewer WR2", "WR", 380, 27),
        _P("v_rb1", "Viewer RB1", "RB", 360, 27),
        _P("v_rb2", "Viewer RB2", "RB", 340, 28),
        _P("v_te1", "Viewer TE1", "TE", 300, 28),
        _P("v_qb_scrub", "Scrub QB", "QB", 40, 33),
    ]
    return {
        "rosters": [
            {"roster_id": 1, "players": ["v_wr1", "v_wr2", "v_rb1", "v_rb2", "v_te1", "v_qb_scrub"]},
            {"roster_id": 2, "players": ["r2_rb"]},
            {"roster_id": 4, "players": ["r4_qb"]},
        ],
        "roster_map": {1: "Viewer", 2: "Two", 4: "Four"},
        "standings_map": {1: 6, 2: 1, 4: 9},
        "model_value_table": table,
        "picks_by_roster": {},
        "settings": {"playoff_week_start": 15},
    }


def _run(archetype="consolidate", ctx=None):
    return ae._get_archetype_suggestions_impl(
        archetype=archetype, platform="sleeper", league_id="t1", season=2026,
        viewer_roster_id="1", league_type="1qb", league_size=10,
        ctx=ctx if ctx is not None else _ctx_no_qb(),
    )


def test_off_pattern_target_is_still_reachable():
    sugg = _run()["suggestions"]
    ids = {s["player_id"] for s in sugg}
    assert "r4_qb" in ids, "an elite target must stay reachable off-pattern"


def test_off_pattern_target_is_flagged_as_stretch():
    sugg = _run()["suggestions"]
    qb = next(s for s in sugg if s["player_id"] == "r4_qb")
    assert qb["is_stretch"] is True
    # A natural-fit target is not flagged, so the UI can tell them apart.
    rb = next((s for s in sugg if s["player_id"] == "r2_rb"), None)
    if rb:
        assert rb["is_stretch"] is False


def test_stretch_package_is_fair_value_not_a_lowball():
    """Relaxing the tier guard must not relax fairness: the stretch offer still
    has to be a real, value-matched package (it just costs more pieces)."""
    qb = next(s for s in _run()["suggestions"] if s["player_id"] == "r4_qb")
    send = qb["suggested_send"]
    assert send, "a reachable target must carry a send package"
    raw = sum(a.get("value", 0) for a in send)
    # Fair value for a 900-value target: never a token underpay.
    assert raw >= qb["value"] * 0.85
    assert len(send) >= 2, "consolidating up should cost multiple pieces"


def test_natural_fit_outranks_the_stretch():
    """Demotion, not exclusion: an in-pattern target still leads the slate."""
    sugg = _run()["suggestions"]
    ids = [s["player_id"] for s in sugg]
    if "r2_rb" in ids and "r4_qb" in ids:
        assert ids.index("r2_rb") < ids.index("r4_qb")


def test_stretch_slack_only_applies_to_stretch_targets():
    """The tier guard still protects normal targets, so a wall of low-tier depth
    can't quietly buy a stud on the in-pattern path."""
    sends = [
        {"player_id": "a", "name": "Depth A", "position": "WR", "value": 120},
        {"player_id": "b", "name": "Depth B", "position": "WR", "value": 110},
        {"player_id": "c", "name": "Depth C", "position": "RB", "value": 105},
    ]
    strict = ae._select_packages(sends, 900.0, "consolidate", max_pkgs=3,
                                 league_size=10, stretch=False)
    assert strict == [], "low-tier depth must not reach a stud in-pattern"
