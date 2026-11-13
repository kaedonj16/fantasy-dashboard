"""Trade calc logic v3 (2026-11-12): calibrated pick pricing on every
trade surface, the partner lineup-drop acceptance cap in the
calculator, PPG-weighted roster fit from the sim, and QB POS_WEIGHTS
hygiene.

Pins:
- Suggestion surfaces never price a pick from a flat per-round
  constant: a pick the calibrated table cannot price is excluded.
- _partner_lineup_drop_cap generalises the suggestions cap to N
  surrendered players; the original single-target
  _partner_acceptance_cap remains as an equivalent wrapper.
- _fit_roster_ppg_score maps the sim's optimal-lineup PPG delta onto
  the same 5-centred 0-10 shape as _fit_playoff_score.
"""
from __future__ import annotations

import ast
import os

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

_HERE = os.path.dirname(__file__)
_ROOT = os.path.join(_HERE, "..")


@pytest.fixture(scope="module")
def appmod():
    import app as app_module

    return app_module


@pytest.fixture(scope="module")
def arch():
    import dashboard_services.archetype_engine as archetype_engine

    return archetype_engine


# ── PPG-weighted roster fit ──────────────────────────────────────────

def test_roster_ppg_score_anchors(appmod):
    f = appmod._fit_roster_ppg_score
    assert f(None) is None
    assert f({}) is None
    assert f({"ppg_before": 0, "ppg_after": 10}) is None  # no baseline
    assert f({"ppg_before": 150.0, "ppg_after": 150.0}) == 5   # flat
    assert f({"ppg_before": 150.0, "ppg_after": 153.0}) == 8   # +3 PPG
    assert f({"ppg_before": 150.0, "ppg_after": 155.0}) == 10  # saturates
    assert f({"ppg_before": 150.0, "ppg_after": 145.0}) == 0   # -5 PPG
    assert f({"ppg_before": 150.0, "ppg_after": 148.4}) == 3   # -1.6 PPG
    assert f({"ppg_before": "junk", "ppg_after": 10}) is None


# ── Partner lineup-drop cap ──────────────────────────────────────────

def _values_by_id():
    rows = {
        "qb1": ("QB", 400), "s1": ("RB", 500), "rb2": ("RB", 320),
        "rb3": ("RB", 100), "s2": ("WR", 500), "wr2": ("WR", 330),
        "wr3": ("WR", 90), "te1": ("TE", 200),
        "bn1": ("RB", 80), "bn2": ("WR", 70),
        "in1": ("RB", 240), "in2": ("WR", 230),
    }
    return {
        pid: {"position": pos, "value": val}
        for pid, (pos, val) in rows.items()
    }


_PARTNER_PIDS = ["qb1", "s1", "rb2", "rb3", "s2", "wr2", "wr3", "te1",
                 "bn1", "bn2"]
_PKG = [
    {"player_id": "in1", "value": 240, "position": "RB"},
    {"player_id": "in2", "value": 230, "position": "WR"},
]


def test_lineup_drop_cap_multi_surrender(arch):
    # Partner surrenders two 500 starters for two bench pieces: the
    # optimal lineup drops ~22%, a hard cap, even though a per-target
    # read (removing one starter at a time) understates the hit.
    cap = arch._partner_lineup_drop_cap(
        _PKG, ["s1", "s2"], 500.0, _PARTNER_PIDS, _values_by_id(), "1qb")
    assert cap == 20


def test_lineup_drop_cap_wrapper_matches_single_target(arch):
    single_pkg = [{"player_id": "in1", "value": 240, "position": "RB"}]
    via_wrapper = arch._partner_acceptance_cap(
        single_pkg, "s1", 500.0, _PARTNER_PIDS, _values_by_id(), "1qb")
    via_general = arch._partner_lineup_drop_cap(
        single_pkg, ["s1"], 500.0, _PARTNER_PIDS, _values_by_id(), "1qb")
    assert via_wrapper == via_general
    assert via_wrapper == 20


def test_lineup_drop_cap_exemptions(arch):
    vbi = _values_by_id()
    # Rebuilding partner: selling starters is the point.
    assert arch._partner_lineup_drop_cap(
        _PKG, ["s1", "s2"], 500.0, _PARTNER_PIDS, vbi, "1qb",
        partner_arch="rebuilding") is None
    # A 1st within two drafts exempts the package.
    pick_pkg = _PKG + [{
        "player_id": "pick_2027_1_mid", "value": 300, "position": "PICK",
        "is_pick": True, "pick_season": 2027, "pick_round": 1,
    }]
    assert arch._partner_lineup_drop_cap(
        pick_pkg, ["s1", "s2"], 500.0, _PARTNER_PIDS, vbi, "1qb",
        season=2026) is None
    # A genuine overpay (>= 1.15x) exempts; 1.10x softens one tier.
    assert arch._partner_lineup_drop_cap(
        _PKG, ["s1", "s2"], 500.0, _PARTNER_PIDS, vbi, "1qb",
        eff_ratio=1.15) is None
    assert arch._partner_lineup_drop_cap(
        _PKG, ["s1", "s2"], 500.0, _PARTNER_PIDS, vbi, "1qb",
        eff_ratio=1.10) == 35
    # A near-equal headliner coming back exempts.
    stud_pkg = [{"player_id": "in1", "value": 450, "position": "RB"}]
    vbi2 = _values_by_id()
    vbi2["in1"] = {"position": "RB", "value": 450}
    assert arch._partner_lineup_drop_cap(
        stud_pkg, ["s1", "s2"], 500.0, _PARTNER_PIDS, vbi2, "1qb") is None


def test_lineup_drop_cap_no_drop_no_cap(arch):
    # Partner surrenders a bench piece only: lineup unchanged.
    cap = arch._partner_lineup_drop_cap(
        _PKG, ["bn1"], 80.0, _PARTNER_PIDS, _values_by_id(), "1qb")
    assert cap is None


# ── Pick send candidates: calibrated only ────────────────────────────

def test_pick_send_candidates_unpriced_pick_excluded(arch, monkeypatch):
    import dashboard_services.picks as picks_mod

    monkeypatch.setattr(
        picks_mod, "load_pick_value_table", lambda **kw: {})
    out = arch._pick_send_candidates(
        [{"season": "2027", "round": 1}], num_teams=12)
    assert out == []  # never the old flat 650.0


def test_pick_send_candidates_use_calibrated_table(arch, monkeypatch):
    import dashboard_services.picks as picks_mod

    monkeypatch.setattr(
        picks_mod, "load_pick_value_table",
        lambda **kw: {"2027_1_mid": 310.0, "2027_2_mid": 120.0})
    out = arch._pick_send_candidates(
        [{"season": "2027", "round": 1}, {"season": "2027", "round": 2}],
        num_teams=12)
    vals = {c["pick_round"]: c["value"] for c in out}
    assert vals == {1: 310.0, 2: 120.0}


# ── Flat pick constants are gone from trade paths (source pin) ──────

def test_no_flat_pick_fallbacks_in_trade_paths():
    with open(os.path.join(_ROOT, "app.py")) as fh:
        app_src = fh.read()
    with open(os.path.join(
            _ROOT, "dashboard_services", "archetype_engine.py")) as fh:
        arch_src = fh.read()
    for needle in (
        "650.0 if rnd == 1", "{1: 650.0", "650.0, 2: 220.0",
        "{1: 300.0, 2: 150.0", "300.0 if rnd == 1",
        "450.0 if rnd == 1", "{1: 220.0, 2: 130.0}",
    ):
        assert needle not in app_src, needle
        assert needle not in arch_src, needle


# ── QB POS_WEIGHTS sum to 1.00 like every other position ────────────

def test_pos_weights_sum_to_one():
    path = os.path.join(_ROOT, "data_building", "player_value.py")
    with open(path) as fh:
        tree = ast.parse(fh.read())
    weights = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "POS_WEIGHTS"
                for t in node.targets):
            weights = ast.literal_eval(node.value)
            break
    assert weights is not None
    for pos, w in weights.items():
        assert abs(sum(w.values()) - 1.0) < 1e-6, pos
