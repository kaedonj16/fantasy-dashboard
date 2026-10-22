"""Tests for dashboard_services.landing_demos (guest landing demo data).

The module is stdlib-only; the DB values loader is stubbed out so the real
data files (data/model_values.json, cache/matchup_ratings_s*.json) drive the
invariants deterministically.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

landing_demos = pytest.importorskip("dashboard_services.landing_demos")

RANK_LABEL = re.compile(r"(QB|RB|WR|TE)\d")


@pytest.fixture(autouse=True)
def _force_file_values(monkeypatch):
    """Skip the DB values loader; exercise the model_values.json fallback."""
    monkeypatch.setattr(landing_demos, "_db_value_rows", lambda: [])


@pytest.fixture(autouse=True)
def _no_network_week(monkeypatch):
    """Keep week derivation off the network unless a test opts in."""
    monkeypatch.setattr(landing_demos, "_nfl_state_week", lambda: None)
    monkeypatch.setattr(landing_demos, "_demo_season_week", lambda: (None, None))
    monkeypatch.setattr(
        landing_demos, "_live_projection_rows",
        lambda season, week: {pos: [] for pos in ("QB", "RB", "WR", "TE")},
    )


def _ratings_doc():
    path = landing_demos.MATCHUP_RATINGS_PATH
    return json.loads(path.read_text(encoding="utf-8"))


def _ordinal(n):
    if 11 <= n % 100 <= 13:
        suf = "th"
    else:
        suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def _fpts_rank(ratings, opp, pos):
    rows = sorted(
        ((t, float(by[pos]["fpts"])) for t, by in ratings.items()),
        key=lambda kv: -kv[1],
    )
    for i, (t, f) in enumerate(rows, 1):
        if t == opp:
            return f, i
    raise AssertionError(f"{opp} not in ratings")


# ---------------------------------------------------------------------------
# Value pairs
# ---------------------------------------------------------------------------

def test_value_pairs_invariants():
    data = landing_demos.build_landing_demo_data()
    pairs = data["value_pairs"]
    assert 1 <= len(pairs) <= 8
    seen = {}
    for p in pairs:
        a, b = p["a"], p["b"]
        # Winner is always side "a" pre-flip, and strictly more valuable.
        assert p["pick"] == 0
        assert a["v"] > b["v"]
        assert 5 <= a["v"] - b["v"] <= 60
        # No rank label may leak into the question or the player meta.
        assert not RANK_LABEL.search(p["q"]), p["q"]
        assert not RANK_LABEL.search(a["m"]), a["m"]
        assert not RANK_LABEL.search(b["m"]), b["m"]
        assert re.fullmatch(r"(QB|RB|WR|TE) · [A-Z]{2,4}", a["m"])
        assert re.fullmatch(r"(QB|RB|WR|TE) · [A-Z]{2,4}", b["m"])
        # The reveal carries both values.
        assert f"{a['v']:,}" in p["why"] and f"{b['v']:,}" in p["why"]
        for n in (a["n"], b["n"]):
            seen[n] = seen.get(n, 0) + 1
    # Per-player usage cap.
    assert all(c <= 2 for c in seen.values())


def test_value_pairs_star_pool_floor():
    pairs = landing_demos.build_landing_demo_data()["value_pairs"]
    for p in pairs:
        assert p["b"]["v"] >= 660 - 1  # rounded display of a >=660 value


# ---------------------------------------------------------------------------
# Start/sit pairs (live projections, fixture-driven)
# ---------------------------------------------------------------------------

def _fixture_rows():
    """Projection rows in the exact shape _live_projection_rows returns:
    {pos: [(name, team, opp, loc, ppr_pts), ...]} sorted best first."""
    return {
        "QB": [
            ("Alpha QB", "BUF", "MIA", "vs", 24.6),
            ("Beta QB", "BAL", "KC", "@", 24.1),
            ("Gamma QB", "CIN", "PIT", "vs", 22.0),
            ("Delta QB", "LAC", "DEN", "vs", 20.5),
            ("Epsilon QB", "DET", "CLE", "vs", 19.0),
        ],
        "RB": [
            ("Alpha RB", "DET", "CLE", "vs", 19.8),
            ("Beta RB", "BAL", "KC", "@", 19.5),
            ("Gamma RB", "MIA", "BUF", "@", 15.0),
        ],
        "WR": [
            ("Alpha WR", "CIN", "PIT", "vs", 18.2),
            ("Beta WR", "ATL", "WAS", "vs", 18.0),
        ],
        "TE": [
            ("Alpha TE", "DET", "CLE", "vs", 12.4),
            ("Beta TE", "BAL", "KC", "@", 12.1),
        ],
    }


def _fixture_ratings():
    """Ratings doc in the real file's shape. RB points allowed are set so
    the RB why-line is checkable: CLE softest, KC stiffest of the set."""
    teams = ["MIA", "KC", "PIT", "DEN", "CLE", "WAS"]
    ratings = {
        t: {pos: {"fpts": 10.0 + i} for pos in ("QB", "RB", "WR", "TE")}
        for i, t in enumerate(teams)
    }
    ratings["CLE"]["RB"] = {"fpts": 29.6}
    ratings["KC"]["RB"] = {"fpts": 17.6}
    ratings["MIA"]["QB"] = {"fpts": 21.4}
    ratings["KC"]["QB"] = {"fpts": 16.2}
    return ratings


@pytest.fixture()
def _live_demo(monkeypatch, tmp_path):
    """Opt into the live start/sit path with fixture data everywhere."""
    ratings_path = tmp_path / "ratings.json"
    ratings_path.write_text(json.dumps(
        {"ratings": _fixture_ratings(), "through_week": 6}))
    monkeypatch.setattr(landing_demos, "MATCHUP_RATINGS_PATH", ratings_path)
    monkeypatch.setattr(landing_demos, "_demo_season_week", lambda: (2026, 7))
    monkeypatch.setattr(
        landing_demos, "_live_projection_rows",
        lambda season, week: _fixture_rows())
    return ratings_path


def test_start_sit_pairs_come_from_live_rows(_live_demo):
    data = landing_demos.build_landing_demo_data()
    pairs = data["start_sit_pairs"]
    assert len(pairs) == 5
    assert [p["a"]["n"] for p in pairs] == [
        "Alpha QB", "Alpha RB", "Alpha WR", "Alpha TE", "Gamma QB",
    ]
    for p in pairs:
        assert p["pick"] in (0, 1)
        winner = p["a"] if p["pick"] == 0 else p["b"]
        loser = p["b"] if p["pick"] == 0 else p["a"]
        assert float(winner["pr"]) > float(loser["pr"])
        assert re.fullmatch(r"[A-Z]{2,4} · (vs|@) [A-Z]{2,4}", p["a"]["m"])
        assert re.fullmatch(r"[A-Z]{2,4} · (vs|@) [A-Z]{2,4}", p["b"]["m"])
    qb = pairs[0]
    assert qb["a"] == {"n": "Alpha QB", "m": "BUF · vs MIA", "pr": "24.6"}
    assert qb["b"] == {"n": "Beta QB", "m": "BAL · @ KC", "pr": "24.1"}
    assert qb["q"] == "Week 7 · One QB spot. Who do you start?"
    # The week label names the projections' week (7), not the ratings
    # file's through_week (6).
    assert data["week_label"] == "Week 7"


def test_start_sit_superflex_excludes_qb_pair(_live_demo):
    pairs = landing_demos.build_landing_demo_data()["start_sit_pairs"]
    sf = pairs[-1]
    assert sf["q"] == "Week 7 · Your superflex spot. Who do you start?"
    assert {sf["a"]["n"], sf["b"]["n"]} == {"Gamma QB", "Delta QB"}


def test_start_sit_why_matches_ratings(_live_demo):
    ratings = _fixture_ratings()
    fw, rw = _fpts_rank(ratings, "CLE", "RB")
    fl, rl = _fpts_rank(ratings, "KC", "RB")
    expected = (
        f"<b>Start Alpha RB.</b> +0.3 projected. "
        f"In our ratings CLE allows {fw:.1f} RB pts/g ({_ordinal(rw)}-most) "
        f"and KC allows {fl:.1f} ({_ordinal(rl)}-most)."
    )
    pairs = landing_demos.build_landing_demo_data()["start_sit_pairs"]
    rb = next(p for p in pairs if p["a"]["n"] == "Alpha RB")
    assert rb["pick"] == 0
    assert rb["why"] == expected


def test_start_sit_lopsided_pool_skips_position(monkeypatch, _live_demo):
    rows = _fixture_rows()
    rows["RB"] = [("Alpha RB", "DET", "CLE", "vs", 20.0),
                  ("Beta RB", "BAL", "KC", "@", 15.0)]
    monkeypatch.setattr(
        landing_demos, "_live_projection_rows", lambda season, week: rows)
    pairs = landing_demos.build_landing_demo_data()["start_sit_pairs"]
    assert all("RB spot" not in p["q"] for p in pairs)
    assert len(pairs) == 4


def test_start_sit_empty_without_projections(_live_demo):
    pairs = landing_demos.build_landing_demo_data()["start_sit_pairs"]
    assert len(pairs) == 5  # fixture sanity: the opt-in path is live


def test_no_frozen_board_in_source():
    """The demo must never go back to a baked-in snapshot of names/numbers."""
    src = Path(landing_demos.__file__).read_text(encoding="utf-8")
    assert "_BOARD" not in src
    assert "Josh Allen" not in src
    assert "_live_projection_rows" in src
    assert "get_week_projections_cached" in src


# ---------------------------------------------------------------------------
# Week label + robustness
# ---------------------------------------------------------------------------

def test_week_label_falls_back_to_this_week(monkeypatch):
    monkeypatch.setattr(landing_demos, "_nfl_state_week", lambda: None)
    data = landing_demos.build_landing_demo_data()
    # The checked-in ratings file has through_week = None.
    assert _ratings_doc().get("through_week") is None
    assert data["week_label"] == "This week"
    assert data["start_sit_pairs"] == []


def test_week_label_from_nfl_state(monkeypatch):
    monkeypatch.setattr(landing_demos, "_nfl_state_week", lambda: 7)
    data = landing_demos.build_landing_demo_data()
    assert data["week_label"] == "Week 7"


def test_missing_files_graceful(monkeypatch, tmp_path):
    monkeypatch.setattr(landing_demos, "MODEL_VALUES_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(landing_demos, "MATCHUP_RATINGS_PATH", tmp_path / "nope2.json")
    monkeypatch.setattr(landing_demos, "_nfl_state_week", lambda: None)
    data = landing_demos.build_landing_demo_data()
    assert data == {"value_pairs": [], "start_sit_pairs": [], "week_label": "This week"}
