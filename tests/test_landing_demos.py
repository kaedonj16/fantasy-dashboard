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
# Start/sit pairs
# ---------------------------------------------------------------------------

def test_start_sit_count_and_shape():
    data = landing_demos.build_landing_demo_data()
    pairs = data["start_sit_pairs"]
    assert len(pairs) == 5
    assert [p["a"]["n"] for p in pairs] == [
        "Josh Allen",
        "Jahmyr Gibbs",
        "Drake London",
        "Sam LaPorta",
        "Justin Herbert",
    ]
    for p in pairs:
        assert p["pick"] in (0, 1)
        winner = p["a"] if p["pick"] == 0 else p["b"]
        loser = p["b"] if p["pick"] == 0 else p["a"]
        assert float(winner["pr"]) > float(loser["pr"])
        assert re.fullmatch(r"[A-Z]{2,4} · (vs|@) [A-Z]{2,4}", p["a"]["m"])
        assert re.fullmatch(r"[A-Z]{2,4} · (vs|@) [A-Z]{2,4}", p["b"]["m"])


def test_start_sit_why_matches_ratings_file():
    """Recompute the RB pair's reasoning independently from the ratings file."""
    ratings = _ratings_doc()["ratings"]
    fw, rw = _fpts_rank(ratings, "CLE", "RB")
    fl, rl = _fpts_rank(ratings, "KC", "RB")
    expected = (
        f"<b>Start Jahmyr Gibbs.</b> +2.2 projected. "
        f"In our ratings CLE allows {fw:.1f} RB pts/g ({_ordinal(rw)}-most) "
        f"and KC allows {fl:.1f} ({_ordinal(rl)}-most)."
    )
    pairs = landing_demos.build_landing_demo_data()["start_sit_pairs"]
    rb = next(p for p in pairs if p["a"]["n"] == "Jahmyr Gibbs")
    assert rb["pick"] == 0
    assert rb["a"] == {"n": "Jahmyr Gibbs", "m": "DET · vs CLE", "pr": "19.8"}
    assert rb["b"] == {"n": "Derrick Henry", "m": "BAL · @ KC", "pr": "17.6"}
    assert rb["why"] == expected


def test_start_sit_home_away_markers():
    pairs = landing_demos.build_landing_demo_data()["start_sit_pairs"]
    by_name = {p["a"]["n"]: p for p in pairs}
    # Chase and Burrow are home vs PIT; the KC trio are away.
    assert by_name["Drake London"]["b"]["m"] == "CIN · vs PIT"
    assert by_name["Justin Herbert"]["b"]["m"] == "CIN · vs PIT"
    assert by_name["Josh Allen"]["b"]["m"] == "BAL · @ KC"


# ---------------------------------------------------------------------------
# Week label + robustness
# ---------------------------------------------------------------------------

def test_week_label_falls_back_to_this_week(monkeypatch):
    monkeypatch.setattr(landing_demos, "_nfl_state_week", lambda: None)
    data = landing_demos.build_landing_demo_data()
    # The checked-in ratings file has through_week = None.
    assert _ratings_doc().get("through_week") is None
    assert data["week_label"] == "This week"
    assert data["start_sit_pairs"][0]["q"].startswith("This week · ")


def test_week_label_from_nfl_state(monkeypatch):
    monkeypatch.setattr(landing_demos, "_nfl_state_week", lambda: 7)
    data = landing_demos.build_landing_demo_data()
    assert data["week_label"] == "Week 7"
    assert data["start_sit_pairs"][0]["q"] == "Week 7 · One QB spot. Who do you start?"
    assert data["start_sit_pairs"][-1]["q"] == "Week 7 · Your superflex spot. Who do you start?"


def test_missing_files_graceful(monkeypatch, tmp_path):
    monkeypatch.setattr(landing_demos, "MODEL_VALUES_PATH", tmp_path / "nope.json")
    monkeypatch.setattr(landing_demos, "MATCHUP_RATINGS_PATH", tmp_path / "nope2.json")
    monkeypatch.setattr(landing_demos, "_nfl_state_week", lambda: None)
    data = landing_demos.build_landing_demo_data()
    assert data == {"value_pairs": [], "start_sit_pairs": [], "week_label": "This week"}
