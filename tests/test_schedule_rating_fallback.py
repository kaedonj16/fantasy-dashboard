from __future__ import annotations

import json

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("openai")

import app as appmod
from utils.defensive_matchup_ratings import scoring_profile_hash


def _write(path, ratings, profile=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scoring_profile": profile, "ratings": ratings}))


def _reset():
    appmod._MATCHUP_RATINGS_CACHE.clear()
    appmod._MATCHUP_RATINGS_TS.clear()
    appmod._MATCHUP_RATINGS_META.clear()


def test_exact_profile_is_preferred_and_cache_is_profile_isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset()
    ppr, half = {"rec": 1.0}, {"rec": 0.5}
    default = {"BUF": {"RB": {"ease": 10}}}
    exact = {"BUF": {"RB": {"adjusted_multiplier": 1.2}}}
    _write(tmp_path / "cache/matchup_ratings_s2026.json", default, "default")
    _write(tmp_path / f"cache/matchup_ratings_s2026_{scoring_profile_hash(ppr)}.json",
           exact, scoring_profile_hash(ppr))

    assert appmod._load_matchup_ratings(2026, ppr) == exact
    assert appmod._matchup_ratings_metadata(2026, ppr)["rating_source"] == "exact-profile"
    assert appmod._load_matchup_ratings(2026, half) == default
    assert appmod._matchup_ratings_metadata(2026, half)["rating_source"] == "default-profile-fallback"
    assert len(appmod._MATCHUP_RATINGS_CACHE) == 2


def test_default_fallback_restores_legacy_ranks_sos_with_estimated_percent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset()
    ratings = {
        "BUF": {"RB": {"z": 0.8, "ease": 90, "n": 8, "fpts": 25}},
        "MIA": {"RB": {"z": -0.5, "ease": 20, "n": 8, "fpts": 14}},
    }
    _write(tmp_path / "cache/matchup_ratings_s2026.json", ratings, "standard-ppr")
    ranks, total, info, _ = appmod._matchup_rank_table(2026, "RB", {"rec": 0.5})

    assert (ranks, total) == ({"BUF": 1, "MIA": 2}, 2)
    assert info["BUF"]["rating_source"] == "default-profile-fallback"
    assert info["BUF"]["rank_value"] == 90
    # Z-score schema synthesizes an estimated multiplier (1.0 + z * 0.10) so the
    # Schedule Assistant "Adj Avg" column shows values instead of N/A.
    assert info["BUF"]["multiplier"] == pytest.approx(1.08)
    assert info["BUF"]["adjusted_percent"] == pytest.approx(8.0)
    assert info["MIA"]["multiplier"] == pytest.approx(0.95)
    assert info["MIA"]["adjusted_percent"] == pytest.approx(-5.0)
    # SOS consumes the same rank value and omits the bye represented by None.
    from utils.defensive_matchup_ratings import rank_team_schedules
    assert rank_team_schedules({"NE": ["BUF", None], "NYJ": ["MIA", "BUF"]},
                               {t: row["rank_value"] for t, row in info.items()})["NE"][2] == 90


def test_unavailable_and_new_multiplier_schema(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _reset()
    assert appmod._load_matchup_ratings(2026, {"rec": 1}) == {}
    assert appmod._matchup_ratings_metadata(2026, {"rec": 1})["rating_source"] == "unavailable"

    _reset()
    ratings = {"BUF": {"RB": {"adjusted_multiplier": 1.125,
                                "adjusted_percent": 12.5,
                                "raw_allowed_per_game": 24,
                                "sample_size": 7, "confidence": "medium"}}}
    _write(tmp_path / "cache/matchup_ratings_s2026.json", ratings, "standard-ppr")
    ranks, _, info, _ = appmod._matchup_rank_table(2026, "RB", {"rec": 0})
    assert ranks == {"BUF": 1}
    assert info["BUF"]["adjusted_percent"] == 12.5
    assert info["BUF"]["fpts"] == 24

