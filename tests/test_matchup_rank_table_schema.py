"""_matchup_rank_table must rank teams from the current ratings schema.

The ratings pipeline stores an opponent-adjusted z-score / 0-100 ``ease`` per
team (higher == easier for the position). An older schema used
``adjusted_multiplier``. Ranking only by ``adjusted_multiplier`` left every rank
empty on the current files, which stripped the schedule-rank chips off the
player-modal game log.
"""
from __future__ import annotations

import pytest

# app imports pandas (first) then openai/stripe/etc. at module load, so gate on
# the heavy deps before importing it -- the lint CI job runs the unit suite with
# a minimal dependency set and must skip this module cleanly, not error on it.
pytest.importorskip("flask")
pytest.importorskip("pandas")
pytest.importorskip("openai")

import app as appmod


def _run(monkeypatch, ratings):
    monkeypatch.setattr(appmod, "_load_matchup_ratings", lambda *a, **k: ratings)
    return appmod._matchup_rank_table(2026, "RB")


def test_ranks_by_ease_when_no_multiplier(monkeypatch):
    ratings = {
        "CIN": {"RB": {"z": 0.57, "ease": 100.0, "n": 15, "fpts": 29.6}},
        "DEN": {"RB": {"z": 0.02, "ease": 50.0, "n": 15, "fpts": 22.0}},
        "SEA": {"RB": {"z": -0.60, "ease": 13.3, "n": 15, "fpts": 15.0}},
    }
    ranks, total, info, _is_z = _run(monkeypatch, ratings)
    assert total == 3
    # Higher ease == easier matchup == rank 1.
    assert ranks["CIN"] == 1
    assert ranks["DEN"] == 2
    assert ranks["SEA"] == 3
    # The real fpts is preserved for info consumers, not clobbered to None.
    assert info["CIN"]["fpts"] == 29.6


def test_still_ranks_legacy_multiplier_schema(monkeypatch):
    ratings = {
        "CIN": {"RB": {"adjusted_multiplier": 1.20, "raw_allowed_per_game": 29.6}},
        "SEA": {"RB": {"adjusted_multiplier": 0.80, "raw_allowed_per_game": 15.0}},
    }
    ranks, total, info, _is_z = _run(monkeypatch, ratings)
    assert total == 2
    assert ranks["CIN"] == 1 and ranks["SEA"] == 2
    # Legacy schema keeps its percent-above-expected.
    assert round(info["CIN"]["adjusted_percent"], 1) == 20.0


def test_empty_ratings_yield_no_ranks(monkeypatch):
    ranks, total, info, _is_z = _run(monkeypatch, {})
    assert ranks == {} and total == 0
