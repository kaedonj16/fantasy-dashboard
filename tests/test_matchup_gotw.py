import json

import pytest

# weekly_recap imports pandas at module load. Keep this module in the full-stack
# shard so the minimal unit-test job can collect without pandas.
pytest.importorskip("pandas")

from dashboard_services.ai import weekly_recap
from dashboard_services import matchups


def _selection(**overrides):
    value = {
        "platform": "sleeper", "league_id": "league-1", "season": "2025",
        "source_week": 1, "target_week": 2, "matchup_id": 7,
        "roster_ids": ["10", "20"],
    }
    value.update(overrides)
    return value


def _matchup(mid, left, right):
    return {
        "matchup_id": mid,
        "left": {"roster_id": left},
        "right": {"roster_id": right},
    }


def _context_key(selection, **overrides):
    context = dict(
        loaded=True, platform="sleeper", league_id="league-1",
        season=2025, week=2,
    )
    context.update(overrides)
    return matchups.gotw_identity_for_context(selection, **context)


def test_cached_gotw_is_scoped_to_platform_league_season_and_target_week(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly_recap, "AI_CACHE_DIR", tmp_path)
    key = weekly_recap._recap_cache_key("league-1", 2025, 1)
    (tmp_path / f"{key}.json").write_text(json.dumps({
        "content": "cached recap", "metadata": {"gotw_selection": _selection()},
    }))
    selected = weekly_recap.get_cached_gotw_selection("sleeper", "league-1", 2025, 2)
    assert selected and selected["matchup_id"] == 7
    assert weekly_recap.get_cached_gotw_selection("espn", "league-1", 2025, 2) is None
    assert weekly_recap.get_cached_gotw_selection("sleeper", "league-1", 2025, 3) is None


@pytest.mark.parametrize("selection", [None, {}, {"matchup_id": None}])
def test_missing_or_null_gotw_produces_zero_badges(selection):
    key = _context_key(selection)
    assert matchups.matchup_gotw_flags([_matchup(1, 10, 20), _matchup(2, 30, 40)], key) == [False, False]


def test_exactly_one_matching_matchup_and_nonmatches_are_not_selected():
    rows = [_matchup(6, 1, 2), _matchup("7", 10, 20), _matchup(8, 30, 40)]
    assert matchups.matchup_gotw_flags(rows, _context_key(_selection())) == [False, True, False]


def test_missing_ids_cannot_match_via_undefined_equality():
    assert matchups.normalized_matchup_identity({}) is None
    assert matchups.matchup_matches_gotw({}, {}) is False
    assert matchups.matchup_gotw_flags([{}, {}], None) == [False, False]


def test_numeric_and_string_matchup_ids_normalize_to_same_key():
    assert matchups.matchup_matches_gotw(
        _matchup(7, 99, 100), {"matchup_id": "7", "roster_ids": ["10", "20"]},
    )


def test_reversed_team_order_matches_with_deterministic_fallback():
    selection = _selection(matchup_id=None)
    row = _matchup(None, 20, 10)
    assert matchups.normalized_matchup_identity(selection) == "teams:10:20"
    assert matchups.matchup_matches_gotw(row, selection)


@pytest.mark.parametrize("field,value", [
    ("platform", "espn"), ("league_id", "league-2"),
    ("season", 2024), ("week", 3),
])
def test_gotw_from_another_context_is_rejected(field, value):
    assert _context_key(_selection(), **{field: value}) is None


def test_not_loaded_rejects_selection_and_prevents_stale_week_badge():
    old_key = _context_key(_selection())
    assert matchups.matchup_gotw_flags([_matchup(7, 10, 20)], old_key) == [True]
    new_key = _context_key(_selection(), loaded=False, week=3)
    assert new_key is None
    assert matchups.matchup_gotw_flags([_matchup(7, 10, 20)], new_key) == [False]


def test_duplicate_matchup_data_can_never_render_more_than_one_badge():
    rows = [_matchup(7, 10, 20), _matchup("7", 10, 20)]
    flags = matchups.matchup_gotw_flags(rows, _context_key(_selection()))
    rendered = "".join("GAME OF THE WEEK" if flag else "" for flag in flags)
    assert flags == [True, False]
    assert rendered.count("GAME OF THE WEEK") == 1
