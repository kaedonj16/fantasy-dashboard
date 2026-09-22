"""``get_players_map`` must share one full-universe map across league contexts.

A member of many leagues builds one ``build_league_context`` per league, each
of which calls ``get_players_map`` on the same process-shared player global.
Rebuilding and retaining a fresh ~11k-entry {pid: {name, team, pos}} dict per
league was a per-league memory cost that scaled with league count and drove the
portfolio OOM. The map is read-only for every caller, so it is memoized on the
source's identity and shared. These tests guard both the sharing invariant and
that the derived fields did not regress when the builder was split out.
"""
from dashboard_services.players import get_players_map


def _sample_players():
    return {
        "1": {"full_name": "Josh Allen", "team": "BUF", "position": "QB"},
        "2": {"first_name": "A.J.", "last_name": "Brown", "team": "PHI",
              "fantasy_positions": ["WR"]},
        "3": {"search_full_name": "jaylen waddle"},
    }


def test_same_source_returns_the_same_shared_instance():
    players = _sample_players()
    first = get_players_map(players)
    second = get_players_map(players)
    # Identity, not just equality: a copy per call would defeat the memory fix.
    assert first is second


def test_distinct_source_objects_rebuild():
    a = get_players_map(_sample_players())
    b = get_players_map(_sample_players())
    assert a is not b


def test_empty_or_missing_source_is_safe():
    assert get_players_map(None) == {}
    assert get_players_map({}) == {}


def test_derived_fields_unchanged_after_builder_split():
    m = get_players_map(_sample_players())
    assert m["1"] == {"name": "Josh Allen", "team": "BUF", "pos": "QB"}
    # first/last name join, position pulled from fantasy_positions
    assert m["2"] == {"name": "A.J. Brown", "team": "PHI", "pos": "WR"}
    # search_full_name fallback for name; missing team defaults to FA
    assert m["3"]["name"] == "jaylen waddle"
    assert m["3"]["team"] == "FA"


def test_source_assertion_memoization_is_in_place():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1]
           / "dashboard_services" / "players.py").read_text()
    assert "_PLAYERS_MAP_CACHE" in src
    assert "def _build_players_map" in src
