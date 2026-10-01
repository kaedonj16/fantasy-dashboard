"""Regression tests for game-log schedule matching with history-form teams.

team_for_week() resolves a player's weekly team in nflverse form (Rams =
LA, and player_team_history canonicalises site LAR back to LA), while the
schedule week files use site form (LAR / WAS / JAX). The game-log loop used
to compare them by raw string equality, so a Rams game never matched:

- played weeks rendered with a blank Date and "-" Opp (stats still showed,
  they join by week), and no matchup-rank chip;
- an unplayed week (no stats, no matched game) satisfied the bye test, so
  it rendered as a BYE row - and that phantom bye counted as an "actual"
  week, suppressing the week's projection entirely.

Seen live 2026-10-01 on Terrance Ferguson (LAR): W1-W3 dateless, a fake
bye in Week 4 (@PHI Oct 4), and no Week 4 projection row.

_game_log_matchup is the route's matching step; these tests pin its
alias behaviour. On main the helper does not exist, so they fail there.
"""
from __future__ import annotations

import inspect

import pytest

# app.py imports pandas (and flask/openai) at module load. The fast "lint"
# CI shard has flask but not pandas, so guard here - otherwise importing
# app raises at COLLECTION and aborts the whole run.
pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")

import app as appmod

# 2026 Rams schedule (site-form codes, as the schedule files store them).
W1 = [{"home": "LAR", "away": "SF", "gameDate": "20260910"},
      {"home": "SEA", "away": "NE", "gameDate": "20260909"}]
W3 = [{"home": "DEN", "away": "LAR", "gameDate": "20260927"}]
W4 = [{"home": "PHI", "away": "LAR", "gameDate": "20261004"}]
W11_BYE = [{"home": "SEA", "away": "NE", "gameDate": "20261115"}]


def test_history_form_la_matches_schedule_lar_home():
    opp, is_away, date, site = appmod._game_log_matchup("LA", W1)
    assert (opp, is_away, date) == ("SF", False, "20260910")
    assert site == "LAR"  # season header label uses site form


def test_history_form_la_matches_schedule_lar_away():
    opp, is_away, date, _ = appmod._game_log_matchup("LA", W3)
    assert (opp, is_away, date) == ("DEN", True, "20260927")


def test_unplayed_week_with_a_game_is_not_a_bye():
    # The phantom-bye regression: Week 4 @PHI is unplayed (no stats yet),
    # but the Rams DO have a game. The route marks a bye only when the
    # matchup comes back empty, so this must return the game.
    opp, is_away, date, _ = appmod._game_log_matchup("LA", W4)
    assert (opp, is_away, date) == ("PHI", True, "20261004")


def test_site_form_input_matches_history_form_schedule():
    # Reverse direction: a schedule file storing LA still matches a
    # site-form LAR team (team_abbr_keys covers both spellings).
    games = [{"home": "LA", "away": "SF", "gameDate": "20260910"}]
    opp, is_away, date, _ = appmod._game_log_matchup("LAR", games)
    assert (opp, is_away, date) == ("SF", False, "20260910")


def test_wsh_and_jac_history_forms_match_site_schedule():
    was = [{"home": "NYG", "away": "WAS", "gameDate": "20261112"}]
    opp, is_away, _, site = appmod._game_log_matchup("WSH", was)
    assert (opp, is_away, site) == ("NYG", True, "WAS")

    jax = [{"home": "JAX", "away": "GB", "gameDate": "20261125"}]
    opp, is_away, _, site = appmod._game_log_matchup("JAC", jax)
    assert (opp, is_away, site) == ("GB", False, "JAX")


def test_real_bye_week_returns_empty():
    opp, is_away, date, site = appmod._game_log_matchup("LA", W11_BYE)
    assert (opp, is_away, date) == ("", False, "")
    assert site == "LAR"


def test_unaliased_team_matches_as_before():
    games = [{"home": "DEN", "away": "KC", "gameDate": "20260927"}]
    opp, is_away, date, site = appmod._game_log_matchup("DEN", games)
    assert (opp, is_away, date, site) == ("KC", False, "20260927", "DEN")


def test_no_team_returns_empty():
    assert appmod._game_log_matchup("", W1) == ("", False, "", "")
    assert appmod._game_log_matchup(None, W1) == ("", False, "", "")


def test_route_uses_alias_aware_matchup():
    src = inspect.getsource(appmod.api_player_game_logs)
    assert "_game_log_matchup(" in src
