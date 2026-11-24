"""Tests for the finished-season targets fetch fix.

Two proven failure modes on finished seasons (live-verified against
CFBD payloads for LSU / Ohio State / Georgia 2023):

- Full-name resolution gap: the no-jersey text form ("pass complete
  to Kaleb Jackson") was reduced to (surname, first initial), and the
  roster map drops (surname, initial) keys shared by two teammates
  (LSU 2023 carried Kaleb AND Kylin Jackson), so those receivers'
  receptions vanished from the parse and the box-score REC equality
  discarded the whole game.  The text was never ambiguous - it spells
  the full name - so an exact full-name roster hit now wins first.
- Bare-incompletion coverage discards: some feeds record
  incompletions without naming a target ("Jayden Daniels pass
  incomplete"), so a sound parse (every completion named) scores
  named coverage under the 0.85 gate and the game is discarded whole.
  For a FINISHED season the exact REC equality against the final box
  score is the stronger proof, so it may override the coverage gate
  (subject to a 0.5 floor).  Current seasons keep the strict gate.

Synthetic data + fake transport only; no network, no real DB.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import datetime as _dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_building.rookie_pipeline import ingestion as ing


ROSTER = [
    {"firstName": "Kaleb", "lastName": "Jackson", "jersey": 5},
    {"firstName": "Kylin", "lastName": "Jackson", "jersey": 6},
    {"firstName": "Alpha", "lastName": "Receiver", "jersey": 1},
    {"firstName": "Beta", "lastName": "Receiver", "jersey": 2},
]


def _play(pid, text, gid=999):
    return {"id": pid, "gameId": gid, "week": 1, "offense": "State",
            "playText": text, "playType": "Pass"}


def _box(rec):
    """rec: {athlete_name: receptions} for team State in game 999."""
    return [{"id": 999, "teams": [{
        "team": "State",
        "categories": [{"name": "receiving", "types": [{
            "name": "REC",
            "athletes": [{"name": n, "stat": c} for n, c in rec.items()],
        }]}],
    }]}]


@pytest.fixture()
def fetch_env(monkeypatch):
    """Fake DB helpers + fake CFBD transport. state holds the plays
    payload and box payload the transport serves for week 1, and
    records every request."""
    state = {"requests": [], "plays": [], "box": []}
    monkeypatch.setattr(ing, "CFBD_KEY", "test-key")
    monkeypatch.setattr(ing, "_today", lambda: _dt.date(2026, 9, 26))
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ing, "_get_existing_targets",
                        lambda draft_year, seasons: {})
    monkeypatch.setattr(ing, "_get_fetched_weeks",
                        lambda kind, seasons: {})
    monkeypatch.setattr(ing, "_get_team_targets_baselines",
                        lambda seasons: {})
    monkeypatch.setattr(ing, "_mark_weeks_fetched", lambda *a, **k: None)
    monkeypatch.setattr(ing, "_record_team_targets", lambda *a, **k: None)
    monkeypatch.setattr(ing, "_candidate_weeks",
                        lambda season, today: [1])

    def fake_cfbd_get(path, params=None, retries=2):
        params = params or {}
        state["requests"].append((path, dict(params)))
        if path == "/roster":
            return ROSTER
        if path == "/plays":
            if "week" not in params:
                return None  # season-level call: CFBD 400s (week required)
            return state["plays"]
        if path == "/games/players":
            return state["box"]
        raise AssertionError(f"Unexpected path in fake transport: {path}")

    monkeypatch.setattr(ing, "_cfbd_get", fake_cfbd_get)
    return state


def _fetch(state, season=2023, **kw):
    return ing.fetch_cfbd_targets_from_plays(
        2024, {season: ["State"]}, **kw)


# ── Full-name resolution ────────────────────────────────────────────

class TestFullNameResolution:
    def test_colliding_surname_initial_resolves_by_full_name(self):
        by_jersey, by_name, names = ing._build_roster_maps(ROSTER)
        # The (surname, initial) key is genuinely ambiguous on this
        # roster and must stay out of by_name...
        assert ("jackson", "k") not in by_name
        plays = [
            _play(1, "Qb One pass complete to Kaleb Jackson for 9 yds"),
            _play(2, "Qb One pass complete to Kaleb Jackson for 6 yds"),
            _play(3, "Qb One pass complete to #6 K.Jackson for 4 yds"),
        ]
        derived = ing._derive_game_targets(plays, "State", by_jersey,
                                           by_name, names)
        # ...while the full-name text resolves Kaleb, and the jersey
        # form still resolves Kylin.
        assert derived["receptions"] == {"kaleb jackson": 2,
                                         "kylin jackson": 1}
        assert derived["named"] == 3

    def test_full_name_miss_falls_back_to_initial_map(self):
        by_jersey, by_name, names = ing._build_roster_maps(ROSTER)
        # "Al Receiver" is not a roster full name; the (surname,
        # initial) path must still get its chance (and miss cleanly).
        assert ing._resolve_receiver(None, "A", "Receiver", by_jersey,
                                     by_name, first_full="Al",
                                     roster_names=names) == "alpha receiver"


# ── Coverage fallback (finished seasons only) ───────────────────────

def _bare_incompletion_game(named_completions=5, bare=5):
    """named_completions completions split Alpha 4 / Beta 1 (for 5),
    plus `bare` target-less incompletions."""
    plays = []
    pid = 0
    split = [(4, "Alpha Receiver"), (1, "Beta Receiver")]
    made = 0
    for count, who in split:
        for _ in range(count):
            if made >= named_completions:
                break
            pid += 1
            plays.append(_play(
                pid, f"Qb One pass complete to {who} for 8 yds"))
            made += 1
    for _ in range(bare):
        pid += 1
        plays.append(_play(pid, "Qb One pass incomplete"))
    return plays


class TestCoverageFallback:
    def test_finished_season_bare_incompletions_validate_on_rec(self,
                                                              fetch_env):
        # 5 named of 10 attempts: coverage 0.5, under the 0.85 gate,
        # at the fallback floor; REC equality exact.
        fetch_env["plays"] = _bare_incompletion_game()
        fetch_env["box"] = _box({"Alpha Receiver": 4, "Beta Receiver": 1})
        result = _fetch(fetch_env)
        assert result["alpha receiver"][2023]["targets"] == 4
        assert result["beta receiver"][2023]["targets"] == 1

    def test_current_season_same_game_still_fails(self, fetch_env):
        # Identical payload in the CURRENT season: the strict gate
        # applies (box scores are not final truth yet), so nothing is
        # collected.  Regression pin for the scoping of the fix.
        fetch_env["plays"] = _bare_incompletion_game()
        fetch_env["box"] = _box({"Alpha Receiver": 4, "Beta Receiver": 1})
        result = _fetch(fetch_env, season=2026)
        assert result == {}

    def test_finished_season_below_floor_still_fails(self, fetch_env):
        # 3 named of 10 attempts: coverage 0.3 < floor 0.5 - REC
        # equality alone must not rescue a degenerate payload.
        fetch_env["plays"] = _bare_incompletion_game(named_completions=3,
                                                     bare=7)
        fetch_env["box"] = _box({"Alpha Receiver": 3})
        result = _fetch(fetch_env)
        assert result == {}

    def test_finished_season_rec_mismatch_still_fails(self, fetch_env):
        # Coverage in the fallback band but the box disagrees: the
        # fallback is REC equality deciding, not coverage waived.
        fetch_env["plays"] = _bare_incompletion_game()
        fetch_env["box"] = _box({"Alpha Receiver": 5, "Beta Receiver": 1})
        result = _fetch(fetch_env)
        assert result == {}

    def test_fallback_composes_with_team_seasons_scope(self, fetch_env):
        fetch_env["plays"] = _bare_incompletion_game()
        fetch_env["box"] = _box({"Alpha Receiver": 4, "Beta Receiver": 1})
        result = ing.fetch_cfbd_targets_from_plays(
            2024, {2023: ["State", "Other"]},
            team_seasons={("State", 2023)})
        assert result["alpha receiver"][2023]["targets"] == 4
        plays_teams = {p["team"] for (path, p) in fetch_env["requests"]
                       if path == "/plays"}
        assert plays_teams == {"State"}
