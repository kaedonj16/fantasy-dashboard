"""Stage 2 resilience for the weekly rookie pipeline (2026-11-23).

On Sunday 2026-11-22 the weekly rookie refresh died for the third
straight week: Postgres was in recovery mode, so the CFBD persistent
store missed and everything fetched live; CFBD answered 429s and 15s
timeouts; /stats/player/season loaded 0 players for 2025 and 2026; and
the play-by-play targets sweep kept retrying until the step's 10800s
timeout killed the whole step. The board stayed at its Nov 1 grades.

Two defenses are pinned here:

1. Stage 2's live fetch is fail-soft (pipeline._stage2_live_stats):
   a fetch that raises, returns nothing, or returns no rows for the
   in-progress season is "not usable" - the staged pipeline then skips
   the stats write and grades from the season stats already stored in
   the DB (Stage 6 loads from the DB either way).

2. The play-by-play targets sweep takes a wall-clock budget
   (fetch_cfbd_targets_from_plays time_budget_s): when it expires the
   sweep stops starting new work and returns stored totals plus the
   additions validated so far, instead of consuming the step.
"""

import pytest

pytest.importorskip("pandas")  # noqa: F401  (keeps CI shard conventions)

import datetime as _dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_building.rookie_pipeline.ingestion as ing
import data_building.rookie_pipeline.pipeline as pl


# ─────────────────────────────────────────────────────────────────────
# 1. Stage 2 usability judgment
# ─────────────────────────────────────────────────────────────────────

_PROSPECTS = [{"player_id": "ROOKIE_2027_a"}, {"player_id": "ROOKIE_2027_b"}]


def _patch_fetch(monkeypatch, result=None, raises=None):
    calls = {}

    def fake_fetch(draft_year, **kwargs):
        calls["draft_year"] = draft_year
        calls["kwargs"] = kwargs
        if raises is not None:
            raise raises
        return result

    monkeypatch.setattr(ing, "fetch_cfbd_college_stats", fake_fetch)
    return calls


class TestStage2LiveStats:
    def test_empty_fetch_is_unusable(self, monkeypatch):
        _patch_fetch(monkeypatch, result={})
        stats, usable = pl._stage2_live_stats(2027, _PROSPECTS, False)
        assert stats == {}
        assert usable is False

    def test_raising_fetch_is_unusable(self, monkeypatch):
        _patch_fetch(monkeypatch, raises=RuntimeError("CFBD down"))
        stats, usable = pl._stage2_live_stats(2027, _PROSPECTS, False)
        assert stats == {}
        assert usable is False

    def test_no_current_season_rows_is_unusable(self, monkeypatch):
        # The 2026-11-22 signature: past seasons came back, the
        # in-progress season (2026 for the 2027 class) loaded nobody.
        _patch_fetch(monkeypatch, result={
            "some player": [{"season": 2025, "receiving_yards": 900}],
            "other player": [{"season": 2024, "receiving_yards": 500}],
        })
        stats, usable = pl._stage2_live_stats(2027, _PROSPECTS, False)
        assert usable is False

    def test_current_season_rows_are_usable(self, monkeypatch):
        _patch_fetch(monkeypatch, result={
            "some player": [{"season": 2025, "receiving_yards": 900},
                            {"season": 2026, "receiving_yards": 300}],
        })
        stats, usable = pl._stage2_live_stats(2027, _PROSPECTS, False)
        assert usable is True
        assert stats

    def test_completed_season_accepts_any_rows(self, monkeypatch):
        # Post-January the class's seasons are all finished; any
        # non-empty fetch is the real thing.
        _patch_fetch(monkeypatch, result={
            "some player": [{"season": 2026, "receiving_yards": 900}],
        })
        stats, usable = pl._stage2_live_stats(2027, _PROSPECTS, True)
        assert usable is True

    def test_fetch_is_bounded_and_unchanged_in_shape(self, monkeypatch):
        calls = _patch_fetch(monkeypatch, result={
            "some player": [{"season": 2026, "receiving_yards": 1}],
        })
        pl._stage2_live_stats(2027, _PROSPECTS, False)
        assert calls["draft_year"] == 2027
        assert calls["kwargs"]["fetch_games_played"] is True
        assert calls["kwargs"]["player_ids"] == ["ROOKIE_2027_a", "ROOKIE_2027_b"]
        assert calls["kwargs"]["targets_time_budget_s"] == \
            pl._STAGE2_TARGETS_TIME_BUDGET_S


class TestStage2Wiring:
    def test_staged_pipeline_routes_stage2_through_the_helper(self):
        import pathlib
        src = (pathlib.Path(pl.__file__)
               .read_text(encoding="utf-8"))
        staged = src[src.index("def run_rookie_pipeline_staged"):]
        assert "_stage2_live_stats(" in staged
        assert "STAGE 2 FALLBACK" in staged
        assert '"stats_fallback": stats_fallback' in staged


# ─────────────────────────────────────────────────────────────────────
# 2. Targets sweep time budget
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture()
def targets_env(monkeypatch):
    """Fake stored-target state + fake CFBD transport for the plays
    sweep, pinned to 2026-09-26 like the other targets fixtures.
    state["plays_calls"] counts /plays calls. The fake clock advances
    50s per read, so elapsed time is a deterministic function of how
    many budget checks the sweep has made."""
    state = {"plays_calls": 0, "requests": [], "clock_reads": 0}

    def fake_monotonic():
        state["clock_reads"] += 1
        return state["clock_reads"] * 50.0

    monkeypatch.setattr(ing, "CFBD_KEY", "test-key")
    monkeypatch.setattr(ing, "_today", lambda: _dt.date(2026, 9, 26))
    monkeypatch.setattr(ing.time, "sleep", lambda _s: None)
    monkeypatch.setattr(ing.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(
        ing, "_get_existing_targets",
        lambda draft_year, seasons: {
            "jeremiah smith": {2026: {"targets": 40, "team": "Ohio State"}},
        })
    monkeypatch.setattr(
        ing, "_get_fetched_weeks",
        lambda kind, seasons: {2026: {"Ohio State": {1, 2, 3, 4}}})
    monkeypatch.setattr(
        ing, "_get_team_targets_baselines",
        lambda seasons: {(2026, "Ohio State"): 200})

    def fake_cfbd_get(path, params=None, retries=2):
        params = params or {}
        state["requests"].append((path, dict(params)))
        if path == "/plays":
            state["plays_calls"] += 1
            return []
        if path in ("/roster", "/games/players"):
            return []
        raise AssertionError(f"Unexpected path in fake transport: {path}")

    monkeypatch.setattr(ing, "_cfbd_get", fake_cfbd_get)
    return state


class TestTargetsTimeBudget:
    def test_zero_budget_returns_stored_without_fetching(self, targets_env,
                                                         capsys):
        result = ing.fetch_cfbd_targets_from_plays(
            2027, {2026: ["Ohio State"]}, time_budget_s=0)
        # Stored totals ride back untouched; not one network call made.
        assert result["jeremiah smith"][2026]["targets"] == 40
        assert result["jeremiah smith"][2026]["target_share"] == \
            round(40 / 200, 4)
        assert targets_env["requests"] == []
        assert "TIME BUDGET EXCEEDED" in capsys.readouterr().out

    def test_budget_stops_the_sweep_mid_run(self, targets_env, capsys):
        # A finished season, two teams, transport that yields nothing:
        # without a budget the first team burns a season-level call
        # plus a per-week fallback call for every candidate week.
        # The fake clock (+50s per read) passes the season and team
        # checks (50s, 100s elapsed of the 100s budget), then blows
        # the budget at the first per-week fallback check: the sweep
        # must abandon the loop, never touch the second team (teams
        # run in sorted order, so Michigan is first, Ohio State is
        # the one left untouched), and still return (with nothing
        # new validated).
        result = ing.fetch_cfbd_targets_from_plays(
            2027, {2025: ["Ohio State", "Michigan"]}, time_budget_s=100)
        assert result == {}
        teams_called = {p["team"] for (path, p) in targets_env["requests"]
                        if path == "/plays"}
        assert teams_called == {"Michigan"}
        assert targets_env["plays_calls"] == 1  # season-level call only
        assert "TIME BUDGET EXCEEDED" in capsys.readouterr().out

    def test_no_budget_keeps_run_to_completion_shape(self, targets_env):
        # Default (no budget): the zero-new-weeks stored path is
        # untouched - stored totals return, week 5 is attempted for
        # the current season exactly as before.
        result = ing.fetch_cfbd_targets_from_plays(
            2027, {2026: ["Ohio State"]})
        assert result["jeremiah smith"][2026]["targets"] == 40
