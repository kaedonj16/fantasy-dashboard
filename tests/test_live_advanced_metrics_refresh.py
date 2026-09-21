"""Coverage for the live (as-a-game-finishes) Advanced Metrics refresh path.

Verifies the mechanics that let a finished game reach Advanced Metrics within
minutes instead of at the next daily cron:

  * ``fetch_week_stats(force=True)`` bypasses the normally-frozen populated cache.
  * ``build_usage_map_for_season`` forces a refetch only for the named week.
  * ``build_advanced_metrics_snapshot`` threads ``force_weeks`` to a capable
    builder and never hands the kwarg to a legacy 2-arg builder.
  * ``resolve_adv_metrics_completed_week`` includes the in-progress week the
    moment one of its games goes final, so the live refresh and the daily cron
    agree on which week the day's snapshot row covers.
"""

import json
from pathlib import Path


# --------------------------------------------------------------------------- #
# fetch_week_stats force refetch
# --------------------------------------------------------------------------- #
def test_force_refetches_populated_cache(monkeypatch, tmp_path):
    import data_building.external_data.sleeper_bulk_stats as sbs

    cache_file = tmp_path / "sleeper_stats_s2026_w3.json"
    cache_file.write_text(json.dumps({"stale": {"rec": 1}}))
    monkeypatch.setattr(sbs, "_week_cache_path", lambda s, w: str(cache_file))

    calls = {"n": 0}

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"fresh": {"rec": 9}}

    def fake_get(url, timeout=20):
        calls["n"] += 1
        return _Resp()

    monkeypatch.setattr(sbs.requests, "get", fake_get)

    # Without force, the populated cache wins and Sleeper is never hit.
    assert sbs.fetch_week_stats(2026, 3) == {"stale": {"rec": 1}}
    assert calls["n"] == 0

    # With force, it refetches and the new data overwrites the cache.
    assert sbs.fetch_week_stats(2026, 3, force=True) == {"fresh": {"rec": 9}}
    assert calls["n"] == 1
    assert json.loads(cache_file.read_text()) == {"fresh": {"rec": 9}}


# --------------------------------------------------------------------------- #
# build_usage_map_for_season forces only the named week
# --------------------------------------------------------------------------- #
def test_usage_map_forces_only_named_weeks(monkeypatch):
    import data_building.external_data.sleeper_usage as su

    seen = {}

    def fake_fetch(season, week, force=False):
        seen[int(week)] = force
        return {}

    monkeypatch.setattr(su, "fetch_week_stats", fake_fetch)
    monkeypatch.setattr(su, "fetch_season_redzone_stats", lambda season: {})
    monkeypatch.setattr(su, "load_players_index", lambda: {})

    su.build_usage_map_for_season(2026, [1, 2, 3], force_weeks=[3])

    assert seen == {1: False, 2: False, 3: True}


# --------------------------------------------------------------------------- #
# snapshot threads force_weeks to a capable builder, never to a 2-arg builder
# --------------------------------------------------------------------------- #
def _usage(games=1):
    return {
        "games": games, "avg_targets": 8, "avg_receptions": 5,
        "avg_rec_yards": 70, "avg_rec_tds": 0, "avg_carries": 0,
        "avg_rush_yards": 0, "avg_rush_tds": 0, "avg_pass_att": 0,
        "avg_pass_cmp": 0, "avg_pass_yds": 0, "avg_pass_tds": 0,
        "avg_pass_int": 0, "avg_off_snap_pct": .7,
    }


def test_snapshot_passes_force_weeks_to_capable_builder(monkeypatch):
    import data_building.advanced_metrics as am

    got = {}

    def builder(season, weeks, force_weeks=None):
        got["force_weeks"] = force_weeks
        return {"wr1": _usage()}

    monkeypatch.setattr(am, "load_matchup_ease", lambda season: {})
    monkeypatch.setattr(am, "finalize_role_scores_v2", lambda *a: None)
    monkeypatch.setattr(am, "save_metrics_snapshot",
                        lambda rows, dt, season=None, return_counts=False: (len(rows), 0))

    am.build_advanced_metrics_snapshot(
        2026, 3, as_of_date="2026-09-28",
        players_index={"wr1": {"pos": "WR", "team": "KC"}},
        usage_builder=builder, force_weeks=[3],
    )
    assert got["force_weeks"] == [3]


def test_build_weekly_metrics_forces_only_named_weeks(monkeypatch):
    import data_building.weekly_metrics as wm

    seen = {}

    def fake_fetch(season, week, force=False):
        seen[int(week)] = force
        return {}  # empty -> loop continues, no DB writes

    monkeypatch.setattr(wm, "init_weekly_metrics_db", lambda: None)
    monkeypatch.setattr(wm, "load_players_index", lambda: {})
    monkeypatch.setattr(wm, "fetch_week_stats", fake_fetch)

    wm.build_weekly_metrics(2026, weeks=[1, 2, 3], force_weeks=[3])
    assert seen == {1: False, 2: False, 3: True}


def test_snapshot_never_hands_force_weeks_to_legacy_builder(monkeypatch):
    import data_building.advanced_metrics as am

    # A 2-arg builder would raise TypeError if called with force_weeks.
    def legacy_builder(season, weeks):
        return {"wr1": _usage()}

    monkeypatch.setattr(am, "load_matchup_ease", lambda season: {})
    monkeypatch.setattr(am, "finalize_role_scores_v2", lambda *a: None)
    monkeypatch.setattr(am, "save_metrics_snapshot",
                        lambda rows, dt, season=None, return_counts=False: (len(rows), 0))

    # Passing force_weeks must not blow up even though the builder can't take it.
    result = am.build_advanced_metrics_snapshot(
        2026, 3, as_of_date="2026-09-28",
        players_index={"wr1": {"pos": "WR", "team": "KC"}},
        usage_builder=legacy_builder, force_weeks=[3],
    )
    assert result["players_inserted"] == 1


# --------------------------------------------------------------------------- #
# resolve_adv_metrics_completed_week
# --------------------------------------------------------------------------- #
def test_completed_week_includes_current_week_once_a_game_is_final(monkeypatch):
    import utils.utils as uu

    # Week 5 has a final game -> include week 5.
    monkeypatch.setattr(uu, "week_has_final_game", lambda season, week: int(week) == 5)
    assert uu.resolve_adv_metrics_completed_week(2026, 5) == 5


def test_completed_week_falls_back_when_no_current_finals(monkeypatch):
    import utils.utils as uu

    monkeypatch.setattr(uu, "week_has_final_game", lambda season, week: False)
    assert uu.resolve_adv_metrics_completed_week(2026, 5) == 4
    # Never below zero even at the very start of the season.
    assert uu.resolve_adv_metrics_completed_week(2026, 0) == 0
    assert uu.resolve_adv_metrics_completed_week(2026, 1) == 0


def test_week_has_final_game_reads_schedule_status(monkeypatch):
    import utils.utils as uu

    sched = [
        {"gameID": "g1", "gameStatusCode": "2"},   # final -> post
        {"gameID": "g2", "gameStatusCode": "0"},   # scheduled -> pre
    ]
    monkeypatch.setattr(uu, "load_week_schedule", lambda s, w: sched)
    # Neutralize kickoff-time inference so status comes from the code fallback.
    monkeypatch.setattr(uu, "normalize_game_status_from_tank01",
                        lambda g, now=None: "post" if str(g.get("gameStatusCode")) == "2" else "pre")
    assert uu.finished_game_ids_for_week(2026, 5) == ["g1"]
    assert uu.week_has_final_game(2026, 5) is True


# --------------------------------------------------------------------------- #
# refresh_live_advanced_metrics.main orchestration
# --------------------------------------------------------------------------- #
def _patch_script(monkeypatch, tmp_path, *, season_type="reg", week=5,
                  finished=("g1",)):
    import scripts.refresh_live_advanced_metrics as live
    monkeypatch.setattr(live, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(live, "get_nfl_state",
                        lambda: {"season": 2026, "week": week, "season_type": season_type})
    monkeypatch.setattr(live, "finished_game_ids_for_week",
                        lambda s, w: list(finished))
    monkeypatch.setattr(live, "resolve_adv_metrics_completed_week",
                        lambda s, w: w if finished else w - 1)
    monkeypatch.setattr(live, "load_players_index", lambda: {})
    monkeypatch.setattr(live, "_flush_app_caches", lambda: None)

    builds = {"n": 0}
    import data_building.advanced_metrics as am
    monkeypatch.setattr(am, "build_advanced_metrics_snapshot",
                        lambda *a, **k: builds.__setitem__("n", builds["n"] + 1) or {"ok": 1})
    # The live build branch also refreshes the per-week usage rows; stub it out so
    # the unit test touches no DB / Sleeper.
    import data_building.weekly_metrics as wm
    monkeypatch.setattr(wm, "build_weekly_metrics",
                        lambda *a, **k: builds.__setitem__("weekly", builds.get("weekly", 0) + 1) or 0)
    # Count nflverse pulls (return 0 rows so it never writes) so the window /
    # throttle gate can be asserted.
    monkeypatch.setattr(
        live, "_refresh_nflverse_weekly",
        lambda s, idx: builds.__setitem__("nflverse", builds.get("nflverse", 0) + 1) or 0)
    return live, builds


def test_live_main_refreshes_weekly_rows_on_build(monkeypatch, tmp_path):
    # A rebuild must also refresh the per-week usage rows that power the week
    # filter, not just the season snapshot.
    live, builds = _patch_script(monkeypatch, tmp_path, finished=("g1",))
    assert live.main(["--no-nflverse"]) == 0
    assert builds["n"] == 1
    assert builds.get("weekly") == 1


def test_live_main_skips_in_offseason(monkeypatch, tmp_path):
    live, builds = _patch_script(monkeypatch, tmp_path, season_type="off")
    assert live.main(["--no-nflverse"]) == 0
    assert builds["n"] == 0


def test_live_main_skips_when_no_games_finished(monkeypatch, tmp_path):
    live, builds = _patch_script(monkeypatch, tmp_path, finished=())
    assert live.main(["--no-nflverse"]) == 0
    assert builds["n"] == 0


def test_live_main_builds_then_skips_until_new_final(monkeypatch, tmp_path):
    live, builds = _patch_script(monkeypatch, tmp_path, finished=("g1",))
    # First run: a game is final and nothing built today yet -> rebuild.
    assert live.main(["--no-nflverse"]) == 0
    assert builds["n"] == 1
    # Second run, same finals, same day -> no rebuild.
    assert live.main(["--no-nflverse"]) == 0
    assert builds["n"] == 1
    # A new game goes final -> rebuild again.
    monkeypatch.setattr(live, "finished_game_ids_for_week", lambda s, w: ["g1", "g2"])
    assert live.main(["--no-nflverse"]) == 0
    assert builds["n"] == 2


def test_live_main_force_rebuilds_regardless(monkeypatch, tmp_path):
    live, builds = _patch_script(monkeypatch, tmp_path, finished=("g1",))
    assert live.main(["--no-nflverse"]) == 0
    assert builds["n"] == 1
    # --force ignores the unchanged-finals guard.
    assert live.main(["--no-nflverse", "--force"]) == 0
    assert builds["n"] == 2


def _set_weekday(monkeypatch, live, weekday):
    """Freeze live.datetime.now() to a date with the given weekday (Mon=0)."""
    import datetime as _dt

    class _FakeDateTime(_dt.datetime):
        @classmethod
        def now(cls, tz=None):
            # ISO weekday is 1=Mon..7=Sun; our weekday arg is 0=Mon..6=Sun.
            return _dt.datetime.fromisocalendar(2026, 39, weekday + 1)

    monkeypatch.setattr(live, "datetime", _FakeDateTime)


def test_nflverse_runs_inside_tue_wed_window(monkeypatch, tmp_path):
    live, builds = _patch_script(monkeypatch, tmp_path, finished=("g1",))
    _set_weekday(monkeypatch, live, 1)  # Tuesday
    assert live.main([]) == 0
    assert builds.get("nflverse") == 1


def test_nflverse_skipped_outside_window(monkeypatch, tmp_path):
    live, builds = _patch_script(monkeypatch, tmp_path, finished=("g1",))
    _set_weekday(monkeypatch, live, 6)  # Sunday
    assert live.main([]) == 0
    assert builds.get("nflverse") is None  # never pulled


def test_nflverse_force_overrides_window(monkeypatch, tmp_path):
    live, builds = _patch_script(monkeypatch, tmp_path, finished=("g1",))
    _set_weekday(monkeypatch, live, 6)  # Sunday
    assert live.main(["--force"]) == 0
    assert builds.get("nflverse") == 1
