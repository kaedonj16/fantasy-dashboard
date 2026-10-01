"""v6 reconstruction backfill: as-of overrides from nflverse-shaped rows,
runner injection, and the reconstruct_week driver rules.

The builder's output feeds the runner's OWN injury-context map and pure
scorer unchanged, so these tests drive the built feed override through
``weekly_runner._injury_context_map`` and compare against live semantics:
Out/IR/Doubtful ahead of a player opens his role; Questionable does not.
"""
from __future__ import annotations

import sys
import types
from datetime import date

import pytest

from data_building.breakout_engine import reconstruction as recon
from data_building.breakout_engine import weekly_runner as wr

AS_OF = date(2026, 9, 20)  # the original week-2 run's date
WEEK = 2

CROSSWALK = {
    "G1": "101", "G2": "102", "G3": "103", "G4": "104", "G5": "105",
    "G6": "106", "G7": "107", "G8": "108", "G9": "109", "G10": "110",
    "G11": "111",
}


def _roster(week, gsis, team, pos, name, status="ACT", years_exp="3",
            rookie_year="2024", entry_year="2024", birth_date="2001-06-15"):
    return {
        "season": "2026", "week": str(week), "team": team, "position": pos,
        "full_name": name, "gsis_id": gsis, "sleeper_id": "",
        "status": status, "years_exp": years_exp, "rookie_year": rookie_year,
        "entry_year": entry_year, "birth_date": birth_date,
    }


def _injury(week, gsis, status):
    return {"week": str(week), "gsis_id": gsis, "report_status": status}


def _depth(dt, team, gsis, pos_abb, rank, name=""):
    return {"dt": dt, "team": team, "gsis_id": gsis, "pos_abb": pos_abb,
            "pos_rank": str(rank), "player_name": name}


def _fixtures():
    rosters = [
        _roster(2, "G1", "AAA", "RB", "Sam Starter"),
        _roster(2, "G2", "AAA", "RB", "Ben Backup"),
        _roster(2, "G3", "AAA", "QB", "Quinn Questionable"),
        _roster(2, "G4", "AAA", "QB", "Bob Backup"),
        _roster(2, "G5", "BBB", "RB", "Ollie Other"),
        _roster(2, "G6", "BBB", "RB", "Omar Other"),
        _roster(2, "G7", "AAA", "WR", "Walt Wideout"),
        _roster(2, "G8", "CCC", "TE", "Ted Tight"),
        _roster(2, "G9", "CCC", "TE", "Tim Tight"),
        # Ted lands on RES in week 3: the IR designation the week-2
        # Sleeper feed would have shown after his week-2 injury.
        _roster(3, "G8", "CCC", "TE", "Ted Tight", status="RES"),
        _roster(3, "G9", "CCC", "TE", "Tim Tight"),
        _roster(2, "G10", "DDD", "WR", "Ian Ingame"),
        _roster(2, "G11", "DDD", "WR", "Ike Ingame"),
    ]
    injuries = [
        _injury(2, "G1", "Out"),
        _injury(2, "G3", "Questionable"),
        _injury(2, "G5", "Out"),
        # Ian has no week-2 designation; his week-3 report (Out) is the
        # in-game injury fallback.
        _injury(3, "G10", "Out"),
    ]
    # Two snapshots: Sep 19 (on/before the run date, used) and Sep 21
    # (after the run date, must be ignored even though it flips the
    # AAA running back order).
    depth = []
    for dt, rb_order in (("2026-09-19", ("G1", "G2")), ("2026-09-21", ("G2", "G1"))):
        depth += [
            _depth(dt, "AAA", rb_order[0], "RB", 1),
            _depth(dt, "AAA", rb_order[1], "RB", 2),
            _depth(dt, "AAA", "G3", "QB", 1),
            _depth(dt, "AAA", "G4", "QB", 2),
            _depth(dt, "AAA", "G7", "WR", 1),
            _depth(dt, "BBB", "G5", "RB", 1),
            _depth(dt, "BBB", "G6", "RB", 2),
            _depth(dt, "CCC", "G8", "TE", 1),
            _depth(dt, "CCC", "G9", "TE", 2),
            _depth(dt, "DDD", "G10", "WR", 1),
            _depth(dt, "DDD", "G11", "WR", 2),
        ]
    return rosters, injuries, depth


def _overrides():
    rosters, injuries, depth = _fixtures()
    return recon.build_asof_overrides(
        roster_rows=rosters, injury_rows=injuries, depth_rows=depth,
        gsis_to_sleeper=CROSSWALK, week=WEEK, as_of_date=AS_OF)


def test_overrides_carry_identity_feed_fields_and_asof_depth():
    index, feed = _overrides()

    assert index["102"] == {
        "name": "Ben Backup", "full_name": "Ben Backup",
        "pos": "RB", "position": "RB", "team": "AAA",
    }
    entry = feed["102"]
    assert entry["years_exp"] == 3
    assert entry["rookie_year"] == 2024
    assert entry["draft_year"] == 2024  # entry_year proxy
    assert entry["age"] == pytest.approx(25.3, abs=0.1)
    # From the Sep 19 snapshot, NOT the post-run Sep 21 flip.
    assert entry["depth_chart_order"] == 2
    assert feed["101"]["depth_chart_order"] == 1


def test_injury_context_matches_live_semantics():
    _index, feed = _overrides()
    ctx = wr._injury_context_map(feed)

    # Starter Out ahead of him: vacated, same source-string shape as live.
    assert ctx["102"] == {"vacated": True, "source": "Sam Starter (Out)"}
    # Questionable ahead: NOT an opening, exactly like the live feed.
    assert "104" not in ctx
    # A different position on the same team is unaffected.
    assert "107" not in ctx
    # Another team's injury only opens that team's chart.
    assert ctx["106"] == {"vacated": True, "source": "Ollie Other (Out)"}
    # RES on the week W+1 roster reads as IR.
    assert ctx["109"] == {"vacated": True, "source": "Ted Tight (IR)"}
    # No week-2 designation, week-3 report Out: the in-game fallback.
    assert ctx["111"] == {"vacated": True, "source": "Ian Ingame (Out)"}
    # The injured starters themselves get no context.
    for pid in ("101", "105", "108", "110"):
        assert pid not in ctx


def test_overrides_skip_unmapped_and_non_skill_rows():
    rosters, injuries, depth = _fixtures()
    rosters.append(_roster(2, "GX", "AAA", "RB", "Unmapped Player"))
    rosters.append(_roster(2, "G12", "AAA", "LS", "Long Snapper"))
    crosswalk = dict(CROSSWALK, G12="112")
    index, feed = recon.build_asof_overrides(
        roster_rows=rosters, injury_rows=injuries, depth_rows=depth,
        gsis_to_sleeper=crosswalk, week=WEEK, as_of_date=AS_OF)
    names = {entry["name"] for entry in index.values()}
    assert "Unmapped Player" not in names
    assert "Long Snapper" not in names
    assert "112" not in index
    assert set(index) == set(feed)


# ---------------------------------------------------------------------------
# runner injection: overrides replace the live inputs entirely
# ---------------------------------------------------------------------------

def _wk(week, snap=60.0, tgt=6.0, car=8.0, ppr=12.0, snaps=60):
    return {"week": week, "snap_pct": snap, "targets": tgt, "carries": car,
            "ppr_pts": ppr, "snaps": snaps}


def test_runner_uses_overrides_and_stamps_run_detail(monkeypatch):
    wm = types.ModuleType("data_building.weekly_metrics")
    series = {"1": [_wk(1, 25, 8, 2), _wk(2, 28, 9, 3),
                    _wk(3, 58, 20, 7), _wk(4, 68, 24, 8)]}
    wm.build_weekly_metrics = lambda season, weeks=None: 0
    wm.get_weekly_series_by_player = lambda season, through: series
    monkeypatch.setitem(sys.modules, "data_building.weekly_metrics", wm)
    import data_building
    monkeypatch.setattr(data_building, "weekly_metrics", wm, raising=False)

    # The live index loader must never be consulted on the override path.
    uu = types.ModuleType("utils.utils")

    def _boom():
        raise AssertionError("live players index used despite override")

    uu.load_players_index = _boom
    monkeypatch.setitem(sys.modules, "utils.utils", uu)

    from data_building.breakout_engine import weekly_store
    published = {}
    monkeypatch.setattr(
        weekly_store, "publish_weekly_snapshot",
        lambda season, week, results, **kwargs: published.update(
            season=season, week=week, results=results, **kwargs) or len(results))
    monkeypatch.setattr(weekly_store, "record_run", lambda *a, **k: None)
    monkeypatch.setattr(weekly_store, "load_previous_week_scores",
                        lambda season, before: {})
    monkeypatch.setattr(weekly_store, "load_weekly_over_expected",
                        lambda season, cutoff: {})

    ctx = wr.ScoringContext(season=2026, mode=wr.MODE_WEEKLY,
                            as_of_date=date(2026, 10, 1), cutoff_week=4,
                            completed_weeks=[1, 2, 3, 4])
    summary = wr.run_weekly_breakout(
        ctx, refresh=False,
        players_index_override={
            "1": {"name": "Test WR", "full_name": "Test WR",
                  "pos": "WR", "position": "WR", "team": "AAA"}},
        full_players_override={
            "1": {"full_name": "Test WR", "team": "AAA", "position": "WR",
                  "years_exp": 1, "rookie_year": 2025}},
        run_detail_extra={"reconstructed": True, "as_of_week": 4,
                          "injury_source": "nflverse"},
    )

    assert summary["status"] == "completed"
    assert summary["reconstructed"] is True
    assert published["week"] == 4
    assert published["detail"]["reconstructed"] is True
    assert published["detail"]["injury_source"] == "nflverse"
    assert published["detail"]["as_of_week"] == 4


# ---------------------------------------------------------------------------
# reconstruct_week driver rules
# ---------------------------------------------------------------------------

def _patch_store(monkeypatch, serving):
    from data_building.breakout_engine import weekly_store
    monkeypatch.setattr(weekly_store, "init_weekly_breakout_db", lambda: None)
    monkeypatch.setattr(weekly_store, "get_serving_run",
                        lambda season, week: serving)
    return weekly_store


def test_reconstruct_week_skips_already_current(monkeypatch):
    _patch_store(monkeypatch, {
        "id": 21, "scoring_version": "weekly-v6", "detail": {},
        "as_of_date": date(2026, 10, 4),
    })
    monkeypatch.setattr(wr, "run_weekly_breakout",
                        lambda *a, **k: pytest.fail("runner must not run"))
    out = recon.reconstruct_week(2026, 4, rows={}, crosswalk={"G": "1"})
    assert out["status"] == "already_current"


def test_reconstruct_week_skips_week_without_original(monkeypatch):
    _patch_store(monkeypatch, None)
    out = recon.reconstruct_week(2026, 3, rows={}, crosswalk={"G": "1"})
    assert out["status"] == "skipped"
    assert "no completed original run" in out["reason"]


def test_reconstruct_week_rebuilds_old_version_week(monkeypatch):
    _patch_store(monkeypatch, {
        "id": 7, "scoring_version": "weekly-v5", "detail": {},
        "as_of_date": date(2026, 9, 20),
    })
    rosters, injuries, depth = _fixtures()
    rows = {"rosters": rosters, "injuries": injuries, "depth": depth}
    captured = {}

    def _fake_run(context, **kwargs):
        captured["context"] = context
        captured["kwargs"] = kwargs
        return {"status": "completed", "records_saved": 42}

    monkeypatch.setattr(wr, "run_weekly_breakout", _fake_run)
    out = recon.reconstruct_week(2026, 2, rows=rows, crosswalk=CROSSWALK)

    assert out["status"] == "completed"
    assert out["records_saved"] == 42
    context = captured["context"]
    assert context.cutoff_week == 2
    # The reconstruction is dated with the ORIGINAL run's date.
    assert context.as_of_date == date(2026, 9, 20)
    kwargs = captured["kwargs"]
    assert kwargs["refresh"] is False
    extra = kwargs["run_detail_extra"]
    assert extra["reconstructed"] is True
    assert extra["as_of_week"] == 2
    assert extra["injury_source"] == "nflverse"
    assert extra["original_scoring_version"] == "weekly-v5"
    assert extra["original_run_id"] == 7
    assert "102" in kwargs["players_index_override"]
    assert kwargs["full_players_override"]["102"]["depth_chart_order"] == 2
