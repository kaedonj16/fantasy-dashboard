"""Breakout player payload: per-week series for the modal's weekly charts.

The /api/breakout/player payload gains ``weekly_series`` (snap share +
targets per played week) from the weekly-metrics store, so the Breakout
tab can draw the snap-share and targets charts with trend lines. Bye
weeks have no row in the store; no-usage (DNP) weeks must never leak in.
"""
import pytest

pytest.importorskip("flask")

from dashboard_services import breakout_api


def _series_rows():
    # Weeks 1-5 played (week 3 is the bye: simply absent, like the store),
    # plus a week-6 no-usage row that must be filtered out.
    return [
        {"week": 1, "snap_pct": 86.0, "snaps": 60, "team_snaps": 70,
         "targets": 6, "carries": 0, "pass_att": 0},
        {"week": 2, "snap_pct": 86.2, "snaps": 61, "team_snaps": 71,
         "targets": 5, "carries": 1, "pass_att": 0},
        {"week": 4, "snap_pct": 92.2, "snaps": 64, "team_snaps": 69,
         "targets": 4, "carries": 0, "pass_att": 0},
        {"week": 5, "snap_pct": 88.9, "snaps": 62, "team_snaps": 70,
         "targets": 9, "carries": 0, "pass_att": 0},
        {"week": 6, "snap_pct": 0.0, "snaps": 0, "team_snaps": 70,
         "targets": 0, "carries": 0, "pass_att": 0},
    ]


def test_weekly_series_shape_and_played_weeks(monkeypatch):
    monkeypatch.setattr(
        "data_building.weekly_metrics.get_player_weekly_series",
        lambda pid, season: _series_rows(),
    )
    out = breakout_api._weekly_series_for_payload("12514", 2026)
    assert [p["week"] for p in out] == [1, 2, 4, 5]  # bye absent, DNP dropped
    # Carries/touches/pass_att ride along (touches unset in these rows ->
    # None) so the Team tab usage card can pick its metric by position.
    assert out[0] == {"week": 1, "snap_share": 86.0, "targets": 6,
                      "carries": 0, "touches": None, "pass_att": 0}
    assert out[-1] == {"week": 5, "snap_share": 88.9, "targets": 9,
                       "carries": 0, "touches": None, "pass_att": 0}
    for point in out:
        assert set(point.keys()) == {"week", "snap_share", "targets",
                                     "carries", "touches", "pass_att"}


def test_weekly_series_keeps_unknown_snap_share_none(monkeypatch):
    rows = _series_rows()
    rows[0]["snap_pct"] = None  # played (targets > 0) but no snap data
    monkeypatch.setattr(
        "data_building.weekly_metrics.get_player_weekly_series",
        lambda pid, season: rows,
    )
    out = breakout_api._weekly_series_for_payload("12514", 2026)
    assert out[0]["week"] == 1
    assert out[0]["snap_share"] is None  # unknown stays unknown, never 0
    assert out[0]["targets"] == 6


def test_weekly_series_store_failure_returns_empty(monkeypatch):
    def _boom(pid, season):
        raise RuntimeError("db down")

    monkeypatch.setattr(
        "data_building.weekly_metrics.get_player_weekly_series", _boom)
    assert breakout_api._weekly_series_for_payload("12514", 2026) == []


def test_weekly_detail_attaches_series(monkeypatch):
    monkeypatch.setattr(
        "data_building.breakout_engine.weekly_store.get_weekly_candidate",
        lambda pid, season: {
            "player_id": pid, "breakout_score": 78, "confidence": 64,
            "classification": "emerging_breakout", "evidence": {},
        },
    )
    monkeypatch.setattr(
        "data_building.weekly_metrics.get_player_weekly_series",
        lambda pid, season: _series_rows(),
    )
    monkeypatch.setattr(breakout_api, "_breakout_ranks", lambda season: {})
    detail = breakout_api.get_weekly_breakout_detail("12514", 2026)
    assert detail["available"] is True
    assert [p["week"] for p in detail["weekly_series"]] == [1, 2, 4, 5]
    assert detail["weekly_series"][1]["snap_share"] == 86.2


def test_weekly_detail_missing_candidate_has_no_series(monkeypatch):
    monkeypatch.setattr(
        "data_building.breakout_engine.weekly_store.get_weekly_candidate",
        lambda pid, season: None,
    )
    detail = breakout_api.get_weekly_breakout_detail("12514", 2026)
    assert detail["available"] is False
    assert "weekly_series" not in detail
