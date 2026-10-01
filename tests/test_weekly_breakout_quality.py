"""Production-quality confidence modifier (weekly-v6).

The breakout SCORE stays purely role-based; confidence additionally asks
whether the player is producing with the role, via the recent window's
per-game PPR points over expected (``ppr_over_expected`` on the weekly
rows). Positive production raises confidence, negative lowers it, and
missing production data leaves confidence EXACTLY as the pre-v6 engine
computed it (unknown is neutral, never a penalty).
"""
from data_building.breakout_engine import weekly_breakout as wb


def row(week, snap, share=0, targets=0, carries=0, routes=None,
        team_dropbacks=None, ppoe="absent"):
    out = {"week": week, "snap_pct": snap, "target_share": share,
           "targets": targets, "carries": carries, "routes": routes,
           "team_dropbacks": team_dropbacks, "ppr_pts": 5}
    if ppoe != "absent":
        out["ppr_over_expected"] = ppoe
    return out


def player(position="WR", **extra):
    return {"player_id": "fixture", "position": position, "season": 2026,
            "years_exp": 1, **extra}


def _series(ppoe_by_week=None):
    """5-game WR with a clear role jump in weeks 4-5 (the recent window)."""
    ppoe_by_week = ppoe_by_week or {}
    weeks = [
        row(1, 30, 7, 2, routes=10, team_dropbacks=35),
        row(2, 33, 8, 3, routes=11, team_dropbacks=36),
        row(3, 35, 9, 3, routes=12, team_dropbacks=36),
        row(4, 72, 22, 8, routes=28, team_dropbacks=38),
        row(5, 75, 24, 9, routes=30, team_dropbacks=39),
    ]
    for w in weeks:
        if w["week"] in ppoe_by_week:
            w["ppr_over_expected"] = ppoe_by_week[w["week"]]
    return weeks


def _quality(result):
    return result["confidence_detail"]["production_quality"]


def test_positive_production_raises_confidence_not_score():
    base = wb.score_player(player(), _series(), cutoff_week=5)
    boosted = wb.score_player(
        player(), _series({1: 0.5, 2: 1.5, 3: 3.0, 4: 3.0, 5: 3.0}),
        cutoff_week=5)
    assert boosted["confidence"] > base["confidence"]
    assert boosted["confidence"] == round(base["confidence"] + 6.0, 1)
    assert boosted["breakout_score"] == base["breakout_score"]
    q = _quality(boosted)
    assert q["available"] is True
    assert q["recent"] == 3.0          # weeks 3-5 (the recent window)
    assert q["baseline"] == 1.0        # weeks 1-2
    assert q["adjustment"] == 6.0


def test_negative_production_lowers_confidence_not_score():
    base = wb.score_player(player(), _series(), cutoff_week=5)
    dragged = wb.score_player(
        player(), _series({1: 0.0, 2: -1.0, 3: -3.0, 4: -3.0, 5: -3.0}),
        cutoff_week=5)
    assert dragged["confidence"] < base["confidence"]
    assert dragged["confidence"] == round(base["confidence"] - 6.0, 1)
    assert dragged["breakout_score"] == base["breakout_score"]
    q = _quality(dragged)
    assert q["available"] is True
    assert q["recent"] == -3.0
    assert q["adjustment"] == -6.0


def test_missing_production_data_leaves_confidence_identical():
    absent = wb.score_player(player(), _series(), cutoff_week=5)
    explicit_none = wb.score_player(
        player(), _series({1: None, 2: None, 3: None, 4: None, 5: None}),
        cutoff_week=5)
    zero_quality = wb.score_player(
        player(), _series({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}),
        cutoff_week=5)
    assert absent["confidence"] == explicit_none["confidence"]
    assert _quality(absent) == {"baseline": None, "recent": None,
                                "adjustment": 0.0, "available": False}
    assert _quality(explicit_none) == _quality(absent)
    # Production exactly at expectation is real data with a zero adjustment.
    assert _quality(zero_quality)["available"] is True
    assert _quality(zero_quality)["adjustment"] == 0.0
    assert zero_quality["confidence"] == absent["confidence"]


def test_adjustment_is_bounded_at_ten_points():
    hot = wb.score_player(
        player(), _series({4: 20.0, 5: 20.0}), cutoff_week=5)
    cold = wb.score_player(
        player(), _series({4: -20.0, 5: -20.0}), cutoff_week=5)
    assert _quality(hot)["adjustment"] == wb.QUALITY_MAX_ADJUSTMENT == 10.0
    assert _quality(cold)["adjustment"] == -wb.QUALITY_MAX_ADJUSTMENT


def test_quality_uses_the_split_windows_not_the_whole_series():
    # Huge early-season underproduction must not leak into the recent mean;
    # the baseline mean is context only.
    result = wb.score_player(
        player(), _series({1: -30.0, 2: -30.0, 3: -30.0, 4: 2.0, 5: 2.0}),
        cutoff_week=5)
    q = _quality(result)
    assert result["recent_weeks"] == [3, 4, 5]
    # split_windows(5 games): recent = weeks 3-5, baseline = weeks 1-2.
    assert q["recent"] == -8.7         # (-30 + 2 + 2) / 3
    assert q["baseline"] == -30.0
    assert q["adjustment"] == -10.0    # -8.7 * 2, clamped at the bound


def test_store_loader_maps_rows_and_fails_soft(monkeypatch):
    from data_building.breakout_engine import weekly_store

    class Result:
        def fetchall(self):
            return [
                {"player_id": "p1", "week": 1, "ppr_over_expected": 2.5},
                {"player_id": "p1", "week": 2, "ppr_over_expected": -1.0},
                {"player_id": "p2", "week": 2, "ppr_over_expected": 0.0},
            ]

    class Conn:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def execute(self, query, params):
            assert "player_weekly_advanced_metrics" in query
            assert "ppr_over_expected" in query
            assert params == (2026, 4)
            return Result()

    monkeypatch.setattr(weekly_store, "get_conn", lambda: Conn())
    out = weekly_store.load_weekly_over_expected(2026, 4)
    assert out == {"p1": {1: 2.5, 2: -1.0}, "p2": {2: 0.0}}

    class BrokenConn:
        def __enter__(self): raise RuntimeError("no table")
        def __exit__(self, *args): return False

    monkeypatch.setattr(weekly_store, "get_conn", lambda: BrokenConn())
    assert weekly_store.load_weekly_over_expected(2026, 4) == {}
