"""Guards trade-intel context surfacing (buyer/seller records, split, velocity).

_trade_intel_extras reads the crawler's trade_context snapshots (per-side
record at trade time) and turns them into UI-ready intel. These tests pin the
classification, sample selection, buyer split, and velocity trend with a
stubbed DB — no Flask server, no Postgres.
"""
import datetime as _dt

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConn:
    def __init__(self, row):
        self._row = row

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        return _FakeResult(self._row)


def _run(monkeypatch, meta, pkgs, sigs, result_packages, values_by_id, vel_row):
    import app
    import dashboard_services.db as db

    monkeypatch.setattr(db, "get_conn", lambda *a, **k: _FakeConn(vel_row))
    return app._trade_intel_extras(
        trade_meta=meta,
        trade_pkgs=pkgs,
        sig_counts=sigs,
        result_packages=result_packages,
        values_by_id=values_by_id,
        target_player_id="p1",
        is_sf=False,
        num_teams=12,
    )


def _meta(recv="a", buyer_rec=None, seller_rec=None, week=9, created=None):
    buyer_rec = buyer_rec or {"wins": 9, "losses": 2, "ties": 0}
    seller_rec = seller_rec or {"wins": 2, "losses": 8, "ties": 0}
    ctx = {"a": buyer_rec, "b": seller_rec} if recv == "a" else {"a": seller_rec, "b": buyer_rec}
    return {
        "recv_side": recv,
        "ctx": ctx,
        "week": week,
        "created_at": created or _dt.datetime(2026, 10, 1),
    }


def _values():
    return {"p2": {"position": "WR", "value": 900.0}}


def test_sample_classifies_buyer_and_seller(monkeypatch):
    import app

    tier = app._asset_tier(900.0)
    meta = {"t1": _meta()}
    pkgs = {"t1": [
        {"asset_type": "pick", "sent_player_id": None, "pick_round": 1, "pick_season": 2026, "pick_order": None},
        {"asset_type": "player", "sent_player_id": "p2", "pick_round": None, "pick_season": None, "pick_order": None},
    ]}
    sigs = {("K",): ["t1"]}
    result = [{"sig": ["K"], "send": []}]
    vel, split = _run(monkeypatch, meta, pkgs, sigs, result, _values(), {"recent": 0, "prior": 0})

    sample = result[0]["trade_context_sample"]
    assert sample["buyer_record"] == "9-2"
    assert sample["buyer_class"] == "contender"
    assert sample["seller_record"] == "2-8"
    assert sample["seller_class"] == "rebuilder"
    assert sample["week"] == 9
    # buyer split uses the same classification
    assert split["contender"]["total"] == 1
    assert split["contender"]["label"] == f"2026 1st + WR{tier}"
    assert "rebuilder" not in split  # seller side never buys here
    assert vel == {"recent_90d": 0, "prior_90d": 0, "trend": None}


def test_sample_picks_most_recent_trade_with_context(monkeypatch):
    meta = {
        "t_old": _meta(week=6, created=_dt.datetime(2026, 9, 1)),
        "t_new": _meta(week=9, created=_dt.datetime(2026, 10, 1)),
        "t_naked": {"recv_side": "a", "ctx": {}, "week": 3,
                    "created_at": _dt.datetime(2026, 10, 5)},
    }
    pkgs = {tid: [] for tid in meta}
    sigs = {("K",): ["t_old", "t_naked", "t_new"]}
    result = [{"sig": ["K"], "send": []}]
    _run(monkeypatch, meta, pkgs, sigs, result, {}, {"recent": 0, "prior": 0})
    # t_naked is newest but has no context; t_new wins on recency among usable
    assert result[0]["trade_context_sample"]["week"] == 9


def test_mid_pack_and_ties_classify_honestly(monkeypatch):
    meta = {"t1": _meta(
        buyer_rec={"wins": 4, "losses": 4, "ties": 0},
        seller_rec={"wins": 1, "losses": 1, "ties": 1},
        week=25,  # out of range: must be omitted, not invented
    )}
    pkgs = {"t1": []}
    sigs = {("K",): ["t1"]}
    result = [{"sig": ["K"], "send": []}]
    _run(monkeypatch, meta, pkgs, sigs, result, {}, {"recent": 0, "prior": 0})
    sample = result[0]["trade_context_sample"]
    assert sample["buyer_class"] == "mid-pack"
    assert sample["buyer_record"] == "4-4"
    assert sample["seller_record"] == "1-1-1"
    assert "week" not in sample


def test_buyer_split_top_shape_per_class(monkeypatch):
    def _pick(season, rnd):
        return {"asset_type": "pick", "sent_player_id": None,
                "pick_round": rnd, "pick_season": season, "pick_order": None}

    meta = {
        "t1": _meta(created=_dt.datetime(2026, 10, 1)),   # contender buyer
        "t2": _meta(created=_dt.datetime(2026, 9, 15)),   # contender buyer
        "t3": _meta(created=_dt.datetime(2026, 9, 1)),    # contender buyer
        # rebuilder buyer: recv side b holds the 3-9 record
        "t4": _meta(recv="b", buyer_rec={"wins": 3, "losses": 9, "ties": 0},
                    created=_dt.datetime(2026, 8, 1)),
    }
    pkgs = {
        "t1": [_pick(2026, 1), _pick(2026, 1)],
        "t2": [_pick(2026, 1), _pick(2026, 1)],
        "t3": [_pick(2027, 1)],
        "t4": [_pick(2026, 2)],
    }
    sigs = {}
    _vel, split = _run(monkeypatch, meta, pkgs, sigs, [], {}, {"recent": 0, "prior": 0})
    assert split["contender"] == {"label": "2× 2026 1st", "count": 2, "total": 3}
    assert split["rebuilder"] == {"label": "2026 2nd", "count": 1, "total": 1}


def test_velocity_trend_needs_minimum_sample(monkeypatch):
    meta, pkgs, sigs = {"t1": _meta()}, {"t1": []}, {}
    vel, _split = _run(monkeypatch, meta, pkgs, sigs, [], {}, {"recent": 11, "prior": 4})
    assert vel == {"recent_90d": 11, "prior_90d": 4, "trend": "heating_up"}
    vel, _split = _run(monkeypatch, meta, pkgs, sigs, [], {}, {"recent": 1, "prior": 5})
    assert vel["trend"] == "cooling_off"
    vel, _split = _run(monkeypatch, meta, pkgs, sigs, [], {}, {"recent": 4, "prior": 4})
    assert vel["trend"] == "steady"
    # too few trades: count shown, no trend claim
    vel, _split = _run(monkeypatch, meta, pkgs, sigs, [], {}, {"recent": 3, "prior": 0})
    assert vel == {"recent_90d": 3, "prior_90d": 0, "trend": None}
