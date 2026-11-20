"""Production regression tests for /api/player-role (PR #3193 follow-up).

The shipped endpoint OOM-killed the web worker in production: the
first cold request downloaded the full ~372-column nflverse PBP
parquet on the request path, materialised it with to_dict("records")
and copied every row three times, peaking at ~1.65 GB RSS for a
partial-season frame; concurrent first-hits each launched their own
build. These tests pin the fixes:

* parquet reads are pruned to player_roles._PBP_COLUMNS and the
  pruned build produces aggregates identical to the full-width build;
* get_pbp_tables builds at most once under concurrent first-hits;
* an expired tables-cache entry rebuilds instead of returning None;
* the route's error bodies carry a machine-readable code.
"""
import csv
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

import pandas as pd  # noqa: E402

from dashboard_services import player_roles as pr  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "pbp_slice_2026.csv"


def _rows():
    with open(FIXTURE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _league_sums(tables):
    return {
        "recv_primary": sum(c["primary"] for c in tables["receiving"]["league"].values()),
        "pass_primary": sum(c["primary"] for c in tables["passing"]["league"].values()),
        "rush_carries": sum(c["carries"] for c in tables["rushing"]["league"].values()),
    }


@pytest.fixture(autouse=True)
def _clear_tables_cache():
    with pr._CACHE_LOCK:
        pr._PBP_TABLES_CACHE.clear()
    yield
    with pr._CACHE_LOCK:
        pr._PBP_TABLES_CACHE.clear()


def test_pruned_build_matches_full_width_build():
    rows = _rows()
    weeks = {int(float(r["week"])) for r in rows}
    kw = dict(last4_weeks={max(weeks)}, base_weeks=weeks - {max(weeks)},
              window_weeks=weeks)
    full = pr.build_pbp_tables_from_rows(rows, {}, **kw)
    # The pandas entry point is the production path: it must prune the
    # frame internally and still produce identical aggregates.
    via_frame = pr.build_pbp_tables(pd.DataFrame(rows), {}, **kw)
    assert _league_sums(via_frame) == _league_sums(full)
    assert via_frame["receiving"]["totals"] == full["receiving"]["totals"]
    assert via_frame["rushing"]["lanes"] == full["rushing"]["lanes"]


def test_accumulators_read_only_pruned_columns():
    rows = _rows()
    weeks = {int(float(r["week"])) for r in rows}
    kw = dict(last4_weeks={max(weeks)}, base_weeks=weeks - {max(weeks)},
              window_weeks=weeks)
    pruned_rows = [{k: r.get(k) for k in pr._PBP_COLUMNS} for r in rows]
    full = pr.build_pbp_tables_from_rows(rows, {}, **kw)
    pruned = pr.build_pbp_tables_from_rows(pruned_rows, {}, **kw)
    assert _league_sums(pruned) == _league_sums(full)
    assert pruned["receiving"]["totals"] == full["receiving"]["totals"]


def test_load_pbp_prunes_columns_and_filters_reg(tmp_path, monkeypatch):
    rows = _rows()
    df = pd.DataFrame(rows)
    df["season_type"] = "REG"  # the slice extraction dropped this column
    assert len(df.columns) > len(pr._PBP_COLUMNS)  # fixture is full-width
    path = tmp_path / "play_by_play_2099.parquet"
    df.to_parquet(path)
    monkeypatch.setattr(pr, "_PBP_CACHE_DIR", str(tmp_path))
    out = pr.load_pbp(2099)
    assert out is not None and len(out) > 0
    assert set(out.columns) <= set(pr._PBP_COLUMNS)
    assert set(out["season_type"].astype(str).str.upper()) == {"REG"}


def _tiny_frame():
    rows = [r for r in _rows() if r["play_type"] in ("pass", "run")][:50]
    return pd.DataFrame(rows)


def test_get_pbp_tables_builds_once_under_concurrency(monkeypatch):
    calls = {"n": 0}

    def slow_load(season):
        calls["n"] += 1
        time.sleep(0.3)
        return _tiny_frame()

    monkeypatch.setattr(pr, "load_pbp", slow_load)
    monkeypatch.setattr(pr, "get_gsis_crosswalk", lambda: {})
    results, errors = [], []

    def hit():
        try:
            results.append(pr.get_pbp_tables(2026, "season"))
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=hit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert calls["n"] == 1
    assert len(results) == 8
    assert all(r is results[0] for r in results)


def test_get_pbp_tables_rebuilds_after_ttl_expiry(monkeypatch):
    calls = {"n": 0}

    def load(season):
        calls["n"] += 1
        return _tiny_frame()

    monkeypatch.setattr(pr, "load_pbp", load)
    monkeypatch.setattr(pr, "get_gsis_crosswalk", lambda: {})
    with pr._CACHE_LOCK:
        pr._PBP_TABLES_CACHE[(2026, "season")] = (0.0, {"stale": True})
    tables = pr.get_pbp_tables(2026, "season")
    assert tables is not None and "stale" not in tables
    assert calls["n"] == 1


def _series_row(week, **kw):
    base = {"week": week, "snap_pct": 60.0, "snaps": 36, "team_snaps": 60,
            "targets": 0, "receptions": 0, "carries": 0, "touches": 0,
            "target_share": None, "carry_share": None, "ppr_pts": 0.0,
            "rec_yards": 0.0, "rush_yards": 0.0, "pass_att": 0,
            "rec_tds": 0, "rush_tds": 0, "pass_tds": 0,
            "rz_targets": 0, "rz_carries": 0}
    base.update(kw)
    return base


def test_timeline_matches_full_pool_brute_force():
    # build_timeline now restricts its per-week compute_pool calls to
    # the target's position (a player's job depends only on
    # same-position peers). Pin that the segments are identical to the
    # old every-position brute force.
    def wr(pid, team, per_week):
        return {"pos": "WR", "team": team, "name": pid,
                "series": [_series_row(w, targets=t, receptions=t // 2,
                                       target_share=ts, snap_pct=sp)
                           for w, (t, ts, sp) in enumerate(per_week, start=1)]}

    pool = {
        "w1": wr("w1", "DAL", [(10, 25.0, 90.0)] * 3 + [(4, 10.0, 55.0)] * 3),
        "w2": wr("w2", "DAL", [(8, 20.0, 80.0)] * 6),
        "w3": wr("w3", "NYJ", [(6, 15.0, 70.0)] * 6),
        "r1": {"pos": "RB", "team": "DAL", "name": "r1",
               "series": [_series_row(w, carries=15, target_share=5.0,
                                      carry_share=50.0) for w in range(1, 7)]},
        "q1": {"pos": "QB", "team": "DAL", "name": "q1",
               "series": [_series_row(w, pass_att=35) for w in range(1, 7)]},
    }
    got = pr.build_timeline("w1", pool, {})

    weeks = list(range(1, 7))
    segments, prev_job, prev_share = [], None, None
    for w in weeks:
        sub_pool = {}
        for p, e in pool.items():
            sub = [r for r in e["series"] if int(r["week"]) <= w]
            if sub:
                sub_pool[p] = {**e, "series": sub}
        comp = pr.compute_pool(sub_pool, {})
        job = pr.job_for("WR", comp["w1"]["score"], comp["w1"]["feats"])
        share = comp["w1"]["feats"]["opp_share"]
        if job != prev_job:
            reason = None
            if prev_job is not None:
                if prev_share is not None and share > prev_share + 2:
                    reason = f"opportunity share jumped after Week {w - 1}"
                elif prev_share is not None and share < prev_share - 2:
                    reason = f"opportunity share fell after Week {w - 1}"
                else:
                    reason = f"role changed after Week {w - 1}"
            segments.append({"from_week": w, "to_week": w, "job": job,
                             "reason": reason})
            prev_job = job
        else:
            segments[-1]["to_week"] = w
        prev_share = share
    assert got == segments
    assert len(got) >= 2  # the fixture's share drop must flip the job


def _client(monkeypatch, build):
    from routes.player_roles_bp import player_roles_bp
    from flask import Flask

    monkeypatch.setattr(pr, "build_player_payload", build)
    app = Flask(__name__)
    app.register_blueprint(player_roles_bp)
    return app.test_client()


def test_route_error_body_carries_code(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("kaboom")

    client = _client(monkeypatch, boom)
    resp = client.get("/api/player-role/6786?window=season")
    assert resp.status_code == 500
    body = resp.get_json()
    assert body["code"] == "role_build_failed"
    assert body["error"]


def test_route_no_data_body_carries_code(monkeypatch):
    client = _client(monkeypatch, lambda *a, **k: None)
    resp = client.get("/api/player-role/6786?window=season")
    assert resp.status_code == 404
    assert resp.get_json()["code"] == "no_role_data"
