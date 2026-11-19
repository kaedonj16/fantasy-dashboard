"""API shape test for GET /api/player-role/<sleeper_id>.

Loaders are monkeypatched: the pool is hand-built weekly series and the
PBP tables come from the real 2026 slice fixture with GSIS ids mapped
onto the fake Sleeper ids used by the pool.
"""
import csv
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from dashboard_services import player_roles as pr  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "pbp_slice_2026.csv"

QB_ID, WR_ID, RB_ID = "9001", "9002", "9003"


def _row(week, **kw):
    base = {"week": week, "snap_pct": 50.0, "snaps": 30, "team_snaps": 60,
            "targets": 0, "receptions": 0, "carries": 0, "touches": 0,
            "target_share": None, "carry_share": None, "ppr_pts": 0.0,
            "rec_yards": 0.0, "rush_yards": 0.0, "pass_att": 0,
            "rec_tds": 0, "rush_tds": 0, "pass_tds": 0,
            "rz_targets": 0, "rz_carries": 0}
    base.update(kw)
    return base


def _pool():
    weeks = [1, 2, 3, 4, 5, 6]
    pool = {
        QB_ID: {"pos": "QB", "team": "DAL", "name": "Dak Testcott",
                "series": [_row(w, snap_pct=98.0, snaps=62, pass_att=36,
                                pass_tds=2, rush_yards=18.0, ppr_pts=21.0)
                           for w in weeks]},
        "9004": {"pos": "QB", "team": "NYG", "name": "Other Qb",
                 "series": [_row(w, snap_pct=97.0, snaps=60, pass_att=28,
                                 pass_tds=1, rush_yards=8.0, ppr_pts=15.0)
                            for w in weeks]},
        WR_ID: {"pos": "WR", "team": "DAL", "name": "CeeDee Test",
                "series": [_row(w, snap_pct=92.0, snaps=55, targets=10,
                                receptions=7, target_share=27.0,
                                rec_yards=95.0, ppr_pts=17.0, rz_targets=1)
                           for w in weeks]},
        "9005": {"pos": "WR", "team": "DAL", "name": "Second Receiver",
                 "series": [_row(w, snap_pct=70.0, snaps=42, targets=5,
                                 receptions=3, target_share=13.0,
                                 rec_yards=45.0, ppr_pts=8.0)
                            for w in weeks]},
        "9006": {"pos": "WR", "team": "NYG", "name": "Rival Receiver",
                 "series": [_row(w, snap_pct=80.0, snaps=48, targets=7,
                                 receptions=5, target_share=19.0,
                                 rec_yards=65.0, ppr_pts=11.0, rz_targets=1)
                            for w in weeks]},
        RB_ID: {"pos": "RB", "team": "DET", "name": "Jahmyr Test",
                "series": [_row(w, snap_pct=62.0, snaps=38, carries=13,
                                targets=5, receptions=4, carry_share=48.0,
                                target_share=14.0, rush_yards=70.0,
                                rec_yards=38.0, ppr_pts=16.0, rz_carries=2)
                           for w in weeks]},
        "9007": {"pos": "RB", "team": "DET", "name": "Other Back",
                 "series": [_row(w, snap_pct=40.0, snaps=24, carries=8,
                                 targets=2, receptions=1, carry_share=30.0,
                                 target_share=6.0, rush_yards=34.0,
                                 rec_yards=9.0, ppr_pts=7.0, rz_carries=1)
                            for w in weeks]},
    }
    return pool


def _tables():
    with open(FIXTURE, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    qb_gsis = next(r["passer_player_id"] for r in rows
                   if r["passer_player_name"] == "D.Prescott")
    wr_gsis = next(r["receiver_player_id"] for r in rows
                   if r["receiver_player_name"] == "C.Lamb")
    rb_gsis = next(r["rusher_player_id"] for r in rows
                   if r["rusher_player_name"] == "J.Gibbs")
    xwalk = {qb_gsis: QB_ID, wr_gsis: WR_ID, rb_gsis: RB_ID}
    weeks = {int(float(r["week"])) for r in rows}
    return pr.build_pbp_tables_from_rows(
        rows, xwalk, last4_weeks={max(weeks)}, base_weeks=weeks - {max(weeks)})


def _pa_rows(pid):
    """Fake FTN weekly store rows: low PA early, high PA late."""
    if pid != QB_ID:
        return []
    rows = []
    for w in range(1, 7):
        late = w >= 3
        rows.append({"week": w,
                     "play_action_rate": 30.0 if late else 10.0,
                     "play_action_epa": 0.35 if late else 0.05,
                     "w_dropbacks": 40})
    return rows


@pytest.fixture()
def client(monkeypatch):
    from flask import Flask

    import routes.player_roles_bp as bp_mod

    monkeypatch.setattr(pr, "load_pool", lambda season: _pool())
    monkeypatch.setattr(pr, "get_pbp_tables", lambda season, window: _tables())
    monkeypatch.setattr(pr, "_pa_series", lambda pid, season: _pa_rows(pid))
    bp_mod.clear_cache()
    app = Flask(__name__)
    app.register_blueprint(bp_mod.player_roles_bp)
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_wr_payload_shape(client):
    resp = client.get(f"/api/player-role/{WR_ID}?season=2026&window=season")
    assert resp.status_code == 200
    p = resp.get_json()
    for key in ("player_id", "name", "pos", "team", "season", "window",
                "weeks", "role", "timeline", "target_map", "gap_map",
                "qb_rush", "comparables", "twin"):
        assert key in p, key
    assert p["player_id"] == WR_ID and p["pos"] == "WR"
    role = p["role"]
    for key in ("job", "tags", "security", "security_note", "confidence",
                "trend", "axes", "games", "snaps"):
        assert key in role, key
    assert isinstance(role["tags"], list)
    tm = p["target_map"]
    assert tm is not None and tm["kind"] == "receiving"
    assert len(tm["zones"]) == 12
    assert [(z["row"], z["col"]) for z in tm["zones"][:3]] == [
        ("Deep", "left"), ("Deep", "middle"), ("Deep", "right")]
    z = tm["zones"][0]
    for key in ("primary", "secondary", "share_primary", "share_secondary",
                "yards", "td", "int", "first_downs", "avg_depth",
                "catch_rate", "rating", "league_catch_rate", "league_rating",
                "trend_last4", "trend_base", "trend_label"):
        assert key in z, key
    share_sum = sum(z["share_primary"] for z in tm["zones"])
    assert share_sum == pytest.approx(100.0, abs=2.0)
    for key in ("adot", "air_yards_share", "catch_rate", "ypt_or_ypa",
                "deep_rate", "comp_pct"):
        assert key in tm["summary"], key
    assert isinstance(p["timeline"], list) and p["timeline"]
    assert isinstance(p["comparables"], list)
    for c in p["comparables"]:
        for key in ("player_id", "name", "pos", "team", "similarity",
                    "match_line", "differ_line"):
            assert key in c, key


def test_qb_payload_shape(client):
    resp = client.get(f"/api/player-role/{QB_ID}?season=2026&window=season")
    assert resp.status_code == 200
    p = resp.get_json()
    assert p["target_map"] is not None
    assert p["target_map"]["kind"] == "passing"
    assert p["target_map"]["summary"]["comp_pct"] > 0
    assert "Mobility" in p["role"]["axes"]
    assert "qb_rush" in p  # may be null if the slice has no QB runs


def test_qb_play_action_matches_ftn_store(client):
    resp = client.get(f"/api/player-role/{QB_ID}?season=2026&window=season")
    assert resp.status_code == 200
    summary = resp.get_json()["target_map"]["summary"]
    # Dropback-weighted over the fake store: (10*80 + 30*160) / 240.
    assert summary["play_action_rate"] == pytest.approx(23.3)
    assert summary["play_action_epa"] == pytest.approx(0.25)


def test_qb_play_action_last4_window(client):
    resp = client.get(f"/api/player-role/{QB_ID}?season=2026&window=last4")
    assert resp.status_code == 200
    summary = resp.get_json()["target_map"]["summary"]
    # Last-4 window covers weeks 3-6 only: the high-PA weeks.
    assert summary["play_action_rate"] == pytest.approx(30.0)


def test_wr_summary_has_no_play_action(client):
    resp = client.get(f"/api/player-role/{WR_ID}?season=2026&window=season")
    assert resp.status_code == 200
    assert "play_action_rate" not in resp.get_json()["target_map"]["summary"]


def test_rb_payload_shape(client):
    resp = client.get(f"/api/player-role/{RB_ID}?season=2026&window=season")
    assert resp.status_code == 200
    p = resp.get_json()
    gm = p["gap_map"]
    assert gm is not None
    assert [lane["lane"] for lane in gm["lanes"]] == pr.LANES
    for key in ("carries", "ypc", "success_rate", "stuffed_pct",
                "explosive_pct", "rush_td"):
        assert key in gm["summary"], key
    assert gm["summary"]["carries"] > 0
    for key in ("left", "middle", "right"):
        assert key in gm["tendency"]
    lane = gm["lanes"][0]
    for key in ("share", "carries", "yards", "ypc", "td", "first_downs",
                "stuffed_pct", "explosive_pct", "longest", "league_share",
                "league_ypc", "trend_last4", "trend_base"):
        assert key in lane, key


def test_unknown_player_404(client):
    resp = client.get("/api/player-role/424242?season=2026")
    assert resp.status_code == 404


def test_bad_window_defaults_to_season(client):
    resp = client.get(f"/api/player-role/{WR_ID}?season=2026&window=bogus")
    assert resp.status_code == 200
    assert resp.get_json()["window"] == "season"
