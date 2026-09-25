"""Sunday drama: flip detection, snapshot throttling, turning-point archive.

Pure tests for dashboard_services.sunday_drama plus render-level checks that
the matchup slide plays the lead-change moment on a flip and shows archived
turning points after finalization. DB access goes through a fake
list-backed connection, so no psycopg is needed.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from dashboard_services import sunday_drama as drama


# ── flip detection ────────────────────────────────────────────────────────────

def test_leader_of_ties_go_left():
    assert drama.leader_of(0.5) == "left"
    assert drama.leader_of(0.51) == "left"
    assert drama.leader_of(0.49) == "right"


def test_detect_flip_fires_on_favorite_change():
    flip = drama.detect_flip(0.68, 0.41)
    assert flip == {"from": "left", "to": "right", "delta": pytest.approx(-0.27)}


def test_detect_flip_ignores_same_leader():
    assert drama.detect_flip(0.68, 0.62) is None
    assert drama.detect_flip(0.3, 0.2) is None


def test_detect_flip_ignores_rounding_flicker():
    # 49.9 -> 50.1 crosses 0.5 but clears no deadband: no moment.
    assert drama.detect_flip(0.499, 0.501) is None


def test_detect_flip_none_inputs():
    assert drama.detect_flip(None, 0.6) is None
    assert drama.detect_flip(0.6, None) is None


def test_matchup_key_prefers_provider_id():
    m = {"matchup_id": "7", "left": {"roster_id": "3"}, "right": {"roster_id": "1"}}
    assert drama.matchup_key_for(m) == "mid:7"


def test_matchup_key_falls_back_to_sorted_pair():
    m = {"left": {"roster_id": "3"}, "right": {"roster_id": "1"}}
    assert drama.matchup_key_for(m) == "pair:1-3"
    assert drama.matchup_key_for({}) == "pair:-"
    assert drama.matchup_key_for(None) == "pair:-"


# ── fake DB ───────────────────────────────────────────────────────────────────

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    """List-backed stand-in for a psycopg connection (dict rows)."""

    def __init__(self):
        self.rows = []
        self.inserts = 0

    def execute(self, sql, params=None):
        flat = " ".join(str(sql).split())
        upper = flat.upper()
        if "CREATE TABLE" in upper or "CREATE INDEX" in upper:
            return _FakeResult([])
        if upper.startswith("SELECT") and "LIMIT 1" in upper:
            league_id, season, week, key = params
            cands = [
                r for r in self.rows
                if r["league_id"] == league_id and r["season"] == season
                and r["week"] == week and r["matchup_key"] == key
            ]
            cands.sort(key=lambda r: (str(r["observed_at"]), r["id"]), reverse=True)
            return _FakeResult(cands[:1])
        if "INSERT INTO MATCHUP_MOMENTS" in upper:
            (league_id, season, week, key, lrid, rrid, lname, rname,
             prob, lpts, rpts, gl) = params
            self.rows.append({
                "id": len(self.rows) + 1,
                "league_id": league_id, "season": season, "week": week,
                "matchup_key": key,
                "left_roster_id": lrid, "right_roster_id": rrid,
                "left_name": lname, "right_name": rname,
                "observed_at": datetime.now(timezone.utc),
                "left_win_prob": prob, "left_pts": lpts, "right_pts": rpts,
                "games_live": gl,
            })
            self.inserts += 1
            return _FakeResult([])
        if upper.startswith("SELECT"):
            league_id, season, week, key = params
            cands = [
                r for r in self.rows
                if r["league_id"] == league_id and r["season"] == season
                and r["week"] == week and r["matchup_key"] == key
            ]
            cands.sort(key=lambda r: (str(r["observed_at"]), r["id"]))
            return _FakeResult(cands)
        raise AssertionError(f"unexpected SQL: {flat[:80]}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture()
def fake_db(monkeypatch):
    conn = _FakeConn()
    drama.reset_table_cache()
    monkeypatch.setattr(drama, "_get_conn", lambda: conn)
    return conn


def _snap_kwargs(**over):
    kw = dict(
        league_id="L1", season="2026", week=3, matchup_key="mid:7",
        left_roster_id="1", right_roster_id="2",
        left_name="Team A", right_name="Team B",
        left_win_prob=0.68, left_pts=90.0, right_pts=80.0,
    )
    kw.update(over)
    return kw


def test_record_snapshot_first_row_always_records(fake_db):
    res = drama.record_snapshot(**_snap_kwargs())
    assert res is not None
    assert res["flip"] is None and res["leader"] == "left"
    assert isinstance(res["observed_at"], datetime)
    assert fake_db.inserts == 1


def test_record_snapshot_throttles_quiet_polls(fake_db):
    drama.record_snapshot(**_snap_kwargs(left_win_prob=0.68))
    # Same leader, seconds later, tiny move: throttled.
    res = drama.record_snapshot(**_snap_kwargs(left_win_prob=0.685))
    assert res is None
    assert fake_db.inserts == 1


def test_record_snapshot_records_meaningful_move_after_gap(fake_db):
    drama.record_snapshot(**_snap_kwargs(left_win_prob=0.68))
    # Age the stored row past the throttle gap, then move 3pp: records.
    fake_db.rows[0]["observed_at"] -= timedelta(seconds=601)
    res = drama.record_snapshot(**_snap_kwargs(left_win_prob=0.65))
    assert res is not None and res["flip"] is None
    assert fake_db.inserts == 2


def test_record_snapshot_flip_always_records(fake_db):
    drama.record_snapshot(**_snap_kwargs(left_win_prob=0.68))
    res = drama.record_snapshot(**_snap_kwargs(left_win_prob=0.41))
    assert res is not None
    assert res["flip"] == {"from": "left", "to": "right", "delta": pytest.approx(-0.27)}
    assert res["leader"] == "right"
    assert fake_db.inserts == 2


def test_record_snapshot_never_raises(monkeypatch):
    drama.reset_table_cache()

    def _boom():
        raise RuntimeError("db down")

    monkeypatch.setattr(drama, "_get_conn", _boom)
    assert drama.record_snapshot(**_snap_kwargs()) is None
    assert drama.get_turning_points("L1", "2026", 3, "mid:7") == []


def _seed_row(fake_db, prob, minutes_ago, lpts=90.0, rpts=80.0):
    drama.record_snapshot(**_snap_kwargs(left_win_prob=prob, left_pts=lpts, right_pts=rpts))
    fake_db.rows[-1]["observed_at"] = datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)


def test_get_turning_points_ranks_biggest_swings(fake_db):
    _seed_row(fake_db, 0.60, 50)
    _seed_row(fake_db, 0.38, 40)   # -0.22 swing
    _seed_row(fake_db, 0.40, 30)   # +0.02 noise, filtered
    _seed_row(fake_db, 0.75, 20)   # +0.35 swing
    _seed_row(fake_db, 0.70, 10)   # -0.05 swing
    points = drama.get_turning_points("L1", "2026", 3, "mid:7", limit=4)
    # Biggest swings kept; chronological for display.
    assert [round(p["swing"], 2) for p in points] == [-0.22, 0.35, -0.05]
    assert [p["at"] for p in points] == sorted(p["at"] for p in points)
    assert points[0]["to"] == "right"
    assert points[1]["to"] == "left"


def test_get_turning_points_respects_limit(fake_db):
    for i, prob in enumerate([0.6, 0.3, 0.7, 0.4, 0.8]):
        _seed_row(fake_db, prob, 50 - i * 10)
    assert len(drama.get_turning_points("L1", "2026", 3, "mid:7", limit=2)) == 2


def test_get_turning_points_empty_without_rows(fake_db):
    assert drama.get_turning_points("L1", "2026", 3, "mid:7") == []


# ── copy / html ───────────────────────────────────────────────────────────────

def test_flip_banner_copy():
    out = drama.flip_banner_html(
        "Caleb's Casting Couch", 68,
        datetime(2026, 9, 27, 19, 42, tzinfo=timezone.utc),
    )
    assert "Lead change" in out
    assert "Caleb&#x27;s Casting Couch now favored at 68%" in out
    assert "Sun 3:42 PM" in out  # ET rendering of 19:42 UTC
    assert "data-br-moment=\"leadchange\"" in out
    assert "\u2014" not in out


def test_flip_banner_without_timestamp_still_renders():
    out = drama.flip_banner_html("Team A", 51)
    assert "Lead change" in out
    assert "m-drama-flip-time" not in out


def test_flip_banner_escapes_names():
    out = drama.flip_banner_html("<script>alert(1)</script>", 55)
    assert "<script>" not in out


def test_turning_points_copy_verbs():
    pts = [
        {"at": datetime(2026, 9, 27, 17, 24, tzinfo=timezone.utc),
         "swing": 0.22, "before": 0.41, "after": 0.63, "to": "left",
         "left_name": "Team A", "right_name": "Team B",
         "left_pts": 88.2, "right_pts": 104.6},
        {"at": datetime(2026, 9, 27, 18, 5, tzinfo=timezone.utc),
         "swing": 0.10, "before": 0.63, "after": 0.73, "to": "left",
         "left_name": "Team A", "right_name": "Team B",
         "left_pts": 95.0, "right_pts": 104.6},
        {"at": datetime(2026, 9, 27, 19, 40, tzinfo=timezone.utc),
         "swing": -0.08, "before": 0.73, "after": 0.65, "to": "left",
         "left_name": "Team A", "right_name": "Team B",
         "left_pts": 101.3, "right_pts": 110.2},
    ]
    out = drama.turning_points_html(pts)
    assert "Turning points" in out
    assert "took the lead" in out
    assert "pulled away" in out
    assert "fell back" in out
    assert "(41% to 63%)" in out
    assert "Sun 1:24 PM" in out  # ET rendering of 17:24 UTC
    assert "88.2 to 104.6" in out
    assert "\u2014" not in out


def test_format_moment_time():
    assert drama.format_moment_time("nope") == ""
    naive = datetime(2026, 9, 27, 17, 24)  # naive treated as UTC
    assert drama.format_moment_time(naive) == "Sun 1:24 PM"


# ── render-level ──────────────────────────────────────────────────────────────

def _slide_modules():
    pytest.importorskip("flask")
    pytest.importorskip("requests")
    from dashboard_services import matchups as mmod
    return mmod


def _live_fixtures(monkeypatch, m):
    monkeypatch.setattr(
        "dashboard_services.api.get_nfl_state",
        lambda: {"season": 2026, "season_type": "reg"},
    )
    monkeypatch.setattr(m, "load_week_stats", lambda *a, **k: {})
    monkeypatch.setattr(m, "load_week_schedule", lambda *a, **k: [])
    monkeypatch.setattr(m, "load_teams_index", lambda: {})
    monkeypatch.setattr(m, "build_offense_rankings", lambda *a: {})
    monkeypatch.setattr(m, "get_nfl_scores_for_date", lambda *a: None)
    monkeypatch.setattr("utils.utils.load_week_projection", lambda *a, **k: {})


def _drama_matchup():
    half = {"gameStatusCode": "1", "lineScore": {"period": "2"}, "gameClock": "0:00"}
    return {
        "matchup_id": "7",
        "left": {
            "roster_id": "1", "name": "Team A", "avatar": "", "record": "1-0",
            "pts_total": 40.0, "proj_total": 110.0,
            "starters": [{"pid": "p1", "name": "QB One", "pos": "QB",
                          "pts": 12.0, "nfl": "KC"}],
        },
        "right": {
            "roster_id": "2", "name": "Team B", "avatar": "", "record": "0-1",
            "pts_total": 30.0, "proj_total": 100.0,
            "starters": [{"pid": "p2", "name": "QB Two", "pos": "QB",
                          "pts": 9.0, "nfl": "BUF"}],
        },
        "h2h": {},
    }, half


def test_slide_plays_lead_change_moment_on_flip(monkeypatch):
    m = _slide_modules()
    _live_fixtures(monkeypatch, m)
    matchup, half = _drama_matchup()

    def _fake_record(**kwargs):
        assert kwargs["league_id"] == "L1"
        assert kwargs["matchup_key"] == "mid:7"
        return {"flip": {"from": "right", "to": "left", "delta": 0.2},
                "leader": "left",
                "observed_at": datetime.now(timezone.utc)}

    monkeypatch.setattr(drama, "record_snapshot", _fake_record)
    html_out = m.render_matchup_slide(
        "2026", matchup, w=3, proj_week=2,
        status_by_pid={"p1": m.STATUS_IN_PROGRESS, "p2": m.STATUS_IN_PROGRESS},
        projections={3: {"projections": {"p1": 22.0, "p2": 20.0}}},
        players={}, teams={},
        team_game_lookup={"KC": half, "BUF": half},
        league_id="L1",
    )
    assert "m-drama-flip" in html_out
    assert "Lead change" in html_out
    assert "Team A now favored" in html_out
    assert "data-br-moment=\"leadchange\"" in html_out


def test_slide_skips_drama_without_league_id(monkeypatch):
    m = _slide_modules()
    _live_fixtures(monkeypatch, m)
    matchup, half = _drama_matchup()

    def _boom(**kwargs):
        raise AssertionError("record_snapshot must not run without league_id")

    monkeypatch.setattr(drama, "record_snapshot", _boom)
    html_out = m.render_matchup_slide(
        "2026", matchup, w=3, proj_week=2,
        status_by_pid={"p1": m.STATUS_IN_PROGRESS, "p2": m.STATUS_IN_PROGRESS},
        projections={3: {"projections": {"p1": 22.0, "p2": 20.0}}},
        players={}, teams={},
        team_game_lookup={"KC": half, "BUF": half},
    )
    assert "m-drama-flip" not in html_out


def test_slide_shows_archived_turning_points_after_final(monkeypatch):
    m = _slide_modules()
    _live_fixtures(monkeypatch, m)
    matchup, _half = _drama_matchup()

    def _fake_points(league_id, season, week, matchup_key, limit=4):
        assert (league_id, season, week, matchup_key) == ("L1", "2026", 2, "mid:7")
        return [{
            "at": datetime(2026, 9, 20, 17, 24, tzinfo=timezone.utc),
            "swing": 0.22, "before": 0.41, "after": 0.63, "to": "left",
            "left_name": "Team A", "right_name": "Team B",
            "left_pts": 88.2, "right_pts": 104.6,
        }]

    monkeypatch.setattr(drama, "get_turning_points", _fake_points)
    html_out = m.render_matchup_slide(
        "2026", matchup, w=2, proj_week=2,
        status_by_pid={},
        projections={2: {"projections": {}}},
        players={}, teams={},
        team_game_lookup={},
        league_id="L1",
    )
    assert "Turning points" in html_out
    assert "took the lead" in html_out
    assert "m-win-bar" not in html_out  # completed weeks have no win bar


def test_completed_slide_without_archive_stays_clean(monkeypatch):
    m = _slide_modules()
    _live_fixtures(monkeypatch, m)
    matchup, _half = _drama_matchup()
    monkeypatch.setattr(drama, "get_turning_points", lambda *a, **k: [])
    html_out = m.render_matchup_slide(
        "2026", matchup, w=2, proj_week=2,
        status_by_pid={},
        projections={2: {"projections": {}}},
        players={}, teams={},
        team_game_lookup={},
        league_id="L1",
    )
    assert "m-drama-history" not in html_out
