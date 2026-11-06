"""Last 2 / Last 4 as per-player last games on Advanced Metrics.

The two quick-range shortcuts used to slice the last N calendar weeks,
so a player whose team had a bye (or who missed a game) silently got a
smaller sample than everyone else. They now send ``last_n_games`` to the
leaderboard API and every weekly builder family aggregates each player
over his own N most recent game weeks. Custom ranges keep the exact
calendar week_start/week_end path.
"""
import contextlib
import re
import shutil
import sqlite3
import subprocess

import pytest

from dashboard_services.pages.advanced_metrics_page import build_advanced_metrics_body
from data_building.advanced_metrics import LEADERBOARD_METRICS

SEASON = 2026


# ── SQLite-backed fake conn for the SQL builders ─────────────────────
# The builders speak psycopg (%s params, ::float casts); translating both
# lets the real generated SQL run against an in-memory SQLite database so
# the tests assert aggregated VALUES, not SQL strings.

class _SqliteConn:
    def __init__(self, db):
        self._db = db
        self._cur = None

    def execute(self, sql, params=()):
        sql = sql.replace("%s", "?").replace("::float", "")
        self._cur = self._db.execute(sql, tuple(params))
        return self

    def fetchall(self):
        return self._cur.fetchall()

    def fetchone(self):
        return self._cur.fetchone()


def _db():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    return db


def _patch_conn(monkeypatch, metrics, db):
    @contextlib.contextmanager
    def conn():
        yield _SqliteConn(db)

    monkeypatch.setattr(metrics, "get_conn", conn)
    monkeypatch.setattr(metrics, "init_weekly_advanced_metrics_db", lambda: None)


def _patch_index(monkeypatch):
    import utils.utils as uu
    monkeypatch.setattr(uu, "load_players_index", lambda: {
        "1001": {"name": "Player A", "team": "KC"},
        "1002": {"name": "Player B", "team": "BUF"},
    })


def _by_id(rows):
    return {r["player_id"]: r for r in rows}


# ── Usage builder (player_weekly_metrics) ────────────────────────────

def _usage_db():
    db = _db()
    db.execute(
        "CREATE TABLE player_weekly_metrics (player_id TEXT, season INT,"
        " week INT, position TEXT, targets REAL, receptions REAL,"
        " carries REAL, snaps REAL, touches REAL)"
    )
    # Player A: bye in week 8 (no row). Targets fall through the season.
    a_targets = {1: 10, 2: 10, 3: 10, 4: 10, 5: 8, 6: 6, 7: 4, 9: 2}
    for wk, tgt in a_targets.items():
        db.execute("INSERT INTO player_weekly_metrics VALUES"
                   " ('1001', ?, ?, 'RB', ?, 0, 0, 0, 0)", (SEASON, wk, tgt))
    # Player B: only two games all season, weeks 8 and 9.
    for wk, tgt in ((8, 12), (9, 14)):
        db.execute("INSERT INTO player_weekly_metrics VALUES"
                   " ('1002', ?, ?, 'WR', ?, 0, 0, 0, 0)", (SEASON, wk, tgt))
    return db


def test_usage_last_n_games_reaches_past_the_bye(monkeypatch):
    import data_building.advanced_metrics as metrics
    _patch_conn(monkeypatch, metrics, _usage_db())
    _patch_index(monkeypatch)

    rows = _by_id(metrics.get_weekly_range_leaderboard(
        "targets_per_game", season=SEASON, last_n_games=4))
    # A's last 4 game weeks are 5, 6, 7, 9 (bye week 8 skipped): full
    # sample of 4 games, avg (8+6+4+2)/4.
    assert rows["1001"]["games"] == 4
    assert rows["1001"]["value"] == pytest.approx(5.0)
    # B has only 2 games total: he contributes what he has.
    assert rows["1002"]["games"] == 2
    assert rows["1002"]["value"] == pytest.approx(13.0)

    rows2 = _by_id(metrics.get_weekly_range_leaderboard(
        "targets_per_game", season=SEASON, last_n_games=2))
    # A's last 2 game weeks are 7 and 9.
    assert rows2["1001"]["games"] == 2
    assert rows2["1001"]["value"] == pytest.approx(3.0)


def test_usage_calendar_range_unchanged(monkeypatch):
    import data_building.advanced_metrics as metrics
    _patch_conn(monkeypatch, metrics, _usage_db())
    _patch_index(monkeypatch)

    rows = _by_id(metrics.get_weekly_range_leaderboard(
        "targets_per_game", season=SEASON, week_start=6, week_end=9))
    # Calendar weeks 6-9 hold only 3 of A's games (6, 7, 9): the old
    # behavior, preserved for Custom ranges.
    assert rows["1001"]["games"] == 3
    assert rows["1001"]["value"] == pytest.approx(4.0)


# ── Adv-weekly builder (player_weekly_advanced_metrics, totals) ──────

def _adv_db():
    db = _db()
    db.execute(
        "CREATE TABLE player_weekly_advanced_metrics (player_id TEXT,"
        " season INT, week INT, position TEXT, receiving_epa REAL,"
        " w_targets REAL, w_carries REAL, w_receptions REAL,"
        " w_dropbacks REAL, w_pass_att REAL)"
    )
    # Player A: bye week 8; receiving_epa 1..4 over weeks 5, 6, 7, 9.
    for wk, epa in ((5, 1.0), (6, 2.0), (7, 3.0), (9, 4.0)):
        db.execute("INSERT INTO player_weekly_advanced_metrics VALUES"
                   " ('1001', ?, ?, 'WR', ?, 10, 0, 0, 0, 0)", (SEASON, wk, epa))
    return db


def test_adv_weekly_last_n_games(monkeypatch):
    import data_building.advanced_metrics as metrics
    _patch_conn(monkeypatch, metrics, _adv_db())
    _patch_index(monkeypatch)

    rows = _by_id(metrics.get_adv_weekly_range_leaderboard(
        "receiving_epa", season=SEASON, last_n_games=4))
    assert rows["1001"]["games"] == 4
    assert rows["1001"]["value"] == pytest.approx(10.0)

    cal = _by_id(metrics.get_adv_weekly_range_leaderboard(
        "receiving_epa", season=SEASON, week_start=6, week_end=9))
    assert cal["1001"]["games"] == 3
    assert cal["1001"]["value"] == pytest.approx(9.0)


def test_adv_weekly_dispatch_passes_last_n_to_sep_and_derived(monkeypatch):
    import data_building.advanced_metrics as metrics
    seen = {}
    monkeypatch.setattr(
        metrics, "get_sep_score_range_leaderboard",
        lambda **kw: seen.setdefault("sep", kw) or [])
    monkeypatch.setattr(
        metrics, "_get_derived_per_game_leaderboard",
        lambda *a, **kw: seen.setdefault("derived", kw) or [])

    metrics.get_adv_weekly_range_leaderboard(
        "ngs_avg_sep_score", season=SEASON, last_n_games=4)
    assert seen["sep"]["last_n_games"] == 4
    derived_key = next(iter(metrics._ADV_WEEKLY_DERIVED_METRICS))
    metrics.get_adv_weekly_range_leaderboard(
        derived_key, season=SEASON, last_n_games=2)
    assert seen["derived"]["last_n_games"] == 2


# ── Sep-score pool (last-N variant of the range pool) ────────────────

def _sep_db():
    db = _db()
    db.execute(
        "CREATE TABLE player_weekly_advanced_metrics (player_id TEXT,"
        " season INT, week INT, position TEXT, ngs_avg_separation REAL,"
        " w_targets REAL)"
    )
    for wk, sep in ((5, 2.0), (6, 3.0), (7, 4.0), (9, 5.0)):
        db.execute("INSERT INTO player_weekly_advanced_metrics VALUES"
                   " ('1001', ?, ?, 'WR', ?, 10)", (SEASON, wk, sep))
    return db


def test_sep_last_n_pool_aggregates_last_game_weeks(monkeypatch):
    import data_building.advanced_metrics as metrics
    _patch_conn(monkeypatch, metrics, _sep_db())

    pool = metrics._sep_last_n_pool(SEASON, 4)
    assert len(pool) == 1
    row = pool[0]
    # All 4 game weeks (5, 6, 7, 9): target-weighted sep (20+30+40+50)/40.
    assert row["weeks"] == 4
    assert row["total_targets"] == 40
    assert row["ngs_avg_separation"] == pytest.approx(3.5)

    cal = metrics._sep_range_pool(SEASON, 6, 9)
    assert cal[0]["weeks"] == 3
    assert cal[0]["ngs_avg_separation"] == pytest.approx(4.0)


# ── Derived per-game builder (xTD/G): denominator is games played ───

def _derived_db():
    db = _db()
    db.execute(
        "CREATE TABLE player_weekly_metrics (player_id TEXT, season INT,"
        " week INT, position TEXT, targets REAL, receptions REAL,"
        " carries REAL, snaps REAL, touches REAL)"
    )
    db.execute(
        "CREATE TABLE player_weekly_advanced_metrics (player_id TEXT,"
        " season INT, week INT, position TEXT, expected_tds REAL,"
        " w_targets REAL, w_carries REAL, w_receptions REAL,"
        " w_dropbacks REAL, w_pass_att REAL)"
    )
    for wk in (5, 6, 7, 9):  # games played (bye week 8)
        db.execute("INSERT INTO player_weekly_metrics VALUES"
                   " ('1001', ?, ?, 'RB', 0, 0, 0, 0, 0)", (SEASON, wk))
    for wk, xtd in ((5, 0.5), (6, 0.5), (7, 0.5), (9, 1.5)):
        db.execute("INSERT INTO player_weekly_advanced_metrics VALUES"
                   " ('1001', ?, ?, 'RB', ?, 0, 0, 0, 0, 0)", (SEASON, wk, xtd))
    return db


def test_derived_per_game_last_n_games(monkeypatch):
    import data_building.advanced_metrics as metrics
    _patch_conn(monkeypatch, metrics, _derived_db())
    _patch_index(monkeypatch)

    rows = _by_id(metrics.get_adv_weekly_range_leaderboard(
        "xtd_per_game", season=SEASON, last_n_games=4))
    # Numerator over his last 4 game weeks (5, 6, 7, 9): 3.0 over 4 games.
    assert rows["1001"]["games"] == 4
    assert rows["1001"]["value"] == pytest.approx(0.75)

    cal = _by_id(metrics.get_adv_weekly_range_leaderboard(
        "xtd_per_game", season=SEASON, week_start=6, week_end=9))
    assert cal["1001"]["games"] == 3
    assert cal["1001"]["value"] == pytest.approx(2.5 / 3)


# ── Situational shares (play-by-play, per-player windows) ────────────

def _pbp_frame():
    import pandas as pd
    rows = []

    def rush(pid, name, week, yardline=50):
        rows.append({
            "game_id": f"g{week}", "play_id": len(rows), "week": week,
            "season_type": "REG", "posteam": "KC", "defteam": "DEN",
            "down": 1, "ydstogo": 10, "yardline_100": yardline,
            "qtr": 2, "game_half": "First Half",
            "game_seconds_remaining": 2000, "play_type": "run",
            "rusher_player_id": pid, "rusher_player_name": name,
            "receiver_player_id": None, "receiver_player_name": None,
            "rush_attempt": 1, "pass_attempt": 0, "complete_pass": 0,
        })

    # A touches the ball weeks 1, 2, 3, 5 (bye week 4), one goalline rush
    # in week 1. B plays every week 1-5. Four rushes each per week.
    for wk in (1, 2, 3, 5):
        for i in range(4):
            rush("00-001", "A.Back", wk, yardline=5 if (wk == 1 and i == 0) else 50)
    for wk in (1, 2, 3, 4, 5):
        for _ in range(4):
            rush("00-002", "B.Back", wk)
    return pd.DataFrame(rows)


def test_situational_last_n_games(monkeypatch):
    pytest.importorskip("pandas")
    import utils.rb_usage as rb

    monkeypatch.setattr(rb, "_load_pbp", lambda season: _pbp_frame())
    monkeypatch.setattr(rb, "_get_rb_gsis_ids_by_team", lambda season: {})

    shares = rb.get_all_player_situational_shares(
        SEASON, rush_only=True, last_n_games=4)
    a = shares["00-001"]
    # A's last 4 game weeks are 1, 2, 3, 5 despite the week-4 bye; his
    # week-1 goalline rush is inside the window and he owns it outright.
    assert a["games"] == 4
    assert a["goalline_share"] == pytest.approx(100.0)
    assert a["early_down_share"] == pytest.approx(50.0)
    assert shares["00-002"]["games"] == 4

    last2 = rb.get_all_player_situational_shares(
        SEASON, rush_only=True, last_n_games=2)
    # A's last 2 game weeks are 3 and 5: the goalline rush falls out.
    assert last2["00-001"]["games"] == 2
    assert last2["00-001"]["goalline_share"] == pytest.approx(0.0)

    cal = rb.get_all_player_situational_shares(
        SEASON, week_start=4, week_end=5, rush_only=True)
    assert "games" not in cal["00-001"]


# ── Route: param plumbing, payload, cache key ────────────────────────

class _Policy:
    games_min = 4
    completed_weeks = (1, 2, 3, 4)
    provisional = False

    def minimum(self, col, full):
        return full

    def note(self):
        return None


def _route_client(monkeypatch, captured):
    pytest.importorskip("flask")
    from flask import Flask
    import data_building.advanced_metrics as metrics
    import routes.advanced_metrics_bp as route
    import utils.season_qualification as sq

    def fake_builder(*a, **kw):
        captured.append(kw)
        return []

    monkeypatch.setattr(metrics, "get_weekly_range_leaderboard", fake_builder)
    monkeypatch.setattr(metrics, "get_metric_leaderboard", fake_builder)
    monkeypatch.setattr(sq, "qualification_policy", lambda *a, **kw: _Policy())
    monkeypatch.setattr(route, "get_players_global", lambda: {})
    route._LB_CACHE.clear()
    app = Flask(__name__)
    app.register_blueprint(route.advanced_metrics_bp)
    return app.test_client()


def test_route_last_n_games_reaches_builder_and_payload(monkeypatch):
    captured = []
    client = _route_client(monkeypatch, captured)
    payload = client.get(
        "/api/advanced-metrics/leaderboard?metric=snap_share"
        f"&season={SEASON}&last_n_games=4").get_json()
    assert payload["is_week_filtered"] is True
    assert payload["last_n_games"] == 4
    assert payload["week_start"] is None and payload["week_end"] is None
    assert captured and captured[0]["last_n_games"] == 4


def test_route_custom_range_still_calendar(monkeypatch):
    captured = []
    client = _route_client(monkeypatch, captured)
    payload = client.get(
        "/api/advanced-metrics/leaderboard?metric=snap_share"
        f"&season={SEASON}&week_start=6&week_end=9").get_json()
    assert payload["is_week_filtered"] is True
    assert payload["last_n_games"] is None
    assert (payload["week_start"], payload["week_end"]) == (6, 9)
    assert captured[0]["last_n_games"] is None
    assert captured[0]["week_start"] == 6 and captured[0]["week_end"] == 9


def test_route_last_n_cache_key_is_distinct(monkeypatch):
    captured = []
    client = _route_client(monkeypatch, captured)
    url = ("/api/advanced-metrics/leaderboard?metric=snap_share"
           f"&season={SEASON}&last_n_games=4")
    client.get(url)
    client.get(url)
    assert len(captured) == 1  # second request served from the cache
    client.get("/api/advanced-metrics/leaderboard?metric=snap_share"
               f"&season={SEASON}&week_start=6&week_end=9")
    assert len(captured) == 2  # calendar mode is a different cache entry


def test_route_last_n_ignored_for_nonweekly_metric(monkeypatch):
    captured = []
    client = _route_client(monkeypatch, captured)
    payload = client.get(
        "/api/advanced-metrics/leaderboard?metric=air_yards_per_game"
        f"&season={SEASON}&last_n_games=4").get_json()
    assert payload["is_week_filtered"] is False
    assert payload["last_n_games"] is None


# ── Client: fetch params + labels (node) ─────────────────────────────

def _page_js():
    html = build_advanced_metrics_body(False, LEADERBOARD_METRICS)
    return html[html.index("<script>"):html.index("</script>")]


def _extract_fn(js, name):
    m = re.search(r"function %s\([^)]*\) \{.*?\n  \}" % re.escape(name), js, re.S)
    assert m, f"{name} not found in page script"
    return m.group(0)


def _run_node(script):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    proc = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr or proc.stdout


_CLIENT_FNS = ("amParseSeasons", "amSelectedSeasons", "amIsMultiSeason",
               "amSeasonLabel", "availableWeeks", "isMetricWeeklyCapable",
               "resolveWeekRange", "amRangeFetchParams", "_amApplyRangeParams",
               "_amRangeNoteLabel", "amUpdateContextLine")


def test_client_last_n_fetch_params_and_labels():
    js = _page_js()
    fns = "\n".join(_extract_fn(js, name) for name in _CLIENT_FNS)
    script = r"""
var cfg = { availableWeeksBySeason: { '2026': [1,2,3,4,5,6,7,8,9] },
            metrics: { snap_share: { weeklyCapable: true } }, seasons: [2026] };
var state = { metric: 'snap_share', season: '2026', combine: false,
              weekRange: '', weekStart: null, weekEnd: null,
              team: '', sortDir: 'desc' };
var _ctxEl = { textContent: '' };
var document = { getElementById: function() { return _ctxEl; } };
function check(name, cond) { if (!cond) { console.error('FAIL ' + name); process.exit(1); } }

state.weekRange = 'last4';
var r = amRangeFetchParams();
check('last4 lastN', r.lastN === 4 && r.ws === null && r.we === null);
var p = _amApplyRangeParams(new URLSearchParams());
check('last4 param', p.get('last_n_games') === '4' && !p.has('week_start') && !p.has('week_end'));
// The calendar resolver still resolves the span (week bar / URL sync).
var cal = resolveWeekRange();
check('last4 calendar span kept', cal.ws === 6 && cal.we === 9);
check('last4 note label', _amRangeNoteLabel() === 'Last 4 games');
amUpdateContextLine();
check('last4 context line', _ctxEl.textContent.indexOf('Last 4 games · per player') !== -1);

state.weekRange = 'last2';
check('last2 lastN', amRangeFetchParams().lastN === 2);

state.weekRange = 'custom'; state.weekStart = 6; state.weekEnd = 9;
var rc = amRangeFetchParams();
check('custom bounds', rc.lastN === null && rc.ws === 6 && rc.we === 9);
var pc = _amApplyRangeParams(new URLSearchParams());
check('custom params', pc.get('week_start') === '6' && pc.get('week_end') === '9' && !pc.has('last_n_games'));
check('custom note label', _amRangeNoteLabel() === 'W6' + '–' + 'W9');
amUpdateContextLine();
check('custom context line', _ctxEl.textContent.indexOf('Weeks 6-9') !== -1);

state.weekRange = '';
var rs = amRangeFetchParams();
check('season empty', rs.lastN === null && rs.ws === null && rs.we === null);
""" + "\n" + fns
    _run_node(script)


def test_client_copy_says_games_not_weeks():
    html = build_advanced_metrics_body(False, LEADERBOARD_METRICS)
    assert 'data-range="last2">Last 2 games</button>' in html
    assert 'data-range="last4">Last 4 games</button>' in html
    js = _page_js()
    assert "weekLbl = 'Last 4 games · per player'" in js
    assert "return 'Last 4 games';" in js  # active-filter chip label
    assert "return 'Last 4 weeks';" not in js
