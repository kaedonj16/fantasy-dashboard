"""Trade DB growth: full-season historical crawls + never-crawled guarantee.

Covers the two reasons prior seasons held ~0 trades while the current
season grew: (1) every league was crawled only up to the *current*
season's week, truncating completed seasons, and marked with that
week; (2) in crawl_mode="both" the velocity sort ranked never-crawled
leagues (NULL last_trade_at) behind every league that ever traded, so
newly discovered leagues - prior-season ancestors included - starved.
Plus the backfill script that drains the historical backlog.
"""
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("requests")

from data_building.trade_intel import trade_crawler as tc

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_backfill():
    spec = importlib.util.spec_from_file_location(
        "backfill_historical_trades",
        REPO_ROOT / "scripts" / "backfill_historical_trades.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(lid, season=2026, last_crawled_week=None):
    return {"league_id": lid, "season": season,
            "last_crawled_week": last_crawled_week, "league_type": 2}


# ---------------------------------------------------------------------------
# Per-season end week
# ---------------------------------------------------------------------------

def test_season_end_week_completed_season_is_full():
    assert tc._season_end_week(2025, 2026, 9) == tc._WEEKS_PER_SEASON == 18
    assert tc._season_end_week(2023, 2026, 9) == 18


def test_season_end_week_current_and_future_use_current_week():
    assert tc._season_end_week(2026, 2026, 9) == 9
    assert tc._season_end_week(2027, 2026, 9) == 9
    assert tc._season_end_week(None, 2026, 9) == 9
    assert tc._season_end_week("2025", 2026, 9) == 18


def test_crawl_league_default_end_week_is_season_aware():
    weeks = []

    def fake_fetch(league_id, week):
        weeks.append(week)
        return week, []

    with patch.object(tc, "_fetch_week", side_effect=fake_fetch), \
         patch.object(tc, "_current_nfl_season", return_value=2026), \
         patch.object(tc, "_current_nfl_week", return_value=9):
        assert tc.crawl_league("lg", 2025, start_week=1) == 0
        assert max(weeks) == 18  # completed season fetched in full
        weeks.clear()
        assert tc.crawl_league("lg", 2026, start_week=1) == 0
        assert max(weeks) == 9  # current season stops at the live week


# ---------------------------------------------------------------------------
# _crawl_one start week: week 0 for first crawls
# ---------------------------------------------------------------------------

def test_crawl_one_never_crawled_starts_at_week_zero():
    captured = {}

    def fake_crawl(league_id, season, start_week=None, end_week=None):
        captured.update(start=start_week, end=end_week)
        return 0

    with patch.object(tc, "crawl_league", side_effect=fake_crawl):
        tc._crawl_one(_row("a", last_crawled_week=None), end_week=18)
    assert captured == {"start": 0, "end": 18}


def test_crawl_one_incremental_and_override():
    captured = {}

    def fake_crawl(league_id, season, start_week=None, end_week=None):
        captured.update(start=start_week, end=end_week)
        return 0

    with patch.object(tc, "crawl_league", side_effect=fake_crawl):
        tc._crawl_one(_row("a", last_crawled_week=10), end_week=18)
        assert captured["start"] == 11
        tc._crawl_one(_row("a", last_crawled_week=10), end_week=18,
                      override_start_week=0)
        assert captured["start"] == 0


# ---------------------------------------------------------------------------
# Never-crawled guarantee in "both" mode
# ---------------------------------------------------------------------------

def test_merge_crawl_batches_guarantees_new_leagues():
    velocity = [_row(f"v{i}", last_crawled_week=8) for i in range(20)]
    reserve = [_row("old1", last_crawled_week=3)]
    new = [_row(f"n{i}") for i in range(10)]
    merged = tc._merge_crawl_batches(velocity, reserve, batch_size=20,
                                     new_rows=new)
    ids = [r["league_id"] for r in merged]
    assert len(ids) == 20 and len(set(ids)) == 20
    # batch 20 -> 5 guaranteed never-crawled slots, oldest first
    assert [f"n{i}" for i in range(5)] == [i for i in ids if i.startswith("n")][:5]
    assert len([i for i in ids if i.startswith("n")]) >= 5
    assert "old1" in ids


def test_both_mode_includes_never_crawled_rows():
    new_rows = [_row(f"n{i}") for i in range(3)]
    reserve_rows = [_row("old1", last_crawled_week=3)]
    velocity_rows = [_row(f"v{i}", last_crawled_week=8) for i in range(10)]

    def fake_execute(query, params=None):
        fake = MagicMock()
        if "last_trade_at DESC" in query:
            fake.fetchall.return_value = list(velocity_rows)
        elif "discovered_at ASC" in query:
            fake.fetchall.return_value = list(new_rows)
        else:
            fake.fetchall.return_value = list(reserve_rows)
        return fake

    fake_conn = MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.execute.side_effect = fake_execute

    with patch.object(tc, "get_conn", return_value=fake_conn):
        rows = tc._leagues_to_crawl(batch_size=10, crawl_mode="both",
                                    recrawl_days=2)
    ids = [r["league_id"] for r in rows]
    assert len(ids) == 10
    assert "n0" in ids and "n1" in ids  # batch 10 -> 2 guaranteed new slots
    assert "old1" in ids


def test_both_mode_new_pick_query_shape():
    captured = []

    def fake_execute(query, params=None):
        captured.append((query, params))
        fake = MagicMock()
        fake.fetchall.return_value = []
        return fake

    fake_conn = MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.execute.side_effect = fake_execute

    with patch.object(tc, "get_conn", return_value=fake_conn):
        tc._leagues_to_crawl(batch_size=2000, crawl_mode="both",
                             recrawl_days=2)
    new_q, new_params = captured[0]
    assert "last_crawled_week IS NULL" in new_q
    assert "discovered_at ASC" in new_q
    assert "IN (0, 1, 2)" in new_q
    assert new_params == (500,)  # 2000 // 4 guaranteed slots


# ---------------------------------------------------------------------------
# Historical backfill script
# ---------------------------------------------------------------------------

def test_backfill_start_week_rules():
    mod = _load_backfill()
    assert mod._start_week_for(_row("a", season=2025, last_crawled_week=None), full=False) == 0
    assert mod._start_week_for(_row("a", season=2025, last_crawled_week=10), full=False) == 11
    assert mod._start_week_for(_row("a", season=2025, last_crawled_week=10), full=True) == 0


def test_backfill_league_query_only_incomplete_prior_seasons():
    mod = _load_backfill()
    captured = {}

    def fake_execute(query, params=None):
        captured["query"] = query
        captured["params"] = params
        fake = MagicMock()
        fake.fetchall.return_value = []
        return fake

    fake_conn = MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.execute.side_effect = fake_execute

    with patch("dashboard_services.db.get_conn", return_value=fake_conn):
        mod._historical_leagues([2025, 2024], batch_size=100)
    q = captured["query"]
    assert "season = ANY" in q
    assert "last_crawled_week IS NULL OR last_crawled_week < 18" in q
    assert "IN (0, 1, 2)" in q
    assert captured["params"] == ([2025, 2024], 100)

    with patch("dashboard_services.db.get_conn", return_value=fake_conn):
        mod._historical_leagues([2025], batch_size=100, full=True)
    assert "last_crawled_week < 18" not in captured["query"]


def test_backfill_run_crawls_full_season_and_marks_week_18():
    mod = _load_backfill()
    rows = [_row("h1", season=2025, last_crawled_week=10),
            _row("h2", season=2024, last_crawled_week=None)]
    crawled = []
    marked = []

    def fake_crawl(league_id, season, start_week=None, end_week=None):
        crawled.append((league_id, season, start_week, end_week))
        return 3

    with patch.object(mod, "_historical_leagues", return_value=rows), \
         patch.object(tc, "_current_nfl_season", return_value=2026), \
         patch.object(tc, "crawl_league", side_effect=fake_crawl), \
         patch.object(tc, "_mark_crawled_batch",
                      side_effect=lambda batch: marked.extend(batch)):
        result = mod.run_backfill(seasons=[2025, 2024], batch_size=10,
                                  workers=2)
    assert result["new_trades"] == 6
    assert result["by_season"] == {2025: 3, 2024: 3}
    by_id = {c[0]: c for c in crawled}
    assert by_id["h1"][2:] == (11, 18)  # resumes at its frontier, full end
    assert by_id["h2"][2:] == (0, 18)   # never crawled: week 0..18
    assert {m[1] for m in marked} == {"h1", "h2"}
    assert all(m[0] == 18 for m in marked)


def test_backfill_run_skips_current_season():
    mod = _load_backfill()
    with patch.object(tc, "_current_nfl_season", return_value=2026):
        result = mod.run_backfill(seasons=[2026], batch_size=10)
    assert result == {"leagues": 0, "new_trades": 0, "by_season": {}}


# ---------------------------------------------------------------------------
# render.yaml growth config
# ---------------------------------------------------------------------------

def test_render_yaml_trade_growth_config():
    source = (REPO_ROOT / "render.yaml").read_text()
    recrawl = source.split("name: trade-recrawl", 1)[1].split("\n  - type:", 1)[0]
    assert "schedule: 0 */2 * * *" in recrawl
    assert "--discover-target 1000" in recrawl
    assert "--batch-size 2000" in recrawl

    backfill = source.split("name: trade-history-backfill", 1)[1].split("\n  - type:", 1)[0]
    assert "plan: standard" in backfill
    assert "backfill_historical_trades.py" in backfill
