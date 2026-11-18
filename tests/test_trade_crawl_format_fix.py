"""Trade crawl format-starvation fix (Nov 2026).

Regression set for the 13 days (2026-11-05 onward) when the nightly
recrawl inserted zero dynasty and zero redraft trades while keeper
flowed: (1) _mark_crawled_batch stamped last_trade_at with the crawl
time (NOW()), so the velocity-ordered batch was captured forever by
whichever format yielded last; (2) _crawl_one started at
last_crawled_week + 1 while leagues are marked AT the current week,
so trades made later in an already-marked week were never fetched;
(3) "both"-mode batch selection had no per-format guarantee. Plus
the catch-up script that backfills the missed dynasty/redraft trades.
"""
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("requests")

from data_building.trade_intel import trade_crawler as tc

REPO_ROOT = Path(__file__).resolve().parents[1]
TS = datetime(2026, 11, 16, 15, 47, tzinfo=timezone.utc)


def _load_catchup():
    spec = importlib.util.spec_from_file_location(
        "catchup_trade_crawl",
        REPO_ROOT / "scripts" / "catchup_trade_crawl.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(lid, season=2026, last_crawled_week=8, league_type=2):
    return {"league_id": lid, "season": season,
            "last_crawled_week": last_crawled_week,
            "league_type": league_type}


def _capturing_conn(captured):
    def fake_execute(query, params=None):
        captured["query"] = query
        captured["params"] = params
        return MagicMock()

    fake_conn = MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.execute.side_effect = fake_execute
    return fake_conn


# ---------------------------------------------------------------------------
# last_trade_at comes from the trade, never from the crawl clock
# ---------------------------------------------------------------------------

def test_mark_crawled_batch_uses_real_trade_timestamp():
    captured = {}
    with patch.object(tc, "get_conn",
                      return_value=_capturing_conn(captured)):
        tc._mark_crawled_batch([(11, "lg1", 3, TS), (11, "lg2", 0, None)])
    q = " ".join(captured["query"].split())
    # No crawl-time stamp anywhere in the last_trade_at assignment.
    assert "THEN NOW()" not in q
    assert "last_trade_at = CASE" in q
    assert "WHEN v.last_trade_at IS NULL THEN t.last_trade_at" in q
    assert ("GREATEST(COALESCE(t.last_trade_at, v.last_trade_at), "
            "v.last_trade_at)") in q
    # Provided timestamps are bound through, in row order.
    assert captured["params"] == [11, "lg1", 3, TS, 11, "lg2", 0, None]


def test_mark_crawled_batch_legacy_three_tuple_leaves_last_trade_at():
    captured = {}
    with patch.object(tc, "get_conn",
                      return_value=_capturing_conn(captured)):
        tc._mark_crawled_batch([(18, "old", 2)])
    assert captured["params"] == [18, "old", 2, None]


# ---------------------------------------------------------------------------
# _crawl_one refetches the last marked week + reports the trade timestamp
# ---------------------------------------------------------------------------

def test_crawl_one_refetches_last_marked_week():
    captured = {}

    def fake_crawl(league_id, season, start_week=None, end_week=None):
        captured.update(start=start_week, end=end_week)
        return 5, TS

    with patch.object(tc, "crawl_league_detailed", side_effect=fake_crawl):
        result = tc._crawl_one(_row("a", last_crawled_week=7), end_week=11)
    assert captured == {"start": 7, "end": 11}
    assert result == ("a", 5, 2, TS)


def test_crawl_one_beyond_end_week_crawls_nothing():
    with patch.object(tc, "crawl_league_detailed") as fake:
        result = tc._crawl_one(_row("a", last_crawled_week=18), end_week=11)
    assert result == ("a", 0, 2, None)
    fake.assert_not_called()


# ---------------------------------------------------------------------------
# Per-format guarantees in "both" mode
# ---------------------------------------------------------------------------

def test_merge_with_format_guarantees_tops_up_starved_formats():
    merged = [_row(f"k{i}", league_type=1) for i in range(9)]
    per_format = {
        0: [_row(f"r{i}", league_type=0) for i in range(4)],
        1: [_row(f"k{i}", league_type=1) for i in range(4)],
        2: [_row(f"d{i}", league_type=2) for i in range(4)],
    }
    result = tc._merge_with_format_guarantees(merged, per_format, 9)
    ids = [r["league_id"] for r in result]
    assert len(ids) == 9 and len(set(ids)) == 9
    counts = {}
    for r in result:
        counts[r["league_type"]] = counts.get(r["league_type"], 0) + 1
    # batch 9 -> each format is guaranteed ceil(9/3) = 3 slots
    assert counts == {0: 3, 1: 3, 2: 3}


def test_both_mode_selection_guarantees_each_format():
    new_rows = [_row("n0", league_type=2)]
    reserve_rows = [_row("old0", last_crawled_week=3, league_type=2)]
    per_format = {
        0: [_row(f"r{i}", league_type=0) for i in range(6)],
        1: [_row(f"k{i}", league_type=1) for i in range(6)],
        2: [_row(f"d{i}", league_type=2) for i in range(6)],
    }
    # The global velocity pick is keeper-captured, as in production.
    global_velocity = [_row(f"k{i}", league_type=1) for i in range(14)]

    def fake_execute(query, params=None):
        fake = MagicMock()
        if "league_type = %s" in query:
            fake.fetchall.return_value = list(per_format[params[0]])
        elif "discovered_at ASC" in query:
            fake.fetchall.return_value = list(new_rows)
        elif "last_trade_at DESC" in query:
            fake.fetchall.return_value = list(global_velocity)
        else:
            fake.fetchall.return_value = list(reserve_rows)
        return fake

    fake_conn = MagicMock()
    fake_conn.__enter__.return_value = fake_conn
    fake_conn.execute.side_effect = fake_execute

    with patch.object(tc, "get_conn", return_value=fake_conn):
        rows = tc._leagues_to_crawl(batch_size=12, crawl_mode="both",
                                    recrawl_days=2)
    ids = [r["league_id"] for r in rows]
    assert len(ids) == 12 and len(set(ids)) == 12
    counts = {}
    for r in rows:
        counts[r["league_type"]] = counts.get(r["league_type"], 0) + 1
    # batch 12 -> every format holds at least ceil(12/3) = 4 slots,
    # even though the global velocity pick was 100% keeper.
    assert counts == {0: 4, 1: 4, 2: 4}
    # The pre-existing new/reserve guarantees survive the top-up.
    assert "n0" in ids and "old0" in ids


# ---------------------------------------------------------------------------
# Catch-up script
# ---------------------------------------------------------------------------

def test_catchup_league_query_filters_season_formats_and_active():
    mod = _load_catchup()
    captured = {}
    fake_conn = _capturing_conn(captured)
    with patch("dashboard_services.db.get_conn", return_value=fake_conn):
        mod._catchup_leagues(2026, batch_size=500)
    q = " ".join(captured["query"].split())
    assert "crawl_enabled = TRUE" in q
    assert "season = %s" in q
    assert "league_type IN (0, 2)" in q
    assert "total_trades > 0" in q
    assert "last_crawled_at ASC NULLS FIRST" in q
    assert captured["params"] == (2026, 500)


def test_catchup_run_full_rescan_marks_with_real_timestamp():
    mod = _load_catchup()
    rows = [_row("d1", league_type=2), _row("r1", league_type=0)]
    crawled = []
    marked = []

    def fake_detailed(league_id, season, start_week=None, end_week=None):
        crawled.append((league_id, season, start_week, end_week))
        return (4, TS) if league_id == "d1" else (2, TS)

    with patch.object(mod, "_catchup_leagues",
                      side_effect=[rows, []]), \
         patch.object(tc, "_current_nfl_season", return_value=2026), \
         patch.object(tc, "_current_nfl_week", return_value=11), \
         patch.object(tc, "crawl_league_detailed",
                      side_effect=fake_detailed), \
         patch.object(tc, "_mark_crawled_batch",
                      side_effect=lambda batch: marked.extend(batch)):
        result = mod.run_catchup(season=2026, batch_size=10, workers=2)
    assert result["new_trades"] == 6
    assert result["by_format"] == {0: 2, 2: 4}
    assert result["batches"] == 1
    # Full rescan: week 0 through the current week, for every league.
    assert {c[0]: c[2:] for c in crawled} == {"d1": (0, 11), "r1": (0, 11)}
    # Marked at the season end week with the real newest-trade time.
    assert {m[1]: m for m in marked} == {
        "d1": (11, "d1", 4, TS), "r1": (11, "r1", 2, TS)}


def test_catchup_dry_run_crawls_nothing():
    mod = _load_catchup()
    rows = [_row("d1", league_type=2), _row("r1", league_type=0)]
    with patch.object(mod, "_catchup_leagues", return_value=rows), \
         patch.object(tc, "_current_nfl_season", return_value=2026), \
         patch.object(tc, "_current_nfl_week", return_value=11), \
         patch.object(tc, "crawl_league_detailed") as fake:
        result = mod.run_catchup(season=2026, batch_size=10, dry_run=True)
    assert result["new_trades"] == 0
    assert result["by_format"] == {0: 1, 2: 1}
    fake.assert_not_called()
