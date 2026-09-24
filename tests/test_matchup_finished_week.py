"""Regression tests for finished-week matchup board fixes.

1. Final game lines read "Final 24-31 @ LV" (score before the @ opponent),
   falling back to a bare "Final" when no score is available.
2. Past weeks backfill the Sleeper per-player stats file once, so players
   missing from the Footballguys scrape (e.g. Brock Purdy in 2026 Week 2)
   get a box-score line instead of "Stats unavailable".
"""
import dashboard_services.matchups as matchups


def test_format_final_game_line_score_before_opponent():
    assert matchups._format_final_game_line("24-31", "@ LV") == "Final 24-31 @ LV"
    assert matchups._format_final_game_line("34-3", "vs ATL") == "Final 34-3 vs ATL"


def test_format_final_game_line_bare_final_without_score():
    assert matchups._format_final_game_line("", "@ LV") == "Final"


def test_week_stats_for_slide_backfills_sleeper_for_past_weeks(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "data_building.external_data.sleeper_bulk_stats.fetch_week_stats",
        lambda season, week, **k: calls.append((season, week)) or {},
    )
    matchups._week_stats_for_slide(2026, 2, ensure_sleeper=True)
    assert calls == [(2026, 2)]


def test_week_stats_for_slide_skips_backfill_for_live_week(monkeypatch):
    def _boom(season, week, **k):
        raise AssertionError("must not fetch for the live week")
    monkeypatch.setattr(
        "data_building.external_data.sleeper_bulk_stats.fetch_week_stats", _boom
    )
    matchups._week_stats_for_slide(2026, 2, ensure_sleeper=False)
