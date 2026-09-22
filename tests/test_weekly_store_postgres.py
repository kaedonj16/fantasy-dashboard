"""PostgreSQL regression coverage for joined weekly snapshot readers."""
from __future__ import annotations

import os
import uuid
from contextlib import contextmanager

import pytest


def test_candidate_detail_join_is_unambiguous_and_selects_completed_snapshot(monkeypatch):
    psycopg = pytest.importorskip("psycopg")
    url = os.getenv("TEST_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL/DATABASE_URL is required for PostgreSQL integration")
    from psycopg.rows import dict_row
    from data_building.breakout_engine import weekly_store
    from data_building.breakout_engine.weekly_breakout import SCORING_VERSION

    schema = "test_weekly_" + uuid.uuid4().hex
    conn = psycopg.connect(url, row_factory=dict_row)
    conn.autocommit = True
    conn.execute(f'CREATE SCHEMA "{schema}"')
    conn.execute(f'SET search_path TO "{schema}"')
    conn.execute("""CREATE TABLE weekly_breakout_runs (
        id BIGSERIAL PRIMARY KEY, season INT, as_of_week INT, scoring_version TEXT,
        status TEXT, completed_at TIMESTAMP, expected_row_count INT,
        inserted_row_count INT, calculated_at TIMESTAMP DEFAULT NOW(), detail JSONB,
        as_of_date DATE)""")
    conn.execute("""CREATE TABLE weekly_breakout_scores (
        id BIGSERIAL PRIMARY KEY, player_id TEXT, player_name TEXT, season INT,
        as_of_week INT, scoring_version TEXT, run_id BIGINT, as_of_date DATE,
        breakout_score NUMERIC, confidence NUMERIC, classification TEXT,
        team TEXT, position TEXT, reasons TEXT, risks TEXT, evidence JSONB)""")

    def add(week, version, status="completed", player="love"):
        run_id = conn.execute(
            "INSERT INTO weekly_breakout_runs "
            "(season,as_of_week,scoring_version,status,completed_at,expected_row_count,inserted_row_count,as_of_date) "
            "VALUES (2026,%s,%s,%s,CASE WHEN %s='completed' THEN NOW() END,1,1,CURRENT_DATE) RETURNING id",
            (week, version, status, status),
        ).fetchone()["id"]
        conn.execute(
            "INSERT INTO weekly_breakout_scores "
            "(player_id,player_name,season,as_of_week,scoring_version,run_id,as_of_date,breakout_score,confidence,evidence) "
            "VALUES (%s,'Jeremiyah Love',2026,%s,%s,%s,CURRENT_DATE,77,80,'{}')",
            (player, week, version, run_id),
        )

    add(5, SCORING_VERSION)
    add(6, "incompatible-version")
    add(8, SCORING_VERSION, status="running")
    add(7, SCORING_VERSION)

    @contextmanager
    def test_conn():
        yield conn

    monkeypatch.setattr(weekly_store, "get_conn", test_conn)
    monkeypatch.setattr(weekly_store, "_INIT_DONE", True)
    try:
        assert weekly_store.get_weekly_candidate("love", 2026, 5)["as_of_week"] == 5
        assert weekly_store.get_weekly_candidate("love", 2026)["as_of_week"] == 7

        # Exercise the dashboard service, not merely the store helper.
        from dashboard_services import breakout_api
        monkeypatch.setattr(breakout_api, "_breakout_ranks", lambda season: {"love": 1})
        detail = breakout_api.get_weekly_breakout_detail("love", 2026)
        assert detail["available"] is True
        assert detail["player_name"] == "Jeremiyah Love"
        assert detail["breakout_rank"] == 1
    finally:
        conn.execute("SET search_path TO public")
        conn.execute(f'DROP SCHEMA "{schema}" CASCADE')
        conn.close()
