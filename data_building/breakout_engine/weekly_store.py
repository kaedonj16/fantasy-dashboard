"""
Persistence for in-season weekly breakout results.

Deliberately a SEPARATE table from the offseason ``breakout_opportunity_scores``
board: weekly and offseason results are computed from different evidence, mean
different things, and must never overwrite or masquerade as one another. The
weekly table is keyed on (player_id, season, as_of_week) so each week's snapshot
is its own row and the history is queryable.

A companion ``weekly_breakout_runs`` table records one row per scoring run
(season, cutoff week, mode, coverage, counts, status). It lets the API expose
freshness ("scored through week 6, 1 week stale") and lets the runner PRESERVE
the last good snapshot when a refresh fails - it records a skipped/stale run
without deleting the previous week's scores.

All DB access is confined to this module; the scorer itself is pure.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, List, Optional

from dashboard_services.db import get_conn

WEEKLY_SCORES_TABLE = "weekly_breakout_scores"
WEEKLY_RUNS_TABLE = "weekly_breakout_runs"

_INIT_DONE = False


def init_weekly_breakout_db() -> None:
    """Create the weekly tables and indexes if absent. Idempotent; safe to call
    on every run (acts as the migration for existing databases)."""
    global _INIT_DONE
    if _INIT_DONE:
        return
    with get_conn() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {WEEKLY_SCORES_TABLE} (
                id               SERIAL PRIMARY KEY,
                player_id        VARCHAR(50) NOT NULL,
                player_name      VARCHAR(255),
                season           INTEGER NOT NULL,
                as_of_week       INTEGER NOT NULL,
                as_of_date       DATE NOT NULL,
                team             VARCHAR(10),
                position         VARCHAR(5),
                scoring_version  VARCHAR(40) NOT NULL,
                classification   VARCHAR(30),
                breakout_score   NUMERIC,
                confidence       NUMERIC,
                provisional      BOOLEAN DEFAULT FALSE,
                baseline_source  VARCHAR(20),
                evaluated_weeks  INTEGER[],
                recent_weeks     INTEGER[],
                baseline_weeks   INTEGER[],
                recent_games     INTEGER,
                baseline_games   INTEGER,
                coverage_fraction NUMERIC,
                reasons          TEXT,
                risks            TEXT,
                evidence         JSONB,
                calculated_at    TIMESTAMP DEFAULT NOW(),
                UNIQUE (player_id, season, as_of_week)
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_wbs_season_week "
            f"ON {WEEKLY_SCORES_TABLE} (season, as_of_week)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_wbs_score "
            f"ON {WEEKLY_SCORES_TABLE} (breakout_score DESC)"
        )
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {WEEKLY_RUNS_TABLE} (
                id               SERIAL PRIMARY KEY,
                season           INTEGER NOT NULL,
                as_of_week       INTEGER NOT NULL,
                as_of_date       DATE NOT NULL,
                scoring_version  VARCHAR(40),
                mode             VARCHAR(20),
                status           VARCHAR(20),
                candidates_scored INTEGER DEFAULT 0,
                records_saved    INTEGER DEFAULT 0,
                weeks_covered    INTEGER DEFAULT 0,
                detail           JSONB,
                calculated_at    TIMESTAMP DEFAULT NOW(),
                UNIQUE (season, as_of_week, status)
            )
            """
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_wbr_season "
            f"ON {WEEKLY_RUNS_TABLE} (season, calculated_at DESC)"
        )
    _INIT_DONE = True


def _result_row(result: Dict[str, Any], season: int, as_of_week: int, as_of_date: date) -> Dict[str, Any]:
    evidence = {
        "signals": result.get("signals"),
        "sample": result.get("sample"),
        "coverage": result.get("coverage"),
        "confidence_detail": result.get("confidence_detail"),
        "fantasy": result.get("fantasy"),
        "reasons": result.get("reasons"),
        "risks": result.get("risks"),
    }
    sample = result.get("sample") or {}
    return {
        "player_id": str(result.get("player_id") or ""),
        "player_name": result.get("player_name"),
        "season": int(season),
        "as_of_week": int(as_of_week),
        "as_of_date": as_of_date,
        "team": result.get("team"),
        "position": result.get("position"),
        "scoring_version": result.get("scoring_version"),
        "classification": result.get("classification"),
        "breakout_score": result.get("breakout_score"),
        "confidence": result.get("confidence"),
        "provisional": bool(result.get("provisional")),
        "baseline_source": result.get("baseline_source"),
        "evaluated_weeks": result.get("evaluated_weeks") or [],
        "recent_weeks": result.get("recent_weeks") or [],
        "baseline_weeks": result.get("baseline_weeks") or [],
        "recent_games": sample.get("recent_games"),
        "baseline_games": sample.get("baseline_games"),
        "coverage_fraction": (result.get("coverage") or {}).get("fraction"),
        "reasons": "\n".join(result.get("reasons") or []),
        "risks": "\n".join(result.get("risks") or []),
        "evidence": json.dumps(evidence),
    }


def save_weekly_scores(
    season: int,
    as_of_week: int,
    results: List[Dict[str, Any]],
    as_of_date: Optional[date] = None,
) -> int:
    """Replace the snapshot for (season, as_of_week) with these results.

    Deletes the existing rows for that exact (season, as_of_week) first so a
    player who drops off the list is not left behind, then inserts. Other weeks'
    snapshots are untouched, so history is preserved. Returns rows written.

    Caller must NOT invoke this with an empty list when it wants to preserve the
    previous snapshot - an empty save wipes the week. The runner guards this.
    """
    if as_of_date is None:
        as_of_date = date.today()
    init_weekly_breakout_db()
    rows = [_result_row(r, season, as_of_week, as_of_date) for r in results]
    cols = [
        "player_id", "player_name", "season", "as_of_week", "as_of_date", "team",
        "position", "scoring_version", "classification", "breakout_score",
        "confidence", "provisional", "baseline_source", "evaluated_weeks",
        "recent_weeks", "baseline_weeks", "recent_games", "baseline_games",
        "coverage_fraction", "reasons", "risks", "evidence",
    ]
    placeholders = ", ".join(
        f"%({c})s::jsonb" if c == "evidence" else f"%({c})s" for c in cols
    )
    insert = f"INSERT INTO {WEEKLY_SCORES_TABLE} ({', '.join(cols)}) VALUES ({placeholders})"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {WEEKLY_SCORES_TABLE} WHERE season = %s AND as_of_week = %s",
                (int(season), int(as_of_week)),
            )
            if rows:
                cur.executemany(insert, rows)
            return len(rows)


def record_run(
    season: int,
    as_of_week: int,
    *,
    mode: str,
    status: str,
    candidates_scored: int = 0,
    records_saved: int = 0,
    weeks_covered: int = 0,
    detail: Optional[Dict[str, Any]] = None,
    as_of_date: Optional[date] = None,
) -> None:
    """Record one run's metadata. status is 'success', 'skipped', or 'stale'."""
    if as_of_date is None:
        as_of_date = date.today()
    init_weekly_breakout_db()
    with get_conn() as conn:
        conn.execute(
            f"""
            INSERT INTO {WEEKLY_RUNS_TABLE}
                (season, as_of_week, as_of_date, scoring_version, mode, status,
                 candidates_scored, records_saved, weeks_covered, detail)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
            ON CONFLICT (season, as_of_week, status) DO UPDATE SET
                as_of_date = EXCLUDED.as_of_date,
                scoring_version = EXCLUDED.scoring_version,
                mode = EXCLUDED.mode,
                candidates_scored = EXCLUDED.candidates_scored,
                records_saved = EXCLUDED.records_saved,
                weeks_covered = EXCLUDED.weeks_covered,
                detail = EXCLUDED.detail,
                calculated_at = NOW()
            """,
            (
                int(season), int(as_of_week), as_of_date,
                (detail or {}).get("scoring_version"), mode, status,
                int(candidates_scored), int(records_saved), int(weeks_covered),
                json.dumps(detail or {}),
            ),
        )


def latest_scored_week(season: int) -> Optional[int]:
    """The most recent as_of_week that actually has saved score rows, or None."""
    init_weekly_breakout_db()
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT MAX(as_of_week) AS w FROM {WEEKLY_SCORES_TABLE} WHERE season = %s",
            (int(season),),
        ).fetchone()
    return int(row["w"]) if row and row.get("w") is not None else None


def get_latest_run(season: int) -> Optional[Dict[str, Any]]:
    """Most recent run row for a season (any status), for freshness reporting."""
    init_weekly_breakout_db()
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT * FROM {WEEKLY_RUNS_TABLE} WHERE season = %s "
            f"ORDER BY calculated_at DESC LIMIT 1",
            (int(season),),
        ).fetchone()
    return dict(row) if row else None


def load_weekly_candidates(
    season: int,
    as_of_week: Optional[int] = None,
    *,
    min_score: float = 0.0,
    classifications: Optional[List[str]] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Load the weekly board for a season.

    When as_of_week is None, uses the latest snapshot that has rows (so a failed
    refresh still serves the last good week rather than an empty board). Returns
    a payload with candidates plus freshness metadata.
    """
    init_weekly_breakout_db()
    week = as_of_week if as_of_week is not None else latest_scored_week(season)
    if week is None:
        return {
            "season": season, "as_of_week": None, "candidates": [], "count": 0,
            "data_available": False, "data_status": "unavailable",
        }

    params: List[Any] = [int(season), int(week), float(min_score)]
    clause = ""
    if classifications:
        clause = " AND classification = ANY(%s)"
        params.append(list(classifications))
    query = (
        f"SELECT * FROM {WEEKLY_SCORES_TABLE} "
        f"WHERE season = %s AND as_of_week = %s AND breakout_score >= %s{clause} "
        f"ORDER BY breakout_score DESC, confidence DESC"
    )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]

    if limit and limit > 0:
        rows = rows[:limit]

    run = get_latest_run(season)
    latest_run_week = run.get("as_of_week") if run else week
    weeks_stale = max(0, int(latest_run_week or week) - int(week)) if latest_run_week else 0

    as_of_date = rows[0]["as_of_date"] if rows else (run or {}).get("as_of_date")
    return {
        "season": season,
        "as_of_week": int(week),
        "as_of_date": as_of_date.isoformat() if hasattr(as_of_date, "isoformat") else as_of_date,
        "candidates": rows,
        "count": len(rows),
        "data_available": True,
        "data_status": "stale" if weeks_stale >= 1 else "ok",
        "weeks_stale": weeks_stale,
        "scoring_version": (run or {}).get("scoring_version"),
        "last_run_status": (run or {}).get("status"),
    }


def get_weekly_candidate(
    player_id: str,
    season: int,
    as_of_week: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """One player's latest weekly row (or the given week's)."""
    init_weekly_breakout_db()
    week = as_of_week if as_of_week is not None else latest_scored_week(season)
    if week is None:
        return None
    with get_conn() as conn:
        row = conn.execute(
            f"SELECT * FROM {WEEKLY_SCORES_TABLE} "
            f"WHERE player_id = %s AND season = %s AND as_of_week = %s",
            (str(player_id), int(season), int(week)),
        ).fetchone()
    return dict(row) if row else None
