"""
Weekly snapshots of each team's total roster value, so the team modal can show
a value-over-time trend (who's building vs. aging out).

There's no historical-roster record to reconstruct from — rosters change with
trades and adds — so we snapshot the *current* roster's total value each week
going forward, keyed by (league, season, week, roster). The team modal both
records the current week (throttled to one row via upsert) and reads the series.
Best-effort: a failure returns an empty series and never breaks the modal.

Mirrors dashboard_services.playoff_odds_history (the weekly-snapshot pattern).
"""
from typing import Dict, List, Optional


def get_conn():
    """Lazy DB handle: importing this module (e.g. under the pure test suite,
    which has no psycopg) must not pull in the driver until a query runs."""
    from dashboard_services.db import get_conn as _get_conn
    return _get_conn()


_TABLE_READY = False


def _ensure_table() -> None:
    global _TABLE_READY
    if _TABLE_READY:
        return
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS team_value_history (
                league_id      TEXT    NOT NULL,
                season         INTEGER NOT NULL,
                week           INTEGER NOT NULL,
                roster_id      INTEGER NOT NULL,
                total_value    NUMERIC NOT NULL,
                sf_total_value NUMERIC,
                created_at     TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (league_id, season, week, roster_id)
            )
            """
        )
    _TABLE_READY = True


def record_and_series(
    league_id: str,
    season: int,
    week: int,
    roster_id,
    total_value: float,
    sf_total_value: Optional[float] = None,
    write: bool = True,
) -> Dict[str, object]:
    """Upsert this week's total for the team, then return its season series.

    Returns {"series": [{"week": w, "value": v}], "current": v, "delta": Δ}
    where delta is this week minus the previous recorded week (None if there's
    no earlier week). Returns an empty series on any failure or when week < 1."""
    empty = {"series": [], "current": None, "delta": None}
    try:
        week = int(week)
    except (TypeError, ValueError):
        return empty
    if week < 1:
        return empty

    try:
        _ensure_table()
        with get_conn() as conn:
            if write and total_value is not None:
                conn.execute(
                    """
                    INSERT INTO team_value_history
                        (league_id, season, week, roster_id, total_value, sf_total_value)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (league_id, season, week, roster_id)
                    DO UPDATE SET total_value = EXCLUDED.total_value,
                                  sf_total_value = EXCLUDED.sf_total_value,
                                  created_at = NOW()
                    """,
                    (str(league_id), int(season), week, int(roster_id),
                     float(total_value),
                     None if sf_total_value is None else float(sf_total_value)),
                )

            rows = conn.execute(
                "SELECT week, total_value FROM team_value_history "
                "WHERE league_id = %s AND season = %s AND roster_id = %s "
                "ORDER BY week ASC",
                (str(league_id), int(season), int(roster_id)),
            ).fetchall()

        series = [{"week": int(r["week"]), "value": round(float(r["total_value"]), 1)}
                  for r in (rows or []) if r["total_value"] is not None]
        if not series:
            return empty
        current = series[-1]["value"]
        delta = round(current - series[-2]["value"], 1) if len(series) >= 2 else None
        return {"series": series, "current": current, "delta": delta}
    except Exception:
        import logging
        logging.getLogger(__name__).debug(
            "team_value_history: record/series failed", exc_info=True)
        return empty
