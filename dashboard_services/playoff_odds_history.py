"""
Weekly playoff-odds snapshots so the playoff-odds table can show movement arrows.

When the odds are (re)simulated we upsert the current week's rows, then load the
most recent earlier week to compute per-team movement in the odds ranking
(sorted by playoff probability, avg final wins as the tiebreaker — the same order
the table renders in). Failures are swallowed by the caller: arrows are
decorative and must never break the endpoint.

Mirrors dashboard_services.power_rank_history; the table is the one defined in
migrations/003_analytics.sql.
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
            CREATE TABLE IF NOT EXISTS playoff_odds (
                league_id                 TEXT        NOT NULL,
                season                    INTEGER     NOT NULL,
                week                      INTEGER     NOT NULL,
                roster_id                 INTEGER     NOT NULL,
                team_name                 TEXT,
                current_wins              INTEGER,
                current_losses            INTEGER,
                current_ties              INTEGER     DEFAULT 0,
                playoff_probability       DECIMAL(5,2),
                first_seed_probability    DECIMAL(5,2),
                bye_probability           DECIMAL(5,2),
                miss_playoffs_probability DECIMAL(5,2),
                avg_final_wins            DECIMAL(5,2),
                avg_final_losses          DECIMAL(5,2),
                num_simulations           INTEGER     DEFAULT 10000,
                calculated_at             TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (league_id, season, week, roster_id)
            )
            """
        )
    _TABLE_READY = True


_DAILY_TABLE_READY = False


def _ensure_daily_table() -> None:
    """Daily playoff-odds snapshots — the basis for the movement arrows. Keyed by
    calendar date (not week) so movement works year-round: a trade in the
    offseason moves a team's odds and the next day's snapshot reflects it, exactly
    like a week-over-week shift does in-season."""
    global _DAILY_TABLE_READY
    if _DAILY_TABLE_READY:
        return
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS playoff_odds_daily (
                league_id           TEXT        NOT NULL,
                season              INTEGER     NOT NULL,
                snap_date           DATE        NOT NULL,
                roster_id           INTEGER     NOT NULL,
                playoff_probability DECIMAL(5,2),
                avg_final_wins      DECIMAL(5,2),
                calculated_at       TIMESTAMP   DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (league_id, season, snap_date, roster_id)
            )
            """
        )
    _DAILY_TABLE_READY = True


def record_daily_and_movement(
    league_id: str,
    season: int,
    odds: List[dict],
    write: bool = True,
) -> Dict[str, float]:
    """Snapshot today's playoff odds (once per day per team) and return each
    team's probability movement vs the most recent *earlier* daily snapshot.

    Returns {roster_id: delta_pct} where delta is this projection's playoff
    probability minus the previous snapshot's, in points (positive = improved).
    Works in-season and in the offseason, so a roster move (e.g. a trade) shows a
    ▲/▼ once the next day's snapshot is taken. Returns {} when there's no earlier
    snapshot to compare against yet. Writes are best-effort and only taken when
    ``write`` is True (the caller passes True on a fresh sim, False on a cache hit
    so the table isn't hammered on every view)."""
    if not odds:
        return {}
    try:
        from datetime import date, timezone, datetime
        _ensure_daily_table()
        today = datetime.now(timezone.utc).date().isoformat()

        cur = {
            str(o["roster_id"]): float(o.get("playoff_pct") or 0.0)
            for o in odds
            if o.get("roster_id") is not None
        }
        if not cur:
            return {}

        with get_conn() as conn:
            if write:
                for o in odds:
                    if o.get("roster_id") is None:
                        continue
                    conn.execute(
                        """
                        INSERT INTO playoff_odds_daily (
                            league_id, season, snap_date, roster_id,
                            playoff_probability, avg_final_wins, calculated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (league_id, season, snap_date, roster_id)
                        DO UPDATE SET
                            playoff_probability = EXCLUDED.playoff_probability,
                            avg_final_wins = EXCLUDED.avg_final_wins,
                            calculated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            str(league_id), int(season), today, int(o["roster_id"]),
                            o.get("playoff_pct"), o.get("avg_final_wins"),
                        ),
                    )

            prev = conn.execute(
                "SELECT MAX(snap_date) AS d FROM playoff_odds_daily "
                "WHERE league_id = %s AND season = %s AND snap_date < %s",
                (str(league_id), int(season), today),
            ).fetchone()
            prev_date = prev and prev["d"]
            if not prev_date:
                return {}

            prev_rows = conn.execute(
                "SELECT roster_id, playoff_probability FROM playoff_odds_daily "
                "WHERE league_id = %s AND season = %s AND snap_date = %s",
                (str(league_id), int(season), prev_date),
            ).fetchall()

        prev_map = {
            str(r["roster_id"]): float(r["playoff_probability"] or 0.0)
            for r in (prev_rows or [])
        }
        return {rid: round(cur[rid] - prev_map[rid], 1)
                for rid in cur if rid in prev_map}
    except Exception:
        import logging
        logging.getLogger(__name__).debug(
            "playoff_odds_history: daily movement failed", exc_info=True)
        return {}


def _rank(rows: List[dict]) -> Dict[str, int]:
    """roster_id -> 1..N rank by (playoff prob desc, avg final wins desc)."""
    ordered = sorted(
        rows,
        key=lambda r: (
            -float(r.get("playoff_probability") or 0.0),
            -float(r.get("avg_final_wins") or 0.0),
        ),
    )
    return {str(r["roster_id"]): i + 1 for i, r in enumerate(ordered)}


def get_series(league_id: str, season: int, roster_id) -> Dict[str, object]:
    """A team's playoff-probability across recorded weeks, for a trend sparkline.

    Returns {"series": [{"week": w, "pct": p}], "current": p, "delta": Δpct}
    (delta = this week's probability minus the previous recorded week's, in
    points). Empty on any failure or when nothing has been recorded yet."""
    empty = {"series": [], "current": None, "delta": None}
    try:
        _ensure_table()
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT week, playoff_probability FROM playoff_odds "
                "WHERE league_id = %s AND season = %s AND roster_id = %s "
                "ORDER BY week ASC",
                (str(league_id), int(season), int(roster_id)),
            ).fetchall()
        series = [{"week": int(r["week"]), "pct": round(float(r["playoff_probability"]), 1)}
                  for r in (rows or []) if r["playoff_probability"] is not None]
        if not series:
            return empty
        current = series[-1]["pct"]
        delta = round(current - series[-2]["pct"], 1) if len(series) >= 2 else None
        return {"series": series, "current": current, "delta": delta}
    except Exception:
        import logging
        logging.getLogger(__name__).debug(
            "playoff_odds_history: get_series failed", exc_info=True)
        return empty


def record_and_movement(
    league_id: str,
    season: int,
    week: int,
    odds: List[dict],
    write: bool = True,
) -> Dict[str, Optional[int]]:
    """Optionally upsert this week's odds, then return movement vs the prior week.

    ``odds`` are the rows from simulate_playoff_odds (playoff_pct, first_seed_pct,
    bye_pct, miss_pct, avg_final_wins/losses, wins/losses/ties, roster_id, ...).

    Returns {roster_id: delta} where positive = climbed in the odds ranking since
    the previous recorded week, negative = dropped, 0 = unchanged. Returns {} when
    there's no in-season week to compare (week < 1) or no earlier snapshot yet.
    Writes are best-effort and are only taken when ``write`` is True (the caller
    passes True on a fresh simulation, False on a cache hit, so the table isn't
    hammered on every view)."""
    try:
        week = int(week)
    except (TypeError, ValueError):
        return {}
    if week < 1 or not odds:
        return {}

    try:
        _ensure_table()

        # Normalize the simulator's rows onto the table's column names once, so
        # both the write and the current-week ranking read the same shape.
        cur_norm = [
            {
                "roster_id": o.get("roster_id"),
                "team_name": o.get("team_name"),
                "current_wins": o.get("wins"),
                "current_losses": o.get("losses"),
                "current_ties": o.get("ties") or 0,
                "playoff_probability": o.get("playoff_pct"),
                "first_seed_probability": o.get("first_seed_pct"),
                "bye_probability": o.get("bye_pct"),
                "miss_playoffs_probability": o.get("miss_pct"),
                "avg_final_wins": o.get("avg_final_wins"),
                "avg_final_losses": o.get("avg_final_losses"),
                "num_simulations": o.get("n_sims") or 10000,
            }
            for o in odds
            if o.get("roster_id") is not None
        ]
        if not cur_norm:
            return {}

        with get_conn() as conn:
            if write:
                for r in cur_norm:
                    conn.execute(
                        """
                        INSERT INTO playoff_odds (
                            league_id, season, week, roster_id, team_name,
                            current_wins, current_losses, current_ties,
                            playoff_probability, first_seed_probability,
                            bye_probability, miss_playoffs_probability,
                            avg_final_wins, avg_final_losses, num_simulations,
                            calculated_at
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            CURRENT_TIMESTAMP
                        )
                        ON CONFLICT (league_id, season, week, roster_id)
                        DO UPDATE SET
                            team_name = EXCLUDED.team_name,
                            current_wins = EXCLUDED.current_wins,
                            current_losses = EXCLUDED.current_losses,
                            current_ties = EXCLUDED.current_ties,
                            playoff_probability = EXCLUDED.playoff_probability,
                            first_seed_probability = EXCLUDED.first_seed_probability,
                            bye_probability = EXCLUDED.bye_probability,
                            miss_playoffs_probability = EXCLUDED.miss_playoffs_probability,
                            avg_final_wins = EXCLUDED.avg_final_wins,
                            avg_final_losses = EXCLUDED.avg_final_losses,
                            num_simulations = EXCLUDED.num_simulations,
                            calculated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            str(league_id), int(season), week, int(r["roster_id"]),
                            r["team_name"], r["current_wins"], r["current_losses"],
                            r["current_ties"], r["playoff_probability"],
                            r["first_seed_probability"], r["bye_probability"],
                            r["miss_playoffs_probability"], r["avg_final_wins"],
                            r["avg_final_losses"], r["num_simulations"],
                        ),
                    )

            prev = conn.execute(
                "SELECT MAX(week) AS w FROM playoff_odds "
                "WHERE league_id = %s AND season = %s AND week < %s",
                (str(league_id), int(season), week),
            ).fetchone()
            prev_week = prev and prev["w"]
            if not prev_week:
                return {}

            prev_rows = conn.execute(
                "SELECT roster_id, playoff_probability, avg_final_wins "
                "FROM playoff_odds "
                "WHERE league_id = %s AND season = %s AND week = %s",
                (str(league_id), int(season), int(prev_week)),
            ).fetchall()

        prev_rank = _rank([dict(r) for r in (prev_rows or [])])
        cur_rank = _rank(cur_norm)
        return {rid: (prev_rank[rid] - cur_rank[rid])
                for rid in cur_rank if rid in prev_rank}
    except Exception:
        # Best-effort: never let snapshotting break the odds endpoint.
        import logging
        logging.getLogger(__name__).debug(
            "playoff_odds_history: record/movement failed", exc_info=True)
        return {}
