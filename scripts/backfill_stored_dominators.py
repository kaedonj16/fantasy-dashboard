#!/usr/bin/env python3
"""Backfill stored dominator_rating values that were never computed.

Background (the defect, PR fixing the write path): a season row in
rookie_prospect_source_data was persisted WITHOUT a dominator_rating
whenever the CFBD team-totals map was empty for that season at build
time (ingestion._build_cfbd_season computes the rating only from the
in-memory team totals). The incremental fetch gate then treated the
row's mere existence as coverage, so past seasons were never
refetched and the NULL froze - while the current season, refreshed
every run, picked up a rating. Jeremiah Smith is the visible case:
his stored 2024/2025 rows carry no dominator, so both the modal's
breakout age and the model's breakout component read his first
"breakout" as 2026. The write path now derives the rating from a
row's own stored inputs (ingestion.derive_season_dominator); this
script repairs the rows already stored.

What it does: for every stored season row that has real player
production but a missing (NULL) or zero dominator, recompute the
rating with the pipeline's existing formula
(ingestion.compute_dominator_rating - the formula is NOT changed)
from the stored player lines plus team totals. Team totals come
from the row itself when stored; when the row's totals are also
missing, they are fetched from CFBD /stats/season for that
(season, team) - the pipeline's own source - and written back with
the rating so the row is self-consistent.

Safety:
- Dry-run by default; nothing is written without --write.
- Fails closed: no DATABASE_URL -> exit 1, nothing touched. If any
  candidate row needs CFBD team totals and CFBD_API_KEY is not set
  -> exit 1, nothing touched (no partial repairs).
- Idempotent / resumable: repaired rows stop matching the candidate
  predicate, and every UPDATE is guarded on the row still lacking a
  dominator, so re-running is a no-op for finished rows.
- A row whose team totals exist nowhere (not stored, not on CFBD)
  is reported with the missing input named and left untouched -
  missing data is never fabricated into a rating.

Usage:
    python scripts/backfill_stored_dominators.py [--dry-run]
    python scripts/backfill_stored_dominators.py --write
    python scripts/backfill_stored_dominators.py --player-id ROOKIE_2027_jeremiah_smith

On Render (web shell), dry run first, then the write detached:
    python3 scripts/backfill_stored_dominators.py
    nohup python3 scripts/backfill_stored_dominators.py --write > /tmp/backfill_dominators.log 2>&1 &
    tail -f /tmp/backfill_dominators.log
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("backfill_stored_dominators")

_CANDIDATE_SQL = """
SELECT player_id, season, source, team,
       receptions, receiving_yards, receiving_tds,
       rush_attempts, rush_yards, rush_tds,
       team_total_yards, team_total_tds, dominator_rating
FROM rookie_prospect_source_data
WHERE source != 'rookie_eval'
  AND (dominator_rating IS NULL OR dominator_rating = 0)
  AND (COALESCE(receiving_yards, 0) + COALESCE(rush_yards, 0)
     + COALESCE(receiving_tds, 0) + COALESCE(rush_tds, 0)) > 0
ORDER BY player_id, season
"""

_UPDATE_SQL = """
UPDATE rookie_prospect_source_data
SET dominator_rating = %s,
    team_total_yards = COALESCE(NULLIF(team_total_yards, 0), %s),
    team_total_tds   = COALESCE(NULLIF(team_total_tds, 0), %s)
WHERE player_id = %s AND season = %s AND source = %s
  AND (dominator_rating IS NULL OR dominator_rating = 0)
"""


def _f(v):
    """DB numerics arrive as Decimal/int/None; normalize to float|None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _has_totals(team_total_yards, team_total_tds) -> bool:
    return bool((team_total_yards or 0) or (team_total_tds or 0))


def plan_repairs(rows, cfbd_totals):
    """Pure repair planner (unit-tested; no I/O).

    rows: candidate stored season rows (dicts with the _CANDIDATE_SQL
    columns). cfbd_totals: {(season, team_lower): (team_total_yards,
    team_total_tds)} fetched from CFBD, consulted only for rows whose
    stored totals are missing.

    Returns (repairs, unrepairable, unchanged):
    - repairs: dicts ready to write (old/new dominator, the totals
      used, and where they came from).
    - unrepairable: rows whose team totals exist neither stored nor
      on CFBD; "missing" names the absent input exactly.
    - unchanged: rows whose recomputed rating equals the stored one
      (a genuine 0.0 - production is real but rounds to no share).
    """
    from data_building.rookie_pipeline.ingestion import compute_dominator_rating

    repairs, unrepairable, unchanged = [], [], []
    for row in rows:
        # Candidacy mirrors the SQL predicate; guard here too so the
        # planner is safe on any row set it is handed.
        _dom = _f(row.get("dominator_rating"))
        if _dom is not None and _dom != 0:
            continue
        _prod = sum(_f(row.get(k)) or 0 for k in (
            "receiving_yards", "rush_yards", "receiving_tds", "rush_tds"))
        if _prod <= 0:
            continue
        stored_yds = _f(row.get("team_total_yards"))
        stored_tds = _f(row.get("team_total_tds"))
        totals_source = None
        if _has_totals(stored_yds, stored_tds):
            t_yds, t_tds, totals_source = stored_yds, stored_tds, "stored"
        else:
            key = (int(row.get("season") or 0),
                   str(row.get("team") or "").strip().lower())
            fetched = (cfbd_totals or {}).get(key)
            if fetched and _has_totals(fetched[0], fetched[1]):
                t_yds, t_tds, totals_source = fetched[0], fetched[1], "cfbd"
        if totals_source is None:
            unrepairable.append({
                "player_id": row.get("player_id"),
                "season": row.get("season"),
                "source": row.get("source"),
                "team": row.get("team"),
                "missing": (
                    "team totals: stored team_total_yards/team_total_tds "
                    "are empty and CFBD /stats/season returned no totals "
                    f"for team '{row.get('team')}' season {row.get('season')}"
                ),
            })
            continue
        new_dom = compute_dominator_rating(
            _f(row.get("receiving_yards")), _f(row.get("rush_yards")),
            _f(row.get("receiving_tds")), _f(row.get("rush_tds")),
            t_yds, t_tds)
        old_dom = _f(row.get("dominator_rating"))
        entry = {
            "player_id": row.get("player_id"),
            "season": row.get("season"),
            "source": row.get("source"),
            "team": row.get("team"),
            "old": old_dom,
            "new": new_dom,
            "totals_source": totals_source,
            "team_total_yards": int(t_yds) if t_yds is not None else None,
            "team_total_tds": int(t_tds) if t_tds is not None else None,
        }
        if new_dom is not None and old_dom is not None and new_dom == old_dom:
            unchanged.append(entry)
        else:
            repairs.append(entry)
    return repairs, unrepairable, unchanged


def _fetch_cfbd_team_totals(needed):
    """Fetch team totals from CFBD /stats/season for {(season, team)}
    pairs, via the pipeline's own client and stat names (the same
    totals _build_cfbd_season consumes: netPassingYards + rushingYards,
    passingTDs + rushingTDs). Returns {(season, team_lower): (yds, tds)}
    for the requested teams only."""
    from data_building.rookie_pipeline.ingestion import _cfbd_get, _safe

    by_season = {}
    for season, _team in needed:
        by_season.setdefault(int(season), set()).add(_team)
    out = {}
    for season, teams in sorted(by_season.items()):
        data = _cfbd_get("/stats/season",
                         {"year": season, "seasonType": "both"})
        if not data:
            log.warning("[backfill] CFBD returned no team stats for %s", season)
            continue
        grouped = {}
        for r in data:
            t = r.get("team", "")
            grouped.setdefault(t, {})[r.get("statName", "")] = _safe(r.get("statValue"), 0)
        wanted = {str(t).strip().lower(): t for t in teams}
        for cfbd_team, stats in grouped.items():
            key = str(cfbd_team).strip().lower()
            if key not in wanted:
                continue
            t_yds = (stats.get("netPassingYards", 0) or 0) + (stats.get("rushingYards", 0) or 0)
            t_tds = (stats.get("passingTDs", 0) or 0) + (stats.get("rushingTDs", 0) or 0)
            out[(season, key)] = (t_yds, t_tds)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="apply the repairs (default is a dry run)")
    ap.add_argument("--player-id", default=None,
                    help="restrict to one player_id (e.g. a Smith check)")
    ap.add_argument("--season", type=int, default=None,
                    help="restrict to one season year")
    args = ap.parse_args(argv)

    # Fail closed: no database, no run.
    if not os.getenv("DATABASE_URL", "").strip():
        log.error("[backfill] DATABASE_URL is not set - failing closed, nothing touched.")
        return 1

    from dashboard_services.db import get_conn

    sql = _CANDIDATE_SQL
    params = []
    if args.player_id:
        sql = sql.replace("ORDER BY", "AND player_id = %s ORDER BY")
        params.append(args.player_id)
    if args.season:
        sql = sql.replace("ORDER BY", "AND season = %s ORDER BY")
        params.append(int(args.season))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        log.info("[backfill] No candidate rows (every stored season with "
                 "production already carries a dominator). Nothing to do.")
        return 0

    # Rows whose stored totals are missing need CFBD before ANY write
    # decision: without the key the run fails closed rather than
    # repairing a subset.
    need_cfbd = {
        (int(r.get("season") or 0), str(r.get("team") or ""))
        for r in rows
        if not _has_totals(_f(r.get("team_total_yards")), _f(r.get("team_total_tds")))
    }
    cfbd_totals = {}
    if need_cfbd:
        if not os.getenv("CFBD_API_KEY", "").strip():
            log.error("[backfill] %d candidate row(s) lack stored team totals "
                      "and CFBD_API_KEY is not set - failing closed, nothing "
                      "touched. Rows: %s",
                      len(need_cfbd),
                      ", ".join(f"{season} '{team}'" for season, team in sorted(need_cfbd)))
            return 1
        cfbd_totals = _fetch_cfbd_team_totals(need_cfbd)

    repairs, unrepairable, unchanged = plan_repairs(rows, cfbd_totals)

    for rep in repairs:
        old = "NULL" if rep["old"] is None else f"{rep['old']:.4f}"
        log.info("[backfill] %s %s (%s): dominator %s -> %.4f [totals: %s]",
                 rep["player_id"], rep["season"], rep["team"],
                 old, rep["new"], rep["totals_source"])
    for bad in unrepairable:
        log.info("[backfill] SKIP %s %s (%s): cannot recompute - missing input: %s",
                 bad["player_id"], bad["season"], bad["team"], bad["missing"])
    log.info("[backfill] candidates=%d repairs=%d unrepairable=%d unchanged=%d",
             len(rows), len(repairs), len(unrepairable), len(unchanged))

    if not args.write:
        log.info("[backfill] DRY RUN - no rows written. Re-run with --write to apply.")
        return 0

    written = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for rep in repairs:
                cur.execute(_UPDATE_SQL, (
                    rep["new"], rep["team_total_yards"], rep["team_total_tds"],
                    rep["player_id"], rep["season"], rep["source"],
                ))
                written += cur.rowcount or 0
        conn.commit()
    log.info("[backfill] WROTE %d row(s). Re-running is a no-op for these rows.", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
