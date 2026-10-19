#!/usr/bin/env python3
"""Recompute stored dominator_rating values onto the receiving basis.

Background (model v3.5): dominator_rating was defined as share of
TOTAL offense - (receiving + rushing) over (netPassingYards +
rushingYards) and (passingTDs + rushingTDs). For WR/TE that basis
structurally taxes the player: the denominator includes rushing
production he cannot touch, so Jeremiah Smith's 2024 (1,315 yards,
15 TD for Ohio State) stored 0.208 while the conventional
receiving-share basis computes 0.345 from the same season - and the
model's own breakout anchors (0.35 saturation, 0.4877 recorded best)
were always receiving-basis numbers, so inputs and anchors sat on
different bases. v3.5 redefines the WR/TE rating as receiving share:

    0.65 * (receiving_yards / team_pass_yards)
  + 0.35 * (receiving_tds  / team_pass_tds)

(ingestion.compute_dominator_rating is the single source of truth;
it prices new/derived rows on this basis from v3.5 on.) RB ratings
are scrimmage share and DO NOT CHANGE - this script never writes an
RB (or QB) row, enforced in the planner, in an assert on the write
path, and in tests.

What it does: for every stored WR/TE season row in
rookie_prospect_source_data (source != 'rookie_eval'), recompute the
rating from the row's stored team pass totals (team_pass_yards /
team_pass_tds) when both are stored; when either is missing, fetch
the totals from CFBD /stats/season for that (season, team) - the
pipeline's own source - and write them back with the rating so the
row is self-consistent (the same write-back pattern as
scripts/backfill_stored_dominators.py). A row whose pass totals
exist nowhere is reported with the missing input named and left
untouched - missing data is never fabricated into a rating.

Safety:
- Dry-run by default; nothing is written without --write.
- Fails closed: no DATABASE_URL -> exit 1, nothing touched. If any
  candidate row needs CFBD pass totals and CFBD_API_KEY is not set
  -> exit 1, nothing touched (no partial recomputes).
- Idempotent: a row already on the receiving basis recomputes to
  within rounding of its stored value and is classified unchanged;
  --write only updates rows whose value moves by > 0.0005.

Usage:
    python scripts/recompute_dominators.py [--dry-run]
    python scripts/recompute_dominators.py --write
    python scripts/recompute_dominators.py --player-id ROOKIE_2027_jeremiah_smith

On Render (web shell), after the v3.5 merge + backtest gate:
    python3 scripts/recompute_dominators.py
    nohup python3 scripts/recompute_dominators.py --write > /tmp/recompute_dominators.log 2>&1 &
    tail -f /tmp/recompute_dominators.log
Run scripts/build_reference_population.py AFTER this script so its
stored advanced marks carry the corrected basis.
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
log = logging.getLogger("recompute_dominators")

_CANDIDATE_SQL = """
SELECT sd.player_id, sd.season, sd.source, sd.team, p.position,
       sd.receiving_yards, sd.receiving_tds,
       sd.rush_yards, sd.rush_tds,
       sd.team_pass_yards, sd.team_pass_tds, sd.dominator_rating
FROM rookie_prospect_source_data sd
LEFT JOIN rookie_prospects p ON p.player_id = sd.player_id
WHERE sd.source != 'rookie_eval'
ORDER BY sd.player_id, sd.season
"""

_UPDATE_SQL = """
UPDATE rookie_prospect_source_data
SET dominator_rating = %s,
    team_pass_yards = COALESCE(NULLIF(team_pass_yards, 0), %s),
    team_pass_tds   = COALESCE(team_pass_tds, %s)
WHERE player_id = %s AND season = %s AND source = %s
"""

_EPSILON = 0.0005


def _f(v):
    """DB numerics arrive as Decimal/int/None; normalize to float|None."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _complete_pass_totals(team_pass_yards, team_pass_tds) -> bool:
    """Both denominators present and positive: the row can be priced
    exactly from its own stored inputs."""
    return bool((team_pass_yards or 0) > 0 and (team_pass_tds or 0) > 0)


def plan_recompute(rows, cfbd_pass_totals):
    """Pure recompute planner (unit-tested; no I/O).

    rows: stored season rows (dicts with the _CANDIDATE_SQL columns,
    position joined from rookie_prospects). cfbd_pass_totals:
    {(season, team_lower): (team_pass_yards, team_pass_tds)} fetched
    from CFBD, consulted only for rows whose stored pass totals are
    incomplete.

    Returns (changed, unchanged, unrepairable, skipped):
    - changed: WR/TE rows whose recomputed receiving-basis rating
      differs from the stored one by > 0.0005 (old may be None),
      ready to write.
    - unchanged: WR/TE rows already within rounding of the
      receiving-basis value.
    - unrepairable: WR/TE rows whose team pass totals exist neither
      stored complete nor on CFBD (or whose position is unknown);
      "missing" names the absent input exactly.
    - skipped: rows at any other position. RB dominator is the
      scrimmage definition and is never recomputed; these rows are
      counted, never written.
    """
    from data_building.rookie_pipeline.ingestion import compute_dominator_rating

    changed, unchanged, unrepairable, skipped = [], [], [], []
    for row in rows:
        pos = str(row.get("position") or "").upper()
        base = {
            "player_id": row.get("player_id"),
            "season": row.get("season"),
            "source": row.get("source"),
            "team": row.get("team"),
            "position": pos or None,
        }
        if pos not in ("WR", "TE"):
            if not pos:
                unrepairable.append({**base, "missing": (
                    "position: no rookie_prospects row for this "
                    "player_id, so no dominator basis can be chosen")})
            else:
                skipped.append(base)
            continue

        stored_py = _f(row.get("team_pass_yards"))
        stored_pt = _f(row.get("team_pass_tds"))
        totals_source = None
        if _complete_pass_totals(stored_py, stored_pt):
            t_py, t_pt, totals_source = stored_py, stored_pt, "stored"
        else:
            key = (int(row.get("season") or 0),
                   str(row.get("team") or "").strip().lower())
            fetched = (cfbd_pass_totals or {}).get(key)
            if fetched and _complete_pass_totals(fetched[0], fetched[1]):
                t_py, t_pt, totals_source = fetched[0], fetched[1], "cfbd"
        if totals_source is None:
            missing_bits = []
            if not (stored_py or 0) > 0:
                missing_bits.append("team_pass_yards")
            if not (stored_pt or 0) > 0:
                missing_bits.append("team_pass_tds")
            unrepairable.append({**base, "missing": (
                "team pass totals: stored "
                + (" and ".join(missing_bits) if missing_bits
                   else "team_pass_yards/team_pass_tds")
                + " incomplete and CFBD /stats/season returned no "
                f"passing totals for team '{row.get('team')}' "
                f"season {row.get('season')}")})
            continue

        new_dom = compute_dominator_rating(
            _f(row.get("receiving_yards")), _f(row.get("rush_yards")),
            _f(row.get("receiving_tds")), _f(row.get("rush_tds")),
            None, None,
            position=pos, team_pass_yards=t_py, team_pass_tds=t_pt)
        if new_dom is None:
            unrepairable.append({**base, "missing": (
                "team pass totals: resolved totals were not positive")})
            continue
        old_dom = _f(row.get("dominator_rating"))
        entry = {
            **base,
            "old": old_dom,
            "new": new_dom,
            "totals_source": totals_source,
            "team_pass_yards": int(t_py) if t_py is not None else None,
            "team_pass_tds": int(t_pt) if t_pt is not None else None,
        }
        if old_dom is not None and abs(new_dom - old_dom) <= _EPSILON:
            unchanged.append(entry)
        else:
            changed.append(entry)
    return changed, unchanged, unrepairable, skipped


def _fetch_cfbd_pass_totals(needed):
    """Fetch team PASSING totals from CFBD /stats/season for
    {(season, team)} pairs, via the pipeline's own client and stat
    names (netPassingYards / passingTDs - the denominators
    _build_cfbd_season stamps as team_pass_yards / team_pass_tds).
    Returns {(season, team_lower): (pass_yards, pass_tds)} for the
    requested teams only."""
    from data_building.rookie_pipeline.ingestion import _cfbd_get, _safe

    by_season = {}
    for season, _team in needed:
        by_season.setdefault(int(season), set()).add(_team)
    out = {}
    for season, teams in sorted(by_season.items()):
        data = _cfbd_get("/stats/season",
                         {"year": season, "seasonType": "both"})
        if not data:
            log.warning("[recompute] CFBD returned no team stats for %s", season)
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
            out[(season, key)] = (stats.get("netPassingYards", 0) or 0,
                                  stats.get("passingTDs", 0) or 0)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="apply the recomputes (default is a dry run)")
    ap.add_argument("--player-id", default=None,
                    help="restrict to one player_id (e.g. a Smith check)")
    ap.add_argument("--season", type=int, default=None,
                    help="restrict to one season year")
    args = ap.parse_args(argv)

    # Fail closed: no database, no run.
    if not os.getenv("DATABASE_URL", "").strip():
        log.error("[recompute] DATABASE_URL is not set - failing closed, nothing touched.")
        return 1

    from dashboard_services.db import get_conn

    sql = _CANDIDATE_SQL
    params = []
    if args.player_id:
        sql = sql.replace("ORDER BY", "AND sd.player_id = %s ORDER BY")
        params.append(args.player_id)
    if args.season:
        sql = sql.replace("ORDER BY", "AND sd.season = %s ORDER BY")
        params.append(int(args.season))

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = [dict(r) for r in cur.fetchall()]

    if not rows:
        log.info("[recompute] No stored season rows. Nothing to do.")
        return 0

    # WR/TE rows whose stored pass totals are incomplete need CFBD
    # before ANY write decision: without the key the run fails closed
    # rather than recomputing a subset.
    need_cfbd = {
        (int(r.get("season") or 0), str(r.get("team") or ""))
        for r in rows
        if str(r.get("position") or "").upper() in ("WR", "TE")
        and not _complete_pass_totals(_f(r.get("team_pass_yards")),
                                      _f(r.get("team_pass_tds")))
    }
    cfbd_pass_totals = {}
    if need_cfbd:
        if not os.getenv("CFBD_API_KEY", "").strip():
            log.error("[recompute] %d WR/TE row(s) lack complete stored "
                      "team pass totals and CFBD_API_KEY is not set - "
                      "failing closed, nothing touched. Rows: %s",
                      len(need_cfbd),
                      ", ".join(f"{season} '{team}'" for season, team in sorted(need_cfbd)))
            return 1
        cfbd_pass_totals = _fetch_cfbd_pass_totals(need_cfbd)

    changed, unchanged, unrepairable, skipped = plan_recompute(rows, cfbd_pass_totals)

    for entry in changed:
        old = "NULL" if entry["old"] is None else f"{entry['old']:.4f}"
        log.info("[recompute] %s %s (%s): dominator %s -> %.4f [totals: %s]",
                 entry["player_id"], entry["season"], entry["team"],
                 old, entry["new"], entry["totals_source"])
    for bad in unrepairable:
        log.info("[recompute] SKIP %s %s (%s): cannot recompute - missing input: %s",
                 bad["player_id"], bad["season"], bad["team"], bad["missing"])
    skipped_by_pos = {}
    for s in skipped:
        skipped_by_pos[s["position"]] = skipped_by_pos.get(s["position"], 0) + 1
    log.info("[recompute] rows=%d changed=%d unchanged=%d unrepairable=%d "
             "skipped_non_wr_te=%d %s",
             len(rows), len(changed), len(unchanged), len(unrepairable),
             len(skipped),
             {k: skipped_by_pos[k] for k in sorted(skipped_by_pos)})

    if not args.write:
        log.info("[recompute] DRY RUN - no rows written. Re-run with --write to apply.")
        return 0

    written = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            for entry in changed:
                # Defense in depth: the planner only emits WR/TE
                # changes; an RB/QB row reaching this loop is a bug,
                # and its scrimmage-basis rating must never be
                # overwritten by a receiving-basis value.
                assert entry["position"] in ("WR", "TE"), entry
                cur.execute(_UPDATE_SQL, (
                    entry["new"], entry["team_pass_yards"], entry["team_pass_tds"],
                    entry["player_id"], entry["season"], entry["source"],
                ))
                written += cur.rowcount or 0
        conn.commit()
    log.info("[recompute] WROTE %d row(s). Re-running is a no-op for these rows.", written)
    return 0


if __name__ == "__main__":
    sys.exit(main())
