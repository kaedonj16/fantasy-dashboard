"""
Per-player per-week usage metrics (snap share, target share, touches).

Season-level snapshots in player_advanced_metrics tell you what a player IS;
weekly rows tell you where he's HEADING. A player averaging 50% snaps could be
declining from 70% or climbing toward 70% — opposite fantasy conclusions that
the season average hides.

Data source: the same cached Sleeper weekly stat files the game logs use
(cache/sleeper_stats/sleeper_stats_s{season}_w{week}.json via fetch_week_stats).
Team target totals are approximated by grouping player targets by the team in
the players index.

Consumers: waivers usage risers, metrics leaderboard trend column, player modal
weekly sparklines, start/sit usage factor, breakout role trajectory.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import json
import os

from dashboard_services.db import get_conn
from data_building.external_data.sleeper_bulk_stats import fetch_week_stats
from data_building.external_data.player_team_history import team_for_week, canon_team
from utils.utils import load_players_index, path_week_schedule

_OPP_MAP_CACHE: Dict[tuple, Dict[str, str]] = {}


def week_opponent_map(season: int, week: int) -> Dict[str, str]:
    """{team_abbr: opponent_abbr} for a season/week from the cached nflverse
    schedule files (see external_data/nflverse_schedules.py). Both directions are
    mapped (home↔away). Empty when the schedule file is missing/unreadable."""
    key = (int(season), int(week))
    cached = _OPP_MAP_CACHE.get(key)
    if cached is not None:
        return cached
    out: Dict[str, str] = {}
    try:
        path = path_week_schedule(int(season), int(week))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                games = json.load(f)
            for g in games or []:
                home, away = g.get("home"), g.get("away")
                if home and away:
                    out[home] = away
                    out[away] = home
    except Exception:
        pass
    _OPP_MAP_CACHE[key] = out
    return out

_POSITIONS = {"QB", "RB", "WR", "TE"}
_TABLE_READY = False


def init_weekly_metrics_db() -> None:
    global _TABLE_READY
    if _TABLE_READY:
        return
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS player_weekly_metrics (
                player_id    TEXT    NOT NULL,
                season       INTEGER NOT NULL,
                week         INTEGER NOT NULL,
                position     TEXT,
                snap_pct     NUMERIC,
                snaps        INTEGER,
                team_snaps   INTEGER,
                targets      INTEGER,
                receptions   INTEGER,
                rec_yards    NUMERIC,
                rec_tds      INTEGER,
                carries      INTEGER,
                rush_yards   NUMERIC,
                rush_tds     INTEGER,
                touches      INTEGER,
                rz_targets   INTEGER,
                rz_carries   INTEGER,
                target_share NUMERIC,
                ppr_pts      NUMERIC,
                pass_att     INTEGER,
                pass_tds     INTEGER,
                PRIMARY KEY (player_id, season, week)
            )
            """
        )
        # Migration for databases created before pass_att existed (needed by the
        # weekly breakout engine's QB passing-opportunity signal). Safe no-op when
        # the column is already present.
        conn.execute(
            "ALTER TABLE player_weekly_metrics ADD COLUMN IF NOT EXISTS pass_att INTEGER"
        )
        for column in ("rec_tds", "rush_tds", "pass_tds", "rz_targets", "rz_carries"):
            conn.execute(
                f"ALTER TABLE player_weekly_metrics ADD COLUMN IF NOT EXISTS {column} INTEGER"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pwm_season_week "
            "ON player_weekly_metrics (season, week)"
        )
    _TABLE_READY = True


def _f(v: Any) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def build_weekly_metrics(season: int, weeks: Optional[List[int]] = None,
                         force_weeks: Optional[List[int]] = None) -> int:
    """Compute and upsert weekly usage rows for the given weeks.

    When `weeks` is None, builds weeks 1-18, skipping weeks already in the DB
    except the two most recent stored weeks (which are rebuilt to pick up
    stat corrections), plus any weeks with missing red-zone columns).

    `force_weeks` names weeks whose Sleeper box-score cache must be refetched even
    when it is already populated. The in-progress week's cache otherwise freezes
    on its first (pre-game / partial) fetch, so the weekly-filter usage rows would
    never reflect a game that finished later the same week. The live refresh
    passes the current week here so a just-final slate lands in the week view now.
    """
    init_weekly_metrics_db()
    idx = load_players_index() or {}
    force_set = {int(w) for w in (force_weeks or [])}

    if weeks is None:
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT week, BOOL_OR(rz_targets IS NULL OR rz_carries IS NULL) AS needs_rz "
                "FROM player_weekly_metrics WHERE season = %s GROUP BY week",
                (int(season),),
            ).fetchall()
        have = sorted(int(r["week"]) for r in rows)
        refresh = set(have[-2:])  # rebuild the latest two stored weeks
        refresh.update(int(r["week"]) for r in rows if r["needs_rz"])
        weeks = [w for w in range(1, 19) if w not in have or w in refresh]

    total = 0
    for week in weeks:
        try:
            # Only pass force when actually forcing, so callers/tests that stub
            # fetch_week_stats with the original 2-arg signature keep working.
            if int(week) in force_set:
                stats = fetch_week_stats(int(season), int(week), force=True) or {}
            else:
                stats = fetch_week_stats(int(season), int(week)) or {}
        except Exception as exc:
            print(f"[weekly_metrics] fetch failed s{season} w{week}: {exc}")
            continue
        if not stats:
            continue

        # Approximate team weekly target totals per team. Use the team the player
        # was on THAT week (team_for_week), not his current team - otherwise a
        # traded player's targets are pooled under his new team, which both
        # mis-computes target share and, when his current team is transiently
        # blank in the index, writes a NULL share that drops him from the
        # target-share leaderboard entirely (raw-stat metrics still show him).
        # Falls back to the current index team when no historical team is known.
        team_targets: Dict[str, float] = {}
        player_rows: List[tuple] = []
        for pid, st in stats.items():
            if not isinstance(st, dict):
                continue
            meta = idx.get(str(pid)) or {}
            pos = (meta.get("pos") or "").upper()
            if pos not in _POSITIONS:
                continue
            tgt = _f(st.get("rec_tgt"))
            team = team_for_week(str(pid), int(season), int(week)) \
                or canon_team(meta.get("team"))
            if team and tgt:
                team_targets[team] = team_targets.get(team, 0.0) + tgt
            player_rows.append((str(pid), pos, team, st))

        count = 0
        with get_conn() as conn:
            for pid, pos, team, st in player_rows:
                snaps = _f(st.get("off_snp"))
                team_snaps = _f(st.get("tm_off_snp"))
                targets = _f(st.get("rec_tgt"))
                carries = _f(st.get("rush_att"))
                pass_att = _f(st.get("pass_att"))
                if snaps <= 0 and targets <= 0 and carries <= 0 and pass_att <= 0:
                    continue  # inactive / no usage
                receptions = _f(st.get("rec"))
                touches = carries + receptions
                snap_pct = round(snaps / team_snaps * 100, 1) if team_snaps > 0 else None
                tgt_share = (
                    round(targets / team_targets[team] * 100, 1)
                    if team and team_targets.get(team) else None
                )
                conn.execute(
                    """
                    INSERT INTO player_weekly_metrics
                        (player_id, season, week, position, snap_pct, snaps, team_snaps,
                         targets, receptions, rec_yards, rec_tds,
                         carries, rush_yards, rush_tds, touches, rz_targets, rz_carries,
                         target_share, ppr_pts, pass_att, pass_tds)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON CONFLICT (player_id, season, week) DO UPDATE SET
                        position = EXCLUDED.position,
                        snap_pct = EXCLUDED.snap_pct,
                        snaps = EXCLUDED.snaps,
                        team_snaps = EXCLUDED.team_snaps,
                        targets = EXCLUDED.targets,
                        receptions = EXCLUDED.receptions,
                        rec_yards = EXCLUDED.rec_yards,
                        rec_tds = EXCLUDED.rec_tds,
                        carries = EXCLUDED.carries,
                        rush_yards = EXCLUDED.rush_yards,
                        rush_tds = EXCLUDED.rush_tds,
                        touches = EXCLUDED.touches,
                        rz_targets = EXCLUDED.rz_targets,
                        rz_carries = EXCLUDED.rz_carries,
                        target_share = EXCLUDED.target_share,
                        ppr_pts = EXCLUDED.ppr_pts,
                        pass_att = EXCLUDED.pass_att,
                        pass_tds = EXCLUDED.pass_tds
                    """,
                    (
                        pid, int(season), int(week), pos, snap_pct, int(snaps),
                        int(team_snaps), int(targets), int(receptions),
                        _f(st.get("rec_yd")), int(_f(st.get("rec_td"))),
                        int(carries), _f(st.get("rush_yd")), int(_f(st.get("rush_td"))),
                        int(touches), int(_f(st.get("rec_rz_tgt"))),
                        int(_f(st.get("rush_rz_att"))), tgt_share,
                        _f(st.get("pts_ppr")), int(pass_att), int(_f(st.get("pass_td"))),
                    ),
                )
                count += 1
        if count:
            print(f"[weekly_metrics] s{season} w{week}: {count} players")
        total += count
    return total


def get_player_weekly_series(player_id: str, season: int) -> List[Dict[str, Any]]:
    """Weekly usage rows for one player, oldest week first."""
    init_weekly_metrics_db()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT week, snap_pct, snaps, team_snaps, targets, receptions, carries,
                   touches, target_share, ppr_pts, rec_yards, rush_yards, pass_att,
                   rec_tds, rush_tds, pass_tds, rz_targets, rz_carries
            FROM player_weekly_metrics
            WHERE player_id = %s AND season = %s
            ORDER BY week
            """,
            (str(player_id), int(season)),
        ).fetchall()
    return [dict(r) for r in rows]


def get_weekly_series_by_player(season: int, through_week: int) -> Dict[str, List[Dict[str, Any]]]:
    """Load the raw scoring universe in one query, including unmatched IDs.

    Iterating the player index first made identity misses invisible and issued a
    query per player.  This raw-first shape is both faster and makes the first
    two pipeline stages auditable.
    """
    init_weekly_metrics_db()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT player_id, week, position, snap_pct, snaps, team_snaps,
                   targets, receptions, carries, touches, target_share, ppr_pts,
                   rec_yards, rush_yards, pass_att, rec_tds, rush_tds, pass_tds,
                   rz_targets, rz_carries
            FROM player_weekly_metrics
            WHERE season=%s AND week <= %s
            ORDER BY player_id, week
            """,
            (int(season), int(through_week)),
        ).fetchall()
    out: Dict[str, List[Dict[str, Any]]] = {}
    for raw in rows:
        row = dict(raw)
        player_id = str(row.pop("player_id"))
        out.setdefault(player_id, []).append(row)
    return out


def get_usage_trends(season: int) -> Dict[str, Dict[str, Any]]:
    """Per-player usage trend map for the season.

    For each player with 2+ active weeks, returns the last-6-week series for
    the position's key usage stat (QB: snap %, RB: touches, WR/TE: targets),
    plus the delta of the last-3-week average vs the season average, expressed
    in the stat's own units. Positive delta = usage is rising.
    """
    init_weekly_metrics_db()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT player_id, week, position, snap_pct, targets, touches, target_share
            FROM player_weekly_metrics
            WHERE season = %s
            ORDER BY player_id, week
            """,
            (int(season),),
        ).fetchall()

    by_pid: Dict[str, List[dict]] = {}
    for r in rows:
        by_pid.setdefault(str(r["player_id"]), []).append(dict(r))

    out: Dict[str, Dict[str, Any]] = {}
    for pid, weeks in by_pid.items():
        if len(weeks) < 2:
            continue
        pos = (weeks[-1].get("position") or "").upper()
        stat = "snap_pct" if pos == "QB" else ("touches" if pos == "RB" else "targets")
        vals = [float(w.get(stat) or 0) for w in weeks]
        snap_vals = [float(w.get("snap_pct") or 0) for w in weeks if w.get("snap_pct") is not None]

        season_avg = sum(vals) / len(vals)
        recent = vals[-3:]
        recent_avg = sum(recent) / len(recent)
        snap_delta = None
        if len(snap_vals) >= 2:
            snap_season = sum(snap_vals) / len(snap_vals)
            snap_recent = sum(snap_vals[-3:]) / len(snap_vals[-3:])
            snap_delta = round(snap_recent - snap_season, 1)

        out[pid] = {
            "position": pos,
            "stat": stat,
            "series": [round(v, 1) for v in vals[-6:]],
            "series_weeks": [int(w["week"]) for w in weeks[-6:]],
            "season_avg": round(season_avg, 1),
            "recent_avg": round(recent_avg, 1),
            "delta": round(recent_avg - season_avg, 1),
            "snap_delta": snap_delta,
            "weeks_played": len(weeks),
        }
    return out


def get_recent_momentum(player_id: str, season: int) -> Optional[float]:
    """Snap-share momentum for one player: last-3-week avg minus season avg,
    in percentage points. None when there isn't enough weekly data."""
    series = get_player_weekly_series(player_id, season)
    snaps = [float(w["snap_pct"]) for w in series if w.get("snap_pct") is not None]
    if len(snaps) < 3:
        return None
    season_avg = sum(snaps) / len(snaps)
    recent_avg = sum(snaps[-3:]) / 3
    return round(recent_avg - season_avg, 1)
