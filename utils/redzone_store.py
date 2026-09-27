"""Server-side RedZone play store.

One elected poller (see ``start_redzone_store_thread``) fetches play-by-play
upstream every ~15s while NFL games are live and upserts into ``redzone_plays``.
Page loads (``_redzone_collect``) and the TD notifier read from this table, so
upstream ESPN/Tank01 is hit once per interval total -- never once per viewer
per worker.

Plays are stored exactly as ``fetch_alt_pbp_plays`` returns them (pid-resolved
via the global player index, which is league-agnostic), so readers get the same
shape the client already consumes.
"""
from __future__ import annotations

import json
import logging
import time

logger = logging.getLogger(__name__)

_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS redzone_plays (
    season      INTEGER NOT NULL,
    game_id     TEXT NOT NULL,
    play_id     TEXT NOT NULL,
    seq         INTEGER NOT NULL DEFAULT 0,
    is_td       BOOLEAN NOT NULL DEFAULT FALSE,
    payload     JSONB NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (season, game_id, play_id)
)
"""
_INDEX_DDL = [
    "CREATE INDEX IF NOT EXISTS idx_redzone_plays_game_seq "
    "ON redzone_plays (season, game_id, seq)",
    "CREATE INDEX IF NOT EXISTS idx_redzone_plays_td_seen "
    "ON redzone_plays (season, observed_at) WHERE is_td",
    "CREATE INDEX IF NOT EXISTS idx_redzone_plays_observed "
    "ON redzone_plays (observed_at)",
]

# Advisory-lock key electing the single store poller across gunicorn workers.
# Fixed bigint; pg_try_advisory_lock is per-connection, so the leader holds a
# dedicated connection for the life of the thread. If the leader dies the
# connection drops, the lock releases, and another worker takes over.
_LEADER_LOCK_KEY = 934172001

_WATERMARK_KEY = "redzone_td_poll_watermark"


# Process-level guard: the migration creates the table at deploy; the in-code
# DDL is a safety net for environments where it hasn't run yet. Run it once
# per database per process instead of on every store call.
_ENSURED_TABLES: set = set()


def _ensure_table(conn) -> None:
    try:
        dbname = getattr(getattr(conn, "info", None), "dbname", None)
    except Exception:
        dbname = None
    key = dbname or "default"
    if key in _ENSURED_TABLES:
        return
    conn.execute(_TABLE_DDL)
    for ddl in _INDEX_DDL:
        conn.execute(ddl)
    _ENSURED_TABLES.add(key)


def upsert_plays(season: int, game_id: str, plays: list[dict]) -> int:
    """Upsert one game's plays. Idempotent; revisions overwrite. Returns rows written."""
    from dashboard_services.db import get_conn

    rows = [
        (
            int(season),
            str(game_id),
            str(p.get("play_id") or p.get("seq") or ""),
            int(p.get("seq") or 0),
            bool(p.get("is_td")),
            json.dumps(p),
        )
        for p in (plays or [])
        if isinstance(p, dict) and (p.get("play_id") or p.get("seq") is not None)
    ]
    if not rows:
        return 0
    try:
        with get_conn() as conn:
            _ensure_table(conn)
            # psycopg3 Connections have no executemany; it lives on Cursor.
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO redzone_plays
                       (season, game_id, play_id, seq, is_td, payload)
                   VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                   ON CONFLICT (season, game_id, play_id) DO UPDATE SET
                       seq = EXCLUDED.seq,
                       is_td = EXCLUDED.is_td,
                       payload = EXCLUDED.payload,
                       -- Only re-stamp when the play actually changed (a
                       -- revision or a scoring flip). Unchanged re-upserts
                       -- must not look "new" to the TD watermark query.
                       observed_at = CASE
                           WHEN redzone_plays.payload IS DISTINCT FROM EXCLUDED.payload
                           THEN NOW() ELSE redzone_plays.observed_at END""",
                    rows,
                )
            conn.commit()
        return len(rows)
    except Exception as exc:
        logger.warning("[redzone-store] upsert failed game=%s: %s", game_id, exc)
        return 0


def get_plays(season: int, game_ids: list[str]) -> dict[str, list[dict]]:
    """Return {game_id: [play dicts]} ordered by seq. Missing games -> absent keys."""
    from dashboard_services.db import get_conn

    gids = [str(g) for g in (game_ids or []) if g]
    if not gids:
        return {}
    try:
        with get_conn() as conn:
            _ensure_table(conn)
            rows = conn.execute(
                """SELECT game_id, payload FROM redzone_plays
                   WHERE season = %s AND game_id = ANY(%s)
                   ORDER BY game_id, seq""",
                (int(season), gids),
            ).fetchall()
    except Exception as exc:
        logger.warning("[redzone-store] read failed: %s", exc)
        return {}
    out: dict[str, list[dict]] = {}
    for r in rows:
        gid = r["game_id"] if isinstance(r, dict) else r[0]
        payload = r["payload"] if isinstance(r, dict) else r[1]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue
        if isinstance(payload, dict):
            out.setdefault(str(gid), []).append(payload)
    return out


def get_td_plays_since(season: int, since_ts: float) -> list[tuple[str, dict, float]]:
    """TD plays first observed after ``since_ts`` (epoch seconds).

    Returns [(game_id, play_dict, observed_at_epoch)]. Watermark callers on the
    max observed_at, never wall-clock now, so a play stored mid-check can't be
    skipped by the watermark advancing past it.
    """
    from dashboard_services.db import get_conn

    try:
        with get_conn() as conn:
            _ensure_table(conn)
            rows = conn.execute(
                """SELECT game_id, payload,
                          EXTRACT(EPOCH FROM observed_at) AS ts
                   FROM redzone_plays
                   WHERE season = %s AND is_td
                     AND observed_at > to_timestamp(%s)
                   ORDER BY observed_at""",
                (int(season), float(since_ts)),
            ).fetchall()
    except Exception as exc:
        logger.warning("[redzone-store] td-since failed: %s", exc)
        return []
    out = []
    for r in rows:
        gid = r["game_id"] if isinstance(r, dict) else r[0]
        payload = r["payload"] if isinstance(r, dict) else r[1]
        ts = r["ts"] if isinstance(r, dict) else r[2]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue
        if isinstance(payload, dict):
            try:
                ts = float(ts or 0)
            except Exception:
                ts = 0.0
            out.append((str(gid), payload, ts))
    return out


def get_watermark() -> float:
    from dashboard_services.db import get_conn

    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT value FROM app_state WHERE key = %s", (_WATERMARK_KEY,)
            ).fetchone()
    except Exception:
        return 0.0
    try:
        return float((row["value"] if isinstance(row, dict) else row[0]) or 0)
    except Exception:
        return 0.0


def set_watermark(ts: float) -> None:
    from dashboard_services.db import get_conn

    try:
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO app_state (key, value) VALUES (%s, %s)
                   ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value""",
                (_WATERMARK_KEY, str(float(ts))),
            )
            conn.commit()
    except Exception as exc:
        logger.warning("[redzone-store] watermark write failed: %s", exc)


def prune_plays(retention_days: int = 7) -> int:
    """Delete plays older than ``retention_days``. Returns rows deleted."""
    from dashboard_services.db import get_conn

    try:
        with get_conn() as conn:
            _ensure_table(conn)
            row = conn.execute(
                "DELETE FROM redzone_plays WHERE observed_at < NOW() - (%s || ' days')::interval RETURNING 1",
                (str(int(retention_days)),),
            ).fetchall()
            conn.commit()
            return len(row)
    except Exception as exc:
        logger.warning("[redzone-store] prune failed: %s", exc)
        return 0


def discover_live_games() -> list[dict]:
    """League-agnostic live/final game discovery via the ESPN scoreboard.

    Returns [{game_id (Tank01 'YYYYMMDD_AWAY@HOME' form), live, final}].
    """
    from utils.redzone_alt_pbp import (
        _ESPN_SCOREBOARD,
        _UA,
        extract_espn_scoreboard_lookup,
    )

    try:
        import requests

        resp = requests.get(
            _ESPN_SCOREBOARD,
            params={"xhr": "1"},
            headers={"User-Agent": _UA, "Accept": "application/json"},
            timeout=10,
        )
        if resp.status_code != 200:
            return []
        lookup = extract_espn_scoreboard_lookup(resp.json())
    except Exception as exc:
        logger.debug("[redzone-store] scoreboard discovery failed: %s", exc)
        return []
    seen: dict[str, dict] = {}
    for game in lookup.values():
        if not isinstance(game, dict):
            continue
        gid = str(game.get("gameID") or "")
        if not gid or gid in seen:
            continue
        code = str(game.get("gameStatusCode") or "")
        if code not in ("1", "2"):
            continue
        seen[gid] = {"game_id": gid, "live": code == "1", "final": code == "2"}
    return list(seen.values())


def _build_name_maps(nfl_players: dict, teams: set[str]):
    """pid-resolution maps for one game's teams (mirrors _redzone_collect)."""
    from utils.redzone_pbp import _normalize_name, _extract_first_initial_last
    from utils.utils import canon_team

    wanted = {str(t).upper() for t in teams if t}
    name_to_pid: dict[str, str] = {}
    player_meta_by_pid: dict[str, dict] = {}
    for pid, p in (nfl_players or {}).items():
        if not isinstance(p, dict):
            continue
        team = canon_team(p.get("team")) or ""
        if str(team).upper() not in wanted:
            continue
        full_name = p.get("full_name") or ""
        pos = p.get("position", "")
        if not full_name:
            continue
        if pos != "DEF":
            player_meta_by_pid[str(pid)] = {
                "name": full_name,
                "team": team,
                "position": pos,
                **{
                    k: p.get(k)
                    for k in (
                        "player_id", "sleeper_id", "tank01_id", "espn_id",
                        "yahoo_id", "mfl_id", "fleaflicker_id", "gsis_id",
                        "sportradar_id",
                    )
                    if p.get(k) not in (None, "")
                },
            }
        normalized = _normalize_name(str(full_name).lower())
        name_to_pid[normalized] = str(pid)
        abbrev = _extract_first_initial_last(str(full_name).lower())
        if abbrev and abbrev != normalized:
            name_to_pid[abbrev] = str(pid)
        parts = str(full_name).split()
        if len(parts) >= 2:
            fi, last = parts[0][0], " ".join(parts[1:])
            for alias in (f"{fi}.{last}", f"{fi}. {last}", f"{fi} {last}"):
                name_to_pid[_normalize_name(alias)] = str(pid)
    return name_to_pid, player_meta_by_pid


def poll_once() -> dict:
    """One store iteration: discover games, fetch PBP, upsert. Returns stats."""
    from dashboard_services.api import get_nfl_players, get_nfl_state
    from utils.redzone_alt_pbp import fetch_alt_pbp_plays, parse_tank_game_id

    stats = {"games": 0, "plays": 0}
    try:
        state = get_nfl_state() or {}
        season = int(state.get("season") or 0)
        week = int(state.get("week") or 1)
    except Exception:
        return stats
    if not season:
        return stats

    games = discover_live_games()
    if not games:
        return stats

    # Finals already in the store keep polling for closing-drive catch-up
    # (fetch_alt_pbp_plays force-refreshes until ESPN reports complete);
    # finals never seen are skipped -- nothing new to learn.
    live = [g for g in games if g["live"]]
    finals = [g for g in games if g["final"]]
    if finals:
        known = set(get_plays(season, [g["game_id"] for g in finals]).keys())
        finals = [g for g in finals if g["game_id"] in known]

    nfl_players = get_nfl_players() or {}
    for g in live + finals:
        gid = g["game_id"]
        try:
            _date_part, away, home = parse_tank_game_id(gid)
        except Exception:
            continue
        if not away or not home:
            continue
        try:
            name_to_pid, meta = _build_name_maps(nfl_players, {away, home})
            plays = fetch_alt_pbp_plays(
                gid,
                season=season,
                week=week,
                name_to_pid=name_to_pid,
                player_meta_by_pid=meta,
                live=g["live"],
                final=g["final"],
                providers=("espn", "sleeper"),
            ) or []
        except Exception as exc:
            logger.debug("[redzone-store] pbp failed game=%s: %s", gid, exc)
            continue
        n = upsert_plays(season, gid, plays)
        stats["games"] += 1
        stats["plays"] += n
    return stats


def _leader_loop(poll_interval: float = 15.0, idle_interval: float = 60.0) -> None:
    """Run forever on the advisory-lock leader; losers retry election.

    Uses a dedicated (non-pooled) connection: the lock is bound to the
    session, so holding it must not consume a pool slot forever.
    """
    import psycopg
    from psycopg.rows import dict_row

    from dashboard_services.db import get_database_url

    prune_at = 0.0
    while True:
        try:
            conn = psycopg.connect(get_database_url(), row_factory=dict_row)
        except Exception as exc:
            logger.warning("[redzone-store] db connect failed, retrying: %s", exc)
            time.sleep(30)
            continue
        try:
            conn.autocommit = True
            row = conn.execute(
                "SELECT pg_try_advisory_lock(%s)", (_LEADER_LOCK_KEY,)
            ).fetchone()
            leader = bool(row["pg_try_advisory_lock"])
            if not leader:
                logger.info("[redzone-store] lock held elsewhere; retrying election in 60s")
                conn.close()
                time.sleep(60)
                continue
            logger.info("[redzone-store] elected leader; polling every %ss", poll_interval)
            try:
                while True:
                    try:
                        stats = poll_once()
                    except Exception as exc:
                        logger.warning("[redzone-store] poll iteration failed: %s", exc)
                        stats = {"games": 0, "plays": 0}
                    if stats.get("games"):
                        # Near-instant TD alerts: check the plays just stored.
                        # The 1-min cron remains as backstop + digest flusher.
                        try:
                            from utils.push_notifications import _redzone_td_check

                            _redzone_td_check()
                        except Exception as exc:
                            logger.warning("[redzone-store] td check failed: %s", exc)
                    now = time.time()
                    if now >= prune_at:
                        try:
                            prune_plays()
                        except Exception:
                            pass
                        prune_at = now + 86400
                    time.sleep(poll_interval if stats.get("games") else idle_interval)
            finally:
                try:
                    conn.execute("SELECT pg_advisory_unlock(%s)", (_LEADER_LOCK_KEY,))
                except Exception:
                    pass
                conn.close()
        except Exception as exc:
            logger.warning("[redzone-store] leader loop error, retrying: %s", exc)
            try:
                conn.close()
            except Exception:
                pass
            time.sleep(30)


def start_redzone_store_thread() -> None:
    """Start the daemon poller thread. Safe to call in every gunicorn worker;
    the advisory lock ensures only one actually polls."""
    import threading

    t = threading.Thread(
        target=_leader_loop, name="redzone-store", daemon=True
    )
    t.start()
