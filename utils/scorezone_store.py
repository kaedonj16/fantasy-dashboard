"""Server-side ScoreZone play store.

One elected poller (see ``start_scorezone_store_thread``) fetches play-by-play
upstream every ~15s while NFL games are live and upserts into ``redzone_plays``.
Page loads (``_scorezone_collect``) and the TD notifier read from this table, so
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
    week        INTEGER,
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
    "CREATE INDEX IF NOT EXISTS idx_redzone_plays_week "
    "ON redzone_plays (season, week)",
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


def upsert_plays(season: int, game_id: str, plays: list[dict], week: int | None = None) -> int:
    """Upsert one game's plays. Idempotent; revisions overwrite. Returns rows written."""
    from dashboard_services.db import get_conn

    wk = int(week) if week is not None else None
    rows = [
        (
            int(season),
            str(game_id),
            str(p.get("play_id") or p.get("seq") or ""),
            int(p.get("seq") or 0),
            bool(p.get("is_td")),
            wk,
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
                       (season, game_id, play_id, seq, is_td, week, payload)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                   ON CONFLICT (season, game_id, play_id) DO UPDATE SET
                       seq = EXCLUDED.seq,
                       is_td = EXCLUDED.is_td,
                       -- Never let a week-less re-upsert wipe a stamped week.
                       week = COALESCE(EXCLUDED.week, redzone_plays.week),
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
        logger.warning("[scorezone-store] upsert failed game=%s: %s", game_id, exc)
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
        logger.warning("[scorezone-store] read failed: %s", exc)
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
        logger.warning("[scorezone-store] td-since failed: %s", exc)
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


def get_plays_for_pids(season: int, pids: list[str], days: int = 5, week: int | None = None) -> list[dict]:
    """Plays from the last ``days`` involving any of ``pids``.

    When ``week`` is given, only plays stamped for that NFL week are
    returned (rows collected before week-stamping, i.e. week IS NULL, are
    excluded). Omit it for the legacy recency-only behavior.

    Returns play payload dicts (with game_id attached) ordered by observed_at
    descending. Used by ScoreZone Moments to surface a matchup's big plays.
    """
    from dashboard_services.db import get_conn

    pid_list = [str(p) for p in (pids or []) if p]
    if not pid_list:
        return []
    week_clause = "AND week = %s" if week is not None else ""
    params = [int(season), str(int(days)), pid_list]
    if week is not None:
        params.append(int(week))
    try:
        with get_conn() as conn:
            _ensure_table(conn)
            rows = conn.execute(
                f"""SELECT game_id, payload,
                          EXTRACT(EPOCH FROM observed_at) AS ts
                   FROM redzone_plays
                   WHERE season = %s
                     AND observed_at >= NOW() - (%s || ' days')::INTERVAL
                     AND payload->>'pid' = ANY(%s)
                     {week_clause}
                   ORDER BY observed_at DESC""",
                tuple(params),
            ).fetchall()
    except Exception as exc:
        logger.warning("[scorezone-store] plays-for-pids failed: %s", exc)
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
            payload = dict(payload)
            payload["_observed_ts"] = ts
            out.append(payload)
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
        logger.warning("[scorezone-store] watermark write failed: %s", exc)


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
        logger.warning("[scorezone-store] prune failed: %s", exc)
        return 0


def discover_live_games(current_week: int | None = None) -> list[dict]:
    """League-agnostic live/final game discovery via the ESPN scoreboard.

    Returns [{game_id (Tank01 'YYYYMMDD_AWAY@HOME' form), live, final, week}].

    The scoreboard is fetched for the current NFL week AND the previous week
    explicitly (``?week=N``). Relying on the unparameterized scoreboard alone
    goes blind to last week's games the moment ESPN flips it forward, so
    collection gaps from the previous week could never be backfilled. Each
    game is tagged with the week it was discovered under, so the collector
    stamps the correct week even when backfilling an older game.
    """
    from utils.scorezone_alt_pbp import (
        _ESPN_SCOREBOARD,
        _UA,
        extract_espn_scoreboard_lookup,
    )

    def _fetch(week: int | None) -> dict:
        params = {"xhr": "1"}
        if week:
            params["week"] = str(week)
        try:
            import requests

            resp = requests.get(
                _ESPN_SCOREBOARD,
                params=params,
                headers={"User-Agent": _UA, "Accept": "application/json"},
                timeout=10,
            )
            if resp.status_code != 200:
                return {}
            return extract_espn_scoreboard_lookup(resp.json())
        except Exception as exc:
            logger.debug("[scorezone-store] scoreboard discovery failed: %s", exc)
            return {}

    # (scoreboard week to fetch, NFL week to tag). The explicit week param
    # keeps working before and after ESPN flips its default scoreboard, and
    # the tag tells the collector which week a backfilled game belongs to.
    fetches: list[tuple[int | None, int | None]] = []
    wk = int(current_week) if current_week else 0
    if wk >= 1:
        fetches.append((wk, wk))
        if wk > 1:
            fetches.append((wk - 1, wk - 1))
    else:
        # Week unknown (e.g. state fetch failed): legacy single fetch.
        fetches.append((None, None))

    seen: dict[str, dict] = {}
    for sb_week, tag in fetches:
        for game in _fetch(sb_week).values():
            if not isinstance(game, dict):
                continue
            gid = str(game.get("gameID") or "")
            if not gid or gid in seen:
                continue
            code = str(game.get("gameStatusCode") or "")
            if code not in ("1", "2"):
                continue
            seen[gid] = {
                "game_id": gid,
                "live": code == "1",
                "final": code == "2",
                "week": tag,
            }
    return list(seen.values())


def _build_name_maps(nfl_players: dict, teams: set[str]):
    """pid-resolution maps for one game's teams (mirrors _scorezone_collect)."""
    from utils.scorezone_pbp import _normalize_name, _extract_first_initial_last
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
    from utils.scorezone_alt_pbp import fetch_alt_pbp_plays, parse_tank_game_id

    stats = {"games": 0, "plays": 0}
    try:
        state = get_nfl_state() or {}
        season = int(state.get("season") or 0)
        week = int(state.get("week") or 1)
    except Exception:
        return stats
    if not season:
        return stats

    games = discover_live_games(current_week=week)
    if not games:
        return stats

    # Finals already in the store keep polling for closing-drive catch-up
    # (fetch_alt_pbp_plays force-refreshes until ESPN reports complete).
    # Finals never seen are normally skipped -- nothing new to learn -- but a
    # recent final may have gone final while the poller was down (deploy,
    # outage). Self-heal: backfill unseen finals from the last 7 days so a
    # poller gap can't permanently lose a game's plays (and its TDs). The
    # window matches the play-prune horizon: anything older is pruned anyway.
    live = [g for g in games if g["live"]]
    finals = [g for g in games if g["final"]]
    if finals:
        known = set(get_plays(season, [g["game_id"] for g in finals]).keys())
        try:
            from datetime import datetime, timedelta, timezone
            _cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y%m%d")
        except Exception:
            _cutoff = ""
        def _is_recent_unseen(gid: str) -> bool:
            if not _cutoff:
                return False
            try:
                _date_part = parse_tank_game_id(gid)[0]
            except Exception:
                return False
            return bool(_date_part) and _date_part >= _cutoff
        finals = [
            g for g in finals
            if g["game_id"] in known or _is_recent_unseen(g["game_id"])
        ]

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
            logger.debug("[scorezone-store] pbp failed game=%s: %s", gid, exc)
            continue
        # Stamp the game's own week, not the current NFL week: a backfilled
        # previous-week final must not be stamped as this week, or the
        # week-scoped moments query would serve last week's plays again.
        n = upsert_plays(season, gid, plays, week=g.get("week") or week)
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
            logger.warning("[scorezone-store] db connect failed, retrying: %s", exc)
            time.sleep(30)
            continue
        try:
            conn.autocommit = True
            row = conn.execute(
                "SELECT pg_try_advisory_lock(%s)", (_LEADER_LOCK_KEY,)
            ).fetchone()
            leader = bool(row["pg_try_advisory_lock"])
            if not leader:
                logger.info("[scorezone-store] lock held elsewhere; retrying election in 60s")
                conn.close()
                time.sleep(60)
                continue
            logger.info("[scorezone-store] elected leader; polling every %ss", poll_interval)
            try:
                while True:
                    try:
                        stats = poll_once()
                    except Exception as exc:
                        logger.warning("[scorezone-store] poll iteration failed: %s", exc)
                        stats = {"games": 0, "plays": 0}
                    if stats.get("games"):
                        # Near-instant TD alerts: check the plays just stored.
                        # The 1-min cron remains as backstop + digest flusher.
                        try:
                            from utils.push_notifications import _scorezone_td_check

                            _scorezone_td_check()
                        except Exception as exc:
                            logger.warning("[scorezone-store] td check failed: %s", exc)
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
            logger.warning("[scorezone-store] leader loop error, retrying: %s", exc)
            try:
                conn.close()
            except Exception:
                pass
            time.sleep(30)


def start_scorezone_store_thread() -> None:
    """Start the daemon poller thread. Safe to call in every gunicorn worker;
    the advisory lock ensures only one actually polls."""
    import threading

    t = threading.Thread(
        target=_leader_loop, name="scorezone-store", daemon=True
    )
    t.start()
