from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Iterable

from dashboard_services.db import get_conn


def init_value_history_db() -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS player_value_history (
                    as_of_date DATE NOT NULL,
                    player_id TEXT NOT NULL,
                    name TEXT,
                    position TEXT,
                    team TEXT,
                    value NUMERIC NOT NULL,
                    source TEXT NOT NULL DEFAULT 'model',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (as_of_date, player_id, source)
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_player_value_history_player_date
                ON player_value_history (player_id, as_of_date DESC)
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_player_value_history_date
                ON player_value_history (as_of_date DESC)
                """
            )


def record_model_value_snapshot(
        players: Iterable[dict],
        *,
        as_of: Optional[date] = None,
        source: str = "model",
        ema_alpha: float = 0.70,
        min_change_pct: float = 0.005,
) -> int:
    """
    Write a smoothed daily value snapshot using EMA blending.

    ema_alpha: weight for new value (0.70 = 70% new, 30% previous).
      Softens step-function jumps when the model is retrained.
    min_change_pct: skip writing if the blended value changed less than
      this fraction from the previous snapshot (reduces DB noise).
    """
    init_value_history_db()

    snapshot_date = (as_of or date.today()).isoformat()

    player_list = []
    for p in players or []:
        if not isinstance(p, dict):
            continue
        pid = str(p.get("id") or "").strip()
        if not pid:
            continue
        try:
            value = float(p.get("value") or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        player_list.append((pid, p.get("name"), p.get("position"), p.get("team"), value))

    if not player_list:
        return 0

    # Fetch the most recent previous value for every player in one batch query
    all_pids = [row[0] for row in player_list]
    prev_values: dict[str, float] = {}
    with get_conn() as conn:
        # Get the latest value per player before today using a lateral/distinct-on query
        rows = conn.execute(
            """
            SELECT DISTINCT ON (player_id)
                player_id, value
            FROM player_value_history
            WHERE source = %s
              AND player_id = ANY(%s)
              AND as_of_date < %s
            ORDER BY player_id, as_of_date DESC
            """,
            (source, all_pids, snapshot_date),
        ).fetchall()
        for r in rows:
            prev_values[r["player_id"]] = float(r["value"])

    # Build smoothed rows, skipping tiny changes
    rows_to_insert: list[tuple] = []
    for pid, name, position, team, new_val in player_list:
        prev = prev_values.get(pid)
        if prev is not None and prev > 0:
            blended = ema_alpha * new_val + (1.0 - ema_alpha) * prev
            # Skip if change from previous is below threshold
            if abs(blended - prev) / prev < min_change_pct:
                continue
        else:
            blended = new_val

        rows_to_insert.append((snapshot_date, pid, name, position, team, round(blended, 2), source))

    if not rows_to_insert:
        return 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO player_value_history
                    (as_of_date, player_id, name, position, team, value, source)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT(as_of_date, player_id, source)
                DO UPDATE SET
                    name = excluded.name,
                    position = excluded.position,
                    team = excluded.team,
                    value = excluded.value
                """,
                rows_to_insert,
            )

    return len(rows_to_insert)


def get_latest_snapshot_date(source: str = "model") -> Optional[str]:
    init_value_history_db()
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT MAX(as_of_date) AS latest_date
            FROM player_value_history
            WHERE source = %s
            """,
            (source,),
        ).fetchone()
    return row["latest_date"] if row and row["latest_date"] else None


def get_player_value_history(
        player_id: str,
        *,
        days: int = 30,
        source: str = "model",
) -> list[dict]:
    init_value_history_db()

    latest_date = get_latest_snapshot_date(source=source)
    if not latest_date:
        return []

    cutoff = (date.fromisoformat(latest_date) - timedelta(days=max(days, 1) - 1)).isoformat()

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                as_of_date,
                player_id,
                name,
                position,
                team,
                value,
                source
            FROM player_value_history
            WHERE source = %s
              AND player_id = %s
              AND as_of_date >= %s
            ORDER BY as_of_date ASC
            """,
            (source, str(player_id), cutoff),
        ).fetchall()

    out: list[dict] = []
    prev_val: Optional[float] = None
    for r in rows:
        val = float(r["value"])
        delta = None if prev_val is None else round(val - prev_val, 1)
        out.append(
            {
                "as_of_date": r["as_of_date"],
                "player_id": r["player_id"],
                "name": r["name"],
                "position": r["position"],
                "team": r["team"],
                "value": round(val, 1),
                "delta_from_prev": delta,
                "source": r["source"],
            }
        )
        prev_val = val

    return out


def load_latest_value_snapshot(source: str = "model") -> list[dict]:
    """
    Load the most recent value snapshot from the database.
    Returns a list of player dicts in the same format as the JSON file.
    Includes computed fields like search_name and pos_rank.
    """
    from utils.utils import normalize_name

    init_value_history_db()

    latest_date = get_latest_snapshot_date(source=source)
    if not latest_date:
        return []

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                player_id as id,
                name,
                position,
                team,
                value,
                sf_value,
                value_8,
                value_12,
                value_14,
                sf_value_8,
                sf_value_12,
                sf_value_14
            FROM player_value_history
            WHERE source = %s
              AND as_of_date = %s
            ORDER BY value DESC
            LIMIT 600
            """,
            (source, latest_date),
        ).fetchall()

    # Convert to dicts and add computed fields
    players = []
    for row in rows:
        player = dict(row)
        # Add search_name for fuzzy matching
        player["search_name"] = normalize_name(player.get("name", ""))
        players.append(player)

    # Calculate position ranks
    pos_to_indices = {}
    for idx, player in enumerate(players):
        pos = str(player.get("position") or "").upper()
        if not pos or pos == "PICK":
            continue
        pos_to_indices.setdefault(pos, []).append(idx)

    # Standard position ranks (by value)
    for pos, indices in pos_to_indices.items():
        indices.sort(key=lambda i: float(players[i].get("value") or 0.0), reverse=True)
        rank = 1
        for i in indices:
            players[i]["pos_rank"] = rank
            players[i]["pos_rank_label"] = f"{pos}{rank}"
            rank += 1

    # Superflex position ranks (by sf_value)
    sf_pos_to_indices = {}
    for idx, player in enumerate(players):
        pos = str(player.get("position") or "").upper()
        if not pos or pos == "PICK":
            continue
        sf_pos_to_indices.setdefault(pos, []).append(idx)

    for pos, indices in sf_pos_to_indices.items():
        indices.sort(key=lambda i: float(players[i].get("sf_value") or 0.0), reverse=True)
        rank = 1
        for i in indices:
            players[i]["sf_pos_rank"] = rank
            players[i]["sf_pos_rank_label"] = f"{pos}{rank}"
            rank += 1

    # Enrich with age from players_index
    try:
        from utils.utils import load_players_index
        from dashboard_services.service import age_from_bday
        players_index = load_players_index() or {}
        for player in players:
            pid = str(player.get("id") or player.get("player_id") or "")
            if pid and pid != "" and pid in players_index:
                # Use age_from_bday function for consistent 1 decimal place precision
                bday_str = players_index[pid].get("bDay")
                player["age"] = age_from_bday(bday_str)
            else:
                player["age"] = None
    except Exception as e:
        print(f"[load_latest_value_snapshot] Failed to enrich ages: {e}")
        for player in players:
            player["age"] = None

    return players


def load_current_values_from_db() -> list[dict]:
    """
    Load current player values from the player_values table (one row per player,
    updated daily by cron_daily).  Falls back gracefully if the table doesn't
    exist yet or the DB is unavailable.
    """
    from utils.utils import normalize_name

    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        player_id  AS id,
                        COALESCE(calibrated_value_1qb, value_1qb) AS value,
                        COALESCE(calibrated_value_sf,  value_sf)  AS sf_value,
                        value_1qb  AS model_value,
                        value_sf   AS model_sf_value,
                        calibration_source,
                        calibration_weight,
                        position,
                        pos_rank,
                        pos_rank_label,
                        age,
                        team,
                        years_exp,
                        last_updated,
                        rank_change_7d,
                        pos_rank_change_7d
                    FROM player_values
                    ORDER BY COALESCE(calibrated_value_1qb, value_1qb) DESC NULLS LAST
                    LIMIT 800
                    """
                )
                rows = cur.fetchall()
    except Exception as e:
        print(f"[load_current_values_from_db] Query failed: {e}")
        return []

    if not rows:
        return []

    players = []
    for row in rows:
        player = dict(row)
        # Normalise field names expected by the rest of the app
        player.setdefault("sf_value", player.get("value") or 0.0)
        player["search_name"] = normalize_name(str(player.get("id") or ""))
        # Add league-size variants as the current value (no per-size data in player_values)
        val    = float(player.get("value")    or 0.0)
        sf_val = float(player.get("sf_value") or 0.0)
        for sz in (8, 12, 14):
            player.setdefault(f"value_{sz}",    val)
            player.setdefault(f"sf_value_{sz}", sf_val)
        players.append(player)

    # Compute pos_rank if not already stored
    if players and players[0].get("pos_rank") is None:
        from collections import defaultdict
        pos_groups: dict[str, list[int]] = defaultdict(list)
        for i, p in enumerate(players):
            pos = str(p.get("position") or "").upper()
            if pos and pos != "PICK":
                pos_groups[pos].append(i)
        for pos, idxs in pos_groups.items():
            idxs.sort(key=lambda i: float(players[i].get("value") or 0.0), reverse=True)
            for rank, i in enumerate(idxs, 1):
                players[i]["pos_rank"] = rank
                players[i]["pos_rank_label"] = f"{pos}{rank}"

    return players


def load_calibration_overrides() -> dict[str, dict]:
    """
    Return {player_id: {value, sf_value}} for every player that has been
    market-calibrated.  Used to overlay trade-data-adjusted values on top
    of the raw model values without touching the model pipeline.
    Falls back to empty dict if the DB is unavailable or the columns don't exist.
    """
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT player_id,
                       calibrated_value_1qb  AS value,
                       COALESCE(calibrated_value_sf, calibrated_value_1qb) AS sf_value
                FROM player_values
                WHERE calibrated_value_1qb IS NOT NULL
                  AND calibrated_value_1qb > 0
                """
            ).fetchall()
        return {
            r["player_id"]: {
                "value":    float(r["value"]),
                "sf_value": float(r["sf_value"]),
            }
            for r in rows
        }
    except Exception:
        return {}


def get_top_movers(
        *,
        days: int = 7,
        limit: int = 15,
        source: str = "model",
) -> dict:
    """
    Try requested window first (ex: 7 days).
    If no baseline exists, fall back to 6, then 5, ... down to 1.
    """
    init_value_history_db()

    latest_date = get_latest_snapshot_date(source=source)
    if not latest_date:
        return {
            "latest_date": None,
            "comparison_date": None,
            "requested_days": days,
            "used_days": None,
            "risers": [],
            "fallers": [],
        }

    max_days = max(int(days), 1)

    with get_conn() as conn:
        with conn.cursor() as cur:
            comparison_date = None
            used_days = None

            for candidate_days in range(max_days, 0, -1):
                target_date = latest_date - timedelta(days=candidate_days)

                cur.execute(
                    """
                    SELECT MAX(as_of_date) AS comparison_date
                    FROM player_value_history
                    WHERE source = %s
                      AND as_of_date <= %s
                    """,
                    (source, target_date),
                )
                row = cur.fetchone()
                candidate_date = row["comparison_date"] if row else None

                if candidate_date and candidate_date < latest_date:
                    comparison_date = candidate_date
                    used_days = candidate_days
                    break

            if comparison_date is None:
                return {
                    "latest_date": latest_date.isoformat(),
                    "comparison_date": None,
                    "requested_days": max_days,
                    "used_days": None,
                    "risers": [],
                    "fallers": [],
                }

            cur.execute(
                """
                WITH latest_rows AS (
                    SELECT DISTINCT ON (player_id)
                        player_id,
                        name,
                        position,
                        team,
                        value,
                        as_of_date
                    FROM player_value_history
                    WHERE source = %s
                      AND as_of_date = %s
                    ORDER BY player_id, as_of_date DESC
                ),
                baseline_rows AS (
                    SELECT DISTINCT ON (player_id)
                        player_id,
                        value,
                        as_of_date
                    FROM player_value_history
                    WHERE source = %s
                      AND as_of_date = %s
                    ORDER BY player_id, as_of_date DESC
                )
                SELECT
                    l.player_id,
                    l.name,
                    l.position,
                    l.team,
                    ROUND(b.value::numeric, 1) AS old_value,
                    ROUND(l.value::numeric, 1) AS new_value,
                    ROUND((l.value - b.value)::numeric, 1) AS delta
                FROM latest_rows l
                JOIN baseline_rows b
                  ON b.player_id = l.player_id
                ORDER BY delta DESC, new_value DESC
                """
                , (source, latest_date, source, comparison_date))

            rows = cur.fetchall()

    # Load players index to get names for records with NULL/empty names
    from utils.utils import load_players_index
    players_index = load_players_index() or {}

    movers = []
    for row in rows:
        row_dict = dict(row)
        # If name is None or empty, get it from players_index
        if not row_dict.get("name") or row_dict.get("name") == "Unknown":
            player_info = players_index.get(str(row_dict["player_id"])) or {}
            row_dict["name"] = player_info.get("name", "Unknown")
        movers.append(row_dict)
    
    risers = movers[:limit]
    fallers = sorted(movers, key=lambda x: (x["delta"], x["new_value"]))[:limit]

    return {
        "latest_date": latest_date.isoformat(),
        "comparison_date": comparison_date.isoformat(),
        "requested_days": max_days,
        "used_days": used_days,
        "risers": risers,
        "fallers": fallers,
    }
