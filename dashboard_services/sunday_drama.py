"""Sunday drama: live lead-change moments and archived turning points.

While games are live, each poll of the Weekly Hub matchup slide records a
throttled win-probability snapshot for the matchup. Comparing the current
poll's favorite against the last recorded snapshot lets the slide play an
in-page "lead change" moment when the favorite flips sides. The snapshot
trail also keeps the biggest win-probability swings as archived turning
points, viewable on the slide after the week finalizes (the "when did I
actually lose this matchup" view).

Storage is the managed Postgres (Render has no persistent disk), declared in
migrations/038_matchup_moments.sql and created idempotently here too. The
win-probability model itself is untouched; this module only observes its
output. Every public entry point swallows its own failures: drama must never
break a matchup render.
"""

import html
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)

_ET = ZoneInfo("America/New_York")

# Breadcrumb cadence: at most one snapshot per matchup per gap unless the
# favorite flips (a flip always records) or the probability moves enough to
# matter. Keeps a Sunday of 60s polls to a few dozen small rows per matchup.
_SNAPSHOT_MIN_GAP_S = 600
_SNAPSHOT_MIN_MOVE = 0.02

# A favorite only "flips" when the leader changes sides AND the swing clears
# this deadband, so 49.9/50.1 rounding flicker does not play a moment.
_FLIP_MIN_DELTA = 0.03

# Consecutive-snapshot swings smaller than this never make the archived
# turning-points list.
_TURNING_POINT_MIN_SWING = 0.05

_TABLE_READY = False


def _get_conn():
    """Lazy DB handle: importing this module (e.g. under the pure test suite,
    which has no psycopg) must not pull in the driver until a query runs."""
    from dashboard_services.db import get_conn as _gc

    return _gc()


def ensure_table() -> None:
    """Create the matchup_moments table if it is missing (mirrors
    migrations/038_matchup_moments.sql; post-deploy applies that file, this
    covers any path that reaches the table first)."""
    global _TABLE_READY
    if _TABLE_READY:
        return
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS matchup_moments (
                id BIGSERIAL PRIMARY KEY,
                league_id TEXT NOT NULL,
                season TEXT NOT NULL,
                week INTEGER NOT NULL,
                matchup_key TEXT NOT NULL,
                left_roster_id TEXT NOT NULL,
                right_roster_id TEXT NOT NULL,
                left_name TEXT,
                right_name TEXT,
                observed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                left_win_prob DOUBLE PRECISION NOT NULL,
                left_pts DOUBLE PRECISION NOT NULL DEFAULT 0,
                right_pts DOUBLE PRECISION NOT NULL DEFAULT 0,
                games_live INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_matchup_moments_lookup
                ON matchup_moments (league_id, season, week, matchup_key, observed_at)
            """
        )
    _TABLE_READY = True


def reset_table_cache() -> None:
    """Test hook: forget the ensure_table memo so a fresh fake conn is used."""
    global _TABLE_READY
    _TABLE_READY = False


def matchup_key_for(m: Optional[dict]) -> str:
    """Stable per-week matchup key. Prefers the provider matchup id; falls
    back to the sorted roster-id pair so leagues without one still key."""
    m = m or {}
    mid = m.get("matchup_id")
    if mid not in (None, ""):
        return f"mid:{mid}"
    rids = sorted([
        str(((m.get("left") or {}).get("roster_id")) or ""),
        str(((m.get("right") or {}).get("roster_id")) or ""),
    ])
    return f"pair:{rids[0]}-{rids[1]}"


def leader_of(left_prob: float) -> str:
    return "left" if float(left_prob) >= 0.5 else "right"


def detect_flip(
    prev_prob: Optional[float],
    cur_prob: Optional[float],
    min_delta: float = _FLIP_MIN_DELTA,
) -> Optional[Dict[str, Any]]:
    """Return {"from", "to", "delta"} when the favorite changed sides between
    two polls with a swing clearing the deadband; otherwise None."""
    if prev_prob is None or cur_prob is None:
        return None
    prev_leader, cur_leader = leader_of(prev_prob), leader_of(cur_prob)
    if prev_leader == cur_leader:
        return None
    if abs(float(cur_prob) - float(prev_prob)) < min_delta:
        return None
    return {"from": prev_leader, "to": cur_leader, "delta": float(cur_prob) - float(prev_prob)}


def _latest_snapshot(conn, league_id: str, season: str, week: int, matchup_key: str) -> Optional[dict]:
    rows = conn.execute(
        """
        SELECT observed_at, left_win_prob, left_name, right_name
          FROM matchup_moments
         WHERE league_id = %s AND season = %s AND week = %s AND matchup_key = %s
         ORDER BY observed_at DESC, id DESC
         LIMIT 1
        """,
        (league_id, season, week, matchup_key),
    ).fetchall()
    return dict(rows[0]) if rows else None


def _observed_age_s(row: dict) -> float:
    ts = row.get("observed_at")
    if not isinstance(ts, datetime):
        return float("inf")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - ts).total_seconds()


def record_snapshot(
    *,
    league_id: str,
    season: str,
    week: int,
    matchup_key: str,
    left_roster_id: str,
    right_roster_id: str,
    left_name: str,
    right_name: str,
    left_win_prob: float,
    left_pts: float,
    right_pts: float,
    games_live: int = 1,
) -> Optional[Dict[str, Any]]:
    """Record one win-probability snapshot for a live matchup.

    Throttled: the first snapshot always records, a favorite flip always
    records, otherwise at most one row per _SNAPSHOT_MIN_GAP_S unless the
    probability moved by _SNAPSHOT_MIN_MOVE.

    Returns {"flip": {...}|None, "leader": "left"|"right"} on a recorded row,
    None when throttled or on any failure.
    """
    try:
        ensure_table()
    except Exception:
        log.debug("sunday drama ensure_table failed", exc_info=True)
        return None
    try:
        with _get_conn() as conn:
            prev = _latest_snapshot(conn, league_id, season, week, matchup_key)
            flip = None
            if prev is not None:
                flip = detect_flip(prev.get("left_win_prob"), left_win_prob)
                if flip is None:
                    moved = abs(float(left_win_prob) - float(prev.get("left_win_prob") or 0.0))
                    if _observed_age_s(prev) < _SNAPSHOT_MIN_GAP_S and moved < _SNAPSHOT_MIN_MOVE:
                        return None
            conn.execute(
                """
                INSERT INTO matchup_moments
                    (league_id, season, week, matchup_key,
                     left_roster_id, right_roster_id, left_name, right_name,
                     left_win_prob, left_pts, right_pts, games_live)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    league_id, season, int(week), matchup_key,
                    left_roster_id, right_roster_id, left_name, right_name,
                    float(left_win_prob), float(left_pts), float(right_pts), int(games_live),
                ),
            )
            return {"flip": flip, "leader": leader_of(left_win_prob),
                    "observed_at": datetime.now(timezone.utc)}
    except Exception:
        log.debug("sunday drama record_snapshot failed", exc_info=True)
        return None


def get_turning_points(
    league_id: str,
    season: str,
    week: int,
    matchup_key: str,
    limit: int = 4,
    min_swing: float = _TURNING_POINT_MIN_SWING,
) -> List[Dict[str, Any]]:
    """Archived turning points: the largest win-probability swings between
    consecutive snapshots for a matchup, oldest first. Each entry carries the
    timestamp, swing, which side it favored, and the live-projected score at
    that moment. Returns [] on any failure."""
    try:
        ensure_table()
    except Exception:
        log.debug("sunday drama ensure_table failed", exc_info=True)
        return []
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                """
                SELECT observed_at, left_win_prob, left_pts, right_pts,
                       left_name, right_name
                  FROM matchup_moments
                 WHERE league_id = %s AND season = %s AND week = %s AND matchup_key = %s
                 ORDER BY observed_at ASC, id ASC
                """,
                (league_id, season, week, matchup_key),
            ).fetchall()
    except Exception:
        log.debug("sunday drama get_turning_points failed", exc_info=True)
        return []
    rows = [dict(r) for r in rows]
    swings: List[Dict[str, Any]] = []
    for prev, cur in zip(rows, rows[1:]):
        try:
            swing = float(cur["left_win_prob"]) - float(prev["left_win_prob"])
        except (TypeError, ValueError, KeyError):
            continue
        if abs(swing) < min_swing:
            continue
        swings.append(
            {
                "at": cur.get("observed_at"),
                "swing": swing,
                "before": float(prev["left_win_prob"]),
                "after": float(cur["left_win_prob"]),
                "to": leader_of(cur["left_win_prob"]),
                "left_name": cur.get("left_name") or prev.get("left_name") or "Left team",
                "right_name": cur.get("right_name") or prev.get("right_name") or "Right team",
                "left_pts": cur.get("left_pts") or 0.0,
                "right_pts": cur.get("right_pts") or 0.0,
            }
        )
    top = sorted(swings, key=lambda s: abs(s["swing"]), reverse=True)[: max(1, int(limit))]
    return sorted(top, key=lambda s: (str(s["at"]),))


def format_moment_time(dt: Any) -> str:
    """'Sun 1:24 PM' in America/New_York; '' when unparseable."""
    if not isinstance(dt, datetime):
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.astimezone(_ET).strftime("%a %-I:%M %p")
    except Exception:
        return ""


def flip_banner_html(new_leader_name: str, new_leader_pct: int,
                     observed_at: Any = None) -> str:
    """In-page lead-change moment. data-br-moment="leadchange" plays the pop
    through the existing moment machinery when it scrolls into view. The
    timestamp is when the flip was first observed (this poll)."""
    name = html.escape(str(new_leader_name or ""))
    try:
        pct = int(new_leader_pct)
    except (TypeError, ValueError):
        pct = 0
    pct = max(0, min(100, pct))
    when = format_moment_time(observed_at) if observed_at is not None else ""
    when_html = f"<span class=\"m-drama-flip-time\">{html.escape(when)}</span>" if when else ""
    return (
        "<div class=\"m-drama-flip\" data-br-moment=\"leadchange\" role=\"status\">"
        "<span class=\"m-drama-flip-tag\">Lead change</span>"
        f"{when_html}"
        f"<span class=\"m-drama-flip-text\">{name} now favored at {pct}%</span>"
        "</div>"
    )


def turning_points_html(points: List[Dict[str, Any]]) -> str:
    """Archived turning points for a finalized week."""
    items = []
    for p in points:
        when = html.escape(format_moment_time(p.get("at")))
        swing = float(p.get("swing") or 0.0)
        before = round(float(p.get("before") or 0.0) * 100)
        after = round(float(p.get("after") or 0.0) * 100)
        to = p.get("to")
        name = html.escape(str(p.get("left_name") if to == "left" else p.get("right_name") or ""))
        crossed = (before < 50) != (after < 50)
        if crossed:
            verb = "took the lead"
        else:
            # Swing moved the leader's number up (pulled away) or down (fell back).
            leader_gained = (to == "left") == (swing > 0)
            verb = "pulled away" if leader_gained else "fell back"
        try:
            lp = float(p.get("left_pts") or 0.0)
            rp = float(p.get("right_pts") or 0.0)
            score = f"{lp:.1f} to {rp:.1f}"
        except (TypeError, ValueError):
            score = ""
        score_html = f" <span class=\"m-drama-tp-score\">{html.escape(score)}</span>" if score else ""
        items.append(
            "<li>"
            f"<span class=\"m-drama-tp-time\">{when}</span> "
            f"<b>{name}</b> {verb} "
            f"<span class=\"m-drama-tp-wp\">({before}% to {after}%)</span>"
            f"{score_html}"
            "</li>"
        )
    return (
        "<div class=\"m-drama-history\">"
        "<div class=\"m-drama-history-title\">Turning points</div>"
        f"<ul class=\"m-drama-tp-list\">{''.join(items)}</ul>"
        "</div>"
    )
