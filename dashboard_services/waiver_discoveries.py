"""League-agnostic "unexpected big game" discovery pipeline.

Glue between the existing weekly ingestion (data_building.weekly_metrics, which
already writes player_weekly_metrics from the cached Sleeper stat files) and the
shared, pure detector in utils.waiver_big_game. It:

  * saves a *pregame* projection snapshot (source + timestamp) so a game can be
    honestly graded against what was expected going in, never a number revised
    after the game (item 6/8);
  * assembles a GameContext per player from this week's usage row plus the prior
    weeks' rows as a clearly-labeled historical baseline;
  * runs the shared detector and upserts the result idempotently keyed by
    (season, week, player_id), so re-running during a game, after final stats,
    and after a stat correction updates one row rather than creating duplicates
    (item 8).

The shared performance/role assessment here is deliberately league-independent
(raw usage + PPR points as the fantasy proxy). League-specific availability and
roster fit are layered on by the waiver route, not here.

Data limitations (honest fallbacks, item 8):
  * player_weekly_metrics does not carry touchdowns or longest-play yardage, so
    the TD-dependence / one-big-play cautions only fire when a caller supplies
    those via ``extra_features``; otherwise they're left unknown, not fabricated.
  * fantasy points use PPR (pts_ppr) as a shared proxy; a caller with the raw
    stat line and league scoring can override per-player via ``league_points``.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from utils.waiver_big_game import (
    BigGameAssessment,
    GameContext,
    assess_big_game,
    discovery_key,
    merge_assessment,
)

logger = logging.getLogger(__name__)

_SNAP_TABLE_READY = False
_DISC_TABLE_READY = False


# ---------------------------------------------------------------------------
# Pure transforms (no DB — unit-testable)
# ---------------------------------------------------------------------------

def _mean(vals) -> Optional[float]:
    vals = [float(v) for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def historical_baseline(prior_rows: List[dict]) -> dict:
    """A clearly-labeled fantasy-point baseline from a player's prior weeks.

    Uses the trailing weeks' PPR points. Returns ppg, the sample size, and a
    source label so callers never present an invented baseline as established
    (item 6). Empty prior history -> ppg None, games 0.
    """
    pts = [r.get("ppr_pts") for r in (prior_rows or []) if r.get("ppr_pts") is not None]
    if not pts:
        return {"ppg": None, "games": 0, "source": "none"}
    # Prefer a trailing-4 window when there's enough history, else season-to-date.
    window = pts[-4:] if len(pts) >= 4 else pts
    src = "trailing4" if len(pts) >= 4 else "season_avg"
    return {"ppg": _mean(window), "games": len(pts), "source": src}


def game_context_from_rows(player_id: str, position: str, season: int, week: int,
                           week_row: dict, prior_rows: List[dict], *,
                           pregame: Optional[dict] = None,
                           position_baseline_ppg: Optional[float] = None,
                           teammate_out: bool = False,
                           teammate_out_share: Optional[float] = None,
                           is_rookie: bool = False,
                           returning_from_injury: bool = False,
                           league_points: Optional[float] = None,
                           extra_features: Optional[dict] = None,
                           status: str = "final") -> GameContext:
    """Build a GameContext from plain metric rows (no DB). ``week_row`` is this
    game's player_weekly_metrics row; ``prior_rows`` are earlier weeks this
    season. snap_pct / target_share are stored 0-100 and converted to 0-1 here.

    ``pregame`` is ``{"pts", "source", "saved_at"}`` from the saved snapshot when
    available (the honest expectation); otherwise a labeled historical baseline is
    used. ``extra_features`` may supply touchdowns / longest_play_yards /
    yards_per_touch that the weekly table doesn't carry.
    """
    week_row = week_row or {}
    extra = extra_features or {}
    base = historical_baseline(prior_rows)

    def _pct(v):
        return (float(v) / 100.0) if v is not None else None

    actual = league_points if league_points is not None else week_row.get("ppr_pts")
    rec_y = week_row.get("rec_yards") or 0
    rush_y = week_row.get("rush_yards") or 0
    total_yards = (float(rec_y) + float(rush_y)) if (rec_y or rush_y) else None

    limited = base["games"] <= 3

    return GameContext(
        player_id=str(player_id),
        position=str(position or "").upper(),
        season=int(season),
        week=int(week),
        actual_points=(float(actual) if actual is not None else None),
        pregame_projection=(pregame or {}).get("pts"),
        projection_source=(pregame or {}).get("source"),
        projection_saved_at=(pregame or {}).get("saved_at"),
        baseline_ppg=base["ppg"],
        baseline_source=base["source"] if base["ppg"] is not None else None,
        baseline_games=base["games"],
        position_baseline_ppg=position_baseline_ppg,
        snap_share=_pct(week_row.get("snap_pct")),
        snap_share_prev=_pct(_mean([r.get("snap_pct") for r in prior_rows])),
        targets=week_row.get("targets"),
        targets_prev=_mean([r.get("targets") for r in prior_rows]),
        target_share=_pct(week_row.get("target_share")),
        target_share_prev=_pct(_mean([r.get("target_share") for r in prior_rows])),
        routes=week_row.get("routes"),
        routes_prev=_mean([r.get("routes") for r in prior_rows]),
        carries=week_row.get("carries"),
        carries_prev=_mean([r.get("carries") for r in prior_rows]),
        pass_attempts=week_row.get("pass_att"),
        pass_attempts_prev=_mean([r.get("pass_att") for r in prior_rows]),
        touches=week_row.get("touches"),
        touches_prev=_mean([r.get("touches") for r in prior_rows]),
        redzone_touches=(extra.get("redzone_touches") if extra.get("redzone_touches") is not None
                         else _sum_known(week_row.get("rz_targets"), week_row.get("rz_carries"))),
        total_yards=total_yards,
        touchdowns=extra.get("touchdowns"),
        td_points=extra.get("td_points"),
        longest_play_yards=extra.get("longest_play_yards"),
        yards_per_touch=extra.get("yards_per_touch"),
        yards_per_route=extra.get("yards_per_route"),
        teammate_out=teammate_out,
        teammate_out_share=teammate_out_share,
        is_rookie=is_rookie,
        returning_from_injury=returning_from_injury,
        limited_history=limited,
        status=status,
    )


def _sum_known(*values) -> Optional[float]:
    known = [float(value) for value in values if value is not None]
    return sum(known) if known else None


# ---------------------------------------------------------------------------
# Pregame projection snapshots (source + timestamp)
# ---------------------------------------------------------------------------

def _ensure_snapshot_table():
    global _SNAP_TABLE_READY
    if _SNAP_TABLE_READY:
        return
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS weekly_projection_snapshots (
                season      INT  NOT NULL,
                week        INT  NOT NULL,
                player_id   TEXT NOT NULL,
                proj_pts    DOUBLE PRECISION,
                source      TEXT,
                saved_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (season, week, player_id)
            )
            """
        )
        conn.commit()
    _SNAP_TABLE_READY = True


def save_pregame_snapshot(season: int, week: int, source: str = "sleeper",
                          overwrite: bool = False) -> int:
    """Persist the CURRENT weekly projections as this week's pregame snapshot.

    Meant to run from the scheduled pipeline BEFORE the week's games. Idempotent:
    by default it will not overwrite an existing snapshot for the week (so a
    mid-week re-run can't replace the true pregame numbers with revised ones).
    Returns the number of rows written; 0 when a snapshot already exists.
    """
    try:
        _ensure_snapshot_table()
        from dashboard_services.db import get_conn
        if not overwrite:
            with get_conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM weekly_projection_snapshots "
                    "WHERE season=%s AND week=%s",
                    (int(season), int(week)),
                ).fetchone()
            existing = (row["n"] if isinstance(row, dict) else row[0]) if row else 0
            if existing:
                logger.info("[waiver-discoveries] pregame snapshot s%sw%s already exists (%s rows)",
                            season, week, existing)
                return 0
        from utils.fantasy_scoring import weekly_projection_points
        from utils.utils import load_week_projection
        proj = load_week_projection(int(season), int(week)) or {}
        written = 0
        with get_conn() as conn:
            for pid in list(proj.keys()):
                pts = weekly_projection_points(proj, pid)
                if pts is None:
                    continue
                conn.execute(
                    """
                    INSERT INTO weekly_projection_snapshots
                        (season, week, player_id, proj_pts, source, saved_at)
                    VALUES (%s,%s,%s,%s,%s, NOW())
                    ON CONFLICT (season, week, player_id) DO UPDATE SET
                        proj_pts = EXCLUDED.proj_pts, source = EXCLUDED.source,
                        saved_at = EXCLUDED.saved_at
                    """,
                    (int(season), int(week), str(pid), float(pts), source),
                )
                written += 1
            conn.commit()
        logger.info("[waiver-discoveries] saved %s pregame snapshots s%sw%s", written, season, week)
        return written
    except Exception:
        logger.warning("[waiver-discoveries] pregame snapshot failed", exc_info=True)
        return 0


def load_pregame_snapshot(season: int, week: int) -> Dict[str, dict]:
    """{player_id: {"pts", "source", "saved_at"}} for a week's pregame snapshot."""
    try:
        _ensure_snapshot_table()
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT player_id, proj_pts, source, saved_at "
                "FROM weekly_projection_snapshots WHERE season=%s AND week=%s",
                (int(season), int(week)),
            ).fetchall()
        out = {}
        for r in rows:
            d = dict(r)
            out[str(d["player_id"])] = {
                "pts": d.get("proj_pts"),
                "source": d.get("source"),
                "saved_at": (d.get("saved_at").isoformat()
                             if hasattr(d.get("saved_at"), "isoformat") else d.get("saved_at")),
            }
        return out
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return {}


# ---------------------------------------------------------------------------
# Idempotent discovery persistence
# ---------------------------------------------------------------------------

def _ensure_disc_table():
    global _DISC_TABLE_READY
    if _DISC_TABLE_READY:
        return
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS waiver_big_game_discoveries (
                discovery_key         TEXT PRIMARY KEY,
                season                INT  NOT NULL,
                week                  INT  NOT NULL,
                player_id             TEXT NOT NULL,
                category              TEXT,
                performance_surprise  DOUBLE PRECISION,
                role_sustainability   DOUBLE PRECISION,
                status                TEXT,
                payload               JSONB NOT NULL DEFAULT '{}'::jsonb,
                updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        conn.commit()
    _DISC_TABLE_READY = True


def _load_one(key: str) -> Optional[dict]:
    from dashboard_services.db import get_conn
    with get_conn() as conn:
        row = conn.execute(
            "SELECT payload, status FROM waiver_big_game_discoveries WHERE discovery_key=%s",
            (key,),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    payload = d.get("payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    return payload if isinstance(payload, dict) else {}


def upsert_discovery(a: BigGameAssessment) -> bool:
    """Idempotently store/refresh one discovery (item 8).

    Keyed by (season, week, player_id) via discovery_key, so during-game,
    final, and stat-correction passes update one row. A less-final status never
    clobbers a more-final one (merge_assessment decides). Returns True if a write
    happened, False if the incoming assessment was superseded by what's stored.
    """
    try:
        _ensure_disc_table()
        key = discovery_key(a.player_id, a.season, a.week)
        existing_payload = _load_one(key)
        existing = _assessment_from_payload(existing_payload) if existing_payload else None
        merged = merge_assessment(existing, a)
        if existing is not None and merged is existing:
            return False  # incoming was superseded (e.g. late live after final)
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO waiver_big_game_discoveries
                    (discovery_key, season, week, player_id, category,
                     performance_surprise, role_sustainability, status, payload, updated_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb, NOW())
                ON CONFLICT (discovery_key) DO UPDATE SET
                    category = EXCLUDED.category,
                    performance_surprise = EXCLUDED.performance_surprise,
                    role_sustainability = EXCLUDED.role_sustainability,
                    status = EXCLUDED.status,
                    payload = EXCLUDED.payload,
                    updated_at = NOW()
                """,
                (key, merged.season, merged.week, merged.player_id, merged.category,
                 merged.performance_surprise, merged.role_sustainability, merged.status,
                 json.dumps(merged.to_dict())),
            )
            conn.commit()
        return True
    except Exception:
        logger.warning("[waiver-discoveries] upsert failed for %s", a.player_id, exc_info=True)
        return False


def _assessment_from_payload(p: dict) -> Optional[BigGameAssessment]:
    if not isinstance(p, dict) or not p.get("player_id"):
        return None
    try:
        return BigGameAssessment(
            player_id=str(p["player_id"]), season=int(p.get("season") or 0),
            week=int(p.get("week") or 0), category=p.get("category") or "none",
            performance_surprise=float(p.get("performance_surprise") or 0),
            role_sustainability=float(p.get("role_sustainability") or 0),
            absolute_score=float(p.get("absolute_score") or 0),
            expectation=p.get("expectation"),
            expectation_basis=p.get("expectation_basis") or "none",
            role_confirmed=bool(p.get("role_confirmed")),
            cautions=tuple(p.get("cautions") or ()),
            factors=tuple(p.get("factors") or ()),
            status=p.get("status") or "final",
        )
    except Exception:
        return None


def get_week_discoveries(season: int, week: int,
                         categories=("priority", "speculative", "watchlist")) -> List[dict]:
    """Stored discovery dicts for a week, most surprising first."""
    try:
        _ensure_disc_table()
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM waiver_big_game_discoveries "
                "WHERE season=%s AND week=%s AND category = ANY(%s) "
                "ORDER BY performance_surprise DESC",
                (int(season), int(week), list(categories)),
            ).fetchall()
        out = []
        for r in rows:
            payload = dict(r).get("payload")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    continue
            if isinstance(payload, dict):
                out.append(payload)
        return out
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Build + detect for a completed (or in-progress) week
# ---------------------------------------------------------------------------

def detect_week(season: int, week: int, *, status: str = "final",
                positions=("QB", "RB", "WR", "TE"), persist: bool = True) -> List[dict]:
    """Run the shared detector across a week's player_weekly_metrics rows and
    (optionally) persist the discoveries idempotently. Returns the surfaced
    discovery dicts (priority / speculative / watchlist). Best-effort: any data
    failure yields [] rather than raising, so callers/pages never break."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT player_id, week, position, snap_pct, targets, touches, carries, "
                "target_share, ppr_pts, rec_yards, rush_yards, pass_att, "
                "rz_targets, rz_carries, rec_tds, rush_tds "
                "FROM player_weekly_metrics WHERE season=%s AND week<=%s ORDER BY player_id, week",
                (int(season), int(week)),
            ).fetchall()
    except Exception:
        logger.debug("suppressed exception", exc_info=True)
        return []

    by_pid: Dict[str, list] = {}
    for r in rows:
        by_pid.setdefault(str(r["player_id"]), []).append(dict(r))

    snapshot = load_pregame_snapshot(season, week)
    out: List[dict] = []
    for pid, weeks in by_pid.items():
        this = next((w for w in weeks if int(w["week"]) == int(week)), None)
        if this is None:
            continue
        pos = str(this.get("position") or "").upper()
        if pos not in positions:
            continue
        prior = [w for w in weeks if int(w["week"]) < int(week)]
        ctx = game_context_from_rows(
            pid, pos, season, week, this, prior,
            pregame=snapshot.get(str(pid)), status=status,
            extra_features={
                "redzone_touches": _sum_known(this.get("rz_targets"), this.get("rz_carries")),
                "touchdowns": _sum_known(this.get("rec_tds"), this.get("rush_tds")),
            })
        assessment = assess_big_game(ctx)
        if assessment.category == "none":
            continue
        if persist:
            upsert_discovery(assessment)
        out.append(assessment.to_dict())
    out.sort(key=lambda d: d.get("performance_surprise", 0), reverse=True)
    return out
