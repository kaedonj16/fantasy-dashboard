"""Start/Sit advice tracking: snapshot the advice, grade it after the week.

Until now the Start/Sit advisor was memoryless: ``/api/start-sit-options``
computed a payload per request and nothing was ever persisted, so there was
no way to answer "how are the start/sit calls performing?" This module closes
that loop the same way the waiver big-game detector does:

* :func:`build_snapshot_payload` distils one options payload (the exact rows
  the viewer was shown: verdict, start score, projection, close-call pairing,
  and the optimal-lineup swaps) into a compact snapshot.
* :func:`save_snapshot` persists it keyed by
  ``(season, week, platform, league_id, roster_id)``. Captures refresh freely
  while the week is live (the latest pre-grade view is the most informed
  advice the viewer saw), but a snapshot locks the moment a grade exists for
  it, so a grade can never be rewritten by a later view.
* :func:`grade_snapshot_payload` is the pure grader. Given the snapshot and
  the week's actual league-scored points per player (the provider matchup
  row's ``players_points``, already in league scoring), it reports what the
  recommended lineup actually scored vs. the lineup the viewer had set at
  snapshot time vs. the optimal lineup, grades every recommended swap and
  every close call, and measures projection error per player.
* :func:`grade_league_week` / :func:`grade_pending` load actuals through the
  provider matchup API and persist grades idempotently. A week whose
  matchup data is incomplete stays ungraded and is retried on the next run;
  nothing is graded from partial data.
* :func:`summarize_tracking` aggregates snapshots + grades into the season
  view the Start/Sit page renders: per-week results, season totals, swap and
  close-call records, and a calibration table (close-call hit rate by the
  starter's 0-100 start-score band).

All DB access is best-effort: a tracking failure must never break the
Start/Sit page, so public entry points swallow and log errors.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_TABLES_READY = False


def _get_conn():
    from dashboard_services.db import get_conn

    return get_conn()


# ---------------------------------------------------------------------------
# Pure transforms (no DB, unit-testable)
# ---------------------------------------------------------------------------

def snapshot_key(season: int, week: int, platform: str, league_id: str,
                 roster_id) -> str:
    return (
        f"{int(season)}:{int(week)}:{str(platform).strip().lower()}:"
        f"{str(league_id).strip()}:{str(roster_id).strip()}"
    )


def _num(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_snapshot_payload(*, season: int, week: int, platform: str,
                           league_id: str, roster_id, positions: dict,
                           lineup_advice: Optional[dict],
                           roster_positions: Optional[list],
                           current_starters: Optional[list] = None,
                           captured_at: Optional[str] = None) -> dict:
    """Distil one Start/Sit options payload into a snapshot dict.

    ``positions`` is the payload's ``positions`` mapping (pos -> row dicts as
    built by ``_build_start_sit_options``). Guest rows are excluded: guests
    were never the viewer's decision. ``lineup_advice`` is the payload's
    optimal-lineup advice block; its swaps and its score totals (measured in
    start score, not fantasy points) are kept for context, while grading is
    always done in actual fantasy points.
    """
    players: List[dict] = []
    for pos, rows in (positions or {}).items():
        for row in rows or []:
            if not isinstance(row, dict) or row.get("guest"):
                continue
            pid = str(row.get("player_id") or "")
            if not pid:
                continue
            is_start = bool(row.get("start"))
            is_flex = bool(row.get("flex_start"))
            verdict = "FLEX" if is_flex else ("START" if is_start else "BENCH")
            close = row.get("close_call") or None
            players.append({
                "player_id": pid,
                "name": row.get("name") or pid,
                "pos": str(pos or row.get("pos") or "").upper(),
                "team": row.get("team"),
                "verdict": verdict,
                "is_recommended": is_start,
                "flex_slot": row.get("flex_slot"),
                "start_score": _num(row.get("start_score")),
                "start_score_pct": row.get("start_score_pct"),
                "proj_pts": _num(row.get("proj_pts")),
                "on_bye": bool(row.get("on_bye")),
                "close_call": (
                    {
                        "vs_player_id": str(close.get("vs_player_id") or ""),
                        "vs_name": close.get("vs_name"),
                        "win_prob": _num(close.get("win_prob")),
                    }
                    if close and close.get("vs_player_id")
                    else None
                ),
            })
    advice = lineup_advice or {}
    swaps: List[dict] = []
    for swap in advice.get("swaps") or []:
        if not isinstance(swap, dict):
            continue
        start = swap.get("start") or {}
        sit = swap.get("sit") or {}
        if not start.get("player_id"):
            continue
        swaps.append({
            "start_player_id": str(start.get("player_id")),
            "start_name": start.get("name"),
            "sit_player_id": str(sit.get("player_id")) if sit.get("player_id") else None,
            "sit_name": sit.get("name"),
            "slot": swap.get("slot"),
            "proj_gain": _num(swap.get("gain")),
        })
    recommended = [p["player_id"] for p in players if p["is_recommended"]]
    roster_pids = {p["player_id"] for p in players}
    if current_starters is not None:
        # The caller's exact current-starter set, restricted to roster rows
        # the snapshot actually carries (a starter with no row cannot be
        # graded, so he is dropped rather than silently scored as zero).
        current = [str(pid) for pid in current_starters
                   if str(pid) in roster_pids]
    else:
        # Fallback derivation from the advice: current starters are the
        # recommended players who needed no swap, plus every swap's "sit"
        # player. Exact when the swap pairing covered every difference.
        swap_ins = {s["start_player_id"] for s in swaps}
        swap_outs = {s["sit_player_id"] for s in swaps if s["sit_player_id"]}
        current = [pid for pid in recommended if pid not in swap_ins]
        current.extend(sorted(swap_outs))
    return {
        "key": snapshot_key(season, week, platform, league_id, roster_id),
        "season": int(season),
        "week": int(week),
        "platform": str(platform or "sleeper").strip().lower(),
        "league_id": str(league_id),
        "roster_id": str(roster_id),
        "captured_at": captured_at or datetime.now(timezone.utc).isoformat(),
        "roster_positions": list(roster_positions or []),
        "players": players,
        "recommended_starters": recommended,
        "current_starters": current,
        "has_current": bool(advice.get("has_current")),
        "advice_score_delta": _num(advice.get("delta")),
        "swaps": swaps,
    }


def _score_band(pct) -> Optional[str]:
    pct = _num(pct)
    if pct is None:
        return None
    if pct >= 90:
        return "90-100"
    if pct >= 80:
        return "80-89"
    if pct >= 70:
        return "70-79"
    if pct >= 60:
        return "60-69"
    return "Under 60"


def grade_snapshot_payload(snapshot: dict, players_points: dict,
                           optimal_fn: Optional[Callable] = None) -> dict:
    """Grade one snapshot against actual league-scored points.

    ``players_points`` maps player_id -> actual fantasy points for the week
    (the provider matchup row's ``players_points``). Returns
    ``{"complete": False, "missing_player_ids": [...]}`` when any rostered
    player in the snapshot has no score yet; the caller must not persist an
    incomplete grade.
    """
    players = snapshot.get("players") or []
    points: Dict[str, float] = {}
    missing: List[str] = []
    for p in players:
        pid = str(p.get("player_id") or "")
        raw = (players_points or {}).get(pid)
        if raw is None:
            raw = (players_points or {}).get(str(pid))
        val = _num(raw)
        if val is None:
            missing.append(pid)
        else:
            points[pid] = val
    if missing:
        return {"complete": False, "missing_player_ids": missing}

    by_pid = {str(p.get("player_id")): p for p in players}
    recommended = [pid for pid in snapshot.get("recommended_starters") or []
                   if pid in points]
    current = [pid for pid in snapshot.get("current_starters") or []
               if pid in points]

    def _total(pids) -> float:
        return round(sum(points.get(pid, 0.0) for pid in pids), 2)

    recommended_actual = _total(recommended)
    current_actual = _total(current) if snapshot.get("has_current") else None

    optimal_actual = None
    optimal_starters: List[str] = []
    try:
        if optimal_fn is None:
            from utils.optimal_lineup import compute_optimal_lineup as optimal_fn  # type: ignore
        pos_map = {pid: (p.get("pos") or "") for pid, p in by_pid.items()}
        opt_set, opt_total = optimal_fn(
            points, pos_map, snapshot.get("roster_positions") or [],
            list(points.keys()))
        optimal_starters = sorted(str(pid) for pid in (opt_set or []))
        optimal_actual = round(float(opt_total), 2)
    except Exception:
        logger.debug("start/sit grading: optimal lineup failed", exc_info=True)

    graded_swaps: List[dict] = []
    for swap in snapshot.get("swaps") or []:
        in_pid = swap.get("start_player_id")
        out_pid = swap.get("sit_player_id")
        in_actual = points.get(str(in_pid)) if in_pid else None
        if in_actual is None:
            continue
        if out_pid:
            out_actual = points.get(str(out_pid))
            if out_actual is None:
                continue
            delta = round(in_actual - out_actual, 2)
            outcome = "hit" if delta > 0 else ("push" if delta == 0 else "miss")
        else:
            out_actual = None
            delta = round(in_actual, 2)
            outcome = "empty_slot"
        graded_swaps.append({
            "start_player_id": in_pid,
            "start_name": swap.get("start_name"),
            "sit_player_id": out_pid,
            "sit_name": swap.get("sit_name"),
            "slot": swap.get("slot"),
            "actual_start": round(in_actual, 2),
            "actual_sit": (round(out_actual, 2) if out_actual is not None else None),
            "delta": delta,
            "outcome": outcome,
        })

    graded_close: List[dict] = []
    for p in players:
        close = p.get("close_call")
        if not close or not close.get("vs_player_id"):
            continue
        pid = str(p.get("player_id"))
        vs_pid = str(close.get("vs_player_id"))
        if pid not in points or vs_pid not in points:
            continue
        delta = round(points[pid] - points[vs_pid], 2)
        graded_close.append({
            "player_id": pid,
            "name": p.get("name"),
            "pos": p.get("pos"),
            "start_score_pct": p.get("start_score_pct"),
            "band": _score_band(p.get("start_score_pct")),
            "win_prob": close.get("win_prob"),
            "vs_player_id": vs_pid,
            "vs_name": close.get("vs_name"),
            "actual": round(points[pid], 2),
            "vs_actual": round(points[vs_pid], 2),
            "delta": delta,
            "outcome": "hit" if delta > 0 else ("push" if delta == 0 else "miss"),
        })

    graded_players = [{
        "player_id": str(p.get("player_id")),
        "name": p.get("name"),
        "pos": p.get("pos"),
        "verdict": p.get("verdict"),
        "proj_pts": p.get("proj_pts"),
        "actual_pts": round(points[str(p.get("player_id"))], 2),
        "proj_error": (
            round(points[str(p.get("player_id"))] - float(p["proj_pts"]), 2)
            if p.get("proj_pts") is not None else None
        ),
    } for p in players]

    return {
        "complete": True,
        "week": snapshot.get("week"),
        "season": snapshot.get("season"),
        "recommended_actual": recommended_actual,
        "current_actual": current_actual,
        "optimal_actual": optimal_actual,
        "optimal_starters": optimal_starters,
        "advice_delta": (
            round(recommended_actual - current_actual, 2)
            if current_actual is not None else None
        ),
        "efficiency": (
            round(recommended_actual / optimal_actual * 100, 1)
            if optimal_actual else None
        ),
        "swaps": graded_swaps,
        "close_calls": graded_close,
        "players": graded_players,
    }


def summarize_tracking(entries: List[dict]) -> dict:
    """Aggregate snapshot/grade entries into the page-facing summary.

    ``entries`` is a list of ``{"snapshot": ..., "grade": ... | None}`` for one
    league + roster, any week order. Pure: no DB, no provider calls.
    """
    entries = sorted(entries or [],
                     key=lambda e: int((e.get("snapshot") or {}).get("week") or 0))
    weeks: List[dict] = []
    pending: List[int] = []
    tot_rec = tot_cur = tot_opt = 0.0
    have_cur = have_opt = False
    swap_hits = swap_push = swap_total = 0
    close_hits = close_push = close_total = 0
    proj_abs_err: List[float] = []
    bands: Dict[str, dict] = {}
    for entry in entries:
        snap = entry.get("snapshot") or {}
        grade = entry.get("grade") or None
        week = int(snap.get("week") or 0)
        if not grade or not grade.get("complete"):
            pending.append(week)
            weeks.append({
                "week": week,
                "captured_at": snap.get("captured_at"),
                "graded": False,
                "recommended_count": len(snap.get("recommended_starters") or []),
                "swap_count": len(snap.get("swaps") or []),
            })
            continue
        tot_rec += float(grade.get("recommended_actual") or 0.0)
        if grade.get("current_actual") is not None:
            tot_cur += float(grade["current_actual"])
            have_cur = True
        if grade.get("optimal_actual") is not None:
            tot_opt += float(grade["optimal_actual"])
            have_opt = True
        w_swaps = [s for s in grade.get("swaps") or []
                   if s.get("outcome") in ("hit", "miss", "push")]
        w_close = grade.get("close_calls") or []
        swap_hits += sum(1 for s in w_swaps if s["outcome"] == "hit")
        swap_push += sum(1 for s in w_swaps if s["outcome"] == "push")
        swap_total += len(w_swaps)
        close_hits += sum(1 for c in w_close if c["outcome"] == "hit")
        close_push += sum(1 for c in w_close if c["outcome"] == "push")
        close_total += len(w_close)
        for c in w_close:
            band = c.get("band") or _score_band(c.get("start_score_pct"))
            if not band:
                continue
            b = bands.setdefault(band, {"band": band, "n": 0, "hits": 0})
            b["n"] += 1
            if c["outcome"] == "hit":
                b["hits"] += 1
        for gp in grade.get("players") or []:
            if gp.get("proj_error") is not None:
                proj_abs_err.append(abs(float(gp["proj_error"])))
        weeks.append({
            "week": week,
            "captured_at": snap.get("captured_at"),
            "graded": True,
            "recommended_actual": grade.get("recommended_actual"),
            "current_actual": grade.get("current_actual"),
            "optimal_actual": grade.get("optimal_actual"),
            "advice_delta": grade.get("advice_delta"),
            "efficiency": grade.get("efficiency"),
            "swaps": grade.get("swaps") or [],
            "swap_record": {
                "hits": sum(1 for s in w_swaps if s["outcome"] == "hit"),
                "total": len(w_swaps),
            },
            "close_calls": w_close,
            "close_record": {
                "hits": sum(1 for c in w_close if c["outcome"] == "hit"),
                "total": len(w_close),
            },
        })
    graded_weeks = [w for w in weeks if w["graded"]]
    band_order = ["90-100", "80-89", "70-79", "60-69", "Under 60"]
    calibration = [bands[b] for b in band_order if b in bands]
    for row in calibration:
        row["hit_rate"] = round(row["hits"] / row["n"] * 100, 1) if row["n"] else None
    if not entries:
        state = "no_snapshots"
    elif not graded_weeks:
        state = "awaiting_grades"
    else:
        state = "graded"
    return {
        "state": state,
        "weeks": list(reversed(weeks)),
        "pending_weeks": pending,
        "season": {
            "weeks_graded": len(graded_weeks),
            "weeks_snapshotted": len(entries),
            "recommended_actual": round(tot_rec, 1) if graded_weeks else None,
            "current_actual": round(tot_cur, 1) if have_cur else None,
            "optimal_actual": round(tot_opt, 1) if have_opt else None,
            "advice_delta": (
                round(tot_rec - tot_cur, 1) if have_cur and graded_weeks else None
            ),
            "efficiency": (
                round(tot_rec / tot_opt * 100, 1) if have_opt and tot_opt else None
            ),
            "swap_record": {"hits": swap_hits, "pushes": swap_push, "total": swap_total},
            "close_record": {"hits": close_hits, "pushes": close_push, "total": close_total},
            "proj_mae": (
                round(sum(proj_abs_err) / len(proj_abs_err), 2)
                if proj_abs_err else None
            ),
        },
        "calibration": calibration,
    }


# ---------------------------------------------------------------------------
# Persistence (best-effort; failures log and return empty/False)
# ---------------------------------------------------------------------------

def _ensure_tables() -> bool:
    global _TABLES_READY
    if _TABLES_READY:
        return True
    try:
        with _get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS start_sit_snapshots (
                    snapshot_key TEXT PRIMARY KEY,
                    season       INT  NOT NULL,
                    week         INT  NOT NULL,
                    platform     TEXT NOT NULL,
                    league_id    TEXT NOT NULL,
                    roster_id    TEXT NOT NULL,
                    source       TEXT,
                    captured_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    payload      JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sss_league_week "
                "ON start_sit_snapshots (platform, league_id, season, week)"
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS start_sit_grades (
                    snapshot_key TEXT PRIMARY KEY,
                    graded_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    payload      JSONB NOT NULL DEFAULT '{}'::jsonb
                )
                """
            )
            conn.commit()
        _TABLES_READY = True
        return True
    except Exception:
        logger.debug("start_sit_tracking: table init failed", exc_info=True)
        return False


def _row_payload(row) -> dict:
    if not row:
        return {}
    d = dict(row)
    payload = d.get("payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}
    return payload if isinstance(payload, dict) else {}


def save_snapshot(snapshot: dict, source: str = "page_view") -> bool:
    """Persist one snapshot. Returns False when a grade already locks it."""
    if not snapshot or not snapshot.get("key"):
        return False
    if not _ensure_tables():
        return False
    try:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT snapshot_key FROM start_sit_grades WHERE snapshot_key=%s",
                (snapshot["key"],),
            ).fetchone()
            if row:
                return False
            conn.execute(
                """
                INSERT INTO start_sit_snapshots
                    (snapshot_key, season, week, platform, league_id,
                     roster_id, source, captured_at, payload)
                VALUES (%s,%s,%s,%s,%s,%s,%s,NOW(),%s)
                ON CONFLICT (snapshot_key) DO UPDATE SET
                    source = EXCLUDED.source,
                    captured_at = NOW(),
                    payload = EXCLUDED.payload
                """,
                (
                    snapshot["key"], int(snapshot["season"]), int(snapshot["week"]),
                    snapshot["platform"], snapshot["league_id"],
                    snapshot["roster_id"], source,
                    json.dumps(snapshot),
                ),
            )
            conn.commit()
        return True
    except Exception:
        logger.debug("start_sit_tracking: save_snapshot failed", exc_info=True)
        return False


def _load_entries(platform: str, league_id: str, season: int,
                  roster_id=None) -> List[dict]:
    if not _ensure_tables():
        return []
    try:
        with _get_conn() as conn:
            if roster_id is not None:
                rows = conn.execute(
                    "SELECT s.payload AS payload, g.payload AS grade_payload "
                    "FROM start_sit_snapshots s "
                    "LEFT JOIN start_sit_grades g "
                    "  ON g.snapshot_key = s.snapshot_key "
                    "WHERE s.platform=%s AND s.league_id=%s AND s.season=%s "
                    "  AND s.roster_id=%s ORDER BY s.week",
                    (str(platform).lower(), str(league_id), int(season),
                     str(roster_id)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT s.payload AS payload, g.payload AS grade_payload "
                    "FROM start_sit_snapshots s "
                    "LEFT JOIN start_sit_grades g "
                    "  ON g.snapshot_key = s.snapshot_key "
                    "WHERE s.platform=%s AND s.league_id=%s AND s.season=%s "
                    "ORDER BY s.week",
                    (str(platform).lower(), str(league_id), int(season)),
                ).fetchall()
        entries = []
        for r in rows:
            d = dict(r)
            snap = d.get("payload")
            if isinstance(snap, str):
                try:
                    snap = json.loads(snap)
                except Exception:
                    snap = {}
            grade = d.get("grade_payload")
            if isinstance(grade, str):
                try:
                    grade = json.loads(grade)
                except Exception:
                    grade = None
            if isinstance(snap, dict) and snap:
                entries.append({
                    "snapshot": snap,
                    "grade": grade if isinstance(grade, dict) else None,
                })
        return entries
    except Exception:
        logger.debug("start_sit_tracking: load entries failed", exc_info=True)
        return []


def _save_grade(key: str, grade: dict) -> bool:
    try:
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO start_sit_grades (snapshot_key, graded_at, payload)
                VALUES (%s, NOW(), %s)
                ON CONFLICT (snapshot_key) DO UPDATE SET
                    graded_at = NOW(), payload = EXCLUDED.payload
                """,
                (key, json.dumps(grade)),
            )
            conn.commit()
        return True
    except Exception:
        logger.debug("start_sit_tracking: save grade failed", exc_info=True)
        return False


# ---------------------------------------------------------------------------
# Grading drivers (provider actuals -> persisted grades)
# ---------------------------------------------------------------------------

def grade_league_week(platform: str, league_id: str, season: int, week: int,
                      matchups: Optional[list] = None) -> int:
    """Grade every snapshot for one league-week. Returns grades written.

    Actuals come from the provider matchup rows: each row's
    ``players_points`` is already scored in that league's scoring, which is
    the only honest basis for grading advice given in that league.
    """
    if not _ensure_tables():
        return 0
    entries = [e for e in _load_entries(platform, league_id, season)
               if int((e.get("snapshot") or {}).get("week") or 0) == int(week)
               and not (e.get("grade") or {}).get("complete")]
    if not entries:
        return 0
    if matchups is None:
        try:
            from dashboard_services.platform_api import get_matchups

            matchups = get_matchups(platform, league_id, int(week), int(season)) or []
        except Exception:
            logger.debug("start_sit_tracking: matchups unavailable", exc_info=True)
            return 0
    by_roster: Dict[str, dict] = {}
    for row in matchups or []:
        if isinstance(row, dict) and row.get("roster_id") is not None:
            by_roster[str(row.get("roster_id"))] = row
    written = 0
    for entry in entries:
        snap = entry["snapshot"]
        row = by_roster.get(str(snap.get("roster_id")))
        if not row:
            continue
        grade = grade_snapshot_payload(snap, row.get("players_points") or {})
        if not grade.get("complete"):
            continue
        if _save_grade(str(snap.get("key")), grade):
            written += 1
    return written


def grade_pending_for_league(platform: str, league_id: str, season: int,
                             before_week: int) -> int:
    """Grade all of one league's snapshots from weeks before ``before_week``."""
    entries = _load_entries(platform, league_id, season)
    weeks = sorted({
        int((e.get("snapshot") or {}).get("week") or 0)
        for e in entries
        if not (e.get("grade") or {}).get("complete")
    })
    written = 0
    for week in weeks:
        if 0 < week < int(before_week):
            written += grade_league_week(platform, league_id, season, week)
    return written


def grade_pending(season: int, current_week: int) -> dict:
    """Cron driver: grade every pending snapshot from completed weeks.

    Groups by (platform, league, week) straight from the snapshot table, so
    it covers exactly the leagues where advice was actually shown. Weeks at
    or after ``current_week`` are still live and are never graded.
    """
    summary = {"graded": 0, "leagues": 0}
    if not _ensure_tables():
        return summary
    try:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT DISTINCT platform, league_id, week "
                "FROM start_sit_snapshots WHERE season=%s AND week<%s "
                "ORDER BY week",
                (int(season), int(current_week)),
            ).fetchall()
    except Exception:
        logger.debug("start_sit_tracking: pending scan failed", exc_info=True)
        return summary
    seen = set()
    for r in rows:
        d = dict(r)
        platform = str(d.get("platform") or "sleeper")
        league_id = str(d.get("league_id") or "")
        week = int(d.get("week") or 0)
        if not league_id or not week:
            continue
        summary["graded"] += grade_league_week(platform, league_id, season, week)
        if (platform, league_id) not in seen:
            seen.add((platform, league_id))
            summary["leagues"] += 1
    return summary


def get_tracking_summary(platform: str, league_id: str, season: int,
                         roster_id) -> dict:
    """Page-facing summary for one viewer's team in one league."""
    entries = _load_entries(platform, league_id, season, roster_id)
    out = summarize_tracking(entries)
    out["platform"] = str(platform or "sleeper").lower()
    out["league_id"] = str(league_id)
    out["season_year"] = int(season)
    return out
