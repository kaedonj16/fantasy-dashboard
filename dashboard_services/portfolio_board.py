"""Portfolio Front Office board: done states, verification, grading, helpers.

Powers the cross-league "Front Office" board on My Leagues (the v25 redesign):

- Pure helpers shared by the page renderer, the board API and the tests:
  receipts tags/tiles, posture, weakest room, heatmap tiers, the client
  move-key derivation (must match the JS pfKey/pfHash in app.py), board
  grouping with global ranks, and lock timing.
- The per-account Done store (``portfolio_move_states``), in the same family
  as ``user_action_dismissals``: a Done click is verified against a platform
  read-back where one exists (Sleeper roster/IR state), and only then is the
  move labelled Verified. Anything else is stored as marked with an honest
  reason, never as verified.
- Board snapshots + self-grading (``portfolio_board_snapshots`` /
  ``portfolio_board_grades``), mirroring start_sit_tracking.py: the served
  board is snapshotted per account/week, and after the week completes the
  gradeable moves (lineup swaps, whose actuals live in the provider matchup
  row's ``players_points``, already in league scoring) are graded. Moves
  that cannot be graded stay ungraded, never faked.

DB access mirrors dashboard_services.playoff_odds_history: the driver is
imported lazily so importing this module never requires psycopg.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

TOSSUP_BAND = 3.0
POSITIONS = ("QB", "RB", "WR", "TE")
_PLAT_LABELS = {
    "sleeper": "Sleeper", "yahoo": "Yahoo", "espn": "ESPN",
    "mfl": "MFL", "fleaflicker": "Fleaflicker",
}


def get_conn():
    """Lazy DB handle (see playoff_odds_history.get_conn)."""
    from dashboard_services.db import get_conn as _get_conn
    return _get_conn()


# ── Receipts (summary projected tile) ────────────────────────────────────────

def receipt_tag(my_proj, opp_proj) -> Optional[str]:
    """W / L / Toss-up for one projected matchup. Toss-up is a projected
    margin within TOSSUP_BAND points; the receipts panel header states the
    band so the tag reads as math, not opinion."""
    try:
        mine = float(my_proj)
        theirs = float(opp_proj)
    except (TypeError, ValueError):
        return None
    if abs(mine - theirs) <= TOSSUP_BAND:
        return "Toss-up"
    return "W" if mine > theirs else "L"


def receipts_tile(rows) -> Dict[str, int]:
    """Aggregate W-L-Tossup counts for the projected tile. Only rows with a
    computable tag count, so the tile always reconciles to the panel."""
    out = {"w": 0, "l": 0, "t": 0}
    for row in rows or []:
        tag = row.get("tag") if isinstance(row, dict) else None
        if tag == "W":
            out["w"] += 1
        elif tag == "L":
            out["l"] += 1
        elif tag == "Toss-up":
            out["t"] += 1
    return out


def tile_str(tile: Dict[str, int]) -> str:
    return f"{tile.get('w', 0)}-{tile.get('l', 0)}"


# ── League card helpers ──────────────────────────────────────────────────────

def posture_for(odds_pct=None, wins=None, losses=None):
    """(label, css-class) posture for a league: Contend / Fringe / Rebuild.

    Playoff odds drive it when known; otherwise the win rate stands in. The
    thresholds are the mock's: a coin-flip-or-better team contends, a team
    with a real but minority path is fringe, the rest are rebuilding."""
    pct = None
    try:
        if odds_pct is not None:
            pct = float(odds_pct)
    except (TypeError, ValueError):
        pct = None
    if pct is not None:
        if pct >= 55:
            return ("Contend", "contend")
        if pct >= 30:
            return ("Fringe", "fringe")
        return ("Rebuild", "rebuild")
    try:
        w = float(wins or 0)
        l = float(losses or 0)
    except (TypeError, ValueError):
        return (None, "")
    if w + l <= 0:
        return (None, "")
    rate = w / (w + l)
    if rate >= 0.55:
        return ("Contend", "contend")
    if rate >= 0.40:
        return ("Fringe", "fringe")
    return ("Rebuild", "rebuild")


def weakest_room(pos_ranks: dict, total_teams) -> Optional[dict]:
    """The user's worst in-league position rank, for the card's weakest line.

    Highest rank number is weakest; ties break toward the later position in
    QB/RB/WR/TE order so the answer is deterministic."""
    try:
        total = int(total_teams)
    except (TypeError, ValueError):
        total = 0
    best = None
    for idx, pos in enumerate(POSITIONS):
        raw = (pos_ranks or {}).get(pos)
        try:
            rank = int(raw)
        except (TypeError, ValueError):
            continue
        if rank <= 0:
            continue
        key = (rank, idx)
        if best is None or key > best[0]:
            best = (key, {"pos": pos, "rank": rank, "total": total or None})
    return best[1] if best else None


def tier_for_rank(rank, total) -> str:
    """good / mid / weak by thirds, matching the card strength strip."""
    try:
        r = int(rank)
        n = int(total)
    except (TypeError, ValueError):
        return "mid"
    if n <= 1 or r <= 0:
        return "mid"
    frac = (r - 1) / (n - 1)
    if frac <= 1 / 3 + 1e-9:
        return "good"
    if frac >= 2 / 3 - 1e-9:
        return "weak"
    return "mid"


def heatmap_rows(leagues) -> List[dict]:
    """Leagues x positions rank rows for the Positional Strength heatmap.

    Input rows are league summary dicts (name + pos_user_rank + total_teams);
    leagues without any rank data are skipped rather than padded with
    placeholder ranks."""
    rows = []
    for lg in leagues or []:
        ranks = lg.get("pos_user_rank") or {}
        total = lg.get("total_teams")
        cells = {}
        for pos in POSITIONS:
            raw = ranks.get(pos)
            try:
                rank = int(raw)
            except (TypeError, ValueError):
                rank = None
            if rank is None or rank <= 0:
                cells[pos] = None
            else:
                cells[pos] = {"rank": rank, "tier": tier_for_rank(rank, total)}
        if any(cells.values()):
            rows.append({"name": lg.get("name") or "", "cells": cells})
    return rows


# ── Move keys (must match the client pfKey/pfHash derivation) ───────────────

def _js_hash(text: str) -> str:
    """Replicates the client's pfHash: h = ((h << 5) - h + code) | 0 over
    UTF-16 code units, rendered as unsigned base-36."""
    h = 0
    data = str(text or "").encode("utf-16-le", "surrogatepass")
    units = [data[i] | (data[i + 1] << 8) for i in range(0, len(data), 2)]
    for unit in units:
        h = ((h << 5) - h + unit) & 0xFFFFFFFF
        if h >= 0x80000000:
            h -= 0x100000000
    u = h & 0xFFFFFFFF
    if u == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while u:
        out = digits[u % 36] + out
        u //= 36
    return out


def _kind_key(kind) -> str:
    import re
    return re.sub(r"[^a-z0-9_-]", "", str(kind or "").lower()) or "action"


def move_key(action: dict) -> str:
    """The board's stable per-move key: the generator's dismiss_key when it
    set one, else the same kind+title hash the client derives.

    Keys are sanitized to the dismissal store's item_key alphabet
    ([A-Za-z0-9:._+-]) so a key round-trips through both stores: the swap
    factory's "swap:<out>><in>" becomes "swap:<out>-<in>"."""
    import re
    dk = str((action or {}).get("dismiss_key") or "")
    if dk:
        return re.sub(r"[^A-Za-z0-9:._+-]", "-", dk)[:96]
    kind = str((action or {}).get("kind") or "")
    title = str((action or {}).get("title") or "")
    return f"pf:{_kind_key(kind)}:{_js_hash(kind + '|' + title)}"


# ── Timing / locked state ────────────────────────────────────────────────────

def _fmt_et_time(epoch) -> str:
    try:
        from zoneinfo import ZoneInfo
        dt = datetime.fromtimestamp(float(epoch), ZoneInfo("America/New_York"))
    except Exception:
        return ""
    return dt.strftime("%-I:%M %p")


def timing_for(action: dict, now_ts: float) -> dict:
    """Timing label + lock state for one board move.

    The scan stamps ``lock_epoch`` when it knows the relevant kickoff (the
    involved players' earliest game, else the week's first Sunday kickoff).
    A move locks when that time passes; waiver/calendar/trade moves have no
    kickoff lock and never render locked."""
    kind = str((action or {}).get("kind") or "")
    lock_epoch = (action or {}).get("lock_epoch")
    try:
        lock_epoch = float(lock_epoch) if lock_epoch else None
    except (TypeError, ValueError):
        lock_epoch = None
    if kind == "waiver":
        return {"label": "Before waivers", "locked": False, "lock_label": ""}
    if kind in ("calendar", "trade"):
        return {"label": "This week", "locked": False, "lock_label": ""}
    if kind in ("injury", "roster"):
        label = "Do now"
    else:
        label = "Before Sunday"
    if lock_epoch is not None:
        when = _fmt_et_time(lock_epoch)
        if now_ts >= lock_epoch:
            return {"label": label, "locked": True,
                    "lock_label": f"Locked · {when}" if when else "Locked"}
        # Same-day future kickoff reads as a lock countdown, per the mock.
        try:
            from zoneinfo import ZoneInfo
            now_d = datetime.fromtimestamp(now_ts, ZoneInfo("America/New_York")).date()
            lock_d = datetime.fromtimestamp(lock_epoch, ZoneInfo("America/New_York")).date()
        except Exception:
            now_d = lock_d = None
        if when and now_d is not None and now_d == lock_d:
            return {"label": f"Locks {when}", "locked": False, "lock_label": ""}
    return {"label": label, "locked": False, "lock_label": ""}


# ── Board grouping ───────────────────────────────────────────────────────────

def group_actions(actions, meta_by_key: Optional[dict] = None) -> List[dict]:
    """Group ranked actions into league groups with global rank numbers.

    ``actions`` arrive priority-ranked (rank_cross_league_actions order);
    ranks are assigned globally in that order, groups keep first-appearance
    order. ``meta_by_key`` maps "platform|league_id" to league header meta
    (record line, odds, posture, proj tag); missing meta degrades to the
    action's own league_name."""
    groups: List[dict] = []
    by_key: Dict[str, dict] = {}
    rank = 0
    for action in actions or []:
        if not isinstance(action, dict) or not action.get("title"):
            continue
        rank += 1
        plat = str(action.get("platform") or "sleeper").lower()
        lid = str(action.get("league_id") or "")
        key = f"{plat}|{lid}"
        group = by_key.get(key)
        if group is None:
            meta = dict((meta_by_key or {}).get(key) or {})
            group = {
                "key": key,
                "platform": plat,
                "league_id": lid,
                "season": action.get("season"),
                "league_name": meta.get("league_name") or action.get("league_name") or lid,
                "meta": meta,
                "moves": [],
            }
            by_key[key] = group
            groups.append(group)
        move = dict(action)
        move["rank"] = rank
        move["move_key"] = move_key(action)
        group["moves"].append(move)
    return groups


# ── Done-state verification ──────────────────────────────────────────────────

def verify_done(action: dict, roster: Optional[dict]) -> Dict[str, object]:
    """Decide whether a Done claim can be Verified from a roster read-back.

    ``roster`` is the viewer's current roster row from the (cached) league
    context: players / starters / reserve pid lists. Only checks with an
    actual read-back verify; everything else returns an honest marked
    reason. Never claims verified without the check passing."""
    plat = _PLAT_LABELS.get(str((action or {}).get("platform") or "sleeper").lower(),
                            "the platform")
    spec = (action or {}).get("verify") or {}
    vtype = str(spec.get("type") or "")
    if roster is None:
        return {"verified": False, "label": "League data unavailable for read-back"}
    players = {str(p) for p in (roster.get("players") or []) if p}
    starters = {str(p) for p in (roster.get("starters") or []) if p}
    reserve = {str(p) for p in (roster.get("reserve") or []) if p}
    if vtype == "swap":
        in_pid = str(spec.get("in") or "")
        out_pid = str(spec.get("out") or "")
        if in_pid and in_pid in starters and (not out_pid or out_pid not in starters):
            return {"verified": True, "label": f"Lineup set in {plat}"}
        return {"verified": False, "label": "Read-back does not show the swap yet"}
    if vtype == "ir":
        pid = str(spec.get("pid") or "")
        if pid and pid in reserve:
            return {"verified": True, "label": f"IR slot set in {plat}"}
        return {"verified": False, "label": "Read-back does not show the IR move yet"}
    if vtype == "activate":
        pid = str(spec.get("pid") or "")
        if pid and pid in players and pid not in reserve:
            return {"verified": True, "label": f"Player active in {plat}"}
        return {"verified": False, "label": "Read-back does not show the activation yet"}
    if vtype == "drop":
        pid = str(spec.get("pid") or "")
        if pid and pid not in players:
            return {"verified": True, "label": f"Player dropped in {plat}"}
        return {"verified": False, "label": "Read-back still shows the player rostered"}
    if vtype == "claim":
        return {"verified": False, "label": "Claims have not processed yet"}
    return {"verified": False, "label": "No read-back available for this move"}


# ── Done store ───────────────────────────────────────────────────────────────

_DONE_TABLE_READY = False


def _ensure_done_table() -> bool:
    global _DONE_TABLE_READY
    if _DONE_TABLE_READY:
        return True
    try:
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_move_states (
                    user_key     TEXT NOT NULL,
                    platform     TEXT NOT NULL,
                    league_id    TEXT NOT NULL,
                    season       INTEGER NOT NULL,
                    week         INTEGER NOT NULL,
                    move_key     TEXT NOT NULL,
                    verified     BOOLEAN NOT NULL DEFAULT FALSE,
                    verify_label TEXT NOT NULL DEFAULT '',
                    done_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_key, platform, league_id, season, week, move_key)
                )
                """
            )
            conn.commit()
        _DONE_TABLE_READY = True
    except Exception:
        logger.debug("portfolio_board: done table init failed", exc_info=True)
    return _DONE_TABLE_READY


def load_done_states(user_key: str, season: int, week: int) -> Dict[tuple, dict]:
    """{(platform, league_id, move_key): {verified, label, done_at}} for a week."""
    if not user_key or not _ensure_done_table():
        return {}
    try:
        with get_conn() as conn:
            rows = conn.execute(
                """
                SELECT platform, league_id, move_key, verified, verify_label, done_at
                FROM portfolio_move_states
                WHERE user_key = %s AND season = %s AND week = %s
                """,
                (str(user_key), int(season), int(week)),
            ).fetchall()
    except Exception:
        logger.debug("portfolio_board: done load failed", exc_info=True)
        return {}
    out: Dict[tuple, dict] = {}
    for row in rows or []:
        d = row if isinstance(row, dict) else {
            "platform": row[0], "league_id": row[1], "move_key": row[2],
            "verified": row[3], "verify_label": row[4], "done_at": row[5],
        }
        done_at = d.get("done_at")
        out[(str(d["platform"]), str(d["league_id"]), str(d["move_key"]))] = {
            "verified": bool(d.get("verified")),
            "label": str(d.get("verify_label") or ""),
            "done_at": done_at.isoformat() if hasattr(done_at, "isoformat") else str(done_at or ""),
        }
    return out


def set_done(user_key: str, platform: str, league_id: str, season: int,
             week: int, key: str, verified: bool, label: str) -> bool:
    if not user_key or not _ensure_done_table():
        return False
    try:
        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_move_states
                    (user_key, platform, league_id, season, week, move_key,
                     verified, verify_label, done_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (user_key, platform, league_id, season, week, move_key)
                DO UPDATE SET verified = EXCLUDED.verified,
                              verify_label = EXCLUDED.verify_label,
                              done_at = NOW()
                """,
                (str(user_key), str(platform).lower(), str(league_id),
                 int(season), int(week), str(key), bool(verified), str(label or "")),
            )
            conn.commit()
        return True
    except Exception:
        logger.debug("portfolio_board: set_done failed", exc_info=True)
        return False


def clear_done(user_key: str, platform: str, league_id: str, season: int,
               week: int, key: str) -> bool:
    if not user_key or not _ensure_done_table():
        return False
    try:
        with get_conn() as conn:
            conn.execute(
                """
                DELETE FROM portfolio_move_states
                WHERE user_key = %s AND platform = %s AND league_id = %s
                  AND season = %s AND week = %s AND move_key = %s
                """,
                (str(user_key), str(platform).lower(), str(league_id),
                 int(season), int(week), str(key)),
            )
            conn.commit()
        return True
    except Exception:
        logger.debug("portfolio_board: clear_done failed", exc_info=True)
        return False


# ── Playoff odds for one league (cache-only) ─────────────────────────────────

def odds_for_ctx(ctx: dict, platform: str, roster_id) -> Dict[str, object]:
    """{pct, delta} for the viewer's roster from the sim cache.

    block=False throughout: a cold sim backgrounds and this returns Nones,
    so a card or board header simply shows odds on a later load (the
    Front Office report's convention). ``delta`` is points of movement vs
    the most recent earlier daily snapshot in playoff_odds_daily."""
    out: Dict[str, object] = {"pct": None, "delta": None}
    if not ctx or roster_id in (None, ""):
        return out
    try:
        from app import _playoff_sim_cached
        rows = _playoff_sim_cached(ctx, str(platform or "sleeper"), block=False) or []
    except Exception:
        return out
    mine = None
    for row in rows:
        if str(row.get("roster_id")) == str(roster_id):
            mine = row
            break
    if not mine or mine.get("playoff_pct") is None:
        return out
    try:
        out["pct"] = round(float(mine["playoff_pct"]), 1)
    except (TypeError, ValueError):
        return out
    try:
        from dashboard_services.playoff_odds_history import record_daily_and_movement
        movement = record_daily_and_movement(
            str(ctx.get("league_id") or ""), int(ctx.get("season") or 0),
            rows, write=False,
        ) or {}
        d = movement.get(str(roster_id))
        if d is not None:
            out["delta"] = round(float(d), 1)
    except Exception:
        pass
    return out


# ── Board snapshots + self-grading ───────────────────────────────────────────

_SNAP_TABLE_READY = False


def _ensure_snap_tables() -> bool:
    global _SNAP_TABLE_READY
    if _SNAP_TABLE_READY:
        return True
    try:
        with get_conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_board_snapshots (
                    user_key   TEXT NOT NULL,
                    season     INTEGER NOT NULL,
                    week       INTEGER NOT NULL,
                    payload    TEXT NOT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_key, season, week)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_board_grades (
                    user_key     TEXT NOT NULL,
                    season       INTEGER NOT NULL,
                    week         INTEGER NOT NULL,
                    followed     INTEGER NOT NULL,
                    total        INTEGER NOT NULL,
                    delta_pts    DECIMAL(6,1),
                    skipped_cost DECIMAL(6,1),
                    graded_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (user_key, season, week)
                )
                """
            )
            conn.commit()
        _SNAP_TABLE_READY = True
    except Exception:
        logger.debug("portfolio_board: snapshot table init failed", exc_info=True)
    return _SNAP_TABLE_READY


def save_board_snapshot(user_key: str, season: int, week: int,
                        groups: List[dict]) -> bool:
    """Persist the served board (move keys + verify specs) for later grading.

    Only what grading needs is stored: per move, its scope, key, kind and
    verify spec. Done/dismissed state is read from the stores at grade time."""
    if not user_key or not _ensure_snap_tables():
        return False
    moves = []
    for group in groups or []:
        for move in group.get("moves") or []:
            moves.append({
                "platform": group.get("platform") or move.get("platform") or "sleeper",
                "league_id": group.get("league_id") or move.get("league_id") or "",
                "move_key": move.get("move_key") or move_key(move),
                "kind": move.get("kind") or "",
                "verify": move.get("verify") or {},
                "roster_id": (group.get("meta") or {}).get("roster_id") or "",
            })
    if not moves:
        return False
    try:
        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_board_snapshots (user_key, season, week, payload, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (user_key, season, week)
                DO UPDATE SET payload = EXCLUDED.payload, updated_at = NOW()
                """,
                (str(user_key), int(season), int(week), json.dumps({"moves": moves})),
            )
            conn.commit()
        return True
    except Exception:
        logger.debug("portfolio_board: snapshot save failed", exc_info=True)
        return False


def _load_snapshot(user_key: str, season: int, week: int) -> Optional[dict]:
    if not _ensure_snap_tables():
        return None
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT payload FROM portfolio_board_snapshots "
                "WHERE user_key = %s AND season = %s AND week = %s",
                (str(user_key), int(season), int(week)),
            ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    payload = row["payload"] if isinstance(row, dict) else row[0]
    try:
        return json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, ValueError):
        return None


def grade_snapshot_moves(moves: List[dict], done_keys: set,
                         points_by_scope: Dict[tuple, dict]) -> Optional[dict]:
    """Pure grading core. ``points_by_scope`` maps (platform, league_id) to
    the viewer's provider matchup row ``players_points`` (league scoring).

    Only swap moves grade: delta = actual(in) - actual(out). Followed swaps
    sum into ``delta_pts`` (what following the board was worth vs leaving
    the lineup as it was); skipped swaps with a positive delta sum into
    ``skipped_cost``. Returns None when nothing on the board is gradeable,
    so an ungradeable week never renders a fabricated grade."""
    followed = 0
    gradeable = 0
    delta_pts = 0.0
    skipped_cost = 0.0
    for move in moves or []:
        key = (str(move.get("platform") or "sleeper"),
               str(move.get("league_id") or ""), str(move.get("move_key") or ""))
        is_done = key in done_keys
        if is_done:
            followed += 1
        spec = move.get("verify") or {}
        if str(spec.get("type") or "") != "swap":
            continue
        points = points_by_scope.get((key[0], key[1])) or {}
        in_pid, out_pid = str(spec.get("in") or ""), str(spec.get("out") or "")
        if in_pid not in points or out_pid not in points:
            continue
        try:
            delta = float(points[in_pid]) - float(points[out_pid])
        except (TypeError, ValueError):
            continue
        gradeable += 1
        if is_done:
            delta_pts += delta
        elif delta > 0:
            skipped_cost += delta
    if not gradeable:
        return None
    return {
        "followed": followed,
        "total": len(moves or []),
        "delta_pts": round(delta_pts, 1),
        "skipped_cost": round(skipped_cost, 1),
        "gradeable": gradeable,
    }


def grade_board_week(user_key: str, season: int, week: int,
                     points_loader=None) -> Optional[dict]:
    """Grade one snapshotted board week; persist and return the grade.

    ``points_loader(platform, league_id, season, week, roster_id)`` must
    return the viewer's matchup-row players_points (or None). Defaults to
    the provider matchup fetch start_sit_tracking grades from."""
    if not user_key or not _ensure_snap_tables():
        return None
    existing = load_board_grade(user_key, season, week)
    if existing:
        return existing
    snap = _load_snapshot(user_key, season, week)
    if not snap:
        return None
    moves = snap.get("moves") or []
    if not moves:
        return None
    if points_loader is None:
        def points_loader(platform, league_id, season_, week_, roster_id):
            try:
                from dashboard_services.platform_api import get_matchups
                rows = get_matchups(platform, league_id, int(week_), int(season_)) or []
            except Exception:
                return None
            for row in rows:
                if str(row.get("roster_id")) == str(roster_id):
                    return row.get("players_points") or {}
            return None
    scopes = {(str(m.get("platform") or "sleeper"), str(m.get("league_id") or ""))
              for m in moves if (m.get("verify") or {}).get("type") == "swap"}
    points_by_scope: Dict[tuple, dict] = {}
    roster_by_scope = {}
    for m in moves:
        roster_by_scope[(str(m.get("platform") or "sleeper"),
                         str(m.get("league_id") or ""))] = m.get("roster_id") or ""
    for scope in scopes:
        pts = points_loader(scope[0], scope[1], season, week, roster_by_scope.get(scope, ""))
        if pts is not None:
            points_by_scope[scope] = {str(k): v for k, v in pts.items()}
    done = load_done_states(user_key, season, week)
    done_keys = set(done.keys())
    grade = grade_snapshot_moves(moves, done_keys, points_by_scope)
    if grade is None:
        return None
    grade["week"] = int(week)
    try:
        with get_conn() as conn:
            conn.execute(
                """
                INSERT INTO portfolio_board_grades
                    (user_key, season, week, followed, total, delta_pts, skipped_cost, graded_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (user_key, season, week) DO NOTHING
                """,
                (str(user_key), int(season), int(week), grade["followed"],
                 grade["total"], grade["delta_pts"], grade["skipped_cost"]),
            )
            conn.commit()
    except Exception:
        logger.debug("portfolio_board: grade save failed", exc_info=True)
    return grade


def load_board_grade(user_key: str, season: int, week: int) -> Optional[dict]:
    if not user_key or not _ensure_snap_tables():
        return None
    try:
        with get_conn() as conn:
            row = conn.execute(
                "SELECT followed, total, delta_pts, skipped_cost "
                "FROM portfolio_board_grades "
                "WHERE user_key = %s AND season = %s AND week = %s",
                (str(user_key), int(season), int(week)),
            ).fetchone()
    except Exception:
        return None
    if not row:
        return None
    d = row if isinstance(row, dict) else {
        "followed": row[0], "total": row[1], "delta_pts": row[2], "skipped_cost": row[3],
    }
    return {
        "week": int(week),
        "followed": int(d["followed"] or 0),
        "total": int(d["total"] or 0),
        "delta_pts": float(d["delta_pts"]) if d["delta_pts"] is not None else None,
        "skipped_cost": float(d["skipped_cost"]) if d["skipped_cost"] is not None else None,
    }


def grade_line(grade: Optional[dict]) -> str:
    """The board's self-grade footer line, or "" when there is no grade."""
    if not grade:
        return ""
    week = grade.get("week")
    line = (f"Week {week} board: you followed {grade.get('followed', 0)} of "
            f"{grade.get('total', 0)} moves")
    delta = grade.get("delta_pts")
    if delta is not None:
        line += f" · worth {delta:+.1f} pts vs leaving the lineups as they were."
    else:
        line += "."
    skipped = grade.get("skipped_cost")
    if skipped:
        line += f" The moves you skipped cost {skipped:.1f}."
    return line
