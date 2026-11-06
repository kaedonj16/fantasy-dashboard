"""Draft Database: a Sleeper-only draft corpus browser, modeled on the
Trade Database (routes/trade_bp.py page + /api/trade-database in app.py).

Routes:
    /draft-database                                  (guest page)
    /<platform>/<season>/<league_id>/draft-database  (league-context page)
    /api/draft-database                              (paginated list)
    /api/draft-database/count                        (cached corpus count)
    /api/draft-database/<draft_id>/board             (full board)
    /api/draft-database/adp                          (ADP view data)

Data: draft_adp_drafts / draft_adp_picks / draft_adp (migrations 015,
034, 056), filled by data_building/trade_intel/draft_adp_crawler.py.
Corpus = completed Sleeper drafts only. Grade superlatives come from the
persisted grade_summary (utils/draft_db_grades.py) and always name the
family they came from (at_draft vs performance).

YOU rule: a viewer is marked on a board ONLY when one of their Sleeper
user ids owns a roster slot in that draft (persisted roster_owners).
Everyone else gets zero YOU markers and no "your" language.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone

from flask import Blueprint, jsonify, request, session

logger = logging.getLogger(__name__)

draft_db_bp = Blueprint("draft_db", __name__)

# Cached COUNT keyed by filter fingerprint (Trade DB policy: counts only
# change when the crawl ingests, so a 1h TTL is safe).
_COUNT_CACHE: dict = {}
_COUNT_TTL = 3600

# ADP view: prefer drafts from the last 30 days; below this many drafts
# in the window the segment is too thin to quote, so the view falls back
# to the season aggregate table and says so on the basis line.
ADP_RECENT_WINDOW_DAYS = 30
ADP_RECENT_MIN_DRAFTS = 10

TEAM_SIZES = (8, 10, 12, 14, 16)
DRAFT_TYPES = ("startup", "rookie", "redraft")
LEAGUE_FORMATS = ("dynasty", "redraft", "keeper")
SORTS = ("recent", "most_picks", "best_grade")

_GRADE_VAL = {"A+": 5, "A": 4, "B": 3, "C": 2, "D": 1, "F": 0}

POS_COLORS = {"QB": "#3b82f6", "RB": "#22c55e", "WR": "#f59e0b",
              "TE": "#8b5cf6", "K": "#c92c68", "DEF": "#475569"}


# ── Filter parsing / SQL ────────────────────────────────────────────────────

def parse_filters(args) -> dict:
    """Normalize the list/ADP query params. Unknown values degrade to
    'all'/defaults rather than erroring, mirroring /api/trade-database."""
    def _s(key, default="all"):
        return (args.get(key) or default).strip().lower()

    try:
        page = max(0, int(args.get("page") or 0))
    except (TypeError, ValueError):
        page = 0
    try:
        limit = min(max(1, int(args.get("limit") or 20)), 50)
    except (TypeError, ValueError):
        limit = 20
    try:
        season = int(args.get("season") or 0) or None
    except (TypeError, ValueError):
        season = None
    try:
        num_teams = int(args.get("num_teams") or 0) or None
    except (TypeError, ValueError):
        num_teams = None
    if num_teams not in TEAM_SIZES:
        num_teams = num_teams if num_teams else None
    draft_type = _s("draft_type")
    if draft_type not in DRAFT_TYPES:
        draft_type = "all"
    league_format = _s("format")
    if league_format not in LEAGUE_FORMATS:
        league_format = "all"
    league_type = _s("league_type")
    if league_type not in ("1qb", "sf"):
        league_type = "all"
    sort = _s("sort", "recent")
    if sort not in SORTS:
        sort = "recent"
    return {
        "q": (args.get("q") or "").strip(),
        "season": season,
        "draft_type": draft_type,
        "format": league_format,
        "league_type": league_type,
        "num_teams": num_teams,
        "sort": sort,
        "page": page,
        "limit": limit,
    }


def build_filter_sql(f: dict, match_ids: list | None = None):
    """(where_sql, params) over draft_adp_drafts aliased ``d``.

    q matches a league name substring, or any draft where one of the
    resolved player ids was picked. NULL metadata never matches a
    specific format filter (honest NULLs, migration 056)."""
    clauses = ["d.status = 'complete'"]
    params: list = []
    if f.get("season"):
        clauses.append("d.season = %s")
        params.append(f["season"])
    if f.get("draft_type") not in (None, "all"):
        clauses.append("d.draft_type = %s")
        params.append(f["draft_type"])
    if f.get("format") not in (None, "all"):
        clauses.append("d.league_format = %s")
        params.append(f["format"])
    if f.get("league_type") == "sf":
        clauses.append("d.is_superflex = TRUE")
    elif f.get("league_type") == "1qb":
        clauses.append("d.is_superflex = FALSE")
    if f.get("num_teams"):
        clauses.append("d.num_teams = %s")
        params.append(f["num_teams"])
    if f.get("q"):
        sub = ["d.league_name ILIKE %s"]
        params.append(f"%{f['q']}%")
        if match_ids:
            sub.append(
                "d.draft_id IN (SELECT draft_id FROM draft_adp_picks "
                "WHERE player_id = ANY(%s))")
            params.append(list(match_ids))
        clauses.append("(" + " OR ".join(sub) + ")")
    return " AND ".join(clauses), params


def _fingerprint(f: dict, match_ids: list | None) -> str:
    raw = json.dumps({
        "q": (f.get("q") or "").lower(),
        "season": f.get("season"), "draft_type": f.get("draft_type"),
        "format": f.get("format"), "league_type": f.get("league_type"),
        "num_teams": f.get("num_teams"),
        "match": sorted(match_ids or []),
    }, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def _cached_count(conn, f: dict, match_ids: list | None) -> int:
    key = _fingerprint(f, match_ids)
    now = time.time()
    hit = _COUNT_CACHE.get(key)
    if hit and (now - hit[0]) < _COUNT_TTL:
        return hit[1]
    where, params = build_filter_sql(f, match_ids)
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM draft_adp_drafts d WHERE {where}",
        params,
    ).fetchone()
    total = int(row["n"]) if row else 0
    _COUNT_CACHE[key] = (now, total)
    if len(_COUNT_CACHE) > 200:
        for k in list(_COUNT_CACHE)[:50]:
            del _COUNT_CACHE[k]
    return total


def _as_json(value):
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


def _players_map():
    from utils.utils import load_players_index
    return load_players_index() or {}


def _player_out(pid, players_map) -> dict:
    info = players_map.get(str(pid)) or {}
    espn_id = str(info.get("espnID") or info.get("espn_id") or "")
    return {
        "player_id": str(pid),
        "name": info.get("name") or str(pid),
        "position": info.get("pos") or info.get("position") or "",
        "team": info.get("team") or "",
        "espn_id": espn_id or None,
        "headshot": (
            f"https://a.espncdn.com/i/headshots/nfl/players/full/{espn_id}.png"
            if espn_id else None
        ),
    }


def _enrich_pick(pick, players_map):
    if not pick:
        return None
    out = dict(pick)
    out.update({k: v for k, v in _player_out(pick.get("player_id"), players_map).items()
                if k != "player_id"})
    return out


# ── Viewer identity (YOU rule) ──────────────────────────────────────────────

def viewer_sleeper_ids(sess=None) -> set:
    """The viewer's Sleeper user ids: the session's viewer id plus any
    Sleeper ids linked to their account (the same sources the draft
    endpoints use: session viewer_user_id + account platform links)."""
    sess = sess if sess is not None else session
    ids = set()
    vid = str(sess.get("viewer_user_id") or "")
    if vid:
        ids.add(vid)
    account_id = sess.get("account_id")
    if account_id:
        try:
            from dashboard_services.accounts import list_account_platform_ids
            for x in list_account_platform_ids(account_id, "sleeper") or []:
                if x:
                    ids.add(str(x))
        except Exception:
            logger.debug("[draft-db] linked sleeper ids lookup failed",
                         exc_info=True)
    return ids


def viewer_roster_ids(draft_row: dict, sess=None) -> set:
    """Roster ids in this draft owned by the viewer (empty when the
    viewer is not in the draft or ownership is unknown)."""
    owners = _as_json(draft_row.get("roster_owners")) or {}
    if not isinstance(owners, dict) or not owners:
        return set()
    mine = viewer_sleeper_ids(sess)
    if not mine:
        return set()
    return {str(rid) for rid, owner in owners.items() if str(owner) in mine}


# ── List API ────────────────────────────────────────────────────────────────

_SORT_SQL = {
    "recent": "d.draft_started_at DESC NULLS LAST, d.crawled_at DESC",
    "most_picks": "d.total_picks DESC NULLS LAST, d.draft_started_at DESC NULLS LAST",
    # Best grade: order by the persisted best-draft letter value via the
    # grade summary JSON (letters map to the endpoint's grade values).
    "best_grade": (
        "(CASE d.grade_summary->'at_draft'->'best_draft'->>'grade' "
        "WHEN 'A+' THEN 5 WHEN 'A' THEN 4 WHEN 'B' THEN 3 "
        "WHEN 'C' THEN 2 WHEN 'D' THEN 1 WHEN 'F' THEN 0 END) "
        "DESC NULLS LAST, d.draft_started_at DESC NULLS LAST"
    ),
}


@draft_db_bp.route("/api/draft-database")
def api_draft_database():
    """Paginated draft list. Mirrors /api/trade-database semantics:
    page is 0-based, limit defaults to 20 and caps at 50, and the total
    comes from a COUNT cached by filter fingerprint for 1 hour."""
    try:
        f = parse_filters(request.args)
        players_map = _players_map()
        match_ids: list = []
        if f["q"]:
            from utils.players import name_match_query as _nmq, name_match_score as _nms
            _pq = _nmq(f["q"].lower())
            match_ids = [
                pid for pid, info in players_map.items()
                if _nms(info.get("name") or "", _pq) > 0
            ]
        from dashboard_services.db import get_conn
        where, params = build_filter_sql(f, match_ids)
        with get_conn() as conn:
            total = _cached_count(conn, f, match_ids)
            rows = conn.execute(
                f"""
                SELECT d.draft_id, d.league_id, d.league_name, d.season,
                       d.draft_type, d.league_format, d.scoring_type,
                       d.is_superflex, d.num_teams, d.rounds, d.total_picks,
                       d.draft_started_at, d.grade_summary
                FROM draft_adp_drafts d
                WHERE {where}
                ORDER BY {_SORT_SQL[f['sort']]}
                LIMIT %s OFFSET %s
                """,
                params + [f["limit"] + 1, f["page"] * f["limit"]],
            ).fetchall()
        has_more = len(rows) > f["limit"]
        rows = rows[: f["limit"]]

        from utils.draft_db_grades import headline_superlatives
        out = []
        for r in rows:
            summary = _as_json(r.get("grade_summary"))
            head = headline_superlatives(summary)
            started = r.get("draft_started_at")
            out.append({
                "draft_id": r["draft_id"],
                "league_id": r["league_id"],
                "league_name": r.get("league_name") or f"League {r['league_id']}",
                "league_name_known": bool(r.get("league_name")),
                "season": r["season"],
                "draft_type": r["draft_type"],
                "league_format": r.get("league_format"),
                "scoring_type": r.get("scoring_type"),
                "is_superflex": bool(r["is_superflex"]),
                "num_teams": r["num_teams"],
                "rounds": r["rounds"],
                "total_picks": r["total_picks"],
                "draft_started_at": started.isoformat()
                    if hasattr(started, "isoformat") else None,
                "best_draft": head["best_draft"],
                "best_draft_family": head["best_draft_family"],
                "biggest_steal": _enrich_pick(head["biggest_steal"], players_map),
                "biggest_bust": _enrich_pick(head["biggest_bust"], players_map),
                "superlative_family": head["family"],
                "performance_note": (summary or {}).get("performance_note"),
            })
        total_pages = (total + f["limit"] - 1) // f["limit"] if total else 0
        return jsonify({
            "drafts": out,
            "total": total,
            "has_more": has_more,
            "pagination": {
                "current_page": f["page"] + 1,
                "per_page": f["limit"],
                "total_players": total,
                "total_pages": total_pages,
                "has_next": has_more,
                "has_prev": f["page"] > 0,
            },
        })
    except Exception:
        logger.exception("[draft-database] list error")
        return jsonify({"error": "Internal error"}), 500


@draft_db_bp.route("/api/draft-database/count")
def api_draft_database_count():
    """Corpus count for the page header (cached, filter-aware)."""
    try:
        f = parse_filters(request.args)
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            total = _cached_count(conn, f, None)
        return jsonify({"count": total})
    except Exception:
        logger.exception("[draft-database] count error")
        return jsonify({"count": 0})


# ── Board API ───────────────────────────────────────────────────────────────

@draft_db_bp.route("/api/draft-database/<draft_id>/board")
def api_draft_database_board(draft_id: str):
    """Full board for one draft, in R1 cell order: teams are columns in
    slot order, rounds are rows, snake rounds reverse. Per-cell grades
    come from the persisted grade summary (at-draft letter plus the
    performance verdict when that family exists)."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            d = conn.execute(
                "SELECT * FROM draft_adp_drafts WHERE draft_id = %s",
                (draft_id,),
            ).fetchone()
            if not d:
                return jsonify({"error": "Draft not found"}), 404
            picks = conn.execute(
                "SELECT player_id, pick_no, round, pick_in_round, roster_id "
                "FROM draft_adp_picks WHERE draft_id = %s ORDER BY pick_no",
                (draft_id,),
            ).fetchall()
        players_map = _players_map()
        summary = _as_json(d.get("grade_summary")) or {}
        per_pick = summary.get("per_pick") or {}
        roster_names = _as_json(d.get("roster_names")) or {}
        slot_map = _as_json(d.get("slot_to_roster_id")) or {}

        # Column order: persisted slot -> roster map first; any roster
        # that drafted but has no slot entry appends in roster order.
        columns: list[str] = []
        if isinstance(slot_map, dict):
            for slot in sorted(slot_map, key=lambda k: int(k) if str(k).isdigit() else 0):
                rid = str(slot_map[slot])
                if rid not in columns:
                    columns.append(rid)
        for p in picks:
            rid = str(p.get("roster_id") or "")
            if rid and rid not in columns:
                columns.append(rid)
        mine = viewer_roster_ids(d)
        teams = [{
            "slot": i + 1,
            "roster_id": rid,
            "name": (roster_names.get(rid) or f"Team {i + 1}"),
            "is_you": rid in mine,
        } for i, rid in enumerate(columns)]

        order_type = (d.get("draft_order_type") or "").lower()
        rounds_map: dict[int, dict[str, dict]] = {}
        for p in picks:
            rid = str(p.get("roster_id") or "")
            rnd = int(p.get("round") or 0)
            if not rnd or rid not in columns:
                continue
            info = _player_out(p["player_id"], players_map)
            grades = per_pick.get(str(p["player_id"])) or {}
            rounds_map.setdefault(rnd, {})[rid] = {
                "pick_no": p["pick_no"],
                "round": rnd,
                "pick_in_round": p.get("pick_in_round"),
                "roster_id": rid,
                **info,
                "grade": grades.get("grade"),
                "adp_diff": grades.get("adp_diff"),
                "verdict": grades.get("verdict"),
                "is_you": rid in mine,
            }
        rounds = []
        for rnd in sorted(rounds_map):
            row_cells = [rounds_map[rnd].get(rid) for rid in columns]
            if order_type == "snake" and rnd % 2 == 0:
                row_cells = list(reversed(row_cells))
            rounds.append({"round": rnd, "cells": row_cells})

        started = d.get("draft_started_at")
        return jsonify({
            "draft": {
                "draft_id": d["draft_id"],
                "league_id": d["league_id"],
                "league_name": d.get("league_name") or f"League {d['league_id']}",
                "league_name_known": bool(d.get("league_name")),
                "season": d["season"],
                "draft_type": d["draft_type"],
                "league_format": d.get("league_format"),
                "scoring_type": d.get("scoring_type"),
                "is_superflex": bool(d["is_superflex"]),
                "num_teams": d["num_teams"],
                "rounds": d["rounds"],
                "total_picks": d["total_picks"],
                "draft_order_type": d.get("draft_order_type"),
                "draft_started_at": started.isoformat()
                    if hasattr(started, "isoformat") else None,
                "performance_note": summary.get("performance_note"),
            },
            "teams": teams,
            "rounds": rounds,
            "viewer_in_draft": bool(mine),
        })
    except Exception:
        logger.exception("[draft-database] board error")
        return jsonify({"error": "Internal error"}), 500


# ── ADP API ─────────────────────────────────────────────────────────────────

def choose_adp_basis(recent_draft_count: int) -> str:
    """'last_30_days' when the recent window has enough drafts to quote,
    else 'season_aggregate' (the draft_adp table)."""
    return ("last_30_days" if recent_draft_count >= ADP_RECENT_MIN_DRAFTS
            else "season_aggregate")


def _segment_label(f: dict) -> str:
    parts = []
    if f.get("draft_type") not in (None, "all"):
        parts.append(f["draft_type"])
    if f.get("league_type") == "sf":
        parts.append("Superflex")
    elif f.get("league_type") == "1qb":
        parts.append("1QB")
    if f.get("num_teams"):
        parts.append(f"{f['num_teams']}-team")
    if f.get("season"):
        parts.append(str(f["season"]))
    return ", ".join(parts) if parts else "all drafts"


@draft_db_bp.route("/api/draft-database/adp")
def api_draft_database_adp():
    """ADP for the filtered segment. Last-30-days cut from raw picks
    when at least ADP_RECENT_MIN_DRAFTS drafts in the window match;
    otherwise the season aggregate (draft_adp). The basis line names
    the exact filters and which basis was used."""
    try:
        f = parse_filters(request.args)
        season = f.get("season") or datetime.now().year
        draft_type = f["draft_type"]
        if draft_type == "all":
            draft_type = "startup"
        is_sf = True if f["league_type"] == "sf" else False
        num_teams = f.get("num_teams") or 12
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            days=ADP_RECENT_WINDOW_DAYS)
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            recent = conn.execute(
                "SELECT COUNT(*) AS n FROM draft_adp_drafts "
                "WHERE draft_type = %s AND season = %s AND is_superflex = %s "
                "AND num_teams = %s AND draft_started_at >= %s",
                (draft_type, season, is_sf, num_teams, cutoff),
            ).fetchone()
            recent_n = int(recent["n"]) if recent else 0
            basis = choose_adp_basis(recent_n)
            if basis == "last_30_days":
                rows = conn.execute(
                    """
                    SELECT p.player_id,
                           ROUND(AVG(p.pick_no)::numeric, 2) AS avg_pick,
                           ROUND(STDDEV(p.pick_no)::numeric, 2) AS std_pick,
                           ROUND(AVG(p.round)::numeric, 2) AS avg_round,
                           COUNT(DISTINCT p.draft_id) AS drafts,
                           COUNT(*) AS sample_size
                    FROM draft_adp_picks p
                    JOIN draft_adp_drafts d ON d.draft_id = p.draft_id
                    WHERE d.draft_type = %s AND d.season = %s
                      AND d.is_superflex = %s AND d.num_teams = %s
                      AND d.draft_started_at >= %s
                    GROUP BY p.player_id
                    ORDER BY avg_pick ASC
                    LIMIT 300
                    """,
                    (draft_type, season, is_sf, num_teams, cutoff),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT player_id, avg_pick, std_pick, avg_round,
                           sample_size,
                           (SELECT COUNT(*) FROM draft_adp_drafts
                            WHERE draft_type = %s AND season = %s
                              AND is_superflex = %s AND num_teams = %s)
                               AS drafts
                    FROM draft_adp
                    WHERE draft_type = %s AND season = %s
                      AND is_superflex = %s AND num_teams = %s
                    ORDER BY avg_pick ASC
                    LIMIT 300
                    """,
                    (draft_type, season, is_sf, num_teams,
                     draft_type, season, is_sf, num_teams),
                ).fetchall()
        players_map = _players_map()
        items = []
        seg_drafts = recent_n
        for r in rows:
            info = _player_out(r["player_id"], players_map)
            if r.get("drafts") is not None:
                seg_drafts = max(seg_drafts, int(r["drafts"] or 0))
            items.append({
                **info,
                "adp": float(r["avg_pick"]) if r.get("avg_pick") is not None else None,
                "avg_round": float(r["avg_round"]) if r.get("avg_round") is not None else None,
                "std_pick": float(r["std_pick"]) if r.get("std_pick") is not None else None,
                "drafts": int(r.get("drafts") or recent_n),
                "sample_size": int(r.get("sample_size") or 0),
            })
        seg = _segment_label({**f, "draft_type": draft_type,
                              "season": season, "num_teams": num_teams,
                              "league_type": "sf" if is_sf else "1qb"})
        if basis == "last_30_days":
            basis_line = (
                f"Based on {recent_n} drafts from the last "
                f"{ADP_RECENT_WINDOW_DAYS} days matching these filters: {seg}")
        else:
            basis_line = (
                f"Season aggregate: based on {seg_drafts} drafts matching "
                f"these filters: {seg}. The last "
                f"{ADP_RECENT_WINDOW_DAYS} days had only {recent_n} drafts "
                f"in this segment (minimum {ADP_RECENT_MIN_DRAFTS} to quote "
                f"a recent window).")
        return jsonify({
            "basis": basis,
            "basis_line": basis_line,
            "recent_drafts": recent_n,
            "segment_drafts": seg_drafts,
            "items": items,
        })
    except Exception:
        logger.exception("[draft-database] adp error")
        return jsonify({"error": "Internal error"}), 500


# ── Page ────────────────────────────────────────────────────────────────────

_PAGE_CSS = """
      .ddb-page-card { max-width: 960px; }
      @media(min-width: 1100px) { .ddb-page-card { max-width: 1100px; } }
      .ddb-page-header { border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 0; }
      .ddb-page-header h2 { margin: 0 0 4px; font-size: 22px; }
      .ddb-page-sub { font-size: 13px; color: var(--text-muted); }
      .ddb-page-body { padding-top: 20px; }
      .ddb-tabs { display: flex; gap: 3px; background: var(--bg-alt); border: 1px solid var(--border); border-radius: 10px; padding: 3px; margin-bottom: 14px; width: fit-content; max-width: 100%; }
      .ddb-tab { font: inherit; font-size: 12px; font-weight: 700; color: var(--text-muted); background: transparent; border: none; border-radius: 7px; padding: 6px 12px; cursor: pointer; white-space: nowrap; }
      .ddb-tab:hover { color: var(--text); }
      .ddb-tab.active { background: var(--accent); color: #fff; }
      .ddb-toolbar { display: block; margin-bottom: 16px; background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 14px; box-shadow: 0 1px 3px rgba(18,45,75,.05); }
      .ddb-search-row { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; }
      .ddb-search-wrap { flex: 1; min-width: 200px; display: flex; flex-direction: column; gap: 6px; }
      .ddb-side-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); line-height: 15px; }
      .ddb-search-outer { border: 1px solid var(--border); border-radius: 8px; background: var(--card); }
      .ddb-search-outer:focus-within { border-color: var(--accent, #3b82f6); }
      .ddb-search { width: 100%; height: 38px; padding: 0 12px; border: none !important; background: transparent; color: var(--text); font-size: 15px; outline: none; box-sizing: border-box; border-radius: 8px; appearance: none; }
      .ddb-filter-row { display: flex; gap: 28px; align-items: flex-end; flex-wrap: wrap; margin-top: 13px; padding-top: 13px; border-top: 1px solid color-mix(in srgb, var(--border) 55%, transparent); }
      .ddb-fgroup { display: flex; flex-direction: column; gap: 6px; }
      .ddb-fgroup-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); }
      .ddb-fgroup-sort { margin-left: auto; }
      .ddb-chip-row { display: flex; gap: 6px; flex-wrap: wrap; }
      .ddb-select { height: 32px; padding: 0 10px; border: 1px solid var(--border); border-radius: 8px; background: var(--card); color: var(--text); font: inherit; font-size: 13px; font-weight: 600; cursor: pointer; }
      .ddb-status { font-size: 13px; color: var(--text-muted); margin-bottom: 14px; min-height: 16px; }
      .ddb-summary-line { font-size: 13px; color: var(--text-muted); margin: 0 0 10px; }
      .ddb-list { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
      @media(max-width: 600px) { .ddb-list { grid-template-columns: 1fr; } }
      .ddb-card { border: 1px solid var(--border); border-radius: 14px; background: var(--card); overflow: hidden; display: flex; flex-direction: column; box-shadow: 0 1px 3px rgba(18,45,75,.05); }
      .ddb-card-head { display: flex; align-items: center; justify-content: space-between; gap: 10px; padding: 10px 14px 0; }
      .ddb-card-title { font-size: 15.5px; font-weight: 800; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .ddb-plat { font-size: 10.5px; font-weight: 800; color: var(--text-muted); border: 1px solid var(--border); border-radius: var(--radius-pill); padding: 2px 8px; background: var(--bg-alt); white-space: nowrap; }
      .ddb-card-meta { font-size: 12px; color: var(--text-muted); padding: 2px 14px 8px; border-bottom: 1px solid color-mix(in srgb, var(--border) 55%, transparent); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .ddb-summary { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0; padding: 6px 0 8px; }
      .ddb-sum-item { padding: 8px 14px 9px; min-width: 0; }
      .ddb-sum-item + .ddb-sum-item { border-left: 1px solid var(--border); }
      .ddb-sum-k { font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin-bottom: 3px; }
      .ddb-sum-v { font-size: 14px; font-weight: 800; line-height: 1.25; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
      .ddb-sum-sub { display: flex; align-items: center; gap: 6px; margin-top: 4px; min-height: 19px; font-size: 11.5px; color: var(--text-muted); font-weight: 600; line-height: 1; }
      .ddb-grade { display: inline-block; font-weight: 800; color: var(--gc, #64748b); background: color-mix(in srgb, var(--gc, #64748b) 10%, transparent); border: 1px solid color-mix(in srgb, var(--gc, #64748b) 35%, transparent); border-radius: var(--radius-pill); padding: 0 7px; font-size: 12.5px; line-height: 19px; white-space: nowrap; }
      .ddb-card-note { margin: 0 14px 10px; font-size: 11.5px; color: var(--text-muted); background: var(--bg-alt); border-radius: 8px; padding: 7px 10px; }
      .ddb-card-foot { margin-top: auto; border-top: 1px solid var(--border); padding: 10px 14px; display: flex; align-items: center; justify-content: flex-end; }
      .ddb-board-btn { display: inline-flex; align-items: center; justify-content: center; gap: 6px; height: 32px; padding: 0 14px; font: inherit; font-size: 13px; font-weight: 700; line-height: 1; white-space: nowrap; color: var(--text); background: var(--card); border: 1px solid var(--border); border-radius: var(--radius-pill); cursor: pointer; }
      .ddb-board-btn:hover { border-color: var(--accent); }
      .ddb-empty { border: 1px dashed var(--border); border-radius: 12px; padding: 22px; text-align: center; color: var(--text-muted); font-size: 13px; grid-column: 1 / -1; }
      .ddb-back { display: inline-flex; align-items: center; gap: 6px; height: 32px; padding: 0 14px; font: inherit; font-size: 13px; font-weight: 700; line-height: 1; color: var(--text-muted); background: transparent; border: 1px solid var(--border); border-radius: var(--radius-pill); cursor: pointer; margin-bottom: 12px; }
      .ddb-back:hover { border-color: var(--accent); color: var(--text); }
      .ddb-board-head { padding: 4px 0 14px; }
      .ddb-board-head h3 { margin: 0 0 4px; font-size: 20px; font-weight: 800; }
      .ddb-board-note { font-size: 12px; color: var(--text-muted); margin-top: 8px; }
      .ddb-adp-basis { font-size: 13px; color: var(--text); background: var(--bg-alt); border: 1px solid var(--border); border-radius: 8px; padding: 9px 12px; margin-bottom: 12px; }
      .ddb-adp-scroll { overflow-x: auto; }
      .ddb-adp-table { width: 100%; border-collapse: collapse; min-width: 600px; }
      .ddb-adp-table th, .ddb-adp-table td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--border); font-size: 13px; }
      .ddb-adp-table th { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); }
      .ddb-adp-big { font-weight: 800; color: var(--text); }
      .ddb-adp-tm { color: var(--text-muted); font-size: 12px; font-weight: 600; }
"""

_PAGE_JS = """
    (function() {
      let view = 'drafts'; // drafts | adp | board
      let currentPage = 1;
      let paginationData = null;
      let abortCtl = null;
      const state = { q: '', season: DDB_SEASON, draft_type: 'all', format: 'all',
                      league_type: 'all', num_teams: '', sort: 'recent' };
      const listEl = document.getElementById('ddbList');
      const statusEl = document.getElementById('ddbStatus');

      fetch('/api/draft-database/count')
        .then(r => r.ok ? r.json() : null)
        .then(d => { if (d && d.count) document.querySelectorAll('[data-draft-count]').forEach(el => { el.textContent = Number(d.count).toLocaleString(); }); })
        .catch(() => {});

      function esc(s) { return String(s == null ? '' : s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
      function gradeColor(g) {
        if (!g) return '#64748b';
        if (g.charAt(0) === 'A') return '#16a34a';
        if (g.charAt(0) === 'B') return '#65a30d';
        if (g.charAt(0) === 'C') return '#d97706';
        return '#dc2626';
      }
      function params(page0) {
        const p = new URLSearchParams({ page: page0, limit: 20, sort: state.sort });
        if (state.q) p.set('q', state.q);
        if (state.season) p.set('season', state.season);
        if (state.draft_type !== 'all') p.set('draft_type', state.draft_type);
        if (state.format !== 'all') p.set('format', state.format);
        if (state.league_type !== 'all') p.set('league_type', state.league_type);
        if (state.num_teams) p.set('num_teams', state.num_teams);
        return p;
      }
      function scoreLabel(s) { return ({ ppr: 'PPR', half: 'Half PPR', std: 'Standard' })[String(s || '').toLowerCase()] || ''; }
      function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : ''; }

      function loadPage(page) {
        if (typeof page === 'string') {
          if (page === 'prev' && currentPage > 1) page = currentPage - 1;
          else if (page === 'next' && paginationData && paginationData.has_next) page = currentPage + 1;
          else return;
        }
        currentPage = page;
        if (abortCtl) abortCtl.abort();
        abortCtl = new AbortController();
        statusEl.textContent = 'Loading drafts...';
        fetch('/api/draft-database?' + params(page - 1), { signal: abortCtl.signal })
          .then(r => r.json())
          .then(data => {
            if (data.error) throw new Error(data.error);
            const drafts = data.drafts || [];
            paginationData = data.pagination;
            document.getElementById('ddbSummary').textContent =
              Number(data.total || 0).toLocaleString() + ' Sleeper drafts';
            if (!drafts.length) {
              listEl.innerHTML = '<div class="ddb-empty">No Sleeper drafts match these filters yet. The corpus grows with every crawl; try a wider segment.</div>';
              document.getElementById('ddbPagination').style.display = 'none';
              statusEl.textContent = '';
              return;
            }
            renderCards(drafts);
            updatePagination();
            statusEl.textContent = '';
          })
          .catch(err => { if (err && err.name !== 'AbortError') statusEl.textContent = 'Error loading drafts'; });
      }

      function superCell(label, pick, fam) {
        if (!pick) return '<div class="ddb-sum-item"><div class="ddb-sum-k">' + label + '</div><div class="ddb-sum-v">-</div><div class="ddb-sum-sub"></div></div>';
        const sub = [];
        if (pick.pick_no) sub.push('Pick ' + pick.pick_no);
        if (fam === 'performance' && pick.verdict) sub.push(esc(pick.verdict));
        else if (fam === 'at_draft' && pick.grade) sub.push('Grade ' + esc(pick.grade));
        return '<div class="ddb-sum-item"><div class="ddb-sum-k">' + label + '</div>' +
          '<div class="ddb-sum-v" title="' + esc(pick.name || '') + '">' + esc(pick.name || '-') + '</div>' +
          '<div class="ddb-sum-sub"><span>' + sub.join(' · ') + '</span></div></div>';
      }

      function renderCards(drafts) {
        listEl.innerHTML = '';
        drafts.forEach(d => {
          const meta = [
            d.season,
            d.draft_started_at ? new Date(d.draft_started_at).toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' }) : '',
            d.num_teams ? d.num_teams + '-team' : '',
            d.is_superflex ? 'SF' : '1QB',
            d.league_format ? cap(d.league_format) : '',
            d.draft_type ? cap(d.draft_type) : '',
            scoreLabel(d.scoring_type),
            d.rounds ? d.rounds + ' rounds' : '',
            d.total_picks ? d.total_picks + ' picks' : '',
          ].filter(Boolean).join(' · ');
          const bd = d.best_draft;
          const bestCell = bd
            ? '<div class="ddb-sum-item"><div class="ddb-sum-k">Best draft</div><div class="ddb-sum-v" title="' + esc(bd.team || '') + '">' + esc(bd.team || '-') + '</div><div class="ddb-sum-sub">' + (bd.grade ? '<span class="ddb-grade" style="--gc:' + gradeColor(bd.grade) + '">' + esc(bd.grade) + '</span>' : '') + '</div></div>'
            : '<div class="ddb-sum-item"><div class="ddb-sum-k">Best draft</div><div class="ddb-sum-v">-</div><div class="ddb-sum-sub"></div></div>';
          const card = document.createElement('article');
          card.className = 'ddb-card';
          card.innerHTML =
            '<div class="ddb-card-head"><div class="ddb-card-title" title="' + esc(d.league_name) + '">' + esc(d.league_name) + '</div><span class="ddb-plat">Sleeper</span></div>' +
            '<div class="ddb-card-meta">' + esc(meta) + '</div>' +
            '<div class="ddb-summary">' + bestCell +
              superCell('Biggest steal', d.biggest_steal, d.superlative_family) +
              superCell('Biggest bust', d.biggest_bust, d.superlative_family) +
            '</div>' +
            (d.performance_note ? '<div class="ddb-card-note">' + esc(d.performance_note) + '</div>' : '') +
            '<div class="ddb-card-foot"><button class="ddb-board-btn" data-board="' + esc(d.draft_id) + '">View board</button></div>';
          listEl.appendChild(card);
        });
        listEl.querySelectorAll('[data-board]').forEach(btn => {
          btn.addEventListener('click', () => openBoard(btn.getAttribute('data-board')));
        });
      }

      function updatePagination() {
        if (!paginationData) return;
        const prev = document.getElementById('ddbPrevBtn');
        const next = document.getElementById('ddbNextBtn');
        prev.disabled = !paginationData.has_prev;
        next.disabled = !paginationData.has_next;
        const start = (paginationData.current_page - 1) * paginationData.per_page + 1;
        const end = Math.min(paginationData.current_page * paginationData.per_page, paginationData.total_players);
        document.getElementById('ddbPaginationText').textContent = 'Showing ' + start + '-' + end + ' of ' + paginationData.total_players + ' drafts';
        const nums = document.getElementById('ddbPageNumbers');
        nums.innerHTML = '';
        const maxPages = 5;
        let sp = Math.max(1, paginationData.current_page - 2);
        let ep = Math.min(paginationData.total_pages, sp + maxPages - 1);
        if (ep - sp < maxPages - 1) sp = Math.max(1, ep - maxPages + 1);
        for (let i = sp; i <= ep; i++) {
          const b = document.createElement('button');
          b.className = 'ti-page-number' + (i === paginationData.current_page ? ' active' : '');
          b.textContent = i;
          b.onclick = () => loadPage(i);
          nums.appendChild(b);
        }
        document.getElementById('ddbPagination').style.display = 'flex';
      }

      // ── Board ──
      const POS_COLORS = { QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b', TE: '#8b5cf6', K: '#c92c68', DEF: '#475569' };
      const VERDICT_COLORS = { Steal: '#16a34a', Hit: '#22c55e', 'As drafted': '#64748b', Reach: '#f59e0b', Bust: '#dc2626', Injured: '#64748b', 'No stats': '#94a3b8' };
      function initials(name) {
        const parts = String(name || '').trim().split(/\\s+/).filter(Boolean);
        if (!parts.length) return '?';
        return (parts[0].charAt(0) + (parts.length > 1 ? parts[parts.length - 1].charAt(0) : '')).toUpperCase();
      }
      function openBoard(draftId) {
        view = 'board';
        document.getElementById('ddbListView').style.display = 'none';
        document.getElementById('ddbAdpView').style.display = 'none';
        const bv = document.getElementById('ddbBoardView');
        bv.style.display = '';
        bv.innerHTML = '<button class="ddb-back" id="ddbBackBtn">Back to drafts</button><div class="ddb-status">Loading board...</div>';
        document.getElementById('ddbBackBtn').addEventListener('click', showList);
        fetch('/api/draft-database/' + encodeURIComponent(draftId) + '/board')
          .then(r => r.json())
          .then(data => { if (data.error) throw new Error(data.error); renderBoard(data); })
          .catch(() => { bv.innerHTML = '<button class="ddb-back" id="ddbBackBtn">Back to drafts</button><div class="ddb-empty">This board could not be loaded.</div>'; document.getElementById('ddbBackBtn').addEventListener('click', showList); });
      }
      function renderBoard(data) {
        const bv = document.getElementById('ddbBoardView');
        const d = data.draft;
        const cols = data.teams.length;
        let html = '<button class="ddb-back" id="ddbBackBtn">Back to drafts</button>' +
          '<div class="ddb-board-head"><h3>' + esc(d.league_name) + '</h3>' +
          '<div class="ddb-card-meta" style="padding:0;border:none;">' + esc([d.season, d.num_teams ? d.num_teams + '-team' : '', d.is_superflex ? 'SF' : '1QB', d.league_format ? cap(d.league_format) : '', d.draft_type ? cap(d.draft_type) : '', scoreLabel(d.scoring_type)].filter(Boolean).join(' · ')) + '</div></div>';
        html += '<div class="dr-board-scroll"><div class="dr-board" style="grid-template-columns:44px repeat(' + cols + ', minmax(168px, 1fr));">';
        html += '<div class="dr-colhead dr-rowhead dr-corner"></div>';
        data.teams.forEach(t => {
          html += '<div class="dr-colhead' + (t.is_you ? ' dr-colhead-you' : '') + '">' + (t.is_you ? '<span class="dr-youpill">You</span> ' : '') + esc(t.name) + '</div>';
        });
        data.rounds.forEach(row => {
          html += '<div class="dr-rowhead">R' + row.round + '</div>';
          row.cells.forEach(c => {
            if (!c) { html += '<div class="dr-cell"></div>'; return; }
            const pos = String(c.position || '').toUpperCase();
            const col = POS_COLORS[pos] || '#94a3b8';
            const photo = c.headshot
              ? '<span class="dr-ph"><img loading="lazy" alt="" src="' + esc(c.headshot) + '" onerror="this.parentElement.style.display=\'none\'"></span>'
              : '<span class="dr-ph-init" style="--pos:' + col + '">' + esc(initials(c.name)) + '</span>';
            let chip = '';
            if (c.grade || c.verdict) {
              chip = '<div class="dr-cell-chip">' +
                (c.grade ? '<span class="dr-cell-grade">Grade ' + esc(c.grade) + '</span>' : '') +
                (c.verdict ? '<span class="dr-cell-verdict" style="color:' + (VERDICT_COLORS[c.verdict] || '#64748b') + '">' + esc(c.verdict) + '</span>' : '') +
                '</div>';
            }
            html += '<div class="dr-cell dr-cell-filled' + (c.is_you ? ' dr-cell-mine' : '') + '" style="--pos:' + col + '">' +
              '<span class="stripe"></span>' + photo +
              '<span class="dr-cell-pick">' + (c.pick_in_round ? row.round + '.' + String(c.pick_in_round).padStart(2, '0') : (c.pick_no || '')) + '</span>' +
              '<div class="dr-cell-body"><div class="dr-cell-name">' + esc(c.name) + '</div>' +
              '<div class="dr-cell-meta"><span class="dr-posbadge">' + esc(pos || '?') + '</span><span>' + esc(c.team || '') + '</span></div>' +
              chip + '</div></div>';
          });
        });
        html += '</div></div>';
        if (d.performance_note) html += '<div class="ddb-board-note">' + esc(d.performance_note) + '</div>';
        html += '<div class="ddb-board-note">Grades are the at-draft letter for each pick; verdicts appear when season performance grades exist.</div>';
        bv.innerHTML = html;
        document.getElementById('ddbBackBtn').addEventListener('click', showList);
      }
      function showList() {
        view = 'drafts';
        document.getElementById('ddbBoardView').style.display = 'none';
        document.getElementById('ddbAdpView').style.display = 'none';
        document.getElementById('ddbListView').style.display = '';
        setTab('drafts');
      }

      // ── ADP ──
      function loadAdp() {
        const el = document.getElementById('ddbAdpBody');
        el.innerHTML = '<div class="ddb-status">Loading ADP...</div>';
        const p = params(0);
        fetch('/api/draft-database/adp?' + p)
          .then(r => r.json())
          .then(data => {
            if (data.error) throw new Error(data.error);
            if (!data.items.length) {
              el.innerHTML = '<div class="ddb-adp-basis">' + esc(data.basis_line) + '</div><div class="ddb-empty">No ADP data for this segment yet.</div>';
              return;
            }
            let html = '<div class="ddb-adp-basis">' + esc(data.basis_line) + '</div>' +
              '<div class="ddb-adp-scroll"><table class="ddb-adp-table"><thead><tr><th>Player</th><th>ADP</th><th>Avg round</th><th>Drafts</th><th>Std dev</th></tr></thead><tbody>';
            data.items.forEach(it => {
              html += '<tr><td><span class="ddb-adp-big">' + esc(it.name) + '</span> <span class="ddb-adp-tm">' + esc([it.position, it.team].filter(Boolean).join(' · ')) + '</span></td>' +
                '<td class="ddb-adp-big">' + (it.adp != null ? it.adp.toFixed(1) : '-') + '</td>' +
                '<td>' + (it.avg_round != null ? it.avg_round.toFixed(1) : '-') + '</td>' +
                '<td>' + it.drafts + '</td>' +
                '<td>' + (it.std_pick != null ? it.std_pick.toFixed(1) : '-') + '</td></tr>';
            });
            html += '</tbody></table></div>';
            el.innerHTML = html;
          })
          .catch(() => { el.innerHTML = '<div class="ddb-empty">ADP could not be loaded for this segment.</div>'; });
      }

      function setTab(t) {
        view = t;
        document.querySelectorAll('.ddb-tab').forEach(b => b.classList.toggle('active', b.dataset.view === t));
        if (t === 'adp') {
          document.getElementById('ddbListView').style.display = 'none';
          document.getElementById('ddbBoardView').style.display = 'none';
          document.getElementById('ddbAdpView').style.display = '';
          loadAdp();
        } else {
          document.getElementById('ddbAdpView').style.display = 'none';
          document.getElementById('ddbBoardView').style.display = 'none';
          document.getElementById('ddbListView').style.display = '';
        }
      }
      document.querySelectorAll('.ddb-tab').forEach(b => b.addEventListener('click', () => setTab(b.dataset.view)));

      function chipWire(cls, key) {
        document.querySelectorAll('.' + cls).forEach(b => b.addEventListener('click', () => {
          document.querySelectorAll('.' + cls).forEach(x => x.classList.toggle('active', x === b));
          state[key] = b.dataset.v;
          if (view === 'adp') loadAdp(); else if (view !== 'board') loadPage(1);
        }));
      }
      chipWire('ddb-dt', 'draft_type');
      chipWire('ddb-fm', 'format');
      chipWire('ddb-lt', 'league_type');
      chipWire('ddb-tm', 'num_teams');
      document.getElementById('ddbSeason').addEventListener('change', function() {
        state.season = this.value;
        if (view === 'adp') loadAdp(); else if (view !== 'board') loadPage(1);
      });
      document.getElementById('ddbSort').addEventListener('change', function() {
        state.sort = this.value;
        loadPage(1);
      });
      let deb = null;
      document.getElementById('ddbSearch').addEventListener('input', function() {
        clearTimeout(deb);
        deb = setTimeout(() => { state.q = this.value.trim(); if (view !== 'board') loadPage(1); }, 300);
      });
      window.ddbLoadPage = loadPage;
      loadPage(1);
    })();
"""


@draft_db_bp.route("/<platform>/<int:season>/<league_id>/draft-database")
def page_draft_database(platform: str, season: int, league_id: str):
    from app import render_page
    _season_opts = "".join(
        f'<option value="{y}"{" selected" if y == season else ""}>{y}</option>'
        for y in range(season + 1, season - 5, -1)
    )
    body_html = f"""
    <div class="card central ddb-page-card">
      <div class="card-header ddb-page-header">
        <h2>Draft Database</h2>
        <div class="ddb-page-sub">
          Explore real Sleeper drafts: full boards, at-draft grades, performance verdicts and ADP.
          <br><span data-draft-count>…</span> completed Sleeper drafts indexed
        </div>
      </div>
      <div class="card-body ddb-page-body">
        <div class="ddb-tabs">
          <button class="ddb-tab active" data-view="drafts">Drafts</button>
          <button class="ddb-tab" data-view="adp">ADP</button>
        </div>
        <div class="ddb-toolbar">
          <div class="ddb-search-row">
            <div class="ddb-search-wrap">
              <div class="ddb-side-label">Search</div>
              <div class="ddb-search-outer">
                <input id="ddbSearch" type="text" placeholder="Search player or league…" class="ddb-search" autocomplete="off">
              </div>
            </div>
          </div>
          <div class="ddb-filter-row">
            <div class="ddb-fgroup">
              <div class="ddb-fgroup-label">Season</div>
              <select id="ddbSeason" class="ddb-select">{_season_opts}</select>
            </div>
            <div class="ddb-fgroup">
              <div class="ddb-fgroup-label">Draft type</div>
              <div class="ddb-chip-row">
                <button class="otc-day-filter ddb-dt active" data-v="all">All</button>
                <button class="otc-day-filter ddb-dt" data-v="startup">Startup</button>
                <button class="otc-day-filter ddb-dt" data-v="rookie">Rookie</button>
                <button class="otc-day-filter ddb-dt" data-v="redraft">Redraft</button>
              </div>
            </div>
            <div class="ddb-fgroup">
              <div class="ddb-fgroup-label">League type</div>
              <div class="ddb-chip-row">
                <button class="otc-day-filter ddb-fm active" data-v="all">All</button>
                <button class="otc-day-filter ddb-fm" data-v="dynasty">Dynasty</button>
                <button class="otc-day-filter ddb-fm" data-v="redraft">Redraft</button>
                <button class="otc-day-filter ddb-fm" data-v="keeper">Keeper</button>
              </div>
            </div>
            <div class="ddb-fgroup">
              <div class="ddb-fgroup-label">Format</div>
              <div class="ddb-chip-row">
                <button class="otc-day-filter ddb-lt active" data-v="all">All</button>
                <button class="otc-day-filter ddb-lt" data-v="1qb">1QB</button>
                <button class="otc-day-filter ddb-lt" data-v="sf">SF</button>
              </div>
            </div>
            <div class="ddb-fgroup">
              <div class="ddb-fgroup-label">Teams</div>
              <div class="ddb-chip-row">
                <button class="otc-day-filter ddb-tm active" data-v="">All</button>
                <button class="otc-day-filter ddb-tm" data-v="8">8</button>
                <button class="otc-day-filter ddb-tm" data-v="10">10</button>
                <button class="otc-day-filter ddb-tm" data-v="12">12</button>
                <button class="otc-day-filter ddb-tm" data-v="14">14</button>
                <button class="otc-day-filter ddb-tm" data-v="16">16</button>
              </div>
            </div>
            <div class="ddb-fgroup ddb-fgroup-sort">
              <label class="ddb-fgroup-label" for="ddbSort">Sort</label>
              <select id="ddbSort" class="ddb-select">
                <option value="recent">Most recent</option>
                <option value="most_picks">Most picks</option>
                <option value="best_grade">Best grade</option>
              </select>
            </div>
          </div>
        </div>

        <div id="ddbListView">
          <div id="ddbStatus" class="ddb-status"></div>
          <div id="ddbSummary" class="ddb-summary-line"></div>
          <div id="ddbList" class="ddb-list"></div>
          <div id="ddbPagination" class="ti-pagination" style="display:none;">
            <div class="ti-pagination-info"><span id="ddbPaginationText"></span></div>
            <div class="ti-pagination-controls">
              <button id="ddbPrevBtn" class="ti-pagination-btn" onclick="ddbLoadPage('prev')"><i class="fa-solid fa-chevron-left"></i> Previous</button>
              <div id="ddbPageNumbers" class="ti-page-numbers"></div>
              <button id="ddbNextBtn" class="ti-pagination-btn" onclick="ddbLoadPage('next')">Next <i class="fa-solid fa-chevron-right"></i></button>
            </div>
          </div>
        </div>
        <div id="ddbBoardView" style="display:none;"></div>
        <div id="ddbAdpView" style="display:none;"><div id="ddbAdpBody"></div></div>
      </div>
    </div>
    <style>{_PAGE_CSS}</style>
    <script>const DDB_SEASON = {season};</script>
    <script>{_PAGE_JS}</script>
    """
    return render_page(
        "Fantasy Football Draft Database - Real Sleeper Draft Boards and ADP | BR Fantasy",
        league_id, "draft-database", body_html, platform, season,
        description=(
            "Browse completed Sleeper drafts with full boards, at-draft grades, "
            "performance verdicts and segment ADP for dynasty, redraft and keeper leagues."
        ),
    )


@draft_db_bp.route("/draft-database")
def page_draft_database_guest():
    from app import get_nfl_state
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    return page_draft_database(platform="sleeper", season=current_season,
                               league_id=None)
