"""
Rookie prospect API blueprint.

Endpoints:
    GET  /api/prospects/rankings?year=2026&pos=WR&league_type=1qb
    GET  /api/prospects/player/<player_id>
    GET  /api/prospects/active-class
    POST /api/prospects/prospects        (add/update one or more prospects)
    POST /api/prospects/refresh          (triggers pipeline re-run)
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from flask import Blueprint, jsonify, request

from dashboard_services.admin_auth import admin_required

log = logging.getLogger(__name__)

rookie_bp = Blueprint("prospects", __name__, url_prefix="/api/prospects")

# In-memory cache so we don't re-run the pipeline on every page load.
# Invalidated on refresh or on first hit per process.
_cache: dict[Any, list[dict[str, Any]]] = {}

# FantasyCalc ADP fallback - keyed by ("sf"|"1qb", YYYY-MM-DD), lives for the day.
_FC_ADP_CACHE: dict[tuple, list] = {}


def _nfl_draft_complete(draft_year: int) -> bool:
    from data_building.rookie_pipeline.pipeline import is_draft_complete
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            return is_draft_complete(draft_year, conn)
    except Exception:
        return is_draft_complete(draft_year)


def _get_rankings(draft_year: int) -> list[dict[str, Any]]:
    draft_done = _nfl_draft_complete(draft_year)
    cache_key = (draft_year, draft_done)
    if cache_key not in _cache:
        from data_building.rookie_pipeline.pipeline import get_rookie_rankings_from_db
        rows = get_rookie_rankings_from_db(draft_year, filter_undrafted=draft_done)
        _cache[cache_key] = rows
        # Auto-link unlinked prospects in the background so future requests
        # can use sleeper_id directly without falling back to name matching.
        _auto_link_unlinked(rows)
    return _cache[cache_key]


def _auto_link_unlinked(rows: list[dict[str, Any]]) -> None:
    """
    For every prospect row missing sleeper_id, attempt a name-based match
    against players_index.json.  Persists successful matches to the DB and
    updates the in-memory row in place so the current request also benefits.
    Runs silently - any failure is a no-op.
    """
    unlinked = [r for r in rows if not r.get("sleeper_id")]
    if not unlinked:
        return

    try:
        import json as _json, re as _re
        from utils.paths import CACHE_DIR as _CD
        _pi_path = _CD / "players_index.json"
        if not _pi_path.exists():
            return
        _pi = _json.loads(_pi_path.read_text())

        def _norm(n: str) -> str:
            n = n.lower()
            n = _re.sub(r"['\.\-]", "", n)
            n = _re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
            return _re.sub(r"\s+", " ", n).strip()

        # Build norm_name -> sleeper_id map from players_index
        _name_to_sid: dict[str, str] = {}
        for _sid, _pdata in _pi.items():
            _pn = _pdata.get("name", "")
            if _pn:
                _name_to_sid[_norm(_pn)] = _sid

        links: dict[str, str] = {}  # prospect player_id -> sleeper_id
        for row in unlinked:
            prospect_name = row.get("name", "")
            if not prospect_name:
                continue
            sid = _name_to_sid.get(_norm(prospect_name))
            if sid:
                row["sleeper_id"] = sid
                links[row["player_id"]] = sid

        if not links:
            return

        # Persist links to DB (best-effort)
        try:
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                for pid, sid in links.items():
                    conn.execute(
                        "UPDATE rookie_prospects SET sleeper_id = %s, updated_at = NOW() "
                        "WHERE player_id = %s AND sleeper_id IS NULL",
                        (sid, pid),
                    )
                conn.commit()
            log.info("[rookie_api] Auto-linked %d unlinked prospects", len(links))
        except Exception as db_exc:
            log.debug("[rookie_api] Auto-link DB persist skipped: %s", db_exc)

    except Exception as exc:
        log.debug("[rookie_api] _auto_link_unlinked error: %s", exc)


def _safe_float(v, default=None):
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _row_to_dict(row: dict) -> dict:
    """Serialise a row dict to JSON-safe types."""
    out = {}
    for k, v in row.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


# ── Prospect redesign: detail helpers ─────────────────────────────────────────
# The Advanced metrics section shows ONLY metrics the pipeline actually stores.
# Raw values come straight from the DB tables; grades are simple documented
# 0-100 display scalings for the bars/radar (they are not model scores).

_BREAKOUT_DOM_THRESH = {"WR": 0.25, "RB": 0.275, "TE": 0.12}
_COMPOSITE_COL_CANDIDATES = (
    "composite_stars", "recruit_stars", "stars_247", "recruiting_stars",
)
_composite_col_cache: dict[str, Any] = {}


def _clip100(v):
    # Display grades: one decimal, capped below 100 (no one grades a 100).
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return round(max(0.0, min(99.9, v)), 1)


def _grade_dominator(d):
    # Average (0.20) -> C (75); 0.35+ elite -> 99.9.
    if d is None:
        return None
    return _clip100(d / 0.267 * 100.0)


_BREAKOUT_AGE_DISPLAY = (
    # (age, grade) anchors for DISPLAY grading. An average breakout age
    # (~20.5-21) reads as C, not F. The model's linear component scale in
    # prospect_model is untouched; this only affects the modal's letter.
    (18.5, 100.0), (19.0, 97.0), (19.5, 90.0), (20.0, 85.0),
    (20.5, 78.0), (21.0, 70.0), (21.5, 60.0), (22.0, 50.0), (22.5, 40.0),
)


def _grade_breakout_age(age, pos):
    if age is None:
        return None
    try:
        a = float(age)
    except (TypeError, ValueError):
        return None
    if (pos or "").upper() == "QB":
        a -= 0.7  # QBs break out later; grade on a shifted scale
    pts = _BREAKOUT_AGE_DISPLAY
    if a <= pts[0][0]:
        return 100.0
    for (a0, g0), (a1, g1) in zip(pts, pts[1:]):
        if a <= a1:
            return _clip100(g0 + (g1 - g0) * (a - a0) / (a1 - a0))
    return _clip100(pts[-1][1])


def _grade_ypr(v):
    return _clip100(v / 18.0 * 100.0) if v is not None else None


def _grade_ypc(v):
    return _clip100(v / 7.0 * 100.0) if v is not None else None


def _grade_market_share(v):
    # Average (0.20) -> C (75); 0.35+ elite -> 99.9.
    return _clip100(v / 0.267 * 100.0) if v is not None else None


def _grade_speed_score(v):
    return _clip100(v / 120.0 * 100.0) if v is not None else None


def _grade_cmp(v):
    return _clip100(v / 70.0 * 100.0) if v is not None else None


def _grade_td_int(v):
    # Average (2.0) -> C (75); 4.0+ elite -> 99.9.
    return _clip100(v / 2.67 * 100.0) if v is not None else None


def _grade_scrim(v):
    # Average (80 yds/gm) -> C (75); 140+ elite -> 99.9.
    return _clip100(v / 106.7 * 100.0) if v is not None else None


def _grade_aya(v):
    # Adjusted yards per attempt; 10.5+ is elite college efficiency.
    return _clip100(v / 10.5 * 100.0) if v is not None else None


def _grade_td_share(v):
    # Average (0.15) -> C (75); 0.30+ elite -> 99.9.
    return _clip100(v / 0.20 * 100.0) if v is not None else None


def _grade_ypa(v):
    return _clip100(v / 9.5 * 100.0) if v is not None else None


def _grade_tpg(v):
    # Average (6.5) -> C (75); 10+ elite -> 99.9.
    return _clip100(v / 8.67 * 100.0) if v is not None else None


def _grade_catch_rate(v):
    # Catch rate; 75%+ is elite hands.
    return _clip100(v / 0.75 * 100.0) if v is not None else None

def _rescale_mid_to_c(v):
    """Rescale a 0-100 model score so average (50) -> C (75).

    Used for Efficiency/Production/WEPA which are raw model outputs
    where 50 is mid-pack, not failing.
    """
    if v is None:
        return None
    try:
        v = float(v)  # psycopg NUMERIC arrives as Decimal; Decimal - float raises
    except (TypeError, ValueError):
        return None
    return _clip100(50.0 + (v - 50.0) * 0.5)


def _compute_breakout(seasons, age, position):
    """(breakout_age, breakout_season) for the first breakout season.

    Mirrors the model's breakout-age logic (prospect_model). QBs use the
    pass-yard share proxy (>= 60% of team yards) since receiving dominator is
    meaningless for them. (None, None) when the player never broke out.
    """
    if not seasons or age is None:
        return None, None
    pos = (position or "").upper()
    try:
        cur_age = float(age)
    except (TypeError, ValueError):
        return None, None
    ordered = sorted(seasons, key=lambda s: s.get("season") or 0)
    if not ordered:
        return None, None
    try:
        current_year = int(ordered[-1].get("season") or 0)
    except (TypeError, ValueError):
        return None, None
    if current_year <= 0:
        return None, None
    for s in ordered:
        try:
            s_year = int(s.get("season") or 0)
        except (TypeError, ValueError):
            continue
        if s_year <= 0:
            continue
        broke_out = False
        if pos == "QB":
            try:
                team_yds = float(s.get("team_total_yards") or 0)
                broke_out = team_yds > 0 and float(s.get("pass_yards") or 0) / team_yds >= 0.60
            except (TypeError, ValueError):
                broke_out = False
        else:
            thresh = _BREAKOUT_DOM_THRESH.get(pos)
            try:
                broke_out = thresh is not None and float(s.get("dominator_rating") or 0) >= thresh
            except (TypeError, ValueError):
                broke_out = False
        if broke_out:
            return round(cur_age - (current_year - s_year), 1), s_year
    return None, None


def _compute_breakout_age(seasons, age, position):
    """Age in the first season the player hit the breakout threshold."""
    return _compute_breakout(seasons, age, position)[0]


def _composite_col(conn) -> Any:
    """Name of the 247-composite stars column if the parallel track added one."""
    if "col" in _composite_col_cache:
        return _composite_col_cache["col"]
    col = None
    try:
        row = conn.execute(
            """SELECT column_name FROM information_schema.columns
               WHERE table_name = 'rookie_prospects'
                 AND column_name = ANY(%s) LIMIT 1""",
            (list(_COMPOSITE_COL_CANDIDATES),),
        ).fetchone()
        if row:
            col = row["column_name"]
    except Exception:
        col = None
    _composite_col_cache["col"] = col
    return col


def _fmt_pct1(v):
    try:
        return "%d%%" % round(float(v) * 100)
    except (TypeError, ValueError):
        return None


def _fmt1(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return ("%g" % round(f, 1))


def _season_shares(s):
    """(dominator, yard_share, td_share) for one stored season row.

    Dominator prefers the stored rating; when absent it is derived with
    the pipeline's own definition: 0.65 * yard share + 0.35 * TD share.
    """
    ys = _safe_float(s.get("market_share_yards"))
    ts = _safe_float(s.get("market_share_tds"))
    dom = _safe_float(s.get("dominator_rating"))
    if dom is None and ys is not None and ts is not None:
        dom = 0.65 * ys + 0.35 * ts
    return dom, ys, ts


def _advanced_season_rows(seasons):
    """Per-season advanced rows for the detail payload (v3.2).

    Shares are fractions (0-1); per-game rates use the season's own games
    played so partial seasons stay honest.
    """
    out = []
    for s in sorted(seasons or [], key=lambda x: _safe_float(x.get("season"), 0) or 0):
        dom, ys, ts = _season_shares(s)
        gp = _safe_float(s.get("games_played"))

        def _pg(v, gp=gp):
            return round(v / gp, 2) if (v is not None and gp) else None

        out.append({
            "season": s.get("season"),
            "dominator": round(dom, 4) if dom is not None else None,
            "yard_share": round(ys, 4) if ys is not None else None,
            "td_share": round(ts, 4) if ts is not None else None,
            "yds_per_reception": _safe_float(s.get("yds_per_reception")),
            "receptions_per_game": _pg(_safe_float(s.get("receptions"))),
            "rec_yards_per_game": _pg(_safe_float(s.get("receiving_yards"))),
            "rec_tds_per_game": _pg(_safe_float(s.get("receiving_tds"))),
        })
    return out


def _advanced_marks(seasons):
    """Best/career marks for the detail payload (v3.2).

    Career shares are the mean of the season shares; career dominator is
    recomposed from them (0.65 yard + 0.35 TD). Career Yds/Rec is total
    receiving yards over total receptions. Latest rates come from the
    most recent season. Shares are fractions; the TD-over-yard gap is in
    percentage points.
    """
    marks = {
        "best_dominator": None,
        "career_dominator": None,
        "career_yds_per_rec": None,
        "latest_rec_yards_per_game": None,
        "latest_rec_tds_per_game": None,
        "td_share_over_yard_share_pts": None,
    }
    rows = [s for s in (seasons or [])]
    if not rows:
        return marks
    doms, yss, tss = [], [], []
    tot_rec, tot_yds = 0.0, 0.0
    for s in rows:
        dom, ys, ts = _season_shares(s)
        if dom is not None:
            doms.append(dom)
        if ys is not None:
            yss.append(ys)
        if ts is not None:
            tss.append(ts)
        rec = _safe_float(s.get("receptions"))
        yds = _safe_float(s.get("receiving_yards"))
        if rec is not None and yds is not None:
            tot_rec += rec
            tot_yds += yds
    if doms:
        marks["best_dominator"] = round(max(doms), 4)
    if yss and tss:
        marks["career_dominator"] = round(
            0.65 * (sum(yss) / len(yss)) + 0.35 * (sum(tss) / len(tss)), 4)
    if tot_rec > 0:
        marks["career_yds_per_rec"] = round(tot_yds / tot_rec, 2)
    latest = max(rows, key=lambda x: _safe_float(x.get("season"), 0) or 0)
    gp = _safe_float(latest.get("games_played"))
    if gp:
        ry = _safe_float(latest.get("receiving_yards"))
        rt = _safe_float(latest.get("receiving_tds"))
        if ry is not None:
            marks["latest_rec_yards_per_game"] = round(ry / gp, 2)
        if rt is not None:
            marks["latest_rec_tds_per_game"] = round(rt / gp, 2)
    _, lys, lts = _season_shares(latest)
    if lys is not None and lts is not None:
        marks["td_share_over_yard_share_pts"] = round((lts - lys) * 100.0, 1)
    return marks


def _build_advanced_metrics(position, seasons, athleticism, row, wepa_score=None):
    """Position-correct advanced metric sets: [{label, raw, grade}].

    Efficiency's grade is the component score itself (no mid-to-C rescale),
    so the advanced table and the Grade Breakdown bar show one number for
    the component. Production keeps the rescale.
    """
    pos = (position or "").upper()
    best = {}
    for s in seasons or []:
        for k in ("dominator_rating", "market_share_yards", "market_share_tds",
                  "yds_per_reception",
                  "yds_per_carry", "completion_pct", "td_int_ratio", "yds_per_attempt",
                  "targets", "receptions", "rush_tds"):
            try:
                v = float(s.get(k)) if s.get(k) is not None else None
            except (TypeError, ValueError):
                v = None
            if v is not None and (k not in best or v > best[k]):
                best[k] = v
        # Derived per-season bests (all from real stored fields).
        try:
            gp = float(s.get("games_played") or 0)
            if gp > 0:
                scrim = (float(s.get("rush_yards") or 0) +
                         float(s.get("receiving_yards") or 0)) / gp
                if "scrimmage_per_game" not in best or scrim > best["scrimmage_per_game"]:
                    best["scrimmage_per_game"] = scrim
                # QB rushing per game (v3.2 advanced rows)
                rpg = float(s.get("rush_yards") or 0) / gp
                if "rush_yards_per_game" not in best or rpg > best["rush_yards_per_game"]:
                    best["rush_yards_per_game"] = rpg
                # Targets per game and catch rate (WR/TE opportunity metrics).
                # A stored targets value below receptions is a pipeline
                # failure, not data: skip the season so Tgt/Gm and Catch%
                # stay hidden instead of printing a garbage ratio.
                tgt = float(s.get("targets") or 0)
                rec = float(s.get("receptions") or 0)
                if tgt > 0 and tgt >= rec:
                    tpg = tgt / gp
                    if "targets_per_game" not in best or tpg > best["targets_per_game"]:
                        best["targets_per_game"] = tpg
                    catch_rate = rec / tgt if tgt > 0 else None
                    if catch_rate is not None and ("catch_rate" not in best or catch_rate > best["catch_rate"]):
                        best["catch_rate"] = catch_rate
            att = float(s.get("pass_attempts") or 0)
            if att > 0:
                aya = (float(s.get("pass_yards") or 0)
                       + 20.0 * float(s.get("pass_tds") or 0)
                       - 45.0 * float(s.get("interceptions") or 0)) / att
                if "aya" not in best or aya > best["aya"]:
                    best["aya"] = aya
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    try:
        speed = float(athleticism.get("speed_score")) if athleticism.get("speed_score") is not None else None
    except (TypeError, ValueError):
        speed = None
    if speed is None:
        # Derive from forty + weight when the pipeline stored measurables
        # but no official speed score (common pre-combine).
        try:
            forty = float(athleticism.get("forty_yard")) if athleticism.get("forty_yard") is not None else None
            wt = float(row.get("weight_lbs")) if row.get("weight_lbs") is not None else None
            if forty and wt:
                speed = round(wt * 200.0 / (forty ** 4), 1)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    eff = _safe_float(row.get("efficiency_score"))
    prod = _safe_float(row.get("production_score"))
    breakout_age = _compute_breakout_age(
        seasons, row.get("age"), pos)
    # No Recruiting row here (Kaedon's 2026-10-11 ruling): recruiting is a
    # model component shown in the Grade Breakdown, not an advanced metric.
    # WEPA opponent-adjusted efficiency (Patreon-tier CFBD; None when unavailable).
    try:
        wepa = float(wepa_score) if wepa_score is not None else None
    except (TypeError, ValueError):
        wepa = None
    wepa_row = (("WEPA", _fmt1(wepa), _rescale_mid_to_c(wepa))
                if wepa is not None else None)
    marks = _advanced_marks(seasons)
    metrics = []
    if pos == "QB":
        metrics = [
            ("Cmp%", _fmt_pct1(best.get("completion_pct") / 100) if best.get("completion_pct") is not None else None,
             _grade_cmp(best.get("completion_pct"))),
            ("TD:INT", _fmt1(best.get("td_int_ratio")), _grade_td_int(best.get("td_int_ratio"))),
            ("Yds/Att", _fmt1(best.get("yds_per_attempt")), _grade_ypa(best.get("yds_per_attempt"))),
            ("AY/A", _fmt1(best.get("aya")), _grade_aya(best.get("aya"))),
            # v3.2: QB rushing rows. No display-grade helper exists for
            # rush yds/g or rush TDs, so those two are raw-only.
            ("Rush Yds/Gm", _fmt1(best.get("rush_yards_per_game")), None),
            ("Yds/Carry", _fmt1(best.get("yds_per_carry")), _grade_ypc(best.get("yds_per_carry"))),
            ("Rush TDs", str(int(best["rush_tds"])) if best.get("rush_tds") is not None else None, None),
            ("Breakout Age", _fmt1(breakout_age), _grade_breakout_age(breakout_age, pos)),
            ("Efficiency", _fmt1(eff), _clip100(eff)),
            ("Production", _fmt1(prod), _rescale_mid_to_c(prod)),
        ]
        if wepa_row:
            metrics.append(wepa_row)
    elif pos == "RB":
        metrics = [
            ("Dominator", _fmt_pct1(best.get("dominator_rating")), _grade_dominator(best.get("dominator_rating"))),
            ("Breakout Age", _fmt1(breakout_age), _grade_breakout_age(breakout_age, pos)),
            ("Yds/Carry", _fmt1(best.get("yds_per_carry")), _grade_ypc(best.get("yds_per_carry"))),
            ("Scrim Yds/Gm", _fmt1(best.get("scrimmage_per_game")), _grade_scrim(best.get("scrimmage_per_game"))),
            ("Mkt Share", _fmt_pct1(best.get("market_share_yards")), _grade_market_share(best.get("market_share_yards"))),
            ("TD Share", _fmt_pct1(best.get("market_share_tds")), _grade_td_share(best.get("market_share_tds"))),
            ("Speed Score", _fmt1(speed), _grade_speed_score(speed)),
        ]
        if wepa_row:
            metrics.append(wepa_row)
        metrics.extend([
            ("Efficiency", _fmt1(eff), _clip100(eff)),
        ])
    else:  # WR / TE
        metrics = [
            ("Dominator", _fmt_pct1(best.get("dominator_rating")), _grade_dominator(best.get("dominator_rating"))),
            ("Breakout Age", _fmt1(breakout_age), _grade_breakout_age(breakout_age, pos)),
            ("Yds/Rec", _fmt1(best.get("yds_per_reception")), _grade_ypr(best.get("yds_per_reception"))),
            ("Mkt Share", _fmt_pct1(best.get("market_share_yards")), _grade_market_share(best.get("market_share_yards"))),
            ("TD Share", _fmt_pct1(best.get("market_share_tds")), _grade_td_share(best.get("market_share_tds"))),
            ("Tgt/Gm", _fmt1(best.get("targets_per_game")), _grade_tpg(best.get("targets_per_game"))),
            ("Catch%", _fmt_pct1(best.get("catch_rate")), _grade_catch_rate(best.get("catch_rate"))),
            ("Speed Score", _fmt1(speed), _grade_speed_score(speed)),
        ]
        # v3.2 fix: the WEPA row was built above but never appended in
        # this branch, so WR/TE prospects could never show it.
        if wepa_row:
            metrics.append(wepa_row)
        metrics.extend([
            ("Efficiency", _fmt1(eff), _clip100(eff)),
            # Best/career marks, graded with the same-stat display formulas
            # already used above: dominator, Yds/Rec, and the yards-per-game
            # grade (the Scrim Yds/Gm formula). No new anchors invented.
            ("Best Dominator", _fmt_pct1(marks.get("best_dominator")),
             _grade_dominator(marks.get("best_dominator"))),
            ("Career Yds/Rec", _fmt1(marks.get("career_yds_per_rec")),
             _grade_ypr(marks.get("career_yds_per_rec"))),
            ("Rec Yds/Gm", _fmt1(marks.get("latest_rec_yards_per_game")),
             _grade_scrim(marks.get("latest_rec_yards_per_game"))),
        ])
    # Numeric value behind each row, for the population percentiles the
    # modal's Advanced tab renders. The Efficiency/Production rows are
    # component scores, not stored-stat metrics: flagged "component" so
    # percentile-only surfaces exclude them (the percentile design
    # supersedes the old advanced-grade cell for Efficiency).
    _value_of = {
        "Cmp%": best.get("completion_pct"),
        "TD:INT": best.get("td_int_ratio"),
        "Yds/Att": best.get("yds_per_attempt"),
        "AY/A": best.get("aya"),
        "Rush Yds/Gm": best.get("rush_yards_per_game"),
        "Yds/Carry": best.get("yds_per_carry"),
        "Rush TDs": best.get("rush_tds"),
        "Breakout Age": breakout_age,
        "Dominator": best.get("dominator_rating"),
        "Scrim Yds/Gm": best.get("scrimmage_per_game"),
        "Mkt Share": best.get("market_share_yards"),
        "TD Share": best.get("market_share_tds"),
        "Speed Score": speed,
        "WEPA": wepa,
        "Yds/Rec": best.get("yds_per_reception"),
        "Tgt/Gm": best.get("targets_per_game"),
        "Catch%": best.get("catch_rate"),
        "Best Dominator": marks.get("best_dominator"),
        "Career Yds/Rec": marks.get("career_yds_per_rec"),
        "Rec Yds/Gm": marks.get("latest_rec_yards_per_game"),
    }
    out = []
    for label, raw, grade in metrics:
        out.append({"label": label, "raw": raw,
                    "grade": grade if grade is not None else None,
                    "value": _value_of.get(label),
                    "component": label in ("Efficiency", "Production")})
    return out


# ── Percentile distributions for the modal's Advanced / Production /
# Testing surfaces. All three are computed against the same graded
# reference population as the component percentiles (current model
# versions, grouped by position) and reuse the <30 hide rule. Cached per
# position for the life of the process, like _population_cache.

_ADV_LOWER_BETTER = {"Breakout Age"}
_advanced_dist_cache: dict[str, dict] = {}
_season_rate_cache: dict[str, list] = {}
_testing_dist_cache: dict[str, dict] = {}


def _season_cols_sql():
    return ", ".join(_SEASON_COLS)


def _percentile_of(value, values, lower_better=False):
    """component_percentile with an inverted direction for lower-is-better
    metrics (breakout age, timed tests): the percentile is the share of
    the pool the player beats."""
    if value is None:
        return None
    if lower_better:
        return component_percentile(
            -float(value), sorted(-float(v) for v in values))
    return component_percentile(value, values)


def _population_player_rows(position):
    """Rankings rows for the graded reference population at a position,
    with the bio fields _build_advanced_metrics needs (age, weight)."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT rr.player_id, rp.age, rp.weight_lbs,
                          rr.production_score, rr.efficiency_score
                   FROM rookie_rankings rr
                   JOIN rookie_prospects rp ON rp.player_id = rr.player_id
                   WHERE rp.position = %s
                     AND rr.model_version = ANY(%s)""",
                (position, list(_PCT_MODEL_VERSIONS)),
            ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _advanced_distributions(position):
    """{metric label: sorted [values]} across the reference population.

    Built by running _build_advanced_metrics itself over every population
    player, so the player's row and the pool can never drift apart.
    Component-score rows are excluded by construction."""
    pos = (position or "").upper()
    if pos in _advanced_dist_cache:
        return _advanced_dist_cache[pos]
    dist: dict[str, list] = {}
    rows = _population_player_rows(pos)
    if rows:
        ids = [r["player_id"] for r in rows if r.get("player_id")]
        season_rows, ath_rows, wepa_rows = [], [], []
        try:
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                season_rows = conn.execute(
                    f"SELECT player_id, {_season_cols_sql()} "
                    "FROM rookie_prospect_source_data "
                    "WHERE player_id = ANY(%s)",
                    (ids,),
                ).fetchall()
                ath_rows = conn.execute(
                    "SELECT player_id, forty_yard, ras_score, speed_score "
                    "FROM rookie_prospect_athleticism "
                    "WHERE player_id = ANY(%s)",
                    (ids,),
                ).fetchall()
                wepa_rows = conn.execute(
                    "SELECT player_id, MAX(adj_efficiency_score) AS wepa "
                    "FROM rookie_prospect_wepa WHERE player_id = ANY(%s) "
                    "GROUP BY player_id",
                    (ids,),
                ).fetchall()
        except Exception:
            season_rows, ath_rows, wepa_rows = [], [], []
        seasons_by: dict[str, list] = {}
        for s in season_rows:
            seasons_by.setdefault(s["player_id"], []).append(dict(s))
        ath_by = {a["player_id"]: dict(a) for a in ath_rows}
        wepa_by = {w["player_id"]: _safe_float(w.get("wepa"))
                   for w in wepa_rows}
        for r in rows:
            pid = r.get("player_id")
            for a in _build_advanced_metrics(
                    pos, seasons_by.get(pid, []), ath_by.get(pid, {}),
                    r, wepa_by.get(pid)):
                if a.get("component") or a.get("value") is None:
                    continue
                dist.setdefault(a["label"], []).append(float(a["value"]))
        for vals in dist.values():
            vals.sort()
    _advanced_dist_cache[pos] = dist
    return dist


def _season_primary_rate(position, s):
    """The position's headline per-game rate for one stored season."""
    gp = _safe_float(s.get("games_played"))
    if not gp:
        return None
    pos = (position or "").upper()
    if pos == "QB":
        v = _safe_float(s.get("pass_yards"))
    elif pos == "RB":
        ry = _safe_float(s.get("rush_yards"))
        cy = _safe_float(s.get("receiving_yards"))
        v = (ry or 0.0) + (cy or 0.0) if (ry is not None or cy is not None) else None
    else:
        v = _safe_float(s.get("receiving_yards"))
    return round(v / gp, 1) if v is not None else None


_RATE_LABEL = {"QB": ("Pass Yds/G", "pass yds/g"),
               "RB": ("Scrim Yds/G", "scrim yds/g")}
_RATE_LABEL_DEFAULT = ("Rec Yds/G", "rec yds/g")
# A season needs a real sample before its rate can anchor a percentile.
_SEASON_MIN_GP = 4


def _season_rate_distributions(position):
    """Sorted primary per-game rates of completed population seasons."""
    pos = (position or "").upper()
    if pos in _season_rate_cache:
        return _season_rate_cache[pos]
    values: list[float] = []
    rows = _population_player_rows(pos)
    if rows:
        ids = [r["player_id"] for r in rows if r.get("player_id")]
        try:
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                season_rows = conn.execute(
                    f"SELECT player_id, {_season_cols_sql()} "
                    "FROM rookie_prospect_source_data "
                    "WHERE player_id = ANY(%s)",
                    (ids,),
                ).fetchall()
        except Exception:
            season_rows = []
        for s in season_rows:
            gp = _safe_float(s.get("games_played"))
            if not gp or gp < _SEASON_MIN_GP:
                continue
            if is_in_season(s.get("season"), gp):
                continue
            rate = _season_primary_rate(pos, s)
            if rate is not None:
                values.append(rate)
        values.sort()
    _season_rate_cache[pos] = values
    return values


def _season_lines(position, seasons):
    """Season-line rows for the modal: per-game rate, in-progress state,
    and a percentile for completed seasons only (an in-progress season
    never prints one)."""
    pos = (position or "").upper()
    label, unit = _RATE_LABEL.get(pos, _RATE_LABEL_DEFAULT)
    dist = _season_rate_distributions(pos)
    lines = []
    for s in sorted(seasons or [],
                    key=lambda x: _safe_float(x.get("season"), 0) or 0):
        gp = _safe_float(s.get("games_played"))
        rate = _season_primary_rate(pos, s)
        in_prog = is_in_season(s.get("season"), gp)
        pct = None
        if not in_prog and gp and gp >= _SEASON_MIN_GP and rate is not None:
            pct = component_percentile(rate, dist)
        lines.append({
            "season": s.get("season"),
            "team": s.get("team"),
            "gp": int(gp) if gp else None,
            "rate": rate,
            "rate_label": label,
            "rate_unit": unit,
            "in_progress": bool(in_prog),
            "percentile": pct,
        })
    return lines


def _fmt_broad_jump(v):
    total = round(v)
    return f"{total // 12}'{total % 12}\""


_TEST_SPECS = (
    ("forty_yard", "40-Yard Dash", True, lambda v: f"{v:.2f}"),
    ("vertical_inches", "Vertical Jump", False, lambda v: f"{v:.1f}\""),
    ("broad_jump_in", "Broad Jump", False, _fmt_broad_jump),
    ("three_cone", "Three-Cone", True, lambda v: f"{v:.2f}"),
    ("short_shuttle", "Short Shuttle", True, lambda v: f"{v:.2f}"),
    ("bench_reps", "Bench Reps", False, lambda v: f"{int(v)}"),
    ("speed_score", "Speed Score", False, lambda v: f"{v:.1f}"),
    ("ras_score", "RAS", False, lambda v: f"{v:.2f}"),
)


def _testing_distributions(position):
    """{test field: sorted [values]} from tested population players only."""
    pos = (position or "").upper()
    if pos in _testing_dist_cache:
        return _testing_dist_cache[pos]
    dist: dict[str, list] = {}
    rows = _population_player_rows(pos)
    if rows:
        ids = [r["player_id"] for r in rows if r.get("player_id")]
        try:
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                ath_rows = conn.execute(
                    "SELECT * FROM rookie_prospect_athleticism "
                    "WHERE player_id = ANY(%s)",
                    (ids,),
                ).fetchall()
        except Exception:
            ath_rows = []
        for a in ath_rows:
            for field, _label, _lower, _fmt in _TEST_SPECS:
                v = _safe_float(a.get(field))
                if v is not None:
                    dist.setdefault(field, []).append(v)
        for vals in dist.values():
            vals.sort()
    _testing_dist_cache[pos] = dist
    return dist


def _testing_rows(athleticism, position):
    """Testing table rows for one player: only tests he actually has a
    stored result for; percentile only where the tested pool supports
    one (>= 30 tested at the position), else None (the UI prints a dash)."""
    if not athleticism:
        return []
    dist = _testing_distributions(position)
    rows = []
    for field, label, lower, fmt in _TEST_SPECS:
        v = _safe_float(athleticism.get(field))
        if v is None:
            continue
        rows.append({
            "label": label,
            "result": fmt(v),
            "percentile": _percentile_of(v, dist.get(field, []), lower),
        })
    return rows


def _breakout_phrase(seasons, breakout_season, breakout_age):
    """Plain-language breakout line from which stored season he broke out
    in ('Broke out as a true freshman' etc.)."""
    if breakout_season is None or breakout_age is None:
        return None
    years = sorted({s.get("season") for s in (seasons or [])
                    if s.get("season")})
    try:
        idx = years.index(breakout_season)
    except ValueError:
        return None
    if idx == 0:
        return ("Broke out as a true freshman" if breakout_age <= 19.0
                else "Broke out in his first season")
    if idx == 1:
        return "Broke out as a sophomore"
    if idx == 2:
        return "Broke out as a junior"
    return "Broke out as a senior"


def _get_rank_deltas(year) -> dict[str, int]:
    """Rank movement: oldest snapshot in the trailing window vs current rank.

    Positive delta = moved up the board. Returns {} when history is absent.
    """
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT player_id, overall_rank, snapshot_date
                   FROM rookie_value_history
                   WHERE draft_class_year = %s
                     AND snapshot_date >= CURRENT_DATE - INTERVAL '40 days'
                     AND overall_rank IS NOT NULL
                   ORDER BY player_id, snapshot_date""",
                (year,),
            ).fetchall()
    except Exception:
        return {}
    oldest: dict[str, int] = {}
    for r in rows:
        pid = r.get("player_id")
        if pid and pid not in oldest:
            try:
                oldest[pid] = int(r["overall_rank"])
            except (TypeError, ValueError):
                continue
    return oldest


def _get_mock_trend(player_id: str) -> Any:
    """Month trend of mock-draft consensus pick. Positive = rising up boards."""
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            r = conn.execute(
                """SELECT
                     AVG(projected_pick) FILTER (
                       WHERE mock_date >= CURRENT_DATE - INTERVAL '16 days') AS recent,
                     AVG(projected_pick) FILTER (
                       WHERE mock_date >= CURRENT_DATE - INTERVAL '46 days'
                         AND mock_date < CURRENT_DATE - INTERVAL '16 days') AS older
                   FROM rookie_mock_draft_entries
                   WHERE player_id = %s AND projected_pick IS NOT NULL""",
                (player_id,),
            ).fetchone()
        if r and r.get("recent") is not None and r.get("older") is not None:
            return int(round(float(r["older"]) - float(r["recent"])))
    except Exception:
        pass
    return None


# ── Percentile display model (prospect page rework, 2026-10-17) ─────────────
# The board and modal show NO numeric prospect grade. Each player shows his
# rank (still the internal prospect_score order; the score itself stays in
# the payload for downstream consumers like rookie values), a band tag from
# the internal score, and component percentiles. Percentiles are display
# language only: they never feed the grade or the rank.
#
# A component percentile is computed within position against a reference
# population of graded prospects: the player's share of that population at
# or below his stored component score, x100, rounded. The population source
# is reference_population() alone, so the v4 anchor table can replace it
# later without touching display code. A percentile is omitted (hidden,
# never faked) when the population at the position is thinner than
# PCT_MIN_POPULATION or the player's component score is missing.

PCT_COMPONENTS = ("production", "efficiency", "breakout", "competition",
                  "athleticism")
_PCT_SCORE_FIELD = {
    "production": "production_score",
    "efficiency": "efficiency_score",
    "breakout": "breakout_profile_score",
    "competition": "competition_score",
    "athleticism": "athleticism_score",
}
PCT_MIN_POPULATION = 30
# Only model versions whose component scores are on a comparable scale.
_PCT_MODEL_VERSIONS = ("3.2", "3.3")

_population_cache: dict = {}


def reference_population(position, rows=None):
    """Reference population for percentile computation at one position.

    Returns {"values": {component: sorted [scores]}, "classes": [years],
    "n": int}. `rows` may be injected (tests); otherwise every stored
    rookie_rankings row at the position graded by a comparable model
    version (3.2/3.3), across every draft class stored, is used. The
    athleticism value list only includes players with recorded testing
    (a forty time or RAS on file), so untested players, whose stored
    athleticism score is a floor, never anchor the athleticism scale.
    Fail-soft: an empty population on any DB error.
    """
    pos = (position or "").upper()
    injected = rows is not None
    if rows is None:
        if pos in _population_cache:
            return _population_cache[pos]
        rows = []
        try:
            from dashboard_services.db import get_conn
            with get_conn() as conn:
                rows = conn.execute(
                    """SELECT rr.draft_class_year, rr.production_score,
                              rr.efficiency_score, rr.breakout_profile_score,
                              rr.competition_score, rr.athleticism_score,
                              (rpa.forty_yard IS NOT NULL
                               OR rpa.ras_score IS NOT NULL) AS tested
                       FROM rookie_rankings rr
                       JOIN rookie_prospects rp ON rp.player_id = rr.player_id
                       LEFT JOIN rookie_prospect_athleticism rpa
                         ON rpa.player_id = rr.player_id
                       WHERE UPPER(rp.position) = %s
                         AND rr.model_version IN ('3.2', '3.3')""",
                    (pos,),
                ).fetchall() or []
        except Exception:
            log.debug("[rookie_api] reference population failed for %s",
                      pos, exc_info=True)
            rows = []
    values: dict = {c: [] for c in PCT_COMPONENTS}
    classes: set = set()
    for r in rows:
        for comp, field in _PCT_SCORE_FIELD.items():
            if comp == "athleticism" and not r.get("tested"):
                continue
            v = _safe_float(r.get(field))
            if v is not None:
                values[comp].append(v)
        yr = r.get("draft_class_year")
        if yr is not None:
            try:
                classes.add(int(yr))
            except (TypeError, ValueError):
                pass
    pop = {
        "values": {c: sorted(v) for c, v in values.items()},
        "classes": sorted(classes),
        "n": len(rows),
    }
    if not injected and rows:
        _population_cache[pos] = pop
    return pop


def component_percentile(value, values):
    """Share of the population at or below `value`, x100, rounded half up.

    None when the value is missing or the population is too thin to print
    (fewer than PCT_MIN_POPULATION players): the percentile is hidden
    rather than faked.
    """
    import bisect
    import math
    if value is None or not values or len(values) < PCT_MIN_POPULATION:
        return None
    v = _safe_float(value)
    if v is None:
        return None
    share = bisect.bisect_right(values, v) / len(values) * 100.0
    return int(math.floor(share + 0.5))


def player_has_testing(row) -> bool:
    """True when the player has recorded testing (forty time or RAS).

    The stored athleticism_score alone cannot say this: untested players
    carry a floor score in v3.x, so the gate reads the testing fields.
    """
    return (row.get("forty_yard") is not None
            or row.get("ras_score") is not None)


def component_percentiles(row, population) -> dict:
    """Full percentile panel for one player against his position population.

    The athleticism percentile is present only when the player has testing
    data; it is omitted entirely otherwise (never a placeholder).
    """
    out: dict = {}
    tested = player_has_testing(row)
    for comp, field in _PCT_SCORE_FIELD.items():
        if comp == "athleticism" and not tested:
            out[comp] = None
            continue
        out[comp] = component_percentile(
            row.get(field), population["values"].get(comp))
    return out


def percentile_pool_line(positions=("QB", "RB", "WR", "TE")):
    """Board header line describing the reference population actually used.

    Exact format: "Percentiles vs graded prospects at the position,
    classes 2025-2027" (span computed from the classes really present).
    None when no population exists at all.
    """
    classes: set = set()
    for pos in positions:
        classes.update(reference_population(pos)["classes"])
    if not classes:
        return None
    yrs = sorted(classes)
    span = f"{yrs[0]}-{yrs[-1]}" if len(yrs) > 1 else str(yrs[0])
    return f"Percentiles vs graded prospects at the position, classes {span}"


def prospect_tag(score):
    """Band tag from the internal prospect_score (v4 band language).

    92+ Elite Prospect, 80-92 Great Prospect, 70-80 Average Prospect,
    below 70 Day 3. The bands describe what the internal number means;
    they are not buckets and nothing is forced into one. Returns
    {"label", "band"} or None when there is no score.
    """
    s = _safe_float(score)
    if s is None:
        return None
    if s >= 92:
        return {"label": "Elite Prospect", "band": "elite"}
    if s >= 80:
        return {"label": "Great Prospect", "band": "great"}
    if s >= 70:
        return {"label": "Average Prospect", "band": "average"}
    return {"label": "Day 3", "band": "day3"}


def current_college_season(today=None):
    """The college season in progress (or most recently played) for a date.

    College seasons are named by their fall year and run August through
    mid-January, so January still belongs to the previous fall's season.
    """
    from datetime import date as _date
    today = today or _date.today()
    return today.year if today.month >= 8 else today.year - 1


def is_in_season(latest_season, games_played, today=None) -> bool:
    """Data-state check for the In-Season chip.

    The chip is driven by data state: the player's latest stored season
    IS the current college season and it is still partial (1-11 games
    played). A completed season (12+ games) or an older latest season
    never reads as in-season. (The v3.4 stored in_season flag records the
    grading mode at grading time; the display chip stays data-state
    driven, per the native board/modal build.)
    """
    if latest_season is None or games_played is None:
        return False
    try:
        season = int(latest_season)
        gp = int(games_played)
    except (TypeError, ValueError):
        return False
    return season == current_college_season(today) and 1 <= gp <= 11


def _get_latest_season_map(player_ids) -> dict:
    """{player_id: (season, games_played)} for each player's latest stored
    season. Fail-soft: {} on any error (In-Season chips simply do not
    render)."""
    if not player_ids:
        return {}
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT DISTINCT ON (player_id) player_id, season,
                          games_played
                   FROM rookie_prospect_source_data
                   WHERE player_id = ANY(%s)
                   ORDER BY player_id, season DESC""",
                (list(player_ids),),
            ).fetchall() or []
        return {r["player_id"]: (r.get("season"), r.get("games_played"))
                for r in rows if r.get("player_id")}
    except Exception:
        log.debug("[rookie_api] latest-season map failed", exc_info=True)
        return {}


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


_PCT_LABEL = {
    "production": "production",
    "efficiency": "efficiency",
    "breakout": "breakout",
    "competition": "competition",
    "athleticism": "athleticism",
}


def strongest_evidence_receipt(position, seasons, athleticism, percentiles,
                               sp_sos=None):
    """Strongest-evidence line for the modal's Prospect Tag card.

    Names the player's best component percentile and the stored stat line
    behind it, e.g. "99th percentile production: 1,315 receiving yards in
    2024." Every number comes from stored season or athleticism rows;
    returns None when no candidate can be stated honestly.
    """
    if not percentiles:
        return None
    pos = (position or "").upper()
    seasons = [s for s in (seasons or []) if s]
    athleticism = athleticism or {}
    # (percentile, tie-break order, component, stat line)
    candidates: list = []

    def _best_season(field, min_gp=1):
        best = None
        for s in seasons:
            v = _safe_float(s.get(field))
            gp = s.get("games_played") or 0
            if v is None or gp < min_gp:
                continue
            if best is None or v > _safe_float(best.get(field)):
                best = s
        return best

    p = percentiles.get("production")
    if p is not None:
        field = {"QB": "pass_yards", "RB": "rush_yards"}.get(
            pos, "receiving_yards")
        label = {"QB": "passing yards", "RB": "rushing yards"}.get(
            pos, "receiving yards")
        s = _best_season(field)
        if s is not None:
            candidates.append((p, 0, "production",
                               f"{int(_safe_float(s.get(field))):,} {label} "
                               f"in {s.get('season')}"))
    p = percentiles.get("efficiency")
    if p is not None:
        field = {"QB": "yds_per_attempt", "RB": "yds_per_carry"}.get(
            pos, "yds_per_reception")
        label = {"QB": "yards per attempt", "RB": "yards per carry"}.get(
            pos, "yards per reception")
        s = _best_season(field, min_gp=5)
        if s is not None:
            candidates.append((p, 1, "efficiency",
                               f"{_safe_float(s.get(field)):.1f} {label} "
                               f"in {s.get('season')}"))
    p = percentiles.get("breakout")
    if p is not None:
        s = _best_season("dominator_rating")
        if s is not None:
            candidates.append((p, 2, "breakout",
                               f"{round(_safe_float(s.get('dominator_rating')) * 100):.0f}% "
                               f"dominator rating in {s.get('season')}"))
    p = percentiles.get("athleticism")
    if p is not None:
        forty = _safe_float(athleticism.get("forty_yard"))
        ras = _safe_float(athleticism.get("ras_score"))
        if forty is not None:
            candidates.append((p, 3, "athleticism",
                               f"{forty:.2f} forty-yard dash"))
        elif ras is not None:
            candidates.append((p, 3, "athleticism", f"{ras:.1f} RAS"))
    p = percentiles.get("competition")
    if p is not None and sp_sos is not None and seasons:
        latest = max(seasons, key=lambda s: s.get("season") or 0)
        candidates.append((p, 4, "competition",
                           f"{_safe_float(sp_sos):.1f} SP+ strength of "
                           f"schedule in {latest.get('season')}"))
    if not candidates:
        return None
    pct, _order, comp, stat = max(candidates, key=lambda c: (c[0], -c[1]))
    return f"{_ordinal(pct)} percentile {_PCT_LABEL[comp]}: {stat}."


_SEASON_COLS = (
    "season", "games_played", "team", "conference",
    "pass_yards", "pass_tds", "pass_attempts", "completions", "interceptions",
    "rush_attempts", "rush_yards", "rush_tds",
    "receptions", "targets", "receiving_yards", "receiving_tds",
    "dominator_rating", "market_share_yards", "market_share_tds",
    "yds_per_carry", "yds_per_reception", "yds_per_attempt",
    "completion_pct", "td_int_ratio",
)


def _get_prospect_detail(player_id: str, row: dict) -> dict:
    """Everything the prospect modal needs beyond the rankings row.

    All null-safe: missing tables/rows/columns degrade to None/[].
    """
    detail: dict[str, Any] = {
        "seasons": [],
        "athleticism": {},
        "utilization_score": None,
        "experience_score": None,
        "breakout_age": None,
        "breakout_season": None,
        "breakout_phrase": None,
        "season_lines": [],
        "testing": [],
        "mock_trend": None,
        "advanced": [],
        "advanced_seasons": [],
        "advanced_marks": {},
        "archetype": None,
        "archetype_receipt": None,
        "archetype_confidence": None,
        "talent_score": None,
        "capital_delta": None,
        "composite_stars": None,
        "in_season": False,
    }
    seasons: list[dict[str, Any]] = []
    athleticism: dict[str, Any] = {}
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            try:
                srows = conn.execute(
                    "SELECT " + ", ".join(_SEASON_COLS) +
                    " FROM rookie_prospect_source_data"
                    " WHERE player_id = %s ORDER BY season",
                    (player_id,),
                ).fetchall()
                seasons = [_row_to_dict(s) for s in srows]
            except Exception:
                seasons = []
            try:
                arow = conn.execute(
                    """SELECT forty_yard, vertical_inches, broad_jump_in,
                              three_cone, short_shuttle, bench_reps,
                              speed_score, ras_score
                       FROM rookie_prospect_athleticism
                       WHERE player_id = %s""",
                    (player_id,),
                ).fetchone()
                athleticism = _row_to_dict(arow) if arow else {}
            except Exception:
                athleticism = {}
            comp_col = _composite_col(conn)
            if comp_col:
                try:
                    prow = conn.execute(
                        "SELECT " + comp_col + " AS composite_stars"
                        " FROM rookie_prospects WHERE player_id = %s",
                        (player_id,),
                    ).fetchone()
                    if prow and prow.get("composite_stars") is not None:
                        detail["composite_stars"] = int(prow["composite_stars"])
                except Exception:
                    pass
            # WEPA opponent-adjusted efficiency (best normalized score, any season).
            try:
                wrow = conn.execute(
                    """SELECT adj_efficiency_score FROM rookie_prospect_wepa
                       WHERE player_id = %s AND adj_efficiency_score IS NOT NULL
                       ORDER BY adj_efficiency_score DESC LIMIT 1""",
                    (player_id,),
                ).fetchone()
                if wrow and wrow.get("adj_efficiency_score") is not None:
                    detail["wepa_score"] = float(wrow["adj_efficiency_score"])
            except Exception:
                pass
            # Team context: SP+ rating + SOS for the latest season's school.
            try:
                school = row.get("school")
                latest_season = max(
                    (int(s.get("season") or 0) for s in seasons), default=0)
                if school and latest_season:
                    trow = conn.execute(
                        """SELECT sp_rating, sp_sos, talent_composite
                           FROM rookie_team_context
                           WHERE school = %s AND season = %s""",
                        (school, latest_season),
                    ).fetchone()
                    if trow:
                        detail["sp_rating"] = _safe_float(trow.get("sp_rating"))
                        detail["sp_sos"] = _safe_float(trow.get("sp_sos"))
                        detail["talent_composite"] = _safe_float(
                            trow.get("talent_composite"))
            except Exception:
                pass
    except Exception:
        pass

    detail["seasons"] = seasons
    detail["athleticism"] = athleticism
    # Component scores the rankings table does not persist: compute on the fly
    # with the model's own functions from the season lines.
    try:
        from data_building.rookie_pipeline.prospect_model import (
            calc_utilization_score, calc_experience_score,
        )
        if seasons:
            detail["utilization_score"] = round(
                float(calc_utilization_score(seasons, row.get("position") or "")), 1)
            detail["experience_score"] = round(
                float(calc_experience_score(seasons, row.get("position") or "")), 1)
    except Exception:
        pass
    try:
        detail["breakout_age"], detail["breakout_season"] = _compute_breakout(
            seasons, row.get("age"), row.get("position"))
        detail["breakout_phrase"] = _breakout_phrase(
            seasons, detail["breakout_season"], detail["breakout_age"])
    except Exception:
        detail["breakout_age"] = None
        detail["breakout_season"] = None
        detail["breakout_phrase"] = None
    try:
        detail["mock_trend"] = _get_mock_trend(player_id)
    except Exception:
        detail["mock_trend"] = None
    try:
        detail["advanced"] = _build_advanced_metrics(
            row.get("position"), seasons, athleticism, row,
            wepa_score=detail.get("wepa_score"))
    except Exception:
        detail["advanced"] = []
    # True population percentiles for the Advanced tab (percentile-only
    # presentation): each stored-stat row gets the share of the graded
    # population at the position at or below its value. Component-score
    # rows never get one; a thin pool (< 30) leaves it None (dash).
    try:
        _adv_dist = _advanced_distributions(row.get("position"))
        for _a in detail["advanced"]:
            _a["percentile"] = None
            if _a.get("component") or _a.get("value") is None:
                continue
            _a["percentile"] = _percentile_of(
                _a["value"], _adv_dist.get(_a["label"]) or [],
                _a["label"] in _ADV_LOWER_BETTER)
    except Exception:
        for _a in detail.get("advanced") or []:
            _a.setdefault("percentile", None)
    # v3.2: per-season advanced rows + best/career marks from stored rows.
    try:
        detail["advanced_seasons"] = _advanced_season_rows(seasons)
        detail["advanced_marks"] = _advanced_marks(seasons)
    except Exception:
        detail["advanced_seasons"] = []
        detail["advanced_marks"] = {}
    # v3.2 transparent capital: stored with the score at grading time.
    detail["talent_score"] = _safe_float(row.get("talent_score"))
    detail["capital_delta"] = _safe_float(row.get("capital_delta"))
    # v3.4: the row's grading mode, stored with the score at grading time.
    detail["in_season"] = bool(row.get("in_season"))
    # v3.2 archetype: the stored label wins; rows graded before v3.2 are
    # classified live from their season lines so the chip still resolves.
    detail["archetype"] = row.get("archetype")
    detail["archetype_receipt"] = row.get("archetype_receipt")
    detail["archetype_confidence"] = row.get("archetype_confidence")
    if not detail["archetype"] and seasons:
        try:
            from data_building.rookie_pipeline.prospect_archetypes import (
                classify_archetype,
            )
            _arch_profile = {
                "weight_lbs": row.get("weight_lbs"),
                "forty_yard": (athleticism or {}).get("forty_yard"),
                "projected_pick": row.get("projected_pick"),
                "age": row.get("age"),
            }
            _arch = classify_archetype(
                row.get("position") or "", seasons, profile=_arch_profile)
            detail["archetype"] = _arch.get("archetype")
            detail["archetype_receipt"] = _arch.get("archetype_receipt")
            detail["archetype_confidence"] = _arch.get("archetype_confidence")
        except Exception:
            pass
    # Percentile display model: full panel, band tag, in-season state and
    # the strongest-evidence receipt for the Tag card. Testing presence is
    # read off the athleticism row itself (forty/RAS), never the stored
    # score, so an untested player gets no athleticism percentile.
    _pct_row = dict(row)
    _pct_row["forty_yard"] = (athleticism or {}).get("forty_yard")
    _pct_row["ras_score"] = (athleticism or {}).get("ras_score")
    _pcts = component_percentiles(
        _pct_row, reference_population(row.get("position")))
    detail["percentiles"] = _pcts
    detail["tag"] = prospect_tag(row.get("prospect_score"))
    _latest = (max(seasons, key=lambda s: s.get("season") or 0)
               if seasons else None)
    detail["in_season"] = bool(_latest) and is_in_season(
        _latest.get("season"), _latest.get("games_played"))
    detail["evidence_receipt"] = strongest_evidence_receipt(
        row.get("position"), seasons, athleticism, _pcts,
        sp_sos=detail.get("sp_sos"))
    # Season lines (rate + completed-season percentile) and the Testing
    # table rows for the modal's Production / Overview surfaces.
    try:
        detail["season_lines"] = _season_lines(row.get("position"), seasons)
    except Exception:
        detail["season_lines"] = []
    try:
        detail["testing"] = _testing_rows(athleticism, row.get("position"))
    except Exception:
        detail["testing"] = []
    return detail


@rookie_bp.route("/active-class")
def active_class():
    from data_building.rookie_pipeline.pipeline import get_active_rookie_class
    year = get_active_rookie_class()
    return jsonify({"draft_class_year": year})


@rookie_bp.route("/rankings")
def rankings():
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from data_building.rookie_pipeline.value_translation import format_draft_capital

        year = request.args.get("year", type=int) or get_active_rookie_class()
        pos  = (request.args.get("pos") or "").upper() or None
        league_type = (request.args.get("league_type") or "1qb").lower()
        league_size = request.args.get("league_size", type=int) or 10

        # Build mv_map: JSON model → DB player_values (source of truth).
        # player_values contains calibrated values for all players including
        # rookies that have been linked to a sleeper_id. Keyed by player_id
        # (= sleeper_id), so the lookup below by sleeper_id finds them directly.
        try:
            from utils.utils import load_model_value_table as _lmvt
            _json_list = list(_lmvt() or [])
        except Exception:
            _json_list = []
        try:
            from dashboard_services.player_value_history import load_current_values_from_db as _lcvdb
            _db_list = list(_lcvdb() or [])
        except Exception:
            _db_list = []
        mv_map: dict = {str(p["id"]): p for p in _json_list if p.get("id")}
        for p in _db_list:  # DB values overwrite JSON (DB is calibrated source of truth)
            if p.get("id"):
                mv_map[str(p["id"])] = p

        rows = _get_rankings(year)

        # Optional server-side position filter (client can also filter)
        if pos:
            rows = [r for r in rows if (r.get("position") or "").upper() == pos]

        total_players = len(rows)

        # Build response list with value field chosen by league settings
        result = []
        for r in rows:
            d = _row_to_dict(r)

            # Use same model values as player rankings page; fall back to rookie_value.
            # Write calibrated values back into value/sf_value so JS (rkGetValue +
            # the modal) can find them without knowing about display_value.
            mv = mv_map.get(str(d.get("sleeper_id") or "")) or {}
            if mv:
                val_1qb = mv.get(f"value_{league_size}") or mv.get("value")
                val_sf  = mv.get(f"sf_value_{league_size}") or mv.get("sf_value")
                if val_1qb:
                    d["value"] = val_1qb
                    if league_size != 10:
                        d[f"value_{league_size}"] = val_1qb
                if val_sf:
                    d["sf_value"] = val_sf
                    if league_size != 10:
                        d[f"sf_value_{league_size}"] = val_sf

            if league_type == "sf":
                d["display_value"] = (d.get("sf_value" if league_size == 10 else f"sf_value_{league_size}")
                                      or d.get("sf_value")
                                      or d.get("rookie_sf_value" if league_size == 10 else f"rookie_sf_value_{league_size}")
                                      or d.get("rookie_sf_value"))
            else:
                d["display_value"] = (d.get("value" if league_size == 10 else f"value_{league_size}")
                                      or d.get("value")
                                      or d.get("rookie_value" if league_size == 10 else f"rookie_value_{league_size}")
                                      or d.get("rookie_value"))

            # Draft capital label
            d["draft_capital_label"] = format_draft_capital(
                d.get("projected_round"),
                d.get("projected_pick"),
                d.get("projected_pick_low"),
                d.get("projected_pick_high"),
            )
            
            # Add headshot URL as espnHeadshot for modal compatibility
            if d.get("headshot_url"):
                d["espnHeadshot"] = d["headshot_url"]

            result.append(d)

        # Rank movement vs the oldest board snapshot in the trailing window.
        # Positive delta = moved up. Absent history degrades to null.
        try:
            _deltas = _get_rank_deltas(year)
            for d in result:
                pid = d.get("player_id")
                cur = d.get("overall_rank")
                old = _deltas.get(pid) if pid else None
                if old is not None and cur:
                    try:
                        d["rank_delta"] = int(old) - int(cur)
                    except (TypeError, ValueError):
                        d["rank_delta"] = None
                else:
                    d["rank_delta"] = None
        except Exception:
            for d in result:
                d.setdefault("rank_delta", None)

        # Percentile display model (board rework): band tag, component
        # percentiles within position, and the In-Season data-state chip.
        # The internal prospect_score stays in the payload (it ranks the
        # board and feeds rookie values); it is simply never rendered.
        try:
            _pops: dict = {}
            _season_map = _get_latest_season_map(
                [d.get("player_id") for d in result])
            for d in result:
                _pos = (d.get("position") or "").upper()
                if _pos not in _pops:
                    _pops[_pos] = reference_population(_pos)
                d["percentiles"] = component_percentiles(d, _pops[_pos])
                d["tag"] = prospect_tag(d.get("prospect_score"))
                _ls = _season_map.get(d.get("player_id"))
                d["in_season"] = bool(_ls) and is_in_season(_ls[0], _ls[1])
        except Exception:
            log.debug("[rookie_api] percentile enrichment skipped",
                      exc_info=True)

        # Overlay dynasty rookie ADP - read directly from dated cache files,
        # no DB connection required. Falls back to adp_service chain if files absent.
        try:
            import glob as _glob, json as _adpj
            from utils.paths import DATA_DIR as _DATA_DIR

            def _load_adp_file(is_sf: bool) -> dict:
                suffix = "sf" if is_sf else "1qb"
                plain = _DATA_DIR / f"league_adp_rookie_{suffix}_{year}.json"
                if plain.exists():
                    return _adpj.loads(plain.read_text())
                dated = sorted(_glob.glob(str(_DATA_DIR / f"league_adp_rookie_{suffix}_{year}_*.json")))
                if dated:
                    return _adpj.loads(open(dated[-1]).read())
                return {}

            def _extract(raw: dict) -> dict[str, float]:
                out: dict[str, float] = {}
                for pid, entry in raw.items():
                    if isinstance(entry, dict):
                        v = entry.get("avg_pick")
                    else:
                        v = float(entry) if entry else None
                    if v:
                        out[str(pid)] = float(v)
                return out

            sf_map  = _extract(_load_adp_file(True))
            qb1_map = _extract(_load_adp_file(False))

            # Name-based fallback using players_index for prospects missing sleeper_id
            def _norm(n: str) -> str:
                import re as _r
                n = n.lower()
                n = _r.sub(r"['\.\-]", "", n)
                n = _r.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
                return _r.sub(r"\s+", " ", n).strip()

            _sid_to_norm: dict[str, str] = {}
            try:
                from utils.paths import CACHE_DIR as _CD
                _pi_path = _CD / "players_index.json"
                if _pi_path.exists():
                    _pi = _adpj.loads(_pi_path.read_text())
                    for _sid, _pdata in _pi.items():
                        _pn = _pdata.get("name", "")
                        if _pn:
                            _sid_to_norm[str(_sid)] = _norm(_pn)
            except Exception:
                logging.getLogger(__name__).debug("suppressed exception", exc_info=True)

            sf_by_name:  dict[str, float] = {_sid_to_norm[s]: v for s, v in sf_map.items()  if s in _sid_to_norm}
            qb1_by_name: dict[str, float] = {_sid_to_norm[s]: v for s, v in qb1_map.items() if s in _sid_to_norm}

            for d in result:
                sid = str(d.get("sleeper_id") or "")
                if sid:
                    if sid in sf_map:  d["sf_avg_pick"] = sf_map[sid]
                    if sid in qb1_map: d["avg_pick"]    = qb1_map[sid]
                else:
                    pname = _norm(d.get("name") or "")
                    if pname:
                        if pname in sf_by_name:  d["sf_avg_pick"] = sf_by_name[pname]
                        if pname in qb1_by_name: d["avg_pick"]    = qb1_by_name[pname]
        except Exception:
            logging.getLogger(__name__).debug("suppressed exception", exc_info=True)


        # Per-game production trajectory (last 3 seasons) for the dense table trend column.
        # One query for all players; per-game rates avoid comparing partial seasons raw.
        try:
            from dashboard_services.db import get_conn as _get_conn_traj
            _pids = [d.get("player_id") for d in result if d.get("player_id")]
            if _pids:
                with _get_conn_traj() as _tconn:
                    with _tconn.cursor() as _tcur:
                        _tcur.execute(
                            """SELECT player_id, season, games_played,
                                      receiving_yards, rush_yards, pass_yards
                               FROM rookie_prospect_source_data
                               WHERE player_id = ANY(%s)
                               ORDER BY player_id, season""",
                            (_pids,),
                        )
                        _trows = _tcur.fetchall() or []
                _pos_by_pid = {
                    d.get("player_id"): (d.get("position") or "").upper()
                    for d in result
                }
                _traj_map: dict = {}
                for _tr in _trows:
                    _pid = _tr.get("player_id")
                    _gp = _tr.get("games_played") or 0
                    _pos = _pos_by_pid.get(_pid, "")
                    if _pos == "QB":
                        _yds = _tr.get("pass_yards") or 0
                        _lbl = "pass yds/g"
                    elif _pos == "RB":
                        _yds = _tr.get("rush_yards") or 0
                        _lbl = "rush yds/g"
                    else:
                        _yds = _tr.get("receiving_yards") or 0
                        _lbl = "rec yds/g"
                    _pg = round(float(_yds) / _gp, 1) if _gp else 0.0
                    _traj_map.setdefault(_pid, []).append(
                        {"season": _tr.get("season"), "per_game": _pg, "label": _lbl,
                         "gp": _gp, "is_current": _tr.get("season") == 2026})
                for d in result:
                    _t = _traj_map.get(d.get("player_id")) or []
                    # Last 3 seasons only
                    _t = _t[-3:]
                    if _t:
                        d["trajectory"] = {
                            "values": [x["per_game"] for x in _t],
                            "seasons": [x["season"] for x in _t],
                            "label": _t[0]["label"],
                            "is_current_last": _t[-1]["is_current"],
                        }
        except Exception:
            logging.getLogger(__name__).debug("trajectory overlay skipped", exc_info=True)

        # Sort: tier ascending, then display_value descending within each tier
        result.sort(key=lambda x: (x.get("tier") or 99, -(x.get("display_value") or 0)))

        paused = (os.environ.get("ROOKIE_PIPELINE_PAUSED") or "0").strip().lower() in (
            "1", "true", "yes", "on",
        )
        payload = {
            "draft_class_year": year,
            "total_players": total_players,
            "rankings": result,
            "paused": paused,
        }
        # Pool line for the board header, generated from the reference
        # population actually used for the percentiles above.
        try:
            payload["percentile_pool"] = percentile_pool_line()
        except Exception:
            payload["percentile_pool"] = None
        as_of = None
        for row in result:
            lu = row.get("last_updated")
            if lu and (as_of is None or str(lu) > str(as_of)):
                as_of = lu
        if as_of:
            payload["last_updated"] = str(as_of)[:10]
        if paused:
            payload["reason"] = (
                "Rookie prospect rankings are paused until the next draft cycle."
            )
        return jsonify(payload)

    except Exception as exc:
        log.exception("[rookie_api] /rankings error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/player/<player_id>")
def player_detail(player_id: str):
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        year = request.args.get("year", type=int) or get_active_rookie_class()
        rows = _get_rankings(year)
        row  = next((r for r in rows if r["player_id"] == player_id), None)
        if not row:
            return jsonify({"error": "Player not found"}), 404
        
        player_data = _row_to_dict(row)

        # Add headshot URL as espnHeadshot for modal compatibility
        if player_data.get("headshot_url"):
            player_data["espnHeadshot"] = player_data["headshot_url"]

        # Modal detail bundle: seasons, athleticism, computed extras. Additive
        # and null-safe; the page renders fine when pieces are missing.
        try:
            player_data.update(_get_prospect_detail(player_id, row))
        except Exception as exc:
            log.error("[rookie_api] /player detail failed for %s: %s", player_id, exc, exc_info=True)
            player_data["_detail_error"] = str(exc)

        return jsonify(player_data)
    except Exception as exc:
        log.exception("[rookie_api] /player error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/prospects", methods=["POST"])
@admin_required
def add_prospects():
    """
    Add or update one or more prospects with their full data.

    Accepts a single prospect object or {"prospects": [...]}.

    Required fields per prospect:  name, position, draft_class_year
    Optional fields:               player_id, school, age, height_inches,
                                   weight_lbs, early_declare, transfer_history,
                                   headshot_url, seasons, athleticism

    seasons[] fields:
        season, games_played,
        pass_yards, pass_tds, pass_attempts, completions, interceptions,
        rush_attempts, rush_yards, rush_tds,
        receptions, targets, receiving_yards, receiving_tds,
        yds_per_carry, yds_per_reception, yds_per_attempt,
        completion_pct, td_int_ratio, dominator_rating,
        market_share_yards, market_share_tds,
        team, conference, team_pass_rate

    athleticism fields:
        forty_yard, vertical_inches, broad_jump_in, three_cone,
        short_shuttle, bench_reps, speed_score, ras_score

    Returns: {"added": N, "prospects": [scored_row, ...]}
    Each scored row includes all component scores, values, tier, and rank.
    """
    try:
        body = request.json or {}

        # Accept single prospect dict or {"prospects": [...]}
        if "prospects" in body:
            incoming = body["prospects"]
        elif "name" in body:
            incoming = [body]
        else:
            return jsonify({"error": 'Expected a prospect object or {"prospects": [...]}'}), 400

        if not incoming:
            return jsonify({"error": "No prospects provided"}), 400

        from data_building.rookie_pipeline.ingestion import normalize_prospect
        from data_building.rookie_pipeline.prospect_model import (
            active_season_year_for_run,
            score_prospect,
        )
        from data_building.rookie_pipeline.mock_draft_consensus import build_mock_draft_consensus
        from data_building.rookie_pipeline.value_translation import translate_score_to_value, format_draft_capital
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class

        def _make_player_id(name: str, draft_year: int) -> str:
            slug = re.sub(r"[^A-Z0-9]+", "_", name.upper()).strip("_")
            return f"ROOKIE_{draft_year}_{slug}"

        scored_rows = []

        for raw in incoming:
            if not raw.get("name"):
                return jsonify({"error": "Each prospect must have a 'name'"}), 400
            if not raw.get("position"):
                return jsonify({"error": f"Prospect '{raw['name']}' is missing 'position'"}), 400

            draft_year = int(raw.get("draft_class_year") or get_active_rookie_class())
            raw["draft_class_year"] = draft_year

            if not raw.get("player_id"):
                raw["player_id"] = _make_player_id(raw["name"], draft_year)

            prospect = normalize_prospect(raw)

            # Fetch any existing mock draft consensus for this player
            consensus_map = build_mock_draft_consensus(draft_year)
            dc = consensus_map.get(prospect["player_id"])

            # Score and translate to dynasty values. v3.4: live regrades
            # use the same run-calendar active season year as the pipeline.
            scores = score_prospect(
                prospect, dc,
                _active_season_year=active_season_year_for_run(),
            )
            values = translate_score_to_value(scores, prospect, dc)

            # Build a flat row matching the shape returned by _merge_inmemory_result
            ath = prospect.get("athleticism") or {}
            row: dict[str, Any] = {
                "player_id":                     prospect["player_id"],
                "draft_class_year":              draft_year,
                "name":                          prospect.get("name"),
                "position":                      prospect.get("position"),
                "school":                        prospect.get("school"),
                "age":                           prospect.get("age"),
                "height_inches":                 prospect.get("height_inches"),
                "weight_lbs":                    prospect.get("weight_lbs"),
                "early_declare":                 prospect.get("early_declare"),
                "transfer_history":              prospect.get("transfer_history"),
                "overall_rank":                  None,   # filled after re-sort below
                "position_rank":                 None,
                "prospect_score":                scores.get("prospect_score"),
                "rookie_value":                  values.get("rookie_value"),
                "rookie_sf_value":               values.get("rookie_sf_value"),
                "rookie_value_8":                values.get("rookie_value_8"),
                "rookie_value_12":               values.get("rookie_value_12"),
                "rookie_value_14":               values.get("rookie_value_14"),
                "rookie_sf_value_8":             values.get("rookie_sf_value_8"),
                "rookie_sf_value_12":            values.get("rookie_sf_value_12"),
                "rookie_sf_value_14":            values.get("rookie_sf_value_14"),
                "tier":                          values.get("tier"),
                "tier_label":                    values.get("tier_label"),
                "key_reasons":                   scores.get("key_reasons"),
                "production_score":              scores.get("production_score"),
                "efficiency_score":              scores.get("efficiency_score"),
                "age_score":                     scores.get("age_score"),
                "breakout_profile_score":        scores.get("breakout_profile_score"),
                "athleticism_score":             scores.get("athleticism_score"),
                "competition_score":             scores.get("competition_score"),
                "environment_adjustment":        scores.get("environment_adjustment"),
                "durability_score":              scores.get("durability_score"),
                "projected_draft_capital_score": scores.get("projected_draft_capital_score"),
                "fantasy_translation_score":     scores.get("fantasy_translation_score"),
                "confidence_score":              scores.get("confidence_score"),
                "in_season":                     bool(scores.get("in_season")),
                "calculated_at":                 None,
                "projected_round":               dc.get("projected_round") if dc else None,
                "projected_pick":                dc.get("projected_pick") if dc else None,
                "projected_pick_low":            dc.get("projected_pick_low") if dc else None,
                "projected_pick_high":           dc.get("projected_pick_high") if dc else None,
                "num_mocks_used":                dc.get("num_mocks_used") if dc else None,
                "consensus_confidence":          dc.get("consensus_confidence") if dc else None,
                "forty_yard":                    ath.get("forty_yard"),
                "ras_score":                     ath.get("ras_score"),
            }
            row["draft_capital_label"] = format_draft_capital(
                row["projected_round"], row["projected_pick"],
                row["projected_pick_low"], row["projected_pick_high"],
            )

            # Merge into the in-memory rankings cache for this year,
            # replacing any existing entry with the same player_id.
            current = _get_rankings(draft_year)
            current = [r for r in current if r.get("player_id") != prospect["player_id"]]
            current.append(row)

            # Re-sort by prospect_score and re-assign overall + position ranks
            current.sort(key=lambda x: x.get("prospect_score") or 0.0, reverse=True)
            pos_counters: dict[str, int] = {}
            for i, r in enumerate(current):
                r["overall_rank"] = i + 1
                pos = (r.get("position") or "UNK").upper()
                pos_counters[pos] = pos_counters.get(pos, 0) + 1
                r["position_rank"] = pos_counters[pos]

            _cache[draft_year] = current

            # Retrieve the newly ranked row for the response
            updated = next((r for r in current if r["player_id"] == prospect["player_id"]), row)
            scored_rows.append(_row_to_dict(updated))

            # Persist to DB (best-effort - non-fatal if DB is unavailable)
            try:
                from data_building.rookie_pipeline.pipeline import upsert_prospects, upsert_rankings
                from dashboard_services.db import get_conn
                with get_conn() as conn:
                    upsert_prospects([prospect], conn)
                    upsert_rankings([scores], [values], conn)
                    conn.commit()
                log.info("[rookie_api] Persisted prospect %s to DB", prospect["player_id"])
            except Exception as db_exc:
                log.warning("[rookie_api] DB upsert skipped (DB unavailable): %s", db_exc)

        return jsonify({"added": len(scored_rows), "prospects": scored_rows})

    except Exception as exc:
        log.exception("[rookie_api] POST /prospects error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/comparables/<player_id>")
def comparables(player_id: str):
    """Return historical prospects at the same position with a similar prospect score."""
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from dashboard_services.db import get_conn

        year = request.args.get("year", type=int) or get_active_rookie_class()
        rows = _get_rankings(year)
        prospect = next((r for r in rows if r["player_id"] == player_id), None)

        if not prospect:
            return jsonify({"comparables": []})

        position = (prospect.get("position") or "").upper()
        score = float(prospect.get("prospect_score") or 0)
        band = 5.0  # ±16 points prospect_score

        try:
            with get_conn() as conn:
                db_rows = conn.execute(
                    """
                    SELECT player_id, name, position, draft_class_year, school,
                           prospect_score, tier, tier_label, overall_rank, position_rank,
                           actual_pick, actual_round, actual_nfl_team, headshot_url
                    FROM historical_prospect_grades
                    WHERE position = %s
                      AND prospect_score BETWEEN %s AND %s
                      AND draft_class_year < %s
                    ORDER BY ABS(prospect_score - %s) ASC, draft_class_year DESC
                    LIMIT 5
                    """,
                    (position, score - band, score + band, year, score),
                ).fetchall()

            result = []
            for r in db_rows:
                result.append({
                    "player_id":        r["player_id"],
                    "name":             r["name"],
                    "position":         r["position"],
                    "draft_class_year": r["draft_class_year"],
                    "school":           r["school"],
                    "prospect_score":   float(r["prospect_score"] or 0),
                    "tag":              prospect_tag(r["prospect_score"]),
                    "tier":             r["tier"],
                    "tier_label":       r["tier_label"],
                    "overall_rank":     r["overall_rank"],
                    "position_rank":    r["position_rank"],
                    "actual_pick":      r["actual_pick"],
                    "actual_round":     r["actual_round"],
                    "actual_nfl_team":  r["actual_nfl_team"],
                    "headshot_url":     r["headshot_url"],
                })
        except Exception as db_exc:
            log.warning("[rookie_api] comparables DB error: %s", db_exc)
            result = []

        return jsonify({"comparables": result})

    except Exception as exc:
        log.exception("[rookie_api] /comparables error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/by-sleeper/<sleeper_id>")
def by_sleeper(sleeper_id: str):
    """Return prospect data for a player identified by their Sleeper player ID."""
    try:
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from data_building.rookie_pipeline.value_translation import format_draft_capital
        from dashboard_services.db import get_conn

        # Check active class rankings first (in-memory)
        year = get_active_rookie_class()
        for y in [year, year - 1]:
            rows = _get_rankings(y)
            row = next((r for r in rows if str(r.get("sleeper_id") or "") == str(sleeper_id)), None)
            if row:
                d = _row_to_dict(row)
                d["draft_capital_label"] = format_draft_capital(
                    d.get("projected_round"), d.get("projected_pick"),
                    d.get("projected_pick_low"), d.get("projected_pick_high"),
                )
                if d.get("headshot_url"):
                    d["espnHeadshot"] = d["headshot_url"]
                return jsonify(d)

        # Fallback: query DB directly
        try:
            with get_conn() as conn:
                row = conn.execute(
                    """
                    SELECT rp.*, rr.prospect_score, rr.tier, rr.tier_label,
                           rr.overall_rank, rr.position_rank,
                           rr.production_score, rr.efficiency_score, rr.age_score,
                           rr.breakout_profile_score, rr.athleticism_score,
                           rr.competition_score, rr.projected_draft_capital_score,
                           rr.confidence_score, rr.key_reasons,
                           rr.rookie_value, rr.rookie_sf_value,
                           rr.rookie_value_8, rr.rookie_value_12, rr.rookie_value_14,
                           rr.rookie_sf_value_8, rr.rookie_sf_value_12, rr.rookie_sf_value_14,
                           rmc.projected_round, rmc.projected_pick,
                           rmc.projected_pick_low, rmc.projected_pick_high,
                           rmc.num_mocks_used,
                           rpa.forty_yard, rpa.ras_score
                    FROM rookie_prospects rp
                    JOIN rookie_rankings rr ON rp.player_id = rr.player_id
                    LEFT JOIN rookie_mock_draft_consensus rmc ON rp.player_id = rmc.player_id
                    LEFT JOIN rookie_prospect_athleticism rpa ON rp.player_id = rpa.player_id
                    WHERE rp.sleeper_id = %s
                    ORDER BY rr.draft_class_year DESC
                    LIMIT 1
                    """,
                    (str(sleeper_id),),
                ).fetchone()

                if row:
                    d = dict(row)
                    d["draft_capital_label"] = format_draft_capital(
                        d.get("projected_round"), d.get("projected_pick"),
                        d.get("projected_pick_low"), d.get("projected_pick_high"),
                    )
                    if d.get("headshot_url"):
                        d["espnHeadshot"] = d["headshot_url"]
                    return jsonify(_row_to_dict(d))
        except Exception as db_exc:
            log.warning("[rookie_api] by-sleeper DB error: %s", db_exc)

        return jsonify({"error": "Prospect not found for sleeper_id"}), 404

    except Exception as exc:
        log.exception("[rookie_api] /by-sleeper error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/link-sleeper", methods=["POST"])
@admin_required
def link_sleeper():
    """Link a prospect's rookie player_id to their Sleeper player ID and optionally promote to player_values."""
    try:
        body = request.json or {}
        player_id = str(body.get("player_id") or "").strip()
        sleeper_id = str(body.get("sleeper_id") or "").strip()

        if not player_id or not sleeper_id:
            return jsonify({"error": "player_id and sleeper_id are required"}), 400

        from dashboard_services.db import get_conn
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class

        # Update DB
        try:
            with get_conn() as conn:
                conn.execute(
                    "UPDATE rookie_prospects SET sleeper_id = %s, updated_at = NOW() WHERE player_id = %s",
                    (sleeper_id, player_id),
                )
                conn.commit()
        except Exception as db_exc:
            log.warning("[rookie_api] link-sleeper DB error: %s", db_exc)
            return jsonify({"error": "Database update failed"}), 500

        # Update in-memory cache
        year = get_active_rookie_class()
        for y in [year, year - 1]:
            rows = _get_rankings(y)
            for r in rows:
                if r.get("player_id") == player_id:
                    r["sleeper_id"] = sleeper_id
                    break

        # Optionally promote: insert into player_values so the player appears in the main system
        promote = body.get("promote", True)
        promoted = False
        if promote:
            try:
                rows = _get_rankings(year)
                row = next((r for r in rows if r["player_id"] == player_id), None)
                if not row:
                    for y in [year - 1]:
                        row = next((r for r in _get_rankings(y) if r["player_id"] == player_id), None)
                        if row:
                            break

                if row:
                    val_1qb = float(row.get("rookie_value") or 0)
                    val_sf = float(row.get("rookie_sf_value") or 0)
                    pos = row.get("position", "")
                    name = row.get("name", "")
                    pos_rank = row.get("position_rank")
                    pos_rank_label = f"{pos}{pos_rank}" if pos and pos_rank else ""

                    with get_conn() as conn:
                        conn.execute(
                            """
                            INSERT INTO player_values
                                (player_id, value_1qb, value_sf, calibrated_value_1qb, calibrated_value_sf,
                                 position, pos_rank, pos_rank_label, last_updated)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
                            ON CONFLICT (player_id) DO UPDATE SET
                                value_1qb = EXCLUDED.value_1qb,
                                value_sf = EXCLUDED.value_sf,
                                calibrated_value_1qb = EXCLUDED.calibrated_value_1qb,
                                calibrated_value_sf = EXCLUDED.calibrated_value_sf,
                                position = EXCLUDED.position,
                                pos_rank = EXCLUDED.pos_rank,
                                pos_rank_label = EXCLUDED.pos_rank_label,
                                last_updated = EXCLUDED.last_updated
                            """,
                            (sleeper_id, val_1qb, val_sf, val_1qb, val_sf,
                             pos, pos_rank, pos_rank_label),
                        )
                        # Seed a value history row so the chart has at least one point
                        conn.execute(
                            """
                            INSERT INTO player_value_history
                                (as_of_date, player_id, name, position, value, source)
                            VALUES (CURRENT_DATE, %s, %s, %s, %s, 'model')
                            ON CONFLICT (as_of_date, player_id, source) DO UPDATE SET
                                value = EXCLUDED.value
                            """,
                            (sleeper_id, name, pos, val_1qb),
                        )
                        conn.commit()
                    promoted = True
            except Exception as prom_exc:
                log.warning("[rookie_api] link-sleeper promote error: %s", prom_exc)

        return jsonify({"ok": True, "player_id": player_id, "sleeper_id": sleeper_id, "promoted": promoted})

    except Exception as exc:
        log.exception("[rookie_api] /link-sleeper error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/auto-link/<player_id>")
def auto_link(player_id: str):
    """Auto-match a prospect to their Sleeper ID via players_index.json name lookup, then promote."""
    try:
        from utils.utils import load_players_index
        from data_building.rookie_pipeline.pipeline import get_active_rookie_class
        from dashboard_services.db import get_conn

        year = get_active_rookie_class()
        row = None
        for y in [year, year - 1]:
            row = next((r for r in _get_rankings(y) if r.get("player_id") == player_id), None)
            if row:
                break

        if not row:
            return jsonify({"ok": False, "error": "Prospect not found"}), 404

        if row.get("sleeper_id"):
            return jsonify({"ok": True, "sleeper_id": row["sleeper_id"], "already_linked": True})

        prospect_name = row.get("name", "")
        if not prospect_name:
            return jsonify({"ok": False, "error": "Prospect has no name"}), 400

        players_index = load_players_index() or {}

        def _norm(n: str) -> str:
            n = n.lower()
            n = re.sub(r"['\.\-]", "", n)
            n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
            return re.sub(r"\s+", " ", n).strip()

        norm_prospect = _norm(prospect_name)
        sleeper_id = None
        for sid, pdata in players_index.items():
            if _norm(pdata.get("name", "")) == norm_prospect:
                sleeper_id = sid
                break

        if not sleeper_id:
            return jsonify({"ok": False, "error": f"No match for '{prospect_name}'"})

        try:
            with get_conn() as conn:
                conn.execute(
                    "UPDATE rookie_prospects SET sleeper_id = %s, updated_at = NOW() WHERE player_id = %s",
                    (sleeper_id, player_id),
                )
                conn.commit()
        except Exception as db_exc:
            log.warning("[rookie_api] auto-link DB error: %s", db_exc)
            return jsonify({"error": "Database update failed"}), 500

        for y in [year, year - 1]:
            for r in _get_rankings(y):
                if r.get("player_id") == player_id:
                    r["sleeper_id"] = sleeper_id
                    break

        promoted = False
        try:
            val_1qb = float(row.get("rookie_value") or 0)
            val_sf = float(row.get("rookie_sf_value") or 0)
            pos = row.get("position", "")
            name = row.get("name", "")
            pos_rank = row.get("position_rank")
            pos_rank_label = f"{pos}{pos_rank}" if pos and pos_rank else ""

            with get_conn() as conn:
                conn.execute(
                    """
                    INSERT INTO player_values
                        (player_id, value_1qb, value_sf, calibrated_value_1qb, calibrated_value_sf,
                         position, pos_rank, pos_rank_label, last_updated)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_DATE)
                    ON CONFLICT (player_id) DO UPDATE SET
                        value_1qb = EXCLUDED.value_1qb,
                        value_sf = EXCLUDED.value_sf,
                        calibrated_value_1qb = EXCLUDED.calibrated_value_1qb,
                        calibrated_value_sf = EXCLUDED.calibrated_value_sf,
                        position = EXCLUDED.position,
                        pos_rank = EXCLUDED.pos_rank,
                        pos_rank_label = EXCLUDED.pos_rank_label,
                        last_updated = EXCLUDED.last_updated
                    """,
                    (sleeper_id, val_1qb, val_sf, val_1qb, val_sf, pos, pos_rank, pos_rank_label),
                )
                conn.execute(
                    """
                    INSERT INTO player_value_history
                        (as_of_date, player_id, name, position, value, source)
                    VALUES (CURRENT_DATE, %s, %s, %s, %s, 'model')
                    ON CONFLICT (as_of_date, player_id, source) DO UPDATE SET
                        value = EXCLUDED.value
                    """,
                    (sleeper_id, name, pos, val_1qb),
                )
                conn.commit()
            promoted = True
        except Exception as prom_exc:
            log.warning("[rookie_api] auto-link promote error: %s", prom_exc)

        return jsonify({"ok": True, "sleeper_id": sleeper_id, "already_linked": False, "promoted": promoted})

    except Exception as exc:
        log.exception("[rookie_api] /auto-link error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/draft-status", methods=["GET"])
def draft_status():
    """Check if the draft is complete for a given year."""
    try:
        import datetime as _dt
        year = request.args.get("year", type=int)
        if year is None:
            from data_building.rookie_pipeline.pipeline import get_active_rookie_class
            year = get_active_rookie_class()

        from data_building.rookie_pipeline.pipeline import is_draft_complete
        from dashboard_services.db import get_conn

        draft_date = None
        with get_conn() as conn:
            draft_complete = is_draft_complete(year, conn)
            try:
                row = conn.execute(
                    "SELECT draft_date FROM rookie_active_class WHERE draft_class_year = %s",
                    (year,),
                ).fetchone()
                if row and row["draft_date"]:
                    d = row["draft_date"]
                    if isinstance(d, str):
                        d = _dt.datetime.strptime(d[:10], "%Y-%m-%d").date()
                    draft_date = d
            except Exception:
                logging.getLogger(__name__).debug("suppressed exception", exc_info=True)

        if draft_date is None:
            draft_date = _dt.date(year, 4, 26)  # typical end-of-draft fallback

        today = _dt.date.today()
        days_since = (today - draft_date).days if draft_complete else None

        return jsonify({
            "draft_year":       year,
            "draft_complete":   draft_complete,
            "days_since_draft": days_since,
        })
    except Exception as exc:
        log.exception("[rookie_api] /draft-status error")
        return jsonify({"error": str(exc)}), 500


@rookie_bp.route("/refresh", methods=["POST"])
@admin_required
def refresh():
    """Re-run the pipeline and bust the in-memory cache."""
    try:
        from data_building.rookie_pipeline.pipeline import (
            get_active_rookie_class, run_rookie_pipeline,
        )
        year = request.json.get("year") if request.json else None
        if year is None:
            year = get_active_rookie_class()
        year = int(year)

        _cache.pop(year, None)
        run_rookie_pipeline(year)
        _cache.pop(year, None)  # force fresh DB read on next request

        return jsonify({"status": "ok", "draft_class_year": year})
    except Exception as exc:
        log.exception("[rookie_api] /refresh error")
        return jsonify({"error": str(exc)}), 500


def register_rookie_routes(app):
    app.register_blueprint(rookie_bp)
    return app
