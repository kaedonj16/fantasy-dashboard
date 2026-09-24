"""Front Office Report v2: structured, data-grounded team report.

Replaces the prose-only GM memo on the Season Hub. The report is two parts:

- Card summary: verdict stamp, headline, key numbers, top move, and a
  "View full report" button. Rendered into the existing Season Hub card.
- Full report: opened as a modal. Verdict, since-last-week changes, roster
  table, computed positional grades, trade targets with prefilled Trade
  Analyzer links, waiver targets, cut candidates, and a GM alert.

Design rules (from the product review):
- Grades, ranks, values, and tiers are computed server-side. The model only
  writes narrative around numbers it is given. It never invents players,
  stats, or ranks.
- Copy uses no em dashes. Records use hyphens (2-0).
"""

from __future__ import annotations

import html
import json
import logging

from dashboard_services.ai.cache import (
    build_ai_cache_key,
    load_cached_ai_text,
    save_cached_ai_text,
)
from dashboard_services.ai.client import AIRateLimitError, AIUnavailableError
from dashboard_services.ai.context_builders import (
    _ctx_is_sf,
    build_model_value_lookup,
    build_team_gm_context,
    build_trade_suggestions_context,
    ctx_scoring_type,
)
from dashboard_services.ai.prompts import (
    build_front_office_prompt_payload,
    generate_front_office_report_result,
    normalize_trade_scoring_type,
)
from dashboard_services.ai.renderer import (
    _ai_error_notice,
    _ctx_with_playoff_odds,
    _emit_ai_html,
    ai_available,
)
from dashboard_services.providers.espn_api import safe_float
from utils.lineup_slots import canonicalize_slots, count_lineup_slots
from utils.roster_strength import STARTER_THRESHOLD

logger = logging.getLogger(__name__)

CACHE_VERSION = "v1"
_SKILL_POS = ("QB", "RB", "WR", "TE")


# ──────────────────────────────────────────────────────────────────────────────
# Data builders (all computed, no model judgment)
# ──────────────────────────────────────────────────────────────────────────────

def _safe_str(v) -> str:
    return "" if v is None else str(v)


def _roster_lookup(ctx: dict, viewer_roster_id: str) -> dict | None:
    for r in ctx.get("rosters") or []:
        if str(r.get("roster_id")) == str(viewer_roster_id):
            return r
    return None


def _build_roster_rows(ctx: dict, roster: dict, model_value_lookup: dict) -> list[dict]:
    """Every rostered player with market data, sorted by value desc."""
    players_index = ctx.get("players_index") or {}
    players_map = ctx.get("players_map") or {}
    players_full = ctx.get("players") or {}
    rows: list[dict] = []
    for pid in roster.get("players") or []:
        spid = str(pid)
        mv = model_value_lookup.get(spid) or {}
        meta = players_index.get(spid) or players_map.get(spid) or {}
        full = players_full.get(spid) or {}
        name = meta.get("full_name") or meta.get("name") or mv.get("name") or "Unknown"
        pos = str(meta.get("position") or meta.get("pos") or mv.get("position") or "?").upper()
        team = meta.get("team") or mv.get("team") or "FA"
        value = round(safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value")), 1)
        threshold = STARTER_THRESHOLD.get(pos, 350)
        role = "Starter" if value >= threshold else "Depth"
        raw_status = str(full.get("injury_status") or full.get("status") or "").strip().upper()
        injury = "" if raw_status in ("", "ACTIVE", "ACT") else raw_status
        trend = mv.get("rank_change_7d")
        try:
            trend = None if trend is None else float(trend)
        except (TypeError, ValueError):
            trend = None
        rows.append({
            "id": spid,
            "name": name,
            "position": pos,
            "team": team,
            "age": meta.get("age") if meta.get("age") not in (None, "") else mv.get("age"),
            "value": value,
            "pos_rank": mv.get("pos_rank"),
            "pos_rank_label": mv.get("pos_rank_label") or "",
            "role": role,
            "trend_7d": trend,
            "injury": injury,
        })
    rows.sort(key=lambda r: r["value"], reverse=True)
    return rows


def _grade_for_percentile(pct: float) -> str:
    if pct >= 90:
        return "A"
    if pct >= 75:
        return "B"
    if pct >= 55:
        return "C"
    if pct >= 30:
        return "D"
    return "F"


def _positional_grades(ctx: dict, viewer_roster_id: str, model_value_lookup: dict) -> list[dict]:
    """League-relative positional grades. Score = sum of top-N values per
    position where N tracks the league's starter slots. Rank is explicit so
    the grade is auditable."""
    slots = count_lineup_slots(canonicalize_slots(ctx.get("roster_positions") or []))
    need = {
        "QB": max(1, slots.get("QB", 1) + slots.get("SUPER_FLEX", 0)),
        "RB": max(2, slots.get("RB", 2) + slots.get("FLEX", 0)),
        "WR": max(2, slots.get("WR", 2) + slots.get("FLEX", 0)),
        "TE": max(1, slots.get("TE", 1)),
    }
    rosters = ctx.get("rosters") or []
    n_teams = len(rosters)

    def _team_pos_sum(roster: dict, pos: str, k: int) -> float:
        vals = []
        for pid in roster.get("players") or []:
            mv = model_value_lookup.get(str(pid)) or {}
            p = str(mv.get("position") or "").upper()
            if p != pos:
                continue
            vals.append(safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value")))
        vals.sort(reverse=True)
        return round(sum(vals[:k]), 1)

    grades = []
    for pos in _SKILL_POS:
        k = need[pos]
        scored = sorted(
            ((_team_pos_sum(r, pos, k), str(r.get("roster_id"))) for r in rosters),
            key=lambda t: t[0],
            reverse=True,
        )
        viewer_score = next((s for s, rid in scored if rid == str(viewer_roster_id)), 0.0)
        rank = next((i + 1 for i, (_, rid) in enumerate(scored) if rid == str(viewer_roster_id)), n_teams)
        pct = 100.0 * (n_teams - rank) / max(n_teams - 1, 1) if n_teams > 1 else 50.0
        grades.append({
            "pos": pos,
            "grade": _grade_for_percentile(pct),
            "rank": rank,
            "of": n_teams,
            "score": viewer_score,
        })
    return grades


def _last_week_result(ctx: dict, viewer_roster_id: str) -> dict | None:
    """Best-effort: last completed week's result from df_weekly."""
    try:
        dfw = ctx.get("df_weekly")
        week = int(ctx.get("current_week") or 0)
        if dfw is None or week <= 1 or getattr(dfw, "empty", True):
            return None
        cols = set(dfw.columns)
        if not {"week", "points"}.issubset(cols):
            return None
        rid_col = "roster_id" if "roster_id" in cols else ("owner" if "owner" in cols else None)
        if not rid_col:
            return None
        last = week - 1
        wk = dfw[dfw["week"] == last]
        if wk.empty:
            return None
        me = wk[wk[rid_col].astype(str) == str(viewer_roster_id)]
        if me.empty:
            return None
        me_row = me.iloc[0]
        my_pts = round(float(me_row.get("points") or 0.0), 1)
        opp_pts = None
        opp_name = ""
        mid_col = "matchup_id" if "matchup_id" in cols else None
        if mid_col is not None:
            mid = me_row.get(mid_col)
            foes = wk[(wk[mid_col] == mid) & (wk[rid_col].astype(str) != str(viewer_roster_id))]
            if not foes.empty:
                foe = foes.iloc[0]
                opp_pts = round(float(foe.get("points") or 0.0), 1)
                opp_name = _safe_str(foe.get("owner") or foe.get("team_name") or "")
        result = "W" if opp_pts is not None and my_pts > opp_pts else ("L" if opp_pts is not None and my_pts < opp_pts else "T")
        return {"week": last, "result": result, "pf": my_pts, "pa": opp_pts, "opponent": opp_name}
    except Exception:
        logger.debug("[front-office] last-week lookup failed", exc_info=True)
        return None


def _this_week_matchup(ctx: dict, viewer_roster_id: str) -> dict | None:
    """Best-effort: current week's opponent and their record."""
    try:
        dfw = ctx.get("df_weekly")
        week = int(ctx.get("current_week") or 0)
        if dfw is None or week < 1 or getattr(dfw, "empty", True):
            return None
        cols = set(dfw.columns)
        rid_col = "roster_id" if "roster_id" in cols else ("owner" if "owner" in cols else None)
        mid_col = "matchup_id" if "matchup_id" in cols else None
        if not rid_col or not mid_col:
            return None
        wk = dfw[dfw["week"] == week]
        me = wk[wk[rid_col].astype(str) == str(viewer_roster_id)]
        if me.empty:
            return None
        mid = me.iloc[0].get(mid_col)
        foes = wk[(wk[mid_col] == mid) & (wk[rid_col].astype(str) != str(viewer_roster_id))]
        if foes.empty:
            return None
        foe = foes.iloc[0]
        return {
            "opponent": _safe_str(foe.get("owner") or foe.get("team_name") or ""),
            "opponent_roster_id": _safe_str(foe.get("roster_id") or ""),
        }
    except Exception:
        logger.debug("[front-office] this-week lookup failed", exc_info=True)
        return None


def _trade_targets(ctx: dict, viewer_roster_id: str, scoring_type: str) -> list[dict]:
    """Top 3 trade packages from the existing suggestions engine, with
    prefilled Trade Analyzer deep links. Side A = you get, Side B = you give."""
    try:
        sugg = build_trade_suggestions_context(_ctx_with_playoff_odds(ctx, block=False), viewer_roster_id)
    except Exception:
        logger.debug("[front-office] trade suggestions failed", exc_info=True)
        return []
    if not sugg:
        return []
    partners = sorted(
        sugg.get("top_partners") or [],
        key=lambda p: float(p.get("suggestion_score") or 0),
        reverse=True,
    )[:3]
    platform = _safe_str(ctx.get("platform") or "sleeper")
    season = ctx.get("current_season") or ctx.get("season") or ""
    league_id = _safe_str(ctx.get("league_id") or "")
    targets = []
    for p in partners:
        gets = (p.get("targets_they_have") or [])[:2]
        gives = (p.get("targets_viewer_sends") or [])[:3]
        if not gets or not gives:
            continue
        get_ids = ",".join(str(g.get("id")) for g in gets if g.get("id"))
        give_ids = ",".join(str(g.get("id")) for g in gives if g.get("id"))
        url = (
            f"/{platform}/{season}/{league_id}/trade"
            f"?a={get_ids}&b={give_ids}"
        )
        targets.append({
            "partner": _safe_str(p.get("team_name") or ""),
            "gets": [
                {
                    "id": str(g.get("id")),
                    "name": _safe_str(g.get("name")),
                    "position": _safe_str(g.get("position")),
                    "age": g.get("age"),
                    "value": round(safe_float(g.get("value")), 1),
                }
                for g in gets
            ],
            "gives": [
                {
                    "id": str(g.get("id")),
                    "name": _safe_str(g.get("name")),
                    "position": _safe_str(g.get("position")),
                    "value": round(safe_float(g.get("value")), 1),
                }
                for g in gives
            ],
            "fairness": p.get("fairness"),
            "analyzer_url": url,
        })
    return targets


def _waiver_targets(ctx: dict, viewer_roster_id: str, model_value_lookup: dict,
                   scoring_type: str, weakest: list[str]) -> list[dict]:
    """Best free agents at the team's weakest positions, gated by the
    existing waiver quality bar."""
    try:
        from utils.cross_league_actions import waiver_add_clears_quality_bar
    except Exception:
        logger.debug("[front-office] waiver bar import failed", exc_info=True)
        return []
    is_redraft = scoring_type == "redraft"
    is_sf = bool(_ctx_is_sf(ctx))
    n_teams = len(ctx.get("rosters") or []) or 12
    rostered: set[str] = set()
    for r in ctx.get("rosters") or []:
        rostered.update(str(pid) for pid in (r.get("players") or []))
    tbl = ctx.get("model_value_table") or []
    cands = []
    for row in tbl:
        pid = str(row.get("id") or "")
        if not pid or pid in rostered:
            continue
        pos = str(row.get("position") or "").upper()
        if pos not in ("QB", "RB", "WR", "TE"):
            continue
        val = safe_float(row.get("value") or row.get("model_value") or row.get("trade_value"))
        try:
            pos_rank = row.get("pos_rank")
            pos_rank = int(pos_rank) if pos_rank not in (None, "") else None
        except (TypeError, ValueError):
            pos_rank = None
        try:
            age = float(row.get("age") or 0.0)
        except (TypeError, ValueError):
            age = 0.0
        gap = 1.0 if pos in weakest else 0.0
        try:
            ok = waiver_add_clears_quality_bar(
                pos=pos, pos_rank=pos_rank, value=val,
                is_redraft=is_redraft, is_sf=is_sf, n_teams=n_teams,
                age=age, starter_gap=gap,
            )
        except Exception:
            continue
        if not ok:
            continue
        cands.append({
            "id": pid,
            "name": _safe_str(row.get("name")),
            "position": pos,
            "team": _safe_str(row.get("team") or "FA"),
            "age": row.get("age"),
            "value": round(val, 1),
            "pos_rank_label": _safe_str(row.get("pos_rank_label") or ""),
        })
    # Prefer weakest positions first, then value.
    weak_set = set(weakest)
    cands.sort(key=lambda c: (0 if c["position"] in weak_set else 1, -c["value"]))
    return cands[:4]


def _cut_candidates(roster_rows: list[dict]) -> list[dict]:
    skill = [r for r in roster_rows if r["position"] in ("QB", "RB", "WR", "TE")]
    return skill[-3:][::-1] if skill else []


def build_front_office_data(ctx: dict, viewer_roster_id: str) -> dict | None:
    """Assemble every computed input the report needs. Returns None when the
    roster cannot be resolved."""
    team_ctx = build_team_gm_context(_ctx_with_playoff_odds(ctx, block=False), viewer_roster_id)
    if not team_ctx:
        return None
    scoring_type = normalize_trade_scoring_type(team_ctx.get("scoring_type") or ctx_scoring_type(ctx))
    roster = _roster_lookup(ctx, viewer_roster_id)
    if not roster:
        return None
    model_value_lookup = build_model_value_lookup(
        ctx.get("model_value_table") or [],
        is_sf=_ctx_is_sf(ctx),
        scoring_type=scoring_type,
    )
    roster_rows = _build_roster_rows(ctx, roster, model_value_lookup)
    grades = _positional_grades(ctx, viewer_roster_id, model_value_lookup)
    weakest = list(team_ctx.get("weakest_positions") or [])
    if not weakest:
        weakest = [g["pos"] for g in sorted(grades, key=lambda g: g["rank"], reverse=True)[:2]]

    def _movers(rows: list[dict], *, rising: bool, n: int = 3) -> list[dict]:
        cands = [r for r in rows if r.get("trend_7d")]
        cands.sort(key=lambda r: r["trend_7d"], reverse=rising)
        return [
            {"name": r["name"], "position": r["position"], "trend_7d": r["trend_7d"]}
            for r in cands[:n]
        ]

    last_week = _last_week_result(ctx, viewer_roster_id)
    this_week = _this_week_matchup(ctx, viewer_roster_id)
    data = {
        "team_name": team_ctx.get("team_name"),
        "record": team_ctx.get("record"),
        "season": team_ctx.get("season"),
        "week": team_ctx.get("week"),
        "season_phase": team_ctx.get("season_phase"),
        "scoring_type": scoring_type,
        "direction": team_ctx.get("direction"),
        "playoff_pct": team_ctx.get("playoff_pct"),
        "playoff_status": team_ctx.get("playoff_status"),
        "playoff_rank": team_ctx.get("playoff_rank"),
        "points_for": team_ctx.get("points_for"),
        "points_against": team_ctx.get("points_against"),
        "weakest_positions": weakest,
        "grades": grades,
        "roster_rows": roster_rows,
        "risers_7d": _movers(roster_rows, rising=True),
        "fallers_7d": _movers(roster_rows, rising=False),
        "last_week": last_week,
        "this_week": this_week,
        "trade_targets": _trade_targets(ctx, viewer_roster_id, scoring_type),
        "waiver_targets": _waiver_targets(ctx, viewer_roster_id, model_value_lookup, scoring_type, weakest),
        "cut_candidates": _cut_candidates(roster_rows),
        "draft_grade": team_ctx.get("draft_grade"),
    }
    if scoring_type != "redraft":
        data["future_picks"] = team_ctx.get("future_picks") or []
    return data


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def _fallback_ai(data: dict, reason: str = "") -> dict:
    direction = _safe_str(data.get("direction") or "bubble")
    return {
        "verdict": None,
        "headline": f"{data.get('team_name') or 'This team'} profiles as a {direction} team.",
        "posture": "",
        "top_move": "",
        "trade_notes": {},
        "waiver_notes": {},
        "gm_alert": "",
        "_fallback": reason or True,
    }


def get_front_office_report(ctx: dict, viewer_roster_id: str, force_refresh: bool = False) -> dict:
    """Build the full report. Returns {card_html, report_html, verdict, cached}.

    Never blocks on a cold playoff Monte Carlo (see renderer._ctx_with_playoff_odds).
    """
    data = build_front_office_data(ctx, viewer_roster_id)
    if not data:
        return {"card_html": "", "report_html": "", "verdict": None, "cached": False}

    cache_key = build_ai_cache_key("front_office_report", {
        "rid": str(viewer_roster_id),
        "week": data.get("week"),
        "season": data.get("season"),
        "league": _safe_str(ctx.get("league_id")),
        "record": data.get("record"),
        "grades": [(g["pos"], g["grade"], g["rank"]) for g in data.get("grades") or []],
        "top_values": [r["value"] for r in (data.get("roster_rows") or [])[:8]],
    }, CACHE_VERSION)
    if not force_refresh:
        cached = load_cached_ai_text(cache_key)
        if cached:
            try:
                obj = json.loads(cached)
                return {
                    "card_html": _emit_ai_html(obj.get("card_html") or ""),
                    "report_html": _emit_ai_html(obj.get("report_html") or ""),
                    "verdict": obj.get("verdict"),
                    "cached": True,
                }
            except Exception:
                logger.debug("[front-office] bad cache entry, regenerating", exc_info=True)

    if not ai_available():
        ai = _fallback_ai(data, reason="ai_disabled")
        card_html = render_front_office_card_html(data, ai)
        report_html = render_front_office_report_html(data, ai)
    else:
        try:
            ai = generate_front_office_report_result(data)
            card_html = render_front_office_card_html(data, ai)
            report_html = render_front_office_report_html(data, ai)
        except (AIRateLimitError, AIUnavailableError) as e:
            reason = "rate limited" if isinstance(e, AIRateLimitError) else "service unavailable"
            logger.warning("[front-office] %s: %s", reason, e)
            ai = _fallback_ai(data, reason=reason)
            notice = _ai_error_notice(reason)
            card_html = render_front_office_card_html(data, ai)
            report_html = notice + render_front_office_report_html(data, ai)
        except Exception:
            logger.exception("[front-office] unexpected error")
            ai = _fallback_ai(data, reason="error")
            notice = _ai_error_notice()
            card_html = render_front_office_card_html(data, ai)
            report_html = notice + render_front_office_report_html(data, ai)

    save_cached_ai_text(cache_key, json.dumps({
        "card_html": card_html,
        "report_html": report_html,
        "verdict": ai.get("verdict"),
    }, ensure_ascii=False))
    return {
        "card_html": _emit_ai_html(card_html),
        "report_html": _emit_ai_html(report_html),
        "verdict": ai.get("verdict"),
        "cached": False,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Renderers
# ──────────────────────────────────────────────────────────────────────────────

_VERDICT_STYLE = {
    "BUY": ("#1a7f4b", "BUY"),
    "HOLD": ("#8a6d1b", "HOLD"),
    "SELL DEPTH": ("#b3541e", "SELL DEPTH"),
    "PRIORITIZE WAIVERS": ("#0369a1", "PRIORITIZE WAIVERS"),
    "SELL VETERANS": ("#b3541e", "SELL VETERANS"),
    "REBUILD AGGRESSIVELY": ("#a33333", "REBUILD AGGRESSIVELY"),
}


def _verdict_stamp(verdict: str | None, size: str = "md") -> str:
    if not verdict:
        return ""
    color, label = _VERDICT_STYLE.get(str(verdict).upper(), ("#334155", str(verdict).upper()))
    cls = "for-stamp for-stamp-sm" if size == "sm" else "for-stamp"
    return (
        f"<div class='{cls}' style='border-color:{color};color:{color};'>"
        f"<span>{html.escape(label)}</span></div>"
    )


def _fmt_trend(trend) -> str:
    if trend is None:
        return ""
    try:
        t = float(trend)
    except (TypeError, ValueError):
        return ""
    if t > 0:
        return f"<span class='for-trend-up'>▲{t:g}</span>"
    if t < 0:
        return f"<span class='for-trend-down'>▼{abs(t):g}</span>"
    return ""


def render_front_office_card_html(data: dict, ai: dict) -> str:
    """Condensed summary for the Season Hub card."""
    verdict = ai.get("verdict")
    headline = html.escape(str(ai.get("headline") or ""))
    top_move = html.escape(str(ai.get("top_move") or ""))
    week = data.get("week")
    week_lbl = f"Week {week}" if week else ""
    team = html.escape(str(data.get("team_name") or ""))

    keys: list[str] = []
    pct = data.get("playoff_pct")
    if pct is not None:
        try:
            keys.append(("Playoff odds", f"{float(pct):.0f}%"))
        except (TypeError, ValueError):
            pass
    grades = data.get("grades") or []
    if grades:
        worst = max(grades, key=lambda g: g["rank"])
        keys.append(("Weakest room", f"{worst['pos']} ranks {worst['rank']} of {worst['of']}"))
    lw = data.get("last_week") or {}
    if lw.get("result") and lw.get("pf") is not None:
        opp = f" vs {html.escape(lw['opponent'])}" if lw.get("opponent") else ""
        keys.append((f"Week {lw.get('week')} result", f"{lw['result']} {lw['pf']}-{lw['pa'] or 0}{opp}"))
    keys = keys[:3]
    keys_html = "".join(
        f"<li><span class='for-key-k'>{html.escape(k)}</span>"
        f"<span class='for-key-v'>{v}</span></li>"
        for k, v in keys
    )

    move_html = ""
    if top_move:
        move_html = (
            "<div class='for-card-move'>"
            "<span class='for-lbl'>Top move</span>"
            f"<div>{top_move}</div></div>"
        )
    stamp = _verdict_stamp(verdict, size="sm")
    verdict_block = f"<div class='for-card-verdict'>{stamp}<div class='for-card-headline'>{headline}</div></div>" if (stamp or headline) else ""
    foot = f"{html.escape(team)}" if team else ""
    if week_lbl:
        foot = f"{foot} · {week_lbl}" if foot else week_lbl
    return f"""
    <div class='for-card-summary'>
      {verdict_block}
      <ul class='for-keys'>{keys_html}</ul>
      {move_html}
      <button type='button' class='for-view-full' id='forViewFullBtn'>View full report →</button>
      <div class='for-card-foot'>{foot}</div>
    </div>
    """


def _grades_html(grades: list[dict]) -> str:
    cards = []
    for g in grades:
        pct = 100.0 * (g["of"] - g["rank"]) / max(g["of"] - 1, 1) if g["of"] > 1 else 50.0
        cards.append(
            f"<div class='for-grade-card'>"
            f"<div class='for-grade-pos'>{html.escape(g['pos'])}</div>"
            f"<div class='for-grade-letter for-grade-{html.escape(g['grade'].lower())}'>{html.escape(g['grade'])}</div>"
            f"<div class='for-grade-rank'>{g['rank']} of {g['of']}</div>"
            f"<div class='for-grade-bar'><span style='width:{pct:.0f}%'></span></div>"
            f"</div>"
        )
    return f"<div class='for-grades'>{''.join(cards)}</div>"


def _roster_table_html(rows: list[dict]) -> str:
    body = []
    for r in rows:
        age = "" if r.get("age") in (None, "") else f"{float(r['age']):.1f}".rstrip("0").rstrip(".")
        inj = f" <span class='for-inj'>{html.escape(r['injury'])}</span>" if r.get("injury") else ""
        body.append(
            "<tr>"
            f"<td class='for-td-name'>{html.escape(r['name'])}{inj}</td>"
            f"<td>{html.escape(r['position'])}</td>"
            f"<td>{html.escape(r['team'])}</td>"
            f"<td class='for-td-num'>{age}</td>"
            f"<td class='for-td-num'>{r['value']:g}</td>"
            f"<td>{html.escape(r['pos_rank_label'])}</td>"
            f"<td><span class='for-role for-role-{r['role'].lower()}'>{r['role']}</span></td>"
            f"<td class='for-td-num'>{_fmt_trend(r.get('trend_7d'))}</td>"
            "</tr>"
        )
    return (
        "<div class='for-table-wrap'><table class='for-table'>"
        "<thead><tr><th>Player</th><th>Pos</th><th>Team</th><th>Age</th>"
        "<th>Value</th><th>Pos rank</th><th>Role</th><th>7d</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _changes_html(data: dict) -> str:
    items = []
    lw = data.get("last_week") or {}
    if lw.get("result"):
        opp = f" vs {html.escape(lw['opponent'])}" if lw.get("opponent") else ""
        pa = lw.get("pa")
        score = f"{lw['pf']}-{pa}" if pa is not None else f"{lw['pf']}"
        items.append(f"<li><strong>Week {lw['week']}:</strong> {lw['result']} {score}{opp}</li>")
    for m in data.get("risers_7d") or []:
        items.append(
            f"<li><strong>{html.escape(m['name'])}</strong> "
            f"({html.escape(m['position'])}) up {m['trend_7d']:g} spots this week</li>"
        )
    for m in data.get("fallers_7d") or []:
        items.append(
            f"<li><strong>{html.escape(m['name'])}</strong> "
            f"({html.escape(m['position'])}) down {abs(m['trend_7d']):g} spots this week</li>"
        )
    if not items:
        return ""
    return (
        "<div class='for-sec'><div class='for-sec-title'>Since last week</div>"
        f"<ul class='for-list'>{''.join(items)}</ul></div>"
    )


def _trade_targets_html(targets: list[dict], trade_notes: dict) -> str:
    if not targets:
        return ""
    cards = []
    for t in targets:
        get = t["gets"][0]
        give_names = ", ".join(
            f"{html.escape(g['name'])} ({html.escape(g['position'])})" for g in t["gives"]
        )
        note = html.escape(str(trade_notes.get(get["id"]) or ""))
        note_html = f"<div class='for-target-note'>{note}</div>" if note else ""
        cards.append(
            "<div class='for-target-card'>"
            f"<div class='for-target-head'><strong>{html.escape(get['name'])}</strong> "
            f"<span class='for-target-meta'>{html.escape(get['position'])}"
            + (f" · age {get['age']}" if get.get("age") not in (None, "") else "")
            + f" · value {get['value']:g}</span></div>"
            f"<div class='for-target-meta'>From {html.escape(t['partner'])}</div>"
            f"<div class='for-target-give'>You give: {give_names}</div>"
            f"{note_html}"
            f"<a class='for-analyze' href='{html.escape(t['analyzer_url'], quote=True)}'>Analyze this trade →</a>"
            "</div>"
        )
    return (
        "<div class='for-sec'><div class='for-sec-title'>Trade targets</div>"
        f"<div class='for-targets'>{''.join(cards)}</div></div>"
    )


def _waivers_cuts_html(data: dict, ai: dict) -> str:
    wnotes = ai.get("waiver_notes") or {}
    w_items = []
    for w in data.get("waiver_targets") or []:
        note = html.escape(str(wnotes.get(w["id"]) or ""))
        rank = f" · {html.escape(w['pos_rank_label'])}" if w.get("pos_rank_label") else ""
        w_items.append(
            f"<li><strong>{html.escape(w['name'])}</strong> "
            f"({html.escape(w['position'])}, {html.escape(w['team'])}{rank})"
            + (f" <span class='for-note'>{note}</span>" if note else "")
            + "</li>"
        )
    c_items = []
    for c in data.get("cut_candidates") or []:
        c_items.append(
            f"<li><strong>{html.escape(c['name'])}</strong> "
            f"({html.escape(c['position'])}, value {c['value']:g})</li>"
        )
    if not w_items and not c_items:
        return ""
    out = "<div class='for-sec'><div class='for-sec-title'>Waivers and cuts</div><div class='for-two-col'>"
    if w_items:
        out += f"<div><div class='for-sub'>Add</div><ul class='for-list'>{''.join(w_items)}</ul></div>"
    if c_items:
        out += f"<div><div class='for-sub'>Cut candidates</div><ul class='for-list'>{''.join(c_items)}</ul></div>"
    return out + "</div></div>"


def render_front_office_report_html(data: dict, ai: dict) -> str:
    """Full modal report."""
    verdict = ai.get("verdict")
    headline = html.escape(str(ai.get("headline") or ""))
    posture = html.escape(str(ai.get("posture") or ""))
    gm_alert = html.escape(str(ai.get("gm_alert") or ""))
    team = html.escape(str(data.get("team_name") or "Front Office Report"))
    week = data.get("week")
    week_lbl = f" · Week {week}" if week else ""
    rec = data.get("record")
    rec_lbl = f" · {html.escape(rec)}" if rec else ""
    pct = data.get("playoff_pct")
    odds_lbl = ""
    if pct is not None:
        try:
            odds_lbl = f" · {float(pct):.0f}% playoff odds"
        except (TypeError, ValueError):
            pass

    grades_html = _grades_html(data.get("grades") or [])
    changes_html = _changes_html(data)
    targets_html = _trade_targets_html(data.get("trade_targets") or [], ai.get("trade_notes") or {})
    waivers_html = _waivers_cuts_html(data, ai)
    roster_html = _roster_table_html(data.get("roster_rows") or [])

    alert_html = ""
    if gm_alert:
        alert_html = (
            "<div class='for-sec'><div class='for-sec-title'>GM alert</div>"
            f"<p class='for-alert'>{gm_alert}</p></div>"
        )
    posture_html = f"<p class='for-posture'>{posture}</p>" if posture else ""
    stamp = _verdict_stamp(verdict)

    return f"""
    <div class='for-report'>
      <div class='for-report-head'>
        {stamp}
        <div>
          <div class='for-report-team'>{team}</div>
          <div class='for-report-meta'>Front Office Report{week_lbl}{rec_lbl}{odds_lbl}</div>
        </div>
      </div>
      <h3 class='for-headline'>{headline}</h3>
      {posture_html}
      {changes_html}
      <div class='for-sec'><div class='for-sec-title'>Positional grades</div>{grades_html}</div>
      <div class='for-sec'><div class='for-sec-title'>Roster</div>{roster_html}</div>
      {targets_html}
      {waivers_html}
      {alert_html}
    </div>
    """
