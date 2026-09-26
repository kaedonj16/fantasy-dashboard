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
    AI_CACHE_FALLBACK_TTL,
    build_ai_cache_key,
    load_cached_ai_text,
    save_cached_ai_text,
)
from dashboard_services.ai.client import AIRateLimitError, AIUnavailableError
from dashboard_services.ai.context_builders import (
    _ctx_is_sf,
    _record_pf_pa_for_roster,
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
from utils.roster_strength import STARTER_THRESHOLD

logger = logging.getLogger(__name__)

CACHE_VERSION = "v2"
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
    if pct >= 80:
        return "A"
    if pct >= 60:
        return "B"
    if pct >= 40:
        return "C"
    if pct >= 20:
        return "D"
    return "F"


def _positional_grades(ctx: dict, viewer_roster_id: str, model_value_lookup: dict) -> list[dict]:
    """League-relative positional grades.

    Ranks come from rank_rosters_by_position, the same starter-weighted
    ranking the Teams page shows, so the two surfaces always agree. The
    grade is a coarse letter off the rank percentile (2nd of 10 is an A).
    """
    from utils.roster_strength import rank_rosters_by_position
    from utils.utils import count_roster_positions

    rosters = ctx.get("rosters") or []
    n_teams = len(rosters)
    team_pos_values: dict[str, dict[str, list[float]]] = {}
    for r in rosters:
        rid = str(r.get("roster_id"))
        buckets: dict[str, list[float]] = {pos: [] for pos in _SKILL_POS}
        for pid in r.get("players") or []:
            mv = model_value_lookup.get(str(pid)) or {}
            pos = str(mv.get("position") or "").upper()
            if pos not in buckets:
                continue
            val = safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value"))
            if val <= 0:
                continue
            buckets[pos].append(val)
        team_pos_values[rid] = buckets
    slot_counts = count_roster_positions(ctx.get("roster_positions") or [])
    strengths, ranks = rank_rosters_by_position(
        team_pos_values, slot_counts, positions=list(_SKILL_POS)
    )
    viewer = str(viewer_roster_id)
    grades = []
    for pos in _SKILL_POS:
        rank = (ranks.get(pos) or {}).get(viewer, n_teams)
        pct = 100.0 * (n_teams - rank) / max(n_teams - 1, 1) if n_teams > 1 else 50.0
        grades.append({
            "pos": pos,
            "grade": _grade_for_percentile(pct),
            "rank": rank,
            "of": n_teams,
            "score": round(float((strengths.get(viewer) or {}).get(pos) or 0.0), 1),
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
        # Partner acceptance context: why they'd say yes. The suggestion
        # engine already pays from bench surplus at positions the partner
        # needs, so the overlap below is usually non-empty; the computed
        # line keeps the AI's note honest about the partner's motivation.
        rid = str(p.get("roster_id") or "")
        partner_record = ""
        try:
            partner_record = _record_pf_pa_for_roster(ctx, rid)[0] or ""
        except Exception:
            logger.debug("[front-office] partner record failed", exc_info=True)
        partner_needs = [str(n).upper() for n in (p.get("partner_needs") or [])]
        give_positions = [
            str(g.get("position") or "").upper() for g in gives
        ]
        need_overlap = [pos for pos in give_positions if pos in partner_needs]
        value_get = safe_float(p.get("value_you_get"))
        value_give = safe_float(p.get("value_you_give"))
        if need_overlap:
            why_they_say_yes = (
                f"Fills their {'/'.join(dict.fromkeys(need_overlap))} need."
            )
        elif value_give > value_get:
            why_they_say_yes = "They win the value math."
        else:
            why_they_say_yes = "Straight value swap at a spot they can use."
        targets.append({
            "partner": _safe_str(p.get("team_name") or ""),
            "partner_record": partner_record,
            "partner_needs": partner_needs,
            "why_they_say_yes": why_they_say_yes,
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


def _trade_deadline_info(ctx: dict) -> dict | None:
    """Real trade-deadline countdown from league settings.

    Sleeper keeps ``trade_deadline`` (week number) on league.settings; ESPN
    maps tradeSettings.deadlineDate onto ``trade_deadline_ts``. Returns
    {"deadline_week", "weeks_remaining"} or None when unknown/passed.
    """
    settings: dict = {}
    try:
        settings.update(ctx.get("league_settings") or {})
        settings.update((ctx.get("league") or {}).get("settings") or {})
    except Exception:
        return None
    try:
        current_week = int(ctx.get("current_week") or ctx.get("week") or 0)
    except (TypeError, ValueError):
        return None
    try:
        deadline = int(settings.get("trade_deadline") or 0)
    except (TypeError, ValueError):
        deadline = 0
    if 0 < deadline < 30:
        if current_week > deadline:
            return None  # deadline passed; the window is closed
        return {"deadline_week": deadline, "weeks_remaining": deadline - current_week}
    try:
        deadline_ts = int(settings.get("trade_deadline_ts") or 0)
    except (TypeError, ValueError):
        deadline_ts = 0
    if deadline_ts > 0:
        import time as _time

        remaining = deadline_ts - _time.time()
        if remaining < 0:
            return None
        weeks = int(remaining // (7 * 86400))
        return {
            "deadline_week": max(1, current_week + weeks),
            "weeks_remaining": weeks,
        }
    return None


def _urgent_needs(ctx: dict, roster: dict, roster_rows: list[dict],
                  week: int | None) -> list[dict]:
    """Starters who won't be available soon: serious injury, or a bye in the
    next two weeks. These holes jump the queue for trade/waiver priority."""
    try:
        from utils.lineup_issues import SERIOUS_INJURY_STATUSES
    except Exception:
        SERIOUS_INJURY_STATUSES = set()
    bye_by_team: dict[str, int] = {}
    try:
        from utils.utils import path_teams_index, read_json_cached

        teams = read_json_cached(path_teams_index()) or {}
        for abv, meta in teams.items():
            if not isinstance(meta, dict) or meta.get("byeWeek") is None:
                continue
            try:
                bye_by_team[str(abv).strip().upper()] = int(meta["byeWeek"])
            except (TypeError, ValueError):
                continue
    except Exception:
        logger.debug("[front-office] bye lookup failed", exc_info=True)
    rows_by_id = {str(r.get("id")): r for r in roster_rows}
    urgent: list[dict] = []
    for pid in roster.get("starters") or []:
        row = rows_by_id.get(str(pid))
        if not row:
            continue
        injury = str(row.get("injury") or "").strip().upper()
        if injury and injury in SERIOUS_INJURY_STATUSES:
            urgent.append({
                "position": row.get("position"),
                "player": row.get("name"),
                "reason": "injury",
                "detail": f"{row.get('name')} is {injury}",
            })
            continue
        bye = bye_by_team.get(str(row.get("team") or "").strip().upper())
        if bye and week and bye in (week + 1, week + 2):
            urgent.append({
                "position": row.get("position"),
                "player": row.get("name"),
                "reason": "bye",
                "detail": f"{row.get('name')} on bye week {bye}",
            })
    return urgent


def _apply_urgency(trade_targets: list[dict], waiver_targets: list[dict],
                   urgent_needs: list[dict]) -> None:
    """Flag + bubble up targets that fill an urgent (bye/injury) hole."""
    urgent_by_pos: dict[str, str] = {}
    for u in urgent_needs:
        pos = str(u.get("position") or "").upper()
        if pos and pos not in urgent_by_pos:
            urgent_by_pos[pos] = str(u.get("detail") or "")
    if not urgent_by_pos:
        return
    for t in trade_targets:
        positions = {
            str(g.get("position") or "").upper() for g in (t.get("gets") or [])
        }
        hit = next((p for p in positions if p in urgent_by_pos), "")
        if hit:
            t["urgent"] = True
            t["urgent_reason"] = urgent_by_pos[hit]
    trade_targets.sort(key=lambda t: (0 if t.get("urgent") else 1))
    for w in waiver_targets:
        pos = str(w.get("position") or "").upper()
        if pos in urgent_by_pos:
            w["urgent"] = True
            w["urgent_reason"] = urgent_by_pos[pos]
    waiver_targets.sort(key=lambda w: (0 if w.get("urgent") else 1))


def _annotate_waiver_alternatives(trade_targets: list[dict],
                                 waiver_targets: list[dict]) -> None:
    """Don't trade for what the wire can fix: if a free agent at the same
    position is worth at least half the trade target, flag it so the note
    can say 'add X for free instead'."""
    for t in trade_targets:
        gets = t.get("gets") or []
        if not gets:
            continue
        primary = gets[0]
        pos = str(primary.get("position") or "").upper()
        val = safe_float(primary.get("value"))
        if val <= 0:
            continue
        alt = next(
            (
                w for w in waiver_targets
                if str(w.get("position") or "").upper() == pos
                and safe_float(w.get("value")) >= 0.5 * val
            ),
            None,
        )
        if alt:
            t["waiver_alternative"] = {
                "name": _safe_str(alt.get("name")),
                "value": alt.get("value"),
            }


def _drop_add_pairs(cut_candidates: list[dict],
                    waiver_targets: list[dict]) -> list[dict]:
    """Pair each waiver add with its cleanest drop: same-position cut first,
    else the cheapest remaining cut. This is the actual move, not two lists."""
    pairs: list[dict] = []
    used: set[str] = set()
    for w in waiver_targets or []:
        cands = [c for c in cut_candidates if str(c.get("id")) not in used]
        if not cands:
            break
        same_pos = [
            c for c in cands
            if str(c.get("position") or "").upper()
            == str(w.get("position") or "").upper()
        ]
        pool = same_pos or cands
        drop = min(pool, key=lambda c: safe_float(c.get("value")))
        used.add(str(drop.get("id")))
        pairs.append({
            "drop": {
                "name": _safe_str(drop.get("name")),
                "position": _safe_str(drop.get("position")),
                "value": drop.get("value"),
            },
            "add": {
                "name": _safe_str(w.get("name")),
                "position": _safe_str(w.get("position")),
                "value": w.get("value"),
            },
        })
    return pairs


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
    week = team_ctx.get("week")
    trade_targets = _trade_targets(ctx, viewer_roster_id, scoring_type)
    waiver_targets = _waiver_targets(ctx, viewer_roster_id, model_value_lookup,
                                     scoring_type, weakest)
    cut_candidates = _cut_candidates(roster_rows)
    urgent_needs = _urgent_needs(ctx, roster, roster_rows, week)
    _apply_urgency(trade_targets, waiver_targets, urgent_needs)
    _annotate_waiver_alternatives(trade_targets, waiver_targets)
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
        "trade_targets": trade_targets,
        "waiver_targets": waiver_targets,
        "cut_candidates": cut_candidates,
        "drop_add_pairs": _drop_add_pairs(cut_candidates, waiver_targets),
        "urgent_needs": urgent_needs,
        "trade_deadline": _trade_deadline_info(ctx),
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

    cache_ttl = None
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
            # Transient failure: cache briefly so the next view retries the AI
            # call instead of serving the "unavailable" notice for 12 hours.
            cache_ttl = AI_CACHE_FALLBACK_TTL
        except Exception:
            logger.exception("[front-office] unexpected error")
            ai = _fallback_ai(data, reason="error")
            notice = _ai_error_notice()
            card_html = render_front_office_card_html(data, ai)
            report_html = notice + render_front_office_report_html(data, ai)
            cache_ttl = AI_CACHE_FALLBACK_TTL

    save_cached_ai_text(cache_key, json.dumps({
        "card_html": card_html,
        "report_html": report_html,
        "verdict": ai.get("verdict"),
    }, ensure_ascii=False), ttl=cache_ttl)
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
        letter = str(g["grade"]).lower()
        cards.append(
            f"<div class='for-grade-card for-grade-{html.escape(letter)}'>"
            f"<div class='for-grade-pos'>{html.escape(g['pos'])}</div>"
            f"<div class='for-grade-letter'>{html.escape(g['grade'])}</div>"
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
        "<th>Value</th><th>Pos rank</th><th>Role</th><th>Trend</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _changes_html(data: dict) -> str:
    moves = []
    lw = data.get("last_week") or {}
    score_html = ""
    if lw.get("result"):
        opp = f" vs {html.escape(lw['opponent'])}" if lw.get("opponent") else ""
        pa = lw.get("pa")
        score = f"{lw['pf']}-{pa}" if pa is not None else f"{lw['pf']}"
        badge = "for-score-w" if str(lw["result"]).upper().startswith("W") else "for-score-l"
        score_html = (
            "<div class='for-score-row'>"
            f"<span class='{badge}'>{html.escape(str(lw['result']))}</span>"
            f"<span><strong>Week {lw['week']}</strong> {html.escape(str(score))}"
            f"<span class='for-muted'>{opp}</span></span>"
            "</div>"
        )
    for m in data.get("risers_7d") or []:
        moves.append(_move_row(m, up=True))
    for m in data.get("fallers_7d") or []:
        moves.append(_move_row(m, up=False))
    if not score_html and not moves:
        return ""
    moves_html = f"<ul class='for-moves'>{''.join(moves)}</ul>" if moves else ""
    return (
        "<div class='for-sec'><div class='for-sec-title'>Since last week</div>"
        f"{score_html}{moves_html}"
        "<div class='for-sec-note'>Spots gained or lost in dynasty trade-value rank vs 7 days ago.</div>"
        "</div>"
    )


def _move_row(m: dict, up: bool) -> str:
    delta = abs(m["trend_7d"]) if m.get("trend_7d") is not None else 0
    cls = "for-up" if up else "for-down"
    arrow = "&#9650;" if up else "&#9660;"
    return (
        "<li class='for-move'>"
        f"<span class='for-move-delta {cls}'>{arrow} {delta:g}</span>"
        f"<span class='for-move-body'><strong>{html.escape(m['name'])}</strong>"
        f"<span class='for-muted'> {html.escape(m['position'])}</span></span>"
        "</li>"
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
        why = t.get("why_they_say_yes") or ""
        partner_rec = t.get("partner_record") or ""
        why_html = ""
        if why:
            why_html = (
                "<div class='for-target-why'><span class='for-lbl-inline'>Why they'd say yes</span> "
                f"{html.escape(why)}"
                + (f" <span class='for-muted'>({html.escape(partner_rec)})</span>" if partner_rec else "")
                + "</div>"
            )
        urgent_html = ""
        if t.get("urgent"):
            urgent_html = (
                "<div class='for-target-urgent'><span class='for-lbl-inline'>Urgent</span> "
                f"{html.escape(str(t.get('urgent_reason') or ''))}</div>"
            )
        alt = t.get("waiver_alternative") or {}
        alt_html = ""
        if alt.get("name"):
            alt_html = (
                "<div class='for-target-alt'><span class='for-lbl-inline'>Free alternative</span> "
                f"{html.escape(str(alt['name']))} is on waivers.</div>"
            )
        cards.append(
            "<div class='for-target-card'>"
            f"<div class='for-target-top'><div class='for-target-head'><strong>{html.escape(get['name'])}</strong> "
            f"<span class='for-muted'>{html.escape(get['position'])}"
            + (f" · age {get['age']}" if get.get("age") not in (None, "") else "")
            + f" · value {get['value']:g}</span></div>"
            f"<span class='for-target-from'>From {html.escape(t['partner'])}</span></div>"
            f"<div class='for-target-give'><span class='for-lbl-inline'>You give</span> {give_names}</div>"
            f"{urgent_html}"
            f"{why_html}"
            f"{alt_html}"
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
        note_html = f"<div class='for-pick-note'>{note}</div>" if note else ""
        urgent_html = ""
        if w.get("urgent"):
            urgent_html = (
                f" <span class='for-lbl-inline'>Urgent:</span> "
                f"<span class='for-muted'>{html.escape(str(w.get('urgent_reason') or ''))}</span>"
            )
        w_items.append(
            "<li class='for-pick'>"
            "<span class='for-pick-badge for-add'>+</span>"
            f"<div class='for-pick-body'><strong>{html.escape(w['name'])}</strong> "
            f"<span class='for-muted'>{html.escape(w['position'])}, {html.escape(w['team'])}{rank}</span>"
            f"{urgent_html}{note_html}</div></li>"
        )
    c_items = []
    for c in data.get("cut_candidates") or []:
        c_items.append(
            "<li class='for-pick'>"
            "<span class='for-pick-badge for-cut'>&minus;</span>"
            f"<div class='for-pick-body'><strong>{html.escape(c['name'])}</strong> "
            f"<span class='for-muted'>{html.escape(c['position'])} · value {c['value']:g}</span></div></li>"
        )
    pair_items = []
    for p in data.get("drop_add_pairs") or []:
        drop = p.get("drop") or {}
        add = p.get("add") or {}
        pair_items.append(
            "<li class='for-pick'>"
            f"<div class='for-pick-body'><strong>Drop {html.escape(str(drop.get('name')))}</strong> "
            f"<span class='for-muted'>({html.escape(str(drop.get('position')))})</span>"
            f" → <strong>Add {html.escape(str(add.get('name')))}</strong> "
            f"<span class='for-muted'>({html.escape(str(add.get('position')))})</span></div></li>"
        )
    if not w_items and not c_items and not pair_items:
        return ""
    out = "<div class='for-sec'><div class='for-sec-title'>Waivers and cuts</div><div class='for-two-col'>"
    if w_items:
        out += f"<div><div class='for-sub'>Add</div><ul class='for-picklist'>{''.join(w_items)}</ul></div>"
    if c_items:
        out += f"<div><div class='for-sub'>Cut candidates</div><ul class='for-picklist'>{''.join(c_items)}</ul></div>"
    out += "</div>"
    if pair_items:
        out += (
            "<div class='for-sub'>Paired moves</div>"
            f"<ul class='for-picklist'>{''.join(pair_items)}</ul>"
        )
    return out + "</div>"


def render_front_office_report_html(data: dict, ai: dict) -> str:
    """Full modal report."""
    verdict = ai.get("verdict")
    headline = html.escape(str(ai.get("headline") or ""))
    posture = html.escape(str(ai.get("posture") or ""))
    gm_alert = html.escape(str(ai.get("gm_alert") or ""))
    team = html.escape(str(data.get("team_name") or "Front Office Report"))
    week = data.get("week")
    rec = data.get("record")
    pct = data.get("playoff_pct")

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
    chips_html = _hero_chips_html(week, rec, pct, data.get("trade_deadline"))

    return f"""
    <div class='for-report'>
      <div class='for-hero'>
        {stamp}
        <div class='for-hero-main'>
          <div class='for-hero-team'>{team}</div>
          {chips_html}
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


def _hero_chips_html(week, rec, pct, trade_deadline=None) -> str:
    """Small stat pills under the team name: week, record, playoff odds,
    trade deadline countdown."""
    chips = []
    if week:
        chips.append(f"<span class='for-chip'>Week {html.escape(str(week))}</span>")
    if rec:
        chips.append(f"<span class='for-chip'>{html.escape(str(rec))}</span>")
    if pct is not None:
        try:
            chips.append(
                f"<span class='for-chip for-chip-hot'>{float(pct):.0f}% playoff odds</span>"
            )
        except (TypeError, ValueError):
            pass
    if isinstance(trade_deadline, dict):
        try:
            weeks_left = int(trade_deadline.get("weeks_remaining"))
        except (TypeError, ValueError):
            weeks_left = None
        if weeks_left is not None and weeks_left >= 0:
            label = (
                "Trade deadline: this week" if weeks_left == 0
                else f"Trade deadline: {weeks_left} wk" if weeks_left == 1
                else f"Trade deadline: {weeks_left} wks"
            )
            hot = " for-chip-hot" if weeks_left <= 3 else ""
            chips.append(f"<span class='for-chip{hot}'>{html.escape(label)}</span>")
    if not chips:
        return ""
    return f"<div class='for-hero-chips'>{''.join(chips)}</div>"
