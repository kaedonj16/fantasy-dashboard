"""Front Office Report v2: structured, data-grounded team report.

Replaces the prose-only GM memo on the Season Hub. The report is two parts:

- Card summary: verdict stamp, headline, key numbers, top move, and a
  "View full report" button. Rendered into the existing Season Hub card.
- Full report: opened as a modal. Verdict, top move, since-last-week changes,
  roster table, computed positional grades, trade targets with prefilled
  Trade Analyzer links, waiver targets, cut candidates, and a GM alert.

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

CACHE_VERSION = "v3"
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


def _potential_trade_targets(ctx: dict, viewer_roster_id: str,
                             model_value_lookup: dict,
                             positions: list[str],
                             limit: int = 6, per_position: int = 2) -> list[dict]:
    """Players worth pursuing at the viewer's need positions, rostered by
    leaguemates. Used when the suggestions engine cannot construct a fair
    package: the report still names who to go get instead of silently
    dropping the trade section. No give side is invented here; these are
    targets, not priced deals."""
    priority: list[str] = []
    for pos in positions or []:
        p = str(pos or "").upper()
        if p in ("QB", "RB", "WR", "TE") and p not in priority:
            priority.append(p)
    if not priority:
        return []
    prio = {pos: i for i, pos in enumerate(priority)}
    players_index = ctx.get("players_index") or {}
    players_map = ctx.get("players_map") or {}
    roster_map = ctx.get("roster_map") or {}
    viewer_rid = str(viewer_roster_id)
    cands: list[dict] = []
    for r in ctx.get("rosters") or []:
        rid = str(r.get("roster_id") or "")
        if not rid or rid == viewer_rid:
            continue
        partner = _safe_str(roster_map.get(rid) or r.get("team_name") or f"Team {rid}")
        for pid in r.get("players") or []:
            spid = str(pid)
            mv = model_value_lookup.get(spid) or {}
            meta = players_index.get(spid) or players_map.get(spid) or {}
            pos = str(meta.get("position") or meta.get("pos") or mv.get("position") or "").upper()
            if pos not in prio:
                continue
            value = round(safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value")), 1)
            if value <= 0:
                continue
            cands.append({
                "id": spid,
                "name": _safe_str(meta.get("full_name") or meta.get("name") or mv.get("name") or spid),
                "position": pos,
                "team": _safe_str(meta.get("team") or mv.get("team") or "FA"),
                "age": meta.get("age") if meta.get("age") not in (None, "") else mv.get("age"),
                "value": value,
                "partner": partner,
            })
    cands.sort(key=lambda c: (prio[c["position"]], -c["value"]))
    out: list[dict] = []
    per_pos_count: dict[str, int] = {}
    for c in cands:
        if per_pos_count.get(c["position"], 0) >= per_position:
            continue
        per_pos_count[c["position"]] = per_pos_count.get(c["position"], 0) + 1
        out.append(c)
        if len(out) >= limit:
            break
    return out


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
    """Starters who won't be available soon: serious injury, a QUESTIONABLE
    tag (game-time call), or a bye in the next two weeks. Bench players with
    serious injuries are included too, since "move to IR" is the action.
    These holes jump the queue for trade/waiver priority."""
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
    starter_ids = {str(pid) for pid in roster.get("starters") or []}

    def _injury_entry(row: dict, on_bench: bool) -> dict | None:
        injury = str(row.get("injury") or "").strip().upper()
        if not injury:
            return None
        canon = _inj_canonical(injury)
        # Non-injury statuses (e.g. a healthy game-day "Inactive") never
        # count as injuries.
        if not _inj_is_reportable(canon):
            return None
        if canon in SERIOUS_INJURY_STATUSES:
            detail = f"{row.get('name')} is {injury}"
            if on_bench:
                detail = f"{row.get('name')} ({row.get('position')}, bench) is {injury}"
            return {
                "position": row.get("position"),
                "player": row.get("name"),
                "reason": "injury",
                "detail": detail,
            }
        if canon == "QUESTIONABLE" and not on_bench:
            return {
                "position": row.get("position"),
                "player": row.get("name"),
                "reason": "injury",
                "detail": f"{row.get('name')} is QUESTIONABLE, game-time call",
            }
        return None

    for pid in roster.get("starters") or []:
        row = rows_by_id.get(str(pid))
        if not row:
            continue
        entry = _injury_entry(row, on_bench=False)
        if entry:
            urgent.append(entry)
            continue
        bye = bye_by_team.get(str(row.get("team") or "").strip().upper())
        if bye and week and bye in (week + 1, week + 2):
            urgent.append({
                "position": row.get("position"),
                "player": row.get("name"),
                "reason": "bye",
                "detail": f"{row.get('name')} on bye week {bye}",
            })
    for pid in roster.get("players") or []:
        if str(pid) in starter_ids:
            continue
        row = rows_by_id.get(str(pid))
        if not row:
            continue
        entry = _injury_entry(row, on_bench=True)
        if entry:
            urgent.append(entry)
    return urgent


# ── Injury helpers ──────────────────────────────────────────────────────────
# Severity order for sorting: IR-like > OUT > DOUBTFUL > QUESTIONABLE.
_INJ_SEVERITY = {
    "IR": 0, "PUP": 0, "NFI": 0, "SUSP": 0, "SUS": 0,
    "OUT": 1, "O": 1,
    "DOUBTFUL": 2, "D": 2,
    "QUESTIONABLE": 3, "Q": 3,
}


def _inj_canonical(status: str) -> str:
    """Single-letter Sleeper codes -> canonical designation.

    Self-contained (no module globals) so AST-extracted unit tests can exec
    it standalone.
    """
    s = str(status or "").strip().upper()
    return {"O": "OUT", "D": "DOUBTFUL", "Q": "QUESTIONABLE"}.get(s, s)


# Canonical injury designations for the report. Mirrors
# dashboard_services.injuries.INJURY_STATUSES (that module imports pandas, so
# the tuple is inlined here and in _inj_is_reportable to keep this module
# importable without pandas and exec-able by AST-extracted unit tests).
_INJURY_DESIGNATIONS = (
    "IR", "PUP", "NFI", "SUSP", "SUS", "OUT", "DOUBTFUL", "QUESTIONABLE",
)


def _inj_is_reportable(designation: str, body: str = "") -> bool:
    """True when a designation/body pair belongs on the Injury Report.

    Guards against non-injury statuses (e.g. a healthy game-day "Inactive")
    being treated as injuries. Self-contained so AST-extracted unit tests
    can exec it standalone.
    """
    if _inj_canonical(designation) in (
        "IR", "PUP", "NFI", "SUSP", "SUS", "OUT", "DOUBTFUL", "QUESTIONABLE",
    ):
        return True
    return bool(str(body or "").strip())


def _inj_severity(status: str) -> int:
    return _INJ_SEVERITY.get(str(status or "").strip().upper(), 9)


def _fmt_weeks_out(weeks_out) -> str:
    """'~3 wks', '~1 wk', 'this week', or '' when unknown."""
    try:
        w = float(weeks_out)
    except (TypeError, ValueError):
        return ""
    if w <= 0.5:
        return "this week"
    return f"~{w:g} {'wk' if w == 1 else 'wks'}"


def _injury_lookup(ctx: dict) -> dict[str, str]:
    """player_id -> raw injury designation, from the Sleeper players index.

    players_full first, then the lighter indexes; first non-empty wins.
    """
    lookup: dict[str, str] = {}
    for src in (ctx.get("players") or {}, ctx.get("players_index") or {},
                ctx.get("players_map") or {}):
        if not isinstance(src, dict):
            continue
        for pid, info in src.items():
            if not isinstance(info, dict):
                continue
            spid = str(pid)
            if spid in lookup:
                continue
            raw = str(info.get("injury_status") or info.get("status") or "").strip().upper()
            if raw and raw not in ("", "ACTIVE", "ACT"):
                lookup[spid] = raw
    return lookup


def _injury_df_map(ctx: dict) -> dict[str, dict] | None:
    """player_id -> {designation, body} from the league context's injury_df.

    injury_df is built by build_injury_report() during league-context
    construction from the same players snapshot the roster rows use, so it is
    the canonical injury source. Duck-typed (no pandas import) so this module
    stays importable in the pandas-free test env. Returns None when the frame
    is missing or unreadable so callers can fall back to the players index.
    """
    df = ctx.get("injury_df")
    if df is None:
        return None
    try:
        if bool(getattr(df, "empty", False)):
            return None
        records = df.to_dict("records")
    except Exception:
        logger.debug("[front-office] injury_df read failed", exc_info=True)
        return None
    out: dict[str, dict] = {}
    for rec in records or []:
        if not isinstance(rec, dict):
            continue
        pid = str(rec.get("PlayerID") or "")
        if not pid or pid in out:
            continue
        out[pid] = {
            "designation": str(rec.get("Injury") or rec.get("Status") or "").strip(),
            "body": str(rec.get("Body") or "").strip(),
        }
    return out


def _roster_injury_map(ctx: dict) -> dict[str, dict]:
    """player_id -> {designation, body} for every injured player.

    Starts from the Sleeper players index scan, then overlays the context's
    injury_df (build_injury_report(), the canonical source) where present so
    its curated designations and body parts win. The merged map covers free
    agents too, which the df (built with include_free_agents=False) omits.
    """
    players_full = ctx.get("players") or {}
    out: dict[str, dict] = {}
    for pid, designation in _injury_lookup(ctx).items():
        info = players_full.get(pid) or {}
        if not isinstance(info, dict):
            info = {}
        out[pid] = {
            "designation": designation,
            "body": str(info.get("injury_body_part") or "").strip(),
        }
    df_map = _injury_df_map(ctx)
    if df_map:
        out.update(df_map)
    return out


def _build_injury_rows(ctx: dict, roster_rows: list[dict]) -> list[dict]:
    """Every rostered player with an injury designation, sorted by severity.

    Designations and body parts come from _roster_injury_map (the context's
    injury_df from build_injury_report() overlaid on the Sleeper players
    index). Each row carries the ESPN-derived expected return and the roster
    action from the shared injury_plan kernel. Pure data; renders even when
    the AI is down.
    """
    try:
        from dashboard_services.injury_return import (
            injury_roster_verdict,
            weeks_out_for_player,
        )
    except Exception:
        logger.debug("[front-office] injury_return import failed", exc_info=True)
        injury_roster_verdict = None
        weeks_out_for_player = None
    inj_map = _roster_injury_map(ctx)
    rows_by_id = {str(r.get("id") or ""): r for r in roster_rows}
    rows: list[dict] = []
    for pid, info in inj_map.items():
        r = rows_by_id.get(pid)
        if r is None:
            continue
        designation = str(info.get("designation") or "").strip()
        body = str(info.get("body") or "").strip()
        if not _inj_is_reportable(designation, body):
            continue
        canon = _inj_canonical(designation)
        # A row can be reportable via its body part alone (e.g. designation
        # "Active" with a lingering body note); don't print a non-injury
        # designation as the label in that case.
        inj = canon if canon in _INJURY_DESIGNATIONS else ""
        weeks_out = None
        if weeks_out_for_player and pid:
            try:
                weeks_out = weeks_out_for_player(pid)
            except Exception:
                logger.debug("[front-office] weeks_out failed for %s", pid, exc_info=True)
        action = ""
        if injury_roster_verdict:
            try:
                verdict = injury_roster_verdict(
                    status=inj,
                    weeks_out=weeks_out,
                    player_value=r.get("value"),
                )
                action = str(verdict.get("label") or "")
            except Exception:
                logger.debug("[front-office] injury verdict failed for %s", pid, exc_info=True)
        return_label = _fmt_weeks_out(weeks_out)
        rows.append({
            "id": pid,
            "name": r.get("name"),
            "position": r.get("position"),
            "team": r.get("team"),
            "role": r.get("role"),
            "injury": inj,
            "body": body,
            "weeks_out": weeks_out,
            "return_label": f"out {return_label}" if return_label else "",
            "action": action,
        })
    rows.sort(key=lambda r: (_inj_severity(r["injury"]), str(r.get("name") or "")))
    return rows


def _opponent_injuries(ctx: dict, this_week: dict | None) -> dict:
    """This week's opponent: top 3 injured starters by severity.

    Injury data comes from _roster_injury_map (the context's injury_df from
    build_injury_report() overlaid on the Sleeper players index).
    """
    out = {"entries": [], "line": ""}
    try:
        opp_rid = str((this_week or {}).get("opponent_roster_id") or "")
        if not opp_rid:
            return out
        roster = next(
            (r for r in ctx.get("rosters") or []
             if str(r.get("roster_id")) == opp_rid),
            None,
        )
        if not roster:
            return out
        inj_map = _roster_injury_map(ctx)
        players_index = ctx.get("players_index") or {}
        players_map = ctx.get("players_map") or {}
        entries = []
        for pid in roster.get("starters") or []:
            spid = str(pid)
            info = inj_map.get(spid)
            if not info:
                continue
            designation = str(info.get("designation") or "").strip()
            body = str(info.get("body") or "").strip()
            if not _inj_is_reportable(designation, body):
                continue
            canon = _inj_canonical(designation)
            name = ""
            for src in (players_index, players_map):
                pinfo = (src or {}).get(spid) or {}
                name = pinfo.get("full_name") or pinfo.get("name") or name
                if name:
                    break
            entries.append({
                "name": name or spid,
                "injury": canon if canon in _INJURY_DESIGNATIONS else "",
                "body": body,
            })
        entries.sort(key=lambda e: _inj_severity(e["injury"]))
        entries = entries[:3]
        out["entries"] = entries
        out["line"] = ", ".join(
            f"{e['name']} ({e['injury'] or e['body'] or 'injury'})"
            for e in entries
        )
    except Exception:
        logger.debug("[front-office] opponent injuries failed", exc_info=True)
    return out


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


def _exclude_hurt_waiver_targets(waiver_targets: list[dict],
                                 scoring_type: str) -> list[dict]:
    """Dart hardening: seriously-hurt players can never be waiver "add"
    candidates in redraft.

    Uses the canonical utils.waiver_score.SERIOUS_INJURY_STATUSES set (lazy
    import, like _urgent_needs, so this stays importable without the full
    app stack). Dynasty keeps hurt players but labels them stash-only so the
    report prices them as stashes, never as immediate adds.
    """
    try:
        from utils.waiver_score import SERIOUS_INJURY_STATUSES
    except Exception:
        SERIOUS_INJURY_STATUSES = set()

    def _serious(w: dict) -> bool:
        return _inj_canonical(str(w.get("injury") or "")) in SERIOUS_INJURY_STATUSES

    targets = list(waiver_targets or [])
    if scoring_type == "redraft":
        return [w for w in targets if not _serious(w)]
    for w in targets:
        if _serious(w):
            w["stash_only"] = True
    return targets


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
            {"id": r.get("id"), "name": r["name"], "position": r["position"], "trend_7d": r["trend_7d"]}
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
    # Potential targets backstop: when no fair package exists, still name
    # the players worth pursuing at the need positions so the full report
    # always carries a trade targets section.
    if trade_targets:
        potential_targets: list[dict] = []
    else:
        need_positions = [u.get("position") for u in urgent_needs] + list(weakest)
        potential_targets = _potential_trade_targets(
            ctx, viewer_roster_id, model_value_lookup, need_positions,
        )
    injury_rows = _build_injury_rows(ctx, roster_rows)
    # Roster table shows the ESPN return estimate next to the injury pill.
    wo_by_id = {r["id"]: r["weeks_out"] for r in injury_rows}
    for r in roster_rows:
        if r.get("injury"):
            r["weeks_out"] = wo_by_id.get(r["id"])
    # Injury pills on the players the report recommends acquiring. Gated on
    # reportable designations so non-injury statuses never get a pill.
    inj_map = _roster_injury_map(ctx)
    for t in trade_targets:
        for g in t.get("gets") or []:
            info = inj_map.get(str(g.get("id")) or "") or {}
            desig = str(info.get("designation") or "")
            g["injury"] = desig if _inj_is_reportable(desig, info.get("body")) else ""
    for w in waiver_targets:
        info = inj_map.get(str(w.get("id")) or "") or {}
        desig = str(info.get("designation") or "")
        w["injury"] = desig if _inj_is_reportable(desig, info.get("body")) else ""
    for p in potential_targets:
        info = inj_map.get(str(p.get("id")) or "") or {}
        desig = str(info.get("designation") or "")
        p["injury"] = desig if _inj_is_reportable(desig, info.get("body")) else ""
    # Dart hardening: a seriously-hurt player can never be an "add"
    # candidate in redraft; dynasty keeps them labeled stash-only.
    waiver_targets = _exclude_hurt_waiver_targets(waiver_targets, scoring_type)
    # Waiver alternatives were annotated from the pre-filter list; recompute
    # so a hurt free agent is never pitched as the reason to skip a trade.
    for t in trade_targets:
        t.pop("waiver_alternative", None)
    _annotate_waiver_alternatives(trade_targets, waiver_targets)
    opponent_injuries = _opponent_injuries(ctx, this_week)
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
        "potential_trade_targets": potential_targets,
        "waiver_targets": waiver_targets,
        "cut_candidates": cut_candidates,
        "drop_add_pairs": _drop_add_pairs(cut_candidates, waiver_targets),
        "urgent_needs": urgent_needs,
        "injury_rows": injury_rows,
        "opponent_injuries": opponent_injuries,
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
        # Injury designations change on their own schedule; without this a
        # status change would serve the 12h-cached report with stale injuries.
        "injuries": sorted(
            f"{r['id']}:{_inj_canonical(r.get('injury') or '')}"
            for r in (data.get("roster_rows") or []) if r.get("injury")
        ),
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


def _injury_section_html(rows: list[dict]) -> str:
    """Full injury report section for the modal. Deterministic: renders from
    computed data with no AI dependency."""
    if not rows:
        return (
            "<div class='for-sec'><div class='for-sec-title'>Injury report</div>"
            "<div class='for-muted'>No rostered players carry an injury designation.</div></div>"
        )
    items = []
    for r in rows:
        meta_bits = [p for p in (r.get("body"), r.get("return_label")) if p]
        meta = (
            f"<div class='for-inj-meta'>{html.escape(' · '.join(meta_bits))}</div>"
            if meta_bits else ""
        )
        action = (
            f"<div class='for-inj-action'>{html.escape(r['action'])}</div>"
            if r.get("action") else ""
        )
        items.append(
            "<li class='for-inj-row'>"
            f"<div><span class='for-inj-name'>{html.escape(str(r.get('name') or ''))}</span> "
            f"<span class='for-muted'>{html.escape(str(r.get('position') or ''))}"
            f" · {html.escape(str(r.get('team') or ''))}</span> "
            f"<span class='for-inj'>{html.escape(str(r.get('injury') or ''))}</span>{meta}</div>"
            f"{action}</li>"
        )
    return (
        "<div class='for-sec'><div class='for-sec-title'>Injury report</div>"
        f"<ul class='for-inj-list'>{''.join(items)}</ul></div>"
    )


def _injury_card_html(rows: list[dict]) -> str:
    """Compact injury summary for the Season Hub card."""
    if not rows:
        return ""
    items = []
    for r in rows[:3]:
        meta = " · ".join(
            p for p in (r.get("body"), r.get("return_label"), r.get("action")) if p
        )
        items.append(
            "<li><strong>" + html.escape(str(r.get("name") or "")) + "</strong> "
            f"<span class='for-inj'>{html.escape(str(r.get('injury') or ''))}</span>"
            + (f" <span class='for-muted'>{html.escape(meta)}</span>" if meta else "")
            + "</li>"
        )
    more = ""
    if len(rows) > 3:
        more = f"<li class='for-muted'>+{len(rows) - 3} more in the full report</li>"
    return (
        "<div class='for-card-inj'><span class='for-lbl'>Injuries</span>"
        f"<ul>{''.join(items)}{more}</ul></div>"
    )


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
    inj_html = _injury_card_html(data.get("injury_rows") or [])
    foot = f"{html.escape(team)}" if team else ""
    if week_lbl:
        foot = f"{foot} · {week_lbl}" if foot else week_lbl
    return f"""
    <div class='for-card-summary'>
      {verdict_block}
      {inj_html}
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
        inj = ""
        if r.get("injury"):
            label = str(r["injury"])
            wo = _fmt_weeks_out(r.get("weeks_out"))
            if wo:
                label = f"{label} · {wo}"
            inj = f" <span class='for-inj'>{html.escape(label)}</span>"
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
    ups = [_move_row(m, up=True) for m in data.get("risers_7d") or []]
    downs = [_move_row(m, up=False) for m in data.get("fallers_7d") or []]
    if not score_html and not ups and not downs:
        return ""
    cols = []
    if ups:
        cols.append(
            "<div class='for-moves-col'><div class='for-moves-col-title for-up'>Risers</div>"
            f"<ul class='for-moves'>{''.join(ups)}</ul></div>"
        )
    if downs:
        cols.append(
            "<div class='for-moves-col'><div class='for-moves-col-title for-down'>Fallers</div>"
            f"<ul class='for-moves'>{''.join(downs)}</ul></div>"
        )
    moves_html = f"<div class='for-two-col for-moves-cols'>{''.join(cols)}</div>" if cols else ""
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
    name = html.escape(m["name"])
    pid = str(m.get("id") or "")
    if pid:
        name_html = (
            f"<span class='player-clickable' data-player-id='{html.escape(pid, quote=True)}'"
            f" data-player-name='{html.escape(m['name'], quote=True)}'><strong>{name}</strong></span>"
        )
    else:
        name_html = f"<strong>{name}</strong>"
    return (
        "<li class='for-move'>"
        f"<span class='for-move-delta {cls}'>{arrow} {delta:g}</span>"
        f"<span class='for-move-body'>{name_html}"
        f"<span class='for-muted'> {html.escape(m['position'])}</span></span>"
        "</li>"
    )


def _trade_targets_html(targets: list[dict], trade_notes: dict) -> str:
    if not targets:
        return ""
    cards = []
    for t in targets:
        get = t["gets"][0]
        get_inj = (
            f" <span class='for-inj'>{html.escape(str(get['injury']))}</span>"
            if get.get("injury") else ""
        )
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
            f"<div class='for-target-top'><div class='for-target-head'><strong>{html.escape(get['name'])}</strong>{get_inj} "
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


def _potential_trade_targets_html(potential: list[dict]) -> str:
    """Fallback trade section for when the suggestions engine prices no
    fair deal: name the players worth pursuing at the team's need
    positions, without an invented give side. With no targets at all,
    render the section with an explicit empty state (same contract as
    the Injury report section) instead of vanishing."""
    if not potential:
        return (
            "<div class='for-sec'><div class='for-sec-title'>Trade targets</div>"
            "<div class='for-muted'>No clear trade targets right now. "
            "No leaguemate surplus matches your needs at a fair price.</div></div>"
        )
    cards = []
    for p in potential:
        inj = (
            f" <span class='for-inj'>{html.escape(str(p['injury']))}</span>"
            if p.get("injury") else ""
        )
        cards.append(
            "<div class='for-target-card'>"
            f"<div class='for-target-top'><div class='for-target-head'><strong>{html.escape(str(p.get('name') or ''))}</strong>{inj} "
            f"<span class='for-muted'>{html.escape(str(p.get('position') or ''))}"
            + (f" · age {p['age']}" if p.get("age") not in (None, "") else "")
            + f" · value {p.get('value', 0):g}</span></div>"
            f"<span class='for-target-from'>On {html.escape(str(p.get('partner') or ''))}</span></div>"
            "</div>"
        )
    return (
        "<div class='for-sec'><div class='for-sec-title'>Potential trade targets</div>"
        f"<div class='for-targets'>{''.join(cards)}</div></div>"
    )


def _waivers_cuts_html(data: dict, ai: dict) -> str:
    wnotes = ai.get("waiver_notes") or {}
    w_items = []
    for w in data.get("waiver_targets") or []:
        note = html.escape(str(wnotes.get(w["id"]) or ""))
        rank = f" · {html.escape(w['pos_rank_label'])}" if w.get("pos_rank_label") else ""
        w_inj = (
            f" <span class='for-inj'>{html.escape(str(w['injury']))}</span>"
            if w.get("injury") else ""
        )
        w_stash = (
            " <span class='for-lbl-inline'>IR stash only</span>"
            if w.get("stash_only") else ""
        )
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
            f"<div class='for-pick-body'><strong>{html.escape(w['name'])}</strong>{w_inj}{w_stash} "
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
    top_move = html.escape(str(ai.get("top_move") or ""))
    gm_alert = html.escape(str(ai.get("gm_alert") or ""))
    team = html.escape(str(data.get("team_name") or "Front Office Report"))
    week = data.get("week")
    rec = data.get("record")
    pct = data.get("playoff_pct")

    grades_html = _grades_html(data.get("grades") or [])
    changes_html = _changes_html(data)
    targets_html = _trade_targets_html(data.get("trade_targets") or [], ai.get("trade_notes") or {})
    if not targets_html:
        targets_html = _potential_trade_targets_html(data.get("potential_trade_targets") or [])
    waivers_html = _waivers_cuts_html(data, ai)
    roster_html = _roster_table_html(data.get("roster_rows") or [])
    injury_html = _injury_section_html(data.get("injury_rows") or [])

    opp = data.get("opponent_injuries") or {}
    opp_html = ""
    if opp.get("entries"):
        opp_html = (
            "<div class='for-opp-inj'><span class='for-lbl-inline'>Opponent missing</span> "
            f"{html.escape(str(opp.get('line') or ''))}</div>"
        )

    alert_html = ""
    if gm_alert:
        alert_html = (
            "<div class='for-sec'><div class='for-sec-title'>GM alert</div>"
            f"<p class='for-alert'>{gm_alert}</p></div>"
        )
    posture_html = f"<p class='for-posture'>{posture}</p>" if posture else ""
    top_move_html = ""
    if top_move:
        top_move_html = (
            "<div class='for-sec'><div class='for-sec-title'>Top move</div>"
            f"<div class='for-card-move'><div>{top_move}</div></div></div>"
        )
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
      {injury_html}
      {opp_html}
      <h3 class='for-headline'>{headline}</h3>
      {posture_html}
      {top_move_html}
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
