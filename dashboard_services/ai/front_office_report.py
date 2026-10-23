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
from urllib.parse import quote_plus

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

CACHE_VERSION = "v7"
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
        base = {
            "partner": _safe_str(p.get("team_name") or ""),
            "partner_record": partner_record,
            "partner_needs": partner_needs,
            "partner_direction": _safe_str(p.get("partner_direction") or ""),
            "why_they_say_yes": why_they_say_yes,
            "depth_warnings": [str(w) for w in (p.get("depth_warnings") or [])],
            "depth_checked": bool(p.get("depth_checked")),
        }
        targets.append({
            **base,
            "gets": [
                {
                    "id": str(g.get("id")),
                    "name": _safe_str(g.get("name")),
                    "position": _safe_str(g.get("position")),
                    "age": g.get("age"),
                    "value": round(safe_float(g.get("value")), 1),
                    "tier": _safe_str(g.get("tier") or ""),
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
            "fairness_label": _fairness_label(p.get("fairness"),
                                              p.get("value_you_get"),
                                              p.get("value_you_give")),
            "acceptance_rate": p.get("acceptance_rate"),
            "acceptance_label": _acceptance_label(p.get("acceptance_rate")),
            "analyzer_url": url,
        })
        # Second deal with the same partner (different need), when the
        # engine found one: it renders as its own card.
        alt = p.get("alt_deal") or {}
        alt_gets = (alt.get("gets") or [])[:2]
        alt_gives = (alt.get("gives") or [])[:3]
        if alt_gets and alt_gives:
            a_get_ids = ",".join(str(g.get("id")) for g in alt_gets if g.get("id"))
            a_give_ids = ",".join(str(g.get("id")) for g in alt_gives if g.get("id"))
            targets.append({
                **base,
                "gets": [
                    {
                        "id": str(g.get("id")),
                        "name": _safe_str(g.get("name")),
                        "position": _safe_str(g.get("position")),
                        "age": g.get("age"),
                        "value": round(safe_float(g.get("value")), 1),
                        "tier": _safe_str(g.get("tier") or ""),
                    }
                    for g in alt_gets
                ],
                "gives": [
                    {
                        "id": str(g.get("id")),
                        "name": _safe_str(g.get("name")),
                        "position": _safe_str(g.get("position")),
                        "value": round(safe_float(g.get("value")), 1),
                    }
                    for g in alt_gives
                ],
                "fairness": alt.get("fairness"),
                "fairness_label": _fairness_label(
                    alt.get("fairness"),
                    sum(safe_float(g.get("value")) for g in alt_gets),
                    sum(safe_float(g.get("value")) for g in alt_gives),
                ),
                "acceptance_rate": alt.get("acceptance_rate"),
                "acceptance_label": _acceptance_label(alt.get("acceptance_rate")),
                "depth_warnings": [str(w) for w in (alt.get("depth_warnings") or [])],
                "depth_checked": bool(alt.get("depth_checked")),
                "analyzer_url": (
                    f"/{platform}/{season}/{league_id}/trade"
                    f"?a={a_get_ids}&b={a_give_ids}"
                ),
            })
    return targets[:4]


def _fairness_label(fairness, value_get=None, value_give=None) -> str:
    """Plain-language fairness for the card pill: 'Fair deal, within 5%'."""
    try:
        f = float(fairness)
    except (TypeError, ValueError):
        return ""
    if f <= 0:
        return ""
    off = max(0, int(round((1.0 - f) * 100)))
    if off <= 2:
        return "Fair deal, even"
    label = f"Fair deal, within {off}%"
    try:
        get_v, give_v = float(value_get or 0), float(value_give or 0)
    except (TypeError, ValueError):
        get_v = give_v = 0.0
    if give_v > get_v > 0:
        label = f"Fair deal, you add about {off}%"
    return label


def _acceptance_label(rate) -> str:
    """Acceptance estimate (5-90 scale) as a card pill label."""
    try:
        r = int(rate)
    except (TypeError, ValueError):
        return ""
    if r <= 0:
        return ""
    if r >= 60:
        return "Likely to accept"
    if r >= 40:
        return "Possible"
    return "Long shot"


def _starter_bar(pos: str) -> float:
    """Starter-caliber value bar for a position, tolerant of test envs
    where STARTER_THRESHOLD is stubbed to a non-dict."""
    try:
        return float(STARTER_THRESHOLD.get(pos, 350))
    except Exception:
        return 350.0


def _position_slots(ctx: dict) -> dict[str, int]:
    """Dedicated starting slots per skill position from roster_positions.

    FLEX is fungible and not attributed (same convention as the trade
    engine's depth model); SUPER_FLEX counts toward QB. Defaults to a
    standard 1QB lineup when the league shape is unknown."""
    rp = [str(p or "").upper() for p in (ctx.get("roster_positions") or [])]
    if not rp:
        return {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    return {
        "QB": max(1, rp.count("QB") + rp.count("SUPER_FLEX")),
        "RB": max(1, rp.count("RB")),
        "WR": max(1, rp.count("WR")),
        "TE": max(1, rp.count("TE")),
    }


_FLEX_SLOT_NAMES = ("FLEX", "WRRB_FLEX")


def _flex_slot_count(ctx: dict) -> int:
    """RB/WR/TE-eligible flex slots in the league's roster positions.
    SUPER_FLEX is not counted here: it is QB-eligible and already raises
    the QB slot count in _position_slots."""
    rp = [str(p or "").upper() for p in (ctx.get("roster_positions") or [])]
    return sum(1 for p in rp if p in _FLEX_SLOT_NAMES)


def _skill_starter_capacity(ctx: dict) -> int:
    """How many RB/WR/TE players a team starts each week: dedicated
    RB + WR + TE slots plus flex slots. A skill player ranked below
    this on his own roster is genuinely behind the starters; with two
    flex spots, a WR3 is usually NOT (Kaedon's correction, 2026-10-14)."""
    slots = _position_slots(ctx)
    return slots.get("RB", 0) + slots.get("WR", 0) + slots.get("TE", 0) + _flex_slot_count(ctx)


def _potential_trade_targets(ctx: dict, viewer_roster_id: str,
                             model_value_lookup: dict,
                             positions: list[str],
                             limit: int = 4, per_position: int = 2,
                             qb_add_allowed: bool = True) -> list[dict]:
    """Players worth pursuing at the viewer's need positions, rostered by
    leaguemates. Used when the suggestions engine cannot construct a fair
    package: the report still names who to go get instead of silently
    dropping the trade section.

    Reworked ranking: a candidate must not be seriously injured, and is
    ranked by how gettable he is: genuine surplus for his owner first,
    then depth on the partner's roster, then how close the viewer's own
    surplus package gets to fair value, never by raw value alone.
    Surplus is FLEX-AWARE: a skill player is surplus only when his rank
    among all of the partner's RB/WR/TE players falls below their total
    starting capacity (dedicated slots + flex slots), because flex
    spots absorb exactly that depth. A partner's starter can still
    appear, but only flagged as a Reach with the reason attached. The
    gap to fair and a starting-point piece from the viewer's surplus
    are reported honestly instead of an invented priced deal."""
    priority: list[str] = []
    for pos in positions or []:
        p = str(pos or "").upper()
        if p in ("QB", "RB", "WR", "TE") and p not in priority:
            priority.append(p)
    if not priority:
        return []
    if not qb_add_allowed:
        priority = [p for p in priority if p != "QB"]
        if not priority:
            return []
    prio = {pos: i for i, pos in enumerate(priority)}
    slots = _position_slots(ctx)
    players_index = ctx.get("players_index") or {}
    players_map = ctx.get("players_map") or {}
    roster_map = ctx.get("roster_map") or {}
    inj_lookup = _injury_lookup(ctx)
    viewer_rid = str(viewer_roster_id)

    def _ordered_players(roster: dict, pos: str) -> list[dict]:
        out = []
        for pid in roster.get("players") or []:
            spid = str(pid)
            mv = model_value_lookup.get(spid) or {}
            meta = players_index.get(spid) or players_map.get(spid) or {}
            p = str(meta.get("position") or meta.get("pos") or mv.get("position") or "").upper()
            if p != pos:
                continue
            value = round(safe_float(mv.get("value") or mv.get("model_value") or mv.get("trade_value")), 1)
            if value <= 0:
                continue
            out.append({
                "id": spid,
                "name": _safe_str(meta.get("full_name") or meta.get("name") or mv.get("name") or spid),
                "position": pos,
                "team": _safe_str(meta.get("team") or mv.get("team") or "FA"),
                "age": meta.get("age") if meta.get("age") not in (None, "") else mv.get("age"),
                "value": value,
            })
        return sorted(out, key=lambda x: -x["value"])

    def _is_serious(pid: str) -> bool:
        return _inj_canonical(inj_lookup.get(pid) or "") in (
            "IR", "PUP", "NFI", "SUSP", "SUS", "OUT", "DOUBTFUL", "NA",
        )

    capacity = _skill_starter_capacity(ctx)
    flex_n = _flex_slot_count(ctx)

    def _skill_rank_map(roster: dict) -> dict[str, int]:
        """Player id -> rank (1 = best) among the roster's RB/WR/TE
        players by value. Compared against the flex-aware capacity to
        decide who is really behind the starters."""
        combined: list[dict] = []
        for pos in ("RB", "WR", "TE"):
            combined.extend(_ordered_players(roster, pos))
        combined.sort(key=lambda pl: -pl["value"])
        return {pl["id"]: i + 1 for i, pl in enumerate(combined)}

    # Viewer's sendable surplus: startable players who would not start
    # for the viewer either. Skill positions use the flex-aware capacity
    # (a flex starter is not a free chip); QB keeps the slot rule.
    viewer_roster = _roster_lookup(ctx, viewer_roster_id) or {}
    viewer_skill_ranks = _skill_rank_map(viewer_roster)
    surplus_pool: list[dict] = []
    surplus_by_pos: dict[str, list[dict]] = {}
    for pos in ("QB", "RB", "WR", "TE"):
        ordered = _ordered_players(viewer_roster, pos)
        bar = _starter_bar(pos)
        startable = [pl for pl in ordered if pl["value"] >= bar and not _is_serious(pl["id"])]
        if pos == "QB":
            sendable = startable[slots.get(pos, 1):]
        else:
            sendable = [
                pl for pl in startable
                if viewer_skill_ranks.get(pl["id"], 0) > capacity
            ]
        surplus_by_pos[pos] = sendable
        surplus_pool.extend(sendable)
    surplus_pool.sort(key=lambda pl: pl["value"])

    def _package_for(target_val: float) -> tuple[float, list[dict]]:
        """Fairest cheapest-first prefix of the surplus pool (max 3)."""
        best_val, best_pkg, running = 0.0, [], 0.0
        best_fair = 0.0
        for i, pl in enumerate(surplus_pool[:3]):
            running += pl["value"]
            if running <= 0 or target_val <= 0:
                continue
            fair = min(running, target_val) / max(running, target_val)
            if fair > best_fair:
                best_fair, best_val, best_pkg = fair, running, surplus_pool[: i + 1]
        return best_val, best_pkg

    cands: list[dict] = []
    for r in ctx.get("rosters") or []:
        rid = str(r.get("roster_id") or "")
        if not rid or rid == viewer_rid:
            continue
        partner = _safe_str(roster_map.get(rid) or r.get("team_name") or f"Team {rid}")
        # Partner holes: positions where they cannot field their starters.
        partner_holes: list[str] = []
        for pos in ("QB", "RB", "WR", "TE"):
            ordered_all = _ordered_players(r, pos)
            bar = _starter_bar(pos)
            n_start = sum(1 for pl in ordered_all if pl["value"] >= bar)
            if n_start < slots.get(pos, 1):
                partner_holes.append(pos)
        partner_skill_ranks = _skill_rank_map(r)
        for pos in priority:
            ordered = _ordered_players(r, pos)
            bar = _starter_bar(pos)
            n_startable = sum(1 for pl in ordered if pl["value"] >= bar)
            for idx, cand in enumerate(ordered):
                if cand["value"] < bar:
                    continue
                depth_rank = idx + 1
                # Surplus is flex-aware: for skill positions he must fall
                # below the partner's TOTAL starting capacity (dedicated
                # slots + flex), not just his position's dedicated slots.
                # A WR3 in a two-flex league is a flex starter, not depth.
                # Starters are not excluded, but they can only ever be
                # flagged Reach (see attainability below).
                if pos == "QB":
                    skill_rank = None
                    buried = depth_rank - slots.get("QB", 1)
                    is_surplus = buried > 0
                else:
                    skill_rank = partner_skill_ranks.get(cand["id"])
                    buried = (skill_rank or 0) - capacity
                    is_surplus = buried > 0
                if _is_serious(cand["id"]):
                    continue
                pkg_val, pkg = _package_for(cand["value"])
                gap = (min(pkg_val, cand["value"]) / max(pkg_val, cand["value"])
                       if pkg_val > 0 and cand["value"] > 0 else 0.0)
                start_piece = None
                for hole in partner_holes:
                    if surplus_by_pos.get(hole):
                        start_piece = surplus_by_pos[hole][0]
                        break
                if start_piece is None and pkg:
                    start_piece = pkg[0]
                cands.append({
                    **cand,
                    "partner": partner,
                    "partner_holes": partner_holes,
                    "depth_rank": depth_rank,
                    "skill_rank": skill_rank,
                    "buried": buried,
                    "flex_slots": flex_n,
                    "is_surplus": is_surplus,
                    "sole_starter": n_startable <= 1,
                    "gap_pct": int(round(gap * 100)),
                    "starting_point": (
                        {"name": start_piece["name"], "position": start_piece["position"]}
                        if start_piece else None
                    ),
                    "attainability": (
                        "Realistic"
                        if is_surplus and (gap >= 0.75 or (gap >= 0.60 and buried >= 2))
                        else "Reach"
                    ),
                })
    # Gettability first: surplus pieces and the players buried deepest
    # below the partner's starting group (flex included), closer to a
    # fair package, outrank a bigger name who will never move.
    cands.sort(key=lambda c: (
        prio[c["position"]],
        0 if c["is_surplus"] else 1,
        -c["buried"], -c["gap_pct"], -c["value"],
    ))
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
                   scoring_type: str, weakest: list[str],
                   qb_add_allowed: bool = True,
                   hole_positions: list[str] | None = None,
                   direction: str | None = None,
                   playoff_pct=None) -> list[dict]:
    """Best free agents, gated by the existing waiver quality bar.

    Two rework gates on top of the quality bar:
    - QB adds are suppressed entirely when the roster already covers the
      position (see _qb_context): in a 1QB league with a healthy QB on
      the roster, no wire QB is ever a suggestion.
    - The starter_gap stretch inside the quality bar now means a TRUE
      lineup hole (fewer healthy startable players than dedicated
      slots), not merely a weak positional grade. A weak grade alone no
      longer waives the 1QB streamer bar or the startable-rank ceilings.

    Waiver v2 (item 6): the gate ORDER here matches the shared selector
    ``utils.waivers.rank_team_waiver_suggestions`` (availability/injury,
    then this quality bar, then the x1.05 drop bar in
    ``_waiver_bar_filter``, then hurt exclusion). Front Office keeps its
    own gate implementation because its inputs are report-shaped
    (roster_rows, model_value_lookup, QB context), but scoring, horizon
    choice and claim guidance all come from the shared helpers in
    utils.waivers, so the surfaces cannot drift on ranking or economics.
    Candidates are re-scored with horizon weights chosen from the team's
    direction / playoff odds (item 3) instead of weakest-then-value only.
    """
    try:
        from utils.cross_league_actions import waiver_add_clears_quality_bar
    except Exception:
        logger.debug("[front-office] waiver bar import failed", exc_info=True)
        return []
    is_redraft = scoring_type == "redraft"
    is_sf = bool(_ctx_is_sf(ctx))
    n_teams = len(ctx.get("rosters") or []) or 12
    holes = set(hole_positions or [])
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
        if pos == "QB" and not qb_add_allowed:
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
        gap = 1.0 if pos in holes else 0.0
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
        cand = {
            "id": pid,
            "name": _safe_str(row.get("name")),
            "position": pos,
            "team": _safe_str(row.get("team") or "FA"),
            "age": row.get("age"),
            "value": round(val, 1),
            "pos_rank_label": _safe_str(row.get("pos_rank_label") or ""),
        }
        # Rich opportunity fields ride along when the model table has them
        # (item 5), so the shared scorer sees the same inputs as the
        # Waivers page instead of value alone.
        for _k in ("ros_ppg", "own_proj_ppg", "trend_windows", "usage_stat",
                   "usage_delta", "injured_ahead", "vacated", "healthy_ahead",
                   "schedule_ease_rank", "schedule_total", "waiver_surprise",
                   "adds_count", "trending_adds", "adds_48h"):
            if row.get(_k) is not None:
                cand[_k] = row[_k]
        cands.append(cand)
    # Horizon-weighted re-score (items 3, 4, 6): weakest rooms keep a small
    # need bump inside the shared score, then claim guidance is attached.
    weak_set = set(weakest)
    try:
        from utils.waivers import (
            choose_suggestion_horizon as _choose_horizon,
            claim_guidance_for as _claim_guidance_for,
            faab_recommendation as _faab_rec,
            horizon_weights as _horizon_weights,
            waiver_pickup_score as _pickup_score,
        )
        _horizon = _choose_horizon(
            {"is_redraft": is_redraft, "is_dynasty": not is_redraft},
            direction=direction, playoff_pct=playoff_pct)
        _weights = _horizon_weights(_horizon, dynasty=not is_redraft)
        _holes_n = len(holes)
        for cand in cands:
            _score_c = {
                "player_id": cand["id"], "value": cand["value"],
                "age": safe_float(cand.get("age")), "position": cand["position"],
                "need_mult": 1.15 if cand["position"] in weak_set else 1.0,
            }
            for _k in ("ros_ppg", "own_proj_ppg", "trend_windows", "usage_stat",
                       "usage_delta", "injured_ahead", "vacated", "healthy_ahead",
                       "schedule_ease_rank", "schedule_total", "waiver_surprise"):
                if cand.get(_k) is not None:
                    _score_c[_k] = cand[_k]
            try:
                cand["score"] = round(float(_pickup_score(_score_c, {}, w=_weights)), 2)
            except Exception:
                cand["score"] = cand["value"]
            cand["horizon"] = _horizon
            _adds = cand.get("adds_count", cand.get("trending_adds", cand.get("adds_48h")))
            _g = _claim_guidance_for(cand["score"], adds_count=_adds,
                                     competing_holes=_holes_n)
            cand["claim_guidance"] = _g["label"]
            cand["faab"] = _faab_rec(cand["score"], need_mult=_score_c["need_mult"])
        cands.sort(key=lambda c: (0 if c["position"] in weak_set else 1,
                                  -float(c.get("score") or 0), -c["value"]))
    except Exception:
        logger.debug("[front-office] waiver horizon scoring skipped", exc_info=True)
        cands.sort(key=lambda c: (0 if c["position"] in weak_set else 1, -c["value"]))
    return cands[:4]


def _cut_candidates(roster_rows: list[dict]) -> list[dict]:
    skill = [r for r in roster_rows if r["position"] in ("QB", "RB", "WR", "TE")]
    return skill[-3:][::-1] if skill else []


# ── Top Moves: deterministic candidates + hard gates ────────────────────────
# Designations that mean a player cannot help a lineup any time soon. A
# player carrying one can never be an "add" in the Top Moves, in any
# format; dynasty may still list him in Waivers as an IR stash only.
_SERIOUS_DESIGNATIONS = ("IR", "PUP", "NFI", "SUSP", "SUS", "OUT", "DOUBTFUL", "NA")


def _is_serious_designation(injury) -> bool:
    return _inj_canonical(str(injury or "")) in _SERIOUS_DESIGNATIONS


def _qb_context(ctx: dict, roster: dict, roster_rows: list[dict]) -> dict:
    """QB room facts that gate every QB suggestion in the report.

    healthy: QBs with no serious designation and no QUESTIONABLE tag.
    add_allowed: 1QB needs zero healthy QBs before a wire QB is a real
    suggestion; superflex needs fewer healthy QBs than its QB slots.
    excess: healthy QBs beyond the keep count (2 in 1QB, 3 in superflex),
    worst first: the shop candidates."""
    try:
        is_sf = bool(_ctx_is_sf(ctx))
    except Exception:
        is_sf = False
    slots = _position_slots(ctx)
    qbs = [r for r in roster_rows if str(r.get("position") or "").upper() == "QB"]
    healthy, questionable = [], []
    for r in qbs:
        canon = _inj_canonical(str(r.get("injury") or ""))
        if canon in _SERIOUS_DESIGNATIONS:
            continue
        if canon == "QUESTIONABLE":
            questionable.append(r)
        else:
            healthy.append(r)
    starter_ids = {str(pid) for pid in (roster or {}).get("starters") or []}
    starter_q = next((r for r in qbs if str(r.get("id")) in starter_ids), None)
    starter_questionable = bool(
        starter_q
        and _inj_canonical(str(starter_q.get("injury") or "")) == "QUESTIONABLE"
    )
    keep = 3 if is_sf else 2
    movable = sorted(healthy + questionable, key=lambda r: safe_float(r.get("value")))
    excess = movable[: max(0, len(movable) - keep)] if len(movable) > keep else []
    if is_sf:
        add_allowed = len(healthy) < max(2, slots.get("QB", 2))
        covered = len(healthy) >= max(2, slots.get("QB", 2))
    else:
        add_allowed = len(healthy) == 0
        covered = len(healthy) >= 1
    return {
        "is_sf": is_sf,
        "total": len(qbs),
        "healthy": len(healthy),
        "questionable": len(questionable),
        "starter_questionable_name": starter_q.get("name") if starter_questionable else "",
        "covered": covered,
        "add_allowed": add_allowed,
        "excess": excess,
    }


def _hole_positions(ctx: dict, roster_rows: list[dict]) -> list[str]:
    """Positions where the viewer cannot field their dedicated starters
    with healthy, starter-caliber players. This (not a weak grade) is
    what 'starter hole' means for the waiver quality bar."""
    slots = _position_slots(ctx)
    holes = []
    for pos in ("QB", "RB", "WR", "TE"):
        bar = _starter_bar(pos)
        n = sum(
            1 for r in roster_rows
            if str(r.get("position") or "").upper() == pos
            and safe_float(r.get("value")) >= bar
            and not _is_serious_designation(r.get("injury"))
        )
        if n < slots.get(pos, 1):
            holes.append(pos)
    return holes


def _qb_needy_team_count(ctx: dict, viewer_roster_id: str,
                         model_value_lookup: dict) -> int:
    """Other teams thinner at QB than the viewer is: fewer healthy
    starter-caliber QBs than their QB slots. Context for shopping an
    excess QB, computed from the same lookup as everything else."""
    slots = _position_slots(ctx)
    inj = _injury_lookup(ctx)
    bar = _starter_bar("QB")
    count = 0
    for r in ctx.get("rosters") or []:
        rid = str(r.get("roster_id") or "")
        if not rid or rid == str(viewer_roster_id):
            continue
        n = 0
        for pid in r.get("players") or []:
            spid = str(pid)
            mv = model_value_lookup.get(spid) or {}
            if str(mv.get("position") or "").upper() != "QB":
                continue
            if safe_float(mv.get("value")) < bar:
                continue
            if _inj_canonical(inj.get(spid) or "") in _SERIOUS_DESIGNATIONS:
                continue
            n += 1
        if n < slots.get("QB", 1):
            count += 1
    return count


def _waiver_bar_filter(waiver_targets: list[dict], roster_rows: list[dict],
                       holes: list[str]) -> list[dict]:
    """Kaedon's drop-bar rule (2026-10-07), now applied to Front Office
    waivers too: an add is suggested only if it beats the worst rostered
    player at the same position by x1.05, or it fills a true hole.
    Dynasty IR stashes are holds, not lineup adds, and are exempt."""
    worst: dict[str, float] = {}
    for r in roster_rows or []:
        pos = str(r.get("position") or "").upper()
        val = safe_float(r.get("value"))
        if pos in ("QB", "RB", "WR", "TE") and val > 0:
            if pos not in worst or val < worst[pos]:
                worst[pos] = val
    hole_set = set(holes or [])
    out = []
    for w in waiver_targets or []:
        if w.get("stash_only"):
            out.append(w)
            continue
        pos = str(w.get("position") or "").upper()
        if pos in hole_set or pos not in worst:
            out.append(w)
            continue
        if safe_float(w.get("value")) > worst[pos] * 1.05:
            out.append(w)
    return out


def _build_top_moves(*, trade_targets: list[dict], drop_add_pairs: list[dict],
                     qb: dict, injury_rows: list[dict], roster: dict,
                     ctx: dict, grades: list[dict], holes: list[str],
                     qb_needy_count: int,
                     direction: str | None = None) -> list[dict]:
    """The report's Top Moves, computed and ranked in code. The model
    never chooses these; it only narrates around them. Every candidate
    has already passed the hard gates (QB coverage, injury, drop pairing)
    before it can be scored here."""
    moves: list[dict] = []
    worst = max(grades, key=lambda g: g["rank"]) if grades else None
    rebuilding = str(direction or "").strip().lower() in _REBUILD_DIRECTIONS
    top_trade = None
    for cand_t in trade_targets or []:
        cand_get = (cand_t.get("gets") or [{}])[0]
        if _rebuild_age_block(cand_get.get("age"), rebuilding):
            continue
        top_trade = cand_t
        break
    if top_trade:
        t = top_trade
        get = (t.get("gets") or [{}])[0]
        give_names = ", ".join(
            str(g.get("name") or "") for g in (t.get("gives") or [])
        )
        why_bits = []
        if worst and str(get.get("position") or "").upper() == worst.get("pos"):
            why_bits.append(
                f"Fixes your {worst['pos']} room, ranked {worst['rank']} of {worst['of']}."
            )
        if t.get("why_they_say_yes"):
            why_bits.append(str(t["why_they_say_yes"]))
        if t.get("acceptance_label"):
            why_bits.append(f"{t['acceptance_label']}.")
        score = 90.0 + safe_float(t.get("fairness")) * 10.0
        try:
            score += float(t.get("acceptance_rate") or 0) / 20.0
        except (TypeError, ValueError):
            pass
        moves.append({
            "type": "Trade",
            "line": f"Trade for {get.get('name')} ({get.get('position')})",
            "why": (f"You give {give_names}. " if give_names else "") + " ".join(why_bits),
            "_score": score,
        })
    top_pair = None
    for cand_p in drop_add_pairs or []:
        cand_add = cand_p.get("add") or {}
        if _rebuild_age_block(cand_add.get("age"), rebuilding):
            continue
        top_pair = cand_p
        break
    if top_pair:
        p = top_pair
        drop, add = p.get("drop") or {}, p.get("add") or {}
        delta = safe_float(add.get("value")) - safe_float(drop.get("value"))
        why_bits = [
            f"{add.get('name')} beats {drop.get('name')} on value "
            f"({safe_float(add.get('value')):g} vs {safe_float(drop.get('value')):g})."
        ]
        if str(add.get("position") or "").upper() in holes:
            why_bits.append(f"Fills your {add.get('position')} hole.")
        moves.append({
            "type": "Waiver",
            "line": f"Drop {drop.get('name')}, add {add.get('name')}",
            "why": " ".join(why_bits),
            "_score": 80.0 + min(15.0, max(0.0, delta) / 25.0),
        })
    excess = (qb or {}).get("excess") or []
    if excess:
        q = excess[0]
        fmt = "superflex" if qb.get("is_sf") else "1QB"
        why_bits = [
            f"You roster {qb.get('total')} QBs in a {fmt} league. "
            f"{q.get('name')} has no path to your lineup."
        ]
        if qb_needy_count:
            why_bits.append(
                f"{qb_needy_count} team{'s' if qb_needy_count != 1 else ''} in this "
                "league are thinner at QB than you."
            )
        moves.append({
            "type": "Trade",
            "line": f"Shop {q.get('name')}",
            "why": " ".join(why_bits),
            "_score": 70.0,
        })
    rp = [str(p or "").upper() for p in (ctx.get("roster_positions") or [])]
    if "IR" in rp:
        reserve_ids = {str(pid) for pid in (roster or {}).get("reserve") or []}
        for r in injury_rows or []:
            canon = _inj_canonical(str(r.get("injury") or ""))
            if canon in ("IR", "PUP", "NFI", "OUT") and str(r.get("id")) not in reserve_ids:
                moves.append({
                    "type": "Roster",
                    "line": f"Move {r.get('name')} to the IR slot",
                    "why": "Frees a bench spot without dropping anyone. "
                           f"{r.get('name')} stays stashed while out.",
                    "_score": 60.0,
                })
                break
    moves.sort(key=lambda m: -m["_score"])
    out = []
    seen = set()
    for m in moves:
        if m["line"] in seen:
            continue
        seen.add(m["line"])
        out.append({"type": m["type"], "line": m["line"], "why": m["why"]})
        if len(out) >= 3:
            break
    return out


_REBUILD_DIRECTIONS = ("rebuild", "retool")
_REBUILD_AGE_CAP = 30.0


def _rebuild_age_block(age, rebuilding: bool) -> bool:
    """A rebuilding / retooling team is never told to acquire a 30+
    veteran (Kaedon, 2026-10-23: "if I am rebuilding, I shouldnt be
    told to go get a 30+ yr old TE"). Unknown age never blocks."""
    if not rebuilding or age in (None, ""):
        return False
    try:
        return float(age) >= _REBUILD_AGE_CAP
    except (TypeError, ValueError):
        return False


def _build_decision_moves(*, trade_targets: list[dict],
                          potential_targets: list[dict],
                          drop_add_pairs: list[dict],
                          waiver_targets: list[dict],
                          cut_candidates: list[dict],
                          injury_rows: list[dict], roster: dict,
                          roster_rows: list[dict], ctx: dict,
                          grades: list[dict], holes: list[str],
                          direction: str | None = None) -> list[dict]:
    """The modal's decision board (v3, approved 2026-10-23): moves in
    deadline sequence, not score order, because some moves unlock the
    next one. IR first (frees the spot), then the waiver add that uses
    it, then the trade, then the lineup fix for Sunday. Every move is
    assembled from the same gated candidates as Top Moves; the model
    never chooses these either. Each move carries the ids / indices its
    in-place evidence renders from, plus a deterministic After line
    describing the post-move state from data the report already has."""
    moves: list[dict] = []
    worst = max(grades, key=lambda g: g["rank"]) if grades else None
    rebuilding = str(direction or "").strip().lower() in _REBUILD_DIRECTIONS

    # 1. IR move: a seriously-hurt player not already on reserve, when
    # the league has an IR slot. Same gate as Top Moves.
    ir_move: dict | None = None
    rp = [str(p or "").upper() for p in (ctx.get("roster_positions") or [])]
    if "IR" in rp:
        reserve_ids = {str(pid) for pid in (roster or {}).get("reserve") or []}
        for r in injury_rows or []:
            canon = _inj_canonical(str(r.get("injury") or ""))
            if canon in ("IR", "PUP", "NFI", "OUT") and str(r.get("id")) not in reserve_ids:
                if r.get("return_label") and r.get("body"):
                    lead = f"{r.get('name')} is {r['return_label']} ({r['body']}). "
                elif r.get("return_label"):
                    lead = f"{r.get('name')} is {r['return_label']}. "
                elif r.get("body"):
                    lead = f"{r.get('name')} is out ({r['body']}). "
                else:
                    lead = ""
                why = lead + "The IR slot holds him without costing you a bench spot."
                ir_move = {
                    "kind": "ir", "type": "Roster", "timing": "Do now",
                    "line": f"Move {r.get('name')} to the IR slot",
                    "why": why,
                    "after": "One open bench spot, nobody dropped.",
                    "dep": None, "injury_id": str(r.get("id") or ""),
                }
                break
    if ir_move:
        moves.append(ir_move)

    # 2. Waiver move: the top drop/add pair. The pair's add snapshot is
    # thin (name/position/value), so enrich it from the full waiver row.
    # Hard gate, held here as well as upstream: an IR stash is a hold,
    # never an immediate add, so a stash-only target can never become a
    # board move or be paired with a drop (the Pierce bug, 2026-10-23:
    # pairs were empty precisely because the only target was a stash,
    # and an unfiltered fallback resurrected him as Move 1).
    stash_names = {
        str(w.get("name")) for w in waiver_targets or [] if w.get("stash_only")
    }
    pair = None
    for cand in drop_add_pairs or []:
        cand_add = cand.get("add") or {}
        if str(cand_add.get("name")) in stash_names:
            continue
        cand_full = next(
            (w for w in waiver_targets or []
             if str(w.get("name")) == str(cand_add.get("name"))),
            {},
        )
        if _rebuild_age_block(
            cand_full.get("age", cand_add.get("age")), rebuilding,
        ):
            continue
        pair = cand
        break
    if pair:
        p_add, p_drop = pair.get("add") or {}, pair.get("drop") or {}
        full_add = next(
            (w for w in waiver_targets or []
             if str(w.get("name")) == str(p_add.get("name"))),
            {},
        )
        full_drop = next(
            (c for c in cut_candidates or []
             if str(c.get("name")) == str(p_drop.get("name"))),
            {},
        )
        add = {
            "id": str(full_add.get("id") or ""), "name": p_add.get("name"),
            "position": p_add.get("position"), "team": full_add.get("team") or "",
            "value": p_add.get("value"),
            "pos_rank_label": full_add.get("pos_rank_label") or "",
            "injury": full_add.get("injury") or "",
            "claim_guidance": p_add.get("claim_guidance") or full_add.get("claim_guidance") or "",
            "stash_only": bool(full_add.get("stash_only")),
        }
        drop = {
            "id": str(full_drop.get("id") or ""), "name": p_drop.get("name"),
            "position": p_drop.get("position"), "value": p_drop.get("value"),
        }
        pos = str(add.get("position") or "").upper()
        why_bits = [
            f"{add.get('name')} beats {drop.get('name')} on value "
            f"({safe_float(add.get('value')):g} vs {safe_float(drop.get('value')):g}) "
            "and is startable this week."
        ]
        if worst and pos == worst.get("pos"):
            why_bits.append(
                f"Your {pos} room ranks {worst['rank']} of {worst['of']}, your worst room."
            )
        elif pos in (holes or []):
            why_bits.append(f"Fills your {pos} hole.")
        before = sum(
            1 for r in roster_rows or []
            if str(r.get("position") or "").upper() == pos and r.get("role") == "Starter"
        )
        clears = safe_float(add.get("value")) >= STARTER_THRESHOLD.get(pos, 350)
        after_n = before + (1 if clears else 0)
        if clears and before:
            after = f"{after_n} startable {pos}s, up from {before}."
        elif clears:
            after = f"{add.get('name')} gives you a startable {pos}."
        else:
            after = f"{add.get('name')} joins your {pos} room as depth."
        if ir_move:
            after += (
                f" No drop needed with Move {len(moves)} done. "
                f"Without it, drop {drop.get('name')}."
            )
        else:
            after += f" You drop {drop.get('name')}."
        moves.append({
            "kind": "waiver", "type": "Waiver", "timing": "Before waivers",
            "line": f"Add {add.get('name')}",
            "why": " ".join(why_bits), "after": after, "dep": None,
            "add": add, "drop": drop,
        })
    elif waiver_targets:
        w = next(
            (cand_w for cand_w in waiver_targets
             if not cand_w.get("stash_only")
             and not _rebuild_age_block(cand_w.get("age"), rebuilding)),
            None,
        )
        drop = (cut_candidates or [{}])[0]
        if w is not None:
            moves.append({
            "kind": "waiver", "type": "Waiver", "timing": "Before waivers",
            "line": f"Add {w.get('name')}",
            "why": (
                f"Top available {w.get('position')} on waivers"
                + (f", {w.get('pos_rank_label')}" if w.get("pos_rank_label") else "")
                + "."
            ),
            "after": (
                f"You drop {drop.get('name')}." if drop.get("name") else ""
            ),
            "dep": None,
            "add": {
                "id": str(w.get("id") or ""), "name": w.get("name"),
                "position": w.get("position"), "team": w.get("team") or "",
                "value": w.get("value"),
                "pos_rank_label": w.get("pos_rank_label") or "",
                "injury": w.get("injury") or "",
                "claim_guidance": w.get("claim_guidance") or "",
                "stash_only": bool(w.get("stash_only")),
            },
            "drop": {
                "id": str(drop.get("id") or ""), "name": drop.get("name"),
                "position": drop.get("position"), "value": drop.get("value"),
            } if drop.get("name") else None,
        })

    # 3. Trade move: the top priced deal, or the top potential target
    # when the engine priced nothing. When rebuilding, deals whose
    # headline get is a 30+ veteran are skipped for the next deal
    # that fits the timeline.
    trade_idx = None
    plan_b_idx = None
    for i, cand_t in enumerate(trade_targets or []):
        cand_get = (cand_t.get("gets") or [{}])[0]
        if _rebuild_age_block(cand_get.get("age"), rebuilding):
            continue
        if trade_idx is None:
            trade_idx = i
        else:
            plan_b_idx = i
            break
    if trade_idx is not None:
        t = trade_targets[trade_idx]
        get = (t.get("gets") or [{}])[0]
        give_names = " + ".join(
            str(g.get("name") or "") for g in (t.get("gives") or [])
        )
        why_bits = []
        if t.get("fairness_label") and t.get("acceptance_label"):
            why_bits.append(f"{t['fairness_label']}. {t['acceptance_label']}.")
        if worst and str(get.get("position") or "").upper() == worst.get("pos"):
            why_bits.append(
                f"Fixes your {worst['pos']} room, ranked {worst['rank']} of {worst['of']}."
            )
        if t.get("why_they_say_yes"):
            why_bits.append(str(t["why_they_say_yes"]))
        after = ""
        if t.get("depth_warnings"):
            after = str(t["depth_warnings"][0])
        elif t.get("depth_checked"):
            after = "No new hole created after the trade."
        moves.append({
            "kind": "trade", "type": "Trade", "timing": "This week",
            "line": f"Offer {give_names} for {get.get('name')}" if give_names
                    else f"Trade for {get.get('name')}",
            "why": " ".join(why_bits), "after": after, "dep": None,
            "target_index": trade_idx,
            "plan_b_index": plan_b_idx,
        })
    elif potential_targets:
        pot_idx = [
            i for i, cand_p in enumerate(potential_targets)
            if not _rebuild_age_block(cand_p.get("age"), rebuilding)
        ][:2]
        if pot_idx:
            p = potential_targets[pot_idx[0]]
            gap = int(p.get("gap_pct") or 0)
            why_bits = ["Top gettable target at a position you need."]
            if gap:
                why_bits.append(f"Your surplus package reaches about {gap}% of his value.")
            moves.append({
                "kind": "potential", "type": "Trade", "timing": "This week",
                "line": f"Target {p.get('name')} ({p.get('position')})",
                "why": " ".join(why_bits), "after": "", "dep": None,
                "potential_indices": pot_idx,
            })

    # 4. Lineup move: a lineup starter who cannot play (OUT or worse)
    # with a healthy same-position bench player to step in. Questionable
    # starters are game-time calls and live in Hold / Watch instead.
    starter_ids = {str(pid) for pid in (roster or {}).get("starters") or []}
    if starter_ids:
        bench = [r for r in roster_rows or [] if str(r.get("id")) not in starter_ids]
        for r in injury_rows or []:
            if str(r.get("id")) not in starter_ids:
                continue
            canon = _inj_canonical(str(r.get("injury") or ""))
            if canon not in ("OUT", "DOUBTFUL", "IR", "PUP", "NFI"):
                continue
            pos = str(r.get("position") or "").upper()
            fills = [
                b for b in bench
                if str(b.get("position") or "").upper() == pos
                and not _is_serious_designation(b.get("injury"))
                and _inj_canonical(str(b.get("injury") or "")) != "QUESTIONABLE"
            ]
            if not fills:
                continue
            fill = max(fills, key=lambda b: safe_float(b.get("value")))
            bits = [b for b in (r.get("body"), r.get("return_label")) if b]
            why = (
                f"{r.get('name')} is {canon}"
                + (f" ({', '.join(bits)})" if bits else "")
                + f". {fill.get('name')} is already on your bench and is the fill-in."
            )
            others = [
                o for o in injury_rows or []
                if str(o.get("id")) in starter_ids
                and str(o.get("id")) != str(r.get("id"))
                and _inj_canonical(str(o.get("injury") or "")) == "QUESTIONABLE"
            ]
            after = "No zero in the lineup."
            if others:
                names = " or ".join(str(o.get("name")) for o in others[:2])
                verb = "is" if len(others) == 1 else "are"
                after += f" Nothing else changes unless {names} {verb} ruled out."
            moves.append({
                "kind": "lineup", "type": "Lineup", "timing": "Before Sunday",
                "line": f"Bench {r.get('name')}, start {fill.get('name')}",
                "why": why, "after": after, "dep": None,
                "injury_id": str(r.get("id") or ""),
                "detail_injury_ids": [str(r.get("id") or "")]
                + [str(o.get("id") or "") for o in others[:2]],
            })
            break

    # Dependency tags name the actual board positions, so they are only
    # assigned once the sequence is final.
    for i, m in enumerate(moves, start=1):
        m["rank"] = i
    if ir_move is not None:
        for m in moves:
            if m["kind"] == "waiver":
                ir_move["dep"] = f"Unlocks Move {m['rank']}"
                m["dep"] = f"Needs Move {ir_move['rank']}"
    return moves


def _qb_note(qb: dict) -> dict | None:
    """The explicit QB line under Top Moves. When the room is covered it
    says so (and why no QB add appears); when there is a genuine hole it
    says that instead, so silence is never ambiguous."""
    if not qb or not qb.get("total"):
        return None
    healthy = qb.get("healthy") or 0
    if qb.get("starter_questionable_name") and qb.get("covered"):
        return {
            "covered": True,
            "lead": "QB: covered, no move.",
            "text": (
                f"{qb['starter_questionable_name']} is QUESTIONABLE, a game-time call. "
                f"You hold {healthy} healthy QB{'s' if healthy != 1 else ''} behind him, "
                "so no QB add is suggested. A QB on IR cannot play and is never an add here."
            ),
        }
    if qb.get("add_allowed"):
        return {
            "covered": False,
            "lead": "QB: real hole.",
            "text": "You have no healthy QB available to start. QB is a genuine need this week.",
        }
    return None


def _ai_top_move_ok(top_move: str, data: dict) -> bool:
    """Deterministic post-check on the model's free-text top move. The
    prompt already bars injured adds and covered-QB adds; this enforces
    it in code so a bad sentence can never reach the card or report."""
    if not top_move:
        return False
    low = f" {top_move.lower()} "
    qb = data.get("qb_context") or {}
    if qb and not qb.get("add_allowed"):
        if " add " in low and (" qb" in low or "quarterback" in low):
            return False
    for w in data.get("waiver_targets") or []:
        if w.get("stash_only") and str(w.get("name") or "").lower() in low and "add" in low:
            return False
    return True


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
                "age": w.get("age"),
                "claim_guidance": w.get("claim_guidance"),
                "faab": w.get("faab"),
                "horizon": w.get("horizon"),
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
    qb_ctx = _qb_context(ctx, roster, roster_rows)
    hole_positions = _hole_positions(ctx, roster_rows)
    waiver_targets = _waiver_targets(ctx, viewer_roster_id, model_value_lookup,
                                     scoring_type, weakest,
                                     qb_add_allowed=qb_ctx["add_allowed"],
                                     hole_positions=hole_positions,
                                     direction=team_ctx.get("direction"),
                                     playoff_pct=team_ctx.get("playoff_pct"))
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
            qb_add_allowed=qb_ctx["add_allowed"],
        )
        # Deep link into the Trade Hub's Build around search (Targets
        # sub-tab), pre-loaded with this player. The hub reads ?build=.
        _ba_platform = _safe_str(ctx.get("platform") or "sleeper")
        _ba_season = ctx.get("season") or ""
        _ba_league = ctx.get("league_id") or ""
        for _pt in potential_targets:
            _pt["build_url"] = (
                f"/{_ba_platform}/{_ba_season}/{_ba_league}/trade"
                f"?tab=hub&build={_pt.get('id')}&name={quote_plus(str(_pt.get('name') or ''))}"
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
    waiver_targets = _waiver_bar_filter(waiver_targets, roster_rows, hole_positions)
    _annotate_waiver_alternatives(trade_targets, waiver_targets)
    opponent_injuries = _opponent_injuries(ctx, this_week)
    # IR stashes (dynasty) are holds, not adds: they never pair with a
    # drop and never become a Top Move.
    pair_targets = [w for w in waiver_targets if not w.get("stash_only")]
    drop_add_pairs = _drop_add_pairs(cut_candidates, pair_targets)
    top_moves = _build_top_moves(
        trade_targets=trade_targets,
        drop_add_pairs=drop_add_pairs,
        qb=qb_ctx,
        injury_rows=injury_rows,
        roster=roster,
        ctx=ctx,
        grades=grades,
        holes=hole_positions,
        qb_needy_count=_qb_needy_team_count(ctx, viewer_roster_id, model_value_lookup),
        direction=team_ctx.get("direction"),
    )
    decision_moves = _build_decision_moves(
        trade_targets=trade_targets,
        potential_targets=potential_targets,
        drop_add_pairs=drop_add_pairs,
        waiver_targets=waiver_targets,
        cut_candidates=cut_candidates,
        injury_rows=injury_rows,
        roster=roster,
        roster_rows=roster_rows,
        ctx=ctx,
        grades=grades,
        holes=hole_positions,
        direction=team_ctx.get("direction"),
    )
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
        "drop_add_pairs": drop_add_pairs,
        "top_moves": top_moves,
        "decision_moves": decision_moves,
        "qb_note": _qb_note(qb_ctx),
        "qb_context": qb_ctx,
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
    # Long labels (PRIORITIZE WAIVERS, SELL VETERANS, REBUILD AGGRESSIVELY)
    # get the small wrapped treatment on the card so the stamp cannot
    # squeeze the headline (F3, approved 2026-10-14); CSS scopes it to the
    # card masthead, the modal hero stamp is unaffected.
    if len(label) > 10:
        cls += " for-stamp-long"
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


def _injury_row_html(r: dict) -> str:
    # The designation pill rides the meta line, not the name line: in
    # the two-column grid the name line wraps badly when the pill,
    # position, and team all compete for the same narrow row.
    meta_bits = [p for p in (r.get("body"), r.get("return_label")) if p]
    pill = (
        f"<span class='for-inj'>{html.escape(str(r.get('injury') or ''))}</span>"
        if r.get("injury") else ""
    )
    meta_inner = " ".join(x for x in (pill, html.escape(" · ".join(meta_bits))) if x)
    meta = f"<div class='for-inj-meta'>{meta_inner}</div>" if meta_inner else ""
    action = (
        f"<div class='for-inj-action'>{html.escape(r['action'])}</div>"
        if r.get("action") else ""
    )
    return (
        "<li class='for-inj-row'>"
        f"<div><span class='for-inj-name'>{html.escape(str(r.get('name') or ''))}</span> "
        f"<span class='for-muted'>{html.escape(str(r.get('position') or ''))}"
        f" · {html.escape(str(r.get('team') or ''))}</span>{meta}</div>"
        f"{action}</li>"
    )


def _injury_section_html(rows: list[dict], *, title: bool = True) -> str:
    """Full injury report section for the modal. Deterministic: renders from
    computed data with no AI dependency."""
    if not rows:
        if not title:
            return "<div class='for-muted'>No rostered players carry an injury designation.</div>"
        return (
            "<div class='for-sec'><div class='for-sec-title'>Injury report</div>"
            "<div class='for-muted'>No rostered players carry an injury designation.</div></div>"
        )
    items = [_injury_row_html(r) for r in rows]
    grid = f"<ul class='for-inj-list for-inj-grid'>{''.join(items)}</ul>"
    if not title:
        return grid
    return (
        "<div class='for-sec'><div class='for-sec-title'>Injury report</div>"
        f"{grid}</div>"
    )


def _injury_card_html(rows: list[dict]) -> str:
    """Compact injury summary for the Season Hub card."""
    if not rows:
        return ""
    items = []
    for r in rows[:3]:
        meta = " · ".join(
            p for p in (r.get("body"), r.get("return_label")) if p
        )
        action = (
            f"<span class='for-inj-act'>{html.escape(str(r['action']))}</span>"
            if r.get("action") else ""
        )
        items.append(
            "<li class='for-card-inj-row'>"
            f"<span class='for-inj-name'>{html.escape(str(r.get('name') or ''))}</span> "
            f"<span class='for-inj'>{html.escape(str(r.get('injury') or ''))}</span>"
            + (f" <span class='for-muted'>{html.escape(meta)}</span>" if meta else "")
            + action
            + "</li>"
        )
    more = ""
    if len(rows) > 3:
        more = (
            f"<span class='for-muted'>+{len(rows) - 3} more in the full report</span>"
        )
    return (
        "<div class='for-card-inj'><div class='for-card-inj-head'>"
        f"<span class='for-lbl'>Injuries</span>{more}</div>"
        f"<ul>{''.join(items)}</ul></div>"
    )


_POS_AVATAR_BG = {"QB": "#2563eb", "RB": "#1a7f4b", "WR": "#c2410c", "TE": "#7c3aed"}


def _pos_avatar(pos: str) -> str:
    p = html.escape(str(pos or "?").upper())
    bg = _POS_AVATAR_BG.get(str(pos or "").upper(), "#64748b")
    return f"<span class='for-avatar' style='background:{bg}'>{p}</span>"


def _pos_chip(pos: str) -> str:
    p = str(pos or "").upper()
    if p not in _POS_AVATAR_BG:
        return ""
    return f"<span class='for-pos-chip for-pos-{p.lower()}'>{html.escape(p)}</span>"


def _dot_pill(label: str, amber: bool = False) -> str:
    if not label:
        return ""
    cls = "for-dot for-dot-amber" if amber else "for-dot"
    return (
        f"<span class='for-pill'><span class='{cls}'></span>"
        f"{html.escape(label)}</span>"
    )


def _top_moves_html(data: dict, ai: dict) -> str:
    """Top Moves section: the deterministic ranked list when it exists,
    else the model's single top move after the code-level guard. The QB
    note (covered or real hole) renders under the list either way."""
    moves = data.get("top_moves") or []
    note = data.get("qb_note") or {}
    note_html = ""
    if note.get("lead"):
        note_html = (
            "<div class='for-covered'>"
            f"<strong>{html.escape(str(note['lead']))}</strong> "
            f"<span class='for-muted'>{html.escape(str(note.get('text') or ''))}</span></div>"
        )
    if moves:
        cards = []
        for i, m in enumerate(moves, start=1):
            why = (
                f"<div class='for-move-why'>{html.escape(str(m.get('why') or ''))}</div>"
                if m.get("why") else ""
            )
            cards.append(
                "<div class='for-move-card'>"
                f"<div class='for-move-rank'>{i}</div>"
                "<div class='for-move-main'>"
                f"<div class='for-move-line'>{html.escape(str(m.get('line') or ''))}</div>"
                f"{why}</div>"
                f"<span class='for-move-type'>{html.escape(str(m.get('type') or ''))}</span>"
                "</div>"
            )
        return (
            "<div class='for-sec'><div class='for-sec-title'>Top moves</div>"
            "<div class='for-sec-sub'>Ranked by what actually improves your starting lineup. "
            "Every add names its drop.</div>"
            f"<div class='for-moves-list'>{''.join(cards)}</div>{note_html}</div>"
        )
    raw_move = str(ai.get("top_move") or "")
    if raw_move and _ai_top_move_ok(raw_move, data):
        return (
            "<div class='for-sec'><div class='for-sec-title'>Top move</div>"
            f"<div class='for-card-move'><div>{html.escape(raw_move)}</div></div>"
            f"{note_html}</div>"
        )
    if note_html:
        return f"<div class='for-sec'>{note_html}</div>"
    return ""


def render_front_office_card_html(data: dict, ai: dict) -> str:
    """Condensed summary for the Season Hub card."""
    verdict = ai.get("verdict")
    headline = html.escape(str(ai.get("headline") or ""))
    _moves = data.get("top_moves") or []
    if _moves:
        top_move = html.escape(str(_moves[0].get("line") or ""))
    elif _ai_top_move_ok(str(ai.get("top_move") or ""), data):
        top_move = html.escape(str(ai.get("top_move") or ""))
    else:
        top_move = ""
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
            "<div class='for-move-main'>"
            "<span class='for-lbl'>Top move</span>"
            f"<div class='for-move-txt'>{top_move}</div></div>"
            "<span class='for-move-arrow' aria-hidden='true'>→</span></div>"
        )
    stamp = _verdict_stamp(verdict, size="sm")
    # Masthead (L1, approved 2026-10-14): stamp and headline share one row,
    # with the team/week line under the headline in place of a footer.
    sub = f"{team}" if team else ""
    if week_lbl:
        sub = f"{sub} · {week_lbl}" if sub else week_lbl
    mast_main = ""
    if headline or sub:
        mast_main = (
            "<div class='for-card-mast-main'>"
            + (f"<div class='for-card-headline'>{headline}</div>" if headline else "")
            + (f"<div class='for-card-sub'>{sub}</div>" if sub else "")
            + "</div>"
        )
    verdict_block = (
        f"<div class='for-card-verdict'>{stamp}{mast_main}</div>"
        if (stamp or mast_main) else ""
    )
    inj_html = _injury_card_html(data.get("injury_rows") or [])
    return f"""
    <div class='for-card-summary'>
      {verdict_block}
      {inj_html}
      <ul class='for-keys'>{keys_html}</ul>
      {move_html}
      <button type='button' class='for-view-full' id='forViewFullBtn'>View full report →</button>
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


def _swap_player_row(p: dict) -> str:
    inj = (
        f" <span class='for-inj'>{html.escape(str(p['injury']))}</span>"
        if p.get("injury") else ""
    )
    age = f" · age {p['age']}" if p.get("age") not in (None, "") else ""
    return (
        "<div class='for-swap-player'>"
        f"{_pos_avatar(p.get('position'))}"
        f"<span class='for-swap-name'>{html.escape(str(p.get('name') or ''))}</span>{inj} "
        f"{_pos_chip(p.get('position'))}"
        f"<span class='for-swap-meta'>{html.escape(age)}</span>"
        f"<span class='for-swap-val'>{safe_float(p.get('value')):g}</span>"
        "</div>"
    )


def _trade_card_html(t: dict, trade_notes: dict) -> str:
    """One priced deal as a Swap card (the design Kaedon picked): the
    trade itself is the visual, Get and Give boxes with a swap arrow
    between them, then one footer line carrying fairness, the why, and
    the post-trade depth result. No label-row walls."""
    get = t["gets"][0]
    note = str(trade_notes.get(get["id"]) or "")
    why = note or str(t.get("why_they_say_yes") or "")
    if t.get("urgent") and t.get("urgent_reason"):
        why = f"Urgent: {t['urgent_reason']}. {why}".strip()
    alt = t.get("waiver_alternative") or {}
    if alt.get("name"):
        why = (why + f" Free alternative: {alt['name']} is on waivers.").strip()
    meta_bits = [
        str(x) for x in (t.get("partner_record"), t.get("partner_direction")) if x
    ]
    meta_html = (
        f" <span class='for-muted'>({html.escape(' · '.join(meta_bits))})</span>"
        if meta_bits else ""
    )
    tier = str(get.get("tier") or "")
    tier_html = ""
    if tier == "buy_low":
        tier_html = " <span class='for-tier-tag'>Buy-low</span>"
    elif tier == "elite":
        tier_html = " <span class='for-tier-tag'>Elite</span>"
    depth_html = ""
    if t.get("depth_warnings"):
        depth_html = (
            "<div class='for-swap-depth'>"
            f"{html.escape(str(t['depth_warnings'][0]))}</div>"
        )
    elif t.get("depth_checked"):
        depth_html = (
            "<div class='for-swap-depth'>After the trade: no new hole created.</div>"
        )
    get_rows = "".join(_swap_player_row(g) for g in t["gets"])
    give_rows = "".join(_swap_player_row(g) for g in t["gives"])
    return (
        "<div class='for-swap-card'>"
        "<div class='for-swap-head'>"
        f"<div class='for-swap-who'>With <strong>{html.escape(str(t.get('partner') or ''))}</strong>"
        f"{meta_html}{tier_html}</div>"
        f"{_dot_pill(str(t.get('acceptance_label') or ''), amber=(str(t.get('acceptance_label') or '') == 'Long shot'))}"
        "</div>"
        "<div class='for-swap-grid'>"
        f"<div class='for-swap-side for-swap-get'><div class='for-swap-lbl'>You get</div>{get_rows}</div>"
        "<div class='for-swap-arrow'><span>&#8596;</span></div>"
        f"<div class='for-swap-side for-swap-give'><div class='for-swap-lbl'>You give</div>{give_rows}</div>"
        "</div>"
        "<div class='for-swap-foot'>"
        f"{_dot_pill(str(t.get('fairness_label') or ''))}"
        f"<span class='for-swap-why'>{html.escape(why)}</span>"
        f"<a class='for-analyze-lite' href='{html.escape(str(t.get('analyzer_url') or ''), quote=True)}'>Analyze &rarr;</a>"
        "</div>"
        f"{depth_html}"
        "</div>"
    )


def _trade_targets_html(targets: list[dict], trade_notes: dict) -> str:
    if not targets:
        return ""
    cards = [_trade_card_html(t, trade_notes) for t in targets]
    return (
        "<div class='for-sec'><div class='for-sec-title'>Trade targets</div>"
        "<div class='for-sec-sub'>Ranked by fairness and how likely they are to say yes, "
        "not by player value. Every deal is checked so it does not leave you short at a position.</div>"
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
    cards = [_potential_row_html(p) for p in potential]
    return (
        "<div class='for-sec'><div class='for-sec-title'>Potential trade targets</div>"
        "<div class='for-sec-sub'>No fair priced deal yet. Ranked by how gettable they are, "
        "with the honest gap to fair.</div>"
        f"<div class='for-targets'>{''.join(cards)}</div></div>"
    )


def _potential_row_html(p: dict) -> str:
    inj = (
        f" <span class='for-inj'>{html.escape(str(p['injury']))}</span>"
        if p.get("injury") else ""
    )
    pos = str(p.get("position") or "")
    rank = int(p.get("depth_rank") or 0)
    flex_n = int(p.get("flex_slots") or 0)
    flex_txt = f"{flex_n} flex spot{'s' if flex_n != 1 else ''}"
    if p.get("sole_starter"):
        why = (
            f"He is their only startable {pos}, so he does not move "
            f"without a {pos} going back."
        )
    elif p.get("is_surplus"):
        if flex_n:
            why = (
                f"{pos}{rank} on their team, outside their starting group "
                f"even with {flex_txt} filled. Startable depth at a position you need."
            )
        else:
            why = (
                f"{pos}{rank} on their team, behind their starters. "
                "Startable depth at a position you need."
            )
    elif rank:
        if flex_n and pos in ("RB", "WR", "TE"):
            why = (
                f"{pos}{rank} on their team. With {flex_txt}, he is in their "
                "starting group, so this takes a real package."
            )
        else:
            why = f"{pos}{rank} on their team. He starts for them, so this takes a real package."
    else:
        why = "Startable depth at a position you need."
    gap = int(p.get("gap_pct") or 0)
    gap_txt = ""
    if gap > 0:
        gap_txt = f" Your surplus package reaches about {gap}% of his value."
        if gap >= 75:
            gap_txt += " Close."
        elif gap < 65:
            gap_txt += " Not close enough to lead with."
    start = p.get("starting_point") or {}
    start_txt = ""
    if start.get("name") and gap > 0:
        start_txt = f" Start with {start['name']}"
        holes = p.get("partner_holes") or []
        if start.get("position") in holes:
            start_txt += f", where they are thin at {start['position']}"
        start_txt += "."
    attn = str(p.get("attainability") or "")
    build_btn = ""
    if p.get("build_url"):
        build_btn = (
            f"<a class='for-analyze for-prow-btn' "
            f"href='{html.escape(str(p['build_url']), quote=True)}'>Build around</a>"
        )
    return (
        "<div class='for-prow'>"
        f"{_pos_avatar(pos)}"
        "<div class='for-prow-main'>"
        f"<div class='for-prow-top'><span class='for-prow-name'>{html.escape(str(p.get('name') or ''))}</span>{inj} "
        f"{_pos_chip(pos)}"
        f"<span class='for-muted'>{safe_float(p.get('value')):g} · {html.escape(str(p.get('partner') or ''))}</span></div>"
        f"<div class='for-prow-sub'>{html.escape(why)}"
        f"<span class='for-muted'>{html.escape(gap_txt + start_txt)}</span></div>"
        "</div>"
        f"<div class='for-prow-side'>{_dot_pill(attn, amber=(attn == 'Reach'))}{build_btn}</div>"
        "</div>"
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
        if w.get("claim_guidance"):
            note_html += (
                f"<div class='for-pick-note'>{html.escape(str(w['claim_guidance']))}</div>"
            )
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


def _watch_text(data: dict) -> str:
    """The Watch line for the board's hold box: a questionable starter
    (value-based Starter role) at a non-QB position. QB questionables are
    already covered by the QB note; a questionable starter needs no move
    unless he is ruled out, and the fill-in is the best healthy bench
    player at the position."""
    roster_rows = data.get("roster_rows") or []
    for r in data.get("injury_rows") or []:
        if _inj_canonical(str(r.get("injury") or "")) != "QUESTIONABLE":
            continue
        pos = str(r.get("position") or "").upper()
        if not pos or pos == "QB" or r.get("role") != "Starter":
            continue
        fills = [
            b for b in roster_rows
            if str(b.get("position") or "").upper() == pos
            and str(b.get("id")) != str(r.get("id"))
            and not _is_serious_designation(b.get("injury"))
            and _inj_canonical(str(b.get("injury") or "")) != "QUESTIONABLE"
        ]
        fill_txt = ""
        if fills:
            fill = max(fills, key=lambda b: safe_float(b.get("value")))
            fill_txt = f" Your {fill.get('name')} is the fill-in if he sits."
        return (
            f"<strong>{html.escape(str(r.get('name') or ''))} ({pos}) questionable.</strong> "
            f"No {pos} move unless he is ruled out.{fill_txt}"
        )
    return ""


def _hold_watch_html(data: dict, ai: dict) -> str:
    """Hold / Watch / Alert rows under the decision board. The QB note
    renders as Hold when the room is covered and as an Alert when there
    is a real hole; the model's GM alert only appears here when it has
    something to say (it used to sit at the very bottom of the modal)."""
    rows = []
    note = data.get("qb_note") or {}
    if note.get("lead"):
        tag = "Hold" if note.get("covered") else "Alert"
        rows.append(
            "<div class='for-holdrow'>"
            f"<span class='for-holdtag'>{tag}</span>"
            f"<span><strong>{html.escape(str(note['lead']))}</strong> "
            f"{html.escape(str(note.get('text') or ''))}</span></div>"
        )
    watch = _watch_text(data)
    if watch:
        rows.append(
            "<div class='for-holdrow'>"
            f"<span class='for-holdtag'>Watch</span><span>{watch}</span></div>"
        )
    gm_alert = str(ai.get("gm_alert") or "")
    if gm_alert:
        rows.append(
            "<div class='for-holdrow'>"
            "<span class='for-holdtag'>Alert</span>"
            f"<span>{html.escape(gm_alert)}</span></div>"
        )
    if not rows:
        return ""
    return f"<div class='for-holdbox'>{''.join(rows)}</div>"


def _move_evidence_html(move: dict, data: dict, ai: dict) -> str:
    """The evidence that lives inside a board move (v3): each move
    carries the rows or cards its call rests on, so nothing below the
    board has to be re-read to act on it."""
    kind = move.get("kind")
    rows_by_id = {
        str(r.get("id") or ""): r for r in data.get("injury_rows") or []
    }
    if kind == "ir":
        r = rows_by_id.get(str(move.get("injury_id") or ""))
        if not r:
            return ""
        return f"<ul class='for-inj-list'>{_injury_row_html(r)}</ul>"
    if kind == "waiver":
        add, drop = move.get("add") or {}, move.get("drop") or {}
        notes = ai.get("waiver_notes") or {}
        note = str(notes.get(str(add.get("id") or "")) or "") or str(add.get("claim_guidance") or "")
        rank = f" · {html.escape(str(add['pos_rank_label']))}" if add.get("pos_rank_label") else ""
        team = f", {html.escape(str(add['team']))}" if add.get("team") else ""
        a_inj = (
            f" <span class='for-inj'>{html.escape(str(add['injury']))}</span>"
            if add.get("injury") else ""
        )
        a_stash = (
            " <span class='for-lbl-inline'>IR stash only</span>"
            if add.get("stash_only") else ""
        )
        note_html = f"<div class='for-pick-note'>{html.escape(note)}</div>" if note else ""
        items = [
            "<li class='for-pick'>"
            "<span class='for-pick-badge for-add'>+</span>"
            f"<div class='for-pick-body'><strong>{html.escape(str(add.get('name') or ''))}</strong>{a_inj}{a_stash} "
            f"<span class='for-muted'>{html.escape(str(add.get('position') or ''))}{team} · value {safe_float(add.get('value')):g}{rank}</span>"
            f"{note_html}</div></li>"
        ]
        if drop.get("name"):
            label = "Fallback drop" if move.get("dep") else "Drop"
            d_note = (
                "<div class='for-pick-note'>Only if the IR move is skipped "
                "and no bench spot is open.</div>" if move.get("dep") else ""
            )
            items.append(
                "<li class='for-pick'>"
                "<span class='for-pick-badge for-cut'>&minus;</span>"
                f"<div class='for-pick-body'><strong>{label}: {html.escape(str(drop.get('name') or ''))}</strong> "
                f"<span class='for-muted'>{html.escape(str(drop.get('position') or ''))} · value {safe_float(drop.get('value')):g}</span>"
                f"{d_note}</div></li>"
            )
        return f"<ul class='for-picklist'>{''.join(items)}</ul>"
    if kind == "trade":
        targets = data.get("trade_targets") or []
        idx = int(move.get("target_index") or 0)
        if idx >= len(targets):
            return ""
        out = _trade_card_html(targets[idx], ai.get("trade_notes") or {})
        pb = move.get("plan_b_index")
        if pb is not None and int(pb) < len(targets):
            t2 = targets[int(pb)]
            get2 = (t2.get("gets") or [{}])[0]
            pos2 = str(get2.get("position") or "")
            out += (
                "<div class='for-prow' style='margin-top:10px'>"
                f"{_pos_avatar(pos2)}"
                "<div class='for-prow-main'>"
                f"<div class='for-prow-top'><span class='for-prow-name'>Plan B: {html.escape(str(get2.get('name') or ''))}</span> "
                f"{_pos_chip(pos2)}"
                f"<span class='for-muted'>{safe_float(get2.get('value')):g} · {html.escape(str(t2.get('partner') or ''))}</span></div>"
                "<div class='for-prow-sub'>If the offer above is declined. "
                f"<span class='for-muted'>{html.escape(str(t2.get('why_they_say_yes') or ''))}</span></div>"
                "</div>"
                f"<div class='for-prow-side'>{_dot_pill(str(t2.get('acceptance_label') or ''), amber=(str(t2.get('acceptance_label') or '') == 'Long shot'))}"
                f"{_dot_pill(str(t2.get('fairness_label') or ''))}</div>"
                "</div>"
            )
        return out
    if kind == "potential":
        potential = data.get("potential_trade_targets") or []
        rows = [
            _potential_row_html(potential[i])
            for i in (move.get("potential_indices") or [0])
            if int(i) < len(potential)
        ]
        return f"<div class='for-targets'>{''.join(rows)}</div>" if rows else ""
    if kind == "lineup":
        ids = move.get("detail_injury_ids") or [move.get("injury_id")]
        items = [
            _injury_row_html(rows_by_id[str(i)])
            for i in ids if str(i) in rows_by_id
        ]
        if not items:
            return ""
        out = f"<ul class='for-inj-list for-inj-grid'>{''.join(items)}</ul>"
        opp = data.get("opponent_injuries") or {}
        if opp.get("entries"):
            out += (
                "<div class='for-opp-inj'><span class='for-lbl-inline'>Opponent missing</span> "
                f"{html.escape(str(opp.get('line') or ''))}</div>"
            )
        return out
    return ""


def _decision_board_html(data: dict, ai: dict) -> str:
    """The 'Do this first' board at the top of the modal (v3, approved
    2026-10-23): sequenced decision moves with timing pills, dependency
    tags, After lines, and their evidence in place. Returns '' when
    there are no decision moves, and the caller falls back to the
    legacy report layout."""
    moves = data.get("decision_moves") or []
    if not moves:
        return ""
    cards = []
    for m in moves:
        dep = (
            f"<span class='for-dep'>{html.escape(str(m['dep']))}</span>"
            if m.get("dep") else ""
        )
        why = (
            f"<div class='for-decide-why'>{html.escape(str(m['why']))}</div>"
            if m.get("why") else ""
        )
        after = (
            f"<div class='for-after'><strong>After this:</strong> "
            f"{html.escape(str(m['after']))}</div>"
            if m.get("after") else ""
        )
        timing = str(m.get("timing") or "")
        when_cls = "for-when" if timing in ("Before waivers", "Before Sunday") else "for-when soon"
        evidence = _move_evidence_html(m, data, ai)
        detail = ""
        if evidence:
            detail = (
                "<details class='for-detail' open>"
                "<summary><span>Evidence</span></summary>"
                f"<div class='for-detail-body'>{evidence}</div></details>"
            )
        cards.append(
            "<div class='for-decide-row'>"
            f"<div class='for-decide-rank'>{int(m.get('rank') or 0)}</div>"
            "<div class='for-decide-main'>"
            f"<div class='for-decide-line'>{html.escape(str(m.get('line') or ''))}{dep}</div>"
            f"{why}{after}</div>"
            "<div class='for-decide-side'>"
            f"<span class='{when_cls}'>{html.escape(timing)}</span>"
            "</div>"
            f"{detail}"
            "</div>"
        )
    n = len(cards)
    return (
        "<div class='for-decide' id='forBoard'>"
        "<div class='for-decide-head'><div class='for-decide-title'>Do this first</div>"
        f"<span class='for-pill'><span class='for-dot'></span>{n} move{'s' if n != 1 else ''}, in order</span></div>"
        "<div class='for-decide-sub'>In deadline order. The evidence for each move is inside it. "
        "Nothing below this board asks you to decide anything.</div>"
        f"<div class='for-decide-list'>{''.join(cards)}</div>"
        f"{_hold_watch_html(data, ai)}"
        "</div>"
    )


def _reference_html(data: dict, grades_html: str, roster_html: str,
                    injury_grid_html: str, opp_html: str) -> str:
    """The Reference section under the board (v3): grades in the open,
    the full injury report and roster behind collapsed bars. Nothing
    here asks for a decision."""
    inj_rows = data.get("injury_rows") or []
    roster_rows = data.get("roster_rows") or []
    starters_flagged = sum(1 for r in inj_rows if r.get("role") == "Starter")
    inj_summary = (
        f"{len(inj_rows)} player{'s' if len(inj_rows) != 1 else ''}"
        + (f", {starters_flagged} starter{'s' if starters_flagged != 1 else ''} flagged"
           if inj_rows else "")
        + ". The rows behind the board moves are inside those moves."
    )
    bars = (
        "<details class='for-collapse-wrap'>"
        "<summary class='for-collapse'><span><strong>Injury report</strong> "
        f"<span class='for-muted'>· {html.escape(inj_summary)}</span></span>"
        "<span class='for-act'>Show all</span></summary>"
        f"<div class='for-collapse-body'>{opp_html}{injury_grid_html}</div></details>"
        "<details class='for-collapse-wrap'>"
        "<summary class='for-collapse'><span><strong>Roster</strong> "
        f"<span class='for-muted'>· {len(roster_rows)} players</span></span>"
        "<span class='for-act'>Show table</span></summary>"
        f"<div class='for-collapse-body'>{roster_html}</div></details>"
    )
    return (
        "<div class='for-sec' id='forReference'>"
        "<div class='for-sec-head'><div class='for-sec-title'>Reference</div>"
        "<span class='for-xref soft'>Nothing to decide</span></div>"
        f"{grades_html}{bars}"
        "</div>"
    )


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
    top_move_html = _top_moves_html(data, ai)
    stamp = _verdict_stamp(verdict)
    chips_html = _hero_chips_html(week, rec, pct, data.get("trade_deadline"))

    # Decisions-first layout (v3, approved 2026-10-23): when the board
    # has moves, it leads the modal and carries its own evidence; the
    # standalone trade / waiver / injury sections and the bottom GM
    # alert are folded into it, and grades + roster drop to Reference.
    board_html = _decision_board_html(data, ai)
    if board_html:
        jump_html = (
            "<div class='for-jumpbar'><a href='#forBoard'>Do this first</a>"
            "<a href='#forLastWeek'>Since last week</a>"
            "<a href='#forReference'>Reference</a></div>"
        )
        changes_wrapped = (
            f"<div id='forLastWeek'>{changes_html}</div>" if changes_html else ""
        )
        reference_html = _reference_html(
            data, grades_html, roster_html,
            _injury_section_html(data.get("injury_rows") or [], title=False),
            opp_html,
        )
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
      {board_html}
      {jump_html}
      {changes_wrapped}
      {reference_html}
    </div>
    """

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
