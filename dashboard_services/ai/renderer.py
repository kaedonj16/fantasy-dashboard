from __future__ import annotations

import html
import os

from dashboard_services.ai.cache import build_ai_cache_key, load_cached_ai_text, save_cached_ai_text
from dashboard_services.ai.context_builders import (
    build_team_gm_context,
    build_power_rankings_context,
    build_trade_suggestions_context,
    calculate_roster_grade,
    summarize_roster_players,
    build_model_value_lookup,
    detect_team_direction,
)
from dashboard_services.ai.prompts import (
    generate_trade_ai_result,
    generate_team_ai_result,
    generate_power_rankings_result,
    generate_trade_suggestions_result,
)
import logging

from dashboard_services.ai.client import AIRateLimitError, AIUnavailableError
from dashboard_services.providers.espn_api import safe_float
from dashboard_services.rank_medals import rank_mark

logger = logging.getLogger(__name__)

AI_ENABLED = os.getenv("AI_ENABLED", "true").lower() == "true"


def ai_available() -> bool:
    return AI_ENABLED and bool(os.getenv("OPENAI_API_KEY"))


def _ai_error_notice(reason: str = "") -> str:
    msg = "AI analysis temporarily unavailable"
    if reason:
        msg += f" ({reason})"
    return (
        f"<div class='ai-copy ai-error-notice'>"
        f"<span class='ai-error-icon'>&#x26A0;&#xFE0F;</span> {html.escape(msg)}. "
        f"Showing data-based summary below.</div>"
    )


def render_team_ai_result(result: dict, mode: str = "gm_memo") -> str:
    """
    Render the AI-generated team analysis result as HTML.
    mode: 'gm_memo' or 'front_office_briefing'
    """
    if mode == "gm_memo":
        team_identity = html.escape(str(result.get("team_identity") or ""))
        outlook = html.escape(str(result.get("outlook") or ""))
        strength = html.escape(str(result.get("strength") or ""))
        weakness = html.escape(str(result.get("weakness") or ""))
        next_move = html.escape(str(result.get("next_move") or ""))
        trade_posture = html.escape(str(result.get("trade_posture") or ""))
        verdict = html.escape(str(result.get("verdict") or "HOLD").upper())

        return f"""
        <div class="ai-copy">
          <p><strong>{team_identity}</strong></p>
          <p>{outlook}</p>
          <ul>
            <li><strong>Biggest strength:</strong> {strength}</li>
            <li><strong>Biggest weakness:</strong> {weakness}</li>
            <li><strong>Best next move:</strong> {next_move}</li>
          </ul>
          <p>{trade_posture}</p>
        </div>
        """
    else:  # front_office_briefing
        headline = html.escape(str(result.get("headline") or ""))
        posture = html.escape(str(result.get("posture") or ""))
        strongest_room = html.escape(str(result.get("strongest_room") or ""))
        weakest_room = html.escape(str(result.get("weakest_room") or ""))
        next_move = html.escape(str(result.get("next_move") or ""))
        gm_alert = html.escape(str(result.get("gm_alert") or ""))

        return f"""
        <div class="ai-copy">
          <p><strong>{headline}</strong></p>
          <p>{posture}</p>
          <ul>
            <li><strong>Strongest room:</strong> {strongest_room}</li>
            <li><strong>Weakest room:</strong> {weakest_room}</li>
            <li><strong>Most important next move:</strong> {next_move}</li>
          </ul>
          <p><strong>GM Alert:</strong> {gm_alert}</p>
        </div>
        """


def get_team_gm_memo(ctx: dict, viewer_roster_id: str) -> str:
    team_ctx = build_team_gm_context(ctx, viewer_roster_id)
    if not team_ctx:
        return ""

    cache_key = build_ai_cache_key("gm_memo", team_ctx, "v3")
    cached = load_cached_ai_text(cache_key)
    if cached:
        return cached

    if not ai_available():
        top_assets = ", ".join(p["name"] for p in (team_ctx.get("top_assets") or [])[:4]) or "None"
        html_out = f"""
        <div class="ai-copy">
          <p><strong>{html.escape(team_ctx['team_name'])}</strong> profiles as a <strong>{html.escape(team_ctx['direction'])}</strong> team.</p>
          <p>Top assets: {html.escape(top_assets)}.</p>
          <p>Record: {html.escape(str(team_ctx.get('record') or 'N/A'))} | PF: {safe_float(team_ctx.get('points_for')):.1f} | PA: {safe_float(team_ctx.get('points_against')):.1f}</p>
        </div>
        """
        save_cached_ai_text(cache_key, html_out)
        return html_out

    try:
        result = generate_team_ai_result(team_ctx, mode="gm_memo")
        html_out = render_team_ai_result(result, mode="gm_memo")
    except (AIRateLimitError, AIUnavailableError) as e:
        reason = "rate limited" if isinstance(e, AIRateLimitError) else "service unavailable"
        logger.warning("[ai gm_memo] %s: %s", reason, e)
        top_assets = ", ".join(p["name"] for p in (team_ctx.get("top_assets") or [])[:4]) or "None"
        html_out = _ai_error_notice(reason) + f"""
        <div class="ai-copy">
          <p><strong>{html.escape(team_ctx['team_name'])}</strong> profiles as a <strong>{html.escape(team_ctx['direction'])}</strong> team.</p>
          <p>Top assets: {html.escape(top_assets)}.</p>
          <p>Record: {html.escape(str(team_ctx.get('record') or 'N/A'))} | PF: {safe_float(team_ctx.get('points_for')):.1f} | PA: {safe_float(team_ctx.get('points_against')):.1f}</p>
        </div>
        """
    except Exception as e:
        logger.exception("[ai gm_memo] unexpected error: %s", e)
        top_assets = ", ".join(p["name"] for p in (team_ctx.get("top_assets") or [])[:4]) or "None"
        html_out = _ai_error_notice() + f"""
        <div class="ai-copy">
          <p><strong>{html.escape(team_ctx['team_name'])}</strong> profiles as a <strong>{html.escape(team_ctx['direction'])}</strong> team.</p>
          <p>Top assets: {html.escape(top_assets)}.</p>
          <p>Record: {html.escape(str(team_ctx.get('record') or 'N/A'))} | PF: {safe_float(team_ctx.get('points_for')):.1f} | PA: {safe_float(team_ctx.get('points_against')):.1f}</p>
        </div>
        """

    save_cached_ai_text(cache_key, html_out)
    return html_out


def get_front_office_briefing(ctx: dict, viewer_roster_id: str) -> str:
    team_ctx = build_team_gm_context(ctx, viewer_roster_id)
    if not team_ctx:
        return ""

    cache_key = build_ai_cache_key("front_office_briefing", team_ctx, "v2")
    cached = load_cached_ai_text(cache_key)
    if cached:
        return cached

    if not ai_available():
        strong_positions = ", ".join(team_ctx.get("strong_positions") or []) or "None"
        weak_positions = ", ".join(team_ctx.get("weak_positions") or []) or "None"
        html_out = f"""
        <div class="ai-copy">
          <p><strong>Strongest rooms:</strong> {html.escape(strong_positions)}</p>
          <p><strong>Weakest rooms:</strong> {html.escape(str(weak_positions))}</p>
          <p><strong>Direction:</strong> {html.escape(str(team_ctx.get("direction") or "balanced"))}</p>
        </div>
        """
        save_cached_ai_text(cache_key, html_out)
        return html_out

    try:
        result = generate_team_ai_result(team_ctx, mode="front_office_briefing")
        html_out = render_team_ai_result(result, mode="front_office_briefing")
    except (AIRateLimitError, AIUnavailableError) as e:
        reason = "rate limited" if isinstance(e, AIRateLimitError) else "service unavailable"
        logger.warning("[ai front_office] %s: %s", reason, e)
        strong_positions = ", ".join(team_ctx.get("strong_positions") or []) or "None"
        weak_positions = ", ".join(team_ctx.get("weak_positions") or []) or "None"
        html_out = _ai_error_notice(reason) + f"""
        <div class="ai-copy">
          <p><strong>Strongest rooms:</strong> {html.escape(str(strong_positions))}</p>
          <p><strong>Weakest rooms:</strong> {html.escape(str(weak_positions))}</p>
          <p><strong>Direction:</strong> {html.escape(str(team_ctx.get("direction") or "balanced"))}</p>
        </div>
        """
    except Exception as e:
        logger.exception("[ai front_office] unexpected error: %s", e)
        strong_positions = ", ".join(team_ctx.get("strong_positions") or []) or "None"
        weak_positions = ", ".join(team_ctx.get("weak_positions") or []) or "None"
        html_out = _ai_error_notice() + f"""
        <div class="ai-copy">
          <p><strong>Strongest rooms:</strong> {html.escape(str(strong_positions))}</p>
          <p><strong>Weakest rooms:</strong> {html.escape(str(weak_positions))}</p>
          <p><strong>Direction:</strong> {html.escape(str(team_ctx.get("direction") or "balanced"))}</p>
        </div>
        """

    save_cached_ai_text(cache_key, html_out)
    return html_out


def get_trade_ai_analysis(
        ctx: dict,
        viewer_roster_id: str,
        viewer_side: str,
        side_a: dict,
        side_b: dict,
        opponent_roster_id: str = "",
) -> str:
    team_ctx = build_team_gm_context(ctx, viewer_roster_id)
    if not team_ctx or not isinstance(team_ctx, dict):
        return ""

    viewer_side = (viewer_side or "a").lower().strip()

    viewer_gets = side_a if viewer_side == "a" else side_b
    viewer_gives = side_b if viewer_side == "a" else side_a

    # Injury data from Sleeper player objects stored in ctx
    players_full = ctx.get("players") or {}

    def clean_asset(a: dict) -> dict:
        if not isinstance(a, dict):
            return {
                "id": "",
                "name": "Unknown",
                "position": "?",
                "team": "",
                "age": None,
                "value": 0.0,
            }
        pid = str(a.get("id") or "")
        full_p = players_full.get(pid) or {}
        raw_status = (full_p.get("injury_status") or full_p.get("status") or "").strip().upper()
        injury_status = "" if raw_status in ("", "ACTIVE", "ACT") else raw_status
        injury_body_part = (full_p.get("injury_body_part") or "").strip().lower()
        return {
            "id": pid,
            "name": a.get("name") or "Unknown",
            "position": str(a.get("position") or a.get("pos") or "?").upper(),
            "team": a.get("team") or "",
            "age": a.get("age"),
            "value": safe_float(a.get("value")),
            "pos_rank_label": a.get("pos_rank_label") or "",
            "rank_change_7d": a.get("rank_change_7d"),
            "injury_status": injury_status,
            "injury_body_part": injury_body_part,
        }

    def summarize_pick_ids(pick_ids: list) -> dict:
        summary = {
            "count": 0,
            "firsts": 0,
            "seconds": 0,
            "thirds_plus": 0,
            "display": [],
        }
        for raw in pick_ids or []:
            pk = str(raw or "").strip()
            if not pk:
                continue
            summary["count"] += 1
            summary["display"].append(pk.replace("_", " "))
            try:
                parts = pk.split("_")
                if len(parts) >= 2:
                    rnd = int(parts[1])
                    if rnd == 1:
                        summary["firsts"] += 1
                    elif rnd == 2:
                        summary["seconds"] += 1
                    elif rnd >= 3:
                        summary["thirds_plus"] += 1
            except Exception:
                logger.debug("suppressed exception", exc_info=True)
        return summary

    def pos_totals(assets: list[dict]) -> dict[str, float]:
        out: dict[str, float] = {}
        for a in assets:
            pos = str(a.get("position") or "?").upper()
            out[pos] = out.get(pos, 0.0) + safe_float(a.get("value"))
        return {k: round(v, 1) for k, v in out.items()}

    gets_assets = [clean_asset(a) for a in (viewer_gets.get("assets") or viewer_gets.get("breakdown") or [])]
    gives_assets = [clean_asset(a) for a in (viewer_gives.get("assets") or viewer_gives.get("breakdown") or [])]

    gets_pick_summary = summarize_pick_ids(viewer_gets.get("pick_ids") or [])
    gives_pick_summary = summarize_pick_ids(viewer_gives.get("pick_ids") or [])

    gets_pos = pos_totals(gets_assets)
    gives_pos = pos_totals(gives_assets)

    market_delta = round(
        safe_float(viewer_gets.get("effective_total")) - safe_float(viewer_gives.get("effective_total")),
        1,
    )

    # Post-trade roster: remove traded-away players, add incoming players
    gives_ids = {a.get("id") for a in gives_assets}
    current_top = team_ctx.get("top_assets") or []
    remaining = [a for a in current_top if str(a.get("id") or "") not in gives_ids]
    for a in gets_assets:
        if a.get("position") not in ("PICK", None, "?"):
            remaining.append(a)
    remaining.sort(key=lambda x: safe_float(x.get("value")), reverse=True)
    post_trade_roster = [
        {"name": a.get("name"), "position": a.get("position"), "value": round(safe_float(a.get("value")), 1)}
        for a in remaining[:8]
    ]

    # Pick-to-prospect mapping using rookie ADP
    current_season = ctx.get("current_season") or ctx.get("season") or 2026
    num_teams = len(ctx.get("rosters") or []) or 12
    _roster_positions = [str(s).upper() for s in (ctx.get("roster_positions") or [])]
    is_sf = bool(ctx.get("is_sf") or any(s in {"SUPER_FLEX", "SFLEX"} for s in _roster_positions))
    # Starter format the analyst should reason from (QB value swings hard on this).
    _qb_slots = sum(1 for s in _roster_positions if s == "QB")
    _sf_slots = sum(1 for s in _roster_positions if s in {"SUPER_FLEX", "SFLEX"})
    league_format = {
        "qb_format": "Superflex/2QB" if is_sf else "1QB",
        "superflex": is_sf,
        "qb_starter_slots": _qb_slots + _sf_slots,
        "starting_lineup": _roster_positions or None,
        "note": (
            "Superflex/2QB league: quarterbacks carry premium value; weight QB assets and QB "
            "rookie picks up. The market values in this JSON already reflect the league format."
            if is_sf else
            "1QB league: standard single-QB scarcity. The market values in this JSON already "
            "reflect the league format."
        ),
    }
    all_pick_ids = list(viewer_gets.get("pick_ids") or []) + list(viewer_gives.get("pick_ids") or [])

    pick_prospects: dict = {}
    if all_pick_ids:
        try:
            from dashboard_services.adp_service import fetch_league_adp_from_db, build_model_adp_fallback
            adp_raw = fetch_league_adp_from_db(is_sf=is_sf, season=current_season, draft_type="rookie") or {}
            if not adp_raw:
                adp_raw = build_model_adp_fallback(is_sf=is_sf, season=current_season) or {}
            name_by_id = {str(p.get("id")): p.get("name") for p in (ctx.get("model_value_table") or []) if p.get("id")}
            prospects_sorted = sorted(
                [
                    {"sid": sid, "name": name_by_id.get(str(sid)) or "", "avg_pick": float(info.get("avg_pick") or info.get("adp_rank") or 999), "position": info.get("position") or ""}
                    for sid, info in adp_raw.items()
                    if name_by_id.get(str(sid))
                ],
                key=lambda x: x["avg_pick"],
            )
            for pk in all_pick_ids:
                parts = str(pk).split("_")
                if len(parts) < 3:
                    continue
                try:
                    yr, rnd = int(parts[0]), int(parts[1])
                except ValueError:
                    continue
                if yr != current_season or not prospects_sorted:
                    continue
                third = parts[2]
                try:
                    slot = int(third)
                except ValueError:
                    slot = {"early": max(1, round(num_teams * 0.2)), "mid": round(num_teams * 0.5), "late": round(num_teams * 0.8)}.get(third, round(num_teams / 2))
                overall = (rnd - 1) * num_teams + slot
                closest = min(prospects_sorted, key=lambda p: abs(p["avg_pick"] - overall))
                if abs(closest["avg_pick"] - overall) <= num_teams // 2 + 1:
                    pick_prospects[pk] = {"name": closest["name"], "position": closest["position"], "adp_overall": round(closest["avg_pick"], 1)}
        except Exception:
            logger.debug("suppressed exception", exc_info=True)

    # Opponent team context. Prefer the opponent the UI actually bound (reliable
    # even when the return is picks-only or the platform's roster ids don't match
    # the calculator's player ids); only fall back to inferring the partner from
    # the acquired players' owner when no explicit opponent was passed.
    opponent_ctx: dict = {}
    try:
        _roster_ids = {str(r.get("roster_id") or "") for r in (ctx.get("rosters") or [])}
        opp_roster_id = None
        _explicit = str(opponent_roster_id or "").strip()
        if _explicit and _explicit != str(viewer_roster_id) and _explicit in _roster_ids:
            opp_roster_id = _explicit
        else:
            opponent_side = side_b if viewer_side == "a" else side_a
            opp_player_ids = {str(a.get("id")) for a in (opponent_side.get("assets") or []) if a.get("id")}
            for roster in (ctx.get("rosters") or []):
                rid = str(roster.get("roster_id") or "")
                if rid == str(viewer_roster_id):
                    continue
                if opp_player_ids & {str(p) for p in (roster.get("players") or [])}:
                    opp_roster_id = rid
                    break
        if opp_roster_id:
            opp_team_ctx = build_team_gm_context(ctx, opp_roster_id)
            if opp_team_ctx:
                opponent_ctx = {
                    "team_name": opp_team_ctx.get("team_name"),
                    "direction": opp_team_ctx.get("direction"),
                    "record": opp_team_ctx.get("record"),
                    "top_assets": [{"name": a.get("name"), "position": a.get("position"), "value": round(safe_float(a.get("value")), 1)} for a in (opp_team_ctx.get("top_assets") or [])[:5]],
                    "strong_positions": opp_team_ctx.get("strong_positions") or [],
                    "weak_positions": opp_team_ctx.get("weak_positions") or [],
                }
    except Exception:
        logger.debug("suppressed exception", exc_info=True)

    payload = {
        "team_context": {
            "team_name": team_ctx.get("team_name"),
            "direction": team_ctx.get("direction"),
            "roster_health": team_ctx.get("roster_health"),
            "summary_flags": team_ctx.get("summary_flags") or [],
            "record": team_ctx.get("record"),
            "place": team_ctx.get("place"),
            "points_for": team_ctx.get("points_for"),
            "points_against": team_ctx.get("points_against"),
            "strong_positions": team_ctx.get("strong_positions") or [],
            "weak_positions": team_ctx.get("weak_positions") or [],
            "pick_summary": team_ctx.get("pick_summary") or {},
            "market_profile": team_ctx.get("market_profile") or {},
            "starter_profile": team_ctx.get("starter_profile") or {},
            "bench_profile": team_ctx.get("bench_profile") or {},
        },
        "trade": {
            "viewer_side": viewer_side,
            "viewer_gets": {
                "assets": gets_assets,
                "pick_ids": viewer_gets.get("pick_ids") or [],
                "effective_total": safe_float(viewer_gets.get("effective_total")),
                "position_totals": gets_pos,
                "pick_summary": gets_pick_summary,
            },
            "viewer_gives": {
                "assets": gives_assets,
                "pick_ids": viewer_gives.get("pick_ids") or [],
                "effective_total": safe_float(viewer_gives.get("effective_total")),
                "position_totals": gives_pos,
                "pick_summary": gives_pick_summary,
            },
            "post_trade_roster": post_trade_roster,
            "pick_prospects": pick_prospects,
            "market_delta": market_delta,
        },
        "league_format": league_format,
        "opponent_team": opponent_ctx or None,
    }

    # Build cache key for trade analysis
    cache_key = build_ai_cache_key("trade_analysis", payload, "v5")

    # Try to get from cache first
    cached = load_cached_ai_text(cache_key)
    if cached:
        return cached

    if not ai_available():
        verdict = "ACCEPT" if market_delta > 40 else "DECLINE" if market_delta < -40 else "COUNTER"
        fallback = {
            "verdict": verdict,
            "summary": f"This is a fallback analysis for a {team_ctx.get('direction') or 'balanced'} team. Market delta: {market_delta:.1f}.",
            "helps": ["Uses market value and roster context as baseline inputs."],
            "risks": ["AI is currently disabled, so this is not a model-generated explanation."],
            "counter": "Enable AI to get a fuller front-office style recommendation.",
            "confidence": "low",
        }
        html_out = render_trade_ai_html(fallback)
        save_cached_ai_text(cache_key, html_out)
        return html_out

    try:
        result = generate_trade_ai_result(payload)
        html_out = render_trade_ai_html(result)
        save_cached_ai_text(cache_key, html_out)
        return html_out
    except (AIRateLimitError, AIUnavailableError) as e:
        reason = "rate limited" if isinstance(e, AIRateLimitError) else "service unavailable"
        logger.warning("[trade-ai] %s: %s", reason, e)
        verdict = "ACCEPT" if market_delta > 40 else "DECLINE" if market_delta < -40 else "COUNTER"
        fallback = {
            "verdict": verdict,
            "summary": f"AI analysis {reason}. Market delta: {market_delta:.1f} - verdict based on value differential only.",
            "helps": ["Market value calculation is based on current dynasty rankings."],
            "risks": [f"AI is {reason}; this is a data-only estimate without roster context."],
            "counter": "Try again shortly for a full AI-powered front-office recommendation.",
            "confidence": "low",
        }
        html_out = _ai_error_notice(reason) + render_trade_ai_html(fallback)
        save_cached_ai_text(cache_key, html_out)
        return html_out
    except Exception as e:
        logger.exception("[trade-ai] unexpected error: %s", e)
        verdict = "ACCEPT" if market_delta > 40 else "DECLINE" if market_delta < -40 else "COUNTER"
        fallback = {
            "verdict": verdict,
            "summary": f"The trade is evaluated for a {team_ctx.get('direction') or 'balanced'} roster profile. Market delta: {market_delta:.1f}.",
            "helps": ["The return may line up with your current roster direction."],
            "risks": ["AI call failed; this is a simplified market-value estimate."],
            "counter": "Try adjusting the pick side or a secondary asset if the deal feels close.",
            "confidence": "low",
        }
        html_out = _ai_error_notice() + render_trade_ai_html(fallback)
        save_cached_ai_text(cache_key, html_out)
        return html_out


def render_trade_ai_html(result: dict) -> str:
    verdict = html.escape(str(result.get("verdict") or "COUNTER").upper())
    summary = html.escape(str(result.get("summary") or ""))
    helps = result.get("helps") or []
    risks = result.get("risks") or []
    counter = html.escape(str(result.get("counter") or ""))
    confidence = html.escape(str(result.get("confidence") or ""))

    helps_html = "".join(
        f"<li>{html.escape(str(x))}</li>" for x in helps[:4]) or "<li>No specific edge identified.</li>"
    risks_html = "".join(
        f"<li>{html.escape(str(x))}</li>" for x in risks[:4]) or "<li>No specific risk identified.</li>"

    counter_html = ""
    if counter:
        counter_html = f"""
        <div class="trade-ai-block">
          <div class="trade-ai-label">Advice</div>
          <div class="trade-ai-copy-line">{counter}</div>
        </div>
        """

    confidence_html = ""
    if confidence:
        confidence_html = f"""<div class="trade-ai-score">Confidence: {confidence}</div>"""

    return f"""
    <div class="ai-copy trade-ai-wrap">
      <div class="trade-ai-top">
        <div class="trade-ai-verdict trade-ai-verdict-{verdict.lower()}">{verdict}</div>
        {confidence_html}
      </div>

      <div class="trade-ai-block">
        <div class="trade-ai-label">Analysis:</div>
        <div class="trade-ai-copy-line">{summary}</div>
      </div>

      <div class="trade-ai-grid">
        <div class="trade-ai-block">
          <div class="trade-ai-label">Pros:</div>
          <ul class="trade-ai-list">{helps_html}</ul>
        </div>

        <div class="trade-ai-block">
          <div class="trade-ai-label">Cons:</div>
          <ul class="trade-ai-list">{risks_html}</ul>
        </div>
      </div>

      {counter_html}
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# Roster Grade
# ──────────────────────────────────────────────────────────────────────────────

def get_roster_grade(ctx: dict, viewer_roster_id: str) -> dict:
    """Return grade data for a single roster."""
    rosters = ctx.get("rosters") or []
    roster = next((r for r in rosters if str(r.get("roster_id")) == str(viewer_roster_id)), None)
    if not roster:
        return {"grade": "N/A", "score": 0, "win_window": "Unknown", "breakdown": {}}

    model_value_lookup = build_model_value_lookup(ctx.get("model_value_table") or [])
    players_summary = summarize_roster_players(
        roster=roster,
        players_index=ctx.get("players_index") or {},
        players_map=ctx.get("players_map") or {},
        model_value_lookup=model_value_lookup,
    )
    future_picks = ctx.get("picks_by_roster", {}).get(str(viewer_roster_id), [])
    return calculate_roster_grade(players_summary, future_picks)


def render_roster_grade_badge(grade_data: dict) -> str:
    grade = html.escape(str(grade_data.get("grade") or "?"))
    win_window = html.escape(str(grade_data.get("win_window") or ""))
    score = grade_data.get("score") or 0
    bd = grade_data.get("breakdown") or {}
    avg_age = bd.get("avg_age", 0)
    elite_count = bd.get("elite_count", 0)

    grade_class = "grade-a" if grade.startswith("A") else "grade-b" if grade.startswith("B") else "grade-c" if grade.startswith("C") else "grade-d"

    return f"""
    <div class="roster-grade-wrap">
      <div class="roster-grade-badge {grade_class}">{grade}</div>
      <div class="roster-grade-meta">
        <div class="roster-grade-window">{win_window}</div>
        <div class="roster-grade-score">Score: {score:.0f}/100 &bull; Age: {avg_age:.1f} &bull; Elite: {elite_count}</div>
      </div>
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# Power Rankings
# ──────────────────────────────────────────────────────────────────────────────

def get_power_rankings_html(ctx: dict) -> str:
    rankings_ctx = build_power_rankings_context(ctx)
    teams = rankings_ctx.get("teams") or []
    if not teams:
        return "<p>Not enough data for power rankings.</p>"

    cache_key = build_ai_cache_key("power_rankings", {"week": rankings_ctx.get("week"), "season": rankings_ctx.get("season"), "teams": [t["roster_id"] for t in teams]}, "v4")
    cached = load_cached_ai_text(cache_key)
    if cached:
        return cached

    # Build fallback narrative map
    fallback_narratives: dict[str, str] = {}
    for t in teams:
        direction = t.get("direction") or "balanced"
        top = (t.get("top_assets") or [{}])[0].get("name") or "their core"
        fallback_narratives[t["roster_id"]] = f"Led by {top}, this {direction} team sits at #{t['rank']}."

    if not ai_available():
        html_out = _render_power_rankings_html_from_data(teams, fallback_narratives)
        save_cached_ai_text(cache_key, html_out)
        return html_out

    try:
        # Trim context for AI - only send what's needed
        ai_input = {
            "season": rankings_ctx.get("season"),
            "week": rankings_ctx.get("week"),
            "teams": [
                {
                    "roster_id": t["roster_id"],
                    "team_name": t["team_name"],
                    "rank": t["rank"],
                    "wins": t["wins"],
                    "losses": t["losses"],
                    "pf": round(t["pf"], 1),
                    "win_window": t.get("win_window") or t.get("direction") or "balanced",
                    "avg_age": t.get("avg_age"),
                    "first_round_picks": t.get("first_round_picks", 0),
                    "position_strengths": t.get("position_strengths") or {},
                    "top_assets": [{"name": p["name"], "position": p["position"], "value": p["value"]} for p in (t.get("top_assets") or [])[:3]],
                }
                for t in teams
            ],
        }
        result = generate_power_rankings_result(ai_input)
        narratives = {r["roster_id"]: r["narrative"] for r in (result.get("rankings") or [])}
        momentums = {r["roster_id"]: r.get("momentum", "steady") for r in (result.get("rankings") or [])}
        html_out = _render_power_rankings_html_from_data(teams, narratives, momentums)
    except (AIRateLimitError, AIUnavailableError) as e:
        reason = "rate limited" if isinstance(e, AIRateLimitError) else "service unavailable"
        logger.warning("[power-rankings-ai] %s: %s", reason, e)
        html_out = _ai_error_notice(reason) + _render_power_rankings_html_from_data(teams, fallback_narratives)
    except Exception as e:
        logger.exception("[power-rankings-ai] unexpected error: %s", e)
        html_out = _render_power_rankings_html_from_data(teams, fallback_narratives)

    save_cached_ai_text(cache_key, html_out)
    return html_out


def _render_power_rankings_html_from_data(
        teams: list[dict],
        narratives: dict[str, str],
        momentums: dict[str, str] | None = None,
) -> str:
    if momentums is None:
        momentums = {}

    rows_html = ""
    for t in teams:
        rid = t["roster_id"]
        rank = t.get("rank", "?")
        team_name = html.escape(str(t.get("team_name") or ""))
        wins = t.get("wins", 0)
        losses = t.get("losses", 0)
        pf = t.get("pf", 0)
        narrative = html.escape(str(narratives.get(rid) or ""))
        momentum = (momentums.get(rid) or "steady").lower()
        momentum_icon = {"rising": "↑", "falling": "↓", "steady": "→"}.get(momentum, "→")
        momentum_class = f"momentum-{momentum}"

        rows_html += f"""
        <div class="pr-row">
          <div class="pr-rank">{rank_mark(rank)}</div>
          <div class="pr-body">
            <div class="pr-team-line">
              <span class="pr-team-name team-clickable" data-roster-id="{rid}" data-team-name="{team_name}">{team_name}</span>
              <span class="pr-record">{wins}-{losses}</span>
              <span class="pr-pf">{pf:.1f} PF</span>
              <span class="pr-momentum {momentum_class}">{momentum_icon}</span>
            </div>
            <div class="pr-narrative">{narrative}</div>
          </div>
        </div>
        """

    return f"""
    <div class="power-rankings-wrap">
      {rows_html}
    </div>
    """


# ──────────────────────────────────────────────────────────────────────────────
# Trade Suggestions
# ──────────────────────────────────────────────────────────────────────────────

def get_trade_suggestions_html(ctx: dict, viewer_roster_id: str) -> str:
    suggestions_ctx = build_trade_suggestions_context(ctx, viewer_roster_id)
    if not suggestions_ctx:
        return "<p>Could not build trade suggestions context.</p>"

    cache_key = build_ai_cache_key(
        "trade_suggestions",
        {
            "roster_id": viewer_roster_id,
            "needs": suggestions_ctx.get("viewer_needs"),
            "surplus": suggestions_ctx.get("viewer_surplus"),
            "ceiling_needs": suggestions_ctx.get("viewer_ceiling_needs"),
            "direction": suggestions_ctx.get("viewer_direction"),
        },
        "v10",
    )
    cached = load_cached_ai_text(cache_key)
    if cached:
        return cached

    if not ai_available() or (not suggestions_ctx.get("top_partners") and not suggestions_ctx.get("pick_trade_partners")):
        html_out = _render_trade_suggestions_fallback(suggestions_ctx)
        save_cached_ai_text(cache_key, html_out)
        return html_out

    try:
        result = generate_trade_suggestions_result(suggestions_ctx)
        html_out = _render_trade_suggestions_from_data(result.get("suggestions") or [])
    except (AIRateLimitError, AIUnavailableError) as e:
        reason = "rate limited" if isinstance(e, AIRateLimitError) else "service unavailable"
        logger.warning("[trade-suggestions-ai] %s: %s", reason, e)
        html_out = _ai_error_notice(reason) + _render_trade_suggestions_fallback(suggestions_ctx)
    except Exception as e:
        logger.exception("[trade-suggestions-ai] unexpected error: %s", e)
        html_out = _render_trade_suggestions_fallback(suggestions_ctx)

    save_cached_ai_text(cache_key, html_out)
    return html_out


def _fmt_pick_label(pk: dict) -> str:
    season = pk.get("season", "")
    rnd    = int(pk.get("round") or 0)
    slot   = pk.get("slot")
    if slot:
        return f"{season} {rnd}.{int(slot):02d}"
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(rnd, "th")
    return f"{season} {rnd}{suffix} (Mid)"


def _render_trade_suggestions_fallback(ctx: dict) -> str:
    needs = ctx.get("viewer_needs") or []
    surplus = ctx.get("viewer_surplus") or []
    partners = ctx.get("top_partners") or []
    pick_partners = ctx.get("pick_trade_partners") or []
    projected_picks = ctx.get("projected_picks") or []

    needs_str = html.escape(", ".join(needs) if needs else "None identified")
    surplus_str = html.escape(", ".join(surplus) if surplus else "None identified")

    partner_rows = ""
    for p in partners[:3]:
        pname = html.escape(str(p.get("team_name") or ""))
        targets = p.get("targets_they_have") or []
        sends = p.get("targets_viewer_sends") or []
        is_pkg = p.get("is_package_trade", False)
        target_names = html.escape(", ".join(t["name"] for t in targets[:2]) or "players at your needed positions")
        send_names = html.escape(", ".join(t["name"] for t in sends[:2]))
        if is_pkg and len(sends) >= 2:
            title = f"Package Deal: {pname}"
            reasoning = f"Package {send_names} to acquire {target_names} - converts surplus depth into an elite upgrade."
        else:
            title = f"Target: {pname}"
            send_part = f" - offer {send_names}" if send_names else ""
            reasoning = f"They have depth at {html.escape(', '.join(p.get('partner_surplus') or []))} - target {target_names}{send_part}."

        # Higher positional fit + a fairer deal = a more actionable suggestion.
        fairness = p.get("fairness") or 0
        urgency_cls, urgency_txt = ("urgency-low", "worth exploring")
        if p.get("match_score", 0) >= 4 and fairness >= 0.80:
            urgency_cls, urgency_txt = ("urgency-high", "strong fit")
        elif p.get("match_score", 0) >= 2 and fairness >= 0.72:
            urgency_cls, urgency_txt = ("urgency-medium", "good fit")
        partner_rows += f"""
        <div class="suggestion-card">
          <div class="suggestion-title">{html.escape(title)}</div>
          <div class="suggestion-reasoning">{html.escape(reasoning)}</div>
          <div class="suggestion-urgency {urgency_cls}">{urgency_txt}</div>
        </div>
        """

    # Pick-for-player suggestions
    for p in pick_partners[:3]:
        pname = html.escape(str(p.get("team_name") or ""))
        targets = p.get("targets_they_have") or []
        picks = p.get("picks_you_offer") or []
        target_names = html.escape(", ".join(t["name"] for t in targets[:2]))
        pick_labels = ", ".join(_fmt_pick_label(pk) for pk in picks[:2])
        if not target_names or not pick_labels:
            continue
        partner_rows += f"""
        <div class="suggestion-card">
          <div class="suggestion-title">Pick Trade: {pname}</div>
          <div class="suggestion-reasoning">Offer {html.escape(pick_labels)} to acquire {target_names}.</div>
          <div class="suggestion-urgency urgency-low">pick offer</div>
        </div>
        """

    if not partner_rows:
        partner_rows = "<p>No strong trade partners identified based on positional fit.</p>"

    return f"""
    <div class="trade-suggestions-wrap">
      <div class="suggestion-meta">
        <span>Needs: <strong>{needs_str}</strong></span>
        <span>Surplus: <strong>{surplus_str}</strong></span>
      </div>
      {partner_rows}
    </div>
    """


def _render_trade_suggestions_from_data(suggestions: list[dict]) -> str:
    if not suggestions:
        return "<p>No specific trade suggestions generated.</p>"

    cards = ""
    for s in suggestions:
        title = html.escape(str(s.get("title") or "Trade Idea"))
        partner = html.escape(str(s.get("partner_team") or ""))
        reasoning = html.escape(str(s.get("reasoning") or ""))
        urgency = (s.get("urgency") or "medium").lower()
        you_give = [html.escape(str(x)) for x in (s.get("you_give") or [])]
        you_get = [html.escape(str(x)) for x in (s.get("you_get") or [])]

        if not you_give or not you_get:
            continue  # skip incomplete suggestions (no named players on one side)

        give_html = "".join(f"<span class='suggestion-asset give'>{x}</span>" for x in you_give)
        get_html = "".join(f"<span class='suggestion-asset get'>{x}</span>" for x in you_get)

        trade_type = (s.get("trade_type") or "swap").lower()
        _type_labels = {
            "up_tier":   ("Up Tier", "#047857"),
            "down_tier": ("Down Tier", "#b45309"),
            "swap":      ("Swap", "#0369a1"),
        }
        type_label, type_color = _type_labels.get(trade_type, ("Swap", "#0369a1"))
        type_badge = (
            f"<span style='font-size:10px;font-weight:700;padding:2px 7px;border-radius:4px;"
            f"background:{type_color}18;color:{type_color};border:1px solid {type_color}40;'>"
            f"{html.escape(type_label)}</span>"
        )

        cards += f"""
        <div class="suggestion-card">
          <div class="suggestion-header">
            <div class="suggestion-title">{title}</div>
            <div style="display:flex;gap:5px;align-items:center;">{type_badge}<div class="suggestion-urgency urgency-{urgency}">{urgency}</div></div>
          </div>
          <div class="suggestion-partner">Partner: {partner}</div>
          <div class="suggestion-assets">
            <div class="suggestion-side"><div class="suggestion-side-label">You give</div>{give_html}</div>
            <div class="suggestion-arrow">⇄</div>
            <div class="suggestion-side"><div class="suggestion-side-label">You get</div>{get_html}</div>
          </div>
          <div class="suggestion-reasoning">{reasoning}</div>
        </div>
        """

    return f"""<div class="trade-suggestions-wrap">{cards}</div>"""
