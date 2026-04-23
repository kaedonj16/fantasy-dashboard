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
from dashboard_services.providers.espn_api import safe_float

AI_ENABLED = os.getenv("AI_ENABLED", "true").lower() == "true"


def ai_available() -> bool:
    return AI_ENABLED and bool(os.getenv("OPENAI_API_KEY"))


def _wrap_text_html(text: str) -> str:
    return (
        "<div class='ai-copy'>"
        f"<pre style='white-space:pre-wrap;font:inherit;margin:0'>{text}</pre>"
        "</div>"
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
    except Exception as e:
        import traceback
        traceback.print_exc()
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
    except Exception as e:
        print(f"[ai front office] fallback: {e}")
        strong_positions = ", ".join(team_ctx.get("strong_positions") or []) or "None"
        weak_positions = ", ".join(team_ctx.get("weak_positions") or []) or "None"
        html_out = f"""
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
) -> str:
    team_ctx = build_team_gm_context(ctx, viewer_roster_id)
    if not team_ctx or not isinstance(team_ctx, dict):
        return ""

    viewer_side = (viewer_side or "a").lower().strip()

    viewer_gets = side_a if viewer_side == "a" else side_b
    viewer_gives = side_b if viewer_side == "a" else side_a

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
        return {
            "id": str(a.get("id") or ""),
            "name": a.get("name") or "Unknown",
            "position": str(a.get("position") or a.get("pos") or "?").upper(),
            "team": a.get("team") or "",
            "age": a.get("age"),
            "value": safe_float(a.get("value")),
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
                pass
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
            "market_delta": market_delta,
        },
    }

    # Build cache key for trade analysis
    cache_key = build_ai_cache_key("trade_analysis", payload, "v2")

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
    except Exception as e:
        print(f"[trade-ai] fallback: {e}")
        verdict = "ACCEPT" if market_delta > 40 else "DECLINE" if market_delta < -40 else "COUNTER"
        fallback = {
            "verdict": verdict,
            "summary": f"The trade is being judged for a {team_ctx.get('direction') or 'balanced'} roster profile. Market delta: {market_delta:.1f}.",
            "helps": ["The return may line up with your current roster direction."],
            "risks": ["The LLM call failed, so this is a simplified fallback."],
            "counter": "Try adjusting the pick side or a secondary asset if the deal feels close.",
            "confidence": "low",
        }
        html_out = render_trade_ai_html(fallback)
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

    cache_key = build_ai_cache_key("power_rankings", {"week": rankings_ctx.get("week"), "season": rankings_ctx.get("season"), "teams": [t["roster_id"] for t in teams]}, "v1")
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
        # Trim context for AI — only send what's needed
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
                    "direction": t["direction"],
                    "top_assets": [{"name": p["name"], "position": p["position"], "value": p["value"]} for p in (t.get("top_assets") or [])[:3]],
                }
                for t in teams
            ],
        }
        result = generate_power_rankings_result(ai_input)
        narratives = {r["roster_id"]: r["narrative"] for r in (result.get("rankings") or [])}
        momentums = {r["roster_id"]: r.get("momentum", "steady") for r in (result.get("rankings") or [])}
        html_out = _render_power_rankings_html_from_data(teams, narratives, momentums)
    except Exception as e:
        print(f"[power-rankings-ai] fallback: {e}")
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
          <div class="pr-rank">#{rank}</div>
          <div class="pr-body">
            <div class="pr-team-line">
              <span class="pr-team-name">{team_name}</span>
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
        {"roster_id": viewer_roster_id, "needs": suggestions_ctx.get("viewer_needs"), "surplus": suggestions_ctx.get("viewer_surplus")},
        "v6",
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
    except Exception as e:
        print(f"[trade-suggestions-ai] fallback: {e}")
        html_out = _render_trade_suggestions_fallback(suggestions_ctx)

    save_cached_ai_text(cache_key, html_out)
    return html_out


def _fmt_pick_label(pk: dict) -> str:
    season = pk.get("season", "")
    rnd    = pk.get("round", "")
    slot   = pk.get("slot")
    name   = pk.get("proj_name", "")
    pos    = pk.get("proj_pos", "")
    suffix = {1: "1st", 2: "2nd", 3: "3rd"}.get(int(rnd) if rnd else 0, f"{rnd}th")
    slot_str = f".{int(slot):02d}" if slot else ""
    base = f"{season} {suffix} Round Pick{slot_str}"
    if name:
        base += f" (proj. {name}, {pos})"
    return base


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
            reasoning = f"Package {send_names} to acquire {target_names} — converts surplus depth into an elite upgrade."
        else:
            title = f"Target: {pname}"
            send_part = f" — offer {send_names}" if send_names else ""
            reasoning = f"They have depth at {html.escape(', '.join(p.get('partner_surplus') or []))} — target {target_names}{send_part}."
        partner_rows += f"""
        <div class="suggestion-card">
          <div class="suggestion-title">{html.escape(title)}</div>
          <div class="suggestion-reasoning">{html.escape(reasoning)}</div>
          <div class="suggestion-urgency urgency-medium">medium priority</div>
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
