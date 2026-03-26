from __future__ import annotations

import html
import os

from dashboard_services.ai.cache import build_ai_cache_key, load_cached_ai_text, save_cached_ai_text
from dashboard_services.ai.context_builders import (
    build_team_gm_context,
)
from dashboard_services.ai.prompts import generate_trade_ai_result
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


def get_team_gm_memo(ctx: dict, viewer_roster_id: str) -> str:
    team_ctx = build_team_gm_context(ctx, viewer_roster_id)
    if not team_ctx:
        return ""

    cache_key = build_ai_cache_key("gm_memo", team_ctx, "v2")
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
        html_out = render_team_ai_result(result)
    except Exception as e:
        print(f"[ai gm memo] fallback: {e}")
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
        result = generate_team_ai_result(team_ctx, mode="front_office")
        html_out = render_team_ai_result(result)
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
