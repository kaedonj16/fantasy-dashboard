from __future__ import annotations

from dashboard_services.ai.client import ai_enabled, generate_text
from dashboard_services.ai.cache import build_ai_cache_key, load_cached_ai_text, save_cached_ai_text
from dashboard_services.ai.context_builders import build_team_gm_context
from dashboard_services.ai.prompts import GM_MEMO_SYSTEM, build_gm_memo_prompt


def get_team_gm_memo(ctx: dict, viewer_roster_id: str) -> str:
    team_ctx = build_team_gm_context(ctx, viewer_roster_id)
    if not team_ctx:
        return "<p>Could not build GM memo for this roster.</p>"

    cache_key = build_ai_cache_key("gm_memo", team_ctx, prompt_version="v1")
    cached = load_cached_ai_text(cache_key)
    if cached:
        return cached

    if not ai_enabled():
        fallback = (
            f"<p><strong>{team_ctx['team_name']}</strong> is currently profiled as "
            f"<strong>{team_ctx['direction']}</strong>. "
            f"Top assets: {', '.join(p['name'] for p in team_ctx['top_assets'][:4])}.</p>"
        )
        save_cached_ai_text(cache_key, fallback)
        return fallback

    prompt = build_gm_memo_prompt(team_ctx)
    text = generate_text(
        system_prompt=GM_MEMO_SYSTEM,
        user_prompt=prompt,
        model="gpt-5-mini",
    )

    html = f"<div class='gm-memo-text'><pre style='white-space:pre-wrap;font:inherit;margin:0'>{text}</pre></div>"
    save_cached_ai_text(cache_key, html)
    return html