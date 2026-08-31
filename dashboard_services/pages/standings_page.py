"""Standings page HTML builder.

Moved from app.py so the Flask monolith can keep shrinking. Helpers that still
live in app.py are lazy-imported inside the builder (request time).
"""
from __future__ import annotations

def build_standings_body(ctx: dict) -> str:
    from app import (  # noqa: E402  (lazy: avoids a circular import at module load)
        _standings_available_weeks, _standings_panels, _standings_week_selector,
    )

    # Value-blended power ranking (matches the Teams page "Power Rankings" tab)
    # for the live view; the week-selector re-renders with the performance
    # PowerScore, which is what past weeks can reconstruct faithfully.
    try:
        from dashboard_services.ai.context_builders import build_power_rankings_context
        _pr_teams = (build_power_rankings_context(ctx) or {}).get("teams") or []
    except Exception:
        _pr_teams = []

    panels = _standings_panels(ctx, power_rankings=_pr_teams)
    week_bar = _standings_week_selector(ctx, _standings_available_weeks(ctx))

    body = f"""
    {week_bar}
    <div class="standings-main two-col-standings">
      <div class="standings-col">
        <div class="card">
          <div class="card-tabs">
            <div class="tab-strip">
              <button class="tab-btn active" data-tab="standings">Standings</button>
              <button class="tab-btn" data-tab="details">Detailed Stats</button>
              <button class="tab-btn" data-tab="shares">Value Share</button>
            </div>
            <div class="tab-panels">
              <div class="tab-panel active" data-tab="standings">
                <div id="stStandingsInner">{panels['standings']}</div>
              </div>
              <div class="tab-panel" data-tab="details">
                <div id="stDetailsInner">{panels['details']}</div>
                <div class="footer">
                  Default sort: Win% ↓ then PF ↓. Click headers to sort.
                </div>
              </div>
              <div class="tab-panel standings-shares-panel" data-tab="shares">
                <div id="stSharesInner">{panels['shares']}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="standings-col">
        <div id="stPowerInner">{panels['power']}</div>
      </div>
    </div>
    <aside class="overview-sidebar">
      <div id="stSidebarInner">{panels['sidebar']}</div>
    </aside>
    """

    return body

