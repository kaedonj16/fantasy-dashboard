"""Standings page HTML builder.

Moved from app.py so the Flask monolith can keep shrinking. Helpers that still
live in app.py are lazy-imported inside the builder (request time).
"""
from __future__ import annotations

def build_standings_body(ctx: dict) -> str:
    from app import (  # noqa: E402  (lazy: avoids a circular import at module load)
        _standings_available_weeks, _standings_panels, _standings_week_selector,
    )

    # Live Standings Power Rankings card (in-season: all-play, PPG tie-break).
    # The week-selector re-renders with the performance PowerScore, which is
    # what past weeks can reconstruct faithfully.
    try:
        from dashboard_services.ai.context_builders import build_power_rankings_context
        _pr_teams = (build_power_rankings_context(ctx) or {}).get("teams") or []
    except Exception:
        _pr_teams = []

    panels = _standings_panels(ctx, power_rankings=_pr_teams)
    week_bar = _standings_week_selector(ctx, _standings_available_weeks(ctx))

    body = f"""
    <div class="std-rework">
    {week_bar}
    <div id="stTilesInner">{panels.get('tiles', '')}</div>
    <div class="standings-main two-col-standings">
      <div class="standings-col">
        <div class="card">
          <div class="card-tabs">
            <div class="tab-strip">
              <button class="tab-btn active" data-tab="standings">Standings</button>
              <button class="tab-btn" data-tab="shares">Value Share</button>
              <label class="st-detail-toggle" title="Show extra stat columns">
                <input type="checkbox" id="stDetailToggle">
                <span class="st-detail-switch" aria-hidden="true"></span>
                <span>Detailed</span>
              </label>
            </div>
            <div class="tab-panels">
              <div class="tab-panel active" data-tab="standings">
                <div id="stStandingsInner">{panels['standings']}</div>
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
    <div id="stSidebarInner" class="standings-insights-wrap">{panels['sidebar']}</div>
    </div>
    <script>
    (function() {{
      var KEY = 'br-standings-detailed';
      function table() {{ return document.querySelector('table[data-page="standings"]'); }}
      // Re-applied after week-selector swaps re-render the table, so the
      // toggle state survives "Standings through Week N" changes.
      window.applyStandingsDetail = function() {{
        var on = false;
        try {{ on = localStorage.getItem(KEY) === '1'; }} catch (e) {{}}
        var t = table();
        if (t) t.classList.toggle('show-detail', on);
        var cb = document.getElementById('stDetailToggle');
        if (cb) cb.checked = on;
      }};
      function init() {{
        var cb = document.getElementById('stDetailToggle');
        if (!cb || cb.__stDetailBound) return;
        cb.__stDetailBound = true;
        cb.addEventListener('change', function() {{
          try {{ localStorage.setItem(KEY, cb.checked ? '1' : '0'); }} catch (e) {{}}
          var t = table();
          if (t) t.classList.toggle('show-detail', cb.checked);
        }});
        window.applyStandingsDetail();
      }}
      if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
      else init();
    }})();
    </script>
    """

    return body
