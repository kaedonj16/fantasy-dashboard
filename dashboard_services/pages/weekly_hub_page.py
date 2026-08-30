"""Weekly Hub: matchups, scorers, scout, and optimal-lineup tabs.

Moved from app.py so the hub can be imported and tested without loading the
Flask monolith. Helpers that still live in app.py are lazy-imported inside
``build_weekly_hub_body`` (request time), matching teams_page.py.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_weekly_hub_body(ctx: dict) -> str:
    import json
    import pandas as pd
    from dashboard_services.matchups import (
        render_matchup_carousel_weeks,
        render_matchup_slide,
    )
    from dashboard_services.pages.scout_page import build_scout_body, platform_sign_in_hint
    from app import (  # noqa: E402  (lazy: avoids a circular import at module load)
        _compute_fpts_against,
        _scoring_format_from_settings,
        _games_scheduled_today,
        _render_weekly_highlights,
        build_optimal_body,
        render_weekly_top_scorers_for_week,
    )
    league_id = ctx["league_id"]
    platform = ctx["platform"]
    season = ctx["season"]  # viewed season
    rosters = ctx["rosters"]
    users = ctx["users"]
    df_weekly = ctx["df_weekly"]
    roster_map = ctx["roster_map"]
    players_map = ctx["players_map"]
    current_week = int(ctx.get("current_week") or 0)
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]
    proj_by_week = ctx["proj_by_week"]
    weeks = int(ctx["weeks"])
    statuses = ctx["statuses"]
    team_game_lookup = ctx["team_game_lookup"]
    matchups_by_week = ctx["matchups_by_week"]
    season_complete = bool(ctx.get("season_complete", False))
    offseason_mode = bool(ctx.get("offseason_mode", False))

    if (
            df_weekly is not None
            and not df_weekly.empty
            and "finalized" in df_weekly.columns
            and "week" in df_weekly.columns
    ):
        finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    else:
        finalized_df = pd.DataFrame()

    if not finalized_df.empty and "week" in finalized_df.columns:
        last_final_week = int(finalized_df["week"].max())
    else:
        last_final_week = 0  # no finalized weeks yet → show projections for current week

    max_week = max(1, weeks)

    def clamp_week(w: int) -> int:
        return max(1, min(max_week, int(w)))

    if season_complete or offseason_mode:
        default_week = clamp_week(last_final_week)
    else:
        default_week = clamp_week(current_week or 1)

    _hub_vid = str((ctx.get("viewer") or {}).get("viewer_roster_id") or "")
    default_matchups = sorted(
        matchups_by_week.get(default_week, []) or [],
        key=lambda m: 0 if _hub_vid and _hub_vid in (str((m.get("left") or {}).get("roster_id", "")),
                                                     str((m.get("right") or {}).get("roster_id", ""))) else 1,
    )

    # Pre-compute head-to-head records for each current-week matchup
    def _h2h_record(rid_a: str, rid_b: str) -> tuple[int, int]:
        """Return (a_wins, b_wins) across all completed weeks this season."""
        a_wins = b_wins = 0
        for wk, wk_matchups in matchups_by_week.items():
            if int(wk) >= default_week:
                continue  # only count past weeks
            for wm in wk_matchups:
                la = str((wm.get("left") or {}).get("roster_id", ""))
                ra = str((wm.get("right") or {}).get("roster_id", ""))
                if set([la, ra]) == set([rid_a, rid_b]):
                    lp = (wm.get("left") or {}).get("pts_total") or 0
                    rp = (wm.get("right") or {}).get("pts_total") or 0
                    if lp == 0 and rp == 0:
                        continue
                    if (la == rid_a and lp > rp) or (ra == rid_a and rp > lp):
                        a_wins += 1
                    else:
                        b_wins += 1
        return a_wins, b_wins

    for _m in default_matchups:
        _la = str((_m.get("left") or {}).get("roster_id", ""))
        _ra = str((_m.get("right") or {}).get("roster_id", ""))
        if _la and _ra:
            _aw, _bw = _h2h_record(_la, _ra)
            _m["h2h"] = {"left_wins": _aw, "right_wins": _bw}

    _fpts_against_weekly = _compute_fpts_against(
        season,
        scoring=_scoring_format_from_settings(ctx.get("scoring_settings")),
    )
    slides = [
        render_matchup_slide(
            season,
            m,
            default_week,
            last_final_week,
            status_by_pid=(statuses.get(default_week) or {}).get("statuses", {}) or {},
            projections=proj_by_week,
            players=players_index,
            teams=teams_index,
            team_game_lookup=team_game_lookup,
            fpts_against=_fpts_against_weekly,
            viewer_roster_id=_hub_vid,
        )
        for m in default_matchups
    ]
    slides_html = "".join(slides) if slides else "<div class='m-empty'>No matchups</div>"
    slides_by_week = {default_week: slides_html}

    matchup_html = render_matchup_carousel_weeks(
        slides_by_week,
        dashboard=False,
        active_week=default_week,
    )

    options = []
    for w in range(1, max_week + 1):
        sel = " selected" if w == default_week else ""
        options.append(f"<option value='{w}'{sel}>Week {w}</option>")
    week_select_html = "".join(options)

    top_scorers_html = render_weekly_top_scorers_for_week(
        league_id,
        df_weekly,
        roster_map,
        players_map,
        proj_by_week,
        rosters,
        default_week,
        users,
        platform,
        season,
        roster_positions=ctx.get("roster_positions") or [],
    )
    proj_by_roster = ctx.get("proj_by_roster") or {}
    highlights_html = _render_weekly_highlights(
        df_weekly, default_week,
        proj_by_roster=proj_by_roster,
        matchups_by_week=matchups_by_week,
    )

    proj_warn_html = ""
    if not proj_by_week.get("_available"):
        proj_warn_html = (
            "<div class='card' style='margin-bottom:12px;background:#fffbeb;border:1px solid #f59e0b;'>"
            "  <div class='card-body' style='padding:10px 14px;font-size:13px;color:#92400e;'>"
            "    <strong>Projections unavailable</strong> - projected scores can't be loaded right now. "
            "    Actual scores will still appear once games are final."
            "  </div>"
            "</div>"
        )

    main_panel_html = f"""
          <div class="week-main-panel active" data-week="{default_week}">
            {proj_warn_html}
            {top_scorers_html}
          </div>
    """
    side_panel_html = f"""
          <div class="week-side-panel active" data-week="{default_week}">
            {highlights_html}
          </div>
    """

    platform_js = json.dumps(platform)
    season_js = json.dumps(season)
    league_js = json.dumps(league_id)
    # Live/Redzone elements (LIVE badge, Redzone CTA, auto-refresh) only apply
    # when an actual game is scheduled for today. Check the current week's
    # schedule file for a game dated today, gated to the active season (mirrors
    # the Redzone page/nav, which is hidden in the offseason). Past seasons
    # never match today's date, so they resolve to no live UI automatically.
    games_today = (not offseason_mode) and _games_scheduled_today(season, current_week)
    games_today_js = json.dumps(bool(games_today))

    scout_tab_html = ""
    try:
        scout_tab_html = build_scout_body(ctx)
    except Exception:
        logger.debug("weekly: scout body build failed", exc_info=True)

    _scout_sign_in_hint = platform_sign_in_hint(platform)
    _scout_unavail = (
        "<div style='padding:20px;text-align:center;color:var(--muted);font-size:0.9em;'>"
        f"Scout report unavailable - sign in with {_scout_sign_in_hint} to see your opponent's breakdown."
        "</div>"
    )
    scout_panel_content = scout_tab_html if scout_tab_html else _scout_unavail

    optimal_panel_content = ""
    try:
        optimal_panel_content = build_optimal_body(ctx)
    except Exception:
        optimal_panel_content = ("<div style='padding:20px;text-align:center;color:var(--muted);font-size:0.9em;'>"
                                 "Optimal lineup data unavailable.</div>")

    return f"""
    <div class="page-layout weekly-hub">
      <main class="page-main">
        <div class="card">
          <div class="card-header-row">
            <div>
              <h2>Weekly Hub</h2>
            </div>
            <div class="week-selector">
              <select id="hubWeek" class="search">
                {week_select_html}
              </select>
            </div>
          </div>
        </div>

        <div id="weekly-rz-cta" class="weekly-rz-cta" style="display:none">
          <span class="weekly-rz-cta-dot"></span>
          <span class="weekly-rz-cta-text">NFL games are live right now, track your players in real time.</span>
          <a href="./redzone" class="weekly-rz-cta-link">Watch on Redzone →</a>
        </div>

        <div class="card-tabs weekly-hub-tabs" id="weeklyLeftTabs">
          <div class="tab-bar">
            <button class="tab-btn active" data-tab="matchups">Matchups</button>
            <button class="tab-btn" data-tab="scorers">Scorers</button>
            <button class="tab-btn" data-tab="scout">Scout</button>
            <button class="tab-btn" data-tab="optimal">Lineup</button>
          </div>
          <div class="tab-panels">
            <div class="tab-panel active" data-tab="matchups">
              <div class="matchups-shell">
                <div id="weeklyMatchupsContainer">
                  {matchup_html}
                </div>
                <div id="weeklyMatchupsLoading" class="matchups-loading hidden">
                  <div class="matchups-loading-inner">
                    <div class="matchups-spinner"></div>
                  </div>
                </div>
              </div>
            </div>
            <div class="tab-panel" data-tab="scorers">
              <div class="week-leaders-band">
                <div class="week-leaders-title">Week Leaders</div>
                <div class="week-side-panels">
                  {side_panel_html}
                </div>
              </div>
              <div class="week-main-panels">
                {main_panel_html}
              </div>
            </div>
            <div class="tab-panel" data-tab="scout">
              {scout_panel_content}
            </div>
            <div class="tab-panel" data-tab="optimal">
              {optimal_panel_content}
            </div>
          </div>
        </div>
      </main>
    </div>

<script>
// Weekly Hub is a single one-section-at-a-time tab switcher: Matchups (default) /
// Scorers / Scout / Lineup. The matchup cards live in the Matchups panel; the
// Week Leaders band + top scorers live in the Scorers panel. The week-change JS
// below finds .week-main-panels, .week-side-panels and #weeklyMatchupsContainer
// by selector regardless of which panel holds them, so re-renders keep working.

(function() {{
  var leagueId  = {league_js};
  var platform  = {platform_js};
  var season    = {season_js};

  var sel = document.getElementById('hubWeek');
  if (!sel) return;
  if (sel.__hubWeekBound) return;
  sel.__hubWeekBound = true;

  var matchupsContainer = document.getElementById('weeklyMatchupsContainer');
  var loadingOverlay    = document.getElementById('weeklyMatchupsLoading');
  var mainContainer = document.querySelector('.week-main-panels');
  var sideContainer = document.querySelector('.week-side-panels');

  function showLoading() {{
    if (loadingOverlay) loadingOverlay.classList.remove('hidden');
    sel.disabled = true;
  }}

  function hideLoading() {{
    if (loadingOverlay) loadingOverlay.classList.add('hidden');
    sel.disabled = false;
  }}

  var controller = null;
  var requestSeq = 0;

  sel.addEventListener('change', function() {{
    var w = String(this.value || '');
    if (!w) return;

    if (controller) {{
      try {{ controller.abort(); }} catch (e) {{}}
    }}
    controller = (window.AbortController ? new AbortController() : null);

    var mySeq = ++requestSeq;
    showLoading();

    var url =
      '/api/weekly-week?platform=' + encodeURIComponent(platform) +
      '&season=' + encodeURIComponent(season) +
      '&league_id=' + encodeURIComponent(leagueId) +
      '&week=' + encodeURIComponent(w);

    fetch(url, {{
      signal: controller ? controller.signal : undefined
    }})
      .then(function(res) {{
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      }})
      .then(function(data) {{
        if (mySeq !== requestSeq) return;
        if (!data || !data.ok) {{
          console.error('Failed to load week', w, data && data.error);
          return;
        }}

        if (mainContainer && typeof data.top_html === 'string') {{
          mainContainer.innerHTML =
            '<div class="week-main-panel active" data-week="' + w + '">' +
              data.top_html +
            '</div>';
        }}

        if (sideContainer && typeof data.highlights_html === 'string') {{
          sideContainer.innerHTML =
            '<div class="week-side-panel active" data-week="' + w + '">' +
              data.highlights_html +
            '</div>';
          if (window.brInitMoments) window.brInitMoments(sideContainer);
        }}

        if (matchupsContainer && typeof data.matchups_html === 'string') {{
          matchupsContainer.innerHTML = data.matchups_html;

          if (typeof window.resetMatchupCarousels === 'function') {{
            window.resetMatchupCarousels(matchupsContainer);
          }}
          if (typeof window.initPageRoot === 'function') {{
            window.initPageRoot(matchupsContainer);
          }}
          if (window.brInitMoments) window.brInitMoments(matchupsContainer);
        }}
      }})
      .catch(function(err) {{
        if (err && err.name === 'AbortError') return;
        console.error('Error fetching week', w, err);
      }})
      .finally(function() {{
        if (mySeq === requestSeq) hideLoading();
      }});
  }});
}})();

// ── Weekly: game-day auto-refresh ──────────────────────────────────────────────
(function() {{
  var sel = document.getElementById('hubWeek');
  if (!sel) return;

  // Live only when the current week's schedule actually has a game dated today
  // (computed server-side from the schedule file), not merely a typical NFL day.
  function liveActive() {{ return {games_today_js}; }}

  // Show/hide Redzone CTA banner
  var cta = document.getElementById('weekly-rz-cta');
  if (cta && liveActive()) cta.style.display = '';

  // Live badge next to "Weekly Hub" h2
  if (liveActive()) {{
    var h2 = document.querySelector('.card-header-row h2');
    if (h2 && !h2.querySelector('.weekly-live-badge')) {{
      var badge = document.createElement('span');
      badge.className = 'weekly-live-badge';
      badge.textContent = 'LIVE';
      h2.appendChild(badge);
    }}
  }}

  // Auto-refresh the week every 60 s only when a game is happening today
  if (!liveActive()) return;
  var _autoRefreshSeq = 0;
  setInterval(function() {{
    var w = String(sel.value || '');
    if (!w) return;
    var mySeq = ++_autoRefreshSeq;
    var url = '/api/weekly-week?platform=' + encodeURIComponent({platform_js}) +
      '&season=' + encodeURIComponent({season_js}) +
      '&league_id=' + encodeURIComponent({league_js}) +
      '&week=' + encodeURIComponent(w);
    fetch(url).then(function(res) {{
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.json();
    }}).then(function(data) {{
      if (!data || !data.ok || mySeq !== _autoRefreshSeq) return; // stale or error
      var mainContainer = document.querySelector('.week-main-panels');
      var sideContainer = document.querySelector('.week-side-panels');
      var matchupsContainer = document.getElementById('weeklyMatchupsContainer');
      if (mainContainer && typeof data.top_html === 'string') {{
        mainContainer.innerHTML = '<div class="week-main-panel active" data-week="' + w + '">' + data.top_html + '</div>';
      }}
      if (sideContainer && typeof data.highlights_html === 'string') {{
        sideContainer.innerHTML = '<div class="week-side-panel active" data-week="' + w + '">' + data.highlights_html + '</div>';
        if (window.brInitMoments) window.brInitMoments(sideContainer);
      }}
      if (matchupsContainer && typeof data.matchups_html === 'string') {{
        var _applyMatchups = function() {{
          matchupsContainer.innerHTML = data.matchups_html;
          if (typeof window.resetMatchupCarousels === 'function') window.resetMatchupCarousels(matchupsContainer);
          if (typeof window.initPageRoot === 'function') window.initPageRoot(matchupsContainer);
          if (window.brInitMoments) window.brInitMoments(matchupsContainer);
        }};
        // Flash any matchup score that moved since the last refresh (green up /
        // red down); falls back to a plain swap under reduced motion.
        if (window.brFlashUpdates) window.brFlashUpdates(matchupsContainer, '.m-score-val', _applyMatchups);
        else _applyMatchups();
      }}
    }}).catch(function() {{}});
  }}, 60000);
}})();

// Activate a weekly left-tab by name and switch its panel
function wkActivateTab(tab) {{
  var container = document.getElementById('weeklyLeftTabs');
  if (!container) return;
  container.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  container.querySelectorAll('.tab-panel').forEach(function(p) {{ p.classList.remove('active'); }});
  var btn   = container.querySelector('.tab-btn[data-tab="' + tab + '"]');
  var panel = container.querySelector('.tab-panel[data-tab="' + tab + '"]');
  if (btn)   btn.classList.add('active');
  if (panel) panel.classList.add('active');
}}

// Activate left tab from ?tab= query param (e.g. ?tab=scout, ?tab=optimal)
(function() {{
  var tabParam = new URLSearchParams(window.location.search).get('tab');
  if (!tabParam) return;
  var container = document.getElementById('weeklyLeftTabs');
  if (!container) return;
  var btn = container.querySelector('.tab-btn[data-tab="' + tabParam + '"]');
  if (!btn) return;
  container.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
  container.querySelectorAll('.tab-panel').forEach(function(p) {{ p.classList.remove('active'); }});
  btn.classList.add('active');
  var panel = container.querySelector('.tab-panel[data-tab="' + tabParam + '"]');
  if (panel) panel.classList.add('active');
}})();

// Desktop layout (>=1100px): restore the pre-tab-switcher arrangement — the
// Scorers/Scout/Lineup tabs in a left column, the matchup preview as the wide
// right column, and Week Leaders in a right-hand "Weekly Tools" aside. Below
// 1100px it stays the single 4-tab card (the mobile layout). Nodes are moved,
// not duplicated, so the week-change / live-refresh JS keeps finding them.
(function() {{
  var tabs = document.getElementById('weeklyLeftTabs');
  if (!tabs || tabs.__wkReflow) return;
  tabs.__wkReflow = true;
  var pageLayout = tabs.closest('.page-layout');
  var main = tabs.closest('.page-main');
  if (!pageLayout || !main) return;
  var mq = window.matchMedia('(min-width: 1100px)');

  function toDesktop() {{
    if (tabs.__mode === 'desktop') return;
    var bar   = tabs.querySelector('.tab-bar');
    var mPanel = tabs.querySelector('.tab-panel[data-tab="matchups"]');
    var mcEl  = document.getElementById('weeklyMatchupsContainer');
    var shell = mcEl ? mcEl.closest('.matchups-shell') : null;
    var band  = document.querySelector('.week-leaders-band');
    if (!bar || !shell) return;
    var two = document.createElement('div');
    two.className = 'standings-main two-col-standings wk-desktop-two';
    var leftCol = document.createElement('div');  leftCol.className = 'standings-col';
    var rightCol = document.createElement('div'); rightCol.className = 'standings-col';
    main.insertBefore(two, tabs);
    leftCol.appendChild(tabs);          // tabs card -> left column
    rightCol.appendChild(shell);        // matchup preview -> wide right column
    two.appendChild(leftCol);
    two.appendChild(rightCol);
    if (band) {{
      var aside = document.createElement('aside');
      aside.className = 'page-sidebar wk-desktop-aside';
      aside.setAttribute('data-sidebar-label', 'Weekly Tools');
      aside.appendChild(band);          // Week Leaders -> right aside
      pageLayout.appendChild(aside);
    }}
    var mBtn = bar.querySelector('.tab-btn[data-tab="matchups"]');
    if (mBtn) mBtn.style.display = 'none';
    wkActivateTab('scorers');
    tabs.classList.add('wk-desktop');
    tabs.__mode = 'desktop';
  }}

  function toMobile() {{
    if (tabs.__mode === 'mobile') return;
    var bar    = tabs.querySelector('.tab-bar');
    var mPanel = tabs.querySelector('.tab-panel[data-tab="matchups"]');
    var sPanel = tabs.querySelector('.tab-panel[data-tab="scorers"]');
    var mcEl   = document.getElementById('weeklyMatchupsContainer');
    var shell  = mcEl ? mcEl.closest('.matchups-shell') : null;
    var band   = document.querySelector('.week-leaders-band');
    if (shell && mPanel) mPanel.appendChild(shell);           // matchup -> back in its tab
    if (band && sPanel) {{
      var wmp = sPanel.querySelector('.week-main-panels');
      if (wmp) sPanel.insertBefore(band, wmp); else sPanel.appendChild(band);
    }}
    var two = main.querySelector('.wk-desktop-two');
    if (two) {{ main.insertBefore(tabs, two); main.removeChild(two); }}
    var aside = pageLayout.querySelector('.wk-desktop-aside');
    if (aside) pageLayout.removeChild(aside);
    if (bar) {{
      var mBtn = bar.querySelector('.tab-btn[data-tab="matchups"]');
      if (mBtn) mBtn.style.display = '';
    }}
    wkActivateTab('matchups');
    tabs.classList.remove('wk-desktop');
    tabs.__mode = 'mobile';
  }}

  function apply() {{ if (mq.matches) toDesktop(); else toMobile(); }}
  apply();
  if (mq.addEventListener) mq.addEventListener('change', apply);
  else if (mq.addListener) mq.addListener(apply);
}})();
</script>
"""
