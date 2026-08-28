"""In-season dashboard HTML builder.

Moved from app.py so the Flask monolith can keep shrinking. Helpers that still
live in app.py are lazy-imported inside the builder (request time).
"""
from __future__ import annotations

def build_dashboard_body(ctx: dict) -> str:
    from app import (  # noqa: E402  (lazy: avoids a circular import at module load)
        _render_do_next_waiver_card,
        _compute_fpts_against,
        _owner_to_rid_map,
        _playoff_sim_cached,
        _playoff_tile_from_cache,
        _render_bench_check,
        _render_season_review_card,
        _render_usage_movers,
        _roster_moves_alert_html,
        _section_title_link,
        _standings_movement,
        _trade_window_card_html,
        _viewer_lineup_alert_html,
        build_teams_overview,
        compute_awards_season,
        compute_win_prob,
        get_model_value_table_cached,
        get_team_gm_memo,
        has_premium_for_viewer,
        has_request_context,
        html,
        logger,
        render_awards_section,
        render_matchup_carousel_weeks,
        render_matchup_slide,
        render_standings_compact,
        render_dashboard_teams_sidebar,
        session,
        url_for,
    )

    league_id = ctx["league_id"]
    season = ctx["current_season"]
    platform = ctx.get("platform", "sleeper")
    rosters = ctx["rosters"]
    users = ctx["users"]
    current_week = ctx["current_week"]
    players_map = ctx["players_map"]
    df_weekly = ctx["df_weekly"]
    team_stats = ctx["team_stats"]
    players_index = ctx["players_index"]
    teams_index = ctx["teams_index"]
    statuses = ctx["statuses"]
    proj_by_week = ctx["proj_by_week"]
    matchups_by_week = ctx["matchups_by_week"]
    picks_by_roster = ctx["picks_by_roster"]
    team_game_lookup = ctx["team_game_lookup"]

    viewer = ctx.get("viewer") or {}
    viewer_roster_id = viewer.get("viewer_roster_id")

    gm_memo_html = ""
    _fo_premium = False
    if viewer_roster_id and has_request_context():
        _fo_premium = has_premium_for_viewer(
            session.get("viewer_username"), session.get("viewer_user_id"),
            league_id, platform, season,
        )
        if _fo_premium:
            try:
                gm_memo_html = get_team_gm_memo(ctx, str(viewer_roster_id))
            except Exception:
                logger.debug("dashboard: gm memo failed", exc_info=True)

    standings_html = render_standings_compact(
        team_stats, movement=_standings_movement(df_weekly),
        owner_to_rid=_owner_to_rid_map(roster_map=ctx.get("roster_map"), df_weekly=df_weekly),
    )
    usage_movers_html = _render_usage_movers(ctx, viewer_roster_id)
    lineup_alert_html = _viewer_lineup_alert_html(ctx, viewer_roster_id)
    roster_moves_html = _roster_moves_alert_html(ctx, viewer_roster_id)
    trade_window_html = _trade_window_card_html(ctx, viewer_roster_id)
    season_review_html = _render_season_review_card(ctx, viewer_roster_id, df_weekly, team_stats)

    finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    if not finalized_df.empty:
        last_final_week = int(finalized_df["week"].max())
    else:
        last_final_week = 0  # no finalized weeks yet → show projections for current week

    bench_check_html = _render_bench_check(ctx, viewer_roster_id, last_final_week)

    # Merged waiver card (preview + show all) lives in the Actions queue only.
    do_next_waiver_html = ""
    try:
        _wv_table = list(get_model_value_table_cached() or []) or (ctx.get("model_value_table") or [])
        do_next_waiver_html = _render_do_next_waiver_card(
            ctx,
            _wv_table,
            platform=platform,
            season=season,
            league_id=league_id,
        )
    except Exception:
        logger.debug("dashboard waiver card failed", exc_info=True)

    _fpts_against_dash = _compute_fpts_against(season)
    _dash_vid = str(viewer_roster_id or "")
    _dash_matchups = sorted(
        matchups_by_week.get(current_week, []),
        key=lambda m: 0 if _dash_vid and _dash_vid in (str((m.get("left") or {}).get("roster_id", "")),
                                                       str((m.get("right") or {}).get("roster_id", ""))) else 1,
    )
    slides = [
        render_matchup_slide(
            season,
            m,
            current_week,
            last_final_week,
            status_by_pid=statuses[current_week].get("statuses", {}),
            projections=proj_by_week,
            players=players_index,
            teams=teams_index,
            team_game_lookup=team_game_lookup,
            fpts_against=_fpts_against_dash,
        )
        for m in _dash_matchups
    ]
    slides_by_week = {current_week: "".join(slides)}
    matchup_html = render_matchup_carousel_weeks(
        slides_by_week,
        dashboard=True,
        active_week=current_week,
    )

    awards = compute_awards_season(finalized_df, players_map, league_id, platform, season, users, rosters)
    awards_html = render_awards_section(awards)

    teams_ctx = build_teams_overview(
        rosters=rosters,
        users_list=users,
        picks_by_roster=picks_by_roster,
        players=players_map,
        players_index=players_index,
        teams_index=teams_index,
        platform=platform,
    )
    ctx["model_value_table"] = list(get_model_value_table_cached() or []) or (ctx.get("model_value_table") or [])
    teams_sidebar_html, teams_tab_label = render_dashboard_teams_sidebar(
        ctx, teams_ctx, filled_label="Team Values",
    )

    gm_card_html = ""
    if gm_memo_html:
        gm_card_html = f"""
        <div class="card gm-card">
          <div class="card-header">
            <h2>Front Office Report</h2>
            <div class="subtle-label">{viewer.get("viewer_team_name") or "Your Team"}</div>
          </div>
          <div class="card-body">
            {gm_memo_html}
          </div>
        </div>
        """
    elif viewer_roster_id:
        # Free users (and premium when AI is down) get the same Generate control
        # as the offseason hub. The button is PRO-gated in JS and /api/gm-memo.
        gm_card_html = f"""
        <div class="card gm-card">
          <div class="card-header">
            <h2>Front Office Report</h2>
            <div class="subtle-label">{viewer.get("viewer_team_name") or "Your Team"}</div>
            <button type="button" id="generateGmMemoBtn" class="recap-generate-btn"
                    data-league-id="{html.escape(str(league_id))}"
                    data-season="{html.escape(str(season))}"
                    data-platform="{html.escape(str(platform))}"
                    data-viewer-roster-id="{html.escape(str(viewer_roster_id))}">
              Generate Report
            </button>
          </div>
          <div class="card-body">
            <div class="otc-ai-empty" id="gm-memo-empty">
              <div class="otc-ai-empty-sub">
                Get personalized analysis on your roster, trade targets, and standings.
              </div>
            </div>
            <div class="otc-ai-empty" id="gm-memo-loading" style="display:none;">
              <div class="otc-ai-empty-title">Analyzing Your Roster...</div>
              <div class="otc-ai-empty-sub">
                <div class="loading-spinner" style="margin: 10px auto; width: 30px; height: 30px; border: 3px solid var(--border); border-radius: 50%; border-top-color: var(--accent); animation: spin 1s linear infinite; border-right-color: transparent;"></div>
              </div>
            </div>
            <div id="gm-memo-result" style="display:none;"></div>
          </div>
        </div>
        """

    # ---- Hero stat cards — mirror the offseason hub's hero card ----
    from utils.format import ordinal as _dash_ord

    _hero_cards: list = []
    try:
        _hs = team_stats.copy()
        _hs = _hs.sort_values(by=["Wins", "PF", "PA"], ascending=[False, False, True]).reset_index(drop=True)
        _hs["Rank"] = _hs.index + 1
        _n_teams = len(_hs)
        _vname = str(viewer.get("viewer_team_name") or "")
        _vmask = (_hs["owner"].astype(str) == _vname) if _vname else None
        if _vname and _vmask is not None and bool(_vmask.any()):
            _vrow = _hs[_vmask].iloc[0]
            _rec = f"{int(_vrow['Wins'])}-{int(_vrow['Losses'])}"
            if int(_vrow.get("Ties", 0) or 0):
                _rec += f"-{int(_vrow['Ties'])}"
            _hero_cards.append(("Your record", _rec, _dash_ord(int(_vrow['Rank']))))
            _pf = _hs.sort_values("PF", ascending=False).reset_index(drop=True)
            _pf_rank = int(_pf.index[_pf["owner"].astype(str) == _vname][0]) + 1
            _hero_cards.append(("Points for", f"{float(_vrow['PF']):.0f}", _dash_ord(_pf_rank)))
            _streak = str(_vrow.get("Streak") or "").strip()
            # Win probability for the viewer's matchup, from the same model the
            # matchup slides' win bar uses. Only for a live/upcoming week.
            _win_sub = ""
            try:
                if _dash_matchups and current_week > last_final_week:
                    _m0 = _dash_matchups[0]
                    _l0 = _m0.get("left") or {}
                    _r0 = _m0.get("right") or {}
                    if _dash_vid and _dash_vid in (str(_l0.get("roster_id", "")), str(_r0.get("roster_id", ""))):
                        _wp_proj = (proj_by_week.get(current_week) or {}).get("projections") or {}
                        _wp_status = (statuses.get(current_week) or {}).get("statuses", {})
                        _wp = compute_win_prob(_l0, _r0, _wp_status, _wp_proj)
                        if str(_r0.get("roster_id", "")) == _dash_vid:
                            _wp = 1.0 - _wp
                        _win_sub = f"{round(_wp * 100)}% to win"
            except Exception:
                logger.debug("hero win prob failed", exc_info=True)
            if _win_sub and _streak:
                _tw_sub = f"{_streak} streak, {_win_sub}"
            elif _win_sub:
                _tw_sub = _win_sub
            elif _streak:
                _tw_sub = f"{_streak} streak"
            else:
                _tw_sub = f"{len(_dash_matchups)} matchups"
            _hero_cards.append(("This week", f"Week {current_week}", _tw_sub))
        else:
            _top = _hs.iloc[0]
            _hero_cards.append(("League leader", str(_top["owner"]),
                                f"{int(_top['Wins'])}-{int(_top['Losses'])} record"))
            _hi = _hs.sort_values("PF", ascending=False).iloc[0]
            _hero_cards.append(("Most points", f"{float(_hi['PF']):.0f}", str(_hi["owner"])))
            _hero_cards.append(("This week", f"Week {current_week}", f"{len(_dash_matchups)} matchups"))
    except Exception:
        logger.debug("dashboard hero stats failed", exc_info=True)
        _hero_cards = [("This week", f"Week {current_week}", "Live scoring & standings")]

    _hero_tiles = [
        f"""<div class="os-stat-card">
              <div class="os-stat-label">{html.escape(str(_lbl))}</div>
              <div class="os-stat-value">{html.escape(str(_val))}</div>
              <div class="os-stat-sub">{html.escape(str(_sub))}</div>
            </div>"""
        for _lbl, _val, _sub in _hero_cards
    ]

    # Playoff-odds tile — serve a warm cache on first paint; if the sim is
    # cold, kick it off in the background and let the client fill the tile.
    if viewer_roster_id:
        _po_val, _po_sub, _po_loaded = "-", "Simulating&hellip;", ""
        try:
            _warm = _playoff_sim_cached(ctx, platform, block=False) or []
            _filled = _playoff_tile_from_cache(_warm, viewer_roster_id)
            if _filled:
                _po_val = f"{_filled[0]}%"
                _po_sub = html.escape(_filled[1])
                _po_loaded = " is-loaded"
        except Exception:
            logger.debug("dashboard playoff prefetch failed", exc_info=True)
        _playoff_tile_html = f"""<div class="os-stat-card os-stat-playoff{_po_loaded}" id="dash-playoff-tile"
              data-platform="{html.escape(str(platform))}"
              data-league="{html.escape(str(league_id))}"
              data-season="{html.escape(str(season))}"
              data-roster="{html.escape(str(viewer_roster_id))}">
              <div class="os-stat-label">Playoff odds</div>
              <div class="os-stat-value" id="dash-playoff-val">{_po_val}</div>
              <div class="os-stat-sub" id="dash-playoff-sub">{_po_sub}</div>
            </div>"""
        # Slot right after the first (record) tile so it reads prominently.
        _hero_tiles.insert(1, _playoff_tile_html)

    _dash_bulletins_html = ""
    if str(platform or "sleeper").strip().lower() == "sleeper":
        _dash_bulletins_html = f"""
        <section class="os-card" id="leagueBulletinsContainer"
                 data-league="{html.escape(str(league_id))}"
                 data-platform="sleeper"
                 data-season="{html.escape(str(season))}">
          <div class="os-section-head">
            <div class="os-section-head-content">
              <h2 class="os-section-title">League Bulletins</h2>
              <div class="os-section-subtitle">From your Sleeper league board</div>
            </div>
          </div>
          <div class="bulletins-list">Loading&hellip;</div>
        </section>"""

    _hero_stats_html = "".join(_hero_tiles)

    _viewer_team = viewer.get("viewer_team_name")
    _hero_copy = (
        f"Welcome back, {html.escape(str(_viewer_team))}. Here is what changed and what needs your attention."
        if _viewer_team else
        "What changed, what needs attention, and your next moves."
    )

    _action_queue_html = f"""
        <div class="os-action-queue os-tab-panel os-tab-active" id="os-jump-actions">
          {lineup_alert_html}
          {roster_moves_html}
          {trade_window_html}
          {do_next_waiver_html}
        </div>"""

    body = f"""
    <div class="os-layout">
      <aside class="os-left-col os-tab-panel" id="os-jump-standings">
        <section class="os-card os-col-fill">
          <div class="os-section-head">
            <div class="os-section-head-content">
              {_section_title_link("Standings", "league_pages.page_standings", platform, season, league_id)}
              <div class="os-section-subtitle">Where every team sits right now</div>
            </div>
            <div class="os-section-head-actions">
              <button type="button" class="card-collapse-toggle" aria-label="Toggle section" aria-expanded="true" data-target="dash-standings-body">&#9660;</button>
            </div>
          </div>
          <div class="card-collapsible-body" id="dash-standings-body">
            {standings_html}
          </div>
        </section>
        {awards_html}
      </aside>

      <main class="os-main-col">
        <section class="os-hero-card">
          <div class="os-hero-top">
            <div>
              <h1 class="os-hero-title">Season Hub</h1>
              <p class="os-hero-copy">{_hero_copy}</p>
            </div>
          </div>
          <div class="os-hero-stats">
            {_hero_stats_html}
          </div>
        </section>

        <div id="sinceLastVisitCard" class="slv-wrap" data-slv-init="1"></div>

        <nav class="os-jump-nav" aria-label="Jump to section">
          <button type="button" class="active" data-jump="os-jump-actions">Actions</button>
          <button type="button" data-jump="os-jump-report">Report</button>
          <button type="button" data-jump="os-jump-standings">Standings</button>
          <button type="button" data-jump="os-jump-waivers">Waivers</button>
          <button type="button" data-jump="os-jump-teams">{teams_tab_label}</button>
        </nav>

        {_action_queue_html}

        <div id="os-jump-report" class="os-tab-panel">
          {gm_card_html}
          {usage_movers_html}
          {matchup_html}
          {bench_check_html}
          {season_review_html}
          {_dash_bulletins_html}
        </div>
      </main>

      <aside class="os-right-col os-tab-panel" id="os-jump-teams">
        <div class="os-sidebar-shell">
          {teams_sidebar_html}
        </div>
      </aside>
    </div>
    <script>
    (function(){{
      var nav = document.querySelector('.os-jump-nav');
      if (!nav) return;
      var ids = Array.prototype.map.call(nav.querySelectorAll('[data-jump]'), function(b){{ return b.getAttribute('data-jump'); }});
      function activate(id){{
        ids.forEach(function(pid){{
          var p = document.getElementById(pid);
          if (p) p.classList.toggle('os-tab-active', pid === id);
        }});
        nav.querySelectorAll('button').forEach(function(x){{ x.classList.toggle('active', x.getAttribute('data-jump') === id); }});
      }}
      nav.querySelectorAll('[data-jump]').forEach(function(btn){{
        btn.addEventListener('click', function(){{ activate(btn.getAttribute('data-jump')); }});
      }});
      var init = nav.querySelector('button.active') || nav.querySelector('button');
      if (init) activate(init.getAttribute('data-jump'));
    }})();
    </script>
    <script>
    (function() {{
      var el = document.getElementById('dash-playoff-tile');
      if (!el) return;
      if (el.classList.contains('is-loaded')) return;
      var rid = el.getAttribute('data-roster');
      var qs = 'platform=' + encodeURIComponent(el.getAttribute('data-platform')) +
               '&league_id=' + encodeURIComponent(el.getAttribute('data-league')) +
               '&season=' + encodeURIComponent(el.getAttribute('data-season'));
      fetch('/api/playoff-odds?' + qs)
        .then(function(r) {{ return r.ok ? r.json() : null; }})
        .then(function(d) {{
          var odds = (d && d.odds) || [];
          var row = null;
          for (var i = 0; i < odds.length; i++) {{
            if (String(odds[i].roster_id) === String(rid)) {{ row = odds[i]; break; }}
          }}
          var valEl = document.getElementById('dash-playoff-val');
          var subEl = document.getElementById('dash-playoff-sub');
          if (!row) {{
            if (valEl) valEl.textContent = '—';
            if (subEl) subEl.textContent = 'Odds unavailable';
            el.classList.add('is-loaded');
            return;
          }}
          var pct = Math.round(row.playoff_pct || 0);
          if (valEl) {{
            if (window.brCountUp) window.brCountUp(valEl, {{ to: pct, dp: 0, suffix: '%', dur: 800 }});
            else valEl.textContent = pct + '%';
          }}
          var sub;
          if (d.is_complete || row.is_complete) {{
            sub = pct >= 100 ? 'Clinched' : (pct <= 0 ? 'Eliminated' : 'Playoff bound');
          }} else {{
            var first = Math.round(row.first_seed_pct || 0);
            var bye = Math.round(row.bye_pct || 0);
            sub = first > 0 ? (first + '% top seed')
                : (bye > 0 ? (bye + '% first-round bye') : 'to make the playoffs');
          }}
          if (subEl) subEl.textContent = sub;
          el.classList.add('is-loaded');
        }})
        .catch(function() {{
          var valEl = document.getElementById('dash-playoff-val');
          var subEl = document.getElementById('dash-playoff-sub');
          if (valEl) valEl.textContent = '—';
          if (subEl) subEl.textContent = 'Odds unavailable';
          el.classList.add('is-loaded');
        }});
    }})();
    </script>
    """

    return body

