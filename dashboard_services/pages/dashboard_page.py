"""In-season dashboard HTML builder.

Moved from app.py so the Flask monolith can keep shrinking. Helpers that still
live in app.py are lazy-imported inside the builder (request time).
"""
from __future__ import annotations

def build_dashboard_body(ctx: dict) -> str:
    from app import (  # noqa: E402  (lazy: avoids a circular import at module load)
        _scorezone_cta_state,
        _render_next_steps_queue,
        _compute_fpts_against,
        _scoring_format_from_settings,
        _owner_to_rid_map,
        _playoff_sim_cached,
        _playoff_tile_from_cache,
        _render_bench_check,
        _render_season_review_card,
        _render_usage_movers,
        _losing_streak_trade_html,
        _section_title_link,
        _standings_movement,
        _league_is_redraft,
        compute_awards_alltime,
        compute_awards_season,
        compute_win_prob,
        get_model_value_table_cached,
        html,
        logger,
        render_awards_section,
        render_matchup_slide,
        render_standings_compact,
        url_for,
    )
    import pandas as pd
    from utils.league_payload import show_matchup_preview as _show_matchup_preview_for

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

    from utils.standings_divisions import resolve_divisions, division_records
    _div_info = resolve_divisions(ctx)
    standings_html = render_standings_compact(
        team_stats, movement=_standings_movement(df_weekly),
        owner_to_rid=_owner_to_rid_map(roster_map=ctx.get("roster_map"), df_weekly=df_weekly),
        divisions=_div_info,
        div_records=division_records(df_weekly, (_div_info or {}).get("by_rid") or {}),
    )
    usage_movers_html = _render_usage_movers(ctx, viewer_roster_id)
    losing_trade_html = _losing_streak_trade_html(ctx, viewer_roster_id)
    season_review_html = _render_season_review_card(ctx, viewer_roster_id, df_weekly, team_stats)

    # Tour / empty leagues can hand over a columnless or empty frame. Bind
    # finalized_df on every path -- awards still consume it below. Match
    # weekly_hub_page / app.py: missing `finalized` means "no completed weeks".
    if (
        df_weekly is not None
        and not getattr(df_weekly, "empty", True)
        and "finalized" in getattr(df_weekly, "columns", [])
    ):
        finalized_df = df_weekly[df_weekly["finalized"] == True].copy()
    else:
        finalized_df = pd.DataFrame()
    if not finalized_df.empty and "week" in finalized_df.columns:
        last_final_week = int(finalized_df["week"].max())
    else:
        last_final_week = 0

    bench_check_html = _render_bench_check(ctx, viewer_roster_id, last_final_week)

    # Unified Next steps queue: one ranked action list across lineup,
    # waivers, and trades. Replaces the old waiver-only card.
    do_next_waiver_html = ""
    try:
        _wv_table = list(get_model_value_table_cached() or []) or (ctx.get("model_value_table") or [])
        do_next_waiver_html = _render_next_steps_queue(
            ctx,
            viewer_roster_id,
            _wv_table,
            season=season,
            current_week=current_week,
            last_final_week=last_final_week,
        )
    except Exception:
        logger.debug("dashboard next-steps queue failed", exc_info=True)

    _fpts_against_dash = _compute_fpts_against(
        season,
        scoring=_scoring_format_from_settings(ctx.get("scoring_settings")),
    )
    _dash_vid = str(viewer_roster_id or "")
    _league_for_preview = dict(ctx.get("league") or {})
    if ctx.get("league_settings"):
        merged = dict(ctx["league_settings"])
        merged.update(_league_for_preview.get("settings") or {})
        _league_for_preview["settings"] = merged
    _show_matchup_preview = _show_matchup_preview_for(
        _league_for_preview,
        ctx.get("latest_draft"),
        rosters,
        is_dynasty=not _league_is_redraft(ctx),
    )
    from utils.matchup_schedule import resolve_matchup_week
    matchup_week = resolve_matchup_week(current_week, matchups_by_week)
    _dash_matchups = sorted(
        matchups_by_week.get(matchup_week, []) or matchups_by_week.get(str(matchup_week), []) or [],
        key=lambda m: 0 if _dash_vid and _dash_vid in (str((m.get("left") or {}).get("roster_id", "")),
                                                       str((m.get("right") or {}).get("roster_id", ""))) else 1,
    ) if _show_matchup_preview else []
    if _show_matchup_preview:
        from utils.standings_divisions import division_records_for_ctx, div_map_for_ctx
        _dash_div_records = division_records_for_ctx(ctx)
        _dash_div_by_rid = div_map_for_ctx(ctx)
        slides = [
            render_matchup_slide(
                season,
                m,
                matchup_week,
                last_final_week,
                status_by_pid=(statuses.get(matchup_week) or statuses.get(str(matchup_week)) or {}).get("statuses", {}) or {},
                projections=proj_by_week,
                players=players_index,
                teams=teams_index,
                team_game_lookup=team_game_lookup,
                fpts_against=_fpts_against_dash,
                compact=True,
                scoring_settings=ctx.get("raw_scoring_settings") or ctx.get("scoring_settings"),
                div_records=_dash_div_records,
                div_by_rid=_dash_div_by_rid,
            )
            for m in _dash_matchups
        ]
        _dash_preview_slide = slides[0] if slides else ""
        _matchup_href = ""
        try:
            _matchup_href = url_for(
                "page_weekly", platform=platform, season=season, league_id=str(league_id)
            )
        except Exception:
            _matchup_href = ""
        matchup_html = ""  # dashboard redesign: preview card uses slides[0]
    else:
        matchup_html = ""
        bench_check_html = ""
        _dash_preview_slide = ""

    # League Awards: all-time history across seasons; fall back to the
    # single-season card if the all-time computation has nothing (or fails).
    _awards_alltime = compute_awards_alltime(
        platform, league_id, season, players_map, users, rosters, finalized_df,
    )
    if _awards_alltime:
        awards = _awards_alltime
        _awards_is_alltime = True
    else:
        awards = compute_awards_season(finalized_df, players_map, league_id, platform, season, users, rosters)
        _awards_is_alltime = False
    awards_html = render_awards_section(awards, alltime=_awards_is_alltime)

    ctx["model_value_table"] = list(get_model_value_table_cached() or []) or (ctx.get("model_value_table") or [])

    gm_card_html = ""
    if viewer_roster_id:
        # Prompt-only until Generate Report. Serving a cached memo on first
        # paint (or generating one) made the report appear without a click.
        gm_card_html = f"""
        <section class="os-side-plain cc-fo-card" id="os-jump-report-card">
          <div class="cc-fo-head">
            <div class="cc-fo-head-icon"><i class="fa-solid fa-chart-line" aria-hidden="true"></i></div>
            <div>
              <h2 class="cc-fo-title">Front Office Report</h2>
              <div class="cc-fo-sub">Roster, trade targets, and standings analysis</div>
            </div>
            <button type="button" class="card-collapse-toggle cc-head-toggle" aria-label="Toggle section" aria-expanded="true" data-target="dash-fo-body">&#9660;</button>
          </div>
          <div class="card-collapsible-body" id="dash-fo-body">
          <div class="os-ai-copy">
            <div class="otc-ai-empty" id="gm-memo-empty">
              <div class="otc-ai-empty-sub">
                Get personalized analysis on your roster, trade targets, and standings.
              </div>
            </div>
            <button type="button" id="generateGmMemoBtn" class="cc-fo-btn"
                    data-league-id="{html.escape(str(league_id))}"
                    data-season="{html.escape(str(season))}"
                    data-platform="{html.escape(str(platform))}"
                    data-viewer-roster-id="{html.escape(str(viewer_roster_id))}">
              Generate Report
            </button>
            <div class="otc-ai-empty" id="gm-memo-loading" style="display:none;">
              <div class="otc-ai-empty-title">Analyzing Your Roster...</div>
              <div class="otc-ai-empty-sub">
                <div class="loading-spinner" style="margin: 10px auto; width: 30px; height: 30px; border: 3px solid var(--border); border-radius: 50%; border-top-color: var(--accent); animation: spin 1s linear infinite; border-right-color: transparent;"></div>
              </div>
            </div>
            <div id="gm-memo-result" style="display:none;"></div>
          </div>
          </div>
        </section>
        """

    # ---- Hero stat cards -- mirror the offseason hub's hero card ----
    from utils.format import ordinal as _dash_ord

    _hero_cards: list = []
    _vname = ""
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
                if _dash_matchups and matchup_week > last_final_week:
                    _m0 = _dash_matchups[0]
                    _l0 = _m0.get("left") or {}
                    _r0 = _m0.get("right") or {}
                    if _dash_vid and _dash_vid in (str(_l0.get("roster_id", "")), str(_r0.get("roster_id", ""))):
                        _wp_proj = (proj_by_week.get(matchup_week) or {}).get("projections") or {}
                        _wp_status = (statuses.get(matchup_week) or {}).get("statuses", {})
                        from dashboard_services.matchups import make_frac_lookup
                        _wp_frac = make_frac_lookup(team_game_lookup)
                        _wp = compute_win_prob(
                            _l0, _r0, _wp_status, _wp_proj, frac_lookup=_wp_frac,
                        )
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
                _tw_sub = (
                    f"{len(_dash_matchups)} matchups"
                    if _show_matchup_preview else "Waiting on the draft"
                )
            _hero_cards.append(("This week", f"Week {matchup_week}", _tw_sub))
        else:
            _top = _hs.iloc[0]
            _hero_cards.append(("League leader", str(_top["owner"]),
                                f"{int(_top['Wins'])}-{int(_top['Losses'])} record"))
            _hi = _hs.sort_values("PF", ascending=False).iloc[0]
            _hero_cards.append(("Most points", f"{float(_hi['PF']):.0f}", str(_hi["owner"])))
            _hero_cards.append((
                "This week", f"Week {current_week}",
                f"{len(_dash_matchups)} matchups" if _show_matchup_preview else "Waiting on the draft",
            ))
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

    # Playoff-odds tile -- serve a warm cache on first paint; if the sim is
    # cold, kick it off in the background and let the client fill the tile.
    _po_val, _po_sub, _po_loaded = "-", "Simulating&hellip;", ""
    if viewer_roster_id:
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

    _hero_stats_html = "".join(_hero_tiles)

    # Best-ball badge (hoisted: rendered in the hero strip next to the team name).
    _bb_badge = ""
    try:
        from utils.league_format import is_best_ball as _dash_is_bb
        if _dash_is_bb(
            ctx.get("league") or {},
            settings=(ctx.get("league_settings")
                      or (ctx.get("league") or {}).get("settings")
                      or ctx.get("settings") or {}),
        ):
            _bb_badge = (
                '<span class="os-hero-tag" style="display:inline-flex;align-items:center;'
                'margin-left:8px;padding:3px 9px;border-radius:8px;font-size:11px;font-weight:800;'
                'letter-spacing:.03em;vertical-align:middle;color:var(--accent);'
                'background:var(--accent-soft);" title="Best Ball: no weekly lineup locks">'
                'Best Ball</span>'
            )
    except Exception:
        _bb_badge = ""

    # ---- Compact hero strip (dashboard redesign): team identity + 4 stats ----
    _hero_grade_badge = ""
    _hero_archetype_tag = ""
    if viewer_roster_id:
        try:
            # League-context grade (same source as the Teams page) so the
            # hero strip never disagrees with Teams on grade/window.
            from dashboard_services.ai.league_grades import viewer_league_context_grade
            _gd = viewer_league_context_grade(ctx, viewer_roster_id) or {}
            _g = str(_gd.get("grade") or "").strip()
            _w = str(_gd.get("win_window") or "").strip()
            if _g and _g.upper() != "N/A":
                _hero_grade_badge = (
                    f'<span class="os-grade-badge">{html.escape(_g)}</span>'
                )
            if _w and _w.lower() not in ("", "unknown"):
                _hero_archetype_tag = (
                    f'<span class="os-archetype-tag">{html.escape(_w)}</span>'
                )
        except Exception:
            logger.debug("dashboard hero grade failed", exc_info=True)
    _strip_stats = []
    for _sl, _sv, _ss in _hero_cards:
        _strip_stats.append(
            (html.escape(str(_sl)), html.escape(str(_sv)), html.escape(str(_ss)))
        )
    if viewer_roster_id and len(_strip_stats) >= 1:
        _strip_stats.insert(1, ("Playoff odds", html.escape(str(_po_val)), _po_sub))
    def _strip_stat_html(_l, _v, _s):
        _sub = f'<div class="ss">{_s}</div>' if _s else ""
        return (
            f'<div class="os-hero-strip-stat"><div class="sl">{_l}</div>'
            f'<div class="sv">{_v}</div>'
            f"{_sub}</div>"
        )

    _strip_stats_html = "".join(
        _strip_stat_html(_l, _v, _s) for _l, _v, _s in _strip_stats[:4]
    )
    _hero_strip_html = f"""
    <section class="os-hero-strip">
      <div class="os-hero-strip-team">
        <div class="os-hero-strip-name">{html.escape(str(_vname or "Season Hub"))}{_bb_badge}</div>
        <div class="os-hero-strip-badges">{_hero_grade_badge}{_hero_archetype_tag}</div>
      </div>
      <div class="os-hero-strip-div"></div>
      <div class="os-hero-strip-stats">{_strip_stats_html}</div>
    </section>
    """

    # ---- Matchup preview card (dashboard redesign) ----
    _matchup_preview_html = ""
    if _dash_preview_slide:
        _mp_href = ""
        try:
            _mp_href = url_for(
                "page_weekly", platform=platform, season=season,
                league_id=str(league_id),
            )
        except Exception:
            _mp_href = ""
        _matchup_preview_html = f"""
    <section class="os-card os-matchup-preview">
      <div class="os-section-head">
        <div class="os-section-head-content">
          <h2 class="os-section-title">This Week&apos;s Matchup</h2>
          <div class="os-section-subtitle">Week {int(matchup_week)}</div>
        </div>
        <div class="os-section-head-actions">
          <a class="os-section-link" href="{html.escape(_mp_href, quote=True)}">View all &rarr;</a>
        </div>
      </div>
      {_dash_preview_slide}
    </section>
    """

    _viewer_team = viewer.get("viewer_team_name")
    _hero_copy = (
        f"Welcome back, {html.escape(str(_viewer_team))}. Here is what changed and what needs your attention."
        if _viewer_team else
        "What changed, what needs attention, and your next moves."
    )
    _bb_outlook_html = ""
    try:
        from utils.league_format import is_best_ball
        if is_best_ball(
            ctx.get("league") or {},
            settings=(ctx.get("league_settings")
                      or (ctx.get("league") or {}).get("settings")
                      or ctx.get("settings") or {}),
        ):
            if _viewer_team:
                _hero_copy = (
                    f"Welcome back, {html.escape(str(_viewer_team))}. Best Ball mode: "
                    "weekly Start/Sit and lineup locks are hidden. Focus on waivers and the draft."
                )
            else:
                _hero_copy = (
                    "Best Ball mode: weekly Start/Sit and lineup locks are hidden. "
                    "Focus on waivers and the draft."
                )
            # R10.3 thin season outlook -- playoff odds when available; honest v1 copy.
            _outlook_odds = ""
            if viewer_roster_id:
                if _po_val and _po_val != "-":
                    _outlook_odds = (
                        f'<div class="os-section-subtitle" style="margin-top:6px">'
                        f'Playoff odds: <strong>{html.escape(str(_po_val))}</strong>'
                        f'{(" · " + _po_sub) if _po_sub else ""}'
                        f'</div>'
                    )
                else:
                    _outlook_odds = (
                        '<div class="os-section-subtitle" style="margin-top:6px">'
                        "Playoff odds load when the sim is ready."
                        "</div>"
                    )
            _bb_outlook_html = f"""
        <section class="os-card" id="dashBestBallOutlook">
          <div class="os-section-head">
            <div class="os-section-head-content">
              <h2 class="os-section-title">Season outlook</h2>
              <div class="os-section-subtitle">Best Ball uses automatic optimal lineups each week.</div>
            </div>
          </div>
          <p style="margin:0;font-size:13px;line-height:1.45;color:var(--text-muted)">
            Lineup locks and Start/Sit do not apply. Use playoff odds and standings for season-long outlook;
            weekly optimal lineups are automatic.
          </p>
          {_outlook_odds}
        </section>"""
    except Exception:
        _bb_outlook_html = ""

    # The Actions queue is the default-active tab, so it must never render blank.
    # Every card above returns "" with no viewer roster (nothing is personalized)
    # or when the roster is clean this week; without a fallback the landing tab
    # looked empty/broken. Show a "link your team" prompt when unidentified, or an
    # all-clear card when identified but there is nothing to do.
    # Urgency ordering: sort cards by deadline pressure so the most time-sensitive
    # action is on top. Lineup locks at kickoff (highest), waivers process
    # mid-week, trade window is urgent only near the deadline, roster moves have
    # no hard deadline (lowest).
    def _card_urgency(key, card_html):
        if not (card_html or "").strip():
            return 0
        if key == "nextsteps":
            return 75  # unified action queue: ranked by impact internally
        if key == "trade-losing":
            return 85  # losing teams need action now, just below urgent trades
        if key == "roster":
            return 40
        return 0

    _action_cards = [
        ("nextsteps", do_next_waiver_html),
        ("trade-losing", losing_trade_html),
    ]
    _action_cards = sorted(
        [(k, h) for k, h in _action_cards if (h or "").strip()],
        key=lambda kh: -_card_urgency(kh[0], kh[1]),
    )
    if _action_cards:
        _action_inner = "".join(h for _, h in _action_cards)
    # Dismissal: persisted per card type + day so dismissed cards stay hidden
    # until tomorrow. For signed-in accounts the source of truth is the
    # server store (/api/action-dismissals, account-scoped): the script syncs
    # from it on load and writes dismiss/undo through to it, so a dismissal
    # follows the user across devices and is shared with the portfolio page.
    # localStorage remains the paint cache (and the whole store for guests).
    # Individual Next steps actions dismiss the same way, per action id
    # (nsitem- keys in the same map). Always included: the bench-check card
    # (rendered in the matchup panel) is dismissable too.
    if _action_cards or (bench_check_html or "").strip():
        import json as _json

        _dismiss_script = f"""
          <script>
          (function() {{
            var SEASON = {int(season or 0)};
            var WEEK = {int(current_week or 0)};
            var PLATFORM = {_json.dumps(str(platform or "sleeper"))};
            var LEAGUE_ID = {_json.dumps(str(league_id or ""))};
            var LS_KEY = "br-actions-dismissed";
            function dismissedMap() {{
              try {{ return JSON.parse(localStorage.getItem(LS_KEY) || "{{}}"); }}
              catch (e) {{ return {{}}; }}
            }}
            function saveMap(map) {{
              try {{ localStorage.setItem(LS_KEY, JSON.stringify(map)); }} catch (e) {{}}
            }}
            function dayStr() {{
              var d = new Date();
              return d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0") + "-" + String(d.getDate()).padStart(2, "0");
            }}
            function cardKey(type) {{
              return type + "-" + SEASON + "-" + WEEK + "-" + dayStr();
            }}
            function itemKey(id) {{
              return "nsitem-" + id + "-" + SEASON + "-" + WEEK + "-" + dayStr();
            }}
            // An item counts as dismissed when its own key is set, when its
            // shared cross-surface key (data-ns-key, e.g. "wv:<player_id>"
            // from a portfolio dismissal) is set, or when its whole kind was
            // dismissed from the portfolio digest ("cat:<kind>").
            function itemIsDismissed(el, map) {{
              if (!el) return false;
              if (map[itemKey(el.getAttribute("data-ns-item"))]) return true;
              var ck = el.getAttribute("data-ns-key");
              if (ck && map[ck]) return true;
              var kind = el.getAttribute("data-ns-kind");
              if (kind && map["cat:" + kind]) return true;
              return false;
            }}
            // Server sync: /api/action-dismissals is the source of truth for
            // signed-in accounts (keyed by the account, so dismissals follow
            // the user across devices and between the portfolio page and
            // this dashboard). localStorage is only the paint cache here,
            // and the whole store for guests (the endpoint reports
            // synced:false for them and sync stays off).
            var SYNC_MODE = "unknown"; // "unknown" -> "server" | "guest"
            var touchedBare = {{}}; // bare keys changed before sync resolved
            function bareFromPageKey(map, bare, on) {{
              var pageKeys = [];
              if (bare.indexOf("card:") === 0) pageKeys.push(cardKey(bare.slice(5)));
              else if (bare.indexOf(":") !== -1) pageKeys.push(bare);
              else pageKeys.push(itemKey(bare));
              for (var i = 0; i < pageKeys.length; i++) {{
                if (on) map[pageKeys[i]] = true; else delete map[pageKeys[i]];
              }}
            }}
            function persistBare(method, bare) {{
              if (SYNC_MODE === "guest") return;
              if (SYNC_MODE === "unknown") touchedBare[bare] = (method === "POST");
              try {{
                fetch("/api/action-dismissals", {{
                  method: method,
                  headers: {{ "Content-Type": "application/json" }},
                  credentials: "same-origin",
                  body: JSON.stringify({{
                    platform: PLATFORM, league_id: LEAGUE_ID,
                    season: SEASON, week: WEEK, day: dayStr(), item_key: bare
                  }})
                }}).then(function(r) {{ return r.json(); }})
                  .then(function(d) {{ if (d && d.synced === false) SYNC_MODE = "guest"; }})
                  .catch(function() {{}});
              }} catch (e) {{}}
            }}
            function syncFromServer() {{
              var qs = "?platform=" + encodeURIComponent(PLATFORM)
                + "&league_id=" + encodeURIComponent(LEAGUE_ID)
                + "&season=" + SEASON + "&week=" + WEEK
                + "&day=" + encodeURIComponent(dayStr());
              try {{
                fetch("/api/action-dismissals" + qs, {{ credentials: "same-origin" }})
                  .then(function(r) {{ return r.json(); }})
                  .then(function(d) {{
                    if (!d || !d.synced) {{ SYNC_MODE = "guest"; return; }}
                    SYNC_MODE = "server";
                    var map = {{}};
                    var keys = d.dismissed || [];
                    for (var i = 0; i < keys.length; i++) bareFromPageKey(map, keys[i], true);
                    // Re-apply anything dismissed/undone while the fetch
                    // was in flight so a fast click is never overwritten.
                    for (var bare in touchedBare) bareFromPageKey(map, bare, touchedBare[bare]);
                    saveMap(map);
                    hideDismissed();
                    var nsCards = document.querySelectorAll(".ns-card[data-action-card]");
                    for (var j = 0; j < nsCards.length; j++) refreshNsCard(nsCards[j], dismissedMap());
                  }})
                  .catch(function() {{}});
              }} catch (e) {{}}
            }}
            function refreshNsCard(card, map) {{
              if (!card) return null;
              var items = card.querySelectorAll(".ns-item[data-ns-item]");
              if (!items.length) return null;
              var remaining = 0;
              for (var i = 0; i < items.length; i++) {{
                if (!itemIsDismissed(items[i], map)) remaining++;
              }}
              var count = card.querySelector(".ns-count");
              if (count) count.textContent = remaining + " action" + (remaining !== 1 ? "s" : "");
              if (!map[cardKey("nextsteps")]) {{
                card.style.display = remaining ? "" : "none";
              }}
              var promoted = rebalanceNsFrame(card, map);
              fitNsFrame(card, map, remaining);
              return promoted;
            }}
            // Fit-the-row state, refreshed on load, dismiss, and undo.
            // The Show all toggle only earns its place while more
            // actions remain than the frame holds: at or below the
            // frame limit everything left is already visible, so the
            // toggle hides. And when exactly one action sits in the
            // frame it goes long and flat: .ns-frame-single switches
            // the item to its horizontal layout in the stylesheet.
            function fitNsFrame(card, map, remaining) {{
              var limit = parseInt(card.getAttribute("data-ns-frame-limit") || "3", 10) || 3;
              var toggle = card.querySelector(".ns-expand-toggle");
              if (toggle) toggle.style.display = remaining > limit ? "" : "none";
              var frame = card.querySelector(".ns-list:not(.ns-hidden)");
              var extra = card.querySelector(".ns-list.ns-hidden");
              var framed = 0;
              if (frame && (card.classList.contains("ns-collapsed") || !extra)) {{
                var kids = frame.querySelectorAll(".ns-item[data-ns-item]");
                for (var i = 0; i < kids.length; i++) {{
                  if (kids[i].style.display !== "none"
                      && !map[itemKey(kids[i].getAttribute("data-ns-item"))]) framed++;
                }}
              }}
              card.classList.toggle("ns-frame-single", framed === 1);
            }}
            // Collapsed view keeps its frame full: the first N remaining
            // actions (N = the server-rendered preview size, captured on
            // first run) live in the main list and later ones wait in the
            // extra list. Dismissing a framed action promotes the next
            // waiting one into the freed slot; undoing sends it back.
            // Dismissed items themselves never move lists. Returns the
            // promoted element so the caller can animate its entrance,
            // else null.
            function rebalanceNsFrame(card, map) {{
              if (!card.classList.contains("ns-collapsed")) return null;
              var frame = card.querySelector(".ns-list:not(.ns-hidden)");
              var extra = card.querySelector(".ns-list.ns-hidden");
              if (!frame || !extra) return null;
              var limit = parseInt(card.getAttribute("data-ns-frame-limit") || "0", 10);
              if (!limit) {{
                limit = frame.querySelectorAll(".ns-item[data-ns-item]").length;
                if (!limit) return null;
                card.setAttribute("data-ns-frame-limit", String(limit));
                // Stamp the server-rendered order on first run: rebalances
                // move surviving items around hidden dismissed ones, and
                // undo relies on these stamps to restore the exact slots.
                var stampKids = frame.querySelectorAll(".ns-item[data-ns-item]");
                var stampExtra = extra.querySelectorAll(".ns-item[data-ns-item]");
                var stamp = 0;
                for (var si = 0; si < stampKids.length; si++) stampKids[si].setAttribute("data-ns-order", String(stamp++));
                for (var sj = 0; sj < stampExtra.length; sj++) stampExtra[sj].setAttribute("data-ns-order", String(stamp++));
              }}
              var ordered = [];
              var frameKids = frame.querySelectorAll(".ns-item[data-ns-item]");
              var extraKids = extra.querySelectorAll(".ns-item[data-ns-item]");
              for (var i = 0; i < frameKids.length; i++) ordered.push(frameKids[i]);
              for (var j = 0; j < extraKids.length; j++) ordered.push(extraKids[j]);
              ordered.sort(function(x, y) {{
                return (parseInt(x.getAttribute("data-ns-order") || "0", 10) || 0)
                  - (parseInt(y.getAttribute("data-ns-order") || "0", 10) || 0);
              }});
              var remaining = [];
              for (var k = 0; k < ordered.length; k++) {{
                if (!itemIsDismissed(ordered[k], map)) remaining.push(ordered[k]);
              }}
              var promoted = null;
              for (var f = 0; f < remaining.length && f < limit; f++) {{
                if (remaining[f].parentNode !== frame) promoted = remaining[f];
                frame.appendChild(remaining[f]);
              }}
              for (var r = limit; r < remaining.length; r++) {{
                extra.appendChild(remaining[r]);
              }}
              return promoted;
            }}
            function hideDismissed() {{
              var map = dismissedMap();
              var cards = document.querySelectorAll("[data-action-card]");
              for (var i = 0; i < cards.length; i++) {{
                var t = cards[i].getAttribute("data-action-card");
                if (t && map[cardKey(t)]) {{
                  cards[i].style.display = "none";
                }}
              }}
              var items = document.querySelectorAll(".ns-item[data-ns-item]");
              for (var j = 0; j < items.length; j++) {{
                if (itemIsDismissed(items[j], map)) {{
                  items[j].style.display = "none";
                }}
              }}
              var nsCards = document.querySelectorAll(".ns-card[data-action-card]");
              for (var k = 0; k < nsCards.length; k++) {{
                refreshNsCard(nsCards[k], map);
              }}
            }}
            function bindDismiss() {{
              var btns = document.querySelectorAll("[data-dismiss-card]");
              for (var i = 0; i < btns.length; i++) {{
                (function(btn) {{
                  btn.addEventListener("click", function() {{
                    var t = btn.getAttribute("data-dismiss-card");
                    var card = btn.closest("[data-action-card]");
                    if (!t) return;
                    var map = dismissedMap();
                    var key = cardKey(t);
                    map[key] = true;
                    saveMap(map);
                    persistBare("POST", "card:" + t);
                    if (card) {{
                      card.style.transition = "opacity .38s cubic-bezier(0.22, 1, 0.36, 1)";
                      card.style.opacity = "0";
                      setTimeout(function() {{ card.style.display = "none"; }}, 400);
                    }}
                    if (window.brUndoToast) {{
                      window.brUndoToast("Card dismissed.", function() {{
                        var m = dismissedMap();
                        delete m[key];
                        saveMap(m);
                        persistBare("DELETE", "card:" + t);
                        if (card) {{
                          card.style.display = "";
                          requestAnimationFrame(function() {{
                            card.style.opacity = "1";
                          }});
                        }}
                      }});
                    }}
                  }});
                }})(btns[i]);
              }}
            }}
            function reducedMotion() {{
              try {{ return window.matchMedia("(prefers-reduced-motion: reduce)").matches; }}
              catch (e) {{ return false; }}
            }}
            function visibleItemSiblings(item, card) {{
              var out = [];
              if (!card) return out;
              var items = card.querySelectorAll(".ns-item[data-ns-item]");
              for (var i = 0; i < items.length; i++) {{
                if (items[i] !== item && items[i].style.display !== "none") out.push(items[i]);
              }}
              return out;
            }}
            // One shared timing for every Next steps motion (dismiss
            // fade, FLIP glide, refill rise, flatten/unflatten): a
            // gentle .38s ease-out so re-fits land softly, not snappy.
            var NS_EASE = "cubic-bezier(0.22, 1, 0.36, 1)";
            // FLIP: record sibling rects, mutate layout, then glide each
            // moved sibling from its old rect to its new one, animating
            // size as well as position, so dismissals and undo restores
            // re-fit the row smoothly (cards split the freed width on
            // desktop, stack vertically on mobile) instead of jumping.
            function flipSiblings(siblings, mutate) {{
              if (reducedMotion() || !siblings.length) {{ mutate(); return; }}
              var els = [];
              var before = [];
              for (var i = 0; i < siblings.length; i++) {{
                var r = siblings[i].getBoundingClientRect();
                if (r.width || r.height) {{ els.push(siblings[i]); before.push(r); }}
              }}
              mutate();
              var jobs = [];
              for (var j = 0; j < els.length; j++) {{
                var el = els[j];
                if (el.style.display === "none") continue;
                var now = el.getBoundingClientRect();
                if (!now.width && !now.height) continue;
                var dx = before[j].left - now.left;
                var dy = before[j].top - now.top;
                if (!dx && !dy && before[j].width === now.width && before[j].height === now.height) continue;
                jobs.push({{ el: el, old: before[j], now: now, dx: dx, dy: dy }});
              }}
              if (!jobs.length) return;
              for (var a = 0; a < jobs.length; a++) {{
                var job = jobs[a];
                job.el.style.transition = "none";
                job.el.style.flex = "none";
                job.el.style.width = job.old.width + "px";
                job.el.style.height = job.old.height + "px";
                job.el.style.transform = "translate(" + job.dx + "px, " + job.dy + "px)";
              }}
              void jobs[0].el.offsetWidth;
              for (var b = 0; b < jobs.length; b++) {{
                (function(job) {{
                  job.el.style.transition = "transform .38s " + NS_EASE + ", width .38s " + NS_EASE + ", height .38s " + NS_EASE;
                  job.el.style.width = job.now.width + "px";
                  job.el.style.height = job.now.height + "px";
                  job.el.style.transform = "";
                  setTimeout(function() {{
                    job.el.style.transition = "";
                    job.el.style.transform = "";
                    job.el.style.width = "";
                    job.el.style.height = "";
                    job.el.style.flex = "";
                  }}, 440);
                }})(jobs[b]);
              }}
            }}
            // Entrance for a promoted action: rises from just below its
            // slot while fading in, on the same .38s timing as the FLIP
            // glide, so the refill reads as one motion with the slide.
            function animateNsEnter(el) {{
              if (!el || reducedMotion()) return;
              el.style.transition = "none";
              el.style.opacity = "0";
              el.style.transform = "translateY(22px)";
              void el.offsetWidth;
              el.style.transition = "transform .38s " + NS_EASE + ", opacity .38s " + NS_EASE;
              el.style.opacity = "";
              el.style.transform = "";
              setTimeout(function() {{
                el.style.transition = "";
                el.style.transform = "";
                el.style.opacity = "";
              }}, 440);
            }}
            function bindItemDismiss() {{
              var btns = document.querySelectorAll("[data-dismiss-ns-item]");
              for (var i = 0; i < btns.length; i++) {{
                (function(btn) {{
                  btn.addEventListener("click", function() {{
                    var id = btn.getAttribute("data-dismiss-ns-item");
                    var item = btn.closest(".ns-item");
                    var card = btn.closest("[data-action-card]");
                    if (!id || !item) return;
                    var map = dismissedMap();
                    var key = itemKey(id);
                    map[key] = true;
                    // Shared cross-surface key when the action carries one
                    // (waiver adds): the server record uses it, so the same
                    // player add dismissed here stays dismissed in the
                    // portfolio digest, and vice versa.
                    var bare = item.getAttribute("data-ns-key") || id;
                    if (bare !== id) map[bare] = true;
                    saveMap(map);
                    persistBare("POST", bare);
                    var siblings = visibleItemSiblings(item, card);
                    function finishDismiss() {{
                      var promoted = null;
                      flipSiblings(siblings, function() {{
                        item.style.display = "none";
                        promoted = refreshNsCard(card, dismissedMap());
                      }});
                      animateNsEnter(promoted);
                    }}
                    if (reducedMotion()) {{
                      finishDismiss();
                    }} else {{
                      item.style.transition = "opacity .38s " + NS_EASE;
                      item.style.opacity = "0";
                      setTimeout(finishDismiss, 390);
                    }}
                    if (window.brUndoToast) {{
                      window.brUndoToast("Action dismissed.", function() {{
                        var m = dismissedMap();
                        delete m[key];
                        if (bare !== id) delete m[bare];
                        saveMap(m);
                        persistBare("DELETE", bare);
                        flipSiblings(visibleItemSiblings(item, card), function() {{
                          item.style.display = "";
                          refreshNsCard(card, dismissedMap());
                        }});
                        requestAnimationFrame(function() {{
                          item.style.opacity = "1";
                        }});
                      }});
                    }}
                  }});
                }})(btns[i]);
              }}
            }}
            if (document.readyState === "loading") {{
              document.addEventListener("DOMContentLoaded", function() {{ hideDismissed(); bindDismiss(); bindItemDismiss(); syncFromServer(); }});
            }} else {{
              hideDismissed(); bindDismiss(); bindItemDismiss(); syncFromServer();
            }}
          }})();
          </script>"""
        _action_inner += _dismiss_script
        # One-time onboarding coachmark: points first-time users at Next steps.
        _coachmark_script = """
          <script>
          (function() {
            var KEY = "br-coachmark-nextsteps-seen";
            try {
              if (localStorage.getItem(KEY)) return;
            } catch (e) { return; }
            function show() {
              var target = document.querySelector("[data-action-card='nextsteps']");
              if (!target) return;
              var tip = document.createElement("div");
              tip.setAttribute("role", "status");
              tip.style.cssText = "background:var(--accent);color:#fff;border-radius:12px;"
                + "padding:12px 16px;font-size:13px;font-weight:600;margin-bottom:12px;"
                + "display:flex;justify-content:space-between;align-items:center;gap:12px;";
              tip.innerHTML = "<span>Start here - these are your top actions this week.</span>"
                + "<button type='button' aria-label='Dismiss' style='background:none;border:none;color:#fff;"
                + "font-size:16px;cursor:pointer;line-height:1;'>&times;</button>";
              tip.querySelector("button").addEventListener("click", function() {
                tip.remove();
                try { localStorage.setItem(KEY, "1"); } catch (e) {}
              });
              target.parentNode.insertBefore(tip, target);
              setTimeout(function() {
                if (tip.parentNode) {
                  tip.remove();
                  try { localStorage.setItem(KEY, "1"); } catch (e) {}
                }
              }, 15000);
            }
            if (document.readyState === "loading") {
              document.addEventListener("DOMContentLoaded", show);
            } else {
              show();
            }
          })();
          </script>"""
        _action_inner += _coachmark_script
    elif not viewer_roster_id:
        _lm_args = (
            f"'{html.escape(str(platform), quote=True)}', "
            f"'{html.escape(str(league_id), quote=True)}', "
            f"'{html.escape(str(season), quote=True)}'"
        )
        _action_inner = f"""
          <section class="os-card os-actions-empty">
            <div class="os-section-head">
              <div class="os-section-head-content">
                <h2 class="os-section-title">Link your team to see this week's actions</h2>
                <div class="os-section-subtitle">Lineup fixes, roster moves, and waiver targets are personalized to your team.</div>
              </div>
            </div>
            <p style="margin:0 0 12px;font-size:13px;line-height:1.45;color:var(--text-muted)">
              We could not match you to a team in this league yet. Link your team and your weekly to-dos show up here.
            </p>
            <button type="button" class="recap-generate-btn"
                    onclick="if(window.linkMyTeam){{linkMyTeam({_lm_args});}}else if(window.openLinkModal){{openLinkModal();}}">
              Link my team
            </button>
          </section>"""
    else:
        _waivers_href = f"/{html.escape(str(platform), quote=True)}/{int(season)}/{html.escape(str(league_id), quote=True)}/waivers"
        _action_inner = f"""
          <section class="os-card os-actions-empty">
            <div class="os-section-head">
              <div class="os-section-head-content">
                <h2 class="os-section-title">You're all set for Week {html.escape(str(current_week))}</h2>
                <div class="os-section-subtitle">No lineup, roster, or waiver moves need your attention right now.</div>
              </div>
            </div>
            <p style="margin:0;font-size:13px;line-height:1.45;color:var(--text-muted)">
              Check back after games and waivers process, or review your
              <a class="os-section-link" href="{_waivers_href}?tab=startsit">Start/Sit</a> and
              <a class="os-section-link" href="{_waivers_href}">Waivers</a> any time.
            </p>
          </section>"""

    _action_queue_html = f"""
        <div class="os-action-queue" id="os-jump-actions">{_action_inner}
        </div>"""

    # ScoreZone CTA: a calm pregame banner within the hour before kickoff, then a
    # pulsing live banner once a game is in progress. Shown only in that window
    # (server-computed like the nav glow); nothing renders otherwise.
    _rz_cta_html = ""
    try:
        _rz_cta_state = _scorezone_cta_state(season, current_week)
    except Exception:
        logger.debug("dashboard scorezone cta state failed", exc_info=True)
        _rz_cta_state = ""
    if _rz_cta_state:
        try:
            _rz_href = url_for(
                "page_scorezone", platform=platform, season=season, league_id=str(league_id)
            )
        except Exception:
            _rz_href = "./scorezone"
        if _rz_cta_state == "live":
            _rz_msg = "NFL games are live right now, track your players in real time."
            _rz_link = "Watch on ScoreZone"
        else:  # pregame
            _rz_msg = "Kickoff is coming up. Get set to track your players live."
            _rz_link = "Open ScoreZone"
        _rz_cta_html = (
            f'<div class="weekly-rz-cta weekly-rz-cta-{_rz_cta_state}">'
            '<span class="weekly-rz-cta-dot"></span>'
            f'<span class="weekly-rz-cta-text">{_rz_msg}</span>'
            f'<a href="{html.escape(str(_rz_href), quote=True)}" class="weekly-rz-cta-link">{_rz_link} &rarr;</a>'
            '</div>'
        )

    # ---- Command Center: league activity feed (recent trades/waivers) ----
    _activity_html = ""
    try:
        from dashboard_services.service import get_transactions_by_week
        _act_weeks = sorted({int(current_week), max(1, int(current_week) - 1)})
        _tx_by_week = get_transactions_by_week(
            str(league_id), _act_weeks, platform=platform, season=int(season)
        ) or {}
        _all_tx = []
        for _w in _act_weeks:
            for _tx in (_tx_by_week.get(_w) or []):
                if isinstance(_tx, dict):
                    _all_tx.append(_tx)
        # Newest first by created timestamp (ms).
        def _tx_ts(_t):
            try:
                return int(_t.get("created") or 0)
            except (TypeError, ValueError):
                return 0
        _all_tx.sort(key=_tx_ts, reverse=True)
        # Roster id -> display name.
        _rid_to_name = {}
        try:
            for _r in (rosters or []):
                _rid = str(_r.get("roster_id") or _r.get("id") or "")
                _nm = ""
                _owner = _r.get("owner_id") or _r.get("owner")
                for _u in (users or []):
                    if str(_u.get("user_id") or "") == str(_owner):
                        _nm = str(_u.get("display_name") or _u.get("username") or "")
                        break
                _rid_to_name[_rid] = _nm or f"Team {_rid}"
        except Exception:
            pass
        def _pname(_pid):
            try:
                _p = (players_map or {}).get(str(_pid)) or {}
                # get_players_map stores the display name under "name".
                _nm = str(_p.get("name") or _p.get("full_name") or "").strip()
                if _nm and _nm != str(_pid):
                    return _nm
                # Sleeper D/ST entries use the team abbrev as the id (e.g. "CIN").
                _pos = str(_p.get("pos") or "").upper()
                if _pos == "DEF" or (len(str(_pid)) <= 3 and str(_pid).isupper()):
                    return f"{_pid} D/ST"
                return str(_pid)
            except Exception:
                return str(_pid)
        def _rel_time(_ms):
            try:
                import datetime
                _dt = datetime.datetime.fromtimestamp(int(_ms) / 1000, tz=datetime.timezone.utc)
                _delta = datetime.datetime.now(datetime.timezone.utc) - _dt
                _hrs = int(_delta.total_seconds() // 3600)
                if _hrs < 1:
                    return "just now"
                if _hrs < 24:
                    return f"{_hrs}h ago"
                _days = _hrs // 24
                return f"{_days}d ago"
            except Exception:
                return ""
        _act_rows = []
        for _tx in _all_tx[:5]:
            _kind = str(_tx.get("type") or "")
            _ts = _rel_time(_tx.get("created"))
            _txt = ""
            if _kind == "trade":
                _rids = [str(x) for x in (_tx.get("roster_ids") or [])]
                _names = [_rid_to_name.get(r, r) for r in _rids]
                # Summarize: first team sent players to second team.
                _adds = _tx.get("adds") or {}
                _sent = []
                try:
                    for _pid, _rid in list(_adds.items())[:2]:
                        _sent.append(_pname(_pid))
                except Exception:
                    pass
                if len(_names) >= 2 and _sent:
                    _txt = (f"<strong>{html.escape(_names[0])}</strong> traded "
                            f"{html.escape(', '.join(_sent))} to "
                            f"<strong>{html.escape(_names[1])}</strong>")
                elif len(_names) >= 2:
                    _txt = (f"<strong>{html.escape(_names[0])}</strong> traded with "
                            f"<strong>{html.escape(_names[1])}</strong>")
            elif _kind in ("waiver", "free_agent"):
                _adds = _tx.get("adds") or {}
                _drops = _tx.get("drops") or {}
                try:
                    _by = ""
                    _rids = [str(x) for x in (_tx.get("roster_ids") or [])]
                    if _rids:
                        _by = _rid_to_name.get(_rids[0], "")
                    if _adds:
                        _pn = _pname(next(iter(_adds)))
                        _txt = (f"<strong>{html.escape(_by)}</strong> claimed "
                                f"{html.escape(_pn)}" if _by else f"Claimed {html.escape(_pn)}")
                    elif _drops:
                        _pn = _pname(next(iter(_drops)))
                        _txt = (f"<strong>{html.escape(_by)}</strong> dropped "
                                f"{html.escape(_pn)}" if _by else f"Dropped {html.escape(_pn)}")
                except Exception:
                    pass
            if _txt:
                _act_rows.append(
                    '<div class="cc-act-row"><div class="cc-act-tx">' + _txt
                    + (f'<div class="cc-act-time">{html.escape(_ts)}</div>' if _ts else '')
                    + '</div></div>'
                )
        if _act_rows:
            _activity_html = (
                '<section class="os-side-plain cc-act-card">'
                '<div class="cc-act-head">'
                '<div class="cc-act-head-icon"><i class="fa-solid fa-arrows-rotate" aria-hidden="true"></i></div>'
                '<div><h2 class="cc-act-title">League Activity</h2>'
                '<div class="cc-act-sub">Recent moves around the league</div></div>'
                '<button type="button" class="card-collapse-toggle cc-head-toggle" aria-label="Toggle section" aria-expanded="true" data-target="dash-act-body">&#9660;</button>'
                '</div>'
                '<div class="card-collapsible-body" id="dash-act-body">'
                + "".join(_act_rows) +
                '</div></section>'
            )
    except Exception:
        logger.debug("dashboard league activity failed", exc_info=True)
        _activity_html = ""

    # ---- Command Center: injury watch (viewer's roster only) ----
    _injury_watch_html = ""
    try:
        from app import ensure_injury_bits
        ensure_injury_bits(ctx)
        _inj_df = ctx.get("injury_df")
        _vid_s = str(viewer_roster_id or "")
        _inj_rows = []
        if _inj_df is not None and not getattr(_inj_df, "empty", True) and _vid_s:
            _has_rid = "RosterID" in getattr(_inj_df, "columns", [])
            for _, _row in _inj_df.iterrows():
                try:
                    if _has_rid and str(_row.get("RosterID") or "") != _vid_s:
                        continue
                    _st = str(_row.get("Status") or _row.get("Injury") or "").strip()
                    if not _st or _st.lower() in ("active", "healthy", ""):
                        continue
                    _pn = str(_row.get("Player") or "")
                    _pos = str(_row.get("Pos") or "")
                    _inj_rows.append(
                        '<div class="cc-inj-row">'
                        f'<span><strong>{html.escape(_pn)}</strong> '
                        f'<span class="cc-muted">{html.escape(_pos)}</span></span>'
                        f'<span class="cc-inj-status">{html.escape(_st)}</span>'
                        '</div>'
                    )
                except Exception:
                    continue
                if len(_inj_rows) >= 6:
                    break
        if _inj_rows:
            _injury_watch_html = (
                '<section class="os-side-plain cc-inj-card">'
                '<div class="cc-inj-head">'
                '<div class="cc-inj-head-icon"><i class="fa-solid fa-heart-crack" aria-hidden="true"></i></div>'
                '<div><h2 class="cc-inj-title">Injury Watch</h2>'
                '<div class="cc-inj-sub">Your roster</div></div>'
                '<button type="button" class="card-collapse-toggle cc-head-toggle" aria-label="Toggle section" aria-expanded="true" data-target="dash-inj-body">&#9660;</button>'
                '</div>'
                '<div class="card-collapsible-body" id="dash-inj-body">'
                + "".join(_inj_rows) +
                '</div></section>'
            )
    except Exception:
        logger.debug("dashboard injury watch failed", exc_info=True)
        _injury_watch_html = ""

    # Standings card subtitle: "LeagueName · N teams" (character-card header).
    _st_league_name = html.escape(str((ctx.get("league") or {}).get("name") or "League"))
    _st_team_count = len(rosters or [])
    _st_sub = (f"{_st_league_name} &middot; {_st_team_count} team"
               f"{'' if _st_team_count == 1 else 's'}")
    body = f"""
    <div class="os-layout">
      <aside class="os-left-col os-side-rail">
        {gm_card_html}
        <section class="os-side-plain cc-st-card" data-section="standings">
          <div class="cc-st-head">
            <div class="cc-st-head-icon"><i class="fa-solid fa-clipboard-list" aria-hidden="true"></i></div>
            <div class="cc-st-head-title">{_section_title_link("Standings", "league_pages.page_standings", platform, season, league_id)}</div>
            <div class="cc-st-sub">{_st_sub}</div>
            <button type="button" class="card-collapse-toggle cc-head-toggle" aria-label="Toggle section" aria-expanded="true" data-target="dash-standings-body">&#9660;</button>
          </div>
          <div class="card-collapsible-body" id="dash-standings-body">
            {standings_html}
          </div>
        </section>
        {season_review_html}
      </aside>

      <main class="os-main-col">
        {_hero_strip_html}

        {_rz_cta_html}

        {_matchup_preview_html}

        {_action_queue_html}

        {usage_movers_html}

        {_bb_outlook_html}

        <div id="sinceLastVisitCard" class="slv-wrap" data-slv-init="1"></div>

        {bench_check_html}
      </main>

      <aside class="os-right-col os-side-rail">
        {awards_html}
        {_activity_html}
        {_injury_watch_html}
      </aside>
    </div>
    <script>
    (function(){{
      // Dashboard redesign: sections stack, no jump-nav tab switching.
      // Collapse toggles still work via the shared card-collapse delegate.
    }})();
    </script>
    <script>
    (function() {{
      var el = document.getElementById('dash-playoff-tile');
      if (!el) return;
      if (el.classList.contains('is-loaded')) return;
      // Mirror of _fmt_playoff_pct_display in app.py: undecided odds read
      // at one decimal, floored to tenths so a 99.9 never displays as 100;
      // literal 100 / 0 only when the raw value is exact.
      function poFloorTenths(v) {{
        var raw = Number(v) || 0;
        if (raw >= 100) return 100;
        if (raw <= 0) return 0;
        return Math.floor(raw * 10 + 1e-9) / 10;
      }}
      function poFmtPct(v) {{
        var raw = Number(v) || 0;
        if (raw >= 100) return '100';
        if (raw <= 0) return '0';
        return poFloorTenths(raw).toFixed(1);
      }}
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
            if (valEl) valEl.textContent = '--';
            if (subEl) subEl.textContent = 'Odds unavailable';
            el.classList.add('is-loaded');
            return;
          }}
          var rawPct = Number(row.playoff_pct) || 0;
          var pct = poFloorTenths(rawPct);
          var settled = rawPct >= 100 || rawPct <= 0;
          if (valEl) {{
            if (window.brCountUp) window.brCountUp(valEl, {{ to: pct, dp: settled ? 0 : 1, suffix: '%', dur: 800 }});
            else valEl.textContent = poFmtPct(rawPct) + '%';
          }}
          var sub;
          if (d.is_complete || row.is_complete) {{
            sub = rawPct >= 100 ? 'Clinched' : (rawPct <= 0 ? 'Eliminated' : 'Playoff bound');
          }} else {{
            var first = poFloorTenths(row.first_seed_pct);
            var bye = poFloorTenths(row.bye_pct);
            sub = first > 0 ? (poFmtPct(row.first_seed_pct) + '% top seed')
                : (bye > 0 ? (poFmtPct(row.bye_pct) + '% first-round bye') : 'to make the playoffs');
          }}
          if (subEl) subEl.textContent = sub;
          el.classList.add('is-loaded');
        }})
        .catch(function() {{
          var valEl = document.getElementById('dash-playoff-val');
          var subEl = document.getElementById('dash-playoff-sub');
          if (valEl) valEl.textContent = '--';
          if (subEl) subEl.textContent = 'Odds unavailable';
          el.classList.add('is-loaded');
        }});
    }})();
    </script>
    """

    return body

