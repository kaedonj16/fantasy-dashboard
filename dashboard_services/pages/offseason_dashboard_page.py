"""Offseason dashboard HTML builder.

Moved from app.py so the Flask monolith can keep shrinking. Helpers that still
live in app.py are lazy-imported inside the builder (request time).
"""
from __future__ import annotations

from typing import Dict

def build_offseason_dashboard_body(ctx: dict) -> str:
    from app import (  # noqa: E402  (lazy: avoids a circular import at module load)
        EASTERN,
        _build_waiver_targets_rows,
        _league_is_redraft,
        _nfl_regular_season_kickoff_ms,
        _playoff_sim_cached,
        _playoff_tile_from_cache,
        _rank_move_html,
        _ranking_movement,
        _render_season_review_card,
        _safe_int,
        _section_title_link,
        _team_pick_value,
        apply_te_premium,
        build_teams_overview,
        datetime,
        get_model_value_table_cached,
        html,
        load_pick_value_table,
        logger,
        render_teams_sidebar,
        te_premium_from_settings,
        url_for,
    )

    league = ctx["league"]
    platform = ctx["platform"]
    season = ctx["season"]
    rosters = ctx["rosters"]
    users = ctx["users"]
    roster_map = ctx["roster_map"]
    picks_by_roster = ctx.get("picks_by_roster", {})
    players_index = ctx["players_index"]
    players_map = ctx["players_map"]
    # Read the live cached model table directly (the same source the player modal
    # uses) instead of ctx["model_value_table"], which gets pinned into the
    # longer-lived league-context cache and goes stale — making the waiver list
    # and roster values disagree with the modal after a value rebuild.
    model_value_table = list(get_model_value_table_cached() or []) or (ctx.get("model_value_table") or [])

    viewer = ctx.get("viewer") or {}
    viewer_roster_id = viewer.get("viewer_roster_id")

    # Season-in-review of the just-completed season (renders nothing if the
    # offseason ctx has no finalized weekly data for the viewer).
    season_review_html = _render_season_review_card(
        ctx, viewer_roster_id, ctx.get("df_weekly"), ctx.get("team_stats")
    )

    latest_draft = ctx.get("latest_draft")
    countdown_label = "Draft countdown"
    draft_text = "Draft date not set"
    countdown_text = "TBD"
    draft_subtext = "Set once your league schedules the draft."

    # First NFL regular-season game kickoff (from the app's schedule), not the
    # preseason — used for the no-draft-scheduled tile and the post-draft
    # countdown alike. See _nfl_regular_season_kickoff_ms.
    _week1_delta = None
    _week1_date_txt = ""
    _week1_ts_ms = _nfl_regular_season_kickoff_ms(int(season))
    if _week1_ts_ms:
        try:
            _week1_dt = datetime.fromtimestamp(_week1_ts_ms / 1000, tz=EASTERN)
            _week1_delta = (_week1_dt.date() - datetime.now(EASTERN).date()).days
            _week1_date_txt = _week1_dt.strftime("%b %d, %Y")
        except Exception as e:
            logger.info(f"[offseason] kickoff parse failed: {e}")

    draft_ts_ms = None

    if isinstance(latest_draft, dict):
        draft_ts_ms = _safe_int(latest_draft.get("start_time"))

    if draft_ts_ms is None:
        draft_ts_ms = _safe_int(league.get("draft_day"))

    # The timestamp the card counts down to: the draft while it's upcoming, then
    # NFL Week 1 kickoff once the draft is done. JS mirrors this (see below).
    _target_ts_ms = draft_ts_ms or 0
    _draft_status = (
        str((latest_draft or {}).get("status") or "").lower()
        if isinstance(latest_draft, dict) else ""
    )

    if draft_ts_ms:
        try:
            draft_dt = datetime.fromtimestamp(draft_ts_ms / 1000, tz=EASTERN)
            now_dt = datetime.now(EASTERN)
            delta_days = (draft_dt.date() - now_dt.date()).days
            # "Done" = the draft is complete, or its scheduled time is in the past.
            draft_done = _draft_status == "complete" or delta_days < 0

            if not draft_done:
                # Draft hasn't happened yet — count down to it.
                countdown_label = "Draft countdown"
                countdown_text = "Today!" if delta_days == 0 else f"{delta_days} days"
                draft_text = draft_dt.strftime("%b %d, %Y at %I:%M %p %Z")
                draft_subtext = "Countdown to your next league draft."
                _target_ts_ms = draft_ts_ms
            else:
                # Draft is finished — count down to NFL Week 1 kickoff, and show
                # the Week 1 date (not the stale draft date) as the sub-line.
                countdown_label = "Season kickoff"
                _target_ts_ms = _week1_ts_ms or 0
                if _week1_delta is not None:
                    if _week1_delta > 0:
                        countdown_text = f"{_week1_delta} days"
                        draft_text = f"Week 1 kicks off {_week1_date_txt}"
                        draft_subtext = "Countdown to Week 1 kickoff."
                    elif _week1_delta == 0:
                        countdown_text = "Today!"
                        draft_text = "Week 1 starts today"
                        draft_subtext = "Week 1 starts today!"
                    else:
                        countdown_text = "Season started"
                        draft_text = f"Week 1 kicked off {_week1_date_txt}"
                        draft_subtext = "Week 1 is underway!"
                else:
                    countdown_text = "Draft complete"
                    draft_text = "Draft has finished"
                    draft_subtext = "Awaiting season start date."
        except Exception:
            logger.debug("suppressed exception", exc_info=True)
    elif _week1_delta is not None and _week1_delta >= 0:
        # No draft on the calendar: count down to the season itself instead of
        # sitting on a dead "TBD" tile all summer.
        countdown_label = "Season kickoff"
        _target_ts_ms = _week1_ts_ms or 0
        if _week1_delta == 0:
            countdown_text = "Today!"
            draft_text = "Week 1 starts today"
        else:
            countdown_text = f"{_week1_delta} days"
            draft_text = f"Week 1 kicks off {_week1_date_txt}"
        draft_subtext = "Draft date not set - schedule it before kickoff."

    teams_ctx = build_teams_overview(
        rosters=rosters,
        users_list=users,
        picks_by_roster=picks_by_roster,
        players=players_map,
        players_index=players_index,
        teams_index=ctx["teams_index"],
        platform=platform,
    )
    teams_sidebar_html = render_teams_sidebar(teams_ctx)

    values_by_id = {}
    for row in model_value_table:
        if isinstance(row, dict) and row.get("id") is not None:
            try:
                values_by_id[str(row["id"])] = float(row.get("value") or 0.0)
            except Exception:
                values_by_id[str(row["id"])] = 0.0

    # Build pick-value lookup from WLS-derived table (overlays FantasyCalc/DynastyProcess)
    pick_by_key: Dict[str, float] = load_pick_value_table() or {}

    # TE-premium scaling so the snapshot total matches the team modal (and the
    # trade calculator / activity feed), which scale TE values up for leagues
    # that award bonus points per TE reception.
    _tep_snap = te_premium_from_settings(ctx.get("scoring_settings"))

    roster_cards = []

    for r in rosters:
        rid = str(r.get("roster_id"))
        team_name = roster_map.get(rid, f"Roster {rid}")
        player_ids = [str(pid) for pid in (r.get("players") or [])]
        roster_value = sum(
            apply_te_premium(
                values_by_id.get(pid, 0.0),
                (players_index.get(pid) or {}).get("pos"),
                _tep_snap,
            )
            for pid in player_ids
        )
        team_picks = picks_by_roster.get(rid, []) if isinstance(picks_by_roster, dict) else []
        pick_count = len(team_picks)

        # Add pick values to total roster value
        league_id_str = str(ctx.get("league_id") or "")
        roster_value += _team_pick_value(
            team_picks, pick_by_key,
            platform=platform, league_id=league_id_str, season=_safe_int(season, 0),
        )

        first_round_count = 0
        for pk in team_picks:
            try:
                if int(pk.get("round") or 0) == 1:
                    first_round_count += 1
            except Exception:
                logger.debug("suppressed exception", exc_info=True)

        roster_cards.append({
            "team_name": team_name,
            "roster_value": roster_value,
            "pick_count": pick_count,
            "first_round_count": first_round_count,
            "roster_id": rid,
        })

    roster_cards.sort(key=lambda x: x["roster_value"], reverse=True)

    # Day-over-day movement in the value ranking (▲/▼ spots), so a trade shows up.
    # Its own kind: this leaderboard's value math is computed here, independently
    # of the standings value table, so they must not share a daily snapshot.
    _snap_mv = _ranking_movement(
        ctx.get("league_id"), season, "dash_value",
        [c["roster_id"] for c in roster_cards])

    # Value leaderboard: rank medallion (gold/silver/bronze for the top 3), a bar
    # scaled to the leader so magnitudes read at a glance, and canonical chips.
    max_value = roster_cards[0]["roster_value"] if roster_cards and roster_cards[0]["roster_value"] > 0 else 1
    ranked_snapshot_html = []
    for idx, card in enumerate(roster_cards, start=1):
        rv = card["roster_value"]
        pct = max(4, min(100, round(rv / max_value * 100)))
        fr = card["first_round_count"]
        pc = card["pick_count"]
        chips = []
        if fr > 0:
            chips.append(f"<span class='chip chip--sm'>{fr} first{'s' if fr != 1 else ''}</span>")
        if pc > 0:
            chips.append(f"<span class='chip chip--sm'>{pc} future pick{'s' if pc != 1 else ''}</span>")
        chips_html = f"<div class=\"os-snap-chips\">{''.join(chips)}</div>" if chips else ""
        medal_cls = f"is-{idx}" if idx <= 3 else ""
        ranked_snapshot_html.append(
            f"""
            <div class="os-snap-row {medal_cls} team-clickable" style="cursor:pointer;" data-roster-id="{card['roster_id']}" data-team-name="{card['team_name']}">
              <div class="os-snap-medal">{idx}{_rank_move_html(_snap_mv.get(str(card['roster_id'])))}</div>
              <div class="os-snap-body">
                <div class="os-snap-head">
                  <div class="os-snap-name">{card['team_name']}</div>
                  <div class="os-snap-valblock">
                    <div class="os-snap-value">{rv:,.0f}</div>
                    <div class="os-snap-kicker">Total value</div>
                  </div>
                </div>
                <div class="os-snap-bar"><span style="width:{pct}%"></span></div>
                {chips_html}
              </div>
            </div>
            """
        )

    roster_cards_html = "".join(ranked_snapshot_html)

    roster_leader = roster_cards[0]["team_name"] if roster_cards else "N/A"
    highest_roster_value = f"{roster_cards[0]['roster_value']:,.0f}" if roster_cards else "0"

    # Calculate total draft capital across all rosters
    total_draft_capital = 0.0
    for roster in rosters:
        roster_id = str(roster.get("roster_id"))
        player_ids = [str(pid) for pid in (roster.get("players") or [])]
        roster_value = sum(values_by_id.get(pid, 0.0) for pid in player_ids)
        team_picks = picks_by_roster.get(roster_id, []) if isinstance(picks_by_roster, dict) else []
        pick_count = len(team_picks)

        # Add pick values to total roster value
        roster_value += _team_pick_value(team_picks, pick_by_key)

        total_draft_capital += roster_value

    # Hero stat tile #3. Dynasty leagues care about future draft capital; redraft
    # and keeper leagues carry no tradeable picks, so that number collapses to the
    # league's total value and reads as dead weight. Swap it for a League Parity
    # read — how lopsided roster strength is (leader vs. cellar) — which tells a
    # redraft manager how competitive the field is heading into the season.
    _breakdown_href = url_for("page_teams", platform=platform, season=season,
                              league_id=ctx.get("league_id", ""))
    if _league_is_redraft(ctx):
        _rvs = [c["roster_value"] for c in roster_cards if c["roster_value"] > 0]
        _ratio = (_rvs[0] / _rvs[-1]) if len(_rvs) >= 2 and _rvs[-1] > 0 else None
        if _ratio is None:
            _parity_word, _parity_sub = "&mdash;", "Not enough roster data yet"
        else:
            if _ratio > 1.6:
                _parity_word = "Top-heavy"
            elif _ratio > 1.25:
                _parity_word = "Balanced"
            else:
                _parity_word = "Wide open"
            _parity_sub = f"Leader worth {_ratio:.1f}&times; the last-place roster"
        capital_tile_html = f"""
            <div class="os-stat-card">
              <div class="os-stat-label">League Parity</div>
              <div class="os-stat-value">{_parity_word}</div>
              <div class="os-stat-sub">{_parity_sub}</div>
            </div>"""
    else:
        capital_tile_html = f"""
            <div class="os-stat-card">
              <div class="os-stat-label">Draft Capital Index</div>
              <div class="os-stat-value">{total_draft_capital:,.0f}</div>
              <div class="os-stat-sub"><a class="os-stat-sub-link" href="{_breakdown_href}">View team breakdown &rarr;</a></div>
            </div>"""

    # Waiver card rows are shared with the in-season Season Hub.
    top_waiver_assets_html = _build_waiver_targets_rows(ctx, model_value_table)

    gm_card_html = ""
    if viewer_roster_id:
        # Show button to generate GM memo instead of auto-generating
        gm_card_html = f"""
        <section class="os-card">
          <div class="os-section-head">
            <div class="os-section-head-content">
              <h2 class="os-section-title">Front Office Report</h2>
              <div class="os-section-subtitle">{viewer.get("viewer_team_name") or "Your Team"}</div>
            </div>
            <div class="os-section-head-actions">
              <button type="button" id="generateGmMemoBtn" class="recap-generate-btn" 
                      data-league-id="{ctx.get('league_id')}" 
                      data-season="{ctx.get('season')}" 
                      data-platform="{ctx.get('platform')}" 
                      data-viewer-roster-id="{viewer_roster_id}">
                Generate Report
              </button>
              <button type="button" class="card-collapse-toggle" aria-label="Toggle section" aria-expanded="true" data-target="gm-memo-body">▼</button>
            </div>
          </div>
          <div class="os-ai-copy card-collapsible-body" id="gm-memo-body">
            <div class="otc-ai-empty" id="gm-memo-empty">
              <div class="otc-ai-empty-sub">
                Get personalized analysis on your roster, trade targets, and offseason strategy.
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
        </section>
        """

    # Projected playoff-odds tile — serve a warm cache on first paint; a cold
    # sim is kicked off in the background and the client fills or removes the tile.
    _os_playoff_tile_html = ""
    if viewer_roster_id:
        _os_val, _os_sub, _os_loaded = "-", "Projecting&hellip;", ""
        try:
            _os_warm = _playoff_sim_cached(ctx, platform, block=False) or []
            _os_filled = _playoff_tile_from_cache(_os_warm, viewer_roster_id, projected=True)
            if _os_filled:
                _os_val = f"{_os_filled[0]}%"
                _os_sub = html.escape(_os_filled[1])
                _os_loaded = " is-loaded"
        except Exception:
            logger.debug("offseason playoff prefetch failed", exc_info=True)
        _os_playoff_tile_html = f"""<div class="os-stat-card os-stat-playoff{_os_loaded}" id="os-playoff-tile"
              data-platform="{html.escape(str(platform))}"
              data-league="{html.escape(str(ctx.get('league_id', '')))}"
              data-season="{html.escape(str(season))}"
              data-roster="{html.escape(str(viewer_roster_id))}">
              <div class="os-stat-label">Projected playoff odds</div>
              <div class="os-stat-value" id="os-playoff-val">{_os_val}</div>
              <div class="os-stat-sub" id="os-playoff-sub">{_os_sub}</div>
            </div>"""

    body = f"""
    <div class="os-layout">
      <aside class="os-left-col os-tab-panel" id="os-jump-roster">
        <section class="os-card os-card-soft os-col-fill">
          <div class="os-section-head">
            <div class="os-section-head-content">
              {_section_title_link("Offseason Team Snapshot", "page_teams", platform, season, ctx.get("league_id"))}
              <div class="os-section-subtitle">Roster value and future capital across the league</div>
            </div>
            <div class="os-section-head-actions">
              <button type="button" class="card-collapse-toggle" aria-label="Toggle section" aria-expanded="true" data-target="team-snapshot-body">▼</button>
            </div>
          </div>
          <div class="os-snapshot-list card-collapsible-body" id="team-snapshot-body">
            {roster_cards_html or "<p>No offseason roster data available yet.</p>"}
          </div>
        </section>
      </aside>

      <main class="os-main-col">
        <section class="os-hero-card">
          <div class="os-hero-top">
            <div>
              <h1 class="os-hero-title">Offseason Hub</h1>
              <p class="os-hero-copy">
                Focus on roster building, draft prep, waiver value, and trade opportunities.
              </p>
            </div>
          </div>

          <div class="os-hero-stats">
            <div class="os-stat-card" id="osDraftCdCard" data-draft-ts="{draft_ts_ms or 0}" data-week1-ts="{_week1_ts_ms or 0}" data-target-ts="{_target_ts_ms or 0}" data-detect-url="/api/draft/detect?platform={platform}&league_id={ctx.get('league_id', '')}&season={season}">
              <div class="os-stat-label" id="osDraftCdLabel">{countdown_label}</div>
              <div class="os-stat-value" id="osDraftCdVal">{countdown_text}</div>
              <div class="os-stat-sub" id="osDraftCdSub">{draft_text}</div>
            </div>
            <div class="os-stat-card">
              <div class="os-stat-label">Value leader</div>
              <div class="os-stat-value os-stat-value-name">{roster_leader}</div>
              <div class="os-stat-sub">{highest_roster_value} roster value</div>
            </div>
            {capital_tile_html}
            {_os_playoff_tile_html}
          </div>

          <div class="os-hero-footer" id="osDraftSubtext">
            {draft_subtext}
          </div>
        </section>
        <script>
        (function(){{
          var card = document.getElementById('osDraftCdCard');
          if (!card) return;
          var week1Ts  = parseInt(card.getAttribute('data-week1-ts')  || '0', 10);
          var targetTs = parseInt(card.getAttribute('data-target-ts') || card.getAttribute('data-draft-ts') || '0', 10);
          var detectUrl = card.getAttribute('data-detect-url');
          var labelEl = document.getElementById('osDraftCdLabel');
          var valEl = document.getElementById('osDraftCdVal');
          var subEl = document.getElementById('osDraftCdSub');
          var subtextEl = document.getElementById('osDraftSubtext');
          function fmtCountdown(ms){{
            if (ms <= 0) return 'Kickoff!';
            var t = Math.floor(ms / 1000);
            var d = Math.floor(t / 86400);
            var h = Math.floor((t % 86400) / 3600);
            var m = Math.floor((t % 3600) / 60);
            var s = t % 60;
            var pad = function(n){{ return (n < 10 ? '0' : '') + n; }};
            var clock = pad(h) + ':' + pad(m) + ':' + pad(s);
            return d > 0 ? (d + 'd ' + clock) : clock;
          }}
          function tick(){{
            if (!targetTs) return;
            var remaining = targetTs - Date.now();
            if (remaining > 0 && valEl) valEl.textContent = fmtCountdown(remaining);
          }}
          function showDraft(ts){{
            targetTs = ts;
            if (labelEl) labelEl.textContent = 'Draft countdown';
            if (subEl) subEl.textContent = new Date(ts).toLocaleString('en-US', {{ month:'short', day:'numeric', year:'numeric', hour:'numeric', minute:'2-digit', timeZoneName:'short' }});
            if (subtextEl) subtextEl.textContent = 'Countdown to your next league draft.';
            tick();
          }}
          function showWeek1(){{
            targetTs = week1Ts;
            if (labelEl) labelEl.textContent = 'Season kickoff';
            if (week1Ts && subEl) subEl.textContent = 'Week 1 kicks off ' + new Date(week1Ts).toLocaleDateString('en-US', {{ month:'short', day:'numeric', year:'numeric' }});
            if (subtextEl) subtextEl.textContent = 'Countdown to Week 1 kickoff.';
            tick();
          }}
          // Only an UPCOMING, not-yet-complete draft resets the target to the
          // draft. Once every detected draft is complete/past, count to Week 1.
          function applyDrafts(drafts){{
            if (!Array.isArray(drafts)) return;
            var now = Date.now();
            var best = null;
            drafts.forEach(function(d){{
              var st = parseInt(d.start_time || 0, 10);
              var done = String(d.status || '').toLowerCase() === 'complete';
              if (st > now && !done && (best === null || st < best)) best = st;
            }});
            if (best !== null){{
              if (best !== targetTs) showDraft(best);
            }} else if (week1Ts && targetTs !== week1Ts){{
              showWeek1();
            }}
          }}
          function refresh(){{
            if (!detectUrl) return;
            fetch(detectUrl, {{ cache: 'no-store' }}).then(function(r){{ return r.json(); }}).then(function(resp){{
              applyDrafts((resp && resp.drafts) || []);
            }}).catch(function(){{}});
          }}
          tick();
          setInterval(tick, 1000);
          setInterval(refresh, 30000);
          refresh();
        }})();
        </script>
        <script>
        (function() {{
          var el = document.getElementById('os-playoff-tile');
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
              // Only a genuine preseason projection belongs here; if the sim
              // couldn't project (no schedule/rosters yet) there's nothing to show.
              if (!row || !row.is_projected) {{ el.remove(); return; }}
              var pct = Math.round(row.playoff_pct || 0);
              var valEl = document.getElementById('os-playoff-val');
              var subEl = document.getElementById('os-playoff-sub');
              if (valEl) {{
                if (window.brCountUp) window.brCountUp(valEl, {{ to: pct, dp: 0, suffix: '%', dur: 800 }});
                else valEl.textContent = pct + '%';
              }}
              var first = Math.round(row.first_seed_pct || 0);
              if (subEl) subEl.textContent = first > 0
                ? ('Projected · ' + first + '% top seed')
                : 'Projected from current rosters';
              el.classList.add('is-loaded');
            }})
            .catch(function() {{ el.remove(); }});
        }})();
        </script>

        <nav class="os-jump-nav" aria-label="Jump to section">
          <button type="button" class="active" data-jump="os-jump-report">Report</button>
          <button type="button" data-jump="os-jump-roster">Team Values</button>
          <button type="button" data-jump="os-jump-waivers">Waivers</button>
          <button type="button" data-jump="os-jump-teams">Roster</button>
        </nav>

        <div id="sinceLastVisitCard" class="slv-wrap" data-slv-init="1"></div>

        <div id="os-jump-report" class="os-tab-panel os-tab-active">
          {gm_card_html}
          {season_review_html}
        </div>

        <section class="os-card os-col-fill os-tab-panel" id="os-jump-waivers">
          <div class="os-section-head">
            <div class="os-section-head-content">
              {_section_title_link("Waiver Wire Targets", "league_pages.page_waivers", platform, season, ctx.get("league_id"))}
              <div class="os-section-subtitle">Smart pickups based on value + trend + breakout potential</div>
            </div>
            <div class="os-section-head-actions">
              <button type="button" class="card-collapse-toggle" aria-label="Toggle section" aria-expanded="true" data-target="waiver-assets-body">▼</button>
            </div>
          </div>
          <div class="os-waiver-list card-collapsible-body" id="waiver-assets-body">
            {top_waiver_assets_html or "<p>No waiver values available yet.</p>"}
          </div>
        </section>
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
    """
    return body

