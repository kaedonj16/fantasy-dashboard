import logging
from typing import Optional

logger = logging.getLogger(__name__)

SUPPORTED_LEAGUE_SIZES = [8, 10, 12, 14]

SUPPORTED_SCORING_FORMATS = [("ppr", "PPR"), ("half", "Half"), ("std", "STD")]


def build_trade_calculator_body(
        league_id: Optional[str],
        season: Optional[int],
        num_teams: Optional[int] = None,
        scoring_format: Optional[str] = None,
        viewer_roster_id: Optional[str] = None,
        has_premium: bool = False,
        is_superflex: bool = False,
        platform: Optional[str] = None,
) -> str:
    league_val = league_id or ""
    season_val = season if season is not None else ""
    viewer_roster_val = viewer_roster_id or ""
    is_guest = not league_id
    platform_val = platform or "sleeper"

    # Get trade count from database
    trade_count = "150,000+"
    try:
        from dashboard_services.db import get_conn
        with get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trade_intel_trades")
            result = cursor.fetchone()
            # Handle both tuple and dictionary return formats
            if isinstance(result, dict):
                count = result.get('count', 0)
            else:
                count = result[0] if result else 0
            trade_count = f"{count:,}"
    except Exception as e:
        logger.warning("Trade count error: %s", e)
        pass

    # Clamp logged-in league size to nearest supported value
    if num_teams and not is_guest:
        closest = min(SUPPORTED_LEAGUE_SIZES, key=lambda s: abs(s - int(num_teams)))
        num_teams_val = closest
    else:
        num_teams_val = 10  # default for guest

    # Scoring format: use league value when logged in, default PPR for guests
    scoring_format_val = (scoring_format or "ppr").strip().lower()
    if scoring_format_val not in {s for s, _ in SUPPORTED_SCORING_FORMATS}:
        scoring_format_val = "ppr"

    # Create dynamic URLs based on login status
    if is_guest:
        breakouts_url = "#"
        breakouts_link_text = "Sign in to see more →"
        breakouts_link_class = "otc-view-all-link otc-guest-link"
        players_url = "/players"
    else:
        breakouts_url = f"/{platform_val}/{season_val}/{league_val}/breakouts"
        breakouts_link_text = "View All →"
        breakouts_link_class = "otc-view-all-link"
        players_url = f"/{platform_val}/{season_val}/{league_val}/players"

    # ----------------------------------------------------------------
    # Pre-compute all conditional HTML blocks outside the f-string
    # to avoid nested triple-quote SyntaxErrors
    # ----------------------------------------------------------------

    # Show owner tags for all users (updated by viewer side selection)
    side_a_owner_tag = '<div class="otc-team-owner-tag" id="sideAOwnerTag">Your side</div>'
    side_b_owner_tag = '<div class="otc-team-owner-tag otc-team-owner-tag-muted" id="sideBOwnerTag">Other side</div>'

    analyze_btn_disabled = 'disabled' if is_guest else ''
    analyze_btn_label = 'Sign In to Analyze' if is_guest else 'Analyze Trade'

    ai_sub_text = 'AI-powered trade analysis for dynasty leagues' if is_guest else 'Personalized to your team direction and roster lens'

    team_select_block = '' if is_guest else """
                <div class="otc-summary-team-select">
                  <select id="teamSelect" class="otc-team-select-dropdown" required>
                    <option value="">Select your team...</option>
                  </select>
                </div>
    """

    ai_empty_title = 'Sign In for AI Analysis' if is_guest else 'Waiting on a deal'
    ai_empty_sub = (
        'Connect your league to get personalized trade analysis powered by AI.'
        if is_guest else
        'Once both sides have assets, this panel can explain whether the trade fits your team build.'
    )

    is_guest_str = 'true' if is_guest else 'false'
    has_premium_str = 'true' if has_premium else 'false'

    # League type toggle: checkbox with 1QB and SF labels
    sf_checked = ' checked' if is_superflex else ''
    league_type_block = f"""
              <div class="otc-ctrl-group" id="leagueTypeControl">
                <span class="otc-toggle-label">1QB</span>
                <label class="otc-pill-switch">
                  <input type="checkbox" id="leagueTypeToggle"{sf_checked}>
                  <span class="otc-pill-track"></span>
                </label>
                <span class="otc-toggle-label">SF</span>
              </div>"""

    # League size dropdown: always shown
    size_options = ""
    for s in SUPPORTED_LEAGUE_SIZES:
        selected = 'selected' if s == num_teams_val else ''
        size_options += f'<option value="{s}" {selected}>{s}-team</option>\n'
    league_size_block = f"""
              <div class="otc-ctrl-group otc-toggle-divider" id="leagueSizeControl">
                <select class="otc-ctrl-select" id="leagueSizeSelect" name="leagueSize">
                  {size_options}
                </select>
              </div>"""

    # Scoring format dropdown: always shown
    fmt_options = ""
    for val, label in SUPPORTED_SCORING_FORMATS:
        selected = 'selected' if val == scoring_format_val else ''
        fmt_options += f'<option value="{val}" {selected}>{label}</option>\n'
    scoring_format_block = f"""
              <div class="otc-ctrl-group" id="scoringFormatControl">
                <select class="otc-ctrl-select" id="scoringFormatSelect" name="scoringFormat" style="width: 60px;">
                  {fmt_options}
                </select>
              </div>"""

    league_type_js = repr("sf" if is_superflex else "1qb")
    return f"""
    <script>var _leagueType = {league_type_js}; var _leagueSize = {num_teams_val};</script>
    <div class="otc-layout">
      <main class="otc-main">
        <input type="hidden" id="leagueIdInput" value="{league_val}">
        <input type="hidden" id="seasonInput" value="{season_val}">
        <input type="hidden" id="viewerRosterIdInput" value="{viewer_roster_val}">
        <input type="hidden" id="viewerSideInput" value="a">
        <input type="hidden" id="isGuestMode" value="{is_guest_str}">
        <input type="hidden" id="otcHasPremium" value="{has_premium_str}">

        <div class="otc-shell">
          <div class="otc-main-tabs">
            <button class="otc-main-tab is-active" data-tab="calculator">Calculator</button>
            <button class="otc-main-tab" data-tab="suggestions">
              Suggestions <span class="nav-pro-badge">PRO</span>
            </button>
          </div>

          <div id="otcCalcTab">
          <div class="otc-page-head">
            <div class="otc-page-title-wrap">
              <div style="display: flex; align-items: center; gap: 8px;">
                <h1 class="otc-page-title">Trade Calculator</h1>
                <div class="otc-info-tooltip-wrapper">
                  <button type="button" class="otc-info-btn" id="otcInfoBtn">ⓘ</button>
                  <div class="otc-info-tooltip" id="otcInfoTooltip" style="display:none;">
                    <div class="otc-info-tooltip-header">BR Value Model</div>
                    <div class="otc-info-tooltip-body">
                      <p>Player values are built directly from real dynasty trades, capturing how the market prices players and picks in actual deals.</p>
                      <p>We translate over <strong>{trade_count}</strong> trade relationships into a unified value scale, then layer in production, age trajectory, and role stability to sharpen the signal.</p>
                    </div>
                  </div>
                </div>
              </div>
              <p class="otc-page-copy">
                Compare both sides of a deal using BR values, balance, and roster-building context.
              </p>
            </div>
            <div class="otc-page-head-controls">
              <div class="otc-viewer-toggles">
                <label class="otc-viewer-toggle">
                  <input type="radio" name="viewerSide" value="a" checked>
                  <span>Team 1</span>
                </label>
                <label class="otc-viewer-toggle">
                  <input type="radio" name="viewerSide" value="b">
                  <span>Team 2</span>
                </label>
              </div>
              <div class="otc-settings-row">
                {league_size_block}
                {scoring_format_block}
                {league_type_block}
              </div>
            </div>
          </div>

          <div class="otc-builder-grid">
            <section class="otc-team-card">
              <div class="otc-team-head">
                <div>
                  <h2 class="otc-team-title">Team 1 gets...</h2>
                </div>
                <div class="otc-team-head-right">
                  {side_a_owner_tag}
                </div>
              </div>

              <div class="otc-slot">
                <div class="otc-slot-empty" id="sideAEmptyState">
                  <div class="otc-slot-empty-title">Add players or picks</div>
                  <div class="otc-slot-empty-sub">
                    Search players to build the first side of the trade.
                  </div>
                </div>

                <div class="otc-search-wrap">
                  <div class="search-wrapper">
                    <input
                      id="sideASearch"
                      class="otc-search-input"
                      type="text"
                      autocomplete="off"
                      placeholder="Start typing a player name..." />
                    <div id="sideADropdown" class="dropdown otc-search-dropdown" style="display:none;"></div>
                  </div>
                </div>

                <div class="chips" id="sideAChips"></div>
              </div>
            </section>

            <section class="otc-team-card">
              <div class="otc-team-head">
                <div>
                  <h2 class="otc-team-title">Team 2 gets...</h2>
                </div>
                <div class="otc-team-head-right">
                  {side_b_owner_tag}
                </div>
              </div>

              <div class="otc-slot">
                <div class="otc-slot-empty" id="sideBEmptyState">
                  <div class="otc-slot-empty-title">Add players or picks</div>
                  <div class="otc-slot-empty-sub">
                    Search players to build the second side of the trade.
                  </div>
                </div>

                <div class="otc-search-wrap">
                  <div class="search-wrapper">
                    <input
                      id="sideBSearch"
                      class="otc-search-input"
                      type="text"
                      autocomplete="off"
                      placeholder="Start typing a player name..." />
                    <div id="sideBDropdown" class="dropdown otc-search-dropdown" style="display:none;"></div>
                  </div>
                </div>

                <div class="chips" id="sideBChips"></div>
              </div>
            </section>
          </div>

          <div class="otc-lower-grid">
            <section class="otc-summary-card">
              <div class="otc-summary-head">
                <div>
                  <h2 class="otc-summary-title">Trade Summary</h2>
                  <div class="otc-summary-sub">Live balance as you add assets</div>
                </div>
                <div class="otc-summary-actions">
                  {team_select_block}
                  <button type="button" id="clearTradeBtn" class="otc-clear-btn" {analyze_btn_disabled}>{analyze_btn_label}</button>
                  <button type="button" id="shareTradeBtn" class="otc-share-btn" title="Copy shareable link">
                    <svg class="otc-share-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <circle cx="18" cy="5" r="3"></circle>
                      <circle cx="6" cy="12" r="3"></circle>
                      <circle cx="18" cy="19" r="3"></circle>
                      <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
                      <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
                    </svg>
                  </button>
                </div>
              </div>

              <div class="otc-summary-stats">
                <div class="otc-stat-box">
                  <div class="otc-stat-label">Team 1 Receives</div>
                  <div class="otc-stat-value" id="sideATotal">0.0</div>
                </div>
                <div class="otc-stat-box otc-stat-box-highlight">
                  <div class="otc-stat-label">Difference</div>
                  <div class="otc-stat-value" id="tradeDiff">0.0</div>
                </div>
                <div class="otc-stat-box">
                  <div class="otc-stat-label">Team 2 Receives</div>
                  <div class="otc-stat-value" id="sideBTotal">0.0</div>
                </div>
              </div>

              <div class="otc-balance-wrap">
                <div class="otc-balance-bar">
                  <div class="otc-balance-fair" title="Fair range"></div>
                  <div class="trade-bar-indicator" id="tradeBarIndicator"></div>
                </div>
                <div class="otc-balance-labels">
                  <span>Team 1 favored</span>
                  <span class="otc-balance-fair-label">Fair range</span>
                  <span>Team 2 favored</span>
                </div>
              </div>

              <div class="otc-verdict-shell">
                <div id="tradeVerdict" class="otc-verdict">
                  Add players to both sides to see the trade balance.
                </div>
                <div id="tradeScarcityNotes" style="display:none;"></div>
                <div id="errorBox" class="error" style="display:none;"></div>
              </div>
            </section>

            <section class="otc-ai-card" id="tradeAiPanel">
              <div class="otc-ai-head">
                <div>
                  <h2 class="otc-ai-title">BR Trade Analyst</h2>
                  <div class="otc-ai-sub">{ai_sub_text}</div>
                </div>
              </div>

              <div id="tradeAiBody" class="otc-ai-body">
                <div class="otc-ai-empty" id="aiLoadingState" style="display:none;">
                  <div class="otc-ai-empty-title">Analyzing Trade...</div>
                  <div class="otc-ai-empty-sub">
                    <div class="loading-spinner" style="margin: 10px auto; width: 30px; height: 30px; border: 3px solid #f3f4f6; border-radius: 50%; border-top-color: #3498db; animation: spin 1s linear infinite; border-right-color: transparent;"></div>
                  </div>
                </div>
                <div class="otc-ai-empty" id="aiEmptyState">
                  <div class="otc-ai-empty-title">{ai_empty_title}</div>
                  <div class="otc-ai-empty-sub">{ai_empty_sub}</div>
                </div>
                <div id="aiAnalysisResult" style="display:none;"></div>
              </div>
            </section>
          </div>

          <div id="similarTradesSection" style="display:none;margin-top:28px;">
            <div style="margin-bottom:14px;">
              <h3 class="stl-title">Recent Similar Trades</h3>
              <div class="stl-sub">Real dynasty trades where these players moved to opposite sides</div>
            </div>
            <div id="similarTradesList" class="stl-list"></div>
          </div>

          <style>
            .stl-title {{ font-size:15px;font-weight:700;color:var(--text-color);margin:0 0 3px; }}
            .stl-sub   {{ font-size:12px;color:var(--text-muted); }}
            .stl-list  {{ display:grid;grid-template-columns:repeat(2,1fr);gap:10px; }}
            @media(max-width:600px) {{ .stl-list {{ grid-template-columns:1fr; }} }}
            .stl-loading, .stl-empty {{ font-size:13px;color:var(--text-muted);padding:12px 0;grid-column:1/-1; }}

            .stl-card {{
              border:1px solid var(--border-color);
              border-radius:12px;
              overflow:hidden;
              background:var(--card-bg);
            }}
            .stl-card-head {{
              display:flex;
              justify-content:space-between;
              align-items:center;
              padding:7px 12px;
              border-bottom:1px solid var(--border-color);
              background:var(--bg-alt, rgba(0,0,0,.03));
            }}
            .stl-date {{ font-size:11px;color:var(--text-muted);font-weight:500; }}
            .stl-badges {{ display:flex;gap:4px;flex-wrap:wrap; }}
            .stl-badge {{
              font-size:10px;font-weight:700;
              padding:2px 7px;border-radius:6px;
              background:var(--row,#1e293b);
              color:var(--text);
              border:1px solid var(--border-color);
            }}
            .stl-badge-sf {{ background:#7c3aed22;color:#a78bfa;border-color:#7c3aed44; }}

            .stl-card-body {{
              display:grid;
              grid-template-columns:1fr 1px 1fr;
            }}
            .stl-col {{ padding:10px 12px;display:flex;flex-direction:column;gap:4px; }}
            .stl-col-divider {{ background:var(--border-color); }}

            .stl-asset {{
              font-size:14px;
              color:var(--text);
              font-weight:500;
              display:flex;align-items:center;gap:5px;flex-wrap:wrap;
            }}
            .stl-asset.stl-key  {{ font-weight:800;color:var(--accent-color,#3b82f6); }}
            .stl-asset.stl-pick {{ color:var(--text-muted);font-size:14px; }}
            .stl-asset.stl-muted {{ color:var(--text-muted); }}
            .stl-pos {{
              font-size:10px;font-weight:700;
              padding:1px 5px;border-radius:4px;
              background:var(--row,#1e293b);
              color:var(--text);flex-shrink:0;
            }}
            @media(max-width:480px) {{
              .stl-card-body {{ grid-template-columns:1fr; }}
              .stl-col-divider {{ height:1px;width:auto; }}
            }}
            /* ── Gaps | Strategy segmented toggle ─────────────────── */
            .otc-mode-toggle {{
              display:flex;gap:2px;padding:2px;border-radius:9px;
              background:var(--row);border:1px solid var(--border);
            }}
            .otc-mode-btn {{
              padding:5px 14px;font-size:12px;font-weight:700;
              border:none;border-radius:7px;background:transparent;
              color:var(--text-muted);cursor:pointer;transition:all .15s;
            }}
            .otc-mode-btn:hover:not(.is-active) {{ color:var(--text); }}
            .otc-mode-btn.is-active {{
              background:var(--text);color:var(--card);
            }}

            /* ── Archetype chips (2×2 grid) ───────────────────────── */
            #otcArchetypeChips {{
              display:none;grid-template-columns:1fr 1fr;gap:6px;margin-top:10px;
            }}
            .otc-arch-chip {{
              padding:6px 16px;font-size:12px;font-weight:600;
              border-radius:20px;border:1px solid var(--border);
              background:var(--card);color:var(--text-muted);
              cursor:pointer;transition:all .15s;
              white-space:nowrap;flex-shrink:0;
            }}
            .otc-arch-chip:hover {{ border-color:var(--accent);color:var(--text); }}
            .otc-arch-chip.is-active {{
              background:var(--accent);border-color:var(--accent);color:#fff;
            }}

            /* ── Strategy result cards (clean list, matches Gaps) ── */
            .otc-arch-card {{
              padding:10px 14px;border-bottom:1px solid var(--border);
              display:flex;flex-direction:column;gap:7px;
              transition:background .1s;
            }}
            .otc-arch-card:last-child {{ border-bottom:none; }}
            .otc-arch-card:hover {{ background:var(--row); }}
            .otc-arch-head {{ display:flex;align-items:center;gap:8px;min-width:0; }}
            .otc-arch-away-row {{ margin-bottom:2px; }}
            .otc-arch-away {{
              display:inline-block;font-size:8px;font-weight:700;text-transform:uppercase;
              letter-spacing:.06em;color:#f59e0b;
              background:#f59e0b1a;padding:2px 6px;border-radius:3px;
            }}
            .otc-arch-pos {{
              font-size:9px;font-weight:700;padding:2px 5px;border-radius:3px;
              flex-shrink:0;min-width:24px;text-align:center;
            }}
            .otc-arch-name {{
              font-size:13px;font-weight:700;color:var(--text);
              flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
            }}
            .otc-arch-why {{ font-size:11.5px;color:var(--text-muted);line-height:1.45; }}
            .otc-arch-deal {{ font-size:11px;color:var(--text-muted); }}
            .otc-arch-deal-label {{
              font-weight:700;text-transform:uppercase;letter-spacing:.04em;
              font-size:9px;color:var(--text-muted);margin-right:4px;
            }}
            .otc-arch-deal-assets {{ color:var(--text);font-weight:600; }}
            .otc-arch-deal-assets .v {{ color:var(--text-muted);font-weight:400; }}
            .otc-arch-foot {{
              display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:10.5px;
            }}
            .otc-arch-wp {{ font-weight:700;padding:2px 8px;border-radius:5px;font-size:10px; }}
            .otc-arch-wp.pos {{ background:#10b9811f;color:#10b981; }}
            .otc-arch-wp.neg {{ background:#ef44441f;color:#ef4444; }}
            .otc-arch-partner {{
              display:inline-flex;align-items:center;gap:4px;
              color:var(--text-muted);font-weight:600;
            }}
            .otc-arch-partner .dot {{ width:6px;height:6px;border-radius:50%;flex-shrink:0; }}
            .otc-arch-vmatch {{
              padding:2px 6px;border-radius:4px;font-size:10px;font-weight:700;
            }}
            .otc-arch-vmatch-great {{ background:#10b9811f;color:#10b981; }}
            .otc-arch-vmatch-fair  {{ background:var(--border);color:var(--text-muted); }}
            .otc-arch-vmatch-light {{ background:#f59e0b1a;color:#f59e0b; }}
            /* ── Suggestions sub-tab bar ── */
            .otc-sugg-subtab-bar {{
              display:flex;padding:10px 14px;gap:2px;
              background:var(--card);border-bottom:1px solid var(--border);
            }}
            .otc-sugg-subtab-toggle {{
              display:flex;gap:2px;padding:2px;border-radius:9px;
              background:var(--row);border:1px solid var(--border);
            }}
            .otc-sugg-subtab {{
              padding:6px 18px;font-size:12px;font-weight:700;
              border:none;border-radius:7px;background:transparent;
              color:var(--text-muted);cursor:pointer;transition:all .15s;
              white-space:nowrap;
            }}
            .otc-sugg-subtab:hover:not(.is-active) {{ color:var(--text); }}
            .otc-sugg-subtab.is-active {{
              background:var(--text);color:var(--card);
            }}
            /* ── Strategy panel ── */
            .otc-strategy-chips {{
              display:flex;overflow-x:auto;gap:6px;
              padding:10px 14px 8px;
              scrollbar-width:none;
            }}
            .otc-strategy-chips::-webkit-scrollbar {{ display:none; }}
            .otc-strategy-section-head {{
              display:flex;align-items:center;justify-content:space-between;
              padding:8px 14px 4px;
            }}
            .otc-strategy-section-hint {{
              font-size:10px;color:var(--text-muted);font-weight:500;
            }}
            .otc-strategy-clear-filter {{
              font-size:10px;font-weight:700;color:var(--accent);
              background:none;border:none;cursor:pointer;padding:2px 6px;
              border-radius:4px;
            }}
            .otc-strategy-clear-filter:hover {{ background:var(--row); }}
            /* ── Impact table rows ── */
            .otc-strategy-impact-row {{
              display:flex;align-items:center;gap:8px;
              padding:8px 14px;border-bottom:1px solid var(--border);
              cursor:pointer;transition:background .1s;
            }}
            .otc-strategy-impact-row:last-child {{ border-bottom:none; }}
            .otc-strategy-impact-row:hover,
            .otc-strategy-impact-row.is-active {{ background:var(--row); }}
            .otc-strategy-impact-name {{
              flex:1;font-size:13px;font-weight:700;color:var(--text);
              overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
            }}
            .otc-strategy-impact-stats {{
              display:flex;align-items:center;gap:6px;flex-shrink:0;
            }}
            .otc-strategy-impact-badge {{
              padding:2px 7px;border-radius:5px;font-size:10px;font-weight:700;
            }}
            /* ── Strategy cards container ── */
            #otcStrategyCards {{
              padding:10px 14px;display:flex;flex-direction:column;gap:6px;
            }}
            /* partner line inside otc-rt-footer */
            .otc-strategy-partner {{
              display:inline-flex;align-items:center;gap:4px;
              font-size:10px;font-weight:600;color:var(--text-muted);
            }}
            .otc-strategy-partner .dot {{
              width:6px;height:6px;border-radius:50%;flex-shrink:0;
            }}

            /* ── Mobile: 600px ── */
            @media (max-width:600px) {{
              .otc-arch-chip {{
                font-size:11px;
                padding:6px 6px;
              }}
              .otc-strategy-impact-badge {{
                font-size:9px;
                padding:2px 5px;
              }}
              .otc-sugg-subtab {{
                padding:6px 10px;
                font-size:11px;
              }}
            }}

            /* ── Mobile: 480px ── */
            @media (max-width:480px) {{
              .otc-strategy-chips {{
                padding:8px 10px 6px;
              }}
              .otc-arch-chip {{
                font-size:11px;
                padding:5px 12px;
              }}
              #otcStrategyCards {{
                padding:8px 10px;
              }}
              .otc-strategy-impact-row {{
                padding:7px 10px;
                gap:6px;
              }}
              .otc-strategy-impact-stats {{
                gap:4px;
              }}
              .otc-strategy-impact-badge {{
                font-size:9px;
                padding:2px 4px;
              }}
              .otc-strategy-impact-name {{
                font-size:12px;
              }}
              .otc-strategy-section-head {{
                padding:6px 10px 2px;
              }}
              .otc-sugg-subtab-bar {{
                padding:8px 10px;
              }}
              .otc-sugg-subtab {{
                padding:5px 8px;
                font-size:11px;
              }}
            }}
          </style>

          </div><!-- /#otcCalcTab -->

          <div id="otcSuggestionsTab" style="display:none;">
            <!-- Sub-tab bar -->
            <div class="otc-sugg-subtab-bar">
              <div class="otc-sugg-subtab-toggle">
                <button id="otcSubtabBuildAround" class="otc-sugg-subtab is-active">Build Around</button>
                <button id="otcSubtabStrategy" class="otc-sugg-subtab">Strategy</button>
              </div>
            </div>

            <!-- ── Build Around panel ───────────────────────────────────────────── -->
            <div id="otcBuildAroundPanel">
              <div class="otc-sugg-tab-layout">

                <!-- Build Around / Find Returns search -->
                <div class="otc-sugg-build-section">
                  <div class="otc-sugg-build-head">
                    <div class="otc-mode-toggle" id="otcSearchModeToggle">
                      <button id="otcSearchModeGet" class="otc-mode-btn is-active">Build around</button>
                      <button id="otcSearchModeSend" class="otc-mode-btn">Find returns</button>
                    </div>
                    <div class="otc-sugg-search-wrap">
                      <input id="suggPlayerInput" class="otc-sugg-player-input"
                        type="text" autocomplete="off"
                        placeholder="Search any player…" />
                      <div id="suggPlayerDropdown" class="otc-sugg-player-dropdown" style="display:none;"></div>
                    </div>
                  </div>
                  <div id="suggResultsMeta" class="otc-sugg-meta" style="display:none;"></div>
                  <div id="suggResultsList" class="otc-sugg-list"></div>
                </div>

                <!-- Gaps trade targets (always Gaps in Build Around) -->
                <div class="otc-sugg-targets-section">
                  <div class="otc-sugg-section-head">
                    <span class="otc-sugg-section-title">Trade Targets</span>
                  </div>
                  <div id="otcSuggTargetsBody">
                    <div class="otc-movers-empty">Select your team above to see targets.</div>
                  </div>
                  <!-- Excluded players bar -->
                  <div id="otcExcludedBar" style="display:none;border-top:1px solid var(--border);">
                    <div style="padding:8px 14px 4px;font-size:10px;font-weight:700;color:var(--text-muted);letter-spacing:.05em;text-transform:uppercase;">Excluded from suggestions</div>
                    <div id="otcExcludedChips"></div>
                  </div>
                </div>

              </div>
            </div><!-- /#otcBuildAroundPanel -->

            <!-- ── Strategy panel ───────────────────────────────────────────────── -->
            <div id="otcStrategyPanel" style="display:none;">

              <!-- Archetype chips (single scrollable pill row) -->
              <div id="otcStrategyChips" class="otc-strategy-chips">
                <button class="otc-arch-chip" data-arch="contending">Contending</button>
                <button class="otc-arch-chip" data-arch="rebuilding">Rebuilding</button>
                <button class="otc-arch-chip" data-arch="consolidate">Consolidate</button>
                <button class="otc-arch-chip" data-arch="distribute">Distribute</button>
              </div>

              <!-- Impact table (current PO badge is inline on the right) -->
              <div class="otc-strategy-section-head">
                <span class="otc-sugg-section-title">Impact <span id="otcStrategySpinner" style="display:none;font-size:10px;font-weight:500;color:var(--text-muted);"><i class="fa-solid fa-circle-notch" style="animation:spin .9s linear infinite;margin-right:2px;"></i>Simulating…</span></span>
                <div style="display:flex;align-items:center;gap:6px;flex-shrink:0;">
                  <span id="otcCurrentPOBadge" style="display:none;font-size:10px;font-weight:700;padding:2px 7px;border-radius:5px;background:var(--accent,#3b82f6)20;color:var(--accent,#3b82f6);"></span>
                  <span class="otc-strategy-section-hint" id="otcStrategyImpactHint">Win % if acquired</span>
                </div>
              </div>
              <div id="otcStrategyImpact">
                <div class="otc-movers-empty">Pick a strategy above.</div>
              </div>

              <!-- Trade cards -->
              <div class="otc-strategy-section-head" id="otcStrategyCardsHead" style="display:none;">
                <span class="otc-sugg-section-title">Suggested trades</span>
                <button class="otc-strategy-clear-filter" id="otcStrategyClearFilter" style="display:none;">Show all</button>
              </div>
              <div id="otcStrategyCards"></div>

            </div><!-- /#otcStrategyPanel -->

          </div><!-- /#otcSuggestionsTab -->

        </div><!-- /.otc-shell -->
      </main>

      <aside class="otc-side">
        <div class="otc-side-stack">

          <div class="otc-side-panel otc-movers-panel">
            <div class="otc-mini-head">
              <div class="otc-mini-head-row">
                <h3 class="otc-mini-title">Player Insights</h3>
                <div class="otc-mini-tabs">
                  <button class="otc-mini-tab is-active" data-tab="movers">Movers</button>
                  <button class="otc-mini-tab" data-tab="breakouts">Breakouts</button>
                  <button class="otc-mini-tab" data-tab="targets">Targets <i class="fa-solid fa-lock" style="font-size:9px;opacity:0.6;" id="targetsLockIcon"></i></button>
                </div>
              </div>
              <div class="otc-mini-sub" id="moversSub">Biggest 7-day changes in BR value</div>
              <div class="otc-day-filters" style="display: flex; gap: 6px; margin-top: 8px; padding: 0 12px;">
                <button class="otc-day-filter active" data-days="7" onclick="changeMoversDays(7)">7d</button>
                <button class="otc-day-filter" data-days="14" onclick="changeMoversDays(14)">14d</button>
                <button class="otc-day-filter" data-days="30" onclick="changeMoversDays(30)">30d</button>
              </div>
            </div>

            <div id="moversTabContent" class="otc-tab-content is-active">
              <div class="otc-mini-section">
                <div class="otc-mini-section-title">Top Risers</div>
                <div id="otcRisersList" class="otc-mini-list">
                  <div class="otc-movers-empty">Loading movers...</div>
                </div>
              </div>

              <div class="otc-mini-section">
                <div class="otc-mini-section-title">Top Fallers</div>
                <div id="otcFallersList" class="otc-mini-list">
                  <div class="otc-movers-empty">Loading movers...</div>
                </div>
              </div>
            </div>

            <div id="breakoutsTabContent" class="otc-tab-content">
              <div class="otc-mini-section">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                  <div class="otc-mini-section-title">Breakouts</div>
                  <a href="{breakouts_url}" class="{breakouts_link_class}" style="font-size: 12px; color: #3b82f6; text-decoration: none; font-weight: 500;">{breakouts_link_text}</a>
                </div>
                <div id="otcBreakoutsList" class="otc-mini-list">
                  <div class="otc-movers-empty">Loading breakouts...</div>
                </div>
              </div>
            </div>

            <div id="targetsTabContent" class="otc-tab-content">
              <div class="otc-mini-section">
                <div id="tradeTargetsBody" class="otc-mini-list" style="padding:8px 12px;display:flex;flex-direction:column;gap:8px;">
                  <div class="otc-movers-empty">Set your team to see targets.</div>
                </div>
              </div>
            </div>
          </div>

          <div class="otc-spacer" style="height: 16px; width: 100%;"></div>

          <div class="otc-side-panel">
            <div class="otc-side-head">
              <div class="otc-side-title-row">
                <div>
                  <h2 class="otc-side-title">Player Values</h2>
                  <div class="otc-side-sub">Filter by position</div>
                </div>
                <a href="{players_url}" class="otc-view-all-link">View All Players →</a>
              </div>
            </div>

            <div class="otc-filter-row" id="posFilterRow">
              <button class="otc-filter-chip pos-filter is-active" data-pos="ALL">All</button>
              <button class="otc-filter-chip pos-filter" data-pos="QB">QB</button>
              <button class="otc-filter-chip pos-filter" data-pos="RB">RB</button>
              <button class="otc-filter-chip pos-filter" data-pos="WR">WR</button>
              <button class="otc-filter-chip pos-filter" data-pos="TE">TE</button>
              <button class="otc-filter-chip pos-filter" data-pos="PICK">Picks</button>
              <button class="otc-filter-chip pos-filter" data-pos="ROOKIE">Rookies</button>
            </div>

            <div id="allPlayersList" class="otc-values-list">
              <!-- Filled by JS -->
            </div>
          </div>
        </div>
      </aside>

      <div id="tradeLoginModal" class="trade-login-modal" style="display:none;">
        <div class="trade-login-overlay"></div>
        <div class="trade-login-content">
          <button type="button" class="trade-login-close" id="closeLoginModal">&times;</button>
          <h2 class="trade-login-title">Sign In to Analyze Trade</h2>
          <p class="trade-login-subtitle">Connect your Sleeper league to get personalized trade analysis</p>

          <div class="trade-login-form">
            <div class="trade-login-row">
              <label for="tradeUsername">Sleeper Username</label>
              <input type="text" id="tradeUsername" placeholder="Enter your username">
            </div>

            <div class="trade-login-row">
              <button type="button" id="tradeLookupBtn" class="otc-btn otc-btn-primary">Find My Leagues</button>
            </div>

            <div class="trade-login-row" id="tradeLeagueSelectWrap" style="display:none;">
              <label for="tradeLeagueSelect">Choose League</label>
              <select id="tradeLeagueSelect">
                <option value="">Select a league</option>
              </select>
            </div>

            <div class="trade-login-row" id="tradeGoWrap" style="display:none;">
              <button type="button" id="tradeGoBtn" class="otc-btn otc-btn-primary">Open Trade Calculator</button>
            </div>

            <div id="tradeLookupError" class="trade-login-error" style="display:none;"></div>
          </div>
        </div>
      </div>
    </div>
    """
