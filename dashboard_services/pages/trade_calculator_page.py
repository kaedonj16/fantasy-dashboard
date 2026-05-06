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
) -> str:
    league_val = league_id or ""
    season_val = season if season is not None else ""
    viewer_roster_val = viewer_roster_id or ""
    is_guest = not league_id

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

    # Create dynamic breakouts URL based on login status
    if is_guest:
        breakouts_url = "#"
        breakouts_link_text = "Sign in to see more →"
        breakouts_link_class = "otc-view-all-link otc-guest-link"
    else:
        # For logged-in users, link to their league's breakouts page
        platform_val = "sleeper"  # Default platform, can be made dynamic if needed
        breakouts_url = f"/{platform_val}/{season_val}/{league_val}/breakouts"
        breakouts_link_text = "View All →"
        breakouts_link_class = "otc-view-all-link"

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
    league_type_block = f"""
              <div class="otc-ctrl-group" id="leagueTypeControl">
                <span class="otc-toggle-label">1QB</span>
                <label class="otc-pill-switch">
                  <input type="checkbox" id="leagueTypeToggle">
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

    return f"""
    <div class="otc-layout">
      <main class="otc-main">
        <input type="hidden" id="leagueIdInput" value="{league_val}">
        <input type="hidden" id="seasonInput" value="{season_val}">
        <input type="hidden" id="viewerRosterIdInput" value="{viewer_roster_val}">
        <input type="hidden" id="viewerSideInput" value="a">
        <input type="hidden" id="isGuestMode" value="{is_guest_str}">
        <input type="hidden" id="otcHasPremium" value="{has_premium_str}">

        <div class="otc-shell">
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
                  <div class="otc-balance-fair">FAIR RANGE</div>
                  <div class="trade-bar-indicator" id="tradeBarIndicator"></div>
                </div>
                <div class="otc-balance-labels">
                  <span>Team 1 favored</span>
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
          </style>

        </div>
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
                <a href="/players" class="otc-view-all-link">View All Players →</a>
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
