from typing import Optional


def build_trade_calculator_body(league_id: Optional[str], season: Optional[int]) -> str:
    league_val = league_id or ""
    season_val = season if season is not None else ""
    is_guest = not league_id

    # ----------------------------------------------------------------
    # Pre-compute all conditional HTML blocks outside the f-string
    # to avoid nested triple-quote SyntaxErrors
    # ----------------------------------------------------------------

    # Show owner tags for all users (updated by viewer side selection)
    side_a_owner_tag = '<div class="otc-team-owner-tag" id="sideAOwnerTag">Your side</div>'
    side_b_owner_tag = '<div class="otc-team-owner-tag otc-team-owner-tag-muted" id="sideBOwnerTag">Other side</div>'

    analyze_btn_disabled = 'disabled' if is_guest else ''
    analyze_btn_label = 'Log In to Analyze' if is_guest else 'Analyze Trade'

    ai_sub_text = 'AI-powered trade analysis for dynasty leagues' if is_guest else 'Personalized to your team direction and roster lens'

    team_select_block = '' if is_guest else """
                <div class="otc-summary-team-select">
                  <select id="teamSelect" class="otc-team-select-dropdown" required>
                    <option value="">Select your team...</option>
                  </select>
                </div>
    """

    ai_empty_title = 'Log In for AI Analysis' if is_guest else 'Waiting on a deal'
    ai_empty_sub = (
        'Connect your league to get personalized trade analysis powered by AI.'
        if is_guest else
        'Once both sides have assets, this panel can explain whether the trade fits your team build.'
    )

    is_guest_str = 'true' if is_guest else 'false'

    return f"""
    <div class="otc-layout">
      <main class="otc-main">
        <input type="hidden" id="leagueIdInput" value="{league_val}">
        <input type="hidden" id="seasonInput" value="{season_val}">
        <input type="hidden" id="viewerSideInput" value="a">
        <input type="hidden" id="isGuestMode" value="{is_guest_str}">

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
                      <p>Player values derive from a hybrid approach combining production metrics, age-adjusted projections, and market consensus data.</p>
                      <p>The model incorporates positional scarcity adjustments, role stability indicators, and capital investment signals to generate normalized valuations on a standardized scale.</p>
                      <p>Values represent long-term dynasty asset worth rather than weekly fantasy output.</p>
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
                  <span>Team 1 is mine</span>
                </label>
                <label class="otc-viewer-toggle">
                  <input type="radio" name="viewerSide" value="b">
                  <span>Team 2 is mine</span>
                </label>
              </div>
              <div class="otc-viewer-toggles" style="margin-left: 16px;">
                <label class="otc-viewer-toggle">
                  <input type="radio" name="leagueType" value="1qb" checked>
                  <span>1QB</span>
                </label>
                <label class="otc-viewer-toggle">
                  <input type="radio" name="leagueType" value="sf">
                  <span>Superflex</span>
                </label>
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
                </div>
              </div>

              <div class="otc-summary-stats">
                <div class="otc-stat-box">
                  <div class="otc-stat-label">Team 1 Total</div>
                  <div class="otc-stat-value" id="sideATotal">0.0</div>
                </div>
                <div class="otc-stat-box otc-stat-box-highlight">
                  <div class="otc-stat-label">Difference</div>
                  <div class="otc-stat-value" id="tradeDiff">0.0</div>
                </div>
                <div class="otc-stat-box">
                  <div class="otc-stat-label">Team 2 Total</div>
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
        </div>
      </main>

      <aside class="otc-side">
        <div class="otc-side-stack">
          <div class="otc-side-panel otc-movers-panel">
            <div class="otc-mini-head">
              <h3 class="otc-mini-title">Top Movers</h3>
              <div class="otc-mini-sub" id="moversSub">Biggest 7-day changes in BR value</div>
            </div>

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

          <div class="otc-spacer" style="height: 16px; width: 100%;"></div>

          <div class="otc-side-panel">
            <div class="otc-side-head">
              <h2 class="otc-side-title">Player Values</h2>
              <div class="otc-side-sub">Filter by position</div>
            </div>

            <div class="otc-filter-row" id="posFilterRow">
              <button class="otc-filter-chip pos-filter is-active" data-pos="ALL">All</button>
              <button class="otc-filter-chip pos-filter" data-pos="QB">QB</button>
              <button class="otc-filter-chip pos-filter" data-pos="RB">RB</button>
              <button class="otc-filter-chip pos-filter" data-pos="WR">WR</button>
              <button class="otc-filter-chip pos-filter" data-pos="TE">TE</button>
              <button class="otc-filter-chip pos-filter" data-pos="PICK">Picks</button>
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
          <h2 class="trade-login-title">Log In to Analyze Trade</h2>
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
