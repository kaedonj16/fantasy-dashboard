from typing import Optional


def build_trade_calculator_body(league_id: Optional[str], season: Optional[int]) -> str:
    league_val = league_id or ""
    season_val = season if season is not None else ""

    return f"""
    <div class="otc-layout">
      <main class="otc-main">

        <input type="hidden" id="leagueIdInput" value="{league_val}">
        <input type="hidden" id="seasonInput" value="{season_val}">

        <div class="otc-shell">
          <div class="otc-page-head">
            <div class="otc-page-title-wrap">
              <div class="otc-page-kicker">Offseason tool</div>
              <h1 class="otc-page-title">Trade Calculator</h1>
              <p class="otc-page-copy">
                Compare both sides of a deal using BR values, balance, and roster-building context.
              </p>
            </div>
            <div class="otc-page-badge">Dynasty Trade Tool</div>
          </div>

          <div class="otc-builder-grid">
            <section class="otc-team-card">
              <div class="otc-team-head">
                <h2 class="otc-team-title">Team 1 gets...</h2>
                <div class="otc-team-pill">Side A</div>
              </div>

              <div class="otc-slot">
                <div class="otc-slot-empty">
                  <div class="otc-slot-empty-title">Add player</div>
                  <div class="otc-slot-empty-sub">
                    Search players to build the first side of the trade.
                  </div>
                </div>

                <div class="otc-search-wrap">
                  <div class="search-wrapper">
                    <input id="sideASearch"
                           class="otc-search-input"
                           type="text"
                           autocomplete="off"
                           placeholder="Start typing a name..." />
                    <div id="sideADropdown" class="dropdown otc-search-dropdown" style="display:none;"></div>
                  </div>
                </div>

                <div class="chips" id="sideAChips"></div>
              </div>
            </section>

            <section class="otc-team-card">
              <div class="otc-team-head">
                <h2 class="otc-team-title">Team 2 gets...</h2>
                <div class="otc-team-pill">Side B</div>
              </div>

              <div class="otc-slot">
                <div class="otc-slot-empty">
                  <div class="otc-slot-empty-title">Add player</div>
                  <div class="otc-slot-empty-sub">
                    Search players to build the second side of the trade.
                  </div>
                </div>

                <div class="otc-search-wrap">
                  <div class="search-wrapper">
                    <input id="sideBSearch"
                           class="otc-search-input"
                           type="text"
                           autocomplete="off"
                           placeholder="Start typing a name..." />
                    <div id="sideBDropdown" class="dropdown otc-search-dropdown" style="display:none;"></div>
                  </div>
                </div>

                <div class="chips" id="sideBChips"></div>
              </div>
            </section>
          </div>

          <section class="otc-summary-card">
            <div class="otc-summary-head">
              <h2 class="otc-summary-title">Trade Summary</h2>
              <div class="otc-summary-sub">Live balance as you add assets</div>
            </div>

            <div class="otc-summary-stats">
              <div class="otc-stat-box">
                <div class="otc-stat-label">Team 1 Total</div>
                <div class="otc-stat-value" id="sideATotal">0.0</div>
              </div>
              <div class="otc-stat-box">
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

            <div id="tradeVerdict" class="otc-verdict">
              Add players to both sides to see the trade balance.
            </div>
            <div id="errorBox" class="error" style="display:none;"></div>
          </section>
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
            </div>

            <div id="allPlayersList" class="otc-values-list">
              <!-- Filled by JS -->
            </div>
          </div>
        </div>
      </aside>
    </div>
    """
