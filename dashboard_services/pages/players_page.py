"""Player Rankings (/players) page shell.

The static HTML/CSS/JS for the rankings page — filters, skeleton, table header,
and styles. Extracted verbatim from app.py's page_players to shrink the monolith;
the route still assembles the dynamic parts (server-rendered rows, SEO intro,
TE-premium + league-context scripts) around this shell.
"""


def build_players_shell() -> str:
    body_html = """
    <div class="card central">
      <div class="card-header" style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
        <div>
          <h2>Player Rankings</h2>
          <div style="font-size: 14px; color: var(--text-muted); margin-top: 4px;">
            All players ranked by dynasty value.
          </div>
        </div>
        <button onclick="prExportCSV()" title="Download current list as CSV"
          style="margin-top:4px;padding:7px 14px;border-radius:8px;border:1px solid var(--border);
                 background:var(--surface);color:var(--text);font-size:12px;font-weight:600;
                 cursor:pointer;display:flex;align-items:center;gap:6px;white-space:nowrap;">
          <img src="/static/images/download-solid.png" style="width:13px;height:13px;vertical-align:middle;opacity:0.8;" alt=""> Export CSV
        </button>
      </div>
      <div class="card-body" style="padding-top:0;">

        <!-- Controls -->
        <div class="filter-controls-container">
          <!-- Row 1: Primary filters -->
          <div class="filter-row filter-row-primary">
            <!-- Search -->
            <div class="filter-search">
              <input id="prSearch" type="text" placeholder="Search players…" autocomplete="off"
                style="width:100%;padding:8px 32px 8px 34px;border-radius:8px;
                       border:1px solid var(--border);background:var(--card-bg);
                       color:var(--text);font-size:13px;outline:none;box-sizing:border-box;">
              <span style="position:absolute;left:10px;top:50%;transform:translateY(-50%);
                           color:var(--text-muted);font-size:13px;pointer-events:none;"><i class="fa-solid fa-magnifying-glass"></i></span>
              <button id="prSearchClear" onclick="prClearSearch()"
                style="display:none;position:absolute;right:8px;top:50%;transform:translateY(-50%);
                       background:none;border:none;cursor:pointer;color:var(--text-muted);
                       font-size:16px;line-height:1;padding:2px;">&#x2715;</button>
            </div>

            <!-- Position filters -->
            <div class="filter-positions br-chip-pop">
              <button class="pos-pill active" data-pos="ALL" onclick="prTogglePos('ALL')">All</button>
              <button class="pos-pill" data-pos="QB" onclick="prTogglePos('QB')">QB</button>
              <button class="pos-pill" data-pos="RB" onclick="prTogglePos('RB')">RB</button>
              <button class="pos-pill" data-pos="WR" onclick="prTogglePos('WR')">WR</button>
              <button class="pos-pill" data-pos="TE" onclick="prTogglePos('TE')">TE</button>
              <button class="pos-pill" data-pos="PICK" onclick="prTogglePos('PICK')">Picks</button>
            </div>

            <!-- Settings button -->
            <div style="position:relative;">
              <button id="prSettingsBtn" class="filter-settings-btn" onclick="prToggleSettings()">
                Settings
              </button>

              <!-- Settings panel (hidden by default) -->
              <div id="prSettingsPanel" class="filter-settings-panel" style="display:none;">
                <div class="settings-section">
                  <span class="settings-section-label">Scoring Type</span>
                  <div class="settings-toggle-group">
                    <button class="settings-toggle active" data-value="dynasty" onclick="prSetScoringType('dynasty')">Dynasty</button>
                    <button class="settings-toggle" data-value="redraft" onclick="prSetScoringType('redraft')">Redraft</button>
                  </div>
                </div>
                <div class="settings-section">
                  <span class="settings-section-label">League Format</span>
                  <div class="settings-toggle-group">
                    <button class="settings-toggle active" data-value="1qb" onclick="prSetLeagueType('1qb')">1QB</button>
                    <button class="settings-toggle" data-value="sf" onclick="prSetLeagueType('sf')">SF</button>
                  </div>
                </div>
                <div class="settings-section" id="prSizeSection">
                  <span class="settings-section-label">League Size</span>
                  <div class="settings-toggle-group">
                    <button class="settings-toggle" data-value="8" onclick="prSetSize(8)">8</button>
                    <button class="settings-toggle active" data-value="10" onclick="prSetSize(10)">10</button>
                    <button class="settings-toggle" data-value="12" onclick="prSetSize(12)">12</button>
                    <button class="settings-toggle" data-value="14" onclick="prSetSize(14)">14</button>
                  </div>
                </div>
                <div class="settings-section" id="prTepSection">
                  <span class="settings-section-label">TE Premium</span>
                  <div class="settings-toggle-group">
                    <button class="settings-toggle active" data-value="0" onclick="prSetTePremium(0)">Off</button>
                    <button class="settings-toggle" data-value="0.5" onclick="prSetTePremium(0.5)">+0.5</button>
                    <button class="settings-toggle" data-value="1" onclick="prSetTePremium(1)">+1.0</button>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Row 2: Secondary filters -->
          <div class="filter-row filter-row-secondary">
            <div id="prActiveSettings" class="active-settings-indicator">
              <span class="active-setting-tag">10-Team</span>
              <span class="active-setting-tag">1QB</span>
              <span class="active-setting-tag">Dynasty</span>
              <span class="active-setting-tag" id="prTepTag" style="display:none;">TE+</span>
            </div>
            <!-- ADP source + Sort: a paired control group (side by side on
                 mobile). The ADP source only appears when sorting by ADP, and
                 sits to the LEFT so the Sort control stays on the far right. -->
            <div class="filter-sort-group">
              <div class="filter-sort" id="prAdpSrcWrap" style="display:none;">
                <label class="filter-label" for="prAdpSource">ADP source</label>
                <select id="prAdpSource" onchange="prReloadAdpSource()"></select>
              </div>
              <div class="filter-sort">
                <label class="filter-label" for="prSort">Sort by</label>
                <select id="prSort" onchange="prPage=1;prFlipRender()">
                  <option value="value">Value</option>
                  <option value="adp">ADP</option>
                  <option value="age">Age</option>
                  <option value="pos_rank">Pos Rank</option>
                  <option value="ppg">PPG</option>
                  <option value="total_pts">Total Points</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        <!-- Loading: a skeleton list shaped like the ranked rows, so the table
             swaps in without a spinner and without the layout jumping. This
             block lives in a plain (non-f) string, so the rows are literal. -->
        <div id="prLoading" class="sk-list" aria-hidden="true" style="margin-top:8px;">
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-80"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-60"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-80"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-60"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-80"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-60"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-80"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-60"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
        </div>

        <!-- Player count -->
        <div id="prCount" style="font-size:12px;color:var(--text-muted);margin-bottom:8px;display:none;"></div>

        <!-- Table header -->
        <div id="prTableHeader" style="display:none;
             grid-template-columns:54px 42px 1fr 52px 46px 46px 60px;
             gap:0;padding:6px 12px;border-radius:6px;
             background:var(--accent-soft);font-size:11px;
             font-weight:700;color:var(--accent);letter-spacing:0.04em;
             text-transform:uppercase;" class="pr-grid-row">
          <span>#</span>
          <span style="text-align:center;"></span>
          <span>Player</span>
          <span style="text-align:center;">Pos</span>
          <span id="prAgeHeader" style="text-align:center;">Age</span>
          <span style="text-align:right;">Team</span>
          <span id="prSortHeader" style="text-align:right;">Value</span>
        </div>

        <!-- Player rows -->
        <div id="prList"></div>

        <!-- Empty state -->
        <div id="prEmpty" style="display:none;text-align:center;padding:40px;color:var(--text-muted);">
          <div style="font-size:24px;margin-bottom:8px;opacity:0.4;"><i class="fa-solid fa-magnifying-glass"></i></div>
          No players match your filters
        </div>

      </div>
    </div>

    <style>
      .pr-grid-row {
        display: grid;
        grid-template-columns: 54px 42px 1fr 52px 46px 46px 60px;
        align-items: center;
        gap: 0;
      }
      .pr-player-row {
        padding: 9px 12px;
        cursor: pointer;
        transition: background 0.12s ease;
      }
      .pr-player-row:hover { background: var(--accent-soft); }
      .pr-player-row + .pr-player-row { border-top: 1px solid var(--border); }
      .pr-rank {
        font-size: 12px;
        font-weight: 700;
        color: var(--text-muted);
        display: flex;
        align-items: center;
        gap: 3px;
        justify-content: flex-start;
      }
      .pr-rank-arrow {
        font-size: 16px;
        font-weight: 700;
        line-height: 1;
      }
      .pr-rank-arrow.up   { color: #22c55e; }
      .pr-rank-arrow.down { color: #ef4444; }
      /* Overall rank movement shown beside the rank number. */
      .pr-rank-delta {
        font-size: 11px;
        font-weight: 800;
        line-height: 1;
        white-space: nowrap;
      }
      .pr-rank-delta.up   { color: var(--win); }
      .pr-rank-delta.down { color: var(--loss); }
      .pr-arrows {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 10px;
      }
      .pr-name {
        font-size: 13px;
        font-weight: 600;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 5px;
        flex-wrap: wrap;
        min-width: 0;
      }
      .pr-pos-cell {
        text-align: center;
        font-size: 11px;
        font-weight: 700;
        color: var(--text-muted);
      }
      .pr-age {
        text-align: center;
        font-size: 12px;
        color: var(--text-muted);
      }
      .pr-team {
        text-align: right;
        font-size: 11px;
        color: var(--text-muted);
      }
      .pr-value {
        text-align: right;
        font-size: 13px;
        font-weight: 700;
        color: var(--accent);
      }
      /* Per-source ADP columns (sort-by-ADP view) */
      .pr-adp-head {
        text-align: center;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        user-select: none;
        transition: color 0.12s ease;
      }
      .pr-adp-head:hover { color: var(--text); }
      .pr-adp-head-active { color: var(--accent); }
      .pr-adp-sort-caret { font-size: 9px; }
      .pr-adp-cell {
        text-align: center;
        font-size: 13px;
        font-weight: 600;
        color: var(--text-muted);
        font-variant-numeric: tabular-nums;
      }
      .pr-adp-cell-active { color: var(--accent); font-weight: 700; }
      /* Comparison arrow vs the sorted ADP source: green ▲ when this source ranks
         the player higher (earlier pick), red ▼ when lower (later pick). */
      .pr-adp-arrow { font-size: 8px; margin-left: 3px; vertical-align: 1px; }
      .pr-adp-arrow.up   { color: var(--win); }
      .pr-adp-arrow.down { color: var(--loss); }
      /* Compact Pos/Age/Team columns kept in the ADP view on desktop */
      .pr-adp-meta-h {
        text-align: center;
        font-size: 10px;
        opacity: 0.85;
      }
      .pr-adp-meta {
        text-align: center;
        font-size: 11px;
        color: var(--text-muted);
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      /* Sticky table header */
      #prTableHeader {
        position: sticky;
        top: 60px;
        z-index: 5;
        border-radius: 6px;
        margin-bottom: 2px;
      }

      /* Filter Controls */
      .filter-controls-container {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 16px 0 14px;
        border-bottom: 1px solid var(--border);
        margin-bottom: 12px;
      }
      .filter-row {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        justify-content: space-between;
      }
      .filter-row-primary {
        gap: 12px;
      }
      .filter-row-secondary {
        padding-top: 4px;
        flex-wrap: nowrap;
        align-items: center;
      }
      .filter-row-secondary .active-settings-indicator {
        flex: 1;
        min-width: 0;
        overflow-x: auto;
        scrollbar-width: none;
        flex-wrap: nowrap;
      }
      .filter-row-secondary .active-settings-indicator::-webkit-scrollbar { display: none; }
      .filter-row-secondary .filter-sort-group { flex-shrink: 0; }
      .filter-search {
        position: relative;
        flex: 1;
        min-width: 200px;
      }
      .filter-positions {
        display: flex;
        gap: 3px;
        flex-wrap: wrap;
      }
      .pos-pill {
        padding: 6px 12px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: var(--card-bg);
        color: var(--text-muted);
        font-size: 11px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.12s;
        white-space: nowrap;
      }
      .pos-pill.active {
        background: var(--accent);
        color: #fff;
        border-color: var(--accent);
      }
      .filter-settings-btn {
        padding: 7px 14px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--card-bg);
        color: var(--text);
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
        transition: all 0.12s;
      }
      .filter-settings-btn:hover {
        background: var(--accent-soft);
        border-color: var(--accent);
        color: var(--accent);
      }
      .filter-settings-panel {
        position: absolute;
        top: 100%;
        right: 0;
        margin-top: 8px;
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        padding: 16px;
        min-width: 260px;
        z-index: 1000;
      }
      .settings-section {
        margin-bottom: 16px;
      }
      .settings-section:last-of-type {
        margin-bottom: 0;
      }
      .settings-section-label {
        display: block;
        font-size: 11px;
        font-weight: 700;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 8px;
      }
      .settings-toggle-group {
        display: flex;
        gap: 6px;
      }
      .settings-toggle {
        flex: 1;
        padding: 8px 12px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--card-bg);
        color: var(--text-muted);
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.12s;
      }
      .settings-toggle.active {
        background: var(--accent);
        color: #fff;
        border-color: var(--accent);
      }
      .active-settings-indicator {
        display: flex;
        gap: 6px;
        align-items: center;
        flex-wrap: wrap;
      }
      .active-setting-tag {
        padding: 4px 10px;
        border-radius: 999px;
        background: var(--accent-soft);
        color: var(--accent);
        font-size: 11px;
        font-weight: 600;
      }
      .filter-sort-group {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-shrink: 0;
      }
      .filter-sort {
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .filter-sort select {
        padding: 7px 30px 7px 11px;
        border-radius: 9px;
        border: 1px solid var(--border);
        background: var(--card-bg);
        color: var(--text);
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        outline: none;
        min-height: 36px;
        min-width: 128px;
        transition: border-color 0.12s, box-shadow 0.12s;
        /* Custom chevron so both selects match across platforms */
        -webkit-appearance: none;
        appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: right 11px center;
      }
      .filter-sort select:hover {
        border-color: var(--accent);
      }
      .filter-sort select:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px var(--accent-soft);
      }
      .filter-label {
        font-size: 11px;
        font-weight: 600;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }

      @media (max-width: 600px) {
        .filter-row-primary {
          display: grid;
          grid-template-columns: 1fr auto;
          grid-template-rows: auto auto;
          gap: 8px;
        }
        .filter-search {
          grid-column: 1 / -1;
          min-width: 0;
        }
        .filter-positions {
          flex-wrap: nowrap;
          overflow-x: auto;
          scrollbar-width: none;
          min-width: 0;
          align-items: center;
        }
        .filter-positions::-webkit-scrollbar { display: none; }
        .filter-row-secondary { gap: 8px; }
      }

      .pr-tier-divider {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 5px 0 3px;
        pointer-events: none;
      }
      .pr-tier-divider-line {
        flex: 1;
        height: 1px;
        opacity: 0.5;
      }
      .pr-tier-divider-label {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 0.06em;
        opacity: 0.75;
        white-space: nowrap;
      }

      /* Mobile responsive */
      @media (max-width: 768px) {
        .filter-row-primary {
          flex-direction: column;
          align-items: stretch;
        }
        .filter-search {
          max-width: 100%;
        }
        .filter-positions {
          justify-content: flex-start;
          gap: 5px;
        }
        .pos-pill {
          padding: 6px 10px;
          font-size: 11px;
        }
        .filter-label {
          white-space: nowrap;
        }
        .active-settings-indicator {
          justify-content: center;
          order: -1;
          width: 100%;
        }
        .filter-row-secondary {
          flex-wrap: wrap;
          gap: 8px;
        }
        /* Sort + ADP source sit side by side on their own row, each a tidy
           field with the label stacked above a full-width select. Grid columns
           of minmax(0,1fr) keep them equal and never overflow; a hidden ADP
           source leaves one column, so Sort by fills the row. */
        .filter-sort-group {
          display: grid;
          grid-auto-flow: column;
          grid-auto-columns: minmax(0, 1fr);
          gap: 10px;
          width: 100%;
        }
        .filter-sort {
          min-width: 0;
          flex-direction: column;
          align-items: stretch;
          gap: 5px;
        }
        .filter-sort .filter-label {
          font-size: 10px;
        }
        .filter-sort select {
          width: 100%;
          min-width: 0;
        }
        /* Table: hide Age on tablets - rank | arrow | name | pos | team | sort.
           The ADP-source view (.pr-adp-mode) manages its own columns inline, so
           exclude it from these fixed overrides. */
        .pr-grid-row:not(.pr-adp-mode) { grid-template-columns: 50px 42px 1fr 44px 42px 56px !important; }
        .pr-age,  #prAgeHeader  { display: none !important; }
      }
      @media (max-width: 480px) {
        /* Phone: rank | arrow | name | sort - hide pos and team */
        .pr-grid-row:not(.pr-adp-mode) { grid-template-columns: 50px 42px 1fr 56px !important; }
        .pr-pos-cell, #prTableHeader:not(.pr-adp-mode) span:nth-child(4) { display: none !important; }
        .pr-team,     #prTableHeader:not(.pr-adp-mode) span:nth-child(6) { display: none !important; }
        /* ADP-source view on phones: the JS shrinks each source column (and the
           rank column) so all sources fit; shrink the header labels to match so
           "Sleeper" / "BR Fantasy" / "Consensus" read in full instead of
           truncating to "SLEEP…", and keep the numbers tidy. */
        #prTableHeader.pr-adp-mode .pr-adp-head {
          font-size: 9px;
          letter-spacing: 0;
          padding: 0 1px;
        }
        #prTableHeader.pr-adp-mode .pr-adp-sort-caret { font-size: 7px; }
        .pr-adp-mode .pr-adp-cell { font-size: 12px; }
        /* Name shares a tighter row now: clip to one line rather than wrapping
           to a second line that misaligns the numbers. */
        .pr-adp-mode .pr-name {
          flex-wrap: nowrap;
          min-width: 0;
          overflow: hidden;
          white-space: nowrap;
        }
      }
    </style>

    <!-- Player Rankings JS lives in static/rankings.js (injected below, deferred) -->
    """
    return body_html
