"""Player Rankings (/players) page shell.

The static HTML/CSS/JS for the rankings page -- filters, skeleton, table header,
and styles. Extracted verbatim from app.py's page_players to shrink the monolith;
the route still assembles the dynamic parts (server-rendered rows, SEO intro,
TE-premium + league-context scripts) around this shell.
"""


def build_players_shell() -> str:
    body_html = """
    <div class="card central">
      <div class="card-body" style="padding-top:0;">

        <!-- Controls: merged header rows (approved rework). Row 1 pairs the
             title with the Dynasty / Redraft + 1QB / SF toggles and search;
             Row 2 pairs the sliding-pill sort with the position pills and
             Legend / Settings pinned right. On phones the stack collapses to
             title row, search, full-width sort pill, position pills, count. -->
        <div class="filter-controls-container">
          <!-- Row 1: title, format toggles, search -->
          <div class="filter-row filter-row-primary pr-head-row1">
            <div class="pr-titleblock">
              <h2 class="page-title">Player Rankings</h2>
              <p class="page-head-sub"><span id="prSubtitle">All players ranked by dynasty value.</span></p>
            </div>
            <div class="pr-seg" role="group" aria-label="Scoring type" id="prScoringSeg">
              <button type="button" class="pr-segbtn active" data-value="dynasty" onclick="prSetScoringType('dynasty')">Dynasty</button>
              <button type="button" class="pr-segbtn" data-value="redraft" onclick="prSetScoringType('redraft')">Redraft</button>
            </div>
            <div class="pr-seg pr-seg-sm" role="group" aria-label="League format" id="prFormatSeg">
              <button type="button" class="pr-segbtn active" data-value="1qb" onclick="prSetLeagueType('1qb')">1QB</button>
              <button type="button" class="pr-segbtn" data-value="sf" onclick="prSetLeagueType('sf')">SF</button>
            </div>
            <!-- Search -->
            <div class="filter-search">
              <input id="prSearch" type="text" placeholder="Search players…" autocomplete="off"
                aria-label="Search players"
                style="width:100%;padding:8px 32px 8px 34px;border-radius:8px;
                       border:1px solid var(--border);background:var(--card-bg);
                       color:var(--text);font-size:13px;outline:none;box-sizing:border-box;">
              <span style="position:absolute;left:10px;top:50%;transform:translateY(-50%);
                           color:var(--text-muted);font-size:13px;pointer-events:none;"><i class="fa-solid fa-magnifying-glass"></i></span>
              <button id="prSearchClear" onclick="prClearSearch()" type="button"
                aria-label="Clear search"
                style="display:none;position:absolute;right:8px;top:50%;transform:translateY(-50%);
                       background:none;border:none;cursor:pointer;color:var(--text-muted);
                       font-size:15px;line-height:1;padding:2px;">&#x2715;</button>
            </div>

          </div>

          <!-- Row 2: sliding-pill sort, position filters, Legend / Settings -->
          <div class="filter-row filter-row-secondary pr-head-row2">
            <span class="pr-grouplab" id="prSortLabel">Sort</span>
            <div class="teams-mode-toggle pr-sort-toggle" role="group" aria-label="Sort by" id="prSortToggle">
              <span class="teams-mode-thumb" aria-hidden="true"></span>
              <button type="button" class="on" data-sortkey="value" aria-pressed="true" onclick="prSetSortKey('value')">Value</button>
              <button type="button" data-sortkey="ppg" aria-pressed="false" onclick="prSetSortKey('ppg')">PPG</button>
              <button type="button" data-sortkey="total_pts" aria-pressed="false" onclick="prSetSortKey('total_pts')"><span class="pr-sort-full">Total Points</span><span class="pr-sort-short">Total</span></button>
              <button type="button" data-sortkey="age" aria-pressed="false" onclick="prSetSortKey('age')">Age</button>
              <button type="button" data-sortkey="pos_rank" aria-pressed="false" onclick="prSetSortKey('pos_rank')">Pos Rank</button>
            </div>
            <span class="pr-vdiv" aria-hidden="true"></span>
            <!-- Position filters -->
            <div class="otc-day-filters filter-positions br-chip-pop" role="group" aria-label="Filter by position">
              <button type="button" class="otc-day-filter pos-pill active" data-pos="ALL" aria-pressed="true" onclick="prTogglePos('ALL')">All</button>
              <button type="button" class="otc-day-filter pos-pill" data-pos="QB" aria-pressed="false" onclick="prTogglePos('QB')">QB</button>
              <button type="button" class="otc-day-filter pos-pill" data-pos="RB" aria-pressed="false" onclick="prTogglePos('RB')">RB</button>
              <button type="button" class="otc-day-filter pos-pill" data-pos="WR" aria-pressed="false" onclick="prTogglePos('WR')">WR</button>
              <button type="button" class="otc-day-filter pos-pill" data-pos="TE" aria-pressed="false" onclick="prTogglePos('TE')">TE</button>
              <button type="button" class="otc-day-filter pos-pill" data-pos="PICK" aria-pressed="false" onclick="prTogglePos('PICK')">Picks</button>
            </div>
            <!-- ADP source: only appears when sorting by ADP (per-source
                 columns replace it whenever the server sends them). -->
            <div class="filter-sort" id="prAdpSrcWrap" style="display:none;">
              <label class="filter-label" for="prAdpSource">ADP source</label>
              <select id="prAdpSource" onchange="prReloadAdpSource()"></select>
            </div>
            <!-- Hidden sort select: kept as the sync target for column-header
                 clicks and the ADP / Name sorts the pill does not carry. -->
            <div class="filter-sort" style="display:none;">
              <label class="filter-label" for="prSort">Sort by</label>
              <select id="prSort" onchange="prSetSortKey(this.value)">
                <option value="value">Value</option>
                <option value="adp">ADP</option>
                <option value="age">Age</option>
                <option value="pos_rank">Pos Rank</option>
                <option value="ppg">PPG</option>
                <option value="total_pts">Total Points</option>
                <option value="name">Name (A-Z)</option>
              </select>
            </div>
            <span class="pr-pinright">
              <span style="position:relative;">
                <button id="prLegendBtn" class="filter-settings-btn" onclick="prToggleLegend()" type="button">Legend</button>
                <div id="prLegendPanel" class="filter-settings-panel" style="display:none;">
                  <div class="settings-section">
                    <span class="settings-section-label">Value tiers</span>
                    <div class="pr-legend-tiers">
                      <span><i style="background:#10b981"></i>T1 Elite</span>
                      <span><i style="background:#22d3ee"></i>T2 Star</span>
                      <span><i style="background:#3b82f6"></i>T3 High-End Starter</span>
                      <span><i style="background:#8b5cf6"></i>T4 Starter</span>
                      <span><i style="background:#a855f7"></i>T5 Flex</span>
                      <span><i style="background:#f59e0b"></i>T6 Bench</span>
                      <span><i style="background:#f97316"></i>T7 Deep Bench</span>
                      <span><i style="background:#94a3b8"></i>T8 Handcuff</span>
                      <span><i style="background:#64748b"></i>T9 Fringe</span>
                    </div>
                  </div>
                  <div class="settings-section">
                    <span class="settings-section-label">Row marks</span>
                    <p class="pr-legend-note"><span class="pr-rank-delta up">&#9650;3</span> / <span class="pr-rank-delta down">&#9660;3</span> spots moved in the overall rankings over the last 7 days. The small line beside the player is the 7-day value trend.</p>
                  </div>
                </div>
              </span>
              <!-- Settings button -->
              <span style="position:relative;">
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
              </span>
            </span>
          </div>
        </div>

        <!-- Loading: a skeleton list shaped like the ranked rows, so the table
             swaps in without a spinner and without the layout jumping. This
             block lives in a plain (non-f) string, so the rows are literal. -->
        <div id="prLoading" class="sk-list" role="status" aria-live="polite" aria-busy="true" style="margin-top:8px;">
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-80"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-60"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-80"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-60"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-80"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-60"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-80"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-60"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
        </div>

        <!-- Player count + CSV export (one row, per the approved mobile mock;
             app.py's SSR pass swaps this exact string to reveal the row) -->
        <div id="prCountRow" style="display:none;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;">
          <span id="prCount" style="font-size:13px;color:var(--text-muted);"></span>
          <button type="button" onclick="prExportCSV()" title="Download current list as CSV"
            style="padding:5px 10px;border-radius:8px;border:1px solid var(--border);
                   background:var(--surface);color:var(--text);font-size:12px;font-weight:600;
                   cursor:pointer;display:flex;align-items:center;gap:6px;white-space:nowrap;">
            <img src="/static/images/download-solid.png" style="width:12px;height:12px;vertical-align:middle;opacity:0.8;" alt="" aria-hidden="true"> Export CSV
          </button>
        </div>

        <div id="prTableScroll">
        <!-- Table header -->
        <div id="prTableHeader" style="display:none;
             grid-template-columns:50px minmax(0,1fr) 64px 96px 108px;
             padding:7px 12px;border-radius:6px;
             background:var(--accent-soft);font-size:11px;
             font-weight:700;color:var(--accent);letter-spacing:0.04em;
             text-transform:uppercase;" class="pr-grid-row">
          <span>#</span>
          <span data-pr-sort-col="player" role="button" tabindex="0" title="Sort by player name">Player</span>
          <span id="prPpgHeader" data-pr-sort-col="ppg" role="button" tabindex="0" title="Sort by fantasy points per game" style="text-align:center;">PPG</span>
          <span id="prTotalHeader" data-pr-sort-col="total_pts" role="button" tabindex="0" title="Sort by total points" style="text-align:center;">Total Points</span>
          <span id="prSortHeader" data-pr-sort-col="sort" role="button" tabindex="0" title="Sort direction: click to flip" style="text-align:center;">Value</span>
        </div>

        <!-- Player rows -->
        <div id="prList"></div>
        </div>

        <!-- Empty state -->
        <div id="prEmpty" class="empty-state is-compact" style="display:none;" role="status">
          <span class="empty-state-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m20 20-4.7-4.7"/></svg>
          </span>
          <p class="empty-state-title">No players match</p>
          <p class="empty-state-msg">Try clearing a filter or searching a different name.</p>
        </div>

      </div>
    </div>

    <style>
      /* One grid for header + rows (approved rework): rank | player |
         PPG | Total Points | Value. Pos / team / age / sparkline live in
         the player subline now, so the numeric columns are wide enough to
         center exactly under their headers. */
      .pr-grid-row {
        display: grid;
        grid-template-columns: 50px minmax(0, 1fr) 64px 96px 108px;
        align-items: center;
        column-gap: 8px;
      }
      .pr-grid-row.pr-adp-mode { column-gap: 0; }
      .pr-player-row {
        padding: 0 12px;
        min-height: 56px;
        align-items: stretch;
        cursor: pointer;
        transition: background 0.12s ease;
      }
      .pr-player-row:hover { background: var(--accent-soft); }
      .pr-ppg { text-align: center; font-size: 14px; font-weight: 600; font-variant-numeric: tabular-nums; align-self: center; }
      .pr-total { text-align: center; font-size: 13px; font-weight: 600; color: var(--text-muted); font-variant-numeric: tabular-nums; align-self: center; }
      /* Headshot cell (player-modal treatment): full row height, square
         corners, bust flush on the row's bottom hairline. The initials
         disc is the fallback only, revealed when every image source
         fails (see prImgFallback in rankings.js). */
      .pr-photo {
        flex: none;
        align-self: stretch;
        display: flex;
        align-items: flex-end;
        width: 62px;
      }
      .pr-hs {
        width: 62px;
        height: 100%;
        display: block;
        object-fit: cover;
        object-position: top;
        border-radius: 0;
        background: transparent;
      }
      .pr-mono {
        width: 38px;
        height: 38px;
        flex: none;
        border-radius: 50%;
        align-items: center;
        justify-content: center;
        align-self: center;
        margin: 0 auto;
        background: var(--accent);
        color: #fff;
        font-size: 12.5px;
        font-weight: 800;
        letter-spacing: 0.02em;
      }
      .pr-name-text {
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: center;
        gap: 2px;
        min-width: 0;
        align-self: center;
      }
      .pr-age-sep {
        color: var(--text-muted);
        opacity: 0.6;
        font-weight: 600;
        flex: none;
      }
      .pr-age-inline { font-weight: 600; color: var(--text-muted); white-space: nowrap; }
      .pr-player-row + .pr-player-row { border-top: 1px solid var(--border); }
      #prTableScroll.pr-adp-scroll {
        overflow-x: auto;
        overscroll-behavior-x: contain;
        scrollbar-width: thin;
        /* Stay viewport-width so wide ADP rows scroll *inside* this wrapper
           instead of expanding the page (html/body clip overflow-x). */
        width: 100%;
        max-width: 100%;
        min-width: 0;
      }
      #prTableScroll.pr-adp-scroll .pr-adp-mode {
        min-width: max-content;
      }
      #prTableScroll.pr-adp-scroll .pr-name {
        min-width: 150px;
      }
      /* A sticky child cannot track the page viewport through an overflow-x
         scroller reliably: browsers treat that wrapper as its sticky scroll
         container, which made the header float over arbitrary player rows.
         Keep the ADP header at the top of its horizontally scrolling table. */
      #prTableScroll.pr-adp-scroll #prTableHeader {
        position: static;
      }
      .pr-rank {
        font-size: 14px;
        font-weight: 700;
        color: var(--text-muted);
        display: inline-flex;
        align-items: baseline;
        gap: 4px;
        justify-content: flex-start;
        white-space: nowrap;
        align-self: center;
        line-height: 1.25;
        font-variant-numeric: tabular-nums;
      }
      .pr-rank-arrow {
        font-size: 15px;
        font-weight: 700;
        line-height: 1;
      }
      .pr-rank-arrow.up   { color: #22c55e; }
      .pr-rank-arrow.down { color: #ef4444; }
      /* Overall rank movement, inline to the RIGHT of the rank number. */
      .pr-rank-delta {
        font-size: 10.5px;
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
        font-size: 11px;
      }
      .pr-name {
        font-size: 14.5px;
        font-weight: 600;
        color: var(--text);
        display: flex;
        flex-direction: row;
        align-items: stretch;
        gap: 11px;
        min-width: 0;
        height: 100%;
      }
      .pr-name-main {
        display: flex;
        align-items: center;
        gap: 5px;
        flex-wrap: wrap;
        min-width: 0;
      }
      /* Subline under the name at every width (approved rework): sparkline,
         then "POSRANK · TEAM" as plain text exactly like the live join,
         then a muted middle-dot separator, then the age. */
      .pr-sub {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        font-weight: 600;
        color: var(--text-muted);
        white-space: nowrap;
      }
      .pr-value {
        text-align: center;
        font-size: 19px;
        font-weight: 800;
        color: var(--accent);
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
        align-self: center;
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
      .pr-adp-sort-caret { font-size: 11px; }
      .pr-adp-cell {
        text-align: center;
        font-size: 13px;
        font-weight: 600;
        color: var(--text-muted);
        font-variant-numeric: tabular-nums;
      }
      .pr-adp-cell-active { color: var(--accent); font-weight: 700; }
      /* The number keeps its own centered box; the arrow hangs off its right edge
         with absolute positioning so it never shifts the number off-center. */
      .pr-adp-num { position: relative; display: inline-block; }
      /* Comparison arrow vs the sorted ADP source: green ▲ when this source ranks
         the player higher (earlier pick), red ▼ when lower (later pick). */
      .pr-adp-arrow { position: absolute; left: 100%; top: 50%; transform: translateY(-50%); margin-left: 3px; font-size: 11px; line-height: 1; }
      .pr-adp-arrow.up   { color: var(--win); }
      .pr-adp-arrow.down { color: var(--loss); }
      /* Compact Pos/Age/Team columns kept in the ADP view on desktop */
      .pr-adp-meta-h {
        text-align: center;
        font-size: 11px;
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
      /* Clickable sort headers (AM-table pattern): pointer + arrow on active */
      #prTableHeader [data-pr-sort-col] {
        cursor: pointer;
        user-select: none;
        -webkit-tap-highlight-color: transparent;
      }
      #prTableHeader [data-pr-sort-col]:hover { color: var(--text); }
      #prTableHeader [data-pr-sort-col]:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
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
        gap: 6px;
        flex-wrap: wrap;
      }
      .filter-settings-btn {
        padding: 7px 14px;
        border-radius: 8px;
        border: 1px solid var(--border);
        background: var(--card-bg);
        color: var(--text);
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
        transition: transform 0.12s ease,opacity 0.12s ease,background-color 0.12s ease,color 0.12s ease,border-color 0.12s ease,box-shadow 0.12s ease;
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
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
        transition: transform 0.12s ease,opacity 0.12s ease,background-color 0.12s ease,color 0.12s ease,border-color 0.12s ease,box-shadow 0.12s ease;
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
        border-radius: var(--radius-pill, 8px);
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
        font-size: 13px;
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

      /* Merged header rows (approved rework) */
      /* Row 1 is a grid with fixed tracks for the format segments and the
         search box: the title/subtitle column is 1fr, so the subtitle
         getting longer in Redraft ("this-season redraft value") can no
         longer push the Dynasty/Redraft + 1QB/SF toggles and the search
         sideways when switching scoring type. The subtitle itself is
         pinned to one line so it can never wrap and push Row 2 down. */
      .pr-head-row1 {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto auto minmax(190px, 240px);
        align-items: center;
        justify-content: stretch;
        gap: 14px;
      }
      .pr-titleblock { min-width: 0; }
      .pr-titleblock .page-head-sub {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
      }
      .pr-head-row1 .filter-search {
        margin-left: 0;
        flex: none;
        min-width: 0;
        width: 100%;
      }
      .pr-seg {
        display: inline-flex;
        gap: 3px;
        padding: 3px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: var(--bg-alt, #f4f6f8);
        flex: none;
      }
      .pr-segbtn {
        border: 0;
        background: transparent;
        color: var(--text-muted);
        font: inherit;
        font-size: 12.5px;
        font-weight: 700;
        padding: 6px 12px;
        border-radius: 7px;
        cursor: pointer;
        white-space: nowrap;
      }
      .pr-segbtn.active { background: var(--accent); color: #fff; }
      .pr-seg-sm .pr-segbtn { padding: 6px 10px; font-size: 12px; }
      .pr-head-row2 { justify-content: flex-start; gap: 6px; flex-wrap: wrap; }
      .pr-head-row2 .filter-positions { gap: 4px; }
      .pr-head-row2 .pr-vdiv { margin: 0; }
      .pr-grouplab {
        font-size: 10.5px;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-right: 1px;
      }
      /* Sliding-pill sort: the Teams page .teams-mode-toggle /
         .teams-mode-thumb treatment from dashboard.css, verbatim mechanism
         (thumb positioned by prSortThumb in rankings.js). */
      .pr-sort-toggle { flex: none; }
      .pr-sort-short { display: none; }
      .pr-vdiv { width: 1px; height: 22px; background: var(--border); margin: 0 3px; flex: none; }
      .pr-pinright { margin-left: auto; display: flex; gap: 8px; flex: none; align-items: flex-start; }
      .pr-legend-tiers { display: flex; flex-direction: column; gap: 5px; font-size: 12.5px; font-weight: 600; color: var(--text); }
      .pr-legend-tiers i { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
      .pr-legend-note { font-size: 12.5px; line-height: 1.55; color: var(--text-muted); margin: 0; }

      @media (max-width: 600px) {
        .filter-positions {
          flex-wrap: nowrap;
          min-width: 0;
          align-items: center;
        }
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
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.06em;
        opacity: 0.75;
        white-space: nowrap;
      }

      /* Mobile responsive */
      @media (max-width: 768px) {
        .filter-row-primary {
          flex-direction: row;
          flex-wrap: wrap;
          align-items: center;
        }
        .filter-search {
          max-width: 100%;
        }
        .filter-positions {
          justify-content: flex-start;
          gap: 6px;
        }
        .filter-label {
          white-space: nowrap;
        }
        /* Row 2 at tablet widths: sort pill + position pills wrap as one
           group, Legend / Settings stay pinned right. */
        .filter-row-secondary {
          flex-wrap: wrap;
          gap: 8px;
        }
        .filter-sort {
          min-width: 0;
          flex-direction: row;
          align-items: center;
          gap: 6px;
        }
        .filter-sort .filter-label {
          font-size: 10px;
        }
        .filter-sort select {
          width: auto;
          min-width: 0;
          max-width: 148px;
          min-height: 32px;
          height: 32px;
          font-size: 12px;
          padding-top: 0;
          padding-bottom: 0;
        }
        /* The ADP-source view (.pr-adp-mode) manages its own columns inline,
           so it is excluded from the fixed grid overrides. */

        /* ADP board (mobile): pin # + Player like schedule rankings; source
           columns scroll horizontally in #prTableScroll. left offsets match
           PR_ADP_RANK_W_MOBILE / PR_ADP_PLAYER_W_MOBILE in rankings.js. */
        #prTableScroll.pr-adp-scroll {
          -webkit-overflow-scrolling: touch;
        }
        #prTableScroll.pr-adp-scroll .pr-adp-mode.pr-player-row,
        #prTableScroll.pr-adp-scroll #prTableHeader.pr-adp-mode {
          padding-left: 0;
          padding-right: 0;
        }
        #prTableScroll.pr-adp-scroll .pr-adp-mode > .pr-adp-pin-rank,
        #prTableScroll.pr-adp-scroll #prTableHeader.pr-adp-mode > .pr-adp-pin-rank {
          position: sticky;
          left: 0;
          z-index: 2;
          box-sizing: border-box;
          padding-left: 8px;
          background: var(--card);
        }
        #prTableScroll.pr-adp-scroll .pr-adp-mode > .pr-adp-pin-player,
        #prTableScroll.pr-adp-scroll #prTableHeader.pr-adp-mode > .pr-adp-pin-player {
          position: sticky;
          left: 40px;
          z-index: 2;
          box-sizing: border-box;
          padding-right: 8px;
          min-width: 0;
          background: var(--card);
          border-right: 1px solid var(--border);
          box-shadow: 6px 0 7px -5px rgba(15, 23, 42, 0.18);
        }
        #prTableScroll.pr-adp-scroll #prTableHeader.pr-adp-mode > .pr-adp-pin-rank,
        #prTableScroll.pr-adp-scroll #prTableHeader.pr-adp-mode > .pr-adp-pin-player {
          background: var(--accent-soft);
          z-index: 3;
        }
        #prTableScroll.pr-adp-scroll #prTableHeader.pr-adp-mode > .pr-adp-pin-player {
          box-shadow: 6px 0 7px -5px rgba(15, 23, 42, 0.28);
        }
        #prTableScroll.pr-adp-scroll .pr-player-row.pr-adp-mode:hover > .pr-adp-pin {
          background: var(--accent-soft);
        }
        #prTableScroll.pr-adp-scroll .pr-adp-mode > .pr-adp-cell:last-child,
        #prTableScroll.pr-adp-scroll #prTableHeader.pr-adp-mode > .pr-adp-head:last-child {
          padding-right: 8px;
        }
        #prTableScroll.pr-adp-scroll .pr-adp-mode .pr-name {
          flex-wrap: nowrap;
          overflow: hidden;
          white-space: nowrap;
          text-overflow: ellipsis;
        }
      }
      @media (max-width: 640px) {
        /* Phone (approved rework): ONE grid for header + rows --
           rank | player | PPG | Value, tight 3px gaps. The Total Points
           header hides (its sort flips the Value column instead, the
           shipped #2716 semantics), and the rank delta sits inline to the
           RIGHT of the rank number. This page's inline sheet is the ONLY
           source of phone grid rules for this table (dashboard.css carries
           none). */
        .pr-grid-row:not(.pr-adp-mode) {
          grid-template-columns: 40px minmax(0, 1fr) 34px 60px !important;
          column-gap: 3px !important;
        }
        #prTableHeader:not(.pr-adp-mode) span:nth-child(4) { display: none !important; }
        /* Phone rows pad 0 10px, so the header pads to 10px too or its
           tracks sit 2px right of the rows'. */
        #prTableHeader:not(.pr-adp-mode) { padding: 7px 10px !important; }
        /* The Total Points cell hides too: on phones its sort flips the
           Value column (total on top, VAL sublabel), it is not a column. */
        .pr-total { display: none !important; }
        .pr-player-row { padding: 0 10px; min-height: 52px; }
        .pr-photo { width: 50px; }
        .pr-hs { width: 50px; }
        .pr-name { gap: 7px; }
        .pr-name-main { font-size: 13px; }
        .pr-sub { gap: 4px; font-size: 10.5px; }
        .pr-rank { font-size: 14px; }
        .pr-rank-delta { font-size: 10px; }
        .pr-ppg { font-size: 14px; }
        .pr-value { font-size: 15px; }
        /* Phone control stack (approved rework): title row, search,
           full-width sliding sort pill with equal segments, one nowrap
           row of six equal-flex position pills, then the count line.
           No settings/tags row on phones. */
        .pr-head-row1 .page-title { font-size: 20px; }
        /* Phone stack stays the flex layout it was approved as; the
           fixed-track grid above is for the wider header only. The
           subtitle also goes back to wrapping here: at phone widths it
           cannot fit one line next to the scoring segment either way. */
        .pr-head-row1 { display: flex; flex-wrap: wrap; gap: 8px; }
        .pr-titleblock .page-head-sub { white-space: normal; overflow: visible; text-overflow: clip; }
        .pr-titleblock { flex: 1 1 auto; }
        #prFormatSeg { display: none; }
        .pr-head-row1 .filter-search {
          order: 3;
          flex: 1 1 100%;
          min-width: 0;
          margin-left: 0;
        }
        .filter-row-secondary { flex-wrap: wrap; gap: 8px; }
        .pr-grouplab, .pr-vdiv, .pr-pinright { display: none; }
        .pr-sort-toggle { flex: 1 1 100%; }
        .pr-sort-toggle button { flex: 1; padding: 7px 0; font-size: 12px; }
        .pr-sort-full { display: none; }
        .pr-sort-short { display: inline; }
        .filter-positions { flex: 1 1 100%; flex-wrap: nowrap; overflow: visible; }
        .filter-positions .pos-pill { flex: 1; min-height: 40px; }
        /* ADP-source view on phones: compact header labels so "Sleeper" /
           "BR Fantasy" / "Consensus" read clearly in the scrollable pane. */
        #prTableHeader.pr-adp-mode .pr-adp-head {
          font-size: 11px;
          letter-spacing: 0;
          padding: 0 1px;
        }
        #prTableHeader.pr-adp-mode .pr-adp-sort-caret { font-size: 11px; }
        .pr-adp-mode .pr-adp-cell { font-size: 13px; }
      }

      /* Sticky column header -- keep the # / Player / Value labels visible while
         scrolling a long list (mock #3, annotation "Sticky header"). .card.central
         sets overflow:hidden, which would trap position:sticky, so this page opts
         its own card back into overflow:visible (nothing here needs the clip).
         --pr-sticky-top is measured from the sticky top-nav in rankings.js so the
         header docks right under it across desktop, mobile and PWA safe-areas. */
      .card.central { overflow: visible; }
      #prTableHeader {
        position: sticky;
        top: var(--pr-sticky-top, 56px);
        z-index: 20;
        box-shadow: 0 8px 12px -10px rgba(0, 0, 0, 0.28);
      }
    </style>

    <!-- Player Rankings JS lives in static/rankings.js (injected below, deferred) -->
    """
    return body_html
