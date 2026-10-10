"""
Rookies page HTML builder.

Returns the full body HTML for the /<platform>/<season>/<league_id>/rookies route.
Data is loaded client-side from /api/prospects/rankings so the page is fast and
the filters/sorts are instant without round-trips.
"""
from __future__ import annotations


def build_prospects_body(is_admin: bool = False) -> str:
    admin_flag = "true" if is_admin else "false"
    return f"""<script>window.RK_IS_ADMIN = {admin_flag};</script>""" + """
<div class="card central" id="prospectsCard">
  <div class="card-header rk-card-header">
    <div>
      <h2 id="rookiesTitle">Rookie Prospects</h2>
      <div style="font-size: 15px; color: var(--text-muted); margin-top: 4px;">
        Dynasty prospect rankings - production, athleticism, and draft capital combined
      </div>
      <div id="rkPausedBanner" style="display:none;margin-top:10px;padding:8px 12px;border-radius:8px;
           border:1px solid color-mix(in srgb,#b45309 35%,var(--border));background:color-mix(in srgb,#b45309 12%,transparent);
           font-size:13px;color:var(--text);">
        Prospect rankings are paused until the next draft cycle. Showing the last published board
        <span id="rkPausedAsOf"></span>.
      </div>
    </div>
  </div>

  <!-- Rankings panel -->
  <div id="rk-panel-rankings" class="card-body" style="padding-top:0;">

    <!-- Controls -->
    <div class="filter-controls-container">
      <!-- Row 1: Search + pills + settings -->
      <div class="filter-row filter-row-primary">
        <div class="filter-search">
          <input id="rookieSearch" type="text" placeholder="Search prospects…" autocomplete="off"
            style="width:100%;padding:8px 32px 8px 34px;border-radius:8px;
                   border:1px solid var(--border);background:var(--card-bg);
                   color:var(--text);font-size:13px;outline:none;box-sizing:border-box;">
          <span style="position:absolute;left:10px;top:50%;transform:translateY(-50%);
                       color:var(--text-muted);font-size:15px;pointer-events:none;"><i class="fa-solid fa-magnifying-glass" aria-hidden="true"></i></span>
          <button id="rookieSearchClear" onclick="rkClearSearch()"
            style="display:none;position:absolute;right:8px;top:50%;transform:translateY(-50%);
                   background:none;border:none;cursor:pointer;color:var(--text-muted);
                   font-size:15px;padding:2px;">&#x2715;</button>
        </div>
        <div class="otc-day-filters filter-positions">
          <button class="otc-day-filter pos-pill active" data-pos="ALL" onclick="rkTogglePos('ALL')">All</button>
          <button class="otc-day-filter pos-pill" data-pos="QB"  onclick="rkTogglePos('QB')">QB</button>
          <button class="otc-day-filter pos-pill" data-pos="RB"  onclick="rkTogglePos('RB')">RB</button>
          <button class="otc-day-filter pos-pill" data-pos="WR"  onclick="rkTogglePos('WR')">WR</button>
          <button class="otc-day-filter pos-pill" data-pos="TE"  onclick="rkTogglePos('TE')">TE</button>
        </div>
      </div>

      <!-- Row 2: Active setting tags + sort -->
      <div class="filter-row filter-row-secondary" style="justify-content:flex-end;">
        <div class="filter-sort">
          <label class="filter-label">Sort by</label>
          <select id="rkSort" onchange="rkSetSortKey(this.value)"
            style="padding:7px 10px;border-radius:8px;border:1px solid var(--border);
                   background:var(--card-bg);color:var(--text);font-size:13px;
                   cursor:pointer;outline:none;min-height:34px;width:140px;">
            <option value="rank">Overall Rank</option>
            <option value="score">Prospect Score</option>
            <option value="mock">Mock Draft</option>
            <option value="age">Age</option>
            <option value="name">Name (A-Z)</option>
          </select>
        </div>
      </div>
    </div>

    <!-- Count -->
    <div id="rkCount" style="font-size:13px;color:var(--text-muted);margin-bottom:8px;display:none;"></div>

    <!-- Table header -->
    <div id="rkHeader" class="rk-grid-row rk-header" style="display:none;">
      <span data-rk-sort-col="rank" role="button" tabindex="0" title="Sort by overall rank" style="text-align:center;">Rank</span>
      <span data-rk-sort-col="name" role="button" tabindex="0" title="Sort by prospect name">Prospect</span>
      <span><span class="hdot" style="background:#10b981"></span>Prod</span>
      <span><span class="hdot" style="background:#3b82f6"></span>Ath</span>
      <span><span class="hdot" style="background:#8b5cf6"></span>DC</span>
      <span data-rk-sort-col="score" role="button" tabindex="0" title="Sort by prospect score" style="text-align:right;">Score</span>
      <span data-rk-sort-col="mock" role="button" tabindex="0" title="Sort by expected draft spot" style="text-align:center;" class="rk-mock-h">Mock</span>
      <span></span>
    </div>

    <!-- Loading -->
    <div id="rkLoading" class="sk-list" style="margin-top:4px;">
      <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line" style="width:46%"></div><div class="skeleton skeleton-line" style="width:30%;height:9px"></div></div><div class="skeleton sk-chip"></div></div>
      <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line" style="width:38%"></div><div class="skeleton skeleton-line" style="width:26%;height:9px"></div></div><div class="skeleton sk-chip"></div></div>
      <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line" style="width:52%"></div><div class="skeleton skeleton-line" style="width:34%;height:9px"></div></div><div class="skeleton sk-chip"></div></div>
      <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line" style="width:42%"></div><div class="skeleton skeleton-line" style="width:24%;height:9px"></div></div><div class="skeleton sk-chip"></div></div>
      <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line" style="width:48%"></div><div class="skeleton skeleton-line" style="width:30%;height:9px"></div></div><div class="skeleton sk-chip"></div></div>
      <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line" style="width:40%"></div><div class="skeleton skeleton-line" style="width:28%;height:9px"></div></div><div class="skeleton sk-chip"></div></div>
    </div>

    <!-- Rows -->
    <div id="rkList"></div>

    <!-- Empty -->
    <div id="rkEmpty" style="display:none;">
      <div class="empty-state">
        <span class="empty-state-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m20 20-4.7-4.7"/></svg></span>
        <p class="empty-state-title">No prospects match</p>
        <p class="empty-state-msg">Try clearing a filter or widening your position and tier selections.</p>
      </div>
    </div>

    <!-- Pagination -->
    <div id="rkPagination" class="pagination" style="display:none;">
      <div class="pagination-info">
        <span id="rkPaginationText">Showing 1-20 of 100 prospects</span>
      </div>
      <div class="pagination-controls">
        <button id="rkPrevBtn" class="pagination-btn" onclick="rkLoadPage('prev')" disabled>
          <i class="fa-solid fa-chevron-left"></i> Previous
        </button>
        <div id="rkPageNumbers" class="pagination-pages"></div>
        <button id="rkNextBtn" class="pagination-btn" onclick="rkLoadPage('next')" disabled>
          Next <i class="fa-solid fa-chevron-right"></i>
        </button>
      </div>
    </div>

  </div><!-- end rk-panel-rankings -->

</div>

<!-- Prospect detail modal -->
<div id="rkModal" style="display:none;position:fixed;inset:0;z-index:10500;
     display:none;align-items:center;justify-content:center;padding:20px;
     background:rgba(15,23,42,0.7);backdrop-filter:blur(4px);">
  <div id="rkModalContent"
    style="background:var(--card);border-radius:16px;max-width:920px;width:100%;
           max-height:90vh;overflow-y:auto;
           box-shadow:0 24px 48px rgba(15,23,42,0.25);">
    <!-- filled by JS -->
  </div>
</div>

<style>
  /* Prospects card-header: override global 50px height, allow vertical stacking */
  .rk-card-header {
    height: auto !important;
    min-height: 50px;
    align-items: flex-start;
    flex-wrap: wrap;
    padding: 14px 0 0;
    gap: 10px;
  }
  @media (max-width: 600px) {
    .rk-card-header {
      flex-direction: column;
      align-items: flex-start;
      gap: 10px;
    }
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
  .filter-row-secondary .filter-sort { flex-shrink: 0; }
  .filter-search {
    position: relative;
    flex: 1;
    min-width: 200px;
  }
  .filter-positions {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    flex-wrap: nowrap;
    scrollbar-width: none;
    -webkit-overflow-scrolling: touch;
  }
  .filter-positions::-webkit-scrollbar { display: none; }
  .active-setting-tag {
    padding: 4px 10px;
    border-radius: var(--radius-pill, 8px);
    background: var(--accent-soft);
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
  }
  .filter-sort {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .filter-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  /* Mobile responsive */
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
      min-width: 0;
    }
    .filter-row-secondary { gap: 8px; }
    .manager-pills-row .manager-pill:not(.active) { display: none; }
    .manager-pills-row { justify-content: center; }
  }

  /* Table grid: Rank | Player | Prod | Ath | DC | Score | Mock | star */
  .rk-grid-row {
    display: grid;
    grid-template-columns: 64px minmax(0,1fr) 96px 96px 96px 60px 56px 40px;
    align-items: center;
    gap: 8px;
  }
  .rk-header {
    padding: 12px 12px 8px;
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }
  .hdot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 5px; }
  /* Clickable sort headers (AM-table pattern): pointer + arrow on active */
  #rkHeader [data-rk-sort-col] {
    cursor: pointer;
    user-select: none;
    -webkit-tap-highlight-color: transparent;
  }
  #rkHeader [data-rk-sort-col]:hover { color: var(--text); }
  #rkHeader [data-rk-sort-col]:focus-visible {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
  }
  .rk-row {
    padding: 10px 12px;
    cursor: pointer;
    transition: background 0.12s;
    border-top: 1px solid var(--border);
  }
  .rk-row:hover { background: var(--accent-soft); }

  .rk-rank { display: flex; align-items: center; justify-content: center; gap: 7px; }
  .rk-rank-num { font-size: 14px; font-weight: 700; color: var(--text-muted); }
  .rk-pcell { min-width: 0; }
  .rk-pplayer { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .rk-headshot {
    width: 38px; height: 38px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 800; letter-spacing: 0.02em;
    background: color-mix(in srgb, var(--pc, var(--accent)) 15%, transparent);
    color: var(--pc, var(--accent));
    border: 1px solid color-mix(in srgb, var(--pc, var(--accent)) 35%, transparent);
    position: relative; overflow: hidden;
  }
  .rk-headshot img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
  .rk-name { font-size: 13.5px; font-weight: 600; color: var(--text); }
  .rk-name:hover { opacity: 0.72; }
  .rk-meta { font-size: 11px; color: var(--text-muted); margin-top: 1px; }
  .rk-pos {
    display: inline-block; font-size: 10px; font-weight: 700; padding: 1px 6px; border-radius: 4px;
    background: color-mix(in srgb, var(--pc, var(--accent)) 15%, transparent);
    color: var(--pc, var(--accent));
    margin-left: 6px; vertical-align: middle;
  }
  .rk-comps3 { display: contents; }
  .rk-compcol { min-width: 0; }
  .rk-compcol-label { display: none; }
  .rk-compcol-row { display: flex; align-items: center; gap: 6px; }
  .rk-meter-bar { flex: 1; height: 5px; border-radius: 3px; background: color-mix(in srgb, var(--text-muted) 14%, transparent); overflow: hidden; }
  .rk-meter-bar > div { height: 100%; border-radius: 3px; }
  .rk-meter-val { font-size: 11.5px; font-weight: 800; }
  .rk-score { text-align: right; font-size: 14px; font-weight: 700; color: var(--text); }
  .rk-mock { text-align: center; font-size: 12.5px; color: var(--text-muted); }
  .rk-star { background: none; border: none; cursor: pointer; padding: 6px; display: flex; }
  .rk-star svg { width: 17px; height: 17px; fill: none; stroke: var(--text-subtle); stroke-width: 1.8; }
  .rk-star.watching svg { fill: var(--gold); stroke: var(--gold); }

  /* Modal */
  #rkModalContent { border-top: 3px solid transparent; }
  .rk-modal-header {
    padding: 20px 20px 0;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
  }
  .rk-modal-close {
    background: var(--accent-soft);
    border: none;
    width: 32px; height: 32px;
    border-radius: 8px;
    cursor: pointer;
    color: var(--accent);
    font-size: 16px;
    flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
  }
  .rk-modal-body { padding: 14px 20px 20px; }
  .m-headwrap { display: flex; gap: 12px; align-items: center; min-width: 0; }
  .m-headshot {
    width: 52px; height: 52px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px; font-weight: 800;
    background: color-mix(in srgb, var(--pc, var(--accent)) 15%, transparent);
    color: var(--pc, var(--accent));
    border: 1px solid color-mix(in srgb, var(--pc, var(--accent)) 35%, transparent);
    position: relative; overflow: hidden;
  }
  .m-headshot img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
  .m-title-row { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .m-name { font-size: 22px; font-weight: 700; color: var(--text); }
  .m-tier { padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; }
  .m-early {
    padding: 3px 8px; border-radius: 6px; font-size: 11px; font-weight: 700;
    background: color-mix(in srgb, var(--text-muted) 12%, transparent); color: var(--text-muted);
  }
  .m-meta { font-size: 13px; color: var(--text-muted); margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .m-meta .m-rank { font-weight: 700; color: var(--text); }
  .m-transfer { font-size: 12px; color: var(--text-muted); margin-top: 4px; }

  /* Hero row */
  .rk-hero-row {
    display: grid;
    grid-template-columns: 1.2fr 1fr 1fr;
    gap: 8px;
    margin: 14px 0 4px;
  }
  .rk-hero-stat {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 14px;
    text-align: center;
  }
  .rk-hero-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
  }
  .rk-hero-val {
    font-size: 26px;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
  }
  .rk-hero-sub {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 4px;
  }

  /* Section divider */
  .rk-section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 14px 0;
  }
  .m-sec-label { font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 2px; }
  .m-sec-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 2px; }
  .m-conf { font-size: 11px; color: var(--text-muted); }
  .m-conf b { color: var(--text); }
  .m-comp-row { display: grid; grid-template-columns: 96px 1fr 38px auto; gap: 10px; align-items: center; margin-top: 12px; }
  .m-comp-label { font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); }
  .m-comp-row .rk-meter-bar { height: 7px; flex: none; }
  .m-comp-row .rk-meter-val { font-size: 13px; }
  .m-comp-raw { font-size: 12px; color: var(--text-subtle); }
  .m-more-btn {
    background: none; border: none; color: var(--accent); font-size: 12.5px;
    font-weight: 700; cursor: pointer; padding: 10px 0 0;
  }

  /* Advanced metrics */
  .m-adv-grid { display: grid; grid-template-columns: 1fr 230px; gap: 20px; align-items: center; margin-top: 10px; }
  .m-adv-row { display: grid; grid-template-columns: 104px 1fr 52px 30px; gap: 10px; align-items: center; margin-top: 9px; }
  .m-adv-label { font-size: 12px; font-weight: 600; color: var(--text-muted); }
  .m-adv-row .rk-meter-bar { height: 7px; flex: none; }
  .m-adv-score { font-size: 13px; font-weight: 700; text-align: right; color: var(--text); }
  .m-adv-grade { font-size: 13px; font-weight: 800; text-align: right; }
  .m-adv-radar { display: flex; justify-content: center; padding: 10px 14px; }

  /* Two-column modal layout */
  .m-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 0 28px; align-items: start; }
  .m-col { min-width: 0; }
  .m-col .m-season-table { font-size: 12px; }
  .m-col .m-season-table th, .m-col .m-season-table td { padding: 6px 6px; }

  /* Combine grid */
  .rk-meas-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 10px; }
  .rk-meas-cell {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 8px;
    text-align: center;
  }
  .rk-meas-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.03em;
    margin-bottom: 4px;
  }
  .rk-meas-val {
    font-size: 15px;
    font-weight: 700;
    color: var(--text);
  }

  /* Season table */
  .m-season-table { width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 12.5px; color: var(--text); }
  .m-season-table th {
    text-align: right; font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--text-subtle); padding: 6px 8px; border-bottom: 1px solid var(--border);
  }
  .m-season-table th:first-child, .m-season-table td:first-child { text-align: left; }
  .m-season-table td { text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--border); }
  .m-season-table tr:last-child td { border-bottom: none; }
  .m-season-table td.yr { font-weight: 700; color: var(--text-muted); }
  .m-season-table tr.rk-career-row td { border-top: 2px solid var(--border); font-weight: 700; }
  .m-season-table tr.rk-career-row td.yr { color: var(--text); }

  /* Scouting notes */
  .m-notes { font-size: 13px; color: var(--text-muted); line-height: 1.7; margin-top: 8px; }
  .m-notes div { padding: 2px 0; }

  /* Historical comparables */
  .m-sim-row { display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid var(--border); }
  .m-sim-row:last-child { border-bottom: none; }
  .m-sim-disc {
    width: 32px; height: 32px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 800;
    background: color-mix(in srgb, var(--pc, var(--accent)) 15%, transparent);
    color: var(--pc, var(--accent));
    border: 1px solid color-mix(in srgb, var(--pc, var(--accent)) 35%, transparent);
    position: relative; overflow: hidden;
  }
  .m-sim-disc img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }
  .m-sim-name { font-size: 13px; font-weight: 600; color: var(--text); }
  .m-sim-meta { font-size: 11px; color: var(--text-muted); margin-top: 1px; }
  .m-sim-right { margin-left: auto; display: flex; align-items: center; gap: 8px; flex-shrink: 0; }
  .m-sim-score { font-size: 13px; font-weight: 700; color: var(--text); }
  .m-sim-tier { padding: 2px 7px; border-radius: 5px; font-size: 11px; font-weight: 700; }

  /* Tier-change nudge */
  .m-nudge {
    display: flex; align-items: center; gap: 8px; margin-top: 14px; padding: 10px 12px;
    border: 1px dashed var(--border); border-radius: 10px; font-size: 13px; color: var(--text-muted);
    cursor: pointer;
  }
  .m-nudge.on { border-style: solid; border-color: color-mix(in srgb, var(--gold) 45%, var(--border)); color: var(--text); }
  .m-nudge .rk-star { padding: 0; }

  /* Detail loading shimmer */
  .rk-detail-loading { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-muted); padding: 12px 0; }

  /* Mobile: hide Mock column, stack component bars under the row */
  @media (max-width: 768px) {
    .rk-head { display: none; }
    .rk-grid-row { grid-template-columns: 48px minmax(0,1fr) 48px 32px; }
    #rkHeader .rk-mock-h, .rk-row .rk-mock { display: none; }
    .rk-comps3 { display: grid; grid-column: 1 / -1; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 8px; }
    .rk-compcol-label { display: block; font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-subtle); margin-bottom: 4px; }
    .rk-meter-val { font-size: 10.5px; }
    .rk-name { font-size: 12.5px; }
    .rk-headshot { width: 32px; height: 32px; font-size: 11px; }
    .rk-pplayer { gap: 8px; }
    .rk-row { padding: 8px 10px; }
    .m-comp-row { grid-template-columns: 96px 1fr 38px; }
    .m-comp-raw { display: none; }
    .m-name { font-size: 19px; }
    .m-cols { grid-template-columns: 1fr; }
    .m-adv-grid { grid-template-columns: 1fr; }
    .rk-meas-grid { grid-template-columns: repeat(3, 1fr); }

    /* Modal: full-screen sheet on mobile */
    #rkModal { padding: 0 !important; align-items: flex-end !important; }
    #rkModalContent {
      max-width: 100% !important;
      max-height: 94vh !important;
      border-radius: 16px 16px 0 0 !important;
    }
    .rk-modal-header { padding: 16px 16px 0; }
    .rk-modal-body { padding: 12px 16px 24px; }
    .rk-modal-close { width: 36px; height: 36px; }

    /* Hero row: tighter on mobile */
    .rk-hero-row { gap: 6px; margin: 12px 0 2px; }
    .rk-hero-stat { padding: 10px 8px; }
    .rk-hero-val { font-size: 22px; }
    .rk-hero-label { font-size: 10px; }

    /* Season table: horizontal scroll on mobile */
    .m-season-table { font-size: 11.5px; }
    .m-season-table th, .m-season-table td { padding: 6px 5px; }
    .m-col { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    .m-col .m-season-table { min-width: 460px; }

    /* Advanced metrics rows: tighter labels */
    .m-adv-row { grid-template-columns: 88px 1fr 48px 28px; gap: 8px; }
    .m-adv-label { font-size: 11px; }
    .m-adv-radar { padding: 6px 0; }

    /* Comparables: tighter */
    .m-sim-row { gap: 8px; padding: 7px 0; }
    .m-sim-disc { width: 28px; height: 28px; font-size: 10px; }

    /* Scouting notes: comfortable reading size */
    .m-notes { font-size: 13px; line-height: 1.6; }
  }

  /* Very small screens: further tightening */
  @media (max-width: 480px) {
    .rk-grid-row { grid-template-columns: 40px minmax(0,1fr) 44px 28px; gap: 6px; }
    .rk-row { padding: 8px 8px; }
    .rk-rank-num { font-size: 13px; }
    .rk-score { font-size: 13px; }
    .rk-comps3 { gap: 8px; }
    .m-headwrap { gap: 10px; }
    .m-headshot { width: 44px; height: 44px; font-size: 14px; }
    .m-name { font-size: 17px; }
    .m-meta { font-size: 12px; }
    .rk-hero-val { font-size: 20px; }
    .rk-meas-grid { grid-template-columns: repeat(2, 1fr); }
    .rk-meas-val { font-size: 14px; }
    .filter-controls-container { padding: 12px 0 10px; gap: 10px; }
  }

  /* Pagination uses the universal .pagination component in dashboard.css */
</style>

<script>
  var rkAllPlayers = [];   // full unfiltered list - never replaced after load
  var rkPosFilters = new Set();
  var rkSearch    = '';
  var rkLoaded    = false;
  var rkPipelinePaused = false;
  var rkDraftYear = null;
  var rkDraftComplete = false;
  var rkCurrentPage = 1;
  var RK_PER_PAGE = 50;
  var rkSortKey = 'rank';  // active sort key (mirrors the #rkSort dropdown)
  var rkSortDir = 'asc';   // 'asc' | 'desc' -- flipped by clicking a column header
  // Default direction when a sort key is first chosen (AM-table convention).
  var RK_SORT_DIRS = { rank: 'asc', score: 'desc', mock: 'asc', age: 'asc', name: 'asc' };

  // Watchlist (localStorage-backed) with the shared undo toast.
  var rkWatch = {};
  try {
    JSON.parse(localStorage.getItem('rkWatchlistV1') || '[]').forEach(function(id){ rkWatch[id] = true; });
  } catch (e) {}
  function rkSaveWatch() {
    try { localStorage.setItem('rkWatchlistV1', JSON.stringify(Object.keys(rkWatch))); } catch (e) {}
  }

  var RK_POS_COLORS = { QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b', TE: '#8b5cf6' };
  var RK_TIER_COLORS = ['', '#10b981', '#22d3ee', '#3b82f6', '#8b5cf6', '#a855f7', '#f59e0b', '#f97316', '#94a3b8', '#64748b'];
  var RK_TIER_NAMES = { 1: 'ELITE', 2: 'GREAT', 3: 'GOOD' };
  var RK_COMP_NAMES = ['Production', 'Athleticism', 'Draft Capital'];
  var RK_COMP_SHORT = ['Prod', 'Ath', 'DC'];
  var RK_COMP_COLORS = ['#10b981', '#3b82f6', '#8b5cf6'];
  var RK_DISC_COLORS = ['#10b981','#3b82f6','#8b5cf6','#f59e0b','#ef4444','#06b6d4','#f97316','#84cc16'];
  var RK_XCOMP = [['Utilization','#06b6d4'],['Efficiency','#818cf8'],['Durability','#f97316'],['Experience','#84cc16'],['Competition','#f472b6']];
  var RK_SEASON_COLS = {
    QB: ['Year','GP','Cmp%','Yds','Y/A','TD','INT','Dom'],
    RB: ['Year','GP','Att','Yds','YPC','TD','Dom'],
    WR: ['Year','GP','Rec','Yds','YPR','TD','Dom'],
    TE: ['Year','GP','Rec','Yds','YPR','TD','Dom']
  };
  function rkDomCell(s) {
    return s.dominator_rating != null ? Math.round(Number(s.dominator_rating) * 100) + '%' : '-';
  }

  function rkEsc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function rkInitials(name) {
    var parts = String(name || '').split(' ');
    return (((parts[0] || '').charAt(0) + (parts[1] ? parts[1].charAt(0) : '')) || '?').toUpperCase();
  }
  function rkDiscColor(pid) {
    var s = String(pid || ''), hsh = 0;
    for (var i = 0; i < s.length; i++) hsh = (hsh * 31 + s.charCodeAt(i)) >>> 0;
    return RK_DISC_COLORS[hsh % RK_DISC_COLORS.length];
  }
  function rkHeadshot(r, cls) {
    var url = r.headshot_url || r.espnHeadshot;
    var color = rkDiscColor(r.player_id);
    var init = rkInitials(r.name);
    if (url) {
      return '<div class="' + cls + '" style="--pc:' + color + '">' + init +
        '<img src="' + rkEsc(url) + '" alt="" loading="lazy" onerror="this.remove()"></div>';
    }
    return '<div class="' + cls + '" style="--pc:' + color + '">' + init + '</div>';
  }
  function rkStarSVG() {
    return '<svg viewBox="0 0 24 24"><path d="M12 2.5l2.9 6 6.6.9-4.8 4.6 1.2 6.5L12 17.4 6.1 20.5l1.2-6.5L2.5 9.4l6.6-.9z"/></svg>';
  }
  // Rank movement arrow. Positive delta = moved up the board.
  function rkDelta(delta) {
    if (delta == null || delta === '') return '';
    delta = parseInt(delta, 10);
    if (isNaN(delta)) return '';
    if (delta > 0) return '<span class="dvt-change dvt-change-up">&#9650;' + delta + '</span>';
    if (delta < 0) return '<span class="dvt-change dvt-change-down">&#9660;' + Math.abs(delta) + '</span>';
    return '<span class="dvt-change dvt-change-flat">-</span>';
  }
  function rkHeightStr(r) {
    var ht = parseInt(r.height_inches, 10);
    if (isNaN(ht)) return '';
    return Math.floor(ht / 12) + "'" + (ht % 12) + '"';
  }
  function rkWeightStr(r) {
    return r.weight_lbs ? r.weight_lbs + ' lbs' : '';
  }
  function rkLetterGrade(v) {
    if (v >= 97) return ['A+', '#10b981'];
    if (v >= 93) return ['A', '#10b981'];
    if (v >= 90) return ['A-', '#34d399'];
    if (v >= 87) return ['B+', '#34d399'];
    if (v >= 83) return ['B', '#6ee7b7'];
    if (v >= 80) return ['B-', '#6ee7b7'];
    if (v >= 77) return ['C+', '#f59e0b'];
    if (v >= 73) return ['C', '#f59e0b'];
    if (v >= 70) return ['C-', '#fbbf24'];
    if (v >= 65) return ['D+', '#f87171'];
    if (v >= 60) return ['D', '#f87171'];
    return ['F', '#ef4444'];
  }
  // Score display: one decimal, never rounded to whole, never 100.
  function rkScore1(v) {
    var n = parseFloat(v);
    if (isNaN(n)) return '-';
    n = Math.min(n, 99.9);
    return (Math.round(n * 10) / 10).toFixed(1);
  }
  function rkScoreNum(v) {
    var n = parseFloat(v);
    if (isNaN(n)) return 0;
    return Math.min(n, 99.9);
  }
  // Radar label abbreviations: full names live in the rows beside it.
  var RK_RADAR_ABBR = {
    'Dominator': 'DOM', 'Breakout Age': 'BO AGE', 'Yds/Carry': 'YDS/C',
    'Yds/Rec': 'YDS/R', 'Scrim Yds/Gm': 'SCRIM', 'Mkt Share': 'MKT SH',
    'TD Share': 'TD SH', 'Speed Score': 'SPEED', 'Efficiency': 'EFF',
    'Production': 'PROD', 'Recruiting': 'REC', 'WEPA': 'WEPA',
    'Cmp%': 'CMP%', 'TD:INT': 'TD:INT', 'Yds/Att': 'YDS/A', 'AY/A': 'AY/A'
  };
  function rkRadarSVG(labels, values, color) {
    var size = 220, cx = 110, cy = 112, R = 68, n = labels.length;
    if (!n) return '';
    function pt(k, r) {
      var a = (-90 + k * 360 / n) * Math.PI / 180;
      return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
    }
    function pts(k, r) { var q = pt(k, r); return q[0].toFixed(1) + ',' + q[1].toFixed(1); }
    var grid = [25, 50, 75, 100].map(function(g) {
      var pp = []; for (var k = 0; k < n; k++) pp.push(pts(k, R * g / 100));
      return '<polygon points="' + pp.join(' ') + '" fill="none" style="stroke:var(--border)" stroke-width="1"/>';
    }).join('');
    var axes = '';
    for (var k = 0; k < n; k++) {
      var e = pt(k, R);
      axes += '<line x1="' + cx + '" y1="' + cy + '" x2="' + e[0].toFixed(1) + '" y2="' + e[1].toFixed(1) + '" style="stroke:var(--border)" stroke-width="1"/>';
    }
    var vp = [];
    for (var k = 0; k < n; k++) vp.push(pts(k, R * Math.max(0, Math.min(100, values[k] || 0)) / 100));
    var poly = '<polygon points="' + vp.join(' ') + '" fill="' + color + '33" stroke="' + color + '" stroke-width="2"/>';
    var dots = vp.map(function(s) { var xy = s.split(','); return '<circle cx="' + xy[0] + '" cy="' + xy[1] + '" r="2.5" fill="' + color + '"/>'; }).join('');
    var labs = labels.map(function(l, k) {
      var a = (-90 + k * 360 / n) * Math.PI / 180;
      var lx = cx + (R + 20) * Math.cos(a), ly = cy + (R + 20) * Math.sin(a);
      var anchor = Math.abs(Math.cos(a)) < 0.35 ? 'middle' : (Math.cos(a) > 0 ? 'start' : 'end');
      var txt = RK_RADAR_ABBR[l] || String(l).toUpperCase();
      return '<text x="' + lx.toFixed(1) + '" y="' + (ly + 3).toFixed(1) + '" text-anchor="' + anchor + '" font-size="9" font-weight="700" style="fill:var(--text-muted)">' + rkEsc(txt) + '</text>';
    }).join('');
    return '<svg width="' + size + '" height="' + size + '" viewBox="0 0 ' + size + ' ' + size + '" style="overflow:visible" role="img">' + grid + axes + poly + dots + labs + '</svg>';
  }

  // Sort dropdown: choosing a key resets to that key's default direction.
  function rkSetSortKey(key) {
    rkSortKey = key;
    rkSortDir = RK_SORT_DIRS[key] || 'desc';
    rkCurrentPage = 1;
    rkRender();
  }
  // Clickable column headers (AM-table pattern): clicking the active key flips
  // direction, clicking a new key takes its default direction. The dropdown is
  // kept in sync so both controls always agree.
  function rkHeaderSort(key) {
    if (rkSortKey === key) rkSortDir = (rkSortDir === 'desc' ? 'asc' : 'desc');
    else { rkSortKey = key; rkSortDir = RK_SORT_DIRS[key] || 'desc'; }
    var sel = document.getElementById('rkSort');
    if (sel) sel.value = rkSortKey;
    rkCurrentPage = 1;
    rkRender();
  }
  // Arrow on the active sort header (AM-table pattern).
  function rkUpdateSortHeaders() {
    var header = document.getElementById('rkHeader');
    if (!header) return;
    var col = rkSortKey;
    header.querySelectorAll('[data-rk-sort-col]').forEach(function(el) {
      var on = el.getAttribute('data-rk-sort-col') === col;
      el.classList.toggle('sorted-asc', on && rkSortDir === 'asc');
      el.classList.toggle('sorted-desc', on && rkSortDir === 'desc');
      if (on) el.setAttribute('aria-sort', rkSortDir === 'asc' ? 'ascending' : 'descending');
      else el.removeAttribute('aria-sort');
    });
  }

  // Watchlist toggle with the shared undo toast.
  function rkToggleWatch(ev, pid, name) {
    ev.stopPropagation();
    if (rkWatch[pid]) {
      delete rkWatch[pid];
      rkSaveWatch();
      rkRender();
      return;
    }
    rkWatch[pid] = true;
    rkSaveWatch();
    rkRender();
    if (window.brUndoToast) {
      window.brUndoToast('Watching ' + name + '.', function() {
        delete rkWatch[pid];
        rkSaveWatch();
        rkRender();
        var card = document.getElementById('rkModalContent');
        if (card) rkSyncModalWatch(card, pid, name);
      });
    }
  }
  // Sync the open modal's star + nudge row after a watchlist change.
  function rkSyncModalWatch(card, pid, name) {
    var on = !!rkWatch[pid];
    card.querySelectorAll('.rk-star').forEach(function(s) { s.classList.toggle('watching', on); });
    var n = card.querySelector('.m-nudge');
    if (n) {
      n.classList.toggle('on', on);
      var sp = n.querySelector('span');
      if (sp) sp.textContent = on ? 'You will be nudged if ' + name + ' changes tiers.'
                                  : 'Get nudged if ' + name + ' changes tiers.';
    }
  }
  function rkToggleWatchModal(ev, pid, name) {
    ev.stopPropagation();
    if (rkWatch[pid]) {
      delete rkWatch[pid];
      rkSaveWatch();
    } else {
      rkWatch[pid] = true;
      rkSaveWatch();
      if (window.brUndoToast) {
        window.brUndoToast('Watching ' + name + '.', function() {
          delete rkWatch[pid];
          rkSaveWatch();
          rkRender();
          var card = document.getElementById('rkModalContent');
          if (card) rkSyncModalWatch(card, pid, name);
        });
      }
    }
    rkRender();
    var card = document.getElementById('rkModalContent');
    if (card) rkSyncModalWatch(card, pid, name);
  }

  function rkTogglePos(pos) {
    if (pos === 'ALL') {
      rkPosFilters.clear();
    } else {
      if (rkPosFilters.has(pos)) rkPosFilters.delete(pos);
      else rkPosFilters.add(pos);
    }
    document.querySelectorAll('.pos-pill').forEach(function(b) {
      var p = b.getAttribute('data-pos');
      b.classList.toggle('active', p === 'ALL' ? rkPosFilters.size === 0 : rkPosFilters.has(p));
    });
    rkCurrentPage = 1;
    rkRender();
  }

  function rkClearSearch() {
    document.getElementById('rookieSearch').value = '';
    rkSearch = '';
    document.getElementById('rookieSearchClear').style.display = 'none';
    rkCurrentPage = 1;
    rkRender();
  }

  function rkFuzzy(name, q) {
    if (!name || !q) return 0;
    var n = name.toLowerCase(), qL = q.toLowerCase();
    if (n.includes(qL)) return 100 + (100 - n.indexOf(qL));
    var nw = n.split(/[\\s\\-]+/);
    if (nw.some(function(w){ return w.startsWith(qL); })) return 60;
    return 0;
  }

  // Missing sort values always sort last, in either direction.
  function rkSortIsNull(r, sortBy) {
    switch (sortBy) {
      case 'score': return !(parseFloat(r.prospect_score||0) > 0);
      case 'mock':  return r.projected_pick == null;
      case 'age':   return r.age == null;
      case 'name':  return false;
      default:      return !(r.overall_rank > 0); // 'rank'
    }
  }
  function rkMockSortVal(r) {
    return r.projected_pick != null ? parseFloat(r.projected_pick) : 999;
  }
  function rkRender() {
    if (!rkLoaded) return;
    var sortBy = rkSortKey;
    rkUpdateSortHeaders();

    // 1. Start with full list
    var players = rkAllPlayers.slice();

    // 2. Position filter
    if (rkPosFilters.size > 0) {
      players = players.filter(function(r) {
        return rkPosFilters.has((r.position||'').toUpperCase());
      });
    }

    // 3. Search filter (fuzzy match on name)
    if (rkSearch.length > 0) {
      var q = rkSearch;
      players = players
        .map(function(r) { return {r:r, s: rkFuzzy(r.name, q)}; })
        .filter(function(x) { return x.s > 0; })
        .sort(function(a,b) { return b.s - a.s || (parseFloat(b.r.prospect_score||0) - parseFloat(a.r.prospect_score||0)); })
        .map(function(x) { return x.r; });
    } else {
      // 4. Sort (only when not searching). Direction-aware: the ascending-base
      //    comparator is flipped when the header toggle is on 'desc'. Missing
      //    values always sort last, in either direction.
      var rkDirMult = rkSortDir === 'asc' ? 1 : -1;
      players.sort(function(a, b) {
        var an = rkSortIsNull(a, sortBy), bn = rkSortIsNull(b, sortBy);
        if (an && bn) return 0;
        if (an) return 1;
        if (bn) return -1;
        switch (sortBy) {
          case 'score': return (parseFloat(a.prospect_score||0) - parseFloat(b.prospect_score||0)) * rkDirMult;
          case 'mock':  return (rkMockSortVal(a) - rkMockSortVal(b)) * rkDirMult;
          case 'age':   return (parseFloat(a.age||99) - parseFloat(b.age||99)) * rkDirMult;
          case 'name':  return String(a.name||'').localeCompare(String(b.name||'')) * rkDirMult;
          default:      return ((a.overall_rank||999) - (b.overall_rank||999)) * rkDirMult;
        }
      });
    }

    var totalFiltered = players.length;
    var totalPages = Math.max(1, Math.ceil(totalFiltered / RK_PER_PAGE));

    // Clamp page
    if (rkCurrentPage > totalPages) rkCurrentPage = 1;

    // 5. Paginate
    var offset = (rkCurrentPage - 1) * RK_PER_PAGE;
    var pageItems = players.slice(offset, offset + RK_PER_PAGE);

    var list   = document.getElementById('rkList');
    var empty  = document.getElementById('rkEmpty');
    var count  = document.getElementById('rkCount');
    var header = document.getElementById('rkHeader');

    if (pageItems.length === 0) {
      list.innerHTML = '';
      empty.style.display = 'block';
      header.style.display = 'none';
      count.style.display  = 'none';
      document.getElementById('rkPagination').style.display = 'none';
      var emptyTitle = empty.querySelector('.empty-state-title');
      var emptyMsg = empty.querySelector('.empty-state-msg');
      var pausedEmpty = rkPipelinePaused && rkAllPlayers.length === 0 && rkPosFilters.size === 0 && !rkSearch;
      if (emptyTitle) {
        emptyTitle.textContent = pausedEmpty ? 'Prospect rankings paused' : 'No prospects match';
      }
      if (emptyMsg) {
        emptyMsg.textContent = pausedEmpty
          ? 'The rookie pipeline is paused until the next draft cycle. Rankings will return when the class is rebuilt.'
          : 'Try clearing a filter or widening your position and tier selections.';
      }
      return;
    }

    empty.style.display  = 'none';
    header.style.display = 'grid';
    count.style.display  = 'none';

    list.innerHTML = '';

    // Tier dividers (player-rankings style) only make sense on the default
    // board order: all positions, no search, sorted by overall rank.
    var showDividers = rkPosFilters.size === 0 && !rkSearch &&
                       rkSortKey === 'rank' && rkSortDir === 'asc';
    var lastTier = null;

    pageItems.forEach(function(r, idx) {
      var pid = r.player_id || '';

      // Tier break: thin colored line with T-label, reusing .otc-tier-divider.
      if (showDividers && r.tier !== lastTier) {
        lastTier = r.tier;
        var tc = RK_TIER_COLORS[r.tier] || '#9ca3af';
        var div = document.createElement('div');
        div.className = 'otc-tier-divider';
        div.style.padding = '10px 12px 2px';
        div.innerHTML =
          '<div class="otc-tier-divider-line" style="background:' + tc + ';"></div>' +
          '<span class="otc-tier-divider-label" style="color:' + tc + ';">T' + (r.tier || '?') + '</span>' +
          '<div class="otc-tier-divider-line" style="background:' + tc + ';"></div>';
        list.appendChild(div);
      }

      var row = document.createElement('div');
      row.className = 'rk-row rk-grid-row';
      row.setAttribute('data-pid', pid);

      var score = parseFloat(r.prospect_score||0);
      var pos = (r.position || '').toUpperCase();
      var posColor = RK_POS_COLORS[pos] || '#94a3b8';
      var rankNum = r.overall_rank ? '#' + r.overall_rank : (offset + idx + 1);
      var ht = rkHeightStr(r), wt = rkWeightStr(r);
      var metaBits = [];
      if (r.school) metaBits.push(rkEsc(r.school));
      if (ht) metaBits.push(rkEsc(ht));
      if (wt) metaBits.push(rkEsc(wt));
      var mockPick = r.projected_pick != null ? Math.round(parseFloat(r.projected_pick)) : '-';

      var compVals = [r.production_score, r.athleticism_score, r.projected_draft_capital_score];
      var compsHtml = '<div class="rk-comps3">' + compVals.map(function(v, k) {
        var n = rkScoreNum(v), disp = rkScore1(v);
        return '<div class="rk-compcol" title="' + RK_COMP_NAMES[k] + ' ' + disp + '">' +
          '<div class="rk-compcol-label">' + RK_COMP_SHORT[k] + '</div>' +
          '<div class="rk-compcol-row"><div class="rk-meter-bar"><div style="width:' + n + '%;background:' + RK_COMP_COLORS[k] + '"></div></div>' +
          '<span class="rk-meter-val" style="color:' + RK_COMP_COLORS[k] + '">' + disp + '</span></div></div>';
      }).join('') + '</div>';

      row.innerHTML =
        '<div class="rk-rank"><span class="rk-rank-num">' + rankNum + '</span>' + rkDelta(r.rank_delta) + '</div>' +
        '<div class="rk-pcell"><div class="rk-pplayer">' +
          rkHeadshot(r, 'rk-headshot') +
          '<div style="min-width:0"><div class="rk-name">' + rkEsc(r.name || 'Unknown') +
            '<span class="rk-pos" style="--pc:' + posColor + '">' + rkEsc(pos) + '</span></div>' +
          '<div class="rk-meta">' + metaBits.join(' &middot; ') + '</div></div>' +
        '</div></div>' +
        compsHtml +
        '<div class="rk-score">' + rkScore1(score) + '</div>' +
        '<div class="rk-mock" title="Expected draft spot based on mock drafts">' + mockPick + '</div>' +
        '<button class="rk-star' + (rkWatch[pid] ? ' watching' : '') + '"' +
          ' aria-label="Watch ' + rkEsc(r.name || '') + '">' + rkStarSVG() + '</button>';

      row.querySelector('.rk-star').addEventListener('click', function(ev) {
        rkToggleWatch(ev, pid, r.name || '');
      });
      row.addEventListener('click', function() { rkOpenModal(r); });
      list.appendChild(row);
    });

    // 6. Update pagination controls
    rkUpdatePaginationControls(totalFiltered, totalPages);
  }

  // ── Modal ─────────────────────────────────────────────────────────────────────
  function rkLoadingHtml() {
    return '<div class="rk-detail-loading"><div class="loading-spinner" style="width:12px;height:12px;flex-shrink:0;"></div>Loading\u2026</div>';
  }

  function rkToggleMoreComps() {
    var d = document.getElementById('rkMoreComps');
    var btn = document.getElementById('rkMoreCompsBtn');
    if (!d || !btn) return;
    var open = d.style.display === 'none';
    d.style.display = open ? 'block' : 'none';
    btn.textContent = open ? 'Show fewer components' : 'Show all components';
  }

  function rkOpenModal(r) {
    var modal   = document.getElementById('rkModal');
    var content = document.getElementById('rkModalContent');
    var pid  = r.player_id || '';
    var name = r.name || 'Unknown';
    var pos  = (r.position || '').toUpperCase();
    var posColor = RK_POS_COLORS[pos] || '#94a3b8';
    var tc   = RK_TIER_COLORS[r.tier] || '#9ca3af';
    var score = parseFloat(r.prospect_score || 0);
    var conf  = parseFloat(r.confidence_score || 0);
    var age   = r.age != null ? parseFloat(r.age).toFixed(1) : '-';
    var mockPick = r.projected_pick != null ? Math.round(parseFloat(r.projected_pick)) : null;
    var on = !!rkWatch[pid];
    var rankNum = r.overall_rank ? '#' + r.overall_rank : '-';

    var reasons = String(r.key_reasons || '').split('\\n').filter(function(l){ return l.trim(); });
    var notesHtml = reasons.length
      ? '<div class="m-notes">' + reasons.map(function(l){ return '<div>&middot; ' + rkEsc(l.trim().replace(/^[·\\-•*]\\s*/, '')) + '</div>'; }).join('') + '</div>'
      : '<div class="m-notes">No scouting notes yet.</div>';

    var compDefs = [
      { label: 'Production',   val: r.production_score,              color: RK_COMP_COLORS[0], rawId: 'rkRawProd' },
      { label: 'Athleticism',  val: r.athleticism_score,             color: RK_COMP_COLORS[1], rawId: 'rkRawAth' },
      { label: 'Draft Capital', val: r.projected_draft_capital_score, color: RK_COMP_COLORS[2], rawId: 'rkRawDc' }
    ];
    var compsHtml = compDefs.map(function(c) {
      var v = rkScoreNum(c.val), disp = rkScore1(c.val);
      return '<div class="m-comp-row">' +
        '<div class="m-comp-label">' + c.label + '</div>' +
        '<div class="rk-meter-bar"><div style="width:' + v + '%;background:' + c.color + '"></div></div>' +
        '<div class="rk-meter-val" style="color:' + c.color + '">' + disp + '</div>' +
        '<div class="m-comp-raw" id="' + c.rawId + '"></div></div>';
    }).join('');

    content.innerHTML =
      '<div class="rk-modal-header"><div class="m-headwrap">' +
        rkHeadshot(r, 'm-headshot') +
        '<div style="min-width:0">' +
        '<div class="m-title-row">' +
          '<span class="m-name">' + rkEsc(name) + '</span>' +
          '<span class="m-tier" style="background:' + tc + '22;color:' + tc + ';border:1px solid ' + tc + '44;">T' + (r.tier || '?') + '</span>' +
          (r.early_declare ? '<span class="m-early">EARLY</span>' : '') +
          '<button class="rk-star' + (on ? ' watching' : '') + '" data-rk-star="1" aria-label="Watch ' + rkEsc(name) + '">' + rkStarSVG() + '</button>' +
        '</div>' +
        '<div class="m-meta"><span class="m-rank">' + rankNum + '</span>' + rkDelta(r.rank_delta) +
          '<span class="rk-pos" style="--pc:' + posColor + ';margin-left:0">' + rkEsc(pos) + '</span>' +
          (r.school ? '<span>' + rkEsc(r.school) + '</span>' : '') +
          '<span id="rkMetaConf"></span>' +
          (r.age != null ? '<span>&middot;</span><span>' + parseFloat(r.age).toFixed(1) + ' yrs</span>' : '') +
          '<span id="rkMetaStars"></span>' +
          (r.draft_class_year ? '<span>&middot;</span><span>' + r.draft_class_year + ' Draft</span>' : '') + '</div>' +
        (r.transfer_history ? '<div class="m-transfer">Transfer: ' + rkEsc(r.transfer_history) + '</div>' : '') +
        '</div>' +
      '</div><button class="rk-modal-close" onclick="rkCloseModal()" aria-label="Close">\u2715</button></div>' +
      '<div class="rk-modal-body">' +
        '<div class="rk-hero-row">' +
          '<div class="rk-hero-stat" style="background:' + tc + '14;border-color:transparent;">' +
            '<div class="rk-hero-label">Prospect Score</div>' +
            '<div class="rk-hero-val" style="color:' + tc + ';">' + rkScore1(score) + '</div>' +
            '<div class="rk-hero-sub">' + rkEsc(RK_TIER_NAMES[r.tier] || r.tier_label || '') + '</div></div>' +
          '<div class="rk-hero-stat"><div class="rk-hero-label">Breakout Age</div>' +
            '<div class="rk-hero-val" id="rkHeroBreakout">-</div><div class="rk-hero-sub">years</div></div>' +
          '<div class="rk-hero-stat"><div class="rk-hero-label">Mock Draft</div>' +
            '<div class="rk-hero-val">' + (mockPick != null ? 'Pick ' + mockPick : '-') + '</div>' +
            '<div class="rk-hero-sub" id="rkHeroTrend"></div></div>' +
        '</div>' +
        '<hr class="rk-section-divider">' +
        '<div class="m-sec-head"><div class="m-sec-label">Grade breakdown</div>' +
          '<div class="m-conf">Data confidence: <b>' + rkScore1(conf) + '</b></div></div>' +
        compsHtml +
        '<div id="rkMoreComps" style="display:none"></div>' +
        '<button class="m-more-btn" id="rkMoreCompsBtn" onclick="rkToggleMoreComps()">Show all components</button>' +
        '<hr class="rk-section-divider">' +
        '<div class="m-sec-label">Advanced metrics</div>' +
        '<div class="m-adv-grid"><div id="rkAdvBody">' + rkLoadingHtml() + '</div>' +
        '<div class="m-adv-radar" id="rkRadarBody"></div></div>' +
        '<div class="m-cols">' +
          '<div class="m-col">' +
            '<hr class="rk-section-divider">' +
            '<div class="m-sec-label">Combine &amp; measurables</div><div class="rk-meas-grid" id="rkCombineBody">' + rkLoadingHtml() + '</div>' +
          '</div>' +
          '<div class="m-col">' +
            '<hr class="rk-section-divider">' +
            '<div class="m-sec-label">College production</div><div id="rkSeasonsBody">' + rkLoadingHtml() + '</div>' +
          '</div>' +
          '<div class="m-col">' +
            '<hr class="rk-section-divider">' +
            '<div class="m-sec-label">Scouting notes</div>' + notesHtml +
          '</div>' +
          '<div class="m-col">' +
            '<hr class="rk-section-divider">' +
            '<div class="m-sec-label">Historical comparables</div>' +
            '<div id="rkComparablesBody">' + rkLoadingHtml() + '</div>' +
          '</div>' +
        '</div>' +
        '<div class="m-nudge' + (on ? ' on' : '') + '" data-rk-nudge="1">' +
          '<button class="rk-star' + (on ? ' watching' : '') + '" style="pointer-events:none" tabindex="-1">' + rkStarSVG() + '</button>' +
          '<span>' + (on ? 'You will be nudged if ' + rkEsc(name) + ' changes tiers.'
                          : 'Get nudged if ' + rkEsc(name) + ' changes tiers.') + '</span></div>' +
      '</div>';

    // Wire the star + nudge (direct binding, no inline handlers).
    var starBtn = content.querySelector('[data-rk-star]');
    if (starBtn) starBtn.addEventListener('click', function(ev) { rkToggleWatchModal(ev, pid, name); });
    var nudge = content.querySelector('[data-rk-nudge]');
    if (nudge) nudge.addEventListener('click', function(ev) { rkToggleWatchModal(ev, pid, name); });

    modal.style.display = 'flex';
    content.style.borderTop = '3px solid ' + tc;
    document.body.style.overflow = 'hidden';

    // Auto-link to Sleeper ID silently in background
    if (pid) {
      fetch('/api/prospects/auto-link/' + encodeURIComponent(pid)).catch(function(){});
    }

    // Detail bundle: advanced metrics, combine, seasons, extras
    var yearQ = r.draft_class_year || rkDraftYear || '';
    fetch('/api/prospects/player/' + encodeURIComponent(pid) + '?year=' + encodeURIComponent(yearQ))
      .then(function(res){ return res.json(); })
      .then(function(d) { rkFillDetail(r, d || {}); })
      .catch(function() {
        ['rkAdvBody', 'rkCombineBody', 'rkSeasonsBody'].forEach(function(id) {
          var el = document.getElementById(id);
          if (el) el.innerHTML = '<span style="font-size:13px;color:var(--text-muted);">Details unavailable.</span>';
        });
      });

    // Historical comparables from real past draft classes
    fetch('/api/prospects/comparables/' + encodeURIComponent(pid))
      .then(function(res){ return res.json(); })
      .then(function(cd) {
        var cb = document.getElementById('rkComparablesBody');
        if (!cb) return;
        var comps = cd.comparables || [];
        if (!comps.length) {
          cb.innerHTML = '<span style="font-size:13px;color:var(--text-muted);">No close historical comps found.</span>';
          return;
        }
        cb.innerHTML = comps.map(function(c) {
          var ctc = RK_TIER_COLORS[c.tier] || '#9ca3af';
          var meta = [c.draft_class_year,
                      c.actual_pick ? 'Pick ' + c.actual_pick : null,
                      c.school].filter(Boolean).join(' \u00b7 ');
          var discColor = rkDiscColor(c.player_id);
          var disc = '<div class="m-sim-disc" style="--pc:' + discColor + '">' + rkInitials(c.name) +
            (c.headshot_url
              ? '<img src="' + rkEsc(c.headshot_url) + '" alt="" loading="lazy" onerror="this.remove()"></div>'
              : '</div>');
          return '<div class="m-sim-row">' + disc +
            '<div><div class="m-sim-name">' + rkEsc(c.name) + '</div>' +
            '<div class="m-sim-meta">' + rkEsc(meta) + '</div></div>' +
            '<div class="m-sim-right"><span class="m-sim-score">' + rkScore1(c.prospect_score) + '</span>' +
            '<span class="m-sim-tier" style="background:' + ctc + '22;color:' + ctc + ';border:1px solid ' + ctc + '44;">T' + c.tier + '</span></div></div>';
        }).join('');
      })
      .catch(function() {
        var cb2 = document.getElementById('rkComparablesBody');
        if (cb2) cb2.innerHTML = '<span style="font-size:13px;color:var(--text-muted);">Could not load comparables.</span>';
      });
  }

  function rkFillDetail(r, d) {
    function fmtNum(v) {
      if (v == null || v === '') return '-';
      var n = Number(v);
      if (isNaN(n)) return '-';
      return n >= 1000 ? n.toLocaleString() : String(Math.round(n));
    }
    // Conference from the latest season with one recorded
    var mc = document.getElementById('rkMetaConf');
    if (mc) {
      var conf = null;
      (d.seasons || []).forEach(function(s) {
        if (s.conference) conf = s.conference;
      });
      mc.innerHTML = conf ? '<span>&middot;</span><span>' + rkEsc(conf) + '</span>' : '';
    }
    // Composite stars + recruiting details (parallel data track; hidden until present)
    var ms = document.getElementById('rkMetaStars');
    if (ms) {
      var st = (d.composite_stars != null && d.composite_stars !== '')
        ? parseInt(d.composite_stars, 10) : null;
      if (st == null && r.recruit_stars != null && r.recruit_stars !== '')
        st = parseInt(r.recruit_stars, 10);
      if (st) {
        var det = '';
        var posStr = (r.position || '').toUpperCase();
        if (r.recruit_composite_rating != null)
          det += Number(r.recruit_composite_rating).toFixed(4);
        if (r.recruit_national_rank != null)
          det += (det ? ', ' : '') + '#' + r.recruit_national_rank + " nat'l";
        if (r.recruit_position_rank != null && posStr)
          det += (det ? ', ' : '') + '#' + r.recruit_position_rank + ' ' + posStr;
        ms.innerHTML = '<span>&middot;</span><span>' + st + '-star' +
          (det ? ' (' + rkEsc(det) + ')' : '') + '</span>';
      } else {
        ms.innerHTML = '';
      }
    }
    // Breakout age hero
    var hb = document.getElementById('rkHeroBreakout');
    if (hb) {
      var ba = d.breakout_age != null ? Number(d.breakout_age).toFixed(1) : null;
      hb.textContent = ba || '-';
      var baSub = hb.parentElement.querySelector('.rk-hero-sub');
      if (baSub) baSub.style.display = ba ? '' : 'none';
    }
    // Mock draft sub: round + positional rank + month trend
    var ht = document.getElementById('rkHeroTrend');
    if (ht) {
      var bits = [];
      var posStr2 = (r.position || '').toUpperCase();
      if (r.projected_round != null && r.projected_round !== '')
        bits.push('Rd ' + r.projected_round);
      if (r.position_rank != null && r.position_rank !== '' && posStr2)
        bits.push(posStr2 + r.position_rank);
      var t = d.mock_trend, trendHtml = '';
      if (t != null && t !== '') {
        if (t > 0) trendHtml = '<span class="dvt-change dvt-change-up">&#9650;' + t + ' this month</span>';
        else if (t < 0) trendHtml = '<span class="dvt-change dvt-change-down">&#9660;' + Math.abs(t) + ' this month</span>';
        else trendHtml = '<span class="dvt-change dvt-change-flat">no move this month</span>';
      }
      ht.innerHTML = rkEsc(bits.join(' · ')) + (trendHtml ? (bits.length ? ' &middot; ' : '') + trendHtml : '');
    }
    // Raw context lines under the three main components
    function advRaw(label) {
      var found = null;
      (d.advanced || []).forEach(function(a){ if (a.label === label) found = a; });
      return found && found.raw ? found.raw : null;
    }
    function setRaw(id, txt) {
      var el = document.getElementById(id);
      if (el && txt) el.textContent = txt;
    }
    var domRaw = advRaw('Dominator');
    setRaw('rkRawProd', domRaw ? domRaw + ' dominator' : null);
    var ath = d.athleticism || {};
    setRaw('rkRawAth', ath.forty_yard != null ? Number(ath.forty_yard).toFixed(2) + ' forty'
      : (ath.ras_score != null ? Number(ath.ras_score).toFixed(1) + ' RAS' : null));
    setRaw('rkRawDc', r.projected_pick != null ? 'mock pick ' + Math.round(parseFloat(r.projected_pick)) : null);

    // "Show all components" expander: utilization / efficiency / durability /
    // experience / competition (competition carries the SP+ SOS context)
    var more = document.getElementById('rkMoreComps');
    if (more) {
      var sosRaw = d.sp_sos != null ? 'SP+ SOS ' + Number(d.sp_sos).toFixed(2) : '';
      var vals = [d.utilization_score, r.efficiency_score, r.durability_score,
                  d.experience_score, r.competition_score];
      var raws = ['', '', '', '', sosRaw];
      more.innerHTML = vals.map(function(v, k) {
        var n = rkScoreNum(v), disp = rkScore1(v);
        var dd = RK_XCOMP[k];
        return '<div class="m-comp-row">' +
          '<div class="m-comp-label">' + dd[0] + '</div>' +
          '<div class="rk-meter-bar"><div style="width:' + n + '%;background:' + dd[1] + '"></div></div>' +
          '<div class="rk-meter-val" style="color:' + dd[1] + '">' + disp + '</div>' +
          '<div class="m-comp-raw">' + rkEsc(raws[k]) + '</div></div>';
      }).join('');
    }

    // Advanced metrics rows + radar
    var advBody = document.getElementById('rkAdvBody');
    var radarBody = document.getElementById('rkRadarBody');
    var adv = (d.advanced || []).filter(function(a){ return a.raw != null && a.grade != null; });
    if (advBody) {
      advBody.innerHTML = adv.length ? adv.map(function(a) {
        var g = rkLetterGrade(a.grade);
        return '<div class="m-adv-row">' +
          '<div class="m-adv-label">' + rkEsc(a.label) + '</div>' +
          '<div class="rk-meter-bar"><div style="width:' + a.grade + '%;background:' + g[1] + '"></div></div>' +
          '<div class="m-adv-score">' + rkEsc(a.raw) + '</div>' +
          '<div class="m-adv-grade" style="color:' + g[1] + '">' + g[0] + '</div></div>';
      }).join('') : '<span style="font-size:13px;color:var(--text-muted);">No advanced metrics available.</span>';
    }
    if (radarBody) {
      var tc = RK_TIER_COLORS[r.tier] || '#9ca3af';
      radarBody.innerHTML = adv.length
        ? rkRadarSVG(adv.map(function(a){ return a.label; }), adv.map(function(a){ return a.grade; }), tc) : '';
    }

    // Combine grid
    var cb = document.getElementById('rkCombineBody');
    if (cb) {
      var bi = parseInt(ath.broad_jump_in, 10);
      var broad = isNaN(bi) ? '-' : Math.floor(bi / 12) + "'" + (bi % 12) + '"';
      var cells = [
        ['Height', rkHeightStr(r) || '-'],
        ['Weight', rkWeightStr(r) || '-'],
        ['40 Dash', ath.forty_yard != null ? Number(ath.forty_yard).toFixed(2) + 's' : '-'],
        ['Vertical', ath.vertical_inches != null ? Number(ath.vertical_inches).toFixed(0) + '"' : '-'],
        ['Broad Jump', broad],
        ['3-Cone', ath.three_cone != null ? Number(ath.three_cone).toFixed(2) : '-'],
        ['Shuttle', ath.short_shuttle != null ? Number(ath.short_shuttle).toFixed(2) : '-'],
        ['Bench', ath.bench_reps != null ? String(ath.bench_reps) : '-'],
        ['RAS', ath.ras_score != null ? Number(ath.ras_score).toFixed(1) : '-']
      ];
      cb.innerHTML = cells.map(function(c) {
        return '<div class="rk-meas-cell"><div class="rk-meas-label">' + c[0] + '</div>' +
               '<div class="rk-meas-val">' + rkEsc(c[1]) + '</div></div>';
      }).join('');
    }

    // Season-by-season production table (position-correct columns)
    var sb = document.getElementById('rkSeasonsBody');
    if (sb) {
      var pos = (r.position || '').toUpperCase();
      var cols = RK_SEASON_COLS[pos] || RK_SEASON_COLS.WR;
      var seasons = d.seasons || [];
      if (!seasons.length) {
        sb.innerHTML = '<span style="font-size:13px;color:var(--text-muted);">No season data available.</span>';
      } else {
        var rowsHtml = seasons.map(function(s) {
          var cells;
          if (pos === 'QB') {
            cells = [s.season, s.games_played,
              s.completion_pct != null ? Number(s.completion_pct).toFixed(1) + '%' : '-',
              fmtNum(s.pass_yards),
              s.yds_per_attempt != null ? Number(s.yds_per_attempt).toFixed(1) : '-',
              s.pass_tds != null ? s.pass_tds : '-', s.interceptions != null ? s.interceptions : '-',
              rkDomCell(s)];
          } else if (pos === 'RB') {
            cells = [s.season, s.games_played, fmtNum(s.rush_attempts), fmtNum(s.rush_yards),
                     s.yds_per_carry != null ? Number(s.yds_per_carry).toFixed(1) : '-',
                     s.rush_tds != null ? s.rush_tds : '-', rkDomCell(s)];
          } else {
            cells = [s.season, s.games_played, fmtNum(s.receptions), fmtNum(s.receiving_yards),
                     s.yds_per_reception != null ? Number(s.yds_per_reception).toFixed(1) : '-',
                     s.receiving_tds != null ? s.receiving_tds : '-', rkDomCell(s)];
          }
          return '<tr>' + cells.map(function(v, k) {
            var disp = (v == null || v === '') ? '-' : v;
            return k === 0 ? '<td class="yr">' + rkEsc(String(disp)) + '</td>'
                           : '<td>' + rkEsc(String(disp)) + '</td>';
          }).join('') + '</tr>';
        }).join('');
        // Career totals row (counting stats summed, efficiency re-weighted)
        (function() {
          var t = {gp:0, att:0, yds:0, td:0, cmp:0, patt:0, pint:0, rec:0, ry:0, dom:0, domN:0};
          seasons.forEach(function(s) {
            t.gp += Number(s.games_played) || 0;
            t.att += Number(s.rush_attempts) || 0;
            t.yds += Number(s.rush_yards) || 0;
            t.td += (Number(s.rush_tds) || 0) + (Number(s.receiving_tds) || 0);
            t.rec += Number(s.receptions) || 0;
            t.ry += Number(s.receiving_yards) || 0;
            t.cmp += Number(s.completions) || 0;
            t.patt += Number(s.pass_attempts) || 0;
            t.pint += Number(s.interceptions) || 0;
            t.py = (t.py || 0) + (Number(s.pass_yards) || 0);
            t.ptd = (t.ptd || 0) + (Number(s.pass_tds) || 0);
            if (s.dominator_rating != null) { t.dom += Number(s.dominator_rating); t.domN++; }
          });
          var tcells, avgDom = t.domN ? Math.round(t.dom / t.domN * 100) + '%' : '-';
          if (pos === 'QB') {
            tcells = ['Career', t.gp,
              t.patt ? (100 * t.cmp / t.patt).toFixed(1) + '%' : '-',
              fmtNum(t.py), t.patt ? (t.py / t.patt).toFixed(1) : '-',
              t.ptd, t.pint, avgDom];
          } else if (pos === 'RB') {
            tcells = ['Career', t.gp, fmtNum(t.att), fmtNum(t.yds),
              t.att ? (t.yds / t.att).toFixed(1) : '-', t.td, avgDom];
          } else {
            var rtd = 0;
            seasons.forEach(function(s){ rtd += Number(s.receiving_tds) || 0; });
            tcells = ['Career', t.gp, fmtNum(t.rec), fmtNum(t.ry),
              t.rec ? (t.ry / t.rec).toFixed(1) : '-', rtd, avgDom];
          }
          rowsHtml += '<tr class="rk-career-row">' + tcells.map(function(v, k) {
            var disp = (v == null || v === '') ? '-' : v;
            return k === 0 ? '<td class="yr">' + rkEsc(String(disp)) + '</td>'
                           : '<td>' + rkEsc(String(disp)) + '</td>';
          }).join('') + '</tr>';
        })();
        sb.innerHTML = '<table class="m-season-table"><thead><tr>' +
          cols.map(function(c){ return '<th>' + c + '</th>'; }).join('') +
          '</tr></thead><tbody>' + rowsHtml + '</tbody></table>';
      }
    }
  }

  function rkCloseModal() {
    document.getElementById('rkModal').style.display = 'none';
    document.body.style.overflow = '';
  }

  // Close on backdrop click
  document.getElementById('rkModal').addEventListener('click', function(e) {
    if (e.target === this) rkCloseModal();
  });

  // ── Load data ──────────────────────────────────────────────────────────────
  (function() {
    var inp = document.getElementById('rookieSearch');
    var clr = document.getElementById('rookieSearchClear');
    inp.addEventListener('input', function() {
      rkSearch = inp.value.trim();
      clr.style.display = rkSearch.length > 0 ? 'block' : 'none';
      rkCurrentPage = 1;
      rkRender();
    });
  })();

  // Clickable sort headers (AM-table pattern): click a column to sort by it,
  // click again to flip direction. Bound once; targets resolve at event time
  // so soft-nav re-renders keep working.
  if (!window.__rkSortBound) {
    window.__rkSortBound = true;
    (function() {
      function rkHeaderCell(e) {
        var cell = e.target && e.target.closest ? e.target.closest('#rkHeader [data-rk-sort-col]') : null;
        return cell || null;
      }
      function rkHeaderKey(cell) {
        var col = cell.getAttribute('data-rk-sort-col');
        if (col === 'name') return 'name';
        if (col === 'rank') return 'rank';
        if (col === 'score') return 'score';
        if (col === 'mock') return 'mock';
        return rkSortKey;
      }
      document.addEventListener('click', function(e) {
        var cell = rkHeaderCell(e);
        if (cell) rkHeaderSort(rkHeaderKey(cell));
      });
      document.addEventListener('keydown', function(e) {
        if (e.key !== 'Enter' && e.key !== ' ') return;
        var cell = rkHeaderCell(e);
        if (!cell) return;
        e.preventDefault();
        rkHeaderSort(rkHeaderKey(cell));
      });
    })();
  }

  fetch('/api/prospects/active-class')
    .then(function(r){ return r.json(); })
    .then(function(d) {
      rkDraftYear = d.draft_class_year || new Date().getFullYear();
      document.getElementById('rookiesTitle').textContent = rkDraftYear + ' Prospect Rankings';
      
      // Fetch draft status
      return fetch('/api/prospects/draft-status?year=' + rkDraftYear);
    })
    .then(function(r){ return r.json(); })
    .then(function(status) {
      rkDraftComplete = status.draft_complete || false;
      return fetch('/api/prospects/rankings?year=' + rkDraftYear);
    })
    .then(function(r){ return r.json(); })
    .then(function(data) {
      document.getElementById('rkLoading').style.display = 'none';
      rkAllPlayers = data.rankings || [];
      rkLoaded = true;
      rkPipelinePaused = !!data.paused;
      var pausedBanner = document.getElementById('rkPausedBanner');
      if (pausedBanner) {
        if (rkPipelinePaused && rkAllPlayers.length) {
          pausedBanner.style.display = '';
          var asOf = document.getElementById('rkPausedAsOf');
          if (asOf && data.last_updated) asOf.textContent = '(as of ' + data.last_updated + ')';
        } else {
          pausedBanner.style.display = 'none';
        }
      }
      rkRender();
    })
    .catch(function(err) {
      console.error('[rookies] Load error:', err);
      document.getElementById('rkLoading').innerHTML =
        '<div style="color:var(--loss);">Failed to load rookie data. Please refresh.</div>';
    });

  function rkLoadPage(page) {
    if (page === 'prev') page = rkCurrentPage - 1;
    else if (page === 'next') page = rkCurrentPage + 1;
    if (page < 1) return;
    rkCurrentPage = page;
    rkRender();
    window.scrollTo({top: 0, behavior: 'smooth'});
  }

  function rkUpdatePaginationControls(totalFiltered, totalPages) {
    var prevBtn      = document.getElementById('rkPrevBtn');
    var nextBtn      = document.getElementById('rkNextBtn');
    var pageNumbers  = document.getElementById('rkPageNumbers');
    var paginationText = document.getElementById('rkPaginationText');
    var pagination   = document.getElementById('rkPagination');

    if (totalPages <= 1) {
      pagination.style.display = 'none';
      return;
    }

    prevBtn.disabled = rkCurrentPage <= 1;
    nextBtn.disabled = rkCurrentPage >= totalPages;

    var start = (rkCurrentPage - 1) * RK_PER_PAGE + 1;
    var end   = Math.min(rkCurrentPage * RK_PER_PAGE, totalFiltered);
    paginationText.textContent = 'Showing ' + start + '–' + end + ' of ' + totalFiltered + ' prospects';

    pageNumbers.innerHTML = '';
    var maxPages  = 5;
    var startPage = Math.max(1, rkCurrentPage - Math.floor(maxPages / 2));
    var endPage   = Math.min(totalPages, startPage + maxPages - 1);
    if (endPage - startPage < maxPages - 1) startPage = Math.max(1, endPage - maxPages + 1);

    for (var i = startPage; i <= endPage; i++) {
      (function(pageNum) {
        var btn = document.createElement('button');
        btn.className = 'pagination-page' + (pageNum === rkCurrentPage ? ' active' : '');
        btn.textContent = pageNum;
        btn.onclick = function() { rkLoadPage(pageNum); };
        pageNumbers.appendChild(btn);
      })(i);
    }

    pagination.style.display = 'flex';
  }
</script>
"""
