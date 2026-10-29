"""
Rookies page HTML builder.

Returns the full body HTML for the /<platform>/<season>/<league_id>/rookies route.
Data is loaded client-side from /api/prospects/rankings so the page is fast and
the filters/sorts are instant without round-trips.

Display model (2026-10-17): the board and modal show NO numeric grade.
They show rank (ordered by the internal grade), a band tag, and component
percentiles computed within position against the graded reference
population. The board uses the native players-board grammar (card shell,
two-row filter bar, pr-grid rows, tier dividers); the modal follows the
approved native mock set (band-tinted hero, percentile panel with raw
stat sub-lines, class-context strip, season lines, testing table).
"""
from __future__ import annotations


def build_prospects_body(is_admin: bool = False) -> str:
    admin_flag = "true" if is_admin else "false"
    return f"""<script>window.RK_IS_ADMIN = {admin_flag};</script>""" + """
<div class="card central" id="prospectsCard">
  <div class="card-header" style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
    <div>
      <h1 id="rookiesTitle">Rookie Prospects</h1>
      <div style="font-size: 15px; color: var(--text-muted); margin-top: 4px;">
        <span id="rkPoolLine">Dynasty prospect rankings - production, athleticism, and draft capital combined</span>
      </div>
      <div id="rkPausedBanner" style="display:none;margin-top:10px;padding:8px 12px;border-radius:8px;
           border:1px solid color-mix(in srgb,#b45309 35%,var(--border));background:color-mix(in srgb,#b45309 12%,transparent);
           font-size:13px;color:var(--text);">
        Prospect rankings are paused until the next draft cycle. Showing the last published board
        <span id="rkPausedAsOf"></span>.
      </div>
    </div>
    <button type="button" class="btn btn-ghost btn-sm" onclick="rkExportCSV()" title="Export CSV" aria-label="Export CSV"><img src="/static/images/download-solid.png" alt="" style="width:14px;height:14px;vertical-align:-2px;opacity:.75;"> Export</button>
  </div>

  <!-- Rankings panel -->
  <div id="rk-panel-rankings" class="card-body" style="padding-top:0;">

    <!-- Controls -->
    <div class="filter-controls-container">
      <!-- Row 1: Search + pills -->
      <div class="filter-row filter-row-primary">
        <div class="filter-search">
          <input id="rookieSearch" type="text" placeholder="Search prospects…" autocomplete="off">
          <span class="mag"><i class="fa-solid fa-magnifying-glass" aria-hidden="true"></i></span>
          <button id="rookieSearchClear" onclick="rkClearSearch()" aria-label="Clear search"
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
      <div class="filter-row filter-row-secondary">
        <div class="active-settings-indicator" id="rkActiveSettings" style="display:none">
          <span class="active-setting-tag" id="rkClassTag"></span>
          <span class="active-setting-tag" id="rkSeasonTag" style="display:none">In-Season</span>
        </div>
        <div class="filter-sort-group">
          <div class="filter-sort">
            <label class="filter-label" for="rkSort">Sort</label>
            <select id="rkSort" onchange="rkSetSortKey(this.value)">
              <option value="rank">Rank</option>
              <option value="prod">Production %</option>
              <option value="eff">Efficiency %</option>
              <option value="brk">Breakout %</option>
              <option value="name">Name</option>
            </select>
          </div>
        </div>
      </div>
    </div>

    <!-- Count -->
    <div id="rkCount" style="display:none;"></div>

    <!-- Column header (native board grammar; shown by rkRender) -->
    <div id="rkTableHeader" class="pr-grid-row" style="display:none">
      <span>#</span>
      <span data-pr-sort-col="player">Player</span>
      <span class="th-r" id="rkSchoolHeader">School</span>
      <span class="th-r" data-pr-sort-col="prod">Prod</span>
      <span class="th-r" id="rkEffHeader" data-pr-sort-col="eff">Eff</span>
      <span class="th-r" id="rkBrkHeader" data-pr-sort-col="brk">Brk</span>
      <span class="th-r">Tag</span>
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

  </div><!-- end rk-panel-rankings -->

</div>

<!-- Prospect detail modal -->
<div id="rkModal" style="display:none;position:fixed;inset:0;z-index:10500;
     display:none;align-items:center;justify-content:center;padding:20px;
     background:rgba(15,23,42,0.7);backdrop-filter:blur(4px);">
  <div id="rkModalContent"
    style="background:var(--card);border-radius:16px;max-width:900px;width:100%;
           max-height:90vh;display:flex;flex-direction:column;overflow:hidden;
           box-shadow:0 24px 48px rgba(15,23,42,0.25);">
    <!-- filled by JS -->
  </div>
</div>
""" + """
<style>
  /* ===== Card: native board geometry (players-board grammar) ===== */
  #prospectsCard { max-width: 1180px; overflow: visible; }
  /* Real h1 (was h2) with the exact specified values the h2 rendered with,
     so the swap changes semantics only, not the layout. */
  #rookiesTitle { font-size: 18px; font-weight: 700; margin: 0; text-align: left; }
  /* The global h1::after is a 300px centered sky bar meant for standalone
     hero titles; inside this card header it renders as a stray floating
     bar between the title and the subtitle. Suppress it (same opt-out as
     .wl-page-title / .compare-page-title in dashboard.css). */
  #rookiesTitle::after { display: none; }
  /* Export is the only .btn on this page and dashboard.css defines no
     .btn/.btn-ghost/.btn-sm at all, so it fell back to the UA button
     padding (1px 6px, 22px tall) in the global accent fill. Give it real
     ghost-button geometry, scoped to this card. */
  #prospectsCard .btn-ghost { display: inline-flex; align-items: center; gap: 6px; padding: 7px 12px; min-height: 32px; background: transparent; border: 1px solid var(--border); border-radius: 8px; color: var(--text-muted); font-size: 13px; font-weight: 600; line-height: 1; white-space: nowrap; }
  #prospectsCard .btn-ghost:hover { color: var(--text); border-color: var(--accent); }

  /* ===== Filter controls: players-board two-row bar ===== */
  .filter-controls-container { display: flex; flex-direction: column; gap: 10px; padding: 14px 0 12px; border-bottom: 1px solid var(--border); margin-bottom: 10px; }
  .filter-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; justify-content: space-between; }
  .filter-row-primary { gap: 12px; }
  .filter-row-secondary { padding-top: 2px; flex-wrap: nowrap; align-items: center; }
  .filter-search { position: relative; flex: 1; min-width: 200px; }
  .filter-search input { width: 100%; height: 36px; padding: 8px 32px 8px 34px; border-radius: 8px; border: 1px solid var(--border); background: var(--card-bg); color: var(--text); font-size: 13px; outline: none; box-sizing: border-box; font-family: inherit; }
  .filter-search .mag { position: absolute; left: 10px; top: 50%; transform: translateY(-50%); color: var(--text-muted); font-size: 13px; pointer-events: none; }
  .filter-positions { display: flex; gap: 6px; flex-wrap: wrap; }
  .active-settings-indicator { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; flex: 1; min-width: 0; }
  .active-setting-tag { display: inline-flex; align-items: center; height: 24px; padding: 0 10px; line-height: 1; border-radius: var(--radius-pill, 8px); background: var(--accent-soft); color: var(--accent); font-size: 11px; font-weight: 600; white-space: nowrap; }
  .filter-sort-group { display: flex; align-items: center; gap: 10px; flex-shrink: 0; }
  .filter-sort { display: flex; align-items: center; gap: 8px; }
  .filter-label { font-size: 11px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: .04em; }
  .filter-sort select { padding: 7px 30px 7px 11px; border-radius: 9px; border: 1px solid var(--border); background: var(--card-bg); color: var(--text); font-size: 13px; font-weight: 600; cursor: pointer; outline: none; min-height: 36px; min-width: 128px; font-family: inherit; -webkit-appearance: none; appearance: none; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 11px center; }
  .filter-sort select:hover { border-color: var(--accent); }

  /* ===== Board: pr-grid rows ===== */
  #rkCount { font-size: 13px; color: var(--text-muted); margin: 0 0 10px; }
  .pr-grid-row { display: grid; grid-template-columns: 88px minmax(0,1fr) 168px 60px 60px 60px 140px; align-items: center; column-gap: 14px; }
  #rkTableHeader { display: grid; padding: 7px 12px; border-radius: 6px; background: var(--accent-soft); font-size: 11px; font-weight: 700; color: var(--accent); letter-spacing: .04em; text-transform: uppercase; position: sticky; top: 60px; z-index: 20; box-shadow: 0 -8px 0 0 var(--card), 0 10px 14px -12px rgba(0,0,0,.4); margin-bottom: 4px; }
  #rkTableHeader [data-pr-sort-col] { cursor: pointer; user-select: none; }
  #rkTableHeader [data-pr-sort-col]:hover { color: var(--text); }
  #rkTableHeader .th-r { text-align: right; }
  .pr-player-row { padding: 10px 12px; cursor: pointer; transition: background .12s ease; }
  .pr-player-row:hover { background: color-mix(in srgb, var(--accent-soft) 70%, transparent); }
  .pr-player-row + .pr-player-row { border-top: 1px solid color-mix(in srgb, var(--border) 55%, transparent); }
  .pr-player-row:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  .pr-rank { display: flex; align-items: baseline; gap: 6px; justify-content: flex-start; white-space: nowrap; padding-left: 2px; }
  .pr-rank-num { font-size: 15px; font-weight: 800; line-height: 1; color: var(--text); font-variant-numeric: tabular-nums; }
  .pr-rank-num.top { color: var(--gold); }
  .pr-rank-delta { font-size: 11px; font-weight: 800; line-height: 1; white-space: nowrap; }
  .pr-rank-delta.up { color: var(--win); }
  .pr-rank-delta.down { color: var(--loss); }
  .pr-name { font-size: 13px; font-weight: 600; color: var(--text); display: flex; align-items: baseline; gap: 6px; flex-wrap: wrap; min-width: 0; }
  .pr-nm { min-width: 0; line-height: 1.3; }
  /* Player-name hover contract (site-wide): names dim, never underline. */
  .pr-nm:hover { opacity: 0.72; }
  .pr-wl-star { color: var(--gold); font-size: 12px; line-height: 1; margin-left: 5px; }
  .pr-meta { display: none; }
  .pr-meta-txt b { color: var(--text); font-weight: 600; }
  .pr-school { text-align: right; font-size: 11px; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .pr-school b { color: var(--text); font-weight: 600; }
  .pr-sep { margin: 0 7px; color: var(--text-subtle); font-weight: 400; }
  .pr-stat { text-align: right; }
  .pr-stat-val { display: block; font-size: 14px; font-weight: 700; line-height: 1.2; font-variant-numeric: tabular-nums; color: var(--text); }
  .pr-stat-val.sv-lo { color: var(--text-muted); font-weight: 600; }
  .rk-scbar { display: block; height: 3px; background: color-mix(in srgb, var(--text-muted) 14%, transparent); border-radius: 999px; margin-top: 4px; overflow: hidden; }
  .rk-scbar i { display: block; height: 100%; border-radius: 999px; opacity: .92; }
  .pr-tagcell { text-align: right; }
  .pr-tier-divider { display: flex; align-items: center; gap: 8px; padding: 7px 12px 5px; pointer-events: none; }
  .pr-tier-divider-line { flex: 1; height: 1px; opacity: .35; }
  .pr-tier-divider-label { display: inline-block; font-size: 11px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; padding: 2px 7px; border-radius: var(--radius-sm); line-height: 1.3; white-space: nowrap; }
  .td-elite { background: rgba(251,191,36,.15); color: var(--gold); }
  .td-great { background: rgba(16,185,129,.15); color: #34d399; }
  .td-avg { background: rgba(148,163,184,.12); color: #94a3b8; }
  .td-day3 { background: rgba(100,116,139,.15); color: #94a3b8; }

  /* Position chip: tint in the position color */
  .rk-pos { display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 5px; background: color-mix(in srgb, var(--pc, var(--accent)) 16%, transparent); color: var(--pc, var(--accent)); vertical-align: middle; }
  /* Tag chips: one geometry, equal tint/border alpha across the bands */
  .m-tier { display: inline-block; padding: 4px 9px; border-radius: 6px; font-size: 11px; font-weight: 700; line-height: 1.25; white-space: nowrap; text-align: center; min-width: 118px; }
  .tag-chip { display: inline-flex; align-items: center; white-space: nowrap; font-size: 11px; font-weight: 700; letter-spacing: .02em; padding: 4px 9px; border-radius: var(--radius-pill); border: 1px solid var(--border); color: var(--text); }
  .tag-chip.tag-mini { font-size: 10px; padding: 2px 7px; }
  .tag-elite { background: rgba(251,191,36,.14); color: var(--gold); border: 1px solid rgba(251,191,36,.30); }
  .tag-great { background: rgba(16,185,129,.14); color: #34d399; border: 1px solid rgba(16,185,129,.30); }
  .tag-average { background: rgba(148,163,184,.14); color: #94a3b8; border: 1px solid rgba(148,163,184,.30); }
  .tag-day3 { background: rgba(100,116,139,.14); color: #94a3b8; border: 1px solid rgba(100,116,139,.30); }
  .m-tier.tag-elite, .m-tier.tag-great, .m-tier.tag-average, .m-tier.tag-day3 { border-width: 1px; }
  /* In-season chip: quieter wash than the position chip */
  .m-early { display: inline-block; padding: 2px 6px; border-radius: 5px; font-size: 10px; font-weight: 600; background: color-mix(in srgb, var(--text-muted) 10%, transparent); color: var(--text-muted); white-space: nowrap; }
  .state-chip { display: inline-block; padding: 3px 7px; border-radius: 5px; font-size: 10px; font-weight: 600; background: color-mix(in srgb, var(--text-muted) 10%, transparent); color: var(--text-muted); white-space: nowrap; }

  @media (max-width: 1100px) {
    #prospectsCard .pr-grid-row { grid-template-columns: 84px minmax(0,1fr) 156px 56px 56px 56px 132px; column-gap: 12px; }
  }
  @media (max-width: 768px) {
    /* GUI audit Fix 3: 11px floor for board tag/chip labels on mobile
       (position chip and In-Season chip were 10px at this width). */
    .rk-pos, .m-early { font-size: 11px; }
    .filter-row-primary { flex-direction: column; align-items: stretch; }
    .filter-search { max-width: 100%; }
    .filter-positions { justify-content: flex-start; gap: 6px; }
    .filter-row-secondary { flex-wrap: wrap; gap: 8px; }
    .active-settings-indicator { justify-content: center; order: -1; width: 100%; }
    .filter-sort-group { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(0,1fr); gap: 10px; width: 100%; }
    .filter-sort { min-width: 0; flex-direction: column; align-items: stretch; gap: 5px; }
    .filter-sort select { width: 100%; min-width: 0; }
    /* board: school and breakout columns yield first, like Age on the players board
       (scoped to #prospectsCard: dashboard.css carries global players-board
       .pr-grid-row phone rules that must not leak into this board) */
    #prospectsCard .pr-grid-row { grid-template-columns: 72px minmax(0,1fr) 54px 54px 130px !important; column-gap: 10px; }
    .pr-school, .pr-brk, #rkSchoolHeader, #rkBrkHeader { display: none !important; }
  }
  @media (max-width: 600px) {
    .filter-row-primary { display: grid; grid-template-columns: 1fr auto; grid-template-rows: auto auto; gap: 8px; }
    .filter-search { grid-column: 1 / -1; min-width: 0; }
    .filter-positions { flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; min-width: 0; align-items: center; }
    .filter-positions::-webkit-scrollbar { display: none; }
    #prospectsCard .pr-grid-row { grid-template-columns: 56px minmax(0,1fr) 46px 46px 120px !important; column-gap: 9px; }
    .pr-player-row { grid-template-areas: "rank name prod eff tag" "rank meta meta meta tag"; grid-template-rows: auto auto; row-gap: 3px; }
    .pr-name { display: contents; }
    .pr-rank { grid-area: rank; }
    .pr-nm { grid-area: name; align-self: center; }
    .pr-meta { grid-area: meta; }
    .pr-player-row > .pr-stat:not(.pr-eff):not(.pr-brk) { grid-area: prod; }
    .pr-eff { grid-area: eff; }
    .pr-tagcell { grid-area: tag; }
    .pr-player-row { padding: 11px 10px; }
    .pr-tier-divider { padding: 7px 10px 5px; }
    #rkTableHeader { padding: 7px 10px; }
    .pr-name { flex-direction: column; align-items: flex-start; flex-wrap: nowrap; gap: 4px; }
    .pr-nm { max-width: 100%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .pr-name > .rk-pos { display: none; }
    .m-early { display: none; }
    .pr-meta { display: inline-flex; align-items: center; gap: 6px; line-height: 1; max-width: 100%; font-size: 10.5px; font-weight: 500; color: var(--text-muted); }
    .pr-meta .rk-pos { font-size: 11px; padding: 1px 5px; }
    .pr-meta-txt { min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .m-tier { width: 116px; min-width: 116px; }
  }
  @media (max-width: 480px) {
    /* phone: rank | player | production | tag, like rank | name | ppg | value on players */
    #prospectsCard .pr-grid-row { grid-template-columns: 39px minmax(0,1fr) 33px 92px !important; column-gap: 8px; }
    .pr-player-row { grid-template-areas: "rank name prod tag" "rank meta meta tag"; }
    .pr-eff, #rkEffHeader { display: none !important; }
    .pr-rank-num { font-size: 14px; }
    .pr-rank { gap: 4px; padding-left: 0; }
    .pr-name { font-size: 12px; }
    .pr-wl-star { font-size: 11px; margin-left: 4px; }
    /* 11px floor (was 9.5px); width goes auto so the longest label
       ("Average Prospect") grows the pill instead of clipping. */
    .m-tier { width: auto; min-width: 92px; font-size: 11px; padding: 3px 3px; }
    .pr-meta { font-size: 10px; }
    .pr-meta .pr-sep { margin: 0 5px; }
    .pr-player-row { padding: 11px 8px; }
    #rkTableHeader { padding: 7px 8px; }
    .pr-tier-divider { padding: 7px 8px 5px; }
  }

  /* ===== Modal ===== */
  /* Band tint: one variable drives the top border and the Tag card. */
  #rkModalContent { --band: #94a3b8; --band-ink: #94a3b8; border-top: 3px solid var(--band); }
  #rkModalContent.rk-band-elite { --band: #fbbf24; --band-ink: #fbbf24; }
  #rkModalContent.rk-band-great { --band: #34d399; --band-ink: #34d399; }
  #rkModalContent.rk-band-average { --band: #94a3b8; --band-ink: #94a3b8; }
  #rkModalContent.rk-band-day3 { --band: #64748b; --band-ink: #94a3b8; }
  #rkModalContent .pm-hero-primary { background: color-mix(in srgb, var(--band) 10%, var(--card)); border-color: color-mix(in srgb, var(--band) 38%, transparent); }
  #rkModalContent .pm-hero-primary .pm-hero-label { color: var(--band-ink); }
  #rkModalContent .rk-hero-tagval { font-size: 19px; font-weight: 800; line-height: 1.15; color: var(--text); }
  #rkModalContent .rk-hero-receipt { white-space: normal; overflow: visible; font-size: 12px; line-height: 1.4; }
  #rkModalContent .tag-chip, #rkModalContent .state-chip { font-size: 11px; }

  /* Player-modal-style prospect modal: initials tile for the headshot slot
     when a prospect has no photo (mirrors the player modal fallback). */
  .rk-hs-initials { width: 160px; height: 135px; background: var(--accent-soft); color: var(--accent);
    display: flex; align-items: center; justify-content: center; font-size: 40px; font-weight: 800; letter-spacing: .02em; }

  .m-sec-hint { font-size: 11px; color: var(--text-subtle); margin-left: auto; }

  /* Class context strip */
  .ctx-strip { display: flex; flex-wrap: wrap; align-items: center; gap: 5px 7px; background: var(--bg-alt); border: 1px solid var(--border); border-radius: var(--radius); padding: 9px 12px; font-size: 12px; color: var(--text-muted); margin-bottom: 2px; }
  .ctx-strip .stand { font-weight: 700; color: var(--text); }
  .ctx-strip .who { font-weight: 700; color: var(--text); }
  .ctx-sep { opacity: .35; }

  /* Percentile panel: number-first read, railed bar, raw-stat sub-line */
  .pct-item { display: grid; grid-template-columns: 112px 1fr 38px; column-gap: 12px; align-items: center; margin-top: 13px; }
  .pct-label { font-size: 10.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .04em; color: var(--text-muted); }
  .pct-track { display: block; height: 7px; background: color-mix(in srgb, var(--text-muted) 14%, transparent); border-radius: 999px; overflow: hidden; }
  .pct-fill { display: block; height: 100%; border-radius: 999px; opacity: .92; }
  .pct-val { font-size: 14px; font-weight: 700; text-align: right; color: var(--text); font-variant-numeric: tabular-nums; line-height: 1; }
  .pct-sub { grid-column: 2; font-size: 11px; color: var(--text-subtle); margin-top: 4px; line-height: 1.35; }

  /* Season table (modal dialect). The scroll wrapper is the site's
     table treatment: the full per-season stat set scrolls inside its
     own container; the modal/page never overflows horizontally. */
  .rk-table-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; max-width: 100%; }
  .rk-table-scroll .m-season-table { min-width: 100%; }
  .m-season-table { width: 100%; border-collapse: collapse; font-size: 11.5px; font-family: ui-monospace, Menlo, monospace; color: var(--text); }
  .m-season-table th { text-align: right; font-weight: 700; color: var(--text-muted); padding: 7px 8px; border-bottom: 1px solid var(--border); white-space: nowrap; }
  .m-season-table th:first-child, .m-season-table td:first-child { text-align: left; }
  .m-season-table td { text-align: right; padding: 7px 8px; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
  .m-season-table tr:last-child td { border-bottom: none; }
  .m-season-table td.yr { font-weight: 700; color: var(--text-muted); }
  .m-season-table td.pg { font-weight: 700; }
  .m-season-table td.team { text-align: left; font-family: var(--font-sans); font-size: 12px; }
  .m-season-table th.team { text-align: left; }
  .m-season-table td.tst { text-align: left; font-family: var(--font-sans); font-size: 12px; font-weight: 600; }
  .m-season-table th.tst { text-align: left; }
  .inprog { display: inline-block; margin-left: 8px; padding: 2px 6px; border-radius: 5px; font-size: 9.5px; font-weight: 700; letter-spacing: .05em; text-transform: uppercase; font-family: var(--font-sans); background: color-mix(in srgb, var(--accent) 14%, transparent); color: var(--accent); vertical-align: 1px; white-space: nowrap; }
  /* Testing table percentile cell: mini railed bar + value */
  .tpct { display: flex; align-items: center; justify-content: flex-end; gap: 8px; }
  .tpct .pct-track { width: 64px; flex: none; }
  .tpct .n { font-weight: 700; color: var(--text); font-variant-numeric: tabular-nums; }
  .tpct .none { color: var(--text-subtle); }
  /* Season-line percentile cell */
  .pct-mini { display: inline-flex; align-items: center; gap: 7px; justify-content: flex-end; }
  .pct-mini .pct-track { width: 52px; height: 5px; flex: none; }
  .pct-mini b { font-weight: 700; color: var(--text); }
  .pct-mini-none { color: var(--text-subtle); }

  /* Production trajectory bars (live treatment) */
  .rk-traj { display: flex; gap: 14px; align-items: flex-end; margin: 8px 0; width: 100%; }
  .rk-tbar { position: relative; display: flex; flex-direction: column; align-items: center; flex: 1 1 0; min-width: 0; }
  .rk-ttrack { display: flex; align-items: flex-end; justify-content: center; height: 110px; width: 100%; }
  .rk-tfill { width: 100%; border-radius: 6px 6px 0 0; background: var(--text-muted); opacity: .82; }
  .rk-tbar.live .rk-tfill { opacity: 1; }
  .rk-tpg { font-size: 15px; font-weight: 800; font-variant-numeric: tabular-nums; margin-top: 6px; }
  .rk-tyr { font-size: 10.5px; color: var(--text-muted); font-weight: 700; margin-top: 1px; text-align: center; }
  .rk-live-tag { position: absolute; top: -8px; left: 50%; transform: translateX(-50%); font-size: 8.5px; font-weight: 800; letter-spacing: .08em; color: #fff; background: #2563eb; border-radius: 4px; padding: 2px 6px; white-space: nowrap; z-index: 2; }
  .rk-trend-note { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--text-muted); margin-top: 10px; background: var(--bg-alt); border: 1px solid var(--border); border-radius: 8px; padding: 9px 12px; }
  .rk-trend-note b { color: var(--text); }

  /* Advanced metrics: percentile-only table */
  .rk-adv-tbl { width: 100%; border-collapse: collapse; font-size: 12px; }
  .rk-adv-tbl th { text-align: left; font-size: 9.5px; font-weight: 800; letter-spacing: .08em; color: var(--text-subtle); text-transform: uppercase; padding: 5px 8px; border-bottom: 1px solid var(--border); }
  .rk-adv-tbl th.r, .rk-adv-tbl td.r { text-align: right; }
  .rk-adv-tbl td { padding: 7px 8px; border-bottom: 1px solid var(--border); font-family: ui-monospace, Menlo, monospace; }
  .rk-adv-tbl tr:last-child td { border-bottom: none; }
  .rk-adv-tbl td.rk-adv-name { font-family: var(--font-sans); font-weight: 600; color: var(--text); }
  .rk-adv-raw { font-weight: 800; font-size: 13px; }
  .rk-pbar { display: inline-block; width: 72px; height: 5px; background: color-mix(in srgb, var(--text-muted) 14%, transparent); border-radius: 3px; vertical-align: middle; margin-right: 8px; overflow: hidden; }
  .rk-pbar i { display: block; height: 100%; border-radius: 3px; }
  .rk-pctl { font-weight: 700; color: var(--text-muted); font-size: 11.5px; }
  .rk-pctl b { color: var(--text); }

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

  /* Detail loading shimmer */
  .rk-detail-loading { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--text-muted); padding: 12px 0; }

  @media (max-width: 768px) {
    .rk-hs-initials { width: 64px; height: 64px; min-height: 0; font-size: 22px; align-self: flex-start; flex: none; }

    /* Modal: same centered-card treatment as the standard player modal on mobile */
    #rkModal { padding: 16px 4px !important; align-items: center !important; }
    #rkModalContent {
      /* Fixed height (not max-height) so the card stays the same size whether the
         detail content is loading or loaded -- mirrors .player-modal. */
      height: 88dvh;
      height: 88vh;
      max-height: 88dvh;
      max-height: 88vh;
      max-width: calc(100% - 8px) !important;
      border-radius: var(--radius, 16px) !important;
      margin: auto;
      /* The sheet itself never scrolls sideways; wide inner tables keep
         their own scroll wrappers. */
      overflow-x: hidden;
    }

    /* Header meta on phones: the dot separators are separate flex items and
       strand at line ends when the line wraps. Drop them and let the flex
       gap separate the parts instead. */
    #rkModalContent .rk-meta-line { gap: 2px 10px !important; }
    #rkModalContent .rk-meta-sep { display: none; }

    /* Season table: compact padding */
    .m-season-table th, .m-season-table td { padding: 6px 5px; }
    .pct-item { grid-template-columns: 96px 1fr 34px; column-gap: 10px; }
    .pct-mini .pct-track { width: 38px; }
    .rk-pbar { width: 52px; margin-right: 6px; }
    .rk-adv-tbl td { padding: 7px 5px; }
    .rk-adv-tbl th { padding: 5px 5px; }
    .rk-traj { gap: 8px; }
    .rk-tfill { max-width: 64px; }

    /* Comparables: tighter */
    .m-sim-row { gap: 8px; padding: 7px 0; }
    .m-sim-disc { width: 28px; height: 28px; font-size: 10px; }
  }

  /* Phone fit for the Overview hero cards: tracks size to the screen, never
     to the longest word. Scoped to #rkModalContent so the NFL player modal
     sharing the pm- classes is untouched. */
  @media (max-width: 640px) {
    #rkHeroRow { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
    #rkModalContent .pm-hero-stat { min-width: 0; }
    #rkModalContent .pm-hero-val, #rkModalContent .rk-hero-tagval { overflow-wrap: break-word; }
    #rkModalContent .pm-hero-sub {
      white-space: normal; overflow: visible; text-overflow: clip;
      overflow-wrap: break-word;
    }
  }
</style>
""" + """
<script>
  var rkAllPlayers = [];   // full unfiltered list - never replaced after load
  var rkPosFilters = new Set();
  var rkSearch    = '';
  var rkLoaded    = false;
  var rkPipelinePaused = false;
  var rkDraftYear = null;
  var rkDraftComplete = false;
  var rkSortKey = 'rank';  // active sort key (mirrors the #rkSort dropdown)
  var rkSortDir = 'asc';   // 'asc' | 'desc' -- flipped by clicking a column header
  // Default direction when a sort key is first chosen.
  var RK_SORT_DIRS = { rank: 'asc', prod: 'desc', eff: 'desc', brk: 'desc', name: 'asc' };

  // Watchlist (localStorage-backed) with the shared undo toast.
  var rkWatch = {};
  try {
    JSON.parse(localStorage.getItem('rkWatchlistV1') || '[]').forEach(function(id){ rkWatch[id] = true; });
  } catch (e) {}
  function rkSaveWatch() {
    try { localStorage.setItem('rkWatchlistV1', JSON.stringify(Object.keys(rkWatch))); } catch (e) {}
  }

  var RK_POS_COLORS = { QB: '#3b82f6', RB: '#22c55e', WR: '#f59e0b', TE: '#8b5cf6' };
  var RK_DISC_COLORS = ['#10b981','#3b82f6','#8b5cf6','#f59e0b','#ef4444','#06b6d4','#f97316','#84cc16'];
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
  function rkHeadshot(r) {
    var url = r.headshot_url || r.espnHeadshot;
    if (url) {
      return '<img class="player-modal-headshot" src="' + rkEsc(url) +
        '" alt="" loading="lazy" data-initials="' + rkEsc(rkInitials(r.name)) +
        '" onerror="rkHsFallback(this)">';
    }
    return '<div class="rk-hs-initials">' + rkEsc(rkInitials(r.name)) + '</div>';
  }
  // Photo failed to load: swap in the initials tile (player-modal fallback).
  function rkHsFallback(img) {
    var d = document.createElement('div');
    d.className = 'rk-hs-initials';
    d.textContent = img.getAttribute('data-initials') || '';
    if (img.parentNode) img.parentNode.replaceChild(d, img);
  }

  // ── Percentile display ─────────────────────────────────────────────────
  // No numeric prospect grade is rendered anywhere. Percentile values come
  // from the API (computed within position against the graded reference
  // population); a null percentile renders nothing, never a placeholder.
  var RK_PCT_BOARD = [
    { key: 'production', label: 'Production', color: '#10b981' },
    { key: 'efficiency', label: 'Efficiency', color: '#3b82f6' },
    { key: 'breakout', label: 'Breakout', color: '#f59e0b' }
  ];
  var RK_PCT_PANEL = RK_PCT_BOARD.concat([
    { key: 'competition', label: 'Competition', color: '#2dd4bf' },
    { key: 'athleticism', label: 'Athleticism', color: '#8b5cf6' }
  ]);
  var RK_TAG_CLASS = { elite: 'tag-elite', great: 'tag-great', average: 'tag-average', day3: 'tag-day3' };
  var RK_BAND_COLORS = { elite: '#fbbf24', great: '#34d399', average: '#94a3b8', day3: '#64748b' };
  var RK_BAND_SHORT = { elite: 'Elite', great: 'Great', average: 'Average', day3: 'Day 3' };
  var RK_BAND_TD = { elite: 'td-elite', great: 'td-great', average: 'td-avg', day3: 'td-day3' };
  function rkTagChip(tag, mini) {
    if (!tag || !tag.label) return '';
    return '<span class="tag-chip' + (mini ? ' tag-mini' : '') + ' ' +
      (RK_TAG_CLASS[tag.band] || '') + '">' + rkEsc(tag.label) + '</span>';
  }
  function rkPosRankLabel(r) {
    return (r.position && r.position_rank) ? String(r.position).toUpperCase() + r.position_rank : '';
  }
  function rkHeightStr(r) {
    var ht = parseInt(r.height_inches, 10);
    if (isNaN(ht)) return '';
    return Math.floor(ht / 12) + "'" + (ht % 12) + '"';
  }
  function rkWeightStr(r) {
    return r.weight_lbs ? r.weight_lbs + ' lbs' : '';
  }
  function rkPctOf(r, key) {
    var p = r && r.percentiles ? r.percentiles[key] : null;
    if (p == null || p === '' || isNaN(+p)) return null;
    return +p;
  }

  // ── Board ──────────────────────────────────────────────────────────────
  // One percentile stat cell: value over a railed bar. The value prints in
  // the component color at 90+, plain at 60-89, muted below 60. A missing
  // percentile renders an empty cell, never a placeholder.
  function rkStatCell(r, def, extraCls) {
    var p = rkPctOf(r, def.key);
    if (p === null) return '<span class="pr-stat ' + extraCls + '"></span>';
    var cls = p >= 90 ? '' : (p < 60 ? ' sv-lo' : '');
    var style = p >= 90 ? ' style="color:' + def.color + '"' : '';
    return '<span class="pr-stat ' + extraCls + '"><span class="pr-stat-val' + cls + '"' +
      style + '>' + p + '</span>' +
      '<span class="rk-scbar"><i style="width:' + p + '%;background:' + def.color + '"></i></span></span>';
  }

  // One board row (native pr-grid grammar): rank + movement triangles,
  // name with baseline position chip, school column, percentile stats,
  // band tag chip. No numeric grade is rendered.
  function rkBoardRow(r) {
    var pid = r.player_id || '';
    var pos = (r.position || '').toUpperCase();
    var posColor = RK_POS_COLORS[pos] || '#94a3b8';
    var rank = r.overall_rank != null && r.overall_rank !== '' ? +r.overall_rank : null;
    var delta = '';
    var d = r.rank_delta == null || r.rank_delta === '' ? null : parseInt(r.rank_delta, 10);
    if (d !== null && !isNaN(d) && d !== 0) {
      delta = '<span class="pr-rank-delta ' + (d > 0 ? 'up' : 'down') + '" title="' +
        Math.abs(d) + ' spot' + (Math.abs(d) === 1 ? '' : 's') + ' overall">' +
        (d > 0 ? '&#9650;' : '&#9660;') + Math.abs(d) + '</span>';
    }
    var star = rkWatch[pid]
      ? '<span class="pr-wl-star" title="On your watchlist">&#9733;</span>' : '';
    var posRank = rkPosRankLabel(r);
    var schoolBit = '';
    if (posRank && r.school) schoolBit = '<b>' + rkEsc(posRank) + '</b><span class="pr-sep">&middot;</span>' + rkEsc(r.school);
    else if (posRank) schoolBit = '<b>' + rkEsc(posRank) + '</b>';
    else if (r.school) schoolBit = rkEsc(r.school);
    var metaTxt = schoolBit;
    var tag = r.tag && r.tag.label
      ? '<span class="m-tier ' + (RK_TAG_CLASS[r.tag.band] || '') + '">' + rkEsc(r.tag.label) + '</span>' : '';
    return '<div class="pr-player-row pr-grid-row" data-pid="' + rkEsc(pid) + '" role="button" tabindex="0" aria-label="' + rkEsc(r.name || '') + '">' +
      '<span class="pr-rank"><span class="pr-rank-num' + (rank === 1 ? ' top' : '') + '">' +
        (rank !== null && !isNaN(rank) ? rank : '-') + '</span>' + delta + '</span>' +
      '<span class="pr-name"><span class="pr-nm">' + rkEsc(r.name || 'Unknown') + star + '</span>' +
        (pos ? '<span class="rk-pos" style="--pc:' + posColor + '">' + rkEsc(pos) + '</span>' : '') +
        (r.in_season ? '<span class="m-early">In-Season</span>' : '') +
        '<span class="pr-meta">' +
          (pos ? '<span class="rk-pos" style="--pc:' + posColor + '">' + rkEsc(pos) + '</span>' : '') +
          (metaTxt ? '<span class="pr-meta-txt">' + metaTxt + '</span>' : '') +
        '</span>' +
      '</span>' +
      '<span class="pr-school">' + schoolBit + '</span>' +
      rkStatCell(r, RK_PCT_BOARD[0], '') +
      rkStatCell(r, RK_PCT_BOARD[1], 'pr-eff') +
      rkStatCell(r, RK_PCT_BOARD[2], 'pr-brk') +
      '<span class="pr-tagcell">' + tag + '</span>' +
    '</div>';
  }

  // Divider between tag bands on the default board order (short band name
  // in the band hue). Never printed above the first group.
  function rkBandDivider(band) {
    var color = RK_BAND_COLORS[band] || '#94a3b8';
    var label = RK_BAND_SHORT[band] || '';
    return '<div class="pr-tier-divider"><div class="pr-tier-divider-line" style="background:' +
      color + '"></div><span class="pr-tier-divider-label ' + (RK_BAND_TD[band] || '') + '">' +
      rkEsc(label) + '</span><div class="pr-tier-divider-line" style="background:' +
      color + '"></div></div>';
  }

  // Filtered + sorted view of the board (shared by render and CSV export).
  function rkVisiblePlayers() {
    var players = rkAllPlayers.slice();
    if (rkPosFilters.size > 0) {
      players = players.filter(function(r) {
        return rkPosFilters.has((r.position || '').toUpperCase());
      });
    }
    if (rkSearch.length > 0) {
      var q = rkSearch;
      // Shared matcher (brNameMatch, core bundle): normalized / compact /
      // initials / typo tolerant, ranked score -> prospect_score.
      if (typeof brRankPlayersByName === 'function') {
        return brRankPlayersByName(players, q, function(r) {
          return parseFloat(r.prospect_score || 0);
        });
      }
      var qL = q.toLowerCase();
      return players
        .map(function(r) {
          var n = (r.name || '').toLowerCase();
          return { r: r, s: n.includes(qL) ? 100 + (100 - n.indexOf(qL)) : 0 };
        })
        .filter(function(x) { return x.s > 0; })
        .sort(function(a, b) {
          return b.s - a.s ||
            (parseFloat(b.r.prospect_score || 0) - parseFloat(a.r.prospect_score || 0));
        })
        .map(function(x) { return x.r; });
    }
    var sortBy = rkSortKey;
    var rkDirMult = rkSortDir === 'asc' ? 1 : -1;
    players.sort(function(a, b) {
      if (sortBy === 'prod' || sortBy === 'eff' || sortBy === 'brk') {
        var ck = sortBy === 'prod' ? 'production' : (sortBy === 'eff' ? 'efficiency' : 'breakout');
        var av = rkPctOf(a, ck), bv = rkPctOf(b, ck);
        if (av === null && bv === null) return 0;
        if (av === null) return 1;   // missing percentiles always last
        if (bv === null) return -1;
        return (av - bv) * rkDirMult;
      }
      if (sortBy === 'name') {
        return String(a.name || '').localeCompare(String(b.name || '')) * rkDirMult;
      }
      var ar = a.overall_rank > 0 ? +a.overall_rank : null;
      var br = b.overall_rank > 0 ? +b.overall_rank : null;
      if (ar === null && br === null) return 0;
      if (ar === null) return 1;
      if (br === null) return -1;
      return (ar - br) * rkDirMult;
    });
    return players;
  }

  function rkRender() {
    if (!rkLoaded) return;
    var listEl = document.getElementById('rkList');
    var emptyEl = document.getElementById('rkEmpty');
    var countEl = document.getElementById('rkCount');
    var headEl = document.getElementById('rkTableHeader');
    if (!listEl) return;
    var list = rkVisiblePlayers();

    if (!list.length) {
      listEl.innerHTML = '';
      if (emptyEl) {
        emptyEl.style.display = 'block';
        var emptyTitle = emptyEl.querySelector('.empty-state-title');
        var emptyMsg = emptyEl.querySelector('.empty-state-msg');
        var pausedEmpty = rkPipelinePaused && rkAllPlayers.length === 0 &&
          rkPosFilters.size === 0 && !rkSearch;
        if (emptyTitle) {
          emptyTitle.textContent = pausedEmpty ? 'Prospect rankings paused' : 'No prospects match';
        }
        if (emptyMsg) {
          emptyMsg.textContent = pausedEmpty
            ? 'The rookie pipeline is paused until the next draft cycle. Rankings will return when the class is rebuilt.'
            : 'Try clearing a filter or widening your position and tier selections.';
        }
      }
      if (headEl) headEl.style.display = 'none';
      if (countEl) countEl.style.display = 'none';
      return;
    }

    if (emptyEl) emptyEl.style.display = 'none';
    if (headEl) headEl.style.display = '';
    if (countEl) {
      countEl.textContent = 'Showing ' + list.length + ' of ' + rkAllPlayers.length + ' prospects';
      countEl.style.display = '';
    }

    // Band dividers only make sense on the default board order: all
    // positions, no search, sorted by overall rank ascending.
    var showDividers = rkPosFilters.size === 0 && !rkSearch &&
                       rkSortKey === 'rank' && rkSortDir === 'asc';
    var byPid = {};
    list.forEach(function(r) { byPid[r.player_id || ''] = r; });
    var html = '', lastBand = null, firstGroup = true;
    list.forEach(function(r) {
      var band = r.tag ? r.tag.band : null;
      if (showDividers && band && band !== lastBand) {
        if (!firstGroup) html += rkBandDivider(band);
        firstGroup = false;
        lastBand = band;
      }
      html += rkBoardRow(r);
    });
    listEl.innerHTML = html;
    listEl.querySelectorAll('.pr-player-row').forEach(function(rowEl) {
      var r = byPid[rowEl.getAttribute('data-pid')];
      if (!r) return;
      rowEl.addEventListener('click', function() { rkOpenModal(r); });
      rowEl.addEventListener('keydown', function(ev) {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          rkOpenModal(r);
        }
      });
    });
  }

  // Export the current (filtered, sorted) board as CSV.
  function rkExportCSV() {
    var rows = [['Rank', 'Player', 'Position', 'School', 'Pos Rank',
                 'Production Pct', 'Efficiency Pct', 'Breakout Pct', 'Tag']];
    rkVisiblePlayers().forEach(function(r) {
      rows.push([
        r.overall_rank != null ? r.overall_rank : '',
        r.name || '', (r.position || '').toUpperCase(), r.school || '',
        rkPosRankLabel(r),
        rkPctOf(r, 'production') !== null ? rkPctOf(r, 'production') : '',
        rkPctOf(r, 'efficiency') !== null ? rkPctOf(r, 'efficiency') : '',
        rkPctOf(r, 'breakout') !== null ? rkPctOf(r, 'breakout') : '',
        r.tag && r.tag.label ? r.tag.label : ''
      ]);
    });
    var csv = rows.map(function(row) {
      return row.map(function(c) {
        var s = String(c == null ? '' : c);
        return /[",\\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
      }).join(',');
    }).join('\\n');
    var blob = new Blob([csv], { type: 'text/csv' });
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'rookie-prospects-' + (rkDraftYear || 'class') + '.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function() { URL.revokeObjectURL(a.href); }, 1000);
  }

  function rkSetSortKey(key) {
    rkSortKey = key;
    rkSortDir = RK_SORT_DIRS[key] || 'desc';
    rkRender();
  }
  // Clickable column headers: clicking the active key flips direction,
  // clicking a new key takes its default direction. The dropdown is kept
  // in sync so both controls always agree.
  function rkHeaderSort(key) {
    if (rkSortKey === key) rkSortDir = (rkSortDir === 'desc' ? 'asc' : 'desc');
    else { rkSortKey = key; rkSortDir = RK_SORT_DIRS[key] || 'desc'; }
    var sel = document.getElementById('rkSort');
    if (sel) sel.value = rkSortKey;
    rkRender();
  }

  // Sync the open modal's star + nudge row after a watchlist change.
  function rkSyncModalWatch(card, pid, name) {
    var on = !!rkWatch[pid];
    card.querySelectorAll('.rk-star').forEach(function(s) {
      s.classList.toggle('watching', on);
      if (s.classList.contains('player-modal-watchlist-btn')) {
        s.classList.toggle('player-modal-watchlist-btn--active', on);
        s.setAttribute('aria-pressed', on ? 'true' : 'false');
        s.title = on ? 'Remove from watchlist' : 'Add to watchlist';
      }
    });
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
    rkRender();
  }

  function rkClearSearch() {
    document.getElementById('rookieSearch').value = '';
    rkSearch = '';
    document.getElementById('rookieSearchClear').style.display = 'none';
    rkRender();
  }

</script>
""" + """
<script>

  function rkLoadingHtml() {
    return '<div class="rk-detail-loading"><div class="loading-spinner" style="width:12px;height:12px;flex-shrink:0;"></div>Loading\\u2026</div>';
  }

  // ── Player-modal-style prospect modal ─────────────────────────────────────
  // Presentation mirrors static/player_modal.js (pm-* classes from
  // dashboard.css); every field renders only when the API returns it.
  function rkFmtNum(v) {
    if (v == null || v === '') return '-';
    var n = Number(v);
    if (isNaN(n)) return '-';
    return n >= 1000 ? n.toLocaleString() : String(Math.round(n));
  }

  function rkSwitchTab(tab) {
    var content = document.getElementById('rkModalContent');
    if (!content) return;
    content.querySelectorAll('[data-rk-tab]').forEach(function(b) {
      var on = b.getAttribute('data-rk-tab') === tab;
      b.classList.toggle('active', on);
      b.setAttribute('aria-selected', on ? 'true' : 'false');
    });
    content.querySelectorAll('[data-rk-panel]').forEach(function(p) {
      p.classList.toggle('pm-panel-active', p.getAttribute('data-rk-panel') === tab);
    });
    var body = document.getElementById('rkModalBody');
    if (body) body.scrollTop = 0;
  }

  function rkOrdinal(n) {
    var s = ['th', 'st', 'nd', 'rd'], v = n % 100;
    return n + (s[(v - 20) % 10] || s[v] || s[0]);
  }

  // Band tint: one class on the modal content drives --band (top border +
  // Tag card). No tag -> the neutral slate default stays.
  function rkApplyBand(tag) {
    var content = document.getElementById('rkModalContent');
    if (!content) return;
    content.classList.remove('rk-band-elite', 'rk-band-great', 'rk-band-average', 'rk-band-day3');
    if (tag && tag.band) content.classList.add('rk-band-' + tag.band);
  }

  function rkApplyArchetype(r, d) {
    var arch = (d && d.archetype) || r.archetype || null;
    var receipt = (d && d.archetype_receipt) || r.archetype_receipt || '';
    var conf = (d && d.archetype_confidence) || r.archetype_confidence || '';
    var card = document.getElementById('rkHeroArch');
    if (arch && !card) {
      var row = document.getElementById('rkHeroRow');
      if (row) {
        card = document.createElement('div');
        card.className = 'pm-hero-stat';
        card.id = 'rkHeroArch';
        card.innerHTML =
          '<div class="pm-hero-label">Archetype</div>' +
          '<div class="pm-hero-val" id="rkHeroArchVal"></div>' +
          '<div class="pm-hero-sub" id="rkHeroArchSub"></div>';
        row.appendChild(card);
        // Keep the hero grid at one track per visible card.
        var n = row.children.length;
        var cols = [];
        for (var i = 0; i < n; i++) cols.push(i === 0 ? '1.25fr' : '1fr');
        row.style.gridTemplateColumns = cols.join(' ');
      }
    }
    if (card) {
      if (arch) {
        card.style.display = '';
        var v = document.getElementById('rkHeroArchVal');
        if (v) v.textContent = arch;
        var s = document.getElementById('rkHeroArchSub');
        if (s) s.textContent = conf ? conf + ' confidence' : (receipt || '');
      } else {
        card.style.display = 'none';
      }
    }
  }

  // Rookie Draft hero card value: the site's dynasty rookie ADP for this
  // class (avg_pick on the board row), formatted as a draft slot on a
  // 12-team board. No ADP row -> no card, never an invented slot.
  function rkRookiePickLabel(r) {
    var ap = r && r.avg_pick != null && r.avg_pick !== '' ? parseFloat(r.avg_pick) : null;
    if (ap === null || isNaN(ap) || ap <= 0) return '';
    var pick = Math.round(ap);
    var round = Math.floor((pick - 1) / 12) + 1;
    var slot = (pick - 1) % 12 + 1;
    return round + '.' + (slot < 10 ? '0' : '') + slot;
  }

  function rkDisplayPercentiles(r, d) {
    if (d && d.percentiles) return d.percentiles;
    return (r && r.percentiles) || {};
  }

  // Raw-stat sub-line under a percentile row, from stored data only. A
  // component without a sourced driving stat gets no sub-line at all.
  function rkSubLine(key, r, d) {
    d = d || {};
    var pos = (r.position || '').toUpperCase();
    if (key === 'breakout') {
      return d.breakout_age != null ? 'Broke out at ' + (+d.breakout_age).toFixed(1) : '';
    }
    if (key === 'athleticism') {
      var ath = d.athleticism || {};
      var forty = ath.forty_yard != null ? +ath.forty_yard : null;
      var wt = r.weight_lbs ? parseInt(r.weight_lbs, 10) : null;
      if (forty) return forty.toFixed(2) + ' forty' + (wt ? ' at ' + wt + ' lbs' : '');
      var ras = ath.ras_score != null ? +ath.ras_score : null;
      return ras ? ras.toFixed(1) + ' RAS' : '';
    }
    // Production / Efficiency / Competition read one season: the latest
    // completed season line, else the latest line there is.
    var lines = d.season_lines || [];
    var line = null;
    for (var i = lines.length - 1; i >= 0; i--) {
      if (!lines[i].in_progress) { line = lines[i]; break; }
    }
    if (!line && lines.length) line = lines[lines.length - 1];
    if (!line) return '';
    if (key === 'production') {
      return line.rate != null ? (+line.rate).toFixed(1) + ' ' + line.rate_unit + ', ' + line.season : '';
    }
    var srow = null;
    (d.seasons || []).forEach(function(s) { if (s.season == line.season) srow = s; });
    if (!srow) return '';
    if (key === 'efficiency') {
      var field = pos === 'QB' ? 'yds_per_attempt' : (pos === 'RB' ? 'yds_per_carry' : 'yds_per_reception');
      var unit = pos === 'QB' ? 'yds/att' : (pos === 'RB' ? 'yds/carry' : 'yds/rec');
      var v = srow[field];
      return v != null ? (+v).toFixed(1) + ' ' + unit + ', ' + line.season : '';
    }
    if (key === 'competition') {
      return srow.conference ? rkEsc(srow.conference) + ', ' + line.season : '';
    }
    return '';
  }

  // Percentile panel (Overview): full-width rows with railed bars and raw
  // stat sub-lines. No component or overall grade numeral is rendered.
  // Athleticism appears only for players with testing data (the API
  // returns a null percentile otherwise, and the row is omitted).
  function rkRenderPercentiles(r, d) {
    var body = document.getElementById('rkBreakdownBody');
    var section = document.getElementById('rkPctSection');
    if (!body) return;
    var hint = document.getElementById('rkPctHint');
    var pos = (r.position || '').toUpperCase();
    if (hint) hint.textContent = pos ? 'vs graded ' + pos + 's at the position' : '';
    var pcts = rkDisplayPercentiles(r, d);
    var rows = '';
    RK_PCT_PANEL.forEach(function(def) {
      var p = pcts[def.key];
      if (p == null) return;
      var sub = rkSubLine(def.key, r, d);
      var valStyle = p >= 90 ? ' style="color:' + def.color + '"' : '';
      rows += '<div class="pct-item"><span class="pct-label">' + def.label + '</span>' +
        '<div class="pct-main"><span class="pct-track"><span class="pct-fill" style="width:' +
          p + '%;background:' + def.color + '"></span></span>' +
          (sub ? '<div class="pct-sub">' + sub + '</div>' : '') + '</div>' +
        '<span class="pct-val"' + valStyle + '>' + p + '</span></div>';
    });
    body.innerHTML = rows;
    if (section) section.style.display = rows ? '' : 'none';
  }

  // Class context strip: positional standing in the class plus the
  // adjacent players at the position (board order), with their tags. A
  // side that does not exist is omitted, never faked.
  function rkRenderCtxStrip(r) {
    var el = document.getElementById('rkCtxStrip');
    if (!el) return;
    el.style.display = 'none';
    el.innerHTML = '';
    if (!r || !r.position || !rkAllPlayers.length) return;
    var pos = (r.position || '').toUpperCase();
    var peers = rkAllPlayers.filter(function(x) {
      return (x.position || '').toUpperCase() === pos;
    });
    if (peers.length < 2) return;
    peers.sort(function(a, b) {
      var ar = a.position_rank > 0 ? +a.position_rank : (a.overall_rank > 0 ? +a.overall_rank : 9999);
      var br = b.position_rank > 0 ? +b.position_rank : (b.overall_rank > 0 ? +b.overall_rank : 9999);
      return ar - br;
    });
    var idx = -1;
    for (var i = 0; i < peers.length; i++) {
      if (peers[i].player_id === r.player_id) { idx = i; break; }
    }
    if (idx < 0) return;
    var standing = pos + (r.position_rank > 0 ? +r.position_rank : idx + 1);
    var html = '<span class="stand">' + rkEsc(standing) + ' of ' + peers.length + '</span>';
    function side(label, p) {
      return '<span class="ctx-sep">&middot;</span><span>' + label +
        ' <span class="who">' + rkEsc(p.name || '') + '</span></span>' +
        rkTagChip(p.tag, true);
    }
    if (idx > 0) html += side('Chasing', peers[idx - 1]);
    if (idx < peers.length - 1) html += side('Chased by', peers[idx + 1]);
    el.innerHTML = html;
    el.style.display = '';
  }

  // Testing table (Overview, tested players only): stored results with a
  // percentile only where the tested-only pool supports one, else a dash.
  function rkRenderTesting(d) {
    var sec = document.getElementById('rkTestingSection');
    var host = document.getElementById('rkTestingBody');
    if (!sec || !host) return;
    var rows = (d && d.testing) || [];
    if (!rows.length) { sec.style.display = 'none'; host.innerHTML = ''; return; }
    var html = '<table class="m-season-table"><thead><tr><th class="tst">Test</th><th>Result</th><th>Percentile</th></tr></thead><tbody>';
    rows.forEach(function(t) {
      var cell = t.percentile != null
        ? '<span class="tpct"><span class="pct-track"><span class="pct-fill" style="width:' +
            t.percentile + '%;background:#8b5cf6"></span></span><span class="n">' + t.percentile + '</span></span>'
        : '<span class="tpct"><span class="none">-</span></span>';
      html += '<tr><td class="tst">' + rkEsc(t.label) + '</td><td class="pg">' +
        rkEsc(t.result) + '</td><td>' + cell + '</td></tr>';
    });
    host.innerHTML = html + '</tbody></table>';
    sec.style.display = '';
  }

  // Season lines table shared by Overview (no percentile column, the
  // ov subset of stat columns) and the Production tab (the full
  // per-season stat set plus a percentile for completed seasons only;
  // an in-progress season shows the chip and a dash, never a
  // percentile). Every cell is a per-game rate (or stored per-play
  // rate) from the API's season_line_columns/cells; a cell whose
  // source field is not stored for that season dashes. The table
  // scrolls inside its own container when the full set is wider than
  // the modal; the modal itself never overflows.
  function rkSeasonLinesTable(lines, pos, withPct, cols) {
    if (!cols || !cols.length) {
      var label = lines.length ? (lines[0].rate_label || '') : '';
      cols = [{ key: '__rate', label: label, ov: true, dp: 1 }];
    }
    var useCols = withPct ? cols : cols.filter(function(c) { return c.ov; });
    var html = '<div class="rk-table-scroll"><table class="m-season-table"><thead><tr><th>Season</th>' +
      (pos === 'QB' ? '<th class="team">Team</th>' : '') +
      '<th>GP</th>' +
      useCols.map(function(c) { return '<th>' + rkEsc(c.label) + '</th>'; }).join('') +
      (withPct ? '<th>Percentile</th>' : '') + '</tr></thead><tbody>';
    lines.forEach(function(l) {
      html += '<tr><td class="yr">' + l.season +
        (l.in_progress ? ' <span class="inprog">In progress</span>' : '') + '</td>' +
        (pos === 'QB' ? '<td class="team">' + rkEsc(l.team || '') + '</td>' : '') +
        '<td class="pg">' + (l.gp != null ? l.gp : '-') + '</td>' +
        useCols.map(function(c) {
          var v = c.key === '__rate' ? l.rate : (l.cells || {})[c.key];
          return '<td>' + (v != null ? (+v).toFixed(c.dp || 1) : '-') + '</td>';
        }).join('') +
        (withPct ? '<td>' + (l.percentile != null
          ? '<span class="pct-mini"><span class="pct-track"><span class="pct-fill" style="width:' +
              l.percentile + '%;background:#10b981"></span></span><b>' + l.percentile + '</b></span>'
          : '<span class="pct-mini-none">-</span>') + '</td>' : '') +
        '</tr>';
    });
    return html + '</tbody></table></div>';
  }

  function rkRenderSeasonLines(r, d) {
    var lines = (d && d.season_lines) || [];
    var cols = (d && d.season_line_columns) || [];
    var pos = (r.position || '').toUpperCase();
    var hint = document.getElementById('rkProdHint');
    if (hint) hint.textContent = pos ? 'Percentile vs graded ' + pos + 's' : '';
    var ovSec = document.getElementById('rkSeasonLinesSection');
    var ov = document.getElementById('rkSeasonLinesBody');
    var pr = document.getElementById('rkProdLinesBody');
    if (!lines.length) {
      if (ovSec) ovSec.style.display = 'none';
      if (pr) pr.innerHTML = '<div style="font-size:13px;color:var(--text-muted);">No season lines yet.</div>';
      return;
    }
    if (ov) ov.innerHTML = rkSeasonLinesTable(lines, pos, false, cols);
    if (pr) pr.innerHTML = rkSeasonLinesTable(lines, pos, true, cols);
  }

  // Production trajectory: per-game rate bars for the last three seasons,
  // scaled to the peak rate; the in-progress season's bar is blue + LIVE.
  function rkRenderTraj(r, d) {
    var host = document.getElementById('rkTrajBody');
    if (!host) return;
    var lines = ((d && d.season_lines) || []).filter(function(l) { return l.rate != null; });
    var shown = lines.slice(-3);
    if (!shown.length) {
      host.innerHTML = '<div style="font-size:13px;color:var(--text-muted);">No season rates yet.</div>';
      return;
    }
    var peak = 0;
    shown.forEach(function(l) { if (+l.rate > peak) peak = +l.rate; });
    var html = '<div class="rk-traj">';
    shown.forEach(function(l) {
      var h = peak > 0 ? Math.max(6, Math.round((+l.rate) / peak * 90)) : 6;
      html += '<div class="rk-tbar' + (l.in_progress ? ' live' : '') + '">' +
        (l.in_progress ? '<span class="rk-live-tag" style="top:' + (110 - h - 20) + 'px">LIVE</span>' : '') +
        '<div class="rk-ttrack"><div class="rk-tfill" style="height:' + h + 'px' +
          (l.in_progress ? ';background:#2563eb' : '') + '"></div></div>' +
        '<div class="rk-tpg">' + (+l.rate).toFixed(1) + '</div>' +
        '<div class="rk-tyr">' + l.season + ' &middot; ' + rkEsc(l.rate_unit || '') + '</div></div>';
    });
    html += '</div>';
    if (shown.length >= 2) {
      var first = shown[0], last = shown[shown.length - 1];
      if (+first.rate > 0) {
        var chg = Math.round(((+last.rate) - (+first.rate)) / (+first.rate) * 100);
        if (Math.abs(chg) >= 1) {
          var up = chg > 0;
          var col = up ? '#16a34a' : '#dc2626';
          html += '<div class="rk-trend-note"><svg width="11" height="9" viewBox="0 0 8 7" fill="' +
            col + '" aria-hidden="true"><path d="' + (up ? 'M4 0L8 7H0Z' : 'M4 7L0 0H8Z') +
            '"/></svg><span><b style="color:' + col + '">' + (up ? '+' : '') + chg +
            '%</b> per-game vs ' + first.season + '</span></div>';
        }
      }
    }
    host.innerHTML = html;
  }

  // Breakout card (Production tab): hero-receipt treatment, full width.
  function rkRenderBreakout(r, d) {
    var wrap = document.getElementById('rkBreakoutWrap');
    if (!wrap) return;
    var phrase = d && d.breakout_phrase;
    var age = d && d.breakout_age;
    if (!phrase || age == null) { wrap.innerHTML = ''; return; }
    var pos = (r.position || '').toUpperCase();
    var bp = rkDisplayPercentiles(r, d).breakout;
    var receipt = 'Breakout Age ' + (+age).toFixed(1) +
      (bp != null ? ' &middot; ' + rkOrdinal(bp) + ' percentile breakout vs graded ' + pos + 's' : '') + '.';
    wrap.innerHTML = '<hr class="pm-section-divider">' +
      '<div class="pm-hero-stat" style="text-align:left">' +
        '<div class="pm-hero-label">Breakout</div>' +
        '<div class="rk-hero-tagval" style="text-align:left">' + rkEsc(phrase) + '</div>' +
        '<div class="rk-hero-receipt">' + receipt + '</div></div>';
  }

  // Advanced tab: percentile-only table. Component-score rows (Efficiency,
  // Production) never render here; a metric whose pool cannot support a
  // percentile shows a dash.
  function rkRenderAdv(r, d) {
    var host = document.getElementById('rkAdvPctBody');
    if (!host) return;
    var rows = ((d && d.advanced) || []).filter(function(a) {
      return a && a.raw != null && !a.component;
    });
    if (!rows.length) {
      host.innerHTML = '<div style="font-size:13px;color:var(--text-muted);">No advanced metrics stored for this player.</div>';
      return;
    }
    var html = '<table class="rk-adv-tbl"><thead><tr><th>Metric</th><th class="r">Value</th><th class="r">Percentile</th></tr></thead><tbody>';
    rows.forEach(function(a) {
      var cell = a.percentile != null
        ? '<span class="rk-pbar"><i style="width:' + a.percentile + '%;background:#16a34a"></i></span><span class="rk-pctl"><b>' + a.percentile + '</b></span>'
        : '<span class="rk-pctl">-</span>';
      html += '<tr><td class="rk-adv-name">' + rkEsc(a.label) + '</td>' +
        '<td class="r rk-adv-raw">' + rkEsc(a.raw) + '</td>' +
        '<td class="r">' + cell + '</td></tr>';
    });
    host.innerHTML = html + '</tbody></table>';
  }
</script>
""" + """
<script>
  function rkOpenModal(r) {
    var modal   = document.getElementById('rkModal');
    var content = document.getElementById('rkModalContent');
    var pid  = r.player_id || '';
    var name = r.name || 'Unknown';
    var pos  = (r.position || '').toUpperCase();
    var on = !!rkWatch[pid];
    var mockPick = r.projected_pick != null ? Math.round(parseFloat(r.projected_pick)) : null;

    // Header meta line (mock treatment: dot-separated, rank first and bold;
    // the evidence-weight receipt is appended by rkFillMetaEvidence).
    var metaParts = [];
    if (r.position_rank != null && r.position_rank !== '') {
      metaParts.push('<span style="font-weight:600;color:var(--text);">' +
        rkEsc(pos + r.position_rank) + '</span>');
    } else if (r.overall_rank != null) {
      metaParts.push('<span style="font-weight:600;color:var(--text);">#' +
        r.overall_rank + '</span>');
    }
    if (r.school) metaParts.push('<span>' + rkEsc(r.school) + '</span>');
    var ageNum = parseFloat(r.age);
    if (r.age != null && !isNaN(ageNum)) {
      metaParts.push('<span>' + ageNum.toFixed(1) + ' yrs</span>');
    }
    if (r.draft_class_year) {
      metaParts.push('<span>' + r.draft_class_year + ' Class</span>');
    }
    if (mockPick != null) metaParts.push('<span>Mock Pick ' + mockPick + '</span>');
    var ht = rkHeightStr(r), wt = rkWeightStr(r);
    if (ht) metaParts.push('<span>' + rkEsc(ht) + '</span>');
    if (wt) metaParts.push('<span>' + rkEsc(wt) + '</span>');
    var metaHtml = '<div class="rk-meta-line" style="display:flex;align-items:center;flex-wrap:wrap;gap:0;">' +
      metaParts.join('<span class="rk-meta-sep" style="opacity:.35;margin:0 3px;">·</span>') +
      '<span id="rkMetaEvidence"></span></div>';

    // Hero cards (Overview). No numeric grade: the primary card is the
    // band Tag, its sub-line the strongest-evidence receipt (filled in by
    // rkFillDetail from the detail bundle). Cards with no data are omitted
    // outright.
    var tagLabel = (r.tag && r.tag.label) || '';
    var heroCards = [{ fr: '1.25fr', html:
      '<div class="pm-hero-stat pm-hero-primary">' +
        '<div class="pm-hero-label">Prospect Tag</div>' +
        '<div class="rk-hero-tagval">' + rkEsc(tagLabel) + '</div>' +
        '<div class="pm-hero-sub rk-hero-receipt" id="rkHeroReceipt" style="display:none;"></div>' +
      '</div>' }];
    if (r.position_rank != null || r.overall_rank != null) {
      var crVal = (r.position_rank != null && r.position_rank !== '')
        ? pos + r.position_rank : '#' + r.overall_rank;
      var crSub = [r.overall_rank != null ? '#' + r.overall_rank + ' overall' : '',
                   r.draft_class_year ? r.draft_class_year + ' class' : '']
        .filter(Boolean).join(' · ');
      heroCards.push({ fr: '1fr', html: '<div class="pm-hero-stat">' +
        '<div class="pm-hero-label">Class Rank</div>' +
        '<div class="pm-hero-val">' + rkEsc(crVal) + '</div>' +
        (crSub ? '<div class="pm-hero-sub">' + rkEsc(crSub) + '</div>' : '') + '</div>' });
    }
    if (mockPick != null) {
      var round = r.projected_round != null && r.projected_round !== ''
        ? parseInt(r.projected_round, 10) : Math.ceil(mockPick / 32);
      heroCards.push({ fr: '1fr', html: '<div class="pm-hero-stat">' +
        '<div class="pm-hero-label">Mock Pick</div>' +
        '<div class="pm-hero-val">' + mockPick + '</div>' +
        '<div class="pm-hero-sub">Round ' + round + '</div></div>' });
    }
    var rookiePick = rkRookiePickLabel(r);
    if (rookiePick) {
      heroCards.push({ fr: '1fr', html: '<div class="pm-hero-stat">' +
        '<div class="pm-hero-label">Rookie Draft</div>' +
        '<div class="pm-hero-val">' + rookiePick + '</div>' +
        '<div class="pm-hero-sub">Dynasty rookie draft</div></div>' });
    }
    if (r.archetype) {
      heroCards.push({ fr: '1fr', html: '<div class="pm-hero-stat" id="rkHeroArch">' +
        '<div class="pm-hero-label">Archetype</div>' +
        '<div class="pm-hero-val" id="rkHeroArchVal"></div>' +
        '<div class="pm-hero-sub" id="rkHeroArchSub"></div></div>' });
    }
    var heroHtml = heroCards.map(function(c) { return c.html; }).join('');
    var heroCols = heroCards.map(function(c) { return c.fr; }).join(' ');

    // Scouting notes (pm-rk-scout treatment); the card hides when empty.
    var reasons = String(r.key_reasons || '').split('\\n')
      .filter(function(l) { return l.trim(); });
    var notesHtml = reasons.length
      ? '<div class="pm-rk-card pm-rk-scout">' +
          '<div class="pm-rk-card-h">Scouting notes</div>' +
          '<ul style="margin:0;padding:0;list-style:none;">' +
          reasons.map(function(l) {
            return '<li style="font-size:13px;color:var(--text-muted);padding:3px 0 3px 14px;position:relative;line-height:1.5;">' +
              '<span style="position:absolute;left:0;color:var(--accent);">·</span>' +
              rkEsc(l.trim().replace(/^[·\\-•*]\\s*/, '')) + '</li>';
          }).join('') + '</ul></div>'
      : '';

    content.innerHTML =
      '<div class="player-modal-header">' +
        '<div class="player-modal-headshot-container">' + rkHeadshot(r) + '</div>' +
        '<div class="player-modal-title-section"><div class="player-modal-title-text">' +
          '<h2 class="player-modal-name" style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">' +
            '<span>' + rkEsc(name) + '</span>' +
            '<span class="pos-badge ' + rkEsc(pos) + '">' + rkEsc(pos) + '</span>' +
            (tagLabel ? '<span class="tag-chip ' + (RK_TAG_CLASS[r.tag.band] || '') + '" id="rkTagChip">' + rkEsc(tagLabel) + '</span>' : '') +
            (r.in_season ? '<span class="state-chip" id="rkStateChip">In-Season</span>' : '') +
          '</h2>' +
          '<div class="player-modal-meta">' + metaHtml + '</div>' +
        '</div></div>' +
        '<div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">' +
          '<button class="player-modal-watchlist-btn rk-star' +
            (on ? ' watching player-modal-watchlist-btn--active' : '') +
            '" data-rk-star="1" title="' + (on ? 'Remove from watchlist' : 'Add to watchlist') +
            '" aria-pressed="' + (on ? 'true' : 'false') + '" aria-label="Watch ' +
            rkEsc(name) + '"><span class="wl-star-glyph" aria-hidden="true">☆</span></button>' +
          '<button class="player-modal-close" onclick="rkCloseModal()" aria-label="Close">×</button>' +
        '</div>' +
      '</div>' +
      '<div class="pm-tab-bar" role="tablist" aria-label="Prospect details">' +
        '<button type="button" class="pm-tab active" role="tab" aria-selected="true" data-rk-tab="overview">Overview</button>' +
        '<button type="button" class="pm-tab" role="tab" aria-selected="false" data-rk-tab="production">Production</button>' +
        '<button type="button" class="pm-tab" role="tab" aria-selected="false" data-rk-tab="advanced">Advanced</button>' +
      '</div>' +
      '<div class="player-modal-body" id="rkModalBody">' +
        '<div class="pm-panel pm-panel-active" data-rk-panel="overview">' +
          '<div class="pm-hero-row" id="rkHeroRow" style="grid-template-columns:' + heroCols + ';">' +
            heroHtml + '</div>' +
          '<div class="ctx-strip" id="rkCtxStrip" style="display:none"></div>' +
          '<div id="rkPctSection">' +
            '<hr class="pm-section-divider">' +
            '<div class="pm-section-header"><span class="pm-section-label">Percentiles</span><span class="m-sec-hint" id="rkPctHint"></span></div>' +
            '<div id="rkBreakdownBody"></div>' +
          '</div>' +
          '<div id="rkTestingSection" style="display:none">' +
            '<hr class="pm-section-divider">' +
            '<div class="pm-section-header"><span class="pm-section-label">Testing</span></div>' +
            '<div id="rkTestingBody"></div>' +
          '</div>' +
          '<div id="rkSeasonLinesSection">' +
            '<hr class="pm-section-divider">' +
            '<div class="pm-section-header"><span class="pm-section-label">Season Lines</span><span class="m-sec-hint">Per-game rates</span></div>' +
            '<div id="rkSeasonLinesBody">' + rkLoadingHtml() + '</div>' +
          '</div>' +
          notesHtml +
          '<div class="pm-rk-card" style="margin-top:12px;">' +
            '<div class="pm-rk-card-h">Comparables</div>' +
            '<div id="rkComparablesBody">' + rkLoadingHtml() + '</div>' +
          '</div>' +
        '</div>' +
        '<div class="pm-panel" data-rk-panel="production">' +
          '<div class="pm-section-header"><span class="pm-section-label">Production Trajectory</span><span class="m-sec-hint">Per-game rates, not season totals</span></div>' +
          '<div id="rkTrajBody">' + rkLoadingHtml() + '</div>' +
          '<hr class="pm-section-divider">' +
          '<div class="pm-section-header"><span class="pm-section-label">Season Lines</span><span class="m-sec-hint" id="rkProdHint"></span></div>' +
          '<div id="rkProdLinesBody">' + rkLoadingHtml() + '</div>' +
          '<div id="rkBreakoutWrap"></div>' +
        '</div>' +
        '<div class="pm-panel" data-rk-panel="advanced">' +
          '<div class="pm-section-header"><span class="pm-section-label">Advanced Metrics</span><span class="m-sec-hint">Only metrics with data</span></div>' +
          '<div id="rkAdvPctBody">' + rkLoadingHtml() + '</div>' +
        '</div>' +
      '</div>';

    // Wire the tabs + watchlist star (direct binding, no inline handlers).
    content.querySelectorAll('[data-rk-tab]').forEach(function(b) {
      b.addEventListener('click', function() {
        rkSwitchTab(b.getAttribute('data-rk-tab'));
      });
    });
    var starBtn = content.querySelector('[data-rk-star]');
    if (starBtn) starBtn.addEventListener('click', function(ev) {
      rkToggleWatchModal(ev, pid, name);
    });
    rkRenderPercentiles(r, {});
    rkApplyArchetype(r, {});
    rkRenderCtxStrip(r);
    // Modal band tint follows the player's band (one --band variable).
    rkApplyBand(r.tag);

    modal.style.display = 'flex';
    document.body.style.overflow = 'hidden';

    // Auto-link to Sleeper ID silently in background
    if (pid) {
      fetch('/api/prospects/auto-link/' + encodeURIComponent(pid)).catch(function(){});
    }

    // Detail bundle: advanced metrics, seasons, archetype, capital split
    var yearQ = r.draft_class_year || rkDraftYear || '';
    fetch('/api/prospects/player/' + encodeURIComponent(pid) + '?year=' + encodeURIComponent(yearQ))
      .then(function(res){ return res.json(); })
      .then(function(d) { rkFillDetail(r, d || {}); })
      .catch(function() {
        ['rkTrajBody', 'rkProdLinesBody', 'rkAdvPctBody', 'rkSeasonLinesBody'].forEach(function(id) {
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
          cb.innerHTML = '<div style="font-size:13px;color:var(--text-muted);">No close historical comps found.</div>';
          return;
        }
        cb.innerHTML = comps.map(function(c) {
          var meta = [c.draft_class_year,
                      c.actual_pick ? 'Pick ' + c.actual_pick : null,
                      c.school].filter(Boolean).join(' · ');
          var discColor = rkDiscColor(c.player_id);
          var disc = '<div class="m-sim-disc" style="--pc:' + discColor + '">' + rkInitials(c.name) +
            (c.headshot_url
              ? '<img src="' + rkEsc(c.headshot_url) + '" alt="" loading="lazy" onerror="this.remove()"></div>'
              : '</div>');
          return '<div class="m-sim-row">' + disc +
            '<div><div class="m-sim-name">' + rkEsc(c.name) + '</div>' +
            '<div class="m-sim-meta">' + rkEsc(meta) + '</div></div>' +
            '<div class="m-sim-right">' + rkTagChip(c.tag) + '</div></div>';
        }).join('');
      })
      .catch(function() {
        var cb2 = document.getElementById('rkComparablesBody');
        if (cb2) cb2.innerHTML = '<span style="font-size:13px;color:var(--text-muted);">Could not load comparables.</span>';
      });
  }

  // Strongest-evidence receipt under the Prospect Tag (detail bundle).
  // Hidden until a real receipt exists; never a placeholder.
  function rkFillReceipt(d) {
    var el = document.getElementById('rkHeroReceipt');
    if (!el) return;
    var rec = (d && d.evidence_receipt) || '';
    if (rec) {
      el.textContent = rec;
      el.style.display = '';
    }
  }

  // Evidence-weight receipt at the end of the meta line, from the stored
  // seasons in the detail bundle ("3 seasons · 34 games").
  function rkFillMetaEvidence(d) {
    var el = document.getElementById('rkMetaEvidence');
    if (!el) return;
    var seasons = (d && d.seasons) || [];
    if (!seasons.length) return;
    var games = 0;
    seasons.forEach(function(s) { games += parseInt(s.games_played, 10) || 0; });
    var line = el.parentNode;
    var sep = line && line.children.length > 1
      ? '<span class="rk-meta-sep" style="opacity:.35;margin:0 3px;">·</span>' : '';
    el.innerHTML = sep + '<span>' + seasons.length +
      (seasons.length === 1 ? ' season' : ' seasons') + ' · ' + games + ' games</span>';
  }

  function rkFillDetail(r, d) {
    // The detail bundle's tag wins over the board row's when present.
    if (d && d.tag) {
      rkApplyBand(d.tag);
      var chip = document.getElementById('rkTagChip');
      if (chip && d.tag.label) {
        chip.textContent = d.tag.label;
        chip.className = 'tag-chip ' + (RK_TAG_CLASS[d.tag.band] || '');
      }
    }
    rkApplyArchetype(r, d);
    rkFillReceipt(d);
    rkFillMetaEvidence(d);
    rkRenderPercentiles(r, d);
    rkRenderTesting(d);
    rkRenderSeasonLines(r, d);
    rkRenderTraj(r, d);
    rkRenderBreakout(r, d);
    rkRenderAdv(r, d);
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
      rkRender();
    });
  })();

  // Clickable column headers (native board pattern): click a column to
  // sort by it, click again to flip direction. Bound once; targets resolve
  // at event time so soft-nav re-renders keep working.
  if (!window.__rkSortBound) {
    window.__rkSortBound = true;
    (function() {
      function rkHeaderCell(e) {
        var cell = e.target && e.target.closest ? e.target.closest('#rkTableHeader [data-pr-sort-col]') : null;
        return cell || null;
      }
      function rkHeaderKey(cell) {
        var col = cell.getAttribute('data-pr-sort-col');
        if (col === 'player') return 'name';
        if (col === 'prod') return 'prod';
        if (col === 'eff') return 'eff';
        if (col === 'brk') return 'brk';
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
      // Subtitle slot: draft class + the pool line generated server-side
      // from the reference population actually used for the percentiles.
      var poolEl = document.getElementById('rkPoolLine');
      if (poolEl && data.percentile_pool) {
        poolEl.textContent = (rkDraftYear ? rkDraftYear + ' draft class. ' : '') +
          data.percentile_pool + '.';
      }
      // Active-setting tags: the class on show, plus the In-Season tag
      // while any player on the board is mid-season.
      var settingsEl = document.getElementById('rkActiveSettings');
      if (settingsEl && rkDraftYear) {
        document.getElementById('rkClassTag').textContent = rkDraftYear + ' Class';
        var anyInSeason = rkAllPlayers.some(function(p) { return p.in_season; });
        document.getElementById('rkSeasonTag').style.display = anyInSeason ? '' : 'none';
        settingsEl.style.display = '';
      }
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
</script>
"""
