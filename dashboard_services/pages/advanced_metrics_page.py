"""
Advanced Metrics leaderboard page.

A premium page: pick an advanced metric (e.g. yards/carry) and see every player
ranked at that metric in a sortable, searchable table with a relative bar. The
position filter auto-narrows to the positions where the metric is meaningful
(with manual override). Data comes from /api/advanced-metrics/leaderboard.
"""
import json
from html import escape as _esc
from typing import Optional


def build_advanced_metrics_body(
    has_premium: bool,
    metrics_spec: dict,
    league_id: Optional[str] = None,
    season: Optional[int] = None,
    platform: Optional[str] = None,
) -> str:
    from data_building.advanced_metrics import get_available_seasons
    available_seasons: list = get_available_seasons() if has_premium else []
    # Group metrics into <optgroup>s by category (Passing / Rushing / Receiving).
    _CAT_ORDER = ["Passing", "Rushing", "Receiving"]
    groups: dict = {}
    for key, spec in metrics_spec.items():
        cat = spec.get("category", "Other")
        groups.setdefault(cat, []).append((key, spec["label"]))

    def _group_key(cat):
        try:
            return _CAT_ORDER.index(cat)
        except ValueError:
            return len(_CAT_ORDER)

    metric_options = "\n".join(
        '<optgroup label="{label}">{opts}</optgroup>'.format(
            label=cat,
            opts="".join(
                f'<option value="{k}">{lbl}</option>' for k, lbl in groups[cat]
            ),
        )
        for cat in sorted(groups, key=_group_key)
    )

    def _min_vol_cfg(spec: dict) -> Optional[dict]:
        mv = spec.get("min_vol")
        if not mv:
            return None
        return {"label": mv["label"], "opts": mv["opts"]}

    # Glossary: every metric grouped by category (same order as the dropdown).
    _legend_sections = []
    for cat in sorted(groups, key=_group_key):
        _rows = "".join(
            '<div class="am-legend-row">'
            '<div class="am-legend-name">{label}</div>'
            '<div class="am-legend-desc">{desc}</div>'
            '</div>'.format(
                label=_esc(metrics_spec[k]["label"]),
                desc=_esc(metrics_spec[k].get("desc") or "No description available."),
            )
            for k, _lbl in groups[cat]
        )
        _legend_sections.append(
            '<div class="am-legend-group">'
            '<div class="am-legend-grouphead">{head}</div>{rows}</div>'.format(
                head=_esc(cat), rows=_rows,
            )
        )
    legend_html = "".join(_legend_sections)

    cfg = json.dumps({
        "hasPremium": bool(has_premium),
        "leagueId": league_id or "",
        "platform": platform or "sleeper",
        "seasons": available_seasons,
        "metrics": {
            key: {
                "label": spec["label"],
                "positions": spec["positions"],
                "category": spec.get("category", "Other"),
                "lowerBetter": bool(spec.get("lower_better")),
                "efficiency": bool(spec.get("efficiency")),
                "desc": spec.get("desc", ""),
                "minVol": _min_vol_cfg(spec),
            }
            for key, spec in metrics_spec.items()
        },
    })

    # available_seasons only contains seasons with real data, so [0] is always right.
    if season and season in available_seasons:
        _default_season = season
    else:
        _default_season = available_seasons[0] if available_seasons else None

    season_options = "".join(
        f'<option value="{s}"{"selected" if s == _default_season else ""}>{s}</option>'
        for s in available_seasons
    ) or '<option value="">No data</option>'

    html = """
    <div class="card central">
      <div class="card-header">
        <div>
          <h2>Advanced Metrics</h2>
          <div style="font-size:14px;color:var(--text-muted);margin-top:4px;">
            Rank every player by a single advanced metric. Bars are relative to the leader.
          </div>
        </div>
        <button id="amLegendBtn" type="button" class="am-legend-btn"
          onclick="document.getElementById('amLegendModal').style.display='flex'">
          <svg width="14" height="14" viewBox="0 0 14 14" fill="none" style="flex-shrink:0"><circle cx="7" cy="7" r="6" stroke="currentColor" stroke-width="1.5"/><path d="M7 6.5v3M7 4.5h.01" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/></svg>
          Metric Glossary
        </button>
      </div>

      <div id="amLegendModal" class="am-legend-modal" style="display:none;"
        onclick="if(event.target===this)this.style.display='none'">
        <div class="am-legend-card" role="dialog" aria-label="Metric glossary">
          <div class="am-legend-head">
            <span>Metric Glossary</span>
            <button type="button" class="am-legend-close" aria-label="Close"
              onclick="document.getElementById('amLegendModal').style.display='none'">&times;</button>
          </div>
          <div class="am-legend-body">__LEGEND__</div>
        </div>
      </div>
      <div class="card-body" style="padding-top:0;">

        <div class="am-controls">
          <div class="am-ctrl">
            <label class="am-ctrl-label">
              Metric
              <span class="am-info" id="amMetricInfo" tabindex="0" role="button" aria-label="Metric description">
                <i class="fa-solid fa-circle-info"></i>
                <span class="am-info-tip" id="amMetricTip"></span>
              </span>
            </label>
            <select id="amMetric" class="am-select">__METRIC_OPTIONS__</select>
          </div>
          <div class="am-ctrl am-ctrl-search">
            <label class="am-ctrl-label">Search</label>
            <input id="amSearch" type="text" autocomplete="off" placeholder="Search players…" class="am-search">
          </div>
          <div class="am-ctrl am-ctrl-season" id="amSeasonCtrl">
            <label class="am-ctrl-label">Season</label>
            <select id="amSeason" class="am-select am-season-select">__SEASON_OPTIONS__</select>
          </div>
          <div class="am-ctrl am-ctrl-games" id="amGamesCtrl" style="display:none;">
            <label class="am-ctrl-label" id="amVolLabel">Min</label>
            <select id="amMinGames" class="am-select am-season-select"></select>
          </div>
          <div class="am-ctrl">
            <label class="am-ctrl-label">Sort</label>
            <button id="amSortBtn" type="button" class="am-sort-btn">High &rarr; Low</button>
          </div>
        </div>

        <div class="am-subcontrols">
          <div id="amPositions" class="am-positions">
            <button class="am-pos active" data-pos="ALL">All</button>
            <button class="am-pos" data-pos="QB">QB</button>
            <button class="am-pos" data-pos="RB">RB</button>
            <button class="am-pos" data-pos="WR">WR</button>
            <button class="am-pos" data-pos="TE">TE</button>
          </div>
          <label class="am-roster-toggle" id="amRosterToggleWrap" style="display:none;">
            <input type="checkbox" id="amRosterToggle">
            <span>My roster only</span>
          </label>
        </div>

        <div id="amLoading" style="text-align:center;padding:40px 0;color:var(--text-muted);">
          <div class="loading-spinner" style="margin:0 auto 12px;"></div>
          Loading metrics…
        </div>

        <div id="amPaywall" style="display:none;text-align:center;padding:48px 16px;">
          <i class="fa-solid fa-lock" style="font-size:26px;color:var(--text-muted);"></i>
          <div style="font-weight:700;font-size:16px;margin-top:12px;">Advanced Metrics is a premium feature</div>
          <div style="font-size:13px;color:var(--text-muted);margin:6px 0 16px;">
            Unlock per-metric leaderboards across every player.
          </div>
          <button onclick="if(window.showPaywall)showPaywall('advanced-metrics')"
            style="font-size:13px;font-weight:700;padding:8px 18px;border:none;border-radius:10px;
                   background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;cursor:pointer;">
            Upgrade &rarr;
          </button>
        </div>

        <div id="amEmpty" style="display:none;text-align:center;padding:40px 0;color:var(--text-muted);">
          No data for this metric yet.
        </div>

        <div id="amAvgNote" class="am-avg-note" style="display:none;">
          <span class="am-avg-swatch"></span>
          <span id="amAvgNoteText"></span>
        </div>

        <table id="amTable" class="am-table">
          <thead>
            <tr>
              <th class="am-rank">#</th>
              <th class="am-player">Player</th>
              <th class="am-games" title="Games played">G</th>
              <th class="am-barcell"></th>
              <th class="am-val">Value</th>
            </tr>
          </thead>
          <tbody id="amTableBody"></tbody>
        </table>

        <div id="amPagination" class="am-pagination" style="display:none;"></div>

      </div>
    </div>
    """.replace("__METRIC_OPTIONS__", metric_options).replace("__SEASON_OPTIONS__", season_options).replace("__LEGEND__", legend_html)

    style = """
    <style>
      .am-legend-btn {
        flex-shrink:0; display:flex; align-items:center; gap:6px;
        padding:5px 10px; border:1px solid var(--border); border-radius:8px;
        background:var(--card); color:var(--text-muted); font-size:12px;
        cursor:pointer; white-space:nowrap; text-align:left; transition:background .14s;
      }
      .am-legend-btn:hover { background:var(--row); }
      .am-legend-modal {
        position:fixed; inset:0; z-index:1000; display:flex; align-items:center;
        justify-content:center; padding:20px; background:rgba(0,0,0,.5);
      }
      .am-legend-card {
        background:var(--card); border:1px solid var(--border); border-radius:14px;
        width:100%; max-width:560px; max-height:82vh; display:flex; flex-direction:column;
        box-shadow:0 16px 48px rgba(0,0,0,.3); overflow:hidden;
      }
      .am-legend-head {
        display:flex; align-items:center; justify-content:space-between;
        padding:14px 18px; border-bottom:1px solid var(--border);
        font-size:15px; font-weight:800; color:var(--text);
      }
      .am-legend-close {
        border:none; background:none; color:var(--text-muted); font-size:24px;
        line-height:1; cursor:pointer; padding:0 4px;
      }
      .am-legend-close:hover { color:var(--text); }
      .am-legend-body { padding:8px 18px 18px; overflow-y:auto; }
      .am-legend-group { margin-top:14px; }
      .am-legend-grouphead {
        font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
        color:var(--accent); margin-bottom:6px; padding-bottom:4px;
        border-bottom:1px solid var(--border);
      }
      .am-legend-row { padding:7px 0; border-bottom:1px solid var(--border); }
      .am-legend-row:last-child { border-bottom:none; }
      .am-legend-name { font-size:13px; font-weight:700; color:var(--text); }
      .am-legend-desc { font-size:12px; color:var(--text-muted); margin-top:2px; line-height:1.4; }
      @media (max-width:600px) {
        .am-legend-btn { padding:6px 10px; font-size:11px; }
      }
      .am-controls { display:flex; gap:14px; flex-wrap:wrap; align-items:flex-end; margin:16px 0 12px; }
      .am-ctrl { display:flex; flex-direction:column; gap:4px; }
      .am-ctrl-search { flex:1; min-width:160px; }
      .am-ctrl-label { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; color:var(--text-muted); }
      .am-select, .am-search, .am-sort-btn {
        padding:8px 12px; border:1px solid var(--border); border-radius:8px;
        background-color:var(--card); color:var(--text); font-size:13px; outline:none;
      }
      .am-select { min-width:180px; cursor:pointer; }
      .am-season-select { min-width:90px; }
      .am-search { width:100%; box-sizing:border-box; }
      .am-sort-btn { cursor:pointer; font-weight:600; white-space:nowrap; }
      /* Subcontrols row: positions + roster toggle always on one line */
      .am-subcontrols { display:flex; align-items:center; gap:8px; margin-bottom:14px; flex-wrap:nowrap; }
      .am-positions { display:flex; gap:6px; flex:1; min-width:0; overflow-x:auto; padding-bottom:1px; }
      .am-roster-toggle { flex-shrink:0; }
      .am-pos {
        padding:6px 14px; border-radius:20px; border:1px solid var(--border);
        background:var(--card); color:var(--text-muted); cursor:pointer;
        font-size:12px; font-weight:600; transition:all .15s; white-space:nowrap;
      }
      .am-pos.active { background:var(--text); color:var(--card); border-color:var(--text); }
      /* Metric description tooltip */
      .am-info { position:relative; display:inline-flex; margin-left:5px; color:var(--text-muted); cursor:help; vertical-align:middle; }
      .am-info i { font-size:11px; }
      .am-info-tip {
        position:absolute; bottom:calc(100% + 8px); left:50%; transform:translateX(-50%);
        width:240px; background:var(--text); color:var(--card); font-size:12px; font-weight:500;
        line-height:1.45; letter-spacing:normal; text-transform:none; padding:9px 11px; border-radius:8px;
        box-shadow:0 6px 22px rgba(15,23,42,.22); opacity:0; visibility:hidden; transition:opacity .15s; z-index:50; pointer-events:none;
      }
      .am-info-tip::after {
        content:""; position:absolute; top:100%; left:50%; transform:translateX(-50%);
        border:6px solid transparent; border-top-color:var(--text);
      }
      .am-info:hover .am-info-tip, .am-info:focus .am-info-tip { opacity:1; visibility:visible; }
      /* My roster toggle */
      .am-roster-toggle {
        display:inline-flex; align-items:center; gap:6px; padding:6px 12px; border-radius:20px;
        border:1px solid var(--border); background:var(--card); color:var(--text); cursor:pointer;
        font-size:12px; font-weight:600; white-space:nowrap;
      }
      .am-roster-toggle input { width:14px; height:14px; margin:0; accent-color:var(--accent,#2563eb); cursor:pointer; }
      .am-table { width:100%; border-collapse:collapse; }
      .am-table th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em;
        color:var(--text-muted); padding:8px 10px; border-bottom:1px solid var(--border); }
      .am-table td { padding:9px 10px; border-bottom:1px solid var(--border); font-size:14px; }
      .am-row:hover { background:var(--bg-alt, rgba(0,0,0,.03)); }
      .am-row.am-owned { background:rgba(59,130,246,0.08); }
      .am-row.am-owned:hover { background:rgba(59,130,246,0.14); }
      .am-row.am-owned .am-name::after {
        content:"YOURS"; margin-left:7px; font-size:9px; font-weight:800; letter-spacing:.04em;
        color:var(--accent,#2563eb); border:1px solid var(--accent,#2563eb); border-radius:4px;
        padding:1px 4px; vertical-align:middle;
      }
      .am-rank { width:36px; color:var(--text-muted); font-size:12px; }
      .am-games { width:36px; text-align:center; color:var(--text-muted); font-size:12px; white-space:nowrap; }
      .am-barcell { width:42%; }
      .am-val { text-align:right; font-weight:700; white-space:nowrap; width:70px; }
      .am-name { font-weight:600; }
      .am-meta { font-size:11px; color:var(--text-muted); margin-left:4px; }
      .am-bar-track { position:relative; background:var(--bg-alt, rgba(0,0,0,.06)); border-radius:6px; height:10px; width:100%; }
      .am-bar-fill { height:100%; border-radius:6px; }
      /* Positional-average marker on each bar */
      .am-bar-avg { position:absolute; top:-3px; bottom:-3px; width:2px; background:var(--text-muted); opacity:.55; border-radius:1px; }
      .am-bar-avg-lbl {
        position:absolute; bottom:calc(100% + 1px); left:50%; transform:translateX(-50%);
        font-size:8px; font-weight:800; letter-spacing:.06em; color:var(--text-muted);
        opacity:1; white-space:nowrap;
      }
      .am-avg-note { font-size:11px; color:var(--text-muted); margin:0 0 10px; display:flex; align-items:center; gap:6px; }
      .am-avg-note .am-avg-swatch { display:inline-block; width:2px; height:12px; background:var(--text-muted); opacity:.55; }
      @media (max-width:600px){
        .am-barcell, .am-table th.am-barcell,
        .am-games, .am-table th.am-games { display:none; }
        .am-controls { gap:10px; }
        /* Metric takes full width; Search fills the row below it */
        .am-ctrl:first-child { flex:1 0 100%; }
        .am-ctrl-search { flex:1 0 100%; }
        /* Season, Min Games, Sort share a row */
        .am-ctrl-season, .am-ctrl-games { flex:1; }
        .am-select { min-width:0; width:100%; }
        /* Smaller position pills on narrow screens */
        .am-pos { padding:5px 10px; font-size:11px; }
        .am-roster-toggle { padding:5px 10px; font-size:11px; }
      }
      .am-pagination { display:flex; align-items:center; justify-content:center; gap:16px; padding:16px 0 4px; }
      .am-page-btn {
        padding:6px 14px; border-radius:8px; border:1px solid var(--border);
        background:var(--card); color:var(--text); font-size:13px; font-weight:600; cursor:pointer;
      }
      .am-page-btn:disabled { opacity:.35; cursor:not-allowed; }
      .am-page-info { font-size:13px; color:var(--text-muted); white-space:nowrap; }
    </style>
    """

    script = (
        "<script>\n(function(){\n"
        "const AM_CFG = " + cfg + ";\n"
        + _AM_JS +
        "\n})();\n</script>"
    )

    return html + style + script


# Plain JS (template literals use ${...}; kept out of any f-string).
_AM_JS = r"""
  const cfg = AM_CFG;
  const metricSel = document.getElementById('amMetric');
  const posWrap   = document.getElementById('amPositions');
  const searchEl  = document.getElementById('amSearch');
  const sortBtn   = document.getElementById('amSortBtn');
  const seasonSel = document.getElementById('amSeason');
  const seasonCtrl= document.getElementById('amSeasonCtrl');
  const minGamesSel = document.getElementById('amMinGames');
  const gamesCtrl = document.getElementById('amGamesCtrl');
  const rosterWrap= document.getElementById('amRosterToggleWrap');
  const rosterChk = document.getElementById('amRosterToggle');
  const metricTip = document.getElementById('amMetricTip');
  const avgNote   = document.getElementById('amAvgNote');
  const avgNoteTxt= document.getElementById('amAvgNoteText');
  const tbody     = document.getElementById('amTableBody');
  const loading   = document.getElementById('amLoading');
  const empty     = document.getElementById('amEmpty');
  const paywall   = document.getElementById('amPaywall');
  if (!metricSel || !tbody) return;

  // Hide season selector when only one season (or none) is available.
  if (seasonCtrl && (!cfg.seasons || cfg.seasons.length <= 1)) seasonCtrl.style.display = 'none';

  const PAGE_SIZE = 25;
  const VOL_LABELS = {
    games: 'G', total_pass_att: 'Att', total_carries: 'Car',
    total_touches: 'Tch', total_targets: 'Tgt', total_receptions: 'Rec',
  };
  const state = { metric: metricSel.value, position: 'ALL', sortDir: 'desc', rows: [], search: '',
                  season: seasonSel ? (seasonSel.value || '') : '', minVol: '', rosterOnly: false, page: 0,
                  fetching: false, volCol: 'games' };
  let ownedIds = new Set();
  const paginationEl = document.getElementById('amPagination');
  const volLabel = document.getElementById('amVolLabel');

  function relevantPositions(m) {
    return (cfg.metrics[m] && cfg.metrics[m].positions) || ['QB','RB','WR','TE'];
  }
  function isEfficiency(m) { return !!(cfg.metrics[m] && cfg.metrics[m].efficiency); }
  function posColor(p) {
    return ({ QB:'#3b82f6', RB:'#22c55e', WR:'#f59e0b', TE:'#8b5cf6' })[p] || '#888';
  }
  function fmtVal(v) {
    if (v == null) return '-';
    const n = Number(v);
    return Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(2);
  }
  function updateSortBtn() {
    sortBtn.innerHTML = state.sortDir === 'desc' ? 'High &rarr; Low' : 'Low &rarr; High';
  }
  function updateMetricTip() {
    if (!metricTip) return;
    metricTip.textContent = (cfg.metrics[state.metric] && cfg.metrics[state.metric].desc) || '';
  }
  // Lowest threshold for a metric — the sensible default so the leaderboard
  // isn't dominated by tiny-sample players (e.g. 1-carry QBs at 198 yds/carry).
  function defaultVol(m) {
    const spec = cfg.metrics[m] && cfg.metrics[m].minVol;
    return (spec && spec.opts && spec.opts.length) ? String(spec.opts[0]) : '';
  }
  // Volume filter: updates label, options, and visibility based on the metric's min_vol spec.
  // The selected option reflects state.minVol (the source of truth).
  function updateVolCtrl() {
    const spec = cfg.metrics[state.metric] && cfg.metrics[state.metric].minVol;
    if (!gamesCtrl || !minGamesSel) return;
    if (!spec) { gamesCtrl.style.display = 'none'; return; }
    if (volLabel) volLabel.textContent = spec.label;
    const prev = state.minVol;
    minGamesSel.innerHTML = '<option value="">Any</option>'
      + (spec.opts || []).map(v => '<option value="' + v + '"' + (String(v) === prev ? ' selected' : '') + '>' + v + '+</option>').join('');
    gamesCtrl.style.display = '';
  }
  function updatePosButtons() {
    const rel = new Set(relevantPositions(state.metric));
    posWrap.querySelectorAll('[data-pos]').forEach(b => {
      const p = b.dataset.pos;
      const ok = p === 'ALL' || rel.has(p);
      b.disabled = !ok;
      b.style.opacity = ok ? '' : '0.35';
      b.style.cursor = ok ? 'pointer' : 'not-allowed';
      b.classList.toggle('active', p === state.position);
    });
  }
  function render() {
    if (state.fetching) return; // keep the spinner up while a fetch is in flight
    const rel = new Set(relevantPositions(state.metric));
    const up = v => String(v || '').toUpperCase();

    // Sort the full positional set first to establish canonical ranks.
    const posRows = (state.position === 'ALL'
      ? state.rows.filter(r => rel.has(up(r.position)))
      : state.rows.filter(r => up(r.position) === state.position));
    posRows.sort((a, b) => state.sortDir === 'desc'
      ? (Number(b.value) - Number(a.value))
      : (Number(a.value) - Number(b.value)));

    // Rank map so roster/search filters preserve original rank numbers.
    const rankMap = new Map(posRows.map((r, i) => [String(r.player_id), i + 1]));

    // maxAbs from the full positional set keeps bars proportional after filtering.
    const maxAbs = posRows.reduce((m, r) => Math.max(m, Math.abs(Number(r.value) || 0)), 0) || 1;

    // Apply roster/search filters for display only (order already set by posRows sort).
    let displayRows = posRows.slice();
    if (state.rosterOnly) displayRows = displayRows.filter(r => ownedIds.has(String(r.player_id)));
    if (state.search) {
      const q = state.search.toLowerCase();
      displayRows = displayRows.filter(r => (r.name || '').toLowerCase().includes(q));
    }

    loading.style.display = 'none';
    if (!displayRows.length) {
      empty.style.display = ''; tbody.innerHTML = '';
      if (avgNote) avgNote.style.display = 'none';
      if (paginationEl) paginationEl.style.display = 'none';
      empty.textContent = state.rosterOnly ? 'None of your players rank for this metric.' : 'No data for this metric yet.';
      return;
    }
    empty.style.display = 'none';

    // Average marker across all displayed rows (position-filtered but not roster/search filtered).
    let avgPct = null;
    if (posRows.length) {
      const avg = posRows.reduce((s, r) => s + (Number(r.value) || 0), 0) / posRows.length;
      avgPct = Math.max(0, Math.min(100, Math.round(Math.abs(avg) / maxAbs * 100)));
      if (avgNote) {
        avgNote.style.display = '';
        const lbl = state.position !== 'ALL' ? state.position : 'Field';
        avgNoteTxt.textContent = lbl + ' average: ' + fmtVal(avg);
      }
    } else if (avgNote) {
      avgNote.style.display = 'none';
    }

    // Pagination: clamp current page then slice.
    const total = displayRows.length;
    const maxPage = Math.max(0, Math.ceil(total / PAGE_SIZE) - 1);
    if (state.page > maxPage) state.page = maxPage;
    const start = state.page * PAGE_SIZE;
    const pageRows = displayRows.slice(start, start + PAGE_SIZE);

    tbody.innerHTML = pageRows.map((r, i) => {
      const pct = Math.max(2, Math.round(Math.abs(Number(r.value) || 0) / maxAbs * 100));
      const safe = (r.name || '').replace(/'/g, "\\'");
      const col = posColor(r.position);
      const owned = ownedIds.has(String(r.player_id));
      const rank = rankMap.get(String(r.player_id)) || '';
      // Show the avg marker on every bar, but label it only on the first row.
      const avgLbl = (avgPct != null && i === 0)
        ? '<span class="am-bar-avg-lbl">AVG</span>' : '';
      const avgMark = (avgPct != null)
        ? '<div class="am-bar-avg" style="left:' + avgPct + '%" title="' + (state.position !== 'ALL' ? state.position : 'Field') + ' average">' + avgLbl + '</div>'
        : '';
      const gamesCell = '<td class="am-games">' + (r.vol != null ? r.vol : (r.games != null ? r.games : '–')) + '</td>';
      return '<tr class="am-row' + (owned ? ' am-owned' : '') + '" style="cursor:pointer;" onclick="window.openPlayerModal&&openPlayerModal(\'' + r.player_id + '\',\'' + safe + '\')">'
        + '<td class="am-rank">' + rank + '</td>'
        + '<td class="am-player"><span class="am-name">' + (r.name || '') + '</span>'
        + '<span class="am-meta" style="color:' + col + '">' + r.position + '</span>'
        + '<span class="am-meta">' + (r.team || '') + '</span></td>'
        + gamesCell
        + '<td class="am-barcell"><div class="am-bar-track"><div class="am-bar-fill" style="width:' + pct + '%;background:' + col + '"></div>' + avgMark + '</div></td>'
        + '<td class="am-val">' + fmtVal(r.value) + '</td></tr>';
    }).join('');

    // Pagination controls.
    if (paginationEl) {
      if (total > PAGE_SIZE) {
        paginationEl.style.display = '';
        const end = Math.min(start + PAGE_SIZE, total);
        paginationEl.innerHTML = '<button class="am-page-btn" id="amPagePrev"' + (state.page === 0 ? ' disabled' : '') + '>&larr; Prev</button>'
          + '<span class="am-page-info">' + (start + 1) + '–' + end + ' of ' + total + '</span>'
          + '<button class="am-page-btn" id="amPageNext"' + (state.page >= maxPage ? ' disabled' : '') + '>Next &rarr;</button>';
        document.getElementById('amPagePrev').onclick = () => { state.page--; render(); };
        document.getElementById('amPageNext').onclick = () => { state.page++; render(); };
      } else {
        paginationEl.style.display = 'none';
      }
    }
  }
  function updateVolHeader() {
    const th = document.querySelector('#amTable thead th.am-games');
    if (th) {
      const lbl = VOL_LABELS[state.volCol] || 'G';
      th.textContent = lbl;
      th.title = { games: 'Games played', total_pass_att: 'Pass attempts', total_carries: 'Carries',
                   total_touches: 'Touches', total_targets: 'Targets', total_receptions: 'Receptions' }[state.volCol] || lbl;
    }
  }
  function fetchData() {
    if (!cfg.hasPremium) { paywall.style.display = ''; loading.style.display = 'none'; return; }
    state.fetching = true;
    loading.style.display = ''; empty.style.display = 'none'; paywall.style.display = 'none'; tbody.innerHTML = '';
    if (avgNote) avgNote.style.display = 'none';
    const params = new URLSearchParams({ metric: state.metric, platform: cfg.platform });
    if (cfg.leagueId) params.set('league_id', cfg.leagueId);
    if (state.season) params.set('season', state.season);
    if (state.minVol) params.set('min_vol', state.minVol);
    fetch('/api/advanced-metrics/leaderboard?' + params)
      .then(r => { if (r.status === 403) { state.fetching = false; paywall.style.display = ''; loading.style.display = 'none'; return null; } return r.json(); })
      .then(d => { if (!d) return; state.fetching = false; state.rows = d.players || []; state.volCol = d.vol_col || 'games'; updateVolHeader(); render(); })
      .catch(() => { state.fetching = false; loading.style.display = 'none'; empty.style.display = ''; });
  }

  // Load the viewer's roster so owned players can be highlighted / filtered.
  function loadOwnedRoster() {
    if (!cfg.leagueId) return;
    fetch('/api/league-rosters?league_id=' + encodeURIComponent(cfg.leagueId) + '&platform=' + encodeURIComponent(cfg.platform), { cache: 'no-store' })
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (!d || !d.teams) return;
        const vid = String(d.viewer_roster_id || '');
        const mine = (d.teams || []).find(t => String(t.roster_id) === vid);
        if (mine && mine.player_ids && mine.player_ids.length) {
          ownedIds = new Set(mine.player_ids.map(String));
          if (rosterWrap) rosterWrap.style.display = '';
          render();
        }
      })
      .catch(() => {});
  }

  metricSel.addEventListener('change', () => {
    state.metric = metricSel.value; state.page = 0;
    const rel = new Set(relevantPositions(state.metric));
    if (state.position !== 'ALL' && !rel.has(state.position)) state.position = 'ALL';
    state.sortDir = (cfg.metrics[state.metric] && cfg.metrics[state.metric].lowerBetter) ? 'asc' : 'desc';
    state.minVol = defaultVol(state.metric);
    updateSortBtn(); updatePosButtons(); updateMetricTip(); updateVolCtrl(); fetchData();
  });
  posWrap.addEventListener('click', e => {
    const b = e.target.closest('[data-pos]');
    if (!b || b.disabled) return;
    state.position = b.dataset.pos; state.page = 0;
    updatePosButtons(); render();
  });
  searchEl.addEventListener('input', () => { state.search = searchEl.value.trim(); state.page = 0; render(); });
  sortBtn.addEventListener('click', () => {
    state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc'; state.page = 0;
    updateSortBtn(); render();
  });
  if (seasonSel) {
    seasonSel.addEventListener('change', () => { state.season = seasonSel.value || ''; state.page = 0; fetchData(); });
  }
  if (minGamesSel) {
    minGamesSel.addEventListener('change', () => { state.minVol = minGamesSel.value || ''; state.page = 0; fetchData(); });
  }
  if (rosterChk) {
    rosterChk.addEventListener('change', () => { state.rosterOnly = rosterChk.checked; state.page = 0; render(); });
  }

  state.minVol = defaultVol(state.metric);
  updateSortBtn(); updatePosButtons(); updateMetricTip(); updateVolCtrl(); fetchData();
  loadOwnedRoster();
"""
