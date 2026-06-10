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
                "pct": bool(spec.get("pct")),
                "pctFrac": bool(spec.get("pct_frac")),
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
              Primary Metric
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
          <div class="am-ctrl" id="amTeamCtrl">
            <label class="am-ctrl-label">Team</label>
            <select id="amTeamFilter" class="am-select am-season-select">
              <option value="">All Teams</option>
            </select>
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

        <!-- Compare bar: chips for active metrics + add-stat picker -->
        <div id="amCompareBar" class="am-compare-bar">
          <div id="amCompareChips" class="am-compare-chips"></div>
          <button id="amComparePinnedBtn" type="button" class="am-add-stat-btn" style="display:none;">&#8645; Compare Pinned</button>
          <div id="amAddStatWrap" style="position:relative;flex-shrink:0;">
            <button id="amAddStatBtn" type="button" class="am-add-stat-btn">&#43; Add Metric</button>
            <div id="amStatPicker" class="am-stat-picker" style="display:none;"></div>
          </div>
        </div>

        <div id="amCompareModal" class="am-legend-modal" style="display:none;"
          onclick="if(event.target===this)this.style.display='none'">
          <div class="am-legend-card am-cmp-card" role="dialog" aria-label="Compare pinned players">
            <div class="am-legend-head">
              <span>Compare Pinned Players</span>
              <button type="button" class="am-legend-close" aria-label="Close"
                onclick="document.getElementById('amCompareModal').style.display='none'">&times;</button>
            </div>
            <div class="am-legend-body" id="amCompareBody"></div>
          </div>
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

        <div class="am-table-wrap">
        <table id="amTable" class="am-table">
          <thead>
            <tr>
              <th class="am-rank">#</th>
              <th class="am-player">Player</th>
              <th class="am-games" title="Games played">G</th>
              <th class="am-barcell" id="amMetricHeader">—</th>
            </tr>
          </thead>
          <tbody id="amTableBody"></tbody>
        </table>
        </div>

        <div id="amPagination" class="am-pagination" style="display:none;"></div>

      </div>
    </div>
    """.replace("__METRIC_OPTIONS__", metric_options).replace("__SEASON_OPTIONS__", season_options).replace("__LEGEND__", legend_html)

    style = """
    <style>
      /* Widen the card on desktop so 5 metric columns fit comfortably */
      .card.central:has(#amTable) { max-width:1050px; }
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
      .am-table th.am-sortable { cursor:pointer; user-select:none; }
      .am-table th.am-sortable:hover { color:var(--text); }
      .am-table th.am-sort-desc::after { content:' ↓'; }
      .am-table th.am-sort-asc::after { content:' ↑'; }
      .am-table td { padding:9px 10px; border-bottom:1px solid var(--border); font-size:14px; }
      /* Column dividers */
      .am-games, .am-barcell,
      .am-table th.am-games, .am-table th.am-barcell {
        border-left:1px solid var(--border);
      }
      .am-row:hover { background:var(--bg-alt, rgba(0,0,0,.03)); }
      .am-row.am-owned { background:rgba(59,130,246,0.08); }
      .am-row.am-owned:hover { background:rgba(59,130,246,0.14); }
      .am-owned-badge {
        font-size:9px; font-weight:800; letter-spacing:.04em; flex-shrink:0;
        color:var(--accent,#2563eb); border:1px solid var(--accent,#2563eb); border-radius:4px;
        padding:1px 4px; white-space:nowrap;
      }
      .am-rank { width:52px; color:var(--text-muted); font-size:12px; }
      .am-games { width:40px; text-align:center; color:var(--text-muted); font-size:12px; white-space:nowrap; }
      .am-player { width:220px; max-width:220px; }
      .am-barcell { width:auto; min-width:120px; }
      .am-table-wrap { overflow-x:auto; -webkit-overflow-scrolling:touch; }
      /* Player cell: name on left, pos+team on right */
      .am-player td, td.am-player { display:table-cell; }
      .am-player-inner { display:flex; align-items:center; justify-content:space-between; gap:8px; overflow:hidden; }
      .am-name { font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; min-width:0; }
      .am-player-right { display:flex; align-items:center; gap:4px; flex-shrink:0; }
      .am-meta { font-size:11px; color:var(--text-muted); }
      /* Merged metric cell: bar on left, value on right */
      .am-metric-cell { display:flex; align-items:center; gap:10px; }
      .am-metric-bar { flex:1; min-width:0; }
      .am-val { font-weight:700; white-space:nowrap; font-size:13px; flex-shrink:0; min-width:46px; text-align:right; }
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
        .am-games, .am-table th.am-games { display:none; }
        .am-metric-bar { display:none; }
        .am-metric-cell { gap:0; }
        .am-val { min-width:38px; font-size:12px; text-align:center; }
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
        .am-barcell { min-width:52px; }
      }
      .am-pagination { display:flex; align-items:center; justify-content:center; gap:16px; padding:16px 0 4px; }
      .am-page-btn {
        padding:6px 14px; border-radius:8px; border:1px solid var(--border);
        background:var(--card); color:var(--text); font-size:13px; font-weight:600; cursor:pointer;
      }
      .am-page-btn:disabled { opacity:.35; cursor:not-allowed; }
      .am-page-info { font-size:13px; color:var(--text-muted); white-space:nowrap; }

      /* ── Multi-stat compare bar ─────────────────────────────────────── */
      .am-compare-bar {
        display:flex; align-items:center; gap:6px; flex-wrap:wrap;
        margin-bottom:12px; min-height:28px;
      }
      .am-compare-chips { display:flex; flex-wrap:wrap; gap:6px; flex:1; min-width:0; }
      .am-chip {
        display:inline-flex; align-items:center; gap:5px;
        padding:3px 10px; border-radius:14px;
        border:1px solid var(--border); background:var(--bg-alt);
        font-size:12px; font-weight:600; color:var(--text-muted); white-space:nowrap;
      }
      .am-chip.am-chip-primary {
        background:var(--text); color:var(--card); border-color:var(--text);
      }
      .am-chip-x {
        border:none; background:none; padding:0; margin:0; line-height:1;
        font-size:15px; cursor:pointer; color:inherit; opacity:.6;
      }
      .am-chip-x:hover { opacity:1; }
      .am-add-stat-btn {
        display:inline-flex; align-items:center; gap:4px;
        padding:3px 11px; border-radius:14px;
        border:1px dashed var(--border); background:transparent;
        color:var(--text-muted); font-size:12px; font-weight:600; cursor:pointer;
        transition:border-color .15s, color .15s; white-space:nowrap;
      }
      .am-add-stat-btn:hover { border-color:var(--accent,#2563eb); color:var(--accent,#2563eb); }
      .am-add-stat-btn:disabled { opacity:.35; cursor:not-allowed; }
      /* Stat picker dropdown */
      .am-stat-picker {
        position:absolute; top:calc(100% + 6px); right:0; z-index:200;
        background:var(--card); border:1px solid var(--border); border-radius:12px;
        box-shadow:0 8px 32px rgba(0,0,0,.18);
        min-width:220px; width:max-content; max-width:min(320px,90vw); max-height:300px; overflow-y:auto;
        padding:6px 0;
      }
      .am-sp-group { border-bottom:1px solid var(--border); padding:4px 0; }
      .am-sp-group:last-child { border-bottom:none; }
      .am-sp-head {
        font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
        color:var(--text-muted); padding:4px 14px 2px;
      }
      .am-sp-cat-head {
        font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.05em;
        color:var(--accent,#2563eb); padding:8px 14px 4px;
        border-bottom:1px solid var(--border);
      }
      .am-sp-cat-note {
        font-size:11px; color:var(--text-muted); padding:8px 14px;
        border-top:1px solid var(--border); line-height:1.4; font-style:italic;
      }
      .am-sp-item {
        display:flex; align-items:center; gap:8px;
        padding:7px 14px; font-size:13px; color:var(--text); cursor:pointer;
      }
      .am-sp-item:hover { background:var(--row,rgba(0,0,0,.04)); }
      .am-sp-item.am-sp-active { color:var(--accent,#2563eb); font-weight:600; }
      .am-sp-check { width:14px; text-align:center; flex-shrink:0; font-size:12px; }
      /* Multi-stat stacked bar rows in table cell */
      .am-multi { vertical-align:middle; }
      .am-mrow {
        display:flex; align-items:center; gap:8px;
        padding:2px 0;
      }
      .am-mlabel {
        font-size:10px; font-weight:600; color:var(--text-muted);
        width:72px; flex-shrink:0;
        overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
      }
      .am-mbar-wrap { flex:1; min-width:0; }
      .am-mbar-track { background:var(--bg-alt,rgba(0,0,0,.06)); border-radius:4px; height:7px; }
      .am-mbar-fill { height:100%; border-radius:4px; transition:width .3s; }
      .am-mval {
        font-size:11px; font-weight:700; min-width:42px; text-align:right;
        white-space:nowrap; flex-shrink:0;
      }
      /* Skeleton shimmer for loading extra metric columns */
      @keyframes am-shimmer {
        0%   { background-position: 200% 0; }
        100% { background-position: -200% 0; }
      }
      .am-skel-bar {
        background: linear-gradient(90deg,
          var(--bg-alt,rgba(0,0,0,.06)) 25%,
          rgba(0,0,0,.1) 50%,
          var(--bg-alt,rgba(0,0,0,.06)) 75%);
        background-size: 200% 100%;
        animation: am-shimmer 1.2s ease-in-out infinite;
        border-radius: 4px; height: 10px; width: 100%;
      }
      @media (max-width:600px) { .am-skel-bar { display:none; } }
      /* Percentile badge */
      .am-val-wrap { display:flex; flex-direction:column; align-items:flex-end; flex-shrink:0; min-width:54px; }
      .am-pct-badge { font-size:9px; font-weight:700; letter-spacing:.02em; line-height:1.3; white-space:nowrap; opacity:.85; }
      @media (max-width:600px) { .am-pct-badge { display:none; } }
      /* Pinned-player comparison modal */
      .am-cmp-card { max-width:720px; }
      .am-cmp-table { width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }
      .am-cmp-table th, .am-cmp-table td { padding:9px 10px; border-bottom:1px solid var(--border); text-align:left; }
      .am-cmp-table th { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.04em; color:var(--text-muted); }
      .am-cmp-player-head { font-size:13px !important; text-transform:none !important; letter-spacing:0 !important; color:var(--text) !important; font-weight:800 !important; }
      .am-cmp-player-meta { font-size:11px; font-weight:600; color:var(--text-muted); margin-left:5px; }
      .am-cmp-metric { font-weight:600; color:var(--text); white-space:nowrap; }
      .am-cmp-val { font-weight:700; font-variant-numeric:tabular-nums; }
      .am-cmp-best { color:#10b981; }
      .am-cmp-rank { font-size:10px; color:var(--text-muted); margin-left:5px; font-weight:600; }
      .am-cmp-bar { height:5px; border-radius:3px; background:var(--row,rgba(127,127,127,.12)); margin-top:5px; overflow:hidden; max-width:120px; }
      .am-cmp-bar > div { height:100%; border-radius:3px; }
      /* YoY trend arrows — inline beside the value */
      .am-val-row { display:flex; align-items:center; gap:3px; }
      .am-trend-up   { font-size:11px; font-weight:700; color:#10b981; line-height:1; flex-shrink:0; }
      .am-trend-down { font-size:11px; font-weight:700; color:#ef4444; line-height:1; flex-shrink:0; }
      /* Pin button */
      .am-rank-cell { display:flex; align-items:center; gap:3px; }
      .am-pin-btn {
        border:none; background:none; padding:1px 2px; margin:0; line-height:1;
        cursor:pointer; color:var(--text-muted); opacity:0; transition:opacity .12s, color .12s;
        flex-shrink:0;
      }
      .am-row:hover .am-pin-btn { opacity:.55; }
      .am-pin-btn.am-pin-active { opacity:1 !important; color:var(--accent,#2563eb); }
      .am-row.am-pinned { background:rgba(37,99,235,.05); }
      .am-row.am-pinned:hover { background:rgba(37,99,235,.1); }
      .am-row.am-pinned.am-owned { background:rgba(37,99,235,.1); }
      .am-pin-divider td { border-bottom:2px dashed var(--accent,#2563eb) !important; padding:0 !important; height:2px !important; }
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
  const teamSel   = document.getElementById('amTeamFilter');
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
  function _loadPins() { try { return new Set(JSON.parse(localStorage.getItem('am_pins') || '[]')); } catch { return new Set(); } }
  function _savePins() { try { localStorage.setItem('am_pins', JSON.stringify([...state.pinnedIds])); } catch {} }

  const state = { metric: metricSel.value, position: 'ALL', sortDir: 'desc', sortBy: metricSel.value, rows: [], search: '',
                  season: seasonSel ? (seasonSel.value || '') : '', minVol: '', rosterOnly: false, page: 0,
                  fetching: false, volCol: 'games', team: '',
                  extraMetrics: [],     // up to 4 extra metric keys
                  extraData: {},        // key -> { byId:{player_id->value}, maxAbs }
                  prevData: {},         // player_id -> previous-season value (for YoY trend)
                  pinnedIds: _loadPins() };
  const MAX_COMPARE = 4;

  const _PIN_SVG = '<svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor"><path d="M9.828.722a.5.5 0 0 1 .354.146l4.95 4.95a.5.5 0 0 1 0 .707c-.48.48-1.072.588-1.503.588-.177 0-.335-.018-.46-.039l-3.134 3.134a5.927 5.927 0 0 1 .16 1.013c.046.702-.032 1.687-.72 2.375a.5.5 0 0 1-.707 0l-2.829-2.828-3.182 3.182c-.195.195-1.219.902-1.414.707-.195-.195.512-1.22.707-1.414l3.182-3.182-2.828-2.829a.5.5 0 0 1 0-.707c.688-.688 1.673-.767 2.375-.72a5.922 5.922 0 0 1 1.013.16l3.134-3.133a2.772 2.772 0 0 1-.04-.461c0-.43.108-1.022.589-1.503a.5.5 0 0 1 .353-.146z"/></svg>';

  function percentileBadge(rank, total) {
    if (!total || !rank) return '';
    const pct = rank / total;
    let label, color;
    if      (pct <= 0.05) { label = 'Top 5%';  color = '#10b981'; }
    else if (pct <= 0.10) { label = 'Top 10%'; color = '#22c55e'; }
    else if (pct <= 0.25) { label = 'Top 25%'; color = '#3b82f6'; }
    else if (pct <= 0.50) { label = 'Top 50%'; color = '#94a3b8'; }
    else return '';
    return '<span class="am-pct-badge" style="color:' + color + '">' + label + '</span>';
  }

  window.amTogglePin = function(id) {
    const sid = String(id);
    if (state.pinnedIds.has(sid)) state.pinnedIds.delete(sid);
    else state.pinnedIds.add(sid);
    _savePins();
    render();
  };
  let ownedIds = new Set();
  const paginationEl = document.getElementById('amPagination');
  const volLabel = document.getElementById('amVolLabel');

  // ── Multi-stat compare ────────────────────────────────────────────────────
  function updateCompareBar() {
    const chipsEl = document.getElementById('amCompareChips');
    const addBtn  = document.getElementById('amAddStatBtn');
    if (!chipsEl) return;
    const all = [state.metric, ...state.extraMetrics];
    chipsEl.innerHTML = all.map((key, i) => {
      const lbl = (cfg.metrics[key] && cfg.metrics[key].label) || key;
      const primary = i === 0;
      const x = primary ? '' : '<button class="am-chip-x" onclick="event.stopPropagation();amRemoveExtra(\'' + key + '\')" aria-label="Remove">×</button>';
      return '<span class="am-chip' + (primary ? ' am-chip-primary' : '') + '">' + lbl + x + '</span>';
    }).join('');
    if (addBtn) addBtn.disabled = state.extraMetrics.length >= MAX_COMPARE;
  }

  function buildStatPicker() {
    const picker = document.getElementById('amStatPicker');
    if (!picker) return;
    const primaryCat = (cfg.metrics[state.metric] && cfg.metrics[state.metric].category) || 'Other';
    const active = new Set([state.metric, ...state.extraMetrics]);
    const items = Object.entries(cfg.metrics)
      .filter(([, spec]) => (spec.category || 'Other') === primaryCat);
    const otherCats = ['Passing', 'Rushing', 'Receiving'].filter(c => c !== primaryCat);
    let html = '<div class="am-sp-cat-head">' + primaryCat + '</div>';
    for (const [key, spec] of items) {
      const on = active.has(key);
      const isPrimary = key === state.metric;
      html += '<div class="am-sp-item' + (on ? ' am-sp-active' : '') + '" onclick="amPickerClick(\'' + key + '\')">'
        + '<span class="am-sp-check">' + (on ? '&#10003;' : '') + '</span>'
        + spec.label
        + (isPrimary ? ' <span style="font-size:10px;opacity:.6">(primary)</span>' : '')
        + '</div>';
    }
    if (otherCats.length) {
      html += '<div class="am-sp-cat-note">To compare '
        + otherCats.join(' or ')
        + ' stats, switch the Primary Metric</div>';
    }
    picker.innerHTML = html;
  }

  window.amPickerClick = function(key) {
    if (key === state.metric) return;
    if (state.extraMetrics.includes(key)) {
      amRemoveExtra(key);
    } else {
      if (state.extraMetrics.length < MAX_COMPARE) {
        state.extraMetrics.push(key);
        updateCompareBar();
        syncExtraCols();
        render();        // show skeleton column immediately
        fetchExtraData(key);
      }
    }
    buildStatPicker();
  };

  window.amRemoveExtra = function(key) {
    state.extraMetrics = state.extraMetrics.filter(k => k !== key);
    delete state.extraData[key];
    if (state.sortBy === key) {
      state.sortBy = state.metric;
      state.sortDir = (cfg.metrics[state.metric] && cfg.metrics[state.metric].lowerBetter) ? 'asc' : 'desc';
    }
    updateCompareBar();
    buildStatPicker();
    syncExtraCols();
    render();
  };

  function fetchExtraData(key) {
    const params = new URLSearchParams({ metric: key, platform: cfg.platform });
    if (cfg.leagueId) params.set('league_id', cfg.leagueId);
    if (state.season) params.set('season', state.season);
    const vol = defaultVol(key);
    if (vol) params.set('min_vol', vol);
    fetch('/api/advanced-metrics/leaderboard?' + params)
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (!d) return;
        const rows = d.players || [];
        const maxAbs = rows.reduce((m, r) => Math.max(m, Math.abs(Number(r.value) || 0)), 0) || 1;
        state.extraData[key] = { byId: Object.fromEntries(rows.map(r => [String(r.player_id), Number(r.value)])), maxAbs };
        render();
      })
      .catch(() => {});
  }

  // ── Pinned-player comparison ──────────────────────────────────────────────
  function pinnedRows() {
    return state.rows.filter(r => state.pinnedIds.has(String(r.player_id)));
  }
  function updateComparePinnedBtn() {
    const btn = document.getElementById('amComparePinnedBtn');
    if (btn) btn.style.display = pinnedRows().length >= 2 ? '' : 'none';
  }
  function _metricRanks(key) {
    // player_id -> rank for a metric, among the players that have a value.
    const lower = !!(cfg.metrics[key] && cfg.metrics[key].lowerBetter);
    let entries;
    if (key === state.metric) {
      entries = state.rows.map(r => [String(r.player_id), Number(r.value)]);
    } else {
      const ed = state.extraData[key];
      if (!ed) return null; // still loading
      entries = Object.entries(ed.byId).map(([id, v]) => [id, Number(v)]);
    }
    entries.sort((a, b) => lower ? a[1] - b[1] : b[1] - a[1]);
    const ranks = {};
    entries.forEach(([id], i) => { ranks[id] = i + 1; });
    return ranks;
  }
  window.amShowCompare = function() {
    const modal = document.getElementById('amCompareModal');
    const body = document.getElementById('amCompareBody');
    if (!modal || !body) return;
    const players = pinnedRows();
    if (players.length < 2) return;
    const metricsList = [state.metric, ...state.extraMetrics];

    let html = '<table class="am-cmp-table"><thead><tr><th>Metric</th>';
    players.forEach(p => {
      html += '<th class="am-cmp-player-head">' + (p.name || '')
        + '<span class="am-cmp-player-meta" style="color:' + posColor(p.position) + '">' + (p.position || '') + '</span>'
        + '<span class="am-cmp-player-meta">' + (p.team || '') + '</span></th>';
    });
    html += '</tr></thead><tbody>';

    metricsList.forEach(key => {
      const lbl = (cfg.metrics[key] && cfg.metrics[key].label) || key;
      const lower = !!(cfg.metrics[key] && cfg.metrics[key].lowerBetter);
      const ranks = _metricRanks(key);
      const vals = players.map(p => {
        if (key === state.metric) return Number(p.value);
        const ed = state.extraData[key];
        const v = ed ? ed.byId[String(p.player_id)] : undefined;
        return v !== undefined && v !== null ? Number(v) : null;
      });
      const present = vals.filter(v => v != null);
      const best = present.length
        ? (lower ? Math.min(...present) : Math.max(...present))
        : null;
      const barMax = present.length ? Math.max(...present.map(v => Math.abs(v))) || 1 : 1;

      html += '<tr><td class="am-cmp-metric">' + lbl + '</td>';
      players.forEach((p, i) => {
        const v = vals[i];
        if (v == null) { html += '<td><span style="opacity:.4">–</span></td>'; return; }
        const isBest = best != null && v === best && present.length > 1;
        const rk = ranks ? ranks[String(p.player_id)] : null;
        const w = Math.min(100, Math.max(3, Math.round(Math.abs(v) / barMax * 100)));
        html += '<td><span class="am-cmp-val' + (isBest ? ' am-cmp-best' : '') + '">' + fmtVal(v, key) + '</span>'
          + (rk ? '<span class="am-cmp-rank">#' + rk + '</span>' : '')
          + '<div class="am-cmp-bar"><div style="width:' + w + '%;background:' + posColor(p.position) + '"></div></div></td>';
      });
      html += '</tr>';
    });
    html += '</tbody></table>';
    html += '<div style="font-size:11px;color:var(--text-muted);margin-top:10px;">'
      + 'Showing the primary metric plus any added metrics. Add more with + Add Metric, or unpin players from the table.</div>';
    body.innerHTML = html;
    modal.style.display = 'flex';
  };

  function toggleStatPicker() {
    const picker = document.getElementById('amStatPicker');
    if (!picker) return;
    const open = picker.style.display !== 'none' && picker.style.display !== '';
    if (open) { picker.style.display = 'none'; return; }
    buildStatPicker();
    picker.style.display = '';
  }

  // Close picker when clicking outside.
  document.addEventListener('click', function(e) {
    const wrap = document.getElementById('amAddStatWrap');
    if (wrap && !wrap.contains(e.target)) {
      const p = document.getElementById('amStatPicker');
      if (p) p.style.display = 'none';
    }
  });

  document.getElementById('amAddStatBtn').addEventListener('click', function(e) {
    e.stopPropagation();
    toggleStatPicker();
  });

  const _cmpBtn = document.getElementById('amComparePinnedBtn');
  if (_cmpBtn) _cmpBtn.addEventListener('click', function() { amShowCompare(); });

  function relevantPositions(m) {
    return (cfg.metrics[m] && cfg.metrics[m].positions) || ['QB','RB','WR','TE'];
  }
  function isEfficiency(m) { return !!(cfg.metrics[m] && cfg.metrics[m].efficiency); }
  function posColor(p) {
    return ({ QB:'#3b82f6', RB:'#22c55e', WR:'#f59e0b', TE:'#8b5cf6' })[p] || '#888';
  }
  function fmtVal(v, m) {
    if (v == null) return '-';
    const n = Number(v);
    const spec = m ? cfg.metrics[m] : null;
    if (spec && spec.pct) {
      const pct = spec.pctFrac ? n * 100 : n;
      return (Math.abs(pct) >= 10 ? pct.toFixed(1) : pct.toFixed(2)) + '%';
    }
    return Math.abs(n) >= 100 ? n.toFixed(0) : n.toFixed(2);
  }
  function updateSortBtn() {
    sortBtn.innerHTML = state.sortDir === 'desc' ? 'High &rarr; Low' : 'Low &rarr; High';
  }
  function updateSortHeaders() {
    const ths = [
      { el: document.querySelector('#amTable thead th.am-player'), col: 'name' },
      { el: document.querySelector('#amTable thead th.am-games'), col: 'games' },
      { el: document.getElementById('amMetricHeader'), col: state.metric },
    ];
    state.extraMetrics.forEach(function(key) {
      const el = document.getElementById('amExtraHeader_' + key);
      if (el) ths.push({ el: el, col: key });
    });
    ths.forEach(function({ el, col }) {
      if (!el) return;
      el.classList.add('am-sortable');
      el.classList.toggle('am-sort-asc', state.sortBy === col && state.sortDir === 'asc');
      el.classList.toggle('am-sort-desc', state.sortBy === col && state.sortDir === 'desc');
    });
  }
  function sortByCol(col) {
    if (state.sortBy === col) {
      state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc';
    } else {
      state.sortBy = col;
      if (col === 'name' || col === 'games') {
        state.sortDir = 'asc';
      } else {
        state.sortDir = (cfg.metrics[col] && cfg.metrics[col].lowerBetter) ? 'asc' : 'desc';
      }
    }
    state.page = 0;
    updateSortHeaders();
    render();
  }
  function syncExtraCols() {
    const thead = document.querySelector('#amTable thead tr');
    if (!thead) return;
    thead.querySelectorAll('.am-extra-header').forEach(function(el) { el.remove(); });
    state.extraMetrics.forEach(function(key) {
      const th = document.createElement('th');
      th.id = 'amExtraHeader_' + key;
      th.className = 'am-barcell am-extra-header';
      th.textContent = (cfg.metrics[key] && cfg.metrics[key].label) || key;
      th.addEventListener('click', function() { sortByCol(key); });
      thead.appendChild(th);
    });
    updateSortHeaders();
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
  function populateTeamFilter() {
    if (!teamSel) return;
    const teams = [...new Set(state.rows.map(r => r.team || '').filter(Boolean))].sort();
    const prev = state.team;
    teamSel.innerHTML = '<option value="">All Teams</option>'
      + teams.map(t => '<option value="' + t + '"' + (t === prev ? ' selected' : '') + '>' + t + '</option>').join('');
    if (!teams.includes(state.team)) state.team = '';
  }

  function trendArrow(curr, prev) {
    if (prev == null || prev === undefined) return '';
    const isLower = !!(cfg.metrics[state.metric] && cfg.metrics[state.metric].lowerBetter);
    const delta = Number(curr) - Number(prev);
    if (Math.abs(Number(prev)) < 0.001) return '';
    const pct = Math.abs(delta / Number(prev));
    if (pct < 0.03) return '';
    const improved = isLower ? delta < 0 : delta > 0;
    return improved
      ? '<span class="am-trend-up">&#8593;</span>'
      : '<span class="am-trend-down">&#8595;</span>';
  }

  function render() {
    if (state.fetching) return; // keep the spinner up while a fetch is in flight
    const rel = new Set(relevantPositions(state.metric));
    const up = v => String(v || '').toUpperCase();

    // Sort the full positional set first to establish canonical ranks.
    const posRows = (state.position === 'ALL'
      ? state.rows.filter(r => rel.has(up(r.position)))
      : state.rows.filter(r => up(r.position) === state.position));
    posRows.sort((a, b) => {
      let diff;
      if (state.sortBy === 'name') {
        diff = (a.name || '').localeCompare(b.name || '');
      } else if (state.sortBy === 'games') {
        const av = Number(a.vol != null ? a.vol : (a.games != null ? a.games : 0));
        const bv = Number(b.vol != null ? b.vol : (b.games != null ? b.games : 0));
        diff = av - bv;
      } else if (state.sortBy === state.metric || !state.extraData[state.sortBy]) {
        diff = Number(a.value) - Number(b.value);
      } else {
        const ed = state.extraData[state.sortBy];
        const av = ed.byId[String(a.player_id)] ?? 0;
        const bv = ed.byId[String(b.player_id)] ?? 0;
        diff = Number(av) - Number(bv);
      }
      return state.sortDir === 'desc' ? -diff : diff;
    });

    // Rank map so roster/search filters preserve original rank numbers.
    const rankMap = new Map(posRows.map((r, i) => [String(r.player_id), i + 1]));

    // Scale to the true max, but if the leader is a big outlier (>30% above the
    // 95th-percentile value) cap the scale so one player doesn't squish the rest.
    // Bars above the cap clamp at 100% width.
    const _vals = posRows.map(r => Math.abs(Number(r.value) || 0)).sort((a, b) => a - b);
    const _p95 = _vals[Math.min(Math.floor(_vals.length * 0.95), _vals.length - 1)] || 1;
    const _trueMax = _vals[_vals.length - 1] || 1;
    const maxAbs = Math.min(_trueMax, _p95 * 1.3) || 1;

    // Extra columns: same logic — scale to the max among the displayed rows so the leader fills the bar.
    const extraMaxMap = {};
    state.extraMetrics.forEach(function(key) {
      const ed = state.extraData[key];
      if (!ed) return;
      let mx = 0;
      posRows.forEach(function(r) {
        const v = ed.byId[String(r.player_id)];
        if (v != null) mx = Math.max(mx, Math.abs(Number(v) || 0));
      });
      extraMaxMap[key] = mx || 1;
    });



    // Apply roster/search filters for display only (order already set by posRows sort).
    // Pinned players float to the top regardless of sort; ranks still reflect sort order.
    let displayRows = state.pinnedIds.size > 0
      ? [...posRows.filter(r => state.pinnedIds.has(String(r.player_id))),
         ...posRows.filter(r => !state.pinnedIds.has(String(r.player_id)))]
      : posRows.slice();
    if (state.rosterOnly) displayRows = displayRows.filter(r => ownedIds.has(String(r.player_id)));
    if (state.search) {
      const q = state.search.toLowerCase();
      displayRows = displayRows.filter(r => (r.name || '').toLowerCase().includes(q));
    }
    if (state.team) displayRows = displayRows.filter(r => (r.team || '').toUpperCase() === state.team.toUpperCase());

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
        avgNoteTxt.textContent = lbl + ' average: ' + fmtVal(avg, state.metric);
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

    const multiMode = state.extraMetrics.length > 0;
    const totalRanked = posRows.length;
    tbody.innerHTML = pageRows.map((r, i) => {
      const safe = (r.name || '').replace(/'/g, "\\'");
      const col = posColor(r.position);
      const owned = ownedIds.has(String(r.player_id));
      const pinned = state.pinnedIds.has(String(r.player_id));
      const rank = rankMap.get(String(r.player_id)) || '';
      const volNum = r.vol != null ? r.vol : (r.games != null ? r.games : '–');
      const gamesCell = '<td class="am-games">' + volNum + '</td>';
      const ownedBadge = owned ? '<span class="am-owned-badge">YOURS</span>' : '';
      const pinBtn = '<button class="am-pin-btn' + (pinned ? ' am-pin-active' : '') + '" '
        + 'onclick="event.stopPropagation();amTogglePin(\'' + r.player_id + '\')" '
        + 'title="' + (pinned ? 'Unpin' : 'Pin to top') + '">' + _PIN_SVG + '</button>';
      const rankCell = '<td class="am-rank"><div class="am-rank-cell">' + pinBtn + '<span>' + rank + '</span></div></td>';
      const playerCell = '<td class="am-player"><div class="am-player-inner">'
        + '<span class="am-name">' + (r.name || '') + '</span>'
        + ownedBadge
        + '<span class="am-player-right">'
        + '<span class="am-meta">' + (r.team || '') + '</span>'
        + '<span class="am-meta" style="color:' + col + ';font-weight:600">' + r.position + '</span>'
        + '</span></div></td>';

      const badge = percentileBadge(rank, totalRanked);
      const prevVal = state.prevData[String(r.player_id)];
      const trend = trendArrow(r.value, prevVal);
      let metricCell;
      if (!multiMode) {
        const pct = Math.min(100, Math.max(2, Math.round(Math.abs(Number(r.value) || 0) / maxAbs * 100)));
        const avgLbl = (avgPct != null && i === 0) ? '<span class="am-bar-avg-lbl">AVG</span>' : '';
        const avgMark = (avgPct != null)
          ? '<div class="am-bar-avg" style="left:' + avgPct + '%" title="' + (state.position !== 'ALL' ? state.position : 'Field') + ' average">' + avgLbl + '</div>'
          : '';
        metricCell = '<td class="am-barcell"><div class="am-metric-cell">'
          + '<div class="am-metric-bar"><div class="am-bar-track"><div class="am-bar-fill" style="width:' + pct + '%;background:' + col + '"></div>' + avgMark + '</div></div>'
          + '<div class="am-val-wrap"><div class="am-val-row">' + trend + '<span class="am-val">' + fmtVal(r.value, state.metric) + '</span></div>' + badge + '</div>'
          + '</div></td>';
      } else {
        const pct = Math.min(100, Math.max(2, Math.round(Math.abs(Number(r.value) || 0) / maxAbs * 100)));
        const avgLbl2 = (avgPct != null && i === 0) ? '<span class="am-bar-avg-lbl">AVG</span>' : '';
        const avgMark2 = (avgPct != null)
          ? '<div class="am-bar-avg" style="left:' + avgPct + '%" title="' + (state.position !== 'ALL' ? state.position : 'Field') + ' average">' + avgLbl2 + '</div>'
          : '';
        metricCell = '<td class="am-barcell"><div class="am-metric-cell">'
          + '<div class="am-metric-bar"><div class="am-bar-track"><div class="am-bar-fill" style="width:' + pct + '%;background:' + col + '"></div>' + avgMark2 + '</div></div>'
          + '<div class="am-val-wrap"><div class="am-val-row">' + trend + '<span class="am-val">' + fmtVal(r.value, state.metric) + '</span></div>' + badge + '</div>'
          + '</div></td>';
        state.extraMetrics.forEach(function(key) {
          const ed = state.extraData[key];
          if (!ed) {
            metricCell += '<td class="am-barcell"><div class="am-metric-cell">'
              + '<div class="am-metric-bar"><div class="am-skel-bar"></div></div>'
              + '<span class="am-val" style="opacity:.25">—</span>'
              + '</div></td>';
            return;
          }
          const val = ed.byId[String(r.player_id)] !== undefined ? ed.byId[String(r.player_id)] : null;
          const pctBar = val != null ? Math.min(100, Math.max(2, Math.round(Math.abs(Number(val)) / extraMaxMap[key] * 100))) : 2;
          const disp = val != null ? fmtVal(val, key) : '–';
          metricCell += '<td class="am-barcell"><div class="am-metric-cell">'
            + '<div class="am-metric-bar"><div class="am-bar-track"><div class="am-bar-fill" style="width:' + pctBar + '%;background:' + col + '"></div></div></div>'
            + '<span class="am-val">' + disp + '</span>'
            + '</div></td>';
        });
      }

      // Divider row after the last pinned player (when pinned and unpinned are both present).
      const nextRow = pageRows[i + 1];
      const isPinBoundary = pinned && nextRow && !state.pinnedIds.has(String(nextRow.player_id));
      const divider = isPinBoundary
        ? '<tr class="am-pin-divider"><td colspan="99"></td></tr>'
        : '';

      return '<tr class="am-row' + (owned ? ' am-owned' : '') + (pinned ? ' am-pinned' : '') + '" style="cursor:pointer;" '
        + 'onclick="window.openPlayerModal&&openPlayerModal(\'' + r.player_id + '\',\'' + safe + '\',{tab:\'metrics\'})">'
        + rankCell
        + playerCell
        + gamesCell
        + metricCell
        + '</tr>' + divider;
    }).join('');

    updateComparePinnedBtn();

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
    const mh = document.getElementById('amMetricHeader');
    if (mh) mh.textContent = (cfg.metrics[state.metric] && cfg.metrics[state.metric].label) || '—';
    syncExtraCols();
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

    // Determine the previous season to fetch for YoY trend arrows.
    const curSeason = state.season ? parseInt(state.season) : (cfg.seasons && cfg.seasons[0]);
    const prevSeason = curSeason ? curSeason - 1 : null;
    const hasPrevInData = prevSeason && cfg.seasons && cfg.seasons.includes(prevSeason);
    const prevParams = new URLSearchParams({ metric: state.metric, platform: cfg.platform });
    if (cfg.leagueId) prevParams.set('league_id', cfg.leagueId);
    if (prevSeason) prevParams.set('season', String(prevSeason));
    if (state.minVol) prevParams.set('min_vol', state.minVol);

    const mainFetch = fetch('/api/advanced-metrics/leaderboard?' + params)
      .then(r => { if (r.status === 403) return null; return r.json(); });
    const prevFetch = hasPrevInData
      ? fetch('/api/advanced-metrics/leaderboard?' + prevParams).then(r => r.ok ? r.json() : null).catch(() => null)
      : Promise.resolve(null);

    Promise.all([mainFetch, prevFetch])
      .then(([d, pd]) => {
        if (!d) { state.fetching = false; paywall.style.display = ''; loading.style.display = 'none'; return; }
        state.fetching = false;
        state.rows = d.players || [];
        state.volCol = d.vol_col || 'games';
        // Build previous-season lookup for trend arrows.
        if (pd && pd.players) {
          state.prevData = Object.fromEntries(pd.players.map(r => [String(r.player_id), Number(r.value)]));
        } else {
          state.prevData = {};
        }
        updateVolHeader();
        populateTeamFilter();
        // Re-fetch all extra metrics (season / filter may have changed).
        state.extraData = {};
        state.extraMetrics.forEach(k => fetchExtraData(k));
        render();
      })
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
    state.extraMetrics = []; state.extraData = {}; state.prevData = {};
    const rel = new Set(relevantPositions(state.metric));
    if (state.position !== 'ALL' && !rel.has(state.position)) state.position = 'ALL';
    state.sortDir = (cfg.metrics[state.metric] && cfg.metrics[state.metric].lowerBetter) ? 'asc' : 'desc';
    state.sortBy = state.metric;
    state.minVol = defaultVol(state.metric);
    updateSortBtn(); updatePosButtons(); updateMetricTip(); updateVolCtrl(); updateVolHeader();
    updateSortHeaders(); updateCompareBar();
    fetchData();
  });
  posWrap.addEventListener('click', e => {
    const b = e.target.closest('[data-pos]');
    if (!b || b.disabled) return;
    state.position = b.dataset.pos; state.page = 0;
    updatePosButtons(); render();
  });
  searchEl.addEventListener('input', () => { state.search = searchEl.value.trim(); state.page = 0; render(); });
  sortBtn.addEventListener('click', () => {
    state.sortBy = state.metric;
    state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc'; state.page = 0;
    updateSortBtn(); updateSortHeaders(); render();
  });
  if (seasonSel) {
    seasonSel.addEventListener('change', () => { state.season = seasonSel.value || ''; state.page = 0; fetchData(); });
  }
  if (teamSel) {
    teamSel.addEventListener('change', () => { state.team = teamSel.value || ''; state.page = 0; render(); });
  }
  if (minGamesSel) {
    minGamesSel.addEventListener('change', () => { state.minVol = minGamesSel.value || ''; state.page = 0; fetchData(); });
  }
  if (rosterChk) {
    rosterChk.addEventListener('change', () => { state.rosterOnly = rosterChk.checked; state.page = 0; render(); });
  }

  // Wire column-header sort clicks.
  const _thPlayer = document.querySelector('#amTable thead th.am-player');
  const _thGames  = document.querySelector('#amTable thead th.am-games');
  const _thMetric = document.getElementById('amMetricHeader');
  if (_thPlayer) _thPlayer.addEventListener('click', () => sortByCol('name'));
  if (_thGames)  _thGames.addEventListener('click', () => sortByCol('games'));
  if (_thMetric) _thMetric.addEventListener('click', () => sortByCol(state.metric));

  state.minVol = defaultVol(state.metric);
  updateSortBtn(); updatePosButtons(); updateMetricTip(); updateVolCtrl(); updateVolHeader();
  updateSortHeaders(); updateCompareBar(); fetchData(); loadOwnedRoster();
"""
