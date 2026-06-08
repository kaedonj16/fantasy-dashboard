"""
Advanced Metrics leaderboard page.

A premium page: pick an advanced metric (e.g. yards/carry) and see every player
ranked at that metric in a sortable, searchable table with a relative bar. The
position filter auto-narrows to the positions where the metric is meaningful
(with manual override). Data comes from /api/advanced-metrics/leaderboard.
"""
import json
from typing import Optional


def build_advanced_metrics_body(
    has_premium: bool,
    metrics_spec: dict,
    league_id: Optional[str] = None,
    season: Optional[int] = None,
    platform: Optional[str] = None,
) -> str:
    # Group metrics into <optgroup>s by the set of positions they apply to, ordered
    # broadest first (all positions → 3-position groups → 2 → single position).
    _POS_ORDER = ["QB", "RB", "WR", "TE"]
    groups: dict = {}
    for key, spec in metrics_spec.items():
        posset = tuple(p for p in _POS_ORDER if p in spec.get("positions", []))
        groups.setdefault(posset, []).append((key, spec["label"]))

    def _group_key(posset):
        return (-len(posset), [_POS_ORDER.index(p) for p in posset])

    metric_options = "\n".join(
        '<optgroup label="{label}">{opts}</optgroup>'.format(
            label="All Positions" if len(posset) == len(_POS_ORDER) else " / ".join(posset),
            opts="".join(
                f'<option value="{k}">{lbl}</option>' for k, lbl in groups[posset]
            ),
        )
        for posset in sorted(groups, key=_group_key)
    )

    cfg = json.dumps({
        "hasPremium": bool(has_premium),
        "leagueId": league_id or "",
        "platform": platform or "sleeper",
        "metrics": {
            key: {
                "label": spec["label"],
                "positions": spec["positions"],
                "lowerBetter": bool(spec.get("lower_better")),
            }
            for key, spec in metrics_spec.items()
        },
    })

    html = """
    <div class="card central">
      <div class="card-header">
        <h2>Advanced Metrics</h2>
        <div style="font-size:14px;color:var(--text-muted);margin-top:4px;">
          Rank every player by a single advanced metric. Bars are relative to the leader.
        </div>
      </div>
      <div class="card-body" style="padding-top:0;">

        <div class="am-controls">
          <div class="am-ctrl">
            <label class="am-ctrl-label">Metric</label>
            <select id="amMetric" class="am-select">__METRIC_OPTIONS__</select>
          </div>
          <div class="am-ctrl am-ctrl-search">
            <label class="am-ctrl-label">Search</label>
            <input id="amSearch" type="text" autocomplete="off" placeholder="Search players…" class="am-search">
          </div>
          <div class="am-ctrl">
            <label class="am-ctrl-label">Sort</label>
            <button id="amSortBtn" type="button" class="am-sort-btn">High &rarr; Low</button>
          </div>
        </div>

        <div id="amPositions" class="am-positions">
          <button class="am-pos active" data-pos="ALL">All</button>
          <button class="am-pos" data-pos="QB">QB</button>
          <button class="am-pos" data-pos="RB">RB</button>
          <button class="am-pos" data-pos="WR">WR</button>
          <button class="am-pos" data-pos="TE">TE</button>
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

        <table id="amTable" class="am-table">
          <thead>
            <tr>
              <th class="am-rank">#</th>
              <th class="am-player">Player</th>
              <th class="am-barcell"></th>
              <th class="am-val">Value</th>
            </tr>
          </thead>
          <tbody id="amTableBody"></tbody>
        </table>

      </div>
    </div>
    """.replace("__METRIC_OPTIONS__", metric_options)

    style = """
    <style>
      .am-controls { display:flex; gap:14px; flex-wrap:wrap; align-items:flex-end; margin:16px 0 12px; }
      .am-ctrl { display:flex; flex-direction:column; gap:4px; }
      .am-ctrl-search { flex:1; min-width:160px; }
      .am-ctrl-label { font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:.05em; color:var(--text-muted); }
      .am-select, .am-search, .am-sort-btn {
        padding:8px 12px; border:1px solid var(--border); border-radius:8px;
        background-color:var(--card); color:var(--text); font-size:13px; outline:none;
      }
      .am-select { min-width:180px; cursor:pointer; }
      .am-search { width:100%; box-sizing:border-box; }
      .am-sort-btn { cursor:pointer; font-weight:600; white-space:nowrap; }
      .am-positions { display:flex; gap:6px; flex-wrap:wrap; margin-bottom:14px; }
      .am-pos {
        padding:6px 14px; border-radius:20px; border:1px solid var(--border);
        background:var(--card); color:var(--text-muted); cursor:pointer;
        font-size:12px; font-weight:600; transition:all .15s;
      }
      .am-pos.active { background:var(--text); color:var(--card); border-color:var(--text); }
      .am-table { width:100%; border-collapse:collapse; }
      .am-table th { text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.04em;
        color:var(--text-muted); padding:8px 10px; border-bottom:1px solid var(--border); }
      .am-table td { padding:9px 10px; border-bottom:1px solid var(--border); font-size:14px; }
      .am-row:hover { background:var(--bg-alt, rgba(0,0,0,.03)); }
      .am-rank { width:36px; color:var(--text-muted); font-size:12px; }
      .am-barcell { width:42%; }
      .am-val { text-align:right; font-weight:700; white-space:nowrap; width:70px; }
      .am-name { font-weight:600; }
      .am-meta { font-size:11px; color:var(--text-muted); margin-left:4px; }
      .am-bar-track { background:var(--bg-alt, rgba(0,0,0,.06)); border-radius:6px; height:10px; width:100%; overflow:hidden; }
      .am-bar-fill { height:100%; border-radius:6px; }
      @media (max-width:600px){ .am-barcell{ display:none; } .am-table th.am-barcell{ display:none; } }
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
  const tbody     = document.getElementById('amTableBody');
  const loading   = document.getElementById('amLoading');
  const empty     = document.getElementById('amEmpty');
  const paywall   = document.getElementById('amPaywall');
  if (!metricSel || !tbody) return;

  const state = { metric: metricSel.value, position: 'ALL', sortDir: 'desc', rows: [], search: '' };

  function relevantPositions(m) {
    return (cfg.metrics[m] && cfg.metrics[m].positions) || ['QB','RB','WR','TE'];
  }
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
    const rel = new Set(relevantPositions(state.metric));
    const up = v => String(v || '').toUpperCase();
    let rows = state.rows.slice();
    rows = state.position === 'ALL'
      ? rows.filter(r => rel.has(up(r.position)))
      : rows.filter(r => up(r.position) === state.position);
    if (state.search) {
      const q = state.search.toLowerCase();
      rows = rows.filter(r => (r.name || '').toLowerCase().includes(q));
    }
    rows.sort((a, b) => state.sortDir === 'desc'
      ? (Number(b.value) - Number(a.value))
      : (Number(a.value) - Number(b.value)));
    const maxAbs = rows.reduce((m, r) => Math.max(m, Math.abs(Number(r.value) || 0)), 0) || 1;
    loading.style.display = 'none';
    if (!rows.length) { empty.style.display = ''; tbody.innerHTML = ''; return; }
    empty.style.display = 'none';
    tbody.innerHTML = rows.map((r, i) => {
      const pct = Math.max(2, Math.round(Math.abs(Number(r.value) || 0) / maxAbs * 100));
      const safe = (r.name || '').replace(/'/g, "\\'");
      const col = posColor(r.position);
      return '<tr class="am-row" style="cursor:pointer;" onclick="window.openPlayerModal&&openPlayerModal(\'' + r.player_id + '\',\'' + safe + '\')">'
        + '<td class="am-rank">' + (i + 1) + '</td>'
        + '<td class="am-player"><span class="am-name">' + (r.name || '') + '</span>'
        + '<span class="am-meta" style="color:' + col + '">' + r.position + '</span>'
        + '<span class="am-meta">' + (r.team || '') + '</span></td>'
        + '<td class="am-barcell"><div class="am-bar-track"><div class="am-bar-fill" style="width:' + pct + '%;background:' + col + '"></div></div></td>'
        + '<td class="am-val">' + fmtVal(r.value) + '</td></tr>';
    }).join('');
  }
  function fetchData() {
    if (!cfg.hasPremium) { paywall.style.display = ''; loading.style.display = 'none'; return; }
    loading.style.display = ''; empty.style.display = 'none'; paywall.style.display = 'none'; tbody.innerHTML = '';
    const params = new URLSearchParams({ metric: state.metric, platform: cfg.platform });
    if (cfg.leagueId) params.set('league_id', cfg.leagueId);
    fetch('/api/advanced-metrics/leaderboard?' + params)
      .then(r => { if (r.status === 403) { paywall.style.display = ''; loading.style.display = 'none'; return null; } return r.json(); })
      .then(d => { if (!d) return; state.rows = d.players || []; render(); })
      .catch(() => { loading.style.display = 'none'; empty.style.display = ''; });
  }

  metricSel.addEventListener('change', () => {
    state.metric = metricSel.value;
    const rel = new Set(relevantPositions(state.metric));
    if (state.position !== 'ALL' && !rel.has(state.position)) state.position = 'ALL';
    state.sortDir = (cfg.metrics[state.metric] && cfg.metrics[state.metric].lowerBetter) ? 'asc' : 'desc';
    updateSortBtn(); updatePosButtons(); fetchData();
  });
  posWrap.addEventListener('click', e => {
    const b = e.target.closest('[data-pos]');
    if (!b || b.disabled) return;
    state.position = b.dataset.pos;
    updatePosButtons(); render();
  });
  searchEl.addEventListener('input', () => { state.search = searchEl.value.trim(); render(); });
  sortBtn.addEventListener('click', () => {
    state.sortDir = state.sortDir === 'desc' ? 'asc' : 'desc';
    updateSortBtn(); render();
  });

  updateSortBtn(); updatePosButtons(); fetchData();
"""
