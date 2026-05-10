"""
Trade-related page routes.

Routes: /trade, /trade-intel, /trade-database
Also handles: /<platform>/<season>/<league_id>/trade|trade-intel|trade-database
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from flask import Blueprint, session

from dashboard_services.subscriptions import has_premium_access

trade_bp = Blueprint("trade", __name__)


# ── Trade Calculator ───────────────────────────────────────────────────────────

@trade_bp.route("/trade")
@trade_bp.route("/<platform>/<int:season>/<league_id>/trade")
def page_trade(platform: Optional[str] = None, season: Optional[int] = None,
               league_id: Optional[str] = None):
    from app import (
        build_trade_calculator_body, get_league_ctx_from_cache,
        get_nfl_state, get_viewer_session_for_league, render_page,
    )
    user_id = session.get("viewer_username") or None
    if league_id:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        league_id_safe = ctx.get("league_id") or league_id
        season_safe = int(ctx.get("season") or season or datetime.now().year)
        num_teams = ctx.get("total_rosters") or None
        rec = float((ctx.get("scoring_settings") or {}).get("rec") or 0)
        scoring_format = "ppr" if rec >= 1.0 else "half" if rec >= 0.5 else "std"
        viewer = get_viewer_session_for_league(ctx.get("users") or [], ctx.get("rosters") or [])
        viewer_roster_id = viewer.get("viewer_roster_id") or ""
        has_premium = has_premium_access(user_id, league_id, platform or "sleeper")
        body = build_trade_calculator_body(league_id_safe, season_safe, num_teams=num_teams,
                                           scoring_format=scoring_format,
                                           viewer_roster_id=viewer_roster_id,
                                           has_premium=has_premium)
    else:
        state = get_nfl_state() or {}
        current_season = int(state.get("season") or datetime.now().year)
        has_premium = has_premium_access(user_id, None, "sleeper")
        body = build_trade_calculator_body(None, current_season, has_premium=has_premium)

    return render_page("BR Fantasy Trade Calculator", league_id, "trade", body, platform, season)


# ── Trade Intelligence ─────────────────────────────────────────────────────────

@trade_bp.route("/<platform>/<int:season>/<league_id>/trade-intel")
def page_trade_intel(platform: str, season: int, league_id: str):
    from app import render_page
    user_id = session.get("viewer_username")
    has_premium = has_premium_access(user_id, league_id, platform)
    body_html = f"""
    <div class="card central" style="max-width:960px;">
      <div class="card-header" style="border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:0;">
        <h2 style="margin:0 0 4px;font-size:20px;">Trade Intelligence</h2>
        <div style="font-size:13px;color:var(--text-muted);">
          Actionable insights from thousands of real dynasty trades across multiple platforms
        </div>
      </div>
      <div class="card-body" style="padding-top:20px;">

        <div class="ti-controls">
          <div class="ti-tabs">
            <button class="ti-tab active" data-tab="trending" onclick="switchTITab('trending')"><i class="fa-solid fa-fire"></i> Trending</button>
            <button class="ti-tab" data-tab="buylows"  onclick="switchTITab('buylows')"><i class="fa-solid fa-arrow-trend-down"></i> Buy Low</button>
            <button class="ti-tab" data-tab="sellhigh" onclick="switchTITab('sellhigh')"><i class="fa-solid fa-arrow-trend-up"></i> Sell High</button>
          </div>
          <div class="ti-pos-filters">
            <button class="ti-pos active" data-pos="ALL" onclick="filterTI('ALL')">All</button>
            <button class="ti-pos" data-pos="QB"  onclick="filterTI('QB')">QB</button>
            <button class="ti-pos" data-pos="RB"  onclick="filterTI('RB')">RB</button>
            <button class="ti-pos" data-pos="WR"  onclick="filterTI('WR')">WR</button>
            <button class="ti-pos" data-pos="TE"  onclick="filterTI('TE')">TE</button>
          </div>
        </div>

        <div class="ti-key">
          <div class="ti-key-item">
            <span class="ti-key-swatch" style="background:#3b82f6;opacity:.7;border-radius:3px;"></span>
            <span><span class="ti-key-label">Market</span> Real Trade-weighted Median Value</span>
          </div>
          <div class="ti-key-item">
            <span class="ti-key-swatch" style="background:#8b5cf6;opacity:.7;border-radius:3px;"></span>
            <span><span class="ti-key-label">BR Model</span> BR Production Model Value</span>
          </div>
          <div class="ti-key-item">
            <span class="ti-key-swatch ti-key-delta"></span>
            <span><span class="ti-key-label">Delta</span> Market minus BR Model</span>
          </div>
          <div class="ti-key-item">
            <span style="display:inline-flex;align-items:center;vertical-align:middle;">
              <span style="width:8px;height:8px;border-radius:50%;color:#10b981;display:flex;align-items:center;line-height:1;">▲</span>
              <span style="width:8px;height:8px;border-radius:50%;color:#ef4444;display:inline-block;line-height:1;">▼</span>
            </span>
            <span><span class="ti-key-label">Momentum</span> Rising or Falling Market Price</span>
          </div>
        </div>

        <div id="tiPagination" class="ti-pagination" style="display:none;">
          <div class="ti-pagination-info">
            <span id="tiPaginationText">Showing 1-20 of 100 players</span>
          </div>
          <div class="ti-pagination-controls">
            <button id="tiPrevBtn" class="ti-pagination-btn" onclick="loadTIPage('prev')" disabled>
              <i class="fa-solid fa-chevron-left"></i> Previous
            </button>
            <div id="tiPageNumbers" class="ti-page-numbers"></div>
            <button id="tiNextBtn" class="ti-pagination-btn" onclick="loadTIPage('next')" disabled>
              Next <i class="fa-solid fa-chevron-right"></i>
            </button>
          </div>
        </div>

        <div id="tiLoading" style="text-align:center;padding:48px 0;color:var(--text-muted);">
          <div class="loading-spinner" style="margin:0 auto 12px;"></div>
          Loading trade data...
        </div>
        <div id="tiEmpty" style="display:none;text-align:center;padding:48px 0;color:var(--text-muted);">
          No data for this filter yet — analytics need to run to populate this view.
        </div>
        <div id="tiGrid" class="ti-grid" style="display:none;"></div>

      </div>
    </div>

    <!-- Trade History Modal -->
    <div id="tiTradesOverlay" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.55);z-index:1000;align-items:center;justify-content:center;" onclick="if(event.target===this)closeTITradesModal()">
      <div class="ti-trades-modal">
        <div class="ti-trades-header">
          <div>
            <div id="tiTradesName" class="ti-trades-name"></div>
            <div id="tiTradesMeta" class="ti-trades-meta"></div>
          </div>
          <div style="display:flex;align-items:center;gap:8px;flex-shrink:0;">
            <button class="ti-profile-btn" onclick="viewTIPlayerProfile()">View Profile</button>
            <button class="ti-trades-close" onclick="closeTITradesModal()">&#x2715;</button>
          </div>
        </div>
        <div class="ti-trades-lf-bar">
          <button class="ti-lf-btn active" data-lf="all" onclick="switchTILF('all')">All</button>
          <button class="ti-lf-btn" data-lf="sf"  onclick="switchTILF('sf')">Superflex</button>
          <button class="ti-lf-btn" data-lf="1qb" onclick="switchTILF('1qb')">1QB</button>
        </div>
        <div id="tiTradesBody" class="ti-trades-body">
          <div class="ti-trades-msg">Loading trades&hellip;</div>
        </div>
        <div id="tiTradesPager" class="ti-trades-pager" style="display:none;">
          <button id="tiTradesPrev" onclick="prevTITrades()" disabled>&larr; Prev</button>
          <span id="tiTradesPagerInfo"></span>
          <button id="tiTradesNext" onclick="nextTITrades()" disabled>Next &rarr;</button>
        </div>
      </div>
    </div>

    <style>
      .ti-controls {{
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 20px;
        flex-wrap: wrap;
      }}
      .ti-tabs {{
        display: flex;
        background: var(--bg-alt, #f1f5f9);
        border-radius: 10px;
        padding: 3px;
        gap: 2px;
      }}
      .ti-tab {{
        padding: 7px 16px;
        border-radius: 8px;
        border: none;
        background: transparent;
        color: var(--text-muted);
        cursor: pointer;
        font-size: 13px;
        font-weight: 500;
        transition: all .15s;
      }}
      .ti-tab.active {{
        background: var(--card);
        color: var(--text);
        box-shadow: 0 1px 3px rgba(0,0,0,.12);
      }}
      .ti-pos-filters {{
        display: flex;
        gap: 6px;
      }}
      .ti-pos {{
        padding: 6px 13px;
        border-radius: 20px;
        border: 1px solid var(--border);
        background: var(--card);
        color: var(--text-muted);
        cursor: pointer;
        font-size: 12px;
        font-weight: 600;
        transition: all .15s;
      }}
      .ti-pos.active {{
        background: var(--text);
        color: var(--card);
        border-color: var(--text);
      }}
      .ti-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
        gap: 12px;
      }}
      .ti-card {{
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 14px;
        cursor: pointer;
        transition: transform .12s, box-shadow .12s;
        background: var(--card);
      }}
      .ti-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,.12); }}
      .ti-card-top {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:10px; }}
      .ti-name {{ font-weight:700; font-size:14px; line-height:1.3; }}
      .ti-meta {{ font-size:11px; color:var(--text-muted); margin-top:2px; }}
      .ti-chip {{
        font-size:11px; font-weight:700;
        padding:3px 9px; border-radius:10px; white-space:nowrap; flex-shrink:0;
      }}
      .ti-divider {{ height:1px; background:var(--border); margin:8px 0; }}
      .ti-row {{ display:flex; justify-content:space-between; font-size:12px; margin-top:5px; }}
      .ti-row-label {{ color:var(--text-muted); }}
      .ti-row-val {{ font-weight:600; }}
      .ti-delta-pos {{ color:#10b981; }}
      .ti-delta-neg {{ color:#ef4444; }}
      .ti-momentum {{ font-size:11px; font-weight:600; margin-top:6px; display:flex; align-items:center; gap:4px; }}
      .ti-key {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 6px 24px;
        font-size: 12px; color: var(--text-muted);
        background: var(--bg-alt, #f8fafc);
        border: 1px solid var(--border);
        border-radius: 10px; padding: 12px 16px;
        margin-bottom: 20px; line-height: 1.4;
      }}
      .ti-key-item {{
        display: flex; align-items: center; gap: 8px;
      }}
      .ti-key-swatch {{
        display: inline-block; width: 12px; height: 12px;
        flex-shrink: 0; margin-top: 1px;
      }}
      .ti-key-delta {{
        background: linear-gradient(135deg, #10b981 50%, #ef4444 50%);
        border-radius: 3px; opacity: .8;
      }}
      .ti-key-label {{
        font-weight: 600; color: var(--text);
        margin-right: 4px;
      }}
      .ti-pagination {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin: 20px 0;
        padding: 12px 0;
        border-top: 1px solid var(--border);
      }}
      .ti-pagination-info {{
        font-size: 13px;
        color: var(--text-muted);
      }}
      .ti-pagination-controls {{
        display: flex;
        align-items: center;
        gap: 12px;
      }}
      .ti-pagination-btn {{
        padding: 6px 12px;
        border: 1px solid var(--border);
        border-radius: 6px;
        background: var(--card);
        color: var(--text);
        cursor: pointer;
        font-size: 12px;
        font-weight: 500;
        transition: all .15s;
        display: flex;
        align-items: center;
        gap: 4px;
      }}
      .ti-pagination-btn:hover:not(:disabled) {{
        background: var(--bg-alt);
        border-color: var(--accent);
      }}
      .ti-pagination-btn:disabled {{
        opacity: 0.5;
        cursor: not-allowed;
      }}
      .ti-page-numbers {{
        display: flex;
        gap: 4px;
      }}
      .ti-page-number {{
        padding: 4px 8px;
        border: 1px solid var(--border);
        border-radius: 4px;
        background: var(--card);
        color: var(--text);
        cursor: pointer;
        font-size: 12px;
        font-weight: 500;
        min-width: 28px;
        text-align: center;
      }}
      .ti-page-number:hover {{
        background: var(--bg-alt);
      }}
      .ti-page-number.active {{
        background: var(--accent-color);
        color: var(--card);
        border-color: var(--accent-color);
        font-weight: 700;
      }}

      /* ── Trade History Modal ── */
      .ti-trades-modal {{
        background: var(--card);
        border-radius: 16px;
        width: min(600px, 96vw);
        max-height: 82vh;
        display: flex;
        flex-direction: column;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,.35);
      }}
      .ti-trades-header {{
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        padding: 20px 20px 14px;
        border-bottom: 1px solid var(--border);
        flex-shrink: 0;
      }}
      .ti-trades-name {{ font-size: 18px; font-weight: 700; }}
      .ti-trades-meta {{ font-size: 13px; color: var(--text-muted); margin-top: 3px; }}
      .ti-trades-close {{
        background: none; border: none; font-size: 20px;
        color: var(--text-muted); cursor: pointer; padding: 0 4px; line-height: 1;
      }}
      .ti-trades-close:hover {{ color: var(--text); }}
      .ti-profile-btn {{
        padding: 5px 12px; border-radius: 8px; font-size: 12px; font-weight: 600;
        border: 1px solid var(--border); background: var(--bg-alt, #f1f5f9);
        color: var(--text); cursor: pointer; white-space: nowrap;
        transition: opacity .15s;
      }}
      .ti-profile-btn:hover {{ opacity: .75; }}
      .ti-trades-lf-bar {{
        display: flex; gap: 6px; padding: 12px 20px;
        border-bottom: 1px solid var(--border); flex-shrink: 0;
      }}
      .ti-lf-btn {{
        padding: 5px 14px; border-radius: 20px;
        border: 1px solid var(--border); background: var(--card);
        color: var(--text-muted); font-size: 12px; font-weight: 600; cursor: pointer;
        transition: all .15s;
      }}
      .ti-lf-btn.active {{
        background: var(--text); color: var(--card);
        border-color: var(--text);
      }}
      .ti-trades-body {{ overflow-y: auto; flex: 1; padding: 0 20px; }}
      .ti-trades-msg {{ text-align: center; padding: 40px 0; color: var(--text-muted); font-size: 14px; }}
      .ti-trade-item {{
        padding: 14px 0;
        border-bottom: 1px solid var(--border);
      }}
      .ti-trade-item:last-child {{ border-bottom: none; }}
      .ti-trade-date {{
        font-size: 11px; color: var(--text-muted); font-weight: 600;
        text-transform: uppercase; letter-spacing: .05em; margin-bottom: 10px;
      }}
      .ti-trade-sides {{
        display: grid; grid-template-columns: 1fr 28px 1fr; gap: 8px; align-items: start;
      }}
      .ti-trade-side-label {{
        font-size: 10px; font-weight: 700; letter-spacing: .06em;
        color: var(--text-muted); margin-bottom: 6px; text-transform: uppercase;
      }}
      .ti-trade-asset {{ font-size: 13px; padding: 2px 0; line-height: 1.4; }}
      .ti-trade-asset.focus {{ font-weight: 700; }}
      .ti-trade-asset.other {{ color: var(--text-muted); }}
      .ti-trade-asset.pick {{ color: var(--text-muted); font-style: italic; }}
      .ti-trade-arrow {{ text-align: center; color: var(--text-muted); padding-top: 22px; font-size: 15px; }}
      .ti-trades-pager {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 12px 20px; border-top: 1px solid var(--border); flex-shrink: 0;
      }}
      .ti-trades-pager button {{
        padding: 6px 14px; border-radius: 8px;
        border: 1px solid var(--border); background: var(--card);
        color: var(--text); font-size: 13px; cursor: pointer;
      }}
      .ti-trades-pager button:disabled {{ opacity: .4; cursor: default; }}
      #tiTradesPagerInfo {{ font-size: 13px; color: var(--text-muted); }}
      @media (max-width: 480px) {{
        .ti-trades-modal {{ border-radius: 12px 12px 0 0; max-height: 90vh; align-self: flex-end; width: 100%; }}
        #tiTradesOverlay {{ align-items: flex-end !important; }}
      }}
    </style>

    <script>
    (function() {{
      const TI_SEASON = {season};
      const TI_HAS_PREMIUM = {str(has_premium).lower()};
      let currentPage = 1;
      let paginationData = null;
      let currentTab = 'trending';
      let currentPos = 'ALL';

      loadTIPage(1);

      function loadTIPage(page) {{
        if (typeof page === 'string') {{
          if (page === 'prev' && currentPage > 1) {{
            page = currentPage - 1;
          }} else if (page === 'next' && paginationData && paginationData.has_next) {{
            page = currentPage + 1;
          }} else {{
            return;
          }}
        }}
        currentPage = page;
        document.getElementById('tiLoading').style.display = '';
        document.getElementById('tiGrid').style.display = 'none';
        document.getElementById('tiPagination').style.display = 'none';
        fetch('/api/trade-intel/trending?season=' + TI_SEASON + '&page=' + page)
          .then(r => r.json())
          .then(data => {{
            if (data.error) throw new Error(data.error);
            paginationData = data.pagination;
            document.getElementById('tiLoading').style.display = 'none';
            document.getElementById('tiGrid').style.display = '';
            updatePaginationControls();
            renderTI(data.players || []);
          }})
          .catch(() => {{
            document.getElementById('tiLoading').innerHTML =
              '<div style="color:var(--text-muted)">Trade data unavailable.</div>';
          }});
      }}

      function updatePaginationControls() {{
        if (!paginationData) return;
        const prevBtn = document.getElementById('tiPrevBtn');
        const nextBtn = document.getElementById('tiNextBtn');
        const pageNumbers = document.getElementById('tiPageNumbers');
        const paginationText = document.getElementById('tiPaginationText');
        prevBtn.disabled = !paginationData.has_prev;
        nextBtn.disabled = !paginationData.has_next;
        const start = (paginationData.current_page - 1) * paginationData.per_page + 1;
        const end = Math.min(paginationData.current_page * paginationData.per_page, paginationData.total_players);
        paginationText.textContent = `Showing ${{start}}-${{end}} of ${{paginationData.total_players}} players`;
        pageNumbers.innerHTML = '';
        const maxPages = 5;
        let startPage = Math.max(1, paginationData.current_page - Math.floor(maxPages / 2));
        let endPage = Math.min(paginationData.total_pages, startPage + maxPages - 1);
        if (endPage - startPage < maxPages - 1) startPage = Math.max(1, endPage - maxPages + 1);
        for (let i = startPage; i <= endPage; i++) {{
          const pageBtn = document.createElement('button');
          pageBtn.className = 'ti-page-number' + (i === paginationData.current_page ? ' active' : '');
          pageBtn.textContent = i;
          pageBtn.onclick = () => loadTIPage(i);
          pageNumbers.appendChild(pageBtn);
        }}
        document.getElementById('tiPagination').style.display = 'flex';
      }}

      window.switchTITab = function(tab) {{
        currentTab = tab;
        document.querySelectorAll('.ti-tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
        renderTI();
      }};

      window.filterTI = function(pos) {{
        currentPos = pos;
        document.querySelectorAll('.ti-pos').forEach(b => b.classList.toggle('active', b.dataset.pos === pos));
        loadTIPage(currentPage);
      }};

      function renderTI(players = null) {{
        if (!players) {{ loadTIPage(currentPage); return; }}
        let filteredPlayers = currentPos === 'ALL' ? players : players.filter(p => p.position === currentPos);
        if (currentTab !== 'trending') {{
          const withDelta = filteredPlayers.filter(p => p.value_delta != null && p.model_value > 0);
          if (currentTab === 'buylows') {{
            filteredPlayers = withDelta.filter(p => p.value_delta < -5).sort((a, b) => a.value_delta - b.value_delta);
          }} else if (currentTab === 'sellhigh') {{
            filteredPlayers = withDelta.filter(p => p.value_delta > 5).sort((a, b) => b.value_delta - a.value_delta);
          }}
        }}
        const grid  = document.getElementById('tiGrid');
        const empty = document.getElementById('tiEmpty');
        if (filteredPlayers.length === 0) {{ grid.style.display = 'none'; empty.style.display = ''; return; }}
        empty.style.display = 'none';
        grid.style.display = '';
        const FREE_LIMIT = 5;
        const displayPlayers = TI_HAS_PREMIUM ? filteredPlayers : filteredPlayers.slice(0, FREE_LIMIT);
        const showPaywallCard = !TI_HAS_PREMIUM && filteredPlayers.length > FREE_LIMIT;
        grid.innerHTML = displayPlayers.map(p => {{
          const name   = p.name || 'Unknown';
          const pos    = p.position || '?';
          const team   = p.team || '?';
          const cnt7   = p.trade_count_7d  || 0;
          const cnt30  = p.trade_count_30d || 0;
          const cntAll = p.trade_count_all || 0;
          const market = p.market_value != null ? p.market_value.toFixed(1) : '—';
          const model  = p.model_value  != null ? p.model_value.toFixed(1)  : '—';
          const delta  = p.value_delta;
          const trend  = p.market_trend;
          let chipBg, chipColor, chipText;
          if (currentTab === 'trending') {{
            chipBg = '#3b82f620'; chipColor = '#3b82f6'; chipText = cntAll + ' trades';
          }} else if (currentTab === 'buylows') {{
            chipBg = '#10b98120'; chipColor = '#10b981';
            chipText = delta != null ? (delta > 0 ? '+' : '') + Math.round(delta) : '—';
          }} else {{
            chipBg = '#f59e0b20'; chipColor = '#f59e0b';
            chipText = delta != null ? (delta > 0 ? '+' : '') + Math.round(delta) : '—';
          }}
          const deltaHtml = delta != null
            ? `<span class="${{delta >= 0 ? 'ti-delta-pos' : 'ti-delta-neg'}}">${{delta >= 0 ? '+' : ''}}${{Math.round(delta)}}</span>`
            : '<span style="color:var(--text-muted)">—</span>';
          let momentumHtml = '';
          if (trend != null) {{
            if (trend >= 5) momentumHtml = '<span style="color:#10b981;">▲</span> Rising';
            else if (trend <= -5) momentumHtml = '<span style="color:#ef4444;">▼</span> Falling';
          }}
          const player_json = JSON.stringify(p).replace(/&/g, '&amp;').replace(/"/g, '&quot;');
          return `<div class="ti-card" data-player="${{player_json}}" onclick="openTITradesModal(JSON.parse(this.dataset.player))">
            <div class="ti-card-top">
              <div>
                <div class="ti-name">${{name}}</div>
                <div class="ti-meta">${{pos}} · ${{team}}</div>
              </div>
              <div class="ti-chip" style="background:${{chipBg}};color:${{chipColor}};">${{chipText}}</div>
            </div>
            <div class="ti-divider"></div>
            <div class="ti-row"><span class="ti-row-label">Market</span><span class="ti-row-val">${{market}}</span></div>
            <div class="ti-row"><span class="ti-row-label">BR Model</span><span class="ti-row-val">${{model}}</span></div>
            <div class="ti-row"><span class="ti-row-label">Delta</span><span class="ti-row-val">${{deltaHtml}}</span></div>
            <div class="ti-row"><span class="ti-row-label">Trades 7d/30d</span><span class="ti-row-val">${{cnt7}} / ${{cnt30}}</span></div>
            ${{momentumHtml ? `<div class="ti-momentum">${{momentumHtml}}</div>` : ''}}
          </div>`;
        }}).join('') + (showPaywallCard ? `
          <div class="ti-card" onclick="showPaywall('trade-history')" style="cursor:pointer;border:2px dashed var(--border);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;min-height:160px;background:var(--card);">
            <i class="fa-solid fa-lock" style="font-size:22px;color:var(--text-muted);"></i>
            <div style="font-weight:700;font-size:14px;">Unlock Full Access</div>
            <div style="font-size:12px;color:var(--text-muted);text-align:center;">See all players &amp; trade history<br>with a premium subscription</div>
            <span style="font-size:11px;font-weight:700;padding:4px 12px;background:linear-gradient(135deg,#667eea,#764ba2);color:white;border-radius:12px;">Upgrade &rarr;</span>
          </div>` : '');
      }}

      const _tiTrades = {{ player: null, page: 1, leagueFilter: 'all', total: 0, totalPages: 1 }};

      window.openTITradesModal = function(playerData) {{
        if (!TI_HAS_PREMIUM) {{ showPaywall('trade-history'); return; }}
        _tiTrades.player = playerData;
        _tiTrades.page = 1;
        _tiTrades.leagueFilter = 'all';
        document.getElementById('tiTradesName').textContent = playerData.name || 'Player';
        const pos  = playerData.position || '';
        const team = playerData.team || '';
        const cnt  = playerData.trade_count_all;
        const cntTxt = cnt ? ` · ${{cnt}} trades tracked` : '';
        document.getElementById('tiTradesMeta').textContent = [pos, team].filter(Boolean).join(' · ') + cntTxt;
        document.querySelectorAll('.ti-lf-btn').forEach(b => b.classList.toggle('active', b.dataset.lf === 'all'));
        const overlay = document.getElementById('tiTradesOverlay');
        overlay.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        _loadTITrades(1);
      }};

      window.closeTITradesModal = function() {{
        document.getElementById('tiTradesOverlay').style.display = 'none';
        document.body.style.overflow = '';
      }};

      window.viewTIPlayerProfile = function() {{
        const p = _tiTrades.player;
        if (!p) return;
        closeTITradesModal();
        if (p.is_rookie && p.is_rookie !== 'False') {{
          rkOpenModal(p);
        }} else {{
          const name = (p.name || '').replace(/'/g, "\\'");
          openPlayerModal(p.player_id, name);
        }}
      }};

      window.switchTILF = function(lf) {{
        _tiTrades.leagueFilter = lf;
        document.querySelectorAll('.ti-lf-btn').forEach(b => b.classList.toggle('active', b.dataset.lf === lf));
        _loadTITrades(1);
      }};

      window.prevTITrades = function() {{ if (_tiTrades.page > 1) _loadTITrades(_tiTrades.page - 1); }};
      window.nextTITrades = function() {{ if (_tiTrades.page < _tiTrades.totalPages) _loadTITrades(_tiTrades.page + 1); }};

      function _loadTITrades(page) {{
        const p = _tiTrades.player;
        if (!p) return;
        _tiTrades.page = page;
        document.getElementById('tiTradesBody').innerHTML = '<div class="ti-trades-msg">Loading&hellip;</div>';
        document.getElementById('tiTradesPager').style.display = 'none';
        const qs = new URLSearchParams({{ season: TI_SEASON, league_type: _tiTrades.leagueFilter, page, limit: 15 }});
        fetch(`/api/trade-intel/player-trades/${{p.player_id}}?${{qs}}`)
          .then(r => {{ if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); }})
          .then(_renderTITrades)
          .catch(() => {{
            document.getElementById('tiTradesBody').innerHTML =
              '<div class="ti-trades-msg">Failed to load trades.</div>';
          }});
      }}

      function _renderTITrades(data) {{
        const body = document.getElementById('tiTradesBody');
        _tiTrades.total = data.total || 0;
        _tiTrades.totalPages = data.total_pages || 1;
        if (!data.trades || data.trades.length === 0) {{
          body.innerHTML = '<div class="ti-trades-msg">No trades found for this filter.</div>';
          return;
        }}
        function assetHtml(a) {{
          if (a.type === 'pick') return `<div class="ti-trade-asset pick">${{a.name}}</div>`;
          const posTag = a.position && a.position !== '?' ? ` <span style="font-size:11px;opacity:.6;">${{a.position}}</span>` : '';
          const cls = a.is_focus ? 'focus' : 'other';
          return `<div class="ti-trade-asset ${{cls}}">${{a.name}}${{posTag}}</div>`;
        }}
        body.innerHTML = data.trades.map(t => {{
          const sideA = (t.side_a || []).map(assetHtml).join('');
          const sideB = (t.side_b || []).map(assetHtml).join('');
          const fmt   = t.is_superflex ? 'SF' : t.is_superflex === false ? '1QB' : '';
          const teams = t.num_teams ? `${{t.num_teams}}-team` : '';
          const ctx   = [teams, fmt].filter(Boolean).join(' ');
          const meta  = [t.date, ctx].filter(Boolean).join(' · ');
          return `<div class="ti-trade-item">
            <div class="ti-trade-date">${{meta}}</div>
            <div class="ti-trade-sides">
              <div><div class="ti-trade-side-label">Side A</div>${{sideA}}</div>
              <div class="ti-trade-arrow">&#x21C4;</div>
              <div><div class="ti-trade-side-label">Side B</div>${{sideB}}</div>
            </div>
          </div>`;
        }}).join('');
        if (_tiTrades.totalPages > 1 || _tiTrades.total > 0) {{
          document.getElementById('tiTradesPager').style.display = 'flex';
          document.getElementById('tiTradesPrev').disabled = !data.has_prev;
          document.getElementById('tiTradesNext').disabled = !data.has_next;
          document.getElementById('tiTradesPagerInfo').textContent =
            `Page ${{data.page}} of ${{data.total_pages}} · ${{data.total}} trades`;
        }}
      }}

      window.loadTIPage = loadTIPage;
    }})();
    </script>
    """
    return render_page("Trade Intelligence", league_id, "trade-intel", body_html, platform, season)


@trade_bp.route("/trade-intel")
def page_trade_intel_guest():
    from app import get_nfl_state
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    return page_trade_intel(platform="sleeper", season=current_season, league_id=None)


# ── Trade Database ─────────────────────────────────────────────────────────────

@trade_bp.route("/<platform>/<int:season>/<league_id>/trade-database")
def page_trade_database(platform: str, season: int, league_id: str):
    from app import render_page
    body_html = f"""
    <div class="card central" style="max-width:960px;">
      <div class="card-header" style="border-bottom:1px solid var(--border);padding-bottom:16px;margin-bottom:0;">
        <h2 style="margin:0 0 4px;font-size:20px;">Trade Database</h2>
        <div style="font-size:13px;color:var(--text-muted);">
          Explore thousands of real dynasty trades to understand player values and market trends
        </div>
      </div>
      <div class="card-body" style="padding-top:20px;">

        <div class="tdb-toolbar">
          <div class="tdb-search-wrap">
            <span class="tdb-search-icon" aria-hidden="true"></span>
            <input id="tdbSearch" type="text" placeholder="Search by player name..." class="tdb-search">
          </div>
          <div class="tdb-lt-filters">
            <button class="tdb-lt active" data-lt="all" onclick="tdbFilter('all')">All</button>
            <button class="tdb-lt" data-lt="1qb" onclick="tdbFilter('1qb')">1QB</button>
            <button class="tdb-lt" data-lt="sf"  onclick="tdbFilter('sf')">SF</button>
          </div>
        </div>

        <div id="tdbStatus" class="tdb-status"></div>
        <div id="tdbList"   class="tdb-list"></div>

        <div id="tdbLoading" style="text-align:center;padding:48px 0;color:var(--text-muted);display:none;">
          <div class="loading-spinner" style="margin:0 auto 12px;"></div>
          Loading trade data...
        </div>

        <div id="tdbPagination" class="ti-pagination" style="display:none;">
          <div class="ti-pagination-info">
            <span id="tdbPaginationText">Showing 1-20 of 100 trades</span>
          </div>
          <div class="ti-pagination-controls">
            <button id="tdbPrevBtn" class="ti-pagination-btn" onclick="loadTDBPage('prev')" disabled>
              <i class="fa-solid fa-chevron-left"></i> Previous
            </button>
            <div id="tdbPageNumbers" class="ti-page-numbers"></div>
            <button id="tdbNextBtn" class="ti-pagination-btn" onclick="loadTDBPage('next')" disabled>
              Next <i class="fa-solid fa-chevron-right"></i>
            </button>
          </div>
        </div>

      </div>
    </div>

    <style>
      .tdb-toolbar {{
        display: flex; gap: 12px; margin-bottom: 16px;
        flex-wrap: wrap; align-items: center;
      }}
      .tdb-search-wrap {{
        flex: 1; min-width: 200px;
        display: flex; align-items: center;
        border: 1px solid var(--border); border-radius: 8px;
        background: var(--card); padding: 0 12px; gap: 8px;
      }}
      .tdb-search-icon {{
        display: inline-block; width: 14px; height: 14px; flex-shrink: 0;
        background: url('/static/images/magnifying-glass-solid.png') no-repeat center / contain;
        filter: brightness(0) saturate(100%) invert(60%) sepia(0%) saturate(0%) hue-rotate(0deg) brightness(85%) contrast(90%);
        pointer-events: none;
      }}
      .tdb-search {{
        flex: 1; padding: 9px 0; border: none; background: transparent;
        color: var(--text); font-size: 14px; outline: none; min-width: 0;
      }}
      .tdb-search-wrap:focus-within {{ border-color: var(--accent-color, #3b82f6); }}
      .tdb-lt-filters {{ display: flex; gap: 4px; }}
      .tdb-lt {{
        padding: 7px 14px; border-radius: 8px; border: 1px solid var(--border);
        background: var(--card); color: var(--text-muted); cursor: pointer;
        font-size: 13px; font-weight: 600; transition: all .15s;
      }}
      .tdb-lt.active {{
        background: var(--text-color); color: var(--card-bg); border-color: var(--text-color);
      }}
      .tdb-status {{ font-size: 12px; color: var(--text-muted); margin-bottom: 14px; min-height: 16px; }}
      .tdb-list {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
      @media(max-width: 600px) {{ .tdb-list {{ grid-template-columns: 1fr; }} }}
      .tdb-card {{
        border: 1px solid var(--border-color); border-radius: 12px;
        overflow: hidden; background: var(--card-bg);
      }}
      .tdb-card-head {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 14px; border-bottom: 1px solid var(--border-color);
        background: var(--bg-alt, rgba(0,0,0,.03));
      }}
      .tdb-card-date {{ font-size: 11px; color: var(--text-muted); font-weight: 500; }}
      .tdb-badges {{ display: flex; gap: 5px; flex-wrap: wrap; }}
      .tdb-badge {{
        font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 8px;
        background: var(--row, #1e293b); color: var(--text);
        border: 1px solid var(--border-color);
      }}
      .tdb-badge-sf {{ background: #7c3aed22; color: #a78bfa; border-color: #7c3aed44; }}
      .tdb-card-body {{ display: grid; grid-template-columns: 1fr 1px 1fr; }}
      .tdb-col {{ padding: 12px 14px; display: flex; flex-direction: column; gap: 5px; }}
      .tdb-col-divider {{ background: var(--border-color); }}
      .tdb-asset {{
        font-size: 14px; color: var(--text); font-weight: 500;
        display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
      }}
      .tdb-asset.tdb-match {{ font-weight: 800; color: var(--accent-color, #3b82f6); }}
      .tdb-asset.tdb-pick {{ color: var(--text-muted); font-size: 14px; font-weight: 500; }}
      .tdb-pos {{
        font-size: 10px; font-weight: 700; padding: 1px 5px; border-radius: 4px;
        background: var(--row, #1e293b); color: var(--text); flex-shrink: 0;
      }}
      .tdb-vol {{
        font-size: 10px; font-weight: 600; padding: 1px 5px; border-radius: 4px;
        background: #3b82f615; color: #3b82f6; border: 1px solid #3b82f630; flex-shrink: 0;
      }}
      @media(max-width: 480px) {{
        .tdb-card-body {{ grid-template-columns: 1fr; }}
        .tdb-col-divider {{ height: 1px; width: auto; }}
      }}
    </style>

    <script>
    (function() {{
      const TDB_SEASON = {season};
      let currentPage = 1;
      let paginationData = null;
      let leagueType = 'all';
      let searchQuery = '';
      let loading = false;

      const listEl   = document.getElementById('tdbList');
      const statusEl = document.getElementById('tdbStatus');
      const searchEl = document.getElementById('tdbSearch');

      loadTDBPage(1);

      function loadTDBPage(page) {{
        if (loading) return;
        if (typeof page === 'string') {{
          if (page === 'prev' && currentPage > 1) page = currentPage - 1;
          else if (page === 'next' && paginationData && paginationData.has_next) page = currentPage + 1;
          else return;
        }}
        currentPage = page;
        loading = true;
        statusEl.textContent = '';
        listEl.style.display = 'none';
        document.getElementById('tdbLoading').style.display = '';
        document.getElementById('tdbPagination').style.display = 'none';
        const apiPage = page - 1;
        const params = new URLSearchParams({{ page: apiPage, limit: 20, league_type: leagueType, season: TDB_SEASON }});
        if (searchQuery) params.set('q', searchQuery);
        fetch('/api/trade-database?' + params)
          .then(r => r.json())
          .then(data => {{
            if (data.error) throw new Error(data.error);
            const trades = data.trades || [];
            document.getElementById('tdbLoading').style.display = 'none';
            if (trades.length === 0) {{
              listEl.innerHTML = '<div style="color:var(--text-muted);padding:20px 0;text-align:center;grid-column:1/-1;">No trades found.</div>';
              statusEl.textContent = '';
              document.getElementById('tdbPagination').style.display = 'none';
              loading = false;
              return;
            }}
            paginationData = data.pagination;
            statusEl.textContent = '';
            listEl.style.display = '';
            updateTDBPaginationControls();
            renderTDBTrades(trades);
            loading = false;
          }})
          .catch(err => {{
            console.error('Error loading trades:', err);
            document.getElementById('tdbLoading').style.display = 'none';
            statusEl.textContent = 'Error loading trades';
            loading = false;
          }});
      }}

      function updateTDBPaginationControls() {{
        if (!paginationData) return;
        const prevBtn = document.getElementById('tdbPrevBtn');
        const nextBtn = document.getElementById('tdbNextBtn');
        const pageNumbers = document.getElementById('tdbPageNumbers');
        const paginationText = document.getElementById('tdbPaginationText');
        prevBtn.disabled = !paginationData.has_prev;
        nextBtn.disabled = !paginationData.has_next;
        const start = (paginationData.current_page - 1) * paginationData.per_page + 1;
        const end = Math.min(paginationData.current_page * paginationData.per_page, paginationData.total_players);
        paginationText.textContent = `Showing ${{start}}-${{end}} of ${{paginationData.total_players}} trades`;
        pageNumbers.innerHTML = '';
        const maxPages = 5;
        let startPage = Math.max(1, paginationData.current_page - Math.floor(maxPages / 2));
        let endPage = Math.min(paginationData.total_pages, startPage + maxPages - 1);
        if (endPage - startPage < maxPages - 1) startPage = Math.max(1, endPage - maxPages + 1);
        for (let i = startPage; i <= endPage; i++) {{
          const pageBtn = document.createElement('button');
          pageBtn.className = 'ti-page-number' + (i === paginationData.current_page ? ' active' : '');
          pageBtn.textContent = i;
          pageBtn.onclick = () => loadTDBPage(i);
          pageNumbers.appendChild(pageBtn);
        }}
        document.getElementById('tdbPagination').style.display = 'flex';
      }}

      function renderTDBTrades(trades) {{
        listEl.innerHTML = '';
        const lq = (searchQuery || '').toLowerCase();
        trades.forEach(t => {{
          const sfBadge    = t.is_superflex === true  ? '<span class="tdb-badge tdb-badge-sf">SF</span>'
                           : t.is_superflex === false ? '<span class="tdb-badge">1QB</span>' : '';
          const teamsBadge = t.num_teams    ? `<span class="tdb-badge">${{t.num_teams}} Teams</span>` : '';
          const scoreBadge = t.scoring_type ? `<span class="tdb-badge">${{t.scoring_type.toUpperCase()}}</span>` : '';
          function renderAsset(a) {{
            const match = lq && a.name && a.name.toLowerCase().includes(lq);
            const pickCls = a.type === 'pick' ? ' tdb-pick' : '';
            const cls = 'tdb-asset' + pickCls + (match ? ' tdb-match' : '');
            const pos = a.position && a.type === 'player' ? `<span class="tdb-pos">${{a.position}}</span>` : '';
            const vol = a.trade_count && a.type === 'player' ? `<span class="tdb-vol">${{a.trade_count}} trades</span>` : '';
            return `<div class="${{cls}}">${{a.name}}${{pos}}${{vol}}</div>`;
          }}
          const sideA = (t.side_a || []).map(renderAsset).join('') || '<div class="tdb-asset" style="color:var(--text-muted)">—</div>';
          const sideB = (t.side_b || []).map(renderAsset).join('') || '<div class="tdb-asset" style="color:var(--text-muted)">—</div>';
          const card = document.createElement('div');
          card.className = 'tdb-card';
          card.innerHTML = `
            <div class="tdb-card-head">
              <span class="tdb-card-date">${{t.date || '—'}}</span>
              <div class="tdb-badges">${{sfBadge}}${{teamsBadge}}${{scoreBadge}}</div>
            </div>
            <div class="tdb-card-body">
              <div class="tdb-col">${{sideA}}</div>
              <div class="tdb-col-divider"></div>
              <div class="tdb-col">${{sideB}}</div>
            </div>`;
          listEl.appendChild(card);
        }});
      }}

      window.tdbFilter = function(lt) {{
        leagueType = lt;
        document.querySelectorAll('.tdb-lt').forEach(b => b.classList.toggle('active', b.dataset.lt === lt));
        loadTDBPage(1);
      }};

      let debounce;
      searchEl.addEventListener('input', () => {{
        clearTimeout(debounce);
        debounce = setTimeout(() => {{ searchQuery = searchEl.value.trim(); loadTDBPage(1); }}, 350);
      }});

      window.loadTDBPage = loadTDBPage;
    }})();
    </script>
    """
    return render_page("Trade Database", league_id, "trade-database", body_html, platform, season)


@trade_bp.route("/trade-database")
def page_trade_database_guest():
    from app import get_nfl_state
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    return page_trade_database(platform="sleeper", season=current_season, league_id=None)
