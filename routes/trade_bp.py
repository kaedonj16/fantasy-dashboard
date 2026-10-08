"""
Trade-related page routes.

Routes: /trade, /trade-intel, /trade-database
Also handles: /<platform>/<season>/<league_id>/trade|trade-intel|trade-database
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from flask import Blueprint, redirect, session

from dashboard_services.subscriptions import has_premium_access, has_premium_for_viewer

logger = logging.getLogger(__name__)

trade_bp = Blueprint("trade", __name__)


# Crawlable explanatory content shown beneath the public trade calculator. Gives
# search engines (and AdSense reviewers) real text to index instead of a bare widget.
_TRADE_CALCULATOR_SEO_CONTENT = """
    <div style="background:var(--card);border:1px solid var(--border);border-radius:14px;margin-top:18px;padding:22px 24px;">

      <!-- Section bridge header -->
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:20px;">
        <i class="fa-solid fa-circle-info" style="color:var(--text-muted);font-size:13px;"></i>
        <span style="font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:var(--text-muted);">About This Calculator</span>
      </div>

      <div>

        <!-- How it works: 3-step strip -->
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin-bottom:28px;">
          <div style="background:var(--bg-alt,rgba(0,0,0,.03));border:1px solid var(--border);border-radius:12px;padding:16px 18px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
              <span style="width:28px;height:28px;border-radius:8px;background:#3b82f620;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                <i class="fa-solid fa-plus" style="color:#3b82f6;font-size:13px;"></i>
              </span>
              <span style="font-size:13px;font-weight:700;">1. Add Players</span>
            </div>
            <p style="font-size:13px;color:var(--text-muted);margin:0;line-height:1.55;">
              Search for players and picks on each side of the deal using your league's scoring format.
            </p>
          </div>
          <div style="background:var(--bg-alt,rgba(0,0,0,.03));border:1px solid var(--border);border-radius:12px;padding:16px 18px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
              <span style="width:28px;height:28px;border-radius:8px;background:#8b5cf620;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                <i class="fa-solid fa-scale-balanced" style="color:#8b5cf6;font-size:13px;"></i>
              </span>
              <span style="font-size:13px;font-weight:700;">2. Compare Values</span>
            </div>
            <p style="font-size:13px;color:var(--text-muted);margin:0;line-height:1.55;">
              Values are built from thousands of real Sleeper dynasty trades, not guesses.
              They still apply to ESPN, Yahoo, and MFL rosters.
            </p>
          </div>
          <div style="background:var(--bg-alt,rgba(0,0,0,.03));border:1px solid var(--border);border-radius:12px;padding:16px 18px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
              <span style="width:28px;height:28px;border-radius:8px;background:#10b98120;display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                <i class="fa-solid fa-flag-checkered" style="color:#10b981;font-size:13px;"></i>
              </span>
              <span style="font-size:13px;font-weight:700;">3. Get a Verdict</span>
            </div>
            <p style="font-size:13px;color:var(--text-muted);margin:0;line-height:1.55;">
              The calculator shows which side wins and by how much, updated daily as the market shifts.
            </p>
          </div>
        </div>

        <!-- Dynasty vs Redraft: two-column -->
        <div style="border-top:1px solid var(--border);padding-top:22px;margin-bottom:22px;">
          <h2 style="font-size:15px;font-weight:700;margin:0 0 14px;display:flex;align-items:center;gap:8px;">
            <i class="fa-solid fa-arrow-right-arrow-left" style="color:var(--text-muted);font-size:13px;"></i>
            Dynasty vs. Redraft Trade Values
          </h2>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
            <div style="border:1px solid var(--border);border-radius:10px;padding:14px 16px;">
              <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#3b82f6;margin-bottom:6px;">Redraft</div>
              <p style="font-size:13px;color:var(--text-muted);margin:0;line-height:1.6;">
                Only this season matters. Proven veterans and high-floor producers carry the most value.
                Age is a strength, not a liability.
              </p>
            </div>
            <div style="border:1px solid var(--border);border-radius:10px;padding:14px 16px;">
              <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#8b5cf6;margin-bottom:6px;">Dynasty</div>
              <p style="font-size:13px;color:var(--text-muted);margin:0;line-height:1.6;">
                Rosters carry over year after year. Youth, long-term upside, and rookie picks
                weigh much more heavily than near-term production.
              </p>
            </div>
          </div>
          <p style="font-size:13px;color:var(--text-muted);margin:10px 0 0;line-height:1.6;">
            The calculator also supports <strong>superflex</strong> scoring, where quarterbacks jump
            dramatically in value because you can start two of them.
          </p>
        </div>

        <!-- Trade tips: 2x2 grid -->
        <div style="border-top:1px solid var(--border);padding-top:22px;margin-bottom:22px;">
          <h2 style="font-size:15px;font-weight:700;margin:0 0 14px;display:flex;align-items:center;gap:8px;">
            <i class="fa-solid fa-lightbulb" style="color:var(--text-muted);font-size:13px;"></i>
            Tips for Evaluating a Trade
          </h2>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
            <div style="display:flex;gap:10px;align-items:flex-start;">
              <span style="width:28px;height:28px;border-radius:7px;background:#f59e0b35;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;">
                <i class="fa-solid fa-layer-group" style="color:#f59e0b;font-size:13px;"></i>
              </span>
              <p style="font-size:13px;color:var(--text-muted);margin:0;line-height:1.6;">
                Don't trade purely by total value, roster construction matters. Two solid
                starters often beat one star plus a bench piece.
              </p>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start;">
              <span style="width:28px;height:28px;border-radius:7px;background:#ef444435;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;">
                <i class="fa-solid fa-gem" style="color:#ef4444;font-size:13px;"></i>
              </span>
              <p style="font-size:13px;color:var(--text-muted);margin:0;line-height:1.6;">
                Account for positional scarcity. An elite tight end or superflex QB is harder
                to replace than a mid-tier running back.
              </p>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start;">
              <span style="width:28px;height:28px;border-radius:7px;background:#3b82f635;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;">
                <i class="fa-solid fa-clock-rotate-left" style="color:#3b82f6;font-size:13px;"></i>
              </span>
              <p style="font-size:13px;color:var(--text-muted);margin:0;line-height:1.6;">
                In dynasty, weigh your timeline. Contenders should pay a premium for win-now
                talent; rebuilders should bank youth and picks.
              </p>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start;">
              <span style="width:28px;height:28px;border-radius:7px;background:#10b98135;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px;">
                <i class="fa-solid fa-magnifying-glass-chart" style="color:#10b981;font-size:13px;"></i>
              </span>
              <p style="font-size:13px;color:var(--text-muted);margin:0;line-height:1.6;">
                Use real trade comparisons to sanity-check a deal, if similar trades have
                happened before, you'll see how the market actually valued them.
              </p>
            </div>
          </div>
        </div>

        <!-- FAQ accordion -->
        <div style="border-top:1px solid var(--border);padding-top:22px;">
          <h2 style="font-size:15px;font-weight:700;margin:0 0 14px;display:flex;align-items:center;gap:8px;">
            <i class="fa-solid fa-circle-question" style="color:var(--text-muted);font-size:13px;"></i>
            Frequently Asked Questions
          </h2>
          <div class="tc-faq">
            <details class="tc-faq-item">
              <summary class="tc-faq-q">Is the trade calculator free?</summary>
              <p class="tc-faq-a">
                Yes. The trade calculator and player trade values are free to use, no account
                required. Connecting your Sleeper, ESPN, or Yahoo league unlocks personalized
                analysis tailored to your roster and scoring settings.
              </p>
            </details>
            <details class="tc-faq-item">
              <summary class="tc-faq-q">How often are player values updated?</summary>
              <p class="tc-faq-a">
                Values refresh daily based on the latest real trades, news, and expert rankings, so
                they reflect the current market rather than a static preseason list.
              </p>
            </details>
            <details class="tc-faq-item">
              <summary class="tc-faq-q">Does it support superflex and tight-end premium?</summary>
              <p class="tc-faq-a">
                Yes. Toggle superflex to value quarterbacks correctly for two-QB formats, and switch
                between PPR, half-PPR, and standard scoring to match your league settings.
              </p>
            </details>
          </div>
        </div>

      </div>
    </div>

    <style>
      .tc-faq { display: flex; flex-direction: column; gap: 2px; }
      .tc-faq-item {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
      }
      .tc-faq-q {
        padding: 12px 16px;
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
        list-style: none;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
        user-select: none;
        background: var(--card);
        transition: background .12s;
      }
      .tc-faq-q::-webkit-details-marker { display: none; }
      .tc-faq-q::after {
        content: '+';
        font-size: 18px;
        font-weight: 400;
        color: var(--text-muted);
        flex-shrink: 0;
        line-height: 1;
      }
      details.tc-faq-item[open] .tc-faq-q::after { content: '\2212'; }
      .tc-faq-q:hover { background: var(--bg-alt, rgba(0,0,0,.03)); }
      .tc-faq-a {
        font-size: 13px;
        color: var(--text-muted);
        line-height: 1.65;
        margin: 0;
        padding: 12px 16px 14px;
        background: var(--card);
        border-top: 1px solid var(--border);
      }
      @media (max-width: 560px) {
        .tc-tips-grid { grid-template-columns: 1fr !important; }
        .tc-format-grid { grid-template-columns: 1fr !important; }
      }
    </style>
"""


# ── Trade Calculator ───────────────────────────────────────────────────────────

@trade_bp.route("/trade")
@trade_bp.route("/<platform>/<int:season>/<league_id>/trade")
def page_trade(platform: Optional[str] = None, season: Optional[int] = None,
               league_id: Optional[str] = None):
    from app import (
        build_trade_calculator_body, get_league_ctx_from_cache,
        get_nfl_state, get_viewer_session_for_league, render_page,
        _league_is_redraft,
    )
    user_id = session.get("viewer_username") or None
    # Redirect to league-specific URL when user is logged in but hit the public /trade path
    if not league_id and user_id:
        _lid = session.get("last_league_id") or session.get("viewer_league_id") or None
        _plt = session.get("last_platform") or session.get("viewer_platform") or "sleeper"
        _ssn = session.get("last_season") or session.get("viewer_season") or None
        if _lid and _ssn:
            from flask import redirect as _redir
            return _redir(f"/{_plt}/{_ssn}/{_lid}/trade", 302)
    if league_id:
        ctx = get_league_ctx_from_cache(platform, league_id, season)
        league_id_safe = ctx.get("league_id") or league_id
        season_safe = int(ctx.get("season") or season or datetime.now().year)
        num_teams = ctx.get("total_rosters") or None
        _ss = ctx.get("scoring_settings") or {}
        rec = float(_ss.get("rec") or 0)
        scoring_format = "ppr" if rec >= 1.0 else "half" if rec >= 0.5 else "std"
        # Auto-apply the league's TE-premium bonus (Sleeper bonus_rec_te) so TE values
        # reflect the actual scoring without the user having to set it manually.
        te_premium = float(_ss.get("bonus_rec_te") or 0)
        viewer = get_viewer_session_for_league(
            ctx.get("users") or [], ctx.get("rosters") or [],
            platform, league_id, season,
        )
        viewer_roster_id = viewer.get("viewer_roster_id") or ""
        has_premium = has_premium_for_viewer(user_id, session.get("viewer_user_id"), league_id, platform or "sleeper", season)
        _rp = ctx.get("roster_positions") or []
        from utils.lineup_slots import is_superflex_lineup
        _is_sf = is_superflex_lineup(_rp)
        scoring_type = "redraft" if _league_is_redraft(ctx) else "dynasty"
        body = build_trade_calculator_body(league_id_safe, season_safe, num_teams=num_teams,
                                           scoring_format=scoring_format,
                                           viewer_roster_id=viewer_roster_id,
                                           has_premium=has_premium,
                                           is_superflex=_is_sf,
                                           te_premium=te_premium,
                                           platform=platform,
                                           scoring_type=scoring_type)
    else:
        state = get_nfl_state() or {}
        current_season = int(state.get("season") or datetime.now().year)
        has_premium = has_premium_access(user_id, None, "sleeper")
        body = build_trade_calculator_body(None, current_season, has_premium=has_premium,
                                           seo_footer=_TRADE_CALCULATOR_SEO_CONTENT)

    return render_page(
        "Fantasy Football Trade Calculator - Dynasty & Redraft Trade Values | BR Fantasy",
        league_id, "trade", body, platform, season,
        description=(
            "Free fantasy football trade calculator. Compare any trade with dynasty and "
            "redraft player values built from thousands of real Sleeper trades. "
            "Get instant verdicts, superflex values, and pick valuations."
        ),
    )


# ── Trade Intelligence ─────────────────────────────────────────────────────────

from dashboard_services.pages.trade_intel_page import _TI_SOURCE_NOTE


@trade_bp.route("/<platform>/<int:season>/<league_id>/trade-intel")
def page_trade_intel(platform: str, season: int, league_id: str):
    """Trade Intel now lives in the Trade Hub (Market Intel tab)."""
    return redirect(f"/{platform}/{season}/{league_id}/trade?tab=suggestions", code=302)


@trade_bp.route("/trade-intel")
def page_trade_intel_guest():
    """Trade Intel now lives in the Trade Hub (Market Intel tab)."""
    return redirect("/trade?tab=suggestions", code=302)


# ── Trade Database ─────────────────────────────────────────────────────────────

def _trade_db_default_format(platform: str, season: int, league_id: Optional[str]) -> str:
    """Default the Trade DB dynasty/redraft toggle from the open league."""
    if not league_id:
        return "all"
    try:
        from app import _league_is_redraft, get_league_ctx_from_cache
        ctx = get_league_ctx_from_cache(platform, league_id, season) or {}
        if not ctx:
            return "all"
        return "redraft" if _league_is_redraft(ctx) else "dynasty"
    except Exception:
        logger.debug("trade-db default format lookup failed", exc_info=True)
        return "all"


@trade_bp.route("/<platform>/<int:season>/<league_id>/trade-database")
def page_trade_database(platform: str, season: int, league_id: str):
    from app import render_page
    _tdb_fmt = _trade_db_default_format(platform, season, league_id)
    _tdb_all = " active" if _tdb_fmt == "all" else ""
    _tdb_dyn = " active" if _tdb_fmt == "dynasty" else ""
    _tdb_rd = " active" if _tdb_fmt == "redraft" else ""
    body_html = f"""
    <div class="card central tdb-page-card">
      <div class="card-header tdb-page-header">
        <h2>Trade Database</h2>
        <div class="tdb-page-sub">
          Explore thousands of real Sleeper trades. Filter by dynasty or redraft to match the market you care about. {_TI_SOURCE_NOTE}
        </div>
      </div>
      <div class="card-body tdb-page-body">

        <div class="tdb-toolbar">
          <div class="tdb-sides-row tdb-search-card">
            <div class="tdb-side-wrap">
              <div class="tdb-side-label">Side A</div>
              <div class="tdb-search-outer">
                <input id="tdbSideASearch" type="text" placeholder="Search player or pick…" class="tdb-search" autocomplete="off">
                <div id="tdbSideADropdown" class="tdb-dropdown" style="display:none;"></div>
              </div>
              <div id="tdbSideAChip" class="tdb-chip-area" style="display:none;"></div>
            </div>
            <div class="tdb-side-sep">&#8646;</div>
            <div class="tdb-side-wrap">
              <div class="tdb-side-label">Side B</div>
              <div class="tdb-search-outer">
                <input id="tdbSideBSearch" type="text" placeholder="Search player or pick…" class="tdb-search" autocomplete="off">
                <div id="tdbSideBDropdown" class="tdb-dropdown" style="display:none;"></div>
              </div>
              <div id="tdbSideBChip" class="tdb-chip-area" style="display:none;"></div>
            </div>
          </div>
          <div class="tdb-filter-col">
            <div class="otc-day-filters tdb-lt-filters">
              <button class="otc-day-filter tdb-lt active" data-lt="all" onclick="tdbFilter('all')">All</button>
              <button class="otc-day-filter tdb-lt" data-lt="1qb" onclick="tdbFilter('1qb')">1QB</button>
              <button class="otc-day-filter tdb-lt" data-lt="sf"  onclick="tdbFilter('sf')">SF</button>
            </div>
            <div class="otc-day-filters tdb-lf-filters">
              <button class="otc-day-filter tdb-lf{_tdb_all}" data-lf="all" onclick="tdbFormatFilter('all')">All</button>
              <button class="otc-day-filter tdb-lf{_tdb_dyn}" data-lf="dynasty" onclick="tdbFormatFilter('dynasty')">Dynasty</button>
              <button class="otc-day-filter tdb-lf{_tdb_rd}" data-lf="redraft" onclick="tdbFormatFilter('redraft')">Redraft</button>
              <button class="otc-day-filter tdb-lf" data-lf="keeper" onclick="tdbFormatFilter('keeper')">Keeper</button>
            </div>
            <div class="tdb-sort-wrap">
              <label class="tdb-sort-label" for="tdbSort">Sort</label>
              <select id="tdbSort" class="tdb-sort" onchange="tdbSortChange(this.value)">
                <option value="date_desc">Newest first</option>
                <option value="date_asc">Oldest first</option>
                <option value="value_desc">Value: high to low</option>
                <option value="value_asc">Value: low to high</option>
              </select>
            </div>
          </div>
        </div>

        <div id="tdbStatus" class="tdb-status"></div>
        <div id="tdbList"   class="tdb-list"></div>

        <div id="tdbLoading" class="tdb-loading" style="display:none;">
          <div class="tdb-skeleton-grid" id="tdbSkeletonGrid"></div>
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
      /* Trade DB page layout: wider on desktop for the 2-col card grid */
      .tdb-page-card {{ max-width: 960px; }}
      @media(min-width: 1100px) {{ .tdb-page-card {{ max-width: 1100px; }} }}
      .tdb-page-header {{
        border-bottom: 1px solid var(--border);
        padding-bottom: 16px; margin-bottom: 0;
      }}
      .tdb-page-header h2 {{ margin: 0 0 4px; font-size: 22px; }}
      .tdb-page-sub {{ font-size: 13px; color: var(--text-muted); }}
      .tdb-page-body {{ padding-top: 20px; }}
      /* Skeleton loading for trade cards */
      .tdb-loading {{ padding: 8px 0; }}
      .tdb-skeleton-grid {{
        display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px;
      }}
      @media(max-width: 600px) {{ .tdb-skeleton-grid {{ grid-template-columns: 1fr; }} }}
      .tdb-skeleton-card {{
        border: 1px solid var(--border); border-radius: 12px;
        overflow: hidden; background: var(--card);
      }}
      .tdb-skeleton-head {{
        height: 34px; border-bottom: 1px solid var(--border);
        background: var(--bg-alt, rgba(0,0,0,.03));
      }}
      .tdb-skeleton-body {{ display: grid; grid-template-columns: 1fr 1px 1fr; min-height: 96px; }}
      .tdb-skeleton-col {{ padding: 12px 14px; display: flex; flex-direction: column; gap: 8px; }}
      .tdb-skeleton-line {{
        height: 14px; border-radius: 4px;
        background: linear-gradient(90deg, var(--bg-alt) 25%, var(--border) 50%, var(--bg-alt) 75%);
        background-size: 200% 100%;
        animation: tdb-skeleton-shimmer 1.2s infinite;
      }}
      .tdb-skeleton-line.short {{ width: 60%; }}
      @keyframes tdb-skeleton-shimmer {{
        0% {{ background-position: 200% 0; }}
        100% {{ background-position: -200% 0; }}
      }}
      .tdb-toolbar {{
        display: flex; gap: 12px; margin-bottom: 16px;
        flex-wrap: wrap; align-items: flex-start;
      }}
      .tdb-sides-row {{
        display: flex; gap: 12px; flex: 1; min-width: 0; flex-wrap: wrap;
        align-items: flex-start;
      }}
      /* Unified two-sided search card (matches approved mock) */
      .tdb-sides-row.tdb-search-card {{
        background: var(--card); border: 1px solid var(--border);
        border-radius: 14px; padding: 14px; align-items: stretch;
      }}
      .tdb-side-wrap {{
        flex: 1; min-width: 160px; display: flex; flex-direction: column; gap: 6px;
      }}
      .tdb-side-label {{
        font-size: 11px; font-weight: 700; text-transform: uppercase;
        letter-spacing: .05em; color: var(--text-muted);
      }}
      .tdb-side-sep {{
        align-self: center; flex-shrink: 0;
        font-size: 20px; color: var(--text-faint, #cbd5e1);
      }}
      .tdb-search-outer {{
        position: relative;
        border: 1px solid var(--border); border-radius: 8px;
        background: var(--card);
      }}
      .tdb-search {{
        width: 100%; padding: 9px 12px; border: none !important; background: transparent;
        color: var(--text); font-size: 15px; outline: none; box-sizing: border-box;
        border-radius: 8px; appearance: none;
      }}
      .tdb-search-outer:focus-within {{ border-color: var(--accent, #3b82f6); }}
      .tdb-dropdown {{
        position: absolute; top: 100%; left: 0; right: 0;
        background: var(--card); border: 1px solid var(--border);
        border-radius: 0 0 8px 8px; z-index: 100; max-height: 240px;
        overflow-y: auto; box-shadow: 0 4px 16px rgba(0,0,0,.15);
      }}
      .tdb-dropdown-item {{
        padding: 8px 12px; cursor: pointer; display: flex;
        align-items: center; justify-content: space-between;
        border-bottom: 1px solid var(--border);
      }}
      .tdb-dropdown-item:last-child {{ border-bottom: none; }}
      .tdb-dropdown-item:hover {{ background: var(--bg-alt); }}
      .tdb-di-name {{ font-size: 15px; font-weight: 600; color: var(--text); }}
      .tdb-di-pos {{ font-size: 13px; color: var(--text-muted); flex-shrink: 0; }}
      .tdb-chip-area {{ display: flex; gap: 6px; flex-wrap: wrap; }}
      .tdb-chip {{
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(59,130,246,.12); color: var(--accent, #3b82f6);
        border: 1px solid rgba(59,130,246,.3);
        border-radius: 8px; padding: 4px 10px 4px 12px;
        font-size: 13px; font-weight: 600;
      }}
      .tdb-chip-x {{
        background: none; border: none; cursor: pointer;
        color: var(--accent, #3b82f6); font-size: 15px; line-height: 1;
        padding: 0; opacity: .7;
      }}
      .tdb-chip-x:hover {{ opacity: 1; }}
        .tdb-filter-col {{ display: flex; flex-direction: column; gap: 6px; align-self: flex-end; padding-bottom: 1px; }}
        .tdb-lt-filters, .tdb-lf-filters {{ display: flex; gap: 6px; }}
      .tdb-sort-wrap {{ display: flex; align-items: center; gap: 8px; }}
      .tdb-sort-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); }}
      .tdb-sort {{
        padding: 7px 10px; border-radius: 8px; border: 1px solid var(--border);
        background: var(--card); color: var(--text); font-size: 13px; font-weight: 600;
        cursor: pointer; outline: none; min-height: 34px;
      }}
      .tdb-sort:hover {{ border-color: var(--accent, #3b82f6); }}
      .tdb-status {{ font-size: 13px; color: var(--text-muted); margin-bottom: 14px; min-height: 16px; }}
      .tdb-list {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }}
      @media(max-width: 600px) {{ .tdb-list {{ grid-template-columns: 1fr; }} }}
      .tdb-card {{
        border: 1px solid var(--border); border-radius: 12px;
        overflow: hidden; background: var(--card);
        display: flex; flex-direction: column;
      }}
      .tdb-card-head {{
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 14px; border-bottom: 1px solid var(--border);
        background: var(--bg-alt, rgba(0,0,0,.03));
      }}
      .tdb-card-date {{ font-size: 11px; color: var(--text-muted); font-weight: 500; }}
      .tdb-badges {{ display: flex; gap: 5px; flex-wrap: wrap; }}
      .tdb-card-body {{ display: grid; grid-template-columns: 1fr 1px 1fr; }}
      .tdb-col {{ padding: 12px 14px; display: flex; flex-direction: column; gap: 5px; }}
      .tdb-col-divider {{ background: var(--border); }}
      .tdb-side-label {{
        font-size: 10px; font-weight: 800; letter-spacing: .06em;
        color: var(--text-muted); text-transform: uppercase; margin-bottom: 2px;
      }}
      .tdb-asset {{
        font-size: 15px; color: var(--text); font-weight: 500;
        display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
      }}
      .tdb-asset.tdb-match {{ font-weight: 800; color: var(--accent, #3b82f6); }}
      .tdb-asset.tdb-pick {{ color: var(--text-muted); font-size: 15px; font-weight: 500; }}
      .tdb-pos {{
        font-size: 11px; font-weight: 800; padding: 1px 6px; border-radius: 5px;
        background: var(--row); color: var(--text-muted); flex-shrink: 0; letter-spacing: .02em;
      }}
      /* Position colours (match the roster badges site-wide) */
      .tdb-pos.pos-QB {{ background: color-mix(in srgb, #3b82f6 16%, transparent); color: #3b82f6; }}
      .tdb-pos.pos-RB {{ background: color-mix(in srgb, #22c55e 16%, transparent); color: #16a34a; }}
      .tdb-pos.pos-WR {{ background: color-mix(in srgb, #f59e0b 18%, transparent); color: #d97706; }}
      .tdb-pos.pos-TE {{ background: color-mix(in srgb, #8b5cf6 16%, transparent); color: #8b5cf6; }}
      [data-theme="dark"] .tdb-pos.pos-RB {{ color: #22c55e; }}
      [data-theme="dark"] .tdb-pos.pos-WR {{ color: #f59e0b; }}
      /* Card footer: pinned to the card bottom via margin-top:auto so footers
         align in a flat row across equal-height cards; leftover space sits
         above the footer inside the card. */
      .tdb-card-foot {{
        margin-top: auto;
        display: flex; justify-content: space-between; align-items: center;
        padding: 10px 14px; border-top: 1px solid var(--border);
        background: var(--bg-alt, rgba(0,0,0,.02));
      }}
      .tdb-delta {{ font-size: 13px; font-weight: 700; color: #16a34a; }}
      .tdb-delta-fair {{ color: var(--text-muted); font-weight: 600; }}
      .tdb-league-info {{ font-size: 12px; color: var(--text-muted); }}
      .chip--fair {{
        background: color-mix(in srgb, #22c55e 14%, transparent);
        color: #16a34a; border: 1px solid color-mix(in srgb, #22c55e 30%, transparent);
      }}
      [data-theme="dark"] .chip--fair {{ color: #22c55e; }}
      @media(max-width: 480px) {{
        .tdb-card-body {{ grid-template-columns: 1fr; }}
        .tdb-col-divider {{ height: 1px; width: auto; }}
      }}
      @media(max-width: 600px) {{
        .tdb-toolbar {{ flex-direction: column; gap: 10px; margin-bottom: 12px; }}
        .tdb-sides-row {{ flex-direction: column; gap: 10px; width: 100%; }}
        .tdb-sides-row.tdb-search-card .tdb-side-sep {{ transform: rotate(90deg); }}
        .tdb-side-wrap {{ min-width: 0; width: 100%; flex: none; }}
        .tdb-search-outer {{ width: 100%; box-sizing: border-box; }}
        .tdb-chip-area {{ flex-wrap: wrap; gap: 6px; min-height: 0; }}
        .tdb-side-sep {{ align-self: flex-start; padding-top: 0; font-size: 13px; }}
        .tdb-filter-col {{ width: 100%; }}
        .tdb-sort-wrap {{ width: 100%; }}
        .tdb-sort {{ flex: 1; min-width: 0; }}
        .tdb-lt-filters, .tdb-lf-filters {{ width: 100%; display: flex; }}
        .tdb-lt, .tdb-lf {{ flex: 1; text-align: center; }}
      }}
      /* ── Pagination (matches site-wide style) ── */
      .ti-pagination {{
        display: flex; justify-content: space-between; align-items: center;
        margin: 20px 0; padding: 12px 0; border-top: 1px solid var(--border);
      }}
      .ti-pagination-info {{ font-size: 13px; color: var(--text-muted); }}
      .ti-pagination-controls {{ display: flex; align-items: center; gap: 12px; }}
      .ti-pagination-btn {{
        padding: 6px 12px; border: 1px solid var(--border); border-radius: 6px;
        background: var(--card); color: var(--text); cursor: pointer;
        font-size: 13px; font-weight: 500; transition: all .15s;
        display: flex; align-items: center; gap: 4px;
      }}
      .ti-pagination-btn:hover:not(:disabled) {{ background: var(--bg-alt); border-color: var(--accent, #3b82f6); }}
      .ti-pagination-btn:disabled {{ opacity: .5; cursor: not-allowed; }}
      .ti-page-numbers {{ display: flex; gap: 4px; }}
      .ti-page-number {{
        padding: 4px 8px; border: 1px solid var(--border); border-radius: 4px;
        background: var(--card); color: var(--text); cursor: pointer;
        font-size: 13px; font-weight: 500; min-width: 28px; text-align: center;
      }}
      .ti-page-number:hover {{ background: var(--bg-alt); }}
      .ti-page-number.active {{
        background: var(--accent, #3b82f6); color: #fff;
        border-color: var(--accent, #3b82f6); font-weight: 700;
      }}
    </style>

    <script>
    (function() {{
      const TDB_SEASON = {season};
      const TDB_PLATFORM = '{platform}';
      let currentPage = 1;
      let paginationData = null;
      let leagueType = 'all';
      let leagueFormat = '{_tdb_fmt}';
      let tdbSort = 'date_desc'; // date_desc | date_asc | value_desc | value_asc
      // Latest-wins request handling: a new filter change aborts the in-flight
      // trade search instead of being silently dropped (the old `loading`
      // guard left stale single-side results on screen with both chips set
      // when the Side B player was added while the Side A request was in flight).
      let tdbAbort = null;
      let selectedA = []; // [{{ id, name }}, ...]
      let selectedB = [];
      let tdbAllPlayers = null;
      let tdbPlayersPromise = null;

      const listEl   = document.getElementById('tdbList');
      const statusEl = document.getElementById('tdbStatus');

      const initQ = new URLSearchParams(window.location.search).get('q') || '';

      loadTDBPage(1);

      async function ensureTDBPlayers() {{
        if (tdbAllPlayers) return tdbAllPlayers;
        if (!tdbPlayersPromise) {{
          // Fetch a compact list; server-side `q` filtering is used per keystroke
          // via tdbSearchPlayers() below, so we only need a small initial set
          // for the dropdown. Use limit to keep the payload small.
          tdbPlayersPromise = fetch('/api/players?limit=100')
            .then(r => r.json())
            .then(data => {{ tdbAllPlayers = Array.isArray(data) ? data : (data.players || []); return tdbAllPlayers; }});
        }}
        return tdbPlayersPromise;
      }}

      async function tdbSearchPlayers(q) {{
        // Server-side search: returns top matches without downloading the full list
        try {{
          const r = await fetch('/api/players?q=' + encodeURIComponent(q) + '&limit=15');
          const data = await r.json();
          return Array.isArray(data) ? data : (data.players || []);
        }} catch (e) {{
          return [];
        }}
      }}

      function tdbScore(name, q) {{
        if (!name || !q) return 0;
        const n = name.toLowerCase(), query = q.toLowerCase();
        if (n === query) return 4;
        if (n.startsWith(query)) return 3;
        if (n.includes(' ' + query)) return 2;
        if (n.includes(query)) return 1;
        return 0;
      }}

      function renderTDBChips(side) {{
        const arr      = side === 'A' ? selectedA : selectedB;
        const chipArea = document.getElementById(side === 'A' ? 'tdbSideAChip' : 'tdbSideBChip');
        chipArea.style.display = arr.length ? 'flex' : 'none';
        chipArea.innerHTML = arr.map(p =>
          `<div class="tdb-chip">${{p.name}}<button class="tdb-chip-x" onclick="removeTDBPlayer('${{side}}','${{p.id}}')">&#x2715;</button></div>`
        ).join('');
      }}

      window.removeTDBPlayer = function(side, id) {{
        if (side === 'A') selectedA = selectedA.filter(p => p.id !== id);
        else              selectedB = selectedB.filter(p => p.id !== id);
        renderTDBChips(side);
        loadTDBPage(1);
      }};

      function bindTDBSearch(side) {{
        const input    = document.getElementById(side === 'A' ? 'tdbSideASearch' : 'tdbSideBSearch');
        const drop     = document.getElementById(side === 'A' ? 'tdbSideADropdown' : 'tdbSideBDropdown');
        if (!input) return;
        let tdbDebounce = null;

        input.addEventListener('input', function() {{
          clearTimeout(tdbDebounce);
          tdbDebounce = setTimeout(async function() {{
          const q = input.value.trim();
          drop.innerHTML = '';
          drop.style.display = 'none';
          if (!q || q.length < 2) return;

          const arr     = side === 'A' ? selectedA : selectedB;
          const already = new Set(arr.map(p => p.id));
          // Server-side search: fast, no full-list download
          const matches = (await tdbSearchPlayers(q))
            .filter(p => !already.has(String(p.player_id)))
            .slice(0, 15);

          if (!matches.length) return;

          matches.forEach(p => {{
            const item = document.createElement('div');
            item.className = 'tdb-dropdown-item';
            const pos = [p.position, p.team].filter(Boolean).join(' · ');
            item.innerHTML = `<span class="tdb-di-name">${{p.name}}</span><span class="tdb-di-pos">${{pos}}</span>`;
            item.addEventListener('click', () => {{
              const sel = {{ id: String(p.player_id), name: p.name }};
              if (side === 'A') selectedA.push(sel);
              else              selectedB.push(sel);
              input.value = '';
              drop.style.display = 'none';
              renderTDBChips(side);
              loadTDBPage(1);
            }});
            drop.appendChild(item);
          }});
          drop.style.display = 'block';
          }}, 250);
        }});

        input.addEventListener('blur', () => {{
          setTimeout(() => {{ drop.style.display = 'none'; }}, 150);
        }});
      }}

      bindTDBSearch('A');
      bindTDBSearch('B');

      // Auto-select player from ?q= URL param
      if (initQ) {{
        ensureTDBPlayers().then(players => {{
          const q = initQ.toLowerCase();
          const match = players.find(p => p.name && p.name.toLowerCase().includes(q));
          if (match) {{
            selectedA = [{{ id: String(match.player_id), name: match.name }}];
            renderTDBChips('A');
            loadTDBPage(1);
          }}
        }});
      }}

      function showTDBSkeletons() {{
        const grid = document.getElementById('tdbSkeletonGrid');
        let html = '';
        for (let i = 0; i < 6; i++) {{
          html += `<div class="tdb-skeleton-card">
            <div class="tdb-skeleton-head"></div>
            <div class="tdb-skeleton-body">
              <div class="tdb-skeleton-col">
                <div class="tdb-skeleton-line"></div>
                <div class="tdb-skeleton-line short"></div>
                <div class="tdb-skeleton-line"></div>
              </div>
              <div class="tdb-col-divider"></div>
              <div class="tdb-skeleton-col">
                <div class="tdb-skeleton-line short"></div>
                <div class="tdb-skeleton-line"></div>
              </div>
            </div>
          </div>`;
        }}
        grid.innerHTML = html;
      }}
      showTDBSkeletons();

      function loadTDBPage(page) {{
        if (tdbAbort) tdbAbort.abort();
        tdbAbort = new AbortController();
        const tdbSignal = tdbAbort.signal;
        if (typeof page === 'string') {{
          if (page === 'prev' && currentPage > 1) page = currentPage - 1;
          else if (page === 'next' && paginationData && paginationData.has_next) page = currentPage + 1;
          else return;
        }}
        currentPage = page;
        statusEl.textContent = '';
        listEl.style.display = 'none';
        document.getElementById('tdbLoading').style.display = '';
        document.getElementById('tdbPagination').style.display = 'none';
        const params = new URLSearchParams({{ page: page - 1, limit: 20, league_type: leagueType, season: TDB_SEASON, league_format: leagueFormat, sort: tdbSort }});
        if (selectedA.length) params.set('player_a', selectedA.map(p => p.id).join(','));
        if (selectedB.length) params.set('player_b', selectedB.map(p => p.id).join(','));
        fetch('/api/trade-database?' + params, {{ signal: tdbSignal }})
          .then(r => r.json())
          .then(data => {{
            if (data.error) throw new Error(data.error);
            const trades = data.trades || [];
            document.getElementById('tdbLoading').style.display = 'none';
            if (trades.length === 0) {{
              listEl.innerHTML = '<div style="color:var(--text-muted);padding:20px 0;text-align:center;grid-column:1/-1;">No trades found.</div>';
              listEl.style.display = '';
              document.getElementById('tdbPagination').style.display = 'none';
              return;
            }}
            paginationData = data.pagination;
            listEl.style.display = '';
            updateTDBPaginationControls();
            renderTDBTrades(trades);
          }})
          .catch(err => {{
            if (err && err.name === 'AbortError') return; // superseded by a newer request
            console.error('Error loading trades:', err);
            document.getElementById('tdbLoading').style.display = 'none';
            statusEl.textContent = 'Error loading trades';
          }});
      }}

      function updateTDBPaginationControls() {{
        if (!paginationData) return;
        const prevBtn        = document.getElementById('tdbPrevBtn');
        const nextBtn        = document.getElementById('tdbNextBtn');
        const pageNumbers    = document.getElementById('tdbPageNumbers');
        const paginationText = document.getElementById('tdbPaginationText');
        prevBtn.disabled = !paginationData.has_prev;
        nextBtn.disabled = !paginationData.has_next;
        const start = (paginationData.current_page - 1) * paginationData.per_page + 1;
        const end   = Math.min(paginationData.current_page * paginationData.per_page, paginationData.total_players);
        paginationText.textContent = `Showing ${{start}}-${{end}} of ${{paginationData.total_players}} trades`;
        pageNumbers.innerHTML = '';
        const maxPages = 5;
        let startPage = Math.max(1, paginationData.current_page - Math.floor(maxPages / 2));
        let endPage   = Math.min(paginationData.total_pages, startPage + maxPages - 1);
        if (endPage - startPage < maxPages - 1) startPage = Math.max(1, endPage - maxPages + 1);
        for (let i = startPage; i <= endPage; i++) {{
          const btn = document.createElement('button');
          btn.className = 'ti-page-number' + (i === paginationData.current_page ? ' active' : '');
          btn.textContent = i;
          btn.onclick = () => loadTDBPage(i);
          pageNumbers.appendChild(btn);
        }}
        document.getElementById('tdbPagination').style.display = 'flex';
      }}

      function renderTDBTrades(trades) {{
        const matchIdsA = new Set(selectedA.map(p => p.id));
        const matchIdsB = new Set(selectedB.map(p => p.id));
        listEl.innerHTML = '';
        trades.forEach(t => {{
          const sfBadge    = t.is_superflex === true  ? '<span class="chip chip--sm chip--accent">SF</span>'
                           : t.is_superflex === false ? '<span class="chip chip--sm">1QB</span>' : '';
          const teamsBadge = t.num_teams    ? `<span class="chip chip--sm">${{t.num_teams}} Teams</span>` : '';
          const scoreBadge = t.scoring_type ? `<span class="chip chip--sm">${{t.scoring_type.toUpperCase()}}</span>` : '';
          const weekBadge  = t.week ? `<span class="chip chip--sm">Wk ${{t.week}}</span>` : '';
          function renderAsset(a) {{
            const pid   = a.player_id ? String(a.player_id) : '';
            const match = pid && (matchIdsA.has(pid) || matchIdsB.has(pid));
            const cls = 'tdb-asset' + (a.type === 'pick' ? ' tdb-pick' : '') + (match ? ' tdb-match' : '');
            const pos = a.position && a.type === 'player' ? `<span class="tdb-pos pos-${{a.position}}">${{a.position}}</span>` : '';
            return `<div class="${{cls}}">${{a.name}}${{pos}}</div>`;
          }}
          const sideA = (t.side_a || []).map(renderAsset).join('') || '<div class="tdb-asset" style="color:var(--text-muted)">-</div>';
          const sideB = (t.side_b || []).map(renderAsset).join('') || '<div class="tdb-asset" style="color:var(--text-muted)">-</div>';
          // Value delta footer + Fair badge (mock-approved)
          const valA = parseFloat(t.side_a_value) || 0;
          const valB = parseFloat(t.side_b_value) || 0;
          const valDelta = valA - valB;
          const valMax = Math.max(valA, valB, 1);
          const isFair = Math.abs(valDelta) <= Math.max(25, 0.1 * valMax);
          const fairBadge = isFair ? '<span class="chip chip--sm chip--fair">Fair</span>' : '';
          let deltaText = '';
          if (isFair) {{
            deltaText = '<span class="tdb-delta tdb-delta-fair">Fair deal</span>';
          }} else if (valDelta > 0) {{
            deltaText = `<span class="tdb-delta">+${{valDelta.toFixed(1)}} value to A</span>`;
          }} else {{
            deltaText = `<span class="tdb-delta">+${{Math.abs(valDelta).toFixed(1)}} value to B</span>`;
          }}
          const leagueInfo = [t.num_teams ? t.num_teams + '-team' : '', t.scoring_type ? t.scoring_type.toUpperCase() : '']
            .filter(Boolean).join(' ');
          const card = document.createElement('div');
          card.className = 'tdb-card';
          card.innerHTML = `
            <div class="tdb-card-head">
              <span class="tdb-card-date">${{t.date || '-'}}</span>
              <div class="tdb-badges">${{weekBadge}}${{sfBadge}}${{fairBadge}}</div>
            </div>
            <div class="tdb-card-body">
              <div class="tdb-col"><div class="tdb-side-label">Team A got</div>${{sideA}}</div>
              <div class="tdb-col-divider"></div>
              <div class="tdb-col"><div class="tdb-side-label">Team B got</div>${{sideB}}</div>
            </div>
            <div class="tdb-card-foot">${{deltaText}}${{leagueInfo ? `<span class="tdb-league-info">${{leagueInfo}}</span>` : ''}}</div>
`;
          listEl.appendChild(card);
        }});
      }}

      window.tdbFilter = function(lt) {{
        leagueType = lt;
        document.querySelectorAll('.tdb-lt').forEach(b => b.classList.toggle('active', b.dataset.lt === lt));
        loadTDBPage(1);
      }};

      window.tdbFormatFilter = function(lf) {{
        leagueFormat = lf;
        document.querySelectorAll('.tdb-lf').forEach(b => b.classList.toggle('active', b.dataset.lf === lf));
        loadTDBPage(1);
      }};

      window.tdbSortChange = function(v) {{
        tdbSort = v;
        loadTDBPage(1);
      }};

      window.loadTDBPage = loadTDBPage;
    }})();
    </script>
    """
    return render_page(
        "Fantasy Football Trade Database - Search Real Dynasty Trades | BR Fantasy",
        league_id, "trade-database", body_html, platform, season,
        description=(
            "Search thousands of real Sleeper dynasty trades to see how players and draft "
            "picks are actually valued. Filter by player to study dynasty and redraft market "
            "prices. Values still apply to ESPN, Yahoo, and MFL rosters."
        ),
    )


@trade_bp.route("/trade-database")
def page_trade_database_guest():
    from app import get_nfl_state
    nfl_state = get_nfl_state() or {}
    current_season = int(nfl_state.get("season") or datetime.now().year)
    return page_trade_database(platform="sleeper", season=current_season, league_id=None)
