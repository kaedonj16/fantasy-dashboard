"""
Standalone Draft Room (Draft Assistant) page.

Phase 2: a dedicated, self-contained draft board that supersedes the old
Prospects→Draft tab. Supports manual drafting for both startup (all players)
and rookie drafts, with snake / linear / third-round-reversal pick order.
Live Sleeper sync, persistence/history, and the full command-center panels
land in later phases; this establishes the standalone page + board grid +
best-available picker + the pickOrder foundation.

The page is self-contained: its CSS/JS are inlined here so nothing depends on
the prospects-page IIFE. Server values are passed via a small window.__draftCfg
JSON blob so the JS body needs no f-string brace escaping.
"""
from __future__ import annotations

import json
from typing import Optional


def build_draft_room_body(
    league_id: Optional[str],
    season: Optional[int],
    platform: Optional[str] = None,
    *,
    is_guest: bool = False,
    num_teams: Optional[int] = None,
    is_superflex: bool = False,
    roster_positions: Optional[list] = None,
    viewer_user_id: Optional[str] = None,
) -> str:
    cfg = {
        "leagueId": league_id or "",
        "season": int(season) if season else None,
        "platform": platform or "sleeper",
        "isGuest": bool(is_guest),
        "numTeams": int(num_teams) if num_teams else None,
        "isSuperflex": bool(is_superflex),
        "rosterPositions": list(roster_positions) if roster_positions else None,
        "viewerUserId": str(viewer_user_id) if viewer_user_id else "",
    }
    cfg_json = json.dumps(cfg)
    return (
        f'<script>window.__draftCfg = {cfg_json};</script>\n'
        + _DRAFT_ROOM_HTML
    )


# Plain (non-f) string — safe to contain { } freely.
_DRAFT_ROOM_HTML = r"""
<div class="dr-wrap">
  <div class="dr-hero">
    <h1 class="dr-title">Draft Room</h1>
    <p class="dr-sub">Mock against ADP-driven CPU teams, draft manually, or sync a live Sleeper draft,
      with best-available, recommendations, tiers, and your live draft grade.</p>
  </div>

  <!-- Setup -->
  <div class="dr-setup" id="drSetup">
    <div class="dr-setup-card">
      <div class="dr-setup-header">
        <h2 class="dr-setup-title">Set up your draft</h2>
        <p class="dr-setup-desc">Run a mock against ADP-driven CPU teams, draft manually, or connect a live Sleeper draft.</p>
      </div>

      <div class="dr-setup-section">
        <div class="dr-setup-section-label">Format</div>
        <div class="dr-setup-grid">
          <div class="dr-field"><span>Draft Type</span>
            <select id="drType">
              <option value="startup">Startup (Dynasty)</option>
              <option value="rookie">Rookie (Dynasty)</option>
              <option value="redraft">Redraft</option>
            </select>
          </div>
          <div class="dr-field"><span>Scoring</span>
            <select id="drSf">
              <option value="0">1QB</option>
              <option value="1">Superflex</option>
            </select>
          </div>
          <div class="dr-field"><span>Pick Order</span>
            <select id="drOrder">
              <option value="snake">Snake</option>
              <option value="linear">Linear</option>
              <option value="3rr">3rd Round Reversal</option>
            </select>
          </div>
        </div>
      </div>

      <div class="dr-setup-section">
        <div class="dr-setup-section-label">League</div>
        <div class="dr-setup-grid">
          <div class="dr-field"><span>Teams</span>
            <select id="drTeams">
              <option>8</option><option>10</option><option selected>12</option><option>14</option>
            </select>
          </div>
          <div class="dr-field"><span>Rounds</span>
            <input id="drRounds" type="number" min="1" max="40" value="15">
          </div>
          <div class="dr-field"><span>Your Pick</span>
            <select id="drSlot"></select>
          </div>
        </div>
      </div>

      <div class="dr-setup-section">
        <div class="dr-setup-section-label">Roster Slots</div>
        <div id="drRosterSection"></div>
      </div>

      <div class="dr-setup-cta">
        <button class="dr-btn dr-btn-primary dr-btn-lg" id="drStartSim">&#9654;&nbsp; Start Mock Draft</button>
        <button class="dr-btn dr-btn-lg" id="drStart">Draft Manually</button>
        <button class="dr-btn dr-btn-ghost" id="drConnect">Connect Live Draft</button>
      </div>
      <div class="dr-live-list" id="drLiveList" style="display:none;"></div>
    </div>
  </div>

  <!-- Board + side -->
  <div class="dr-main" id="drMain" style="display:none;">
    <div class="dr-statusbar">
      <div class="dr-status-left">
        <span class="dr-pill" id="drRoundPill">Round 1</span>
        <span class="dr-pill" id="drPickPill">Pick 1</span>
        <span class="dr-onclock">On the clock: <b id="drOnClock">Team 1</b></span>
        <span class="dr-pill dr-pill-you" id="drNextPill" style="display:none;"></span>
        <span class="dr-pill dr-pill-grade" id="drGradePill" style="display:none;"></span>
        <span class="dr-pill dr-pill-live" id="drLiveBadge" style="display:none;">&#9679; LIVE</span>
        <span class="dr-progress" id="drProgress"></span>
        <span class="dr-save" id="drSave"></span>
      </div>
      <div class="dr-status-right">
        <select class="dr-sim-speed" id="drSimSpeed" style="display:none;" title="Simulation speed">
          <option value="1400">Slow</option>
          <option value="700" selected>Normal</option>
          <option value="300">Fast</option>
          <option value="60">Instant</option>
        </select>
        <button class="dr-btn dr-btn-ghost" id="drSimToggle" style="display:none;">Pause</button>
        <button class="dr-btn dr-btn-ghost" id="drSummaryBtn" style="display:none;">Summary</button>
        <button class="dr-btn dr-btn-ghost" id="drShare">Share</button>
        <button class="dr-btn dr-btn-ghost" id="drUndo">Undo</button>
        <button class="dr-btn dr-btn-ghost" id="drEdit">Edit Setup</button>
        <button class="dr-btn dr-btn-ghost dr-btn-danger" id="drReset">Reset</button>
      </div>
    </div>

    <div class="dr-cols">
      <div class="dr-board-wrap">
        <div class="dr-board" id="drBoard"></div>
      </div>
      <aside class="dr-side">
        <div class="otc-main-tabs dr-side-tabs" id="drSideTabs">
          <button class="otc-main-tab is-active" data-stab="best">Players</button>
          <button class="otc-main-tab" data-stab="rec">Recs</button>
          <button class="otc-main-tab" data-stab="future">Future</button>
          <button class="otc-main-tab" data-stab="queue">Queue</button>
          <button class="otc-main-tab" data-stab="needs">Team</button>
        </div>
        <div class="dr-side-head" id="drBestControls">
          <div class="dr-side-controls">
            <input id="drSearch" type="search" placeholder="Search…" autocomplete="off">
            <select id="drBaSort">
              <option value="value">Value</option>
              <option value="adp">ADP</option>
              <option value="steals">Steals</option>
            </select>
          </div>
          <div class="dr-pos-filters" id="drPosFilters">
            <button class="dr-pos active" data-pos="ALL">All</button>
            <button class="dr-pos" data-pos="QB">QB</button>
            <button class="dr-pos" data-pos="RB">RB</button>
            <button class="dr-pos" data-pos="WR">WR</button>
            <button class="dr-pos" data-pos="TE">TE</button>
            <button class="dr-pos dr-pos-kdef" data-pos="K" style="display:none;">K</button>
            <button class="dr-pos dr-pos-kdef" data-pos="DEF" style="display:none;">DEF</button>
          </div>
          <div class="dr-adp-src" id="drAdpSrc"></div>
        </div>
        <div class="dr-ba-list" id="drBaList">
          <div class="dr-loading"><div class="loading-spinner" style="width:22px;height:22px;"></div><span>Loading players…</span></div>
        </div>
      </aside>
    </div>
  </div>

  <!-- Player preview / draft confirm -->
  <div class="dr-preview-overlay" id="drPreview" style="display:none;">
    <div class="dr-preview-card" id="drPreviewCard"></div>
  </div>

  <!-- End-of-draft summary -->
  <div class="dr-summary-overlay" id="drSummary" style="display:none;">
    <div class="dr-summary-card" id="drSummaryCard"></div>
  </div>
</div>

<style>
  .dr-wrap { max-width: 1640px; margin: 0 auto; padding: 10px 12px 40px; }
  .dr-hero { margin-bottom: 12px; }
  .dr-title { font-size: clamp(20px,4vw,28px); font-weight: 800; color: var(--text); margin: 0 0 4px; }
  .dr-sub { font-size: 14px; color: var(--text-muted); margin: 0; }
  /* ── Setup (redesigned) ── */
  .dr-setup { display: flex; justify-content: center; padding: 8px 0; }
  .dr-setup-card { width: 100%; max-width: 720px; background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 24px 24px 22px; box-shadow: 0 8px 30px rgba(0,0,0,.10); }
  .dr-setup-header { margin-bottom: 18px; }
  .dr-setup-title { font-size: 20px; font-weight: 800; color: var(--text); margin: 0 0 4px; }
  .dr-setup-desc { font-size: 13px; color: var(--text-muted); margin: 0; line-height: 1.5; }
  .dr-setup-section { padding: 14px 0; border-top: 1px solid var(--border); }
  .dr-setup-section:first-of-type { border-top: none; padding-top: 0; }
  .dr-setup-section-label { font-size: 11px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase;
    color: var(--text-muted); margin-bottom: 10px; }
  .dr-setup-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap: 12px; }
  .dr-field { display: flex; flex-direction: column; gap: 6px; font-size: 12px; font-weight: 700; color: var(--text-muted); }
  .dr-field select, .dr-field input {
    padding: 9px 11px; border-radius: 9px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); font-size: 14px; font-weight: 600; outline: none; min-height: 40px;
  }
  .dr-field select:focus, .dr-field input:focus { border-color: var(--accent,#38bdf8); }
  .dr-setup-cta { margin-top: 20px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .dr-btn-lg { padding: 12px 22px; font-size: 14px; border-radius: 10px; }
  .dr-sim-speed { padding: 6px 8px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg);
    color: var(--text); font-size: 12px; font-weight: 600; }
  .dr-btn {
    padding: 9px 16px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer;
    border: 1px solid var(--border); background: var(--bg); color: var(--text); white-space: nowrap;
  }
  .dr-btn-primary { background: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8); color: #fff; }
  .dr-btn-ghost { background: transparent; font-weight: 600; }
  .dr-btn-danger { color: #ef4444; border-color: rgba(239,68,68,.4); }
  .dr-statusbar {
    display: flex; align-items: center; justify-content: space-between; gap: 10px; flex-wrap: wrap;
    padding: 10px 12px; margin-bottom: 12px; border: 1px solid var(--border); border-radius: 10px;
    background: var(--card);
    position: sticky; top: 56px; z-index: 30;
  }
  .dr-status-left { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .dr-status-right { display: flex; align-items: center; gap: 6px; }
  .dr-pill { font-size: 12px; font-weight: 700; padding: 3px 9px; border-radius: 999px;
    background: rgba(56,189,248,.14); color: var(--accent,#38bdf8); }
  .dr-pill-you { background: rgba(34,197,94,.16); color: #22c55e; }
  .dr-pill-live { background: rgba(239,68,68,.16); color: #ef4444; animation: drPulse 1.6s ease-in-out infinite; }
  .dr-progress { font-size: 12px; color: var(--text-muted); }
  .dr-save { font-size: 11px; color: #22c55e; }
  .dr-live-list { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
  .dr-live-head { font-size: 12px; font-weight: 700; color: var(--text-muted); }
  .dr-live-item { text-align: left; padding: 9px 12px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); font-size: 13px; cursor: pointer; }
  .dr-live-item:hover { border-color: var(--accent,#38bdf8); }
  .dr-live-status { font-size: 10px; font-weight: 800; text-transform: uppercase; padding: 1px 6px; border-radius: 999px; margin-right: 6px; }
  .dr-ls-drafting { background: rgba(239,68,68,.16); color: #ef4444; }
  .dr-ls-pre_draft { background: rgba(245,158,11,.16); color: #f59e0b; }
  .dr-ls-complete { background: rgba(148,163,184,.16); color: #94a3b8; }
  .dr-onclock { font-size: 13px; color: var(--text-muted); }
  .dr-onclock b { color: var(--text); }
  .dr-cols { display: grid; grid-template-columns: 1fr 340px; gap: 14px; align-items: start; }
  .dr-board-wrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--card); padding: 6px; }
  .dr-board { display: grid; gap: 5px; min-width: max-content; }
  .dr-cell {
    border: 1px solid var(--border); border-radius: 8px; padding: 5px 6px 0; min-height: 50px;
    background: var(--bg); display: flex; align-items: flex-end; gap: 6px; position: relative; overflow: hidden;
  }
  .dr-cell-body { padding-bottom: 6px; }
  .dr-cell-empty { opacity: .45; }
  .dr-cell-filled { background: linear-gradient(180deg, rgba(56,189,248,.05), var(--bg)); }
  .dr-cell-current { outline: 2px solid var(--accent,#38bdf8); animation: drPulse 1.6s ease-in-out infinite; }
  @keyframes drPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(56,189,248,.0); } 50% { box-shadow: 0 0 0 3px rgba(56,189,248,.18); } }
  .dr-cell-mine { box-shadow: inset 3px 0 0 var(--accent,#38bdf8); opacity: 1; }
  .dr-cell-mine.dr-cell-empty { opacity: 1; background: linear-gradient(180deg, rgba(56,189,248,.10), var(--bg)); }
  .dr-cell-claimed { box-shadow: inset 3px 0 0 #f59e0b; }     /* traded-in pick */
  .dr-cell-claimable { cursor: pointer; }
  .dr-cell-claimable:hover { outline: 1px dashed var(--accent,#38bdf8); outline-offset: -2px; }
  .dr-cell-mineflag { position: absolute; top: 2px; right: 5px; font-size: 8px; font-weight: 800;
    letter-spacing: .04em; color: var(--accent,#38bdf8); }
  .dr-cell-claimed .dr-cell-mineflag { color: #f59e0b; }
  .dr-cell-just { animation: drPop .35s ease; }
  @keyframes drPop { 0% { transform: scale(.92); opacity: .3; } 100% { transform: scale(1); opacity: 1; } }
  .dr-cell-val { position: absolute; top: 2px; right: 5px; font-size: 9px; font-weight: 800; color: var(--accent,#38bdf8); }
  .dr-cell-num { position: absolute; top: 2px; left: 5px; font-size: 9px; font-weight: 700; color: var(--text-muted); }
  .dr-hs { width: 40px; height: 40px; border-radius: 8px 8px 0 0; object-fit: cover; object-position: top center;
    flex-shrink: 0; background: transparent; align-self: flex-end; }
  .dr-cell-body { min-width: 0; line-height: 1.2; }
  .dr-cell-name { font-size: 12px; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 96px; }
  .dr-cell-meta { font-size: 10px; color: var(--text-muted); }
  .dr-posbadge { font-size: 9px; font-weight: 700; color: #fff; border-radius: 3px; padding: 1px 4px; }
  .dr-colhead { font-size: 11px; font-weight: 700; color: var(--text-muted); text-align: center; padding: 2px 0; white-space: nowrap; }
  .dr-colhead-you { color: var(--accent,#38bdf8); }
  .dr-side { border: 1px solid var(--border); border-radius: 10px; background: var(--card); display: flex; flex-direction: column;
    position: sticky; top: 120px; align-self: start; max-height: calc(100vh - 134px); z-index: 20; overflow: hidden; }
  /* Reuse the trade-calculator pill tabs (otc-main-tabs), fit to the panel + scroll */
  .dr-side-tabs.otc-main-tabs { width: auto; margin: 8px; overflow-x: auto; -webkit-overflow-scrolling: touch; }
  .dr-side-tabs .otc-main-tab { padding: 7px 12px; flex: 0 0 auto; }
  .dr-side-head { padding: 10px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 8px; }
  /* command-center panels (team / runs) */
  .dr-panel { padding: 12px; overflow-y: auto; }
  .dr-roster { padding: 10px; display: flex; flex-direction: column; gap: 6px; overflow-y: auto; }
  .dr-roster-div { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin: 8px 0 2px; }
  .dr-rslot { display: flex; align-items: center; gap: 8px; padding: 4px 8px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg); min-height: 42px; overflow: hidden; }
  .dr-rslot-open { opacity: .65; border-style: dashed; }
  .dr-rslot-pos { width: 36px; flex-shrink: 0; text-align: center; font-size: 10px; font-weight: 800; color: #fff; border-radius: 4px; padding: 3px 0; }
  .dr-rslot-hs { width: 30px; height: 30px; border-radius: 6px 6px 0 0; object-fit: cover; object-position: top center; align-self: flex-end; background: transparent; flex-shrink: 0; }
  .dr-rslot-body { flex: 1; min-width: 0; line-height: 1.2; }
  .dr-rslot-name { font-size: 12px; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-rslot-meta { font-size: 10px; color: var(--text-muted); }
  .dr-rslot-val { font-size: 12px; font-weight: 800; color: var(--text); flex-shrink: 0; }
  .dr-rslot-empty { font-size: 12px; color: var(--text-muted); font-style: italic; }
  .dr-run-line { font-size: 12px; color: var(--text-muted); margin-bottom: 10px; }
  .dr-run-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; }
  .dr-run-chip { font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: 999px; background: rgba(127,127,127,.14); color: var(--text); }
  .dr-run-hot { background: rgba(239,68,68,.16); color: #ef4444; }
  .dr-run-banner { margin: 10px 10px 4px; padding: 8px 10px; border-radius: 8px; font-size: 12px;
    background: rgba(239,68,68,.12); color: #ef4444; border: 1px solid rgba(239,68,68,.3); }
  .dr-run-banner b { color: #ef4444; }
  .dr-prev-score-hero { border: 1px solid; border-radius: 10px; padding: 12px 10px 10px; margin-bottom: 12px; text-align: center; }
  .dr-prev-score-num { font-size: 44px; font-weight: 900; line-height: 1; }
  .dr-prev-score-lbl { font-size: 9px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin-top: 2px; }
  .dr-prev-score-reason { font-size: 12px; font-weight: 600; color: var(--text-muted); margin-top: 6px; }
  .dr-empty-note { padding: 22px 14px; font-size: 12px; color: var(--text-muted); text-align: center; }
  /* tiers */
  .dr-tier { font-size: 9px; font-weight: 800; padding: 1px 5px; border-radius: 999px; margin-left: 6px;
    background: rgba(127,127,127,.18); color: var(--text-muted); vertical-align: middle; }
  .dr-tier-cliff { background: rgba(239,68,68,.16); color: #ef4444; }
  /* pick score */
  .dr-ba-reason { font-size: 10px; color: var(--text-muted); margin-top: 3px; font-weight: 600; }
  /* draft grade */
  .dr-pill-grade { background: rgba(34,197,94,.16); color: #22c55e; }
  .dr-grade-card { display: flex; align-items: center; gap: 12px; padding: 12px; margin: 10px 10px 4px;
    border: 1px solid var(--border); border-radius: 10px; background: var(--bg); }
  .dr-grade-letter { font-size: 34px; font-weight: 900; color: var(--accent,#38bdf8); line-height: 1; min-width: 48px; text-align: center; }
  .dr-grade-meta { flex: 1; min-width: 0; }
  .dr-grade-pace { font-size: 12px; font-weight: 700; color: var(--text); margin-bottom: 6px; }
  .dr-gbar-row { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
  .dr-gbar-lbl { font-size: 10px; color: var(--text-muted); width: 50px; flex-shrink: 0; }
  .dr-gbar { flex: 1; height: 6px; border-radius: 999px; background: rgba(127,127,127,.18); overflow: hidden; }
  .dr-gbar-fill { height: 100%; border-radius: 999px; background: var(--accent,#38bdf8); }
  /* player preview */
  .dr-preview-overlay { position: fixed; inset: 0; z-index: 1000; background: rgba(0,0,0,.45);
    display: flex; align-items: center; justify-content: center; padding: 16px; }
  .dr-preview-card { position: relative; width: 100%; max-width: 360px; background: var(--card);
    border: 1px solid var(--border); border-radius: 14px; padding: 18px; box-shadow: 0 16px 50px rgba(0,0,0,.3); }
  .dr-prev-close { position: absolute; top: 8px; right: 10px; background: none; border: none; font-size: 22px;
    line-height: 1; color: var(--text-muted); cursor: pointer; }
  .dr-prev-top { display: flex; align-items: flex-end; gap: 12px; margin-bottom: 14px; }
  .dr-prev-hs { width: 64px; height: 64px; border-radius: 10px 10px 0 0; object-fit: cover; object-position: top center; background: transparent; flex-shrink: 0; }
  .dr-prev-name { font-size: 18px; font-weight: 800; color: var(--text); }
  .dr-prev-meta { font-size: 12px; color: var(--text-muted); margin-top: 3px; }
  .dr-prev-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 14px; }
  .dr-prev-stat { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 8px 4px; text-align: center; }
  .dr-prev-stat-v { font-size: 15px; font-weight: 800; color: var(--text); }
  .dr-prev-stat-l { font-size: 9px; text-transform: uppercase; letter-spacing: .04em; color: var(--text-muted); margin-top: 2px; }
  .dr-prev-draft { width: 100%; }
  .dr-prev-note { font-size: 12px; color: var(--text-muted); text-align: center; padding: 6px 0; }
  /* queue star */
  .dr-star { background: none; border: none; cursor: pointer; font-size: 15px; line-height: 1; flex-shrink: 0;
    color: var(--text-muted); padding: 2px 2px 0; }
  .dr-star.on { color: #f59e0b; }
  .dr-side-title { font-size: 14px; font-weight: 800; color: var(--text); }
  .dr-side-controls { display: flex; gap: 6px; }
  .dr-side-controls input { flex: 1; min-width: 0; padding: 7px 9px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 12px; }
  .dr-side-controls select { padding: 7px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 12px; }
  .dr-pos-filters { display: flex; gap: 4px; flex-wrap: wrap; }
  .dr-pos { font-size: 11px; font-weight: 700; padding: 4px 9px; border-radius: 999px; border: 1px solid var(--border); background: var(--bg); color: var(--text-muted); cursor: pointer; }
  .dr-pos.active { background: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8); color: #fff; }
  .dr-adp-src { font-size: 10px; color: var(--text-muted); }
  .dr-ba-list { overflow-y: auto; flex: 1; }
  .dr-ba-row { display: flex; align-items: center; gap: 10px; padding: 8px 12px 8px 10px; border-bottom: 1px solid var(--border); cursor: pointer; transition: background .12s; }
  .dr-ba-row:hover { background: rgba(56,189,248,.06); }
  .dr-ba-hs { width: 40px; height: 40px; border-radius: 9px 9px 0 0; object-fit: cover; object-position: top center;
    flex-shrink: 0; background: transparent; align-self: flex-end; }
  .dr-ba-body { min-width: 0; flex: 1; line-height: 1.3; }
  .dr-ba-name { font-size: 13.5px; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-ba-meta { font-size: 11px; color: var(--text-muted); display: flex; align-items: center; gap: 6px; margin-top: 2px; }
  .dr-ba-right { text-align: right; flex-shrink: 0; display: flex; flex-direction: column; align-items: flex-end; gap: 3px; min-width: 52px; }
  .dr-ba-val { font-size: 15px; font-weight: 800; color: var(--text); line-height: 1; }
  .dr-ba-sub { font-size: 10.5px; color: var(--text-muted); line-height: 1; white-space: nowrap; }
  /* Compact pick-score chip shown in the row */
  .dr-ba-pschip { flex-shrink: 0; width: 34px; text-align: center; border-radius: 7px; padding: 4px 0;
    font-size: 13px; font-weight: 800; line-height: 1; }
  .dr-ba-pschip small { display: block; font-size: 7.5px; font-weight: 700; letter-spacing: .06em; opacity: .8; margin-top: 2px; }
  .dr-ba-draft { padding: 6px 13px; border-radius: 8px; border: 1px solid var(--accent,#122d4b);
    background: transparent; color: var(--accent,#122d4b); font-size: 11.5px; font-weight: 800;
    cursor: pointer; flex-shrink: 0; transition: background .12s, color .12s; white-space: nowrap; }
  .dr-ba-row:hover .dr-ba-draft, .dr-ba-draft:hover { background: var(--accent,#122d4b); color: #fff; }
  /* ── Future picks panel ── */
  .dr-fp-intro { font-size: 11px; color: var(--text-muted); padding: 10px 12px 6px; line-height: 1.5; }
  .dr-fp-card { margin: 8px 10px 12px; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; background: var(--card); }
  .dr-fp-head { display: flex; align-items: center; gap: 8px; padding: 9px 12px; background: rgba(56,189,248,.06); border-bottom: 1px solid var(--border); }
  .dr-fp-pick { font-size: 14px; font-weight: 800; color: var(--text); }
  .dr-fp-rd { font-size: 10px; font-weight: 700; color: var(--text-muted); background: rgba(127,127,127,.14); padding: 2px 7px; border-radius: 999px; }
  .dr-fp-when { margin-left: auto; font-size: 11px; font-weight: 700; color: var(--text-muted); }
  .dr-fp-when.dr-fp-now { color: var(--accent,#38bdf8); }
  .dr-fp-insights { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1px; background: var(--border); }
  .dr-fp-ins { background: var(--card); padding: 7px 9px; min-width: 0; }
  .dr-fp-ins-l { font-size: 8.5px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); }
  .dr-fp-ins-v { font-size: 11.5px; font-weight: 700; color: var(--text); margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-fp-ins-p { font-size: 9.5px; font-weight: 800; margin-top: 1px; }
  .dr-fp-scarce { font-size: 10.5px; font-weight: 700; color: #f59e0b; padding: 6px 12px; background: rgba(245,158,11,.08); border-top: 1px solid var(--border); }
  .dr-fp-list { padding: 2px 0; }
  .dr-fp-row { display: flex; align-items: center; gap: 9px; padding: 6px 12px; cursor: pointer; transition: background .12s; }
  .dr-fp-row:hover { background: rgba(56,189,248,.06); }
  .dr-fp-prob { flex-shrink: 0; width: 40px; text-align: center; font-size: 11px; font-weight: 800; padding: 3px 0; border-radius: 6px; }
  .dr-fp-name { flex: 1; min-width: 0; font-size: 12.5px; font-weight: 600; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: flex; align-items: center; gap: 6px; }
  .dr-fp-pos { font-size: 9px; font-weight: 800; }
  .dr-fp-vor { flex-shrink: 0; font-size: 11px; font-weight: 700; color: var(--text-muted); min-width: 34px; text-align: right; }
  .dr-fp-val { flex-shrink: 0; font-size: 12.5px; font-weight: 800; color: var(--text); min-width: 38px; text-align: right; }
  .dr-fp-empty { font-size: 11px; color: var(--text-muted); font-style: italic; padding: 10px 12px; }
  .dr-fp-gone { font-size: 10.5px; color: var(--text-muted); padding: 7px 12px; border-top: 1px solid var(--border); line-height: 1.4; }
  .dr-fp-gone-lbl { font-weight: 800; color: #ef4444; }
  .dr-loading { display: flex; align-items: center; gap: 10px; padding: 24px; color: var(--text-muted); font-size: 13px; justify-content: center; }
  @media (max-width: 900px) {
    .dr-cols { grid-template-columns: 1fr; }
    .dr-side { position: static; max-height: none; order: -1; }
    .dr-statusbar { top: 0; }
  }
  @media (max-width: 640px) {
    .dr-wrap { padding: 8px 8px 32px; }
    .dr-setup-card { padding: 16px; }
    .dr-statusbar { padding: 8px; gap: 6px; }
    .dr-status-left { gap: 6px; }
    .dr-onclock { width: 100%; }
    .dr-side-tabs .otc-main-tab { padding: 8px 12px; }
    .dr-side { max-height: 60vh; }            /* keep the board reachable below */
    .dr-ba-list { max-height: 52vh; }
    .dr-board-wrap { padding: 4px; }
    .dr-cta, .dr-setup-cta { flex-direction: column; align-items: stretch; }
    .dr-setup-cta .dr-btn { width: 100%; }
    .dr-prev-stats { grid-template-columns: repeat(2, 1fr); }
  }
  /* Summary overlay */
  .dr-summary-overlay { position:fixed; inset:0; z-index:1001; background:rgba(0,0,0,.55);
    display:flex; align-items:flex-start; justify-content:center; padding:16px; overflow-y:auto; }
  .dr-summary-card { position:relative; width:100%; max-width:480px; margin:auto; background:var(--card);
    border:1px solid var(--border); border-radius:16px; padding:20px 18px;
    box-shadow:0 16px 60px rgba(0,0,0,.35); }
  .dr-sum-header { text-align:center; margin-bottom:12px; }
  .dr-sum-title { font-size:12px; font-weight:800; text-transform:uppercase; letter-spacing:.06em; color:var(--text-muted); }
  .dr-sum-grade { font-size:52px; font-weight:900; line-height:1; margin:6px 0 4px; }
  .dr-sum-pace { font-size:12px; color:var(--text-muted); }
  .dr-sum-section { font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.06em;
    color:var(--text-muted); margin:12px 0 4px; padding-top:4px; }
  .dr-sum-row { display:flex; align-items:center; gap:8px; padding:6px 0; border-bottom:1px solid var(--border); }
  .dr-sum-slot-badge { font-size:9px; font-weight:800; color:#fff; border-radius:4px; padding:3px 0;
    width:34px; flex-shrink:0; text-align:center; }
  .dr-sum-hs { width:30px; height:30px; border-radius:5px 5px 0 0; object-fit:cover;
    object-position:top center; flex-shrink:0; align-self:flex-end; background:transparent; }
  .dr-sum-body { flex:1; min-width:0; line-height:1.3; }
  .dr-sum-name { font-size:12px; font-weight:700; color:var(--text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .dr-sum-meta { font-size:10px; color:var(--text-muted); }
  .dr-sum-reason { font-size:10px; color:var(--text-muted); font-style:italic; }
  .dr-sum-empty { font-size:11px; color:var(--text-muted); font-style:italic; }
  .dr-sum-ps { font-size:14px; font-weight:800; flex-shrink:0; }
  .dr-sum-footer { display:flex; gap:8px; margin-top:16px; }
  .dr-sum-footer .dr-btn { flex:1; text-align:center; }
  /* ── Roster slots (setup page) ── */
  .dr-setup-roster { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:8px; }
  .dr-srow { display:flex; align-items:center; justify-content:space-between; gap:8px;
    background:var(--bg); border:1px solid var(--border); border-radius:9px; padding:8px 11px; min-height:40px; }
  .dr-srow-label { font-size:13px; font-weight:700; color:var(--text); }
  .dr-stepper { display:flex; align-items:center; gap:8px; }
  .dr-step-btn { width:26px; height:26px; border-radius:6px; border:1px solid var(--border);
    background:var(--card); color:var(--text); font-size:16px; font-weight:700; cursor:pointer; line-height:1;
    display:flex; align-items:center; justify-content:center; padding:0; flex-shrink:0; }
  .dr-step-btn:hover { border-color:var(--accent,#38bdf8); color:var(--accent,#38bdf8); }
  .dr-step-val { font-size:14px; font-weight:800; color:var(--text); min-width:18px; text-align:center; }
  /* ── Team tab PS badge ── */
  .dr-rslot-ps { font-size:11px; font-weight:800; flex-shrink:0; margin-right:2px; }
</style>

<script>
(function(){
  var cfg = window.__draftCfg || {};
  var POS_COLOR = { QB:'#f59e0b', RB:'#22c55e', WR:'#3b82f6', TE:'#8b5cf6', K:'#94a3b8', DEF:'#64748b' };
  var posColor = function(p){ return POS_COLOR[(p||'').toUpperCase()] || '#94a3b8'; };
  var hsUrl = function(id){ return 'https://sleepercdn.com/content/nfl/players/' + id + '.jpg'; };

  var sessKey = 'dr_' + location.pathname;
  var state = null;        // { type, teams, rounds, sf, slot, order, picks:{}, current }
  var players = [];        // best-available pool
  var drafted = {};        // id -> true
  var posFilter = 'ALL';
  var justPick = null;     // pick # filled this render (for the pop-in animation)
  var playersById = {};    // id -> player (value lookup for live picks)
  var lastLivePicks = null;// last picks payload from the live feed
  var saveTimer = null;    // debounce for DB autosave
  var pollTimer = null;    // live-draft poll interval
  var sim = false;         // mock-draft simulation active
  var simTimer = null;
  var simSpeed = 700;      // ms between CPU picks
  var simPaused = false;
  var sideTab = 'best';    // best | rec | needs | runs
  var _setupRoster = null; // roster config built on setup page
  var _setupOwned = null;  // claimed picks (pickNumber -> true) built on setup page
  var tierThresholds = {}; // {leagueType:{size:[...]}} from /api/league-players
  var adpSources = {};     // {startup|rookie|redraft: 'Sleeper'|'none'} from /api/league-players
  var _boardSig = null;    // board structure signature (rebuild only when it changes)
  var _summaryShown = false; // auto-open summary only once per draft

  // ── Pick-order helper (snake / linear / 3rr) ───────────────────────────────
  function pickDir(r, order){            // true = forward (slot 1 → N)
    if (order === 'linear') return true;
    if (order === '3rr') {
      if (r === 1) return true;
      if (r === 2 || r === 3) return false;
      return (r % 2 === 0);
    }
    return (r % 2 === 1);                 // snake
  }
  function pickNum(r, slot, teams, order){
    var inRound = pickDir(r, order) ? slot : (teams - slot + 1);
    return (r - 1) * teams + inRound;
  }
  function slotOnClock(pickNo, teams, order){
    var r = Math.ceil(pickNo / teams);
    var posInRound = pickNo - (r - 1) * teams;
    return pickDir(r, order) ? posInRound : (teams - posInRound + 1);
  }
  window.draftPickOrder = { pickDir: pickDir, pickNum: pickNum, slotOnClock: slotOnClock };

  // ── Pick ownership ─────────────────────────────────────────────────────────
  // The user can own any set of picks (multiple firsts via trades, etc.).
  // state.owned maps pickNumber -> true. When absent we fall back to the home
  // slot so older saved sessions keep working.
  function isMyPick(pn){
    if (state && state.owned) return !!state.owned[pn];
    return !!(state && state.slot && slotOnClock(pn, state.teams, state.order) === state.slot);
  }
  function defaultOwned(){
    var o = {};
    if (!state || !state.slot) return o;
    var tot = state.teams * state.rounds;
    for (var pn = 1; pn <= tot; pn++){
      if (slotOnClock(pn, state.teams, state.order) === state.slot) o[pn] = true;
    }
    return o;
  }
  function ensureOwned(){ if (state && !state.owned) state.owned = defaultOwned(); }
  function ownedPicks(){
    ensureOwned();
    if (!state || !state.owned) return [];
    return Object.keys(state.owned).filter(function(k){ return state.owned[k]; })
      .map(Number).sort(function(a, b){ return a - b; });
  }
  function hasOwned(){ return ownedPicks().length > 0; }
  // Upcoming owned picks that have not been made yet (current pick included).
  function upcomingOwnedPicks(){
    return ownedPicks().filter(function(pn){ return pn >= state.current && !state.picks[pn]; });
  }
  function toggleOwned(pn){
    ensureOwned();
    if (state.owned[pn]) delete state.owned[pn]; else state.owned[pn] = true;
    save();
  }

  // ── Persistence ────────────────────────────────────────────────────────────
  function save(){ try { sessionStorage.setItem(sessKey, JSON.stringify(state)); } catch(e){} }
  function load(){ try { return JSON.parse(sessionStorage.getItem(sessKey) || 'null'); } catch(e){ return null; } }

  function fillSlotOptions(teams){
    var sel = document.getElementById('drSlot');
    sel.innerHTML = '';
    for (var i = 1; i <= teams; i++){
      var o = document.createElement('option');
      o.value = i; o.textContent = 'Pick ' + i;
      sel.appendChild(o);
    }
  }

  // ── Setup ────────────────────────────────────────────────────────────────
  function applyCfgDefaults(){
    if (cfg.numTeams) {
      var t = document.getElementById('drTeams');
      var want = String(Math.min(14, Math.max(8, cfg.numTeams)));
      for (var i=0;i<t.options.length;i++){ if (t.options[i].value === want || t.options[i].text === want){ t.selectedIndex = i; break; } }
    }
    if (cfg.isSuperflex) document.getElementById('drSf').value = '1';
    fillSlotOptions(parseInt(document.getElementById('drTeams').value, 10));
  }

  document.getElementById('drTeams').addEventListener('change', function(){
    fillSlotOptions(parseInt(this.value, 10));
  });
  document.getElementById('drType').addEventListener('change', function(){
    document.getElementById('drRounds').value = (this.value === 'rookie') ? '4' : '15';
  });

  // Map a Sleeper-style roster_positions list into our slot counts.
  function rosterFromLeague(){
    var rp = cfg.rosterPositions;
    if (!rp || !rp.length) return null;
    var r = { QB:0, SF:0, RB:0, WR:0, TE:0, FLEX:0, K:0, DEF:0, BN:0 };
    var map = {
      QB:'QB', RB:'RB', WR:'WR', TE:'TE',
      FLEX:'FLEX', WRRB_FLEX:'FLEX', REC_FLEX:'FLEX', WRRBTE_FLEX:'FLEX',
      SUPER_FLEX:'SF', SFLEX:'SF',
      K:'K', DEF:'DEF', DST:'DEF', BN:'BN'
    };
    rp.forEach(function(s){
      var key = map[String(s).toUpperCase()];
      if (key) r[key]++;            // IDP/TAXI/IR positions are ignored
    });
    if (!(r.QB+r.RB+r.WR+r.TE+r.FLEX+r.SF)) return null;  // no usable starters
    return r;
  }
  function defaultRoster(sf, rd){
    if (sf === undefined) sf = state && state.sf;
    if (rd === undefined) rd = state && state.type === 'redraft';
    // Prefer the connected league's actual roster shape when available.
    var lg = rosterFromLeague();
    if (lg){
      // Keep K/DEF only for redraft; dynasty/rookie boards skip them.
      if (!rd){ lg.K = 0; lg.DEF = 0; }
      return lg;
    }
    return { QB:1, SF:sf?1:0, RB:2, WR:3, TE:1, FLEX:sf?0:1,
             K:rd?1:0, DEF:rd?1:0, BN:rd?5:7 };
  }

  function renderSetupRoster(){
    var sf = document.getElementById('drSf').value === '1';
    var rd = document.getElementById('drType').value === 'redraft';
    if (!_setupRoster || _setupRoster._sf !== sf || _setupRoster._rd !== rd){
      var base = defaultRoster(sf, rd);
      base._sf = sf; base._rd = rd;
      _setupRoster = base;
    }
    var rows = [
      { key:'QB',   label:'QB' },
      { key:'SF',   label:'SF',   hide: !sf },
      { key:'RB',   label:'RB' },
      { key:'WR',   label:'WR' },
      { key:'TE',   label:'TE' },
      { key:'FLEX', label:'FLEX', hide: sf },
      { key:'K',    label:'K',    hide: !rd },
      { key:'DEF',  label:'DEF',  hide: !rd },
      { key:'BN',   label:'Bench' }
    ];
    var html = '<div class="dr-setup-roster">';
    rows.forEach(function(r){
      if (r.hide) return;
      var val = _setupRoster[r.key] || 0;
      html += '<div class="dr-srow">'
        + '<span class="dr-srow-label">' + r.label + '</span>'
        + '<div class="dr-stepper">'
        + '<button type="button" class="dr-step-btn" data-key="' + r.key + '" data-d="-1">&#8722;</button>'
        + '<span class="dr-step-val">' + val + '</span>'
        + '<button type="button" class="dr-step-btn" data-key="' + r.key + '" data-d="1">+</button>'
        + '</div></div>';
    });
    html += '</div>';
    document.getElementById('drRosterSection').innerHTML = html;
  }

  function readSetup(){
    var teams = parseInt(document.getElementById('drTeams').value, 10);
    var sf = document.getElementById('drSf').value === '1';
    var rd = document.getElementById('drType').value === 'redraft';
    return {
      type:   document.getElementById('drType').value,
      teams:  teams,
      rounds: Math.max(1, Math.min(40, parseInt(document.getElementById('drRounds').value, 10) || 15)),
      sf:     sf,
      slot:   Math.min(teams, Math.max(1, parseInt(document.getElementById('drSlot').value, 10) || 1)),
      order:  document.getElementById('drOrder').value,
      roster: _setupRoster || defaultRoster(sf, rd),
      picks:  {},
      current: 1,
      queue:  []
    };
  }

  function startDraft(){
    state = readSetup();
    state.owned = _setupOwned || defaultOwned();
    _summaryShown = false;
    drafted = {};
    save();
    showMain();
    loadPlayers();
  }

  function showMain(){
    _boardSig = null;   // always force a full board rebuild when entering the draft view
    document.getElementById('drSetup').style.display = 'none';
    document.getElementById('drBoard').innerHTML = '';
    document.getElementById('drBaList').innerHTML = '<div class="dr-loading">Loading players…</div>';
    document.getElementById('drMain').style.display = '';
  }
  function showSetup(){
    endSim();
    document.getElementById('drMain').style.display = 'none';
    document.getElementById('drSetup').style.display = '';
    renderSetupRoster();
  }

  // ── Data ─────────────────────────────────────────────────────────────────
  function redraftVal(p){
    return (state.sf ? (p.redraft_value_sf != null ? p.redraft_value_sf : p.redraft_value_1qb)
                     : p.redraft_value_1qb) || 0;
  }
  function valOf(p){
    if (state.type === 'redraft') return redraftVal(p);
    return state.sf ? (p.sf_value || p.value || 0) : (p.value || 0);
  }
  function adpOf(p){
    // Sleeper community ADP (server-side, aggregated from real Sleeper drafts).
    // Redraft has no Sleeper feed, so it falls back to a value-derived rank.
    if (state.type === 'rookie') return state.sf ? p.sf_rookie_avg_pick : p.rookie_avg_pick;
    if (state.type === 'redraft'){
      var ra = state.sf ? p.sf_redraft_avg_pick : p.redraft_avg_pick;
      return (ra != null) ? ra : (p._radp != null ? p._radp : null);
    }
    return state.sf ? p.sf_avg_pick : p.avg_pick;
  }

  function loadPlayers(){
    var url = '/api/league-players' + (state.type === 'redraft' ? '?kdef=1' : '');
    fetch(url, { cache: 'no-store' })
      .then(function(r){ return r.json(); })
      .then(function(resp){
        var raw = Array.isArray(resp) ? resp : (resp.players || []);
        tierThresholds = (!Array.isArray(resp) && resp.tier_thresholds) ? resp.tier_thresholds : {};
        adpSources = (!Array.isArray(resp) && resp.adp_sources) ? resp.adp_sources : {};
        players = raw.filter(function(p){
          if (!p || p.id == null) return false;
          var pos = String(p.position || '').toUpperCase();
          if (pos === 'PICK') return false;
          if (state.type === 'rookie') return !!p.is_rookie;
          if (state.type === 'redraft') return redraftVal(p) > 0 || pos === 'K' || pos === 'DEF';
          return ['QB','RB','WR','TE'].indexOf(pos) >= 0 || p.is_rookie;
        });
        // Derive a stable redraft ADP rank (1 = top redraft value).
        if (state.type === 'redraft'){
          players.slice().sort(function(a, b){ return redraftVal(b) - redraftVal(a); })
            .forEach(function(p, i){ p._radp = i + 1; });
        }
        playersById = {};
        players.forEach(function(p){ playersById[String(p.id)] = p; });
        // Live mode: re-apply picks now that values are available; else rebuild
        // the drafted set from saved picks.
        if (state.mode === 'live' && lastLivePicks){
          applyLivePicks(lastLivePicks);
        } else {
          drafted = {};
          Object.keys(state.picks).forEach(function(k){ var pp = state.picks[k]; if (pp) drafted[String(pp.id)] = true; });
        }
        render();
        if (sim) scheduleSim();   // begin CPU picks once players are loaded
      })
      .catch(function(){
        document.getElementById('drBaList').innerHTML =
          '<div class="dr-loading">Could not load players. Refresh to retry.</div>';
      });
  }

  // ── Render ───────────────────────────────────────────────────────────────
  function render(){
    if (state && !state.queue) state.queue = [];
    renderStatus(); renderBoard(); renderSide(); justPick = null; save();
    var _tot = state.teams * state.rounds;
    if (state.current > _tot && !_summaryShown && hasOwned() && !sim){
      _summaryShown = true;
      setTimeout(openSummary, 500);
    }
  }

  // ── Simulation (mock draft) ─────────────────────────────────────────────────
  function simAdp(p){
    var a = adpOf(p);
    return (a != null) ? a : (10000 - (valOf(p) / 100));  // ADP-less players sort after, by value
  }
  function teamCounts(slot){
    var c = { QB:0, RB:0, WR:0, TE:0, K:0, DEF:0 };
    Object.keys(state.picks).forEach(function(k){
      if (slotOnClock(parseInt(k,10), state.teams, state.order) === slot){
        var pos = (state.picks[k].position||'').toUpperCase(); if (c[pos] != null) c[pos]++;
      }
    });
    return c;
  }
  function simPick(){
    var pool = availablePool();
    if (!pool.length) return null;
    // CPU picks follow ADP but are need-aware: needed positions get pulled a bit
    // earlier and already-filled positions pushed later, so rosters stay realistic.
    var slot = slotOnClock(state.current, state.teams, state.order);
    var counts = teamCounts(slot), targets = posTargets();
    pool.forEach(function(p){
      var pos = (p.position||'').toUpperCase();
      var t = targets[pos] || 0, have = counts[pos] || 0;
      var need = t ? Math.max(0, t - have) / t : 0;
      var over = (t && have >= t) ? (have - t + 1) : 0;
      // Keep adjustments small so elite players don't slip far past their ADP.
      p._eff = simAdp(p) * (1 - 0.08 * need) * (1 + 0.22 * over);
    });
    pool.sort(function(a, b){ return a._eff - b._eff; });
    // Tighter window (3) keeps picks near ADP while still allowing light variance.
    var win = Math.min(pool.length, 3);
    var idx = Math.floor(win * (1 - Math.sqrt(1 - Math.random())));
    if (idx >= win) idx = win - 1;
    return pool[idx];
  }
  function scheduleSim(){
    if (!sim || simPaused) return;
    var total = state.teams * state.rounds;
    if (state.current > total){ endSim(); return; }
    if (isMyPick(state.current)) return; // wait for the user
    clearTimeout(simTimer);
    simTimer = setTimeout(simStep, simSpeed);
  }
  function simStep(){
    if (!sim || simPaused) return;
    var total = state.teams * state.rounds;
    if (state.current > total){ endSim(); render(); return; }
    if (isMyPick(state.current)){ render(); return; } // your turn
    var p = simPick();
    if (p) { commitPick(p); render(); }
    scheduleSim();
  }
  function endSim(){
    sim = false; clearTimeout(simTimer);
    var sp = document.getElementById('drSimSpeed'); if (sp) sp.style.display = 'none';
    var tg = document.getElementById('drSimToggle'); if (tg) tg.style.display = 'none';
  }
  function toggleSim(){
    simPaused = !simPaused;
    document.getElementById('drSimToggle').textContent = simPaused ? 'Resume' : 'Pause';
    if (simPaused) clearTimeout(simTimer); else scheduleSim();
  }
  function startMock(){
    state = readSetup();
    state.owned = _setupOwned || defaultOwned();
    _summaryShown = false;
    drafted = {};
    sim = true; simPaused = false;
    var sp = document.getElementById('drSimSpeed');
    simSpeed = parseInt(sp.value, 10) || 700;
    sp.style.display = '';
    var tg = document.getElementById('drSimToggle'); tg.style.display = ''; tg.textContent = 'Pause';
    save();
    showMain();
    loadPlayers();
  }

  // ── Command-center panels ───────────────────────────────────────────────────
  function availablePool(){ return players.filter(function(p){ return !drafted[String(p.id)]; }); }
  function myPicksList(){
    var out = [];
    if (!hasOwned()) return out;
    Object.keys(state.picks).forEach(function(k){
      var pn = parseInt(k, 10);
      if (isMyPick(pn)) out.push(state.picks[k]);
    });
    return out;
  }
  function posTargets(){
    var rs = (state && state.roster) || defaultRoster();
    var flex = rs.FLEX||0, sf = rs.SF||0, bn = rs.BN||0;
    var t = {
      QB: (rs.QB||0) + sf        + Math.round(bn * 0.10),
      RB: (rs.RB||0) + flex      + Math.round(bn * 0.35),
      WR: (rs.WR||0)             + Math.round(bn * 0.40),
      TE: (rs.TE||0)             + Math.round(bn * 0.15)
    };
    if (rs.K)   t.K   = rs.K;
    if (rs.DEF) t.DEF = rs.DEF;
    return t;
  }
  function myPosCounts(){
    var c = { QB:0, RB:0, WR:0, TE:0, K:0, DEF:0 };
    myPicksList().forEach(function(p){ var pos = (p.position||'').toUpperCase(); if (c[pos] != null) c[pos]++; });
    return c;
  }
  function listInto(html){ document.getElementById('drBaList').innerHTML = html; }

  // ── Tiers + cliffs ──────────────────────────────────────────────────────────
  function tierOf(p){
    if (state.type === 'redraft') return null;   // tiers are keyed to dynasty value
    var lt = state.sf ? 'sf' : '1qb';
    var sz = String(state.teams);
    var tbl = (tierThresholds[lt] || {})[sz] || (tierThresholds['1qb'] || {})['10'] || [];
    if (!tbl.length) return null;
    var v = valOf(p);
    for (var i = 0; i < tbl.length; i++){ if (v >= tbl[i]) return i + 1; }
    return tbl.length + 1;
  }
  // Count of still-available players per (position|tier) — drives cliff alerts.
  function posTierCounts(){
    var m = {};
    availablePool().forEach(function(p){
      var t = tierOf(p); if (t == null) return;
      var k = (p.position || '').toUpperCase() + '|' + t;
      m[k] = (m[k] || 0) + 1;
    });
    return m;
  }
  var _ptc = {};   // refreshed each render
  function isTierCliff(p){
    var t = tierOf(p); if (t == null) return false;
    var k = (p.position || '').toUpperCase() + '|' + t;
    return (_ptc[k] || 0) <= 2;     // tier is drying up
  }
  // Top-tier (T1+T2) players still available at a position - scarcity signal.
  function posTopRemaining(pos){
    pos = (pos || '').toUpperCase();
    return (_ptc[pos + '|1'] || 0) + (_ptc[pos + '|2'] || 0);
  }

  // ── Value Over Replacement (VOR) ────────────────────────────────────────────
  // Replacement level = value of the last startable player at a position across
  // the league (teams x starters-per-team). VOR(p) = value(p) - replacement.
  var _repl = {};   // refreshed each render
  function computeReplacement(){
    var rs = (state && state.roster) || defaultRoster();
    var teams = state.teams || 12;
    var flex = rs.FLEX || 0, sf = rs.SF || 0;
    var starters = {
      QB: (rs.QB || 0) + sf * 0.5,
      RB: (rs.RB || 0) + flex * 0.5,
      WR: (rs.WR || 0) + flex * 0.5,
      TE: (rs.TE || 0)
    };
    var byPos = { QB: [], RB: [], WR: [], TE: [] };
    players.forEach(function(p){
      var pos = (p.position || '').toUpperCase();
      if (byPos[pos]) byPos[pos].push(valOf(p));
    });
    var r = {};
    Object.keys(byPos).forEach(function(pos){
      var arr = byPos[pos]; arr.sort(function(a, b){ return b - a; });
      if (!arr.length){ r[pos] = 0; return; }
      var idx = Math.round(teams * (starters[pos] || 1)) - 1;
      if (idx < 0) idx = 0; if (idx >= arr.length) idx = arr.length - 1;
      r[pos] = arr[idx];
    });
    return r;
  }
  function vorOf(p){
    var pos = (p.position || '').toUpperCase();
    if (_repl[pos] == null) return null;
    return Math.round(valOf(p) - _repl[pos]);
  }

  // ── Availability probability ────────────────────────────────────────────────
  // Model a player's actual draft slot as Normal(ADP, sigma); the chance they
  // survive to overall pick `pn` is P(slot >= pn) = 1 - CDF((pn - ADP)/sigma).
  function _normCdf(z){
    var t = 1 / (1 + 0.2316419 * Math.abs(z));
    var d = 0.3989423 * Math.exp(-z * z / 2);
    var p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return z >= 0 ? 1 - p : p;
  }
  function availProb(p, pn){
    var a = adpOf(p);
    if (a == null) return null;
    var sigma = Math.max(4, Math.min(16, 0.16 * a + 4));  // spread widens later in drafts
    var z = (pn - a) / sigma;
    return Math.round((1 - _normCdf(z)) * 100);
  }
  function availColor(pct){ return pct >= 65 ? '#22c55e' : pct >= 40 ? '#f59e0b' : '#ef4444'; }

  // ── Pick Score (composite) ──────────────────────────────────────────────────
  // Fuses talent (value), value-vs-ADP, positional need, tier strength + cliff,
  // momentum and youth into one 0-100 number per available player.
  function clamp01(x){ return x < 0 ? 0 : (x > 1 ? 1 : x); }
  function pickScore(p, maxVal, counts){
    var pos = (p.position || '').toUpperCase();
    var valueNorm = maxVal > 0 ? clamp01(valOf(p) / maxVal) : 0;
    var t = posTargets()[pos];
    // Need ramps in from 0 at pick 1 to full weight by pick 13 — early picks
    // should be pure talent grabs, not roster-need influenced.
    var needRaw = t ? clamp01(Math.max(0, t - (counts[pos] || 0)) / t) : 0;
    var needRamp = clamp01((state.current - 1) / 12);
    // Blend toward neutral (0.5) in early picks rather than zeroing out the weight —
    // dropping to 0 silently removes 18% of the formula and caps elite players at ~82.
    var need = (1 - needRamp) * 0.5 + needRamp * needRaw;
    var adp = adpOf(p);
    // Steal vs reach: CPU sim now picks within ~2-3 spots of ADP, so any fall
    // beyond 4 is a real value signal and a reach beyond 4 is a real cost.
    // Neutral zone tightened from ±8 to ±4 to match sim fidelity.
    var adpVal;
    if (adp != null) {
      var gap = state.current - adp;
      if (gap >= 0)       adpVal = clamp01(gap / 14 + 0.5);           // steal: 0.5 → 1.0 over 7 picks
      else if (gap >= -4) adpVal = 0.5;                                // neutral zone: within 4 of ADP
      else                adpVal = clamp01((gap + 4) / 28 + 0.5);     // reach penalty beyond 4
    } else { adpVal = 0.5; }
    var tier = tierOf(p);
    var tierNorm = tier ? (10 - Math.min(tier, 9)) / 9 : valueNorm;
    if (isTierCliff(p)) tierNorm = clamp01(tierNorm + 0.15);         // urgency bump
    var mom = clamp01((p.rank_change_7d || 0) / 20 + 0.5);
    var age = (p.age != null) ? Number(p.age) : null;
    var youth = (age != null && ['RB','WR','TE','QB'].indexOf(pos) >= 0) ? clamp01((30 - age) / 12) : 0.5;
    // Value + tier are the primary drivers; ADP is a secondary signal.
    var s = 0.48*valueNorm + 0.18*need + 0.10*adpVal + 0.14*tierNorm + 0.05*mom + 0.05*youth;
    return Math.round(s * 100);
  }
  // How many players remain in this player's (position|tier) bucket.
  function tierRemaining(p){
    var t = tierOf(p); if (t == null) return null;
    return _ptc[(p.position || '').toUpperCase() + '|' + t] || 0;
  }
  function pickReason(p, counts){
    var pos = (p.position || '').toUpperCase();
    var t = posTargets()[pos];
    var need = t ? Math.max(0, t - (counts[pos] || 0)) : 0;
    var adp = adpOf(p);
    var fell = (adp != null) ? Math.round(state.current - adp) : null;
    var tier = tierOf(p);
    var left = tierRemaining(p);
    // Tier cliff: the most urgent signal — name the tier and how many remain.
    if (isTierCliff(p) && tier != null){
      if (left <= 1) return 'Last ' + pos + ' in Tier ' + tier + ' — grab now';
      return 'Only ' + left + ' ' + pos + 's left in Tier ' + tier;
    }
    // Big steal: fell well past ADP.
    if (fell != null && fell >= 8) return 'Steal — fell ' + fell + ' picks past ADP';
    // Roster need — only surface this after the first 4 picks.
    if (need > 0 && state.current > 4){
      if (tier != null && tier <= 2) return 'Tier ' + tier + ' ' + pos + ' fills a need';
      return 'Fills ' + pos + ' need (' + need + ' more to target)';
    }
    // Falls a little past ADP without being a full cliff.
    if (fell != null && fell >= 3) return 'Good value — ' + fell + ' past ADP';
    if (tier != null && tier <= 2) return 'Elite tier (T' + tier + ') talent';
    return 'Best available';
  }

  function tierBadge(p){
    var t = tierOf(p); if (t == null) return '';
    var cliff = isTierCliff(p) ? ' dr-tier-cliff' : '';
    return '<span class="dr-tier' + cliff + '">T' + t + '</span>';
  }

  // ── Queue / targets ─────────────────────────────────────────────────────────
  function isQueued(id){ return !!(state.queue && state.queue.indexOf(String(id)) >= 0); }
  function toggleQueue(id){
    id = String(id);
    if (!state.queue) state.queue = [];
    var i = state.queue.indexOf(id);
    if (i >= 0) state.queue.splice(i, 1); else state.queue.push(id);
    save(); renderSide();
  }

  // opts: { reason: text shown under the meta line, sub: overrides the ADP sub-line }
  function playerRowHtml(p, opts){
    opts = opts || {};
    var adp = adpOf(p);
    var ps = pickScoreFor(p);
    var sub = (opts.sub != null) ? opts.sub : (adp != null ? 'ADP ' + Number(adp).toFixed(1) : '');
    var reasonLine = opts.reason ? '<div class="dr-ba-reason">' + esc(opts.reason) + '</div>' : '';
    var psChip = (ps != null)
      ? '<div class="dr-ba-pschip" style="color:' + psColor(ps) + ';background:' + psColor(ps) + '1a;">' + ps + '<small>PS</small></div>'
      : '';
    return '<div class="dr-ba-row" data-id="' + esc(String(p.id)) + '">'
      + '<button class="dr-star' + (isQueued(p.id) ? ' on' : '') + '" data-star="' + esc(String(p.id)) + '" title="Queue" aria-label="Queue">' + (isQueued(p.id) ? '★' : '☆') + '</button>'
      + '<img class="dr-ba-hs" src="' + hsUrl(p.id) + '" alt="" onerror="this.style.visibility=\'hidden\'">'
      + '<div class="dr-ba-body"><div class="dr-ba-name">' + esc(p.name) + tierBadge(p) + '</div>'
      + '<div class="dr-ba-meta"><span class="dr-posbadge" style="background:' + posColor(p.position) + '">' + esc(p.position) + '</span>' + esc(p.team || '') + '</div>'
      + reasonLine + '</div>'
      + '<div class="dr-ba-right"><div class="dr-ba-val">' + Math.round(valOf(p)) + '</div><div class="dr-ba-sub">' + sub + '</div></div>'
      + psChip
      + '<button class="dr-ba-draft" data-draft="' + esc(String(p.id)) + '" title="Draft now">Draft</button>'
      + '</div>';
  }

  function renderQueue(){
    var q = (state.queue || []).map(function(id){ return playersById[String(id)]; })
      .filter(function(p){ return p && !drafted[String(p.id)]; });
    if (!q.length){ listInto('<div class="dr-empty-note">Your queue is empty. Tap the ☆ on any player to add a target.</div>'); return; }
    var html = ''; q.forEach(function(p){ html += playerRowHtml(p); });
    listInto(html);
  }

  function renderSide(){
    _ptc = posTierCounts();
    _repl = computeReplacement();
    var kdef = (state.type === 'redraft');
    var kbtns = document.querySelectorAll('.dr-pos-kdef');
    for (var i = 0; i < kbtns.length; i++){ kbtns[i].style.display = kdef ? '' : 'none'; }
    var bc = document.getElementById('drBestControls');
    if (bc) bc.style.display = (sideTab === 'best') ? '' : 'none';
    if (sideTab === 'rec')      return renderRec();
    if (sideTab === 'future')   return renderFuture();
    if (sideTab === 'queue')    return renderQueue();
    if (sideTab === 'needs')    return renderNeeds();
    return renderBA();
  }


  // Positional-run alert banner (folded into Recs): fires when 3+ of the last 5
  // picks share a position. Returns '' when there's no active run.
  function runBanner(){
    var last5 = { QB:0, RB:0, WR:0, TE:0, K:0, DEF:0 }, n = 0;
    for (var pn = state.current - 1; pn >= 1 && n < 5; pn--){
      var p = state.picks[pn]; if (!p) continue;
      var pos = (p.position||'').toUpperCase(); if (last5[pos] != null) last5[pos]++; n++;
    }
    var hot = '';
    ['RB','WR','QB','TE'].forEach(function(pos){ if (!hot && last5[pos] >= 3) hot = pos; });
    if (!hot) return '';
    return '<div class="dr-run-banner"><i class="fa-solid fa-fire"></i> <b>' + hot + ' run</b>: ' + last5[hot]
      + ' of the last 5 picks. Weigh your ' + hot + ' need before the tier dries up.</div>';
  }

  function renderRec(){
    if (!hasOwned()){ listInto('<div class="dr-empty-note">Set your pick slot to get personalized recommendations.</div>'); return; }
    var counts = myPosCounts();
    var pool = availablePool().slice();
    if (!pool.length){ listInto('<div class="dr-empty-note">No players available.</div>'); return; }
    var maxVal = 0; pool.forEach(function(p){ var v = valOf(p); if (v > maxVal) maxVal = v; });
    pool.forEach(function(p){ p._ps = pickScore(p, maxVal, counts); });
    pool.sort(function(a, b){ return b._ps - a._ps; });
    var html = runBanner();
    for (var i = 0; i < Math.min(pool.length, 50); i++){
      var p = pool[i];
      html += playerRowHtml(p, { reason: pickReason(p, counts) });
    }
    listInto(html);
  }

  // ── Future picks (availability projections) ─────────────────────────────────
  function renderFuture(){
    if (!hasOwned()){ listInto('<div class="dr-empty-note">Claim a draft slot to project your upcoming picks.</div>'); return; }
    var total = state.teams * state.rounds;
    var upcoming = upcomingOwnedPicks();
    if (state.current > total || !upcoming.length){ listInto('<div class="dr-empty-note">No upcoming picks to project.</div>'); return; }
    var pool = availablePool();
    var html = '<div class="dr-fp-intro">Players projected available at your upcoming picks, with the chance each is still on the board.</div>';
    for (var i = 0; i < upcoming.length && i < 4; i++){
      html += futurePickCard(upcoming[i], pool);
    }
    listInto(html);
  }
  function futurePickCard(pn, pool){
    var r = Math.ceil(pn / state.teams);
    var away = pn - state.current;
    var isNow = (pn === state.current);
    var scored = pool.map(function(p){
      return { p: p, prob: availProb(p, pn), val: valOf(p), vor: vorOf(p), adp: adpOf(p) };
    });
    var avail = scored.filter(function(s){ return s.prob != null && s.prob >= 35; })
      .sort(function(a, b){ return b.val - a.val; });
    var gone = scored.filter(function(s){ return s.prob != null && s.prob < 35; })
      .sort(function(a, b){ return b.val - a.val; }).slice(0, 4);
    var bestAvail = avail[0];
    var bestVor = avail.slice().sort(function(a, b){ return (b.vor == null ? -1e9 : b.vor) - (a.vor == null ? -1e9 : a.vor); })[0];
    var bestValue = avail.slice().sort(function(a, b){ return (b.adp - pn) - (a.adp - pn); })[0];

    var when = isNow ? '<span class="dr-fp-when dr-fp-now">On the clock</span>'
                     : '<span class="dr-fp-when">' + away + ' pick' + (away === 1 ? '' : 's') + ' away</span>';
    var h = '<div class="dr-fp-card">'
      + '<div class="dr-fp-head"><span class="dr-fp-pick">Pick #' + pn + '</span>'
      + '<span class="dr-fp-rd">R' + r + '</span>' + when + '</div>';

    h += '<div class="dr-fp-insights">';
    if (bestAvail) h += fpInsight('Best available', bestAvail.p.name, bestAvail.prob);
    if (bestVor && bestVor.vor != null) h += fpInsight('Highest VOR', bestVor.p.name + ' (+' + bestVor.vor + ')', bestVor.prob);
    if (bestValue && bestValue.adp != null && (bestValue.adp - pn) > 0) h += fpInsight('Best value', bestValue.p.name, bestValue.prob);
    h += '</div>';

    var scarce = [];
    ['QB','RB','WR','TE'].forEach(function(pos){
      var rem = posTopRemaining(pos);
      if (rem > 0 && rem <= 3) scarce.push(pos + ' (' + rem + ' elite left)');
    });
    if (scarce.length) h += '<div class="dr-fp-scarce">&#9888; Scarce now: ' + esc(scarce.join(', ')) + '</div>';

    h += '<div class="dr-fp-list">';
    if (avail.length){ avail.slice(0, 6).forEach(function(s){ h += fpRow(s); }); }
    else h += '<div class="dr-fp-empty">Hard to project this far out.</div>';
    h += '</div>';

    if (gone.length){
      h += '<div class="dr-fp-gone"><span class="dr-fp-gone-lbl">Likely gone:</span> '
        + esc(gone.map(function(s){ return s.p.name; }).join(', ')) + '</div>';
    }
    return h + '</div>';
  }
  function fpInsight(label, val, prob){
    return '<div class="dr-fp-ins"><div class="dr-fp-ins-l">' + esc(label) + '</div>'
      + '<div class="dr-fp-ins-v">' + esc(val) + '</div>'
      + (prob != null ? '<div class="dr-fp-ins-p" style="color:' + availColor(prob) + '">' + prob + '% there</div>' : '')
      + '</div>';
  }
  function fpRow(s){
    var p = s.p;
    return '<div class="dr-fp-row" data-id="' + esc(String(p.id)) + '">'
      + '<span class="dr-fp-prob" style="color:' + availColor(s.prob) + ';background:' + availColor(s.prob) + '1a;">' + s.prob + '%</span>'
      + '<div class="dr-fp-name">' + esc(p.name) + '<span class="dr-fp-pos" style="color:' + posColor(p.position) + '">' + esc(p.position) + '</span></div>'
      + '<span class="dr-fp-vor" title="Value over replacement">' + (s.vor != null ? (s.vor >= 0 ? '+' + s.vor : s.vor) : '') + '</span>'
      + '<span class="dr-fp-val">' + Math.round(s.val) + '</span>'
      + '</div>';
  }

  // ── My Team (roster slots) ──────────────────────────────────────────────────
  function lineupSlots(){
    var rs = (state && state.roster) || defaultRoster();
    var slots = [];
    ['QB','SF','RB','WR','TE','FLEX','K','DEF'].forEach(function(s){
      var n = rs[s] || 0;
      for (var i = 0; i < n; i++) slots.push(s);
    });
    return slots;
  }
  function slotEligible(slot, pos){
    pos = (pos || '').toUpperCase();
    if (slot === 'FLEX') return pos === 'RB' || pos === 'WR' || pos === 'TE';
    if (slot === 'SF')   return pos === 'QB' || pos === 'RB' || pos === 'WR' || pos === 'TE';
    return slot === pos;
  }
  function slotColor(slot){
    if (slot === 'FLEX') return '#14b8a6';
    if (slot === 'SF')   return '#a78bfa';
    if (slot === 'BN')   return '#64748b';
    return posColor(slot);
  }
  function slotRow(slot, p){
    if (p){
      var psBadge = (p.ps != null) ? '<span class="dr-rslot-ps" style="color:' + psColor(p.ps) + '">' + p.ps + '</span>' : '';
      return '<div class="dr-rslot">'
        + '<span class="dr-rslot-pos" style="background:' + slotColor(slot) + '">' + slot + '</span>'
        + '<img class="dr-rslot-hs" src="' + hsUrl(p.id) + '" alt="" onerror="this.style.visibility=\'hidden\'">'
        + '<div class="dr-rslot-body"><div class="dr-rslot-name">' + esc(p.name) + '</div>'
        + '<div class="dr-rslot-meta">' + esc(p.position) + ' &middot; ' + esc(p.team || '') + '</div></div>'
        + psBadge
        + '<div class="dr-rslot-val">' + (p.val != null ? Math.round(p.val) : '') + '</div>'
        + '</div>';
    }
    return '<div class="dr-rslot dr-rslot-open">'
      + '<span class="dr-rslot-pos" style="background:' + slotColor(slot) + '">' + slot + '</span>'
      + '<span class="dr-rslot-empty">open</span></div>';
  }
  // ── Draft grade / roster strength ───────────────────────────────────────────
  function gradeTeam(){
    if (!hasOwned()) return null;
    var mine = [];
    Object.keys(state.picks).forEach(function(k){
      var pn = parseInt(k, 10);
      if (isMyPick(pn)) mine.push({ pn: pn, p: state.picks[k] });
    });
    if (!mine.length) return null;
    var deltas = [], counts = { QB:0, RB:0, WR:0, TE:0 }, tierTop = 0;
    mine.forEach(function(m){
      var pos = (m.p.position || '').toUpperCase();
      if (counts[pos] != null) counts[pos]++;
      var full = playersById[String(m.p.id)];
      if (full){
        var adp = adpOf(full);
        if (adp != null) {
          var gap = m.pn - adp;
          // ±2 neutral zone — matches sim tolerance; only excess past ±2 counts
          deltas.push(gap > 2 ? gap - 2 : gap < -2 ? gap + 2 : 0);
        }
        var t = tierOf(full); if (t && t <= 2) tierTop++;
      }
    });
    var avgDelta = deltas.length ? deltas.reduce(function(a,b){ return a+b; }, 0) / deltas.length : 0;
    var targets = posTargets(), bsum = 0;
    ['QB','RB','WR','TE'].forEach(function(pos){ var t = targets[pos] || 0; bsum += t ? Math.min(counts[pos] || 0, t) / t : 0; });
    var balanceRamp = Math.min(1, mine.length / 8);
    var tierTarget = mine.length <= 1 ? 1 : mine.length <= 5 ? 2 : 3;
    var tierRaw = clamp01(tierTop / tierTarget);

    var valuePts, balancePts, tierPts;
    if (state.type === 'rookie'){
      // Rookie drafts are pure best-player-available: you're adding to an existing
      // roster, so positional balance barely matters. Reward hitting value and
      // landing elite-tier rookies. Value 55 / Tier 30 / Balance 15.
      valuePts   = Math.round(clamp01((avgDelta + 10) / 20) * 55);
      balancePts = Math.round(((1 - balanceRamp) * 0.9 + balanceRamp * (bsum / 4)) * 15);
      tierPts    = Math.round(tierRaw * 30);
    } else {
      // Value: 0 effective delta (at ADP ±2) = 50% of 40; +8 steal = 100%; -8 reach = 0%
      valuePts = Math.round(clamp01((avgDelta + 10) / 20) * 40);
      // Balance: ramp from a neutral floor (0.85) so 1-2 picks don't score near-zero
      balancePts = Math.round(((1 - balanceRamp) * 0.85 + balanceRamp * (bsum / 4)) * 35);
      // Tier: normalize by what's realistically achievable — at most ~3 elite picks
      // in any draft. 1 T1 pick at pick 1 = full tier credit, not 6/25.
      tierPts = Math.round(tierRaw * 25);
    }
    var total = valuePts + balancePts + tierPts;
    return { score: total, value: valuePts, balance: balancePts, tier: tierPts, count: mine.length };
  }
  function gradeLetter(s){
    if (s>=90) return 'A+'; if (s>=85) return 'A'; if (s>=80) return 'A-';
    if (s>=75) return 'B+'; if (s>=70) return 'B'; if (s>=65) return 'B-';
    if (s>=60) return 'C+'; if (s>=55) return 'C'; if (s>=50) return 'C-';
    if (s>=40) return 'D';  return 'F';
  }
  function gradePace(s){ return s>=75 ? 'Ahead of pace' : (s>=48 ? 'On pace' : 'Behind pace'); }
  function gradeBar(label, val, max){
    var pct = max ? Math.round(val / max * 100) : 0;
    return '<div class="dr-gbar-row"><span class="dr-gbar-lbl">' + label + '</span>'
      + '<div class="dr-gbar"><div class="dr-gbar-fill" style="width:' + pct + '%"></div></div></div>';
  }
  // Per-component max points (rookie drafts weight value/tier over balance).
  function gradeMax(){
    return (state.type === 'rookie') ? { value:55, balance:15, tier:30 }
                                     : { value:40, balance:35, tier:25 };
  }
  function gradeBars(g){
    var m = gradeMax();
    return gradeBar('Value', g.value, m.value) + gradeBar('Balance', g.balance, m.balance) + gradeBar('Tiers', g.tier, m.tier);
  }

  function renderNeeds(){
    if (!hasOwned()){ listInto('<div class="dr-empty-note">Set your pick slot to see your team build.</div>'); return; }
    var mine = myPicksList().slice().sort(function(a, b){ return (b.val || 0) - (a.val || 0); });
    var used = {};
    var html = '';
    var g = gradeTeam();
    if (g){
      html += '<div class="dr-grade-card"><div class="dr-grade-letter">' + gradeLetter(g.score) + '</div>'
        + '<div class="dr-grade-meta"><div class="dr-grade-pace">' + gradePace(g.score) + '</div>'
        + gradeBars(g)
        + '</div></div>';
    }
    html += '<div class="dr-roster">';
    lineupSlots().forEach(function(slot){
      var pick = null;
      for (var i = 0; i < mine.length; i++){ if (!used[i] && slotEligible(slot, mine[i].position)){ pick = mine[i]; used[i] = true; break; } }
      html += slotRow(slot, pick);
    });
    var bench = [];
    for (var i = 0; i < mine.length; i++){ if (!used[i]) bench.push(mine[i]); }
    html += '<div class="dr-roster-div">Bench</div>';
    if (bench.length){ bench.forEach(function(p){ html += slotRow('BN', p); }); }
    else { html += slotRow('BN', null); }
    html += '</div>';
    listInto(html);
  }

  // ── Live draft (P5, Sleeper) ────────────────────────────────────────────────
  function valLookup(id){ var p = playersById[String(id)]; return (p && state) ? Math.round(valOf(p)) : null; }
  function applyLivePicks(picks){
    lastLivePicks = picks;
    state.picks = {}; drafted = {};
    picks.forEach(function(p){
      if (p.pick_no == null) return;
      state.picks[p.pick_no] = { id: p.player_id, name: p.name, position: p.position, team: p.team, val: valLookup(p.player_id) };
      if (p.player_id) drafted[String(p.player_id)] = true;
    });
    state.current = (picks.length || 0) + 1;
    _boardSig = null;   // force a full board rebuild on the next render
  }
  function detectLive(){
    if (cfg.isGuest || !cfg.leagueId){
      alert('Live draft sync requires opening the Draft Room from your league.');
      return;
    }
    var box = document.getElementById('drLiveList');
    box.style.display = ''; box.innerHTML = '<div class="dr-live-head">Detecting drafts…</div>';
    fetch('/api/draft/detect?platform=' + encodeURIComponent(cfg.platform) + '&league_id=' + encodeURIComponent(cfg.leagueId) + '&season=' + (cfg.season || ''))
      .then(function(r){ return r.json(); })
      .then(function(resp){
        if (resp.unsupported){ box.innerHTML = '<div class="dr-live-head">Live sync currently supports Sleeper leagues.</div>'; return; }
        var ds = resp.drafts || [];
        if (!ds.length){ box.innerHTML = '<div class="dr-live-head">No drafts found for this league yet.</div>'; return; }
        var html = '<div class="dr-live-head">Detected drafts. Pick one to connect</div>';
        ds.forEach(function(d){
          html += '<button class="dr-live-item" data-id="' + esc(d.draft_id) + '">'
            + '<span class="dr-live-status dr-ls-' + esc(d.status || '') + '">' + esc(d.status || '') + '</span>'
            + esc((d.type || 'snake') + ' · ' + (d.teams || '?') + ' teams · ' + (d.rounds || '?') + ' rounds') + '</button>';
        });
        box.innerHTML = html;
      })
      .catch(function(){ box.innerHTML = '<div class="dr-live-head">Could not detect drafts.</div>'; });
  }
  function connectLive(draftId){
    fetch('/api/draft/live?platform=' + encodeURIComponent(cfg.platform) + '&draft_id=' + encodeURIComponent(draftId))
      .then(function(r){ return r.json(); })
      .then(function(d){
        if (!d || d.error){ alert('Could not load that draft.'); return; }
        var teams = parseInt(d.teams || 0, 10) || (cfg.numTeams || 12);
        var rounds = parseInt(d.rounds || 0, 10) || 15;
        var slot = 0;
        if (cfg.viewerUserId && d.draft_order && d.draft_order[cfg.viewerUserId]) {
          slot = parseInt(d.draft_order[cfg.viewerUserId], 10) || 0;
        }
        state = {
          type: 'startup', teams: teams, rounds: rounds, sf: !!cfg.isSuperflex,
          slot: slot, order: d.order || 'snake', picks: {}, current: 1,
          mode: 'live', sourceDraftId: draftId, slotNames: d.slot_names || {}, queue: []
        };
        applyLivePicks(d.picks || []);
        showMain();
        document.getElementById('drLiveBadge').style.display = '';
        document.getElementById('drUndo').style.display = 'none';
        loadPlayers();
        startPolling();
        if (String(d.status) === 'complete') stopPolling();
      })
      .catch(function(){ alert('Could not connect to the live draft.'); });
  }
  function startPolling(){
    stopPolling();
    pollTimer = setInterval(function(){
      if (!state || state.mode !== 'live') { stopPolling(); return; }
      fetch('/api/draft/live?platform=' + encodeURIComponent(cfg.platform) + '&draft_id=' + encodeURIComponent(state.sourceDraftId))
        .then(function(r){ return r.json(); })
        .then(function(d){
          if (d && d.picks){ applyLivePicks(d.picks); render(); if (String(d.status) === 'complete') stopPolling(); }
        })
        .catch(function(){});
    }, 5000);
  }
  function stopPolling(){ if (pollTimer){ clearInterval(pollTimer); pollTimer = null; } }


  function userNextPick(){
    var total = state.teams * state.rounds;
    for (var pn = state.current; pn <= total; pn++){
      if (isMyPick(pn)) return pn;
    }
    return null;
  }

  function renderStatus(){
    var total = state.teams * state.rounds;
    var done = state.current > total;
    var r = Math.ceil(state.current / state.teams);
    document.getElementById('drRoundPill').textContent = done ? 'Complete' : ('Round ' + r);
    document.getElementById('drPickPill').textContent  = done ? (total + ' picks') : ('Pick ' + state.current);
    var oc = document.getElementById('drOnClock');
    if (done) { oc.textContent = 'Draft complete'; }
    else {
      var slot = slotOnClock(state.current, state.teams, state.order);
      oc.textContent = isMyPick(state.current) ? ('Team ' + slot + ' (You)') : ('Team ' + slot);
    }
    var nextPill = document.getElementById('drNextPill');
    var np = done ? null : userNextPick();
    if (np){ nextPill.style.display = ''; nextPill.textContent = 'Your next: #' + np + ' (R' + Math.ceil(np / state.teams) + ')'; }
    else { nextPill.style.display = 'none'; }
    document.getElementById('drProgress').textContent = Math.min(state.current - 1, total) + ' / ' + total + ' picks';
    var gp = document.getElementById('drGradePill');
    var g = gradeTeam();
    if (g){ gp.style.display = ''; gp.textContent = 'Grade ' + gradeLetter(g.score); } else { gp.style.display = 'none'; }
    document.getElementById('drSummaryBtn').style.display = done ? '' : 'none';
  }

  // ── Board rendering (incremental) ───────────────────────────────────────────
  function teamName(slot){
    if (state.slotNames && state.slotNames[slot]) return state.slotNames[slot];
    return (slot === state.slot) ? 'You' : ('Team ' + slot);
  }
  function cellClass(pn){
    var slot = slotOnClock(pn, state.teams, state.order);
    var pl = state.picks[pn];
    var mine = isMyPick(pn);
    // A claimed pick that is not your home slot is a traded-in pick.
    var traded = mine && state.slot && slot !== state.slot;
    return 'dr-cell' + (pl ? ' dr-cell-filled' : ' dr-cell-empty')
      + (mine ? ' dr-cell-mine' : '') + (traded ? ' dr-cell-claimed' : '')
      + ((canClaim(pn)) ? ' dr-cell-claimable' : '')
      + (pn === justPick ? ' dr-cell-just' : '');
  }
  // Future, uncommitted picks can be claimed/unclaimed in mock/manual mode.
  function canClaim(pn){
    return state.mode !== 'live' && pn >= state.current && !state.picks[pn];
  }
  function cellInner(pn){
    var pl = state.picks[pn];
    var h = '<span class="dr-cell-num">' + pn + '</span>';
    if (isMyPick(pn) && !pl) h += '<span class="dr-cell-mineflag">YOU</span>';
    if (pl){
      if (pl.val != null) h += '<span class="dr-cell-val">' + Math.round(pl.val) + '</span>';
      h += '<img class="dr-hs" src="' + hsUrl(pl.id) + '" alt="" onerror="this.style.visibility=\'hidden\'">';
      h += '<div class="dr-cell-body"><div class="dr-cell-name">' + esc(pl.name) + '</div>'
        + '<div class="dr-cell-meta"><span class="dr-posbadge" style="background:' + posColor(pl.position) + '">' + esc(pl.position) + '</span> ' + esc(pl.team || '') + '</div></div>';
    }
    return h;
  }
  function buildBoard(){
    var board = document.getElementById('drBoard');
    var teams = state.teams, rounds = state.rounds;
    board.style.gridTemplateColumns = '30px repeat(' + teams + ', minmax(108px, 1fr))';
    var html = '<div class="dr-colhead"></div>';
    for (var s = 1; s <= teams; s++){
      var you = (s === state.slot) ? ' dr-colhead-you' : '';
      html += '<div class="dr-colhead' + you + '">' + esc(teamName(s)) + (s === state.slot ? ' ★' : '') + '</div>';
    }
    for (var rnd = 1; rnd <= rounds; rnd++){
      html += '<div class="dr-colhead">R' + rnd + '</div>';
      for (var slot = 1; slot <= teams; slot++){
        var pn = pickNum(rnd, slot, teams, state.order);
        html += '<div class="' + cellClass(pn) + '" id="dc' + pn + '" data-pn="' + pn + '">' + cellInner(pn) + '</div>';
      }
    }
    board.innerHTML = html;
    _boardSig = boardSig();
    refreshCurrent();
  }
  function paintCell(pn){
    var el = document.getElementById('dc' + pn);
    if (!el) return;
    el.className = cellClass(pn);
    el.innerHTML = cellInner(pn);
  }
  function refreshCurrent(){
    var prev = document.querySelector('.dr-cell-current');
    if (prev) prev.classList.remove('dr-cell-current');
    var cur = document.getElementById('dc' + state.current);
    if (cur){
      cur.classList.add('dr-cell-current');
      requestAnimationFrame(function(){
        try { cur.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' }); }
        catch (e) { try { cur.scrollIntoView(); } catch (_) {} }
      });
    }
  }
  function boardSig(){ return [state.teams, state.rounds, state.order, state.slot, state.mode || 'm', ownedPicks().join(',')].join('|'); }
  function renderBoard(){
    // Full rebuild only when the board structure changes; otherwise picks are
    // painted incrementally by commitPick/undo for smooth (Instant) sims.
    if (_boardSig !== boardSig()) buildBoard();
  }

  function renderBA(){
    var srcEl = document.getElementById('drAdpSrc');
    if (srcEl){ srcEl.textContent = 'ADP source: ' + (adpSources[state.type] || 'unavailable'); }
    var sortBy = document.getElementById('drBaSort').value;
    var q = (document.getElementById('drSearch').value || '').trim().toLowerCase();
    var pool = availablePool().filter(function(p){
      if (posFilter !== 'ALL' && String(p.position||'').toUpperCase() !== posFilter) return false;
      if (q && String(p.name||'').toLowerCase().indexOf(q) < 0) return false;
      return true;
    });
    function steal(p){ var a = adpOf(p); return (a != null) ? (state.current - a) : -99999; }
    pool.sort(function(a, b){
      if (sortBy === 'adp'){
        var aa = adpOf(a), ba = adpOf(b);
        return (aa != null ? aa : 99999) - (ba != null ? ba : 99999);
      }
      if (sortBy === 'steals'){ return steal(b) - steal(a); }   // biggest fallers vs ADP first
      return valOf(b) - valOf(a);
    });
    if (!pool.length){ listInto('<div class="dr-empty-note">No players match.</div>'); return; }
    var html = '';
    for (var i = 0; i < Math.min(pool.length, 200); i++){
      var p = pool[i];
      var opts = null;
      if (sortBy === 'steals'){
        var d = steal(p);
        if (d > 0) opts = { sub: '+' + Math.round(d) + ' vs ADP' };
      }
      html += playerRowHtml(p, opts);
    }
    listInto(html);
  }

  // Pick Score for a single player (computes the pool max + your roster counts).
  function pickScoreFor(p){
    var pool = availablePool();
    var maxVal = 0; pool.forEach(function(x){ var v = valOf(x); if (v > maxVal) maxVal = v; });
    return pickScore(p, maxVal, myPosCounts());
  }

  function esc(s){ return String(s == null ? '' : s).replace(/[&<>"]/g, function(c){
    return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]; }); }

  // ── Summary overlay ─────────────────────────────────────────────────────────
  function openSummary(){
    if (!state || !state.slot) return;
    var g = gradeTeam();
    var mine = myPicksList().slice().sort(function(a, b){ return (b.val || 0) - (a.val || 0); });
    var used = {};
    var slots = lineupSlots();
    var starters = [];
    slots.forEach(function(slot){
      var pick = null;
      for (var i = 0; i < mine.length; i++){ if (!used[i] && slotEligible(slot, mine[i].position)){ pick = mine[i]; used[i] = true; break; } }
      starters.push({ slot: slot, p: pick });
    });
    var bench = [];
    for (var i = 0; i < mine.length; i++){ if (!used[i]) bench.push(mine[i]); }

    function sumRow(slot, p){
      if (!p) return '<div class="dr-sum-row"><span class="dr-sum-slot-badge" style="background:' + slotColor(slot) + '">' + slot + '</span>'
        + '<span class="dr-sum-empty">open</span></div>';
      var roundPicked = Object.keys(state.picks).filter(function(k){ return state.picks[k] && state.picks[k].id === p.id; }).map(function(k){ return Math.ceil(parseInt(k,10)/state.teams); })[0] || '';
      var psStr = (p.ps != null) ? '<span class="dr-sum-ps" style="color:' + psColor(p.ps) + '">' + p.ps + '</span>' : '';
      return '<div class="dr-sum-row">'
        + '<span class="dr-sum-slot-badge" style="background:' + slotColor(slot) + '">' + slot + '</span>'
        + '<img class="dr-sum-hs" src="' + hsUrl(p.id) + '" alt="" onerror="this.style.visibility=\'hidden\'">'
        + '<div class="dr-sum-body"><div class="dr-sum-name">' + esc(p.name) + '</div>'
        + '<div class="dr-sum-meta">' + esc(p.position) + (p.team ? ' \xb7 ' + esc(p.team) : '') + (roundPicked ? ' \xb7 Rd ' + roundPicked : '') + '</div>'
        + (p.reason ? '<div class="dr-sum-reason">' + esc(p.reason) + '</div>' : '')
        + '</div>' + psStr + '</div>';
    }

    var gradeHtml = g
      ? '<div class="dr-sum-grade">' + gradeLetter(g.score) + '</div><div class="dr-sum-pace">' + gradePace(g.score) + '</div>'
        + gradeBars(g)
      : '';

    var html = '<button class="dr-prev-close" id="drSumClose" aria-label="Close">&times;</button>'
      + '<div class="dr-sum-header"><div class="dr-sum-title">Draft Complete</div>' + gradeHtml + '</div>'
      + '<div class="dr-sum-section">Starters</div>';
    starters.forEach(function(s){ html += sumRow(s.slot, s.p); });
    html += '<div class="dr-sum-section">Bench</div>';
    if (bench.length){ bench.forEach(function(p){ html += sumRow('BN', p); }); }
    else { html += sumRow('BN', null); }
    html += '<div class="dr-sum-footer">'
      + '<button class="dr-btn dr-btn-primary" id="drSumShare">Share</button>'
      + '<button class="dr-btn" id="drSumCloseBtn">Close</button>'
      + '</div>';

    var card = document.getElementById('drSummaryCard');
    card.innerHTML = html;
    document.getElementById('drSummary').style.display = '';
    document.getElementById('drSumClose').addEventListener('click', closeSummary);
    document.getElementById('drSumCloseBtn').addEventListener('click', closeSummary);
    document.getElementById('drSumShare').addEventListener('click', function(){ closeSummary(); shareDraft(); });
  }
  function closeSummary(){ document.getElementById('drSummary').style.display = 'none'; }

  // ── Actions ──────────────────────────────────────────────────────────────
  function commitPick(p){
    var pn = state.current;
    var ps = pickScoreFor(p);
    var reason = pickReason(p, myPosCounts());
    state.picks[pn] = { id: p.id, name: p.name, position: p.position, team: p.team, val: Math.round(valOf(p)), ps: ps, reason: reason };
    drafted[String(p.id)] = true;
    justPick = pn;
    state.current++;
    paintCell(pn);        // fill just-picked cell (incremental)
    refreshCurrent();     // move the on-the-clock highlight + auto-scroll
  }
  function draftPlayer(id){
    if (state.mode === 'live') return;   // live board is driven by the platform
    var total = state.teams * state.rounds;
    if (state.current > total) return;
    if (sim && !isMyPick(state.current)) return; // not your turn
    var p = playersById[String(id)];
    if (!p || drafted[String(id)]) return;
    commitPick(p);
    render();
    if (sim) scheduleSim();   // resume CPU picks after your selection
  }
  function undo(){
    if (state.current <= 1 || state.mode === 'live') return;
    state.current--;
    var p = state.picks[state.current];
    if (p) { delete drafted[String(p.id)]; delete state.picks[state.current]; }
    paintCell(state.current);
    refreshCurrent();
    render();
  }
  function resetDraft(){
    if (!confirm('Reset the draft board?')) return;
    try { sessionStorage.removeItem(sessKey); } catch(e){}
    state = null;
    showSetup();
  }

  // ── Share a draft image ─────────────────────────────────────────────────────
  function shareDraft(){
    if (!state || !hasOwned()){ alert('Pick a draft slot to build and share your team.'); return; }
    var mine = myPicksList().slice().sort(function(a, b){ return (b.val || 0) - (a.val || 0); });
    var used = {}, rows = [];
    lineupSlots().forEach(function(slot){
      var pick = null;
      for (var i = 0; i < mine.length; i++){ if (!used[i] && slotEligible(slot, mine[i].position)){ pick = mine[i]; used[i] = true; break; } }
      rows.push({ slot: slot, p: pick });
    });
    for (var i = 0; i < mine.length; i++){ if (!used[i]) rows.push({ slot: 'BN', p: mine[i] }); }

    var W = 720, pad = 30, lineH = 40;
    var H = pad * 2 + 118 + rows.length * lineH;
    var c = document.createElement('canvas'); c.width = W; c.height = H;
    var ctx = c.getContext('2d');
    ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#38bdf8'; ctx.font = 'bold 30px system-ui,Arial';
    ctx.fillText('My ' + (state.type.charAt(0).toUpperCase() + state.type.slice(1)) + ' Draft', pad, pad + 30);
    ctx.fillStyle = '#94a3b8'; ctx.font = '14px system-ui,Arial';
    ctx.fillText((state.sf ? 'Superflex' : '1QB') + ' · ' + state.teams + ' teams · BR Fantasy', pad, pad + 54);
    var g = gradeTeam();
    if (g){ ctx.fillStyle = '#22c55e'; ctx.font = 'bold 16px system-ui,Arial';
      ctx.fillText('Grade ' + gradeLetter(g.score) + ' · ' + gradePace(g.score), pad, pad + 84); }
    var y = pad + 118;
    var POSC = { QB:'#f59e0b', RB:'#22c55e', WR:'#3b82f6', TE:'#8b5cf6', K:'#94a3b8', DEF:'#64748b', FLEX:'#14b8a6', SF:'#a78bfa', BN:'#64748b' };
    rows.forEach(function(r){
      ctx.fillStyle = POSC[r.slot] || '#94a3b8'; ctx.font = 'bold 12px system-ui,Arial';
      ctx.fillText(r.slot, pad, y + 14);
      ctx.fillStyle = r.p ? '#e5e7eb' : '#64748b'; ctx.font = r.p ? 'bold 16px system-ui,Arial' : 'italic 15px system-ui,Arial';
      ctx.fillText(r.p ? r.p.name : 'open', pad + 64, y + 14);
      if (r.p){ ctx.fillStyle = '#94a3b8'; ctx.font = '13px system-ui,Arial';
        ctx.textAlign = 'right'; ctx.fillText((r.p.position || '') + '  ' + (r.p.team || ''), W - pad, y + 14); ctx.textAlign = 'left'; }
      y += lineH;
    });
    c.toBlob(function(blob){
      if (!blob) return;
      try {
        var file = new File([blob], 'br-draft.png', { type: 'image/png' });
        if (navigator.share && navigator.canShare && navigator.canShare({ files: [file] })){
          navigator.share({ files: [file], title: 'My BR Fantasy Draft' }).catch(function(){});
          return;
        }
      } catch (e) {}
      var a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'br-draft.png'; a.click();
      setTimeout(function(){ URL.revokeObjectURL(a.href); }, 1000);
    }, 'image/png');
  }

  // ── Player preview / draft confirm ──────────────────────────────────────────
  function statBox(label, val){
    return '<div class="dr-prev-stat"><div class="dr-prev-stat-v">' + val + '</div><div class="dr-prev-stat-l">' + label + '</div></div>';
  }
  function isYourTurn(){
    if (state.mode === 'live') return false;
    if (state.current > state.teams * state.rounds) return false;
    if (sim && !isMyPick(state.current)) return false;
    return true;
  }
  function psColor(ps){ return ps >= 90 ? '#22c55e' : ps >= 75 ? '#38bdf8' : ps >= 60 ? '#f59e0b' : '#ef4444'; }
  function openPreview(id){
    var p = playersById[String(id)]; if (!p) return;
    var adp = adpOf(p), t = tierOf(p), ps = pickScoreFor(p);
    var posRank = state.sf ? (p.sf_pos_rank_label || '') : (p.pos_rank_label || '');
    var adpGap = (adp != null) ? (state.current - adp) : null;
    var vsAdp = adpGap != null ? (adpGap >= 0 ? ('+' + Math.round(adpGap)) : String(Math.round(adpGap))) : '–';
    var sc = psColor(ps);
    var c = document.getElementById('drPreviewCard');
    var h = '<button class="dr-prev-close" id="drPrevClose" aria-label="Close">&times;</button>'
      // Player identity row
      + '<div class="dr-prev-top">'
      + '<img class="dr-prev-hs" src="' + hsUrl(p.id) + '" alt="" onerror="this.style.visibility=\'hidden\'">'
      + '<div class="dr-prev-id"><div class="dr-prev-name">' + esc(p.name) + (t ? (' <span class="dr-tier' + (isTierCliff(p) ? ' dr-tier-cliff' : '') + '">T' + t + '</span>') : '') + '</div>'
      + '<div class="dr-prev-meta"><span class="dr-posbadge" style="background:' + posColor(p.position) + '">' + esc(p.position) + '</span> ' + esc(p.team || '') + (posRank ? (' &middot; ' + esc(posRank)) : '') + '</div>'
      + '</div></div>'
      // Pick Score hero
      + '<div class="dr-prev-score-hero" style="border-color:' + sc + ';background:' + sc + '1a;">'
      + '<div class="dr-prev-score-num" style="color:' + sc + '">' + ps + '</div>'
      + '<div class="dr-prev-score-lbl">Pick Score</div>'
      + '<div class="dr-prev-score-reason">' + esc(pickReason(p, myPosCounts())) + '</div>'
      + '</div>'
      // Stats row
      + '<div class="dr-prev-stats">'
      + statBox('Value', Math.round(valOf(p)))
      + statBox('ADP', adp != null ? Number(adp).toFixed(1) : '–')
      + statBox('vs ADP', vsAdp)
      + (p.age != null ? statBox('Age', Number(p.age).toFixed(1)) : statBox('Pos Rank', posRank || '–'))
      + '</div>';
    if (state.mode === 'live'){
      h += '<div class="dr-prev-note">Live draft. Picks come from the platform.</div>';
    } else if (isYourTurn()){
      h += '<button class="dr-btn dr-btn-primary dr-btn-lg dr-prev-draft" data-id="' + esc(String(p.id)) + '">Draft ' + esc(p.name) + '</button>';
    } else if (sim){
      h += '<div class="dr-prev-note">Not your pick yet. A CPU team is on the clock.</div>';
    } else {
      h += '<button class="dr-btn dr-btn-primary dr-btn-lg dr-prev-draft" data-id="' + esc(String(p.id)) + '">Draft ' + esc(p.name) + '</button>';
    }
    c.innerHTML = h;
    document.getElementById('drPreview').style.display = '';
  }
  function closePreview(){ document.getElementById('drPreview').style.display = 'none'; }

  // ── Wire up ──────────────────────────────────────────────────────────────
  document.getElementById('drStart').addEventListener('click', startDraft);
  document.getElementById('drStartSim').addEventListener('click', startMock);
  document.getElementById('drSimToggle').addEventListener('click', toggleSim);
  document.getElementById('drSimSpeed').addEventListener('change', function(){
    simSpeed = parseInt(this.value, 10) || 700;
  });
  document.getElementById('drSideTabs').addEventListener('click', function(e){
    var b = e.target.closest('.otc-main-tab'); if (!b) return;
    sideTab = b.getAttribute('data-stab');
    this.querySelectorAll('.otc-main-tab').forEach(function(x){ x.classList.toggle('is-active', x === b); });
    renderSide();
  });
  document.getElementById('drBoard').addEventListener('click', function(e){
    var cell = e.target.closest('[data-pn]'); if (!cell) return;
    var pn = parseInt(cell.getAttribute('data-pn'), 10);
    if (!canClaim(pn)) return;          // only future, uncommitted picks (mock/manual)
    toggleOwned(pn);
    _boardSig = null;                   // ownership changed: force board rebuild
    render();
  });
  document.getElementById('drConnect').addEventListener('click', detectLive);
  document.getElementById('drLiveList').addEventListener('click', function(e){
    var b = e.target.closest('.dr-live-item'); if (b) connectLive(b.getAttribute('data-id'));
  });
  document.getElementById('drUndo').addEventListener('click', undo);
  document.getElementById('drReset').addEventListener('click', resetDraft);
  document.getElementById('drEdit').addEventListener('click', showSetup);
  document.getElementById('drBaSort').addEventListener('change', renderBA);
  document.getElementById('drSearch').addEventListener('input', renderBA);
  document.getElementById('drBaList').addEventListener('click', function(e){
    var star = e.target.closest('[data-star]');
    if (star){ e.stopPropagation(); toggleQueue(star.getAttribute('data-star')); return; }
    var draft = e.target.closest('[data-draft]');
    if (draft){ e.stopPropagation(); draftPlayer(draft.getAttribute('data-draft')); return; }
    var row = e.target.closest('.dr-ba-row') || e.target.closest('.dr-fp-row');
    if (row) openPreview(row.getAttribute('data-id'));
  });
  document.getElementById('drSummaryBtn').addEventListener('click', openSummary);
  document.getElementById('drSummary').addEventListener('click', function(e){
    if (e.target === this) closeSummary();
  });
  document.getElementById('drShare').addEventListener('click', shareDraft);
  document.getElementById('drPreview').addEventListener('click', function(e){
    if (e.target === this || e.target.closest('#drPrevClose')){ closePreview(); return; }
    var d = e.target.closest('.dr-prev-draft');
    if (d){ var id = d.getAttribute('data-id'); closePreview(); draftPlayer(id); }
  });
  document.getElementById('drPosFilters').addEventListener('click', function(e){
    var b = e.target.closest('.dr-pos'); if (!b) return;
    posFilter = b.getAttribute('data-pos');
    this.querySelectorAll('.dr-pos').forEach(function(x){ x.classList.toggle('active', x === b); });
    renderBA();
  });

  applyCfgDefaults();
  renderSetupRoster();
  document.getElementById('drSf').addEventListener('change', function(){ _setupRoster = null; renderSetupRoster(); });
  document.getElementById('drType').addEventListener('change', function(){ _setupRoster = null; renderSetupRoster(); });
  document.getElementById('drRosterSection').addEventListener('click', function(e){
    var step = e.target.closest('.dr-step-btn');
    if (!step) return;
    e.stopPropagation();
    var key = step.getAttribute('data-key');
    var d = parseInt(step.getAttribute('data-d'), 10);
    if (!_setupRoster) _setupRoster = defaultRoster();
    _setupRoster[key] = Math.max(0, (_setupRoster[key] || 0) + d);
    renderSetupRoster();
  });

  function resumeFromSession(){
    var saved = load();
    if (saved && saved.teams && saved.picks){
      state = saved;
      if (!state.roster) state.roster = defaultRoster();
      if (!state.owned) state.owned = defaultOwned();
      if (state.mode === 'live'){
        document.getElementById('drLiveBadge').style.display = '';
        document.getElementById('drUndo').style.display = 'none';
      }
      showMain();
      loadPlayers();
      if (state.mode === 'live' && state.sourceDraftId) startPolling();
    }
  }

  // Open a specific league draft from history (?live=<draft_id>), else resume
  // the in-progress session draft.
  var urlLive = new URLSearchParams(location.search).get('live');
  if (urlLive){
    connectLive(urlLive);
  } else {
    resumeFromSession();
  }
})();
</script>
"""


def build_draft_history_body(
    league_id: Optional[str],
    season: Optional[int],
    platform: Optional[str] = None,
) -> str:
    """Draft History page: the league's real drafts (from Sleeper), openable by
    any league member to review the board."""
    has_league = bool(league_id and platform and season)
    base = f"/{platform}/{int(season)}/{league_id}/draft" if has_league else "/draft"
    cfg = {
        "base": base,
        "leagueId": league_id or "",
        "platform": platform or "sleeper",
        "season": int(season) if season else None,
        "hasLeague": has_league,
    }
    cfg_json = json.dumps(cfg)
    return (
        f'<script>window.__draftHistCfg = {cfg_json};</script>\n'
        + _DRAFT_HISTORY_HTML
    )


_DRAFT_HISTORY_HTML = r"""
<div class="dr-wrap">
  <div class="dr-hero">
    <h1 class="dr-title">Draft History</h1>
    <p class="dr-sub">Your league's drafts. Open any board to review the picks pick-by-pick.</p>
  </div>
  <div id="drHistList" class="dr-hist-list">
    <div class="dr-loading"><div class="loading-spinner" style="width:22px;height:22px;"></div><span>Loading…</span></div>
  </div>
</div>

<style>
  .dr-wrap { max-width: 900px; margin: 0 auto; padding: 12px 14px 48px; }
  .dr-hero { margin-bottom: 14px; }
  .dr-title { font-size: clamp(20px,4vw,28px); font-weight: 800; color: var(--text); margin: 0 0 4px; }
  .dr-sub { font-size: 14px; color: var(--text-muted); margin: 0; }
  .dr-hist-list { display: flex; flex-direction: column; gap: 10px; }
  .dr-hist-card { display: flex; align-items: center; gap: 12px; padding: 14px 16px; border: 1px solid var(--border);
    border-radius: 10px; background: var(--card); }
  .dr-hist-body { flex: 1; min-width: 0; }
  .dr-hist-title { font-size: 15px; font-weight: 700; color: var(--text); }
  .dr-hist-meta { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
  .dr-hist-tag { font-size: 10px; font-weight: 800; text-transform: uppercase; padding: 1px 7px; border-radius: 999px;
    background: rgba(56,189,248,.14); color: var(--accent,#38bdf8); margin-right: 6px; }
  .dr-hist-tag-live { background: rgba(239,68,68,.16); color: #ef4444; }
  .dr-hist-tag-complete { background: rgba(148,163,184,.16); color: #94a3b8; }
  .dr-hist-actions { display: flex; gap: 6px; flex-shrink: 0; }
  .dr-btn { padding: 8px 14px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer;
    border: 1px solid var(--border); background: var(--bg); color: var(--text); text-decoration: none; }
  .dr-btn-primary { background: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8); color: #fff; }
  .dr-btn-danger { color: #ef4444; border-color: rgba(239,68,68,.4); background: transparent; }
  .dr-loading { display: flex; align-items: center; gap: 10px; padding: 24px; color: var(--text-muted); font-size: 13px; justify-content: center; }
  .dr-hist-empty { padding: 28px; text-align: center; color: var(--text-muted); font-size: 14px; }
</style>

<script>
(function(){
  var cfg = window.__draftHistCfg || { base: '/draft', hasLeague: false };
  var listEl = document.getElementById('drHistList');

  function esc(s){ return String(s == null ? '' : s).replace(/[&<>"]/g, function(c){
    return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]; }); }

  function statusTag(s){
    var c = (s === 'drafting') ? 'dr-hist-tag-live' : (s === 'complete' ? 'dr-hist-tag-complete' : '');
    var label = (s === 'drafting') ? 'Live now' : (s === 'pre_draft' ? 'Upcoming' : (s === 'complete' ? 'Complete' : (s || '')));
    return '<span class="dr-hist-tag ' + c + '">' + esc(label) + '</span>';
  }

  function render(drafts){
    if (!drafts.length){
      listEl.innerHTML = '<div class="dr-hist-empty">No drafts found for this league yet.</div>';
      return;
    }
    // Live/upcoming first, then completed.
    var rank = { drafting: 0, pre_draft: 1, complete: 2 };
    drafts.sort(function(a, b){ return (rank[a.status] != null ? rank[a.status] : 3) - (rank[b.status] != null ? rank[b.status] : 3); });
    var html = '';
    drafts.forEach(function(d){
      var title = (d.type ? (d.type.charAt(0).toUpperCase() + d.type.slice(1)) : 'Draft')
        + ' · ' + (d.teams || '?') + ' teams · ' + (d.rounds || '?') + ' rounds';
      html += '<div class="dr-hist-card">'
        + '<div class="dr-hist-body"><div class="dr-hist-title">' + statusTag(d.status) + esc(title) + '</div>'
        + '<div class="dr-hist-meta">' + esc((d.order || 'snake')) + ' order' + (d.season ? (' · ' + esc(String(d.season))) : '') + '</div></div>'
        + '<div class="dr-hist-actions">'
        + '<a class="dr-btn dr-btn-primary" href="' + esc(cfg.base) + '?live=' + encodeURIComponent(d.draft_id) + '">Open board</a>'
        + '</div></div>';
    });
    listEl.innerHTML = html;
  }

  function loadList(){
    if (!cfg.hasLeague){
      listEl.innerHTML = '<div class="dr-hist-empty">Open Draft History from your league to see its drafts. '
        + 'You can still run a mock in the <a href="' + esc(cfg.base) + '">Draft Room</a>.</div>';
      return;
    }
    fetch('/api/draft/detect?platform=' + encodeURIComponent(cfg.platform)
        + '&league_id=' + encodeURIComponent(cfg.leagueId) + '&season=' + (cfg.season || ''), { cache: 'no-store' })
      .then(function(r){ return r.json(); })
      .then(function(resp){
        if (resp.unsupported){ listEl.innerHTML = '<div class="dr-hist-empty">Draft history is available for Sleeper leagues.</div>'; return; }
        render(resp.drafts || []);
      })
      .catch(function(){ listEl.innerHTML = '<div class="dr-hist-empty">Could not load drafts.</div>'; });
  }

  loadList();
})();
</script>
"""
