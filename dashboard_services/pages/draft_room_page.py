"""
Standalone Draft Room (Draft Assistant) page.

Phase 2: a dedicated, self-contained draft board that supersedes the old
Prospects→Draft tab. Supports manual drafting for both startup (all players)
and rookie drafts, with snake / linear / third-round-reversal pick order.
Live Sleeper sync, persistence/history, and the full command-center panels
land in later phases; this establishes the standalone page + board grid +
best-available picker + the pickOrder foundation.

The page is self-contained: its CSS is inlined here and its JS lives in
static/draft_room.js (loaded as a deferred external script so the browser caches
it across visits instead of re-receiving ~210KB inline on every load). Server
values are passed via a small window.__draftCfg JSON blob the script reads on
start; the JS file needs no f-string brace escaping.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

# Cache-busting hash for the external Draft Room script (mirrors app.js's ?v=).
# The 4k-line draft IIFE lives in static/draft_room.js so the browser caches it
# across visits instead of re-receiving it inline on every Draft Room load.
_DRAFT_ROOM_JS_V: Optional[str] = None


def _draft_room_js_v() -> str:
    global _DRAFT_ROOM_JS_V
    if _DRAFT_ROOM_JS_V is None:
        try:
            _p = Path(__file__).resolve().parents[2] / "static" / "draft_room.js"
            _DRAFT_ROOM_JS_V = hashlib.md5(_p.read_bytes()).hexdigest()[:8]
        except OSError:
            _DRAFT_ROOM_JS_V = "0"
    return _DRAFT_ROOM_JS_V


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
    num_rounds_rookie: Optional[int] = None,
    num_rounds_startup: Optional[int] = None,
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
        "numRoundsRookie":  int(num_rounds_rookie)  if num_rounds_rookie  else None,
        "numRoundsStartup": int(num_rounds_startup) if num_rounds_startup else None,
    }
    cfg_json = json.dumps(cfg)
    # cfg is a plain inline script so it runs during parse, before the deferred
    # external draft_room.js reads window.__draftCfg. The page is a full document
    # (render_page), so a deferred external script executes normally.
    return (
        f'<script>window.__draftCfg = {cfg_json};</script>\n'
        + _DRAFT_ROOM_HTML
        + f'\n<script src="/static/draft_room.js?v={_draft_room_js_v()}" defer></script>\n'
    )


# Plain (non-f) string — safe to contain { } freely.
_DRAFT_ROOM_HTML = r"""
<div class="dr-wrap">
  <div class="dr-hero" id="drHero">
    <h1 class="dr-title">Draft Room</h1>
    <p class="dr-sub">Mock against CPU teams, draft manually, or sync a live Sleeper draft. Best-available, tiers, and a live draft grade.</p>
  </div>

  <!-- Setup -->
  <div class="dr-setup" id="drSetup">
    <div class="dr-setup-card">

      <div class="dr-step">
        <div class="dr-step-num">Step 1</div>
        <div class="dr-step-title">Format</div>
        <div class="dr-setup-grid">
          <div class="dr-field"><span>Draft Type</span>
            <select id="drType">
              <option value="startup">Startup (Dynasty)</option>
              <option value="rookie">Rookie (Dynasty)</option>
              <option value="redraft">Redraft</option>
            </select>
          </div>
          <div class="dr-field"><span>QB Format</span>
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
          <div class="dr-field"><span>PPR</span>
            <select id="drPpr">
              <option value="1" selected>Full PPR</option>
              <option value="0.5">Half PPR</option>
              <option value="0">Standard</option>
            </select>
          </div>
          <div class="dr-field"><span>TE Premium</span>
            <select id="drTep">
              <option value="0" selected>None</option>
              <option value="0.5">+0.5 PPR</option>
              <option value="1">+1.0 PPR</option>
            </select>
          </div>
        </div>
      </div>

      <div class="dr-step">
        <div class="dr-step-num">Step 2</div>
        <div class="dr-step-title">Roster Slots</div>
        <div id="drRosterSection"></div>
      </div>

      <div class="dr-step">
        <div class="dr-step-num">Step 3</div>
        <div class="dr-step-title">League</div>
        <div class="dr-setup-grid">
          <div class="dr-field"><span>Teams</span>
            <select id="drTeams">
              <option>8</option><option>10</option><option selected>12</option><option>14</option>
            </select>
          </div>
          <div class="dr-field" id="drRoundsField" style="display:none;"><span>Rounds</span>
            <input id="drRounds" type="number" min="1" max="40" value="3">
          </div>
          <div class="dr-field"><span>Your Pick</span>
            <select id="drSlot"></select>
          </div>
        </div>
      </div>

      <div class="dr-step">
        <div class="dr-step-num">Step 4</div>
        <div class="dr-step-title">Draft Capital</div>
        <p class="dr-setup-desc" style="margin-bottom:8px;">Defaults to your slot's picks. Tap + on a round to add a traded-in pick, or click a pick to remove one you traded away.</p>
        <div id="drCapitalSection"></div>
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
    <div class="dr-start-banner" id="drStartBanner" style="display:none;"></div>
    <div class="dr-statusbar">
      <div class="dr-status-info">
        <div class="dr-onclock" id="drOnClockWrap">
          <span class="dr-onclock-label">On the clock</span>
          <b id="drOnClock">Team 1</b>
        </div>
        <div class="dr-status-pills">
          <span class="dr-ss-stat" id="drPickPill">Pick: 1.01</span>
          <span class="dr-pick-timer" id="drPickTimer" style="display:none;"></span>
          <span class="dr-pill dr-pill-live" id="drLiveBadge" style="display:none;">&#9679; LIVE</span>
          <span class="dr-pill dr-pill-upcoming" id="drUpcomingBadge" style="display:none;">Upcoming</span>
          <span class="dr-progress" id="drProgress"></span>
          <span class="dr-save" id="drSave"></span>
          <span class="dr-poll-status" id="drPollStatus" style="display:none;"></span>
        </div>
      </div>
      <div class="dr-status-right">
        <span class="dr-pill dr-pill-you" id="drNextPill" style="display:none;"></span>
        <span class="dr-pill dr-pill-grade" id="drGradePill" style="display:none;cursor:pointer;" title="View your draft report card"></span>
        <span class="dr-sr-gap"></span>
        <button class="dr-btn dr-btn-primary" id="drSimStart" style="display:none;">&#9654;&nbsp; Start Draft</button>
        <button class="dr-btn dr-btn-ghost" id="drSimToggle" style="display:none;">Pause</button>
        <button class="dr-btn dr-btn-ghost" id="drAutoBtn" style="display:none;" title="Auto-draft best available on your picks">Auto Draft</button>
        <button class="dr-btn dr-btn-ghost" id="drPractice" style="display:none;">Practice Mock</button>
      </div>
    </div>

    <div class="dr-cols">
      <div class="dr-board-wrap">
        <div class="dr-board-toolbar">
          <div class="dr-cell-toggle" id="drCellToggle" title="Toggle between dynasty value and pick score">
            <span class="dr-ct-opt is-active" data-mode="val">Value</span>
            <span class="dr-ct-opt" data-mode="ps">Pick Score</span>
          </div>
        </div>
        <div class="dr-board-scroll"><div class="dr-board" id="drBoard"></div></div>
      </div>
      <aside class="dr-side" id="drSide">
        <button class="dr-sheet-handle" id="drSheetHandle" aria-label="Resize panel"><span class="dr-sheet-grip"></span></button>
        <div class="otc-main-tabs dr-side-tabs" id="drSideTabs">
          <button class="otc-main-tab is-active" data-stab="best">Players</button>
          <button class="otc-main-tab" data-stab="rec">Recs</button>
          <button class="otc-main-tab" data-stab="queue">Queue</button>
          <button class="otc-main-tab" data-stab="needs">Team</button>
          <button class="otc-main-tab" data-stab="league">League</button>
          <div class="dr-side-opts">
            <button class="dr-opts-trigger" id="drPickTradeBtn" aria-label="Pick trade evaluator" title="Pick trade evaluator"><i class="fa-solid fa-right-left"></i></button>
            <button class="dr-opts-trigger" id="drOptsBtn" aria-label="Settings" title="Settings"><i class="fa-solid fa-gear"></i></button>
            <div class="dr-opts-panel" id="drOptsPanel">
              <select class="dr-sim-speed" id="drSimSpeed" style="display:none;" title="Simulation speed">
                <option value="1400">Slow</option>
                <option value="700" selected>Normal</option>
                <option value="300">Fast</option>
                <option value="60">Instant</option>
              </select>
              <button class="dr-btn dr-btn-ghost" id="drSummaryBtn" style="display:none;">Summary</button>
              <button class="dr-btn dr-btn-ghost" id="drShare">Share</button>
              <button class="dr-btn dr-btn-ghost" id="drUndo">Undo</button>
              <button class="dr-btn dr-btn-ghost" id="drEdit">Edit Setup</button>
              <button class="dr-btn dr-btn-ghost dr-btn-danger" id="drReset">Reset</button>
            </div>
          </div>
        </div>
        <div class="dr-side-head" id="drBestControls">
          <div class="dr-side-controls">
            <select id="drBaSort">
              <option value="value">Value</option>
              <option value="adp" selected>ADP</option>
              <option value="steals">Steals</option>
              <option value="ps">Pick Score</option>
            </select>
            <input id="drSearch" type="search" placeholder="Search…" autocomplete="off">
            <button class="dr-help-btn" id="drHelpBtn" type="button" aria-label="What do these terms mean?" title="What do these terms mean?">?</button>
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
        <div id="drBestChips" style="display:none;"></div>
        <div class="dr-ba-list" id="drBaList">
          <div class="dr-loading"><div class="loading-spinner" style="width:22px;height:22px;"></div><span>Loading players…</span></div>
        </div>
        <div id="drCompleteBar" style="display:none;">
          <button class="dr-btn dr-btn-primary" id="drCompleteSummaryBtn" style="width:100%;">Draft Summary</button>
          <button class="dr-btn" id="drCompleteShareBtn" style="width:100%;">Share</button>
        </div>
      </aside>
    </div>
  </div>

  <!-- Player preview / draft confirm -->
  <div class="dr-preview-overlay" id="drPreview" style="display:none;">
    <div class="dr-preview-card" id="drPreviewCard"></div>
  </div>

  <!-- Player comparison -->
  <div class="dr-cmp-overlay" id="drCompare" style="display:none;">
    <div class="dr-cmp-card" id="drCompareCard"></div>
  </div>
  <!-- Team needs tooltip (board cell hover) -->
  <div id="drTeamTip" style="display:none;position:fixed;z-index:300;pointer-events:none;"></div>

  <!-- End-of-draft summary -->
  <div class="dr-summary-overlay" id="drSummary" style="display:none;">
    <div class="dr-summary-card" id="drSummaryCard"></div>
  </div>

  <!-- Glossary / term explainer -->
  <div class="dr-gloss-overlay" id="drGloss" style="display:none;">
    <div class="dr-gloss-card">
      <button class="dr-gloss-close" id="drGlossClose" aria-label="Close">&times;</button>
      <div class="dr-gloss-title">What the numbers mean</div>
      <div id="drGlossBody"></div>
    </div>
  </div>

  <!-- Share preview -->
  <div class="dr-shareview-overlay" id="drShareView" style="display:none;">
    <div class="dr-shareview-card">
      <button class="dr-prev-close" id="drShareViewClose" aria-label="Close">&times;</button>
      <div class="dr-shareview-tabs" id="drShareViewTabs">
        <button class="dr-shareview-tab is-active" data-sv="dark">Dark</button>
        <button class="dr-shareview-tab" data-sv="light">Light</button>
      </div>
      <img class="dr-shareview-img" id="drShareViewImg" alt="Draft preview">
      <div class="dr-shareview-footer">
        <button class="dr-btn dr-btn-primary" id="drShareViewShare">Share</button>
        <button class="dr-btn" id="drShareViewDl">Download</button>
      </div>
    </div>
  </div>

  <!-- Custom modal (replaces browser confirm/alert) -->
  <div id="drModal" style="display:none;position:fixed;inset:0;z-index:9999;background:rgba(0,0,0,.52);align-items:center;justify-content:center;padding:20px;">
    <div class="dr-modal-box">
      <div class="dr-modal-msg" id="drModalMsg"></div>
      <div class="dr-modal-btns" id="drModalBtns"></div>
    </div>
  </div>
</div>

<style>
  .dr-wrap { max-width: 1640px; margin: 0 auto; padding: 10px 12px 40px; }
  .dr-hero { margin: 4px 0 16px; text-align: center; }
  .dr-title { font-size: clamp(22px,4vw,30px); font-weight: 800; color: var(--text); margin: 0 0 6px; }
  .dr-sub { font-size: 14px; color: var(--text-muted); margin: 0 auto; max-width: 560px; line-height: 1.5; }
  /* ── Setup (redesigned) ── */
  .dr-setup { display: flex; justify-content: center; padding: 0 0 8px; }
  .dr-setup-card { width: 100%; max-width: 720px; background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 22px 24px; box-shadow: 0 8px 30px rgba(0,0,0,.10); }
  .dr-setup-desc { font-size: 13px; color: var(--text-muted); margin: 0; line-height: 1.5; }
  .dr-step { padding: 22px 0; border-top: 1px solid var(--border); }
  .dr-step:first-child { border-top: none; padding-top: 0; }
  .dr-step-num { font-size: 10px; font-weight: 900; text-transform: uppercase; letter-spacing: .12em; color: var(--accent,#38bdf8); margin-bottom: 4px; }
  .dr-step-title { font-size: 22px; font-weight: 900; color: var(--text); margin-bottom: 16px; line-height: 1.1; }
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
  /* Settings gear button — sits beside the side-panel tabs — + dropdown panel */
  .dr-side-opts { position: relative; flex: 0 0 auto; display: flex; align-items: stretch; }
  .dr-opts-trigger { display: flex; align-items: center; justify-content: center;
    background: transparent; border: none; cursor: pointer; color: var(--text-muted);
    font-size: 14px; padding: 0 9px; border-radius: 8px; }
  .dr-opts-trigger:hover, .dr-opts-trigger[aria-expanded="true"] {
    color: var(--accent,#38bdf8); background: rgba(56,189,248,.12); }
  .dr-opts-panel {
    display: none; flex-direction: column; gap: 2px;
    position: absolute; top: calc(100% + 6px); right: 0;
    background: var(--card, #1a1a1a); border: 1px solid var(--border, #333); border-radius: 12px;
    padding: 6px; z-index: 200; min-width: 155px;
    box-shadow: 0 8px 32px rgba(0,0,0,.3);
  }
  .dr-opts-panel .dr-btn { width: 100%; text-align: left; padding: 9px 14px; border-radius: 8px; font-size: 13px;
    background: var(--bg, #0f0f0f); color: var(--text, #fff); border: 1px solid var(--border, #333); }
  .dr-opts-panel .dr-sim-speed { width: 100%; margin: 2px 0; padding: 6px 8px; border-radius: 8px;
    border: 1px solid var(--border, #333); background: var(--bg, #0f0f0f); color: var(--text, #fff); font-size: 13px; }
  .dr-btn-danger { color: #ef4444; border-color: rgba(239,68,68,.4); }
  .dr-statusbar {
    position: relative;
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    padding: 10px 14px; margin-bottom: 12px; border: 1px solid var(--border); border-radius: 12px;
    background: var(--card);
    position: sticky; top: 89px; z-index: 30;
  }
  .dr-status-info { display: flex; align-items: center; gap: 14px; min-width: 0; flex: 1; }
  .dr-status-pills { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; min-width: 0; }
  /* Prominent round/pick stats inside status bar */
  .dr-ss-stat { font-size: 15px; font-weight: 800; color: var(--text); white-space: nowrap; }
  .dr-ss-sep { font-size: 13px; color: var(--text-muted); font-weight: 700; }
  .dr-status-right { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
  .dr-sr-gap { flex: 1; }
  /* On-the-clock hero chip */
  .dr-onclock { display: flex; flex-direction: column; gap: 1px; padding: 6px 14px; border-radius: 10px;
    background: var(--bg); border: 1px solid var(--border); flex-shrink: 0; line-height: 1.2; }
  .dr-onclock-label { font-size: 9px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em;
    color: var(--text-muted); }
  .dr-onclock b { font-size: 15px; font-weight: 800; color: var(--text); white-space: nowrap; }
  .dr-onclock.dr-onclock-you { background: rgba(34,197,94,.1); border-color: rgba(34,197,94,.4); }
  .dr-onclock.dr-onclock-you b { color: #22c55e; }
  .dr-pill { font-size: 12px; font-weight: 700; padding: 3px 9px; border-radius: 999px;
    background: rgba(56,189,248,.14); color: var(--accent,#38bdf8); white-space: nowrap; }
  .dr-pill-you { background: rgba(34,197,94,.16); color: #22c55e; }
  .dr-pill-live { background: rgba(239,68,68,.16); color: #ef4444; animation: drPulse 1.6s ease-in-out infinite; }
  .dr-pill-upcoming { background: rgba(245,158,11,.16); color: #f59e0b; }
  .dr-pill-paused   { background: rgba(148,163,184,.16); color: #94a3b8; }
  .dr-pick-timer { font-size: 14px; font-weight: 800; color: var(--text); font-variant-numeric: tabular-nums;
    min-width: 40px; padding: 2px 8px; border-radius: 7px; background: rgba(127,127,127,.1); text-align: center; }
  .dr-pick-timer.urgent { color: #fff; background: #ef4444; animation: drPulse 1s ease-in-out infinite; }
  .dr-progress { font-size: 12px; color: var(--text-muted); white-space: nowrap; }
  .dr-save { font-size: 11px; color: #22c55e; }
  .dr-start-banner { display: flex; align-items: center; gap: 13px; margin: 0 0 12px; padding: 12px 16px; border-radius: 12px;
    background: linear-gradient(90deg, rgba(56,189,248,.18), rgba(56,189,248,.05)); border: 1px solid var(--accent,#38bdf8); }
  .dr-start-banner.is-live { background: linear-gradient(90deg, rgba(34,197,94,.18), rgba(34,197,94,.05)); border-color: #22c55e; }
  .dr-banner-ic { font-size: 22px; flex-shrink: 0; display: inline-flex; align-items: center; }
  .dr-banner-ic-live { animation: drPulse 1.4s ease-in-out infinite; }
  .dr-banner-txt { display: flex; flex-direction: column; line-height: 1.35; min-width: 0; flex: 1; }
  .dr-banner-txt b { font-size: 15px; font-weight: 800; color: var(--text); }
  .dr-banner-txt span { font-size: 12px; color: var(--text-muted); }
  .dr-start-cd { font-variant-numeric: tabular-nums; }
  .dr-banner-join { flex-shrink: 0; margin-left: auto; display: inline-flex; align-items: center; gap: 7px; white-space: nowrap;
    background: var(--accent,#38bdf8); color: #fff; font-weight: 700; font-size: 13px; text-decoration: none; padding: 8px 14px; border-radius: 8px; }
  .dr-start-banner.is-live .dr-banner-join { background: #22c55e; }
  .dr-banner-join i { font-size: 11px; }
  .dr-poll-status { font-size: 11px; color: var(--text-muted); display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
  .dr-poll-status .dr-poll-dot { width: 6px; height: 6px; border-radius: 50%; background: #22c55e; flex-shrink: 0; }
  .dr-poll-status.is-syncing .dr-poll-dot { background: var(--accent,#38bdf8); animation: drPulse 1s ease-in-out infinite; }
  /* Bottom-sheet drag handle (mobile only) */
  .dr-sheet-handle { display: none; }
  .dr-live-list { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
  .dr-live-head { font-size: 12px; font-weight: 700; color: var(--text-muted); }
  .dr-live-item { text-align: left; padding: 9px 12px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); font-size: 13px; cursor: pointer; }
  .dr-live-item:hover { border-color: var(--accent,#38bdf8); }
  .dr-live-status { font-size: 10px; font-weight: 800; text-transform: uppercase; padding: 1px 6px; border-radius: 999px; margin-right: 6px; }
  .dr-ls-drafting { background: rgba(239,68,68,.16); color: #ef4444; }
  .dr-ls-pre_draft { background: rgba(245,158,11,.16); color: #f59e0b; }
  .dr-ls-complete { background: rgba(148,163,184,.16); color: #94a3b8; }
  .dr-cols { display: grid; grid-template-columns: 1fr 375px; gap: 14px; align-items: start; }
  /* min-width:0 lets this grid item shrink to its track instead of growing to
     the wide board's width (the inner scroll, not the card, holds the overflow). */
  .dr-board-wrap { position: relative; min-width: 0; border: 1px solid var(--border); border-radius: 10px; background: var(--card); padding: 6px; }
  /* Only the board scrolls horizontally; the toolbar (Value/Pick Score toggle)
     stays pinned to the card so it doesn't drift when you scroll the grid. */
  .dr-board-scroll { overflow-x: auto; min-width: 0; }
  .dr-board { display: grid; gap: 5px; min-width: max-content; }
  .dr-cell {
    border: 1px solid var(--border); border-radius: 8px; padding: 5px 6px 0; min-height: 50px;
    background: var(--bg); display: flex; align-items: flex-end; gap: 6px; position: relative; overflow: hidden;
  }
  .dr-cell-body { padding: 5px; }
  .dr-cell-empty { opacity: .45; }
  .dr-cell-filled { background: linear-gradient(180deg, rgba(56,189,248,.05), var(--bg)); }
  .dr-cell-current { box-shadow: inset 0 0 0 2px var(--accent,#38bdf8); animation: drPulse 1.6s ease-in-out infinite; }
  @keyframes drPulse { 0%,100% { box-shadow: inset 0 0 0 2px var(--accent,#38bdf8); } 50% { box-shadow: inset 0 0 0 2px var(--accent,#38bdf8), 0 0 10px rgba(56,189,248,.2); } }
  .dr-cell-mine { box-shadow: inset 3px 0 0 var(--accent,#38bdf8); opacity: 1; }
  .dr-cell-mine.dr-cell-empty { opacity: 1; background: linear-gradient(180deg, rgba(56,189,248,.10), var(--bg)); }
  .dr-cell-claimed { box-shadow: inset 3px 0 0 #f59e0b; }     /* traded-in pick */
  .dr-cell-claimable { cursor: pointer; }
  .dr-cell-claimable:hover { outline: 1px dashed var(--accent,#38bdf8); outline-offset: -2px; }
  .dr-cell-mineflag { position: absolute; top: 2px; right: 5px; font-size: 8px; font-weight: 800;
    letter-spacing: .04em; color: var(--accent,#38bdf8); }
  .dr-cell-claimed .dr-cell-mineflag { color: #f59e0b; }
  /* Traded pick: who the pick was dealt to (shown on another team's seat). */
  .dr-cell-owner { position: absolute; top: 2px; right: 5px; font-size: 8px; font-weight: 800;
    letter-spacing: .04em; color: #f59e0b;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; pointer-events: none; }
  .dr-cell-just { animation: drPop .35s ease; }
  @keyframes drPop { 0% { transform: scale(.92); opacity: .3; } 100% { transform: scale(1); opacity: 1; } }
  .dr-cell-val { position: absolute; bottom: 2px; right: 5px; font-size: 9px; font-weight: 800; color: var(--accent,#38bdf8); }
  .dr-cell-num { position: absolute; top: 2px; left: 5px; font-size: 9px; font-weight: 700; color: var(--text-muted); }
  .dr-board-toolbar { display: flex; align-items: center; justify-content: flex-end; padding: 4px 6px 2px; }
  .dr-cell-toggle { display: flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; font-size: 10px; font-weight: 700; }
  .dr-ct-opt { padding: 3px 9px; cursor: pointer; color: var(--text-muted); transition: background .15s, color .15s; }
  .dr-ct-opt.is-active { background: var(--accent,#38bdf8); color: #fff; }
  .dr-ct-opt:not(.is-active):hover { background: var(--bg2,rgba(127,127,127,.12)); color: var(--text); }
  .dr-hs { width: 40px; height: 40px; border-radius: 8px 8px 0 0; object-fit: cover; object-position: top center;
    flex-shrink: 0; background: transparent; align-self: flex-end; }
  .dr-cell-body { min-width: 0; line-height: 1.2; }
  .dr-cell-name { font-size: 12px; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 96px; }
  .dr-cell-meta { font-size: 10px; color: var(--text-muted); }
  .dr-posbadge { font-size: 9px; font-weight: 700; color: #fff; border-radius: 3px; padding: 1px 4px; }
  .dr-colhead { font-size: 11px; font-weight: 700; color: var(--text-muted); text-align: center; padding: 2px 0; white-space: nowrap; }
  .dr-colhead-you { color: var(--accent,#38bdf8); }
  /* Round-label column sticks to the left so "R11" stays visible while you
     scroll the board horizontally through team columns. */
  .dr-rowhead { position: sticky; left: 0; z-index: 2; background: var(--card);
    box-shadow: 2px 0 4px -2px rgba(0,0,0,.25);
    display: flex; align-items: center; justify-content: center; }
  .dr-corner { z-index: 4; }
  .dr-side { border: 1px solid var(--border); border-radius: 10px; background: var(--card); display: flex; flex-direction: column;
    position: sticky; top: 158px; align-self: start; max-height: calc(100vh - 166px); z-index: 20; overflow: hidden; }
  /* Reuse the trade-calculator pill tabs (otc-main-tabs), evenly spread across panel */
  .dr-side-tabs.otc-main-tabs { width: auto; margin: 8px; }
  .dr-side-tabs .otc-main-tab { flex: 1; display: flex; align-items: center; justify-content: center;
    text-align: center; padding: 7px 4px; font-size: 12px; }
  /* Team needs hover tooltip */
  .dr-team-tip { background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 10px 12px; box-shadow: 0 8px 28px rgba(0,0,0,.28); min-width: 160px; }
  .dr-team-tip-name { font-size: 12px; font-weight: 800; color: var(--text); margin-bottom: 7px; }
  .dr-team-tip-pos-row { display: flex; gap: 5px; flex-wrap: wrap; }
  .dr-team-tip-pos { display: flex; flex-direction: column; align-items: center; padding: 4px 7px;
    border-radius: 7px; border: 1px solid transparent; }
  .dr-team-tip-pos-lbl { font-size: 8px; font-weight: 800; text-transform: uppercase; letter-spacing: .04em; }
  .dr-team-tip-pos-cnt { font-size: 13px; font-weight: 900; line-height: 1.3; }
  .dr-team-tip-next { font-size: 10px; color: var(--text-muted); margin-top: 7px; }
  .dr-team-tip-stats { display: flex; gap: 6px; margin-bottom: 7px; }
  .dr-team-tip-stat { flex: 1; text-align: center; background: var(--bg); border: 1px solid var(--border);
    border-radius: 7px; padding: 5px 6px; }
  .dr-team-tip-stat-v { font-size: 14px; font-weight: 900; color: var(--text); line-height: 1; }
  .dr-team-tip-stat-l { font-size: 8px; font-weight: 700; text-transform: uppercase; letter-spacing: .03em;
    color: var(--text-muted); margin-top: 3px; }
  .dr-team-tip-picks { display: flex; flex-direction: column; gap: 3px; max-height: 150px; overflow-y: auto; }
  .dr-team-tip-pick { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text); }
  .dr-team-tip-pick-pos { font-size: 8px; font-weight: 800; padding: 1px 4px; border-radius: 4px; flex-shrink: 0; }
  .dr-team-tip-pick-nm { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-team-tip-pick-tier { font-size: 9px; font-weight: 700; color: var(--accent,#38bdf8); margin-left: auto; flex-shrink: 0; }
  .dr-team-tip-empty { font-size: 11px; color: var(--text-muted); font-style: italic; }
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
  .dr-cliff-banner { background: rgba(245,158,11,.12); color: #f59e0b; border-color: rgba(245,158,11,.35); }
  .dr-cliff-banner b { color: #f59e0b; }
  .dr-strat-tag { margin-left: 6px; font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .04em; color: var(--text-muted); border: 1px solid var(--border);
    border-radius: 999px; padding: 1px 7px; vertical-align: middle; white-space: nowrap; }
  /* Pick trade evaluator (inside drModal) */
  .dr-pt-title { font-size: 15px; font-weight: 800; margin-bottom: 4px; }
  .dr-pt-sub { font-size: 12px; color: var(--text-muted); margin-bottom: 12px; line-height: 1.45; }
  .dr-pt-lbl { display: block; font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .05em; color: var(--text-muted); margin: 8px 0 3px; }
  .dr-pt-input { width: 100%; padding: 8px 10px; border: 1px solid var(--border); border-radius: 8px;
    background: var(--card); color: var(--text); font-size: 13px; outline: none; }
  .dr-pt-result { margin-top: 12px; }
  .dr-pt-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .dr-pt-side-h { font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .05em; color: var(--text-muted); margin-bottom: 4px; }
  .dr-pt-row { font-size: 13px; padding: 2px 0; }
  .dr-pt-proxy { font-size: 11px; color: var(--text-muted); }
  .dr-pt-verdict { margin-top: 10px; font-size: 14px; font-weight: 800; }
  .dr-prev-score-hero { border: 1px solid; border-radius: 10px; padding: 12px 10px 10px; margin-bottom: 12px; text-align: center; }
  .dr-prev-score-num { font-size: 44px; font-weight: 900; line-height: 1; }
  .dr-prev-score-lbl { font-size: 9px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin-top: 2px; }
  .dr-prev-score-reason { font-size: 12px; font-weight: 600; color: var(--text-muted); margin-top: 6px; }
  .dr-empty-note { padding: 22px 14px; font-size: 12px; color: var(--text-muted); text-align: center; }
  /* tiers */
  .dr-tier { font-size: 9px; font-weight: 800; padding: 1px 5px; border-radius: 999px;
    background: rgba(127,127,127,.18); color: var(--text-muted); flex-shrink: 0; }
  .dr-tier-cliff { background: rgba(239,68,68,.16); color: #ef4444; }
  /* pick score */
  .dr-ba-reason { font-size: 10px; color: var(--text-muted); margin-top: 3px; font-weight: 600; }
  .dr-ba-wait { font-size: 9.5px; color: #22c55e; margin-top: 2px; font-weight: 700; }
  .dr-prev-wait { display: flex; align-items: center; gap: 10px; border: 1px solid; border-radius: 9px;
    padding: 9px 12px; margin-bottom: 12px; }
  .dr-prev-wait-p { font-size: 18px; font-weight: 900; flex-shrink: 0; }
  .dr-prev-wait-t { font-size: 12px; font-weight: 600; color: var(--text); line-height: 1.35; }
  /* draft grade */
  .dr-pill-grade { background: rgba(34,197,94,.16); color: #22c55e; }
  .dr-grade-card { display: flex; align-items: center; gap: 12px; padding: 12px; margin: 10px 10px 4px;
    border: 1px solid var(--border); border-radius: 10px; background: var(--bg); }
  .dr-grade-letter { font-size: 34px; font-weight: 900; color: var(--accent,#38bdf8); line-height: 1; min-width: 48px; text-align: center; }
  .dr-grade-meta { flex: 1; min-width: 0; }
  .dr-grade-pace { font-size: 12px; font-weight: 700; color: var(--text); margin-bottom: 6px; }
  .dr-gbar-row { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
  .dr-gbar-lbl { font-size: 10px; color: var(--text-muted); width: 76px; flex-shrink: 0; }
  .dr-gbar { flex: 1; height: 6px; border-radius: 999px; background: rgba(127,127,127,.18); overflow: hidden; }
  .dr-gbar-fill { height: 100%; border-radius: 999px; }
  .dr-gbar-pct { font-size: 10px; font-weight: 800; width: 26px; text-align: right; flex-shrink: 0; }
  /* inline info-icon tooltip (ⓘ) */
  .dr-info { display:inline-flex; align-items:center; justify-content:center; width:13px; height:13px; border-radius:50%;
    border:1px solid var(--border); color:var(--text-muted); font-size:9px; font-weight:800; font-style:normal;
    cursor:help; margin-left:4px; position:relative; vertical-align:middle; line-height:1; flex-shrink:0; }
  .dr-info:hover, .dr-info:focus { border-color:var(--accent,#38bdf8); color:var(--accent,#38bdf8); outline:none; }
  .dr-info::after { content: attr(data-tip); position:absolute; top:calc(100% + 6px); left:50%; transform:translateX(-50%);
    width:max-content; max-width:210px; background:var(--card); color:var(--text); border:1px solid var(--border);
    border-radius:8px; padding:7px 9px; font-size:11px; font-weight:500; font-style:normal; line-height:1.4; text-align:left;
    box-shadow:0 8px 24px rgba(0,0,0,.28); opacity:0; pointer-events:none; transition:opacity .12s; z-index:600; white-space:normal; }
  .dr-info:hover::after, .dr-info:focus::after { opacity:1; }
  /* glossary popover */
  .dr-help-btn { width:26px; height:26px; border-radius:7px; border:1px solid var(--border); background:var(--bg);
    color:var(--text-muted); font-size:13px; font-weight:800; cursor:pointer; flex-shrink:0; line-height:1; }
  .dr-help-btn:hover { border-color:var(--accent,#38bdf8); color:var(--accent,#38bdf8); }
  .dr-gloss-overlay { position:fixed; inset:0; z-index:9998; background:rgba(0,0,0,.52); display:flex;
    align-items:center; justify-content:center; padding:18px; }
  .dr-gloss-card { background:var(--card); border:1px solid var(--border); border-radius:14px; width:100%; max-width:440px;
    max-height:82vh; overflow-y:auto; padding:18px 18px 22px; position:relative; box-shadow:0 24px 70px rgba(0,0,0,.4); }
  .dr-gloss-title { font-size:16px; font-weight:800; color:var(--text); margin:0 0 12px; padding-right:28px; }
  .dr-gloss-close { position:absolute; top:12px; right:12px; width:28px; height:28px; border-radius:8px; border:1px solid var(--border);
    background:var(--bg); color:var(--text-muted); font-size:18px; cursor:pointer; line-height:1; }
  .dr-gloss-item { padding:9px 0; border-top:1px solid var(--border); }
  .dr-gloss-item:first-of-type { border-top:none; }
  .dr-gloss-term { font-size:12.5px; font-weight:800; color:var(--text); margin-bottom:2px; }
  .dr-gloss-def { font-size:12px; font-weight:500; color:var(--text-muted); line-height:1.45; }
  /* player preview */
  .dr-preview-overlay { position: fixed; inset: 0; z-index: 1000; background: rgba(0,0,0,.45);
    display: flex; align-items: flex-start; justify-content: center; padding: 16px; overflow-y: auto; }
  .dr-preview-card { position: relative; width: 100%; max-width: 380px; background: var(--card);
    border: 1px solid var(--border); border-radius: 16px; padding: 18px 18px 16px; box-shadow: 0 18px 56px rgba(0,0,0,.34); margin: auto; }
  .dr-prev-close { position: absolute; top: 10px; right: 12px; width: 28px; height: 28px; background: var(--bg);
    border: 1px solid var(--border); border-radius: 999px; font-size: 17px; line-height: 1;
    color: var(--text-muted); cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: background .12s, color .12s; }
  .dr-prev-close:hover { background: rgba(239,68,68,.12); color: #ef4444; }
  .dr-prev-top { display: flex; align-items: flex-end; gap: 13px; margin-bottom: 14px; padding-right: 28px; }
  .dr-prev-hs { width: 66px; height: 66px; border-radius: 12px 12px 0 0; object-fit: cover; object-position: top center; background: rgba(127,127,127,.08); flex-shrink: 0; }
  .dr-prev-name { font-size: 19px; font-weight: 800; color: var(--text); line-height: 1.15; }
  .dr-prev-meta { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
  .dr-prev-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; margin-bottom: 14px; }
  .dr-prev-stat { background: var(--bg); border: 1px solid var(--border); border-radius: 9px; padding: 9px 4px 8px; text-align: center; }
  .dr-prev-stat-v { font-size: 16px; font-weight: 800; color: var(--text); letter-spacing: -.01em; line-height: 1; }
  .dr-prev-stat-l { font-size: 8.5px; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin-top: 4px; font-weight: 700; }
  .dr-prev-stat-sub { font-size: 8.5px; color: var(--text-muted); margin-top: 1px; opacity: .7; font-weight: 600; }
  .dr-prev-btns { display: flex; flex-direction: column; gap: 8px; margin-top: 4px; }
  .dr-prev-draft { width: 100%; }
  .dr-prev-profile { display: block; width: 100%; text-align: center; text-decoration: none; box-sizing: border-box; }
  .dr-prev-note { font-size: 12px; color: var(--text-muted); text-align: center; padding: 6px 0; }
  /* queue star */
  .dr-star { background: none; border: none; cursor: pointer; font-size: 15px; line-height: 1; flex-shrink: 0;
    color: var(--text-muted); padding: 2px 2px 0; }
  .dr-star.on { color: #f59e0b; }
  .dr-side-title { font-size: 14px; font-weight: 800; color: var(--text); }
  .dr-side-controls { display: flex; gap: 6px; }
  .dr-side-controls input { flex: 1; min-width: 0; padding: 7px 9px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 12px; }
  .dr-side-controls select { padding: 7px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 12px; flex-shrink: 0; max-width: 110px; }
  .dr-pos-filters { display: flex; gap: 4px; flex-wrap: wrap; }
  .dr-pos { font-size: 11px; font-weight: 700; padding: 4px 9px; border-radius: 999px; border: 1px solid var(--border); background: var(--bg); color: var(--text-muted); cursor: pointer; }
  .dr-pos.active { background: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8); color: #fff; }
  .dr-adp-src { font-size: 10px; color: var(--text-muted); }
  .dr-ba-list { overflow-y: auto; flex: 1; }
  .dr-ba-row { display: flex; align-items: center; gap: 10px; padding: 8px 12px 8px 5px; border-bottom: 1px solid var(--border); cursor: pointer; transition: background .12s; }
  .dr-ba-row:hover { background: rgba(56,189,248,.06); }
  .dr-ba-hs { width: 65px; height: 65px; border-radius: 9px 9px 0 0; object-fit: cover; object-position: top center;
    flex-shrink: 0; background: transparent; align-self: flex-end; margin-bottom: -8px; }
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
  .dr-ba-right-col { display: flex; flex-direction: column; align-items: flex-end; gap: 5px; flex-shrink: 0; }
  .dr-ba-metrics { display: flex; align-items: center; gap: 8px; justify-content: flex-end; }
  .dr-ba-actions { display: flex; align-items: center; gap: 6px; }
  .dr-ba-draft { padding: 5px 12px; border-radius: 7px; border: 1px solid var(--accent,#122d4b);
    background: transparent; color: var(--accent,#122d4b); font-size: 11px; font-weight: 800;
    cursor: pointer; white-space: nowrap; transition: background .12s, color .12s; }
  .dr-ba-row:hover .dr-ba-draft, .dr-ba-draft:hover { background: var(--accent,#122d4b); color: #fff; }
  /* ── Player availability indicators (Players tab) ── */
  .dr-ba-row.dr-avail-hi { box-shadow: inset 3px 0 0 #22c55e; }
  .dr-ba-row.dr-avail-md { box-shadow: inset 3px 0 0 #f59e0b; }
  .dr-ba-row.dr-avail-lo { box-shadow: inset 3px 0 0 #ef4444; }
  .dr-ba-avail { font-size: 9.5px; font-weight: 700; margin-top: 2px; }
  /* ── Preview modal availability track ── */
  .dr-prev-avail-track { margin-bottom: 12px; }
  .dr-prev-avail-label { font-size: 10px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }
  .dr-prev-avail-picks { display: flex; flex-wrap: wrap; gap: 6px; }
  .dr-prev-avail-pick { display: inline-flex; align-items: baseline; gap: 4px; padding: 5px 10px; border-radius: 8px; }
  .dr-prev-avail-pn { font-size: 10px; font-weight: 600; color: var(--text-muted); }
  .dr-loading { display: flex; align-items: center; gap: 10px; padding: 24px; color: var(--text-muted); font-size: 13px; justify-content: center; }
  @media (max-width: 900px) {
    .dr-cols { grid-template-columns: 1fr; padding-bottom: 52vh; }
    .dr-statusbar { top: 0; }
    /* The side panel becomes a draggable bottom sheet */
    .dr-side {
      position: fixed; left: 0; right: 0; bottom: 0; top: auto;
      width: 100%; height: 92vh; max-height: 92vh; align-self: auto; order: 0;
      border-radius: 18px 18px 0 0; border-bottom: none;
      box-shadow: 0 -10px 40px rgba(0,0,0,.28); z-index: 50;
      transform: translateY(42vh);          /* default: ~50vh visible (mid snap) */
      transition: transform .3s cubic-bezier(.32,.72,0,1);
    }
    .dr-side.dragging { transition: none; }
    .dr-sheet-handle {
      display: flex; align-items: center; justify-content: center;
      width: 100%; height: 26px; padding: 0; border: none; background: none;
      cursor: grab; flex-shrink: 0; touch-action: none;
    }
    .dr-sheet-handle:active { cursor: grabbing; }
    .dr-sheet-grip { width: 40px; height: 5px; border-radius: 999px; background: var(--border);
      transition: background .12s; }
    .dr-sheet-handle:active .dr-sheet-grip { background: var(--accent,#38bdf8); }
    .dr-ba-list { max-height: none; }
    .dr-board-wrap { max-width: calc(100vw - 16px); }
  }
  @media (max-width: 480px) {
    /* Hide player headshots in board cells on very small screens so columns stay readable */
    .dr-hs { display: none; }
    .dr-cell { min-height: 38px; }
  }
  @media (max-width: 640px) {
    .dr-wrap { padding: 8px 8px 32px; }
    .dr-setup-card { padding: 16px; }
    /* Status bar: two compact rows */
    .dr-statusbar { padding: 6px 10px; gap: 5px; flex-direction: column; align-items: stretch; border-radius: 10px; }
    /* Row 1: on-clock inline + pills scroll */
    .dr-status-info { width: 100%; flex-wrap: nowrap; gap: 8px; align-items: center; }
    .dr-onclock { flex-direction: row; align-items: center; gap: 5px; padding: 4px 10px; flex-shrink: 0; }
    .dr-onclock-label { font-size: 8px; }
    .dr-onclock b { font-size: 13px; white-space: nowrap; }
    .dr-status-pills { gap: 4px; flex-wrap: nowrap; overflow-x: auto; scrollbar-width: none; min-width: 0; flex: 1; }
    .dr-status-pills::-webkit-scrollbar { display: none; }
    .dr-ss-stat { font-size: 13px; }
    .dr-pill { font-size: 10px; padding: 2px 7px; }
    .dr-pick-timer { font-size: 12px; min-width: 32px; padding: 2px 6px; }
    .dr-progress, .dr-save { font-size: 10px; white-space: nowrap; }
    /* Row 2: buttons scroll */
    .dr-status-right {
      width: 100%; gap: 5px; overflow-x: auto; -webkit-overflow-scrolling: touch;
      flex-wrap: nowrap; padding-bottom: 2px; scrollbar-width: none;
    }
    .dr-status-right::-webkit-scrollbar { display: none; }
    .dr-status-right .dr-btn { flex: 0 0 auto; padding: 7px 11px; font-size: 12px; }
    .dr-side-tabs .otc-main-tab { font-size: 11px; padding: 6px 2px; }
    .dr-opts-trigger { font-size: 13px; padding: 0 8px; }
    .dr-board-wrap { padding: 4px; max-width: calc(100vw - 16px); }
    .dr-cta, .dr-setup-cta { flex-direction: column; align-items: stretch; }
    .dr-setup-cta .dr-btn { width: 100%; }
    .dr-prev-stats { grid-template-columns: repeat(2, 1fr); }
  }
  /* Summary overlay */
  .dr-summary-overlay { position:fixed; inset:0; z-index:1001; background:rgba(0,0,0,.6);
    display:flex; align-items:flex-start; justify-content:center; padding:20px 16px; overflow-y:auto; }
  .dr-summary-card { position:relative; width:100%; max-width:500px; margin:auto; background:var(--card);
    border:1px solid var(--border); border-radius:20px; overflow:hidden;
    box-shadow:0 24px 80px rgba(0,0,0,.5); }
  /* Grade ring + bars header */
  .dr-sum-header { padding:20px 20px 0; }
  .dr-sum-title { font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.1em;
    color:var(--text-muted); text-align:center; margin-bottom:14px; }
  .dr-sum-grade-wrap { display:flex; align-items:center; gap:18px; padding-bottom:16px; }
  .dr-sum-grade-ring { width:76px; height:76px; border-radius:50%; border:3px solid;
    display:flex; align-items:center; justify-content:center; flex-shrink:0; }
  .dr-sum-grade { font-size:30px; font-weight:900; line-height:1; }
  .dr-sum-grade-bars { flex:1; display:flex; flex-direction:column; gap:5px; }
  /* Stats strip */
  .dr-sum-stats { display:flex; border-top:1px solid var(--border); border-bottom:1px solid var(--border); }
  .dr-sum-stat { flex:1; text-align:center; padding:13px 4px; }
  .dr-sum-stat:not(:last-child) { border-right:1px solid var(--border); }
  .dr-sum-stat-v { font-size:20px; font-weight:900; color:var(--text); line-height:1; }
  .dr-sum-stat-l { font-size:9px; color:var(--text-muted); margin-top:3px; text-transform:uppercase; letter-spacing:.04em; }
  /* Archetype / window strip */
  .dr-sum-arch { display:flex; align-items:center; justify-content:center; gap:14px; flex-wrap:wrap;
    padding:11px 16px; border-bottom:1px solid var(--border); }
  .dr-sum-arch-item { display:flex; flex-direction:column; align-items:center; gap:4px; }
  .dr-sum-arch-tag { font-size:9px; font-weight:800; text-transform:uppercase; letter-spacing:.06em; color:var(--text-muted); }
  .dr-sum-arch-label { font-size:14px; font-weight:900; color:var(--accent); line-height:1.1; }
  .dr-sum-arch-div { width:1px; height:32px; background:var(--border); flex-shrink:0; }
  /* Competitive window chips */
  .dr-sum-win { font-size:12px; font-weight:800; padding:4px 10px; border-radius:999px; white-space:nowrap; }
  .dr-win-winnow { background:rgba(34,197,94,.16); color:#22c55e; }
  .dr-win-balanced { background:rgba(245,158,11,.16); color:#f59e0b; }
  .dr-win-future { background:rgba(56,189,248,.16); color:#38bdf8; }
  /* Body wrapper + section labels */
  .dr-sum-body-wrap { padding:0 16px 4px; }
  .dr-sum-section { font-size:9px; font-weight:800; text-transform:uppercase; letter-spacing:.08em;
    color:var(--text-muted); margin:14px 0 6px; }
  /* Player rows */
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
  /* Footer */
  .dr-sum-footer { display:flex; gap:8px; padding:14px 16px 16px; }
  .dr-sum-footer .dr-btn { flex:1; text-align:center; }
  /* Share preview overlay */
  .dr-shareview-overlay { position:fixed; inset:0; z-index:1002; background:rgba(0,0,0,.6);
    display:flex; align-items:center; justify-content:center; padding:16px; }
  .dr-shareview-card { position:relative; background:var(--card); border:1px solid var(--border);
    border-radius:16px; padding:20px; max-width:520px; width:100%; max-height:calc(100vh - 32px); overflow-y:auto;
    box-shadow:0 24px 60px rgba(0,0,0,.4); display:flex; flex-direction:column; gap:14px; }
  .dr-shareview-tabs { display:flex; gap:6px; }
  .dr-shareview-tab { padding:6px 16px; border-radius:8px; border:1px solid var(--border);
    background:var(--bg); color:var(--text-muted); font-size:13px; font-weight:600; cursor:pointer; }
  .dr-shareview-tab.is-active { background:var(--accent,#38bdf8); border-color:var(--accent,#38bdf8); color:#fff; }
  .dr-shareview-img { width:100%; border-radius:10px; border:1px solid var(--border); display:block; }
  .dr-shareview-footer { display:flex; gap:10px; }
  .dr-shareview-footer .dr-btn { flex:1; text-align:center; }
  /* Custom modal */
  .dr-modal-box { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:24px 28px;
    max-width:380px; width:100%; box-shadow:0 24px 60px rgba(0,0,0,.45); }
  .dr-modal-msg { font-size:15px; color:var(--text); line-height:1.55; margin-bottom:20px; }
  .dr-modal-btns { display:flex; gap:10px; justify-content:flex-end; }
  /* Complete-draft sidebar footer */
  #drCompleteBar { padding:10px; border-top:1px solid var(--border); display:flex; flex-direction:column; gap:7px; flex-shrink:0; }
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
  .dr-step-val-ro { font-size:14px; font-weight:800; color:var(--text-muted); min-width:18px; text-align:center; }
  .dr-roster-src { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
  .dr-roster-src-tag { font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.04em;
    padding:2px 8px; border-radius:999px; background:rgba(56,189,248,.14); color:var(--accent,#38bdf8); }
  .dr-roster-src-custom { background:rgba(168,85,247,.14); color:#a855f7; }
  .dr-roster-src-btn { font-size:11px; font-weight:700; color:var(--text-muted); background:none; border:1px solid var(--border);
    border-radius:6px; padding:2px 9px; cursor:pointer; line-height:1.6; }
  .dr-roster-src-btn:hover { color:var(--accent,#38bdf8); border-color:var(--accent,#38bdf8); }
  /* ── Draft capital (setup) ── */
  .dr-cap-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
  .dr-cap-count { font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase; letter-spacing:.04em; }
  .dr-cap-list { max-height:440px; overflow-y:auto; border:1px solid var(--border); border-radius:12px;
    background:var(--bg); padding:4px; }
  .dr-cap-list::-webkit-scrollbar { width:8px; }
  .dr-cap-list::-webkit-scrollbar-thumb { background:rgba(127,127,127,.28); border-radius:8px; }
  .dr-cap-row { display:flex; align-items:center; gap:10px; padding:7px 8px; border-radius:9px;
    transition:background .12s; }
  .dr-cap-row:hover { background:rgba(127,127,127,.06); }
  .dr-cap-row.is-open { background:rgba(56,189,248,.06); }
  .dr-cap-rlabel { font-size:11px; font-weight:900; color:var(--text); width:54px; flex-shrink:0;
    letter-spacing:.02em; }
  .dr-cap-rpicks { flex:1; min-width:0; display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
  .dr-cap-none { font-size:11.5px; color:var(--text-muted); opacity:.6; }
  .dr-cap-pill { display:inline-flex; align-items:center; gap:3px; font-size:12px; font-weight:700;
    padding:4px 9px; border-radius:999px; background:rgba(56,189,248,.12); color:var(--text);
    cursor:pointer; transition:background .12s, color .12s; user-select:none; }
  .dr-cap-pill:hover { background:#ef4444; color:#fff; }
  .dr-cap-pill-x { font-style:normal; font-size:13px; line-height:1; opacity:0; width:0; overflow:hidden;
    transition:opacity .12s, width .12s; }
  .dr-cap-pill:hover .dr-cap-pill-x { opacity:1; width:11px; }
  .dr-cap-pill-traded { background:rgba(245,158,11,.16); }
  .dr-cap-addbtn { width:26px; height:26px; flex-shrink:0; border:none; border-radius:7px;
    background:rgba(127,127,127,.1); color:var(--text-muted); font-size:17px; font-weight:600; line-height:1;
    cursor:pointer; display:flex; align-items:center; justify-content:center; padding:0; transition:all .12s; }
  .dr-cap-addbtn:hover { background:var(--accent,#38bdf8); color:#fff; }
  .dr-cap-row.is-open .dr-cap-addbtn { background:var(--accent,#38bdf8); color:#fff; }
  .dr-cap-picker { display:flex; align-items:center; gap:10px; padding:4px 8px 10px 72px; }
  .dr-cap-picker-lbl { font-size:10px; font-weight:700; color:var(--text-muted); text-transform:uppercase;
    letter-spacing:.04em; flex-shrink:0; }
  .dr-cap-slots { display:flex; flex-wrap:wrap; gap:5px; }
  .dr-cap-slot { width:28px; height:28px; border:1px solid var(--border); border-radius:7px; background:var(--card);
    color:var(--text-muted); font-size:11px; font-weight:700; cursor:pointer; transition:all .12s; padding:0; }
  .dr-cap-slot:hover { border-color:var(--accent,#38bdf8); color:var(--accent,#38bdf8); }
  .dr-cap-slot.home { border-style:dashed; }
  .dr-cap-slot.on { background:var(--accent,#38bdf8); border-color:var(--accent,#38bdf8); color:#fff; }
  .dr-cap-late { border-top:1px solid var(--border); margin-top:4px; }
  .dr-cap-latehead { width:100%; display:flex; align-items:center; gap:10px; padding:9px 8px; border:none;
    background:none; cursor:pointer; transition:background .12s; border-radius:9px; }
  .dr-cap-latehead:hover { background:rgba(127,127,127,.06); }
  .dr-cap-latecount { flex:1; text-align:left; font-size:11.5px; color:var(--text-muted); }
  .dr-cap-chev { font-style:normal; font-size:9px; color:var(--text-muted); }
  .dr-cap-latebody { padding-bottom:2px; }
  /* ── Team tab PS badge ── */
  .dr-rslot-ps { font-size:11px; font-weight:800; flex-shrink:0; margin-right:2px; }
  /* ── Positional scarcity bar ── */
  .dr-scarcity { display: flex; border-bottom: 1px solid var(--border); }
  .dr-scar-pos { flex: 1; display: flex; flex-direction: column; align-items: center; padding: 5px 2px;
    cursor: pointer; transition: background .12s; }
  .dr-scar-pos:hover { background: rgba(56,189,248,.07); }
  .dr-scar-pos:not(:last-child) { border-right: 1px solid var(--border); }
  .dr-scar-count { font-size: 14px; font-weight: 900; line-height: 1; }
  .dr-scar-label { font-size: 8px; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin-top: 1px; }
  /* ── Best-at-position chips + T1-2 counts ── */
  .dr-bchips-header { display: flex; align-items: center; justify-content: space-between;
    padding: 5px 10px 4px; cursor: pointer; transition: background .12s; border-bottom: 1px solid var(--border); }
  .dr-bchips-header:hover { background: rgba(127,127,127,.05); }
  .dr-bchips-label { font-size: 10px; font-weight: 800; text-transform: uppercase;
    letter-spacing: .06em; color: var(--text-muted); }
  .dr-bchips-hint { font-size: 9px; color: var(--text-muted); opacity: .7; }
  .dr-bchips-section-title { font-size: 9px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .06em; color: var(--text-muted); padding: 5px 10px 2px; }
  .dr-bchips { display: flex; gap: 6px; padding: 4px 8px 7px; overflow-x: auto; -webkit-overflow-scrolling: touch;
    border-bottom: 1px solid var(--border); }
  .dr-bchip { display: flex; align-items: flex-end; gap: 6px; padding: 5px 8px 5px; border-radius: 9px;
    border: 1px solid var(--border); background: var(--bg); cursor: pointer; flex-shrink: 0;
    transition: border-color .12s, background .12s; }
  .dr-bchip:hover { border-color: var(--accent,#38bdf8); background: rgba(56,189,248,.06); }
  .dr-bchip-img { width: 30px; height: 30px; border-radius: 5px 5px 0 0; object-fit: cover;
    object-position: top center; align-self: flex-end; flex-shrink: 0; }
  .dr-bchip-body { min-width: 0; line-height: 1.3; }
  .dr-bchip-name { font-size: 11px; font-weight: 700; color: var(--text); white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; max-width: 68px; }
  .dr-bchip-adp { font-size: 9px; color: var(--text-muted); }
  /* ── Balance alert ── */
  .dr-bal-alert { margin: 8px 10px 2px; padding: 7px 10px; border-radius: 8px; font-size: 11.5px;
    background: rgba(245,158,11,.12); color: #b45309; border: 1px solid rgba(245,158,11,.3);
    line-height: 1.4; }
  .dr-bal-alert b { color: #f59e0b; }
  /* ── Bye week conflict flag ── */
  .dr-bye-flag { font-size: 9px; font-weight: 800; padding: 1px 5px; border-radius: 4px;
    background: rgba(239,68,68,.14); color: #ef4444; margin-left: 5px; white-space: nowrap; }
  /* ── Compare button in rows ── */
  .dr-cmp-btn { background: none; border: none; cursor: pointer; font-size: 10px; font-weight: 800;
    line-height: 1; color: var(--text-muted); padding: 3px 5px; border-radius: 5px;
    border: 1px solid transparent; transition: all .12s; flex-shrink: 0; letter-spacing: .02em; }
  .dr-cmp-btn:hover, .dr-cmp-btn.on { color: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8);
    background: rgba(56,189,248,.1); }
  /* ── Player comparison overlay ── */
  .dr-cmp-overlay { position: fixed; inset: 0; z-index: 1000; background: rgba(0,0,0,.45);
    display: flex; align-items: flex-start; justify-content: center; padding: 16px; overflow-y: auto; }
  .dr-cmp-card { position: relative; width: 100%; max-width: 540px; background: var(--card);
    border: 1px solid var(--border); border-radius: 14px; padding: 18px 16px 16px;
    box-shadow: 0 16px 50px rgba(0,0,0,.3); margin: auto; }
  .dr-cmp-close { position: absolute; top: 8px; right: 10px; background: none; border: none;
    font-size: 22px; line-height: 1; color: var(--text-muted); cursor: pointer; }
  .dr-cmp-title { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .06em;
    color: var(--text-muted); text-align: center; margin-bottom: 12px; }
  .dr-cmp-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .dr-cmp-player { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 10px; }
  .dr-cmp-top { display: flex; align-items: flex-end; gap: 8px; margin-bottom: 8px; }
  .dr-cmp-hs { width: 44px; height: 44px; border-radius: 8px 8px 0 0; object-fit: cover;
    object-position: top center; flex-shrink: 0; }
  .dr-cmp-name { font-size: 12px; font-weight: 800; color: var(--text); line-height: 1.2; }
  .dr-cmp-meta { font-size: 10px; color: var(--text-muted); margin-top: 2px; }
  .dr-cmp-ps { font-size: 30px; font-weight: 900; line-height: 1; text-align: center; margin: 6px 0 0; }
  .dr-cmp-ps-lbl { font-size: 8px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em;
    color: var(--text-muted); text-align: center; margin-bottom: 8px; }
  .dr-cmp-stats { display: flex; flex-direction: column; gap: 3px; }
  .dr-cmp-stat { display: flex; justify-content: space-between; align-items: center; padding: 3px 5px;
    border-radius: 5px; }
  .dr-cmp-stat-lbl { font-size: 10px; color: var(--text-muted); font-weight: 600; }
  .dr-cmp-stat-val { font-size: 12px; font-weight: 800; color: var(--text); }
  .dr-cmp-stat.win { background: rgba(34,197,94,.12); }
  .dr-cmp-stat.win .dr-cmp-stat-val { color: #22c55e; }
  .dr-cmp-actions { display: flex; gap: 8px; justify-content: center; margin-top: 14px; flex-wrap: wrap; }
  /* ── League tab ── */
  .dr-lg-wrap { padding: 10px; display: flex; flex-direction: column; gap: 6px; overflow-y: auto; }
  .dr-lg-row { border: 1px solid var(--border); border-radius: 9px; padding: 9px 10px; background: var(--bg); }
  .dr-lg-mine { border-color: var(--accent,#38bdf8); background: rgba(56,189,248,.05); }
  .dr-lg-onclock { border-color: #22c55e; background: rgba(34,197,94,.05); animation: drPulse 1.6s ease-in-out infinite; }
  .dr-lg-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 7px; gap: 6px; }
  .dr-lg-team { font-size: 12px; font-weight: 800; color: var(--text); }
  .dr-lg-next { font-size: 10px; color: var(--text-muted); flex-shrink: 0; }
  .dr-lg-next-you { color: #22c55e; font-weight: 700; }
  .dr-lg-pos-row { display: flex; gap: 4px; flex-wrap: wrap; }
  .dr-lg-pos { display: flex; flex-direction: column; align-items: center; padding: 3px 7px;
    border-radius: 6px; border: 1px solid; min-width: 36px; }
  .dr-lg-pos-label { font-size: 8px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; }
  .dr-lg-pos-count { font-size: 11px; font-weight: 700; color: var(--text); margin-top: 1px; }
  .dr-lg-need { font-size: 10px; color: var(--text-muted); margin-top: 6px; }
  .dr-lg-need b { font-weight: 800; }
  .dr-lg-picks { font-size: 10px; color: var(--text-muted); margin-top: 4px; line-height: 1.4; }
  /* ── Roster projection card ── */
  .dr-proj-card { margin: 10px 0 2px; padding: 10px 12px; border-radius: 10px;
    border: 1px solid var(--border); background: var(--bg); }
  .dr-proj-title { font-size: 9px; font-weight: 800; text-transform: uppercase;
    letter-spacing: .06em; color: var(--text-muted); margin-bottom: 8px; }
  .dr-proj-stats { display: flex; gap: 10px; }
  .dr-proj-stat { flex: 1; text-align: center; }
  .dr-proj-val { font-size: 19px; font-weight: 900; color: var(--text); line-height: 1; }
  .dr-proj-lbl { font-size: 9px; color: var(--text-muted); margin-top: 2px; }
  .dr-proj-bar-wrap { margin-top: 8px; }
  .dr-proj-bar-bg { height: 5px; border-radius: 3px; background: rgba(127,127,127,.15); overflow: hidden; }
  .dr-proj-bar-fill { height: 100%; border-radius: 3px; background: var(--accent,#38bdf8); }
  .dr-proj-bar-lbl { font-size: 9px; color: var(--text-muted); margin-top: 3px; }
  /* League grades list */
  .dr-sum-league { display: flex; flex-direction: column; gap: 3px; }
  .dr-sum-lrow { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-radius: 10px;
    background: var(--bg); border: 1px solid var(--border); cursor: pointer; transition: background .12s; }
  .dr-sum-lrow:hover { background: rgba(127,127,127,.08); }
  .dr-sum-lrow.is-me { border-color: var(--accent,#38bdf8); background: rgba(56,189,248,.08); }
  .dr-sum-lrank { width: 20px; flex-shrink: 0; font-size: 12px; font-weight: 900; color: var(--text-muted); text-align: center; }
  .dr-sum-lrank.gold { color: #f59e0b; }
  .dr-sum-lrank.silver { color: #94a3b8; }
  .dr-sum-lrank.bronze { color: #cd7c2f; }
  .dr-sum-lname { flex: 1; min-width: 0; font-size: 13px; font-weight: 700; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-sum-lrow.is-me .dr-sum-lname { color: var(--accent,#38bdf8); }
  .dr-sum-lwin { font-size: 9.5px; font-weight: 800; padding: 2px 7px; border-radius: 999px; white-space: nowrap; flex-shrink: 0; }
  .dr-sum-lgrade { font-size: 18px; font-weight: 900; flex-shrink: 0; width: 32px; text-align: right; }
  .dr-sum-lchev { font-size: 9px; color: var(--text-muted); flex-shrink: 0; transition: transform .2s; }
  .dr-sum-lrow.is-open .dr-sum-lchev { transform: rotate(180deg); }
  /* Expandable team starter detail */
  .dr-sum-ldtl { display: none; padding: 4px 6px 8px 38px; }
  .dr-sum-ldtl.is-open { display: block; }
  .dr-sum-ldtl-row { display: flex; align-items: center; gap: 6px; padding: 3px 0; }
  .dr-sum-ldtl-slot { font-size: 8px; font-weight: 800; color: #fff; border-radius: 3px; padding: 2px 0;
    width: 28px; flex-shrink: 0; text-align: center; }
  .dr-sum-ldtl-name { flex: 1; font-size: 11px; font-weight: 600; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-sum-ldtl-pick { font-size: 9px; color: var(--text-muted); flex-shrink: 0; }
  .dr-sum-ldtl-ps { font-size: 11px; font-weight: 800; flex-shrink: 0; }
</style>

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
      var typeLabel = d.draft_type ? (d.draft_type.charAt(0).toUpperCase() + d.draft_type.slice(1)) : 'Draft';
      var title = typeLabel + ' Draft'
        + ' · ' + (d.teams || '?') + ' teams · ' + (d.rounds || '?') + ' rounds';
      html += '<div class="dr-hist-card">'
        + '<div class="dr-hist-body"><div class="dr-hist-title">' + esc(title) + statusTag(d.status) + '</div>'
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
