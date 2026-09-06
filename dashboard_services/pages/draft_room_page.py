"""
Standalone Draft Room page.

Supports manual drafting for both startup (all players) and rookie drafts,
with snake / linear / third-round-reversal pick order. Live Sleeper sync,
ESPN live companion sync (observe-only), persistence/history, and the full
command-center panels.

The page is self-contained: its CSS is inlined here and its JS lives in
static/draft_room.js (loaded as a deferred external script so the browser caches
it across visits instead of re-receiving ~210KB inline on every load). Server
values are passed via a small window.__draftCfg JSON blob the script reads on
start; the JS file needs no f-string brace escaping.
"""
from __future__ import annotations

import hashlib
import json
import os
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


def _static_v(name: str) -> str:
    try:
        _p = Path(__file__).resolve().parents[2] / "static" / name
        return hashlib.md5(_p.read_bytes()).hexdigest()[:8]
    except OSError:
        return "0"


def build_draft_room_body(
        league_id: Optional[str],
        season: Optional[int],
        platform: Optional[str] = None,
        *,
        is_guest: bool = False,
        num_teams: Optional[int] = None,
        is_superflex: bool = False,
        roster_positions: Optional[list] = None,
        scoring: Optional[dict] = None,
        viewer_user_id: Optional[str] = None,
        viewer_roster_id: Optional[str] = None,
        num_rounds_rookie: Optional[int] = None,
        num_rounds_startup: Optional[int] = None,
        keepers: Optional[dict] = None,
        show_keeper: bool = True,
        has_premium: bool = False,
        is_auction: bool = False,
        auction_budget: Optional[float] = None,
        is_best_ball: bool = False,
) -> str:
    _dr_has_league = bool(league_id and platform and season)
    cfg = {
        "leagueId": league_id or "",
        "season": int(season) if season else None,
        "platform": platform or "sleeper",
        # Link target for the Draft History page (league-scoped when available).
        "historyUrl": (
            f"/{platform}/{int(season)}/{league_id}/draft/history"
            if _dr_has_league else "/draft/history"
        ),
        "cheatSheetUrl": (
            f"/{platform}/{int(season)}/{league_id}/draft/cheat-sheet"
            if _dr_has_league else "/draft/cheat-sheet"
        ),
        "cheatSheetEmbedUrl": (
            f"/{platform}/{int(season)}/{league_id}/draft/cheat-sheet/embed"
            if _dr_has_league else "/draft/cheat-sheet/embed"
        ),
        "isGuest": bool(is_guest),
        "numTeams": int(num_teams) if num_teams else None,
        "isSuperflex": bool(is_superflex),
        "rosterPositions": list(roster_positions) if roster_positions else None,
        "scoring": scoring or None,
        "viewerUserId": str(viewer_user_id) if viewer_user_id else "",
        "viewerRosterId": str(viewer_roster_id) if viewer_roster_id else "",
        "numRoundsRookie": int(num_rounds_rookie) if num_rounds_rookie else None,
        "numRoundsStartup": int(num_rounds_startup) if num_rounds_startup else None,
        # League keepers (from the keeper tool) to drop from the board. Omitted /
        # empty for non-keeper leagues, where the draft room behaves exactly as before.
        "keepers": keepers or None,
        # Whether to offer the Keeper draft type at all. False for dynasty and
        # plain redraft leagues, where keepers do not apply; draft_room.js then
        # removes the Keeper option and its fields.
        "showKeeper": bool(show_keeper),
        # hasPremium still gates Draft Deep Dive and custom-board persistence.
        # Live cheat-sheet overlay / sync is free.
        "hasPremium": bool(has_premium),
        # Auction detection (R02.1): snake UX stays default; auction leagues get
        # an honest banner until auction grades/values ship.
        "isAuction": bool(is_auction),
        "auctionBudget": float(auction_budget) if auction_budget is not None else None,
        "isBestBall": bool(is_best_ball),
        "chromeExtensionStoreUrl": (os.environ.get("CHROME_EXTENSION_URL") or "").strip(),
        "chromeExtensionZipUrl": "/static/extension/br-fantasy-espn-connector.zip",
    }
    cfg_json = json.dumps(cfg)
    # cfg is a plain inline script so it runs during parse, before the deferred
    # external draft_room.js reads window.__draftCfg. The page is a full document
    # (render_page), so a deferred external script executes normally.
    return (
            f'<script>window.__draftCfg = {cfg_json};</script>\n'
            + _DRAFT_ROOM_HTML
            # draft_grade_curve.js is intentionally not loaded: live grades are absolute
            # (no field curve). The file remains for backtests + parity tests only.
            + f'\n<script src="/static/pick_score.js?v={_static_v("pick_score.js")}" defer></script>\n'
            + f'\n<script src="/static/draft_board_core.js?v={_static_v("draft_board_core.js")}" defer></script>\n'
            + f'\n<script src="/static/draft_grade_team.js?v={_static_v("draft_grade_team.js")}" defer></script>\n'
            + f'\n<script src="/static/draft_room.js?v={_draft_room_js_v()}" defer></script>\n'
    )


# Plain (non-f) string — safe to contain { } freely.
_DRAFT_ROOM_HTML = r"""
<div class="dr-wrap">
  <div class="dr-hero" id="drHero">
    <h1 class="dr-title">Draft Room</h1>
    <p class="dr-sub">Mock against CPU teams, draft manually, or sync a live Sleeper or ESPN draft with best-available ranks, tiers, and a live grade.</p>
    <div class="dr-hero-actions">
      <a class="dr-hero-link" id="drToCheatSheet" href="/draft/cheat-sheet">Cheat Sheet</a>
      <a class="dr-hero-link" id="drToHistory" href="/draft/history">Draft History</a>
    </div>
    <div class="dr-auction-note" id="drAuctionNote" hidden style="margin-top:12px;padding:10px 12px;border-radius:10px;background:var(--accent-soft,rgba(37,99,235,.08));border:1px solid var(--border);font-size:13px;line-height:1.45;color:var(--text);">
      <strong>Auction league detected.</strong> Recommendation Rank and Pick Score still help nominations. Suggested $ amounts are guidance from BR values, not clearing prices. Snake-round draft grades are disabled for auction.
    </div>
  </div>

  <!-- Setup -->
  <div class="dr-setup" id="drSetup">
    <div class="dr-setup-card" id="drSetupCard">
      <header class="dr-setup-modal-head" id="drSetupModalHead" hidden>
        <div>
          <div class="dr-setup-modal-kicker">Current draft</div>
          <h2 class="dr-setup-modal-title" id="drEditTitle">Edit Setup</h2>
        </div>
        <button type="button" class="dr-setup-modal-close" id="drEditClose" aria-label="Close">&times;</button>
      </header>
      <p class="dr-setup-desc" id="drEditNote" hidden>Changes apply to this draft. Picks stay on the board unless you change teams, pick order, or your slot. Reset wipes the board and returns to setup.</p>

      <div class="dr-step">
        <div class="dr-step-head">
          <span class="dr-step-num">1</span>
          <div class="dr-step-title">Format</div>
        </div>
        <div class="dr-setup-grid">
          <div class="dr-field"><span>Draft Type</span>
            <select id="drType">
              <option value="startup">Startup (Dynasty)</option>
              <option value="rookie">Rookie (Dynasty)</option>
              <option value="redraft">Redraft</option>
              <option value="keeper">Keeper</option>
            </select>
          </div>
          <!-- Keeper-only options; shown when Draft Type is Keeper. A keeper
               draft is a redraft where each kept player costs that team the pick
               at his keeper round, so those picks come off the board up front. -->
          <div class="dr-field dr-keeper-only" style="display:none;"><span>Keepers</span>
            <select id="drKeeperSource">
              <option value="assistant">Use Keeper Assistant</option>
              <option value="manual">Pick my own</option>
            </select>
          </div>
          <div class="dr-field dr-keeper-only" style="display:none;"><span>Keepers / Team</span>
            <input id="drKeeperCount" type="number" min="0" max="10" step="1" value="2">
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
            <select id="drPpr" aria-label="Reception scoring" title="Projected PPG uses this reception scoring (full, half, or standard).">
              <option value="1" selected>Full PPR</option>
              <option value="0.5">Half PPR</option>
              <option value="0">Standard</option>
            </select>
          </div>
          <div class="dr-field"><span>TE Premium</span>
            <select id="drTep" aria-label="Tight end premium" title="Projected PPG for tight ends includes this TE premium.">
              <option value="0" selected>None</option>
              <option value="0.5">+0.5 PPR</option>
              <option value="1">+1.0 PPR</option>
            </select>
          </div>
          <div class="dr-field"><span>Passing TDs</span>
            <select id="drPassTd" aria-label="Points per passing touchdown" title="Adjusts quarterback projected PPG, recommendations, and pick grades">
              <option value="4" selected>4 points</option>
              <option value="6">6 points</option>
            </select>
          </div>
        </div>
      </div>

      <div class="dr-step">
        <div class="dr-step-head">
          <span class="dr-step-num">2</span>
          <div class="dr-step-title">Roster Slots</div>
        </div>
        <div id="drRosterSection"></div>
      </div>

      <div class="dr-step">
        <div class="dr-step-head">
          <span class="dr-step-num">3</span>
          <div class="dr-step-title">League</div>
        </div>
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
          <!-- Which ADP the CPU opponents draft against. Consensus (blended
               Sleeper/BR/ESPN/MFL/Yahoo) is the default and most platform-neutral;
               pick a single platform to mock a board that drafts like that site.
               This is a simulation rule, independent of the in-draft "ADP source"
               display selector. A source with no data for the chosen format falls
               back to consensus, so no pick is ever left without an ADP. -->
          <div class="dr-field"><span>CPU drafts from</span>
            <select id="drCpuAdpSource" title="Which ADP source the CPU opponents draft against. Consensus blends every platform. Live (7d) is recent BR Fantasy drafts only.">
              <option value="consensus" selected>Consensus (all platforms)</option>
              <option value="sleeper">Sleeper</option>
              <option value="brfantasy">BR Fantasy</option>
              <option value="brfantasy_live">BR Fantasy Live (7d)</option>
              <option value="espn">ESPN</option>
              <option value="mfl">MFL</option>
              <option value="yahoo">Yahoo</option>
            </select>
          </div>
        </div>
      </div>

      <div class="dr-step">
        <div class="dr-step-head">
          <span class="dr-step-num">4</span>
          <div class="dr-step-title">Draft Capital</div>
        </div>
        <p class="dr-setup-desc" style="margin-bottom:8px;">Defaults to your slot's picks. Tap + on a round to add a traded-in pick, or click a pick to remove one you traded away.</p>
        <div id="drCapitalSection"></div>
      </div>

      <div class="dr-setup-cta" id="drSetupStartCta">
        <button class="dr-btn dr-btn-primary dr-btn-lg" id="drStartSim">&#9654;&nbsp; Start Mock Draft</button>
        <button class="dr-btn dr-btn-lg" id="drStart">Draft Manually</button>
        <button class="dr-btn dr-btn-ghost" id="drConnect">Connect Live Draft</button>
      </div>
      <div class="dr-setup-cta dr-setup-edit-cta" id="drSetupEditCta" hidden>
        <button type="button" class="dr-btn dr-btn-ghost dr-btn-danger" id="drEditReset"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>Reset Draft</button>
        <span class="dr-setup-edit-spacer"></span>
        <button type="button" class="dr-btn dr-btn-ghost" id="drEditCancel">Cancel</button>
        <button type="button" class="dr-btn dr-btn-primary" id="drEditApply">Apply Settings</button>
      </div>
      <div class="dr-live-list" id="drLiveList" style="display:none;"></div>
    </div>
  </div>

  <!-- Board + side -->
  <div class="dr-main" id="drMain" style="display:none;">
    <div class="dr-start-banner" id="drStartBanner" style="display:none;"></div>
    <div class="dr-start-banner dr-espn-fallback" id="drEspnFallback" style="display:none;" hidden></div>
    <div class="dr-espn-tools" id="drEspnTools" style="display:none;" hidden></div>
    <div class="dr-statusbar">
      <div class="dr-status-info">
        <div class="dr-onclock" id="drOnClockWrap">
          <span class="dr-onclock-label">On the clock</span>
          <b id="drOnClock">Team 1</b>
        </div>
        <div class="dr-status-pills">
          <span class="dr-ss-stat" id="drPickPill">Pick: 1.01</span>
          <button type="button" class="dr-league-meta" id="drLeagueMeta" hidden></button>
          <span class="dr-pick-timer" id="drPickTimer" style="display:none;"></span>
          <span class="dr-pill dr-pill-live" id="drLiveBadge" style="display:none;">&#9679; LIVE</span>
          <span class="dr-pill dr-pill-upcoming" id="drUpcomingBadge" style="display:none;">Upcoming</span>
          <span class="dr-pill dr-pill-espn" id="drEspnSync" style="display:none;" hidden>ESPN Draft</span>
          <button type="button" class="dr-pill-reconnect" id="drEspnReconnect" style="display:none;" hidden title="Reestablish extension sync">↻ Reconnect</button>
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
          <button class="otc-main-tab" data-stab="queue">Queue</button>
          <button class="otc-main-tab" data-stab="needs">Team</button>
          <button class="otc-main-tab" data-stab="league">League</button>
          <div class="dr-side-opts">
            <button class="dr-opts-trigger dr-undo-trigger" id="drUndo" aria-label="Undo last pick" title="Undo last pick"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg></button>
            <a class="dr-opts-trigger dr-cs-trigger" id="drOptsCheatSheet" href="/draft/cheat-sheet" rel="noopener" title="Open your value board / cheat sheet (Cmd/Ctrl-click for a new tab)" aria-label="Cheat Sheet"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="8" y="2" width="8" height="4" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M9 12h6M9 16h4"/></svg><span class="dr-cs-trigger-lbl">Cheat</span></a>
            <button class="dr-opts-trigger dr-pt-trigger" id="drPickTradeBtn" aria-label="Pick trade evaluator" title="Pick trade evaluator">Trade</button>
            <button class="dr-opts-trigger" id="drOptsBtn" aria-label="Settings" title="Settings"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="vertical-align:-2px;"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg></button>
            <div class="dr-opts-panel" id="drOptsPanel">
              <!-- Auto-draft settings: mocks only, collapsed by default so the
                   menu is not cluttered by three selectors most people set once. -->
              <div class="dr-opts-auto" id="drAutoSettings" style="display:none;">
                <button type="button" class="dr-opts-expander" id="drAutoSettingsToggle" aria-expanded="false" aria-controls="drAutoSettingsBody">
                  <span>Auto-draft settings</span>
                  <svg class="dr-opts-chev" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="6 9 12 15 18 9"/></svg>
                </button>
                <div class="dr-opts-auto-body" id="drAutoSettingsBody" hidden>
                  <select class="dr-sim-speed" id="drSimSpeed" title="Simulation speed">
                    <option value="1400">Speed: Slow</option>
                    <option value="700" selected>Speed: Normal</option>
                    <option value="300">Speed: Fast</option>
                    <option value="60">Speed: Instant</option>
                  </select>
                  <select class="dr-sim-speed" id="drMyStrat" title="Strategy your auto-draft follows on your picks">
                    <option value="">Auto: Balanced</option>
                    <option value="rb_heavy">Auto: RB heavy</option>
                    <option value="wr_heavy">Auto: WR heavy</option>
                    <option value="zero_rb">Auto: Zero RB</option>
                    <option value="hero_rb">Auto: Hero RB</option>
                    <option value="elite_te">Auto: Elite TE</option>
                    <option value="early_qb">Auto: Early QB</option>
                  </select>
                  <select class="dr-sim-speed" id="drMyAgeLean" title="Age lean your auto-draft follows on your picks">
                    <option value="">Age: Neutral</option>
                    <option value="win_now">Age: Win now</option>
                    <option value="youth">Age: Youth</option>
                  </select>
                </div>
              </div>
              <div class="dr-opts-sec">
                <button class="dr-btn dr-btn-ghost" id="drSummaryBtn" style="display:none;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>Summary</button>
                <button class="dr-btn dr-btn-ghost" id="drShare"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>Share</button>
                <button class="dr-btn dr-btn-ghost" id="drEdit"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>Edit Setup</button>
                <button class="dr-btn dr-btn-ghost dr-btn-danger" id="drReset"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>Reset</button>
              </div>
            </div>
          </div>
        </div>
        <div class="dr-side-head" id="drBestControls">
          <div class="dr-side-controls">
            <!-- Sort control: a custom dropdown (the native <select> popup
                 mis-anchors inside the transformed mobile sheet). data-val holds
                 the current sort; renderBA reads it. -->
            <div class="dr-sortsel" id="drBaSortUI">
              <button type="button" class="dr-sortsel-btn" id="drBaSortBtn" data-val="ps" aria-haspopup="listbox" aria-expanded="false">
                <span id="drBaSortLbl">Recommendation Rank</span>
                <svg class="dr-sortsel-caret" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>
              </button>
              <div class="dr-sortsel-menu" id="drBaSortMenu" role="listbox" hidden>
                <button type="button" class="dr-sortsel-opt" role="option" data-val="ps">Recommendation Rank</button>
                <button type="button" class="dr-sortsel-opt" role="option" data-val="pickscore">Pick Score</button>
                <button type="button" class="dr-sortsel-opt" role="option" data-val="value">Value</button>
                <button type="button" class="dr-sortsel-opt" role="option" data-val="ppg">Proj PPG</button>
                <button type="button" class="dr-sortsel-opt" role="option" data-val="adp">ADP</button>
              </div>
            </div>
            <input id="drSearch" type="search" placeholder="Search…" autocomplete="off">
            <button class="dr-help-btn" id="drHelpBtn" type="button" aria-label="What do these terms mean?" title="What do these terms mean?">?</button>
          </div>
          <div class="otc-day-filters dr-pos-filters" id="drPosFilters">
            <button class="otc-day-filter dr-pos active" data-pos="ALL">All</button>
            <button class="otc-day-filter dr-pos" data-pos="QB">QB</button>
            <button class="otc-day-filter dr-pos" data-pos="RB">RB</button>
            <button class="otc-day-filter dr-pos" data-pos="WR">WR</button>
            <button class="otc-day-filter dr-pos" data-pos="TE">TE</button>
            <button class="otc-day-filter dr-pos dr-pos-kdef" data-pos="K" style="display:none;">K</button>
            <button class="otc-day-filter dr-pos dr-pos-kdef" data-pos="DEF" style="display:none;">DEF</button>
          </div>
          <div class="dr-adp-src" id="drAdpSrc"></div>
        </div>
        <div id="drBestChips" style="display:none;"></div>
        <div class="dr-ba-list" id="drBaList">
          <div class="sk-list" aria-hidden="true">
            <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-60"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
            <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-80"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
            <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-60"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
            <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-80"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
            <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-60"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
            <div class="sk-card-row"><div class="skeleton sk-av"></div><div class="sk-lines"><div class="skeleton skeleton-line w-80"></div><div class="skeleton skeleton-line w-40"></div></div><div class="skeleton sk-chip"></div></div>
          </div>
        </div>
        <div id="drCompleteBar" style="display:none;">
          <button class="dr-btn dr-btn-primary" id="drCompleteSummaryBtn" style="width:100%;">Draft Summary</button>
          <button class="dr-btn dr-btn-deepdive" id="drCompleteDeepDiveBtn" style="width:100%;"><svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="vertical-align:-2px;margin-right:5px;"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/></svg>Deep Dive<span class="dr-dd-prochip">PRO</span></button>
          <button class="dr-btn" id="drCompleteShareBtn" style="width:100%;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="vertical-align:-1px;margin-right:4px;"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>Share</button>
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

  <!-- Deep Dive analyzer (Pro) -->
  <div class="dr-dd-overlay" id="drDeepDive" style="display:none;">
    <div class="dr-dd-card" id="drDeepDiveCard"></div>
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
        <button class="dr-btn dr-btn-primary" id="drShareViewShare"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" style="vertical-align:-1px;margin-right:4px;"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg>Share</button>
        <button class="dr-btn" id="drShareViewDl">Download</button>
      </div>
    </div>
  </div>

  <!-- In-draft cheat sheet (chrome-less iframe embed) -->
  <div class="dr-cheat-overlay" id="drCheatSheet" role="dialog" aria-modal="true" aria-labelledby="drCheatTitle" style="display:none;">
    <div class="dr-cheat-card">
      <div class="dr-cheat-head">
        <span class="dr-cheat-title" id="drCheatTitle">Cheat Sheet</span>
        <a class="dr-cheat-pop" id="drCheatPop" href="/draft/cheat-sheet" target="_blank" rel="noopener" title="Open in a new tab">Open in tab &#8599;</a>
        <button class="dr-cheat-close" id="drCheatClose" aria-label="Close">&times;</button>
      </div>
      <iframe class="dr-cheat-frame" id="drCheatFrame" title="Draft cheat sheet"></iframe>
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
  .dr-wrap {
    max-width: 1640px; margin: 0 auto; padding: 14px 14px 48px;
  }
  .dr-hero { margin: 2px 0 22px; text-align: center; }

  .dr-title {
    font-size: clamp(28px, 4.4vw, 40px); font-weight: 800; color: var(--text);
    margin: 0 0 8px; letter-spacing: -0.03em; line-height: 1.1;
  }
  .dr-sub {
    font-size: 15px; color: var(--text-muted); margin: 0 auto; max-width: 540px; line-height: 1.55;
  }
  .dr-hero-actions {
    display: inline-flex; flex-wrap: wrap; gap: 8px; justify-content: center; margin-top: 14px;
  }
  .dr-hero-link {
    display: inline-flex; align-items: center; padding: 7px 12px; font-size: 12px; font-weight: 700;
    color: var(--text-muted); text-decoration: none; border: 1px solid var(--border);
    border-radius: var(--radius-pill, 8px); background: color-mix(in srgb, var(--card) 80%, transparent);
    transition: color .15s, border-color .15s, background .15s;
  }
  .dr-hero-link:hover {
    color: var(--brand-blue, #3b82f6); border-color: color-mix(in srgb, var(--brand-blue, #3b82f6) 45%, var(--border));
    background: color-mix(in srgb, var(--brand-blue, #3b82f6) 8%, transparent); text-decoration: none;
  }
  /* ── Setup (redesigned) ── */
  .dr-setup { display: flex; justify-content: center; padding: 0 0 8px; }
  .dr-setup-card {
    position: relative; width: 100%; max-width: 740px; border: 1px solid var(--border);
    border-radius: 18px; padding: 24px 26px; box-shadow: var(--shadow, 0 8px 30px rgba(0,0,0,.10));
    background:
      linear-gradient(180deg, color-mix(in srgb, var(--brand-blue, #3b82f6) 5%, var(--card)) 0%, var(--card) 88px),
      var(--card);
  }
  .dr-setup-desc { font-size: 13px; color: var(--text-muted); margin: 0; line-height: 1.5; }
  #drEditNote { margin-bottom: 12px; }
  .dr-step { padding: 22px 0; border-top: 1px solid var(--border); }
  .dr-setup-card > .dr-step:first-of-type { border-top: none; padding-top: 0; }
  .dr-setup-is-modal .dr-setup-card > .dr-step:first-of-type { border-top: 1px solid var(--border); padding-top: 22px; }
  .dr-step-head { display: flex; align-items: center; gap: 10px; margin-bottom: 14px; }
  .dr-step-num {
    width: 26px; height: 26px; border-radius: 8px; display: inline-flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 900; color: var(--on-accent, #fff);
    background: var(--accent, #122d4b); flex-shrink: 0;
  }
  .dr-step-title { font-size: 20px; font-weight: 800; color: var(--text); margin: 0; line-height: 1.15; letter-spacing: -0.02em; }
  .dr-setup-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap: 12px; }
  .dr-field { display: flex; flex-direction: column; gap: 6px; font-size: 12px; font-weight: 700; color: var(--text-muted); }
  .dr-field select, .dr-field input {
    padding: 9px 11px; border-radius: 9px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); font-size: 14px; font-weight: 600; outline: none; min-height: 40px;
  }
  .dr-field select:focus, .dr-field input:focus {
    border-color: var(--brand-blue, #3b82f6);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--brand-blue, #3b82f6) 16%, transparent);
  }
  .dr-setup-cta { margin-top: 20px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .dr-setup-edit-cta { margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border); }
  .dr-setup-edit-cta .dr-btn { display: inline-flex; align-items: center; gap: 7px; }
  .dr-setup-edit-spacer { flex: 1; min-width: 8px; }
  /* Author display:flex rules beat the UA [hidden] stylesheet; force collapse. */
  #drSetup [hidden], .dr-league-meta[hidden] { display: none !important; }
  .dr-setup-modal-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; margin-bottom: 14px; }
  .dr-setup-modal-kicker {
    font-size: 10px; font-weight: 900; text-transform: uppercase; letter-spacing: .12em;
    color: var(--brand-blue, #3b82f6); margin-bottom: 4px;
  }
  .dr-setup-modal-title { font-size: 22px; font-weight: 900; color: var(--text); margin: 0; line-height: 1.1; }
  .dr-setup-modal-close {
    width: 28px; height: 28px; flex-shrink: 0; background: var(--bg); border: 1px solid var(--border);
    border-radius: 12px; font-size: 17px; line-height: 1; color: var(--text-muted); cursor: pointer;
    display: flex; align-items: center; justify-content: center;
  }
  .dr-setup-modal-close:hover { background: color-mix(in srgb, var(--loss) 12%, transparent); color: var(--loss); }
  .dr-setup-is-modal {
    display: flex !important; position: fixed; inset: 0; z-index: 1100;
    background: rgba(0,0,0,.58); align-items: flex-start; justify-content: center;
    overflow-y: auto; padding: calc(env(safe-area-inset-top) + 16px) 16px calc(env(safe-area-inset-bottom) + 20px);
  }
  .dr-setup-is-modal .dr-setup-card {
    margin: 8px auto; max-width: 740px; width: 100%;
    box-shadow: 0 24px 80px rgba(0,0,0,.45);
  }
  .dr-setup-is-modal .dr-live-list { display: none !important; }
  body.dr-edit-open { overflow: hidden; }
  .dr-league-meta {
    display: inline-flex; align-items: center; gap: 5px; flex-wrap: nowrap;
    min-width: 0; padding: 3px 6px; border-radius: 8px;
    border: 1px solid transparent; background: transparent; color: var(--text-muted);
    font-family: inherit; font-size: 12px; font-weight: 700; line-height: 1.3; white-space: nowrap;
    cursor: default; flex-shrink: 0; appearance: none;
  }
  .dr-league-meta.is-editable { cursor: pointer; }
  .dr-league-meta.is-editable:hover {
    color: var(--text); border-color: var(--border); background: var(--bg);
  }
  .dr-lm-chip {
    display: inline-flex; align-items: center; padding: 1px 7px; border-radius: 6px;
    background: var(--row, var(--bg)); border: 1px solid var(--grid, var(--border));
    color: var(--text-muted); font-size: 11px; font-weight: 700; line-height: 1.45;
    white-space: nowrap; flex-shrink: 0;
  }
  .dr-btn-lg { padding: 12px 22px; font-size: 14px; border-radius: 10px; }
  .dr-sim-speed { padding: 6px 8px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg);
    color: var(--text); font-size: 12px; font-weight: 600; }
  .dr-btn {
    padding: 9px 16px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer;
    border: 1px solid var(--border); background: var(--bg); color: var(--text); white-space: nowrap;
    transition: background .15s, border-color .15s, color .15s, box-shadow .15s, transform .15s;
  }
  .dr-btn:hover { border-color: color-mix(in srgb, var(--accent) 45%, var(--border)); }
  .dr-btn-primary {
    background: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8); color: var(--on-accent, #fff);
  }
  .dr-btn-primary:hover {
    box-shadow: 0 6px 16px color-mix(in srgb, var(--accent) 28%, transparent);
    transform: translateY(-1px);
  }
  .dr-btn-ghost { background: transparent; font-weight: 600; }
  /* Settings gear button — sits beside the side-panel tabs — + dropdown panel */
  .dr-side-opts { position: relative; flex: 0 0 auto; display: flex; align-items: stretch; }
  .dr-opts-trigger { display: flex; align-items: center; justify-content: center; gap: 5px;
    background: transparent; border: none; cursor: pointer; color: var(--text-muted);
    font-size: 14px; padding: 0 9px; border-radius: 8px; text-decoration: none; }
  a.dr-opts-trigger { color: var(--text-muted); }
  .dr-cs-trigger { font-size: 12px; font-weight: 700; white-space: nowrap; }
  .dr-cs-trigger-lbl { line-height: 1; }
  .dr-opts-trigger:hover, .dr-opts-trigger[aria-expanded="true"] {
    color: var(--accent,#38bdf8); background: color-mix(in srgb, var(--accent) 12%, transparent); }
  .dr-opts-panel {
    display: none; flex-direction: column; gap: 2px;
    position: absolute; top: calc(100% + 6px); right: 0;
    background: var(--card, #1a1a1a); border: 1px solid var(--border, #333); border-radius: 12px;
    padding: 6px; z-index: 200; min-width: 155px;
    box-shadow: 0 8px 32px rgba(0,0,0,.3);
  }
  .dr-opts-panel .dr-btn { width: 100%; display: flex; align-items: center; gap: 7px; text-align: left; padding: 9px 14px; border-radius: 8px; font-size: 13px;
    background: var(--bg, #0f0f0f); color: var(--text, #fff); border: 1px solid var(--border, #333); }
  .dr-opts-panel .dr-btn svg { flex-shrink: 0; opacity: .8; }
  .dr-opts-panel .dr-sim-speed { width: 100%; margin: 0; padding: 6px 8px; border-radius: 8px;
    border: 1px solid var(--border, #333); background: var(--bg, #0f0f0f); color: var(--text, #fff); font-size: 13px; }
  /* Grouped sections: a labelled block per kind, with hairline dividers between. */
  .dr-opts-sec { display: flex; flex-direction: column; gap: 2px; }
  .dr-opts-sec + .dr-opts-sec { margin-top: 6px; padding-top: 6px; border-top: 1px solid var(--border, #333); }
  .dr-opts-label { font-size: 10px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    color: var(--text-muted); padding: 2px 14px 3px; }
  /* Auto-draft settings: collapsible, with its own divider below when shown. */
  .dr-opts-auto { display: flex; flex-direction: column; gap: 4px;
    margin-bottom: 6px; padding-bottom: 6px; border-bottom: 1px solid var(--border, #333); }
  .dr-opts-expander { width: 100%; display: flex; align-items: center; justify-content: space-between; gap: 8px;
    text-align: left; padding: 9px 14px; border-radius: 8px; font-size: 13px; font-weight: 600; cursor: pointer;
    background: var(--bg, #0f0f0f); color: var(--text, #fff); border: 1px solid var(--border, #333); }
  .dr-opts-expander:hover { border-color: var(--accent, #38bdf8); color: var(--accent, #38bdf8); }
  .dr-opts-chev { flex-shrink: 0; opacity: .75; transition: transform .15s ease; }
  .dr-opts-expander[aria-expanded="true"] .dr-opts-chev { transform: rotate(180deg); }
  .dr-opts-auto-body { display: flex; flex-direction: column; gap: 4px; padding-top: 4px; }
  .dr-opts-auto-body[hidden] { display: none; }
  .dr-btn-danger { color: var(--loss); border-color: color-mix(in srgb, var(--loss) 40%, transparent); }
  .dr-sim-error {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    margin-bottom: 12px; padding: 10px 14px; border-radius: 10px;
    border: 1px solid rgba(239, 68, 68, .45); background: rgba(239, 68, 68, .12);
    color: var(--text); font-size: 13px; line-height: 1.4;
  }
  .dr-sim-error b { color: #ef4444; }
  .dr-sim-error-x {
    margin-left: auto; background: none; border: none; cursor: pointer;
    color: var(--text-muted); font-size: 20px; line-height: 1; padding: 0 4px;
  }
  .dr-statusbar {
    position: relative;
    display: flex; align-items: center; justify-content: space-between; gap: 12px;
    padding: 10px 14px; margin-bottom: 12px; border: 1px solid var(--border); border-radius: 14px;
    background:
      linear-gradient(180deg, color-mix(in srgb, var(--brand-blue, #3b82f6) 4%, var(--card)), var(--card));
    box-shadow: var(--shadow-sm, 0 2px 8px rgba(15, 23, 42, 0.05));
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
  .dr-onclock.dr-onclock-you { background: color-mix(in srgb, var(--win) 10%, transparent); border-color: color-mix(in srgb, var(--win) 40%, transparent); }
  .dr-onclock.dr-onclock-you b { color: var(--win); }
  .dr-pill { display:inline-flex; align-items:center; font-size:12px; font-weight:700; padding:3px 9px;
    border-radius:var(--radius-pill, 8px); background:color-mix(in srgb, var(--accent) 14%, transparent);
    color:var(--accent,#38bdf8); white-space:nowrap;
    border:1px solid color-mix(in srgb, currentColor 35%, transparent); }
  .dr-pill-you { background: color-mix(in srgb, var(--win) 16%, transparent); color: var(--win); }
  .dr-pill-live { background: color-mix(in srgb, var(--loss) 16%, transparent); color: var(--loss); animation: drPulse 1.6s ease-in-out infinite; }
  .dr-pill-upcoming { background: color-mix(in srgb, var(--warning) 16%, transparent); color: var(--warning); }
  .dr-pill-paused   { background: rgba(148,163,184,.16); color: var(--text-subtle); }
  .dr-pill-espn { background: color-mix(in srgb, var(--accent,#38bdf8) 14%, transparent); color: var(--accent,#38bdf8); font-variant-numeric: tabular-nums; }
  .dr-pill-espn.is-live { background: color-mix(in srgb, var(--loss) 16%, transparent); color: var(--loss); animation: drPulse 1.6s ease-in-out infinite; }
  .dr-pill-espn.is-ok { background: color-mix(in srgb, var(--win) 16%, transparent); color: var(--win); }
  .dr-pill-espn.is-warn { background: color-mix(in srgb, var(--warning) 16%, transparent); color: var(--warning); }
  .dr-pill-espn.is-muted { background: rgba(148,163,184,.16); color: var(--text-subtle); animation: none; }
  .dr-pill-reconnect {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 4px 10px; border-radius: 999px; border: 1px solid color-mix(in srgb, var(--accent,#38bdf8) 35%, transparent);
    background: color-mix(in srgb, var(--accent,#38bdf8) 10%, transparent);
    color: var(--accent,#38bdf8); font: 700 11px/1.2 inherit; cursor: pointer;
  }
  .dr-pill-reconnect:hover { background: color-mix(in srgb, var(--accent,#38bdf8) 18%, transparent); }
  .dr-pill-reconnect.is-busy { opacity: .65; cursor: wait; }
  .dr-espn-fallback { background: linear-gradient(90deg, color-mix(in srgb, var(--warning) 16%, transparent), color-mix(in srgb, var(--warning) 5%, transparent)); border-color: var(--warning); }
  .dr-espn-fallback .dr-banner-join { background: var(--warning); color: #111; cursor: pointer; border: 0; font: inherit; }
  /* ESPN/Yahoo sync helpers — compact promo strip above the status bar */
  .dr-espn-tools {
    position: relative;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 14px 18px;
    margin: 0 0 12px;
    padding: 13px 44px 13px 14px;
    border-radius: 14px;
    border: 1px solid color-mix(in srgb, var(--accent,#38bdf8) 24%, var(--border));
    background:
      linear-gradient(135deg,
        color-mix(in srgb, var(--accent,#38bdf8) 10%, var(--card, var(--bg))),
        color-mix(in srgb, var(--accent,#38bdf8) 3%, var(--card, var(--bg))));
    box-shadow: var(--shadow-sm, 0 2px 8px rgba(15, 23, 42, 0.05));
  }
  .dr-espn-tools.is-unavailable {
    border-color: color-mix(in srgb, var(--warning) 38%, var(--border));
    background:
      linear-gradient(135deg,
        color-mix(in srgb, var(--warning) 12%, var(--card, var(--bg))),
        color-mix(in srgb, var(--warning) 4%, var(--card, var(--bg))));
  }
  .dr-espn-tools-body {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    flex: 1 1 240px;
    min-width: 0;
  }
  .dr-espn-tools-ic {
    width: 40px; height: 40px; border-radius: 11px; flex-shrink: 0;
    display: inline-flex; align-items: center; justify-content: center;
    background: linear-gradient(145deg,
      color-mix(in srgb, var(--accent,#38bdf8) 22%, transparent),
      color-mix(in srgb, var(--accent,#38bdf8) 8%, transparent));
    border: 1px solid color-mix(in srgb, var(--accent,#38bdf8) 28%, transparent);
    color: var(--accent,#38bdf8); font-size: 16px;
    box-shadow: 0 2px 10px color-mix(in srgb, var(--accent,#38bdf8) 14%, transparent);
  }
  .dr-espn-tools.is-unavailable .dr-espn-tools-ic {
    background: linear-gradient(145deg,
      color-mix(in srgb, var(--warning) 24%, transparent),
      color-mix(in srgb, var(--warning) 10%, transparent));
    border-color: color-mix(in srgb, var(--warning) 32%, transparent);
    color: var(--warning);
    box-shadow: 0 2px 10px color-mix(in srgb, var(--warning) 12%, transparent);
  }
  .dr-espn-tools-copy { min-width: 0; flex: 1; padding-top: 1px; }
  .dr-espn-tools-kicker {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: .06em;
    color: var(--accent,#38bdf8); margin: 0 0 4px;
  }
  .dr-espn-tools.is-unavailable .dr-espn-tools-kicker { color: var(--warning); }
  .dr-espn-tools-kicker-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor; opacity: .85;
  }
  .dr-espn-tools-copy b {
    display: block; font-size: 15px; font-weight: 800; letter-spacing: -0.02em;
    color: var(--text); line-height: 1.25; margin: 0 0 4px;
  }
  .dr-espn-tools-copy span {
    display: block; font-size: 12.5px; color: var(--text-muted); line-height: 1.45;
    max-width: 52ch;
  }
  .dr-espn-tools-x {
    position: absolute; top: 10px; right: 10px;
    width: 28px; height: 28px; margin: 0;
    border: 0; border-radius: 8px; background: transparent; color: var(--text-muted);
    cursor: pointer; font-size: 18px; line-height: 1;
    display: inline-flex; align-items: center; justify-content: center;
  }
  .dr-espn-tools-x:hover { background: rgba(127,127,127,.12); color: var(--text); }
  .dr-espn-tools-actions {
    display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
    flex: 1 1 100%; width: 100%;
  }
  .dr-espn-tools-actions.is-split { /* unavailable: same row layout */ }
  .dr-espn-tools-actions .dr-banner-join {
    margin: 0; width: auto; justify-content: center;
    border: 0; cursor: pointer; font: inherit; text-decoration: none;
    padding: 9px 14px; border-radius: 10px; font-size: 13px; font-weight: 700;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.06);
  }
  .dr-espn-tools-actions .dr-banner-join:not(.is-ghost):not(.is-link) {
    background: var(--accent,#38bdf8);
    color: var(--on-accent, #fff);
  }
  .dr-espn-tools-actions .dr-banner-join:not(.is-ghost):not(.is-link):hover {
    filter: brightness(1.05);
  }
  .dr-espn-tools-actions .dr-banner-join.is-ghost {
    background: var(--card, var(--bg)); color: var(--text);
    border: 1px solid color-mix(in srgb, var(--accent,#38bdf8) 28%, var(--border));
    box-shadow: none;
  }
  .dr-espn-tools-actions .dr-banner-join.is-ghost:hover {
    border-color: color-mix(in srgb, var(--accent,#38bdf8) 45%, var(--border));
    background: color-mix(in srgb, var(--accent,#38bdf8) 6%, var(--card, var(--bg)));
  }
  .dr-espn-tools.is-unavailable .dr-espn-tools-actions .dr-banner-join.is-ghost {
    border-color: color-mix(in srgb, var(--warning) 36%, var(--border));
  }
  .dr-espn-tools-actions .dr-banner-join.is-link {
    background: transparent; color: var(--accent,#38bdf8);
    border: 0; padding: 9px 10px; font-weight: 650; font-size: 12.5px;
    box-shadow: none; margin-left: auto;
  }
  .dr-espn-tools-actions .dr-banner-join.is-link:hover {
    text-decoration: underline;
    filter: brightness(1.08);
  }
  .dr-espn-tools-actions .dr-banner-join.is-link i { opacity: .75; font-size: 10px; }
  @media (min-width: 720px) {
    .dr-espn-tools { flex-wrap: nowrap; padding: 12px 44px 12px 14px; }
    .dr-espn-tools-body { flex: 1 1 auto; }
    .dr-espn-tools-actions {
      flex: 0 0 auto; width: auto; justify-content: flex-end;
      padding-left: 12px;
      border-left: 1px solid color-mix(in srgb, var(--border) 80%, transparent);
    }
    .dr-espn-tools-actions .dr-banner-join.is-link {
      margin-left: 0;
      padding-left: 14px;
      border-left: 1px solid color-mix(in srgb, var(--border) 80%, transparent);
    }
  }
  @media (max-width: 719px) {
    .dr-espn-tools-actions .dr-banner-join { flex: 1 1 calc(50% - 4px); min-width: 0; }
    .dr-espn-tools-actions .dr-banner-join.is-link {
      flex: 1 1 100%; justify-content: center; margin-left: 0;
      padding-top: 4px;
    }
  }
  .dr-pick-timer { font-size: 14px; font-weight: 800; color: var(--text); font-variant-numeric: tabular-nums;
    min-width: 40px; padding: 2px 8px; border-radius: 7px; background: rgba(127,127,127,.1); text-align: center; }
  .dr-pick-timer.urgent { color: #fff; background: var(--loss); animation: drPulse 1s ease-in-out infinite; }
  .dr-progress { font-size: 12px; color: var(--text-muted); white-space: nowrap; }
  .dr-save { font-size: 11px; color: var(--win); }
  .dr-start-banner { display: flex; align-items: center; gap: 13px; margin: 0 0 12px; padding: 12px 16px; border-radius: 12px;
    background: linear-gradient(90deg, color-mix(in srgb, var(--accent) 18%, transparent), color-mix(in srgb, var(--accent) 5%, transparent)); border: 1px solid var(--accent,#38bdf8); }
  .dr-start-banner.is-live { background: linear-gradient(90deg, color-mix(in srgb, var(--win) 18%, transparent), color-mix(in srgb, var(--win) 5%, transparent)); border-color: var(--win); }
  .dr-banner-ic { font-size: 22px; flex-shrink: 0; display: inline-flex; align-items: center; }
  .dr-banner-ic-live { animation: drPulse 1.4s ease-in-out infinite; }
  .dr-banner-txt { display: flex; flex-direction: column; line-height: 1.35; min-width: 0; flex: 1; }
  .dr-banner-txt b { font-size: 15px; font-weight: 800; color: var(--text); }
  .dr-banner-txt span { font-size: 12px; color: var(--text-muted); }
  .dr-start-cd { font-variant-numeric: tabular-nums; }
  .dr-banner-join { flex-shrink: 0; margin-left: auto; display: inline-flex; align-items: center; gap: 7px; white-space: nowrap;
    background: var(--accent,#38bdf8); color: var(--on-accent, #fff); font-weight: 700; font-size: 13px; text-decoration: none; padding: 8px 14px; border-radius: 8px; }
  .dr-start-banner.is-live .dr-banner-join { background: var(--win); }
  .dr-banner-join i { font-size: 11px; }
  .dr-poll-status { font-size: 11px; color: var(--text-muted); display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
  .dr-poll-status .dr-poll-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--win); flex-shrink: 0; }
  .dr-poll-status.is-syncing .dr-poll-dot { background: var(--accent,#38bdf8); animation: drPulse 1s ease-in-out infinite; }
  /* Bottom-sheet drag handle (mobile only) */
  .dr-sheet-handle { display: none; }
  .dr-live-list { margin-top: 12px; display: flex; flex-direction: column; gap: 6px; }
  .dr-live-head { font-size: 12px; font-weight: 700; color: var(--text-muted); }
  .dr-live-item { text-align: left; padding: 9px 12px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); font-size: 13px; cursor: pointer; }
  .dr-live-item:hover { border-color: var(--accent,#38bdf8); }
  .dr-live-status { font-size: 10px; font-weight: 800; text-transform: uppercase; padding: 1px 6px; border-radius: var(--radius-pill, 8px); margin-right: 6px; }
  .dr-ls-drafting { background: color-mix(in srgb, var(--loss) 16%, transparent); color: var(--loss); }
  .dr-ls-pre_draft { background: color-mix(in srgb, var(--warning) 16%, transparent); color: var(--warning); }
  .dr-ls-complete { background: rgba(148,163,184,.16); color: var(--text-subtle); }
  .dr-cols { display: grid; grid-template-columns: 1fr 375px; gap: 14px; align-items: start; }
  /* min-width:0 lets this grid item shrink to its track instead of growing to
     the wide board's width (the inner scroll, not the card, holds the overflow). */
  .dr-board-wrap { position: relative; min-width: 0; border: 1px solid var(--border); border-radius: 14px; background: var(--card); padding: 8px; box-shadow: var(--shadow-sm, 0 2px 8px rgba(15, 23, 42, 0.05)); }
  /* Only the board scrolls horizontally; the toolbar (Value/Pick Score toggle)
     stays pinned to the card so it doesn't drift when you scroll the grid. */
  .dr-board-scroll { overflow-x: auto; min-width: 0; }
  .dr-board { display: grid; gap: 5px; min-width: max-content; }
  .dr-cell {
    border: 1px solid var(--border); border-radius: 8px; padding: 5px 6px 0; min-height: 50px;
    background: var(--bg); display: flex; align-items: flex-end; gap: 6px; position: relative; overflow: hidden;
  }
  .dr-cell-body { padding: 5px; }
  /* Empty slot: reads as an open board cell with its round.pick centered, rather
     than a washed-out box. */
  .dr-cell-empty { background: var(--card); border-style: dashed; }
  .dr-cell-rp { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center;
    justify-content: center; line-height: 1.05; font-size: 11px; font-weight: 700; color: var(--text-muted);
    font-variant-numeric: tabular-nums; letter-spacing: .01em; }
  .dr-cell-rp-ov { font-size: 8px; font-weight: 600; opacity: .55; margin-top: 1px; }
  /* Filled pick: tint the whole cell by its POSITION colour (--pos, set per-cell)
     with a matching left stripe, so a column reads as a roster shape at a glance.
     The ownership (.dr-cell-mine) and current-pick rules below still win their stripe/ring. */
  .dr-cell-filled { background: color-mix(in srgb, var(--pos, var(--accent)) 14%, var(--bg));
    box-shadow: inset 3px 0 0 var(--pos, var(--accent)); }
  .dr-cell-current { box-shadow: inset 0 0 0 2px var(--accent,#38bdf8); animation: drPulse 1.6s ease-in-out infinite; }
  @keyframes drPulse { 0%,100% { box-shadow: inset 0 0 0 2px var(--accent,#38bdf8); } 50% { box-shadow: inset 0 0 0 2px var(--accent,#38bdf8), 0 0 10px color-mix(in srgb, var(--accent) 20%, transparent); } }
  .dr-cell-mine { box-shadow: inset 3px 0 0 var(--accent,#38bdf8); opacity: 1; }
  .dr-cell-mine.dr-cell-empty { opacity: 1; background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 10%, transparent), var(--bg)); }
  .dr-cell-claimed { box-shadow: inset 3px 0 0 var(--warning); }     /* traded-in pick */
  .dr-cell-claimable { cursor: pointer; }
  .dr-cell-claimable:hover { outline: 1px dashed var(--accent,#38bdf8); outline-offset: -2px; }
  .dr-cell-mineflag { position: absolute; top: 2px; right: 5px; font-size: 8px; font-weight: 800;
    letter-spacing: .04em; color: var(--accent,#38bdf8); }
  .dr-cell-claimed .dr-cell-mineflag { color: var(--warning); }
  /* Keeper: same position tint as a live pick; the KEEP flag is the marker.
     Do not wash the cell green — that hides WR/QB/TE color. */
  .dr-cell-keepflag { position: absolute; top: 2px; right: 5px; font-size: 8px; font-weight: 800;
    letter-spacing: .04em; color: var(--win,#15803d); }
  /* Traded pick: who the pick was dealt to (shown on another team's seat). */
  .dr-cell-owner { position: absolute; top: 2px; right: 5px; font-size: 8px; font-weight: 800;
    letter-spacing: .04em; color: var(--warning);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; pointer-events: none; }
  .dr-cell-just { animation: drPop .35s ease; }
  @keyframes drPop { 0% { transform: scale(.92); opacity: .3; } 100% { transform: scale(1); opacity: 1; } }
  .dr-cell-val { position: absolute; bottom: 3px; right: 4px; font-size: 9px; font-weight: 800; color: var(--accent,#38bdf8);
    background: color-mix(in srgb, var(--card) 70%, transparent); padding: 0 4px; border-radius: 5px; font-variant-numeric: tabular-nums; }
  .dr-cell-num { position: absolute; top: 2px; left: 5px; font-size: 9px; font-weight: 700; color: var(--text-muted); }
  .dr-board-toolbar { display: flex; align-items: center; justify-content: flex-end; padding: 4px 6px 2px; }
  .dr-cell-toggle { display: flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; font-size: 10px; font-weight: 700; }
  .dr-ct-opt { padding: 3px 9px; cursor: pointer; color: var(--text-muted); transition: background .15s, color .15s; }
  .dr-ct-opt.is-active { background: var(--accent,#38bdf8); color: var(--on-accent, #fff); }
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
  .dr-side { border: 1px solid var(--border); border-radius: 14px; background: var(--card); display: flex; flex-direction: column;
    position: sticky; top: 158px; align-self: start; max-height: calc(100vh - 166px); z-index: 20; overflow: hidden;
    box-shadow: var(--shadow-sm, 0 2px 8px rgba(15, 23, 42, 0.05)); }
  /* Reuse the trade-calculator pill tabs (otc-main-tabs), evenly spread across panel */
  .dr-side-tabs.otc-main-tabs { width: auto; margin: 8px; }
  .dr-side-tabs .otc-main-tab { flex: 1; display: flex; align-items: center; justify-content: center;
    text-align: center; padding: 7px 4px; font-size: 12px; }
  /* Team needs hover tooltip */
  .dr-team-tip { background: var(--tooltip-bg,var(--card)); color: var(--tooltip-fg,var(--text)); border: 1px solid var(--tooltip-border,var(--border)); border-radius: var(--tooltip-radius,10px);
    padding: 10px 12px; box-shadow: var(--tooltip-shadow,0 8px 28px rgba(0,0,0,.28)); min-width: 160px; }
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
  .dr-run-chip { font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: var(--radius-pill, 8px); background: rgba(127,127,127,.14); color: var(--text); border: 1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dr-run-hot { background: color-mix(in srgb, var(--loss) 16%, transparent); color: var(--loss); }
  .dr-run-banner { margin: 10px 10px 4px; padding: 8px 10px; border-radius: 8px; font-size: 12px;
    background: color-mix(in srgb, var(--loss) 12%, transparent); color: var(--loss); border: 1px solid color-mix(in srgb, var(--loss) 30%, transparent); }
  .dr-run-banner b { color: var(--loss); }
  .dr-cliff-banner { background: color-mix(in srgb, var(--warning) 12%, transparent); color: var(--warning); border-color: color-mix(in srgb, var(--warning) 35%, transparent); }
  .dr-cliff-banner b { color: var(--warning); }
  .dr-strat-tag { margin-left: 6px; font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .04em; color: var(--text-muted); border: 1px solid var(--border);
    border-radius: var(--radius-pill, 8px); padding: 1px 7px; vertical-align: middle; white-space: nowrap; }
  /* Pick trade evaluator (inside drModal) */
  .dr-pt-trigger { font-size: 12px; font-weight: 700; white-space: nowrap; }
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
  .dr-pt-row { display: flex; align-items: center; gap: 7px; font-size: 13px; padding: 3px 0; }
  .dr-pt-pk { font-weight: 800; color: var(--text); font-variant-numeric: tabular-nums; flex: 0 0 auto; min-width: 30px; }
  .dr-pt-pos { font-size: 9px; font-weight: 800; color: #fff; border-radius: 4px; padding: 1px 5px; flex: 0 0 auto; }
  .dr-pt-pos-QB { background: #e0483f; } .dr-pt-pos-RB { background: #199a4d; }
  .dr-pt-pos-WR { background: #2f6df0; } .dr-pt-pos-TE { background: #b5730b; }
  .dr-pt-nm { flex: 1 1 auto; min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: var(--text-muted); }
  .dr-pt-val { flex: 0 0 auto; font-weight: 800; color: var(--text); font-variant-numeric: tabular-nums; }
  .dr-pt-empty { color: var(--text-muted); font-style: italic; }
  .dr-pt-proxy { font-size: 11px; color: var(--text-muted); }
  .dr-pt-chips { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; margin: 7px 0 2px; }
  .dr-pt-chips-lbl { font-size: 11px; font-weight: 700; color: var(--text-muted); margin-right: 2px; }
  .dr-pt-chip { font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: var(--radius-pill, 8px); border: 1px solid var(--border); background: var(--bg); color: var(--text); cursor: pointer; }
  .dr-pt-chip:hover { border-color: var(--accent,#38bdf8); color: var(--accent,#38bdf8); }
  .dr-pt-picker { display: flex; gap: 6px; align-items: center; margin: 6px 0 4px; }
  .dr-pt-sel { flex: 1 1 auto; min-width: 0; padding: 8px 9px; border-radius: 8px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 13px; }
  .dr-pt-add { flex: 0 0 auto; white-space: nowrap; padding: 8px 14px; }
  .dr-pt-chiprow { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; min-height: 20px; margin: 2px 0; }
  .dr-pt-tok { display: inline-flex; align-items: center; gap: 2px; font-size: 12px; font-weight: 800;
    padding: 3px 4px 3px 10px; border-radius: var(--radius-pill, 8px); font-variant-numeric: tabular-nums;
    background: color-mix(in srgb, var(--accent,#38bdf8) 15%, transparent); color: var(--accent,#38bdf8);
    border: 1px solid color-mix(in srgb, currentColor 35%, transparent); }
  .dr-pt-tokx { background: none; border: 0; cursor: pointer; color: inherit; font-size: 15px; line-height: 1; padding: 0 4px; opacity: .75; }
  .dr-pt-tokx:hover { opacity: 1; }
  .dr-pt-bar { display: flex; height: 8px; border-radius: 12px; overflow: hidden; margin: 14px 0 10px; background: var(--border); }
  .dr-pt-bar-g { background: color-mix(in srgb, var(--text-muted) 60%, transparent); }
  .dr-pt-bar-r { background: #22c55e; }
  .dr-pt-verdict { font-size: 14px; font-weight: 800; }
  .dr-pt-vpct { font-size: 12px; font-weight: 700; opacity: .85; }
  .dr-prev-score-hero { border: 1px solid; border-radius: 10px; padding: 12px 10px 10px; margin-bottom: 12px; text-align: center; }
  .dr-prev-score-num { font-size: 44px; font-weight: 900; line-height: 1; }
  .dr-prev-score-lbl { font-size: 9px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin-top: 2px; }
  .dr-prev-score-reason { font-size: 12px; font-weight: 600; color: var(--text-muted); margin-top: 6px; }
  .dr-empty-note {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 6px; padding: 28px 16px; text-align: center;
  }
  .dr-empty-note-icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 40px; height: 40px; border-radius: 50%;
    background: color-mix(in srgb, var(--text-muted) 10%, transparent);
    color: var(--text-muted); margin-bottom: 2px;
  }
  .dr-empty-note-icon svg { width: 20px; height: 20px; display: block; }
  .dr-empty-note-title { font-size: 13px; font-weight: 800; color: var(--text); margin: 0; }
  .dr-empty-note-msg { font-size: 12px; color: var(--text-muted); line-height: 1.45; max-width: 34ch; margin: 0; }
  .dr-loading {
    display: flex; flex-direction: column; align-items: stretch; gap: 8px;
    padding: 14px 10px;
  }
  .dr-loading-msg {
    display: flex; align-items: center; justify-content: center; gap: 8px;
    padding: 22px 14px; color: var(--text-muted); font-size: 13px;
  }
  .dr-loading-msg .loading-spinner { width: 14px; height: 14px; margin: 0; flex-shrink: 0; }
  /* In-draft cheat sheet overlay (iframes the chrome-less cheat sheet). */
  .dr-cheat-overlay { position: fixed; inset: 0; z-index: 12000; background: rgba(0,0,0,.55);
    display: flex; align-items: center; justify-content: center; padding: 18px; }
  .dr-cheat-card { width: min(1180px, 96vw); height: min(90vh, 920px); background: var(--card);
    border: 1px solid var(--border); border-radius: 14px; display: flex; flex-direction: column;
    overflow: hidden; min-width: 0; box-shadow: 0 20px 60px rgba(0,0,0,.4); }
  .dr-cheat-head { display: flex; align-items: center; gap: 12px; padding: 10px 14px;
    border-bottom: 1px solid var(--border); flex: 0 0 auto; }
  .dr-cheat-title { font-weight: 800; font-size: 15px; color: var(--text); }
  .dr-cheat-pop { margin-left: auto; font-size: 12px; font-weight: 700; color: var(--accent,#38bdf8); text-decoration: none; }
  .dr-cheat-pop:hover { text-decoration: underline; }
  .dr-cheat-close { background: none; border: 0; font-size: 24px; line-height: 1; color: var(--text-muted); cursor: pointer; padding: 0 4px; }
  .dr-cheat-close:hover { color: var(--text); }
  .dr-cheat-frame { display: block; flex: 1 1 auto; width: 100%; min-width: 0; min-height: 0;
    border: 0; background: var(--bg); }
  /* tiers */
  .dr-tier { font-size: 9px; font-weight: 800; padding: 1px 5px; border-radius: var(--radius-pill, 8px);
    background: rgba(127,127,127,.18); color: var(--text-muted); flex-shrink: 0;
    border: 1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dr-tier-cliff { background: color-mix(in srgb, var(--loss) 16%, transparent); color: var(--loss); }
  /* pick score */
  .dr-ba-reason { font-size: 10px; color: var(--text-muted); margin-top: 5px; font-weight: 600;
    display: flex; align-items: center; gap: 5px; line-height: 1.25; }
  .dr-ba-recchip { color: var(--accent,#38bdf8); background: color-mix(in srgb, var(--accent) 11%, transparent);
    font-size: 12px; font-weight: 900; }
  .dr-ba-wait { font-size: 9.5px; color: var(--win); margin-top: 2px; font-weight: 700; }
  .dr-prev-wait { display: flex; align-items: center; gap: 10px; border: 1px solid; border-radius: 9px;
    padding: 9px 12px; margin-bottom: 12px; }
  .dr-prev-wait-p { font-size: 18px; font-weight: 900; flex-shrink: 0; }
  .dr-prev-wait-t { font-size: 12px; font-weight: 600; color: var(--text); line-height: 1.35; }
  /* draft grade */
  .dr-pill-grade { background: color-mix(in srgb, var(--win) 16%, transparent); color: var(--win); }
  .dr-grade-card { display: flex; align-items: center; gap: 12px; padding: 12px; margin: 10px 10px 4px;
    border: 1px solid var(--border); border-radius: 10px; background: var(--bg); }
  .dr-grade-letter { font-size: 34px; font-weight: 900; color: var(--accent,#38bdf8); line-height: 1; min-width: 48px; text-align: center; }
  .dr-grade-mark { display: flex; flex-direction: column; align-items: center; min-width: 48px; flex-shrink: 0; }
  .dr-grade-early { font-size: 9px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: var(--text-muted); margin-top: 3px; }
  .dr-grade-early-inline { font-size: 9px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; color: var(--text-muted); }
  .dr-grade-meta { flex: 1; min-width: 0; }
  .dr-grade-pace { font-size: 12px; font-weight: 700; color: var(--text); margin-bottom: 6px; }
  .dr-gbar-row { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
  .dr-gbar-lbl { font-size: 10px; color: var(--text-muted); width: 76px; flex-shrink: 0; }
  .dr-gbar { flex: 1; height: 6px; border-radius: 12px; background: rgba(127,127,127,.18); overflow: hidden; }
  .dr-gbar-fill { height: 100%; border-radius: 12px; }
  .dr-gbar-pct { font-size: 10px; font-weight: 800; width: 26px; text-align: right; flex-shrink: 0; }
  /* inline info-icon tooltip (ⓘ) */
  .dr-info { display:inline-flex; align-items:center; justify-content:center; width:13px; height:13px; border-radius:50%;
    border:1px solid var(--border); color:var(--text-muted); font-size:9px; font-weight:800; font-style:normal;
    cursor:help; margin-left:4px; position:relative; vertical-align:middle; line-height:1; flex-shrink:0; }
  .dr-info:hover, .dr-info:focus { border-color:var(--accent,#38bdf8); color:var(--accent,#38bdf8); outline:none; }
  /* Anchor the tooltip's left edge to the icon and extend rightward. These info
     icons all sit on the LEFT of their label, so a centered tooltip overflowed
     the panel's left edge and got clipped by its overflow:hidden ancestor. */
  .dr-info::after { content: attr(data-tip); position:absolute; top:calc(100% + 6px); left:0; transform:none;
    width:max-content; max-width:210px; background:var(--tooltip-bg,var(--card)); color:var(--tooltip-fg,var(--text)); border:1px solid var(--tooltip-border,var(--border));
    border-radius:var(--tooltip-radius,10px); padding:var(--tooltip-pad,8px 12px); font-size:var(--tooltip-fs,12px); font-weight:500; font-style:normal; line-height:var(--tooltip-lh,1.45); text-align:left;
    box-shadow:var(--tooltip-shadow,0 8px 24px rgba(0,0,0,.28)); opacity:0; pointer-events:none; transition:opacity .12s; z-index:600; white-space:normal; }
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
  .dr-preview-card { position: relative; width: 100%; max-width: 420px; background: var(--card);
    border: 1px solid var(--border); border-radius: 16px; padding: 18px 18px 16px; box-shadow: 0 18px 56px rgba(0,0,0,.34); margin: auto; }
  .dr-prev-close { position: absolute; top: 10px; right: 12px; width: 28px; height: 28px; background: var(--bg);
    border: 1px solid var(--border); border-radius: 12px; font-size: 17px; line-height: 1;
    color: var(--text-muted); cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: background .12s, color .12s; }
  .dr-prev-close:hover { background: color-mix(in srgb, var(--loss) 12%, transparent); color: var(--loss); }
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
  .dr-star.on { color: var(--warning); }
  .dr-side-title { font-size: 14px; font-weight: 800; color: var(--text); }
  .dr-side-controls { display: flex; gap: 6px; }
  .dr-side-controls input { flex: 1; min-width: 0; padding: 7px 9px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 12px; }
  .dr-side-controls select { padding: 7px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 12px; flex-shrink: 0; max-width: 110px; }
  /* Custom sort dropdown (replaces the native <select> popup, which mis-anchors
     inside the transformed mobile sheet). */
  .dr-sortsel { position: relative; flex-shrink: 0; }
  .dr-sortsel-btn {
    display: flex; align-items: center; gap: 6px; width: 100%;
    padding: 7px 9px; border-radius: 7px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); font-size: 12px; font-weight: 600;
    cursor: pointer; white-space: nowrap; line-height: 1;
  }
  .dr-sortsel-caret { color: var(--text-muted); transition: transform .15s; flex-shrink: 0; }
  .dr-sortsel-btn[aria-expanded="true"] .dr-sortsel-caret { transform: rotate(180deg); }
  .dr-sortsel-menu {
    position: absolute; top: calc(100% + 4px); left: 0; z-index: 60;
    min-width: 100%; width: max-content; padding: 4px;
    background: var(--card); border: 1px solid var(--border); border-radius: 9px;
    box-shadow: 0 10px 30px rgba(0,0,0,.22); display: flex; flex-direction: column; gap: 2px;
  }
  .dr-sortsel-menu[hidden] { display: none; }
  .dr-sortsel-opt {
    display: block; width: 100%; text-align: left; padding: 8px 12px; border: none;
    border-radius: 6px; background: none; color: var(--text); font-size: 13px;
    font-weight: 600; cursor: pointer; white-space: nowrap;
  }
  .dr-sortsel-opt:hover { background: color-mix(in srgb, var(--accent) 10%, transparent); }
  .dr-sortsel-opt.is-active { background: var(--accent,#38bdf8); color: var(--on-accent, #fff); }
  .dr-pos-filters { display: flex; gap: 6px; flex-wrap: wrap; }
  .dr-adp-src { font-size: 10px; color: var(--text-muted); display: flex; align-items: center; gap: 6px; }
  .dr-adp-src-label { font-size: 10px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .dr-adp-src-select { padding: 4px 7px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 11px; cursor: pointer; outline: none; }
  .dr-ba-list { overflow-y: auto; flex: 1; }
  .dr-ba-row { display: flex; align-items: center; gap: 10px; padding: 8px 12px 8px 5px; border-bottom: 1px solid var(--border); cursor: pointer; transition: background .12s; }
  .dr-ba-row:hover { background: color-mix(in srgb, var(--accent) 6%, transparent); }
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
  .dr-ba-row.dr-avail-hi { box-shadow: inset 3px 0 0 var(--win); }
  .dr-ba-row.dr-avail-md { box-shadow: inset 3px 0 0 var(--warning); }
  .dr-ba-row.dr-avail-lo { box-shadow: inset 3px 0 0 var(--loss); }
  .dr-ba-avail { font-size: 9.5px; font-weight: 700; margin-top: 2px; }
  /* ── Preview modal availability track ── */
  .dr-prev-avail-track { margin-bottom: 12px; }
  .dr-prev-avail-label { font-size: 10px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }
  .dr-prev-avail-picks { display: flex; flex-wrap: wrap; gap: 6px; }
  .dr-prev-avail-pick { display: inline-flex; align-items: baseline; gap: 4px; padding: 5px 10px; border-radius: 8px; }
  .dr-prev-avail-pn { font-size: 10px; font-weight: 600; color: var(--text-muted); }
  @media (max-width: 768px) {
    /* Treat the in-draft sheet as a real mobile screen. A centered desktop modal
       leaves too little room for the controls and can sit behind the app dock. */
    body.dr-cheat-open { overflow: hidden; }
    .dr-cheat-overlay { padding: 0; align-items: stretch; background: var(--card); }
    .dr-cheat-card { width: 100%; height: 100vh; height: 100dvh; max-width: none;
      border: 0; border-radius: 0; box-shadow: none; }
    .dr-cheat-frame { height: 0; }
    .dr-cheat-head { min-height: 54px; padding: max(10px, env(safe-area-inset-top))
      max(12px, env(safe-area-inset-right)) 10px max(12px, env(safe-area-inset-left)); }
    .dr-cheat-title { font-size: 16px; }
    .dr-cheat-pop { font-size: 13px; }
    .dr-cheat-close { min-width: 38px; min-height: 38px; font-size: 28px; }
    /* The global mobile tab bar (56px, fixed at the bottom) overlaps the draft
       sheet. Pad the scrollable list so its content always clears the bar; when
       the sheet is dragged to full, hide the bar so the sheet uses the whole
       screen (per the "full covers the bar" behavior). */
    .dr-side .dr-ba-list { padding-bottom: calc(56px + env(safe-area-inset-bottom) + 6px); }
    body.dr-sheet-expanded .br-tabbar { display: none; }
    body.dr-sheet-expanded .dr-side .dr-ba-list { padding-bottom: calc(env(safe-area-inset-bottom) + 6px); }
  }
  @media (max-width: 900px) {
    .dr-cols { grid-template-columns: 1fr; padding-bottom: 52vh; }
    .dr-statusbar { top: 0; }
    /* The side panel becomes a draggable bottom sheet */
    .dr-side {
      /* Anchored to the bottom so a full sheet still covers the tab bar. Height
         is capped below full-viewport so the top of a fully-raised sheet stops
         under the page header + draft status bar (whose-pick / pick number)
         instead of covering them. */
      position: fixed; left: 0; right: 0; bottom: 0; top: auto;
      width: 100%; height: 85vh; max-height: 85vh; align-self: auto; order: 0;
      border-radius: 18px 18px 0 0; border-bottom: none;
      box-shadow: 0 -10px 40px rgba(0,0,0,.28); z-index: 50;
      transform: translateY(42vh);          /* default: ~43vh visible (mid snap) */
      transition: transform .3s cubic-bezier(.32,.72,0,1);
    }
    .dr-side.dragging { transition: none; }
    .dr-sheet-handle {
      display: flex; align-items: center; justify-content: center;
      width: 100%; height: 26px; padding: 0; border: none; background: none;
      cursor: grab; flex-shrink: 0; touch-action: none;
    }
    .dr-sheet-handle:active { cursor: grabbing; }
    .dr-sheet-grip { width: 40px; height: 5px; border-radius: 12px; background: var(--border);
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
    /* Start / ESPN-fallback banners: stack the CTA under the copy so the
       long "Switch to Manual Tracking" label cannot crush the text into a
       one-word-wide column beside it. */
    .dr-start-banner {
      flex-wrap: wrap;
      align-items: flex-start;
      gap: 10px 12px;
      padding: 12px;
    }
    .dr-start-banner .dr-banner-txt { flex: 1 1 0; min-width: 0; }
    .dr-start-banner .dr-banner-txt span { overflow-wrap: anywhere; }
    .dr-start-banner .dr-banner-join {
      flex: 1 1 100%;
      margin-left: 0;
      width: 100%;
      justify-content: center;
      white-space: normal;
      text-align: center;
    }
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
    .dr-league-meta { font-size: 11px; padding: 2px 4px; }
    .dr-lm-chip { font-size: 10px; padding: 1px 6px; }
    .dr-pill, .dr-roster-src-tag, .dr-cap-pill { font-size: 10px; padding: 2px 7px; }
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
  /* Summary overlay: capped height so Deep Dive / Share / Close stay on screen
     while the roster list scrolls inside the card. */
  .dr-summary-overlay { position:fixed; inset:0; z-index:1001; background:rgba(0,0,0,.6);
    display:flex; align-items:center; justify-content:center; overflow:hidden;
    /* Clear the status bar / dynamic island at the top and the home indicator at the bottom. */
    padding:calc(env(safe-area-inset-top) + 16px) 16px calc(env(safe-area-inset-bottom) + 20px); }
  .dr-summary-card { position:relative; width:100%; max-width:500px; margin:0 auto; background:var(--card);
    border:1px solid var(--border); border-radius:20px; overflow:hidden;
    box-shadow:0 24px 80px rgba(0,0,0,.5); display:flex; flex-direction:column;
    max-height:min(620px, calc(100dvh - 48px)); }
  /* Grade ring + bars header */
  .dr-sum-header { padding:16px 20px 0; flex-shrink:0; }
  .dr-sum-title { font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.1em;
    color:var(--text-muted); text-align:center; margin-bottom:10px; }
  .dr-sum-grade-wrap { display:flex; align-items:center; gap:18px; padding-bottom:12px; }
  .dr-sum-grade-ring { width:76px; height:76px; border-radius:50%; border:3px solid;
    display:flex; align-items:center; justify-content:center; flex-shrink:0; }
  .dr-sum-grade { font-size:30px; font-weight:900; line-height:1; }
  .dr-sum-grade-bars { flex:1; display:flex; flex-direction:column; gap:5px; }
  /* Stats strip */
  .dr-sum-stats { display:flex; border-top:1px solid var(--border); border-bottom:1px solid var(--border); flex-shrink:0; }
  .dr-sum-stat { flex:1; text-align:center; padding:10px 4px; }
  .dr-sum-stat:not(:last-child) { border-right:1px solid var(--border); }
  .dr-sum-stat-v { font-size:20px; font-weight:900; color:var(--text); line-height:1; }
  .dr-sum-stat-l { font-size:9px; color:var(--text-muted); margin-top:3px; text-transform:uppercase; letter-spacing:.04em; }
  /* Archetype / window strip */
  .dr-sum-arch { display:flex; align-items:center; justify-content:center; gap:14px; flex-wrap:wrap;
    padding:10px 16px; border-bottom:1px solid var(--border); flex-shrink:0; }
  .dr-sum-arch-item { display:flex; flex-direction:column; align-items:center; gap:4px; }
  .dr-sum-arch-tag { font-size:9px; font-weight:800; text-transform:uppercase; letter-spacing:.06em; color:var(--text-muted); }
  .dr-sum-arch-label { font-size:14px; font-weight:900; color:var(--accent); line-height:1.1; }
  .dr-sum-arch-div { width:1px; height:32px; background:var(--border); flex-shrink:0; }
  /* Competitive window chips */
  .dr-sum-win { font-size:12px; font-weight:800; padding:4px 10px; border-radius:var(--radius-pill, 8px); white-space:nowrap; border:1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dr-win-winnow { background:color-mix(in srgb, var(--win) 16%, transparent); color:var(--win); }
  .dr-win-balanced { background:color-mix(in srgb, var(--warning) 16%, transparent); color:var(--warning); }
  .dr-win-future { background:color-mix(in srgb, var(--accent) 16%, transparent); color:var(--accent); }
  /* Roster list scrolls; header + footer stay put. */
  .dr-sum-body-wrap { padding:0 16px 4px; flex:1 1 auto; min-height:0; overflow-y:auto;
    -webkit-overflow-scrolling:touch; overscroll-behavior:contain; }
  .dr-sum-section { font-size:9px; font-weight:800; text-transform:uppercase; letter-spacing:.08em;
    color:var(--text-muted); margin:14px 0 6px; }
  .dr-sum-section:first-child { margin-top:10px; }
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
  /* Footer stays pinned to the bottom of the card. */
  .dr-sum-footer { display:flex; gap:8px; padding:12px 16px 14px; flex-shrink:0;
    border-top:1px solid var(--border); background:var(--card); position:sticky; bottom:0; z-index:2; }
  .dr-sum-footer .dr-btn { flex:1; text-align:center; }
  @media (max-width: 640px) {
    .dr-summary-overlay { padding:calc(env(safe-area-inset-top) + 8px) 10px calc(env(safe-area-inset-bottom) + 10px);
      align-items:center; }
    .dr-summary-card { max-height:min(78dvh, calc(100dvh - 16px)); border-radius:16px; }
    .dr-sum-footer { padding:10px 12px calc(10px + env(safe-area-inset-bottom)); }
  }
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
  .dr-modal-box.is-wide { max-width:560px; max-height:min(86vh, 720px); overflow:auto; padding:22px 22px 18px; }
  .dr-modal-msg { font-size:15px; color:var(--text); line-height:1.55; margin-bottom:20px; }
  .dr-modal-box.is-wide .dr-modal-msg { margin-bottom:14px; }
  .dr-modal-btns { display:flex; gap:10px; justify-content:flex-end; flex-wrap:wrap; }
  .dr-msync-title { font-size:18px; font-weight:800; letter-spacing:-0.02em; margin:0 0 6px; color:var(--text); display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
  .dr-msync-ver { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:800; letter-spacing:0.02em;
    border:1px solid var(--border); color:var(--text-muted); background:color-mix(in srgb, var(--text) 6%, transparent); }
  .dr-msync-lead { font-size:13px; color:var(--text-muted); margin:0 0 14px; line-height:1.5; }
  .dr-msync-warn { font-size:12px; line-height:1.45; padding:10px 12px; border-radius:10px; margin:0 0 14px;
    background:color-mix(in srgb, var(--warning) 14%, transparent); border:1px solid color-mix(in srgb, var(--warning) 35%, transparent); color:var(--text); }
  .dr-msync-sec { margin:0 0 14px; }
  .dr-msync-sec h4 { font-size:13px; font-weight:800; margin:0 0 6px; color:var(--text); }
  .dr-msync-sec ol { margin:0; padding-left:1.2em; font-size:13px; color:var(--text); line-height:1.55; }
  .dr-msync-sec li { margin:0 0 4px; }
  .dr-msync-sec p { margin:0; font-size:12px; color:var(--text-muted); line-height:1.45; }
  .dr-msync-status { font-size:12px; color:var(--win); min-height:1.2em; margin:4px 0 0; }
  /* Complete-draft sidebar footer */
  #drCompleteBar { padding:10px; border-top:1px solid var(--border); display:flex; flex-direction:column; gap:7px; flex-shrink:0; }
  .dr-btn-deepdive { background:color-mix(in srgb, var(--accent) 12%, transparent); border-color:color-mix(in srgb, var(--accent) 35%, var(--border)); color:var(--accent); font-weight:700; display:flex; align-items:center; justify-content:center; }
  .dr-btn-deepdive:hover { background:color-mix(in srgb, var(--accent) 20%, transparent); }
  .dr-dd-prochip, .dr-sum-prolock { font-size:9px; font-weight:800; letter-spacing:.06em; background:var(--accent); color:var(--on-accent,#fff); border-radius:4px; padding:1px 5px; margin-left:7px; }
  /* ── Deep Dive analyzer ── */
  .dr-dd-overlay { position:fixed; inset:0; z-index:12500; background:rgba(0,0,0,.62); display:flex; align-items:flex-start; justify-content:center; overflow-y:auto; padding:calc(env(safe-area-inset-top) + 14px) 14px calc(env(safe-area-inset-bottom) + 18px); }
  .dr-dd-card { position:relative; width:100%; max-width:940px; margin:0 auto; background:var(--bg); border:1px solid var(--border); border-radius:20px; overflow:hidden; box-shadow:0 24px 80px rgba(0,0,0,.5); display:flex; flex-direction:column; max-height:calc(100vh - 40px); }
  .dr-dd-card .dr-prev-close { z-index:3; }
  .dd-head { padding:20px 22px 16px; border-bottom:1px solid var(--border); background:var(--card); }
  .dd-kicker { font-family:"Archivo",sans-serif; font-size:11px; font-weight:800; letter-spacing:.11em; text-transform:uppercase; color:var(--text-muted); display:flex; align-items:center; }
  .dd-pro { font-size:9px; font-weight:800; letter-spacing:.06em; background:var(--accent); color:var(--on-accent,#fff); border-radius:4px; padding:1px 6px; margin-left:9px; }
  .dd-sub { font-size:13px; color:var(--text-muted); margin-top:4px; }
  .dd-scroll { overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:14px; }
  .dd-foot { padding:12px 16px; border-top:1px solid var(--border); background:var(--card); display:flex; justify-content:flex-end; }
  .dd-foot .dr-btn { min-width:120px; }
  .dd-card { background:var(--card); border:1px solid var(--border); border-radius:15px; padding:18px; }
  .dd-note { color:var(--text-muted); font-size:13.5px; }
  .dd-sec { margin-bottom:14px; }
  .dd-sec h4 { margin:0; font-family:"Archivo",sans-serif; font-size:16px; font-weight:800; color:var(--text); }
  .dd-h-sub { display:inline-block; margin-left:8px; font-size:10px; font-weight:700; letter-spacing:.06em;
    text-transform:uppercase; color:var(--text-muted); vertical-align:baseline; position:relative; top:.18em; }
  .dd-sec p { margin:4px 0 0; font-size:12.5px; color:var(--text-muted); }
  /* overview */
  .dd-ov-top { display:grid; grid-template-columns:auto 1fr; gap:18px 22px; align-items:center; }
  .dd-ring { position:relative; width:104px; height:104px; border-radius:50%; flex:none; background:conic-gradient(var(--gc) calc(var(--pct)*1%), var(--border) 0); display:grid; place-items:center; }
  .dd-ring::after { content:""; position:absolute; inset:8px; border-radius:50%; background:var(--card); }
  .dd-ring b { position:relative; z-index:1; font-family:"Archivo",sans-serif; font-weight:800; font-size:38px; line-height:1; text-align:center; }
  .dd-ring b small { display:block; font-size:12px; color:var(--text-muted); font-weight:600; margin-top:2px; }
  .dd-ov-txt h3 { margin:0; font-family:"Archivo",sans-serif; font-size:22px; font-weight:800; letter-spacing:-.01em; }
  .dd-rankline { margin-top:5px; font-size:13px; color:var(--text-muted); }
  .dd-rankline b { color:var(--text); }
  .dd-say { margin-top:8px; font-size:13.5px; color:var(--text); border-left:3px solid var(--accent); padding-left:11px; }
  .dd-meters { grid-column:1 / -1; display:flex; flex-direction:column; gap:11px; margin-top:4px; }
  .dd-meter { display:grid; grid-template-columns:150px 1fr auto; gap:13px; align-items:center; }
  .dd-meter-lab { font-size:13px; font-weight:600; }
  .dd-meter-lab small { display:block; font-weight:500; color:var(--text-subtle,var(--text-muted)); font-size:11px; }
  .dd-track { height:8px; border-radius:99px; background:var(--border); overflow:hidden; }
  .dd-track i { display:block; height:100%; border-radius:99px; }
  .dd-meter-val { font-family:"Archivo",sans-serif; font-weight:700; font-size:14px; font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; }
  .dd-meter-val span { font-size:11px; color:var(--text-muted); font-weight:600; }
  .dd-rankpill { display:inline-block; font-size:11px; font-weight:700; padding:2px 7px; border-radius:var(--radius-pill, 8px); margin-left:6px; border:1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dd-rk-top { background:color-mix(in srgb,#22c55e 16%,transparent); color:#16a34a; }
  .dd-rk-mid { background:color-mix(in srgb,var(--accent) 15%,transparent); color:var(--accent); }
  .dd-rk-low { background:color-mix(in srgb,#ef4444 15%,transparent); color:#dc2626; }
  .dd-tiles { display:grid; grid-template-columns:repeat(4,1fr); gap:11px; margin-top:16px; }
  .dd-tile { border:1px solid var(--border); border-radius:12px; padding:13px 14px; background:var(--bg); }
  .dd-tile-v { font-family:"Archivo",sans-serif; font-weight:800; font-size:23px; line-height:1; font-variant-numeric:tabular-nums; }
  .dd-tile-l { font-size:11.5px; color:var(--text-muted); margin-top:6px; }
  .dd-tile.good .dd-tile-v { color:#16a34a; } .dd-tile.bad .dd-tile-v { color:#dc2626; }
  /* legend + chart */
  .dd-legend { display:flex; gap:13px; flex-wrap:wrap; font-size:12px; color:var(--text-muted); margin-bottom:10px; }
  .dd-legend span { display:inline-flex; align-items:center; gap:6px; }
  .dd-dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
  .dd-sq { width:11px; height:11px; border-radius:3px; display:inline-block; }
  .dd-chart-hint { display:none; }
  .dd-chartscroll, .dd-tablescroll {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    overscroll-behavior-x: contain;
    scrollbar-width: thin;
  }
  .dd-chartscroll svg { display:block; max-width:none; }
  .dd-tl-dot:hover { stroke:var(--text); stroke-width:1.6; }
  .dd-tip { position:fixed; z-index:12800; pointer-events:none; background:var(--tooltip-bg,var(--card)); color:var(--tooltip-fg,var(--text)); border:1px solid var(--tooltip-border,var(--border)); box-shadow:var(--tooltip-shadow,0 12px 40px rgba(0,0,0,.4)); border-radius:var(--tooltip-radius,10px); padding:var(--tooltip-pad,8px 12px); font-size:var(--tooltip-fs,12px); line-height:var(--tooltip-lh,1.45); opacity:0; transform:translateY(4px); transition:opacity .12s; max-width:280px; }
  .dd-tip.show { opacity:1; transform:none; }
  .dd-tip b { font-family:"Archivo",sans-serif; }
  .dd-tip-r { display:flex; justify-content:space-between; gap:16px; color:var(--text-muted); margin-top:3px; }
  .dd-tip-r b { color:var(--text); font-family:inherit; }
  .dd-tip-opp { margin-top:6px; font-size:11.5px; color:var(--text); line-height:1.4; }
  /* tables */
  .dd-ledger { width:100%; border-collapse:collapse; font-size:13px; }
  .dd-ledger th, .dd-ledger td { padding:9px 11px; text-align:left; border-bottom:1px solid var(--border); white-space:nowrap; }
  .dd-ledger thead th { font-size:10.5px; letter-spacing:.05em; text-transform:uppercase; color:var(--text-subtle,var(--text-muted)); cursor:pointer; user-select:none; }
  .dd-ledger thead th:hover { color:var(--text); }
  .dd-ledger thead th.dd-sorted { color:var(--accent); }
  .dd-ledger thead th.r, .dd-ledger tbody td.r { text-align:center; font-variant-numeric:tabular-nums; }
  .dd-ledger .num { font-variant-numeric:tabular-nums; }
  .dd-ledger tbody tr:hover { background:color-mix(in srgb,var(--accent) 5%,transparent); }
  .dd-ledger td.dd-plcell { white-space:normal; min-width:140px; }
  .dd-plname { font-weight:600; }
  .dd-pl-sub { margin-top:3px; font-size:11px; font-weight:500; color:var(--text-muted); line-height:1.35; max-width:280px; }
  .dd-opp-sev { display:inline-block; margin-left:4px; font-size:10px; font-weight:700; letter-spacing:.02em; text-transform:uppercase; }
  .dd-opp-modest { color:#b45309; }
  .dd-opp-material { color:#c2410c; }
  .dd-opp-severe { color:#dc2626; }
  .dd-facets { display:flex; flex-direction:column; gap:6px; margin-top:14px; }
  .dd-facet { font-size:12.5px; color:var(--text); line-height:1.4; padding:8px 11px; border-left:3px solid color-mix(in srgb,var(--accent) 55%,var(--border)); background:color-mix(in srgb,var(--accent) 6%,transparent); }
  .dd-facet-line { margin:8px 0 0; font-size:12.5px; line-height:1.4; }
  .dd-posbadge { display:inline-block; min-width:30px; text-align:center; font-size:10px; font-weight:800; color:#fff; padding:2px 6px; border-radius:5px; }
  .dd-diff { display:inline-block; min-width:6.2ch; text-align:right; font-weight:800;
    font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1; }
  .dd-diff.p { color:#16a34a; } .dd-diff.n { color:#dc2626; } .dd-diff.z { color:var(--text-muted); }
  .dd-verd { font-size:10.5px; font-weight:800; padding:3px 9px; border-radius:var(--radius-pill, 8px); border:1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dd-v-steal { background:color-mix(in srgb,#22c55e 16%,transparent); color:#16a34a; }
  .dd-v-value { background:color-mix(in srgb,var(--accent) 14%,transparent); color:var(--accent); }
  .dd-v-fair { background:var(--bg); color:var(--text-muted); border:1px solid var(--border); }
  .dd-v-aggressive { background:color-mix(in srgb,#f59e0b 14%,transparent); color:#d97706; }
  .dd-v-reach { background:color-mix(in srgb,#ef4444 14%,transparent); color:#dc2626; }
  .dd-v-keep { background:color-mix(in srgb,var(--text-muted) 12%,transparent); color:var(--text-muted); border:1px solid var(--border); }
  .dd-v-na { color:var(--text-subtle,var(--text-muted)); }
  /* league board */
  .dd-league tbody tr.dd-me { background:color-mix(in srgb,var(--accent) 9%,transparent); }
  .dd-youtag { font-size:9px; font-weight:800; background:var(--accent); color:var(--on-accent,#fff); border-radius:4px; padding:1px 5px; margin-left:6px; }
  .dd-gletter { font-family:"Archivo",sans-serif; font-weight:800; font-size:15px; }
  .dd-odds { display:flex; align-items:center; gap:8px; min-width:130px; }
  .dd-odds-track { flex:1; height:7px; border-radius:99px; background:var(--border); overflow:hidden; }
  .dd-odds-track i { display:block; height:100%; border-radius:99px; }
  .dd-odds .num { font-variant-numeric:tabular-nums; font-weight:600; font-size:12.5px; min-width:34px; text-align:right; }
  .dd-odds-pending { color:var(--text-muted); font-size:12px; font-style:italic; }
  /* construction */
  .dd-two { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
  .dd-cap-row { display:grid; grid-template-columns:40px 1fr 78px; gap:11px; align-items:center; margin-bottom:10px; }
  .dd-cap-pos { font-size:12px; font-weight:800; }
  .dd-cap-track { position:relative; height:20px; border-radius:6px; background:var(--border); overflow:visible; }
  .dd-cap-track i { display:block; height:100%; border-radius:6px; }
  .dd-cap-lg { position:absolute; top:-3px; width:2px; height:26px; background:var(--text); opacity:.55; }
  .dd-cap-val { font-family:"Archivo",sans-serif; font-weight:700; font-size:13px; text-align:right; font-variant-numeric:tabular-nums; }
  .dd-cap-val small { display:block; font-family:inherit; font-weight:500; color:var(--text-subtle,var(--text-muted)); font-size:10px; }
  .dd-st-row { display:grid; grid-template-columns:auto 1fr auto auto; gap:10px; align-items:center; padding:7px 0; border-bottom:1px solid var(--border); }
  .dd-slotbadge { font-size:10px; font-weight:800; padding:3px 7px; border-radius:6px; border:1px solid var(--border); min-width:40px; text-align:center; }
  .dd-st-name { font-size:13px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .dd-st-ppg { font-family:"Archivo",sans-serif; font-weight:700; font-size:13px; font-variant-numeric:tabular-nums; }
  .dd-st-ppg small { font-family:"Inter",sans-serif; font-weight:500; color:var(--text-muted); font-size:10px; margin-left:2px; }
  .dd-st-rank { font-size:11.5px; color:var(--text-muted); white-space:nowrap; }
  .dd-st-rank b { color:var(--text); }
  .dd-hist-pct { font-family:"Archivo",sans-serif; font-weight:800; font-variant-numeric:tabular-nums; }
  .dd-hist-pct.is-strong { color:#16a34a; }
  .dd-hist-vs.up { color:#16a34a; font-weight:700; }
  .dd-hist-vs.down { color:#dc2626; font-weight:700; }
  /* Historical trends (Deep Dive) — scoped so early-ADP “high bar” never reads as a miss */
  .dd-hist { display:flex; flex-direction:column; gap:16px; }
  .dd-hist > .dd-sec { margin-bottom:0; }
  .dd-hist-stats {
    display:grid; grid-template-columns:1.35fr repeat(3, minmax(0, 1fr)); gap:10px;
  }
  .dd-hist-stat {
    border:1px solid var(--border); border-radius:12px; padding:12px 13px;
    background:color-mix(in srgb, var(--bg) 88%, var(--card));
    min-width:0;
  }
  .dd-hist-stat.is-lead {
    background:
      linear-gradient(135deg, color-mix(in srgb, var(--accent) 12%, transparent), transparent 62%),
      var(--bg);
    border-color:color-mix(in srgb, var(--accent) 28%, var(--border));
  }
  .dd-hist-stat.is-good .dd-hist-stat-v { color:#16a34a; }
  .dd-hist-stat.is-muted .dd-hist-stat-v { color:var(--text-muted); font-weight:700; }
  .dd-hist-stat.is-info {
    background:transparent;
    border-style:dashed;
  }
  .dd-hist-stat-v {
    font-family:"Archivo",sans-serif; font-weight:800; font-size:22px; line-height:1;
    font-variant-numeric:tabular-nums; letter-spacing:-.02em; color:var(--text);
  }
  .dd-hist-stat.is-lead .dd-hist-stat-v { font-size:26px; }
  .dd-hist-stat-l {
    margin-top:6px; font-size:11px; font-weight:650; line-height:1.3;
    color:var(--text-muted);
  }
  .dd-hist-callouts {
    display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:12px;
  }
  .dd-hist-callout {
    border:1px solid var(--border); border-radius:13px; padding:13px 14px 14px;
    background:var(--bg); display:flex; flex-direction:column; gap:0; min-width:0;
  }
  .dd-hist-callout.is-ahead {
    border-color:color-mix(in srgb, #22c55e 34%, var(--border));
    background:
      linear-gradient(180deg, color-mix(in srgb, #22c55e 8%, transparent), transparent 48%),
      var(--bg);
  }
  .dd-hist-callout.is-bar {
    border-color:color-mix(in srgb, var(--accent) 26%, var(--border));
    background:
      linear-gradient(180deg, color-mix(in srgb, var(--accent) 8%, transparent), transparent 48%),
      var(--bg);
  }
  .dd-hist-callout-k {
    font-size:10.5px; font-weight:800; letter-spacing:.05em; text-transform:uppercase;
    color:var(--text-muted);
  }
  .dd-hist-callout.is-ahead .dd-hist-callout-k { color:#16a34a; }
  .dd-hist-callout.is-bar .dd-hist-callout-k { color:var(--accent); }
  .dd-hist-callout-pl {
    font-family:"Archivo",sans-serif; font-weight:800; font-size:17px;
    letter-spacing:-.01em; margin-top:6px; color:var(--text);
  }
  .dd-hist-callout-sub { font-size:12px; color:var(--text-muted); margin-top:2px; }
  .dd-hist-compare {
    display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px;
  }
  .dd-hist-compare-col {
    border-radius:10px; padding:9px 10px;
    background:color-mix(in srgb, var(--card) 70%, var(--bg));
    border:1px solid var(--border);
    display:flex; flex-direction:column; gap:4px; min-width:0;
  }
  .dd-hist-compare-col.is-hist {
    border-color:color-mix(in srgb, #22c55e 22%, var(--border));
  }
  .dd-hist-compare-col.is-adp {
    border-color:color-mix(in srgb, var(--accent) 22%, var(--border));
  }
  .dd-hist-compare-k {
    font-size:10px; font-weight:700; letter-spacing:.03em; text-transform:uppercase;
    color:var(--text-muted); line-height:1.25;
  }
  .dd-hist-compare-v {
    font-family:"Archivo",sans-serif; font-weight:800; font-size:18px;
    font-variant-numeric:tabular-nums; letter-spacing:-.02em; color:var(--text);
  }
  .dd-hist-callout-say {
    margin:10px 0 0; font-size:12.5px; line-height:1.45; color:var(--text-muted);
  }
  /* overflow-x must stay auto — overflow:hidden here used to clip the
     dd-tablescroll horizontal swipe on narrow phones. */
  .dd-hist-tablewrap {
    margin-top:2px; border:1px solid var(--border); border-radius:12px;
    overflow-x:auto; overflow-y:hidden;
    -webkit-overflow-scrolling:touch;
    overscroll-behavior-x:contain;
    scrollbar-width:thin;
  }
  .dd-hist-table { margin:0; width:max-content; min-width:100%; }
  .dd-hist-table thead th {
    background:color-mix(in srgb, var(--card) 82%, var(--bg));
    position:sticky; top:0; z-index:1;
  }
  .dd-hist-table tbody tr:last-child td { border-bottom:none; }
  .dd-hist-pick { color:var(--text-muted); font-weight:600; }
  .dd-hist-mkt { color:var(--text-muted); font-variant-numeric:tabular-nums; font-weight:600; }
  .dd-hist-vs {
    display:inline-block; min-width:7.5ch; font-size:12px; font-weight:700;
    font-variant-numeric:tabular-nums; color:var(--text-muted);
  }
  .dd-hist-vs.is-up {
    color:#16a34a;
    background:color-mix(in srgb, #22c55e 12%, transparent);
    border:1px solid color-mix(in srgb, #22c55e 28%, transparent);
    border-radius:999px; padding:2px 8px; min-width:0;
  }
  .dd-hist-vs.is-flat {
    color:var(--text-muted);
    background:var(--bg);
    border:1px solid var(--border);
    border-radius:999px; padding:2px 8px; min-width:0;
  }
  .dd-hist-vs.is-bar {
    color:var(--accent);
    background:color-mix(in srgb, var(--accent) 11%, transparent);
    border:1px solid color-mix(in srgb, var(--accent) 26%, transparent);
    border-radius:999px; padding:2px 8px; min-width:0;
  }
  /* edges + flags */
  .dd-edges { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
  .dd-edge { padding:14px; border-radius:12px; border:1px solid var(--border); background:var(--bg); }
  .dd-edge-k { font-size:10.5px; font-weight:800; letter-spacing:.04em; text-transform:uppercase; }
  .dd-edge.win .dd-edge-k { color:#16a34a; } .dd-edge.winb .dd-edge-k { color:var(--accent); } .dd-edge.bad .dd-edge-k { color:#dc2626; }
  .dd-edge-pl { font-family:"Archivo",sans-serif; font-weight:700; font-size:16px; margin-top:7px; }
  .dd-edge-sub { font-size:12px; color:var(--text-muted); margin-top:2px; }
  .dd-edge-say { font-size:12px; color:var(--text); margin-top:8px; }
  .dd-flags { display:flex; flex-direction:column; gap:9px; margin-top:13px; }
  .dd-flag { display:grid; grid-template-columns:auto 1fr; gap:11px; padding:12px 13px; border-radius:11px; border:1px solid var(--border); background:var(--bg); }
  .dd-flag-ic { width:30px; height:30px; border-radius:8px; display:grid; place-items:center; font-weight:800; flex:none; }
  .dd-flag-crit { border-color:color-mix(in srgb,#ef4444 40%,var(--border)); }
  .dd-flag-crit .dd-flag-ic { background:color-mix(in srgb,#ef4444 15%,transparent); color:#dc2626; }
  .dd-flag-warn .dd-flag-ic { background:color-mix(in srgb,#f59e0b 16%,transparent); color:#d97706; }
  .dd-flag-ttl { font-weight:700; font-size:13px; }
  .dd-flag-ds { font-size:12px; color:var(--text-muted); margin-top:2px; }
  @media (max-width:720px){
    .dr-dd-card { max-width:100%; border-radius:14px; }
    .dd-ov-top { grid-template-columns:1fr; text-align:center; }
    .dd-ring { margin:0 auto; }
    .dd-say { text-align:left; }
    .dd-meter { grid-template-columns:120px 1fr auto; }
    .dd-tiles { grid-template-columns:repeat(2,1fr); }
    .dd-hist-stats { grid-template-columns:repeat(2, minmax(0, 1fr)); }
    .dd-hist-stat.is-lead { grid-column:1 / -1; }
    .dd-hist-callouts { grid-template-columns:1fr; }
    .dd-two { grid-template-columns:1fr; }
    .dd-edges { grid-template-columns:1fr; }
    .dd-card { padding:14px; }
    .dd-legend { gap:8px 11px; }
    .dd-legend span:nth-last-child(3) { margin-left:0 !important; }
    .dd-chart-hint { display:block; margin:-2px 0 7px; color:var(--text-muted); font-size:11px; font-weight:650; }
    .dd-chartscroll { margin:0 -6px -4px; padding:0 6px 4px; -webkit-overflow-scrolling:touch; scroll-snap-type:x proximity; }
    .dd-chartscroll svg { touch-action:pan-x; scroll-snap-align:start; }
  }
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
  .dr-roster-presets { display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin-bottom:8px; }
  .dr-roster-presets-label { font-size:10px; font-weight:800; color:var(--text-muted); text-transform:uppercase; letter-spacing:.05em; margin-right:2px; }
  .dr-roster-preset { font-size:11px; font-weight:750; color:var(--text-muted); background:var(--bg);
    border:1px solid var(--border); border-radius:6px; padding:4px 9px; cursor:pointer; }
  .dr-roster-preset:hover, .dr-roster-preset.is-active { color:var(--accent,#38bdf8); border-color:var(--accent,#38bdf8);
    background:color-mix(in srgb,var(--accent,#38bdf8) 9%,transparent); }
  .dr-roster-src { display:flex; align-items:center; gap:8px; margin-bottom:8px; }
  /* Setup source and draft-pick labels mirror the site's canonical .chip. */
  .dr-roster-src-tag, .dr-cap-pill { display:inline-flex; align-items:center; gap:4px;
    background:var(--row); border:1px solid var(--grid); border-radius:6px; padding:2px 8px;
    color:var(--text-muted); font-size:11px; font-weight:700; line-height:1.45; white-space:nowrap; }
  .dr-roster-src-tag { text-transform:none; letter-spacing:normal; }
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
  .dr-cap-row.is-open { background:color-mix(in srgb, var(--accent) 6%, transparent); }
  .dr-cap-rlabel { font-size:11px; font-weight:900; color:var(--text); width:54px; flex-shrink:0;
    letter-spacing:.02em; }
  .dr-cap-rpicks { flex:1; min-width:0; display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
  .dr-cap-none { font-size:11.5px; color:var(--text-muted); opacity:.6; }
  .dr-cap-pill { cursor:pointer; transition:background .12s, color .12s; user-select:none; }
  .dr-cap-pill:hover { background:var(--loss); color:#fff; }
  .dr-cap-pill-x { font-style:normal; font-size:13px; line-height:1; opacity:0; width:0; overflow:hidden;
    transition:opacity .12s, width .12s; }
  .dr-cap-pill:hover .dr-cap-pill-x { opacity:1; width:11px; }
  .dr-cap-pill-traded { background:color-mix(in srgb, var(--warning) 16%, transparent); }
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
  .dr-scar-pos:hover { background: color-mix(in srgb, var(--accent) 7%, transparent); }
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
  .dr-bchip:hover { border-color: var(--accent,#38bdf8); background: color-mix(in srgb, var(--accent) 6%, transparent); }
  .dr-bchip-img { width: 30px; height: 30px; border-radius: 5px 5px 0 0; object-fit: cover;
    object-position: top center; align-self: flex-end; flex-shrink: 0; }
  .dr-bchip-body { min-width: 0; line-height: 1.3; }
  .dr-bchip-name { font-size: 11px; font-weight: 700; color: var(--text); white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; max-width: 68px; }
  .dr-bchip-adp { font-size: 9px; color: var(--text-muted); }
  /* ── Balance alert ── */
  .dr-bal-alert { margin: 8px 10px 2px; padding: 7px 10px; border-radius: 8px; font-size: 11.5px;
    background: color-mix(in srgb, var(--warning) 12%, transparent); color: var(--gold); border: 1px solid color-mix(in srgb, var(--warning) 30%, transparent);
    line-height: 1.4; }
  .dr-bal-alert b { color: var(--warning); }
  /* ── Bye week conflict flag ── */
  .dr-bye-flag { font-size: 9px; font-weight: 800; padding: 1px 5px; border-radius: 4px;
    background: color-mix(in srgb, var(--loss) 14%, transparent); color: var(--loss); margin-left: 5px; white-space: nowrap; }
  /* ── Compare button in rows ── */
  .dr-cmp-btn { background: none; border: none; cursor: pointer; font-size: 10px; font-weight: 800;
    line-height: 1; color: var(--text-muted); padding: 3px 5px; border-radius: 5px;
    border: 1px solid transparent; transition: all .12s; flex-shrink: 0; letter-spacing: .02em; }
  .dr-cmp-btn:hover, .dr-cmp-btn.on { color: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8);
    background: color-mix(in srgb, var(--accent) 10%, transparent); }
  /* ── Player comparison overlay ── */
  .dr-cmp-overlay { position: fixed; inset: 0; z-index: 1000; background: rgba(0,0,0,.45);
    display: flex; align-items: flex-start; justify-content: center; padding: 16px; overflow-y: auto; }
  .dr-cmp-card { position: relative; width: 100%; max-width: 580px; background: var(--card);
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
  .dr-cmp-stat.win { background: color-mix(in srgb, var(--win) 12%, transparent); }
  .dr-cmp-stat.win .dr-cmp-stat-val { color: var(--win); }
  .dr-cmp-actions { display: flex; gap: 8px; justify-content: center; margin-top: 14px; flex-wrap: wrap; }
  /* ── League tab ── */
  .dr-lg-wrap { padding: 10px; display: flex; flex-direction: column; gap: 6px; overflow-y: auto; }
  .dr-lg-row { border: 1px solid var(--border); border-radius: 9px; padding: 9px 10px; background: var(--bg); }
  .dr-lg-mine { border-color: var(--accent,#38bdf8); background: color-mix(in srgb, var(--accent) 5%, transparent); }
  .dr-lg-onclock { border-color: var(--win); background: color-mix(in srgb, var(--win) 5%, transparent); animation: drPulse 1.6s ease-in-out infinite; }
  .dr-lg-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 7px; gap: 6px; }
  .dr-lg-team { font-size: 12px; font-weight: 800; color: var(--text); }
  .dr-lg-next { font-size: 10px; color: var(--text-muted); flex-shrink: 0; }
  .dr-lg-next-you { color: var(--win); font-weight: 700; }
  .dr-lg-pos-row { display: flex; gap: 4px; flex-wrap: wrap; }
  .dr-lg-pos { display: flex; flex-direction: column; align-items: center; padding: 3px 7px;
    border-radius: 6px; border: 1px solid; min-width: 36px; }
  .dr-lg-pos-label { font-size: 8px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; }
  .dr-lg-pos-count { font-size: 11px; font-weight: 700; color: var(--text); margin-top: 1px; }
  .dr-lg-need { font-size: 10px; color: var(--text-muted); margin-top: 6px; }
  .dr-lg-need b { font-weight: 800; }
  .dr-lg-picks { font-size: 10px; color: var(--text-muted); margin-top: 4px; line-height: 1.4; }
  /* ── Roster projection card ── */
  .dr-proj-card { margin: 6px 10px 2px; padding: 10px 12px; border-radius: 10px;
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
  /* Draft recap — biggest steals & reaches across the whole draft */
  .dr-league-body { padding: 8px 14px 14px; }
  .dr-recap { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 18px; margin-bottom: 16px; }
  @media (max-width: 460px) { .dr-recap { grid-template-columns: 1fr; gap: 14px; } }
  .dr-recap-h { display: flex; align-items: center; gap: 6px; font-size: 10px; text-transform: uppercase;
    letter-spacing: .06em; font-weight: 800; color: var(--text-muted); margin: 0 0 8px; }
  .dr-recap-ic { width: 12px; height: 12px; flex-shrink: 0; }
  .dr-recap-grades-h { margin: 4px 0 8px; }
  .dr-recap-row { display: flex; align-items: center; gap: 8px; padding: 5px 0;
    border-bottom: 1px solid var(--border); }
  .dr-recap-row:last-child { border-bottom: none; }
  .dr-recap-pos { font-size: 9px; font-weight: 800; padding: 1px 5px; border-radius: 5px; flex-shrink: 0;
    color: #fff; min-width: 24px; text-align: center; }
  .dr-recap-main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
  .dr-recap-name { font-size: 12px; font-weight: 700; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-recap-sub { font-size: 10px; color: var(--text-muted); }
  .dr-recap-ps { font-size: 13px; font-weight: 900; flex-shrink: 0; font-variant-numeric: tabular-nums; }
  .dr-recap-nums-h { margin: 4px 0 8px; }
  .dr-recap-nums { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
  .dr-recap-tile { background: var(--card-soft, var(--bg)); border: 1px solid var(--border); border-radius: 11px; padding: 10px 11px; }
  .dr-recap-tlbl { font-size: 9.5px; text-transform: uppercase; letter-spacing: .05em; font-weight: 800; color: var(--text-subtle); }
  .dr-recap-tbig { font-size: 14px; font-weight: 800; color: var(--text); margin-top: 2px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-recap-tsub { font-size: 10.5px; color: var(--text-muted); margin-top: 1px; }

  /* League grades list */
  .dr-sum-league { display: flex; flex-direction: column; gap: 3px; }
  .dr-sum-lrow { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-radius: 10px;
    background: var(--bg); border: 1px solid var(--border); cursor: pointer; transition: background .12s; }
  .dr-sum-lrow:hover { background: rgba(127,127,127,.08); }
  .dr-sum-lrow.is-me { border-color: var(--accent,#38bdf8); background: color-mix(in srgb, var(--accent) 8%, transparent); }
  .dr-sum-lrank { width: 20px; flex-shrink: 0; font-size: 12px; font-weight: 900; color: var(--text-muted); text-align: center; }
  .dr-sum-lrank.gold { color: var(--warning); }
  .dr-sum-lrank.silver { color: var(--text-subtle); }
  .dr-sum-lrank.bronze { color: #cd7c2f; }
  .dr-sum-lrank.has-medal { width: 30px; display: inline-flex; align-items: center; justify-content: center; }
  .dr-sum-lname { flex: 1; min-width: 0; font-size: 13px; font-weight: 700; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-sum-lrow.is-me .dr-sum-lname { color: var(--accent,#38bdf8); }
  .dr-sum-lwin { font-size: 9.5px; font-weight: 800; padding: 2px 7px; border-radius: var(--radius-pill, 8px); white-space: nowrap; flex-shrink: 0; border: 1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dr-sum-lgrade { font-size: 18px; font-weight: 900; flex-shrink: 0; width: 32px; text-align: right; }
  /* Projected playoff-odds chip (completed draft only) */
  .dr-sum-lpo { font-size: 11px; font-weight: 800; flex-shrink: 0; width: 38px; text-align: right; font-variant-numeric: tabular-nums; }
  .dr-sum-lpo-pending { color: var(--text-muted); font-weight: 600; }
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
    <p class="dr-sub">Every draft in your league's history. Open any board to review the picks pick-by-pick.</p>
    <div class="dr-hero-actions">
      <a class="dr-hero-link" id="drHistToRoom" href="/draft">&larr; Draft Room</a>
    </div>
  </div>
  <div id="drHistList" class="dr-hist-list">
    <div class="dr-loading"><div class="loading-spinner" style="width:22px;height:22px;"></div><span>Loading…</span></div>
  </div>
</div>

<style>
  .dr-wrap { max-width: 900px; margin: 0 auto; padding: 14px 14px 48px; }
  .dr-hero { margin-bottom: 18px; }
  .dr-title { font-size: clamp(24px,4vw,34px); font-weight: 800; color: var(--text); margin: 0 0 6px; letter-spacing: -0.03em; }
  .dr-sub { font-size: 15px; color: var(--text-muted); margin: 0; line-height: 1.5; max-width: 520px; }
  .dr-hero-actions { display: inline-flex; gap: 8px; margin-top: 12px; }
  .dr-hero-link {
    display: inline-flex; align-items: center; padding: 7px 12px; font-size: 12px; font-weight: 700;
    color: var(--text-muted); text-decoration: none; border: 1px solid var(--border);
    border-radius: var(--radius-pill, 8px); background: color-mix(in srgb, var(--card) 80%, transparent);
  }
  .dr-hero-link:hover {
    color: var(--brand-blue, #3b82f6); border-color: color-mix(in srgb, var(--brand-blue, #3b82f6) 45%, var(--border));
    text-decoration: none;
  }
  .dr-hist-list { display: flex; flex-direction: column; gap: 10px; position: relative; z-index: 1; }
  .dr-hist-card { display: flex; align-items: center; gap: 12px; padding: 14px 16px; border: 1px solid var(--border);
    border-radius: 12px; background: var(--card); box-shadow: var(--shadow-sm, 0 2px 8px rgba(15, 23, 42, 0.05)); }
  .dr-hist-body { flex: 1; min-width: 0; }
  .dr-hist-title { font-size: 15px; font-weight: 700; color: var(--text); }
  .dr-hist-meta { font-size: 12px; color: var(--text-muted); margin-top: 2px; }
  .dr-hist-tag { font-size: 10px; font-weight: 800; text-transform: uppercase; padding: 1px 7px; border-radius: var(--radius-pill, 8px);
    background: color-mix(in srgb, var(--accent) 14%, transparent); color: var(--accent,#38bdf8); margin-right: 6px;
    border: 1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dr-hist-tag-live { background: color-mix(in srgb, var(--loss) 16%, transparent); color: var(--loss); }
  .dr-hist-tag-complete { background: rgba(148,163,184,.16); color: var(--text-subtle); }
  .dr-hist-actions { display: flex; gap: 6px; flex-shrink: 0; }
  .dr-btn { padding: 8px 14px; border-radius: 8px; font-size: 13px; font-weight: 700; cursor: pointer;
    border: 1px solid var(--border); background: var(--bg); color: var(--text); text-decoration: none; }
  .dr-btn-primary { background: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8); color: var(--on-accent, #fff); }
  .dr-btn-danger { color: var(--loss); border-color: color-mix(in srgb, var(--loss) 40%, transparent); background: transparent; }
  .dr-loading {
    display: flex; align-items: center; justify-content: center; gap: 10px;
    padding: 28px 16px; color: var(--text-muted); font-size: 13px;
  }
  .dr-hist-empty {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 6px; padding: 36px 18px; text-align: center; color: var(--text-muted); font-size: 13px;
    line-height: 1.45;
  }
</style>

<script>
(function(){
  var cfg = window.__draftHistCfg || { base: '/draft', hasLeague: false };
  var listEl = document.getElementById('drHistList');
  // Point the hero's Draft Room link at the league-scoped board when available.
  var _toRoom = document.getElementById('drHistToRoom');
  if (_toRoom && cfg.base) _toRoom.setAttribute('href', cfg.base);

  function esc(s){ return String(s == null ? '' : s).replace(/[&<>"]/g, function(c){
    return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'})[c]; }); }

  function statusTag(s){
    var c = (s === 'drafting') ? 'dr-hist-tag-live' : (s === 'complete' ? 'dr-hist-tag-complete' : '');
    var label = (s === 'drafting') ? 'Live now' : (s === 'pre_draft' ? 'Upcoming' : (s === 'complete' ? 'Complete' : (s || '')));
    return '<span class="dr-hist-tag ' + c + '">' + esc(label) + '</span>';
  }

  function histEmpty(title, message, htmlMsg){
    if (htmlMsg) {
      listEl.innerHTML = '<div class="empty-state is-compact">'
        + '<p class="empty-state-title">' + esc(title) + '</p>'
        + '<p class="empty-state-msg">' + htmlMsg + '</p></div>';
      return;
    }
    if (window.brEmptyState) {
      window.brEmptyState(listEl, { icon: 'empty', title: title, message: message, compact: true });
      return;
    }
    listEl.innerHTML = '<div class="empty-state is-compact"><p class="empty-state-title">'
      + esc(title) + '</p><p class="empty-state-msg">' + esc(message) + '</p></div>';
  }

  function render(drafts){
    if (!drafts.length){
      histEmpty('No drafts yet', 'Drafts for this league will show up here once they are created.');
      return;
    }
    // Live/upcoming first, then completed; newest season first within each.
    var rank = { drafting: 0, pre_draft: 1, complete: 2 };
    drafts.sort(function(a, b){
      var ra = (rank[a.status] != null ? rank[a.status] : 3), rb = (rank[b.status] != null ? rank[b.status] : 3);
      if (ra !== rb) return ra - rb;
      return (Number(b.season) || 0) - (Number(a.season) || 0);
    });
    var html = '';
    drafts.forEach(function(d){
      var typeLabel = d.draft_type ? (d.draft_type.charAt(0).toUpperCase() + d.draft_type.slice(1)) : 'Draft';
      var title = (d.season ? (String(d.season) + ' ') : '') + typeLabel + ' Draft'
        + ' · ' + (d.teams || '?') + ' teams · ' + (d.rounds || '?') + ' rounds';
      html += '<div class="dr-hist-card">'
        + '<div class="dr-hist-body"><div class="dr-hist-title">' + esc(title) + ' ' + statusTag(d.status) + '</div>'
        + '<div class="dr-hist-meta">' + esc((d.order || 'snake')) + ' order</div></div>'
        + '<div class="dr-hist-actions">'
        + '<a class="dr-btn dr-btn-primary" href="' + esc(cfg.base) + '?live=' + encodeURIComponent(d.draft_id) + '">Open board</a>'
        + '</div></div>';
    });
    listEl.innerHTML = html;
  }

  function loadList(){
    if (!cfg.hasLeague){
      histEmpty('Open from your league', '',
        'Open Draft History from your league to see its drafts. '
        + 'You can still run a mock in the <a href="' + esc(cfg.base) + '">Draft Room</a>.');
      return;
    }
    fetch('/api/draft/detect?history=1&platform=' + encodeURIComponent(cfg.platform)
        + '&league_id=' + encodeURIComponent(cfg.leagueId) + '&season=' + (cfg.season || ''), { cache: 'no-store' })
      .then(function(r){ return r.json(); })
      .then(function(resp){
        if (resp.unsupported){
          var plat = String(cfg.platform || '').toLowerCase();
          if (plat === 'espn') {
            histEmpty('ESPN drafts are not listed here', 'Open the Draft Room to run a mock. Live ESPN draft boards are not imported into Draft History.');
          } else if (!cfg.hasLeague) {
            histEmpty('Open from your league', '',
              'Open Draft History from your league to see its drafts. '
              + 'You can still run a mock in the <a href="' + esc(cfg.base) + '">Draft Room</a>.');
          } else {
            histEmpty('Sleeper only', 'Draft history is available for Sleeper leagues. Other platforms can still run a mock in the Draft Room.');
          }
          return;
        }
        render(resp.drafts || []);
      })
      .catch(function(){
        if (window.brErrorState) window.brErrorState(listEl, 'Could not load drafts.', loadList, { compact: true });
        else listEl.innerHTML = '<div class="dr-hist-empty">Could not load drafts.</div>';
      });
  }

  loadList();
})();
</script>
"""
