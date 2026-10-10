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

import json
import os
from typing import Optional

# Cache-busting hash for the external Draft Room script (mirrors app.js's ?v=).
# The 4k-line draft IIFE lives in static/draft_room.js so the browser caches it
# across visits instead of re-receiving it inline on every Draft Room load.


def _static_src(name: str) -> str:
    from utils.static_minify import served_name
    return served_name(name)


def _static_v(name: str) -> str:
    from utils.static_minify import served_name, static_hash
    return static_hash(served_name(name))


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
            + f'\n<script src="/static/{_static_src("pick_score.js")}?v={_static_v("pick_score.js")}" defer></script>\n'
            + f'\n<script src="/static/{_static_src("draft_board_core.js")}?v={_static_v("draft_board_core.js")}" defer></script>\n'
            + f'\n<script src="/static/{_static_src("draft_grade_team.js")}?v={_static_v("draft_grade_team.js")}" defer></script>\n'
            + f'\n<script src="/static/{_static_src("draft_room.js")}?v={_static_v("draft_room.js")}" defer></script>\n'
            # Setup wizard (Phase 1 of the draft room visual redesign). Runs after
            # draft_room.js and drives the original setup inputs, so every setting
            # keeps its exact semantics.
            + f'\n<script src="/static/{_static_src("draft_room_wizard.js")}?v={_static_v("draft_room_wizard.js")}" defer></script>\n'
    )


# Plain (non-f) string -- safe to contain { } freely.
_DRAFT_ROOM_HTML = r"""
<div class="dr-wrap">
  <div class="dr-hero" id="drHero">
    <div class="dr-hero-row">
      <div>
        <h1 class="dr-title">Draft Room</h1>
        <p class="dr-sub">Mock against CPU teams, draft manually, or sync a live draft with best-available ranks, tiers, and a live grade.</p>
      </div>
      <div class="dr-hero-actions">
        <a class="dr-hero-link" id="drToCheatSheet" href="/draft/cheat-sheet">Cheat Sheet</a>
        <a class="dr-hero-link" id="drToHistory" href="/draft/history">Draft History</a>
      </div>
    </div>
    <div class="dr-auction-note" id="drAuctionNote" hidden style="margin-top:12px;padding:10px 12px;border-radius:10px;background:var(--accent-soft,rgba(37,99,235,.08));border:1px solid var(--border);font-size:13px;line-height:1.45;color:var(--text);">
      <strong>Auction league detected.</strong> Recommendation Rank and Pick Score still help nominations. Suggested $ amounts are guidance from BR values, not clearing prices. Snake-round draft grades are disabled for auction.
    </div>
  </div>

  <!-- Setup -->
  <!-- Setup: 3-step wizard (Phase 1 visual redesign). All original setup inputs
       stay in the DOM inside .wz-orig (visually hidden); the wizard writes
       through to them so every setting keeps its exact semantics. -->
  <div class="dr-setup" id="drSetup">
    <div class="dr-setup-card wz" id="drSetupCard">
      <header class="dr-setup-modal-head" id="drSetupModalHead" hidden>
        <div>
          <div class="dr-setup-modal-kicker">Current draft</div>
          <h2 class="dr-setup-modal-title" id="drEditTitle">Edit Setup</h2>
        </div>
        <button type="button" class="dr-setup-modal-close" id="drEditClose" aria-label="Close">&times;</button>
      </header>
      <p class="dr-setup-desc" id="drEditNote" hidden>Changes apply to this draft. Picks stay on the board unless you change teams, pick order, or your slot. Reset wipes the board and returns to setup.</p>

      <!-- Presets: collapsed by default -->
      <div class="wz-preset-wrap" id="wzPresetWrap">
        <button type="button" class="wz-preset-head" id="wzPresetHead" aria-expanded="false" aria-controls="wzPresetPanel">
          <span>
            <span class="wz-lt">Start from a preset</span>
            <span class="wz-ls"><span id="wzPresetCurName">Custom</span><span class="wz-opt">optional</span></span>
          </span>
          <svg class="wz-chev" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>
        </button>
        <div class="wz-preset-panel" id="wzPresetPanel">
          <div class="wz-preset-grid" aria-label="Presets">
            <button type="button" class="wz-pcard" data-preset="ppr10">
              <span class="wz-pname">10-Team PPR</span>
              <span class="wz-pdesc">Redraft, 1QB, snake, full PPR</span>
              <span class="wz-ppills"><span>10 teams</span><span>1QB</span><span>15 rds</span><span>Full PPR</span></span>
              <span class="wz-pcheck" aria-hidden="true">&#10003;</span>
            </button>
            <button type="button" class="wz-pcard" data-preset="sf12">
              <span class="wz-pname">12-Team SF TEP</span>
              <span class="wz-pdesc">Redraft, superflex, snake, TE premium</span>
              <span class="wz-ppills"><span>12 teams</span><span>SF</span><span>15 rds</span><span>+1.0 TEP</span></span>
              <span class="wz-pcheck" aria-hidden="true">&#10003;</span>
            </button>
            <button type="button" class="wz-pcard" data-preset="dynasty">
              <span class="wz-pname">Dynasty Startup</span>
              <span class="wz-pdesc">Startup, superflex, snake, full PPR</span>
              <span class="wz-ppills"><span>12 teams</span><span>SF</span><span>25 rds</span><span>Dynasty</span></span>
              <span class="wz-pcheck" aria-hidden="true">&#10003;</span>
            </button>
            <button type="button" class="wz-pcard" data-preset="keeper2">
              <span class="wz-pname">Keeper 2</span>
              <span class="wz-pdesc">Redraft with 2 keepers, half PPR</span>
              <span class="wz-ppills"><span>10 teams</span><span>1QB</span><span>15 rds</span><span>2 keepers</span></span>
              <span class="wz-pcheck" aria-hidden="true">&#10003;</span>
            </button>
            <button type="button" class="wz-pcard" data-preset="custom" hidden>
              <span class="wz-pname">Custom</span>
              <span class="wz-pdesc">Your tweaks, tracked as you go</span>
              <span class="wz-ppills"><span id="wzCustomPills">Custom</span></span>
              <span class="wz-pcheck" aria-hidden="true">&#10003;</span>
            </button>
          </div>
        </div>
      </div>

      <!-- Live draft connect: one button for the connected league -->
      <div class="wz-live-card" id="wzLiveCard">
        <div class="wz-live-simple">
          <span class="wz-livedot" aria-hidden="true"></span>
          <span>
            <span class="wz-lt" id="wzLiveTitle">Sync your live draft</span>
            <span class="wz-ls">Connected league. Setup comes from your live draft.</span>
          </span>
          <button type="button" class="wz-btn" id="wzLiveGo">Connect live draft</button>
        </div>
      </div>
      <div class="dr-live-list wz-live-list" id="drLiveList" style="display:none;"></div>

      <div class="wz-steps" aria-label="Setup progress">
        <div class="wz-step on" id="wzStepDot1"><span class="n">1</span>Format</div>
        <div class="wz-step-line"></div>
        <div class="wz-step" id="wzStepDot2"><span class="n">2</span>League</div>
        <div class="wz-step-line"></div>
        <div class="wz-step" id="wzStepDot3"><span class="n">3</span>Your Picks</div>
      </div>

      <!-- STEP 1: FORMAT -->
      <div class="wz-card" id="wzStep1">
        <h2>Format</h2>
        <p class="wz-sub">What kind of draft this is and how scoring works.</p>

        <div class="wz-field">
          <label id="wzLblDtype">Draft type</label>
          <div class="wz-segrow">
            <div class="wz-seg" data-wz-seg="dtype" role="group" aria-labelledby="wzLblDtype"><button type="button" data-val="startup">Startup Dynasty</button><button type="button" data-val="rookie">Rookie Dynasty</button><button type="button" data-val="redraft">Redraft</button><button type="button" data-val="keeper" id="wzSegKeeper">Keeper</button></div>
          </div>
        </div>

        <div class="wz-keeper-box" id="wzKeeperBox" hidden>
          <div class="wz-field">
            <label id="wzLblKsrc">Keepers source</label>
            <div class="wz-segrow">
              <div class="wz-seg" data-wz-seg="ksrc" role="group" aria-labelledby="wzLblKsrc"><button type="button" data-val="assistant">Use Keeper Assistant</button><button type="button" data-val="manual">Pick my own</button></div>
            </div>
          </div>
          <div class="wz-field">
            <label id="wzLblKeepers">Keepers per team</label>
            <div class="wz-stepper" data-wz-stepper="keepers" data-min="0" data-max="10" role="group" aria-labelledby="wzLblKeepers"><button type="button" data-dir="-1" aria-label="Fewer keepers">&minus;</button><span class="val">2</span><button type="button" data-dir="1" aria-label="More keepers">+</button></div>
            <p class="wz-fhint">Keeper picks are removed from the board before the draft starts.</p>
          </div>
        </div>

        <div class="wz-field-row">
          <div class="wz-field">
            <label id="wzLblQb">QB format</label>
            <div class="wz-segrow">
              <div class="wz-seg" data-wz-seg="qb" role="group" aria-labelledby="wzLblQb"><button type="button" data-val="0">1QB</button><button type="button" data-val="1">Superflex</button></div>
            </div>
          </div>
          <div class="wz-field">
            <label id="wzLblOrder">Pick order</label>
            <div class="wz-segrow">
              <div class="wz-seg" data-wz-seg="order" role="group" aria-labelledby="wzLblOrder"><button type="button" data-val="snake">Snake</button><button type="button" data-val="linear">Linear</button><button type="button" data-val="3rr">3rd Round Reversal</button></div>
            </div>
          </div>
        </div>

        <div class="wz-field-row">
          <div class="wz-field">
            <label id="wzLblPpr">PPR</label>
            <div class="wz-segrow">
              <div class="wz-seg" data-wz-seg="ppr" role="group" aria-labelledby="wzLblPpr"><button type="button" data-val="1">Full</button><button type="button" data-val="0.5">Half</button><button type="button" data-val="0">Standard</button></div>
            </div>
          </div>
          <div class="wz-field">
            <label id="wzLblTep">TE premium</label>
            <div class="wz-segrow">
              <div class="wz-seg" data-wz-seg="tep" role="group" aria-labelledby="wzLblTep"><button type="button" data-val="0">None</button><button type="button" data-val="0.5">+0.5</button><button type="button" data-val="1">+1.0</button></div>
            </div>
          </div>
          <div class="wz-field">
            <label id="wzLblPtd">Passing TDs</label>
            <div class="wz-segrow">
              <div class="wz-seg" data-wz-seg="ptd" role="group" aria-labelledby="wzLblPtd"><button type="button" data-val="4">4 pts</button><button type="button" data-val="6">6 pts</button></div>
            </div>
          </div>
        </div>

        <div class="wz-nav">
          <span class="wz-edit-later">You can change all of this mid-draft.</span>
          <span class="wz-spacer"></span>
          <button type="button" class="wz-btn wz-btn-primary" id="wzToStep2">Continue</button>
        </div>
      </div>

      <!-- STEP 2: LEAGUE AND ROSTER -->
      <div class="wz-card" id="wzStep2" hidden>
        <h2>League</h2>
        <p class="wz-sub">League size, your seat at the table, and roster slots.</p>

        <div class="wz-field-row">
          <div class="wz-field">
            <label id="wzLblTeams">Teams</label>
            <div class="wz-stepper" data-wz-stepper="teams" data-min="6" data-max="16" role="group" aria-labelledby="wzLblTeams"><button type="button" data-dir="-1" aria-label="Fewer teams">&minus;</button><span class="val">12</span><button type="button" data-dir="1" aria-label="More teams">+</button></div>
          </div>
          <div class="wz-field">
            <label id="wzLblRounds">Rounds</label>
            <div class="wz-stepper" data-wz-stepper="rounds" data-min="1" data-max="40" role="group" aria-labelledby="wzLblRounds"><button type="button" data-dir="-1" aria-label="Fewer rounds">&minus;</button><span class="val">15</span><button type="button" data-dir="1" aria-label="More rounds">+</button></div>
          </div>
          <div class="wz-field">
            <label for="drCpuAdpSource">CPU drafts from</label>
            <!-- Original select: draft_room.js rebuilds its options per draft type. -->
            <select class="wz-inline" id="drCpuAdpSource" title="Which ADP source the CPU opponents draft against. Consensus blends every platform. Live (7d) is recent BR Fantasy drafts only.">
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

        <div class="wz-field">
          <label id="wzLblSlot">Your draft slot</label>
          <div class="wz-slots" id="wzSlotPicker" role="radiogroup" aria-labelledby="wzLblSlot"></div>
          <p class="wz-slot-note" id="wzSlotNote"></p>
        </div>

        <div class="wz-field">
          <label id="wzLblRoster">Roster slots</label>
          <div class="wz-roster-lock" id="wzRosterLock" hidden></div>
          <div class="wz-rostergrid" id="wzRosterGrid" role="group" aria-labelledby="wzLblRoster"></div>
          <p class="wz-fhint">Starter slots plus bench. Set SF to 1 for superflex lineups.</p>
        </div>

        <div class="wz-nav">
          <button type="button" class="wz-btn" id="wzBackToStep1">Back</button>
          <span class="wz-spacer"></span>
          <button type="button" class="wz-btn wz-btn-primary" id="wzToStep3">Continue</button>
        </div>
      </div>

      <!-- STEP 3: YOUR PICKS -->
      <div class="wz-card" id="wzStep3" hidden>
        <h2>Your picks</h2>
        <p class="wz-sub">Your draft capital. Tweak it for any traded picks.</p>
        <p class="wz-cap-summary" id="wzCapSummary"></p>
        <div class="wz-picklist" id="wzPickList"></div>
        <p class="wz-fhint">Defaults to your slot's picks for the chosen order. Use x to remove a pick you traded away, or + add pick for one you traded in.</p>
        <div class="dr-setup-cta wz-cta3" id="drSetupStartCta">
          <button type="button" class="wz-btn wz-btn-primary wz-btn-lg" id="drStartSim">Start Mock Draft</button>
          <button type="button" class="wz-btn wz-btn-lg" id="drStart">Draft Manually</button>
          <button type="button" class="wz-btn wz-btn-live" id="drConnect">Connect Live Draft</button>
        </div>
        <div class="wz-nav">
          <button type="button" class="wz-btn" id="wzBackToStep2">Back</button>
          <span class="wz-spacer"></span>
          <span class="wz-edit-later">Edit later in settings.</span>
        </div>
      </div>

      <div class="dr-setup-cta dr-setup-edit-cta" id="drSetupEditCta" hidden>
        <button type="button" class="dr-btn dr-btn-ghost dr-btn-danger" id="drEditReset"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>Reset Draft</button>
        <span class="dr-setup-edit-spacer"></span>
        <button type="button" class="dr-btn dr-btn-ghost" id="drEditCancel">Cancel</button>
        <button type="button" class="dr-btn dr-btn-primary" id="drEditApply">Apply Settings</button>
      </div>

      <p class="wz-resume" id="wzResumeWrap" hidden>Already in a draft? <a href="#" id="wzResume">Resume where you left off</a></p>

      <!-- Original setup controls. Kept in the DOM (visually hidden) so
           draft_room.js keeps working unchanged; the wizard writes through to
           these inputs and reads back the sections it renders into. -->
      <div class="wz-orig" aria-hidden="true">
        <select id="drType" tabindex="-1">
          <option value="startup">Startup (Dynasty)</option>
          <option value="rookie">Rookie (Dynasty)</option>
          <option value="redraft">Redraft</option>
          <option value="keeper">Keeper</option>
        </select>
        <!-- Keeper-only options; shown when Draft Type is Keeper. A keeper
             draft is a redraft where each kept player costs that team the pick
             at his keeper round, so those picks come off the board up front. -->
        <div class="dr-keeper-only" style="display:none;">
          <select id="drKeeperSource" tabindex="-1">
            <option value="assistant">Use Keeper Assistant</option>
            <option value="manual">Pick my own</option>
          </select>
        </div>
        <div class="dr-keeper-only" style="display:none;">
          <input id="drKeeperCount" type="number" min="0" max="10" step="1" value="2" tabindex="-1">
        </div>
        <select id="drSf" tabindex="-1">
          <option value="0">1QB</option>
          <option value="1">Superflex</option>
        </select>
        <select id="drOrder" tabindex="-1">
          <option value="snake">Snake</option>
          <option value="linear">Linear</option>
          <option value="3rr">3rd Round Reversal</option>
        </select>
        <select id="drPpr" tabindex="-1" aria-label="Reception scoring" title="Projected PPG uses this reception scoring (full, half, or standard).">
          <option value="1" selected>Full PPR</option>
          <option value="0.5">Half PPR</option>
          <option value="0">Standard</option>
        </select>
        <select id="drTep" tabindex="-1" aria-label="Tight end premium" title="Projected PPG for tight ends includes this TE premium.">
          <option value="0" selected>None</option>
          <option value="0.5">+0.5 PPR</option>
          <option value="1">+1.0 PPR</option>
        </select>
        <select id="drPassTd" tabindex="-1" aria-label="Points per passing touchdown" title="Adjusts quarterback projected PPG, recommendations, and pick grades">
          <option value="4" selected>4 points</option>
          <option value="6">6 points</option>
        </select>
        <select id="drTeams" tabindex="-1">
          <option value="6">6</option><option value="7">7</option><option value="8">8</option><option value="9">9</option><option value="10">10</option><option value="11">11</option><option value="12" selected>12</option><option value="13">13</option><option value="14">14</option><option value="15">15</option><option value="16">16</option>
        </select>
        <div class="dr-field" id="drRoundsField" style="display:none;">
          <input id="drRounds" type="number" min="1" max="40" value="3" tabindex="-1">
        </div>
        <select id="drSlot" tabindex="-1"></select>
        <div id="drRosterSection"></div>
        <div id="drCapitalSection"></div>
      </div>
    </div>
  </div>

  <!-- Board + side -->
  <div class="dr-main" id="drMain" style="display:none;">
    <div class="dr-start-banner" id="drStartBanner" style="display:none;"></div>
    <div class="dr-start-banner dr-espn-fallback" id="drEspnFallback" style="display:none;" hidden></div>
    <div class="dr-espn-tools" id="drEspnTools" style="display:none;" hidden></div>
    <!-- Slim sticky command bar (Phase 2 draft-view redesign) -->
    <header class="dr-cmdbar" id="drCmdbar">
      <span class="dr-cb-name" id="drDraftName">Draft</span>
      <span class="dr-cb-pills">
        <button type="button" class="dr-league-meta" id="drLeagueMeta" hidden></button>
        <span class="dr-pill dr-pill-live" id="drLiveBadge" style="display:none;">&#9679; LIVE</span>
        <span class="dr-pill dr-pill-upcoming" id="drUpcomingBadge" style="display:none;">Upcoming</span>
        <span class="dr-pill dr-pill-espn" id="drEspnSync" style="display:none;" hidden>ESPN Draft</span>
        <button type="button" class="dr-pill-reconnect" id="drEspnReconnect" style="display:none;" hidden title="Reestablish extension sync">↻ Reconnect</button>
        <span class="dr-poll-status" id="drPollStatus" style="display:none;"></span>
        <span class="dr-save" id="drSave"></span>
      </span>
      <div class="dr-cb-onclock" id="drOnClockWrap">
        <span class="dr-timer-ring" aria-hidden="true">
          <svg viewBox="0 0 40 40"><circle class="dr-tr-bg" cx="20" cy="20" r="16"/><circle class="dr-tr-fg" id="drTimerRingFg" cx="20" cy="20" r="16"/></svg>
          <span class="dr-pick-timer" id="drPickTimer" style="display:none;"></span>
        </span>
        <span class="dr-cb-who">
          <span class="dr-onclock-label">On the clock</span>
          <b id="drOnClock">Team 1</b>
          <span class="dr-cb-sub dr-ss-stat" id="drPickPill">Pick: 1.01</span>
        </span>
      </div>
      <div class="dr-cb-youclock" id="drYouClock" hidden>
        <span class="dr-you-flag">YOU</span>
        <span class="dr-you-txt">YOU ARE ON THE CLOCK</span>
        <span class="dr-timer-ring you" aria-hidden="true">
          <svg viewBox="0 0 40 40"><circle class="dr-tr-bg" cx="20" cy="20" r="16"/><circle class="dr-tr-fg" id="drYouTimerFg" cx="20" cy="20" r="16"/></svg>
          <span class="dr-pick-timer" id="drYouTimer" style="display:none;"></span>
        </span>
      </div>
      <span class="dr-your-next" id="drYourNext" style="display:none;"></span>
      <span class="dr-cb-spacer"></span>
      <button class="dr-btn dr-btn-primary" id="drSimStart" style="display:none;">&#9654;&nbsp; Start Draft</button>
      <button class="dr-btn dr-btn-ghost" id="drSimToggle" style="display:none;">Pause</button>
      <button class="dr-btn dr-btn-ghost" id="drAutoBtn" style="display:none;" title="Auto-draft best available on your picks">Auto Draft</button>
      <button class="dr-btn dr-btn-ghost" id="drPractice" style="display:none;">Practice Mock</button>
      <span class="dr-pill dr-pill-you" id="drNextPill" style="display:none;"></span>
      <span class="dr-pill dr-pill-grade" id="drGradePill" style="display:none;cursor:pointer;" title="View your draft report card"></span>
      <div class="dr-side-opts">
        <button type="button" class="dr-cb-icon" id="drThemeToggle" title="Toggle light/dark mode" aria-label="Toggle theme"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg></button>
        <button type="button" class="dr-cb-icon dr-undo-trigger" id="drUndo" aria-label="Undo last pick" title="Undo last pick"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 14 4 9l5-5"/><path d="M4 9h10.5a5.5 5.5 0 0 1 0 11H11"/></svg></button>
        <a class="dr-opts-trigger dr-cs-trigger" id="drOptsCheatSheet" href="/draft/cheat-sheet" rel="noopener" title="Open your value board / cheat sheet (Cmd/Ctrl-click for a new tab)" aria-label="Cheat Sheet"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="8" y="2" width="8" height="4" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="M9 12h6M9 16h4"/></svg><span class="dr-cs-trigger-lbl">Cheat</span></a>
        <button type="button" class="dr-cb-icon dr-pt-trigger" id="drPickTradeBtn" aria-label="Pick trade evaluator" title="Pick trade evaluator">Trade</button>
        <button type="button" class="dr-cb-icon" id="drShare" aria-label="Share draft board" title="Share draft board"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/></svg></button>
        <button type="button" class="dr-cb-icon" id="drOptsBtn" aria-label="Settings" title="Settings"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/></svg></button>
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
            <button class="dr-btn dr-btn-ghost" id="drEdit"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/></svg>Edit Setup</button>
            <button class="dr-btn dr-btn-ghost dr-btn-danger" id="drReset"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"/></svg>Reset</button>
          </div>
        </div>
      </div>
    </header>

    <!-- Draft progress line -->
    <div class="dr-progressline" id="drProgressLine" aria-label="Draft progress">
      <div class="dr-pl-labels">
        <span class="dr-pl-round" id="drPlRound"></span>
        <span class="dr-pl-you"><i aria-hidden="true"></i><span id="drPlNext"></span></span>
        <span class="dr-progress" id="drProgress"></span>
      </div>
      <div class="dr-pl-track" id="drPlTrack">
        <div class="dr-pl-fill" id="drPlFill"></div>
        <span id="drPlTicks" aria-hidden="true"></span>
        <span class="dr-pl-you-mk" id="drPlYou" title="Your next pick"></span>
      </div>
    </div>

    <div class="dr-cols">
      <section class="dr-panel dr-board-panel" aria-label="Draft board">
        <div class="dr-panel-head">
          <h2>Draft Board</h2>
          <span class="dr-count" id="drBoardCount"></span>
          <span class="dr-panel-kicker" id="drBoardKicker"></span>
          <span class="dr-panel-sp"></span>
          <div class="dr-board-toolbar">
            <div class="dr-cell-toggle" id="drCellToggle" title="Toggle between dynasty value and pick score">
              <span class="dr-ct-opt is-active" data-mode="val">Value</span>
              <span class="dr-ct-opt" data-mode="ps">Pick Score</span>
            </div>
          </div>
        </div>
        <div class="dr-board-scroll"><div class="dr-board" id="drBoard"></div></div>
      </section>
      <section class="dr-panel dr-pool-panel dr-side" id="drSide" aria-label="Player pool">
        <button class="dr-sheet-handle" id="drSheetHandle" aria-label="Resize panel"><span class="dr-sheet-grip"></span></button>
        <div class="dr-panel-head" id="drPoolHead">
          <h2>Best Available</h2>
          <span class="dr-count" id="drBaCount"></span>
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
      </section>
      <!-- Assistant rail: Queue / My Team / Team Needs / League Activity -->
      <aside class="dr-rail" id="drRail" aria-label="Assistant rail">
        <section class="dr-panel" aria-label="Queue">
          <div class="dr-panel-head"><h2>Queue</h2><span class="dr-count" id="drQueueCount"></span></div>
          <div class="dr-queue-list" id="drQueueList"></div>
        </section>
        <section class="dr-panel" aria-label="My team">
          <div class="dr-panel-head"><h2>My Team</h2><span class="dr-count" id="drMyTeamCount"></span></div>
          <div class="dr-myteam" id="drMyTeamList"></div>
        </section>
        <section class="dr-panel" aria-label="Team needs">
          <div class="dr-panel-head"><h2>Team Needs</h2><span class="dr-count" id="drNeedsCount"></span></div>
          <div class="dr-needs-matrix" id="drNeedsMatrix"></div>
        </section>
        <section class="dr-panel" aria-label="League activity">
          <div class="dr-panel-head"><h2>League Activity</h2></div>
          <div class="dr-feed" id="drLeagueFeed"></div>
        </section>
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
  .dr-hero { margin: 2px 0 18px; }
  .dr-hero-row { display: flex; align-items: flex-end; justify-content: space-between; gap: 12px; flex-wrap: wrap; }

  .dr-title {
    font-size: 24px; font-weight: 800; color: var(--text);
    margin: 0 0 4px; letter-spacing: -0.02em; line-height: 1.15;
  }
  .dr-sub {
    font-size: 13px; color: var(--text-muted); margin: 0; max-width: 560px; line-height: 1.5;
  }
  .dr-hero-actions {
    display: inline-flex; flex-wrap: wrap; gap: 8px; justify-content: flex-end; margin: 0; flex-shrink: 0;
  }
  .dr-hero-link {
    display: inline-flex; align-items: center; padding: 7px 12px; font-size: 13px; font-weight: 700;
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
    font-size: 13px; font-weight: 900; color: var(--on-accent, #fff);
    background: var(--accent, #122d4b); flex-shrink: 0;
  }
  .dr-step-title { font-size: 22px; font-weight: 800; color: var(--text); margin: 0; line-height: 1.15; letter-spacing: -0.02em; }
  .dr-setup-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(150px,1fr)); gap: 12px; }
  .dr-field { display: flex; flex-direction: column; gap: 6px; font-size: 13px; font-weight: 700; color: var(--text-muted); }
  .dr-field select, .dr-field input {
    padding: 9px 11px; border-radius: 9px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); font-size: 15px; font-weight: 600; outline: none; min-height: 40px;
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
    font-size: 11px; font-weight: 900; text-transform: uppercase; letter-spacing: .12em;
    color: var(--brand-blue, #3b82f6); margin-bottom: 4px;
  }
  .dr-setup-modal-title { font-size: 22px; font-weight: 900; color: var(--text); margin: 0; line-height: 1.1; }
  .dr-setup-modal-close {
    width: 28px; height: 28px; flex-shrink: 0; background: var(--bg); border: 1px solid var(--border);
    border-radius: 12px; font-size: 18px; line-height: 1; color: var(--text-muted); cursor: pointer;
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
  /* ── Setup wizard (Phase 1 visual redesign) ── */
  .dr-setup-card.wz {
    max-width: 820px; background: var(--wz-panel); border-color: var(--wz-border);
    border-radius: 16px; padding: 0; overflow: visible;
    --wz-panel: #ffffff;
    --wz-panel2: #eef2f8;
    --wz-border: #e2e8f1;
    --wz-text: #17222f;
    --wz-muted: #5d6d86;
    --wz-faint: #7e8da9;
    --wz-accent: #2f6df6;
    --wz-accent-soft: rgba(47,109,246,.10);
    --wz-you: #9a6b0a;
    --wz-you-soft: rgba(190,140,20,.12);
    --wz-good: #1d9e6b;
    --wz-bad: #d64545;
    --wz-shadow: 0 1px 2px rgba(16,24,40,.04),0 8px 24px rgba(16,24,40,.06);
    color: var(--wz-text);
  }
  :root[data-theme="dark"] .dr-setup-card.wz {
    --wz-panel: #151c2c;
    --wz-panel2: #1a2338;
    --wz-border: #26304a;
    --wz-text: #e8ecf5;
    --wz-muted: #93a0bb;
    --wz-faint: #5d6a88;
    --wz-accent: #4f8cff;
    --wz-accent-soft: rgba(79,140,255,.14);
    --wz-you: #ffd166;
    --wz-you-soft: rgba(255,209,102,.08);
    --wz-good: #4fd08d;
    --wz-bad: #f07a7a;
    --wz-shadow: 0 10px 28px rgba(0,0,0,.4);
  }
  .dr-setup-card.wz .dr-setup-modal-head { padding: 22px 22px 0; }
  .dr-setup-card.wz #drEditNote { padding: 0 22px; }
  /* Original controls stay in the DOM for draft_room.js; visually hidden. */
  .wz-orig { position: absolute !important; width: 1px; height: 1px; overflow: hidden;
    clip: rect(0 0 0 0); clip-path: inset(50%); white-space: nowrap; }
  /* Presets: collapsible, collapsed by default */
  .wz-preset-wrap { border: 1px solid var(--wz-border); border-radius: 12px; background: var(--wz-panel);
    margin: 0 22px 18px; box-shadow: var(--wz-shadow); overflow: hidden; }
  .wz-preset-head { display: flex; align-items: center; gap: 12px; padding: 13px 16px; cursor: pointer;
    text-align: left; width: 100%; border: 0; background: none; color: var(--wz-text); font-family: inherit; }
  .wz-preset-wrap:hover .wz-preset-head { background: var(--wz-accent-soft); }
  .wz-preset-head .wz-lt { font-weight: 800; font-size: 14px; display: block; }
  .wz-preset-head .wz-ls { color: var(--wz-muted); font-size: 12.5px; margin-top: 2px; display: block; }
  .wz-preset-head .wz-opt { font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em;
    color: var(--wz-faint); border: 1px solid var(--wz-border); border-radius: 999px; padding: 2px 7px; margin-left: 6px; }
  .wz-preset-head .wz-chev { margin-left: auto; color: var(--wz-faint); transition: transform .18s; flex: none; }
  .wz-preset-wrap.open .wz-chev { transform: rotate(180deg); }
  .wz-preset-panel { max-height: 0; overflow: hidden; transition: max-height .25s ease; }
  .wz-preset-panel.open { max-height: 900px; }
  .wz-preset-grid { display: grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap: 10px; padding: 2px 14px 14px; }
  @media (max-width: 560px){ .wz-preset-grid { grid-template-columns: 1fr; } }
  .wz-pcard { position: relative; text-align: left; border: 1.5px solid var(--wz-border); border-radius: 12px;
    background: var(--wz-panel); padding: 12px 14px; cursor: pointer; box-shadow: var(--wz-shadow);
    display: flex; flex-direction: column; gap: 4px; color: var(--wz-text); font-family: inherit; }
  .wz-pcard:hover { border-color: var(--wz-accent); }
  .wz-pcard.on { border-color: var(--wz-accent); background: var(--wz-accent-soft); }
  .wz-pname { font-size: 13.5px; font-weight: 800; }
  .wz-pdesc { font-size: 11.5px; color: var(--wz-muted); }
  .wz-ppills { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }
  .wz-ppills span { font-size: 10px; font-weight: 700; color: var(--wz-muted); background: var(--wz-panel2);
    border: 1px solid var(--wz-border); border-radius: 999px; padding: 3px 8px; }
  .wz-pcheck { position: absolute; top: 10px; right: 10px; width: 20px; height: 20px; border-radius: 50%;
    background: var(--wz-accent); color: #fff; font-size: 12px; font-weight: 800; display: none;
    align-items: center; justify-content: center; }
  .wz-pcard.on .wz-pcheck { display: flex; }
  /* Live connect card */
  .wz-live-card { border: 1px solid var(--wz-border); border-radius: 12px; background: var(--wz-panel);
    margin: 0 22px 18px; overflow: hidden; }
  /* Prominent only when a draft is actually live (JS adds .is-live). */
  .wz-live-card.is-live { border: 1.5px dashed var(--wz-accent); box-shadow: var(--wz-shadow); }
  .wz-live-card .wz-livedot { display: none; }
  .wz-live-card.is-live .wz-livedot { display: block; }
  .wz-live-simple { display: flex; align-items: center; gap: 12px; padding: 14px 16px; }
  .wz-live-simple .wz-lt { font-weight: 800; font-size: 14px; display: block; color: var(--wz-text); }
  .wz-live-simple .wz-ls { color: var(--wz-muted); font-size: 12.5px; margin-top: 2px; display: block; }
  .wz-live-simple .wz-btn { margin-left: auto; flex: none; }
  .wz-livedot { width: 10px; height: 10px; border-radius: 50%; background: var(--wz-good); flex: none;
    animation: wz-livedotpulse 1.6s ease-in-out infinite; }
  @keyframes wz-livedotpulse { 0%,100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--wz-good) 50%, transparent); } 50% { box-shadow: 0 0 0 8px transparent; } }
  .wz-live-list { margin: 0 22px 18px; }
  .wz-live-list .dr-live-item { display: block; width: 100%; text-align: left; margin-top: 8px; }
  /* Step indicator */
  .wz-steps { display: flex; align-items: center; gap: 0; margin: 0 22px 16px; }
  .wz-step { display: flex; align-items: center; gap: 8px; font-size: 12.5px; font-weight: 700; color: var(--wz-faint); }
  .wz-step .n { width: 26px; height: 26px; border-radius: 50%; border: 2px solid var(--wz-border);
    display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 800;
    background: var(--wz-panel); }
  .wz-step.on { color: var(--wz-text); }
  .wz-step.on .n { border-color: var(--wz-accent); background: var(--wz-accent); color: #fff; }
  .wz-step.done .n { border-color: var(--wz-good); background: var(--wz-good); color: #fff; }
  .wz-step-line { flex: 1; height: 2px; background: var(--wz-border); margin: 0 12px; border-radius: 1px; }
  /* Step cards */
  .wz-card { background: var(--wz-panel); border: 1px solid var(--wz-border); border-radius: 16px;
    box-shadow: var(--wz-shadow); padding: 22px; margin: 0 22px; overflow: hidden; }
  /* Steps are sections of the outer setup card, not nested cards. */
  .dr-setup-card.wz .wz-card {
    background: transparent; border: none; border-radius: 0;
    box-shadow: none; margin: 0 22px; padding: 4px 0 24px; overflow: visible;
  }
  .wz-card h2 { font-size: 16px; font-weight: 800; margin: 0 0 4px; color: var(--wz-text); }
  .wz-card .wz-sub { font-size: 12.5px; color: var(--wz-muted); margin: 0 0 18px; }
  .wz-field { margin-bottom: 18px; }
  .wz-field > label { display: block; font-size: 11px; font-weight: 800; letter-spacing: .06em;
    text-transform: uppercase; color: var(--wz-muted); margin-bottom: 8px; }
  .wz-fhint { font-size: 11px; color: var(--wz-faint); font-weight: 500; margin-top: 10px; line-height: 1.5; }
  .wz-segrow { display: flex; gap: 6px; flex-wrap: wrap; }
  .wz-seg { display: inline-flex; background: var(--wz-panel2); border: 1px solid var(--wz-border);
    border-radius: 999px; padding: 3px; gap: 2px; flex: 1; min-width: 0; }
  .wz-seg button { border: 0; background: transparent; color: var(--wz-muted); font-size: 12.5px; font-weight: 700;
    padding: 9px 6px; border-radius: 999px; cursor: pointer; flex: 1; white-space: nowrap; font-family: inherit; }
  .wz-seg button.on { background: var(--wz-panel); color: var(--wz-text); box-shadow: var(--wz-shadow); }
  .wz-seg button:hover:not(.on) { color: var(--wz-text); }
  .wz-stepper { display: inline-flex; align-items: center; gap: 0; border: 1px solid var(--wz-border);
    border-radius: 10px; overflow: hidden; background: var(--wz-panel2); }
  .wz-stepper button { width: 38px; height: 38px; border: 0; background: transparent; font-size: 18px;
    font-weight: 700; color: var(--wz-muted); cursor: pointer; font-family: inherit; }
  .wz-stepper button:hover { color: var(--wz-accent); background: var(--wz-accent-soft); }
  .wz-stepper .val { min-width: 56px; text-align: center; font-size: 15px; font-weight: 800;
    font-variant-numeric: tabular-nums; color: var(--wz-text); }
  .wz-field-row { display: flex; gap: 18px; flex-wrap: wrap; }
  .wz-field-row .wz-field { flex: 1; min-width: 200px; }
  .wz-keeper-box { border: 1px dashed var(--wz-border); border-radius: 12px; padding: 14px;
    background: var(--wz-panel2); margin-bottom: 18px; }
  .wz-keeper-box .wz-field:last-child { margin-bottom: 0; }
  /* Slot picker */
  .wz-slots { display: grid; grid-template-columns: repeat(10,1fr); gap: 6px; margin-top: 2px; }
  .wz-slotbox { aspect-ratio: 1/1.15; border: 1.5px solid var(--wz-border); border-radius: 9px;
    background: var(--wz-panel2); font-weight: 800; font-size: 13px; color: var(--wz-muted); cursor: pointer;
    display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 1px;
    font-variant-numeric: tabular-nums; font-family: inherit; padding: 0; }
  .wz-slotbox small { font-size: 8.5px; font-weight: 700; letter-spacing: .04em; color: var(--wz-faint); }
  .wz-slotbox:hover { border-color: var(--wz-accent); }
  .wz-slotbox.sel { background: var(--wz-accent); border-color: var(--wz-accent); color: #fff; }
  .wz-slotbox.sel small { color: rgba(255,255,255,.8); }
  .wz-slotbox.random { aspect-ratio: auto; grid-column: span 2; font-size: 11px; }
  .wz-slot-note { font-size: 11.5px; color: var(--wz-muted); margin-top: 8px; }
  .wz-slot-note b { color: var(--wz-text); }
  select.wz-inline { background: var(--wz-panel2); color: var(--wz-text); border: 1px solid var(--wz-border);
    border-radius: 10px; font-size: 13px; font-weight: 600; padding: 10px 12px; min-width: 220px;
    font-family: inherit; }
  /* Roster grid */
  .wz-rostergrid { display: grid; grid-template-columns: repeat(3,minmax(0,1fr)); gap: 10px; }
  .wz-rslot { border: 1px solid var(--wz-border); border-radius: 10px; padding: 8px 10px; display: flex;
    align-items: center; justify-content: space-between; background: var(--wz-panel2); }
  .wz-rslot .wz-rl { font-size: 12px; font-weight: 800; color: var(--wz-muted); letter-spacing: .03em; }
  .wz-mini-stepper { display: inline-flex; align-items: center; border: 1px solid var(--wz-border);
    border-radius: 8px; overflow: hidden; background: var(--wz-panel); }
  .wz-mini-stepper button { width: 28px; height: 30px; border: 0; background: transparent; font-size: 15px;
    font-weight: 700; color: var(--wz-muted); cursor: pointer; font-family: inherit; }
  .wz-mini-stepper button:hover { color: var(--wz-accent); background: var(--wz-accent-soft); }
  .wz-mini-stepper .val { min-width: 34px; text-align: center; font-size: 14px; font-weight: 800;
    font-variant-numeric: tabular-nums; color: var(--wz-text); }
  .wz-mini-stepper button:disabled { opacity: .35; cursor: default; }
  .wz-mini-stepper button:disabled:hover { color: var(--wz-muted); background: transparent; }
  .wz-roster-lock { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; background: var(--wz-panel2);
    border: 1px solid var(--wz-border); border-radius: 10px; padding: 10px 12px; font-size: 12.5px;
    color: var(--wz-muted); margin-bottom: 12px; }
  .wz-roster-lock .wz-linkbtn { margin-left: auto; }
  /* Draft capital */
  .wz-cap-summary { font-size: 13.5px; font-weight: 600; color: var(--wz-text); background: var(--wz-panel2);
    border: 1px solid var(--wz-border); border-radius: 10px; padding: 10px 14px; margin: 0 0 12px; line-height: 1.5; }
  .wz-cap-summary b { color: var(--wz-accent); font-weight: 800; }
  .wz-picklist { border: 1px solid var(--wz-border); border-radius: 12px; overflow: hidden; margin-bottom: 4px;
    max-height: 420px; overflow-y: auto; }
  .wz-prow { display: flex; align-items: center; gap: 12px; padding: 8px 14px; background: var(--wz-panel); }
  .wz-prow:nth-child(even) { background: var(--wz-panel2); }
  .wz-prow .wz-rd { width: 62px; flex: none; font-size: 10.5px; font-weight: 800; color: var(--wz-faint);
    text-transform: uppercase; letter-spacing: .05em; }
  .wz-prow .wz-picks { display: flex; gap: 8px; flex-wrap: wrap; flex: 1; align-items: center; }
  .wz-pbadge { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; font-weight: 800;
    background: var(--wz-panel); border: 1.5px solid var(--wz-accent); color: var(--wz-text); border-radius: 9px;
    padding: 5px 10px; font-variant-numeric: tabular-nums; }
  .wz-pbadge .wz-ov { font-size: 11px; font-weight: 600; color: var(--wz-muted); }
  .wz-pbadge.traded { border-color: var(--wz-border); color: var(--wz-faint); background: transparent; }
  .wz-pbadge.traded .wz-pk { text-decoration: line-through; }
  .wz-pbadge .wz-tag { font-size: 9.5px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em;
    color: var(--wz-accent); }
  .wz-pbadge.extra { border-style: dashed; }
  .wz-prow .wz-acts { margin-left: auto; display: flex; gap: 6px; align-items: center; flex: none; }
  .wz-mini-btn { font-size: 11px; font-weight: 700; border: 1px solid var(--wz-border); background: var(--wz-panel);
    color: var(--wz-muted); border-radius: 8px; padding: 6px 10px; cursor: pointer; white-space: nowrap;
    font-family: inherit; }
  .wz-mini-btn:hover { color: var(--wz-accent); border-color: var(--wz-accent); }
  .wz-mini-btn.undo { color: var(--wz-accent); border-color: var(--wz-accent); }
  .wz-addgrid { display: flex; flex-wrap: wrap; gap: 5px; padding: 4px 14px 12px 88px; background: var(--wz-panel); }
  .wz-addgrid .wz-slotbox { width: 30px; aspect-ratio: 1; font-size: 11px; }
  .wz-addgrid .wz-slotbox.on { background: var(--wz-accent); border-color: var(--wz-accent); color: #fff; }
  .wz-addgrid .wz-slotbox.home { border-style: dashed; }
  /* Nav + CTAs */
  .wz-nav { display: flex; align-items: center; gap: 10px; margin-top: 20px; }
  .wz-nav .wz-spacer { flex: 1; }
  .wz-cta3 { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 20px; }
  .wz-btn { border: 1px solid var(--wz-border); background: var(--wz-panel); color: var(--wz-text);
    border-radius: 10px; padding: 11px 20px; font-size: 13.5px; font-weight: 700; cursor: pointer;
    font-family: inherit; display: inline-flex; align-items: center; gap: 8px; }
  .wz-btn:hover { border-color: var(--wz-faint); }
  .wz-btn-primary { background: var(--wz-accent); border-color: var(--wz-accent); color: #fff;
    box-shadow: var(--wz-shadow); }
  .wz-btn-primary:hover { filter: brightness(1.06); border-color: var(--wz-accent); }
  .wz-btn-lg { padding: 12px 22px; font-size: 14px; }
  .wz-btn-live { border-color: var(--wz-good); color: var(--wz-good); }
  .wz-btn-live:hover { background: color-mix(in srgb, var(--wz-good) 8%, transparent); border-color: var(--wz-good); }
  .wz-btn:disabled { opacity: .55; cursor: default; }
  .wz-edit-later { font-size: 11.5px; color: var(--wz-faint); }
  .wz-linkbtn { border: 0; background: none; color: var(--wz-accent); font-size: 13px; font-weight: 700;
    cursor: pointer; padding: 8px 4px; font-family: inherit; }
  .wz-linkbtn:hover { text-decoration: underline; }
  .wz-resume { text-align: center; margin: 14px 22px 26px; font-size: 12.5px; color: var(--wz-muted); }
  .wz-resume a { color: var(--wz-accent); font-weight: 700; text-decoration: none; }
  .wz-resume a:hover { text-decoration: underline; }
  .wz-connect-loading { display: none; align-items: center; gap: 10px; padding: 0 16px 14px; font-weight: 700;
    font-size: 13.5px; color: var(--wz-text); }
  .wz-connect-loading.show { display: flex; }
  .wz-spinner { width: 16px; height: 16px; border-radius: 50%; border: 2px solid var(--wz-border);
    border-top-color: var(--wz-accent); animation: wz-spin .7s linear infinite; flex: none; }
  @keyframes wz-spin { to { transform: rotate(360deg); } }
  @media (max-width: 760px){
    .wz-slots { grid-template-columns: repeat(5,1fr); }
    .wz-card { padding: 16px; margin: 0 12px; }
    .wz-rostergrid { grid-template-columns: repeat(2,minmax(0,1fr)); }
    .wz-steps { margin-left: 12px; margin-right: 12px; padding-left: 0; padding-right: 0; }
    .wz-preset-wrap, .wz-live-card { margin-left: 12px; margin-right: 12px; }
    .wz-live-list { margin-left: 12px; margin-right: 12px; }
    .wz-resume { margin-left: 12px; margin-right: 12px; }
  }
  body.dr-edit-open { overflow: hidden; }
  .dr-league-meta {
    display: inline-flex; align-items: center; gap: 5px; flex-wrap: nowrap;
    min-width: 0; padding: 3px 6px; border-radius: 8px;
    border: 1px solid transparent; background: transparent; color: var(--text-muted);
    font-family: inherit; font-size: 13px; font-weight: 700; line-height: 1.3; white-space: nowrap;
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
  .dr-btn-lg { padding: 12px 22px; font-size: 15px; border-radius: 10px; }
  .dr-sim-speed { padding: 6px 8px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg);
    color: var(--text); font-size: 13px; font-weight: 600; }
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
  /* Settings gear button -- sits beside the side-panel tabs -- + dropdown panel */
  .dr-side-opts { position: relative; flex: 0 0 auto; display: flex; align-items: stretch; }
  .dr-opts-trigger { display: flex; align-items: center; justify-content: center; gap: 5px;
    background: transparent; border: none; cursor: pointer; color: var(--text-muted);
    font-size: 15px; padding: 0 9px; border-radius: 8px; text-decoration: none; }
  a.dr-opts-trigger { color: var(--text-muted); }
  .dr-cs-trigger { font-size: 13px; font-weight: 700; white-space: nowrap; }
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
  .dr-opts-label { font-size: 11px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
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
    color: var(--text-muted); font-size: 22px; line-height: 1; padding: 0 4px;
  }
  /* ── Slim sticky command bar (Phase 2 draft-view redesign) ── */
  .dr-cmdbar {
    position: sticky; top: 89px; z-index: 50;
    display: flex; align-items: center; gap: 14px;
    height: 58px; padding: 0 14px;
    background: var(--card); border: 1px solid var(--border); border-bottom: none;
    border-radius: 14px 14px 0 0;
    box-shadow: var(--shadow-sm, 0 2px 8px rgba(15, 23, 42, 0.05));
    overflow-x: auto; scrollbar-width: none;
  }
  .dr-cmdbar::-webkit-scrollbar { display: none; }
  .dr-cmdbar > * { flex-shrink: 0; }
  .dr-cb-name { font-weight: 800; font-size: 14px; white-space: nowrap; color: var(--text); }
  .dr-cb-pills { display: flex; align-items: center; gap: 6px; min-width: 0; }
  .dr-cb-onclock { display: flex; align-items: center; gap: 10px; padding-left: 14px;
    border-left: 1px solid var(--border); }
  .dr-cb-who { display: flex; flex-direction: column; line-height: 1.25; min-width: 0; }
  .dr-onclock-label { font-size: 10px; font-weight: 800; text-transform: uppercase; letter-spacing: .07em;
    color: var(--text-muted); }
  .dr-cb-who b { font-size: 14px; font-weight: 800; color: var(--text); white-space: nowrap; }
  .dr-cb-sub { font-size: 11px; color: var(--text-muted); font-weight: 600; white-space: nowrap;
    font-variant-numeric: tabular-nums; }
  .dr-ss-stat { font-size: 11px; font-weight: 700; color: var(--text-muted); white-space: nowrap; }
  /* Pick timer ring */
  .dr-timer-ring { position: relative; width: 38px; height: 38px; flex: none; }
  .dr-timer-ring svg { width: 38px; height: 38px; transform: rotate(-90deg); display: block; }
  .dr-tr-bg { fill: none; stroke: var(--border); stroke-width: 4; }
  .dr-tr-fg { fill: none; stroke: var(--accent,#38bdf8); stroke-width: 4; stroke-linecap: round;
    stroke-dasharray: 100.53; stroke-dashoffset: 0; }
  .dr-timer-ring .dr-pick-timer { position: absolute; inset: 0; display: flex; align-items: center;
    justify-content: center; font-size: 10.5px; font-weight: 800; color: var(--text);
    font-variant-numeric: tabular-nums; }
  .dr-pick-timer.urgent { color: var(--loss); }
  /* You-are-on-the-clock state: pulsing banner swaps in for the on-clock cluster */
  .dr-cb-youclock { display: none; align-items: center; gap: 10px; padding: 7px 14px; border-radius: 10px;
    background: color-mix(in srgb, var(--warning) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--warning) 55%, transparent);
    font-weight: 800; font-size: 13px; color: var(--warning); letter-spacing: .02em; white-space: nowrap;
    animation: drYouPulse 1.4s ease-in-out infinite; }
  .dr-cb-youclock[hidden] { display: none; }
  .dr-cmdbar.dr-you-on .dr-cb-youclock { display: flex; }
  .dr-cmdbar.dr-you-on #drOnClockWrap, .dr-cmdbar.dr-you-on #drYourNext { display: none; }
  .dr-cmdbar.dr-you-on #drSimStart { animation: drYouPulse 1.4s ease-in-out infinite; }
  @keyframes drYouPulse {
    0%,100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--warning) 45%, transparent); }
    50% { box-shadow: 0 0 0 7px transparent; } }
  .dr-you-flag { font-size: 10px; font-weight: 800; letter-spacing: .1em; background: var(--warning);
    color: #fff; border-radius: 6px; padding: 3px 8px; }
  .dr-cb-youclock .dr-tr-fg { stroke: var(--warning); }
  .dr-your-next { font-size: 12px; color: var(--text-muted); white-space: nowrap; }
  .dr-your-next b { color: var(--text); font-variant-numeric: tabular-nums; }
  .dr-cb-spacer { flex: 1; }
  .dr-cb-icon { min-width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center;
    gap: 4px; border: 1px solid var(--border); background: var(--card); border-radius: 10px;
    color: var(--text-muted); cursor: pointer; text-decoration: none; font-size: 12px; font-weight: 700;
    flex: none; padding: 0 8px; }
  .dr-cb-icon:hover, .dr-cb-icon[aria-expanded="true"] { color: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8); }
  a.dr-cb-icon { color: var(--text-muted); }
  .dr-cs-trigger-lbl { line-height: 1; }
  /* Cheat-sheet link keeps its legacy classes (test contract); in the command
     bar it gets the same icon-button treatment as its neighbors. */
  .dr-cmdbar a.dr-opts-trigger.dr-cs-trigger {
    min-width: 36px; height: 36px; display: inline-flex; align-items: center; justify-content: center;
    gap: 4px; border: 1px solid var(--border); background: var(--card); border-radius: 10px;
    color: var(--text-muted); cursor: pointer; text-decoration: none; font-size: 12px; font-weight: 700;
    flex: none; padding: 0 8px; }
  .dr-cmdbar a.dr-opts-trigger.dr-cs-trigger:hover { color: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8); }
  /* ── Draft progress line ── */
  .dr-progressline { position: sticky; top: 147px; z-index: 49; background: var(--card);
    border: 1px solid var(--border); border-radius: 0 0 14px 14px;
    padding: 8px 14px 10px; margin-bottom: 12px;
    box-shadow: var(--shadow-sm, 0 2px 8px rgba(15, 23, 42, 0.05)); }
  .dr-pl-labels { display: flex; justify-content: space-between; align-items: baseline; gap: 10px;
    font-size: 11px; color: var(--text-muted); margin-bottom: 6px;
    font-variant-numeric: tabular-nums; white-space: nowrap; }
  .dr-pl-labels b { color: var(--text); }
  .dr-pl-you { display: inline-flex; align-items: center; gap: 6px; min-width: 0;
    overflow: hidden; text-overflow: ellipsis; }
  .dr-pl-you i { width: 8px; height: 8px; background: var(--warning); border-radius: 2px;
    transform: rotate(45deg); flex: none; }
  .dr-pl-track { position: relative; height: 6px; border-radius: 3px; background: var(--border); }
  .dr-pl-fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px;
    background: var(--accent,#38bdf8); width: 0; transition: width .3s; }
  #drPlTicks { position: absolute; inset: 0; }
  .dr-pl-tick { position: absolute; top: -2px; width: 2px; height: 10px; background: var(--text-muted);
    opacity: .45; border-radius: 1px; transform: translateX(-50%); }
  .dr-pl-you-mk { position: absolute; top: -4px; width: 10px; height: 14px; background: var(--warning);
    border-radius: 3px; transform: translateX(-50%) rotate(45deg); box-shadow: 0 0 0 2px var(--card); }
  .dr-pill { display:inline-flex; align-items:center; font-size:13px; font-weight:700; padding:3px 9px;
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
  /* ESPN/Yahoo sync helpers -- compact promo strip above the status bar */
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
    color: var(--accent,#38bdf8); font-size: 15px;
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
    font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .06em;
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
    display: block; font-size: 13px; color: var(--text-muted); line-height: 1.45;
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
    border: 0; padding: 9px 10px; font-weight: 650; font-size: 13px;
    box-shadow: none; margin-left: auto;
  }
  .dr-espn-tools-actions .dr-banner-join.is-link:hover {
    text-decoration: underline;
    filter: brightness(1.08);
  }
  .dr-espn-tools-actions .dr-banner-join.is-link i { opacity: .75; font-size: 11px; }
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
  .dr-pick-timer { font-size: 15px; font-weight: 800; color: var(--text); font-variant-numeric: tabular-nums;
    min-width: 40px; padding: 2px 8px; border-radius: 7px; background: rgba(127,127,127,.1); text-align: center; }
  .dr-pick-timer.urgent { color: #fff; background: var(--loss); animation: drPulse 1s ease-in-out infinite; }
  .dr-progress { font-size: 13px; color: var(--text-muted); white-space: nowrap; }
  .dr-save { font-size: 11px; color: var(--win); }
  .dr-start-banner { display: flex; align-items: center; gap: 13px; margin: 0 0 12px; padding: 12px 16px; border-radius: 12px;
    background: linear-gradient(90deg, color-mix(in srgb, var(--accent) 18%, transparent), color-mix(in srgb, var(--accent) 5%, transparent)); border: 1px solid var(--accent,#38bdf8); }
  .dr-start-banner.is-live { background: linear-gradient(90deg, color-mix(in srgb, var(--win) 18%, transparent), color-mix(in srgb, var(--win) 5%, transparent)); border-color: var(--win); }
  .dr-banner-ic { font-size: 22px; flex-shrink: 0; display: inline-flex; align-items: center; }
  .dr-banner-ic-live { animation: drPulse 1.4s ease-in-out infinite; }
  .dr-banner-txt { display: flex; flex-direction: column; line-height: 1.35; min-width: 0; flex: 1; }
  .dr-banner-txt b { font-size: 15px; font-weight: 800; color: var(--text); }
  .dr-banner-txt span { font-size: 13px; color: var(--text-muted); }
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
  .dr-live-head { font-size: 13px; font-weight: 700; color: var(--text-muted); }
  .dr-live-item { text-align: left; padding: 9px 12px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); font-size: 13px; cursor: pointer; }
  .dr-live-item:hover { border-color: var(--accent,#38bdf8); }
  .dr-live-status { font-size: 11px; font-weight: 800; text-transform: uppercase; padding: 1px 6px; border-radius: var(--radius-pill, 8px); margin-right: 6px; }
  .dr-ls-drafting { background: color-mix(in srgb, var(--loss) 16%, transparent); color: var(--loss); }
  .dr-ls-pre_draft { background: color-mix(in srgb, var(--warning) 16%, transparent); color: var(--warning); }
  .dr-ls-complete { background: rgba(148,163,184,.16); color: var(--text-subtle); }
  /* ── Draft view: 3-column layout (board / Best Available / assistant rail) ── */
  .dr-cols { display: grid; grid-template-columns: minmax(0,1.35fr) minmax(0,1fr) 300px; gap: 12px; align-items: start; }
  .dr-panel { background: var(--card); border: 1px solid var(--border); border-radius: 14px; overflow: hidden;
    box-shadow: var(--shadow-sm, 0 2px 8px rgba(15, 23, 42, 0.05)); min-width: 0; }
  .dr-panel-head { display: flex; align-items: center; gap: 10px; padding: 10px 14px;
    border-bottom: 1px solid var(--border); flex: none; }
  .dr-panel-head h2 { font-size: 12px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase;
    color: var(--text-muted); margin: 0; }
  .dr-count { font-size: 11px; font-weight: 700; background: var(--bg); border: 1px solid var(--border);
    border-radius: 999px; padding: 2px 8px; color: var(--text); font-variant-numeric: tabular-nums;
    white-space: nowrap; }
  .dr-panel-kicker { font-size: 10.5px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase;
    color: var(--text-muted); white-space: nowrap; }
  .dr-panel-sp { flex: 1; }
  /* min-width:0 lets this grid item shrink to its track instead of growing to
     the wide board's width (the inner scroll, not the card, holds the overflow). */
  .dr-board-panel { position: relative; min-width: 0; }
  .dr-board-panel .dr-board-toolbar { padding: 0; }
  /* The board scrolls on both axes; the header row sticks to the top and the
     round-label column (+ corner cell) sticks to the left. */
  .dr-board-scroll { overflow: auto; min-width: 0; max-height: calc(100vh - 330px); min-height: 300px; }
  .dr-board { display: grid; gap: 5px; min-width: max-content; padding: 8px; }
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
  .dr-cell-rp-ov { font-size: 11px; font-weight: 600; opacity: .55; margin-top: 1px; }
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
  .dr-cell-mineflag { position: absolute; top: 2px; right: 5px; font-size: 11px; font-weight: 800;
    letter-spacing: .04em; color: var(--accent,#38bdf8); }
  .dr-cell-claimed .dr-cell-mineflag { color: var(--warning); }
  /* Keeper: same position tint as a live pick; the KEEP flag is the marker.
     Do not wash the cell green -- that hides WR/QB/TE color. */
  .dr-cell-keepflag { position: absolute; top: 2px; right: 5px; font-size: 11px; font-weight: 800;
    letter-spacing: .04em; color: var(--win,#15803d); }
  /* Traded pick: who the pick was dealt to (shown on another team's seat). */
  .dr-cell-owner { position: absolute; top: 2px; right: 5px; font-size: 11px; font-weight: 800;
    letter-spacing: .04em; color: var(--warning);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; pointer-events: none; }
  .dr-cell-just { animation: drPop .35s ease; }
  @keyframes drPop { 0% { transform: scale(.92); opacity: .3; } 100% { transform: scale(1); opacity: 1; } }
  .dr-cell-val { position: absolute; bottom: 3px; right: 4px; font-size: 11px; font-weight: 800; color: var(--accent,#38bdf8);
    background: color-mix(in srgb, var(--card) 70%, transparent); padding: 0 4px; border-radius: 5px; font-variant-numeric: tabular-nums; }
  .dr-cell-num { position: absolute; top: 2px; left: 5px; font-size: 11px; font-weight: 700; color: var(--text-muted); }
  .dr-board-toolbar { display: flex; align-items: center; justify-content: flex-end; padding: 4px 6px 2px; }
  .dr-cell-toggle { display: flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; font-size: 11px; font-weight: 700; }
  .dr-ct-opt { padding: 3px 9px; cursor: pointer; color: var(--text-muted); transition: background .15s, color .15s; }
  .dr-ct-opt.is-active { background: var(--accent,#38bdf8); color: var(--on-accent, #fff); }
  .dr-ct-opt:not(.is-active):hover { background: var(--bg2,rgba(127,127,127,.12)); color: var(--text); }
  /* ── Headshots: mock .hs treatment (Phase 3) ──
     Initials circle with a position-color ring (--ring) and team tint
     (t-XXX classes below, scoped to the draft room). The <img> overlays the
     circle when the photo loads and removes itself on error, so the styled
     initials are always the fallback. Both themes covered via the
     :root[data-theme="dark"] overrides. */
  .dr-hs { position: relative; width: 40px; height: 40px; flex: none; border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 12px; letter-spacing: .02em;
    border: 2.5px solid var(--ring, var(--text-muted));
    background: var(--tint, rgba(127,127,127,.12)); color: var(--tfg, var(--text)); }
  .dr-hs-ini { line-height: 1; }
  .dr-hs-img { position: absolute; inset: 0; width: 100%; height: 100%; border-radius: 50%;
    object-fit: cover; object-position: top center; }
  .dr-hs-tm { position: absolute; right: -5px; bottom: -5px; font-size: 8.5px; font-weight: 800;
    background: var(--card); border: 1px solid var(--border); border-radius: 4px;
    padding: 0 3px; color: var(--text-muted); line-height: 1.5; }
  .dr-hs-lg { width: 44px; height: 44px; font-size: 13px; }
  .dr-hs-sm { width: 30px; height: 30px; font-size: 10px; border-width: 2px; }
  .dr-hs-sm .dr-hs-tm { display: none; }
  /* Team tints for the headshot circles, all 32 teams. Dark overrides follow. */
  .dr-wrap .t-ARI{--tint:#fbe7e7;--tfg:#8f2323}.dr-wrap .t-ATL{--tint:#fbe7e7;--tfg:#8f2323}
  .dr-wrap .t-BAL{--tint:#ece4f7;--tfg:#4b2d86}.dr-wrap .t-BUF{--tint:#dbe7fb;--tfg:#1a448f}
  .dr-wrap .t-CAR{--tint:#dde7f5;--tfg:#274b7d}.dr-wrap .t-CHI{--tint:#fbe7d9;--tfg:#8f4a1d}
  .dr-wrap .t-CIN{--tint:#fbe7d9;--tfg:#8f4a1d}.dr-wrap .t-CLE{--tint:#f5e6d3;--tfg:#7a4a1a}
  .dr-wrap .t-DAL{--tint:#dde7f5;--tfg:#274b7d}.dr-wrap .t-DEN{--tint:#fbe7d9;--tfg:#8f4a1d}
  .dr-wrap .t-DET{--tint:#dbe7fb;--tfg:#1c4d8f}.dr-wrap .t-GB{--tint:#ddeddf;--tfg:#1f5c2d}
  .dr-wrap .t-HOU{--tint:#dde7f5;--tfg:#274b7d}.dr-wrap .t-IND{--tint:#dbe7fb;--tfg:#1a448f}
  .dr-wrap .t-JAX{--tint:#dcf3f1;--tfg:#0f6b66}.dr-wrap .t-KC{--tint:#fbe3e3;--tfg:#8f1d1d}
  .dr-wrap .t-LAC{--tint:#dbe7fb;--tfg:#1a448f}.dr-wrap .t-LAR{--tint:#dfeafb;--tfg:#1c4d8f}
  .dr-wrap .t-LV{--tint:#e8e8ec;--tfg:#3a3a44}.dr-wrap .t-MIA{--tint:#dcf3f1;--tfg:#0f6b66}
  .dr-wrap .t-MIN{--tint:#ece4f7;--tfg:#4b2d86}.dr-wrap .t-NE{--tint:#dde7f5;--tfg:#274b7d}
  .dr-wrap .t-NO{--tint:#f5edd6;--tfg:#7a5c14}.dr-wrap .t-NYG{--tint:#dde7f5;--tfg:#274b7d}
  .dr-wrap .t-NYJ{--tint:#ddeddf;--tfg:#1f5c2d}.dr-wrap .t-PHI{--tint:#ddeddf;--tfg:#1f5c2d}
  .dr-wrap .t-PIT{--tint:#f5edd6;--tfg:#7a5c14}.dr-wrap .t-SF{--tint:#fbe7e7;--tfg:#8f2323}
  .dr-wrap .t-SEA{--tint:#dcf0e4;--tfg:#14532d}.dr-wrap .t-TB{--tint:#fbe7e7;--tfg:#8f2323}
  .dr-wrap .t-TEN{--tint:#dde7f5;--tfg:#274b7d}.dr-wrap .t-WAS{--tint:#f5e6e6;--tfg:#7d2f2f}
  :root[data-theme="dark"] .dr-wrap .t-ARI{--tint:#3a2326;--tfg:#f0b0b0}:root[data-theme="dark"] .dr-wrap .t-ATL{--tint:#3a2326;--tfg:#f0b0b0}
  :root[data-theme="dark"] .dr-wrap .t-BAL{--tint:#2a2545;--tfg:#c4b5f5}:root[data-theme="dark"] .dr-wrap .t-BUF{--tint:#1d2c47;--tfg:#a9c8f5}
  :root[data-theme="dark"] .dr-wrap .t-CAR{--tint:#1f2a44;--tfg:#a9bfe8}:root[data-theme="dark"] .dr-wrap .t-CHI{--tint:#3d2f1c;--tfg:#f5cf96}
  :root[data-theme="dark"] .dr-wrap .t-CIN{--tint:#3d2f1c;--tfg:#f5cf96}:root[data-theme="dark"] .dr-wrap .t-CLE{--tint:#3a2c1c;--tfg:#e8c896}
  :root[data-theme="dark"] .dr-wrap .t-DAL{--tint:#1f2a44;--tfg:#a9bfe8}:root[data-theme="dark"] .dr-wrap .t-DEN{--tint:#3d2f1c;--tfg:#f5cf96}
  :root[data-theme="dark"] .dr-wrap .t-DET{--tint:#1d2c47;--tfg:#a9c8f5}:root[data-theme="dark"] .dr-wrap .t-GB{--tint:#22382a;--tfg:#a9dcb2}
  :root[data-theme="dark"] .dr-wrap .t-HOU{--tint:#1f2a44;--tfg:#a9bfe8}:root[data-theme="dark"] .dr-wrap .t-IND{--tint:#1d2c47;--tfg:#a9c8f5}
  :root[data-theme="dark"] .dr-wrap .t-JAX{--tint:#1c3833;--tfg:#9fe0d8}:root[data-theme="dark"] .dr-wrap .t-KC{--tint:#3a2326;--tfg:#f0b0b0}
  :root[data-theme="dark"] .dr-wrap .t-LAC{--tint:#1d2c47;--tfg:#a9c8f5}:root[data-theme="dark"] .dr-wrap .t-LAR{--tint:#1d2c47;--tfg:#a9c8f5}
  :root[data-theme="dark"] .dr-wrap .t-LV{--tint:#2b2b33;--tfg:#c9c9d4}:root[data-theme="dark"] .dr-wrap .t-MIA{--tint:#1c3833;--tfg:#9fe0d8}
  :root[data-theme="dark"] .dr-wrap .t-MIN{--tint:#2c2545;--tfg:#c9b3f0}:root[data-theme="dark"] .dr-wrap .t-NE{--tint:#1f2a44;--tfg:#a9bfe8}
  :root[data-theme="dark"] .dr-wrap .t-NO{--tint:#3a301c;--tfg:#f0d896}:root[data-theme="dark"] .dr-wrap .t-NYG{--tint:#1f2a44;--tfg:#a9bfe8}
  :root[data-theme="dark"] .dr-wrap .t-NYJ{--tint:#1f3826;--tfg:#a9dcb2}:root[data-theme="dark"] .dr-wrap .t-PHI{--tint:#1f3826;--tfg:#a9dcb2}
  :root[data-theme="dark"] .dr-wrap .t-PIT{--tint:#3a301c;--tfg:#f0d896}:root[data-theme="dark"] .dr-wrap .t-SF{--tint:#3a2326;--tfg:#f0b0b0}
  :root[data-theme="dark"] .dr-wrap .t-SEA{--tint:#1c3a2c;--tfg:#9fe0b8}:root[data-theme="dark"] .dr-wrap .t-TB{--tint:#3a2326;--tfg:#f0b0b0}
  :root[data-theme="dark"] .dr-wrap .t-TEN{--tint:#1f2a44;--tfg:#a9bfe8}:root[data-theme="dark"] .dr-wrap .t-WAS{--tint:#3d2626;--tfg:#e8a8a8}
  .dr-cell-body { min-width: 0; line-height: 1.2; }
  .dr-cell-name { font-size: 13px; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 96px; }
  .dr-cell-meta { font-size: 11px; color: var(--text-muted); }
  .dr-posbadge { font-size: 11px; font-weight: 700; color: #fff; border-radius: 3px; padding: 1px 4px; }
  .dr-colhead { position: sticky; top: 0; z-index: 3; background: var(--card);
    font-size: 11px; font-weight: 700; color: var(--text-muted); text-align: center; padding: 6px 0;
    white-space: nowrap; border-bottom: 2px solid var(--border); }
  .dr-colhead-you { color: var(--accent,#38bdf8); }
  /* Your seat's column header: a clear YOU pill (plus a star when you only
     hold traded-in picks in the column). */
  .dr-youpill { display: inline-block; font-size: 10px; font-weight: 800; letter-spacing: .1em;
    text-transform: uppercase; background: var(--accent,#38bdf8); color: #fff;
    border-radius: 6px; padding: 3px 10px; }
  /* Round-label column sticks to the left so "R11" stays visible while you
     scroll the board horizontally through team columns. */
  .dr-rowhead { position: sticky; left: 0; z-index: 2; background: var(--card);
    box-shadow: 2px 0 4px -2px rgba(0,0,0,.25);
    display: flex; align-items: center; justify-content: center; }
  /* Corner cell sticks on BOTH axes during horizontal + vertical scroll. */
  .dr-corner { z-index: 4; }
  /* Best Available panel: keeps the old .dr-side id/class so the mobile sheet
     logic keeps working; the tab strip is gone, so this is the player pool. */
  .dr-pool-panel { display: flex; flex-direction: column;
    position: sticky; top: 199px; align-self: start; max-height: calc(100vh - 211px); z-index: 20; }
  /* Slim assistant rail */
  .dr-rail { display: flex; flex-direction: column; gap: 12px; min-width: 0;
    position: sticky; top: 199px; align-self: start; max-height: calc(100vh - 211px);
    overflow-y: auto; scrollbar-width: thin; }
  .dr-rail .dr-panel { flex: none; }
  .dr-queue-list .dr-ba-row { cursor: pointer; }
  .dr-feed { max-height: 260px; overflow-y: auto; }
  .dr-myteam { padding: 2px 0 0; }
  .dr-needs-line { padding: 10px 14px; font-size: 11.5px; color: var(--text-muted);
    border-top: 1px solid var(--border); background: color-mix(in srgb, var(--warning) 8%, transparent); }
  .dr-needs-line b { color: var(--warning); }
  /* Team needs matrix: teams x QB/RB/WR/TE filled dots per drafted starter */
  .dr-needs-matrix { padding: 2px 0 6px; }
  table.dr-matrix { width: 100%; border-collapse: collapse; font-size: 11px; }
  table.dr-matrix th { font-size: 9.5px; font-weight: 800; letter-spacing: .05em; color: var(--text-muted);
    text-transform: uppercase; padding: 6px 4px; border-bottom: 1px solid var(--border); text-align: center; }
  table.dr-matrix th:first-child, table.dr-matrix td:first-child { text-align: left; padding-left: 14px;
    font-weight: 700; color: var(--text); white-space: nowrap; max-width: 104px;
    overflow: hidden; text-overflow: ellipsis; }
  table.dr-matrix td { text-align: center; padding: 6px 4px; border-bottom: 1px solid var(--border); }
  table.dr-matrix tr:last-child td { border-bottom: 0; }
  table.dr-matrix tr.dr-myou td { background: color-mix(in srgb, var(--warning) 10%, transparent); }
  table.dr-matrix tr.dr-myou td:first-child { color: var(--warning); }
  .dr-mdots { display: inline-flex; gap: 3px; }
  .dr-mdot { width: 7px; height: 7px; border-radius: 50%; border: 1.5px solid var(--text-muted); opacity: .45; }
  .dr-mdot.f { background: var(--accent,#38bdf8); border-color: var(--accent,#38bdf8); opacity: 1; }
  /* (draft-view tab strip removed in the Phase 2 redesign; panes render side by side) */
  /* Team needs hover tooltip */
  .dr-team-tip { background: var(--tooltip-bg,var(--card)); color: var(--tooltip-fg,var(--text)); border: 1px solid var(--tooltip-border,var(--border)); border-radius: var(--tooltip-radius,10px);
    padding: 10px 12px; box-shadow: var(--tooltip-shadow,0 8px 28px rgba(0,0,0,.28)); min-width: 160px; }
  .dr-team-tip-name { font-size: 13px; font-weight: 800; color: var(--text); margin-bottom: 7px; }
  .dr-team-tip-pos-row { display: flex; gap: 5px; flex-wrap: wrap; }
  .dr-team-tip-pos { display: flex; flex-direction: column; align-items: center; padding: 4px 7px;
    border-radius: 7px; border: 1px solid transparent; }
  .dr-team-tip-pos-lbl { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .04em; }
  .dr-team-tip-pos-cnt { font-size: 13px; font-weight: 900; line-height: 1.3; }
  .dr-team-tip-next { font-size: 11px; color: var(--text-muted); margin-top: 7px; }
  .dr-team-tip-stats { display: flex; gap: 6px; margin-bottom: 7px; }
  .dr-team-tip-stat { flex: 1; text-align: center; background: var(--bg); border: 1px solid var(--border);
    border-radius: 7px; padding: 5px 6px; }
  .dr-team-tip-stat-v { font-size: 15px; font-weight: 900; color: var(--text); line-height: 1; }
  .dr-team-tip-stat-l { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .03em;
    color: var(--text-muted); margin-top: 3px; }
  .dr-team-tip-picks { display: flex; flex-direction: column; gap: 3px; max-height: 150px; overflow-y: auto; }
  .dr-team-tip-pick { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--text); }
  .dr-team-tip-pick-pos { font-size: 11px; font-weight: 800; padding: 1px 4px; border-radius: 4px; flex-shrink: 0; }
  .dr-team-tip-pick-nm { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-team-tip-pick-tier { font-size: 11px; font-weight: 700; color: var(--accent,#38bdf8); margin-left: auto; flex-shrink: 0; }
  .dr-team-tip-empty { font-size: 11px; color: var(--text-muted); font-style: italic; }
  .dr-side-head { padding: 10px; border-bottom: 1px solid var(--border); display: flex; flex-direction: column; gap: 8px; }
  /* command-center panels (team / runs) */
  .dr-panel { padding: 12px; overflow-y: auto; }
  .dr-roster { padding: 10px; display: flex; flex-direction: column; gap: 6px; overflow-y: auto; }
  .dr-roster-div { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin: 8px 0 2px; }
  .dr-rslot { display: flex; align-items: center; gap: 8px; padding: 4px 8px; border: 1px solid var(--border); border-radius: 8px; background: var(--bg); min-height: 42px; overflow: hidden; }
  .dr-rslot-open { opacity: .65; border-style: dashed; }
  .dr-rslot-pos { width: 36px; flex-shrink: 0; text-align: center; font-size: 11px; font-weight: 800; color: #fff; border-radius: 4px; padding: 3px 0; }
  .dr-rslot-body { flex: 1; min-width: 0; line-height: 1.2; }
  .dr-rslot-name { font-size: 13px; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-rslot-meta { font-size: 11px; color: var(--text-muted); }
  .dr-rslot-val { font-size: 13px; font-weight: 800; color: var(--text); flex-shrink: 0; }
  .dr-rslot-empty { font-size: 13px; color: var(--text-muted); font-style: italic; }
  .dr-run-line { font-size: 13px; color: var(--text-muted); margin-bottom: 10px; }
  .dr-run-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; }
  .dr-run-chip { font-size: 11px; font-weight: 700; padding: 3px 9px; border-radius: var(--radius-pill, 8px); background: rgba(127,127,127,.14); color: var(--text); border: 1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dr-run-hot { background: color-mix(in srgb, var(--loss) 16%, transparent); color: var(--loss); }
  .dr-run-banner { margin: 10px 10px 4px; padding: 8px 10px; border-radius: 8px; font-size: 13px;
    background: color-mix(in srgb, var(--loss) 12%, transparent); color: var(--loss); border: 1px solid color-mix(in srgb, var(--loss) 30%, transparent);
    display: flex; align-items: center; gap: 8px; }
  .dr-run-banner b { color: var(--loss); }
  .dr-run-x { margin-left: auto; border: 0; background: none; color: var(--text-muted); font-size: 16px;
    cursor: pointer; line-height: 1; padding: 2px 4px; flex: none; }
  .dr-run-x:hover { color: var(--loss); }
  .dr-cliff-banner { background: color-mix(in srgb, var(--warning) 12%, transparent); color: var(--warning); border-color: color-mix(in srgb, var(--warning) 35%, transparent); }
  .dr-cliff-banner b { color: var(--warning); }
  .dr-strat-tag { margin-left: 6px; font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: .04em; color: var(--text-muted); border: 1px solid var(--border);
    border-radius: var(--radius-pill, 8px); padding: 1px 7px; vertical-align: middle; white-space: nowrap; }
  /* Pick trade evaluator (inside drModal) */
  .dr-pt-trigger { font-size: 13px; font-weight: 700; white-space: nowrap; }
  .dr-pt-title { font-size: 15px; font-weight: 800; margin-bottom: 4px; }
  .dr-pt-sub { font-size: 13px; color: var(--text-muted); margin-bottom: 12px; line-height: 1.45; }
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
  .dr-pt-pos { font-size: 11px; font-weight: 800; color: #fff; border-radius: 4px; padding: 1px 5px; flex: 0 0 auto; }
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
  .dr-pt-picker .csd-wrap { flex: 1 1 auto; min-width: 0; }
  .dr-pt-add { flex: 0 0 auto; white-space: nowrap; padding: 8px 14px; }
  .dr-pt-chiprow { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; min-height: 20px; margin: 2px 0; }
  .dr-pt-tok { display: inline-flex; align-items: center; gap: 2px; font-size: 13px; font-weight: 800;
    padding: 3px 4px 3px 10px; border-radius: var(--radius-pill, 8px); font-variant-numeric: tabular-nums;
    background: color-mix(in srgb, var(--accent,#38bdf8) 15%, transparent); color: var(--accent,#38bdf8);
    border: 1px solid color-mix(in srgb, currentColor 35%, transparent); }
  .dr-pt-tokx { background: none; border: 0; cursor: pointer; color: inherit; font-size: 15px; line-height: 1; padding: 0 4px; opacity: .75; }
  .dr-pt-tokx:hover { opacity: 1; }
  .dr-pt-bar { display: flex; height: 8px; border-radius: 12px; overflow: hidden; margin: 14px 0 10px; background: var(--border); }
  .dr-pt-bar-g { background: color-mix(in srgb, var(--text-muted) 60%, transparent); }
  .dr-pt-bar-r { background: #22c55e; }
  .dr-pt-verdict { font-size: 15px; font-weight: 800; }
  .dr-pt-vpct { font-size: 13px; font-weight: 700; opacity: .85; }
  .dr-prev-score-hero { border: 1px solid; border-radius: 10px; padding: 12px 10px 10px; margin-bottom: 12px; text-align: center; }
  .dr-prev-score-num { font-size: 44px; font-weight: 900; line-height: 1; }
  .dr-prev-score-lbl { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin-top: 2px; }
  .dr-prev-score-reason { font-size: 13px; font-weight: 600; color: var(--text-muted); margin-top: 6px; }
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
  .dr-empty-note-msg { font-size: 13px; color: var(--text-muted); line-height: 1.45; max-width: 34ch; margin: 0; }
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
  .dr-cheat-pop { margin-left: auto; font-size: 13px; font-weight: 700; color: var(--accent,#38bdf8); text-decoration: none; }
  .dr-cheat-pop:hover { text-decoration: underline; }
  .dr-cheat-close { background: none; border: 0; font-size: 22px; line-height: 1; color: var(--text-muted); cursor: pointer; padding: 0 4px; }
  .dr-cheat-close:hover { color: var(--text); }
  .dr-cheat-frame { display: block; flex: 1 1 auto; width: 100%; min-width: 0; min-height: 0;
    border: 0; background: var(--bg); }
  /* tiers */
  .dr-tier { font-size: 11px; font-weight: 800; padding: 1px 5px; border-radius: var(--radius-pill, 8px);
    background: rgba(127,127,127,.18); color: var(--text-muted); flex-shrink: 0;
    border: 1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dr-tier-cliff { background: color-mix(in srgb, var(--loss) 16%, transparent); color: var(--loss); }
  /* pick score */
  .dr-ba-reason { font-size: 11px; color: var(--accent,#38bdf8); margin-top: 4px; font-weight: 600;
    line-height: 1.35; }
  .dr-ba-recchip { color: var(--accent,#38bdf8); background: color-mix(in srgb, var(--accent) 11%, transparent);
    font-size: 13px; font-weight: 900; }
  .dr-ba-wait { font-size: 11px; color: var(--win); margin-top: 2px; font-weight: 700; }
  .dr-prev-wait { display: flex; align-items: center; gap: 10px; border: 1px solid; border-radius: 9px;
    padding: 9px 12px; margin-bottom: 12px; }
  .dr-prev-wait-p { font-size: 18px; font-weight: 900; flex-shrink: 0; }
  .dr-prev-wait-t { font-size: 13px; font-weight: 600; color: var(--text); line-height: 1.35; }
  /* draft grade */
  .dr-pill-grade { background: color-mix(in srgb, var(--win) 16%, transparent); color: var(--win); }
  .dr-grade-card { display: flex; align-items: center; gap: 12px; padding: 12px; margin: 10px 10px 4px;
    border: 1px solid var(--border); border-radius: 10px; background: var(--bg); }
  .dr-grade-letter { font-size: 28px; font-weight: 900; color: var(--accent,#38bdf8); line-height: 1; min-width: 48px; text-align: center; }
  .dr-grade-mark { display: flex; flex-direction: column; align-items: center; min-width: 48px; flex-shrink: 0; }
  .dr-grade-early { font-size: 11px; font-weight: 800; letter-spacing: .06em; text-transform: uppercase; color: var(--text-muted); margin-top: 3px; }
  .dr-grade-early-inline { font-size: 11px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase; color: var(--text-muted); }
  .dr-grade-meta { flex: 1; min-width: 0; }
  .dr-grade-pace { font-size: 13px; font-weight: 700; color: var(--text); margin-bottom: 6px; }
  .dr-gbar-row { display: flex; align-items: center; gap: 6px; margin-bottom: 3px; }
  .dr-gbar-lbl { font-size: 11px; color: var(--text-muted); width: 76px; flex-shrink: 0; }
  .dr-gbar { flex: 1; height: 6px; border-radius: 12px; background: rgba(127,127,127,.18); overflow: hidden; }
  .dr-gbar-fill { height: 100%; border-radius: 12px; }
  .dr-gbar-pct { font-size: 11px; font-weight: 800; width: 26px; text-align: right; flex-shrink: 0; }
  /* inline info-icon tooltip (ⓘ) */
  .dr-info { display:inline-flex; align-items:center; justify-content:center; width:13px; height:13px; border-radius:50%;
    border:1px solid var(--border); color:var(--text-muted); font-size:11px; font-weight:800; font-style:normal;
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
  .dr-gloss-title { font-size:15px; font-weight:800; color:var(--text); margin:0 0 12px; padding-right:28px; }
  .dr-gloss-close { position:absolute; top:12px; right:12px; width:28px; height:28px; border-radius:8px; border:1px solid var(--border);
    background:var(--bg); color:var(--text-muted); font-size:18px; cursor:pointer; line-height:1; }
  .dr-gloss-item { padding:9px 0; border-top:1px solid var(--border); }
  .dr-gloss-item:first-of-type { border-top:none; }
  .dr-gloss-term { font-size:13px; font-weight:800; color:var(--text); margin-bottom:2px; }
  .dr-gloss-def { font-size:13px; font-weight:500; color:var(--text-muted); line-height:1.45; }
  /* player preview */
  .dr-preview-overlay { position: fixed; inset: 0; z-index: 1000; background: rgba(0,0,0,.45);
    display: flex; align-items: flex-start; justify-content: center; padding: 16px; overflow-y: auto; }
  .dr-preview-card { position: relative; width: 100%; max-width: 420px; background: var(--card);
    border: 1px solid var(--border); border-radius: 16px; padding: 18px 18px 16px; box-shadow: 0 18px 56px rgba(0,0,0,.34); margin: auto; }
  .dr-prev-close { position: absolute; top: 10px; right: 12px; width: 28px; height: 28px; background: var(--bg);
    border: 1px solid var(--border); border-radius: 12px; font-size: 18px; line-height: 1;
    color: var(--text-muted); cursor: pointer; display: flex; align-items: center; justify-content: center;
    transition: background .12s, color .12s; }
  .dr-prev-close:hover { background: color-mix(in srgb, var(--loss) 12%, transparent); color: var(--loss); }
  .dr-prev-top { display: flex; align-items: flex-end; gap: 13px; margin-bottom: 14px; padding-right: 28px; }
  .dr-prev-hs { width: 66px; height: 66px; border-radius: 12px 12px 0 0; object-fit: cover; object-position: top center; background: rgba(127,127,127,.08); flex-shrink: 0; }
  .dr-prev-name { font-size: 18px; font-weight: 800; color: var(--text); line-height: 1.15; }
  .dr-prev-meta { font-size: 13px; color: var(--text-muted); margin-top: 4px; }
  .dr-prev-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; margin-bottom: 14px; }
  .dr-prev-stat { background: var(--bg); border: 1px solid var(--border); border-radius: 9px; padding: 9px 4px 8px; text-align: center; }
  .dr-prev-stat-v { font-size: 15px; font-weight: 800; color: var(--text); letter-spacing: -.01em; line-height: 1; }
  .dr-prev-stat-l { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin-top: 4px; font-weight: 700; }
  .dr-prev-stat-sub { font-size: 11px; color: var(--text-muted); margin-top: 1px; opacity: .7; font-weight: 600; }
  .dr-prev-btns { display: flex; flex-direction: column; gap: 8px; margin-top: 4px; }
  .dr-prev-draft { width: 100%; }
  .dr-prev-profile { display: block; width: 100%; text-align: center; text-decoration: none; box-sizing: border-box; }
  .dr-prev-note { font-size: 13px; color: var(--text-muted); text-align: center; padding: 6px 0; }
  /* queue star */
  .dr-star { background: none; border: none; cursor: pointer; font-size: 15px; line-height: 1; flex-shrink: 0;
    color: var(--text-muted); padding: 2px 2px 0; }
  .dr-star.on { color: var(--warning); }
  .dr-side-title { font-size: 15px; font-weight: 800; color: var(--text); }
  .dr-side-controls { display: flex; gap: 6px; }
  .dr-side-controls input { flex: 1; min-width: 0; padding: 7px 9px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 13px; }
  .dr-side-controls select { padding: 7px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 13px; flex-shrink: 0; max-width: 110px; }
  /* Custom sort dropdown (replaces the native <select> popup, which mis-anchors
     inside the transformed mobile sheet). */
  .dr-sortsel { position: relative; flex-shrink: 0; }
  .dr-sortsel-btn {
    display: flex; align-items: center; gap: 6px; width: 100%;
    padding: 7px 9px; border-radius: 7px; border: 1px solid var(--border);
    background: var(--bg); color: var(--text); font-size: 13px; font-weight: 600;
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
  .dr-adp-src { font-size: 11px; color: var(--text-muted); display: flex; align-items: center; gap: 6px; }
  .dr-adp-src-label { font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .dr-adp-src-select { padding: 4px 7px; border-radius: 7px; border: 1px solid var(--border); background: var(--bg); color: var(--text); font-size: 11px; cursor: pointer; outline: none; }
  .dr-ba-list { overflow-y: auto; flex: 1; }
  .dr-ba-row { display: flex; align-items: center; gap: 10px; padding: 8px 12px 8px 5px; border-bottom: 1px solid var(--border); cursor: pointer; transition: background .12s; }
  .dr-ba-row:hover { background: color-mix(in srgb, var(--accent) 6%, transparent); }
  .dr-ba-body { min-width: 0; flex: 1; line-height: 1.3; }
  .dr-ba-name { font-size: 13px; font-weight: 700; color: var(--text); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-ba-meta { font-size: 11px; color: var(--text-muted); display: flex; align-items: center; gap: 6px; margin-top: 2px; }
  .dr-ba-right { text-align: right; flex-shrink: 0; display: flex; flex-direction: column; align-items: flex-end; gap: 3px; min-width: 52px; }
  .dr-ba-val { font-size: 15px; font-weight: 800; color: var(--text); line-height: 1; }
  .dr-ba-sub { font-size: 11px; color: var(--text-muted); line-height: 1; white-space: nowrap; }
  /* Recommendation-rank badge (mock: big number + "rec" label box) */
  .dr-ba-pschip { flex-shrink: 0; width: 44px; text-align: center; border-radius: 10px; padding: 6px 0;
    font-size: 16px; font-weight: 800; line-height: 1; border: 1px solid var(--border); background: var(--bg); }
  .dr-ba-pschip small { display: block; font-size: 8.5px; font-weight: 700; letter-spacing: .06em;
    text-transform: uppercase; opacity: .75; margin-top: 3px; }
  /* Tier dividers + shading bands in the Best Available list */
  .dr-tier-div { display: flex; align-items: center; gap: 10px; padding: 10px 14px 4px;
    font-size: 10.5px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase;
    color: var(--text-muted); }
  .dr-tier-div::after { content: ""; flex: 1; height: 1px; background: var(--border); }
  .dr-tier-n { font-size: 10px; font-weight: 800; letter-spacing: .02em; text-transform: none;
    background: color-mix(in srgb, var(--accent) 12%, transparent); color: var(--accent,#38bdf8);
    border-radius: 5px; padding: 1px 6px; }
  .dr-ba-row.dr-tier-band { background: color-mix(in srgb, var(--accent) 3.5%, transparent); }
  .dr-ba-row.dr-top3 { background: color-mix(in srgb, var(--accent) 5%, transparent); }
  .dr-ba-row.dr-top3 .dr-ba-recchip { border: 1px solid var(--accent,#38bdf8); }
  .dr-ba-right-col { display: flex; flex-direction: column; align-items: flex-end; gap: 5px; flex-shrink: 0; }
  .dr-ba-metrics { display: flex; align-items: center; gap: 8px; justify-content: flex-end; }
  .dr-ba-actions { display: flex; align-items: center; gap: 6px; }
  .dr-ba-draft { margin-top: 4px; padding: 2px; border: none; background: none; cursor: pointer;
    color: var(--accent,#38bdf8); font-size: 10.5px; font-weight: 800; letter-spacing: .03em;
    white-space: nowrap; }
  .dr-ba-draft:hover { text-decoration: underline; }
  /* ── Player availability indicators (Players tab) ── */
  .dr-ba-row.dr-avail-hi { box-shadow: inset 3px 0 0 var(--win); }
  .dr-ba-row.dr-avail-md { box-shadow: inset 3px 0 0 var(--warning); }
  .dr-ba-row.dr-avail-lo { box-shadow: inset 3px 0 0 var(--loss); }
  .dr-ba-avail { font-size: 11px; font-weight: 700; margin-top: 2px; }
  /* ── Preview modal availability track ── */
  .dr-prev-avail-track { margin-bottom: 12px; }
  .dr-prev-avail-label { font-size: 11px; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }
  .dr-prev-avail-picks { display: flex; flex-wrap: wrap; gap: 6px; }
  .dr-prev-avail-pick { display: inline-flex; align-items: baseline; gap: 4px; padding: 5px 10px; border-radius: 8px; }
  .dr-prev-avail-pn { font-size: 11px; font-weight: 600; color: var(--text-muted); }
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
    .dr-cheat-title { font-size: 15px; }
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
  /* Phase 2: board spans full width on top at <=1100px; pool + rail sit beneath. */
  @media (max-width: 1100px) {
    .dr-cols { grid-template-columns: minmax(0,1fr) 300px; }
    .dr-board-panel { grid-column: 1 / -1; }
    .dr-board-scroll { max-height: 320px; }
    .dr-pool-panel, .dr-rail { position: static; max-height: none; }
    .dr-rail { overflow: visible; }
    .dr-ba-list { max-height: 480px; }
    .dr-feed { max-height: 220px; }
  }
  @media (max-width: 900px) {
    .dr-cols { grid-template-columns: 1fr; padding-bottom: 52vh; }
    .dr-cmdbar { top: 0; height: 52px; }
    .dr-progressline { top: 52px; }
    /* The pool panel becomes a draggable bottom sheet */
    .dr-side {
      /* Anchored to the bottom so a full sheet still covers the tab bar. Height
         is capped below full-viewport so the top of a fully-raised sheet stops
         under the page header + command bar instead of covering them. */
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
    .dr-board-panel { max-width: calc(100vw - 16px); }
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
    /* Command bar: compress to one scrollable row */
    .dr-cmdbar { gap: 8px; padding: 0 10px; }
    .dr-cb-name { font-size: 13px; }
    #drYourNext { display: none; }
    .dr-cb-onclock { padding-left: 8px; gap: 8px; }
    .dr-timer-ring { width: 32px; height: 32px; }
    .dr-timer-ring svg { width: 32px; height: 32px; }
    .dr-cb-icon { min-width: 32px; height: 32px; padding: 0 6px; }
    .dr-cb-icon .dr-cs-trigger-lbl, .dr-cmdbar .dr-cs-trigger-lbl { display: none; }
    .dr-cmdbar .dr-btn { flex: 0 0 auto; padding: 7px 11px; font-size: 13px; }
    .dr-cmdbar .dr-pill { font-size: 11px; padding: 2px 7px; }
    .dr-progressline { padding: 6px 10px 8px; }
    .dr-pl-labels { font-size: 10.5px; }
    .dr-pl-you { display: none; }
    .dr-ss-stat { font-size: 11px; }
    .dr-league-meta { font-size: 11px; padding: 2px 4px; }
    .dr-lm-chip { font-size: 11px; padding: 1px 6px; }
    .dr-pick-timer { font-size: 13px; }
    .dr-progress, .dr-save { font-size: 11px; white-space: nowrap; }
    .dr-board-panel { max-width: calc(100vw - 16px); }
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
  .dr-sum-title { font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:.1em;
    color:var(--text-muted); text-align:center; margin-bottom:10px; }
  .dr-sum-grade-wrap { display:flex; align-items:center; gap:18px; padding-bottom:12px; }
  .dr-sum-grade-ring { width:76px; height:76px; border-radius:50%; border:3px solid;
    display:flex; align-items:center; justify-content:center; flex-shrink:0; }
  .dr-sum-grade { font-size:28px; font-weight:900; line-height:1; }
  .dr-sum-grade-bars { flex:1; display:flex; flex-direction:column; gap:5px; }
  /* Stats strip */
  .dr-sum-stats { display:flex; border-top:1px solid var(--border); border-bottom:1px solid var(--border); flex-shrink:0; }
  .dr-sum-stat { flex:1; text-align:center; padding:10px 4px; }
  .dr-sum-stat:not(:last-child) { border-right:1px solid var(--border); }
  .dr-sum-stat-v { font-size:22px; font-weight:900; color:var(--text); line-height:1; }
  .dr-sum-stat-l { font-size:11px; color:var(--text-muted); margin-top:3px; text-transform:uppercase; letter-spacing:.04em; }
  /* Archetype / window strip */
  .dr-sum-arch { display:flex; align-items:center; justify-content:center; gap:14px; flex-wrap:wrap;
    padding:10px 16px; border-bottom:1px solid var(--border); flex-shrink:0; }
  .dr-sum-arch-item { display:flex; flex-direction:column; align-items:center; gap:4px; }
  .dr-sum-arch-tag { font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:.06em; color:var(--text-muted); }
  .dr-sum-arch-label { font-size:15px; font-weight:900; color:var(--accent); line-height:1.1; }
  .dr-sum-arch-div { width:1px; height:32px; background:var(--border); flex-shrink:0; }
  /* Competitive window chips */
  .dr-sum-win { font-size:13px; font-weight:800; padding:4px 10px; border-radius:var(--radius-pill, 8px); white-space:nowrap; border:1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dr-win-winnow { background:color-mix(in srgb, var(--win) 16%, transparent); color:var(--win); }
  .dr-win-balanced { background:color-mix(in srgb, var(--warning) 16%, transparent); color:var(--warning); }
  .dr-win-future { background:color-mix(in srgb, var(--accent) 16%, transparent); color:var(--accent); }
  /* Roster list scrolls; header + footer stay put. */
  .dr-sum-body-wrap { padding:0 16px 4px; flex:1 1 auto; min-height:0; overflow-y:auto;
    -webkit-overflow-scrolling:touch; overscroll-behavior:contain; }
  .dr-sum-section { font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:.08em;
    color:var(--text-muted); margin:14px 0 6px; }
  .dr-sum-section:first-child { margin-top:10px; }
  /* Player rows */
  .dr-sum-row { display:flex; align-items:center; gap:8px; padding:6px 0; border-bottom:1px solid var(--border); }
  .dr-sum-slot-badge { font-size:11px; font-weight:800; color:#fff; border-radius:4px; padding:3px 0;
    width:34px; flex-shrink:0; text-align:center; }
  .dr-sum-hs { width:30px; height:30px; border-radius:5px 5px 0 0; object-fit:cover;
    object-position:top center; flex-shrink:0; align-self:flex-end; background:transparent; }
  .dr-sum-body { flex:1; min-width:0; line-height:1.3; }
  .dr-sum-name { font-size:13px; font-weight:700; color:var(--text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .dr-sum-meta { font-size:11px; color:var(--text-muted); }
  .dr-sum-reason { font-size:11px; color:var(--text-muted); font-style:italic; }
  .dr-sum-empty { font-size:11px; color:var(--text-muted); font-style:italic; }
  .dr-sum-ps { font-size:15px; font-weight:800; flex-shrink:0; }
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
  .dr-msync-warn { font-size:13px; line-height:1.45; padding:10px 12px; border-radius:10px; margin:0 0 14px;
    background:color-mix(in srgb, var(--warning) 14%, transparent); border:1px solid color-mix(in srgb, var(--warning) 35%, transparent); color:var(--text); }
  .dr-msync-sec { margin:0 0 14px; }
  .dr-msync-sec h4 { font-size:13px; font-weight:800; margin:0 0 6px; color:var(--text); }
  .dr-msync-sec ol { margin:0; padding-left:1.2em; font-size:13px; color:var(--text); line-height:1.55; }
  .dr-msync-sec li { margin:0 0 4px; }
  .dr-msync-sec p { margin:0; font-size:13px; color:var(--text-muted); line-height:1.45; }
  .dr-msync-status { font-size:13px; color:var(--win); min-height:1.2em; margin:4px 0 0; }
  /* Complete-draft sidebar footer */
  #drCompleteBar { padding:10px; border-top:1px solid var(--border); display:flex; flex-direction:column; gap:7px; flex-shrink:0; }
  .dr-btn-deepdive { background:color-mix(in srgb, var(--accent) 12%, transparent); border-color:color-mix(in srgb, var(--accent) 35%, var(--border)); color:var(--accent); font-weight:700; display:flex; align-items:center; justify-content:center; }
  .dr-btn-deepdive:hover { background:color-mix(in srgb, var(--accent) 20%, transparent); }
  .dr-dd-prochip, .dr-sum-prolock { font-size:11px; font-weight:800; letter-spacing:.06em; background:var(--accent); color:var(--on-accent,#fff); border-radius:4px; padding:1px 5px; margin-left:7px; }
  /* ── Deep Dive analyzer ── */
  .dr-dd-overlay { position:fixed; inset:0; z-index:12500; background:rgba(0,0,0,.62); display:flex; align-items:flex-start; justify-content:center; overflow-y:auto; padding:calc(env(safe-area-inset-top) + 14px) 14px calc(env(safe-area-inset-bottom) + 18px); }
  .dr-dd-card { position:relative; width:100%; max-width:940px; margin:0 auto; background:var(--bg); border:1px solid var(--border); border-radius:20px; overflow:hidden; box-shadow:0 24px 80px rgba(0,0,0,.5); display:flex; flex-direction:column; max-height:calc(100vh - 40px); }
  .dr-dd-card .dr-prev-close { z-index:3; }
  .dd-head { padding:20px 22px 16px; border-bottom:1px solid var(--border); background:var(--card); }
  .dd-kicker { font-family:"Archivo",sans-serif; font-size:11px; font-weight:800; letter-spacing:.11em; text-transform:uppercase; color:var(--text-muted); display:flex; align-items:center; }
  .dd-pro { font-size:11px; font-weight:800; letter-spacing:.06em; background:var(--accent); color:var(--on-accent,#fff); border-radius:4px; padding:1px 6px; margin-left:9px; }
  .dd-sub { font-size:13px; color:var(--text-muted); margin-top:4px; }
  .dd-scroll { overflow-y:auto; padding:16px; display:flex; flex-direction:column; gap:14px; }
  .dd-foot { padding:12px 16px; border-top:1px solid var(--border); background:var(--card); display:flex; justify-content:flex-end; }
  .dd-foot .dr-btn { min-width:120px; }
  .dd-card { background:var(--card); border:1px solid var(--border); border-radius:15px; padding:18px; }
  .dd-note { color:var(--text-muted); font-size:13px; }
  .dd-sec { margin-bottom:14px; }
  .dd-sec h4 { margin:0; font-family:"Archivo",sans-serif; font-size:15px; font-weight:800; color:var(--text); }
  .dd-h-sub { display:inline-block; margin-left:8px; font-size:11px; font-weight:700; letter-spacing:.06em;
    text-transform:uppercase; color:var(--text-muted); vertical-align:baseline; position:relative; top:.18em; }
  .dd-sec p { margin:4px 0 0; font-size:13px; color:var(--text-muted); }
  /* overview */
  .dd-ov-top { display:grid; grid-template-columns:auto 1fr; gap:18px 22px; align-items:center; }
  .dd-ring { position:relative; width:104px; height:104px; border-radius:50%; flex:none; background:conic-gradient(var(--gc) calc(var(--pct)*1%), var(--border) 0); display:grid; place-items:center; }
  .dd-ring::after { content:""; position:absolute; inset:8px; border-radius:50%; background:var(--card); }
  .dd-ring b { position:relative; z-index:1; font-family:"Archivo",sans-serif; font-weight:800; font-size:28px; line-height:1; text-align:center; }
  .dd-ring b small { display:block; font-size:13px; color:var(--text-muted); font-weight:600; margin-top:2px; }
  .dd-ov-txt h3 { margin:0; font-family:"Archivo",sans-serif; font-size:22px; font-weight:800; letter-spacing:-.01em; }
  .dd-rankline { margin-top:5px; font-size:13px; color:var(--text-muted); }
  .dd-rankline b { color:var(--text); }
  .dd-say { margin-top:8px; font-size:13px; color:var(--text); border-left:3px solid var(--accent); padding-left:11px; }
  .dd-meters { grid-column:1 / -1; display:flex; flex-direction:column; gap:11px; margin-top:4px; }
  .dd-meter { display:grid; grid-template-columns:150px 1fr auto; gap:13px; align-items:center; }
  .dd-meter-lab { font-size:13px; font-weight:600; }
  .dd-meter-lab small { display:block; font-weight:500; color:var(--text-subtle,var(--text-muted)); font-size:11px; }
  .dd-track { height:8px; border-radius:99px; background:var(--border); overflow:hidden; }
  .dd-track i { display:block; height:100%; border-radius:99px; }
  .dd-meter-val { font-family:"Archivo",sans-serif; font-weight:700; font-size:15px; font-variant-numeric:tabular-nums; text-align:right; white-space:nowrap; }
  .dd-meter-val span { font-size:11px; color:var(--text-muted); font-weight:600; }
  .dd-rankpill { display:inline-block; font-size:11px; font-weight:700; padding:2px 7px; border-radius:var(--radius-pill, 8px); margin-left:6px; border:1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dd-rk-top { background:color-mix(in srgb,#22c55e 16%,transparent); color:#16a34a; }
  .dd-rk-mid { background:color-mix(in srgb,var(--accent) 15%,transparent); color:var(--accent); }
  .dd-rk-low { background:color-mix(in srgb,#ef4444 15%,transparent); color:#dc2626; }
  .dd-tiles { display:grid; grid-template-columns:repeat(4,1fr); gap:11px; margin-top:16px; }
  .dd-tile { border:1px solid var(--border); border-radius:12px; padding:13px 14px; background:var(--bg); }
  .dd-tile-v { font-family:"Archivo",sans-serif; font-weight:800; font-size:22px; line-height:1; font-variant-numeric:tabular-nums; }
  .dd-tile-l { font-size:11px; color:var(--text-muted); margin-top:6px; }
  .dd-tile.good .dd-tile-v { color:#16a34a; } .dd-tile.bad .dd-tile-v { color:#dc2626; }
  /* legend + chart */
  .dd-legend { display:flex; gap:13px; flex-wrap:wrap; font-size:13px; color:var(--text-muted); margin-bottom:10px; }
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
  .dd-tip-opp { margin-top:6px; font-size:11px; color:var(--text); line-height:1.4; }
  /* tables */
  .dd-ledger { width:100%; border-collapse:collapse; font-size:13px; }
  .dd-ledger th, .dd-ledger td { padding:9px 11px; text-align:left; border-bottom:1px solid var(--border); white-space:nowrap; }
  .dd-ledger thead th { font-size:11px; letter-spacing:.05em; text-transform:uppercase; color:var(--text-subtle,var(--text-muted)); cursor:pointer; user-select:none; }
  .dd-ledger thead th:hover { color:var(--text); }
  .dd-ledger thead th.dd-sorted { color:var(--accent); }
  .dd-ledger thead th.r, .dd-ledger tbody td.r { text-align:center; font-variant-numeric:tabular-nums; }
  .dd-ledger .num { font-variant-numeric:tabular-nums; }
  .dd-ledger tbody tr:hover { background:color-mix(in srgb,var(--accent) 5%,transparent); }
  .dd-ledger td.dd-plcell { white-space:normal; min-width:140px; }
  /* Sticky identifying column (Player / Team): the leading Pick / # column scrolls under it. */
  #drDdLedger thead th[data-k="name"],
  #drDdLedger tbody td.dd-plcell,
  .dd-ledger.dd-league thead th:nth-child(2),
  .dd-ledger.dd-league tbody td.dd-plname,
  .dd-ledger.dd-hist-table thead th:nth-child(2),
  .dd-ledger.dd-hist-table tbody td.dd-plname {
    position:sticky; left:0; z-index:2;
    background:var(--card);
    border-right:1px solid var(--border);
  }
  #drDdLedger thead th[data-k="name"],
  .dd-ledger.dd-league thead th:nth-child(2),
  .dd-ledger.dd-hist-table thead th:nth-child(2) { z-index:3; }
  /* Row states, resolved opaque against the card so scrolled columns stay hidden. */
  .dd-ledger tbody tr:hover td.dd-plcell,
  .dd-ledger tbody tr:hover td.dd-plname {
    background:color-mix(in srgb,var(--accent) 5%,var(--card));
  }
  .dd-ledger.dd-league tbody tr.dd-me td.dd-plname {
    background:color-mix(in srgb,var(--accent) 9%,var(--card));
  }
  .dd-plname { font-weight:600; }
  .dd-pl-sub { margin-top:3px; font-size:11px; font-weight:500; color:var(--text-muted); line-height:1.35; max-width:280px; }
  .dd-opp-sev { display:inline-block; margin-left:4px; font-size:11px; font-weight:700; letter-spacing:.02em; text-transform:uppercase; }
  .dd-opp-modest { color:#b45309; }
  .dd-opp-material { color:#c2410c; }
  .dd-opp-severe { color:#dc2626; }
  .dd-facets { display:flex; flex-direction:column; gap:6px; margin-top:14px; }
  .dd-facet { font-size:13px; color:var(--text); line-height:1.4; padding:8px 11px; border-left:3px solid color-mix(in srgb,var(--accent) 55%,var(--border)); background:color-mix(in srgb,var(--accent) 6%,transparent); }
  .dd-facet-line { margin:8px 0 0; font-size:13px; line-height:1.4; }
  .dd-posbadge { display:inline-block; min-width:30px; text-align:center; font-size:11px; font-weight:800; color:#fff; padding:2px 6px; border-radius:5px; }
  .dd-diff { display:inline-block; min-width:6.2ch; text-align:right; font-weight:800;
    font-variant-numeric:tabular-nums; font-feature-settings:"tnum" 1; }
  .dd-diff.p { color:#16a34a; } .dd-diff.n { color:#dc2626; } .dd-diff.z { color:var(--text-muted); }
  .dd-verd { font-size:11px; font-weight:800; padding:3px 9px; border-radius:var(--radius-pill, 8px); border:1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dd-v-steal { background:color-mix(in srgb,#22c55e 16%,transparent); color:#16a34a; }
  .dd-v-value { background:color-mix(in srgb,var(--accent) 14%,transparent); color:var(--accent); }
  .dd-v-fair { background:var(--bg); color:var(--text-muted); border:1px solid var(--border); }
  .dd-v-aggressive { background:color-mix(in srgb,#f59e0b 14%,transparent); color:#d97706; }
  .dd-v-reach { background:color-mix(in srgb,#ef4444 14%,transparent); color:#dc2626; }
  .dd-v-keep { background:color-mix(in srgb,var(--text-muted) 12%,transparent); color:var(--text-muted); border:1px solid var(--border); }
  .dd-v-na { color:var(--text-subtle,var(--text-muted)); }
  /* league board */
  .dd-league tbody tr.dd-me { background:color-mix(in srgb,var(--accent) 9%,transparent); }
  .dd-youtag { font-size:11px; font-weight:800; background:var(--accent); color:var(--on-accent,#fff); border-radius:4px; padding:1px 5px; margin-left:6px; }
  .dd-gletter { font-family:"Archivo",sans-serif; font-weight:800; font-size:15px; }
  .dd-odds { display:flex; align-items:center; gap:8px; min-width:130px; }
  .dd-odds-track { flex:1; height:7px; border-radius:99px; background:var(--border); overflow:hidden; }
  .dd-odds-track i { display:block; height:100%; border-radius:99px; }
  .dd-odds .num { font-variant-numeric:tabular-nums; font-weight:600; font-size:13px; min-width:34px; text-align:right; }
  .dd-odds-pending { color:var(--text-muted); font-size:13px; font-style:italic; }
  /* construction */
  .dd-two { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
  .dd-cap-row { display:grid; grid-template-columns:40px 1fr 78px; gap:11px; align-items:center; margin-bottom:10px; }
  .dd-cap-pos { font-size:13px; font-weight:800; }
  .dd-cap-track { position:relative; height:20px; border-radius:6px; background:var(--border); overflow:visible; }
  .dd-cap-track i { display:block; height:100%; border-radius:6px; }
  .dd-cap-lg { position:absolute; top:-3px; width:2px; height:26px; background:var(--text); opacity:.55; }
  .dd-cap-val { font-family:"Archivo",sans-serif; font-weight:700; font-size:13px; text-align:right; font-variant-numeric:tabular-nums; }
  .dd-cap-val small { display:block; font-family:inherit; font-weight:500; color:var(--text-subtle,var(--text-muted)); font-size:11px; }
  .dd-st-row { display:grid; grid-template-columns:auto 1fr auto auto; gap:10px; align-items:center; padding:7px 0; border-bottom:1px solid var(--border); }
  .dd-slotbadge { font-size:11px; font-weight:800; padding:3px 7px; border-radius:6px; border:1px solid var(--border); min-width:40px; text-align:center; }
  .dd-st-name { font-size:13px; font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .dd-st-ppg { font-family:"Archivo",sans-serif; font-weight:700; font-size:13px; font-variant-numeric:tabular-nums; }
  .dd-st-ppg small { font-family:"Inter",sans-serif; font-weight:500; color:var(--text-muted); font-size:11px; margin-left:2px; }
  .dd-st-rank { font-size:11px; color:var(--text-muted); white-space:nowrap; }
  .dd-st-rank b { color:var(--text); }
  .dd-hist-pct { font-family:"Archivo",sans-serif; font-weight:800; font-variant-numeric:tabular-nums; }
  .dd-hist-pct.is-strong { color:#16a34a; }
  .dd-hist-vs.up { color:#16a34a; font-weight:700; }
  .dd-hist-vs.down { color:#dc2626; font-weight:700; }
  /* Historical trends (Deep Dive) -- scoped so early-ADP “high bar” never reads as a miss */
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
  .dd-hist-stat.is-lead .dd-hist-stat-v { font-size:28px; }
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
    font-size:11px; font-weight:800; letter-spacing:.05em; text-transform:uppercase;
    color:var(--text-muted);
  }
  .dd-hist-callout.is-ahead .dd-hist-callout-k { color:#16a34a; }
  .dd-hist-callout.is-bar .dd-hist-callout-k { color:var(--accent); }
  .dd-hist-callout-pl {
    font-family:"Archivo",sans-serif; font-weight:800; font-size:18px;
    letter-spacing:-.01em; margin-top:6px; color:var(--text);
  }
  .dd-hist-callout-sub { font-size:13px; color:var(--text-muted); margin-top:2px; }
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
    font-size:11px; font-weight:700; letter-spacing:.03em; text-transform:uppercase;
    color:var(--text-muted); line-height:1.25;
  }
  .dd-hist-compare-v {
    font-family:"Archivo",sans-serif; font-weight:800; font-size:18px;
    font-variant-numeric:tabular-nums; letter-spacing:-.02em; color:var(--text);
  }
  .dd-hist-callout-say {
    margin:10px 0 0; font-size:13px; line-height:1.45; color:var(--text-muted);
  }
  /* overflow-x must stay auto -- overflow:hidden here used to clip the
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
    display:inline-block; min-width:7.5ch; font-size:13px; font-weight:700;
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
  .dd-edge-k { font-size:11px; font-weight:800; letter-spacing:.04em; text-transform:uppercase; }
  .dd-edge.win .dd-edge-k { color:#16a34a; } .dd-edge.winb .dd-edge-k { color:var(--accent); } .dd-edge.bad .dd-edge-k { color:#dc2626; }
  .dd-edge-pl { font-family:"Archivo",sans-serif; font-weight:700; font-size:15px; margin-top:7px; }
  .dd-edge-sub { font-size:13px; color:var(--text-muted); margin-top:2px; }
  .dd-edge-say { font-size:13px; color:var(--text); margin-top:8px; }
  .dd-flags { display:flex; flex-direction:column; gap:9px; margin-top:13px; }
  .dd-flag { display:grid; grid-template-columns:auto 1fr; gap:11px; padding:12px 13px; border-radius:11px; border:1px solid var(--border); background:var(--bg); }
  .dd-flag-ic { width:30px; height:30px; border-radius:8px; display:grid; place-items:center; font-weight:800; flex:none; }
  .dd-flag-crit { border-color:color-mix(in srgb,#ef4444 40%,var(--border)); }
  .dd-flag-crit .dd-flag-ic { background:color-mix(in srgb,#ef4444 15%,transparent); color:#dc2626; }
  .dd-flag-warn .dd-flag-ic { background:color-mix(in srgb,#f59e0b 16%,transparent); color:#d97706; }
  .dd-flag-ttl { font-weight:700; font-size:13px; }
  .dd-flag-ds { font-size:13px; color:var(--text-muted); margin-top:2px; }
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
    background:var(--card); color:var(--text); font-size:15px; font-weight:700; cursor:pointer; line-height:1;
    display:flex; align-items:center; justify-content:center; padding:0; flex-shrink:0; }
  .dr-step-btn:hover { border-color:var(--accent,#38bdf8); color:var(--accent,#38bdf8); }
  .dr-step-val { font-size:15px; font-weight:800; color:var(--text); min-width:18px; text-align:center; }
  .dr-step-val-ro { font-size:15px; font-weight:800; color:var(--text-muted); min-width:18px; text-align:center; }
  .dr-roster-presets { display:flex; align-items:center; gap:6px; flex-wrap:wrap; margin-bottom:8px; }
  .dr-roster-presets-label { font-size:11px; font-weight:800; color:var(--text-muted); text-transform:uppercase; letter-spacing:.05em; margin-right:2px; }
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
  .dr-cap-none { font-size:11px; color:var(--text-muted); opacity:.6; }
  .dr-cap-pill { cursor:pointer; transition:background .12s, color .12s; user-select:none; }
  .dr-cap-pill:hover { background:var(--loss); color:#fff; }
  .dr-cap-pill-x { font-style:normal; font-size:13px; line-height:1; opacity:0; width:0; overflow:hidden;
    transition:opacity .12s, width .12s; }
  .dr-cap-pill:hover .dr-cap-pill-x { opacity:1; width:11px; }
  .dr-cap-pill-traded { background:color-mix(in srgb, var(--warning) 16%, transparent); }
  .dr-cap-addbtn { width:26px; height:26px; flex-shrink:0; border:none; border-radius:7px;
    background:rgba(127,127,127,.1); color:var(--text-muted); font-size:18px; font-weight:600; line-height:1;
    cursor:pointer; display:flex; align-items:center; justify-content:center; padding:0; transition:all .12s; }
  .dr-cap-addbtn:hover { background:var(--accent,#38bdf8); color:#fff; }
  .dr-cap-row.is-open .dr-cap-addbtn { background:var(--accent,#38bdf8); color:#fff; }
  .dr-cap-picker { display:flex; align-items:center; gap:10px; padding:4px 8px 10px 72px; }
  .dr-cap-picker-lbl { font-size:11px; font-weight:700; color:var(--text-muted); text-transform:uppercase;
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
  .dr-cap-latecount { flex:1; text-align:left; font-size:11px; color:var(--text-muted); }
  .dr-cap-chev { font-style:normal; font-size:11px; color:var(--text-muted); }
  .dr-cap-latebody { padding-bottom:2px; }
  /* ── Team tab PS badge ── */
  .dr-rslot-ps { font-size:11px; font-weight:800; flex-shrink:0; margin-right:2px; }
  /* ── Positional scarcity bar ── */
  .dr-scarcity { display: flex; border-bottom: 1px solid var(--border); }
  .dr-scar-pos { flex: 1; display: flex; flex-direction: column; align-items: center; padding: 5px 2px;
    cursor: pointer; transition: background .12s; }
  .dr-scar-pos:hover { background: color-mix(in srgb, var(--accent) 7%, transparent); }
  .dr-scar-pos:not(:last-child) { border-right: 1px solid var(--border); }
  .dr-scar-count { font-size: 15px; font-weight: 900; line-height: 1; }
  .dr-scar-label { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; color: var(--text-muted); margin-top: 1px; }
  /* ── Best-at-position chips + T1-2 counts ── */
  .dr-bchips-header { display: flex; align-items: center; justify-content: space-between;
    padding: 5px 10px 4px; cursor: pointer; transition: background .12s; border-bottom: 1px solid var(--border); }
  .dr-bchips-header:hover { background: rgba(127,127,127,.05); }
  .dr-bchips-label { font-size: 11px; font-weight: 800; text-transform: uppercase;
    letter-spacing: .06em; color: var(--text-muted); }
  .dr-bchips-hint { font-size: 11px; color: var(--text-muted); opacity: .7; }
  .dr-bchips-section-title { font-size: 11px; font-weight: 700; text-transform: uppercase;
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
  .dr-bchip-adp { font-size: 11px; color: var(--text-muted); }
  /* ── Balance alert ── */
  .dr-bal-alert { margin: 8px 10px 2px; padding: 7px 10px; border-radius: 8px; font-size: 11px;
    background: color-mix(in srgb, var(--warning) 12%, transparent); color: var(--gold); border: 1px solid color-mix(in srgb, var(--warning) 30%, transparent);
    line-height: 1.4; }
  .dr-bal-alert b { color: var(--warning); }
  /* ── Bye week conflict flag ── */
  .dr-bye-flag { font-size: 11px; font-weight: 800; padding: 1px 5px; border-radius: 4px;
    background: color-mix(in srgb, var(--loss) 14%, transparent); color: var(--loss); margin-left: 5px; white-space: nowrap; }
  /* ── Compare button in rows ── */
  .dr-cmp-btn { background: none; border: none; cursor: pointer; font-size: 11px; font-weight: 800;
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
  .dr-cmp-name { font-size: 13px; font-weight: 800; color: var(--text); line-height: 1.2; }
  .dr-cmp-meta { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
  .dr-cmp-ps { font-size: 28px; font-weight: 900; line-height: 1; text-align: center; margin: 6px 0 0; }
  .dr-cmp-ps-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em;
    color: var(--text-muted); text-align: center; margin-bottom: 8px; }
  .dr-cmp-stats { display: flex; flex-direction: column; gap: 3px; }
  .dr-cmp-stat { display: flex; justify-content: space-between; align-items: center; padding: 3px 5px;
    border-radius: 5px; }
  .dr-cmp-stat-lbl { font-size: 11px; color: var(--text-muted); font-weight: 600; }
  .dr-cmp-stat-val { font-size: 13px; font-weight: 800; color: var(--text); }
  .dr-cmp-stat.win { background: color-mix(in srgb, var(--win) 12%, transparent); }
  .dr-cmp-stat.win .dr-cmp-stat-val { color: var(--win); }
  .dr-cmp-actions { display: flex; gap: 8px; justify-content: center; margin-top: 14px; flex-wrap: wrap; }
  /* ── League tab ── */
  .dr-lg-wrap { padding: 10px; display: flex; flex-direction: column; gap: 6px; overflow-y: auto; }
  .dr-lg-row { border: 1px solid var(--border); border-radius: 9px; padding: 9px 10px; background: var(--bg); }
  .dr-lg-mine { border-color: var(--accent,#38bdf8); background: color-mix(in srgb, var(--accent) 5%, transparent); }
  .dr-lg-onclock { border-color: var(--win); background: color-mix(in srgb, var(--win) 5%, transparent); animation: drPulse 1.6s ease-in-out infinite; }
  .dr-lg-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 7px; gap: 6px; }
  .dr-lg-team { font-size: 13px; font-weight: 800; color: var(--text); }
  .dr-lg-next { font-size: 11px; color: var(--text-muted); flex-shrink: 0; }
  .dr-lg-next-you { color: var(--win); font-weight: 700; }
  .dr-lg-pos-row { display: flex; gap: 4px; flex-wrap: wrap; }
  .dr-lg-pos { display: flex; flex-direction: column; align-items: center; padding: 3px 7px;
    border-radius: 6px; border: 1px solid; min-width: 36px; }
  .dr-lg-pos-label { font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: .05em; }
  .dr-lg-pos-count { font-size: 11px; font-weight: 700; color: var(--text); margin-top: 1px; }
  .dr-lg-need { font-size: 11px; color: var(--text-muted); margin-top: 6px; }
  .dr-lg-need b { font-weight: 800; }
  .dr-lg-picks { font-size: 11px; color: var(--text-muted); margin-top: 4px; line-height: 1.4; }
  /* ── Roster projection card ── */
  .dr-proj-card { margin: 6px 10px 2px; padding: 10px 12px; border-radius: 10px;
    border: 1px solid var(--border); background: var(--bg); }
  .dr-proj-title { font-size: 11px; font-weight: 800; text-transform: uppercase;
    letter-spacing: .06em; color: var(--text-muted); margin-bottom: 8px; }
  .dr-proj-stats { display: flex; gap: 10px; }
  .dr-proj-stat { flex: 1; text-align: center; }
  .dr-proj-val { font-size: 18px; font-weight: 900; color: var(--text); line-height: 1; }
  .dr-proj-lbl { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
  .dr-proj-bar-wrap { margin-top: 8px; }
  .dr-proj-bar-bg { height: 5px; border-radius: 3px; background: rgba(127,127,127,.15); overflow: hidden; }
  .dr-proj-bar-fill { height: 100%; border-radius: 3px; background: var(--accent,#38bdf8); }
  .dr-proj-bar-lbl { font-size: 11px; color: var(--text-muted); margin-top: 3px; }
  /* Draft recap -- biggest steals & reaches across the whole draft */
  .dr-league-body { padding: 8px 14px 14px; }
  .dr-recap { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 18px; margin-bottom: 16px; }
  @media (max-width: 460px) { .dr-recap { grid-template-columns: 1fr; gap: 14px; } }
  .dr-recap-h { display: flex; align-items: center; gap: 6px; font-size: 11px; text-transform: uppercase;
    letter-spacing: .06em; font-weight: 800; color: var(--text-muted); margin: 0 0 8px; }
  .dr-recap-ic { width: 12px; height: 12px; flex-shrink: 0; }
  .dr-recap-grades-h { margin: 4px 0 8px; }
  .dr-recap-row { display: flex; align-items: center; gap: 8px; padding: 5px 0;
    border-bottom: 1px solid var(--border); }
  .dr-recap-row:last-child { border-bottom: none; }
  .dr-recap-pos { font-size: 11px; font-weight: 800; padding: 1px 5px; border-radius: 5px; flex-shrink: 0;
    color: #fff; min-width: 24px; text-align: center; }
  .dr-recap-main { flex: 1; min-width: 0; display: flex; flex-direction: column; }
  .dr-recap-name { font-size: 13px; font-weight: 700; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-recap-sub { font-size: 11px; color: var(--text-muted); }
  .dr-recap-ps { font-size: 13px; font-weight: 900; flex-shrink: 0; font-variant-numeric: tabular-nums; }
  .dr-recap-nums-h { margin: 4px 0 8px; }
  .dr-recap-nums { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
  .dr-recap-tile { background: var(--card-soft, var(--bg)); border: 1px solid var(--border); border-radius: 11px; padding: 10px 11px; }
  .dr-recap-tlbl { font-size: 11px; text-transform: uppercase; letter-spacing: .05em; font-weight: 800; color: var(--text-subtle); }
  .dr-recap-tbig { font-size: 15px; font-weight: 800; color: var(--text); margin-top: 2px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-recap-tsub { font-size: 11px; color: var(--text-muted); margin-top: 1px; }

  /* League grades list */
  .dr-sum-league { display: flex; flex-direction: column; gap: 3px; }
  .dr-sum-lrow { display: flex; align-items: center; gap: 8px; padding: 8px 10px; border-radius: 10px;
    background: var(--bg); border: 1px solid var(--border); cursor: pointer; transition: background .12s; }
  .dr-sum-lrow:hover { background: rgba(127,127,127,.08); }
  .dr-sum-lrow.is-me { border-color: var(--accent,#38bdf8); background: color-mix(in srgb, var(--accent) 8%, transparent); }
  .dr-sum-lrank { width: 20px; flex-shrink: 0; font-size: 13px; font-weight: 900; color: var(--text-muted); text-align: center; }
  .dr-sum-lrank.gold { color: var(--warning); }
  .dr-sum-lrank.silver { color: var(--text-subtle); }
  .dr-sum-lrank.bronze { color: #cd7c2f; }
  .dr-sum-lrank.has-medal { width: 30px; display: inline-flex; align-items: center; justify-content: center; }
  .dr-sum-lname { flex: 1; min-width: 0; font-size: 13px; font-weight: 700; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-sum-lrow.is-me .dr-sum-lname { color: var(--accent,#38bdf8); }
  .dr-sum-lwin { font-size: 11px; font-weight: 800; padding: 2px 7px; border-radius: var(--radius-pill, 8px); white-space: nowrap; flex-shrink: 0; border: 1px solid color-mix(in srgb, currentColor 30%, transparent); }
  .dr-sum-lgrade { font-size: 18px; font-weight: 900; flex-shrink: 0; width: 32px; text-align: right; }
  /* Projected playoff-odds chip (completed draft only) */
  .dr-sum-lpo { font-size: 11px; font-weight: 800; flex-shrink: 0; width: 38px; text-align: right; font-variant-numeric: tabular-nums; }
  .dr-sum-lpo-pending { color: var(--text-muted); font-weight: 600; }
  .dr-sum-lchev { font-size: 11px; color: var(--text-muted); flex-shrink: 0; transition: transform .2s; }
  .dr-sum-lrow.is-open .dr-sum-lchev { transform: rotate(180deg); }
  /* Expandable team starter detail */
  .dr-sum-ldtl { display: none; padding: 4px 6px 8px 38px; }
  .dr-sum-ldtl.is-open { display: block; }
  .dr-sum-ldtl-row { display: flex; align-items: center; gap: 6px; padding: 3px 0; }
  .dr-sum-ldtl-slot { font-size: 11px; font-weight: 800; color: #fff; border-radius: 3px; padding: 2px 0;
    width: 28px; flex-shrink: 0; text-align: center; }
  .dr-sum-ldtl-name { flex: 1; font-size: 11px; font-weight: 600; color: var(--text);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dr-sum-ldtl-pick { font-size: 11px; color: var(--text-muted); flex-shrink: 0; }
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
  .dr-title { font-size: 24px; font-weight: 800; color: var(--text); margin: 0 0 4px; letter-spacing: -0.02em; }
  .dr-sub { font-size: 15px; color: var(--text-muted); margin: 0; line-height: 1.5; max-width: 520px; }
  .dr-hero-actions { display: inline-flex; gap: 8px; margin-top: 12px; }
  .dr-hero-link {
    display: inline-flex; align-items: center; padding: 7px 12px; font-size: 13px; font-weight: 700;
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
  .dr-hist-meta { font-size: 13px; color: var(--text-muted); margin-top: 2px; }
  .dr-hist-tag { font-size: 11px; font-weight: 800; text-transform: uppercase; padding: 1px 7px; border-radius: var(--radius-pill, 8px);
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
