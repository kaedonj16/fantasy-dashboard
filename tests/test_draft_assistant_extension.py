"""Guards for the docked Draft Assistant overlay in the League Connector."""
from pathlib import Path
import json
import re

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "extension"


def test_overlay_is_mv3_safe_extension_page():
    html = (EXT / "overlay.html").read_text(encoding="utf-8")
    assert 'src="overlay.js"' in html
    assert 'src="overlay_score.js"' in html
    assert 'src="pick_score.js"' in html
    assert 'src="draft_board_core.js"' in html
    assert "BROverlayScore" in (EXT / "overlay_score.js").read_text(encoding="utf-8")
    assert "BRPickScore" in (EXT / "overlay_score.js").read_text(encoding="utf-8")
    assert "decisionScore" in (EXT / "overlay_score.js").read_text(encoding="utf-8")
    assert "futurePickDecisionScore" not in (EXT / "overlay_score.js").read_text(encoding="utf-8")
    assert "return ctx.current || 1;" in (EXT / "overlay_score.js").read_text(encoding="utf-8")
    assert "Gone before #" not in (EXT / "overlay_score.js").read_text(encoding="utf-8")
    assert 'href="overlay.css"' in html
    assert 'class="br-da-embed"' in html
    assert not re.search(r"<script>(?!\s*</script>)", html)
    assert (EXT / "overlay.js").is_file()
    assert (EXT / "overlay.css").is_file()
    js = (EXT / "overlay.js").read_text(encoding="utf-8")
    assert "ingestLive" in js
    assert "isCompletedHostPick" in js
    assert '__br: "br-da"' in js
    assert "never submits" in js.lower() or "never submit" in js.lower()


def test_manifest_docks_overlay_on_host_drafts():
    manifest = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "1.5.23"
    hosts = " ".join(manifest.get("host_permissions") or [])
    assert "sleeper.app" in hosts
    assert "api.sleeper.app" in hosts
    war = manifest.get("web_accessible_resources") or []
    resources = " ".join(" ".join(block.get("resources") or []) for block in war)
    assert "overlay.html" in resources
    sleeper_js = None
    for block in manifest["content_scripts"]:
        joined = " ".join(block.get("matches") or [])
        if "sleeper.com/draft" in joined or "sleeper.app/draft" in joined:
            sleeper_js = block["js"]
    assert sleeper_js == ["draft_slot.js", "assistant_inject.js", "sleeper_draft.js"]
    inject = (EXT / "assistant_inject.js").read_text(encoding="utf-8")
    sleeper = (EXT / "sleeper_draft.js").read_text(encoding="utf-8")
    espn_iso = (EXT / "espn_draft.js").read_text(encoding="utf-8")
    yahoo_iso = (EXT / "yahoo_draft.js").read_text(encoding="utf-8")
    assert "overlay.html" in inject
    assert "__brDaPushPicks" in inject
    assert "never submits" in inject.lower() or "Reads host picks" in inject
    assert "api.sleeper.app/v1/draft" in sleeper
    assert "__brDaPushPicks" in sleeper
    assert "brfantasy:assistant-reconnect" in espn_iso
    assert "brfantasy:assistant-reconnect" in yahoo_iso
    assert "feedAssistant" in espn_iso
    assert "feedAssistant" in yahoo_iso
    assert "br-fantasy-espn-sync-chip" in inject
    csp = (manifest.get("content_security_policy") or {}).get("extension_pages") or ""
    assert "script-src 'self'" in csp
    assert "unsafe-eval" not in csp
    assert "img-src" in csp
    assert "sleepercdn.com" in csp
    assert "espncdn.com" in csp
    assert "brfantasyfootball.com" in csp


def test_overlay_uses_live_br_player_pool_and_headshots():
    overlay = (EXT / "overlay.js").read_text(encoding="utf-8")
    css = (EXT / "overlay.css").read_text(encoding="utf-8")
    html = (EXT / "overlay.html").read_text(encoding="utf-8")
    inject = (EXT / "assistant_inject.js").read_text(encoding="utf-8")
    background = (EXT / "background.js").read_text(encoding="utf-8")
    sleeper = (EXT / "sleeper_draft.js").read_text(encoding="utf-8")
    assert "ingestPool" in overlay
    assert 'msg.type === "pool"' in overlay
    assert "Loading BR Fantasy ranks" in overlay
    assert "has-photo" in overlay
    assert "sleepercdn.com/content/nfl/players/" in overlay
    assert "has-photo" in css
    assert "object-fit: cover" in css
    assert "fetchDraftPool" in inject
    assert "adpSource" in inject
    assert "queuedPool" in inject
    assert "/api/league-players" in background
    assert "adp_source=" in background
    assert "espnHeadshot" in background
    assert "redraft_value_1qb" in background
    assert "redraft_avg_pick" in background
    assert "compactDraftPlayer" in background
    assert "rank_change_7d" in background
    assert "BROverlayScore.rankPool" in overlay
    assert "sleepercdn.com/content/nfl/players/" in background
    assert "slots_super_flex" in sleeper
    assert "adpSel" in html
    assert "ADP source" in html
    assert 'data-link="room"' not in html
    assert "Open Draft Room" not in html
    assert 'aria-label="Draft Room"' not in html
    assert 'data-link="sheet"' in html
    assert "draft-cta" not in overlay
    assert "draft-cta" not in css
    assert "[data-draft]" not in overlay
    assert "dr-ba-draft" not in overlay
    assert 'id="ovOtc"' in html
    assert "boardControls" in html
    assert "searchInp" in html
    assert "Players" in html
    assert "Recommendation Rank" in html
    assert 'loading="lazy"' in overlay
    assert "liveFingerprint" in overlay
    assert "hadPool" in inject
    assert 'postToHost("open"' in overlay
    assert 'postToHost("adp"' in overlay
    assert 'msg.type === "adp"' in inject
    assert "adp_source_options" in background
    assert "adpOptionsFromBody" in background


def test_collapsed_overlay_has_reopen_control():
    inject = (EXT / "assistant_inject.js").read_text(encoding="utf-8")
    overlay = (EXT / "overlay.js").read_text(encoding="utf-8")
    html = (EXT / "overlay.html").read_text(encoding="utf-8")
    css = (EXT / "overlay.css").read_text(encoding="utf-8")
    assert "br-fantasy-assistant-expand" in inject
    assert "Open Draft Assistant" in inject
    assert "setCollapsed(false)" in inject
    assert "html.br-da-collapsed" in inject
    assert "translateX" in inject
    assert "applyDockShift" in inject
    assert "data-br-da-shifted" in inject
    assert "transition:transform" in inject.replace(" ", "")
    assert "br-da-ready" in inject
    assert "prefers-reduced-motion" in inject
    assert "collapseBtn" in html
    assert html.index('id="reconnectBtn"') < html.index('class="ov-foot"')
    assert "sync-cluster" in html
    assert "sync-reconnect" in css
    assert "setCollapsedUi" in overlay
    assert 'msg.type === "collapsed"' in overlay
    assert "br-da-rail" in css
    assert "br-da-rail" in overlay


def test_overlay_uses_site_logo():
    html = (EXT / "overlay.html").read_text(encoding="utf-8")
    css = (EXT / "overlay.css").read_text(encoding="utf-8")
    inject = (EXT / "assistant_inject.js").read_text(encoding="utf-8")
    pack = (EXT / "pack_extension.py").read_text(encoding="utf-8")
    manifest = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    assert 'src="icons/br-logo.png"' in html
    assert 'src="icons/br-logo-dark.png"' in html
    assert "site-logo-light" in css
    assert "site-logo-dark" in css
    assert "icons/br-logo-dark.png" in inject
    assert "chrome.runtime.getURL" in inject
    assert (EXT / "icons/br-logo.png").is_file()
    assert (EXT / "icons/br-logo-dark.png").is_file()
    assert "icons/br-logo.png" in pack
    resources = " ".join(
        " ".join(block.get("resources") or [])
        for block in (manifest.get("web_accessible_resources") or [])
    )
    assert "icons/br-logo-dark.png" in resources
    assert "icons/br-logo.png" in resources


def test_overlay_autodetects_slot_and_keeps_header_on_one_line():
    overlay = (EXT / "overlay.js").read_text(encoding="utf-8")
    html = (EXT / "overlay.html").read_text(encoding="utf-8")
    css = (EXT / "overlay.css").read_text(encoding="utf-8")
    helper = (EXT / "draft_slot.js").read_text(encoding="utf-8")
    espn_iso = (EXT / "espn_draft.js").read_text(encoding="utf-8")
    yahoo_iso = (EXT / "yahoo_draft.js").read_text(encoding="utf-8")
    espn_main = (EXT / "espn_draft_main.js").read_text(encoding="utf-8")
    yahoo_main = (EXT / "yahoo_draft_main.js").read_text(encoding="utf-8")
    sleeper = (EXT / "sleeper_draft.js").read_text(encoding="utf-8")
    assert "function detectDomSlot" in helper
    assert "function detectYahooSlot" in helper
    assert "function slotFromYahooClock" in helper
    assert "function slotFromTeamId" in helper
    assert "function compactSync" in helper
    assert "YOU " in helper
    assert "resolveMySlot" in espn_iso
    assert "resolveMySlot" in yahoo_iso
    assert "compactSync" in espn_iso
    assert "compactSync" in yahoo_iso
    assert "compactSync" in sleeper
    assert "rememberEspnUser" in espn_main
    assert "computeMySlot" in espn_main
    assert "view=mTeam" in espn_main
    assert "rememberYahooUser" in yahoo_main
    assert "computeMySlot" in yahoo_main
    assert "state.slotAuto" in overlay
    assert "formatSyncChip" in overlay
    assert "YOU " in overlay
    assert "rp.teamId || ownerOf" not in overlay
    assert "ownerOf(pn)" in overlay
    assert 'id="slotLab"' in html
    assert "flex-wrap: nowrap" in css
    assert "white-space: nowrap" in css
    assert "BR Fantasy · connected" not in overlay
    assert "160 PICKS" not in html.upper()
    assert "\u2014" not in overlay
    assert "\u2014" not in html
    assert "\u2014" not in (EXT / "popup.html").read_text(encoding="utf-8")
    assert "\u2014" not in (EXT / "popup.js").read_text(encoding="utf-8")
    assert "\u2014" not in (EXT / "content.js").read_text(encoding="utf-8")
    assert "\u2014" not in (EXT / "background.js").read_text(encoding="utf-8")
    assert "\u2014" not in json.dumps(json.loads((EXT / "manifest.json").read_text(encoding="utf-8")))


def test_overlay_does_not_end_live_espn_draft_after_each_round():
    overlay = (EXT / "overlay.js").read_text(encoding="utf-8")
    espn_iso = (EXT / "espn_draft.js").read_text(encoding="utf-8")
    yahoo_iso = (EXT / "yahoo_draft.js").read_text(encoding="utf-8")
    espn_main = (EXT / "espn_draft_main.js").read_text(encoding="utf-8")
    yahoo_main = (EXT / "yahoo_draft_main.js").read_text(encoding="utf-8")
    assert "Math.ceil(max / teams)" not in espn_iso
    assert "Math.ceil(max / teams)" not in yahoo_iso
    assert "hostInProgress" in overlay
    assert "hostDrafted" in overlay
    assert "hostInProgress === true" in overlay
    assert "Math.min(SPOTS, season.length)" in overlay
    assert "isHostDraftRoom" in (EXT / "assistant_inject.js").read_text(encoding="utf-8")
    assert "mockdraftlobby" in (EXT / "draft_slot.js").read_text(encoding="utf-8")
    assert "r === inferred && r < 10" in overlay
    assert "lineupSlotCounts" in espn_main
    assert "rosterRoundsFromLineupSlots" in espn_main
    assert "detectedRounds" in espn_main
    assert "detectedRounds" in yahoo_main
    assert "inProgress: detail.inProgress" in espn_iso
    assert "inProgress: detail.inProgress" in yahoo_iso


def test_overlay_reads_league_settings_and_compares_players():
    helper = (EXT / "draft_slot.js").read_text(encoding="utf-8")
    overlay = (EXT / "overlay.js").read_text(encoding="utf-8")
    score = (EXT / "overlay_score.js").read_text(encoding="utf-8")
    html = (EXT / "overlay.html").read_text(encoding="utf-8")
    css = (EXT / "overlay.css").read_text(encoding="utf-8")
    espn_main = (EXT / "espn_draft_main.js").read_text(encoding="utf-8")
    yahoo_main = (EXT / "yahoo_draft_main.js").read_text(encoding="utf-8")
    espn_iso = (EXT / "espn_draft.js").read_text(encoding="utf-8")
    yahoo_iso = (EXT / "yahoo_draft.js").read_text(encoding="utf-8")
    sleeper = (EXT / "sleeper_draft.js").read_text(encoding="utf-8")
    inject = (EXT / "assistant_inject.js").read_text(encoding="utf-8")
    assert "function rosterFromEspnSlots" in helper
    assert "function rosterFromSleeperSettings" in helper
    assert "function rosterFromYahooPositions" in helper
    assert "function settingsLabel" in helper
    assert "function scoringFromSleeperSettings" in helper
    assert "rosterFromEspnSlots" in espn_main
    assert "scoringFromEspnSettings" in espn_main
    assert "detectedRoster" in espn_main
    assert "roster: detectedRoster" in espn_main
    assert "ppr: detectedPpr" in espn_main
    assert "rosterFromYahooPositions" in yahoo_main
    assert "roster: detectedRoster" in yahoo_main
    assert "ppr: detectedPpr" in yahoo_main
    assert "roster: detail.roster" in espn_iso
    assert "passTd: detail.passTd" in espn_iso
    assert "roster: detail.roster" in yahoo_iso
    assert "passTd: detail.passTd" in yahoo_iso
    assert "rosterFromSleeperSettings" in sleeper
    assert "api.sleeper.app/v1/league/" in sleeper
    assert "applyLeagueSettings" in overlay
    assert "leagueSettingsLabel" in overlay
    assert "scoreCtx" in overlay
    assert "pickOwners: state.pickOwners" in overlay
    assert "ctx.pickOwners" in score
    assert "roster: state.roster" in overlay
    assert "function rosterOf" in score
    assert "function slotList" in overlay
    assert "function slotEligible" in overlay
    assert "toggleCompare" in overlay
    assert "openCompare" in overlay
    assert "function draftPlayerFacts" in overlay
    assert "dr-cmp-player" in overlay
    assert "Pick Score" in overlay
    assert "Survive" in overlay
    assert "Proj Pts" in overlay
    assert "Mkt vs ADP" in overlay
    assert "Pos Rank" in overlay
    assert "infoIcon" in overlay
    assert 'data-cmp' in overlay
    assert "Compare Players" in overlay
    assert 'id="cmpModal"' in html
    assert "dr-cmp-overlay" in html
    assert 'src="draft_slot.js"' in html
    assert "dr-cmp-btn" in css
    assert "dr-cmp-stat.win" in css
    assert "dr-info" in css
    assert "settings-line" in css
    assert "BRDraftSlot.rosterKey" in inject
    assert "last_ppg" in (EXT / "background.js").read_text(encoding="utf-8")
    assert "vorp" in (EXT / "background.js").read_text(encoding="utf-8")
    assert "data-cmp-draft" not in overlay
    assert "[data-draft]" not in overlay
    assert "Open Draft Room" not in html
    assert "\u2014" not in overlay
    assert "\u2014" not in helper
    assert "\u2014" not in html


def test_sleeper_detects_live_pick_slot_from_several_signals():
    sleeper = (EXT / "sleeper_draft.js").read_text(encoding="utf-8")
    helper = (EXT / "draft_slot.js").read_text(encoding="utf-8")
    assert "function detectSleeperSlot" in helper
    assert "function collectSleeperIdentity" in helper
    assert "function slotFromSleeperDraftOrder" in helper
    assert "function slotFromSleeperPickedBy" in helper
    assert "function slotFromSleeperRosterMap" in helper
    assert "function slotFromSleeperClock" in helper
    assert "function detectSleeperDomSlot" in helper
    assert "is-me" in helper
    assert "detectSleeperSlot" in sleeper
    assert "resolveMySlot" in sleeper
    assert "resolveSleeperUserId" in sleeper
    assert "api.sleeper.app/v1/user/" in sleeper
    assert "api.sleeper.app/v1/league/" in sleeper
    assert "/rosters" in sleeper
    assert "/users" in sleeper
    assert "pickedBy" in sleeper
    assert "teamNamesFromSleeperDraft" in sleeper
    assert "visibilitychange" in sleeper
    assert "POLL_DRAFTING_MS" in sleeper
    assert "function teamNamesFromSleeperDraft" in helper
    assert "function sleeperPickOwners" in helper
    assert "function scrapeHostClockSeconds" in helper
    assert "traded_picks" in sleeper
    assert "__brDaPushClock" in sleeper
    assert "state.pickOwners" in (EXT / "overlay.js").read_text(encoding="utf-8")
    assert "function paintLiveClock" in (EXT / "overlay.js").read_text(encoding="utf-8")
    assert "__brDaPushClock" in (EXT / "assistant_inject.js").read_text(encoding="utf-8")
    assert "teamNames: detail.teamNames" in (EXT / "espn_draft.js").read_text(encoding="utf-8")
    assert "teamNames: detail.teamNames" in (EXT / "yahoo_draft.js").read_text(encoding="utf-8")
    assert 'mySlot: EMBEDDED ? 1 : 7' in (EXT / "overlay.js").read_text(encoding="utf-8")
    assert "12-team PPR · snake · round " not in (EXT / "overlay.js").read_text(encoding="utf-8")
    assert "if (!EMBEDDED)" in (EXT / "overlay.js").read_text(encoding="utf-8")
    assert "\u2014" not in sleeper


def test_sleeper_slot_helpers_unit():
    import subprocess

    script = r"""
const fs = require("fs");
const vm = require("vm");
const store = {
  _d: {},
  getItem(k) { return Object.prototype.hasOwnProperty.call(this._d, k) ? this._d[k] : null; },
  setItem(k, v) { this._d[k] = String(v); },
  key(i) { return Object.keys(this._d)[i]; },
  get length() { return Object.keys(this._d).length; },
};
const ctx = {
  window: {},
  localStorage: store,
  sessionStorage: store,
  document: {
    cookie: "",
    querySelectorAll() { return []; },
    querySelector() { return null; },
    createTreeWalker: null,
    body: null,
  },
  location: { pathname: "/draft/nfl/abc", href: "" },
};
ctx.window = ctx;
vm.runInNewContext(fs.readFileSync("extension/draft_slot.js", "utf8"), ctx);
const B = ctx.BRDraftSlot;
if (B.slotFromSleeperDraftOrder({ "111111": 3 }, ["111111"]) !== 3) process.exit(1);
if (B.slotFromSleeperPickedBy(
  [{ pickedBy: "111111", slot: 5, overallPickNumber: 5 }],
  ["111111"]
) !== 5) process.exit(1);
if (B.slotFromSleeperRosterMap({ "4": 9 }, { "111111": 9 }, ["111111"]) !== 4) process.exit(1);
if (B.slotFromSleeperClock("You're on the clock", 8, 12) !== 8) process.exit(1);
if (B.slotFromSleeperClock("You're on the clock", 13, 12) !== 12) process.exit(1);
const slot = B.detectSleeperSlot({
  draft: { draft_order: { "999999": 2 } },
  identity: { userIds: ["999999"] },
  teams: 12,
  skipDom: true,
});
if (slot !== 2) process.exit(1);
const clockWins = B.detectSleeperSlot({
  draft: { draft_order: { "999999": 5 } },
  identity: { userIds: ["999999"] },
  teams: 12,
  currentPick: 7,
  clockText: "You're on the clock",
  skipDom: true,
});
if (clockWins !== 7) process.exit(1);
const viaPicks = B.detectSleeperSlot({
  draft: {},
  picks: [{ picked_by: "999999", draft_slot: 7, overallPickNumber: 7 }],
  identity: { userIds: ["999999"] },
  teams: 12,
  skipDom: true,
});
if (viaPicks !== 7) process.exit(1);
const auction = B.detectSleeperSlot({
  draft: { type: "auction" },
  identity: { userIds: [] },
  teams: 12,
  currentPick: 8,
  clockText: "You're on the clock",
  skipDom: true,
});
if (auction !== 0) process.exit(1);
const names = B.teamNamesFromSleeperDraft(
  { draft_order: { "111111": 3, "222222": 1 } },
  [
    { user_id: "111111", display_name: "Night Owls" },
    { user_id: "222222", metadata: { team_name: "Gridiron" } },
  ]
);
if (names[3] !== "Night Owls" || names[1] !== "Gridiron") process.exit(1);
const fromIds = B.teamNamesFromTeamIds(
  { "10": "East", "11": "West" },
  { "10": 1 },
  [{ overallPickNumber: 2, teamId: "11" }],
  12
);
if (fromIds[1] !== "East" || fromIds[2] !== "West") process.exit(1);
if (B.parseClockSeconds("1:15 left") !== 75) process.exit(1);
const owners = B.sleeperPickOwners({
  teams: 4,
  rounds: 2,
  draft: { slot_to_roster_id: { "1": 1, "2": 2, "3": 3, "4": 4 } },
  ownerToRoster: {},
  tradedPicks: [{ roster_id: 1, round: 2, owner_id: 3 }],
  picks: [],
});
if (owners[1] !== 1) process.exit(1);
if (owners[8] !== 3) process.exit(1);
if (B.sleeperClockRemaining({ settings: { pick_timer: 90 }, last_picked: Date.now() - 10000 }, Date.now()) !== 80) process.exit(1);
console.log("ok");
"""
    out = subprocess.run(
        ["node", "-e", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr + out.stdout


def test_overlay_ranks_the_on_the_clock_pick_when_owners_are_traded():
    import subprocess

    script = r"""
const fs = require("fs");
const vm = require("vm");
const ctx = { window: {}, self: null };
ctx.window = ctx;
ctx.self = ctx;
vm.runInNewContext(fs.readFileSync("extension/overlay_score.js", "utf8"), ctx);
const S = ctx.BROverlayScore;
const traded = { current: 7, teams: 12, rounds: 15, mySlot: 5, pickOwners: { 7: 5, 20: 5 } };
if (S.recommendationPickNo(traded) !== 7) process.exit(1);
const snakeWait = { current: 7, teams: 12, rounds: 15, mySlot: 5 };
if (S.recommendationPickNo(snakeWait) !== 7) process.exit(1);
const onClock = { current: 7, teams: 12, rounds: 15, mySlot: 7 };
if (S.recommendationPickNo(onClock) !== 7) process.exit(1);
const stringKeys = { current: 7, teams: 12, rounds: 15, mySlot: 5, pickOwners: { "7": 5 } };
if (S.recommendationPickNo(stringKeys) !== 7) process.exit(1);
console.log("ok");
"""
    out = subprocess.run(
        ["node", "-e", script],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr + out.stdout
