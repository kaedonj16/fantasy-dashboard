"""Guards for Yahoo live-draft extension + Draft Room wiring."""
from pathlib import Path
import json

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "extension"
ROOM_JS = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")


def test_extension_manifest_includes_yahoo_draft_scripts():
    manifest = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "1.5.28"
    assert "cookies" in manifest["permissions"]
    assert "tabs" in manifest["permissions"]
    hosts = " ".join(manifest.get("host_permissions") or [])
    assert "fantasysports.yahoo.com" in hosts
    scripts = manifest["content_scripts"]
    main_js = None
    iso_js = None
    for block in scripts:
        joined = " ".join(block.get("matches") or [])
        if "fantasysports.yahoo.com" in joined and block.get("world") == "MAIN":
            main_js = block["js"]
        if "fantasysports.yahoo.com" in joined and block.get("world") != "MAIN":
            iso_js = block["js"]
    assert main_js == ["draft_slot.js", "yahoo_draft_main.js"]
    assert iso_js == ["draft_slot.js", "assistant_inject.js", "yahoo_draft.js"]
    main_block = next(
        s for s in scripts
        if s.get("world") == "MAIN" and "yahoo_draft_main.js" in s.get("js", [])
    )
    iso_block = next(
        s for s in scripts
        if s.get("world", "ISOLATED") != "MAIN" and "yahoo_draft.js" in s.get("js", [])
    )
    assert main_block.get("all_frames") is True
    assert not iso_block.get("all_frames")
    assert (EXT / "yahoo_draft_main.js").is_file()
    assert (EXT / "yahoo_draft.js").is_file()


def test_yahoo_extension_relay_message_contract():
    bg = (EXT / "background.js").read_text(encoding="utf-8")
    main = (EXT / "yahoo_draft_main.js").read_text(encoding="utf-8")
    iso = (EXT / "yahoo_draft.js").read_text(encoding="utf-8")
    content = (EXT / "content.js").read_text(encoding="utf-8")
    assert "yahooDraftRelay" in bg
    assert "brfantasy:yahoo-draft-raw" in main
    assert "brfantasy:yahoo-draft-raw" in iso
    assert "relayToBackground" in main
    assert "chrome.runtime.sendMessage" in main
    assert "finishReconnect" in iso
    assert "draftRelayResult" in iso
    assert "reconnect sent" not in iso
    assert "brfantasy:yahoo-draft-relay" in content
    assert "overallPickNumber" in main
    assert "br-fantasy-yahoo-sync-chip" in iso
    assert "player_key" in main
    assert "playerName" in main
    assert "yahooPlayerName" in main
    assert "rememberYahooUser" in main
    assert "computeMySlot" in main
    assert "userTeamId" in main
    assert "resolveMySlot" in iso
    assert "BRDraftSlot" in iso
    assert "detectYahooSlot" in iso
    assert "pollYahooLive" in iso
    assert "draftclient" in main
    helper = (EXT / "draft_slot.js").read_text(encoding="utf-8")
    assert "function detectYahooSlot" in helper
    assert "function slotFromYahooClock" in helper
    assert "you(?:'re| are) up in" in helper
    assert "yahooClientTeamId" in helper
    inject = (EXT / "assistant_inject.js").read_text(encoding="utf-8")
    assert "function applyDockShift" in inject
    assert "data-br-da-shifted" in inject
    assert "br-da-" in inject
    assert "--br-da-shift" in inject
    assert "display:flex" in inject
    assert "looksFullBleed" in inject
    assert "100vw" in inject
    assert "br-da-yahoo #root" in inject
    assert "br-da-sleeper #root" in inject


def test_draft_room_yahoo_live_wiring():
    assert "function applyYahooExtensionRelay(detail)" in ROOM_JS
    assert "/api/draft/yahoo-relay" in ROOM_JS
    assert "brfantasy:yahoo-draft-relay" in ROOM_JS
    assert "function yahooDraftUrl()" in ROOM_JS
    assert "Live sync currently supports Sleeper, ESPN, and Yahoo leagues." in ROOM_JS
    assert "function isExtLiveSource()" in ROOM_JS
    assert "plat === 'espn' || plat === 'yahoo'" in ROOM_JS
    assert "Sync Yahoo picks automatically" in ROOM_JS or "Sync ' + who + ' picks automatically" in ROOM_JS


def test_yahoo_live_picks_accumulate_and_scrape_board():
    main = (EXT / "yahoo_draft_main.js").read_text(encoding="utf-8")
    iso = (EXT / "yahoo_draft.js").read_text(encoding="utf-8")
    helper = (EXT / "draft_slot.js").read_text(encoding="utf-8")
    overlay = (EXT / "overlay.js").read_text(encoding="utf-8")
    inject = (EXT / "assistant_inject.js").read_text(encoding="utf-8")
    assert "pickAccumulator" in main
    assert "function mergeIntoAccumulator" in main
    assert "function collectPickArrays" in main
    assert "function inspectText" in main
    assert "window.WebSocket" in main
    assert "function scrapeYahooDomPicks" in main
    assert "function scanAll" in main
    assert "return true;" not in main.split("function walkForPicks")[1].split("function scanReact")[0]
    assert "relayPending();" in iso
    assert "scrapeYahooBoard" in iso
    assert "pushMergedPicks" in iso
    assert "completedFromYahooClock" in iso
    assert "function scrapeYahooBoard" in helper
    assert "function parseYahooDraftResultsHtml" in helper
    assert "function sameOriginDocuments" in helper
    assert "function parseYahooLooseName" in helper
    assert "function parseYahooCompactPick" in helper
    assert "function filterYahooPicksToClock" in helper
    assert "matchAbbrevName" in overlay
    assert "harvestPageJson" in main
    assert "pollDraftResultPages" in main
    assert "window.top.postMessage" in main
    assert "sameOriginDocuments" in main
    assert "looksDraftedYahoo" in main
    assert "filterYahooPicksToClock" in main
    assert "filterYahooPicksToClock" in iso
    assert "parseYahooCompactPick" in helper
    assert "detail.current" in overlay or "clockPn" in overlay
    assert "draftedPlayers" in main
    assert "lastPicks && lastPicks.length" in iso
    assert "function mergeYahooPicks" in helper
    assert "function parseYahooNamePos" in helper
    assert "function completedFromYahooClock" in helper
    assert "last.playerName || last.name" in overlay
    assert "last.playerName || last.name" in inject
    assert "last.playerName || last.name" in iso
    assert "\u2014" not in main
    assert "\u2014" not in iso
    assert "\u2014" not in helper


def test_yahoo_pick_helpers_merge_and_parse():
    import subprocess
    script = r"""
const fs = require("fs");
const vm = require("vm");
const ctx = {
  window: {},
  document: {
    cookie: "",
    querySelectorAll() { return []; },
    querySelector() { return null; },
    createTreeWalker: null,
    body: null,
  },
  location: { pathname: "/draftclient/f1/10288933/8", href: "" },
};
ctx.window = ctx;
vm.runInNewContext(fs.readFileSync("extension/draft_slot.js", "utf8"), ctx);
const B = ctx.BRDraftSlot;
const merged = B.mergeYahooPicks(
  [{ overallPickNumber: 1, playerName: "Ja'Marr Chase", playerId: "31002", pos: "WR" }],
  [{ overallPickNumber: 2, playerName: "Bijan Robinson", pos: "RB", nflTeam: "ATL" }]
);
if (merged.length !== 2 || merged[1].playerName.indexOf("Bijan") < 0) {
  console.error("merge", merged);
  process.exit(1);
}
const np = B.parseYahooNamePos("Ja'Marr Chase WR CIN");
if (!np || np.name.indexOf("Chase") < 0 || np.pos !== "WR") {
  console.error(np);
  process.exit(1);
}
const slot = B.slotFromYahooClock("You're up in 7 Picks Round 1, Pick 1", 12);
if (slot !== 8) {
  console.error("slot", slot);
  process.exit(1);
}
const late = B.parseYahooClock("Jane's Pick • You're up in 4 Picks Round 14, Pick 185");
if (late.overall !== 185) {
  console.error("overall", late);
  process.exit(1);
}
const htmlRows = B.parseYahooDraftResultsHtml("1 Ja'Marr Chase WR CIN 2 Bijan Robinson RB ATL");
if (htmlRows.length < 2 || htmlRows[0].playerName.indexOf("Chase") < 0) {
  console.error("html", htmlRows);
  process.exit(1);
}
const kf = B.parseYahooNamePos("K. Fairbairn K HOU");
if (!kf || kf.pos !== "K") {
  console.error("kf", kf);
  process.exit(1);
}
const loose = B.parseYahooLooseName("Ja'Marr Chase");
if (!loose || loose.name.indexOf("Chase") < 0) {
  console.error("loose", loose);
  process.exit(1);
}
const docs = B.sameOriginDocuments(ctx.document);
if (!docs || docs.length < 1) {
  console.error("docs", docs);
  process.exit(1);
}
const compact = B.parseYahooCompactPick("1.1 J. GIBBS RB DET", 12);
if (!compact || compact.overallPickNumber !== 1 || compact.playerName.indexOf("GIBBS") < 0) {
  console.error("compact", compact);
  process.exit(1);
}
const compact2 = B.parseYahooCompactPick("1.2 B. ROBINSON RB ATL", 12);
if (!compact2 || compact2.overallPickNumber !== 2) {
  console.error("compact2", compact2);
  process.exit(1);
}
const kept = B.mergeYahooPicks(merged, [{ overallPickNumber: 1, playerName: "Ja" }]);
if (kept[0].playerName !== "Ja'Marr Chase" || kept.length !== 2) {
  console.error("shrink", kept);
  process.exit(1);
}
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
