"""Guards for the ESPN live-draft extension relay files."""
from pathlib import Path
import json

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "extension"


def test_extension_manifest_includes_draft_scripts():
    manifest = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "1.5.27"
    assert "cookies" in manifest["permissions"]
    assert "scripting" in manifest["permissions"]
    assert "tabs" in manifest["permissions"]
    scripts = manifest["content_scripts"]
    worlds = {(tuple(s["matches"]), s.get("world", "ISOLATED")): s["js"] for s in scripts}
    main_js = None
    iso_js = None
    for (matches, world), js in worlds.items():
        joined = " ".join(matches)
        if "fantasy.espn.com/football/draft" in joined and world == "MAIN":
            main_js = js
        if "fantasy.espn.com/football/draft" in joined and world != "MAIN":
            iso_js = js
    assert main_js == ["draft_slot.js", "espn_draft_main.js"]
    assert iso_js == ["draft_slot.js", "assistant_inject.js", "espn_draft.js"]
    main_block = next(
        s for s in scripts
        if s.get("world") == "MAIN" and "espn_draft_main.js" in s.get("js", [])
    )
    iso_block = next(
        s for s in scripts
        if s.get("world", "ISOLATED") != "MAIN" and "espn_draft.js" in s.get("js", [])
    )
    assert not main_block.get("all_frames")
    assert not iso_block.get("all_frames")
    assert (EXT / "espn_draft_main.js").is_file()
    assert (EXT / "espn_draft.js").is_file()
    assert (EXT / "pack_extension.py").is_file()


def test_extension_relay_message_contract():
    manifest = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    bg = (EXT / "background.js").read_text(encoding="utf-8")
    main = (EXT / "espn_draft_main.js").read_text(encoding="utf-8")
    iso = (EXT / "espn_draft.js").read_text(encoding="utf-8")
    yahoo_iso = (EXT / "yahoo_draft.js").read_text(encoding="utf-8")
    content = (EXT / "content.js").read_text(encoding="utf-8")
    assert 'type: "espnDraftRelay"' in bg or "type: \"espnDraftRelay\"" in bg or "espnDraftRelay" in bg
    assert "yahooDraftRelay" in bg
    assert "brfantasy:espn-draft-raw" in main
    assert "brfantasy:espn-draft-raw" in iso
    assert "relayToBackground" in main
    assert "chrome.runtime.sendMessage" in main
    assert "brfantasy:espn-relay-status" in main
    assert "brfantasy:espn-observer-ready" in main
    assert "pollEspnApi" in main
    assert "scrapeDomPicks" in main
    assert "scope.querySelectorAll" in main
    assert "typeof scope.querySelectorAll" in main
    assert "isEspnDraftRoom" in main
    assert "mockdraftlobby" in main
    assert "startEspnDraftObserver" in main
    assert "!document.body" in main
    assert "playerIdFromImg" in main
    assert "dom-scrape" in main
    assert "pickAccumulator" in main
    assert "mergeIntoAccumulator" in main
    assert "findBestDraftDetail" in main
    assert "pickSources" in main
    assert "isTraversableObject" in main
    assert "safeProp" in main
    assert "instanceof Window" in main
    assert "function isWindowLike" in main
    assert "function hostBrand" in main
    assert "[object Window]" in main
    assert "isWindowLike(obj)" in main
    assert "isWindowLike(node)" in main
    assert "emitAccumulated" in main
    assert "watchDom" in main
    assert "deepFindDraftDetail" in main
    assert "playerPoolEntry" in main
    assert "ensureEspnDraftObserver" in bg
    assert "ensureEspnDraftObserver" in iso or "requestObserverInject" in iso
    assert "mainObserverReady" in iso
    assert "observer not loaded" in iso
    assert "reconnect sent" not in iso
    assert "nudgeDraftTabScan" in bg
    assert "draftRelayResult" in bg
    assert "notifyDraftTabRelayResult" in bg
    assert "relayPending" in iso
    assert "brfantasy:espn-draft-relay" in content
    assert "brfantasy:yahoo-draft-relay" in content
    assert "playerName" in main
    assert "rememberPlayerMeta" in main
    assert "br-fantasy-espn-sync-chip" in iso
    assert "lastDelivered" in iso
    assert "scheduleRetry" in iso
    assert "lastDelivered" in yahoo_iso
    assert "playerIdSelected" in main
    assert "pickLooksMade" in main
    assert "isPlaceholderPlayerName" in main
    assert "Empty ESPN seats" in main
    assert "teamId != null" not in main.split("function isPickRow")[1].split("function normalizePick")[0]
    assert "chrome.scripting.executeScript" in bg
    assert "brDraftRoomTabs" in bg
    assert "registerBrDraftRoomTab" in bg
    assert "queryBrDraftRoomTabs" in bg
    assert "persistBrDraftRoomTabs" in bg
    assert "chrome.storage.session" in bg
    assert "forceReplay" in bg
    assert "lastRelaySuccessAt" in iso or "relaySuccessSticky" in iso
    assert "announceDraftRoom" in content
    assert "relayFailureText" in iso
    assert "forceResend" in iso
    assert "manualReconnect" in iso
    assert "injectPageEvent" in bg
    assert "composed: true" in bg
    assert "dispatchToPage" in content
    assert "RECONNECT_COOLDOWN_MS" in bg
    assert "reconnectDraftRelay" in bg
    assert "forceDraftRelay" in bg
    assert "brfantasy:request-extension-reconnect" in content
    assert "brfantasy:extension-reconnect" in content
    assert "brfantasy:draft-rescan" in main
    popup_html = (EXT / "popup.html").read_text(encoding="utf-8")
    assert "reconnectBtn" in popup_html
    assert "openAssistantBtn" in popup_html
    assert "rememberEspnUser" in main
    assert "computeMySlot" in main
    assert "userTeamId" in main
    assert "view=mTeam" in main
    assert "lineupSlotCounts" in main
    assert "rosterRoundsFromLineupSlots" in main
    assert "detectedRounds" in main
    assert "resolveMySlot" in iso
    assert "BRDraftSlot" in iso


def test_pack_extension_strips_localhost():
    import subprocess
    import sys
    import zipfile

    subprocess.check_call([sys.executable, str(EXT / "pack_extension.py")], cwd=str(EXT.parent))
    zips = list((EXT.parent / "artifacts").glob("br-fantasy-espn-connector-v*.zip"))
    assert zips
    with zipfile.ZipFile(sorted(zips)[-1]) as zf:
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    blob = json.dumps(manifest)
    assert "localhost" not in blob
    assert "127.0.0.1" not in blob
    assert "draft_slot.js" in names
    assert "overlay.html" in names
    assert "overlay.css" in names
    assert "overlay.js" in names
    assert "overlay_score.js" in names
    assert "pick_score.js" in names
    assert "draft_board_core.js" in names
    assert "assistant_inject.js" in names
    assert "sleeper_draft.js" in names
