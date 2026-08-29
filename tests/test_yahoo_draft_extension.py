"""Guards for Yahoo live-draft extension + Draft Room wiring."""
from pathlib import Path
import json

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "extension"
ROOM_JS = (REPO / "static" / "draft_room.js").read_text(encoding="utf-8")


def test_extension_manifest_includes_yahoo_draft_scripts():
    manifest = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "1.3.0"
    assert "cookies" in manifest["permissions"]
    assert "tabs" not in manifest.get("permissions", [])
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
    assert main_js == ["yahoo_draft_main.js"]
    assert iso_js == ["yahoo_draft.js"]
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
    assert "brfantasy:yahoo-draft-relay" in content
    assert "overallPickNumber" in main
    assert "br-fantasy-yahoo-sync-chip" in iso
    assert "player_key" in main


def test_draft_room_yahoo_live_wiring():
    assert "function applyYahooExtensionRelay(detail)" in ROOM_JS
    assert "/api/draft/yahoo-relay" in ROOM_JS
    assert "brfantasy:yahoo-draft-relay" in ROOM_JS
    assert "function yahooDraftUrl()" in ROOM_JS
    assert "Live sync currently supports Sleeper, ESPN, and Yahoo leagues." in ROOM_JS
    assert "function isExtLiveSource()" in ROOM_JS
    assert "plat === 'espn' || plat === 'yahoo'" in ROOM_JS
    assert "Sync Yahoo picks automatically" in ROOM_JS or "Sync ' + who + ' picks automatically" in ROOM_JS
