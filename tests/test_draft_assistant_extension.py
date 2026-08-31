"""Guards for the docked Draft Assistant overlay in the League Connector."""
from pathlib import Path
import json
import re

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "extension"


def test_overlay_is_mv3_safe_extension_page():
    html = (EXT / "overlay.html").read_text(encoding="utf-8")
    assert 'src="overlay.js"' in html
    assert 'href="overlay.css"' in html
    assert 'class="br-da-embed"' in html
    assert not re.search(r"<script>(?!\s*</script>)", html)
    assert (EXT / "overlay.js").is_file()
    assert (EXT / "overlay.css").is_file()
    js = (EXT / "overlay.js").read_text(encoding="utf-8")
    assert "ingestLive" in js
    assert '__br: "br-da"' in js
    assert "never submits" in js.lower() or "never submit" in js.lower()


def test_manifest_docks_overlay_on_host_drafts():
    manifest = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "1.5.4"
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
    assert sleeper_js == ["assistant_inject.js", "sleeper_draft.js"]
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
    assert "sleepercdn.com/content/nfl/players/" in background
    assert "slots_super_flex" in sleeper
    assert "adpSel" in html
    assert "ADP source" in html
    assert 'data-link="room"' in html
    assert 'data-link="sheet"' in html
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
    assert "transition:transform" in inject.replace(" ", "")
    assert "br-da-ready" in inject
    assert "prefers-reduced-motion" in inject
    assert "collapseBtn" in html
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
