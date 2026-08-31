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
    assert manifest["version"] == "1.5.0"
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
