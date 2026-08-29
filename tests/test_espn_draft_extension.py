"""Guards for the ESPN live-draft extension relay files."""
from pathlib import Path
import json

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "extension"


def test_extension_manifest_includes_draft_scripts():
    manifest = json.loads((EXT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["version"] == "1.2.0"
    assert "cookies" in manifest["permissions"]
    assert "tabs" not in manifest.get("permissions", [])
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
    assert main_js == ["espn_draft_main.js"]
    assert iso_js == ["espn_draft.js"]
    assert (EXT / "espn_draft_main.js").is_file()
    assert (EXT / "espn_draft.js").is_file()
    assert (EXT / "pack_extension.py").is_file()


def test_extension_relay_message_contract():
    bg = (EXT / "background.js").read_text(encoding="utf-8")
    main = (EXT / "espn_draft_main.js").read_text(encoding="utf-8")
    iso = (EXT / "espn_draft.js").read_text(encoding="utf-8")
    content = (EXT / "content.js").read_text(encoding="utf-8")
    assert 'type: "espnDraftRelay"' in bg or "type: \"espnDraftRelay\"" in bg
    assert "brfantasy:espn-draft-raw" in main
    assert "brfantasy:espn-draft-raw" in iso
    assert "brfantasy:espn-draft-relay" in content
    assert "overallPickNumber" in main
    assert "br-fantasy-espn-sync-chip" in iso


def test_pack_extension_strips_localhost():
    import subprocess
    import sys
    import zipfile

    subprocess.check_call([sys.executable, str(EXT / "pack_extension.py")], cwd=str(EXT.parent))
    zips = list((EXT.parent / "artifacts").glob("br-fantasy-espn-connector-v*.zip"))
    assert zips
    with zipfile.ZipFile(sorted(zips)[-1]) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    blob = json.dumps(manifest)
    assert "localhost" not in blob
    assert "127.0.0.1" not in blob
