#!/usr/bin/env python3
"""Build a Chrome Web Store / AMO zip of the BR Fantasy league connector.

Strips localhost host permissions for the production package. Usage:

    python3 extension/pack_extension.py
    # writes:
    #   artifacts/br-fantasy-espn-connector-vX.Y.Z.zip
    #   static/extension/br-fantasy-espn-connector.zip  (stable download URL)
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT_DIR = REPO / "artifacts"
STATIC_DIR = REPO / "static" / "extension"

INCLUDE = [
    "manifest.json",
    "background.js",
    "content.js",
    "content.css",
    "espn_draft.js",
    "espn_draft_main.js",
    "yahoo_draft.js",
    "yahoo_draft_main.js",
    "popup.html",
    "popup.js",
    "popup.css",
    "icons/icon16.png",
    "icons/icon32.png",
    "icons/icon48.png",
    "icons/icon128.png",
]


def build_manifest() -> dict:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    hosts = [
        h
        for h in manifest.get("host_permissions", [])
        if "localhost" not in h and "127.0.0.1" not in h
    ]
    manifest["host_permissions"] = hosts
    for block in manifest.get("content_scripts", []):
        block["matches"] = [
            m
            for m in block.get("matches", [])
            if "localhost" not in m and "127.0.0.1" not in m
        ]
    return manifest


def write_zip(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Some cloud/VM checkouts stamp files at unix epoch; ZIP requires >= 1980.
    safe_date = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            zipfile.ZipInfo("manifest.json", date_time=safe_date),
            json.dumps(manifest, indent=2) + "\n",
        )
        for rel in INCLUDE:
            if rel == "manifest.json":
                continue
            src = ROOT / rel
            if not src.is_file():
                raise SystemExit(f"missing {rel}")
            info = zipfile.ZipInfo(rel, date_time=safe_date)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, src.read_bytes())


def main() -> None:
    manifest = build_manifest()
    version = str(manifest.get("version") or "0.0.0")
    versioned = OUT_DIR / f"br-fantasy-espn-connector-v{version}.zip"
    stable = STATIC_DIR / "br-fantasy-espn-connector.zip"
    write_zip(versioned, manifest)
    write_zip(stable, manifest)
    # Tiny readme so the download folder is self-explanatory in the repo.
    readme = STATIC_DIR / "README.md"
    readme.write_text(
        "# BR Fantasy league connector (download)\n\n"
        "`br-fantasy-espn-connector.zip` is the production package users download "
        "from Draft Room (ESPN + Yahoo live draft relay). Rebuild with:\n\n"
        "```bash\npython3 extension/pack_extension.py\n```\n",
        encoding="utf-8",
    )
    print(versioned)
    print(stable)


if __name__ == "__main__":
    main()
