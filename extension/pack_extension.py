#!/usr/bin/env python3
"""Build a Chrome Web Store / AMO zip of the BR Fantasy ESPN extension.

Strips localhost host permissions for the production package. Usage:

    python3 extension/pack_extension.py
    # writes artifacts/br-fantasy-espn-connector-vX.Y.Z.zip
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT_DIR = REPO / "artifacts"

INCLUDE = [
    "manifest.json",
    "background.js",
    "content.js",
    "content.css",
    "espn_draft.js",
    "espn_draft_main.js",
    "popup.html",
    "popup.js",
    "popup.css",
    "icons/icon16.png",
    "icons/icon32.png",
    "icons/icon48.png",
    "icons/icon128.png",
]


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    hosts = [
        h
        for h in manifest.get("host_permissions", [])
        if "localhost" not in h and "127.0.0.1" not in h
    ]
    manifest["host_permissions"] = hosts
    # Drop localhost from content_scripts matches too.
    for block in manifest.get("content_scripts", []):
        block["matches"] = [
            m
            for m in block.get("matches", [])
            if "localhost" not in m and "127.0.0.1" not in m
        ]
    version = str(manifest.get("version") or "0.0.0")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"br-fantasy-espn-connector-v{version}.zip"
    # Some cloud/VM checkouts stamp files at unix epoch; ZIP requires >= 1980.
    safe_date = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            zipfile.ZipInfo("manifest.json", date_time=safe_date),
            json.dumps(manifest, indent=2) + "\n",
        )
        for rel in INCLUDE:
            if rel == "manifest.json":
                continue
            path = ROOT / rel
            if not path.is_file():
                raise SystemExit(f"missing {rel}")
            info = zipfile.ZipInfo(rel, date_time=safe_date)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
    print(out)


if __name__ == "__main__":
    main()
