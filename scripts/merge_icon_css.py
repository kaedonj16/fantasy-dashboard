#!/usr/bin/env python3
"""Re-merge icons.css + font-awesome.css into static/dashboard.css.

dashboard.css ends with two MERGED sections (icons.css, then font-awesome.css,
in the same order the <link> tags used to load). This script rebuilds those
sections from the source files. Run after editing static/icons.css or
static/font-awesome.css.

Usage: python3 scripts/merge_icon_css.py
"""
from pathlib import Path

STATIC = Path(__file__).parent.parent / "static"

ICONS_MARKER = "MERGED: icons.css"
FA_MARKER = "MERGED: font-awesome.css"


def main() -> None:
    dash_path = STATIC / "dashboard.css"
    text = dash_path.read_text(encoding="utf-8")

    # Cut everything from the first merge marker onward
    idx = text.find(ICONS_MARKER)
    if idx == -1:
        raise SystemExit("icons.css merge marker not found in dashboard.css")
    # Back up to the start of the comment block containing the marker
    comment_start = text.rfind("/*", 0, idx)
    base = text[:comment_start].rstrip() + "\n"

    icons_text = (STATIC / "icons.css").read_text(encoding="utf-8").strip()
    fa_text = (STATIC / "font-awesome.css").read_text(encoding="utf-8").strip()
    # Strip the DEPRECATED header (it describes the source file, not the merge)
    for _hdr in ("/* DEPRECATED",):
        if icons_text.startswith(_hdr):
            _end = icons_text.find("*/") + 2
            icons_text = icons_text[_end:].strip()
        if fa_text.startswith(_hdr):
            _end = fa_text.find("*/") + 2
            fa_text = fa_text[_end:].strip()

    merged = (
        base
        + "\n/* ============================================================\n"
        + "   MERGED: icons.css (PNG-based icon system)\n"
        + "   Do not edit here -- edit static/icons.css and re-run the merge.\n"
        + "   See scripts/merge_icon_css.py\n"
        + "   ============================================================ */\n"
        + icons_text
        + "\n/* ============================================================\n"
        + "   MERGED: font-awesome.css (Font Awesome 6.5.1 base)\n"
        + "   Do not edit here -- edit static/font-awesome.css and re-run the merge.\n"
        + "   See scripts/merge_icon_css.py\n"
        + "   ============================================================ */\n"
        + fa_text
        + "\n"
    )
    dash_path.write_text(merged, encoding="utf-8")
    print(f"merged: {len(merged)} bytes")


if __name__ == "__main__":
    main()
