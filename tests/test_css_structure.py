"""Stylesheet structural sanity: balanced braces, nothing swallowed.

A truncated rule in landing_lite.css (`.pos-bar-inner {` with its body
dropped during a hand extraction) left one unclosed block that swallowed
every rule after it, including the whole guest-landing block, in the sheet
guests actually load. These tests pin the structure so a bad extraction
fails CI instead of shipping.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LITE = (ROOT / "static" / "landing_lite.css").read_text(encoding="utf-8")
DASH = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


def _brace_depth(text: str) -> int:
    """Net { } depth skipping comments and quoted strings."""
    depth = 0
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "/" and i + 1 < n and text[i + 1] == "*":
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if ch in "\"'":
            j = i + 1
            while j < n and text[j] != ch:
                j += 2 if text[j] == "\\" else 1
            i = j + 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            assert depth >= 0, "closing brace without an open block"
        i += 1
    return depth


def test_stylesheets_have_balanced_braces():
    assert _brace_depth(LITE) == 0
    assert _brace_depth(DASH) == 0


def test_lite_board_mirror_rules_carry_their_bodies():
    # The mirrored board block must match dashboard.css, bodies included.
    assert ".pos-bar-inner {\n    height: 100%;" in LITE
    assert ".pos-bar-inner {\n    height: 100%;" in DASH


def test_landing_block_follows_the_mirror_in_lite():
    # If anything above the landing block is left unclosed, browsers drop
    # the whole block; pin the ordering the parser depends on.
    assert LITE.index(".pos-bar-inner") < LITE.index(".ld-hero-card")
    assert ".ld-hero-card {" in LITE
