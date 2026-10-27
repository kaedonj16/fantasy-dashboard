"""Regression pin: Weekly Recap "Top scorers by position" cards keep their
white card background.

The .rc-pos-col cards never carried their own background; they read as
white cards only because the guest-home section's late `body` rule
(background: var(--bg), white) sat after the main body rule
(background: var(--bg-alt)) and won the cascade on every page. PR #2831's
font fix (commit 1961f46b) scoped that late rule to body:has(.home-hero),
so app pages fell back to the main rule's gray-blue --bg-alt and the
transparent pos cards went flat while real .card cards (Injury Report)
stayed white.

The cards now carry `background: var(--card)` themselves, like every
other card on the page, so a future CSS sweep or cascade change cannot
silently flatten them again. This test fails on the tree without the
declaration and passes with it.
"""
import os
import re

_ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(*parts):
    with open(os.path.join(_ROOT, *parts), encoding="utf-8") as fh:
        return fh.read()


def _pos_col_rule():
    src = _read("dashboard_services", "pages", "recap_page.py")
    match = re.search(r"\.rc-pos-col \{\{([^}]+)\}\}", src)
    assert match, "could not find the .rc-pos-col rule in recap_page.py"
    return match.group(1)


def test_pos_col_cards_carry_their_own_card_background():
    rule = _pos_col_rule()
    assert re.search(r"background:\s*var\(--card\)", rule), (
        ".rc-pos-col must paint its own card background; it used to borrow "
        "the white body background that PR #2831's body:has(.home-hero) "
        "scoping withdrew"
    )


def test_pos_col_rule_keeps_its_card_frame():
    rule = _pos_col_rule()
    assert "border:1px solid var(--border)" in rule
    assert "border-radius:10px" in rule
