"""Guard: ADP board on mobile pins # / Player and scrolls source columns.

The schedule-rankings pattern (sticky left + overflow-x on the right) must stay
wired for Sort-by-ADP on phones. Without the pin classes + CSS left offsets,
wide ADP source columns clip player names or push them off-screen.
"""
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_JS = os.path.join(_ROOT, "static", "rankings.js")
_SHELL = os.path.join(_ROOT, "dashboard_services", "pages", "players_page.py")


def test_adp_mobile_pins_rank_and_player():
    js = open(_JS, encoding="utf-8").read()
    shell = open(_SHELL, encoding="utf-8").read()

    assert "PR_ADP_RANK_W_MOBILE" in js
    assert "PR_ADP_PLAYER_W_MOBILE" in js
    assert "pr-adp-pin-rank" in js
    assert "pr-adp-pin-player" in js

    # Mobile sticky CSS must pin both columns and scroll inside #prTableScroll.
    assert "pr-adp-pin-rank" in shell
    assert "pr-adp-pin-player" in shell
    assert "left: 40px" in shell  # matches PR_ADP_RANK_W_MOBILE
    assert re.search(
        r"#prTableScroll\.pr-adp-scroll\s*\{[^}]*-webkit-overflow-scrolling:\s*touch",
        shell,
        re.S,
    ), "mobile ADP scroller missing touch scrolling"
    assert "min-width: max-content" in shell
