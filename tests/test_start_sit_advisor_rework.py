"""Start/Sit Advisor rework (2026-11-05): search card, badge-only pills,
grouped evidence, and the slim compare panel with a shared-scale Range.

Kaedon's approved mock: the verdict badge is the only filled pill left on a
row, the expanded detail groups into Why this call / Range / Game context,
and compare Range labels live in reserved lines so they cannot overlap.
"""

import re
from pathlib import Path

import pytest

from dashboard_services.pages.waivers_page import build_waivers_body

ROOT = Path(__file__).resolve().parent.parent
CSS = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def body():
    return build_waivers_body("sleeper", 2026, "league", {})


@pytest.fixture(scope="module")
def script(body):
    scripts = re.findall(r"<script>(.*?)</script>", body, re.S)
    matches = [s for s in scripts if "function wvRenderStartSit(" in s]
    assert matches, "waivers inline script not found"
    return matches[0]


def test_search_is_a_compare_card(body):
    assert 'placeholder="Compare a player"' in body
    assert "Add any player (free agent or another team)" not in body
    assert "wv-guest-ic" in body
    assert "Free agents and other rosters join as Guests and get a verdict." in body
    # Results carry the honest /api/players signal: rank label + model value.
    assert "wv-guest-av" in body
    assert "pos_rank_label" in body
    # Guest chips gain the real verdict once start-sit rows load.
    assert "Would start" in body
    assert "wv-guest-verdict" in body


def test_badge_is_the_only_pill_on_start_sit_rows(script):
    # Matchup renders as dot + plain text in a Start/Sit-scoped class, and
    # the demotion renders the same way. Neither uses the filled chip.
    assert 'class="wv-ss-mu mt${tier}"' in script
    assert 'class="wv-ss-demote"' in script
    assert "wv-cx-chip bad" not in script
    assert "wvSsMatchupChip(p.def_rank, p.def_total)" in script
    # The filled chip class still exists for waiver rows that share it.
    assert ".wv-cx-chip" in CSS
    assert ".wv-ss-mu.mt3 .wv-ss-dot" in CSS


def test_evidence_groups_into_three_sections(script):
    assert "WHY THIS CALL" in script
    assert "GAME CONTEXT" in script
    assert "wvSsWhySentence" in script
    assert "wvSsFactors" in script
    # The flat 3-col evidence cells are gone from the Start/Sit detail.
    assert "FLOOR - CEIL" not in script
    # Game context keeps the pinned uppercase labels.
    assert ">VENUE<" in script
    assert ">TEAMMATES OUT<" in script
    assert ">OPP DEFENSE OUT<" in script
    # Actions: Open player is the single primary, the rest quiet.
    assert "wv-cx-act-primary" in script
    assert "wv-cx-act-quiet" in script
    # The evidence container is a stacked column now, never a 3-col grid.
    assert "grid-template-columns: repeat(3, 1fr)" not in CSS.split(".wv-cx-evidence {")[1].split("}")[0]


def test_range_labels_use_reserved_lines(script):
    # wvRangeBar reserves a fixed line above the bar for the Proj label and
    # one below for the L4 label; both labels clamp inside the bar width.
    assert "function wvRangeBar(o)" in script
    assert script.count("wv-rg-lbl-line") >= 2
    assert ".wv-rg-lbl-line { position: relative; height: 16px; }" in CSS
    # The compare Range draws both players through the same helper.
    assert "wvRangeBar({ floor: dp.floorNum, ceiling: dp.ceilNum" in script


def test_compare_header_chips_and_mobile_rows(body, script):
    # Picked chips live in the header with full names and a remove button.
    assert "wv-cmp-pick" in body
    assert "wvRemoveCompareSlot" in script
    # Mobile: chips drop to their own full-width row under Compare + Clear.
    assert ".wv-cmp-picks { order: 3; flex-basis: 100%; }" in body
    assert "@media (max-width: 600px)" in body


def test_compare_slim_and_grouped_sections(script):
    assert "GAME CONTEXT" in script
    assert "wv-cmp-sec-h" in script
    # Verdict logic untouched: raw start_score, toss-up band, factor reasons.
    assert "const sa = a.start_score, sb = b.start_score;" in script
    # Slim tokens: verdict padding and the 5px factor tracks.
    assert "padding: 8px 12px" in CSS
    assert ".wv-cmp2-track { flex: 1; height: 5px;" in CSS
    assert ".wv-cmp2-head .pj { font-size: 22px;" in CSS
