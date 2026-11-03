"""Week Leaders segment redesign (2026-11-03).

Pins the uniform segment skeleton emitted by _render_weekly_highlights:
head row, hero row, score-share bar, team zone, in that order for all four
cards, with Closest Game and Biggest Blowout structurally identical
(they are both matchup cards; only accent, icon, label, margin, bar width
and teams differ).
"""
import re

import pytest

# app.py imports pandas (and flask) at module load. The fast "lint" CI shard
# has flask but not pandas, so guard on pandas first, otherwise importing app
# here raises at COLLECTION and aborts the whole run.
pytest.importorskip("pandas")
pytest.importorskip("flask")
pytest.importorskip("openai")  # app.py pulls openai via dashboard_services.ai.client

import pandas as pd  # noqa: E402

from app import _render_weekly_highlights  # noqa: E402


def _fixture_df():
    # Week 9, two matchups. Triple B II 121.3 vs Fearless 74.2 is both the
    # highest/lowest pair and the blowout; Spartyhater 100.0 vs Tua 95.9 is
    # the closest game.
    rows = [
        {"week": 9, "owner": "Triple B II", "roster_id": "1", "matchup_id": 1,
         "points": 121.3, "finalized": True},
        {"week": 9, "owner": "Fearless And Ready To Rumble", "roster_id": "2",
         "matchup_id": 1, "points": 74.2, "finalized": True},
        {"week": 9, "owner": "Spartyhater", "roster_id": "3", "matchup_id": 2,
         "points": 100.0, "finalized": True},
        {"week": 9, "owner": "Tua in the Pink", "roster_id": "4", "matchup_id": 2,
         "points": 95.9, "finalized": True},
    ]
    return pd.DataFrame(rows)


def _cards(html):
    parts = re.split(r'(?=<div class="whl-card whl-seg )', html)
    return [p for p in parts if p.startswith('<div class="whl-card whl-seg ')]


def _card(html, seg):
    for part in _cards(html):
        if f"whl-seg-{seg}" in part.split(">", 1)[0]:
            return part
    raise AssertionError(f"segment {seg} not found in: {html[:400]}")


@pytest.fixture(scope="module")
def html():
    return _render_weekly_highlights(_fixture_df(), 9)


def test_all_four_segment_classes_present(html):
    for seg in ("high", "low", "close", "blow"):
        assert f"whl-card whl-seg whl-seg-{seg}" in html


def test_skeleton_order_head_hero_bar_teams(html):
    for part in _cards(html):
        head = part.index("whl-head")
        hero = part.index("whl-hero")
        bar = part.index("whl-bar")
        teams = part.index("whl-teams")
        assert head < hero < bar < teams


def test_closest_and_blowout_structure_identical(html):
    def class_sequence(fragment):
        seq = re.findall(r'class="([^"]+)"', fragment)
        # Mask the per-segment icon class; everything else must match.
        return [re.sub(r"fa-solid fa-[a-z-]+", "fa-solid fa-ICON", c) for c in seq]

    def tag_sequence(fragment):
        return re.findall(r"<(div|span|i)[ >]", fragment)

    close = _card(html, "close")
    blow = _card(html, "blow")
    # Mask the segment-specific class on the outer card, then the class
    # sequences and tag sequences must be identical.
    assert class_sequence(close)[1:] == class_sequence(blow)[1:]
    assert tag_sequence(close) == tag_sequence(blow)


def test_bar_widths_are_game_score_shares(html):
    # Winner/focal share of the game's total points.
    assert 'style="width:62.0%"' in _card(html, "high")   # 121.3 / 195.5
    assert 'style="width:38.0%"' in _card(html, "low")    # 74.2 / 195.5
    assert 'style="width:51.0%"' in _card(html, "close")  # 100.0 / 195.9
    assert 'style="width:62.0%"' in _card(html, "blow")   # 121.3 / 195.5


def test_single_captions_name_opponent_and_score(html):
    assert "Beat Fearless And Ready To Rumble 74.2" in _card(html, "high")
    assert "Lost to Triple B II 121.3" in _card(html, "low")


def test_matchup_hero_is_the_margin(html):
    assert "4.1<small>pt margin</small>" in _card(html, "close")
    assert "47.1<small>pt margin</small>" in _card(html, "blow")


def test_highest_keeps_countup_moment_attrs(html):
    high = _card(html, "high")
    assert "br-hs" in high.split(">", 1)[0]
    assert 'data-br-moment="highscore"' in high
    assert 'data-mo-count="121.3"' in high
    assert 'data-mo-dp="1"' in high


def test_avatar_initials_skip_stopwords(html):
    assert ">TB<" in _card(html, "high")
    assert ">FR<" in _card(html, "low")
    assert ">TP<" in _card(html, "close")  # Tua in the Pink -> TP, not TI


def test_no_old_margin_markup(html):
    assert "whl-margin" not in html


def test_avatar_initials_skip_punctuation():
    df = pd.DataFrame([
        {"week": 9, "owner": "Hit the (Sa)Quon", "roster_id": "1",
         "matchup_id": 1, "points": 121.3, "finalized": True},
        {"week": 9, "owner": "Other Team", "roster_id": "2",
         "matchup_id": 1, "points": 74.2, "finalized": True},
    ])
    out = _render_weekly_highlights(df, 9)
    assert ">HS<" in out  # Hit (Sa)Quon -> HS, never "H("
    assert ">H(<" not in out


def test_real_avatar_images_when_resolved():
    out = _render_weekly_highlights(
        _fixture_df(), 9,
        avatars_by_roster={"1": "https://sleepercdn.com/avatars/thumbs/abc"},
    )
    high = _card(out, "high")
    assert ('<img class="whl-ava whl-ava-img" '
            'src="https://sleepercdn.com/avatars/thumbs/abc"') in high
    # A roster with no resolved picture keeps the initials disc.
    assert ">FR<" in _card(out, "low")
