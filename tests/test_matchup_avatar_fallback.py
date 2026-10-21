"""Matchup header must stay symmetric when a team has no logo.

Kaedon's Week 7 ESPN case (2026-10-21): "Kyle Pitts Sucks Ass" has no
team logo, "Cook and Ball Torture" does. The header rendered lopsided --
left score flush to the card edge, right score + avatar packed into the
middle, right third of the row empty.

Root cause: the avatar <img> carried onerror="this.style.display='none'".
When the logo URL fails (ESPN serves logo URLs that 404 for logo-less
teams), display:none removes the grid item, grid auto-placement shifts
the remaining items one track left, and the 1fr scores reflow into the
middle. Verified in Chromium on the pre-fix markup: score-l collapsed to
content width at the row's left edge and the right avatar sat at ~55%
of the row width instead of the far right.

Fix: a missing or failed logo swaps in the generated monogram crest
(team_crest_data_uri -- the same last-resort identity team_avatar uses)
and the score row's avatar tracks are a fixed 52px, so the geometry
never depends on a logo existing or loading.
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _matchups_mod():
    import dashboard_services.matchups as m
    return m


def _team(name, username, record, avatar, pts, proj, rid):
    return {
        "name": name, "username": username, "record": record,
        "avatar": avatar, "pts_total": pts, "proj_total": proj,
        "roster_id": rid, "owner_id": f"u{rid}", "starters": [],
    }


def _render(m, left_avatar, right_avatar):
    left = _team("Kyle Pitts Sucks Ass", "jtripp17", "2-4",
                 left_avatar, 129.1, 125.5, 1)
    right = _team("Cook and Ball Torture", "ESPNfan58801560", "5-1",
                  right_avatar, 114.3, 114.8, 2)
    return m.render_matchup_slide(
        "2026", {"left": left, "right": right, "h2h": {}}, 7, 7,
        status_by_pid={}, projections={}, players={}, teams={},
        team_game_lookup={}, compact=True,
    )


def _score_row_imgs(html):
    i = html.find("m-espn-score-row")
    assert i >= 0, "score row missing"
    return re.findall(r"<img class='m-espn-av'[^>]*>", html[i:i + 3000])


def test_missing_logo_renders_crest_not_empty_slot():
    m = _matchups_mod()
    html = _render(m, "", "https://example.com/logo.png")
    imgs = _score_row_imgs(html)
    assert len(imgs) == 2
    # Left (logo-less) team: crest data URI in the same img slot.
    assert "data:image/svg+xml" in imgs[0]
    # No invisible placeholder div holding the slot.
    assert "<div class='m-espn-av'></div>" not in html


def test_broken_logo_url_swaps_to_crest_on_error():
    m = _matchups_mod()
    html = _render(m, "https://invalid.example.com/nope.png",
                   "https://example.com/logo.png")
    imgs = _score_row_imgs(html)
    assert len(imgs) == 2
    for img in imgs:
        # The old collapse path must never come back.
        assert "display='none'" not in img and 'display="none"' not in img
        assert "this.style.display" not in img
    # Left img keeps its real URL but falls back to the crest on error.
    assert "invalid.example.com/nope.png" in imgs[0]
    assert "this.onerror=null" in imgs[0]
    assert "data:image/svg+xml" in imgs[0]


def test_both_logos_render_plain_imgs_with_crest_fallback():
    m = _matchups_mod()
    html = _render(m, "https://example.com/a.png", "https://example.com/b.png")
    imgs = _score_row_imgs(html)
    assert len(imgs) == 2
    assert "example.com/a.png" in imgs[0]
    assert "example.com/b.png" in imgs[1]


def test_score_row_avatar_tracks_are_fixed_width():
    css = open("static/dashboard.css", encoding="utf-8").read()
    rule = re.search(r"\.m-espn-score-row \{([^}]*)\}", css)
    assert rule, "score row rule missing"
    assert "grid-template-columns: 52px 1fr 1fr 52px" in rule.group(1)
