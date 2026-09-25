"""Page-source tests for the waivers personalization banner + K/DST tabs.

The banner shows the exact "Link your team" copy whenever ?rid= didn't resolve
to a roster (missing or stale), is dismissible per league, and routes through
the dashboard's team-link flow. K/DST pills stay hidden until the league is
known to start those positions, and selecting them fetches a position-filtered
waiver-candidate list.
"""
import re

from dashboard_services.pages.waivers_page import build_waivers_body

BANNER_COPY = ("Link your team for personalized ranks, drop suggestions, "
               "and lineup-gain projections.")


def _html():
    return build_waivers_body("sleeper", 2026, "123", {})


def test_banner_copy_exact_and_no_em_dash():
    html = _html()
    assert BANNER_COPY in html
    banner = html.split('id="wvLinkBanner"')[1].split("</div>", 1)[0]
    assert "\u2014" not in banner


def test_banner_dismissible_per_league():
    html = _html()
    # Dismissal is remembered per league in localStorage; the X wires to it.
    assert "wv_link_banner_dismissed:" in html
    assert "wvDismissLinkBanner()" in html
    assert "localStorage.setItem('wv_link_banner_dismissed:'" in html


def test_banner_cta_uses_team_link_flow():
    html = _html()
    assert "wvLinkTeam()" in html
    assert "window.linkMyTeam" in html
    # Banner visibility is driven by the API's explicit personalized flag.
    assert "wvRenderLinkBanner(d.personalized === true)" in html


def test_kd_pills_hidden_until_league_known_to_use_them():
    html = _html()
    # Both pills start hidden; the sync functions reveal them from either the
    # start-sit requirements or the waiver-candidates league flags.
    assert re.search(r'data-pos="K"[^>]*hidden', html)
    assert re.search(r'data-pos="DEF"[^>]*hidden', html)
    assert "wvUsesK = wvUsesK || d.league_uses_k === true" in html
    assert "wvUsesDef = wvUsesDef || d.league_uses_def === true" in html


def test_kd_tabs_fetch_position_filtered_candidates():
    html = _html()
    # Selecting K/DEF refetches with ?position= so the tabs are server-ranked
    # by the streaming rankers instead of filtering skill-position rows.
    assert "requestedPos" in html
    assert "'&position=' + encodeURIComponent(requestedPos)" in html


def test_streaming_strip_removed():
    html = _html()
    assert "wvStreamWrap" not in html
    assert "wvRenderStreaming" not in html
    assert "/api/streaming-options" not in html
