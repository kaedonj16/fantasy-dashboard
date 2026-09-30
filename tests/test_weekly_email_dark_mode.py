"""Dark-mode email theme + de-duplicated cross-league action kickers."""
from __future__ import annotations

import re

from utils.cross_league_actions import make_action
from utils.digest_sections import (
    _dark_mode_css,
    email_shell,
    greeting_html,
    heading,
    league_summary_html,
    leagues_snapshot_table_html,
    thursday_alert_html,
    waiver_html,
)
from utils.weekly_email import cross_league_digest_html


def _sample_actions():
    return [
        make_action(
            kind="lineup", platform="sleeper", season=2026, league_id="l1",
            league_name="Revenge of the Picks",
            title="Injured starter needs a swap",
            detail="Mike Evans is listed Out", severity=9.0,
        ),
        make_action(
            kind="lineup", platform="sleeper", season=2026, league_id="l2",
            league_name="Big Dawg League",
            title="Injured starter needs a swap",
            detail="Justin Jefferson is listed Out", severity=8.0,
        ),
    ]


def test_email_shell_is_full_document():
    html = email_shell("<p>body</p>", subtitle="Test")
    assert html.startswith("<!DOCTYPE html>")
    assert "<head>" in html and "</head>" in html
    assert "<body" in html and "</body>" in html
    assert html.rstrip().endswith("</html>")


def test_email_shell_declares_color_scheme():
    html = email_shell("<p>body</p>", subtitle="Test")
    assert 'name="color-scheme" content="light dark"' in html
    assert 'name="supported-color-schemes" content="light dark"' in html


def test_email_shell_has_dark_media_query_and_ogsc():
    html = email_shell("<p>body</p>", subtitle="Test")
    assert "@media (prefers-color-scheme:dark)" in html
    assert "[data-ogsc]" in html


def test_dark_rules_use_important():
    css = _dark_mode_css()
    for sel, _decls in [
        (".em-wrap", None), (".em-card", None), (".em-t", None),
        (".em-sect", None), (".em-alert", None),
    ]:
        assert sel in css
    assert "!important" in css


def test_theme_classes_used_by_sections_are_covered():
    rendered = "".join([
        email_shell("<p>x</p>", subtitle="T"),
        greeting_html("Kaedon"),
        heading("Your leagues"),
        league_summary_html(league_name="L", rank=2, wins=3, losses=1,
                            format_label="SF", stakes_line="Win and in"),
        leagues_snapshot_table_html([{
            "name": "L", "href": "https://x", "chip": "SF",
            "standing": "#2 · 3-1", "focus": "vs Opp", "urgent": True,
        }]),
        thursday_alert_html([{"name": "Chase", "team": "CIN"}], compact=True),
        thursday_alert_html([{"name": "Chase", "team": "CIN", "kickoff": "8:15 PM"}]),
        waiver_html([{"name": "Waddle", "pos": "WR", "reason": "Target hog"}]),
        cross_league_digest_html(_sample_actions(), base_url="https://x", limit=4),
    ])
    used = set(re.findall(r'class="([^"]+)"', rendered))
    classes = {c for group in used for c in group.split()}
    themed = {c for c in classes if c.startswith("em-")}
    assert themed, "expected em-* theme classes in digest markup"
    css = _dark_mode_css()
    missing = sorted(c for c in themed if f".{c}" not in css)
    assert not missing, f"theme classes without dark rules: {missing}"


def test_cross_league_kicker_is_just_the_title():
    html = cross_league_digest_html(_sample_actions(), base_url="https://x", limit=4)
    assert "Across leagues" not in html
    kickers = re.findall(r'<div class="em-k"[^>]*>(.*?)</div>', html)
    assert len(kickers) == 2
    assert all(k == "Injured starter needs a swap" for k in kickers)


def test_cross_league_body_leads_with_bold_league():
    html = cross_league_digest_html(_sample_actions(), base_url="https://x", limit=4)
    assert "<strong>Revenge of the Picks</strong> · Mike Evans is listed Out" in html
    assert "<strong>Big Dawg League</strong> · Justin Jefferson is listed Out" in html


def test_cross_league_action_without_league_still_renders():
    acts = [make_action(kind="lineup", platform="sleeper", season=2026,
                        league_id="l1", title="Set your lineup",
                        detail="2 starters on bye", severity=5.0)]
    html = cross_league_digest_html(acts, base_url="https://x", limit=4)
    assert "Set your lineup" in html
    assert "2 starters on bye" in html


def test_light_mode_defaults_unchanged():
    html = email_shell(greeting_html("Kaedon"), subtitle="Test")
    assert "background:#ffffff" in html  # card stays white in light mode
    assert "color:#0f172a" in html  # primary text stays dark in light mode
    assert "background:#0b1220" in html  # navy masthead unchanged
