"""Nav dropdown reorganization contract (2026-11-02).

The league Weekly dropdown (and its mobile More-sheet panel) is grouped
into Play and Review with ScoreZone second; NFL Team Rankings moved from
Players to Stats in the league nav only. The guest nav has no Stats
dropdown, so NFL Team Rankings stays in guest Players. The mobile dock
tabs do not change.
"""
import re

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")


@pytest.fixture()
def nav_env(monkeypatch):
    import app

    monkeypatch.setattr(
        app, "get_nfl_state",
        lambda: {"season": "2026", "season_type": "reg", "week": 4},
    )
    monkeypatch.setattr(app, "get_league_ctx_from_cache", lambda *a, **k: {})
    monkeypatch.setattr(app, "has_draft_ended", lambda *a, **k: True)
    monkeypatch.setattr(app, "_games_live_or_imminent", lambda *a, **k: False)
    monkeypatch.setattr(app, "_breakouts_are_new", lambda: False)
    monkeypatch.setattr(app, "_nav_show_keeper", lambda *a, **k: True)
    monkeypatch.setattr(app, "_nav_is_best_ball", lambda *a, **k: False)
    return app


def _clean(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Strip trailing badge text (PRO / NEW) from item labels.
    while text.endswith(" PRO") or text.endswith(" NEW"):
        text = text.rsplit(" ", 1)[0]
    return text


def _menu_tokens(html, menu_id):
    """Ordered (kind, label) tokens for one desktop dropdown menu.

    kind is "header" for a group header span, "item" for a link.
    """
    menu = re.search(rf"id='{menu_id}'.*?</div>", html, re.S)
    assert menu, f"{menu_id} not rendered"
    tokens = []
    for m in re.finditer(
        r"<span class='nav-pill-dropdown-header'>([^<]+)</span>"
        r"|<a class='nav-pill-dropdown-item[^']*'[^>]*>(.*?)</a>",
        menu.group(0), re.S,
    ):
        if m.group(1) is not None:
            tokens.append(("header", m.group(1)))
        else:
            tokens.append(("item", _clean(m.group(2))))
    return tokens


def _panel_tokens(html, panel):
    """Ordered (kind, label) tokens for one mobile More-sheet panel.

    kind is "label" for a group label row, "link" for a sheet link.
    """
    start = html.index(f"data-br-sheet-panel='{panel}'")
    panel_html = html[start:html.index("</section>", start)]
    tokens = []
    for m in re.finditer(
        r"<div class='br-sheet-group-label'>([^<]+)</div>"
        r"|<a class='br-sheet-link[^']*'[^>]*>.*?<span>([^<]+)</span>",
        panel_html, re.S,
    ):
        if m.group(1) is not None:
            tokens.append(("label", m.group(1)))
        else:
            tokens.append(("link", m.group(2)))
    return tokens


def _league_nav(nav_env, active="dashboard"):
    app = nav_env
    with app.app.test_request_context("/sleeper/2026/L/dashboard"):
        app.session["account_id"] = 42
        return app.build_nav("L", active, "sleeper", 2026)


def _league_sheet(nav_env, active="dashboard"):
    app = nav_env
    with app.app.test_request_context("/sleeper/2026/L/dashboard"):
        app.session["account_id"] = 42
        return app._mobile_nav(active, "L", "sleeper", 2026)


WEEKLY_DESKTOP = [
    ("header", "Play"),
    ("item", "Matchups"),
    ("item", "ScoreZone"),
    ("item", "Waivers & Start/Sit"),
    ("item", "Lineup Lab"),
    ("item", "Opponent Scout"),
    ("item", "Schedule Rankings"),
    ("header", "Review"),
    ("item", "Weekly Recap"),
    ("item", "Lineup Efficiency"),
]

WEEKLY_SHEET = [
    ("label", "Play"),
    ("link", "Matchups"),
    ("link", "ScoreZone"),
    ("link", "Waivers & Start/Sit"),
    ("link", "Lineup Lab"),
    ("link", "Opponent Scout"),
    ("link", "Schedule Rankings"),
    ("label", "Review"),
    ("link", "Weekly Recap"),
    ("link", "Lineup Efficiency"),
]

PLAYERS_ORDER = [
    "Player Rankings", "Top Movers", "Compare Players",
    "Advanced Metrics", "Breakout Engine", "Prospect Rankings",
]

STATS_ORDER = ["Awards", "Graphs", "League History", "NFL Team Rankings"]


def test_desktop_weekly_grouped_play_then_review(nav_env):
    html = _league_nav(nav_env)
    assert _menu_tokens(html, "weeklyNavMenu") == WEEKLY_DESKTOP


def test_sheet_weekly_grouped_play_then_review(nav_env):
    html = _league_sheet(nav_env)
    assert _panel_tokens(html, "weekly") == WEEKLY_SHEET


def test_desktop_weekly_best_ball_keeps_groups_without_lab(nav_env, monkeypatch):
    monkeypatch.setattr(nav_env, "_nav_is_best_ball", lambda *a, **k: True)
    tokens = _menu_tokens(_league_nav(nav_env), "weeklyNavMenu")
    labels = [label for _kind, label in tokens]
    assert ("header", "Play") in tokens and ("header", "Review") in tokens
    assert "Waivers" in labels and "Waivers & Start/Sit" not in labels
    assert "Lineup Lab" not in labels
    assert labels.index("ScoreZone") == 2  # header, Matchups, ScoreZone


def test_sheet_weekly_best_ball_keeps_groups_without_lab(nav_env, monkeypatch):
    monkeypatch.setattr(nav_env, "_nav_is_best_ball", lambda *a, **k: True)
    tokens = _panel_tokens(_league_sheet(nav_env), "weekly")
    labels = [label for _kind, label in tokens]
    assert ("label", "Play") in tokens and ("label", "Review") in tokens
    assert "Waivers" in labels and "Waivers & Start/Sit" not in labels
    assert "Lineup Lab" not in labels


def test_desktop_players_order_without_nfl_teams(nav_env):
    html = _league_nav(nav_env)
    tokens = _menu_tokens(html, "playersNavMenu")
    assert [label for _kind, label in tokens] == PLAYERS_ORDER
    assert all(kind == "item" for kind, _label in tokens)


def test_sheet_players_order_without_nfl_teams(nav_env):
    html = _league_sheet(nav_env)
    tokens = _panel_tokens(html, "players")
    assert [label for _kind, label in tokens] == PLAYERS_ORDER


def test_desktop_stats_ends_with_nfl_team_rankings(nav_env):
    html = _league_nav(nav_env)
    tokens = _menu_tokens(html, "statsNavMenu")
    assert [label for _kind, label in tokens] == STATS_ORDER


def test_sheet_stats_ends_with_nfl_team_rankings(nav_env):
    html = _league_sheet(nav_env)
    tokens = _panel_tokens(html, "stats")
    assert [label for _kind, label in tokens] == STATS_ORDER


def test_nfl_teams_page_lights_stats_pill_in_league_nav(nav_env):
    html = _league_nav(nav_env, active="nfl-teams")
    assert "class='nav-pill active' id='statsNavBtn'" in html
    assert "class='nav-pill active' id='playersNavBtn'" not in html


def test_guest_desktop_players_still_has_nfl_team_rankings(nav_env):
    app = nav_env
    with app.app.test_request_context("/"):
        html = app.build_nav(None, "players", "sleeper", 2026)
    labels = [label for _kind, label in _menu_tokens(html, "playersNavMenu")]
    assert "NFL Team Rankings" in labels


def test_guest_sheet_players_still_has_nfl_team_rankings(nav_env):
    app = nav_env
    with app.app.test_request_context("/players"):
        html = app._mobile_nav_guest("players")
    labels = [label for _kind, label in _panel_tokens(html, "players")]
    assert "NFL Team Rankings" in labels


def test_inseason_dock_unchanged(nav_env):
    app = nav_env
    with app.app.test_request_context("/x"):
        html = app._mobile_nav("dashboard", "L", "sleeper", 2026)
    labels = re.findall(r"br-tabbar-lbl'>([^<]+)<", html)
    assert labels == ["Home", "Matchups", "Trades", "Teams", "More"]


def test_group_header_styles_exist():
    from pathlib import Path
    css = (Path(__file__).resolve().parents[1] / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert ".nav-pill-dropdown-header" in css
    assert ".br-sheet-group-label" in css
    # Both rules land before the merged icons section.
    marker = css.index("MERGED: icons.css")
    assert css.index(".nav-pill-dropdown-header") < marker
    assert css.index(".br-sheet-group-label") < marker
