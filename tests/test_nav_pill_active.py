"""Top-nav active-pill contract: every page lights its own nav pill.

The desktop outline is the gliding .nav-pill-ind, which app.js positions
behind the one pill carrying .nav-pill.active. If build_nav never grants
.active, the page shows no outline at all. That is what happened on Advanced
Metrics and the Trade Hub: both are items inside a dropdown whose hand-kept
active_keys list omitted their keys (players dropdown missed
"advanced-metrics", trades missed "trade-suggestions"), in BOTH the league
nav and the global nav.

These tests render build_nav for every nav page and assert exactly one pill
is active and it is the right parent. They also pin the source contract
(item keys subset of each dropdown's active_keys) so the lists cannot drift
again, and Draft History's active key (it passed "draft", so the Draft Room
item highlighted instead of Draft History).
"""
import re
from pathlib import Path

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[1]

# Dropdown button id -> the label a user would name for that section.
LEAGUE_PARENT = {
    "tradesNavBtn": "Trades",
    "weeklyNavBtn": "Weekly",
    "teamsNavBtn": "League",
    "playersNavBtn": "Players",
    "draftNavBtn": "Draft",
    "statsNavBtn": "Stats",
}
GLOBAL_PARENT = {
    "tradesNavBtn": "Trades",
    "playersNavBtn": "Players",
    "draftNavBtn": "Draft",
    "learnNavBtn": "Learn",
}


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


def _active_pills(html):
    """(dropdown btn ids carrying .active, plain anchor-pill labels carrying .active).

    Dropdown *items* also render as anchors, but with the distinct
    nav-pill-dropdown-item class, so plain pills are matched by their exact
    class attribute ('nav-pill' / 'nav-pill active') only.
    """
    btns = [
        m.group(2)
        for m in re.finditer(r"class='(nav-pill[^']*)' id='(\w+Btn)'", html)
        if "active" in m.group(1).split()
    ]
    anchors = [
        m.group(2).strip()
        for m in re.finditer(r"<a class='(nav-pill active|nav-pill)'[^>]*>([^<]+)</a>", html)
        if m.group(1) == "nav-pill active"
    ]
    return btns, anchors


def _assert_only_parent(html, parents, btn_id=None, anchor_label=None):
    btns, anchors = _active_pills(html)
    if btn_id is not None:
        assert btns == [btn_id], f"expected only {parents[btn_id]} active, got {btns}"
        assert anchors == [], f"no anchor pill should be active, got {anchors}"
    else:
        assert btns == [], f"no dropdown should be active, got {btns}"
        assert anchors == [anchor_label], f"expected {anchor_label!r} active, got {anchors}"
    # Exactly one active pill overall: the gliding indicator needs exactly one.
    assert html.count("nav-pill active") == 1


LEAGUE_DROPDOWN_CASES = [
    ("players", "playersNavBtn"), ("compare", "playersNavBtn"),
    ("top-movers", "playersNavBtn"), ("advanced-metrics", "playersNavBtn"),
    ("nfl-teams", "playersNavBtn"), ("breakouts", "playersNavBtn"),
    ("prospects", "playersNavBtn"),
    ("trade", "tradesNavBtn"), ("trade-database", "tradesNavBtn"),
    ("standings", "teamsNavBtn"), ("teams", "teamsNavBtn"),
    ("activity", "teamsNavBtn"), ("league_health", "teamsNavBtn"),
    ("weekly", "weeklyNavBtn"), ("recap", "weeklyNavBtn"),
    ("waivers", "weeklyNavBtn"), ("lineup-lab", "weeklyNavBtn"),
    ("schedule", "weeklyNavBtn"),
    ("scorezone", "weeklyNavBtn"),
    ("draft", "draftNavBtn"), ("draft-cheat-sheet", "draftNavBtn"),
    ("draft-history", "draftNavBtn"), ("keeper", "draftNavBtn"),
    ("awards", "statsNavBtn"), ("graphs", "statsNavBtn"),
    ("history", "statsNavBtn"),
]


@pytest.mark.parametrize("active,btn_id", LEAGUE_DROPDOWN_CASES)
def test_league_nav_lights_parent_pill(nav_env, active, btn_id):
    app = nav_env
    with app.app.test_request_context("/sleeper/2026/L/dashboard"):
        app.session["account_id"] = 42
        html = app.build_nav("L", active, "sleeper", 2026)
    _assert_only_parent(html, LEAGUE_PARENT, btn_id=btn_id)


@pytest.mark.parametrize("active,label", [("dashboard", "Dashboard"), ("portfolio", "My Leagues")])
def test_league_nav_plain_pills(nav_env, active, label):
    app = nav_env
    with app.app.test_request_context("/sleeper/2026/L/dashboard"):
        app.session["account_id"] = 42
        html = app.build_nav("L", active, "sleeper", 2026)
    _assert_only_parent(html, LEAGUE_PARENT, anchor_label=label)


@pytest.mark.parametrize("path,active,btn_id", [
    ("/sleeper/2026/L/trade?tab=suggestions", "trade", "tradesNavBtn"),
    ("/sleeper/2026/L/weekly?tab=scout", "weekly", "weeklyNavBtn"),
    ("/sleeper/2026/L/weekly?tab=optimal", "weekly", "weeklyNavBtn"),
    ("/sleeper/2026/L/waivers?tab=lab", "waivers", "weeklyNavBtn"),
])
def test_league_nav_tab_subpages_light_parent(nav_env, path, active, btn_id):
    app = nav_env
    with app.app.test_request_context(path):
        app.session["account_id"] = 42
        html = app.build_nav("L", active, "sleeper", 2026)
    _assert_only_parent(html, LEAGUE_PARENT, btn_id=btn_id)


GLOBAL_DROPDOWN_CASES = [
    ("players", "playersNavBtn"), ("compare", "playersNavBtn"),
    ("top-movers", "playersNavBtn"), ("advanced-metrics", "playersNavBtn"),
    ("nfl-teams", "playersNavBtn"), ("breakouts", "playersNavBtn"),
    ("prospects", "playersNavBtn"),
    ("trade", "tradesNavBtn"), ("trade-database", "tradesNavBtn"),
    ("draft", "draftNavBtn"), ("draft-cheat-sheet", "draftNavBtn"),
    ("draft-history", "draftNavBtn"),
    ("guides", "learnNavBtn"), ("glossary", "learnNavBtn"),
    ("faq", "learnNavBtn"), ("about", "learnNavBtn"),
]


@pytest.mark.parametrize("active,btn_id", GLOBAL_DROPDOWN_CASES)
def test_global_nav_lights_parent_pill(nav_env, active, btn_id):
    app = nav_env
    with app.app.test_request_context("/"):
        app.session["viewer_username"] = "kaedon"
        html = app.build_nav(None, active, "sleeper", 2026)
    _assert_only_parent(html, GLOBAL_PARENT, btn_id=btn_id)


@pytest.mark.parametrize("active,label", [("home", "Home"), ("portfolio", "My Leagues")])
def test_global_nav_plain_pills(nav_env, active, label):
    app = nav_env
    with app.app.test_request_context("/"):
        app.session["viewer_username"] = "kaedon"
        html = app.build_nav(None, active, "sleeper", 2026)
    _assert_only_parent(html, GLOBAL_PARENT, anchor_label=label)


def test_global_nav_trade_hub_tab_lights_trades(nav_env):
    app = nav_env
    with app.app.test_request_context("/trade?tab=suggestions"):
        html = app.build_nav(None, "trade", "sleeper", 2026)
    _assert_only_parent(html, GLOBAL_PARENT, btn_id="tradesNavBtn")


def test_dropdown_item_keys_covered_by_active_keys():
    """Source contract: no dropdown item key may be missing from the
    active_keys list handed to the same dropdown call (the original drift)."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    nav_src = src[src.index("def build_nav"):src.index("def render_page")]
    calls = re.findall(
        r"(?:simple_dropdown|nav_pill_dropdown)\(\s*\"([^\"]+)\",\s*\[(.*?)\]\s*,\s*\[([^\]]*)\]",
        nav_src, re.S,
    )
    assert calls, "expected to find static dropdown calls in build_nav"
    for label, items_block, active_block in calls:
        item_keys = []
        for tup in re.findall(r"\((.*?)\)", items_block, re.S):
            strs = re.findall(r'"([^"]+)"', tup)
            if len(strs) >= 3:
                item_keys.append(strs[2])
        active_keys = re.findall(r'"([^"]+)"', active_block)
        missing = [k for k in item_keys if k not in active_keys]
        assert not missing, f"{label} dropdown items missing from active_keys: {missing}"
    # The dynamic Weekly/Draft league dropdowns build their item lists first;
    # their literal active lists must cover every key those lists can add
    # (ScoreZone is appended to _weekly_items in season; Keeper to _draft_items
    # for keeper leagues).
    weekly_active = re.search(
        r'"Weekly", _weekly_items,\s*\[([^\]]*)\]', nav_src, re.S)
    assert weekly_active, "league Weekly dropdown call not found"
    weekly_keys = re.findall(r'"([^"]+)"', weekly_active.group(1))
    for key in ("weekly", "recap", "scout", "optimal", "waivers", "lineup-lab", "schedule", "scorezone"):
        assert key in weekly_keys, f"Weekly active_keys missing {key}"
    draft_active = re.search(
        r'nav_pill_dropdown\("Draft", _draft_items,\s*\[([^\]]*)\]', nav_src, re.S)
    assert draft_active, "league Draft dropdown call not found"
    draft_keys = re.findall(r'"([^"]+)"', draft_active.group(1))
    for key in ("draft", "draft-cheat-sheet", "draft-history", "keeper"):
        assert key in draft_keys, f"Draft active_keys missing {key}"


def test_dropdown_helpers_derive_active_from_items():
    """Belt and braces: even if a future caller's list drifts again, the
    helpers themselves light the parent for any of their own item keys."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    nav_src = src[src.index("def build_nav"):src.index("def render_page")]
    assert nav_src.count("active in item_keys") == 2  # simple_dropdown + nav_pill_dropdown


def test_draft_history_page_uses_its_own_active_key():
    src = (ROOT / "routes" / "tool_pages_bp.py").read_text(encoding="utf-8")
    fn = src[src.index("def page_draft_history"):]
    fn = fn[:fn.index("\n@tool_pages_bp", 1)] if "\n@tool_pages_bp" in fn[1:] else fn
    assert '"draft-history"' in fn
    assert 'league_id, "draft",' not in fn


def _weekly_menu_items(html):
    """(label, href, cls) for each item in the league Weekly dropdown, in order."""
    menu = re.search(r"id='weeklyNavMenu'.*?</div>", html, re.S)
    assert menu, "Weekly dropdown menu not rendered"
    return re.findall(
        r"<a class='(nav-pill-dropdown-item[^']*)'(?: aria-current='page')? "
        r"href='([^']*)'>([^<]+)</a>",
        menu.group(0),
    )


def test_lineup_lab_item_sits_under_start_sit_in_weekly_dropdown(nav_env):
    app = nav_env
    with app.app.test_request_context("/sleeper/2026/L/waivers"):
        app.session["account_id"] = 42
        html = app.build_nav("L", "waivers", "sleeper", 2026)
    items = _weekly_menu_items(html)
    labels = [label for _cls, _href, label in items]
    assert "Lineup Lab" in labels
    i = labels.index("Lineup Lab")
    # Directly beneath Waivers & Start/Sit, deep-linking into the Lab view.
    assert labels[i - 1] == "Waivers & Start/Sit"
    assert items[i][1].endswith("/waivers?tab=lab")


def test_lineup_lab_item_is_the_active_one_on_lab_deep_link(nav_env):
    app = nav_env
    with app.app.test_request_context("/sleeper/2026/L/waivers?tab=lab"):
        app.session["account_id"] = 42
        html = app.build_nav("L", "waivers", "sleeper", 2026)
    _assert_only_parent(html, LEAGUE_PARENT, btn_id="weeklyNavBtn")
    items = {label: cls for cls, _href, label in _weekly_menu_items(html)}
    assert "active" in items["Lineup Lab"].split()
    assert "active" not in items["Waivers & Start/Sit"].split()


def test_mobile_sheet_lineup_lab_row_under_waivers(nav_env):
    app = nav_env
    with app.app.test_request_context("/sleeper/2026/L/waivers?tab=lab"):
        app.session["account_id"] = 42
        sheet = app._mobile_nav("waivers", "L", "sleeper", 2026)
    # The Lab row deep-links to the Lab view and is the marked row on
    # ?tab=lab; the Waivers row is not.
    assert ("<a class='br-sheet-link active' aria-current='page' "
            "href='/sleeper/2026/L/waivers?tab=lab'>") in sheet
    assert "<a class='br-sheet-link' href='/sleeper/2026/L/waivers'>" in sheet
    assert sheet.index("Waivers & Start/Sit") < sheet.index("Lineup Lab")
    assert sheet.index("Lineup Lab") < sheet.index("Schedule Assistant")
