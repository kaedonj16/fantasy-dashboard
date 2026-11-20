"""Guest ROS rankings polish (2026-11-20): explainer dropdown, table
panel CSS, and the lite-shell no-reload swapper.

The ROS explainer renders as a collapsed native <details> (prose kept in
the HTML for SEO, no "optional read" label), the shared .rnk-* table is
one unified panel with a sticky header and theme-aware hover, and
logged-out lite shells carry the lite_nav.js swapper plus the br-shell
marker it checks on both ends of a soft navigation.
"""
import os

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fake_ros_payload():
    players = {
        "rb1": {"ros_ppg": 20.0, "ros_total": 140.0, "ros_games": 7,
                "ros_sched_rank": 4, "ros_pos_rank": 1},
        "rb2": {"ros_ppg": 18.0, "ros_total": 108.0, "ros_games": 6,
                "ros_sched_rank": 9, "ros_pos_rank": 2},
    }
    table = [
        {"id": "rb1", "name": "Back One", "team": "DET", "position": "RB"},
        {"id": "rb2", "name": "Back Two", "team": "MIA", "position": "RB"},
    ]
    return ({"players": players, "window": {"start_week": 11, "end_week": 17},
             "updated": "November 20, 2026"}, table)


# ── Explainer dropdown ───────────────────────────────────────────────────

@pytest.mark.parametrize("pos", [None, "QB", "RB", "WR", "TE"])
def test_ros_explainer_is_collapsed_details(pos):
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    payload, table = _fake_ros_payload()
    body = build_ros_rankings_body(payload, table, position=pos)
    # Native details, collapsed by default (no open attribute), heading
    # inside the summary, prose still present in the HTML.
    assert '<details class="rnk-analysis rnk-analysis-details">' in body
    assert '<details class="rnk-analysis rnk-analysis-details" open' not in body
    assert '<summary class="rnk-analysis-summary">' in body
    assert 'class="rnk-analysis-title"' in body
    assert 'class="rnk-analysis-chevron"' in body
    assert '<div class="rnk-analysis-body">' in body
    assert "optional read" not in body.lower()
    # The old always-open section markup is gone from ROS pages.
    assert '<section class="rnk-analysis">' not in body


def test_ros_explainer_keeps_prose_for_seo():
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    payload, table = _fake_ros_payload()
    body = build_ros_rankings_body(payload, table, position="RB")
    assert "Rest-of-season RB rankings" in body
    assert "volume bet" in body  # RB prose, inside the closed details


def test_ros_table_primary_and_sched_markup():
    from dashboard_services.pages.dynasty_pages import build_ros_rankings_body

    payload, table = _fake_ros_payload()
    body = build_ros_rankings_body(payload, table, position="RB")
    # ROS Pts is the primary column in header and cells; Sched is a pill;
    # Exp. Games / Sched carry the phone-drop column classes.
    assert '<th class="rnk-val rnk-val-primary">ROS Pts</th>' in body
    assert '<td class="rnk-val rnk-val-primary">140.0</td>' in body
    assert '<span class="rnk-sched-pill">#4</span>' in body
    assert '<th class="rnk-val rnk-col-games">Exp. Games</th>' in body
    assert '<th class="rnk-age rnk-col-sched">Sched</th>' in body


# ── CSS pins ─────────────────────────────────────────────────────────────

def _rnk_css_block():
    css = open(os.path.join(_ROOT, "static", "dashboard.css"),
               encoding="utf-8").read()
    start = css.index("/* ── Rankings Hub")
    end = css.index("/* ── Share Card Modal")
    return css[start:end]


def test_rnk_css_panel_and_theme_fixes():
    block = _rnk_css_block()
    # The verified defects are gone from the rankings block.
    assert "rgba(30, 58, 95" not in block   # heavy hardcoded row lines
    assert "#38bdf8" not in block          # hardcoded sky-blue active nav
    # Unified panel + sticky header + scroll region behaviour.
    assert ".rnk-table-wrap" in block
    assert "overscroll-behavior: contain" in block
    assert "scrollbar-width: thin" in block
    assert ".rnk-table thead th" in block
    assert "position: sticky" in block
    # Theme-aware hover and hairline rows.
    assert ".rnk-table tr:hover td" in block
    assert "background: var(--row)" in block
    assert "border-bottom: 1px solid var(--border)" in block
    # Segmented nav on accent tokens, primary value, sched pill, phone drop.
    assert ".rnk-nav-btn-active" in block
    assert "background: var(--accent)" in block
    assert ".rnk-val-primary" in block
    assert ".rnk-sched-pill" in block
    assert ".rnk-col-games" in block and ".rnk-col-sched" in block
    assert "display: none" in block
    # Details dropdown chrome.
    assert ".rnk-analysis-summary" in block
    assert ".rnk-analysis-chevron" in block
    assert "prefers-reduced-motion" in block


# ── Lite shell swapper ───────────────────────────────────────────────────

def test_lite_nav_js_is_conservative():
    src = open(os.path.join(_ROOT, "static", "lite_nav.js"),
               encoding="utf-8").read()
    assert 'meta[name="br-shell"]' in src       # both-ends lite check
    assert "application/ld+json" in src          # only scripts allowed in a swap
    assert "page-root" in src                    # the shell content wrapper
    assert "pushState" in src
    assert "popstate" in src
    assert "location.href = url" in src          # full-load fallback


def _stub_ros_route(monkeypatch):
    """The offline env has no ROS data files (the route 503s without
    them), so stub the map + value table the route reads at call time."""
    import app as app_mod

    payload, table = _fake_ros_payload()
    monkeypatch.setattr(
        "utils.ros_rankings.load_ros_map", lambda season=None: payload)
    monkeypatch.setattr(
        app_mod, "get_model_value_table_cached", lambda: table)
    monkeypatch.setattr(app_mod, "get_nfl_state", lambda: {"season": 2026})


def test_guest_ros_page_carries_lite_nav_shell(offline_client, monkeypatch):
    import app as app_mod
    if not getattr(app_mod, "_FEATURES_JS_FILE", None):
        pytest.skip("app-features.js bundle not built in this environment")
    _stub_ros_route(monkeypatch)

    r = offline_client.get("/rankings/rest-of-season-rb")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert '<meta name="br-shell" content="lite">' in html
    assert "lite_nav" in html
    assert 'id="page-root"' in html
    # The explainer dropdown ships in the served page too.
    assert '<details class="rnk-analysis rnk-analysis-details">' in html


def test_signed_in_ros_page_has_no_lite_nav(offline_client, monkeypatch):
    import app as app_mod
    if not getattr(app_mod, "_FEATURES_JS_FILE", None):
        pytest.skip("app-features.js bundle not built in this environment")
    _stub_ros_route(monkeypatch)

    with offline_client.session_transaction() as sess:
        sess["viewer_username"] = "tester"
        sess["viewer_user_id"] = "u1"

    r = offline_client.get("/rankings/rest-of-season-rb")
    assert r.status_code == 200
    html = r.get_data(as_text=True)
    assert 'name="br-shell"' not in html
    assert "lite_nav" not in html
