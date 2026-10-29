"""Direction C (2026-10-29): the Portfolio insights sections live as
collapsed accordion rows inside ONE card, the Rooting Guide row is
game-day only, hydration is lazy on first open, and the summary strip
keeps its full data-hook contract under its polish."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

pytest.importorskip("flask")
pytest.importorskip("pandas")


def _league(lid="1", name="Alpha", **kw):
    lg = {
        "league_id": lid,
        "name": name,
        "platform": "sleeper",
        "season": 2026,
        "wins": 5,
        "losses": 2,
        "ties": 0,
        "record": "5-2",
        "rank": 2,
        "total_teams": 10,
        "pf": 812.4,
        "total_value": 4200.0,
        "all_players": {},
        "streak": ["W", "W", "L"],
        "urgency": 3,
        "pos_user_vals": {},
        "pos_league_avgs": {},
        "pos_user_pctile": {},
        "pos_user_rank": {"QB": 4, "RB": 9, "WR": 2, "TE": 6},
        "offseason": False,
        "team_name": "Team " + name,
        "is_favorite": False,
    }
    lg.update(kw)
    return lg


_HOLDINGS = [
    {"pid": "p1", "name": "Top Wideout", "position": "WR", "value": 2980,
     "pos_rank": "WR4", "shares": 3, "in_leagues": ["Alpha", "Beta", "Gamma"],
     "bye_week": 10, "status": "Questionable", "rank_change_7d": 12},
]


def _render(leagues, game_day, **kw):
    import app as app_mod
    orig = app_mod._games_scheduled_today
    app_mod._games_scheduled_today = lambda s, w: game_day
    try:
        tw = sum(l["wins"] for l in leagues)
        tl = sum(l["losses"] for l in leagues)
        args = dict(
            num_leagues=len(leagues), holdings=[], nfl_exposure=[],
            cross_pos={}, total_wins=tw, total_losses=tl, total_ties=0,
            current_week=8,
        )
        args.update(kw)
        with app_mod.app.test_request_context("/"):
            return app_mod.build_portfolio_body(
                "kaedon", leagues, leagues, 2026, **args)
    finally:
        app_mod._games_scheduled_today = orig


def test_insights_one_card_four_collapsed_rows():
    html = _render([_league()], game_day=True, holdings=_HOLDINGS)
    # Exactly one insights card, carrying the title once.
    assert "class='card pf-ins-card'" in html
    assert html.count("class='card pf-ins-card'") == 1
    assert "<div class='pf-ins-head'><h2>Portfolio insights</h2></div>" in html
    # No standalone section cards remain.
    assert "class='card' id='pfSec" not in html
    # All four rows: accordion section, collapsed header button.
    for sec in ("pfSecHeat", "pfSecRooting", "pfSecBye", "pfSecHoldings"):
        assert f"class='pf-acc' id='{sec}'" in html
        assert f"data-pf-acc='{sec}' aria-expanded='false'" in html
    assert html.count("data-pf-acc='") == 4
    # Row order: Strength, Rooting, Bye, Holdings.
    order = [html.index(f"id='{s}'") for s in
             ("pfSecHeat", "pfSecRooting", "pfSecBye", "pfSecHoldings")]
    assert order == sorted(order)
    # Bodies start hidden; section contents and ids are untouched.
    assert "<div class='pf-acc-b' hidden>" in html  # heatmap + holdings
    assert "id='pfRootingBody' hidden" in html
    assert "id='pfByeBody' hidden" in html
    assert "pf-hm-chip" in html
    assert "id='pfTable'" in html and "id='pfBody'" in html
    assert "id='pfSearch'" in html
    assert "data-bye-sub" in html and "data-rooting-upd" in html
    # The accordion wiring ships with the card: headers toggle, the
    # shared opener expands, and opening dispatches the lazy loader.
    assert "window.__pfOpenSection=function(id)" in html
    assert "window.__pfLoadSection)window.__pfLoadSection(id)" in html
    # Phone tabs expand their row, not just scroll to it.
    assert "window.__pfOpenSection)window.__pfOpenSection(tid)" in html


def test_rooting_row_and_tab_are_game_day_only():
    on = _render([_league()], game_day=True, holdings=_HOLDINGS)
    assert "id='pfSecRooting'" in on
    assert "id='pfRootingBody'" in on
    assert "data-t='pfSecRooting'" in on  # phone tab exists too

    off = _render([_league()], game_day=False, holdings=_HOLDINGS)
    assert "id='pfSecRooting'" not in off
    assert "id='pfRootingBody'" not in off
    assert "data-t='pfSecRooting'" not in off
    # The other rows are unconditional.
    for sec in ("pfSecHeat", "pfSecBye", "pfSecHoldings"):
        assert f"id='{sec}'" in off


def test_summary_strip_hooks_contract_unchanged():
    html = _render([_league()], game_day=False)
    assert "class='card pf-summary-card'" in html
    assert "data-portfolio-league-count" in html
    assert re.search(
        r"data-portfolio-agg-record data-wins='5' data-losses='2' "
        r"data-ties='0'", html)
    assert "data-portfolio-lw-record" in html  # Last Week tile, week 8
    assert "id='pfProjTile'" in html and "aria-expanded='false'" in html
    assert "id='pfReceipts' hidden" in html
    assert "data-receipts-rows" in html
    assert "Toss-up = within 3 pts" in html
    assert "data-portfolio-proj-record" in html
    assert "data-portfolio-next-deadline" in html
    assert "Signed in as" in html and "href='/reset-user'" in html
    assert "openNotifPrefs" in html
    # The polish: proj tile carries a chevron affordance in its label.
    assert ".pf-stat-btn .pf-stat-label::after" in html


def test_app_js_fetches_only_inside_lazy_loaders():
    js = (ROOT / "static" / "app.js").read_text()
    # Each insights endpoint is fetched from exactly one place, inside
    # its loader function; nothing calls the loaders at page load.
    ins = js.index("fetcher('/api/portfolio-insights'")
    assert js.index("function loadInsights(attempt, startedAt)") < ins
    assert ins < js.index("function loadBye(attempt)")
    bye = js.index("fetcher('/api/portfolio-bye'")
    assert js.index("function loadBye(attempt)") < bye
    assert bye < js.index("window.__pfLoadSection")
    assert js.count("fetcher('/api/portfolio-insights'") == 1
    assert js.count("fetcher('/api/portfolio-bye'") == 1
    assert "document.addEventListener('DOMContentLoaded', loadBye)" not in js
    assert "kickInsights" not in js
