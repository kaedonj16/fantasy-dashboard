"""Focused tests for the Portfolio Front Office v25 redesign page:
unpaginated compact league cards with odds, the receipts summary, the
board card shell, the heatmap, and the reworked holdings table."""
import pytest

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


def _render(leagues, **kw):
    from app import app, build_portfolio_body
    tw = sum(l["wins"] for l in leagues)
    tl = sum(l["losses"] for l in leagues)
    args = dict(
        num_leagues=len(leagues), holdings=[], nfl_exposure=[],
        cross_pos={}, total_wins=tw, total_losses=tl, total_ties=0,
        current_week=8,
    )
    args.update(kw)
    with app.test_request_context("/"):
        return build_portfolio_body("kaedon", leagues, leagues, 2026, **args)


def test_cards_render_all_leagues_without_pager():
    leagues = [_league(str(i), f"League {i}") for i in range(1, 8)]
    html = _render(leagues)
    assert "pfLgPager" not in html
    assert "pfLgPrev" not in html and "pfLgNext" not in html
    assert html.count("data-summary-card") >= 7
    # Sort control with the three modes from the mock.
    assert "id='pfSortSel'" in html
    assert "Urgency" in html and "Playoff odds" in html


def test_leagues_header_and_sort_share_one_line():
    html = _render([_league()])
    # Title and sort live in the same header row, not stacked rows.
    assert "card-header pf-leagues-head'><h2>My Leagues</h2>" in html
    assert "<div class='pf-sortbar'><label for='pfSortSel'>Sort</label>" in html
    assert ".pf-leagues-head{flex-wrap:nowrap;}" in html
    assert ".pf-sortbar{display:flex;align-items:center;gap:8px;flex:0 0 auto;}" in html
    assert "justify-content:flex-end;padding:10px 2px 0" not in html


def test_card_shows_odds_delta_weakest_and_posture():
    lg = _league(playoff_odds_pct=74.0, playoff_odds_delta=9.0,
                 posture="Contend", posture_cls="contend")
    html = _render([lg])
    assert "data-odds='74.0'" in html
    # Odds read at one decimal (floored): only an exact sim 100/0 may read
    # as a whole 100% / 0%; an undecided 74.0 reads "74.0%", never "74%".
    assert "74.0%" in html
    assert "▲9" in html
    assert "Playoff odds" in html
    assert "Weakest room:" in html
    assert "RB · 9th of 10" in html
    assert "pf-posture contend" in html
    assert ">Contend</span>" in html
    # Whole card is the link; the favorite button stays a separate control.
    assert "data-href='/sleeper/2026/1/dashboard'" in html
    assert "Front Office" in html


def test_card_hides_delta_without_history():
    lg = _league(playoff_odds_pct=61.0, playoff_odds_delta=None,
                 posture="Contend", posture_cls="contend")
    html = _render([lg])
    assert "61.0%" in html
    assert "data-odds-delta hidden" in html


def test_card_odds_never_round_up_to_100():
    # A sim-capped 99.9 is undecided, not clinched: it must read 99.9%.
    lg = _league(playoff_odds_pct=99.9, playoff_odds_delta=None,
                 posture="Contend", posture_cls="contend")
    html = _render([lg])
    assert "data-odds-val>99.9%</span>" in html
    assert "data-odds-val>100%</span>" not in html
    # And a capped 0.1 is alive, not eliminated: it must read 0.1%.
    lg = _league(playoff_odds_pct=0.1, playoff_odds_delta=None,
                 posture="Long Shot", posture_cls="long-shot")
    html = _render([lg])
    assert "data-odds-val>0.1%</span>" in html


def test_card_without_odds_shows_dash():
    html = _render([_league()])
    assert "data-odds='-1'" in html
    assert "data-odds-val>—</span>" in html


def test_summary_bar_five_stats_and_receipts_shell():
    html = _render([_league()], current_week=8)
    assert "pf-stat-bar pf-stat-bar--5" in html
    assert "data-portfolio-proj-record" in html
    assert "This week (proj)" in html
    assert "data-portfolio-next-deadline" in html
    assert "Next deadline" in html
    # Receipts panel: closed by default, header states the toss-up band.
    assert "id='pfReceipts' hidden" in html
    assert "Toss-up = within 3 pts" in html
    assert "id='pfProjTile'" in html
    assert "aria-expanded='false'" in html


def test_board_card_shell_and_bootstrap():
    html = _render([_league()])
    assert "id='pfMovesCard'" in html
    assert "id='pfMovesBody'" in html
    assert "id='pfScanLine'" in html
    assert "id='pfGenBtn'" in html
    assert "Generate full report" in html
    assert "id='pfGradeLine'" in html
    assert "id='pfHiddenFooter'" in html
    assert "__pfBoardInit" in html
    assert "__pfBoardWeek=8" in html


def test_insights_heatmap_rooting_bye_and_tabs():
    html = _render([_league()])
    assert "id='pfSecHeat'" in html
    assert "pf-hm-chip" in html
    assert "id='pfSecRooting'" in html
    assert "id='pfRootingBody'" in html
    assert "id='pfSecBye'" in html
    assert "id='pfByeBody'" in html
    assert "id='pfPhoneTabs'" in html
    assert "Rooting guide" in html and ">Byes</button>" in html
    # Movers is gone from the page.
    assert "Portfolio Movers" not in html
    # NFL Exposure is replaced by the Rooting Guide.
    assert "NFL Exposure" not in html


def test_holdings_columns_flag_and_no_pager():
    holdings = [
        {"pid": "p1", "name": "Top Wideout", "position": "WR", "value": 2980,
         "pos_rank": "WR4", "shares": 3, "in_leagues": ["Alpha", "Beta", "Gamma"],
         "bye_week": 10, "status": "Questionable", "rank_change_7d": 12},
        {"pid": "p2", "name": "Hurt Back", "position": "RB", "value": 720,
         "pos_rank": "RB27", "shares": 1, "in_leagues": ["Alpha"],
         "bye_week": 9, "status": "Out", "rank_change_7d": -4},
    ]
    html = _render([_league()], holdings=holdings)
    assert "id='pfSecHoldings'" in html
    for col in ("data-sort='bye'", "data-sort='status'", "data-sort='trend'"):
        assert col in html
    assert "Wk 10" in html
    assert "pf-hstatus warn" in html and "Questionable" in html
    assert "pf-hstatus bad" in html and ">Out</span>" in html
    assert "▲12" in html and "▼4" in html
    # Concentration flag for 3+ leagues.
    assert "pf-conc" in html and "3 leagues" in html
    # No holdings pager.
    assert "id='pfPrev'" not in html and "id='pfNext'" not in html
    assert "id='pfPageInfo'" not in html


def test_holdings_sortable_columns_and_row_keys():
    """Every holdings column header carries a data-sort key, Value is
    the marked default (descending), and rows carry the typed sort
    keys the click handler reads (never display text): shares/bye as
    ints with a large sentinel for a missing bye, rank as the numeric
    part of the position-rank label, status lowercased, 7d signed."""
    holdings = [
        {"pid": "p1", "name": "Top Wideout", "position": "WR", "value": 2980,
         "pos_rank": "WR4", "shares": 3, "in_leagues": ["Alpha", "Beta", "Gamma"],
         "bye_week": 10, "status": "Questionable", "rank_change_7d": 12},
        {"pid": "p2", "name": "Hurt Back", "position": "RB", "value": 720,
         "pos_rank": "RB27", "shares": 1, "in_leagues": ["Alpha"],
         "bye_week": None, "status": "", "rank_change_7d": -4},
    ]
    html = _render([_league()], holdings=holdings)
    for key in ("pos", "name", "rank", "shares", "bye", "status",
                "trend", "value"):
        assert f"data-sort='{key}'" in html
    assert "data-sort='value' class='pf-sorted' aria-sort='descending'" in html
    assert "data-shares='3'" in html and "data-shares='1'" in html
    assert "data-bye='10'" in html and "data-bye='999'" in html
    assert "data-rank='4'" in html and "data-rank='27'" in html
    assert "data-status='questionable'" in html and "data-status=''" in html
    assert "data-trend='12'" in html and "data-trend='-4'" in html
    # The sort handler itself: comparator + header wiring + cap re-deal.
    assert "function cmpRows(a,b)" in html
    assert "function applySort()" in html
    assert "pf-row-extra',i>=25" in html
    assert "aria-sort" in html
