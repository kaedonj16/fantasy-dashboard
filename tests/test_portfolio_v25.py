"""Focused tests for the Portfolio Front Office v25 redesign page:
unpaginated compact league cards, the receipts summary, the
board card shell, the heatmap, and the reworked holdings table."""
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
    # Sort control: Urgency / Record / Standings (the Playoff odds mode
    # left with the card stat, which never populated in production).
    assert "id='pfSortSel'" in html
    assert "Urgency" in html and "Standings" in html
    assert "<option value='standings'>Standings</option>" in html
    assert "<option value='odds'>" not in html


def test_leagues_header_and_sort_share_one_line():
    html = _render([_league()])
    # Title and sort live in the same header row, not stacked rows.
    assert "card-header pf-leagues-head'><h2>My Leagues</h2>" in html
    assert "<div class='pf-sortbar'><label for='pfSortSel'>Sort</label>" in html
    assert ".pf-leagues-head{flex-wrap:nowrap;}" in html
    assert ".pf-sortbar{display:flex;align-items:center;gap:8px;flex:0 0 auto;}" in html
    assert "justify-content:flex-end;padding:10px 2px 0" not in html


def test_card_shows_fix_best_and_posture_without_odds():
    # The league dict can still carry sim odds (the team modal uses them);
    # the My Leagues card must not render a Playoff odds stat or the
    # data-odds sort attribute (2026-10-27: the stat never populated).
    lg = _league(playoff_odds_pct=74.0, playoff_odds_delta=9.0,
                 posture="Contend", posture_cls="contend")
    html = _render([lg])
    assert "Playoff odds" not in html
    assert "data-odds" not in html
    assert "74.0%" not in html
    # Standings sort source: the card carries its rank up front.
    assert "data-rank='2'" in html
    # Rooms in two columns from pos_user_rank {QB 4, RB 9, WR 2, TE 6}:
    # Fix is the weakest (RB 9th), Best the strongest (WR 2nd).
    assert "Fix: " in html and "RB · 9th of 10" in html
    assert "Best: " in html and "WR · 2nd of 10" in html
    assert "Weakest room:" not in html
    assert "pf-posture contend" in html
    assert ">Contend</span>" in html
    # Whole card is the link; the favorite button stays a separate control.
    assert "data-href='/sleeper/2026/1/dashboard'" in html
    # No footer on the card: no Front Office link, no Open button.
    assert "class='pf-lg-fo'" not in html
    assert "pf-lg-open" not in html


def test_card_without_rank_gets_sort_sentinel():
    lg = _league(rank="?")
    html = _render([lg])
    assert "data-rank='999'" in html


def test_standings_sort_comparator_and_hydration_hook():
    html = _render([_league()])
    # Ascending by data-rank, wins desc on ties, unranked (999) last.
    assert "sortMode==='standings'" in html
    assert "parseInt(a.getAttribute('data-rank'),10)" in html
    assert "sortMode==='odds'" not in html
    # The inline sort exposes its render so app.js can re-apply the order
    # as card summaries hydrate.
    assert "window.__pfApplySort=render;" in html
    js = (ROOT / "static" / "app.js").read_text()
    assert "card.dataset.rank" in js
    assert "card.dataset.odds" not in js
    assert "window.__pfApplySort" in js
    rs = js[js.index("function renderSummary("):]
    rs = rs[:rs.index("function updateAggregateRecord()")]
    assert "Playoff odds" not in rs


def test_phonetabs_sections_carry_scroll_margin():
    html = _render([_league()])
    # Measured sticky stack at 390px: nav 59px, phonetabs bar stuck bottom
    # 98px; margin is the stack + ~10px so a tap lands the card header
    # fully below both bars instead of underneath them.
    assert ("#pfSecHeat,#pfSecRooting,#pfSecBye,#pfSecHoldings"
            "{scroll-margin-top:108px !important;}") in html
    # One bounded re-align when the insights bodies re-render right after
    # a tap (content growth above the target), never a polling loop.
    assert "pfRootingBody" in html and "MutationObserver" in html
    assert "_pfTap" in html


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


def test_board_card_paints_a_loading_state_not_a_hidden_card():
    # Regression (2026-10-30): the card rendered `hidden` and its only
    # in-progress content was one static muted line, while the scan does
    # not even start until the league cards settle (or 8s) and then runs
    # the heaviest read on the page. The section read as missing for the
    # whole cold scan. It must paint visible, with a real loading state
    # whose sub-line the JS can update with scan progress.
    html = _render([_league()])
    assert "id='pfMovesCard' style='margin-bottom:14px;' hidden" not in html
    assert "Scanning your leagues" in html
    assert "id='pfLoadSub'" in html
    assert "pf-load-skel" in html
    assert "<div class='pf-moves-loading'>Checking lineups…</div>" not in html


def test_board_js_reports_scan_progress_and_reinits_per_card():
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    # Incomplete passes surface their scanned/total counts into the
    # loading sub-line instead of being discarded until the final render.
    assert "pfLoadProgress(scan)" in js
    assert "pfLoadProgress(null)" in js
    assert "function pfLoadingHtml()" in js
    # The one-shot init guard is per card element, so a soft-nav return
    # (fresh card, surviving closure) re-initialises instead of leaving
    # the new card dead for the visit.
    assert "startedCard === card" in js
    # A final failure offers an in-place Retry, not only "refresh".
    assert "__pfRefreshBoard" in js and ">Retry</button>" in js


def test_insights_heatmap_rooting_bye_and_tabs(monkeypatch):
    # The Rooting Guide row renders only on game days: the shared schedule
    # check reads the real calendar, so an unpinned render drops the row on
    # non-game days and this test went red every Fri/Sat. Pin the check to a
    # game day for the presence assertions, then pin it off to prove the
    # gate still hides the row.
    import app as _app_module
    monkeypatch.setattr(_app_module, "_games_scheduled_today",
                        lambda *a, **k: True)
    html = _render([_league()])
    monkeypatch.setattr(_app_module, "_games_scheduled_today",
                        lambda *a, **k: False)
    assert "id='pfSecRooting'" not in _render([_league()])
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
