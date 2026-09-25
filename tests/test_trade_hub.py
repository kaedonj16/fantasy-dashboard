"""Trade Hub: shared why-lines, shop repricer, and hub markup.

Covers dashboard_services/trade_hub.py (why_line_* helpers + shop_package),
the extracted Trade Intelligence page builder, and the tabbed hub markup in
the trade calculator page.
"""

import pytest

from dashboard_services.trade_hub import (
    _cheapest_fair_return,
    shop_package,
    why_line_for_market,
    why_line_for_shop,
    why_line_for_suggestion,
    why_line_for_target,
)


# ---------------------------------------------------------------------------
# why-line helpers
# ---------------------------------------------------------------------------

def test_why_line_suggestion_uses_fit_note_and_acceptance():
    line = why_line_for_suggestion({
        "fit_note": "Fills their RB hole",
        "acceptance_pct": 72,
    })
    assert "Fills their RB hole." in line
    assert "72%" in line
    assert "\u2014" not in line  # no em dashes in user-facing copy


def test_why_line_suggestion_falls_back_to_why():
    line = why_line_for_suggestion({"why": "Cheapest upgrade at WR"})
    assert "Cheapest upgrade at WR." in line


def test_why_line_target_names_owner_and_need():
    line = why_line_for_target({
        "name": "Puka Nacua",
        "position": "WR",
        "value": 800,
        "surplus_value": 900,
        "owner_team": "Pilots",
        "owner_needs": ["RB"],
    })
    assert "Pilots" in line
    assert "RB" in line
    assert "\u2014" not in line


def test_why_line_market_mentions_volume_and_trend():
    line = why_line_for_market({
        "trade_count_7d": 31,
        "market_trend": 5.2,
        "value_delta": 40,
        "model_value": 500,
        "buy_sell_ratio": 2.1,
    })
    assert "31" in line
    assert "rising" in line
    assert "Sell-high" in line
    assert "\u2014" not in line


def test_why_line_shop_names_needs_and_fairness():
    line = why_line_for_shop({
        "team_needs": ["RB", "WR"],
        "suggested_get": [{"name": "Mid RB"}, {"name": "Mid WR"}],
        "fairness": 1.10,
    })
    assert "RB" in line and "WR" in line
    assert "Mid RB" in line
    assert "win the value math" in line
    assert "\u2014" not in line


def test_why_lines_never_contain_em_dashes():
    lines = [
        why_line_for_suggestion({"fit_note": "x", "acceptance_pct": 50}),
        why_line_for_target({"owner_team": "T", "owner_needs": ["QB"], "value": 1, "surplus_value": 2}),
        why_line_for_market({"trade_count_7d": 5, "market_trend": -3, "value_delta": -10,
                             "model_value": 100, "buy_sell_ratio": 0.5}),
        why_line_for_shop({"team_needs": ["TE"], "suggested_get": [{"name": "X"}],
                           "fairness": 0.9}),
    ]
    for line in lines:
        assert "\u2014" not in line


# ---------------------------------------------------------------------------
# _cheapest_fair_return
# ---------------------------------------------------------------------------

def _c(cid, value, need_fit=False):
    return {"id": cid, "name": cid, "value": value, "need_fit": need_fit}


def test_cheapest_fair_return_prefers_closest_to_parity():
    chosen, total = _cheapest_fair_return(
        [_c("a", 500), _c("b", 480), _c("c", 350), _c("d", 300)], 900)
    assert total >= 900 * 0.85
    assert abs(total / 900 - 1.0) < 0.15


def test_cheapest_fair_return_prefers_fewer_pieces_on_tie():
    chosen, total = _cheapest_fair_return([_c("a", 900), _c("b", 450), _c("c", 450)], 900)
    assert [x["id"] for x in chosen] == ["a"]


def test_cheapest_fair_return_empty_when_nothing_reaches_floor():
    chosen, total = _cheapest_fair_return([_c("a", 100)], 900)
    assert chosen == [] and total == 0.0


def test_cheapest_fair_return_zero_or_empty_target():
    assert _cheapest_fair_return([_c("a", 500)], 0) == ([], 0.0)
    assert _cheapest_fair_return([], 900) == ([], 0.0)


# ---------------------------------------------------------------------------
# shop_package
# ---------------------------------------------------------------------------

def _hub_ctx():
    def p(pid, name, pos, val):
        return {"id": pid, "name": name, "position": pos, "value": val, "team": "FAKE"}

    mvt = [
        p("1", "Stud RB", "RB", 900), p("2", "Mid RB", "RB", 500),
        p("3", "Weak RB", "RB", 200), p("4", "Stud WR", "WR", 850),
        p("5", "Mid WR", "WR", 480), p("6", "Weak WR", "WR", 180),
        p("7", "Stud QB", "QB", 700), p("8", "Mid QB", "QB", 350),
        p("9", "Stud TE", "TE", 550), p("10", "Mid TE", "TE", 300),
        p("11", "RB2", "RB", 620), p("12", "WR2", "WR", 600),
    ]
    return {
        "rosters": [
            {"roster_id": "1", "players": ["1", "4", "7", "9"]},
            {"roster_id": "2", "players": ["3", "6", "8", "10"]},
            {"roster_id": "3", "players": ["2", "5", "8", "10"]},
            {"roster_id": "4", "players": ["11", "12", "7", "9"]},
        ],
        "roster_map": {"1": "You", "2": "Team B", "3": "Team C", "4": "Team D"},
        "model_value_table": mvt,
        "players_index": {},
        "roster_positions": ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN", "BN"],
        "picks_by_roster": {},
    }


def test_shop_package_skips_viewer_and_prices_every_team():
    res = shop_package(_hub_ctx(), viewer_roster_id="1", send_ids=["1"])
    assert res["send_value"] == pytest.approx(900.0)
    teams = {t["team"] for t in res["teams"]}
    assert "You" not in teams
    assert {"Team B", "Team C", "Team D"} <= teams


def test_shop_package_rows_carry_needs_fairness_and_why_line():
    res = shop_package(_hub_ctx(), viewer_roster_id="1", send_ids=["1"])
    assert res["teams"], "expected at least one shoppable team"
    for t in res["teams"]:
        assert t["team_needs"], "row should name the partner's needs"
        assert t["suggested_get"], "row should include a value-matched return"
        assert 0.85 <= t["fairness"], "return must clear the 85% fairness floor"
        assert t["why_line"], "row should carry the shared why-line"
        assert "\u2014" not in t["why_line"]


def test_shop_package_empty_send_ids():
    res = shop_package(_hub_ctx(), viewer_roster_id="1", send_ids=[])
    assert res["teams"] == []


# ---------------------------------------------------------------------------
# Trade Intelligence page builder (extracted)
# ---------------------------------------------------------------------------

def test_trade_intel_builder_standalone_and_embedded():
    from dashboard_services.pages.trade_intel_page import build_trade_intel_body

    full = build_trade_intel_body(platform="sleeper", season=2026, league_id="x",
                                  has_premium=True, embedded=False)
    assert "tiGrid" in full
    assert 'class="card central"' in full  # standalone wrapper present
    assert "ti-embed" not in full
    assert "Why this" in full  # shared why-line CSS/component

    embedded = build_trade_intel_body(platform="sleeper", season=2026, league_id="x",
                                      has_premium=True, embedded=True)
    assert "tiGrid" in embedded
    assert 'class="card central"' not in embedded  # no standalone wrapper
    assert "ti-embed" in embedded


# ---------------------------------------------------------------------------
# Trade calculator page: hub tabs
# ---------------------------------------------------------------------------

def test_trade_calculator_hub_tabs_for_pro():
    from dashboard_services.pages.trade_calculator_page import build_trade_calculator_body

    html = build_trade_calculator_body("L1", 2026, has_premium=True)
    for tab_id in ("otcSubtabSuggestions", "otcSubtabTargets",
                   "otcSubtabMarket", "otcSubtabSaved"):
        assert tab_id in html
    assert "otcSavedPanel" in html
    assert "otcMarketIntelPanel" in html
    assert "Trade Hub" in html
    # Market intel embedded for PRO
    assert "tiGrid" in html


def test_trade_calculator_hub_gated_for_non_pro():
    from dashboard_services.pages.trade_calculator_page import build_trade_calculator_body

    html = build_trade_calculator_body(None, 2026, has_premium=False)
    assert "otcSuggPaywall" in html
    assert "tiGrid" not in html  # no market intel markup for non-PRO
