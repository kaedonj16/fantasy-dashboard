"""Trade calculator pre-selects dynasty vs redraft from the league."""
from pathlib import Path

from dashboard_services.pages.trade_calculator_page import build_trade_calculator_body

ROOT = Path(__file__).resolve().parents[1]


def _selected_scoring_type(html: str) -> str:
    import re

    match = re.search(
        r'<select[^>]*id="scoringTypeSelect"[^>]*>(.*?)</select>',
        html,
        re.S,
    )
    assert match, "scoringTypeSelect missing"
    block = match.group(1)
    selected = re.search(r'<option value="(dynasty|redraft)"[^>]*selected', block)
    assert selected, f"no selected scoring type in {block!r}"
    return selected.group(1)


def test_trade_calculator_defaults_to_dynasty():
    html = build_trade_calculator_body(None, 2026)
    assert _selected_scoring_type(html) == "dynasty"


def test_trade_calculator_preselects_redraft_when_league_is_redraft():
    html = build_trade_calculator_body("L1", 2026, scoring_type="redraft")
    assert _selected_scoring_type(html) == "redraft"
    assert 'value="dynasty" selected' not in html


def test_trade_calculator_exposes_platform_for_roster_filter():
    html = build_trade_calculator_body("L1", 2026, platform="espn", scoring_type="redraft")
    assert 'id="platformInput"' in html
    assert 'value="espn"' in html


def test_trade_calculator_exposes_league_scoring_type():
    html = build_trade_calculator_body("L1", 2026, platform="espn", scoring_type="redraft")
    assert 'id="leagueScoringTypeInput"' in html
    assert 'value="redraft"' in html
    assert "var _scoringType = 'redraft'" in html
    assert "Top-3 Pick" not in html


def test_strategy_impact_and_cards_are_two_columns():
    html = build_trade_calculator_body("L1", 2026)
    assert "#otcStrategyImpact" in html
    assert "#otcStrategyCards" in html
    # Both lists sit side-by-side on desktop and collapse at 600px.
    assert "grid-template-columns:1fr 1fr" in html
    assert html.count("grid-template-columns:1fr 1fr") >= 2
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert "#otcStrategyImpact {\n    display: grid;\n    grid-template-columns: 1fr 1fr;" in css
    assert "#otcStrategyCards {\n    display: grid;\n    grid-template-columns: 1fr 1fr;" in css
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert "const _STRATEGY_PAGE_SIZE = 6;" in js


def test_trade_page_wires_league_redraft_into_calculator():
    src = (ROOT / "routes" / "trade_bp.py").read_text(encoding="utf-8")
    assert "_league_is_redraft" in src
    assert 'scoring_type = "redraft" if _league_is_redraft(ctx) else "dynasty"' in src
    assert "scoring_type=scoring_type" in src
    assert "get_viewer_session_for_league(" in src
    assert "platform, league_id, season" in src


def test_trade_calculator_defaults_to_1qb_toggle():
    html = build_trade_calculator_body("L1", 2026)
    # No is_superflex passed -> SF toggle unchecked, page seeds _leagueType='1qb'.
    assert 'id="leagueTypeToggle">' in html and 'id="leagueTypeToggle" checked' not in html
    assert "var _leagueType = '1qb'" in html


def test_trade_calculator_preselects_superflex_when_league_is_sf():
    html = build_trade_calculator_body("L1", 2026, is_superflex=True)
    # SF league -> toggle pre-checked and _leagueType='sf', so the Strategy tab's
    # getLeagueType() reads "sf" and the archetype engine runs in superflex mode
    # (2 QB slots, SF QB values/scarcity). This is what surfaces QB suggestions
    # for a QB-starved superflex roster.
    assert 'id="leagueTypeToggle" checked' in html
    assert "var _leagueType = 'sf'" in html


def test_refresh_page_trade_render_forwards_superflex():
    """/api/refresh-page rebuilds the trade page body. It must pass is_superflex
    (and viewer/premium) like the canonical render in routes/trade_bp.py -
    omitting it defaulted every league to 1QB after a refresh, so a superflex
    league's Strategy tab ran in 1QB mode and stopped suggesting QB upgrades."""
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    marker = 'elif page == "trade":'
    start = src.index(marker)
    # The trade branch ends where the next branch (else:) begins.
    branch = src[start:src.index("else:", start)]
    assert "build_trade_calculator_body(" in branch
    assert "is_superflex=" in branch, "refresh render must seed the superflex toggle"
    assert "_is_superflex_lineup(" in branch
    assert "has_premium=" in branch
    assert "viewer_roster_id=" in branch


def test_otc_info_tooltip_shows_trade_count():
    html = build_trade_calculator_body(None, 2026)
    assert 'id="otcInfoTooltip"' in html
    assert 'id="tradeCount"' in html
    assert "We translate over" in html
    assert "trade relationships" in html

    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    assert 'id=\\"tradeCount\\"' in js or 'id="tradeCount"' in js
    assert "We translate over" in js
    # syncScoringTypeUi used to rewrite dynasty copy as "those trade
    # relationships" and drop the live count. Guard against that regression.
    assert "those trade relationships" not in js
    assert "cachedTradeCountLabel" in js
    assert "setTradeCountLabel" in js
