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


def test_trade_page_wires_league_redraft_into_calculator():
    src = (ROOT / "routes" / "trade_bp.py").read_text(encoding="utf-8")
    assert "_league_is_redraft" in src
    assert 'scoring_type = "redraft" if _league_is_redraft(ctx) else "dynasty"' in src
    assert "scoring_type=scoring_type" in src
    assert "get_viewer_session_for_league(" in src
    assert "platform, league_id, season" in src
