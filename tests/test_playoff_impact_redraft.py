"""Redraft Playoff Impact drops dynasty signals (top-3 pick, roster age)."""
from pathlib import Path

import pytest

# simulate_playoff_odds imports numpy at module load; the lint-only CI job
# installs just pytest, so skip cleanly there like other heavy-dep suites.
pytest.importorskip("numpy")

from data_building.simulate_playoff_odds import shape_playoff_impact_for_league
from dashboard_services.pages.trade_calculator_page import build_trade_calculator_body

ROOT = Path(__file__).resolve().parents[1]


def _raw_impact():
    return {
        "available": True,
        "before": {"playoff_pct": 46.3, "avg_final_wins": 7.1, "avg_ppg": 112.0, "top3_pick_pct": 18.2},
        "after": {"playoff_pct": 52.0, "avg_final_wins": 7.6, "avg_ppg": 115.4, "top3_pick_pct": 9.0},
        "delta": {"playoff_pct": 5.7, "avg_final_wins": 0.5, "avg_ppg": 3.4, "top3_pick_pct": -9.2},
        "outlook": {"age_delta": -1.4, "value_delta": 80},
    }


def test_shape_playoff_impact_strips_dynasty_fields_for_redraft():
    out = shape_playoff_impact_for_league(_raw_impact(), is_redraft=True)
    assert out["scoring_type"] == "redraft"
    assert out["outlook"] is None
    assert "top3_pick_pct" not in out["before"]
    assert "top3_pick_pct" not in out["after"]
    assert "top3_pick_pct" not in out["delta"]
    assert out["before"]["playoff_pct"] == 46.3
    assert out["delta"]["playoff_pct"] == 5.7


def test_shape_playoff_impact_keeps_dynasty_fields():
    src = _raw_impact()
    out = shape_playoff_impact_for_league(src, is_redraft=False)
    assert out["scoring_type"] == "dynasty"
    assert out["before"]["top3_pick_pct"] == 18.2
    assert out["outlook"]["age_delta"] == -1.4


def test_redraft_trade_calc_tooltip_omits_pick_and_age():
    html = build_trade_calculator_body("L1", 2026, scoring_type="redraft")
    assert "Playoff Odds" in html
    assert "Proj. PPG" in html
    assert "Top-3 Pick" not in html
    assert "Roster Age" not in html
    assert "Prime Yrs Left" not in html


def test_dynasty_trade_calc_tooltip_keeps_pick_and_age():
    html = build_trade_calculator_body("L1", 2026, scoring_type="dynasty")
    assert "Top-3 Pick" in html
    assert "Roster Age" in html
    assert "Prime Yrs Left" in html


def test_playoff_impact_js_skips_future_outlook_in_redraft():
    js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    start = js.find("async function fetchPlayoffImpact")
    end = js.find("async function fetchTradeIntel")
    body = js[start:end]
    assert 'const isRedraft = (data.scoring_type || getScoringType()) === "redraft"' in body
    assert "Playoff Upgrade" in body
    assert "Playoff Downgrade" in body
    assert "!isRedraft && data.before.top3_pick_pct" in body
    assert 'pi-section-label">Future Outlook' in body


def test_api_skips_future_outlook_for_redraft():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def api_trade_eval_playoff_impact")
    end = src.find("def _trade_future_outlook")
    body = src[start:end]
    assert "shape_playoff_impact_for_league" in body
    assert "_league_is_redraft(ctx)" in body
    assert "if not is_redraft:" in body
    assert "_trade_future_outlook(" in body
