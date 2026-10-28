"""Dashboard duplicate-work fixes: one slide render, shared sim helper."""
import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_dashboard_renders_only_the_first_matchup_slide():
    src = (ROOT / "dashboard_services" / "pages" / "dashboard_page.py").read_text(encoding="utf-8")
    block = src[src.index("if _show_matchup_preview:"):src.index("matchup_html = ")]
    # Exactly one slide render, applied to the first (viewer-first) matchup;
    # the other league matchups' renders were discarded by the redesign.
    assert block.count("render_matchup_slide(") == 1
    assert "_dash_matchups[0]" in block


def test_playoff_odds_api_uses_shared_sim_helper():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    body = src[src.index("def api_playoff_odds():"):src.index('@app.route("/api/playoff-scenarios")')]
    assert "odds = _playoff_sim_cached(ctx, platform, block=True)" in body
    assert "simulate_playoff_odds(ctx" not in body
    # The snapshot writes stay gated on a read-only freshness probe.
    assert "_fresh_sim = not (" in body
    assert "write=_fresh_sim" in body


def test_offseason_standings_uses_shared_sim_helper():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    body = src[src.index("def _build_offseason_standings_body"):src.index("def ", src.index("def _build_offseason_standings_body") + 10)]
    assert "odds_list = _playoff_sim_cached(ctx, platform)" in body
    assert "simulate_playoff_odds(ctx" not in body
