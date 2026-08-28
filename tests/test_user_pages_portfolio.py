"""My Leagues pending-card viewer matching guards."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_portfolio_uses_account_viewer_resolution():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    fn = source.split("def _league_summary")[1].split("\n    leagues_data")[0]
    assert "resolve_account_viewer_for_league" in fn
    assert "match_viewer_roster" in fn
    assert 'owner_id=viewer_user_id if lg_platform == "sleeper"' in fn
    assert 'str(r.get("owner_id")) == str(viewer_user_id)' not in fn


def test_portfolio_positional_strength_uses_in_league_percentiles():
    source = (ROOT / "routes" / "user_pages_bp.py").read_text()
    summary = source.split("def _league_summary")[1].split("\n    leagues_data")[0]
    assert "from utils.roster_strength import strength_percentile" in summary
    assert "pos_user_pctile[pos] = strength_percentile(" in summary
    assert '"pos_user_pctile": pos_user_pctile' in summary

    blend = source.split("Cross-league positional strength")[1].split("valid_leagues.sort")[0]
    assert "average_league_percentiles" in blend
    assert "pos_user_pctile" in blend
    # Ratio-vs-median blend is what made stacked leagues read negative.
    assert "u / a" not in blend
