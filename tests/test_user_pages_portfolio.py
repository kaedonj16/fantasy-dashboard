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
