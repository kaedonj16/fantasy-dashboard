from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_signed_in_home_separates_saved_leagues_from_connect_flow():
    source = (ROOT / "app.py").read_text()
    assert "Your leagues" in source
    assert 'id="connectLeagueFlow"{% if session.get(\'account_id\') %} hidden{% endif %}' in source
    assert 'aria-controls="connectLeagueFlow"' in source


def test_saved_leagues_are_links_with_an_explicit_open_action():
    source = (ROOT / "static" / "app.js").read_text()
    assert 'class="signed-home-league" href="${url}"' in source
    assert 'signed-home-league-open' in source
    assert 'flow.hidden = !willOpen' in source
