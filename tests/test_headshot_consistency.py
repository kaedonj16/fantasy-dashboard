"""Regression coverage for player headshots shared by search and the modal."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_league_player_payload_includes_modal_headshot():
    source = (ROOT / "app.py").read_text(encoding="utf-8")

    assert 'player["espnHeadshot"] = player_data.get("espnHeadshot") or ""' in source


def test_nav_search_uses_canonical_espn_headshot():
    source = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
    nav_search = source.split("// ── Nav-wide player search", 1)[1]

    assert "headshot: String(p.espnHeadshot || '')" in nav_search
    assert "_hiResHeadshot(p.headshot, 80)" in nav_search
    assert "sleepercdn.com/content/nfl/players/thumb" not in nav_search
