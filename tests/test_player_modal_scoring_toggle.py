"""Player modal Dynasty/Redraft + PPR/Half/STD value toggles."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_player_details_exposes_redraft_values_and_defaults():
    src = (ROOT / "app.py").read_text(encoding="utf-8")
    start = src.find("def api_player_details(player_id: str):")
    end = src.find("def api_player_game_logs", start)
    assert start > 0 and end > start
    body = src[start:end]

    assert '"redraft_value_1qb"' in body
    assert '"redraft_value_sf"' in body
    assert '"redraft_value_ovr_rank"' in body
    assert '"redraft_sf_value_ovr_rank"' in body
    assert '"default_scoring_type"' in body
    assert '"default_scoring_format"' in body
    assert '"scoring_format_ranks"' in body
    assert "SCORING_MULTS" in body
    assert "_league_is_redraft" in body
    # ESPN short-circuits to redraft without needing settings.type.
    assert 'platform or "").strip().lower() == "espn"' in body
    assert '_default_scoring = "redraft"' in body
    # Format default from league reception scoring.
    assert '_default_format = "half"' in body
    assert '_default_format = "std"' in body


def test_player_modal_has_dynasty_redraft_and_ppr_toggles():
    src = (ROOT / "static" / "player_modal.js").read_text(encoding="utf-8")

    assert 'id="pmScoringTypeToggle"' in src
    assert 'id="pmScoringFormatToggle"' in src
    assert 'data-scoring="dynasty"' in src
    assert 'data-scoring="redraft"' in src
    assert 'data-format="ppr"' in src
    assert 'data-format="half"' in src
    assert 'data-format="std"' in src
    assert "PM_SCORING_MULTS" in src
    assert "redraft_value_1qb" in src
    assert "default_scoring_type" in src
    assert "default_scoring_format" in src
    assert "scoring_format_ranks" in src
    assert "pmHero1qbVal" in src
    assert "pmHeroSfVal" in src
    assert "Dynasty Value History" in src
    assert "_wireScoringToggle" in src


def test_player_modal_scoring_toggle_css():
    css = (ROOT / "static" / "dashboard.css").read_text(encoding="utf-8")
    assert ".pm-scoring-toggle" in css
    assert ".pm-scoring-toggles" in css
