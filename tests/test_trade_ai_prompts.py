"""Trade-analyst prompts switch on dynasty vs redraft.

Does not need Flask; skipped if openai (imported by the prompt module) is absent.
"""
import pytest

pytest.importorskip("openai")

from dashboard_services.ai.prompts import (  # noqa: E402
    REDRAFT_HONESTY_RULES,
    build_trade_ai_system_prompt,
    build_trade_ai_user_prompt,
    normalize_trade_scoring_type,
)


def test_normalize_trade_scoring_type():
    assert normalize_trade_scoring_type("REDRAFT") == "redraft"
    assert normalize_trade_scoring_type("dynasty") == "dynasty"
    assert normalize_trade_scoring_type("") == "dynasty"
    assert normalize_trade_scoring_type(None) == "dynasty"
    assert normalize_trade_scoring_type("keeper") == "dynasty"


def test_redraft_trade_prompt_forbids_draft_pick_counters():
    redraft = build_trade_ai_system_prompt("redraft")
    redraft_l = redraft.lower()
    assert "redraft" in redraft_l
    assert "never recommend" in redraft_l
    assert "never suggest a draft pick" in redraft_l
    assert "1.01" not in redraft
    assert "cannot be traded" in redraft_l

    user = build_trade_ai_user_prompt({"scoring_type": "redraft"}, "redraft").lower()
    assert "redraft" in user
    assert "draft picks cannot be traded" in user
    assert "analyze this dynasty trade" not in user
    assert "playoff_status" in redraft_l
    assert "playoff_pct" in redraft_l
    assert "retool" in redraft_l
    assert "record context is missing" in redraft_l
    assert REDRAFT_HONESTY_RULES.splitlines()[0] in redraft
    assert "playoff_status" in user
    assert "weakest_positions" in user


def test_dynasty_trade_prompt_keeps_pick_valuation():
    dynasty = build_trade_ai_system_prompt("dynasty")
    assert "1.01" in dynasty
    assert "dynasty" in dynasty.lower()
    user = build_trade_ai_user_prompt({"scoring_type": "dynasty"}, "dynasty").lower()
    assert "analyze this dynasty trade" in user
    assert "draft picks cannot be traded" not in user


def test_redraft_trade_suggestions_prompt_uses_honesty_rules():
    src = open("dashboard_services/ai/prompts.py", encoding="utf-8").read()
    start = src.find("def generate_trade_suggestions_result")
    end = src.find("def generate_team_ai_result")
    body = src[start:end]
    assert "REDRAFT_HONESTY_RULES" in body
    assert "playoff_status" in body
    assert "playoff_pct" in body
