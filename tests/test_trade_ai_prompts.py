"""Trade-analyst prompts switch on dynasty vs redraft.

Does not need Flask; skipped if openai (imported by the prompt module) is absent.
"""
import pytest

pytest.importorskip("openai")

from dashboard_services.ai.prompts import (  # noqa: E402
    POS_RANK_LABEL_NOTE,
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


# ── Rank/tier label prompt rewrite: whitelist, worked examples, self-check ──

_SELF_CHECK = (
    "Before returning, scan your text: any position+number that is not "
    "\"the <pos_rank_label> overall\" or a role phrase copied from "
    "pos_tier_label must be rewritten."
)

_LABELED_PAYLOAD = {
    "scoring_type": "dynasty",
    "trade": {
        "viewer_gives": {"assets": [
            {"name": "Drake London", "pos_rank_label": "WR9",
             "pos_tier_label": "WR1"},
            {"name": "Jaylen Warren", "pos_rank_label": "RB24"},
        ]},
        "viewer_gets": {"assets": [
            {"name": "Omarion Hampton", "pos_rank_label": "RB18",
             "pos_tier_label": "RB2"},
            {"name": "Bijan Robinson", "pos_rank_label": "RB2",
             "pos_tier_label": "RB1"},
        ]},
    },
}


def test_label_note_is_a_two_construction_whitelist():
    note = POS_RANK_LABEL_NOTE
    assert "EXACTLY TWO CONSTRUCTIONS" in note
    assert "\"the <pos_rank_label> overall\"" in note
    assert "copied exactly from pos_tier_label" in note
    assert "Every other placement is forbidden" in note


def test_label_note_examples_do_not_collide():
    note = POS_RANK_LABEL_NOTE
    # The conversion is shown as an explicit pair for ONE player.
    assert "pos_rank_label WR23 is pos_tier_label WR2" in note
    # The banned exact-rank shapes are named, spaced and hyphenated.
    assert "\"WR23-type\"" in note
    assert "\"WR23 type\"" in note
    assert "\"WR23 tier asset\"" in note
    # The old colliding definition (WR2 = exact rank, "never as a tier")
    # is gone.
    assert "NOT a WR2 starter slot" not in note
    assert "2nd-most-valuable" not in note


def test_user_prompt_carries_worked_examples_for_this_trade():
    user = build_trade_ai_user_prompt(_LABELED_PAYLOAD, "dynasty")
    assert (
        "HOW TO REFERENCE THESE PLAYERS: "
        "Drake London — rank claim: \"the WR9 overall\"; role: \"WR1 asset\""
    ) in user
    assert (
        "Omarion Hampton — rank claim: \"the RB18 overall\"; "
        "role: \"RB2 asset\""
    ) in user
    assert "Bijan Robinson — rank claim: \"the RB2 overall\"" in user
    # Warren carries no tier label, so he gets no worked example.
    assert "Jaylen Warren — rank claim" not in user
    # Examples sit immediately before the trade context JSON.
    assert user.index("HOW TO REFERENCE THESE PLAYERS") < user.index(
        "Trade context:"
    )


def test_user_prompt_ends_with_the_self_check():
    user = build_trade_ai_user_prompt(_LABELED_PAYLOAD, "dynasty")
    assert user.endswith(_SELF_CHECK)


def test_examples_block_empty_without_labelled_assets():
    from dashboard_services.ai.prompts import _label_examples_block

    assert _label_examples_block({}) == ""
    assert _label_examples_block({"trade": {}}) == ""
    assert _label_examples_block({"trade": {"viewer_gives": {"assets": [
        {"name": "Jaylen Warren", "pos_rank_label": "RB24"},
    ]}}}) == ""
    user = build_trade_ai_user_prompt({"scoring_type": "dynasty"}, "dynasty")
    assert "HOW TO REFERENCE THESE PLAYERS" not in user
    assert user.endswith(_SELF_CHECK)
