"""AI prose must not echo JSON field names like playoff_pct."""
from dashboard_services.ai.prose import scrub_ai_prose_field_names, scrub_ai_result_strings


def test_scrub_playoff_pct_sits_at():
    raw = (
        "With no games played yet, playoff_pct sits at 78.5%, and the roster is "
        "built to win now around Ja'Marr Chase."
    )
    out = scrub_ai_prose_field_names(raw)
    assert "playoff_pct" not in out
    assert "playoff odds sit at 78.5%" in out


def test_scrub_ai_result_strings_all_fields():
    scrubbed = scrub_ai_result_strings({
        "outlook": "playoff_pct sits at 46.3% with a bubble playoff_status.",
        "verdict": "HOLD",
        "rank": 2,
    })
    assert "playoff_pct" not in scrubbed["outlook"]
    assert "playoff_status" not in scrubbed["outlook"]
    assert "playoff odds" in scrubbed["outlook"].lower()
    assert "playoff standing" in scrubbed["outlook"].lower()
    assert scrubbed["verdict"] == "HOLD"
    assert scrubbed["rank"] == 2


def test_scrub_before_render_matches_screenshot_leak():
    """Cached GM memos that already leaked playoff_pct get cleaned on render path."""
    result = scrub_ai_result_strings({
        "team_identity": "Team 5 is a preseason contending redraft roster",
        "outlook": (
            "With no games played yet, playoff_pct sits at 78.5%, and the roster "
            "is built to win now around Ja'Marr Chase."
        ),
        "trade_posture": "This is a contender with 78.5% playoff_pct.",
    })
    assert "playoff_pct" not in result["outlook"]
    assert "playoff odds sit at 78.5%" in result["outlook"]
    assert "78.5% playoff odds" in result["trade_posture"]


def test_scrub_bare_out_leadin_and_missing_odds_narration():
    """Screenshot: outlook opened with 'out, with playoff odds not provided here…'."""
    raw = (
        "out, with playoff odds not provided here and a roster shape that leans "
        "heavily on James Cook III, Chase Brown, and Breece Hall."
    )
    out = scrub_ai_prose_field_names(raw)
    assert not out.lower().startswith("out,")
    assert "not provided" not in out.lower()
    assert "This team looks out of playoff contention" in out
    assert "James Cook III" in out
    # Leading clause removal should leave a grammatical continuation.
    assert ", and a roster shape" in out or "and a roster shape" in out


def test_scrub_bare_bubble_and_contend_leadins():
    assert scrub_ai_prose_field_names("bubble, with thin TE.").startswith(
        "This team is on the playoff bubble"
    )
    assert scrub_ai_prose_field_names("contend, with elite WRs.").startswith(
        "This team is built to contend"
    )


def test_redraft_honesty_forbids_echoing_playoff_pct_key():
    # Import prompts lazily — it pulls OpenAI via client; skip if unavailable.
    import pytest
    pytest.importorskip("openai")
    from dashboard_services.ai.prompts import REDRAFT_HONESTY_RULES, build_gm_memo_prompt

    assert "NEVER write the raw key name playoff_pct" in REDRAFT_HONESTY_RULES
    assert "Never start a sentence with the bare labels" in REDRAFT_HONESTY_RULES
    assert 'Never write that odds were "not provided"' in REDRAFT_HONESTY_RULES
    prompt = build_gm_memo_prompt({"scoring_type": "redraft", "team_name": "A"}, "redraft")
    assert 'never write the key name "playoff_pct"' in prompt
    assert "78.5% playoff odds" in prompt
    assert "never say they were not provided" in prompt
