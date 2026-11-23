"""Trade Analyst verdict guard: a decisive value win cannot be a DECLINE.

Bug (Kaedon, Week 11 2026, Blackedraw): the calculator showed Caleb's
Casting Couch (8-2, league-high points) receiving Joe Burrow (428.0) +
James Cook III (620.4) = 982.1 after the depth adjustment, and Fire Riley
receiving Caleb Williams (403.5) + Emeka Egbuka (313.7) + Quinshon Judkins
(217.7) = 755.8, a +226.3 depth-adjusted delta (raw +113.5). The BR Trade
Analyst still returned DECLINE / High confidence, on age-curve grounds
alone ("young dynasty insulation" for "two older, proven pieces").

The system prompt already forbids that (market delta is factor 1; age /
direction must never override a decisive delta; only a post-trade roster
that cannot field a starter may decline a value win), but the model
disobeyed it. apply_trade_verdict_guard enforces the framework server-side.
These tests pin the exact reported case shape plus the controls: a
decisively negative trade, and a genuinely catastrophic value-positive
trade, can still be DECLINE.
"""
import pytest

pytest.importorskip("flask")

# NOTE: dashboard_services imports stay inside the tests. The renderer chain
# imports the openai package at module level, which the pure-Python lint CI
# shard does not install; a module-level import here would fail collection
# there (pytest exit 2) even though this file is auto-marked integration.

GETS = [
    {"name": "Joe Burrow", "position": "QB", "value": 428.0},
    {"name": "James Cook III", "position": "RB", "value": 620.4},
]
GIVES = [
    {"name": "Caleb Williams", "position": "QB", "value": 403.5},
    {"name": "Emeka Egbuka", "position": "WR", "value": 313.7},
    {"name": "Quinshon Judkins", "position": "RB", "value": 217.7},
]
# Kaedon's pre-trade Blackedraw body counts (QB: Caleb/Purdy/Nix, etc).
PRE_COUNTS = {"QB": 3, "RB": 6, "WR": 8, "TE": 2}
# Blackedraw starting lineup shape: 1 QB, 2 RB, 3 WR, 1 TE, 1 FLEX.
SLOTS = {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 1}

DECLINE_RESULT = {
    "verdict": "DECLINE",
    "summary": (
        "You would be shipping out a lot of young dynasty insulation for "
        "two older, proven pieces."
    ),
    "helps": ["Joe Burrow gives you a premier quarterback anchor."],
    "risks": ["You lose a promising young QB1 asset, a WR2 asset, and a RB3 asset."],
    "counter": "",
    "confidence": "high",
}


def _guard(result, *, delta, gets=GETS, gives=GIVES, pre=PRE_COUNTS, slots=SLOTS):
    from dashboard_services.ai.renderer import (
        apply_trade_verdict_guard,
        post_trade_position_counts,
    )

    return apply_trade_verdict_guard(
        result,
        market_delta=delta,
        gets_assets=gets,
        gives_assets=gives,
        pos_counts=post_trade_position_counts(pre, gets, gives),
        slot_counts=slots,
    )


def test_reported_case_decline_is_overridden_to_accept():
    out = _guard(dict(DECLINE_RESULT), delta=226.3)
    assert out["verdict"] == "ACCEPT"
    # The narrative must not contradict the new verdict.
    assert "226.3" in out["summary"]
    assert "insulation" not in out["summary"]
    assert out["counter"] == ""
    rendered_helps = " ".join(out["helps"])
    assert "Joe Burrow" in rendered_helps and "James Cook III" in rendered_helps


def test_reported_case_counter_is_overridden_at_raw_delta_too():
    result = dict(DECLINE_RESULT, verdict="COUNTER", counter="Ask for a 1st.")
    out = _guard(result, delta=113.5)
    assert out["verdict"] == "ACCEPT"
    assert out["counter"] == ""


def test_reported_case_renders_accept_badge_not_decline():
    from dashboard_services.ai.renderer import render_trade_ai_html

    out = _guard(dict(DECLINE_RESULT), delta=226.3)
    html_out = render_trade_ai_html(out)
    assert 'data-verdict="ACCEPT"' in html_out
    assert "trade-ai-verdict-badge-accept" in html_out
    assert "insulation" not in html_out


def test_decisively_negative_delta_can_still_decline():
    out = _guard(dict(DECLINE_RESULT), delta=-226.3)
    assert out["verdict"] == "DECLINE"
    assert out["summary"] == DECLINE_RESULT["summary"]


def test_slight_loss_counter_stands():
    result = dict(DECLINE_RESULT, verdict="COUNTER", counter="Ask for a 2nd.")
    out = _guard(result, delta=-60.0)
    assert out["verdict"] == "COUNTER"
    assert out["counter"] == "Ask for a 2nd."


def test_fair_trade_decline_is_overridden():
    out = _guard(dict(DECLINE_RESULT), delta=0.0)
    assert out["verdict"] == "ACCEPT"


def test_accept_verdict_is_untouched():
    result = dict(DECLINE_RESULT, verdict="ACCEPT")
    out = _guard(result, delta=226.3)
    assert out["summary"] == DECLINE_RESULT["summary"]


def test_catastrophic_value_win_can_still_decline():
    # Viewer sends his only QB and gets no QB back: the post-trade roster
    # cannot field a starting QB at all. That is the prompt's one
    # deterministic catastrophe, so the guard leaves the DECLINE alone.
    gives = [
        {"name": "Caleb Williams", "position": "QB", "value": 403.5},
        {"name": "Quinshon Judkins", "position": "RB", "value": 217.7},
    ]
    gets = [
        {"name": "James Cook III", "position": "RB", "value": 620.4},
        {"name": "Kenneth Walker III", "position": "RB", "value": 500.0},
    ]
    out = _guard(
        dict(DECLINE_RESULT), delta=226.3,
        gets=gets, gives=gives, pre={"QB": 1, "RB": 6, "WR": 8, "TE": 2},
    )
    assert out["verdict"] == "DECLINE"


def test_post_trade_position_counts_ignores_picks():
    from dashboard_services.ai.renderer import post_trade_position_counts

    counts = post_trade_position_counts(
        PRE_COUNTS,
        GETS + [{"name": "2028 2nd", "position": "PICK", "value": 100.0}],
        GIVES,
    )
    assert counts == {"QB": 3, "RB": 6, "WR": 7, "TE": 2}


def test_structural_catastrophe_needs_a_real_hole():
    from dashboard_services.ai.renderer import trade_structural_catastrophe

    assert not trade_structural_catastrophe(
        {"QB": 3, "RB": 6, "WR": 7, "TE": 2}, SLOTS)
    assert trade_structural_catastrophe(
        {"QB": 0, "RB": 6, "WR": 7, "TE": 2}, SLOTS)
    # Flex cannot be fielded: exactly the direct starters and nothing more,
    # with two flex slots to fill.
    assert trade_structural_catastrophe(
        {"QB": 1, "RB": 2, "WR": 3, "TE": 1},
        {"QB": 1, "RB": 2, "WR": 3, "TE": 1, "FLEX": 2},
    )
    # Unknown slots prove nothing.
    assert not trade_structural_catastrophe({"QB": 0, "RB": 0, "WR": 0, "TE": 0}, {})


# ── v2: percentage-hybrid bands ─────────────────────────────────────────────

def _assessment(**kw):
    from dashboard_services.ai.renderer import trade_verdict_assessment

    kw.setdefault("pos_counts", {"QB": 3, "RB": 6, "WR": 7, "TE": 2})
    kw.setdefault("slot_counts", SLOTS)
    return trade_verdict_assessment(**kw)


def test_big_trade_small_pct_win_is_fair_not_decisive():
    # +60 on a ~3000-value trade is 2%: inside the site's fair band
    # (7% of 3000 = 210), so it earns fair treatment, never decisive
    # treatment (the old flat +50 line called this a clear win).
    a = _assessment(adjusted_delta=60.0, raw_delta=60.0, larger_side=3000.0)
    assert a["fair_band"] == 210.0
    assert a["decisive_bar"] == 210.0
    assert a["decisive"] is False
    # A DECLINE there is overridden as a fair trade, but the confidence
    # is deterministic and low -- not the model's, and not "high".
    out = _guard(dict(DECLINE_RESULT), delta=60.0)
    assert out["verdict"] == "ACCEPT"
    from dashboard_services.ai.renderer import trade_deterministic_confidence
    assert trade_deterministic_confidence(
        "ACCEPT", 60.0, a["fair_band"], a["decisive_bar"],
        decisive_ok=a["decisive"]) == "low"


def test_decisive_needs_absolute_floor_and_percentage():
    # +60 on a 400-value trade clears both bars (fair 28, decisive 50).
    a = _assessment(adjusted_delta=60.0, raw_delta=60.0, larger_side=400.0)
    assert a["decisive"] is True
    # +40 on the same trade is a clear win but not decisive.
    a = _assessment(adjusted_delta=40.0, raw_delta=40.0, larger_side=400.0)
    assert a["decisive"] is False
    assert a["assigned_verdict"] == "ACCEPT"


# ── v2: raw + adjusted double check ─────────────────────────────────────────

def test_slot_penalty_only_win_is_not_decisive():
    # Adjusted +226.3 (the reported shape) but the raw delta is a loss:
    # the "win" is manufactured by the roster-slot penalty, so it is
    # never decisive and never earns high confidence.
    a = _assessment(adjusted_delta=226.3, raw_delta=-20.0, larger_side=982.1)
    assert a["decisive"] is False
    from dashboard_services.ai.renderer import trade_deterministic_confidence
    assert trade_deterministic_confidence(
        "ACCEPT", 226.3, a["fair_band"], a["decisive_bar"],
        decisive_ok=a["decisive"]) == "medium"


def test_slot_penalty_only_win_with_real_raw_loss_is_not_forced():
    # Adjusted +80 but raw -100 on a 1000-value trade (fair band 70):
    # the raw market says a real loss, so a COUNTER stands untouched.
    from dashboard_services.ai.renderer import apply_trade_verdict_guard

    result = dict(DECLINE_RESULT, verdict="COUNTER", counter="Ask for a 2nd.")
    out = apply_trade_verdict_guard(
        result,
        market_delta=80.0,
        raw_delta=-100.0,
        larger_side=1000.0,
        gets_assets=GETS,
        gives_assets=GIVES,
        pos_counts={"QB": 3, "RB": 6, "WR": 7, "TE": 2},
        slot_counts=SLOTS,
    )
    assert out["verdict"] == "COUNTER"
    assert out["counter"] == "Ask for a 2nd."


# ── v2: starter-caliber catastrophe ─────────────────────────────────────────

# 10-team thresholds from derive_league_thresholds (500/350/350/200 x 12/10).
THRESHOLDS_10 = {"QB": 600, "RB": 420, "WR": 420, "TE": 240}


def test_backup_qb_does_not_count_as_a_starter():
    from dashboard_services.ai.renderer import starter_caliber_counts

    roster = [
        {"id": "qb1", "position": "QB", "value": 120.0},   # backup body
        {"id": "rb1", "position": "RB", "value": 500.0},
    ]
    counts = starter_caliber_counts(roster, THRESHOLDS_10, {})
    assert counts["QB"] == 0
    assert counts["RB"] == 1


def test_ir_starter_does_not_count_toward_caliber():
    from dashboard_services.ai.renderer import starter_caliber_counts

    roster = [{"id": "qb1", "position": "QB", "value": 700.0}]
    counts = starter_caliber_counts(roster, THRESHOLDS_10, {"qb1": "IR"})
    assert counts["QB"] == 0


def test_fair_band_trade_stripping_only_caliber_qb_is_not_forced_accept():
    # Fair-band trade (+20 on an 800-value deal, fair band 56) that sends
    # the viewer's only starter-caliber QB and gets none back: the
    # catastrophe exception applies across the whole accept band, so the
    # DECLINE is NOT overridden (the old guard only excepted >= +50).
    from dashboard_services.ai.renderer import (
        apply_trade_verdict_guard,
        post_trade_caliber_counts,
    )

    pre_caliber = {"QB": 1, "RB": 3, "WR": 4, "TE": 1}
    gives = [{"name": "Caleb Williams", "position": "QB", "value": 650.0}]
    gets = [{"name": "James Cook III", "position": "RB", "value": 670.0}]
    post_caliber = post_trade_caliber_counts(
        pre_caliber, gets, gives, THRESHOLDS_10)
    assert post_caliber["QB"] == 0
    out = apply_trade_verdict_guard(
        dict(DECLINE_RESULT),
        market_delta=20.0,
        raw_delta=20.0,
        larger_side=800.0,
        gets_assets=gets,
        gives_assets=gives,
        pos_counts={"QB": 2, "RB": 6, "WR": 8, "TE": 2},
        slot_counts=SLOTS,
        caliber_counts=post_caliber,
    )
    assert out["verdict"] == "DECLINE"


def test_decisive_win_with_caliber_intact_still_overrides():
    from dashboard_services.ai.renderer import (
        apply_trade_verdict_guard,
        post_trade_caliber_counts,
    )

    pre_caliber = {"QB": 2, "RB": 3, "WR": 4, "TE": 1}
    post_caliber = post_trade_caliber_counts(
        pre_caliber, GETS, GIVES, THRESHOLDS_10)
    # Burrow (428) is below the 10-team QB bar (600) on the 1QB scale the
    # calculator displays, Caleb (403.5) too: neither moves caliber, and
    # the room keeps its caliber bodies, so no catastrophe.
    out = apply_trade_verdict_guard(
        dict(DECLINE_RESULT),
        market_delta=226.3,
        raw_delta=113.5,
        larger_side=982.1,
        gets_assets=GETS,
        gives_assets=GIVES,
        pos_counts={"QB": 3, "RB": 6, "WR": 7, "TE": 2},
        slot_counts=SLOTS,
        caliber_counts=post_caliber,
    )
    assert out["verdict"] == "ACCEPT"
    assert out["confidence"] == "high"


# ── v2: deterministic confidence + override logging ─────────────────────────

def test_override_confidence_is_deterministic_not_inherited(caplog):
    import logging

    from dashboard_services.ai.renderer import apply_trade_verdict_guard

    # Model said DECLINE / high at +30 (fair 25, decisive 50 on a small
    # trade): the override must not inherit "high".
    with caplog.at_level(logging.INFO, logger="dashboard_services.ai.renderer"):
        out = apply_trade_verdict_guard(
            dict(DECLINE_RESULT),
            market_delta=30.0,
            raw_delta=30.0,
            larger_side=300.0,
            gets_assets=GETS,
            gives_assets=GIVES,
            pos_counts={"QB": 3, "RB": 6, "WR": 7, "TE": 2},
            slot_counts=SLOTS,
        )
    assert out["verdict"] == "ACCEPT"
    assert out["confidence"] == "medium"
    override_logs = [
        r.getMessage() for r in caplog.records
        if "[trade-guard] override" in r.getMessage()
    ]
    assert override_logs, "guard override must be logged for calibration"
    assert "pre=DECLINE" in override_logs[0]
    assert "post=ACCEPT" in override_logs[0]
    assert "adjusted_delta=30.0" in override_logs[0]
    assert "raw_delta=30.0" in override_logs[0]


# ── v2: partner acceptance split ────────────────────────────────────────────

def test_partner_acceptance_labels():
    from dashboard_services.ai.renderer import partner_acceptance_label

    # The reported case: partner asked to lose +226.3 of adjusted value.
    assert partner_acceptance_label(226.3, 982.1) == "Long shot"
    # Partner breaks even or gains: Likely.
    assert partner_acceptance_label(0.0, 982.1) == "Likely"
    assert partner_acceptance_label(-40.0, 982.1) == "Likely"
    # Partner loses between the fair band and the decisive bar: Possible
    # (400-value trade: fair 28, decisive 50).
    assert partner_acceptance_label(40.0, 400.0) == "Possible"
    assert partner_acceptance_label(60.0, 400.0) == "Long shot"


def test_partner_acceptance_renders_as_its_own_block():
    from dashboard_services.ai.renderer import (
        partner_acceptance_note,
        render_trade_ai_html,
    )

    result = dict(DECLINE_RESULT, verdict="ACCEPT")
    result["partner_acceptance"] = "Long shot"
    result["partner_acceptance_note"] = partner_acceptance_note("Long shot", 226.3)
    html_out = render_trade_ai_html(result)
    assert "Partner acceptance" in html_out
    assert "Long shot" in html_out
    assert 'data-verdict="ACCEPT"' in html_out


# ── v2: verdict-first assigned verdict in the prompt ────────────────────────

def test_assigned_verdict_is_injected_into_user_prompt():
    from dashboard_services.ai.prompts import build_trade_ai_user_prompt

    payload = {
        "scoring_type": "dynasty",
        "trade": {
            "assigned_verdict": "ACCEPT",
            "viewer_gets": {"assets": []},
            "viewer_gives": {"assets": []},
        },
    }
    prompt = build_trade_ai_user_prompt(payload, "dynasty")
    assert "ASSIGNED VERDICT: ACCEPT" in prompt
    payload["trade"]["assigned_verdict"] = None
    prompt = build_trade_ai_user_prompt(payload, "dynasty")
    assert "ASSIGNED VERDICT" not in prompt


def test_prompt_thresholds_use_bands_not_flat_numbers():
    from dashboard_services.ai.prompts import build_trade_ai_system_prompt

    for scoring_type in ("dynasty", "redraft"):
        system = build_trade_ai_system_prompt(scoring_type)
        assert "fair_band" in system and "decisive_bar" in system
        assert "delta >= +250" not in system
        assert "delta -10 to +10" not in system


# ── v2: direction reads the contention window first ─────────────────────────

def _players(specs):
    return [
        {"age": age, "value": value} for age, value in specs
    ]


def test_direction_contender_from_window_despite_old_roster():
    from dashboard_services.ai.context_builders import detect_team_direction

    # Old (avg 31), no elite assets: the age-only read never returns
    # "contender" for this roster. An 8-2 record, league-high PF, and
    # contend-level odds must win anyway.
    players = _players([(31, 400), (32, 380), (30, 350), (31, 300)])
    direction = detect_team_direction(
        players, [], scoring_type="dynasty",
        playoff_pct=85.0, wins=8, losses=2, pf_rank=1, num_teams=10,
    )
    assert direction == "contender"


def test_direction_young_losing_team_is_not_a_contender():
    from dashboard_services.ai.context_builders import detect_team_direction

    # Young with elite assets: the age-only read calls this a contender.
    # At 2-8 with 10% playoff odds it is not one.
    players = _players([(24, 800), (23, 780), (25, 760), (24, 500)])
    picks = [{"display": "2027 1st"}, {"display": "2028 1st"}]
    direction = detect_team_direction(
        players, picks, scoring_type="dynasty",
        playoff_pct=10.0, wins=2, losses=8, pf_rank=9, num_teams=10,
    )
    assert direction != "contender"
    assert direction == "rebuild"


def test_direction_without_window_evidence_keeps_asset_read():
    from dashboard_services.ai.context_builders import detect_team_direction

    players = _players([(25, 800), (24, 780), (26, 760), (25, 600), (24, 580), (25, 560)])
    assert detect_team_direction(players, [], scoring_type="dynasty") == "contender"
    # Redraft branch is untouched by the window kwargs.
    assert detect_team_direction(
        players, [], scoring_type="redraft",
        playoff_pct=5.0, wins=1, losses=9,
    ) in ("contend", "bubble", "out")
