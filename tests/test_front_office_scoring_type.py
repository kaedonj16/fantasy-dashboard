"""Front Office Report / GM memo switches dynasty vs redraft by league type.

ESPN is always redraft (no dynasty product). Other platforms use settings.type.
The report used to hardcode dynasty prompts and dynasty values regardless.
"""
import pytest

pytest.importorskip("openai")

from dashboard_services.ai.context_builders import (  # noqa: E402
    build_model_value_lookup,
    build_team_gm_context,
    ctx_scoring_type,
    detect_team_direction,
)
from dashboard_services.ai.prompts import (  # noqa: E402
    GM_MEMO_SYSTEM,
    GM_MEMO_SYSTEM_REDRAFT,
    build_front_office_brief_prompt,
    build_gm_memo_prompt,
)


def test_ctx_scoring_type_espn_is_always_redraft():
    # Platform alone — ESPN has no dynasty leagues.
    assert ctx_scoring_type({"platform": "espn"}) == "redraft"
    assert ctx_scoring_type({"platform": "ESPN", "league_settings": {}}) == "redraft"
    # Even a bogus type must not flip ESPN to dynasty.
    assert ctx_scoring_type({"platform": "espn", "league_settings": {"type": 2}}) == "redraft"


def test_ctx_scoring_type_uses_settings_type_for_other_platforms():
    assert ctx_scoring_type({"league_settings": {"type": 0}}) == "redraft"
    assert ctx_scoring_type({"league_settings": {"type": 1}}) == "redraft"
    assert ctx_scoring_type({"league_settings": {"type": 2}}) == "dynasty"
    assert ctx_scoring_type({"platform": "sleeper", "league_settings": {"type": 2}}) == "dynasty"
    assert ctx_scoring_type({}) == "dynasty"


def test_build_model_value_lookup_uses_redraft_values():
    rows = [{
        "id": "1",
        "value": 100,
        "sf_value": 120,
        "redraft_value_1qb": 400,
        "redraft_value_sf": 450,
    }]
    dynasty = build_model_value_lookup(rows, is_sf=False, scoring_type="dynasty")
    assert dynasty["1"]["value"] == 100

    redraft = build_model_value_lookup(rows, is_sf=False, scoring_type="redraft")
    assert redraft["1"]["value"] == 400

    redraft_sf = build_model_value_lookup(rows, is_sf=True, scoring_type="redraft")
    assert redraft_sf["1"]["value"] == 450


def test_build_team_gm_context_espn_uses_redraft_values_and_no_picks():
    ctx = {
        "league_id": "espn-1",
        "platform": "espn",
        "league_settings": {},  # type optional — ESPN is redraft by platform
        "current_season": 2026,
        "current_week": 1,
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX"],
        "rosters": [{"roster_id": 1, "players": ["p1"]}],
        "roster_map": {"1": "Team A"},
        "standings_map": {"1": {"record": "0-0", "PF": 0, "PA": 0}},
        "picks_by_roster": {"1": [{"display": "2027 1.01", "season": "2027", "round": 1}]},
        "players_index": {"p1": {"full_name": "Star RB", "position": "RB", "age": 24}},
        "players_map": {},
        "model_value_table": [{
            "id": "p1",
            "name": "Star RB",
            "position": "RB",
            "value": 200,
            "redraft_value_1qb": 880,
            "redraft_value_sf": 880,
        }],
    }
    team_ctx = build_team_gm_context(ctx, "1")
    assert team_ctx is not None
    assert team_ctx["scoring_type"] == "redraft"
    assert team_ctx["future_picks"] == []
    assert team_ctx["aging_assets"] == []
    assert team_ctx["top_assets"][0]["value"] == 880.0


def test_build_team_gm_context_dynasty_keeps_picks_and_dynasty_values():
    ctx = {
        "league_id": "sl-1",
        "league_settings": {"type": 2},
        "current_season": 2026,
        "rosters": [{"roster_id": 1, "players": ["p1"]}],
        "roster_map": {"1": "Team A"},
        "standings_map": {},
        "picks_by_roster": {"1": [{"display": "2027 1.01"}]},
        "players_index": {"p1": {"full_name": "Young WR", "position": "WR", "age": 23}},
        "players_map": {},
        "model_value_table": [{
            "id": "p1",
            "value": 700,
            "redraft_value_1qb": 500,
        }],
    }
    team_ctx = build_team_gm_context(ctx, "1")
    assert team_ctx["scoring_type"] == "dynasty"
    assert team_ctx["future_picks"]
    assert team_ctx["top_assets"][0]["value"] == 700.0


def test_detect_team_direction_redraft_ignores_future_firsts():
    weak = [{"value": 100, "age": 24} for _ in range(5)]
    picks = [{"display": "2027 1.01"}, {"display": "2028 1.02"}, {"display": "2029 1.03"}]
    assert detect_team_direction(weak, picks, scoring_type="redraft") != "rebuild"
    assert detect_team_direction(weak, picks, scoring_type="dynasty") == "rebuild"


def test_redraft_gm_memo_prompt_forbids_dynasty_framing():
    prompt = build_gm_memo_prompt({"scoring_type": "redraft", "team_name": "A"}, "redraft")
    prompt_l = prompt.lower()
    assert "redraft" in prompt_l
    assert "not a dynasty league" in prompt_l
    assert "prioritize waivers" in prompt_l
    assert "rebuild aggressively" not in prompt_l
    assert "personalized dynasty gm memo" not in prompt_l

    assert "redraft" in GM_MEMO_SYSTEM_REDRAFT.lower()
    assert "dynasty" in GM_MEMO_SYSTEM.lower()


def test_dynasty_gm_memo_prompt_keeps_rebuild_verdict():
    prompt = build_gm_memo_prompt({"scoring_type": "dynasty"}, "dynasty").lower()
    assert "personalized dynasty gm memo" in prompt
    assert "rebuild aggressively" in prompt
    assert "prioritize waivers" not in prompt


def test_redraft_front_office_brief_prompt():
    prompt = build_front_office_brief_prompt({"scoring_type": "redraft"}, "redraft").lower()
    assert "redraft team" in prompt
    assert "not a dynasty league" in prompt
    assert "never mention draft picks" in prompt
