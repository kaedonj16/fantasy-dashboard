"""Front Office Report / GM memo switches dynasty vs redraft by league type.

ESPN is always redraft (no dynasty product). Other platforms use settings.type.
The report used to hardcode dynasty prompts and dynasty values regardless.
"""
import pytest

pytest.importorskip("openai")

from dashboard_services.ai.context_builders import (  # noqa: E402
    build_model_value_lookup,
    build_power_rankings_context,
    build_team_gm_context,
    ctx_scoring_type,
    detect_team_direction,
    redraft_window_label,
)
from dashboard_services.ai.prompts import (  # noqa: E402
    GM_MEMO_SYSTEM,
    GM_MEMO_SYSTEM_REDRAFT,
    REDRAFT_HONESTY_RULES,
    build_front_office_brief_prompt,
    build_gm_memo_prompt,
    build_power_rankings_prompt,
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


def test_ctx_scoring_type_yahoo_defaults_to_redraft():
    """Yahoo has no dynasty product. A missing type must not silently use
    dynasty values for trade suggestions (the calculator already defaults
    redraft via app._league_is_redraft)."""
    assert ctx_scoring_type({"platform": "yahoo"}) == "redraft"
    assert ctx_scoring_type({"platform": "yahoo", "league_settings": {}}) == "redraft"
    assert ctx_scoring_type({"platform": "yahoo", "league_settings": {"type": 0}}) == "redraft"
    assert ctx_scoring_type({"platform": "yahoo", "league_settings": {"type": 2}}) == "dynasty"
    assert ctx_scoring_type({
        "platform": "yahoo",
        "league_settings": {"league_type": "dynasty"},
    }) == "dynasty"


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
    assert detect_team_direction(weak, picks, scoring_type="redraft") == "out"
    assert detect_team_direction(weak, picks, scoring_type="redraft") != "retool"
    assert detect_team_direction(weak, picks, scoring_type="dynasty") == "rebuild"


def test_redraft_gm_memo_prompt_forbids_dynasty_framing():
    prompt = build_gm_memo_prompt({"scoring_type": "redraft", "team_name": "A"}, "redraft")
    prompt_l = prompt.lower()
    assert "redraft" in prompt_l
    assert "not a dynasty league" in prompt_l
    assert "prioritize waivers" in prompt_l
    assert "rebuild aggressively" not in prompt_l
    assert "personalized dynasty gm memo" not in prompt_l
    assert "record context is missing" in prompt_l
    assert "playoff_status" in prompt_l
    assert "retool" in prompt_l
    assert REDRAFT_HONESTY_RULES.splitlines()[0] in GM_MEMO_SYSTEM_REDRAFT

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
    assert "playoff_status" in prompt
    assert "weakest_positions" in prompt
    assert "retool" in prompt


def test_redraft_power_rankings_prompt_uses_odds_not_windows():
    system, user = build_power_rankings_prompt({
        "scoring_type": "redraft",
        "season_phase": "preseason",
        "teams": [{"roster_id": "1", "playoff_status": "bubble", "playoff_pct": 46.3}],
    })
    blob = (system + "\n" + user).lower()
    assert "redraft" in blob
    assert "playoff_status" in blob
    assert "retool" in blob
    assert "never mention draft capital" in blob
    assert "win_window guide" not in blob


def test_dynasty_power_rankings_prompt_keeps_windows():
    system, user = build_power_rankings_prompt({"scoring_type": "dynasty"})
    blob = (system + "\n" + user).lower()
    assert "win_window" in blob
    assert "full rebuild" in blob


def test_build_team_gm_context_attaches_playoff_snapshot():
    ctx = {
        "league_id": "espn-1",
        "platform": "espn",
        "league_settings": {"playoff_teams": 6},
        "current_season": 2026,
        "current_week": 0,
        "roster_positions": ["QB", "RB", "WR", "TE"],
        "rosters": [{"roster_id": 1, "players": ["p1"]}],
        "roster_map": {"1": "Team A"},
        "standings_map": {"1": {"record": "0-0", "PF": 0, "PA": 0}},
        "picks_by_roster": {},
        "players_index": {"p1": {"full_name": "Star RB", "position": "RB", "age": 24}},
        "players_map": {},
        "model_value_table": [{
            "id": "p1", "name": "Star RB", "position": "RB",
            "value": 200, "redraft_value_1qb": 880, "redraft_value_sf": 880,
        }],
        "playoff_odds": [
            {"roster_id": 1, "playoff_pct": 46.3},
            {"roster_id": 2, "playoff_pct": 80.0},
        ],
        "team_grades": {"1": {"grade": "B-"}},
    }
    team_ctx = build_team_gm_context(ctx, "1")
    assert team_ctx["season_phase"] == "preseason"
    assert team_ctx["playoff_pct"] == 46.3
    assert team_ctx["playoff_rank"] == 2
    assert team_ctx["playoff_status"] == "bubble"
    assert team_ctx["direction"] == "bubble"
    assert team_ctx["draft_grade"] == "B-"
    assert "RB" in team_ctx["weakest_positions"] or team_ctx["weakest_positions"]


def test_build_power_rankings_context_redraft_uses_odds_not_picks():
    ctx = {
        "league_id": "espn-1",
        "platform": "espn",
        "league_settings": {"playoff_teams": 6},
        "current_season": 2026,
        "current_week": 0,
        "rosters": [
            {"roster_id": 1, "players": ["p1"], "settings": {"wins": 0, "losses": 0, "fpts": 0}},
            {"roster_id": 2, "players": ["p2"], "settings": {"wins": 0, "losses": 0, "fpts": 0}},
        ],
        "roster_map": {"1": "Team A", "2": "Team B"},
        "standings_map": {},
        "picks_by_roster": {"1": [{"display": "2027 1.01"}]},
        "players_index": {
            "p1": {"full_name": "Star RB", "position": "RB", "age": 24},
            "p2": {"full_name": "Other WR", "position": "WR", "age": 25},
        },
        "players_map": {},
        "model_value_table": [
            {"id": "p1", "name": "Star RB", "position": "RB",
             "value": 200, "redraft_value_1qb": 880, "redraft_value_sf": 880},
            {"id": "p2", "name": "Other WR", "position": "WR",
             "value": 200, "redraft_value_1qb": 400, "redraft_value_sf": 400},
        ],
        "playoff_odds": [
            {"roster_id": 1, "playoff_pct": 46.3},
            {"roster_id": 2, "playoff_pct": 80.0},
        ],
    }
    out = build_power_rankings_context(ctx)
    assert out["scoring_type"] == "redraft"
    assert out["season_phase"] == "preseason"
    by_id = {t["roster_id"]: t for t in out["teams"]}
    assert by_id["1"]["playoff_pct"] == 46.3
    assert by_id["1"]["playoff_status"] == "bubble"
    assert by_id["1"]["direction"] == "bubble"
    assert by_id["1"]["future_picks"] == []
    assert by_id["1"]["win_window"] == "Bubble"
    assert by_id["2"]["playoff_status"] == "contend"
    assert by_id["2"]["direction"] == "contend"
    assert by_id["2"]["win_window"] == "Contend"


def test_redraft_window_label_uses_odds_then_percentile():
    assert redraft_window_label(playoff_pct=87) == "Contend"
    assert redraft_window_label(playoff_pct=46.3) == "Bubble"
    assert redraft_window_label(playoff_pct=12) == "Long Shot"
    # No odds: same bands on 0–1 redraft-value percentile.
    assert redraft_window_label(redraft_pct=0.80) == "Contend"
    assert redraft_window_label(redraft_pct=0.50) == "Bubble"
    assert redraft_window_label(redraft_pct=0.10) == "Long Shot"
    # Odds win when both are present — a mid roster with 80% odds is Contend.
    assert redraft_window_label(playoff_pct=80, redraft_pct=0.10) == "Contend"
    assert redraft_window_label() == ""
    assert "Retool" not in (redraft_window_label(playoff_pct=20) or "")
    assert "Window" not in (redraft_window_label(playoff_pct=90) or "")
