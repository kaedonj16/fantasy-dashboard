"""Regression coverage for conservative narrative rushing and replay results."""
from utils.player_identity import PlayerIdentityResolver
from utils.redzone_pbp import (
    PLAY_STATE_CORRECTED, PLAY_STATE_OVERTURNED, _detect_play_state,
    extract_pbp_plays,
)


PLAYERS = {
    "9509": {"name": "Kenneth Walker III", "team": "SEA", "pos": "RB"},
    "other": {"name": "Ken Walker", "team": "MIA", "pos": "RB"},
}


def _box(text, **extra):
    return {"allPlayByPlay": [{
        "playId": "sea-1", "play": text, "possession": "SEA", **extra,
    }]}


def test_narrative_only_zero_yard_touchdown_resolves_verified_alias():
    plays = extract_pbp_plays(
        _box("Ken Walker rushed up the middle for 0 yards TOUCHDOWN"),
        "game", player_meta_by_pid=PLAYERS,
    )
    assert len(plays) == 1
    assert plays[0]["pid"] == "9509"
    assert plays[0]["stat_line"] == {"carries": 1, "rush_yds": 0, "rush_td": 1}


def test_narrative_rusher_requires_team_role_evidence_and_stays_neutral():
    plays = extract_pbp_plays(
        _box("Alex Smith rushed for 2 yards TOUCHDOWN"),
        "game", player_meta_by_pid=PLAYERS,
    )
    assert len(plays) == 1
    assert plays[0]["pid"] == ""
    assert plays[0]["stat_line"] == {}


def test_verified_alias_is_not_cross_team_fuzzy_matching():
    resolver = PlayerIdentityResolver(PLAYERS)
    assert resolver.resolve(name="Ken Walker", team="SEA", role="rusher")["canonical_player_id"] == "9509"
    assert resolver.resolve(name="Ken Walker", team="NYJ", role="rusher")["canonical_player_id"] == ""


def test_replay_reversal_uses_final_result_in_both_directions():
    assert _detect_play_state(
        {"playStatus": "overturned", "finalResult": "touchdown"},
        "Ruling overturned, result is a touchdown",
    ) == PLAY_STATE_CORRECTED
    assert _detect_play_state(
        {"playStatus": "overturned"}, "Touchdown ruling overturned on replay",
    ) == PLAY_STATE_OVERTURNED


def test_corrected_touchdown_is_emitted_once_per_payload():
    plays = extract_pbp_plays(
        _box("Ken Walker ruling overturned to touchdown for 1 yard", playStatus="overturned", finalResult="touchdown"),
        "game", player_meta_by_pid=PLAYERS,
    )
    # Narrative grammar intentionally does not invent a carry when the corrected
    # sentence does not identify the player as rushing.
    assert len(plays) == 1
    assert plays[0]["play_state"] == PLAY_STATE_CORRECTED
