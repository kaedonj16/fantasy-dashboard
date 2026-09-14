from utils.player_identity import PlayerIdentityResolver, normalize_player_name
from utils.redzone_pbp import extract_pbp_plays, pbp_boxscore_mismatches


PLAYERS = {
    "q1": {"name": "Drake Maye", "team": "NE", "pos": "QB", "tank01_id": "tank-q"},
    "w1": {"name": "Stefon Diggs Jr.", "team": "NE", "pos": "WR", "espn_id": "espn-w"},
    "a1": {"name": "John Smith", "team": "NYJ", "pos": "WR"},
    "a2": {"name": "John Smith", "team": "NYJ", "pos": "TE"},
}


def test_id_crosswalk_suffix_and_team_alias_resolution():
    r = PlayerIdentityResolver(PLAYERS)
    assert r.resolve(tank01_id="tank-q")["canonical_player_id"] == "q1"
    assert r.resolve(provider_player_id="espn-w")["canonical_player_id"] == "w1"
    found = r.resolve(name="Stefon Diggs", team="NWE", role="receiver")
    assert found["canonical_player_id"] == "w1"
    assert found["resolution_method"] == "team_name_position"
    assert normalize_player_name("Stefon Diggs, Jr.") == "stefon diggs"


def test_same_name_collision_stays_ambiguous_without_role():
    result = PlayerIdentityResolver(PLAYERS).resolve(name="John Smith", team="NYJ")
    assert result["canonical_player_id"] == ""
    assert result["confidence"] == "ambiguous"


def test_role_disambiguates_same_name_without_guessing():
    result = PlayerIdentityResolver(PLAYERS).resolve(name="John Smith", team="NYJ", position="TE")
    assert result["canonical_player_id"] == "a2"


def test_structured_pass_actors_use_canonical_ids():
    box = {"allPlayByPlay": [{"playId": "p1", "play": "D.Maye pass to S.Diggs for 8 yards",
        "playerStats": [
            {"playerID": "tank-q", "longName": "Drake Maye", "teamAbv": "NWE", "Passing": {"passYds": 8}},
            {"longName": "Stefon Diggs", "teamAbv": "NE", "Receiving": {"rec": 1, "recYds": 8, "targets": 1}},
        ]}]}
    plays = extract_pbp_plays(box, "g", player_meta_by_pid=PLAYERS)
    assert {(p["pid"], p["actor_role"]) for p in plays} == {("q1", "passer"), ("w1", "receiver")}


def test_boxscore_reconciliation_reports_bad_attribution_without_overwrite():
    plays = [{"pid": "w1", "play_state": "VALID", "stat_line": {"rec": 1},
              "identity": {"confidence": "strong"}}]
    mismatches = pbp_boxscore_mismatches(plays, {"w1": {"Receiving": {"rec": 2}}})
    assert mismatches == [{"player_id": "w1", "stat": "rec", "pbp": 1.0, "boxscore": 2.0}]


def test_corrected_play_retains_same_canonical_actor():
    resolver = PlayerIdentityResolver(PLAYERS)
    original = resolver.resolve(tank01_id="tank-q", name="Drake Maye", team="NE")
    corrected = resolver.resolve(tank01_id="tank-q", name="D. Maye", team="NWE")
    assert original["canonical_player_id"] == corrected["canonical_player_id"] == "q1"


def test_ambiguous_incomplete_target_is_not_awarded_points():
    box = {"allPlayByPlay": [{"playId": "inc", "play": "Pass incomplete to John Smith",
                              "playerStats": []}]}
    plays = extract_pbp_plays(box, "g", player_meta_by_pid=PLAYERS)
    assert not any(p.get("pid") in {"a1", "a2"} for p in plays)
