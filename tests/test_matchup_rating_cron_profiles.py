from data_building.matchup_ratings import (
    COMMON_SCORING_PROFILES,
    build_matchup_rating_profiles,
)
from utils.defensive_matchup_ratings import scoring_profile_hash


def test_default_and_deduplicated_common_profiles_isolate_optional_failures(monkeypatch):
    # Exercise hash deduplication explicitly without expanding the production
    # job list: the duplicate represents two labels for one supported profile.
    monkeypatch.setattr("data_building.matchup_ratings.COMMON_SCORING_PROFILES", (
        *COMMON_SCORING_PROFILES, ("half-PPR alias", {"rec": 0.5})))
    calls = []

    def builder(season, scoring_settings=None):
        calls.append(scoring_settings)
        if scoring_settings == {"rec": 0.5}:
            raise RuntimeError("optional half profile failed")
        return {"ratings": {"BUF": {}}}

    logs = []
    statuses = build_matchup_rating_profiles(2026, builder=builder, logger=logs.append)

    assert calls[0] is None  # reliable default is always attempted first
    default_hash = scoring_profile_hash({"rec": 1.0})
    assert {scoring_profile_hash(s) for s in calls[1:]} == {
        scoring_profile_hash(settings) for _, settings in COMMON_SCORING_PROFILES
        if scoring_profile_hash(settings) != default_hash}
    assert next(s for s in statuses if s["label"] == "full PPR")["status"] == "deduplicated"
    assert next(s for s in statuses if s["label"] == "half-PPR")["status"] == "failed"
    assert next(s for s in statuses if s["label"] == "half-PPR alias")["status"] == "deduplicated"
    assert next(s for s in statuses if s["label"] == "standard TEP")["status"] == "built"
    assert all("hash=" in line and "label=" in line and "teams=" in line and "status=" in line
               for line in logs)
    assert any("reason=optional half profile failed" in line for line in logs)


def test_default_result_materializes_full_ppr_profile_without_second_build(tmp_path, monkeypatch):
    calls = []

    def builder(season, scoring_settings=None):
        calls.append(scoring_settings)
        return {"season": season, "scoring_profile": scoring_profile_hash(
            scoring_settings or {"rec": 1.0}), "ratings": {"BUF": {}}}

    monkeypatch.setattr("data_building.matchup_ratings.CACHE_DIR", tmp_path)
    monkeypatch.setattr("data_building.matchup_ratings.build_matchup_ratings", builder)
    statuses = build_matchup_rating_profiles(2026, logger=lambda _message: None)

    assert calls.count(None) == 1
    assert {"rec": 1.0} not in calls
    full_ppr = tmp_path / (
        f"matchup_ratings_s2026_{scoring_profile_hash({'rec': 1.0})}.json")
    assert full_ppr.exists()
    assert next(s for s in statuses if s["label"] == "full PPR")["team_count"] == 1
