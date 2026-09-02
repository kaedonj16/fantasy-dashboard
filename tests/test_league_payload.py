"""Unit tests for utils.league_payload.

Pure logic — no app / DB import — so these run anywhere pytest does.
"""
from utils.league_payload import (
    build_roster_map,
    draft_countdown_copy,
    draft_start_ms,
    format_sleeper_league_option,
    get_most_recent_valid_draft_for_season,
    rosters_look_undrafted,
    show_matchup_preview,
    startup_draft_pending,
    startup_draft_phase,
    top_board_preview,
)


# ---- format_sleeper_league_option -----------------------------------------

def test_format_full_league():
    lg = {"league_id": 123, "name": "Dynasty Warriors", "season": 2024,
          "total_rosters": 12, "avatar": "abc"}
    out = format_sleeper_league_option(lg)
    assert out["league_id"] == "123"
    assert out["name"] == "Dynasty Warriors"
    assert out["season"] == "2024"
    assert out["total_rosters"] == 12
    assert out["avatar"] == "abc"
    assert out["label"] == "Dynasty Warriors (2024) • 12 teams"


def test_format_defaults_and_settings_fallback():
    lg = {"settings": {"num_teams": 10}}
    out = format_sleeper_league_option(lg)
    assert out["name"] == "Unnamed League"
    assert out["total_rosters"] == 10
    assert out["label"] == "Unnamed League () • 10 teams"


def test_format_unknown_team_count():
    out = format_sleeper_league_option({"name": "X", "season": 2023})
    assert out["label"] == "X (2023) • ? teams"


# ---- get_most_recent_valid_draft_for_season -------------------------------

def test_empty_or_non_list_returns_none():
    assert get_most_recent_valid_draft_for_season([], 2024) is None
    assert get_most_recent_valid_draft_for_season(None, 2024) is None


def test_picks_newest_by_best_timestamp():
    drafts = [
        {"draft_id": "a", "season": "2024", "start_time": 100},
        {"draft_id": "b", "season": "2024", "created": 200},
    ]
    out = get_most_recent_valid_draft_for_season(drafts, 2024)
    assert out["draft_id"] == "b"


def test_newest_from_older_season_returns_none():
    drafts = [{"draft_id": "old", "season": "2023", "start_time": 999}]
    assert get_most_recent_valid_draft_for_season(drafts, 2024) is None


def test_ignores_non_dict_entries():
    drafts = ["junk", {"draft_id": "a", "season": "2024", "created": 5}]
    out = get_most_recent_valid_draft_for_season(drafts, 2024)
    assert out["draft_id"] == "a"


# ---- build_roster_map -----------------------------------------------------

def test_roster_metadata_team_name_wins():
    users = [{"user_id": "1", "display_name": "Kae"}]
    rosters = [{"roster_id": 10, "owner_id": "1", "metadata": {"team_name": "Champs"}}]
    assert build_roster_map(users, rosters) == {"10": "Champs"}


def test_falls_back_to_user_name_chain():
    users = [{"user_id": "1", "metadata": {}, "username": "kae"}]
    rosters = [{"roster_id": 10, "owner_id": "1", "metadata": {}}]
    assert build_roster_map(users, rosters) == {"10": "kae"}


def test_orphan_roster_uses_roster_id_label():
    users = []
    rosters = [{"roster_id": 7, "owner_id": "99", "metadata": {}}]
    assert build_roster_map(users, rosters) == {"7": "Roster 7"}


def _empty_rosters(n=12):
    return [{"roster_id": i, "players": []} for i in range(n)]


def _filled_rosters(n=12, players=15):
    return [{"roster_id": i, "players": [f"p{i}-{j}" for j in range(players)]} for i in range(n)]


def test_empty_rosters_look_undrafted():
    assert rosters_look_undrafted([]) is True
    assert rosters_look_undrafted(_empty_rosters()) is True


def test_full_rosters_do_not_look_undrafted():
    assert rosters_look_undrafted(_filled_rosters()) is False


def test_keeper_stubs_still_look_undrafted():
    stubs = [{"roster_id": i, "players": ["a", "b"]} for i in range(12)]
    assert rosters_look_undrafted(stubs) is True


def test_startup_phase_empty_shells_are_predraft_even_if_marked_complete():
    # Yahoo/MFL/Flea and ESPN's no-date fallback report complete before a pick.
    assert startup_draft_phase(
        {"status": "in_season"},
        {"status": "complete", "start_time": 1},
        _empty_rosters(),
    ) == "predraft"


def test_startup_phase_live_draft_with_empty_rosters():
    assert startup_draft_phase(
        {"status": "drafting"}, {}, _empty_rosters(),
    ) == "drafting"


def test_dynasty_pre_draft_with_full_rosters_stays_drafted():
    # Rookie-draft waiting room: last year's team is still the team.
    assert startup_draft_phase(
        {"status": "pre_draft"},
        {"status": "pre_draft", "start_time": 9_999_999_999_000},
        _filled_rosters(),
    ) == "drafted"


def test_keeper_predraft_with_full_rosters_is_still_predraft():
    # Fleaflicker keepers retain last year's roster until the new draft runs.
    league = {
        "settings": {
            "type": 1,
            "league_type": "keeper",
            "draft_status": "NOT_YET_DRAFTED",
        },
    }
    assert startup_draft_phase(
        league, {"status": "pre_draft"}, _filled_rosters(),
    ) == "predraft"
    assert startup_draft_pending(league, {"status": "pre_draft"}, _filled_rosters()) is True


def test_keeper_full_rosters_with_omitted_flea_status_stay_predraft():
    """Provider now persists NOT_YET_DRAFTED when Fleaflicker omits the enum."""
    league = {
        "settings": {
            "type": 1,
            "league_type": "keeper",
            "draft_status": "NOT_YET_DRAFTED",
        },
    }
    # Even a leftover "complete" draft record (last year's board) must not
    # override the official pre-draft status on a keeper league.
    assert startup_draft_phase(
        league, {"status": "complete"}, _filled_rosters(),
    ) == "predraft"


def test_flea_keeper_raw_draft_status_without_draft_record():
    league = {
        "settings": {
            "type": 1,
            "league_type": "keeper",
            "draft_status": "NOT_YET_DRAFTED",
        },
    }
    assert startup_draft_phase(league, None, _filled_rosters()) == "predraft"


def test_sleeper_redraft_complete_draft_stays_drafted_if_league_still_predraft():
    # Sleeper often leaves league.status at pre_draft after a summer draft.
    league = {"status": "pre_draft", "settings": {"type": 0, "league_type": "redraft"}}
    assert startup_draft_phase(
        league, {"status": "complete"}, _filled_rosters(),
    ) == "drafted"


def test_keeper_live_draft_with_full_rosters_is_drafting():
    league = {"settings": {"type": 1, "league_type": "keeper"}}
    assert startup_draft_phase(
        league, {"status": "drafting"}, _filled_rosters(),
    ) == "drafting"


def test_show_matchup_preview_hides_undrafted_keeper_even_with_rosters():
    league = {
        "settings": {
            "type": 1,
            "league_type": "keeper",
            "draft_status": "NOT_YET_DRAFTED",
        },
    }
    assert show_matchup_preview(
        league, {"status": "pre_draft"}, _filled_rosters(),
    ) is False
    assert show_matchup_preview(
        league, {"status": "pre_draft"}, _filled_rosters(), is_dynasty=False,
    ) is False


def test_show_matchup_preview_always_on_for_dynasty():
    empty = _empty_rosters()
    dynasty = {"status": "pre_draft", "settings": {"type": 2, "league_type": "dynasty"}}
    assert show_matchup_preview(dynasty, {"status": "pre_draft"}, empty) is True
    assert show_matchup_preview(
        {"settings": {"type": 1, "league_type": "keeper"}},
        {"status": "pre_draft"},
        empty,
        is_dynasty=True,
    ) is True


def test_show_matchup_preview_after_keeper_draft():
    league = {"settings": {"type": 1, "league_type": "keeper"}}
    assert show_matchup_preview(
        league, {"status": "complete"}, _filled_rosters(),
    ) is True


def test_draft_start_ms_converts_seconds_and_prefers_draft_record():
    assert draft_start_ms({}, {"start_time": 1_700_000_000}) == 1_700_000_000_000
    assert draft_start_ms({"draft_day": 1_800_000_000_000}, {"start_time": 0}) == 1_800_000_000_000
    assert draft_start_ms({}, {}) is None


def test_countdown_copy_formats_days_and_live():
    start = 1_700_000_000_000
    now = start - (2 * 86400 + 5) * 1000
    copy = draft_countdown_copy(start, now_ms=now, phase="predraft")
    assert copy["label"] == "Draft countdown"
    assert copy["value"].startswith("2d ")
    assert copy["sub"]  # date string

    live = draft_countdown_copy(start, now_ms=now, phase="drafting")
    assert live["value"] == "Live now"

    missing = draft_countdown_copy(None, phase="predraft")
    assert missing["value"] == "TBD"


def test_top_board_preview_ranks_skill_positions_and_prefers_sf_value():
    table = [
        {"id": "1", "name": "QB A", "position": "QB", "value": 100, "sf_value": 900},
        {"id": "2", "name": "WR B", "pos": "WR", "value": 400, "sf_value": 200},
        {"id": "3", "name": "Kicker", "position": "K", "value": 999, "sf_value": 999},
        {"id": "4", "name": "Zero", "position": "RB", "value": 0},
    ]
    one_qb = top_board_preview(table, is_sf=False, limit=10)
    assert [p["name"] for p in one_qb] == ["WR B", "QB A"]
    sf = top_board_preview(table, is_sf=True, limit=10)
    assert [p["name"] for p in sf] == ["QB A", "WR B"]
