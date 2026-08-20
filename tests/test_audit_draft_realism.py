from scripts.audit_draft_realism import (
    _draft_filters,
    render_inventory,
    render_markdown,
    summarize_cohort,
)


def test_summarize_cohort_tracks_depth_timing_and_position_counts():
    drafts = [{"draft_id": "d1", "draft_type": "redraft", "is_superflex": False,
               "num_teams": 2, "rounds": 6}]
    positions = {"q1": "QB", "q2": "QB", "r1": "RB", "w1": "WR", "t1": "TE", "k1": "K"}
    picks = [
        {"draft_id": "d1", "player_id": "r1", "pick_no": 1, "round": 1, "roster_id": "1"},
        {"draft_id": "d1", "player_id": "w1", "pick_no": 3, "round": 2, "roster_id": "1"},
        {"draft_id": "d1", "player_id": "q1", "pick_no": 5, "round": 3, "roster_id": "1"},
        {"draft_id": "d1", "player_id": "t1", "pick_no": 7, "round": 4, "roster_id": "1"},
        {"draft_id": "d1", "player_id": "q2", "pick_no": 9, "round": 5, "roster_id": "1"},
        {"draft_id": "d1", "player_id": "k1", "pick_no": 11, "round": 6, "roster_id": "1"},
    ]

    summary = summarize_cohort(drafts, picks, positions)

    assert summary.qb_rounds == {"QB1": 3, "QB2": 5, "QB3": None}
    assert summary.te_rounds["TE1"] == 4
    assert summary.position_counts["QB"] == 2
    assert summary.k_first_round == 6
    assert summary.resolved_pct == 100
    assert "Redraft — 1QB" in render_markdown([summary], "Filters: test.")


def test_unresolved_players_are_reported_but_not_counted():
    drafts = [{"draft_id": "d1", "draft_type": "startup", "is_superflex": True,
               "num_teams": 12, "rounds": 2}]
    picks = [
        {"draft_id": "d1", "player_id": "known", "pick_no": 1, "round": 1, "roster_id": "1"},
        {"draft_id": "d1", "player_id": "missing", "pick_no": 2, "round": 2, "roster_id": "1"},
    ]

    summary = summarize_cohort(drafts, picks, {"known": "QB"})

    assert summary.resolved_pct == 50
    assert summary.picks == 2
    assert summary.teams == 1


def test_db_filters_include_legacy_rows_without_status():
    where, params = _draft_filters([2024, 2025], ["redraft", "startup"], alias="d")

    assert where == "d.draft_type = ANY(%s) AND d.season = ANY(%s)"
    assert "status" not in where
    assert params == [["redraft", "startup"], [2024, 2025]]


def test_inventory_explains_which_cohorts_are_available():
    text = render_inventory([
        {"season": 2026, "draft_type": "startup", "is_superflex": True, "drafts": 42}
    ])

    assert "2026" in text
    assert "startup" in text
    assert "Superflex" in text
    assert "42" in text
