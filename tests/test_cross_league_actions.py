from __future__ import annotations

from utils.cross_league_actions import (
    action_priority,
    calendar_action,
    injury_stash_action,
    lineup_actions_from_issues,
    make_action,
    rank_cross_league_actions,
    roster_slot_action,
    waiver_pickup_action,
    waiver_value_threshold,
)


def test_action_priority_orders_kinds():
    assert action_priority("lineup") > action_priority("injury") > action_priority("waiver")


def test_rank_sorts_by_priority_then_league_name():
    actions = [
        make_action(
            kind="injury", platform="espn", season=2025, league_id="2",
            league_name="Beta", title="Stash: X", severity=0.5,
        ),
        make_action(
            kind="lineup", platform="sleeper", season=2025, league_id="1",
            league_name="Alpha", title="Empty starting slot", severity=1.0,
        ),
        make_action(
            kind="lineup", platform="yahoo", season=2025, league_id="3",
            league_name="Charlie", title="Starter on bye", severity=0.7,
        ),
    ]
    ranked = rank_cross_league_actions(actions)
    assert [a["league_id"] for a in ranked] == ["1", "3", "2"]


def test_lineup_actions_empty_slot_is_highest_severity():
    issues = [{"kind": "empty", "pid": "0", "name": "", "detail": "Empty starting slot"}]
    acts = lineup_actions_from_issues(
        issues, platform="sleeper", season=2025, league_id="lg1", league_name="My Team",
    )
    assert len(acts) == 1
    assert acts[0]["kind"] == "lineup"
    assert acts[0]["title"] == "Empty starting slot"
    assert "/sleeper/2025/lg1/waivers?tab=startsit" in acts[0]["href"]
    assert acts[0]["priority"] >= action_priority("lineup", severity=0.9)


def test_lineup_actions_injury_and_bye_titles():
    inj = lineup_actions_from_issues(
        [{"kind": "injury", "pid": "1", "name": "A", "detail": "A is listed Out"}],
        platform="espn", season=2025, league_id="9", league_name="E",
    )
    assert inj[0]["title"] == "Injured starter needs a swap"
    bye = lineup_actions_from_issues(
        [{"kind": "bye", "pid": "2", "name": "B", "detail": "B is on bye"}],
        platform="yahoo", season=2025, league_id="8", league_name="Y",
    )
    assert bye[0]["title"] == "Starter on bye"


def test_injury_stash_action_filters_unknown_verdicts():
    assert injury_stash_action(
        platform="espn", season=2025, league_id="1", league_name="N",
        player_name="X", verdict="Monitor",
    ) is None
    act = injury_stash_action(
        platform="espn", season=2025, league_id="1", league_name="N",
        player_name="Injured Guy", verdict="Stash", weeks_label="~3 wk",
    )
    assert act is not None
    assert act["kind"] == "injury"
    assert act["title"] == "Stash: Injured Guy"
    assert "Approx return ~3 wk" in act["detail"] or "~3 wk" in act["detail"]


def test_injury_stash_action_skips_players_already_on_ir():
    assert injury_stash_action(
        platform="sleeper", season=2025, league_id="1", league_name="N",
        player_name="Already Stashed", verdict="Stash", already_on_ir=True,
    ) is None
    assert injury_stash_action(
        platform="sleeper", season=2025, league_id="1", league_name="N",
        player_name="Already Stashed", verdict="IR", already_on_ir=True,
    ) is None
    # Drop from IR can still free a slot.
    drop = injury_stash_action(
        platform="sleeper", season=2025, league_id="1", league_name="N",
        player_name="Drop Me", verdict="Drop candidate", already_on_ir=True,
    )
    assert drop is not None
    assert drop["title"] == "Drop candidate: Drop Me"


def test_waiver_threshold_is_higher_for_dynasty():
    assert waiver_value_threshold(25.0, is_redraft=True) < waiver_value_threshold(
        25.0, is_redraft=False
    )
    # Tied to the shared floor, not hard-coded.
    assert waiver_value_threshold(50.0, is_redraft=True) == 50.0 * 1.4


def test_waiver_pickup_action_labels_format():
    rd = waiver_pickup_action(
        platform="sleeper", season=2025, league_id="1", league_name="N",
        player_name="Rookie WR", position="wr", is_redraft=True,
        pos_rank_label="WR48", value=120.0,
    )
    assert rd["kind"] == "waiver"
    assert rd["title"] == "Add Rookie WR (WR)"
    assert "redraft value" in rd["detail"]
    assert "WR48" in rd["detail"]
    assert "/sleeper/2025/1/waivers" in rd["href"]
    dyn = waiver_pickup_action(
        platform="sleeper", season=2025, league_id="1", league_name="N",
        player_name="Young RB", position="RB", is_redraft=False,
    )
    assert "dynasty value" in dyn["detail"]


def test_roster_slot_action_prefers_most_actionable():
    issues = [
        {"kind": "taxi_stash", "pid": "3", "name": "Rook", "detail": "taxi open"},
        {"kind": "ir_activate", "pid": "1", "name": "Back", "detail": "no longer IR"},
        {"kind": "ir_stash", "pid": "2", "name": "Hurt", "detail": "move to IR"},
    ]
    act = roster_slot_action(
        issues, platform="sleeper", season=2025, league_id="42", league_name="N",
    )
    assert act is not None
    assert act["kind"] == "roster"
    assert act["title"] == "Activate or drop a recovered IR player"
    assert act["detail"] == "no longer IR"
    assert "/sleeper/2025/42/teams" in act["href"]
    assert roster_slot_action(
        [], platform="sleeper", season=2025, league_id="42", league_name="N",
    ) is None


def test_calendar_action_deadline_precedes_playoffs():
    # Deadline in 1 week wins over a playoff countdown.
    act = calendar_action(
        platform="sleeper", season=2025, league_id="1", league_name="N",
        week=11, trade_deadline=12, playoff_week_start=13,
    )
    assert act["kind"] == "calendar"
    assert act["title"] == "Trade deadline in 1 week"
    assert "/trade" in act["href"]
    # This-week deadline phrasing.
    now = calendar_action(
        platform="sleeper", season=2025, league_id="1", league_name="N",
        week=12, trade_deadline=12,
    )
    assert now["title"] == "Trade deadline is this week"
    # Playoffs only (no/expired deadline).
    po = calendar_action(
        platform="sleeper", season=2025, league_id="1", league_name="N",
        week=13, trade_deadline=0, playoff_week_start=14,
    )
    assert po["title"] == "Playoffs start in 1 week"
    assert "/matchups" in po["href"]
    # Nothing near, or out of season.
    assert calendar_action(
        platform="sleeper", season=2025, league_id="1", league_name="N",
        week=5, trade_deadline=12, playoff_week_start=14,
    ) is None
    assert calendar_action(
        platform="sleeper", season=2025, league_id="1", league_name="N",
        week=0, trade_deadline=12,
    ) is None
