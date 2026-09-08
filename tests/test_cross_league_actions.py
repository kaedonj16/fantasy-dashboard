from __future__ import annotations

from utils.cross_league_actions import (
    action_priority,
    calendar_action,
    injury_stash_action,
    lineup_actions_from_issues,
    make_action,
    parse_pos_rank,
    rank_cross_league_actions,
    roster_slot_action,
    select_waiver_add,
    waiver_add_clears_quality_bar,
    waiver_pickup_action,
    waiver_rank_ceiling,
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
    assert waiver_value_threshold(50.0, is_redraft=True) == 50.0 * 1.6


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
    need = waiver_pickup_action(
        platform="sleeper", season=2025, league_id="1", league_name="N",
        player_name="Dalton Kincaid", position="TE", is_redraft=True,
        pos_rank_label="TE11", starter_gap=1.0, pos_rank=11,
    )
    assert "Fills a TE need" in need["detail"]
    assert "Top available" not in need["detail"]


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


def test_parse_pos_rank_from_label_and_number():
    assert parse_pos_rank(35, "RB35") == 35
    assert parse_pos_rank(None, "WR44") == 44
    assert parse_pos_rank("", "TE11") == 11
    assert parse_pos_rank(None, "") is None


def test_waiver_rank_ceiling_scales_and_raises_sf_qb():
    assert waiver_rank_ceiling("RB", is_redraft=True, n_teams=12) == 28
    assert waiver_rank_ceiling("QB", is_redraft=True, is_sf=False) == 14
    assert waiver_rank_ceiling("QB", is_redraft=True, is_sf=True) == 24
    assert waiver_rank_ceiling("TE", is_redraft=False) == 14
    assert waiver_rank_ceiling("RB", is_redraft=True, n_teams=10) < 28


def test_quality_bar_drops_leftover_rb35_and_streamer_qb():
    # Screenshot-style leftovers: highest remaining value, not a great add.
    assert not waiver_add_clears_quality_bar(
        pos="RB", pos_rank=35, value=80, is_redraft=True,
    )
    assert not waiver_add_clears_quality_bar(
        pos="RB", pos_rank=36, value=80, is_redraft=True, starter_gap=1.0,
    )
    assert not waiver_add_clears_quality_bar(
        pos="QB", pos_rank=21, value=90, is_redraft=True, is_sf=False,
    )
    assert not waiver_add_clears_quality_bar(
        pos="WR", pos_rank=44, value=70, is_redraft=True,
    )
    # Aging dynasty WR35 is not a stash.
    assert not waiver_add_clears_quality_bar(
        pos="WR", pos_rank=35, value=200, is_redraft=False, age=32,
    )
    # Below the format value floor.
    assert not waiver_add_clears_quality_bar(
        pos="TE", pos_rank=8, value=20, is_redraft=True,
    )


def test_quality_bar_keeps_startable_adds():
    assert waiver_add_clears_quality_bar(
        pos="TE", pos_rank=11, value=90, is_redraft=True, starter_gap=1.0,
    )
    assert waiver_add_clears_quality_bar(
        pos="WR", pos_rank=28, value=80, is_redraft=True,
    )
    assert waiver_add_clears_quality_bar(
        pos="RB", pos_rank=22, value=100, is_redraft=True,
    )
    # Superflex can use a QB21; 1QB cannot.
    assert waiver_add_clears_quality_bar(
        pos="QB", pos_rank=21, value=90, is_redraft=True, is_sf=True,
    )
    # Young dynasty WR inside the ceiling.
    assert waiver_add_clears_quality_bar(
        pos="WR", pos_rank=28, value=180, is_redraft=False, age=23,
    )


def _fa(pid, name, pos, rank, value, age=24, team="PHI"):
    return {
        "id": pid, "name": name, "position": pos, "team": team,
        "value": value, "pos_rank": rank, "pos_rank_label": f"{pos}{rank}",
        "age": age,
    }


def test_select_waiver_add_skips_screenshot_leftovers():
    rows = [
        _fa("g", "Kenneth Gainwell", "RB", 35, 80),
        _fa("d", "Sam Darnold", "QB", 21, 90),
        _fa("c", "KC Concepcion", "WR", 44, 60),
        _fa("t", "Tyreek Hill", "WR", 35, 200, age=32),
    ]
    # 1QB redraft with a full roster — nobody is a great add.
    hit = select_waiver_add(
        rows, set(),
        value_key="value", is_redraft=True, is_sf=False, n_teams=12,
        roster_players=["qb1", "rb1", "rb2", "wr1", "wr2", "te1"],
        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        pidx={
            "qb1": {"position": "QB"}, "rb1": {"position": "RB"},
            "rb2": {"position": "RB"}, "wr1": {"position": "WR"},
            "wr2": {"position": "WR"}, "te1": {"position": "TE"},
        },
    )
    assert hit is None


def test_select_waiver_add_prefers_startable_te_need_over_leftover_rb():
    rows = [
        _fa("g", "Kenneth Gainwell", "RB", 35, 200),
        _fa("k", "Dalton Kincaid", "TE", 11, 110),
    ]
    hit = select_waiver_add(
        rows, set(),
        value_key="value", is_redraft=True, is_sf=False, n_teams=12,
        roster_players=["qb1", "rb1", "rb2", "wr1", "wr2"],
        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "BN"],
        pidx={
            "qb1": {"position": "QB"}, "rb1": {"position": "RB"},
            "rb2": {"position": "RB"}, "wr1": {"position": "WR"},
            "wr2": {"position": "WR"},
        },
    )
    assert hit is not None
    assert hit["name"] == "Dalton Kincaid"
    assert hit["position"] == "TE"
    assert "TE need" in hit["reason"] or "Startable" in hit["reason"] or "TE11" in hit["reason"]


def test_select_waiver_add_skips_owned_injured_and_free_agents():
    rows = [
        _fa("k", "Dalton Kincaid", "TE", 11, 110),
        _fa("x", "Ghost", "RB", 18, 150, team="FA"),
        _fa("y", "Hurt", "WR", 20, 140),
    ]
    hit = select_waiver_add(
        rows, {"k"},
        value_key="value", is_redraft=True,
        injured_status_by_pid={"y": "IR"},
        roster_players=[],
        roster_positions=["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"],
    )
    assert hit is None

