from __future__ import annotations

from utils.cross_league_actions import (
    action_priority,
    injury_stash_action,
    lineup_actions_from_issues,
    make_action,
    rank_cross_league_actions,
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
