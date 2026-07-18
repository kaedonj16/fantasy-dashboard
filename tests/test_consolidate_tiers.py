"""Guards the roster-aware consolidation ceiling in archetype_engine.

The tier a consolidation aims for should depend on what the viewer already has at
that position: a team with only flex-worthy pieces gets steered to a pure
starter, never an unrealistic reach for a top-3 elite; a team that already
rosters pure starters can consolidate them into an elite. Tiers are
league-specific (starter demand scales with size and format).

Pure module-level functions, so this runs in the base suite (no Flask/pandas).
"""
from dashboard_services.archetype_engine import (
    _consolidate_target_allowed,
    _pos_category,
    _pos_starter_line,
)


def test_starter_line_scales_with_league():
    # More teams => more startable players at a position.
    assert _pos_starter_line("WR", 12, "1qb") > _pos_starter_line("WR", 8, "1qb")
    # Superflex roughly doubles QB starter demand vs 1QB.
    assert _pos_starter_line("QB", 12, "sf") > _pos_starter_line("QB", 12, "1qb")
    # TE (one slot, small flex share) has far fewer starters than WR.
    assert _pos_starter_line("TE", 12, "1qb") < _pos_starter_line("WR", 12, "1qb")


def test_position_categories():
    n, lt = 12, "1qb"
    line = _pos_starter_line("WR", n, lt)  # 35 in a 12-team 1QB league
    assert _pos_category("WR", 1, n, lt) == "elite"       # top 3
    assert _pos_category("WR", 3, n, lt) == "elite"
    assert _pos_category("WR", 4, n, lt) == "starter"     # just below elite
    assert _pos_category("WR", line, n, lt) == "starter"  # last startable
    assert _pos_category("WR", line + 1, n, lt) == "flex" # first flex/depth
    assert _pos_category("WR", None, n, lt) == "depth"    # not rostered


def test_flex_only_team_cannot_reach_an_elite():
    # A team whose best WR is only flex/depth should not be aimed at a top-3 WR.
    assert _consolidate_target_allowed("elite", "flex") is False
    assert _consolidate_target_allowed("elite", "depth") is False


def test_team_with_a_starter_can_consolidate_into_an_elite():
    assert _consolidate_target_allowed("elite", "starter") is True
    assert _consolidate_target_allowed("elite", "elite") is True


def test_pure_starter_targets_are_always_allowed():
    # Steering a flex-heavy team toward a pure starter is exactly the goal.
    for best in ("depth", "flex", "starter", "elite"):
        assert _consolidate_target_allowed("starter", best) is True
        assert _consolidate_target_allowed("flex", best) is True
