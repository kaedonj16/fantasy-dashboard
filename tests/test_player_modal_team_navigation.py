"""Player-modal Team panel navigation uses its explicit interactive contract."""
from pathlib import Path


SRC = (Path(__file__).resolve().parents[1] / "static" / "player_modal.js").read_text()


def _wire():
    return SRC.split("function _pmWireTeamPanel(", 1)[1].split("function pmPrefetchTabs", 1)[0]


def test_click_and_keyboard_share_narrow_player_target_helper():
    wire = _wire()
    assert "function (target)" in wire
    assert "closest('[data-pid][role=\"button\"]')" in wire
    assert "const nav = findPlayerNavTarget(target)" in wire
    assert "const row=findPlayerNavTarget(e.target)" in wire
    assert "closest('[data-pid]')" not in wire


def test_metadata_pid_blocks_remain_non_navigation_metadata():
    assert 'id="pmAdpBlock" data-pid=' in SRC
    assert 'id="pmWeeklyTrendsWrap" data-position=' in SRC
    wire = _wire()
    # Exactly one open call lives in the one navigation helper; ordinary Team
    # controls continue through their existing branches.
    assert wire.count("openTeammate(nav)") == 1
    assert wire.count("openTeammate(row)") == 1
    for control in (".pm-team-adv-toggle", ".pm-team-sched-toggle",
                    ".pm-schedule-toggle", ".pm-boxscore-team-pill"):
        assert control in wire
