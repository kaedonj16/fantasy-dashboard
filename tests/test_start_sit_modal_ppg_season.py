"""Start/Sit modal rows: the "Def vs pos" PPG must share the matchup
rank's season basis.

Regression: `_startsit_compare_extras` (the player modal's Start/Sit tab)
took its matchup rank from `_matchup_rank_table`, the current-season
ratings table, but called `_fpts_against_effective` without a completed
week. That function's default (0) silently means "last season only":
zero current-season blend weight and no current-season weeks counted,
so the PPG printed on the row was the prior season's full-season line
next to a current-season rank (the same pairing the Start/Sit page rows
build in `_build_start_sit_options`). The fix passes the season's
completed week through at both call sites.

The fake `_fpts_against_effective` below mirrors the real blend's
contract: completed_week=0 anchors on last season alone, and at 6+
completed weeks the value is the current season's. On the pre-fix code
the modal call never passes a week, so the row comes back with the
prior-season PPG and this test fails.
"""

import pytest

pytest.importorskip("flask")
pytest.importorskip("pandas")

SEASON = 2026
COMPLETED = 9  # weeks 1-9 final, week 10 upcoming
PRIOR_RB_PPG = 30.0    # BUF vs RB, prior season full-season line
CURRENT_RB_PPG = 12.0  # BUF vs RB, current season through week 9


class _Policy:
    completed_weeks = tuple(range(1, COMPLETED + 1))


@pytest.fixture
def modal_env(monkeypatch):
    import app as appmod
    import utils.season_qualification as sq
    import data_building.start_score_bundle as ssb
    import data_building.weekly_metrics as wm

    monkeypatch.setattr(sq, "qualification_policy", lambda season: _Policy())
    cache = getattr(appmod, "_COMPLETED_WEEK_CACHE", None)
    if isinstance(cache, dict):
        cache.clear()

    calls = {}

    def fake_fpts_against(season, scoring_settings=None, completed_week=0,
                          **kwargs):
        calls["completed_week"] = completed_week
        if completed_week >= 6:
            return {"BUF": {"RB": CURRENT_RB_PPG, "games": COMPLETED}}
        return {"BUF": {"RB": PRIOR_RB_PPG, "games": 17}}

    monkeypatch.setattr(appmod, "_fpts_against_effective", fake_fpts_against)
    # The rank beside the PPG: current-season ratings table, BUF is the
    # #2-easiest defense vs RB.
    monkeypatch.setattr(appmod, "_matchup_rank_table",
                        lambda *a, **k: ({"BUF": 2}, 32, {}, True))
    # Silence the extras' other best-effort inputs.
    monkeypatch.setattr(appmod, "_load_season_weekly_points",
                        lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_load_team_play_volume", lambda *a, **k: {})
    monkeypatch.setattr(appmod, "_oline_for_player", lambda *a, **k: None)
    monkeypatch.setattr(appmod, "_ss_game_env", lambda *a, **k: None)
    monkeypatch.setattr(ssb, "derive_start_sit_inputs", lambda *a, **k: {})
    monkeypatch.setattr(wm, "get_usage_trends", lambda *a, **k: {})
    return appmod, calls


def test_modal_row_ppg_comes_from_current_season(modal_env):
    appmod, calls = modal_env
    out = appmod._startsit_compare_extras(
        "rb1", "RB", "KC", SEASON, COMPLETED + 1, {},
        proj_pts=15.0, opponent="BUF", home_team="KC")

    # The completed week was passed through, not left at the
    # last-season-only default.
    assert calls["completed_week"] == COMPLETED
    # Rank and PPG on the same row now describe the same season: the
    # #2-easiest current-season defense is the one allowing 12.0 PPG
    # this season, not the 30.0 it allowed last season.
    assert out["def_rank"] == 2
    assert out["def_total"] == 32
    assert out["fpts_against"] == CURRENT_RB_PPG
