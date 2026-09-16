"""Sleeper-only current-season usage aggregation regressions."""

import subprocess
import sys


def test_usage_module_does_not_import_heavy_optional_providers():
    code = (
        "import sys; import data_building.external_data.sleeper_usage; "
        "assert 'pandas' not in sys.modules; assert 'nfl_data_py' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_sleeper_totals_drive_opportunity_and_target_share(monkeypatch):
    import data_building.external_data.sleeper_usage as usage

    weeks = {
        1: {
            "rb": {"team": "JAC", "rush_att": 5, "rec_tgt": 6, "pts_ppr": 1},
            "wr": {"team": "JAX", "rec_tgt": 49, "pts_ppr": 1},
        },
        # Only the WR records a stat in Week 2. The RB denominator must still be
        # the full team-season 70, not 60 and not a sum of per-game averages.
        2: {"wr": {"team": "JAX", "rec_tgt": 10, "pts_ppr": 1}},
    }
    monkeypatch.setattr(usage, "fetch_week_stats", lambda season, week: weeks[week])
    monkeypatch.setattr(usage, "fetch_season_redzone_stats", lambda season: {})
    monkeypatch.setattr(usage, "load_players_index", lambda: {
        "rb": {"team": "JAX", "pos": "RB"}, "wr": {"team": "JAX", "pos": "WR"},
    })

    result = usage.build_usage_map_for_season(2026, [1, 2])
    assert result["rb"]["team_opportunities"] == 70
    assert result["rb"]["season_targets"] + result["rb"]["season_carries"] == 11
    assert round(100 * 11 / result["rb"]["team_opportunities"], 1) == 15.7
    assert result["wr"]["target_share"] == 59 / 65
