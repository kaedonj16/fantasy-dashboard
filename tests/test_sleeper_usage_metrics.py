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


def test_weekly_red_zone_stats_populate_without_per_player_requests(monkeypatch):
    import data_building.external_data.sleeper_usage as usage

    weeks = {
        1: {
            "wr": {"team": "KC", "rec_tgt": 6, "rec_rz_tgt": 2, "pts_ppr": 1},
            "rb": {"team": "KC", "rush_att": 10, "rush_rz_att": 3, "pts_ppr": 1},
        },
        2: {
            "wr": {"team": "KC", "rec_tgt": 4, "pts_ppr": 1},
            "rb": {"team": "KC", "rush_att": 8, "rush_rz_att": 1, "pts_ppr": 1},
        },
    }
    monkeypatch.setattr(usage, "fetch_week_stats", lambda season, week: weeks[week])
    monkeypatch.setattr(
        usage, "fetch_season_redzone_stats",
        lambda season: (_ for _ in ()).throw(AssertionError("fallback must not run")),
    )
    monkeypatch.setattr(usage, "load_players_index", lambda: {
        "wr": {"team": "KC", "pos": "WR"}, "rb": {"team": "KC", "pos": "RB"},
    })

    result = usage.build_usage_map_for_season(2026, [1, 2])
    assert result["wr"]["rec_rz_tgt_pg"] == 1.0
    assert result["wr"]["rush_rz_att_pg"] == 0.0
    assert result["rb"]["rush_rz_att_pg"] == 2.0
    assert result["wr"]["red_zone_available"] is True


def test_carry_and_touch_share_populate_for_rbs(monkeypatch):
    import data_building.external_data.sleeper_usage as usage

    weeks = {
        1: {
            "rb1": {"team": "BUF", "rush_att": 20, "rec_tgt": 5, "pts_ppr": 1},
            "rb2": {"team": "BUF", "rush_att": 10, "rec_tgt": 2, "pts_ppr": 1},
            "wr": {"team": "BUF", "rec_tgt": 8, "pts_ppr": 1},
        },
        2: {
            "rb1": {"team": "BUF", "rush_att": 20, "rec_tgt": 5, "pts_ppr": 1},
            "rb2": {"team": "BUF", "rush_att": 10, "rec_tgt": 2, "pts_ppr": 1},
            "wr": {"team": "BUF", "rec_tgt": 8, "pts_ppr": 1},
        },
    }
    monkeypatch.setattr(usage, "fetch_week_stats", lambda season, week: weeks[week])
    monkeypatch.setattr(usage, "fetch_season_redzone_stats", lambda season: {})
    monkeypatch.setattr(usage, "load_players_index", lambda: {
        "rb1": {"team": "BUF", "pos": "RB"},
        "rb2": {"team": "BUF", "pos": "RB"},
        "wr": {"team": "BUF", "pos": "WR"},
    })

    result = usage.build_usage_map_for_season(2026, [1, 2])
    # Team carries: (20 + 10) * 2 = 60.
    assert result["rb1"]["carry_share"] == 40 / 60
    assert result["rb2"]["carry_share"] == 20 / 60
    # Team opportunities: targets (5 + 2 + 8) * 2 = 30 plus carries 60 = 90.
    # rb1 touches: (20 + 5) * 2 = 50; rb2 touches: (10 + 2) * 2 = 24.
    assert result["rb1"]["touch_share"] == 50 / 90
    assert result["rb2"]["touch_share"] == 24 / 90
    # Target share still correct alongside the new shares.
    assert result["rb1"]["target_share"] == 10 / 30


def test_share_denominators_use_meta_team_when_weekly_rows_lack_team(monkeypatch):
    # Sleeper's weekly rows carry no team field, so attribution falls back to
    # the players-index meta team. A player with targets must not end up with
    # a 0/0 denominator (which rendered as 0%).
    import data_building.external_data.sleeper_usage as usage

    weeks = {
        1: {
            # No "team" on either row, mirroring the real 2026 weekly feed.
            "rb1": {"rush_att": 10, "rec_tgt": 4, "pts_ppr": 1},
            "wr": {"rec_tgt": 6, "pts_ppr": 1},
        },
    }
    monkeypatch.setattr(usage, "fetch_week_stats", lambda season, week: weeks[week])
    monkeypatch.setattr(usage, "fetch_season_redzone_stats", lambda season: {})
    monkeypatch.setattr(usage, "load_players_index", lambda: {
        "rb1": {"team": "BUF", "pos": "RB"},
        "wr": {"team": "BUF", "pos": "WR"},
    })

    result = usage.build_usage_map_for_season(2026, [1])
    # Both players attributed to BUF via meta: team targets = 10, carries = 10,
    # opportunities = 20.
    assert result["rb1"]["target_share"] == 4 / 10
    assert result["rb1"]["carry_share"] == 10 / 10
    assert result["rb1"]["touch_share"] == 14 / 20
