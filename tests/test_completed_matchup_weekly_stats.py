from dashboard_services import matchups as mmod


def test_position_stat_lines_include_requested_weekly_volume():
    stats = {
        "BUF": {
            "QB": {"josh allen": {"pass_cmp": 20, "pass_att": 30, "pass_yds": 250, "pass_td": 2, "int": 1, "rush_yds": 35, "rush_td": 1}},
            "RB": {"james cook": {"rush_att": 15, "rush_yds": 70, "rush_td": 1, "rec": 4, "tgt": 5, "rec_yds": 28, "rec_td": 0}},
            "WR": {"keon coleman": {"rec": 3, "tgt": 7, "rec_yds": 51, "rec_td": 1}},
        }
    }
    qb = mmod.format_player_stats(stats, "BUF", "QB", "Josh Allen")
    rb = mmod.format_player_stats(stats, "BUF", "RB", "James Cook")
    wr = mmod.format_player_stats(stats, "BUF", "WR", "Keon Coleman")
    assert "20/30 cmp/att" in qb and "250 yds" in qb and "1 int" in qb
    assert "CAR 15" in rb and "TGT 5" in rb and "REC YD/TD 28/0" in rb
    assert "REC 3" in wr and "TGT 7" in wr and "REC YD/TD 51/1" in wr


def test_completed_week_renders_starter_and_bench_stats_and_zero_points(monkeypatch):
    weekly = {"BUF": {"QB": {"starter qb": {"pass_yds": 200}}, "WR": {"bench wr": {"rec": 2, "tgt": 4, "rec_yds": 20}}}}
    monkeypatch.setattr(mmod, "load_teams_index", lambda: {})
    monkeypatch.setattr(mmod, "build_offense_rankings", lambda *_: {})
    monkeypatch.setattr(mmod, "load_week_stats", lambda season, week: weekly)
    monkeypatch.setattr(mmod, "load_week_schedule", lambda *_: [])
    monkeypatch.setattr(mmod, "build_team_schedule_lookup", lambda *_: {})
    matchup = {
        "left": {"name": "Left", "roster_id": "1", "starters": [{"pid": "1", "name": "Starter QB", "pos": "QB", "nfl": "BUF", "pts": 12.0}], "bench": [{"pid": "2", "name": "Bench WR", "pos": "WR", "nfl": "BUF", "pts": 0.0}], "pts_total": 12.0},
        "right": {"name": "Right", "roster_id": "2", "starters": [], "bench": [], "pts_total": 0.0},
    }
    html = mmod.render_matchup_slide("2025", matchup, 2, 2, {}, {}, {}, {}, {})
    assert "Starter QB" in html and "200 yds" in html
    assert "Bench WR" in html and "TGT 4" in html
    assert "m-row--bench" in html and ">0.0</span>" in html
